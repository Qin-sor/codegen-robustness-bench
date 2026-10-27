"""报告生成（设计文档 §4.14）：把 ``results-<tag>.jsonl`` 变成 ``summary-<tag>.md`` / ``.csv`` 和一张图。

``build_report(ws, tag)`` 的步骤：

1. 读 ``results-<tag>.jsonl``，**校验每条记录的输入哈希**（设计文档 §6.6）：schema 版本、prompt 模板摘要、
   评测代码摘要、混淆副本的源码树哈希、spec 摘要、oracle gate、对应的生成记录（回复有没有被重新生成或删掉）、
   对应的任务（提示词 / 功能规格有没有变）都要与当前的文件对得上；
   对不上的记录默认不进统计，在报告里列出数量和原因（``allow_stale=True`` 才纳入）；
2. 用 :mod:`bench.metrics` 汇总：model × level × corpus 明细、model × level 合计、相对 L0 的配对变化；
3. 渲染 Markdown（总表、相对 L0 的变化、失败归因分布、按 stub-kill 杀死方式分层、分语料明细、可信度记录、局限、来源）、
   CSV，以及一张 SVG 折线图（通过率与编译通过率随混淆级别的变化，带 Wilson 95% 置信区间色带）。
   结果文件里的坏行、evaluate 记下的缺失样本都会写进报告，命令也据此返回非零（``--allow-missing`` 放行）。

关于图：设计文档 v0.1 写的是 matplotlib 存 PNG（可选），v0.2 改成标准库直接生成 SVG：图只是汇总表的补充，
不新增依赖（matplotlib 对 numpy 版本敏感，装不装得上取决于环境），断网也能打开，浏览器里直接看；
需要 PNG 时用任意工具转换即可。图遵循 dataviz 约定：线 2px、圆点直径 8px 且带 2px 底色描边、
色带用 10% 不透明度、网格是 1px 实线的低对比灰；调色板取自已通过校验的 8 色类别色板，
浅色和深色各一套，随系统主题切换；每个点带 ``<title>`` 悬停提示，数值在汇总表里一样能查到（提示只是增强，不是唯一来源）。
"""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence
from xml.sax.saxutils import escape, quoteattr

from bench.limitations import LIMITATIONS, REPLY_SHARING_NOTE
from bench.metrics import (
    ALL, LEVEL_ORDER, Cell, Drop, csv_text, drop_vs_baseline, load_results_checked, overview, summarize,
    summarize_all,
)
from bench.prompt import PROMPT_DIGEST
from bench.provenance import code_digest
from bench.store import SCHEMA_VERSION, Workspace, digest, read_json, read_jsonl, spec_digest
from bench.vocab import CATEGORIES, LEVELS

log = logging.getLogger(__name__)

__all__ = ["Validation", "ReportResult", "validate_records", "gather_context", "gather_provenance", "load_completeness",
           "render_markdown", "render_svg", "build_report"]

#: 混淆级别的中文说明，图和表的脚注用
LEVEL_NOTES = {"L0": "原样", "L1": "符号", "L2": "+结构", "L3": "+语义"}

#: 归因类别的中文说明
CATEGORY_NOTES = {
    "pass": "通过", "extract_fail": "抽不出代码", "policy_violation": "违反硬性约束", "compile_syntax": "语法错误",
    "compile_undeclared": "未声明",
    "compile_type": "类型错误", "compile_other": "其它编译错误", "link_error": "链接错误",
    "runtime_crash": "运行崩溃", "runtime_timeout": "运行超时", "test_fail": "测试失败",
}


# ----------------------------------------------------------------------
# 输入哈希校验
# ----------------------------------------------------------------------
@dataclass
class Validation:
    total: int
    valid: list[dict]
    stale: list[tuple[dict, str]] = field(default_factory=list)

    def reasons(self) -> Counter:
        return Counter(reason for _, reason in self.stale)


#: 评测记录 ``inputs`` 里必须有的字段（旧版本产出缺这些，没法核对上游，一律判过期）
REQUIRED_INPUTS = ("spec", "tree", "prompt", "gen", "messages", "reply", "code")


def _latest_by(records: Sequence[dict], key: tuple[str, ...]) -> dict[tuple, dict]:
    out: dict[tuple, dict] = {}
    for r in records:
        out[tuple(r.get(k) for k in key)] = r
    return out


def validate_records(records: Sequence[dict], ws: Workspace, specs: Optional[Mapping[str, Any]] = None
                     ) -> Validation:
    """逐条校验记录的输入哈希，返回（通过的记录, [(过期记录, 原因)]）。

    校验项：schema_version；``inputs`` 的必备字段；``inputs.prompt`` 等于当前 prompt 模板摘要；``inputs.code`` 等于当前评测相关
    源文件的内容哈希；对应的 ``obf-<corpus>-<level>.json`` 存在，``inputs.tree`` 等于它记录的副本源码树哈希，
    ``inputs.spec`` 等于它记录的 spec 摘要，``inputs.gate`` 等于它的 oracle gate 输入哈希；
    ``specs``（语料名 → spec）给了的话，再与当前 spec 的摘要比较；
    对应的生成记录（``generations-<model>.jsonl``）还在、输入哈希与回复文本都没变（重新 describe / generate 之后忘了重跑
    evaluate，旧回复的结果不能当有效数据）；``tasks.jsonl`` 里该任务的消息哈希没变（提示词 / 功能规格 / 目标集没变）。
    """
    cur_prompt = PROMPT_DIGEST[:16]
    cur_code = code_digest("evaluate")
    obf_cache: dict[tuple, Optional[dict]] = {}
    gens_cache: dict[str, dict[tuple, dict]] = {}
    tasks: Optional[dict[tuple, dict]] = None
    valid: list[dict] = []
    stale: list[tuple[dict, str]] = []

    def gens(model: str) -> dict[tuple, dict]:
        if model not in gens_cache:
            gens_cache[model] = _latest_by(read_jsonl(ws.generations_file(model)), ("task_id", "sample_idx"))
        return gens_cache[model]

    for rec in records:
        reason: Optional[str] = None
        inputs = rec.get("inputs") or {}
        if rec.get("schema_version") != SCHEMA_VERSION:
            reason = "schema_version 不符"
        elif inputs.get("prompt") != cur_prompt:
            reason = "prompt 模板已变"
        elif [k for k in REQUIRED_INPUTS if k not in inputs]:
            reason = "记录缺少输入哈希字段（旧版本产出）"
        elif inputs.get("code") != cur_code:
            reason = "评测代码已变（判定逻辑改过）"
        else:
            key = (rec.get("corpus"), rec.get("level"))
            if key not in obf_cache:
                obf_cache[key] = read_json(ws.obf_file(*key)) if all(key) else None
            obf = obf_cache[key]
            if not isinstance(obf, dict):
                reason = "缺少对应的混淆副本记录"
            elif inputs.get("tree") != obf.get("tree_hash"):
                reason = "混淆副本已变（源码树哈希不符）"
            elif inputs.get("spec") != (obf.get("inputs") or {}).get("spec"):
                reason = "spec 与生成副本时不一致"
            elif specs and rec.get("corpus") in specs and inputs.get("spec") != spec_digest(specs[rec["corpus"]]):
                reason = "spec 已变"
            elif inputs.get("gate") != str(((obf.get("oracle_gate") or {}).get("input_hash")) or ""):
                reason = "oracle gate 已重跑或被清除"
            else:
                g = gens(str(rec.get("model"))).get((rec.get("task_id"), rec.get("sample_idx")))
                if g is None:
                    reason = "对应的生成记录已不存在"
                elif g.get("input_hash") != inputs.get("gen") or digest(g.get("text") or "") != inputs.get("reply"):
                    reason = "回复已变（重新生成过，评测结果对应的是旧回复）"
                else:
                    if tasks is None:
                        tasks = _latest_by(read_jsonl(ws.tasks_file), ("task_id",))
                    t = tasks.get((rec.get("task_id"),))
                    if t is None:
                        reason = "任务已不在 tasks.jsonl 里（被作废或重新出题）"
                    elif t.get("messages_hash") != inputs.get("messages"):
                        reason = "提示词 / 功能规格已变（任务的消息哈希不符）"
        if reason is None:
            valid.append(rec)
        else:
            stale.append((rec, reason))
    return Validation(total=len(records), valid=valid, stale=stale)


# ----------------------------------------------------------------------
# 可信度记录的数据来源
# ----------------------------------------------------------------------
def gather_context(ws: Workspace, corpora: Sequence[str]) -> dict[str, Any]:
    """从工作区读报告要用的旁路信息：目标集与 stub-kill、混淆副本与 oracle gate、功能规格、出题数。

    注意这些读的是**当前**文件，与结果表里的评测记录不一定是同一时刻的产出——所以校验过的输入哈希（validate_records）
    才是结果可信的依据，这里的数字只是背景；报告末尾的「来源」一节列出了各份产出的摘要，供核对。
    """
    descs: dict[str, dict] = {}
    for rec in read_jsonl(ws.descriptions_file):
        if rec.get("target_id"):
            descs[rec["target_id"]] = rec
    tasks = Counter((r.get("corpus"), r.get("level")) for r in read_jsonl(ws.tasks_file))
    ctx: dict[str, Any] = {"corpora": {}}
    for c in corpora:
        targets = read_json(ws.targets_summary_file(c))
        levels = {}
        for lv in LEVELS:
            obf = read_json(ws.obf_file(c, lv))
            if isinstance(obf, dict):
                gate = obf.get("oracle_gate") or {}
                sym = (obf.get("stats") or {}).get("symbols") or {}
                levels[lv] = {
                    "verify_ok": obf.get("ok"), "renamed": len(obf.get("rename_map") or {}),
                    "strict": obf.get("strict_stringify"), "gate_checked": gate.get("checked"),
                    "gate_passed": gate.get("passed"), "gate_failed": [f.get("target_id") for f in gate.get("failed", [])],
                    "copy_invalid": gate.get("copy_invalid"), "tasks": tasks.get((c, lv), 0),
                    "problems": obf.get("problems") or [],
                    "stringified_renamed": len(sym.get("stringified_names_renamed") or []),
                    "tree": obf.get("tree_hash"), "seed": obf.get("seed"), "obf_hash": obf.get("input_hash"),
                    "gate_hash": gate.get("input_hash"),
                    "clang": (obf.get("inputs") or {}).get("clang"),
                }
        mine = [d for t, d in descs.items() if d.get("corpus") == c]
        ctx["corpora"][c] = {
            "targets": targets if isinstance(targets, dict) else None,
            "levels": levels,
            "descriptions": {"n": len(mine), "failed": sum(1 for d in mine if d.get("failed")),
                             "leak_retries": sum(int(d.get("leak_retries") or 0) for d in mine),
                             "phrase_warnings": sum(1 for d in mine if d.get("phrase_hits")),
                             "own_name_hits": sum(1 for d in mine if d.get("own_name_hit")),
                             "exempt_hits": sum(len(d.get("exempt_hits") or []) for d in mine),
                             "models": sorted({str(d.get("model")) for d in mine})},
        }
    return ctx


def gather_provenance(ws: Workspace, used: Sequence[dict]) -> dict[str, Any]:
    """报告的「来源」：参与统计的记录用了哪些沙箱、哪些模型配置、哪些采样参数。"""
    sandboxes = {digest(r.get("sandbox") or {}): r.get("sandbox") or {} for r in used}
    models: dict[str, dict[str, Any]] = {}
    for m in sorted({str(r.get("model")) for r in used}):
        gens = read_jsonl(ws.generations_file(m))
        models[m] = {
            "api_model": sorted({str(g.get("api_model")) for g in gens if g.get("api_model")}),
            "llm_fp": sorted({str(g.get("llm_fp")) for g in gens if g.get("llm_fp")}),
            "temperature": sorted({g.get("temperature") for g in gens if g.get("temperature") is not None}),
            "max_tokens": sorted({g.get("max_tokens") for g in gens if g.get("max_tokens") is not None}),
            "system_fingerprint": sorted({str(g.get("system_fingerprint")) for g in gens if g.get("system_fingerprint")}),
            "reply_model": sorted({str(g.get("reply_model")) for g in gens if g.get("reply_model")}),
        }
    return {"sandboxes": list(sandboxes.values()), "models": models, "prompt_digest": PROMPT_DIGEST[:16],
            "eval_code": code_digest("evaluate")}


def load_completeness(ws: Workspace, tag: str) -> list[dict]:
    """evaluate 写的完整性记录里的格（没有文件返回空列表）。"""
    data = read_json(ws.completeness_file(tag))
    return list(data.get("cells", [])) if isinstance(data, dict) else []


# ----------------------------------------------------------------------
# Markdown
# ----------------------------------------------------------------------
def _pct(x: Optional[float], digits: int = 1) -> str:
    return "-" if x is None else f"{x * 100:.{digits}f}%"


def _pp(x: Optional[float]) -> str:
    """比例差 → 带符号的百分点。"""
    return "-" if x is None else f"{x * 100:+.1f}"


def _num(x: Optional[float], digits: int = 3) -> str:
    return "-" if x is None else f"{x:.{digits}f}"


def _cell_text(x: Any) -> str:
    return str(x).replace("|", "\\|").replace("\n", " ")


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]], right_from: int = 2) -> str:
    """Markdown 表；从第 right_from 列起右对齐（数字列）。"""
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(("---:" if i >= right_from else "---") for i in range(len(headers))) + " |"
    body = ["| " + " | ".join(_cell_text(c) for c in row) + " |" for row in rows]
    return "\n".join([head, sep, *body])


def _missing_by_cell(completeness: Sequence[Mapping[str, Any]]) -> dict[tuple, Mapping[str, Any]]:
    return {(c.get("model"), c.get("level"), c.get("corpus")): c for c in completeness}


def _missing_for(completeness: Sequence[Mapping[str, Any]], model: str, level: str) -> tuple[int, int]:
    """（缺失数, 应有数）：某模型某级别在所有语料上的合计；没有完整性记录返回 (0, 0)。"""
    miss = exp = 0
    for c in completeness:
        if c.get("model") == model and c.get("level") == level:
            miss += int(c.get("missing") or 0)
            exp += int(c.get("expected") or 0)
    return miss, exp


def render_markdown(tag: str, cells_all: Sequence[Cell], cells: Sequence[Cell], drops: Sequence[Drop],
                    info: Mapping[str, Any], validation: Validation, context: Mapping[str, Any],
                    allow_stale: bool = False, *, completeness: Sequence[Mapping[str, Any]] = (),
                    bad_lines: int = 0, provenance: Optional[Mapping[str, Any]] = None) -> str:
    out: list[str] = [f"# 评测汇总 · {tag}", ""]
    out += _warning_banner(context, completeness, bad_lines)
    if not info["n_records"]:
        out += ["没有可用的评测记录。", ""]
        if validation.stale:
            out += _validation_section(validation, allow_stale)
        return "\n".join(out).rstrip() + "\n"

    out += [
        f"- 样本记录 {info['n_records']} 条；模型 {', '.join(info['models'])}；"
        f"语料 {', '.join(info['corpora'])}；混淆级别 {', '.join(info['levels'])}；不同目标 {info['n_targets']} 个。",
        "- 编译通过率 = 构建与链接成功、至少进入项目测试的比例（stage ≥ test）；通过率 = 项目自带测试全部通过的比例，"
        "区间是 Wilson 95% 置信区间。",
        "- 混淆级别：" + "；".join(f"{lv} {LEVEL_NOTES[lv]}" for lv in LEVELS)
        + "（高一级包含低一级的全部变换）。L0 通过率是上限参考：功能规格由 LLM 生成，规格歧义会抬高所有级别的失败率。",
        "",
    ]
    if validation.stale:
        out += _validation_section(validation, allow_stale)

    out += ["## 1. 总表：模型 × 混淆级别", ""]
    rows = []
    for c in cells_all:
        miss, exp = _missing_for(completeness, c.model, c.level)
        rows.append([c.model, c.level, c.n, "不完整（缺 %d/%d）" % (miss, exp) if miss else ("完整" if exp else "-"),
                     c.n_targets, _pct(c.compile_rate), _pct(c.pass_rate),
                     f"[{_pct(c.pass_ci[0], 0)}, {_pct(c.pass_ci[1], 0)}]", _pct(c.macro_pass_rate),
                     _num(c.mean_similarity), _pct(c.same_reply_as_l1) if c.level in ("L2", "L3") else "-",
                     c.prompt_tokens, c.completion_tokens, c.n_cached])
    out.append(_md_table(
        ["模型", "级别", "n", "样本完整性", "目标数", "编译通过率", "通过率", "95% 区间", "语料宏平均通过率", "平均相似度",
         "回复与 L1 相同", "提示 token", "补全 token", "命中缓存"], rows, right_from=2))
    out += ["", "平均相似度是候选代码与（对应混淆级的）原函数的归一化相似度，作为记忆痕迹指标：标识符一律归一，"
            "所以单纯改名不改变它；L0 → L1 相似度明显下降说明 L0 的高相似可能来自记忆。", "",
            "**L2 / L3 的数字请和「回复与 L1 相同」一起看**：" + REPLY_SHARING_NOTE, ""]

    out += ["## 2. 相对 L0 的变化（配对样本）", ""]
    ds = [d for d in drops if d.corpus == ALL]
    if ds:
        out.append(_md_table(
            ["模型", "级别", "配对数", "L0 通过率", "本级通过率", "Δ通过率(百分点)", "相对下降", "仅 L0 通过", "仅本级通过",
             "McNemar p", "Δ编译通过率(百分点)", "L0 相似度", "本级相似度", "回复与 L1 逐字相同的配对数"],
            [[d.model, d.level, d.n_pairs, _pct(d.pass_base), _pct(d.pass_level), _pp(d.delta_pass),
              _pct(d.rel_drop), d.only_base, d.only_level, _num(d.p_value), _pp(d.delta_compile),
              _num(d.sim_base), _num(d.sim_level),
              "-" if d.n_same_as_l1 is None else f"{d.n_same_as_l1}/{d.n_pairs}"] for d in ds], right_from=2))
        out += ["", "只比较 L0 与本级都有记录的 (语料, 目标, 样本)：oracle gate 会让不同级别作废不同的目标，直接比总体比例会把"
                "目标集合的差异当成混淆的效果；某一侧缺失（生成调用失败等）的样本自然不在配对里。McNemar p 是精确双侧检验，"
                "仅在两边都有不一致配对时才有值；同一目标有多个样本时样本并不独立，p 值偏乐观，只作参考。", "",
                "> **脚注**：" + REPLY_SHARING_NOTE, ""]
    else:
        out += ["没有同时含 L0 与其它级别的记录，无法计算相对 L0 的变化。", ""]

    out += ["## 3. 失败归因分布（占本格样本数的百分比）", ""]
    out.append(_md_table(
        ["模型", "级别", "n"] + [f"{c}（{CATEGORY_NOTES.get(c, c)}）" for c in CATEGORIES],
        [[c.model, c.level, c.n] + [_pct(c.shares.get(cat, 0.0)) for cat in CATEGORIES] for c in cells_all],
        right_from=2))
    out += ["", "一个样本只归一类，判定顺序见 `bench/attribute.py`。policy_violation = 候选调用了提示词明令禁止的函数"
            "（exit / abort 等）、写了内联汇编或 #define 等，不进沙箱、按失败计。", ""]
    out += ["### 回复规范性", "",
            _md_table(["模型", "级别", "n", "多个函数", "签名不符", "多余的 #include", "疑似截断", "违反硬性约束",
                       "libclang 检查没做"],
                      [[c.model, c.level, c.n] + [_pct(c.issue_counts.get(k, 0) / c.n if c.n else 0.0)
                                                  for k in ("multiple_functions", "signature_mismatch",
                                                            "added_include", "truncated", "policy_violation")]
                       + [c.n_analysis_skipped] for c in cells_all], right_from=2),
            "", "后处理记下的问题，一个回复可以同时有多个，所以同一行的百分比加起来可以超过 100%；没有代码块的回复在上表的 "
            "extract_fail 里。除「违反硬性约束」外，这些问题只记录、不丢弃代码，是否真的出错由编译器和测试判定。"
            "「libclang 检查没做」是解析候选代码失败、没能做多个函数 / 签名 / 禁用调用检查的回复数（0 才说明这几项检查是完整的）。", ""]
    out += _kill_mode_section(cells_all)

    out += ["## 4. 分语料明细", ""]
    drop_by = {(d.model, d.level, d.corpus): d for d in drops}
    miss_by = _missing_by_cell(completeness)
    rows = []
    for c in cells:
        d = drop_by.get((c.model, c.level, c.corpus))
        m = miss_by.get((c.model, c.level, c.corpus))
        rows.append([c.model, c.corpus, c.level, c.n, ("缺 %d" % m["missing"]) if m and m.get("missing") else "-",
                     _pct(c.compile_rate), _pct(c.pass_rate),
                     f"[{_pct(c.pass_ci[0], 0)}, {_pct(c.pass_ci[1], 0)}]", _pp(d.delta_pass) if d else "-",
                     _num(c.mean_similarity)])
    out.append(_md_table(["模型", "语料", "级别", "n", "缺失", "编译通过率", "通过率", "95% 区间", "Δ通过率(百分点)",
                          "平均相似度"], rows, right_from=3))
    out += [""]

    out += _credibility_section(context)
    out += _limitations_section()
    out += _provenance_section(context, provenance)
    return "\n".join(out).rstrip() + "\n"


def _warning_banner(context: Mapping[str, Any], completeness: Sequence[Mapping[str, Any]], bad_lines: int) -> list[str]:
    """报告开头的醒目警告：目标集没做 stub-kill / 执行过滤、结果文件有坏行、样本缺失。没有问题返回空列表。"""
    lines: list[str] = []
    unverified = []
    for name, c in (context.get("corpora") or {}).items():
        t = c.get("targets") or {}
        flt = t.get("filter") or {}
        if t.get("unverified") or flt.get("require_stub_kill") is False or flt.get("require_exec") is False:
            what = [w for w, off in (("stub-kill", flt.get("require_stub_kill") is False),
                                     ("执行次数过滤", flt.get("require_exec") is False)) if off]
            unverified.append(f"{name}（未做 {'、'.join(what) or '部分过滤'}）")
    if unverified:
        lines.append("> **警告：目标集未经完整的可信度过滤** —— " + "；".join(unverified)
                     + "。没做 stub-kill 的目标里混有「测试覆盖不住」的函数，通过率失去意义，这份报告只能用于调试，"
                     "不能当正式结果。")
    if bad_lines:
        lines.append(f"> **警告：结果文件有 {bad_lines} 条坏行**（例如进程被杀时留下的半行），对应样本没有计入统计，重新运行 evaluate 会补上。")
    miss = sum(int(c.get("missing") or 0) for c in completeness)
    if miss:
        exp = sum(int(c.get("expected") or 0) for c in completeness)
        failed = sum(int(c.get("failed_calls") or 0) for c in completeness)
        lines.append(f"> **警告：样本不完整** —— 应有 {exp} 个样本，缺 {miss} 个（其中 {failed} 个是生成调用失败）。"
                     "调用失败往往与提示词长度或难度相关，幸存样本偏容易，会抬高通过率；总表里标「不完整」的格不要直接和别的模型比。")
    return lines + [""] if lines else []


def _kill_mode_section(cells_all: Sequence[Cell]) -> list[str]:
    rows = []
    for c in cells_all:
        for mode in ("test_fail", "runtime_crash", "runtime_timeout"):
            slot = c.kill_modes.get(mode)
            if slot and slot["n"]:
                rows.append([c.model, c.level, mode, slot["n"], slot["n_pass"], _pct(slot["n_pass"] / slot["n"])])
    if not rows:
        return []
    return ["### 按 stub-kill 杀死方式分层的通过率", "",
            _md_table(["模型", "级别", "目标被杀死的方式", "n", "通过数", "通过率"], rows, right_from=3), "",
            "目标被零值桩杀死的方式：test_fail = 测试断言失败（测试检查了函数的返回内容）；runtime_crash / runtime_timeout = 桩让测试程序崩溃或超时，"
            "这类目标的测试并没有断言返回内容，任何不崩溃的错误实现都能通过，通过率会被系统性高估。读结果时优先看 test_fail 一行，"
            "崩溃型和超时型单列。", ""]


def _validation_section(validation: Validation, allow_stale: bool) -> list[str]:
    reasons = validation.reasons()
    verb = "仍然纳入统计" if allow_stale else "已排除，不进统计"
    lines = [f"> **输入校验**：共 {validation.total} 条记录，{len(validation.stale)} 条的输入哈希与当前不符，{verb}。"]
    for reason, n in reasons.most_common():
        lines.append(f"> - {reason}：{n} 条")
    lines.append("")
    return lines


def _credibility_section(context: Mapping[str, Any]) -> list[str]:
    out = ["## 5. 可信度记录", ""]
    corp = context.get("corpora") or {}
    if not corp:
        return out + ["没有可读的旁路记录（targets / obf / descriptions）。", ""]

    rows = []
    for name, c in corp.items():
        t = c.get("targets")
        if not t:
            continue
        ex = t.get("excluded") or {}
        flt = t.get("filter") or {}
        unchecked = flt.get("require_stub_kill") is False
        rows.append([name, t.get("extracted"), t.get("candidates"), t.get("selected"),
                     "未做" if unchecked else ex.get("stub_survived", 0),
                     ex.get("not_executed", 0), ex.get("few_statements", 0) + ex.get("too_short", 0)
                     + ex.get("too_long", 0), ex.get("stub_build_error", 0) + ex.get("stub_unsupported", 0),
                     ex.get("not_sampled", 0), "未做 stub-kill" if unchecked else _modes_text(t.get("stub_kill_modes"))])
    if rows:
        out += ["### 目标集与 stub-kill", "",
                _md_table(["语料", "提取函数数", "候选", "入选", "测试覆盖不住(stub 存活)", "未被测试执行", "规模不合格",
                           "桩无法构建", "名额已满未入选", "入选目标的杀死方式"], rows, right_from=1), "",
                "stub 存活 = 把函数体换成零值桩后项目测试仍通过，说明测试覆盖不住它，不能当目标；"
                "名额已满未入选 = 通过了全部过滤，但每个语料入选数有上限，按 seed 抽样落选；"
                "杀死方式 test_fail 是测试断言失败，runtime_crash 是桩让测试程序崩溃，runtime_timeout 是桩让测试超时。", ""]

    rows = []
    for name, c in corp.items():
        for lv, info in (c.get("levels") or {}).items():
            gate = "-" if info.get("gate_checked") is None else f"{info.get('gate_passed')}/{info.get('gate_checked')}"
            rows.append([name, lv, "通过" if info.get("verify_ok") else "未通过", info.get("renamed"),
                         "是" if info.get("strict") else "否", info.get("stringified_renamed", 0), gate,
                         len(info.get("gate_failed") or []), info.get("tasks")])
    if rows:
        out += ["### 混淆副本与 oracle gate", "",
                _md_table(["语料", "级别", "副本 verify", "改名数", "字符串化宏用 strict", "字符串化实参被改名数",
                           "oracle gate 通过/检查", "作废目标数", "出题数"], rows, right_from=3), "",
                "副本 verify = 项目自带测试全过且每个目标能按映射名重新定位；oracle gate = 用混淆后的原函数文本走一遍"
                "完整 evaluate 必须通过，不过的目标在该级别作废。字符串化实参被改名数 = 出现在 # 宏调用实参里、却照常改了名的拼写"
                "（只可能是测试文件里的；库源码里的一律排除）：这些名字被字符串化后的输出文字随之变化，而项目测试若不逐字比对输出，"
                "verify 发现不了；该列为 0 或使用 strict 才说明副本的输出文字与原件完全一致。", ""]

    rows = []
    for name, c in corp.items():
        d = c.get("descriptions") or {}
        if d.get("n"):
            rows.append([name, d["n"], d["failed"], d["leak_retries"], d.get("phrase_warnings", 0),
                         d.get("own_name_hits", 0), d.get("exempt_hits", 0), ", ".join(d.get("models") or [])])
    if rows:
        out += ["### 功能规格", "",
                _md_table(["语料", "规格数", "describe_failed", "泄漏重试次数", "词组警告的规格数",
                           "含目标自身函数名的规格数", "放行词命中总数", "生成模型"], rows, right_from=1), "",
                "describe_failed = 重试用尽仍泄漏项目标识符，该目标被排除。放行词 = 常见英文词 / spec.allow_words 里的项目标识符，"
                "出现在描述里不拦截但如实计数；含目标自身函数名 = 描述里用了函数自己的名字（只可能是常见英文词）；"
                "词组警告 = 多词名字的各个词以空格相邻出现。", ""]
    return out


def _limitations_section() -> list[str]:
    out = ["## 6. 局限", "", "与 README 第 8 节同一份内容（`bench/limitations.py` 单点维护）：", ""]
    for i, (title, text) in enumerate(LIMITATIONS, 1):
        out.append(f"{i}. **{title}。** {text}")
    return out + [""]


def _provenance_section(context: Mapping[str, Any], prov: Optional[Mapping[str, Any]]) -> list[str]:
    out = ["## 7. 来源", ""]
    if not prov:
        return out + ["没有可用的来源信息。", ""]
    out += [f"- prompt 模板摘要 {prov.get('prompt_digest')}；评测相关代码摘要 {prov.get('eval_code')}。"]
    for sb in prov.get("sandboxes", []):
        out.append("- 沙箱：" + ", ".join(f"{k}={v}" for k, v in sb.items()) if sb else "- 沙箱：（记录里没有沙箱信息）")
    for m, info in (prov.get("models") or {}).items():
        out.append(f"- 模型 {m}：" + "；".join(f"{k}={','.join(map(str, v))}" for k, v in info.items() if v))
    rows = []
    for name, c in (context.get("corpora") or {}).items():
        t = c.get("targets") or {}
        for lv, info in (c.get("levels") or {}).items():
            rows.append([name, lv, (t.get("inputs") or {}).get("seed", t.get("seed")), info.get("tree"), info.get("gate_hash"),
                         (info.get("clang") or {}).get("builtin_include") if info.get("clang") else "-"])
    if rows:
        out += ["", _md_table(["语料", "级别", "seed", "副本源码树哈希", "oracle gate 输入哈希", "宿主内置头摘要"], rows, right_from=2)]
    return out + [""]


def _modes_text(modes: Optional[Mapping[str, int]]) -> str:
    if not modes:
        return "-"
    return ", ".join(f"{k}={v}" for k, v in sorted(modes.items(), key=lambda kv: str(kv[0])))


# ----------------------------------------------------------------------
# SVG 折线图
# ----------------------------------------------------------------------
# 取自 dataviz 参考调色板（8 个类别色，已按相邻色对在浅色 / 深色表面上通过校验），按模型名排序后依次取色
_LIGHT = {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9",
          "axis": "#c3c2b7",
          "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]}
_DARK = {"surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a",
         "axis": "#383835",
         "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]}
_FONT = 'system-ui, -apple-system, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif'
MAX_SERIES = 8
BAND_MAX_SERIES = 4          # 超过这个线数就不画置信区间色带（叠在一起看不清），区间仍在悬停提示和汇总表里


def _css_vars(theme: Mapping[str, Any]) -> str:
    items = [f"--{k}:{theme[k]}" for k in ("surface", "ink", "ink2", "muted", "grid", "axis")]
    items += [f"--s{i + 1}:{c}" for i, c in enumerate(theme["series"])]
    return ";".join(items)


def _text_width(text: str, size: float) -> float:
    """粗略估算文字宽度：CJK 全角算 1 个字号，其余算 0.58 个字号（只用来排图例，不求精确）。"""
    return sum(size if ord(ch) > 0x2E80 else size * 0.58 for ch in text)


def render_svg(cells_all: Sequence[Cell], title: str = "") -> Optional[str]:
    """通过率与编译通过率随混淆级别的折线图（两个并排的面板）；没有数据返回 None。

    x 轴是混淆级别（有序），一条线一个模型；点是 model × level 合计的比例，色带是 Wilson 95% 区间。
    ≥2 个模型时画图例；每条线只在终点标一个数值，互相挤在一起（间距 < 13px）的终点标签不画，交给图例、悬停提示和汇总表。
    """
    cells = [c for c in cells_all if c.corpus == ALL and c.n > 0]
    if not cells:
        return None
    models = sorted({c.model for c in cells})
    note = ""
    if len(models) > MAX_SERIES:
        note = f"（模型多于 {MAX_SERIES} 个，只画前 {MAX_SERIES} 个）"
        models = models[:MAX_SERIES]
    levels = sorted({c.level for c in cells}, key=lambda lv: LEVEL_ORDER.index(lv) if lv in LEVEL_ORDER else 99)
    by = {(c.model, c.level): c for c in cells}

    pw, ph = 340, 236                 # 单个面板的宽高
    ml, mr, mt, mb = 46, 44, 46, 34
    gap, pad = 22, 16
    width = pad * 2 + pw * 2 + gap
    band_note = ("色带：Wilson 95% 置信区间；" if len(models) <= BAND_MAX_SERIES and len(levels) > 1
                 else "Wilson 95% 置信区间见悬停提示；")
    legend_rows = _legend_layout(models, width - 2 * pad) if len(models) >= 2 else []
    legend_h = 8 + 22 * len(legend_rows) if legend_rows else 0
    top = 62
    height = top + ph + legend_h + 34

    def panel(x0: float, y0: float, name: str, metric: str) -> list[str]:
        parts = [f'<g transform="translate({x0},{y0})">',
                 f'<text x="{ml}" y="22" class="ptitle">{escape(name)}</text>']
        plot_w, plot_h = pw - ml - mr, ph - mt - mb

        def yy(v: float) -> float:
            return mt + plot_h * (1 - v)

        def xx(i: int) -> float:
            return ml + (plot_w * i / (len(levels) - 1) if len(levels) > 1 else plot_w / 2)

        for tick in (0, 0.25, 0.5, 0.75, 1.0):
            y = yy(tick)
            cls = "axis" if tick == 0 else "grid"
            parts.append(f'<line x1="{ml}" x2="{ml + plot_w}" y1="{y:.1f}" y2="{y:.1f}" class="{cls}"/>')
            parts.append(f'<text x="{ml - 8}" y="{y + 4:.1f}" class="tick" text-anchor="end">{int(tick * 100)}%</text>')
        for i, lv in enumerate(levels):
            parts.append(f'<text x="{xx(i):.1f}" y="{mt + plot_h + 20}" class="tick" '
                         f'text-anchor="middle">{escape(lv)}</text>')
        ends: list[tuple[float, str, float]] = []
        for si, m in enumerate(models):
            pts = [(i, by[(m, lv)]) for i, lv in enumerate(levels) if (m, lv) in by]
            if not pts:
                continue
            col = f"var(--s{si + 1})"
            rate = lambda c: c.pass_rate if metric == "pass" else c.compile_rate        # noqa: E731
            ci = lambda c: c.pass_ci if metric == "pass" else c.compile_ci               # noqa: E731
            if len(pts) > 1:
                if len(models) <= BAND_MAX_SERIES:
                    upper = [f"{xx(i):.1f},{yy(ci(c)[1]):.1f}" for i, c in pts]
                    lower = [f"{xx(i):.1f},{yy(ci(c)[0]):.1f}" for i, c in reversed(pts)]
                    parts.append(f'<polygon points="{" ".join(upper + lower)}" fill="{col}" fill-opacity="0.10"/>')
                line = " ".join(f"{xx(i):.1f},{yy(rate(c)):.1f}" for i, c in pts)
                parts.append(f'<polyline points="{line}" fill="none" stroke="{col}" stroke-width="2" '
                             f'stroke-linejoin="round" stroke-linecap="round"/>')
            for i, c in pts:
                k = c.n_pass if metric == "pass" else c.n_compiled
                tip = (f"{m} · {c.level}\n{name} {rate(c) * 100:.1f}%（{k}/{c.n}）\n"
                       f"95% 区间 [{ci(c)[0] * 100:.0f}%, {ci(c)[1] * 100:.0f}%]")
                parts.append(f'<circle cx="{xx(i):.1f}" cy="{yy(rate(c)):.1f}" r="4" fill="{col}" '
                             f'stroke="var(--surface)" stroke-width="2"/>')
                parts.append(f'<circle cx="{xx(i):.1f}" cy="{yy(rate(c)):.1f}" r="12" fill="transparent">'
                             f'<title>{escape(tip)}</title></circle>')
            last_i, last = pts[-1]
            ends.append((yy(rate(last)), f"{rate(last) * 100:.0f}%", xx(last_i)))
        placed: list[float] = []
        for y, label, x in sorted(ends):
            if all(abs(y - p) >= 13 for p in placed):
                placed.append(y)
                parts.append(f'<text x="{x + 10:.1f}" y="{y + 4:.1f}" class="val">{label}</text>')
        parts.append("</g>")
        return parts

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'role="img" aria-label={quoteattr(title or "通过率随混淆级别的变化")}>',
        "<style>"
        f"svg{{{_css_vars(_LIGHT)}}}"
        f"@media (prefers-color-scheme: dark){{svg{{{_css_vars(_DARK)}}}}}"
        f"text{{font-family:{_FONT};fill:var(--ink2)}}"
        ".title{font-size:16px;font-weight:600;fill:var(--ink)}.sub{font-size:12px;fill:var(--muted)}"
        ".ptitle{font-size:13px;font-weight:600;fill:var(--ink)}.tick{font-size:11px;fill:var(--muted)}"
        ".val{font-size:11px;fill:var(--ink2)}.leg{font-size:12px;fill:var(--ink2)}"
        ".grid{stroke:var(--grid);stroke-width:1}.axis{stroke:var(--axis);stroke-width:1}"
        "</style>",
        f'<rect width="{width}" height="{height}" fill="var(--surface)"/>',
        f'<text x="{pad}" y="26" class="title">{escape(title or "通过率随混淆级别的变化")}</text>',
        f'<text x="{pad}" y="46" class="sub">点：该模型在该级别的样本合计比例；{escape(band_note)}'
        f'悬停看具体数值{escape(note)}</text>',
    ]
    svg += panel(pad, top - 8, "测试通过率", "pass")
    svg += panel(pad + pw + gap, top - 8, "编译通过率", "compile")
    if legend_rows:
        y = top - 8 + ph + 12
        for row in legend_rows:
            x = pad
            for si, label in row:
                svg.append(f'<circle cx="{x + 5}" cy="{y}" r="5" fill="var(--s{si + 1})"/>')
                svg.append(f'<text x="{x + 16}" y="{y + 4}" class="leg">{escape(label)}</text>')
                x += 16 + _text_width(label, 12) + 22
            y += 22
    svg.append(f'<text x="{pad}" y="{height - 12}" class="sub">'
               + escape("级别：" + "　".join(f"{lv} {LEVEL_NOTES.get(lv, '')}" for lv in levels)) + "</text>")
    svg.append("</svg>")
    return "\n".join(svg) + "\n"


def _legend_layout(models: Sequence[str], max_width: float) -> list[list[tuple[int, str]]]:
    rows: list[list[tuple[int, str]]] = [[]]
    used = 0.0
    for si, m in enumerate(models):
        w = 16 + _text_width(m, 12) + 22
        if rows[-1] and used + w > max_width:
            rows.append([])
            used = 0.0
        rows[-1].append((si, m))
        used += w
    return rows


# ----------------------------------------------------------------------
# 入口
# ----------------------------------------------------------------------
@dataclass
class ReportResult:
    tag: str
    n_records: int                  # results 文件里的记录条数
    n_used: int                     # 进入统计的记录条数
    n_stale: int                    # 输入哈希对不上的记录条数
    paths: dict[str, Path] = field(default_factory=dict)       # md / csv / svg -> 文件
    stale_reasons: dict[str, int] = field(default_factory=dict)
    n_bad_lines: int = 0            # 结果文件里解析不了的行数（半行等），对应样本没计入
    n_missing: int = 0              # evaluate 记下的缺失样本数（生成调用失败 / 没生成 / 回复过期）


def build_report(ws: Workspace, tag: str, *, specs: Optional[Mapping[str, Any]] = None, allow_stale: bool = False,
                 formats: Sequence[str] = ("md", "csv", "svg"), records: Optional[Sequence[dict]] = None
                 ) -> ReportResult:
    """生成 ``summary-<tag>.md`` / ``.csv`` / ``fig-<tag>.svg``（formats 里选）。

    ``records`` 给了就直接用（测试用），否则读 ``results-<tag>.jsonl``；``specs`` 用来多校验一层 spec 摘要。
    """
    if records is not None:
        raw, bad = list(records), 0
    else:
        raw, bad = load_results_checked(ws.results_file(tag))
    validation = validate_records(raw, ws, specs)
    used = raw if allow_stale else validation.valid
    cells_all, cells = summarize_all(used), summarize(used)
    drops = drop_vs_baseline(used)
    info = overview(used)
    ctx = gather_context(ws, info["corpora"] or sorted({r.get("corpus") for r in raw if r.get("corpus")}))
    completeness = load_completeness(ws, tag)
    prov = gather_provenance(ws, used)
    result = ReportResult(tag=tag, n_records=len(raw), n_used=len(used), n_stale=len(validation.stale),
                          stale_reasons=dict(validation.reasons()), n_bad_lines=bad,
                          n_missing=sum(int(c.get("missing") or 0) for c in completeness))
    ws.results.mkdir(parents=True, exist_ok=True)
    if "md" in formats:
        p = ws.summary_file(tag, ".md")
        p.write_text(render_markdown(tag, cells_all, cells, drops, info, validation, ctx, allow_stale,
                                     completeness=completeness, bad_lines=bad, provenance=prov), encoding="utf-8")
        result.paths["md"] = p
    if "csv" in formats:
        p = ws.summary_file(tag, ".csv")
        p.write_text(csv_text(cells_all + cells, drops), encoding="utf-8")
        result.paths["csv"] = p
    if "svg" in formats:
        svg = render_svg(cells_all, f"通过率随混淆级别的变化 · {tag}")
        if svg is None:
            log.info("没有可画的数据，跳过图")
        else:
            p = ws.summary_file(tag, ".svg")
            p.write_text(svg, encoding="utf-8")
            result.paths["svg"] = p
    return result
