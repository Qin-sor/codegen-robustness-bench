"""实验记录生成：把真实语料的 prepare → coverage → targets → obfuscate → verify 产出汇总成 Markdown。

``docs/实验记录.md`` 的数据部分（环境来源、各阶段的表）由本模块从工作区的 ``results/*.json`` 生成，手写的说明和观察放在标记之外，
重新生成时只替换标记之间的内容::

    python3 -m bench.experiment_record --update docs/实验记录.md      # 读默认工作区的 results/，替换文件里的生成区
    python3 -m bench.experiment_record --workspace DIR --corpus cjson sds   # 只打印到标准输出

只读数据文件（和重新计算泄漏检查口径下的放行情况需要的源码树），不碰沙箱、不联网、不调用模型。
没有任何模型参与，所以这里没有通过率之类的数字；那些由 ``python3 -m bench report`` 产出。

缺陷修复任务的 ``mutate`` 阶段（``results/mutants-<corpus>.summary.json`` 与各级的 ``mutants-<corpus>-<level>.jsonl``）有产出时，
生成区末尾多一节「缺陷修复任务的变异体」：各语料各级的出题数 / 放弃数与原因、算子分布、与 L0 同算子 / 同位点的比例、杀死方式（含变异体怎么死的
与目标被零值桩杀死的方式的交叉表）、复核与记账、各类变异体占的沙箱时间、提示词上下文被裁掉条目的题、字符串字面量里的原名；
``check-mutants`` 回放过（``results/mutants-<corpus>.check.json``）就再加一张回放结果表。没有任何语料有 mutate 产出就不出这一节。
文档里这一节的数字都由这里从数据文件算出，不手写。
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from bench import corpus
from bench.metrics import comparable_by_level, repair_subsets
from bench.store import Workspace, read_json, read_jsonl
from bench.vocab import KILL_MODES, LEVELS, MUTATION_OPERATORS, MUTATION_TITLES

log = logging.getLogger(__name__)

__all__ = ["BEGIN", "END", "collect", "render", "update_document", "main"]

BEGIN = "<!-- BEGIN GENERATED: python3 -m bench.experiment_record -->"
END = "<!-- END GENERATED -->"

#: targets 汇总里的排除原因 → 表里的列（顺序即列序）
_EXCLUDE_COLUMNS = (
    ("few_statements", "语句数不足"), ("too_short", "行数过短"), ("too_long", "行数过长"), ("spec_excluded", "spec 排除"),
    ("not_executed", "未被测试执行"), ("stub_survived", "stub 存活"), ("stub_build_error", "桩无法构建"),
    ("stub_unsupported", "返回类型无法造桩"), ("not_sampled", "名额已满未入选"),
)
_KILL_MODES = ("test_fail", "runtime_crash", "runtime_timeout")


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]], align: Optional[Sequence[str]] = None) -> list[str]:
    """Markdown 表格；align 给每列 'l' 或 'r'（默认第一列左对齐，其余右对齐）。"""
    align = align or ["l"] + ["r"] * (len(headers) - 1)
    sep = [":---:" if a == "c" else ("---:" if a == "r" else "---") for a in align]
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(sep) + " |"]
    out += ["| " + " | ".join(str(x) for x in row) + " |" for row in rows]
    return out


def _get(d: Optional[Mapping[str, Any]], *path: str, default: Any = None) -> Any:
    cur: Any = d
    for key in path:
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur


def collect(ws: Workspace, corpora: Sequence[str]) -> dict[str, dict[str, Any]]:
    """读各语料的阶段产出；缺的文件记为 None / 空。"""
    data: dict[str, dict[str, Any]] = {}
    for name in corpora:
        data[name] = {
            "prepare": read_json(ws.prepare_file(name)),
            "coverage": read_json(ws.coverage_file(name)),
            "summary": read_json(ws.targets_summary_file(name)),
            "targets": read_jsonl(ws.targets_file(name)),
            "obf": {lv: read_json(ws.obf_file(name, lv)) for lv in LEVELS},
            "mutants": read_json(ws.mutants_summary_file(name)),
            "mutant_records": {lv: read_jsonl(ws.mutants_file(name, lv)) for lv in LEVELS},
            "mutants_check": read_json(ws.mutants_check_file(name)),
        }
    return data


def _distinct(values: Sequence[Any]) -> str:
    """一组值去重后拼起来；全一样时就是那个值，不一样时都列出来（读者一眼能看到来源不一致）。"""
    seen: list[str] = []
    for v in values:
        if v not in (None, "") and str(v) not in seen:
            seen.append(str(v))
    return " / ".join(f"`{v}`" for v in seen) if seen else "-"


def _provenance(data: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """环境与来源：种子、沙箱镜像、宿主 libclang 环境、各阶段的评测相关代码摘要。"""
    summaries = [d["summary"] for d in data.values() if d["summary"]]
    obfs = [r for d in data.values() for r in d["obf"].values() if r]
    gates = [r["oracle_gate"] for r in obfs if r.get("oracle_gate")]
    rows = [
        ("随机种子", _distinct([s.get("seed") for s in summaries])),
        ("沙箱镜像", _distinct([_get(s, "sandbox", "image") for s in summaries])),
        ("镜像 ID", _distinct([_get(s, "sandbox", "image_id") for s in summaries])),
        ("沙箱资源", _distinct([f"cpus={_get(s, 'sandbox', 'cpus')} memory={_get(s, 'sandbox', 'memory')} "
                              f"pids={_get(s, 'sandbox', 'pids')}" for s in summaries])),
        ("宿主 libclang（python 绑定 / 库文件 / 内置头摘要）",
         _distinct([f"{_get(r, 'inputs', 'clang', 'python_binding')} / {_get(r, 'inputs', 'clang', 'library')} / "
                    f"{_get(r, 'inputs', 'clang', 'builtin_include') or '-'}" for r in obfs if r["level"] != "L0"])),
        ("代码摘要：prepare", _distinct([_get(d["prepare"], "inputs", "code") for d in data.values()])),
        ("代码摘要：coverage", _distinct([_get(d["coverage"], "inputs", "code") for d in data.values()])),
        ("代码摘要：targets", _distinct([_get(s, "inputs", "code") for s in summaries])),
        ("代码摘要：obfuscate", _distinct([_get(r, "inputs", "code") for r in obfs])),
        ("代码摘要：oracle gate", _distinct([_get(g, "inputs", "code") for g in gates])),
    ]
    return _table(["项", "值"], rows, ["l", "l"])


def _prepare_coverage(data: Mapping[str, Mapping[str, Any]], specs: Mapping[str, Any]) -> list[str]:
    rows = []
    for name, d in data.items():
        p, c = d["prepare"], d["coverage"]
        spec = specs.get(name)
        commit = (spec.commit if spec else (p or {}).get("commit")) or ""
        rows.append((
            name, f"`{commit[:12]}`" if commit else "-",
            f"`{_get(p, 'inputs', 'tree', default='-')}`",
            "通过" if p and p.get("ok") else ("未通过" if p else "缺"),
            f"{p.get('duration_s', 0):.1f}" if p else "-",
            _get(c, "n_gcda", default="-"), _get(c, "n_gcda_failed", default="-"),
            _get(c, "n_functions", default="-"), _get(c, "n_executed", default="-"),
        ))
    return _table(["语料", "commit", "源码树哈希", "沙箱里构建与测试", "耗时(s)", ".gcda 数", "gcov 失败", "函数数", "执行过的函数数"], rows,
                  ["l", "l", "l", "l"] + ["r"] * 5)


def _targets(data: Mapping[str, Mapping[str, Any]]) -> list[str]:
    rows = []
    total = {k: 0 for k in ("extracted", "candidates", "selected")}
    total_ex = {k: 0 for k, _ in _EXCLUDE_COLUMNS}
    total_modes = {k: 0 for k in _KILL_MODES}
    for name, d in data.items():
        s = d["summary"]
        if not s:
            rows.append((name,) + ("缺",) * (3 + len(_EXCLUDE_COLUMNS) + 1))
            continue
        ex = s.get("excluded") or {}
        modes = s.get("stub_kill_modes") or {}
        for k in total:
            total[k] += int(s.get(k) or 0)
        for k, _ in _EXCLUDE_COLUMNS:
            total_ex[k] += int(ex.get(k, 0))
        for k in _KILL_MODES:
            total_modes[k] += int(modes.get(k, 0))
        rows.append((name, s.get("extracted"), s.get("candidates"), s.get("selected"),
                     *[ex.get(k, 0) for k, _ in _EXCLUDE_COLUMNS],
                     ", ".join(f"{k}={modes[k]}" for k in _KILL_MODES if modes.get(k)) or "-"))
    rows.append(("**合计**", total["extracted"], total["candidates"], total["selected"],
                 *[total_ex[k] for k, _ in _EXCLUDE_COLUMNS],
                 ", ".join(f"{k}={total_modes[k]}" for k in _KILL_MODES if total_modes[k]) or "-"))
    heads = ["语料", "提取函数数", "候选", "入选"] + [label for _, label in _EXCLUDE_COLUMNS] + ["入选目标的杀死方式"]
    return _table(heads, rows, ["l"] + ["r"] * (len(heads) - 2) + ["l"])


def _obfuscation(data: Mapping[str, Mapping[str, Any]]) -> list[str]:
    rows = []
    for name, d in data.items():
        for lv in LEVELS:
            r = d["obf"][lv]
            if not r:
                rows.append((name, lv) + ("缺",) * 9)
                continue
            sym = _get(r, "stats", "symbols", default={}) or {}
            gate = r.get("oracle_gate") or {}
            ver = r.get("verify") or {}
            renamed = sym.get("entities_renamed")
            rows.append((
                name, lv,
                renamed if lv != "L0" else "-",
                sym.get("entities_excluded", "-") if lv != "L0" else "-",
                ("是" if r.get("strict_stringify") else "否") if lv != "L0" else "-",
                len(sym.get("stringified_names_renamed") or []) if lv != "L0" else "-",
                _get(r, "stats", "structure", "total", default="-"),
                _get(r, "stats", "semantic", "total", default="-"),
                ("通过" if ver.get("ok") else "未通过") + f"（定位 {ver.get('relocated', '-')}/{ver.get('n_targets', '-')}）" if ver else "-",
                f"{gate.get('passed', '-')}/{gate.get('checked', '-')}" if gate else "-",
            ))
    heads = ["语料", "级别", "改名数", "未改名的项目拼写", "字符串化宏用 strict", "字符串化实参被改名数", "结构变换次数", "语义变换次数",
             "副本 verify", "oracle gate 通过/检查"]
    return _table(heads, rows, ["l", "l"] + ["r"] * 6 + ["l", "r"])


def _problems(data: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """需要读者注意的事：回退到 strict 的、gate 没过的目标、副本 verify 的问题。"""
    notes: list[str] = []
    for name, d in data.items():
        for lv in LEVELS:
            r = d["obf"][lv]
            if not r:
                continue
            attempts = _get(r, "stats", "attempts", default=[]) or []
            if len(attempts) > 1:
                notes.append(f"- {name}@{lv}：符号混淆先用非 strict，verify 未过，退回 strict（共 {len(attempts)} 次尝试）。")
            for prob in r.get("problems") or []:
                notes.append(f"- {name}@{lv}：{prob}")
            gate = r.get("oracle_gate") or {}
            if gate.get("copy_invalid"):
                notes.append(f"- {name}@{lv}：副本 verify 没过，整级作废。")
            for f in gate.get("failed") or []:
                notes.append(f"- {name}@{lv}：目标 `{f.get('target_id')}` 没过 oracle gate（{f.get('stage', '?')}）。")
            not_renamed = _get(r, "stats", "targets_not_renamed", default=[]) or []
            if not_renamed:
                notes.append(f"- {name}@{lv}：{len(not_renamed)} 个目标函数名没被改掉（出题时作废）：{', '.join(not_renamed[:6])}。")
    return notes or ["- 没有：所有副本 verify 通过、所有目标通过 oracle gate、没有回退到 strict。"]


def _checkers(data: Mapping[str, Mapping[str, Any]], ws: Workspace, specs: Mapping[str, Any]) -> dict[str, Any]:
    """每个语料的泄漏检查器（项目标识符 + spec.allow_words）；找不到 spec 或原始源码的语料没有。
    放行情况（:func:`_exempt_audit`）和修复提示词里的名字泄漏检查（:func:`_name_leaks`）共用，libclang 分析只做一次。"""
    from bench.describe import LeakChecker
    from bench.pipeline import project_names
    out: dict[str, Any] = {}
    for name in data:
        spec, root = specs.get(name), ws.pristine(name)
        if spec is not None and root.is_dir():
            out[name] = LeakChecker(project_names(spec, root), spec.allow_words)
    return out


def _exempt_audit(data: Mapping[str, Mapping[str, Any]], checkers: Mapping[str, Any]) -> list[str]:
    """泄漏检查口径下的放行情况：项目标识符里有多少被常见英文词表 / spec.allow_words 放行，入选目标自己的函数名有多少在其中。"""
    rows = []
    for name, d in data.items():
        checker = checkers.get(name)
        if checker is None:
            rows.append((name, "缺", "-", "-", "-", "-"))
            continue
        own = sorted({t["name"] for t in d["targets"] if t.get("name") in checker.exempt})
        allow = sorted(n for n in checker.exempt if n.casefold() in checker.allow_words)
        rows.append((name, len(checker.names), len(checker.exempt), len(allow), len(own), ", ".join(own) or "-"))
    return _table(["语料", "参与检查的项目标识符（≥3 字符）", "被放行", "其中靠 spec.allow_words", "入选目标自己的函数名被放行", "这些函数名"],
                  rows, ["l", "r", "r", "r", "r", "l"])


def _mutant_entry(summary: Optional[Mapping[str, Any]], level: str) -> Mapping[str, Any]:
    levels = (summary or {}).get("levels") if isinstance(summary, Mapping) else None
    ent = levels.get(level) if isinstance(levels, Mapping) else None
    return ent if isinstance(ent, Mapping) else {}


def _fmt_counts(d: Mapping[str, Any], keys: Sequence[str]) -> str:
    return ", ".join(f"{k}={d[k]}" for k in keys if d.get(k)) or "-"


def _ratio(k: int, n: int) -> str:
    return f"{k}/{n}（{k / n * 100:.1f}%）" if n else "-"


def _corpus_rows(data: Mapping[str, Mapping[str, Any]], make_row: Callable[[Mapping[str, Any]], Sequence[Any]],
                 n_cols: int) -> list[tuple]:
    """每个语料一行：有 mutate 汇总的用 ``make_row(汇总)``，没有的标「缺」（不是 0）。"""
    rows: list[tuple] = []
    for name, d in data.items():
        m = d.get("mutants")
        rows.append((name, *make_row(m)) if isinstance(m, Mapping) else (name,) + ("缺",) * n_cols)
    return rows


def _records_by_target(d: Mapping[str, Any]) -> dict[str, dict[str, Mapping[str, Any]]]:
    """某个语料的 mutants 记录 -> {target_id: {级别: 记录}}。"""
    out: dict[str, dict[str, Mapping[str, Any]]] = {}
    for lv in LEVELS:
        for r in (d.get("mutant_records") or {}).get(lv) or []:
            out.setdefault(str(r.get("target_id")), {})[lv] = r
    return out


_LITERAL_RE = re.compile(r'"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'')
_WORD_RE = re.compile(r"[A-Za-z_]\w*")
_LONG_WORD_RE = re.compile(r"[A-Za-z_]{3,}")


def _name_leaks(data: Mapping[str, Mapping[str, Any]], checkers: Mapping[str, Any]) -> list[str]:
    """修复提示词里的函数文本是否还带着符号混淆本该隐藏的原名：L1–L3 的变异体代码（已去注释）里，字符串 / 字符常量之外
    出现被改名的原拼写的题数（应当是 0）；字符串 / 字符常量里出现这类拼写（长度 ≥ 3 的词）的题数，再按泄漏检查器的放行名单分成两类：
    词全在放行名单里的（常见英文词，或 spec.allow_words 里的词；放行不等于与领域无关，mpc 的 ``Regex`` / ``Grammar`` 就是库的领域词）
    和含放行名单之外的项目词的，以及目标函数**自己的原名**出现在字面量里的题数（这类题出题时作废，``name_in_literal``）。
    字面量改了就改变行为，去不掉。"""
    rows = []
    tot = [0] * 6
    for name, d in data.items():
        if not any((d.get("mutant_records") or {}).get(lv) for lv in LEVELS[1:]):
            rows.append((name, "缺", "缺", "缺", "缺", "缺", "缺"))                    # 没有变异体记录：没检查过，不是「0 个泄漏」
            continue
        exempt = getattr(checkers.get(name), "exempt", None)
        n = outside = inside = common = domain = own = 0
        for lv in LEVELS[1:]:
            obf = (d.get("obf") or {}).get(lv) or {}
            renamed = set((obf.get("rename_map") or {}))
            for r in (d.get("mutant_records") or {}).get(lv) or []:
                code = str(r.get("mutant_code") or "")
                words = {w for w in _LONG_WORD_RE.findall(" ".join(_LITERAL_RE.findall(code))) if w in renamed}
                target_name = str(r.get("target_id") or "").rsplit(":", 1)[-1]
                n += 1
                outside += int(any(w in renamed for w in _WORD_RE.findall(_LITERAL_RE.sub(" ", code))))
                inside += int(bool(words))
                common += int(bool(words) and exempt is not None and all(w in exempt for w in words))
                domain += int(bool(words) and (exempt is None or any(w not in exempt for w in words)))
                own += int(target_name in words)
        rows.append((name, n, outside, inside, common, domain, own))
        tot = [x + y for x, y in zip(tot, (n, outside, inside, common, domain, own))]
    rows.append(("**合计**", *tot))
    return _table(["语料", "L1–L3 的题数", "字面量之外仍含被改名的原名", "字符串 / 字符常量里含原名（≥ 3 字符的词）",
                   "其中词全在放行名单里（常见英文词 / spec.allow_words）", "其中含放行名单之外的项目词", "目标函数自己的原名在字面量里"], rows)


def _comment_audit(data: Mapping[str, Mapping[str, Any]], ws: Workspace) -> list[str]:
    """函数体里的注释：入选目标里有多少个的函数文本带注释，其中多少个的注释提到了会被 L1 改名的项目标识符（长度 ≥ 3 的词）。
    符号混淆只改标识符，注释里仍是原名，原样放进修复提示词会泄漏本该隐藏的名字，所以变异体先去注释（附录 §12 #7）。
    按目标 L0 的原函数文本（原始源码树里按偏移切出来）统计。"""
    rows = []
    tot = [0, 0, 0]
    for name, d in data.items():
        root = ws.pristine(name)
        renamed = set(((d.get("obf") or {}).get("L1") or {}).get("rename_map") or {})
        if not d["targets"] or not root.is_dir() or not renamed:
            rows.append((name, "缺", "缺", "缺"))
            continue
        n = with_comment = mention = 0
        for t in d["targets"]:
            try:
                code = (root / t["file"]).read_bytes()[t["start"]:t["end"]].decode("utf-8", errors="replace")
            except OSError:
                continue
            n += 1
            comments = _comment_text(code)
            with_comment += bool(comments)
            mention += any(w in renamed for w in _LONG_WORD_RE.findall(comments))
        rows.append((name, n, with_comment, mention))
        tot = [tot[0] + n, tot[1] + with_comment, tot[2] + mention]
    rows.append(("**合计**", *tot))
    return _table(["语料", "入选目标", "函数体带注释", "其中注释里提到了会被 L1 改名的标识符"], rows)


def _comment_text(code: str) -> str:
    """函数文本里的注释部分（原文减去去掉注释后的文本，逐段拼起来）；没有注释返回空串。"""
    from bench.extract import strip_c_comments
    stripped = strip_c_comments(code)
    if stripped == code:
        return ""
    return " ".join(m.group(1) for m in _COMMENT_RE.finditer(code) if m.group(1))


_COMMENT_RE = re.compile(r'"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'|(/\*.*?\*/|//[^\n]*)', re.S)


def _overview(data: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """出题与放弃：各级出题数、放弃数与原因、出题率、四级齐全 / 同算子 / 同位点的目标数、有注释的题、耗时。"""
    tot = {"targets": 0, "killed": 0, "searched": 0, "abandoned": 0, "full": 0, "same4": 0, "site4": 0, "comment": 0,
           "checked": 0, "failed": 0, "elapsed": 0.0}
    per_level = {lv: 0 for lv in LEVELS}
    tot_why: dict[str, int] = {}
    searched_of: dict[str, int] = {}

    def row(m: Mapping[str, Any]) -> tuple:
        ents = {lv: _mutant_entry(m, lv) for lv in LEVELS}
        killed = {lv: int(ents[lv].get("n_killed") or 0) for lv in LEVELS}
        searched = sum(int(e.get("n_searched") or 0) for e in ents.values())
        abandoned = sum(int(e.get("n_abandoned") or 0) for e in ents.values())
        why: dict[str, int] = {}
        for e in ents.values():
            for k, v in (e.get("outcomes") or {}).items():
                if k != "killed":
                    why[k] = why.get(k, 0) + int(v)
        four = m.get("four_level") or {}
        comment = sum(int(e.get("n_with_comments") or 0) for e in ents.values())
        checked = sum(int(e.get("n_strip_checked") or 0) for e in ents.values())
        failed = sum(int(e.get("n_strip_failed") or 0) for e in ents.values())
        tot["targets"] += int(m.get("n_targets") or 0)
        tot["killed"] += sum(killed.values())
        tot["searched"] += searched
        tot["abandoned"] += abandoned
        tot["full"] += int(four.get("n_all_levels") or 0)
        tot["same4"] += int(four.get("n_same_operator") or 0)
        tot["site4"] += int(four.get("n_same_site") or 0)
        tot["comment"] += comment
        tot["checked"] += checked
        tot["failed"] += failed
        tot["elapsed"] += float(m.get("elapsed_s") or 0)
        for lv in LEVELS:
            per_level[lv] += killed[lv]
        for k, v in why.items():
            tot_why[k] = tot_why.get(k, 0) + v
        searched_of["n"] = searched_of.get("n", 0) + searched
        return (m.get("n_targets"), " / ".join(str(killed[lv]) for lv in LEVELS), sum(killed.values()), abandoned,
                _fmt_counts(why, sorted(why)), f"{sum(killed.values()) / searched * 100:.1f}%" if searched else "-",
                four.get("n_all_levels", "-"), four.get("n_same_operator", "-"), four.get("n_same_site", "-"),
                f"{comment} / {checked} / {failed}", f"{float(m.get('elapsed_s') or 0):.1f}")

    rows = _corpus_rows(data, row, 10)
    rows.append(("**合计**", tot["targets"], " / ".join(str(per_level[lv]) for lv in LEVELS), tot["killed"], tot["abandoned"],
                 _fmt_counts(tot_why, sorted(tot_why)), f"{tot['killed'] / tot['searched'] * 100:.1f}%" if tot["searched"] else "-",
                 tot["full"], tot["same4"], tot["site4"], f"{tot['comment']} / {tot['checked']} / {tot['failed']}", f"{tot['elapsed']:.1f}"))
    return ["#### 出题与放弃", ""] + _table(
        ["语料", "目标数", "出题数 L0 / L1 / L2 / L3", "出题数合计", "放弃数", "不出题的原因", "出题率", "四级都出了题的目标",
         "其中四级同算子", "其中四级同位点", "有注释的题 / 去注释后复验 / 复验失败", "耗时(s)"], rows,
        ["l", "r", "r", "r", "r", "l", "r", "r", "r", "r", "r", "r"])


def _operator_tables(data: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """算子分布：语料 × 算子（四级合计），以及全部语料按级别。"""
    by_level = {lv: {op: 0 for op in MUTATION_OPERATORS} for lv in LEVELS}
    total = {op: 0 for op in MUTATION_OPERATORS}

    def row(m: Mapping[str, Any]) -> tuple:
        ops = {op: 0 for op in MUTATION_OPERATORS}
        for lv in LEVELS:
            for op, n in (_mutant_entry(m, lv).get("operators") or {}).items():
                if op in ops:
                    ops[op] += int(n)
                    by_level[lv][op] += int(n)
                    total[op] += int(n)
        return (*[ops[op] for op in MUTATION_OPERATORS], sum(ops.values()))

    rows = _corpus_rows(data, row, len(MUTATION_OPERATORS) + 1)
    rows.append(("**合计**", *[total[op] for op in MUTATION_OPERATORS], sum(total.values())))
    heads = ["语料"] + list(MUTATION_OPERATORS) + ["合计"]
    return (["", "#### 算子分布（出了题的变异体，L0–L3 合计）", ""] + _table(heads, rows)
            + ["", "#### 算子分布（全部语料，按级别）", ""]
            + _table(["级别"] + list(MUTATION_OPERATORS) + ["合计"],
                     [(lv, *[by_level[lv][op] for op in MUTATION_OPERATORS], sum(by_level[lv].values())) for lv in LEVELS]))


def _site_tables(data: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """L1–L3 的变异体跟 L0 的关系：同算子（复用算子保证的，几乎恒为全部）、同位点（mutate 先试 L0 的位点）、同（算子, 序号）、
    以及同位点是怎么来的（锚点的去向）。"""
    out: list[str] = []
    for key, title in (("same_operator_as_L0", "#### 与 L0 同算子的比例（same_operator_as_L0：本级算子 = 该目标在 L0 选中的算子）"),
                       ("same_site_as_L0", "#### 与 L0 同位点的比例（same_site_as_L0：本级选中的就是 L0 位点在本级的对应位点）")):
        tsame = {lv: [0, 0] for lv in LEVELS[1:]}

        def row(m: Mapping[str, Any], key: str = key, tsame: dict = tsame) -> tuple:
            cells = []
            for lv in LEVELS[1:]:
                same = _mutant_entry(m, lv).get(key) or {}
                n_, k_ = int(same.get("n") or 0), int(same.get("same") or 0)
                tsame[lv][0] += n_
                tsame[lv][1] += k_
                cells.append(_ratio(k_, n_))
            return tuple(cells)

        rows = _corpus_rows(data, row, len(LEVELS) - 1)
        rows.append(("**合计**", *[_ratio(tsame[lv][1], tsame[lv][0]) for lv in LEVELS[1:]]))
        out += ["", title, ""] + _table(["语料"] + list(LEVELS[1:]), rows)

    # 同（算子, 序号）：L1 的位点表与 L0 一一对应，序号相同就是同一个位点；L2 / L3 改写了函数体，序号会漂，只作对照
    rows = []
    tot = {lv: [0, 0] for lv in LEVELS[1:]}
    for name, d in data.items():
        per = _records_by_target(d)
        if not per:
            rows.append((name,) + ("缺",) * (len(LEVELS) - 1))
            continue
        cells = []
        for lv in LEVELS[1:]:
            pairs = [(r["L0"], r[lv]) for r in per.values() if "L0" in r and lv in r]
            k = sum(1 for a, b in pairs if (a.get("operator"), a.get("site_index")) == (b.get("operator"), b.get("site_index")))
            tot[lv][0] += len(pairs)
            tot[lv][1] += k
            cells.append(_ratio(k, len(pairs)))
        rows.append((name, *cells))
    rows.append(("**合计**", *[_ratio(tot[lv][1], tot[lv][0]) for lv in LEVELS[1:]]))
    out += ["", "#### 与 L0 选中同一个（算子, 算子内序号）的比例（L1 精确对应；L2 / L3 的序号会漂，只作对照）", ""]
    out += _table(["语料"] + list(LEVELS[1:]), rows)

    # 锚点的去向
    rows = []
    totals = {lv: Counter() for lv in LEVELS[1:]}
    for name, d in data.items():
        m = d.get("mutants")
        if not isinstance(m, Mapping):
            rows.append((name,) + ("缺",) * (len(LEVELS) - 1))
            continue
        cells = []
        for lv in LEVELS[1:]:
            anchor = _mutant_entry(m, lv).get("anchor") or {}
            totals[lv].update({k: int(v) for k, v in anchor.items()})
            cells.append(_fmt_counts(anchor, ("chosen", "tried", "no_image", "none")))
        rows.append((name, *cells))
    rows.append(("**合计**", *[_fmt_counts(totals[lv], ("chosen", "tried", "no_image", "none")) for lv in LEVELS[1:]]))
    out += ["", "#### L0 位点在本级的去向（chosen = 对应位点被选中；tried = 试了但不是稳定的杀死，退回算子级复用；"
                "no_image = 本级找不到对应位点；none = 没有参照）", ""]
    out += _table(["语料"] + list(LEVELS[1:]), rows, ["l", "l", "l", "l"])
    return out


def _subset_table(data: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """报告里几种「跨级别可比」的目标集合有多大（``metrics.repair_subsets``，与报告同一个函数，这里喂的是 mutants 记录）：
    四级都出了题、四级同算子（恒等于前者）、四级同位点、四级的提示词上下文都没有被裁条目、可比子集（后两者的交集），
    以及「本级与 L0 同位点、两级上下文都干净」的配对目标数（相对 L0 的配对比较用）。"""
    def as_result(d: Mapping[str, Any]) -> list[dict]:
        return [{"corpus": r.get("corpus"), "target_id": r.get("target_id"), "level": lv,
                 "mutation": {"operator": r.get("operator")}, "same_site_as_L0": r.get("same_site_as_L0"),
                 "context_orphans": r.get("context_orphans")}
                for lv in LEVELS for r in (d.get("mutant_records") or {}).get(lv) or []]

    rows = []
    tot = [0] * 9
    for name, d in data.items():
        recs = as_result(d)
        if not recs:
            rows.append((name,) + ("缺",) * 8)
            continue
        s = repair_subsets(recs)
        pairs = comparable_by_level(recs)
        vals = [len(s.all_targets), len(s.four_levels), len(s.same_operator), len(s.same_site), len(s.clean_context),
                len(s.comparable), *[len(pairs[lv]) for lv in LEVELS[1:]]]
        rows.append((name, *vals[:6], " / ".join(str(v) for v in vals[6:]), ""))
        tot = [a + b for a, b in zip(tot, [*vals, 0, 0][:9])]
    rows = [r[:-1] for r in rows]
    rows.append(("**合计**", *tot[:6], " / ".join(str(v) for v in tot[6:9])))
    return ["", "#### 可比子集的大小（报告里「可比子集」一节用同一组函数）", ""] + _table(
        ["语料", "目标", "四级都出了题", "四级同算子", "四级同位点", "四级上下文没有被裁条目", "可比子集（同位点且上下文干净）",
         "与 L0 配对可比的目标数 L1 / L2 / L3"], rows, ["l", "r", "r", "r", "r", "r", "r", "r"])


def _kill_tables(data: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """变异体被杀死的方式（按级别），以及它与目标被零值桩杀死的方式的交叉表：两个口径不等价。"""
    tmodes = {lv: {k: 0 for k in KILL_MODES} for lv in LEVELS}

    def row(m: Mapping[str, Any]) -> tuple:
        cells = []
        for lv in LEVELS:
            modes = _mutant_entry(m, lv).get("kill_modes") or {}
            for k in KILL_MODES:
                tmodes[lv][k] += int(modes.get(k) or 0)
            cells.append(" / ".join(str(int(modes.get(k) or 0)) for k in KILL_MODES))
        return tuple(cells)

    rows = _corpus_rows(data, row, len(LEVELS))
    rows.append(("**合计**", *[" / ".join(str(tmodes[lv][k]) for k in KILL_MODES) for lv in LEVELS]))
    out = ["", "#### 变异体被杀死的方式（" + " / ".join(KILL_MODES) + "）", ""] + _table(["语料"] + list(LEVELS), rows)

    cross: Counter = Counter()
    for d in data.values():
        stub = {t.get("target_id"): t.get("stub_kill_mode") for t in d.get("targets") or []}
        for lv in LEVELS:
            for r in (d.get("mutant_records") or {}).get(lv) or []:
                cross[(stub.get(r.get("target_id")) or "-", r.get("kill_mode") or "-")] += 1
    if cross:
        rows = []
        for sm in (*KILL_MODES, "-"):
            vals = [cross[(sm, km)] for km in KILL_MODES]
            if any(vals):
                rows.append((sm, *vals, sum(vals)))
        n_all = sum(cross.values())
        rows.append(("**合计**", *[sum(cross[(sm, km)] for sm in (*KILL_MODES, "-")) for km in KILL_MODES], n_all))
        weak_dies_hard = sum(cross[("test_fail", km)] for km in KILL_MODES[1:])
        strong_dies_soft = sum(cross[(sm, "test_fail")] for sm in KILL_MODES[1:])
        n_test = sum(cross[("test_fail", km)] for km in KILL_MODES)
        n_nontest = sum(cross[(sm, km)] for sm in KILL_MODES[1:] for km in KILL_MODES)
        out += ["", "#### 两个口径的交叉表：目标被零值桩杀死的方式（行）× 变异体怎么死的（列），L0–L3 合计", ""]
        out += _table(["目标的 stub_kill_mode \\ 变异体的 kill_mode", *KILL_MODES, "合计"], rows)
        out += ["", f"目标的桩是 test_fail 的 {n_test} 个变异体里，有 {weak_dies_hard} 个（{_pct(weak_dies_hard, n_test)}）是崩溃 / 超时杀死的；"
                    f"目标的桩是崩溃 / 超时的 {n_nontest} 个变异体里，有 {strong_dies_soft} 个（{_pct(strong_dies_soft, n_nontest)}）记为 test_fail"
                    f"（共 {n_all} 个变异体）。所以「变异体怎么死」不能当作「测试断不断言返回值」的代用口径，报告按两个口径各分一层。"]
    return out


def _pct(k: int, n: int) -> str:
    return f"{k / n * 100:.1f}%" if n else "-"


def _accounting_tables(data: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """kill 检查的记账（四级合计）：位点、进沙箱试的、等价、编译失败、解析失败，以及复核的次数、不稳定的个数、杀死方式在复核时变了的个数。"""
    keys = ("n_sites", "tried_sites", "equivalent_count", "compile_fail_count", "parse_fail_count", "n_rechecked",
            "unstable_count", "n_kill_label_changed")

    totals = [0] * len(keys)

    def row(m: Mapping[str, Any]) -> tuple:
        ents = [_mutant_entry(m, lv) for lv in LEVELS]
        vals = tuple(sum(int(e.get(k) or 0) for e in ents) for k in keys)
        for i, v in enumerate(vals):
            totals[i] += v
        return vals

    rows = _corpus_rows(data, row, len(keys))
    rows.append(("**合计**", *totals))
    out = ["", "#### kill 检查的记账（四级合计）", ""] + _table(
        ["语料", "可改位点总数", "进沙箱试的位点", "等价变异体（编译通过、测试仍全过）", "编译失败", "重新解析有新 error 而没进沙箱",
         "复核次数", "复核时不稳定（第一次判为杀死、复核时没失败）", "杀死方式在复核时变了"], rows)
    by_level = []
    for lv in LEVELS:
        ents = [_mutant_entry(d.get("mutants"), lv) for d in data.values()]
        by_level.append((lv, *[sum(int(e.get(k) or 0) for e in ents) for k in ("n_killed", "tried_sites", "equivalent_count", "n_rechecked")]))
    return out + ["", "#### kill 检查的记账（按级别，全部语料）", ""] + _table(
        ["级别", "出题数", "进沙箱试的位点", "其中等价变异体", "复核次数"], by_level)


def _sandbox_time_table(data: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """kill 检查（含复核与去注释验证）的沙箱运行时间按归因类别累计：各类变异体占多少评测时间。"""
    secs: Counter = Counter()
    for d in data.values():
        m = d.get("mutants")
        for lv in LEVELS:
            for k, v in (_mutant_entry(m, lv).get("sandbox_seconds") or {}).items():
                secs[k] += float(v)
    total = sum(secs.values())
    if not total:
        return []
    rows = [(k, f"{v:.1f}", f"{v / total * 100:.1f}%") for k, v in sorted(secs.items(), key=lambda kv: -kv[1])]
    per_corpus, n_timeout_all, s_all, s_timeout_all = [], 0, 0.0, 0.0
    for name, d in data.items():
        m = d.get("mutants")
        if not isinstance(m, Mapping):
            per_corpus.append((name, "缺", "缺", "缺"))
            continue
        s: Counter = Counter()
        n_timeout = 0
        for lv in LEVELS:
            ent = _mutant_entry(m, lv)
            for k, v in (ent.get("sandbox_seconds") or {}).items():
                s[k] += float(v)
            n_timeout += int((ent.get("kill_modes") or {}).get("runtime_timeout") or 0)
        per_corpus.append((name, n_timeout, f"{sum(s.values()):.1f}", f"{s['runtime_timeout']:.1f}"))
        n_timeout_all += n_timeout
        s_all += sum(s.values())
        s_timeout_all += s["runtime_timeout"]
    per_corpus.append(("**合计**", n_timeout_all, f"{s_all:.1f}", f"{s_timeout_all:.1f}"))
    return ["", "#### 沙箱运行时间按归因类别（kill 检查、复核与去注释验证的运行时间之和，全部语料）", ""] + _table(
        ["归因类别", "沙箱运行时间(s)", "占比"], rows) + [
        "", f"沙箱运行时间合计 {total:.1f} 秒（各容器并发，不等于墙钟时间）。pass = 等价变异体、去注释的原函数；runtime_timeout = 要等满 spec 的 "
            "timeout_s 的那些运行。",
        "", "#### 沙箱运行时间按语料（超时型运行集中在哪几个语料）", ""] + _table(
        ["语料", "runtime_timeout 型变异体", "沙箱运行时间合计(s)", "其中 runtime_timeout 的运行(s)"], per_corpus)


def _orphans_table(data: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """提示词上下文被裁掉条目的题（缺陷让原型 / 宏 / 全局变得多余，prompt.prune_context）：按算子和按级别。"""
    by_op: dict[str, list[int]] = {op: [0, 0] for op in MUTATION_OPERATORS}
    by_level = {lv: [0, 0] for lv in LEVELS}
    names: Counter = Counter()
    for d in data.values():
        for lv in LEVELS:
            for r in (d.get("mutant_records") or {}).get(lv) or []:
                has = bool(r.get("context_orphans"))
                op = str(r.get("operator"))
                if op in by_op:
                    by_op[op][0] += 1
                    by_op[op][1] += has
                by_level[lv][0] += 1
                by_level[lv][1] += has
                names.update(r.get("context_orphans") or [])
    if not any(n for n, _ in by_level.values()):
        return []
    rows = [(f"{op}（{MUTATION_TITLES.get(op, op)}）", n, k, _pct(k, n)) for op, (n, k) in by_op.items() if n]
    out = ["", "#### 提示词上下文里因缺陷而多余的条目（原型 / 宏 / 全局）：有这类条目的题（按算子，L0–L3 合计）", ""]
    out += _table(["算子", "题数", "上下文被裁掉条目的题", "占比"], rows)
    out += ["", "按级别：" + "；".join(f"{lv} {k}/{n}" for lv, (n, k) in by_level.items()) +
            "。这些条目留在提示词里等于把答案线索写给模型（例如 `return f(a, b);` 被换成 `return NULL;` 之后上下文仍列着 f 的原型），"
            "所以出题时裁掉，这些题不进可比子集。"]
    return out


def _check_table(data: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """``check-mutants`` 回放复核的结果（有产出的语料才出表）。"""
    rows = []
    tot: Counter = Counter()
    with_oracle = False
    for name, d in data.items():
        c = d.get("mutants_check")
        if not isinstance(c, Mapping):
            continue
        total = c.get("total") or {}
        tot.update({k: int(v) for k, v in total.items()})
        with_oracle = with_oracle or bool(c.get("with_oracle"))
        rows.append((name, total.get("replayed", 0), total.get("killed", 0), total.get("same_mode", 0), total.get("passed", 0),
                     total.get("other", 0), len(c.get("flipped") or []),
                     total.get("oracle_failed", 0) if c.get("with_oracle") else "-"))
    if not rows:
        return []
    rows.append(("**合计**", tot["replayed"], tot["killed"], tot["same_mode"], tot["passed"], tot["other"],
                 sum(len((d.get("mutants_check") or {}).get("flipped") or []) for d in data.values()),
                 tot["oracle_failed"] if with_oracle else "-"))
    return ["", "#### 回放复核（`python3 -m bench check-mutants`）：变异体当作候选再评测一遍", ""] + _table(
        ["语料", "回放的变异体", "仍被杀死", "杀死方式与 mutate 记录相同", "回放通过了测试（不合格）", "其它（不合格）",
         "杀死方式变了", "去注释的原函数没通过"], rows)


def _mutants(data: Mapping[str, Mapping[str, Any]], checkers: Optional[Mapping[str, Any]] = None,
             ws: Optional[Workspace] = None) -> list[str]:
    """缺陷修复任务的变异体（mutate 阶段的汇总）：出题与放弃、算子分布、与 L0 的关系（同算子 / 同位点）、杀死方式、复核与记账、
    沙箱时间、上下文被裁的题、名字泄漏检查、回放复核。"""
    out = _overview(data) + _operator_tables(data) + _site_tables(data) + _subset_table(data) + _kill_tables(data)
    out += _accounting_tables(data)
    out += _sandbox_time_table(data) + _orphans_table(data)
    out += ["", "#### 修复提示词里的名字泄漏检查", ""] + _name_leaks(data, checkers or {})
    if ws is not None:
        out += ["", "#### 函数体里的注释（变异体先去注释再出题）", ""] + _comment_audit(data, ws)
    out += _check_table(data)
    return out


def render(ws: Workspace, corpora: Sequence[str]) -> str:
    """生成区的 Markdown（不含标记行）。"""
    data = collect(ws, corpora)
    specs: dict[str, Any] = {}
    for name in corpora:
        try:
            specs[name] = corpus.load_spec(name)
        except (FileNotFoundError, corpus.SpecError):
            log.warning("找不到语料 %s 的 spec", name)
    out: list[str] = ["### 环境与来源", ""] + _provenance(data)
    out += ["", "### prepare 与 coverage", ""] + _prepare_coverage(data, specs)
    out += ["", "### 目标选择与 stub-kill", ""] + _targets(data)
    out += ["", "### 混淆副本与 oracle gate", ""] + _obfuscation(data)
    out += ["", "### 需要注意的事", ""] + _problems(data)
    checkers = _checkers(data, ws, specs)
    out += ["", "### 泄漏检查的放行情况", ""] + _exempt_audit(data, checkers)
    if any(isinstance(d.get("mutants"), Mapping) for d in data.values()):
        out += ["", "### 缺陷修复任务的变异体（mutate）", ""] + _mutants(data, checkers, ws)
    return "\n".join(out) + "\n"


def update_document(path: Path, generated: str) -> None:
    """把文件里 BEGIN / END 两行标记之间的内容换成 generated；缺标记就报错（不猜位置）。"""
    text = Path(path).read_text(encoding="utf-8")
    try:
        head, rest = text.split(BEGIN, 1)
        _old, tail = rest.split(END, 1)
    except ValueError:
        raise SystemExit(f"{path} 里没有生成区标记：\n{BEGIN}\n{END}") from None
    Path(path).write_text(f"{head}{BEGIN}\n\n{generated}\n{END}{tail}", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m bench.experiment_record", description=__doc__.split("\n")[0])
    ap.add_argument("--workspace", metavar="DIR", help="工作区目录（默认用项目目录）")
    ap.add_argument("--corpus", nargs="+", metavar="NAME", help="语料名（默认 corpus/specs/ 下的全部）")
    ap.add_argument("--update", metavar="FILE", help="替换这个 Markdown 文件里标记之间的生成区；不给就打印到标准输出")
    args = ap.parse_args(argv)
    ws = Workspace.under(args.workspace) if args.workspace else Workspace.default()
    text = render(ws, args.corpus or corpus.list_specs())
    if args.update:
        update_document(Path(args.update), text)
        print(f"已更新 {args.update}")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
