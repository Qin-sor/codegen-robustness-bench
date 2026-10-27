"""指标汇总（设计文档 §4.14）：把 ``results-<tag>.jsonl`` 的逐样本记录汇总成 model × level × corpus 的指标。

输入是 ``stage_evaluate`` 写出的记录（dict），用到的字段：``model`` ``level`` ``corpus`` ``target_id`` ``sample_idx``
``stage`` ``category`` ``similarity`` ``prompt_tokens`` ``completion_tokens`` ``cached``。本模块不读文件系统以外的东西，
也不依赖 libclang 和沙箱，纯标准库，方便单测。

指标口径：

* ``n``：样本数（每个 (目标, 样本编号) 一条）；``n_targets``：其中不同目标的个数；
* ``compile_rate``：stage 达到 test 或 pass 的比例，即「设计文档的 stage ≥ test」。stage 的先后顺序是
  extract（抽不出代码）< policy（违反硬性约束，没进沙箱）< syntax < build < test < pass；
* ``pass_rate``：category == pass 的比例；两者都附 Wilson 95% 置信区间（小样本下比正态近似可靠）；
* 归因占比：每个 category 的样本数除以 n，所有占比加起来为 1；
* ``mean_similarity``：有候选代码的样本的归一化相似度均值（记忆痕迹指标，越接近 1 越像原函数）；
* token 用量：回复记录里的 prompt / completion token 之和（命中缓存的回复也计入，``n_cached`` 单独给出个数）；
* 回复规范性：后处理记下的 issue（多个函数、签名不符、多余的 #include、疑似截断……）各占多少样本，
  一个回复可以同时有多个问题，所以占比加起来可以超过 1；
* ``corpus == "ALL"`` 的行是把全部语料的样本合在一起算（按样本加权），同时给 ``macro_pass_rate``：各语料通过率的算术平均，
  避免大语料（目标多）淹没小语料。

「相对 L0 的下降」按配对样本算：只比较 L0 与 Lk 都有记录的 (语料, 目标, 样本编号)，因为 oracle gate 会让不同级别
作废不同的目标，直接比总体比例会把目标集合的差异当成混淆的效果。配对比较给出两个级别各自的通过率、差值、
只在一边通过的样本数，以及精确 McNemar 检验的双侧 p 值（样本数小时比正态近似可靠）。
注意同一目标的多个样本（samples > 1）并不独立，p 值在那种情况下偏乐观，只作参考。
"""
from __future__ import annotations

import csv
import io
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

from bench.store import read_jsonl_checked
from bench.vocab import CATEGORIES, ISSUES, LEVELS

__all__ = [
    "COMPILED_STAGES", "ISSUE_NAMES", "ALL", "Cell", "Drop", "load_results", "load_results_checked",
    "wilson_interval", "mcnemar_exact", "summarize", "summarize_all", "drop_vs_baseline", "overview",
    "dedupe_records",
]

#: stage 的先后顺序是 extract（抽不出代码）< policy（违反硬性约束，没进沙箱）< syntax < build < test < pass；
#: 只有 test 和 pass 算「编译通过」
COMPILED_STAGES = frozenset({"test", "pass"})
ISSUE_NAMES = ISSUES          # 回复后处理记下的问题（词表在 bench.vocab，与 postprocess.ISSUES 是同一个对象）
ALL = "ALL"
LEVEL_ORDER = LEVELS


# ----------------------------------------------------------------------
# 统计工具
# ----------------------------------------------------------------------
def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """比例 k/n 的 Wilson 置信区间（默认 95%）；n == 0 时返回 (0, 0)。"""
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar_exact(b: int, c: int) -> Optional[float]:
    """配对二分类的精确 McNemar 检验（双侧）：b、c 是两种不一致配对的个数。没有不一致配对时返回 None。"""
    n = b + c
    if n == 0:
        return None
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def _mean(values: Sequence[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


# ----------------------------------------------------------------------
# 读入与整理
# ----------------------------------------------------------------------
def load_results_checked(path: Path | str) -> tuple[list[dict], int]:
    """读 results-<tag>.jsonl，返回（记录, 坏行数）。坏行（例如进程被杀时留下的半行）会打警告并计数，
    调用方（report）据此在报告里写明、并让命令返回非零，不能悄悄少算样本。文件不存在返回（空列表, 0）。"""
    return read_jsonl_checked(path)


def load_results(path: Path | str) -> list[dict]:
    """读 results-<tag>.jsonl（坏行打警告后跳过）。文件不存在返回空列表。"""
    return load_results_checked(path)[0]


def _key(rec: dict) -> tuple:
    return rec.get("model"), rec.get("level"), rec.get("corpus"), rec.get("target_id"), rec.get("sample_idx")


def dedupe_records(records: Iterable[dict]) -> list[dict]:
    """同一 (模型, 级别, 语料, 目标, 样本编号) 多条记录时只留最后一条（追加写文件里后写的是重做的结果）。"""
    latest: dict[tuple, dict] = {}
    for rec in records:
        latest[_key(rec)] = rec
    return list(latest.values())


def _level_sort(level: str) -> int:
    return LEVEL_ORDER.index(level) if level in LEVEL_ORDER else len(LEVEL_ORDER)


def _compiled(rec: dict) -> bool:
    return rec.get("stage") in COMPILED_STAGES


def _passed(rec: dict) -> bool:
    return rec.get("category") == "pass"


# ----------------------------------------------------------------------
# 单元格
# ----------------------------------------------------------------------
@dataclass
class Cell:
    model: str
    level: str
    corpus: str                         # "ALL" 表示合计
    n: int = 0
    n_targets: int = 0
    n_compiled: int = 0
    n_pass: int = 0
    compile_rate: float = 0.0
    pass_rate: float = 0.0
    compile_ci: tuple[float, float] = (0.0, 0.0)
    pass_ci: tuple[float, float] = (0.0, 0.0)
    categories: dict[str, int] = field(default_factory=dict)
    shares: dict[str, float] = field(default_factory=dict)
    mean_similarity: Optional[float] = None
    n_similarity: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    n_cached: int = 0
    macro_pass_rate: Optional[float] = None     # 只有 corpus == "ALL" 才有
    issue_counts: dict[str, int] = field(default_factory=dict)     # issue 名 -> 带这个问题的样本数
    # L2 / L3 的提示词与 L1 逐字相同（只改函数体，提示词里没有函数体），temperature=0 下回复也相同：
    # 这两级衡量的是「同一份回复放回被结构 / 语义混淆过的项目后还能不能编过、测过」，不是模型对混淆代码的理解。
    # 这里给出该格样本里「回复与同一样本在 L1 的回复逐字相同」的比例（按回复摘要比较），读 L2 / L3 的数字时要看它。
    same_reply_as_l1: Optional[float] = None
    n_reply_compared: int = 0
    # stub-kill 杀死方式（test_fail 是断言失败，runtime_crash / runtime_timeout 是桩让测试程序崩溃 / 超时）→ {n, n_pass}。
    # 崩溃型 / 超时型目标的测试并没有断言函数返回了什么，任何不崩溃的错误实现都能通过，通过率会被系统性高估，所以分层给出
    kill_modes: dict[str, dict[str, int]] = field(default_factory=dict)
    n_analysis_skipped: int = 0          # libclang 解析失败、multiple_functions / signature_mismatch / 禁用调用检查没做的回复数

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model, "level": self.level, "corpus": self.corpus, "n": self.n,
            "n_targets": self.n_targets, "n_compiled": self.n_compiled, "n_pass": self.n_pass,
            "compile_rate": self.compile_rate, "pass_rate": self.pass_rate,
            "compile_ci_low": self.compile_ci[0], "compile_ci_high": self.compile_ci[1],
            "pass_ci_low": self.pass_ci[0], "pass_ci_high": self.pass_ci[1],
            "categories": dict(self.categories), "shares": dict(self.shares),
            "mean_similarity": self.mean_similarity, "n_similarity": self.n_similarity,
            "prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens,
            "n_cached": self.n_cached, "macro_pass_rate": self.macro_pass_rate,
            "issue_counts": dict(self.issue_counts),
            "same_reply_as_l1": self.same_reply_as_l1, "n_reply_compared": self.n_reply_compared,
            "kill_modes": {k: dict(v) for k, v in self.kill_modes.items()},
            "n_analysis_skipped": self.n_analysis_skipped,
        }


def _cell(model: str, level: str, corpus: str, recs: Sequence[dict]) -> Cell:
    n = len(recs)
    n_compiled = sum(_compiled(r) for r in recs)
    n_pass = sum(_passed(r) for r in recs)
    cats: dict[str, int] = {c: 0 for c in CATEGORIES}
    for r in recs:
        c = r.get("category") or "unknown"
        cats[c] = cats.get(c, 0) + 1
    sims = [float(r["similarity"]) for r in recs if r.get("similarity") is not None]
    issues: dict[str, int] = {k: 0 for k in ISSUE_NAMES}
    for r in recs:
        for name in set(r.get("issues") or []):
            issues[name] = issues.get(name, 0) + 1
    kill: dict[str, dict[str, int]] = {}
    for r in recs:
        mode = r.get("stub_kill_mode")
        if mode:
            slot = kill.setdefault(str(mode), {"n": 0, "n_pass": 0})
            slot["n"] += 1
            slot["n_pass"] += int(_passed(r))
    return Cell(
        model=model, level=level, corpus=corpus, n=n,
        n_targets=len({r.get("target_id") for r in recs}),
        n_compiled=n_compiled, n_pass=n_pass,
        compile_rate=n_compiled / n if n else 0.0, pass_rate=n_pass / n if n else 0.0,
        compile_ci=wilson_interval(n_compiled, n), pass_ci=wilson_interval(n_pass, n),
        categories=cats, shares={c: (v / n if n else 0.0) for c, v in cats.items()},
        mean_similarity=_mean(sims), n_similarity=len(sims),
        prompt_tokens=sum(int(r.get("prompt_tokens") or 0) for r in recs),
        completion_tokens=sum(int(r.get("completion_tokens") or 0) for r in recs),
        n_cached=sum(1 for r in recs if r.get("cached")),
        issue_counts=issues,
        kill_modes=kill,
        n_analysis_skipped=sum(1 for r in recs if r.get("analysis_skipped")),
    )


def _reply_digest(rec: dict) -> Optional[str]:
    inputs = rec.get("inputs")
    return inputs.get("reply") if isinstance(inputs, dict) else None


def _sample_key(rec: dict) -> tuple:
    return rec.get("model"), rec.get("corpus"), rec.get("target_id"), rec.get("sample_idx")


def _l1_replies(records: Iterable[dict]) -> dict[tuple, str]:
    """(模型, 语料, 目标, 样本) -> 该样本在 L1 的回复摘要。"""
    return {_sample_key(r): d for r in records if r.get("level") == "L1" and (d := _reply_digest(r))}


def _apply_reply_sharing(cell: Cell, recs: Sequence[dict], l1: dict[tuple, str]) -> None:
    """L2 / L3 的格：填「回复与 L1 逐字相同」的比例。记录里没有回复摘要（旧记录）或没有对应的 L1 记录的样本不参与。"""
    if cell.level in ("L0", "L1"):
        return
    n = same = 0
    for r in recs:
        d = _reply_digest(r)
        ref = l1.get(_sample_key(r))
        if d is None or ref is None:
            continue
        n += 1
        same += int(d == ref)
    cell.n_reply_compared = n
    cell.same_reply_as_l1 = same / n if n else None


def _group(records: Iterable[dict], key: Callable[[dict], tuple]) -> dict[tuple, list[dict]]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        groups[key(r)].append(r)
    return groups


def summarize(records: Iterable[dict]) -> list[Cell]:
    """model × level × corpus 的明细单元格，按 (模型, 级别, 语料) 排序。"""
    recs = dedupe_records(records)
    l1 = _l1_replies(recs)
    groups = _group(recs, lambda r: (r.get("model"), r.get("level"), r.get("corpus")))
    keys = sorted(groups, key=lambda k: (str(k[0]), _level_sort(str(k[1])), str(k[2])))
    out = []
    for m, lv, c in keys:
        cell = _cell(str(m), str(lv), str(c), groups[(m, lv, c)])
        _apply_reply_sharing(cell, groups[(m, lv, c)], l1)
        out.append(cell)
    return out


def summarize_all(records: Iterable[dict]) -> list[Cell]:
    """model × level 的合计单元格（corpus == "ALL"），带语料宏平均通过率。"""
    recs = dedupe_records(records)
    l1 = _l1_replies(recs)
    groups = _group(recs, lambda r: (r.get("model"), r.get("level")))
    keys = sorted(groups, key=lambda k: (str(k[0]), _level_sort(str(k[1]))))
    out = []
    for m, lv in keys:
        cell = _cell(str(m), str(lv), ALL, groups[(m, lv)])
        _apply_reply_sharing(cell, groups[(m, lv)], l1)
        per_corpus = _group(groups[(m, lv)], lambda r: (r.get("corpus"),))
        rates = [sum(_passed(r) for r in rs) / len(rs) for rs in per_corpus.values() if rs]
        cell.macro_pass_rate = _mean(rates)
        out.append(cell)
    return out


def overview(records: Iterable[dict]) -> dict[str, Any]:
    recs = dedupe_records(records)
    return {
        "n_records": len(recs),
        "models": sorted({str(r.get("model")) for r in recs}),
        "levels": sorted({str(r.get("level")) for r in recs}, key=_level_sort),
        "corpora": sorted({str(r.get("corpus")) for r in recs}),
        "n_targets": len({(r.get("corpus"), r.get("target_id")) for r in recs}),
    }


# ----------------------------------------------------------------------
# 相对 L0 的变化
# ----------------------------------------------------------------------
@dataclass
class Drop:
    model: str
    level: str
    corpus: str                          # "ALL" 或具体语料
    baseline: str = "L0"
    n_pairs: int = 0
    pass_base: float = 0.0
    pass_level: float = 0.0
    delta_pass: float = 0.0              # pass_level - pass_base（比例差，不是百分点；渲染时 ×100）
    rel_drop: Optional[float] = None     # (pass_base - pass_level) / pass_base，基线为 0 时 None
    only_base: int = 0                   # 基线通过、该级别不通过的配对数
    only_level: int = 0                  # 该级别通过、基线不通过的配对数
    p_value: Optional[float] = None
    compile_base: float = 0.0
    compile_level: float = 0.0
    delta_compile: float = 0.0
    sim_base: Optional[float] = None
    sim_level: Optional[float] = None
    delta_sim: Optional[float] = None
    n_same_as_l1: Optional[int] = None    # L2 / L3 的配对里，本级回复与 L1 回复逐字相同的样本数（L1 与其它级别为 None）

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _drop(model: str, level: str, corpus: str, baseline: str, pairs: Sequence[tuple[dict, dict]],
          l1: Optional[dict[tuple, str]] = None) -> Drop:
    n = len(pairs)
    pb = sum(_passed(a) for a, _ in pairs)
    pl = sum(_passed(b) for _, b in pairs)
    only_base = sum(1 for a, b in pairs if _passed(a) and not _passed(b))
    only_level = sum(1 for a, b in pairs if not _passed(a) and _passed(b))
    cb = sum(_compiled(a) for a, _ in pairs)
    cl = sum(_compiled(b) for _, b in pairs)
    # 相似度只在两边都有候选代码的配对上比较
    sims = [(float(a["similarity"]), float(b["similarity"])) for a, b in pairs
            if a.get("similarity") is not None and b.get("similarity") is not None]
    sim_b, sim_l = _mean([s[0] for s in sims]), _mean([s[1] for s in sims])
    pass_base, pass_level = (pb / n if n else 0.0), (pl / n if n else 0.0)
    same: Optional[int] = None
    if l1 is not None and level in ("L2", "L3"):
        same = sum(1 for _, b in pairs if _reply_digest(b) is not None and _reply_digest(b) == l1.get(_sample_key(b)))
    return Drop(
        model=model, level=level, corpus=corpus, baseline=baseline, n_pairs=n,
        pass_base=pass_base, pass_level=pass_level, delta_pass=pass_level - pass_base,
        rel_drop=((pass_base - pass_level) / pass_base) if pass_base > 0 else None,
        only_base=only_base, only_level=only_level, p_value=mcnemar_exact(only_base, only_level),
        compile_base=cb / n if n else 0.0, compile_level=cl / n if n else 0.0,
        delta_compile=(cl - cb) / n if n else 0.0,
        sim_base=sim_b, sim_level=sim_l,
        delta_sim=(sim_l - sim_b) if sim_b is not None and sim_l is not None else None,
        n_same_as_l1=same)


def drop_vs_baseline(records: Iterable[dict], baseline: str = "L0", per_corpus: bool = True) -> list[Drop]:
    """每个模型、每个非基线级别相对基线级别的配对变化；``per_corpus`` 为真时再按语料各给一行。"""
    recs = dedupe_records(records)
    l1 = _l1_replies(recs)
    by_model_level: dict[tuple, dict[tuple, dict]] = defaultdict(dict)
    for r in recs:
        by_model_level[(r.get("model"), r.get("level"))][(r.get("corpus"), r.get("target_id"), r.get("sample_idx"))] = r
    out: list[Drop] = []
    models = sorted({m for m, _ in by_model_level}, key=str)
    for m in models:
        base = by_model_level.get((m, baseline))
        if not base:
            continue
        levels = sorted({lv for mm, lv in by_model_level if mm == m and lv != baseline},
                        key=lambda lv: _level_sort(str(lv)))
        for lv in levels:
            other = by_model_level[(m, lv)]
            keys = sorted(set(base) & set(other), key=str)
            if not keys:
                continue
            out.append(_drop(str(m), str(lv), ALL, baseline, [(base[k], other[k]) for k in keys], l1))
            if per_corpus:
                for c in sorted({k[0] for k in keys}, key=str):
                    ks = [k for k in keys if k[0] == c]
                    out.append(_drop(str(m), str(lv), str(c), baseline, [(base[k], other[k]) for k in ks], l1))
    return out


# ----------------------------------------------------------------------
# CSV
# ----------------------------------------------------------------------
CSV_COLUMNS = (
    ["model", "level", "corpus", "n", "n_targets", "compile_rate", "pass_rate", "pass_ci_low", "pass_ci_high"]
    + [f"cat_{c}" for c in CATEGORIES]
    + [f"issue_{k}" for k in ISSUE_NAMES]
    + ["mean_similarity", "prompt_tokens", "completion_tokens", "n_cached", "macro_pass_rate",
       "delta_pass_vs_L0", "delta_compile_vs_L0", "delta_similarity_vs_L0", "mcnemar_p", "n_pairs_vs_L0"]
)


def _fmt(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def csv_text(cells: Sequence[Cell], drops: Sequence[Drop]) -> str:
    """summary CSV：每行一个单元格（明细 + 合计），归因占比是比例（0–1），delta 列是与 L0 的配对差。"""
    drop_by = {(d.model, d.level, d.corpus): d for d in drops}
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(CSV_COLUMNS)
    for c in cells:
        d = drop_by.get((c.model, c.level, c.corpus))
        row = [c.model, c.level, c.corpus, c.n, c.n_targets, c.compile_rate, c.pass_rate, c.pass_ci[0], c.pass_ci[1]]
        row += [c.shares.get(cat, 0.0) for cat in CATEGORIES]
        row += [(c.issue_counts.get(k, 0) / c.n if c.n else 0.0) for k in ISSUE_NAMES]
        row += [c.mean_similarity, c.prompt_tokens, c.completion_tokens, c.n_cached, c.macro_pass_rate]
        row += [d.delta_pass if d else None, d.delta_compile if d else None, d.delta_sim if d else None,
                d.p_value if d else None, d.n_pairs if d else None]
        w.writerow([_fmt(x) for x in row])
    return buf.getvalue()
