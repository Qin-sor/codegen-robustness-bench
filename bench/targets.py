"""目标函数选择与 stub-kill（设计文档 §4.5）。

``select_targets`` 的过滤顺序，以及每一步被排除时记入 ``Selection.excluded`` 的原因名：

======================  =====================================================================
spec_excluded           spec.exclude_targets 明确排除的函数（如编进目标文件的测试驱动）
few_statements          语句数 < min_statements
too_short / too_long    定义行数 < min_lines / > max_lines
not_executed            项目测试里执行次数为 0（覆盖率数据里没有该函数也算 0）；只在 require_exec 时检查
stub_survived           用零值桩替换后项目测试仍然通过：测试覆盖不住这个函数，不能作为目标
stub_build_error        桩没能通过构建（构建阶段失败），无法判断，保守起见排除
stub_unsupported        返回类型无法构造桩（匿名 struct 等）
not_sampled             候选已经够 max_per_corpus 个，后面的没有再检查
======================  =====================================================================

（``main`` 与「单语句的 static inline」由 ``extract.extract_functions`` 先行排除，不在这里计数。）

抽样：候选按 ``sha256(f"{seed}|{target_id}")`` 排序，再按这个顺序逐个做 stub-kill，凑够 ``max_per_corpus`` 个
被杀死的函数为止。排序键只和 seed、target_id 有关，所以同一个 seed 在任何机器、任何并发度下选出同一批目标。
默认只检查到凑够名额为止，所以 ``stub_survived`` 等计数只覆盖「检查过的」候选，没检查的记在 ``not_sampled``；
要得到整个语料「测试覆盖不住的函数」的完整数字（写进报告用），传 ``scan_all=True``：所有候选都做 stub-kill，
入选的目标与默认模式完全相同，``not_sampled`` 变成「杀得死但名额已满」的个数。

桩的构造（设计文档 §4.5）：只替换函数体 ``[body_start, body_end)``，函数头保持原文（宏、属性、注释都不动）：

* 返回 void   → ``{ }``
* 返回指针或算术类型（含 enum、_Bool、浮点）→ ``{ return 0; }``
* 返回 struct / union / complex → ``{ static T _bench_zero; return _bench_zero; }``。
  静态存储期的变量天然零初始化，不依赖 ``memset`` / ``<string.h>``，也不用管 ``= {0}`` 的括号规则；
  返回类型的拼写取自 libclang（去掉 const / volatile）。

判定：桩替换后跑构建 + 项目测试，阶段为 test（测试失败、崩溃或超时）才算「杀死」；
阶段为 build（桩没编过）单独记 stub_build_error，不会因为构建失败而被误当成杀死。
杀死方式记在 ``StubVerdict.signal / timed_out``，``kill_mode`` 按 attribute 的归因顺序（崩溃信号、超时、其余）换算成
runtime_crash / runtime_timeout / test_fail，写进 targets-<corpus>.jsonl 的 ``stub_kill_mode``：同一份桩交给
evaluate + attribute 得到的类别应当与它一致，集成自测（tests/test_e2e_mock.py）靠这一点交叉验证两条独立的实现路径。
已经算好的函数执行次数（key 为 ``file:name``）可以用 ``select(..., precomputed_counts=...)`` 传入，免得重复跑覆盖率构建。
入选前还会在未改动的副本上跑一次基线（构建 + 测试必须通过），否则所有桩都会「被杀死」，结果没有意义。
"""
from __future__ import annotations

import hashlib
import logging
import re
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import clang.cindex as ci

from bench import extract
from bench.corpus import CorpusSpec, materialize
from bench.attribute import CRASH_SIGNALS
from bench.coverage import function_exec_counts
from bench.extract import FunctionTarget
from bench.runner import new_workdir, run_project, splice
from bench.store import SCHEMA_VERSION

log = logging.getLogger(__name__)

__all__ = [
    "TargetFilter", "Selection", "StubVerdict", "ReturnInfo", "TargetsError",
    "select_targets", "select", "stub_kill", "stub_verdict", "return_infos", "stub_body", "build_stub",
]

class TargetsError(RuntimeError):
    """基线就不通过等，无法继续选目标。"""


@dataclass
class TargetFilter:
    min_statements: int = 3
    min_lines: int = 4
    max_lines: int = 120
    max_per_corpus: int = 50
    require_exec: bool = True
    require_stub_kill: bool = True


@dataclass
class ReturnInfo:
    kind: str           # void / scalar / pointer / record / unsupported
    spelling: str       # 返回类型的拼写（record 时用来声明零值变量）


@dataclass
class StubVerdict:
    verdict: str        # killed / survived / build_error / unsupported
    stage: str | None = None    # 桩版本跑出的阶段（build / test / pass）；unsupported 时为 None
    detail: str = ""
    signal: int | None = None   # killed 时测试程序收到的信号（崩溃型杀死）
    timed_out: bool = False     # killed 时测试是否超时（死循环型杀死）

    @property
    def killed(self) -> bool:
        return self.verdict == "killed"

    @property
    def kill_mode(self) -> str | None:
        """杀死方式，取值与 attribute 的归因类别一致：runtime_crash / runtime_timeout / test_fail。

        同一份桩代码交给 evaluate + attribute 得到的类别应当与它相同（先判崩溃信号，再判超时，
        其余为测试失败），所以集成自测可以用它预先确定「wrong 模式的候选应归哪一类」。未被杀死返回 None。
        """
        if not self.killed:
            return None
        if self.signal in CRASH_SIGNALS:
            return "runtime_crash"
        if self.timed_out:
            return "runtime_timeout"
        return "test_fail"


@dataclass
class Selection:
    corpus: str
    seed: int
    filter: dict[str, Any]
    targets: list[FunctionTarget]                       # 入选目标，按 (file, 偏移) 排序
    exec_counts: dict[str, int]                         # target_id -> 执行次数（require_exec=False 时为空）
    stub_verdicts: dict[str, StubVerdict]               # target_id -> 桩判定（检查过的全部候选，含被排除的）
    excluded: dict[str, int]                            # 原因 -> 个数
    excluded_ids: dict[str, list[str]]                  # 原因 -> target_id 列表（排查用）
    n_extracted: int                                    # extract_functions 给出的函数数
    n_candidates: int                                   # 通过静态过滤与执行次数过滤的函数数

    def summary(self) -> dict[str, Any]:
        return {"corpus": self.corpus, "seed": self.seed, "filter": self.filter,
                "extracted": self.n_extracted, "candidates": self.n_candidates,
                "selected": len(self.targets), "excluded": dict(self.excluded)}

    def records(self) -> list[dict[str, Any]]:
        """results/targets-<corpus>.jsonl 的行：FunctionTarget 全字段 + exec_count + stub_killed + schema_version。"""
        out = []
        for t in self.targets:
            rec = t.to_dict()
            rec["exec_count"] = self.exec_counts.get(t.target_id)
            v = self.stub_verdicts.get(t.target_id)
            rec["stub_killed"] = None if v is None else v.killed
            rec["stub_kill_mode"] = None if v is None else v.kill_mode
            rec["schema_version"] = SCHEMA_VERSION
            out.append(rec)
        return out


# ----------------------------------------------------------------------
# 返回类型与桩
# ----------------------------------------------------------------------
def _kinds(*names: str) -> frozenset:
    return frozenset(getattr(ci.TypeKind, n) for n in names if hasattr(ci.TypeKind, n))


_SCALAR_KINDS = _kinds(
    "BOOL", "CHAR_U", "UCHAR", "CHAR16", "CHAR32", "USHORT", "UINT", "ULONG", "ULONGLONG", "UINT128",
    "CHAR_S", "SCHAR", "WCHAR", "SHORT", "INT", "LONG", "LONGLONG", "INT128", "FLOAT", "DOUBLE",
    "LONGDOUBLE", "HALF", "FLOAT128", "ENUM")
_POINTER_KINDS = _kinds("POINTER", "BLOCKPOINTER")
_RECORD_KINDS = _kinds("RECORD", "COMPLEX")
_QUALIFIER_RE = re.compile(r"\b(?:const|volatile|restrict)\b\s*")


def _classify(rtype: ci.Type) -> ReturnInfo:
    canon = rtype.get_canonical()
    kind = canon.kind
    if kind == ci.TypeKind.VOID:
        return ReturnInfo("void", "void")
    if kind in _POINTER_KINDS:
        return ReturnInfo("pointer", rtype.spelling)
    if kind in _SCALAR_KINDS:
        return ReturnInfo("scalar", rtype.spelling)
    if kind in _RECORD_KINDS:
        spelling = " ".join(_QUALIFIER_RE.sub("", rtype.spelling).split())
        if not spelling or re.search(r"anonymous|unnamed", spelling):
            return ReturnInfo("unsupported", rtype.spelling)
        return ReturnInfo("record", spelling)
    return ReturnInfo("unsupported", rtype.spelling)


def return_infos(spec: CorpusSpec, src_root: Path, targets: list[FunctionTarget]) -> dict[str, ReturnInfo]:
    """用 libclang 判定每个目标的返回类型，key = target_id。同一文件只解析一次。"""
    by_file: dict[str, list[FunctionTarget]] = defaultdict(list)
    for t in targets:
        by_file[t.file].append(t)
    out: dict[str, ReturnInfo] = {}
    for file, ts in by_file.items():
        tu = extract.parse_file(spec, src_root, file)
        defs: dict[str, list[ci.Cursor]] = defaultdict(list)
        for cur in tu.cursor.get_children():
            if cur.kind == ci.CursorKind.FUNCTION_DECL and cur.is_definition():
                defs[cur.spelling].append(cur)
        for t in ts:
            cands = defs.get(t.name, [])
            pick = None
            for cur in cands:
                ext = cur.extent
                if ext.start.offset <= t.body_start < ext.end.offset:
                    pick = cur
                    break
            if pick is None and len(cands) == 1:
                pick = cands[0]
            out[t.target_id] = _classify(pick.result_type) if pick is not None else ReturnInfo("unsupported", "")
    return out


def stub_body(info: ReturnInfo) -> str | None:
    """桩的函数体（含花括号）；返回类型不支持时 None。"""
    if info.kind == "void":
        return "{ }"
    if info.kind in ("scalar", "pointer"):
        return "{ return 0; }"
    if info.kind == "record":
        return "{ static %s _bench_zero; return _bench_zero; }" % info.spelling
    return None


def build_stub(src_root: Path, target: FunctionTarget, info: ReturnInfo) -> str | None:
    """整个定义 [start, end) 的替换文本：原函数头（逐字节照抄）+ 桩函数体；不支持时 None。"""
    body = stub_body(info)
    if body is None:
        return None
    data = (Path(src_root) / target.file).read_bytes()
    header = data[target.start:target.body_start].decode("utf-8", errors="replace")
    return header + body


# ----------------------------------------------------------------------
# stub-kill
# ----------------------------------------------------------------------
def stub_verdict(spec: CorpusSpec, src_root: Path, target: FunctionTarget, sandbox: Any, *,
                 info: ReturnInfo | None = None, work_root: Path | None = None) -> StubVerdict:
    """把目标函数换成零值桩，沙箱里构建并跑项目测试，给出判定（killed / survived / build_error / unsupported）。"""
    src_root = Path(src_root)
    if info is None:
        info = return_infos(spec, src_root, [target])[target.target_id]
    code = build_stub(src_root, target, info)
    if code is None:
        return StubVerdict("unsupported", None, f"返回类型无法构造桩: {info.spelling!r}")
    workdir = new_workdir(work_root, "stub")
    try:
        splice(src_root, target, code, workdir)
        run = run_project(spec, workdir, sandbox)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    if run.stage == "pass":
        return StubVerdict("survived", run.stage, "桩替换后项目测试仍通过")
    if run.stage == "build":
        first = next((d for d in run.diagnostics if d.get("kind") in ("error", "fatal error")), None)
        msg = first["message"] if first else (run.stderr_text.strip().splitlines() or [""])[-1]
        log.warning("%s: 桩没能通过构建: %s", target.target_id, msg[:200])
        return StubVerdict("build_error", run.stage, msg[:300])
    how = "超时" if run.timed_out else (f"信号 {run.signal}" if run.signal else f"rc={run.rc}")
    return StubVerdict("killed", run.stage, f"桩替换后项目测试失败（{how}）",
                       signal=run.signal, timed_out=run.timed_out)


def stub_kill(spec: CorpusSpec, src_root: Path, target: FunctionTarget, sandbox: Any) -> bool:
    """用「返回零值 / 空函数体」桩替换目标函数，项目测试必须失败才算 True。"""
    return stub_verdict(spec, src_root, target, sandbox).killed


# ----------------------------------------------------------------------
# 选择
# ----------------------------------------------------------------------
def _sample_key(seed: int, target_id: str) -> str:
    return hashlib.sha256(f"{seed}|{target_id}".encode("utf-8")).hexdigest()


def _check_baseline(spec: CorpusSpec, src_root: Path, sandbox: Any, work_root: Path | None) -> None:
    workdir = new_workdir(work_root, "base")
    try:
        materialize(src_root, workdir)
        run = run_project(spec, workdir, sandbox)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    if run.stage != "pass":
        tail = (run.stderr_text or run.stdout)[-300:].strip()
        raise TargetsError(f"{spec.name}: 未改动的源码树构建 / 测试就不通过（stage={run.stage}, rc={run.rc}），"
                           f"stub-kill 没有意义: {tail}")


def select(spec: CorpusSpec, src_root: Path, sandbox: Any, flt: TargetFilter | None = None, seed: int = 0, *,
           jobs: int = 1, scan_all: bool = False, work_root: Path | None = None,
           precomputed_counts: dict[str, int] | None = None) -> Selection:
    """选目标并返回完整记录（入选目标、执行次数、桩判定、排除原因计数）。select_targets 是只取目标列表的简写。

    jobs：stub-kill 的并发度；scan_all：对所有候选做 stub-kill（见模块说明），不改变入选结果；
    precomputed_counts：已经算好的函数执行次数（coverage 阶段的产出，key 为 ``file:name``），给了就不再跑覆盖率构建；
    注意它与返回值 Selection.exec_counts 不是一回事：后者的 key 是 target_id，只含入选目标。
    """
    flt = flt or TargetFilter()
    src_root = Path(src_root)
    extracted = extract.extract_functions(spec, src_root)
    excluded: Counter = Counter()
    excluded_ids: dict[str, list[str]] = defaultdict(list)

    def exclude(t: FunctionTarget, reason: str) -> None:
        excluded[reason] += 1
        excluded_ids[reason].append(t.target_id)

    # 1. 静态过滤
    static_ok: list[FunctionTarget] = []
    for t in sorted(extracted, key=lambda x: (x.file, x.start)):
        if spec.is_excluded_target(t.file, t.name):
            exclude(t, "spec_excluded")
        elif t.n_statements < flt.min_statements:
            exclude(t, "few_statements")
        elif t.n_lines < flt.min_lines:
            exclude(t, "too_short")
        elif t.n_lines > flt.max_lines:
            exclude(t, "too_long")
        else:
            static_ok.append(t)

    # 2. 执行次数（有候选才跑覆盖率构建）
    counts: dict[str, int] = {}
    candidates = static_ok
    if flt.require_exec and static_ok:
        counts = dict(precomputed_counts) if precomputed_counts is not None else \
            function_exec_counts(spec, src_root, sandbox, work_root=work_root)
        candidates = []
        for t in static_ok:
            if counts.get(f"{t.file}:{t.name}", 0) <= 0:
                exclude(t, "not_executed")
            else:
                candidates.append(t)

    # 3. 按 seed 抽样 + stub-kill
    ordered = sorted(candidates, key=lambda t: _sample_key(seed, t.target_id))
    limit = flt.max_per_corpus if flt.max_per_corpus and flt.max_per_corpus > 0 else len(ordered)
    selected: list[FunctionTarget] = []
    verdicts: dict[str, StubVerdict] = {}
    consumed = 0
    if not flt.require_stub_kill:
        selected = ordered[:limit]
        consumed = len(selected)
    elif ordered and limit > 0:
        _check_baseline(spec, src_root, sandbox, work_root)
        infos = return_infos(spec, src_root, ordered)
        reason_of = {"survived": "stub_survived", "build_error": "stub_build_error",
                     "unsupported": "stub_unsupported"}
        with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
            while consumed < len(ordered) and (scan_all or len(selected) < limit):
                chunk = ordered[consumed:consumed + max(1, jobs)]
                results = list(pool.map(
                    lambda t: stub_verdict(spec, src_root, t, sandbox, info=infos[t.target_id],
                                           work_root=work_root), chunk))
                for t, v in zip(chunk, results):
                    if not scan_all and len(selected) >= limit:
                        break          # 本批里超出名额的结果丢弃，保证结果与 jobs 无关
                    consumed += 1
                    verdicts[t.target_id] = v
                    if not v.killed:
                        exclude(t, reason_of[v.verdict])
                    elif len(selected) < limit:
                        selected.append(t)
                    else:
                        exclude(t, "not_sampled")      # scan_all：杀得死，但名额已满
    for t in ordered[consumed:]:
        exclude(t, "not_sampled")

    selected.sort(key=lambda t: (t.file, t.start))
    sel = Selection(
        corpus=spec.name, seed=seed, filter=asdict(flt), targets=selected,
        exec_counts={t.target_id: counts.get(f"{t.file}:{t.name}", 0) for t in selected} if flt.require_exec else {},
        stub_verdicts=verdicts, excluded=dict(excluded), excluded_ids=dict(excluded_ids),
        n_extracted=len(extracted), n_candidates=len(candidates))
    log.info("%s: 提取 %d 个函数，候选 %d 个，入选 %d 个；排除原因 %s", spec.name, len(extracted),
             len(candidates), len(selected), dict(excluded))
    return sel


def select_targets(spec: CorpusSpec, src_root: Path, sandbox: Any, flt: TargetFilter, seed: int, *,
                   jobs: int = 1, scan_all: bool = False, stats: dict | None = None,
                   work_root: Path | None = None) -> list[FunctionTarget]:
    """按 TargetFilter 选出目标函数（含 stub-kill），每个语料最多 flt.max_per_corpus 个，seed 决定抽样。

    jobs：stub-kill 的并发度（每个桩一次沙箱运行）；scan_all：对所有候选做 stub-kill，只影响 stats 里的计数；
    stats：传入 dict 时填入 Selection.summary()。需要执行次数、桩判定等完整记录请用 :func:`select`。
    """
    sel = select(spec, src_root, sandbox, flt, seed, jobs=jobs, scan_all=scan_all, work_root=work_root)
    if stats is not None:
        stats.update(sel.summary())
    return sel.targets
