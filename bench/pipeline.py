"""流水线各阶段（设计文档 §2）：prepare → coverage → targets → obfuscate → verify → describe → generate → evaluate → report。

缺陷修复任务（docs/设计-缺陷修复任务.md）在 verify 之后多一个 ``mutate`` 阶段（``stage_mutate``）：在每一级副本的目标函数里注入一个
被项目测试抓住的缺陷；``generate`` / ``evaluate`` 用 ``task="repair"`` 处理修复任务（``plan_repair_tasks`` 出题），
与生成任务共用同一套实现，只是出题、提示词、数据文件名和结果记录里的几个字段不同（见 :class:`TaskKind`）。

每个阶段是一个 ``stage_*`` 函数，吃 :class:`Context`（工作区、沙箱、并发度、种子），返回 :class:`StageResult`
（或它的列表）。cli.py 只负责解析参数和调用这里的函数，e2e 自测和单元测试也直接调用它们。

通用约定：

* **幂等**：每个阶段把「输入哈希」写进自己的产出，下次先重新算输入哈希，产出存在且哈希一致就跳过
  （``StageResult.skipped``）；``Context.force`` 强制重做。哈希里包括 spec、源码树、种子、过滤条件、prompt 模板摘要、
  各阶段自己的版本号（``STAGE_VERSIONS``）、阶段相关源文件的内容哈希（``provenance.code_digest``，改了判定逻辑旧产出自动失效）、
  沙箱镜像 ID（``provenance.sandbox_digest``，镜像是浮动标签）、模型实例的配置指纹（``LLM.fingerprint``，思考开关 / 端点 / 实际模型名）。
* **确定性**：所有随机性由 ``Context.seed`` 决定；LLM 调用 temperature=0（多样本时按样本编号传 seed）并走缓存。
* **并发**：沙箱运行和 LLM 调用放进线程池（``Context.jobs``）。libclang 相关的工作（locate、后处理）在主线程里先做完，
  线程池里只做沙箱和网络，结果按提交顺序落盘，所以产出文件的行序是确定的。
* **只在沙箱里执行语料代码**：所有构建、测试、模型生成的代码都经 ``Context.sandbox``（DockerSandbox）。
* **用的时候再验一遍**：出题（``plan_tasks``）时重新核对 oracle gate 对应的目标集 / 副本、功能规格的输入哈希与泄漏检查、
  stub-kill 记录，对不上的任务作废并计数——不能因为中间某个阶段被跳过或重跑，就悄悄用上过期的上游产出。

数据文件格式见 :class:`bench.store.Workspace` 的各个 ``*_file`` 方法与 README 的「数据文件」一节。
"""
from __future__ import annotations

import logging
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Optional, Sequence, TypeVar

from bench import clangcfg, corpus
from bench.attribute import attribute
from bench.corpus import CorpusSpec
from bench.coverage import function_exec_counts
from bench.describe import DESCRIBE_PROMPT_DIGEST, LeakChecker, describe_detailed
from bench.extract import FunctionTarget, locate
from bench.llm import LLM, Cache, LLMError, LLMHTTPError, MissingAPIKeyError, chat_with_meta, estimate_tokens, make_llm
from bench.metrics import complete_targets, uniform_targets, unreliable_tasks
from bench.mutate import DEFAULT_MAX_TRIES, MutantSearch, analyze, choose_for_levels, reference_for, strip_function_comments
from bench.obfuscate import LEVELS, load_obf_result, obfuscate
from bench.postprocess import Candidate, has_policy_directives, process_reply
from bench.prompt import (
    PROMPT_DIGEST, PROMPT_VERSION, REPAIR_PROMPT_DIGEST, REPAIR_PROMPT_VERSION, RepairTask, Task, build_messages,
    build_repair_messages, prune_context,
)
from bench.provenance import code_digest, sandbox_digest, sandbox_identity
from bench.runner import EvalResult, evaluate, read_function
from bench.similarity import lex, normalized_equal, normalized_similarity
from bench.store import (
    SCHEMA_VERSION, Workspace, append_jsonl, digest, now_iso, read_json, read_jsonl, spec_digest, write_json,
    write_jsonl,
)
from bench.targets import TargetFilter, select
from bench.vocab import KILL_MODES, MUTATION_OPERATORS

log = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_SEED", "STAGE_VERSIONS", "TASKS", "StageError", "Context", "StageResult", "PlannedTask", "TaskKind",
    "TASK_KINDS", "stage_prepare", "stage_coverage", "stage_targets", "stage_obfuscate", "stage_verify", "stage_mutate",
    "stage_check_mutants", "stage_describe", "stage_generate", "stage_evaluate", "load_targets", "load_target_records", "load_descriptions",
    "load_generations", "load_mutants", "plan_tasks", "plan_repair_tasks", "resolve_tasks", "repair_inputs_present",
    "build_llm", "project_names", "run_ordered",
]

DEFAULT_SEED = 20261027

#: 各阶段的版本号：阶段的产出语义变了就加一，旧的结果文件随之失效（代码内容的变化另由 code_digest 覆盖）
STAGE_VERSIONS = {"prepare": 2, "coverage": 2, "targets": 2, "oracle_gate": 2, "describe": 2, "generate": 2,
                  "evaluate": 2, "mutate": 1}

#: 任务类型（``--task`` 的取值是它们加 ``all``）：generate 是功能规格生成任务，repair 是缺陷修复任务
TASKS = ("generate", "repair")

#: 没有 oracle gate 结果的混淆副本不能用来出题（设计文档 §6.1）
GATE_MODEL = "oracle-gate"

#: ``check-mutants`` 回放变异体时走 evaluate 用的模型名（只出现在 EvalResult 里，不进任何数据文件）
REPLAY_MODEL = "mutation-replay"

T = TypeVar("T")
R = TypeVar("R")


class StageError(RuntimeError):
    """阶段无法进行（缺上游产出、副本被改动等），消息里写明该先做什么。"""


@dataclass
class Context:
    ws: Workspace
    sandbox: Any = None
    jobs: int = 4
    seed: int = DEFAULT_SEED
    force: bool = False
    #: 允许跳过可信度措施（targets 不做 stub-kill）并使用未做过 stub-kill 的目标：只给调试用，结果不可作为正式数据
    unsafe_debug: bool = False
    #: 允许使用 mock-* 模型写的功能规格（e2e 自测和测试）：真实工作区里误跑 describe --model mock-oracle 的话，
    #: 固定的 MOCK_DESCRIPTION 会被当成真描述用于后续 generate
    allow_mock: bool = False
    #: evaluate 时有样本缺失（生成调用失败等）不算阶段失败
    allow_missing: bool = False
    #: generate 不读 LLM 缓存（照常写入）：``--force`` 配合它才会真的重新采样
    no_cache: bool = False
    _names: dict = field(default_factory=dict, repr=False, compare=False)

    def need_sandbox(self) -> Any:
        if self.sandbox is None:
            raise StageError("这个阶段需要 Docker 沙箱（ctx.sandbox 为空）")
        return self.sandbox

    def env(self, full: bool = False) -> str:
        """沙箱身份的哈希：``full=False`` 只看工具链（镜像标签与镜像 ID），``True`` 再加资源上限。"""
        return sandbox_digest(self.sandbox, full=full)

    def names(self, spec: CorpusSpec, root: Path) -> set[str]:
        """项目里声明的、符号混淆会改名的全部拼写（libclang 分析较慢，同一个 Context 里每个语料只算一次）。"""
        key = (spec.name, spec_digest(spec), corpus.tree_digest(Path(root))[:16])
        if key not in self._names:
            self._names[key] = project_names(spec, root)
        return self._names[key]

    def checker(self, spec: CorpusSpec, root: Path) -> LeakChecker:
        """当前配置下的泄漏检查器（项目标识符 + spec.allow_words）。"""
        return LeakChecker(self.names(spec, root), spec.allow_words)


@dataclass
class StageResult:
    stage: str
    corpus: str
    ok: bool = True
    skipped: bool = False
    detail: dict = field(default_factory=dict)
    seconds: float = 0.0
    message: str = ""

    def line(self) -> str:
        state = "跳过(输入未变)" if self.skipped else ("完成" if self.ok else "失败")
        extra = f" {self.message}" if self.message else ""
        brief = ", ".join(f"{k}={v}" for k, v in self.detail.items() if not isinstance(v, (dict, list)))
        return f"[{self.stage}] {self.corpus}: {state}{extra}" + (f"（{brief}）" if brief else "") + \
            f" {self.seconds:.1f}s"


# ----------------------------------------------------------------------
# 小工具
# ----------------------------------------------------------------------
def run_ordered(fn: Callable[[T], R], items: Sequence[T], jobs: int, label: str = "",
                every: int = 20) -> Iterator[R]:
    """线程池里并发执行 fn，结果按 items 的顺序依次产出（边完成边交付，可以边跑边落盘）。"""
    total = len(items)
    if total == 0:
        return
    t0 = time.monotonic()

    def note(done: int) -> None:
        if label and (done % every == 0 or done == total):
            log.info("%s: %d/%d（%.0fs）", label, done, total, time.monotonic() - t0)

    if jobs <= 1 or total == 1:
        for i, item in enumerate(items, 1):
            yield fn(item)
            note(i)
        return
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for i, res in enumerate(pool.map(fn, items), 1):
            yield res
            note(i)


def _tree16(root: Path) -> str:
    return corpus.tree_digest(Path(root))[:16]


def _pristine(ctx: Context, spec: CorpusSpec) -> Path:
    root = ctx.ws.pristine(spec.name)
    if not root.is_dir():
        raise StageError(f"{spec.name}: 找不到原始源码 {root}，请先运行 prepare")
    return root


def _levels_arg(levels: Optional[Iterable[str]]) -> list[str]:
    out = list(levels) if levels else list(LEVELS)
    bad = [lv for lv in out if lv not in LEVELS]
    if bad:
        raise StageError(f"未知混淆级别 {bad}，可选 {LEVELS}")
    return [lv for lv in LEVELS if lv in out]


def _timed(stage: str, corpus_name: str, t0: float, **kw: Any) -> StageResult:
    return StageResult(stage=stage, corpus=corpus_name, seconds=round(time.monotonic() - t0, 2), **kw)


# ----------------------------------------------------------------------
# prepare：取源码 + 沙箱里确认构建与测试可用
# ----------------------------------------------------------------------
def stage_prepare(ctx: Context, spec: CorpusSpec, *, local_src: Optional[Path] = None,
                  check_build: bool = True) -> StageResult:
    """把语料源码放到 ``ws.pristine(name)``，并在沙箱里确认 build_cmd 按 pass_criteria 通过。

    有 url 的 spec 走 ``corpus.prepare``（复制本地镜像或克隆）；``local_src`` 给了（本地夹具、没有 git 仓库的语料）
    就从该目录复制。结果记在 ``results/prepare-<corpus>.json``。
    """
    t0 = time.monotonic()
    dest = ctx.ws.pristine(spec.name)
    if local_src is not None:
        local_src = Path(local_src)
        if not dest.is_dir() or corpus.tree_digest(dest) != corpus.tree_digest(local_src):
            corpus.materialize(local_src, dest)
    elif spec.url:
        corpus.prepare(spec, ctx.ws.corpus_src)
    elif not dest.is_dir():
        raise StageError(f"{spec.name}: spec 没有 url，请用 --src 指定源码目录")
    if not check_build:
        return _timed("prepare", spec.name, t0, detail={"tree": _tree16(dest), "build_checked": False})

    inputs = {"spec": spec_digest(spec), "tree": _tree16(dest), "version": STAGE_VERSIONS["prepare"],
              "code": code_digest("prepare"), "env": ctx.env()}
    ihash = digest(inputs)
    path = ctx.ws.prepare_file(spec.name)
    prev = read_json(path)
    if not ctx.force and prev and prev.get("input_hash") == ihash and prev.get("ok"):
        return _timed("prepare", spec.name, t0, skipped=True, detail={"tree": inputs["tree"]})
    ok, run = corpus.verify_build(spec, dest, ctx.need_sandbox(), work_root=ctx.ws.work)
    write_json(path, {
        "schema_version": SCHEMA_VERSION, "corpus": spec.name, "commit": spec.commit, "created": now_iso(),
        "inputs": inputs, "input_hash": ihash, "ok": ok, "rc": run.rc, "timed_out": run.timed_out,
        "sandbox": sandbox_identity(ctx.sandbox),
        "duration_s": round(run.duration_s, 2), "stdout_tail": run.stdout[-1500:], "stderr_tail": run.stderr[-1500:],
    })
    msg = "" if ok else f"构建或测试未通过 (rc={run.rc}, timed_out={run.timed_out})：{(run.stderr or run.stdout)[-200:].strip()}"
    return _timed("prepare", spec.name, t0, ok=ok, message=msg, detail={"tree": inputs["tree"], "rc": run.rc})


# ----------------------------------------------------------------------
# coverage：函数执行次数
# ----------------------------------------------------------------------
def stage_coverage(ctx: Context, spec: CorpusSpec) -> StageResult:
    """沙箱里带 --coverage 构建并跑测试，gcov 得到每个目标函数的执行次数，写 ``results/coverage-<corpus>.json``。

    gcov 有失败的 .gcda、或覆盖率构建下项目测试没通过，都算阶段失败（记在 ``problems`` 里）：失败 .gcda 对应的函数
    不会出现在执行次数里，会被 targets 阶段当成「没被测试执行」排除，基础设施故障就这样悄悄变成目标集变小。
    """
    t0 = time.monotonic()
    root = _pristine(ctx, spec)
    inputs = {"spec": spec_digest(spec), "tree": _tree16(root), "version": STAGE_VERSIONS["coverage"],
              "code": code_digest("coverage"), "env": ctx.env()}
    ihash = digest(inputs)
    path = ctx.ws.coverage_file(spec.name)
    prev = read_json(path)
    if not ctx.force and prev and prev.get("input_hash") == ihash and prev.get("ok", True):
        return _timed("coverage", spec.name, t0, skipped=True, detail={"functions": len(prev.get("counts", {}))})
    stats: dict[str, Any] = {}
    counts = function_exec_counts(spec, root, ctx.need_sandbox(), work_root=ctx.ws.work, stats=stats)
    problems: list[str] = []
    if not stats.get("tests_passed", True):
        problems.append("覆盖率构建下构建 / 测试没通过")
    if stats.get("n_failed"):
        problems.append(f"{stats['n_failed']}/{stats.get('n_gcda')} 个 .gcda 的 gcov 失败: "
                        + "; ".join(stats.get("failed", [])[:3]))
    write_json(path, {
        "schema_version": SCHEMA_VERSION, "corpus": spec.name, "created": now_iso(), "inputs": inputs,
        "input_hash": ihash, "ok": not problems, "problems": problems, "n_gcda": stats.get("n_gcda"),
        "n_gcda_failed": stats.get("n_failed"), "n_functions": len(counts),
        "n_executed": sum(1 for v in counts.values() if v > 0), "counts": counts,
    })
    return _timed("coverage", spec.name, t0, ok=not problems, message="; ".join(problems),
                  detail={"functions": len(counts), "executed": sum(1 for v in counts.values() if v > 0),
                          "gcda": stats.get("n_gcda"), "gcda_failed": stats.get("n_failed")})


# ----------------------------------------------------------------------
# targets：选目标函数
# ----------------------------------------------------------------------
def load_target_records(ws: Workspace, spec_name: str) -> list[dict]:
    """results/targets-<corpus>.jsonl 的原始行（含 exec_count、stub_killed、stub_kill_mode）。"""
    return read_jsonl(ws.targets_file(spec_name))


def load_targets(ws: Workspace, spec_name: str, *, root: Optional[Path] = None) -> list[FunctionTarget]:
    """入选的目标函数（按 (file, 偏移) 排序）；还没跑 targets 阶段则报错。

    给了 ``root``（当前的原始源码树）就核对 targets 产出时的源码树哈希：目标记录里的字节偏移是对着当时的源码树算的，
    源码变了而没重跑 targets，切出来的函数文本会错位。
    """
    recs = load_target_records(ws, spec_name)
    if not recs:
        raise StageError(f"{spec_name}: 没有 {ws.targets_file(spec_name).name}，请先运行 targets")
    if root is not None:
        summary = read_json(ws.targets_summary_file(spec_name)) or {}
        was = (summary.get("inputs") or {}).get("tree")
        if was is None:
            log.warning("%s: 没有 targets 摘要文件，无法核对目标偏移对应的源码树", spec_name)
        elif was != _tree16(root):
            raise StageError(f"{spec_name}: targets 产出时的源码树（{was}）与当前原始源码不一致，"
                             "目标的字节偏移已经失效，请先重跑 targets")
    return [FunctionTarget.from_dict(r) for r in recs]


def stage_targets(ctx: Context, spec: CorpusSpec, flt: Optional[TargetFilter] = None, *, scan_all: bool = True
                  ) -> StageResult:
    """选目标：静态过滤 → 执行次数 > 0 → 按 seed 抽样 + stub-kill。

    产出 ``targets-<corpus>.jsonl``（入选目标，FunctionTarget 全字段 + exec_count + stub_killed + stub_kill_mode）
    与 ``targets-<corpus>.summary.json``（提取数、候选数、各排除原因的计数与 target_id，stub-kill 的完整数字）。
    ``scan_all`` 默认开：对所有候选都做 stub-kill，才能报出整个语料里「测试覆盖不住」的函数个数
    （入选的目标与关掉时完全相同，只是多花几分钟）。覆盖率数据优先用 coverage 阶段的产出，没有就先跑 coverage。

    stub-kill 是设计文档 §6.2 规定不可裁掉的可信度措施：``flt.require_stub_kill`` 为假（``--no-stub-kill``）需要
    ``ctx.unsafe_debug``，产出里的 ``stub_killed`` 是 None，出题时这些目标会被当成「没做 stub-kill」作废
    （同样要 ``unsafe_debug`` 才放行），报告开头也会写明。
    """
    t0 = time.monotonic()
    flt = flt or TargetFilter()
    if not flt.require_stub_kill and not ctx.unsafe_debug:
        raise StageError("跳过 stub-kill 会让「测试覆盖不住的函数」混进目标集，通过率失去意义；"
                         "确实只是调试，请加 --unsafe-debug（产出不可用于正式报告）")
    root = _pristine(ctx, spec)
    counts: Optional[dict[str, int]] = None
    cov_hash: Optional[str] = None
    if flt.require_exec:
        cov_res = stage_coverage(ctx, spec)
        if not cov_res.ok:
            raise StageError(f"{spec.name}: 覆盖率阶段未通过，目标集不可信: {cov_res.message}")
        cov = read_json(ctx.ws.coverage_file(spec.name)) or {}
        counts, cov_hash = cov.get("counts"), cov.get("input_hash")
        if counts is None:
            raise StageError(f"{spec.name}: 覆盖率阶段没有产出 ({cov_res.message})")
    inputs = {"spec": spec_digest(spec), "tree": _tree16(root), "seed": ctx.seed, "filter": asdict(flt),
              "scan_all": bool(scan_all), "coverage": cov_hash, "version": STAGE_VERSIONS["targets"],
              "code": code_digest("targets"), "env": ctx.env()}
    ihash = digest(inputs)
    summary_path, jsonl_path = ctx.ws.targets_summary_file(spec.name), ctx.ws.targets_file(spec.name)
    prev = read_json(summary_path)
    if not ctx.force and prev and prev.get("input_hash") == ihash and jsonl_path.is_file():
        return _timed("targets", spec.name, t0, skipped=True, detail={"selected": prev.get("selected")})
    sel = select(spec, root, ctx.need_sandbox(), flt, ctx.seed, jobs=ctx.jobs, scan_all=scan_all,
                 work_root=ctx.ws.work, precomputed_counts=counts)
    records = sel.records()
    write_jsonl(jsonl_path, ({**r, "input_hash": ihash} for r in records))
    modes = Counter(r.get("stub_kill_mode") for r in records)
    summary = sel.summary()
    summary.update({
        "schema_version": SCHEMA_VERSION, "created": now_iso(), "inputs": inputs, "input_hash": ihash,
        "scan_all": bool(scan_all), "excluded_ids": sel.excluded_ids,
        "stub_kill_modes": dict(modes),
        "stub_checked": len(sel.stub_verdicts),
        "stub_verdicts": dict(Counter(v.verdict for v in sel.stub_verdicts.values())),
        "unverified": not (flt.require_stub_kill and flt.require_exec),
        "sandbox": sandbox_identity(ctx.sandbox),
    })
    write_json(summary_path, summary)
    return _timed("targets", spec.name, t0,
                  detail={"extracted": sel.n_extracted, "candidates": sel.n_candidates, "selected": len(sel.targets)})


# ----------------------------------------------------------------------
# obfuscate：生成 L0–L3 副本并验证
# ----------------------------------------------------------------------
def stage_obfuscate(ctx: Context, spec: CorpusSpec, levels: Optional[Iterable[str]] = None, *,
                    strict_stringify: Optional[bool] = None) -> list[StageResult]:
    """对每个级别生成混淆副本到 ``ws.obf_root(corpus, level)`` 并在沙箱里 verify，结果写 ``obf-<corpus>-<level>.json``。

    某一级出错（改名后重新解析有 error 等）只让这一级失败，不影响其它级别；失败原因放在 ``StageResult.message``。
    """
    root = _pristine(ctx, spec)
    targets = load_targets(ctx.ws, spec.name, root=root)
    out: list[StageResult] = []
    for level in _levels_arg(levels):
        t0 = time.monotonic()
        try:
            res = obfuscate(spec, root, level, ctx.seed, ctx.ws.obf_root(spec.name, level),
                            sandbox=ctx.need_sandbox(), targets=targets, strict_stringify=strict_stringify,
                            results_dir=ctx.ws.results, force=ctx.force, work_root=ctx.ws.work)
        except Exception as exc:                      # 单级失败不拖垮其它级别；原因留在消息里
            log.exception("%s %s 混淆失败", spec.name, level)
            out.append(_timed("obfuscate", f"{spec.name}@{level}", t0, ok=False, message=f"{type(exc).__name__}: {exc}"))
            continue
        detail = {"renamed": len(res.rename_map), "tree": res.tree_hash,
                  "strict": res.strict_stringify, "verify": "未跑" if res.verify is None else
                  ("通过" if res.ok else "未通过")}
        out.append(_timed("obfuscate", f"{spec.name}@{level}", t0, ok=res.ok, skipped=res.skipped,
                          message="; ".join(res.problems), detail=detail))
    return out


# ----------------------------------------------------------------------
# verify：oracle gate（逐目标，设计文档 §6.1）
# ----------------------------------------------------------------------
def _obf_or_raise(ctx: Context, spec: CorpusSpec, level: str) -> dict:
    obf = load_obf_result(ctx.ws.obf_file(spec.name, level))
    if obf is None:
        raise StageError(f"{spec.name}@{level}: 没有 {ctx.ws.obf_file(spec.name, level).name}，请先运行 obfuscate")
    root = ctx.ws.obf_root(spec.name, level)
    if not root.is_dir() or _tree16(root) != obf.get("tree_hash"):
        raise StageError(f"{spec.name}@{level}: 混淆副本 {root} 缺失或与记录的哈希不符，请用 obfuscate --force 重做")
    return obf


def _target_ids_hash(targets: Sequence[FunctionTarget]) -> str:
    return digest(sorted(t.target_id for t in targets))


def stage_verify(ctx: Context, spec: CorpusSpec, levels: Optional[Iterable[str]] = None) -> list[StageResult]:
    """oracle gate：用混淆后的原函数文本走一遍完整的 evaluate（替换 → 语法检查 → 构建 → 项目测试），必须 pass。

    对每个级别的每个目标各做一次；不过的目标记入 ``oracle_gate.failed``，该级别的这个目标作废（出题时跳过并计数）。
    副本本身的 verify 没过的级别整体作废，不再逐目标跑。结果写回 ``obf-<corpus>-<level>.json`` 的 ``oracle_gate``，
    其中 ``checked_ids`` / ``passed_ids`` 记下检查过哪些目标、``inputs`` 记下对应的副本与目标集——出题时据此核对，
    目标集变了而没重跑 verify 会直接报错。
    """
    root = _pristine(ctx, spec)
    targets = load_targets(ctx.ws, spec.name, root=root)
    ids_hash = _target_ids_hash(targets)
    out: list[StageResult] = []
    for level in _levels_arg(levels):
        t0 = time.monotonic()
        obf = _obf_or_raise(ctx, spec, level)
        inputs = {"obf": obf.get("input_hash"), "tree": obf.get("tree_hash"), "targets": ids_hash,
                  "version": STAGE_VERSIONS["oracle_gate"], "code": code_digest("oracle_gate"),
                  "env": ctx.env(full=True)}
        ihash = digest(inputs)
        prev = obf.get("oracle_gate")
        if not ctx.force and prev and prev.get("input_hash") == ihash:
            out.append(_timed("verify", f"{spec.name}@{level}", t0, skipped=True, ok=not prev.get("copy_invalid"),
                              detail={"checked": prev.get("checked"), "passed": prev.get("passed")}))
            continue
        gate = _run_gate(ctx, spec, level, obf, targets)
        gate.update({"inputs": inputs, "input_hash": ihash, "created": now_iso(),
                     "sandbox": sandbox_identity(ctx.sandbox)})
        obf["oracle_gate"] = gate
        write_json(ctx.ws.obf_file(spec.name, level), obf)
        # 个别目标没过 gate 不算阶段失败：它们被作废并计数（出题时跳过）；副本本身不可用才算失败
        msg = ""
        if gate["copy_invalid"]:
            msg = "副本本身的 verify 未通过，整级作废"
        elif gate["failed"]:
            msg = f"{len(gate['failed'])} 个目标未通过 oracle gate，已作废"
        out.append(_timed("verify", f"{spec.name}@{level}", t0, ok=not gate["copy_invalid"], message=msg,
                          detail={"checked": gate["checked"], "passed": gate["passed"]}))
    return out


def _run_gate(ctx: Context, spec: CorpusSpec, level: str, obf: dict, targets: list[FunctionTarget]) -> dict:
    gate: dict[str, Any] = {"checked": 0, "passed": 0, "failed": [], "copy_invalid": False,
                            "checked_ids": sorted(t.target_id for t in targets), "passed_ids": []}
    if not obf.get("ok"):
        gate["copy_invalid"] = True
        return gate
    root = ctx.ws.obf_root(spec.name, level)
    mapping = obf.get("rename_map") or {}
    work: list[tuple[FunctionTarget, FunctionTarget, str]] = []
    failed: list[dict] = []
    for t in targets:                                 # libclang 相关的定位在主线程里先做完
        located = locate(spec, root, mapping.get(t.name, t.name), t.file)
        if located is None:
            failed.append({"target_id": t.target_id, "category": "not_relocated",
                           "detail": f"在 {level} 副本里找不到 {mapping.get(t.name, t.name)}", "stage": None})
            continue
        work.append((t, located, read_function(root, located)))

    def one(job: tuple[FunctionTarget, FunctionTarget, str]) -> dict:
        t, located, text = job
        ev = evaluate(spec, root, located, text, ctx.need_sandbox(), GATE_MODEL, f"{t.target_id}@{level}", 0,
                      work_root=ctx.ws.work)
        category, detail = attribute(ev, Candidate(text, []))
        return {"target_id": t.target_id, "category": category, "detail": detail, "stage": ev.stage}

    for res in run_ordered(one, work, ctx.jobs, f"oracle gate {spec.name}@{level}"):
        if res["category"] != "pass":
            failed.append(res)
    failed_ids = {f["target_id"] for f in failed}
    gate["checked"] = len(targets)
    gate["passed"] = len(targets) - len(failed)
    gate["failed"] = failed
    gate["passed_ids"] = sorted(t.target_id for t in targets if t.target_id not in failed_ids)
    return gate


# ----------------------------------------------------------------------
# mutate：缺陷修复任务的变异体（docs/设计-缺陷修复任务.md §3、§4、§7）
# ----------------------------------------------------------------------
@dataclass
class _LevelOutcome:
    """一个（目标, 级别）在 mutate 阶段的结果。``search`` 为 None 表示没有进入搜索（副本无效、gate 没过、找不到目标）。"""

    outcome: str                    # killed / no_sites / exhausted / gave_up / strip_unsafe / copy_invalid / oracle_failed / not_relocated
    search: Optional[MutantSearch] = None


def load_mutants(ws: Workspace, corpus_name: str, level: str) -> tuple[Optional[dict], dict[str, dict]]:
    """某语料某一级的变异体：返回（汇总里该级的条目, {target_id: 记录}）。汇总文件或该级条目缺失时条目为 None。"""
    summary = read_json(ws.mutants_summary_file(corpus_name))
    levels = summary.get("levels") if isinstance(summary, dict) else None
    entry = levels.get(level) if isinstance(levels, dict) else None
    recs = {r["target_id"]: r for r in read_jsonl(ws.mutants_file(corpus_name, level)) if r.get("target_id")}
    return (entry if isinstance(entry, dict) else None), recs


def _mutate_inputs(ctx: Context, spec: CorpusSpec, root: Path, ids_hash: str, max_tries: int, recheck: bool) -> dict:
    """mutate 各级共有的输入：spec、原始树、目标集、种子、尝试上限、是否复核、阶段版本与代码摘要、沙箱（含资源上限：超时会决定 kill 结果）、
    宿主 libclang 环境（位点识别依赖它）。出题（``plan_repair_tasks``）使用变异体时按同一个公式重算一遍，对不上就拒绝。"""
    return {"spec": spec_digest(spec), "pristine": _tree16(root), "targets": ids_hash, "seed": ctx.seed,
            "max_tries": max_tries, "recheck": recheck, "version": STAGE_VERSIONS["mutate"], "code": code_digest("mutate"),
            "env": ctx.env(full=True), "clang": clangcfg.environment(), "strip_comments": True}


def _four_level_stats(ws: Workspace, corpus_name: str, target_ids: Sequence[str]) -> dict[str, int]:
    """四级都选出了变异体的目标数，以及其中四级算子相同 / 四级是同一个位点的个数（附录 §4 / §6）。
    判定函数与报告共用（``metrics.uniform_targets``），只是这里读的是 mutants 文件。"""
    ops: dict[str, dict[str, str]] = {}
    sites: dict[str, dict[str, bool]] = {}
    for lv in LEVELS:
        for r in read_jsonl(ws.mutants_file(corpus_name, lv)):
            tid = str(r.get("target_id"))
            ops.setdefault(tid, {})[lv] = str(r.get("operator"))
            sites.setdefault(tid, {})[lv] = r.get("same_site_as_L0") is True
    wanted = set(target_ids)
    ops = {k: v for k, v in ops.items() if k in wanted}
    sites = {k: v for k, v in sites.items() if k in wanted}
    full = complete_targets(ops)
    return {"n_targets": len(wanted), "n_all_levels": len(full), "n_same_operator": len(uniform_targets(ops)),
            "n_same_site": sum(1 for per in complete_targets(sites).values() if all(per.values()))}


def _mutate_entry(ihash: str, inputs: dict, n_targets: int, outcomes: Sequence[tuple[FunctionTarget, _LevelOutcome]],
                  records: list[dict]) -> dict:
    """汇总文件里某一级的条目：出题数、放弃数与原因、算子 / 杀死方式分布、same_operator_as_L0 / same_site_as_L0 比例、
    记账（试了几个位点、复核了几个……）、各类变异体占的沙箱时间。"""
    counts: Counter = Counter(lo.outcome for _, lo in outcomes)
    searched = [lo.search for _, lo in outcomes if lo.search is not None]
    ops = Counter(r["operator"] for r in records)
    modes = Counter(r["kill_mode"] for r in records)
    flagged = [r["same_operator_as_L0"] for r in records if r.get("same_operator_as_L0") is not None]
    sited = [r["same_site_as_L0"] for r in records if r.get("same_site_as_L0") is not None]
    abandoned = []
    for t, lo in outcomes:
        if lo.search is not None and lo.outcome != "killed":
            sr = lo.search
            abandoned.append({"target_id": t.target_id, "outcome": lo.outcome, "n_sites": sr.n_sites,
                              "tried_sites": sr.tried_sites, "equivalent_count": sr.equivalent_count,
                              "compile_fail_count": sr.compile_fail_count, "parse_fail_count": sr.parse_fail_count,
                              "unstable_count": sr.unstable_count})
    seconds: Counter = Counter()
    for s in searched:
        seconds.update(s.sandbox_seconds)
    return {
        "input_hash": ihash, "inputs": inputs, "n_targets": n_targets, "n_searched": len(searched),
        "n_killed": len(records), "n_abandoned": len(abandoned), "outcomes": dict(counts),
        "operators": {op: ops.get(op, 0) for op in MUTATION_OPERATORS}, "kill_modes": dict(modes),
        "same_operator_as_L0": {"n": len(flagged), "same": sum(1 for f in flagged if f)},
        "same_site_as_L0": {"n": len(sited), "same": sum(1 for f in sited if f)},
        "anchor": dict(Counter(s.anchor for s in searched)),
        "n_sites": sum(s.n_sites for s in searched),
        "tried_sites": sum(s.tried_sites for s in searched),
        "equivalent_count": sum(s.equivalent_count for s in searched),
        "compile_fail_count": sum(s.compile_fail_count for s in searched),
        "parse_fail_count": sum(s.parse_fail_count for s in searched),
        "unstable_count": sum(s.unstable_count for s in searched),
        "n_rechecked": sum(s.rechecked for s in searched),
        "n_kill_label_changed": sum(1 for r in records if r.get("kill_recheck") not in (None, r.get("kill_mode"))),
        "n_with_orphans": sum(1 for r in records if r.get("context_orphans")),
        "n_with_comments": sum(1 for s in searched if s.has_comments),
        "n_strip_checked": sum(1 for s in searched if s.strip_checked),
        "n_strip_failed": counts.get("strip_unsafe", 0),
        "sandbox_seconds": {k: round(v, 1) for k, v in sorted(seconds.items())},
        "abandoned": abandoned,
    }


def stage_mutate(ctx: Context, spec: CorpusSpec, levels: Optional[Iterable[str]] = None, *,
                 max_tries: int = DEFAULT_MAX_TRIES, recheck: bool = True) -> StageResult:
    """缺陷修复任务的出题：在每一级混淆副本的每个入选目标函数里注入一个能被项目测试捕获的缺陷（docs/设计-缺陷修复任务.md §3、§4）。

    只处理 oracle gate 通过的（目标, 级别）：先 verify，再 mutate。每个（目标, 级别）按 :func:`bench.mutate.choose_for_levels`
    选变异体：L0 先选，其余级别先试 L0 选定的位点在本级的对应位点（``same_site_as_L0``）、再优先复用 L0 选定的算子
    （``same_operator_as_L0``）；变异体当作候选走一遍完整的 evaluate，编译通过且在测试阶段失败才算杀死，``recheck`` 为真（默认）时
    杀死的候选再评测一次，第二次仍失败才出题；最多试 ``max_tries`` 个位点，试满仍没有杀死就放弃（不出题，计数并记原因）。
    变异体的函数文本先去注释再做 kill 检查（符号混淆只改标识符，注释里仍是原名，留着会泄漏被隐藏的名字）；
    函数里有注释的，先把去注释后的**原函数**放回副本过一遍项目测试，不通过（``strip_unsafe``）就放弃这个（目标, 级别），
    不搜索，也不拿它当 L1–L3 的参照。

    产出：``results/mutants-<corpus>-<level>.jsonl``（每个出了题的（目标, 级别）一行，字段见附录 §7）和
    ``results/mutants-<corpus>.summary.json``（各级的输入哈希、出题数、放弃数及原因、算子与杀死方式分布、same_operator_as_L0 /
    same_site_as_L0 比例、记账、各类变异体占的沙箱时间、耗时）。幂等：每级的输入哈希（spec、原始树、目标集、副本、oracle gate、种子、
    ``max_tries``、``recheck``、mutate 相关代码、沙箱、宿主 libclang；L1–L3 另含 L0 的输入哈希）没变且文件完好就跳过。
    只重做 ``levels`` 里过期的级别；不含 L0 时要求 L0 的产出有效（其它级别拿它选定的位点当参照）。
    """
    t0 = time.monotonic()
    if max_tries < 1:
        raise StageError("max_tries 必须 >= 1")
    root = _pristine(ctx, spec)
    targets = load_targets(ctx.ws, spec.name, root=root)
    ids_hash = _target_ids_hash(targets)
    wanted = _levels_arg(levels)
    needed = [lv for lv in LEVELS if lv in wanted or (lv == "L0" and wanted != ["L0"])]
    obfs: dict[str, dict] = {}
    gates: dict[str, Optional[dict]] = {}
    for lv in needed:
        obfs[lv] = _obf_or_raise(ctx, spec, lv)
        gates[lv] = _gate_or_raise(spec, lv, obfs[lv], ids_hash) if obfs[lv].get("ok") else None
    common = _mutate_inputs(ctx, spec, root, ids_hash, max_tries, recheck)
    expected: dict[str, tuple[dict, str]] = {}
    for lv in needed:
        inputs = {**common, "level": lv, "obf": obfs[lv].get("input_hash"), "tree": obfs[lv].get("tree_hash"),
                  "gate": str((gates[lv] or {}).get("input_hash") or ""),
                  "ref": expected["L0"][1] if lv != "L0" else None}
        expected[lv] = (inputs, digest(inputs))

    summary = read_json(ctx.ws.mutants_summary_file(spec.name))
    prev_levels: dict[str, dict] = dict((summary or {}).get("levels") or {}) if isinstance(summary, dict) else {}

    def is_fresh(lv: str) -> bool:
        ent, path = prev_levels.get(lv), ctx.ws.mutants_file(spec.name, lv)
        return bool(ent) and ent.get("input_hash") == expected[lv][1] and path.is_file() \
            and len(read_jsonl(path)) == ent.get("n_killed")

    todo = [lv for lv in wanted if ctx.force or not is_fresh(lv)]
    if not todo:
        killed = sum(prev_levels[lv]["n_killed"] for lv in wanted)
        gone = sum(prev_levels[lv]["n_abandoned"] for lv in wanted)
        return _timed("mutate", spec.name, t0, skipped=True, detail={"killed": killed, "abandoned": gone})
    if "L0" not in todo and not is_fresh("L0"):
        raise StageError(f"{spec.name}: L1–L3 拿 L0 选定的位点当参照，但 L0 的 mutants 缺失或已过期，请先对 L0 运行 mutate")
    ref_recs: dict[str, dict] = {}
    if "L0" not in todo and "L0" in needed:
        ref_recs = {str(r["target_id"]): r for r in read_jsonl(ctx.ws.mutants_file(spec.name, "L0"))}

    # libclang 相关的定位在主线程里先做完（locate 不在 mutate 的那把锁里），线程池里只做位点枚举与沙箱运行
    passed = {lv: set((gates[lv] or {}).get("passed_ids") or []) for lv in todo}
    items: list[tuple] = []
    for t in targets:
        trees: dict[str, tuple[Path, FunctionTarget]] = {}
        pre: dict[str, str] = {}
        for lv in todo:
            if not obfs[lv].get("ok"):
                pre[lv] = "copy_invalid"
            elif t.target_id not in passed[lv]:
                pre[lv] = "oracle_failed"
            else:
                oroot = ctx.ws.obf_root(spec.name, lv)
                located = locate(spec, oroot, (obfs[lv].get("rename_map") or {}).get(t.name, t.name), t.file)
                if located is None:
                    pre[lv] = "not_relocated"
                else:
                    trees[lv] = (oroot, located)
        ref_src = None
        if t.target_id in ref_recs and trees:       # 这次不重做 L0：参照要从 L0 的副本和它选定的位点现算
            root0 = ctx.ws.obf_root(spec.name, "L0")
            located0 = locate(spec, root0, (obfs["L0"].get("rename_map") or {}).get(t.name, t.name), t.file)
            if located0 is None:
                raise StageError(f"{spec.name}: L0 副本里找不到 {t.target_id}，L0 的 mutants 已过期，请先对 L0 运行 mutate")
            ref_src = (root0, located0, ref_recs[t.target_id])
        items.append((t, trees, pre, ref_src))

    sandbox = ctx.need_sandbox()

    def one(item: tuple) -> dict[str, _LevelOutcome]:
        t, trees, pre, ref_src = item
        out = {lv: _LevelOutcome(why) for lv, why in pre.items()}
        if not trees:
            return out
        reference = None
        if ref_src is not None:
            root0, located0, rec0 = ref_src
            reference = reference_for(analyze(spec, root0, located0), str(rec0["operator"]), int(rec0["site_index"]))
            if reference is None:
                raise StageError(f"{t.target_id}: L0 的 mutants 记录的位点 {rec0['operator']}#{rec0['site_index']} 在 L0 副本里已不存在，"
                                 "请先对 L0 运行 mutate")
        found = choose_for_levels(spec, trees, ctx.seed, sandbox, target_id=t.target_id, levels=list(trees),
                                  max_tries=max_tries, reference=reference, work_root=ctx.ws.work, strip_comments=True,
                                  recheck=recheck)
        out.update({lv: _LevelOutcome(res.outcome, res) for lv, res in found.items()})
        return out

    results = list(run_ordered(one, items, ctx.jobs, f"mutate {spec.name}", every=5))

    entries = dict(prev_levels)
    for lv in todo:
        inputs, ihash = expected[lv]
        pairs = [(item[0], per[lv]) for item, per in zip(items, results)]
        recs: list[dict] = []
        for t, lo in pairs:
            if lo.outcome != "killed" or lo.search is None or lo.search.mutant is None:
                continue
            recs.append({**lo.search.mutant.to_dict(), "schema_version": SCHEMA_VERSION, "corpus": spec.name,
                         "outcome": "killed", "n_sites": lo.search.n_sites, "prefer_operator": lo.search.prefer_operator,
                         "anchor": lo.search.anchor, "has_comments": lo.search.has_comments,
                         "strip_checked": lo.search.strip_checked, "input_hash": ihash})
        write_jsonl(ctx.ws.mutants_file(spec.name, lv), recs)
        entries[lv] = _mutate_entry(ihash, inputs, len(targets), pairs, recs)
    write_json(ctx.ws.mutants_summary_file(spec.name), {
        "schema_version": SCHEMA_VERSION, "corpus": spec.name, "created": now_iso(), "seed": ctx.seed,
        "max_tries": max_tries, "recheck": recheck, "elapsed_s": round(time.monotonic() - t0, 2), "levels_computed": todo,
        "n_targets": len(targets), "levels": {lv: entries[lv] for lv in LEVELS if lv in entries},
        "four_level": _four_level_stats(ctx.ws, spec.name, [t.target_id for t in targets]),
        "sandbox": sandbox_identity(ctx.sandbox)})

    killed = sum(entries[lv]["n_killed"] for lv in todo)
    gone = sum(entries[lv]["n_abandoned"] for lv in todo)
    voided = sum(entries[lv]["n_targets"] - entries[lv]["n_searched"] for lv in todo)
    unstable = sum(entries[lv]["unstable_count"] for lv in todo)
    why: Counter = Counter()
    for lv in todo:
        why.update({k: v for k, v in entries[lv]["outcomes"].items() if k != "killed"})
    msg = (f"{gone + voided} 个（目标, 级别）不出题（" + ", ".join(f"{k}={v}" for k, v in sorted(why.items())) + "）") \
        if why else ""
    return _timed("mutate", spec.name, t0, message=msg,
                  detail={"targets": len(targets), "levels": "/".join(todo), "killed": killed, "abandoned": gone,
                          "voided": voided, "unstable": unstable})


# ----------------------------------------------------------------------
# describe：功能规格（一次生成、四级共用）
# ----------------------------------------------------------------------
def project_names(spec: CorpusSpec, root: Path) -> set[str]:
    """项目里声明的、符号混淆会改名的全部拼写：描述里不允许出现它们（含 L0 的真名，设计文档 §6.4）。

    返回 ``IdentifierSet``（``set`` 的子类，带种类等信息）。
    """
    from bench.obfuscate.symbols import collect_project_identifiers   # 延迟导入：拉起 libclang
    return collect_project_identifiers(spec, root)


def load_descriptions(ws: Workspace) -> dict[str, dict]:
    """results/descriptions.jsonl 里每个 target_id 的最后一条记录（追加写，后写的覆盖先写的）。"""
    latest: dict[str, dict] = {}
    for rec in read_jsonl(ws.descriptions_file):
        if rec.get("target_id"):
            latest[rec["target_id"]] = rec
    return latest


def _describe_inputs(t: FunctionTarget, code: str, tree: str, model: str, checker: str) -> dict:
    """决定一份功能规格的全部输入：目标函数原文、源码树、描述模型、描述提示词、泄漏检查器的配置（词表、阈值、allow_words、
    项目标识符集合）、描述阶段的代码。检查器收紧后旧描述的输入哈希对不上，出题时会被重新判定。"""
    return {"target": t.target_id, "code": digest(code), "tree": tree, "model": model,
            "prompt": DESCRIBE_PROMPT_DIGEST[:16], "checker": checker, "src": code_digest("describe"),
            "version": STAGE_VERSIONS["describe"]}


def stage_describe(ctx: Context, spec: CorpusSpec, llm: LLM, *, model_name: Optional[str] = None) -> StageResult:
    """为每个入选目标生成一份「不含任何项目标识符」的英文功能规格，追加写 ``descriptions.jsonl``。

    输入（目标函数原文、源码树、描述模型、描述 prompt 摘要、泄漏检查器配置）没变的目标跳过；泄漏检查重试用尽仍泄漏的目标
    记 ``failed=true``（出题时排除）。LLM 自身的错误（如缺 API key）向上抛——但只有真有目标要生成时才会发请求，
    所有描述都已是最新时不需要 API key。mock-* 模型只在 ``ctx.allow_mock`` 时允许（真实工作区里误写入固定的 mock 描述，
    会被当成真描述用于后续 generate）。
    """
    t0 = time.monotonic()
    root = _pristine(ctx, spec)
    targets = load_targets(ctx.ws, spec.name, root=root)
    model = model_name or getattr(llm, "model", "") or "unknown"
    if model.startswith("mock-") and not ctx.allow_mock:
        raise StageError(f"{model} 是离线自测用的假模型，会把固定的占位描述写进 descriptions.jsonl；"
                         "只有 e2e 自测用得着它（真要在这个工作区里用，请加 --unsafe-debug）")
    tree = _tree16(root)
    checker = ctx.checker(spec, root)
    cdig = checker.digest()
    existing = load_descriptions(ctx.ws)
    todo: list[tuple[FunctionTarget, str, str]] = []
    for t in targets:
        code = read_function(root, t)
        ihash = digest(_describe_inputs(t, code, tree, model, cdig))
        rec = existing.get(t.target_id)
        if not ctx.force and rec and rec.get("input_hash") == ihash:
            continue
        todo.append((t, code, ihash))
    if not todo:
        failed = sum(1 for t in targets if existing.get(t.target_id, {}).get("failed"))
        return _timed("describe", spec.name, t0, skipped=True, detail={"targets": len(targets), "failed": failed})

    names = ctx.names(spec, root)

    def one(item: tuple[FunctionTarget, str, str]):
        t, code, _ = item
        return describe_detailed(t, code, llm, names, allow_words=spec.allow_words)

    n_failed = n_leak_retries = p_tok = c_tok = n_phrase = n_own = 0
    for (t, code, ihash), res in zip(todo, run_ordered(one, todo, ctx.jobs, f"describe {spec.name}", every=10)):
        n_failed += int(res.failed)
        n_leak_retries += res.leak_retries
        n_phrase += int(bool(res.phrase_hits))
        n_own += int(res.own_name_hit)
        p_tok += res.prompt_tokens
        c_tok += res.completion_tokens
        append_jsonl(ctx.ws.descriptions_file, {
            "schema_version": SCHEMA_VERSION, "target_id": t.target_id, "corpus": spec.name,
            "description": "" if res.failed else res.description,
            "rejected_text": res.description if res.failed else None,
            "model": model, "leak_retries": res.leak_retries, "attempts": res.attempts,
            "failed": res.failed, "reason": res.reason, "leaked": res.leaked, "n_words": res.n_words,
            "length_ok": res.length_ok, "prompt_tokens": res.prompt_tokens,
            "completion_tokens": res.completion_tokens, "describe_prompt": DESCRIBE_PROMPT_DIGEST[:16],
            "phrase_hits": res.phrase_hits, "exempt_hits": res.exempt_hits, "own_name_hit": res.own_name_hit,
            "checker": res.checker or cdig,
            "inputs": {"code": digest(code), "tree": tree, "model": model, "checker": cdig},
            "input_hash": ihash, "created": now_iso(),
        })
    return _timed("describe", spec.name, t0, ok=True,
                  message=(f"{n_failed} 个目标 describe_failed，已排除" if n_failed else ""),
                  detail={"described": len(todo), "failed": n_failed, "leak_retries": n_leak_retries,
                          "phrase_warnings": n_phrase, "own_name_hits": n_own,
                          "prompt_tokens": p_tok, "completion_tokens": c_tok})


# ----------------------------------------------------------------------
# 出题：每个（目标, 级别）一个 Task
# ----------------------------------------------------------------------
@dataclass
class PlannedTask:
    task: Task | RepairTask         # 生成任务是 Task，缺陷修复任务是 RepairTask（两者的 task_id / target_id / corpus / level 等同名）
    spec: CorpusSpec
    original: FunctionTarget        # L0 原始树上的目标
    located: FunctionTarget         # 该级副本里重新定位的目标（偏移对应副本，splice 用它）
    oracle: str                     # 该级副本里原函数的文本
    obf_root: Path
    tree_hash: str                  # 副本源码树哈希（obf json 里记录的）
    spec_hash: str
    gate_hash: str = ""             # oracle gate 的输入哈希（出题时核对过，评测结果里记下）
    stub_kill_mode: Optional[str] = None    # 该目标被零值桩杀死的方式（test_fail / runtime_crash / runtime_timeout）
    allow_directives: bool = False          # 原函数自己带 #define / #pragma：候选里的这类指令不算违反约束
    # 以下只有缺陷修复任务用得上（生成任务里都是空值）；字段名与任务 / 结果记录里的同名字段含义一致
    mutation: Optional[dict] = None         # 注入的缺陷：{operator, site_index, original_text, mutated_text}
    kill_mode: Optional[str] = None         # 变异体被杀死的方式（kill 检查里的归因类别）
    same_operator_as_L0: Optional[bool] = None      # 变异体的算子与 L0 选定的是否相同
    same_site_as_L0: Optional[bool] = None          # 变异体是不是 L0 选定位点在本级的对应位点（四级是同一个缺陷）
    mutants_hash: str = ""                  # 变异体所在 mutants 文件的输入哈希（评测结果里记下，report 据此核对）


@dataclass
class _LevelView:
    """一个级别出题时用到的上游信息：混淆记录、oracle gate 的结论、改名表、副本目录。"""

    level: str
    obf: dict
    gate: Optional[dict]
    root: Path

    @property
    def mapping(self) -> dict:
        return self.obf.get("rename_map") or {}

    @property
    def gate_failed(self) -> set:
        return {f["target_id"] for f in (self.gate or {}).get("failed", [])}

    @property
    def gate_passed(self) -> set:
        return set((self.gate or {}).get("passed_ids") or [])


def _level_view(ctx: Context, spec: CorpusSpec, level: str, ids_hash: str, require_gate: bool) -> Optional[_LevelView]:
    """某一级出题要用的上游信息；副本本身 verify 没过（整级作废）返回 None。副本缺失 / 哈希对不上 / gate 过期直接报 StageError。"""
    obf = _obf_or_raise(ctx, spec, level)
    if not obf.get("ok"):
        return None
    return _LevelView(level, obf, _gate_or_raise(spec, level, obf, ids_hash, require_gate), ctx.ws.obf_root(spec.name, level))


def _common_void(ctx: Context, view: _LevelView, trec: dict, t: FunctionTarget, require_gate: bool) -> Optional[str]:
    """两类任务共用的作废判定（在使用上游产出的这一刻重新核对，设计文档 §6.8）→ 作废原因，None 表示没问题。"""
    if not ctx.unsafe_debug and trec.get(t.target_id, {}).get("stub_killed") is not True:
        return "stub_unchecked"
    if t.target_id in view.gate_failed:
        return "oracle_failed"
    if require_gate and t.target_id not in view.gate_passed:
        return "oracle_unchecked"
    if view.level != "L0" and t.name not in view.mapping:
        return "name_not_renamed"
    return None


def _relocate(spec: CorpusSpec, view: _LevelView, t: FunctionTarget) -> Optional[FunctionTarget]:
    """目标在该级副本里按映射后的名字重新定位（偏移对应副本）。"""
    return locate(spec, view.root, view.mapping.get(t.name, t.name), t.file)


def _planned(task: Task | RepairTask, spec: CorpusSpec, t: FunctionTarget, located: FunctionTarget, view: _LevelView,
             trec: dict, spec_hash: str, **repair_fields: Any) -> PlannedTask:
    """两类任务共用的 PlannedTask 构造（``repair_fields`` 是只有修复任务才有的那几个字段）。"""
    oracle = read_function(view.root, located)
    return PlannedTask(
        task=task, spec=spec, original=t, located=located, oracle=oracle, obf_root=view.root,
        tree_hash=view.obf["tree_hash"], spec_hash=spec_hash, gate_hash=str((view.gate or {}).get("input_hash") or ""),
        stub_kill_mode=trec.get(t.target_id, {}).get("stub_kill_mode"), allow_directives=has_policy_directives(oracle),
        **repair_fields)


def plan_tasks(ctx: Context, spec: CorpusSpec, levels: Optional[Iterable[str]] = None, *, require_gate: bool = True
               ) -> tuple[list[PlannedTask], dict[str, dict[str, int]]]:
    """为 ``levels`` 里每个级别、每个入选目标准备 Task（上下文取自该级副本，名字已按映射改写）。

    返回（任务列表, {级别: {作废原因: 个数}}）。作废原因：
    ``copy_invalid``（副本 verify 没过，整级作废）、``no_description`` / ``describe_failed``（没有可用规格）、
    ``describe_mock``（规格是 mock 模型写的）、``describe_stale``（规格的输入哈希与现状不符：目标函数、源码树、描述提示词
    或泄漏检查器配置变了）、``describe_leak``（用当前检查器重新检查，规格里有项目标识符）、
    ``stub_unchecked``（目标没做过 stub-kill，只有 ``ctx.unsafe_debug`` 才放行）、``oracle_failed`` / ``oracle_unchecked``
    （oracle gate 没过 / 没检查这个目标）、``name_not_renamed``（L1+ 里目标函数名没被改掉，提示词会带真名）、
    ``not_relocated``（副本里找不到）。

    副本缺失、哈希对不上、还没跑 oracle gate、或 oracle gate 对应的目标集 / 副本与现状不符（重跑了 targets 或 obfuscate
    而没重跑 verify），都直接报 StageError——使用上游产出的时候要重新核对，不能因为中间某个阶段被跳过就用上过期数据。
    """
    root = _pristine(ctx, spec)
    targets = load_targets(ctx.ws, spec.name, root=root)
    trec = {r.get("target_id"): r for r in load_target_records(ctx.ws, spec.name)}
    descs = load_descriptions(ctx.ws)
    sh = spec_digest(spec)
    ids_hash = _target_ids_hash(targets)
    tree = _tree16(root)
    checker: Optional[LeakChecker] = None
    cdig = ""
    planned: list[PlannedTask] = []
    voids: dict[str, dict[str, int]] = {}
    desc_verdict: dict[str, Optional[str]] = {}

    def check_description(t: FunctionTarget) -> Optional[str]:
        """功能规格能不能用 → 不能用的原因（None 表示可用）。结果按目标缓存，四个级别只算一次。"""
        nonlocal checker, cdig
        if t.target_id in desc_verdict:
            return desc_verdict[t.target_id]
        desc = descs.get(t.target_id)
        why: Optional[str] = None
        if desc is None:
            why = "no_description"
        elif desc.get("failed") or not desc.get("description"):
            why = "describe_failed"
        elif str(desc.get("model", "")).startswith("mock-") and not (ctx.allow_mock or ctx.unsafe_debug):
            why = "describe_mock"
        else:
            if checker is None:
                checker = ctx.checker(spec, root)
                cdig = checker.digest()
            want = digest(_describe_inputs(t, read_function(root, t), tree, str(desc.get("model", "")), cdig))
            if desc.get("input_hash") != want:
                why = "describe_stale"
            elif checker.leaks(desc["description"]):
                why = "describe_leak"
        desc_verdict[t.target_id] = why
        return why

    for level in _levels_arg(levels):
        view = _level_view(ctx, spec, level, ids_hash, require_gate)
        if view is None:
            voids[level] = {"copy_invalid": len(targets)}
            continue
        void: Counter = Counter()
        for t in targets:
            why = check_description(t) or _common_void(ctx, view, trec, t, require_gate)
            if why is not None:
                void[why] += 1
                continue
            located = _relocate(spec, view, t)
            if located is None:
                void["not_relocated"] += 1
                continue
            task = Task(
                task_id=f"{t.target_id}@{level}", target_id=t.target_id, corpus=spec.name, level=level,
                obf_name=located.name, obf_signature=located.signature, context=located.deps,
                description=descs[t.target_id]["description"], includes=list(located.deps.includes),
                language_std=spec.std or "c99")
            planned.append(_planned(task, spec, t, located, view, trec, sh))
        voids[level] = dict(void)
    return planned, voids


_GATE_WHAT = {"obf": "混淆副本记录", "tree": "副本源码树", "targets": "目标集"}


def _gate_or_raise(spec: CorpusSpec, level: str, obf: dict, ids_hash: str, require: bool = True) -> Optional[dict]:
    """某一级的 oracle gate 结果。``require`` 为真时：还没跑过 verify，或 gate 对着的副本 / 目标集与现状不符
    （重跑了 targets 或 obfuscate 而没重跑 verify），直接报错；为假时有什么返回什么（可能是 None）。"""
    gate = obf.get("oracle_gate")
    if gate is None and require:
        raise StageError(f"{spec.name}@{level}: 还没有 oracle gate 结果，请先运行 verify")
    if gate is not None and require:
        gi = gate.get("inputs") or {}
        expect = {"obf": obf.get("input_hash"), "tree": obf.get("tree_hash"), "targets": ids_hash}
        stale = [k for k, v in expect.items() if gi.get(k) != v]
        if stale:
            raise StageError(f"{spec.name}@{level}: oracle gate 是对着另一份{'、'.join(_GATE_WHAT[k] for k in stale)}"
                             "跑的（重跑了 targets 或 obfuscate 而没重跑 verify？），请先运行 verify")
    return gate


_MUTANT_WHAT = {"obf": "混淆副本记录", "tree": "副本源码树", "targets": "目标集", "spec": "语料 spec",
                "gate": "oracle gate 结果", "ref": "L0 的变异体", "code": "mutate 相关代码", "seed": "随机种子（--seed）",
                "version": "mutate 阶段版本", "clang": "宿主 libclang 环境", "env": "沙箱（镜像或资源上限）",
                "strip_comments": "去注释设置"}


def _own_name_in_literals(name: str, code: str) -> bool:
    """函数文本的字符串 / 字符常量里有没有出现 ``name``（整词）。L1 以上目标函数已经改名，字面量里（错误提示里写着函数名之类）
    出现原名，就把本该隐藏的名字原样放进了提示词，字面量改了会改变行为，去不掉。"""
    pat = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])")
    return any(kind in ("string", "char") and pat.search(text) for kind, text in lex(code))


def plan_repair_tasks(ctx: Context, spec: CorpusSpec, levels: Optional[Iterable[str]] = None, *,
                      require_gate: bool = True) -> tuple[list[PlannedTask], dict[str, dict[str, int]]]:
    """为 ``levels`` 里每个级别、每个入选目标准备缺陷修复任务（``RepairTask``，上下文取自该级副本，名字已按映射改写）。

    返回（任务列表, {级别: {作废原因: 个数}}）。变异体取自 mutate 阶段的 ``mutants-<corpus>-<level>.jsonl``，
    不需要功能规格，所以与 describe 阶段无关。作废原因：``copy_invalid``（副本 verify 没过，整级作废）、
    ``stub_unchecked``（目标没做过 stub-kill，只有 ``ctx.unsafe_debug`` 才放行）、``oracle_failed`` / ``oracle_unchecked``、
    ``name_not_renamed``（L1+ 里目标函数名没被改掉，签名会带真名）、``no_killed_mutant``（mutate 没为这个（目标, 级别）选出
    被杀死的变异体：没有位点、全是等价变异体、试满上限，或去注释后的原函数通不过测试）、``name_in_literal``（L1+ 的变异体函数
    文本的字符串 / 字符常量里带着目标函数自己的原名，隐藏失败）、``not_relocated``。

    提示词里的依赖上下文取自注入缺陷后的函数（``prompt.prune_context``）：被删掉 / 换掉的那部分代码才用到的原型、宏、全局不进提示词，
    被裁掉的名字记在 ``RepairTask.context_orphans``。

    和 :func:`plan_tasks` 一样在使用上游产出的时候重新核对：还没跑 mutate、mutate 对着的副本 / 目标集 / spec / oracle gate /
    mutate 相关代码 / 沙箱（有沙箱时）/ 种子 / 宿主 libclang 与现状不符、L1–L3 参照的 L0 变异体已重做，都直接报 StageError，
    不会悄悄用上过期的变异体。
    """
    root = _pristine(ctx, spec)
    targets = load_targets(ctx.ws, spec.name, root=root)
    trec = {r.get("target_id"): r for r in load_target_records(ctx.ws, spec.name)}
    sh = spec_digest(spec)
    ids_hash = _target_ids_hash(targets)
    l0_entry = load_mutants(ctx.ws, spec.name, "L0")[0]
    expect_common: dict[str, Any] = {
        "spec": sh, "targets": ids_hash, "code": code_digest("mutate"), "seed": ctx.seed, "version": STAGE_VERSIONS["mutate"],
        "clang": clangcfg.environment(), "strip_comments": True}
    if ctx.sandbox is not None:                       # generate 不碰沙箱，沙箱身份留给 evaluate 核对
        expect_common["env"] = ctx.env(full=True)
    planned: list[PlannedTask] = []
    voids: dict[str, dict[str, int]] = {}
    for level in _levels_arg(levels):
        view = _level_view(ctx, spec, level, ids_hash, require_gate)
        if view is None:
            voids[level] = {"copy_invalid": len(targets)}
            continue
        entry, mutants = load_mutants(ctx.ws, spec.name, level)
        path = ctx.ws.mutants_file(spec.name, level)
        if entry is None or not path.is_file():
            raise StageError(f"{spec.name}@{level}: 还没有 {path.name}，请先运行 mutate")
        mi = entry.get("inputs") or {}
        expect = {**expect_common, "obf": view.obf.get("input_hash"), "tree": view.obf.get("tree_hash")}
        if view.gate is not None:
            expect["gate"] = str(view.gate.get("input_hash") or "")
        stale = [k for k, v in expect.items() if mi.get(k) != v]
        if level != "L0" and l0_entry is not None and mi.get("ref") != l0_entry.get("input_hash"):
            stale.append("ref")
        if stale:
            raise StageError(f"{spec.name}@{level}: 变异体是对着另一份{'、'.join(_MUTANT_WHAT[k] for k in stale)}生成的"
                             "（上游产出、mutate 相关代码、沙箱镜像、--seed 或 L0 的 mutate 变了，而没重跑 mutate？），请先运行 mutate")
        if len(mutants) != entry.get("n_killed") or any(r.get("input_hash") != entry.get("input_hash")
                                                       for r in mutants.values()):
            raise StageError(f"{spec.name}@{level}: {path.name} 与汇总文件对不上（被改动过？），请用 mutate --force 重做")
        void: Counter = Counter()
        for t in targets:
            why = _common_void(ctx, view, trec, t, require_gate)
            rec = mutants.get(t.target_id)
            if why is None and rec is None:
                why = "no_killed_mutant"
            if why is None and level != "L0" and _own_name_in_literals(t.name, str(rec["mutant_code"])):
                why = "name_in_literal"
            if why is not None:
                void[why] += 1
                continue
            located = _relocate(spec, view, t)
            if located is None:
                void["not_relocated"] += 1
                continue
            context, orphans = prune_context(located.deps, str(rec["mutant_code"]))
            if "context_orphans" in rec and sorted(rec["context_orphans"]) != sorted(orphans):
                raise StageError(f"{t.target_id}@{level}: 变异体记录里的 context_orphans 与现在重新算的不一致"
                                 f"（{rec['context_orphans']} 对 {orphans}），请用 mutate --force 重做")
            task = RepairTask(
                task_id=f"{t.target_id}@{level}#repair", target_id=t.target_id, corpus=spec.name, level=level,
                obf_name=located.name, obf_signature=located.signature, context=context,
                includes=list(located.deps.includes), language_std=spec.std or "c99",
                mutant_code=str(rec["mutant_code"]), operator=str(rec["operator"]), context_orphans=orphans)
            planned.append(_planned(
                task, spec, t, located, view, trec, sh,
                mutation={"operator": rec["operator"], "site_index": rec["site_index"],
                          "original_text": rec["original_text"], "mutated_text": rec["mutated_text"]},
                kill_mode=rec.get("kill_mode"), same_operator_as_L0=rec.get("same_operator_as_L0"),
                same_site_as_L0=rec.get("same_site_as_L0"), mutants_hash=str(entry.get("input_hash") or "")))
        voids[level] = dict(void)
    return planned, voids


def repair_inputs_present(ws: Workspace, specs: Sequence[CorpusSpec], levels: Optional[Iterable[str]] = None) -> bool:
    """所选语料 × 级别里是不是已有 mutate 的产出（任何一个 mutants-<语料>-<级别>.jsonl 存在）。"""
    return any(ws.mutants_file(s.name, lv).is_file() for s in specs for lv in _levels_arg(levels))


def resolve_tasks(ctx: Context, specs: Sequence[CorpusSpec], levels: Optional[Iterable[str]], task: str = "all"
                  ) -> tuple[list[str], list[str]]:
    """``--task`` 的取值 → （要处理的任务列表, 给用户看的说明）。

    ``generate`` / ``repair`` 就是它自己（修复任务缺 mutate 产出时，出题那一步会报错并提示先运行 mutate）；
    ``all``：生成任务照常处理，修复任务只在 mutate 已经有产出时一并处理，没有就在说明里写明「已跳过」，
    免得只想做生成任务的人因为还没跑 mutate 而被拦住。
    """
    if task in TASKS:
        return [task], []
    if task != "all":
        raise StageError(f"未知任务类型 {task!r}，可选 {', '.join(TASKS)}、all")
    if repair_inputs_present(ctx.ws, specs, levels):
        return list(TASKS), []
    return ["generate"], ["修复任务：还没有 mutate 的产出（results/mutants-<语料>-<级别>.jsonl），已跳过；"
                          "要处理修复任务先运行 python3 -m bench mutate（或 --task generate 去掉这条提示）"]


# ----------------------------------------------------------------------
# generate：调模型，缓存并落盘回复
# ----------------------------------------------------------------------
def build_llm(name: str, ws: Workspace, *, oracle: Optional[dict[str, str]] = None, use_cache: bool = True) -> LLM:
    """按 --model 的取值构造 LLM：mock-* 用 MockLLM；其余走 OpenAICompat 预设，缓存和 usage 日志落在工作区的 results/ 下。

    API key 不在这里检查（``require_key=False``），到第一次真正发请求时才检查：所有回复都已齐全、整段被跳过，
    或只回放缓存的场景不需要 key。``use_cache=False`` 时不读缓存（照常写入），``--force`` 才能真的重新采样。
    """
    if name.startswith("mock-"):
        return make_llm(name, oracle=oracle)
    return make_llm(name, cache=Cache(ws.cache_dir, read=use_cache), usage_path=ws.usage_file, require_key=False)


@dataclass(frozen=True)
class TaskKind:
    """两类任务（功能规格生成 / 缺陷修复）在 generate、evaluate 两个阶段里不同的那几处：出题函数、提示词、数据文件名。
    其余流程（调模型、缓存、续跑、后处理、沙箱评测、归因、完整性记录）完全共用。"""

    name: str                                                       # generate / repair
    plan: Callable[..., tuple[list[PlannedTask], dict[str, dict[str, int]]]]    # plan_tasks / plan_repair_tasks
    messages: Callable[[Any], list[dict]]                                           # build_messages / build_repair_messages
    prompt_digest: str                                              # 提示词模板摘要（全长）
    prompt_version: str
    tasks_file: Callable[[Workspace], Path]
    generations_file: Callable[[Workspace, str], Path]
    failures_file: Callable[[Workspace, str], Path]
    results_file: Callable[[Workspace, str], Path]
    completeness_file: Callable[[Workspace, str], Path]

    @property
    def suffix(self) -> str:
        """阶段结果里附在模型名后面的标记（生成任务不加，保持原有输出）。"""
        return "" if self.name == "generate" else f" [{self.name}]"


TASK_KINDS: dict[str, TaskKind] = {
    "generate": TaskKind(
        "generate", plan_tasks, build_messages, PROMPT_DIGEST, PROMPT_VERSION, lambda ws: ws.tasks_file,
        lambda ws, m: ws.generations_file(m), lambda ws, m: ws.generation_failures_file(m),
        lambda ws, tag: ws.results_file(tag), lambda ws, tag: ws.completeness_file(tag)),
    "repair": TaskKind(
        "repair", plan_repair_tasks, build_repair_messages, REPAIR_PROMPT_DIGEST, REPAIR_PROMPT_VERSION,
        lambda ws: ws.repair_tasks_file, lambda ws, m: ws.repair_generations_file(m),
        lambda ws, m: ws.repair_generation_failures_file(m), lambda ws, tag: ws.repair_results_file(tag),
        lambda ws, tag: ws.repair_completeness_file(tag)),
}


def _kind(task: str) -> TaskKind:
    try:
        return TASK_KINDS[task]
    except KeyError:
        raise StageError(f"未知任务类型 {task!r}，可选 {', '.join(TASKS)}") from None


def _task_record(pt: PlannedTask, messages_hash: str, kind: TaskKind) -> dict:
    base = {**pt.task.to_dict(), "schema_version": SCHEMA_VERSION, "prompt_digest": kind.prompt_digest[:16],
            "prompt_version": kind.prompt_version, "messages_hash": messages_hash, "tree": pt.tree_hash}
    if kind.name == "repair":
        base.update({"task": "repair", "mutation": pt.mutation, "kill_mode": pt.kill_mode,
                     "same_operator_as_L0": pt.same_operator_as_L0, "same_site_as_L0": pt.same_site_as_L0,
                     "mutants": pt.mutants_hash})
    return base


def _write_tasks(ws: Workspace, planned: list[PlannedTask], messages_hash: dict[str, str],
                 scope: set[tuple[str, str]], kind: TaskKind) -> None:
    """把本次出的题并入 tasks.jsonl（修复任务是 tasks-repair.jsonl）：scope（语料, 级别）内的旧记录整体替换，其它保留。"""
    path = kind.tasks_file(ws)
    kept = [r for r in read_jsonl(path) if (r.get("corpus"), r.get("level")) not in scope]
    new = [_task_record(pt, messages_hash[pt.task.task_id], kind) for pt in planned]
    order = {lv: i for i, lv in enumerate(LEVELS)}
    merged = sorted(kept + new, key=lambda r: (r.get("corpus", ""), r.get("target_id", ""),
                                               order.get(r.get("level"), 9)))
    write_jsonl(path, merged)


def _gen_inputs_hash(messages_hash: str, model: str, temperature: float, max_tokens: int, seed: Optional[int],
                     llm_fp: str = "") -> str:
    """一条生成记录的输入哈希：提示词、模型标签、模型实例的配置指纹（实际模型名、端点、思考开关等）、采样参数。"""
    return digest({"messages": messages_hash, "model": model, "llm": llm_fp, "temperature": float(temperature),
                   "max_tokens": int(max_tokens), "seed": seed, "version": STAGE_VERSIONS["generate"]})


def _dry_run_generate(ctx: Context, planned: list[PlannedTask], messages: dict[str, list[dict]],
                      mhash: dict[str, str], model_names: Sequence[str], samples: int, temperature: float,
                      max_tokens: int, void_note: str, kind: TaskKind) -> list[StageResult]:
    """generate 的预估：每个模型还要发多少次请求（扣掉已有的回复）、去重后有多少个不同的请求、提示 token 约多少。
    只做离线计算：构造模型对象不联网、不要 API key，也不写任何文件。"""
    out: list[StageResult] = []
    for name in model_names:
        t0 = time.monotonic()
        fp = build_llm(name, ctx.ws, use_cache=not ctx.no_cache).fingerprint()
        have = {(r.get("task_id"), r.get("sample_idx")): r.get("input_hash")
                for r in read_jsonl(kind.generations_file(ctx.ws, name))}
        todo = unique = tokens = 0
        seen: set[tuple[str, Optional[int]]] = set()
        for pt in planned:
            for idx in range(samples):
                seed_i = idx if temperature > 0 else None
                key = (pt.task.task_id, idx)
                if have.get(key) == _gen_inputs_hash(mhash[pt.task.task_id], name, temperature, max_tokens, seed_i, fp) \
                        and not ctx.force:
                    continue
                todo += 1
                if (mhash[pt.task.task_id], seed_i) not in seen:
                    seen.add((mhash[pt.task.task_id], seed_i))
                    unique += 1
                    tokens += sum(estimate_tokens(str(m.get("content", ""))) for m in messages[pt.task.task_id])
        out.append(_timed("generate", f"{name}{kind.suffix}（预估，未调用）", t0, message=void_note,
                          detail={"tasks": len(planned), "to_call": todo, "unique_requests": unique,
                                  "est_prompt_tokens": tokens, "max_completion_tokens": unique * max_tokens}))
    return out


def _update_failures(path: Path, scope_keys: set[tuple], new: list[dict]) -> None:
    """generation-failures-<model>.jsonl：这次涉及的（任务, 样本）已有的失败记录整体替换成这次的结果
    （成功补上的从文件里消失），其它范围的记录原样保留。没有任何记录且文件不存在就不创建。"""
    old = read_jsonl(path)
    kept = [r for r in old if (r.get("task_id"), r.get("sample_idx")) not in scope_keys]
    merged = kept + new
    if merged or old:
        write_jsonl(path, merged)


def stage_generate(ctx: Context, specs: Sequence[CorpusSpec], model_names: Sequence[str],
                   levels: Optional[Iterable[str]] = None, *, samples: int = 1, temperature: float = 0.0,
                   max_tokens: int = 4096, dry_run: bool = False, task: str = "generate") -> list[StageResult]:
    """按（目标, 级别, 模型, 样本）组织 prompt、调模型，回复追加写 ``generations-<model>.jsonl``，并更新 ``tasks.jsonl``。

    ``task="repair"`` 处理缺陷修复任务：题目来自 :func:`plan_repair_tasks`（先运行 mutate），提示词是
    ``build_repair_messages``，文件名换成 ``tasks-repair.jsonl`` / ``generations-repair-<model>.jsonl`` /
    ``generation-failures-repair-<model>.jsonl``，生成记录多 ``task`` 与 ``mutation`` 字段；其余行为完全一致。

    断点续跑：已有记录且输入哈希（prompt、模型标签、模型实例配置指纹、温度、max_tokens、seed）没变的样本不再调用。
    多样本（``samples > 1``）要求 temperature > 0；只要 temperature > 0，样本 i 就把 i 当 seed 传给模型
    （seed 参与缓存键，所以每个样本单独请求、单独缓存；temperature = 0 时多样本只会命中同一份缓存，没有意义）。
    L1–L3 的提示词逐字相同（见 README 的局限），它们共用同一个缓存键，所以真实调用数约为 L0 + L1 两份。
    ``dry_run=True`` 只估算：还要调用多少次、其中不同的请求有几个、提示 token 约多少，不联网、不写任何文件。

    调用失败的样本（重试用尽的 5xx / 超时 / 400 等）不中断阶段：记进 ``generation-failures-<model>.jsonl``，
    阶段结果标失败，evaluate 与 report 会把它们算作「缺失样本」，不会悄悄从分母里消失；重新运行会续跑。
    401 / 402 / 403（key 无效、余额不足、没有权限）整批必然失败，直接中止阶段，已落盘的样本不受影响。
    """
    if samples < 1:
        raise StageError("samples 必须 >= 1")
    if samples > 1 and temperature <= 0:
        raise StageError("samples > 1 时 temperature 必须 > 0，否则所有样本都是同一份回复")
    kind = _kind(task)
    levels_l = _levels_arg(levels)
    planned: list[PlannedTask] = []
    void_total: Counter = Counter()
    for spec in specs:
        got, voids = kind.plan(ctx, spec, levels_l)
        planned += got
        for per_level in voids.values():
            void_total.update(per_level)
    void_note = (f"作废 {sum(void_total.values())} 个任务（" + ", ".join(f"{k}={v}" for k, v in sorted(void_total.items()))
                 + "）") if void_total else ""
    messages = {pt.task.task_id: kind.messages(pt.task) for pt in planned}
    mhash = {tid: digest(m) for tid, m in messages.items()}
    if dry_run:
        return _dry_run_generate(ctx, planned, messages, mhash, model_names, samples, temperature, max_tokens,
                                 void_note, kind)
    scope = {(spec.name, lv) for spec in specs for lv in levels_l}
    _write_tasks(ctx.ws, planned, mhash, scope, kind)
    oracle = {pt.task.task_id: pt.oracle for pt in planned}
    results: list[StageResult] = []
    for name in model_names:
        t0 = time.monotonic()
        llm = build_llm(name, ctx.ws, oracle=oracle, use_cache=not ctx.no_cache)
        fp = llm.fingerprint()
        path = kind.generations_file(ctx.ws, name)
        want: dict[tuple[str, int], tuple[PlannedTask, str, Optional[int]]] = {}
        for pt in planned:
            for idx in range(samples):
                seed_i = idx if temperature > 0 else None
                ihash = _gen_inputs_hash(mhash[pt.task.task_id], name, temperature, max_tokens, seed_i, fp)
                want[(pt.task.task_id, idx)] = (pt, ihash, seed_i)
        kept, dropped, have = [], 0, set()
        for rec in read_jsonl(path):
            task_id, idx = rec.get("task_id"), rec.get("sample_idx")
            key = (task_id, idx)
            if (rec.get("corpus"), rec.get("level")) not in scope:
                keep = True                                   # 这次不涉及的语料 / 级别：原样保留
            elif key in want:
                keep = rec.get("input_hash") == want[key][1] and not ctx.force and key not in have
                if keep:
                    have.add(key)
            else:
                # 这次没要的样本编号（例如之前跑过 3 个样本、这次只要 1 个）：提示词和参数都没变就留着，别白扔付过费的回复
                keep = (not ctx.force and task_id in mhash and isinstance(idx, int) and rec.get("input_hash")
                        == _gen_inputs_hash(mhash[task_id], name, temperature, max_tokens,
                                            idx if temperature > 0 else None, fp))
            if keep:
                kept.append(rec)
            else:
                dropped += 1                      # 过期（prompt 或参数变了）、重复、任务已作废、或被 --force 要求重做
        if dropped:
            write_jsonl(path, kept)
        todo = [(key, *want[key]) for key in want if key not in have]
        if not todo:
            _update_failures(kind.failures_file(ctx.ws, name), set(want), [])
            results.append(_timed("generate", name + kind.suffix, t0, skipped=True, message=void_note,
                                  detail={"tasks": len(planned)}))
            continue

        def one(item):
            (task_id, idx), pt, _ih, seed_i = item
            meta = {"task_id": task_id, "target_id": pt.task.target_id, "level": pt.task.level,
                    "obf_signature": pt.task.obf_signature, "purpose": "generate", "sample_idx": idx}
            if kind.name == "repair":          # MockLLM 的 unchanged 模式靠它们原样返回变异体
                meta.update(task="repair", mutant_code=pt.task.mutant_code)
            try:
                return chat_with_meta(llm, messages[task_id], temperature=temperature, max_tokens=max_tokens,
                                      seed=seed_i, meta=meta), None
            except MissingAPIKeyError:
                raise
            except LLMHTTPError as exc:
                if exc.fatal:
                    raise                          # key 无效 / 余额不足 / 没有权限：整批都会失败，别再逐个请求
                return None, exc
            except LLMError as exc:
                return None, exc

        n_fail = n_cached = p_tok = c_tok = 0
        failures: list[dict] = []
        for item, (reply, err) in zip(todo, run_ordered(one, todo, ctx.jobs, f"generate {name}", every=25)):
            (task_id, idx), pt, ihash, seed_i = item
            if err is not None:
                n_fail += 1
                log.error("%s 的 %s #%d 调用失败，已跳过（重新运行会续跑）: %s", name, task_id, idx, err)
                failures.append({"schema_version": SCHEMA_VERSION, "task_id": task_id, "sample_idx": idx,
                                 "target_id": pt.task.target_id, "corpus": pt.task.corpus, "level": pt.task.level,
                                 "model": name, "error": str(err)[:300], "status": getattr(err, "status", None),
                                 "created": now_iso(), **({"task": "repair"} if kind.name == "repair" else {})})
                continue
            n_cached += int(reply.cached)
            p_tok += reply.prompt_tokens
            c_tok += reply.completion_tokens
            append_jsonl(path, {
                "schema_version": SCHEMA_VERSION, "task_id": task_id, "target_id": pt.task.target_id,
                "corpus": pt.task.corpus, "level": pt.task.level, "model": name, "sample_idx": idx,
                "text": reply.text, "prompt_tokens": reply.prompt_tokens, "completion_tokens": reply.completion_tokens,
                "cached": reply.cached, "latency_s": round(reply.latency_s, 3), "finish_reason": reply.finish_reason,
                "temperature": float(temperature), "max_tokens": int(max_tokens), "seed": seed_i,
                "api_model": getattr(llm, "api_model", None), "llm_fp": fp,
                "reply_model": (reply.raw or {}).get("model"), "system_fingerprint": reply.system_fingerprint,
                "messages_hash": mhash[task_id], "input_hash": ihash, "created": now_iso(),
                **({"task": "repair", "mutation": pt.mutation} if kind.name == "repair" else {}),
            })
        _update_failures(kind.failures_file(ctx.ws, name), set(want), failures)
        results.append(_timed(
            "generate", name + kind.suffix, t0, ok=n_fail == 0,
            message="；".join(m for m in (f"{n_fail} 次调用失败，重新运行会续跑" if n_fail else "", void_note) if m),
            detail={"tasks": len(planned), "calls": len(todo) - n_fail, "cached": n_cached,
                    "prompt_tokens": p_tok, "completion_tokens": c_tok}))
    return results


# ----------------------------------------------------------------------
# evaluate：后处理 → 替换 → 沙箱编译与测试 → 归因
# ----------------------------------------------------------------------
def load_generations(ws: Workspace, model: str, task: str = "generate") -> list[dict]:
    """generations-<model>.jsonl（修复任务是 generations-repair-<model>.jsonl）里每个（task_id, sample_idx）的最后一条记录，
    保持文件顺序。"""
    latest: dict[tuple, dict] = {}
    for rec in read_jsonl(_kind(task).generations_file(ws, model)):
        latest[(rec.get("task_id"), rec.get("sample_idx"))] = rec
    return list(latest.values())


def _result_hash(ctx: Context, gen: dict, pt: PlannedTask, kind: TaskKind) -> str:
    """一条评测结果的输入哈希：回复（生成记录的输入哈希 + 回复文本）、提示词、副本、spec、oracle gate、
    沙箱（镜像 ID + 资源上限）、评测相关源文件的内容哈希、阶段版本；修复任务另含变异体所在 mutants 文件的输入哈希
    和 mutate 相关代码的摘要（变异体是按那份判定逻辑杀死的）。"""
    inputs = {"gen": gen.get("input_hash"), "reply": digest(gen.get("text") or ""), "tree": pt.tree_hash,
              "spec": pt.spec_hash, "prompt": kind.prompt_digest[:16], "gate": pt.gate_hash,
              "env": ctx.env(full=True), "code": code_digest("evaluate"), "version": STAGE_VERSIONS["evaluate"]}
    if kind.name == "repair":
        inputs["mutants"] = pt.mutants_hash
        inputs["mutate_code"] = code_digest("mutate")
    return digest(inputs)


def _eval_record(ctx: Context, tag: str, gen: dict, pt: PlannedTask, cand: Candidate, ev: Optional[EvalResult],
                 category: str, detail: str, similarity: Optional[float], ihash: str, kind: TaskKind) -> dict:
    d = ev.to_dict() if ev is not None else {}
    policy = ev is None and bool(cand.violations)
    rec = {
        "schema_version": SCHEMA_VERSION, "task": kind.name, "tag": tag, "task_id": pt.task.task_id,
        "target_id": pt.task.target_id,
        "corpus": pt.task.corpus, "level": pt.task.level, "model": gen["model"], "sample_idx": gen["sample_idx"],
        "stage": ev.stage if ev is not None else ("policy" if policy else "extract"),
        "rc": ev.rc if ev is not None else None, "timed_out": ev.timed_out if ev is not None else False,
        "signal": ev.signal if ev is not None else None, "diagnostics": d.get("diagnostics", []),
        "test_stdout_tail": d.get("test_stdout_tail", ""),
        "duration_s": round(ev.duration_s, 3) if ev is not None else 0.0,
        "category": category, "detail": detail,
        "similarity": None if similarity is None else round(similarity, 4),
        "issues": list(cand.issues), "violations": list(cand.violations), "analysis_skipped": cand.analysis_skipped,
        "prompt_tokens": gen.get("prompt_tokens", 0),
        "completion_tokens": gen.get("completion_tokens", 0), "cached": bool(gen.get("cached")),
        "stub_kill_mode": pt.stub_kill_mode,
        "temperature": gen.get("temperature"), "max_tokens": gen.get("max_tokens"), "seed": gen.get("seed"),
        "api_model": gen.get("api_model"),
        "inputs": {"spec": pt.spec_hash, "tree": pt.tree_hash, "prompt": kind.prompt_digest[:16],
                   "gen": gen.get("input_hash"), "messages": gen.get("messages_hash"),
                   "reply": digest(gen.get("text") or ""), "gate": pt.gate_hash,
                   "env": ctx.env(full=True), "code": code_digest("evaluate")},
        "sandbox": sandbox_identity(ctx.sandbox),
        "input_hash": ihash, "created": now_iso(),
    }
    if kind.name == "repair":
        # 候选与变异体规范化后相同 = 模型什么也没改；与该级原函数规范化后相同 = 逐字还原（只比排版与注释之外的内容）。
        # 注意单点缺陷（ROR / AOR / LOR / OBO / NEG）的任何正确修复都逐字等于原函数，restored 约等于 pass，
        # similarity 在候选没大改时天然接近 1：修复任务里这两个量不是「凭记忆还原」的指标（报告里有说明）
        rec.update({
            "mutation": pt.mutation, "kill_mode": pt.kill_mode, "same_operator_as_L0": pt.same_operator_as_L0,
            "same_site_as_L0": pt.same_site_as_L0, "context_orphans": list(pt.task.context_orphans),
            "unchanged": normalized_equal(cand.code, pt.task.mutant_code),
            "restored": normalized_equal(cand.code, pt.oracle)})
        rec["inputs"]["mutants"] = pt.mutants_hash
        rec["inputs"]["mutate_code"] = code_digest("mutate")
    return rec


def stage_evaluate(ctx: Context, specs: Sequence[CorpusSpec], model_names: Sequence[str],
                   levels: Optional[Iterable[str]] = None, *, tag: str = "run", task: str = "generate"
                   ) -> list[StageResult]:
    """对 generations 里的每个样本：后处理出候选代码 → 替换回该级混淆副本 → 沙箱编译并跑项目测试 → 归因 → 相似度，
    追加写 ``results-<tag>.jsonl``。

    ``task="repair"`` 评测缺陷修复任务：读 ``generations-repair-<model>.jsonl``，写 ``results-repair-<tag>.jsonl`` 与
    ``completeness-repair-<tag>.json``；判定流程与生成任务完全相同，记录多 ``task`` / ``mutation`` / ``kill_mode`` /
    ``same_operator_as_L0`` / ``unchanged`` / ``restored`` 字段（候选与变异体、与该级原函数规范化后是否相同）。

    只评测与当前 prompt 对得上的回复（回复记录里的 messages 哈希与重新出的题一致）；已评测且输入哈希没变的样本跳过。
    抽不出代码的回复不进沙箱，直接归 extract_fail（stage 记为 extract）；违反硬性约束的回复（调用 exit 等）同样不进沙箱，
    归 policy_violation（stage 记为 policy）。

    评测完把「应有多少样本、实际评测了多少、缺多少」按（模型, 级别, 语料）写进 ``completeness-<tag>.json``：
    缺的样本（生成调用失败、没生成、回复过期）会让阶段失败（``ctx.allow_missing`` 放行），报告里也会标「不完整」。
    """
    kind = _kind(task)
    levels_l = _levels_arg(levels)
    plan: dict[str, PlannedTask] = {}
    for spec in specs:
        got, _ = kind.plan(ctx, spec, levels_l)
        plan.update({pt.task.task_id: pt for pt in got})
    mhash = {tid: digest(kind.messages(pt.task)) for tid, pt in plan.items()}
    scope_corpora = {spec.name for spec in specs}
    path = kind.results_file(ctx.ws, tag)
    models = set(model_names)

    def in_scope(rec: dict) -> bool:
        return rec.get("model") in models and rec.get("corpus") in scope_corpora and rec.get("level") in levels_l

    # 先整理旧结果：范围外的保留；范围内输入哈希还对得上的保留，其余丢弃重做
    gens_by_model = {m: load_generations(ctx.ws, m, kind.name) for m in model_names}
    expected: dict[tuple, tuple[dict, PlannedTask, str]] = {}
    stale_gen = Counter()
    for m, gens in gens_by_model.items():
        for gen in gens:
            pt = plan.get(gen.get("task_id"))
            if pt is None:
                continue
            if gen.get("messages_hash") != mhash[pt.task.task_id]:
                stale_gen[m] += 1
                continue
            expected[(gen["task_id"], m, gen["sample_idx"])] = (gen, pt, _result_hash(ctx, gen, pt, kind))
    kept, dropped, have = [], 0, set()
    for rec in read_jsonl(path):
        key = (rec.get("task_id"), rec.get("model"), rec.get("sample_idx"))
        if not in_scope(rec):
            kept.append(rec)
        elif key in expected and rec.get("input_hash") == expected[key][2] and not ctx.force and key not in have:
            kept.append(rec)
            have.add(key)
        else:
            dropped += 1
    if dropped:
        write_jsonl(path, kept)

    results: list[StageResult] = []
    for m in model_names:
        t0 = time.monotonic()
        todo = [(key, *val) for key, val in expected.items() if key[1] == m and key not in have]
        missing, comp_cells = _completeness(ctx, m, plan, gens_by_model[m], have, todo, kind)
        if not todo:
            bad = _unreliable_tasks(path, m, in_scope)
            results.append(_timed("evaluate", f"{m}/{tag}{kind.suffix}", t0, skipped=True,
                                  ok=(ctx.allow_missing or missing["missing"] == 0) and not bad,
                                  message="；".join(x for x in (missing["message"], _unreliable_note(bad)) if x),
                                  detail={"records": sum(1 for k in have if k[1] == m),
                                          "stale_generations": stale_gen[m], "missing": missing["missing"]}))
            _write_completeness(ctx.ws, tag, comp_cells, {m}, set(levels_l), scope_corpora, kind)
            continue
        # 串行：后处理（libclang）与归因输入；线程池里只做沙箱评测
        prepared: list[tuple] = []
        for key, gen, pt, ihash in todo:
            cand = process_reply(gen.get("text") or "", pt.task, finish_reason=gen.get("finish_reason"),
                                 allow_directives=pt.allow_directives)
            prepared.append((gen, pt, ihash, cand))

        def one(item: tuple) -> Optional[EvalResult]:
            gen, pt, _ih, cand = item
            if cand.code is None or cand.violations:
                return None                       # 抽不出代码 / 违反硬性约束：不进沙箱
            return evaluate(pt.spec, pt.obf_root, pt.located, cand.code, ctx.need_sandbox(), m, pt.task.task_id,
                            gen["sample_idx"], work_root=ctx.ws.work)

        cats: Counter = Counter()
        for item, ev in zip(prepared, run_ordered(one, prepared, ctx.jobs, f"evaluate {m}", every=20)):
            gen, pt, ihash, cand = item
            category, detail = attribute(ev, cand)
            sim = normalized_similarity(cand.code, pt.oracle) if cand.code else None
            cats[category] += 1
            append_jsonl(path, _eval_record(ctx, tag, gen, pt, cand, ev, category, detail, sim, ihash, kind))
        others = ", ".join(f"{k}={v}" for k, v in sorted(cats.items()) if k != "pass")
        bad = _unreliable_tasks(path, m, in_scope)
        msg = "；".join(x for x in (others, missing["message"], _unreliable_note(bad)) if x)
        results.append(_timed("evaluate", f"{m}/{tag}{kind.suffix}", t0,
                              ok=(ctx.allow_missing or missing["missing"] == 0) and not bad, message=msg,
                              detail={"evaluated": len(prepared), "pass": cats["pass"],
                                      "stale_generations": stale_gen[m], "missing": missing["missing"],
                                      "categories": {k: v for k, v in cats.items()}}))
        _write_completeness(ctx.ws, tag, comp_cells, {m}, set(levels_l), scope_corpora, kind)
    return results


def _unreliable_tasks(path: Path, model: str, in_scope: Callable[[dict], bool]) -> set[tuple[str, str]]:
    """这个模型在范围内的评测记录里，``unchanged`` 的候选却通过了项目测试的修复题（生成任务的记录没有 ``unchanged``，恒为空）。"""
    return unreliable_tasks(r for r in read_jsonl(path) if r.get("model") == model and in_scope(r))


def _unreliable_note(bad: set[tuple[str, str]]) -> str:
    """评测阶段的提示：这些修复题的变异体没有被测试稳定地杀死（模型什么也没改，测试却过了），报告会把它们整题剔除，
    阶段按失败处理（不能悄悄把它们算进通过率）。"""
    if not bad:
        return ""
    names = ", ".join(sorted(tid for _, tid in bad)[:3])
    return f"{len(bad)} 道修复题的候选与变异体相同（unchanged）却通过了项目测试，变异体不可靠（{names}{' 等' if len(bad) > 3 else ''}）"


def _completeness(ctx: Context, model: str, plan: dict[str, PlannedTask], gens: list[dict], have: set,
                  todo: list, kind: TaskKind) -> tuple[dict, list[dict]]:
    """某个模型在 plan 范围内「应有多少样本、评测了多少、缺多少」。

    应有 = 出了题的任务数 × 每个任务的样本数（样本数取这个模型已有回复和失败记录里最大的样本编号 + 1，至少 1）。
    缺的样本分三种：生成调用失败（generation-failures 里有记录）、回复过期（提示词变了）、还没生成。
    todo 里的样本马上就会被评测，不算缺。
    """
    failures = {(r.get("task_id"), r.get("sample_idx")) for r in read_jsonl(kind.failures_file(ctx.ws, model))
                if r.get("task_id") in plan}
    idxs = [g.get("sample_idx", 0) for g in gens if g.get("task_id") in plan and isinstance(g.get("sample_idx"), int)]
    idxs += [i for (tid, i) in failures if isinstance(i, int)]
    k = (max(idxs) + 1) if idxs else 1
    evaluated = {(key[0], key[2]) for key in have if key[1] == model} | {(key[0], key[2]) for key, *_ in todo}
    valid_gen = {(g.get("task_id"), g.get("sample_idx")) for g in gens}
    cells: dict[tuple, dict] = {}
    for tid, pt in plan.items():
        cell = cells.setdefault((pt.task.level, pt.task.corpus), {
            "model": model, "level": pt.task.level, "corpus": pt.task.corpus, "expected": 0, "have": 0,
            "failed_calls": 0, "stale_generations": 0, "not_generated": 0})
        for idx in range(k):
            cell["expected"] += 1
            key = (tid, idx)
            if key in evaluated:
                cell["have"] += 1
            elif key in failures:
                cell["failed_calls"] += 1
            elif key in valid_gen:
                cell["stale_generations"] += 1
            else:
                cell["not_generated"] += 1
    rows = list(cells.values())
    for r in rows:
        r["missing"] = r["expected"] - r["have"]
    tot = Counter()
    for r in rows:
        tot.update({x: r[x] for x in ("expected", "have", "failed_calls", "stale_generations", "not_generated", "missing")})
    msg = ""
    if tot["missing"]:
        parts = [f"{n} 个{what}" for n, what in ((tot["failed_calls"], "调用失败"), (tot["not_generated"], "未生成"),
                                                (tot["stale_generations"], "回复过期")) if n]
        msg = f"缺 {tot['missing']}/{tot['expected']} 个样本（{'、'.join(parts)}）"
    return {"missing": tot["missing"], "message": msg}, rows


def _write_completeness(ws: Workspace, tag: str, rows: list[dict], models: set, levels: set, corpora: set,
                        kind: Optional[TaskKind] = None) -> None:
    """把这次范围内的完整性记录并入 completeness-<tag>.json（修复任务是 completeness-repair-<tag>.json；范围外的格原样保留）。"""
    path = (kind or TASK_KINDS["generate"]).completeness_file(ws, tag)
    old = (read_json(path) or {}).get("cells", [])
    kept = [c for c in old if not (c.get("model") in models and c.get("level") in levels and c.get("corpus") in corpora)]
    write_json(path, {"schema_version": SCHEMA_VERSION, "tag": tag, "created": now_iso(),
                      "cells": sorted(kept + rows, key=lambda c: (c["model"], c["level"], c["corpus"]))})


# ----------------------------------------------------------------------
# check-mutants：把 mutate 选定的变异体在沙箱里回放复核
# ----------------------------------------------------------------------
def stage_check_mutants(ctx: Context, spec: CorpusSpec, levels: Optional[Iterable[str]] = None, *,
                        with_oracle: bool = False) -> StageResult:
    """回放复核 mutate 选定的变异体：每个出了题的（目标, 级别），把变异体当作候选再走一遍 evaluate（相当于 ``mock-unchanged`` 的
    评测，但不经过 generate / evaluate 两个阶段，也不写评测结果），要求都在测试阶段失败；``with_oracle`` 为真时，再把**去注释后的
    原函数**放回副本跑一遍，要求都通过（oracle 回放）。

    回放用 :func:`plan_repair_tasks` 取题，所以会像出题一样核对 mutate 产出有没有过期（过期直接报 StageError）。统计：

    * ``killed``：回放仍在测试阶段失败（归因属于 ``test_fail`` / ``runtime_crash`` / ``runtime_timeout``）；
    * ``same_mode``：回放的归因与 mutate 记下的 ``kill_mode`` 相同，``flipped`` 是不同的（崩溃 / 超时 / OOM 的结局随负载变，
      两种结局都算被杀死，只影响分层标签；``--jobs 1`` 在安静的机器上回放可以看到哪些标签不稳定）；
    * ``passed``：回放通过了项目测试（变异体没有被稳定地杀死，**阶段失败**）；``other``：既不是失败于测试阶段也没通过（编译 / 构建
      没过等，同样算失败）；
    * ``oracle_failed``：去注释的原函数没通过（``with_oracle``）。

    结果写进 ``results/mutants-<corpus>.check.json``（``experiment_record`` 的实验记录据此出表），不改动 mutate 的任何产出。
    """
    t0 = time.monotonic()
    planned, voids = plan_repair_tasks(ctx, spec, levels)
    sandbox = ctx.need_sandbox()

    def one(pt: PlannedTask) -> dict[str, Any]:
        task = pt.task
        ev = evaluate(pt.spec, pt.obf_root, pt.located, task.mutant_code, sandbox, REPLAY_MODEL, f"{task.task_id}#replay", 0,
                      work_root=ctx.ws.work)
        category = attribute(ev, Candidate(task.mutant_code, []))[0]
        row: dict[str, Any] = {"task_id": task.task_id, "level": task.level, "recorded": pt.kill_mode, "replay": category,
                               "stage": ev.stage}
        if with_oracle:
            original = strip_function_comments(pt.oracle)
            ev0 = evaluate(pt.spec, pt.obf_root, pt.located, original, sandbox, REPLAY_MODEL, f"{task.task_id}#oracle", 0,
                           work_root=ctx.ws.work)
            row["oracle"] = attribute(ev0, Candidate(original, []))[0]
        return row

    rows = list(run_ordered(one, planned, ctx.jobs, f"check-mutants {spec.name}", every=20))
    per_level: dict[str, Counter] = {}
    flipped: list[dict] = []
    bad: list[dict] = []
    for row in rows:
        slot = per_level.setdefault(row["level"], Counter())
        slot["replayed"] += 1
        killed = row["stage"] == "test" and row["replay"] in KILL_MODES
        slot["killed"] += killed
        slot["same_mode"] += row["replay"] == row["recorded"]
        slot["passed"] += row["replay"] == "pass"
        slot["other"] += not killed and row["replay"] != "pass"
        slot["oracle_failed"] += "oracle" in row and row["oracle"] != "pass"
        if killed and row["replay"] != row["recorded"]:
            flipped.append({k: row[k] for k in ("task_id", "recorded", "replay")})
        if not killed or row.get("oracle", "pass") != "pass":
            bad.append(row)
    total = Counter()
    for slot in per_level.values():
        total.update(slot)
    hashes = {pt.task.level: pt.mutants_hash for pt in planned}
    write_json(ctx.ws.mutants_check_file(spec.name), {
        "schema_version": SCHEMA_VERSION, "corpus": spec.name, "created": now_iso(), "with_oracle": with_oracle,
        "inputs": {"mutants": hashes, "env": ctx.env(full=True), "seed": ctx.seed}, "sandbox": sandbox_identity(sandbox),
        "levels": {lv: dict(per_level[lv]) for lv in LEVELS if lv in per_level}, "total": dict(total),
        "flipped": flipped, "failed": bad, "voided": voids})
    ok = not bad
    msg = ""
    if bad:
        msg = f"{len(bad)} 个变异体回放不合格（" + ", ".join(f"{r['task_id']}={r['replay']}" for r in bad[:3]) + \
              (" 等" if len(bad) > 3 else "") + "）：没有被稳定地杀死，或去注释的原函数没通过"
    elif flipped:
        msg = f"{len(flipped)} 个变异体的杀死方式与 mutate 记录的不同（两种结局都算被杀死，只影响分层标签）"
    return _timed("check-mutants", spec.name, t0, ok=ok, message=msg,
                  detail={"replayed": total["replayed"], "killed": total["killed"], "same_mode": total["same_mode"],
                          "passed": total["passed"], "other": total["other"], "flipped": len(flipped),
                          **({"oracle_failed": total["oracle_failed"]} if with_oracle else {})})

