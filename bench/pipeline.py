"""流水线各阶段（设计文档 §2）：prepare → coverage → targets → obfuscate → verify → describe → generate → evaluate → report。

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
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Optional, Sequence, TypeVar

from bench import corpus
from bench.attribute import attribute
from bench.corpus import CorpusSpec
from bench.coverage import function_exec_counts
from bench.describe import DESCRIBE_PROMPT_DIGEST, LeakChecker, describe_detailed
from bench.extract import FunctionTarget, locate
from bench.llm import LLM, Cache, LLMError, LLMHTTPError, MissingAPIKeyError, chat_with_meta, estimate_tokens, make_llm
from bench.obfuscate import LEVELS, load_obf_result, obfuscate
from bench.postprocess import Candidate, has_policy_directives, process_reply
from bench.prompt import PROMPT_DIGEST, PROMPT_VERSION, Task, build_messages
from bench.provenance import code_digest, sandbox_digest, sandbox_identity
from bench.runner import EvalResult, evaluate, read_function
from bench.similarity import normalized_similarity
from bench.store import (
    SCHEMA_VERSION, Workspace, append_jsonl, digest, now_iso, read_json, read_jsonl, spec_digest, write_json,
    write_jsonl,
)
from bench.targets import TargetFilter, select

log = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_SEED", "STAGE_VERSIONS", "StageError", "Context", "StageResult", "PlannedTask",
    "stage_prepare", "stage_coverage", "stage_targets", "stage_obfuscate", "stage_verify", "stage_describe",
    "stage_generate", "stage_evaluate", "load_targets", "load_target_records", "load_descriptions",
    "load_generations", "plan_tasks", "build_llm", "project_names", "run_ordered",
]

DEFAULT_SEED = 20261027

#: 各阶段的版本号：阶段的产出语义变了就加一，旧的结果文件随之失效（代码内容的变化另由 code_digest 覆盖）
STAGE_VERSIONS = {"prepare": 2, "coverage": 2, "targets": 2, "oracle_gate": 2, "describe": 2, "generate": 2,
                  "evaluate": 2}

#: 没有 oracle gate 结果的混淆副本不能用来出题（设计文档 §6.1）
GATE_MODEL = "oracle-gate"

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
    task: Task
    spec: CorpusSpec
    original: FunctionTarget        # L0 原始树上的目标
    located: FunctionTarget         # 该级副本里重新定位的目标（偏移对应副本，splice 用它）
    oracle: str                     # 该级副本里原函数的文本
    obf_root: Path
    tree_hash: str                  # 副本源码树哈希（obf json 里记录的）
    spec_hash: str
    gate_hash: str = ""             # oracle gate 的输入哈希（出题时核对过，评测结果里记下）
    kill_mode: Optional[str] = None  # 该目标的 stub-kill 杀死方式（test_fail / runtime_crash / runtime_timeout）
    allow_directives: bool = False   # 原函数自己带 #define / #pragma：候选里的这类指令不算违反约束


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
        obf = _obf_or_raise(ctx, spec, level)
        void = Counter()
        if not obf.get("ok"):
            void["copy_invalid"] = len(targets)
            voids[level] = dict(void)
            continue
        gate = obf.get("oracle_gate")
        if gate is None and require_gate:
            raise StageError(f"{spec.name}@{level}: 还没有 oracle gate 结果，请先运行 verify")
        if gate is not None and require_gate:
            gi = gate.get("inputs") or {}
            expect = {"obf": obf.get("input_hash"), "tree": obf.get("tree_hash"), "targets": ids_hash}
            stale = [k for k, v in expect.items() if gi.get(k) != v]
            if stale:
                raise StageError(f"{spec.name}@{level}: oracle gate 是对着另一份{'、'.join(_GATE_WHAT[k] for k in stale)}"
                                 "跑的（重跑了 targets 或 obfuscate 而没重跑 verify？），请先运行 verify")
        gate_failed = {f["target_id"] for f in (gate or {}).get("failed", [])}
        gate_passed = set((gate or {}).get("passed_ids") or [])
        mapping = obf.get("rename_map") or {}
        oroot = ctx.ws.obf_root(spec.name, level)
        for t in targets:
            desc = descs.get(t.target_id)
            why = check_description(t)
            if why is not None:
                void[why] += 1
            elif not (ctx.unsafe_debug) and trec.get(t.target_id, {}).get("stub_killed") is not True:
                void["stub_unchecked"] += 1
            elif t.target_id in gate_failed:
                void["oracle_failed"] += 1
            elif require_gate and t.target_id not in gate_passed:
                void["oracle_unchecked"] += 1
            elif level != "L0" and t.name not in mapping:
                void["name_not_renamed"] += 1
            else:
                located = locate(spec, oroot, mapping.get(t.name, t.name), t.file)
                if located is None:
                    void["not_relocated"] += 1
                    continue
                task = Task(
                    task_id=f"{t.target_id}@{level}", target_id=t.target_id, corpus=spec.name, level=level,
                    obf_name=located.name, obf_signature=located.signature, context=located.deps,
                    description=desc["description"], includes=list(located.deps.includes),
                    language_std=spec.std or "c99")
                planned.append(PlannedTask(task=task, spec=spec, original=t, located=located,
                                           oracle=read_function(oroot, located), obf_root=oroot,
                                           tree_hash=obf["tree_hash"], spec_hash=sh,
                                           gate_hash=str((gate or {}).get("input_hash") or ""),
                                           kill_mode=trec.get(t.target_id, {}).get("stub_kill_mode"),
                                           allow_directives=has_policy_directives(read_function(oroot, located))))
        voids[level] = dict(void)
    return planned, voids


_GATE_WHAT = {"obf": "混淆副本记录", "tree": "副本源码树", "targets": "目标集"}


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


def _task_record(pt: PlannedTask, messages_hash: str) -> dict:
    return {**pt.task.to_dict(), "schema_version": SCHEMA_VERSION, "prompt_digest": PROMPT_DIGEST[:16],
            "prompt_version": PROMPT_VERSION, "messages_hash": messages_hash, "tree": pt.tree_hash}


def _write_tasks(ws: Workspace, planned: list[PlannedTask], messages_hash: dict[str, str],
                 scope: set[tuple[str, str]]) -> None:
    """把本次出的题并入 tasks.jsonl：scope（语料, 级别）内的旧记录整体替换，其它保留。"""
    kept = [r for r in read_jsonl(ws.tasks_file) if (r.get("corpus"), r.get("level")) not in scope]
    new = [_task_record(pt, messages_hash[pt.task.task_id]) for pt in planned]
    order = {lv: i for i, lv in enumerate(LEVELS)}
    merged = sorted(kept + new, key=lambda r: (r.get("corpus", ""), r.get("target_id", ""),
                                               order.get(r.get("level"), 9)))
    write_jsonl(ws.tasks_file, merged)


def _gen_inputs_hash(messages_hash: str, model: str, temperature: float, max_tokens: int, seed: Optional[int],
                     llm_fp: str = "") -> str:
    """一条生成记录的输入哈希：提示词、模型标签、模型实例的配置指纹（实际模型名、端点、思考开关等）、采样参数。"""
    return digest({"messages": messages_hash, "model": model, "llm": llm_fp, "temperature": float(temperature),
                   "max_tokens": int(max_tokens), "seed": seed, "version": STAGE_VERSIONS["generate"]})


def _dry_run_generate(ctx: Context, planned: list[PlannedTask], messages: dict[str, list[dict]],
                      mhash: dict[str, str], model_names: Sequence[str], samples: int, temperature: float,
                      max_tokens: int, void_note: str) -> list[StageResult]:
    """generate 的预估：每个模型还要发多少次请求（扣掉已有的回复）、去重后有多少个不同的请求、提示 token 约多少。
    只做离线计算：构造模型对象不联网、不要 API key，也不写任何文件。"""
    out: list[StageResult] = []
    for name in model_names:
        t0 = time.monotonic()
        fp = build_llm(name, ctx.ws, use_cache=not ctx.no_cache).fingerprint()
        have = {(r.get("task_id"), r.get("sample_idx")): r.get("input_hash")
                for r in read_jsonl(ctx.ws.generations_file(name))}
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
        out.append(_timed("generate", f"{name}（预估，未调用）", t0, message=void_note,
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
                   max_tokens: int = 4096, dry_run: bool = False) -> list[StageResult]:
    """按（目标, 级别, 模型, 样本）组织 prompt、调模型，回复追加写 ``generations-<model>.jsonl``，并更新 ``tasks.jsonl``。

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
    levels_l = _levels_arg(levels)
    planned: list[PlannedTask] = []
    void_total: Counter = Counter()
    for spec in specs:
        got, voids = plan_tasks(ctx, spec, levels_l)
        planned += got
        for per_level in voids.values():
            void_total.update(per_level)
    void_note = (f"作废 {sum(void_total.values())} 个任务（" + ", ".join(f"{k}={v}" for k, v in sorted(void_total.items()))
                 + "）") if void_total else ""
    messages = {pt.task.task_id: build_messages(pt.task) for pt in planned}
    mhash = {tid: digest(m) for tid, m in messages.items()}
    if dry_run:
        return _dry_run_generate(ctx, planned, messages, mhash, model_names, samples, temperature, max_tokens,
                                 void_note)
    scope = {(spec.name, lv) for spec in specs for lv in levels_l}
    _write_tasks(ctx.ws, planned, mhash, scope)
    oracle = {pt.task.task_id: pt.oracle for pt in planned}
    results: list[StageResult] = []
    for name in model_names:
        t0 = time.monotonic()
        llm = build_llm(name, ctx.ws, oracle=oracle, use_cache=not ctx.no_cache)
        fp = llm.fingerprint()
        path = ctx.ws.generations_file(name)
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
            _update_failures(ctx.ws.generation_failures_file(name), set(want), [])
            results.append(_timed("generate", name, t0, skipped=True, message=void_note, detail={"tasks": len(planned)}))
            continue

        def one(item):
            (task_id, idx), pt, _ih, seed_i = item
            meta = {"task_id": task_id, "target_id": pt.task.target_id, "level": pt.task.level,
                    "obf_signature": pt.task.obf_signature, "purpose": "generate", "sample_idx": idx}
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
                                 "created": now_iso()})
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
            })
        _update_failures(ctx.ws.generation_failures_file(name), set(want), failures)
        results.append(_timed(
            "generate", name, t0, ok=n_fail == 0,
            message="；".join(m for m in (f"{n_fail} 次调用失败，重新运行会续跑" if n_fail else "", void_note) if m),
            detail={"tasks": len(planned), "calls": len(todo) - n_fail, "cached": n_cached,
                    "prompt_tokens": p_tok, "completion_tokens": c_tok}))
    return results


# ----------------------------------------------------------------------
# evaluate：后处理 → 替换 → 沙箱编译与测试 → 归因
# ----------------------------------------------------------------------
def load_generations(ws: Workspace, model: str) -> list[dict]:
    """generations-<model>.jsonl 里每个（task_id, sample_idx）的最后一条记录，保持文件顺序。"""
    latest: dict[tuple, dict] = {}
    for rec in read_jsonl(ws.generations_file(model)):
        latest[(rec.get("task_id"), rec.get("sample_idx"))] = rec
    return list(latest.values())


def _result_hash(ctx: Context, gen: dict, pt: PlannedTask) -> str:
    """一条评测结果的输入哈希：回复（生成记录的输入哈希 + 回复文本）、提示词、副本、spec、oracle gate、
    沙箱（镜像 ID + 资源上限）、评测相关源文件的内容哈希、阶段版本。"""
    return digest({"gen": gen.get("input_hash"), "reply": digest(gen.get("text") or ""), "tree": pt.tree_hash,
                   "spec": pt.spec_hash, "prompt": PROMPT_DIGEST[:16], "gate": pt.gate_hash,
                   "env": ctx.env(full=True), "code": code_digest("evaluate"),
                   "version": STAGE_VERSIONS["evaluate"]})


def _eval_record(ctx: Context, tag: str, gen: dict, pt: PlannedTask, cand: Candidate, ev: Optional[EvalResult],
                 category: str, detail: str, similarity: Optional[float], ihash: str) -> dict:
    d = ev.to_dict() if ev is not None else {}
    policy = ev is None and bool(cand.violations)
    return {
        "schema_version": SCHEMA_VERSION, "tag": tag, "task_id": pt.task.task_id, "target_id": pt.task.target_id,
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
        "stub_kill_mode": pt.kill_mode,
        "temperature": gen.get("temperature"), "max_tokens": gen.get("max_tokens"), "seed": gen.get("seed"),
        "api_model": gen.get("api_model"),
        "inputs": {"spec": pt.spec_hash, "tree": pt.tree_hash, "prompt": PROMPT_DIGEST[:16],
                   "gen": gen.get("input_hash"), "messages": gen.get("messages_hash"),
                   "reply": digest(gen.get("text") or ""), "gate": pt.gate_hash,
                   "env": ctx.env(full=True), "code": code_digest("evaluate")},
        "sandbox": sandbox_identity(ctx.sandbox),
        "input_hash": ihash, "created": now_iso(),
    }


def stage_evaluate(ctx: Context, specs: Sequence[CorpusSpec], model_names: Sequence[str],
                   levels: Optional[Iterable[str]] = None, *, tag: str = "run") -> list[StageResult]:
    """对 generations 里的每个样本：后处理出候选代码 → 替换回该级混淆副本 → 沙箱编译并跑项目测试 → 归因 → 相似度，
    追加写 ``results-<tag>.jsonl``。

    只评测与当前 prompt 对得上的回复（回复记录里的 messages 哈希与重新出的题一致）；已评测且输入哈希没变的样本跳过。
    抽不出代码的回复不进沙箱，直接归 extract_fail（stage 记为 extract）；违反硬性约束的回复（调用 exit 等）同样不进沙箱，
    归 policy_violation（stage 记为 policy）。

    评测完把「应有多少样本、实际评测了多少、缺多少」按（模型, 级别, 语料）写进 ``completeness-<tag>.json``：
    缺的样本（生成调用失败、没生成、回复过期）会让阶段失败（``ctx.allow_missing`` 放行），报告里也会标「不完整」。
    """
    levels_l = _levels_arg(levels)
    plan: dict[str, PlannedTask] = {}
    for spec in specs:
        got, _ = plan_tasks(ctx, spec, levels_l)
        plan.update({pt.task.task_id: pt for pt in got})
    mhash = {tid: digest(build_messages(pt.task)) for tid, pt in plan.items()}
    scope_corpora = {spec.name for spec in specs}
    path = ctx.ws.results_file(tag)
    models = set(model_names)

    def in_scope(rec: dict) -> bool:
        return rec.get("model") in models and rec.get("corpus") in scope_corpora and rec.get("level") in levels_l

    # 先整理旧结果：范围外的保留；范围内输入哈希还对得上的保留，其余丢弃重做
    gens_by_model = {m: load_generations(ctx.ws, m) for m in model_names}
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
            expected[(gen["task_id"], m, gen["sample_idx"])] = (gen, pt, _result_hash(ctx, gen, pt))
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
        missing, comp_cells = _completeness(ctx, m, plan, gens_by_model[m], have, todo)
        if not todo:
            results.append(_timed("evaluate", f"{m}/{tag}", t0, skipped=True,
                                  ok=ctx.allow_missing or missing["missing"] == 0, message=missing["message"],
                                  detail={"records": sum(1 for k in have if k[1] == m),
                                          "stale_generations": stale_gen[m], "missing": missing["missing"]}))
            _write_completeness(ctx.ws, tag, comp_cells, {m}, set(levels_l), scope_corpora)
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
            append_jsonl(path, _eval_record(ctx, tag, gen, pt, cand, ev, category, detail, sim, ihash))
        others = ", ".join(f"{k}={v}" for k, v in sorted(cats.items()) if k != "pass")
        msg = "；".join(x for x in (others, missing["message"]) if x)
        results.append(_timed("evaluate", f"{m}/{tag}", t0, ok=ctx.allow_missing or missing["missing"] == 0,
                              message=msg,
                              detail={"evaluated": len(prepared), "pass": cats["pass"],
                                      "stale_generations": stale_gen[m], "missing": missing["missing"],
                                      "categories": {k: v for k, v in cats.items()}}))
        _write_completeness(ctx.ws, tag, comp_cells, {m}, set(levels_l), scope_corpora)
    return results


def _completeness(ctx: Context, model: str, plan: dict[str, PlannedTask], gens: list[dict], have: set,
                  todo: list) -> tuple[dict, list[dict]]:
    """某个模型在 plan 范围内「应有多少样本、评测了多少、缺多少」。

    应有 = 出了题的任务数 × 每个任务的样本数（样本数取这个模型已有回复和失败记录里最大的样本编号 + 1，至少 1）。
    缺的样本分三种：生成调用失败（generation-failures 里有记录）、回复过期（提示词变了）、还没生成。
    todo 里的样本马上就会被评测，不算缺。
    """
    failures = {(r.get("task_id"), r.get("sample_idx")) for r in read_jsonl(ctx.ws.generation_failures_file(model))
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


def _write_completeness(ws: Workspace, tag: str, rows: list[dict], models: set, levels: set, corpora: set) -> None:
    """把这次范围内的完整性记录并入 completeness-<tag>.json（范围外的格原样保留）。"""
    path = ws.completeness_file(tag)
    old = (read_json(path) or {}).get("cells", [])
    kept = [c for c in old if not (c.get("model") in models and c.get("level") in levels and c.get("corpus") in corpora)]
    write_json(path, {"schema_version": SCHEMA_VERSION, "tag": tag, "created": now_iso(),
                      "cells": sorted(kept + rows, key=lambda c: (c["model"], c["level"], c["corpus"]))})
