"""命令行（设计文档 §4.15）：``python3 -m bench <cmd> [选项]``。

子命令与流水线阶段一一对应（实现都在 :mod:`bench.pipeline`，这里只解析参数、调用、打印结果）::

    prepare    取语料源码到 corpus/src/<name>，沙箱里确认构建与测试可用
    coverage   沙箱里带 --coverage 构建并跑测试，得到每个函数的执行次数
    targets    选目标函数：静态过滤 → 执行次数 > 0 → 按 seed 抽样 + stub-kill
    obfuscate  生成 L0–L3 混淆副本（--level 选级别）并 verify
    verify     oracle gate：用混淆后的原函数文本逐目标走一遍完整 evaluate
    mutate     缺陷修复任务的出题：在每级副本的目标函数里注入一个能被项目测试捕获的缺陷（--levels --max-tries）
    check-mutants  把 mutate 选定的变异体在沙箱里回放复核：都要稳定地让项目测试失败（--levels --oracle）
    describe   用 LLM 为每个目标生成不含标识符的功能规格（--model）
    generate   按（目标, 级别, 模型, 样本）出题并调模型（--model --levels --samples --task）
    evaluate   后处理 → 替换回副本 → 沙箱编译与测试 → 归因（--model --tag --task）
    report     汇总指标，输出 summary-<tag>.md / .csv 与图（--tag --task）
    e2e        在 tests/fixtures/mini 上用 MockLLM 一键跑通全链路并断言（--backend mock）

``generate`` / ``evaluate`` / ``report`` 的 ``--task`` 选任务类型：``generate`` 是功能规格生成任务，``repair`` 是缺陷修复任务
（先运行 ``mutate``），``all``（默认）两类都处理；修复任务在 mutate 还没有产出时，``all`` 只给出提示并跳过它。

所有命令都接受 ``--corpus``（可多选，默认全部语料）、``--seed``、``--jobs``，并且幂等：输出存在且输入哈希
一致就跳过（``--force`` 强制重做）。输出文件默认在项目的 ``results/``，``--workspace DIR`` 可以把一切
（results、语料副本、混淆副本、临时目录）放进另一个目录。

退出码：0 成功；1 某个阶段失败（含 evaluate / report 发现缺样本）或 e2e 断言不成立；2 用法错误或缺少前置条件
（缺上游产出、没有 API key、docker 不可用等）；130 被 Ctrl-C 中断（已落盘的产出可以续跑）。

``--unsafe-debug`` 是唯一的「放宽可信度措施」开关（跳过 stub-kill、允许 mock 描述、放行没做过 stub-kill 的目标），
只给调试用，产出不可用于正式报告。
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

from bench import corpus
from bench.llm import LLMError
from bench.mutate import DEFAULT_MAX_TRIES
from bench.obfuscate import LEVELS
from bench.pipeline import (
    DEFAULT_SEED, TASKS, Context, StageError, StageResult, build_llm, load_target_records, resolve_tasks,
    stage_check_mutants, stage_coverage, stage_describe, stage_evaluate, stage_generate, stage_mutate, stage_obfuscate,
    stage_prepare, stage_targets, stage_verify,
)
from bench.report import ReportResult, build_report
from bench.store import PROJECT_ROOT, Workspace, now_iso, read_json, read_jsonl
from bench.targets import TargetFilter
from bench.vocab import KILL_MODES

log = logging.getLogger("bench")

#: e2e 的夹具：tests/fixtures/mini（含 spec.json，语料名 mini）
MINI_FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "mini"
E2E_MODELS = ("mock-oracle", "mock-broken", "mock-wrong")
#: 缺陷修复任务的 e2e 多三个模式：unchanged（原样返回变异体，模型什么也没改）、reformat（原函数换排版加注释，应当通过且 restored）、
#: touched（变异体加一条无害语句，缺陷还在，应当失败且 unchanged / restored 都为假）
E2E_REPAIR_MODELS = (*E2E_MODELS, "mock-unchanged", "mock-reformat", "mock-touched")
#: e2e 里 unchanged 的归因与变异体的 kill_mode 一致的比例至少是多少（个别依赖负载的超时 / OOM 除外）
E2E_MIN_SAME_MODE = 0.95
TASK_CHOICES = (*TASKS, "all")
E2E_TAG = "e2e"
E2E_MARKER = ".bench-e2e"
#: describe 的默认模型：设计文档要求「最强可用模型」；deepseek-chat / deepseek-reasoner 在官方文档里有弃用通知，不再当默认
DEFAULT_DESCRIBE_MODEL = "deepseek-v4-pro"
DEFAULT_GENERATE_MODEL = "deepseek-flash"


EPILOG = """\
流水线顺序：prepare → coverage → targets → obfuscate → verify → [mutate → check-mutants] → describe → generate → evaluate → report
（mutate、check-mutants 只给缺陷修复任务用；generate / evaluate / report 的 --task 选任务类型，默认 all）
所有命令幂等（输出存在且输入哈希一致就跳过，--force 强制重做），都接受 --corpus / --seed / --jobs。
先用离线自测确认链路：python3 -m bench e2e --backend mock
退出码：0 成功；1 阶段失败或 e2e 断言不成立；2 用法错误或缺少前置条件；130 被中断。"""


def default_jobs() -> int:
    """默认并发度：CPU 数的一半，最多 6（每个沙箱容器占 1 个 CPU、至多 1 GB 内存）。"""
    return max(1, min(6, (os.cpu_count() or 2) // 2))


# ----------------------------------------------------------------------
# 参数解析
# ----------------------------------------------------------------------
def _common_parent() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    g = p.add_argument_group("通用选项")
    g.add_argument("--corpus", nargs="+", metavar="NAME", help="语料名（可多选，也可以是 spec 的 .json 路径）；默认全部")
    g.add_argument("--seed", type=int, default=DEFAULT_SEED, help="随机种子（默认 %(default)s）：目标抽样、改名表、变换位置都由它决定")
    g.add_argument("--jobs", type=int, default=default_jobs(), help="沙箱与 LLM 调用的并发度（默认 %(default)s）")
    g.add_argument("--force", action="store_true", help="忽略已有产出，强制重做")
    g.add_argument("--workspace", metavar="DIR", help="把 results、语料副本、混淆副本、临时目录都放进 DIR（默认用项目目录）")
    g.add_argument("--image", default="gcc:13", help="沙箱镜像（默认 %(default)s）")
    g.add_argument("--cpus", type=float, default=1.0, help="每个沙箱容器的 CPU 数（默认 %(default)s）")
    g.add_argument("--memory", default="1g", help="每个沙箱容器的内存上限（默认 %(default)s）")
    g.add_argument("--unsafe-debug", action="store_true",
                   help="放宽可信度措施（targets 可跳过 stub-kill、允许 mock 描述、放行没做过 stub-kill 的目标），"
                        "只给调试用，产出不可用于正式报告")
    g.add_argument("-v", "--verbose", action="store_true", help="输出调试日志")
    return p


def build_parser() -> argparse.ArgumentParser:
    parent = _common_parent()
    ap = argparse.ArgumentParser(prog="python3 -m bench", description="大模型代码生成鲁棒性评测平台", epilog=EPILOG,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", metavar="<cmd>", required=True)

    def add(name: str, help_: str) -> argparse.ArgumentParser:
        return sub.add_parser(name, parents=[parent], help=help_, description=help_)

    p = add("prepare", "取语料源码并在沙箱里确认构建与测试可用")
    p.add_argument("--src", metavar="DIR", help="本地源码目录（没有 git 仓库的语料或夹具用；只能配合单个 --corpus）")
    p.add_argument("--no-build-check", action="store_true", help="只取源码，不在沙箱里跑构建与测试")

    add("coverage", "沙箱里带 --coverage 构建并跑测试，得到函数执行次数")

    p = add("targets", "选目标函数（静态过滤、执行次数、stub-kill）")
    d = TargetFilter()
    p.add_argument("--max-per-corpus", type=int, default=d.max_per_corpus, help="每个语料最多入选的目标数（默认 %(default)s）")
    p.add_argument("--min-statements", type=int, default=d.min_statements, help="最少语句数（默认 %(default)s）")
    p.add_argument("--min-lines", type=int, default=d.min_lines, help="最少行数（默认 %(default)s）")
    p.add_argument("--max-lines", type=int, default=d.max_lines, help="最多行数（默认 %(default)s）")
    p.add_argument("--no-exec-filter", action="store_true", help="不要求目标在项目测试里被执行过")
    p.add_argument("--no-stub-kill", action="store_true",
                   help="不做 stub-kill（设计文档规定不可裁掉，必须同时加 --unsafe-debug，产出不可用于正式报告）")
    p.add_argument("--lazy", action="store_true", help="stub-kill 凑够名额就停，不统计整个语料「测试覆盖不住」的函数数")

    p = add("obfuscate", "生成混淆副本并 verify")
    p.add_argument("--level", "--levels", dest="levels", nargs="+", choices=LEVELS, metavar="LEVEL",
                   help=f"级别，可选 {' '.join(LEVELS)}（默认全部）")
    p.add_argument("--strict-stringify", choices=["auto", "on", "off"], default="auto",
                   help="符号混淆对字符串化宏的处理：auto 先非 strict、验证失败再退回 strict（默认）；on / off 固定一种")

    p = add("verify", "oracle gate：混淆后的原函数逐目标走一遍完整 evaluate")
    p.add_argument("--level", "--levels", dest="levels", nargs="+", choices=LEVELS, metavar="LEVEL", help="级别（默认全部）")

    p = add("mutate", "缺陷修复任务的出题：在每级副本的目标函数里注入一个能被项目测试捕获的缺陷")
    p.add_argument("--level", "--levels", dest="levels", nargs="+", choices=LEVELS, metavar="LEVEL",
                   help="级别（默认全部）；不含 L0 时要求 L0 的 mutate 产出有效（其它级别优先复用 L0 选定的算子）")
    p.add_argument("--max-tries", type=int, default=DEFAULT_MAX_TRIES,
                   help="每个（目标, 级别）最多试多少个位点的 kill 检查，试满仍没有被杀死的变异体就放弃（默认 %(default)s）")

    p = add("check-mutants", "回放复核 mutate 选定的变异体：都要稳定地让项目测试失败，杀死方式有没有变")
    p.add_argument("--level", "--levels", dest="levels", nargs="+", choices=LEVELS, metavar="LEVEL", help="级别（默认全部）")
    p.add_argument("--oracle", action="store_true",
                   help="另把去注释后的原函数放回副本再跑一遍，都要通过（相当于 mock-oracle 的评测）")

    p = add("describe", "用 LLM 为每个目标生成功能规格（一次生成、四级共用）")
    p.add_argument("--model", default=DEFAULT_DESCRIBE_MODEL,
                   help="描述模型（默认 %(default)s；mock-oracle 等 mock-* 用于离线自测）")

    p = add("generate", "按（目标, 级别, 模型, 样本）出题并调模型")
    p.add_argument("--model", nargs="+", default=[DEFAULT_GENERATE_MODEL], metavar="MODEL", help="生成模型，可多个（默认 %(default)s）")
    p.add_argument("--levels", "--level", dest="levels", nargs="+", choices=LEVELS, metavar="LEVEL", help="级别（默认全部）")
    p.add_argument("--samples", type=int, default=1, help="每个任务的样本数（默认 %(default)s；>1 需要 --temperature > 0）")
    p.add_argument("--temperature", type=float, default=0.0, help="采样温度（默认 %(default)s）")
    p.add_argument("--max-tokens", type=int, default=4096, help="每次回复的 max_tokens（默认 %(default)s）")
    p.add_argument("--dry-run", action="store_true", help="只估算要调用多少次、多少提示 token，不调模型、不写文件")
    p.add_argument("--no-cache", action="store_true",
                   help="不读 LLM 缓存（照常写入）：--force 只作废已有记录，缓存命中仍会原样复用旧回复，配合 --no-cache 才会重新采样")
    p.add_argument("--task", choices=TASK_CHOICES, default="all",
                   help="任务类型（默认 %(default)s）：generate = 功能规格生成，repair = 缺陷修复（先运行 mutate），"
                        "all = 两类都处理（修复任务在 mutate 没有产出时给出提示并跳过）")

    p = add("evaluate", "评测回复：替换回副本、沙箱编译与测试、归因")
    p.add_argument("--model", nargs="+", required=True, metavar="MODEL", help="要评测的模型（generations 里已有回复）")
    p.add_argument("--levels", "--level", dest="levels", nargs="+", choices=LEVELS, metavar="LEVEL", help="级别（默认全部）")
    p.add_argument("--tag", default="run", help="结果标签，写入 results-<tag>.jsonl（默认 %(default)s）")
    p.add_argument("--allow-missing", action="store_true", help="有样本缺失（生成调用失败、没生成、回复过期）也不算失败")
    p.add_argument("--task", choices=TASK_CHOICES, default="all",
                   help="任务类型（默认 %(default)s）：generate = 功能规格生成，repair = 缺陷修复（先运行 mutate），"
                        "all = 两类都处理（修复任务在 mutate 没有产出时给出提示并跳过）；"
                        "修复任务的结果写进 results-repair-<tag>.jsonl")

    p = add("report", "汇总指标，输出 summary-<tag>.md / .csv 与图")
    p.add_argument("--tag", default="run", help="结果标签（默认 %(default)s）")
    p.add_argument("--task", choices=TASK_CHOICES, default="all",
                   help="任务类型（默认 %(default)s）：generate / repair 只汇总该类任务的结果，all = 两类并列（哪一类没有结果文件就只出另一类）")
    p.add_argument("--format", nargs="+", choices=["md", "csv", "svg"], default=["md", "csv", "svg"], dest="formats",
                   help="输出格式（默认全部）")
    p.add_argument("--allow-stale", action="store_true", help="输入哈希对不上的记录也纳入统计（默认排除并在报告里列出）")
    p.add_argument("--allow-missing", action="store_true", help="有样本缺失或结果文件有坏行时，命令也返回 0（报告里照常标注）")

    p = add("e2e", "在 mini 夹具上用 MockLLM 一键跑通全链路并断言")
    p.add_argument("--backend", choices=["mock"], default="mock", help="模型后端（目前只有 mock）")
    p.add_argument("--keep", action="store_true", help="保留上次的工作目录，不先清空（用来演示幂等：第二次几乎全部跳过）；"
                   "工作目录用 --workspace 指定（默认 tmp/e2e），非空且不是 e2e 建的目录会被拒绝")
    p.add_argument("--levels", "--level", dest="levels", nargs="+", choices=LEVELS, metavar="LEVEL", help="级别（默认全部）")
    return ap


# ----------------------------------------------------------------------
# 辅助
# ----------------------------------------------------------------------
def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S", force=True)


def _make_sandbox(args: argparse.Namespace) -> Any:
    from bench.sandbox import DockerSandbox
    if not DockerSandbox.docker_available(args.image):
        raise StageError(f"docker 不可用，或本地没有镜像 {args.image}")
    return DockerSandbox(image=args.image, cpus=args.cpus, memory=args.memory)


def _workspace(args: argparse.Namespace, create: bool = True) -> Workspace:
    ws = Workspace.under(args.workspace) if args.workspace else Workspace.default()
    return ws.ensure() if create else ws


def _resolve_specs(names: Optional[Sequence[str]]) -> list[Any]:
    wanted = list(names) if names else corpus.list_specs()
    if not wanted:
        raise StageError("没有可用的语料 spec（corpus/specs/*.json）")
    return [corpus.load_spec(n) for n in wanted]


def _print_results(results: Sequence[StageResult]) -> bool:
    for r in results:
        print(r.line(), flush=True)
    return all(r.ok for r in results)


# ----------------------------------------------------------------------
# 各子命令
# ----------------------------------------------------------------------
def _cmd_prepare(args: argparse.Namespace, ctx: Context, specs: list[Any]) -> bool:
    if args.src and len(specs) != 1:
        raise StageError("--src 只能配合单个 --corpus 使用")
    ctx.sandbox = None if args.no_build_check else _make_sandbox(args)
    return _print_results([stage_prepare(ctx, s, local_src=Path(args.src) if args.src else None,
                                         check_build=not args.no_build_check) for s in specs])


def _cmd_coverage(args: argparse.Namespace, ctx: Context, specs: list[Any]) -> bool:
    ctx.sandbox = _make_sandbox(args)
    return _print_results([stage_coverage(ctx, s) for s in specs])


def _cmd_targets(args: argparse.Namespace, ctx: Context, specs: list[Any]) -> bool:
    ctx.sandbox = _make_sandbox(args)
    flt = TargetFilter(min_statements=args.min_statements, min_lines=args.min_lines, max_lines=args.max_lines,
                       max_per_corpus=args.max_per_corpus, require_exec=not args.no_exec_filter,
                       require_stub_kill=not args.no_stub_kill)
    return _print_results([stage_targets(ctx, s, flt, scan_all=not args.lazy) for s in specs])


def _cmd_obfuscate(args: argparse.Namespace, ctx: Context, specs: list[Any]) -> bool:
    ctx.sandbox = _make_sandbox(args)
    strict = {"auto": None, "on": True, "off": False}[args.strict_stringify]
    results: list[StageResult] = []
    for s in specs:
        results += stage_obfuscate(ctx, s, args.levels, strict_stringify=strict)
    return _print_results(results)


def _cmd_verify(args: argparse.Namespace, ctx: Context, specs: list[Any]) -> bool:
    ctx.sandbox = _make_sandbox(args)
    results: list[StageResult] = []
    for s in specs:
        results += stage_verify(ctx, s, args.levels)
    return _print_results(results)


def _cmd_mutate(args: argparse.Namespace, ctx: Context, specs: list[Any]) -> bool:
    ctx.sandbox = _make_sandbox(args)
    ok = True
    for s in specs:                      # 每个语料要跑几分钟，做完一个就打印一个
        ok &= _print_results([stage_mutate(ctx, s, args.levels, max_tries=args.max_tries)])
    return ok


def _cmd_check_mutants(args: argparse.Namespace, ctx: Context, specs: list[Any]) -> bool:
    ctx.sandbox = _make_sandbox(args)
    ok = True
    for s in specs:                      # 每个语料要跑几分钟，做完一个就打印一个
        ok &= _print_results([stage_check_mutants(ctx, s, args.levels, with_oracle=args.oracle)])
    return ok


def _cmd_describe(args: argparse.Namespace, ctx: Context, specs: list[Any]) -> bool:
    llm = build_llm(args.model, ctx.ws)
    return _print_results([stage_describe(ctx, s, llm, model_name=args.model) for s in specs])


def _cmd_generate(args: argparse.Namespace, ctx: Context, specs: list[Any]) -> bool:
    ctx.no_cache = args.no_cache
    tasks, notes = resolve_tasks(ctx, specs, args.levels, args.task)
    for note in notes:
        print(f"[generate] {note}", flush=True)
    ok = True
    for task in tasks:
        ok &= _print_results(stage_generate(ctx, specs, args.model, args.levels, samples=args.samples,
                                            temperature=args.temperature, max_tokens=args.max_tokens,
                                            dry_run=args.dry_run, task=task))
    return ok


def _cmd_evaluate(args: argparse.Namespace, ctx: Context, specs: list[Any]) -> bool:
    ctx.sandbox = _make_sandbox(args)
    ctx.allow_missing = args.allow_missing
    tasks, notes = resolve_tasks(ctx, specs, args.levels, args.task)
    for note in notes:
        print(f"[evaluate] {note}", flush=True)
    ok = True
    for task in tasks:
        ok &= _print_results(stage_evaluate(ctx, specs, args.model, args.levels, tag=args.tag, task=task))
    return ok


def _cmd_report(args: argparse.Namespace, ctx: Context, specs: list[Any]) -> bool:
    known = {s.name: s for s in specs}                          # 记录所属语料有 spec 的，多校验一层 spec 摘要
    res = build_report(ctx.ws, args.tag, specs=known, allow_stale=args.allow_stale, formats=args.formats,
                       task=args.task)
    print(f"[report] {args.tag}: 记录 {res.n_records} 条，纳入统计 {res.n_used} 条，输入哈希不符 {res.n_stale} 条", flush=True)
    if len(res.by_task) > 1:
        print("         " + "；".join(f"{t}：记录 {n['records']} 条，纳入 {n['used']} 条" for t, n in res.by_task.items()),
              flush=True)
    for reason, n in res.stale_reasons.items():
        print(f"         - {reason}: {n} 条", flush=True)
    if res.n_bad_lines:
        print(f"         结果文件有 {res.n_bad_lines} 条坏行（例如进程被杀时留下的半行），这些样本没有计入", flush=True)
    if res.n_missing:
        print(f"         缺 {res.n_missing} 个样本（生成调用失败 / 没生成 / 回复过期），表里的 n 不完整", flush=True)
    if res.n_unreliable:
        print(f"         {res.n_unreliable} 道修复题的变异体不可靠（候选与变异体相同却通过了项目测试），已整题从统计里剔除", flush=True)
    for kind, path in res.paths.items():
        print(f"         {kind}: {path}", flush=True)
    if res.n_records == 0:
        return False
    return args.allow_missing or not (res.n_bad_lines or res.n_missing)


# ----------------------------------------------------------------------
# e2e
# ----------------------------------------------------------------------
@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class E2EResult:
    ws: Workspace
    stages: list[StageResult] = field(default_factory=list)
    checks: list[Check] = field(default_factory=list)
    distributions: dict[str, dict[str, int]] = field(default_factory=dict)   # 模型（修复任务加 repair/ 前缀）-> {类别: 个数}
    wrong_modes: dict[str, int] = field(default_factory=dict)                # wrong 模式下各类别的个数
    report: Optional[ReportResult] = None
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks) and all(s.ok for s in self.stages)

    def format(self) -> str:
        lines = ["e2e（mock 后端，mini 夹具）：" + ("全部通过" if self.ok else "有断言不成立")]
        for model, dist in self.distributions.items():
            total = sum(dist.values())
            lines.append(f"  {model:19s} {total:4d} 个样本  " + ", ".join(f"{k}={v}" for k, v in sorted(dist.items())))
        for c in self.checks:
            lines.append(f"  [{'通过' if c.ok else '失败'}] {c.name}" + (f"：{c.detail}" if c.detail else ""))
        lines.append(f"  用时 {self.seconds:.0f}s，输出在 {self.ws.results}")
        return "\n".join(lines)


def run_e2e(work_dir: Optional[Path] = None, *, jobs: Optional[int] = None, seed: int = DEFAULT_SEED,
            levels: Optional[Sequence[str]] = None, sandbox: Any = None, fresh: bool = True,
            fixture: Optional[Path] = None) -> E2EResult:
    """mini 夹具上的全链路自测：L0–L3 × oracle / broken / wrong 三种 MockLLM 模式，并断言（设计文档 §4.15、§6.3，v0.2 口径）。

    断言：

    * oracle 模式 100% pass：混淆后的原函数文本经「抽取 → 替换 → 沙箱编译与测试」原样回来，必须全部通过；
    * broken 模式 100% compile_syntax；
    * wrong 模式：没有一个样本通过，并且每个样本的类别恰好等于 stub-kill 记录的杀死方式
      （``targets-mini.jsonl`` 的 ``stub_kill_mode``）。wrong 候选与 stub-kill 的桩是同一种东西（返回零值 / 空函数体），
      targets.stub_verdict 与 runner.evaluate + attribute 两条路径对同一个函数必须给出同一个结论。
      注意设计文档 v0.1 写的是「wrong 模式 100% test_fail」，v0.2 改成了现在的口径：绝大多数目标是 test_fail，
      个别目标的桩会让后面的测试代码读到未初始化的状态而崩溃（runtime_crash），那是 stub-kill 已经记下的事实，不是归因出错。
      两条路径共用同一份崩溃信号集合（``attribute.CRASH_SIGNALS``），所以这项交叉验证检查的是桩的构造与替换、沙箱运行、
      阶段判定是否一致，信号分类规则本身不在检验范围内（由 tests/test_attribute.py 逐条覆盖）；
    * 每个级别的每个目标都出了题（没有被 oracle gate 作废），副本 verify 与 oracle gate 全过；
    * 缺陷修复任务（mutate → generate --task repair → evaluate --task repair）同样跑 L0–L3，模式多三个：
      oracle 模式 100% pass 且 restored（候选与该级原函数规范化后相同）；unchanged 模式（原样返回变异体）每个样本都不通过、
      stage 为 test、unchanged 为真，归因与变异体的 kill_mode 一致的比例不低于 ``E2E_MIN_SAME_MODE``；reformat 模式（原函数换排版、
      加注释）100% pass、restored 为真、unchanged 为假（规范化比较确实只忽略排版与注释）；touched 模式（变异体加一条无害语句）每个样本
      都不通过、stage 为 test、unchanged 与 restored 都为假（缺陷还在，改了别处不算「什么也没改」）；broken 模式 100% compile_syntax；
      wrong 模式每个样本都不通过且类别等于 stub-kill 记录；每个被杀死了变异体的（目标, 级别）都出了题，出题率不低于 90%；
      ``check-mutants --oracle`` 回放：每个变异体都仍被杀死、没有一个通过，去注释的原函数都通过；
    * 再把所有阶段重跑一遍，必须全部「跳过」（幂等）；
    * 报告文件生成，且没有输入哈希过期的记录。
    """
    t_start = time.monotonic()
    fixture = Path(fixture) if fixture else MINI_FIXTURE
    base = Path(work_dir) if work_dir else PROJECT_ROOT / "tmp" / "e2e"
    # 只动自己建的目录：非空且没有标记文件的目录可能是别人的，既不清空也不往里写
    if base.exists() and any(base.iterdir()) and not (base / E2E_MARKER).exists():
        raise StageError(f"{base} 非空且不是 e2e 的工作目录（没有 {E2E_MARKER}），拒绝使用；请换一个目录")
    if fresh and base.exists():
        shutil.rmtree(base)
    ws = Workspace.under(base).ensure()
    (base / E2E_MARKER).write_text("bench e2e 的工作目录，可以整个删除\n", encoding="utf-8")
    spec = corpus.load_spec_file(fixture / "spec.json")
    if sandbox is None:
        from bench.sandbox import DockerSandbox
        sandbox = DockerSandbox()
    ctx = Context(ws=ws, sandbox=sandbox, jobs=jobs or default_jobs(), seed=seed, allow_mock=True)
    levels_l = list(levels) if levels else list(LEVELS)
    res = E2EResult(ws=ws)

    def run_stages(verbose: bool) -> list[StageResult]:
        out: list[StageResult] = []

        def keep(items: Any) -> None:
            for r in (items if isinstance(items, list) else [items]):
                out.append(r)
                if verbose:
                    print(r.line(), flush=True)

        keep(stage_prepare(ctx, spec, local_src=fixture))
        keep(stage_coverage(ctx, spec))
        keep(stage_targets(ctx, spec))
        keep(stage_obfuscate(ctx, spec, levels_l))
        keep(stage_verify(ctx, spec, levels_l))
        keep(stage_describe(ctx, spec, build_llm("mock-oracle", ws), model_name="mock-oracle"))
        keep(stage_generate(ctx, [spec], E2E_MODELS, levels_l))
        keep(stage_evaluate(ctx, [spec], E2E_MODELS, levels_l, tag=E2E_TAG))
        keep(stage_mutate(ctx, spec, levels_l))
        keep(stage_generate(ctx, [spec], E2E_REPAIR_MODELS, levels_l, task="repair"))
        keep(stage_evaluate(ctx, [spec], E2E_REPAIR_MODELS, levels_l, tag=E2E_TAG, task="repair"))
        return out

    res.stages = run_stages(verbose=True)
    check = stage_check_mutants(ctx, spec, levels_l, with_oracle=True)     # 回放不是幂等阶段（每次都重新跑沙箱），不放进 run_stages
    print(check.line(), flush=True)
    res.stages.append(check)
    # 幂等：再来一遍，全部应当跳过（没有新的输入，就不会再调用模型或沙箱）
    again = run_stages(verbose=False)
    not_skipped = [f"{r.stage}:{r.corpus}" for r in again if not r.skipped]
    res.checks.append(Check("幂等：重跑所有阶段全部跳过", not not_skipped, "、".join(not_skipped)))

    targets = load_target_records(ws, spec.name)
    kill_mode = {t["target_id"]: t.get("stub_kill_mode") for t in targets}
    records = read_jsonl(ws.results_file(E2E_TAG))
    n_expected = len(targets) * len(levels_l)
    res.checks.append(Check("入选目标全部被 stub-kill 杀死", bool(targets) and all(t.get("stub_killed") for t in targets),
                            f"{len(targets)} 个目标"))
    res.checks.append(Check("每个级别的每个目标都出了题（没有作废）",
                            all(sum(1 for r in records if r["model"] == m) == n_expected for m in E2E_MODELS),
                            f"每个模型应有 {n_expected} 条记录"))
    for lv in levels_l:
        obf = read_json(ws.obf_file(spec.name, lv))
        obf = obf if isinstance(obf, dict) else None
        gate = (obf or {}).get("oracle_gate") or {}
        res.checks.append(Check(f"{lv} 副本 verify 与 oracle gate 全过",
                                bool(obf and obf.get("ok")) and gate.get("passed") == gate.get("checked") == len(targets),
                                f"gate {gate.get('passed')}/{gate.get('checked')}"))

    for m in E2E_MODELS:
        res.distributions[m] = dict(Counter(r["category"] for r in records if r["model"] == m))
    oracle = [r for r in records if r["model"] == "mock-oracle"]
    broken = [r for r in records if r["model"] == "mock-broken"]
    wrong = [r for r in records if r["model"] == "mock-wrong"]
    res.checks.append(Check("oracle 100% pass", bool(oracle) and all(r["category"] == "pass" for r in oracle),
                            f"{sum(r['category'] == 'pass' for r in oracle)}/{len(oracle)}"))
    res.checks.append(Check("broken 100% compile_syntax",
                            bool(broken) and all(r["category"] == "compile_syntax" for r in broken),
                            f"{sum(r['category'] == 'compile_syntax' for r in broken)}/{len(broken)}"))
    mismatched = [r["task_id"] for r in wrong
                  if r["stage"] != "test" or r["category"] != kill_mode.get(r["target_id"])]
    res.wrong_modes = dict(Counter(r["category"] for r in wrong))
    n_plain = sum(r["category"] == "test_fail" for r in wrong)
    others = ", ".join(f"{k}={v}" for k, v in sorted(res.wrong_modes.items()) if k != "test_fail")
    detail = f"test_fail {n_plain}/{len(wrong)}" + (f"，其余 {others}" if others else "")
    if mismatched:
        detail += f"；不一致: {mismatched[:3]}"
    res.checks.append(Check("wrong 全部在测试阶段失败，且类别与 stub-kill 记录一致", bool(wrong) and not mismatched, detail))

    _check_repair(res, ws, spec.name, levels_l)
    res.report = build_report(ws, E2E_TAG, specs={spec.name: spec})
    res.checks.append(Check("报告文件已生成且没有过期记录",
                            {"md", "csv", "svg"} <= set(res.report.paths) and res.report.n_stale == 0
                            and all(p.stat().st_size > 0 for p in res.report.paths.values()),
                            f"{res.report.n_used}/{res.report.n_records} 条纳入统计"))
    res.seconds = time.monotonic() - t_start
    return res


def _check_repair(res: E2EResult, ws: Workspace, corpus_name: str, levels: Sequence[str]) -> None:
    """缺陷修复任务的 e2e 断言（见 run_e2e 的文档）。"""
    mutants = [r for lv in levels for r in read_jsonl(ws.mutants_file(corpus_name, lv))]
    n_pairs = len(load_target_records(ws, corpus_name)) * len(levels)
    records = read_jsonl(ws.repair_results_file(E2E_TAG))
    by_model = {m: [r for r in records if r["model"] == m] for m in E2E_REPAIR_MODELS}
    for m, rows in by_model.items():
        res.distributions[f"repair/{m}"] = dict(Counter(r["category"] for r in rows))
    res.checks.append(Check("repair：每个杀死了变异体的（目标, 级别）都出了题",
                            bool(mutants) and all(len(rows) == len(mutants) for rows in by_model.values()),
                            f"每个模型应有 {len(mutants)} 条记录"))
    res.checks.append(Check("repair：出题率不低于 90%（有变异体的（目标, 级别）占比）",
                            len(mutants) >= 0.9 * n_pairs, f"{len(mutants)}/{n_pairs}"))
    oracle, broken = by_model["mock-oracle"], by_model["mock-broken"]
    wrong, unchanged = by_model["mock-wrong"], by_model["mock-unchanged"]
    reformat, touched = by_model["mock-reformat"], by_model["mock-touched"]
    res.checks.append(Check("repair：oracle 100% pass 且 restored",
                            bool(oracle) and all(r["category"] == "pass" and r["restored"] and not r["unchanged"]
                                                 for r in oracle),
                            f"{sum(r['category'] == 'pass' for r in oracle)}/{len(oracle)}"))
    same_mode = sum(r["category"] == r["kill_mode"] for r in unchanged)
    res.checks.append(Check(
        "repair：unchanged 100% 不通过（stage 为 test、归因属于杀死方式）且 unchanged=True",
        bool(unchanged) and all(r["category"] != "pass" and r["stage"] == "test" and r["category"] in KILL_MODES
                                and r["unchanged"] and not r["restored"] for r in unchanged),
        f"{len(unchanged)} 条，归因与变异体的 kill_mode 一致 {same_mode} 条"))
    res.checks.append(Check(
        f"repair：unchanged 的归因与变异体的 kill_mode 一致的比例不低于 {E2E_MIN_SAME_MODE:.0%}",
        bool(unchanged) and same_mode >= E2E_MIN_SAME_MODE * len(unchanged), f"{same_mode}/{len(unchanged)}"))
    res.checks.append(Check(
        "repair：reformat（原函数换排版加注释）100% pass 且 restored、unchanged 为假",
        bool(reformat) and all(r["category"] == "pass" and r["restored"] and not r["unchanged"] for r in reformat),
        f"{sum(r['category'] == 'pass' for r in reformat)}/{len(reformat)}"))
    res.checks.append(Check(
        "repair：touched（变异体加一条无害语句）100% 不通过（stage 为 test），unchanged 与 restored 都为假",
        bool(touched) and all(r["category"] != "pass" and r["stage"] == "test" and r["category"] in KILL_MODES
                              and not r["unchanged"] and not r["restored"] for r in touched),
        f"{len(touched)} 条，{sum(r['category'] != 'pass' for r in touched)} 条不通过"))
    res.checks.append(Check("repair：broken 100% compile_syntax",
                            bool(broken) and all(r["category"] == "compile_syntax" for r in broken),
                            f"{sum(r['category'] == 'compile_syntax' for r in broken)}/{len(broken)}"))
    bad = [r["task_id"] for r in wrong if r["stage"] != "test" or r["category"] != r.get("stub_kill_mode")]
    res.checks.append(Check("repair：wrong 全部在测试阶段失败，且类别与 stub-kill 记录一致", bool(wrong) and not bad,
                            f"{len(wrong)} 条" + (f"；不一致: {bad[:3]}" if bad else "")))
    check = read_json(ws.mutants_check_file(corpus_name)) or {}
    total = check.get("total") or {}
    res.checks.append(Check(
        "repair：check-mutants 回放：变异体都仍被杀死、没有一个通过，去注释的原函数都通过",
        bool(total.get("replayed")) and total.get("replayed") == total.get("killed") == len(mutants)
        and not total.get("passed") and not total.get("oracle_failed"),
        f"回放 {total.get('replayed', 0)}，仍被杀死 {total.get('killed', 0)}，杀死方式一致 {total.get('same_mode', 0)}"))


def _cmd_e2e(args: argparse.Namespace, ctx: Context, specs: list[Any]) -> bool:
    sandbox = _make_sandbox(args)
    base = Path(args.workspace) if args.workspace else None
    result = run_e2e(base, jobs=args.jobs, seed=args.seed, levels=args.levels, sandbox=sandbox, fresh=not args.keep)
    print(result.format(), flush=True)
    return result.ok


# ----------------------------------------------------------------------
# 入口
# ----------------------------------------------------------------------
def _specs_for(args: argparse.Namespace) -> list[Any]:
    """命令要用的语料 spec。e2e 自带夹具；report 只拿 spec 多校验一层摘要，没指定 --corpus 时找不到 spec 也能出报告。"""
    if args.cmd in _NO_SPEC:
        return []
    if args.cmd == "report" and not args.corpus:
        try:
            return _resolve_specs(None)
        except (StageError, FileNotFoundError, corpus.SpecError) as exc:
            log.warning("读不到语料 spec（%s），报告不校验 spec 摘要", exc)
            return []
    return _resolve_specs(args.corpus)


_COMMANDS = {"prepare": _cmd_prepare, "coverage": _cmd_coverage, "targets": _cmd_targets, "obfuscate": _cmd_obfuscate,
             "verify": _cmd_verify, "mutate": _cmd_mutate, "check-mutants": _cmd_check_mutants, "describe": _cmd_describe,
             "generate": _cmd_generate, "evaluate": _cmd_evaluate, "report": _cmd_report, "e2e": _cmd_e2e}
#: 不需要读语料 spec、也不预先建工作区目录的命令（e2e 自己管理工作目录）
_NO_SPEC = {"e2e"}


def _save_traceback(args: argparse.Namespace, exc: BaseException) -> Optional[Path]:
    """把未预期异常的完整堆栈追加写进工作区 results/ 下的 bench-error.log（长跑中途失败事后要能排查）。"""
    import traceback
    try:
        ws = _workspace(args, create=True)
        path = ws.results / "bench-error.log"
        with path.open("a", encoding="utf-8") as f:
            f.write(f"=== {now_iso()} python3 -m bench {args.cmd}\n")
            f.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)) + "\n")
        return path
    except OSError:
        return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    try:
        ws = _workspace(args, create=args.cmd not in _NO_SPEC)
        ctx = Context(ws=ws, jobs=max(1, args.jobs), seed=args.seed, force=args.force,
                      unsafe_debug=args.unsafe_debug, allow_mock=args.unsafe_debug)
        ok = _COMMANDS[args.cmd](args, ctx, _specs_for(args))
    except (StageError, FileNotFoundError, corpus.SpecError, LLMError, ValueError) as exc:
        if args.verbose:
            log.exception("命令失败")
        print(f"错误：{exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("已中断（已落盘的产出可以续跑）", file=sys.stderr)
        return 130
    except Exception as exc:                  # 沙箱不可用、libclang 解析失败等运行期故障：给一行说明，堆栈落进日志文件
        if args.verbose:
            log.exception("命令失败")
        where = _save_traceback(args, exc)
        print(f"运行失败：{type(exc).__name__}: {exc}（完整堆栈{'在 ' + str(where) if where else '加 -v 查看'}；"
              "已落盘的产出可以续跑）", file=sys.stderr)
        return 1
    return 0 if ok else 1
