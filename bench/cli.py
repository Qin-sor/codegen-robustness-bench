"""命令行（设计文档 §4.15）：``python3 -m bench <cmd> [选项]``。

子命令与流水线阶段一一对应（实现都在 :mod:`bench.pipeline`，这里只解析参数、调用、打印结果）::

    prepare    取语料源码到 corpus/src/<name>，沙箱里确认构建与测试可用
    coverage   沙箱里带 --coverage 构建并跑测试，得到每个函数的执行次数
    targets    选目标函数：静态过滤 → 执行次数 > 0 → 按 seed 抽样 + stub-kill
    obfuscate  生成 L0–L3 混淆副本（--level 选级别）并 verify
    verify     oracle gate：用混淆后的原函数文本逐目标走一遍完整 evaluate
    describe   用 LLM 为每个目标生成不含标识符的功能规格（--model）
    generate   按（目标, 级别, 模型, 样本）出题并调模型（--model --levels --samples）
    evaluate   后处理 → 替换回副本 → 沙箱编译与测试 → 归因（--model --tag）
    report     汇总指标，输出 summary-<tag>.md / .csv 与图（--tag）
    e2e        在 tests/fixtures/mini 上用 MockLLM 一键跑通全链路并断言（--backend mock）

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
from bench.obfuscate import LEVELS
from bench.pipeline import (
    DEFAULT_SEED, Context, StageError, StageResult, build_llm, load_target_records, stage_coverage,
    stage_describe, stage_evaluate, stage_generate, stage_obfuscate, stage_prepare, stage_targets, stage_verify,
)
from bench.report import ReportResult, build_report
from bench.store import now_iso
from bench.store import PROJECT_ROOT, Workspace, read_json, read_jsonl
from bench.targets import TargetFilter

log = logging.getLogger("bench")

#: e2e 的夹具：tests/fixtures/mini（含 spec.json，语料名 mini）
MINI_FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "mini"
E2E_MODELS = ("mock-oracle", "mock-broken", "mock-wrong")
E2E_TAG = "e2e"
E2E_MARKER = ".bench-e2e"
#: describe 的默认模型：设计文档要求「最强可用模型」；deepseek-chat / deepseek-reasoner 在官方文档里有弃用通知，不再当默认
DEFAULT_DESCRIBE_MODEL = "deepseek-v4-pro"
DEFAULT_GENERATE_MODEL = "deepseek-flash"


EPILOG = """\
流水线顺序：prepare → coverage → targets → obfuscate → verify → describe → generate → evaluate → report
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

    p = add("evaluate", "评测回复：替换回副本、沙箱编译与测试、归因")
    p.add_argument("--model", nargs="+", required=True, metavar="MODEL", help="要评测的模型（generations 里已有回复）")
    p.add_argument("--levels", "--level", dest="levels", nargs="+", choices=LEVELS, metavar="LEVEL", help="级别（默认全部）")
    p.add_argument("--tag", default="run", help="结果标签，写入 results-<tag>.jsonl（默认 %(default)s）")
    p.add_argument("--allow-missing", action="store_true", help="有样本缺失（生成调用失败、没生成、回复过期）也不算失败")

    p = add("report", "汇总指标，输出 summary-<tag>.md / .csv 与图")
    p.add_argument("--tag", default="run", help="结果标签（默认 %(default)s）")
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


def _cmd_describe(args: argparse.Namespace, ctx: Context, specs: list[Any]) -> bool:
    llm = build_llm(args.model, ctx.ws)
    return _print_results([stage_describe(ctx, s, llm, model_name=args.model) for s in specs])


def _cmd_generate(args: argparse.Namespace, ctx: Context, specs: list[Any]) -> bool:
    ctx.no_cache = args.no_cache
    return _print_results(stage_generate(ctx, specs, args.model, args.levels, samples=args.samples,
                                         temperature=args.temperature, max_tokens=args.max_tokens,
                                         dry_run=args.dry_run))


def _cmd_evaluate(args: argparse.Namespace, ctx: Context, specs: list[Any]) -> bool:
    ctx.sandbox = _make_sandbox(args)
    ctx.allow_missing = args.allow_missing
    return _print_results(stage_evaluate(ctx, specs, args.model, args.levels, tag=args.tag))


def _cmd_report(args: argparse.Namespace, ctx: Context, specs: list[Any]) -> bool:
    known = {s.name: s for s in specs}                          # 记录所属语料有 spec 的，多校验一层 spec 摘要
    res = build_report(ctx.ws, args.tag, specs=known, allow_stale=args.allow_stale, formats=args.formats)
    print(f"[report] {args.tag}: 记录 {res.n_records} 条，纳入统计 {res.n_used} 条，输入哈希不符 {res.n_stale} 条", flush=True)
    for reason, n in res.stale_reasons.items():
        print(f"         - {reason}: {n} 条", flush=True)
    if res.n_bad_lines:
        print(f"         结果文件有 {res.n_bad_lines} 条坏行（例如进程被杀时留下的半行），这些样本没有计入", flush=True)
    if res.n_missing:
        print(f"         缺 {res.n_missing} 个样本（生成调用失败 / 没生成 / 回复过期），表里的 n 不完整", flush=True)
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
    distributions: dict[str, dict[str, int]] = field(default_factory=dict)   # 模型 -> {类别: 个数}
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
            lines.append(f"  {model:12s} {total:4d} 个样本  " + ", ".join(f"{k}={v}" for k, v in sorted(dist.items())))
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
        return out

    res.stages = run_stages(verbose=True)
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

    res.report = build_report(ws, E2E_TAG, specs={spec.name: spec})
    res.checks.append(Check("报告文件已生成且没有过期记录",
                            {"md", "csv", "svg"} <= set(res.report.paths) and res.report.n_stale == 0
                            and all(p.stat().st_size > 0 for p in res.report.paths.values()),
                            f"{res.report.n_used}/{res.report.n_records} 条纳入统计"))
    res.seconds = time.monotonic() - t_start
    return res


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
             "verify": _cmd_verify, "describe": _cmd_describe, "generate": _cmd_generate, "evaluate": _cmd_evaluate,
             "report": _cmd_report, "e2e": _cmd_e2e}
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
