"""流水线阶段测试共用的夹具与假件（test_e2e_mock.py、test_integrity.py、test_repair_pipeline.py 共用）。

* ``planned`` 夹具：一个有 4 个目标、L0 与 L1 副本（未跑 verify）的工作区，targets 记录与 oracle gate 按流水线真实的格式手工写好；
* ``Recorder``：记录每次调用参数的假模型；``CountingSandbox``：数沙箱运行次数，可以伪造指定文件的语法错误；
* ``FakeKillSandbox``：不需要 docker 的假沙箱，把候选放回副本后的 .c 文件按「有没有改动」判成通过 / 测试失败 / 语法错误，
  缺陷修复任务的 mutate / generate / evaluate 阶段逻辑靠它在没有 docker 的环境里也能测。
"""
from __future__ import annotations

import re
import threading
from pathlib import Path

from bench import corpus, pipeline, store
from bench.extract import extract_functions, strip_c_comments
from bench.llm import LLM, MOCK_DESCRIPTION, Reply
from bench.obfuscate import obfuscate
from bench.pipeline import Context
from bench.sandbox import DockerSandbox, RunResult
from bench.store import SCHEMA_VERSION, Workspace
from tests.helpers import MINI

__all__ = ["PICKED", "Recorder", "CountingSandbox", "FakeKillSandbox", "write_targets", "write_gate", "add_level",
           "describe_all", "make_planned_workspace"]

PICKED = ("calc_gcd", "calc_median3", "array_find", "parse_digits")


class Recorder(LLM):
    """记录每次调用参数的假模型（描述与生成都返回合格的固定文本）。"""

    def __init__(self, model: str = "recorder") -> None:
        self.model = model
        self.calls: list[dict] = []
        self._lock = threading.Lock()

    def chat(self, messages, temperature=0.0, max_tokens=4096, seed=None, meta=None) -> Reply:
        meta = dict(meta or {})
        with self._lock:
            self.calls.append({"temperature": temperature, "seed": seed, "max_tokens": max_tokens, "meta": meta})
        text = MOCK_DESCRIPTION if meta.get("purpose") == "describe" else "```c\nint f(void) { return 0; }\n```"
        return Reply(text=text, model=self.model, prompt_tokens=10, completion_tokens=5,
                     raw={"choices": [{"finish_reason": "stop"}]})


def write_targets(ws: Workspace, spec, targets, *, stub_killed=True, kill_mode="test_fail", **filter_kw) -> None:
    """按 stage_targets 的格式写 targets 记录和摘要（摘要里的源码树哈希是出题时核对偏移用的）。"""
    recs = [{**t.to_dict(), "exec_count": 5, "stub_killed": stub_killed, "stub_kill_mode": kill_mode if stub_killed else None,
             "schema_version": SCHEMA_VERSION} for t in targets]
    store.write_jsonl(ws.targets_file(spec.name), recs)
    flt = {"require_exec": True, "require_stub_kill": stub_killed is not None, **filter_kw}
    store.write_json(ws.targets_summary_file(spec.name), {
        "schema_version": SCHEMA_VERSION, "corpus": spec.name, "selected": len(recs), "excluded": {}, "filter": flt,
        "inputs": {"tree": pipeline._tree16(ws.pristine(spec.name))}, "stub_kill_modes": {kill_mode: len(recs)}})


def write_gate(ws: Workspace, name: str, level: str, targets, failed=(), copy_invalid=False) -> None:
    """按 stage_verify 的格式把 oracle gate 写进 obf 记录（inputs 对着当前的副本与目标集，出题时会重新核对）。"""
    obf = store.read_json(ws.obf_file(name, level))
    ids = sorted(t.target_id for t in targets)
    failed_ids = {f["target_id"] for f in failed}
    inputs = {"obf": obf["input_hash"], "tree": obf["tree_hash"], "targets": pipeline._target_ids_hash(targets)}
    obf["oracle_gate"] = {"checked": len(ids), "passed": len(ids) - len(failed_ids), "failed": list(failed),
                          "copy_invalid": copy_invalid, "checked_ids": ids,
                          "passed_ids": [i for i in ids if i not in failed_ids],
                          "inputs": inputs, "input_hash": store.digest(inputs)}
    store.write_json(ws.obf_file(name, level), obf)


def describe_all(ctx, spec) -> None:
    pipeline.stage_describe(ctx, spec, Recorder("desc"), model_name="desc")


def add_level(ws: Workspace, spec, targets, level: str) -> None:
    """给 planned 工作区再补一个级别的副本（未跑 verify）和手工写好的 gate。"""
    obfuscate(spec, ws.pristine("mini"), level, 3, ws.obf_root("mini", level), verify=False, results_dir=ws.results)
    write_gate(ws, "mini", level, targets)


class CountingSandbox(DockerSandbox):
    """数沙箱运行次数；fail_syntax 里列出的文件名，语法检查直接伪造一个 gcc 错误（模拟「这个文件上的 oracle 过不了」）。"""

    def __init__(self, fail_syntax: tuple[str, ...] = ()) -> None:
        super().__init__()
        self.runs = 0
        self.fail_syntax = fail_syntax
        self._count_lock = threading.Lock()

    def run(self, workdir, cmd, env=None, timeout_s=None):
        with self._count_lock:
            self.runs += 1
        if "-fsyntax-only" in cmd and any(f in cmd for f in self.fail_syntax):
            from bench.sandbox import RunResult
            return RunResult(rc=1, stdout="", stderr='[{"kind": "error", "message": "boom: injected", "locations": []}]',
                             duration_s=0.0, timed_out=False, signal=None)
        return super().run(workdir, cmd, env=env, timeout_s=timeout_s)




_GCD_PLAIN = "int calc_gcd(int a, int b)\n{\n    if (b == 0) {\n        return a < 0 ? -a : a;\n    }\n"
_GCD_COMMENTED = ("int calc_gcd(int a, int b)\n{\n    /* calc_gcd: Euclid's algorithm */\n    if (b == 0) {\n"
                  "        return a < 0 ? -a : a; // abs of a, the calc_gcd base case\n    }\n")


def make_planned_workspace(scratch, mini_spec, *, comments: bool = False):
    """一个有 4 个目标、L0 与 L1 副本（未跑 verify）的工作区，targets 记录与 oracle gate 按流水线的格式手工写好（全过）。
    返回 (Context, Workspace, 目标列表)。conftest.py 的 ``planned`` 夹具调用它。

    ``comments=True``：原始树里 ``calc_gcd`` 的函数体加了一条块注释和一条行注释（注释里写着函数自己的名字），
    副本和目标都从改过的原始树来。缺陷修复任务要求提示词里没有注释，靠它测去注释。"""
    ws = Workspace.under(scratch).ensure()
    corpus.materialize(MINI, ws.pristine("mini"))
    if comments:
        calc = ws.pristine("mini") / "calc.c"
        text = calc.read_text(encoding="utf-8")
        assert text.count(_GCD_PLAIN) == 1
        calc.write_text(text.replace(_GCD_PLAIN, _GCD_COMMENTED), encoding="utf-8")
    targets = [t for t in extract_functions(mini_spec, ws.pristine("mini")) if t.name in PICKED]
    assert len(targets) == len(PICKED)
    write_targets(ws, mini_spec, targets)
    for lv in ("L0", "L1"):
        obfuscate(mini_spec, ws.pristine("mini"), lv, 3, ws.obf_root("mini", lv), verify=False, results_dir=ws.results)
        write_gate(ws, "mini", lv, targets)
    return Context(ws=ws, sandbox=None, jobs=2, seed=3), ws, targets


_SYNTAX_ERROR = '[{"kind": "error", "message": "expected \';\' before \'}\' token", "locations": []}]'


def _normalized(text: str) -> str:
    """忽略注释和空白后的源码：判断「候选放回去以后，文件有没有真的改动」。"""
    return re.sub(r"\s+", "", strip_c_comments(text))


class FakeKillSandbox:
    """假沙箱（不需要 docker）：只看候选放回副本之后工作目录里的 .c 文件。

    ``originals`` 是各级副本里每个文件的原文（忽略注释与空白）；文件内容与其中之一相同就算「没有改动」。
    判定：全部文件都没有改动 → 通过（原函数、去了注释的原函数）；有改动且括号 / 花括号不配对 → 语法检查失败；
    有改动的其余情况（变异体、零值桩）→ 测试失败（rc=1，没有信号）。
    ``reject`` 给了就先用它判：``reject(文件名 -> 文本)`` 为真时一律测试失败（用来模拟「去注释的原函数通不过测试」）。
    """

    def __init__(self, ws: Workspace, corpus_name: str, levels=("L0", "L1"), reject=None) -> None:
        self.originals: dict[str, set[str]] = {}
        for lv in levels:
            for f in sorted(ws.obf_root(corpus_name, lv).glob("*.c")):
                self.originals.setdefault(f.name, set()).add(_normalized(f.read_text(encoding="utf-8")))
        self.reject = reject
        self.runs = 0
        self._lock = threading.Lock()

    def identity(self) -> dict:
        return {"class": "FakeKillSandbox", "image": "fake", "image_id": "sha256:fake", "cpus": 1.0}

    def run(self, workdir, cmd, env=None, timeout_s=None) -> RunResult:
        with self._lock:
            self.runs += 1
        texts = {f.name: f.read_text(encoding="utf-8") for f in sorted(Path(workdir).glob("*.c"))}
        changed = [n for n, t in texts.items() if _normalized(t) not in self.originals.get(n, set())]
        ok = dict(rc=0, stdout="", stderr="[]", duration_s=0.0, timed_out=False, signal=None)
        unbalanced = any(texts[n].count("{") != texts[n].count("}") or texts[n].count("(") != texts[n].count(")")
                         for n in changed)
        if "-fsyntax-only" in cmd:
            return RunResult(**{**ok, "rc": 1, "stderr": _SYNTAX_ERROR}) if unbalanced else RunResult(**ok)
        if changed or (self.reject is not None and self.reject(texts)):
            return RunResult(**{**ok, "rc": 1, "stdout": "FAIL\n"})
        return RunResult(**{**ok, "stdout": "ok\n"})
