"""流水线阶段测试共用的夹具与假件（test_e2e_mock.py 与 test_integrity.py 共用）。

* ``planned`` 夹具：一个有 4 个目标、L0 与 L1 副本（未跑 verify）的工作区，targets 记录与 oracle gate 按流水线真实的格式手工写好；
* ``Recorder``：记录每次调用参数的假模型；``CountingSandbox``：数沙箱运行次数，可以伪造指定文件的语法错误。
"""
from __future__ import annotations

import threading

from bench import corpus, pipeline, store
from bench.extract import extract_functions
from bench.llm import LLM, MOCK_DESCRIPTION, Reply
from bench.obfuscate import obfuscate
from bench.pipeline import Context
from bench.sandbox import DockerSandbox
from bench.store import SCHEMA_VERSION, Workspace
from tests.helpers import MINI

__all__ = ["PICKED", "Recorder", "CountingSandbox", "write_targets", "write_gate", "add_level", "describe_all",
           "make_planned_workspace"]

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




def make_planned_workspace(scratch, mini_spec):
    """一个有 4 个目标、L0 与 L1 副本（未跑 verify）的工作区，targets 记录与 oracle gate 按流水线的格式手工写好（全过）。
    返回 (Context, Workspace, 目标列表)。conftest.py 的 ``planned`` 夹具调用它。"""
    ws = Workspace.under(scratch).ensure()
    corpus.materialize(MINI, ws.pristine("mini"))
    targets = [t for t in extract_functions(mini_spec, MINI) if t.name in PICKED]
    assert len(targets) == len(PICKED)
    write_targets(ws, mini_spec, targets)
    for lv in ("L0", "L1"):
        obfuscate(mini_spec, ws.pristine("mini"), lv, 3, ws.obf_root("mini", lv), verify=False, results_dir=ws.results)
        write_gate(ws, "mini", lv, targets)
    return Context(ws=ws, sandbox=None, jobs=2, seed=3), ws, targets
