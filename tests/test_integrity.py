"""可信度措施在「使用上游产出」那一刻的核对：出题（plan_tasks）、生成、评测各阶段不能因为中间某个阶段被跳过或重跑，
就悄悄用上过期的上游产出；样本不能悄悄从分母里消失；放宽措施的开关必须显式。

每一条对应一个以前能蒙混过关的洞：

* oracle gate 对应的目标集 / 副本变了（重跑了 targets 或 obfuscate 而没重跑 verify）；
* 功能规格的输入哈希、泄漏检查器配置、mock 模型写的占位描述；
* 没做 stub-kill 的目标；
* 生成阶段的输入哈希漏掉模型实例配置（思考开关、端点、实际模型名）；
* 生成调用失败的样本静默从分母里消失，401 / 402 / 403 还逐个样本重试；
* 候选里写 ``exit(0)`` 骗过只看退出码的语料。

需要 docker 的部分用 ``needs_docker`` 标记（缺 docker 时跳过，``BENCH_REQUIRE_DOCKER=1`` 时失败）。
"""
from __future__ import annotations

import dataclasses

import pytest

from bench import corpus, pipeline, report, store
from bench.extract import extract_functions
from bench.llm import LLMError, LLMHTTPError, MissingAPIKeyError, Reply
from bench.pipeline import Context, StageError
from bench.prompt import build_messages
from bench.store import Workspace
from tests.helpers import MINI, needs_docker
from tests.pipeline_support import CountingSandbox, Recorder, describe_all, write_gate, write_targets


# ----------------------------------------------------------------------
# plan_tasks：oracle gate 在使用端核对
# ----------------------------------------------------------------------
class TestGateIsVerifiedWhereItIsUsed:
    def test_targets_changed_after_verify_is_refused(self, planned, mini_spec):
        """gate 只检查了旧目标集；换了目标集（重跑 targets 换种子 / 名额）直接 generate，以前照常为新目标出题。"""
        ctx, ws, targets = planned
        describe_all(ctx, mini_spec)
        extra = [t for t in extract_functions(mini_spec, MINI) if t.name in ("point_make", "rect_area")]
        write_targets(ws, mini_spec, targets + extra)               # 目标集变了，gate 没重跑
        with pytest.raises(StageError, match="oracle gate 是对着另一份目标集跑的"):
            pipeline.plan_tasks(ctx, mini_spec, ["L0"])
        # 重新写一份对得上的 gate 才放行，并且新目标也在 passed_ids 里
        for lv in ("L0", "L1"):
            write_gate(ws, "mini", lv, targets + extra)
        tasks, voids = pipeline.plan_tasks(ctx, mini_spec, ["L0"])
        assert len(tasks) == 4 and voids == {"L0": {"no_description": 2}}      # 新目标还没有功能规格，照常作废并计数

    def test_obfuscated_copy_regenerated_after_verify_is_refused(self, planned, mini_spec):
        ctx, ws, targets = planned
        describe_all(ctx, mini_spec)
        pipeline.obfuscate(mini_spec, ws.pristine("mini"), "L1", 99, ws.obf_root("mini", "L1"), verify=False,
                           results_dir=ws.results, force=True)          # 换种子重做副本，obf 记录的输入哈希变了，gate 没跟着变
        data = store.read_json(ws.obf_file("mini", "L1"))
        data["oracle_gate"] = store.read_json(ws.obf_file("mini", "L0"))["oracle_gate"]   # 沿用别的副本的 gate
        store.write_json(ws.obf_file("mini", "L1"), data)
        with pytest.raises(StageError, match="oracle gate 是对着另一份"):
            pipeline.plan_tasks(ctx, mini_spec, ["L1"])

    def test_target_the_gate_never_looked_at_is_voided_as_unchecked(self, planned, mini_spec):
        ctx, ws, targets = planned
        describe_all(ctx, mini_spec)
        data = store.read_json(ws.obf_file("mini", "L0"))
        data["oracle_gate"]["passed_ids"] = [i for i in data["oracle_gate"]["passed_ids"] if not i.endswith(":array_find")]
        store.write_json(ws.obf_file("mini", "L0"), data)
        tasks, voids = pipeline.plan_tasks(ctx, mini_spec, ["L0"])
        assert voids["L0"] == {"oracle_unchecked": 1} and len(tasks) == 3

    def test_targets_recorded_for_another_source_tree_are_refused(self, planned, mini_spec):
        """targets 记录里的字节偏移是对着当时的源码树算的；源码变了而没重跑 targets，以前会切出错位的函数文本。"""
        ctx, ws, _ = planned
        describe_all(ctx, mini_spec)
        calc = ws.pristine("mini") / "calc.c"
        calc.write_text("/* 向前挪了几个字节 */\n" + calc.read_text(encoding="utf-8"), encoding="utf-8")
        with pytest.raises(StageError, match="重跑 targets"):
            pipeline.plan_tasks(ctx, mini_spec, ["L0"])
        with pytest.raises(StageError, match="重跑 targets"):
            pipeline.stage_describe(ctx, mini_spec, Recorder("desc"), model_name="desc")

    def test_read_function_refuses_a_misaligned_slice(self, planned):
        from bench.runner import SpliceError, read_function
        ctx, ws, targets = planned
        t = dataclasses.replace(targets[0], start=0, end=5)                # 文件开头的几个字节里不可能有函数名
        with pytest.raises(SpliceError):
            read_function(ws.pristine("mini"), t)


# ----------------------------------------------------------------------
# plan_tasks：stub-kill 记录
# ----------------------------------------------------------------------
class TestStubKillIsNotOptional:
    def test_targets_without_a_stub_kill_verdict_are_voided(self, planned, mini_spec):
        ctx, ws, targets = planned
        write_targets(ws, mini_spec, targets, stub_killed=None)       # --no-stub-kill 选出来的目标：stub_killed 是 None
        for lv in ("L0", "L1"):
            write_gate(ws, "mini", lv, targets)
        describe_all(ctx, mini_spec)
        tasks, voids = pipeline.plan_tasks(ctx, mini_spec, ["L0", "L1"])
        assert tasks == [] and voids == {"L0": {"stub_unchecked": 4}, "L1": {"stub_unchecked": 4}}

    def test_unsafe_debug_lets_them_through(self, planned, mini_spec):
        ctx, ws, targets = planned
        write_targets(ws, mini_spec, targets, stub_killed=None)
        for lv in ("L0", "L1"):
            write_gate(ws, "mini", lv, targets)
        describe_all(ctx, mini_spec)
        ctx.unsafe_debug = True
        tasks, voids = pipeline.plan_tasks(ctx, mini_spec, ["L0"])
        assert len(tasks) == 4 and voids == {"L0": {}}

    def test_a_survived_stub_is_not_a_killed_target_either(self, planned, mini_spec):
        ctx, ws, targets = planned
        recs = store.read_jsonl(ws.targets_file("mini"))
        recs[0]["stub_killed"] = False
        store.write_jsonl(ws.targets_file("mini"), recs)
        describe_all(ctx, mini_spec)
        _, voids = pipeline.plan_tasks(ctx, mini_spec, ["L0"])
        assert voids["L0"] == {"stub_unchecked": 1}

    def test_stage_targets_refuses_to_skip_stub_kill_without_the_explicit_switch(self, scratch, mini_spec):
        from bench.targets import TargetFilter
        ws = Workspace.under(scratch).ensure()
        corpus.materialize(MINI, ws.pristine("mini"))
        flt = TargetFilter(max_per_corpus=3, require_exec=False, require_stub_kill=False)
        with pytest.raises(StageError, match="unsafe-debug"):
            pipeline.stage_targets(Context(ws=ws, sandbox=object(), jobs=1), mini_spec, flt)
        assert not ws.targets_file("mini").exists()

    def test_kill_mode_travels_with_the_planned_task(self, planned, mini_spec):
        ctx, ws, targets = planned
        recs = store.read_jsonl(ws.targets_file("mini"))
        recs[0]["stub_kill_mode"] = "runtime_crash"
        store.write_jsonl(ws.targets_file("mini"), recs)
        describe_all(ctx, mini_spec)
        tasks, _ = pipeline.plan_tasks(ctx, mini_spec, ["L0"])
        modes = {pt.original.target_id: pt.kill_mode for pt in tasks}
        assert modes[recs[0]["target_id"]] == "runtime_crash" and set(modes.values()) == {"runtime_crash", "test_fail"}


# ----------------------------------------------------------------------
# 功能规格：使用时重新核对
# ----------------------------------------------------------------------
class TestDescriptionsAreRevalidatedOnUse:
    def test_up_to_date_descriptions_pass(self, planned, mini_spec):
        ctx, _, _ = planned
        describe_all(ctx, mini_spec)
        tasks, voids = pipeline.plan_tasks(ctx, mini_spec, ["L0"])
        assert len(tasks) == 4 and voids == {"L0": {}}

    def test_description_made_for_other_inputs_is_stale(self, planned, mini_spec):
        ctx, ws, _ = planned
        describe_all(ctx, mini_spec)
        recs = store.read_jsonl(ws.descriptions_file)
        recs[0]["input_hash"] = "0" * 16                           # 目标函数 / 源码树 / 描述提示词 / 检查器配置变了
        store.write_jsonl(ws.descriptions_file, recs)
        tasks, voids = pipeline.plan_tasks(ctx, mini_spec, ["L0"])
        assert voids["L0"] == {"describe_stale": 1} and len(tasks) == 3

    def test_tightening_the_checker_invalidates_old_descriptions(self, planned, mini_spec):
        """检查器配置（词表、阈值、allow_words、项目标识符集合）进了描述阶段的输入哈希：收紧检查器后，旧描述要重新生成。"""
        ctx, ws, _ = planned
        llm = Recorder("desc")
        pipeline.stage_describe(ctx, mini_spec, llm, model_name="desc")
        assert len(llm.calls) == 4
        assert pipeline.stage_describe(ctx, mini_spec, llm, model_name="desc").skipped
        looser = dataclasses.replace(mini_spec, allow_words=["gcd"])      # allow_words 不进 spec 哈希，但进检查器摘要
        again = pipeline.stage_describe(Context(ws=ws, jobs=2, seed=3), looser, llm, model_name="desc")
        assert not again.skipped and len(llm.calls) == 8
        _, voids = pipeline.plan_tasks(ctx, mini_spec, ["L0"])             # 用原来的 spec 出题：描述是按另一套检查器写的
        assert voids["L0"] == {"describe_stale": 4}

    def test_a_description_that_leaks_under_the_current_checker_is_voided(self, planned, mini_spec):
        ctx, ws, _ = planned
        describe_all(ctx, mini_spec)
        recs = store.read_jsonl(ws.descriptions_file)
        target = pipeline.load_targets(ws, "mini")[0]
        recs[0]["description"] += " It behaves like CALC-GCD in spirit."          # 检查器收紧后能认出的变体写法
        store.write_jsonl(ws.descriptions_file, recs)
        tasks, voids = pipeline.plan_tasks(ctx, mini_spec, ["L0"])
        # 描述文本变了但输入哈希还对得上（输入不含描述文本）：重新跑一遍泄漏检查才发现
        assert voids["L0"].get("describe_leak") == 1 and len(tasks) == 3, (voids, target.name)

    def test_mock_descriptions_are_refused_outside_e2e(self, planned, mini_spec):
        ctx, ws, _ = planned
        with pytest.raises(StageError, match="离线自测"):
            pipeline.stage_describe(ctx, mini_spec, pipeline.build_llm("mock-oracle", ws), model_name="mock-oracle")
        assert not ws.descriptions_file.exists()
        ctx.allow_mock = True
        pipeline.stage_describe(ctx, mini_spec, pipeline.build_llm("mock-oracle", ws), model_name="mock-oracle")
        ctx.allow_mock = False
        tasks, voids = pipeline.plan_tasks(ctx, mini_spec, ["L0"])
        assert tasks == [] and voids["L0"] == {"describe_mock": 4}            # 之后在真实工作区里误用也会被拦住
        ctx.unsafe_debug = True
        assert len(pipeline.plan_tasks(ctx, mini_spec, ["L0"])[0]) == 4

    def test_description_records_carry_audit_fields(self, planned, mini_spec):
        ctx, ws, _ = planned
        describe_all(ctx, mini_spec)
        recs = store.read_jsonl(ws.descriptions_file)
        assert all(r["checker"] and isinstance(r["exempt_hits"], list) and isinstance(r["phrase_hits"], list)
                   and r["own_name_hit"] is False for r in recs)
        assert all(set(r["inputs"]) >= {"code", "tree", "model", "checker"} for r in recs)

    def test_up_to_date_descriptions_need_no_api_key(self, planned, mini_spec, monkeypatch):
        ctx, ws, _ = planned
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        pipeline.stage_describe(ctx, mini_spec, Recorder("deepseek-v4-pro"), model_name="deepseek-v4-pro")
        llm = pipeline.build_llm("deepseek-v4-pro", ws)                    # 以前构造时就要 key，哪怕根本没有要生成的描述
        assert pipeline.stage_describe(ctx, mini_spec, llm, model_name="deepseek-v4-pro").skipped


# ----------------------------------------------------------------------
# generate：输入哈希、失败样本、缺 key、缓存
# ----------------------------------------------------------------------
class ConfigurableRecorder(Recorder):
    """带可改的实例配置指纹的假模型。"""

    def __init__(self, model: str, fp: str) -> None:
        super().__init__(model)
        self.fp = fp

    def fingerprint(self) -> str:
        return self.fp


class FlakyLLM(Recorder):
    """前 fail_first 次调用失败（可指定异常），之后正常。"""

    def __init__(self, model: str, fail_first: int, exc: Exception) -> None:
        super().__init__(model)
        self.fail_first, self.exc, self.n = fail_first, exc, 0

    def chat(self, messages, temperature=0.0, max_tokens=4096, seed=None, meta=None) -> Reply:
        with self._lock:
            self.n += 1
            n = self.n
        if n <= self.fail_first:
            raise self.exc
        return super().chat(messages, temperature=temperature, max_tokens=max_tokens, seed=seed, meta=meta)


class TestGenerationInputs:
    def patch_llm(self, monkeypatch, llm):
        calls = []

        def build(name, ws, oracle=None, use_cache=True):
            calls.append(use_cache)
            return llm

        monkeypatch.setattr(pipeline, "build_llm", build)
        return calls

    def test_model_configuration_is_part_of_the_generation_hash(self, planned, mini_spec, monkeypatch):
        """换了 PRESETS 里的思考开关 / 实际模型名 / 端点之后重跑 generate，旧回复不能因为哈希相同被原样复用。"""
        ctx, ws, _ = planned
        describe_all(ctx, mini_spec)
        first = ConfigurableRecorder("rec", "thinking-off")
        self.patch_llm(monkeypatch, first)
        pipeline.stage_generate(ctx, [mini_spec], ["rec"], ["L0"])
        assert len(first.calls) == 4
        assert pipeline.stage_generate(ctx, [mini_spec], ["rec"], ["L0"])[0].skipped
        second = ConfigurableRecorder("rec", "thinking-on")
        self.patch_llm(monkeypatch, second)
        res = pipeline.stage_generate(ctx, [mini_spec], ["rec"], ["L0"])[0]
        assert not res.skipped and res.detail["calls"] == 4 and len(second.calls) == 4
        gens = store.read_jsonl(ws.generations_file("rec"))
        assert len(gens) == 4 and {g["llm_fp"] for g in gens} == {"thinking-on"}

    def test_generation_records_say_what_produced_the_reply(self, planned, mini_spec, monkeypatch):
        ctx, ws, _ = planned
        describe_all(ctx, mini_spec)

        class Tagged(ConfigurableRecorder):
            api_model = "real-api-name"

            def chat(self, messages, **kw):
                reply = super().chat(messages, **kw)
                reply.raw = {"model": "server-says", "system_fingerprint": "fp_abc", "choices": [{"finish_reason": "stop"}]}
                return reply

        self.patch_llm(monkeypatch, Tagged("rec", "cfg"))
        pipeline.stage_generate(ctx, [mini_spec], ["rec"], ["L0"])
        (g, *_) = store.read_jsonl(ws.generations_file("rec"))
        assert (g["api_model"], g["reply_model"], g["system_fingerprint"], g["llm_fp"]) == (
            "real-api-name", "server-says", "fp_abc", "cfg")
        assert (g["temperature"], g["max_tokens"], g["seed"]) == (0.0, 4096, None)

    def test_no_cache_flag_reaches_the_llm_builder(self, planned, mini_spec, monkeypatch):
        ctx, ws, _ = planned
        describe_all(ctx, mini_spec)
        calls = self.patch_llm(monkeypatch, Recorder("rec"))
        pipeline.stage_generate(ctx, [mini_spec], ["rec"], ["L0"])
        ctx.no_cache = True
        ctx.force = True
        pipeline.stage_generate(ctx, [mini_spec], ["rec"], ["L0"])
        assert calls == [True, False]

    def test_build_llm_never_needs_the_key_up_front_and_can_skip_reading_the_cache(self, scratch, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        ws = Workspace.under(scratch).ensure()
        llm = pipeline.build_llm("deepseek-flash", ws)
        assert llm.cache.read is True and pipeline.build_llm("deepseek-flash", ws, use_cache=False).cache.read is False
        with pytest.raises(MissingAPIKeyError):
            llm.check_credentials()
        assert pipeline.build_llm("mock-oracle", ws, oracle={}).model == "mock-oracle"

    def test_generate_needs_no_key_when_every_reply_is_already_there(self, planned, mini_spec, monkeypatch):
        ctx, ws, _ = planned
        describe_all(ctx, mini_spec)
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        fp = pipeline.build_llm("deepseek-flash", ws).fingerprint()
        tasks, _ = pipeline.plan_tasks(ctx, mini_spec, ["L0"])
        rows = []
        for pt in tasks:
            mh = store.digest(build_messages(pt.task))
            rows.append({"task_id": pt.task.task_id, "target_id": pt.task.target_id, "corpus": "mini", "level": "L0",
                         "model": "deepseek-flash", "sample_idx": 0, "text": "```c\nint f(void) { return 0; }\n```",
                         "messages_hash": mh, "input_hash": pipeline._gen_inputs_hash(mh, "deepseek-flash", 0.0, 4096, None, fp)})
        store.write_jsonl(ws.generations_file("deepseek-flash"), rows)
        (res,) = pipeline.stage_generate(ctx, [mini_spec], ["deepseek-flash"], ["L0"])
        assert res.skipped and res.ok                          # 以前：构造模型时就抛 MissingAPIKeyError

    def test_missing_key_surfaces_at_the_first_real_request(self, planned, mini_spec, monkeypatch):
        ctx, ws, _ = planned
        describe_all(ctx, mini_spec)
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        with pytest.raises(MissingAPIKeyError, match="DEEPSEEK_API_KEY"):
            pipeline.stage_generate(ctx, [mini_spec], ["deepseek-flash"], ["L0"])
        assert not ws.generations_file("deepseek-flash").exists()


class TestFailedCallsAreNotSilentlyDropped:
    def test_failures_are_recorded_cleared_on_success_and_make_the_stage_fail(self, planned, mini_spec, monkeypatch):
        ctx, ws, _ = planned
        ctx.jobs = 1
        describe_all(ctx, mini_spec)
        flaky = FlakyLLM("rec", 2, LLMError("调用 rec 失败，已重试 5 次仍未成功: HTTP 503"))
        monkeypatch.setattr(pipeline, "build_llm", lambda name, ws_, oracle=None, use_cache=True: flaky)
        (res,) = pipeline.stage_generate(ctx, [mini_spec], ["rec"], ["L0"])
        assert not res.ok and "2 次调用失败" in res.message and res.detail["calls"] == 2
        failures = store.read_jsonl(ws.generation_failures_file("rec"))
        assert len(failures) == 2 and all("503" in f["error"] and f["level"] == "L0" for f in failures)
        assert len(store.read_jsonl(ws.generations_file("rec"))) == 2
        (again,) = pipeline.stage_generate(ctx, [mini_spec], ["rec"], ["L0"])
        assert again.ok and again.detail["calls"] == 2
        assert store.read_jsonl(ws.generation_failures_file("rec")) == []        # 补上的样本从失败记录里消失

    def test_fatal_http_errors_abort_the_whole_stage(self, planned, mini_spec, monkeypatch):
        """401 / 402 / 403：key 无效、余额不足、没有权限，整批必然失败；以前每个样本还各打一条错误日志、白发全部请求。"""
        ctx, ws, _ = planned
        ctx.jobs = 1
        describe_all(ctx, mini_spec)
        llm = FlakyLLM("rec", 1000, LLMHTTPError(402, "余额不足"))
        monkeypatch.setattr(pipeline, "build_llm", lambda name, ws_, oracle=None, use_cache=True: llm)
        with pytest.raises(LLMHTTPError) as exc:
            pipeline.stage_generate(ctx, [mini_spec], ["rec"], ["L0", "L1"])
        assert exc.value.status == 402 and llm.n == 1                       # 第一个样本就中止，没有再发 7 个请求
        # 400 之类只影响单个样本的错误照旧逐样本失败
        llm2 = FlakyLLM("rec", 1000, LLMHTTPError(400, "提示词过长"))
        monkeypatch.setattr(pipeline, "build_llm", lambda name, ws_, oracle=None, use_cache=True: llm2)
        (res,) = pipeline.stage_generate(ctx, [mini_spec], ["rec"], ["L0"])
        assert not res.ok and llm2.n == 4

    def test_evaluate_reports_missing_samples_and_keeps_them_out_of_the_denominator_silently_no_more(
            self, planned, mini_spec, monkeypatch):
        ctx, ws, _ = planned
        ctx.sandbox = sandbox = CountingSandbox()
        ctx.jobs = 1
        describe_all(ctx, mini_spec)
        flaky = FlakyLLM("rec", 3, LLMError("HTTP 503"))
        monkeypatch.setattr(pipeline, "build_llm", lambda name, ws_, oracle=None, use_cache=True: flaky)
        pipeline.stage_generate(ctx, [mini_spec], ["rec"], ["L0", "L1"])           # 8 个任务，前 3 个调用失败
        (res,) = pipeline.stage_evaluate(ctx, [mini_spec], ["rec"], ["L0", "L1"], tag="miss")
        assert not res.ok and res.detail["missing"] == 3 and "缺 3/8" in res.message and "3 个调用失败" in res.message
        comp = store.read_json(ws.completeness_file("miss"))
        cells = {(c["model"], c["level"], c["corpus"]): c for c in comp["cells"]}
        assert sum(c["expected"] for c in cells.values()) == 8 and sum(c["missing"] for c in cells.values()) == 3
        assert sum(c["failed_calls"] for c in cells.values()) == 3 and sum(c["have"] for c in cells.values()) == 5
        # --allow-missing 放行
        ctx.allow_missing = True
        (res2,) = pipeline.stage_evaluate(ctx, [mini_spec], ["rec"], ["L0", "L1"], tag="miss")
        assert res2.ok and res2.skipped and res2.detail["missing"] == 3
        ctx.allow_missing = False
        # 补齐之后完整性记录更新，阶段通过
        monkeypatch.setattr(pipeline, "build_llm", lambda name, ws_, oracle=None, use_cache=True: Recorder("rec"))
        pipeline.stage_generate(ctx, [mini_spec], ["rec"], ["L0", "L1"])
        (res3,) = pipeline.stage_evaluate(ctx, [mini_spec], ["rec"], ["L0", "L1"], tag="miss")
        assert res3.ok and res3.detail["missing"] == 0
        comp = store.read_json(ws.completeness_file("miss"))
        assert sum(c["missing"] for c in comp["cells"]) == 0 and sum(c["expected"] for c in comp["cells"]) == 8
        assert sandbox.runs > 0


# ----------------------------------------------------------------------
# evaluate：违反硬性约束的候选不进沙箱
# ----------------------------------------------------------------------
class ExitLLM(Recorder):
    """生成阶段对每个任务都回一个函数体只有 exit(0) 的定义（描述阶段照常）。"""

    def chat(self, messages, temperature=0.0, max_tokens=4096, seed=None, meta=None) -> Reply:
        reply = super().chat(messages, temperature=temperature, max_tokens=max_tokens, seed=seed, meta=meta)
        if (meta or {}).get("purpose") == "generate":
            sig = meta["obf_signature"]
            reply.text = f"```c\n{sig}\n{{\n    exit(0);\n}}\n```"
        return reply


@needs_docker
class TestPolicyViolationsNeverReachTheSandbox:
    def test_exit_zero_candidates_fail_without_running_anything(self, planned, mini_spec, monkeypatch):
        ctx, ws, _ = planned
        ctx.sandbox = sandbox = CountingSandbox()
        describe_all(ctx, mini_spec)
        monkeypatch.setattr(pipeline, "build_llm", lambda name, ws_, oracle=None, use_cache=True: ExitLLM("exit-llm"))
        pipeline.stage_generate(ctx, [mini_spec], ["exit-llm"], ["L0", "L1"])
        before = sandbox.runs
        (res,) = pipeline.stage_evaluate(ctx, [mini_spec], ["exit-llm"], ["L0", "L1"], tag="pol")
        assert res.ok and res.detail["categories"] == {"policy_violation": 8} and res.detail["pass"] == 0
        assert sandbox.runs == before                               # 一次沙箱都没跑
        recs = store.read_jsonl(ws.results_file("pol"))
        assert len(recs) == 8 and all(r["stage"] == "policy" and r["category"] == "policy_violation" for r in recs)
        assert all("call:exit" in r["violations"] and "policy_violation" in r["issues"] for r in recs)
        assert all(r["detail"].startswith("forbidden construct") for r in recs)
        # 报告里：不算编译通过，通过率 0，归因分布里有这一类
        out = report.build_report(ws, "pol", formats=("md",))
        assert out.n_used == 8 and out.n_stale == 0, out.stale_reasons
        md = out.paths["md"].read_text(encoding="utf-8")
        assert "policy_violation" in md and "违反硬性约束" in md

    def test_oracle_candidates_with_directives_are_allowed_when_the_original_has_them(self, planned, mini_spec):
        ctx, ws, _ = planned
        describe_all(ctx, mini_spec)
        tasks, _ = pipeline.plan_tasks(ctx, mini_spec, ["L0"])
        assert not any(pt.allow_directives for pt in tasks)           # mini 的原函数里没有 #define / #pragma
