"""缺陷修复任务的流水线阶段：mutate、出题（plan_repair_tasks）、generate / evaluate 的 repair 分支、``--task`` 的取舍。

全部用假沙箱（``tests/pipeline_support.py`` 的 ``FakeKillSandbox``），不需要 docker：假沙箱按「候选放回副本之后文件有没有真的改动」
判通过 / 测试失败 / 语法错误，足以把阶段自己的逻辑测清楚——哪些（目标, 级别）出题、放弃的原因怎么记、输入哈希变了怎么失效、
上游产出过期时怎么拒绝、记录里有哪些字段。真实沙箱 + 真实评测的全链路在 ``tests/test_e2e_mock.py`` 的 e2e 里。

用的工作区是 ``planned`` 夹具的变体：mini 里挑 4 个目标、L0 与 L1 副本、手工写好的 oracle gate；``comments=True`` 时
``calc_gcd`` 的函数体里带注释（注释里写着函数自己的名字），用来测去注释。
"""
from __future__ import annotations

import dataclasses

import pytest

from bench import cli, pipeline, store
from bench.extract import strip_c_comments
from bench.mutate import OPERATORS
from bench.pipeline import Context, StageError
from bench.prompt import REPAIR_PROMPT_DIGEST, RepairTask, build_repair_messages
from bench.report import validate_records
from bench.runner import read_function
from bench.store import Workspace
from tests.pipeline_support import FakeKillSandbox, PICKED, make_planned_workspace, write_gate, write_targets

LEVELS2 = ["L0", "L1"]


@pytest.fixture()
def rp(scratch, mini_spec):
    """(Context, Workspace, 目标, spec)：带注释的 calc_gcd，假沙箱接在 ctx 上。"""
    ctx, ws, targets = make_planned_workspace(scratch, mini_spec, comments=True)
    ctx.sandbox = FakeKillSandbox(ws, "mini")
    return ctx, ws, targets, mini_spec


def mutants_of(ws: Workspace, level: str) -> dict[str, dict]:
    return {r["target_id"]: r for r in store.read_jsonl(ws.mutants_file("mini", level))}


def summary_of(ws: Workspace) -> dict:
    return store.read_json(ws.mutants_summary_file("mini"))


def by_name(targets, name):
    return next(t for t in targets if t.name == name)


# ----------------------------------------------------------------------
# mutate
# ----------------------------------------------------------------------
class TestStageMutate:
    def test_writes_one_killed_mutant_per_target_and_level(self, rp):
        ctx, ws, targets, spec = rp
        res = pipeline.stage_mutate(ctx, spec, LEVELS2)
        assert res.ok and not res.skipped and res.detail["killed"] == 8 and res.detail["abandoned"] == 0
        for lv in LEVELS2:
            recs = mutants_of(ws, lv)
            assert set(recs) == {t.target_id for t in targets}
            for r in recs.values():
                # 附录 §7 列的字段都在
                assert {"target_id", "level", "operator", "site_index", "original_text", "mutated_text", "mutant_code",
                        "kill_mode", "tried_sites", "equivalent_count", "compile_fail_count", "input_hash"} <= set(r)
                assert r["operator"] in OPERATORS and r["level"] == lv and r["outcome"] == "killed"
                assert r["kill_mode"] == "test_fail" and r["tried_sites"] == 1 and r["corpus"] == "mini"
                assert r["schema_version"] == store.SCHEMA_VERSION
        summary = summary_of(ws)
        assert summary["levels_computed"] == LEVELS2 and summary["n_targets"] == 4 and summary["max_tries"] == 12
        for lv in LEVELS2:
            ent = summary["levels"][lv]
            assert (ent["n_targets"], ent["n_searched"], ent["n_killed"], ent["n_abandoned"]) == (4, 4, 4, 0)
            assert ent["input_hash"] == mutants_of(ws, lv)[targets[0].target_id]["input_hash"]
            assert sum(ent["operators"].values()) == 4 and ent["kill_modes"] == {"test_fail": 4}
        assert summary["four_level"] == {"n_targets": 4, "n_all_levels": 0, "n_same_operator": 0, "n_same_site": 0}   # 这次只做了两级

    def test_mutant_code_is_the_function_with_exactly_the_site_changed_and_without_comments(self, rp):
        ctx, ws, targets, spec = rp
        pipeline.stage_mutate(ctx, spec, LEVELS2)
        gcd = by_name(targets, "calc_gcd")
        for lv in LEVELS2:
            rec = mutants_of(ws, lv)[gcd.target_id]
            assert "/*" not in rec["mutant_code"] and "//" not in rec["mutant_code"]
            assert rec["has_comments"] is True and rec["strip_checked"] is True
            root = ws.obf_root("mini", lv)
            name = store.read_json(ws.obf_file("mini", lv))["rename_map"].get("calc_gcd", "calc_gcd")
            located = pipeline.locate(spec, root, name, gcd.file)
            original = read_function(root, located)
            assert rec["mutant_code"] != strip_c_comments(original) and rec["original_text"] in original
        l1 = mutants_of(ws, "L1")[gcd.target_id]["mutant_code"]
        assert "calc_gcd" not in l1                   # 注释里的原名随注释一起去掉了，L1 的提示词不再泄漏它
        others = [r for tid, r in mutants_of(ws, "L1").items() if tid != gcd.target_id]
        assert all(r["has_comments"] is False and r["strip_checked"] is False for r in others)
        ent = summary_of(ws)["levels"]["L1"]
        assert (ent["n_with_comments"], ent["n_strip_checked"], ent["n_strip_failed"]) == (1, 1, 0)

    def test_l1_reuses_the_operator_chosen_on_l0(self, rp):
        ctx, ws, _, spec = rp
        pipeline.stage_mutate(ctx, spec, LEVELS2)
        l0, l1 = mutants_of(ws, "L0"), mutants_of(ws, "L1")
        assert all(l1[tid]["operator"] == l0[tid]["operator"] for tid in l0)
        assert all(r["same_operator_as_L0"] is True for r in l0.values())
        assert all(r["same_operator_as_L0"] is True for r in l1.values())
        assert summary_of(ws)["levels"]["L1"]["same_operator_as_L0"] == {"n": 4, "same": 4}

    def test_second_run_is_skipped_without_touching_the_sandbox_and_force_redoes_it(self, rp):
        ctx, ws, _, spec = rp
        pipeline.stage_mutate(ctx, spec, LEVELS2)
        runs, before = ctx.sandbox.runs, mutants_of(ws, "L1")
        again = pipeline.stage_mutate(ctx, spec, LEVELS2)
        assert again.skipped and again.detail == {"killed": 8, "abandoned": 0} and ctx.sandbox.runs == runs
        ctx.force = True
        redone = pipeline.stage_mutate(ctx, spec, LEVELS2)
        assert not redone.skipped and ctx.sandbox.runs > runs and mutants_of(ws, "L1") == before   # 同输入同结果

    def test_seed_and_max_tries_are_part_of_the_inputs(self, rp):
        ctx, ws, _, spec = rp
        pipeline.stage_mutate(ctx, spec, LEVELS2)
        assert not pipeline.stage_mutate(ctx, spec, LEVELS2, max_tries=5).skipped
        assert pipeline.stage_mutate(ctx, spec, LEVELS2, max_tries=5).skipped
        other = Context(ws=ws, sandbox=ctx.sandbox, jobs=2, seed=ctx.seed + 1)
        assert not pipeline.stage_mutate(other, spec, LEVELS2, max_tries=5).skipped
        with pytest.raises(StageError, match="max_tries"):
            pipeline.stage_mutate(ctx, spec, LEVELS2, max_tries=0)

    def test_only_stale_levels_are_redone_and_l1_needs_a_valid_l0(self, rp):
        ctx, ws, _, spec = rp
        with pytest.raises(StageError, match="L0"):
            pipeline.stage_mutate(ctx, spec, ["L1"])                  # 先要有 L0 的产出
        pipeline.stage_mutate(ctx, spec, ["L0"])
        assert not ws.mutants_file("mini", "L1").exists()
        res = pipeline.stage_mutate(ctx, spec, ["L1"])               # L0 有效，只做 L1，并复用 L0 选定的算子
        assert not res.skipped and res.detail["levels"] == "L1"
        l0, l1 = mutants_of(ws, "L0"), mutants_of(ws, "L1")
        assert all(l1[t]["operator"] == l0[t]["operator"] and l1[t]["same_operator_as_L0"] for t in l0)
        assert set(summary_of(ws)["levels"]) == {"L0", "L1"}
        assert pipeline.stage_mutate(ctx, spec, LEVELS2).skipped
        # L0 自己的输入变了（它的 oracle gate 重跑过）：L1 参照的 L0 哈希随之变了，两级都要重做
        data = store.read_json(ws.obf_file("mini", "L0"))
        data["oracle_gate"]["input_hash"] = "regate"
        store.write_json(ws.obf_file("mini", "L0"), data)
        res = pipeline.stage_mutate(ctx, spec, LEVELS2)
        assert not res.skipped and res.detail["levels"] == "L0/L1"

    def test_targets_that_failed_the_gate_and_invalid_copies_are_not_searched(self, rp):
        ctx, ws, targets, spec = rp
        median = by_name(targets, "calc_median3")
        write_gate(ws, "mini", "L1", targets, failed=[{"target_id": median.target_id, "category": "test_fail"}])
        data = store.read_json(ws.obf_file("mini", "L0"))
        data["ok"] = False
        store.write_json(ws.obf_file("mini", "L0"), data)
        res = pipeline.stage_mutate(ctx, spec, LEVELS2)
        assert res.ok and "oracle_failed=1" in res.message and "copy_invalid=4" in res.message
        assert mutants_of(ws, "L0") == {} and median.target_id not in mutants_of(ws, "L1")
        ent = summary_of(ws)["levels"]
        assert ent["L1"]["outcomes"] == {"killed": 3, "oracle_failed": 1} and ent["L1"]["n_searched"] == 3
        assert ent["L0"]["outcomes"] == {"copy_invalid": 4} and ent["L0"]["n_searched"] == 0
        assert ctx.sandbox.runs > 0

    def test_gate_is_required_and_checked_for_freshness(self, rp):
        ctx, ws, targets, spec = rp
        data = store.read_json(ws.obf_file("mini", "L1"))
        data["oracle_gate"] = None
        store.write_json(ws.obf_file("mini", "L1"), data)
        with pytest.raises(StageError, match="verify"):
            pipeline.stage_mutate(ctx, spec, LEVELS2)
        write_gate(ws, "mini", "L1", targets)
        write_targets(ws, spec, targets[:3])                           # 目标集变了，gate 没重跑
        with pytest.raises(StageError, match="oracle gate 是对着另一份目标集跑的"):
            pipeline.stage_mutate(ctx, spec, LEVELS2)

    def test_pairs_without_a_killed_mutant_are_counted_with_their_reason(self, rp):
        ctx, ws, targets, spec = rp
        ctx.sandbox = NothingKilledSandbox(ws)                          # 什么改动都测不出来：变异体全是等价变异体
        res = pipeline.stage_mutate(ctx, spec, LEVELS2, max_tries=2)
        assert res.ok and res.detail["killed"] == 0 and res.detail["abandoned"] == 8
        ent = summary_of(ws)["levels"]["L1"]
        assert ent["n_killed"] == 0 and ent["n_abandoned"] == 4 and ent["tried_sites"] == 8        # 每个目标只试 2 个就放弃
        assert ent["equivalent_count"] == 8 and ent["compile_fail_count"] == 0
        assert {a["outcome"] for a in ent["abandoned"]} == {"gave_up"} and len(ent["abandoned"]) == 4
        assert store.read_jsonl(ws.mutants_file("mini", "L1")) == []   # 文件在（空的），表示这一级做过了
        assert pipeline.stage_mutate(ctx, spec, LEVELS2, max_tries=2).skipped

    def test_a_stripped_original_that_fails_the_tests_gives_up_the_pair(self, rp):
        ctx, ws, targets, spec = rp
        # 模拟「去掉注释后的原函数通不过项目测试」：calc.c 里 Euclid 那条注释没了就判失败
        ctx.sandbox = FakeKillSandbox(ws, "mini", reject=lambda texts: "Euclid" not in texts.get("calc.c", "Euclid"))
        res = pipeline.stage_mutate(ctx, spec, LEVELS2)
        gcd = by_name(targets, "calc_gcd")
        assert "strip_unsafe=2" in res.message
        for lv in LEVELS2:
            assert gcd.target_id not in mutants_of(ws, lv) and len(mutants_of(ws, lv)) == 3
            ent = summary_of(ws)["levels"][lv]
            assert ent["n_strip_failed"] == 1 and ent["outcomes"]["strip_unsafe"] == 1
            assert [a["target_id"] for a in ent["abandoned"]] == [gcd.target_id]
            assert ent["abandoned"][0]["outcome"] == "strip_unsafe"


class NothingKilledSandbox(FakeKillSandbox):
    """所有候选都通过测试（变异体全是等价变异体）。"""

    def __init__(self, ws: Workspace) -> None:
        super().__init__(ws, "mini")

    def run(self, workdir, cmd, env=None, timeout_s=None):
        from bench.sandbox import RunResult
        with self._lock:
            self.runs += 1
        return RunResult(rc=0, stdout="ok\n", stderr="[]", duration_s=0.0, timed_out=False, signal=None)


# ----------------------------------------------------------------------
# 出题
# ----------------------------------------------------------------------
class TestPlanRepairTasks:
    @pytest.fixture()
    def mutated(self, rp):
        ctx, ws, targets, spec = rp
        pipeline.stage_mutate(ctx, spec, LEVELS2)
        return ctx, ws, targets, spec

    def test_tasks_carry_the_mutant_of_their_level_and_hide_the_original_names(self, mutated):
        ctx, ws, targets, spec = mutated
        planned, voids = pipeline.plan_repair_tasks(ctx, spec, LEVELS2)
        assert len(planned) == 8 and voids == {"L0": {}, "L1": {}}
        gcd0 = next(p for p in planned if p.task.level == "L0" and p.original.name == "calc_gcd")
        gcd1 = next(p for p in planned if p.task.level == "L1" and p.original.name == "calc_gcd")
        new_name = store.read_json(ws.obf_file("mini", "L1"))["rename_map"]["calc_gcd"]
        assert isinstance(gcd0.task, RepairTask) and gcd0.task.task_id == "mini:calc.c:calc_gcd@L0#repair"
        assert gcd1.task.target_id == gcd0.task.target_id and gcd1.task.obf_name == new_name != "calc_gcd"
        for pt in (gcd0, gcd1):
            rec = mutants_of(ws, pt.task.level)[pt.task.target_id]
            assert pt.task.mutant_code == rec["mutant_code"] and pt.task.operator == rec["operator"]
            assert pt.mutation == {k: rec[k] for k in ("operator", "site_index", "original_text", "mutated_text")}
            assert pt.kill_mode == "test_fail" and pt.stub_kill_mode == "test_fail"       # 前者是变异体的死法，后者是目标的 stub-kill 方式
            assert pt.same_operator_as_L0 is True and pt.same_site_as_L0 is True
            assert pt.mutants_hash == summary_of(ws)["levels"][pt.task.level]["input_hash"]
            assert pt.oracle == read_function(pt.obf_root, pt.located) and pt.oracle not in (pt.task.mutant_code,)
        user = build_repair_messages(gcd1.task)[1]["content"]
        assert new_name in user and "calc_gcd" not in user and "Euclid" not in user       # 名字和注释都不在提示词里
        assert gcd0.task.language_std == "c99" and gcd0.task.includes == gcd0.located.deps.includes

    def test_no_description_is_needed(self, mutated):
        ctx, ws, _, spec = mutated
        assert not ws.descriptions_file.exists()
        planned, _ = pipeline.plan_repair_tasks(ctx, spec, ["L0"])
        assert len(planned) == 4

    def test_missing_mutants_name_the_stage_to_run(self, rp):
        ctx, _, _, spec = rp
        with pytest.raises(StageError, match="mutate"):
            pipeline.plan_repair_tasks(ctx, spec, ["L0"])

    def test_void_reasons_are_counted(self, mutated):
        ctx, ws, targets, spec = mutated
        median, parse = by_name(targets, "calc_median3"), by_name(targets, "parse_digits")
        # 1. mutants 里少一个目标（模拟 mutate 没为它选出变异体）：改 jsonl 与汇总里的 n_killed，输入哈希不动
        recs = [r for r in store.read_jsonl(ws.mutants_file("mini", "L1")) if r["target_id"] != median.target_id]
        store.write_jsonl(ws.mutants_file("mini", "L1"), recs)
        summary = summary_of(ws)
        summary["levels"]["L1"]["n_killed"] = 3
        store.write_json(ws.mutants_summary_file("mini"), summary)
        # 2. 目标没做过 stub-kill
        trec = store.read_jsonl(ws.targets_file("mini"))
        for r in trec:
            if r["target_id"] == parse.target_id:
                r["stub_killed"] = None
        store.write_jsonl(ws.targets_file("mini"), trec)
        planned, voids = pipeline.plan_repair_tasks(ctx, spec, LEVELS2)
        assert voids == {"L0": {"stub_unchecked": 1}, "L1": {"stub_unchecked": 1, "no_killed_mutant": 1}}
        assert len(planned) == 3 + 2
        # unsafe_debug 放行没做过 stub-kill 的目标
        ctx.unsafe_debug = True
        planned, voids = pipeline.plan_repair_tasks(ctx, spec, LEVELS2)
        assert voids == {"L0": {}, "L1": {"no_killed_mutant": 1}} and len(planned) == 4 + 3

    def test_names_that_were_not_renamed_are_voided(self, mutated):
        ctx, ws, targets, spec = mutated
        gcd = by_name(targets, "calc_gcd")
        data = store.read_json(ws.obf_file("mini", "L1"))
        del data["rename_map"]["calc_gcd"]
        store.write_json(ws.obf_file("mini", "L1"), data)
        _, voids = pipeline.plan_repair_tasks(ctx, spec, LEVELS2)
        assert voids["L1"] == {"name_not_renamed": 1} and voids["L0"] == {}
        assert gcd.target_id in mutants_of(ws, "L1")                      # 变异体本身还在，只是这级提示词会带真名，不出题

    def test_invalid_copy_voids_the_whole_level(self, mutated):
        ctx, ws, _, spec = mutated
        data = store.read_json(ws.obf_file("mini", "L1"))
        data["ok"] = False
        store.write_json(ws.obf_file("mini", "L1"), data)
        planned, voids = pipeline.plan_repair_tasks(ctx, spec, LEVELS2)
        assert voids["L1"] == {"copy_invalid": 4} and {p.task.level for p in planned} == {"L0"}

    def test_stale_inputs_are_refused_where_the_mutants_are_used(self, mutated):
        ctx, ws, targets, spec = mutated
        # 副本重做（换种子）：gate 与 mutants 都对着旧副本
        pipeline.obfuscate(spec, ws.pristine("mini"), "L1", 99, ws.obf_root("mini", "L1"), verify=False,
                           results_dir=ws.results, force=True)
        write_gate(ws, "mini", "L1", targets)                            # gate 重新对上了新副本，mutants 没有
        with pytest.raises(StageError, match="变异体是对着另一份.*生成的.*mutate"):
            pipeline.plan_repair_tasks(ctx, spec, ["L1"])

    def test_mutants_file_tampering_and_a_redone_l0_are_refused(self, mutated):
        ctx, ws, _, spec = mutated
        recs = store.read_jsonl(ws.mutants_file("mini", "L1"))
        store.write_jsonl(ws.mutants_file("mini", "L1"), recs[:-1])
        with pytest.raises(StageError, match="与汇总文件对不上"):
            pipeline.plan_repair_tasks(ctx, spec, ["L1"])
        store.write_jsonl(ws.mutants_file("mini", "L1"), recs)
        assert pipeline.plan_repair_tasks(ctx, spec, ["L1"])[0]
        data = store.read_json(ws.obf_file("mini", "L0"))
        data["oracle_gate"]["input_hash"] = "regate"                     # L0 的输入变了，只重做 L0：L1 参照的 L0 哈希对不上了
        store.write_json(ws.obf_file("mini", "L0"), data)
        pipeline.stage_mutate(ctx, spec, ["L0"])
        with pytest.raises(StageError, match="L0 的变异体"):
            pipeline.plan_repair_tasks(ctx, spec, ["L1"])


# ----------------------------------------------------------------------
# mutate 产出里新增的字段（跨级别同位点、复核、上下文里多余的名字）与分次运行的一致性
# ----------------------------------------------------------------------
class TestMutateRecordsAndSummary:
    def test_records_carry_the_cross_level_and_recheck_fields(self, rp):
        ctx, ws, _, spec = rp
        pipeline.stage_mutate(ctx, spec, LEVELS2)
        l0, l1 = mutants_of(ws, "L0"), mutants_of(ws, "L1")
        for tid, r in l0.items():
            assert r["same_site_as_L0"] is True and r["anchor"] == "none" and r["unstable_count"] == 0
            assert r["kill_recheck"] == r["kill_mode"] == "test_fail" and isinstance(r["context_orphans"], list)
            assert l1[tid]["same_site_as_L0"] is True and l1[tid]["anchor"] == "chosen"
            assert (l1[tid]["operator"], l1[tid]["site_index"]) == (r["operator"], r["site_index"])
        ent = summary_of(ws)["levels"]["L1"]
        assert ent["same_site_as_L0"] == {"n": 4, "same": 4} and ent["anchor"] == {"chosen": 4}
        assert ent["n_rechecked"] == 4 and ent["unstable_count"] == 0 and ent["n_kill_label_changed"] == 0
        assert summary_of(ws)["recheck"] is True and ent["inputs"]["recheck"] is True
        assert set(ent["sandbox_seconds"]) <= {"pass", "test_fail"}

    def test_recheck_is_part_of_the_inputs(self, rp):
        ctx, ws, _, spec = rp
        pipeline.stage_mutate(ctx, spec, LEVELS2)
        redone = pipeline.stage_mutate(ctx, spec, LEVELS2, recheck=False)
        assert not redone.skipped and summary_of(ws)["levels"]["L1"]["n_rechecked"] == 0
        assert all(r["kill_recheck"] is None for r in mutants_of(ws, "L1").values())

    def test_redoing_only_l1_gives_the_same_records_as_the_whole_run(self, rp):
        ctx, ws, _, spec = rp
        pipeline.stage_mutate(ctx, spec, LEVELS2)
        before = mutants_of(ws, "L1")
        ctx.force = True
        res = pipeline.stage_mutate(ctx, spec, ["L1"])
        ctx.force = False
        assert not res.skipped and res.detail["levels"] == "L1"
        assert mutants_of(ws, "L1") == before

    def test_a_pair_whose_l0_is_strip_unsafe_has_no_reference_in_the_whole_run_and_in_a_split_run(self, rp):
        ctx, ws, targets, spec = rp
        gcd = by_name(targets, "calc_gcd")
        # 只有 L0 的去注释原函数通不过：L0 的文件里 calc_gcd 还叫原名，且注释已经没了
        ctx.sandbox = FakeKillSandbox(ws, "mini", reject=lambda texts: "calc_gcd" in texts.get("calc.c", "") and "Euclid" not in texts.get("calc.c", ""))
        pipeline.stage_mutate(ctx, spec, LEVELS2)
        assert gcd.target_id not in mutants_of(ws, "L0")
        l1 = mutants_of(ws, "L1")
        assert gcd.target_id in l1 and l1[gcd.target_id]["anchor"] == "none"
        assert l1[gcd.target_id]["same_operator_as_L0"] is None and l1[gcd.target_id]["same_site_as_L0"] is None
        ctx.force = True
        pipeline.stage_mutate(ctx, spec, ["L1"])                    # 只重做 L1：L0 没有 gcd 的记录，也就没有参照
        ctx.force = False
        assert mutants_of(ws, "L1") == l1

    def test_a_stale_l0_record_that_points_at_a_missing_site_is_refused(self, rp):
        ctx, ws, targets, spec = rp
        pipeline.stage_mutate(ctx, spec, ["L0"])
        recs = store.read_jsonl(ws.mutants_file("mini", "L0"))
        recs[0]["site_index"] = 999                                  # 位点已经不存在（L0 副本变了而记录没跟上）
        store.write_jsonl(ws.mutants_file("mini", "L0"), recs)
        with pytest.raises(StageError, match="L0"):
            pipeline.stage_mutate(ctx, spec, ["L1"])


# ----------------------------------------------------------------------
# 出题时重新核对 mutate 记下的全部输入
# ----------------------------------------------------------------------
class OtherSandbox(FakeKillSandbox):
    """换了一个沙箱身份（镜像 ID 不同），行为与 FakeKillSandbox 一样。"""

    def identity(self) -> dict:
        return {**super().identity(), "image_id": "sha256:another"}


class TestPlanRefusesStaleMutantsByEveryInput:
    @pytest.fixture()
    def mutated(self, rp):
        ctx, ws, targets, spec = rp
        pipeline.stage_mutate(ctx, spec, LEVELS2)
        return ctx, ws, targets, spec

    def test_changed_mutate_code_is_refused(self, mutated, monkeypatch):
        ctx, _, _, spec = mutated
        real = pipeline.code_digest
        monkeypatch.setattr(pipeline, "code_digest", lambda stage: "changed" if stage == "mutate" else real(stage))
        with pytest.raises(StageError, match="变异体是对着另一份.*mutate 相关代码.*生成的.*mutate"):
            pipeline.plan_repair_tasks(ctx, spec, ["L0"])

    def test_changed_seed_is_refused(self, mutated):
        ctx, ws, _, spec = mutated
        other = Context(ws=ws, sandbox=ctx.sandbox, jobs=2, seed=ctx.seed + 1)
        with pytest.raises(StageError, match="随机种子"):
            pipeline.plan_repair_tasks(other, spec, ["L0"])

    def test_changed_sandbox_is_refused_but_generate_has_no_sandbox_to_compare(self, mutated):
        ctx, ws, _, spec = mutated
        other = Context(ws=ws, sandbox=OtherSandbox(ws, "mini"), jobs=2, seed=ctx.seed)
        with pytest.raises(StageError, match="沙箱"):
            pipeline.plan_repair_tasks(other, spec, ["L1"])
        generate_side = Context(ws=ws, sandbox=None, jobs=2, seed=ctx.seed)               # generate 不碰沙箱：沙箱身份留给 evaluate 核对
        assert len(pipeline.plan_repair_tasks(generate_side, spec, ["L1"])[0]) == 4

    def test_changed_host_libclang_environment_is_refused(self, mutated, monkeypatch):
        ctx, _, _, spec = mutated
        monkeypatch.setattr(pipeline.clangcfg, "environment", lambda: {"python_binding": "99", "library": "x", "builtin_include": "y"})
        with pytest.raises(StageError, match="libclang"):
            pipeline.plan_repair_tasks(ctx, spec, ["L0"])

    @pytest.mark.parametrize("key, value, what", [("version", 99, "mutate 阶段版本"), ("strip_comments", False, "去注释设置")])
    def test_changed_stage_version_or_strip_setting_is_refused(self, mutated, key, value, what):
        ctx, ws, _, spec = mutated
        summary = summary_of(ws)
        summary["levels"]["L0"]["inputs"][key] = value
        store.write_json(ws.mutants_summary_file("mini"), summary)
        with pytest.raises(StageError, match=what):
            pipeline.plan_repair_tasks(ctx, spec, ["L0"])

    def test_an_untouched_mutate_output_is_accepted_by_every_check(self, mutated):
        ctx, _, _, spec = mutated
        assert len(pipeline.plan_repair_tasks(ctx, spec, LEVELS2)[0]) == 8


# ----------------------------------------------------------------------
# 出题：上下文按注入缺陷后的函数裁剪；L1 以上字面量里带着目标函数原名的题作废
# ----------------------------------------------------------------------
class TestContextPruningAndNameInLiteral:
    @pytest.fixture()
    def mutated(self, rp):
        ctx, ws, targets, spec = rp
        pipeline.stage_mutate(ctx, spec, LEVELS2)
        return ctx, ws, targets, spec

    def edit(self, ws, level, target, **changes):
        recs = store.read_jsonl(ws.mutants_file("mini", level))
        for r in recs:
            if r["target_id"] == target.target_id:
                r.update(changes)
        store.write_jsonl(ws.mutants_file("mini", level), recs)

    def without_the_assignment(self, ws, spec, target, level="L0"):
        root = ws.obf_root("mini", level)
        located = pipeline.locate(spec, root, target.name, target.file)
        code = strip_c_comments(read_function(root, located))
        line = '    g_last_error = "parse error";\n'
        assert line in code
        return code.replace(line, "")

    def test_the_context_loses_what_only_the_removed_code_used(self, mutated):
        ctx, ws, targets, spec = mutated
        parse = by_name(targets, "parse_digits")
        mutant = self.without_the_assignment(ws, spec, parse)
        self.edit(ws, "L0", parse, mutant_code=mutant, context_orphans=["g_last_error"])
        planned, _ = pipeline.plan_repair_tasks(ctx, spec, ["L0"])
        pt = next(p for p in planned if p.original.name == "parse_digits")
        assert pt.task.context_orphans == ["g_last_error"] and pt.task.context.globals == []
        assert pt.located.deps.globals == ["extern const char *g_last_error;"]              # 目标本身的依赖闭包没被动
        user = build_repair_messages(pt.task)[1]["content"]
        assert "g_last_error" not in user and "Global variables" not in user
        for other in planned:
            if other is not pt:
                assert other.task.context_orphans == [] and other.task.context == other.located.deps

    def test_orphans_recorded_by_mutate_must_match_what_planning_recomputes(self, mutated):
        ctx, ws, targets, spec = mutated
        parse = by_name(targets, "parse_digits")
        self.edit(ws, "L0", parse, mutant_code=self.without_the_assignment(ws, spec, parse), context_orphans=[])
        with pytest.raises(StageError, match="context_orphans"):
            pipeline.plan_repair_tasks(ctx, spec, ["L0"])

    def test_task_and_result_records_keep_the_orphans(self, mutated):
        ctx, ws, targets, spec = mutated
        parse = by_name(targets, "parse_digits")
        self.edit(ws, "L0", parse, mutant_code=self.without_the_assignment(ws, spec, parse), context_orphans=["g_last_error"])
        ctx.allow_mock = True
        pipeline.stage_generate(ctx, [spec], ["mock-unchanged"], ["L0"], task="repair")
        rec = next(t for t in store.read_jsonl(ws.repair_tasks_file) if t["target_id"] == parse.target_id)
        assert rec["context_orphans"] == ["g_last_error"] and rec["context"]["globals"] == []
        pipeline.stage_evaluate(ctx, [spec], ["mock-unchanged"], ["L0"], tag="t", task="repair")
        res = {r["target_id"]: r for r in store.read_jsonl(ws.repair_results_file("t"))}
        assert res[parse.target_id]["context_orphans"] == ["g_last_error"]
        assert all(r["context_orphans"] == [] for tid, r in res.items() if tid != parse.target_id)

    def test_the_original_name_in_a_string_literal_voids_l1_but_not_l0(self, mutated):
        ctx, ws, targets, spec = mutated
        gcd = by_name(targets, "calc_gcd")
        for level in LEVELS2:
            code = store.read_jsonl(ws.mutants_file("mini", level))
            rec = next(r for r in code if r["target_id"] == gcd.target_id)
            self.edit(ws, level, gcd, mutant_code=rec["mutant_code"].replace("{", '{\n    const char *where = "calc_gcd";', 1),
                      context_orphans=rec["context_orphans"])
        planned, voids = pipeline.plan_repair_tasks(ctx, spec, LEVELS2)
        assert voids == {"L0": {}, "L1": {"name_in_literal": 1}}
        assert gcd.target_id in {p.original.target_id for p in planned if p.task.level == "L0"}
        assert gcd.target_id not in {p.original.target_id for p in planned if p.task.level == "L1"}

    def test_the_name_inside_an_identifier_or_a_comment_like_literal_is_not_a_leak(self, mutated):
        from bench.pipeline import _own_name_in_literals
        assert _own_name_in_literals("calc_gcd", 'x = "calc_gcd";') and _own_name_in_literals("calc_gcd", "c = 'x'; s = \"see calc_gcd.\";")
        assert not _own_name_in_literals("calc_gcd", 'x = "calc_gcd2"; y = "my_calc_gcd";')
        assert not _own_name_in_literals("calc_gcd", "int calc_gcd(int a) { return a; }")            # 代码里的名字另有 name_not_renamed 管
        assert not _own_name_in_literals("calc_gcd", "/* calc_gcd */ int f(void) { return 1; }")


# ----------------------------------------------------------------------
# check-mutants：回放复核
# ----------------------------------------------------------------------
class CrashSandbox(FakeKillSandbox):
    """改动过的文件一律段错误（rc=139，signal=11）：回放时归因是 runtime_crash，与 mutate 记下的 test_fail 不同。"""

    def run(self, workdir, cmd, env=None, timeout_s=None):
        res = super().run(workdir, cmd, env=env, timeout_s=timeout_s)
        if "-fsyntax-only" not in cmd and res.rc == 1:
            return dataclasses.replace(res, rc=139, signal=11)
        return res


class TestCheckMutants:
    @pytest.fixture()
    def mutated(self, rp):
        ctx, ws, targets, spec = rp
        pipeline.stage_mutate(ctx, spec, LEVELS2)
        return ctx, ws, targets, spec

    def test_replay_confirms_every_mutant_and_writes_the_check_file(self, mutated):
        ctx, ws, _, spec = mutated
        res = pipeline.stage_check_mutants(ctx, spec, LEVELS2, with_oracle=True)
        assert res.ok and res.stage == "check-mutants" and not res.message
        assert res.detail == {"replayed": 8, "killed": 8, "same_mode": 8, "passed": 0, "other": 0, "flipped": 0, "oracle_failed": 0}
        data = store.read_json(ws.mutants_check_file("mini"))
        assert data["total"]["replayed"] == 8 and data["levels"]["L1"]["killed"] == 4
        assert data["flipped"] == [] and data["failed"] == [] and data["with_oracle"] is True
        assert data["inputs"]["mutants"] == {lv: summary_of(ws)["levels"][lv]["input_hash"] for lv in LEVELS2}
        assert ws.mutants_check_file("mini").name == "mutants-mini.check.json"

    def test_replay_without_the_oracle_does_not_run_the_originals(self, mutated):
        ctx, _, _, spec = mutated
        before = ctx.sandbox.runs
        res = pipeline.stage_check_mutants(ctx, spec, ["L0"])
        assert res.ok and "oracle_failed" not in res.detail and ctx.sandbox.runs - before == 8        # 4 个变异体 × (语法检查 + 构建)

    def test_a_mutant_that_passes_on_replay_fails_the_stage(self, mutated):
        ctx, ws, _, spec = mutated
        ctx.sandbox = NothingKilledSandbox(ws)
        res = pipeline.stage_check_mutants(ctx, spec, LEVELS2)
        assert not res.ok and res.detail["passed"] == 8 and res.detail["killed"] == 0 and "回放不合格" in res.message
        data = store.read_json(ws.mutants_check_file("mini"))
        assert len(data["failed"]) == 8 and all(r["replay"] == "pass" for r in data["failed"])

    def test_an_original_that_fails_fails_the_oracle_replay(self, mutated):
        ctx, ws, _, spec = mutated
        ctx.sandbox = FakeKillSandbox(ws, "mini", reject=lambda texts: True)                   # 什么都不让过，包括去注释的原函数
        res = pipeline.stage_check_mutants(ctx, spec, ["L0"], with_oracle=True)
        assert not res.ok and res.detail["oracle_failed"] == 4 and res.detail["killed"] == 4

    def test_a_changed_kill_mode_is_reported_but_is_not_a_failure(self, mutated):
        ctx, ws, _, spec = mutated
        ctx.sandbox = CrashSandbox(ws, "mini")
        res = pipeline.stage_check_mutants(ctx, spec, ["L0"])
        assert res.ok and res.detail["killed"] == 4 and res.detail["same_mode"] == 0 and res.detail["flipped"] == 4
        assert "杀死方式" in res.message
        data = store.read_json(ws.mutants_check_file("mini"))
        assert {(f["recorded"], f["replay"]) for f in data["flipped"]} == {("test_fail", "runtime_crash")}

    def test_stale_mutants_are_refused_before_anything_is_replayed(self, mutated, monkeypatch):
        ctx, ws, _, spec = mutated
        real = pipeline.code_digest
        monkeypatch.setattr(pipeline, "code_digest", lambda stage: "changed" if stage == "mutate" else real(stage))
        before = ctx.sandbox.runs
        with pytest.raises(StageError, match="mutate"):
            pipeline.stage_check_mutants(ctx, spec, ["L0"])
        assert ctx.sandbox.runs == before and not ws.mutants_check_file("mini").exists()

    def test_the_cli_command_is_wired_up(self):
        ns = cli.build_parser().parse_args(["check-mutants", "--levels", "L1", "--oracle", "--corpus", "mini", "--jobs", "2"])
        assert (ns.cmd, ns.levels, ns.oracle, ns.corpus, ns.jobs) == ("check-mutants", ["L1"], True, ["mini"], 2)
        assert cli._COMMANDS["check-mutants"] is cli._cmd_check_mutants


# ----------------------------------------------------------------------
# unchanged 的候选却通过了测试：变异体不可靠，评测阶段点名并失败
# ----------------------------------------------------------------------
class TestUnreliableMutants:
    def test_an_unchanged_candidate_that_passes_fails_the_evaluate_stage_and_is_named(self, rp):
        ctx, ws, _, spec = rp
        ctx.allow_mock = True
        pipeline.stage_mutate(ctx, spec, LEVELS2)
        pipeline.stage_generate(ctx, [spec], ["mock-unchanged", "mock-oracle"], LEVELS2, task="repair")
        ctx.sandbox = NothingKilledSandbox(ws)                       # 评测时测试什么都测不出来：原样返回变异体的候选也过了
        ev = pipeline.stage_evaluate(ctx, [spec], ["mock-unchanged", "mock-oracle"], LEVELS2, tag="t", task="repair")
        by_model = {r.corpus.split("/")[0]: r for r in ev}
        bad = by_model["mock-unchanged"]
        assert not bad.ok and "8 道修复题" in bad.message and "unchanged" in bad.message and "calc_gcd" in bad.message
        assert by_model["mock-oracle"].ok                           # oracle 的候选不是 unchanged：通过是应当的
        recs = [r for r in store.read_jsonl(ws.repair_results_file("t")) if r["model"] == "mock-unchanged"]
        assert all(r["unchanged"] and r["category"] == "pass" for r in recs)
        again = pipeline.stage_evaluate(ctx, [spec], ["mock-unchanged"], LEVELS2, tag="t", task="repair")
        assert again[0].skipped and not again[0].ok and "unchanged" in again[0].message           # 跳过也照样提示

    def test_generation_results_are_never_flagged(self, planned):
        ctx, ws, _ = planned
        assert pipeline._unreliable_tasks(ws.results_file("x"), "m", lambda r: True) == set()


# ----------------------------------------------------------------------
# 生成任务的 tag / 模型名不能与修复任务的文件撞名
# ----------------------------------------------------------------------
class TestRepairPrefixCollision:
    def test_names_that_would_collide_with_the_repair_files_are_refused(self, scratch):
        ws = Workspace.under(scratch)
        for call in (lambda: ws.results_file("repair-x"), lambda: ws.completeness_file("repair-x"),
                     lambda: ws.generations_file("repair-m"), lambda: ws.generation_failures_file("repair-m")):
            with pytest.raises(ValueError, match="repair-"):
                call()
        assert ws.repair_results_file("x") == ws.results / "results-repair-x.jsonl"
        assert ws.results_file("pilot").name == "results-pilot.jsonl" and ws.generations_file("deepseek-flash").name == "generations-deepseek-flash.jsonl"
        assert ws.results_file("Repair-x").name == "results-Repair-x.jsonl"                    # 只认小写的 repair- 前缀（文件名区分大小写）

    def test_the_command_line_reports_it_as_a_usage_error(self, scratch, capsys):
        ws = Workspace.under(scratch).ensure()
        assert cli.main(["report", "--tag", "repair-x", "--workspace", str(ws.results.parent)]) == 2
        assert "repair-" in capsys.readouterr().err


# ----------------------------------------------------------------------
# generate / evaluate 的 repair 分支
# ----------------------------------------------------------------------
MODELS = ("mock-oracle", "mock-broken", "mock-wrong", "mock-unchanged")


class TestGenerateAndEvaluateRepair:
    @pytest.fixture()
    def evaluated(self, rp):
        ctx, ws, targets, spec = rp
        ctx.allow_mock = True
        pipeline.stage_mutate(ctx, spec, LEVELS2)
        gen = pipeline.stage_generate(ctx, [spec], MODELS, LEVELS2, task="repair")
        ev = pipeline.stage_evaluate(ctx, [spec], MODELS, LEVELS2, tag="t", task="repair")
        return ctx, ws, targets, spec, gen, ev

    def test_generate_writes_the_repair_files_and_leaves_the_generate_files_alone(self, evaluated):
        ctx, ws, _, _, gen, _ = evaluated
        assert all(r.ok and "[repair]" in r.corpus for r in gen) and all(r.detail["calls"] == 8 for r in gen)
        assert not ws.tasks_file.exists() and not any(ws.generations_file(m).exists() for m in MODELS)
        tasks = store.read_jsonl(ws.repair_tasks_file)
        assert len(tasks) == 8 and {t["task"] for t in tasks} == {"repair"}
        assert all(t["prompt_digest"] == REPAIR_PROMPT_DIGEST[:16] and t["mutation"]["operator"] == t["operator"]
                   and t["mutants"] and t["kill_mode"] == "test_fail" for t in tasks)
        for m in MODELS:
            rows = store.read_jsonl(ws.repair_generations_file(m))
            assert len(rows) == 8 and {r["task"] for r in rows} == {"repair"} and all(r["mutation"] for r in rows)
        assert RepairTask.from_dict(tasks[0]).mutant_code == tasks[0]["mutant_code"]

    def test_each_mock_mode_gets_the_documented_verdict(self, evaluated):
        ctx, ws, _, _, _, ev = evaluated
        assert all(r.ok for r in ev)
        recs = store.read_jsonl(ws.repair_results_file("t"))
        assert len(recs) == 32 and not ws.results_file("t").exists()
        by = {m: [r for r in recs if r["model"] == m] for m in MODELS}
        assert all(r["category"] == "pass" and r["restored"] is True and r["unchanged"] is False for r in by["mock-oracle"])
        assert all(r["category"] != "pass" and r["unchanged"] is True and r["restored"] is False
                   and r["stage"] == "test" and r["category"] == r["kill_mode"] for r in by["mock-unchanged"])
        assert all(r["category"] == "compile_syntax" and r["unchanged"] is False for r in by["mock-broken"])
        assert all(r["category"] == "test_fail" and r["stage"] == "test" and not r["unchanged"] and not r["restored"]
                   for r in by["mock-wrong"])

    def test_result_records_carry_the_repair_fields_and_inputs(self, evaluated):
        ctx, ws, _, _, _, _ = evaluated
        recs = store.read_jsonl(ws.repair_results_file("t"))
        tasks = {t["task_id"]: t for t in store.read_jsonl(ws.repair_tasks_file)}
        for r in recs:
            assert r["task"] == "repair" and r["task_id"].endswith("#repair")
            t = tasks[r["task_id"]]
            assert r["mutation"] == t["mutation"] and r["kill_mode"] == t["kill_mode"]
            assert r["same_operator_as_L0"] is True and r["stub_kill_mode"] == "test_fail"
            assert r["inputs"]["prompt"] == REPAIR_PROMPT_DIGEST[:16] and r["inputs"]["mutants"] == t["mutants"]
            assert r["inputs"]["messages"] == t["messages_hash"] and set(r["inputs"]) >= {"spec", "tree", "gen", "reply", "gate"}
        assert validate_records(recs, ws, task="repair").valid == recs
        assert len(validate_records(recs, ws).valid) == 0                  # 当成生成任务校验：prompt 摘要对不上

    def test_completeness_and_failures_use_their_own_files(self, evaluated):
        _, ws, _, _, _, _ = evaluated
        data = store.read_json(ws.repair_completeness_file("t"))
        assert data and not ws.completeness_file("t").exists()
        assert sum(c["expected"] for c in data["cells"]) == 32 and sum(c["missing"] for c in data["cells"]) == 0

    def test_everything_is_skipped_on_the_second_run(self, evaluated):
        ctx, ws, _, spec, _, _ = evaluated
        runs = ctx.sandbox.runs
        assert all(r.skipped for r in pipeline.stage_generate(ctx, [spec], MODELS, LEVELS2, task="repair"))
        assert all(r.skipped for r in pipeline.stage_evaluate(ctx, [spec], MODELS, LEVELS2, tag="t", task="repair"))
        assert ctx.sandbox.runs == runs

    def test_a_redone_mutate_makes_the_old_results_stale(self, evaluated):
        ctx, ws, _, spec, _, _ = evaluated
        recs = store.read_jsonl(ws.repair_results_file("t"))
        ctx.force = True
        pipeline.stage_mutate(ctx, spec, LEVELS2, max_tries=7)           # 输入变了（max_tries），mutants 的输入哈希随之变
        ctx.force = False
        stale = validate_records(recs, ws, task="repair")
        assert not stale.valid and {r for _, r in stale.stale} == {"变异体已重做（mutate 重跑过，评测结果对应的是旧变异体）"}
        # 重新 generate / evaluate 之后恢复有效：回复不用重新生成（提示词没变），评测结果的输入哈希变了所以重做
        pipeline.stage_generate(ctx, [spec], MODELS, LEVELS2, task="repair")
        pipeline.stage_evaluate(ctx, [spec], MODELS, LEVELS2, tag="t", task="repair")
        fresh = store.read_jsonl(ws.repair_results_file("t"))
        assert len(fresh) == 32 and len(validate_records(fresh, ws, task="repair").valid) == 32

    def test_dry_run_estimates_without_writing(self, rp):
        ctx, ws, _, spec = rp
        pipeline.stage_mutate(ctx, spec, LEVELS2)
        (res,) = pipeline.stage_generate(ctx, [spec], ["deepseek-flash"], LEVELS2, task="repair", dry_run=True)
        assert res.detail["tasks"] == 8 and res.detail["to_call"] == 8 and "[repair]" in res.corpus
        assert not ws.repair_tasks_file.exists() and not ws.repair_generations_file("deepseek-flash").exists()

    def test_unknown_task_kind_is_refused(self, rp):
        ctx, _, _, spec = rp
        with pytest.raises(StageError, match="未知任务类型"):
            pipeline.stage_generate(ctx, [spec], ["mock-oracle"], LEVELS2, task="nope")


# ----------------------------------------------------------------------
# --task 的取舍与命令行
# ----------------------------------------------------------------------
class TestTaskSelection:
    def test_all_runs_the_repair_half_only_when_mutate_has_produced_something(self, rp):
        ctx, ws, _, spec = rp
        assert pipeline.resolve_tasks(ctx, [spec], LEVELS2, "generate") == (["generate"], [])
        assert pipeline.resolve_tasks(ctx, [spec], LEVELS2, "repair") == (["repair"], [])
        tasks, notes = pipeline.resolve_tasks(ctx, [spec], LEVELS2, "all")
        assert tasks == ["generate"] and len(notes) == 1 and "mutate" in notes[0] and "已跳过" in notes[0]
        pipeline.stage_mutate(ctx, spec, ["L0"])
        assert pipeline.resolve_tasks(ctx, [spec], LEVELS2, "all") == (["generate", "repair"], [])
        assert pipeline.resolve_tasks(ctx, [spec], ["L1"], "all")[0] == ["generate"]       # 所选级别里还没有 mutate 的产出
        with pytest.raises(StageError, match="未知任务类型"):
            pipeline.resolve_tasks(ctx, [spec], LEVELS2, "both")

    def test_cli_parses_the_new_command_and_option(self):
        ap = cli.build_parser()
        ns = ap.parse_args(["mutate"])
        assert (ns.cmd, ns.levels, ns.max_tries) == ("mutate", None, 12)
        ns = ap.parse_args(["mutate", "--levels", "L1", "L2", "--max-tries", "5", "--corpus", "a", "--jobs", "3"])
        assert (ns.levels, ns.max_tries, ns.corpus, ns.jobs) == (["L1", "L2"], 5, ["a"], 3)
        for cmd in ("generate", "evaluate", "report"):
            extra = ["--model", "m"] if cmd == "evaluate" else []
            assert ap.parse_args([cmd, *extra]).task == "all"
            assert ap.parse_args([cmd, *extra, "--task", "repair"]).task == "repair"
            with pytest.raises(SystemExit):
                ap.parse_args([cmd, *extra, "--task", "both"])

    def test_cli_generate_all_skips_repair_with_a_note_and_repair_alone_fails_clearly(self, rp, capsys):
        ctx, ws, _, spec = rp
        base = ["--corpus", str(cli.MINI_FIXTURE / "spec.json"), "--workspace", str(ws.results.parent), "--seed", "3",
                "--jobs", "2", "--unsafe-debug"]
        # describe + generate（生成任务）照常；repair 在没有 mutate 产出时只给提示
        ctx.allow_mock = True
        pipeline.stage_describe(ctx, spec, pipeline.build_llm("mock-oracle", ws), model_name="mock-oracle")
        assert cli.main(["generate", "--model", "mock-oracle", "--levels", "L0", "L1", *base]) == 0
        out = capsys.readouterr().out
        assert "修复任务：还没有 mutate 的产出" in out and "calls=8" in out and "[repair]" not in out
        assert cli.main(["generate", "--model", "mock-oracle", "--levels", "L0", "L1", "--task", "repair", *base]) == 2
        assert "mutate" in capsys.readouterr().err
        pipeline.stage_mutate(ctx, spec, LEVELS2)
        assert cli.main(["generate", "--model", "mock-oracle", "--levels", "L0", "L1", *base]) == 0
        out = capsys.readouterr().out
        assert "[repair]" in out and "修复任务：还没有" not in out
        assert ws.repair_generations_file("mock-oracle").is_file()
        assert cli.main(["generate", "--model", "mock-oracle", "--levels", "L0", "L1", "--task", "generate", *base]) == 0
        assert "[repair]" not in capsys.readouterr().out


def test_the_mutate_stage_digest_covers_the_code_that_decides_what_a_mutant_is():
    """判定逻辑（位点、改写、kill 检查走的评测与归因）改了，旧 mutants 记录要自动失效：它们都在 mutate 阶段的代码摘要里。"""
    from bench import provenance
    files = provenance._files_for("mutate")
    assert all(f.is_file() for f in files)
    assert {f.name for f in files} >= {"mutate.py", "runner.py", "attribute.py", "extract.py", "corpus.py", "sandbox.py",
                                       "clangcfg.py", "_edits.py", "postprocess.py", "vocab.py", "prompt.py", "similarity.py"}
    assert provenance.code_digest("mutate") != provenance.code_digest("evaluate")


def test_the_mutate_stage_digest_really_changes_with_the_content_of_a_covered_file(scratch, monkeypatch):
    """摘要不只是列了文件名：被覆盖的文件内容变了摘要就变，没被覆盖的文件变了不影响（在一份拷贝上验证，不碰真实源文件）。"""
    import shutil
    from bench import provenance
    pkg = scratch / "pkg"
    for f in provenance._files_for("mutate"):
        dest = pkg / f.relative_to(provenance._PKG_DIR)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(f, dest)
    (pkg / "describe.py").write_text("# 与 mutate 阶段无关\n", encoding="utf-8")
    monkeypatch.setattr(provenance, "_PKG_DIR", pkg)
    digest = provenance.code_digest.__wrapped__                       # 绕开 lru_cache
    base = digest("mutate")
    assert digest("mutate") == base
    (pkg / "describe.py").write_text("# 改了，但 mutate 不覆盖它\n", encoding="utf-8")
    assert digest("mutate") == base
    for covered in ("mutate.py", "prompt.py", "similarity.py"):
        path = pkg / covered
        old = path.read_text(encoding="utf-8")
        path.write_text(old + "\n# changed\n", encoding="utf-8")
        assert digest("mutate") != base, covered
        path.write_text(old, encoding="utf-8")
        assert digest("mutate") == base


def test_picked_targets_are_what_the_tests_assume(rp):
    _, _, targets, _ = rp
    assert sorted(t.name for t in targets) == sorted(PICKED)
