"""缺陷修复任务的指标与报告：metrics 的 unchanged / restored / 按算子与杀死方式分层 / 四级同算子子集，
report 的两类任务并列汇总、输入校验、CSV 与图，以及实验记录里的 mutate 一节。全部用合成记录，不需要 docker。
"""
from __future__ import annotations

import csv
import io
import xml.etree.ElementTree as ET

import pytest

from bench import cli
from bench import experiment_record as er
from bench import metrics, report, store
from bench.limitations import LIMITATIONS, TASK_INPUT_LINES
from bench.metrics import (
    comparable_by_level, four_level_comparable, four_level_same_operator, four_level_same_site, repair_subsets, subset,
    unreliable_tasks,
)
from bench.prompt import PROMPT_DIGEST, REPAIR_PROMPT_DIGEST
from bench.provenance import code_digest
from bench.store import SCHEMA_VERSION, Workspace
from bench.vocab import KILL_MODES, LEVELS, MUTATION_OPERATORS
from tests.helpers import BANNED


def rrec(model="m1", level="L0", corpus="c1", target="t0", idx=0, *, op="ROR", category="pass", stage=None,
         kill_mode="test_fail", unchanged=False, restored=False, sim=0.9, stub_kill_mode="test_fail", same_site=True,
         orphans=(), **extra) -> dict:
    """一条合成的缺陷修复任务评测记录（只含 metrics / report 用到的字段）。"""
    out = {"task": "repair", "model": model, "level": level, "corpus": corpus, "target_id": f"{corpus}:{target}",
           "task_id": f"{corpus}:{target}@{level}#repair", "sample_idx": idx,
           "stage": stage or ("pass" if category == "pass" else "test"), "category": category, "similarity": sim,
           "prompt_tokens": 100, "completion_tokens": 20, "cached": False,
           "mutation": {"operator": op, "site_index": 0, "original_text": "<", "mutated_text": "<="},
           "kill_mode": kill_mode, "unchanged": unchanged, "restored": restored, "stub_kill_mode": stub_kill_mode,
           "same_operator_as_L0": True, "same_site_as_L0": same_site, "context_orphans": list(orphans)}
    out.update(extra)
    return out


def grec(model="m1", level="L0", corpus="c1", target="t0", idx=0, category="pass", **extra) -> dict:
    """一条合成的生成任务评测记录。"""
    out = {"model": model, "level": level, "corpus": corpus, "target_id": f"{corpus}:{target}",
           "task_id": f"{corpus}:{target}@{level}", "sample_idx": idx,
           "stage": "pass" if category == "pass" else "test", "category": category, "similarity": 0.8,
           "prompt_tokens": 100, "completion_tokens": 20, "cached": False}
    out.update(extra)
    return out


# ----------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------
class TestRepairMetrics:
    def test_unchanged_and_restored_rates(self):
        recs = [rrec(target="a", unchanged=True, category="test_fail"), rrec(target="b", restored=True),
                rrec(target="c"), rrec(target="d", category="test_fail")]
        (cell,) = metrics.summarize(recs, reply_sharing=False)
        assert (cell.task, cell.n_unchanged, cell.n_restored) == ("repair", 1, 1)
        assert (cell.unchanged_rate, cell.restored_rate) == (0.25, 0.25)
        assert cell.to_dict()["unchanged_rate"] == 0.25 and cell.to_dict()["task"] == "repair"
        (gen,) = metrics.summarize([grec()])
        assert gen.task == "generate" and gen.n_unchanged == gen.n_restored == 0 and gen.operators == {}

    def test_strata_by_operator_and_by_the_mutants_kill_mode(self):
        recs = [rrec(target="a", op="ROR"), rrec(target="b", op="ROR", category="test_fail"),
                rrec(target="c", op="DST", kill_mode="runtime_crash"),
                rrec(target="d", op="DST", kill_mode="runtime_crash", category="test_fail")]
        (cell,) = metrics.summarize(recs, reply_sharing=False)
        assert cell.operators == {"ROR": {"n": 2, "n_pass": 1}, "DST": {"n": 2, "n_pass": 1}}
        # 修复任务按变异体的 kill_mode 分层，而不是目标的 stub_kill_mode（这里全是 test_fail）
        assert cell.kill_modes == {"test_fail": {"n": 2, "n_pass": 1}, "runtime_crash": {"n": 2, "n_pass": 1}}
        assert metrics.stratum_mode(rrec(kill_mode="runtime_timeout")) == "runtime_timeout"
        assert metrics.stratum_mode(grec(stub_kill_mode="runtime_crash", kill_mode="test_fail")) == "runtime_crash"
        assert metrics.record_task(rrec()) == "repair" and metrics.record_task(grec()) == "generate"

    def test_reply_sharing_is_off_for_repair_tasks(self):
        def r(level, reply):
            return rrec(level=level, inputs={"reply": reply})
        recs = [r("L0", "a"), r("L1", "b"), r("L2", "b"), r("L3", "b")]
        cells = {c.level: c for c in metrics.summarize_all(recs, reply_sharing=False)}
        assert cells["L2"].same_reply_as_l1 is None and cells["L2"].n_reply_compared == 0
        assert {d.level: d.n_same_as_l1 for d in metrics.drop_vs_baseline(recs, per_corpus=False, reply_sharing=False)} \
            == {"L1": None, "L2": None, "L3": None}
        with_sharing = {c.level: c for c in metrics.summarize_all(recs)}
        assert with_sharing["L2"].same_reply_as_l1 == 1.0                  # 默认行为不变

    def test_drops_know_their_task(self):
        recs = [rrec(level="L0"), rrec(level="L1", category="test_fail")]
        (d,) = metrics.drop_vs_baseline(recs, per_corpus=False, reply_sharing=False)
        assert d.task == "repair" and d.delta_pass == -1.0
        (g,) = metrics.drop_vs_baseline([grec(level="L0"), grec(level="L1")], per_corpus=False)
        assert g.task == "generate"

    def test_four_level_same_operator_subset(self):
        recs = []
        for tgt, ops in {"a": "ROR", "b": "ROR"}.items():                        # 四级同一个算子
            recs += [rrec(level=lv, target=tgt, op=ops) for lv in LEVELS]
        recs += [rrec(level=lv, target="c", op=op) for lv, op in zip(LEVELS, ("ROR", "ROR", "NEG", "ROR"))]   # 算子不全相同
        recs += [rrec(level=lv, target="d") for lv in LEVELS[:3]]                # 缺 L3
        recs += [rrec(level=lv, target="e", model="m2") for lv in LEVELS]       # 另一个模型的记录同样汇总（变异体与模型无关）
        recs += [rrec(level="L0", target="f", op=None) | {"mutation": None}]    # 没有 mutation 的记录不参与
        ids = four_level_same_operator(recs)
        assert ids == {("c1", "c1:a"), ("c1", "c1:b"), ("c1", "c1:e")}
        sub = subset(recs, ids)
        assert len(sub) == 12 and {r["target_id"] for r in sub} == {"c1:a", "c1:b", "c1:e"}
        assert four_level_same_operator([]) == set()

    def test_same_site_comparable_and_the_other_subsets_are_told_apart(self):
        recs = []
        for lv in LEVELS:                                                     # a：四级同位点、上下文干净 -> 可比
            recs.append(rrec(level=lv, target="a"))
        for lv in LEVELS:                                                     # b：同算子，但 L2 不是同一个位点
            recs.append(rrec(level=lv, target="b", same_site=lv != "L2"))
        for lv in LEVELS:                                                     # c：同位点，但 L1 的上下文被裁了条目
            recs.append(rrec(level=lv, target="c", orphans=["helper"] if lv == "L1" else ()))
        for lv in LEVELS:                                                     # d：L3 换了算子（不可能同位点）
            recs.append(rrec(level=lv, target="d", op="NEG" if lv == "L3" else "ROR", same_site=lv != "L3"))
        recs += [rrec(level=lv, target="e") for lv in LEVELS[:3]]            # e：缺 L3
        recs += [rrec(level="L0", target="f", same_site=None) | {"mutation": None}]      # 没有 mutation 的记录不参与
        s = repair_subsets(recs)
        key = lambda t: ("c1", f"c1:{t}")                                      # noqa: E731
        assert s.all_targets == {key(t) for t in "abcde"}
        assert s.four_levels == {key(t) for t in "abcd"}
        assert s.same_operator == {key(t) for t in "abc"}
        assert s.same_site == {key(t) for t in "ac"}
        assert s.clean_context == {key(t) for t in "abd"}
        assert s.comparable == {key("a")}
        assert four_level_same_site(recs) == s.same_site and four_level_comparable(recs) == s.comparable
        assert four_level_same_operator(recs) == s.same_operator

    def test_same_operator_is_the_whole_set_whenever_the_operator_is_reused(self):
        """prefer_operator 让 L1–L3 复用 L0 的算子：只要四级都出了题，「四级同算子」就恒等于「四级齐全」，不构成对照。"""
        recs = [rrec(level=lv, target=t, same_site=lv in ("L0", "L1")) for t in ("a", "b", "c") for lv in LEVELS]
        s = repair_subsets(recs)
        assert s.same_operator == s.four_levels and s.same_site == set()

    def test_comparable_by_level_needs_l0_and_the_level_to_agree_but_not_the_other_levels(self):
        recs = ([rrec(level=lv, target="a", same_site=lv != "L3") for lv in LEVELS]
                + [rrec(level=lv, target="b", orphans=["h"] if lv == "L0" else ()) for lv in LEVELS]
                + [rrec(level="L1", target="c")])                                              # c 没有 L0 的记录
        got = comparable_by_level(recs)
        assert got["L1"] == {("c1", "c1:a")} and got["L2"] == {("c1", "c1:a")} and got["L3"] == set()      # b 的 L0 上下文被裁了

    def test_unreliable_tasks_are_the_ones_whose_unchanged_candidates_pass(self):
        recs = [rrec(target="a", unchanged=True, category="test_fail"),                      # 正常：什么也没改，没过
                rrec(target="b", unchanged=True, category="pass"),                           # 不可靠
                rrec(target="c", restored=True), rrec(target="d", category="test_fail"),
                rrec(model="m2", level="L1", target="e", unchanged=True, category="pass"),
                grec(target="g", category="pass") | {"unchanged": True}]                     # 生成任务的记录不看
        assert unreliable_tasks(recs) == {("c1", "c1:b@L0#repair"), ("c1", "c1:e@L1#repair")}
        assert unreliable_tasks([]) == set()

    def test_stub_mode_strata_and_the_context_strata(self):
        recs = [rrec(target="a", stub_kill_mode="runtime_crash", kill_mode="test_fail"),
                rrec(target="b", stub_kill_mode="runtime_crash", kill_mode="runtime_timeout", category="test_fail"),
                rrec(target="c", stub_kill_mode="test_fail", kill_mode="runtime_crash", orphans=["h"]),
                rrec(target="d", stub_kill_mode="test_fail", kill_mode="test_fail", orphans=["h"], category="test_fail")]
        (cell,) = metrics.summarize(recs, reply_sharing=False)
        # 两个口径各自分层，不混：变异体怎么死的 vs 目标被零值桩杀死的方式
        assert cell.kill_modes == {"test_fail": {"n": 2, "n_pass": 1}, "runtime_timeout": {"n": 1, "n_pass": 0},
                                   "runtime_crash": {"n": 1, "n_pass": 1}}
        assert cell.stub_modes == {"runtime_crash": {"n": 2, "n_pass": 1}, "test_fail": {"n": 2, "n_pass": 1}}
        assert cell.context == {"without": {"n": 2, "n_pass": 1}, "with": {"n": 2, "n_pass": 1}}
        assert cell.to_dict()["stub_modes"] == cell.stub_modes and cell.to_dict()["context"] == cell.context
        assert metrics.stub_mode(rrec(stub_kill_mode="runtime_timeout")) == "runtime_timeout" and metrics.stub_mode(grec()) is None
        (gen,) = metrics.summarize([grec(stub_kill_mode="runtime_crash")])
        assert gen.kill_modes == gen.stub_modes == {"runtime_crash": {"n": 1, "n_pass": 1}} and gen.context == {}

    def test_a_model_missing_one_level_does_not_shrink_the_subset(self):
        recs = [rrec(level=lv, target="a") for lv in LEVELS] + [rrec(model="m2", level="L1", target="a")]
        assert four_level_same_operator(recs) == {("c1", "c1:a")}
        assert four_level_same_operator([r for r in recs if r["level"] != "L2"]) == set()

    def test_csv_has_a_task_column_and_repair_only_rates(self):
        recs = [rrec(level="L0", target="a", unchanged=True, category="test_fail"), rrec(level="L1", target="a")]
        cells = metrics.summarize_all(recs, reply_sharing=False) + metrics.summarize(recs, reply_sharing=False)
        gen = [grec(level="L0"), grec(level="L1", category="test_fail")]
        text = metrics.csv_text(cells + metrics.summarize_all(gen),
                                metrics.drop_vs_baseline(recs, reply_sharing=False) + metrics.drop_vs_baseline(gen))
        rows = list(csv.DictReader(io.StringIO(text)))
        assert list(rows[0]) == metrics.CSV_COLUMNS and metrics.CSV_COLUMNS[-3:] == ["task", "unchanged_rate", "restored_rate"]
        rep_all = {r["level"]: r for r in rows if r["task"] == "repair" and r["corpus"] == "ALL"}
        gen_all = {r["level"]: r for r in rows if r["task"] == "generate" and r["corpus"] == "ALL"}
        assert float(rep_all["L0"]["unchanged_rate"]) == 1.0 and rep_all["L0"]["restored_rate"] == "0.0000"
        assert gen_all["L0"]["unchanged_rate"] == "" and gen_all["L1"]["delta_pass_vs_L0"] == "-1.0000"
        assert rep_all["L1"]["delta_pass_vs_L0"] == "1.0000"                 # 两类任务的 delta 不串


# ----------------------------------------------------------------------
# report
# ----------------------------------------------------------------------
def write_obf(ws: Workspace, corpus: str, level: str, tree="tree0", spec="spec0", gate="gate0") -> None:
    store.write_json(ws.obf_file(corpus, level), {
        "schema_version": SCHEMA_VERSION, "corpus": corpus, "level": level, "tree_hash": tree, "ok": True,
        "inputs": {"spec": spec}, "rename_map": {"a": "v_1"}, "strict_stringify": False,
        "oracle_gate": {"checked": 3, "passed": 3, "failed": [], "copy_invalid": False, "input_hash": gate}})


def stamp_repair(r: dict, *, mutants="mut0", prompt=None, reply="reply") -> dict:
    r = dict(r)
    r["schema_version"] = SCHEMA_VERSION
    tid, idx = r["task_id"], r["sample_idx"]
    r["inputs"] = {"spec": "spec0", "tree": "tree0", "prompt": prompt or REPAIR_PROMPT_DIGEST[:16], "gen": f"gen:{tid}:{idx}",
                   "messages": f"msg:{tid}", "reply": store.digest(reply), "gate": "gate0", "code": code_digest("evaluate"),
                   "mutants": mutants, "mutate_code": code_digest("mutate")}
    return r


def stamp_generate(r: dict) -> dict:
    r = dict(r)
    r["schema_version"] = SCHEMA_VERSION
    tid, idx = r["task_id"], r["sample_idx"]
    r["inputs"] = {"spec": "spec0", "tree": "tree0", "prompt": PROMPT_DIGEST[:16], "gen": f"gen:{tid}:{idx}",
                   "messages": f"msg:{tid}", "reply": store.digest("reply"), "gate": "gate0", "code": code_digest("evaluate")}
    return r


def put_repair(ws: Workspace, tag: str, records: list[dict], mutants="mut0", *, summary=True) -> None:
    """写修复任务的结果文件，并让上游文件（repair 生成记录、tasks-repair.jsonl、mutate 汇总）与之对得上。"""
    store.write_jsonl(ws.repair_results_file(tag), records)
    gens: dict[str, list[dict]] = {}
    tasks: dict[str, dict] = {}
    for r in records:
        inp = r["inputs"]
        gens.setdefault(r["model"], []).append({"task_id": r["task_id"], "sample_idx": r["sample_idx"],
                                                "input_hash": inp["gen"], "text": "reply", "model": r["model"]})
        tasks[r["task_id"]] = {"task_id": r["task_id"], "corpus": r["corpus"], "level": r["level"],
                               "messages_hash": inp["messages"]}
    for m, rows in gens.items():
        store.write_jsonl(ws.repair_generations_file(m), rows)
    store.write_jsonl(ws.repair_tasks_file, list(tasks.values()))
    if summary:
        for c in {r["corpus"] for r in records}:
            store.write_json(ws.mutants_summary_file(c), {"levels": {lv: {"input_hash": mutants} for lv in LEVELS}})


def put_generate(ws: Workspace, tag: str, records: list[dict]) -> None:
    store.write_jsonl(ws.results_file(tag), records)
    gens: dict[str, list[dict]] = {}
    tasks: dict[str, dict] = {}
    for r in records:
        inp = r["inputs"]
        gens.setdefault(r["model"], []).append({"task_id": r["task_id"], "sample_idx": r["sample_idx"],
                                                "input_hash": inp["gen"], "text": "reply", "model": r["model"]})
        tasks[r["task_id"]] = {"task_id": r["task_id"], "corpus": r["corpus"], "level": r["level"],
                               "messages_hash": inp["messages"]}
    for m, rows in gens.items():
        store.write_jsonl(ws.generations_file(m), rows)
    store.write_jsonl(ws.tasks_file, list(tasks.values()))


def repair_records() -> list[dict]:
    """两个模型 × 四级 × 两个语料 × 6 个目标；模型 B 的通过率随级别下降。

    每个目标四级同算子，除了 t5 在 L2 换了算子（同时不是同一个位点）；t4 在 L3 算子没变但不是同一个位点；t3 在 L1 的提示词上下文被裁掉了条目。
    所以每个语料里：四级齐全 6 个，同算子 5 个（t5 不算），同位点 4 个（t4、t5 不算），上下文没被裁的 5 个（t3 不算），可比子集 3 个（t0 t1 t2）。
    t4 的变异体是崩溃型杀死（kill_mode），t1 与 t4 的目标被零值桩杀死的方式是崩溃（stub_kill_mode）：两个口径不一样。"""
    out = []
    for model, p in (("model-a", {"L0": 6, "L1": 6, "L2": 5, "L3": 5}), ("model-b", {"L0": 5, "L1": 4, "L2": 3, "L3": 2})):
        for lv in LEVELS:
            for c in ("c1", "c2"):
                for i in range(6):
                    ok = i < p[lv]
                    op = "NEG" if (i == 5 and lv == "L2") else ("ROR" if i % 2 == 0 else "DST")
                    out.append(stamp_repair(rrec(
                        model=model, level=lv, corpus=c, target=f"t{i}", op=op,
                        category="pass" if ok else "test_fail", kill_mode="runtime_crash" if i == 4 else "test_fail",
                        stub_kill_mode="runtime_crash" if i in (1, 4) else "test_fail",
                        same_site=not ((i == 5 and lv == "L2") or (i == 4 and lv == "L3")),
                        orphans=["helper"] if (i == 3 and lv == "L1") else (),
                        unchanged=(not ok and i == 3), restored=ok)))
    return out


class TestValidateRepairRecords:
    @pytest.fixture()
    def ws(self, scratch):
        ws = Workspace.under(scratch).ensure()
        write_obf(ws, "c1", "L0")
        return ws

    def good(self, ws, **kw):
        r = stamp_repair(rrec(), **kw)
        put_repair(ws, "t", [r])
        return r

    def test_matching_record_is_valid(self, ws):
        r = self.good(ws)
        v = report.validate_records([r], ws, task="repair")
        assert len(v.valid) == 1 and not v.stale

    @pytest.mark.parametrize("mutate, reason", [
        (lambda r: r["inputs"].update(prompt=PROMPT_DIGEST[:16]), "prompt 模板已变"),          # 生成任务的摘要不能冒充修复任务的
        (lambda r: r["inputs"].update(mutants="other"), "变异体已重做"),
        (lambda r: r["inputs"].update(mutate_code="old"), "mutate 相关代码已变"),               # 汇总文件里的哈希是 mutate 当时存的，要拿当前代码重算
        (lambda r: r["inputs"].pop("mutants"), "记录缺少输入哈希字段"),
        (lambda r: r["inputs"].pop("mutate_code"), "记录缺少输入哈希字段"),
        (lambda r: r["inputs"].update(gate="other"), "oracle gate 已重跑"),
        (lambda r: r["inputs"].update(tree="other"), "混淆副本已变"),
        (lambda r: r["inputs"].update(gen="other"), "回复已变"),
        (lambda r: r["inputs"].update(messages="other"), "提示词 / 功能规格已变"),
    ])
    def test_each_kind_of_mismatch_is_reported(self, ws, mutate, reason):
        r = self.good(ws)
        mutate(r)
        v = report.validate_records([r], ws, task="repair")
        assert not v.valid and v.stale[0][1].startswith(reason)

    def test_mutate_code_is_compared_with_the_current_digest_not_with_a_stored_one(self, ws, monkeypatch):
        r = self.good(ws)
        assert report.validate_records([r], ws, task="repair").valid == [r]
        monkeypatch.setattr(report, "code_digest", lambda stage: "changed" if stage == "mutate" else code_digest(stage))
        v = report.validate_records([r], ws, task="repair")
        assert not v.valid and v.stale[0][1].startswith("mutate 相关代码已变")
        # 生成任务的记录不看 mutate 代码
        assert report.validate_records([dict(r, inputs={k: v for k, v in r["inputs"].items() if k != "mutate_code"})], ws).stale[0][1] \
            == "prompt 模板已变"

    def test_the_mutants_hash_comes_from_the_mutate_summary(self, ws):
        r = self.good(ws)
        store.write_json(ws.mutants_summary_file("c1"), {"levels": {"L0": {"input_hash": "after-rerun"}}})
        assert report.validate_records([r], ws, task="repair").stale[0][1].startswith("变异体已重做")
        ws.mutants_summary_file("c1").unlink()
        assert report.validate_records([r], ws, task="repair").stale[0][1].startswith("变异体已重做")

    def test_repair_records_read_the_repair_files(self, ws):
        r = self.good(ws)
        ws.repair_generations_file("m1").unlink()
        assert report.validate_records([r], ws, task="repair").stale[0][1] == "对应的生成记录已不存在"
        put_repair(ws, "t", [r])
        store.write_jsonl(ws.repair_tasks_file, [])
        assert report.validate_records([r], ws, task="repair").stale[0][1].startswith("任务已不在 tasks-repair.jsonl")


class TestCombinedReport:
    @pytest.fixture()
    def ws(self, scratch):
        ws = Workspace.under(scratch).ensure()
        for c in ("c1", "c2"):
            for lv in LEVELS:
                write_obf(ws, c, lv)
        return ws

    def gen_records(self):
        return [stamp_generate(grec(model=m, level=lv, corpus=c, target=f"t{i}", category="pass" if i < 4 else "test_fail"))
                for m in ("model-a", "model-b") for lv in LEVELS for c in ("c1", "c2") for i in range(6)]

    def test_both_tasks_are_summarised_side_by_side(self, ws):
        put_generate(ws, "t", self.gen_records())
        put_repair(ws, "t", repair_records())
        res = report.build_report(ws, "t")
        assert (res.n_records, res.n_used, res.n_stale) == (96 + 96, 96 + 96, 0)
        assert res.by_task == {"generate": {"records": 96, "used": 96, "stale": 0},
                               "repair": {"records": 96, "used": 96, "stale": 0}}
        md = res.paths["md"].read_text(encoding="utf-8")
        # 开头说明两类任务的输入差异
        head = md.split("## 1.")[0]
        assert "**两类任务的输入差异**" in head and all(line in head for line in TASK_INPUT_LINES)
        assert "- 生成任务：样本记录 96 条" in head and "- 修复任务：样本记录 96 条" in head
        for heading in ("## 1. 总表", "## 2. 相对 L0 的变化", "## 3. 失败归因分布", "## 4. 分语料明细",
                        "## 5. 缺陷修复任务（repair）", "## 6. 可信度记录", "## 7. 局限", "## 8. 来源"):
            assert heading in md
        section = md.split("## 5. 缺陷修复任务（repair）")[1].split("## 6.")[0]
        for sub in ("### 5.1 总表", "### 5.2 生成与修复并列对照", "### 5.3 相对 L0 的变化", "### 5.4 可比子集（四级是同一个缺陷）",
                    "### 5.5 失败归因分布", "### 5.6 按变异体怎么死的（kill_mode）分层的通过率",
                    "### 5.7 按目标被零值桩杀死的方式（stub_kill_mode）分层的通过率", "### 5.8 按提示词上下文有没有被裁条目分层的通过率",
                    "### 5.9 按变异算子分层的通过率", "### 5.10 分语料明细"):
            assert sub in section
        # 修复任务的总表：model-b L3 通过 2/6，unchanged 与 restored 率
        row = next(ln for ln in section.splitlines() if ln.startswith("| model-b | L3 | 12 |"))
        cols = [c.strip() for c in row.strip("|").split("|")]
        head_cols = [c.strip() for c in next(ln for ln in section.splitlines() if ln.startswith("| 模型 | 级别 | n | 样本完整性")).strip("|").split("|")]
        got = dict(zip(head_cols, cols))
        assert got["通过率"] == "33.3%" and got["restored 率"] == "33.3%" and got["unchanged 率"] == "16.7%"      # 4/12 通过；4/12 restored；2/12 unchanged
        assert "| 回复与 L1 相同 |" not in section and "回复与 L1 逐字相同的配对数" not in section
        # 并列对照里有两类任务的通过率
        pair = next(ln for ln in section.splitlines() if ln.startswith("| model-a | L0 |") and "66.7%" in ln)
        assert "100.0%" in pair
        # 局限一节是同一份，且新增的几条都在
        lim = md.split("## 7. 局限")[1].split("## 8.")[0]
        for title, text in LIMITATIONS:
            assert f"**{title}。**" in lim and text in lim
        for title in ("崩溃型与超时型杀死的修复任务，测试可能不检查返回内容", "修复任务的提示词上下文被裁剪过",
                      "修复任务的相似度和 restored 率不是记忆指标", "NEG 缺陷的显眼程度随级别变化"):
            assert any(title == t for t, _ in LIMITATIONS) and f"**{title}。**" in lim

    def test_the_comparable_subset_is_same_site_and_clean_context_not_same_operator(self, ws):
        put_repair(ws, "t", repair_records())
        md = report.build_report(ws, "t", formats=("md",), task="repair").paths["md"].read_text(encoding="utf-8")
        section = md.split("### 1.3 可比子集（四级是同一个缺陷）")[1].split("### 1.4")[0]
        # 每个语料：同算子 5（t5 在 L2 换了算子）、同位点 4（t4 在 L3、t5 在 L2 不是同一个位点）、上下文没被裁 5（t3 在 L1 被裁）、可比 3
        assert "修复任务涉及的目标 12 个；四级（L0–L3）都出了题的 12 个" in section
        assert "四级算子完全相同的 10 个" in section and "不构成对照" in section
        assert "四级是同一个位点的（L1–L3 的缺陷都是 L0 那个位点在本级的对应位点）8 个" in section
        assert "四级的提示词上下文都没有被裁掉条目的 10 个" in section
        assert "可比子集 = 同一个位点 ∩ 上下文没有被裁条目，共 6 个" in section
        assert "最可比" not in md
        row = next(ln for ln in section.splitlines() if ln.startswith("| model-b | L0 |"))
        assert "| 6 | 6 |" in row                                          # n 与目标数都是 6（3 个目标 × 2 个语料，一个样本编号）
        row3 = next(ln for ln in section.splitlines() if ln.startswith("| model-b | L3 |"))
        assert "| 66.7% |" in row3 and "-33.3" in row3                    # 可比子集里 model-b 在 L3 通过 4/6，Δ 是相对 L0 的配对差

    def test_pairs_of_the_same_defect_only_table(self, ws):
        put_repair(ws, "t", repair_records())
        md = report.build_report(ws, "t", formats=("md",), task="repair").paths["md"].read_text(encoding="utf-8")
        section = md.split("### 1.3 可比子集（四级是同一个缺陷）")[1].split("### 1.4")[0]
        assert "只留「本级的缺陷与 L0 是同一个位点" in section
        table = section.split("只留「本级的缺陷与 L0 是同一个位点")[1]
        rows = {tuple(c.strip() for c in ln.strip("|").split("|")[:3]): ln for ln in table.splitlines() if ln.startswith("| model-a |")}
        # L1：t3 的 L1 上下文被裁 -> 12 个目标里少 2 个 = 10；L2：t5 不是同一个位点 -> 10；L3：t4 -> 10
        assert set(rows) == {("model-a", "L1", "10"), ("model-a", "L2", "10"), ("model-a", "L3", "10")}

    def test_operator_and_kill_mode_strata_are_two_different_axes(self, ws):
        put_repair(ws, "t", repair_records())
        md = report.build_report(ws, "t", formats=("md",), task="repair").paths["md"].read_text(encoding="utf-8")
        ops = md.split("### 1.8 按变异算子分层的通过率")[1].split("### 1.9")[0]
        assert "| model-a | ROR（关系算子替换） |" in ops and "| model-a | DST（删语句） |" in ops
        assert "| model-a | NEG（条件取反） |" in ops                       # L2 里 t5 换成了 NEG
        assert "NEG 的难度随级别变化有一部分与混淆无关" in ops and "`!(c)`" in ops
        modes = md.split("### 1.5 按变异体怎么死的（kill_mode）分层的通过率")[1].split("### 1.6")[0]
        assert "| model-a | L0 | runtime_crash | 2 | 2 | 100.0% |" in modes          # t4：2 个语料
        assert "这个具体缺陷" in modes and "不等于" in modes and "任何不崩溃的错误实现" not in modes
        stubs = md.split("### 1.6 按目标被零值桩杀死的方式（stub_kill_mode）分层的通过率")[1].split("### 1.7")[0]
        assert "| model-a | L0 | runtime_crash | 4 | 4 | 100.0% |" in stubs          # t1、t4 × 2 个语料
        assert "任何不崩溃的错误实现都能通过" in stubs and "测试有没有断言返回内容" in stubs
        # 同一批记录按两个口径切出来是两张不同的表：同一格（model-b L1 的崩溃型）两边的 n 和通过数都不一样
        assert "| model-b | L1 | runtime_crash | 2 | 0 | 0.0% |" in modes
        assert "| model-b | L1 | runtime_crash | 4 | 2 | 50.0% |" in stubs

    def test_context_strata_and_the_note(self, ws):
        put_repair(ws, "t", repair_records())
        md = report.build_report(ws, "t", formats=("md",), task="repair").paths["md"].read_text(encoding="utf-8")
        ctx = md.split("### 1.7 按提示词上下文有没有被裁条目分层的通过率")[1].split("### 1.8")[0]
        # model-a L1：t3 被裁（2 个语料）且通过 -> 有被裁条目 2/2；其余 10 个也都通过
        assert "| model-a | L1 | 10/10（100.0%） | 2/2（100.0%） |" in ctx
        assert "| model-a | L0 | 12/12（100.0%） | - |" in ctx
        assert "不进可比子集" in ctx and "context_orphans" in ctx

    def test_task_selection(self, ws):
        put_generate(ws, "t", self.gen_records())
        put_repair(ws, "t", repair_records())
        only_gen = report.build_report(ws, "t", formats=("md", "csv"), task="generate")
        assert only_gen.n_records == 96 and set(only_gen.by_task) == {"generate"}
        md = only_gen.paths["md"].read_text(encoding="utf-8")
        assert "（repair）" not in md and "两类任务的输入差异" not in md and "缺陷修复任务的变异体（mutate）" not in md
        only_rep = report.build_report(ws, "t", formats=("md", "csv"), task="repair")
        assert only_rep.n_records == 96 and set(only_rep.by_task) == {"repair"}
        md = only_rep.paths["md"].read_text(encoding="utf-8")
        assert "## 1. 缺陷修复任务（repair）" in md and "## 1. 总表" not in md and "## 2. 局限" not in md
        assert "## 2. 可信度记录" in md and "## 3. 局限" in md and "## 4. 来源" in md
        rows = list(csv.DictReader(only_rep.paths["csv"].open(encoding="utf-8")))
        assert {r["task"] for r in rows} == {"repair"}
        both = report.build_report(ws, "t", formats=("csv",))
        assert {r["task"] for r in csv.DictReader(both.paths["csv"].open(encoding="utf-8"))} == {"generate", "repair"}

    def test_a_missing_repair_file_leaves_the_generate_report_untouched(self, ws):
        put_generate(ws, "t", self.gen_records())
        with_all = report.build_report(ws, "t", formats=("md",)).paths["md"].read_text(encoding="utf-8")
        only_gen = report.build_report(ws, "t", formats=("md",), task="generate").paths["md"].read_text(encoding="utf-8")
        assert with_all == only_gen and "（repair）" not in with_all and "两类任务的输入差异" not in with_all
        none = report.build_report(ws, "nothing", task="repair")
        assert none.n_records == 0 and "没有可用的评测记录" in none.paths["md"].read_text(encoding="utf-8")

    def test_stale_repair_records_are_excluded_and_listed_separately(self, ws):
        put_repair(ws, "t", repair_records())
        recs = repair_records()
        recs[0]["inputs"]["mutants"] = "old"
        put_repair(ws, "t", recs)
        res = report.build_report(ws, "t", formats=("md",))
        assert (res.n_records, res.n_used, res.n_stale) == (96, 95, 1)
        assert list(res.stale_reasons) == ["修复任务：变异体已重做（mutate 重跑过，评测结果对应的是旧变异体）"]
        md = res.paths["md"].read_text(encoding="utf-8")
        assert "输入校验（修复任务）" in md and "已排除，不进统计" in md
        assert report.build_report(ws, "t", formats=("md",), allow_stale=True).n_used == 96

    def test_unchanged_candidates_that_pass_remove_the_whole_task_and_are_named(self, ws):
        recs = repair_records()
        victim = next(r for r in recs if r["model"] == "model-a" and r["level"] == "L2" and r["target_id"] == "c1:t1")
        victim.update(unchanged=True, category="pass")                         # model-a 什么也没改却通过了：这道题（所有模型）都不可靠
        put_repair(ws, "t", recs)
        res = report.build_report(ws, "t", formats=("md",))
        assert res.n_unreliable == 1 and res.n_records == 96 and res.n_used == 96 - 2        # 同一道题 model-b 的记录一起剔除
        md = res.paths["md"].read_text(encoding="utf-8")
        head = md.split("## 1.")[0]
        assert "**警告：1 道修复题的变异体不可靠**" in head and "c1:c1:t1@L2#repair" in head and "check-mutants" in head
        assert "- 修复任务：样本记录 94 条" in md
        row = next(ln for ln in md.splitlines() if ln.startswith("| model-a | L2 | 11 |"))      # 12 个 - 1
        assert row
        # 其余级别不受影响，剔除的是题而不是目标：同一个目标的 L0 / L1 / L3 还在
        assert any(ln.startswith("| model-a | L1 | 12 |") for ln in md.splitlines())

    def test_the_report_command_prints_how_many_unreliable_tasks_were_dropped(self, ws, capsys):
        recs = repair_records()
        next(r for r in recs if r["model"] == "model-a" and r["level"] == "L2" and r["target_id"] == "c1:t1").update(
            unchanged=True, category="pass")
        put_repair(ws, "t", recs)
        assert cli.main(["report", "--tag", "t", "--task", "repair", "--format", "md", "--workspace", str(ws.results.parent)]) == 0
        out = capsys.readouterr().out
        assert "1 道修复题的变异体不可靠" in out and "整题从统计里剔除" in out

    def test_bad_lines_and_missing_samples_from_both_tasks_are_called_out(self, ws):
        put_generate(ws, "t", self.gen_records())
        put_repair(ws, "t", repair_records())
        with ws.repair_results_file("t").open("a", encoding="utf-8") as f:
            f.write('{"half": ')
        store.write_json(ws.repair_completeness_file("t"), {"cells": [
            {"model": "model-a", "level": "L1", "corpus": "c1", "expected": 8, "have": 6, "failed_calls": 2,
             "stale_generations": 0, "not_generated": 0, "missing": 2}]})
        res = report.build_report(ws, "t", formats=("md",))
        assert res.n_bad_lines == 1 and res.n_missing == 2
        md = res.paths["md"].read_text(encoding="utf-8")
        head = md.split("## 1.")[0]
        assert "1 条坏行" in head and "生成任务 0 条，修复任务 1 条" in head and "缺 2 个" in head
        assert "不完整（缺 2/8）" in md.split("\n## 5. ")[1]

    def test_svg_draws_a_line_per_task_and_model(self, ws):
        put_generate(ws, "t", self.gen_records())
        put_repair(ws, "t", repair_records())
        svg = report.build_report(ws, "t", formats=("svg",)).paths["svg"].read_text(encoding="utf-8")
        ET.fromstring(svg)
        for name in ("model-a（生成）", "model-a（修复）", "model-b（生成）", "model-b（修复）"):
            assert name in svg

    def test_provenance_lists_both_prompt_digests_and_the_mutate_code(self, ws):
        put_generate(ws, "t", self.gen_records())
        put_repair(ws, "t", repair_records())
        md = report.build_report(ws, "t", formats=("md",)).paths["md"].read_text(encoding="utf-8")
        src = md.split("## 8. 来源")[1]
        assert PROMPT_DIGEST[:16] in src and f"修复任务 {REPAIR_PROMPT_DIGEST[:16]}" in src
        assert f"变异体（mutate）相关代码摘要 {code_digest('mutate')}" in src
        assert "模型 model-a：" in src and "模型 model-a（修复）：" in src

    def test_mutate_summary_and_repair_task_counts_show_up_in_the_credibility_section(self, ws):
        put_repair(ws, "t", repair_records())
        ent = {"input_hash": "mut0", "n_targets": 6, "n_searched": 6, "n_killed": 5, "n_abandoned": 1,
               "operators": {"ROR": 3, "DST": 2}, "kill_modes": {"test_fail": 4, "runtime_crash": 1},
               "same_operator_as_L0": {"n": 5, "same": 4}, "same_site_as_L0": {"n": 5, "same": 3}, "unstable_count": 2,
               "outcomes": {"killed": 5, "gave_up": 1}}
        store.write_json(ws.mutants_summary_file("c1"), {"levels": {lv: ent for lv in LEVELS}})
        store.append_jsonl(ws.repair_tasks_file, {"task_id": "extra", "corpus": "c1", "level": "L0", "messages_hash": "x"})
        md = report.build_report(ws, "t", formats=("md",), task="repair").paths["md"].read_text(encoding="utf-8")
        assert "### 缺陷修复任务的变异体（mutate）" in md
        row = next(ln for ln in md.splitlines() if ln.startswith("| c1 | L1 | 6 | 6 | 5 | 1 | 83.3% |"))
        assert "ROR=3, DST=2" in row and "| 4/5 | 3/5 |" in row and "test_fail=4, runtime_crash=1" in row and "| 2 |" in row and "gave_up=1" in row
        assert "修复任务出题数" in md and "与 L0 同位点" in md and "复核不稳定" in md

    def test_generated_text_has_no_banned_words(self, ws):
        put_generate(ws, "t", self.gen_records())
        put_repair(ws, "t", repair_records())
        res = report.build_report(ws, "t")
        for path in res.paths.values():
            text = path.read_text(encoding="utf-8")
            assert not [w for w in BANNED if w in text], path.name


# ----------------------------------------------------------------------
# 实验记录里的 mutate 一节
# ----------------------------------------------------------------------
def summary_for(killed=(5, 5, 5, 4), same=(None, (5, 5), (5, 5), (4, 5)), site=(None, (5, 5), (4, 5), (3, 5))) -> dict:
    levels = {}
    for lv, k, sm, st in zip(LEVELS, killed, same, site):
        levels[lv] = {
            "input_hash": f"h-{lv}", "n_targets": 5, "n_searched": 5, "n_killed": k, "n_abandoned": 5 - k,
            "outcomes": {"killed": k, **({"exhausted": 5 - k} if k < 5 else {})},
            "operators": {op: (k if op == "ROR" else 0) for op in MUTATION_OPERATORS},
            "kill_modes": {"test_fail": k - 1, "runtime_crash": 1},
            "same_operator_as_L0": {"n": sm[1], "same": sm[0]} if sm else {"n": 0, "same": 0},
            "same_site_as_L0": {"n": st[1], "same": st[0]} if st else {"n": 0, "same": 0},
            "anchor": ({"none": k} if lv == "L0" else {"chosen": st[0], "no_image": st[1] - st[0]}),
            "n_sites": 40, "tried_sites": 7, "equivalent_count": 2, "compile_fail_count": 1, "parse_fail_count": 0,
            "unstable_count": 1, "n_rechecked": k, "n_kill_label_changed": 0,
            "sandbox_seconds": {"test_fail": 8.0, "runtime_timeout": 30.0, "pass": 2.0},
            "n_with_comments": 1, "n_strip_checked": 1, "n_strip_failed": 0}
    return {"corpus": "mini", "n_targets": 5, "elapsed_s": 12.34, "levels": levels,
            "four_level": {"n_targets": 5, "n_all_levels": 4, "n_same_operator": 3, "n_same_site": 2}}


def mutant_rec(target: str, level: str, op="ROR", idx=0, kill_mode="test_fail", orphans=(), code="int f(void) { return 0; }") -> dict:
    return {"target_id": f"mini:calc.c:{target}", "level": level, "operator": op, "site_index": idx, "kill_mode": kill_mode,
            "context_orphans": list(orphans), "mutant_code": code}


class TestExperimentRecordMutateSection:
    @pytest.fixture()
    def data(self):
        return {"mini": {"mutants": summary_for()}, "other": {"mutants": None}}

    def test_tables_state_counts_rates_and_time(self, data):
        text = "\n".join(er._mutants(data))
        assert "| mini | 5 | 5 / 5 / 5 / 4 | 19 | 1 | exhausted=1 | 95.0% | 4 | 3 | 2 | 4 / 4 / 0 | 12.3 |" in text
        assert "| mini | 19 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 19 |" in text                      # 算子分布：ROR 19，其余 0
        assert "| mini | 5/5（100.0%） | 5/5（100.0%） | 4/5（80.0%） |" in text            # 同算子
        assert "| mini | 5/5（100.0%） | 4/5（80.0%） | 3/5（60.0%） |" in text            # 同位点
        assert "| mini | 4 / 1 / 0 | 4 / 1 / 0 | 4 / 1 / 0 | 3 / 1 / 0 |" in text
        assert "| mini | 160 | 28 | 8 | 4 | 0 | 19 | 4 | 0 |" in text                        # 记账：位点 / 试 / 等价 / 编译失败 / 解析失败 / 复核 / 不稳定 / 方式变了
        assert "| other | 缺 |" in text                                                       # 没有 mutate 产出的语料标「缺」，不是 0
        assert text.count("**合计**") >= 6 and "(test_fail / runtime_crash / runtime_timeout)" not in text
        assert " / ".join(KILL_MODES) in text

    def test_anchor_fate_and_sandbox_time_shares_come_from_the_summary(self, data):
        text = "\n".join(er._mutants(data))
        assert "chosen=5" in text and "chosen=3, no_image=2" in text
        row = next(ln for ln in text.splitlines() if ln.startswith("| runtime_timeout |"))
        assert row == "| runtime_timeout | 120.0 | 75.0% |"                                  # 4 级 × 30 s，占 160 s 的 75%
        assert "沙箱运行时间合计 160.0 秒" in text

    def test_same_operator_index_ratio_and_the_cross_table_come_from_the_records(self):
        d = {"mutants": summary_for(),
             "targets": [{"target_id": f"mini:calc.c:{t}", "stub_kill_mode": m}
                         for t, m in (("a", "test_fail"), ("b", "test_fail"), ("c", "runtime_crash"), ("d", "runtime_crash"))],
             "mutant_records": {
                 "L0": [mutant_rec("a", "L0", "ROR", 0), mutant_rec("b", "L0", "DST", 2, "runtime_crash"), mutant_rec("c", "L0", "RVR", 1),
                        mutant_rec("d", "L0", "NEG", 3, "runtime_crash")],
                 "L1": [mutant_rec("a", "L1", "ROR", 0), mutant_rec("b", "L1", "DST", 2, "runtime_crash"), mutant_rec("c", "L1", "RVR", 1),
                        mutant_rec("d", "L1", "NEG", 3, "runtime_crash")],
                 "L2": [mutant_rec("a", "L2", "ROR", 5), mutant_rec("b", "L2", "DST", 2, "runtime_crash"), mutant_rec("c", "L2", "RVR", 1)],
                 "L3": []}}
        text = "\n".join(er._site_tables({"mini": d}) + er._kill_tables({"mini": d}))
        assert "| mini | 4/4（100.0%） | 2/3（66.7%） | - |" in text                          # L1 精确；L2 的 a 序号漂了；L3 没有记录
        # a：桩 test_fail、变异体 test_fail ×3；b：桩 test_fail、变异体崩溃 ×3；c：桩崩溃、变异体 test_fail ×3；d：桩崩溃、变异体崩溃 ×2
        assert "| test_fail | 3 | 3 | 0 | 6 |" in text and "| runtime_crash | 3 | 2 | 0 | 5 |" in text
        assert "目标的桩是 test_fail 的 6 个变异体里，有 3 个（50.0%）是崩溃 / 超时杀死的" in text
        assert "目标的桩是崩溃 / 超时的 5 个变异体里，有 3 个（60.0%）记为 test_fail（共 11 个变异体）" in text

    def test_orphan_table_counts_per_operator_and_level(self):
        d = {"mutants": summary_for(), "mutant_records": {
            "L0": [mutant_rec("a", "L0", "DST", orphans=["helper"]), mutant_rec("b", "L0", "ROR")],
            "L1": [mutant_rec("a", "L1", "DST", orphans=["f_1"]), mutant_rec("b", "L1", "ROR")], "L2": [], "L3": []}}
        text = "\n".join(er._orphans_table({"mini": d}))
        assert "| DST（删语句） | 2 | 2 | 100.0% |" in text and "| ROR（关系算子替换） | 2 | 0 | 0.0% |" in text
        assert "L0 1/2；L1 1/2；L2 0/0；L3 0/0" in text
        assert er._orphans_table({"mini": {"mutants": summary_for(), "mutant_records": {lv: [] for lv in LEVELS}}}) == []

    def test_subset_table_uses_the_same_functions_as_the_report(self):
        def recs(target, ops=("ROR",) * 4, sites=(True,) * 4, orphans=((),) * 4):
            return {lv: mutant_rec(target, lv, op, 0, orphans=o) | {"corpus": "mini", "same_site_as_L0": s}
                    for lv, op, s, o in zip(LEVELS, ops, sites, orphans)}
        per_target = {"a": recs("a"),                                                        # 可比
                      "b": recs("b", sites=(True, True, False, True)),                        # L2 不是同一个位点
                      "c": recs("c", orphans=((), ("h",), (), ())),                           # L1 的上下文被裁了条目
                      "d": recs("d", ops=("ROR", "ROR", "NEG", "ROR"), sites=(True, True, False, True))}
        mutant_records = {lv: [per_target[t][lv] for t in per_target] for lv in LEVELS}
        text = "\n".join(er._subset_table({"mini": {"mutant_records": mutant_records}, "other": {"mutant_records": {}}}))
        # 目标 4；四级齐全 4；同算子 3（d 换了算子）；同位点 2（a、c）；上下文干净 3（c 除外）；可比 1（a）；
        # 配对可比：L1 3（a、b、d；c 的 L1 上下文被裁）/ L2 2（a、c；b、d 的 L2 不是同一个位点）/ L3 4（四个目标 L3 都同位点且干净）
        assert "| mini | 4 | 4 | 3 | 2 | 3 | 1 | 3 / 2 / 4 |" in text
        assert "| other | 缺 | 缺 | 缺 | 缺 | 缺 | 缺 | 缺 |" in text and "| **合计** | 4 | 4 | 3 | 2 | 3 | 1 | 3 / 2 / 4 |" in text

    def test_replay_check_table_lists_each_corpus_and_the_total(self):
        check = {"with_oracle": True, "total": {"replayed": 8, "killed": 8, "same_mode": 7, "passed": 0, "other": 0, "oracle_failed": 0},
                 "flipped": [{"task_id": "x", "recorded": "test_fail", "replay": "runtime_crash"}]}
        text = "\n".join(er._check_table({"mini": {"mutants_check": check}, "other": {"mutants_check": None}}))
        assert "| mini | 8 | 8 | 7 | 0 | 0 | 1 | 0 |" in text and "| **合计** | 8 | 8 | 7 | 0 | 0 | 1 | 0 |" in text
        assert "other" not in text and er._check_table({"mini": {}}) == []

    def test_name_leak_check_separates_code_from_literals_and_common_words_from_domain_words(self):
        rename = {"calc_gcd": "f_1", "buffer": "v_2", "x": "v_3", "Regex": "t_4"}
        recs = {
            "L1": [{"target_id": "mini:calc.c:calc_gcd", "mutant_code": "int f_1(void) { return calc_gcd(1); }"},           # 代码里还有原名：泄漏
                   {"target_id": "mini:calc.c:other", "mutant_code": 'int f_2(void) { puts("buffer is full"); return 0; }'},    # 只在字符串里：通用词
                   {"target_id": "mini:calc.c:other", "mutant_code": 'int f_3(void) { puts("Regex failed"); return \'x\'; }'},    # 领域词；单字符常量不算
                   {"target_id": "mini:calc.c:calc_gcd", "mutant_code": 'int f_4(void) { puts("calc_gcd: bad"); return 0; }'}],   # 目标自己的原名
            "L2": [{"target_id": "mini:calc.c:other", "mutant_code": "int f_5(void) { return 0; }"}],
        }
        data = {"mini": {"obf": {lv: {"rename_map": rename} for lv in ("L1", "L2")}, "mutant_records": recs}}
        checker = type("C", (), {"exempt": frozenset({"buffer", "calc_gcd"})})()          # 放行名单：buffer 是常见英文词；calc_gcd 假装也在里面
        text = "\n".join(er._name_leaks(data, {"mini": checker}))
        # 题数 5；字面量之外含原名 1；字面量里含原名 3（buffer / Regex / calc_gcd）；其中全是通用词 2（buffer、calc_gcd），含领域词 1（Regex）；自己的原名 1
        assert "| mini | 5 | 1 | 3 | 2 | 1 | 1 |" in text and "| **合计** | 5 | 1 | 3 | 2 | 1 | 1 |" in text
        assert "| other | 缺 | 缺 | 缺 | 缺 | 缺 | 缺 |" in "\n".join(er._name_leaks({**data, "other": {}}, {"mini": checker}))
        # 没有检查器（找不到 spec / 源码）时无法区分通用词和领域词，宁可都算领域词
        assert "| mini | 5 | 1 | 3 | 0 | 3 | 1 |" in "\n".join(er._name_leaks(data, {}))

    def test_comment_audit_counts_functions_with_comments_and_the_ones_naming_renamed_identifiers(self, scratch):
        ws = Workspace.under(scratch).ensure()
        src = ("int plain(int a) { return a; }\n"
               "int hinted(int a) { /* hinted wraps helper_fn */ return a; }\n"
               "int quiet(int a) { return a; // nothing special here\n}\n"
               'int literal(void) { return puts("/* helper_fn */"); }\n')
        (ws.pristine("mini")).mkdir(parents=True)
        (ws.pristine("mini") / "x.c").write_text(src, encoding="utf-8")
        data_bytes = src.encode()

        def target(name: str) -> dict:
            start = data_bytes.index(f"int {name}(".encode())
            end = data_bytes.index(b"}", start) + 1 if name != "quiet" else data_bytes.index(b"}\n", start) + 1
            return {"target_id": f"mini:x.c:{name}", "name": name, "file": "x.c", "start": start, "end": end}

        data = {"mini": {"targets": [target(n) for n in ("plain", "hinted", "quiet", "literal")],
                         "obf": {"L1": {"rename_map": {"helper_fn": "f_1", "plain": "f_2"}}}},
                "other": {"targets": [], "obf": {}}}
        text = "\n".join(er._comment_audit(data, ws))
        assert "| mini | 4 | 2 | 1 |" in text and "| **合计** | 4 | 2 | 1 |" in text         # 带注释的 hinted、quiet；提到 helper_fn 的只有 hinted
        assert "| other | 缺 | 缺 | 缺 |" in text                                             # 没有目标 / 改名表：没统计过，不是 0

    def test_the_section_is_omitted_when_no_corpus_has_mutants(self, scratch, monkeypatch):
        from bench import corpus
        import shutil
        from tests.helpers import MINI
        specs = scratch / "specs"
        specs.mkdir()
        shutil.copy(MINI / "spec.json", specs / "mini.json")
        monkeypatch.setattr(corpus, "SPEC_DIR", specs)
        ws = Workspace.under(scratch).ensure()
        shutil.copytree(MINI, ws.pristine("mini"))
        assert "缺陷修复任务的变异体" not in er.render(ws, ["mini"])
        store.write_json(ws.mutants_summary_file("mini"), summary_for())
        text = er.render(ws, ["mini"])
        assert "### 缺陷修复任务的变异体（mutate）" in text and text.index("### 泄漏检查的放行情况") < text.index("### 缺陷修复任务的变异体")
        assert not [w for w in BANNED if w in text]
