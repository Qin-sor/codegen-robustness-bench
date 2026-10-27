"""集成自测：存储 / 指标 / 报告 / 混淆入口 / 出题与生成 / 命令行，以及 mini 夹具上的全链路 e2e（三种 MockLLM 模式）。

分两类（docker 与镜像 gcc:13 不可用时，第二类整组跳过）：

* 不需要沙箱：store 的 JSON / JSONL / 哈希，metrics 的统计口径，report 的输入哈希校验与 Markdown / CSV / SVG 输出，
  ``obfuscate()`` 入口（含 verify 失败后退回 strict 的重试逻辑，用脚本化的假沙箱），出题（作废原因的计数）、
  描述、生成阶段的幂等与断点续跑，真实的 OpenAICompat 客户端对着本机假服务经过 generate 阶段（L1–L3 提示词相同所以共用
  缓存、usage 日志只记真实调用、API key 只出现在请求头里、调用失败不中断且可续跑），命令行参数与错误处理；
* 需要沙箱：``obfuscate()`` 的 verify，oracle gate（用注入的编译错误模拟个别目标过不了），增量重算
  （换了功能规格后旧回复和旧结果作废），用 main() 把每条子命令按顺序走一遍，以及 mini 上 L0–L3 × oracle / broken / wrong
  的全链路 e2e。e2e 的断言是设计文档 §4.15 / §6.3 的验收标准：oracle 100% pass、broken 100% compile_syntax、
  wrong 不通过且归因与 stub-kill 记录的杀死方式一致，加上幂等（重跑全部跳过）、副本 verify 与 oracle gate 全过、
  报告文件齐全且没有过期记录、L1–L3 的提示词里不再有目标函数的真名。

全部输出放在项目的 ``tmp/pytest-e2e/`` 下，测试结束即清理，不碰 ``results/``。
"""
from __future__ import annotations

import csv
import dataclasses
import http.server
import io
import json
import re
import shutil
import subprocess
import sys
import threading
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from bench import corpus, metrics, report, store
from bench.attribute import CATEGORIES
from bench.cli import build_parser, main, run_e2e
from bench.obfuscate import LEVELS, ObfResult, obfuscate
from bench.prompt import PROMPT_DIGEST
from bench.provenance import code_digest
from bench.sandbox import DockerSandbox
from bench.store import SCHEMA_VERSION, Workspace
from tests.helpers import BANNED, MINI, ROOT, needs_docker


# ----------------------------------------------------------------------
# 公共构造器（scratch / mini_spec 夹具在 conftest.py）
# ----------------------------------------------------------------------
def rec(model="m1", level="L0", corpus_name="c1", target="t0", idx=0, stage="pass", category="pass", sim=0.9,
        ptok=100, ctok=20, cached=False, **extra) -> dict:
    """一条合成的评测记录（只含 metrics / report 用到的字段）。"""
    out = {"model": model, "level": level, "corpus": corpus_name, "target_id": f"{corpus_name}:{target}",
           "task_id": f"{corpus_name}:{target}@{level}", "sample_idx": idx, "stage": stage, "category": category,
           "similarity": sim, "prompt_tokens": ptok, "completion_tokens": ctok, "cached": cached}
    out.update(extra)
    return out


# ----------------------------------------------------------------------
# store
# ----------------------------------------------------------------------
class TestStore:
    def test_digest_is_canonical_and_short(self):
        assert store.digest({"a": 1, "b": [1, 2]}) == store.digest({"b": [1, 2], "a": 1})
        assert store.digest({"a": 1}) != store.digest({"a": 2})
        assert len(store.digest("x")) == 16 and len(store.digest("x", 8)) == 8
        assert store.digest("中文") == store.digest("中文")      # 不因编码丢信息

    def test_spec_digest_for_spec_and_dict(self, mini_spec):
        assert store.spec_digest(mini_spec) == mini_spec.digest()[:16]
        a, b = corpus.CorpusSpec(name="x", cflags=["-O0"]), corpus.CorpusSpec(name="x", cflags=["-O2"])
        assert store.spec_digest(a) != store.spec_digest(b)
        assert store.spec_digest({"name": "x", "cflags": ["-O0"]}) == store.spec_digest({"cflags": ["-O0"], "name": "x"})
        # 不影响构建与测试的字段（notes / url / license / allow_words）不进哈希
        assert store.spec_digest(a) == store.spec_digest(dataclasses.replace(a, notes="补一句备注", license="BSD"))

    def test_safe_name(self):
        assert store.safe_name("deepseek-v4-pro") == "deepseek-v4-pro"
        assert store.safe_name("a/b c:d") == "a_b_c_d"
        assert store.safe_name("///") == "_"

    def test_read_jsonl_skips_blank_and_broken_lines(self, scratch):
        p = scratch / "x.jsonl"
        p.write_text('{"a": 1}\n\n{"a": 2\n{"a": 3}\n[1, 2]\n', encoding="utf-8")
        assert store.read_jsonl(p) == [{"a": 1}, {"a": 3}]
        assert store.read_jsonl(scratch / "missing.jsonl") == []

    def test_read_json_missing_or_corrupt_is_none(self, scratch):
        assert store.read_json(scratch / "nope.json") is None
        (scratch / "bad.json").write_text("{oops", encoding="utf-8")
        assert store.read_json(scratch / "bad.json") is None

    def test_write_json_roundtrip_is_atomic(self, scratch):
        p = store.write_json(scratch / "d" / "a.json", {"中": [1, 2.5, None]})
        assert store.read_json(p) == {"中": [1, 2.5, None]}
        assert "中" in p.read_text(encoding="utf-8")                  # 不转义中文
        assert [f.name for f in p.parent.iterdir()] == ["a.json"]    # 没有残留的临时文件

    def test_append_jsonl_is_thread_safe(self, scratch):
        p = scratch / "log.jsonl"

        def work(k: int) -> None:
            for i in range(50):
                store.append_jsonl(p, {"k": k, "i": i, "pad": "x" * 200})

        threads = [threading.Thread(target=work, args=(k,)) for k in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        rows = store.read_jsonl(p)
        assert len(rows) == 400 and len({(r["k"], r["i"]) for r in rows}) == 400

    def test_append_jsonl_repairs_a_half_written_last_line(self, scratch):
        # 进程被杀时可能留下没有换行的半行；下一条记录接在它后面会被 read_jsonl 连同半行一起当坏行丢掉
        p = scratch / "log.jsonl"
        store.append_jsonl(p, {"id": 1})
        with p.open("a", encoding="utf-8") as f:
            f.write('{"id": 2, "tru')                                   # 半行，没有换行
        store.append_jsonl(p, {"id": 3})
        assert [r["id"] for r in store.read_jsonl(p)] == [1, 3]         # id=3 没有被半行拖累
        store.append_jsonl(p, {"id": 4})
        assert [r["id"] for r in store.read_jsonl(p)] == [1, 3, 4]

    def test_read_jsonl_checked_counts_bad_lines(self, scratch):
        p = scratch / "x.jsonl"
        p.write_text('{"a": 1}\n{oops\n[1]\n\n{"a": 2}\n', encoding="utf-8")
        assert store.read_jsonl_checked(p) == ([{"a": 1}, {"a": 2}], 2)
        assert store.read_jsonl_checked(scratch / "none.jsonl") == ([], 0)

    def test_lone_surrogates_do_not_break_writing(self, scratch):
        bad = "ok \ud83d tail"
        store.append_jsonl(scratch / "a.jsonl", {"text": bad})
        store.write_jsonl(scratch / "b.jsonl", [{"text": bad}])
        store.write_json(scratch / "c.json", {"text": bad})
        assert store.read_jsonl(scratch / "a.jsonl")[0]["text"] == bad
        assert store.read_jsonl(scratch / "b.jsonl")[0]["text"] == bad
        assert store.read_json(scratch / "c.json")["text"] == bad

    def test_write_jsonl_replaces_whole_file(self, scratch):
        p = scratch / "t.jsonl"
        store.write_jsonl(p, [{"a": 1}, {"a": 2}])
        store.write_jsonl(p, [{"a": 3}])
        assert store.read_jsonl(p) == [{"a": 3}]

    def test_workspace_under_layout(self, scratch):
        ws = Workspace.under(scratch).ensure()
        assert ws.pristine("c") == scratch / "corpus_src" / "c"
        assert ws.obf_root("c", "L2") == scratch / "obf" / "c" / "L2"
        assert ws.obf_file("c", "L1").name == "obf-c-L1.json"
        assert ws.targets_file("c").name == "targets-c.jsonl"
        assert ws.generations_file("a/b").name == "generations-a_b.jsonl"
        assert ws.results_file("t 1").name == "results-t_1.jsonl"
        assert ws.generation_failures_file("a/b").name == "generation-failures-a_b.jsonl"
        assert ws.completeness_file("t 1").name == "completeness-t_1.json"
        assert ws.summary_file("t", ".md").name == "summary-t.md"
        assert ws.summary_file("t", ".svg").name == "fig-t.svg"
        assert all(d.is_dir() for d in (ws.results, ws.corpus_src, ws.obf, ws.work))

    def test_workspace_default_honours_environment(self, monkeypatch, scratch):
        monkeypatch.setenv("BENCH_RESULTS_DIR", str(scratch / "r"))
        monkeypatch.setenv("BENCH_OBF_DIR", str(scratch / "o"))
        ws = Workspace.default()
        assert ws.results == scratch / "r" and ws.obf == scratch / "o"
        monkeypatch.delenv("BENCH_RESULTS_DIR")
        monkeypatch.delenv("BENCH_OBF_DIR")
        ws = Workspace.default()
        assert ws.results == store.PROJECT_ROOT / "results" and ws.obf == store.PROJECT_ROOT / "tmp" / "obf"


# ----------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------
class TestMetrics:
    def test_wilson_interval_known_values(self):
        lo, hi = metrics.wilson_interval(0, 10)
        assert lo == 0.0 and hi == pytest.approx(3.8416 / 13.8416, abs=1e-4)
        lo, hi = metrics.wilson_interval(5, 10)
        assert (lo, hi) == (pytest.approx(0.2366, abs=1e-3), pytest.approx(0.7634, abs=1e-3))
        assert metrics.wilson_interval(10, 10)[1] == 1.0
        assert metrics.wilson_interval(0, 0) == (0.0, 0.0)
        lo, hi = metrics.wilson_interval(3, 7)
        assert 0 <= lo < 3 / 7 < hi <= 1

    def test_mcnemar_exact_known_values(self):
        assert metrics.mcnemar_exact(0, 0) is None
        assert metrics.mcnemar_exact(5, 1) == pytest.approx(0.21875)
        assert metrics.mcnemar_exact(1, 5) == pytest.approx(0.21875)
        assert metrics.mcnemar_exact(10, 0) == pytest.approx(2 / 1024)
        assert metrics.mcnemar_exact(3, 3) == 1.0

    def test_cell_rates_shares_and_tokens(self):
        recs = [
            rec(target="a", stage="pass", category="pass", sim=1.0),
            rec(target="b", stage="test", category="test_fail", sim=0.5, cached=True),
            rec(target="c", stage="syntax", category="compile_syntax", sim=0.25),
            rec(target="d", stage="extract", category="extract_fail", sim=None),
        ]
        (cell,) = metrics.summarize(recs)
        assert (cell.n, cell.n_targets, cell.n_compiled, cell.n_pass) == (4, 4, 2, 1)
        assert cell.compile_rate == 0.5 and cell.pass_rate == 0.25      # stage ≥ test 才算编译通过
        assert cell.categories["extract_fail"] == 1 and cell.categories["link_error"] == 0
        assert sum(cell.shares.values()) == pytest.approx(1.0) and set(cell.shares) >= set(CATEGORIES)
        assert cell.mean_similarity == pytest.approx((1.0 + 0.5 + 0.25) / 3) and cell.n_similarity == 3
        assert (cell.prompt_tokens, cell.completion_tokens, cell.n_cached) == (400, 80, 1)
        assert cell.pass_ci[0] < 0.25 < cell.pass_ci[1]

    def test_shared_vocabulary_is_one_object_everywhere(self):
        """层级 / 回复问题名 / 归因类别只在 bench.vocab 定义一份，各模块导出的是同一个对象（不会漂移）。"""
        from bench import attribute, postprocess, vocab
        assert metrics.ISSUE_NAMES is vocab.ISSUES and postprocess.ISSUES is vocab.ISSUES
        assert LEVELS is vocab.LEVELS and metrics.LEVEL_ORDER is vocab.LEVELS and report.LEVELS is vocab.LEVELS
        assert attribute.CATEGORIES is vocab.CATEGORIES and report.CATEGORIES is vocab.CATEGORIES
        assert "policy_violation" in vocab.ISSUES and "policy_violation" in vocab.CATEGORIES
        assert set(report.CATEGORY_NOTES) == set(vocab.CATEGORIES)          # 报告里每个类别都有中文说明

    def test_reply_issues_are_counted_per_sample_and_match_postprocess(self):
        from bench.postprocess import ISSUES
        assert metrics.ISSUE_NAMES == tuple(ISSUES)                    # 不与后处理的 issue 清单漂移
        recs = [rec(target="a", issues=["multiple_functions", "truncated"]),
                rec(target="b", issues=["multiple_functions", "multiple_functions"]), rec(target="c", issues=[]),
                rec(target="d")]                                       # 没有 issues 字段的旧记录当作没有问题
        (cell,) = metrics.summarize(recs)
        assert cell.issue_counts == {"no_code_block": 0, "multiple_functions": 2, "signature_mismatch": 0,
                                     "added_include": 0, "truncated": 1, "policy_violation": 0}
        rows = list(csv.DictReader(io.StringIO(metrics.csv_text(metrics.summarize(recs), []))))
        assert float(rows[0]["issue_multiple_functions"]) == 0.5 and float(rows[0]["issue_truncated"]) == 0.25

    def test_policy_violations_are_failures_that_never_reached_the_compiler(self):
        recs = [rec(target="a", stage="policy", category="policy_violation", issues=["policy_violation"], sim=0.7),
                rec(target="b")]
        (cell,) = metrics.summarize(recs)
        assert (cell.n, cell.n_compiled, cell.n_pass) == (2, 1, 1) and cell.categories["policy_violation"] == 1
        assert cell.issue_counts["policy_violation"] == 1

    def test_analysis_skipped_replies_are_counted(self):
        (cell,) = metrics.summarize([rec(target="a", analysis_skipped=True), rec(target="b"), rec(target="c")])
        assert cell.n_analysis_skipped == 1

    def test_stratification_by_stub_kill_mode(self):
        """崩溃型 / 超时型目标的测试没断言返回内容，通过率会被高估：按杀死方式分层给出。"""
        recs = [rec(target="a", stub_kill_mode="test_fail", category="test_fail", stage="test"),
                rec(target="b", stub_kill_mode="test_fail"),
                rec(target="c", stub_kill_mode="runtime_crash"), rec(target="d", stub_kill_mode="runtime_crash"),
                rec(target="e")]
        (cell,) = metrics.summarize(recs)
        assert cell.kill_modes == {"test_fail": {"n": 2, "n_pass": 1}, "runtime_crash": {"n": 2, "n_pass": 2}}

    def test_reply_sharing_between_l1_and_l2_l3(self):
        """L1 / L2 / L3 的提示词逐字相同，回复（temperature=0）也相同：L2 / L3 的数字只反映「同一份回复放进别的副本」。"""
        def r(level, target, reply):
            return rec(level=level, target=target, inputs={"reply": reply})
        recs = [r("L0", "a", "A0"), r("L1", "a", "A1"), r("L2", "a", "A1"), r("L3", "a", "A1"),
                r("L0", "b", "B0"), r("L1", "b", "B1"), r("L2", "b", "B2"), r("L3", "b", "B1"),
                rec(level="L2", target="old")]                     # 没有回复摘要的旧记录不参与
        cells = {c.level: c for c in metrics.summarize_all(recs)}
        assert cells["L0"].same_reply_as_l1 is None and cells["L1"].same_reply_as_l1 is None
        assert cells["L2"].same_reply_as_l1 == 0.5 and cells["L2"].n_reply_compared == 2
        assert cells["L3"].same_reply_as_l1 == 1.0
        ds = {d.level: d for d in metrics.drop_vs_baseline(recs, per_corpus=False)}
        assert ds["L1"].n_same_as_l1 is None and ds["L2"].n_same_as_l1 == 1 and ds["L3"].n_same_as_l1 == 2

    def test_unknown_category_is_kept_not_dropped(self):
        (cell,) = metrics.summarize([rec(category="brand_new")])
        assert cell.categories["brand_new"] == 1 and cell.shares["brand_new"] == 1.0

    def test_dedupe_keeps_the_last_record(self):
        first = rec(target="a", category="test_fail", stage="test")
        again = rec(target="a", category="pass", stage="pass")
        (cell,) = metrics.summarize([first, again])
        assert cell.n == 1 and cell.n_pass == 1

    def test_levels_sort_in_obfuscation_order(self):
        recs = [rec(level=lv) for lv in ("L3", "L1", "L0", "L2")]
        assert [c.level for c in metrics.summarize(recs)] == ["L0", "L1", "L2", "L3"]
        assert [c.level for c in metrics.summarize_all(recs)] == ["L0", "L1", "L2", "L3"]

    def test_all_row_pools_corpora_and_gives_macro_average(self):
        recs = [rec(corpus_name="big", target=f"t{i}", category="pass" if i < 9 else "test_fail",
                    stage="pass" if i < 9 else "test") for i in range(10)]
        recs += [rec(corpus_name="small", target="s0", category="test_fail", stage="test")]
        (cell,) = metrics.summarize_all(recs)
        assert cell.corpus == metrics.ALL and cell.n == 11 and cell.n_pass == 9
        assert cell.pass_rate == pytest.approx(9 / 11)
        assert cell.macro_pass_rate == pytest.approx((0.9 + 0.0) / 2)      # 各语料等权，不被大语料淹没
        assert len(metrics.summarize(recs)) == 2

    def test_drop_vs_baseline_pairs_only_common_samples(self):
        l0 = [rec(level="L0", target="t0", category="pass"), rec(level="L0", target="t1", category="pass"),
              rec(level="L0", target="t2", category="test_fail", stage="test")]
        # L1 比 L0 少 t0（被 oracle gate 作废），多一个 t3；配对只看 t1、t2
        l1 = [rec(level="L1", target="t1", category="test_fail", stage="test"),
              rec(level="L1", target="t2", category="pass"), rec(level="L1", target="t3", category="pass")]
        (d,) = [x for x in metrics.drop_vs_baseline(l0 + l1, per_corpus=False)]
        assert (d.level, d.corpus, d.n_pairs) == ("L1", metrics.ALL, 2)
        assert d.pass_base == 0.5 and d.pass_level == 0.5 and d.delta_pass == 0.0
        assert (d.only_base, d.only_level, d.p_value) == (1, 1, 1.0)

    def test_drop_vs_baseline_reports_drop_and_relative_drop(self):
        recs = []
        for i in range(10):
            recs.append(rec(level="L0", target=f"t{i}", category="pass"))
            recs.append(rec(level="L1", target=f"t{i}", category="pass" if i < 6 else "test_fail",
                            stage="pass" if i < 6 else "test", sim=0.4))
        recs.append(rec(model="m2", level="L1", target="x"))       # m2 没有 L0：不产生比较行
        ds = metrics.drop_vs_baseline(recs)
        assert {d.model for d in ds} == {"m1"}
        (d,) = [x for x in ds if x.corpus == metrics.ALL]
        assert d.delta_pass == pytest.approx(-0.4) and d.rel_drop == pytest.approx(0.4)
        assert (d.only_base, d.only_level) == (4, 0)
        assert d.p_value == pytest.approx(2 * (1 / 16))
        assert d.delta_compile == pytest.approx(0.0)               # 失败的 4 个仍进入了测试阶段
        assert d.delta_sim == pytest.approx(0.4 - 0.9)

    def test_drop_per_corpus_rows_and_zero_baseline(self):
        recs = [rec(level="L0", corpus_name="a", category="test_fail", stage="test"),
                rec(level="L1", corpus_name="a", category="pass"),
                rec(level="L0", corpus_name="b", category="pass"), rec(level="L1", corpus_name="b", category="pass")]
        ds = metrics.drop_vs_baseline(recs)
        assert [(d.level, d.corpus) for d in ds] == [("L1", "ALL"), ("L1", "a"), ("L1", "b")]
        a = next(d for d in ds if d.corpus == "a")
        assert a.rel_drop is None and a.delta_pass == 1.0           # 基线为 0 时没有相对下降

    def test_overview(self):
        info = metrics.overview([rec(model="m2", level="L1"), rec(model="m1", level="L0", corpus_name="z")])
        assert info == {"n_records": 2, "models": ["m1", "m2"], "levels": ["L0", "L1"], "corpora": ["c1", "z"],
                        "n_targets": 2}

    def test_load_results_skips_bad_lines_but_reports_how_many(self, scratch, caplog):
        p = scratch / "r.jsonl"
        p.write_text(json.dumps(rec()) + "\n{bad\n\n" + json.dumps(rec(target="t9")) + "\n", encoding="utf-8")
        with caplog.at_level("WARNING"):
            assert len(metrics.load_results(p)) == 2 and metrics.load_results(scratch / "none.jsonl") == []
        assert "不是合法 JSON" in caplog.text                              # 以前静默 continue，少算样本没人知道
        records, bad = metrics.load_results_checked(p)
        assert len(records) == 2 and bad == 1

    def test_csv_columns_rows_and_delta_fill(self):
        recs = [rec(level="L0", target="t0"), rec(level="L1", target="t0", category="test_fail", stage="test")]
        text = metrics.csv_text(metrics.summarize_all(recs) + metrics.summarize(recs), metrics.drop_vs_baseline(recs))
        rows = list(csv.DictReader(io.StringIO(text)))
        assert list(rows[0]) == metrics.CSV_COLUMNS and len(rows) == 4
        l1_all = next(r for r in rows if r["level"] == "L1" and r["corpus"] == "ALL")
        assert float(l1_all["pass_rate"]) == 0.0 and float(l1_all["delta_pass_vs_L0"]) == -1.0
        assert float(l1_all["cat_test_fail"]) == 1.0 and l1_all["n_pairs_vs_L0"] == "1"
        l0 = next(r for r in rows if r["level"] == "L0" and r["corpus"] == "ALL")
        assert l0["delta_pass_vs_L0"] == ""                        # L0 自己没有「相对 L0」


# ----------------------------------------------------------------------
# report
# ----------------------------------------------------------------------
def stamp(r: dict, tree="tree0", spec="spec0", prompt=None, gate="gate0", reply="reply") -> dict:
    """给合成记录补上 evaluate 阶段会写的 schema_version 与 inputs（必备字段见 report.REQUIRED_INPUTS）。"""
    r = dict(r)
    r["schema_version"] = SCHEMA_VERSION
    tid, idx = r.get("task_id"), r.get("sample_idx")
    r["inputs"] = {"spec": spec, "tree": tree, "prompt": prompt or PROMPT_DIGEST[:16], "gen": f"gen:{tid}:{idx}",
                   "messages": f"msg:{tid}", "reply": store.digest(reply), "gate": gate, "code": code_digest("evaluate")}
    return r


def write_obf(ws: Workspace, corpus_name: str, level: str, tree="tree0", spec="spec0", **extra) -> None:
    store.write_json(ws.obf_file(corpus_name, level), {
        "schema_version": SCHEMA_VERSION, "corpus": corpus_name, "level": level, "tree_hash": tree, "ok": True,
        "inputs": {"spec": spec}, "rename_map": {"a": "v_1"}, "strict_stringify": False,
        "oracle_gate": {"checked": 3, "passed": 3, "failed": [], "copy_invalid": False, "input_hash": "gate0"},
        **extra})


def sync_files(ws: Workspace, records: list[dict], reply: str = "reply") -> None:
    """写出与 stamp() 的记录一致的生成记录和 tasks.jsonl：report 会拿结果记录的 inputs 去对它们。"""
    gens: dict[str, list[dict]] = {}
    tasks: dict[str, dict] = {}
    for r in records:
        inp = r.get("inputs") or {}
        gens.setdefault(str(r["model"]), []).append({"task_id": r["task_id"], "sample_idx": r["sample_idx"],
                                                     "input_hash": inp.get("gen"), "text": reply, "model": r["model"]})
        tasks[r["task_id"]] = {"task_id": r["task_id"], "corpus": r["corpus"], "level": r["level"],
                               "messages_hash": inp.get("messages")}
    for m, rows in gens.items():
        store.write_jsonl(ws.generations_file(m), rows)
    store.write_jsonl(ws.tasks_file, list(tasks.values()))


def put_results(ws: Workspace, tag: str, records: list[dict]) -> None:
    """写结果文件，并让上游文件（生成记录、tasks.jsonl）与它们对得上。"""
    store.write_jsonl(ws.results_file(tag), records)
    sync_files(ws, records)


def synthetic_records() -> list[dict]:
    """两个模型 × L0/L1 × 两个语料；模型 B 在 L1 明显变差。"""
    out = []
    for model, p1 in (("model-a", 6), ("model-b", 3)):
        for lv in ("L0", "L1"):
            for c in ("c1", "c2"):
                for i in range(6):
                    ok = i < (6 if lv == "L0" else p1)
                    out.append(stamp(rec(model=model, level=lv, corpus_name=c, target=f"t{i}",
                                         category="pass" if ok else "compile_undeclared",
                                         stage="pass" if ok else "syntax")))
    return out


class TestValidateRecords:
    @pytest.fixture()
    def ws(self, scratch):
        ws = Workspace.under(scratch).ensure()
        write_obf(ws, "c1", "L0")
        return ws

    def good(self, ws, **kw):
        r = stamp(rec(), **kw)
        sync_files(ws, [r])
        return r

    def test_matching_record_is_valid(self, ws):
        v = report.validate_records([self.good(ws)], ws)
        assert len(v.valid) == 1 and not v.stale and v.total == 1

    @pytest.mark.parametrize("mutate, reason", [
        (lambda r: r.update(schema_version=SCHEMA_VERSION + 1), "schema_version 不符"),
        (lambda r: r["inputs"].update(prompt="0" * 16), "prompt 模板已变"),
        (lambda r: r["inputs"].update(tree="other"), "混淆副本已变"),
        (lambda r: r["inputs"].update(spec="other"), "spec 与生成副本时不一致"),
        (lambda r: r["inputs"].update(code="0" * 16), "评测代码已变"),
        (lambda r: r["inputs"].update(gate="other"), "oracle gate 已重跑"),
        (lambda r: r["inputs"].update(gen="other"), "回复已变"),
        (lambda r: r["inputs"].update(reply="other"), "回复已变"),
        (lambda r: r["inputs"].update(messages="other"), "提示词 / 功能规格已变"),
        (lambda r: r["inputs"].pop("gen"), "记录缺少输入哈希字段"),
        (lambda r: r.update(corpus="nowhere", level="L2"), "缺少对应的混淆副本记录"),
    ])
    def test_each_kind_of_mismatch_is_reported(self, ws, mutate, reason):
        r = self.good(ws)
        mutate(r)
        v = report.validate_records([r], ws)
        assert not v.valid and len(v.stale) == 1 and v.stale[0][1].startswith(reason)
        assert v.reasons() == {v.stale[0][1]: 1}

    def test_live_spec_digest_is_checked_when_given(self, ws):
        live = corpus.CorpusSpec(name="c1", cflags=["-O0"])
        good = stamp(rec(), spec=store.spec_digest(live))
        sync_files(ws, [good])
        write_obf(ws, "c1", "L0", spec=store.spec_digest(live))
        assert report.validate_records([good], ws, {"c1": live}).valid
        changed = corpus.CorpusSpec(name="c1", cflags=["-O2"])
        v = report.validate_records([good], ws, {"c1": changed})
        assert not v.valid and v.stale[0][1] == "spec 已变"

    def test_record_without_inputs_is_stale(self, ws):
        assert not report.validate_records([rec()], ws).valid

    # -- 上游文件变了：以前只比 tree / spec / prompt，重新 describe 或 generate 之后忘了重跑 evaluate，旧结果照样算有效 --
    def test_deleting_the_generations_makes_every_result_stale(self, ws):
        r = self.good(ws)
        ws.generations_file("m1").unlink()
        v = report.validate_records([r], ws)
        assert not v.valid and v.stale[0][1] == "对应的生成记录已不存在"

    def test_regenerated_reply_makes_the_old_result_stale(self, ws):
        r = self.good(ws)
        sync_files(ws, [r], reply="a different reply")             # 重新 generate 了，回复文本变了
        assert report.validate_records([r], ws).stale[0][1].startswith("回复已变")

    def test_regenerated_with_other_parameters_makes_the_old_result_stale(self, ws):
        r = self.good(ws)
        rows = store.read_jsonl(ws.generations_file("m1"))
        rows[0]["input_hash"] = "changed-by-new-llm-config"          # 换了模型配置 / 温度：生成记录的输入哈希变了
        store.write_jsonl(ws.generations_file("m1"), rows)
        assert report.validate_records([r], ws).stale[0][1].startswith("回复已变")

    def test_changed_description_changes_the_messages_hash(self, ws):
        r = self.good(ws)
        tasks = store.read_jsonl(ws.tasks_file)
        tasks[0]["messages_hash"] = "msg-after-new-description"
        store.write_jsonl(ws.tasks_file, tasks)
        assert report.validate_records([r], ws).stale[0][1].startswith("提示词 / 功能规格已变")

    def test_task_that_was_voided_since_is_stale(self, ws):
        r = self.good(ws)
        store.write_jsonl(ws.tasks_file, [])
        assert report.validate_records([r], ws).stale[0][1].startswith("任务已不在 tasks.jsonl")

    def test_rerunning_the_oracle_gate_makes_old_results_stale(self, ws):
        r = self.good(ws)
        write_obf(ws, "c1", "L0", oracle_gate={"checked": 3, "passed": 3, "failed": [], "copy_invalid": False,
                                                 "input_hash": "gate-after-rerun"})
        assert report.validate_records([r], ws).stale[0][1].startswith("oracle gate 已重跑")


class TestReport:
    @pytest.fixture()
    def ws(self, scratch):
        ws = Workspace.under(scratch).ensure()
        for c in ("c1", "c2"):
            for lv in ("L0", "L1"):
                write_obf(ws, c, lv)
        return ws

    def test_build_report_writes_md_csv_svg(self, ws):
        records = synthetic_records()
        put_results(ws, "t", records)
        res = report.build_report(ws, "t")
        assert (res.n_records, res.n_used, res.n_stale) == (48, 48, 0)
        assert set(res.paths) == {"md", "csv", "svg"} and all(p.stat().st_size > 0 for p in res.paths.values())
        md = res.paths["md"].read_text(encoding="utf-8")
        for heading in ("# 评测汇总 · t", "## 1. 总表", "## 2. 相对 L0 的变化", "## 3. 失败归因分布",
                        "## 4. 分语料明细", "## 5. 可信度记录", "## 6. 局限", "## 7. 来源"):
            assert heading in md
        assert "model-a" in md and "model-b" in md and "compile_undeclared" in md
        # 模型 b 的 L1 通过率 50%，相对 L0 下降 50%（12 个配对里 6 个只在 L0 通过）
        section2 = md.split("## 2.")[1].split("## 3.")[0]
        row = next(ln for ln in section2.splitlines() if ln.startswith("| model-b | L1 | 12 |"))
        assert "100.0% | 50.0% | -50.0 | 50.0% | 6 | 0 |" in row
        rows = list(csv.DictReader(res.paths["csv"].open(encoding="utf-8")))
        assert len(rows) == 4 + 8        # 合计行 2 模型 × 2 级别，明细行 2 模型 × 2 级别 × 2 语料
        assert {r["corpus"] for r in rows} == {"ALL", "c1", "c2"}

    def test_markdown_has_reply_issue_table(self, ws):
        records = synthetic_records()
        for r in records[:6]:
            r["issues"] = ["signature_mismatch"]
        records[6]["issues"] = ["multiple_functions", "truncated"]
        put_results(ws, "t", records)
        md = report.build_report(ws, "t", formats=("md",)).paths["md"].read_text(encoding="utf-8")
        assert "### 回复规范性" in md
        section = md.split("### 回复规范性")[1].split("## 4.")[0]
        assert "| model-a | L0 | 12 | 8.3% | 50.0% | 0.0% | 8.3% | 0.0% | 0 |" in section   # 1/12、6/12、0、1/12、0、0

    def test_markdown_credibility_section_reads_side_files(self, ws):
        put_results(ws, "t", synthetic_records())
        store.write_json(ws.targets_summary_file("c1"), {
            "extracted": 40, "candidates": 30, "selected": 6, "excluded": {"stub_survived": 3, "not_executed": 2},
            "stub_kill_modes": {"test_fail": 5, "runtime_crash": 1}})
        store.append_jsonl(ws.descriptions_file, {"target_id": "c1:t0", "corpus": "c1", "failed": True, "leak_retries": 2,
                                                  "model": "desc-m"})
        md = report.build_report(ws, "t").paths["md"].read_text(encoding="utf-8")
        assert "### 目标集与 stub-kill" in md and "| c1 | 40 | 30 | 6 | 3 | 2 |" in md
        assert "runtime_crash=1, test_fail=5" in md
        assert "### 混淆副本与 oracle gate" in md and "3/3" in md
        assert "### 功能规格" in md and "desc-m" in md

    def test_limitations_section_is_the_shared_list_and_says_l1_l2_l3_share_replies(self, ws):
        from bench.limitations import LIMITATIONS, REPLY_SHARING_NOTE
        put_results(ws, "t", synthetic_records())
        md = report.build_report(ws, "t", formats=("md",)).paths["md"].read_text(encoding="utf-8")
        section = md.split("## 6. 局限")[1].split("## 7.")[0]
        for title, text in LIMITATIONS:
            assert f"**{title}。**" in section and text in section
        # 「L1 / L2 / L3 共用同一批回复」不能只藏在 README 里：总表、相对 L0 的变化表下面都要有提示
        assert md.count(REPLY_SHARING_NOTE) >= 2
        assert "回复与 L1 相同" in md and "回复与 L1 逐字相同的配对数" in md
        assert any("OOM" in t or "负载" in t for t, _ in LIMITATIONS)

    def test_reply_sharing_columns_are_filled_from_reply_digests(self, ws):
        recs = []
        for lv, reply in (("L0", "r0"), ("L1", "r1"), ("L2", "r1"), ("L3", "r1")):
            r = stamp(rec(level=lv, target="t0"), reply=reply)
            recs.append(r)
        for lv in ("L2", "L3"):
            write_obf(ws, "c1", lv)
        put_results(ws, "t", recs)
        texts = {"L0": "r0", "L1": "r1", "L2": "r1", "L3": "r1"}           # 生成记录里的回复文本要和 stamp 的摘要对上
        rows = store.read_jsonl(ws.generations_file("m1"))
        for row in rows:
            row["text"] = texts[row["task_id"].rsplit("@", 1)[1]]
        store.write_jsonl(ws.generations_file("m1"), rows)
        res = report.build_report(ws, "t", formats=("md",))
        assert res.n_used == 4, res.stale_reasons
        md = res.paths["md"].read_text(encoding="utf-8")
        row = next(ln for ln in md.splitlines() if ln.startswith("| m1 | L2 |"))
        assert "100.0%" in row                                     # L2 的回复与 L1 全部相同
        row0 = next(ln for ln in md.splitlines() if ln.startswith("| m1 | L0 |"))
        assert row0.count("| - |") >= 1                            # L0 / L1 没有这一列的值

    def test_unverified_targets_get_a_loud_warning(self, ws):
        put_results(ws, "t", synthetic_records())
        store.write_json(ws.targets_summary_file("c1"), {
            "extracted": 40, "candidates": 30, "selected": 6, "excluded": {}, "unverified": True,
            "filter": {"require_stub_kill": False, "require_exec": True}, "stub_kill_modes": {None: 6}})
        md = report.build_report(ws, "t", formats=("md",)).paths["md"].read_text(encoding="utf-8")
        head = md.split("## 1.")[0]
        assert "警告：目标集未经完整的可信度过滤" in head and "c1（未做 stub-kill）" in head
        assert "未做" in md.split("### 目标集与 stub-kill")[1].split("###")[0]

    def test_missing_samples_and_bad_lines_are_called_out(self, ws):
        records = synthetic_records()
        put_results(ws, "t", records)
        with ws.results_file("t").open("a", encoding="utf-8") as f:
            f.write('{"half": ')                                   # 进程被杀时留下的半行
        store.write_json(ws.completeness_file("t"), {"cells": [
            {"model": "model-b", "level": "L1", "corpus": "c1", "expected": 8, "have": 6, "failed_calls": 2,
             "stale_generations": 0, "not_generated": 0, "missing": 2},
            {"model": "model-a", "level": "L1", "corpus": "c1", "expected": 6, "have": 6, "failed_calls": 0,
             "stale_generations": 0, "not_generated": 0, "missing": 0}]})
        res = report.build_report(ws, "t", formats=("md",))
        assert res.n_bad_lines == 1 and res.n_missing == 2 and res.n_records == 48
        md = res.paths["md"].read_text(encoding="utf-8")
        head = md.split("## 1.")[0]
        assert "1 条坏行" in head and "样本不完整" in head and "缺 2 个（其中 2 个是生成调用失败）" in head
        assert "不完整（缺 2/8）" in md and "缺 2" in md.split("## 4.")[1]

    def test_cells_are_complete_when_the_completeness_file_says_so(self, ws):
        put_results(ws, "t", synthetic_records())
        store.write_json(ws.completeness_file("t"), {"cells": [
            {"model": "model-a", "level": "L0", "corpus": c, "expected": 6, "have": 6, "failed_calls": 0,
             "stale_generations": 0, "not_generated": 0, "missing": 0} for c in ("c1", "c2")]})
        res = report.build_report(ws, "t", formats=("md",))
        md = res.paths["md"].read_text(encoding="utf-8")
        assert res.n_missing == 0 and "样本不完整" not in md
        assert "| model-a | L0 | 12 | 完整 |" in md

    def test_kill_mode_stratification_section(self, ws):
        records = synthetic_records()
        for i, r in enumerate(records):
            r["stub_kill_mode"] = "runtime_crash" if int(r["target_id"].split(":t")[1]) % 3 == 0 else "test_fail"
        put_results(ws, "t", records)
        md = report.build_report(ws, "t", formats=("md",)).paths["md"].read_text(encoding="utf-8")
        section = md.split("### 按 stub-kill 杀死方式分层的通过率")[1].split("## 4.")[0]
        assert "| model-a | L0 | runtime_crash | 4 | 4 | 100.0% |" in section
        assert "| model-b | L1 | test_fail |" in section and "任何不崩溃的错误实现都能通过" in section

    def test_provenance_section_lists_the_environment(self, ws):
        records = synthetic_records()
        for r in records:
            r["sandbox"] = {"image": "gcc:13", "image_id": "sha256:abc", "cpus": 1.0}
        put_results(ws, "t", records)
        store.write_jsonl(ws.generations_file("model-a"), [
            {"task_id": r["task_id"], "sample_idx": r["sample_idx"], "input_hash": r["inputs"]["gen"], "text": "reply",
             "api_model": "deepseek-flash", "llm_fp": "fp123", "temperature": 0.0, "max_tokens": 4096,
             "system_fingerprint": "fp_x", "model": "model-a"} for r in records if r["model"] == "model-a"])
        md = report.build_report(ws, "t", formats=("md",)).paths["md"].read_text(encoding="utf-8")
        section = md.split("## 7. 来源")[1]
        assert PROMPT_DIGEST[:16] in section and "image_id=sha256:abc" in section
        assert "模型 model-a" in section and "deepseek-flash" in section and "fp123" in section
        assert str(ROOT) not in md                                      # 来源信息不带本机绝对路径

    def test_stale_records_are_excluded_and_listed(self, ws):
        records = synthetic_records()
        records[0]["inputs"]["tree"] = "tampered"
        records[1]["inputs"]["prompt"] = "0" * 16
        put_results(ws, "t", records)
        res = report.build_report(ws, "t", formats=("md",))
        assert (res.n_records, res.n_used, res.n_stale) == (48, 46, 2)
        assert len(res.stale_reasons) == 2
        md = res.paths["md"].read_text(encoding="utf-8")
        assert "输入校验" in md and "已排除，不进统计" in md and "prompt 模板已变：1 条" in md
        res = report.build_report(ws, "t", formats=("md",), allow_stale=True)
        assert res.n_used == 48
        assert "仍然纳入统计" in res.paths["md"].read_text(encoding="utf-8")

    def test_empty_results_still_produce_a_readable_summary(self, ws):
        res = report.build_report(ws, "none")
        assert res.n_records == 0 and "svg" not in res.paths
        assert "没有可用的评测记录" in res.paths["md"].read_text(encoding="utf-8")

    def test_all_stale_says_so(self, ws):
        records = [stamp(rec(), prompt="0" * 16)]
        res = report.build_report(ws, "t", records=records, formats=("md",))
        md = res.paths["md"].read_text(encoding="utf-8")
        assert res.n_used == 0 and "没有可用的评测记录" in md and "prompt 模板已变" in md

    def test_generated_text_has_no_banned_words(self, ws):
        put_results(ws, "t", synthetic_records())
        res = report.build_report(ws, "t")
        for path in res.paths.values():
            text = path.read_text(encoding="utf-8")
            assert not [w for w in BANNED if w in text], path.name


class TestSvg:
    def cells(self, models=("a", "b"), levels=("L0", "L1", "L2", "L3")):
        recs = []
        for mi, m in enumerate(models):
            for li, lv in enumerate(levels):
                for i in range(10):
                    ok = i < 10 - li - mi
                    recs.append(rec(model=m, level=lv, target=f"t{i}", category="pass" if ok else "test_fail",
                                    stage="pass" if ok else "test"))
        return metrics.summarize_all(recs)

    def parse(self, svg: str) -> ET.Element:
        return ET.fromstring(svg)

    def test_svg_is_well_formed_with_points_titles_and_legend(self):
        svg = report.render_svg(self.cells(), "标题 <&>")
        root = self.parse(svg)                                        # 能被 XML 解析器接受，转义无误
        ns = {"s": "http://www.w3.org/2000/svg"}
        markers = [c for c in root.iterfind(".//s:circle", ns) if c.get("r") == "4"]
        assert len(markers) == 2 * 4 * 2                              # 2 个模型 × 4 个级别 × 2 个面板
        assert len(root.findall(".//s:title", ns)) == 2 * 4 * 2       # 每个点一个悬停提示
        assert len(root.findall(".//s:polyline", ns)) == 4            # 每个面板每个模型一条线
        assert "标题 <&>" not in svg and "标题 &lt;&amp;&gt;" in svg
        legend = [t.text for t in root.iterfind(".//s:text", ns) if t.get("class") == "leg"]
        assert legend == ["a", "b"]

    def test_marks_follow_the_spec(self):
        svg = report.render_svg(self.cells())
        assert 'stroke-width="2"' in svg and 'stroke-linejoin="round"' in svg      # 线 2px、圆角
        assert 'r="4"' in svg and 'stroke="var(--surface)"' in svg                 # 圆点 8px、2px 底色描边
        assert 'fill-opacity="0.10"' in svg                                          # 色带 10% 不透明度
        assert "stroke-dasharray" not in svg                                         # 网格是实线
        assert "prefers-color-scheme: dark" in svg and "<script" not in svg and "http://" not in svg.replace(
            "http://www.w3.org/2000/svg", "")                                        # 自带深色主题，不引外部资源

    def test_single_model_has_no_legend(self):
        svg = report.render_svg(self.cells(models=("only",)))
        assert 'class="leg"' not in svg and "only" in svg                           # 单个系列不画图例（标题说明）

    def test_many_models_drop_bands_and_cap_at_eight(self):
        svg = report.render_svg(self.cells(models=tuple("abcdefghij")))
        assert "fill-opacity" not in svg and "只画前 8 个" in svg
        assert svg.count("<polyline") == 2 * 8

    def test_single_level_draws_points_only(self):
        svg = report.render_svg(self.cells(levels=("L0",)))
        assert "<polyline" not in svg and 'r="4"' in svg
        self.parse(svg)

    def test_no_data_returns_none(self):
        assert report.render_svg([]) is None
        assert report.render_svg([c for c in self.cells() if c.corpus != metrics.ALL]) is None

    def test_series_colors_follow_model_order(self):
        svg = report.render_svg(self.cells())
        assert "var(--s1)" in svg and "var(--s2)" in svg and "var(--s3)" not in svg.split("</style>")[1]


# ----------------------------------------------------------------------
# obfuscate() 入口（不跑 verify，不需要 docker）
# ----------------------------------------------------------------------
class TestObfuscateEntry:
    SEED = 5

    def run(self, spec, level, out, results, **kw) -> ObfResult:
        return obfuscate(spec, MINI, level, kw.pop("seed", self.SEED), out, verify=False, results_dir=results, **kw)

    def test_levels_constant_and_bad_arguments(self, scratch, mini_spec):
        assert LEVELS == ["L0", "L1", "L2", "L3"]
        with pytest.raises(ValueError, match="未知混淆级别"):
            self.run(mini_spec, "L9", scratch / "o", scratch)
        with pytest.raises(FileNotFoundError):
            obfuscate(mini_spec, scratch / "nope", "L0", 1, scratch / "o", verify=False, results_dir=scratch)

    def test_l0_is_a_plain_copy_with_a_result_file(self, scratch, mini_spec):
        res = self.run(mini_spec, "L0", scratch / "o" / "L0", scratch)
        assert isinstance(res, ObfResult) and res.level == "L0" and res.ok and res.verify is None
        assert res.rename_map == {} and not res.skipped
        assert corpus.tree_digest(res.out_root) == corpus.tree_digest(MINI)
        saved = store.read_json(scratch / "obf-mini-L0.json")
        assert saved["schema_version"] == SCHEMA_VERSION and saved["corpus"] == "mini" and saved["level"] == "L0"
        assert saved["tree_hash"] == res.tree_hash == corpus.tree_digest(MINI)[:16]
        assert saved["rename_map"] == {} and saved["stats"]["level"] == "L0" and saved["verify"] is None
        assert saved["inputs"]["spec"] == store.spec_digest(mini_spec) and saved["oracle_gate"] is None

    def test_each_level_adds_a_transform_on_top_of_the_previous(self, scratch, mini_spec):
        out = {lv: self.run(mini_spec, lv, scratch / "o" / lv, scratch) for lv in LEVELS}
        hashes = [out[lv].tree_hash for lv in LEVELS]
        assert len(set(hashes)) == 4                                   # 每一级的副本都不同
        assert out["L1"].rename_map and out["L1"].rename_map == out["L2"].rename_map == out["L3"].rename_map
        assert set(out["L1"].stats) >= {"level", "seed", "symbols", "attempts", "targets_not_renamed"}
        assert "structure" not in out["L1"].stats and "structure" in out["L2"].stats
        assert "semantic" not in out["L2"].stats and "semantic" in out["L3"].stats
        assert out["L1"].stats["symbols"]["entities_renamed"] == len(out["L1"].rename_map)
        assert out["L2"].stats["structure"]["total"] > 0 and out["L3"].stats["semantic"]["total"] > 0
        # 改名后函数名不再出现在源码里（注释与字符串除外，这里只看定义头）
        calc = (out["L1"].out_root / "calc.c").read_text(encoding="utf-8")
        assert "int calc_gcd(" not in calc and f"{out['L1'].rename_map['calc_gcd']}(" in calc

    def test_same_seed_same_tree_other_seed_other_tree(self, scratch, mini_spec):
        a = self.run(mini_spec, "L3", scratch / "a", scratch / "ra")
        b = self.run(mini_spec, "L3", scratch / "b", scratch / "rb")
        c = self.run(mini_spec, "L3", scratch / "c", scratch / "rc", seed=self.SEED + 1)
        assert a.tree_hash == b.tree_hash and a.rename_map == b.rename_map
        assert c.tree_hash != a.tree_hash and c.rename_map != a.rename_map

    def test_second_call_is_skipped_force_and_damage_rebuild(self, scratch, mini_spec):
        out = scratch / "o"
        first = self.run(mini_spec, "L1", out, scratch)
        again = self.run(mini_spec, "L1", out, scratch)
        assert not first.skipped and again.skipped and again.rename_map == first.rename_map
        assert again.tree_hash == first.tree_hash and again.stats == first.stats
        assert not self.run(mini_spec, "L1", out, scratch, force=True).skipped
        (out / "calc.c").write_text("/* 被改坏了 */\n", encoding="utf-8")           # 副本被改动 → 不能沿用
        rebuilt = self.run(mini_spec, "L1", out, scratch)
        assert not rebuilt.skipped and rebuilt.tree_hash == first.tree_hash
        assert self.run(mini_spec, "L1", out, scratch, seed=99).skipped is False      # 种子变了 → 重做

    def test_write_result_false_leaves_no_file(self, scratch, mini_spec):
        res = obfuscate(mini_spec, MINI, "L1", 1, scratch / "o", verify=False, write_result=False)
        assert res.result_path is None and not list(scratch.glob("obf-*.json"))

    def test_strict_stringify_excludes_more_names(self, scratch, mini_spec):
        loose = self.run(mini_spec, "L1", scratch / "a", scratch / "ra", strict_stringify=False)
        strict = self.run(mini_spec, "L1", scratch / "b", scratch / "rb", strict_stringify=True)
        assert strict.strict_stringify and not loose.strict_stringify
        assert len(strict.rename_map) < len(loose.rename_map)


class ScriptedSandbox:
    """按脚本返回构建结果的假沙箱（只用来测 obfuscate() 的 verify 重试逻辑，不跑任何命令）。"""

    def __init__(self, *rcs: int) -> None:
        self.rcs = list(rcs)
        self.calls = 0

    def run(self, workdir, cmd, env=None, timeout_s=None):
        from bench.sandbox import RunResult
        rc = self.rcs[min(self.calls, len(self.rcs) - 1)]
        self.calls += 1
        return RunResult(rc=rc, stdout="", stderr="", duration_s=0.0, timed_out=False, signal=None)


class TestObfuscateStrictFallback:
    def run(self, spec, level, scratch, sandbox, **kw) -> ObfResult:
        return obfuscate(spec, MINI, level, 5, scratch / "o", sandbox=sandbox, targets=[], results_dir=scratch,
                         work_root=scratch / "work", **kw)

    def test_failed_verify_retries_once_in_strict_mode(self, scratch, mini_spec):
        sandbox = ScriptedSandbox(1, 0)
        res = self.run(mini_spec, "L1", scratch, sandbox)
        assert res.ok and res.strict_stringify is True and sandbox.calls == 2
        assert [a["strict_stringify"] for a in res.stats["attempts"]] == [False, True]
        assert [a["ok"] for a in res.stats["attempts"]] == [False, True]
        loose = obfuscate(mini_spec, MINI, "L1", 5, scratch / "loose", verify=False, results_dir=scratch / "r2",
                          strict_stringify=False)
        assert len(res.rename_map) < len(loose.rename_map)            # strict 多排除了出现在 # 宏实参里的名字
        assert store.read_json(scratch / "obf-mini-L1.json")["strict_stringify"] is True

    def test_fixed_mode_and_l0_never_retry(self, scratch, mini_spec):
        for kw in ({"strict_stringify": False}, {"strict_stringify": True}):
            sandbox = ScriptedSandbox(1)
            res = self.run(mini_spec, "L1", scratch, sandbox, force=True, **kw)
            assert not res.ok and sandbox.calls == 1 and len(res.stats["attempts"]) == 1
            assert res.strict_stringify is kw["strict_stringify"]
        sandbox = ScriptedSandbox(1)
        res = self.run(mini_spec, "L0", scratch, sandbox, force=True)
        assert not res.ok and sandbox.calls == 1 and res.stats["attempts"] == [
            {"strict_stringify": None, "ok": False, "build_rc": 1, "missing": 0}]

    def test_both_modes_failing_leaves_the_strict_attempt_as_the_result(self, scratch, mini_spec):
        sandbox = ScriptedSandbox(1, 1)
        res = self.run(mini_spec, "L2", scratch, sandbox)
        assert not res.ok and sandbox.calls == 2 and res.strict_stringify is True
        assert res.problems and "verify 未通过" in res.problems[0]
        assert store.read_json(scratch / "obf-mini-L2.json")["ok"] is False


# ----------------------------------------------------------------------
# 出题、描述、生成（不需要 docker：沙箱评测之前的所有阶段）
# ----------------------------------------------------------------------
from bench import pipeline                                              # noqa: E402
from bench.extract import extract_functions                             # noqa: E402
from bench.llm import MOCK_DESCRIPTION                                  # noqa: E402
from bench.pipeline import Context, StageError                          # noqa: E402
from tests.pipeline_support import CountingSandbox, Recorder, add_level, describe_all  # noqa: E402


class TestPlanTasks:
    def test_missing_obf_and_missing_gate_name_the_stage_to_run(self, planned, mini_spec):
        ctx, ws, _ = planned
        describe_all(ctx, mini_spec)
        ws.obf_file("mini", "L2").unlink(missing_ok=True)
        with pytest.raises(StageError, match="obfuscate"):
            pipeline.plan_tasks(ctx, mini_spec, ["L2"])
        data = store.read_json(ws.obf_file("mini", "L1"))
        data["oracle_gate"] = None
        store.write_json(ws.obf_file("mini", "L1"), data)
        with pytest.raises(StageError, match="verify"):
            pipeline.plan_tasks(ctx, mini_spec, ["L1"])
        assert pipeline.plan_tasks(ctx, mini_spec, ["L1"], require_gate=False)[0]

    def test_modified_copy_is_refused(self, planned, mini_spec):
        ctx, ws, _ = planned
        describe_all(ctx, mini_spec)
        (ws.obf_root("mini", "L0") / "calc.c").write_text("/* x */\n", encoding="utf-8")
        with pytest.raises(StageError, match="哈希不符"):
            pipeline.plan_tasks(ctx, mini_spec, ["L0"])

    def test_targets_without_a_usable_description_or_gate_are_voided_and_counted(self, planned, mini_spec):
        ctx, ws, targets = planned
        describe_all(ctx, mini_spec)
        by_name = {t.name: t for t in targets}
        # array_find 没有规格；parse_digits 规格生成失败；calc_median3 没过 oracle gate
        recs = [r for r in store.read_jsonl(ws.descriptions_file) if r["target_id"] != by_name["array_find"].target_id]
        for r in recs:
            if r["target_id"] == by_name["parse_digits"].target_id:
                r.update(failed=True, description="", reason="leak")
        store.write_jsonl(ws.descriptions_file, recs)
        for lv in ("L0", "L1"):
            data = store.read_json(ws.obf_file("mini", lv))
            data["oracle_gate"]["failed"] = [{"target_id": by_name["calc_median3"].target_id, "category": "test_fail"}]
            store.write_json(ws.obf_file("mini", lv), data)
        tasks, voids = pipeline.plan_tasks(ctx, mini_spec, ["L0", "L1"])
        assert sorted({pt.task.obf_name for pt in tasks if pt.task.level == "L0"}) == ["calc_gcd"]
        assert [pt.task.task_id for pt in tasks if pt.task.level == "L1"] == ["mini:calc.c:calc_gcd@L1"]
        expect = {"no_description": 1, "describe_failed": 1, "oracle_failed": 1}
        assert voids == {"L0": expect, "L1": expect}

    def test_invalid_copy_voids_the_whole_level(self, planned, mini_spec):
        ctx, ws, _ = planned
        describe_all(ctx, mini_spec)
        data = store.read_json(ws.obf_file("mini", "L1"))
        data["ok"] = False
        store.write_json(ws.obf_file("mini", "L1"), data)
        tasks, voids = pipeline.plan_tasks(ctx, mini_spec, ["L0", "L1"])
        assert {pt.task.level for pt in tasks} == {"L0"} and voids["L1"] == {"copy_invalid": 4}

    def test_name_not_renamed_is_voided_only_from_l1(self, planned, mini_spec):
        ctx, ws, _ = planned
        describe_all(ctx, mini_spec)
        data = store.read_json(ws.obf_file("mini", "L1"))
        del data["rename_map"]["calc_gcd"]
        store.write_json(ws.obf_file("mini", "L1"), data)
        tasks, voids = pipeline.plan_tasks(ctx, mini_spec, ["L0", "L1"])
        assert voids["L1"] == {"name_not_renamed": 1} and "L0" in voids and voids["L0"] == {}
        assert sum(pt.task.level == "L0" for pt in tasks) == 4 and sum(pt.task.level == "L1" for pt in tasks) == 3

    def test_task_fields_come_from_the_obfuscated_copy(self, planned, mini_spec):
        ctx, ws, targets = planned
        describe_all(ctx, mini_spec)
        tasks, _ = pipeline.plan_tasks(ctx, mini_spec, ["L0", "L1"])
        gcd0 = next(pt for pt in tasks if pt.task.level == "L0" and pt.original.name == "calc_gcd")
        gcd1 = next(pt for pt in tasks if pt.task.level == "L1" and pt.original.name == "calc_gcd")
        new_name = store.read_json(ws.obf_file("mini", "L1"))["rename_map"]["calc_gcd"]
        assert gcd0.task.obf_name == "calc_gcd" and gcd1.task.obf_name == new_name != "calc_gcd"
        assert gcd0.task.task_id == "mini:calc.c:calc_gcd@L0" and gcd1.task.target_id == gcd0.task.target_id
        assert new_name in gcd1.task.obf_signature and "calc_gcd" not in gcd1.task.obf_signature
        assert gcd0.task.description == gcd1.task.description == MOCK_DESCRIPTION.strip()
        assert gcd0.task.language_std == "c99" and gcd0.task.includes == gcd0.located.deps.includes
        assert gcd1.oracle.startswith("int ") and new_name in gcd1.oracle and "calc_gcd" not in gcd1.oracle
        assert gcd0.obf_root == ws.obf_root("mini", "L0") and gcd1.tree_hash == store.read_json(
            ws.obf_file("mini", "L1"))["tree_hash"]


class TestDescribeAndGenerateStages:
    def test_describe_is_idempotent_and_redoes_on_changes(self, planned, mini_spec):
        ctx, ws, _ = planned
        llm = Recorder("desc")
        first = pipeline.stage_describe(ctx, mini_spec, llm, model_name="desc")
        assert not first.skipped and first.detail["described"] == 4 and len(llm.calls) == 4
        assert all(c["meta"]["purpose"] == "describe" and c["temperature"] == 0.0 for c in llm.calls)
        recs = store.read_jsonl(ws.descriptions_file)
        assert len(recs) == 4 and not any(r["failed"] for r in recs)
        assert {r["corpus"] for r in recs} == {"mini"} and all(r["schema_version"] == SCHEMA_VERSION for r in recs)
        again = pipeline.stage_describe(ctx, mini_spec, llm, model_name="desc")
        assert again.skipped and len(llm.calls) == 4
        forced = pipeline.stage_describe(Context(ws=ws, jobs=1, seed=3, force=True), mini_spec, llm, model_name="desc")
        assert not forced.skipped and len(llm.calls) == 8
        other = pipeline.stage_describe(ctx, mini_spec, llm, model_name="other-model")      # 换描述模型 → 重做
        assert not other.skipped and len(llm.calls) == 12
        assert pipeline.load_descriptions(ws)[next(iter(pipeline.load_descriptions(ws)))]["model"] == "other-model"

    def test_description_leaking_a_project_name_is_retried_then_marked_failed(self, planned, mini_spec):
        ctx, ws, _ = planned

        class Leaky(Recorder):
            def chat(self, messages, **kw):
                reply = super().chat(messages, **kw)
                reply.text = "This function mixes calc_gcd with array_find and returns something useful. " * 12
                return reply

        res = pipeline.stage_describe(ctx, mini_spec, Leaky("leaky"), model_name="leaky")
        recs = store.read_jsonl(ws.descriptions_file)
        assert res.detail["failed"] == 4 and all(r["failed"] and r["description"] == "" for r in recs)
        assert all(r["leaked"] and r["reason"] == "leak" and r["leak_retries"] == 3 for r in recs)
        assert all("calc_gcd" in (r["rejected_text"] or "") for r in recs)         # 被拒的原文只放在 rejected_text 里

    def test_generate_requires_sane_sampling_arguments(self, planned, mini_spec):
        ctx, _, _ = planned
        with pytest.raises(StageError, match="temperature"):
            pipeline.stage_generate(ctx, [mini_spec], ["mock-oracle"], samples=2, temperature=0.0)
        with pytest.raises(StageError, match="samples"):
            pipeline.stage_generate(ctx, [mini_spec], ["mock-oracle"], samples=0)
        with pytest.raises(StageError, match="未知混淆级别"):
            pipeline.stage_generate(ctx, [mini_spec], ["mock-oracle"], levels=["L7"])

    def test_generate_writes_tasks_and_generations_and_is_resumable(self, planned, mini_spec):
        ctx, ws, _ = planned
        describe_all(ctx, mini_spec)
        (res,) = pipeline.stage_generate(ctx, [mini_spec], ["mock-oracle"], ["L0", "L1"])
        assert not res.skipped and res.detail["tasks"] == 8 and res.detail["calls"] == 8
        tasks = store.read_jsonl(ws.tasks_file)
        assert [t["task_id"] for t in tasks[:2]] == ["mini:calc.c:calc_gcd@L0", "mini:calc.c:calc_gcd@L1"]
        assert len(tasks) == 8 and all(t["prompt_digest"] == PROMPT_DIGEST[:16] and t["messages_hash"] for t in tasks)
        gens = store.read_jsonl(ws.generations_file("mock-oracle"))
        assert len(gens) == 8 and all(g["text"].startswith("```c") and g["sample_idx"] == 0 for g in gens)
        assert all(g["seed"] is None and g["temperature"] == 0.0 and g["finish_reason"] == "stop" for g in gens)
        (again,) = pipeline.stage_generate(ctx, [mini_spec], ["mock-oracle"], ["L0", "L1"])
        assert again.skipped
        # 掉了最后一条（模拟中断）→ 只补这一条
        lines = ws.generations_file("mock-oracle").read_text(encoding="utf-8").splitlines()
        ws.generations_file("mock-oracle").write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
        (resumed,) = pipeline.stage_generate(ctx, [mini_spec], ["mock-oracle"], ["L0", "L1"])
        assert resumed.detail["calls"] == 1 and len(store.read_jsonl(ws.generations_file("mock-oracle"))) == 8

    def test_dry_run_estimates_without_calling_or_writing(self, planned, mini_spec, monkeypatch):
        ctx, ws, targets = planned
        add_level(ws, mini_spec, targets, "L2")
        describe_all(ctx, mini_spec)
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)             # 估算是纯离线计算：不需要 key，不联网
        (res,) = pipeline.stage_generate(ctx, [mini_spec], ["deepseek-flash"], ["L0", "L1", "L2"], dry_run=True)
        assert "预估" in res.corpus and res.detail["tasks"] == 12 and res.detail["to_call"] == 12
        # L1 与 L2 只改函数体，提示词逐字相同：12 个任务只有 8 个不同的请求
        assert res.detail["unique_requests"] == 8 and res.detail["est_prompt_tokens"] > 0
        assert res.detail["max_completion_tokens"] == 8 * 4096
        assert not ws.tasks_file.exists() and not ws.generations_file("deepseek-flash").exists()
        assert not ws.cache_dir.exists() and not ws.usage_file.exists()
        # 多样本：每个样本是单独的请求（seed 不同）
        (multi,) = pipeline.stage_generate(ctx, [mini_spec], ["deepseek-flash"], ["L1", "L2"], samples=2,
                                           temperature=0.5, dry_run=True)
        assert multi.detail["to_call"] == 16 and multi.detail["unique_requests"] == 8

    def test_generate_reports_voided_tasks(self, planned, mini_spec):
        ctx, ws, _ = planned
        describe_all(ctx, mini_spec)
        recs = [r for r in store.read_jsonl(ws.descriptions_file) if not r["target_id"].endswith(":array_find")]
        store.write_jsonl(ws.descriptions_file, recs)
        (res,) = pipeline.stage_generate(ctx, [mini_spec], ["mock-oracle"], ["L0", "L1"])
        assert res.detail["tasks"] == 6 and "作废 2 个任务" in res.message and "no_description=2" in res.message
        (again,) = pipeline.stage_generate(ctx, [mini_spec], ["mock-oracle"], ["L0", "L1"])
        assert again.skipped and "no_description=2" in again.message

    def test_generate_scopes_merge_and_stale_records_are_replaced(self, planned, mini_spec):
        ctx, ws, _ = planned
        describe_all(ctx, mini_spec)
        pipeline.stage_generate(ctx, [mini_spec], ["mock-oracle"], ["L1"])
        pipeline.stage_generate(ctx, [mini_spec], ["mock-oracle"], ["L0"])
        assert {t["level"] for t in store.read_jsonl(ws.tasks_file)} == {"L0", "L1"}          # 后一次没有抹掉前一次
        assert len(store.read_jsonl(ws.generations_file("mock-oracle"))) == 8
        # 换 max_tokens → 输入哈希变了，旧回复整体换成新的，不是追加成两份
        (res,) = pipeline.stage_generate(ctx, [mini_spec], ["mock-oracle"], ["L0", "L1"], max_tokens=2048)
        assert res.detail["calls"] == 8
        gens = store.read_jsonl(ws.generations_file("mock-oracle"))
        assert len(gens) == 8 and {g["max_tokens"] for g in gens} == {2048}

    def test_multiple_samples_get_distinct_seeds_when_temperature_is_positive(self, planned, mini_spec, monkeypatch):
        ctx, ws, _ = planned
        describe_all(ctx, mini_spec)
        rec_llm = Recorder("rec")
        monkeypatch.setattr(pipeline, "build_llm", lambda name, ws_, oracle=None, use_cache=True: rec_llm)
        pipeline.stage_generate(ctx, [mini_spec], ["rec"], ["L0"], samples=3, temperature=0.7)
        by_task: dict[str, list] = {}
        for c in rec_llm.calls:
            by_task.setdefault(c["meta"]["task_id"], []).append((c["meta"]["sample_idx"], c["seed"]))
            assert c["temperature"] == 0.7 and c["meta"]["purpose"] == "generate" and c["meta"]["obf_signature"]
        assert len(by_task) == 4 and all(sorted(v) == [(0, 0), (1, 1), (2, 2)] for v in by_task.values())
        gens = store.read_jsonl(ws.generations_file("rec"))
        assert len(gens) == 12 and {g["sample_idx"] for g in gens} == {0, 1, 2}

    def test_extra_samples_from_an_earlier_run_are_kept_while_valid(self, planned, mini_spec, monkeypatch):
        ctx, ws, _ = planned
        describe_all(ctx, mini_spec)
        monkeypatch.setattr(pipeline, "build_llm", lambda name, ws_, oracle=None, use_cache=True: Recorder(name))
        pipeline.stage_generate(ctx, [mini_spec], ["rec"], ["L0"], samples=3, temperature=0.7)
        path = ws.generations_file("rec")
        assert len(store.read_jsonl(path)) == 12
        (again,) = pipeline.stage_generate(ctx, [mini_spec], ["rec"], ["L0"], samples=1, temperature=0.7)
        assert again.skipped and len(store.read_jsonl(path)) == 12            # 样本 1、2 还有效，留着
        # 温度变了：全部过期，包括这次没要的样本编号；只重新生成这次要的
        (res,) = pipeline.stage_generate(ctx, [mini_spec], ["rec"], ["L0"], samples=1, temperature=0.9)
        assert res.detail["calls"] == 4 and len(store.read_jsonl(path)) == 4
        assert {g["temperature"] for g in store.read_jsonl(path)} == {0.9}
        # --force：这次涉及的范围内全部重做
        ctx.force = True
        (res,) = pipeline.stage_generate(ctx, [mini_spec], ["rec"], ["L0"], samples=1, temperature=0.9)
        assert res.detail["calls"] == 4 and len(store.read_jsonl(path)) == 4

    def test_oracle_dict_gives_each_task_the_function_text_of_its_level(self, planned, mini_spec):
        ctx, ws, _ = planned
        describe_all(ctx, mini_spec)
        pipeline.stage_generate(ctx, [mini_spec], ["mock-oracle"], ["L0", "L1"])
        gens = {g["task_id"]: g["text"] for g in store.read_jsonl(ws.generations_file("mock-oracle"))}
        new = store.read_json(ws.obf_file("mini", "L1"))["rename_map"]["calc_gcd"]
        assert "int calc_gcd(" in gens["mini:calc.c:calc_gcd@L0"]
        assert f"{new}(" in gens["mini:calc.c:calc_gcd@L1"] and "calc_gcd" not in gens["mini:calc.c:calc_gcd@L1"]


def make_git_mirror(base: Path, url: str) -> Path:
    """在 base/mirror/mini 下造一个只有一次提交的本地 git 仓库（origin 指向 url），当作 CORPUS_MIRROR 用。"""
    repo = base / "mirror" / "mini"
    shutil.copytree(MINI, repo)

    def git(*args: str) -> None:
        subprocess.run(["git", "-c", "core.hooksPath=/dev/null", "-c", "user.name=t", "-c", "user.email=t@example.invalid",
                        *args], cwd=repo, check=True, capture_output=True)

    git("init", "-q")
    git("add", "-A")
    git("commit", "-q", "-m", "mini")
    git("remote", "add", "origin", url)
    return base / "mirror"


class TestPrepareAndTargetsStages:
    URL = "https://example.invalid/mini.git"

    def test_spec_without_url_needs_a_source_directory(self, scratch, mini_spec):
        ctx = Context(ws=Workspace.under(scratch).ensure(), sandbox=None)
        with pytest.raises(StageError, match="--src"):
            pipeline.stage_prepare(ctx, mini_spec, check_build=False)

    def test_local_source_is_copied_and_a_damaged_copy_is_restored(self, scratch, mini_spec):
        ws = Workspace.under(scratch).ensure()
        ctx = Context(ws=ws, sandbox=None)
        first = pipeline.stage_prepare(ctx, mini_spec, local_src=MINI, check_build=False)
        assert first.ok and not first.skipped and first.detail == {"tree": corpus.tree_digest(MINI)[:16],
                                                                   "build_checked": False}
        assert corpus.tree_digest(ws.pristine("mini")) == corpus.tree_digest(MINI)
        (ws.pristine("mini") / "calc.c").write_text("/* 被改坏了 */\n", encoding="utf-8")
        pipeline.stage_prepare(ctx, mini_spec, local_src=MINI, check_build=False)
        assert corpus.tree_digest(ws.pristine("mini")) == corpus.tree_digest(MINI)
        assert not ws.prepare_file("mini").exists()                      # 没做构建检查就不写状态文件
        assert pipeline.stage_prepare(ctx, mini_spec, check_build=False).ok         # 之后可以不再给 --src

    def test_url_spec_is_copied_from_the_local_mirror_and_pins_the_commit(self, scratch, mini_spec, monkeypatch):
        monkeypatch.setenv("CORPUS_MIRROR", str(make_git_mirror(scratch, self.URL)))
        spec = dataclasses.replace(mini_spec, url=self.URL, commit="")
        ws = Workspace.under(scratch / "ws").ensure()
        ctx = Context(ws=ws, sandbox=None)
        res = pipeline.stage_prepare(ctx, spec, check_build=False)
        assert res.ok and (ws.pristine("mini") / ".git").exists() and (ws.pristine("mini") / "calc.c").is_file()
        assert re.fullmatch(r"[0-9a-f]{40}", spec.commit)               # 没钉提交的 spec 用镜像的 HEAD 钉上
        assert corpus.tree_digest(ws.pristine("mini")) == corpus.tree_digest(MINI)    # .git 不参与哈希
        assert pipeline.stage_prepare(ctx, spec, check_build=False).ok

    @needs_docker
    def test_build_check_is_recorded_and_skipped_when_nothing_changed(self, scratch, mini_spec):
        ws = Workspace.under(scratch).ensure()
        ctx = Context(ws=ws, sandbox=CountingSandbox())
        first = pipeline.stage_prepare(ctx, mini_spec, local_src=MINI)
        assert first.ok and not first.skipped and ctx.sandbox.runs == 1
        saved = store.read_json(ws.prepare_file("mini"))
        assert saved["ok"] is True and saved["rc"] == 0 and saved["inputs"]["tree"] == first.detail["tree"]
        assert "mini_test" in saved["stdout_tail"] and saved["corpus"] == "mini"
        again = pipeline.stage_prepare(ctx, mini_spec, local_src=MINI)
        assert again.skipped and ctx.sandbox.runs == 1
        ctx.force = True
        assert not pipeline.stage_prepare(ctx, mini_spec, local_src=MINI).skipped and ctx.sandbox.runs == 2

    @needs_docker
    def test_failing_build_is_reported_not_raised_and_retried(self, scratch, mini_spec):
        ws = Workspace.under(scratch).ensure()
        broken = scratch / "broken"
        corpus.materialize(MINI, broken)
        (broken / "calc.c").write_text("int this is not C;\n", encoding="utf-8")
        ctx = Context(ws=ws, sandbox=CountingSandbox())
        res = pipeline.stage_prepare(ctx, mini_spec, local_src=broken)
        assert not res.ok and "构建或测试未通过" in res.message and store.read_json(ws.prepare_file("mini"))["ok"] is False
        assert not pipeline.stage_prepare(ctx, mini_spec, local_src=broken).skipped   # 失败的不当作「已完成」

    def test_static_only_selection_needs_neither_coverage_nor_sandbox(self, scratch, mini_spec):
        from bench.targets import TargetFilter
        ws = Workspace.under(scratch).ensure()
        corpus.materialize(MINI, ws.pristine("mini"))
        ctx = Context(ws=ws, sandbox=object(), jobs=1, seed=5, unsafe_debug=True)     # 跳过 stub-kill 必须显式放行
        flt = TargetFilter(max_per_corpus=5, require_exec=False, require_stub_kill=False)
        res = pipeline.stage_targets(ctx, mini_spec, flt)
        assert res.ok and res.detail["selected"] == 5 and not ws.coverage_file("mini").exists()
        recs = store.read_jsonl(ws.targets_file("mini"))
        assert len(recs) == 5 and all(r["exec_count"] is None and r["stub_killed"] is None
                                      and r["stub_kill_mode"] is None for r in recs)
        summary = store.read_json(ws.targets_summary_file("mini"))
        assert summary["stub_checked"] == 0 and summary["excluded"] == {"few_statements": 13, "not_sampled": 22}
        assert summary["unverified"] is True                              # 没做 stub-kill / 执行过滤：摘要里如实标记
        assert summary["input_hash"] == recs[0]["input_hash"] and summary["filter"]["max_per_corpus"] == 5
        assert pipeline.stage_targets(ctx, mini_spec, flt).skipped
        assert not pipeline.stage_targets(ctx, mini_spec, flt, scan_all=False).skipped      # scan_all 是输入的一部分
        first = {r["target_id"] for r in recs}
        again = pipeline.stage_targets(Context(ws=ws, sandbox=object(), jobs=1, seed=6, unsafe_debug=True), mini_spec, flt)
        assert not again.skipped                                               # 换种子：重新抽样
        assert {r["target_id"] for r in store.read_jsonl(ws.targets_file("mini"))} != first
        back = pipeline.stage_targets(ctx, mini_spec, flt)                     # 换回种子：同样的输入得到同样的目标
        assert {r["target_id"] for r in store.read_jsonl(ws.targets_file("mini"))} == first and not back.skipped

    def test_load_targets_without_a_targets_stage_names_the_command(self, scratch):
        with pytest.raises(StageError, match="targets"):
            pipeline.load_targets(Workspace.under(scratch).ensure(), "mini")


class TestStubKillMode:
    """targets.StubVerdict.kill_mode 是 e2e 里「wrong 的归因必须与 stub-kill 一致」这条断言的依据，这里单独钉住它的口径。"""

    @pytest.mark.parametrize("signal, timed_out", [
        (None, False), (11, False), (6, False), (8, False), (7, False), (4, False), (9, False), (9, True),
        (None, True), (11, True),
    ])
    def test_kill_mode_agrees_with_attribution_of_the_same_observation(self, signal, timed_out):
        from bench.attribute import attribute
        from bench.postprocess import Candidate
        from bench.runner import EvalResult
        from bench.targets import StubVerdict
        ev = EvalResult(task_id="t", model="m", sample_idx=0, stage="test", rc=128 + signal if signal else 1,
                        timed_out=timed_out, signal=signal, diagnostics=[], test_stdout_tail="", duration_s=1.0)
        category, _ = attribute(ev, Candidate("int f(void) { return 0; }", []))
        assert StubVerdict("killed", "test", signal=signal, timed_out=timed_out).kill_mode == category

    def test_only_killed_verdicts_have_a_mode(self):
        from bench.targets import StubVerdict
        assert StubVerdict("survived", "pass").kill_mode is None
        assert StubVerdict("build_error", "build").kill_mode is None and StubVerdict("unsupported").kill_mode is None
        assert StubVerdict("killed", "test").killed and StubVerdict("killed", "test").kill_mode == "test_fail"

    def test_precomputed_counts_replace_the_coverage_build(self, mini_spec):
        from bench import targets as T
        counts = {f"{t.file}:{t.name}": (0 if t.name == "calc_gcd" else 3) for t in extract_functions(mini_spec, MINI)}
        flt = T.TargetFilter(require_stub_kill=False)
        sel = T.select(mini_spec, MINI, object(), flt, 1, precomputed_counts=counts)     # 沙箱是个摆设：用到就会报错
        assert "calc_gcd" not in {t.name for t in sel.targets} and sel.excluded["not_executed"] == 1
        assert sel.exec_counts and set(sel.exec_counts.values()) == {3}
        recs = sel.records()
        assert all(r["exec_count"] == 3 and r["stub_killed"] is None and r["stub_kill_mode"] is None for r in recs)


class TinyServer:
    """本机回环地址上的最小 OpenAI 兼容服务：对每个请求返回同一段代码块，记录收到的请求体与请求头。"""

    REPLY = "```c\nint f(void) { return 0; }\n```"

    def __init__(self) -> None:
        self.requests: list[dict] = []
        self._lock = threading.Lock()
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                with outer._lock:
                    outer.requests.append({"body": json.loads(raw), "auth": self.headers.get("Authorization", "")})
                body = json.dumps({"id": "x", "model": "srv", "usage": {"prompt_tokens": 40, "completion_tokens": 8},
                                   "choices": [{"index": 0, "finish_reason": "stop",
                                                "message": {"role": "assistant", "content": TinyServer.REPLY}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


class TestRealClientThroughTheStages:
    """真实的 OpenAICompat 客户端（对着本机假服务）经过 describe / generate 阶段：缓存、去重、usage 日志、key 不落盘。"""

    KEY = "bench-test-secret-0123456789"

    @pytest.fixture()
    def served(self, planned, mini_spec, monkeypatch):
        from bench.llm import Cache, OpenAICompat
        ctx, ws, targets = planned
        ctx.jobs = 1                                # 单线程：L1 与 L2 的相同请求不会同时发出，缓存去重才是确定的
        add_level(ws, mini_spec, targets, "L2")
        server = TinyServer()
        monkeypatch.setenv("BENCH_TEST_API_KEY", self.KEY)
        llm = OpenAICompat(server.url, "BENCH_TEST_API_KEY", "fake-model", cache=Cache(ws.cache_dir),
                           usage_path=ws.usage_file, use_proxy=False)
        monkeypatch.setattr(pipeline, "build_llm", lambda name, ws_, oracle=None, use_cache=True: llm)
        yield ctx, ws, server
        server.close()

    def test_identical_prompts_of_l1_and_l2_cost_one_http_request(self, served, mini_spec):
        ctx, ws, server = served
        describe_all(ctx, mini_spec)                                 # Recorder 写的描述，不经过 HTTP
        (res,) = pipeline.stage_generate(ctx, [mini_spec], ["fake-model"], ["L0", "L1", "L2"])
        assert res.detail["tasks"] == 12 and res.detail["calls"] == 12
        assert len(server.requests) == 8 and res.detail["cached"] == 4      # L0 四个 + L1 四个；L2 全部命中缓存
        gens = store.read_jsonl(ws.generations_file("fake-model"))
        assert sum(g["cached"] for g in gens) == 4 and all(g["text"] == TinyServer.REPLY for g in gens)
        assert {g["level"] for g in gens if g["cached"]} == {"L2"}
        # 请求体：模型名、温度 0、max_tokens、没有 seed；system + user 两条消息
        body = server.requests[0]["body"]
        assert body["model"] == "fake-model" and body["temperature"] == 0.0 and body["max_tokens"] == 4096
        assert "seed" not in body and [m["role"] for m in body["messages"]] == ["system", "user"]
        # usage 日志只记真实调用（命中缓存的不产生费用，不记）
        usage = store.read_jsonl(ws.usage_file)
        assert len(usage) == 8 and {u.get("purpose") for u in usage} == {"generate"}
        assert all(u["prompt_tokens"] == 40 and u["completion_tokens"] == 8 for u in usage)
        # 再跑一遍：全部跳过，不再发请求
        (again,) = pipeline.stage_generate(ctx, [mini_spec], ["fake-model"], ["L0", "L1", "L2"])
        assert again.skipped and len(server.requests) == 8

    def test_key_goes_to_the_authorization_header_and_nowhere_else(self, served, mini_spec):
        ctx, ws, server = served
        describe_all(ctx, mini_spec)
        pipeline.stage_generate(ctx, [mini_spec], ["fake-model"], ["L0"])
        assert {r["auth"] for r in server.requests} == {f"Bearer {self.KEY}"}
        for path in ws.results.rglob("*"):
            if path.is_file():
                assert self.KEY not in path.read_text(encoding="utf-8"), path.name

    def test_samples_are_separate_requests_and_still_deduplicated_across_levels(self, served, mini_spec):
        ctx, ws, server = served
        describe_all(ctx, mini_spec)
        (res,) = pipeline.stage_generate(ctx, [mini_spec], ["fake-model"], ["L1", "L2"], samples=2, temperature=0.7)
        assert res.detail["calls"] == 16 and len(server.requests) == 8        # 4 个目标 × 2 个样本，L2 命中 L1 的缓存
        assert sorted({r["body"].get("seed") for r in server.requests}) == [0, 1]
        assert all(r["body"]["temperature"] == 0.7 for r in server.requests)

    def test_api_errors_do_not_abort_the_stage_and_a_rerun_resumes(self, served, mini_spec):
        ctx, ws, server = served
        describe_all(ctx, mini_spec)
        first_status = {"n": 0}

        def flaky_chat(self_llm, *a, **k):          # 前两次调用抛出不可重试的错误，之后恢复正常
            first_status["n"] += 1
            if first_status["n"] <= 2:
                from bench.llm import LLMHTTPError
                raise LLMHTTPError(400, "bad request (injected)")
            return orig(self_llm, *a, **k)

        from bench.llm import OpenAICompat
        orig = OpenAICompat.chat
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(OpenAICompat, "chat", flaky_chat)
            (res,) = pipeline.stage_generate(ctx, [mini_spec], ["fake-model"], ["L0"])
        assert not res.ok and "2 次调用失败" in res.message and res.detail["calls"] == 2
        assert len(store.read_jsonl(ws.generations_file("fake-model"))) == 2
        (resumed,) = pipeline.stage_generate(ctx, [mini_spec], ["fake-model"], ["L0"])
        assert resumed.ok and resumed.detail["calls"] == 2 and len(store.read_jsonl(ws.generations_file("fake-model"))) == 4


# ----------------------------------------------------------------------
# 命令行（不需要 docker 的部分）
# ----------------------------------------------------------------------
COMMANDS = ("prepare", "coverage", "targets", "obfuscate", "verify", "describe", "generate", "evaluate", "report", "e2e")


class TestCli:
    def test_every_command_accepts_the_common_options(self):
        ap = build_parser()
        for cmd in COMMANDS:
            extra = ["--model", "m"] if cmd == "evaluate" else []
            ns = ap.parse_args([cmd, "--corpus", "a", "b", "--seed", "5", "--jobs", "3", "--force", *extra])
            assert (ns.cmd, ns.corpus, ns.seed, ns.jobs, ns.force) == (cmd, ["a", "b"], 5, 3, True)

    def test_defaults(self):
        ap = build_parser()
        ns = ap.parse_args(["targets"])
        assert ns.corpus is None and ns.seed == pipeline.DEFAULT_SEED and ns.jobs >= 1 and not ns.force
        assert (ns.max_per_corpus, ns.min_statements, ns.min_lines, ns.max_lines) == (50, 3, 4, 120)
        ns = ap.parse_args(["generate"])
        assert (ns.samples, ns.temperature, ns.max_tokens, ns.levels) == (1, 0.0, 4096, None)
        assert ns.model == ["deepseek-flash"]
        assert ap.parse_args(["describe"]).model == "deepseek-v4-pro"
        assert ap.parse_args(["obfuscate"]).strict_stringify == "auto"
        assert ap.parse_args(["e2e"]).backend == "mock"
        assert ap.parse_args(["report"]).formats == ["md", "csv", "svg"]

    def test_default_model_names_are_current_presets(self):
        from bench import cli
        from bench.llm import PRESETS
        for name in (cli.DEFAULT_DESCRIBE_MODEL, cli.DEFAULT_GENERATE_MODEL):
            assert name in PRESETS and not PRESETS[name].legacy          # 不能默认用官方有弃用通知的旧模型名

    def test_levels_option_accepts_both_spellings_and_rejects_unknown(self):
        ap = build_parser()
        assert ap.parse_args(["obfuscate", "--level", "L1", "L3"]).levels == ["L1", "L3"]
        assert ap.parse_args(["generate", "--levels", "L0"]).levels == ["L0"]
        with pytest.raises(SystemExit) as exc:
            ap.parse_args(["obfuscate", "--level", "L9"])
        assert exc.value.code == 2

    def test_unknown_command_and_missing_command_exit_2(self):
        for argv in (["frobnicate"], []):
            with pytest.raises(SystemExit) as exc:
                build_parser().parse_args(argv)
            assert exc.value.code == 2

    def test_python_dash_m_bench_lists_all_commands(self):
        out = subprocess.run([sys.executable, "-m", "bench", "--help"], cwd=ROOT, capture_output=True, text=True,
                             timeout=120)
        assert out.returncode == 0
        assert all(c in out.stdout for c in COMMANDS)

    def test_report_without_results_returns_1(self, scratch, capsys):
        assert main(["report", "--tag", "nothing", "--workspace", str(scratch)]) == 1
        assert "记录 0 条" in capsys.readouterr().out

    def test_report_on_synthetic_results(self, scratch, capsys):
        ws = Workspace.under(scratch).ensure()
        for c in ("c1", "c2"):
            for lv in ("L0", "L1"):
                write_obf(ws, c, lv)
        put_results(ws, "syn", synthetic_records())
        assert main(["report", "--tag", "syn", "--workspace", str(scratch), "--format", "md", "csv"]) == 0
        out = capsys.readouterr().out
        assert "纳入统计 48 条" in out and "md:" in out and "csv:" in out and "svg:" not in out
        assert ws.summary_file("syn", ".md").is_file() and not ws.summary_file("syn", ".svg").exists()

    def test_report_works_without_any_corpus_spec_unless_one_is_named(self, scratch, capsys, monkeypatch):
        ws = Workspace.under(scratch).ensure()
        write_obf(ws, "c1", "L0")
        put_results(ws, "syn", [stamp(rec(target=f"t{i}")) for i in range(3)])
        monkeypatch.setattr(corpus, "SPEC_DIR", scratch / "no-specs-here")
        assert main(["report", "--tag", "syn", "--workspace", str(scratch)]) == 0
        assert "纳入统计 3 条" in capsys.readouterr().out
        assert main(["report", "--tag", "syn", "--workspace", str(scratch), "--corpus", "ghost"]) == 2

    def test_report_flags_stale_records_and_allow_stale_keeps_them(self, scratch, capsys):
        ws = Workspace.under(scratch).ensure()
        write_obf(ws, "c1", "L0")
        records = [stamp(rec(target=f"t{i}")) for i in range(4)] + [stamp(rec(target="old"), tree="stale")]
        put_results(ws, "syn", records)
        assert main(["report", "--tag", "syn", "--workspace", str(scratch)]) == 0
        out = capsys.readouterr().out
        assert "纳入统计 4 条" in out and "输入哈希不符 1 条" in out and "混淆副本已变" in out
        main(["report", "--tag", "syn", "--workspace", str(scratch), "--allow-stale"])
        assert "纳入统计 5 条" in capsys.readouterr().out

    def test_report_exit_code_is_nonzero_for_bad_lines_or_missing_samples_unless_allowed(self, scratch, capsys):
        ws = Workspace.under(scratch).ensure()
        for c in ("c1", "c2"):
            for lv in ("L0", "L1"):
                write_obf(ws, c, lv)
        put_results(ws, "clean", synthetic_records())
        assert main(["report", "--tag", "clean", "--workspace", str(scratch)]) == 0         # 干净的结果：0
        capsys.readouterr()
        # 坏行（进程被杀时留下的半行）：默认非零，--allow-missing 放行；命令行里照样点出来
        put_results(ws, "bad", synthetic_records())
        with ws.results_file("bad").open("a", encoding="utf-8") as f:
            f.write('{"half": ')
        argv = ["report", "--tag", "bad", "--workspace", str(scratch)]
        assert main(argv) == 1
        assert "1 条坏行" in capsys.readouterr().out
        assert main([*argv, "--allow-missing"]) == 0
        assert "1 条坏行" in capsys.readouterr().out
        # 缺样本（completeness 里有缺口）：同样默认非零
        put_results(ws, "gap", synthetic_records())
        store.write_json(ws.completeness_file("gap"), {"cells": [
            {"model": "model-b", "level": "L1", "corpus": "c1", "expected": 8, "have": 6, "failed_calls": 2,
             "stale_generations": 0, "not_generated": 0, "missing": 2}]})
        argv = ["report", "--tag", "gap", "--workspace", str(scratch)]
        assert main(argv) == 1
        assert "样本不完整" in ws.summary_file("gap", ".md").read_text(encoding="utf-8")
        capsys.readouterr()
        assert main([*argv, "--allow-missing"]) == 0
        # 一条记录都没有：--allow-missing 也救不了
        assert main(["report", "--tag", "none", "--workspace", str(scratch), "--allow-missing"]) == 1

    def test_usage_errors_exit_2_with_a_clear_message(self, scratch, capsys, monkeypatch):
        spec = str(MINI / "spec.json")
        base = ["--corpus", spec, "--workspace", str(scratch)]
        assert main(["generate", "--model", "mock-oracle", "--samples", "2", *base]) == 2
        assert "temperature" in capsys.readouterr().err
        assert main(["describe", "--model", "mock-oracle", *base]) == 2           # 还没有 prepare
        assert "prepare" in capsys.readouterr().err
        assert main(["describe", "--model", "no-such-model", *base]) == 2
        assert "no-such-model" in capsys.readouterr().err
        assert main(["report", "--tag", "x", "--corpus", str(scratch / "none.json"), "--workspace", str(scratch)]) == 2
        assert main(["prepare", "--src", str(MINI), "--corpus", spec, str(scratch / "x.json"), "--workspace",
                     str(scratch)]) == 2

    def test_missing_api_key_is_reported_at_the_first_real_request_without_echoing_anything_secret(
            self, planned, capsys, monkeypatch):
        ctx, ws, _ = planned
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        rc = main(["describe", "--model", "deepseek-flash", "--corpus", str(MINI / "spec.json"),
                   "--workspace", str(ws.results.parent)])
        err = capsys.readouterr().err
        assert rc == 2 and "DEEPSEEK_API_KEY" in err and "没有设置" in err
        assert not ws.descriptions_file.exists()                          # 没发出任何请求，也没写任何描述

    def test_unsafe_debug_is_the_only_switch_for_relaxing_the_trust_measures(self):
        ap = build_parser()
        assert ap.parse_args(["targets"]).unsafe_debug is False
        assert ap.parse_args(["generate", "--unsafe-debug"]).unsafe_debug is True
        assert "--no-cache" in ap.format_help() or ap.parse_args(["generate", "--no-cache"]).no_cache is True
        assert ap.parse_args(["evaluate", "--model", "m", "--allow-missing"]).allow_missing is True
        assert ap.parse_args(["report", "--allow-missing"]).allow_missing is True

    def test_skipping_stub_kill_is_refused_without_unsafe_debug(self, planned, capsys):
        ctx, ws, _ = planned
        base = ["--corpus", str(MINI / "spec.json"), "--workspace", str(ws.results.parent)]
        assert main(["targets", "--no-stub-kill", *base]) == 2
        err = capsys.readouterr().err
        assert "stub-kill" in err and "--unsafe-debug" in err

    def test_mock_models_are_refused_for_describe_without_unsafe_debug(self, planned, capsys):
        ctx, ws, _ = planned
        base = ["--corpus", str(MINI / "spec.json"), "--workspace", str(ws.results.parent)]
        assert main(["describe", "--model", "mock-oracle", *base]) == 2
        assert "离线自测" in capsys.readouterr().err and not ws.descriptions_file.exists()
        assert main(["describe", "--model", "mock-oracle", "--unsafe-debug", *base]) == 0
        assert ws.descriptions_file.is_file()

    def test_unexpected_errors_leave_a_traceback_file(self, scratch, capsys, monkeypatch):
        from bench import cli

        def boom(args, ctx, specs):
            raise RuntimeError("docker 挂了")

        monkeypatch.setitem(cli._COMMANDS, "report", boom)
        assert main(["report", "--tag", "x", "--workspace", str(scratch)]) == 1
        err = capsys.readouterr().err
        log_file = Workspace.under(scratch).results / "bench-error.log"
        assert "bench-error.log" in err and log_file.is_file()
        text = log_file.read_text(encoding="utf-8")
        assert "RuntimeError: docker 挂了" in text and "Traceback" in text

    def test_unexpected_runtime_errors_are_reported_not_raised(self, scratch, capsys, monkeypatch):
        from bench import cli

        def boom(args, ctx, specs):
            raise RuntimeError("docker 挂了")

        monkeypatch.setitem(cli._COMMANDS, "report", boom)
        assert main(["report", "--tag", "x", "--workspace", str(scratch)]) == 1
        assert "运行失败：RuntimeError: docker 挂了" in capsys.readouterr().err

    def test_e2e_refuses_to_use_or_wipe_a_foreign_directory(self, scratch):
        (scratch / "precious.txt").write_text("keep me", encoding="utf-8")
        for fresh in (True, False):
            with pytest.raises(StageError, match="拒绝使用"):
                run_e2e(scratch, sandbox=object(), fresh=fresh)
        assert (scratch / "precious.txt").read_text(encoding="utf-8") == "keep me"

    def test_stage_result_line_format(self):
        ok = pipeline.StageResult("targets", "c", detail={"selected": 5, "nested": {"a": 1}}, seconds=2.0)
        assert ok.line() == "[targets] c: 完成（selected=5） 2.0s"
        skipped = pipeline.StageResult("targets", "c", skipped=True)
        assert "跳过" in skipped.line()
        bad = pipeline.StageResult("obfuscate", "c@L1", ok=False, message="boom")
        assert "失败" in bad.line() and "boom" in bad.line()

    def test_run_ordered_keeps_order_and_runs_concurrently(self):
        active, peak = [0], [0]
        lock = threading.Lock()

        def slow(x):
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            import time
            time.sleep(0.05)
            with lock:
                active[0] -= 1
            return x * x

        assert list(pipeline.run_ordered(slow, list(range(12)), 4)) == [x * x for x in range(12)]
        assert peak[0] > 1
        assert list(pipeline.run_ordered(lambda x: x + 1, [1, 2, 3], 1)) == [2, 3, 4]
        assert list(pipeline.run_ordered(lambda x: x, [], 4)) == []


# ----------------------------------------------------------------------
# 需要 docker：obfuscate() 的 verify
# ----------------------------------------------------------------------
@needs_docker
class TestObfuscateVerify:
    def test_verify_passes_is_recorded_and_skipped_on_rerun(self, scratch, mini_spec):
        sandbox = DockerSandbox()
        kw = dict(sandbox=sandbox, results_dir=scratch, work_root=scratch / "work")
        res = obfuscate(mini_spec, MINI, "L3", 5, scratch / "o", **kw)
        assert res.ok and res.verify["ok"] and res.verify["tests_passed"]
        assert res.verify["relocated"] == res.verify["n_targets"] == 40 and res.verify["missing"] == []
        assert res.stats["attempts"] == [{"strict_stringify": False, "ok": True, "build_rc": 0, "missing": 0}]
        saved = store.read_json(scratch / "obf-mini-L3.json")
        assert saved["ok"] is True and saved["verify"]["ok"] is True and saved["problems"] == []
        again = obfuscate(mini_spec, MINI, "L3", 5, scratch / "o", **kw)
        assert again.skipped and again.verify == res.verify

    def test_unlocatable_target_fails_verify_and_the_copy_is_redone_next_time(self, scratch, mini_spec):
        sandbox = DockerSandbox()
        targets = [t for t in extract_functions(mini_spec, MINI) if t.name == "calc_gcd"]
        bogus = {"target_id": "mini:calc.c:no_such_function", "file": "calc.c", "name": "no_such_function"}
        kw = dict(sandbox=sandbox, results_dir=scratch, work_root=scratch / "work", targets=targets + [bogus])
        res = obfuscate(mini_spec, MINI, "L1", 5, scratch / "o", **kw)
        assert not res.ok and res.verify["tests_passed"] and res.verify["missing"] == [bogus["target_id"]]
        assert res.verify["relocated"] == 1 and res.problems and "无法按映射名重新定位" in res.problems[0]
        assert store.read_json(scratch / "obf-mini-L1.json")["ok"] is False
        assert not obfuscate(mini_spec, MINI, "L1", 5, scratch / "o", **kw).skipped     # 失败的副本下次重做

    def test_broken_project_tests_fail_verify(self, scratch, mini_spec):
        """pristine 本身的测试过不了（被改坏）时，verify 如实报告测试未通过。"""
        bad = scratch / "bad"
        corpus.materialize(MINI, bad)
        text = (bad / "calc.c").read_text(encoding="utf-8")
        head = "int calc_gcd(int a, int b)\n{"
        assert head in text
        (bad / "calc.c").write_text(text.replace(head, head + "\n    return 7;", 1), encoding="utf-8")
        res = obfuscate(mini_spec, bad, "L0", 1, scratch / "o", sandbox=DockerSandbox(), results_dir=scratch,
                        work_root=scratch / "work", targets=[])
        assert not res.ok and res.verify["tests_passed"] is False and "项目自带测试未通过" in res.problems[0]

    def test_default_sandbox_is_created_when_none_given(self, scratch, mini_spec):
        res = obfuscate(mini_spec, MINI, "L0", 1, scratch / "o", results_dir=scratch, work_root=scratch / "work",
                        targets=[])
        assert res.ok and res.verify["build_rc"] == 0


# ----------------------------------------------------------------------
# 需要 docker：mini 夹具上的全链路 e2e
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def e2e():
    """跑一次完整的 e2e（约 2 分钟），模块内所有断言共用它的产出；结束后清理。"""
    base = ROOT / "tmp" / "pytest-e2e" / f"run-{uuid.uuid4().hex[:8]}"
    result = run_e2e(base, jobs=6, seed=20261027)
    result.base = base                                                   # type: ignore[attr-defined]
    yield result
    shutil.rmtree(base, ignore_errors=True)


def read_results(e2e_result) -> list[dict]:
    return store.read_jsonl(e2e_result.ws.results_file("e2e"))


@needs_docker
class TestE2E:
    N_MODELS = 3

    def test_built_in_checks_all_pass(self, e2e):
        assert e2e.ok, e2e.format()
        assert all(s.ok for s in e2e.stages), [s.line() for s in e2e.stages if not s.ok]
        assert len(e2e.checks) >= 10 and all(c.ok for c in e2e.checks)

    def test_all_levels_times_three_modes_were_run(self, e2e):
        recs = read_results(e2e)
        n_targets = len(store.read_jsonl(e2e.ws.targets_file("mini")))
        assert n_targets >= 20                                           # mini 的候选里绝大多数都入选
        cells = {(r["model"], r["level"]) for r in recs}
        assert cells == {(m, lv) for m in ("mock-oracle", "mock-broken", "mock-wrong") for lv in LEVELS}
        for m in ("mock-oracle", "mock-broken", "mock-wrong"):
            for lv in LEVELS:
                assert sum(r["model"] == m and r["level"] == lv for r in recs) == n_targets
        assert len(recs) == n_targets * len(LEVELS) * self.N_MODELS

    def test_oracle_mode_is_100_percent_pass_with_identical_code(self, e2e):
        oracle = [r for r in read_results(e2e) if r["model"] == "mock-oracle"]
        assert oracle and all(r["category"] == "pass" and r["stage"] == "pass" for r in oracle)
        assert all(r["similarity"] == 1.0 and r["issues"] == [] and r["detail"] == "" for r in oracle)

    def test_broken_mode_is_100_percent_compile_syntax(self, e2e):
        broken = [r for r in read_results(e2e) if r["model"] == "mock-broken"]
        assert broken and all(r["category"] == "compile_syntax" and r["stage"] == "syntax" for r in broken)
        assert all(r["rc"] != 0 and any(d.get("kind") == "error" for d in r["diagnostics"]) for r in broken)

    def test_wrong_mode_never_passes_and_matches_the_stub_kill_record(self, e2e):
        wrong = [r for r in read_results(e2e) if r["model"] == "mock-wrong"]
        kill = {t["target_id"]: t["stub_kill_mode"] for t in store.read_jsonl(e2e.ws.targets_file("mini"))}
        assert wrong and not any(r["category"] == "pass" for r in wrong)
        assert all(r["stage"] == "test" for r in wrong)                  # 编译通过、在项目测试阶段才失败
        assert all(r["category"] == kill[r["target_id"]] for r in wrong)
        assert sum(r["category"] == "test_fail" for r in wrong) >= 0.9 * len(wrong)
        # 同一个目标在四个级别上的归因必须一致：混淆不改变行为
        for tid in kill:
            assert len({r["category"] for r in wrong if r["target_id"] == tid}) == 1

    def test_every_target_was_killed_by_its_stub(self, e2e):
        targets = store.read_jsonl(e2e.ws.targets_file("mini"))
        assert all(t["stub_killed"] is True and t["exec_count"] > 0 for t in targets)
        assert {t["stub_kill_mode"] for t in targets} <= {"test_fail", "runtime_crash", "runtime_timeout"}
        summary = store.read_json(e2e.ws.targets_summary_file("mini"))
        assert summary["selected"] == len(targets) and summary["stub_verdicts"] == {"killed": summary["stub_checked"]}
        assert summary["excluded"].get("stub_survived", 0) == 0 and summary["scan_all"] is True

    def test_obf_files_record_verify_gate_and_tree_hash(self, e2e):
        n = len(store.read_jsonl(e2e.ws.targets_file("mini")))
        pristine = corpus.tree_digest(e2e.ws.pristine("mini"))[:16]
        hashes = set()
        for lv in LEVELS:
            obf = store.read_json(e2e.ws.obf_file("mini", lv))
            assert obf["ok"] and obf["verify"]["ok"] and obf["problems"] == [] and obf["level"] == lv
            assert obf["oracle_gate"]["checked"] == obf["oracle_gate"]["passed"] == n
            assert obf["oracle_gate"]["failed"] == [] and obf["oracle_gate"]["copy_invalid"] is False
            assert corpus.tree_digest(e2e.ws.obf_root("mini", lv))[:16] == obf["tree_hash"]
            assert obf["inputs"]["pristine"] == pristine and obf["inputs"]["seed"] == 20261027
            hashes.add(obf["tree_hash"])
        assert len(hashes) == 4 and store.read_json(e2e.ws.obf_file("mini", "L0"))["tree_hash"] == pristine
        assert store.read_json(e2e.ws.obf_file("mini", "L0"))["rename_map"] == {}
        assert len(store.read_json(e2e.ws.obf_file("mini", "L1"))["rename_map"]) > 100

    def test_prompts_hide_original_names_from_l1_on(self, e2e):
        """符号混淆 + 描述 + 出题接起来之后：L1–L3 的提示词里不能再出现目标函数的真名，L0 里必须有。"""
        from bench.prompt import Task, build_messages
        tasks = [Task.from_dict(t) for t in store.read_jsonl(e2e.ws.tasks_file)]
        orig = {t["target_id"]: t["name"] for t in store.read_jsonl(e2e.ws.targets_file("mini"))}
        assert len(tasks) == len(orig) * len(LEVELS)
        for t in tasks:
            user = build_messages(t)[1]["content"]
            pattern = re.compile(rf"\b{re.escape(orig[t.target_id])}\b")
            if t.level == "L0":
                assert pattern.search(user) and t.obf_name == orig[t.target_id]
            else:
                assert not pattern.search(user), (t.task_id, "提示词泄漏了原函数名")
                assert t.obf_name != orig[t.target_id] and re.fullmatch(r"f_\d+", t.obf_name)

    def test_descriptions_are_shared_by_levels_and_leak_free(self, e2e):
        from bench.describe import leaks
        from bench.pipeline import project_names
        descs = store.read_jsonl(e2e.ws.descriptions_file)
        targets = store.read_jsonl(e2e.ws.targets_file("mini"))
        assert len(descs) == len(targets) and not any(d["failed"] for d in descs)
        names = project_names(corpus.load_spec_file(MINI / "spec.json"), e2e.ws.pristine("mini"))
        assert names and all(not leaks(d["description"], names) for d in descs)
        by_target = {d["target_id"]: d["description"] for d in descs}
        for t in store.read_jsonl(e2e.ws.tasks_file):
            assert t["description"] == by_target[t["target_id"]]          # 四个级别共用同一份规格

    def test_every_record_carries_input_hashes_and_no_host_paths(self, e2e):
        recs = read_results(e2e)
        spec_hash = store.spec_digest(corpus.load_spec_file(MINI / "spec.json"))
        assert all(r["schema_version"] == SCHEMA_VERSION and r["inputs"]["prompt"] == PROMPT_DIGEST[:16] for r in recs)
        assert all(r["inputs"]["spec"] == spec_hash and r["input_hash"] for r in recs)
        # 上游的哈希都带上了：回复（生成记录的输入哈希 + 回复摘要）、提示词、oracle gate、评测代码、沙箱
        assert all(set(report.REQUIRED_INPUTS) <= set(r["inputs"]) for r in recs)
        assert all(r["inputs"]["code"] == code_digest("evaluate") and r["inputs"]["env"] for r in recs)
        gens = {(m, g["task_id"], g["sample_idx"]): g for m in ("mock-oracle", "mock-broken", "mock-wrong")
                for g in store.read_jsonl(e2e.ws.generations_file(m))}
        assert all(gens[(r["model"], r["task_id"], r["sample_idx"])]["input_hash"] == r["inputs"]["gen"] for r in recs)
        assert all(r["sandbox"]["image"] == "gcc:13" and r["sandbox"]["image_id"].startswith("sha256:") for r in recs)
        assert all(r["temperature"] == 0.0 and r["max_tokens"] == 4096 and r["seed"] is None for r in recs)
        assert {r["stub_kill_mode"] for r in recs} <= {"test_fail", "runtime_crash", "runtime_timeout"}
        assert all(g["llm_fp"] and g["temperature"] == 0.0 and "system_fingerprint" in g for g in gens.values())
        for lv in LEVELS:
            tree = store.read_json(e2e.ws.obf_file("mini", lv))["tree_hash"]
            assert all(r["inputs"]["tree"] == tree for r in recs if r["level"] == lv)
        needles = (str(ROOT), "/" + "home/", "sk" + "-")        # 项目绝对路径、家目录路径、API key 的常见前缀
        for path in e2e.ws.results.iterdir():
            if path.is_file():
                text = path.read_text(encoding="utf-8")
                assert not [n for n in needles if n in text], path.name

    def test_rerunning_every_stage_skips_everything(self, e2e):
        assert next(c for c in e2e.checks if c.name.startswith("幂等")).ok
        assert e2e.report.n_stale == 0

    def test_report_files_and_contents(self, e2e):
        paths = e2e.report.paths
        md = paths["md"].read_text(encoding="utf-8")
        for m in ("mock-oracle", "mock-broken", "mock-wrong"):
            assert m in md
        assert "| mock-oracle | L3 |" in md and "100.0%" in md and "输入校验" not in md
        assert "runtime_crash" in md                                    # 归因分布表里有这一类
        rows = list(csv.DictReader(paths["csv"].open(encoding="utf-8")))
        oracle_all = [r for r in rows if r["model"] == "mock-oracle" and r["corpus"] == "ALL"]
        assert len(oracle_all) == 4 and all(float(r["pass_rate"]) == 1.0 for r in oracle_all)
        wrong_all = [r for r in rows if r["model"] == "mock-wrong" and r["corpus"] == "ALL"]
        assert all(float(r["pass_rate"]) == 0.0 and float(r["compile_rate"]) == 1.0 for r in wrong_all)
        root = ET.fromstring(paths["svg"].read_text(encoding="utf-8"))
        assert root.tag.endswith("svg")
        text = "\n".join(p.read_text(encoding="utf-8") for p in paths.values())
        assert not [w for w in BANNED if w in text]

    def test_changes_to_upstream_files_are_detected_by_the_report_and_by_evaluate(self, e2e):
        ws = e2e.ws
        recs = read_results(e2e)
        assert report.validate_records(recs, ws).valid == recs               # 基线：全部有效
        # 1. 结果记录里的副本哈希对不上
        tampered = [{**recs[0], "inputs": {**recs[0]["inputs"], "tree": "tampered"}}, *recs[1:]]
        v = report.validate_records(tampered, ws)
        assert len(v.valid) == len(recs) - 1 and v.stale[0][1].startswith("混淆副本已变")
        # 2. 上游的生成记录被改（重新 generate 过）：对应结果过期；删光 generations 则全部过期
        gen_path = ws.generations_file("mock-oracle")
        original = gen_path.read_text(encoding="utf-8")
        try:
            rows = store.read_jsonl(gen_path)
            rows[0]["text"] += "\n/* regenerated */"
            store.write_jsonl(gen_path, rows)
            v = report.validate_records(recs, ws)
            assert len(v.stale) == 1 and v.stale[0][1].startswith("回复已变")
            gen_path.unlink()
            v = report.validate_records(recs, ws)
            assert {r["model"] for r, _ in v.stale} == {"mock-oracle"} and len(v.stale) == len(recs) // 3
        finally:
            gen_path.write_text(original, encoding="utf-8")
        # 3. 提示词 / 功能规格变了（tasks.jsonl 里的消息哈希不同）
        tasks_path = ws.tasks_file
        original_tasks = tasks_path.read_text(encoding="utf-8")
        try:
            rows = store.read_jsonl(tasks_path)
            rows[0]["messages_hash"] = "changed"
            store.write_jsonl(tasks_path, rows)
            assert report.validate_records(recs, ws).stale
        finally:
            tasks_path.write_text(original_tasks, encoding="utf-8")
        # 4. 真的动了混淆副本：评测（以及 verify / 出题）直接拒绝，不会拿被改动过的副本出数据
        victim = ws.obf_root("mini", "L1") / "calc.c"
        data = victim.read_bytes()
        try:
            victim.write_bytes(data + b"\n/* tampered */\n")
            ctx = Context(ws=ws, sandbox=DockerSandbox(), jobs=2, seed=20261027, allow_mock=True)
            with pytest.raises(StageError, match="哈希不符"):
                pipeline.stage_evaluate(ctx, [corpus.load_spec_file(MINI / "spec.json")], ["mock-oracle"], ["L1"],
                                        tag="e2e")
        finally:
            victim.write_bytes(data)
        assert report.validate_records(read_results(e2e), ws).valid == recs

    def test_report_command_on_the_e2e_workspace(self, e2e, capsys):
        base = e2e.base
        assert main(["report", "--tag", "e2e", "--workspace", str(base), "--format", "md", "csv"]) == 0
        out = capsys.readouterr().out
        assert f"纳入统计 {len(read_results(e2e))} 条" in out and "输入哈希不符 0 条" in out

    def test_evaluate_command_is_idempotent_and_force_redoes_it(self, e2e, capsys):
        base = e2e.base
        # 工作区里的功能规格是 mock 模型写的，正式命令会把它判为作废；自测用 --unsafe-debug 放行
        argv = ["evaluate", "--model", "mock-oracle", "--tag", "e2e", "--corpus", str(MINI / "spec.json"),
                "--workspace", str(base), "--jobs", "6", "--levels", "L0", "--unsafe-debug"]
        before = store.read_jsonl(e2e.ws.results_file("e2e"))
        assert main(argv) == 0
        assert "跳过" in capsys.readouterr().out and store.read_jsonl(e2e.ws.results_file("e2e")) == before
        assert main([*argv, "--force"]) == 0
        out = capsys.readouterr().out
        n = sum(r["model"] == "mock-oracle" and r["level"] == "L0" for r in before)
        assert f"evaluated={n}" in out and f"pass={n}" in out
        after = store.read_jsonl(e2e.ws.results_file("e2e"))
        assert len(after) == len(before)                                 # 重做的是替换，不是追加成两份
        assert [r["task_id"] for r in after if r["model"] != "mock-oracle" or r["level"] != "L0"] == \
               [r["task_id"] for r in before if r["model"] != "mock-oracle" or r["level"] != "L0"]


# ----------------------------------------------------------------------
# 需要 docker：oracle gate、增量重算、逐条命令走一遍
# ----------------------------------------------------------------------
@needs_docker
class TestOracleGate:
    def test_gate_failures_void_only_the_affected_targets_and_are_counted(self, planned, mini_spec):
        ctx, ws, _ = planned
        ctx.sandbox = CountingSandbox(fail_syntax=("util.c",))
        results = pipeline.stage_verify(ctx, mini_spec, ["L0", "L1"])
        assert [r.ok for r in results] == [True, True] and all("2 个目标未通过 oracle gate" in r.message for r in results)
        for lv in ("L0", "L1"):
            gate = store.read_json(ws.obf_file("mini", lv))["oracle_gate"]
            assert (gate["checked"], gate["passed"], gate["copy_invalid"]) == (4, 2, False)
            assert sorted(f["target_id"].split(":")[-1] for f in gate["failed"]) == ["array_find", "parse_digits"]
            assert all(f["category"] == "compile_other" and "boom" in f["detail"] for f in gate["failed"])
            assert gate["inputs"]["targets"] and gate["input_hash"]
        describe_all(ctx, mini_spec)
        tasks, voids = pipeline.plan_tasks(ctx, mini_spec, ["L0", "L1"])
        assert voids == {"L0": {"oracle_failed": 2}, "L1": {"oracle_failed": 2}}
        assert {pt.original.name for pt in tasks} == {"calc_gcd", "calc_median3"} and len(tasks) == 4

    def test_gate_is_skipped_when_nothing_changed_and_redone_on_force(self, planned, mini_spec):
        ctx, ws, _ = planned
        ctx.sandbox = sandbox = CountingSandbox()
        first = pipeline.stage_verify(ctx, mini_spec, ["L0"])
        assert not first[0].skipped and first[0].detail == {"checked": 4, "passed": 4} and sandbox.runs >= 8
        n = sandbox.runs
        assert pipeline.stage_verify(ctx, mini_spec, ["L0"])[0].skipped and sandbox.runs == n
        ctx.force = True
        assert not pipeline.stage_verify(ctx, mini_spec, ["L0"])[0].skipped and sandbox.runs > n

    def test_invalid_copy_is_voided_without_running_anything(self, planned, mini_spec):
        ctx, ws, _ = planned
        ctx.sandbox = sandbox = CountingSandbox()
        data = store.read_json(ws.obf_file("mini", "L1"))
        data["ok"] = False
        store.write_json(ws.obf_file("mini", "L1"), data)
        (res,) = pipeline.stage_verify(ctx, mini_spec, ["L1"])
        assert not res.ok and "整级作废" in res.message and sandbox.runs == 0
        assert store.read_json(ws.obf_file("mini", "L1"))["oracle_gate"]["copy_invalid"] is True

    def test_missing_or_damaged_copy_is_refused(self, planned, mini_spec):
        ctx, ws, _ = planned
        ctx.sandbox = CountingSandbox()
        (ws.obf_root("mini", "L1") / "util.c").write_text("/* x */\n", encoding="utf-8")
        with pytest.raises(StageError, match="哈希不符"):
            pipeline.stage_verify(ctx, mini_spec, ["L1"])
        ws.obf_file("mini", "L0").unlink()
        with pytest.raises(StageError, match="obfuscate"):
            pipeline.stage_verify(ctx, mini_spec, ["L0"])


@needs_docker
class TestIncrementalRecompute:
    def test_changing_the_description_invalidates_generations_and_results_only(self, planned, mini_spec):
        ctx, ws, _ = planned
        ctx.sandbox = sandbox = CountingSandbox()
        models = ["mock-oracle", "mock-wrong"]
        describe_all(ctx, mini_spec)
        pipeline.stage_verify(ctx, mini_spec, ["L0", "L1"])
        pipeline.stage_generate(ctx, [mini_spec], models, ["L0", "L1"])
        first = pipeline.stage_evaluate(ctx, [mini_spec], models, ["L0", "L1"], tag="inc")
        assert [r.detail["evaluated"] for r in first] == [8, 8] and first[0].detail["pass"] == 8
        old = store.read_jsonl(ws.results_file("inc"))
        assert len(old) == 16 and {r["category"] for r in old if r["model"] == "mock-oracle"} == {"pass"}
        n = sandbox.runs
        assert all(r.skipped for r in pipeline.stage_evaluate(ctx, [mini_spec], models, ["L0", "L1"], tag="inc"))
        assert sandbox.runs == n                                           # 什么都没变：一次沙箱都不跑

        # 换一份功能规格 → 提示词变了：旧回复作废，旧评测结果跟着作废
        class Changed(Recorder):
            def chat(self, messages, **kw):
                reply = super().chat(messages, **kw)
                reply.text = MOCK_DESCRIPTION + " The function never allocates memory and never changes global state."
                return reply

        pipeline.stage_describe(ctx, mini_spec, Changed("desc2"), model_name="desc2")
        stale = pipeline.stage_evaluate(ctx, [mini_spec], models, ["L0", "L1"], tag="inc")
        assert all(r.skipped and r.detail["stale_generations"] == 8 for r in stale)
        assert sandbox.runs == n and store.read_jsonl(ws.results_file("inc")) == []        # 过期的结果被清掉，不进统计
        gen = pipeline.stage_generate(ctx, [mini_spec], models, ["L0", "L1"])
        assert [r.detail["calls"] for r in gen] == [8, 8]
        again = pipeline.stage_evaluate(ctx, [mini_spec], models, ["L0", "L1"], tag="inc")
        assert [r.detail["evaluated"] for r in again] == [8, 8]
        new = store.read_jsonl(ws.results_file("inc"))
        assert len(new) == 16 and {r["input_hash"] for r in new}.isdisjoint({r["input_hash"] for r in old})
        assert {r["category"] for r in new if r["model"] == "mock-oracle"} == {"pass"}

    def test_evaluating_a_subset_leaves_other_levels_untouched(self, planned, mini_spec):
        ctx, ws, _ = planned
        ctx.sandbox = CountingSandbox()
        describe_all(ctx, mini_spec)
        pipeline.stage_verify(ctx, mini_spec, ["L0", "L1"])
        pipeline.stage_generate(ctx, [mini_spec], ["mock-oracle"], ["L0", "L1"])
        pipeline.stage_evaluate(ctx, [mini_spec], ["mock-oracle"], ["L1"], tag="sub")
        assert {r["level"] for r in store.read_jsonl(ws.results_file("sub"))} == {"L1"}
        pipeline.stage_evaluate(ctx, [mini_spec], ["mock-oracle"], ["L0"], tag="sub")
        recs = store.read_jsonl(ws.results_file("sub"))
        assert [r["level"] for r in recs] == ["L1"] * 4 + ["L0"] * 4 and all(r["category"] == "pass" for r in recs)


@needs_docker
class TestCliPipeline:
    """用 main() 把每条子命令按流水线顺序走一遍（L0 + L1、每个语料最多 6 个目标，比 e2e 小得多）。"""

    @pytest.fixture(scope="class")
    def cli_ws(self):
        base = ROOT / "tmp" / "pytest-e2e" / f"cli-{uuid.uuid4().hex[:8]}"
        base.mkdir(parents=True)
        yield base
        shutil.rmtree(base, ignore_errors=True)

    def call(self, cli_ws, *args) -> int:
        spec = str(MINI / "spec.json")
        return main([args[0], "--corpus", spec, "--workspace", str(cli_ws), "--jobs", "6", "--seed", "11", *args[1:]])

    def test_pipeline_through_main(self, cli_ws, capsys):
        ws = Workspace.under(cli_ws)
        assert self.call(cli_ws, "prepare", "--src", str(MINI)) == 0
        assert "[prepare] mini: 完成" in capsys.readouterr().out
        assert self.call(cli_ws, "prepare", "--src", str(MINI)) == 0
        assert "跳过" in capsys.readouterr().out
        assert self.call(cli_ws, "targets", "--max-per-corpus", "6") == 0
        assert "selected=6" in capsys.readouterr().out
        assert self.call(cli_ws, "targets", "--max-per-corpus", "6") == 0 and "跳过" in capsys.readouterr().out
        assert len(store.read_jsonl(ws.targets_file("mini"))) == 6 and ws.coverage_file("mini").is_file()
        assert self.call(cli_ws, "obfuscate", "--level", "L0", "L1") == 0
        assert capsys.readouterr().out.count("verify=通过") == 2 and not ws.obf_file("mini", "L2").exists()
        assert self.call(cli_ws, "verify", "--level", "L0", "L1") == 0
        assert "checked=6, passed=6" in capsys.readouterr().out
        # 没有生成过功能规格就出题 → 每个目标都因 no_description 作废，但这不是错误；先 describe 再 generate
        # mock 模型只用于自测：不加 --unsafe-debug，describe 直接拒绝（以免固定的占位描述混进真实工作区）
        assert self.call(cli_ws, "describe", "--model", "mock-oracle") == 2
        assert "离线自测" in capsys.readouterr().err
        assert self.call(cli_ws, "describe", "--model", "mock-oracle", "--unsafe-debug") == 0
        assert "described=6" in capsys.readouterr().out
        assert self.call(cli_ws, "generate", "--model", "mock-oracle", "mock-wrong", "--levels", "L0", "L1",
                         "--unsafe-debug") == 0
        assert capsys.readouterr().out.count("calls=12") == 2
        assert self.call(cli_ws, "evaluate", "--model", "mock-oracle", "mock-wrong", "--levels", "L0", "L1",
                         "--tag", "cli", "--unsafe-debug") == 0
        out = capsys.readouterr().out
        assert "evaluated=12" in out and "pass=12" in out and "pass=0" in out
        assert self.call(cli_ws, "report", "--tag", "cli", "--format", "md", "csv") == 0
        assert "纳入统计 24 条" in capsys.readouterr().out
        recs = store.read_jsonl(ws.results_file("cli"))
        assert len(recs) == 24 and {r["level"] for r in recs} == {"L0", "L1"}
        assert all(r["category"] == "pass" for r in recs if r["model"] == "mock-oracle")
        assert not any(r["category"] == "pass" for r in recs if r["model"] == "mock-wrong")
        md = ws.summary_file("cli", ".md").read_text(encoding="utf-8")
        assert "| mock-oracle | L1 |" in md and "## 2. 相对 L0 的变化" in md
        assert ws.summary_file("cli", ".csv").is_file() and not ws.summary_file("cli", ".svg").exists()

    def test_coverage_command_alone_is_idempotent(self, cli_ws, capsys):
        assert self.call(cli_ws, "coverage") == 0
        assert "跳过" in capsys.readouterr().out
        assert self.call(cli_ws, "coverage", "--force") == 0
        assert "executed=" in capsys.readouterr().out

    def test_changing_the_seed_invalidates_everything_downstream(self, cli_ws, capsys):
        """种子变了：目标集重选，混淆副本重做，旧的 oracle gate 随副本作废，必须重新 verify 才能出题。"""
        assert self.call(cli_ws, "targets", "--max-per-corpus", "6", "--seed", "12") == 0
        assert "跳过" not in capsys.readouterr().out
        assert self.call(cli_ws, "obfuscate", "--level", "L0", "L1", "--seed", "12") == 0
        assert "跳过" not in capsys.readouterr().out
        assert self.call(cli_ws, "generate", "--model", "mock-oracle", "--levels", "L0", "L1", "--seed", "12") == 2
        assert "verify" in capsys.readouterr().err
        assert self.call(cli_ws, "verify", "--level", "L0", "L1", "--seed", "12") == 0
        assert "跳过" not in capsys.readouterr().out
