"""bench.targets 与 bench.coverage 的测试。

不需要 docker 的部分：过滤器默认值、桩的构造（按返回类型分类）、抽样的确定性；
需要 docker 的部分：mini 夹具上的覆盖率执行次数、stub-kill 判定、select_targets 的整体行为。

mini 里每个被测函数都有「零值桩会被抓住」的断言，所以夹具本身没有测试覆盖不住的函数。
这里复制一份 mini，在副本里追加两个函数：一个只被调用、不校验返回值（桩杀不死它），
一个从未被调用（执行次数为 0），用来验证 select_targets 的排除逻辑。
所有编译和运行只走 docker 沙箱；临时文件放在项目 tmp/ 下，测试结束即清理。
"""
from __future__ import annotations

import dataclasses
import shutil
import uuid
from pathlib import Path

import pytest

from bench import corpus
from bench.coverage import CoverageError, function_exec_counts, parse_gcov_json
from bench.corpus import CorpusSpec
from bench.extract import FunctionTarget, extract_file, extract_functions
from bench.runner import splice
from bench.sandbox import DockerSandbox
from bench.targets import (
    ReturnInfo,
    Selection,
    TargetFilter,
    TargetsError,
    build_stub,
    return_infos,
    select,
    select_targets,
    stub_body,
    stub_kill,
    stub_verdict,
)
from bench.targets import _sample_key

from tests.helpers import MINI, ROOT, needs_docker

EXTRA_C = """

/* 绝对值求和：测试里只调用、不校验返回值，所以零值桩杀不死它 */
int array_sum_abs(const int *a, size_t n)
{
    int total = 0;
    size_t i;

    for (i = 0; i < n; i++) {
        total += a[i] < 0 ? -a[i] : a[i];
    }
    return total;
}

/* 从未被调用：执行次数为 0 */
int never_called(int x)
{
    int y = x * 2;

    y += 3;
    return y;
}
"""


def make_variant(dest: Path, break_test: bool = False) -> Path:
    """复制 mini，追加 array_sum_abs（只调用不校验）和 never_called（从不调用）。"""
    corpus.materialize(MINI, dest)
    util = dest / "util.c"
    util.write_text(util.read_text(encoding="utf-8") + EXTRA_C, encoding="utf-8")
    header = dest / "mini.h"
    h = header.read_text(encoding="utf-8")
    assert "#endif /* MINI_H */" in h
    header.write_text(h.replace("#endif /* MINI_H */",
                                "int array_sum_abs(const int *a, size_t n);\n\n#endif /* MINI_H */"), encoding="utf-8")
    tm = dest / "test_main.c"
    t = tm.read_text(encoding="utf-8")
    anchor = "    CHECK(sum_array(a, 0) == 0);\n"
    assert anchor in t
    t = t.replace(anchor, anchor + "    (void)array_sum_abs(b, 4);   /* 只调用，不校验 */\n")
    if break_test:
        t = t.replace("    CHECK(calc_gcd(12, 18) == 6);\n", "    CHECK(calc_gcd(12, 18) == 7);\n")
    tm.write_text(t, encoding="utf-8")
    return dest


@pytest.fixture(scope="module")
def tmp_base():
    base = ROOT / "tmp" / "pytest" / uuid.uuid4().hex[:10]
    base.mkdir(parents=True)
    yield base
    shutil.rmtree(base, ignore_errors=True)


@pytest.fixture(scope="module")
def mini_targets(mini_spec) -> dict[str, FunctionTarget]:
    return {t.name: t for t in extract_functions(mini_spec, MINI)}


@pytest.fixture(scope="module")
def sandbox() -> DockerSandbox:
    return DockerSandbox()


@pytest.fixture(scope="module")
def variant(tmp_base) -> Path:
    return make_variant(tmp_base / "variant")


@pytest.fixture(scope="module")
def variant_targets(mini_spec, variant) -> dict[str, FunctionTarget]:
    return {t.name: t for t in extract_functions(mini_spec, variant)}


class NoSandbox:
    """任何调用都算失败：用来证明某条路径不碰沙箱。"""

    def run(self, *args, **kwargs):
        raise AssertionError("不应该调用沙箱")


# ----------------------------------------------------------------------
# 不需要 docker
# ----------------------------------------------------------------------
class TestFilterAndSampling:
    def test_filter_defaults_follow_the_design_doc(self):
        f = TargetFilter()
        assert (f.min_statements, f.min_lines, f.max_lines, f.max_per_corpus) == (3, 4, 120, 50)
        assert f.require_exec is True and f.require_stub_kill is True

    def test_sample_key_depends_only_on_seed_and_id(self):
        assert _sample_key(1, "mini:calc.c:a") == _sample_key(1, "mini:calc.c:a")
        assert _sample_key(1, "mini:calc.c:a") != _sample_key(2, "mini:calc.c:a")
        assert _sample_key(1, "mini:calc.c:a") != _sample_key(1, "mini:calc.c:b")

    def test_selection_without_exec_and_stub_never_touches_the_sandbox(self, mini_spec):
        flt = TargetFilter(max_per_corpus=5, require_exec=False, require_stub_kill=False)
        sel = select(mini_spec, MINI, NoSandbox(), flt, seed=3)
        assert len(sel.targets) == 5
        assert all(t.n_statements >= 3 and t.n_lines >= 4 for t in sel.targets)
        assert sel.exec_counts == {} and sel.stub_verdicts == {}
        assert sel.excluded["not_sampled"] == sel.n_candidates - 5
        # 每个提取出来的函数要么入选，要么有且仅有一个排除原因
        assert len(sel.targets) + sum(sel.excluded.values()) == sel.n_extracted
        again = select(mini_spec, MINI, NoSandbox(), flt, seed=3)
        assert [t.target_id for t in again.targets] == [t.target_id for t in sel.targets]
        other = select(mini_spec, MINI, NoSandbox(), flt, seed=4)
        assert [t.target_id for t in other.targets] != [t.target_id for t in sel.targets]

    def test_static_filters_report_their_reasons(self, mini_spec):
        flt = TargetFilter(min_statements=5, min_lines=10, max_lines=14, max_per_corpus=1000,
                           require_exec=False, require_stub_kill=False)
        sel = select(mini_spec, MINI, NoSandbox(), flt, seed=0)
        assert sel.excluded.get("few_statements", 0) > 0
        assert sel.excluded.get("too_long", 0) > 0
        assert all(5 <= t.n_statements and 10 <= t.n_lines <= 14 for t in sel.targets)
        assert len(sel.targets) + sum(sel.excluded.values()) == sel.n_extracted

    def test_select_targets_returns_the_target_list_and_fills_stats(self, mini_spec):
        flt = TargetFilter(max_per_corpus=3, require_exec=False, require_stub_kill=False)
        stats: dict = {}
        got = select_targets(mini_spec, MINI, NoSandbox(), flt, 7, stats=stats)
        assert [t.target_id for t in got] == [t.target_id for t in select(mini_spec, MINI, NoSandbox(), flt, 7).targets]
        assert stats["selected"] == 3 and stats["corpus"] == "mini" and stats["seed"] == 7
        assert stats["excluded"]["not_sampled"] > 0

    def test_selected_targets_are_sorted_by_position(self, mini_spec):
        flt = TargetFilter(max_per_corpus=10, require_exec=False, require_stub_kill=False)
        got = select_targets(mini_spec, MINI, NoSandbox(), flt, 1)
        assert [(t.file, t.start) for t in got] == sorted((t.file, t.start) for t in got)

    def test_records_have_schema_fields(self, mini_spec):
        flt = TargetFilter(max_per_corpus=2, require_exec=False, require_stub_kill=False)
        recs = select(mini_spec, MINI, NoSandbox(), flt, 1).records()
        assert len(recs) == 2
        for r in recs:
            assert r["schema_version"] == 1 and r["exec_count"] is None and r["stub_killed"] is None
            assert {"target_id", "corpus", "file", "name", "signature", "start", "end", "body_start", "body_end",
                    "n_statements", "n_lines", "callees", "deps", "doc_comment"} <= set(r)
            FunctionTarget.from_dict(r)         # 能还原


class TestStubConstruction:
    def test_stub_body_by_kind(self):
        assert stub_body(ReturnInfo("void", "void")) == "{ }"
        assert stub_body(ReturnInfo("scalar", "int")) == "{ return 0; }"
        assert stub_body(ReturnInfo("pointer", "char *")) == "{ return 0; }"
        assert stub_body(ReturnInfo("record", "struct rect")) == "{ static struct rect _bench_zero; return _bench_zero; }"
        assert stub_body(ReturnInfo("unsupported", "struct (anonymous)")) is None

    def test_return_kinds_on_mini(self, mini_spec, mini_targets):
        wanted = {
            "calc_gcd": ("scalar", "int"), "calc_div_round": ("scalar", "int"), "str_dup_upper": ("pointer", None),
            "op_name": ("pointer", None), "array_reverse": ("void", None), "point_make": ("record", "point_t"),
            "point_add": ("record", "point_t"), "rect_normalize": ("void", None), "calc_abs_diff": ("scalar", "int"),
        }
        infos = return_infos(mini_spec, MINI, [mini_targets[n] for n in wanted])
        for name, (kind, spelling) in wanted.items():
            got = infos[mini_targets[name].target_id]
            assert got.kind == kind, name
            if spelling:
                assert got.spelling == spelling, name

    def test_build_stub_keeps_the_original_header_verbatim(self, mini_spec, mini_targets):
        infos = return_infos(mini_spec, MINI, list(mini_targets.values()))
        t = mini_targets["calc_abs_diff"]          # 返回类型是宏 MINI_RET(int)
        text = build_stub(MINI, t, infos[t.target_id])
        assert text == "MINI_RET(int) calc_abs_diff(int a, int b)\n{ return 0; }"
        t = mini_targets["point_add"]
        text = build_stub(MINI, t, infos[t.target_id])
        assert text == "point_t point_add(point_t a, point_t b)\n{ static point_t _bench_zero; return _bench_zero; }"
        t = mini_targets["rect_contains"]          # 参数分多行写
        text = build_stub(MINI, t, infos[t.target_id])
        assert text.startswith("int rect_contains(const struct rect *r,\n                  int x,\n") \
            and text.endswith("{ return 0; }")

    def test_unsupported_return_type_has_no_stub(self, mini_targets):
        assert build_stub(MINI, mini_targets["calc_gcd"], ReturnInfo("unsupported", "")) is None

    SPECIAL_C = """#include <stddef.h>
struct rect { int w, h; };
typedef struct { int a; } anon_t;
enum color { RED, GREEN };
typedef int (*cb_t)(int);

const struct rect ret_const_rect(int x) { struct rect r; r.w = x; r.h = x; return r; }
anon_t ret_anon(int x) { anon_t a; a.a = x; return a; }
enum color ret_enum(int x) { if (x) { return RED; } return GREEN; }
float ret_float(int x) { float f = (float)x; f += 1.0f; return f; }
_Bool ret_bool(int x) { int y = x; y++; return y != 0; }
cb_t ret_cb(int x) { (void)x; return 0; }
unsigned long long ret_ull(int x) { unsigned long long v = (unsigned long long)x; v++; return v; }
static void ret_void(int *p) { *p = 1; (*p)++; }
char *ret_ptr(char *s) { char *t = s; t++; return s; }
long double ret_ld(int x) { long double v = x; v += 1; return v; }
"""
    EXPECTED = {
        "ret_const_rect": ("record", "struct rect"), "ret_anon": ("record", "anon_t"),
        "ret_enum": ("scalar", None), "ret_float": ("scalar", None), "ret_bool": ("scalar", None),
        "ret_cb": ("pointer", None), "ret_ull": ("scalar", None), "ret_void": ("void", None),
        "ret_ptr": ("pointer", None), "ret_ld": ("scalar", None),
    }

    @pytest.fixture()
    def special(self, tmp_base):
        root = tmp_base / f"special-{uuid.uuid4().hex[:6]}"
        root.mkdir()
        (root / "t.c").write_text(self.SPECIAL_C)
        spec = CorpusSpec(name="special", project_files=["t.c"], target_files=["t.c"], cflags=["-std=c99"])
        return spec, root, {t.name: t for t in extract_file(spec, root, "t.c")}

    def test_special_return_types(self, special):
        spec, root, ts = special
        assert set(ts) == set(self.EXPECTED)
        infos = return_infos(spec, root, list(ts.values()))
        for name, (kind, spelling) in self.EXPECTED.items():
            got = infos[ts[name].target_id]
            assert got.kind == kind, (name, got)
            if spelling:
                assert got.spelling == spelling, name

    @needs_docker
    def test_every_stub_compiles(self, special, sandbox, tmp_base):
        spec, root, ts = special
        infos = return_infos(spec, root, list(ts.values()))
        for name, t in ts.items():
            work = tmp_base / f"stubcc-{name}"
            splice(root, t, build_stub(root, t, infos[t.target_id]), work)
            res = sandbox.run(work, "gcc -std=c99 -Wall -Wextra -fsyntax-only t.c", timeout_s=60)
            shutil.rmtree(work, ignore_errors=True)
            assert res.rc == 0, (name, res.stderr[-400:])
            assert "error" not in res.stderr, name


class TestGcovParsing:
    DOC = {
        "format_version": "1", "current_working_directory": "/work", "data_file": "x.gcda",
        "files": [
            {"file": "tests/../cJSON.c", "functions": [
                {"name": "a", "execution_count": 3}, {"name": "b.constprop.0", "execution_count": 2},
                {"name": "b", "execution_count": 1}, {"name": "zero", "execution_count": 0}]},
            {"file": "/work/util.c", "functions": [{"name": "c", "execution_count": 7}]},
            {"file": "/usr/include/x86_64-linux-gnu/bits/stdio.h", "functions": [{"name": "d", "execution_count": 9}]},
        ],
    }

    def test_paths_are_normalized_against_the_recorded_cwd(self):
        import json
        got = list(parse_gcov_json(json.dumps(self.DOC).encode()))
        assert ("cJSON.c", "a", 3) in got and ("util.c", "c", 7) in got
        assert ("cJSON.c", "zero", 0) in got
        assert ("/usr/include/x86_64-linux-gnu/bits/stdio.h", "d", 9) in got

    def test_clone_suffixes_are_folded_into_the_base_name(self):
        import json
        names = [n for f, n, c in parse_gcov_json(json.dumps(self.DOC)) if f == "cJSON.c"]
        assert names == ["a", "b", "b", "zero"]

    def test_gzip_input(self):
        import gzip, json
        blob = gzip.compress(json.dumps(self.DOC).encode())
        assert len(list(parse_gcov_json(blob))) == 6


# ----------------------------------------------------------------------
# 覆盖率（docker）
# ----------------------------------------------------------------------
@needs_docker
class TestExecCounts:
    def test_mini_counts_are_positive_and_keyed_by_file_and_name(self, mini_spec, sandbox):
        counts = function_exec_counts(mini_spec, MINI, sandbox)
        assert counts["calc.c:calc_gcd"] > 0 and counts["util.c:sum_array"] > 0
        assert counts["calc.c:point_make"] > 0 and counts["util.c:str_dup_upper"] > 0
        assert set(f.split(":")[0] for f in counts) == {"calc.c", "util.c"}      # 只含目标文件
        assert not any(k.startswith("test_main.c:") for k in counts)
        # 计的是函数被执行的次数，不是「是否执行过」：测试里至少调了 4 次 calc_gcd，递归还会再加
        assert counts["calc.c:calc_gcd"] >= 4 and counts["calc.c:calc_gcd"] > counts["calc.c:mini_version"]
        assert all(isinstance(v, int) and v >= 0 for v in counts.values())

    def test_never_called_function_has_count_zero(self, mini_spec, variant, sandbox):
        counts = function_exec_counts(mini_spec, variant, sandbox)
        assert counts["util.c:never_called"] == 0
        assert counts["util.c:array_sum_abs"] >= 1

    def test_does_not_modify_the_source_tree(self, mini_spec, sandbox):
        before = corpus.tree_digest(MINI)
        function_exec_counts(mini_spec, MINI, sandbox)
        assert corpus.tree_digest(MINI) == before
        assert not list(MINI.glob("*.gc*"))

    def test_broken_build_raises_coverage_error(self, mini_spec, sandbox, tmp_base):
        broken = tmp_base / "broken"
        corpus.materialize(MINI, broken)
        (broken / "calc.c").write_text("this is not C\n")
        with pytest.raises(CoverageError):
            function_exec_counts(mini_spec, broken, sandbox)

    def test_failing_tests_under_coverage_warn_but_still_report(self, mini_spec, sandbox, tmp_base, caplog):
        # 项目测试在覆盖率构建下没通过，但 .gcda 已经生成：统计照常返回，同时打警告
        broken = make_variant(tmp_base / "cov_broken_test", break_test=True)
        with caplog.at_level("WARNING", logger="bench.coverage"):
            counts = function_exec_counts(mini_spec, broken, sandbox)
        assert counts["calc.c:calc_gcd"] > 0
        assert any("没有通过" in r.getMessage() for r in caplog.records)

    def test_coverage_uses_the_spec_coverage_env(self, mini_spec, sandbox):
        # coverage.env 去掉后没有 --coverage，构建正常但不产生 .gcda
        spec = dataclasses.replace(mini_spec, coverage={})
        with pytest.raises(CoverageError):
            function_exec_counts(spec, MINI, sandbox)


# ----------------------------------------------------------------------
# stub-kill 与选择（docker）
# ----------------------------------------------------------------------
@needs_docker
class TestStubKill:
    @pytest.mark.parametrize("name", ["calc_gcd", "point_make", "str_dup_upper", "array_reverse", "op_name",
                                      "rect_normalize", "calc_div_round"])
    def test_well_tested_functions_are_killed(self, mini_spec, mini_targets, sandbox, name):
        t = mini_targets[name]
        v = stub_verdict(mini_spec, MINI, t, sandbox)
        assert v.verdict == "killed" and v.stage == "test", v
        assert stub_kill(mini_spec, MINI, t, sandbox) is True

    def test_untested_function_survives(self, mini_spec, variant, variant_targets, sandbox):
        t = variant_targets["array_sum_abs"]
        v = stub_verdict(mini_spec, variant, t, sandbox)
        assert v.verdict == "survived" and v.stage == "pass"
        assert stub_kill(mini_spec, variant, t, sandbox) is False

    def test_stub_that_does_not_build_is_not_a_kill(self, mini_spec, mini_targets, sandbox):
        t = mini_targets["point_make"]
        v = stub_verdict(mini_spec, MINI, t, sandbox, info=ReturnInfo("record", "no_such_type_t"))
        assert v.verdict == "build_error" and v.stage == "build"
        assert "no_such_type_t" in v.detail
        assert v.killed is False

    def test_unsupported_return_type_is_reported_without_running_anything(self, mini_spec, mini_targets):
        v = stub_verdict(mini_spec, MINI, mini_targets["calc_gcd"], NoSandbox(), info=ReturnInfo("unsupported", "?"))
        assert v.verdict == "unsupported" and v.stage is None

    def test_source_tree_is_untouched(self, mini_spec, mini_targets, sandbox):
        before = corpus.tree_digest(MINI)
        stub_kill(mini_spec, MINI, mini_targets["calc_gcd"], sandbox)
        assert corpus.tree_digest(MINI) == before


@needs_docker
class TestSelectTargets:
    @pytest.fixture(scope="class")
    def full(self, mini_spec, variant, sandbox) -> Selection:
        return select(mini_spec, variant, sandbox, TargetFilter(max_per_corpus=200), seed=1, jobs=4)

    def test_weak_and_unexecuted_functions_are_excluded(self, full):
        names = {t.name for t in full.targets}
        assert "array_sum_abs" not in names and "never_called" not in names
        assert full.excluded["stub_survived"] == 1
        assert full.excluded_ids["stub_survived"] == ["mini:util.c:array_sum_abs"]
        assert full.excluded["not_executed"] == 1
        assert full.excluded_ids["not_executed"] == ["mini:util.c:never_called"]
        assert full.excluded["few_statements"] >= 1

    def test_well_tested_functions_are_selected(self, full):
        names = {t.name for t in full.targets}
        for name in ("calc_gcd", "calc_median3", "point_make", "point_add", "str_dup_upper", "array_reverse",
                     "rect_normalize", "stack_push", "parse_digits", "sum_array"):
            assert name in names, name
        # mini 里每个候选函数都有「零值桩会被抓住」的断言（见 test_main.c），只有追加的 array_sum_abs 杀不死
        assert len(full.targets) == full.n_candidates - 1

    def test_every_extracted_function_is_accounted_for(self, full):
        assert len(full.targets) + sum(full.excluded.values()) == full.n_extracted
        assert full.n_candidates == len(full.targets) + full.excluded["stub_survived"]

    def test_records_carry_exec_count_and_stub_killed(self, full):
        recs = full.records()
        assert len(recs) == len(full.targets)
        for r in recs:
            assert r["exec_count"] > 0 and r["stub_killed"] is True and r["schema_version"] == 1
        assert full.stub_verdicts["mini:util.c:array_sum_abs"].verdict == "survived"

    def test_summary(self, full):
        s = full.summary()
        assert s["selected"] == len(full.targets) and s["excluded"]["stub_survived"] == 1
        assert s["filter"]["max_per_corpus"] == 200

    def test_max_per_corpus_and_determinism_across_jobs_and_seeds(self, mini_spec, sandbox):
        flt = TargetFilter(max_per_corpus=4, require_exec=False)
        a = select(mini_spec, MINI, sandbox, flt, seed=11, jobs=1)
        b = select(mini_spec, MINI, sandbox, flt, seed=11, jobs=3)
        c = select(mini_spec, MINI, sandbox, flt, seed=12, jobs=3)
        ids = lambda s: [t.target_id for t in s.targets]  # noqa: E731
        assert len(a.targets) == 4 and ids(a) == ids(b)
        assert a.excluded == b.excluded                    # 计数也与并发度无关
        assert ids(c) != ids(a)
        assert all(v.killed for v in a.stub_verdicts.values())
        assert a.excluded["not_sampled"] == a.n_candidates - 4

    def test_scan_all_gives_complete_counts_without_changing_the_selection(self, mini_spec, variant, sandbox):
        flt = TargetFilter(max_per_corpus=4)
        lazy = select(mini_spec, variant, sandbox, flt, seed=5, jobs=4)
        full = select(mini_spec, variant, sandbox, flt, seed=5, jobs=4, scan_all=True)
        assert [t.target_id for t in full.targets] == [t.target_id for t in lazy.targets]
        assert len(full.targets) == 4
        # 完整扫描：每个候选都有桩判定，array_sum_abs 这个存活的一定被发现；名额外杀得死的记 not_sampled
        assert len(full.stub_verdicts) == full.n_candidates
        assert full.excluded["stub_survived"] == 1
        assert full.excluded["not_sampled"] == full.n_candidates - 1 - 4
        # 默认模式只检查到凑够名额为止
        assert len(lazy.stub_verdicts) < len(full.stub_verdicts)
        assert lazy.excluded["not_sampled"] == full.n_candidates - len(lazy.stub_verdicts)
        assert len(full.targets) + sum(full.excluded.values()) == full.n_extracted

    def test_spec_excluded_targets_never_enter_the_selection(self, mini_spec, variant, sandbox):
        # 编进目标文件的测试驱动（sds 的 sdsTest）：换成桩汇总行就没了，会被误判成「杀死」，必须靠 spec 明确排除
        spec = dataclasses.replace(mini_spec, exclude_targets=["calc_gcd", "util.c:array_find"])
        sel = select(spec, variant, sandbox, TargetFilter(max_per_corpus=500, require_exec=False), seed=1)
        names = {t.name for t in sel.targets}
        assert "calc_gcd" not in names and "array_find" not in names and "calc_median3" in names
        assert sel.excluded["spec_excluded"] == 2

    def test_stub_kill_can_be_switched_off(self, mini_spec, variant, sandbox):
        flt = TargetFilter(max_per_corpus=500, require_stub_kill=False)
        sel = select(mini_spec, variant, sandbox, flt, seed=1)
        names = {t.name for t in sel.targets}
        assert "array_sum_abs" in names and "never_called" not in names       # 执行次数过滤仍然生效
        assert sel.stub_verdicts == {}

    def test_failing_baseline_aborts_the_selection(self, mini_spec, tmp_base, sandbox):
        broken = make_variant(tmp_base / "broken_baseline", break_test=True)
        with pytest.raises(TargetsError):
            select(mini_spec, broken, sandbox, TargetFilter(max_per_corpus=3, require_exec=False), seed=1)
