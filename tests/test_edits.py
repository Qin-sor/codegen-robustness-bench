"""bench.obfuscate._edits（L2 结构混淆与 L3 语义混淆共用的基础设施）的单元测试。

以前 structure.py 与 semantic.py 各抄了一套 Edit / 渲染 / 回滚 / 统计 / 递归上限，已经分叉；
现在只有这一份，两边的行为靠这里和各自的测试一起锁住。不需要 docker。
"""
from __future__ import annotations

import sys
from collections import Counter

from bench import clangcfg, corpus
from bench.obfuscate import _edits as E


class TestEditsAndRender:
    def test_str_parts_are_stored_as_latin1_bytes(self):
        e = E.Edit("x", 0, 3, ["ab", b"cd", (0, 1)])
        assert e.parts == [b"ab", b"cd", (0, 1)]
        assert E.Edit("x", 0, 1, ["é"]).parts == ["é".encode("latin-1")]        # 语义混淆按 latin-1 解码源码，往返不丢字节

    def test_nested_edits_render_from_the_outside_in(self):
        data = b"AAA BBB CCC\n"
        outer = E.Edit("x", 0, 11, [b"<", (4, 7), b">"])
        inner = E.Edit("y", 4, 7, [b"[", (5, 6), b"]"])
        assert E.render(data, [inner, outer]) == b"<[B]>\n"

    def test_laminar_check_drops_partially_overlapping_and_out_of_span_edits(self):
        a = E.Edit("a", 0, 5, [b"A"])
        crossing = E.Edit("b", 3, 8, [b"B"])                 # 与 a 部分重叠
        inside = E.Edit("c", 1, 2, [b"C"])                   # 落在 a 里，但 a 没有任何原文子区间
        outside = E.Edit("d", 20, 30, [(20, 30)])
        kept, dropped = E.check_laminar([a, crossing, inside, outside])
        assert kept == [a, outside] and dropped == 2
        assert E.render(b"0123456789", [a]) == b"A56789"

    def test_nested_edit_inside_a_declared_sub_range_is_kept(self):
        outer = E.Edit("o", 0, 10, [b"{", (2, 8), b"}"])
        inner = E.Edit("i", 3, 5, [b"##"])
        kept, dropped = E.check_laminar([inner, outer])
        assert kept == [outer, inner] and dropped == 0

    def test_replacement_that_loses_newlines_gets_them_back(self):
        data = b"a\nb\nc"
        new = E.render(data, [E.Edit("x", 0, 3, [b"X"])])
        assert new == b"X\n\nc" and new.count(b"\n") == data.count(b"\n")

    def test_crlf_files_get_crlf_padding(self):
        data = b"a\r\nb\r\nc"
        new = E.render(data, [E.Edit("x", 0, 4, [b"X"])])
        assert new == b"X\r\n" + data[4:] and new.count(b"\n") == data.count(b"\n")      # 补回去的换行也是 CRLF

    def test_render_without_edits_is_the_identity(self):
        assert E.render(b"int a;\n", []) == b"int a;\n"


class TestMatchClose:
    """括号配对：structure 的 ``_FileCtx`` 与缺陷修复任务的 mutate 共用这一份（以前各抄一份）。"""

    def test_pairs_nested_brackets_of_every_kind(self):
        t = ["f", "(", "a", "[", "1", "]", ",", "{", "x", "}", ")", ";"]
        assert E.match_close(t, 1) == 10 and E.match_close(t, 3) == 5 and E.match_close(t, 7) == 9

    def test_returns_none_when_the_brackets_do_not_pair(self):
        assert E.match_close(["(", "a"], 0) is None                       # 没有闭括号
        assert E.match_close(["(", "[", ")", "]"], 0) is None            # 交叉
        assert E.match_close(["(", "a", "}"], 0) is None                  # 种类不符
        assert E.match_close([")", "("], 0) is None                        # 起点就是闭括号

    def test_starts_from_any_index_and_ignores_what_is_before_it(self):
        t = [")", "(", "x", ")"]
        assert E.match_close(t, 1) == 3

    def test_the_structure_context_delegates_to_it(self):
        from bench.obfuscate import structure
        ctx = structure._FileCtx.__new__(structure._FileCtx)
        ctx.T = ["{", "(", ")", "}", "["]
        assert ctx.match_close(0) == 3 == E.match_close(ctx.T, 0) and ctx.match_close(4) is None


class TestMacroIntervals:
    def test_nested_and_overlapping_calls_merge_and_touching_ones_stay_apart(self):
        m = E.MacroIntervals([(10, 20), (12, 15), (18, 30), (40, 50), (50, 60)])
        assert m.intervals == [(10, 30), (40, 50), (50, 60)]
        assert E.MacroIntervals([]).intervals == [] and not E.MacroIntervals([]) and E.MacroIntervals([(0, 1)])

    def test_a_range_inside_a_macro_call_or_overlapping_it_conflicts(self):
        m = E.MacroIntervals([(10, 20)])
        assert m.conflicts(12, 15)                 # 宏实参
        assert m.conflicts(5, 12) and m.conflicts(18, 25)         # 部分重叠
        assert m.conflicts(10, 20)                 # 宏生成的整条语句：范围相等不放行
        assert not m.conflicts(0, 10) and not m.conflicts(20, 30)        # 紧挨着不算重叠
        assert not m.conflicts(5, 25) and not m.conflicts(0, 100)        # 整个宏调用落在范围里面：放行

    def test_allow_equal_lets_a_condition_that_is_exactly_one_macro_call_through(self):
        m = E.MacroIntervals([(10, 20)])
        assert m.conflicts(10, 20) and not m.conflicts(10, 20, allow_equal=True)
        assert m.conflicts(12, 15, allow_equal=True) and m.conflicts(5, 12, allow_equal=True)       # 其它冲突不受影响

    def test_the_structure_context_and_mutate_agree_on_every_range(self):
        """structure（require_not_macro）与 mutate（macro_conflict）判定宏冲突用的是同一个实现：穷举小区间核对。"""
        from bench import mutate
        from bench.obfuscate import structure
        raw = [(3, 9), (5, 7), (12, 14), (14, 18)]
        sctx = structure._FileCtx.__new__(structure._FileCtx)
        sctx.macro_iv = E.MacroIntervals(raw)
        mctx = mutate._FuncCtx.__new__(mutate._FuncCtx)
        mctx.macros = E.MacroIntervals(raw)
        for s in range(0, 22):
            for e in range(s + 1, 23):
                try:
                    sctx.require_not_macro(s, e)
                    structure_says = False
                except structure._Reject:
                    structure_says = True
                assert mctx.macro_conflict(s, e) is structure_says, (s, e)


class TestRollback:
    def edits(self, n=8):
        return [E.Edit("k", i * 10, i * 10 + 5, [b"ok"]) for i in range(n)]

    def validator(self, bad_starts, calls=None):
        def validate(eds):
            if calls is not None:
                calls.append(len(eds))
            hit = [e for e in eds if e.start in bad_starts]
            return (not hit), (["boom"] if hit else [])
        return validate

    def test_all_valid_is_one_validation(self):
        calls: list[int] = []
        rolled = Counter()
        eds = self.edits()
        assert E.rollback_invalid(eds, self.validator(set(), calls), rolled, "f.c") == eds
        assert calls == [8] and not rolled

    def test_only_the_bad_edits_are_rolled_back_and_counted_by_kind(self):
        rolled = Counter()
        eds = self.edits()
        good = E.rollback_invalid(eds, self.validator({20, 60}), rolled, "f.c")
        assert [e.start for e in good] == [0, 10, 30, 40, 50, 70]
        assert rolled["k"] == 2

    def test_everything_bad_gives_nothing(self):
        rolled = Counter()
        assert E.rollback_invalid(self.edits(3), lambda eds: (False, ["boom"]), rolled, "f.c") == []
        assert rolled["k"] == 3

    def test_empty_input(self):
        assert E.rollback_invalid([], lambda eds: (False, []), Counter(), "f.c") == []


class TestStats:
    KINDS = ("a", "b")

    def test_file_stats_defaults_and_dict(self):
        fs = E.FileStats(self.KINDS)
        assert set(fs.skipped) == set(self.KINDS) and fs.to_dict()["applied"] == {}
        fs.applied["a"] += 2
        fs.host = "m.c"
        assert fs.to_dict()["applied"] == {"a": 2} and fs.to_dict()["host"] == "m.c"

    def test_merge_accumulates_and_records_per_file_detail(self):
        total = E.empty_stats(self.KINDS, 7, 0.5, ["a"], functions=0, opaque_ratio=1.0)
        assert total["functions"] == 0 and total["opaque_ratio"] == 1.0 and total["kinds"] == ["a"]
        fs = E.FileStats(self.KINDS)
        fs.applied["a"] = 3
        fs.candidates["a"] = 5
        fs.skipped["b"]["why"] = 2
        fs.functions = 4
        fs.host = "m.c"
        E.merge_stats(total, "x.h", fs, changed=True)
        E.merge_stats(total, "y.c", E.FileStats(self.KINDS, skipped_file="解析失败"), changed=False)
        assert total["a"] == 3 and total["total"] == 3 and total["candidates"]["a"] == 5
        assert total["skipped"]["b"] == {"why": 2} and total["functions"] == 4
        assert total["files_total"] == 2 and total["files_changed"] == 1
        assert total["files_skipped"] == {"y.c": "解析失败"}
        assert total["files"]["x.h"] == {"applied": {"a": 3}, "host": "m.c"}

    def test_functions_key_is_optional(self):
        total = E.empty_stats(self.KINDS, 1, 1.0, self.KINDS)
        assert "functions" not in total
        E.merge_stats(total, "x.c", E.FileStats(self.KINDS), changed=False)       # 没有 functions 键时不报错


def test_recursion_guard_raises_and_restores_the_limit():
    old = sys.getrecursionlimit()
    with E.RecursionGuard(old + 5000):
        assert sys.getrecursionlimit() == old + 5000
    assert sys.getrecursionlimit() == old


class TestFiles:
    def spec(self, **kw):
        return corpus.CorpusSpec(name="t", project_files=["*.c", "*.h", "v/*.c", "notes.txt"], target_files=["a.c"], **kw)

    def make(self, root):
        for rel in ("a.c", "b.c", "c.h", "v/lib.c", "notes.txt"):
            f = root / rel
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text('#include "c.h"\n' if rel.endswith(".c") else "x\n")

    def test_plan_files_filters_vendored_suffixes_and_wanted(self, scratch):
        self.make(scratch)
        spec = self.spec(vendored=["v"])
        all_files, todo = E.plan_files(spec, scratch)
        assert [p.name for p in all_files] == ["a.c", "b.c", "c.h", "notes.txt", "lib.c"] or len(all_files) == 5
        assert [rel for rel, _ in todo] == ["a.c", "b.c", "c.h"]
        assert [rel for rel, _ in E.plan_files(spec, scratch, include_vendored=True)[1]] == ["a.c", "b.c", "c.h", "v/lib.c"]
        assert [rel for rel, _ in E.plan_files(spec, scratch, files=["b.c", "v/lib.c"])[1]] == ["b.c"]

    def test_host_map_lists_the_c_files_that_include_a_header(self, scratch):
        self.make(scratch)
        hosts = E.host_map([scratch / "a.c", scratch / "b.c"])
        assert hosts == {"c.h": [scratch / "a.c", scratch / "b.c"]}

    def test_safe_to_write_refuses_symlinks_and_paths_outside_the_root(self, scratch):
        root = scratch / "root"
        root.mkdir()
        (root / "real.c").write_text("x")
        (scratch / "out.c").write_text("y")
        (root / "link.c").symlink_to(scratch / "out.c")
        assert E.safe_to_write(root.resolve(), (root / "real.c").resolve())
        assert not E.safe_to_write(root.resolve(), root / "link.c")
        assert not E.safe_to_write(root.resolve(), (scratch / "out.c").resolve())

    def test_run_files_writes_back_merges_and_survives_a_crashing_transform(self, scratch):
        root = scratch / "r"
        root.mkdir()
        (root / "good.c").write_text("old")
        (root / "boom.c").write_text("old")
        (scratch / "out.c").write_text("old")
        (root / "link.c").symlink_to(scratch / "out.c")
        stats = E.empty_stats(("k",), 1, 1.0, ["k"])

        def transform(path, data, rel):
            if rel == "boom.c":
                raise RuntimeError("坏了")
            fs = E.FileStats(("k",))
            fs.applied["k"] += 1
            return b"new", fs

        todo = [(rel, root / rel) for rel in ("good.c", "boom.c", "link.c")]
        E.run_files(todo, root.resolve(), ("k",), stats, transform, "测试")
        assert (root / "good.c").read_text() == "new"
        assert (root / "boom.c").read_text() == "old"                  # 单个文件出意外：保持原样，记下来，不拖垮整体
        assert (scratch / "out.c").read_text() == "old"                # 符号链接不写穿
        assert "处理出错" in stats["files_skipped"]["boom.c"] and "符号链接" in stats["files_skipped"]["link.c"]
        assert stats["files_total"] == 3 and stats["files_changed"] == 1 and stats["k"] == 1


class TestParserHosting:
    def test_header_that_does_not_stand_alone_is_parsed_through_a_host(self, scratch):
        (scratch / "t.h").write_text("static int f(va_list ap) { return 0; }\n")
        (scratch / "m.c").write_text('#include <stdarg.h>\n#include "t.h"\nint main(void) { return 0; }\n')
        parser = E.Parser(["-x", "c", "-w", "-ferror-limit=0", *clangcfg.builtin_include_args()])
        data = (scratch / "t.h").read_bytes()
        unit, tu, errs = parser.open(scratch / "t.h", data, [scratch / "m.c"])
        assert errs == [] and unit.host == scratch / "m.c" and E.file_name(tu, scratch / "t.h")
        # 没有宿主可借：错误如实返回
        unit2, _, errs2 = parser.open(scratch / "t.h", data, [])
        assert errs2 and unit2.host is None
        ok, errs3 = parser.valid(unit, data + b"int broken(;\n")
        assert not ok and errs3
