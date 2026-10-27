"""bench.attribute 与 bench.similarity 的测试：不需要 docker，全部是纯 Python 用例。

* attribute：每个类别至少一个用例，外加类别之间的优先级、诊断字典两种写法（gcc 原始 JSON / runner 压平后）、
  detail 内容；
* similarity：同一函数改名后相似度 > 0.9，不同函数 < 0.5，注释与空白不影响结果。

被测函数用 tests/fixtures/mini 里的真实源码，不手写玩具代码。
"""
from __future__ import annotations

import re
import signal as sig

import pytest

from bench.attribute import CATEGORIES, CRASH_SIGNALS, attribute, compile_category
from bench.runner import EvalResult
from bench.similarity import C_KEYWORDS, normalized_similarity, normalized_tokens

from bench.postprocess import Candidate
from tests.helpers import MINI, mini_spec

CODE = "int f(int a) { return a; }"


def ev(stage: str = "test", rc: int = 1, timed_out: bool = False, signal: int | None = None,
       diagnostics: list[dict] | None = None, tail: str = "", duration: float = 0.5) -> EvalResult:
    return EvalResult(task_id="t:f.c:f", model="mock", sample_idx=0, stage=stage, rc=rc, timed_out=timed_out,
                      signal=signal, diagnostics=diagnostics or [], test_stdout_tail=tail, duration_s=duration)


def cand(code: str | None = CODE, issues: list[str] | None = None, violations: list[str] | None = None) -> Candidate:
    return Candidate(code=code, issues=list(issues or []), violations=list(violations or []))


def err(message: str, kind: str = "error", **extra) -> dict:
    """runner 压平后的诊断字典。"""
    return {"kind": kind, "message": message, "stage": "syntax", **extra}


# ----------------------------------------------------------------------
# attribute：每个类别
# ----------------------------------------------------------------------
class TestEveryCategory:
    def test_pass(self):
        assert attribute(ev("pass", rc=0), cand()) == ("pass", "")

    def test_extract_fail_when_no_code(self):
        cat, detail = attribute(ev("syntax", rc=-1), cand(None, ["no_code_block"]))
        assert cat == "extract_fail"
        assert "no_code_block" in detail

    def test_extract_fail_needs_no_eval_result(self):
        # 抽不出代码的样本不会进沙箱，ev 可以是 None
        assert attribute(None, cand(None, ["no_code_block"]))[0] == "extract_fail"

    def test_extract_fail_for_blank_code(self):
        assert attribute(ev("pass", rc=0), cand("  \n "))[0] == "extract_fail"

    @pytest.mark.parametrize("message", [
        "expected ';' before '}' token",
        "expected expression before ')' token",
        "expected declaration or statement at end of input",
        "stray '\\342' in program",
        "syntax error before 'x'",
        "missing terminating \" character",
        "unterminated comment",
    ])
    def test_compile_syntax(self, message):
        assert attribute(ev("syntax", rc=1, diagnostics=[err(message)]), cand())[0] == "compile_syntax"

    @pytest.mark.parametrize("message", [
        "'total' undeclared (first use in this function)",
        "implicit declaration of function 'helper'",
        "unknown type name 'node_t'",
        "'struct rect' has no member named 'z'",
    ])
    def test_compile_undeclared(self, message):
        assert attribute(ev("syntax", rc=1, diagnostics=[err(message)]), cand())[0] == "compile_undeclared"

    @pytest.mark.parametrize("message", [
        "incompatible types when returning type 'point_t' but 'int' was expected",
        "too few arguments to function 'calc_gcd'",
        "too many arguments to function 'calc_gcd'",
        "passing argument 1 of 'f' makes pointer from integer without a cast",
        "invalid operands to binary + (have 'int *' and 'int *')",
    ])
    def test_compile_type(self, message):
        assert attribute(ev("syntax", rc=1, diagnostics=[err(message)]), cand())[0] == "compile_type"

    @pytest.mark.parametrize("message", [
        "label 'out' used but not defined",
        "redefinition of 'g_calls'",
        "conflicting types for 'calc_gcd'",
        "lvalue required as left operand of assignment",
    ])
    def test_compile_other(self, message):
        assert attribute(ev("syntax", rc=1, diagnostics=[err(message)]), cand())[0] == "compile_other"

    def test_link_error(self):
        tail = ("/usr/bin/ld: /tmp/ccAbC.o: in function `calc_gcd':\n"
                "calc.c:(.text+0x21): undefined reference to `bench_missing_helper'\n"
                "collect2: error: ld returned 1 exit status\n")
        cat, detail = attribute(ev("build", rc=2, tail=tail), cand())
        assert cat == "link_error"
        assert "undefined reference to `bench_missing_helper'" in detail

    @pytest.mark.parametrize("number", sorted(CRASH_SIGNALS))
    def test_runtime_crash_for_each_signal(self, number):
        cat, detail = attribute(ev("test", rc=128 + number, signal=number), cand())
        assert cat == "runtime_crash"
        assert sig.Signals(number).name in detail

    def test_runtime_timeout(self):
        cat, detail = attribute(ev("test", rc=137, timed_out=True, signal=9, duration=30.2), cand())
        assert cat == "runtime_timeout"
        assert "30.2" in detail

    def test_test_fail_with_nonzero_exit(self):
        cat, detail = attribute(ev("test", rc=1, tail="FAIL calc.c:42: calc_gcd(12, 18) == 6\n"), cand())
        assert cat == "test_fail"
        assert "exit code 1" in detail and "calc_gcd" in detail

    def test_test_fail_when_exit_code_is_zero(self):
        # parson / mpc / vec：测试失败只体现在输出里，退出码仍是 0
        assert attribute(ev("test", rc=0, tail="Tests failed: 3\n"), cand())[0] == "test_fail"

    def test_every_category_is_reachable(self):
        seen = {
            attribute(ev("pass", rc=0), cand())[0],
            attribute(ev(), cand(None))[0],
            attribute(ev("syntax", diagnostics=[err("expected ';'")]), cand())[0],
            attribute(ev("syntax", diagnostics=[err("'x' undeclared")]), cand())[0],
            attribute(ev("syntax", diagnostics=[err("too few arguments")]), cand())[0],
            attribute(ev("syntax", diagnostics=[err("label 'x' used but not defined")]), cand())[0],
            attribute(ev("build", tail="undefined reference to `f'"), cand())[0],
            attribute(ev("test", rc=139, signal=11), cand())[0],
            attribute(ev("test", rc=137, signal=9, timed_out=True), cand())[0],
            attribute(ev("test", rc=1), cand())[0],
            attribute(None, cand(violations=["call:exit"], issues=["policy_violation"]))[0],
        }
        assert seen == set(CATEGORIES)

    def test_policy_violation_beats_everything_except_missing_code(self):
        # 违反硬性约束的候选不进沙箱，ev 可以是 None；即使有评测结果（比如旧记录），也不能因为「碰巧通过」被放过
        cat, detail = attribute(None, cand(violations=["call:exit", "directive:#define"], issues=["policy_violation"]))
        assert cat == "policy_violation" and "call:exit" in detail and "directive:#define" in detail
        assert attribute(ev("pass", rc=0), cand(violations=["call:exit"]))[0] == "policy_violation"
        assert attribute(ev("pass", rc=0), cand(None, violations=["call:exit"]))[0] == "extract_fail"


# ----------------------------------------------------------------------
# attribute：优先级与边界
# ----------------------------------------------------------------------
class TestPrecedence:
    def test_syntax_beats_undeclared_beats_type_beats_other(self):
        d_other = err("redefinition of 'x'")
        d_type = err("too few arguments to function 'f'")
        d_undeclared = err("'y' undeclared (first use in this function)")
        d_syntax = err("expected ';' before '}' token")
        # 类别按规则顺序比较，不看诊断出现的先后
        assert attribute(ev("syntax", diagnostics=[d_other, d_type, d_undeclared, d_syntax]), cand())[0] \
            == "compile_syntax"
        assert attribute(ev("syntax", diagnostics=[d_other, d_type, d_undeclared]), cand())[0] \
            == "compile_undeclared"
        assert attribute(ev("syntax", diagnostics=[d_other, d_type]), cand())[0] == "compile_type"
        assert attribute(ev("syntax", diagnostics=[d_other]), cand())[0] == "compile_other"

    def test_warnings_and_notes_are_ignored(self):
        diags = [err("implicit declaration of function 'foo'", kind="warning"),
                 err("expected 'int' but argument is of type 'char *'", kind="note"),
                 err("redefinition of 'x'")]
        assert attribute(ev("syntax", diagnostics=diags), cand())[0] == "compile_other"

    def test_only_warnings_does_not_make_compile_failure(self):
        diags = [err("unused variable 'x'", kind="warning")]
        assert attribute(ev("test", rc=1, diagnostics=diags), cand())[0] == "test_fail"

    def test_compile_error_beats_link_error_text(self):
        tail = "undefined reference to `f'"
        assert attribute(ev("build", diagnostics=[err("expected ';'")], tail=tail), cand())[0] == "compile_syntax"

    def test_implicitly_declared_function_that_fails_to_link_is_compile_undeclared(self):
        warn = err("implicit declaration of function 'helper'", kind="warning", file="calc.c", line=9, column=5)
        tail = "calc.c:(.text+0x21): undefined reference to `helper'\ncollect2: error: ld returned 1 exit status\n"
        cat, detail = attribute(ev("build", rc=2, diagnostics=[warn], tail=tail), cand())
        assert cat == "compile_undeclared"
        assert "calc.c:9:5: implicit declaration of function 'helper'" in detail and "link" in detail

    def test_implicit_declaration_of_another_symbol_stays_link_error(self):
        warn = err("implicit declaration of function 'strlen'", kind="warning")
        tail = "calc.c:(.text+0x21): undefined reference to `helper'\n"
        assert attribute(ev("build", rc=2, diagnostics=[warn], tail=tail), cand())[0] == "link_error"

    def test_implicit_declaration_warning_alone_is_harmless(self):
        warn = err("implicit declaration of function 'strlen'", kind="warning")
        assert attribute(ev("test", rc=1, diagnostics=[warn], tail="FAIL x\n"), cand())[0] == "test_fail"

    def test_error_level_diagnostics_still_come_first(self):
        warn = err("implicit declaration of function 'helper'", kind="warning")
        tail = "undefined reference to `helper'"
        assert attribute(ev("build", diagnostics=[warn, err("expected ';' before '}' token")], tail=tail),
                         cand())[0] == "compile_syntax"

    def test_link_error_beats_runtime_rules(self):
        assert attribute(ev("test", rc=139, signal=11, tail="undefined reference to `f'"), cand())[0] == "link_error"

    def test_crash_signal_beats_timeout_and_test_fail(self):
        assert attribute(ev("test", rc=134, signal=6, timed_out=True), cand())[0] == "runtime_crash"

    def test_sigkill_without_timeout_is_plain_test_fail(self):
        # 137 可能是 OOM，不在崩溃信号集合里，也不是超时
        assert attribute(ev("test", rc=137, signal=9), cand())[0] == "test_fail"

    def test_crash_signal_outside_test_stage_is_not_runtime_crash(self):
        assert attribute(ev("build", rc=139, signal=11), cand())[0] == "compile_other"

    def test_extract_fail_beats_everything(self):
        assert attribute(ev("test", rc=139, signal=11, diagnostics=[err("expected ';'")]), cand(None))[0] \
            == "extract_fail"

    def test_syntax_stage_without_recognizable_diagnostics(self):
        cat, detail = attribute(ev("syntax", rc=1, tail="gcc: internal error"), cand())
        assert cat == "compile_other" and "syntax stage failed" in detail

    def test_build_stage_timeout_is_not_a_runtime_timeout(self):
        # 连编译都没走完就超时：不是「运行超时」，归 compile_other（不计入编译通过率），detail 写明是哪个阶段
        cat, detail = attribute(ev("build", rc=137, timed_out=True, signal=9), cand())
        assert cat == "compile_other" and "build stage timed out" in detail
        assert attribute(ev("syntax", rc=137, timed_out=True, signal=9), cand())[0] == "compile_other"


class TestDiagnosticShapes:
    def test_raw_gcc_json_shape(self):
        raw = {"kind": "error", "message": "expected ';' before '}' token", "children": [],
               "locations": [{"caret": {"file": "calc.c", "line": 111, "column": 11}}]}
        cat, detail = attribute(ev("syntax", diagnostics=[raw]), cand())
        assert cat == "compile_syntax"
        assert detail.startswith("calc.c:111:11: expected ';'")

    def test_flattened_shape_has_location_in_detail(self):
        d = err("'x' undeclared (first use in this function)", file="util.c", line=7, column=3)
        cat, detail = attribute(ev("syntax", diagnostics=[d]), cand())
        assert cat == "compile_undeclared"
        assert detail.startswith("util.c:7:3: 'x' undeclared")

    def test_missing_kind_counts_as_error(self):
        assert attribute(ev("syntax", diagnostics=[{"message": "expected ';'"}]), cand())[0] == "compile_syntax"

    def test_compile_category_helper(self):
        assert compile_category([]) is None
        assert compile_category([err("warn", kind="warning")]) is None
        assert compile_category([err("expected ';'")])[0] == "compile_syntax"

    def test_was_expected_is_a_type_error_not_a_syntax_error(self):
        msg = "incompatible types when returning type 'point_t' but 'int' was expected"
        assert attribute(ev("syntax", diagnostics=[err(msg)]), cand())[0] == "compile_type"
        # 真正的语法错误仍然归 compile_syntax，即便同一样本里还有类型错误
        assert attribute(ev("syntax", diagnostics=[err(msg), err("expected ';' before '}' token")]), cand())[0] \
            == "compile_syntax"

    def test_message_matching_is_case_insensitive(self):
        assert attribute(ev("syntax", diagnostics=[err("Expected ';'")]), cand())[0] == "compile_syntax"


class TestDetail:
    def test_detail_is_short_and_single_line(self):
        long_msg = "expected ';' " + "x" * 1000
        _, detail = attribute(ev("syntax", diagnostics=[err(long_msg)]), cand())
        assert len(detail) <= 300 and "\n" not in detail

    def test_candidate_issues_are_appended(self):
        _, detail = attribute(ev("syntax", diagnostics=[err("expected ';'")]), cand(issues=["truncated"]))
        assert "truncated" in detail

    def test_crash_detail_has_last_output_line(self):
        tail = "running tests\nmini_test: calc.c:113: calc_gcd: Assertion `a == 1' failed.\n"
        _, detail = attribute(ev("test", rc=134, signal=6, tail=tail), cand())
        assert "SIGABRT" in detail and "Assertion" in detail

    def test_attribute_returns_a_tuple_of_two_strings(self):
        out = attribute(ev("test", rc=1), cand())
        assert isinstance(out, tuple) and len(out) == 2 and all(isinstance(x, str) for x in out)


# ----------------------------------------------------------------------
# similarity
# ----------------------------------------------------------------------
def _mini_functions() -> dict[str, str]:
    """mini 项目里每个函数的原文，按函数名索引。"""
    from bench.extract import extract_functions

    out = {}
    for t in extract_functions(mini_spec(), MINI):
        data = (MINI / t.file).read_bytes()
        out[t.name] = data[t.start:t.end].decode("utf-8")
    return out


@pytest.fixture(scope="module")
def funcs() -> dict[str, str]:
    return _mini_functions()


_STR_OR_ID = re.compile(r'"(?:\\.|[^"\\])*"|[A-Za-z_]\w*')


def rename_identifiers(code: str, rename) -> str:
    """逐个标识符改名（字符串字面量原样保留）。rename 是 dict（没有的名字不变）或 callable。"""
    fn = rename if callable(rename) else (lambda name: rename.get(name, name))

    def sub(m: re.Match) -> str:
        text = m.group()
        return text if text.startswith('"') else fn(text)

    return _STR_OR_ID.sub(sub, code)


class TestSimilarity:
    def test_renamed_function_is_nearly_identical(self, funcs):
        src = funcs["array_max_gap"]
        renamed = rename_identifiers(src, {"array_max_gap": "f_17", "a": "v_1", "n": "v_2", "best": "v_3",
                                           "i": "v_4", "gap": "v_5", "calc_abs_diff": "f_3", "size_t": "t_9"})
        assert renamed != src
        assert normalized_similarity(src, renamed) > 0.9

    def test_renaming_every_non_keyword_gives_exactly_one(self, funcs):
        for name in ("calc_gcd", "str_dup_upper", "rect_normalize", "parse_digits"):
            src = funcs[name]
            renamed = rename_identifiers(src, lambda n: n if n in C_KEYWORDS else "zz" + n)
            assert normalized_similarity(src, renamed) == pytest.approx(1.0), name

    @pytest.mark.parametrize("a, b", [
        ("calc_gcd", "str_dup_upper"),
        ("rect_normalize", "parse_digits"),
        ("array_reverse", "calc_median3"),
        ("op_name", "stack_push"),
        ("calc_add_sat", "rand_next"),
    ])
    def test_different_functions_are_dissimilar(self, funcs, a, b):
        assert normalized_similarity(funcs[a], funcs[b]) < 0.5
        assert normalized_similarity(funcs[b], funcs[a]) < 0.5

    def test_identical_code_is_one(self, funcs):
        assert normalized_similarity(funcs["calc_gcd"], funcs["calc_gcd"]) == 1.0

    def test_whitespace_and_comments_do_not_matter(self):
        a = "int f(int a)\n{\n    /* 注释 */\n    return a + 1; // tail\n}\n"
        b = "int f(int a){return a+1;}"
        assert normalized_similarity(a, b) == pytest.approx(1.0)

    def test_comment_text_with_identifiers_is_dropped(self):
        a = "int f(int a) { return a; }"
        b = "int f(int a) { /* helper_one helper_two helper_three */ return a; }"
        assert normalized_similarity(a, b) == pytest.approx(1.0)

    def test_string_literals_are_kept(self):
        a = 'const char *f(void) { return "division by zero"; }'
        b = 'const char *f(void) { return "bad op"; }'
        assert normalized_similarity(a, b) < 1.0

    def test_structural_change_lowers_similarity(self, funcs):
        src = funcs["sum_array"]
        # L2 风格：for 改写成 while
        rewritten = ("int sum_array(const int *a, size_t n)\n{\n int total = 0; size_t i;\n"
                     " { i = 0; while (i < n) { total = total + a[i]; i++; } }\n return total;\n}\n")
        sim = normalized_similarity(src, rewritten)
        assert 0.4 < sim < 0.95

    def test_value_range_and_empty_inputs(self, funcs):
        for a in funcs.values():
            assert 0.0 <= normalized_similarity(a, funcs["calc_gcd"]) <= 1.0
        assert normalized_similarity("", "") == 1.0
        assert normalized_similarity("", funcs["calc_gcd"]) == 0.0
        assert normalized_similarity(funcs["calc_gcd"], "/* only a comment */") == 0.0

    def test_degenerate_huge_inputs_are_bounded(self):
        import time
        from bench.similarity import MAX_TOKENS
        junk = "x = x + y; " * 20000          # 12 万个 token
        t0 = time.time()
        v = normalized_similarity(junk, junk + "return 0;")
        assert time.time() - t0 < 20 and v == pytest.approx(1.0)      # 只比较前 MAX_TOKENS 个
        assert len(normalized_tokens(junk)) > MAX_TOKENS

    def test_long_functions_stay_fast_and_accurate(self):
        # 500+ token 的函数：autojunk 打开时 difflib 会把高频 token 当垃圾，比值失真
        body = "".join(f"  total += a[{i}] * {i};\n" for i in range(150))
        src = "int big(const int *a) {\n  int total = 0;\n" + body + "  return total;\n}\n"
        renamed = src.replace("big", "g_1").replace("total", "v_1").replace("a[", "w_2[")
        assert normalized_similarity(src, renamed) == pytest.approx(1.0)
        shuffled = "int big(const int *a) {\n  int total = 0;\n" + body[len(body) // 2:] + body[:len(body) // 2] \
            + "  return total;\n}\n"
        assert normalized_similarity(src, shuffled) > 0.5

    def test_tokens(self):
        assert normalized_tokens("int f(int a){ return a+1; /* c */ } // x") == \
            ["int", "ID", "(", "int", "ID", ")", "{", "return", "ID", "+", "1", ";", "}"]
        assert normalized_tokens("x->y += 0x1fUL;") == ["ID", "->", "ID", "+=", "0x1fUL", ";"]
        assert normalized_tokens('s = "a // not comment" "b";') == ["ID", "=", '"a // not comment"', '"b"', ";"]
