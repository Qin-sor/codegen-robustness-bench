"""bench.similarity 的词法接口（lex / exact_tokens / identifiers）与 normalized_equal（缺陷修复任务里 unchanged / restored 的判定）。

normalized_equal 以前放在 postprocess 里、测试放在 test_llm_repair.py；它是纯词法比较，现在和词法器一起放在 similarity，测试随之搬来。
全部离线、不需要 docker。
"""
from __future__ import annotations

import pytest

from bench.similarity import (
    MAX_TOKENS, exact_tokens, identifiers, lex, normalized_equal, normalized_similarity, normalized_tokens,
)

MUTANT = "int calc_gcd(int a, int b)\n{\n    return b != 0 ? a : calc_gcd(b, a % b);\n}"      # ROR：== 改成 !=
ORIGINAL = "int calc_gcd(int a, int b)\n{\n    return b == 0 ? a : calc_gcd(b, a % b);\n}"


class TestLexer:
    def test_lex_drops_comments_and_whitespace_and_tags_each_token(self):
        code = '/* c */ int f(void) { return puts("a // b") + \'x\' + 0x1F; } // end'
        kinds = [k for k, _ in lex(code)]
        texts = [t for _, t in lex(code)]
        assert texts == ["int", "f", "(", "void", ")", "{", "return", "puts", "(", '"a // b"', ")", "+", "'x'", "+", "0x1F", ";", "}"]
        assert kinds.count("string") == 1 and kinds.count("char") == 1 and kinds.count("number") == 1
        assert [t for k, t in lex(code) if k == "ident"] == ["int", "f", "void", "return", "puts"]
        assert lex("") == [] and lex(None) == []        # type: ignore[arg-type]

    def test_exact_tokens_keep_identifiers_and_normalized_tokens_do_not(self):
        code = "int f(int a) { return a + g(1); }"
        assert exact_tokens(code) == [t for _, t in lex(code)]
        assert "a" in exact_tokens(code) and "a" not in normalized_tokens(code)
        assert normalized_tokens(code).count("ID") == 4          # f a a g

    def test_identifiers_skip_keywords_literals_and_comments(self):
        code = 'int f(int a) { /* hidden */ return g("lit") + a; } // tail'
        assert identifiers(code) == {"f", "a", "g"}
        assert identifiers("") == set()


SUB = "int f(int a, int b)\n{\n    return a - b;\n}"


class TestNormalizedEqual:
    def test_identical_code_is_equal(self):
        assert normalized_equal(SUB, SUB) and normalized_equal(SUB, str(SUB))

    def test_layout_only_differences_are_ignored(self):
        squeezed = "int   f( int a,int b ){return a-b;}"
        tabs_and_crlf = "int f(int a, int b)\r\n\t{\r\n\t\treturn a - b;\r\n\t}\r\n\r\n"
        for other in (squeezed, tabs_and_crlf, "\n\n  " + SUB + "  \n"):
            assert normalized_equal(SUB, other) and normalized_equal(other, SUB)

    def test_comments_are_ignored(self):
        commented = "/* head */ int f(int a, int b) // trailing\n{\n    /* multi\n       line */ return a - b; /* done */\n}"
        assert normalized_equal(SUB, commented) and normalized_equal(commented, SUB)

    def test_comment_markers_inside_string_literals_are_code(self):
        with_text = 'const char *u(void) { return "/* not a comment */"; }'
        without = 'const char *u(void) { return ""; }'
        assert not normalized_equal(with_text, without)
        assert normalized_equal(with_text, with_text.replace("{ return", "{\n    return"))

    def test_identifiers_are_compared_verbatim(self):
        """与 similarity 不同，标识符不归一成 ID：交换操作数、换一个变量都算改动。"""
        swapped = SUB.replace("a - b", "b - a")
        assert not normalized_equal(SUB, swapped)
        assert normalized_similarity(SUB, swapped) == 1.0           # similarity 认不出这种差别，所以不能拿它判 unchanged / restored
        assert not normalized_equal(SUB, SUB.replace("int f(", "int g("))
        assert not normalized_equal("int k(int *x, int i, int j) { return x[i]; }", "int k(int *x, int i, int j) { return x[j]; }")

    @pytest.mark.parametrize(
        "old, new",
        [("a - b", "a + b"), ("a - b", "a - b - 1"), ("return a", "return -a"), ("int f", "long f"), ("a - b;", "a - b"),
         ("a - b", "(a - b)"), ("return a - b;", "return a - b; return 0;"), ("{", "{ if (a)"), ("a - b", "a-- b")],
    )
    def test_any_change_of_operators_literals_or_structure_counts(self, old, new):
        assert old in SUB
        assert not normalized_equal(SUB, SUB.replace(old, new, 1))

    def test_string_and_character_literals_are_compared_exactly(self):
        assert not normalized_equal('int g(void) { return puts("a  b"); }', 'int g(void) { return puts("a b"); }')
        assert not normalized_equal("char h(void) { return 'x'; }", "char h(void) { return 'y'; }")
        assert not normalized_equal("int n(void) { return 10; }", "int n(void) { return 0xA; }")

    def test_is_symmetric(self):
        for a, b in ((SUB, SUB.replace("-", "+")), (SUB, "  " + SUB), (MUTANT, ORIGINAL)):
            assert normalized_equal(a, b) == normalized_equal(b, a)

    @pytest.mark.parametrize(
        "a, b",
        [(None, None), (None, SUB), (SUB, None), ("", ""), ("", SUB), (SUB, ""), ("  \n\t", "\n"), ("/* c */", "/* c */"),
         ("// only a comment", "// only a comment"), ("   ", "   ")],
    )
    def test_missing_or_empty_code_is_never_equal(self, a, b):
        assert normalized_equal(a, b) is False

    def test_returns_a_bool(self):
        assert normalized_equal(SUB, SUB) is True and normalized_equal(SUB, "int z;") is False

    def test_differences_beyond_the_similarity_token_cap_are_still_seen(self):
        body = "x = x + 1;\n" * MAX_TOKENS                           # 远多于 MAX_TOKENS 个 token
        a = f"void k(int x)\n{{\n{body}}}"
        b = f"void k(int x)\n{{\n{body}x = 0;\n}}"
        assert normalized_equal(a, a + "\n/* tail */") and not normalized_equal(a, b)
