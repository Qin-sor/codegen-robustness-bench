"""bench.describe 与 bench.prompt 的测试：泄漏检查、描述生成的重试逻辑、生成阶段 prompt 的结构。

全部用 ScriptedLLM（按脚本回复）或 MockLLM，不联网；mini 夹具只被 libclang 解析，不执行。
"""
from __future__ import annotations

import json
import re

import pytest

from bench import describe as D
from bench.describe import (
    COMMON_WORDS,
    DescribeFailed,
    clean_description,
    describe,
    describe_detailed,
    leaks,
    word_count,
)
from bench.extract import Dependencies, FunctionTarget, extract_functions
from bench.llm import LLM, MOCK_DESCRIPTION, MockLLM, Reply
from bench.prompt import (
    CONSTRAINT_LINES,
    PROMPT_DIGEST,
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    USER_TEMPLATE,
    Task,
    build_messages,
    format_context,
)

from bench import corpus
from bench.pipeline import project_names
from tests.helpers import MINI, mini_spec, require_real_corpus


# ---------------------------------------------------------------------------
# 夹具与小工具
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def mini_targets() -> dict[str, FunctionTarget]:
    return {t.name: t for t in extract_functions(mini_spec(), MINI)}


@pytest.fixture(scope="module")
def mini_names() -> set[str]:
    """mini 源码里出现过的全部标识符拼写（含 libc 名字，比真实的项目标识符集合更宽，测试更严）。"""
    names: set[str] = set()
    for f in MINI.glob("*.[ch]"):
        names |= set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", f.read_text(encoding="utf-8")))
    return names


def source_of(t: FunctionTarget) -> str:
    return (MINI / t.file).read_bytes()[t.start : t.end].decode("utf-8")


# 只由常见英文词组成的文本，保证不会误触泄漏检查
_SAFE_WORDS = (
    "the function reads the first argument and then returns a value that the caller can use "
    "when the input is not valid it returns zero and does not change anything else"
).split()


def prose(n: int) -> str:
    """恰好 n 个词的安全文本。"""
    return " ".join(_SAFE_WORDS[i % len(_SAFE_WORDS)] for i in range(n)) + "."


class ScriptedLLM(LLM):
    """按脚本依次回复（用完后重复最后一条），并记录每次调用。"""

    def __init__(self, replies, model: str = "scripted") -> None:
        self.model = model
        self.replies = list(replies)
        self.calls: list[dict] = []

    def chat(self, messages, temperature=0.0, max_tokens=4096, seed=None, meta=None) -> Reply:
        self.calls.append(
            {"messages": [dict(m) for m in messages], "temperature": temperature, "max_tokens": max_tokens, "meta": meta}
        )
        item = self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]
        return Reply(item, self.model, 100, 50)


# ---------------------------------------------------------------------------
# leaks
# ---------------------------------------------------------------------------
class TestLeaks:
    NAMES = {"cJSON_Delete", "cJSON", "foo", "ab", "x", "buf_len", "stack_push", "MAX_DEPTH", "parse_value"}

    def test_exact_identifier_is_found(self):
        assert leaks("It then calls cJSON_Delete() on the result.", self.NAMES) == ["cJSON_Delete"]

    @pytest.mark.parametrize("text", ["foobar", "foo_bar", "xfoo", "foo2", "_foo", "a_foo", "foo_"])
    def test_longer_identifiers_are_not_matches(self, text):
        assert leaks(text, {"foo"}) == []

    @pytest.mark.parametrize(
        "text",
        ["(foo)", "foo,", "foo.", "a->foo", "foo->bar", '"foo"', "foo()", "foo[0]", "*foo", "&foo", "`foo`", "foo;", "x=foo+1", "foo-bar", "re-foo"],
    )
    def test_punctuation_neighbours_still_match(self, text):
        assert leaks(text, {"foo"}) == ["foo"]

    def test_digit_prefixed_token_is_not_an_identifier(self):
        assert leaks("2foo", {"foo"}) == []

    def test_non_ascii_neighbours_do_not_hide_a_match(self):
        assert leaks("中foo文", {"foo"}) == ["foo"]

    def test_short_names_match_case_sensitively(self):
        # 规范化后短于 5 个字符的名字只做精确匹配（区分大小写），免得 foo / Foo 这类短词到处误报
        assert leaks("Foo FOO fOO", {"foo"}) == []
        assert leaks("the cJSON library", {"cJSON"}) == ["cJSON"]

    def test_long_names_match_ignoring_case_and_separators(self):
        # 规范化匹配：去掉 _ 和 -、统一大小写后比较；这是防止「改个大小写 / 换个分隔符」规避检查
        assert leaks("cjson", {"cJSON"}) == ["cJSON"]
        assert leaks("CJSON", {"cJSON"}) == ["cJSON"]

    def test_short_names_are_never_checked_even_with_variants(self):
        assert leaks("AB Ab ab x X", {"ab", "x"}) == []

    def test_short_names_are_never_checked(self):
        assert leaks("ab x ab x", {"ab", "x"}) == []
        assert leaks("abc", {"abc"}) == ["abc"]   # 长度 3 刚好参与

    def test_common_english_words_are_not_leaks(self):
        names = {"item", "Item", "VALUE", "Node", "buffer", "length", "items", "parsed", "entries", "errors", "NULL", "true"}
        text = "The item value in the node buffer has a length; items are parsed, entries errors NULL true."
        assert leaks(text, names) == []

    def test_compound_identifiers_made_of_common_words_still_count(self):
        # buf_len 整体不是常见词；描述里原样写出它就是泄漏，写成「buffer length」则不是
        assert leaks("the buf_len field", self.NAMES) == ["buf_len"]
        assert leaks("the buffer length", self.NAMES) == []

    def test_order_is_first_appearance_and_deduplicated(self):
        assert leaks("beta_fn alpha_fn beta_fn gamma_fn alpha_fn", {"gamma_fn", "alpha_fn", "beta_fn"}) == ["beta_fn", "alpha_fn", "gamma_fn"]

    def test_empty_inputs(self):
        assert leaks("", self.NAMES) == []
        assert leaks("calls stack_push", set()) == []

    def test_different_name_sets_do_not_share_results(self):
        assert leaks("foo", {"foo"}) == ["foo"]
        assert leaks("foo", {"bar"}) == []
        assert leaks("foo", {"foo"}) == ["foo"]

    def test_multiline_text(self):
        assert leaks("line one\nthen parse_value\n\nand MAX_DEPTH", self.NAMES) == ["parse_value", "MAX_DEPTH"]

    def test_accepts_frozenset(self):
        assert leaks("stack_push", frozenset({"stack_push"})) == ["stack_push"]

    def test_mini_sources_are_flagged_but_safe_prose_is_not(self, mini_names):
        assert "stack_push" in leaks("it ends up in stack_push or intstack_t", mini_names)
        assert leaks(prose(100), mini_names) == []


class TestCommonWords:
    def test_list_is_lowercase_and_reasonably_large(self):
        assert len(COMMON_WORDS) > 800
        assert all(w == w.lower() and w.strip() == w for w in COMMON_WORDS)

    @pytest.mark.parametrize("w", ["item", "value", "string", "buffer", "length", "size", "index", "count", "null", "error", "result", "node", "malloc", "size_t", "NULL"])
    def test_representative_words(self, w):
        assert D._is_common(w)

    @pytest.mark.parametrize("w", ["items", "parsed", "parsing", "entries", "boxes", "values", "freed"])
    def test_simple_inflections_are_recognised(self, w):
        assert D._is_common(w)

    @pytest.mark.parametrize("w", ["cJSON", "stack_push", "intstack_t", "mpc_parser_t", "jsmn", "tinyexpr", "re_compile"])
    def test_project_specific_names_are_not_common(self, w):
        assert not D._is_common(w)

    def test_mock_description_is_safe_for_any_name_set(self):
        # 描述里每个词都在词表里，所以任何项目标识符集合都不可能让它泄漏——哪怕集合就是整张词表
        assert leaks(MOCK_DESCRIPTION, set(COMMON_WORDS)) == []
        assert leaks(MOCK_DESCRIPTION, {w.upper() for w in COMMON_WORDS} | {w.capitalize() for w in COMMON_WORDS}) == []
        tokens = {t for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", MOCK_DESCRIPTION) if len(t) >= 3}
        assert all(D._is_common(t) for t in tokens)

    def test_mock_description_meets_length_requirement(self):
        assert D.MIN_WORDS <= word_count(MOCK_DESCRIPTION) <= D.MAX_WORDS


class TestTextHelpers:
    def test_word_count(self):
        assert word_count("") == 0
        assert word_count("one two  three\nfour") == 4
        assert word_count("a well-known don't") == 3

    def test_word_count_ignores_list_markers_and_lone_punctuation(self):
        assert word_count("- one\n- two\n-  three - four ' — ...") == 4
        assert word_count("NULL-terminated string, e.g. buf_len") == 5

    def test_prose_helper_is_exact(self):
        assert word_count(prose(100)) == 100

    def test_clean_removes_fences_backticks_and_title(self):
        raw = "```text\nSpecification:\nThe `first` argument is read.\n```"
        assert clean_description(raw) == "The first argument is read."

    def test_clean_keeps_ordinary_text_and_collapses_blank_lines(self):
        assert clean_description("A.\n\n\n\nB.  \r\nC.") == "A.\n\nB.\nC."

    def test_clean_handles_none_and_empty(self):
        assert clean_description("") == ""
        assert clean_description(None) == ""  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# describe 的提示词
# ---------------------------------------------------------------------------
class TestDescribePrompt:
    def test_user_message_contains_code_context_and_forbidden_names(self, mini_targets):
        t = mini_targets["stack_push"]
        msgs = D.build_describe_messages(t, source_of(t), ["stack_push", "intstack_t"])
        assert [m["role"] for m in msgs] == ["system", "user"]
        user = msgs[1]["content"]
        assert "int stack_push(intstack_t *s, int v)" in user
        assert "#define MINI_STACK_CAP 8" in user
        assert "must not appear in your text: stack_push, intstack_t." in user

    def test_prerendered_context_text_is_used_as_is(self, mini_targets):
        t = mini_targets["stack_push"]
        user = D.build_describe_messages(t, source_of(t), context_text="/* rendered once */")[1]["content"]
        assert "/* rendered once */" in user and "MINI_STACK_CAP" not in user.split("relies on")[1].split("```c")[1].split("```")[0]

    def test_digest_is_hex_and_changes_with_the_version(self):
        import hashlib

        assert re.fullmatch(r"[0-9a-f]{64}", D.DESCRIBE_PROMPT_DIGEST)
        assert D.DESCRIBE_PROMPT_DIGEST != hashlib.sha256(D.DESCRIBE_SYSTEM.encode()).hexdigest()

    def test_no_forbidden_block_when_list_is_empty(self, mini_targets):
        t = mini_targets["stack_push"]
        assert "must not appear in your text" not in D.build_describe_messages(t, source_of(t))[1]["content"]

    def test_doc_comment_is_included_when_present(self, mini_targets):
        t = mini_targets["calc_apply"]
        assert t.doc_comment
        user = D.build_describe_messages(t, source_of(t))[1]["content"]
        assert "Existing comment on the function" in user and "运算类型非法" in user

    def test_system_prompt_states_the_requirements(self):
        s = D.DESCRIBE_SYSTEM
        for needle in ("English", "80 to 250 words", "the first argument", "Never write any identifier", "backticks"):
            assert needle in s

    def test_templates_have_only_known_placeholders(self):
        fields = {f for _, f, _, _ in __import__("string").Formatter().parse(D.DESCRIBE_USER_TEMPLATE) if f}
        assert fields == {"code", "doc_block", "context", "forbidden_block"}
        assert "{" not in D.DESCRIBE_SYSTEM and "}" not in D.DESCRIBE_SYSTEM

    def test_code_with_braces_does_not_break_formatting(self, mini_targets):
        t = mini_targets["stack_push"]
        user = D.build_describe_messages(t, "int f(void) { return '{'; } /* {x} {0} */")[1]["content"]
        assert "{x} {0}" in user

    def test_oversized_context_is_truncated(self, mini_targets):
        t = mini_targets["stack_push"]
        big = Dependencies(macros=[f"#define BIG_{i} {i}" for i in range(5000)])
        t2 = FunctionTarget(**{**t.__dict__, "deps": big})
        user = D.build_describe_messages(t2, "int f(void) { return 0; }")[1]["content"]
        assert "(truncated)" in user and len(user) < D.MAX_CONTEXT_CHARS + 3000


# ---------------------------------------------------------------------------
# describe / describe_detailed
# ---------------------------------------------------------------------------
class TestDescribeFlow:
    def test_clean_first_attempt(self, mini_targets, mini_names):
        t = mini_targets["stack_push"]
        llm = ScriptedLLM([prose(100)])
        assert describe(t, source_of(t), llm, mini_names) == prose(100)
        assert len(llm.calls) == 1
        call = llm.calls[0]
        assert call["temperature"] == 0.0
        assert call["meta"]["purpose"] == "describe" and call["meta"]["target_id"] == t.target_id
        assert [m["role"] for m in call["messages"]] == ["system", "user"]

    def test_prompt_lists_identifiers_found_in_the_material_but_not_common_words(self, mini_targets, mini_names):
        t = mini_targets["stack_push"]
        llm = ScriptedLLM([prose(100)])
        describe(t, source_of(t), llm, mini_names)
        user = llm.calls[0]["messages"][1]["content"]
        listed = user.split("must not appear in your text: ")[1].split(".\n")[0].split(", ")
        for must in ("stack_push", "intstack_t", "g_last_error", "MINI_STACK_CAP"):
            assert must in listed
        for never in ("len", "items", "size_t", "int", "NULL"):
            assert never not in listed

    def test_leak_triggers_retry_with_feedback(self, mini_targets, mini_names):
        t = mini_targets["stack_push"]
        leaky = prose(100) + " It calls stack_push on an intstack_t."
        llm = ScriptedLLM([leaky, prose(100)])
        res = describe_detailed(t, source_of(t), llm, mini_names)
        assert (res.failed, res.attempts, res.leak_retries, res.leaked) == (False, 2, 1, [])
        assert res.description == prose(100)
        second = llm.calls[1]["messages"]
        assert [m["role"] for m in second] == ["system", "user", "assistant", "user"]
        assert second[2]["content"] == leaky
        feedback = second[3]["content"]
        assert "stack_push, intstack_t" in feedback and "80" in feedback and "250" in feedback

    def test_gives_up_after_three_retries(self, mini_targets, mini_names):
        t = mini_targets["stack_push"]
        llm = ScriptedLLM([prose(100) + " stack_push"])
        with pytest.raises(DescribeFailed) as ei:
            describe(t, source_of(t), llm, mini_names)
        assert len(llm.calls) == 4   # 首次 + 最多 3 次重新生成
        r = ei.value.result
        assert (r.failed, r.reason, r.attempts, r.leak_retries, r.leaked) == (True, "leak", 4, 3, ["stack_push"])
        assert "stack_push" in str(ei.value) and t.target_id in str(ei.value)

    def test_describe_detailed_reports_failure_without_raising(self, mini_targets, mini_names):
        t = mini_targets["stack_push"]
        res = describe_detailed(t, source_of(t), ScriptedLLM([prose(100) + " stack_push"]), mini_names)
        assert res.failed and res.reason == "leak" and res.model == "scripted"
        assert res.prompt_tokens == 400 and res.completion_tokens == 200   # 4 次调用累计

    def test_max_retries_zero_means_single_attempt(self, mini_targets, mini_names):
        t = mini_targets["stack_push"]
        llm = ScriptedLLM([prose(100) + " stack_push"])
        res = describe_detailed(t, source_of(t), llm, mini_names, max_retries=0)
        assert res.failed and len(llm.calls) == 1

    def test_too_short_is_retried_with_word_count_feedback(self, mini_targets, mini_names):
        t = mini_targets["stack_push"]
        llm = ScriptedLLM([prose(20), prose(100)])
        res = describe_detailed(t, source_of(t), llm, mini_names)
        assert (res.failed, res.attempts, res.leak_retries, res.length_ok, res.n_words) == (False, 2, 0, True, 100)
        assert "20 words" in llm.calls[1]["messages"][-1]["content"]

    def test_too_long_is_retried(self, mini_targets, mini_names):
        t = mini_targets["stack_push"]
        llm = ScriptedLLM([prose(400), prose(120)])
        assert describe(t, source_of(t), llm, mini_names) == prose(120)
        assert "400 words" in llm.calls[1]["messages"][-1]["content"]

    def test_persistent_length_problem_without_leak_is_accepted(self, mini_targets, mini_names):
        t = mini_targets["stack_push"]
        llm = ScriptedLLM([prose(20)])
        res = describe_detailed(t, source_of(t), llm, mini_names)
        assert not res.failed and not res.length_ok and res.description == prose(20)
        assert len(llm.calls) == 4
        assert describe(t, source_of(t), ScriptedLLM([prose(20)]), mini_names) == prose(20)

    def test_clean_but_short_reply_is_kept_as_fallback_when_later_attempts_leak(self, mini_targets, mini_names):
        t = mini_targets["stack_push"]
        leaky = prose(100) + " stack_push"
        llm = ScriptedLLM([prose(60), leaky, leaky, leaky])
        res = describe_detailed(t, source_of(t), llm, mini_names)
        assert not res.failed and res.description == prose(60) and not res.length_ok and res.leaked == []
        assert (res.attempts, res.leak_retries, res.n_words) == (4, 2, 60)

    def test_fallback_prefers_the_reply_closest_to_the_allowed_range(self, mini_targets, mini_names):
        t = mini_targets["stack_push"]
        llm = ScriptedLLM([prose(300), prose(60), prose(400), prose(10)])
        res = describe_detailed(t, source_of(t), llm, mini_names)
        assert res.description == prose(60) and res.n_words == 60 and not res.failed

    def test_all_attempts_leaking_or_empty_still_fails(self, mini_targets, mini_names):
        t = mini_targets["stack_push"]
        res = describe_detailed(t, source_of(t), ScriptedLLM(["", prose(100) + " stack_push"]), mini_names)
        assert res.failed and res.reason == "leak"

    def test_llm_without_model_attribute_is_tolerated(self, mini_targets, mini_names):
        class Bare:
            def chat(self, messages, temperature=0.0, max_tokens=4096, seed=None, meta=None):
                return Reply(prose(100), "bare", 1, 1)

        t = mini_targets["stack_push"]
        assert describe_detailed(t, source_of(t), Bare(), mini_names).model == ""

    def test_leak_plus_wrong_length_reports_both(self, mini_targets, mini_names):
        t = mini_targets["stack_push"]
        llm = ScriptedLLM(["stack_push is short.", prose(100)])
        describe(t, source_of(t), llm, mini_names)
        feedback = llm.calls[1]["messages"][-1]["content"]
        assert "stack_push" in feedback and "words; it must be between" in feedback

    def test_empty_reply_is_retried_then_reported(self, mini_targets, mini_names):
        t = mini_targets["stack_push"]
        llm = ScriptedLLM(["", prose(100)])
        assert describe(t, source_of(t), llm, mini_names) == prose(100)
        assert "no specification text" in llm.calls[1]["messages"][-1]["content"]
        with pytest.raises(DescribeFailed) as ei:
            describe(t, source_of(t), ScriptedLLM(["  \n "]), mini_names)
        assert ei.value.result.reason == "empty"

    def test_fences_and_backticks_are_cleaned_before_checking(self, mini_targets, mini_names):
        t = mini_targets["stack_push"]
        llm = ScriptedLLM(["```\n" + prose(100) + "\n```"])
        assert describe(t, source_of(t), llm, mini_names) == prose(100)
        assert len(llm.calls) == 1

    def test_backticked_identifier_is_still_a_leak(self, mini_targets, mini_names):
        t = mini_targets["stack_push"]
        llm = ScriptedLLM([prose(100) + " Use `stack_push` here.", prose(100)])
        describe(t, source_of(t), llm, mini_names)
        assert len(llm.calls) == 2

    def test_llm_errors_propagate(self, mini_targets, mini_names):
        class Boom(LLM):
            model = "boom"

            def chat(self, *a, **k):
                raise RuntimeError("network down")

        t = mini_targets["stack_push"]
        with pytest.raises(RuntimeError, match="network down"):
            describe(t, source_of(t), Boom(), mini_names)

    def test_identical_inputs_give_identical_prompts(self, mini_targets, mini_names):
        t = mini_targets["calc_apply"]
        a, b = ScriptedLLM([prose(100)]), ScriptedLLM([prose(100)])
        describe(t, source_of(t), a, mini_names)
        describe(t, source_of(t), b, mini_names)
        assert a.calls[0]["messages"] == b.calls[0]["messages"]


class TestDescribeWithMockLLM:
    @pytest.mark.parametrize("mode", ["oracle", "broken", "wrong", "empty"])
    def test_every_mini_target_gets_a_valid_description_in_every_mode(self, mini_targets, mini_names, mode):
        llm = MockLLM(mode, {})
        for t in mini_targets.values():
            res = describe_detailed(t, source_of(t), llm, mini_names)
            assert not res.failed and res.attempts == 1 and res.leak_retries == 0, t.name
            assert res.description == MOCK_DESCRIPTION
            assert leaks(res.description, mini_names) == []
        assert all(c["meta"]["purpose"] == "describe" for c in llm.calls)


# ---------------------------------------------------------------------------
# 生成阶段 prompt
# ---------------------------------------------------------------------------
def make_task(**kw) -> Task:
    base = dict(
        task_id="mini:util.c:stack_push:L0",
        target_id="mini:util.c:stack_push",
        corpus="mini",
        level="L0",
        obf_name="stack_push",
        obf_signature="int stack_push(intstack_t *s, int v)",
        context=Dependencies(
            decls=["int helper(int x);"],
            types=["typedef struct intstack {\n    int items[MINI_STACK_CAP];\n    size_t len;\n} intstack_t;"],
            globals=["extern const char *g_last_error;"],
            macros=["#define MINI_STACK_CAP 8"],
            includes=["#include <stddef.h>", "#include <string.h>"],
        ),
        description="Pushes a value onto the stack.",
        includes=[],
        language_std="c99",
    )
    base.update(kw)
    return Task(**base)


class TestBuildMessages:
    def test_two_messages_system_then_user(self):
        msgs = build_messages(make_task())
        assert [m["role"] for m in msgs] == ["system", "user"]
        assert msgs[0]["content"] == SYSTEM_PROMPT
        assert all(isinstance(m["content"], str) and m["content"] for m in msgs)

    def test_system_prompt_follows_the_design(self):
        s = SYSTEM_PROMPT
        assert "C programmer" in s and "```c" in s
        assert "do not add #include" in s.lower() and "explanation" in s
        assert "signature" in s

    def test_user_sections_are_in_the_documented_order(self):
        user = build_messages(make_task())[1]["content"]
        marks = ["## Functional specification", "Pushes a value onto the stack.", "## Project context", "## Function to implement",
                 "int stack_push(intstack_t *s, int v)", "## Constraints"]
        positions = [user.index(m) for m in marks]
        assert positions == sorted(positions)

    def test_context_sections_order_and_content(self):
        user = build_messages(make_task())[1]["content"]
        marks = ["/* Headers already included */", "/* Macros */", "/* Types */", "/* Global variables */", "/* Functions you may call */"]
        positions = [user.index(m) for m in marks]
        assert positions == sorted(positions)
        assert "#define MINI_STACK_CAP 8" in user and "extern const char *g_last_error;" in user and "int helper(int x);" in user
        assert "typedef struct intstack {\n    int items[MINI_STACK_CAP];" in user

    def test_constraints_carry_language_std_name_and_forbidden_calls(self):
        user = build_messages(make_task(language_std="c89"))[1]["content"]
        constraints = user.split("## Constraints")[1]
        assert "Language standard: c89." in constraints and "gcc -std=c89" in constraints
        assert "definition of stack_push" in constraints
        for fn in ("system", "popen", "fork", "exec*", "abort", "exit"):
            assert fn in constraints
        assert "#include" in constraints

    def test_signature_is_shown_without_semicolon(self):
        user = build_messages(make_task(obf_signature="int stack_push(intstack_t *s, int v);"))[1]["content"]
        assert "```c\nint stack_push(intstack_t *s, int v)\n```" in user

    def test_task_includes_override_context_includes(self):
        user = build_messages(make_task(includes=["#include <stdlib.h>"]))[1]["content"]
        assert "#include <stdlib.h>" in user and "#include <stddef.h>" not in user
        user2 = build_messages(make_task())[1]["content"]
        assert "#include <stddef.h>" in user2

    def test_empty_context_gets_placeholder(self):
        user = build_messages(make_task(context=Dependencies()))[1]["content"]
        assert "/* (no additional declarations) */" in user

    def test_braces_in_description_are_safe(self):
        user = build_messages(make_task(description="Returns {x} and {0} and a literal {}."))[1]["content"]
        assert "Returns {x} and {0} and a literal {}." in user

    def test_deterministic(self):
        assert build_messages(make_task()) == build_messages(make_task())

    def test_template_is_identical_across_levels(self):
        """四级 prompt 模板完全相同：把 L1 任务里的名字换回去，就和 L0 的 prompt 一字不差。"""
        l0 = build_messages(make_task())
        l1_task = make_task(
            level="L1",
            task_id="mini:util.c:stack_push:L1",
            obf_name="f_17",
            obf_signature="int f_17(t_4 *v_1, int v_2)",
            context=Dependencies(
                decls=["int f_3(int v_9);"],
                types=["typedef struct t_5 {\n    int m_1[M_2];\n    size_t m_3;\n} t_4;"],
                globals=["extern const char *v_8;"],
                macros=["#define M_2 8"],
                includes=["#include <stddef.h>", "#include <string.h>"],
            ),
        )
        l1 = build_messages(l1_task)
        assert l1[0] == l0[0]
        back = l1[1]["content"]
        for new, old in [("f_17", "stack_push"), ("t_4", "intstack_t"), ("v_1", "s"), ("v_2", "v"), ("f_3", "helper"), ("v_9", "x"),
                         ("t_5", "intstack"), ("m_1", "items"), ("M_2", "MINI_STACK_CAP"), ("m_3", "len"), ("v_8", "g_last_error")]:
            back = back.replace(new, old)
        assert back == l0[1]["content"]
        for text in (l0[1]["content"], l1[1]["content"]):
            assert "L0" not in text and "L1" not in text and "obfus" not in text.lower()

    def test_templates_only_use_known_placeholders(self):
        fmt = __import__("string").Formatter()
        assert {f for _, f, _, _ in fmt.parse(USER_TEMPLATE) if f} == {"description", "context", "signature", "constraints"}
        for line in CONSTRAINT_LINES:
            assert {f for _, f, _, _ in fmt.parse(line) if f} <= {"language_std", "name"}
        assert "{" not in SYSTEM_PROMPT and "}" not in SYSTEM_PROMPT

    def test_version_and_digest(self):
        assert re.fullmatch(r"[a-z0-9.\-]+", PROMPT_VERSION)
        assert re.fullmatch(r"[0-9a-f]{64}", PROMPT_DIGEST)

    def test_digest_covers_templates_constraints_and_context_headings(self):
        """摘要的口径固定下来：模板、约束、上下文各段的标题与分隔符任何一处改动，摘要都会变。"""
        import hashlib

        from bench import prompt as P

        parts = [PROMPT_VERSION, SYSTEM_PROMPT, USER_TEMPLATE, *CONSTRAINT_LINES, P.EMPTY_CONTEXT, P.CONTEXT_JOINER,
                 *(f"{title}\x1e{sep}" for title, sep in P.CONTEXT_SECTIONS)]
        assert PROMPT_DIGEST == hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()
        assert [title for title, _ in P.CONTEXT_SECTIONS] == [
            "Headers already included", "Macros", "Types", "Global variables", "Functions you may call"]


class TestFormatContext:
    def test_order_comments_and_dedupe(self):
        ctx = Dependencies(
            decls=["int f(void);", "int f(void);"],
            types=["struct a { int x; };"],
            globals=["extern int g;"],
            macros=["#define A 1"],
            includes=["#include <a.h>", "#include <a.h>", "#include <b.h>"],
        )
        text = format_context(ctx)
        assert text.count("#include <a.h>") == 1 and text.count("int f(void);") == 1
        assert text.index("<b.h>") < text.index("#define A") < text.index("struct a") < text.index("extern int g;") < text.index("int f(void);")
        plain = format_context(ctx, comments=False)
        assert "/*" not in plain
        assert plain == "#include <a.h>\n#include <b.h>\n\n#define A 1\n\nstruct a { int x; };\n\nextern int g;\n\nint f(void);"

    def test_empty_sections_are_omitted(self):
        assert format_context(Dependencies()) == ""
        assert format_context(Dependencies(macros=["#define A 1"]), comments=False) == "#define A 1"

    def test_explicit_includes_replace_context_includes(self):
        ctx = Dependencies(includes=["#include <a.h>"])
        assert format_context(ctx, includes=["#include <z.h>"], comments=False) == "#include <z.h>"


class TestTask:
    def test_dict_and_json_roundtrip(self):
        t = make_task()
        d = json.loads(json.dumps(t.to_dict(), ensure_ascii=False))
        assert Task.from_dict(d) == t

    def test_from_dict_tolerates_missing_and_extra_keys(self):
        t = Task.from_dict({"task_id": "a", "target_id": "b", "corpus": "c", "level": "L0", "obf_name": "f", "obf_signature": "int f(void)",
                            "description": "d", "context": {"macros": ["#define A 1"]}, "extra": 1})
        assert t.context.macros == ["#define A 1"] and t.context.decls == [] and t.includes == [] and t.language_std == "c99"

    def test_field_order_matches_design(self):
        assert list(Task.__dataclass_fields__) == [
            "task_id", "target_id", "corpus", "level", "obf_name", "obf_signature", "context", "description", "includes", "language_std",
        ]


# ---------------------------------------------------------------------------
# 泄漏检查：大小写 / 分隔符规避、词组警告、放行词记录
# ---------------------------------------------------------------------------
class TestLeakEvasion:
    """改大小写、换连字符、写成驼峰是绕过精确匹配的常见办法；规范化匹配要把它们挡住。"""

    NAMES = {"cJSON", "cJSON_Parse", "buffer_skip_whitespace", "parse_value", "skip_whitespace"}

    @pytest.mark.parametrize("text, expected", [
        ("cjson_parse", ["cJSON_Parse"]),
        ("CJSON_PARSE", ["cJSON_Parse"]),
        ("the cjson library", ["cJSON"]),
        ("buffer-skip-whitespace", ["buffer_skip_whitespace"]),
        ("bufferSkipWhitespace", ["buffer_skip_whitespace"]),
        ("the BufferSkipWhitespace helper", ["buffer_skip_whitespace"]),
        ("cJSON Parse", ["cJSON"]),
    ])
    def test_variants_are_caught(self, text, expected):
        assert leaks(text, self.NAMES) == expected

    def test_plain_phrases_are_not_hard_leaks_but_are_warned_about(self):
        checker = D.LeakChecker(self.NAMES)
        for text in ("parse value and skip whitespace", "it will skip whitespace first"):
            assert checker.leaks(text) == []
        assert checker.phrase_hits("Parse the value, then skip whitespace.") == ["skip_whitespace"]
        assert checker.phrase_hits("Parse value, then skip whitespace.") == ["parse_value", "skip_whitespace"]
        assert checker.phrase_hits("parse value and skip whitespace") == ["parse_value", "skip_whitespace"]
        assert checker.phrase_hits("nothing to see") == []

    def test_phrase_hits_only_count_adjacent_words(self):
        checker = D.LeakChecker(self.NAMES)
        assert checker.phrase_hits("skip some whitespace") == []

    def test_common_word_names_never_leak_in_any_spelling(self):
        names = {"buffer_value", "Item", "item"}
        # buffer_value 整体不是常见词，会被抓；Item / item 是常见英文词，任何拼写都不算泄漏
        assert leaks("the BUFFER-VALUE and the ITEM", names) == ["buffer_value"]

    def test_allow_words_extend_the_common_word_list(self):
        names = {"expr", "te_expr", "regex"}
        assert leaks("an expr and a regex", names) == ["expr", "regex"]
        assert leaks("an expr and a regex", names, allow_words=["expr", "regex"]) == []

    def test_exempt_hits_are_recorded_not_blocked(self):
        checker = D.LeakChecker({"power", "base", "te_pow"}, allow_words=[])
        assert checker.leaks("raise the base to the power of the exponent") == []
        assert checker.exempt_hits("raise the base to the power of the exponent") == ["base", "power"]
        assert checker.exempt_hits("nothing") == []

    def test_digest_changes_with_names_and_allow_words(self):
        a = D.LeakChecker({"foo_bar"}).digest()
        assert a == D.LeakChecker({"foo_bar"}).digest()
        assert a != D.LeakChecker({"foo_bar", "baz_qux"}).digest()
        assert a != D.LeakChecker({"foo_bar"}, allow_words=["foo_bar"]).digest()

    def test_domain_words_are_not_in_the_general_list(self):
        # 领域词（expr / regex / ast ……）不放进通用词表，要在 spec.allow_words 里逐个声明
        for w in ("expr", "regex", "grammar", "ast", "codepoint", "anchor"):
            assert w not in COMMON_WORDS


class TestExemptionAudit:
    """常见英文词放行是有意的（描述里要能写出 add / base 这类词），但词表不能悄悄长大：
    每个真实语料被放行的项目标识符个数有上限（上限是 2026-10 审计时的个数加约 15% 余量，见 docs/实验记录.md 的放行情况表）。
    收紧或放宽 COMMON_WORDS / allow_words 时，要在这里同步改并重新审计。"""

    CAPS = {"cjson": 125, "inih": 22, "jsmn": 24, "mpc": 100, "parson": 60, "sds": 30, "tiny-regex-c": 18, "tinyexpr": 63, "vec": 6}

    @pytest.mark.parametrize("name", sorted(CAPS))
    def test_exempt_identifiers_stay_within_the_audited_cap(self, name):
        root = require_real_corpus(name)
        spec = corpus.load_spec(name)
        checker = D.LeakChecker(project_names(spec, root), spec.allow_words)
        assert len(checker.exempt) <= self.CAPS[name], sorted(checker.exempt)

    def test_every_spec_is_audited(self):
        assert sorted(self.CAPS) == sorted(corpus.list_specs())


class TestDescribeRecordsAudit:
    """放行词命中、词组警告、检查器摘要要随描述结果一起带出来，pipeline 写进 descriptions.jsonl。"""

    def test_result_carries_audit_fields(self, mini_targets, mini_names):
        t = mini_targets["calc_gcd"]
        text = prose(100) + " It uses the greatest common divisor of two numbers."
        llm = ScriptedLLM([text])
        res = describe_detailed(t, source_of(t), llm, {"calc_gcd", "stack_push"})
        assert not res.failed and res.exempt_hits == []
        assert re.fullmatch(r"[0-9a-f]{16}", res.checker)

    def test_own_name_in_description_is_reported_when_the_name_is_a_common_word(self, mini_targets):
        t = mini_targets["calc_gcd"]
        renamed = FunctionTarget(**{**t.__dict__, "name": "power"})
        text = prose(100) + " Raise it to the power of two."
        res = describe_detailed(renamed, source_of(t), ScriptedLLM([text]), {"power", "calc_gcd"})
        assert not res.failed and res.exempt_hits == ["power"] and res.own_name_hit is True

    def test_phrase_hits_are_collected(self, mini_targets):
        t = mini_targets["calc_gcd"]
        text = prose(100) + " Compute the stack push result."
        res = describe_detailed(t, source_of(t), ScriptedLLM([text]), {"stack_push", "calc_gcd"})
        assert not res.failed and res.phrase_hits == ["stack_push"]

    def test_variant_spelling_triggers_a_retry(self, mini_targets):
        t = mini_targets["calc_gcd"]
        bad = prose(100) + " Same as calc-gcd in spirit."
        good = prose(100)
        llm = ScriptedLLM([bad, good])
        res = describe_detailed(t, source_of(t), llm, {"calc_gcd"})
        assert not res.failed and res.leak_retries == 1 and len(llm.calls) == 2


class TestChatWithMeta:
    """设计文档 §4.7 的 LLM.chat 签名没有 meta：按那个签名写的子类也要能用。"""

    def test_llm_without_meta_parameter_is_called_without_it(self):
        from bench.llm import chat_with_meta

        class Plain(LLM):
            model = "plain"

            def chat(self, messages, temperature=0.0, max_tokens=4096, seed=None):
                return Reply("ok", "plain", 1, 1)

        assert chat_with_meta(Plain(), [{"role": "user", "content": "x"}], meta={"purpose": "describe"}).text == "ok"

    def test_llm_with_meta_gets_it(self):
        from bench.llm import chat_with_meta
        llm = ScriptedLLM(["hello"])
        chat_with_meta(llm, [{"role": "user", "content": "x"}], meta={"purpose": "describe"})
        assert llm.calls[0]["meta"] == {"purpose": "describe"}

    def test_describe_works_with_a_plain_llm(self, mini_targets):
        class Plain(LLM):
            model = "plain"

            def chat(self, messages, temperature=0.0, max_tokens=4096, seed=None):
                return Reply(prose(100), "plain", 1, 1)

        t = mini_targets["calc_gcd"]
        assert not describe_detailed(t, source_of(t), Plain(), {"calc_gcd"}).failed
