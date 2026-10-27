"""bench.postprocess 的测试：代码块抽取、签名文本解析、normalize 的各类 issue、与 MockLLM 的联动。

签名比对用 libclang，上下文取自 mini 夹具真实提取出来的依赖闭包（tests/fixtures/mini）；
不执行任何代码，也不需要 docker。
"""
from __future__ import annotations

import pytest

from bench.extract import Dependencies, FunctionTarget, extract_functions
from bench.llm import MockLLM
from bench.postprocess import (
    ISSUES,
    Candidate,
    extract_code,
    has_policy_directives,
    normalize,
    parse_blocks,
    process_reply,
    return_type_text,
    split_signature,
)
from bench.prompt import Task, build_messages
from tests.helpers import MINI, mini_spec, needs_docker


@pytest.fixture(scope="module")
def mini_targets() -> dict[str, FunctionTarget]:
    return {t.name: t for t in extract_functions(mini_spec(), MINI)}


def source_of(t: FunctionTarget) -> str:
    """目标函数在源文件里的原始定义（按字节偏移切）。"""
    return (MINI / t.file).read_bytes()[t.start : t.end].decode("utf-8")


FUNC = "int f(int x)\n{\n    return x + 1;\n}"


# ---------------------------------------------------------------------------
# extract_code / parse_blocks
# ---------------------------------------------------------------------------
class TestExtractCode:
    @pytest.mark.parametrize(
        "fence_open, fence_close",
        [
            ("```c", "```"),
            ("```C", "```"),
            ("``` c", "```"),
            ("```", "```"),
            ("```{.c .numberLines}", "```"),
            ("```language-c", "```"),
            ("```c99", "```"),
            ("~~~c", "~~~"),
            ("````c", "````"),
        ],
    )
    def test_fence_variants(self, fence_open, fence_close):
        reply = f"Here is the function:\n\n{fence_open}\n{FUNC}\n{fence_close}\n\nHope this helps."
        assert extract_code(reply, "f") == FUNC

    def test_no_block_returns_none(self):
        assert extract_code("Sorry, I cannot do that.") is None
        assert extract_code("") is None
        assert extract_code(None) is None  # type: ignore[arg-type]

    def test_bare_code_without_fence_is_not_a_block(self):
        assert extract_code(FUNC, "f") is None

    def test_inline_triple_backticks_on_one_line_is_not_a_fence(self):
        assert extract_code("```int f(void) { return 0; }```", "f") is None

    def test_blank_blocks_are_ignored(self):
        reply = f"```c\n\n```\n```c\n{FUNC}\n```"
        assert extract_code(reply, "f") == FUNC

    def test_picks_block_that_defines_target(self):
        usage = "int main(void)\n{\n    return f(1);\n}"
        reply = f"Usage:\n```c\n{usage}\n```\nImplementation:\n```c\n{FUNC}\n```"
        assert extract_code(reply, "f") == FUNC

    def test_definition_beats_mere_mention(self):
        caller = "int g(int y)\n{\n    return f(y) * 2;\n}"
        reply = f"```c\n{caller}\n```\n```c\n{FUNC}\n```"
        assert extract_code(reply, "f") == FUNC
        assert extract_code(reply, "g") == caller

    def test_mention_beats_unrelated(self):
        other = "int unrelated(void)\n{\n    return 3;\n}"
        usage = "x = f(2);"
        reply = f"```c\n{other}\n```\n```c\n{usage}\n```"
        assert extract_code(reply, "f") == usage

    def test_c_block_beats_other_languages(self):
        reply = f"```bash\ngcc -c f.c   # builds f\n```\n```c\n{FUNC}\n```"
        assert extract_code(reply, "f") == FUNC

    def test_first_of_equal_candidates_wins(self):
        second = "int f(int x)\n{\n    return x + 2;\n}"
        reply = f"```c\n{FUNC}\n```\n```c\n{second}\n```"
        assert extract_code(reply, "f") == FUNC

    def test_without_name_picks_first_function_like_block(self):
        reply = f"```c\nint x = 3;\n```\n```c\n{FUNC}\n```"
        assert extract_code(reply) == FUNC

    def test_call_inside_if_is_not_a_definition(self):
        # `if (f(x)) {` 的括号后面也有花括号，但它在函数体里面（花括号深度 > 0），不是 f 的定义
        wrapper = "int g(int x)\n{\n    if (f(x)) {\n        return 1;\n    }\n    return 0;\n}"
        reply = f"```c\n{wrapper}\n```\n```c\n{FUNC}\n```"
        assert extract_code(reply, "f") == FUNC

    def test_unterminated_fence_keeps_the_rest(self):
        partial = "int f(int x)\n{\n    if (x) {\n        return 1;"
        code = extract_code(f"```c\n{partial}", "f")
        assert code == partial
        blocks = parse_blocks(f"```c\n{partial}")
        assert [b.closed for b in blocks] == [False]

    def test_inner_shorter_fence_does_not_close_longer_one(self):
        reply = "````c\nint f(void) { return 0; }\n```\nint g(void) { return 1; }\n````"
        code = extract_code(reply, "f")
        assert "int g(void)" in code and "```" in code

    def test_crlf_and_indentation(self):
        reply = "Answer:\r\n    ```c\r\n    int f(int x)\r\n    {\r\n        return x;\r\n    }\r\n    ```\r\n"
        assert extract_code(reply, "f") == "int f(int x)\n{\n    return x;\n}"

    def test_parse_blocks_reports_language_and_closed(self):
        blocks = parse_blocks("```C\na\n```\ntext\n```\nb\n```\n~~~sh\nc\n~~~")
        assert [(b.lang, b.code, b.closed) for b in blocks] == [("c", "a", True), ("", "b", True), ("sh", "c", True)]

    def test_non_ascii_content_survives(self):
        code = "int f(int x)\n{\n    /* 中文注释：加一 */\n    return x + 1;\n}"
        assert extract_code(f"```c\n{code}\n```", "f") == code


# ---------------------------------------------------------------------------
# split_signature / return_type_text
# ---------------------------------------------------------------------------
class TestSignatureText:
    @pytest.mark.parametrize(
        "sig, ret, name, params",
        [
            ("int add(int a, int b)", "int", "add", "int a, int b"),
            ("static const char *op_name(op_kind_t op)", "static const char *", "op_name", "op_kind_t op"),
            ("MINI_RET(int) calc_abs_diff(int a, int b)", "MINI_RET(int)", "calc_abs_diff", "int a, int b"),
            ("CJSON_PUBLIC(cJSON *) cJSON_Parse(const char *value)", "CJSON_PUBLIC(cJSON *)", "cJSON_Parse", "const char *value"),
            ("mpc_err_t *mpca_lang_file(int flags, FILE *f, ...)", "mpc_err_t *", "mpca_lang_file", "int flags, FILE *f, ..."),
            ("void f(void);", "void", "f", "void"),
            ("int g(int (*cb)(int), int n) {", "int", "g", "int (*cb)(int), int n"),
        ],
    )
    def test_split(self, sig, ret, name, params):
        parts = split_signature(sig)
        assert parts is not None
        assert (parts.ret, parts.name, parts.params) == (ret, name, params)

    @pytest.mark.parametrize("sig", ["", "int x", "int (*get_op(int which))(int, int)", "(void)"])
    def test_split_unrecognized(self, sig):
        assert split_signature(sig) is None

    @pytest.mark.parametrize(
        "ret, expected",
        [
            ("int", "int"),
            ("static const char *", "const char *"),
            ("static inline int", "int"),
            ("CJSON_PUBLIC(cJSON *)", "cJSON *"),
            ("CJSON_PUBLIC(const char*)", "const char *"),
            ("MINI_RET(int)", "int"),
            ("M_17(t_5 *)", "t_5 *"),
            ("JSMN_API int", "int"),
            ("M_23 void", "void"),
            ("EXPORT sds", "sds"),
            ("static mpc_state_t", "mpc_state_t"),
            ("static unsigned long long int", "unsigned long long int"),
            ("static enum patch_operation", "enum patch_operation"),
            ("struct rect", "struct rect"),
            ("const struct rect *", "const struct rect *"),
            ("static void *", "void *"),
            ("__attribute__((unused)) static int", "int"),
            ("", ""),
        ],
    )
    def test_return_type_text(self, ret, expected):
        assert return_type_text(ret) == expected


# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------
class TestNormalizeBasics:
    def test_none_and_blank_mean_no_code_block(self):
        for code in (None, "", "   \n\t "):
            cand = normalize(code, "int f(int x)")
            assert cand.code is None and cand.issues == ["no_code_block"]

    def test_clean_code_without_context(self):
        cand = normalize(FUNC, "int f(int x)")
        assert cand == Candidate(FUNC, [])

    def test_surrounding_blank_lines_and_crlf_are_normalized(self):
        cand = normalize("\n\n" + FUNC.replace("\n", "\r\n") + "\n\n", "int f(int x)")
        assert cand.code == FUNC and cand.issues == []

    def test_candidate_default_issues_is_independent_list(self):
        a, b = Candidate("x"), Candidate("y")
        a.issues.append("truncated")
        assert b.issues == []

    def test_issue_vocabulary_matches_design(self):
        assert set(ISSUES) == {"no_code_block", "multiple_functions", "signature_mismatch", "added_include", "truncated",
                               "policy_violation"}

    @pytest.mark.parametrize("name", ["calc_gcd", "stack_push", "rect_contains", "point_add", "array_count_even", "op_name"])
    def test_original_source_is_clean(self, mini_targets, name):
        t = mini_targets[name]
        assert normalize(source_of(t), t.signature, True, t.deps) == Candidate(source_of(t).strip(), [])

    def test_every_mini_target_original_is_clean(self, mini_targets):
        dirty = {}
        for name, t in mini_targets.items():
            cand = normalize(source_of(t), t.signature, True, t.deps)
            if cand.issues:
                dirty[name] = cand.issues
        assert dirty == {}


class TestIncludes:
    CTX = Dependencies(includes=["#include <stddef.h>", "#include <limits.h>"])

    def test_added_include_is_stripped_and_flagged(self):
        code = "#include <string.h>\n" + FUNC
        cand = normalize(code, "int f(int x)", True, self.CTX)
        assert cand.code == FUNC
        assert cand.issues == ["added_include"]

    def test_redundant_include_is_stripped_silently(self):
        cand = normalize("#include <stddef.h>\n" + FUNC, "int f(int x)", True, self.CTX)
        assert cand.code == FUNC and cand.issues == []

    def test_without_context_every_include_counts_as_added(self):
        cand = normalize("#include <stddef.h>\n" + FUNC, "int f(int x)")
        assert cand.issues == ["added_include"]

    @pytest.mark.parametrize("line", ["# include <string.h>", "#include<string.h>", '#include "mini.h"', "  #include <string.h>  // for strlen"])
    def test_include_spelling_variants(self, line):
        cand = normalize(line + "\n" + FUNC, "int f(int x)", True, self.CTX)
        assert cand.code == FUNC and "added_include" in cand.issues

    def test_forbidden_includes_false_keeps_lines_and_reports_nothing(self):
        code = "#include <string.h>\n" + FUNC
        cand = normalize(code, "int f(int x)", False, self.CTX)
        assert cand.code == code and cand.issues == []

    def test_only_includes_means_no_usable_code(self):
        cand = normalize("#include <string.h>\n", "int f(int x)")
        assert cand.code is None and cand.issues == ["no_code_block", "added_include"]


class TestTruncated:
    @pytest.mark.parametrize(
        "code",
        [
            "int f(int x)\n{\n    if (x) {\n        return 1;",           # 花括号没闭合
            "int f(int x)\n{",                                              # 刚写出开头
            "int f(int x)\n{\n    return x; /* unfinished",                # 停在块注释里
            'int f(int x)\n{\n    puts("unfinished',                       # 停在字符串里
        ],
    )
    def test_truncated_detected(self, code):
        assert "truncated" in normalize(code, "int f(int x)").issues

    def test_finish_reason_length_forces_truncated(self):
        assert normalize(FUNC, "int f(int x)", finish_reason="length").issues == ["truncated"]
        assert normalize(FUNC, "int f(int x)", finish_reason="stop").issues == []

    def test_unbalanced_parentheses_alone_is_not_truncation(self):
        broken = "int f(int x)\n{\n    int t = 0\n    return (t;\n}"
        assert normalize(broken, "int f(int x)").issues == []

    def test_extra_closing_brace_is_not_truncation(self):
        assert "truncated" not in normalize(FUNC + "\n}", "int f(int x)").issues

    def test_braces_in_strings_chars_and_comments_are_ignored(self):
        code = (
            "int f(int x)\n{\n"
            '    const char *open = "{{{";\n'
            "    char close = '}';\n"
            "    /* { unbalanced in comment */\n"
            "    // } also here\n"
            "    return open[0] + close + x;\n}"
        )
        assert normalize(code, "int f(int x)").issues == []

    def test_escaped_quote_inside_string(self):
        code = 'int f(int x)\n{\n    const char *s = "a\\"{b";\n    return s[0] + x;\n}'
        assert normalize(code, "int f(int x)").issues == []


class TestMultipleFunctions:
    def test_helper_plus_target(self):
        code = "static int helper(int y)\n{\n    return y + 1;\n}\n\n" + FUNC
        assert normalize(code, "int f(int x)").issues == ["multiple_functions"]

    def test_prototype_plus_definition_is_one_function(self):
        assert normalize("int f(int x);\n" + FUNC, "int f(int x)").issues == []

    def test_two_definitions_of_target(self):
        assert "multiple_functions" in normalize(FUNC + "\n" + FUNC.replace("x + 1", "x + 2"), "int f(int x)").issues

    def test_example_main_after_target(self):
        code = FUNC + "\n\nint main(void)\n{\n    return f(1) == 2 ? 0 : 1;\n}"
        assert normalize(code, "int f(int x)").issues == ["multiple_functions"]


# 候选签名与 mini 目标期望签名比对：(用例名, 目标, 候选代码, 是否应记 signature_mismatch)
MISMATCH_CASES = [
    ("identical", "calc_gcd", "int calc_gcd(int a, int b) { return 0; }", False),
    ("parameter names do not matter", "calc_gcd", "int calc_gcd(int x, int y) { return 0; }", False),
    ("qualifier order does not matter", "rect_area", "int rect_area(struct rect const *r) { return 0; }", False),
    ("array parameter decays to pointer", "sum_array", "int sum_array(const int a[], size_t n) { return 0; }", False),
    ("size_t spelled as unsigned long", "sum_array", "int sum_array(const int *a, unsigned long n) { return 0; }", False),
    ("typedef and its struct are the same type", "stack_push", "int stack_push(struct intstack *s, int v) { return 0; }", False),
    ("function pointer typedef expanded", "array_count_if", "int array_count_if(const int *a, size_t n, int (*p)(int)) { return 0; }", False),
    ("macro-wrapped return type kept", "calc_abs_diff", "MINI_RET(int) calc_abs_diff(int a, int b) { return 0; }", False),
    ("macro-wrapped return type expanded", "calc_abs_diff", "int calc_abs_diff(int a, int b) { return 0; }", False),
    ("struct by value", "point_add", "point_t point_add(point_t a, point_t b) { return a; }", False),
    ("static kept, inline dropped", "sign_of", "static int sign_of(int x) { return 0; }", False),
    ("parameter type changed", "calc_gcd", "int calc_gcd(long a, int b) { return 0; }", True),
    ("const dropped from pointee", "sum_array", "int sum_array(int *a, size_t n) { return 0; }", True),
    ("fewer parameters", "calc_gcd", "int calc_gcd(int a) { return 0; }", True),
    ("more parameters", "calc_gcd", "int calc_gcd(int a, int b, int c) { return 0; }", True),
    ("return type changed", "calc_gcd", "long calc_gcd(int a, int b) { return 0; }", True),
    ("void return", "calc_gcd", "void calc_gcd(int a, int b) { }", True),
    ("function renamed", "calc_gcd", "int calc_gcd2(int a, int b) { return 0; }", True),
    ("only a prototype", "calc_gcd", "int calc_gcd(int a, int b);", True),
    ("static added", "calc_gcd", "static int calc_gcd(int a, int b) { return 0; }", True),
    ("static dropped", "sign_of", "int sign_of(int x) { return 0; }", True),
    ("unknown type name in head", "stack_push", "int stack_push(Stack *s, int v) { return 0; }", True),
    ("struct by value turned into pointer", "point_add", "point_t point_add(point_t *a, point_t b) { return b; }", True),
    ("enum parameter replaced by int", "calc_div_round", "int calc_div_round(int a, int b, int mode) { return 0; }", True),
    ("pointer depth changed", "op_name", "const char **op_name(op_kind_t op) { return 0; }", True),
]


class TestSignatureMismatch:
    @pytest.mark.parametrize("label, name, code, mismatch", MISMATCH_CASES, ids=[c[0] for c in MISMATCH_CASES])
    def test_with_project_context(self, mini_targets, label, name, code, mismatch):
        t = mini_targets[name]
        issues = normalize(code, t.signature, True, t.deps).issues
        assert ("signature_mismatch" in issues) is mismatch, (label, issues)

    def test_variadic_flag_matters(self):
        sig = "int logf2(const char *fmt, ...)"
        assert normalize("int logf2(const char *fmt, ...) { return 0; }", sig).issues == []
        assert normalize("int logf2(const char *fmt) { return 0; }", sig).issues == ["signature_mismatch"]

    def test_without_context_project_types_degrade_to_name_and_count(self, mini_targets):
        t = mini_targets["stack_push"]
        # 没给上下文：intstack_t 认不出来，只比较名字与参数个数
        assert normalize(source_of(t), t.signature).issues == []
        assert normalize("int stack_push(Whatever *s, int v) { return 0; }", t.signature).issues == []
        assert normalize("int stack_push(intstack_t *s) { return 0; }", t.signature).issues == ["signature_mismatch"]
        assert normalize("int stack_pop(intstack_t *s, int v) { return 0; }", t.signature).issues == ["signature_mismatch"]

    def test_non_ascii_comment_before_head_does_not_confuse_offsets(self, mini_targets):
        t = mini_targets["stack_push"]
        code = "/* 中文注释，让字节偏移与字符偏移错开 */\nint stack_push(Stack *s, int v) { return 0; }"
        assert normalize(code, t.signature, True, t.deps).issues == ["signature_mismatch"]
        ok = "/* 中文注释，让字节偏移与字符偏移错开 */\n" + source_of(t)
        assert normalize(ok, t.signature, True, t.deps).issues == []

    def test_broken_body_keeps_signature_check_working(self, mini_targets):
        t = mini_targets["calc_gcd"]
        broken = "int calc_gcd(int a, int b)\n{\n    int bench_tmp_ = 0\n    return (bench_tmp_;\n}"
        assert normalize(broken, t.signature, True, t.deps).issues == []
        wrong_head = "int calc_gcd(int a)\n{\n    int bench_tmp_ = 0\n    return (bench_tmp_;\n}"
        assert normalize(wrong_head, t.signature, True, t.deps).issues == ["signature_mismatch"]


class TestIssueOrdering:
    def test_issues_follow_the_documented_order(self, mini_targets):
        t = mini_targets["calc_gcd"]
        code = "#include <string.h>\nstatic int h(int y)\n{\n    return y;\n}\nint calc_gcd2(int a, int b)\n{\n    if (a) {"
        assert normalize(code, t.signature, True, t.deps).issues == [
            "multiple_functions",
            "signature_mismatch",
            "added_include",
            "truncated",
        ]


# ---------------------------------------------------------------------------
# process_reply 与 MockLLM 联动
# ---------------------------------------------------------------------------
def make_task(t: FunctionTarget, description: str = "Returns something.") -> Task:
    return Task(
        task_id=f"{t.target_id}:L0",
        target_id=t.target_id,
        corpus="mini",
        level="L0",
        obf_name=t.name,
        obf_signature=t.signature,
        context=t.deps,
        description=description,
        includes=[],
        language_std="c99",
    )


class TestProcessReply:
    def test_prose_around_block(self, mini_targets):
        t = mini_targets["calc_gcd"]
        reply = f"Sure! Here is the code.\n\n```c\n{source_of(t)}\n```\n\nIt uses Euclid's algorithm."
        cand = process_reply(reply, make_task(t))
        assert cand == Candidate(source_of(t).strip(), [])

    def test_picks_block_by_task_function_name(self, mini_targets):
        t = mini_targets["calc_gcd"]
        example = "int main(void)\n{\n    return calc_gcd(4, 6) == 2 ? 0 : 1;\n}"
        reply = f"```c\n{example}\n```\n```c\n{source_of(t)}\n```"
        assert process_reply(reply, make_task(t)).code == source_of(t).strip()

    def test_no_block(self, mini_targets):
        cand = process_reply("I cannot help with that.", make_task(mini_targets["calc_gcd"]))
        assert cand == Candidate(None, ["no_code_block"])

    def test_finish_reason_length(self, mini_targets):
        t = mini_targets["calc_gcd"]
        cand = process_reply(f"```c\n{source_of(t)}\n```", make_task(t), finish_reason="length")
        assert cand.issues == ["truncated"]

    def test_task_includes_take_precedence_over_context_includes(self, mini_targets):
        t = mini_targets["sum_array"]
        assert "#include <stdlib.h>" in t.deps.includes
        task = make_task(t)
        task.includes = ["#include <stddef.h>", "#include <math.h>"]   # prompt 里告诉模型已包含的头文件
        reply = f"```c\n#include <math.h>\n#include <stdlib.h>\n{source_of(t)}\n```"
        cand = process_reply(reply, task)
        assert cand.issues == ["added_include"]            # math.h 是提供的；stdlib.h 不在 task.includes 里，算新增
        assert cand.code == source_of(t).strip()
        # 没有 task.includes 时仍按 context.includes 判断
        task2 = make_task(t)
        assert process_reply(reply, task2).issues == ["added_include"]    # math.h 不在 context.includes 里
        assert process_reply(f"```c\n#include <stdlib.h>\n{source_of(t)}\n```", task2).issues == []

    def test_task_includes_feed_the_libclang_prefix(self, mini_targets):
        t = mini_targets["sum_array"]
        task = make_task(t)
        task.context = Dependencies()                    # 上下文里没有任何头文件
        task.includes = ["#include <stddef.h>"]          # size_t 只能靠 task.includes 认出来
        bad = "int sum_array(const int *a, long n) { return 0; }"
        assert process_reply(f"```c\n{source_of(t)}\n```", task).issues == []
        assert process_reply(f"```c\n{bad}\n```", task).issues == ["signature_mismatch"]

    def test_unclosed_fence_is_truncated_unless_the_caller_says_stop(self, mini_targets):
        t = mini_targets["calc_gcd"]
        task = make_task(t)
        unclosed = f"```c\n{source_of(t)}"
        assert process_reply(unclosed, task).issues == ["truncated"]
        assert process_reply(unclosed, task, finish_reason="length").issues == ["truncated"]
        assert process_reply(unclosed, task, finish_reason="stop").issues == []
        assert process_reply(unclosed + "\n```", task).issues == []

    def test_block_with_only_includes_has_no_usable_code(self, mini_targets):
        task = make_task(mini_targets["calc_gcd"])
        cand = process_reply("```c\n#include <string.h>\n```", task)
        assert cand == Candidate(None, ["no_code_block", "added_include"])

    def test_includes_policy_is_forwarded(self, mini_targets):
        t = mini_targets["calc_gcd"]
        reply = f"```c\n#include <string.h>\n{source_of(t)}\n```"
        assert process_reply(reply, make_task(t)).issues == ["added_include"]
        assert process_reply(reply, make_task(t), forbidden_includes=False).issues == []


class TestWithMockLLM:
    """四种 MockLLM 模式的输出走完 build_messages → chat → process_reply，对 mini 全部目标都符合各自的预期。"""

    @pytest.fixture(scope="class")
    def tasks_and_oracle(self, mini_targets):
        tasks = [make_task(t) for t in mini_targets.values()]
        oracle = {make_task(t).task_id: source_of(t) for t in mini_targets.values()}
        return tasks, oracle

    @staticmethod
    def run(mode, tasks, oracle):
        llm = MockLLM(mode, oracle)
        out = []
        for task in tasks:
            meta = {"task_id": task.task_id, "level": task.level, "obf_signature": task.obf_signature}
            reply = llm.chat(build_messages(task), temperature=0.0, max_tokens=1024, meta=meta)
            out.append((task, reply, process_reply(reply.text, task)))
        return out

    @pytest.mark.parametrize("mode", ["oracle", "wrong", "broken"])
    def test_code_modes_are_clean_candidates(self, tasks_and_oracle, mode):
        tasks, oracle = tasks_and_oracle
        results = self.run(mode, tasks, oracle)
        assert len(results) == len(tasks) >= 30
        dirty = {task.obf_name: cand.issues for task, _, cand in results if cand.issues or cand.code is None}
        assert dirty == {}

    def test_oracle_candidate_is_the_original_source(self, tasks_and_oracle):
        tasks, oracle = tasks_and_oracle
        for task, _, cand in self.run("oracle", tasks, oracle):
            assert cand.code == oracle[task.task_id].strip()

    def test_wrong_candidate_differs_from_original_but_keeps_signature(self, tasks_and_oracle):
        tasks, oracle = tasks_and_oracle
        for task, _, cand in self.run("wrong", tasks, oracle):
            assert cand.code != oracle[task.task_id].strip()
            assert cand.code.startswith(task.obf_signature)

    def test_empty_mode_yields_no_code_block(self, tasks_and_oracle):
        tasks, oracle = tasks_and_oracle
        for _, reply, cand in self.run("empty", tasks, oracle):
            assert "```" not in reply.text
            assert cand == Candidate(None, ["no_code_block"])


class TestThreadSafety:
    def test_concurrent_normalize_gives_consistent_results(self, mini_targets):
        import threading

        names = ["calc_gcd", "stack_push", "rect_contains", "point_add", "array_count_even", "op_name"]
        expected = {n: normalize(source_of(mini_targets[n]), mini_targets[n].signature, True, mini_targets[n].deps) for n in names}
        errors: list[str] = []

        def work(seed: int) -> None:
            try:
                for i in range(30):
                    n = names[(seed + i) % len(names)]
                    t = mini_targets[n]
                    cand = normalize(source_of(t), t.signature, True, t.deps)
                    bad = normalize(source_of(t).replace(n, n + "_x", 1), t.signature, True, t.deps)
                    if cand != expected[n] or bad.issues != ["signature_mismatch"]:
                        errors.append(f"{n}: {cand.issues} / {bad.issues}")
            except Exception as e:   # noqa: BLE001
                errors.append(repr(e))

        threads = [threading.Thread(target=work, args=(k,)) for k in range(6)]
        [th.start() for th in threads]
        [th.join() for th in threads]
        assert errors == []


# ---------------------------------------------------------------------------
# 政策检查：提示词里的硬性约束（禁用函数、内联汇编、#define 等）
# ---------------------------------------------------------------------------
class TestPolicyViolations:
    """候选里只写一句 exit(0) 就能让只看退出码的语料误判为通过，所以这类候选不进沙箱，归 policy_violation。"""

    SIG = "static int op_add(int a, int b)"

    def cand(self, mini_targets, body: str, **kw):
        t = mini_targets["op_add"]
        code = f"{t.signature}\n{{\n{body}\n}}"
        return normalize(code, t.signature, True, t.deps, **kw)

    @pytest.mark.parametrize("body, what", [
        ("    exit(0);", "call:exit"),
        ("    abort();\n    return 0;", "call:abort"),
        ("    _exit(0);", "call:_exit"),
        ("    _Exit(0);", "call:_Exit"),
        ("    quick_exit(0);", "call:quick_exit"),
        ("    atexit(0);\n    return a + b;", "call:atexit"),
        ("    system(\"true\");\n    return 0;", "call:system"),
        ("    execl(\"/bin/true\", \"true\", 0);\n    return 0;", "call:execl"),
        ("    execvp(\"true\", 0);\n    return 0;", "call:execvp"),
        ("    fork();\n    return 0;", "call:fork"),
        ("    signal(2, 0);\n    return 0;", "call:signal"),
        ("    raise(11);\n    return 0;", "call:raise"),
        ("    kill(0, 9);\n    return 0;", "call:kill"),
        ("    longjmp(0, 1);\n    return 0;", "call:longjmp"),
        ("    socket(2, 1, 0);\n    return 0;", "call:socket"),
        ("    void (*f)(int) = exit;\n    f(0);\n    return 0;", "call:exit"),
        ("    __builtin_trap();\n    return 0;", "call:__builtin_trap"),
        ("    __asm__ volatile (\"nop\");\n    return a + b;", "asm"),
    ])
    def test_forbidden_constructs_are_flagged(self, mini_targets, body, what):
        c = self.cand(mini_targets, body)
        assert "policy_violation" in c.issues and what in c.violations, (c.issues, c.violations)
        assert c.code is not None and not c.analysis_skipped

    def test_constructor_attribute_is_flagged(self, mini_targets):
        t = mini_targets["op_add"]
        code = f"__attribute__((constructor)) static void early(void) {{ exit(0); }}\n{t.signature}\n{{ return a + b; }}"
        c = normalize(code, t.signature, True, t.deps)
        assert "attribute:constructor" in c.violations and "policy_violation" in c.issues

    @pytest.mark.parametrize("line, what", [
        ("#define assert(x) ((void)0)", "directive:#define"),
        ("#undef NULL", "directive:#undef"),
        ("#pragma GCC optimize (\"O3\")", "directive:#pragma"),
    ])
    def test_directives_are_flagged_unless_the_original_needs_them(self, mini_targets, line, what):
        body = f"    {line}\n    return a + b;"
        c = self.cand(mini_targets, body)
        assert what in c.violations
        ok = self.cand(mini_targets, body, allow_directives=True)
        assert ok.violations == [] and "policy_violation" not in ok.issues

    def test_has_policy_directives(self):
        assert has_policy_directives("int f(void) {\n#define X 1\n return X; }")
        assert has_policy_directives("int f(void) {\n  # pragma once\n return 0; }")
        assert not has_policy_directives("int f(void) {\n#ifdef X\n return 1;\n#endif\n return 0; }")
        assert not has_policy_directives('int f(void) { return puts("#define x"); } /* #define y */')

    @pytest.mark.parametrize("body", [
        '    printf("exit(0) abort() #define x\\n");\n    return a + b;',            # 字符串字面量里的字不算
        "    /* exit(0); */\n    return a + b;",                                    # 注释里的不算
        "    int exit_code = a + b;\n    return exit_code;",                       # 标识符只是前缀相同
        "    struct { int abort; } s;\n    s.abort = a;\n    return s.abort + b;",    # 结构体成员
        "    int send = a;\n    return send + b;",                                   # 网络函数名只在调用位置才算
    ])
    def test_lookalikes_are_not_flagged(self, mini_targets, body):
        c = self.cand(mini_targets, body)
        assert c.violations == [] and "policy_violation" not in c.issues, c.violations

    def test_member_function_pointer_named_like_a_forbidden_function_is_fine(self, mini_targets):
        t = mini_targets["op_add"]
        code = (f"{t.signature}\n{{\n    struct ops {{ int (*exit)(int); }} o;\n    o.exit = 0;\n"
                "    if (o.exit) return o.exit(a);\n    return a + b;\n}")
        assert normalize(code, t.signature, True, t.deps).violations == []

    def test_project_function_with_a_forbidden_name_is_not_flagged(self, mini_targets):
        from bench.extract import Dependencies
        t = mini_targets["op_add"]
        deps = Dependencies(decls=["int signal(int level);"], types=[], globals=[], macros=[], includes=[])
        code = f"{t.signature}\n{{\n    return signal(a) + b;\n}}"
        assert normalize(code, t.signature, True, deps).violations == []

    def test_libclang_failure_falls_back_to_identifier_scan(self, mini_targets, monkeypatch):
        import bench.postprocess as pp
        monkeypatch.setattr(pp, "_analyze", lambda *a, **k: None)
        c = self.cand(mini_targets, "    exit(0);")
        assert c.analysis_skipped is True
        assert "call:exit" in c.violations and "policy_violation" in c.issues
        ok = self.cand(mini_targets, "    return a + b;")
        assert ok.analysis_skipped is True and ok.violations == []

    def test_process_reply_threads_the_flags_through(self, mini_targets):
        t = mini_targets["op_add"]
        reply = f"```c\n{t.signature}\n{{\n    exit(0);\n}}\n```"
        c = process_reply(reply, make_task(t))
        assert c.violations == ["call:exit"]
        d = process_reply(f"```c\n{t.signature}\n{{\n#define Q 1\n    return a + b;\n}}\n```", make_task(t),
                          allow_directives=True)
        assert d.violations == []

    def test_unmodified_original_functions_never_trip_the_policy(self, mini_targets):
        for name, t in mini_targets.items():
            c = normalize(source_of(t), t.signature, True, t.deps, allow_directives=has_policy_directives(source_of(t)))
            assert c.violations == [], (name, c.violations)


# ---------------------------------------------------------------------------
# MockLLM → process_reply → runner.evaluate（docker 沙箱）→ attribute：设计文档 §4.15 的 e2e 预期
# ---------------------------------------------------------------------------
@needs_docker
class TestMockThroughRunner:
    """四种模式的输出真的送进沙箱评测，归因结果要符合预期：
    oracle → pass；broken → compile_syntax；wrong → test_fail；empty → extract_fail。
    只取几个项目测试一定覆盖得到的函数（mini 的 stack_init 是例外：空函数体会让后面的测试读到未初始化的栈而崩溃，
    归因为 runtime_crash，所以不放进这组 test_fail 断言）。"""

    NAMES = ["calc_gcd", "op_name", "rect_normalize", "point_make"]

    def test_expected_category_per_mode(self, mini_targets):
        from concurrent.futures import ThreadPoolExecutor

        from bench import attribute, runner
        from bench.sandbox import DockerSandbox

        spec = mini_spec()
        sandbox = DockerSandbox(timeout_s=120)
        tasks = {n: make_task(mini_targets[n]) for n in self.NAMES}
        oracle = {tasks[n].task_id: source_of(mini_targets[n]) for n in self.NAMES}
        expected = {"oracle": "pass", "broken": "compile_syntax", "wrong": "test_fail", "empty": "extract_fail"}

        def run(job):
            name, mode = job
            task = tasks[name]
            reply = MockLLM(mode, oracle).chat(
                build_messages(task), meta={"task_id": task.task_id, "level": "L0", "obf_signature": task.obf_signature})
            cand = process_reply(reply.text, task)
            ev = runner.evaluate(spec, MINI, mini_targets[name], cand.code, sandbox, f"mock-{mode}", task.task_id, 0)
            return name, mode, attribute.attribute(ev, cand)[0]

        with ThreadPoolExecutor(3) as pool:
            got = list(pool.map(run, [(n, m) for n in self.NAMES for m in expected]))
        assert [(n, m, c) for n, m, c in got if c != expected[m]] == []
