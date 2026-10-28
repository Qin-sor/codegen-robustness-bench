"""bench.prompt 里缺陷修复任务的部分：RepairTask、build_repair_messages、REPAIR_PROMPT_VERSION / REPAIR_PROMPT_DIGEST。

约定见 docs/设计-缺陷修复任务.md §5：system 要求返回修正后的完整函数；user 依次是依赖上下文（小节格式与顺序同生成任务）、
``Buggy function:`` 加注入缺陷后的函数、约束行。没有功能规格，不透露缺陷的位置与算子。
全部离线，只拼字符串，不需要 docker。
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import string

import pytest

from bench import prompt as P
from bench.extract import Dependencies
from bench.prompt import (
    CONSTRAINT_LINES,
    PROMPT_DIGEST,
    PROMPT_VERSION,
    REPAIR_PROMPT_DIGEST,
    REPAIR_PROMPT_VERSION,
    REPAIR_SYSTEM_PROMPT,
    REPAIR_USER_TEMPLATE,
    SYSTEM_PROMPT,
    RepairTask,
    Task,
    build_messages,
    build_repair_messages,
    format_context,
    prune_context,
)

# 附录 §5 规定的 system 文字（逐字）
APPENDIX_SYSTEM = (
    "You are an experienced C programmer. The function below contains a bug that makes the project's tests fail. "
    "Return the corrected function only, as one ```c code block containing the complete definition. "
    "Keep the signature. Do not add #include lines. No explanation."
)

# L0 的变异体：``>=`` 被改成 ``>``
MUTANT = (
    "int stack_push(intstack_t *s, int v)\n"
    "{\n"
    "    if (s->len > MINI_STACK_CAP) {\n"
    "        g_last_error = \"stack full\";\n"
    "        return -1;\n"
    "    }\n"
    "    s->items[s->len++] = helper(v);\n"
    "    return 0;\n"
    "}"
)


def make_context() -> Dependencies:
    return Dependencies(
        decls=["int helper(int x);"],
        types=["typedef struct intstack {\n    int items[MINI_STACK_CAP];\n    size_t len;\n} intstack_t;"],
        globals=["extern const char *g_last_error;"],
        macros=["#define MINI_STACK_CAP 8"],
        includes=["#include <stddef.h>", "#include <string.h>"],
    )


def make_repair_task(**kw) -> RepairTask:
    base = dict(
        task_id="mini:util.c:stack_push@L0",
        target_id="mini:util.c:stack_push",
        corpus="mini",
        level="L0",
        obf_name="stack_push",
        obf_signature="int stack_push(intstack_t *s, int v)",
        context=make_context(),
        includes=[],
        language_std="c99",
        mutant_code=MUTANT,
        operator="ROR",
    )
    base.update(kw)
    return RepairTask(**base)


def make_generation_task(**kw) -> Task:
    """与 make_repair_task 同一个目标、同一份上下文的生成任务（拿来对照两套 prompt 共用的部分）。"""
    base = dict(
        task_id="mini:util.c:stack_push:L0",
        target_id="mini:util.c:stack_push",
        corpus="mini",
        level="L0",
        obf_name="stack_push",
        obf_signature="int stack_push(intstack_t *s, int v)",
        context=make_context(),
        description="Pushes a value onto the stack.",
        includes=[],
        language_std="c99",
    )
    base.update(kw)
    return Task(**base)


def user_of(task: RepairTask) -> str:
    return build_repair_messages(task)[1]["content"]


# ---------------------------------------------------------------------------
# RepairTask
# ---------------------------------------------------------------------------
class TestRepairTask:
    def test_field_names_and_order_follow_the_appendix(self):
        assert [f.name for f in dataclasses.fields(RepairTask)] == [
            "task_id", "target_id", "corpus", "level", "obf_name", "obf_signature", "context", "includes",
            "language_std", "mutant_code", "operator", "context_orphans"]

    def test_there_is_no_description_field(self):
        assert "description" not in {f.name for f in dataclasses.fields(RepairTask)}
        with pytest.raises(TypeError):
            make_repair_task(description="Pushes a value onto the stack.")

    def test_positional_construction_matches_the_documented_order(self):
        ctx = make_context()
        t = RepairTask("id", "tid", "mini", "L2", "f", "int f(void)", ctx, ["#include <a.h>"], "gnu99", "int f(void) { return 1; }", "NEG")
        assert (t.task_id, t.target_id, t.corpus, t.level, t.obf_name, t.obf_signature) == ("id", "tid", "mini", "L2", "f", "int f(void)")
        assert (t.context, t.includes, t.language_std, t.mutant_code, t.operator) == (
            ctx, ["#include <a.h>"], "gnu99", "int f(void) { return 1; }", "NEG")

    def test_defaults_match_the_generation_task(self):
        t = RepairTask("id", "tid", "mini", "L0", "f", "int f(void)", Dependencies(), mutant_code="int f(void) { return 1; }")
        g = Task("id", "tid", "mini", "L0", "f", "int f(void)", Dependencies(), "d")
        assert (t.includes, t.language_std) == (g.includes, g.language_std) == ([], "c99")
        other = RepairTask("id2", "tid", "mini", "L0", "f", "int f(void)", Dependencies())
        assert other.includes is not t.includes                           # 默认列表不在实例之间共享

    def test_dict_and_json_roundtrip(self):
        t = make_repair_task(includes=["#include <stdlib.h>"], language_std="c89")
        d = json.loads(json.dumps(t.to_dict(), ensure_ascii=False))
        assert RepairTask.from_dict(d) == t
        assert set(d) == {f.name for f in dataclasses.fields(RepairTask)}
        assert d["context"] == dataclasses.asdict(make_context())

    def test_from_dict_tolerates_missing_and_extra_keys(self):
        # 真实记录里还有 task / mutation / kill_mode / prompt_digest 等字段，读回来时忽略
        rec = {"task_id": "a", "target_id": "b", "corpus": "c", "level": "L1", "obf_name": "f_1", "obf_signature": "int f_1(void)",
               "context": {"macros": ["#define A 1"]}, "mutant_code": "int f_1(void) { return 0; }", "operator": "RVR",
               "task": "repair", "mutation": {"operator": "RVR", "site_index": 3}, "kill_mode": "test_fail", "prompt_digest": "x"}
        t = RepairTask.from_dict(rec)
        assert t.context.macros == ["#define A 1"] and t.context.decls == [] and t.includes == [] and t.language_std == "c99"
        assert (t.mutant_code, t.operator, t.level) == ("int f_1(void) { return 0; }", "RVR", "L1")

    def test_from_dict_turns_null_includes_into_an_empty_list(self):
        rec = make_repair_task().to_dict()
        rec["includes"] = None
        assert RepairTask.from_dict(rec).includes == []

    def test_context_orphans_default_to_empty_roundtrip_and_are_not_shared(self):
        t = RepairTask("id", "tid", "mini", "L0", "f", "int f(void)", Dependencies(), mutant_code="int f(void) { return 1; }")
        assert t.context_orphans == [] and RepairTask("id2", "tid", "mini", "L0", "f", "int f(void)", Dependencies()).context_orphans is not t.context_orphans
        t2 = make_repair_task(context_orphans=["helper", "MAXN"])
        assert RepairTask.from_dict(json.loads(json.dumps(t2.to_dict()))) == t2
        rec = t2.to_dict()
        rec["context_orphans"] = None
        assert RepairTask.from_dict(rec).context_orphans == []

    def test_both_task_classes_share_the_record_helpers(self):
        assert issubclass(Task, P._TaskBase) and issubclass(RepairTask, P._TaskBase)
        assert Task.from_dict.__func__ is RepairTask.from_dict.__func__          # 同一份 from_dict，不是各抄一份

    def test_the_task_record_does_not_share_state_with_the_task(self):
        t = make_repair_task(includes=["#include <a.h>"])
        d = t.to_dict()
        d["includes"].append("#include <b.h>")
        d["context"]["macros"].append("#define B 2")
        assert t.includes == ["#include <a.h>"] and t.context.macros == ["#define MINI_STACK_CAP 8"]


# ---------------------------------------------------------------------------
# build_repair_messages
# ---------------------------------------------------------------------------
class TestBuildRepairMessages:
    def test_two_messages_system_then_user(self):
        msgs = build_repair_messages(make_repair_task())
        assert [m["role"] for m in msgs] == ["system", "user"]
        assert msgs[0]["content"] == REPAIR_SYSTEM_PROMPT
        assert all(isinstance(m["content"], str) and m["content"] for m in msgs)

    def test_system_prompt_is_the_appendix_text(self):
        assert REPAIR_SYSTEM_PROMPT == APPENDIX_SYSTEM
        assert build_repair_messages(make_repair_task())[0]["content"] == APPENDIX_SYSTEM
        assert REPAIR_SYSTEM_PROMPT != SYSTEM_PROMPT

    def test_user_sections_are_in_the_documented_order(self):
        user = user_of(make_repair_task())
        marks = ["## Project context", "/* Headers already included */", "/* Macros */", "/* Types */", "/* Global variables */",
                 "/* Functions you may call */", "Buggy function:", "int stack_push(intstack_t *s, int v)", "## Constraints"]
        positions = [user.index(m) for m in marks]
        assert positions == sorted(positions) and len(set(positions)) == len(positions)

    def test_buggy_function_block_holds_the_mutant_verbatim(self):
        user = user_of(make_repair_task())
        assert f"\nBuggy function:\n```c\n{MUTANT}\n```\n" in user
        assert user.count(MUTANT) == 1
        assert user.count("int stack_push(intstack_t *s, int v)") == 1       # 签名只随函数出现一次，没有单独的签名小节

    def test_blank_lines_around_the_mutant_are_trimmed(self):
        padded = make_repair_task(mutant_code="\n\n" + MUTANT + "\n\n  \n")
        assert user_of(padded) == user_of(make_repair_task())

    def test_indentation_inside_the_mutant_is_kept(self):
        indented = "static int f(int a)\n{\n\tif (a)\n\t\treturn 1;\n    return 0;\n}"
        user = user_of(make_repair_task(mutant_code=indented))
        assert f"Buggy function:\n```c\n{indented}\n```" in user

    def test_there_is_no_functional_specification(self):
        user = user_of(make_repair_task())
        assert "Functional specification" not in user and "Implement the following" not in user
        assert "Pushes a value" not in user and "## Function to implement" not in user

    def test_nothing_reveals_the_defect_or_the_pipeline(self):
        task = make_repair_task(level="L1", operator="LOR", task_id="mini:util.c:stack_push@L1#repair")
        text = "\n".join(m["content"] for m in build_repair_messages(task))
        for needle in ("LOR", "mini:util.c", "#repair", "@L1"):
            assert needle not in text, needle
        assert not re.search(r"\bL[0-3]\b", text)                                 # 不透露混淆级
        lowered = text.lower()
        for word in ("mutant", "mutation", "operator", "obfus", "test output", "failing test"):
            assert word not in lowered, word

    def test_context_is_rendered_by_format_context_in_the_generation_format(self):
        task = make_repair_task()
        expected = format_context(task.context, includes=None)
        assert f"## Project context\nThese declarations are already in scope where the function will be placed. Do not repeat them.\n```c\n{expected}\n```\n" in user_of(task)
        marks = ["/* Headers already included */", "/* Macros */", "/* Types */", "/* Global variables */", "/* Functions you may call */"]
        user = user_of(task)
        assert [user.index(m) for m in marks] == sorted(user.index(m) for m in marks)
        assert "#define MINI_STACK_CAP 8" in user and "extern const char *g_last_error;" in user and "int helper(int x);" in user
        assert "typedef struct intstack {\n    int items[MINI_STACK_CAP];" in user

    def test_context_section_is_word_for_word_the_generation_one(self):
        """``## Project context`` 小节（标题、说明句、代码块）和生成任务的逐字相同，不论上下文是否为空、includes 是否被覆盖。"""
        pattern = re.compile(r"## Project context\n.*?\n```\n", re.S)
        for ctx, includes in ((make_context(), []), (make_context(), ["#include <stdlib.h>"]), (Dependencies(), [])):
            gen = build_messages(make_generation_task(context=ctx, includes=includes))[1]["content"]
            rep = user_of(make_repair_task(context=ctx, includes=includes))
            m_gen, m_rep = pattern.search(gen), pattern.search(rep)
            assert m_gen and m_rep and m_gen.group() == m_rep.group()

    def test_constraints_are_the_generation_constraint_lines(self):
        for std, name in (("c99", "stack_push"), ("c89", "f_17"), ("gnu99", "x")):
            sig = f"int {name}(int a)"
            gen = build_messages(make_generation_task(language_std=std, obf_name=name, obf_signature=sig))[1]["content"]
            rep = user_of(make_repair_task(language_std=std, obf_name=name, obf_signature=sig))
            assert rep.split("## Constraints\n", 1)[1] == gen.split("## Constraints\n", 1)[1]
            lines = rep.split("## Constraints\n", 1)[1].splitlines()
            assert lines == ["- " + line.format(language_std=std, name=name) for line in CONSTRAINT_LINES]

    def test_constraints_carry_language_std_name_and_forbidden_calls(self):
        constraints = user_of(make_repair_task(language_std="c89")).split("## Constraints")[1]
        assert "Language standard: c89." in constraints and "gcc -std=c89" in constraints
        assert "definition of stack_push" in constraints
        for fn in ("system", "popen", "fork", "exec*", "abort", "exit", "longjmp"):
            assert fn in constraints
        assert "#include" in constraints and "inline assembly" in constraints and "networking" in constraints
        assert set(P.FORBIDDEN_CALLS) <= set(re.split(r"[,\s]+", constraints))

    def test_task_includes_override_context_includes(self):
        user = user_of(make_repair_task(includes=["#include <stdlib.h>"]))
        assert "#include <stdlib.h>" in user and "#include <stddef.h>" not in user
        assert "#include <stddef.h>" in user_of(make_repair_task())

    def test_empty_context_gets_the_placeholder(self):
        assert "/* (no additional declarations) */" in user_of(make_repair_task(context=Dependencies()))

    def test_braces_and_format_markers_in_the_code_are_safe(self):
        code = ('int f(int a)\n{\n    const char *s = "{context} {code} {constraints} {0} {} %s";\n'
                '    struct { int x; } p = {0};\n    return p.x + (int)s[0] + a;\n}')
        user = user_of(make_repair_task(mutant_code=code, obf_name="f", obf_signature="int f(int a)"))
        assert f"Buggy function:\n```c\n{code}\n```" in user
        assert user.count("{constraints}") == 1 and user.count("{context}") == 1       # 代码里的字面量原样出现，没有被二次替换

    def test_deterministic_and_pure(self):
        task = make_repair_task()
        first = build_repair_messages(task)
        assert first == build_repair_messages(task) == build_repair_messages(make_repair_task())
        first[1]["content"] = "tampered"
        first.append({"role": "assistant", "content": "x"})
        assert build_repair_messages(task) == build_repair_messages(make_repair_task())

    def test_operator_and_ids_do_not_change_the_prompt(self):
        base = build_repair_messages(make_repair_task())
        assert build_repair_messages(make_repair_task(operator="DGS", task_id="another-id", target_id="other:t")) == base

    def test_mutant_code_changes_the_prompt(self):
        other = MUTANT.replace("> MINI_STACK_CAP", ">= MINI_STACK_CAP")
        assert build_repair_messages(make_repair_task(mutant_code=other)) != build_repair_messages(make_repair_task())

    @pytest.mark.parametrize("code", ["", " ", "\n\n", "  \n \t\n", None])
    def test_an_empty_mutant_is_rejected(self, code):
        with pytest.raises(ValueError, match="mutant_code"):
            build_repair_messages(make_repair_task(mutant_code=code))

    def test_a_mutant_that_still_carries_comments_is_rejected(self):
        """L1 以上注释里仍是项目原名：带注释的函数文本不能进提示词（最后一道关；mutate 阶段本来就先去注释）。"""
        for commented in (MUTANT.replace("{\n", "{\n    /* stack_push guards against overflow */\n", 1),
                          MUTANT.replace("return -1;", "return -1; // stack_push failed")):
            with pytest.raises(ValueError, match="注释"):
                build_repair_messages(make_repair_task(mutant_code=commented))

    def test_comment_markers_inside_string_literals_are_not_comments(self):
        code = 'const char *url(void)\n{\n    return "http://example.org/*x*/";\n}'
        user = user_of(make_repair_task(mutant_code=code, obf_name="url", obf_signature="const char *url(void)"))
        assert code in user

    def test_the_template_is_identical_across_levels(self):
        """四级 prompt 模板完全相同：把 L1 任务里的名字换回去，就和 L0 的 prompt 一字不差。"""
        l0 = build_repair_messages(make_repair_task())
        l1_mutant = (
            "int f_17(t_4 *v_1, int v_2)\n"
            "{\n"
            "    if (v_1->m_3 > M_2) {\n"
            "        v_8 = \"stack full\";\n"
            "        return -1;\n"
            "    }\n"
            "    v_1->m_1[v_1->m_3++] = f_3(v_2);\n"
            "    return 0;\n"
            "}"
        )
        l1_task = make_repair_task(
            level="L1",
            task_id="mini:util.c:stack_push@L1",
            obf_name="f_17",
            obf_signature="int f_17(t_4 *v_1, int v_2)",
            context=Dependencies(
                decls=["int f_3(int v_9);"],
                types=["typedef struct t_5 {\n    int m_1[M_2];\n    size_t m_3;\n} t_4;"],
                globals=["extern const char *v_8;"],
                macros=["#define M_2 8"],
                includes=["#include <stddef.h>", "#include <string.h>"],
            ),
            mutant_code=l1_mutant,
        )
        l1 = build_repair_messages(l1_task)
        assert l1[0] == l0[0]
        back = l1[1]["content"]
        for new, old in [("f_17", "stack_push"), ("t_4", "intstack_t"), ("v_1", "s"), ("v_2", "v"), ("f_3", "helper"), ("v_9", "x"),
                         ("t_5", "intstack"), ("m_1", "items"), ("M_2", "MINI_STACK_CAP"), ("m_3", "len"), ("v_8", "g_last_error")]:
            back = back.replace(new, old)
        assert back == l0[1]["content"]

    def test_templates_only_use_known_placeholders(self):
        fmt = string.Formatter()
        assert {f for _, f, _, _ in fmt.parse(REPAIR_USER_TEMPLATE) if f} == {"context", "code", "constraints"}
        assert "{" not in REPAIR_SYSTEM_PROMPT and "}" not in REPAIR_SYSTEM_PROMPT

    def test_the_two_prompts_are_separate_templates(self):
        assert REPAIR_USER_TEMPLATE != P.USER_TEMPLATE and REPAIR_SYSTEM_PROMPT != P.SYSTEM_PROMPT
        gen = build_messages(make_generation_task())
        assert "Buggy function" not in gen[1]["content"] and "bug" not in gen[0]["content"].lower()


# ---------------------------------------------------------------------------
# 版本与摘要
# ---------------------------------------------------------------------------
class TestRepairPromptDigest:
    def test_version_and_digest_are_named_apart_from_the_generation_ones(self):
        assert REPAIR_PROMPT_VERSION == "repair-v2"
        assert re.fullmatch(r"[a-z0-9.\-]+", REPAIR_PROMPT_VERSION)
        assert re.fullmatch(r"[0-9a-f]{64}", REPAIR_PROMPT_DIGEST)
        assert REPAIR_PROMPT_VERSION != PROMPT_VERSION and REPAIR_PROMPT_DIGEST != PROMPT_DIGEST
        assert PROMPT_VERSION == "gen-v1"                                      # 生成任务的版本没被动

    def test_digest_formula_is_pinned(self):
        """摘要的口径固定下来：版本、两段模板、共用的约束行、空上下文占位、上下文各段的标题与分隔符。"""
        parts = [REPAIR_PROMPT_VERSION, REPAIR_SYSTEM_PROMPT, REPAIR_USER_TEMPLATE, *CONSTRAINT_LINES, P.EMPTY_CONTEXT, P.CONTEXT_JOINER,
                 *(f"{title}\x1e{sep}" for title, sep in P.CONTEXT_SECTIONS)]
        expected = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()
        assert REPAIR_PROMPT_DIGEST == expected == P._repair_prompt_digest()
        # 生成任务与修复任务由同一个 _digest 算：生成任务的摘要没有因为重构变（它和已有的结果、缓存的提示词摘要对应）
        parts[0:3] = [PROMPT_VERSION, SYSTEM_PROMPT, P.USER_TEMPLATE]
        assert PROMPT_DIGEST == hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()

    @pytest.mark.parametrize(
        "name, value",
        [
            ("REPAIR_PROMPT_VERSION", "repair-v3"),
            ("REPAIR_SYSTEM_PROMPT", REPAIR_SYSTEM_PROMPT + " Be brief."),
            ("REPAIR_SYSTEM_PROMPT", REPAIR_SYSTEM_PROMPT.replace("No explanation.", "No explanations.")),
            ("REPAIR_USER_TEMPLATE", REPAIR_USER_TEMPLATE.replace("Buggy function:", "Faulty function:")),
            ("REPAIR_USER_TEMPLATE", REPAIR_USER_TEMPLATE + "\n"),
            ("CONSTRAINT_LINES", (*CONSTRAINT_LINES, "Be careful.")),
            ("CONSTRAINT_LINES", CONSTRAINT_LINES[1:]),
            ("EMPTY_CONTEXT", "/* nothing */"),
            ("CONTEXT_JOINER", "\n"),
            ("CONTEXT_SECTIONS", (("Header files", "\n"), *P.CONTEXT_SECTIONS[1:])),
            ("CONTEXT_SECTIONS", ((P.CONTEXT_SECTIONS[0][0], "\n\n"), *P.CONTEXT_SECTIONS[1:])),
        ],
    )
    def test_digest_changes_with_every_piece_of_template_text(self, monkeypatch, name, value):
        assert getattr(P, name) != value
        monkeypatch.setattr(P, name, value)
        assert P._repair_prompt_digest() != REPAIR_PROMPT_DIGEST

    @pytest.mark.parametrize(
        "name, value",
        [("PROMPT_VERSION", "gen-v2"), ("SYSTEM_PROMPT", SYSTEM_PROMPT + " Be brief."), ("USER_TEMPLATE", P.USER_TEMPLATE + "\n")],
    )
    def test_digest_ignores_the_generation_only_templates(self, monkeypatch, name, value):
        monkeypatch.setattr(P, name, value)
        assert P._repair_prompt_digest() == REPAIR_PROMPT_DIGEST

    def test_digest_is_recomputed_from_the_current_values_every_time(self, monkeypatch):
        before = P._repair_prompt_digest()
        monkeypatch.setattr(P, "REPAIR_SYSTEM_PROMPT", "changed")
        changed = P._repair_prompt_digest()
        monkeypatch.undo()
        assert changed != before == P._repair_prompt_digest() == REPAIR_PROMPT_DIGEST


# ---------------------------------------------------------------------------
# prune_context：上下文只留注入缺陷后的函数还用得到的原型 / 宏 / 全局
# ---------------------------------------------------------------------------
def deps(**kw) -> Dependencies:
    base = dict(decls=[], types=[], globals=[], macros=[], includes=["#include <stddef.h>"])
    base.update(kw)
    return Dependencies(**base)


class TestPruneContext:
    FULL = dict(
        decls=["int mpc_expect(mpc_parser_t *a, const char *e);", "static int helper(int x);"],
        macros=["#define MAXN 10", "#define TWICE(x) ((x) * 2)"],
        globals=["extern int g_count;", "extern const char *names[12];"],
        types=["typedef struct { int items[MAXN]; } box_t;"])

    def test_entries_only_the_removed_code_used_are_dropped_and_named(self):
        # 原函数是 return mpc_expect(a, e);，缺陷把它换成了 return NULL;
        pruned, orphans = prune_context(deps(**self.FULL), "mpc_parser_t *f(mpc_parser_t *a, const char *e) { return NULL; }")
        assert pruned.decls == [] and pruned.globals == []
        assert pruned.macros == ["#define MAXN 10"]                  # 类型定义（不裁）里的数组长度还用着它，留下
        assert orphans == ["mpc_expect", "helper", "TWICE", "g_count", "names"]       # 先原型、再宏、再全局，各自保持原来的顺序

    def test_whatever_the_mutant_still_uses_stays_in_its_original_order(self):
        code = "int f(int a) { return helper(TWICE(a)) + g_count + mpc_expect(0, 0); }"
        pruned, orphans = prune_context(deps(**self.FULL), code)
        assert pruned.decls == self.FULL["decls"]
        assert pruned.macros == ["#define MAXN 10", "#define TWICE(x) ((x) * 2)"]       # MAXN 靠类型定义保留
        assert pruned.globals == ["extern int g_count;"] and orphans == ["names"]

    def test_types_and_includes_are_never_pruned(self):
        pruned, _ = prune_context(deps(**self.FULL), "void f(void) { }")
        assert pruned.types == self.FULL["types"] and pruned.includes == ["#include <stddef.h>"]

    def test_reachability_goes_through_the_bodies_of_kept_entries(self):
        d = deps(macros=["#define OUTER(x) INNER((x) + 1)", "#define INNER(y) ((y) * 3)", "#define UNUSED 7"],
                 decls=["int hook(int v);"], globals=["extern int g_table[4];"])
        pruned, orphans = prune_context(d, "int f(int a) { return OUTER(a); }")
        assert pruned.macros == d.macros[:2] and orphans == ["hook", "UNUSED", "g_table"]
        # 缺陷把 OUTER 的使用删了：它和它带进来的 INNER 一起被裁
        pruned, orphans = prune_context(d, "int f(int a) { return a; }")
        assert pruned.macros == [] and set(orphans) == {"OUTER", "INNER", "UNUSED", "hook", "g_table"}

    def test_names_in_strings_and_comments_do_not_keep_an_entry(self):
        d = deps(decls=["int helper(int x);"], macros=["#define MAXN 10"])
        pruned, orphans = prune_context(d, 'int f(void) { return puts("helper MAXN"); /* helper, MAXN */ }')
        assert pruned.decls == [] and pruned.macros == [] and orphans == ["helper", "MAXN"]

    def test_function_pointer_and_array_globals_are_named_correctly(self):
        d = deps(globals=["extern void (*handler)(int);", "extern const char *names[12];", "extern int total;"],
                 decls=["void __attribute__((noreturn)) die(int code);", "static te_expr *new_expr(const int type, const te_expr *parameters[]);"])
        pruned, orphans = prune_context(d, "int f(void) { handler(1); die(2); return total; }")
        assert pruned.globals == [d.globals[0], d.globals[2]] and pruned.decls == [d.decls[0]]
        assert orphans == ["new_expr", "names"]

    def test_entries_whose_name_cannot_be_read_are_kept(self):
        d = deps(decls=["???"], macros=["#pragma once"], globals=[";"])
        pruned, orphans = prune_context(d, "int f(void) { return 0; }")
        assert (pruned.decls, pruned.macros, pruned.globals, orphans) == (d.decls, d.macros, d.globals, [])

    def test_nothing_to_prune_returns_an_equal_context_that_does_not_alias_the_input(self):
        d = deps(**self.FULL)
        code = "int f(int a) { return helper(a) + TWICE(a) + mpc_expect(0, 0) + g_count + names[0][0]; }"
        pruned, orphans = prune_context(d, code)
        assert pruned == d and orphans == [] and pruned.decls is not d.decls and pruned.types is not d.types

    def test_the_pruned_context_is_what_the_prompt_renders(self):
        d = deps(**self.FULL)
        pruned, _ = prune_context(d, "int f(int a) { return a; }")
        task = make_repair_task(context=pruned, mutant_code="int stack_push(intstack_t *s, int v)\n{\n    return 0;\n}")
        user = user_of(task)
        assert "mpc_expect" not in user and "helper" not in user and "g_count" not in user
        assert "typedef struct" in user and "#include <stddef.h>" in user
