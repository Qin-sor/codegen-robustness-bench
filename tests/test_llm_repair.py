"""bench.llm.MockLLM 在缺陷修复任务上的行为，以及它们与 similarity.normalized_equal（unchanged / restored 的判定）的衔接。

约定见 docs/设计-缺陷修复任务.md §8：修复任务靠 ``meta["task"] == "repair"`` 区分；oracle 返回该级的原函数，unchanged
原样返回 ``meta["mutant_code"]``（模型什么也没改），reformat 返回换了排版、加了注释的原函数，touched 返回加了一条无害语句的变异体，
broken / wrong 与生成任务相同，empty 没有代码块。normalized_equal 本身的测试在 tests/test_similarity.py。
全部离线、不需要 docker：回复只经过 postprocess 的代码块抽取与签名比对（libclang 解析），不编译、不运行。
"""
from __future__ import annotations

import copy

import pytest

from bench import llm as L
from bench.extract import FunctionTarget, extract_functions
from bench.llm import LLMError, MockLLM, MockLLMError, chat_with_meta, make_llm
from bench.mutate import strip_function_comments
from bench.postprocess import extract_code, process_reply
from bench.prompt import RepairTask, build_repair_messages
from bench.similarity import normalized_equal
from tests.helpers import MINI, mini_spec

MESSAGES = [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "Fix the bug."}]
SIG = "int calc_gcd(int a, int b)"
ORIGINAL = "int calc_gcd(int a, int b)\n{\n    return b == 0 ? a : calc_gcd(b, a % b);\n}"
MUTANT = "int calc_gcd(int a, int b)\n{\n    return b != 0 ? a : calc_gcd(b, a % b);\n}"      # ROR：== 改成 !=
TASK_ID = "mini:calc.c:calc_gcd@L1#repair"


def repair_meta(**kw) -> dict:
    """流水线给修复任务的 meta：生成任务的那几项，加上 task / mutant_code / operator。关键字参数覆盖默认值，传 ``...`` 表示去掉这一项。"""
    meta = {"task_id": TASK_ID, "target_id": "mini:calc.c:calc_gcd", "level": "L1", "obf_signature": SIG,
            "purpose": "generate", "sample_idx": 0, "task": "repair", "mutant_code": MUTANT, "operator": "ROR"}
    meta.update(kw)
    return {k: v for k, v in meta.items() if v is not ...}


def fenced(code: str) -> str:
    return f"```c\n{code}\n```"


def code_of(reply_text: str) -> str:
    """回复里 ```c 代码块的内容（不经过任何规整）。"""
    return reply_text.split("```c\n", 1)[1].rsplit("\n```", 1)[0]


def flags(text: str, name: str = "calc_gcd") -> tuple[bool, bool]:
    """回复 → (unchanged, restored)：抽出代码，分别和变异体、原函数比。"""
    code = extract_code(text, name)
    return normalized_equal(code, MUTANT), normalized_equal(code, ORIGINAL)


# ---------------------------------------------------------------------------
# 模式清单
# ---------------------------------------------------------------------------
class TestModeRegistry:
    def test_repair_only_modes_are_appended_to_the_old_modes(self):
        assert L.MOCK_MODES == ("oracle", "broken", "wrong", "empty", "unchanged", "reformat", "touched")

    @pytest.mark.parametrize("mode", ["unchanged", "reformat", "touched"])
    def test_make_llm_builds_the_repair_only_modes(self, mode):
        m = make_llm(f"mock-{mode}")
        assert isinstance(m, MockLLM)
        assert (m.mode, m.model, m.fingerprint()) == (mode, f"mock-{mode}", f"mock:{mode}")

    def test_error_messages_list_unchanged(self):
        with pytest.raises(ValueError, match="unchanged"):
            MockLLM("cheerful")
        with pytest.raises(ValueError, match="unchanged"):
            make_llm("mock-nonsense")


# ---------------------------------------------------------------------------
# unchanged：原样返回变异体
# ---------------------------------------------------------------------------
class TestUnchanged:
    def test_returns_the_mutant_verbatim_in_a_c_block(self):
        r = MockLLM("unchanged").chat(MESSAGES, meta=repair_meta())
        assert r.text == fenced(MUTANT)
        assert (r.model, r.cached, r.finish_reason) == ("mock-unchanged", False, "stop")
        assert r.prompt_tokens > 0 and r.completion_tokens > 0
        assert r.raw["mock"]["mode"] == "unchanged"

    def test_extract_code_gets_the_mutant_back(self):
        text = MockLLM("unchanged").chat(MESSAGES, meta=repair_meta()).text
        assert extract_code(text, "calc_gcd") == MUTANT
        assert flags(text) == (True, False)

    def test_surrounding_blank_lines_are_trimmed_like_in_the_other_modes(self):
        padded = MockLLM("unchanged").chat(MESSAGES, meta=repair_meta(mutant_code="\n\n" + MUTANT + "\n\n")).text
        assert padded == fenced(MUTANT)

    def test_needs_neither_an_oracle_nor_a_signature(self):
        meta = repair_meta(obf_signature=..., task_id=...)
        assert "obf_signature" not in meta and "task_id" not in meta
        assert MockLLM("unchanged").chat(MESSAGES, meta=meta).text == fenced(MUTANT)

    def test_ignores_the_oracle_entry_of_the_same_task(self):
        llm = MockLLM("unchanged", {TASK_ID: ORIGINAL})
        assert llm.chat(MESSAGES, meta=repair_meta()).text == fenced(MUTANT)

    @pytest.mark.parametrize("purpose", ["generate", "repair", ...])
    def test_the_purpose_label_does_not_matter(self, purpose):
        assert MockLLM("unchanged").chat(MESSAGES, meta=repair_meta(purpose=purpose)).text == fenced(MUTANT)

    @pytest.mark.parametrize(
        "meta",
        [
            None,
            {},
            {"task_id": TASK_ID},
            {"task": "generate", "mutant_code": MUTANT},
            {"task": None, "mutant_code": MUTANT},
            {"task_id": TASK_ID, "mutant_code": MUTANT},                    # 没有 task 字段：不猜
            {"task_id": TASK_ID, "purpose": "generate", "obf_signature": SIG},
        ],
    )
    def test_refuses_anything_that_is_not_a_repair_task(self, meta):
        with pytest.raises(MockLLMError, match="repair"):
            MockLLM("unchanged").chat(MESSAGES, meta=meta)

    @pytest.mark.parametrize("bad", [..., "", "  \n\t ", None, 5, ["int f(void) { return 0; }"]])
    def test_refuses_a_repair_task_without_a_usable_mutant(self, bad):
        with pytest.raises(MockLLMError, match="mutant_code"):
            MockLLM("unchanged").chat(MESSAGES, meta=repair_meta(mutant_code=bad))

    def test_the_failure_is_an_llm_error(self):
        with pytest.raises(LLMError):
            MockLLM("unchanged").chat(MESSAGES, meta={"task_id": "x"})

    def test_describe_purpose_still_returns_the_canned_spec(self):
        r = MockLLM("unchanged").chat(MESSAGES, meta={"purpose": "describe", "target_id": "x"})
        assert r.text == L.MOCK_DESCRIPTION

    def test_works_through_chat_with_meta(self):
        r = chat_with_meta(MockLLM("unchanged"), MESSAGES, temperature=0.0, max_tokens=512, seed=None, meta=repair_meta())
        assert r.text == fenced(MUTANT)

    def test_is_deterministic_and_does_not_modify_the_meta(self):
        llm = MockLLM("unchanged")
        meta = repair_meta()
        snapshot = copy.deepcopy(meta)
        first = llm.chat(MESSAGES, meta=meta).text
        assert meta == snapshot
        assert llm.chat(MESSAGES, meta=meta).text == first

    def test_calls_record_the_repair_meta(self):
        llm = MockLLM("unchanged")
        llm.chat(MESSAGES, meta=repair_meta())
        assert llm.calls[-1]["meta"]["task"] == "repair" and llm.calls[-1]["meta"]["mutant_code"] == MUTANT


# ---------------------------------------------------------------------------
# reformat / touched：检验 unchanged / restored 的判定没有被排版或无关改动骗过
# ---------------------------------------------------------------------------
class TestReformatAndTouched:
    def test_reformat_is_the_original_with_other_layout_and_a_comment(self):
        text = MockLLM("reformat", {TASK_ID: ORIGINAL}).chat(MESSAGES, meta=repair_meta()).text
        code = code_of(text)
        assert code != ORIGINAL and "/*" in code and "\t" in code
        assert extract_code(text, "calc_gcd") == code
        assert flags(text) == (False, True)                      # 规范化比较认它是逐字还原

    def test_reformat_needs_the_oracle_and_a_repair_task(self):
        with pytest.raises(MockLLMError, match="TASK_ID|calc_gcd"):
            MockLLM("reformat").chat(MESSAGES, meta=repair_meta())
        with pytest.raises(MockLLMError, match="repair"):
            MockLLM("reformat", {TASK_ID: ORIGINAL}).chat(MESSAGES, meta=repair_meta(task="generate"))

    def test_reformat_keeps_every_token_of_the_original(self):
        from bench.similarity import exact_tokens
        code = L.mock_reformat_code(ORIGINAL)
        assert exact_tokens(code) == exact_tokens(ORIGINAL)

    def test_touched_adds_a_harmless_statement_to_the_mutant(self):
        text = MockLLM("touched").chat(MESSAGES, meta=repair_meta()).text
        code = code_of(text)
        assert "(void)0;" in code and code != MUTANT
        assert extract_code(text, "calc_gcd") == code
        assert flags(text) == (False, False)                     # 既不是 unchanged 也不是 restored：缺陷还在，别处被碰过

    def test_touched_refuses_anything_that_is_not_a_repair_task(self):
        with pytest.raises(MockLLMError, match="repair"):
            MockLLM("touched").chat(MESSAGES, meta=repair_meta(task="generate"))
        with pytest.raises(MockLLMError, match="mutant_code"):
            MockLLM("touched").chat(MESSAGES, meta=repair_meta(mutant_code=...))


# ---------------------------------------------------------------------------
# oracle：返回该级原函数
# ---------------------------------------------------------------------------
class TestOracleForRepairTasks:
    def test_returns_the_original_function_of_the_level(self):
        r = MockLLM("oracle", {TASK_ID: ORIGINAL}).chat(MESSAGES, meta=repair_meta())
        assert r.text == fenced(ORIGINAL)
        assert extract_code(r.text, "calc_gcd") == ORIGINAL
        assert flags(r.text) == (False, True)

    def test_levels_are_told_apart_by_the_repair_task_id(self):
        llm = MockLLM("oracle", {"x@L0#repair": ORIGINAL, "x@L1#repair": ORIGINAL.replace("calc_gcd", "f_7")})
        assert "f_7" in llm.chat(MESSAGES, meta=repair_meta(task_id="x@L1#repair")).text
        assert "f_7" not in llm.chat(MESSAGES, meta=repair_meta(task_id="x@L0#repair")).text

    def test_a_missing_task_is_a_loud_error(self):
        with pytest.raises(MockLLMError, match="other@L1"):
            MockLLM("oracle", {TASK_ID: ORIGINAL}).chat(MESSAGES, meta=repair_meta(task_id="other@L1"))
        with pytest.raises(MockLLMError, match="task_id"):
            MockLLM("oracle", {TASK_ID: ORIGINAL}).chat(MESSAGES, meta=repair_meta(task_id=...))

    def test_does_not_fall_back_to_the_mutant(self):
        """oracle 里没有这个任务时宁可报错，不能悄悄把变异体当成答案（那样 oracle 会变成 unchanged，100% 通过的断言形同虚设）。"""
        with pytest.raises(MockLLMError):
            MockLLM("oracle").chat(MESSAGES, meta=repair_meta())

    def test_one_instance_serves_generation_and_repair_tasks(self):
        llm = MockLLM("oracle", {"g@L1": ORIGINAL, TASK_ID: ORIGINAL})
        gen = llm.chat(MESSAGES, meta={"task_id": "g@L1", "level": "L1", "obf_signature": SIG, "purpose": "generate"})
        assert gen.text == llm.chat(MESSAGES, meta=repair_meta()).text == fenced(ORIGINAL)


# ---------------------------------------------------------------------------
# broken / wrong / empty：修复任务上的回复与生成任务一样
# ---------------------------------------------------------------------------
class TestBrokenWrongEmptyForRepairTasks:
    def test_broken_has_the_right_signature_and_a_syntax_error(self):
        text = MockLLM("broken").chat(MESSAGES, meta=repair_meta()).text
        code = code_of(text)
        assert code == f"{SIG}\n{{\n    int bench_tmp_ = 0\n    return (bench_tmp_;\n}}"
        assert code.count("(") != code.count(")") and code.count("{") == code.count("}")
        assert extract_code(text, "calc_gcd") == code
        assert flags(text) == (False, False)

    def test_wrong_is_the_zero_stub_with_the_right_signature(self):
        text = MockLLM("wrong").chat(MESSAGES, meta=repair_meta()).text
        assert code_of(text) == f"{SIG}\n{{\n    int bench_zero_ = {{0}};\n    return bench_zero_;\n}}"
        assert extract_code(text, "calc_gcd") == code_of(text)
        assert flags(text) == (False, False)

    def test_wrong_gives_an_empty_body_for_void_functions(self):
        mutant = "void bump(int *p)\n{\n    *p += 2;\n}"
        text = MockLLM("wrong").chat(MESSAGES, meta=repair_meta(obf_signature="void bump(int *p)", mutant_code=mutant)).text
        assert code_of(text) == "void bump(int *p)\n{\n}"

    def test_empty_has_no_code_block(self):
        text = MockLLM("empty").chat(MESSAGES, meta=repair_meta()).text
        assert text and "```" not in text
        assert extract_code(text, "calc_gcd") is None
        assert flags(text) == (False, False)

    @pytest.mark.parametrize("mode, stub", [("broken", "int bench_tmp_ = 0"), ("wrong", "int bench_zero_ = {0};")])
    def test_signature_comes_from_meta_then_oracle_then_mutant_then_placeholder(self, mode, stub):
        mutant = "int sig_from_mutant(int a, int b)\n{\n    return a;\n}"
        oracle = {TASK_ID: "int sig_from_oracle(int a, int b)\n{\n    return b;\n}"}
        full = repair_meta(obf_signature="int sig_from_meta(int a, int b)", mutant_code=mutant)
        no_sig = repair_meta(obf_signature=..., mutant_code=mutant)
        chain = [
            (MockLLM(mode, oracle), full, "int sig_from_meta(int a, int b)"),
            (MockLLM(mode, oracle), no_sig, "int sig_from_oracle(int a, int b)"),
            (MockLLM(mode), no_sig, "int sig_from_mutant(int a, int b)"),
            (MockLLM(mode), repair_meta(obf_signature=..., mutant_code=...), f"int mock_{mode}(void)"),
        ]
        for llm, meta, header in chain:
            code = code_of(llm.chat(MESSAGES, meta=meta).text)
            assert code.startswith(header + "\n{"), (header, code)
            assert stub in code


# ---------------------------------------------------------------------------
# unchanged / restored 的判定矩阵
# ---------------------------------------------------------------------------
class TestFlagMatrix:
    #: 模式 -> (候选与变异体规范化后相同: unchanged, 候选与原函数规范化后相同: restored)
    EXPECTED = {"oracle": (False, True), "unchanged": (True, False), "broken": (False, False), "wrong": (False, False),
                "empty": (False, False), "reformat": (False, True), "touched": (False, False)}

    @pytest.mark.parametrize("mode", sorted(EXPECTED))
    def test_flags_per_mode(self, mode):
        llm = MockLLM(mode, {TASK_ID: ORIGINAL})
        assert flags(llm.chat(MESSAGES, meta=repair_meta()).text) == self.EXPECTED[mode]

    def test_every_mode_has_a_row(self):
        assert sorted(self.EXPECTED) == sorted(L.MOCK_MODES)

    def test_the_mutant_really_differs_from_the_original(self):
        assert not normalized_equal(MUTANT, ORIGINAL) and not normalized_equal(ORIGINAL, MUTANT)


# ---------------------------------------------------------------------------
# mini 夹具上的全流程：RepairTask -> build_repair_messages -> MockLLM -> process_reply -> unchanged / restored
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def mini_targets() -> dict[str, FunctionTarget]:
    return {t.name: t for t in extract_functions(mini_spec(), MINI)}


def source_of(t: FunctionTarget) -> str:
    """目标函数在源文件里的原始定义（按字节偏移切）。"""
    return (MINI / t.file).read_bytes()[t.start : t.end].decode("utf-8")


def inject_defect(code: str) -> str:
    """在函数体末尾加一条空语句：不是真缺陷，只保证变异体与原函数不同、签名不变；这里测的是各环节的衔接。"""
    head, brace, tail = code.rpartition("}")
    return head + "    (void)0;\n" + brace + tail


def repair_task_of(t: FunctionTarget) -> RepairTask:
    return RepairTask(
        task_id=f"{t.target_id}@L0#repair", target_id=t.target_id, corpus="mini", level="L0", obf_name=t.name,
        obf_signature=t.signature, context=t.deps, includes=[], language_std="c99",
        mutant_code=strip_function_comments(inject_defect(source_of(t))), operator="DST")


def meta_of(task: RepairTask) -> dict:
    return {"task_id": task.task_id, "target_id": task.target_id, "level": task.level, "obf_signature": task.obf_signature,
            "purpose": "generate", "sample_idx": 0, "task": "repair", "mutant_code": task.mutant_code,
            "operator": task.operator}


class TestRepairLoopOnMiniTargets:
    @pytest.fixture(scope="class")
    def tasks(self, mini_targets):
        return [repair_task_of(t) for t in mini_targets.values()]

    @pytest.fixture(scope="class")
    def oracle(self, mini_targets):
        return {repair_task_of(t).task_id: source_of(t) for t in mini_targets.values()}

    EXPECTED = {"oracle": (False, True), "unchanged": (True, False), "broken": (False, False), "wrong": (False, False),
                "reformat": (False, True), "touched": (False, False)}

    def test_the_fixture_is_big_enough_and_every_defect_differs_from_its_original(self, tasks, oracle):
        assert len(tasks) >= 30
        assert all(not normalized_equal(t.mutant_code, oracle[t.task_id]) for t in tasks)

    @pytest.mark.parametrize("mode", sorted(EXPECTED))
    def test_every_mode_gives_a_clean_candidate_for_every_target(self, tasks, oracle, mode):
        llm = MockLLM(mode, oracle)
        dirty = {}
        for task in tasks:
            reply = llm.chat(build_repair_messages(task), temperature=0.0, max_tokens=1024, meta=meta_of(task))
            cand = process_reply(reply.text, task)
            if cand.code is None or cand.issues:
                dirty[task.obf_name] = cand.issues
        assert dirty == {}
        assert len(llm.calls) == len(tasks)

    @pytest.mark.parametrize("mode", sorted(EXPECTED))
    def test_unchanged_and_restored_flags_for_every_target(self, tasks, oracle, mode):
        llm = MockLLM(mode, oracle)
        wrong = {}
        for task in tasks:
            reply = llm.chat(build_repair_messages(task), meta=meta_of(task))
            cand = process_reply(reply.text, task)
            got = (normalized_equal(cand.code, task.mutant_code), normalized_equal(cand.code, oracle[task.task_id]))
            if got != self.EXPECTED[mode]:
                wrong[task.obf_name] = got
        assert wrong == {}

    def test_the_candidates_are_the_expected_texts(self, tasks, oracle):
        def candidate(mode: str, task: RepairTask) -> str | None:
            reply = MockLLM(mode, oracle).chat(build_repair_messages(task), meta=meta_of(task))
            return process_reply(reply.text, task).code

        for task in tasks:
            assert candidate("oracle", task) == oracle[task.task_id].strip()
            assert candidate("unchanged", task) == task.mutant_code.strip()
            assert candidate("wrong", task).startswith(task.obf_signature)

    def test_the_prompt_carries_the_mutant_and_not_the_original(self, tasks, oracle):
        for task in tasks:
            user = build_repair_messages(task)[1]["content"]
            assert task.mutant_code in user and "(void)0;" in user
            assert oracle[task.task_id] not in user

    def test_process_reply_judges_includes_of_a_repair_task_like_those_of_a_generation_task(self, mini_targets):
        t = mini_targets["sum_array"]
        assert "#include <stdlib.h>" in t.deps.includes
        task = repair_task_of(t)
        task.includes = ["#include <stddef.h>", "#include <math.h>"]      # prompt 里告诉模型已包含的头文件
        reply = f"```c\n#include <math.h>\n#include <stdlib.h>\n{task.mutant_code}\n```"
        cand = process_reply(reply, task)
        assert cand.issues == ["added_include"]                           # stdlib.h 不在 task.includes 里，算新增
        assert normalized_equal(cand.code, task.mutant_code)             # 删掉 #include 行之后就是原样的变异体
