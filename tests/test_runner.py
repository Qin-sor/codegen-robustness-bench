"""bench.runner 的测试。

* 不需要 docker 的部分：splice 的字节偏移语义、gcc 诊断解析、阶段与信号的判定（用假沙箱喂预设输出）；
* 需要 docker 的部分：mini 夹具上真实地替换、编译、跑测试，覆盖 pass / 语法错误 / 未声明 / 类型错误 /
  其它编译错误 / 链接错误 / 测试失败 / 崩溃 / 超时，并把结果交给 attribute 归因。

mini 的编译与运行只走 docker 沙箱（--network none、--cap-drop ALL 等），宿主机不直接 make。
临时文件放在项目的 tmp/ 下，测试结束即清理。
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from bench import corpus
from bench.attribute import attribute
from bench.corpus import CorpusSpec
from bench.extract import FunctionTarget, extract_file, extract_functions
from bench.runner import (
    CFLAGS_INJECTION,
    EvalResult,
    SandboxError,
    SpliceError,
    evaluate,
    parse_text_diagnostics,
    read_function,
    run_project,
    split_gcc_json,
    splice,
    syntax_check_command,
    syntax_flags,
)
from bench.postprocess import Candidate
from bench.sandbox import DockerSandbox, RunResult
from tests.helpers import MINI, needs_docker


# ----------------------------------------------------------------------
# 公共夹具
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def mini_targets(mini_spec) -> dict[str, FunctionTarget]:
    return {t.name: t for t in extract_functions(mini_spec, MINI)}


@pytest.fixture(scope="module")
def sandbox() -> DockerSandbox:
    return DockerSandbox()


def with_body(target: FunctionTarget, body: str, root: Path = MINI) -> str:
    """原函数头（逐字节照抄）+ 新函数体，得到完整的候选定义。"""
    data = (root / target.file).read_bytes()
    return data[target.start:target.body_start].decode("utf-8") + body


def run_eval(spec, target, code, sandbox, scratch, sample_idx=0, root: Path = MINI) -> EvalResult:
    return evaluate(spec, root, target, code, sandbox, model="mock", task_id=f"{target.target_id}@L0",
                    sample_idx=sample_idx, work_root=scratch / "work")


def classify(ev: EvalResult, code: str | None) -> str:
    return attribute(ev, Candidate(code=code, issues=[]))[0]


class FakeSandbox:
    """按顺序返回预设的 RunResult，并记录每次调用的命令。"""

    def __init__(self, *results: RunResult) -> None:
        self.results = list(results)
        self.calls: list[tuple[str, int | None]] = []

    def run(self, workdir, cmd, env=None, timeout_s=None) -> RunResult:
        assert Path(workdir).is_dir()
        self.calls.append((cmd, timeout_s))
        return self.results.pop(0)


def rr(rc=0, stdout="", stderr="", timed_out=False, signal=None, duration_s=0.1, **extra) -> RunResult:
    if signal is None and 128 < rc <= 192:
        signal = rc - 128
    return RunResult(rc=rc, stdout=stdout, stderr=stderr, duration_s=duration_s, timed_out=timed_out, signal=signal,
                     **extra)


def fake_spec(**over) -> CorpusSpec:
    d = dict(name="fake", cflags=["-std=c99", "-I."], build_cmd="make test", test_cmd=None,
             pass_criteria={"exit_code": 0}, timeout_s=33)
    d.update(over)
    return CorpusSpec(**d)


def json_diag(message: str, kind: str = "error", file: str = "calc.c", line: int = 5, col: int = 3) -> str:
    return json.dumps([{"kind": kind, "message": message, "children": [], "column-origin": 1,
                        "locations": [{"caret": {"file": file, "line": line, "column": col}}]}])


# ----------------------------------------------------------------------
# splice
# ----------------------------------------------------------------------
class TestSplice:
    def test_splice_error_is_an_internal_error_not_a_usage_error(self):
        # 以前继承 ValueError，cli.main 把它当用法错误报「错误：…」并返回 2；它是内部错误，不该被当成用法问题
        assert issubclass(SpliceError, RuntimeError) and not issubclass(SpliceError, ValueError)

    def test_read_function_refuses_offsets_that_do_not_belong_to_the_source(self, mini_targets, scratch):
        t = mini_targets["calc_gcd"]
        other = scratch / "other"
        other.mkdir()
        (other / t.file).write_text("int unrelated(void) { return 0; }\n" * 40)
        with pytest.raises(SpliceError, match="函数名"):
            read_function(other, t)
        assert t.name in read_function(MINI, t)

    def test_splicing_original_text_leaves_every_file_unchanged(self, mini_targets, scratch):
        for name in ("calc_gcd", "point_add", "str_dup_upper", "array_reverse", "sum_array", "stack_pop"):
            t = mini_targets[name]
            dest = splice(MINI, t, read_function(MINI, t), scratch / f"rt_{name}")
            for f in sorted(p for p in MINI.iterdir() if p.is_file()):
                assert (dest / f.name).read_bytes() == f.read_bytes(), (name, f.name)

    def test_replaces_exactly_the_byte_range(self, mini_targets, scratch):
        t = mini_targets["calc_gcd"]
        new = "int calc_gcd(int a, int b) { return 42; }"
        dest = splice(MINI, t, new, scratch / "d")
        old = (MINI / t.file).read_bytes()
        got = (dest / t.file).read_bytes()
        assert got == old[:t.start] + new.encode() + old[t.end:]
        assert got.count(b"return 42;") == 1
        # 别的文件不动
        assert (dest / "util.c").read_bytes() == (MINI / "util.c").read_bytes()

    def test_offsets_are_bytes_not_characters(self, mini_targets, scratch):
        # util.c 开头有中文注释，字节偏移与字符偏移不同
        data = (MINI / "util.c").read_bytes()
        assert len(data) > len(data.decode("utf-8"))
        t = mini_targets["sum_array"]
        new = "int sum_array(const int *a, size_t n) { return 7; }"
        dest = splice(MINI, t, new, scratch / "d")
        got = (dest / "util.c").read_bytes()
        assert got == data[:t.start] + new.encode() + data[t.end:]
        got.decode("utf-8")  # 前后的多字节字符没有被切坏

    def test_multibyte_candidate_is_written_as_utf8(self, mini_targets, scratch):
        t = mini_targets["calc_gcd"]
        new = "int calc_gcd(int a, int b) { /* 中文注释 */ return 1; }"
        dest = splice(MINI, t, new, scratch / "d")
        assert "中文注释" in (dest / "calc.c").read_text(encoding="utf-8")

    def test_does_not_modify_the_source_tree(self, mini_targets, scratch):
        before = corpus.tree_digest(MINI)
        splice(MINI, mini_targets["calc_gcd"], "int calc_gcd(int a, int b) { return 0; }", scratch / "d")
        assert corpus.tree_digest(MINI) == before

    def test_existing_dest_is_replaced(self, mini_targets, scratch):
        dest = scratch / "d"
        dest.mkdir()
        (dest / "stale.txt").write_text("old")
        splice(MINI, mini_targets["calc_gcd"], "int calc_gcd(int a, int b) { return 0; }", dest)
        assert not (dest / "stale.txt").exists() and (dest / "calc.c").is_file()

    def test_rejects_offsets_that_do_not_match_the_function(self, mini_targets, scratch):
        t = mini_targets["calc_gcd"]
        wrong_file = dataclasses.replace(t, file="util.c")
        with pytest.raises(SpliceError):
            splice(MINI, wrong_file, "x", scratch / "a")
        out_of_range = dataclasses.replace(t, end=10**9)
        with pytest.raises(SpliceError):
            splice(MINI, out_of_range, "x", scratch / "b")
        shifted = dataclasses.replace(t, start=t.start - 40, end=t.start - 1)
        with pytest.raises(SpliceError):
            splice(MINI, shifted, "x", scratch / "c")

    def test_rejects_path_escape(self, mini_targets, scratch):
        t = dataclasses.replace(mini_targets["calc_gcd"], file="../calc.c")
        with pytest.raises(SpliceError):
            splice(MINI, t, "x", scratch / "d")

    def test_read_function_returns_the_definition_text(self, mini_targets):
        text = read_function(MINI, mini_targets["calc_gcd"])
        assert text.startswith("int calc_gcd(int a, int b)") and text.rstrip().endswith("}")


# ----------------------------------------------------------------------
# gcc 诊断与命令
# ----------------------------------------------------------------------
class TestGccPlumbing:
    def test_syntax_flags_keep_only_gcc_friendly_options(self):
        flags = syntax_flags(["-std=c99", "-I.", "-DFOO=1", "-I", "src", "-O2", "-include", "x.h", "-lm",
                              "-Wall", "-isystem", "inc", "-UBAR", "-fno-builtin"])
        assert flags == ["-std=c99", "-I.", "-DFOO=1", "-I", "src", "-isystem", "inc", "-UBAR"]
        assert syntax_flags([]) == []

    def test_syntax_check_command(self, mini_spec):
        cmd = syntax_check_command(mini_spec, "calc.c")
        assert cmd == ("gcc -x c -fsyntax-only -fdiagnostics-format=json -Wimplicit-function-declaration "
                       "-std=c99 -I. calc.c")
        weird = fake_spec(cflags=["-DNAME=a b"])
        assert "'-DNAME=a b'" in syntax_check_command(weird, "src/x.c")

    def test_split_gcc_json(self):
        d = json_diag("expected ';' before '}' token")
        text = f"[]\n{d}\nsome linker line\n[1, 2, 3]\n" + '[{"kind": "error", "message": "cut off ...\n'
        diags, rest, saw = split_gcc_json(text)
        assert saw is True
        assert [x["message"] for x in diags] == ["expected ';' before '}' token"]
        assert rest.splitlines() == ["some linker line", "[1, 2, 3]"]

    def test_split_gcc_json_without_any_json(self):
        diags, rest, saw = split_gcc_json("make: *** [Makefile:9: test] Error 1\n")
        assert diags == [] and saw is False and "Error 1" in rest

    def test_parse_text_diagnostics(self):
        text = ("calc.c: In function 'calc_gcd':\n"
                "calc.c:111:11: error: expected expression before ';' token\n"
                "calc.c:111:13: warning: implicit declaration of function 'foo'\n"
                "calc.c:3: fatal error: nope.h: No such file or directory\n")
        got = parse_text_diagnostics(text)
        assert [(d["kind"], d["line"], d["column"]) for d in got] == \
            [("error", 111, 11), ("warning", 111, 13), ("fatal error", 3, None)]
        assert got[0]["message"].startswith("expected expression")


# ----------------------------------------------------------------------
# run_project / evaluate 的阶段与信号判定（假沙箱）
# ----------------------------------------------------------------------
class TestStageClassification:
    def run(self, spec, *results, tmp):
        fake = FakeSandbox(*results)
        return run_project(spec, tmp, fake), fake

    def test_pass(self, scratch):
        run, fake = self.run(fake_spec(), rr(0, "117 checks, 0 failures\n"), tmp=scratch)
        assert (run.stage, run.rc, run.timed_out, run.signal) == ("pass", 0, False, None)
        assert fake.calls[0][0] == CFLAGS_INJECTION + "make test"
        assert fake.calls[0][1] == 33          # 超时取 spec.timeout_s

    def test_cflags_injection_keeps_existing_cflags(self):
        assert '${CFLAGS:+$CFLAGS }-fdiagnostics-format=json' in CFLAGS_INJECTION

    def test_compile_error_in_json_is_build_stage(self, scratch):
        err = json_diag("expected ';' before '}' token") + "\nmake: *** [Makefile:12: mini_test] Error 1\n"
        run, _ = self.run(fake_spec(), rr(2, "gcc -o mini_test\n", err), tmp=scratch)
        assert run.stage == "build"
        assert run.diagnostics[0]["message"].startswith("expected ';'")
        assert run.diagnostics[0]["stage"] == "build" and run.diagnostics[0]["line"] == 5

    def test_link_error_is_build_stage(self, scratch):
        err = ("[]\n/usr/bin/ld: x.o: in function `f':\nx.c:(.text+0x5): undefined reference to `g'\n"
               "collect2: error: ld returned 1 exit status\nmake: *** [Makefile:12: mini_test] Error 1\n")
        run, _ = self.run(fake_spec(), rr(2, "", err), tmp=scratch)
        assert run.stage == "build"
        assert "undefined reference to `g'" in run.stderr_text

    def test_failing_test_is_test_stage(self, scratch):
        run, _ = self.run(fake_spec(), rr(2, "FAIL calc.c:12: x\n117 checks, 3 failures\n",
                                          "[]\n[]\nmake: *** [Makefile:9: test] Error 1\n"), tmp=scratch)
        assert (run.stage, run.rc, run.signal) == ("test", 2, None)

    def test_test_output_that_looks_like_a_compile_error_does_not_count(self, scratch):
        # mpc 的测试自己会在 stdout 打印 "file:1:4: error: expected ..." 这类解析器报错
        stdout = "<test>:1:4: error: expected one of 'a'\nFailed 1\n"
        run, _ = self.run(fake_spec(), rr(1, stdout, "[]\n[]\n"), tmp=scratch)
        assert run.stage == "test"

    @pytest.mark.parametrize("line, number", [
        ("make: *** [Makefile:9: test] Segmentation fault (core dumped)", 11),
        ("make: *** [Makefile:9: test] Aborted (core dumped)", 6),
        ("make[1]: *** [Makefile:9: test] Floating point exception (core dumped)", 8),
        ("make: *** [Makefile:9: test] Bus error (core dumped)", 7),
        ("bash: line 3:    13 Segmentation fault      (core dumped) ./a.out", 11),
        ("bash: line 1:    12 Aborted                 (core dumped) ./a", 6),
        ("Aborted (core dumped)\nmake: *** [Makefile:9: test] Error 134", 6),           # dash 的写法
        ("Segmentation fault\nmake: *** [Makefile:9: test] Error 139", 11),
        ("make: *** [Makefile:9: test] Error 139", 11),                                 # 只剩 128+n
    ])
    def test_signal_is_recovered_from_make_and_bash_messages(self, scratch, line, number):
        run, _ = self.run(fake_spec(), rr(2, "./mini_test\n", f"[]\n[]\n{line}\n"), tmp=scratch)
        assert run.stage == "test" and run.signal == number

    def test_program_output_mentioning_a_signal_is_not_a_crash(self, scratch):
        stdout = "Aborted by user request\nSegmentation fault handler installed\n"
        run, _ = self.run(fake_spec(), rr(1, stdout, "[]\n"), tmp=scratch)
        assert run.stage == "test" and run.signal is None
        # stderr 里整行不是状态文字、make 退出码也不是 128+n，同样不算
        stderr = "[]\nAborted by user request\nmake: *** [Makefile:9: test] Error 1\n"
        run, _ = self.run(fake_spec(), rr(2, "", stderr), tmp=scratch)
        assert run.stage == "test" and run.signal is None

    def test_sandbox_signal_wins(self, scratch):
        run, _ = self.run(fake_spec(), rr(139, "", "[]\n"), tmp=scratch)
        assert run.stage == "test" and run.signal == 11

    def test_timeout_is_test_stage_with_timed_out(self, scratch):
        run, _ = self.run(fake_spec(), rr(137, "", "[]\n", timed_out=True), tmp=scratch)
        assert (run.stage, run.timed_out, run.signal) == ("test", True, 9)

    def test_timeout_with_some_output_is_a_test_timeout(self, scratch):
        # 有输出证据（程序打印过东西）说明编译早就完成，是测试里死循环
        run, _ = self.run(fake_spec(), rr(137, "1 - Create a string\n", "", timed_out=True), tmp=scratch)
        assert run.stage == "test" and run.timed_out

    def test_timeout_without_any_evidence_that_a_compile_finished_is_a_build_timeout(self, scratch):
        # gcc 每编完一个文件会在 stderr 打一行 JSON 诊断；stdout 也没有任何输出：连第一个文件都没编完就超时了。
        # 这种样本不能算「通过编译」（stage 记 build），stub-kill 也不能把它当成「杀死」
        run, _ = self.run(fake_spec(), rr(137, "", "", timed_out=True), tmp=scratch)
        assert (run.stage, run.timed_out) == ("build", True)

    def test_output_truncated_by_the_sandbox_cannot_vouch_for_a_pass(self, scratch, caplog):
        # 靠输出文字的判据（这里是 stdout_not_regex）在被截断的文本上不可靠：FAILED 行可能落在被丢弃的中间部分
        spec = fake_spec(pass_criteria={"exit_code": 0, "stdout_not_regex": "FAILED"})
        run, _ = self.run(spec, rr(0, "lots of output\n", "[]\n", stdout_truncated=True), tmp=scratch)
        assert run.stage == "test" and "截断" in run.stderr_text
        # 只看退出码的语料不受影响
        run, _ = self.run(fake_spec(), rr(0, "lots of output\n", "[]\n", stdout_truncated=True), tmp=scratch)
        assert run.stage == "pass"

    def test_truncated_output_in_either_of_two_runs_counts_for_a_separate_test_cmd(self, scratch):
        spec = fake_spec(test_cmd="./t", pass_criteria={"exit_code": 0, "stdout_regex": "ok"})
        run, _ = self.run(spec, rr(0, "build\n", "[]\n", stdout_truncated=True), rr(0, "ok\n"), tmp=scratch)
        assert run.stage == "test"

    def test_pipe_read_errors_are_infrastructure_failures_not_sample_failures(self, scratch):
        with pytest.raises(SandboxError, match="输出不完整"):
            self.run(fake_spec(), rr(0, "x", "[]\n", io_error="stdout OSError: boom"), tmp=scratch)

    def test_exit_zero_but_pass_criteria_unmet_is_test_stage(self, scratch):
        spec = fake_spec(pass_criteria={"exit_code": 0, "stdout_regex": "ALL TESTS PASSED"})
        run, _ = self.run(spec, rr(0, "SOME TESTS FAILED\n", "[]\n"), tmp=scratch)
        assert (run.stage, run.rc) == ("test", 0)
        run, _ = self.run(spec, rr(0, "ALL TESTS PASSED\n", "[]\n"), tmp=scratch)
        assert run.stage == "pass"

    def test_warnings_alone_do_not_fail_the_build(self, scratch):
        warn = json_diag("unused variable 'x'", kind="warning")
        run, _ = self.run(fake_spec(), rr(0, "ok\n", warn), tmp=scratch)
        assert run.stage == "pass" and run.diagnostics[0]["kind"] == "warning"

    def test_text_format_diagnostics_are_used_when_cflags_are_ignored(self, scratch):
        err = "calc.c:7:5: error: 'x' undeclared (first use in this function)\nmake: *** [Makefile:12: m] Error 1\n"
        run, _ = self.run(fake_spec(), rr(2, "", err), tmp=scratch)
        assert run.stage == "build" and run.diagnostics[0]["message"].startswith("'x' undeclared")

    def test_separate_test_cmd_runs_after_a_successful_build(self, scratch):
        spec = fake_spec(build_cmd="make mini_test", test_cmd="./mini_test")
        run, fake = self.run(spec, rr(0, "gcc -o mini_test\n", "[]\n"), rr(1, "FAIL x\n", ""), tmp=scratch)
        assert run.stage == "test" and run.rc == 1
        assert fake.calls[0][0] == CFLAGS_INJECTION + "make mini_test"
        assert fake.calls[1][0] == "./mini_test"

    def test_separate_test_cmd_not_run_after_build_failure(self, scratch):
        spec = fake_spec(build_cmd="make mini_test", test_cmd="./mini_test")
        run, fake = self.run(spec, rr(2, "", json_diag("expected ';'")), tmp=scratch)
        assert run.stage == "build" and len(fake.calls) == 1

    def test_separate_test_cmd_pass_criteria_sees_both_outputs(self, scratch):
        spec = fake_spec(build_cmd="make mini_test", test_cmd="./mini_test",
                         pass_criteria={"exit_code": 0, "stdout_regex": "BUILD-MARK[\\s\\S]*TEST-MARK"})
        run, _ = self.run(spec, rr(0, "BUILD-MARK\n", "[]\n"), rr(0, "TEST-MARK\n", ""), tmp=scratch)
        assert run.stage == "pass"

    def test_docker_failure_raises_instead_of_being_blamed_on_the_candidate(self, scratch):
        with pytest.raises(SandboxError):
            self.run(fake_spec(), rr(125, "", "docker: Error response from daemon: boom"), tmp=scratch)


class TestEvaluateWithFakeSandbox:
    def test_syntax_failure_short_circuits(self, mini_spec, mini_targets, scratch):
        err = json_diag("expected ';' before '}' token", line=111, col=11)
        fake = FakeSandbox(rr(1, "", err))
        ev = evaluate(mini_spec, MINI, mini_targets["calc_gcd"], "int calc_gcd(int a, int b) { return a }",
                      fake, "m", "tid", 3, work_root=scratch / "w")
        assert len(fake.calls) == 1                         # 没有进入构建
        assert fake.calls[0][0] == syntax_check_command(mini_spec, "calc.c")
        assert fake.calls[0][0].endswith(" -std=c99 -I. calc.c")
        assert (ev.task_id, ev.model, ev.sample_idx, ev.stage, ev.rc) == ("tid", "m", 3, "syntax", 1)
        assert ev.diagnostics == [{"kind": "error", "message": "expected ';' before '}' token",
                                   "file": "calc.c", "line": 111, "column": 11, "stage": "syntax"}]
        assert not (scratch / "w").exists() or list((scratch / "w").iterdir()) == []

    def test_syntax_then_build_then_pass_uses_spec_timeout(self, mini_spec, mini_targets, scratch):
        spec = dataclasses.replace(mini_spec, timeout_s=47)
        fake = FakeSandbox(rr(0, "", "[]\n"), rr(0, "117 checks, 0 failures\n", "[]\n[]\n[]\n"))
        ev = evaluate(spec, MINI, mini_targets["calc_gcd"], read_function(MINI, mini_targets["calc_gcd"]),
                      fake, "m", "tid", 0, work_root=scratch / "w")
        assert ev.stage == "pass" and ev.rc == 0 and ev.diagnostics == []
        assert [t for _, t in fake.calls] == [47, 47]       # 语法检查（上限 60）与构建都取 spec.timeout_s
        assert "117 checks" in ev.test_stdout_tail

    def test_none_candidate_never_reaches_the_sandbox(self, mini_spec, mini_targets):
        fake = FakeSandbox()
        ev = evaluate(mini_spec, MINI, mini_targets["calc_gcd"], None, fake, "m", "tid", 0)
        assert fake.calls == [] and ev.stage == "syntax" and ev.rc == -1

    def test_warnings_from_the_syntax_stage_are_kept_and_deduplicated(self, mini_spec, mini_targets, scratch):
        warn = json_diag("unused variable 'x'", kind="warning")
        fake = FakeSandbox(rr(0, "", warn), rr(0, "ok\n", warn))
        ev = evaluate(mini_spec, MINI, mini_targets["calc_gcd"], read_function(MINI, mini_targets["calc_gcd"]),
                      fake, "m", "tid", 0, work_root=scratch / "w")
        assert ev.stage == "pass" and len(ev.diagnostics) == 1

    def test_diagnostics_are_capped_with_errors_first(self, mini_spec, mini_targets, scratch):
        many = json.dumps([{"kind": "warning", "message": f"w{i}", "locations": []} for i in range(100)]
                          + [{"kind": "error", "message": "boom", "locations": []}])
        fake = FakeSandbox(rr(1, "", many))
        ev = evaluate(mini_spec, MINI, mini_targets["calc_gcd"], "int calc_gcd(int a, int b) { }",
                      fake, "m", "tid", 0, work_root=scratch / "w")
        assert len(ev.diagnostics) == 30 and ev.diagnostics[0]["message"] == "boom"

    def test_result_is_json_serializable(self, mini_spec, mini_targets):
        fake = FakeSandbox(rr(1, "", json_diag("expected ';'")))
        ev = evaluate(mini_spec, MINI, mini_targets["calc_gcd"], "int calc_gcd(int a, int b) { return a }",
                      fake, "m", "tid", 0)
        assert json.loads(json.dumps(ev.to_dict()))["stage"] == "syntax"


# ----------------------------------------------------------------------
# mini 夹具上的真实评测（docker）
# ----------------------------------------------------------------------
@needs_docker
class TestEvaluateOnMini:
    @pytest.mark.parametrize("name", ["calc_gcd", "point_add", "str_dup_upper", "array_reverse", "sum_array"])
    def test_original_function_passes(self, mini_spec, mini_targets, sandbox, scratch, name):
        t = mini_targets[name]
        code = read_function(MINI, t)
        ev = run_eval(mini_spec, t, code, sandbox, scratch)
        assert ev.stage == "pass", ev
        assert ev.rc == 0 and not ev.timed_out and ev.signal is None
        assert "0 failures" in ev.test_stdout_tail
        assert classify(ev, code) == "pass"
        assert ev.duration_s > 0

    def test_workdir_is_removed_and_source_untouched(self, mini_spec, mini_targets, sandbox, scratch):
        before = corpus.tree_digest(MINI)
        run_eval(mini_spec, mini_targets["calc_gcd"], read_function(MINI, mini_targets["calc_gcd"]),
                 sandbox, scratch)
        assert list((scratch / "work").iterdir()) == []
        assert corpus.tree_digest(MINI) == before

    def test_keep_workdir_keeps_the_spliced_copy(self, mini_spec, mini_targets, sandbox, scratch):
        t = mini_targets["calc_gcd"]
        code = with_body(t, "{ return 42; }")
        evaluate(mini_spec, MINI, t, code, sandbox, "m", "tid", 0, work_root=scratch / "work", keep_workdir=True)
        kept = list((scratch / "work").iterdir())
        assert len(kept) == 1 and "return 42;" in (kept[0] / "calc.c").read_text(encoding="utf-8")

    def test_syntax_error_is_compile_syntax(self, mini_spec, mini_targets, sandbox, scratch):
        t = mini_targets["calc_gcd"]
        code = with_body(t, "{ if (b == 0) { return a } return calc_gcd(b, a % b); }")
        ev = run_eval(mini_spec, t, code, sandbox, scratch)
        assert ev.stage == "syntax" and ev.rc != 0
        err = [d for d in ev.diagnostics if d["kind"] == "error"][0]
        assert "expected ';'" in err["message"] and err["file"] == "calc.c" and err["stage"] == "syntax"
        assert classify(ev, code) == "compile_syntax"

    def test_undeclared_identifier_is_compile_undeclared(self, mini_spec, mini_targets, sandbox, scratch):
        t = mini_targets["calc_gcd"]
        code = with_body(t, "{ return bench_undefined_symbol + a + b; }")
        ev = run_eval(mini_spec, t, code, sandbox, scratch)
        assert ev.stage == "syntax"
        assert classify(ev, code) == "compile_undeclared"

    def test_unknown_type_is_compile_undeclared(self, mini_spec, mini_targets, sandbox, scratch):
        t = mini_targets["calc_gcd"]
        code = with_body(t, "{ bench_missing_t x = 0; return a + b + x; }")
        ev = run_eval(mini_spec, t, code, sandbox, scratch)
        assert classify(ev, code) == "compile_undeclared"

    def test_type_error_is_compile_type(self, mini_spec, mini_targets, sandbox, scratch):
        t = mini_targets["calc_gcd"]
        code = with_body(t, "{ return point_make(a, b); }")
        ev = run_eval(mini_spec, t, code, sandbox, scratch)
        assert ev.stage == "syntax"
        assert any("incompatible types" in d["message"] for d in ev.diagnostics)
        assert classify(ev, code) == "compile_type"

    def test_too_many_arguments_is_compile_type(self, mini_spec, mini_targets, sandbox, scratch):
        t = mini_targets["calc_gcd"]
        code = with_body(t, "{ return calc_clamp(a, b, 1, 2, 3); }")
        ev = run_eval(mini_spec, t, code, sandbox, scratch)
        assert classify(ev, code) == "compile_type"

    def test_other_compile_error_is_compile_other(self, mini_spec, mini_targets, sandbox, scratch):
        t = mini_targets["calc_gcd"]
        code = with_body(t, "{ goto nowhere; return a + b; }")
        ev = run_eval(mini_spec, t, code, sandbox, scratch)
        assert ev.stage == "syntax"
        assert classify(ev, code) == "compile_other"

    def test_undefined_function_is_link_error(self, mini_spec, mini_targets, sandbox, scratch):
        t = mini_targets["calc_gcd"]
        code = with_body(t, "{ extern int bench_missing_helper(int); return bench_missing_helper(a) + b; }")
        ev = run_eval(mini_spec, t, code, sandbox, scratch)
        assert ev.stage == "build", ev            # 单文件语法检查通过，到链接才失败
        assert "undefined reference" in ev.test_stdout_tail
        assert classify(ev, code) == "link_error"

    def test_calling_an_undeclared_function_is_compile_undeclared(self, mini_spec, mini_targets, sandbox, scratch):
        # gcc 13 只把隐式函数声明当 warning：单文件语法检查能过，死在链接上；归因要把它认作「用了没声明的东西」
        t = mini_targets["calc_gcd"]
        code = with_body(t, "{ return bench_missing_helper(a) + b; }")
        ev = run_eval(mini_spec, t, code, sandbox, scratch)
        assert ev.stage == "build", ev
        assert "undefined reference to `bench_missing_helper'" in ev.test_stdout_tail
        warn = [d for d in ev.diagnostics if d["kind"] == "warning"]
        assert any("implicit declaration of function 'bench_missing_helper'" in d["message"] for d in warn)
        cat, detail = attribute(ev, Candidate(code=code, issues=[]))
        assert cat == "compile_undeclared" and "bench_missing_helper" in detail and "link" in detail

    def test_zero_stub_is_test_fail(self, mini_spec, mini_targets, sandbox, scratch):
        t = mini_targets["calc_gcd"]
        code = with_body(t, "{ return 0; }")
        ev = run_eval(mini_spec, t, code, sandbox, scratch)
        assert ev.stage == "test" and ev.rc != 0 and ev.signal is None and not ev.timed_out
        assert "FAIL" in ev.test_stdout_tail
        assert classify(ev, code) == "test_fail"

    def test_abort_is_runtime_crash(self, mini_spec, mini_targets, sandbox, scratch):
        t = mini_targets["calc_gcd"]
        code = with_body(t, "{ __builtin_abort(); return 0; }")
        ev = run_eval(mini_spec, t, code, sandbox, scratch)
        assert ev.stage == "test" and ev.signal == 6, ev       # make 下信号靠 "Aborted" 文字补出
        assert classify(ev, code) == "runtime_crash"

    def test_null_dereference_is_runtime_crash(self, mini_spec, mini_targets, sandbox, scratch):
        t = mini_targets["calc_gcd"]
        code = with_body(t, "{ volatile int *p = 0; return *p + a + b; }")
        ev = run_eval(mini_spec, t, code, sandbox, scratch)
        assert ev.stage == "test" and ev.signal == 11, ev
        assert classify(ev, code) == "runtime_crash"

    def test_division_by_zero_is_runtime_crash(self, mini_spec, mini_targets, sandbox, scratch):
        t = mini_targets["calc_gcd"]
        code = with_body(t, "{ volatile int z = 0; return (a + b) / z; }")
        ev = run_eval(mini_spec, t, code, sandbox, scratch)
        assert ev.stage == "test" and ev.signal == 8, ev
        assert classify(ev, code) == "runtime_crash"

    def test_infinite_loop_is_runtime_timeout(self, mini_spec, mini_targets, sandbox, scratch):
        spec = dataclasses.replace(mini_spec, timeout_s=6)      # 正常构建 + 测试不到 2 s
        t = mini_targets["calc_gcd"]
        code = with_body(t, "{ while (1) { } return 0; }")
        ev = run_eval(spec, t, code, sandbox, scratch)
        assert ev.stage == "test" and ev.timed_out is True, ev
        assert 5 <= ev.duration_s < 30
        assert classify(ev, code) == "runtime_timeout"

    def test_exit_zero_with_unmet_pass_criteria_is_test_fail(self, mini_spec, mini_targets, sandbox, scratch):
        spec = dataclasses.replace(mini_spec, pass_criteria={"exit_code": 0, "stdout_regex": "THIS NEVER APPEARS"})
        t = mini_targets["calc_gcd"]
        code = read_function(MINI, t)
        ev = run_eval(spec, t, code, sandbox, scratch)
        assert ev.stage == "test" and ev.rc == 0
        assert classify(ev, code) == "test_fail"

    def test_separate_build_and_test_commands(self, mini_spec, mini_targets, sandbox, scratch):
        spec = dataclasses.replace(mini_spec, build_cmd="make mini_test", test_cmd="./mini_test")
        t = mini_targets["calc_gcd"]
        ok = run_eval(spec, t, read_function(MINI, t), sandbox, scratch)
        assert ok.stage == "pass"
        stub = with_body(t, "{ return 0; }")
        assert run_eval(spec, t, stub, sandbox, scratch).stage == "test"
        # 单文件语法检查过得去、链接失败：阶段是 build，测试命令不会运行
        link = with_body(t, "{ extern int bench_missing_helper(int); return bench_missing_helper(a) + b; }")
        ev = run_eval(spec, t, link, sandbox, scratch)
        assert ev.stage == "build" and classify(ev, link) == "link_error"
        crash = with_body(t, "{ __builtin_abort(); return 0; }")
        ev = run_eval(spec, t, crash, sandbox, scratch)
        assert ev.stage == "test" and ev.signal == 6          # 直接运行时退出码 134 → 沙箱换算出信号

    def test_extract_fail_never_needs_evaluation(self, mini_spec, mini_targets):
        ev = evaluate(mini_spec, MINI, mini_targets["calc_gcd"], None, DockerSandbox(), "m", "tid", 0)
        assert attribute(ev, Candidate(code=None, issues=["no_code_block"]))[0] == "extract_fail"


@needs_docker
class TestHeaderOnlyTarget:
    """jsmn 这类头文件实现的库：目标函数在 .h 里，语法检查要按 C 源码而不是预编译头处理。"""

    LIB_H = """#ifndef LIB_H
#define LIB_H

static int lib_double(int x)
{
    int r = x;

    r += x;
    return r;
}

#endif
"""
    TEST_C = """#include <stdio.h>
#include "lib.h"

int main(void)
{
    if (lib_double(21) != 42) {
        printf("FAIL lib_double\\n");
        return 1;
    }
    printf("all ok\\n");
    return 0;
}
"""

    @pytest.fixture()
    def hdr(self, scratch):
        root = scratch / "hdr"
        root.mkdir()
        (root / "lib.h").write_text(self.LIB_H)
        (root / "test.c").write_text(self.TEST_C)
        spec = CorpusSpec(name="hdr", project_files=["lib.h", "test.c"], target_files=["lib.h"],
                          cflags=["-std=c99"], timeout_s=60,
                          build_cmd="gcc -std=c99 -O0 $CFLAGS -o t test.c && ./t")
        t = next(x for x in extract_file(spec, root, "lib.h") if x.name == "lib_double")
        return spec, root, t

    def test_undeclared_call_is_detected_under_c89_too(self, hdr, sandbox, scratch):
        # -std=c89 下 gcc 默认不报隐式声明；语法检查带 -Wimplicit-function-declaration，所以结论与 c99 一致
        spec, root, t = hdr
        spec = dataclasses.replace(spec, cflags=["-std=c89"],
                                   build_cmd="gcc -std=c89 -O0 $CFLAGS -o t test.c && ./t")
        code = with_body(t, "{ return bench_missing_helper(x) + x; }", root)
        ev = run_eval(spec, t, code, sandbox, scratch, root=root)
        assert ev.stage == "build", ev
        assert classify(ev, code) == "compile_undeclared"
        assert classify(run_eval(spec, t, read_function(root, t), sandbox, scratch, root=root),
                        read_function(root, t)) == "pass"

    def test_header_target_round_trip_and_failures(self, hdr, sandbox, scratch):
        spec, root, t = hdr
        orig = read_function(root, t)
        ev = run_eval(spec, t, orig, sandbox, scratch, root=root)
        assert ev.stage == "pass", ev
        stub = with_body(t, "{ return 0; }", root)
        ev = run_eval(spec, t, stub, sandbox, scratch, root=root)
        assert ev.stage == "test" and classify(ev, stub) == "test_fail"
        bad = with_body(t, "{ return x + ; }", root)
        ev = run_eval(spec, t, bad, sandbox, scratch, root=root)
        assert ev.stage == "syntax" and ev.diagnostics[0]["file"] == "lib.h"
        assert classify(ev, bad) == "compile_syntax"
