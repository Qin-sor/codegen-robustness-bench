"""语料模块测试：9 个 spec 的 schema、文件展开、沙箱构建、覆盖率；prepare / materialize 的行为。

分三块：
1. 纯逻辑（不需要语料、docker、libclang）：CorpusSpec、校验、展开、materialize、用本地 git 仓库验证 prepare。
2. 真实语料（corpus/src，缺失时按 spec 从本地镜像复制或克隆；都不行就跳过）：schema、展开、includes 闭包、libclang 零 error。
3. 真实语料 + docker（gcc:13）：build_cmd 退出码 0、注入 --coverage 能生成 .gcda、pass_criteria 能识别被破坏的实现。

临时目录一律放在项目 tmp/ 下。
"""
from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
from pathlib import Path

import clang.cindex as ci
import pytest

from bench import clangcfg, corpus
from bench.corpus import CorpusSpec, SpecError
from bench.sandbox import DockerSandbox, RunResult
from tests.helpers import PROJECT_ROOT, REQUIRE_CORPUS, needs_docker

SPEC_NAMES = ["cjson", "inih", "jsmn", "mpc", "parson", "sds", "tiny-regex-c", "tinyexpr", "vec"]

GIT_ENV = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@example.com",
               GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@example.com", GIT_TERMINAL_PROMPT="0")


def git(*args: str, cwd: Path | None = None) -> str:
    res = subprocess.run(["git", "-c", "commit.gpgsign=false", *args], cwd=cwd, env=GIT_ENV, text=True,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return res.stdout.strip()


def make_upstream(root: Path, n_commits: int = 2) -> tuple[Path, list[str]]:
    """造一个本地上游仓库，返回（路径，各提交的 SHA）。lib.c 的内容随提交变化。"""
    repo = root / "foo"
    repo.mkdir(parents=True)
    git("init", "-q", cwd=repo)
    commits = []
    for i in range(n_commits):
        (repo / "lib.c").write_text(f"int f(void) {{ return {i}; }}\n")
        (repo / "lib.h").write_text("int f(void);\n")
        git("add", "-A", cwd=repo)
        git("commit", "-q", "-m", f"c{i}", cwd=repo)
        commits.append(git("rev-parse", "HEAD", cwd=repo))
    return repo, commits


def foo_spec(url: str, commit: str = "") -> CorpusSpec:
    return CorpusSpec(name="foo", url=url, commit=commit, license="MIT",
                      project_files=["*.c", "*.h"], target_files=["lib.c"], build_cmd="true")


# ======================================================================
# 1. 纯逻辑
# ======================================================================
def minimal_dict(**kw) -> dict:
    d = {"name": "x", "url": "u", "commit": "", "license": "MIT", "project_files": ["a.c"],
         "target_files": ["a.c"], "cflags": [], "build_cmd": "true", "test_cmd": None,
         "pass_criteria": {"exit_code": 0}, "coverage": {}, "timeout_s": 10, "notes": ""}
    d.update(kw)
    return d


class TestCorpusSpec:
    def test_defaults_and_roundtrip(self):
        spec = CorpusSpec(name="a")
        assert spec.test_cmd is None and spec.pass_criteria == {"exit_code": 0}
        assert spec.vendored == [] and spec.keep_names == [] and spec.timeout_s == 120
        assert spec.allow_words == [] and spec.exclude_targets == []
        again = CorpusSpec.from_dict(spec.to_dict())
        assert again == spec
        assert again.digest() == spec.digest()
        assert CorpusSpec(name="b").digest() != spec.digest()

    def test_digest_ignores_fields_that_do_not_affect_builds_or_tests(self):
        base = CorpusSpec(name="a", url="https://x/y.git", license="MIT", notes="n", build_cmd="make")
        same = CorpusSpec(name="a", url="https://other/y.git", license="BSD", notes="补一句备注", build_cmd="make",
                          allow_words=["expr"])
        assert base.digest() == same.digest()
        # 改了构建 / 测试 / 目标选择语义的字段，哈希必须变
        for patch in ({"commit": "1" * 40}, {"cflags": ["-O2"]}, {"build_cmd": "make all"}, {"test_cmd": "./t"},
                      {"pass_criteria": {"exit_code": 1}}, {"coverage": {"env": {"A": "b"}}}, {"timeout_s": 5},
                      {"vendored": ["x"]}, {"keep_names": ["K"]}, {"project_files": ["a.c"]},
                      {"target_files": ["a.c"]}, {"exclude_targets": ["main"]}, {"name": "b"}):
            changed = CorpusSpec.from_dict({**base.to_dict(), **patch})
            assert changed.digest() != base.digest(), patch

    def test_exclude_targets_match_by_name_or_file_and_name(self):
        spec = CorpusSpec(name="a", exclude_targets=["sdsTest", "x.c:helper"])
        assert spec.is_excluded_target("sds.c", "sdsTest")
        assert spec.is_excluded_target("x.c", "helper") and not spec.is_excluded_target("y.c", "helper")
        assert not spec.is_excluded_target("sds.c", "sdsnew")

    def test_gcda_glob_is_no_longer_a_spec_field(self):
        # 以前每个 spec 都写了 coverage.gcda_glob，却没有任何代码读它（gcov 脚本硬编码 find -name '*.gcda'）
        errs = corpus.validate_spec(CorpusSpec.from_dict(minimal_dict(coverage={"gcda_glob": "**/*.gcda"})))
        assert any("gcda_glob" in e for e in errs)

    def test_repo_dir_name_comes_from_the_url(self):
        assert CorpusSpec(name="cjson", url="https://github.com/DaveGamble/cJSON.git").repo_dir_name == "cJSON"
        assert CorpusSpec(name="vec", url="https://github.com/rxi/vec/").repo_dir_name == "vec"
        assert CorpusSpec(name="mini").repo_dir_name == "mini"

    def test_judge_refuses_output_criteria_on_truncated_output(self):
        crit = CorpusSpec(name="a", pass_criteria={"exit_code": 0, "stdout_not_regex": "FAILED"})
        assert crit.judge(0, "ok\n") == (True, False)
        # 输出被沙箱从中间截断：FAILED 行可能恰好落在被丢弃的部分，靠输出的判据不可信，一律不认通过
        assert crit.judge(0, "ok\n", truncated=True) == (False, True)
        rc_only = CorpusSpec(name="b", pass_criteria={"exit_code": 0})
        assert rc_only.judge(0, "x", truncated=True) == (True, False)       # 只看退出码的不受影响
        assert crit.judge(1, "ok\n", truncated=True) == (False, False)      # 本来就不通过，不算「因截断而拒绝」

    def test_from_dict_rejects_unknown_but_ignores_comments(self):
        with pytest.raises(SpecError, match="未知字段"):
            CorpusSpec.from_dict(minimal_dict(bogus=1))
        assert CorpusSpec.from_dict(minimal_dict(_comment="随便写")).name == "x"
        with pytest.raises(SpecError, match="name"):
            CorpusSpec.from_dict({"url": "u"})

    @pytest.mark.parametrize("patch, needle", [
        ({"name": "Bad Name"}, "name"),
        ({"commit": "abc"}, "commit"),
        ({"project_files": []}, "project_files"),
        ({"project_files": ["/etc/passwd"]}, "绝对路径"),
        ({"target_files": ["../x.c"]}, "绝对路径或 .."),
        ({"cflags": "-O2"}, "cflags"),
        ({"build_cmd": "  "}, "build_cmd"),
        ({"test_cmd": ""}, "test_cmd"),
        ({"pass_criteria": {}}, "exit_code"),
        ({"pass_criteria": {"exit_code": 0, "stdout_regex": "("}}, "正则"),
        ({"pass_criteria": {"exit_code": 0, "extra": 1}}, "未知键"),
        ({"coverage": {"env": {"CFLAGS": 1}}}, "coverage.env"),
        ({"coverage": {"oops": 1}}, "未知键"),
        ({"timeout_s": 0}, "timeout_s"),
        ({"vendored": ["../x"]}, "vendored"),
    ])
    def test_validation_catches(self, patch, needle):
        spec = CorpusSpec.from_dict(minimal_dict(**patch))
        errs = corpus.validate_spec(spec)
        assert errs and any(needle in e for e in errs), errs

    def test_valid_minimal_spec_passes_and_strict_demands_pin(self):
        spec = CorpusSpec.from_dict(minimal_dict())
        assert corpus.validate_spec(spec) == []
        strict = corpus.validate_spec(spec, strict=True)
        assert any("commit" in e for e in strict)

    def test_load_spec_file_reports_path_and_reason(self, scratch):
        bad = scratch / "x.json"
        bad.write_text("{not json")
        with pytest.raises(SpecError, match="JSON 解析失败"):
            corpus.load_spec_file(bad)
        bad.write_text(json.dumps(minimal_dict(build_cmd="")))
        with pytest.raises(SpecError, match="build_cmd"):
            corpus.load_spec_file(bad)

    def test_load_spec_checks_name_matches_filename(self, scratch, monkeypatch):
        (scratch / "foo.json").write_text(json.dumps(minimal_dict(name="bar")))
        monkeypatch.setattr(corpus, "SPEC_DIR", scratch)
        with pytest.raises(SpecError, match="不一致"):
            corpus.load_spec("foo")
        with pytest.raises(FileNotFoundError):
            corpus.load_spec("nope")
        assert corpus.list_specs() == ["foo"]
        # 传 .json 路径时直接读那个文件（名字与文件名可以不同，夹具用）
        assert corpus.load_spec(str(scratch / "foo.json")).name == "bar"

    def test_std_property(self):
        assert CorpusSpec(name="a", cflags=["-I.", "-std=c89", "-DX"]).std == "c89"
        assert CorpusSpec(name="a", cflags=["-DX"]).std is None

    def test_is_vendored(self):
        spec = CorpusSpec(name="a", vendored=["tests/unity", "third/*.c", "single.h"])
        assert spec.is_vendored("tests/unity/src/unity.c")
        assert spec.is_vendored("tests/unity")
        assert spec.is_vendored("third/x.c") and not spec.is_vendored("third/x.h")
        assert spec.is_vendored("single.h")
        assert not spec.is_vendored("tests/unity_setup.c")   # 前缀相同但不是同一个目录
        assert not spec.is_vendored("tests/other.c")

    def test_command_line_macros(self):
        spec = CorpusSpec(
            name="a", cflags=["-std=c99", "-DFROM_CFLAGS", "-I."],
            build_cmd='gcc -DA=1 -D B -UC $CFLAGS x.c -o t -Wno-DNOT_A_MACRO\nfor d in "-DD_IN_LOOP"; do :; done',
            test_cmd="./t -DE", coverage={"build_cmd": "gcc -DCOV_ONLY x.c"})
        assert spec.command_line_macros() == {"FROM_CFLAGS", "A", "B", "C", "D_IN_LOOP", "E", "COV_ONLY"}

    def test_passes(self):
        spec = CorpusSpec(name="a", pass_criteria={"exit_code": 0, "stdout_regex": r"(\d+)/\1 ok",
                                                   "stdout_not_regex": "FAIL"})
        assert spec.passes(0, "3/3 ok\n")
        assert not spec.passes(1, "3/3 ok\n")             # 退出码不对
        assert not spec.passes(0, "3/4 ok\n")             # 缺少必须出现的输出
        assert not spec.passes(0, "3/3 ok\nFAIL x\n")     # 出现了禁止的输出
        assert CorpusSpec(name="b", pass_criteria={"exit_code": 2}).passes(2, "")

    def test_coverage_command_prefers_its_own_build_cmd(self):
        spec = CorpusSpec(name="a", build_cmd="make", coverage={"env": {}})
        assert spec.coverage_command() == "make"
        spec.coverage["build_cmd"] = "make CFLAGS=--coverage"
        assert spec.coverage_command() == "make CFLAGS=--coverage"

    def test_run_command_chains_test_cmd_after_build(self):
        assert CorpusSpec(name="a", build_cmd="make").run_command() == "make"
        cmd = CorpusSpec(name="a", build_cmd="make\nmake x", test_cmd="./t").run_command()
        assert cmd == "{\nmake\nmake x\n} && {\n./t\n}"


class TestExpandAndMaterialize:
    def make_tree(self, root: Path) -> None:
        for rel in ["a.c", "sub/b.c", "sub/deep/c.c", "sub/readme.md", ".git/x.c", "sub/dir.c/inner.c"]:
            f = root / rel
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(rel)

    def test_expand_files_dedups_sorts_and_skips_dirs_and_git(self, scratch):
        self.make_tree(scratch)
        spec = CorpusSpec(name="a", project_files=["sub/**/*.c", "a.c", "*.c", "sub/*.c", "sub/readme.md"],
                          target_files=["a.c"])
        got = corpus.expand_files(spec, scratch)
        rels = [p.relative_to(scratch).as_posix() for p in got]
        assert rels == ["a.c", "sub/b.c", "sub/deep/c.c", "sub/dir.c/inner.c", "sub/readme.md"]
        assert all(p.is_absolute() == scratch.is_absolute() and p.is_file() for p in got)
        assert corpus.expand_patterns(spec.project_files, scratch) == rels
        assert [p.name for p in corpus.expand_target_files(spec, scratch)] == ["a.c"]

    def test_expand_files_missing_pattern_yields_nothing(self, scratch):
        spec = CorpusSpec(name="a", project_files=["nope/*.c"])
        assert corpus.expand_files(spec, scratch) == []

    def test_expand_patterns_can_warn_or_insist(self, scratch, caplog):
        (scratch / "a.c").write_text("int a;")
        assert corpus.expand_patterns(["a.c", "nope/*.c"], scratch) == ["a.c"]            # 默认静默
        with caplog.at_level("WARNING"):
            assert corpus.expand_patterns(["nope/*.c"], scratch, warn=True) == []
        assert "没有匹配到文件" in caplog.text
        with pytest.raises(FileNotFoundError):
            corpus.expand_patterns(["nope.c"], scratch, must_exist=True)         # 字面路径缺失才报错
        assert corpus.expand_patterns(["nope/*.c"], scratch, must_exist=True) == []

    def test_materialize_excludes_git_keeps_modes_and_symlinks(self, scratch):
        src = scratch / "src"
        self.make_tree(src)
        (src / "run.sh").write_text("#!/bin/sh\n")
        (src / "run.sh").chmod(0o755)
        (src / "link.c").symlink_to("a.c")
        out = corpus.materialize(src, scratch / "w" / "copy")
        assert out == scratch / "w" / "copy"
        assert (out / "sub/deep/c.c").read_text() == "sub/deep/c.c"
        assert not (out / ".git").exists()
        assert os.access(out / "run.sh", os.X_OK)
        assert (out / "link.c").is_symlink()
        assert (src / ".git/x.c").exists()                 # 源目录原样不动

    def test_materialize_replaces_existing_dest(self, scratch):
        src = scratch / "src"
        self.make_tree(src)
        dest = scratch / "dest"
        dest.mkdir()
        (dest / "stale.o").write_text("old build output")
        corpus.materialize(src, dest)
        assert not (dest / "stale.o").exists() and (dest / "a.c").exists()

    def test_materialize_refuses_nested_paths(self, scratch):
        src = scratch / "src"
        self.make_tree(src)
        with pytest.raises(ValueError):
            corpus.materialize(src, src / "inside")
        with pytest.raises(ValueError):
            corpus.materialize(src, scratch)               # dest 是 src 的祖先，清理它会删掉源码
        with pytest.raises(ValueError):
            corpus.materialize(src, src)
        with pytest.raises(FileNotFoundError):
            corpus.materialize(scratch / "missing", scratch / "d")
        assert (src / "a.c").exists()

    def test_tree_digest(self, scratch):
        a, b = scratch / "a", scratch / "b"
        self.make_tree(a)
        corpus.materialize(a, b)
        assert corpus.tree_digest(a) == corpus.tree_digest(b)      # .git 不参与，mtime 不参与
        (b / "a.c").write_text("changed")
        assert corpus.tree_digest(a) != corpus.tree_digest(b)

    def test_clang_args_resolve_relative_paths_against_root(self, scratch):
        (scratch / "src").mkdir()
        (scratch / "pre.h").write_text("")
        spec = CorpusSpec(name="a", cflags=["-std=c99", "-I.", "-Isrc", "-isystem", "inc", "-iquote", "/abs",
                                            "-include", "pre.h", "-include", "stdio.h", "-DX=1"])
        args = clangcfg.clang_args(spec.cflags, scratch)
        root = str(scratch.resolve())
        assert args[:5] == ["-x", "c", "-fparse-all-comments", "-w", "-ferror-limit=0"]
        rest = args[5:5 + 12]
        assert rest == ["-std=c99", f"-I{root}", f"-I{root}/src", "-isystem", f"{root}/inc", "-iquote", "/abs",
                        "-include", f"{root}/pre.h", "-include", "stdio.h", "-DX=1"]

    def test_clang_args_drop_flags_that_have_nothing_to_do_with_parsing(self, scratch):
        # 以前 corpus 与 extract 各有一套参数装配，同一份 cflags 得到的参数不同；现在只有 clangcfg.clang_args 一套
        args = clangcfg.clang_args(["-Werror", "-o", "x.o", "-lm", "-MD", "-std=c89"], scratch)
        assert "-std=c89" in args
        for dropped in ("-Werror", "-o", "x.o", "-lm", "-MD"):
            assert dropped not in args

    def test_builtin_include_args_point_at_a_dir_with_stddef(self):
        args = clangcfg.builtin_include_args()
        assert args == () or (args[0] == "-isystem" and (Path(args[1]) / "stddef.h").is_file())

    def test_mirror_dir_needs_the_environment_variable(self, monkeypatch, scratch):
        monkeypatch.delenv("CORPUS_MIRROR", raising=False)
        assert corpus.mirror_dir() is None                      # 不猜任何本机目录
        monkeypatch.setenv("CORPUS_MIRROR", str(scratch))
        assert corpus.mirror_dir() == scratch

    def test_default_work_root_is_defined_once(self, monkeypatch):
        from bench import runner, store
        monkeypatch.delenv("BENCH_WORK_DIR", raising=False)
        assert store.default_work_root() == store.PROJECT_ROOT / "tmp" / "work"
        assert store.Workspace.default().work == store.default_work_root()
        monkeypatch.setenv("BENCH_WORK_DIR", "/nonexistent/w")
        assert store.default_work_root() == Path("/nonexistent/w")
        assert not hasattr(corpus, "WORK_DIR") and not hasattr(runner, "WORK_ROOT")


class TestPrepare:
    """用本地 git 仓库验证 prepare，全程不联网。"""

    @pytest.fixture()
    def env(self, scratch, monkeypatch):
        upstream, commits = make_upstream(scratch)
        monkeypatch.delenv("CORPUS_KEEP_PROXY", raising=False)
        monkeypatch.setenv("CORPUS_MIRROR", str(scratch / "no-mirror"))     # 默认：没有镜像，只能克隆
        return {"scratch": scratch, "upstream": upstream, "commits": commits,
                "url": f"file://{upstream}", "root": scratch / "src"}

    def add_mirror(self, env, monkeypatch, shallow: bool = True, origin: str | None = None) -> Path:
        mirror = env["scratch"] / "mirror"
        mirror.mkdir()
        args = ["clone", "-q"] + (["--depth", "1"] if shallow else []) + [env["url"], str(mirror / "foo")]
        git(*args)
        if origin:
            git("remote", "set-url", "origin", origin, cwd=mirror / "foo")
        monkeypatch.setenv("CORPUS_MIRROR", str(mirror))
        return mirror / "foo"

    def test_clone_without_mirror_fills_in_head(self, env):
        spec = foo_spec(env["url"])
        dest = corpus.prepare(spec, env["root"])
        assert dest == env["root"] / "foo"
        assert git("rev-parse", "HEAD", cwd=dest) == env["commits"][-1] == spec.commit
        assert (dest / "lib.c").read_text() == "int f(void) { return 1; }\n"
        assert [p.name for p in env["root"].iterdir()] == ["foo"]          # 暂存目录没有残留

    def test_clone_pinned_older_commit(self, env):
        spec = foo_spec(env["url"], env["commits"][0])
        dest = corpus.prepare(spec, env["root"])
        assert git("rev-parse", "HEAD", cwd=dest) == env["commits"][0]
        assert (dest / "lib.c").read_text() == "int f(void) { return 0; }\n"

    def test_mirror_is_copied_without_touching_the_network(self, env, monkeypatch):
        fake_url = "https://example.invalid/some/foo.git"
        self.add_mirror(env, monkeypatch, origin=fake_url)
        spec = foo_spec(fake_url)
        dest = corpus.prepare(spec, env["root"])        # 若试图克隆 example.invalid 会失败
        assert spec.commit == env["commits"][-1] == git("rev-parse", "HEAD", cwd=dest)
        assert (dest / ".git").is_dir() and (dest / "lib.h").exists()

    def test_mirror_with_other_origin_is_ignored(self, env, monkeypatch):
        self.add_mirror(env, monkeypatch, origin="https://example.invalid/other/foo.git")
        spec = foo_spec(env["url"])                      # 镜像 origin 对不上 → 退回克隆本地上游
        assert corpus.prepare(spec, env["root"]).is_dir()
        assert spec.commit == env["commits"][-1]

    def test_pinned_commit_missing_in_shallow_mirror_falls_back_to_clone(self, env, monkeypatch):
        self.add_mirror(env, monkeypatch, shallow=True)            # 镜像只有最新提交
        spec = foo_spec(env["url"], env["commits"][0])
        dest = corpus.prepare(spec, env["root"])
        assert git("rev-parse", "HEAD", cwd=dest) == env["commits"][0]

    def test_pinned_commit_found_in_full_mirror_is_checked_out(self, env, monkeypatch):
        self.add_mirror(env, monkeypatch, shallow=False)
        spec = foo_spec(env["url"], env["commits"][0])
        dest = corpus.prepare(spec, env["root"])
        assert git("rev-parse", "HEAD", cwd=dest) == env["commits"][0]
        assert (dest / "lib.c").read_text() == "int f(void) { return 0; }\n"

    def test_unknown_commit_raises_and_leaves_nothing(self, env):
        spec = foo_spec(env["url"], "1" * 40)
        with pytest.raises(RuntimeError):
            corpus.prepare(spec, env["root"])
        assert list(env["root"].iterdir()) == []

    def test_existing_checkout_is_reused_not_recloned(self, env):
        spec = foo_spec(env["url"])
        dest = corpus.prepare(spec, env["root"])
        (dest / ".git" / "bench_marker").write_text("x")
        again = corpus.prepare(foo_spec(env["url"], env["commits"][-1]), env["root"])
        assert again == dest and (dest / ".git" / "bench_marker").exists()

    def test_dirty_checkout_is_restored(self, env):
        spec = foo_spec(env["url"])
        dest = corpus.prepare(spec, env["root"])
        (dest / "lib.c").write_text("garbage")
        (dest / "libfoo.a").write_text("build output")
        (dest / "build").mkdir()
        (dest / "build" / "x.o").write_text("o")
        corpus.prepare(spec, env["root"])
        assert (dest / "lib.c").read_text() == "int f(void) { return 1; }\n"
        assert not (dest / "libfoo.a").exists() and not (dest / "build").exists()

    def test_commit_change_in_spec_replaces_checkout(self, env):
        dest = corpus.prepare(foo_spec(env["url"]), env["root"])
        assert git("rev-parse", "HEAD", cwd=dest) == env["commits"][-1]
        corpus.prepare(foo_spec(env["url"], env["commits"][0]), env["root"])
        assert git("rev-parse", "HEAD", cwd=dest) == env["commits"][0]

    def test_empty_commit_is_written_back_to_spec_file(self, env, monkeypatch):
        spec_dir = env["scratch"] / "specs"
        spec_dir.mkdir()
        data = minimal_dict(name="foo", url=env["url"], project_files=["*.c"], target_files=["lib.c"])
        path = spec_dir / "foo.json"
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
        monkeypatch.setattr(corpus, "SPEC_DIR", spec_dir)
        spec = corpus.load_spec("foo")
        assert spec.commit == ""
        corpus.prepare(spec, env["root"])
        written = json.loads(path.read_text())
        assert written["commit"] == env["commits"][-1] == spec.commit
        assert {k: v for k, v in written.items() if k != "commit"} == {k: v for k, v in data.items() if k != "commit"}
        # 已经钉住的 spec 再 prepare 不改文件
        before = path.read_text()
        corpus.prepare(corpus.load_spec("foo"), env["root"])
        assert path.read_text() == before

    def test_unreachable_remote_fails_fast_with_timeout(self, env, monkeypatch):
        # 用一个会一直卡住的 "git"（睡眠脚本）模拟连不上 GitHub，确认超时后直接报错而不是再重试
        real_git = shutil.which("git")
        fake_bin = env["scratch"] / "bin"
        fake_bin.mkdir()
        (fake_bin / "git").write_text(
            f'#!/bin/sh\ncase "$*" in *fetch*|*clone*) sleep 30;; *) exec {real_git} "$@";; esac\n')
        (fake_bin / "git").chmod(0o755)
        monkeypatch.setenv("PATH", f"{fake_bin}{os.pathsep}{os.environ['PATH']}")
        monkeypatch.setenv("CORPUS_GIT_TIMEOUT", "1")
        import time
        t0 = time.monotonic()
        with pytest.raises(corpus.GitTimeout):
            corpus.prepare(foo_spec(env["url"], env["commits"][0]), env["root"])
        assert time.monotonic() - t0 < 10                 # 没有在「浅取失败 → 完整克隆」上再等第二轮
        assert list(env["root"].iterdir()) == []

    def test_repo_hooks_are_never_executed_on_the_host(self, env, monkeypatch):
        # 镜像里放一个 post-checkout hook；prepare 做 checkout / reset 时不能执行它
        mirror_repo = self.add_mirror(env, monkeypatch, shallow=False)
        marker = env["scratch"] / "hook-ran"
        hook = mirror_repo / ".git" / "hooks" / "post-checkout"
        hook.write_text(f"#!/bin/sh\ntouch {marker}\n")
        hook.chmod(0o755)
        spec = foo_spec(env["url"], env["commits"][0])        # 需要 checkout 到较早的提交
        dest = corpus.prepare(spec, env["root"])
        assert git("rev-parse", "HEAD", cwd=dest) == env["commits"][0]
        assert not marker.exists()

    @pytest.fixture()
    def clean_proxy_cache(self, monkeypatch):
        """清掉环境里已有的代理变量（本机的代理进程可能正在跑），并让探测结果的缓存不跨测试。"""
        for k in list(os.environ):
            if k.lower() in ("http_proxy", "https_proxy", "all_proxy"):
                monkeypatch.delenv(k)
        corpus._drop_proxy_for_git.cache_clear()
        yield
        corpus._drop_proxy_for_git.cache_clear()

    def test_dead_proxy_is_stripped_for_git(self, monkeypatch, clean_proxy_cache):
        # 端口 1 上没有任何服务：代理配了但连不上，git 联网时临时清掉（以前无条件清，需要代理的网络上 clone 会直接失败）
        for k in ("http_proxy", "HTTPS_PROXY", "ALL_PROXY"):
            monkeypatch.setenv(k, "http://127.0.0.1:1")
        monkeypatch.delenv("CORPUS_KEEP_PROXY", raising=False)
        monkeypatch.delenv("CORPUS_NO_PROXY", raising=False)
        env = corpus._git_env()
        assert not any(k.lower() in ("http_proxy", "https_proxy", "all_proxy") for k in env)

    def test_live_proxy_is_kept_for_git(self, monkeypatch, clean_proxy_cache):
        srv = socket.socket()
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        try:
            url = f"http://127.0.0.1:{srv.getsockname()[1]}"
            monkeypatch.setenv("http_proxy", url)
            monkeypatch.delenv("CORPUS_KEEP_PROXY", raising=False)
            monkeypatch.delenv("CORPUS_NO_PROXY", raising=False)
            assert corpus._git_env()["http_proxy"] == url
        finally:
            srv.close()

    def test_forcing_switches_beat_the_probe(self, monkeypatch, clean_proxy_cache):
        monkeypatch.setenv("http_proxy", "http://127.0.0.1:1")
        monkeypatch.setenv("CORPUS_KEEP_PROXY", "1")
        assert corpus._git_env()["http_proxy"] == "http://127.0.0.1:1"
        monkeypatch.delenv("CORPUS_KEEP_PROXY")
        srv = socket.socket()
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        try:
            monkeypatch.setenv("http_proxy", f"http://127.0.0.1:{srv.getsockname()[1]}")
            monkeypatch.setenv("CORPUS_NO_PROXY", "1")
            assert "http_proxy" not in corpus._git_env()
        finally:
            srv.close()

    def test_without_any_proxy_nothing_is_stripped(self, monkeypatch, clean_proxy_cache):
        monkeypatch.delenv("CORPUS_KEEP_PROXY", raising=False)
        monkeypatch.delenv("CORPUS_NO_PROXY", raising=False)
        assert corpus._drop_proxy_for_git_now() is False


# ======================================================================
# 2. 真实语料：schema / 展开 / includes 闭包 / libclang
# ======================================================================
_SOURCES: dict[str, tuple[CorpusSpec, Path]] = {}


def source(name: str) -> tuple[CorpusSpec, Path]:
    """加载 spec 并准备语料（缓存）。本地镜像和网络都拿不到时跳过。"""
    if name not in _SOURCES:
        spec = corpus.load_spec(name)
        try:
            root = corpus.prepare(spec)
        except (corpus.GitTimeout, RuntimeError) as exc:
            mirror = corpus.mirror_dir()
            if mirror is not None and mirror.is_dir():
                raise                       # 有本地镜像还失败，是真问题，不能跳过
            msg = f"语料 {name} 不可用（没设 CORPUS_MIRROR，克隆失败）: {exc}"
            if REQUIRE_CORPUS:
                pytest.fail(msg, pytrace=False)
            pytest.skip(msg)
        _SOURCES[name] = (spec, root)
    return _SOURCES[name]


def test_there_are_exactly_the_nine_specs():
    assert corpus.list_specs() == SPEC_NAMES


@pytest.mark.parametrize("name", SPEC_NAMES)
class TestSpecSchema:
    def test_schema(self, name):
        spec = corpus.load_spec(name)
        assert corpus.validate_spec(spec, strict=True) == []
        raw = json.loads((corpus.SPEC_DIR / f"{name}.json").read_text(encoding="utf-8"))
        assert set(raw) == set(CorpusSpec.__dataclass_fields__)      # 字段齐全，没有多余的
        assert spec.name == name
        assert re.fullmatch(r"https://github\.com/[\w.-]+/[\w.-]+\.git", spec.url)
        assert re.fullmatch(r"[0-9a-f]{40}", spec.commit)
        assert spec.test_cmd is None or isinstance(spec.test_cmd, str)
        assert spec.pass_criteria["exit_code"] == 0
        assert 10 <= spec.timeout_s <= 600
        assert len(spec.notes) > 40

    def test_coverage_field(self, name):
        cov = corpus.load_spec(name).coverage
        assert "--coverage" in cov["env"]["CFLAGS"] and "-O0" in cov["env"]["CFLAGS"]
        assert "--coverage" in cov["env"]["LDFLAGS"]
        assert "gcda_glob" not in cov

    def test_build_cmd_conventions(self, name):
        spec = corpus.load_spec(name)
        cmd = spec.build_cmd
        # 注入诊断格式 / --coverage 都靠环境变量，所以构建命令必须自己引用它们
        assert "$CFLAGS" in cmd and "$LDFLAGS" in cmd
        # 多条命令的脚本必须 set -e，否则前面的失败会被后面的成功盖掉
        assert cmd.startswith("set -e")
        assert not re.search(r"-Werror|-pedantic-errors", cmd)
        assert re.search(r"-std=\w+", cmd) and spec.std is not None
        assert f"-std={spec.std}" in cmd, "构建用的标准必须和 libclang 用的一致"
        assert "-Werror" not in spec.cflags

    def test_every_command_line_macro_is_in_keep_names(self, name):
        spec = corpus.load_spec(name)
        assert spec.command_line_macros() <= set(spec.keep_names)

    def test_pass_criteria_are_not_trivial_when_exit_code_cannot_be_trusted(self, name):
        # 这几个库的测试程序在失败时仍然退出 0，只能靠输出判断
        spec = corpus.load_spec(name)
        if name in {"parson", "mpc", "vec"}:
            assert spec.pass_criteria.get("stdout_not_regex")

    def test_every_corpus_can_tell_that_the_tests_really_ran_to_the_end(self, name):
        """候选在被测函数里提前 exit(0) 时程序以 0 退出：只看退出码的语料会误判通过。每个语料都要有「测试跑完」的正向证据——
        输出里的汇总行（stdout_regex），或构建命令自己核对每个测试程序的汇总行（cjson 的 Unity），或输出与基线逐字对比（inih）。"""
        spec = corpus.load_spec(name)
        if name == "cjson":
            assert "Tests 0 Failures" in spec.build_cmd
        elif name == "inih":
            assert "diff tests/baseline_" in spec.build_cmd
        else:
            assert spec.pass_criteria.get("stdout_regex"), f"{name} 只看退出码，提前 exit(0) 的候选会误判通过"

    def test_sds_test_driver_is_not_a_target(self, name):
        spec = corpus.load_spec(name)
        if name == "sds":
            assert spec.exclude_targets == ["sdsTest"]

    def test_files_expand_and_targets_are_subset(self, name):
        spec, root = source(name)
        files = corpus.expand_files(spec, root)
        assert files, "project_files 展开为空"
        rels = [p.relative_to(root).as_posix() for p in files]
        assert rels == sorted(set(rels)), "必须去重并稳定排序"
        assert corpus.expand_files(spec, root) == files                       # 确定性
        for pat in spec.project_files:                                       # 每个 pattern 都要有匹配
            assert corpus.expand_files(CorpusSpec(name="t", project_files=[pat]), root), pat
        targets = corpus.expand_target_files(spec, root)
        assert targets and set(targets) <= set(files)
        for pat in spec.target_files:
            assert corpus.expand_files(CorpusSpec(name="t", project_files=[pat]), root), pat
        assert all(p.suffix in {".c", ".h"} for p in files)
        # vendored 条目必须真实存在，且确实覆盖了至少一个项目文件
        for entry in spec.vendored:
            assert (root / entry).exists(), entry
            assert any(spec.is_vendored(r) for r in rels)
        assert not any(spec.is_vendored(p.relative_to(root).as_posix()) for p in targets)

    def test_checkout_is_pinned_and_pristine(self, name):
        spec, root = source(name)
        assert git("rev-parse", "HEAD", cwd=root) == spec.commit
        assert git("status", "--porcelain", "--ignored", cwd=root) == ""
        assert not any(p.name == ".git" for p in corpus.expand_files(spec, root))


@pytest.mark.parametrize("name", SPEC_NAMES)
def test_libclang_parses_project_without_errors(name, monkeypatch):
    """零 error 的口径：
    * 每个 .c 文件整体（含它 include 进来的项目头文件）零 error；
    * target_files 单独解析零 error（包括 jsmn.h 这种头文件实现的目标）；
    * 没被任何 .c 包含的项目头文件也要能单独解析。
    非目标头文件如果不是自含的（jsmn 的 test/testutil.h 用了 va_list 却没 include stdarg.h），
    只要求它在被包含的上下文里干净，不要求单独解析干净。
    """
    spec, root = source(name)
    monkeypatch.chdir(PROJECT_ROOT)       # 故意不在语料根下，证明 cflags 不依赖进程当前目录
    args = clangcfg.clang_args(spec.cflags, root)
    index = clangcfg.index()
    files = corpus.expand_files(spec, root)
    project = {p.resolve() for p in files}
    targets = {p.resolve() for p in corpus.expand_target_files(spec, root)}
    root_real = root.resolve()
    problems: list[str] = []
    outside: set[str] = set()
    included_by_some_tu: set[Path] = set()

    def errors_of(tu) -> list[str]:
        return [f"{d.location.file}:{d.location.line}: {d.spelling}"
                for d in tu.diagnostics if d.severity >= ci.Diagnostic.Error]

    def parse(f: Path):
        return index.parse(str(f), args=args, options=clangcfg.DEFAULT_PARSE_OPTIONS)

    for f in files:
        if f.suffix != ".c":
            continue
        tu = parse(f)
        problems += errors_of(tu)
        for inc in tu.get_includes():
            if inc.include is None:
                continue
            p = Path(inc.include.name).resolve()
            if root_real in p.parents:
                included_by_some_tu.add(p)
                if p not in project:
                    outside.add(p.relative_to(root_real).as_posix())
    for f in files:
        real = f.resolve()
        if f.suffix == ".h" and real not in targets and real in included_by_some_tu:
            continue
        if f.suffix == ".c" and real not in targets:
            continue
        tu = parse(f)
        problems += errors_of(tu)
        for inc in tu.get_includes():
            if inc.include is not None:
                p = Path(inc.include.name).resolve()
                if root_real in p.parents and p not in project:
                    outside.add(p.relative_to(root_real).as_posix())
    assert problems == []
    # 构建实际用到的、位于仓库内的头文件都必须在 project_files 里，否则混淆只改一半会编不过
    assert sorted(outside) == []


# ======================================================================
# 3. 真实语料 + docker
# ======================================================================
@pytest.fixture(scope="module")
def sb() -> DockerSandbox:
    return DockerSandbox()


@needs_docker
@pytest.mark.parametrize("name", SPEC_NAMES)
def test_build_cmd_passes_in_sandbox(name, sb, scratch):
    spec, root = source(name)
    ok, res = corpus.verify_build(spec, root, sb, work_root=scratch)
    assert ok, f"rc={res.rc} timed_out={res.timed_out}\n{res.stdout[-1500:]}\n{res.stderr[-1500:]}"


@needs_docker
def test_verify_build_reports_failure_for_a_broken_tree(sb, scratch):
    spec, root = source("vec")
    broken = corpus.materialize(root, scratch / "broken")
    (broken / "src" / "vec.c").write_text("this is not C\n")
    ok, res = corpus.verify_build(spec, broken, sb, work_root=scratch / "w")
    assert not ok and res.rc != 0


def count_gcda(workdir: Path) -> int:
    return len(list(workdir.rglob("*.gcda")))


@needs_docker
@pytest.mark.parametrize("name", SPEC_NAMES)
def test_coverage_env_builds_and_writes_gcda(name, sb, scratch):
    spec, root = source(name)
    work = corpus.materialize(root, scratch / name)
    res = sb.run(work, spec.run_command(), env=spec.coverage["env"], timeout_s=spec.timeout_s)
    assert spec.passes(res.rc, res.stdout), res.stdout[-800:] + res.stderr[-800:]
    assert count_gcda(work) >= 1
    assert list(work.rglob("*.gcno")), "编译期应当产生 .gcno"


@needs_docker
@pytest.mark.parametrize("name, func", [("cjson", "cJSON_Parse"), ("parson", "json_parse_string"),
                                        ("sds", "sdsnew")])
def test_gcov_reports_executed_target_functions(name, func, sb, scratch):
    spec, root = source(name)
    work = corpus.materialize(root, scratch / name)
    res = sb.run(work, spec.run_command(), env=spec.coverage["env"], timeout_s=spec.timeout_s)
    assert spec.passes(res.rc, res.stdout)
    probe = (f"for g in $(find . -name '*.gcda'); do gcov -f -n $g 2>/dev/null; done"
             f" | grep -A1 -h \"^Function '{func}'$\"")
    out = sb.run(work, probe, timeout_s=60).stdout
    executed = [float(m) for m in re.findall(r"Lines executed:([\d.]+)%", out)]
    assert executed and max(executed) > 0, out


# 每个语料一个「把某个函数改坏」的变异：build_cmd + pass_criteria 必须判失败。
# expect_rc0 标出那些即使被改坏退出码仍是 0、只能靠输出里的 pass_criteria 识别的情形。
MUTATIONS = {
    "cjson": ("cJSON.c",
              "CJSON_PUBLIC(int) cJSON_GetArraySize(const cJSON *array)\n{\n    cJSON *child = NULL;\n    size_t size = 0;\n",
              "CJSON_PUBLIC(int) cJSON_GetArraySize(const cJSON *array)\n{\n    cJSON *child = NULL;\n    size_t size = 0;\n    return 0;\n",
              False),
    "parson": ("parson.c",
               "int json_value_get_boolean(const JSON_Value *value) {\n    return json_value_get_type(value) == JSONBoolean ? value->value.boolean : -1;",
               "int json_value_get_boolean(const JSON_Value *value) {\n    return -1;",
               True),
    "sds": ("sds.c",
            "int sdscmp(const sds s1, const sds s2) {\n    size_t l1, l2, minlen;\n    int cmp;\n",
            "int sdscmp(const sds s1, const sds s2) {\n    size_t l1, l2, minlen;\n    int cmp;\n    return 0;\n",
            False),
    "jsmn": ("jsmn.h",
             "  token->type = type;\n  token->start = start;",
             "  token->type = JSMN_UNDEFINED;\n  token->start = start;",
             False),
    "tinyexpr": ("tinyexpr.c",
                 "const double ret = n ? te_eval(n) : NAN;",
                 "const double ret = 0;",
                 False),
    "mpc": ("mpc.c",
            "void mpc_err_delete(mpc_err_t *x) {",
            "void mpc_err_delete(mpc_err_t *x) { *(volatile int *)0 = 1;",
            True),     # ptest 的信号处理函数 exit(0)：崩溃也是退出码 0
    "tiny-regex-c": ("re.c",
                     "static int matchdigit(char c)\n{\n",
                     "static int matchdigit(char c)\n{\n  return 0;\n",
                     False),
    "inih": ("ini.c",
             "    while (*s && isspace((unsigned char)(*s)))\n        s++;\n    return (char*)s;",
             "    return (char*)s;",
             False),
    "vec": ("src/vec.c",
            "void vec_splice_(char **data, int *length, int *capacity, int memsz,\n                 int start, int count\n) {\n  (void) capacity;",
            "void vec_splice_(char **data, int *length, int *capacity, int memsz,\n                 int start, int count\n) {\n  (void) capacity; return;",
            True),
}


# 每个语料一个「在被测函数开头提前 exit(0)」的变异：程序以 0 退出，只看退出码会误判通过，所以判据必须能识别「测试没跑完」。
# 锚点与 MUTATIONS 同类（项目测试一定会走到的函数），新文本把 return / 破坏换成 exit(0)。
EXIT_MUTATIONS = {
    "cjson": ("cJSON.c",
              "CJSON_PUBLIC(int) cJSON_GetArraySize(const cJSON *array)\n{\n    cJSON *child = NULL;\n    size_t size = 0;\n",
              "CJSON_PUBLIC(int) cJSON_GetArraySize(const cJSON *array)\n{\n    cJSON *child = NULL;\n    size_t size = 0;\n    exit(0);\n"),
    "parson": ("parson.c",
               "int json_value_get_boolean(const JSON_Value *value) {\n    return json_value_get_type(value) == JSONBoolean ? value->value.boolean : -1;",
               "int json_value_get_boolean(const JSON_Value *value) {\n    exit(0);\n    return -1;"),
    "sds": ("sds.c",
            "int sdscmp(const sds s1, const sds s2) {\n    size_t l1, l2, minlen;\n    int cmp;\n",
            "int sdscmp(const sds s1, const sds s2) {\n    size_t l1, l2, minlen;\n    int cmp;\n    exit(0);\n"),
    "jsmn": ("jsmn.h",
             "  token->type = type;\n  token->start = start;",
             "  exit(0);\n  token->type = type;\n  token->start = start;"),
    "tinyexpr": ("tinyexpr.c",
                 "const double ret = n ? te_eval(n) : NAN;",
                 "exit(0); const double ret = 0;"),
    "mpc": ("mpc.c",
            "void mpc_err_delete(mpc_err_t *x) {",
            "void mpc_err_delete(mpc_err_t *x) { exit(0);"),
    "tiny-regex-c": ("re.c",
                     "static int matchdigit(char c)\n{\n",
                     "static int matchdigit(char c)\n{\n  exit(0);\n"),
    "inih": ("ini.c",
             "    while (*s && isspace((unsigned char)(*s)))\n        s++;\n    return (char*)s;",
             "    exit(0);\n    return (char*)s;"),
    "vec": ("src/vec.c",
            "void vec_splice_(char **data, int *length, int *capacity, int memsz,\n                 int start, int count\n) {\n  (void) capacity;",
            "void vec_splice_(char **data, int *length, int *capacity, int memsz,\n                 int start, int count\n) {\n  (void) capacity; exit(0);"),
}


def test_every_spec_has_a_mutation():
    assert set(MUTATIONS) == set(SPEC_NAMES)
    assert set(EXIT_MUTATIONS) == set(SPEC_NAMES)


@needs_docker
@pytest.mark.parametrize("name", SPEC_NAMES)
def test_pass_criteria_detects_a_broken_function(name, sb, scratch):
    spec, root = source(name)
    rel, old, new, expect_rc0 = MUTATIONS[name]
    work = corpus.materialize(root, scratch / name)
    text = (work / rel).read_text()
    assert text.count(old) == 1, f"变异锚点在 {rel} 里必须恰好出现一次"
    (work / rel).write_text(text.replace(old, new))
    res = sb.run(work, spec.run_command(), timeout_s=spec.timeout_s)
    assert not res.timed_out
    assert not spec.passes(res.rc, res.stdout), "被改坏的实现居然通过了"
    if expect_rc0:
        assert res.rc == 0, "这个语料的测试程序失败时本应退出 0，靠输出里的 pass_criteria 识别"


@needs_docker
@pytest.mark.parametrize("name", SPEC_NAMES)
def test_pass_criteria_detect_an_early_exit_zero(name, sb, scratch):
    """提前 exit(0) 的候选：程序以退出码 0 结束，没有跑完测试。每个语料都必须判失败（以前 cjson / jsmn / sds 只看退出码，会判通过）。"""
    spec, root = source(name)
    rel, old, new = EXIT_MUTATIONS[name]
    work = corpus.materialize(root, scratch / name)
    text = (work / rel).read_text()
    assert text.count(old) == 1, f"变异锚点在 {rel} 里必须恰好出现一次"
    (work / rel).write_text(text.replace(old, new))
    res = sb.run(work, spec.run_command(), timeout_s=spec.timeout_s)
    assert not res.timed_out
    assert not spec.passes(res.rc, res.stdout), f"提前 exit(0) 的实现居然通过了（rc={res.rc}）:\n{res.stdout[-400:]}"


def test_verify_build_refuses_truncated_output_for_output_judged_specs(scratch):
    class Fake:
        def __init__(self, res):
            self.res = res

        def run(self, workdir, cmd, env=None, timeout_s=None):
            return self.res

    (scratch / "src").mkdir()
    (scratch / "src" / "a.c").write_text("int a;")
    spec = CorpusSpec(name="a", build_cmd="true", pass_criteria={"exit_code": 0, "stdout_not_regex": "FAILED"})
    ok_res = RunResult(rc=0, stdout="fine\n", stderr="", duration_s=0.1, timed_out=False, signal=None)
    assert corpus.verify_build(spec, scratch / "src", Fake(ok_res), work_root=scratch / "w")[0] is True
    cut = RunResult(rc=0, stdout="fine\n", stderr="", duration_s=0.1, timed_out=False, signal=None, stdout_truncated=True)
    assert corpus.verify_build(spec, scratch / "src", Fake(cut), work_root=scratch / "w")[0] is False
    broken = RunResult(rc=125, stdout="", stderr="docker: Error response from daemon: boom", duration_s=0.1,
                       timed_out=False, signal=None)
    from bench.sandbox import SandboxError
    with pytest.raises(SandboxError):
        corpus.verify_build(spec, scratch / "src", Fake(broken), work_root=scratch / "w")


@needs_docker
def test_source_trees_stay_pristine_after_all_the_above():
    for name in SPEC_NAMES:
        spec, root = source(name)
        assert git("status", "--porcelain", "--ignored", cwd=root) == "", name
        assert git("rev-parse", "HEAD", cwd=root) == spec.commit
