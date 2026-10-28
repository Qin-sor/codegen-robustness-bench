"""仓库卫生检查：违例用语、本机绝对路径、临时文件位置，以及 README / 设计文档与实现是否一致。

都不需要 docker，毫秒级。以前只有 6 个 bench 模块有用语检查，其余模块、测试、README、docs 没有任何守卫；
README 里的环境变量清单、命令行选项、目录结构、局限列表、数据文件表全靠人记着改，改漏了没人发现。
这里把它们都变成断言：代码里多了一个环境变量 / 选项 / 模块而 README 没写，这个文件里的测试就会失败。
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pytest

from bench import cli, limitations, vocab
from bench.llm import PRESETS
from bench.store import Workspace
from tests.helpers import BANNED, ROOT

README = ROOT / "README.zh-CN.md"
DESIGN = ROOT / "docs" / "设计.md"
EXPERIMENT_LOG = ROOT / "docs" / "实验记录.md"

#: 本机绝对路径的特征（拼接着写，免得这个文件自己被 grep 命中）
HOST_PATH_NEEDLES = ["/" + "home/", "/" + "Users/", "/" + "root/", "C:" + "\\" + "Users"]

_TEXT_SUFFIXES = {".py", ".md", ".json", ".sh", ".toml", ".c", ".h", ".txt", ".cfg", ".ini", ".yml", ".yaml"}
_SKIP_DIRS = {"__pycache__", ".git", ".pytest_cache", "tmp", "results"}


def _project_text_files() -> list[Path]:
    """要检查的文本文件：bench / tests / scripts / docs / corpus/specs、根目录的 README 与配置。
    不含语料源码（corpus/src，第三方代码）、tmp/ 与 results/（运行产物，另有各自的测试）。"""
    files: list[Path] = []
    for top in ("bench", "tests", "scripts", "docs", "corpus/specs"):
        base = ROOT / top
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*")):
            if p.is_file() and not (set(p.relative_to(ROOT).parts) & _SKIP_DIRS) and p.suffix in _TEXT_SUFFIXES:
                files.append(p)
    for name in ("README.md", "README.zh-CN.md", "pyproject.toml", ".gitignore"):
        if (ROOT / name).is_file():
            files.append(ROOT / name)
    return files


FILES = _project_text_files()


def _rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


def _readme_section(title_part: str) -> str:
    """README 里标题含 title_part 的那一节（到下一个同级或更高级标题为止）。"""
    text = README.read_text(encoding="utf-8")
    m = re.search(rf"^(#{{1,6}}) [^\n]*{re.escape(title_part)}[^\n]*$", text, flags=re.M)
    assert m, f"README 里找不到标题含「{title_part}」的一节"
    level = len(m.group(1))
    rest = text[m.end():]
    nxt = re.search(rf"^#{{1,{level}}} ", rest, flags=re.M)
    return rest[:nxt.start()] if nxt else rest


# ----------------------------------------------------------------------
# 用语与路径
# ----------------------------------------------------------------------
class TestStyle:
    def test_the_file_list_covers_every_part_of_the_project(self):
        rels = {_rel(p) for p in FILES}
        assert {"README.md", "README.zh-CN.md", "docs/设计.md", "pyproject.toml", "scripts/ci.sh", "bench/pipeline.py",
                "bench/obfuscate/symbols.py", "tests/test_hygiene.py", "corpus/specs/cjson.json"} <= rels

    @pytest.mark.parametrize("path", FILES, ids=_rel)
    def test_no_translationese_terms(self, path):
        text = path.read_text(encoding="utf-8")
        assert [w for w in BANNED if w in text] == []

    @pytest.mark.parametrize("path", FILES, ids=_rel)
    def test_no_host_absolute_paths(self, path):
        text = path.read_text(encoding="utf-8")
        assert [n for n in HOST_PATH_NEEDLES + [str(ROOT)] if n in text] == []

    def test_no_api_key_like_strings_or_phone_numbers(self):
        key_re = re.compile(r"\bsk-[A-Za-z0-9]{16,}\b")
        phone_re = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
        for path in FILES:
            text = path.read_text(encoding="utf-8")
            assert not key_re.search(text), _rel(path)
            assert not phone_re.search(text), _rel(path)

    def test_test_modules_share_the_common_fixtures(self):
        """ROOT / MINI / IMAGE、needs_docker 标记、scratch 夹具只在 tests/helpers.py 与 tests/conftest.py 里各定义一份。"""
        offenders = []
        for path in sorted((ROOT / "tests").glob("test_*.py")):
            text = path.read_text(encoding="utf-8")
            for pattern, what in ((r"^(ROOT|MINI|IMAGE|PROJECT_ROOT)\s*=", "重定义共用常量"),
                                  (r"skipif\([^)]*docker", "自带 docker 跳过标记"),
                                  (r"^def (scratch|scratch_module)\b", "自带 scratch 夹具"),
                                  (r"importorskip\([\"']bench", "对 bench 自己的模块用 importorskip（模块坏了会变成 skip 而不是失败）")):
                if re.search(pattern, text, flags=re.M | re.I):
                    offenders.append(f"{path.name}: {what}")
        assert offenders == []

    def test_temp_files_stay_under_the_project(self):
        """bench 里临时目录 / 文件都要显式给 dir=（落在工作区或项目 tmp/ 下），不能默认落到系统 /tmp。"""
        offenders = []
        for path in (ROOT / "bench").rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for m in re.finditer(r"(mkdtemp|mkstemp|TemporaryDirectory|NamedTemporaryFile)\(([^)]*)\)", text):
                if "dir=" not in m.group(2):
                    offenders.append(f"{_rel(path)}: {m.group(0)}")
            if "gettempdir" in text:
                offenders.append(f"{_rel(path)}: gettempdir")
        assert offenders == []


# ----------------------------------------------------------------------
# 打包与版本
# ----------------------------------------------------------------------
class TestPackaging:
    def test_python_version_is_declared_and_checked_consistently(self):
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        assert re.search(r'^requires-python\s*=\s*">=3\.10"', pyproject, flags=re.M)
        assert 'dependencies = ["clang==14.0"]' in pyproject            # 运行时唯一的第三方依赖
        init = (ROOT / "bench" / "__init__.py").read_text(encoding="utf-8")
        assert "sys.version_info < (3, 10)" in init
        assert sys.version_info >= (3, 10)
        assert "Python 3.10" in README.read_text(encoding="utf-8")

    def test_pytest_markers_and_ci_script(self):
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        assert "needs_docker" in pyproject and 'pythonpath = ["."]' in pyproject
        ci = (ROOT / "scripts" / "ci.sh").read_text(encoding="utf-8")
        assert "BENCH_REQUIRE_DOCKER=1" in ci and "BENCH_REQUIRE_CORPUS=1" in ci
        assert "skipped" in ci and "python3 -m bench e2e --backend mock" in ci and "pyflakes" in ci


# ----------------------------------------------------------------------
# README 与实现一致
# ----------------------------------------------------------------------
_ENV_TOKEN_RE = re.compile(r"\b(?:BENCH|CORPUS)_[A-Z0-9_]+\b|\bLIBCLANG_PATH\b")
_ENV_LITERAL_RE = re.compile(r"""["']((?:BENCH|CORPUS)_[A-Z0-9_]+|LIBCLANG_PATH)["']""")   # 代码里的字符串字面量就是变量名
_ENV_READ_RE = re.compile(r"""(?:environ(?:\.get)?\s*[(\[]|getenv\()\s*["']((?:BENCH|CORPUS)_[A-Z0-9_]+|LIBCLANG_PATH)["']""")
_ENV_EXPORT_RE = re.compile(r"^\s*export\s+((?:BENCH|CORPUS)_[A-Z0-9_]+)=", flags=re.M)


def _env_vars_read_by_the_code() -> set[str]:
    """代码实际读的环境变量名（BENCH_* / CORPUS_* / LIBCLANG_PATH）：bench 里出现的字符串字面量，
    tests 里 os.environ / getenv 读的（测试里 monkeypatch 出来的假变量名不算），加上脚本里 export 的。"""
    names: set[str] = set()
    for path in FILES:
        top = path.relative_to(ROOT).parts[0]
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".py" and top == "bench":
            names |= set(_ENV_LITERAL_RE.findall(text))
        elif path.suffix == ".py" and top == "tests":
            names |= set(_ENV_READ_RE.findall(text))
        elif path.suffix == ".sh":
            names |= set(_ENV_EXPORT_RE.findall(text))
    return names


def _cli_options() -> dict[str, set[str]]:
    ap = cli.build_parser()
    sub = next(a for a in ap._actions if isinstance(a, argparse._SubParsersAction))
    return {name: {s for a in p._actions for s in a.option_strings if s.startswith("--") and s != "--help"}
            for name, p in sub.choices.items()}


class TestReadmeMatchesTheCode:
    def test_every_environment_variable_the_code_reads_is_documented(self):
        readme = README.read_text(encoding="utf-8")
        read = _env_vars_read_by_the_code()
        assert {"BENCH_ROOT", "BENCH_REQUIRE_DOCKER", "CORPUS_MIRROR", "LIBCLANG_PATH"} <= read    # 抽取本身没坏
        assert sorted(n for n in read if f"`{n}`" not in readme) == []

    def test_readme_documents_no_environment_variable_the_code_does_not_read(self):
        read = _env_vars_read_by_the_code()
        mentioned = set(_ENV_TOKEN_RE.findall(README.read_text(encoding="utf-8")))
        assert sorted(mentioned - read) == []

    def test_api_key_variables_of_every_preset_are_documented(self):
        readme = README.read_text(encoding="utf-8")
        assert sorted({p.api_key_env for p in PRESETS.values() if f"`{p.api_key_env}`" not in readme and
                       p.api_key_env not in readme}) == []

    def test_every_subcommand_and_option_is_documented(self):
        readme = README.read_text(encoding="utf-8")
        opts = _cli_options()
        assert set(opts) == set(cli._COMMANDS)
        missing = [f"`{name}`" for name in opts if f"| `{name}` |" not in readme]
        missing += [f"{name} {flag}" for name, flags in opts.items() for flag in sorted(flags) if flag not in readme]
        assert missing == []

    def test_readme_mentions_no_option_that_does_not_exist(self):
        known = set().union(*_cli_options().values())
        foreign = {"--network", "--cap-drop", "--pids-limit", "--security-opt", "--coverage", "--memory-swap",
                   "--ulimit", "--rm", "--name",                        # docker / gcc 的选项，README 在讲沙箱时会提到
                   "--update",                                           # python3 -m bench.experiment_record 的选项
                   "--durations", "--junitxml"}                          # pytest 的选项（README 第 11 节讲怎么量耗时）
        mentioned = set(re.findall(r"(?<![\w-])--[a-z][a-z0-9-]+", README.read_text(encoding="utf-8")))
        assert sorted(mentioned - known - foreign) == []

    def test_exit_codes_are_documented_the_way_the_cli_states_them(self):
        readme = README.read_text(encoding="utf-8")
        for code, what in (("0", "成功"), ("1", "失败"), ("2", "用法错误"), ("130", "中断")):
            assert re.search(rf"\b{code}\b[^；。\n]*{what}", readme), (code, what)
        assert "130" in cli.__doc__ and "130" in cli.EPILOG

    def test_every_module_appears_in_the_readme_directory_listing(self):
        section = _readme_section("目录结构")
        mods = [p for p in (ROOT / "bench").rglob("*.py") if "__pycache__" not in p.parts]
        missing = [p.relative_to(ROOT / "bench").as_posix() for p in mods
                   if p.stem not in ("__init__", "__main__") and not re.search(rf"\b{re.escape(p.stem)}(\.py)?\b", section)]
        assert missing == []
        for extra in ("__main__.py", "docker/", "scripts/", "pyproject.toml", "docs/", "tests/", "corpus/specs"):
            assert extra in section, extra

    def test_every_module_appears_in_the_design_document(self):
        design = DESIGN.read_text(encoding="utf-8")
        mods = [p for p in (ROOT / "bench").rglob("*.py") if "__pycache__" not in p.parts]
        assert sorted(p.stem for p in mods if p.stem not in ("__init__", "__main__") and p.stem not in design) == []

    def test_limitations_are_the_shared_list(self):
        section = _readme_section("局限")
        numbered = re.findall(r"^\d+\. \*\*(.+?)\*\*", section, flags=re.M)
        titles = [t.rstrip("。") for t, _ in limitations.LIMITATIONS]
        assert [t.rstrip("。") for t in numbered] == titles
        for _title, body in limitations.LIMITATIONS:
            # 正文的每个句子（按句号切）都要原样出现在 README 里，README 不另写一份口径
            flat = re.sub(r"\s+", "", section)
            for sentence in filter(None, (s.strip() for s in body.split("。"))):
                assert re.sub(r"\s+", "", sentence) in flat, sentence[:40]

    def test_data_file_table_matches_the_workspace_layout(self):
        ws = Workspace.under(ROOT / "tmp" / "hygiene-not-created")
        pairs = {
            "prepare-<corpus>.json": ws.prepare_file("c").name == "prepare-c.json",
            "coverage-<corpus>.json": ws.coverage_file("c").name == "coverage-c.json",
            "targets-<corpus>.jsonl": ws.targets_file("c").name == "targets-c.jsonl",
            "targets-<corpus>.summary.json": ws.targets_summary_file("c").name == "targets-c.summary.json",
            "obf-<corpus>-<level>.json": ws.obf_file("c", "L1").name == "obf-c-L1.json",
            "descriptions.jsonl": ws.descriptions_file.name == "descriptions.jsonl",
            "tasks.jsonl": ws.tasks_file.name == "tasks.jsonl",
            "generations-<model>.jsonl": ws.generations_file("m").name == "generations-m.jsonl",
            "generation-failures-<model>.jsonl": ws.generation_failures_file("m").name == "generation-failures-m.jsonl",
            "results-<tag>.jsonl": ws.results_file("t").name == "results-t.jsonl",
            "completeness-<tag>.json": ws.completeness_file("t").name == "completeness-t.json",
            "usage.jsonl": ws.usage_file.name == "usage.jsonl",
            "summary-<tag>.md": ws.summary_file("t", ".md").name == "summary-t.md",
            "fig-<tag>.svg": ws.summary_file("t", ".svg").name == "fig-t.svg",
            # 缺陷修复任务（附录 §7）
            "mutants-<corpus>-<level>.jsonl": ws.mutants_file("c", "L1").name == "mutants-c-L1.jsonl",
            "mutants-<corpus>.summary.json": ws.mutants_summary_file("c").name == "mutants-c.summary.json",
            "mutants-<corpus>.check.json": ws.mutants_check_file("c").name == "mutants-c.check.json",
            "tasks-repair.jsonl": ws.repair_tasks_file.name == "tasks-repair.jsonl",
            "generations-repair-<model>.jsonl": ws.repair_generations_file("m").name == "generations-repair-m.jsonl",
            "generation-failures-repair-<model>.jsonl":
                ws.repair_generation_failures_file("m").name == "generation-failures-repair-m.jsonl",
            "results-repair-<tag>.jsonl": ws.repair_results_file("t").name == "results-repair-t.jsonl",
            "completeness-repair-<tag>.json": ws.repair_completeness_file("t").name == "completeness-repair-t.json",
        }
        assert [k for k, ok in pairs.items() if not ok] == []              # 表里写的文件名确实是 Workspace 给出的
        section = _readme_section("数据文件")
        assert [k for k in pairs if k not in section] == []                # 每个文件都在 README 的数据文件表里

    def test_task_input_difference_is_the_shared_text(self):
        """报告开头说明两类任务输入差异的文字（bench/limitations.py 单点维护）也要原样出现在 README 里。"""
        flat = re.sub(r"\s+", "", README.read_text(encoding="utf-8"))
        assert [line[:30] for line in limitations.TASK_INPUT_LINES if re.sub(r"\s+", "", line) not in flat] == []

    def test_repair_task_is_described_where_users_look(self):
        readme = README.read_text(encoding="utf-8")
        for needle in ("python3 -m bench mutate", "--task repair", "same_operator_as_L0", "四级同算子", "unchanged", "restored",
                       "mock-unchanged", "kill_mode"):
            assert needle in readme, needle
        design = DESIGN.read_text(encoding="utf-8")
        assert "设计-缺陷修复任务.md" in design and "`mutate`" in design and "v0.3" in design
        assert (ROOT / "docs" / "设计-缺陷修复任务.md").is_file()

    def test_readme_has_no_stale_claims(self):
        text = README.read_text(encoding="utf-8")
        assert "个用例" not in text                                          # 用例数很快过期，不写死
        assert "样本 i 把 i 当 seed 传给模型" not in text                       # DeepSeek 预设不发 seed，只进缓存键
        assert "本机没有出口网络时" not in text                                  # 本地镜像只认 CORPUS_MIRROR，与有没有网络无关


class TestDesignDocument:
    def test_version_and_deviation_record(self):
        design = DESIGN.read_text(encoding="utf-8")
        assert re.search(r"版本 v0\.[2-9]", design)
        assert "实现偏差记录" in design

    def test_vocabularies_in_the_design_document_match_the_code(self):
        design = DESIGN.read_text(encoding="utf-8")
        for cat in vocab.CATEGORIES:
            assert cat in design, cat
        for issue in vocab.ISSUES:
            assert issue in design, issue
        for level in vocab.LEVELS:
            assert level in design

    def test_experiment_record_exists_and_covers_every_corpus(self):
        from bench import corpus
        text = EXPERIMENT_LOG.read_text(encoding="utf-8")
        assert [n for n in corpus.list_specs() if n not in text] == []
        for needle in ("stub 存活", "oracle gate", "代码摘要"):
            assert needle in text, needle
