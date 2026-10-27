"""语料管理：读 spec、准备源码（复制本地镜像或克隆）、展开文件列表、生成工作副本。

语料不提交进仓库，``prepare`` 把它放到 ``corpus/src/<name>``（仓库根下，已在 .gitignore 里）。
来源有两种，按顺序尝试：

1. 本地镜像：**只有**设置了环境变量 ``CORPUS_MIRROR`` 才会用，它指向的目录里按仓库名放着 git 克隆。
   存在就直接复制，不联网。没设置这个变量就不会去猜任何本机目录。
2. 按 spec 里的 url 克隆。联网命令默认 180 s 超时（``CORPUS_GIT_TIMEOUT`` 可改）。想强制走克隆，
   不设 ``CORPUS_MIRROR`` 即可。

代理：环境变量（http_proxy / https_proxy / all_proxy）里配了代理时，默认先探测它连不连得上——连得上就照常使用，
一个都连不上（代理进程没起，常见于换了网络环境的机器）才临时清掉代理变量并给 git 加 ``-c http.proxy=`` 等参数绕开，
免得 clone 卡满超时。``CORPUS_KEEP_PROXY=1`` 强制保留代理，``CORPUS_NO_PROXY=1`` 强制清掉。

spec 约定（除设计文档 3.1 的字段外多了两个可选字段）：

* ``vendored``：第三方代码目录（相对项目根，例如 cJSON 的 ``tests/unity``）。这些文件仍在
  ``project_files`` 里、会被 libclang 解析，但符号混淆不改它们的名字。
* ``keep_names``：必须保持原拼写的名字。主要是只在构建命令里用 ``-DNAME`` 传入、同时源码里又有
  ``#define NAME`` 默认值的配置宏（inih 的 ``INI_MAX_LINE`` 一类）——改了名，命令行开关就失效了。
* ``build_cmd`` 是一段在项目根下由 ``bash -c`` 执行的脚本（可多行），一律显式引用
  ``$CFLAGS`` / ``$LDFLAGS``：运行器注入诊断格式、覆盖率注入 ``--coverage`` 都靠环境变量，
  所以不依赖仓库 Makefile 是否吃这两个变量。
* 构建命令里不带 ``-Werror`` / ``-pedantic-errors``：L2 / L3 变换和模型生成的代码都可能引入新警告，
  把警告当错误会让失败原因和被测能力无关。
* ``cflags`` 只给 libclang / ``gcc -fsyntax-only`` 用；里面相对的 ``-I`` 路径相对项目根，
  用 :func:`bench.clangcfg.clang_args` 转成绝对路径再传给 libclang（它还会补上缺失的 builtin 头目录）。
* ``exclude_targets``：不能当目标函数的名字（``name`` 或 ``file:name``）。典型是编进目标文件的测试驱动
  （sds 的 ``sdsTest``）：把它换成桩，测试程序的汇总行就没了，会被误判成「桩被杀死」。
* ``allow_words``：描述（功能规格）的泄漏检查里，除通用英文词表外额外放行的词。只用于这个语料里确实是领域通用词、
  又恰好是项目标识符的少数几个（如 tinyexpr 的 ``expr``），必须逐个有理由；描述记录会如实记下放行命中，
  审计时可以核对。它不参与 :meth:`CorpusSpec.digest`（不影响构建与测试，改它不该让构建类产出失效）。

另外提供 :func:`verify_build`：在沙箱里对源码树的干净副本跑 ``build_cmd`` 并按 ``pass_criteria`` 判定，
``prepare`` 之后用它确认构建与测试可用。

环境变量：``CORPUS_MIRROR``（本地镜像目录）、``CORPUS_KEEP_PROXY`` / ``CORPUS_NO_PROXY``（git 联网时强制保留 / 清掉代理）、
``CORPUS_GIT_TIMEOUT``（联网 git 命令超时秒数，默认 180）、``BENCH_ROOT`` / ``BENCH_SPEC_DIR`` /
``BENCH_CORPUS_SRC``（覆盖项目根、spec 目录、语料目录，默认由本文件位置推出）。
"""
from __future__ import annotations

import fnmatch
import functools
import glob
import hashlib
import json
import logging
import os
import re
import shutil
import socket
import subprocess
import tempfile
import urllib.parse
from dataclasses import dataclass, field, fields
from pathlib import Path, PurePosixPath
from typing import Any

from bench.sandbox import raise_if_docker_failed
from bench.store import PROJECT_ROOT, default_work_root

log = logging.getLogger(__name__)

SPEC_DIR = Path(os.environ.get("BENCH_SPEC_DIR") or PROJECT_ROOT / "corpus" / "specs")
CORPUS_SRC = Path(os.environ.get("BENCH_CORPUS_SRC") or PROJECT_ROOT / "corpus" / "src")

# 绕开失效代理的 git 参数（设计文档 3 节）
GIT_NOPROXY_ARGS = ["-c", "http.proxy=", "-c", "https.proxy=",
                    "-c", "http.https://github.com/.proxy=",
                    "-c", "http.https://github.com.proxy="]
_PROXY_ENV_KEYS = ("http_proxy", "https_proxy", "all_proxy")

SPEC_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
PASS_CRITERIA_KEYS = {"exit_code", "stdout_regex", "stdout_not_regex"}
COVERAGE_KEYS = {"env", "build_cmd"}
#: 不影响构建、测试与目标选择的字段：改了它们不该让已有产出（prepare / coverage / targets / 混淆副本 / 评测结果）作废
DIGEST_EXCLUDED_FIELDS = ("url", "license", "notes", "allow_words")
_DEFINE_RE = re.compile(r"(?<![\w-])-[DU]\s*([A-Za-z_]\w*)")


class SpecError(ValueError):
    """spec 文件不合法。"""


# ----------------------------------------------------------------------
# 数据类
# ----------------------------------------------------------------------
@dataclass
class CorpusSpec:
    name: str
    url: str = ""
    commit: str = ""
    license: str = ""
    project_files: list[str] = field(default_factory=list)
    target_files: list[str] = field(default_factory=list)
    cflags: list[str] = field(default_factory=list)
    build_cmd: str = ""
    test_cmd: str | None = None
    pass_criteria: dict = field(default_factory=lambda: {"exit_code": 0})
    coverage: dict = field(default_factory=dict)
    timeout_s: int = 120
    notes: str = ""
    # 以下四个是设计文档 4.1 字段表之外的可选字段（vendored / keep_names 见 4.6，allow_words 见 4.8，exclude_targets 见 4.5）
    vendored: list[str] = field(default_factory=list)
    keep_names: list[str] = field(default_factory=list)
    allow_words: list[str] = field(default_factory=list)
    exclude_targets: list[str] = field(default_factory=list)

    # -- 构造与序列化 ---------------------------------------------------
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CorpusSpec":
        """从 dict 构造；未知键报错（防止拼错字段名），以下划线开头的键视为注释忽略。"""
        known = {f.name for f in fields(cls)}
        extra = [k for k in data if k not in known and not k.startswith("_")]
        if extra:
            raise SpecError(f"spec 含未知字段: {extra}")
        if "name" not in data:
            raise SpecError("spec 缺少 name")
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def digest(self) -> str:
        """影响构建与测试语义的字段的内容哈希，结果文件用它校验输入没变。

        ``url`` / ``license`` / ``notes`` / ``allow_words`` 不参与：它们不改变构建、测试和目标选择，
        给 notes 补一句备注就让整套产出作废（prepare、coverage、targets、混淆副本、评测结果全部重跑）得不偿失。
        """
        data = {k: v for k, v in self.to_dict().items() if k not in DIGEST_EXCLUDED_FIELDS}
        blob = json.dumps(data, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    # -- 便捷方法 -------------------------------------------------------
    def is_excluded_target(self, file: str, name: str) -> bool:
        """函数（file 相对项目根，name 是函数名）是否被 exclude_targets 明确排除。"""
        return name in self.exclude_targets or f"{file}:{name}" in self.exclude_targets

    def is_vendored(self, rel_path: str) -> bool:
        """rel_path（相对项目根）是否落在 vendored 目录里。条目可以是目录或 glob。"""
        rel = PurePosixPath(rel_path)
        for entry in self.vendored:
            ent = entry.rstrip("/")
            if any(ch in ent for ch in "*?["):
                if fnmatch.fnmatch(rel.as_posix(), ent):
                    return True
            elif rel == PurePosixPath(ent) or PurePosixPath(ent) in rel.parents:
                return True
        return False

    @property
    def std(self) -> str | None:
        """cflags 里 -std= 的取值（例如 "c89"），没有则为 None；给 prompt 的 language_std 用。"""
        for flag in self.cflags:
            if flag.startswith("-std="):
                return flag[len("-std="):]
        return None

    def command_line_macros(self) -> set[str]:
        """构建、测试、覆盖率命令和 cflags 里用 -D / -U 指定的宏名（这些名字不能被改）。"""
        texts = [self.build_cmd, self.test_cmd or "", self.coverage.get("build_cmd") or "",
                 " ".join(self.cflags)]
        names: set[str] = set()
        for text in texts:
            names.update(_DEFINE_RE.findall(text))
        return names

    @property
    def repo_dir_name(self) -> str:
        """url 末段的仓库名（本地镜像目录里按它放克隆），没有 url 时用 spec 名。"""
        stem = self.url.rstrip("/").rsplit("/", 1)[-1]
        stem = stem[:-4] if stem.endswith(".git") else stem
        return stem or self.name

    def passes(self, rc: int, stdout: str) -> bool:
        """按 pass_criteria 判断一次构建 + 测试是否通过。"""
        crit = self.pass_criteria or {"exit_code": 0}
        if rc != crit.get("exit_code", 0):
            return False
        pattern = crit.get("stdout_regex")
        if pattern and not re.search(pattern, stdout, re.MULTILINE):
            return False
        pattern = crit.get("stdout_not_regex")
        if pattern and re.search(pattern, stdout, re.MULTILINE):
            return False
        return True

    @property
    def output_judged(self) -> bool:
        """通过判据里有没有靠输出文字的部分（stdout_regex / stdout_not_regex）。"""
        crit = self.pass_criteria or {}
        return bool(crit.get("stdout_regex") or crit.get("stdout_not_regex"))

    def judge(self, rc: int, stdout: str, *, truncated: bool = False) -> tuple[bool, bool]:
        """判定一次运行是否通过 → (是否通过, 是否因输出被截断而拒绝)。

        沙箱把超过 64 KB 的输出从中间截断；靠输出文字做的判据（例如 ``stdout_not_regex`` 找 FAILED 行）
        如果恰好落在被丢弃的中间部分就会漏判，所以输出被截断时一律不认通过。基线（未改动的项目）的输出都远小于上限。
        """
        ok = self.passes(rc, stdout)
        if truncated and self.output_judged:
            return False, ok
        return ok, False

    def coverage_command(self) -> str:
        """覆盖率构建用的命令：spec 单独写了 coverage.build_cmd 就用它，否则同 run_command()。"""
        return self.coverage.get("build_cmd") or self.run_command()

    def run_command(self) -> str:
        """构建 + 测试的完整命令：build_cmd；有 test_cmd 时用 && 接在后面（构建失败就不跑测试）。"""
        if self.test_cmd:
            return f"{{\n{self.build_cmd}\n}} && {{\n{self.test_cmd}\n}}"
        return self.build_cmd


# ----------------------------------------------------------------------
# 校验与加载
# ----------------------------------------------------------------------
def _is_str_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(x, str) and x for x in value)


def _bad_path(p: str) -> bool:
    return p.startswith("/") or ".." in PurePosixPath(p).parts


def validate_spec(spec: CorpusSpec, strict: bool = False) -> list[str]:
    """返回问题列表，空表示合法。只检查结构，不碰文件系统。

    strict=True 用于正式语料：url / license 必须非空，commit 必须已钉成 40 位；
    测试夹具（本地小项目）用默认的宽松模式。
    """
    errs: list[str] = []
    if not isinstance(spec.name, str) or not SPEC_NAME_RE.match(spec.name):
        errs.append(f"name 不合法: {spec.name!r}")
    if not isinstance(spec.url, str) or (strict and not spec.url):
        errs.append("url 缺失")
    if not isinstance(spec.commit, str) or (spec.commit and not COMMIT_RE.match(spec.commit)):
        errs.append(f"commit 必须是空串或 40 位小写十六进制: {spec.commit!r}")
    elif strict and not spec.commit:
        errs.append("commit 未钉")
    if not isinstance(spec.license, str) or (strict and not spec.license):
        errs.append("license 缺失")
    for key in ("project_files", "target_files"):
        value = getattr(spec, key)
        if not _is_str_list(value) or not value:
            errs.append(f"{key} 必须是非空的字符串列表")
        elif any(_bad_path(p) for p in value):
            errs.append(f"{key} 含绝对路径或 ..")
    for key in ("cflags", "vendored", "keep_names", "allow_words", "exclude_targets"):
        if not _is_str_list(getattr(spec, key)):
            errs.append(f"{key} 必须是字符串列表")
    if _is_str_list(spec.vendored) and any(_bad_path(p) for p in spec.vendored):
        errs.append("vendored 含绝对路径或 ..")
    if not isinstance(spec.build_cmd, str) or not spec.build_cmd.strip():
        errs.append("build_cmd 为空")
    if spec.test_cmd is not None and (not isinstance(spec.test_cmd, str) or not spec.test_cmd.strip()):
        errs.append("test_cmd 必须是 null 或非空字符串")
    crit = spec.pass_criteria
    if not isinstance(crit, dict) or not isinstance(crit.get("exit_code"), int) \
            or isinstance(crit.get("exit_code"), bool):
        errs.append("pass_criteria 必须是 dict 且含整数 exit_code")
    else:
        extra = set(crit) - PASS_CRITERIA_KEYS
        if extra:
            errs.append(f"pass_criteria 含未知键: {sorted(extra)}")
        for key in ("stdout_regex", "stdout_not_regex"):
            if key in crit:
                try:
                    re.compile(crit[key])
                except (re.error, TypeError) as exc:
                    errs.append(f"pass_criteria.{key} 不是合法正则: {exc}")
    cov = spec.coverage
    if not isinstance(cov, dict):
        errs.append("coverage 必须是 dict")
    else:
        extra = set(cov) - COVERAGE_KEYS
        if extra:
            errs.append(f"coverage 含未知键: {sorted(extra)}")
        env = cov.get("env", {})
        if not isinstance(env, dict) or not all(isinstance(k, str) and isinstance(v, str)
                                                for k, v in env.items()):
            errs.append("coverage.env 必须是 str->str 的 dict")
        if "build_cmd" in cov and not (isinstance(cov["build_cmd"], str) and cov["build_cmd"].strip()):
            errs.append("coverage.build_cmd 必须是非空字符串")
    if not isinstance(spec.timeout_s, int) or isinstance(spec.timeout_s, bool) or spec.timeout_s <= 0:
        errs.append("timeout_s 必须是正整数")
    if not isinstance(spec.notes, str):
        errs.append("notes 必须是字符串")
    return errs


def load_spec_file(path: Path) -> CorpusSpec:
    """读任意路径的 spec JSON（测试夹具用），并做结构校验。"""
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SpecError(f"{path}: JSON 解析失败: {exc}") from exc
    if not isinstance(data, dict):
        raise SpecError(f"{path}: 顶层必须是对象")
    spec = CorpusSpec.from_dict(data)
    errs = validate_spec(spec)
    if errs:
        raise SpecError(f"{path}: " + "; ".join(errs))
    return spec


def load_spec(name: str) -> CorpusSpec:
    """按名字读 ``SPEC_DIR/<name>.json``；传入的是 .json 路径（含目录分隔符）时直接读该文件，方便测试夹具。"""
    if name.endswith(".json") or os.sep in name:
        return load_spec_file(Path(name))
    path = SPEC_DIR / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(f"找不到语料 spec: {path}（可用: {', '.join(list_specs()) or '无'}）")
    spec = load_spec_file(path)
    if spec.name != name:
        raise SpecError(f"{path}: name 字段 {spec.name!r} 与文件名不一致")
    return spec


def list_specs() -> list[str]:
    if not SPEC_DIR.is_dir():
        return []
    return sorted(p.stem for p in SPEC_DIR.glob("*.json"))


# ----------------------------------------------------------------------
# 文件展开
# ----------------------------------------------------------------------
def expand_patterns(patterns: list[str], root: Path, *, must_exist: bool = False, warn: bool = False) -> list[str]:
    """展开 glob 模式，返回相对 root 的 POSIX 路径（规范化、去重、稳定排序），不含 .git 下的内容，只含普通文件。

    全项目唯一的 glob 展开实现。``warn``：模式一个文件都没匹配上时记一条警告；
    ``must_exist``：不含通配符的字面路径缺失时直接抛 FileNotFoundError。
    """
    root = Path(root)
    found: set[str] = set()
    for pat in patterns:
        files = []
        for rel in glob.glob(pat, root_dir=str(root), recursive=True):
            if ".git" in PurePosixPath(Path(rel).as_posix()).parts:
                continue
            if (root / rel).is_file():
                files.append(rel)
        if not files and (warn or must_exist):
            msg = f"模式 {pat!r} 在 {root} 下没有匹配到文件"
            if must_exist and not glob.has_magic(pat):
                raise FileNotFoundError(msg)
            log.warning(msg)
        found.update(Path(os.path.normpath(rel)).as_posix() for rel in files)
    return sorted(found)


def expand_files(spec: CorpusSpec, root: Path) -> list[Path]:
    """展开 project_files 的 glob：去重、按相对路径排序，返回 ``root / 相对路径``。

    只返回普通文件，不含 .git 下的内容。要拿相对路径用 ``p.relative_to(root)``。
    """
    return [Path(root) / r for r in expand_patterns(spec.project_files, root)]


def expand_target_files(spec: CorpusSpec, root: Path) -> list[Path]:
    """展开 target_files（语义同 expand_files）。"""
    return [Path(root) / r for r in expand_patterns(spec.target_files, root)]


def tree_digest(root: Path, exclude: tuple[str, ...] = (".git",)) -> str:
    """源码树内容哈希：按相对路径排序，对（路径，内容哈希）整体取 sha256。"""
    root = Path(root)
    h = hashlib.sha256()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in exclude)
        for fn in sorted(filenames):
            full = Path(dirpath) / fn
            rel = full.relative_to(root).as_posix()
            h.update(rel.encode("utf-8") + b"\0")
            if full.is_symlink():
                h.update(b"L" + os.readlink(full).encode("utf-8", "surrogateescape"))
            else:
                h.update(hashlib.sha256(full.read_bytes()).digest())
    return h.hexdigest()


# ----------------------------------------------------------------------
# git 与 prepare
# ----------------------------------------------------------------------
def _env_proxies() -> list[str]:
    """环境变量里配置的代理地址（去重）。"""
    seen: list[str] = []
    for key, val in os.environ.items():
        if key.lower() in _PROXY_ENV_KEYS and val.strip() and val.strip() not in seen:
            seen.append(val.strip())
    return seen


def _proxy_reachable(url: str, timeout: float = 1.0) -> bool:
    """代理地址的 TCP 端口连不连得上（只试连接，不发请求）。解析不了的地址按连得上处理，交给 git 自己报错。"""
    try:
        parts = urllib.parse.urlsplit(url if "://" in url else "http://" + url)
        host, port = parts.hostname, parts.port or (443 if parts.scheme == "https" else 80)
    except ValueError:
        return True
    if not host:
        return True
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@functools.lru_cache(maxsize=1)
def _drop_proxy_for_git() -> bool:
    """git 联网时要不要绕开代理：``CORPUS_KEEP_PROXY=1`` 不绕开，``CORPUS_NO_PROXY=1`` 一定绕开，
    都没设时探测环境变量里的代理——配了但一个都连不上才绕开（连得上就用，需要代理的网络不受影响）。"""
    if os.environ.get("CORPUS_KEEP_PROXY") == "1":
        return False
    if os.environ.get("CORPUS_NO_PROXY") == "1":
        return True
    proxies = _env_proxies()
    if proxies and not any(_proxy_reachable(u) for u in proxies):
        log.info("环境变量里的代理（%s）都连不上，git 联网时忽略代理（CORPUS_KEEP_PROXY=1 可强制保留）",
                 ", ".join(proxies))
        return True
    return False


def _git_env() -> dict[str, str]:
    env = dict(os.environ)
    if _drop_proxy_for_git_now():
        for key in list(env):
            if key.lower() in _PROXY_ENV_KEYS:
                del env[key]
    env["GIT_TERMINAL_PROMPT"] = "0"
    return env


def _drop_proxy_for_git_now() -> bool:
    """每次调用都重新读两个强制开关（测试和长进程里会改环境变量），探测结果才用缓存。"""
    if os.environ.get("CORPUS_KEEP_PROXY") == "1":
        return False
    if os.environ.get("CORPUS_NO_PROXY") == "1":
        return True
    return _drop_proxy_for_git()


class GitTimeout(RuntimeError):
    """git 命令超时（联网命令在连不上 GitHub 时会一直卡住）。"""


def _git(args: list[str], cwd: Path | None = None, network: bool = False,
         check: bool = True, timeout: float | None = None) -> subprocess.CompletedProcess:
    """跑 git。联网命令默认 180 s 超时（CORPUS_GIT_TIMEOUT 可改），本地命令 600 s。"""
    # core.hooksPath 指到 /dev/null：语料仓库里即使带了 hook 脚本，reset / checkout 时也不会在宿主机上执行
    cmd = ["git", "-c", "core.hooksPath=/dev/null"]
    if network and _drop_proxy_for_git_now():
        cmd += GIT_NOPROXY_ARGS
    cmd += args
    if timeout is None:
        timeout = float(os.environ.get("CORPUS_GIT_TIMEOUT", "180")) if network else 600.0
    try:
        res = subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=_git_env(), text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise GitTimeout(f"git {' '.join(args)} 超过 {timeout:g} s 仍未完成（网络不通？）") from exc
    if check and res.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} 失败(rc={res.returncode}): {res.stderr.strip()[:400]}")
    return res


def _head(repo: Path) -> str:
    return _git(["rev-parse", "HEAD"], cwd=repo).stdout.strip()


def _norm_url(url: str) -> str:
    url = url.strip().rstrip("/")
    if url.endswith(".git"):
        url = url[:-4]
    return url.lower()


def mirror_dir() -> Path | None:
    """本地镜像根目录：只认环境变量 CORPUS_MIRROR，没设置返回 None（不猜任何本机目录）。"""
    env = os.environ.get("CORPUS_MIRROR")
    return Path(env) if env else None


def _mirror_repo(spec: CorpusSpec) -> Path | None:
    """在镜像目录里找这个 spec 对应的 git 仓库（目录名取 url 末段，兼容大小写）。"""
    base = mirror_dir()
    if base is None or not base.is_dir():
        return None
    wanted = [c for c in (spec.repo_dir_name, spec.name) if c]
    lowered = {c.lower() for c in wanted}
    names = list(dict.fromkeys(
        wanted + sorted(d.name for d in base.iterdir() if d.is_dir() and d.name.lower() in lowered)))
    for cand in names:
        repo = base / cand
        if (repo / ".git").exists():
            res = _git(["remote", "get-url", "origin"], cwd=repo, check=False)
            if res.returncode == 0 and spec.url and _norm_url(res.stdout) != _norm_url(spec.url):
                log.warning("镜像 %s 的 origin(%s) 与 spec.url(%s) 不一致，不使用", repo,
                            res.stdout.strip(), spec.url)
                continue
            return repo
    return None


def _make_pristine(repo: Path) -> None:
    """丢弃工作区里的一切改动和未跟踪文件，保证树和 HEAD 完全一致。"""
    status = _git(["status", "--porcelain", "--ignored"], cwd=repo).stdout.strip()
    if status:
        log.info("%s 工作区不干净，恢复到 HEAD:\n%s", repo.name, status[:300])
        _git(["reset", "--hard", "-q"], cwd=repo)
        _git(["clean", "-fdxq"], cwd=repo)


def _checkout(repo: Path, commit: str) -> None:
    has = _git(["cat-file", "-e", f"{commit}^{{commit}}"], cwd=repo, check=False).returncode == 0
    if not has:
        raise LookupError(f"{repo} 里没有提交 {commit}")
    _git(["checkout", "-q", "--detach", commit], cwd=repo)


def _fill_from_mirror(spec: CorpusSpec, mirror: Path, staging: Path) -> None:
    shutil.copytree(mirror, staging, symlinks=True)
    if spec.commit and _head(staging) != spec.commit:
        _checkout(staging, spec.commit)
    _make_pristine(staging)


def _fill_from_clone(spec: CorpusSpec, staging: Path) -> None:
    if not spec.commit:
        _git(["clone", "-q", "--depth", "1", spec.url, str(staging)], network=True)
        return
    # 钉了提交：优先浅取单个提交，服务器不允许按 SHA 取时退回完整克隆
    staging.mkdir(parents=True, exist_ok=True)
    _git(["init", "-q"], cwd=staging)
    _git(["remote", "add", "origin", spec.url], cwd=staging)
    try:
        _git(["fetch", "-q", "--depth", "1", "origin", spec.commit], cwd=staging, network=True)
        _git(["checkout", "-q", "--detach", "FETCH_HEAD"], cwd=staging)
    except GitTimeout:
        raise                      # 网络不通，再试完整克隆只会再等一轮
    except RuntimeError as exc:
        log.warning("按 SHA 浅取失败(%s)，改用完整克隆", exc)
        shutil.rmtree(staging)
        _git(["clone", "-q", spec.url, str(staging)], network=True)
        _checkout(staging, spec.commit)


def _persist_commit(spec: CorpusSpec) -> None:
    """spec 文件里 commit 为空时，把解析到的 HEAD 写回去（只改这一行，不动其它格式）。"""
    path = SPEC_DIR / f"{spec.name}.json"
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8")
    new, n = re.subn(r'("commit"\s*:\s*)""', rf'\g<1>"{spec.commit}"', text, count=1)
    if n:
        path.write_text(new, encoding="utf-8")
        log.info("已把 %s 的 commit 写入 %s", spec.name, path)


def _reusable(dest: Path, spec: CorpusSpec) -> bool:
    if not (dest / ".git").exists():
        return False
    res = _git(["rev-parse", "HEAD"], cwd=dest, check=False)
    if res.returncode != 0:
        return False
    head = res.stdout.strip()
    if spec.commit and head != spec.commit:
        log.info("%s 现有提交 %s 与 spec 钉的 %s 不同，重新准备", spec.name, head[:12], spec.commit[:12])
        return False
    return True


def prepare(spec: CorpusSpec, root: Path = CORPUS_SRC) -> Path:
    """准备 ``root/<name>`` 下的源码并返回该目录。

    已存在且提交一致就复用（顺手把工作区恢复成干净状态）；否则先复制本地镜像，镜像没有或提交对不上
    再按 url 克隆。spec.commit 为空时，用解析到的 HEAD 填入（内存里的 spec 和 spec 文件都会更新）。
    先在 ``root`` 下的暂存目录里准备好，再整体改名，所以中途失败不会留下半成品。
    """
    root = Path(root)
    dest = root / spec.name
    if _reusable(dest, spec):
        _make_pristine(dest)
        if not spec.commit:
            spec.commit = _head(dest)
            _persist_commit(spec)
        return dest

    root.mkdir(parents=True, exist_ok=True)
    holder = Path(tempfile.mkdtemp(prefix=f".{spec.name}.", dir=root))
    staging = holder / "src"
    try:
        mirror = _mirror_repo(spec)
        done = False
        if mirror is not None:
            try:
                _fill_from_mirror(spec, mirror, staging)
                log.info("%s: 复制本地镜像 %s", spec.name, mirror)
                done = True
            except (LookupError, RuntimeError, OSError) as exc:
                log.warning("%s: 本地镜像不可用(%s)，改为克隆", spec.name, exc)
                shutil.rmtree(staging, ignore_errors=True)
        if not done:
            try:
                _fill_from_clone(spec, staging)
            except LookupError as exc:
                raise RuntimeError(f"{spec.name}: {exc}") from exc
            _make_pristine(staging)
            log.info("%s: 已从 %s 克隆", spec.name, spec.url)
        head = _head(staging)
        if spec.commit and head != spec.commit:
            raise RuntimeError(f"{spec.name}: 准备好的提交 {head} 与 spec 钉的 {spec.commit} 不一致")
        # 并发保护：暂存期间别的进程可能已经准备好了同一份，能复用就不要把人家正在读的目录换掉
        if _reusable(dest, spec):
            log.info("%s: 已被其它进程准备好，丢弃暂存副本", spec.name)
        else:
            if dest.exists():
                shutil.rmtree(dest)
            os.rename(staging, dest)
    finally:
        shutil.rmtree(holder, ignore_errors=True)
    if not spec.commit:
        spec.commit = _head(dest)
        _persist_commit(spec)
    return dest


# ----------------------------------------------------------------------
# 工作副本与构建校验
# ----------------------------------------------------------------------
def materialize(src_root: Path, dest: Path) -> Path:
    """复制一份工作副本到 dest（不含 .git），返回 dest。

    dest 已存在时先清掉再复制，保证结果和 src_root 完全一致（不残留上次构建产物）。
    """
    src_root, dest = Path(src_root), Path(dest)
    if not src_root.is_dir():
        raise FileNotFoundError(f"源码目录不存在: {src_root}")
    src_real, dest_real = src_root.resolve(), dest.resolve()
    if dest_real == src_real or dest_real in src_real.parents or src_real in dest_real.parents:
        raise ValueError(f"dest 与 src_root 互相包含，拒绝复制: {dest} / {src_root}")
    if dest.is_symlink() or dest.is_file():
        dest.unlink()
    elif dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src_root, dest, symlinks=True, ignore=shutil.ignore_patterns(".git"))
    return dest


def verify_build(spec: CorpusSpec, src_root: Path, sandbox, env: dict | None = None,
                 work_root: Path | None = None, keep: bool = False):
    """在沙箱里对 src_root 的干净副本跑 build_cmd（和 test_cmd），按 pass_criteria 判定。

    返回 ``(ok, RunResult)``。副本放在 ``work_root``（默认 ``store.default_work_root()``），跑完删除，keep=True 保留。
    """
    base = Path(work_root) if work_root else default_work_root()
    base.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix=f"verify-{spec.name}-", dir=base))
    try:
        materialize(src_root, workdir)
        res = sandbox.run(workdir, spec.run_command(), env=env, timeout_s=spec.timeout_s)
        raise_if_docker_failed(res)
        ok, _ = spec.judge(res.rc, res.stdout, truncated=getattr(res, "stdout_truncated", False))
        return ok and not res.timed_out, res
    finally:
        if not keep:
            shutil.rmtree(workdir, ignore_errors=True)
