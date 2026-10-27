"""libclang 配置：统一设置库文件路径，所有解析代码从这里拿 Index 和解析参数。

库文件路径优先取环境变量 LIBCLANG_PATH，否则在常见位置里找；
解析选项默认带 PARSE_DETAILED_PROCESSING_RECORD，这样宏定义与宏展开才会出现在 AST 里。

这里也是 libclang 解析参数的**唯一出口**（以前 corpus 与 extract 各有一套，同一份 cflags 得到的参数不同，
符号混淆和结构 / 语义混淆看到的 AST 口径也就不一致）：

* :func:`clang_args`：把 ``spec.cflags`` 换成适合 libclang 的参数；
* :func:`builtin_include_args`：libclang 找不到编译器内置头（stddef.h / stdarg.h 等）时，借宿主编译器的目录补上。
  只装了 libclang 运行库、没装 libclang-common-dev 的环境会遇到这个问题；装全了的环境返回空元组。
  环境变量 ``BENCH_CLANG_INCLUDE`` 可显式指定目录。
"""
from __future__ import annotations

import functools
import glob
import logging
import os
import subprocess
from pathlib import Path
from typing import Sequence

import clang.cindex as ci

log = logging.getLogger(__name__)

LIBCLANG_CANDIDATES = [
    "/usr/lib/llvm-14/lib/libclang.so.1",
    "/usr/lib/x86_64-linux-gnu/libclang-14.so.1",
    "/usr/lib/llvm-14/lib/libclang.so",
]

DEFAULT_PARSE_OPTIONS = ci.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD

_configured = False
_library: str | None = None


def configure() -> None:
    """只执行一次：定位 libclang 并告诉 cindex。"""
    global _configured, _library
    if _configured:
        return
    path = os.environ.get("LIBCLANG_PATH")
    if not path:
        for cand in LIBCLANG_CANDIDATES:
            if Path(cand).exists():
                path = cand
                break
    if path:
        ci.Config.set_library_file(path)
    _library = path
    _configured = True


def index() -> ci.Index:
    """返回配置好的 Index，调用方用 index().parse(...)。"""
    configure()
    return ci.Index.create()


def kind_of(c: ci.Cursor) -> ci.CursorKind | None:
    """cursor.kind；绑定不认识的新 cursor 种类会抛 ValueError，这里当作 None。全项目共用这一个包装。"""
    try:
        return c.kind
    except ValueError:
        return None


# --------------------------------------------------------------------------- 编译器内置头

def _probe_clean(extra: Sequence[str]) -> bool:
    """用一小段包含内置头的代码试探：没有致命诊断就说明这套参数够用。"""
    src = "#include <stddef.h>\n#include <stdarg.h>\n#include <limits.h>\n#include <stdint.h>\n#include <float.h>\nint probe_ok;\n"
    tu = index().parse("__bench_probe__.c", args=["-x", "c", *extra],
                       unsaved_files=[("__bench_probe__.c", src)], options=0)
    return not any(d.severity >= ci.Diagnostic.Fatal for d in tu.diagnostics)


def _compiler_include_dirs() -> list[str]:
    """候选的编译器内置头目录：显式配置 > clang 资源目录 > gcc / cc 的 include 目录。"""
    cands: list[str] = []
    env = os.environ.get("BENCH_CLANG_INCLUDE")
    if env:
        cands += [p for p in env.split(os.pathsep) if p]
    for pat in ("/usr/lib/llvm-*/lib/clang/*/include", "/usr/lib/clang/*/include", "/usr/local/lib/clang/*/include"):
        cands += sorted(glob.glob(pat), reverse=True)
    for cmd in (["gcc", "-print-file-name=include"], ["cc", "-print-file-name=include"], ["clang", "-print-resource-dir"]):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            continue
        if out and os.path.isabs(out):
            cands.append(os.path.join(out, "include") if cmd[0] == "clang" else out)
    cands += sorted(glob.glob("/usr/lib/gcc/*/*/include"), reverse=True)
    seen: set[str] = set()
    return [c for c in cands if os.path.isdir(c) and not (c in seen or seen.add(c))]


@functools.lru_cache(maxsize=1)
def builtin_include_args() -> tuple[str, ...]:
    """让 libclang 能找到 stddef.h / stdarg.h / limits.h 等编译器内置头所需的额外参数。

    先试探 libclang 自己能不能找到（装了 libclang-common 时可以），找不到再逐个尝试候选目录。
    借来的目录来自宿主机的编译器，可能与沙箱里 gcc:13 的内置头版本不同；这只影响「哪些系统头拼写被当成
    系统实体而不改名」的判定，所以符号混淆的结果记录里要带上这个参数（见 ``bench.obfuscate``）。
    """
    if _probe_clean([]):
        return ()
    for cand in _compiler_include_dirs():
        if _probe_clean(["-isystem", cand]):
            log.info("libclang 找不到内置头，改用 %s", cand)
            return ("-isystem", cand)
    log.warning("libclang 找不到编译器内置头，也没有可用的候选目录；解析可能出现致命诊断（可设 BENCH_CLANG_INCLUDE）")
    return ()


def environment() -> dict[str, str]:
    """宿主 libclang 的环境摘要（不含本机绝对路径）：python 绑定版本、库文件名、借用的内置头目录的内容摘要。

    符号混淆的排除集靠 libclang 在宿主机上解析：系统头里声明过的拼写一律不改名，而宿主的系统头 / 借来的 gcc 内置头
    与沙箱里 gcc:13 的并不相同，换一台机器或升级系统后改名表可能不同——所以混淆结果的输入里要带上这份摘要，
    环境变了旧副本就会被判定为过期。
    """
    import hashlib
    configure()
    inc = builtin_include_args()
    listing = ""
    if inc:
        d = Path(inc[-1])
        entries = sorted((f.name, f.stat().st_size) for f in d.glob("*.h")) if d.is_dir() else []
        listing = hashlib.sha256(repr(entries).encode()).hexdigest()[:16]
    try:
        from importlib.metadata import version
        binding = version("clang")
    except Exception:  # noqa: BLE001 - 包元数据缺失不影响功能
        binding = ""
    return {"python_binding": binding, "library": Path(_library).name if _library else "", "builtin_include": listing}


# --------------------------------------------------------------------------- 解析参数

_DIR_OPTS = ("-I", "-isystem", "-iquote", "-idirafter")   # 参数是目录：相对路径一律按 src_root 展开
_FILE_OPTS = ("-include", "-imacros")                      # 参数是文件：只有 src_root 下真有这个文件才展开（-include stdio.h 要保持原样）
_DROP_EXACT = {"-c", "-pipe", "-MD", "-MMD", "-MP", "-pedantic-errors"}
_DROP_WITH_ARG = {"-o", "-MF", "-MT", "-MQ"}
_DROP_PREFIX = ("-l", "-L", "-Wl,", "-Werror")


def clang_args(cflags: Sequence[str], root: Path) -> list[str]:
    """把 spec.cflags 换成适合 libclang 的参数：

    相对的包含路径按 src_root 展开成绝对路径（不依赖当前工作目录）；
    去掉链接、依赖输出、-Werror 这类与解析无关或会误伤的选项；
    固定按 C 解析（头文件目标也能解析）、打开全部注释解析、压掉警告、不限错误数。
    """
    root = Path(root)
    args = ["-x", "c", "-fparse-all-comments", "-w", "-ferror-limit=0"]

    def absolutize(p: str) -> str:
        return p if os.path.isabs(p) else str((root / p).resolve())

    def absolutize_file(p: str) -> str:
        return absolutize(p) if (root / p).is_file() else p

    it = iter(cflags)
    for tok in it:
        if tok in _DROP_EXACT or tok.startswith(_DROP_PREFIX):
            continue
        if tok in _DROP_WITH_ARG:
            next(it, None)
            continue
        if tok in _DIR_OPTS or tok in _FILE_OPTS:
            val = next(it, None)
            if val is not None:
                args += [tok, absolutize_file(val) if tok in _FILE_OPTS else absolutize(val)]
            continue
        if tok.startswith("-I") and len(tok) > 2:
            args.append("-I" + absolutize(tok[2:]))
            continue
        args.append(tok)
    args += builtin_include_args()
    return args
