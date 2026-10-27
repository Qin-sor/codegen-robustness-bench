"""L1 符号混淆（设计文档 §4.6）：把项目里自己声明的标识符整体改成无语义的新名字。

对外接口（签名与设计文档一致）::

    collect_project_identifiers(spec, src_root) -> set[str]   # 要改名的拼写
    build_rename_map(names, seed) -> dict[str, str]           # 双射：f_<n> t_<n> v_<n> M_<n> m_<n>
    apply_renames(spec, src_root, mapping) -> dict[str, int]  # 逐文件 token 级替换，返回每文件替换数

在此之外的补充（都是新增，不改上面三个的用法）：

* ``collect_project_identifiers`` 返回 ``set`` 的子类 :class:`IdentifierSet`，带 ``kinds``（拼写 -> 种类，
  ``build_rename_map`` 靠它选前缀）、``reserved``（新名字不得取用的拼写）、``excluded``（原因 -> 被排除的项目拼写）、
  ``stats``（统计，含 ``excluded_stringify``）；关键字参数 ``strict_stringify`` 见下；
* :func:`obfuscate_symbols` 把三步串起来，原地改写 ``src_root``，返回 :class:`SymbolsResult` （映射表、stats、每文件替换数），
  供 ``obfuscate(level="L1")`` 直接调用（verify 与「非 strict 失败后退回 strict」的回退逻辑只在 ``bench.obfuscate.obfuscate`` 里有一份）；
* ``load_file_info`` / ``rewrite_bytes``：取单个文件的 token 流、不落盘地按映射表改写，只给测试和调试用，生产路径不调用；
* ``build_rename_map(names, seed, kinds=..., reserved=...)``：别的模块要给新插入的变量取「像 L1 风格」的新名字时，
  可传入 ``kinds={"_k": "var"}`` 与已有的全部拼写作为 ``reserved``。

收集（collect）
---------------
* 遍历 ``project_files`` 里每个文件对应的 TU（头文件也单独当根解析），取声明 cursor
  FUNCTION_DECL / VAR_DECL / PARM_DECL / FIELD_DECL / STRUCT_DECL / UNION_DECL / ENUM_DECL /
  ENUM_CONSTANT_DECL / TYPEDEF_DECL / LABEL_STMT / MACRO_DEFINITION，且 ``location.file`` 在可改名文件内
  （``spec.vendored`` 里的第三方文件只解析、不改名）。
* 解析按两套参数各走一遍：``spec.cflags``，以及再加上构建命令里出现过的全部 ``-D`` 的「并集配置」，
  这样 ``#ifdef JSMN_PARENT_LINKS`` 之类条件编译块里的声明也能看见。
* 排除集（绝不改名）：
  - C 关键字、``main`` 等入口名、以 ``__`` 或 ``_大写`` 开头的保留拼写（含 ``_GNU_SOURCE`` 这类特性开关）；
  - 所有系统头 / vendored 文件里声明过的拼写（``location.file`` 不在可改名文件内：函数、变量、类型、
    枚举常量、字段、宏）。系统头里只看文件作用域名字和成员，函数参数、局部变量不算（不会和项目代码互相影响）；
  - 被 vendored 代码引用的项目实体、vendored 宏体里的自由标识符、vendored 条件编译指令里出现的标识符；
  - 以隐式声明方式调用的函数（C89 里没 include 就调用的 libc 函数），除非项目自己有同名函数定义；
  - 项目里只有声明、没有定义的函数 / 变量（链接到项目外的符号，改名就链接不上）；
  - 构建命令和 Makefile 里用 ``-D`` / ``-U`` 指定的宏、spec.keep_names、``NDEBUG`` 这类被系统头消费的开关；
  - 出现在 ``#pragma`` 行里的拼写；
  - 宏体含 ``#`` 或 ``##`` 的宏所涉及的拼写，记进 ``stats["excluded_stringify"]``：宏名、形参、宏体里的标识符；
    ``##`` 宏的调用实参、由 ``##`` 拼出来的名字（声明或引用该名字的 cursor 落在拼接宏的展开里）、
    X-macro 列表里被拼接宏吃进去的列表项（``LIST(AS_ENUM)`` 的 ``X(RED, 1)`` 里的 RED）。
* ``#`` 宏（字符串化）的调用实参分两种处理：**目标文件（``spec.target_files``，库自己的源码）里的一律排除**，
  库代码里被字符串化的名字会变成运行时输出的一部分，改名等于改了库的行为，而项目测试多半不逐字比对这类文字，
  verify 发现不了；**测试文件里的**默认照常改名——字符串化的结果只出现在失败提示之类的文字里，而测试里大量
  ``CHECK(f(x) == 1)`` 这样的断言会让所有被调函数都改不了。后一部分如实记在 ``stats["stringified_names_renamed"]``
  （报告的「混淆副本」表里也有）；传 ``strict_stringify=True`` 则连同它们一起排除（输出文字完全不变，
  代价是排除数大幅上升）。项目测试若逐字比对了字符串化的输出，用 strict。
* 同一拼写全项目映射到同一个新名字（按拼写的双射）。

替换（apply）
-------------
逐文件 ``tu.get_tokens(extent=整文件)``，只改 ``TokenKind.IDENTIFIER`` 且拼写在映射表里的 token，
按字节偏移从后往前写回；字符串字面量与注释原样保留（所以注释里还留着原名）。不碰的 token：
``#include <...>`` 里的头文件名（原始词法器会把 ``stdio`` ``h`` 当成两个标识符）、``#pragma`` / ``#error`` /
``#warning`` / ``#line`` 行、``__attribute__((...))`` / ``__declspec(...)`` 里的属性名。改名后重新解析每个
target_file（两套配置），severity >= 3 的诊断数不得比改名前多（改名前为 0 就必须仍为 0），否则抛
:class:`RenameError` 并把所有文件恢复原样。新名字与源码里已有的任何标识符重名、映射表不是双射、
新名字是关键字，抛 ``ValueError``（此时文件未动）。

环境变量：无；libclang 路径由 ``bench.clangcfg`` 统一处理。
"""
from __future__ import annotations

import bisect
import logging
import os
import random
import re
import shlex
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import clang.cindex as ci

from bench import clangcfg
from bench.clangcfg import clang_args, kind_of
from bench.corpus import CorpusSpec, expand_patterns

log = logging.getLogger(__name__)

CK = ci.CursorKind
TK = ci.TokenKind

__all__ = [
    "IdentifierSet",
    "RenameError",
    "SymbolError",
    "SymbolsResult",
    "analyze_project",
    "apply_renames",
    "load_file_info",
    "rewrite_bytes",
    "build_rename_map",
    "collect_project_identifiers",
    "obfuscate_symbols",
    "KIND_PREFIX",
    "C_KEYWORDS",
]

# --------------------------------------------------------------------------- 常量

#: C89 ~ C23 的全部关键字，外加 GNU 的 asm / typeof 和预处理的 defined（它们不可能是项目实体名）。
#: 与 bench.similarity.C_KEYWORDS 是两张不同的表：这里是「不能改名 / 不能当新名字」的拼写（含 C23 的 bool / true / false /
#: nullptr 等，不含 __asm__ 这类双下划线拼写，它们由保留前缀规则排除），相似度那张只关心语法结构记号
C_KEYWORDS = frozenset(
    """auto break case char const continue default do double else enum extern float for goto if int long
    register return short signed sizeof static struct switch typedef union unsigned void volatile while
    inline restrict _Bool _Complex _Imaginary _Alignas _Alignof _Atomic _Generic _Noreturn _Static_assert
    _Thread_local alignas alignof bool constexpr false nullptr static_assert thread_local true typeof
    typeof_unqual asm defined""".split()
)
#: 程序入口 / 平台入口，名字由运行时约定
ENTRY_NAMES = frozenset({"main", "wmain", "WinMain", "wWinMain", "DllMain", "_start"})
#: 被系统头或编译器「消费」而不是「声明」的开关与常见平台宏：项目里即使 #define 了也不能改名
KNOWN_EXTERNAL_NAMES = frozenset(
    {"NDEBUG", "WIN32", "WIN64", "NULL", "EOF", "errno", "linux", "unix", "i386", "WINAPI", "CALLBACK",
     "environ", "optarg", "optind", "opterr", "optopt"}
)
#: 种类 -> 新名字前缀
KIND_PREFIX = {"func": "f", "type": "t", "var": "v", "macro": "M", "field": "m"}
#: 同一拼写同时有多种实体时，取优先级最高的种类决定前缀
KIND_PRIORITY = ("macro", "func", "type", "field", "var")

_DECL_LABEL = {
    CK.FUNCTION_DECL: "func",
    CK.VAR_DECL: "var",
    CK.PARM_DECL: "var",
    CK.FIELD_DECL: "field",
    CK.STRUCT_DECL: "type",
    CK.UNION_DECL: "type",
    CK.ENUM_DECL: "type",
    CK.ENUM_CONSTANT_DECL: "var",
    CK.TYPEDEF_DECL: "type",
    CK.LABEL_STMT: "var",
    CK.MACRO_DEFINITION: "macro",
}
_REF_KINDS = frozenset({CK.DECL_REF_EXPR, CK.MEMBER_REF_EXPR, CK.MEMBER_REF, CK.TYPE_REF, CK.CALL_EXPR,
                        CK.LABEL_REF, CK.MACRO_INSTANTIATION})
#: 系统头里只记这些「会被项目代码直接按名字引用」的声明；成员（字段、枚举常量）从类型声明往下取
_EXT_TOP_KINDS = frozenset({CK.FUNCTION_DECL, CK.VAR_DECL})
_EXT_TYPE_KINDS = frozenset({CK.TYPEDEF_DECL, CK.STRUCT_DECL, CK.UNION_DECL, CK.ENUM_DECL, CK.FIELD_DECL,
                             CK.ENUM_CONSTANT_DECL})

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_RESERVED_RE = re.compile(r"^(?:__|_[A-Z])")
_C_EXTS = (".c", ".h", ".inc", ".def")
_NL_RE = re.compile(rb"\n")
_WS_BYTES = b" \t\r\f\v"
_DFLAG_RE = re.compile(r"(?<![\w-])(-D\s*[A-Za-z_]\w*(?:=\S+)?)")
_DNAME_RE = re.compile(r"(?<![\w-])-[DU]\s*([A-Za-z_]\w*)")
_SCRIPT_RE = re.compile(r"^(?:GNUmakefile|[Mm]akefile.*|.*\.mk|.*\.sh|CMakeLists\.txt|.*\.cmake|meson\.build|"
                        r"configure(?:\.ac)?)$")
_ATTR_KEYWORDS = frozenset({"__attribute__", "__attribute", "__declspec"})
_FUNC_NAME_MACROS = frozenset({"__func__", "__FUNCTION__", "__PRETTY_FUNCTION__"})
#: 预处理里的探测函数，括号里是头文件名 / 属性名 / 内建名，不是项目标识符
_PROBE_CALLS = frozenset({"__has_include", "__has_include_next", "__has_attribute", "__has_cpp_attribute",
                          "__has_c_attribute", "__has_builtin", "__has_feature", "__has_extension",
                          "__has_warning"})
_ASM_KEYWORDS = frozenset({"asm", "__asm", "__asm__"})
#: 这些指令后面的内容不是代码，整行不改
_DIRECTIVE_SKIP = frozenset({"pragma", "error", "warning", "line", "ident", "sccs", "assert", "unassert"})
_DIRECTIVE_INCLUDE = frozenset({"include", "include_next", "import"})
#: 互斥统计「被排除」原因时的优先顺序（一个拼写可能同时命中多个原因，只记第一个）
_REASON_ORDER = ("keyword", "entry", "reserved_prefix", "known_external", "external_decl", "implicit_decl",
                 "no_definition", "vendored_use", "cmdline_macro", "keep_names", "stringify", "pragma")


class SymbolError(RuntimeError):
    """符号混淆过程中的不可恢复错误（libclang 无法加载文件等）。"""


class RenameError(SymbolError):
    """改名后重新解析目标文件出现了新的 error 级诊断（文件已恢复原样）。"""

    def __init__(self, message: str, diagnostics: Optional[list[str]] = None) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics or []


# --------------------------------------------------------------------------- spec 小工具


def project_file_lists(spec: CorpusSpec, root: Path) -> tuple[list[str], list[str]]:
    """(可改名文件, vendored 文件)，相对路径（POSIX）、排序、只含 C 源文件 / 头文件。"""
    root = Path(root)
    rels = set(expand_patterns(spec.project_files, root))
    extra = set(expand_patterns(spec.target_files, root)) - rels
    if extra:
        log.warning("target_files 里有文件不在 project_files 中，一并处理: %s", sorted(extra))
        rels |= extra
    renamable, vendored = [], []
    for rel in sorted(rels):
        if not rel.lower().endswith(_C_EXTS):
            continue
        (vendored if spec.is_vendored(rel) else renamable).append(rel)
    return renamable, vendored


def _spec_texts(spec: CorpusSpec) -> list[str]:
    return [spec.build_cmd or "", spec.test_cmd or "", spec.coverage.get("build_cmd") or ""]


def _cmdline_macros(spec: CorpusSpec, root: Path) -> set[str]:
    """构建命令、cflags 以及 src_root 里 Makefile / shell / cmake 脚本中用 -D / -U 指定的宏名。"""
    names: set[str] = set(spec.command_line_macros())
    # Makefile 里的 -D 开关 spec 看不到，扫一遍构建脚本兜底
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != ".git"]
        depth = len(Path(dirpath).relative_to(root).parts)
        if depth > 3:
            dirnames[:] = []
            continue
        for fn_ in filenames:
            if not _SCRIPT_RE.match(fn_):
                continue
            p = Path(dirpath) / fn_
            try:
                if p.stat().st_size > 256 * 1024:
                    continue
                names.update(_DNAME_RE.findall(p.read_text(encoding="utf-8", errors="replace")))
            except OSError as exc:
                log.warning("读构建脚本 %s 失败（%s），其中的 -D 开关不在排除集里", p, exc)
                continue
    return names


def _config_cflags(spec: CorpusSpec) -> list[list[str]]:
    """libclang 要跑的 cflags 配置：基础配置，以及「基础 + 构建命令里全部 -D」的并集配置（有新增才有）。"""
    base = list(spec.cflags)
    have = set(_DNAME_RE.findall(" ".join(base)))
    extra: list[str] = []
    for text in _spec_texts(spec):
        for tok in _DFLAG_RE.findall(text):
            tok = re.sub(r"^-D\s+", "-D", tok)          # "-D NAME" 写法
            try:
                parts = shlex.split(tok)
            except ValueError:
                continue
            for part in parts:
                m = _DNAME_RE.match(part)
                if m and m.group(1) not in have:
                    have.add(m.group(1))
                    extra.append(part if "=" in part else part + "=1")
    return [base] + ([base + extra] if extra else [])


# --------------------------------------------------------------------------- 词法层


@dataclass(slots=True)
class Tok:
    """一个原始 token。kind: id 标识符 / kw 关键字 / punct 标点 / lit 字面量 / cmt 注释。"""

    kind: str
    text: str          # id / kw / punct 的拼写；字面量和注释不取
    start: int         # 字节偏移 [start, end)
    end: int
    line: int = 0      # 逻辑行号（续行与跨行注释不增加）


@dataclass
class MacroDef:
    name: str
    params: list[str]
    body: list[Tok]
    function_like: bool
    cls: str                       # R 可改名文件 / V vendored / E 系统头
    has_hash: bool = False         # 宏体含 #（字符串化）
    has_paste: bool = False        # 宏体含 ##（拼接）


@dataclass
class FileInfo:
    rel: str
    path: Path
    cls: str                                   # R / V
    data: bytes = b""
    toks: list[Tok] = field(default_factory=list)
    protected: set[int] = field(default_factory=set)       # 不得改名的 token 起始偏移
    pragma_ids: set[str] = field(default_factory=set)
    directive_ids: set[str] = field(default_factory=set)   # 条件编译 / define 等指令行里的标识符
    ident_at: dict[int, str] = field(default_factory=dict)
    macros: list[MacroDef] = field(default_factory=list)
    directive_lines: set[int] = field(default_factory=set)
    tainted_paste: set[int] = field(default_factory=set)   # 展开里含 ## 宏的调用起点
    tainted_hash: set[int] = field(default_factory=set)    # 展开里含 # 宏的调用起点
    parse_errors: int = 0


def _lex(tu: ci.TranslationUnit, path: Path, data: bytes) -> list[Tok]:
    """整文件的原始 token 流（含注释）。"""
    if not data:
        return []
    ext = tu.get_extent(str(path), (0, len(data)))
    out: list[Tok] = []
    for t in tu.get_tokens(extent=ext):
        k = t.kind
        e = t.extent
        s, en = e.start.offset, e.end.offset
        if k == TK.IDENTIFIER:
            out.append(Tok("id", t.spelling, s, en))
        elif k == TK.KEYWORD:
            out.append(Tok("kw", t.spelling, s, en))
        elif k == TK.PUNCTUATION:
            out.append(Tok("punct", t.spelling, s, en))
        elif k == TK.LITERAL:
            out.append(Tok("lit", "", s, en))
        else:
            out.append(Tok("cmt", "", s, en))
    return out


def _assign_lines(data: bytes, toks: list[Tok]) -> None:
    """给 token 标逻辑行号：转义换行（反斜杠接换行）和块注释内部的换行都不算行结束。"""
    cmt_starts = [t.start for t in toks if t.kind == "cmt"]
    cmt_ends = [t.end for t in toks if t.kind == "cmt"]
    breaks: list[int] = []
    for m in _NL_RE.finditer(data):
        p = m.start()
        q = p
        while q > 0 and data[q - 1] in _WS_BYTES:
            q -= 1
        if q > 0 and data[q - 1] == 0x5C:           # 行尾反斜杠：续行
            continue
        i = bisect.bisect_right(cmt_starts, p) - 1
        if i >= 0 and p < cmt_ends[i]:              # 注释内部的换行
            continue
        breaks.append(p)
    for t in toks:
        t.line = bisect.bisect_right(breaks, t.start)


def _macro_from_tokens(toks: list[Tok], cls: str) -> Optional[MacroDef]:
    """toks 是 #define 之后、不含注释的 token：宏名、可选的形参表、宏体。"""
    if not toks or toks[0].kind not in ("id", "kw"):
        return None
    name = toks[0].text
    params: list[str] = []
    function_like = False
    k = 1
    if len(toks) > 1 and toks[1].kind == "punct" and toks[1].text == "(" and toks[1].start == toks[0].end:
        function_like = True
        k = 2
        while k < len(toks) and not (toks[k].kind == "punct" and toks[k].text == ")"):
            if toks[k].kind == "id":
                params.append(toks[k].text)
            k += 1
        k += 1
    body = toks[k:]
    has_hash = any(t.kind == "punct" and t.text == "#" for t in body)
    has_paste = any(t.kind == "punct" and t.text == "##" for t in body)
    return MacroDef(name=name, params=params, body=body, function_like=function_like, cls=cls,
                    has_hash=has_hash, has_paste=has_paste)


def _scan_structure(fi: FileInfo) -> None:
    """预处理指令（保护区、pragma、宏定义）与属性名的位置。"""
    toks = fi.toks
    n = len(toks)
    prev_line = -1
    i = 0
    while i < n:
        t = toks[i]
        if t.kind == "cmt":
            i += 1
            continue
        if t.kind == "punct" and t.text == "#" and t.line != prev_line:
            j = i + 1
            while j < n and toks[j].line == t.line:
                j += 1
            _scan_directive(fi, [x for x in toks[i + 1:j] if x.kind != "cmt"], t.line)
            prev_line = t.line
            i = j
            continue
        prev_line = t.line
        i += 1
    _scan_attributes(fi)
    _scan_asm(fi)


def _scan_asm(fi: FileInfo) -> None:
    """asm 语句里 [name] 形式的符号操作数名（和模板字符串里的 %[name] 配对，不是 C 标识符）。"""
    toks = [t for t in fi.toks if t.kind != "cmt"]
    n = len(toks)
    for i, t in enumerate(toks):
        if t.kind not in ("id", "kw") or t.text not in _ASM_KEYWORDS:
            continue
        k = i + 1
        while k < n and toks[k].kind in ("id", "kw") and toks[k].text in ("volatile", "__volatile__", "goto", "inline"):
            k += 1
        if k >= n or toks[k].text != "(" or toks[k].kind != "punct":
            continue
        end = _paren_end(toks, k)
        if end < 0:
            continue
        for m in range(k + 1, end - 2):
            if (toks[m].kind == "punct" and toks[m].text == "[" and toks[m + 1].kind == "id"
                    and toks[m + 2].kind == "punct" and toks[m + 2].text == "]"):
                fi.protected.add(toks[m + 1].start)


def _scan_directive(fi: FileInfo, body: list[Tok], line: int) -> None:
    fi.directive_lines.add(line)
    if not body or body[0].kind not in ("id", "kw"):
        return
    name = body[0].text
    rest = body[1:]
    if name in _DIRECTIVE_INCLUDE:
        if rest and rest[0].kind == "punct" and rest[0].text == "<":
            for x in rest:                       # <stdio.h> 在原始词法里是 < stdio . h >，整段保护
                fi.protected.add(x.start)
                if x.kind == "punct" and x.text == ">":
                    break
        return
    if name in _DIRECTIVE_SKIP:
        for x in rest:
            fi.protected.add(x.start)
            if name == "pragma" and x.kind in ("id", "kw"):
                fi.pragma_ids.add(x.text)
        return
    if name == "define":
        macro = _macro_from_tokens(rest, fi.cls)
        if macro is not None:
            fi.macros.append(macro)
            if fi.cls == "V":
                body_ids = {x.text for x in macro.body if x.kind == "id"} - set(macro.params)
                fi.directive_ids |= body_ids
        return
    if fi.cls == "V" and name in ("if", "ifdef", "ifndef", "elif", "undef"):
        fi.directive_ids |= {x.text for x in rest if x.kind == "id"}


def _paren_end(toks: list[Tok], open_idx: int) -> int:
    """toks[open_idx] 是 '('，返回配对的 ')' 的下标；没配平返回 -1。"""
    depth = 0
    for j in range(open_idx, len(toks)):
        x = toks[j]
        if x.kind == "punct":
            if x.text == "(":
                depth += 1
            elif x.text == ")":
                depth -= 1
                if depth == 0:
                    return j
    return -1


def _scan_attributes(fi: FileInfo) -> None:
    """不是项目标识符的位置记入保护集：__attribute__((a, b(1))) / __declspec(x) 里每一项的属性名，
    __has_include(<x.h>) / __has_attribute(x) 之类探测函数的整个括号，asm 操作数的符号名 [name]。"""
    toks = [t for t in fi.toks if t.kind != "cmt"]
    n = len(toks)
    for i, t in enumerate(toks):
        if t.kind not in ("id", "kw") or i + 1 >= n or toks[i + 1].text != "(" or toks[i + 1].kind != "punct":
            continue
        if t.text in _PROBE_CALLS:
            end = _paren_end(toks, i + 1)
            for k in range(i + 2, end if end > 0 else i + 2):
                fi.protected.add(toks[k].start)
            continue
        if t.text not in _ATTR_KEYWORDS:
            continue
        j = _paren_end(toks, i + 1)
        if j < 0:
            continue
        lo, hi = i + 2, j                          # 括号内部 [lo, hi)
        if t.text != "__declspec" and lo < hi and toks[lo].text == "(" and toks[hi - 1].text == ")":
            lo, hi = lo + 1, hi - 1                # __attribute__ 的第二层括号
        item_start = True
        depth = 0
        for k in range(lo, hi):
            x = toks[k]
            if item_start and x.kind in ("id", "kw"):
                fi.protected.add(x.start)
            item_start = False
            if x.kind == "punct":
                if x.text == "(":
                    depth += 1
                elif x.text == ")":
                    depth -= 1
                elif x.text == "," and depth == 0:
                    item_start = True


def _build_fileinfo(index: ci.Index, args: list[str], root: Path, rel: str, cls: str
                    ) -> tuple[FileInfo, ci.TranslationUnit]:
    """解析一个文件，取 token、逻辑行、预处理指令 / 属性的保护区；返回 (FileInfo, tu)。"""
    path = root / rel
    data = path.read_bytes()
    try:
        tu = index.parse(str(path), args=args, options=clangcfg.DEFAULT_PARSE_OPTIONS)
    except ci.TranslationUnitLoadError as exc:
        raise SymbolError(f"libclang 无法加载 {path}: {exc}") from exc
    fi = FileInfo(rel=rel, path=path, cls=cls, data=data)
    fi.toks = _lex(tu, path, data)
    _assign_lines(data, fi.toks)
    fi.ident_at = {t.start: t.text for t in fi.toks if t.kind == "id"}
    _scan_structure(fi)
    fi.parse_errors = sum(1 for d in tu.diagnostics if d.severity >= ci.Diagnostic.Error)
    return fi, tu


def load_file_info(spec: CorpusSpec, src_root: Path, rel: str, cls: str = "R") -> FileInfo:
    """取 src_root 下一个文件的词法信息（token 流、保护区等），测试和调试用。"""
    root = Path(src_root).resolve()
    args = clang_args(list(spec.cflags), root)
    return _build_fileinfo(clangcfg.index(), args, root, rel, cls)[0]


def rewrite_bytes(fi: FileInfo, mapping: dict[str, str]) -> tuple[bytes, int]:
    """按映射表改写一个文件的内容（不落盘）：只换未受保护的 IDENTIFIER token，返回 (新内容, 替换数)。"""
    return _rewrite(fi.data, fi.toks, fi.protected, mapping)


def _closure(allm: dict[str, list[MacroDef]], seeds: set[str]) -> set[str]:
    """宏体里提到 seeds 中某个宏名的宏，也算被牵连（包装宏）。"""
    tainted = set(seeds)
    changed = True
    while changed:
        changed = False
        for name, defs in allm.items():
            if name in tainted:
                continue
            for d in defs:
                if any(t.kind == "id" and t.text in tainted for t in d.body):
                    tainted.add(name)
                    changed = True
                    break
    return tainted


def _is_implicit_decl(r: ci.Cursor) -> bool:
    """被引用的 FUNCTION_DECL 是不是隐式声明：真声明的 extent 至少盖住「类型 + 名字 + 参数」，隐式的为空或只有名字。"""
    try:
        e = r.extent
        return (e.end.offset - e.start.offset) <= len(r.spelling)
    except Exception as exc:  # noqa: BLE001 - libclang 偶发异常
        # 这是「别给 libc 的隐式声明函数改名」的安全闸：判不出来时宁可当隐式声明处理（只会少改一个名字），
        # 而不是放行（可能把 libc 函数改名，链接不上）
        log.warning("判断 %r 是否隐式声明时出错（%s: %s），按隐式声明处理", getattr(r, "spelling", "?"), type(exc).__name__, exc)
        return True


# --------------------------------------------------------------------------- 收集


@dataclass
class Analysis:
    """collect 的完整结果。"""

    root: Path
    r_files: list[str]
    v_files: list[str]
    kinds: dict[str, str]                      # 将被改名的拼写 -> 种类
    names: set[str]
    excluded: dict[str, list[str]]             # 原因 -> 被排除的项目拼写（可多原因重复出现）
    reserved: set[str]                         # 新名字不得使用的拼写
    stats: dict[str, Any]


class IdentifierSet(set):
    """collect_project_identifiers 的返回值：就是 ``set[str]``（要改名的拼写），多带几个属性。

    * ``kinds``：拼写 -> func / type / var / macro / field，build_rename_map 据此选前缀；
    * ``reserved``：新名字不得取用的拼写（系统头、vendored、源码里出现过的所有标识符、关键字等）；
    * ``excluded``：原因 -> 项目里声明过但被排除的拼写；
    * ``stats``：统计信息（可 JSON 序列化）。
    """

    kinds: dict[str, str]
    reserved: frozenset[str]
    excluded: dict[str, list[str]]
    stats: dict[str, Any]

    def __init__(self, iterable: Iterable[str] = ()) -> None:
        super().__init__(iterable)
        self.kinds = {}
        self.reserved = frozenset()
        self.excluded = {}
        self.stats = {}

    def __repr__(self) -> str:                    # 默认的 set 表示会把几百个名字全打出来
        return f"IdentifierSet({len(self)} 个拼写)"


class _Analyzer:
    def __init__(self, spec: CorpusSpec, src_root: Path, strict_stringify: bool = False) -> None:
        self.spec = spec
        self.root = Path(src_root).resolve()
        self.strict = strict_stringify
        self.r_rel, self.v_rel = project_file_lists(spec, self.root)
        self.cls_by_rel = {**{r: "R" for r in self.r_rel}, **{v: "V" for v in self.v_rel}}
        self.real2rel = {os.path.realpath(self.root / rel): rel for rel in self.cls_by_rel}
        self.configs = _config_cflags(spec)
        self.index = clangcfg.index()
        self.files: dict[str, FileInfo] = {}
        self._fcls: dict[str, tuple[str, Optional[str]]] = {}
        # 遍历 AST 时累积
        self.cand: dict[str, set[str]] = defaultdict(set)      # 拼写 -> 种类集合（只含可改名文件里的声明）
        self.fs_decl: dict[str, set[str]] = defaultdict(set)   # 文件作用域的函数 / 变量声明
        self.fs_def: set[str] = set()                          # 有定义的函数 / 变量
        self.ext: set[str] = set()                             # 系统头 / 其它项目外文件里声明的拼写
        self.vdecl: set[str] = set()                           # vendored 文件里声明的拼写
        self.vref: set[str] = set()                            # 被 vendored 代码引用的项目实体
        self.implicit: set[str] = set()                        # 被隐式声明调用的函数名
        self.macros_e: dict[str, MacroDef] = {}                # 项目源码里出现过名字的系统头宏
        self.events: dict[tuple[str, int], set[str]] = defaultdict(set)
        self._seen_tops: set[tuple] = set()
        self.all_idents: set[str] = set()

    # -- 文件分类 ---------------------------------------------------------
    def classify(self, f: Optional[ci.File]) -> tuple[str, Optional[str]]:
        if f is None:
            return "E", None
        name = f.name
        hit = self._fcls.get(name)
        if hit is None:
            rel = self.real2rel.get(os.path.realpath(name))
            hit = (self.cls_by_rel[rel], rel) if rel else ("E", None)
            self._fcls[name] = hit
        return hit

    # -- 阶段 A：词法 -----------------------------------------------------
    def lex_all(self) -> list[tuple[str, ci.TranslationUnit]]:
        """解析每个文件（基础配置）并取 token；返回 (rel, tu) 供阶段 B 复用。"""
        args = clang_args(self.configs[0], self.root)
        tus: list[tuple[str, ci.TranslationUnit]] = []
        for rel in self.r_rel + self.v_rel:
            fi, tu = _build_fileinfo(self.index, args, self.root, rel, self.cls_by_rel[rel])
            self.files[rel] = fi
            tus.append((rel, tu))
        for fi in self.files.values():
            self.all_idents |= {t.text for t in fi.toks if t.kind == "id"}
        return tus

    # -- 阶段 B：遍历 AST -------------------------------------------------
    def walk_tu(self, tu: ci.TranslationUnit, cfgi: int) -> None:
        for top in tu.cursor.get_children():
            self._visit_top(tu, top, cfgi)

    def _visit_top(self, tu: ci.TranslationUnit, top: ci.Cursor, cfgi: int) -> None:
        k = kind_of(top)
        if k is None:
            return
        cls, rel = self.classify(top.location.file)
        if cls == "E":
            self._visit_external(tu, top, k)
            return
        ext = top.extent
        key = (cfgi, rel, ext.start.offset, ext.end.offset, k, top.spelling)
        if key in self._seen_tops:                 # 同一份头文件被很多测试 TU 包含：只走一遍
            return
        self._seen_tops.add(key)
        self._visit_project(top)

    def _visit_external(self, tu: ci.TranslationUnit, top: ci.Cursor, k: ci.CursorKind) -> None:
        if k == CK.MACRO_DEFINITION:
            name = top.spelling
            if _IDENT_RE.match(name):
                self.ext.add(name)
                if name in self.all_idents and name not in self.macros_e and top.location.file is not None:
                    self._fetch_external_macro(tu, top, name)
            return
        if k in _EXT_TOP_KINDS:
            name = top.spelling
            if _IDENT_RE.match(name):
                self.ext.add(name)
            return
        if k not in _EXT_TYPE_KINDS:
            return
        stack = [top]
        while stack:
            c = stack.pop()
            kk = kind_of(c)
            if kk not in _EXT_TYPE_KINDS:
                continue
            name = c.spelling
            if _IDENT_RE.match(name):
                self.ext.add(name)
            stack.extend(c.get_children())

    def _fetch_external_macro(self, tu: ci.TranslationUnit, top: ci.Cursor, name: str) -> None:
        toks: list[Tok] = []
        try:
            for t in tu.get_tokens(extent=top.extent):
                k = t.kind
                if k == TK.IDENTIFIER:
                    toks.append(Tok("id", t.spelling, t.extent.start.offset, t.extent.end.offset))
                elif k == TK.KEYWORD:
                    toks.append(Tok("kw", t.spelling, t.extent.start.offset, t.extent.end.offset))
                elif k == TK.PUNCTUATION:
                    toks.append(Tok("punct", t.spelling, t.extent.start.offset, t.extent.end.offset))
                elif k == TK.LITERAL:
                    toks.append(Tok("lit", "", t.extent.start.offset, t.extent.end.offset))
        except Exception as exc:  # noqa: BLE001 - 取不到宏体就当没有 # / ##，系统宏的影响只是调用实参，影响很小
            log.warning("取系统头宏 %s 的宏体失败（%s: %s），按没有 # / ## 处理", name, type(exc).__name__, exc)
            return
        macro = _macro_from_tokens(toks, "E")
        if macro is not None and macro.name == name:
            self.macros_e[name] = macro

    def _visit_project(self, top: ci.Cursor) -> None:
        stack: list[tuple[ci.Cursor, bool]] = [(top, False)]
        while stack:
            c, in_func = stack.pop()
            k = kind_of(c)
            if k is None:
                continue
            loc = c.location
            cls, rel = self.classify(loc.file)
            if k in _DECL_LABEL:
                name = c.spelling
                if name and _IDENT_RE.match(name):
                    if cls == "R":
                        self._record_decl(c, k, name, rel, loc.offset, in_func)
                    elif cls == "V":
                        if k == CK.MACRO_DEFINITION or not in_func:
                            self.vdecl.add(name)
                    elif k == CK.MACRO_DEFINITION or (k in (CK.FUNCTION_DECL, CK.VAR_DECL) and not in_func):
                        self.ext.add(name)
            elif k in _REF_KINDS:
                self._record_ref(c, k, cls, rel, loc.offset)
            child_in_func = in_func or k == CK.FUNCTION_DECL
            for ch in c.get_children():
                stack.append((ch, child_in_func))

    def _record_decl(self, c: ci.Cursor, k: ci.CursorKind, name: str, rel: str, offset: int,
                     in_func: bool) -> None:
        self.cand[name].add(_DECL_LABEL[k])
        if k in (CK.FUNCTION_DECL, CK.VAR_DECL) and not in_func:
            self.fs_decl[name].add(_DECL_LABEL[k])
            if k == CK.FUNCTION_DECL:
                if c.is_definition():
                    self.fs_def.add(name)
            elif c.is_definition() or c.storage_class != ci.StorageClass.EXTERN:
                self.fs_def.add(name)              # 含暂存定义（int x;），static 变量也算定义
        tok = self.files[rel].ident_at.get(offset)
        if tok is not None and tok != name:        # 名字不是写在这个位置上的：宏展开出来的声明
            self.events[(rel, offset)].add(name)

    def _record_ref(self, c: ci.Cursor, k: ci.CursorKind, cls: str, rel: Optional[str], offset: int) -> None:
        if cls == "E":
            return
        if k == CK.MACRO_INSTANTIATION:
            if cls == "V":
                r = c.referenced
                if r is not None and self.classify(r.location.file)[0] == "R":
                    self.vref.add(c.spelling)
            return
        r = c.referenced
        if r is None:
            return
        rname = r.spelling
        if not rname or not _IDENT_RE.match(rname):
            return
        if k == CK.CALL_EXPR and kind_of(r) == CK.FUNCTION_DECL and _is_implicit_decl(r):
            self.implicit.add(rname)
        if cls == "V":
            if self.classify(r.location.file)[0] == "R":
                self.vref.add(rname)
        elif cls == "R":
            tok = self.files[rel].ident_at.get(offset)
            if tok is not None and tok != rname:   # 位置上是别的标识符（宏名）：引用出现在宏展开里
                self.events[(rel, offset)].add(rname)

    # -- 阶段 C：汇总 -----------------------------------------------------
    def finalize(self) -> Analysis:
        """汇总：收集宏 → 找出被 # / ## 牵连的拼写 → 给每个候选拼写判排除原因 → 出统计。"""
        allm, macros_rv = self._gather_macros()
        excl_stringify, at_risk = self._stringify_sets(allm, macros_rv)

        pragma_ids: set[str] = set()
        vendor_use: set[str] = set(self.vref)
        for fi in self.files.values():
            pragma_ids |= fi.pragma_ids
            if fi.cls == "V":
                vendor_use |= fi.directive_ids

        # 源码里真正以（未保护的）标识符 token 出现过的拼写
        seen_in_r: set[str] = set()
        for fi in self.files.values():
            if fi.cls == "R":
                seen_in_r |= {t.text for t in fi.toks if t.kind == "id" and t.start not in fi.protected}

        cmdline = _cmdline_macros(self.spec, self.root)
        keep = set(self.spec.keep_names)
        reasons, declared, dropped_unseen = self._exclusion_reasons(
            seen_in_r, excl_stringify, pragma_ids, vendor_use, cmdline, keep)

        names = {n for n in self.cand if n in seen_in_r and n not in reasons}
        kinds = {n: next(k for k in KIND_PRIORITY if k in self.cand[n]) for n in names}
        excluded: dict[str, list[str]] = defaultdict(list)
        first_reason: dict[str, int] = defaultdict(int)
        for name, why in reasons.items():
            for w in why:
                excluded[w].append(name)
            first_reason[min(why, key=_REASON_ORDER.index)] += 1

        reserved = set(C_KEYWORDS) | ENTRY_NAMES | KNOWN_EXTERNAL_NAMES | self.ext | self.vdecl | self.all_idents
        reserved |= self.implicit | set(self.macros_e) | cmdline | keep | pragma_ids
        # 非 strict 时：出现在测试文件 # 宏调用实参里、却照常改了名的拼写（字符串化的输出会随之变化，测试若比对输出就会挂）
        stringified = sorted(n for n in at_risk if n in names) if not self.strict else []
        func_name_uses = sum(1 for fi in self.files.values() if fi.cls == "R" for t in fi.toks
                             if t.kind in ("id", "kw") and t.text in _FUNC_NAME_MACROS)
        stats: dict[str, Any] = {
            "files_total": len(self.r_rel),
            "files_vendored": len(self.v_rel),
            "configs": len(self.configs),
            "entities_declared": declared,
            "entities_renamed": len(names),
            "entities_excluded": len(reasons),
            "dropped_unseen": dropped_unseen,
            "renamed_by_kind": {k: sum(1 for v in kinds.values() if v == k) for k in KIND_PRIORITY},
            "excluded_by_reason": {r: first_reason[r] for r in _REASON_ORDER if first_reason.get(r)},
            "excluded_stringify": sorted(excluded.get("stringify", [])),
            "stringified_names_renamed": stringified,
            "func_name_uses": func_name_uses,       # __func__ 之类在可改名文件里出现的次数（输出里会带函数名）
            "strict_stringify": self.strict,
            "exclusion_set_size": len(reserved),
            "external_names": len(self.ext | self.vdecl),
            "parse_error_files": {rel: fi.parse_errors for rel, fi in sorted(self.files.items())
                                  if fi.parse_errors},
        }
        return Analysis(root=self.root, r_files=self.r_rel, v_files=self.v_rel, kinds=kinds, names=names,
                        excluded={k: sorted(v) for k, v in excluded.items()}, reserved=reserved, stats=stats)

    def _gather_macros(self) -> tuple[dict[str, list[MacroDef]], list[MacroDef]]:
        """(所有可见宏的名字 -> 定义列表, 项目内 + vendored 的宏定义)。"""
        allm: dict[str, list[MacroDef]] = defaultdict(list)
        macros_rv: list[MacroDef] = []
        for fi in self.files.values():
            for m in fi.macros:
                allm[m.name].append(m)
                macros_rv.append(m)
        for name, m in self.macros_e.items():
            allm[name].append(m)
        return allm, macros_rv

    def _stringify_sets(self, allm: dict[str, list[MacroDef]], macros_rv: list[MacroDef]
                        ) -> tuple[set[str], set[str]]:
        """# / ## 宏牵连的拼写 → (一定排除的, 非 strict 时仍照常改名的)。

        一定排除：``##`` 宏（改了就拼不出声明过的名字，编译不过）；宏体含 ``#`` / ``##`` 的宏本身涉及的拼写；
        **目标文件（``target_files``，库自己的源码）里 ``#`` 宏的调用实参**——库代码里被字符串化的名字会变成
        运行时输出的一部分，改名等于改了库的行为，而项目测试多半不逐字比对这类文字，verify 发现不了。
        只有测试文件里 ``#`` 宏的实参（``CHECK(f(x) == 1)`` 之类断言宏，字符串化结果只出现在失败提示里）在非 strict 时照常改名，
        否则所有被测函数都改不了；这部分如实记在 ``stringified_names_renamed``，strict 模式连它们一起排除。
        """
        function_like = {n for n, ds in allm.items() if any(d.function_like for d in ds)}
        paste_seed = {n for n, ds in allm.items() if any(d.has_paste for d in ds)}
        hash_seed = {n for n, ds in allm.items() if any(d.has_hash for d in ds)}
        paste_t = _closure(allm, paste_seed)
        hash_t = _closure(allm, hash_seed)

        # 宏体含 # / ## 的（项目内或 vendored）宏所涉及的拼写：宏名、形参、宏体里的标识符
        involve: set[str] = set()
        for m in macros_rv:
            if m.has_hash or m.has_paste:
                involve.add(m.name)
                involve.update(m.params)
                involve.update(t.text for t in m.body if t.kind == "id")

        # 调用点：## 宏与 # 宏的调用实参，以及展开里含这类宏的调用起点
        lib_files = set(expand_patterns(self.spec.target_files, self.root))
        paste_args: set[str] = set()
        hash_args: set[str] = set()          # 测试等非目标文件里的 # 宏调用实参
        hash_args_lib: set[str] = set()      # 目标文件里的
        for fi in self.files.values():
            if fi.cls != "R":
                continue
            self._scan_invocations(fi, allm, function_like, paste_t, hash_t, paste_args,
                                   hash_args_lib if fi.rel in lib_files else hash_args)
        gen_paste: set[str] = set()
        gen_hash: set[str] = set()
        gen_hash_lib: set[str] = set()
        for (rel, off), names in self.events.items():
            fi = self.files[rel]
            if off in fi.tainted_paste:
                gen_paste |= names
            if off in fi.tainted_hash:
                (gen_hash_lib if rel in lib_files else gen_hash).update(names)
        excl_stringify = involve | paste_args | gen_paste | hash_args_lib | gen_hash_lib
        at_risk = (hash_args | gen_hash) - excl_stringify
        if self.strict:
            excl_stringify |= at_risk
        return excl_stringify, at_risk

    def _exclusion_reasons(self, seen_in_r: set[str], excl_stringify: set[str], pragma_ids: set[str],
                           vendor_use: set[str], cmdline: set[str], keep: set[str]
                           ) -> tuple[dict[str, list[str]], int, int]:
        """每个候选拼写的排除原因 → (拼写 -> 原因列表, 声明过的拼写数, 因为源码里没真正出现而丢掉的数)。"""
        no_def = {n for n, kinds in self.fs_decl.items() if n not in self.fs_def}
        implicit = {n for n in self.implicit if n not in self.fs_def}
        ext_decl = self.ext | self.vdecl
        reasons: dict[str, list[str]] = {}
        declared = 0
        dropped_unseen = 0
        for name in sorted(self.cand):
            if name not in seen_in_r:
                dropped_unseen += 1
                continue
            declared += 1
            why: list[str] = []
            if name in C_KEYWORDS:
                why.append("keyword")
            if name in ENTRY_NAMES:
                why.append("entry")
            if _RESERVED_RE.match(name):
                why.append("reserved_prefix")
            if name in KNOWN_EXTERNAL_NAMES:
                why.append("known_external")
            if name in ext_decl:
                why.append("external_decl")
            if name in implicit:
                why.append("implicit_decl")
            if name in no_def:
                why.append("no_definition")
            if name in vendor_use:
                why.append("vendored_use")
            if name in cmdline:
                why.append("cmdline_macro")
            if name in keep:
                why.append("keep_names")
            if name in excl_stringify:
                why.append("stringify")
            if name in pragma_ids:
                why.append("pragma")
            if why:
                reasons[name] = why
        return reasons, declared, dropped_unseen

    def _scan_invocations(self, fi: FileInfo, allm: dict[str, list[MacroDef]], function_like: set[str],
                          paste_t: set[str], hash_t: set[str], paste_args: set[str],
                          hash_args: set[str]) -> None:
        ct = [t for t in fi.toks if t.kind != "cmt"]
        n = len(ct)
        for i, t in enumerate(ct):
            if t.kind != "id" or t.text not in allm or t.start in fi.protected:
                continue
            j = i
            if t.text in function_like and i + 1 < n and ct[i + 1].kind == "punct" and ct[i + 1].text == "(":
                depth = 0
                j = i + 1
                in_directive = t.line in fi.directive_lines
                while j < n:
                    x = ct[j]
                    if in_directive and x.line != t.line:
                        j = i
                        break
                    if x.kind == "punct":
                        if x.text == "(":
                            depth += 1
                        elif x.text == ")":
                            depth -= 1
                            if depth == 0:
                                break
                    j += 1
                if j >= n:
                    j = i
            span = ct[i:j + 1]
            span_ids = {x.text for x in span if x.kind == "id"}
            if span_ids & paste_t:
                fi.tainted_paste.add(t.start)
            if span_ids & hash_t:
                fi.tainted_hash.add(t.start)
            if j > i:
                args = {x.text for x in ct[i + 2:j] if x.kind == "id"}
                if t.text in paste_t:
                    paste_args.update(args)
                if t.text in hash_t:
                    hash_args.update(args)
                # 高阶宏（X-macro 列表）：实参传的是另一个拼接 / 字符串化宏的名字，如 COLOR_LIST(AS_ENUM)，
                # 列表宏体里 X(RED, 1) 的 RED 会被那个宏拼接 / 字符串化，所以宏体里的标识符也算被牵连
                if (args & paste_t) - {t.text}:
                    paste_args.update(x.text for d in allm[t.text] for x in d.body if x.kind == "id")
                if (args & hash_t) - {t.text}:
                    hash_args.update(x.text for d in allm[t.text] for x in d.body if x.kind == "id")


def analyze_project(spec: CorpusSpec, src_root: Path, *, strict_stringify: bool = False) -> Analysis:
    """完整的收集流程，返回 :class:`Analysis`。"""
    an = _Analyzer(spec, src_root, strict_stringify=strict_stringify)
    tus = an.lex_all()
    for _rel, tu in tus:
        an.walk_tu(tu, 0)
    del tus
    for cfgi, cflags in enumerate(an.configs[1:], start=1):
        args = clang_args(cflags, an.root)
        for rel in an.r_rel + an.v_rel:
            path = an.root / rel
            try:
                tu = an.index.parse(str(path), args=args, options=clangcfg.DEFAULT_PARSE_OPTIONS)
            except ci.TranslationUnitLoadError:
                log.warning("并集配置下无法解析 %s，跳过", rel)
                continue
            an.walk_tu(tu, cfgi)
    result = an.finalize()
    log.info("%s: 声明 %d 个拼写，改名 %d，排除 %d（%s）", Path(an.root).name,
             result.stats["entities_declared"], result.stats["entities_renamed"],
             result.stats["entities_excluded"], result.stats["excluded_by_reason"])
    return result


def collect_project_identifiers(spec: CorpusSpec, src_root: Path, *, strict_stringify: bool = False) -> IdentifierSet:
    """项目文件内声明的函数、变量、参数、字段、类型、枚举常量、标签、宏的拼写（已去掉排除集）。

    返回 ``set[str]`` 的子类 :class:`IdentifierSet`，带 kinds / reserved / excluded / stats 属性。
    """
    res = analyze_project(spec, src_root, strict_stringify=strict_stringify)
    out = IdentifierSet(res.names)
    out.kinds = dict(res.kinds)
    out.reserved = frozenset(res.reserved)
    out.excluded = res.excluded
    out.stats = res.stats
    return out


# --------------------------------------------------------------------------- 改名表


def _guess_kind(name: str) -> str:
    """没有种类信息（传入的是普通 set）时的兜底：全大写当宏，其余当变量。"""
    letters = [ch for ch in name if ch.isalpha()]
    return "macro" if letters and all(ch.isupper() for ch in letters) else "var"


def build_rename_map(names: Iterable[str], seed: Any, *, kinds: Optional[dict[str, str]] = None,
                     reserved: Optional[Iterable[str]] = None) -> dict[str, str]:
    """旧拼写 -> 新拼写的双射表：函数 f_<n>、类型 t_<n>、变量 v_<n>、宏 M_<n>、字段 m_<n>，n 由 seed 打乱。

    * 种类默认取 ``names.kinds``（IdentifierSet），没有就按拼写兜底；同一前缀内先按拼写排序再分配，
      所以结果只取决于（名字集合, 种类, seed），与集合迭代顺序无关；
    * 新名字不会与 ``reserved``（默认取 ``names.reserved``）、也不会与任何旧名字相同。
    """
    if kinds is None:
        kinds = getattr(names, "kinds", None) or {}
    if reserved is None:
        reserved = getattr(names, "reserved", None) or ()
    old = sorted(set(names))
    taken = set(reserved) | set(old)
    groups: dict[str, list[str]] = defaultdict(list)
    for name in old:
        kind = kinds.get(name) or _guess_kind(name)
        if kind not in KIND_PREFIX:
            raise ValueError(f"{name!r} 的种类 {kind!r} 不在 {sorted(KIND_PREFIX)} 里")
        groups[KIND_PREFIX[kind]].append(name)
    mapping: dict[str, str] = {}
    for prefix in sorted(groups):
        group = groups[prefix]
        rng = random.Random(f"{seed}:{prefix}")
        pool = list(range(1, len(group) + 1))
        rng.shuffle(pool)
        queue = deque(pool)
        extra = len(group)
        for name in group:
            while True:
                if queue:
                    n = queue.popleft()
                else:                              # 有编号撞了保留拼写：往后顺延
                    extra += 1
                    n = extra
                new = f"{prefix}_{n}"
                if new not in taken:
                    break
            taken.add(new)
            mapping[name] = new
    return mapping


# --------------------------------------------------------------------------- 替换


def _validate_mapping(mapping: dict[str, str]) -> None:
    bad = [k for k, v in mapping.items() if not _IDENT_RE.match(k) or not _IDENT_RE.match(v)]
    if bad:
        raise ValueError(f"映射表里有不是合法标识符的项: {bad[:5]}")
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("映射表不是双射：有多个旧名字映射到同一个新名字")
    overlap = set(mapping) & set(mapping.values())
    if overlap:
        raise ValueError(f"新名字与旧名字重叠: {sorted(overlap)[:5]}")
    kw = [v for v in mapping.values() if v in C_KEYWORDS]
    if kw:
        raise ValueError(f"新名字是 C 关键字: {kw[:5]}")


def _rewrite(data: bytes, toks: list[Tok], protected: set[int], mapping: dict[str, str]) -> tuple[bytes, int]:
    """token 级替换：按偏移从后往前写回。"""
    repl = [(t.start, t.end, mapping[t.text]) for t in toks
            if t.kind == "id" and t.text in mapping and t.start not in protected]
    if not repl:
        return data, 0
    out = bytearray(data)
    for s, e, new in reversed(repl):
        out[s:e] = new.encode("ascii")
    return bytes(out), len(repl)


def _diag_text(d: ci.Diagnostic, root: Path) -> str:
    try:
        f = d.location.file.name if d.location.file else "<unknown>"
        f = os.path.relpath(f, root) if os.path.isabs(f) else f
        return f"{f}:{d.location.line}: {d.spelling}"
    except UnicodeDecodeError:                     # 诊断文字里带了非 UTF-8 字节
        return f"<诊断文字无法解码>:{d.location.line}"


def _error_counts(index: ci.Index, configs: list[list[str]], root: Path, files: list[str]
                  ) -> dict[tuple[str, int], tuple[int, list[str]]]:
    """每个（文件，配置序号）解析后 error 级以上诊断的（个数，文字）。"""
    out: dict[tuple[str, int], tuple[int, list[str]]] = {}
    for ci_, cflags in enumerate(configs):
        args = clang_args(cflags, root)
        for rel in files:
            try:
                tu = index.parse(str(root / rel), args=args, options=clangcfg.DEFAULT_PARSE_OPTIONS)
            except ci.TranslationUnitLoadError as exc:
                raise SymbolError(f"libclang 无法加载 {root / rel}: {exc}") from exc
            msgs = [_diag_text(d, root) for d in tu.diagnostics if d.severity >= ci.Diagnostic.Error]
            out[(rel, ci_)] = (len(msgs), msgs)
    return out


def apply_renames(spec: CorpusSpec, src_root: Path, mapping: dict[str, str]) -> dict[str, int]:
    """对 src_root 里的可改名文件做 token 级替换（原地修改），返回 {相对路径: 替换次数}。

    失败（新名字与现有拼写冲突、改名后目标文件出现新的 error 诊断）时所有文件恢复原样：
    冲突抛 ValueError，解析出错抛 :class:`RenameError`。
    """
    return _apply(spec, src_root, mapping)[0]


def _apply(spec: CorpusSpec, src_root: Path, mapping: dict[str, str]) -> tuple[dict[str, int], dict[str, int]]:
    """apply_renames 的实现；第二个返回值是目标文件改名前后的 error 诊断总数（各配置之和）。"""
    root = Path(src_root).resolve()
    _validate_mapping(mapping)
    r_files, v_files = project_file_lists(spec, root)
    info = {"baseline_errors": 0, "reparse_errors": 0}
    if not mapping:
        return {rel: 0 for rel in r_files}, info
    index = clangcfg.index()
    configs = _config_cflags(spec)
    args = clang_args(configs[0], root)
    targets = expand_patterns(spec.target_files, root)
    baseline = _error_counts(index, configs, root, targets) if targets else {}
    info["baseline_errors"] = sum(n for n, _m in baseline.values())

    # 1) 先在内存里算好每个文件的新内容
    plans: dict[str, tuple[bytes, bytes, int]] = {}
    present: set[str] = set()
    v_set = set(v_files)
    for rel in r_files + v_files:
        fi, _tu = _build_fileinfo(index, args, root, rel, "V" if rel in v_set else "R")
        present |= {t.text for t in fi.toks if t.kind == "id" and t.start not in fi.protected}
        if rel in v_set:
            continue
        new, n = _rewrite(fi.data, fi.toks, fi.protected, mapping)
        plans[rel] = (fi.data, new, n)
    clash = sorted(set(mapping.values()) & present)
    if clash:
        raise ValueError(f"新名字与源码里已有的标识符冲突: {clash[:5]}")

    # 2) 写回；符号链接先换成普通文件，免得改到树外（比如 pristine 源码）
    written: list[str] = []
    try:
        for rel, (old, new, _n) in plans.items():
            if new == old:
                continue
            path = root / rel
            written.append(rel)                    # 先登记再写：写到一半失败也能恢复
            if path.is_symlink():
                path.unlink()
            path.write_bytes(new)
        after = _error_counts(index, configs, root, targets) if targets else {}
        info["reparse_errors"] = sum(n for n, _m in after.values())
        worse = []
        for key, (n_after, msgs) in after.items():
            n_before = baseline.get(key, (0, []))[0]
            if n_after > n_before:
                worse.append((key, n_before, n_after, msgs))
        if worse:
            lines = [m for _k, _b, _a, ms in worse for m in ms][:8]
            raise RenameError("改名后目标文件出现新的 error 诊断: " +
                              "; ".join(f"{k[0]}(配置{k[1]}) {b}->{a}" for k, b, a, _m in worse), lines)
    except BaseException:
        for rel in written:                       # 回滚
            path = root / rel
            if path.is_symlink():
                path.unlink()
            path.write_bytes(plans[rel][0])
        raise
    return {rel: plans[rel][2] for rel in r_files}, info


# --------------------------------------------------------------------------- 一站式入口


@dataclass
class SymbolsResult:
    mapping: dict[str, str]
    stats: dict[str, Any]
    per_file: dict[str, int]


def obfuscate_symbols(spec: CorpusSpec, src_root: Path, seed: Any, *, strict_stringify: bool = False) -> SymbolsResult:
    """collect -> build_rename_map -> apply_renames，原地改写 src_root，返回映射表与 stats。"""
    ids = collect_project_identifiers(spec, src_root, strict_stringify=strict_stringify)
    mapping = build_rename_map(ids, seed)
    per_file, info = _apply(spec, src_root, mapping)
    stats = dict(ids.stats)
    stats.update({
        "level": "L1",
        "seed": seed,
        "files_changed": sum(1 for n in per_file.values() if n),
        "tokens_replaced": sum(per_file.values()),
        "renamed_by_prefix": {p: sum(1 for v in mapping.values() if v.startswith(p + "_"))
                              for p in sorted(KIND_PREFIX.values())},
        "baseline_errors": info["baseline_errors"],        # 改名前目标文件的 error 级诊断数（应为 0）
        "reparse_errors": info["reparse_errors"],          # 改名后重新解析的 error 级诊断数，必须为 0
    })
    return SymbolsResult(mapping=mapping, stats=stats, per_file=per_file)
