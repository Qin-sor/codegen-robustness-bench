"""函数提取与依赖闭包（设计文档 §4.3）。

用 libclang 解析语料里的 C 文件，对每个「有函数体的定义」产出一个 FunctionTarget：
签名、字节偏移、语句数、被调函数，以及让模型能独立重写该函数所需的依赖上下文
（被调函数原型、类型定义、全局变量 extern 声明、项目宏、系统头 include 行）。

约定与要点：

* 所有偏移都是文件里的字节偏移（libclang 的 extent.start.offset / end.offset），
  文件一律按 bytes 读写，不能拿它去切 str。
* 「实体是否属于项目」由 cursor.referenced 加 USR 判定：先扫一遍 TU 顶层，把声明在项目文件内的
  函数、全局变量按 USR 建索引，被引用实体的 USR 在索引里才算项目实体（同一实体的多次声明 USR 相同，
  所以无论引用解析到头文件原型还是定义都能认出来）。类型实体先取 get_definition()，再看它是否落在
  项目文件的某条顶层 typedef / struct / union / enum 语句的范围内（内联在 typedef 里的 struct、
  嵌套 struct 没有独立的顶层条目，只能按范围归属）。系统头里的声明一律不收，系统头靠 includes 带入。
* 类型按 TU 里的出现顺序输出（原代码能编译，所以这个顺序天然满足「被依赖的在前」）。
* 宏通过 PARSE_DETAILED_PROCESSING_RECORD 的 MACRO_INSTANTIATION 找到引用，
  再用 MACRO_DEFINITION 的位置取完整 #define 文本，并沿宏体做传递闭包。
* locate() 用于混淆副本：副本里函数名已被改写，按新名字重新解析并定位。

libclang 的解析参数（spec.cflags 换算、缺失的编译器内置头目录）统一由 :mod:`bench.clangcfg` 提供，见
``clangcfg.clang_args`` / ``clangcfg.builtin_include_args``。
"""
from __future__ import annotations

import bisect
import copy
import ctypes
import functools
import hashlib
import logging
import os
import re
import threading
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence

import clang.cindex as ci

from bench import clangcfg
from bench.clangcfg import kind_of
from bench.corpus import CorpusSpec, expand_patterns

log = logging.getLogger(__name__)

CK = ci.CursorKind

__all__ = [
    "Dependencies",
    "FunctionTarget",
    "ExtractError",
    "parse_file",
    "extract_file",
    "extract_functions",
    "locate",
    "error_diagnostics",
    "strip_c_comments",
    "squash_whitespace",
]


# --------------------------------------------------------------------------- 数据类


@dataclass
class Dependencies:
    """目标函数的依赖闭包，每个列表里的元素都是一段可直接放进 prompt 的 C 文本。

    各列表按它们在 TU 里的出现顺序排列，所以类型定义里被依赖的在前；
    建议拼 prompt 时的顺序是 includes、macros、types、globals、decls（宏是纯文本，放最前最安全）。
    只收声明在项目文件内的实体；系统头里的类型、函数、宏一律不收，靠 includes 带入。
    """

    decls: list[str] = field(default_factory=list)      # 被引用的项目内函数的原型（去掉函数体）：直接调用的，加上只被取地址的（回调）
    types: list[str] = field(default_factory=list)      # 用到的 struct / union / enum / typedef 完整定义，传递闭包，按依赖顺序
    globals: list[str] = field(default_factory=list)    # 用到的全局变量 extern 声明（去掉 static 和初始化器；省略长度的数组补上长度）
    macros: list[str] = field(default_factory=list)     # 用到的项目内宏的完整 #define 行，含宏体里引用的其它项目宏
    includes: list[str] = field(default_factory=list)   # 该文件及它包含的项目头文件里，指向项目外文件（系统头）的 #include 行，按出现顺序


@dataclass
class FunctionTarget:
    """一个有函数体的函数定义，以及重写它所需的上下文。

    字节偏移 [start, end) 覆盖「存储类 + 返回类型 + 名字 + 参数 + 函数体」；
    紧贴在前面的注释、以及展开为空的宏（如空的 EXPORT 宏）不在范围内，替换时保持原样。
    """

    target_id: str          # f"{corpus}:{file}:{name}"
    corpus: str
    file: str               # 相对 src_root 的 POSIX 路径
    name: str
    signature: str          # 不含函数体的定义头，含存储类与返回类型；注释去掉、空白压成单空格，宏调用保持原文
    start: int              # 整个定义的字节偏移 [start, end)
    end: int
    body_start: int         # 函数体（含花括号）的字节偏移 [body_start, body_end)
    body_end: int
    n_statements: int       # COMPOUND_STMT 下直接子节点递归计数（空语句不计）；不带花括号的 if / for 体不单独计
    n_lines: int            # 定义占的行数
    callees: list[str]      # 函数体里直接调用（CALL_EXPR）的项目内函数名：去重、按首次出现排序、不含自身；经函数指针的调用不算
    deps: Dependencies
    doc_comment: Optional[str] = None   # 紧贴在定义前的注释，或头文件原型上的文档注释（/** */ 等）；分隔标题注释不算

    def to_dict(self) -> dict[str, Any]:
        """JSONL 序列化用：全字段平铺成 dict。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FunctionTarget":
        """to_dict 的逆操作；多余的键忽略，deps 缺失的列表按空处理。"""
        fields = {k: v for k, v in d.items() if k in cls.__dataclass_fields__ and k != "deps"}
        deps = d.get("deps") or {}
        fields["deps"] = Dependencies(**{k: list(deps.get(k, [])) for k in Dependencies.__dataclass_fields__})
        return cls(**fields)


class ExtractError(RuntimeError):
    """解析失败（文件缺失、libclang 报致命错误）时抛出，避免悄悄返回不完整的结果。"""


# --------------------------------------------------------------------------- 文本工具

_WS_RE = re.compile(r"\s+")


def squash_whitespace(text: str) -> str:
    """把所有空白（含换行）压成单个空格并去掉首尾空白。"""
    return _WS_RE.sub(" ", text).strip()


def strip_c_comments(text: str) -> str:
    """去掉 C 注释，字符串与字符常量里的 /* // 原样保留。

    块注释换成一个空格（避免粘连两个记号），行注释连同内容删除但保留换行。
    """
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch == '"' or ch == "'":
            # 只在同一行里找收尾引号；找不到就当普通字符（预处理行里的撇号等）
            j = i + 1
            while j < n and text[j] != ch and text[j] != "\n":
                j += 2 if text[j] == "\\" else 1
            if j < n and text[j] == ch:
                out.append(text[i : j + 1])
                i = j + 1
                continue
            out.append(ch)
            i += 1
        elif ch == "/" and text.startswith("/*", i):
            j = text.find("*/", i + 2)
            out.append(" ")
            i = n if j < 0 else j + 2
        elif ch == "/" and text.startswith("//", i):
            j = text.find("\n", i)
            i = n if j < 0 else j
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def _tidy_block(text: str) -> str:
    """多行文本（类型定义、宏）：去注释、统一换行、去掉行尾空白与空行。"""
    text = strip_c_comments(text.replace("\r\n", "\n").replace("\r", "\n"))
    lines = [ln.rstrip() for ln in text.split("\n")]
    return "\n".join(ln for ln in lines if ln.strip())


def _one_line(text: str) -> str:
    """单行文本（签名、原型）：去注释并压缩空白。"""
    return squash_whitespace(strip_c_comments(text))


_BANNER_RE = re.compile(r"^[-=_~+#*]{3,}\s.*\s[-=_~+#*]{3,}$")
_WORD_RE = re.compile(r"[^\W_]")   # 任意文字的字母 / 数字（Unicode）


def _is_banner(comment: str) -> bool:
    """「/* ---- 小节标题 ---- */」这类分隔注释不是函数文档。"""
    inner = re.sub(r"/\*+|\*+/|^\s*\*+|//+", " ", comment, flags=re.M)
    inner = squash_whitespace(inner)
    return not _WORD_RE.search(inner) or bool(_BANNER_RE.match(inner))


def _is_doxygen(comment: str) -> bool:
    return comment.lstrip().startswith(("/**", "/*!", "///", "//!"))


# --------------------------------------------------------------------------- 项目文件集合


@dataclass
class _Project:
    """一次提取共用的项目视图：根目录、项目文件集合与内容摘要（缓存键）。"""

    root: Path
    files: list[str]
    abs_set: set[str]
    digest: str


def _make_project(spec: CorpusSpec, src_root: Path | str) -> _Project:
    root = Path(src_root).resolve()
    rel = sorted(set(expand_patterns(spec.project_files, root, warn=True))
                 | set(expand_patterns(spec.target_files, root, warn=True)))
    h = hashlib.sha1()
    for r in rel:
        h.update(r.encode())
        h.update(b"\0")
        h.update((root / r).read_bytes())
        h.update(b"\0")
    abs_set = {os.path.realpath(root / r) for r in rel}
    return _Project(root=root, files=rel, abs_set=abs_set, digest=h.hexdigest())


# --------------------------------------------------------------------------- 公共：解析


def parse_file(spec: CorpusSpec, src_root: Path | str, file: str) -> ci.TranslationUnit:
    """按 spec.cflags 解析 src_root 下的一个文件，返回 TranslationUnit。

    调用方用 error_diagnostics(tu) 检查诊断；这里不因 error 级诊断抛异常。
    """
    root = Path(src_root).resolve()
    path = root / file
    if not path.is_file():
        raise FileNotFoundError(f"文件不存在: {path}")
    args = clangcfg.clang_args(list(spec.cflags), root)
    try:
        return clangcfg.index().parse(str(path), args=args, options=clangcfg.DEFAULT_PARSE_OPTIONS)
    except ci.TranslationUnitLoadError as e:
        raise ExtractError(f"libclang 无法加载 {path}（参数 {args}）: {e}") from e


def error_diagnostics(tu: ci.TranslationUnit, min_severity: int = ci.Diagnostic.Error) -> list[str]:
    """返回严重程度 >= min_severity 的诊断，格式 'file:line: message'。"""
    out = []
    for d in tu.diagnostics:
        if d.severity >= min_severity:
            f = d.location.file.name if d.location.file else "<unknown>"
            out.append(f"{f}:{d.location.line}: {d.spelling}")
    return out


# --------------------------------------------------------------------------- TU 索引

_TAG_KINDS = (CK.STRUCT_DECL, CK.UNION_DECL, CK.ENUM_DECL)
_TYPE_KINDS = (CK.TYPEDEF_DECL, CK.STRUCT_DECL, CK.UNION_DECL, CK.ENUM_DECL)
# 带 .referenced 的节点：函数体 / 类型定义里靠它们找到所依赖的实体
_REF_KINDS = frozenset({CK.DECL_REF_EXPR, CK.MEMBER_REF_EXPR, CK.MEMBER_REF, CK.TYPE_REF, CK.CALL_EXPR})
_DEFINE_PREFIX_RE = re.compile(rb"#[ \t]*define[ \t]+$")
_INLINE_RE = re.compile(r"\b(?:inline|__inline|__inline__)\b")


@functools.lru_cache(maxsize=1)
def _inline_checker():
    """clang_Cursor_isFunctionInlined：python 绑定没包装，直接从库里取；取不到就返回 None。"""
    try:
        fn = ci.conf.lib.clang_Cursor_isFunctionInlined
        fn.argtypes = [ci.Cursor]
        fn.restype = ctypes.c_uint
        return fn
    except AttributeError:
        return None


@dataclass
class _Top:
    """项目文件里的一条顶层声明。"""

    seq: int
    cursor: ci.Cursor
    file: str
    start: int
    end: int


@dataclass(eq=False)
class _Item:
    """依赖闭包里的一个条目：类型单元 / 函数原型 / 全局变量 / 宏。"""

    kind: str                   # type | func | global | macro
    key: tuple
    seq: int
    text: str
    file: str = ""
    lo: int = 0                 # 该条目的源码范围 [lo, hi)：范围内的宏展开属于它
    hi: int = 0
    roots: tuple = ()           # 需要遍历引用的 AST 根
    skip: Any = None            # 遍历根的直接子节点时的过滤函数（函数原型跳过函数体、全局变量跳过初始化器）
    macro_def: Any = None       # kind == macro 时的 _MacroDef


@dataclass
class _MacroDef:
    name: str
    file: str
    start: int                  # 宏名的偏移
    end: int
    seq: int
    cursor: ci.Cursor


class _TuIndex:
    """一个 TU 的符号索引与依赖闭包计算。只在解析它的那一次调用里存活。"""

    def __init__(self, tu: ci.TranslationUnit, proj: _Project, main_path: str):
        self.tu = tu
        self.proj = proj
        self.main_path = main_path
        self._src: dict[str, bytes] = {}
        self._real: dict[str, str] = {}
        self.func_defs: list[_Top] = []
        self.type_tops: dict[str, list[_Top]] = defaultdict(list)
        self.funcs: dict[str, list[_Top]] = defaultdict(list)
        self.globals: dict[str, list[_Top]] = defaultdict(list)
        self.global_starts: Counter = Counter()
        self.macro_by_name: dict[str, list[_MacroDef]] = defaultdict(list)
        self.macro_defs: dict[tuple[str, int], _MacroDef] = {}
        self.macro_uses: dict[str, list[tuple[int, tuple[str, int]]]] = defaultdict(list)
        self.includes: list[str] = []
        self._memo: dict[tuple, Optional[_Item]] = {}
        self._children: dict[tuple, list[_Item]] = {}
        self._build()

    # ---- 基础读写 ----

    def real(self, name: str) -> str:
        r = self._real.get(name)
        if r is None:
            r = self._real[name] = os.path.realpath(name)
        return r

    def src(self, path: str) -> bytes:
        s = self._src.get(path)
        if s is None:
            try:
                s = Path(path).read_bytes()
            except OSError as exc:
                log.warning("读不到源文件 %s（%s），依赖文本按空串处理", path, exc)
                s = b""
            self._src[path] = s
        return s

    def where(self, c: ci.Cursor) -> tuple[Optional[str], int, int]:
        """cursor 的 (规范化文件路径, 起始字节偏移, 结束字节偏移)；没有文件位置时文件为 None。"""
        ext = c.extent
        s = ext.start
        f = s.file
        if f is None:
            return None, 0, 0
        return self.real(f.name), s.offset, ext.end.offset

    def text(self, path: str, start: int, end: int) -> str:
        return self.src(path)[start:end].decode("utf-8", errors="replace")

    # ---- 建索引：只扫一遍 TU 顶层 ----

    def _build(self) -> None:
        proj = self.proj.abs_set
        decl_seq = 0
        pp_seq = 0
        for c in self.tu.cursor.get_children():
            k = kind_of(c)
            if k is None:
                continue
            if k == CK.MACRO_DEFINITION:
                pp_seq += 1
                f, s, e = self.where(c)
                if f in proj:
                    md = _MacroDef(c.spelling, f, s, e, pp_seq, c)
                    self.macro_defs[(f, s)] = md
                    self.macro_by_name[md.name].append(md)
            elif k == CK.MACRO_INSTANTIATION:
                f, s, _ = self.where(c)
                if f in proj:
                    ref = c.referenced
                    if ref is not None:
                        rf, rs, _ = self.where(ref)
                        if rf in proj:
                            self.macro_uses[f].append((s, (rf, rs)))
            elif k == CK.INCLUSION_DIRECTIVE:
                f, s, e = self.where(c)
                if f in proj:
                    self._note_include(c, f, s, e)
            elif k in _TYPE_KINDS or k in (CK.FUNCTION_DECL, CK.VAR_DECL):
                decl_seq += 1
                f, s, e = self.where(c)
                if f not in proj:
                    continue
                top = _Top(decl_seq, c, f, s, e)
                if k in _TYPE_KINDS:
                    self.type_tops[f].append(top)
                elif k == CK.FUNCTION_DECL:
                    usr = c.get_usr()
                    if usr:
                        self.funcs[usr].append(top)
                    if c.is_definition() and f == self.main_path:
                        self.func_defs.append(top)
                else:
                    usr = c.get_usr()
                    if usr:
                        self.globals[usr].append(top)
                    self.global_starts[(f, s)] += 1
        for uses in self.macro_uses.values():
            uses.sort(key=lambda t: t[0])

    def _note_include(self, c: ci.Cursor, f: str, s: int, e: int) -> None:
        """项目文件里指向「非项目文件」的 #include（系统头等）记入 includes。"""
        try:
            inc = c.get_included_file()
            inc_path = self.real(inc.name) if inc is not None else None
        except Exception as exc:  # noqa: BLE001 - 找不到被包含文件时，绑定可能抛异常
            log.debug("取 #include 指向的文件失败（%s: %s），按指向项目外处理: %s", type(exc).__name__, exc,
                      self.text(f, s, e).strip()[:80])
            inc_path = None
        if inc_path is not None and inc_path in self.proj.abs_set:
            return
        line = _one_line(self.text(f, s, e))
        if line.startswith("#") and line not in self.includes:
            self.includes.append(line)

    # ---- 引用 → 条目 ----

    def resolve(self, ref: ci.Cursor) -> Optional[_Item]:
        k = kind_of(ref)
        if k == CK.FUNCTION_DECL:
            return self._func_item(ref)
        if k == CK.VAR_DECL:
            return self._global_item(ref)
        if k in _TYPE_KINDS:
            return self._type_item(ref)
        if k in (CK.FIELD_DECL, CK.ENUM_CONSTANT_DECL):
            parent = ref.semantic_parent
            if parent is not None and kind_of(parent) in _TYPE_KINDS:
                return self._type_item(parent)
        return None

    def _func_item(self, ref: ci.Cursor) -> Optional[_Item]:
        usr = ref.get_usr()
        decls = self.funcs.get(usr) if usr else None
        if not decls:
            return None
        key = ("func", usr)
        if key in self._memo:
            return self._memo[key]
        first = decls[0]  # TU 顺序里最早的项目内声明：有前置原型就用原型，否则取定义头
        c = first.cursor
        hi = first.end
        skip = None
        if c.is_definition():
            body = next((ch for ch in c.get_children() if kind_of(ch) == CK.COMPOUND_STMT), None)
            if body is not None:
                hi = self.where(body)[1]
                skip = lambda ch: kind_of(ch) == CK.COMPOUND_STMT  # noqa: E731
        text = _one_line(self.text(first.file, first.start, hi))
        if not text.endswith(";"):
            text += ";"
        item = _Item("func", key, first.seq, text, first.file, first.start, hi, (c,), skip)
        self._memo[key] = item
        return item

    def _global_item(self, ref: ci.Cursor) -> Optional[_Item]:
        usr = ref.get_usr()
        decls = self.globals.get(usr) if usr else None
        if not decls:
            return None
        key = ("global", usr)
        if key in self._memo:
            return self._memo[key]
        first = decls[0]
        text, cut = self._extern_decl(first)
        hi = first.end if cut is None else cut

        def skip(ch: ci.Cursor, hi=hi) -> bool:  # 初始化器及其后的子树不属于 extern 声明
            return self.where(ch)[1] >= hi

        item = _Item("global", key, first.seq, text, first.file, first.start, hi, (first.cursor,), skip)
        self._memo[key] = item
        return item

    def _extern_decl(self, top: _Top) -> tuple[str, Optional[int]]:
        """由全局变量的声明生成 extern 声明，返回 (文本, 初始化器 '=' 的偏移)。

        单声明符：直接取源码文本，去掉 static / 初始化器，补 extern。
        同一语句多个声明符（int a = 1, b;）或文本里找不到变量名：改用 libclang 的类型拼写重建。
        """
        c = top.cursor
        name = c.spelling
        cut = self._initializer_offset(top)
        hi = top.end if cut is None else cut
        text = squash_whitespace(strip_c_comments(self.text(top.file, top.start, hi)))
        if self.global_starts[(top.file, top.start)] == 1 and re.search(rf"\b{re.escape(name)}\b", text):
            text = squash_whitespace(re.sub(r"\b(?:static|register|auto)\b\s*", "", text))
            if not re.match(r"extern\b", text):
                text = "extern " + text
            if re.search(r"\[\s*\]", text):
                text = self._complete_array_size(c, text)
            return text + ";", cut
        t = c.type
        if t.kind == ci.TypeKind.CONSTANTARRAY:
            decl = f"extern {t.element_type.spelling} {name}[{t.element_count}];"
        elif t.kind == ci.TypeKind.INCOMPLETEARRAY:
            decl = f"extern {t.element_type.spelling} {name}[];"
        elif "(*)" in t.spelling:
            decl = f"extern {t.spelling.replace('(*)', '(*' + name + ')', 1)};"
        else:
            decl = f"extern {t.spelling} {name};"
        return decl, cut

    @staticmethod
    def _complete_array_size(c: ci.Cursor, text: str) -> str:
        """省略长度的数组（T a[] = {...}）：extern 声明里补上由初始化器推出的长度，否则 sizeof 用不了。"""
        d = c.get_definition() or c
        t = d.type
        if t.kind == ci.TypeKind.CONSTANTARRAY and t.element_count >= 0:
            return re.sub(r"\[\s*\]", f"[{t.element_count}]", text, count=1)
        return text

    def _initializer_offset(self, top: _Top) -> Optional[int]:
        """全局变量初始化器 '=' 在文件里的字节偏移（在原文上扫描，跳过注释）。"""
        data = self.src(top.file)[top.start : top.end]
        depth = 0
        i, n = 0, len(data)
        while i < n:
            b = data[i : i + 1]
            if data.startswith(b"/*", i):
                j = data.find(b"*/", i + 2)
                i = n if j < 0 else j + 2
                continue
            if data.startswith(b"//", i):
                j = data.find(b"\n", i)
                i = n if j < 0 else j
                continue
            if b in (b'"', b"'"):
                j = i + 1
                while j < n and data[j : j + 1] != b and data[j : j + 1] != b"\n":
                    j += 2 if data[j : j + 1] == b"\\" else 1
                i = j + 1
                continue
            if b in (b"(", b"[", b"{"):
                depth += 1
            elif b in (b")", b"]", b"}"):
                depth -= 1
            elif b == b"=" and depth == 0:
                return top.start + i
            i += 1
        return None

    def _type_item(self, ref: ci.Cursor) -> Optional[_Item]:
        """类型实体 → 它所在的最外层顶层类型声明语句（typedef 语句或独立的 struct/union/enum）。"""
        d = ref
        if kind_of(d) in _TAG_KINDS:
            defn = d.get_definition()
            if defn is not None:
                d = defn
        f, s, e = self.where(d)
        if f is None or f not in self.type_tops:
            return None
        memo_key = ("typeref", f, s, e)
        if memo_key in self._memo:
            return self._memo[memo_key]
        tops = self.type_tops[f]
        best: Optional[_Top] = None
        for t in tops:
            if t.start <= s and e <= t.end and (best is None or (t.start, -t.end) < (best.start, -best.end)):
                best = t
        item: Optional[_Item] = None
        if best is not None:
            start = best.start
            end = max(t.end for t in tops if t.start == start)
            key = ("type", f, start)
            if key in self._memo:
                item = self._memo[key]
            else:
                bk = kind_of(best.cursor)
                unnamed = not best.cursor.spelling or "(unnamed" in best.cursor.spelling or "(anonymous" in best.cursor.spelling
                if bk in (CK.STRUCT_DECL, CK.UNION_DECL) and unnamed:
                    item = None  # 不在 typedef 里的匿名 struct / union：没法按名字引用，交给声明它的变量
                else:
                    members = [t for t in tops if start <= t.start and t.end <= end]
                    text = _tidy_block(self.text(f, start, end)) + ";"
                    item = _Item(
                        "type", key, min(t.seq for t in members), text, f, start, end, tuple(t.cursor for t in members)
                    )
                self._memo[key] = item
        self._memo[memo_key] = item
        return item

    def _macro_item(self, key: tuple[str, int]) -> Optional[_Item]:
        mk = ("macro", *key)
        if mk in self._memo:
            return self._memo[mk]
        md = self.macro_defs.get(key)
        item = None
        if md is not None:
            data = self.src(md.file)
            chunk_start = max(0, md.start - 96)
            m = _DEFINE_PREFIX_RE.search(data[chunk_start : md.start])
            begin = chunk_start + m.start() if m else md.start
            text = _tidy_block(data[begin : md.end].decode("utf-8", errors="replace"))
            item = _Item("macro", mk, md.seq, text, md.file, md.start, md.end, (), None, md)
        self._memo[mk] = item
        return item

    # ---- 遍历与闭包 ----

    def _nodes(self, roots: Sequence[ci.Cursor], skip: Any = None) -> Iterator[ci.Cursor]:
        """前序遍历若干根下的全部节点；skip 只作用于根的直接子节点。"""
        for root in roots:
            stack = [(root, True)]
            while stack:
                node, is_root = stack.pop()
                yield node
                kids = list(node.get_children())
                if is_root and skip is not None:
                    kids = [ch for ch in kids if not skip(ch)]
                stack.extend((ch, False) for ch in reversed(kids))

    def _macros_in(self, f: str, lo: int, hi: int) -> Iterator[tuple[str, int]]:
        uses = self.macro_uses.get(f)
        if not uses:
            return
        i = bisect.bisect_left(uses, lo, key=lambda t: t[0])
        while i < len(uses) and uses[i][0] < hi:
            yield uses[i][1]
            i += 1

    def children_of(self, item: _Item) -> list[_Item]:
        """条目直接依赖的其它条目（带缓存）。"""
        cached = self._children.get(item.key)
        if cached is not None:
            return cached
        out: dict[tuple, _Item] = {}
        if item.kind == "macro":
            for it in self._macro_children(item):
                out.setdefault(it.key, it)
        else:
            for node in self._nodes(item.roots, item.skip):
                if kind_of(node) in _REF_KINDS:
                    ref = node.referenced
                    if ref is not None:
                        it = self.resolve(ref)
                        if it is not None:
                            out.setdefault(it.key, it)
            for mkey in self._macros_in(item.file, item.lo, item.hi):
                it = self._macro_item(mkey)
                if it is not None:
                    out.setdefault(it.key, it)
        res = list(out.values())
        self._children[item.key] = res
        return res

    def _macro_children(self, item: _Item) -> Iterator[_Item]:
        """宏体里出现的其它项目宏（排除宏自己的名字与形参）。"""
        md: _MacroDef = item.macro_def
        toks = list(md.cursor.get_tokens())
        if not toks:
            return
        params: set[str] = set()
        body_from = 1
        if len(toks) > 1 and toks[1].spelling == "(" and toks[1].extent.start.offset == toks[0].extent.end.offset:
            for j in range(2, len(toks)):
                body_from = j + 1
                if toks[j].spelling == ")":
                    break
                if toks[j].kind == ci.TokenKind.IDENTIFIER:
                    params.add(toks[j].spelling)
        for t in toks[body_from:]:
            if t.kind != ci.TokenKind.IDENTIFIER or t.spelling in params or t.spelling == md.name:
                continue
            cands = self.macro_by_name.get(t.spelling)
            if not cands:
                continue
            before = [c for c in cands if c.seq < md.seq]
            pick = before[-1] if before else cands[0]
            it = self._macro_item((pick.file, pick.start))
            if it is not None:
                yield it

    def dependencies(self, root_items: list[_Item], exclude: set[tuple]) -> Dependencies:
        """从若干起始条目出发做传递闭包，排除 exclude 里的条目，按 TU 顺序输出各类文本。"""
        seen: dict[tuple, _Item] = {}
        todo = list(root_items)
        while todo:
            it = todo.pop()
            if it.key in seen or it.key in exclude:
                continue
            seen[it.key] = it
            todo.extend(self.children_of(it))
        by_kind: dict[str, list[_Item]] = defaultdict(list)
        for it in seen.values():
            by_kind[it.kind].append(it)

        def texts(kind: str) -> list[str]:
            out: list[str] = []
            for it in sorted(by_kind[kind], key=lambda x: x.seq):
                if it.text not in out:
                    out.append(it.text)
            return out

        return Dependencies(
            decls=texts("func"),
            types=texts("type"),
            globals=texts("global"),
            macros=texts("macro"),
            includes=list(self.includes),
        )

    # ---- 函数目标 ----

    def count_statements(self, body: ci.Cursor) -> int:
        """COMPOUND_STMT 的直接子节点之和，递归整棵子树（空语句不计）。"""
        n = 0
        stack = [body]
        while stack:
            node = stack.pop()
            kids = list(node.get_children())
            if kind_of(node) == CK.COMPOUND_STMT:
                n += sum(1 for k in kids if kind_of(k) != CK.NULL_STMT)
            stack.extend(kids)
        return n

    def doc_comment(self, cur: ci.Cursor, path: str, start: int) -> Optional[str]:
        rc = cur.raw_comment
        if not rc:
            return None
        rc = rc.replace("\r\n", "\n").strip()
        if not rc or _is_banner(rc):
            return None
        data = self.src(path)
        tail = data[max(0, start - len(rc.encode()) - 512) : start].decode("utf-8", errors="replace").replace("\r\n", "\n").rstrip()
        if tail.endswith(rc):
            # 紧贴定义的注释；但注释前面还有代码时是上一行代码的行尾注释，不是本函数的文档
            line_prefix = tail[: len(tail) - len(rc)].rsplit("\n", 1)[-1]
            return None if line_prefix.strip() else rc
        # 注释在别处（比如头文件的原型上）时只认文档风格的注释
        return rc if _is_doxygen(rc) else None

    def build_target(self, spec: CorpusSpec, top: _Top, rel_file: str) -> Optional[tuple[FunctionTarget, bool, bool]]:
        cur = top.cursor
        name = cur.spelling
        f, start, end = top.file, top.start, top.end
        body = next((ch for ch in cur.get_children() if kind_of(ch) == CK.COMPOUND_STMT), None)
        if body is None:
            return None
        bf, bstart, bend = self.where(body)
        data = self.src(f)
        if bf != f or not (start <= bstart < bend <= len(data)) or data[bstart : bstart + 1] != b"{" or data[bend - 1 : bend] != b"}":
            log.debug("跳过 %s:%s：函数体不是源码里直接写出的花括号块（可能由宏生成）", rel_file, name)
            return None
        sig_raw = data[start:bstart].decode("utf-8", errors="replace")
        signature = _one_line(sig_raw)
        if not re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", signature):
            log.debug("跳过 %s:%s：函数名不在源码签名里（由宏拼出）", rel_file, name)
            return None

        usr = cur.get_usr()
        callees: list[str] = []
        roots: dict[tuple, _Item] = {}
        for node in self._nodes([cur]):
            if kind_of(node) not in _REF_KINDS:
                continue
            ref = node.referenced
            if ref is None:
                continue
            it = self.resolve(ref)
            if it is None:
                continue
            roots.setdefault(it.key, it)
            if it.kind == "func" and kind_of(node) == CK.CALL_EXPR and it.key != ("func", usr):
                if ref.spelling not in callees:
                    callees.append(ref.spelling)
        for mkey in self._macros_in(f, start, end):
            it = self._macro_item(mkey)
            if it is not None:
                roots.setdefault(it.key, it)
        deps = self.dependencies(list(roots.values()), {("func", usr)})

        inline_fn = _inline_checker()
        is_inline = bool(inline_fn(cur)) if inline_fn is not None else bool(_INLINE_RE.search(signature))
        is_static = cur.storage_class == ci.StorageClass.STATIC
        target = FunctionTarget(
            target_id=f"{spec.name}:{rel_file}:{name}",
            corpus=spec.name,
            file=rel_file,
            name=name,
            signature=signature,
            start=start,
            end=end,
            body_start=bstart,
            body_end=bend,
            n_statements=self.count_statements(body),
            n_lines=data[start:end].count(b"\n") + 1,
            callees=callees,
            deps=deps,
            doc_comment=self.doc_comment(cur, f, start),
        )
        return target, is_static, is_inline


# --------------------------------------------------------------------------- 公共：提取

_LOCK = threading.RLock()
_CACHE: dict[tuple, list[tuple[FunctionTarget, bool, bool]]] = {}
_CACHE_MAX = 64


def _scan_file(spec: CorpusSpec, proj: _Project, rel_file: str) -> list[tuple[FunctionTarget, bool, bool]]:
    """提取一个文件里全部有函数体的定义（不做排除），带 (是否 static, 是否 inline) 标记。"""
    key = (
        spec.name,
        tuple(spec.cflags),
        str(proj.root),
        rel_file,
        proj.digest,
    )
    with _LOCK:
        hit = _CACHE.get(key)
        if hit is not None:
            return hit
        tu = parse_file(spec, proj.root, rel_file)
        fatal = error_diagnostics(tu, ci.Diagnostic.Fatal)
        if fatal:
            raise ExtractError(f"{rel_file} 解析出现致命错误: " + "; ".join(fatal[:5]))
        errs = error_diagnostics(tu)
        if errs:
            log.warning("%s 有 %d 条 error 级诊断，如 %s", rel_file, len(errs), errs[0])
        main_path = os.path.realpath(proj.root / rel_file)
        index = _TuIndex(tu, proj, main_path)
        found = []
        for top in index.func_defs:
            built = index.build_target(spec, top, rel_file)
            if built is not None:
                found.append(built)
        if len(found) < len(index.func_defs):
            log.info("%s：跳过 %d 个由宏生成、无法按字节偏移替换的函数定义", rel_file, len(index.func_defs) - len(found))
        if len(_CACHE) >= _CACHE_MAX:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[key] = found
        return found


def extract_file(spec: CorpusSpec, src_root: Path | str, file: str) -> list[FunctionTarget]:
    """提取单个文件里全部有函数体的定义，不做 main / static inline 之类的排除。"""
    proj = _make_project(spec, src_root)
    rel = Path(os.path.normpath(file)).as_posix()
    return [copy.deepcopy(t) for t, _, _ in _scan_file(spec, proj, rel)]


def extract_functions(spec: CorpusSpec, src_root: Path | str) -> list[FunctionTarget]:
    """提取 spec.target_files 里的全部目标函数候选。

    排除：main；`static inline` 且只有 0~1 条语句的函数。
    不在 target_files 里的文件（测试文件等）不会被解析。
    """
    proj = _make_project(spec, src_root)
    targets = expand_patterns(spec.target_files, proj.root, must_exist=True)
    out: list[FunctionTarget] = []
    for rel in targets:
        for t, is_static, is_inline in _scan_file(spec, proj, rel):
            if t.name == "main":
                continue
            if is_static and is_inline and t.n_statements <= 1:
                continue
            out.append(copy.deepcopy(t))
    return out


def locate(spec: CorpusSpec, src_root: Path | str, name: str, file: str) -> Optional[FunctionTarget]:
    """在 src_root（通常是混淆副本）的 file 里按 name 重新定位函数定义，找不到返回 None。

    不套用 extract_functions 的排除规则：副本里的语句数可能因变换而变化，
    只要原本被选作目标，就必须能找回来。
    """
    proj = _make_project(spec, src_root)
    rel = Path(os.path.normpath(file)).as_posix()
    for t, _, _ in _scan_file(spec, proj, rel):
        if t.name == name:
            return copy.deepcopy(t)
    return None
