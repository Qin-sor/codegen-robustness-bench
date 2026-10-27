"""结构混淆（L2，设计文档 §4.6）：纯语法层面的等价改写，不依赖任何标识符的名字。

对 ``spec.project_files`` 里的每个 C 源文件 / 头文件（``vendored`` 目录除外，测试文件照改）做五类变换，
只作用于函数体：

==================  ==========================================================================
``for_to_while``    ``for (i; c; s) B``  ->  ``{ i; while (c) { B s; } }``；没有 init 就不加外层花括号，
                    没有 cond 补 ``1``；body 里出现 ``continue``（含嵌套循环里的）一律不转
``if_else_swap``    ``if (c) A else B``  ->  ``if (!(c)) B else A``；非复合语句的分支补花括号，免得悬空 else
``compound_assign`` ``x op= e``          ->  ``x = x op (e)``；x 只能是标识符或 ``a->b`` / ``a.b`` 成员链
``ternary_return``  ``return c ? a : b;`` ->  ``if (c) return a; else return b;``
``decl_split``      ``int a = 1, *b;``   ->  ``int a = 1; int *b;``（同一行内用空格分隔，不增加行数）
==================  ==========================================================================

实现方式
--------
1. 每个文件用 libclang 解析一次，在函数体里收集「可变换位置」。每个位置是一个 :class:`Edit`：
   原文字节区间 + 新文本的配方（字面量与「原文子区间」交替）。渲染时子区间递归套用落在其中的其它变换，
   所以嵌套的位置（循环里的 if、循环头里的复合赋值、条件里的三目）可以同时生效，不需要「嵌套只改外层」，
   也不存在偏移失效问题；新文本的各个组成部分始终取自原文。
2. 每个函数体按 ``seed`` 抽约 ``ratio``（默认 50%）的位置：随机数发生器由 ``(seed, 相对路径, 函数序号)`` 派生，
   只依赖语法结构，不依赖标识符名字，所以在 L1 改名后的副本上选出的位置与改名前一一对应。
3. 变换后用 libclang 重新解析（不落盘），error 数必须为 0；原文件本身就有 error 的整份跳过并记录。
   有 error 时按二分法排查，只回滚引发错误的位置（最坏情况等价于回滚该文件的本轮全部变换），
   次数记在 ``rolled_back``。
4. 每处变换保持原文的总行数不变（缺的换行补在这处变换末尾），变换点之后的行号与 ``__LINE__`` 不变。

不变换的边界（一律跳过并按原因计入 ``skipped``）
------------------------------------------------
* 节点位于宏展开内部、与宏调用范围重合、或与宏调用部分重叠：``#x`` 字符串化、参数多次求值都会受影响。
* 由宏生成的语句（首个记号不是 ``for`` / ``if`` / ``return``）；被 ``#if`` 劈开的语句。
* 被搬动的片段里只允许成对出现的条件编译指令，且指令不能落在 ``else``、括号这类「胶水」位置上。
* ``x op= e``：e 里有函数调用（含宏展开出来的）、GNU 语句表达式、内联汇编时不拆，因为 ``x += f()`` 与
  ``x = x + f()`` 先读 x 还是先调 f 没有确定次序；x 是 volatile / _Atomic 时不拆。
* ``return c ? a : b``：a、b 的类型必须与整个三目的类型一致（或为 0..65535 的整型常量、提升到 int 的小整型、
  指针、void），否则通常算术转换（例如 ``-1`` 与 unsigned 取三目）会让拆开后的返回值不同；三目结果是浮点而
  分支是常量时，常量还得装得进函数返回类型（浮点转窄整型溢出是未定义行为）。
* 多声明拆分只认 ``[* 限定符]* 名字 [数组后缀]* [= 初始化器]`` 形态的声明符；函数指针、属性、
  内联 struct / enum 定义、typedef 等一律跳过。

已知限制
--------
* 靠别的头文件才能自洽的头文件（单独解析有 error）借一个包含它的项目 .c 当宿主解析；找不到干净的宿主就整份跳过，
  记在 ``files_skipped``。
* 不在默认配置里生效的 ``#if`` 分支里的代码 libclang 看不见，所以也不会被改。
* ``for`` 头、``else`` 前后这类「胶水」处的注释不保留；被移动的片段在变换内部的行号可能偏移。
* 极深的 else-if 链（嵌套超过 libclang 的 256 层括号上限）会被回滚逻辑截断。

对外接口
--------
``transform_structure(spec, src_root, seed) -> dict`` 就地修改 ``src_root`` 下的文件，返回五类变换各自的应用次数
（键见 ``KINDS``）、``total``、``candidates``（通过全部检查的可用位置数）、``skipped``（被放弃的位置及原因）、
``rolled_back``、逐文件明细 ``files``。``transform_source`` / ``transform_bytes`` 对单段源码做同样的变换，测试用。
"""
from __future__ import annotations

import bisect
import logging
import random
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import clang.cindex as ci

from bench import clangcfg, corpus
from bench.clangcfg import kind_of
from bench.obfuscate._edits import (
    Edit, FileStats, Parser, RecursionGuard, check_laminar, empty_stats, file_name, host_map,
    libclang_args, merge_stats, plan_files, render, rollback_invalid, run_files,
)

log = logging.getLogger(__name__)

CK = ci.CursorKind
TK = ci.TypeKind

__all__ = [
    "KINDS",
    "DEFAULT_RATIO",
    "transform_structure",
    "transform_source",
    "transform_bytes",
]

# 五类变换的名字，也是统计字典里的键
KINDS: tuple[str, ...] = ("for_to_while", "if_else_swap", "compound_assign", "ternary_return", "decl_split")
DEFAULT_RATIO = 0.5

COMPOUND_OPS = frozenset({"+=", "-=", "*=", "/=", "%=", "<<=", ">>=", "&=", "^=", "|="})
# 出现这些节点的右值不拆：函数调用与 x 的读取之间没有确定的先后次序（被调函数可能改 x），GNU 语句表达式和内联汇编同理
_SIDE_EFFECT_KINDS = frozenset(k for k in (CK.CALL_EXPR, getattr(CK, "StmtExpr", None), CK.ASM_STMT) if k is not None)
# 指针声明符里允许出现的限定符；它们在 ``*`` 右边时属于声明符，在类型左右时属于声明说明符
_QUALIFIERS = frozenset({"const", "volatile", "restrict", "__restrict", "__restrict__", "__const", "__volatile__"})
_OPEN = frozenset({"(", "[", "{"})
_CLOSE = {")": "(", "]": "[", "}": "{"}
# 允许出现在被移动片段里的条件编译指令
_COND_DIRECTIVES = frozenset({"if", "ifdef", "ifndef", "elif", "else", "endif"})
# 语句结尾已经自带终结符（花括号或分号）的种类
_SELF_TERMINATED = frozenset({CK.COMPOUND_STMT, CK.NULL_STMT, CK.DECL_STMT})
# 结尾取决于最后一个子语句的复合语句种类
_WRAPPER_STMTS = frozenset({CK.IF_STMT, CK.FOR_STMT, CK.WHILE_STMT, CK.SWITCH_STMT,
                            CK.LABEL_STMT, CK.CASE_STMT, CK.DEFAULT_STMT})
# return 的父节点是这些种类时，替换成 if/else 不需要再套花括号
_NO_BRACE_PARENTS = frozenset({CK.COMPOUND_STMT, CK.CASE_STMT, CK.DEFAULT_STMT, CK.LABEL_STMT})
_SMALL_INT_KINDS = frozenset({TK.BOOL, TK.CHAR_U, TK.UCHAR, TK.CHAR_S, TK.SCHAR, TK.SHORT, TK.USHORT})
_ARITH_KINDS = frozenset({
    TK.BOOL, TK.CHAR_U, TK.UCHAR, TK.CHAR16, TK.CHAR32, TK.USHORT, TK.UINT, TK.ULONG, TK.ULONGLONG,
    TK.UINT128, TK.CHAR_S, TK.SCHAR, TK.WCHAR, TK.SHORT, TK.INT, TK.LONG, TK.LONGLONG, TK.INT128,
    TK.FLOAT, TK.DOUBLE, TK.LONGDOUBLE,
})


class _Reject(Exception):
    """某个位置不能变换。``args[0]`` 是原因，写进 ``skipped`` 统计。"""


class _NotASite(_Reject):
    """这个节点根本不属于该变换的候选（例如没有 else 的 if、不是三目的 return），不计入跳过统计。"""


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def _children(c: ci.Cursor) -> list[ci.Cursor]:
    return list(c.get_children())


def _walk(c: ci.Cursor, include_self: bool = False) -> Iterator[ci.Cursor]:
    """前序遍历 c 的全部后代（include_self 为真时连 c 本身）；用显式栈，避免深层 AST 触发递归上限。"""
    stack = [c] if include_self else list(reversed(_children(c)))
    while stack:
        cur = stack.pop()
        yield cur
        stack.extend(reversed(_children(cur)))


def _peel_casts(c: ci.Cursor) -> ci.Cursor:
    """去掉隐式转换的外壳（libclang 把它们暴露成只有一个子节点的 UNEXPOSED_EXPR）。"""
    while kind_of(c) == CK.UNEXPOSED_EXPR:
        kids = _children(c)
        if len(kids) != 1:
            break
        c = kids[0]
    return c


def _simple_lvalue(c: ci.Cursor) -> bool:
    """左值是变量标识符，或以变量标识符为根的 ``.`` / ``->`` 成员链（没有下标、解引用、调用）。"""
    c = _peel_casts(c)
    k = kind_of(c)
    if k == CK.DECL_REF_EXPR:
        ref = c.referenced
        return ref is not None and kind_of(ref) in (CK.VAR_DECL, CK.PARM_DECL)
    if k == CK.MEMBER_REF_EXPR:
        kids = _children(c)
        return len(kids) == 1 and _simple_lvalue(kids[0])
    return False


_SIGNED_INT_KINDS = frozenset({TK.CHAR_S, TK.SCHAR, TK.SHORT, TK.INT, TK.LONG, TK.LONGLONG, TK.INT128, TK.WCHAR})
_UNSIGNED_INT_KINDS = frozenset({TK.CHAR_U, TK.UCHAR, TK.USHORT, TK.UINT, TK.ULONG, TK.ULONGLONG, TK.UINT128,
                                 TK.CHAR16, TK.CHAR32})
_FLOAT_KINDS = frozenset({TK.FLOAT, TK.DOUBLE, TK.LONGDOUBLE})


def _fits(value: int, ret: ci.Type) -> bool:
    """非负整数 value 能否无损地放进返回类型 ret（浮点和 _Bool 都按能放下处理）。"""
    t = ret.get_canonical()
    k = t.kind
    if k in _FLOAT_KINDS or k == TK.BOOL:
        return True
    bits = t.get_size() * 8
    if k in _SIGNED_INT_KINDS:
        return value <= (1 << (bits - 1)) - 1
    if k in _UNSIGNED_INT_KINDS:
        return value <= (1 << bits) - 1
    return value <= 127


def _conv_safe(branch: ci.Type, result: ci.Type, ret: ci.Type, const_value: int | None = None) -> bool:
    """三目的一个分支直接转成返回类型 ret，与「先转成三目结果类型 result 再转 ret」相比，值是否必然相同。

    const_value：该分支是 0..65535 的整型常量。整数之间的转换按 2^n 取模，先转 result 再转 ret 与直接转一样；
    若 result 是浮点，则要求常量能装进 ret（浮点转整数溢出是未定义行为，原代码与拆开后的行为可能不同）。
    """
    bk = branch.get_canonical().kind
    rk = result.get_canonical().kind
    if bk == rk:
        return True
    if rk in _ARITH_KINDS:
        if const_value is not None and bk in _ARITH_KINDS:
            return rk not in _FLOAT_KINDS or _fits(const_value, ret)
        # 小整型提升到 int 保值；其它算术转换（有符号转无符号、整数转浮点）可能改变值
        return rk == TK.INT and bk in _SMALL_INT_KINDS
    # 指针（另一支可能是空指针常量 0）、void：转换不改变值
    return rk in (TK.POINTER, TK.VOID)


_INT_LITERAL_RE = re.compile(r"^(0[xX][0-9a-fA-F]+|0[0-7]*|[1-9][0-9]*)[uUlL]*$")


def _int_literal_value(text: str) -> int | None:
    m = _INT_LITERAL_RE.match(text)
    if not m:
        return None
    body = m.group(1)
    if body[:2] in ("0x", "0X"):
        return int(body, 16)
    if len(body) > 1 and body[0] == "0":
        return int(body, 8)
    return int(body)


# ---------------------------------------------------------------------------
# 每个文件的记号索引与解析上下文
# ---------------------------------------------------------------------------
class _FileCtx:
    """一次解析的上下文：源码、去掉注释与预处理指令的记号表、宏调用区间、指令区间。"""

    def __init__(self, path: Path, data: bytes, tu: ci.TranslationUnit, name: str | None = None) -> None:
        self.path = path
        self.name = name or str(path)     # libclang 在这个 TU 里称呼该文件的名字（借宿主解析头文件时可能与 path 的拼写不同）
        self.data = data
        self.tu = tu
        self._newlines = [m.start() for m in re.finditer(b"\n", data)]
        self.S: list[int] = []        # 记号起点
        self.E: list[int] = []        # 记号终点
        self.T: list[str] = []        # 记号文本
        self.ID: list[bool] = []      # 是否标识符（不含关键字）
        self.dirs: list[tuple[int, int, str]] = []   # (起点, 终点, 指令名)，互不重叠、按起点排序
        self._dir_ends: list[int] = []
        self._tok_at: dict[int, int] = {}
        self._tokenize()
        self.macros: list[tuple[int, int, ci.Cursor]] = []
        self.macro_iv: list[tuple[int, int]] = []    # 合并后的最外层宏调用区间
        self._collect_macros()

    # -- 构建 ----------------------------------------------------------
    def line_of(self, off: int) -> int:
        return bisect.bisect_right(self._newlines, off - 1)

    def _directive_end(self, hash_off: int) -> int:
        """从 # 开始找逻辑行末（处理反斜杠续行）。"""
        data, pos = self.data, hash_off
        while True:
            nl = data.find(b"\n", pos)
            if nl == -1:
                return len(data)
            k = nl - 1
            if k >= 0 and data[k] == 0x0D:
                k -= 1
            if k >= 0 and data[k] == 0x5C:
                pos = nl + 1
                continue
            return nl

    def _tokenize(self) -> None:
        if not self.data:
            return
        ext = self.tu.get_extent(self.name, (0, len(self.data)))
        raw = [(t.kind.name, t.spelling, t.extent.start.offset, t.extent.end.offset)
               for t in self.tu.get_tokens(extent=ext)]
        last_line = -1
        i, n = 0, len(raw)
        while i < n:
            kind, text, s, e = raw[i]
            if kind == "COMMENT":
                i += 1
                continue
            if text == "#" and kind == "PUNCTUATION" and self.line_of(s) > last_line:
                dend = self._directive_end(s)
                name = ""
                j = i + 1
                while j < n and raw[j][2] < dend:
                    if not name and raw[j][0] != "COMMENT":
                        name = raw[j][1]
                    j += 1
                self.dirs.append((s, dend, name))
                last_line = self.line_of(max(dend - 1, s))
                i = j
                continue
            self.S.append(s)
            self.E.append(e)
            self.T.append(text)
            self.ID.append(kind == "IDENTIFIER")
            last_line = self.line_of(max(e - 1, s))
            i += 1
        self._tok_at = {s: k for k, s in enumerate(self.S)}
        self._dir_ends = [d[1] for d in self.dirs]

    def _collect_macros(self) -> None:
        for ch in self.tu.cursor.get_children():
            if kind_of(ch) != CK.MACRO_INSTANTIATION:
                continue
            f = ch.location.file
            if f is None or f.name != self.name:
                continue
            ex = ch.extent
            self.macros.append((ex.start.offset, ex.end.offset, ch))
        self.macros.sort(key=lambda m: (m[0], -m[1]))
        merged: list[list[int]] = []
        for s, e, _ in self.macros:
            if merged and s < merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        self.macro_iv = [(a, b) for a, b in merged]
        self._macro_starts = [a for a, _ in self.macro_iv]
        self._mac_starts = [m[0] for m in self.macros]

    # -- 记号查询 ------------------------------------------------------
    def tok_at(self, off: int) -> int | None:
        """起点恰为 off 的记号下标。"""
        return self._tok_at.get(off)

    def first_at_or_after(self, off: int) -> int:
        return bisect.bisect_left(self.S, off)

    def last_ending_by(self, off: int) -> int:
        """终点 <= off 的最后一个记号下标（没有则 -1）。"""
        return bisect.bisect_right(self.E, off) - 1

    def toks_within(self, a: int, b: int) -> range:
        """完全落在 [a, b) 内的记号下标区间。"""
        return range(self.first_at_or_after(a), bisect.bisect_right(self.E, b))

    def span(self, i: int, j: int) -> tuple[int, int] | None:
        """记号下标 [i, j) 对应的原文区间；空则返回 None。"""
        if i >= j:
            return None
        return self.S[i], self.E[j - 1]

    def match_close(self, i: int) -> int | None:
        """T[i] 是 ( [ { 之一，返回与之配对的右括号下标；配不上返回 None。"""
        stack: list[str] = []
        T = self.T
        for j in range(i, len(T)):
            t = T[j]
            if t in _OPEN:
                stack.append(t)
            elif t in _CLOSE:
                if not stack or stack[-1] != _CLOSE[t]:
                    return None
                stack.pop()
                if not stack:
                    return j
        return None

    def split_top(self, i: int, j: int, sep: str) -> list[int]:
        """下标 [i, j) 内、括号深度为 0 的 sep 记号下标。"""
        depth, out = 0, []
        T = self.T
        for k in range(i, j):
            t = T[k]
            if t in _OPEN:
                depth += 1
            elif t in _CLOSE:
                depth -= 1
            elif depth == 0 and t == sep:
                out.append(k)
        return out

    # -- 节点查询 ------------------------------------------------------
    def extent(self, c: ci.Cursor) -> tuple[int, int]:
        ex = c.extent
        f = ex.start.file
        if f is None or f.name != self.name:
            raise _Reject("other_file")
        s, e = ex.start.offset, ex.end.offset
        if e <= s:
            raise _Reject("empty_extent")
        return s, e

    def stmt_end(self, c: ci.Cursor) -> int:
        """语句的真实结尾：libclang 的表达式语句、return、break、do-while 等范围不含结尾的分号，这里补上。"""
        cur = c
        while True:
            k = kind_of(cur)
            if k in _SELF_TERMINATED:
                return cur.extent.end.offset
            if k in _WRAPPER_STMTS:
                kids = _children(cur)
                if kids:
                    cur = kids[-1]
                    continue
            break
        e = cur.extent.end.offset
        j = self.last_ending_by(e)
        if j < 0 or self.T[j] == ";":
            return e                # 范围本身已含分号（如 asm 语句），后面的分号是另一条空语句
        if j + 1 < len(self.S) and self.T[j + 1] == ";" and self.S[j + 1] >= e:
            return self.E[j + 1]
        return e

    # -- 宏与预处理指令 -------------------------------------------------
    def require_not_macro(self, s: int, e: int) -> None:
        """节点 [s, e) 必须与宏调用区间不冲突：与某个最外层宏调用重叠时，只允许完整包住它且范围不相等。"""
        iv = self.macro_iv
        if not iv:
            return
        i = bisect.bisect_right(self._macro_starts, s) - 1
        if i < 0:
            i = 0
        while i < len(iv) and iv[i][0] < e:
            ms, me = iv[i]
            i += 1
            if me <= s:
                continue
            if not (s <= ms and me <= e) or (ms == s and me == e):
                raise _Reject("macro")

    def macro_at(self, s: int, e: int) -> ci.Cursor | None:
        """范围恰为 [s, e) 的宏调用（有嵌套同范围的取最外层）。"""
        for k in range(bisect.bisect_left(self._mac_starts, s), len(self.macros)):
            ms, me, cur = self.macros[k]
            if ms != s:
                break
            if me == e:
                return cur
        return None

    def _dirs_overlapping(self, s: int, e: int) -> list[tuple[int, int, str]]:
        out = []
        for k in range(bisect.bisect_right(self._dir_ends, s), len(self.dirs)):
            if self.dirs[k][0] >= e:
                break
            out.append(self.dirs[k])
        return out

    def _balanced(self, a: int, b: int) -> bool:
        depth = 0
        for ds, _de, name in self._dirs_overlapping(a, b):
            if ds < a:
                continue
            if name in ("if", "ifdef", "ifndef"):
                depth += 1
            elif name == "endif":
                depth -= 1
                if depth < 0:
                    return False
            elif name in ("else", "elif") and depth == 0:
                return False
        return depth == 0

    def check_directives(self, s: int, e: int, allowed: Sequence[tuple[int, int]]) -> None:
        """[s, e) 内的预处理指令只能是成对的条件编译，且完整落在 allowed 区间（被原样搬动的片段）里。"""
        for ds, de, name in self._dirs_overlapping(s, e):
            if name not in _COND_DIRECTIVES:
                raise _Reject("directive")
            if not any(a <= ds and de <= b for a, b in allowed):
                raise _Reject("directive_in_gap")
        for a, b in allowed:
            if not self._balanced(a, b):
                raise _Reject("directive_unbalanced")


# ---------------------------------------------------------------------------
# 可变换位置的识别
# ---------------------------------------------------------------------------
class _SiteFinder:
    """遍历函数体，为五类变换各自生成 :class:`Edit`。"""

    def __init__(self, ctx: _FileCtx, kinds: Sequence[str]) -> None:
        self.ctx = ctx
        self.kinds = set(kinds)
        self.skipped: dict[str, Counter] = {k: Counter() for k in KINDS}
        self.candidates: Counter = Counter()
        self.rejected: list[tuple[str, str, int]] = []     # (种类, 原因, 节点起点)，排查用
        self.ret_type: ci.Type | None = None               # 当前函数的返回类型（find 时设置）
        self._dispatch = {
            CK.FOR_STMT: ("for_to_while", self._for),
            CK.IF_STMT: ("if_else_swap", self._if),
            CK.COMPOUND_ASSIGNMENT_OPERATOR: ("compound_assign", self._compound),
            CK.RETURN_STMT: ("ternary_return", self._return),
            CK.DECL_STMT: ("decl_split", self._decl),
        }

    def find(self, body: ci.Cursor, ret_type: ci.Type) -> list[Edit]:
        """body 是函数体的 COMPOUND_STMT，ret_type 是该函数的返回类型；返回按源码顺序（外层在前）排列的全部可用位置。"""
        self.ret_type = ret_type
        edits: list[Edit] = []
        stack: list[tuple[ci.Cursor, ci.CursorKind | None]] = [(body, None)]
        while stack:
            cur, parent = stack.pop()
            k = kind_of(cur)
            entry = self._dispatch.get(k)
            if entry is not None and entry[0] in self.kinds:
                name, fn = entry
                if k == CK.DECL_STMT and parent != CK.COMPOUND_STMT:
                    pass        # for 头里的声明、case 标签后的声明等：不是语句块里的独立声明
                else:
                    try:
                        edits.append(fn(cur, parent))
                        self.candidates[name] += 1
                    except _NotASite:
                        pass
                    except _Reject as why:
                        self.skipped[name][str(why)] += 1
                        self.rejected.append((name, str(why), cur.extent.start.offset))
                    except Exception as exc:    # noqa: BLE001 - 单个位置出意外不应拖垮整个文件，记录后放弃该位置
                        log.warning("%s: 分析 %s 位置 @%d 时出错，放弃该位置: %r", self.ctx.path.name, name,
                                    cur.extent.start.offset, exc)
                        self.skipped[name]["internal_error"] += 1
            for ch in reversed(_children(cur)):
                stack.append((ch, k))
        return edits

    # -- 公共检查 ------------------------------------------------------
    def _expect(self, i: int | None, text: str, why: str) -> int:
        if i is None or i >= len(self.ctx.T) or self.ctx.T[i] != text:
            raise _Reject(why)
        return i

    def _is_block(self, kid: ci.Cursor, a: int, b: int) -> bool:
        """分支语句是否本身就是花括号复合语句（否则交换时要补花括号）。"""
        return kind_of(kid) == CK.COMPOUND_STMT and self.ctx.data[a:a + 1] == b"{" and self.ctx.data[b - 1:b] == b"}"

    # -- 1. for -> while -----------------------------------------------
    def _for(self, c: ci.Cursor, _parent) -> Edit:
        ctx = self.ctx
        s, _e0 = ctx.extent(c)
        i0 = self._expect(ctx.tok_at(s), "for", "macro_generated")
        ip = self._expect(i0 + 1, "(", "macro_generated")
        iclose = ctx.match_close(ip)
        if iclose is None:
            raise _Reject("paren_mismatch")
        semis = ctx.split_top(ip + 1, iclose, ";")
        if len(semis) != 2:
            raise _Reject("header_shape")
        init = ctx.span(ip + 1, semis[0])
        cond = ctx.span(semis[0] + 1, semis[1])
        step = ctx.span(semis[1] + 1, iclose)
        kids = _children(c)
        if not kids:
            raise _Reject("no_body")
        body = kids[-1]
        bs, _ = ctx.extent(body)
        if iclose + 1 >= len(ctx.S) or ctx.S[iclose + 1] != bs:
            raise _Reject("body_start_mismatch")
        be = ctx.stmt_end(body)
        ctx.require_not_macro(s, be)
        if any(kind_of(d) == CK.CONTINUE_STMT for d in _walk(body, include_self=True)) or \
                any(ctx.T[j] == "continue" for j in ctx.toks_within(bs, be)):
            raise _Reject("has_continue")
        ctx.check_directives(s, be, [(bs, be)])

        parts: list = []
        if init:
            parts += [b"{ ", init, b"; "]
        parts += [b"while (", cond if cond else b"1", b") "]
        if step:
            parts += [b"{ ", (bs, be), b" ", step, b"; }"]
        else:
            parts += [(bs, be)]
        if init:
            parts += [b" }"]
        return Edit("for_to_while", s, be, parts)

    # -- 2. if/else 交换 -----------------------------------------------
    def _if(self, c: ci.Cursor, _parent) -> Edit:
        ctx = self.ctx
        kids = _children(c)
        if len(kids) != 3:
            raise _NotASite("no_else")
        s, _e0 = ctx.extent(c)
        i0 = self._expect(ctx.tok_at(s), "if", "macro_generated")
        ip = self._expect(i0 + 1, "(", "macro_generated")
        iclose = ctx.match_close(ip)
        if iclose is None or iclose == ip + 1:
            raise _Reject("paren_mismatch")
        cond = ctx.span(ip + 1, iclose)
        assert cond is not None
        then_c, else_c = kids[1], kids[2]
        ts, _ = ctx.extent(then_c)
        if iclose + 1 >= len(ctx.S) or ctx.S[iclose + 1] != ts:
            raise _Reject("then_start_mismatch")
        te = ctx.stmt_end(then_c)
        ie = ctx.first_at_or_after(te)
        self._expect(ie, "else", "else_not_found")
        if ie + 1 >= len(ctx.S):
            raise _Reject("else_not_found")
        es = ctx.S[ie + 1]
        if es != ctx.extent(else_c)[0]:
            raise _Reject("else_start_mismatch")
        ee = ctx.stmt_end(else_c)
        ctx.require_not_macro(s, ee)
        ctx.check_directives(s, ee, [(ts, te), (es, ee)])

        then_p: list = [(ts, te)] if self._is_block(then_c, ts, te) else [b"{ ", (ts, te), b" }"]
        else_p: list = [(es, ee)] if self._is_block(else_c, es, ee) else [b"{ ", (es, ee), b" }"]
        parts = [(s, cond[0]), b"!(", cond, b")", (cond[1], ts)] + else_p + [(te, es)] + then_p
        return Edit("if_else_swap", s, ee, parts)

    # -- 3. 复合赋值 ---------------------------------------------------
    def _compound(self, c: ci.Cursor, _parent) -> Edit:
        ctx = self.ctx
        kids = _children(c)
        if len(kids) != 2:
            raise _Reject("shape")
        s, e = ctx.extent(c)
        ctx.require_not_macro(s, e)
        i0 = ctx.tok_at(s)
        if i0 is None:
            raise _Reject("shape")
        # 左值只有 标识符 (. | ->) 标识符 ... 这几种记号，紧接着是复合赋值运算符
        j = i0
        T, ID = ctx.T, ctx.ID
        if not ID[j]:
            raise _Reject("lhs_not_simple")
        j += 1
        while j + 1 < len(T) and T[j] in (".", "->") and ID[j + 1]:
            j += 2
        if j >= len(T) or T[j] not in COMPOUND_OPS or ctx.E[j] > e:
            raise _Reject("lhs_not_simple")
        lhs_type = kids[0].type
        if not _simple_lvalue(kids[0]) or lhs_type.is_volatile_qualified() or "_Atomic" in lhs_type.spelling:
            # volatile 会多读一次，_Atomic 的复合赋值是单次原子读改写，拆开后语义不同
            raise _Reject("lhs_not_simple")
        if j + 1 >= len(T) or ctx.S[j + 1] != ctx.extent(kids[1])[0] or ctx.extent(kids[1])[1] != e:
            raise _Reject("shape")
        rs = ctx.S[j + 1]
        self._require_pure(kids[1])
        ctx.check_directives(s, e, [])
        lhs = (s, ctx.E[j - 1])
        rhs = (rs, e)
        op = T[j][:-1].encode()
        single = len(ctx.toks_within(rs, e)) == 1
        parts = [lhs, b" = ", lhs, b" " + op + b" ", rhs] if single else \
            [lhs, b" = ", lhs, b" " + op + b" (", rhs, b")"]
        return Edit("compound_assign", s, e, parts)

    def _require_pure(self, rhs: ci.Cursor) -> None:
        """右侧不能有函数调用（含宏展开出来的）、GNU 语句表达式、内联汇编。

        ``x += y++`` 之类的自增、赋值不必禁：它们若真的改到 x，原代码本身就是未定义行为（x 的读取与修改无序）。
        函数调用不同：被调函数改 x 是合法的，``x += f()`` 与 ``x = x + f()`` 先读 x 还是先调用 f 却可能不同。
        """
        for d in _walk(rhs, include_self=True):
            if kind_of(d) in _SIDE_EFFECT_KINDS:
                raise _Reject("rhs_side_effect")

    # -- 4. return 三目 -> if/else --------------------------------------
    def _return(self, c: ci.Cursor, parent) -> Edit:
        ctx = self.ctx
        kids = _children(c)
        if len(kids) != 1:
            raise _NotASite("not_ternary")
        node, parens = kids[0], 0
        while True:
            k = kind_of(node)
            sub = _children(node)
            if k == CK.UNEXPOSED_EXPR and len(sub) == 1:
                node = sub[0]
            elif k == CK.PAREN_EXPR and len(sub) == 1:
                node, parens = sub[0], parens + 1
            else:
                break
        if kind_of(node) != CK.CONDITIONAL_OPERATOR:
            raise _NotASite("not_ternary")
        s, _re = ctx.extent(c)
        cs, ce = ctx.extent(node)
        ctx.require_not_macro(cs, ce)
        i0 = self._expect(ctx.tok_at(s), "return", "macro_generated")
        ic0 = ctx.tok_at(cs)
        if ic0 is None or ic0 - i0 - 1 != parens or any(ctx.T[j] != "(" for j in range(i0 + 1, ic0)):
            raise _Reject("paren_shape")
        ic1 = ctx.last_ending_by(ce)
        if ic1 < 0 or ctx.E[ic1] != ce:
            raise _Reject("paren_shape")
        semi = ic1 + parens + 1
        if semi >= len(ctx.T) or any(ctx.T[j] != ")" for j in range(ic1 + 1, semi)) or ctx.T[semi] != ";":
            raise _Reject("paren_shape")
        # 顶层的 ? 与配对的 :
        depth = tern = 0
        q = col = None
        for j in range(ic0, ic1 + 1):
            t = ctx.T[j]
            if t in _OPEN:
                depth += 1
            elif t in _CLOSE:
                depth -= 1
            elif depth == 0 and t == "?":
                if tern == 0 and q is None:
                    q = j
                tern += 1
            elif depth == 0 and t == ":" and tern > 0:
                tern -= 1
                if tern == 0 and col is None:
                    col = j
        if q is None or col is None or q == ic0 or col == q + 1 or col == ic1:
            raise _Reject("ternary_shape")
        cond = ctx.span(ic0, q)
        then_r = ctx.span(q + 1, col)
        else_r = ctx.span(col + 1, ic1 + 1)
        assert cond and then_r and else_r
        tk = _children(node)
        if len(tk) != 3:
            raise _Reject("ternary_shape")
        result_type = node.type
        for br in (tk[1], tk[2]):
            br = _peel_casts(br)
            if not _conv_safe(br.type, result_type, self.ret_type, self._small_const(br)):
                raise _Reject("type_mismatch")
        end = ctx.E[semi]
        ctx.check_directives(s, end, [])
        parts: list = [b"if (", cond, b") return ", then_r, b"; else return ", else_r, b";"]
        if parent not in _NO_BRACE_PARENTS:
            parts = [b"{ "] + parts + [b" }"]
        return Edit("ternary_return", s, end, parts)

    def _small_const(self, c: ci.Cursor) -> int | None:
        """c 若是 0..65535 的整型常量（十进制/十六进制/八进制字面量、枚举常量，或展开成这类字面量的对象宏）返回其值。"""
        ctx = self.ctx
        while kind_of(c) in (CK.PAREN_EXPR, CK.UNEXPOSED_EXPR) and len(_children(c)) == 1:
            c = _children(c)[0]
        k = kind_of(c)
        if k == CK.DECL_REF_EXPR:
            ref = c.referenced
            if ref is not None and kind_of(ref) == CK.ENUM_CONSTANT_DECL and 0 <= ref.enum_value <= 65535:
                return ref.enum_value
            return None
        if k != CK.INTEGER_LITERAL:
            return None
        try:
            s, e = ctx.extent(c)
        except _Reject:
            return None
        toks = [ctx.T[j] for j in ctx.toks_within(s, e)]
        if len(toks) == 1 and ctx.ID[ctx.tok_at(s) or 0]:
            # 对象宏展开出来的字面量（cursor 自己的记号列表在宏展开处不可靠）：到宏定义里取体
            toks = []
            mac = ctx.macro_at(s, e)
            ref = mac.referenced if mac is not None else None
            if ref is not None:
                body = [t.spelling for t in ref.get_tokens()][1:]
                while len(body) >= 3 and body[0] == "(" and body[-1] == ")":
                    body = body[1:-1]
                toks = body
        if len(toks) != 1:
            return None
        v = _int_literal_value(toks[0])
        return v if v is not None and 0 <= v <= 65535 else None

    # -- 5. 多声明拆分 --------------------------------------------------
    def _decl(self, c: ci.Cursor, _parent) -> Edit:
        ctx = self.ctx
        kids = _children(c)
        if len(kids) < 2:
            raise _NotASite("single")
        if any(kind_of(k) != CK.VAR_DECL for k in kids):
            raise _Reject("not_plain_vars")
        s, e = ctx.extent(c)
        ctx.require_not_macro(s, e)
        ctx.check_directives(s, e, [])
        i_first = ctx.tok_at(s)
        i_semi = ctx.last_ending_by(e)
        if i_first is None or i_semi <= i_first or ctx.E[i_semi] != e or ctx.T[i_semi] != ";":
            raise _Reject("shape")
        names = []
        for k in kids:
            idx = ctx.tok_at(k.location.offset)
            if idx is None or ctx.T[idx] != k.spelling or not ctx.ID[idx]:
                raise _Reject("name_not_found")
            names.append(idx)
        commas = ctx.split_top(i_first, i_semi, ",")
        if len(commas) + 1 != len(kids):
            raise _Reject("comma_count")
        # 第一个声明符的起点：从名字往左吃掉 * 和夹在星号之间/左边的限定符，其余都属于声明说明符
        d0 = names[0]
        k = names[0] - 1
        while k >= i_first:
            t = ctx.T[k]
            if t == "*":
                d0 = k
            elif t not in _QUALIFIERS:
                break
            k -= 1
        spec_toks = ctx.T[i_first:d0]
        if not spec_toks or spec_toks.count("(") != spec_toks.count(")") or "{" in spec_toks:
            raise _Reject("declarator_shape")
        bounds = [d0] + [x + 1 for x in commas]
        ends = commas + [i_semi]
        for b0, b1, nm in zip(bounds, ends, names):
            if not self._declarator_ok(b0, b1, nm):
                raise _Reject("declarator_shape")
        specs = (ctx.S[i_first], ctx.E[d0 - 1])
        parts: list = []
        for n, (b0, b1) in enumerate(zip(bounds, ends)):
            if n:
                parts.append(b" ")      # 用空格而不是换行分隔，保持行数不变
            parts += [specs, b" ", (ctx.S[b0], ctx.E[b1 - 1]), b";"]
        return Edit("decl_split", s, e, parts)

    def _declarator_ok(self, i0: int, i1: int, name_idx: int) -> bool:
        """tokens[i0:i1] 是否形如 ``[* | 限定符]* 名字 [数组后缀]* [= 初始化器]``。"""
        ctx = self.ctx
        j = i0
        while j < i1 and (ctx.T[j] == "*" or ctx.T[j] in _QUALIFIERS):
            j += 1
        if j != name_idx:
            return False
        j += 1
        while j < i1 and ctx.T[j] == "[":
            k = ctx.match_close(j)
            if k is None or k >= i1:
                return False
            j = k + 1
        if j == i1:
            return True
        return ctx.T[j] == "=" and j + 1 < i1


# ---------------------------------------------------------------------------
# 文件级流程
# ---------------------------------------------------------------------------
class _Transformer:
    def __init__(self, args: Sequence[str], seed: int, ratio: float, kinds: Sequence[str]) -> None:
        unknown = [k for k in kinds if k not in KINDS]
        if unknown:
            raise ValueError(f"未知的变换种类: {unknown}（可用: {list(KINDS)}）")
        self.parser = Parser(args)
        self.seed = seed
        self.ratio = ratio
        self.kinds = tuple(kinds)

    def transform_bytes(self, path: Path, data: bytes, rel: str, hosts: Sequence[Path] = ()) -> tuple[bytes, FileStats]:
        """变换一个文件的内容，返回 (新内容, 统计)。path 只用于定位 include 与诊断，不要求磁盘上存在。"""
        fs = FileStats(KINDS)
        try:
            unit, tu, base_errs = self.parser.open(path, data, hosts)
        except ci.TranslationUnitLoadError as exc:
            fs.skipped_file = f"解析失败: {exc}"
            log.warning("%s: %s", rel, fs.skipped_file)
            return data, fs
        if base_errs:
            fs.skipped_file = f"原文件解析已有 {len(base_errs)} 个 error，首个: {base_errs[0]}"
            log.warning("%s: %s", rel, fs.skipped_file)
            return data, fs

        if unit.host is not None:
            fs.host = unit.host.name
        ctx = _FileCtx(path, data, tu, file_name(tu, path) or str(path))
        finder = _SiteFinder(ctx, self.kinds)
        chosen: list[Edit] = []
        func_idx = 0
        for fn in self._functions(ctx):
            body = next((k for k in _children(fn) if kind_of(k) == CK.COMPOUND_STMT), None)
            if body is None:
                continue
            rng = random.Random(f"{self.seed}:{rel}:{func_idx}")
            func_idx += 1
            sites = finder.find(body, fn.result_type)
            chosen.extend(ed for ed in sites if rng.random() < self.ratio)
        fs.candidates = finder.candidates
        fs.skipped = finder.skipped

        chosen, fs.overlap = check_laminar(chosen)
        good = rollback_invalid(chosen, lambda eds: self.parser.valid(unit, render(data, eds)), fs.rolled_back, rel)
        if not good:
            return data, fs
        new = render(data, good)
        if len(good) != len(chosen):
            # 走过二分回滚：最终文本再整体验证一次，万一各段单独没问题、合起来有问题，宁可整文件不动
            ok, errs = self.parser.valid(unit, new)
            if not ok:
                fs.skipped_file = f"回滚后仍有 error，该文件保持原样，首个: {errs[0]}"
                log.warning("%s: %s", rel, fs.skipped_file)
                for ed in good:
                    fs.rolled_back[ed.kind] += 1
                return data, fs
        for ed in good:
            fs.applied[ed.kind] += 1
        return new, fs

    @staticmethod
    def _functions(ctx: _FileCtx) -> Iterator[ci.Cursor]:
        for ch in ctx.tu.cursor.get_children():
            if kind_of(ch) == CK.FUNCTION_DECL and ch.is_definition():
                f = ch.location.file
                if f is not None and f.name == ctx.name:
                    yield ch


# ---------------------------------------------------------------------------
# 对外接口
# ---------------------------------------------------------------------------
def transform_bytes(data: bytes, args: Sequence[str] = (), seed: int = 0, *, ratio: float = DEFAULT_RATIO,
                    kinds: Sequence[str] | None = None, filename: str = "snippet.c",
                    workdir: Path | None = None) -> tuple[bytes, dict]:
    """变换一段 C 源码（bytes），返回 (新源码, 统计)。args 是给 libclang 的额外参数（-I、-D、-std 等）。

    filename 相对 workdir（默认当前目录）解析，只用来定位 ``#include "..."``，磁盘上不必真有这个文件。
    """
    kinds = tuple(kinds) if kinds is not None else KINDS
    tr = _Transformer(["-x", "c", "-w", "-ferror-limit=0", *clangcfg.builtin_include_args(), *args], seed, ratio, kinds)
    path = (Path(workdir) if workdir else Path.cwd()).resolve() / filename
    with RecursionGuard():
        new, fs = tr.transform_bytes(path, data, filename)
    stats = empty_stats(KINDS, seed, ratio, kinds)
    merge_stats(stats, filename, fs, new != data)
    return new, stats


def transform_source(source: str, args: Sequence[str] = (), seed: int = 0, *, ratio: float = DEFAULT_RATIO,
                     kinds: Sequence[str] | None = None, filename: str = "snippet.c",
                     workdir: Path | None = None) -> tuple[str, dict]:
    """:func:`transform_bytes` 的字符串版本，供测试和交互使用。"""
    new, stats = transform_bytes(source.encode("utf-8"), args, seed, ratio=ratio, kinds=kinds,
                                 filename=filename, workdir=workdir)
    return new.decode("utf-8"), stats


def transform_structure(spec: corpus.CorpusSpec, src_root: Path, seed: int, *, ratio: float = DEFAULT_RATIO,
                        kinds: Sequence[str] | None = None, files: Iterable[str] | None = None) -> dict:
    """对 src_root 下 spec.project_files 里的 C 文件做 L2 结构混淆（就地修改），返回统计字典。

    返回值里 ``for_to_while`` / ``if_else_swap`` / ``compound_assign`` / ``ternary_return`` / ``decl_split``
    是每类变换的应用次数，``total`` 是总数；``candidates`` 是通过全部检查的可用位置数，``skipped`` 是被放弃的位置
    及原因，``rolled_back`` 是重新解析出错而回滚的次数，``files`` 是逐文件明细。

    可选参数（接口契约之外的扩展）：``ratio`` 每个函数体内的抽样比例；``kinds`` 只启用部分变换；
    ``files`` 只处理这些相对路径（仍受 vendored 过滤）。
    """
    src_root = Path(src_root).resolve()
    if not src_root.is_dir():
        raise FileNotFoundError(f"源码目录不存在: {src_root}")
    kinds = tuple(kinds) if kinds is not None else KINDS
    tr = _Transformer(libclang_args(spec, src_root), seed, ratio, kinds)
    stats = empty_stats(KINDS, seed, ratio, kinds)

    project, todo = plan_files(spec, src_root, files=files)
    hosts = host_map(p for p in project if p.suffix == ".c")
    with RecursionGuard():
        run_files(todo, src_root, KINDS, stats,
                  lambda path, data, rel: tr.transform_bytes(path, data, rel, hosts.get(path.name, ())), "结构混淆")
    log.info("结构混淆 %s: 共 %d 处 %s，%d/%d 个文件有改动", spec.name, stats["total"],
             {k: stats[k] for k in KINDS}, stats["files_changed"], stats["files_total"])
    return stats
