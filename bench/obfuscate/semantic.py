"""语义混淆（L3，设计文档 4.6）：表达式级的等价改写。

在 L1（符号）、L2（结构）之上，对项目文件里的函数体做五类改写，改写前后行为完全等价：

==================  ===========================================================================
``relation_flip``   ``a < b`` → ``b > a``，``a <= b`` → ``b >= a``（反向的 ``>`` / ``>=`` 同理）
``de_morgan``       ``!(a && b)`` → ``(!a || !b)``，``!(a || b)`` → ``(!a && !b)``
``eq_zero``         ``!x`` → ``(x == 0)``，``x == 0`` → ``!x``（x 为整型，不是指针）
``commute``         ``a + b`` → ``b + a``，``a * b`` → ``b * a``（操作数无副作用，非指针运算）
``opaque``          函数体首个 ``{`` 之后插入 ``{ volatile int v_<n> = 0; (void)v_<n>; }``（变量名走 L1 的取名规则）
==================  ===========================================================================

作用范围与结构混淆相同：``spec.project_files`` 里的 ``.c`` / ``.h``（含测试文件）的每个函数体；
第三方框架目录（``spec.vendored``）默认不动，``include_vendored=True`` 才一并处理。
函数体之外（全局初始化、宏定义、类型定义）和宏展开产生的节点一律不碰。

实现要点
--------
* 每个文件用 libclang 解析一次，在 AST 节点上判定可改位置，按字节偏移改源码。文件按 latin-1 解码，
  字符下标与字节偏移一一对应，写回时按 latin-1 编码，原文里的任何字节（含 UTF-8 中文）都原样保留。
* ``BINARY_OPERATOR`` / ``UNARY_OPERATOR`` 的操作符拼写靠记号（token）判定：自带一个小词法器，
  要求「两个操作数之间恰好一个记号，且就是该操作符」「``!`` 后面紧跟操作数」，且 AST 范围必须落在记号边界上。
* 交换律与关系反转只在两个操作数都是 ``DECL_REF_EXPR`` / ``INTEGER_LITERAL`` / ``MEMBER_REF_EXPR``
  （字符、浮点字面量按字面量一并算；成员链里不得有函数调用、``++`` / ``--``、赋值、解引用）、
  且类型是算术类型（不是指针 / 数组 / 函数）时才做；类型取 ``cursor.type`` 的规范类型。
* ``!x`` 与 ``x == 0`` 互换要求 x 为整型（含枚举、bool），不是指针，也不是浮点。
* De Morgan 只处理源码里显式写出的 ``!( ... && ... )`` 与 ``!( ... || ... )``，操作数按优先级补括号；
  ``&&`` / ``||`` 的短路求值顺序保持不变，所以操作数可以带副作用。
* 范围内含宏调用的节点一律跳过（用 libclang 的 MACRO_INSTANTIATION 区间判定）：宏参数、宏体里的节点范围会
  塌缩到宏名上，宏展开后的优先级也看不见，改写它们不安全。含预处理指令行、行拼接、``__LINE__`` 一类
  内建宏的范围同样跳过。
* 嵌套的改写（例如 ``!(a < b && c)`` 外层 De Morgan、内层关系反转）由外向内递归渲染：每处改写只保留操作数
  的原文区间，落在区间里的内层改写自然生效。同一个 ``!`` 节点同时满足 De Morgan 和 eq_zero 时优先 De Morgan。
* 保持行数不变：替换文本丢掉了换行时，在替换文本末尾补回同样数量的换行；opaque 语句不换行。
* opaque 变量名走与 L1 同一张改名表的取名规则（``build_rename_map``，种类按变量，形如 ``v_<n>``），所以和 L1 改过的变量名
  长得一样，不是一眼能认出来的固定模式；由 (seed, 文件) 决定，并避开项目所有文件里出现过的拼写。
* 每个文件改完用 libclang 重新解析（不落盘）：必须零 error。有 error 时按改写二分排查，只回滚出错的改写，
  回滚次数按类别记在 ``rolled_back``；合并后仍失败则整文件放弃。原文件本身解析就有 error 的直接跳过。
* 全程确定：文件按相对路径排序，每个函数体的随机数发生器由 (seed, 相对路径, 函数序号) 派生，只依赖语法结构，
  不依赖标识符名字。

已知限制
--------
* 只改 ``spec.cflags`` 这套配置下生效的代码；``#if`` / ``#ifdef`` 里没有生效的分支原样保留（它们本来就是合法的原文）。
* opaque 块紧跟在函数体 ``{`` 之后，所以在 C89 / ``-std=c89`` 下它出现在局部声明之前，属于「声明与语句混用」。
  gcc / clang 默认放行（只有 ``-pedantic`` 才警告，``-pedantic-errors`` 会报错）；9 个语料的构建命令都不带这两个开关。
* 交换律 / 关系反转只认 ``DECL_REF_EXPR`` / 字面量 / 纯成员链（任务口径），``a[i] < n``、``x < -1`` 这类不改；
  操作数含宏调用（包括 ``INT_MAX`` 这类常量宏）的位置一律放弃，统计里记在 ``skipped``。
* 原文件单独解析就有 error 的文件整份跳过（例如 jsmn 的 ``test/testutil.h`` 不自含，只有作为 ``tests.c`` 的一部分才干净）。

对外接口
--------
``transform_semantic(spec, src_root, seed) -> dict`` 就地修改 ``src_root`` 下的文件（形状与
``structure.transform_structure`` 的返回值一致）：五类改写各自的应用次数（键见 ``KINDS``）、``total``、
``candidates``（通过全部检查的可用位置数）、``skipped``（被放弃的位置及原因）、``rolled_back``、逐文件明细。
``transform_source`` / ``transform_bytes`` 对单段源码做同样的改写，测试用。
"""
from __future__ import annotations

import itertools
import logging
import os
import random
import re
from array import array
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional, Sequence

import clang.cindex as ci

from bench import clangcfg, corpus
from bench.clangcfg import kind_of
from bench.obfuscate._edits import (
    Edit, FileStats, Parser, RecursionGuard, check_laminar, empty_stats, host_map, libclang_args, merge_stats,
    plan_files, render, rollback_invalid, run_files, file_name,
)
from bench.obfuscate.symbols import build_rename_map

log = logging.getLogger(__name__)

CK = ci.CursorKind
TK = ci.TypeKind

__all__ = [
    "KINDS",
    "DEFAULT_RATIO",
    "transform_semantic",
    "transform_source",
    "transform_bytes",
]

# 五类改写的名字，也是统计字典里的键
KINDS: tuple[str, ...] = ("relation_flip", "de_morgan", "eq_zero", "commute", "opaque")
DEFAULT_RATIO = 0.5

# --------------------------------------------------------------------------- 词法器

_TOKEN_RE = re.compile(
    r"""
      (?P<ws>(?:[ \t\r\f\v\n]|\\\r?\n)+)
    | (?P<lc>//[^\n]*)
    | (?P<bc>/\*.*?\*/)
    | (?P<str>(?:u8|u|U|L)?"(?:[^"\\\n]|\\.)*")
    | (?P<chr>(?:u8|u|U|L)?'(?:[^'\\\n]|\\.)*')
    | (?P<num>\.?[0-9](?:[eEpP][+\-]|[A-Za-z0-9_.])*)
    | (?P<id>[A-Za-z_][A-Za-z0-9_]*)
    | (?P<p>%:%:|<<=|>>=|\.\.\.|->|\+\+|--|<<|>>|<=|>=|==|!=|&&|\|\||\+=|-=|\*=|/=|%=|&=|\|=|\^=|\#\#
            |[-+*/%&|^~!<>=?:;,.(){}\[\]\#])
    | (?P<o>.)
    """,
    re.X | re.S,
)
_SKIP_GROUPS = frozenset({"ws", "lc", "bc"})
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# 改写范围里不允许出现的东西：预处理指令行、行拼接、会随位置变化的内建宏
_DIRECTIVE_RE = re.compile(r"(?m)^[ \t]*#")
_BUILTIN_MACRO_RE = re.compile(r"\b__(?:LINE|FILE|BASE_FILE|COUNTER|DATE|TIME|TIMESTAMP|INCLUDE_LEVEL)__\b")


class _Toks:
    """一个文件的记号表（跳过空白和注释）：起止偏移、种类，以及按偏移反查记号下标。"""

    def __init__(self, text: str) -> None:
        self.text = text
        starts: list[int] = []
        ends: list[int] = []
        kinds: list[str] = []
        for m in _TOKEN_RE.finditer(text):
            g = m.lastgroup
            if g in _SKIP_GROUPS:
                continue
            starts.append(m.start())
            ends.append(m.end())
            kinds.append(g)
        self.starts = starts
        self.ends = ends
        self.kinds = kinds
        self.start_idx = {s: i for i, s in enumerate(starts)}
        self.end_idx = {e: i for i, e in enumerate(ends)}

    def spell(self, i: int) -> str:
        return self.text[self.starts[i]:self.ends[i]]

    def aligned(self, s: int, e: int) -> bool:
        """[s, e) 非空，且两端恰好落在记号边界上。"""
        return s < e and s in self.start_idx and e in self.end_idx


# --------------------------------------------------------------------------- libclang 小工具


def _span(c: ci.Cursor) -> tuple[int, int]:
    e = c.extent
    return e.start.offset, e.end.offset


def _types(names: Iterable[str]) -> frozenset:
    return frozenset(getattr(TK, n) for n in names if hasattr(TK, n))


_INT_TYPES = _types(["BOOL", "CHAR_U", "UCHAR", "CHAR16", "CHAR32", "USHORT", "UINT", "ULONG", "ULONGLONG",
                     "UINT128", "CHAR_S", "SCHAR", "WCHAR", "SHORT", "INT", "LONG", "LONGLONG", "INT128", "ENUM"])
_ARITH_TYPES = _INT_TYPES | _types(["FLOAT", "DOUBLE", "LONGDOUBLE"])

# 原子操作数允许的节点种类（成员链另算）
_ATOM_KINDS = frozenset({CK.DECL_REF_EXPR, CK.INTEGER_LITERAL, CK.CHARACTER_LITERAL, CK.FLOATING_LITERAL})
# 前面直接加 ! 不会改变结合方式的节点（基本 / 后缀 / 一元表达式），其余补括号
_NO_PAREN_KINDS = frozenset({CK.DECL_REF_EXPR, CK.INTEGER_LITERAL, CK.CHARACTER_LITERAL, CK.FLOATING_LITERAL,
                             CK.STRING_LITERAL, CK.PAREN_EXPR, CK.CALL_EXPR, CK.MEMBER_REF_EXPR,
                             CK.ARRAY_SUBSCRIPT_EXPR, CK.UNARY_OPERATOR})

_REL_FLIP = {"<": ">", "<=": ">=", ">": "<", ">=": "<="}
_COMMUTE_OPS = frozenset({"+", "*"})


def _canon(node: ci.Cursor):
    try:
        return node.type.get_canonical().kind
    except Exception as exc:  # noqa: BLE001 - 个别节点取不到类型：当作「不是算术类型」，这个位置就不会被改写
        log.debug("取节点类型失败（%s: %s），该位置不改写", type(exc).__name__, exc)
        return None


class _Reject(Exception):
    """某个位置不能改写。``args[0]`` 是原因，写进 ``skipped`` 统计。"""


# 跳过原因（统计里的键）
R_MACRO = "范围含宏调用 / 预处理指令 / 内建宏"
R_NOT_ATOMIC = "操作数不是原子表达式"
R_NOT_ARITH = "操作数类型不是算术类型"
R_NOT_INT = "操作数类型不是整型"
R_SAME = "两侧文本相同"
R_SHAPE = "括号内结构不规整"
R_BRACE = "函数体的花括号不在源码里"


# --------------------------------------------------------------------------- 单个文件的上下文


class _FileCtx:
    """一次解析的上下文：源码（latin-1 字符串）、记号表、宏调用区间前缀和、本文件里的函数体。"""

    def __init__(self, path: Path, data: bytes, tu: ci.TranslationUnit, name: str | None = None) -> None:
        self.path = path
        self.name = name or str(path)       # libclang 在这个 TU 里称呼该文件的名字（借宿主解析头文件时可能与 path 的拼写不同）
        self.real = os.path.realpath(self.name)
        self.text = data.decode("latin-1")
        self.toks = _Toks(self.text)
        self._real_cache: dict[str, bool] = {}
        self.bodies: list[ci.Cursor] = []
        intervals: list[tuple[int, int]] = []
        for c in tu.cursor.get_children():
            k = kind_of(c)
            if k is None or not self._in_main(c):
                continue
            if k == CK.MACRO_INSTANTIATION:
                intervals.append(_span(c))
            elif k == CK.FUNCTION_DECL and c.is_definition():
                body = None
                for ch in c.get_children():
                    if kind_of(ch) == CK.COMPOUND_STMT:
                        body = ch
                if body is not None:
                    self.bodies.append(body)
        n = len(self.text)
        cover = bytearray(n + 1)
        for s, e in intervals:
            s = max(0, min(s, n))
            e = max(s, min(e, n))
            cover[s:e] = b"\x01" * (e - s)
        self.macro_prefix = array("I", itertools.accumulate(cover, initial=0))   # 前缀和：[a, b) 里被宏占的字节数

    def _in_main(self, c: ci.Cursor) -> bool:
        f = c.location.file
        if f is None:
            return False
        hit = self._real_cache.get(f.name)
        if hit is None:
            hit = self._real_cache[f.name] = os.path.realpath(f.name) == self.real
        return hit

    def macro_hit(self, a: int, b: int) -> bool:
        return self.macro_prefix[b] - self.macro_prefix[a] > 0

    def clean(self, a: int, b: int) -> bool:
        """[a, b) 是不是一段「原样的源码」：不含宏调用、预处理指令 / 行拼接、内建宏。"""
        if not (0 <= a < b <= len(self.text)) or self.macro_hit(a, b):
            return False
        seg = self.text[a:b]
        return not ("\\\n" in seg or "\\\r" in seg or _DIRECTIVE_RE.search(seg) or _BUILTIN_MACRO_RE.search(seg))


# --------------------------------------------------------------------------- 找可改位置


class _SiteFinder:
    """在函数体的 AST 里找可改写的位置；每个位置是一组备选改写（同一个节点最多选其中一个）。"""

    def __init__(self, ctx: _FileCtx, kinds: Sequence[str]) -> None:
        self.ctx = ctx
        self.t = ctx.toks
        self.text = ctx.text
        self.kinds = frozenset(kinds)
        self.candidates: Counter = Counter()
        self.skipped: dict[str, Counter] = {k: Counter() for k in KINDS}

    def find(self, body: ci.Cursor) -> list[list[Edit]]:
        sites: list[list[Edit]] = []
        stack = [body]
        while stack:
            n = stack.pop()
            kids = list(n.get_children())
            k = kind_of(n)
            if k == CK.BINARY_OPERATOR:
                opts = self._binary(n, kids)
            elif k == CK.UNARY_OPERATOR:
                opts = self._unary(n, kids)
            else:
                opts = []
            if opts:
                sites.append(opts)
            stack.extend(reversed(kids))
        return sites

    def opaque_site(self, body: ci.Cursor) -> Optional[int]:
        """函数体首个 ``{`` 的偏移；这个花括号不在源码里（宏生成）则记一次跳过并返回 None。"""
        if "opaque" not in self.kinds:
            return None
        bs, be = _span(body)
        c = self.ctx
        if not (0 <= bs < be <= len(self.text)) or self.text[bs] != "{" or self.text[be - 1] != "}" \
                or c.macro_hit(bs, bs + 1):
            self.skipped["opaque"][R_BRACE] += 1
            return None
        self.candidates["opaque"] += 1
        return bs

    # ---- 通用判定 ----------------------------------------------------------------

    def _unwrap(self, node: ci.Cursor) -> ci.Cursor:
        """去掉隐式转换等包装（libclang 把它们暴露成 UNEXPOSED_EXPR，范围与被包装的表达式相同）。"""
        for _ in range(32):
            if kind_of(node) != CK.UNEXPOSED_EXPR:
                break
            kids = list(node.get_children())
            if len(kids) != 1 or _span(kids[0]) != _span(node):
                break
            node = kids[0]
        return node

    def _pure_chain(self, node: ci.Cursor, depth: int = 0) -> bool:
        """MEMBER_REF_EXPR 的整条基表达式链只由成员访问和变量引用组成（没有调用 / 自增 / 赋值 / 解引用 / 下标）。"""
        kids = list(node.get_children())
        if len(kids) != 1 or depth > 12:
            return False
        c = self._unwrap(kids[0])
        k = kind_of(c)
        if k == CK.DECL_REF_EXPR:
            return True
        return k == CK.MEMBER_REF_EXPR and self._pure_chain(c, depth + 1)

    def _pure_atomic(self, node: ci.Cursor) -> Optional[ci.Cursor]:
        """操作数是 DECL_REF_EXPR / 字面量 / 纯成员链则返回去掉包装后的节点，否则 None。"""
        u = self._unwrap(node)
        k = kind_of(u)
        if k in _ATOM_KINDS:
            s, e = _span(u)
            i = self.t.start_idx.get(s)
            return u if i is not None and self.t.ends[i] == e else None   # 原子操作数恰好是一个记号
        if k == CK.MEMBER_REF_EXPR and self._pure_chain(u):
            return u
        return None

    @staticmethod
    def _arith(outer: ci.Cursor, inner: ci.Cursor) -> bool:
        return _canon(outer) in _ARITH_TYPES and _canon(inner) in _ARITH_TYPES

    @staticmethod
    def _integer(outer: ci.Cursor, inner: ci.Cursor) -> bool:
        return _canon(outer) in _INT_TYPES and _canon(inner) in _INT_TYPES

    def _is_zero_literal(self, node: ci.Cursor) -> bool:
        u = self._unwrap(node)
        if kind_of(u) != CK.INTEGER_LITERAL:
            return False
        s, e = _span(u)
        i = self.t.start_idx.get(s)
        return i is not None and self.t.ends[i] == e and self.text[s:e] == "0"

    def _needs_paren(self, node: ci.Cursor) -> bool:
        return kind_of(self._unwrap(node)) not in _NO_PAREN_KINDS

    def _record(self, kind: str, build) -> Optional[Edit]:
        """跑 build()；通过则记候选，被拒绝则记原因。kind 没启用时直接不做。"""
        if kind not in self.kinds:
            return None
        try:
            edit = build()
        except _Reject as why:
            self.skipped[kind][str(why)] += 1
            return None
        self.candidates[kind] += 1
        return edit

    # ---- 二元运算 ----------------------------------------------------------------

    def _binary(self, n: ci.Cursor, kids: list[ci.Cursor]) -> list[Edit]:
        if len(kids) != 2:
            return []
        lhs, rhs = kids
        ns, ne = _span(n)
        ls, le = _span(lhs)
        rs, re_ = _span(rhs)
        t = self.t
        if not (ns == ls and ne == re_ and ls < le <= rs < re_ and t.aligned(ls, le) and t.aligned(rs, re_)):
            return []                                  # 范围对不上记号（多半是宏参数 / 宏体里的节点），无从判断
        i_op = t.end_idx[le] + 1
        if t.start_idx[rs] != i_op + 1 or t.kinds[i_op] != "p":
            return []                                  # 两个操作数之间必须恰好一个标点记号
        op = t.spell(i_op)
        if op in _REL_FLIP:
            kind = "relation_flip"
        elif op in _COMMUTE_OPS:
            kind = "commute"
        elif op == "==" and (self._is_zero_literal(rhs) or self._is_zero_literal(lhs)):
            kind = "eq_zero"
        else:
            return []

        def build() -> Edit:
            if not self.ctx.clean(ns, ne):
                raise _Reject(R_MACRO)
            if kind == "eq_zero":
                return self._eq_to_not(lhs, rhs, (ns, ne), (ls, le), (rs, re_))
            la, ra = self._pure_atomic(lhs), self._pure_atomic(rhs)
            if la is None or ra is None:
                raise _Reject(R_NOT_ATOMIC)
            if not (self._arith(lhs, la) and self._arith(rhs, ra)):
                raise _Reject(R_NOT_ARITH)
            if self.text[ls:le] == self.text[rs:re_]:
                raise _Reject(R_SAME)
            new_op = _REL_FLIP.get(op, op)
            gap = self.text[le:rs]
            off = t.starts[i_op] - le
            new_gap = gap[:off] + new_op + gap[off + len(op):]
            return Edit(kind, ns, ne, [(rs, re_), new_gap, (ls, le)])

        edit = self._record(kind, build)
        return [edit] if edit is not None else []

    def _eq_to_not(self, lhs: ci.Cursor, rhs: ci.Cursor, node_span: tuple[int, int],
                   lspan: tuple[int, int], rspan: tuple[int, int]) -> Edit:
        """``x == 0`` / ``0 == x`` → ``!x``。"""
        x, xspan = (lhs, lspan) if self._is_zero_literal(rhs) else (rhs, rspan)
        xu = self._unwrap(x)
        if not self._integer(x, xu):
            raise _Reject(R_NOT_INT)
        parts: list = ["!(", xspan, ")"] if self._needs_paren(xu) else ["!", xspan]
        return Edit("eq_zero", node_span[0], node_span[1], parts)

    # ---- 一元运算 ----------------------------------------------------------------

    def _unary(self, n: ci.Cursor, kids: list[ci.Cursor]) -> list[Edit]:
        if len(kids) != 1:
            return []
        x = kids[0]
        us, ue = _span(n)
        xs, xe = _span(x)
        t = self.t
        if not (us < xs < xe == ue):
            return []
        i_u = t.start_idx.get(us)
        if i_u is None or t.kinds[i_u] != "p" or t.spell(i_u) != "!":
            return []
        if i_u + 1 >= len(t.starts) or t.starts[i_u + 1] != xs or xe not in t.end_idx:
            return []                                  # ! 后面紧跟操作数，操作数范围对齐记号
        xu = self._unwrap(x)

        def clean() -> None:
            if not self.ctx.clean(us, ue):
                raise _Reject(R_MACRO)

        def build_dm() -> Edit:
            clean()
            return self._de_morgan(xu, (us, ue), (xs, xe))

        def build_eq() -> Edit:
            clean()
            if not self._integer(x, xu):
                raise _Reject(R_NOT_INT)
            return Edit("eq_zero", us, ue, ["(", (xs, xe), " == 0)"])

        out: list[Edit] = []
        if self._dm_shape(xu):
            dm = self._record("de_morgan", build_dm)
            if dm is not None:
                out.append(dm)
        eq = self._record("eq_zero", build_eq)
        if eq is not None:
            out.append(eq)
        return out

    def _dm_op_children(self, paren: ci.Cursor):
        """paren 是 ``( A op B )``、op 为 && / || 时返回 (BINARY 节点, A, B)，否则 None（只看 AST，不看记号）。"""
        if kind_of(paren) != CK.PAREN_EXPR:
            return None
        pk = list(paren.get_children())
        if len(pk) != 1 or kind_of(pk[0]) != CK.BINARY_OPERATOR:
            return None
        bk = list(pk[0].get_children())
        if len(bk) != 2:
            return None
        return pk[0], bk[0], bk[1]

    def _dm_shape(self, paren: ci.Cursor) -> bool:
        """``!`` 的操作数是不是「括号里一个 && / ||」——先粗判，决定没通过的情况要不要记入 de_morgan 的统计。"""
        parts = self._dm_op_children(paren)
        if parts is None:
            return False
        _, a, c = parts
        t = self.t
        i = t.end_idx.get(_span(a)[1])
        if i is None or t.start_idx.get(_span(c)[0]) != i + 2:
            return False
        return t.spell(i + 1) in ("&&", "||")

    def _de_morgan(self, paren: ci.Cursor, node_span: tuple[int, int], pspan: tuple[int, int]) -> Edit:
        """``!( A && B )`` / ``!( A || B )`` → ``(!A || !B)`` / ``(!A && !B)``。"""
        b, a, c = self._dm_op_children(paren)
        t = self.t
        ps, pe = pspan
        bs, be = _span(b)
        as_, ae = _span(a)
        cs, ce = _span(c)
        if not (t.aligned(as_, ae) and t.aligned(cs, ce) and as_ == bs and ce == be and ae <= cs):
            raise _Reject(R_SHAPE)
        i_lp, i_rp = t.start_idx.get(ps), t.end_idx.get(pe)
        if i_lp is None or i_rp is None or t.spell(i_lp) != "(" or t.spell(i_rp) != ")":
            raise _Reject(R_SHAPE)
        if t.starts[i_lp + 1] != bs or t.ends[i_rp - 1] != be:
            raise _Reject(R_SHAPE)                     # 括号紧贴着 A 和 B，里面没有别的记号
        i_op = t.end_idx[ae] + 1
        if t.start_idx[cs] != i_op + 1:
            raise _Reject(R_SHAPE)
        op = t.spell(i_op)
        if op not in ("&&", "||"):
            raise _Reject(R_SHAPE)
        new_op = "||" if op == "&&" else "&&"
        gap = self.text[ae:cs]
        off = t.starts[i_op] - ae
        new_gap = gap[:off] + new_op + gap[off + 2:]
        wa, wc = self._needs_paren(a), self._needs_paren(c)
        parts: list = ["(!" + ("(" if wa else ""), (as_, ae),
                       (")" if wa else "") + new_gap + "!" + ("(" if wc else ""), (cs, ce),
                       (")" if wc else "") + ")"]
        return Edit("de_morgan", node_span[0], node_span[1], parts)


# --------------------------------------------------------------------------- 文件级流程


def _opaque_names(seed: int, rel: str, count: int, taken: set[str]) -> list[str]:
    """每个函数体一个 opaque 变量名：走与 L1 同一张改名表的取名规则（``build_rename_map``，种类按变量），
    所以长得和 L1 改过的变量名一样（``v_<n>``），不是一眼能认出来的固定模式。
    新名字避开 taken（项目所有文件里出现过的拼写）并登记进去，由 (seed, 文件) 决定。"""
    holders = [f"_k{i}" for i in range(count)]
    mapping = build_rename_map(holders, f"{seed}:{rel}", kinds={h: "var" for h in holders}, reserved=taken)
    names = [mapping[h] for h in holders]
    taken.update(names)
    return names


class _Transformer:
    def __init__(self, args: Sequence[str], seed: int, ratio: float, opaque_ratio: float,
                 kinds: Sequence[str]) -> None:
        unknown = [k for k in kinds if k not in KINDS]
        if unknown:
            raise ValueError(f"未知的改写种类: {unknown}（可用: {list(KINDS)}）")
        self.parser = Parser(args)
        self.seed = seed
        self.ratio = ratio
        self.opaque_ratio = opaque_ratio
        self.kinds = tuple(kinds)

    def transform_bytes(self, path: Path, data: bytes, rel: str, taken: set[str],
                        hosts: Sequence[Path] = ()) -> tuple[bytes, FileStats]:
        """改写一个文件的内容，返回 (新内容, 统计)。path 只用于定位 include 与诊断，不要求磁盘上存在。"""
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
        opaque = _opaque_names(self.seed, rel, len(ctx.bodies), taken) if "opaque" in self.kinds else []
        chosen: list[Edit] = []
        for fidx, body in enumerate(ctx.bodies):
            rng = random.Random(f"{self.seed}:{rel}:{fidx}")
            for options in finder.find(body):
                if rng.random() < self.ratio:
                    chosen.append(options[0])        # 同一节点的备选按优先级排列（De Morgan 比 eq_zero 更具体）
            pos = finder.opaque_site(body)
            if pos is not None and rng.random() < self.opaque_ratio:
                name = opaque[fidx]
                chosen.append(Edit("opaque", pos, pos + 1, ["{ { volatile int %s = 0; (void)%s; }" % (name, name)]))
        fs.functions = len(ctx.bodies)
        fs.candidates = finder.candidates
        fs.skipped = finder.skipped

        chosen, fs.overlap = check_laminar(chosen)
        good = rollback_invalid(chosen, lambda eds: self.parser.valid(unit, render(data, eds)), fs.rolled_back, rel)
        if not good:
            return data, fs
        new = render(data, good)
        for e in good:
            fs.applied[e.kind] += 1
        return new, fs


# --------------------------------------------------------------------------- 对外接口


def transform_bytes(data: bytes, args: Sequence[str] = (), seed: int = 0, *, ratio: float = DEFAULT_RATIO,
                    opaque_ratio: float = 1.0, kinds: Optional[Sequence[str]] = None, filename: str = "snippet.c",
                    workdir: Optional[Path] = None) -> tuple[bytes, dict]:
    """改写一段 C 源码（bytes），返回 (新源码, 统计)。args 是给 libclang 的额外参数（-I、-D、-std 等）。

    filename 相对 workdir（默认当前目录）解析，只用来定位 ``#include "..."``，磁盘上不必真有这个文件。
    """
    kinds = tuple(kinds) if kinds is not None else KINDS
    tr = _Transformer(["-x", "c", "-w", "-ferror-limit=0", *clangcfg.builtin_include_args(), *args],
                      seed, ratio, opaque_ratio, kinds)
    path = (Path(workdir) if workdir else Path.cwd()).resolve() / filename
    taken = set(_IDENT_RE.findall(data.decode("latin-1")))
    with RecursionGuard():
        new, fs = tr.transform_bytes(path, data, filename, taken)
    stats = empty_stats(KINDS, seed, ratio, kinds, functions=0, opaque_ratio=opaque_ratio)
    merge_stats(stats, filename, fs, new != data)
    return new, stats


def transform_source(source: str, args: Sequence[str] = (), seed: int = 0, *, ratio: float = DEFAULT_RATIO,
                     opaque_ratio: float = 1.0, kinds: Optional[Sequence[str]] = None, filename: str = "snippet.c",
                     workdir: Optional[Path] = None) -> tuple[str, dict]:
    """:func:`transform_bytes` 的字符串版本（按 UTF-8 编解码），供测试和交互使用。"""
    new, stats = transform_bytes(source.encode("utf-8"), args, seed, ratio=ratio, opaque_ratio=opaque_ratio,
                                 kinds=kinds, filename=filename, workdir=workdir)
    return new.decode("utf-8"), stats


def transform_semantic(spec: corpus.CorpusSpec, src_root: Path, seed: int, *, ratio: float = DEFAULT_RATIO,
                       opaque_ratio: float = 1.0, kinds: Optional[Sequence[str]] = None,
                       files: Optional[Iterable[str]] = None, include_vendored: bool = False) -> dict:
    """对 src_root 下 spec.project_files 里的 C 文件做 L3 语义混淆（就地修改），返回统计字典。

    返回值里 ``relation_flip`` / ``de_morgan`` / ``eq_zero`` / ``commute`` / ``opaque`` 是每类改写的应用次数，
    ``total`` 是总数；``candidates`` 是通过全部检查的可用位置数，``skipped`` 是被放弃的位置及原因，
    ``rolled_back`` 是重新解析出错而回滚的次数（按类别），``files`` 是逐文件明细。

    可选参数（接口契约之外的扩展）：``ratio`` 每个可改位置被选中的概率（opaque 除外）；``opaque_ratio`` 每个函数体
    被插入 opaque 语句的概率，默认 1.0（每个函数都插）；``kinds`` 只启用部分类别；``files`` 只处理这些相对路径；
    ``include_vendored`` 为 True 时连 ``spec.vendored`` 里的第三方框架文件一起改（默认不改）。
    """
    src_root = Path(src_root).resolve()
    if not src_root.is_dir():
        raise FileNotFoundError(f"源码目录不存在: {src_root}")
    kinds = tuple(kinds) if kinds is not None else KINDS
    tr = _Transformer(libclang_args(spec, src_root), seed, ratio, opaque_ratio, kinds)
    stats = empty_stats(KINDS, seed, ratio, kinds, functions=0, opaque_ratio=opaque_ratio)

    all_files, todo = plan_files(spec, src_root, files=files, include_vendored=include_vendored)
    hosts = host_map(p for p in all_files if p.suffix == ".c")
    # opaque 变量名要避开项目所有文件里出现过的拼写（含不处理的文件，它们仍参与编译）
    taken: set[str] = set()
    for path in all_files:
        taken.update(_IDENT_RE.findall(path.read_bytes().decode("latin-1")))

    with RecursionGuard():
        run_files(todo, src_root, KINDS, stats,
                  lambda path, data, rel: tr.transform_bytes(path, data, rel, taken, hosts.get(path.name, ())),
                  "语义混淆")
    log.info("语义混淆 %s: 共 %d 处 %s，%d/%d 个文件有改动", spec.name, stats["total"],
             {k: stats[k] for k in KINDS}, stats["files_changed"], stats["files_total"])
    return stats
