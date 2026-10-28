"""缺陷修复任务的变异算子与变异体选择（docs/设计-缺陷修复任务.md §3、§4、§8）。

在某一级混淆副本（L0–L3）的目标函数体上，用 libclang 的 AST 找出八类单点缺陷的可改位置（:func:`find_sites`），
按偏移改写出变异体（:func:`apply_site`），再把变异体当作候选走一遍完整的 ``runner.evaluate``（替换 → 语法检查 →
构建 → 项目测试），只有「编译通过且稳定地让测试失败」的才算被杀死（:func:`choose_mutant`）。

八个算子（编号按附录的优先序，``OPERATORS``）
------------------------------------------
==========  ==============================================================  ====================================
``ROR``     关系算子替换：``<`` ``<=`` ``>`` ``>=`` ``==`` ``!=``           ``<``↔``<=``、``>``↔``>=``、``==``↔``!=``
``NEG``     条件取反：``if`` / ``while`` / ``for`` 的条件                    ``c`` → ``!(c)``
``LOR``     逻辑算子替换                                                    ``&&``↔``||``
``OBO``     差一：比较或下标里的十进制整数字面量 n                          ``n+1`` 或 ``n-1``（rng 决定）
``DGS``     删守卫：无 else、只含一条 return / break 的 if                   整句删除
``AOR``     算术算子替换                                                    ``+``↔``-``、``*``↔``/``
``RVR``     返回值替换：``return expr;``（expr 不是字面量 0 / NULL）         ``return 0;`` 或 ``return NULL;``
``DST``     删语句：一条赋值、复合赋值、自增自减或调用表达式语句            整条删除
==========  ==============================================================  ====================================

位点的识别与边界
----------------
* :class:`Site` 的 ``start`` / ``end`` 是目标文件里的字节偏移（与 ``FunctionTarget`` 同一口径），``original`` 是该区间的原文，
  ``index`` 是它在本算子全部位点里的序号（按源码位置排序，从 0 起），所以 ``(operator, index)`` 唯一确定一个位点；
  ``detail`` 是给人看的一句话，RVR 的 ``detail`` 就是替换用的整句（``return 0;`` / ``return NULL;``），apply 时直接取用。
  区间：ROR / LOR / AOR 是运算符记号本身；OBO 是字面量记号；NEG 是括号里的条件表达式；
  DGS / DST / RVR 是整条语句（含结尾分号）。
* 一律不碰宏：位点落在宏调用的实参里、由宏展开生成（节点范围与某个宏调用重合）、或与宏调用部分重叠的，都不算位点；
  只有「宏调用整个落在位点范围内」是允许的（``x = MAX(a, b);`` 可以整句删掉，``if (ISDIGIT(c))`` 的条件可以整个取反）。
  范围里含预处理指令行的位点也不要（删到一半会让 ``#if`` 不成对）。宏调用区间的合并与冲突判定在
  ``obfuscate._edits.MacroIntervals``，与 L2 结构混淆共用。
* DGS / DST 只认直接位于复合语句里的语句（``else if`` 链里的 if、没有花括号的循环体不认，删掉会让上下文语法不完整）。
* 变异后的整份文件必须能用 libclang 重新解析，且不比变异前多出新的 error（变异前的文件本来就有的 error 不算，
  与 L1 符号混淆的口径一致）；否则该候选作废，不进沙箱、也不占 ``max_tries``，只计入 ``parse_fail_count``。
* 缩进与换行：DGS / DST 删掉的语句如果独占若干整行，连同缩进和行尾换行一起去掉，不留空行（空行会暴露缺陷位置）；
  同一行还有别的内容时只删语句和紧挨着的一段空白；
  其余算子都在行内改写，行数不变（用 ``_edits.render`` 做偏移改写）。

选择与 kill 检查（附录 §4）
---------------------------
**试探顺序。** ``order_sites``：全部位点先按（算子优先序, 序号）排成规范顺序，再用 ``random.Random(f"{seed}|{target_id}")`` 洗牌，
种子里**没有级别**。L1 相对 L0 只改标识符，位点表与序号一一对应；种子里带上级别的话，同一个目标在 L0 与 L1 会走两条不同的试探路线，
选出不是同一个缺陷的变异体（评审实测只有约一半的目标落在同一个位点），「混淆本身是否影响修复」就被缺陷难度的差异混杂了。
优先序只用来固定洗牌前的规范顺序，不把算子分档（先试完前一档再试后一档，会让多数目标落在 ROR 上：mini 夹具 L0 的 27 个目标里
分档是 18 个 ROR、不分档只有 2 个，附录 §6「按算子分层的通过率」就没有可比的格子了）。

**同一个位点。** L0 选定位点之后，L1–L3 先试「同一个位点」（:class:`SiteRef`），再按 ``prefer_operator`` 复用 L0 的算子：
:func:`match_reference` 把 L0 的函数文本与本级的函数文本按归一化记号序列（标识符一律换成 ``ID``）对齐（``difflib``），
L0 位点的记号在本级对得上、前后各一个记号也对得上、并且本级在那个位置恰好有同一算子的位点，就是同一个位点。L1 只改名字，
记号序列完全相同，对齐是精确的；L2 / L3 改写了函数体的结构与表达式，位点的记号被改掉的（``x == 0`` 换成 ``!x``、``a < b`` 换成
``b > a`` 等）对不上，就退回到算子级复用（各级对得上的比例见 docs/实验记录.md 里「与 L0 同位点的比例」，由 experiment_record 从数据文件算出）。选中的变异体记
``same_site_as_L0``，OBO 的 ±1 也按 L0 位点的序号决定，同一个位点在各级改成同一个方向。

``search_mutant`` 依次试，每个位点：``apply_site`` → 重新解析 → ``evaluate``（沙箱）。
``stage == "test"`` 且归因类别属于 ``test_fail`` / ``runtime_crash`` / ``runtime_timeout`` 才算杀死；
``stage == "pass"`` 是等价变异体（``equivalent_count``）；编译 / 构建失败不算杀死（``compile_fail_count``）。
**复核**：判为杀死的候选再评测一次（``recheck``），第二次仍然失败才出题，记下 ``kill_recheck``；第二次通过了（崩溃 / 超时 / OOM 的
结局会随负载变，删赋值留下的未初始化变量也是）计入 ``unstable_count``，不出题、继续试下一个位点。
最多试 ``max_tries`` 个位点（只数进了沙箱的），用完仍没杀死就放弃。``choose_mutant`` 是附录规定的接口，返回 ``Mutant | None``；
``search_mutant`` 额外给出放弃时的完整记账（试了几个、等价几个、编译失败几个、逐位点的结果），供 mutate 阶段统计放弃数。
``choose_for_levels`` 固化「L1–L3 先试 L0 的位点、再优先复用 L0 的算子」，并记下 ``same_operator_as_L0`` / ``same_site_as_L0``。

**去注释。** ``strip_comments``（默认开）：函数里有注释的，先把**去注释后的原函数**放回副本过一遍项目测试，不通过（``strip_unsafe``）
就放弃这个（目标, 级别），不参照、不搜索；通过了再对去注释后的变异体做 kill 检查，``Mutant.code`` 就是最终放进 prompt 的那段无注释文本。
L1 以上注释里仍是原名，放进修复 prompt 会泄漏被隐藏的名字。

用法（mutate 阶段）::

    trees = {lv: (root_lv, extract.locate(spec, root_lv, mapping_lv.get(t.name, t.name), t.file)) for lv in LEVELS}
    out = choose_for_levels(spec, trees, seed, sandbox, target_id=t.target_id, work_root=ws.work)
    for lv, search in out.items():      # search.mutant 为 None 就是这个 (目标, 级别) 不出题，search.to_dict() 记下放弃的原因与记账

``Mutant.context_orphans``：这个变异体让依赖上下文里哪些原型 / 宏 / 全局变得多余（只有被改掉的代码才用得到，``prompt.prune_context``）；
出题时它们从提示词里裁掉，免得把答案线索留在里面。

确定性：位点枚举只依赖源码；洗牌的 rng 由 (seed, target_id) 派生，每个位点 apply 用的 rng 由 (seed, target_id, 算子, 序号) 派生
（OBO 的 ±1 不随试的先后变；同一个位点在各级用 L0 的序号）；同一输入重复运行得到同一个变异体。
libclang 的解析与遍历放在一把锁里（并发调用 ``choose_mutant`` 时，只有沙箱运行是并行的）。
"""
from __future__ import annotations

import bisect
import difflib
import logging
import os
import random
import re
import threading
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import clang.cindex as ci

from bench.attribute import attribute
from bench.clangcfg import kind_of
from bench.corpus import CorpusSpec
from bench.extract import FunctionTarget, parse_file, strip_c_comments
from bench.obfuscate._edits import Edit, MacroIntervals, Parser, Unit, libclang_args, match_close, render
from bench.postprocess import Candidate
from bench.prompt import prune_context
from bench.runner import evaluate, read_function
from bench.similarity import ID_TOKEN
from bench.vocab import KILL_MODES, LEVELS, MUTATION_OPERATORS

log = logging.getLogger(__name__)

CK = ci.CursorKind
TK = ci.TypeKind

__all__ = [
    "OPERATORS", "DEFAULT_MAX_TRIES", "CHECK_MODEL", "MutateError", "Site", "SiteRef", "Analysis", "Mutant", "MutantSearch",
    "analyze", "find_sites", "count_sites", "reference_for", "match_reference", "mutation_text", "apply_site", "order_sites",
    "strip_function_comments", "search_mutant", "choose_mutant", "choose_for_levels",
]

#: 算子编号，按附录 §4 的优先序（洗牌前的规范顺序，也是报告里各列的顺序；定义在 bench.vocab，报告共用）
OPERATORS: tuple[str, ...] = MUTATION_OPERATORS

#: kill 检查最多试的位点数（附录 §4）
DEFAULT_MAX_TRIES = 12

#: kill 检查走 evaluate 时用的模型名（只出现在 EvalResult 里，不进任何数据文件）
CHECK_MODEL = "mutation-check"


class MutateError(RuntimeError):
    """目标与副本对不上（找不到函数定义、函数体偏移不符等），或内部不变量被破坏（位点区间与源码对不上等）。

    都是内部错误，不是某个变异体的问题，也不是命令行用法错误（cli 把 ValueError 当用法错误，不写堆栈，这里不能用它）。"""


# ---------------------------------------------------------------------------
# 数据类
# ---------------------------------------------------------------------------
@dataclass
class Site:
    """一个可改位置。start / end 是目标文件里的字节偏移，original 是 [start, end) 的原文。"""

    operator: str
    start: int
    end: int
    original: str
    index: int              # 在本算子的全部位点里的序号（按源码位置，从 0 起）
    detail: str = ""        # 给人看的一句话；RVR 的 detail 就是替换用的整句


@dataclass
class Analysis:
    """一级副本里目标函数的位点表，加上归一化记号表（跨级别对齐位点要用）。

    ``tokens`` 是函数范围内的记号序列（不含注释）：标识符一律换成 ``ID``，其余原样；``spans[k]`` 是第 k 个记号在文件里的字节区间。"""

    sites: list[Site]
    tokens: list[str]
    spans: list[tuple[int, int]]

    def token_range(self, start: int, end: int) -> tuple[int, int] | None:
        """完全落在文件偏移 [start, end) 里的记号下标区间 [i, j)；一个记号都没有返回 None。"""
        i = bisect.bisect_left(self.spans, (start, -1))
        j = i
        while j < len(self.spans) and self.spans[j][1] <= end:
            j += 1
        return (i, j) if j > i else None


@dataclass(frozen=True)
class SiteRef:
    """L0 选定位点的参照：L0 目标函数的归一化记号序列，以及位点占的记号下标区间 [first, last)。
    其它级别用它找「同一个位点」（:func:`match_reference`）。"""

    operator: str
    index: int                  # 在 L0 的算子内序号
    tokens: tuple[str, ...]
    first: int
    last: int


@dataclass
class Mutant:
    """选定的变异体。字段与 ``results/mutants-<corpus>-<level>.jsonl`` 一一对应（``code`` 在文件里叫 ``mutant_code``）。"""

    target_id: str
    level: str
    operator: str
    site_index: int
    original_text: str          # 位点区间的原文
    mutated_text: str           # 换成了什么（删除类为空串）
    code: str                   # 变异后的目标函数完整定义（该级副本里 [start, end) 的新文本；strip_comments=True 时已去注释）
    kill_mode: str | None       # test_fail / runtime_crash / runtime_timeout（第一次评测的归因）
    tried_sites: int            # 进沙箱试过的位点数（含被杀死的这一个）
    equivalent_count: int       # 试过的位点里编译通过、测试仍全过的（等价变异体）
    compile_fail_count: int     # 试过的位点里语法检查 / 构建没过的
    # 以下是附录字段之外的附加信息，都带默认值
    site_detail: str = ""
    kill_detail: str = ""       # attribute 给出的杀死细节（退出码、信号等，英文短句）
    parse_fail_count: int = 0   # 因 libclang 重新解析有新 error 而没进沙箱的位点数
    unstable_count: int = 0     # 第一次判为杀死、复核时却没失败的候选个数（不出题，也不算等价变异体）
    kill_recheck: str | None = None     # 复核（同一份候选再评测一次）的归因类别；没复核为 None
    same_operator_as_L0: bool | None = None     # choose_for_levels 填：与 L0 选定的算子是否相同；L0 自己为 True，L0 没选出则为 None
    same_site_as_L0: bool | None = None         # 同上，是不是 L0 选定的那个位点（L1 精确，L2 / L3 靠记号对齐，见模块文档）
    context_orphans: list[str] = field(default_factory=list)    # 依赖上下文里因这个缺陷而多余的名字（prompt.prune_context）

    def to_dict(self) -> dict[str, Any]:
        """文件里的记录形态（``code`` 写作 ``mutant_code``）。"""
        d = asdict(self)
        d["mutant_code"] = d.pop("code")
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Mutant":
        """to_dict 的逆操作；多余的键忽略，``mutant_code`` 与 ``code`` 都认。"""
        fields = {k: v for k, v in d.items() if k in cls.__dataclass_fields__ and k != "code"}
        fields["code"] = d["mutant_code"] if "mutant_code" in d else d["code"]
        return cls(**fields)


@dataclass
class MutantSearch:
    """一次（目标, 级别）的选择过程。``mutant`` 为 None 就是放弃，``outcome`` 说明原因：

    ``killed`` 选出了被杀死的变异体；``no_sites`` 没有任何可用位点（没有位点，或全部重新解析失败）；
    ``exhausted`` 所有可用位点都试过了，没有一个被杀死；``gave_up`` 试满 ``max_tries`` 个仍没有杀死、还有位点没试；
    ``strip_unsafe`` 函数里有注释，去注释后的原函数通不过项目测试，不搜索。

    ``anchor``：L0 位点的参照在本级的情况——``none`` 没有参照（L0 自己，或 L0 没选出变异体）、``no_image`` 有参照但本级找不到
    对应的位点、``tried`` 试了对应的位点但它不是（稳定的）杀死、``chosen`` 选中的就是对应的位点。
    """

    target_id: str
    level: str
    mutant: Mutant | None
    outcome: str
    n_sites: int                    # find_sites 给出的位点总数
    tried_sites: int = 0
    equivalent_count: int = 0
    compile_fail_count: int = 0
    parse_fail_count: int = 0
    unstable_count: int = 0
    prefer_operator: str | None = None
    anchor: str = "none"
    has_comments: bool = False      # 原函数里有注释
    strip_checked: bool = False     # 去注释后的原函数放回副本跑过项目测试
    rechecked: int = 0              # 做了复核的候选个数
    tries: list[dict[str, Any]] = field(default_factory=list)   # 逐个试过的位点：operator / site_index / stage / category
    # 按归因类别累计的沙箱运行耗时（含复核与去注释验证）；每次运行都不同，不参与相等比较
    sandbox_seconds: dict[str, float] = field(default_factory=dict, compare=False)
    analysis: Analysis | None = field(default=None, repr=False, compare=False)     # 本级的位点表与记号表，不进记录

    @property
    def killed(self) -> bool:
        return self.mutant is not None

    def to_dict(self) -> dict[str, Any]:
        """选择过程的记录（放弃的 (目标, 级别) 也留得下试了几个、等价几个）：``mutant`` 展开成 :meth:`Mutant.to_dict` 的形态。"""
        d = asdict(replace(self, mutant=None, analysis=None))
        d.pop("analysis")
        d["mutant"] = None if self.mutant is None else self.mutant.to_dict()
        return d


# ---------------------------------------------------------------------------
# 改写表
# ---------------------------------------------------------------------------
_ROR_MAP = {"<": "<=", "<=": "<", ">": ">=", ">=": ">", "==": "!=", "!=": "=="}
_LOR_MAP = {"&&": "||", "||": "&&"}
_AOR_MAP = {"+": "-", "-": "+", "*": "/", "/": "*"}

_DEC_RE = re.compile(r"^(0|[1-9][0-9]*)([uUlL]{0,3})$")                 # 十进制整数字面量（带可选后缀）
_ZERO_RE = re.compile(r"^(?:0[uUlL]{0,3}|0[xX]0+[uUlL]{0,3}|0\.0*[fFlL]?|0[fF]|NULL|\(void\*\)0)$")      # return 0 / NULL 一类
_DURATION_RE = re.compile(r"after \d+(?:\.\d+)?s")
_DIRECTIVE_RE = re.compile(rb"^[ \t]*#", re.M)

_LINE_OPS = frozenset({"DGS", "DST"})           # 整句删除：独占整行时连缩进与换行一起去掉
_MAX_LITERAL_DEPTH = 16

_SCALAR_KINDS = frozenset(getattr(TK, n) for n in (
    "BOOL", "CHAR_U", "UCHAR", "CHAR16", "CHAR32", "USHORT", "UINT", "ULONG", "ULONGLONG", "UINT128", "CHAR_S",
    "SCHAR", "WCHAR", "SHORT", "INT", "LONG", "LONGLONG", "INT128", "FLOAT", "DOUBLE", "LONGDOUBLE", "ENUM")
    if hasattr(TK, n))

# libclang 的解析与遍历放在一把锁里：Index / TranslationUnit 不保证可以被多个线程同时使用，沙箱运行不受影响
_CLANG_LOCK = threading.RLock()


# ---------------------------------------------------------------------------
# 目标函数的记号表与宏调用区间
# ---------------------------------------------------------------------------
class _FuncCtx:
    """目标函数所在文件的一次解析：源码、函数范围内的记号表（不含注释）、文件里最外层的宏调用区间。"""

    def __init__(self, tu: ci.TranslationUnit, func: ci.Cursor, data: bytes, file_real: str) -> None:
        self.data = data
        ext = func.extent
        self.lo, self.hi = ext.start.offset, ext.end.offset
        self.S: list[int] = []      # 记号起点
        self.E: list[int] = []      # 记号终点
        self.T: list[str] = []      # 记号文本
        self.ID: list[bool] = []    # 是不是标识符（不含关键字）
        for tok in tu.get_tokens(extent=ext):
            if tok.kind == ci.TokenKind.COMMENT:
                continue
            te = tok.extent
            self.S.append(te.start.offset)
            self.E.append(te.end.offset)
            self.T.append(tok.spelling)
            self.ID.append(tok.kind == ci.TokenKind.IDENTIFIER)
        self.tok_at = {s: i for i, s in enumerate(self.S)}
        self.has_null = False
        self.macros = MacroIntervals(self._scan_macros(tu, file_real))

    def _scan_macros(self, tu: ci.TranslationUnit, file_real: str) -> list[tuple[int, int]]:
        """NULL 有没有定义（RVR 用得上）；返回与函数范围有交集的宏调用区间（还没合并，交给 ``MacroIntervals``）。"""
        real_cache: dict[str, str] = {}
        raw: list[tuple[int, int]] = []
        for c in tu.cursor.get_children():
            k = kind_of(c)
            if k == CK.MACRO_DEFINITION:
                if not self.has_null and c.spelling == "NULL":
                    self.has_null = True
            elif k == CK.MACRO_INSTANTIATION:
                f = c.location.file
                if f is None:
                    continue
                real = real_cache.get(f.name)
                if real is None:
                    real = real_cache[f.name] = os.path.realpath(f.name)
                if real != file_real:
                    continue
                ex = c.extent
                s, e = ex.start.offset, ex.end.offset
                if e > self.lo and s < self.hi:
                    raw.append((s, e))
        return raw

    # -- 记号查询 ------------------------------------------------------
    def first_at_or_after(self, off: int) -> int:
        return bisect.bisect_left(self.S, off)

    def match_close(self, i: int) -> int | None:
        """T[i] 是 ( [ { 之一，返回配对的右括号下标；配不上返回 None。"""
        return match_close(self.T, i)

    def semicolon_end(self, off: int) -> int | None:
        """off 之后的第一个记号如果是分号，返回它的终点；否则 None（语句的结尾不在预期位置）。"""
        i = self.first_at_or_after(off)
        if i < len(self.T) and self.T[i] == ";":
            return self.E[i]
        return None

    # -- 宏与预处理指令 ------------------------------------------------
    def macro_conflict(self, s: int, e: int, *, allow_equal: bool = False) -> bool:
        """[s, e) 与某个最外层宏调用冲突（判定见 ``MacroIntervals.conflicts``）。"""
        return self.macros.conflicts(s, e, allow_equal=allow_equal)

    def has_directive(self, s: int, e: int) -> bool:
        return _DIRECTIVE_RE.search(self.data, s, e) is not None


# ---------------------------------------------------------------------------
# 找位点
# ---------------------------------------------------------------------------
def _span(c: ci.Cursor) -> tuple[int, int]:
    ex = c.extent
    return ex.start.offset, ex.end.offset


def _strip_parens(text: str) -> str:
    """去掉整体套着的最外层括号：``((void*)0)`` -> ``(void*)0``；``(a)+(b)`` 不动。"""
    while text.startswith("(") and text.endswith(")"):
        depth = 0
        for i, ch in enumerate(text):
            depth += (ch == "(") - (ch == ")")
            if depth == 0 and i < len(text) - 1:
                return text
        text = text[1:-1]
    return text


def _is_pointer(node: ci.Cursor) -> bool:
    """表达式的类型是不是指针（数组经隐式转换后也是）；取不到类型当作不是。"""
    try:
        return node.type.get_canonical().kind == TK.POINTER
    except Exception as exc:    # noqa: BLE001 - 个别节点取不到类型：AOR 照常处理，不合法的改写交给重新解析去挡
        log.debug("取 @%d 节点的类型失败（%s: %s），当作不是指针", node.extent.start.offset, type(exc).__name__, exc)
        return False


def _aor_valid(text: str, kids: list[ci.Cursor]) -> bool:
    """AOR 改写后类型上是否可能合法：指针相减（ptrdiff_t）改成相加、``整数 + 指针`` 改成 ``整数 - 指针``、
    指针参与乘除都不合法，这类位点一开始就不算（重新解析反正会把它们挡掉，但位点数会虚高）。"""
    left, right = _is_pointer(kids[0]), _is_pointer(kids[1])
    if text == "-":
        return not (left and right)
    if text == "+":
        return not (right and not left)
    return not (left or right)


def _return_kind(rtype: ci.Type) -> str:
    """返回类型：void / scalar（算术、枚举、_Bool）/ pointer / other（struct、union、复数等，RVR 不处理）。"""
    kind = rtype.get_canonical().kind
    if kind == TK.VOID:
        return "void"
    if kind == TK.POINTER:
        return "pointer"
    if kind in _SCALAR_KINDS:
        return "scalar"
    return "other"


class _Finder:
    """遍历函数体，按算子收集位点。每个算子一个字典：(start, end) -> detail。"""

    def __init__(self, ctx: _FuncCtx, ret_kind: str, where: str = "") -> None:
        self.c = ctx
        self.ret = ret_kind
        self.where = where              # 目标的 id 与文件，出错时写进日志（并发处理多个目标时才定位得到）
        self.found: dict[str, dict[tuple[int, int], str]] = {op: {} for op in OPERATORS}

    def add(self, op: str, s: int, e: int, detail: str) -> None:
        self.found[op].setdefault((s, e), detail)

    # -- 遍历 ----------------------------------------------------------
    def run(self, body: ci.Cursor) -> None:
        stack: list[tuple[ci.Cursor, ci.CursorKind | None]] = [(body, None)]
        while stack:
            cur, parent = stack.pop()
            k = kind_of(cur)
            kids = list(cur.get_children())
            try:
                self.visit(cur, k, kids, parent)
            except Exception as exc:    # noqa: BLE001 - 单个节点分析出意外不该拖垮整个函数：记下来，放弃这个节点
                log.warning("%s: 分析 @%d 的 %s 节点时出错，放弃该节点: %r", self.where, cur.extent.start.offset, k, exc)
            for ch in reversed(kids):
                stack.append((ch, k))

    def visit(self, cur: ci.Cursor, k: ci.CursorKind | None, kids: list[ci.Cursor], parent: ci.CursorKind | None) -> None:
        if k == CK.BINARY_OPERATOR:
            self.binary(cur, kids, parent)
        elif k == CK.COMPOUND_ASSIGNMENT_OPERATOR:
            if parent == CK.COMPOUND_STMT:
                self.delete_statement(cur, "复合赋值语句")
        elif k == CK.UNARY_OPERATOR:
            if parent == CK.COMPOUND_STMT and self.incdec(cur, kids):
                self.delete_statement(cur, "自增 / 自减语句")
        elif k == CK.CALL_EXPR:
            if parent == CK.COMPOUND_STMT:
                self.delete_statement(cur, "调用语句")
        elif k == CK.ARRAY_SUBSCRIPT_EXPR:
            if len(kids) == 2:
                self.offset_literals(kids[1])
        elif k == CK.IF_STMT:
            self.negate(cur, kids, "if")
            self.delete_guard(cur, kids, parent)
        elif k == CK.WHILE_STMT:
            self.negate(cur, kids, "while")
        elif k == CK.FOR_STMT:
            self.negate(cur, kids, "for")
        elif k == CK.RETURN_STMT:
            self.replace_return(cur, kids)

    # -- 记号级小工具 --------------------------------------------------
    def binop(self, cur: ci.Cursor, kids: list[ci.Cursor]) -> tuple[str, int, int] | None:
        """二元运算符的 (运算符文本, 起点, 终点)：左右操作数之间必须恰好一个记号。"""
        if len(kids) != 2:
            return None
        (ls, le), (rs, re_) = _span(kids[0]), _span(kids[1])
        if not (ls < le <= rs < re_):
            return None
        c = self.c
        i = c.first_at_or_after(le)
        if i >= len(c.S) or c.E[i] > rs or (i + 1 < len(c.S) and c.S[i + 1] < rs):
            return None
        return c.T[i], c.S[i], c.E[i]

    def incdec(self, cur: ci.Cursor, kids: list[ci.Cursor]) -> bool:
        """UNARY_OPERATOR 是不是自增 / 自减（前缀或后缀）：运算符记号在唯一的操作数之外。"""
        if len(kids) != 1:
            return False
        c = self.c
        (ns, ne), (cs, ce) = _span(cur), _span(kids[0])
        i = c.tok_at.get(ns)
        if i is not None and c.T[i] in ("++", "--") and c.E[i] <= cs:
            return True
        j = c.first_at_or_after(ce)
        return j < len(c.T) and c.T[j] in ("++", "--") and c.E[j] == ne

    # -- ROR / LOR / AOR / OBO ----------------------------------------
    def binary(self, cur: ci.Cursor, kids: list[ci.Cursor], parent: ci.CursorKind | None) -> None:
        op = self.binop(cur, kids)
        if op is None:
            return
        text, s, e = op
        c = self.c
        if text in _ROR_MAP or text in _LOR_MAP or text in _AOR_MAP:
            if not (c.macro_conflict(s, e) or c.has_directive(s, e)):
                if text in _ROR_MAP:
                    self.add("ROR", s, e, f"{text} -> {_ROR_MAP[text]}")
                elif text in _LOR_MAP:
                    self.add("LOR", s, e, f"{text} -> {_LOR_MAP[text]}")
                elif _aor_valid(text, kids):
                    self.add("AOR", s, e, f"{text} -> {_AOR_MAP[text]}")
        if text in _ROR_MAP:
            for operand in kids:
                self.offset_literals(operand)
        elif text == "=" and parent == CK.COMPOUND_STMT:
            self.delete_statement(cur, "赋值语句")

    def offset_literals(self, expr: ci.Cursor) -> None:
        """expr 是比较的操作数或下标：沿隐式转换、括号、强制转换、加减法往下找十进制整数字面量，每个是一个 OBO 位点。"""
        stack: list[tuple[ci.Cursor, int]] = [(expr, 0)]
        while stack:
            node, depth = stack.pop()
            if depth > _MAX_LITERAL_DEPTH:
                continue
            k = kind_of(node)
            kids = list(node.get_children())
            if k == CK.INTEGER_LITERAL:
                self.add_literal(node)
            elif k in (CK.UNEXPOSED_EXPR, CK.PAREN_EXPR) and len(kids) == 1:
                stack.append((kids[0], depth + 1))
            elif k == CK.CSTYLE_CAST_EXPR and kids:
                stack.append((kids[-1], depth + 1))
            elif k == CK.BINARY_OPERATOR:
                op = self.binop(node, kids)
                if op is not None and op[0] in ("+", "-"):
                    stack.extend((ch, depth + 1) for ch in kids)

    def add_literal(self, lit: ci.Cursor) -> None:
        s, e = _span(lit)
        c = self.c
        i = c.tok_at.get(s)
        if i is None or c.E[i] != e or not _DEC_RE.match(c.T[i]) or c.macro_conflict(s, e):
            return
        self.add("OBO", s, e, f"n={int(_DEC_RE.match(c.T[i]).group(1))}: n+1 或 n-1")

    # -- NEG -----------------------------------------------------------
    def negate(self, cur: ci.Cursor, kids: list[ci.Cursor], kw: str) -> None:
        c = self.c
        i = c.tok_at.get(_span(cur)[0])
        if i is None or c.T[i] != kw or i + 1 >= len(c.T) or c.T[i + 1] != "(":
            return
        close = c.match_close(i + 1)
        if close is None or close <= i + 2:
            return
        if kw == "for":
            semis = []
            depth = 0
            for j in range(i + 2, close):
                t = c.T[j]
                if t in ("(", "[", "{"):
                    depth += 1
                elif t in (")", "]", "}"):
                    depth -= 1
                elif t == ";" and depth == 0:
                    semis.append(j)
            if len(semis) != 2 or semis[1] - semis[0] < 2:
                return                      # 不是标准的三段式，或者条件为空（for (;;)）
            first, last = semis[0] + 1, semis[1] - 1
        else:
            first, last = i + 2, close - 1
            if not kids or _span(kids[0]) != (c.S[first], c.E[last]):
                return                      # AST 里的条件与括号内的记号对不上（宏生成等）：不动
        cs, ce = c.S[first], c.E[last]
        if c.macro_conflict(cs, ce, allow_equal=True) or c.has_directive(cs, ce):
            return
        self.add("NEG", cs, ce, f"{kw} 条件 c -> !(c)")

    # -- DGS -----------------------------------------------------------
    def delete_guard(self, cur: ci.Cursor, kids: list[ci.Cursor], parent: ci.CursorKind | None) -> None:
        """形如 ``if (c) return …;`` / ``if (c) { return …; }`` / ``if (c) break;`` 的单语句 if（无 else）。"""
        if parent != CK.COMPOUND_STMT or len(kids) != 2:
            return
        c = self.c
        then = kids[1]
        tk = kind_of(then)
        if tk == CK.COMPOUND_STMT:
            inner = list(then.get_children())
            if len(inner) != 1 or kind_of(inner[0]) not in (CK.RETURN_STMT, CK.BREAK_STMT):
                return
            what = "return" if kind_of(inner[0]) == CK.RETURN_STMT else "break"
            s, e = _span(cur)
            end = e if c.data[e - 1:e] == b"}" else None
        elif tk in (CK.RETURN_STMT, CK.BREAK_STMT):
            what = "return" if tk == CK.RETURN_STMT else "break"
            s, e = _span(cur)
            end = c.semicolon_end(e)
        else:
            return
        i = c.tok_at.get(s)
        if end is None or i is None or c.T[i] != "if" or e <= s:
            return
        if c.macro_conflict(s, e) or c.has_directive(s, end):
            return
        self.add("DGS", s, end, f"删除守卫 if (…) {what}")

    # -- DST -----------------------------------------------------------
    def delete_statement(self, cur: ci.Cursor, what: str) -> None:
        """cur 是复合语句里的一条表达式语句（赋值 / 复合赋值 / 自增自减 / 调用）。"""
        c = self.c
        s, e = _span(cur)
        if e <= s or s not in c.tok_at:
            return
        end = c.semicolon_end(e)
        if end is None or c.macro_conflict(s, e) or c.has_directive(s, end):
            return
        self.add("DST", s, end, f"删除{what}")

    # -- RVR -----------------------------------------------------------
    def replace_return(self, cur: ci.Cursor, kids: list[ci.Cursor]) -> None:
        if len(kids) != 1 or self.ret not in ("scalar", "pointer"):
            return
        c = self.c
        s, e = _span(cur)
        i = c.tok_at.get(s)
        if i is None or c.T[i] != "return" or e <= s:
            return
        end = c.semicolon_end(e)
        if end is None or c.macro_conflict(s, e) or c.has_directive(s, end):
            return
        expr = _strip_parens(re.sub(r"\s+", "", c.data[c.E[i]:e].decode("utf-8", errors="replace")))
        if _ZERO_RE.match(expr):
            return                          # 本来就返回 0 / NULL
        new = "return NULL;" if self.ret == "pointer" and c.has_null else "return 0;"
        self.add("RVR", s, end, new)


def _find_function(tu: ci.TranslationUnit, path: Path, target: FunctionTarget) -> ci.Cursor | None:
    """TU 顶层里与 target 偏移吻合的函数定义。"""
    want = os.path.realpath(path)
    fallback = None
    for cur in tu.cursor.get_children():
        if kind_of(cur) != CK.FUNCTION_DECL or cur.spelling != target.name or not cur.is_definition():
            continue
        f = cur.extent.start.file
        if f is None or os.path.realpath(f.name) != want:
            continue
        s, e = _span(cur)
        if (s, e) == (target.start, target.end):
            return cur
        if s <= target.body_start < e:
            fallback = cur
    return fallback


def _body_of(func: ci.Cursor) -> ci.Cursor | None:
    return next((ch for ch in func.get_children() if kind_of(ch) == CK.COMPOUND_STMT), None)


def _collect(spec: CorpusSpec, root: Path, target: FunctionTarget, data: bytes
             ) -> tuple[_FuncCtx, dict[str, dict[tuple[int, int], str]]]:
    path = root / target.file
    with _CLANG_LOCK:
        tu = parse_file(spec, root, target.file)
        func = _find_function(tu, path, target)
        if func is None:
            raise MutateError(f"{target.target_id}: 在 {target.file} 里找不到偏移 [{target.start}, {target.end}) 的函数定义 "
                              f"{target.name}（目标不是这份副本里 locate 出来的？）")
        body = _body_of(func)
        if body is None or _span(body) != (target.body_start, target.body_end):
            raise MutateError(f"{target.target_id}: 函数体范围 {None if body is None else _span(body)} 与目标记录的 "
                              f"[{target.body_start}, {target.body_end}) 不一致")
        ctx = _FuncCtx(tu, func, data, os.path.realpath(path))
        finder = _Finder(ctx, _return_kind(func.result_type), f"{target.target_id} ({target.file})")
        finder.run(body)
        return ctx, finder.found


def analyze(spec: CorpusSpec, obf_root: Path, target: FunctionTarget) -> Analysis:
    """目标函数的位点表（按（算子优先序, 序号）排序）加上归一化记号表。

    ``target`` 的偏移必须对应 ``obf_root`` 这份副本（L1 以上要先用 ``extract.locate`` 按映射后的名字重新定位）。
    """
    root = Path(obf_root).resolve()
    read_function(root, target)                         # 偏移与源码对不上时抛 SpliceError
    data = (root / target.file).read_bytes()
    ctx, found = _collect(spec, root, target, data)
    sites: list[Site] = []
    for op in OPERATORS:
        for idx, ((s, e), detail) in enumerate(sorted(found[op].items())):
            if not (target.body_start < s < e < target.body_end):
                continue                                # 位点必须严格落在函数体花括号之内
            sites.append(Site(op, s, e, data[s:e].decode("utf-8", errors="replace"), idx, detail))
    tokens = [ID_TOKEN if is_id else text for is_id, text in zip(ctx.ID, ctx.T)]
    return Analysis(sites=sites, tokens=tokens, spans=list(zip(ctx.S, ctx.E)))


def find_sites(spec: CorpusSpec, obf_root: Path, target: FunctionTarget, operator: str | None = None) -> list[Site]:
    """目标函数体里全部可改位置，按（算子优先序, 序号）排序；``operator`` 给了就只返回该算子的位点（偏移的要求同 :func:`analyze`）。"""
    if operator is not None and operator not in OPERATORS:
        raise ValueError(f"未知算子 {operator!r}，可选 {OPERATORS}")
    return [s for s in analyze(spec, obf_root, target).sites if operator is None or s.operator == operator]


# ---------------------------------------------------------------------------
# 跨级别的「同一个位点」
# ---------------------------------------------------------------------------
def reference_for(analysis: Analysis, operator: str, index: int) -> SiteRef | None:
    """``analysis``（通常是 L0 的）里（算子, 序号）对应的位点做成参照；找不到这个位点返回 None。"""
    site = next((s for s in analysis.sites if s.operator == operator and s.index == index), None)
    rng = analysis.token_range(site.start, site.end) if site is not None else None
    if site is None or rng is None:
        return None
    return SiteRef(operator=operator, index=index, tokens=tuple(analysis.tokens), first=rng[0], last=rng[1])


def match_reference(ref: SiteRef, analysis: Analysis) -> Site | None:
    """参照位点在另一级副本里的对应位点；对不上返回 None。

    把参照的函数文本与本级的函数文本按归一化记号序列对齐（``difflib``，标识符一律是 ``ID``，所以符号混淆造成的改名不影响对齐）：
    参照位点占的每个记号都要在本级找到对应的记号、对应的记号连续排列、紧挨着位点前后的两个记号也对得上（单个运算符的位点靠上下文
    才能确认它不是别处恰好长得一样的记号），并且本级在那段记号上恰好有一个同一算子的位点。L1 只改名字，记号序列完全相同，
    对齐是精确的；L2 / L3 改写了结构与表达式，位点的记号被改掉的就对不上（返回 None）。"""
    image: dict[int, int] = {}                      # 参照里的记号下标 -> 本级的记号下标
    matcher = difflib.SequenceMatcher(None, ref.tokens, analysis.tokens, autojunk=False)
    for a, b, n in matcher.get_matching_blocks():
        image.update((a + k, b + k) for k in range(n))
    inside = [image.get(i) for i in range(ref.first, ref.last)]
    if not inside or None in inside:
        return None
    first, last = inside[0], inside[-1]
    if inside != list(range(first, last + 1)):
        return None
    if ref.first > 0 and image.get(ref.first - 1) != first - 1:
        return None
    if ref.last < len(ref.tokens) and image.get(ref.last) != last + 1:
        return None
    start, end = analysis.spans[first][0], analysis.spans[last][1]
    return next((s for s in analysis.sites if s.operator == ref.operator and (s.start, s.end) == (start, end)), None)


def count_sites(sites: Iterable[Site]) -> dict[str, int]:
    """各算子的位点数（按 OPERATORS 的顺序，没有位点的算子也列出，值为 0）。"""
    c = Counter(s.operator for s in sites)
    return {op: c.get(op, 0) for op in OPERATORS}


# ---------------------------------------------------------------------------
# 改写
# ---------------------------------------------------------------------------
def _obo_text(original: str, rng: random.Random) -> str:
    m = _DEC_RE.match(original)
    if m is None:
        raise MutateError(f"OBO 位点的原文 {original!r} 不是十进制整数字面量")
    n, suffix = int(m.group(1)), m.group(2)
    delta = rng.choice((1, -1))
    if n + delta < 0:
        delta = 1                       # 0 不往负数改：数组下标 -1 是未定义行为，杀不杀死取决于内存布局，不可复现
    return f"{n + delta}{suffix}"


def mutation_text(site: Site, rng: random.Random) -> str:
    """位点 [start, end) 的原文换成什么（删除类为空串）。只有 OBO 用到 rng。"""
    op = site.operator
    if op in ("ROR", "LOR", "AOR"):
        table = {"ROR": _ROR_MAP, "LOR": _LOR_MAP, "AOR": _AOR_MAP}[op]
        if site.original not in table:
            raise MutateError(f"{op} 位点的原文 {site.original!r} 不是该算子能改的运算符")
        return table[site.original]
    if op == "NEG":
        return f"!({site.original})"
    if op == "OBO":
        return _obo_text(site.original, rng)
    if op in _LINE_OPS:
        return ""
    if op == "RVR":
        if site.detail not in ("return 0;", "return NULL;"):
            raise MutateError(f"RVR 位点的 detail {site.detail!r} 不是 'return 0;' / 'return NULL;'")
        return site.detail
    raise ValueError(f"未知算子 {op!r}，可选 {OPERATORS}")


def _cut_span(code: bytes, a: int, b: int) -> tuple[int, int]:
    """删掉 [a, b) 这条语句时实际要删的字节区间：独占整行（前面只有缩进、后面只有空白）就连缩进与行尾换行一起删，
    不留空行；同一行还有别的内容时只删语句本身，并带走紧挨着的一段空白（前面还有内容、它在行尾就删前面的，否则删后面的），
    免得留下连续的空格。"""
    ls = code.rfind(b"\n", 0, a) + 1
    le = code.find(b"\n", b)
    tail = code[b:] if le == -1 else code[b:le]
    first, last = not code[ls:a].strip(b" \t"), not tail.strip(b" \t\r")
    if first and last:
        return ls, len(code) if le == -1 else le + 1
    if last:                            # 前面有内容、自己是行尾：删前面的空白
        while a > ls and code[a - 1:a] in (b" ", b"\t"):
            a -= 1
        return a, b
    while b < len(code) and code[b:b + 1] in (b" ", b"\t"):
        b += 1
    return a, b


def apply_site(code: bytes, site: Site, rng: random.Random) -> bytes:
    """把 ``site`` 的改写应用到 ``code`` 上，返回新的字节串。

    ``code`` 是位点所属的整份文件的字节（site 的偏移是文件内的字节偏移）；区间里的原文必须与 ``site.original`` 一致，
    否则抛 MutateError（传错了文件，或文件在 find_sites 之后变过）。行内算子用 ``_edits.render`` 做偏移改写；
    DGS / DST 独占整行时连缩进与换行一起删掉（同行还有别的内容就只删语句和紧挨着的空白）；RVR 整句替换，不补空行。
    """
    a, b = site.start, site.end
    if not (0 <= a < b <= len(code)) or code[a:b].decode("utf-8", errors="replace") != site.original:
        raise MutateError(f"{site.operator}#{site.index} 的区间 [{a}, {b}) 与源码对不上（传错了文件，或源码在找位点之后变过）")
    new = mutation_text(site, rng).encode("utf-8")
    if site.operator in _LINE_OPS:
        a, b = _cut_span(code, a, b)
        return code[:a] + code[b:]
    if site.operator == "RVR":
        return code[:a] + new + code[b:]
    return render(code, [Edit(site.operator, a, b, [new])])


# ---------------------------------------------------------------------------
# 排序与选择
# ---------------------------------------------------------------------------
def _rng(seed: int, *parts: object) -> random.Random:
    # 字符串种子由 random 做 sha512 展开，与 PYTHONHASHSEED、Python 小版本无关
    return random.Random("|".join(str(p) for p in (seed, *parts)))


def order_sites(sites: Sequence[Site], seed: int, target_id: str, prefer_operator: str | None = None,
                anchor: Site | None = None) -> list[Site]:
    """试位点的顺序：先按（算子优先序, 序号）排成规范顺序，再用 rng(seed, target_id) 洗牌——种子里没有级别（见模块文档）；
    ``prefer_operator`` 的位点稳定地排到最前（其余保持洗牌后的相对顺序）；``anchor``（L0 位点在本级的对应位点）排在最最前面。"""
    if prefer_operator is not None and prefer_operator not in OPERATORS:
        raise ValueError(f"未知算子 {prefer_operator!r}，可选 {OPERATORS}")
    rank = {op: i for i, op in enumerate(OPERATORS)}
    order = sorted(sites, key=lambda s: (rank[s.operator], s.index))
    _rng(seed, target_id).shuffle(order)
    if prefer_operator is not None:
        order.sort(key=lambda s: s.operator != prefer_operator)
    if anchor is not None:
        order.sort(key=lambda s: not (s.operator == anchor.operator and s.index == anchor.index))
    return order


def _error_messages(tu: ci.TranslationUnit) -> Counter:
    return Counter(d.spelling for d in tu.diagnostics if d.severity >= ci.Diagnostic.Error)


class _Reparser:
    """变异后重新解析整份目标文件：不比变异前多出新的 error 才算有效。"""

    def __init__(self, spec: CorpusSpec, root: Path, target: FunctionTarget, original: bytes) -> None:
        self.unit = Unit(root / target.file)
        self.where = target.target_id
        with _CLANG_LOCK:
            self.parser = Parser(libclang_args(spec, root))
        self.base = self._messages(original)

    def _messages(self, data: bytes) -> Counter | None:
        with _CLANG_LOCK:
            try:
                return _error_messages(self.parser.parse(self.unit, data, detailed=False))
            except ci.TranslationUnitLoadError as exc:
                log.warning("%s: libclang 无法加载 %s: %s", self.where, self.unit.path.name, exc)
                return None

    def ok(self, data: bytes) -> bool:
        msgs = self._messages(data)
        return msgs is not None and not (msgs - (self.base or Counter()))


def strip_function_comments(code: str) -> str:
    """函数文本去掉注释：块注释换成一个空格，行注释删到行尾，行尾空白去掉，连续空行压成一行。

    符号混淆只改标识符，L1 以上的函数体注释里仍是项目原有的名字，原样放进修复 prompt 会把本该隐藏的名字泄漏给模型；
    四个级别一律去注释，级别之间的差别才只来自混淆本身。换行的个数不保证不变，字符串与字符常量里的 ``/*``、``//`` 原样保留。
    """
    out: list[str] = []
    for line in strip_c_comments(code).split("\n"):
        line = line.rstrip()
        if line or (out and out[-1]):       # 连续的空行只留一个
            out.append(line)
    return "\n".join(out)


def _site_rng(seed: int, target_id: str, operator: str, key: int) -> random.Random:
    """每个位点自己的 rng（只有 OBO 的 ±1 用得上）：不随试的先后变化，同一个位点永远改成同一个样子。
    ``key`` 是位点的序号；L0 位点在其它级别的对应位点用 L0 的序号，所以同一个缺陷在各级改成同一个方向。"""
    return _rng(seed, target_id, operator, key)


def _record(res: MutantSearch, ev: Any, category: str) -> None:
    """沙箱运行耗时按归因类别累计（实验记录里「各类变异体占多少评测时间」的出处）。"""
    res.sandbox_seconds[category] = round(res.sandbox_seconds.get(category, 0.0) + float(ev.duration_s or 0.0), 2)


def search_mutant(spec: CorpusSpec, obf_root: Path, target: FunctionTarget, level: str, seed: int, sandbox: Any,
                  prefer_operator: str | None = None, max_tries: int = DEFAULT_MAX_TRIES, *,
                  target_id: str | None = None, work_root: Path | None = None,
                  strip_comments: bool = True, recheck: bool = True, reference: SiteRef | None = None,
                  analysis: Analysis | None = None) -> MutantSearch:
    """在 ``level`` 这一级副本里为 ``target`` 选一个被杀死的变异体，返回完整的选择过程（见 :class:`MutantSearch`）。

    ``target`` 是该级副本里重新定位出来的目标（偏移对应 ``obf_root``）；``target_id`` 给原始（L0）目标的 id：
    L1 以上 locate 出来的目标名字已被改写，target_id 随之变化，不给就用 ``target.target_id``。它进 rng 的种子和
    结果记录，所以同一个目标的四级用同一个 id，结果才不依赖副本里的名字。``sandbox`` 必须给：没有 kill 检查的变异体不可信。
    ``work_root`` 是 evaluate 的临时工作目录（不给就用 ``store.default_work_root()``）。

    ``reference`` 是 L0 选定位点的参照：本级找得到对应位点就先试它（``same_site_as_L0`` 的依据），再按 ``prefer_operator``
    （没给就取参照的算子）试同一算子的位点。``analysis`` 给了就不再重新解析目标文件（调用方已经分析过这一级）。

    ``strip_comments``（默认开）：函数里有注释的，先验证去注释后的原函数仍通过项目测试（不通过：outcome ``strip_unsafe``，不搜索），
    变异体的函数文本先去注释再做 kill 检查并写进 ``Mutant.code``——kill 检查验证的就是最终放进 prompt 的那段文本。
    位点的选择、重新解析与 ``original_text`` / ``mutated_text`` 不受影响（它们只涉及位点本身）。
    ``recheck``（默认开）：判为杀死的候选再评测一次，第二次仍失败才出题（见模块文档）。
    """
    if level not in LEVELS:
        raise ValueError(f"未知混淆级别 {level!r}，可选 {LEVELS}")
    if sandbox is None:
        raise ValueError("choose_mutant 需要沙箱做 kill 检查（sandbox 为空）")
    tid = target_id or target.target_id
    root = Path(obf_root).resolve()
    ana = analysis if analysis is not None else analyze(spec, root, target)
    sites = ana.sites
    if prefer_operator is None and reference is not None:
        prefer_operator = reference.operator
    res = MutantSearch(target_id=tid, level=level, mutant=None, outcome="no_sites", n_sites=len(sites),
                       prefer_operator=prefer_operator, analysis=ana)
    if not sites or max_tries <= 0:
        return res
    data = (root / target.file).read_bytes()
    original = data[target.start:target.end].decode("utf-8", errors="replace")
    if strip_comments:
        res.has_comments = strip_c_comments(original) != original
        if res.has_comments:
            res.strip_checked = True
            ev = evaluate(spec, root, target, strip_function_comments(original), sandbox, CHECK_MODEL,
                          f"{tid}@{level}#stripped", 0, work_root=work_root)
            _record(res, ev, attribute(ev, Candidate(original, []))[0])
            if ev.stage != "pass":
                res.outcome = "strip_unsafe"
                log.warning("%s@%s: 去注释后的原函数通不过项目测试（stage=%s），放弃这个（目标, 级别）", tid, level, ev.stage)
                return res
    anchor = match_reference(reference, ana) if reference is not None else None
    res.anchor = "none" if reference is None else ("no_image" if anchor is None else "tried")
    reparser = _Reparser(spec, root, target, data)
    seen_code: set[str] = set()
    budget_hit = False
    for site in order_sites(sites, seed, tid, prefer_operator, anchor):
        if res.tried_sites >= max_tries:
            budget_hit = True
            break
        is_anchor = anchor is not None and (site.operator, site.index) == (anchor.operator, anchor.index)
        key = reference.index if is_anchor and reference is not None else site.index
        new = apply_site(data, site, _site_rng(seed, tid, site.operator, key))
        if new == data or not reparser.ok(new):
            res.parse_fail_count += 1
            log.debug("%s@%s: %s#%d 改写后重新解析有新 error（或没有变化），跳过", tid, level, site.operator, site.index)
            continue
        code = new[target.start:target.end + len(new) - len(data)].decode("utf-8", errors="replace")
        if strip_comments:
            code = strip_function_comments(code)
        if code in seen_code:
            continue
        seen_code.add(code)
        res.tried_sites += 1
        task_id = f"{tid}@{level}#{site.operator}{site.index}"
        ev = evaluate(spec, root, target, code, sandbox, CHECK_MODEL, task_id, 0, work_root=work_root)
        category, detail = attribute(ev, Candidate(code, []))
        _record(res, ev, category)
        res.tries.append({"operator": site.operator, "site_index": site.index, "stage": ev.stage, "category": category})
        log.debug("%s: stage=%s category=%s", task_id, ev.stage, category)
        if ev.stage == "test" and category in KILL_MODES:
            second: str | None = None
            if recheck:
                ev2 = evaluate(spec, root, target, code, sandbox, CHECK_MODEL, f"{task_id}#recheck", 0, work_root=work_root)
                second = attribute(ev2, Candidate(code, []))[0]
                _record(res, ev2, second)
                res.rechecked += 1
                res.tries[-1]["recheck"] = second if ev2.stage == "test" else f"{ev2.stage}:{second}"
                if not (ev2.stage == "test" and second in KILL_MODES):
                    res.unstable_count += 1                    # 一次失败一次通过：不是稳定的杀死，不出题
                    log.warning("%s: 第一次评测判为杀死（%s），复核却是 %s / %s，不出题", task_id, category, ev2.stage, second)
                    continue
            res.mutant = Mutant(
                target_id=tid, level=level, operator=site.operator, site_index=site.index, original_text=site.original,
                mutated_text=mutation_text(site, _site_rng(seed, tid, site.operator, key)), code=code, kill_mode=category,
                tried_sites=res.tried_sites, equivalent_count=res.equivalent_count,
                compile_fail_count=res.compile_fail_count, site_detail=site.detail,
                kill_detail=_DURATION_RE.sub("after the time limit", detail),       # 超时的耗时每次不同，写进记录会破坏可复现
                parse_fail_count=res.parse_fail_count, unstable_count=res.unstable_count, kill_recheck=second,
                same_site_as_L0=None if reference is None else is_anchor,
                context_orphans=prune_context(target.deps, code)[1])
            break
        if ev.stage == "pass":
            res.equivalent_count += 1
        else:
            res.compile_fail_count += 1
    if res.mutant is not None:
        res.outcome = "killed"
        if anchor is not None:
            res.anchor = "chosen" if res.mutant.same_site_as_L0 else "tried"
    elif budget_hit:
        res.outcome = "gave_up"
    elif res.tried_sites:
        res.outcome = "exhausted"
    log.info("%s@%s: %s（位点 %d，试 %d，等价 %d，编译失败 %d，解析失败 %d，不稳定 %d）%s", tid, level, res.outcome, res.n_sites,
             res.tried_sites, res.equivalent_count, res.compile_fail_count, res.parse_fail_count, res.unstable_count,
             f"，选中 {res.mutant.operator}#{res.mutant.site_index} ({res.mutant.kill_mode})" if res.mutant else "")
    return res


def choose_mutant(spec: CorpusSpec, obf_root: Path, target: FunctionTarget, level: str, seed: int, sandbox: Any,
                  prefer_operator: str | None = None, max_tries: int = DEFAULT_MAX_TRIES, *,
                  target_id: str | None = None, work_root: Path | None = None,
                  strip_comments: bool = True, recheck: bool = True, reference: SiteRef | None = None) -> Mutant | None:
    """附录 §8 的接口：选出一个被杀死的变异体；没有（无位点、全部等价或编译失败、试满 ``max_tries``）返回 None。
    需要放弃时的记账（试了几个、等价几个……）用 :func:`search_mutant`。"""
    return search_mutant(spec, obf_root, target, level, seed, sandbox, prefer_operator, max_tries,
                         target_id=target_id, work_root=work_root, strip_comments=strip_comments, recheck=recheck,
                         reference=reference).mutant


def choose_for_levels(spec: CorpusSpec, trees: Mapping[str, tuple[Path, FunctionTarget]], seed: int, sandbox: Any, *,
                      target_id: str, levels: Sequence[str] = LEVELS, max_tries: int = DEFAULT_MAX_TRIES,
                      reference: SiteRef | None = None, reference_operator: str | None = None,
                      work_root: Path | None = None, strip_comments: bool = True, recheck: bool = True
                      ) -> dict[str, MutantSearch]:
    """同一个目标在各级副本上选变异体，固化「L1–L3 先试 L0 的位点、再优先复用 L0 的算子」。

    ``trees[level] = (该级副本目录, 该级副本里 locate 出来的目标)``；L0 先选（不指定算子），选出了位点，其余级别拿它当参照
    （``reference``）。``levels`` 里没有 L0 时（续跑别的级别），用调用方给的 ``reference``（L0 位点的参照）或只给算子的
    ``reference_operator`` 代替。每个选出的 Mutant 记下 ``same_operator_as_L0`` / ``same_site_as_L0``：L0 自己为 True，
    L0 没选出（也没给参照）则为 None。
    """
    ref = reference
    op0 = reference.operator if reference is not None else reference_operator
    out: dict[str, MutantSearch] = {}
    for level in [lv for lv in LEVELS if lv in levels]:
        root, target = trees[level]
        res = search_mutant(spec, root, target, level, seed, sandbox, None if level == "L0" else op0, max_tries,
                            target_id=target_id, work_root=work_root, strip_comments=strip_comments, recheck=recheck,
                            reference=None if level == "L0" else ref)
        if level == "L0" and res.mutant is not None:
            op0 = res.mutant.operator
            ref = reference_for(res.analysis, op0, res.mutant.site_index) if res.analysis is not None else None
        if res.mutant is not None:
            res.mutant.same_operator_as_L0 = None if op0 is None else res.mutant.operator == op0
            if level == "L0":
                res.mutant.same_site_as_L0 = True
        out[level] = res
    return out
