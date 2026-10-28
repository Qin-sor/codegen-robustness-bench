"""归一化相似度（设计文档 §4.13）：衡量模型生成的函数与（对应混淆级的）原函数有多像，作为记忆痕迹指标。

算法：
  1. 用一个小型 C 词法器切 token（:func:`lex`，公开接口，postprocess / prompt 等需要切记号的地方都用它，不再各自导入私有的正则），
     注释直接丢弃，字符串 / 字符常量 / 数字原样保留；
  2. 所有非关键字的标识符统一换成 ``ID``（所以单纯改名不会改变相似度）；
  3. 空白在切 token 时就已经消失；
  4. 对两条 token 序列取 ``difflib.SequenceMatcher(...).ratio()``，范围 [0, 1]；每边最多取前 ``MAX_TOKENS`` 个 token。

与设计文档字面描述（「标识符全换成 ID、去空白后的 difflib ratio」）有两处细化：

* difflib 比较的是 token 序列，不是拼接后的字符序列。字符级比较在 200 字符以上会触发 difflib 的 autojunk
  启发式（高频字符被当成垃圾，比值失真），关掉它又会让长函数的耗时按平方增长；token 级既快又不失真；
* C 关键字（int / if / return 等）不算标识符，保留原样，它们是结构信息；其余标识符（包括 libc 函数名、
  typedef 名、宏名）一律归一成 ID。
"""
from __future__ import annotations

import difflib
import re

__all__ = ["normalized_similarity", "normalized_tokens", "normalized_equal", "lex", "exact_tokens", "identifiers", "ID_TOKEN",
           "C_KEYWORDS", "MAX_TOKENS"]

ID_TOKEN = "ID"
# 每边最多比较的 token 数：真实函数（<= 120 行）不到 1000 个 token；这个上限只防模型吐出几万 token 的退化输出拖慢 difflib
MAX_TOKENS = 6000

# C89 / C99 / C11 关键字，外加常见的 GNU 扩展关键字（asm / typeof / __attribute__ 等）。
# 注意与 bench.obfuscate.symbols.C_KEYWORDS 不是同一张表、用途不同：那张表是「符号混淆不能拿来当新名字 / 不能改名的拼写」，
# 含 C23 的 bool / true / false / nullptr 等；这张表只回答「相似度里哪些记号是语法结构、要原样保留，其余标识符一律归一成 ID」。
C_KEYWORDS = frozenset("""
auto break case char const continue default do double else enum extern float for goto if inline int long
register restrict return short signed sizeof static struct switch typedef union unsigned void volatile while
_Bool _Complex _Imaginary _Alignas _Alignof _Atomic _Generic _Noreturn _Static_assert _Thread_local
asm __asm __asm__ typeof __typeof__ __attribute__ __extension__ __inline __inline__ __restrict __restrict__
""".split())

_TOKEN_RE = re.compile(
    r"""
      (?P<comment>/\*.*?\*/|//[^\n]*)
    | (?P<string>(?:u8|[uUL])?"(?:\\.|[^"\\\n])*")
    | (?P<char>[uUL]?'(?:\\.|[^'\\\n])*')
    | (?P<number>0[xX][0-9a-fA-F]+[uUlL]*|(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?[fFlLuU]*)
    | (?P<ident>[A-Za-z_]\w*)
    | (?P<punct>->|\+\+|--|<<=|>>=|\.\.\.|<<|>>|<=|>=|==|!=|&&|\|\||[-+*/%&|^]=|\#\#
               |[-+*/%&|^~!=<>?:;,.(){}\[\]\#])
    | (?P<other>\S)
    """,
    re.VERBOSE | re.DOTALL,
)


def lex(code: str) -> list[tuple[str, str]]:
    """C 代码 → [(种类, 文本)]：注释丢弃，空白在切 token 时消失，字符串 / 字符常量 / 数字原样。

    种类取 ``string`` / ``char`` / ``number`` / ``ident`` / ``punct`` / ``other``；``ident`` 包含关键字（要区分的话查 :data:`C_KEYWORDS`）。
    这是本项目唯一的 C 词法器：相似度、``normalized_equal``、依赖上下文的裁剪（``prompt.prune_context``）都从这里切记号。"""
    return [(m.lastgroup or "other", m.group()) for m in _TOKEN_RE.finditer(code or "") if m.lastgroup != "comment"]


def exact_tokens(code: str) -> list[str]:
    """C 代码 → 记号文本序列：词法器同 :func:`lex`，标识符也原样保留（不像 :func:`normalized_tokens` 那样归一成 ID）。"""
    return [text for _, text in lex(code)]


def identifiers(code: str) -> set[str]:
    """代码里出现的标识符记号（含关键字之外的全部名字；字符串、字符常量和注释里的不算）。"""
    return {text for kind, text in lex(code) if kind == "ident" and text not in C_KEYWORDS}


def normalized_tokens(code: str) -> list[str]:
    """C 代码 → 归一化 token 序列：去注释，非关键字标识符换成 ID，其余原样。"""
    return [ID_TOKEN if kind == "ident" and text not in C_KEYWORDS else text for kind, text in lex(code)]


def normalized_equal(a: str | None, b: str | None) -> bool:
    """两段 C 代码规范化后是否相同；缺陷修复任务的 ``unchanged``（候选与变异体相同，模型什么也没改）与
    ``restored``（候选与该级原函数相同，逐字还原）都靠它判定。

    规范化沿用本模块的词法器：只有排版（空白、换行、缩进）和注释的差别算相同。与 :func:`normalized_tokens`
    不同，标识符**不**归一成 ``ID``，也原样比较：否则 ``a - b`` 与 ``b - a``、``x[i]`` 与 ``x[j]`` 会被当成同一段代码，
    把改错了变量的候选算成「什么也没改」或「逐字还原」。

    任一方是 None（没有抽出代码）或规范化后一个 token 都没有，返回 False：没有代码谈不上「相同」。
    """
    if a is None or b is None:
        return False
    ta = exact_tokens(a)
    return bool(ta) and ta == exact_tokens(b)


def normalized_similarity(a: str, b: str) -> float:
    """两段 C 代码的归一化相似度，[0, 1]。

    * 同一函数只改了标识符名字：1.0；
    * 任一方为空（归一化后没有 token）而另一方不空：0.0；两边都空：1.0（与 difflib 约定一致）。
    """
    ta, tb = normalized_tokens(a)[:MAX_TOKENS], normalized_tokens(b)[:MAX_TOKENS]
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return difflib.SequenceMatcher(None, ta, tb, autojunk=False).ratio()
