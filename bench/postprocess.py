"""模型回复的后处理（设计文档 §4.10）：抽出代码块，检查它是不是「只含目标函数的完整定义」。

流程：``extract_code`` 从回复里挑出一个代码块 → ``normalize`` 做规整与检查 → 得到 ``Candidate``。
``process_reply`` 把两步串起来，直接吃 ``prompt.Task``。

``Candidate.issues`` 取值（按此顺序排列，不重复）：

* ``no_code_block``：没有可用的代码（回复里没有代码块，或代码块里除 ``#include`` 之外什么都没有）。
  此时 ``code`` 为 None，归因为 extract_fail；
* ``multiple_functions``：代码里不止一个函数定义（多出来的辅助函数违反「内含且仅含目标函数」）；
* ``signature_mismatch``：找不到与期望签名同名的定义，或参数个数 / 类型、返回类型、static 与期望的不一致。
  只记录，不丢弃代码，交给编译器判；
* ``added_include``：代码里有题目上下文没有提供的 ``#include``。``forbidden_includes=True`` 时这些行会被删掉
  （上下文已经包含的头文件重复写一遍无害，静默删掉不记问题）；
* ``truncated``：代码像是被截断的——花括号没闭合，或停在注释 / 字符串中间，或调用方说 finish_reason 是 length；
  ``process_reply`` 还会在代码块没有收尾围栏、且调用方没有明说 finish_reason 是 stop 时记它；
* ``policy_violation``：违反提示词里的硬性约束，具体内容在 ``Candidate.violations``：调用或引用了禁用函数
  （``prompt.FORBIDDEN_CALLS`` 与网络函数）、内联汇编、``__attribute__((constructor / destructor))``、
  ``#define`` / ``#undef`` / ``#pragma``。这类候选**不进沙箱**，归因为 policy_violation 并按失败计：
  候选里只写一句 ``exit(0);`` 就能让只看退出码的语料误判为通过，``#define`` 一个断言宏也能让同文件里的测试形同虚设。
  调用扫描靠 libclang（只认候选自己写的部分，上下文里项目自己声明的同名函数不算）；libclang 解析失败时退回按标识符扫描，
  同时 ``Candidate.analysis_skipped`` 置位。

签名比对用 libclang：把 ``task.context`` 的声明拼在代码前面一起解析（这样 ``size_t``、项目 typedef、
包装返回类型的宏都认得），取同名函数定义，比较返回类型、参数个数与每个参数的类型，参数名不比。
类型比较先看 libclang 给出的拼写，拼写不同再看规范类型（``size_t`` 与 ``unsigned long`` 算一样）。
如果期望签名的函数头范围内有 error 级诊断（如未知类型名，说明上下文不全），类型就不可靠，只比较函数名与参数个数；
期望签名解析干净、候选的函数头却有 error 级诊断，算 ``signature_mismatch``。
libclang 一律按默认的 GNU C 标准解析，不跟随任务的 ``-std``：这里只比较函数头、数函数定义，
和语言标准有关的诊断（c89 下的 ``//`` 注释、``inline`` 等）留给后面 gcc 的编译阶段去报。

libclang 找不到编译器内置头时，解析前借 ``clangcfg.builtin_include_args()`` 补上。

缺陷修复任务（docs/设计-缺陷修复任务.md）的候选同样走 ``process_reply``（``RepairTask`` 也有 ``obf_name`` / ``obf_signature`` /
``context`` / ``includes``）。候选与变异体（``unchanged``）、候选与该级原函数（``restored``）是否相同由
``bench.similarity.normalized_equal`` 判定（纯词法比较，放在词法器所在的模块里）。
"""
from __future__ import annotations

import dataclasses
import functools
import logging
import re
import textwrap
import threading
from dataclasses import dataclass, field
from typing import NamedTuple

import clang.cindex as ci

from bench import clangcfg
from bench.clangcfg import builtin_include_args
from bench.extract import Dependencies, strip_c_comments
from bench.prompt import FORBIDDEN_CALLS, RepairTask, Task, format_context
from bench.vocab import ISSUES as _ISSUES

log = logging.getLogger(__name__)

__all__ = [
    "ISSUES",
    "Candidate",
    "CodeBlock",
    "SignatureParts",
    "parse_blocks",
    "extract_code",
    "normalize",
    "process_reply",
    "split_signature",
    "return_type_text",
    "has_policy_directives",
]

ISSUES = _ISSUES


@dataclass
class Candidate:
    code: str | None
    issues: list[str] = field(default_factory=list)   # 取值与顺序见 ISSUES
    violations: list[str] = field(default_factory=list)   # policy_violation 的具体内容，如 ["call:exit", "directive:#define"]
    analysis_skipped: bool = False   # libclang 解析失败，multiple_functions / signature_mismatch / 调用扫描没做


# --------------------------------------------------------------------------- 代码块抽取


@dataclass(frozen=True)
class CodeBlock:
    lang: str      # 语言标记：小写、只取第一个词；没有标记为空串
    code: str
    closed: bool   # 是否见到收尾围栏；False 说明回复在代码块中间结束了


_FENCE_OPEN_RE = re.compile(r"^[ \t]*(`{3,}|~{3,})[ \t]*([^\n]*)$")
# 可以当 C 看的语言标记；其它标记（bash / text / diff / json 等）的块排在后面
_C_LANGS = frozenset({"", "c", "h", "cc", "cpp", "c++", "cxx", "hpp", "c89", "c90", "c99", "c11", "c17", "gnuc", "ansic", "clang"})


def _lang_of(info: str) -> str:
    """围栏后的说明串取语言：```c / ```C / ``` c / ```{.c .numberLines} / ```language-c 都得到 c。"""
    info = info.strip().lstrip("{").lstrip(".")
    tok = re.split(r"[\s{},;]+", info, maxsplit=1)[0].lower()
    for pre in ("language-", "lang-"):
        if tok.startswith(pre):
            tok = tok[len(pre):]
    return tok


def parse_blocks(text: str) -> list[CodeBlock]:
    """逐行扫描 Markdown 围栏代码块（``` 或 ~~~，三个及以上；收尾围栏要同种字符且不短于开头）。

    没有收尾围栏的块（回复被截断）收下到末尾，closed=False。反引号围栏的说明串里不能再有反引号，
    所以单行的 ```foo``` 不当作围栏。
    """
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    blocks: list[CodeBlock] = []
    i, n = 0, len(lines)
    while i < n:
        m = _FENCE_OPEN_RE.match(lines[i])
        if m is None or (m.group(1)[0] == "`" and "`" in m.group(2)):
            i += 1
            continue
        fence, info = m.group(1), m.group(2)
        close_re = re.compile(r"^[ \t]*" + re.escape(fence[0]) + "{" + str(len(fence)) + r",}[ \t]*$")
        body: list[str] = []
        j, closed = i + 1, False
        while j < n:
            if close_re.match(lines[j]):
                closed = True
                break
            body.append(lines[j])
            j += 1
        blocks.append(CodeBlock(_lang_of(info), "\n".join(body), closed))
        i = j + 1
    return blocks


def _skeleton_depth(text: str, upto: int | None = None) -> tuple[int, bool]:
    """扫描 C 文本（跳过注释、字符串、字符常量），返回 (花括号净深度, 是否在末尾停在注释 / 字符串中间)。

    upto 不为 None 时只扫描 text[:upto]，用来求某个位置的嵌套深度。
    """
    if upto is not None:
        text = text[:upto]
    depth = 0
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if ch == "/" and nxt == "/":
            j = text.find("\n", i)
            if j < 0:
                return depth, False   # 行注释一直到末尾：注释本身是完整的
            i = j + 1
        elif ch == "/" and nxt == "*":
            j = text.find("*/", i + 2)
            if j < 0:
                return depth, True
            i = j + 2
        elif ch == '"' or ch == "'":
            j = i + 1
            while j < n and text[j] != ch and text[j] != "\n":
                j += 2 if text[j] == "\\" else 1
            if j >= n:
                return depth, True    # 引号一直没闭合就到了末尾
            i = j + 1 if text[j] == ch else j   # 行内没闭合：当普通字符，别吞掉后面的行
        else:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            i += 1
    return depth, False


def _defines(code: str, name: str) -> bool:
    """code 里是否有 name 的函数定义：顶层（花括号深度 0）出现 ``name(...) {``，调用点不算。"""
    pat = re.compile(r"(?<![\w$])" + re.escape(name) + r"\s*\([^;{}]*\)[^;{}]*\{")
    for m in pat.finditer(code):
        if _skeleton_depth(code, m.start())[0] == 0:
            return True
    return False


def _looks_like_function_definition(code: str) -> bool:
    return re.search(r"\)\s*\{", code) is not None


def _pick_block(blocks: list[CodeBlock], func_name: str | None) -> CodeBlock:
    """多个代码块时选一个：含目标函数定义的 > 只提到目标函数名的 > 其它；C 或无标记的优先于 bash / text 等；
    同分取最前面的（模型通常先给答案、后给调用示例）。"""
    def score(b: CodeBlock) -> int:
        s = 10 if b.lang in _C_LANGS else 0
        if func_name:
            if _defines(b.code, func_name):
                s += 100
            elif re.search(r"(?<![\w$])" + re.escape(func_name) + r"(?![\w$])", b.code):
                s += 50
        elif _looks_like_function_definition(b.code):
            s += 100
        return s + (1 if b.closed else 0)

    return blocks[max(range(len(blocks)), key=lambda i: (score(blocks[i]), -i))]


def _choose_block(reply_text: str, func_name: str | None) -> CodeBlock | None:
    blocks = [b for b in parse_blocks(reply_text or "") if b.code.strip()]
    return _pick_block(blocks, func_name) if blocks else None


def _block_code(block: CodeBlock) -> str:
    return textwrap.dedent(block.code).strip()


def extract_code(reply_text: str, func_name: str | None = None) -> str | None:
    """从回复里抽出代码；没有代码块返回 None。

    处理带 c 标记的、无标记的、``` C / ~~~ / 四个反引号围栏的块；多个块时挑含 func_name 定义的那个
    （func_name 为 None 时挑第一个像函数定义的）；统一换行符并去掉整体缩进与首尾空行。
    只有裸代码、没有围栏的回复按「无代码块」处理——系统提示要求必须给 ```c 代码块。
    """
    chosen = _choose_block(reply_text, func_name)
    return _block_code(chosen) if chosen is not None else None


# --------------------------------------------------------------------------- 签名文本解析（纯字符串，不依赖 libclang）


class SignatureParts(NamedTuple):
    ret: str       # 函数名之前的全部文本：存储类、返回类型、宏包装等，原样
    name: str
    params: str    # 参数括号里的文本


def split_signature(sig: str) -> SignatureParts | None:
    """把 ``static const char *op_name(op_kind_t op)`` 拆成 (返回类型文本, 函数名, 参数文本)。

    规则：以最后一个顶层括号组为参数表，紧挨在它前面的标识符是函数名。返回函数指针的写法
    （``int (*f(int))(int)``）和带尾部属性的写法认不出来，返回 None。
    """
    s = sig.strip()
    while s and s[-1] in ";{":
        s = s[:-1].rstrip()
    if not s.endswith(")"):
        return None
    depth = 0
    open_idx = -1
    for i in range(len(s) - 1, -1, -1):
        if s[i] == ")":
            depth += 1
        elif s[i] == "(":
            depth -= 1
            if depth == 0:
                open_idx = i
                break
    if open_idx < 0:
        return None
    head = s[:open_idx].rstrip()
    m = re.search(r"([A-Za-z_]\w*)$", head)
    if m is None:
        return None
    return SignatureParts(head[: m.start()].strip(), m.group(1), s[open_idx + 1 : -1])


_STORAGE_WORDS = frozenset({"static", "extern", "inline", "__inline", "__inline__", "register", "_Noreturn", "__extension__", "auto"})
_QUALIFIERS = frozenset({"const", "volatile", "restrict", "__restrict", "__restrict__", "_Atomic"})
_BUILTIN_TYPE_WORDS = frozenset({"void", "char", "short", "int", "long", "float", "double", "signed", "unsigned", "_Bool", "_Complex", "__int128"})
_TAG_WORDS = frozenset({"struct", "union", "enum"})
_ATTR_RE = re.compile(r"__attribute__\s*\(\((?:[^()]|\([^()]*\))*\)\)|__declspec\s*\((?:[^()]|\([^()]*\))*\)|_Alignas\s*\([^()]*\)")
_MACRO_CALL_RE = re.compile(r"(?<![\w$])(?!typeof\b|__typeof__\b|__typeof\b)([A-Za-z_]\w*)\s*\(((?:[^()]|\([^()]*\))*)\)")


def return_type_text(ret: str) -> str:
    """从 split_signature 得到的返回类型文本里取出「可以拿来声明局部变量」的纯类型。

    去掉存储类（static / inline / extern）、属性、``NAME(type)`` 形式的宏包装（取括号里的类型），
    以及 ``JSMN_API int`` 这类放在类型前面、展开后可能是 extern / static 的装饰宏。
    例：``static const char *`` → ``const char *``；``CJSON_PUBLIC(cJSON *)`` → ``cJSON *``；
    ``JSMN_API int`` → ``int``；``static inline mpc_state_t`` → ``mpc_state_t``。认不出来返回空串。
    """
    t = _ATTR_RE.sub(" ", ret)
    for _ in range(4):   # 宏包装可能嵌套
        t2 = _MACRO_CALL_RE.sub(lambda m: " " + m.group(2) + " ", t)
        if t2 == t:
            break
        t = t2
    tokens = [w for w in re.findall(r"[A-Za-z_]\w*|\*", t) if w not in _STORAGE_WORDS]
    plain: list[int] = []   # 普通标识符的位置：typedef 名或装饰宏（不含 struct / union / enum 后面的标签名）
    prev = ""
    for i, w in enumerate(tokens):
        if w != "*" and w not in _BUILTIN_TYPE_WORDS and w not in _QUALIFIERS and w not in _TAG_WORDS and prev not in _TAG_WORDS:
            plain.append(i)
        prev = w
    has_keyword_type = any(w in _BUILTIN_TYPE_WORDS or w in _TAG_WORDS for w in tokens)
    # 已经有 int / struct 之类的类型关键字时，其余普通标识符只能是装饰宏；
    # 否则最后一个普通标识符是 typedef 类型名，它前面的是装饰宏
    drop = set(plain) if has_keyword_type else set(plain[:-1])
    return " ".join(w for i, w in enumerate(tokens) if i not in drop)


def _split_params(params: str) -> list[str]:
    """按顶层逗号切参数表；空表和单独的 void 得到空列表。"""
    params = params.strip()
    if not params or params == "void":
        return []
    out, depth, cur = [], 0, []
    for ch in params:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    out.append("".join(cur).strip())
    return out


# --------------------------------------------------------------------------- libclang 分析

_CAND_FILE = "__bench_candidate__.c"
_PARSE_LOCK = threading.Lock()
_ARRAY_KINDS = frozenset({ci.TypeKind.CONSTANTARRAY, ci.TypeKind.INCOMPLETEARRAY, ci.TypeKind.VARIABLEARRAY})


@dataclass(frozen=True)
class _Head:
    """一个函数的头部信息，只含纯字符串 / 数字，可以安全缓存。"""

    name: str
    static: bool
    ret: tuple[str, str]                    # (拼写, 规范类型拼写)
    params: tuple[tuple[str, str], ...]
    variadic: bool
    reliable: bool                          # 头部范围内没有 error 级诊断，类型信息可信


def _decay(spelling: str) -> str:
    """数组形参的类型拼写退化成指针：``const int[]`` → ``const int *``（两边用同一个变换，所以多维数组也只会前后一致）。"""
    base = re.sub(r"\s*\[[^\]]*\]\s*$", "", spelling)
    return base + ("*" if base.endswith("*") else " *")


def _type_key(t: ci.Type) -> tuple[str, str]:
    canon = t.get_canonical()
    if canon.kind in _ARRAY_KINDS:   # int a[] 与 int *a 是一回事
        return _decay(t.spelling), _decay(canon.spelling)
    return t.spelling, canon.spelling


def _parse(text: str) -> ci.TranslationUnit:
    args = ["-x", "c", "-w", "-ferror-limit=0", *builtin_include_args()]
    return clangcfg.index().parse(_CAND_FILE, args=args, unsaved_files=[(_CAND_FILE, text)], options=0)


def _heads(tu: ci.TranslationUnit, prefix_bytes: int, definitions: bool) -> list[_Head]:
    """TU 里位于前缀之后的函数声明 / 定义的头部信息。"""
    errors = [
        d.location.offset
        for d in tu.diagnostics
        if d.severity >= ci.Diagnostic.Error and d.location.file is not None and d.location.file.name == _CAND_FILE
    ]
    out: list[_Head] = []
    for c in tu.cursor.get_children():
        if c.kind != ci.CursorKind.FUNCTION_DECL:
            continue
        loc = c.location
        if loc.file is None or loc.file.name != _CAND_FILE or loc.offset < prefix_bytes:
            continue
        if definitions and not c.is_definition():
            continue
        start = c.extent.start.offset
        body = next((ch for ch in c.get_children() if ch.kind == ci.CursorKind.COMPOUND_STMT), None)
        end = body.extent.start.offset if body is not None else c.extent.end.offset
        out.append(
            _Head(
                name=c.spelling,
                static=c.storage_class == ci.StorageClass.STATIC,
                ret=_type_key(c.result_type),
                params=tuple(_type_key(a.type) for a in c.get_arguments()),
                variadic=bool(c.type.is_function_variadic()),
                reliable=not any(start <= off < end for off in errors),
            )
        )
    return out


@functools.lru_cache(maxsize=512)
def _expected_head(prefix: str, signature: str) -> _Head | None:
    """期望签名的头部信息（同一任务的多个样本共用，所以缓存）。"""
    sig = signature.strip().rstrip(";{ \t\n")
    with _PARSE_LOCK:
        tu = _parse(prefix + sig + ";\n")
        heads = _heads(tu, len(prefix.encode("utf-8")), definitions=False)
    if heads:
        return heads[-1]
    parts = split_signature(sig)   # libclang 一个函数声明都没认出来：退回纯文本，只能比较名字与参数个数
    if parts is None:
        return None
    params = _split_params(parts.params)
    return _Head(
        name=parts.name,
        static="static" in re.findall(r"[A-Za-z_]\w*", parts.ret),
        ret=("", ""),
        params=tuple(("", "") for _ in params),
        variadic=bool(params) and params[-1] == "...",
        reliable=False,
    )


def _same_type(a: tuple[str, str], b: tuple[str, str]) -> bool:
    return a[0] == b[0] or a[1] == b[1]


def _heads_match(exp: _Head, cand: _Head) -> bool:
    if exp.name != cand.name or len(exp.params) != len(cand.params):
        return False
    if exp.variadic != cand.variadic:
        return False
    if not exp.reliable:
        return True    # 期望签名自己都解析不干净（上下文不全）：只比较名字与参数个数
    if not cand.reliable:
        return False   # 同一份上下文下期望签名解析干净、候选的函数头却有错误：类型名对不上之类，算不匹配
    if exp.static != cand.static or not _same_type(exp.ret, cand.ret):
        return False
    return all(_same_type(x, y) for x, y in zip(exp.params, cand.params))


@dataclass
class _Analysis:
    multiple: bool
    mismatch: bool
    calls: list[str]            # 候选里调用 / 引用的禁用函数（libclang 认出来的）


def _analyze(code: str, expected_signature: str, context: Dependencies | None) -> _Analysis | None:
    """libclang 分析：有多个函数定义、签名不匹配、禁用函数的调用；解析失败时返回 None（这几项本次都没做）。"""
    prefix = format_context(context, comments=False) + "\n" if context is not None else ""
    prefix = prefix if prefix.strip() else ""
    prefix_bytes = len(prefix.encode("utf-8"))
    try:
        exp = _expected_head(prefix, expected_signature)
        with _PARSE_LOCK:
            tu = _parse(prefix + code + "\n")
            defs = _heads(tu, prefix_bytes, definitions=True)
            calls = _forbidden_refs(tu, prefix_bytes)
    except (ci.TranslationUnitLoadError, ci.LibclangError, OSError) as e:
        log.warning("libclang 解析候选代码失败（%s），跳过 multiple_functions / signature_mismatch / 禁用调用检查", e)
        return None
    multiple = len(defs) > 1
    if exp is None:
        return _Analysis(multiple, False, calls)
    same_name = [h for h in defs if h.name == exp.name]
    mismatch = not same_name or not any(_heads_match(exp, h) for h in same_name)
    return _Analysis(multiple, mismatch, calls)


# --------------------------------------------------------------------------- 政策检查（提示词里的硬性约束）

_EXEC_RE = re.compile(r"^(?:exec(?:l|le|lp|v|ve|vp|vpe)|fexecve)$")
#: 编译器内建的终止 / 陷入函数
_BUILTIN_STOPS = frozenset({"__builtin_trap", "__builtin_abort", "__builtin_exit", "__builtin__exit"})
#: 网络函数：提示词只写了「不要用网络」，这里列出常见的 BSD socket 入口。这些名字也常被当普通变量名用，
#: 所以只在「调用」位置（后面紧跟 ``(``）且上下文里没有项目自己的同名声明时才算
_NET_NAMES = frozenset("socket connect bind listen accept accept4 send sendto sendmsg recv recvfrom recvmsg "
                       "getaddrinfo gethostbyname".split())
_FORBIDDEN_FUNCS = frozenset(n for n in FORBIDDEN_CALLS if not n.endswith("*"))
_ATTR_BAD_RE = re.compile(r"__attribute__\s*\(\(([^;{}]*?)\)\)", re.S)
_ATTR_BAD_WORDS = re.compile(r"\b(?:constructor|destructor)\b")
_DIRECTIVE_BAD_RE = re.compile(r"^[ \t]*#[ \t]*(define|undef|pragma)\b", re.M)
_ASM_RE = re.compile(r"(?<![\w$])(?:asm|__asm__|__asm)(?![\w$])")
_STRING_RE = re.compile(r'"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'')
_IDENT_TOKEN_RE = re.compile(r"(?<![\w$])[A-Za-z_]\w*")


def _is_forbidden_name(name: str) -> bool:
    """name 是不是提示词禁用的函数（含 exec 系列和编译器内建的终止函数）。"""
    return name in _FORBIDDEN_FUNCS or bool(_EXEC_RE.match(name)) or name in _BUILTIN_STOPS


def _forbidden_refs(tu: ci.TranslationUnit, prefix_bytes: int) -> list[str]:
    """候选自己写的部分（位于上下文前缀之后）里调用或引用的禁用函数，格式 ``call:名字`` / ``asm``。

    引用（取地址）也算：``void (*f)(int) = exit; f(0);`` 与直接调用等效。上下文前缀里项目自己声明的同名函数不算。
    宏展开出来的调用，位置记在宏调用处，所以经项目宏调用 exit 同样能发现。
    """
    found: set[str] = set()
    for top in tu.cursor.get_children():
        loc = top.location
        if loc.file is None or loc.file.name != _CAND_FILE or loc.offset < prefix_bytes:
            continue
        for c in top.walk_preorder():
            try:
                kind = c.kind
            except ValueError:
                continue
            if kind == ci.CursorKind.ASM_STMT:
                found.add("asm")
                continue
            if kind not in (ci.CursorKind.CALL_EXPR, ci.CursorKind.DECL_REF_EXPR):
                continue
            name = c.spelling
            is_net = name in _NET_NAMES
            if not (_is_forbidden_name(name) or (is_net and kind == ci.CursorKind.CALL_EXPR)):
                continue
            ref = c.referenced
            if ref is not None:
                if ref.kind != ci.CursorKind.FUNCTION_DECL:
                    continue          # 引用的是变量、参数、结构体成员（函数指针成员 ``obj->exit(0)``）之类，不是库函数
                rloc = ref.location
                if rloc.file is not None and rloc.file.name == _CAND_FILE and rloc.offset < prefix_bytes:
                    continue          # 上下文里项目自己声明的同名函数
            found.add(f"call:{name}")
    return sorted(found)


def _blank_strings(text: str) -> str:
    """去掉注释，并把字符串 / 字符常量的内容清空（只留引号），免得字面量里的字被当成代码。"""
    return _STRING_RE.sub('""', strip_c_comments(text))


def has_policy_directives(code: str) -> bool:
    """code 里有没有 ``#define`` / ``#undef`` / ``#pragma``（去掉注释和字符串之后）。出题时拿原函数文本调它：
    原函数自己就带这类指令（inih 的 ``ini_parse_stream`` 里有 ``#define HANDLER``，parson 的 ``parson_sprintf`` 里有 ``#pragma``）
    的目标，对应的候选也允许写，否则连原函数本身都过不了约束。"""
    return bool(_DIRECTIVE_BAD_RE.search(_blank_strings(code)))


def _text_violations(code: str, context: Dependencies | None, *, scan_calls: bool,
                     allow_directives: bool = False) -> list[str]:
    """不依赖 libclang 的政策检查：预处理指令（``allow_directives`` 为真时不查）、constructor / destructor 属性、
    内联汇编关键字；``scan_calls`` 为真（libclang 没能分析）时再按标识符扫描禁用函数。"""
    bare = _blank_strings(code)
    out: set[str] = set()
    if not allow_directives:
        for m in _DIRECTIVE_BAD_RE.finditer(bare):
            out.add(f"directive:#{m.group(1)}")
    for m in _ATTR_BAD_RE.finditer(bare):
        w = _ATTR_BAD_WORDS.search(m.group(1))
        if w:
            out.add(f"attribute:{w.group(0)}")
    if _ASM_RE.search(bare):
        out.add("asm")
    if scan_calls:
        own = set()
        if context is not None:
            ctx_text = _blank_strings(format_context(context, comments=False))
            own = {m.group(0) for m in _IDENT_TOKEN_RE.finditer("\n".join(
                ln for ln in ctx_text.splitlines() if not ln.lstrip().startswith("#include")))}
        for m in _IDENT_TOKEN_RE.finditer(bare):
            name = m.group(0)
            if name in own:
                continue
            before = bare[:m.start()].rstrip()
            if before.endswith(".") or before.endswith("->"):
                continue              # 结构体成员，不是函数
            if _is_forbidden_name(name):
                out.add(f"call:{name}")
            elif name in _NET_NAMES and bare[m.end():].lstrip().startswith("("):
                out.add(f"call:{name}")
    return sorted(out)


# --------------------------------------------------------------------------- 规整与检查

_INCLUDE_RE = re.compile(r'^[ \t]*#[ \t]*include[ \t]*([<"][^>"\n]+[>"])[^\n]*$')


def _include_key(line: str) -> str | None:
    m = _INCLUDE_RE.match(line)
    return m.group(1).replace(" ", "") if m else None


def normalize(
    code: str | None,
    expected_signature: str,
    forbidden_includes: bool = True,
    context: Dependencies | None = None,
    *,
    finish_reason: str | None = None,
    allow_directives: bool = False,
) -> Candidate:
    """规整候选代码并检查问题，见模块文档里 issues 的含义。

    * ``forbidden_includes``：True 时删掉代码里所有 ``#include`` 行，其中 ``context.includes`` 没有提供的记
      ``added_include``；False 时原样保留、不记。
    * ``context``：任务的依赖声明，拼在代码前面做 libclang 解析（设计文档 §4.10 的「加上任务的依赖声明作为前缀」）；
      不给也能跑，但项目 typedef / 宏认不出来，类型比较会退化成只比名字与参数个数。
    * ``finish_reason``：Reply.finish_reason；等于 "length" 时记 ``truncated``。
    * ``allow_directives``：原函数自己带 ``#define`` / ``#pragma`` 的目标为真，此时候选里的这类指令不算违反约束
      （见 :func:`has_policy_directives`）。
    * ``code`` 为 None 或全空白时返回 ``Candidate(None, ["no_code_block"])``。
    """
    if code is None or not code.strip():
        return Candidate(None, ["no_code_block"])
    issues: set[str] = set()
    text = code.replace("\r\n", "\n").replace("\r", "\n")

    if forbidden_includes:
        provided = {k for k in (_include_key(ln) for ln in (context.includes if context else [])) if k}
        kept: list[str] = []
        for line in text.split("\n"):
            key = _include_key(line)
            if key is None:
                kept.append(line)
            elif key not in provided:
                issues.add("added_include")
        text = "\n".join(kept)
    text = text.strip("\n")
    if not text.strip():
        return Candidate(None, sorted(issues | {"no_code_block"}, key=ISSUES.index))

    depth, unterminated = _skeleton_depth(text)
    if finish_reason == "length" or depth > 0 or unterminated:
        issues.add("truncated")

    analysis = _analyze(text, expected_signature, context)
    violations = set(_text_violations(text, context, scan_calls=analysis is None, allow_directives=allow_directives))
    if analysis is not None:
        if analysis.multiple:
            issues.add("multiple_functions")
        if analysis.mismatch:
            issues.add("signature_mismatch")
        violations.update(analysis.calls)
    if violations:
        issues.add("policy_violation")
    return Candidate(text, sorted(issues, key=ISSUES.index), sorted(violations), analysis_skipped=analysis is None)


def process_reply(
    reply_text: str,
    task: Task | RepairTask,
    *,
    forbidden_includes: bool = True,
    finish_reason: str | None = None,
    allow_directives: bool = False,
) -> Candidate:
    """extract_code + normalize：回复文本 → Candidate，目标函数名 / 期望签名 / 上下文都取自 task（生成任务的 Task
    或缺陷修复任务的 RepairTask，只用到 ``obf_name`` / ``obf_signature`` / ``context`` / ``includes``）。

    ``task.includes`` 非空时它就是 prompt 里告诉模型「已经包含」的头文件，优先于 ``task.context.includes``
    （与 ``build_messages`` 一致），所以这里按它判断 ``added_include`` 并拼进 libclang 前缀。
    选中的代码块没有收尾围栏、而调用方又没有明说 ``finish_reason == "stop"`` 时，也记 ``truncated``。
    """
    chosen = _choose_block(reply_text, task.obf_name)
    code = _block_code(chosen) if chosen is not None else None
    context = dataclasses.replace(task.context, includes=list(task.includes)) if task.includes else task.context
    cand = normalize(code, task.obf_signature, forbidden_includes, context, finish_reason=finish_reason,
                     allow_directives=allow_directives)
    if chosen is not None and not chosen.closed and finish_reason != "stop" and "truncated" not in cand.issues:
        cand.issues = sorted([*cand.issues, "truncated"], key=ISSUES.index)
    return cand

