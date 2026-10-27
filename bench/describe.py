"""功能规格（描述）生成与泄漏检查（设计文档 §4.8）。

``describe(target, code, llm, project_names)`` 让（最强可用的）模型为一个目标函数写一段英文功能规格：
只讲行为契约，不出现任何项目标识符、不引用代码，80–250 词。描述只生成一次（LLM 层按请求内容缓存），
四个混淆级共用——所以它必须是「名字无关」的，否则 L0 的真名会漏进 L1–L3 的 prompt，抵消符号混淆。

流程：

1. 提示词里除了函数源码和依赖声明，还列出这些材料里出现的项目标识符（``leaks`` 扫出来的），明令禁止出现；
2. 拿到回复后用 ``leaks`` 检查，并检查长度 / 是否为空 / 是否带代码块；
3. 有问题就把「上一次的回复 + 具体问题（泄漏清单、字数）」作为多轮对话的后续再问一次，
   最多重试 ``MAX_LEAK_RETRIES``（=3）次，即最多 4 次调用；
4. 重试用尽后仍有标识符泄漏（或回复为空）→ ``describe`` 抛 ``DescribeFailed``，调用方把该目标标记为
   ``describe_failed`` 并排除；只要有过一次回复没有泄漏、只是长度不合格，就接受其中字数离合格区间最近的那条并记警告
   （规格要求里长度不是排除条件，泄漏才是）。

``describe`` 对 llm 的要求是设计文档 §4.7 的 ``chat(messages, temperature, max_tokens, seed)``；``chat`` 若还接受 ``meta``
（OpenAICompat 用它写 usage 日志，MockLLM 靠 purpose 返回固定的合格描述），调用方就传
``{"purpose": "describe", "target_id": ..., "attempt": 第几次}``，不接受就不传（见 ``llm.chat_with_meta``）。

需要把 model / leak_retries / token 用量写进 ``results/descriptions.jsonl`` 的调用方用 ``describe_detailed``，
它返回 ``DescribeResult`` 而不是抛异常。

``leaks`` 的口径（:class:`LeakChecker`）：

* 参与比对的名字：长度 ≥ 3，且不是常见英文词（``COMMON_WORDS`` 加简单的词形还原，再加 ``spec.allow_words``）——
  ``item``、``value``、``buffer`` 这类词既是 C 标识符又是描述里的正常用语，不能算泄漏。**这是有意的放行**：
  放行命中（描述里出现了这些词，包括目标函数自己的名字）不拦截，但如实记进描述记录的 ``exempt_hits`` / ``own_name_hit``，
  报告里统计，审计时可以核对；
* 精确匹配：把描述切成标识符记号（``[A-Za-z_][A-Za-z0-9_]*``，左边不能紧挨着字母数字下划线），区分大小写地比对；
* 规范化匹配（防改写规避）：记号（含用连字符连起来的）去掉 ``_`` 和 ``-``、统一大小写后再比，规范化后长度 ≥ 5 才算，
  所以 ``CJSON_PARSE`` / ``cjson_parse`` / ``the cjson library`` / ``buffer-skip-whitespace`` / ``bufferSkipWhitespace``
  都能认出来；
* 词组警告：多词名字（``buffer_skip_whitespace``）的各个词在描述里**以空格相邻**出现，只记警告（``phrase_hits``），不拦截，
  因为自然语言里常常需要这样描述功能。
"""
from __future__ import annotations

import functools
import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from bench.extract import FunctionTarget
from bench.llm import LLM, chat_with_meta
from bench.prompt import format_context

log = logging.getLogger(__name__)

__all__ = [
    "DESCRIBE_PROMPT_VERSION",
    "DESCRIBE_PROMPT_DIGEST",
    "DESCRIBE_SYSTEM",
    "DESCRIBE_USER_TEMPLATE",
    "MIN_WORDS",
    "MAX_WORDS",
    "MAX_LEAK_RETRIES",
    "MIN_NAME_LEN",
    "MIN_NORM_LEN",
    "COMMON_WORDS",
    "LeakChecker",
    "DescribeResult",
    "DescribeFailed",
    "leaks",
    "word_count",
    "clean_description",
    "build_describe_messages",
    "describe_detailed",
    "describe",
]

DESCRIBE_PROMPT_VERSION = "desc-v1"

MIN_WORDS = 80
MAX_WORDS = 250
MAX_LEAK_RETRIES = 3          # 重新生成的最多次数（不含第一次），所以最多调用 1 + 3 = 4 次
MIN_NAME_LEN = 3              # 短于此长度的标识符（i、n、fn……）不参与泄漏检查
MIN_NORM_LEN = 5              # 规范化匹配（忽略大小写和 _ -）只对规范化后不短于此长度的名字生效
DESCRIBE_MAX_TOKENS = 4096    # 思考模式下思维链也计入，用推理模型做描述时调用方应调大
MAX_LISTED_NAMES = 80         # 提示词里最多列出多少个禁用标识符
MAX_CONTEXT_CHARS = 8000      # 给描述模型看的依赖声明的字符上限

# --------------------------------------------------------------------------- 常见英文词表
# 这些词既是常见的 C 标识符，也是写功能规格时的正常用语，出现在描述里不算泄漏。
# 词表只收小写基本形；复数 / 过去式 / 进行式由 _is_common 做简单还原，不必逐个列。
# 也收了 C 关键字、标准类型名和常见缩写（len、buf、ptr……）：它们不是能帮助模型「回忆」出项目的名字。
_COMMON_WORDS_TEXT = """
the and for are but not you all any can had her was one our out his has its let may say she too use who why how did get got put set
now off per via yes nor yet this that with from they will have been were what when your than then them these those there their which
would could should shall might must does done here each every both such same very just only also even more most much many some other
another into onto upon over under about above below after before between during through while until since because although though
unless whether either neither within without around against among along across beyond inside outside toward towards

able abort about abs absolute accept access account accumulate action active actual add added addition additional address adjust
advance algorithm align alignment alive allocate allocated allocation allow allowed alpha alphabet alter alternate always amount
analysis ancestor angle answer append apply approach arbitrary area arg args argc argv argument arguments arithmetic array arrays
ascending ascii assert assign assigned assignment associate associated assume assumption atom attempt attribute attributes auto
automatic available average avoid back backward bad balance bar base basic batch begin beginning behavior behaviour best beta big
binary bind bit bits bitmap bitwise black blank block blocks body bool boolean border bottom bound boundary bounds box branch break
bridge buf buffer buffers bug build builder built bulk busy button byte bytes

cache calc calculate calculation call callback called caller calling cancel candidate capacity capital capture carry case cases cast
catch cell center chain change changed changes channel char character characters charset check checked checksum child children choice
choose chunk circle class clean clear clip clock clone close closed closing cluster code collect collection color colour column
columns combine combined comma command comment comments commit common compact compare comparison compile complete completed complex
component compose compress compute concat concatenate condition conditions config configuration confirm conflict connect connection
consecutive consider const constant constants construct constructor consume contain contained container contains content contents
context continue control convert copy core corner correct count counter counts cover create created creation current currently
cursor custom cut cycle

data database date debug decimal declare declaration decode decrement default define defined definition delay delete deleted
delimiter depth descend descending describe description destination destroy detail details detect determine device dict dictionary
diff difference different digit digits direct direction directory disable discard display distance divide division divisor document
dot double down drop dump duplicate dynamic

each edge element elements else empty enable encode encoding end ending endian enough ensure entire entries entry enum environment
equal equals equivalent error errors escape evaluate evaluation event exact example exception exclusive exec execute exist exists
exit expand expect expected explicit exponent export expression extend extension extern external extra extract

fail failed failure false fast fatal feature fetch field fields file filename files fill filter final find finish first fix fixed
flag flags float floating flush folder follow following force form format formatted forward found fraction frame free freed front
full func function functions

general generate generic get getter given global goto grow guard

handle handled handler hash header heap height help helper hex hexadecimal hidden high hold hook hooks host

identifier ignore implementation implicit include increment index indexes indices info inherit init initial initialize inline inner
input insert instance int integer interface internal interpret interval invalid invert item items iterate iteration iterator

join json jump keep key keys keyword kind

label language large largest last later layout lazy leading left len length less level lexer library limit line lines link list
literal little load local locale location lock log logic logical long look lookup loop low lower lowercase

macro magic main make malloc manage many map mark mask match matched matches matching max maximum mean member members memory merge
message method min minimum minus miss missing mode modify module more move multi multiple multiply must

name names native need negate negative nest nested new newline next node nodes non none normal normalize note nothing notify nul null
number numbers numeric

object objects octal offset old only opaque open operand operation operator option optional options order origin original other
out output outside overflow overwrite own owner

pack pad padding page pair param parameter parameters parent parse parsed parser partial pass passed past path pattern pause peek
percent perform period pick piece pipe plain plus point pointer pointers pool pop port pos position positive possible post power
precision prefix prepare present preserve prev previous primary print printf private process produce program prompt property
protect provide public push put

query queue quote quotes raise random range rank raw read reader ready real reason receive record recover recursion recursive reduce
ref reference region register relative release remain remainder remaining remove rename repeat replace report represent
representation request require reserve reset resize resolve resource response rest restore result results resume retain retry
return returned returns reverse rewind right root round route row rule run runtime

safe same sample save scale scan scanner scope search second section seed seek segment select send sentinel separator sequence
serialize server set setter setup shift short show sign signal signature signed simple single size skip slash slice slot small
smallest socket sort source space spaces special specific specification specifier split stack standard start starting startup state
statement static status step stop storage store str stream strict string strings struct structure sub subtract success suffix sum
support swap switch symbol syntax system

tab table tag tail take target temp template temporary term terminal terminate terminator test text then thread throw time
timeout token tokens tolerance top total trace track trailing transform translate traverse tree trim true truncate try tuple type
types typedef

unary undefined underscore unicode union unique unit unknown unlock unsigned until update upper uppercase usage use used user usual
utf utf8 util utility

valid validate validation value values var variable variables variant vector verify version void volatile wait walk warn warning
whitespace width wildcard window word words work wrap wrapper write writer yield zero

anything something nothing everything anyone someone everyone never ever often sometimes usually already again once twice cause
purpose self thing things part parts side sides way ways place places person give gives gave keep kept leave leaves lose lost mean
means meant meet meets notes okay reach reaches refer refers related rely relies remember seem seems sent serve share shared similar
slow speed spend spot stay stick suggest supply sure tell tends think thus together took touch treat turn turns understand unused
various view want wants went whole whose wide wish worth wrong year bigger biggest smaller larger lowest highest second third fourth
fifth three four five six seven eight nine ten hundred thousand million recent recently earlier

malloc calloc realloc memcpy memmove memset memcmp memchr strlen strcmp strncmp strcasecmp strcpy strncpy strcat strncat strchr
strrchr strstr strdup strtol strtoul strtod strtof atoi atol atof sprintf snprintf fprintf vprintf vsnprintf sscanf scanf fopen
fclose fread fwrite fgets fputs fputc fgetc fseek ftell fflush puts putchar getchar isalpha isdigit isalnum isspace isupper islower
isxdigit isprint ispunct iscntrl tolower toupper labs fabs floor ceil sqrt pow exp sin cos tan atan fmod errno stdin stdout stderr
eof size_t ssize_t ptrdiff_t intptr_t uintptr_t uint8_t uint16_t uint32_t uint64_t int8_t int16_t int32_t int64_t va_list va_start
va_end va_arg

break continue default double extern float inline long restrict short signed sizeof typedef while
"""

COMMON_WORDS: frozenset[str] = frozenset(_COMMON_WORDS_TEXT.split())

_SUFFIX_RULES = (("ies", "y"), ("es", ""), ("s", ""), ("ed", ""), ("ed", "e"), ("ing", ""), ("ing", "e"))


def _is_common(word: str) -> bool:
    """word（不分大小写）是不是常见英文词：在词表里，或者去掉 s / es / ed / ing 等词尾后在词表里。"""
    w = word.lower()
    if w in COMMON_WORDS:
        return True
    for suffix, repl in _SUFFIX_RULES:
        if w.endswith(suffix) and len(w) - len(suffix) >= 3 and w[: -len(suffix)] + repl in COMMON_WORDS:
            return True
    return False


# --------------------------------------------------------------------------- 泄漏检查

#: 描述里的「词链」：标识符，以及用连字符连起来的一串（``buffer-skip-whitespace``）
_CHAIN_RE = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z_][A-Za-z0-9_]*(?:-[A-Za-z0-9_]+)*")
#: 把名字 / 描述切成词：驼峰、下划线、连字符、数字边界
_PART_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|[0-9]+")
_SEP_RE = re.compile(r"[_\-]")
_PHRASE_MAX = 5               # 词组警告只看最多这么多个词组成的名字


def _norm(text: str) -> str:
    """规范化：去掉 ``_`` 和 ``-``，统一大小写。"""
    return _SEP_RE.sub("", text).casefold()


def _parts(name: str) -> tuple[str, ...]:
    return tuple(w.casefold() for w in _PART_RE.findall(name))


class LeakChecker:
    """项目标识符泄漏检查器：对一组项目标识符预处理一次，反复检查描述。口径见模块文档。

    * ``leaks(text)``：硬泄漏（精确匹配 + 规范化匹配），命中就要重新生成；
    * ``phrase_hits(text)``：多词名字的各个词以空格相邻出现，只是警告；
    * ``exempt_hits(text)``：放行名字（常见英文词 / spec.allow_words）在描述里出现了哪些，只是记录，不拦截；
    * ``digest()``：检查器配置的摘要（词表、词形规则、长度阈值、allow_words、名字集合），进描述阶段的输入哈希，
      收紧检查器后旧描述会被重新判定。
    """

    def __init__(self, project_names: Iterable[str], allow_words: Iterable[str] = ()) -> None:
        self.allow_words = frozenset(w.casefold() for w in allow_words)
        names = sorted({n for n in project_names if len(n) >= MIN_NAME_LEN})
        self.names = tuple(names)
        self.exempt = frozenset(n for n in names if self._is_exempt(n))
        self.exact = frozenset(n for n in names if n not in self.exempt)
        by_norm: dict[str, list[str]] = {}
        phrases: dict[tuple[str, ...], list[str]] = {}
        for n in sorted(self.exact):
            k = _norm(n)
            if len(k) >= MIN_NORM_LEN:
                by_norm.setdefault(k, []).append(n)
            parts = _parts(n)
            if 2 <= len(parts) <= _PHRASE_MAX:
                phrases.setdefault(parts, []).append(n)
        self._by_norm = by_norm
        self._phrases = phrases
        self._phrase_lens = sorted({len(k) for k in phrases})

    def _is_exempt(self, name: str) -> bool:
        return _is_common(name) or name.casefold() in self.allow_words

    def leaks(self, text: str) -> list[str]:
        """描述里出现的项目标识符，去重、按首次出现的顺序排列。"""
        if not text or not self.exact:
            return []
        found: list[str] = []
        seen: set[str] = set()

        def add(name: str) -> None:
            if name not in seen:
                seen.add(name)
                found.append(name)

        for m in _CHAIN_RE.finditer(text):
            tok = m.group(0)
            for cand in (tok, *tok.split("-")):
                if cand in self.exact:
                    add(cand)
            key = _norm(tok)
            if len(key) >= MIN_NORM_LEN:
                for name in self._by_norm.get(key, ()):
                    add(name)
        return found

    def phrase_hits(self, text: str) -> list[str]:
        """多词名字的各个词以空格相邻出现（``skip whitespace`` 对 ``skip_whitespace``）：只是警告，去重、按名字排序。"""
        if not text or not self._phrases:
            return []
        words = [w.casefold() for w in _PART_RE.findall(text)]
        hits: set[str] = set()
        for n in self._phrase_lens:
            for i in range(len(words) - n + 1):
                for name in self._phrases.get(tuple(words[i:i + n]), ()):
                    hits.add(name)
        return sorted(hits)

    def exempt_hits(self, text: str) -> list[str]:
        """描述里出现的放行名字（精确匹配，区分大小写），去重、按首次出现的顺序。"""
        if not text or not self.exempt:
            return []
        found: list[str] = []
        for m in _CHAIN_RE.finditer(text):
            tok = m.group(0)
            if tok in self.exempt and tok not in found:
                found.append(tok)
        return found

    def digest(self) -> str:
        return _checker_digest(self.names, tuple(sorted(self.allow_words)))


@functools.lru_cache(maxsize=1)
def _static_config_digest() -> str:
    cfg = {"common": sorted(COMMON_WORDS), "suffix": _SUFFIX_RULES, "min_name": MIN_NAME_LEN, "min_norm": MIN_NORM_LEN,
           "phrase_max": _PHRASE_MAX}
    return hashlib.sha256(repr(cfg).encode("utf-8")).hexdigest()


def _checker_digest(names: tuple[str, ...], allow: tuple[str, ...]) -> str:
    blob = repr((_static_config_digest(), names, allow))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


@functools.lru_cache(maxsize=8)
def _checker(names: frozenset[str], allow: frozenset[str]) -> LeakChecker:
    return LeakChecker(names, allow)


def leaks(description: str, project_names: set[str], allow_words: Iterable[str] = ()) -> list[str]:
    """描述里出现的项目标识符，去重、按首次出现的顺序排列；口径见模块文档。"""
    if not description or not project_names:
        return []
    return _checker(frozenset(project_names), frozenset(allow_words)).leaks(description)


_WORD_RE = re.compile(r"[A-Za-z0-9_]+(?:['\-][A-Za-z0-9_]+)*")


def word_count(text: str) -> int:
    """英文词数：词内的连字符 / 撇号不拆开（well-known、don't 各算一个），列表符号 ``-`` 之类的孤立标点不算词。"""
    return len(_WORD_RE.findall(text))


_TITLE_RE = re.compile(r"^\W*(?:function |behavioral |behavioural )?specification\W*$", re.I)


def clean_description(text: str) -> str:
    """去掉围栏行、反引号和单独成行的「Specification:」标题；规整空白。其余内容不动。"""
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln for ln in text.split("\n") if not ln.lstrip().startswith(("```", "~~~"))]
    if lines and _TITLE_RE.match(lines[0].strip()):
        lines = lines[1:]
    text = "\n".join(ln.rstrip() for ln in lines).replace("`", "")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


# --------------------------------------------------------------------------- 提示词

DESCRIBE_SYSTEM = (
    "You write behavioral specifications of C functions.\n"
    "\n"
    "Your reader is a programmer who will re-implement the function from your text alone. The reader sees only the "
    "function's signature and the declarations of the types, macros, globals and helper functions it may use, under "
    "unfamiliar names. The reader never sees the original code, and the re-implementation will be checked against the "
    "project's test suite, so be exact about every observable behavior.\n"
    "\n"
    "Write the specification in English as plain prose (short paragraphs or a compact list), 80 to 250 words. "
    "Cover, as far as they apply:\n"
    "- the purpose of the function in one sentence;\n"
    "- what each argument means, referring to arguments only by position (\"the first argument\", \"the second "
    "argument\", ...), and what the function assumes about each one (not NULL, NUL-terminated, valid range, ownership);\n"
    "- the return value: its meaning and every distinct outcome, with the exact literal values callers can observe;\n"
    "- boundary conditions and special inputs (empty, zero length, NULL, maximum values, overlapping data);\n"
    "- error handling: which inputs are rejected, what is returned or recorded, which condition is reported first when "
    "several apply, and whether partial results are left behind;\n"
    "- side effects: memory allocated or released, objects or globals modified (quote any literal strings or numeric "
    "constants that get stored), and the order of modifications when it matters;\n"
    "- ownership and lifetime of anything returned or stored.\n"
    "\n"
    "Hard rules:\n"
    "1. Never write any identifier from the project: no function names, variable or parameter names, type names, "
    "struct or union member names, enum constants, macro names or labels. Refer to such things by their role or "
    "position instead, for example \"the structure pointed to by the first argument\", \"its second member\", "
    "\"the helper that duplicates a string\", \"the error value for out-of-memory\". Names of C standard library "
    "functions, C keywords, standard type names such as size_t, and the constant NULL are allowed.\n"
    "2. Do not quote code and do not narrate the code line by line: state what the function guarantees, not how it "
    "is written. Do not use code blocks or backticks.\n"
    "3. Do not add a title, a preface or closing remarks. Output only the specification."
)

DESCRIBE_USER_TEMPLATE = (
    "Write the behavioral specification of the following C function.\n"
    "\n"
    "Function:\n"
    "```c\n"
    "{code}\n"
    "```\n"
    "{doc_block}"
    "Declarations that the function relies on (context for you only; do not mention any name from it):\n"
    "```c\n"
    "{context}\n"
    "```\n"
    "{forbidden_block}"
    "Remember: plain English prose of 80 to 250 words, arguments referred to by position, "
    "no project identifiers, no code."
)

_DOC_BLOCK = "Existing comment on the function (may be incomplete; trust the code):\n{doc}\n\n"
_FORBIDDEN_BLOCK = "These project identifiers occur in the material above and must not appear in your text: {names}.\n\n"

_FEEDBACK_LEAK = (
    "Your specification mentions project identifiers: {names}. The specification must not contain these or any other "
    "project identifier. Rewrite the complete specification without them, describing those entities by role or by "
    "position instead."
)
_FEEDBACK_LENGTH = "Your specification has {n} words; it must be between {lo} and {hi} words."
_FEEDBACK_EMPTY = "Your reply contained no specification text. Write the specification now."
_FEEDBACK_TAIL = " Keep it between {lo} and {hi} words, as plain prose without code blocks or backticks."


DESCRIBE_PROMPT_DIGEST = hashlib.sha256(
    "\x1f".join(
        [
            DESCRIBE_PROMPT_VERSION, DESCRIBE_SYSTEM, DESCRIBE_USER_TEMPLATE, _DOC_BLOCK, _FORBIDDEN_BLOCK, _FEEDBACK_LEAK,
            _FEEDBACK_LENGTH, _FEEDBACK_EMPTY, _FEEDBACK_TAIL, str((MIN_WORDS, MAX_WORDS, MAX_LEAK_RETRIES, MAX_LISTED_NAMES)),
        ]
    ).encode("utf-8")
).hexdigest()


def _deps_text(target: FunctionTarget) -> str:
    text = format_context(target.deps)
    if len(text) > MAX_CONTEXT_CHARS:
        text = text[:MAX_CONTEXT_CHARS].rsplit("\n", 1)[0] + "\n/* ... (truncated) */"
    return text or "/* (none) */"


def build_describe_messages(
    target: FunctionTarget, code: str, forbidden: Sequence[str] = (), *, context_text: str | None = None
) -> list[dict]:
    """描述模型的首轮 messages：[system, user]。forbidden 是材料里出现、必须避开的项目标识符；
    context_text 是已经渲染好的依赖声明（describe_detailed 要先用它算 forbidden，传进来省得再渲染一遍）。"""
    doc = (target.doc_comment or "").strip()
    user = DESCRIBE_USER_TEMPLATE.format(
        code=code.strip("\n"),
        doc_block=_DOC_BLOCK.format(doc=doc) if doc else "",
        context=context_text if context_text is not None else _deps_text(target),
        forbidden_block=_FORBIDDEN_BLOCK.format(names=", ".join(forbidden[:MAX_LISTED_NAMES])) if forbidden else "",
    )
    return [{"role": "system", "content": DESCRIBE_SYSTEM}, {"role": "user", "content": user}]


# --------------------------------------------------------------------------- 主流程


@dataclass
class DescribeResult:
    target_id: str
    description: str                 # 最后一次回复清理后的文本；failed 时仅供排查
    model: str
    attempts: int                    # 实际调用 LLM 的次数
    leak_retries: int                # 因标识符泄漏而重新生成的次数（写进 descriptions.jsonl）
    failed: bool
    reason: str = ""                 # failed 时："leak"（重试用尽仍泄漏）或 "empty"（回复为空）
    leaked: list[str] = field(default_factory=list)   # 最后一次回复里仍然泄漏的标识符；成功时为空
    n_words: int = 0
    length_ok: bool = True
    prompt_tokens: int = 0           # 各次调用累计（命中缓存的调用也计入，反映这份描述需要的量）
    completion_tokens: int = 0
    phrase_hits: list[str] = field(default_factory=list)   # 多词名字以空格相邻出现（只是警告）
    exempt_hits: list[str] = field(default_factory=list)   # 描述里出现的放行名字（常见英文词 / allow_words，只是记录）
    own_name_hit: bool = False       # 目标函数自己的名字落在放行集合里、且出现在了描述里
    checker: str = ""                # 检查器配置摘要（LeakChecker.digest）


class DescribeFailed(RuntimeError):
    """重试用尽后描述仍然泄漏项目标识符（或为空）：调用方应把该目标标记为 describe_failed 并排除。"""

    def __init__(self, result: DescribeResult) -> None:
        detail = ("仍泄漏: " + ", ".join(result.leaked[:10])) if result.leaked else "回复为空"
        super().__init__(f"{result.target_id} 的功能规格生成失败（{result.reason}，共调用 {result.attempts} 次）: {detail}")
        self.result = result


def _feedback(leaked: Sequence[str], n_words: int, empty: bool) -> str:
    parts: list[str] = []
    if empty:
        parts.append(_FEEDBACK_EMPTY)
    if leaked:
        parts.append(_FEEDBACK_LEAK.format(names=", ".join(leaked[:MAX_LISTED_NAMES])))
    if not empty and not MIN_WORDS <= n_words <= MAX_WORDS:
        parts.append(_FEEDBACK_LENGTH.format(n=n_words, lo=MIN_WORDS, hi=MAX_WORDS))
    return " ".join(parts) + _FEEDBACK_TAIL.format(lo=MIN_WORDS, hi=MAX_WORDS)


def describe_detailed(
    target: FunctionTarget,
    code: str,
    llm: LLM,
    project_names: set[str],
    *,
    allow_words: Iterable[str] = (),
    max_retries: int = MAX_LEAK_RETRIES,
    max_tokens: int = DESCRIBE_MAX_TOKENS,
) -> DescribeResult:
    """生成并检查功能规格；失败也返回 DescribeResult（failed=True），不抛 DescribeFailed。LLM 自身的错误照常向上抛。"""
    checker = _checker(frozenset(project_names), frozenset(allow_words))   # 只预处理一次，后面每次检查都复用
    result = _describe_loop(target, code, llm, checker, max_retries, max_tokens)
    result.phrase_hits = checker.phrase_hits(result.description)
    result.exempt_hits = checker.exempt_hits(result.description)
    result.own_name_hit = target.name in result.exempt_hits
    result.checker = checker.digest()
    if result.phrase_hits:
        log.info("%s 的描述里有多词名字的词组: %s", target.target_id, ", ".join(result.phrase_hits[:5]))
    if result.own_name_hit:
        log.warning("%s 的描述里出现了目标函数自己的名字 %s（常见英文词，被放行）", target.target_id, target.name)
    return result


def _describe_loop(target: FunctionTarget, code: str, llm: LLM, checker: LeakChecker, max_retries: int,
                   max_tokens: int) -> DescribeResult:
    context_text = _deps_text(target)
    material = "\n".join([code, context_text, target.doc_comment or ""])
    messages = build_describe_messages(target, code, checker.leaks(material), context_text=context_text)
    result = DescribeResult(target.target_id, "", getattr(llm, "model", ""), 0, 0, failed=True)
    best: tuple[int, str] | None = None   # 没有泄漏、非空但长度不合格的回复里，字数离合格区间最近的一条
    for attempt in range(max_retries + 1):
        reply = chat_with_meta(
            llm,
            messages,
            temperature=0.0,
            max_tokens=max_tokens,
            meta={"purpose": "describe", "target_id": target.target_id, "attempt": attempt},
        )
        result.attempts += 1
        result.prompt_tokens += reply.prompt_tokens
        result.completion_tokens += reply.completion_tokens
        text = clean_description(reply.text)
        leaked = checker.leaks(text)
        n = word_count(text)
        result.description, result.leaked, result.n_words = text, leaked, n
        result.length_ok = MIN_WORDS <= n <= MAX_WORDS
        if text and not leaked and result.length_ok:
            result.failed, result.reason = False, ""
            return result
        if text and not leaked:
            gap = max(MIN_WORDS - n, n - MAX_WORDS)
            if best is None or gap < best[0]:
                best = (gap, text)
        if attempt == max_retries:
            break
        if leaked:
            result.leak_retries += 1
        messages = messages + [
            {"role": "assistant", "content": reply.text or "(empty reply)"},
            {"role": "user", "content": _feedback(leaked, n, empty=not text)},
        ]
        log.info("%s 的描述第 %d 次不合格（泄漏 %s，%d 词），重新生成", target.target_id, attempt + 1, leaked[:5], n)
    # 重试用尽：只有长度不合格而没有泄漏的描述照样可用（取离合格区间最近的那条，哪怕不是最后一次的回复）
    if best is not None:
        result.description, result.leaked = best[1], []
        result.n_words, result.length_ok = word_count(best[1]), False
        result.failed, result.reason = False, ""
        log.warning("%s 的描述长度 %d 词不在 [%d, %d]，但没有泄漏，接受", target.target_id, result.n_words, MIN_WORDS, MAX_WORDS)
    else:
        result.reason = "leak" if result.leaked else "empty"
    return result


def describe(target: FunctionTarget, code: str, llm: LLM, project_names: set[str],
             allow_words: Iterable[str] = ()) -> str:
    """返回目标函数的功能规格；重试用尽仍泄漏标识符时抛 DescribeFailed（调用方标记 describe_failed 并排除）。"""
    result = describe_detailed(target, code, llm, project_names, allow_words=allow_words)
    if result.failed:
        raise DescribeFailed(result)
    return result.description
