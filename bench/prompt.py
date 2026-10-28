"""生成阶段的 prompt（设计文档 §4.9），以及缺陷修复任务的 prompt（docs/设计-缺陷修复任务.md §5）。

``build_messages(task)`` 是纯函数：同一个 Task 永远得到同样的 messages。四个混淆级共用同一套模板，
只有内容（描述之外的签名、依赖声明里的名字）不同。模板文本都是本模块的常量，改动任何一处都要同时
升 ``PROMPT_VERSION``；``PROMPT_DIGEST`` 是全部模板文本的哈希，结果文件带上它，report 时可以校验
「prompt 模板没被悄悄改过」（设计文档 §6.6）。

messages 的结构：

* system：C 程序员；只输出一个 ```c 代码块，内含且仅含目标函数的完整定义；不改签名、不加 #include、不写解释。
* user：功能规格 → 项目上下文（已包含的头文件、宏、类型、全局变量、可调用函数的原型）→ 要实现的函数签名 → 约束
  （C 标准、禁止加头文件、禁止调用的函数）。

``format_context`` 同时被 describe（给描述模型看上下文）和 postprocess（给 libclang 当解析前缀）复用，
所以输出顺序固定为 includes、macros、types、globals、decls：宏是纯文本放前面，类型按依赖顺序，原型放最后。

缺陷修复任务另有一套模板，``build_repair_messages(RepairTask)`` 同样是纯函数：

* system：返回修正后的完整函数，不改签名、不加 #include、不写解释；
* user：依赖上下文（``## Project context`` 小节与生成任务由同一段文本 ``CONTEXT_BLOCK`` 拼出，同样走 ``format_context``）→
  ``Buggy function:`` 加该级副本里注入缺陷后的完整函数 → 约束行（直接复用 ``CONSTRAINT_LINES``）。没有功能规格，不给测试输出，
  不透露缺陷的位置与算子。

修复任务的依赖上下文取自**注入缺陷之后**的函数（:func:`prune_context`）：上下文原本来自原函数，被删掉 / 换掉的那部分代码用到的原型、宏、
全局还留在里面，就等于把答案线索写进了提示词（``plan_repair_tasks`` 调它；被裁掉的名字记在 ``RepairTask.context_orphans``）。

修复模板的常量与 ``PROMPT_VERSION`` / ``PROMPT_DIGEST`` 分开命名（``REPAIR_PROMPT_VERSION`` / ``REPAIR_PROMPT_DIGEST``）：
改动修复模板的任何一处都要同时升 ``REPAIR_PROMPT_VERSION``，``PROMPT_DIGEST`` 不受影响。约束行与上下文各段的标题 / 分隔符
是两套 prompt 共用的文本，改了它们，两个摘要一起变。
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

from bench.extract import Dependencies, strip_c_comments
from bench.similarity import identifiers, lex

__all__ = [
    "PROMPT_VERSION",
    "PROMPT_DIGEST",
    "SYSTEM_PROMPT",
    "USER_TEMPLATE",
    "CONSTRAINT_LINES",
    "CONTEXT_SECTIONS",
    "FORBIDDEN_CALLS",
    "Task",
    "format_context",
    "build_messages",
    "CONTEXT_BLOCK",
    "prune_context",
    "REPAIR_PROMPT_VERSION",
    "REPAIR_PROMPT_DIGEST",
    "REPAIR_SYSTEM_PROMPT",
    "REPAIR_USER_TEMPLATE",
    "RepairTask",
    "build_repair_messages",
]

PROMPT_VERSION = "gen-v1"

# 约束里禁止调用的函数：进程控制、终止进程、信号、跳转。这份清单是提示词和后处理共用的唯一来源：
# postprocess.normalize 会按它扫描候选代码（``exec*`` 展开成整个 exec 系列），命中就记 policy_violation，
# 不进沙箱、按失败计。否则候选只写 ``exit(0);`` 就能让只看退出码的语料误判为通过。
# 语料九个库的目标函数都不用它们（tests/test_corpus.py 里有逐语料的核实）。
FORBIDDEN_CALLS = ("system", "popen", "fork", "vfork", "exec*", "posix_spawn", "abort", "exit", "_exit", "_Exit",
                   "quick_exit", "atexit", "signal", "sigaction", "raise", "kill", "longjmp")

SYSTEM_PROMPT = (
    "You are an experienced C programmer. You implement exactly one function inside an existing C project, "
    "strictly following the functional specification and the declarations you are given.\n"
    "Reply with exactly one fenced code block (```c ... ```) that contains the complete definition of that "
    "function and nothing else. Do not change the signature, do not add #include lines, and do not write any "
    "explanation or any text outside the code block."
)

# 约束清单；{language_std} 与 {name} 在 build_messages 里代入
CONSTRAINT_LINES = (
    "Language standard: {language_std}. The code is compiled with gcc -std={language_std}, so it must be valid for that standard.",
    "Output exactly one ```c code block that contains only the complete definition of {name}; "
    "do not define any other function, type, macro or global variable.",
    "Keep the signature exactly as given: the same name, the same parameter types in the same order, "
    "the same return type and the same storage class.",
    "Do not write any #include line. The headers that are already included are listed in the context; "
    "the C library functions they declare are available.",
    "Call only functions that are declared in the context or provided by those headers; "
    "do not rely on implicit declarations.",
    "Do not call " + ", ".join(FORBIDDEN_CALLS) + ", and do not use networking or inline assembly.",
    "Write no explanation and no text outside the code block.",
)

# 生成任务与修复任务的 user 消息里共用的「项目上下文」小节：两套模板都由它拼出，不靠复制保持一致
CONTEXT_BLOCK = (
    "## Project context\n"
    "These declarations are already in scope where the function will be placed. Do not repeat them.\n"
    "```c\n"
    "{context}\n"
    "```\n"
)

USER_TEMPLATE = (
    "Implement the following C function for an existing project.\n"
    "\n"
    "## Functional specification\n"
    "{description}\n"
    "\n"
    + CONTEXT_BLOCK +
    "\n"
    "## Function to implement\n"
    "Write the complete definition: this signature followed by the function body.\n"
    "```c\n"
    "{signature}\n"
    "```\n"
    "\n"
    "## Constraints\n"
    "{constraints}\n"
)

EMPTY_CONTEXT = "/* (no additional declarations) */"

# 上下文各段的 (标题, 条目之间的分隔符)，顺序即渲染顺序；段与段之间用 CONTEXT_JOINER 隔开。
# 这些文本直接出现在 prompt 里，所以和模板一起算进 PROMPT_DIGEST。
CONTEXT_SECTIONS = (
    ("Headers already included", "\n"),
    ("Macros", "\n"),
    ("Types", "\n\n"),
    ("Global variables", "\n"),
    ("Functions you may call", "\n"),
)
CONTEXT_JOINER = "\n\n"



def _digest(version: str, system: str, user: str) -> str:
    """一套 prompt 的全部模板文本的哈希：版本、两段模板、两套 prompt 共用的约束行、空上下文占位、上下文各段的标题与分隔符。
    生成任务与修复任务用同一个公式，各自传自己的版本号和模板。"""
    return hashlib.sha256(
        "\x1f".join(
            [version, system, user, *CONSTRAINT_LINES, EMPTY_CONTEXT, CONTEXT_JOINER,
             *(f"{title}\x1e{sep}" for title, sep in CONTEXT_SECTIONS)]
        ).encode("utf-8")
    ).hexdigest()


PROMPT_DIGEST = _digest(PROMPT_VERSION, SYSTEM_PROMPT, USER_TEMPLATE)


# --------------------------------------------------------------------------- 任务


@dataclass
class _TaskBase:
    """两类任务共有的前 7 个字段，以及 ``to_dict`` / ``from_dict``（子类各自补自己的字段与默认值）。"""

    task_id: str
    target_id: str
    corpus: str
    level: str                  # L0 / L1 / L2 / L3
    obf_name: str               # 该混淆级下目标函数的名字（L0 即原名）
    obf_signature: str          # 该混淆级下的签名（不含函数体、不带分号）
    context: Dependencies       # 该混淆级下的依赖声明（名字已按映射改写）

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]):
        """to_dict 的逆操作；多余的键忽略（记录里的 task / mutation 等字段），context 缺失的列表按空处理。"""
        fields = {k: v for k, v in d.items() if k in cls.__dataclass_fields__ and k != "context"}
        ctx = d.get("context") or {}
        fields["context"] = Dependencies(**{k: list(ctx.get(k, [])) for k in Dependencies.__dataclass_fields__})
        for key in ("includes", "context_orphans"):
            if key in fields:
                fields[key] = list(fields[key] or [])
        return cls(**fields)


@dataclass
class Task(_TaskBase):
    """一个（目标函数, 混淆级）的生成任务；字段与设计文档 §4.9 一致，tasks.jsonl 存它的全部字段。"""

    description: str            # 功能规格（四级共用）
    includes: list[str] = field(default_factory=list)   # 已包含的系统头 #include 行；为空时用 context.includes
    language_std: str = "c99"   # 来自 spec.cflags 的 -std，如 c89 / c99 / gnu99


# --------------------------------------------------------------------------- 渲染


def _dedupe(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for s in items:
        s = s.strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def format_context(
    ctx: Dependencies,
    *,
    includes: Sequence[str] | None = None,
    comments: bool = True,
) -> str:
    """把依赖闭包渲染成一段 C 文本，顺序固定：includes、macros、types、globals、decls。

    ``includes`` 不为空时代替 ``ctx.includes``（Task.includes 优先）；``comments=True`` 给每段加一行
    ``/* 标题 */``（给人和模型看），False 时只输出纯 C（给 libclang 当前缀）。空的段落整段省略。
    """
    items_by_title = {
        "Headers already included": _dedupe(includes if includes else ctx.includes),
        "Macros": _dedupe(ctx.macros),
        "Types": [t.strip("\n") for t in ctx.types if t and t.strip()],
        "Global variables": _dedupe(ctx.globals),
        "Functions you may call": _dedupe(ctx.decls),
    }
    chunks: list[str] = []
    for title, sep in CONTEXT_SECTIONS:
        items = items_by_title[title]
        if not items:
            continue
        body = sep.join(items)
        chunks.append(f"/* {title} */\n{body}" if comments else body)
    return CONTEXT_JOINER.join(chunks)


def _context_and_constraints(task: "Task | RepairTask") -> tuple[str, str]:
    """两类 user 消息共用的两段文本：项目上下文（没有内容时用占位）和约束清单（代入 C 标准与目标函数名）。"""
    context = format_context(task.context, includes=task.includes or None) or EMPTY_CONTEXT
    constraints = "\n".join(
        "- " + line.format(language_std=task.language_std, name=task.obf_name) for line in CONSTRAINT_LINES
    )
    return context, constraints


def build_messages(task: Task) -> list[dict]:
    """生成阶段发给模型的 messages：[system, user]。纯函数，不读任何外部状态。"""
    context, constraints = _context_and_constraints(task)
    user = USER_TEMPLATE.format(
        description=task.description.strip(),
        context=context,
        signature=task.obf_signature.strip().rstrip(";").rstrip(),
        constraints=constraints,
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


# --------------------------------------------------------------------------- 缺陷修复任务（docs/设计-缺陷修复任务.md §5）

REPAIR_PROMPT_VERSION = "repair-v2"

REPAIR_SYSTEM_PROMPT = (
    "You are an experienced C programmer. The function below contains a bug that makes the project's tests fail. "
    "Return the corrected function only, as one ```c code block containing the complete definition. "
    "Keep the signature. Do not add #include lines. No explanation."
)

# 约束行用上面的 CONSTRAINT_LINES（{language_std} 与 {name} 在 build_repair_messages 里代入）；
# ``## Project context`` 小节与生成任务共用 CONTEXT_BLOCK（tests/test_prompt_repair.py 锁住渲染结果）
REPAIR_USER_TEMPLATE = (
    "Fix the bug in the following C function of an existing project.\n"
    "\n"
    + CONTEXT_BLOCK +
    "\n"
    "Buggy function:\n"
    "```c\n"
    "{code}\n"
    "```\n"
    "\n"
    "## Constraints\n"
    "{constraints}\n"
)


def _repair_prompt_digest() -> str:
    """修复 prompt 的全部模板文本的哈希，口径同 ``PROMPT_DIGEST``（同一个 ``_digest``）。每次按模块里的当前值重算
    （``REPAIR_PROMPT_DIGEST`` 是 import 时算出的那一份），测试靠它确认「任何一处模板文本改动，摘要都会变」。"""
    return _digest(REPAIR_PROMPT_VERSION, REPAIR_SYSTEM_PROMPT, REPAIR_USER_TEMPLATE)


REPAIR_PROMPT_DIGEST = _repair_prompt_digest()


@dataclass
class RepairTask(_TaskBase):
    """一个（目标函数, 混淆级）的缺陷修复任务；tasks-repair.jsonl 存它的全部字段（外加变异信息，见附录 §7）。

    与 :class:`Task` 的区别：没有功能规格，多了 ``mutant_code``（该级副本里注入缺陷后的目标函数完整定义，prompt 里给模型看的
    就是它）、``operator``（注入缺陷用的变异算子，只记录，不进 prompt）和 ``context_orphans``（:func:`prune_context` 从上下文里裁掉的
    名字，只记录）。``context`` 是裁剪之后的依赖声明。``obf_name`` / ``obf_signature`` / ``context`` / ``includes`` 的含义同 Task，
    所以 ``postprocess.process_reply`` 吃 RepairTask 也行。
    """

    includes: list[str] = field(default_factory=list)   # 已包含的系统头 #include 行；为空时用 context.includes
    language_std: str = "c99"   # 来自 spec.cflags 的 -std，如 c89 / c99 / gnu99
    mutant_code: str = ""       # 注入缺陷后的完整函数定义（已去注释）；build_repair_messages 要求非空
    operator: str = ""          # 变异算子编号（ROR / OBO / ...），不进 prompt
    context_orphans: list[str] = field(default_factory=list)   # 被裁掉的原型 / 宏 / 全局的名字，不进 prompt


def build_repair_messages(task: RepairTask) -> list[dict]:
    """缺陷修复任务发给模型的 messages：[system, user]。纯函数，不读任何外部状态；四个混淆级共用同一套模板。

    ``task.mutant_code`` 为空（或全是空白）时抛 ValueError：没有被修的函数，这道题没有意义，不能悄悄发一个空代码块出去。
    函数文本里还带着注释时同样抛 ValueError：符号混淆只改标识符，L1 以上的注释里仍是项目原名，放进提示词会泄漏本该隐藏的名字
    （mutate 阶段已经去过注释；这里是最后一道关，旧产出里带注释的变异体要用 ``mutate --force`` 重做）。
    """
    code = (task.mutant_code or "").strip("\n").rstrip()
    if not code.strip():
        raise ValueError(f"RepairTask {task.task_id!r} 的 mutant_code 为空，无法出题")
    if strip_c_comments(code) != code:
        raise ValueError(f"RepairTask {task.task_id!r} 的 mutant_code 带着注释，会泄漏被混淆隐藏的名字；"
                         "变异体的函数文本必须先去注释（mutate 阶段会做，旧产出请用 mutate --force 重做）")
    context, constraints = _context_and_constraints(task)
    user = REPAIR_USER_TEMPLATE.format(context=context, code=code, constraints=constraints)
    return [
        {"role": "system", "content": REPAIR_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


# --------------------------------------------------------------------------- 依赖上下文的裁剪（缺陷修复任务）

_DEFINE_RE = re.compile(r"^\s*#\s*define\s+([A-Za-z_]\w*)")
#: 出现在声明开头、括号里是修饰而不是参数的关键字（取函数名时要跳过它们的括号）
_ATTR_WORDS = frozenset({"__attribute__", "__declspec", "_Alignas", "__asm__", "__asm", "asm"})


def _function_name(prototype: str) -> str | None:
    """函数原型里被声明的名字：第一个（不属于修饰括号的）``(`` 前面的标识符；``(*name)(...)`` 形式取括号里的名字。取不到返回 None。"""
    toks = lex(prototype)
    i = 0
    while i < len(toks):
        kind, tok = toks[i]
        if tok == "(":
            if i > 0 and toks[i - 1][0] == "ident" and toks[i - 1][1] in _ATTR_WORDS:
                depth = 0
                while i < len(toks):                       # 整个修饰括号跳过
                    depth += (toks[i][1] == "(") - (toks[i][1] == ")")
                    i += 1
                    if depth == 0:
                        break
                continue
            if i + 1 < len(toks) and toks[i + 1][1] == "*":
                return next((t for k, t in toks[i + 2:] if k == "ident"), None)
            return toks[i - 1][1] if i > 0 and toks[i - 1][0] == "ident" else None
        i += 1
    return None


def _variable_name(declaration: str) -> str | None:
    """``extern`` 声明里的变量名：数组长度 / 初始化器 / 分号之前的最后一个标识符；``(*name)`` 形式的函数指针取括号里的名字。"""
    toks = lex(declaration)
    for i, (_, tok) in enumerate(toks[:-1]):
        if tok == "(" and toks[i + 1][1] == "*":
            return next((t for k, t in toks[i + 2:] if k == "ident"), None)
    cut = next((i for i, (_, tok) in enumerate(toks) if tok in ("[", "=", ";")), len(toks))
    return next((t for k, t in reversed(toks[:cut]) if k == "ident"), None)


def prune_context(ctx: Dependencies, code: str) -> tuple[Dependencies, list[str]]:
    """裁掉依赖上下文里只有「已经被删掉 / 换掉的那部分代码」才用得到的原型、宏、全局，返回（裁剪后的上下文, 被裁掉的名字）。

    ``ctx`` 来自**原**函数（``extract.locate`` 按原函数算的依赖闭包），缺陷删掉的调用、宏、全局还留在里面，就等于告诉模型该调谁、
    该用什么：例如 ``return mpc_expect(a, e);`` 被换成 ``return NULL;`` 之后，上下文里仍列着 ``mpc_expect`` 的原型。
    这里按注入缺陷后的函数 ``code`` 重新做可达性：从 ``code`` 里的标识符和（不裁剪的）类型定义出发，名字出现过的原型 / 宏 / 全局保留，
    保留下来的条目文本里出现的名字继续往下找，直到不再增加；没被找到的就是被裁掉的。只裁 ``decls`` / ``macros`` / ``globals``：
    ``includes`` 与 ``types`` 原样保留（类型定义不裁，所以只被裁掉的原型用到的类型会留着，线索很弱）；取不到名字的条目也保留。
    没有任何条目被裁时返回的 Dependencies 与 ``ctx`` 内容相同。
    """
    entries: list[tuple[str, str, str | None]] = (
        [("decls", e, _function_name(e)) for e in ctx.decls]
        + [("macros", e, (m.group(1) if (m := _DEFINE_RE.match(e)) else None)) for e in ctx.macros]
        + [("globals", e, _variable_name(e)) for e in ctx.globals])
    reach = identifiers(code) | identifiers("\n".join(ctx.types))
    keep = [False] * len(entries)
    changed = True
    while changed:
        changed = False
        for k, (_, text, name) in enumerate(entries):
            if not keep[k] and (name is None or name in reach):
                keep[k] = changed = True
                reach |= identifiers(text)
    pruned = Dependencies(
        decls=[e for (kind, e, _), ok in zip(entries, keep) if kind == "decls" and ok],
        types=list(ctx.types),
        globals=[e for (kind, e, _), ok in zip(entries, keep) if kind == "globals" and ok],
        macros=[e for (kind, e, _), ok in zip(entries, keep) if kind == "macros" and ok],
        includes=list(ctx.includes))
    return pruned, [name for (_, _, name), ok in zip(entries, keep) if not ok and name]
