"""生成阶段的 prompt（设计文档 §4.9）。

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
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

from bench.extract import Dependencies

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

USER_TEMPLATE = (
    "Implement the following C function for an existing project.\n"
    "\n"
    "## Functional specification\n"
    "{description}\n"
    "\n"
    "## Project context\n"
    "These declarations are already in scope where the function will be placed. Do not repeat them.\n"
    "```c\n"
    "{context}\n"
    "```\n"
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

PROMPT_DIGEST = hashlib.sha256(
    "\x1f".join(
        [
            PROMPT_VERSION, SYSTEM_PROMPT, USER_TEMPLATE, *CONSTRAINT_LINES, EMPTY_CONTEXT, CONTEXT_JOINER,
            *(f"{title}\x1e{sep}" for title, sep in CONTEXT_SECTIONS),
        ]
    ).encode("utf-8")
).hexdigest()


# --------------------------------------------------------------------------- 任务


@dataclass
class Task:
    """一个（目标函数, 混淆级）的生成任务；字段与设计文档 §4.9 一致，tasks.jsonl 存它的全部字段。"""

    task_id: str
    target_id: str
    corpus: str
    level: str                  # L0 / L1 / L2 / L3
    obf_name: str               # 该混淆级下目标函数的名字（L0 即原名）
    obf_signature: str          # 该混淆级下的签名（不含函数体、不带分号）
    context: Dependencies       # 该混淆级下的依赖声明（名字已按映射改写）
    description: str            # 功能规格（四级共用）
    includes: list[str] = field(default_factory=list)   # 已包含的系统头 #include 行；为空时用 context.includes
    language_std: str = "c99"   # 来自 spec.cflags 的 -std，如 c89 / c99 / gnu99

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Task":
        """to_dict 的逆操作；多余的键忽略，context 缺失的列表按空处理。"""
        fields = {k: v for k, v in d.items() if k in cls.__dataclass_fields__ and k != "context"}
        ctx = d.get("context") or {}
        fields["context"] = Dependencies(**{k: list(ctx.get(k, [])) for k in Dependencies.__dataclass_fields__})
        if "includes" in fields:
            fields["includes"] = list(fields["includes"] or [])
        return cls(**fields)


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


def build_messages(task: Task) -> list[dict]:
    """生成阶段发给模型的 messages：[system, user]。纯函数，不读任何外部状态。"""
    context = format_context(task.context, includes=task.includes or None) or EMPTY_CONTEXT
    constraints = "\n".join(
        "- " + line.format(language_std=task.language_std, name=task.obf_name) for line in CONSTRAINT_LINES
    )
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
