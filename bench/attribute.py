"""错误归因（设计文档 §4.12）：把一次评测样本归到唯一一类。

规则（一个样本只归一类，按下列顺序取第一个命中）：

1. 没有可用代码（Candidate.code 为 None 或全空白）               → extract_fail
1b. 候选违反提示词里的硬性约束（调用 exit / abort 等禁用函数、内联汇编、#define 等，
    Candidate.violations 非空）                                    → policy_violation
    这类候选不进沙箱：只看退出码的语料里，候选写一句 ``exit(0);`` 就会被误判为通过
2. 评测阶段是 pass                                                → pass
3. 编译诊断（只看 error 级）的 message，按类别依次匹配：
   含 expected / syntax / stray（``was expected`` 除外，见下）      → compile_syntax
   （另认 unterminated / missing terminating：注释、字符串、字符常量没收尾，也是语法层面的错）
   含 undeclared / implicit declaration / unknown type / has no member → compile_undeclared
   含 incompatible / too few / too many / makes pointer / invalid operands → compile_type
   有 error 但都不在上面                                          → compile_other
   （类别之间按上面的顺序比较，不是按诊断出现的先后：同一样本里既有语法错误又有未声明标识符，
   归 compile_syntax。）
4. 输出里有链接错误（undefined reference 等）                     → link_error
   例外：gcc 13 把「调用未声明的函数」降成 warning（implicit declaration of function 'foo'，gcc 14 起才是 error），
   这样的候选能编过、死在链接上。如果 undefined reference 的符号恰好是被隐式声明的函数，
   归 compile_undeclared（它本质上就是「用了没声明的东西」），detail 里注明是链接时才发现的。
5. 阶段是 test 且 signal 在 {SIGSEGV, SIGABRT, SIGFPE, SIGBUS}      → runtime_crash
6. timed_out 且阶段是 test                                         → runtime_timeout
   构建 / 语法检查阶段超时（连编译都没走完，没有任何测试输出）不是「运行超时」，
   不计入编译通过率，归 compile_other，detail 写明是构建超时
7. 阶段是 test（退出码非 0，或退出码 0 但 pass_criteria 的输出判据不满足）→ test_fail
8. 兜底：阶段是 syntax / build 但没有任何可识别的诊断              → compile_other

和设计文档的差别：

* ``expected`` 不匹配 ``was expected``。gcc 的类型错误 ``incompatible types when returning type 'X' but 'Y' was
  expected`` 里也有 expected，按字面匹配会把它归成语法错误；语法错误的措辞是 ``expected ';' before ...`` 这类，
  ``expected`` 在句首或动词位置，不会前缀 ``was``；
* 语法关键字多认 ``unterminated`` / ``missing terminating``（未收尾的注释、字符串、字符常量），其余关键字完全按文档；
* 链接错误除 ``undefined reference`` 外，还认 ``multiple definition of`` / ``ld returned N exit status`` /
  ``cannot find -l``，它们同样是链接器报的错；
* 规则 7 不要求退出码非 0：mpc / parson / vec 这类用输出判成败的语料，测试失败时退出码仍是 0。

本模块只依赖标准库；EvalResult / Candidate 只用于类型标注。
"""
from __future__ import annotations

import re
import signal as _signal
from typing import TYPE_CHECKING, Any, Iterable

from bench.vocab import CATEGORIES as _CATEGORIES

if TYPE_CHECKING:  # 只做类型标注，避免运行期引入 runner / postprocess 的依赖
    from bench.postprocess import Candidate
    from bench.runner import EvalResult

__all__ = ["CATEGORIES", "CRASH_SIGNALS", "attribute", "compile_category"]

CATEGORIES = _CATEGORIES

CRASH_SIGNALS = frozenset(int(s) for s in (_signal.SIGSEGV, _signal.SIGABRT, _signal.SIGFPE, _signal.SIGBUS))

# 编译诊断关键字（小写子串匹配），顺序即归因优先级
_COMPILE_RULES: list[tuple[str, re.Pattern]] = [
    ("compile_syntax", re.compile(r"(?<!was )expected|syntax|stray|unterminated|missing terminating")),
    ("compile_undeclared", re.compile(r"undeclared|implicit declaration|unknown type|has no member")),
    ("compile_type", re.compile(r"incompatible|too few|too many|makes pointer|invalid operands")),
]

_ERROR_KINDS = frozenset({"error", "fatal error", "internal compiler error", "sorry", "sorry, unimplemented"})

_IMPLICIT_RE = re.compile(r"implicit declaration of function ['‘`]?([A-Za-z_]\w*)")
_UNDEFINED_RE = re.compile(r"undefined reference to [`'‘\"]?([A-Za-z_][\w.@]*)")
_LINK_RE = re.compile(
    r"undefined reference to|multiple definition of|ld returned \d+ exit status|cannot find -l\S+")

DETAIL_LIMIT = 300


# ----------------------------------------------------------------------
# 诊断字典的读取：兼容 gcc 原始 JSON（locations[0].caret）和 runner 压平后的写法（file / line / column）
# ----------------------------------------------------------------------
def _is_error(diag: dict[str, Any]) -> bool:
    kind = str(diag.get("kind", "error")).lower()
    return kind in _ERROR_KINDS or kind.endswith("error")


def _message(diag: dict[str, Any]) -> str:
    return str(diag.get("message", "") or "")


def _location(diag: dict[str, Any]) -> str:
    file, line, col = diag.get("file"), diag.get("line"), diag.get("column")
    if not file:
        locs = diag.get("locations") or []
        caret = (locs[0] or {}).get("caret") if locs and isinstance(locs[0], dict) else None
        if isinstance(caret, dict):
            file, line, col = caret.get("file"), caret.get("line"), caret.get("column")
    if not file:
        return ""
    out = str(file)
    if line:
        out += f":{line}"
        if col:
            out += f":{col}"
    return out


def _fmt_diag(diag: dict[str, Any]) -> str:
    loc = _location(diag)
    msg = _message(diag)
    return f"{loc}: {msg}" if loc else msg


def _clip(text: str, limit: int = DETAIL_LIMIT) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _error_diagnostics(diagnostics: Iterable[dict[str, Any]] | None) -> list[dict[str, Any]]:
    return [d for d in (diagnostics or []) if isinstance(d, dict) and _is_error(d)]


def compile_category(diagnostics: Iterable[dict[str, Any]] | None) -> tuple[str, str] | None:
    """只按编译诊断归类：返回 (compile_*, 命中的那条诊断)，没有 error 级诊断时返回 None。"""
    errors = _error_diagnostics(diagnostics)
    if not errors:
        return None
    for category, pattern in _COMPILE_RULES:
        for diag in errors:
            if pattern.search(_message(diag).lower()):
                return category, _fmt_diag(diag)
    return "compile_other", _fmt_diag(errors[0])


def _signal_name(num: int) -> str:
    try:
        return _signal.Signals(num).name
    except ValueError:
        return f"signal {num}"


def _last_line(text: str) -> str:
    for line in reversed((text or "").splitlines()):
        if line.strip():
            return line.strip()
    return ""


def _link_line(text: str) -> str | None:
    """输出里第一条链接错误所在的行；没有则 None。"""
    for line in (text or "").splitlines():
        if _LINK_RE.search(line):
            return line.strip()
    return None


def _implicit_and_undefined(diagnostics: Iterable[dict[str, Any]] | None, text: str) -> str | None:
    """被隐式声明、而且链接时报 undefined reference 的函数：返回那条 warning 的描述，没有则 None。"""
    undefined = set(_UNDEFINED_RE.findall(text or ""))
    if not undefined:
        return None
    for diag in diagnostics or []:
        if not isinstance(diag, dict):
            continue
        m = _IMPLICIT_RE.search(_message(diag))
        if m and m.group(1) in undefined:
            return _fmt_diag(diag)
    return None


# ----------------------------------------------------------------------
# 入口
# ----------------------------------------------------------------------
def attribute(ev: "EvalResult | None", cand: "Candidate") -> tuple[str, str]:
    """返回 (category, detail)。detail 是给人看的一句话（命中的诊断、信号名、退出码等），不超过 300 字符。

    抽不出代码的样本不会进沙箱，没有 EvalResult：cand.code 为空时 ev 可以传 None，直接得到 extract_fail。
    """
    code = getattr(cand, "code", None)
    issues = list(getattr(cand, "issues", None) or [])
    note = f" [issues: {', '.join(issues)}]" if issues else ""

    # 1. 没有可用代码
    if code is None or not str(code).strip():
        return "extract_fail", _clip("no code block in reply" + note)

    # 1b. 违反硬性约束（不进沙箱，ev 可以为 None）
    violations = list(getattr(cand, "violations", None) or [])
    if violations or "policy_violation" in issues:
        what = ", ".join(violations) or "policy_violation"
        return "policy_violation", _clip(f"forbidden construct in candidate: {what}" + note)

    # 2. 通过
    if ev.stage == "pass":
        return "pass", ""

    # 3. 编译诊断
    hit = compile_category(ev.diagnostics)
    if hit is not None:
        return hit[0], _clip(hit[1] + note)

    # 4. 链接错误
    tail = ev.test_stdout_tail or ""
    link = _link_line(tail)
    if link is not None:
        implicit = _implicit_and_undefined(ev.diagnostics, tail)
        if implicit is not None:
            return "compile_undeclared", _clip(f"{implicit} (undefined at link time)" + note)
        return "link_error", _clip(link + note)

    # 5. 崩溃信号（只看测试阶段）
    if ev.stage == "test" and ev.signal in CRASH_SIGNALS:
        extra = _last_line(tail)
        return "runtime_crash", _clip(f"{_signal_name(ev.signal)}" + (f": {extra}" if extra else "") + note)

    # 6. 超时：只有测试阶段的超时才是运行超时；构建 / 语法检查没走完就超时是编译层面的问题
    if ev.timed_out:
        if ev.stage == "test":
            return "runtime_timeout", _clip(f"timed out after {ev.duration_s:.1f}s (rc={ev.rc})" + note)
        return "compile_other", _clip(f"{ev.stage} stage timed out after {ev.duration_s:.1f}s (rc={ev.rc})" + note)

    # 7. 测试失败
    if ev.stage == "test":
        extra = _last_line(tail)
        return "test_fail", _clip(f"exit code {ev.rc}" + (f": {extra}" if extra else "") + note)

    # 8. 兜底：构建 / 语法阶段失败，但没有可识别的 error 诊断
    extra = _last_line(tail)
    return "compile_other", _clip(f"{ev.stage} stage failed (rc={ev.rc})" + (f": {extra}" if extra else "") + note)
