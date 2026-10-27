"""候选代码替换回项目、沙箱内编译并跑项目自带测试（设计文档 §4.11）。

``evaluate`` 的流程：

1. ``splice``：把混淆副本整棵复制到临时工作目录，按字节偏移 [start, end) 把目标函数换成候选代码
   （目标文件按 bytes 读写，偏移是 libclang 给的字节偏移，不能拿去切 str）。
2. 单文件语法检查（stage=syntax）：沙箱里 ``gcc -x c -fsyntax-only -fdiagnostics-format=json`` 检查目标文件，
   只带 ``spec.cflags`` 里能直接给 gcc 的部分（-I / -D / -U / -std，以及 -isystem / -iquote），其余忽略。
   失败就停在这一步，不再构建。
3. 整体构建（stage=build）：在 ``build_cmd`` 前注入 ``CFLAGS="$CFLAGS -fdiagnostics-format=json"``，
   构建输出里能解析出的 JSON 诊断一并收集。
4. 测试（stage=test / pass）：``spec.test_cmd`` 非空时单独再跑一次；为空时测试已经含在 ``build_cmd`` 里，
   按 ``pass_criteria`` 判定。沙箱超时取 ``spec.timeout_s``。

测试含在 ``build_cmd`` 里时（九个语料都是这样），一次沙箱运行里分不清「构建失败」和「测试失败」，
这里按证据判：输出里有 gcc 的 error 级诊断、链接错误（undefined reference 等）或编译器 / 链接器 / make 的
报错行，就记 build；否则记 test。

信号：直接运行的测试程序崩溃时，沙箱从退出码（128+n）换算出信号；经 ``make`` 运行时退出码只剩 2，
信号名只留在 ``make: *** [...] Segmentation fault`` 这类文字里，所以测试阶段信号为空时会从文字里补。

临时工作目录放在 ``<项目根>/tmp/work``（环境变量 ``BENCH_WORK_DIR`` 可改），评测完即删除。
"""
from __future__ import annotations

import json
import logging
import re
import shlex
import shutil
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

from bench.corpus import CorpusSpec, materialize
from bench.sandbox import SandboxError, raise_if_docker_failed
from bench.store import default_work_root

if TYPE_CHECKING:  # 只做类型标注，runner 本身不需要 libclang
    from bench.extract import FunctionTarget

log = logging.getLogger(__name__)

__all__ = [
    "EvalResult", "ProjectRun", "SpliceError", "SandboxError",
    "splice", "read_function", "evaluate", "run_project", "syntax_check_command", "syntax_flags",
    "split_gcc_json", "parse_text_diagnostics", "new_workdir", "raise_if_docker_failed",
]

JSON_DIAG_FLAG = "-fdiagnostics-format=json"
# 注入到 build_cmd 前面：保留原有 CFLAGS（环境里没有就只有诊断格式一项）
CFLAGS_INJECTION = 'export CFLAGS="${CFLAGS:+$CFLAGS }' + JSON_DIAG_FLAG + '"\n'

TAIL_CHARS_FAIL = 2000      # 失败样本保留的输出尾部长度
TAIL_CHARS_PASS = 400
MAX_DIAGNOSTICS = 30        # 每个样本最多保留的诊断条数（error 优先）
SYNTAX_TIMEOUT_CAP_S = 60   # 单文件语法检查的超时上限

class SpliceError(RuntimeError):
    """偏移与目标函数对不上（文件被改过、传错了副本等），拒绝写入。内部错误，不是用法错误（所以不继承 ValueError）。"""


@dataclass
class EvalResult:
    task_id: str
    model: str
    sample_idx: int
    stage: str                      # syntax / build / test / pass
    rc: int
    timed_out: bool
    signal: int | None
    diagnostics: list[dict] = field(default_factory=list)
    test_stdout_tail: str = ""      # 构建 / 测试输出的尾部（stdout + 去掉 JSON 诊断行的 stderr）
    duration_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProjectRun:
    """一次「构建 + 测试」的结果（evaluate 的第 3、4 步，stub-kill 与基线检查也复用）。"""

    stage: str                      # build / test / pass
    rc: int
    timed_out: bool
    signal: int | None
    diagnostics: list[dict]         # 已压平，见 _flatten_diag
    stdout: str                     # 构建与测试的 stdout（分两次运行时首尾相接）
    stderr_text: str                # stderr 去掉 JSON 诊断行之后的文本
    duration_s: float


# ----------------------------------------------------------------------
# 替换
# ----------------------------------------------------------------------
def read_function(root: Path, target: "FunctionTarget") -> str:
    """root 下目标函数的原文（字节偏移 [start, end) 解码），oracle 用它当「候选代码」。"""
    data = (Path(root) / target.file).read_bytes()
    if not (0 <= target.start <= target.end <= len(data)) or target.name.encode("utf-8") not in data[target.start:target.end]:
        raise SpliceError(f"{target.file} 的 [{target.start}, {target.end}) 里没有函数名 {target.name}："
                          "目标的字节偏移与这份源码对不上（源码变了而没重跑 targets？）")
    return data[target.start:target.end].decode("utf-8", errors="replace")


def splice(obf_root: Path, target: "FunctionTarget", code: str, dest: Path) -> Path:
    """复制 obf_root 到 dest（不含 .git），把目标文件里字节区间 [start, end) 换成 code，返回 dest。

    dest 已存在会被清空重建。偏移必须落在文件内，且区间里要能看到目标函数名，否则抛 SpliceError。
    code 原样写入（UTF-8），不补换行、不修剪，所以用原函数文本替换后文件逐字节不变。
    """
    dest = materialize(Path(obf_root), Path(dest))
    path = (dest / target.file).resolve()
    if dest.resolve() not in path.parents:
        raise SpliceError(f"目标文件跑出了工作目录: {target.file}")
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise SpliceError(f"读不到目标文件 {target.file}: {exc}") from exc
    if not (0 <= target.start <= target.end <= len(data)):
        raise SpliceError(f"{target.file} 长 {len(data)} 字节，偏移 [{target.start}, {target.end}) 越界")
    if target.name.encode("utf-8") not in data[target.start:target.end]:
        raise SpliceError(f"{target.file} 的 [{target.start}, {target.end}) 里没有函数名 {target.name}，"
                          "偏移可能属于另一份副本")
    new = data[:target.start] + code.encode("utf-8", errors="replace") + data[target.end:]
    path.write_bytes(new)
    return dest


def new_workdir(work_root: Path | None, prefix: str) -> Path:
    """在 work_root 下建一个唯一的空目录（名字不含冒号，docker -v 不接受）；不给就用 ``store.default_work_root()``。"""
    base = Path(work_root) if work_root else default_work_root()
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"{prefix}-{uuid.uuid4().hex[:12]}"
    path.mkdir()
    return path


# ----------------------------------------------------------------------
# gcc 诊断
# ----------------------------------------------------------------------
_PASS_ATTACHED = ("-I", "-D", "-U")
_PASS_SEPARATE = ("-I", "-D", "-U", "-isystem", "-iquote")


def syntax_flags(cflags: Iterable[str]) -> list[str]:
    """spec.cflags 里可以直接给 gcc 的部分：-I / -D / -U（附着或分开写）、-std=、-isystem、-iquote；其余忽略。"""
    out: list[str] = []
    it = iter(cflags or [])
    for tok in it:
        if tok in _PASS_SEPARATE:
            val = next(it, None)
            if val is not None:
                out += [tok, val]
        elif tok.startswith(_PASS_ATTACHED) or tok.startswith("-std="):
            out.append(tok)
    return out


def syntax_check_command(spec: CorpusSpec, file: str) -> str:
    """对 file 做单文件语法检查的完整命令行（在沙箱的项目根下执行）。

    ``-x c`` 让 .h 目标（jsmn.h 这类头文件实现）也按 C 源码检查，而不是当成预编译头。
    ``-Wimplicit-function-declaration`` 只多产生 warning，不影响通过与否：gcc 13 里「调用未声明的函数」
    只是 warning（gcc 14 起才是 error），而且 -std=c89 下默认不报；有了这条 warning，归因才能把
    「函数没声明、链接时才发现没定义」认作 compile_undeclared（见 attribute 模块说明）。
    """
    parts = ["gcc", "-x", "c", "-fsyntax-only", JSON_DIAG_FLAG, "-Wimplicit-function-declaration",
             *syntax_flags(spec.cflags), file]
    return " ".join(shlex.quote(p) for p in parts)


def _is_gcc_diag_list(value: Any) -> bool:
    return (isinstance(value, list)
            and all(isinstance(x, dict) and ("message" in x or "kind" in x) for x in value))


def split_gcc_json(text: str) -> tuple[list[dict], str, bool]:
    """把 stderr 拆成 (gcc JSON 诊断, 剩余文本, 是否见过 JSON 数组)。

    ``-fdiagnostics-format=json`` 下 gcc 每编译一个文件在 stderr 打一行 JSON 数组（没有诊断时是 ``[]``）。
    沙箱把超长输出从中间截断，被截断的 JSON 行解析不了，直接丢弃（它们以 ``[{"kind"`` 开头）。
    """
    diags: list[dict] = []
    rest: list[str] = []
    saw = False
    for line in (text or "").splitlines():
        s = line.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                value = json.loads(s)
            except ValueError:
                value = None
            if _is_gcc_diag_list(value):
                saw = True
                diags.extend(value)
                continue
        if s.startswith('[{"kind"'):
            saw = True
            continue
        rest.append(line)
    return diags, "\n".join(rest), saw


_TEXT_DIAG_RE = re.compile(
    r"^(?P<file>[^\s:][^:\n]*):(?P<line>\d+):(?:(?P<col>\d+):)?\s*"
    r"(?P<kind>fatal error|error|warning):\s*(?P<msg>.*?)\s*$", re.M)


def parse_text_diagnostics(text: str) -> list[dict]:
    """文本格式的 gcc 诊断（``file:line:col: error: msg``）→ 压平的诊断字典。

    只在构建根本没吃到我们注入的 CFLAGS（stderr 里一个 JSON 数组都没有）时当后备用。
    """
    out = []
    for m in _TEXT_DIAG_RE.finditer(text or ""):
        out.append({"kind": m.group("kind"), "message": m.group("msg"), "file": m.group("file"),
                    "line": int(m.group("line")), "column": int(m.group("col")) if m.group("col") else None})
    return out


def _is_error_diag(d: dict) -> bool:
    kind = str(d.get("kind", "error")).lower()
    return kind in ("error", "fatal error", "internal compiler error", "sorry", "sorry, unimplemented") \
        or kind.endswith("error")


def _flatten_diag(d: dict, stage: str) -> dict:
    """gcc 的诊断字典 → 精简字典：kind / message / option / file / line / column / stage。"""
    out: dict[str, Any] = {"kind": str(d.get("kind", "error")), "message": str(d.get("message", ""))}
    if d.get("option"):
        out["option"] = d["option"]
    file, line, col = d.get("file"), d.get("line"), d.get("column")
    if not file:
        locs = d.get("locations") or []
        caret = locs[0].get("caret") if locs and isinstance(locs[0], dict) else None
        if isinstance(caret, dict):
            file, line, col = caret.get("file"), caret.get("line"), caret.get("column")
    if file:
        out.update(file=file, line=line, column=col)
    out["stage"] = stage
    return out


def _cap_diagnostics(diags: list[dict]) -> list[dict]:
    """去重后最多保留 MAX_DIAGNOSTICS 条，error 优先、保持原顺序。"""
    seen: set[tuple] = set()
    uniq = []
    for d in diags:
        key = (d.get("kind"), d.get("message"), d.get("file"), d.get("line"), d.get("column"))
        if key not in seen:
            seen.add(key)
            uniq.append(d)
    errors = [d for d in uniq if _is_error_diag(d)]
    others = [d for d in uniq if not _is_error_diag(d)]
    return (errors + others)[:MAX_DIAGNOSTICS] if len(uniq) > MAX_DIAGNOSTICS else uniq


def _collect_diags(stderr: str, stage: str) -> tuple[list[dict], str, bool]:
    """从 stderr 取诊断（JSON 优先，一个 JSON 数组都没有时退回文本格式）→ (压平的诊断, 剩余文本, 是否见过 JSON 数组)。
    见过 JSON 数组说明至少有一个文件编译完了（gcc 每编完一个文件打一行）。"""
    raw, rest, saw = split_gcc_json(stderr)
    if not saw:
        raw = parse_text_diagnostics(stderr)
    return [_flatten_diag(d, stage) for d in raw], rest, saw


# ----------------------------------------------------------------------
# 阶段与信号的判定
# ----------------------------------------------------------------------
# 构建失败的文字证据（只在 stderr 里找；测试程序自己的 stdout 不参与，避免把测试输出误判成编译错误）
_BUILD_FAIL_RE = re.compile(
    r"undefined reference to|multiple definition of|ld returned \d+ exit status|collect2: error"
    r"|^(?:cc1|gcc|cc|ld|as)(?:\[\d+\])?: (?:fatal )?error|compilation terminated|internal compiler error"
    r"|No rule to make target|cannot find -l\S+", re.M)

# make / bash 打印的「被信号杀死」状态行：make 形如 ``make: *** [Makefile:9: test] Aborted (core dumped)``，
# bash 形如 ``bash: line 3:    13 Segmentation fault      (core dumped) ./a``
_CRASH_NAMES = r"(Segmentation fault|Aborted|Floating point exception|Bus error|Illegal instruction|Killed)"
_CRASH_TEXT_RE = re.compile(
    r"(?:\*\*\*\s+\[[^\]\n]*\]\s+|line \d+:\s+\d+\s+)" + _CRASH_NAMES
    # sh（dash）不带前缀、独占一行的写法；要求整行只有状态文字，免得把程序自己输出的「Aborted by user」当成崩溃
    + r"|^" + _CRASH_NAMES + r"(?: \(core dumped\))?[ \t]*$", re.M)
_CRASH_TEXT_SIGNALS = {"Segmentation fault": 11, "Aborted": 6, "Floating point exception": 8,
                       "Bus error": 7, "Illegal instruction": 4, "Killed": 9}
# 配方由 dash 之类的 sh 执行时，make 只看到退出码 128+n：``make: *** [Makefile:9: test] Error 139``
_MAKE_ERROR_RE = re.compile(r"\*\*\*\s+\[[^\]\n]*\]\s+Error (\d+)")


def _is_build_failure(diags: list[dict], stderr_text: str) -> bool:
    return any(_is_error_diag(d) for d in diags) or bool(_BUILD_FAIL_RE.search(stderr_text))


def _infer_signal(sandbox_signal: int | None, stderr_text: str) -> int | None:
    """沙箱没给出信号时，从 make / bash 的状态行里推：信号名，或 make 报的 Error 128+n。"""
    if sandbox_signal is not None:
        return sandbox_signal
    m = _CRASH_TEXT_RE.search(stderr_text)
    if m:
        return _CRASH_TEXT_SIGNALS[m.group(1) or m.group(2)]
    codes = [int(c) for c in _MAKE_ERROR_RE.findall(stderr_text)]
    if codes and 128 < codes[-1] <= 128 + 64:
        return codes[-1] - 128
    return None


def _tail(stdout: str, stderr_text: str, limit: int, keep_lines: int = 3) -> str:
    """stdout 接上（去掉 JSON 的）stderr，取尾部 limit 字符。

    链接错误行通常紧挨着输出末尾，但后面还有别的输出时可能被挤出尾部；
    这时把它们补到前面，归因要靠这几行。
    """
    text = stdout
    if stderr_text.strip():
        text = (text.rstrip("\n") + "\n" if text else "") + stderr_text
    if len(text) <= limit:
        return text
    tail = text[-limit:]
    missing = [ln for ln in text.splitlines()
               if re.search(r"undefined reference to|multiple definition of", ln) and ln not in tail]
    if missing:
        tail = "\n".join(missing[:keep_lines]) + "\n...\n" + tail
    return tail


# ----------------------------------------------------------------------
# 构建 + 测试
# ----------------------------------------------------------------------
TRUNCATED_NOTE = ("bench: stdout 超过沙箱输出上限被截断，stdout_regex / stdout_not_regex 判据无法在残缺的文本上"
                  "可靠求值，按未通过处理")


def run_project(spec: CorpusSpec, workdir: Path, sandbox: Any, timeout_s: int | None = None) -> ProjectRun:
    """在沙箱里对 workdir 跑构建与测试，返回 ProjectRun（stage 为 build / test / pass）。

    spec.test_cmd 为空：build_cmd 一次跑完（测试含在里面），阶段按证据判（见模块说明）。
    spec.test_cmd 非空：先跑 build_cmd，失败记 build；再单独跑 test_cmd，pass_criteria 对「两次 stdout 首尾相接」判定，
    与 corpus.run_command() 把两段用 && 连成一条命令时的语义一致。

    超时的阶段判定：构建与测试在同一次运行里时，超时 + 没有任何输出证据（没见过 gcc 的 JSON 诊断数组、
    stdout 与 stderr 都是空的）说明连第一个文件都没编完，记 build（不计入编译通过率，stub-kill 也不会把它当成「杀死」）；
    有输出证据的超时记 test。
    """
    timeout = int(timeout_s or spec.timeout_s)
    t0 = time.monotonic()

    res = sandbox.run(workdir, CFLAGS_INJECTION + spec.build_cmd, timeout_s=timeout)
    raise_if_docker_failed(res)
    diags, rest, saw_json = _collect_diags(res.stderr, "build")

    if spec.test_cmd:
        if res.rc != 0 or res.timed_out:
            return ProjectRun("build", res.rc, res.timed_out, res.signal, diags, res.stdout, rest,
                              time.monotonic() - t0)
        res_t = sandbox.run(workdir, spec.test_cmd, timeout_s=timeout)
        raise_if_docker_failed(res_t)
        stdout = res.stdout + res_t.stdout
        rest_t = rest + "\n" + res_t.stderr if rest.strip() else res_t.stderr
        ok, refused = spec.judge(res_t.rc, stdout, truncated=res.stdout_truncated or res_t.stdout_truncated)
        ok = ok and not res_t.timed_out
        if refused:
            rest_t = (rest_t.rstrip("\n") + "\n" if rest_t else "") + TRUNCATED_NOTE
            log.warning("%s: %s", spec.name, TRUNCATED_NOTE)
        stage = "pass" if ok else "test"
        sig = None if ok else _infer_signal(res_t.signal, rest_t)
        return ProjectRun(stage, res_t.rc, res_t.timed_out, sig, diags, stdout, rest_t, time.monotonic() - t0)

    ok, refused = spec.judge(res.rc, res.stdout, truncated=res.stdout_truncated)
    if ok and not res.timed_out:
        return ProjectRun("pass", res.rc, False, None, diags, res.stdout, rest, time.monotonic() - t0)
    if refused:
        rest = (rest.rstrip("\n") + "\n" if rest else "") + TRUNCATED_NOTE
        log.warning("%s: %s", spec.name, TRUNCATED_NOTE)
    if _is_build_failure(diags, rest) and not res.timed_out:
        return ProjectRun("build", res.rc, res.timed_out, res.signal, diags, res.stdout, rest,
                          time.monotonic() - t0)
    if res.timed_out and not saw_json and not res.stdout.strip() and not rest.strip():
        log.info("%s: 超时且没有任何输出证据，按构建阶段超时处理", spec.name)
        return ProjectRun("build", res.rc, True, res.signal, diags, res.stdout, rest, time.monotonic() - t0)
    return ProjectRun("test", res.rc, res.timed_out, _infer_signal(res.signal, rest), diags, res.stdout, rest,
                      time.monotonic() - t0)


# ----------------------------------------------------------------------
# 入口
# ----------------------------------------------------------------------
def evaluate(spec: CorpusSpec, obf_root: Path, target: "FunctionTarget", candidate_code: str | None, sandbox: Any,
             model: str, task_id: str, sample_idx: int, *,
             work_root: Path | None = None, keep_workdir: bool = False) -> EvalResult:
    """把候选代码替换进 obf_root 的副本，沙箱里依次做语法检查、构建、测试，返回 EvalResult。

    target 的偏移必须对应 obf_root（混淆副本里要用 extract.locate 重新定位）。
    candidate_code 为 None（抽不出代码）时不进沙箱，直接返回 stage=syntax、rc=-1 的结果；
    归因看到 Candidate.code 为空会先判 extract_fail，不会用到这个结果。
    work_root / keep_workdir 供调试：改临时目录位置、保留工作副本。
    """
    t0 = time.monotonic()

    def result(stage: str, rc: int, timed_out: bool = False, signal: int | None = None,
               diagnostics: list[dict] | None = None, tail: str = "") -> EvalResult:
        return EvalResult(task_id=task_id, model=model, sample_idx=sample_idx, stage=stage, rc=rc,
                          timed_out=timed_out, signal=signal, diagnostics=diagnostics or [],
                          test_stdout_tail=tail, duration_s=time.monotonic() - t0)

    if candidate_code is None:
        return result("syntax", -1, tail="no candidate code")

    workdir = new_workdir(work_root, "eval")
    try:
        try:
            splice(obf_root, target, candidate_code, workdir)
        except SpliceError:
            log.error("%s: 替换候选代码失败（偏移与副本对不上）", task_id)
            raise

        # 1. 单文件语法检查
        syn = sandbox.run(workdir, syntax_check_command(spec, target.file),
                          timeout_s=min(spec.timeout_s, SYNTAX_TIMEOUT_CAP_S))
        raise_if_docker_failed(syn)
        syn_diags, syn_rest, _ = _collect_diags(syn.stderr, "syntax")
        if syn.rc != 0 or syn.timed_out:
            log.debug("%s: 单文件语法检查未过 (rc=%s, timed_out=%s)", task_id, syn.rc, syn.timed_out)
            return result("syntax", syn.rc, syn.timed_out, syn.signal, _cap_diagnostics(syn_diags),
                          _tail(syn.stdout, syn_rest, TAIL_CHARS_FAIL))

        # 2. 构建 + 测试
        run = run_project(spec, workdir, sandbox)
        log.debug("%s: 构建 + 测试 → stage=%s rc=%s timed_out=%s signal=%s (%.1fs)", task_id, run.stage, run.rc,
                  run.timed_out, run.signal, run.duration_s)
        diags = _cap_diagnostics(syn_diags + run.diagnostics)
        limit = TAIL_CHARS_PASS if run.stage == "pass" else TAIL_CHARS_FAIL
        return result(run.stage, run.rc, run.timed_out, run.signal, diags,
                      _tail(run.stdout, run.stderr_text, limit))
    finally:
        if not keep_workdir:
            shutil.rmtree(workdir, ignore_errors=True)
