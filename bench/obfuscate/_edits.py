"""L2（结构混淆）与 L3（语义混淆）共用的基础设施。

两个变换模块各自只保留「找可改位置」（SiteFinder）和各自的统计键，下面这些通用部分只有一份
（以前两边各抄了一套，渲染方式、递归上限、符号链接保护、头文件借宿主解析等都已经分叉）：

* :class:`Edit`：一处改写（原文字节区间 + 新文本配方）。配方里的元素是字面量（bytes，或 latin-1 的 str）
  或 ``(a, b)`` 二元组——后者取原文 [a, b) 并递归渲染落在其中的其它改写，所以嵌套的改写可以同时生效，
  新文本的各个组成部分始终取自原文，不存在偏移失效；
* :func:`check_laminar` / :func:`render`：保证改写互相嵌套或不相交，并渲染成新文本（每处改写保持原文总行数不变）；
* :func:`rollback_invalid`：改写后重新解析必须零 error，有 error 时按二分法回滚出错的改写；
* :func:`match_close` / :class:`MacroIntervals`：libclang 记号表上的括号配对，以及最外层宏调用区间的合并与冲突判定
  （structure 的 ``_FileCtx`` 与缺陷修复任务的 ``mutate`` 共用；semantic 用自己的正则记号表，不在其中）；
* :class:`FileStats` / :func:`empty_stats` / :func:`merge_stats`：逐文件统计与汇总；
* :class:`RecursionGuard`：渲染深层嵌套的改写时临时调高递归上限；
* :class:`Unit` / :class:`Parser`：libclang 解析，头文件单独解析不过（靠别的头才自洽）时借一个包含它的项目 .c 当宿主；
* :func:`plan_files` / :func:`host_map` / :func:`run_files`：要处理的文件列表（过滤 vendored、按 files 限定）、
  宿主映射，以及逐文件的驱动循环（含「符号链接或位于副本之外不改写」的保护）。

文件一律按 bytes 处理：偏移是 libclang 给的字节偏移，改写后的文本也按 bytes 拼接。
"""
from __future__ import annotations

import bisect
import logging
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import clang.cindex as ci

from bench import clangcfg, corpus

log = logging.getLogger(__name__)

__all__ = [
    "SOURCE_SUFFIXES", "Edit", "check_laminar", "render", "rollback_invalid", "FileStats", "empty_stats",
    "merge_stats", "RecursionGuard", "Unit", "Parser", "file_name", "plan_files", "host_map", "run_files",
    "safe_to_write", "libclang_args", "OPEN_TO_CLOSE", "match_close", "MacroIntervals",
]

SOURCE_SUFFIXES = (".c", ".h")


# ---------------------------------------------------------------------------
# 改写与渲染
# ---------------------------------------------------------------------------
@dataclass
class Edit:
    """一处改写：把原文 [start, end) 换成 parts 拼出的新文本。

    parts 里的元素是 bytes（字面量；str 按 latin-1 转成 bytes）或 (a, b) 二元组（取原文 [a, b) 并递归渲染落在其中的其它改写）。
    """

    kind: str
    start: int
    end: int
    parts: list

    def __post_init__(self) -> None:
        self.parts = [p.encode("latin-1") if isinstance(p, str) else p for p in self.parts]

    @property
    def subs(self) -> list[tuple[int, int]]:
        return [p for p in self.parts if isinstance(p, tuple)]


def render(data: bytes, edits: Sequence[Edit]) -> bytes:
    """把一组两两嵌套或不相交的 edits 应用到 data 上，返回新文本。

    每处改写保持原文的总行数不变：新文本丢掉了换行（括号、分号之间的空白被丢弃）时，把缺的换行补在这处改写的末尾，
    所以改写点之后的代码行号（以及 ``__LINE__``）不受影响；被移动的片段在改写内部的行号可能略有偏移。
    """
    ordered = sorted(edits, key=lambda e: (e.start, -e.end))
    starts = [e.start for e in ordered]
    nl = b"\r\n" if b"\r\n" in data[:data.find(b"\n") + 1] else b"\n"

    def render_range(a: int, b: int) -> bytes:
        out: list[bytes] = []
        pos = a
        i = bisect.bisect_left(starts, a)
        while i < len(ordered) and starts[i] < b:
            ed = ordered[i]
            i += 1
            if ed.start < pos:      # 嵌套在上一个已渲染的 edit 里，由它的 parts 负责
                continue
            if ed.end > b:          # 越界（不应出现，校验阶段已排除）
                continue
            out.append(data[pos:ed.start])
            out.append(render_edit(ed))
            pos = ed.end
        out.append(data[pos:b])
        return b"".join(out)

    def render_edit(ed: Edit) -> bytes:
        text = b"".join(p if isinstance(p, bytes) else render_range(p[0], p[1]) for p in ed.parts)
        missing = data.count(b"\n", ed.start, ed.end) - text.count(b"\n")
        return text + nl * missing if missing > 0 else text

    return render_range(0, len(data))


def check_laminar(edits: Sequence[Edit]) -> tuple[list[Edit], int]:
    """保证 edits 构成嵌套族：任意两个要么不相交，要么一个完全落在另一个的某个子区间里。返回 (保留的, 丢弃数)。"""
    kept: list[Edit] = []
    dropped = 0
    stack: list[Edit] = []
    for ed in sorted(edits, key=lambda e: (e.start, -e.end)):
        while stack and stack[-1].end <= ed.start:
            stack.pop()
        if stack:
            top = stack[-1]
            if ed.end > top.end or not any(a <= ed.start and ed.end <= b for a, b in top.subs):
                dropped += 1
                continue
        kept.append(ed)
        stack.append(ed)
    return kept, dropped


def rollback_invalid(chosen: list[Edit], validate: Callable[[list[Edit]], tuple[bool, list[str]]],
                     rolled_back: Counter, rel: str) -> list[Edit]:
    """找出能保持零 error 的最大改写子集：先试全部，失败就二分，单个改写仍失败才回滚它（按类别记入 rolled_back）。

    ``validate(edits)`` 渲染并重新解析，返回 (是否零 error, error 文字)。chosen 须已按 (起点, -终点) 排好序。
    """
    if not chosen:
        return []
    ok, errs = validate(chosen)
    if ok:
        return chosen
    log.warning("%s: 改写后重新解析有 %d 个 error（首个: %s），逐段排查回滚", rel, len(errs), errs[0] if errs else "?")

    def solve(accepted: list[Edit], pending: list[Edit]) -> list[Edit]:
        if not pending:
            return accepted
        good, _ = validate(accepted + pending)
        if good:
            return accepted + pending
        if len(pending) == 1:
            ed = pending[0]
            rolled_back[ed.kind] += 1
            log.warning("%s: 回滚 %s @%d-%d", rel, ed.kind, ed.start, ed.end)
            return accepted
        mid = len(pending) // 2
        return solve(solve(accepted, pending[:mid]), pending[mid:])

    return solve([], chosen)


class RecursionGuard:
    """渲染深层嵌套的改写（极深的 else-if 链等）时递归较深，临时调高递归上限。"""

    def __init__(self, limit: int = 20000) -> None:
        self.limit = limit
        self.old = 0

    def __enter__(self) -> "RecursionGuard":
        self.old = sys.getrecursionlimit()
        if self.old < self.limit:
            sys.setrecursionlimit(self.limit)
        return self

    def __exit__(self, *exc: object) -> None:
        sys.setrecursionlimit(self.old)


# ---------------------------------------------------------------------------
# 记号表上的括号配对与宏调用区间
# ---------------------------------------------------------------------------
OPEN_TO_CLOSE = {"(": ")", "[": "]", "{": "}"}


def match_close(tokens: Sequence[str], i: int) -> int | None:
    """``tokens[i]`` 是 ``(`` ``[`` ``{`` 之一，返回与之配对的右括号下标；括号不配对（或深度回不到 0）返回 None。"""
    stack: list[str] = []
    for j in range(i, len(tokens)):
        t = tokens[j]
        if t in OPEN_TO_CLOSE:
            stack.append(OPEN_TO_CLOSE[t])
        elif t in (")", "]", "}"):
            if not stack or stack.pop() != t:
                return None
            if not stack:
                return j
    return None


class MacroIntervals:
    """一个文件里「最外层宏调用」的字节区间：嵌套和重叠的合并成一个，按起点排好序，供二分查找。

    宏展开出来的代码、宏调用的实参，都不能按偏移改写（改了展开结果就变了），所以 L2 / L3 / 变异算子找到的节点范围要先和它们比一下。
    """

    def __init__(self, raw: Iterable[tuple[int, int]]) -> None:
        merged: list[tuple[int, int]] = []
        for s, e in sorted(raw):
            if merged and s < merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        self.intervals = merged
        self._starts = [a for a, _ in merged]

    def __bool__(self) -> bool:
        return bool(self.intervals)

    def conflicts(self, s: int, e: int, *, allow_equal: bool = False) -> bool:
        """[s, e) 与某个最外层宏调用冲突：部分重叠，或整个落在宏调用里面（宏实参）。

        宏调用整个落在 [s, e) 内部不算冲突（``x = MAX(a, b);`` 整句可以删，``if (ISDIGIT(c))`` 的条件可以整个取反）；
        两者范围恰好相等时，``allow_equal`` 为真才放行：「把整个宏调用包起来」放行，「宏生成的整条语句」不放行。"""
        iv = self.intervals
        i = max(bisect.bisect_right(self._starts, s) - 1, 0)
        while i < len(iv) and iv[i][0] < e:
            ms, me = iv[i]
            i += 1
            if me <= s:
                continue
            if s <= ms and me <= e:
                if (ms, me) == (s, e) and not allow_equal:
                    return True
                continue
            return True
        return False


# ---------------------------------------------------------------------------
# 统计
# ---------------------------------------------------------------------------
@dataclass
class FileStats:
    kinds: tuple[str, ...]
    applied: Counter = field(default_factory=Counter)
    candidates: Counter = field(default_factory=Counter)
    skipped: dict = field(default_factory=dict)
    rolled_back: Counter = field(default_factory=Counter)
    overlap: int = 0
    functions: int = 0
    skipped_file: str | None = None
    host: str | None = None         # 头文件单独解析不过、借来当宿主的 .c 文件名

    def __post_init__(self) -> None:
        if not self.skipped:
            self.skipped = {k: Counter() for k in self.kinds}

    def to_dict(self) -> dict:
        return {
            "applied": {k: self.applied[k] for k in self.kinds if self.applied[k]},
            "candidates": {k: self.candidates[k] for k in self.kinds if self.candidates[k]},
            "skipped": {k: dict(v) for k, v in self.skipped.items() if v},
            "rolled_back": {k: v for k, v in self.rolled_back.items() if v},
            "overlap": self.overlap,
            "skipped_file": self.skipped_file,
            "host": self.host,
        }


def empty_stats(kinds: Sequence[str], seed: int, ratio: float, enabled: Sequence[str], **extra: object) -> dict:
    """汇总统计的初值：各类别的应用次数、候选数、跳过原因、回滚数，以及 seed / ratio / 启用的类别；
    ``extra`` 是各变换自己的附加键（语义混淆的 ``functions`` / ``opaque_ratio``）。"""
    out: dict = {k: 0 for k in kinds}
    out.update({
        "total": 0,
        "candidates": {k: 0 for k in kinds},
        "skipped": {k: {} for k in kinds},
        "rolled_back": {k: 0 for k in kinds},
        "overlap_dropped": 0,
        "files_total": 0,
        "files_changed": 0,
        "files_skipped": {},
        "files": {},
        "seed": seed,
        "ratio": ratio,
        "kinds": list(enabled),
    })
    out.update(extra)
    return out


def merge_stats(total: dict, rel: str, fs: FileStats, changed: bool) -> None:
    total["files_total"] += 1
    if changed:
        total["files_changed"] += 1
    if fs.skipped_file:
        total["files_skipped"][rel] = fs.skipped_file
    if "functions" in total:
        total["functions"] += fs.functions
    for k in fs.kinds:
        total[k] += fs.applied[k]
        total["total"] += fs.applied[k]
        total["candidates"][k] += fs.candidates[k]
        total["rolled_back"][k] += fs.rolled_back[k]
        for why, n in fs.skipped[k].items():
            total["skipped"][k][why] = total["skipped"][k].get(why, 0) + n
    total["overlap_dropped"] += fs.overlap
    detail = fs.to_dict()
    if detail["applied"] or detail["rolled_back"] or detail["skipped_file"]:
        total["files"][rel] = {k: v for k, v in detail.items()
                               if k in ("applied", "rolled_back", "skipped_file", "host") and v}


# ---------------------------------------------------------------------------
# libclang 解析
# ---------------------------------------------------------------------------
def libclang_args(spec: corpus.CorpusSpec, src_root: Path) -> list[str]:
    """spec.cflags 转 libclang 参数（唯一口径见 :func:`bench.clangcfg.clang_args`）。"""
    return clangcfg.clang_args(spec.cflags, Path(src_root))


@dataclass
class Unit:
    """libclang 解析 path 的方式：path 自己当主文件；头文件单独解析不过（靠别的头才自洽）时，借一个包含它的 .c 当宿主。"""

    path: Path
    host: Path | None = None
    host_data: bytes | None = None

    @property
    def main(self) -> Path:
        return self.host or self.path

    def unsaved(self, data: bytes) -> list[tuple[str, bytes]]:
        """data 是 path 的（新）内容；宿主文件用读到的原内容。"""
        if self.host is None:
            return [(str(self.path), data)]
        return [(str(self.host), self.host_data or b""), (str(self.path), data)]


def file_name(tu: ci.TranslationUnit, path: Path) -> str | None:
    """libclang 在这个 TU 里称呼 path 的名字；path 没被这个 TU 用到时返回 None（get_file 此时抛 AssertionError）。"""
    try:
        return tu.get_file(str(path)).name
    except (AssertionError, ci.LibclangError):
        return None


class Parser:
    """带固定参数的 libclang 解析器：解析、数 error、改写后校验、打开文件（必要时借宿主）。"""

    def __init__(self, args: Sequence[str]) -> None:
        self.args = list(args)
        self.index = clangcfg.index()

    def parse(self, unit: Unit, data: bytes, detailed: bool) -> ci.TranslationUnit:
        opts = clangcfg.DEFAULT_PARSE_OPTIONS if detailed else 0
        return self.index.parse(str(unit.main), args=self.args, unsaved_files=unit.unsaved(data), options=opts)

    @staticmethod
    def errors(tu: ci.TranslationUnit) -> list[str]:
        return [f"{d.location.line}:{d.location.column}: {d.spelling}"
                for d in tu.diagnostics if d.severity >= ci.Diagnostic.Error]

    def valid(self, unit: Unit, data: bytes) -> tuple[bool, list[str]]:
        try:
            errs = self.errors(self.parse(unit, data, detailed=False))
        except ci.TranslationUnitLoadError as exc:
            return False, [f"TranslationUnitLoadError: {exc}"]
        return not errs, errs

    def open(self, path: Path, data: bytes, hosts: Sequence[Path]) -> tuple[Unit, ci.TranslationUnit, list[str]]:
        """解析 path；头文件单独解析有 error 时，依次试 hosts（包含它的项目 .c），取第一个解析干净的当宿主。"""
        unit = Unit(path)
        tu = self.parse(unit, data, detailed=True)
        errs = self.errors(tu)
        if errs and path.suffix == ".h":
            for host in list(hosts)[:3]:
                try:
                    cand = Unit(path, host, host.read_bytes())
                    tu2 = self.parse(cand, data, detailed=True)
                except (OSError, ci.TranslationUnitLoadError):
                    continue
                if not self.errors(tu2) and file_name(tu2, path) is not None:
                    log.info("%s 单独解析有 error，改借 %s 当宿主解析", path.name, host.name)
                    return cand, tu2, []
        return unit, tu, errs


# ---------------------------------------------------------------------------
# 文件列表与逐文件驱动
# ---------------------------------------------------------------------------
_INCLUDE_RE = re.compile(rb'^[ \t]*#[ \t]*include[ \t]*[<"]([^">\n]+)[">]', re.M)


def host_map(c_files: Iterable[Path]) -> dict[str, list[Path]]:
    """头文件基本名 -> ``#include`` 了同名头文件的项目 .c（按给定顺序）；头文件单独解析不过时拿来当宿主。"""
    out: dict[str, list[Path]] = {}
    for p in c_files:
        try:
            text = p.read_bytes()
        except OSError:
            continue
        for m in _INCLUDE_RE.finditer(text):
            base = m.group(1).decode("utf-8", "replace").replace("\\", "/").rsplit("/", 1)[-1]
            hosts = out.setdefault(base, [])
            if p not in hosts:
                hosts.append(p)
    return out


def plan_files(spec: corpus.CorpusSpec, src_root: Path, *, files: Iterable[str] | None = None,
               include_vendored: bool = False) -> tuple[list[Path], list[tuple[str, Path]]]:
    """(project_files 展开的全部文件, 要处理的 [(相对路径, 路径)])：只处理 .c / .h，默认跳过 vendored，
    ``files`` 给了就只处理其中的相对路径。"""
    all_files = corpus.expand_files(spec, src_root)
    wanted = None if files is None else {Path(f).as_posix() for f in files}
    todo: list[tuple[str, Path]] = []
    for path in all_files:
        rel = path.relative_to(src_root).as_posix()
        if path.suffix not in SOURCE_SUFFIXES or (not include_vendored and spec.is_vendored(rel)):
            continue
        if wanted is not None and rel not in wanted:
            continue
        todo.append((rel, path))
    return all_files, todo


def safe_to_write(src_root: Path, path: Path) -> bool:
    """path 是不是可以放心写回：工作副本里的符号链接可能指向副本之外（例如原始语料目录），写下去会改坏原件。"""
    return not (path.is_symlink() or Path(src_root) not in path.resolve().parents)


def run_files(todo: Sequence[tuple[str, Path]], src_root: Path, kinds: Sequence[str], stats: dict,
              transform: Callable[[Path, bytes, str], tuple[bytes, FileStats]], label: str) -> None:
    """逐文件驱动：读 → 变换 → 符号链接保护 → 写回 → 并入汇总统计。单个文件出意外不拖垮整个混淆，该文件保持原样。"""
    kinds_t = tuple(kinds)
    for rel, path in todo:
        data = path.read_bytes()
        try:
            new, fs = transform(path, data, rel)
        except Exception as exc:    # noqa: BLE001 - 单个文件出意外不能拖垮整个混淆：记下来，文件保持原样
            log.exception("%s处理 %s 时出错，该文件保持原样", label, rel)
            new, fs = data, FileStats(kinds_t, skipped_file=f"处理出错: {type(exc).__name__}: {exc}")
        changed = new != data
        if changed and not safe_to_write(src_root, path):
            fs = FileStats(kinds_t, skipped_file="符号链接或位于副本之外，不改写")
            log.warning("%s: %s", rel, fs.skipped_file)
            new, changed = data, False
        if changed:
            path.write_bytes(new)
        merge_stats(stats, rel, fs, changed)
        log.debug("%s: %s", rel, dict(fs.applied))
