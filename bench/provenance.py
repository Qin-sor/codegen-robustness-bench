"""结果的出处信息：沙箱身份、阶段代码哈希，进各阶段的输入哈希，也进结果记录和报告的「来源」一节。

为什么需要它们（输入哈希只含 spec / 源码树 / 提示词会漏掉什么）：

* 沙箱镜像是浮动标签（``gcc:13``）。镜像换了版本，gcc 的诊断文字、警告行为、默认标准都可能变，
  旧结果仍被当成有效就会把两个版本的数据混在一起，所以镜像 ID（以及资源上限）要进输入哈希；
* 判定逻辑（runner / postprocess / attribute 等）改了，旧结果同样不该沿用。靠人手工加 ``STAGE_VERSIONS``
  不可靠，这里直接对相关源文件的内容取哈希，改了任何一行都会让对应阶段的旧产出失效
  （宁可多重算，也不留过期数据）。

只依赖标准库与 :mod:`bench.store`，不拉起 libclang。
"""
from __future__ import annotations

import functools
import logging
from pathlib import Path
from typing import Any, Mapping

from bench.store import digest

log = logging.getLogger(__name__)

__all__ = ["CODE_MODULES", "code_digest", "sandbox_identity", "sandbox_digest", "TOOLCHAIN_KEYS"]

_PKG_DIR = Path(__file__).resolve().parent

_JUDGE = ("runner", "postprocess", "attribute", "similarity", "sandbox", "extract", "corpus", "prompt", "clangcfg")

#: 阶段 -> 影响该阶段产出的 bench 模块（相对 bench/ 的路径，不含 .py；目录写 ``obfuscate/*``）
CODE_MODULES: dict[str, tuple[str, ...]] = {
    "prepare": ("corpus", "sandbox", "runner", "store"),
    "coverage": ("coverage", "corpus", "sandbox", "runner"),
    "targets": ("targets", "extract", "coverage", "corpus", "sandbox", "runner", "attribute", "clangcfg"),
    "obfuscate": ("obfuscate/*", "extract", "corpus", "clangcfg"),
    "oracle_gate": _JUDGE,
    "evaluate": _JUDGE,
    "describe": ("describe", "extract", "prompt", "clangcfg"),
    # 缺陷修复任务的变异体：位点识别与改写（mutate、obfuscate/_edits）、kill 检查走的评测与归因（runner、attribute、postprocess）、
    # 副本里重新定位目标（extract、clangcfg）、沙箱与语料判据（sandbox、corpus）、归因类别词表（vocab），
    # 以及变异体记录里的「上下文里多余的名字」（prompt.prune_context 与它用的词法器 similarity）
    "mutate": ("mutate", "runner", "attribute", "extract", "corpus", "sandbox", "clangcfg", "obfuscate/_edits",
               "postprocess", "vocab", "prompt", "similarity"),
}

#: 沙箱身份里决定「工具链是不是同一个」的字段（镜像标签与镜像 ID）；其余是资源上限
TOOLCHAIN_KEYS = ("class", "image", "image_id")


def _files_for(stage: str) -> list[Path]:
    out: list[Path] = []
    for spec in CODE_MODULES[stage]:
        if spec.endswith("/*"):
            out += sorted((_PKG_DIR / spec[:-2]).glob("*.py"))
        else:
            out.append(_PKG_DIR / f"{spec}.py")
    return out


@functools.lru_cache(maxsize=None)
def code_digest(stage: str) -> str:
    """阶段相关源文件的内容哈希（16 位十六进制）。同一进程内只算一次；源文件缺失按空内容算。"""
    if stage not in CODE_MODULES:
        raise KeyError(f"未知阶段 {stage!r}，可选 {sorted(CODE_MODULES)}")
    parts: dict[str, str] = {}
    for f in _files_for(stage):
        try:
            parts[f.relative_to(_PKG_DIR).as_posix()] = digest(f.read_bytes().decode("utf-8", errors="replace"), 16)
        except OSError:
            parts[f.name] = ""
    return digest(parts)


def sandbox_identity(sandbox: Any) -> dict[str, Any]:
    """沙箱的身份信息（镜像标签、镜像 ID、资源上限）。没有沙箱返回空字典；
    沙箱对象没提供 ``identity()``（测试里的假沙箱）时只记类名。"""
    if sandbox is None:
        return {}
    fn = getattr(sandbox, "identity", None)
    if callable(fn):
        try:
            return dict(fn())
        except Exception as exc:  # noqa: BLE001 - 取不到身份不该让阶段失败，但要留痕
            log.warning("读取沙箱身份失败（%s: %s），按未知处理", type(exc).__name__, exc)
    return {"class": type(sandbox).__name__}


def sandbox_digest(sandbox: Any, *, full: bool = True) -> str:
    """沙箱身份的哈希。``full=False`` 只看工具链（镜像标签与镜像 ID），``True`` 再加上资源上限
    （内存上限会决定 OOM 归因，CPU 数影响超时，所以评测与 oracle gate 用 full）。"""
    ident: Mapping[str, Any] = sandbox_identity(sandbox)
    if not full:
        ident = {k: ident[k] for k in TOOLCHAIN_KEYS if k in ident}
    return digest(dict(ident))
