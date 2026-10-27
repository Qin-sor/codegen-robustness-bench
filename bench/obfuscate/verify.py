"""混淆副本的验证（oracle gate 的一部分，设计文档 §4.6 / §6.1）。

``verify`` 做两件事，两件都过才算 ok：

1. 把副本复制到临时工作目录，在 Docker 沙箱里跑 ``spec.build_cmd`` / ``test_cmd``，按
   ``pass_criteria`` 判定项目自带测试全部通过（语料代码只在沙箱里执行，宿主机不 make）；
2. 对每个目标函数，用 :func:`bench.extract.locate` 按「映射后的名字」在副本里重新定位，
   找不到就记入 ``missing``。

临时工作目录用 ``store.default_work_root()``（默认 ``<项目根>/tmp/work``，环境变量 ``BENCH_WORK_DIR`` 可改），验证完即删。
L1 的「非 strict 验证失败后退回 strict」逻辑只在 ``bench.obfuscate.obfuscate`` 里有一份。
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

from bench import corpus
from bench.corpus import CorpusSpec
from bench.extract import ExtractError, locate

log = logging.getLogger(__name__)

__all__ = ["VerifyResult", "verify", "relocate_targets"]

#: stdout / stderr 保留的尾部字符数
TAIL_CHARS = 2000


@dataclass
class VerifyResult:
    ok: bool                    # 测试通过且全部目标都能重新定位
    build_rc: int               # build_cmd（及 test_cmd）的退出码
    stdout_tail: str            # 沙箱输出的尾部；没通过时后面接上 stderr 尾部
    relocated: int              # 按映射名重新定位成功的目标数
    missing: list[str] = field(default_factory=list)   # 找不到的目标（target_id）
    # 以下为附加字段，有默认值，不影响按设计文档的五个字段构造
    tests_passed: bool = False  # pass_criteria 是否满足（与目标定位无关）
    timed_out: bool = False
    duration_s: float = 0.0
    n_targets: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _field(target: Any, name: str, default: Any = None) -> Any:
    if isinstance(target, dict):
        return target.get(name, default)
    return getattr(target, name, default)


def relocate_targets(spec: CorpusSpec, obf_root: Path, targets: Sequence[Any], mapping: dict[str, str]
                     ) -> tuple[int, list[str]]:
    """用 ``mapping`` 换算每个目标在副本里的名字，再用 extract.locate 重新定位。

    ``targets`` 是原始（未混淆）树上提取出的 FunctionTarget（也接受同字段的 dict）。
    返回（定位成功数，找不到的 target_id 列表）。
    """
    relocated = 0
    missing: list[str] = []
    for t in targets:
        name = _field(t, "name")
        file = _field(t, "file")
        tid = _field(t, "target_id") or f"{file}:{name}"
        new_name = mapping.get(name, name)
        try:
            found = locate(spec, obf_root, new_name, file)
        except (ExtractError, OSError) as exc:
            log.warning("定位 %s（%s）时出错: %s", tid, new_name, exc)
            found = None
        if found is None:
            missing.append(tid)
        else:
            relocated += 1
    return relocated, missing


def verify(spec: CorpusSpec, obf_root: Path, targets: Sequence[Any], mapping: dict[str, str], sandbox: Any,
           *, work_root: Optional[Path] = None) -> VerifyResult:
    """项目测试全过 + 每个目标函数能按映射名重新定位。

    ``obf_root`` 不会被改动：构建在它的副本里跑，跑完删除。沙箱超时、构建失败、测试不过都体现在返回值里，不抛异常；
    只有 docker 本身出错（镜像不存在等）才会抛。
    """
    obf_root = Path(obf_root)
    tests_ok, run = corpus.verify_build(spec, obf_root, sandbox, work_root=work_root)
    relocated, missing = relocate_targets(spec, obf_root, targets, mapping)
    tail = run.stdout[-TAIL_CHARS:]
    if not tests_ok and run.stderr:
        tail += "\n--- stderr ---\n" + run.stderr[-TAIL_CHARS:]
    ok = tests_ok and not missing
    log.info("verify %s: ok=%s rc=%s relocated=%d/%d missing=%d %.1fs", getattr(spec, "name", "?"), ok, run.rc,
             relocated, len(targets), len(missing), run.duration_s)
    return VerifyResult(ok=ok, build_rc=run.rc, stdout_tail=tail, relocated=relocated, missing=missing,
                        tests_passed=tests_ok, timed_out=run.timed_out, duration_s=run.duration_s,
                        n_targets=len(targets))
