"""混淆总入口（设计文档 §4.6）：L0 原样复制、L1 符号、L2 = L1 + 结构、L3 = L2 + 语义。

``obfuscate(spec, pristine_root, level, seed, out_root)`` 把原始源码树复制到 ``out_root``，按级别依次施加变换，
并在 Docker 沙箱里跑 ``verify``（项目自带测试全过 + 每个目标函数能按映射后的名字重新定位）。
每级的产出是一个 :class:`ObfResult`，同时把 ``rename_map`` 与 ``stats``、verify 结果、混淆副本的源码树哈希
写进 ``results/obf-<corpus>-<level>.json``（设计文档 §5）。

级别与变换的关系（高一级 = 低一级 + 一类新变换，所以 L3 的副本里三类变换都在）::

    L0  只复制
    L1  复制 + 符号混淆（symbols）
    L2  L1 + 结构混淆（structure）
    L3  L2 + 语义混淆（semantic）

三类变换都只由 ``seed`` 决定（同种子同结果），所以 L2 / L3 不必在 L1 的产物上增量做，每级都从原始树重做一遍，
这样每个副本都能单独复现。

符号混淆遇到字符串化宏（宏体含 ``#``）时，默认先用非 strict 模式，验证不过且确有名字出现在 ``#`` 宏的调用实参里
（字符串化的输出被改了），自动退回 ``strict_stringify=True`` 重做一遍；每次尝试都记进 ``stats["attempts"]``。
也可以用 ``strict_stringify`` 参数固定使用某一种模式。

幂等：``results/obf-<corpus>-<level>.json`` 里记了输入哈希（spec、原始树、级别、种子、strict 设置、目标集、版本）。
输出目录里的源码树哈希也对得上、且上次 verify 通过（或没跑 verify）时直接返回上次的结果（``skipped=True``）；
上次 verify 没过的会重做一次，免得环境抖动留下一个永远作废的副本。

沙箱不可用时传 ``verify=False``；默认 ``verify=True``，没传 sandbox 就用默认参数建一个 DockerSandbox。
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

from bench import clangcfg, corpus
from bench.provenance import code_digest, sandbox_digest
from bench.vocab import LEVELS as _LEVELS
from bench.store import (
    SCHEMA_VERSION, Workspace, digest, now_iso, read_json, spec_digest, write_json,
)

log = logging.getLogger(__name__)

__all__ = ["LEVELS", "OBF_VERSION", "ObfResult", "obfuscate", "obf_inputs", "load_obf_result"]

LEVELS = _LEVELS                       # L0 原样；L1 符号；L2 = L1 + 结构；L3 = L2 + 语义

#: 混淆流程的版本号：改变了任何级别的产出语义时加一，让旧的结果文件失效
OBF_VERSION = 1


@dataclass
class ObfResult:
    level: str
    out_root: Path
    rename_map: dict[str, str]
    stats: dict
    # 以下为文档接口之外的附加字段，都带默认值
    verify: Optional[dict] = None       # VerifyResult.to_dict()；没跑 verify 时为 None
    tree_hash: str = ""                 # out_root 的源码树内容哈希
    input_hash: str = ""
    strict_stringify: bool = False      # 最终采用的符号混淆模式（L0 恒为 False）
    skipped: bool = False               # True 表示沿用上次的结果，这次没有重做
    result_path: Optional[Path] = None  # results/obf-<corpus>-<level>.json；没写文件时为 None
    elapsed_s: float = 0.0
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """副本可用：没跑 verify，或 verify 通过。"""
        return self.verify is None or bool(self.verify.get("ok"))


def _ident(target: Any) -> tuple[str, str]:
    """目标的 (target_id, 函数名)；FunctionTarget 与同字段的 dict 都认。"""
    get = target.get if isinstance(target, dict) else (lambda k, d=None: getattr(target, k, d))
    name = str(get("name"))
    return str(get("target_id") or f"{get('file')}:{name}"), name


def _target_ids(targets: Sequence[Any]) -> list[str]:
    return sorted(_ident(t)[0] for t in targets)


def obf_inputs(spec: corpus.CorpusSpec, pristine_root: Path, level: str, seed: Any, strict_stringify: Optional[bool],
               targets: Optional[Sequence[Any]], *, sandbox: Any = None) -> dict:
    """决定一个混淆副本内容的全部输入（的哈希），幂等判断和 report 校验都用它。

    除了 spec、原始树、级别、种子之外还带：混淆代码本身的内容哈希（``code``，改了变换逻辑旧副本就作废）、
    宿主 libclang 的环境摘要（``clang``，改名表的排除集依赖宿主机的系统头，见 :func:`bench.clangcfg.environment`）、
    以及 verify 用的沙箱工具链（``env``，镜像 ID）。
    """
    return {
        "spec": spec_digest(spec),
        "pristine": corpus.tree_digest(Path(pristine_root))[:16],
        "level": level,
        "seed": seed,
        "strict": strict_stringify,
        "targets": digest(_target_ids(targets)) if targets is not None else None,
        "version": OBF_VERSION,
        "code": code_digest("obfuscate"),
        "clang": clangcfg.environment() if level != "L0" else None,
        "env": sandbox_digest(sandbox, full=False) if sandbox is not None else None,
    }


def load_obf_result(path: Path) -> Optional[dict]:
    """读 results/obf-*.json，文件不存在或损坏返回 None。"""
    data = read_json(path)
    return data if isinstance(data, dict) else None


def _default_result_path(spec: corpus.CorpusSpec, level: str, results_dir: Optional[Path]) -> Path:
    base = Workspace.default().results if results_dir is None else Path(results_dir)
    return base / f"obf-{spec.name}-{level}.json"


def _from_saved(saved: dict, out_root: Path, level: str, path: Optional[Path]) -> ObfResult:
    return ObfResult(
        level=level, out_root=out_root, rename_map=dict(saved.get("rename_map") or {}),
        stats=dict(saved.get("stats") or {}), verify=saved.get("verify"), tree_hash=saved.get("tree_hash", ""),
        input_hash=saved.get("input_hash", ""), strict_stringify=bool(saved.get("strict_stringify")),
        skipped=True, result_path=path, elapsed_s=float(saved.get("elapsed_s") or 0.0),
        problems=list(saved.get("problems") or []))


def obfuscate(spec: corpus.CorpusSpec, pristine_root: Path, level: str, seed: int, out_root: Path, *,
              sandbox: Any = None, targets: Optional[Sequence[Any]] = None, verify: bool = True,
              strict_stringify: Optional[bool] = None, results_dir: Optional[Path] = None,
              write_result: bool = True, force: bool = False, work_root: Optional[Path] = None) -> ObfResult:
    """生成 ``level`` 级混淆副本到 ``out_root``，返回 :class:`ObfResult`。

    ``spec``：CorpusSpec；``pristine_root``：原始源码树（不会被改动）；``out_root``：副本目录，已存在会被清空重建。
    其余是文档签名之外的可选关键字参数：

    * ``verify`` / ``sandbox`` / ``targets`` / ``work_root``：是否在沙箱里验证、用哪个沙箱、验证时要求能重新定位的目标
      （原始树上提取出的 FunctionTarget；不给就用 ``extract_functions`` 提取到的全部候选）、验证临时目录；
    * ``strict_stringify``：None 表示先非 strict、必要时自动退回 strict；True / False 固定一种；
    * ``results_dir`` / ``write_result``：结果文件的目录（默认 ``Workspace.default().results``）、是否写文件；
    * ``force``：忽略上次的结果强制重做。
    """
    if level not in LEVELS:
        raise ValueError(f"未知混淆级别 {level!r}，可选 {LEVELS}")
    pristine_root, out_root = Path(pristine_root), Path(out_root)
    if not pristine_root.is_dir():
        raise FileNotFoundError(f"原始源码目录不存在: {pristine_root}")

    # 延迟导入：这几个模块会拉起 libclang，只做 L0 复制或只读结果文件的场景不需要它们
    from bench.extract import extract_functions
    from bench.obfuscate import semantic, structure, symbols
    from bench.obfuscate import verify as verify_mod

    if verify and targets is None:
        targets = extract_functions(spec, pristine_root)
    result_path = _default_result_path(spec, level, results_dir) if write_result else None
    if verify and sandbox is None:
        from bench.sandbox import DockerSandbox
        sandbox = DockerSandbox()
    inputs = obf_inputs(spec, pristine_root, level, seed, strict_stringify, targets if verify else None,
                        sandbox=sandbox if verify else None)
    input_hash = digest(inputs)

    if not force and result_path is not None:
        saved = load_obf_result(result_path)
        if (saved and saved.get("input_hash") == input_hash and out_root.is_dir()
                and corpus.tree_digest(out_root)[:16] == saved.get("tree_hash")
                and (saved.get("verify") is None or saved["verify"].get("ok"))):
            log.info("%s %s: 输入未变且副本完好，沿用上次的结果", spec.name, level)
            return _from_saved(saved, out_root, level, result_path)

    t0 = time.monotonic()
    modes: list[bool] = [False, True] if strict_stringify is None else [bool(strict_stringify)]
    attempts: list[dict[str, Any]] = []
    mapping: dict[str, str] = {}
    stats: dict[str, Any] = {}
    vr: Optional[dict] = None
    used_strict = False
    for strict in modes:
        corpus.materialize(pristine_root, out_root)
        mapping, stats = {}, {"level": level, "seed": seed}
        if level == "L0":
            used_strict = False
        else:
            sym = symbols.obfuscate_symbols(spec, out_root, seed, strict_stringify=strict)
            mapping = dict(sym.mapping)
            stats["symbols"] = sym.stats
            used_strict = strict
            if level in ("L2", "L3"):
                stats["structure"] = structure.transform_structure(spec, out_root, seed)
            if level == "L3":
                stats["semantic"] = semantic.transform_semantic(spec, out_root, seed)
            # 目标函数名没被改掉（被排除集挡住）的，提示词里还是真名；调用方据此作废这些目标
            stats["targets_not_renamed"] = sorted(
                tid for tid, name in map(_ident, targets or []) if name not in mapping)
        attempt: dict[str, Any] = {"strict_stringify": strict if level != "L0" else None}
        if verify:
            res = verify_mod.verify(spec, out_root, targets or [], mapping, sandbox, work_root=work_root)
            vr = res.to_dict()
            attempt.update(ok=res.ok, build_rc=res.build_rc, missing=len(res.missing))
        attempts.append(attempt)
        if vr is None or vr.get("ok"):
            break
        if level == "L0" or strict:
            break
        if strict_stringify is None and not stats["symbols"].get("stringified_names_renamed"):
            break                                    # strict 不会改变任何东西，没必要重试
        log.warning("%s %s: 非 strict 验证失败，改用 strict_stringify 重做", spec.name, level)
    stats["attempts"] = attempts

    tree_hash = corpus.tree_digest(out_root)[:16]
    problems: list[str] = []
    if vr is not None and not vr.get("ok"):
        why = "项目自带测试未通过" if not vr.get("tests_passed") else f"{len(vr.get('missing') or [])} 个目标无法按映射名重新定位"
        problems.append(f"verify 未通过：{why}")
    result = ObfResult(level=level, out_root=out_root, rename_map=mapping, stats=stats, verify=vr,
                       tree_hash=tree_hash, input_hash=input_hash, strict_stringify=used_strict,
                       skipped=False, result_path=result_path, elapsed_s=round(time.monotonic() - t0, 2),
                       problems=problems)
    if result_path is not None:
        write_json(result_path, {
            "schema_version": SCHEMA_VERSION,
            "corpus": spec.name,
            "level": level,
            "seed": seed,
            "created": now_iso(),
            "inputs": inputs,
            "input_hash": input_hash,
            "strict_stringify": used_strict,
            "tree_hash": tree_hash,
            "ok": result.ok,
            "problems": problems,
            "elapsed_s": result.elapsed_s,
            "rename_map": mapping,
            "stats": stats,
            "verify": vr,
            "oracle_gate": None,       # 由 verify 阶段补上（pipeline.stage_verify）
        })
    log.info("%s %s: 完成（verify=%s，替换 %s 个名字，%.1fs）", spec.name, level,
             "未跑" if vr is None else ("通过" if result.ok else "未通过"), len(mapping), result.elapsed_s)
    return result
