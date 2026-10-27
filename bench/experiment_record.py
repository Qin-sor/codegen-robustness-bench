"""实验记录生成：把真实语料的 prepare → coverage → targets → obfuscate → verify 产出汇总成 Markdown。

``docs/实验记录.md`` 的数据部分（环境来源、各阶段的表）由本模块从工作区的 ``results/*.json`` 生成，手写的说明和观察放在标记之外，
重新生成时只替换标记之间的内容::

    python3 -m bench.experiment_record --update docs/实验记录.md      # 读默认工作区的 results/，替换文件里的生成区
    python3 -m bench.experiment_record --workspace DIR --corpus cjson sds   # 只打印到标准输出

只读数据文件（和重新计算泄漏检查口径下的放行情况需要的源码树），不碰沙箱、不联网、不调用模型。
没有任何模型参与，所以这里没有通过率之类的数字；那些由 ``python3 -m bench report`` 产出。
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from bench import corpus
from bench.store import Workspace, read_json, read_jsonl
from bench.vocab import LEVELS

log = logging.getLogger(__name__)

__all__ = ["BEGIN", "END", "collect", "render", "update_document", "main"]

BEGIN = "<!-- BEGIN GENERATED: python3 -m bench.experiment_record -->"
END = "<!-- END GENERATED -->"

#: targets 汇总里的排除原因 → 表里的列（顺序即列序）
_EXCLUDE_COLUMNS = (
    ("few_statements", "语句数不足"), ("too_short", "行数过短"), ("too_long", "行数过长"), ("spec_excluded", "spec 排除"),
    ("not_executed", "未被测试执行"), ("stub_survived", "stub 存活"), ("stub_build_error", "桩无法构建"),
    ("stub_unsupported", "返回类型无法造桩"), ("not_sampled", "名额已满未入选"),
)
_KILL_MODES = ("test_fail", "runtime_crash", "runtime_timeout")


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]], align: Optional[Sequence[str]] = None) -> list[str]:
    """Markdown 表格；align 给每列 'l' 或 'r'（默认第一列左对齐，其余右对齐）。"""
    align = align or ["l"] + ["r"] * (len(headers) - 1)
    sep = [":---:" if a == "c" else ("---:" if a == "r" else "---") for a in align]
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(sep) + " |"]
    out += ["| " + " | ".join(str(x) for x in row) + " |" for row in rows]
    return out


def _get(d: Optional[Mapping[str, Any]], *path: str, default: Any = None) -> Any:
    cur: Any = d
    for key in path:
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur


def collect(ws: Workspace, corpora: Sequence[str]) -> dict[str, dict[str, Any]]:
    """读各语料的阶段产出；缺的文件记为 None / 空。"""
    data: dict[str, dict[str, Any]] = {}
    for name in corpora:
        data[name] = {
            "prepare": read_json(ws.prepare_file(name)),
            "coverage": read_json(ws.coverage_file(name)),
            "summary": read_json(ws.targets_summary_file(name)),
            "targets": read_jsonl(ws.targets_file(name)),
            "obf": {lv: read_json(ws.obf_file(name, lv)) for lv in LEVELS},
        }
    return data


def _distinct(values: Sequence[Any]) -> str:
    """一组值去重后拼起来；全一样时就是那个值，不一样时都列出来（读者一眼能看到来源不一致）。"""
    seen: list[str] = []
    for v in values:
        if v not in (None, "") and str(v) not in seen:
            seen.append(str(v))
    return " / ".join(f"`{v}`" for v in seen) if seen else "-"


def _provenance(data: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """环境与来源：种子、沙箱镜像、宿主 libclang 环境、各阶段的评测相关代码摘要。"""
    summaries = [d["summary"] for d in data.values() if d["summary"]]
    obfs = [r for d in data.values() for r in d["obf"].values() if r]
    gates = [r["oracle_gate"] for r in obfs if r.get("oracle_gate")]
    rows = [
        ("随机种子", _distinct([s.get("seed") for s in summaries])),
        ("沙箱镜像", _distinct([_get(s, "sandbox", "image") for s in summaries])),
        ("镜像 ID", _distinct([_get(s, "sandbox", "image_id") for s in summaries])),
        ("沙箱资源", _distinct([f"cpus={_get(s, 'sandbox', 'cpus')} memory={_get(s, 'sandbox', 'memory')} "
                              f"pids={_get(s, 'sandbox', 'pids')}" for s in summaries])),
        ("宿主 libclang（python 绑定 / 库文件 / 内置头摘要）",
         _distinct([f"{_get(r, 'inputs', 'clang', 'python_binding')} / {_get(r, 'inputs', 'clang', 'library')} / "
                    f"{_get(r, 'inputs', 'clang', 'builtin_include') or '-'}" for r in obfs if r["level"] != "L0"])),
        ("代码摘要：prepare", _distinct([_get(d["prepare"], "inputs", "code") for d in data.values()])),
        ("代码摘要：coverage", _distinct([_get(d["coverage"], "inputs", "code") for d in data.values()])),
        ("代码摘要：targets", _distinct([_get(s, "inputs", "code") for s in summaries])),
        ("代码摘要：obfuscate", _distinct([_get(r, "inputs", "code") for r in obfs])),
        ("代码摘要：oracle gate", _distinct([_get(g, "inputs", "code") for g in gates])),
    ]
    return _table(["项", "值"], rows, ["l", "l"])


def _prepare_coverage(data: Mapping[str, Mapping[str, Any]], specs: Mapping[str, Any]) -> list[str]:
    rows = []
    for name, d in data.items():
        p, c = d["prepare"], d["coverage"]
        spec = specs.get(name)
        commit = (spec.commit if spec else (p or {}).get("commit")) or ""
        rows.append((
            name, f"`{commit[:12]}`" if commit else "-",
            f"`{_get(p, 'inputs', 'tree', default='-')}`",
            "通过" if p and p.get("ok") else ("未通过" if p else "缺"),
            f"{p.get('duration_s', 0):.1f}" if p else "-",
            _get(c, "n_gcda", default="-"), _get(c, "n_gcda_failed", default="-"),
            _get(c, "n_functions", default="-"), _get(c, "n_executed", default="-"),
        ))
    return _table(["语料", "commit", "源码树哈希", "沙箱里构建与测试", "耗时(s)", ".gcda 数", "gcov 失败", "函数数", "执行过的函数数"], rows,
                  ["l", "l", "l", "l"] + ["r"] * 5)


def _targets(data: Mapping[str, Mapping[str, Any]]) -> list[str]:
    rows = []
    total = {k: 0 for k in ("extracted", "candidates", "selected")}
    total_ex = {k: 0 for k, _ in _EXCLUDE_COLUMNS}
    total_modes = {k: 0 for k in _KILL_MODES}
    for name, d in data.items():
        s = d["summary"]
        if not s:
            rows.append((name,) + ("缺",) * (3 + len(_EXCLUDE_COLUMNS) + 1))
            continue
        ex = s.get("excluded") or {}
        modes = s.get("stub_kill_modes") or {}
        for k in total:
            total[k] += int(s.get(k) or 0)
        for k, _ in _EXCLUDE_COLUMNS:
            total_ex[k] += int(ex.get(k, 0))
        for k in _KILL_MODES:
            total_modes[k] += int(modes.get(k, 0))
        rows.append((name, s.get("extracted"), s.get("candidates"), s.get("selected"),
                     *[ex.get(k, 0) for k, _ in _EXCLUDE_COLUMNS],
                     ", ".join(f"{k}={modes[k]}" for k in _KILL_MODES if modes.get(k)) or "-"))
    rows.append(("**合计**", total["extracted"], total["candidates"], total["selected"],
                 *[total_ex[k] for k, _ in _EXCLUDE_COLUMNS],
                 ", ".join(f"{k}={total_modes[k]}" for k in _KILL_MODES if total_modes[k]) or "-"))
    heads = ["语料", "提取函数数", "候选", "入选"] + [label for _, label in _EXCLUDE_COLUMNS] + ["入选目标的杀死方式"]
    return _table(heads, rows, ["l"] + ["r"] * (len(heads) - 2) + ["l"])


def _obfuscation(data: Mapping[str, Mapping[str, Any]]) -> list[str]:
    rows = []
    for name, d in data.items():
        for lv in LEVELS:
            r = d["obf"][lv]
            if not r:
                rows.append((name, lv) + ("缺",) * 9)
                continue
            sym = _get(r, "stats", "symbols", default={}) or {}
            gate = r.get("oracle_gate") or {}
            ver = r.get("verify") or {}
            renamed = sym.get("entities_renamed")
            rows.append((
                name, lv,
                renamed if lv != "L0" else "-",
                sym.get("entities_excluded", "-") if lv != "L0" else "-",
                ("是" if r.get("strict_stringify") else "否") if lv != "L0" else "-",
                len(sym.get("stringified_names_renamed") or []) if lv != "L0" else "-",
                _get(r, "stats", "structure", "total", default="-"),
                _get(r, "stats", "semantic", "total", default="-"),
                ("通过" if ver.get("ok") else "未通过") + f"（定位 {ver.get('relocated', '-')}/{ver.get('n_targets', '-')}）" if ver else "-",
                f"{gate.get('passed', '-')}/{gate.get('checked', '-')}" if gate else "-",
            ))
    heads = ["语料", "级别", "改名数", "未改名的项目拼写", "字符串化宏用 strict", "字符串化实参被改名数", "结构变换次数", "语义变换次数",
             "副本 verify", "oracle gate 通过/检查"]
    return _table(heads, rows, ["l", "l"] + ["r"] * 6 + ["l", "r"])


def _problems(data: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """需要读者注意的事：回退到 strict 的、gate 没过的目标、副本 verify 的问题。"""
    notes: list[str] = []
    for name, d in data.items():
        for lv in LEVELS:
            r = d["obf"][lv]
            if not r:
                continue
            attempts = _get(r, "stats", "attempts", default=[]) or []
            if len(attempts) > 1:
                notes.append(f"- {name}@{lv}：符号混淆先用非 strict，verify 未过，退回 strict（共 {len(attempts)} 次尝试）。")
            for prob in r.get("problems") or []:
                notes.append(f"- {name}@{lv}：{prob}")
            gate = r.get("oracle_gate") or {}
            if gate.get("copy_invalid"):
                notes.append(f"- {name}@{lv}：副本 verify 没过，整级作废。")
            for f in gate.get("failed") or []:
                notes.append(f"- {name}@{lv}：目标 `{f.get('target_id')}` 没过 oracle gate（{f.get('stage', '?')}）。")
            not_renamed = _get(r, "stats", "targets_not_renamed", default=[]) or []
            if not_renamed:
                notes.append(f"- {name}@{lv}：{len(not_renamed)} 个目标函数名没被改掉（出题时作废）：{', '.join(not_renamed[:6])}。")
    return notes or ["- 没有：所有副本 verify 通过、所有目标通过 oracle gate、没有回退到 strict。"]


def _exempt_audit(data: Mapping[str, Mapping[str, Any]], ws: Workspace, specs: Mapping[str, Any]) -> list[str]:
    """泄漏检查口径下的放行情况：项目标识符里有多少被常见英文词表 / spec.allow_words 放行，入选目标自己的函数名有多少在其中。"""
    from bench.describe import LeakChecker
    from bench.pipeline import project_names
    rows = []
    for name, d in data.items():
        spec = specs.get(name)
        root = ws.pristine(name)
        if spec is None or not root.is_dir():
            rows.append((name, "缺", "-", "-", "-", "-"))
            continue
        checker = LeakChecker(project_names(spec, root), spec.allow_words)
        own = sorted({t["name"] for t in d["targets"] if t.get("name") in checker.exempt})
        allow = sorted(n for n in checker.exempt if n.casefold() in checker.allow_words)
        rows.append((name, len(checker.names), len(checker.exempt), len(allow), len(own), ", ".join(own) or "-"))
    return _table(["语料", "参与检查的项目标识符（≥3 字符）", "被放行", "其中靠 spec.allow_words", "入选目标自己的函数名被放行", "这些函数名"],
                  rows, ["l", "r", "r", "r", "r", "l"])


def render(ws: Workspace, corpora: Sequence[str]) -> str:
    """生成区的 Markdown（不含标记行）。"""
    data = collect(ws, corpora)
    specs: dict[str, Any] = {}
    for name in corpora:
        try:
            specs[name] = corpus.load_spec(name)
        except (FileNotFoundError, corpus.SpecError):
            log.warning("找不到语料 %s 的 spec", name)
    out: list[str] = ["### 环境与来源", ""] + _provenance(data)
    out += ["", "### prepare 与 coverage", ""] + _prepare_coverage(data, specs)
    out += ["", "### 目标选择与 stub-kill", ""] + _targets(data)
    out += ["", "### 混淆副本与 oracle gate", ""] + _obfuscation(data)
    out += ["", "### 需要注意的事", ""] + _problems(data)
    out += ["", "### 泄漏检查的放行情况", ""] + _exempt_audit(data, ws, specs)
    return "\n".join(out) + "\n"


def update_document(path: Path, generated: str) -> None:
    """把文件里 BEGIN / END 两行标记之间的内容换成 generated；缺标记就报错（不猜位置）。"""
    text = Path(path).read_text(encoding="utf-8")
    try:
        head, rest = text.split(BEGIN, 1)
        _old, tail = rest.split(END, 1)
    except ValueError:
        raise SystemExit(f"{path} 里没有生成区标记：\n{BEGIN}\n{END}") from None
    Path(path).write_text(f"{head}{BEGIN}\n\n{generated}\n{END}{tail}", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m bench.experiment_record", description=__doc__.split("\n")[0])
    ap.add_argument("--workspace", metavar="DIR", help="工作区目录（默认用项目目录）")
    ap.add_argument("--corpus", nargs="+", metavar="NAME", help="语料名（默认 corpus/specs/ 下的全部）")
    ap.add_argument("--update", metavar="FILE", help="替换这个 Markdown 文件里标记之间的生成区；不给就打印到标准输出")
    args = ap.parse_args(argv)
    ws = Workspace.under(args.workspace) if args.workspace else Workspace.default()
    text = render(ws, args.corpus or corpus.list_specs())
    if args.update:
        update_document(Path(args.update), text)
        print(f"已更新 {args.update}")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
