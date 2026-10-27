"""bench.experiment_record（docs/实验记录.md 的数据部分由它从 results/ 生成）的单元测试。不需要 docker。"""
from __future__ import annotations

import shutil

import pytest

from bench import corpus, experiment_record as er
from bench.store import Workspace, write_json, write_jsonl
from bench.vocab import LEVELS
from tests.helpers import MINI, REQUIRE_CORPUS, ROOT

SANDBOX = {"image": "gcc:13", "image_id": "sha256:abc", "cpus": 1.0, "memory": "1g", "pids": 256, "timeout_s": 120}


def write_stage_files(ws: Workspace, name: str = "mini", *, attempts: int = 1, gate_failed=(), copy_invalid: bool = False) -> None:
    write_json(ws.prepare_file(name), {"ok": True, "commit": "", "duration_s": 1.5, "inputs": {"tree": "tree0", "code": "codeP"}})
    write_json(ws.coverage_file(name), {"n_gcda": 3, "n_gcda_failed": 0, "n_functions": 41, "n_executed": 40,
                                        "inputs": {"code": "codeC"}})
    write_json(ws.targets_summary_file(name), {
        "seed": 7, "extracted": 40, "candidates": 27, "selected": 20, "sandbox": SANDBOX, "inputs": {"code": "codeT"},
        "excluded": {"few_statements": 13, "stub_survived": 4, "not_sampled": 3},
        "stub_kill_modes": {"test_fail": 18, "runtime_crash": 2}})
    write_jsonl(ws.targets_file(name), [{"target_id": f"{name}:calc.c:calc_gcd", "name": "calc_gcd"}])
    for lv in LEVELS:
        rec = {"level": lv, "strict_stringify": lv == "L2", "ok": not copy_invalid,
               "inputs": {"code": "codeO", "clang": {"python_binding": "14.0", "library": "libclang.so.1", "builtin_include": "inc0"}},
               "verify": {"ok": not copy_invalid, "relocated": 20, "n_targets": 20},
               "oracle_gate": {"checked": 20, "passed": 20 - len(gate_failed), "copy_invalid": copy_invalid,
                               "failed": [{"target_id": t, "stage": "build"} for t in gate_failed],
                               "inputs": {"code": "codeG"}},
               "problems": [], "stats": {"attempts": [{"ok": True}] * attempts, "targets_not_renamed": []}}
        if lv != "L0":
            rec["stats"]["symbols"] = {"entities_renamed": 130, "entities_excluded": 4, "stringified_names_renamed": ["a", "b", "c"]}
        if lv in ("L2", "L3"):
            rec["stats"]["structure"] = {"total": 12}
        if lv == "L3":
            rec["stats"]["semantic"] = {"total": 81}
        write_json(ws.obf_file(name, lv), rec)


@pytest.fixture()
def ws(scratch, monkeypatch):
    specs = scratch / "specs"
    specs.mkdir()
    shutil.copy(MINI / "spec.json", specs / "mini.json")
    monkeypatch.setattr(corpus, "SPEC_DIR", specs)
    w = Workspace.under(scratch).ensure()
    shutil.copytree(MINI, w.pristine("mini"))
    return w


class TestRender:
    def test_every_stage_is_summarised(self, ws):
        write_stage_files(ws)
        md = er.render(ws, ["mini"])
        assert "| 随机种子 | `7` |" in md and "| 镜像 ID | `sha256:abc` |" in md
        assert "| 代码摘要：targets | `codeT` |" in md and "| 代码摘要：oracle gate | `codeG` |" in md
        assert "14.0 / libclang.so.1 / inc0" in md
        assert "| mini | - | `tree0` | 通过 | 1.5 | 3 | 0 | 41 | 40 |" in md
        row = next(ln for ln in md.splitlines() if ln.startswith("| mini | 40 | 27 | 20 |"))
        assert "| 13 | 0 | 0 | 0 | 0 | 4 | 0 | 0 | 3 | test_fail=18, runtime_crash=2 |" in row
        assert any(ln.startswith("| **合计** | 40 | 27 | 20 |") for ln in md.splitlines())
        assert "| mini | L1 | 130 | 4 | 否 | 3 | - | - | 通过（定位 20/20） | 20/20 |" in md
        assert "| mini | L3 | 130 | 4 | 否 | 3 | 12 | 81 | 通过（定位 20/20） | 20/20 |" in md
        assert "没有：所有副本 verify 通过" in md

    def test_problems_are_called_out(self, ws):
        write_stage_files(ws, attempts=2, gate_failed=("mini:calc.c:calc_gcd",))
        md = er.render(ws, ["mini"])
        notes = md.split("### 需要注意的事")[1].split("###")[0]
        assert "先用非 strict，verify 未过，退回 strict（共 2 次尝试）" in notes
        assert "目标 `mini:calc.c:calc_gcd` 没过 oracle gate（build）" in notes
        assert "没有：" not in notes

    def test_invalid_copy_is_reported(self, ws):
        write_stage_files(ws, copy_invalid=True)
        notes = er.render(ws, ["mini"]).split("### 需要注意的事")[1]
        assert "副本 verify 没过，整级作废" in notes

    def test_missing_stage_files_show_up_as_missing_not_as_zero(self, ws):
        md = er.render(ws, ["mini"])
        assert "| mini | 缺 |" in md or "| mini | 缺 | 缺 |" in md
        assert "| mini | L1 | 缺 |" in md

    def test_exemption_audit_counts_the_checker_view_of_the_project(self, ws):
        write_stage_files(ws)
        audit = er.render(ws, ["mini"]).split("### 泄漏检查的放行情况")[1]
        row = next(ln for ln in audit.splitlines() if ln.startswith("| mini |"))
        cells = [c.strip() for c in row.strip("|").split("|")]
        assert int(cells[1]) > 0 and 0 <= int(cells[2]) <= int(cells[1])


class TestUpdateDocument:
    def test_only_the_generated_region_is_replaced(self, scratch):
        doc = scratch / "log.md"
        doc.write_text(f"# 手写的开头\n\n{er.BEGIN}\n旧内容\n{er.END}\n\n手写的结尾\n", encoding="utf-8")
        er.update_document(doc, "新内容\n")
        text = doc.read_text(encoding="utf-8")
        assert text.startswith("# 手写的开头") and text.endswith("手写的结尾\n")
        assert "新内容" in text and "旧内容" not in text
        er.update_document(doc, "新内容\n")                              # 幂等
        assert doc.read_text(encoding="utf-8") == text

    def test_missing_markers_are_an_error(self, scratch):
        doc = scratch / "log.md"
        doc.write_text("没有标记\n", encoding="utf-8")
        with pytest.raises(SystemExit):
            er.update_document(doc, "x")
        assert doc.read_text(encoding="utf-8") == "没有标记\n"


class TestMain:
    def test_prints_to_stdout_or_updates_a_file(self, ws, scratch, capsys):
        write_stage_files(ws)
        base = ["--workspace", str(scratch), "--corpus", "mini"]
        assert er.main(base) == 0
        assert "### 混淆副本与 oracle gate" in capsys.readouterr().out
        doc = scratch / "log.md"
        doc.write_text(f"头\n{er.BEGIN}\n{er.END}\n尾\n", encoding="utf-8")
        assert er.main([*base, "--update", str(doc)]) == 0
        text = doc.read_text(encoding="utf-8")
        assert text.startswith("头\n") and text.endswith("尾\n") and "### 环境与来源" in text


class TestCommittedRecord:
    def test_the_generated_region_matches_the_results_it_was_made_from(self):
        """docs/实验记录.md 的生成区必须等于现在从 results/ 渲染出来的内容：重跑了实验没更新文档（或反过来）就失败。
        更新办法：python3 -m bench.experiment_record --update docs/实验记录.md"""
        ws = Workspace.default()
        names = corpus.list_specs()
        if not all(ws.targets_summary_file(n).is_file() for n in names):
            msg = "results/ 里没有 9 个语料完整的 prepare 到 verify 产出（先跑 python3 -m bench prepare coverage targets obfuscate verify）"
            if REQUIRE_CORPUS:
                pytest.fail(msg, pytrace=False)
            pytest.skip(msg)
        text = (ROOT / "docs" / "实验记录.md").read_text(encoding="utf-8")
        body = text.split(er.BEGIN, 1)[1].split(er.END, 1)[0]
        assert body.strip() == er.render(ws, names).strip()
