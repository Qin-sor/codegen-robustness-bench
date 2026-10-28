"""工作区路径、JSON / JSONL 读写与输入哈希，流水线各阶段共用，只依赖标准库。

约定：

* 所有结果文件都带 ``schema_version``；阶段性产出（targets、obf、tasks 等）整份原子重写，
  需要边跑边记、可断点续跑的产出（generations、results）按行追加；
* 每条产出带 ``inputs``（各输入的内容哈希，人能读）与 ``input_hash``（它们的合并哈希）。
  幂等判断只看 ``input_hash``：输出存在且哈希与重新算出来的一致就跳过，设计文档 §2 / §6.5 / §6.6；
* 哈希统一用 sha256 的前 16 个十六进制字符（64 位）：足够区分同一工作区里的不同输入，读起来也不累。

目录布局由 :class:`Workspace` 决定。默认布局全部落在项目目录下，且都可以用环境变量覆盖：

==============  ==========================  ================================
目录            默认位置                    环境变量
==============  ==========================  ================================
results         ``<项目根>/results``        ``BENCH_RESULTS_DIR``
corpus_src      ``<项目根>/corpus/src``     ``BENCH_CORPUS_SRC``
obf             ``<项目根>/tmp/obf``        ``BENCH_OBF_DIR``
work            ``<项目根>/tmp/work``       ``BENCH_WORK_DIR``
==============  ==========================  ================================

``obf`` 与 ``work`` 都是可以由 seed 重新生成的中间产物，所以放在被忽略的 ``tmp/`` 下。
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

log = logging.getLogger(__name__)

__all__ = [
    "SCHEMA_VERSION",
    "PROJECT_ROOT",
    "default_work_root",
    "Workspace",
    "canonical_json",
    "digest",
    "spec_digest",
    "now_iso",
    "safe_name",
    "REPAIR_PREFIX",
    "atomic_write",
    "read_json",
    "write_json",
    "read_jsonl",
    "read_jsonl_checked",
    "append_jsonl",
    "write_jsonl",
]

SCHEMA_VERSION = 1

# 项目根：由本文件位置推出，不写死绝对路径；BENCH_ROOT 与 corpus.py / llm.py 的约定一致
PROJECT_ROOT = Path(os.environ.get("BENCH_ROOT") or Path(__file__).resolve().parent.parent)


def default_work_root() -> Path:
    """临时工作目录（评测、构建副本、验证副本都放这里）：``BENCH_WORK_DIR`` 优先，默认 ``<项目根>/tmp/work``。
    全项目只有这一个默认值；每次调用重新读环境变量。"""
    return Path(os.environ.get("BENCH_WORK_DIR") or PROJECT_ROOT / "tmp" / "work")


# ----------------------------------------------------------------------
# 哈希与时间
# ----------------------------------------------------------------------
def canonical_json(obj: Any) -> str:
    """键排序、无多余空白的 JSON，做哈希的规范形式（同一内容永远得到同一字符串）。"""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def digest(obj: Any, length: int = 16) -> str:
    """任意可 JSON 化对象的内容哈希（sha256 的前 length 个十六进制字符）。"""
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()[:length]


def spec_digest(spec: Any) -> str:
    """spec 的内容哈希：CorpusSpec 用它自带的 digest()（只含影响构建与测试的字段），dict 按全部字段算。"""
    fn = getattr(spec, "digest", None)
    if callable(fn):
        return str(fn())[:16]
    return digest(dict(spec))


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_name(text: str) -> str:
    """把模型名、tag 之类变成能安全放进文件名的形式。"""
    return _SAFE_RE.sub("_", text).strip("._") or "_"


#: 缺陷修复任务的文件名在生成任务的文件名前面加 ``repair-``（``results-repair-<tag>.jsonl``、``generations-repair-<model>.jsonl``……），
#: 生成任务的 tag / 模型名若也以它开头（``repair-x``），两类任务的文件就会撞名，一类的评测结果被另一类当成范围内的过期记录丢掉
REPAIR_PREFIX = "repair-"


def _plain_name(text: str, what: str) -> str:
    """生成任务的 tag / 模型名：``safe_name`` 之后不许以 ``repair-`` 开头（见 :data:`REPAIR_PREFIX`）。"""
    name = safe_name(text)
    if name.startswith(REPAIR_PREFIX):
        raise ValueError(f"{what} {text!r} 以 {REPAIR_PREFIX!r} 开头，会与缺陷修复任务的文件名撞名（results-repair-<tag>.jsonl 等），请换一个名字")
    return name


# ----------------------------------------------------------------------
# 工作区
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Workspace:
    """一次评测的目录布局。默认布局见模块文档；测试和 e2e 用 :meth:`under` 把一切放进一个临时目录。"""

    results: Path
    corpus_src: Path
    obf: Path
    work: Path

    @classmethod
    def default(cls) -> "Workspace":
        env = os.environ.get
        return cls(
            results=Path(env("BENCH_RESULTS_DIR") or PROJECT_ROOT / "results"),
            corpus_src=Path(env("BENCH_CORPUS_SRC") or PROJECT_ROOT / "corpus" / "src"),
            obf=Path(env("BENCH_OBF_DIR") or PROJECT_ROOT / "tmp" / "obf"),
            work=default_work_root(),
        )

    @classmethod
    def under(cls, base: Path | str) -> "Workspace":
        base = Path(base)
        return cls(results=base / "results", corpus_src=base / "corpus_src", obf=base / "obf", work=base / "work")

    # -- 目录 ----------------------------------------------------------
    def pristine(self, corpus: str) -> Path:
        """语料的原始源码树（L0 的来源）。"""
        return self.corpus_src / corpus

    def obf_root(self, corpus: str, level: str) -> Path:
        """某个混淆级的副本目录。"""
        return self.obf / corpus / level

    @property
    def cache_dir(self) -> Path:
        return self.results / "cache"

    @property
    def usage_file(self) -> Path:
        return self.results / "usage.jsonl"

    def ensure(self) -> "Workspace":
        for d in (self.results, self.corpus_src, self.obf, self.work):
            d.mkdir(parents=True, exist_ok=True)
        return self

    # -- 数据文件（设计文档 §5） ---------------------------------------
    def prepare_file(self, corpus: str) -> Path:
        return self.results / f"prepare-{corpus}.json"

    def coverage_file(self, corpus: str) -> Path:
        return self.results / f"coverage-{corpus}.json"

    def targets_file(self, corpus: str) -> Path:
        return self.results / f"targets-{corpus}.jsonl"

    def targets_summary_file(self, corpus: str) -> Path:
        return self.results / f"targets-{corpus}.summary.json"

    def obf_file(self, corpus: str, level: str) -> Path:
        return self.results / f"obf-{corpus}-{level}.json"

    @property
    def descriptions_file(self) -> Path:
        return self.results / "descriptions.jsonl"

    @property
    def tasks_file(self) -> Path:
        return self.results / "tasks.jsonl"

    def generations_file(self, model: str) -> Path:
        return self.results / f"generations-{_plain_name(model, '模型名')}.jsonl"

    def generation_failures_file(self, model: str) -> Path:
        """generate 阶段重试用尽仍失败的样本（task_id、sample_idx、原因）；成功补上后会从这里删掉。"""
        return self.results / f"generation-failures-{_plain_name(model, '模型名')}.jsonl"

    def results_file(self, tag: str) -> Path:
        return self.results / f"results-{_plain_name(tag, 'tag')}.jsonl"

    def completeness_file(self, tag: str) -> Path:
        """evaluate 写的完整性记录：每个（模型, 级别, 语料）应有多少样本、实际评测了多少、缺多少。"""
        return self.results / f"completeness-{_plain_name(tag, 'tag')}.json"

    def summary_file(self, tag: str, ext: str) -> Path:
        """report 的输出：summary-<tag>.md / .csv，图是 fig-<tag>.png / .svg（ext 带点号）。"""
        stem = "fig" if ext.lower() in (".png", ".svg") else "summary"
        return self.results / f"{stem}-{safe_name(tag)}{ext}"

    # -- 缺陷修复任务的数据文件（docs/设计-缺陷修复任务.md §7） ----------------
    def mutants_file(self, corpus: str, level: str) -> Path:
        """mutate 阶段的产出：某语料某一级副本上选定的变异体，一个（目标, 级别）一行（只含被杀死的）。"""
        return self.results / f"mutants-{corpus}-{level}.jsonl"

    def mutants_summary_file(self, corpus: str) -> Path:
        """mutate 阶段的汇总：各级的输入哈希、出题数、放弃数及原因、算子分布、same_operator_as_L0 比例等。"""
        return self.results / f"mutants-{corpus}.summary.json"

    def mutants_check_file(self, corpus: str) -> Path:
        """``check-mutants`` 的产出：mutate 选定的变异体在沙箱里回放复核的结果（有没有被稳定地杀死、杀死方式有没有变）。"""
        return self.results / f"mutants-{corpus}.check.json"

    @property
    def repair_tasks_file(self) -> Path:
        return self.results / "tasks-repair.jsonl"

    def repair_generations_file(self, model: str) -> Path:
        return self.results / f"generations-repair-{safe_name(model)}.jsonl"

    def repair_generation_failures_file(self, model: str) -> Path:
        return self.results / f"generation-failures-repair-{safe_name(model)}.jsonl"

    def repair_results_file(self, tag: str) -> Path:
        return self.results / f"results-repair-{safe_name(tag)}.jsonl"

    def repair_completeness_file(self, tag: str) -> Path:
        return self.results / f"completeness-repair-{safe_name(tag)}.json"


# ----------------------------------------------------------------------
# JSON / JSONL
# ----------------------------------------------------------------------
_APPEND_LOCK = threading.Lock()


def dumps_line(rec: Mapping[str, Any]) -> str:
    """一条 JSONL 记录（单行、保留中文）。文本里有孤立代理项（UTF-8 编不出来）时退回 ASCII 转义，
    这样写文件不会抛 UnicodeEncodeError，读回来的内容也不变。"""
    text = json.dumps(rec, ensure_ascii=False, separators=(",", ":"), default=str)
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        text = json.dumps(rec, ensure_ascii=True, separators=(",", ":"), default=str)
    return text


def _dumps_pretty(obj: Any) -> str:
    text = json.dumps(obj, ensure_ascii=False, indent=1, default=str)
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        text = json.dumps(obj, ensure_ascii=True, indent=1, default=str)
    return text


def atomic_write(path: Path, text: str) -> None:
    """先写同目录下的临时文件再 os.replace：读的人不会看到写了一半的文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def read_json(path: Path | str) -> Any | None:
    """读 JSON 文件；文件不存在或内容损坏返回 None（调用方按「没有产出」处理，会重新生成）。"""
    path = Path(path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        log.warning("读取 %s 失败（%s），按不存在处理", path, exc)
        return None


def write_json(path: Path | str, obj: Any) -> Path:
    """原子地写 JSON（缩进 1、保留中文、键序保持插入顺序）。"""
    path = Path(path)
    atomic_write(path, _dumps_pretty(obj) + "\n")
    return path


def read_jsonl_checked(path: Path | str) -> tuple[list[dict], int]:
    """读 JSONL，返回（记录, 坏行数）：空行跳过；坏行（例如上次中断时写了一半）记一条警告、计数后跳过。
    文件不存在返回（空列表, 0）。需要知道有没有丢行的调用方（report）用它。"""
    path = Path(path)
    out: list[dict] = []
    bad = 0
    try:
        with path.open("r", encoding="utf-8") as f:
            for no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    log.warning("%s 第 %d 行不是合法 JSON，已跳过", path.name, no)
                    bad += 1
                    continue
                if isinstance(rec, dict):
                    out.append(rec)
                else:
                    bad += 1
    except FileNotFoundError:
        pass
    return out, bad


def read_jsonl(path: Path | str) -> list[dict]:
    """读 JSONL；空行跳过，坏行记一条警告后跳过。文件不存在返回空列表。"""
    return read_jsonl_checked(path)[0]


def _ends_without_newline(path: Path) -> bool:
    """文件非空且最后一个字节不是换行：上一次写入被中断，留下了半行。"""
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            if f.tell() == 0:
                return False
            f.seek(-1, os.SEEK_END)
            return f.read(1) != b"\n"
    except OSError:
        return False


def append_jsonl(path: Path | str, rec: Mapping[str, Any]) -> None:
    """线程安全地追加一行。上次写入被中断、文件末尾没有换行时，先补一个换行，
    免得新记录接在半行后面、连同半行一起被 read_jsonl 当坏行丢掉。"""
    path = Path(path)
    data = (dumps_line(rec) + "\n").encode("utf-8")
    with _APPEND_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        if _ends_without_newline(path):
            data = b"\n" + data
        with path.open("ab") as f:
            f.write(data)


def write_jsonl(path: Path | str, records: Iterable[Mapping[str, Any]]) -> Path:
    """原子地整份重写 JSONL。"""
    path = Path(path)
    text = "".join(dumps_line(r) + "\n" for r in records)
    atomic_write(path, text)
    return path
