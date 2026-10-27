"""函数级执行次数（设计文档 §4.4）：沙箱里带 --coverage 构建并跑项目测试，再用 gcov 取每个函数的 execution_count。

步骤（构建、测试、gcov 全部在 Docker 沙箱里执行，宿主机只读结果文件）：

1. ``corpus.materialize`` 复制一份工作副本到 ``<项目根>/tmp/work/cov-*``；
2. 沙箱里按 ``spec.coverage`` 构建并跑测试：环境变量取 ``spec.coverage["env"]``（``CFLAGS="--coverage -O0"``、
   ``LDFLAGS="--coverage"``），命令取 ``spec.coverage_command()``（有 ``coverage.build_cmd`` 用它，否则就是
   build_cmd + test_cmd）；
3. 沙箱里对工作副本里每个 ``.gcda`` 单独跑一次 ``gcov --json-format <绝对路径>``，输出写进
   ``_bench_gcov/<序号>/``，每个 .gcda 一个子目录，避免同名输出互相覆盖。

为什么直接把 .gcda 路径交给 gcov、而不是按源文件名加 ``-o``：一条 gcc 命令同时编译多个源文件并链接时
（``gcc --coverage -o mini_test calc.c util.c test_main.c``），数据文件名是 ``<输出名>-<源文件名>.gcda``
（``mini_test-calc.gcda``），与源文件名对不上，``gcov -o <目录> calc.c`` 找不到；把 .gcda 路径给 gcov，
它按这个路径找同名的 .gcno，不用猜前缀；
4. 宿主机读 ``*.gcov.json.gz``，把落在目标文件里的函数的 execution_count 累加。
   同一个函数可能出现在多个 .gcda 里（cJSON 的测试各自 ``#include "../cJSON.c"``，函数名相同、文件记为
   ``tests/../cJSON.c``），路径先规范化再按 ``<文件>:<函数名>`` 求和。

返回值 key = ``f"{file}:{name}"``，file 是相对项目根的 POSIX 路径（与 FunctionTarget.file 同一写法）。
目标文件里被编译进来、但一次都没执行的函数值为 0；根本没被编译进任何测试程序的函数不在字典里
（调用方按 0 处理）。
"""
from __future__ import annotations

import gzip
import json
import logging
import posixpath
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

from bench.corpus import CorpusSpec, expand_patterns, materialize
from bench.runner import new_workdir, raise_if_docker_failed
from bench.sandbox import CONTAINER_WORKDIR

log = logging.getLogger(__name__)

__all__ = ["function_exec_counts", "parse_gcov_json", "CoverageError", "GCOV_DIR"]

GCOV_DIR = "_bench_gcov"

# 在容器里对每个 .gcda 单独跑 gcov；任何一个失败只记一行到 stderr，不中断
_GCOV_SCRIPT = r"""
rm -rf {gdir}; mkdir -p {gdir}; i=0; failed=0
while IFS= read -r -d '' g; do
  i=$((i+1)); mkdir -p {gdir}/$i
  if ! (cd {gdir}/$i && gcov --json-format "{work}/${{g#./}}" > gcov.out 2> gcov.err); then
    echo "gcov failed: $g: $(tail -n 3 {gdir}/$i/gcov.err 2>/dev/null | tr '\n' ' ')" >&2; failed=$((failed+1))
  fi
done < <(find . -name '*.gcda' -not -path './{gdir}/*' -print0 | sort -z)
echo "gcda=$i failed=$failed"
"""

# 优化产生的函数克隆后缀（.part.0 / .constprop.1 / .isra.2 / .cold 等）；覆盖率构建用 -O0 通常没有，保险起见归并
_CLONE_SUFFIX_RE = re.compile(r"(?:\.(?:part|constprop|isra|cold|lto_priv|localalias)(?:\.\d+)?)+$")


class CoverageError(RuntimeError):
    """覆盖率构建没产出任何 .gcda，或 gcov 全部失败。"""


def _base_name(name: str) -> str:
    return _CLONE_SUFFIX_RE.sub("", name)


def parse_gcov_json(raw: bytes | str | Path) -> Iterator[tuple[str, str, int]]:
    """解析一份 gcov JSON（.gcov.json.gz 的路径、解压后的 bytes 或 gz 字节），逐个产出 (规范化的文件路径, 函数名, 执行次数)。

    文件路径按文档里的 ``current_working_directory`` 规范化成绝对路径；落在沙箱挂载点 /work 下的，
    返回相对项目根的 POSIX 路径，其余（系统头等）原样返回绝对路径，调用方自然匹配不上目标文件。
    """
    if isinstance(raw, Path):
        raw = raw.read_bytes()
    if isinstance(raw, bytes) and raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    doc = json.loads(raw)
    cwd = doc.get("current_working_directory") or CONTAINER_WORKDIR
    prefix = CONTAINER_WORKDIR.rstrip("/") + "/"
    for entry in doc.get("files", []):
        full = posixpath.normpath(posixpath.join(cwd, entry.get("file", "")))
        rel = full[len(prefix):] if full.startswith(prefix) else full
        for fn in entry.get("functions", []):
            yield rel, _base_name(fn.get("name", "")), int(fn.get("execution_count", 0))


def _target_files(spec: CorpusSpec, root: Path) -> set[str]:
    return set(expand_patterns(spec.target_files, root))


def function_exec_counts(spec: CorpusSpec, src_root: Path, sandbox: Any, *, work_root: Path | None = None,
                         keep_workdir: bool = False, stats: dict | None = None) -> dict[str, int]:
    """每个目标函数在项目自带测试里被执行的次数，key = ``f"{file}:{name}"``。

    src_root 应是未混淆的原始源码树（L0）。覆盖率构建失败（构建命令本身出错、一个 .gcda 都没有）
    抛 CoverageError；测试在覆盖率构建下没通过但生成了 .gcda 时仍然统计已有的数据，但**如实记下来**：
    gcov 部分失败的函数不会出现在结果里，被 targets 阶段当成「没被测试执行」排除，基础设施故障就这样悄悄变成
    目标集变小，所以传入 ``stats`` 字典时会填入 ``n_gcda`` / ``n_failed`` / ``failed``（失败原因的尾部文字）/
    ``tests_passed``，由调用方（pipeline.stage_coverage）决定阶段是否算失败。
    """
    src_root = Path(src_root)
    workdir = new_workdir(work_root, f"cov-{spec.name}")
    try:
        materialize(src_root, workdir)
        env = dict(spec.coverage.get("env") or {})
        cmd = spec.coverage_command()
        timeout = max(int(spec.timeout_s) * 2, 60)

        build = sandbox.run(workdir, cmd, env=env, timeout_s=timeout)
        raise_if_docker_failed(build)
        ok = spec.judge(build.rc, build.stdout, truncated=build.stdout_truncated)[0] and not build.timed_out
        if not ok:
            log.warning("%s: 覆盖率构建下构建 / 测试没有通过 (rc=%s, timed_out=%s)，尾部输出: %s",
                        spec.name, build.rc, build.timed_out, (build.stderr or build.stdout)[-300:].strip())

        script = _GCOV_SCRIPT.format(gdir=GCOV_DIR, work=CONTAINER_WORKDIR)
        gcov = sandbox.run(workdir, script, timeout_s=timeout)
        raise_if_docker_failed(gcov)
        m = re.search(r"gcda=(\d+) failed=(\d+)", gcov.stdout)
        n_gcda, n_failed = (int(m.group(1)), int(m.group(2))) if m else (0, 0)
        if stats is not None:
            stats.update({"n_gcda": n_gcda, "n_failed": n_failed, "tests_passed": ok,
                          "failed": [ln.strip() for ln in gcov.stderr.splitlines() if ln.startswith("gcov failed")][:20]})
        if n_gcda == 0:
            raise CoverageError(f"{spec.name}: 覆盖率构建没有产生 .gcda (build rc={build.rc})，"
                                f"输出尾部: {(build.stderr or build.stdout)[-400:].strip()}")
        if n_failed:
            log.warning("%s: %d / %d 个 .gcda 的 gcov 失败: %s", spec.name, n_failed, n_gcda, gcov.stderr[-600:].strip())
        if n_failed == n_gcda:
            raise CoverageError(f"{spec.name}: 所有 .gcda 的 gcov 都失败: {gcov.stderr[-400:].strip()}")

        targets = _target_files(spec, workdir)
        counts: dict[str, int] = defaultdict(int)
        n_json = 0
        for path in sorted((workdir / GCOV_DIR).glob("*/*.gcov.json.gz")):
            n_json += 1
            for rel, name, count in parse_gcov_json(path):
                if rel in targets and name:
                    counts[f"{rel}:{name}"] += count
        log.info("%s: %d 个 .gcda、%d 份 gcov JSON，目标文件里共 %d 个函数有数据（其中 %d 个执行过）",
                 spec.name, n_gcda, n_json, len(counts), sum(1 for v in counts.values() if v > 0))
        return dict(sorted(counts.items()))
    finally:
        if not keep_workdir:
            shutil.rmtree(workdir, ignore_errors=True)

