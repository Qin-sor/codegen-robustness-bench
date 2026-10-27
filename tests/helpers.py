"""所有测试模块共用的常量与小工具（以前每个模块各抄一份 ROOT / MINI / IMAGE / needs_docker）。

环境变量：

* ``BENCH_REQUIRE_DOCKER=1``：docker 或镜像不可用时，需要沙箱的测试**失败**而不是跳过（CI 里必须设，
  否则没有 docker 的机器上整组沙箱测试悄悄跳过，套件仍是绿的，e2e 自测等于没跑）；
* ``BENCH_REQUIRE_CORPUS=1``：真实语料（``corpus/src/<name>``，或 ``CORPUS_MIRROR`` 指向的本地镜像）找不到时同样失败；
* ``BENCH_SANDBOX_IMAGE``：沙箱镜像（默认 gcc:13）。
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from bench import corpus
from bench.sandbox import DockerSandbox

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT
MINI = ROOT / "tests" / "fixtures" / "mini"
IMAGE = os.environ.get("BENCH_SANDBOX_IMAGE", "gcc:13")
DOCKER_OK = DockerSandbox.docker_available(IMAGE)
REQUIRE_DOCKER = os.environ.get("BENCH_REQUIRE_DOCKER") == "1"
REQUIRE_CORPUS = os.environ.get("BENCH_REQUIRE_CORPUS") == "1"

#: 需要 docker 沙箱的测试（标记由 conftest 的钩子处理：缺 docker 时跳过，BENCH_REQUIRE_DOCKER=1 时失败）
needs_docker = pytest.mark.needs_docker

#: 中文技术写作的违例词表：用拼接的方式写，免得这个文件自己被 grep 命中
BANNED = ["管" + "线", "整" + "备", "旋" + "钮", "治" + "理", "一等" + "公民", "买" + "回"]


def mini_spec() -> corpus.CorpusSpec:
    """tests/fixtures/mini 的 spec（CorpusSpec）。"""
    return corpus.load_spec_file(MINI / "spec.json")


def real_corpus_dir(name: str) -> Path | None:
    """真实语料 ``name`` 的源码目录：``corpus/src/<name>``，没有就看 ``CORPUS_MIRROR``（本地克隆）；都没有返回 None。
    只读用，别往里写。"""
    cands = [corpus.CORPUS_SRC / name]
    mirror = corpus.mirror_dir()
    if mirror is not None:
        try:
            cands.append(mirror / corpus.load_spec(name).repo_dir_name)
        except (FileNotFoundError, corpus.SpecError):
            pass
    for c in cands:
        if c.is_dir() and any(c.iterdir()):
            return c
    return None


def require_real_corpus(name: str) -> Path:
    """真实语料 ``name`` 的源码目录；找不到时跳过（``BENCH_REQUIRE_CORPUS=1`` 时失败）。"""
    d = real_corpus_dir(name)
    if d is None:
        msg = f"找不到语料 {name} 的源码（corpus/src 与 CORPUS_MIRROR 都没有；先 python3 -m bench prepare --corpus {name}）"
        if REQUIRE_CORPUS:
            pytest.fail(msg, pytrace=False)
        pytest.skip(msg)
    return d
