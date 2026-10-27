"""pytest 的共用夹具与钩子。

* ``needs_docker`` 标记：docker 或镜像不可用时跳过；环境变量 ``BENCH_REQUIRE_DOCKER=1`` 时改为失败。
  可信度措施的自测（e2e、oracle gate、stub-kill、沙箱评测）全靠沙箱，CI 必须设这个变量，
  再用 ``scripts/ci.sh`` 检查「跳过数 > 0 即失败」；
* ``scratch`` 夹具：项目 ``tmp/pytest/`` 下的一次性目录，测试结束删除（不碰系统临时目录）；
* ``mini_spec`` 夹具：mini 夹具的 CorpusSpec。
"""
from __future__ import annotations

import shutil
import uuid

import pytest

from tests.helpers import DOCKER_OK, IMAGE, REQUIRE_DOCKER, ROOT, mini_spec as _mini_spec
from tests.pipeline_support import make_planned_workspace


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "needs_docker: 需要 docker 与沙箱镜像（缺则跳过，BENCH_REQUIRE_DOCKER=1 时失败）")


def pytest_runtest_setup(item: pytest.Item) -> None:
    if item.get_closest_marker("needs_docker") is not None and not DOCKER_OK:
        msg = f"docker 或镜像 {IMAGE} 不可用"
        if REQUIRE_DOCKER:
            pytest.fail(msg + "（BENCH_REQUIRE_DOCKER=1：需要沙箱的测试不允许跳过）", pytrace=False)
        pytest.skip(msg)


@pytest.fixture()
def scratch():
    """项目 tmp/ 下的一次性目录，测试结束后删除。"""
    base = ROOT / "tmp" / "pytest"
    base.mkdir(parents=True, exist_ok=True)
    d = base / uuid.uuid4().hex[:12]
    d.mkdir()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture(scope="module")
def scratch_module():
    """同 scratch，但整个模块共用一个目录（给 module 作用域的夹具用）。"""
    base = ROOT / "tmp" / "pytest"
    base.mkdir(parents=True, exist_ok=True)
    d = base / uuid.uuid4().hex[:12]
    d.mkdir()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture(scope="session")
def mini_spec():
    return _mini_spec()


@pytest.fixture()
def planned(scratch, mini_spec):
    """一个有 4 个目标、L0 与 L1 副本（未跑 verify）的工作区，targets 记录与 oracle gate 手工写好（全过）：
    返回 (Context, Workspace, 目标列表)。见 tests/pipeline_support.py。"""
    return make_planned_workspace(scratch, mini_spec)
