"""DockerSandbox 测试。

纯 Python 部分（输出截断、信号换算、docker 参数拼装）不需要 docker；
其余用 gcc:13 镜像真跑，docker 或镜像不可用时整组跳过。
临时目录放在项目的 tmp/ 下，不用系统临时目录。
"""
from __future__ import annotations

import os
import shlex
import subprocess

import pytest

from bench import sandbox as sbx
from bench.sandbox import DockerSandbox, RunResult, _BoundedBuffer, _signal_from_rc

from tests.helpers import needs_docker


@pytest.fixture(scope="module")
def sb() -> DockerSandbox:
    return DockerSandbox()


# ----------------------------------------------------------------------
# 不依赖 docker 的部分
# ----------------------------------------------------------------------
class TestBoundedBuffer:
    def test_small_output_kept_verbatim(self):
        buf = _BoundedBuffer()
        buf.feed(b"hello ")
        buf.feed(b"world\n")
        assert buf.text() == "hello world\n"

    def test_exactly_at_limit_is_not_truncated(self):
        buf = _BoundedBuffer(limit=1000)
        buf.feed(b"x" * 1000)
        assert buf.text() == "x" * 1000

    def test_one_byte_over_limit_is_truncated(self):
        buf = _BoundedBuffer(limit=1000)
        buf.feed(b"x" * 1001)
        text = buf.text()
        assert "truncated" in text
        assert len(text.encode()) <= 1000

    def test_keeps_head_and_tail_and_reports_dropped_bytes(self):
        limit = sbx.OUTPUT_LIMIT
        data = b"".join(b"%07d\n" % i for i in range(60000))     # 约 480 KB
        buf = _BoundedBuffer()
        for i in range(0, len(data), 4099):                        # 故意用不整齐的块大小
            buf.feed(data[i:i + 4099])
        text = buf.text()
        assert len(text.encode()) <= limit
        assert text.startswith("0000000\n0000001\n")
        assert text.endswith("0059999\n")
        marker = [ln for ln in text.splitlines() if "truncated" in ln]
        assert len(marker) == 1
        dropped = int(marker[0].split("truncated ")[1].split(" ")[0])
        kept = len(text.encode()) - len(f"\n...[truncated {dropped} bytes]...\n".encode())
        assert kept + dropped == len(data)

    def test_multibyte_split_at_cut_does_not_raise(self):
        buf = _BoundedBuffer(limit=1000)
        buf.feed("中".encode() * 3000)
        text = buf.text()
        assert "truncated" in text and len(text.encode()) <= 1000 + 64


@pytest.mark.parametrize("rc, expected", [
    (0, None), (1, None), (3, None), (128, None), (129, 1), (134, 6), (137, 9),
    (139, 11), (192, 64), (193, None), (255, None),
])
def test_signal_from_rc(rc, expected):
    assert _signal_from_rc(rc) == expected


class TestDockerArgs:
    def test_fixed_parameters(self, tmp_path):
        sb = DockerSandbox(image="gcc:13", cpus=2.0, memory="512m", pids=64, uid=1234, gid=5678)
        cmd = "echo 'a b'; ls | wc -l && exit 3"
        args = sb.docker_args(tmp_path, cmd, {"A": "1 2", "B": None}, 7, "bench-sbx-test")

        def pair(flag, value):
            return any(args[i] == flag and args[i + 1] == value for i in range(len(args) - 1))

        assert args[:3] == ["docker", "run", "--rm"]
        assert pair("--name", "bench-sbx-test")
        assert pair("--network", "none")
        assert pair("--cpus", "2.0")
        assert pair("--memory", "512m")
        assert pair("--pids-limit", "64")
        assert pair("--security-opt", "no-new-privileges")
        assert pair("--cap-drop", "ALL")
        assert pair("-u", "1234:5678")
        assert pair("-e", "HOME=/tmp")
        assert pair("-e", "A=1 2")
        assert not any(a.startswith("B=") for a in args)
        assert pair("-v", f"{tmp_path.resolve()}:/work")
        assert pair("-w", "/work")
        i = args.index("gcc:13")
        assert args[i + 1:i + 3] == ["bash", "-c"]
        # 超时必须罩住整条命令：内层 bash -c 的参数还原后应当等于原命令
        inner = shlex.split(args[i + 3])
        assert inner[:5] == ["timeout", "-s", "KILL", "7", "bash"]
        assert inner[5] == "-c" and inner[6] == cmd

    def test_default_user_is_host_user(self):
        sb = DockerSandbox()
        assert (sb.uid, sb.gid) == (os.getuid(), os.getgid())
        assert (sb.image, sb.cpus, sb.memory, sb.pids, sb.timeout_s) == ("gcc:13", 1.0, "1g", 256, 120)

    def test_workdir_with_colon_is_rejected(self, tmp_path):
        d = tmp_path / "a:b"
        d.mkdir()
        with pytest.raises(ValueError):
            DockerSandbox().docker_args(d, "true", None, 1, "n")

    def test_missing_workdir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            DockerSandbox().run(tmp_path / "nope", "true")

    @pytest.mark.parametrize("bad", [0, -1, 0.0])
    def test_non_positive_timeout_is_rejected(self, tmp_path, bad):
        # GNU timeout 把 0 当成不限时，必须在到达 docker 之前挡住
        with pytest.raises(ValueError):
            DockerSandbox().run(tmp_path, "true", timeout_s=bad)

    def test_fractional_timeout_is_passed_through(self, tmp_path):
        args = DockerSandbox().docker_args(tmp_path, "true", None, 0.5, "n")
        assert shlex.split(args[-1])[:4] == ["timeout", "-s", "KILL", "0.5"]


# ----------------------------------------------------------------------
# 需要 docker 的部分（gcc:13）
# ----------------------------------------------------------------------
@needs_docker
class TestRun:
    def test_echo_hi(self, sb, scratch):
        r = sb.run(scratch, "echo hi")
        assert isinstance(r, RunResult)
        assert (r.rc, r.stdout, r.stderr) == (0, "hi\n", "")
        assert r.timed_out is False and r.signal is None
        assert 0 < r.duration_s < 30

    def test_sleep_times_out(self, sb, scratch):
        r = sb.run(scratch, "sleep 5", timeout_s=1)
        assert r.timed_out is True
        assert r.rc == 137 and r.signal == 9
        assert 1 <= r.duration_s < 4.8        # 没有真的睡满 5 s

    def test_timeout_default_comes_from_constructor(self, scratch):
        r = DockerSandbox(timeout_s=1).run(scratch, "sleep 5")
        assert r.timed_out is True

    def test_timeout_covers_whole_compound_command(self, sb, scratch):
        # 复合语句：若超时只罩住第一个词，后面的 echo 会在 sleep 被杀后照常执行
        r = sb.run(scratch, "sleep 3; echo done-after-sleep", timeout_s=1)
        assert r.timed_out is True
        assert "done-after-sleep" not in r.stdout

    @pytest.mark.parametrize("sig_name, signum", [("SEGV", 11), ("ABRT", 6), ("FPE", 8), ("BUS", 7)])
    def test_crash_signals_are_converted_from_rc(self, sb, scratch, sig_name, signum):
        r = sb.run(scratch, f"kill -{sig_name} $$")
        assert r.rc == 128 + signum
        assert r.signal == signum
        assert r.timed_out is False

    def test_exit_code_and_stderr(self, sb, scratch):
        r = sb.run(scratch, "echo out; echo err >&2; exit 3")
        assert (r.rc, r.stdout, r.stderr) == (3, "out\n", "err\n")
        assert r.signal is None and r.timed_out is False

    def test_self_sigkill_is_not_a_timeout(self, sb, scratch):
        # rc=137 本身不足以说明超时：程序自己 kill -9（或被 OOM 杀）时耗时远小于上限
        r = sb.run(scratch, "kill -KILL $$", timeout_s=30)
        assert r.rc == 137 and r.signal == 9
        assert r.timed_out is False

    def test_oom_kill_is_not_a_timeout(self, scratch):
        r = DockerSandbox(memory="64m").run(scratch, "python3 -c 'x = bytearray(10**9)'", timeout_s=30)
        assert r.rc == 137 and r.signal == 9
        assert r.timed_out is False

    def test_env_is_passed_without_shell_interpretation(self, sb, scratch):
        r = sb.run(scratch, 'echo "[$FOO][$BAR][$HOME]"', env={"FOO": "a b $x", "BAR": "x=y"})
        assert r.stdout == "[a b $x][x=y][/tmp]\n"

    def test_runs_as_host_user_and_workdir_is_shared(self, sb, scratch):
        r = sb.run(scratch, "id -u; id -g; pwd; echo data > from_container.txt")
        assert r.stdout.split() == [str(os.getuid()), str(os.getgid()), "/work"]
        out = scratch / "from_container.txt"
        assert out.read_text() == "data\n"
        assert out.stat().st_uid == os.getuid()        # 宿主机用户自己能删

    def test_host_files_visible_in_container(self, sb, scratch):
        (scratch / "in.txt").write_text("from host\n")
        assert sb.run(scratch, "cat in.txt").stdout == "from host\n"

    def test_no_network_no_caps_no_new_privs(self, sb, scratch):
        r = sb.run(scratch, "ls /sys/class/net; grep -E '^(CapEff|NoNewPrivs)' /proc/self/status")
        assert r.stdout.split() == ["lo", "CapEff:", "0000000000000000", "NoNewPrivs:", "1"]

    def test_resource_limits_reach_cgroup(self, scratch):
        probe = "cat /sys/fs/cgroup/pids.max /sys/fs/cgroup/memory.max /sys/fs/cgroup/cpu.max"
        r = DockerSandbox(cpus=0.5, memory="256m", pids=64).run(scratch, probe)
        if r.rc != 0:
            pytest.skip("容器内没有 cgroup v2 文件")
        assert r.stdout.split("\n")[:3] == ["64", str(256 * 1024 * 1024), "50000 100000"]

    def test_stdout_and_stderr_truncated_separately(self, sb, scratch):
        cmd = ("head -c 300000 /dev/zero | tr '\\0' 'a'; "
               "head -c 300000 /dev/zero | tr '\\0' 'b' >&2")
        r = sb.run(scratch, cmd)
        for text, ch in ((r.stdout, "a"), (r.stderr, "b")):
            assert len(text) <= sbx.OUTPUT_LIMIT
            assert "truncated" in text
            assert text.startswith(ch * 100) and text.endswith(ch * 100)

    def test_endless_output_is_bounded_and_killed(self, sb, scratch):
        r = sb.run(scratch, "yes", timeout_s=2)
        assert r.timed_out is True
        assert len(r.stdout) <= sbx.OUTPUT_LIMIT

    def test_stdin_is_closed(self, sb, scratch):
        r = sb.run(scratch, "cat; echo eof-reached", timeout_s=20)
        assert r.stdout == "eof-reached\n"

    def test_gcc_compiles_and_runs_c(self, sb, scratch):
        (scratch / "a.c").write_text('#include <stdio.h>\nint main(void){puts("c-ok");return 7;}\n')
        r = sb.run(scratch, "gcc -o a a.c && ./a")
        assert (r.rc, r.stdout) == (7, "c-ok\n")


@needs_docker
def test_outer_timeout_removes_container(scratch, monkeypatch):
    """容器内 timeout 没生效时（模拟：外层等待上限小于容器内超时），要按名字强杀容器。"""
    names: list[str] = []
    orig = DockerSandbox.docker_args

    def spy(self, workdir, cmd, env, timeout_s, name):
        names.append(name)
        return orig(self, workdir, cmd, env, timeout_s, name)

    monkeypatch.setattr(DockerSandbox, "docker_args", spy)
    monkeypatch.setattr(sbx, "_OUTER_GRACE_S", -35.0)
    r = DockerSandbox().run(scratch, "sleep 100", timeout_s=40)     # 容器内最多 40 s，外层只等 5 s
    assert r.timed_out is True and r.rc == 137
    assert 4 < r.duration_s < 20
    gone = subprocess.run(["docker", "inspect", names[0]], stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL).returncode != 0
    assert gone, "外层超时后容器仍然存在"
