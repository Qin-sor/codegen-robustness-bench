"""Docker 沙箱：语料仓库的构建、测试，以及模型生成的 C 代码，只允许在这里编译和执行。

固定的 docker run 参数（设计文档 4.2）：

    docker run --rm --network none --cpus N --memory M --pids-limit P
        --security-opt no-new-privileges --cap-drop ALL -u uid:gid
        -e HOME=/tmp -v <workdir>:/work -w /work <image>
        bash -c "timeout -s KILL <timeout> bash -c '<cmd>'"

在文档参数之外多加了三项，都只收紧、不放宽：
  * ``--memory-swap``：与 ``--memory`` 相同，禁止用 swap，内存超限时直接 OOM 而不是拖垮宿主机；
  * ``--ulimit core=0``：不产生 core 文件（宿主机 core_pattern 可能是 apport，崩溃样本多时会很吵）；
  * ``--name``：每次运行用唯一容器名，外层超时或被中断时能按名字强杀，不留孤儿容器。

超时与信号：
  * ``timeout -s KILL`` 触发时容器退出码是 137（128+9）；``timeout`` 自己的超时码是 124。
  * 退出码 > 128 时按 128+n 换算成信号 n。
  * rc=137 既可能是超时，也可能是 OOM 或程序自己 kill -9；为避免把 OOM 误判成超时，
    rc=137 只有在总耗时 >= timeout_s 时才记 timed_out（rc=124 一律记超时）。
"""
from __future__ import annotations

import logging
import os
import re
import shlex
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

# 每路输出（stdout / stderr）保留的最大字节数
OUTPUT_LIMIT = 64 * 1024
# 截断时中间插入的提示最多占用的字节数（保证首尾 + 提示 不超过 OUTPUT_LIMIT）
_MARKER_BUDGET = 64
# 外层 Python 等待 docker 的额外宽限：容器内 timeout 之外，给 docker 启动和回收留的时间
_OUTER_GRACE_S = 30
# 容器内挂载点
CONTAINER_WORKDIR = "/work"


class SandboxError(RuntimeError):
    """docker 本身出错（镜像缺失、守护进程不可用、读输出管道失败），不是被测代码的问题，不能记成样本失败。"""


@dataclass
class RunResult:
    rc: int
    stdout: str
    stderr: str
    duration_s: float
    timed_out: bool
    signal: int | None
    io_error: str = ""      # 读容器输出的管道时出错（输出已被无声截断）：基础设施故障，调用方不能把它记成样本失败
    stdout_truncated: bool = False   # stdout 超过 OUTPUT_LIMIT，中间部分被丢弃：靠输出文字做的判据在这种文本上不可靠
    stderr_truncated: bool = False


class _BoundedBuffer:
    """流式读管道，只保留头尾，内存占用不随输出量增长。

    输出总量 <= limit 时原样保留；超出时保留开头和结尾各一半，中间换成一行提示。
    """

    def __init__(self, limit: int = OUTPUT_LIMIT) -> None:
        self.limit = limit
        self.head_limit = (limit - _MARKER_BUDGET) // 2
        self.tail_limit = limit - _MARKER_BUDGET - self.head_limit
        self._head = bytearray()   # 前 limit 字节（未超限时就是全部输出）
        self._tail = bytearray()   # 最近的若干字节，周期性裁剪到 tail_limit
        self.total = 0
        self.error = ""            # 读管道出错的原因；非空表示输出不完整

    def feed(self, data: bytes) -> None:
        self.total += len(data)
        room = self.limit - len(self._head)
        if room > 0:
            self._head += data[:room]
        self._tail += data
        if len(self._tail) > 2 * self.tail_limit:
            del self._tail[: len(self._tail) - self.tail_limit]

    @property
    def truncated(self) -> bool:
        return self.total > self.limit

    def text(self) -> str:
        if self.total <= self.limit:
            return bytes(self._head).decode("utf-8", errors="replace")
        dropped = self.total - self.head_limit - self.tail_limit
        marker = f"\n...[truncated {dropped} bytes]...\n".encode()
        head = bytes(self._head[: self.head_limit])
        tail = bytes(self._tail[-self.tail_limit:])
        return (head + marker + tail).decode("utf-8", errors="replace")


def _drain(pipe, buf: _BoundedBuffer) -> None:
    try:
        while True:
            chunk = pipe.read(65536)
            if not chunk:
                break
            buf.feed(chunk)
    except (OSError, ValueError) as exc:
        buf.error = f"{type(exc).__name__}: {exc}"
        log.warning("读沙箱输出管道出错（%s），输出不完整", buf.error)


def raise_if_docker_failed(res: "RunResult") -> None:
    """docker run 自己失败（rc=125 且 stderr 是 docker 的报错），或读容器输出的管道出了错（输出被无声截断）时
    抛 SandboxError：这是基础设施故障，不能被记成样本失败。"""
    if res.rc == 125 and re.search(r"docker:|Unable to find image|Error response from daemon|daemon",
                                   res.stderr, re.I):
        raise SandboxError(f"docker 运行失败: {res.stderr.strip()[-300:]}")
    if getattr(res, "io_error", ""):
        raise SandboxError(f"读沙箱输出时出错，输出不完整: {res.io_error}")


def _signal_from_rc(rc: int) -> int | None:
    """退出码 128+n 换算成信号 n；超出 Linux 信号范围（1..64）的不算信号。"""
    if 128 < rc <= 128 + 64:
        return rc - 128
    return None


class DockerSandbox:
    def __init__(self, image: str = "gcc:13", cpus: float = 1.0, memory: str = "1g",
                 pids: int = 256, timeout_s: int = 120,
                 uid: int | None = None, gid: int | None = None) -> None:
        self.image = image
        self.cpus = cpus
        self.memory = memory
        self.pids = pids
        self.timeout_s = timeout_s
        # 默认用宿主机当前用户，这样容器里写出的文件宿主机也能删
        self.uid = os.getuid() if uid is None else uid
        self.gid = os.getgid() if gid is None else gid
        self._image_id: str | None = None

    # ------------------------------------------------------------------
    def image_id(self) -> str:
        """镜像 ID（``sha256:...``）。镜像标签是浮动的（gcc:13 以后拉到新版本，ID 就变），
        各阶段的输入哈希和结果记录带上它，才能发现「用的已经不是同一个编译器」。查不到返回空串。"""
        cached = getattr(self, "_image_id", None)
        if cached is None:
            try:
                res = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", self.image],
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=30)
                cached = res.stdout.strip() if res.returncode == 0 else ""
            except (OSError, subprocess.SubprocessError):
                cached = ""
            if not cached:
                log.warning("读不到镜像 %s 的 ID，输入哈希里的镜像身份只有标签", self.image)
            self._image_id = cached
        return cached

    def identity(self) -> dict:
        """这台沙箱的身份：镜像标签、镜像 ID 和资源上限（进输入哈希与结果记录）。"""
        return {"image": self.image, "image_id": self.image_id(),
                "cpus": self.cpus, "memory": str(self.memory), "pids": self.pids, "timeout_s": self.timeout_s}

    # ------------------------------------------------------------------
    @staticmethod
    def docker_available(image: str | None = None) -> bool:
        """docker 可用（且镜像已在本地）时返回 True，测试用它决定是否跳过。"""
        try:
            args = ["docker", "image", "inspect", image] if image else ["docker", "version"]
            return subprocess.run(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                  timeout=30).returncode == 0
        except (OSError, subprocess.SubprocessError):
            return False

    def docker_args(self, workdir: Path, cmd: str, env: dict | None, timeout_s: float,
                    name: str) -> list[str]:
        """拼出完整的 docker run 命令行（单独成函数，方便测试固定参数没有被改掉）。"""
        host_dir = str(Path(workdir).resolve())
        if ":" in host_dir:
            raise ValueError(f"workdir 路径含冒号，无法用 -v 挂载: {host_dir}")
        # 超时必须罩住整条命令：cmd 里可能有 a && b; c 这样的复合语句，
        # 直接写 `timeout N <cmd>` 只会罩住第一个词，所以包一层 bash -c。
        inner = f"timeout -s KILL {timeout_s:g} bash -c {shlex.quote(cmd)}"
        args = [
            "docker", "run", "--rm", "--name", name,
            "--network", "none",
            "--cpus", str(self.cpus),
            "--memory", str(self.memory),
            "--memory-swap", str(self.memory),
            "--pids-limit", str(self.pids),
            "--security-opt", "no-new-privileges",
            "--cap-drop", "ALL",
            "--ulimit", "core=0",
            "-u", f"{self.uid}:{self.gid}",
            "-e", "HOME=/tmp",
        ]
        for key, value in (env or {}).items():
            if value is None:
                continue
            args += ["-e", f"{key}={value}"]
        args += ["-v", f"{host_dir}:{CONTAINER_WORKDIR}", "-w", CONTAINER_WORKDIR,
                 self.image, "bash", "-c", inner]
        return args

    def run(self, workdir: Path, cmd: str, env: dict | None = None,
            timeout_s: int | None = None) -> RunResult:
        workdir = Path(workdir)
        if not workdir.is_dir():
            raise FileNotFoundError(f"沙箱工作目录不存在: {workdir}")
        timeout = float(self.timeout_s if timeout_s is None else timeout_s)
        if timeout <= 0:
            # GNU timeout 把 0 当成「不限时」，这里必须挡住
            raise ValueError(f"timeout_s 必须为正数: {timeout_s}")
        name = f"bench-sbx-{uuid.uuid4().hex[:12]}"
        args = self.docker_args(workdir, cmd, env, timeout, name)
        log.debug("docker run %s: %s", name, cmd[:200])

        out, err = _BoundedBuffer(), _BoundedBuffer()
        t0 = time.monotonic()
        proc = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, bufsize=0)
        readers = [threading.Thread(target=_drain, args=(proc.stdout, out), daemon=True),
                   threading.Thread(target=_drain, args=(proc.stderr, err), daemon=True)]
        for t in readers:
            t.start()
        outer_timed_out = False
        try:
            try:
                proc.wait(timeout=timeout + _OUTER_GRACE_S)
            except subprocess.TimeoutExpired:
                # 容器内的 timeout 没生效（docker 卡住等），按名字强杀
                outer_timed_out = True
                log.warning("沙箱 %s 超过外层等待上限，强制清理", name)
                self._force_remove(name)
                proc.kill()
                proc.wait()
        except BaseException:
            # Ctrl-C 等中断：别把容器留在后台
            self._force_remove(name)
            proc.kill()
            raise
        finally:
            for t in readers:
                t.join(timeout=5)
        duration = time.monotonic() - t0

        rc = proc.returncode
        if rc < 0:
            # docker 客户端自己被信号杀掉（Popen 约定返回负数），换成 shell 的 128+n 写法
            rc = 128 - rc
        if outer_timed_out:
            rc = 137
        timed_out = outer_timed_out or rc == 124 or (rc == 137 and duration >= timeout)
        if rc == 125:
            # docker 自己出错（镜像不存在、参数不合法等），不是被测命令的退出码
            log.error("docker run 失败(rc=125): %s", err.text()[-300:])
        io_error = "; ".join(e for e in (out.error and f"stdout {out.error}", err.error and f"stderr {err.error}") if e)
        return RunResult(rc=rc, stdout=out.text(), stderr=err.text(),
                         duration_s=duration, timed_out=timed_out,
                         signal=_signal_from_rc(rc), io_error=io_error,
                         stdout_truncated=out.truncated, stderr_truncated=err.truncated)

    @staticmethod
    def _force_remove(name: str) -> None:
        """强制删除容器，直到 docker inspect 查不到为止（最多试 3 次）。"""
        for attempt in range(3):
            try:
                subprocess.run(["docker", "rm", "-f", name], stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=30)
                gone = subprocess.run(["docker", "inspect", name], stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL, timeout=30).returncode != 0
            except (OSError, subprocess.SubprocessError):
                log.warning("清理容器 %s 失败", name)
                return
            if gone:
                return
            time.sleep(0.5 * (attempt + 1))
        log.warning("容器 %s 在 3 次强制删除后仍然存在", name)
