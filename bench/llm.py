"""LLM 抽象、回复缓存、OpenAI 兼容客户端与 MockLLM（设计文档 §4.7）。

模块里有四样东西：

* ``Reply`` / ``LLM``：统一的回复数据类与抽象基类。``LLM.chat`` 在文档签名之外多了可选参数
  ``meta``（如 ``{"task_id": ..., "level": ..., "obf_signature": ...}``）：OpenAICompat 不拿它改变请求，
  只把其中的 task_id / target_id / level / purpose 写进 usage 日志；MockLLM 用它挑选要返回的代码。
  按设计文档签名（没有 ``meta``）写的 LLM 子类同样可用：流水线一律经 :func:`chat_with_meta` 调用，
  只在 ``chat`` 接受 ``meta`` 时才传。``LLM.fingerprint()`` 返回实例配置的摘要，进 generate 阶段的输入哈希。
* ``Cache``：``results/cache/<sha256>.json``。键是 sha256(model, messages, temperature, max_tokens)，再并入
  调用方传的 seed 和 ``extra``（OpenAICompat 的实例配置：端点 ``base_url``、实际发给 API 的模型名、额外请求参数如 DeepSeek 的
  ``thinking``、max_tokens 下限），否则同一份 prompt 的多个样本、思考 / 非思考两种模式、不同端点会互相命中。
* ``OpenAICompat``：用标准库 urllib 调 ``{base_url}/chat/completions``。429 / 5xx / 网络错误 / 响应坏掉
  按指数退避重试（默认重试 5 次），其它 4xx 立即报错；每次真实调用的 token 用量追加写
  ``results/usage.jsonl``（命中缓存的不记，不产生费用）。API key 只从环境变量读，
  不放在实例属性里，也不进日志、缓存、结果文件。
* ``MockLLM``：离线自测用，四种模式 oracle / broken / wrong / empty，见类的文档。

多样本（``generate --samples N``）：temperature > 0 时，给每个样本传不同的 seed（如 sample_idx）。seed 参与缓存键，
所以每个样本单独请求、单独缓存；DeepSeek 预设不把 seed 发给 API（文档里没有这个参数），样本之间的差异来自
temperature 采样。temperature = 0 时多样本没有意义：同样的请求只会命中同一份缓存。

关于模型名（2026-10 核对过 DeepSeek 官方文档）：设计文档里的 ``deepseek-chat`` / ``deepseek-reasoner``
在官方 Change Log 里有 2026-07-24 起弃用的通知（官方页面上是否仍列着、端点是否还认，以官方当时的文档和端点实际返回为准，
这里不替官方下结论），现行模型是 ``deepseek-flash`` 与 ``deepseek-v4-pro``，
且默认开启思考模式（思考模式下 temperature 不生效）。所以：

* ``OpenAICompat`` 支持 ``extra_body`` 往请求体里并入额外字段（``thinking`` / ``reasoning_effort`` 等）；
* ``PRESETS`` / ``make_llm`` 给出一组命名预设：``deepseek-flash`` 是关闭思考的版本（temperature=0 才有意义，
  对应旧的 deepseek-chat），``deepseek-flash-thinking`` 是开启思考的版本（对应旧的 deepseek-reasoner）；
  旧名字仍保留，使用时会打一条警告。

环境变量：``BENCH_ROOT`` / ``BENCH_RESULTS_DIR``（覆盖项目根与结果目录）、
``BENCH_LLM_NO_PROXY=1``（调用 API 时不走环境变量里的代理）。
"""
from __future__ import annotations

import abc
import functools
import hashlib
import http.client
import inspect
import json
import logging
import os
import random
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from bench.store import PROJECT_ROOT, SCHEMA_VERSION, append_jsonl, atomic_write, digest, dumps_line, now_iso

log = logging.getLogger(__name__)

__all__ = [
    "SCHEMA_VERSION",
    "DEEPSEEK_BASE_URL",
    "DEEPSEEK_KEY_ENV",
    "MOCK_MODES",
    "MOCK_DESCRIPTION",
    "PRESETS",
    "ModelPreset",
    "LLMError",
    "LLMHTTPError",
    "MissingAPIKeyError",
    "MockLLMError",
    "Reply",
    "LLM",
    "Cache",
    "OpenAICompat",
    "MockLLM",
    "make_llm",
    "estimate_tokens",
    "chat_with_meta",
    "results_dir",
    "default_cache_dir",
    "default_usage_path",
    "mock_wrong_code",
    "mock_broken_code",
]

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_KEY_ENV = "DEEPSEEK_API_KEY"

def results_dir() -> Path:
    """结果目录：环境变量 BENCH_RESULTS_DIR 优先，默认 <项目根>/results。每次调用重新读环境变量。"""
    return Path(os.environ.get("BENCH_RESULTS_DIR") or PROJECT_ROOT / "results")


def default_cache_dir() -> Path:
    return results_dir() / "cache"


def default_usage_path() -> Path:
    return results_dir() / "usage.jsonl"


# --------------------------------------------------------------------------- 异常


class LLMError(RuntimeError):
    """调用 LLM 失败：重试用尽、不可重试的 HTTP 错误、响应格式不对等。"""


class MissingAPIKeyError(LLMError):
    """API key 所在的环境变量没设置（只报变量名，不涉及 key 的值）。"""


class LLMHTTPError(LLMError):
    """不可重试的 HTTP 错误（401 / 402 / 400 / 404 / 422 等）。"""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status

    @property
    def fatal(self) -> bool:
        """整批请求必然都会失败的状态码（key 无效 / 余额不足 / 没有权限）：调用方应当中止阶段，而不是逐个样本再试。"""
        return self.status in (401, 402, 403)


class MockLLMError(LLMError):
    """MockLLM 无法给出回复（如 oracle 模式缺少该任务的原函数）。"""


# --------------------------------------------------------------------------- 回复与抽象基类


@dataclass
class Reply:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cached: bool = False
    latency_s: float = 0.0
    raw: dict = field(default_factory=dict)

    @property
    def finish_reason(self) -> str | None:
        """从 raw 里取 finish_reason（"length" 表示被 max_tokens 截断）；取不到返回 None。"""
        try:
            return self.raw["choices"][0].get("finish_reason")
        except (KeyError, IndexError, TypeError, AttributeError):
            return None

    @property
    def system_fingerprint(self) -> str | None:
        """服务端报告的系统指纹（后端版本 / 配置的标识），记进生成记录，方便事后发现「同一个模型名背后换了版本」。"""
        try:
            value = self.raw.get("system_fingerprint")
        except AttributeError:
            return None
        return value if isinstance(value, str) else None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Reply":
        """to_dict 的逆操作；缺的数值字段按 0 / False 处理。"""
        return cls(
            text=str(d.get("text") or ""),
            model=str(d.get("model", "")),
            prompt_tokens=int(d.get("prompt_tokens", 0) or 0),
            completion_tokens=int(d.get("completion_tokens", 0) or 0),
            cached=bool(d.get("cached", False)),
            latency_s=float(d.get("latency_s", 0.0) or 0.0),
            raw=dict(d.get("raw") or {}),
        )


class LLM(abc.ABC):
    """抽象基类。``model`` 是模型标签：缓存键、结果文件名、报告里都用它区分模型。"""

    model: str = ""

    @abc.abstractmethod
    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 4096,
        seed: int | None = None,
        meta: dict | None = None,
    ) -> Reply:
        """发一轮对话，返回回复。meta 只是附加信息，不属于请求内容。"""

    def fingerprint(self) -> str:
        """实例配置的摘要：会影响回复的配置（实际模型名、额外请求参数、端点等）都应并入。
        generate 阶段把它写进输入哈希，配置变了已有回复就作废。默认空串（没有额外配置）。"""
        return ""


@functools.lru_cache(maxsize=64)
def _accepts_meta(cls: type) -> bool:
    try:
        params = inspect.signature(cls.chat).parameters
    except (TypeError, ValueError):
        return False
    return "meta" in params or any(p.kind is p.VAR_KEYWORD for p in params.values())


def chat_with_meta(llm: LLM, messages: list[dict], *, temperature: float = 0.0, max_tokens: int = 4096,
                   seed: int | None = None, meta: dict | None = None) -> Reply:
    """调 ``llm.chat``：``chat`` 接受 ``meta`` 才传（设计文档 §4.7 的签名没有它，按那个签名写的 LLM 子类也要能用）。"""
    kwargs: dict[str, Any] = {"temperature": temperature, "max_tokens": max_tokens, "seed": seed}
    if meta is not None and _accepts_meta(type(llm)):
        kwargs["meta"] = meta
    return llm.chat(messages, **kwargs)


_SURROGATE_RE = re.compile("[\ud800-\udfff]")


def _clean_text(text: str) -> str:
    """去掉孤立代理项：API 返回的 JSON 里单独的 ``\\ud83d`` 转义会被 json.loads 还原成无法用 UTF-8 编码的字符，
    写缓存 / 写 generations 时会抛 UnicodeEncodeError，已经付过费的回复就丢了。换成 U+FFFD。"""
    return _SURROGATE_RE.sub("\ufffd", text)


def estimate_tokens(text: str) -> int:
    """粗略估计 token 数（4 字节约 1 个 token）。只给 MockLLM 填 usage 用，不参与任何计费。"""
    return max(1, (len(text.encode("utf-8")) + 3) // 4)


# --------------------------------------------------------------------------- 缓存


class Cache:
    """按请求内容寻址的回复缓存：``<root>/<sha256>.json``，默认 ``results/cache``。

    键 = sha256(规范化 JSON{model, messages, temperature, max_tokens})；seed 不为 None、
    extra（实例配置：端点、实际模型名、额外请求参数）非空时并入键。写入是原子的，文件损坏按未命中处理。
    ``read=False`` 时 ``get`` 一律未命中（照常写入）：generate 的 ``--no-cache`` 用它重新采样并刷新缓存。
    """

    def __init__(self, root: Path | str | None = None, *, read: bool = True) -> None:
        self.root = Path(root) if root is not None else default_cache_dir()
        self.read = read
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def make_key(
        model: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        seed: int | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": float(temperature),   # 0 与 0.0 必须得到同一个键
            "max_tokens": int(max_tokens),
        }
        if seed is not None:
            payload["seed"] = int(seed)
        if extra:
            payload["extra"] = extra
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def path_for(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def get(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 4096,
        seed: int | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> Reply | None:
        if not self.read:
            with self._lock:
                self.misses += 1
            return None
        key = self.make_key(model, messages, temperature, max_tokens, seed, extra)
        path = self.path_for(key)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or not isinstance(data.get("reply"), dict):
                raise ValueError("文件内容不是缓存记录")
            if data.get("schema_version") != SCHEMA_VERSION or data.get("key") != key:
                raise ValueError("schema_version 或 key 对不上")
            reply = Reply.from_dict(data["reply"])
        except FileNotFoundError:
            with self._lock:
                self.misses += 1
            return None
        except (OSError, ValueError, KeyError, TypeError) as e:
            log.warning("缓存文件 %s 不可用（%s），按未命中处理", path.name, e)
            with self._lock:
                self.misses += 1
            return None
        reply.cached = True
        with self._lock:
            self.hits += 1
        return reply

    def put(
        self,
        model: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        reply: Reply,
        seed: int | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> Path:
        key = self.make_key(model, messages, temperature, max_tokens, seed, extra)
        path = self.path_for(key)
        record = {
            "schema_version": SCHEMA_VERSION,
            "key": key,
            "created": now_iso(),
            "model": model,
            "request": {
                "messages": messages,
                "temperature": float(temperature),
                "max_tokens": int(max_tokens),
                "seed": seed,
                "extra": dict(extra) if extra else None,
            },
            "reply": {
                "text": reply.text,
                "model": reply.model,
                "prompt_tokens": reply.prompt_tokens,
                "completion_tokens": reply.completion_tokens,
                "latency_s": reply.latency_s,
                "raw": reply.raw,
            },
        }
        atomic_write(path, dumps_line(record))
        return path


# --------------------------------------------------------------------------- OpenAI 兼容客户端

_UNSET: Any = object()   # 构造参数的「用默认值」哨兵（None 表示显式关闭）
_RESERVED_BODY_KEYS = frozenset({"model", "messages", "stream"})
_HTTP_HINTS = {
    400: "请求格式不对",
    401: "API key 无效或已过期",
    402: "账户余额不足（Insufficient Balance）",
    403: "没有权限访问该接口或模型",
    404: "接口或模型不存在（检查 base_url 与模型名是否仍然有效）",
    422: "请求参数无效",
}
_RETRYABLE_FINISH = frozenset({"insufficient_system_resource"})
# 只缓存正常结束的回复。被 max_tokens 截断（length）、被内容过滤（content_filter）的回复不进缓存：
# 否则 temperature=0 下同一个请求以后每次都命中这份残缺回复，重跑也救不回来，只有改 max_tokens 才能绕开。
_CACHEABLE_FINISH = frozenset({None, "", "stop", "tool_calls", "function_call"})
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_HTTP_HINTS.update({code: "服务端要求重定向（检查 base_url 的协议 / 路径前缀；为避免把 key 转发给别的主机，不跟随重定向）"
                    for code in (301, 302, 303, 307, 308)})


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """不跟随重定向：urllib 会把 Authorization 头原样带到跳转目标，API key 不能因为一次重定向发给别的主机。"""

    def redirect_request(self, req, fp, code, msg, headers, newurl):   # noqa: D401, ANN001
        return None


class _BadResponse(Exception):
    """HTTP 200 但内容不能用（不是 JSON、缺 choices、服务端中断等），按可重试处理。"""


def _is_retryable_status(status: int) -> bool:
    return status in (408, 425, 429) or status >= 500


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        seconds = float(value.strip())
    except ValueError:
        return None   # HTTP-date 形式不处理，退回指数退避
    return seconds if seconds >= 0 else None


def _proxy_hint(url: str) -> str:
    """网络错误时的排查提示：这个 URL 的请求会经环境变量里的代理转发时，提醒一句（只显示 host:port，不带凭据）。"""
    parts = urllib.parse.urlsplit(url)
    proxies = urllib.request.getproxies()
    proxy = proxies.get(parts.scheme) or proxies.get("all")
    if not proxy or urllib.request.proxy_bypass(parts.hostname or ""):
        return ""
    try:
        u = urllib.parse.urlsplit(proxy)
        shown = f"{u.hostname}:{u.port}" if u.hostname and u.port else (u.hostname or "(无法解析)")
    except ValueError:
        shown = "(无法解析)"
    return f"；本次请求会经环境变量里配置的代理 {shown} 转发，若该代理没在运行，请去掉代理变量或设 BENCH_LLM_NO_PROXY=1"


class OpenAICompat(LLM):
    """调用 OpenAI 兼容的 ``/chat/completions``（DeepSeek、OpenAI、各类网关）。

    参数（位置顺序与设计文档一致：base_url, api_key, model）：

    * ``base_url``：如 ``https://api.deepseek.com``；末尾斜杠会被去掉。只接受 http / https。
    * ``api_key_env``：**存放 API key 的环境变量名**（大写蛇形，如 ``DEEPSEEK_API_KEY``），不是 key 本身；
      设计文档里写的 ``api_key(环境变量名)`` 对应它，``api_key=`` 关键字作为同义别名也接受（值同样是变量名）。
    * ``model``：模型标签，用于缓存键 / 结果文件 / 报告；实际发给 API 的名字默认同它，
      标签与 API 名字不一样时用 ``api_model``（例如 ``deepseek-flash-thinking`` 实际调 ``deepseek-flash``）。
    * ``extra_body``：并入请求体的额外字段（``{"thinking": {"type": "disabled"}}`` 等），
      不能覆盖 model / messages / stream。
    * ``timeout_s``：单次 HTTP 请求的 socket 超时，默认 120 s。
    * ``max_retries`` / ``backoff_base_s`` / ``backoff_max_s`` / ``jitter``：重试次数（默认 5 次，
      即最多 6 次请求）与指数退避：第 k 次重试前等 ``min(max, base * 2**k)`` 秒，429 带 Retry-After 时取较大者，
      再乘 ``1 + jitter * U(0,1)``（U 来自固定种子的随机数发生器，只影响等待时间，不影响任何结果）。
    * ``max_tokens_floor``：把每次请求的 max_tokens 至少抬到这个值。DeepSeek 思考模式下思维链也计入 max_tokens，
      调用方习惯性传的 4096 会把答案截断成空，思考类预设因此默认抬到 32768（只是上限，按实际用量计费）。
      缓存键仍按调用方传入的 max_tokens 算，这个下限作为实例配置并入键的 extra 部分（连同端点 base_url、实际模型名、额外请求参数）。
    * ``send_seed``：调用方传了 seed 时是否把它写进请求体。OpenAI 等端点支持 seed，默认发送；DeepSeek 的 API 文档
      没有列出这个参数，预设里关掉（seed 仍然参与缓存键，所以同一份 prompt 的多个样本各自单独请求、单独缓存）。
    * ``cache``：不传用默认的 ``results/cache``；传 ``None`` 关闭缓存。只缓存 finish_reason 正常（stop）的回复，
      被 max_tokens 截断、被内容过滤的不缓存，下次同样的请求会重新调用。
    * ``usage_path``：不传用 ``results/usage.jsonl``；传 ``None`` 不记 usage。
    * ``require_key``：True（默认）在构造时检查环境变量，缺失立即抛 ``MissingAPIKeyError``；
      False 则拖到第一次真正发请求时才检查（只回放缓存的场景不需要 key）。
    * ``use_proxy``：False 时忽略环境变量里的代理（环境变量里配了代理却没在运行时，请求会卡住或失败）；``BENCH_LLM_NO_PROXY=1`` 同效。
    """

    def __init__(
        self,
        base_url: str,
        api_key_env: str | None = None,
        model: str | None = None,
        *,
        api_key: str | None = None,
        api_model: str | None = None,
        extra_body: Mapping[str, Any] | None = None,
        timeout_s: float = 120.0,
        max_retries: int = 5,
        backoff_base_s: float = 2.0,
        backoff_max_s: float = 60.0,
        jitter: float = 0.25,
        max_tokens_floor: int | None = None,
        send_seed: bool = True,
        cache: Cache | None = _UNSET,
        usage_path: Path | str | None = _UNSET,
        require_key: bool = True,
        use_proxy: bool = True,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not isinstance(base_url, str) or urllib.parse.urlsplit(base_url).scheme not in ("http", "https"):
            raise ValueError(f"base_url 必须以 http:// 或 https:// 开头: {base_url!r}")
        if api_key_env is None:
            api_key_env = api_key
        elif api_key is not None and api_key != api_key_env:
            raise TypeError("api_key 是 api_key_env 的别名，不要同时给两个不同的值")
        if api_key_env is None or not model:
            raise TypeError("OpenAICompat 需要 base_url、api_key_env（存放 API key 的环境变量名）和 model 三个参数")
        if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", api_key_env):
            # 防止有人把 key 本身当成变量名传进来（真 key 几乎都含小写字母或连字符）：报错信息里绝不回显传入的值
            raise ValueError("api_key_env 必须是大写蛇形的环境变量名（如 DEEPSEEK_API_KEY），不是 API key 本身")
        bad = _RESERVED_BODY_KEYS & set(extra_body or {})
        if bad:
            raise ValueError(f"extra_body 不能覆盖 {sorted(bad)}")
        if max_retries < 0:
            raise ValueError("max_retries 不能为负")
        parsed = urllib.parse.urlsplit(base_url)
        if parsed.scheme == "http" and (parsed.hostname or "") not in _LOOPBACK_HOSTS:
            log.warning("base_url 是明文 http 且不是本机地址（%s），API key 会以明文发送", parsed.hostname)
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.model = model
        self.api_model = api_model or model
        self.extra_body: dict[str, Any] = dict(extra_body or {})
        self.timeout_s = float(timeout_s)
        self.max_retries = int(max_retries)
        self.backoff_base_s = float(backoff_base_s)
        self.backoff_max_s = float(backoff_max_s)
        self.jitter = float(jitter)
        self.max_tokens_floor = int(max_tokens_floor) if max_tokens_floor else None
        self.send_seed = bool(send_seed)
        self.cache: Cache | None = Cache() if cache is _UNSET else cache
        if usage_path is _UNSET:
            self.usage_path: Path | None = default_usage_path()
        else:
            self.usage_path = Path(usage_path) if usage_path is not None else None
        self._sleep = sleep
        self._rng = random.Random(0)
        no_proxy = os.environ.get("BENCH_LLM_NO_PROXY", "").strip() not in ("", "0")
        self._proxied = use_proxy and not no_proxy
        handlers: list[Any] = [_NoRedirect]
        if not self._proxied:
            handlers.append(urllib.request.ProxyHandler({}))   # 空的代理表：忽略环境变量里的代理
        self._opener = urllib.request.build_opener(*handlers)
        if require_key:
            self.check_credentials()

    # -- 构造辅助 ----------------------------------------------------------
    @classmethod
    def from_preset(cls, preset: "ModelPreset", **overrides: Any) -> "OpenAICompat":
        if preset.legacy:
            log.warning(
                "模型名 %s 在 DeepSeek 官方 Change Log 里有 2026-07-24 起弃用的通知，现行名称见 PRESETS 里的 "
                "deepseek-flash / deepseek-v4-pro；仍要使用请先确认你的端点还认这个名字", preset.name)
        kwargs: dict[str, Any] = {
            "api_model": preset.api_model,
            "extra_body": preset.extra_body,
            "max_tokens_floor": preset.max_tokens_floor,
            "send_seed": preset.send_seed,
        }
        kwargs.update(overrides)
        base_url = kwargs.pop("base_url", preset.base_url)
        api_key_env = kwargs.pop("api_key_env", preset.api_key_env)
        return cls(base_url, api_key_env, preset.name, **kwargs)

    # -- 凭据 --------------------------------------------------------------
    def _api_key(self) -> str:
        key = os.environ.get(self.api_key_env, "").strip()
        if not key:
            raise MissingAPIKeyError(
                f"环境变量 {self.api_key_env} 没有设置或为空：OpenAICompat 需要它保存 API key（只从环境变量读取，不写进任何文件）")
        return key

    def check_credentials(self) -> None:
        """key 缺失就抛 MissingAPIKeyError。CLI 可在长任务开始前先调一次。"""
        self._api_key()

    def _scrub(self, text: str) -> str:
        """把 key 的原文从要进日志 / 异常信息的文本里抹掉（服务端有时会把请求头回显）。"""
        key = os.environ.get(self.api_key_env, "").strip()
        return text.replace(key, "***") if key else text

    # -- 请求 --------------------------------------------------------------
    @property
    def url(self) -> str:
        return self.base_url + "/chat/completions"

    def _cache_extra(self) -> dict[str, Any]:
        """并入缓存键的实例配置：端点、实际发给 API 的模型名、额外请求参数、max_tokens 下限。
        同一个模型标签换了端点或思考开关，回复就可能不同，不能互相命中。"""
        extra: dict[str, Any] = {"base_url": self.base_url}
        if self.extra_body:
            extra["extra_body"] = self.extra_body
        if self.api_model != self.model:
            extra["api_model"] = self.api_model
        if self.max_tokens_floor:
            extra["max_tokens_floor"] = self.max_tokens_floor
        return extra

    def fingerprint(self) -> str:
        """实例配置的摘要：影响回复的全部配置（端点、实际模型名、额外请求参数、max_tokens 下限、是否发 seed）。"""
        cfg = dict(self._cache_extra())
        cfg["send_seed"] = self.send_seed
        cfg["api_model"] = self.api_model
        return digest(cfg)

    def _build_body(self, messages: list[dict], temperature: float, max_tokens: int, seed: int | None) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.api_model,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": max(int(max_tokens), self.max_tokens_floor or 0),
            "stream": False,
        }
        if seed is not None and self.send_seed:
            body["seed"] = int(seed)
        body.update(self.extra_body)
        return body

    def _post(self, payload: bytes, key: str) -> tuple[int, Mapping[str, str], bytes]:
        """发一次 POST，HTTP 错误状态码也当作正常返回（由调用方分类）；传输层错误向上抛。"""
        req = urllib.request.Request(
            self.url,
            data=payload,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {key}",
                "User-Agent": "codegen-robustness-bench/0.1",
            },
        )
        try:
            with self._opener.open(req, timeout=self.timeout_s) as resp:
                return resp.status, resp.headers, resp.read()
        except urllib.error.HTTPError as e:
            try:
                body = e.read()
            except (OSError, http.client.HTTPException):
                body = b""
            return e.code, e.headers, body

    def _error_message(self, body: bytes) -> str:
        """从错误响应体里取可读信息（DeepSeek / OpenAI 都是 {"error": {"message": ...}}），截断并抹掉 key。"""
        text = body.decode("utf-8", errors="replace").strip()
        try:
            data = json.loads(text)
            err = data.get("error") if isinstance(data, dict) else None
            if isinstance(err, dict) and err.get("message"):
                text = str(err["message"])
            elif isinstance(err, str):
                text = err
        except ValueError:
            pass
        return self._scrub(text)[:300]

    def _parse_ok(self, body: bytes, latency_s: float) -> Reply:
        try:
            data = json.loads(body.decode("utf-8", errors="replace"))   # 开头的空行（keep-alive）json 自己会跳过
        except ValueError as e:
            raise _BadResponse(f"不是合法 JSON（{e}）") from None
        if not isinstance(data, dict):
            raise _BadResponse("顶层不是 JSON 对象")
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            err = data.get("error")
            raise _BadResponse(f"响应里没有 choices（error={self._scrub(str(err))[:200]}）" if err else "响应里没有 choices")
        choice = choices[0]
        finish = choice.get("finish_reason")
        if finish in _RETRYABLE_FINISH:
            raise _BadResponse(f"服务端因资源不足中断了推理（finish_reason={finish}）")
        message = choice.get("message")
        if not isinstance(message, dict):
            raise _BadResponse("choices[0] 缺少 message")
        content = message.get("content")
        if content is None:
            content = ""
        if not isinstance(content, str):
            raise _BadResponse("message.content 不是字符串")
        content = _clean_text(content)
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        reply = Reply(
            text=content,
            model=self.model,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            cached=False,
            latency_s=latency_s,
            raw=data,
        )
        if finish == "length":
            log.warning("模型 %s 的回复被 max_tokens 截断（finish_reason=length，正文 %d 字符）；"
                        "思考模式下思维链也计入 max_tokens，必要时调大", self.model, len(content))
        return reply

    def _backoff_delay(self, attempt: int, retry_after: float | None) -> float:
        delay = min(self.backoff_max_s, self.backoff_base_s * (2 ** attempt))
        if retry_after is not None:
            delay = max(delay, min(retry_after, self.backoff_max_s))
        if self.jitter > 0:
            delay *= 1.0 + self.jitter * self._rng.random()
        return delay

    def _request_with_retry(self, body: dict[str, Any], key: str) -> tuple[Reply, int]:
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        last_err = "未知错误"
        for attempt in range(self.max_retries + 1):
            retry_after: float | None = None
            t0 = time.perf_counter()
            try:
                status, headers, raw_body = self._post(payload, key)
            except (OSError, http.client.HTTPException) as e:
                hint = _proxy_hint(self.url) if self._proxied else ""
                last_err = f"网络错误 {type(e).__name__}: {self._scrub(str(e))[:200]}{hint}"
            else:
                latency = time.perf_counter() - t0
                if status == 200:
                    try:
                        return self._parse_ok(raw_body, latency), attempt + 1
                    except _BadResponse as e:
                        last_err = f"HTTP 200 但响应不可用: {e}"
                elif _is_retryable_status(status):
                    last_err = f"HTTP {status}: {self._error_message(raw_body)}"
                    retry_after = _parse_retry_after(headers.get("Retry-After") if headers else None)
                else:
                    hint = _HTTP_HINTS.get(status, "请求被拒绝")
                    if status == 401:
                        hint += f"（检查环境变量 {self.api_key_env}）"
                    raise LLMHTTPError(status, f"{self.model}: HTTP {status} {hint}: {self._error_message(raw_body)}")
            if attempt < self.max_retries:
                delay = self._backoff_delay(attempt, retry_after)
                log.warning("调用 %s 失败（%s），%.1f s 后重试（第 %d/%d 次）",
                            self.model, last_err, delay, attempt + 1, self.max_retries)
                self._sleep(delay)
        raise LLMError(f"调用 {self.model} 失败，已重试 {self.max_retries} 次仍未成功: {last_err}")

    def _log_usage(self, reply: Reply, attempts: int, meta: Mapping[str, Any] | None) -> None:
        if self.usage_path is None:
            return
        usage = reply.raw.get("usage") if isinstance(reply.raw.get("usage"), dict) else {}
        details = usage.get("completion_tokens_details") if isinstance(usage.get("completion_tokens_details"), dict) else {}
        record: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "ts": now_iso(),
            "model": self.model,
            "api_model": self.api_model,
            "prompt_tokens": reply.prompt_tokens,
            "completion_tokens": reply.completion_tokens,
            "total_tokens": reply.prompt_tokens + reply.completion_tokens,
            "latency_s": round(reply.latency_s, 3),
            "attempts": attempts,
            "finish_reason": reply.finish_reason,
        }
        for k in ("prompt_cache_hit_tokens", "prompt_cache_miss_tokens"):
            if isinstance(usage.get(k), int):
                record[k] = usage[k]
        if isinstance(details.get("reasoning_tokens"), int):
            record["reasoning_tokens"] = details["reasoning_tokens"]
        for k in ("task_id", "target_id", "level", "purpose"):
            v = (meta or {}).get(k)
            if isinstance(v, (str, int)):
                record[k] = v
        try:
            append_jsonl(self.usage_path, record)
        except OSError as e:   # 记账失败不该让已经付过费的调用丢掉回复
            log.error("写 usage 日志 %s 失败: %s", self.usage_path, e)

    # -- 对外接口 ----------------------------------------------------------
    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 4096,
        seed: int | None = None,
        meta: dict | None = None,
    ) -> Reply:
        if not messages:
            raise ValueError("messages 不能为空")
        if not 0 <= temperature <= 2:
            raise ValueError(f"temperature 必须在 [0, 2]: {temperature}")
        if max_tokens < 1:
            raise ValueError(f"max_tokens 必须为正: {max_tokens}")
        extra = self._cache_extra()
        if self.cache is not None:
            hit = self.cache.get(self.model, messages, temperature, max_tokens, seed, extra)
            if hit is not None:
                return hit
        key = self._api_key()
        reply, attempts = self._request_with_retry(self._build_body(messages, temperature, max_tokens, seed), key)
        self._log_usage(reply, attempts, meta)
        if self.cache is not None:
            if reply.finish_reason not in _CACHEABLE_FINISH:
                log.info("回复的 finish_reason=%s，不写缓存（下次同样的请求会重新调用）", reply.finish_reason)
            else:
                try:
                    self.cache.put(self.model, messages, temperature, max_tokens, reply, seed, extra)
                except (OSError, ValueError) as e:     # UnicodeEncodeError 是 ValueError 的子类
                    log.error("写缓存失败（回复照常返回）: %s", e)
        return reply


# --------------------------------------------------------------------------- 命名预设


@dataclass(frozen=True)
class ModelPreset:
    name: str                      # 模型标签：--model 的取值，缓存键 / 结果文件名 / 报告里都用它
    api_model: str                 # 实际发给 API 的模型名
    base_url: str = DEEPSEEK_BASE_URL
    api_key_env: str = DEEPSEEK_KEY_ENV
    extra_body: Mapping[str, Any] = field(default_factory=dict)
    legacy: bool = False           # 官方有弃用通知的旧名字（只保留为预设，使用时给出警告）
    max_tokens_floor: int | None = None   # 思考模式的思维链计入 max_tokens，需要抬高下限
    send_seed: bool = False               # DeepSeek 文档里没有 seed 参数，不发；seed 仍参与缓存键


_NO_THINK = {"thinking": {"type": "disabled"}}
_THINK = {"thinking": {"type": "enabled"}, "reasoning_effort": "high"}

PRESETS: dict[str, ModelPreset] = {
    p.name: p
    for p in (
        # 现行模型（2026-10 官方文档）：默认思考模式，做 temperature=0 的对照实验要显式关闭
        ModelPreset("deepseek-flash", "deepseek-flash", extra_body=_NO_THINK),
        ModelPreset("deepseek-flash-thinking", "deepseek-flash", extra_body=_THINK, max_tokens_floor=32768),
        ModelPreset("deepseek-v4-pro", "deepseek-v4-pro", extra_body=_NO_THINK),
        ModelPreset("deepseek-v4-pro-thinking", "deepseek-v4-pro", extra_body=_THINK, max_tokens_floor=32768),
        # 旧名字：设计文档里用的，官方有 2026-07-24 起弃用的通知
        ModelPreset("deepseek-chat", "deepseek-chat", legacy=True),
        ModelPreset("deepseek-reasoner", "deepseek-reasoner", legacy=True),
    )
}


def make_llm(name: str, *, oracle: dict[str, str] | None = None, **kwargs: Any) -> LLM:
    """按 --model 的取值构造 LLM。

    ``mock-oracle`` / ``mock-broken`` / ``mock-wrong`` / ``mock-empty`` → MockLLM（oracle 参数给 oracle 模式用）；
    ``PRESETS`` 里的名字 → OpenAICompat（kwargs 原样传给构造函数，如 cache=None、usage_path=...）。
    """
    if name.startswith("mock-"):
        return MockLLM(name[len("mock-"):], oracle=oracle, model=name)
    preset = PRESETS.get(name)
    if preset is None:
        raise ValueError(f"未知模型 {name!r}；可选: {sorted(PRESETS)} 或 mock-<{'|'.join(MOCK_MODES)}>")
    return OpenAICompat.from_preset(preset, **kwargs)


# --------------------------------------------------------------------------- MockLLM

MOCK_MODES = ("oracle", "broken", "wrong", "empty")

# describe 阶段（meta["purpose"] == "describe"）不论什么模式都返回这段固定描述：
# 全部用常见英文词写成，不含任何项目标识符，长度在 80–250 词之间，所以一定能通过泄漏检查与长度检查。
MOCK_DESCRIPTION = (
    "This function performs one small and self contained operation of the library. "
    "It reads its arguments in the order given by the signature, and the first argument is the main input. "
    "For valid input it computes the expected result, changes only the objects that the caller passed in for "
    "that purpose, and returns the value that the caller needs. "
    "If an argument is not valid, for example a null pointer, a negative size, or a value outside the allowed "
    "range, the function does not read or write through that argument, and it returns the usual failure value "
    "of its return type, such as zero, null, or a negative number. "
    "Special cases such as empty input, zero length, and the largest or smallest possible value are handled "
    "on purpose and never cause undefined behavior. "
    "The function does not print anything and does not keep any state between calls unless the specification "
    "says so."
)

MOCK_EMPTY_TEXT = "Sorry, I am not able to provide an implementation for this request."


def _fence_c(code: str) -> str:
    return "```c\n" + code.strip("\n") + "\n```"


def mock_wrong_code(signature: str | None) -> str:
    """签名正确、函数体是「返回零值 / 空体」的函数，规则同设计文档 §4.5 的 stub：

    void → 空函数体；其余（指针 / 算术类型 / 结构体）→ 用 ``{0}`` 初始化一个返回类型的局部变量再返回，
    这样不需要知道返回类型到底是标量还是结构体。解析不出返回类型时退回 ``return 0;``。
    """
    from bench.postprocess import return_type_text, split_signature   # 延迟导入：避免 llm 在导入时就拉起 libclang

    sig = (signature or "int mock_wrong(void)").strip().rstrip(";{ \t\n")
    parts = split_signature(sig)
    if parts is None:
        return f"{sig}\n{{\n    return 0;\n}}"
    rtype = return_type_text(parts.ret)
    if not rtype:
        return f"{sig}\n{{\n    return 0;\n}}"
    if rtype == "void":
        return f"{sig}\n{{\n}}"
    return f"{sig}\n{{\n    {rtype} bench_zero_ = {{0}};\n    return bench_zero_;\n}}"


def mock_broken_code(signature: str | None) -> str:
    """签名正确、函数体缺分号且括号不配对的函数；gcc 一定报 ``expected ...``（归因为 compile_syntax）。"""
    sig = (signature or "int mock_broken(void)").strip().rstrip(";{ \t\n")
    return f"{sig}\n{{\n    int bench_tmp_ = 0\n    return (bench_tmp_;\n}}"


class MockLLM(LLM):
    """离线自测用的假模型：不联网、没有随机性，同样的输入永远给同样的输出。

    ``mode``：

    * ``oracle``：返回 ``oracle[meta["task_id"]]``（该任务在对应混淆级下的原函数定义）包在 ```c 代码块里；
      找不到任务就抛 MockLLMError（宁可大声失败，不悄悄返回错代码）。
    * ``broken``：返回签名正确、但缺分号且括号不配对的函数（签名取自 ``meta["obf_signature"]``，
      没有就从 oracle 代码的函数头取，再没有用占位签名）。
    * ``wrong``：返回签名正确、函数体为「返回零值 / 空体」的函数，能编译，但会让项目测试失败。
    * ``empty``：返回一段没有代码块的文字。

    ``meta["purpose"] == "describe"`` 时不论什么模式都返回固定的功能规格 ``MOCK_DESCRIPTION``，
    这样 describe 阶段在 mock 后端下也能走通。
    """

    def __init__(self, mode: str = "oracle", oracle: dict[str, str] | None = None, model: str | None = None) -> None:
        if mode not in MOCK_MODES:
            raise ValueError(f"MockLLM 的 mode 必须是 {MOCK_MODES} 之一: {mode!r}")
        self.mode = mode
        self.oracle: dict[str, str] = dict(oracle or {})
        self.model = model or f"mock-{mode}"
        self._lock = threading.Lock()
        self.calls: list[dict[str, Any]] = []   # 每次调用的 meta 与消息数，供测试检查

    def fingerprint(self) -> str:
        return f"mock:{self.mode}"

    def _signature(self, meta: Mapping[str, Any]) -> str | None:
        sig = meta.get("obf_signature")
        if isinstance(sig, str) and sig.strip():
            return sig.strip()
        code = self.oracle.get(meta.get("task_id"))
        if code and "{" in code:
            return code.split("{", 1)[0].strip() or None
        return None

    def _oracle_code(self, meta: Mapping[str, Any]) -> str:
        task_id = meta.get("task_id")
        if task_id is None:
            raise MockLLMError("oracle 模式需要 meta['task_id']")
        try:
            return self.oracle[task_id]
        except KeyError:
            raise MockLLMError(f"oracle 里没有任务 {task_id!r}（共 {len(self.oracle)} 个任务）") from None

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 4096,
        seed: int | None = None,
        meta: dict | None = None,
    ) -> Reply:
        meta = dict(meta or {})
        t0 = time.perf_counter()
        if meta.get("purpose") == "describe":
            text = MOCK_DESCRIPTION
        elif self.mode == "oracle":
            text = _fence_c(self._oracle_code(meta))
        elif self.mode == "broken":
            text = _fence_c(mock_broken_code(self._signature(meta)))
        elif self.mode == "wrong":
            text = _fence_c(mock_wrong_code(self._signature(meta)))
        else:
            text = MOCK_EMPTY_TEXT
        prompt_tokens = sum(estimate_tokens(str(m.get("content", ""))) for m in messages)
        completion_tokens = estimate_tokens(text)
        with self._lock:
            self.calls.append({"meta": meta, "n_messages": len(messages)})
        return Reply(
            text=text,
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached=False,
            latency_s=time.perf_counter() - t0,
            raw={
                "id": "mock",
                "model": self.model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
                "mock": {"mode": self.mode},
            },
        )
