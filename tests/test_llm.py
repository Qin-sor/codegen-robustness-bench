"""bench.llm 的测试：Reply / Cache / MockLLM / OpenAICompat / 预设。

不联网：OpenAICompat 对着本机回环地址（127.0.0.1）上起的假 OpenAI 兼容服务测，真实走 urllib；
测试里一律 use_proxy=False（环境变量里的代理可能指向没在跑的端口），重试等待用注入的 sleep 记录而不真睡。
MockLLM 产出的 C 代码另有一组 docker 沙箱测试（编译与运行只在沙箱里），docker 或镜像不可用时跳过。
临时文件放项目 tmp/ 下，测试结束删除。
"""
from __future__ import annotations

import hashlib
import http.server
import json
import logging
import re
import shutil
import socket
import threading
import time
import uuid
from pathlib import Path

import pytest

from bench import llm as L
from bench.extract import FunctionTarget, extract_functions
from bench.llm import (
    Cache,
    LLMError,
    LLMHTTPError,
    MissingAPIKeyError,
    MockLLM,
    MockLLMError,
    OpenAICompat,
    Reply,
    make_llm,
    mock_broken_code,
    mock_wrong_code,
)
from bench.sandbox import DockerSandbox
from tests.helpers import MINI, ROOT, mini_spec, needs_docker

KEY_ENV = "BENCH_TEST_API_KEY"
KEY = "sk-test-SECRET-0123456789abcdef"
MESSAGES = [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "Say hi. 你好"}]


# ---------------------------------------------------------------------------
# 夹具与假服务
# ---------------------------------------------------------------------------
@pytest.fixture()
def key(monkeypatch):
    monkeypatch.setenv(KEY_ENV, KEY)
    return KEY


@pytest.fixture(scope="module")
def mini_targets() -> dict[str, FunctionTarget]:
    return {t.name: t for t in extract_functions(mini_spec(), MINI)}


class Step:
    """假服务的一步响应。"""

    def __init__(self, status: int = 200, body: bytes | str | None = None, headers: dict | None = None, delay: float = 0.0):
        self.status = status
        self.body = body.encode("utf-8") if isinstance(body, str) else (body if body is not None else ok_body())
        self.headers = {"Content-Type": "application/json", **(headers or {})}
        self.delay = delay


def ok_body(text: str = "hello", prompt: int = 12, completion: int = 5, finish: str = "stop", model: str = "srv-model", usage_extra: dict | None = None) -> bytes:
    usage = {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion, **(usage_extra or {})}
    body = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish}],
        "usage": usage,
    }
    return json.dumps(body, ensure_ascii=False).encode("utf-8")


class FakeServer:
    """本机回环地址上的假 OpenAI 兼容服务：按脚本依次回复（用完重复最后一步），记录收到的请求。
    脚本里的元素可以是 Step，也可以是 callable(序号, 请求体 dict) -> Step。"""

    def __init__(self, script):
        self.script = list(script) or [Step()]
        self.requests: list[dict] = []
        self._lock = threading.Lock()
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length)
                with outer._lock:
                    idx = len(outer.requests)
                    outer.requests.append({"path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()}, "raw": raw})
                step = outer.script[min(idx, len(outer.script) - 1)]
                if callable(step):
                    step = step(idx, json.loads(raw))
                if step.delay:
                    time.sleep(step.delay)
                try:
                    self.send_response(step.status)
                    for k, v in step.headers.items():
                        self.send_header(k, v)
                    self.send_header("Content-Length", str(len(step.body)))
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.wfile.write(step.body)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, *args):   # 不往 stderr 打访问日志
                pass

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def bodies(self) -> list[dict]:
        return [json.loads(r["raw"]) for r in self.requests]

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture()
def serve():
    servers: list[FakeServer] = []

    def start(*script):
        s = FakeServer(script)
        servers.append(s)
        return s

    yield start
    for s in servers:
        s.close()


@pytest.fixture()
def sleeps():
    return []


@pytest.fixture()
def client(key, scratch, sleeps):
    """工厂：对着给定的假服务造一个 OpenAICompat（缓存与 usage 都放在 scratch 下，不真睡，不抖动）。"""

    def make(server: FakeServer, **kw) -> OpenAICompat:
        params = dict(
            model="m1",
            cache=Cache(scratch / "cache"),
            usage_path=scratch / "usage.jsonl",
            use_proxy=False,
            sleep=sleeps.append,
            jitter=0.0,
        )
        params.update(kw)
        model = params.pop("model")
        return OpenAICompat(server.url, KEY_ENV, model, **params)

    return make


def usage_lines(scratch: Path) -> list[dict]:
    p = scratch / "usage.jsonl"
    return [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines()] if p.exists() else []


# ---------------------------------------------------------------------------
# Reply
# ---------------------------------------------------------------------------
class TestReply:
    def test_roundtrip(self):
        r = Reply("hi 你好", "m", 3, 4, True, 0.25, {"choices": [{"finish_reason": "stop"}], "x": [1, 2]})
        assert Reply.from_dict(json.loads(json.dumps(r.to_dict(), ensure_ascii=False))) == r

    def test_positional_construction_matches_design_field_order(self):
        assert list(Reply.__dataclass_fields__) == ["text", "model", "prompt_tokens", "completion_tokens", "cached", "latency_s", "raw"]
        r = Reply("t", "m", 1, 2)
        assert (r.cached, r.latency_s, r.raw, r.total_tokens) == (False, 0.0, {}, 3)

    def test_default_raw_is_not_shared(self):
        a, b = Reply("a", "m", 0, 0), Reply("b", "m", 0, 0)
        a.raw["x"] = 1
        assert b.raw == {}

    @pytest.mark.parametrize(
        "raw, expected",
        [({"choices": [{"finish_reason": "length"}]}, "length"), ({"choices": []}, None), ({}, None), ({"choices": [{}]}, None), ({"choices": None}, None)],
    )
    def test_finish_reason(self, raw, expected):
        assert Reply("", "m", 0, 0, raw=raw).finish_reason == expected

    def test_from_dict_tolerates_missing_fields(self):
        r = Reply.from_dict({"text": "x"})
        assert (r.text, r.model, r.prompt_tokens, r.completion_tokens, r.cached, r.raw) == ("x", "", 0, 0, False, {})


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
class TestCache:
    def test_key_formula_is_pinned(self):
        payload = {"model": "m", "messages": MESSAGES, "temperature": 0.0, "max_tokens": 4096}
        expected = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
        assert Cache.make_key("m", MESSAGES, 0, 4096) == expected
        assert re.fullmatch(r"[0-9a-f]{64}", expected)

    def test_key_ignores_dict_order_and_int_float_temperature(self):
        a = Cache.make_key("m", [{"role": "user", "content": "x"}], 0, 100)
        b = Cache.make_key("m", [{"content": "x", "role": "user"}], 0.0, 100)
        assert a == b

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(model="m2"),
            dict(messages=[{"role": "user", "content": "other"}]),
            dict(temperature=0.7),
            dict(max_tokens=101),
            dict(seed=1),
            dict(extra={"thinking": {"type": "disabled"}}),
        ],
    )
    def test_key_changes_with_every_input(self, kwargs):
        base = dict(model="m", messages=[{"role": "user", "content": "x"}], temperature=0.0, max_tokens=100)
        assert Cache.make_key(**base) != Cache.make_key(**{**base, **kwargs})

    def test_seed_none_and_empty_extra_keep_the_plain_formula(self):
        base = Cache.make_key("m", MESSAGES, 0.0, 10)
        assert Cache.make_key("m", MESSAGES, 0.0, 10, None, None) == base
        assert Cache.make_key("m", MESSAGES, 0.0, 10, None, {}) == base
        assert Cache.make_key("m", MESSAGES, 0.0, 10, 1) != Cache.make_key("m", MESSAGES, 0.0, 10, 2)

    def test_put_get_roundtrip(self, scratch):
        c = Cache(scratch / "cache")
        assert c.get("m", MESSAGES, 0.0, 100) is None
        reply = Reply("答案 ok", "m", 7, 9, False, 1.5, {"choices": [{"finish_reason": "stop"}]})
        path = c.put("m", MESSAGES, 0.0, 100, reply)
        assert path.parent == scratch / "cache" and re.fullmatch(r"[0-9a-f]{64}\.json", path.name)
        got = c.get("m", MESSAGES, 0.0, 100)
        assert got is not None and got.cached is True
        assert (got.text, got.model, got.prompt_tokens, got.completion_tokens, got.latency_s, got.raw) == (reply.text, "m", 7, 9, 1.5, reply.raw)
        assert (c.hits, c.misses) == (1, 1)
        record = json.loads(path.read_text(encoding="utf-8"))
        assert record["schema_version"] == 1 and record["key"] == path.stem and record["request"]["messages"] == MESSAGES

    def test_different_requests_do_not_collide(self, scratch):
        c = Cache(scratch / "cache")
        c.put("m", MESSAGES, 0.0, 100, Reply("A", "m", 1, 1))
        c.put("m", MESSAGES, 0.0, 200, Reply("B", "m", 1, 1))
        assert c.get("m", MESSAGES, 0.0, 100).text == "A"
        assert c.get("m", MESSAGES, 0.0, 200).text == "B"
        assert c.get("m", MESSAGES, 0.5, 100) is None

    @pytest.mark.parametrize("corrupt", ["", "not json", '{"schema_version": 1, "key": "wrong", "reply": {}}', '{"schema_version": 99}', '{"schema_version": 1}', "[]"])
    def test_corrupt_file_is_a_miss_and_can_be_overwritten(self, scratch, corrupt, caplog):
        c = Cache(scratch / "cache")
        path = c.path_for(Cache.make_key("m", MESSAGES, 0.0, 100))
        path.parent.mkdir(parents=True)
        path.write_text(corrupt, encoding="utf-8")
        with caplog.at_level(logging.WARNING, logger="bench.llm"):
            assert c.get("m", MESSAGES, 0.0, 100) is None
        assert "不可用" in caplog.text
        c.put("m", MESSAGES, 0.0, 100, Reply("fresh", "m", 1, 1))
        assert c.get("m", MESSAGES, 0.0, 100).text == "fresh"

    def test_no_temp_files_left_behind(self, scratch):
        c = Cache(scratch / "cache")
        for i in range(5):
            c.put("m", MESSAGES, 0.0, 100 + i, Reply(str(i), "m", 1, 1))
        assert sorted(p.suffix for p in (scratch / "cache").iterdir()) == [".json"] * 5

    def test_concurrent_writers_of_the_same_key(self, scratch):
        c = Cache(scratch / "cache")
        errors = []

        def work(i):
            try:
                for _ in range(20):
                    c.put("m", MESSAGES, 0.0, 100, Reply(f"v{i}", "m", 1, 1))
                    got = c.get("m", MESSAGES, 0.0, 100)
                    assert got is not None and got.text.startswith("v")
            except Exception as e:   # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=work, args=(i,)) for i in range(6)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        assert errors == []

    def test_default_root_follows_results_dir_env(self, scratch, monkeypatch):
        monkeypatch.setenv("BENCH_RESULTS_DIR", str(scratch / "res"))
        assert Cache().root == scratch / "res" / "cache"
        assert L.default_usage_path() == scratch / "res" / "usage.jsonl"
        monkeypatch.delenv("BENCH_RESULTS_DIR")
        assert Cache().root == L.PROJECT_ROOT / "results" / "cache"


# ---------------------------------------------------------------------------
# MockLLM
# ---------------------------------------------------------------------------
SIG = "int calc_gcd(int a, int b)"
ORACLE_CODE = "int calc_gcd(int a, int b)\n{\n    return b == 0 ? a : calc_gcd(b, a % b);\n}"


def meta_for(task_id="t1", sig=SIG, level="L1"):
    return {"task_id": task_id, "level": level, "obf_signature": sig}


class TestMockLLM:
    def test_oracle_returns_the_oracle_function_in_a_c_block(self):
        llm = MockLLM("oracle", {"t1": ORACLE_CODE})
        r = llm.chat(MESSAGES, meta=meta_for())
        assert r.text == f"```c\n{ORACLE_CODE}\n```"
        assert (r.model, r.cached) == ("mock-oracle", False)
        assert r.prompt_tokens > 0 and r.completion_tokens > 0 and r.finish_reason == "stop"

    def test_oracle_missing_task_is_a_loud_error(self):
        llm = MockLLM("oracle", {"t1": ORACLE_CODE})
        with pytest.raises(MockLLMError, match="t2"):
            llm.chat(MESSAGES, meta=meta_for("t2"))
        with pytest.raises(MockLLMError, match="task_id"):
            llm.chat(MESSAGES)
        with pytest.raises(LLMError):
            MockLLM("oracle").chat(MESSAGES, meta=meta_for())

    def test_oracle_keeps_levels_apart_by_task_id(self):
        llm = MockLLM("oracle", {"x:L0": "int f(void) { return 0; }", "x:L1": "int f_1(void) { return 0; }"})
        assert "f_1" in llm.chat(MESSAGES, meta={"task_id": "x:L1"}).text
        assert "f_1" not in llm.chat(MESSAGES, meta={"task_id": "x:L0"}).text

    def test_broken_has_missing_semicolon_unbalanced_parenthesis_and_the_right_signature(self):
        text = MockLLM("broken").chat(MESSAGES, meta=meta_for()).text
        code = text.split("```c\n")[1].split("\n```")[0]
        assert code.startswith(SIG + "\n{")
        assert code.count("(") != code.count(")")
        assert re.search(r"=\s*0\s*\n\s*return", code)   # 赋值语句后面缺分号
        assert code.count("{") == code.count("}")         # 花括号是平衡的：这不是「被截断」

    def test_wrong_has_the_right_signature_and_a_zero_body(self):
        code = MockLLM("wrong").chat(MESSAGES, meta=meta_for()).text.split("```c\n")[1].split("\n```")[0]
        assert code == f"{SIG}\n{{\n    int bench_zero_ = {{0}};\n    return bench_zero_;\n}}"

    def test_empty_has_no_code_block(self):
        text = MockLLM("empty").chat(MESSAGES, meta=meta_for()).text
        assert text and "```" not in text

    @pytest.mark.parametrize("mode", ["oracle", "broken", "wrong", "empty"])
    def test_describe_purpose_returns_the_canned_spec_in_every_mode(self, mode):
        r = MockLLM(mode, {}).chat(MESSAGES, meta={"purpose": "describe", "target_id": "x"})
        assert r.text == L.MOCK_DESCRIPTION and "```" not in r.text

    def test_signature_falls_back_to_oracle_header_then_placeholder(self):
        wrong = MockLLM("wrong", {"t1": ORACLE_CODE})
        assert wrong.chat(MESSAGES, meta={"task_id": "t1"}).text.startswith("```c\nint calc_gcd(int a, int b)\n{")
        assert "int mock_wrong(void)" in MockLLM("wrong").chat(MESSAGES, meta={}).text
        assert "int mock_broken(void)" in MockLLM("broken").chat(MESSAGES).text

    def test_deterministic_and_stateless(self):
        llm = MockLLM("wrong")
        a = llm.chat(MESSAGES, meta=meta_for()).text
        llm.chat(MESSAGES, meta=meta_for("other", "void g(void)"))
        assert llm.chat(MESSAGES, meta=meta_for()).text == a

    def test_unknown_mode_rejected(self):
        with pytest.raises(ValueError, match="mode"):
            MockLLM("cheerful")

    def test_model_label(self):
        assert MockLLM("wrong").model == "mock-wrong"
        assert MockLLM("wrong", model="custom").chat(MESSAGES).model == "custom"

    def test_calls_are_recorded_thread_safely(self):
        llm = MockLLM("empty")
        threads = [threading.Thread(target=lambda: [llm.chat(MESSAGES, meta={"task_id": "t"}) for _ in range(50)]) for _ in range(8)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        assert len(llm.calls) == 400

    def test_does_not_touch_results_directory(self, scratch, monkeypatch):
        monkeypatch.setenv("BENCH_RESULTS_DIR", str(scratch / "res"))
        MockLLM("oracle", {"t1": ORACLE_CODE}).chat(MESSAGES, meta=meta_for())
        assert not (scratch / "res").exists()


WRONG_CASES = [
    ("int add(int a, int b)", "int"),
    ("static int helper(int x)", "int"),
    ("static inline char sdsReqType(size_t string_size)", "char"),
    ("const char *op_name(op_kind_t op)", "const char *"),
    ("static const te_variable *find_builtin(const char *name, int len)", "const te_variable *"),
    ("point_t point_make(int x, int y)", "point_t"),
    ("static mpc_state_t mpc_state_invalid(void)", "mpc_state_t"),
    ("static unsigned long long int gcd(unsigned long long int a, unsigned long long int b)", "unsigned long long int"),
    ("static enum patch_operation decode_patch_operation(const cJSON * const patch)", "enum patch_operation"),
    ("CJSON_PUBLIC(cJSON *) cJSON_ParseWithOpts(const char *value, const char **end, cJSON_bool flag)", "cJSON *"),
    ("MINI_RET(int) calc_abs_diff(int a, int b)", "int"),
    ("JSMN_API int jsmn_parse(jsmn_parser *parser, const char *js, const size_t len)", "int"),
    ("M_23 f_9(t_1 *v_2)", "f_9"),   # 混淆后装饰宏 + typedef 名：保留最后一个标识符做类型
    ("mpc_err_t *mpca_lang_file(int flags, FILE *f, ...)", "mpc_err_t *"),
    ("void *mpc_malloc(mpc_input_t *i, size_t n)", "void *"),
]


class TestMockWrongCode:
    @pytest.mark.parametrize("sig, rtype", [c for c in WRONG_CASES if c[0] != "M_23 f_9(t_1 *v_2)"])
    def test_zero_stub_per_return_type(self, sig, rtype):
        assert mock_wrong_code(sig) == f"{sig}\n{{\n    {rtype} bench_zero_ = {{0}};\n    return bench_zero_;\n}}"

    @pytest.mark.parametrize("sig", ["void rect_normalize(struct rect *r)", "static void update_offset(printbuffer * const buffer)", "CJSON_PUBLIC(void) cJSON_InitHooks(cJSON_Hooks* hooks)", "JSMN_API void jsmn_init(jsmn_parser *parser)"])
    def test_void_functions_get_an_empty_body(self, sig):
        assert mock_wrong_code(sig) == f"{sig}\n{{\n}}"

    def test_decorated_obfuscated_name_pair(self):
        # 返回类型文本只有「装饰宏 + 一个标识符」时，最后一个标识符被当作类型
        assert "M_23 f_9(t_1 *v_2)" in mock_wrong_code("M_23 f_9(t_1 *v_2)")

    @pytest.mark.parametrize("sig", ["int (*get_op(int which))(int, int)", "garbage", "", None])
    def test_unrecognized_signature_falls_back_to_return_zero(self, sig):
        code = mock_wrong_code(sig)
        assert code.endswith("{\n    return 0;\n}") or "mock_wrong" in code

    def test_trailing_semicolon_in_signature_is_dropped(self):
        assert mock_wrong_code("int f(void);") == "int f(void)\n{\n    int bench_zero_ = {0};\n    return bench_zero_;\n}"


class TestMockBrokenCode:
    def test_shape(self):
        assert mock_broken_code("void f(int a);") == "void f(int a)\n{\n    int bench_tmp_ = 0\n    return (bench_tmp_;\n}"

    def test_placeholder_signature(self):
        assert mock_broken_code(None).startswith("int mock_broken(void)\n{")


# ---------------------------------------------------------------------------
# MockLLM 产出的 C 代码在沙箱里的表现（编译与运行只在 docker 里）
# ---------------------------------------------------------------------------
def make_variant(dest: Path, target: FunctionTarget, new_code: str) -> None:
    shutil.copytree(MINI, dest, ignore=shutil.ignore_patterns("spec.json"))
    path = dest / target.file
    data = path.read_bytes()
    path.write_bytes(data[: target.start] + new_code.encode("utf-8") + data[target.end :])


def original_of(t: FunctionTarget) -> str:
    return (MINI / t.file).read_bytes()[t.start : t.end].decode("utf-8")


# 项目自带测试一定覆盖得到的函数：返回零值 / 空体的桩会让 make test 失败
KILLABLE = ["calc_gcd", "point_make", "rect_normalize", "str_dup_upper", "stack_init", "op_name", "calc_div_round", "counter_next", "prime_at", "array_reverse"]


@needs_docker
class TestMockCodeInSandbox:
    @pytest.fixture(scope="class")
    def outcome(self, mini_targets):
        base = ROOT / "tmp" / "pytest-llm" / f"sbx-{uuid.uuid4().hex[:10]}"
        variants = base / "variants"
        variants.mkdir(parents=True)
        try:
            for name, t in mini_targets.items():
                make_variant(variants / f"syn_wrong_{name}", t, mock_wrong_code(t.signature))
                make_variant(variants / f"syn_broken_{name}", t, mock_broken_code(t.signature))
            for name in KILLABLE:
                make_variant(variants / f"test_wrong_{name}", mini_targets[name], mock_wrong_code(mini_targets[name].signature))
            for name in ("calc_gcd", "rect_normalize"):
                make_variant(variants / f"test_oracle_{name}", mini_targets[name], original_of(mini_targets[name]))
            script = (
                "cd variants\n"
                "for d in syn_*; do ( cd \"$d\" && gcc -std=c99 -fsyntax-only -fdiagnostics-format=json calc.c util.c 2> diag.txt >/dev/null; echo $? > rc.txt ); done\n"
                "for d in test_*; do ( cd \"$d\" && make test > out.txt 2>&1; echo $? > rc.txt ); done\n"
            )
            (base / "run.sh").write_text(script, encoding="utf-8")
            res = DockerSandbox(timeout_s=300).run(base, "bash run.sh")
            assert res.rc == 0, res.stderr[-500:]
            yield variants
        finally:
            shutil.rmtree(base, ignore_errors=True)

    @staticmethod
    def rc(d: Path) -> int:
        return int((d / "rc.txt").read_text().strip())

    @staticmethod
    def errors(d: Path) -> list[str]:
        msgs = []
        for line in (d / "diag.txt").read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line.startswith("["):
                msgs += [x["message"] for x in json.loads(line) if x.get("kind") == "error"]
        return msgs

    def test_wrong_stubs_compile_for_every_mini_target(self, outcome, mini_targets):
        failed = [n for n in mini_targets if self.rc(outcome / f"syn_wrong_{n}") != 0]
        assert failed == []

    def test_broken_stubs_fail_with_expected_diagnostics_for_every_mini_target(self, outcome, mini_targets):
        for n in mini_targets:
            d = outcome / f"syn_broken_{n}"
            assert self.rc(d) != 0, n
            assert any("expected" in m for m in self.errors(d)), (n, self.errors(d))

    def test_wrong_stubs_make_project_tests_fail(self, outcome):
        passed = [n for n in KILLABLE if self.rc(outcome / f"test_wrong_{n}") == 0]
        assert passed == []

    def test_original_code_still_passes(self, outcome):
        for n in ("calc_gcd", "rect_normalize"):
            assert self.rc(outcome / f"test_oracle_{n}") == 0, (outcome / f"test_oracle_{n}" / "out.txt").read_text()[-400:]


# ---------------------------------------------------------------------------
# OpenAICompat（假服务）
# ---------------------------------------------------------------------------
class TestOpenAICompatSuccess:
    def test_request_shape_and_reply(self, serve, client, scratch, key):
        srv = serve(Step(body=ok_body("答案", prompt=21, completion=8)))
        llm = client(srv)
        r = llm.chat(MESSAGES, temperature=0.0, max_tokens=300)
        assert (r.text, r.model, r.prompt_tokens, r.completion_tokens, r.cached) == ("答案", "m1", 21, 8, False)
        assert r.latency_s > 0 and r.raw["model"] == "srv-model" and r.finish_reason == "stop"
        (req,) = srv.requests
        assert req["path"] == "/chat/completions"
        assert req["headers"]["authorization"] == f"Bearer {key}"
        assert req["headers"]["content-type"] == "application/json"
        assert "codegen-robustness-bench" in req["headers"]["user-agent"]
        assert srv.bodies()[0] == {"model": "m1", "messages": MESSAGES, "temperature": 0.0, "max_tokens": 300, "stream": False}

    def test_base_url_trailing_slash_and_version_prefix(self, serve, client):
        srv = serve(Step())
        OpenAICompat(srv.url + "/v1/", KEY_ENV, "m", cache=None, usage_path=None, use_proxy=False).chat(MESSAGES)
        assert srv.requests[0]["path"] == "/v1/chat/completions"

    def test_seed_is_sent_only_when_given(self, serve, client):
        srv = serve(Step())
        llm = client(srv, cache=None)
        llm.chat(MESSAGES)
        llm.chat(MESSAGES, seed=7)
        assert "seed" not in srv.bodies()[0] and srv.bodies()[1]["seed"] == 7

    def test_send_seed_false_keeps_seed_out_of_the_request_but_in_the_cache_key(self, serve, client):
        srv = serve(Step())
        llm = client(srv, send_seed=False)
        llm.chat(MESSAGES, seed=1)
        llm.chat(MESSAGES, seed=2)
        assert all("seed" not in b for b in srv.bodies()) and len(srv.requests) == 2   # 样本各自请求
        assert llm.chat(MESSAGES, seed=1).cached is True

    def test_extra_body_is_merged_and_protected_keys_rejected(self, serve, client):
        srv = serve(Step())
        client(srv, extra_body={"thinking": {"type": "disabled"}, "user_id": "u1"}, cache=None).chat(MESSAGES)
        body = srv.bodies()[0]
        assert body["thinking"] == {"type": "disabled"} and body["user_id"] == "u1" and body["model"] == "m1"
        with pytest.raises(ValueError, match="extra_body"):
            client(srv, extra_body={"messages": []})

    def test_api_model_differs_from_label(self, serve, client):
        srv = serve(Step())
        r = client(srv, model="label-x", api_model="real-y", cache=None).chat(MESSAGES)
        assert srv.bodies()[0]["model"] == "real-y" and r.model == "label-x"

    def test_leading_blank_lines_before_json_are_fine(self, serve, client):
        srv = serve(Step(body=b"\n\n\n" + ok_body("kept alive")))
        assert client(srv).chat(MESSAGES).text == "kept alive"

    def test_null_content_becomes_empty_text(self, serve, client):
        body = json.dumps({"choices": [{"message": {"role": "assistant", "content": None}, "finish_reason": "length"}], "usage": {}}).encode()
        srv = serve(Step(body=body))
        r = client(srv).chat(MESSAGES)
        assert r.text == "" and r.finish_reason == "length" and (r.prompt_tokens, r.completion_tokens) == (0, 0)

    def test_length_truncation_is_logged(self, serve, client, caplog):
        srv = serve(Step(body=ok_body("cut", finish="length")))
        with caplog.at_level(logging.WARNING, logger="bench.llm"):
            assert client(srv).chat(MESSAGES).text == "cut"
        assert "max_tokens" in caplog.text

    def test_meta_does_not_change_the_request(self, serve, client):
        srv = serve(Step())
        llm = client(srv, cache=None)
        llm.chat(MESSAGES, meta={"task_id": "t", "level": "L1", "obf_signature": "int f(void)"})
        llm.chat(MESSAGES)
        a, b = srv.bodies()
        assert a == b


class TestOpenAICompatCacheAndUsage:
    def test_second_identical_call_is_served_from_cache(self, serve, client, scratch):
        srv = serve(Step(body=ok_body("once", prompt=10, completion=3)))
        llm = client(srv)
        first = llm.chat(MESSAGES, max_tokens=50)
        second = llm.chat(MESSAGES, max_tokens=50)
        assert (first.cached, second.cached) == (False, True)
        assert second.text == "once" and second.prompt_tokens == 10
        assert len(srv.requests) == 1
        assert len(usage_lines(scratch)) == 1   # 命中缓存的不记 usage

    def test_cache_survives_a_new_client(self, serve, client):
        srv = serve(Step())
        client(srv).chat(MESSAGES)
        assert client(srv).chat(MESSAGES).cached is True
        assert len(srv.requests) == 1

    @pytest.mark.parametrize("change", [dict(temperature=0.5), dict(max_tokens=77), dict(seed=3)])
    def test_changed_parameters_miss_the_cache(self, serve, client, change):
        srv = serve(Step())
        llm = client(srv)
        llm.chat(MESSAGES)
        assert llm.chat(MESSAGES, **change).cached is False
        assert len(srv.requests) == 2

    @pytest.mark.parametrize("finish", ["length", "content_filter"])
    def test_incomplete_replies_are_not_cached(self, serve, client, scratch, finish):
        srv = serve(Step(body=ok_body("cut off", finish=finish)), Step(body=ok_body("complete")))
        llm = client(srv)
        assert llm.chat(MESSAGES).text == "cut off"
        assert list((scratch / "cache").glob("*.json")) == []      # 残缺回复不进缓存
        second = llm.chat(MESSAGES)                                 # 同样的请求会重新调用
        assert (second.text, second.cached) == ("complete", False) and len(srv.requests) == 2
        assert llm.chat(MESSAGES).cached is True                    # 完整的回复才缓存
        assert len(usage_lines(scratch)) == 2                       # 两次真实调用都记了 usage

    def test_reply_without_finish_reason_is_cached(self, serve, client):
        body = json.dumps({"choices": [{"message": {"role": "assistant", "content": "ok"}}], "usage": {}}).encode()
        srv = serve(Step(body=body))
        llm = client(srv)
        llm.chat(MESSAGES)
        assert llm.chat(MESSAGES).cached is True and len(srv.requests) == 1

    def test_cache_key_is_the_documented_tuple_plus_the_instance_config(self, serve, client, scratch):
        # 键 = 文档里的四元组 + extra（端点 base_url 等实例配置）：同一个模型标签换了端点，回复可能不同，不能互相命中
        srv = serve(Step())
        llm = client(srv)
        assert llm._cache_extra() == {"base_url": srv.url}
        llm.chat(MESSAGES, max_tokens=100)
        assert [p.stem for p in (scratch / "cache").glob("*.json")] == [
            Cache.make_key("m1", MESSAGES, 0.0, 100, None, {"base_url": srv.url})]
        assert Cache.make_key("m1", MESSAGES, 0.0, 100, None, {"base_url": srv.url}) != Cache.make_key("m1", MESSAGES, 0.0, 100)

    def test_other_endpoint_does_not_hit_the_cache_of_this_one(self, serve, client):
        a, b = serve(Step()), serve(Step())
        shared = Cache(client(a).cache.root)
        client(a, cache=shared).chat(MESSAGES)
        assert client(b, cache=shared).chat(MESSAGES).cached is False

    def test_fingerprint_changes_with_everything_that_changes_the_reply(self, serve, client):
        srv = serve(Step())
        base = client(srv).fingerprint()
        assert base == client(srv).fingerprint()
        assert base != client(srv, api_model="other").fingerprint()
        assert base != client(srv, extra_body={"thinking": {"type": "enabled"}}).fingerprint()
        assert base != client(srv, max_tokens_floor=32768).fingerprint()
        assert base != client(srv, send_seed=False).fingerprint()
        assert MockLLM("oracle").fingerprint() != MockLLM("wrong").fingerprint()

    def test_no_read_cache_always_calls_the_server_but_still_refreshes_the_cache(self, serve, client, scratch):
        srv = serve(Step())
        root = scratch / "cache"
        client(srv, cache=Cache(root)).chat(MESSAGES)
        assert len(srv.requests) == 1
        refreshing = client(srv, cache=Cache(root, read=False))
        assert refreshing.chat(MESSAGES).cached is False and len(srv.requests) == 2
        assert client(srv, cache=Cache(root)).chat(MESSAGES).cached is True and len(srv.requests) == 2

    def test_lone_surrogates_in_a_reply_do_not_break_caching_or_logging(self, serve, client, scratch):
        # JSON 里单独的 \ud83d 转义会被 json.loads 还原成无法用 UTF-8 编码的字符：以前写缓存时抛 UnicodeEncodeError，整个阶段中断
        body = ('{"choices":[{"message":{"role":"assistant","content":"bad \\ud83d tail"},"finish_reason":"stop"}],'
                '"usage":{"prompt_tokens":1,"completion_tokens":1}}')
        srv = serve(Step(body=body))
        llm = client(srv)
        reply = llm.chat(MESSAGES)
        assert reply.text == "bad \ufffd tail"
        reply.text.encode("utf-8")
        assert llm.chat(MESSAGES).cached is True

    def test_cache_write_failure_for_unencodable_text_is_not_fatal(self, serve, client, monkeypatch):
        srv = serve(Step())
        llm = client(srv)

        def boom(*a, **k):
            raise UnicodeEncodeError("utf-8", "x", 0, 1, "surrogates not allowed")

        monkeypatch.setattr(llm.cache, "put", boom)
        assert llm.chat(MESSAGES).text == "hello"

    @pytest.mark.parametrize("status, fatal", [(401, True), (402, True), (403, True), (400, False), (404, False), (422, False)])
    def test_http_errors_that_doom_the_whole_batch_are_marked_fatal(self, status, fatal):
        assert LLMHTTPError(status, "x").fatal is fatal

    def test_cache_extra_is_namespaced(self, serve, client):
        srv = serve(Step())
        a = client(srv, model="m", api_model="x")
        b = client(srv, model="m", extra_body={"api_model": "x"})
        assert a._cache_extra() != b._cache_extra()
        assert Cache.make_key("m", MESSAGES, 0.0, 10, None, a._cache_extra()) != Cache.make_key("m", MESSAGES, 0.0, 10, None, b._cache_extra())

    def test_cache_none_always_calls_the_server(self, serve, client):
        srv = serve(Step())
        llm = client(srv, cache=None)
        llm.chat(MESSAGES)
        llm.chat(MESSAGES)
        assert len(srv.requests) == 2

    def test_usage_record_fields(self, serve, client, scratch):
        extra = {"prompt_cache_hit_tokens": 4, "prompt_cache_miss_tokens": 8, "completion_tokens_details": {"reasoning_tokens": 2}}
        srv = serve(Step(body=ok_body("x", prompt=12, completion=5, usage_extra=extra)))
        client(srv).chat(MESSAGES, meta={"task_id": "t9", "level": "L2", "purpose": "generate", "obf_signature": "int secret_sig(void)"})
        (rec,) = usage_lines(scratch)
        assert rec["schema_version"] == 1 and rec["model"] == "m1" and rec["api_model"] == "m1"
        assert (rec["prompt_tokens"], rec["completion_tokens"], rec["total_tokens"]) == (12, 5, 17)
        assert (rec["prompt_cache_hit_tokens"], rec["prompt_cache_miss_tokens"], rec["reasoning_tokens"]) == (4, 8, 2)
        assert (rec["task_id"], rec["level"], rec["purpose"], rec["attempts"], rec["finish_reason"]) == ("t9", "L2", "generate", 1, "stop")
        assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\+00:00", rec["ts"]) and rec["latency_s"] >= 0
        text = (scratch / "usage.jsonl").read_text(encoding="utf-8")
        assert "secret_sig" not in text and "Say hi" not in text   # 只记用量与标识，不记 prompt 内容

    def test_usage_is_appended_not_overwritten(self, serve, client, scratch):
        srv = serve(Step())
        llm = client(srv, cache=None)
        for _ in range(3):
            llm.chat(MESSAGES)
        client(srv, cache=None).chat(MESSAGES)
        assert len(usage_lines(scratch)) == 4

    def test_usage_path_none_writes_nothing(self, serve, client, scratch):
        srv = serve(Step())
        client(srv, usage_path=None).chat(MESSAGES)
        assert not (scratch / "usage.jsonl").exists()

    def test_defaults_land_under_results_dir(self, serve, key, scratch, monkeypatch):
        monkeypatch.setenv("BENCH_RESULTS_DIR", str(scratch / "res"))
        srv = serve(Step())
        OpenAICompat(srv.url, KEY_ENV, "m", use_proxy=False).chat(MESSAGES)
        assert (scratch / "res" / "usage.jsonl").is_file() and len(list((scratch / "res" / "cache").glob("*.json"))) == 1

    def test_concurrent_calls_keep_usage_lines_intact(self, serve, client, scratch):
        srv = serve(lambda i, body: Step(body=ok_body(f"r{body['messages'][1]['content']}")))
        llm = client(srv)
        results: dict[int, str] = {}

        def work(w):
            for j in range(5):
                n = w * 10 + j
                results[n] = llm.chat([{"role": "system", "content": "s"}, {"role": "user", "content": str(n)}], meta={"task_id": f"t{n}"}).text

        threads = [threading.Thread(target=work, args=(w,)) for w in range(8)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        assert {n: f"r{n}" for n in results} == results and len(results) == 40
        lines = usage_lines(scratch)
        assert len(lines) == 40 and {rec["task_id"] for rec in lines} == {f"t{n}" for n in results}
        assert len(list((scratch / "cache").glob("*.json"))) == 40


class TestOpenAICompatRetries:
    @pytest.mark.parametrize("status", [429, 500, 502, 503, 504, 408])
    def test_retryable_status_then_success(self, serve, client, sleeps, status):
        srv = serve(Step(status, '{"error": {"message": "busy"}}'), Step(), )
        r = client(srv).chat(MESSAGES)
        assert r.text == "hello" and len(srv.requests) == 2 and sleeps == [2.0]

    def test_exponential_backoff_sequence_and_exhaustion(self, serve, client, sleeps):
        srv = serve(Step(503, '{"error": {"message": "overloaded"}}'))
        with pytest.raises(LLMError, match="已重试 5 次") as ei:
            client(srv).chat(MESSAGES)
        assert len(srv.requests) == 6   # 首次 + 5 次重试
        assert sleeps == [2.0, 4.0, 8.0, 16.0, 32.0]
        assert "503" in str(ei.value) and "overloaded" in str(ei.value)

    def test_backoff_is_capped(self, serve, client, sleeps):
        srv = serve(Step(500, "oops"))
        with pytest.raises(LLMError):
            client(srv, backoff_max_s=10.0).chat(MESSAGES)
        assert sleeps == [2.0, 4.0, 8.0, 10.0, 10.0]

    def test_custom_retry_count_and_base(self, serve, client, sleeps):
        srv = serve(Step(500, "oops"))
        with pytest.raises(LLMError):
            client(srv, max_retries=2, backoff_base_s=0.5).chat(MESSAGES)
        assert len(srv.requests) == 3 and sleeps == [0.5, 1.0]

    def test_zero_retries_means_single_attempt(self, serve, client, sleeps):
        srv = serve(Step(500, "oops"))
        with pytest.raises(LLMError):
            client(srv, max_retries=0).chat(MESSAGES)
        assert len(srv.requests) == 1 and sleeps == []

    def test_retry_after_header_is_honoured(self, serve, client, sleeps):
        srv = serve(Step(429, "{}", {"Retry-After": "7"}), Step(429, "{}", {"Retry-After": "0"}), Step())
        client(srv).chat(MESSAGES)
        assert sleeps == [7.0, 4.0]   # 取 Retry-After 与指数退避里较大的

    def test_retry_after_is_capped_by_backoff_max(self, serve, client, sleeps):
        srv = serve(Step(429, "{}", {"Retry-After": "9999"}), Step())
        client(srv, backoff_max_s=30.0).chat(MESSAGES)
        assert sleeps == [30.0]

    def test_jitter_stays_within_bounds_and_is_reproducible(self, serve, client, sleeps):
        srv = serve(Step(500, "oops"))
        runs = []
        for _ in range(2):
            sleeps.clear()
            with pytest.raises(LLMError):
                client(srv, jitter=0.25).chat(MESSAGES)
            runs.append(list(sleeps))
        assert runs[0] == runs[1]
        for got, base in zip(runs[0], [2.0, 4.0, 8.0, 16.0, 32.0]):
            assert base <= got <= base * 1.25

    @pytest.mark.parametrize("code", [301, 302, 307, 308])
    def test_redirects_are_not_followed_and_authorization_is_not_forwarded(self, serve, client, sleeps, code):
        target = serve(Step())
        srv = serve(Step(code, "", {"Location": target.url + "/stolen"}))
        with pytest.raises(LLMHTTPError, match="重定向") as ei:
            client(srv).chat(MESSAGES)
        assert ei.value.status == code and target.requests == [] and len(srv.requests) == 1 and sleeps == []

    @pytest.mark.parametrize("status, needle", [(400, "请求格式"), (401, "API key"), (402, "余额"), (403, "权限"), (404, "不存在"), (422, "参数")])
    def test_non_retryable_status_fails_immediately(self, serve, client, sleeps, status, needle):
        srv = serve(Step(status, '{"error": {"message": "server says no"}}'))
        with pytest.raises(LLMHTTPError) as ei:
            client(srv).chat(MESSAGES)
        assert ei.value.status == status and needle in str(ei.value) and "server says no" in str(ei.value)
        assert len(srv.requests) == 1 and sleeps == []

    def test_401_names_the_env_var_but_never_the_key(self, serve, client, key):
        srv = serve(Step(401, '{"error": {"message": "bad key"}}'))
        with pytest.raises(LLMHTTPError) as ei:
            client(srv).chat(MESSAGES)
        assert KEY_ENV in str(ei.value) and key not in str(ei.value)

    def test_connection_refused_is_retried_then_reported(self, key, scratch, sleeps):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()   # 没人监听的端口
        llm = OpenAICompat(f"http://127.0.0.1:{port}", KEY_ENV, "m", cache=None, usage_path=None, use_proxy=False, max_retries=2, jitter=0.0, sleep=sleeps.append)
        with pytest.raises(LLMError, match="网络错误"):
            llm.chat(MESSAGES)
        assert sleeps == [2.0, 4.0]

    def test_timeout_is_retried(self, serve, client, sleeps):
        srv = serve(Step(body=ok_body(), delay=0.6))
        with pytest.raises(LLMError, match="网络错误"):
            client(srv, timeout_s=0.15, max_retries=1).chat(MESSAGES)
        assert len(srv.requests) == 2 and sleeps == [2.0]

    @pytest.mark.parametrize(
        "body, needle",
        [
            ("this is not json", "JSON"),
            ("[1, 2]", "JSON 对象"),
            ('{"choices": []}', "choices"),
            ('{"error": {"message": "internal"}}', "choices"),
            ('{"choices": [{"message": 5}]}', "message"),
            ('{"choices": [{"message": {"content": 5}}]}', "content"),
            ('{"choices": [{"message": {"content": "x"}, "finish_reason": "insufficient_system_resource"}]}', "资源不足"),
        ],
    )
    def test_unusable_200_responses_are_retried_then_reported(self, serve, client, sleeps, body, needle):
        srv = serve(Step(200, body))
        with pytest.raises(LLMError, match="响应不可用") as ei:
            client(srv, max_retries=1).chat(MESSAGES)
        assert needle in str(ei.value) and len(srv.requests) == 2

    def test_recovers_after_a_bad_200(self, serve, client):
        srv = serve(Step(200, "<html>gateway</html>"), Step())
        assert client(srv).chat(MESSAGES).text == "hello"

    def test_usage_records_attempt_count(self, serve, client, scratch):
        srv = serve(Step(500, "x"), Step(500, "x"), Step())
        client(srv).chat(MESSAGES)
        assert usage_lines(scratch)[0]["attempts"] == 3


class TestOpenAICompatCredentials:
    def test_missing_key_raises_at_construction(self, monkeypatch):
        monkeypatch.delenv("BENCH_NO_SUCH_KEY", raising=False)
        with pytest.raises(MissingAPIKeyError, match="BENCH_NO_SUCH_KEY"):
            OpenAICompat("https://example.invalid", "BENCH_NO_SUCH_KEY", "m")

    def test_empty_key_counts_as_missing(self, monkeypatch):
        monkeypatch.setenv(KEY_ENV, "   ")
        with pytest.raises(MissingAPIKeyError):
            OpenAICompat("https://example.invalid", KEY_ENV, "m")

    def test_lazy_check_with_require_key_false(self, serve, monkeypatch, scratch):
        monkeypatch.delenv(KEY_ENV, raising=False)
        srv = serve(Step())
        llm = OpenAICompat(srv.url, KEY_ENV, "m", require_key=False, cache=Cache(scratch / "cache"), usage_path=None, use_proxy=False)
        with pytest.raises(MissingAPIKeyError, match=KEY_ENV):
            llm.chat(MESSAGES)
        assert srv.requests == []
        with pytest.raises(MissingAPIKeyError):
            llm.check_credentials()

    def test_cache_hits_need_no_key(self, serve, monkeypatch, scratch):
        monkeypatch.setenv(KEY_ENV, KEY)
        srv = serve(Step())
        cache = Cache(scratch / "cache")
        OpenAICompat(srv.url, KEY_ENV, "m", cache=cache, usage_path=None, use_proxy=False).chat(MESSAGES)
        monkeypatch.delenv(KEY_ENV)
        replay = OpenAICompat(srv.url, KEY_ENV, "m", require_key=False, cache=cache, usage_path=None, use_proxy=False)
        assert replay.chat(MESSAGES).cached is True

    def test_api_key_keyword_is_an_alias_of_api_key_env(self, serve, key, scratch):
        srv = serve(Step())
        llm = OpenAICompat(base_url=srv.url, api_key=KEY_ENV, model="m", cache=None, usage_path=None, use_proxy=False)
        assert llm.api_key_env == KEY_ENV and llm.chat(MESSAGES).text == "hello"
        same = OpenAICompat(srv.url, KEY_ENV, "m", api_key=KEY_ENV, cache=None, usage_path=None, use_proxy=False)
        assert same.api_key_env == KEY_ENV
        with pytest.raises(TypeError, match="别名"):
            OpenAICompat(srv.url, KEY_ENV, "m", api_key="OTHER_NAME")

    def test_missing_constructor_arguments(self, serve, key):
        srv = serve(Step())
        for kwargs in (dict(), dict(model="m"), dict(api_key_env=KEY_ENV)):
            with pytest.raises(TypeError, match="三个参数"):
                OpenAICompat(srv.url, **kwargs)

    @pytest.mark.parametrize("bad", ["sk-abc123SECRET", "has space", "", "1STARTS_WITH_DIGIT", "a-b", "deepseek_api_key", "abcdef0123456789abcdef0123456789", "AbC"])
    def test_api_key_env_must_be_a_variable_name(self, bad):
        with pytest.raises((ValueError, TypeError)) as ei:
            OpenAICompat("https://example.invalid", bad, "m")
        assert bad not in str(ei.value) or bad == ""   # 不回显传入的值（它可能就是 key）
        with pytest.raises((ValueError, TypeError)) as ei2:
            OpenAICompat("https://example.invalid", api_key=bad, model="m")   # 别名同样检查
        assert bad not in str(ei2.value) or bad == ""

    @pytest.mark.parametrize("url", ["", "ftp://x", "file:///etc/passwd", "api.deepseek.com", "javascript:alert(1)"])
    def test_base_url_must_be_http(self, url, key):
        with pytest.raises(ValueError, match="base_url"):
            OpenAICompat(url, KEY_ENV, "m")

    def test_plain_http_to_a_remote_host_warns(self, key, caplog):
        with caplog.at_level(logging.WARNING, logger="bench.llm"):
            OpenAICompat("http://llm.example.invalid/v1", KEY_ENV, "m", cache=None, usage_path=None)
            OpenAICompat("http://127.0.0.1:9", KEY_ENV, "m", cache=None, usage_path=None)
            OpenAICompat("https://llm.example.invalid", KEY_ENV, "m", cache=None, usage_path=None)
        warnings = [r for r in caplog.records if "明文" in r.getMessage()]
        assert len(warnings) == 1 and "llm.example.invalid" in warnings[0].getMessage()

    def test_argument_validation(self, serve, client):
        llm = client(serve(Step()))
        for kwargs in (dict(temperature=-0.1), dict(temperature=2.5), dict(max_tokens=0)):
            with pytest.raises(ValueError):
                llm.chat(MESSAGES, **kwargs)
        with pytest.raises(ValueError, match="messages"):
            llm.chat([])
        with pytest.raises(ValueError):
            client(serve(Step()), max_retries=-1)

    def test_key_never_reaches_logs_files_or_errors(self, serve, client, scratch, key, caplog):
        echo = json.dumps({"error": {"message": f"invalid credentials {key} for request"}})
        srv = serve(Step(500, echo), Step(), Step(500, echo))
        llm = client(srv, max_retries=1)
        with caplog.at_level(logging.DEBUG):
            r = llm.chat(MESSAGES)   # 第 1 次 500（回显了 key）→ 重试成功
            with pytest.raises(LLMError) as ei:
                llm.chat([{"role": "user", "content": "other"}], max_tokens=10)
        assert key not in caplog.text and key not in str(ei.value)
        assert "***" in caplog.text
        for p in scratch.rglob("*"):
            if p.is_file():
                assert key not in p.read_text(encoding="utf-8", errors="replace"), p
        assert key not in json.dumps(r.to_dict(), ensure_ascii=False)

    def test_repr_has_no_secret(self, serve, client, key):
        assert key not in repr(client(serve(Step())))
        assert key not in repr(vars(client(serve(Step()))))


class TestProxyHandling:
    def test_use_proxy_false_ignores_proxy_env(self, serve, key, monkeypatch):
        for var in ("no_proxy", "NO_PROXY"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("http_proxy", "http://127.0.0.1:1")
        srv = serve(Step())
        assert OpenAICompat(srv.url, KEY_ENV, "m", cache=None, usage_path=None, use_proxy=False).chat(MESSAGES).text == "hello"

    def test_env_switch_disables_proxy(self, serve, key, monkeypatch):
        for var in ("no_proxy", "NO_PROXY"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("http_proxy", "http://127.0.0.1:1")
        monkeypatch.setenv("BENCH_LLM_NO_PROXY", "1")
        srv = serve(Step())
        assert OpenAICompat(srv.url, KEY_ENV, "m", cache=None, usage_path=None).chat(MESSAGES).text == "hello"

    def test_dead_proxy_error_carries_a_hint_without_credentials(self, serve, key, monkeypatch, sleeps):
        for var in ("no_proxy", "NO_PROXY", "BENCH_LLM_NO_PROXY"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("http_proxy", "http://user:hunter2@127.0.0.1:1")
        srv = serve(Step())
        llm = OpenAICompat(srv.url, KEY_ENV, "m", cache=None, usage_path=None, max_retries=0, sleep=sleeps.append)
        with pytest.raises(LLMError) as ei:
            llm.chat(MESSAGES)
        msg = str(ei.value)
        assert "127.0.0.1:1" in msg and "BENCH_LLM_NO_PROXY" in msg and "hunter2" not in msg
        assert srv.requests == []


# ---------------------------------------------------------------------------
# 预设与 make_llm
# ---------------------------------------------------------------------------
class TestPresets:
    def test_preset_table(self):
        p = L.PRESETS
        assert p["deepseek-flash"].extra_body == {"thinking": {"type": "disabled"}}
        assert p["deepseek-flash-thinking"].extra_body["thinking"] == {"type": "enabled"}
        assert p["deepseek-flash-thinking"].api_model == "deepseek-flash"
        assert p["deepseek-v4-pro"].api_model == "deepseek-v4-pro"
        assert p["deepseek-chat"].legacy and p["deepseek-reasoner"].legacy and not p["deepseek-flash"].legacy
        assert all(x.base_url == "https://api.deepseek.com" and x.api_key_env == "DEEPSEEK_API_KEY" for x in p.values())
        assert all(name == x.name for name, x in p.items())

    def test_make_llm_mock(self):
        m = make_llm("mock-wrong")
        assert isinstance(m, MockLLM) and m.mode == "wrong" and m.model == "mock-wrong"
        assert make_llm("mock-oracle", oracle={"t": "int f(void){return 0;}"}).oracle == {"t": "int f(void){return 0;}"}
        with pytest.raises(ValueError):
            make_llm("mock-nonsense")

    def test_make_llm_unknown_name_lists_choices(self):
        with pytest.raises(ValueError, match="deepseek-flash"):
            make_llm("gpt-nonexistent")

    def test_non_thinking_preset_request(self, serve, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", KEY)
        srv = serve(Step())
        llm = make_llm("deepseek-flash", base_url=srv.url, use_proxy=False, cache=None, usage_path=None)
        r = llm.chat(MESSAGES)
        body = srv.bodies()[0]
        assert body["model"] == "deepseek-flash" and body["thinking"] == {"type": "disabled"} and r.model == "deepseek-flash"

    def test_thinking_preset_request_and_label(self, serve, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", KEY)
        srv = serve(Step())
        llm = make_llm("deepseek-flash-thinking", base_url=srv.url, use_proxy=False, cache=None, usage_path=None)
        assert llm.api_key_env == "DEEPSEEK_API_KEY"
        r = llm.chat(MESSAGES, max_tokens=4096)
        body = srv.bodies()[0]
        assert body["model"] == "deepseek-flash" and body["thinking"] == {"type": "enabled"} and body["reasoning_effort"] == "high"
        assert body["max_tokens"] == 32768   # 思考模式：思维链也计入 max_tokens，下限把 4096 抬上去
        assert r.model == "deepseek-flash-thinking" and llm.api_model == "deepseek-flash"

    def test_deepseek_presets_do_not_send_seed(self, serve, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", KEY)
        srv = serve(Step())
        make_llm("deepseek-flash", base_url=srv.url, use_proxy=False, cache=None, usage_path=None).chat(MESSAGES, seed=5)
        assert "seed" not in srv.bodies()[0]
        assert all(not p.send_seed for p in L.PRESETS.values())

    def test_non_thinking_preset_keeps_callers_max_tokens(self, serve, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", KEY)
        srv = serve(Step())
        make_llm("deepseek-flash", base_url=srv.url, use_proxy=False, cache=None, usage_path=None).chat(MESSAGES, max_tokens=4096)
        assert srv.bodies()[0]["max_tokens"] == 4096

    def test_thinking_and_non_thinking_never_share_cache_entries(self, serve, monkeypatch, scratch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", KEY)
        srv = serve(Step())
        cache = Cache(scratch / "cache")
        a = make_llm("deepseek-flash", base_url=srv.url, use_proxy=False, cache=cache, usage_path=None)
        b = make_llm("deepseek-flash-thinking", base_url=srv.url, use_proxy=False, cache=cache, usage_path=None)
        a.chat(MESSAGES)
        assert b.chat(MESSAGES).cached is False
        assert a.chat(MESSAGES).cached is True and b.chat(MESSAGES).cached is True
        assert len(srv.requests) == 2

    def test_max_tokens_floor_never_lowers_and_is_part_of_the_cache_identity(self, serve, client):
        srv = serve(Step())
        shared = Cache(Path(client(srv).cache.root))
        floored = client(srv, cache=shared, max_tokens_floor=9000)
        plain = client(srv, cache=shared)
        floored.chat(MESSAGES, max_tokens=100)
        floored.chat(MESSAGES, max_tokens=20000)
        assert [b["max_tokens"] for b in srv.bodies()] == [9000, 20000]
        assert floored.chat(MESSAGES, max_tokens=100).cached is True     # 缓存键按调用方传入的 max_tokens
        assert plain.chat(MESSAGES, max_tokens=100).cached is False      # 没有下限的同名实例不会误用

    def test_same_label_with_different_extra_body_does_not_collide(self, serve, client):
        srv = serve(Step())
        shared = Cache(Path(client(srv).cache.root))
        a = client(srv, cache=shared, extra_body={"thinking": {"type": "disabled"}})
        b = client(srv, cache=shared, extra_body={"thinking": {"type": "enabled"}})
        a.chat(MESSAGES)
        assert b.chat(MESSAGES).cached is False

    def test_legacy_name_logs_a_warning(self, serve, key, monkeypatch, caplog):
        monkeypatch.setenv("DEEPSEEK_API_KEY", KEY)
        srv = serve(Step())
        with caplog.at_level(logging.WARNING, logger="bench.llm"):
            llm = make_llm("deepseek-chat", base_url=srv.url, use_proxy=False, cache=None, usage_path=None)
        assert "2026-07-24" in caplog.text
        assert llm.chat(MESSAGES).model == "deepseek-chat" and srv.bodies()[0]["model"] == "deepseek-chat"
        assert "thinking" not in srv.bodies()[0]

    def test_missing_default_key_is_reported_with_variable_name(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        with pytest.raises(MissingAPIKeyError, match="DEEPSEEK_API_KEY"):
            make_llm("deepseek-flash")


# ---------------------------------------------------------------------------
# 其它约束
# ---------------------------------------------------------------------------
class TestHygiene:
    def test_no_key_like_literals_in_sources(self):
        for name in ("llm.py", "describe.py", "prompt.py", "postprocess.py"):
            text = (ROOT / "bench" / name).read_text(encoding="utf-8")
            assert not re.search(r"sk-[A-Za-z0-9]{8,}", text), name

    def test_llm_is_abstract(self):
        with pytest.raises(TypeError):
            L.LLM()   # type: ignore[abstract]

    def test_importing_llm_does_not_load_libclang(self):
        import subprocess
        import sys

        code = "import sys; import bench.llm; print('clang.cindex' in sys.modules)"
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, timeout=60)
        assert out.stdout.strip() == "False", out.stderr
