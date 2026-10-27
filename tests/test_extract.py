"""bench.extract 的测试：mini 夹具上的函数清单 / 依赖闭包 / 字节偏移往返，以及 cJSON 的只读提取。

mini 夹具的编译与运行只走 docker 沙箱（--network none、--cap-drop ALL 等），宿主机不直接 make。
临时文件放在项目的 tmp/ 下，测试结束即清理。
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import shutil
import uuid
from pathlib import Path

import clang.cindex as ci
import pytest

from bench import corpus
from bench.clangcfg import builtin_include_args, clang_args
from bench.corpus import CorpusSpec
from bench.extract import (
    Dependencies,
    ExtractError,
    FunctionTarget,
    error_diagnostics,
    extract_file,
    extract_functions,
    locate,
    parse_file,
    squash_whitespace,
    strip_c_comments,
)
from bench.sandbox import DockerSandbox
from tests.helpers import IMAGE as SANDBOX_IMAGE, MINI, ROOT, needs_docker, require_real_corpus

# 手数：calc.c / util.c 里的函数定义（按出现顺序），sq_plus_one 是单语句 static inline，extract_functions 会排除
CALC_FUNCS = [
    "op_add", "op_sub", "op_mul", "op_div", "sign_of", "mini_version", "calc_apply", "op_name",
    "calc_abs_diff", "calc_clamp", "calc_add_sat", "calc_div_round", "calc_gcd", "calc_median3", "calc_in_range",
    "calc_score", "point_make", "point_add", "point_dist2", "rect_area", "rect_normalize", "rect_contains",
]
UTIL_FUNCS = [
    "sum_array", "weighted_sum", "sum_positive", "array_find", "array_max_gap", "array_reverse", "array_count_if",
    "is_even_value", "array_count_even", "str_dup_upper", "parse_digits", "stack_init", "stack_push",
    "stack_pop", "counter_next", "rand_seed", "rand_next", "prime_at",
]


# --------------------------------------------------------------------------- 夹具


@pytest.fixture(scope="module")
def mini_targets(mini_spec) -> dict[str, FunctionTarget]:
    return {t.name: t for t in extract_functions(mini_spec, MINI)}


@pytest.fixture()
def workdir():
    """项目 tmp/ 下的一次性工作目录。"""
    base = ROOT / "tmp" / "pytest"
    base.mkdir(parents=True, exist_ok=True)
    d = base / uuid.uuid4().hex[:12]
    d.mkdir()
    yield d
    shutil.rmtree(d, ignore_errors=True)


def copy_mini(dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    for f in MINI.iterdir():
        if f.is_file():
            shutil.copy2(f, dest / f.name)
    return dest


@pytest.fixture()
def mini_copy(workdir) -> Path:
    return copy_mini(workdir / "mini")


def norm(texts: list[str]) -> list[str]:
    """多行文本压成单行再比较，避免和夹具里的缩进绑死。"""
    return [" ".join(t.split()) for t in texts]


# --------------------------------------------------------------------------- 解析与函数清单


def test_mini_files_parse_without_errors(mini_spec):
    for rel in ("calc.c", "util.c", "test_main.c"):
        tu = parse_file(mini_spec, MINI, rel)
        assert error_diagnostics(tu) == [], rel


def test_function_inventory_matches_hand_count(mini_spec, mini_targets):
    # 手数：独立于 libclang，用正则数每个文件里的定义头——
    # 行首以标识符开头、含 '(' 、不含 ';' 的多行头，加上「签名和花括号体写在同一行」的单行定义
    head_re = re.compile(r"^[A-Za-z_][^;\n]*\([^;\n]*$", re.M)
    oneliner_re = re.compile(r"^[A-Za-z_][^\n]*\([^\n]*\)\s*\{[^\n]*\}\s*$", re.M)
    for rel, names in (("calc.c", CALC_FUNCS), ("util.c", UTIL_FUNCS)):
        text = (MINI / rel).read_text(encoding="utf-8")
        hand = len(head_re.findall(text)) + len(oneliner_re.findall(text))
        all_defs = extract_file(mini_spec, MINI, rel)
        assert [t.name for t in all_defs if t.name != "sq_plus_one"] == names
        assert len(all_defs) == hand, f"{rel}: libclang {len(all_defs)} vs 手数 {hand}"
    assert sorted(mini_targets) == sorted(CALC_FUNCS + UTIL_FUNCS)
    assert len(mini_targets) == 40


def test_static_inline_single_statement_is_excluded(mini_spec, mini_targets):
    assert "sq_plus_one" not in mini_targets          # static inline + 单语句：排除
    assert "sign_of" in mini_targets                   # static inline 但多语句：保留
    unfiltered = {t.name: t for t in extract_file(mini_spec, MINI, "calc.c")}
    sq = unfiltered["sq_plus_one"]
    assert sq.signature == "static inline int sq_plus_one(int x)"
    assert sq.n_statements == 1
    assert norm(sq.deps.macros) == ["#define MINI_SQUARE(x) ((x) * (x))"]


def test_main_and_non_target_files(mini_spec):
    # 默认 target_files 只有 calc.c / util.c：测试文件里的函数根本不会被解析出来
    assert not any(t.file == "test_main.c" for t in extract_functions(mini_spec, MINI))
    # 即使把测试文件当目标，main 也被排除；extract_file 不做排除，能看到 main
    spec = dataclasses.replace(mini_spec, target_files=["test_main.c"])
    names = [t.name for t in extract_functions(spec, MINI)]
    assert "main" not in names
    assert {"str_eq", "is_even", "is_positive", "test_calc_apply"} <= set(names)
    assert "main" in [t.name for t in extract_file(spec, MINI, "test_main.c")]


def test_target_id_corpus_file_and_signature(mini_targets):
    t = mini_targets["calc_apply"]
    assert (t.target_id, t.corpus, t.file) == ("mini:calc.c:calc_apply", "mini", "calc.c")
    assert mini_targets["rand_next"].target_id == "mini:util.c:rand_next"
    sigs = {
        "calc_apply": "int calc_apply(op_kind_t op, int a, int b)",
        "op_name": "const char *op_name(op_kind_t op)",
        "point_add": "point_t point_add(point_t a, point_t b)",
        "str_dup_upper": "char *str_dup_upper(const char *s)",
        "sign_of": "static inline int sign_of(int x)",
        "op_div": "static int op_div(int a, int b)",
        # 返回类型由宏包装：签名保留宏调用原文，宏定义靠 deps.macros 带入
        "calc_abs_diff": "MINI_RET(int) calc_abs_diff(int a, int b)",
        # 多行签名压成一行
        "rect_contains": "int rect_contains(const struct rect *r, int x, int y)",
        # 空宏 MINI_API 不在 libclang 报告的定义范围内（前缀留在原地）
        "mini_version": "int mini_version(void)",
    }
    for name, sig in sigs.items():
        assert mini_targets[name].signature == sig


def test_offsets_are_consistent_with_source(mini_targets):
    for t in mini_targets.values():
        data = (MINI / t.file).read_bytes()
        assert 0 <= t.start < t.body_start < t.body_end <= t.end <= len(data)
        assert data[t.body_start : t.body_start + 1] == b"{" and data[t.body_end - 1 : t.body_end] == b"}"
        assert t.end == t.body_end
        head = data[t.start : t.body_start].decode("utf-8")
        assert squash_whitespace(strip_c_comments(head)) == t.signature
        assert re.search(rf"\b{t.name}\b", t.signature)
        assert t.n_lines == data[t.start : t.end].count(b"\n") + 1
    # 空宏前缀留在 [start, end) 之外
    t = mini_targets["mini_version"]
    data = (MINI / "calc.c").read_bytes()
    assert data[t.start - len(b"MINI_API ") : t.start] == b"MINI_API "
    # 宏包装的返回类型在定义范围之内
    t = mini_targets["calc_abs_diff"]
    assert (MINI / "calc.c").read_bytes()[t.start : t.start + 13] == b"MINI_RET(int)"


def test_offsets_are_byte_offsets_not_char_offsets(mini_targets):
    # util.c 的第一个函数前有中文注释：字节偏移和字符偏移必然错开
    data = (MINI / "util.c").read_bytes()
    t = mini_targets["sum_array"]
    assert len(data[: t.start].decode("utf-8")) < t.start
    assert data[t.start : t.end].decode("utf-8").startswith("int sum_array(")
    text = data.decode("utf-8")
    assert text[t.start : t.start + 14] != "int sum_array("   # 当成字符偏移就会切歪


EXPECTED_STATEMENTS = {
    # calc.c
    "op_add": 1, "op_sub": 1, "op_mul": 1, "op_div": 4, "sign_of": 5, "mini_version": 1,
    "calc_apply": 9, "op_name": 7, "calc_abs_diff": 1, "calc_clamp": 1, "calc_add_sat": 5,
    "calc_div_round": 9, "calc_gcd": 3, "calc_median3": 10, "calc_in_range": 1, "calc_score": 3, "point_make": 4,
    "point_add": 3, "point_dist2": 3, "rect_area": 4, "rect_normalize": 6, "rect_contains": 1,
    # util.c
    "sum_array": 5, "weighted_sum": 5, "sum_positive": 7, "array_find": 5, "array_max_gap": 7, "array_reverse": 11,
    "array_count_if": 6, "is_even_value": 1, "array_count_even": 2, "str_dup_upper": 12,
    "parse_digits": 14, "stack_init": 4, "stack_push": 5, "stack_pop": 6, "counter_next": 2,
    "rand_seed": 1, "rand_next": 2, "prime_at": 3,
}


def test_n_statements_counts_compound_children_recursively(mini_targets):
    assert {n: t.n_statements for n, t in mini_targets.items()} == EXPECTED_STATEMENTS
    # 规则的几个关键点：
    # op_name：函数体 2 条（switch、return）+ switch 的花括号里 5 条（4 个 case + default）；case 后面的子语句不重复计
    assert mini_targets["op_name"].n_statements == 7
    # parse_digits：标号语句 fail: 算 1 条，它后面的 return 是同级兄弟，goto 在各自的 if 花括号里
    assert mini_targets["parse_digits"].n_statements == 14
    # 至少 6 个函数有 3 条以上语句（夹具要求）
    assert sum(1 for t in mini_targets.values() if t.n_statements >= 3) >= 6


def test_callees(mini_targets):
    expected = {
        "calc_score": ["sq_plus_one", "sign_of"],   # 按函数体内首次调用的顺序
        "point_add": ["point_make"],
        "array_max_gap": ["calc_abs_diff"],          # 跨文件调用
        "array_count_even": ["array_count_if"],      # is_even_value 只是被取地址传入，不算直接调用
    }
    for name, t in mini_targets.items():
        assert t.callees == expected.get(name, []), name
    # 自递归不算 callee；经函数指针的调用、系统库函数也不算
    assert mini_targets["calc_gcd"].callees == []
    assert mini_targets["calc_apply"].callees == []
    assert mini_targets["str_dup_upper"].callees == []


# --------------------------------------------------------------------------- 依赖闭包

POINT_T = "typedef struct point { int x; int y; } point_t;"
RECT = "struct rect { point_t lo; point_t hi; };"
INTSTACK = "typedef struct intstack { int items[MINI_STACK_CAP]; size_t len; } intstack_t;"
OP_KIND = "typedef enum { OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_COUNT } op_kind_t;"
BINOP = "typedef int (*binop_fn)(int, int);"
PRED = "typedef int (*pred_fn)(int);"
ROUNDING = "enum rounding { ROUND_DOWN, ROUND_UP };"
G_CALLS = "extern int g_calls;"
G_ERR = "extern const char *g_last_error;"
M_MIN = "#define MINI_MIN(a, b) ((a) < (b) ? (a) : (b))"
M_MAX = "#define MINI_MAX(a, b) ((a) > (b) ? (a) : (b))"
M_CLAMP = "#define MINI_CLAMP(x, lo, hi) MINI_MIN(MINI_MAX((x), (lo)), (hi))"
M_SQUARE = "#define MINI_SQUARE(x) ((x) * (x))"
M_RET = "#define MINI_RET(type) type"
M_CAP = "#define MINI_STACK_CAP 8"

# name -> (decls, types, globals, macros)
EXPECTED_DEPS = {
    "op_div": ([], [], [G_ERR], []),
    "mini_version": ([], [], [], ["#define MINI_VERSION 3"]),
    # 函数指针调用不产生 callee；g_ops 是 calc.c 里的 static 全局，extern 声明去掉了 static 和初始化器
    "calc_apply": ([], [OP_KIND, BINOP], [G_CALLS, G_ERR, "extern const binop_fn g_ops[OP_COUNT];"], []),
    "op_name": ([], [OP_KIND], [], []),
    "calc_abs_diff": ([], [], [], [M_RET]),
    # 宏体里引用的其它项目宏一并带入（传递闭包），按定义顺序
    "calc_clamp": ([], [], [], [M_MIN, M_MAX, M_CLAMP]),
    # INT_MAX / INT_MIN 是系统头宏：不收，靠 includes 带入
    "calc_add_sat": ([], [], [], []),
    "calc_div_round": ([], [ROUNDING], [G_ERR], []),
    "calc_gcd": ([], [], [], []),
    "calc_score": (["static inline int sq_plus_one(int x);", "static inline int sign_of(int x);"], [], [], []),
    "point_make": ([], [POINT_T], [], []),
    "point_add": (["point_t point_make(int x, int y);"], [POINT_T], [G_CALLS], []),
    "point_dist2": ([], [POINT_T], [], [M_SQUARE]),
    # 类型按依赖顺序：被 struct rect 按值包含的 point_t 在前
    "rect_area": ([], [POINT_T, RECT], [], []),
    "rect_normalize": ([], [POINT_T, RECT], [], [M_MIN, M_MAX]),
    "rect_contains": ([], [POINT_T, RECT], [], []),
    "array_max_gap": (["MINI_RET(int) calc_abs_diff(int a, int b);"], [], [], [M_RET]),
    "array_count_if": ([], [PRED], [], []),
    # is_even_value 只被取地址：它的原型进 decls；原型取头文件里最早的声明，否则取定义头
    "array_count_even": (
        ["int array_count_if(const int *a, size_t n, pred_fn pred);", "static int is_even_value(int x);"],
        [PRED], [], [],
    ),
    "str_dup_upper": ([], [], [], []),
    "parse_digits": ([], [], [G_ERR], []),
    # 宏 MINI_STACK_CAP 只出现在类型定义的数组长度里：随类型单元一起带入
    "stack_init": ([], [INTSTACK], [], [M_CAP]),
    "stack_push": ([], [INTSTACK], [G_ERR], [M_CAP]),
    "stack_pop": ([], [INTSTACK], [G_ERR], [M_CAP]),
    "counter_next": ([], [], [G_CALLS], []),
    "rand_seed": ([], [], ["extern unsigned int s_seed;"], []),
    "rand_next": ([], [], ["extern unsigned int s_seed;"], []),
    "prime_at": ([], [], ["extern const int g_primes[5];"], []),
    # 省略长度的 static 数组：extern 声明里补上由初始化器推出的长度，sizeof 才能用
    "weighted_sum": ([], [], ["extern const int s_weights[5];"], []),
}
CALC_INCLUDES = ["#include <limits.h>", "#include <stddef.h>"]
UTIL_INCLUDES = ["#include <ctype.h>", "#include <stdlib.h>", "#include <string.h>", "#include <stddef.h>"]


def test_dependency_closure(mini_targets):
    for name, t in mini_targets.items():
        decls, types_, globals_, macros = EXPECTED_DEPS.get(name, ([], [], [], []))
        got = t.deps
        assert norm(got.decls) == decls, f"{name}.decls"
        assert norm(got.types) == types_, f"{name}.types"
        assert norm(got.globals) == globals_, f"{name}.globals"
        assert norm(got.macros) == macros, f"{name}.macros"
        assert got.includes == (CALC_INCLUDES if t.file == "calc.c" else UTIL_INCLUDES), f"{name}.includes"


def test_dependencies_have_no_system_entities_or_duplicates(mini_targets):
    system_names = ("strlen", "malloc", "toupper", "INT_MAX", "INT_MIN", "NULL", "memset")
    for t in mini_targets.values():
        for field in dataclasses.fields(Dependencies):
            items = getattr(t.deps, field.name)
            assert len(items) == len(set(items)), (t.name, field.name)
            assert all(isinstance(x, str) and x.strip() for x in items)
        for text in t.deps.decls + t.deps.types + t.deps.globals + t.deps.macros:
            for bad in system_names:
                assert not re.search(rf"\b{bad}\b", text), (t.name, text)
        # 目标自己的原型不进 decls（签名里已经有了）
        assert not any(re.search(rf"\b{t.name}\s*\(", d) for d in t.deps.decls), t.name


def test_type_definitions_come_in_translation_unit_order(mini_targets):
    """被依赖的类型在前：point_t 先于按值包含它的 struct rect。"""
    types_ = norm(mini_targets["rect_area"].deps.types)
    assert types_.index(POINT_T) < types_.index(RECT)
    # 所有目标里：只要同时出现，顺序都和头文件里的声明顺序一致
    header_order = [OP_KIND, ROUNDING, POINT_T, RECT, INTSTACK, BINOP, PRED]
    for t in mini_targets.values():
        got = [x for x in norm(t.deps.types) if x in header_order]
        assert got == sorted(got, key=header_order.index), t.name


def test_doc_comments(mini_targets):
    docs = {n: t.doc_comment for n, t in mini_targets.items()}
    assert docs["calc_apply"].startswith("/** 按运算类型对两个整数做运算")   # 文档注释写在头文件原型上，也能取到
    assert docs["op_name"] == "// 运算名：switch + 指针返回"
    assert docs["sum_array"].startswith("/* 对数组求和（中文注释")
    assert docs["array_max_gap"].startswith("/* 相邻元素差的最大值")
    assert docs["is_even_value"].startswith("/* 静态辅助函数")
    assert docs["sign_of"].startswith("/* 多语句的 static inline")
    assert docs["calc_add_sat"].startswith("/* 饱和加法")
    # 头文件里的「/* ---- calc.c ---- */」是分隔标题，不是函数文档
    assert docs["mini_version"] is None
    assert docs["calc_gcd"] is None and docs["point_make"] is None


def test_function_target_dict_roundtrip(mini_targets):
    for t in mini_targets.values():
        again = FunctionTarget.from_dict(json.loads(json.dumps(t.to_dict(), ensure_ascii=False)))
        assert again == t
        assert isinstance(again.deps, Dependencies)


def test_cache_returns_independent_copies(mini_spec):
    a = extract_functions(mini_spec, MINI)
    a[0].callees.append("tampered")
    a[0].deps.types.append("tampered")
    b = extract_functions(mini_spec, MINI)
    assert "tampered" not in b[0].callees and "tampered" not in b[0].deps.types


# --------------------------------------------------------------------------- 偏移往返与替换


def test_offset_roundtrip_leaves_file_bytes_unchanged(mini_spec, mini_copy):
    for rel in ("calc.c", "util.c"):
        path = mini_copy / rel
        data = path.read_bytes()
        targets = extract_file(mini_spec, mini_copy, rel)
        out = data
        for t in reversed(targets):           # 从后往前：前面的偏移不会因后面的改动失效
            piece = out[t.start : t.end]      # 切出来
            out = out[: t.start] + piece + out[t.end :]   # 再原样写回
        path.write_bytes(out)
        assert path.read_bytes() == data
        assert extract_file(mini_spec, mini_copy, rel) == targets


def test_replacing_bodies_by_offsets_keeps_everything_else(mini_spec, mini_copy):
    for rel in ("calc.c", "util.c"):
        path = mini_copy / rel
        data = path.read_bytes()
        before = extract_file(mini_spec, mini_copy, rel)
        # 逐个函数：只替换它的函数体，其它函数的源码文本必须逐字节不变
        for i, t in enumerate(before):
            new = data[: t.body_start] + b"{ }" + data[t.body_end :]
            path.write_bytes(new)
            after = extract_file(mini_spec, mini_copy, rel)
            assert [a.name for a in after] == [b.name for b in before]
            assert after[i].n_statements == 0 and after[i].signature == t.signature
            for j, (b, a) in enumerate(zip(before, after)):
                if j != i:
                    assert new[a.start : a.end] == data[b.start : b.end], (t.name, b.name)
        # 全部函数体一起替换（从后往前）
        out = data
        for t in reversed(before):
            out = out[: t.body_start] + b"{ }" + out[t.body_end :]
        path.write_bytes(out)
        after = extract_file(mini_spec, mini_copy, rel)
        assert [a.name for a in after] == [b.name for b in before]
        assert all(a.n_statements == 0 for a in after)
        path.write_bytes(data)
        assert error_diagnostics(parse_file(mini_spec, mini_copy, rel)) == []


def test_bom_crlf_and_non_ascii_comments_keep_byte_offsets(workdir):
    text = "// 注释\r\n#define K 3\r\nint add(int a, int b)\r\n{\r\n    return a + b + K;\r\n}\r\n\r\n/* é */\r\nstatic int twice(int x) { x *= 2; return x; }\r\n"
    data = b"\xef\xbb\xbf" + text.encode("utf-8")        # UTF-8 BOM + CRLF
    (workdir / "w.c").write_bytes(data)
    spec = CorpusSpec(name="w", project_files=["w.c"], target_files=["w.c"], cflags=[])
    got = {t.name: t for t in extract_functions(spec, workdir)}
    add = got["add"]
    assert data[add.start : add.end] == b"int add(int a, int b)\r\n{\r\n    return a + b + K;\r\n}"
    assert add.signature == "int add(int a, int b)" and add.n_lines == 4 and add.deps.macros == ["#define K 3"]
    assert add.doc_comment is None                      # 注释在 #define 之前，不紧贴函数
    twice = got["twice"]
    assert data[twice.body_start : twice.body_end] == b"{ x *= 2; return x; }"
    assert twice.doc_comment == "/* é */"               # 带重音字母的注释是文档，不是分隔线


# --------------------------------------------------------------------------- locate


def rename_words(root: Path, mapping: dict[str, str]) -> None:
    """模拟符号混淆副本：对副本里所有 .c / .h 按整词改名。"""
    for f in root.iterdir():
        if f.suffix in (".c", ".h"):
            text = f.read_text(encoding="utf-8")
            for old, new in mapping.items():
                text = re.sub(rf"\b{old}\b", new, text)
            f.write_text(text, encoding="utf-8")


def test_locate_in_renamed_copy(mini_spec, mini_copy):
    mapping = {"sum_array": "f_17", "array_find": "f_3", "sq_plus_one": "f_9", "calc_apply": "f_21", "point_t": "t_4"}
    rename_words(mini_copy, mapping)

    t = locate(mini_spec, mini_copy, "f_17", "util.c")
    assert t is not None and t.name == "f_17" and t.target_id == "mini:util.c:f_17"
    assert t.signature == "int f_17(const int *a, size_t n)"
    data = (mini_copy / "util.c").read_bytes()
    assert data[t.start : t.start + len(b"int f_17(")] == b"int f_17("
    assert data[t.body_start : t.body_start + 1] == b"{" and data[t.end - 1 : t.end] == b"}"
    # 旧名字在副本里已不存在
    assert locate(mini_spec, mini_copy, "sum_array", "util.c") is None
    # 单语句 static inline 被 extract_functions 排除，但 locate 仍要能找回来
    sq = locate(mini_spec, mini_copy, "f_9", "calc.c")
    assert sq is not None and sq.n_statements == 1
    assert "f_9" not in [x.name for x in extract_functions(mini_spec, mini_copy)]
    # 依赖里的类型名也跟着副本走
    ca = locate(mini_spec, mini_copy, "f_21", "calc.c")
    assert any("op_kind_t" in x for x in ca.deps.types)
    pa = locate(mini_spec, mini_copy, "point_add", "calc.c")
    assert pa.signature == "t_4 point_add(t_4 a, t_4 b)"
    assert norm(pa.deps.types) == ["typedef struct point { int x; int y; } t_4;"]
    # 混淆前后同一个函数：语句数 / 行数不受改名影响
    assert locate(mini_spec, MINI, "sum_array", "util.c").n_statements == t.n_statements
    # 不是函数的名字 / 不存在的名字
    assert locate(mini_spec, mini_copy, "g_calls", "calc.c") is None
    assert locate(mini_spec, mini_copy, "no_such_function", "calc.c") is None
    with pytest.raises(FileNotFoundError):
        locate(mini_spec, mini_copy, "f_17", "no_such_file.c")


def test_header_implemented_library_targets(workdir):
    """jsmn 这类头文件实现的库：目标函数在 .h 里，也要能提取、定位、按偏移替换。"""
    root = workdir / "hdrlib"
    root.mkdir()
    (root / "hdr.h").write_text(
        """#ifndef HDR_H
#define HDR_H
#include <stddef.h>
#ifdef HDR_STATIC
#define HDR_API static
#else
#define HDR_API extern
#endif

typedef struct { int a; } hdr_t;

#ifndef HDR_HEADER
static int hdr_helper(int x)
{
    int y = x * 2;
    y += 1;
    return y;
}

static inline int hdr_one(int x) { return x; }

HDR_API int hdr_parse(hdr_t *h, int v)
{
    h->a = hdr_helper(v);
    h->a++;
    return h->a;
}
#endif /* HDR_HEADER */
#endif /* HDR_H */
""",
        encoding="utf-8",
    )
    spec = CorpusSpec(name="hdrlib", project_files=["hdr.h"], target_files=["hdr.h"], cflags=["-std=c99", "-I."])
    targets = {t.name: t for t in extract_functions(spec, root)}
    assert sorted(targets) == ["hdr_helper", "hdr_parse"]            # hdr_one 是单语句 static inline，排除
    p = targets["hdr_parse"]
    assert p.signature == "HDR_API int hdr_parse(hdr_t *h, int v)"
    assert p.callees == ["hdr_helper"]
    assert norm(p.deps.types) == ["typedef struct { int a; } hdr_t;"]
    assert p.deps.macros == ["#define HDR_API extern"]
    assert norm(p.deps.decls) == ["static int hdr_helper(int x);"]
    assert p.deps.includes == ["#include <stddef.h>"]
    # 按偏移替换 hdr_helper 的函数体，再重新定位
    data = (root / "hdr.h").read_bytes()
    h = targets["hdr_helper"]
    (root / "hdr.h").write_bytes(data[: h.body_start] + b"{ return x; }" + data[h.body_end :])
    again = locate(spec, root, "hdr_helper", "hdr.h")
    assert again is not None and again.n_statements == 1
    assert locate(spec, root, "hdr_parse", "hdr.h").start > again.end - 1


def test_works_on_the_mini_fixture_spec_loaded_from_file():
    spec = corpus.load_spec_file(MINI / "spec.json")
    assert len(extract_functions(spec, MINI)) == 40


def test_works_with_the_real_cjson_spec():
    cjson = require_real_corpus("cjson")
    real = corpus.load_spec("cjson")                          # 仓库里的真实 spec，以 cJSON 的克隆为根
    names = {t.name for t in extract_functions(real, cjson)}
    assert {"cJSON_Parse", "cJSON_Print", "cJSONUtils_GeneratePatches"} <= names


# --------------------------------------------------------------------------- 错误处理与辅助函数


def test_missing_target_file_raises(workdir):
    (workdir / "a.c").write_text("int f(void) { return 0; }\n")
    spec = CorpusSpec(name="x", project_files=["a.c"], target_files=["nope.c"], cflags=[])
    with pytest.raises(FileNotFoundError):
        extract_functions(spec, workdir)


def test_fatal_parse_error_raises(workdir):
    (workdir / "bad.c").write_text('#include "does_not_exist.h"\nint f(void) { return 0; }\n')
    spec = CorpusSpec(name="x", project_files=["bad.c"], target_files=["bad.c"], cflags=[])
    with pytest.raises(ExtractError):
        extract_functions(spec, workdir)


def test_non_fatal_errors_do_not_hide_other_functions(workdir):
    (workdir / "e.c").write_text("int good(int x) { return x + 1; }\nint bad(void) { return undeclared_symbol; }\n")
    spec = CorpusSpec(name="x", project_files=["e.c"], target_files=["e.c"], cflags=[])
    assert {t.name for t in extract_functions(spec, workdir)} == {"good", "bad"}


def test_builtin_include_args_make_stddef_resolvable(workdir):
    extra = builtin_include_args()
    assert isinstance(extra, tuple)
    if extra:
        assert extra[0] == "-isystem" and Path(extra[1]).is_dir()
    (workdir / "s.c").write_text("#include <stddef.h>\n#include <stdarg.h>\n#include <limits.h>\nsize_t n = INT_MAX;\n")
    spec = CorpusSpec(name="x", project_files=["s.c"], target_files=["s.c"], cflags=[])
    assert error_diagnostics(parse_file(spec, workdir, "s.c"), ci.Diagnostic.Fatal) == []


def test_clang_args_keeps_system_headers_in_include_flag(workdir):
    (workdir / "local_pre.h").write_text("#define PRE 1\n")
    args = clang_args(["-include", "stdio.h", "-include", "local_pre.h"], workdir)
    # 系统头名保持原样；src_root 下真有的文件才展开成绝对路径
    assert args[args.index("-include") + 1] == "stdio.h"
    assert str((workdir / "local_pre.h").resolve()) in args
    (workdir / "p.c").write_text("int f(void) { return fprintf(stdout, \"x\") + PRE; }\n")
    spec = CorpusSpec(name="x", project_files=["p.c"], target_files=["p.c"],
                                 cflags=["-include", "stdio.h", "-include", "local_pre.h"])
    assert error_diagnostics(parse_file(spec, workdir, "p.c")) == []


def test_clang_args_absolutize_and_drop_irrelevant_flags():
    root = Path("/some/root")
    args = clang_args(["-std=c89", "-I.", "-Iinc", "-I", "other", "-isystem", "sys", "-DX=1", "-lm", "-Wl,--foo",
                       "-Werror", "-o", "out", "-O2"], root)
    assert "-std=c89" in args and "-DX=1" in args and "-O2" in args
    assert f"-I{root.resolve()}" in args and f"-I{(root / 'inc').resolve()}" in args
    assert "-I" in args and str((root / "other").resolve()) in args
    assert "-isystem" in args and str((root / "sys").resolve()) in args
    for dropped in ("-lm", "-Wl,--foo", "-Werror", "-o", "out"):
        assert dropped not in args
    assert args[:2] == ["-x", "c"]


def test_strip_comments_respects_literals():
    src = 'int a = 1; /* x */ char *s = "// not a comment /* nor this */"; // tail\nchar c = \'"\'; int b;'
    out = strip_c_comments(src)
    assert '"// not a comment /* nor this */"' in out
    assert "tail" not in out and "x */" not in out
    assert out.count("\n") == 1 and "int b;" in out
    assert squash_whitespace("  a \n\t b   c ") == "a b c"


# --------------------------------------------------------------------------- 沙箱：stub-kill 验证


def sandbox_run(workdir: Path, script: str, timeout: int = 600):
    """在共用的 DockerSandbox 里跑脚本（无网络、丢全部 capability、限 CPU / 内存 / 进程数，超时能强杀容器）。"""
    return DockerSandbox(image=SANDBOX_IMAGE, timeout_s=timeout).run(workdir, script)


def return_kinds(spec, root: Path, rel: str) -> dict[str, tuple[str, str]]:
    """函数名 -> (返回值类别, 返回类型拼写)，类别：void / pointer / record / arith。"""
    tu = parse_file(spec, root, rel)
    path = (root / rel).resolve()
    out = {}
    for c in tu.cursor.get_children():
        if c.kind == ci.CursorKind.FUNCTION_DECL and c.is_definition() and c.location.file \
                and Path(c.location.file.name).resolve() == path:
            canon = c.result_type.get_canonical()
            kind = {ci.TypeKind.VOID: "void", ci.TypeKind.POINTER: "pointer", ci.TypeKind.RECORD: "record"}.get(canon.kind, "arith")
            out[c.spelling] = (kind, c.result_type.spelling)
    return out


def stub_body(kind: str, spelling: str) -> bytes:
    """按设计文档 4.5 的 stub 规则：void 空体；指针 / 算术返回 0；结构体返回零初始化变量。"""
    if kind == "void":
        return b"{ }"
    if kind == "record":
        return f"{{ {spelling} _z; __builtin_memset(&_z, 0, sizeof _z); return _z; }}".encode()
    return b"{ return 0; }"


def standalone_source(t: FunctionTarget, data: bytes, drop: tuple[str, ...] = ()) -> str:
    """includes + macros + types + globals + decls + 目标函数原定义：依赖闭包完整时它能独立通过语法检查。"""
    parts: list[str] = []
    for name in ("includes", "macros", "types", "globals", "decls"):
        if name not in drop:
            parts += getattr(t.deps, name)
    parts.append(data[t.start : t.end].decode("utf-8"))
    return "\n".join(parts) + "\n"


def syntax_check_in_sandbox(files_dir: Path, std: str) -> dict[str, bool]:
    """沙箱里对目录下每个 .c 只做 gcc -fsyntax-only（不链接、不运行），返回 文件名 -> 是否通过。"""
    script = f'for f in *.c; do gcc -std={std} -w -fsyntax-only -x c "$f" >/dev/null 2>&1; echo "$f $?"; done'
    proc = sandbox_run(files_dir, script)
    assert proc.rc == 0, proc.stderr
    return {line.split()[0]: line.split()[1] == "0" for line in proc.stdout.splitlines() if line.strip()}


@needs_docker
def test_mini_tests_pass_and_every_function_is_stub_killable(mini_spec, workdir):
    """基线 make test 必须通过；把任何一个函数（含单语句 static inline）换成零值桩，make test 必须失败。"""
    variants = workdir / "variants"
    base = copy_mini(variants / "baseline")
    expected_failing: list[str] = []
    for rel in ("calc.c", "util.c"):
        kinds = return_kinds(mini_spec, MINI, rel)
        data = (MINI / rel).read_bytes()
        for t in extract_file(mini_spec, MINI, rel):
            vdir = copy_mini(variants / f"v_{t.name}")
            kind, spelling = kinds[t.name]
            new = data[: t.body_start] + stub_body(kind, spelling) + data[t.body_end :]
            (vdir / rel).write_bytes(new)
            expected_failing.append(f"v_{t.name}")
    assert base.is_dir() and len(expected_failing) == 41           # 40 个目标 + sq_plus_one

    # 每个目录：先单独 make mini_test 确认桩本身能编译（编译失败不算「被测试抓住」），再跑 make test 看退出码
    script = (
        'for d in */; do d=${d%/}; ( cd "$d" && make mini_test >/dev/null 2>&1; b=$?; '
        'timeout -s KILL 60 make test >/dev/null 2>&1; t=$?; echo "$d $b $t" ); done'
    )
    proc = sandbox_run(variants, script)
    assert proc.rc == 0, proc.stderr
    rc = {}
    for line in proc.stdout.splitlines():
        if line.strip():
            name, build_rc, test_rc = line.split()
            rc[name] = (int(build_rc), int(test_rc))
    assert rc["baseline"] == (0, 0), "未改动的 mini 项目应该能编译且 make test 通过"
    not_compiled = [v for v in expected_failing if rc[v][0] != 0]
    assert not_compiled == [], f"桩没能编译，验证无效: {not_compiled}"
    survivors = [v for v in expected_failing if rc[v][1] == 0]
    assert survivors == [], f"这些函数换成零值桩后测试仍然通过，说明测试没检验到它们: {survivors}"


@needs_docker
def test_mini_dependency_closure_is_sufficient_to_compile_each_target(mini_spec, workdir):
    """依赖闭包完整性的硬检查：每个目标函数只靠自己的 deps 就能独立通过语法检查。"""
    good, missing_types = workdir / "good", workdir / "missing_types"
    good.mkdir()
    missing_types.mkdir()
    expect_fail: list[str] = []
    n = 0
    for rel in ("calc.c", "util.c"):
        data = (MINI / rel).read_bytes()
        for t in extract_file(mini_spec, MINI, rel):
            (good / f"{t.name}.c").write_text(standalone_source(t, data), encoding="utf-8")
            n += 1
            if t.deps.types:     # 反例对照：去掉类型定义后必须编译失败，证明检查本身不是摆设
                (missing_types / f"{t.name}.c").write_text(standalone_source(t, data, drop=("types",)), encoding="utf-8")
                expect_fail.append(f"{t.name}.c")
    ok = syntax_check_in_sandbox(good, "c99")
    assert len(ok) == n and [f for f, passed in ok.items() if not passed] == []
    bad = syntax_check_in_sandbox(missing_types, "c99")
    assert expect_fail and all(not bad[f] for f in expect_fail)


@needs_docker
def test_mini_builds_and_passes_in_shared_sandbox(mini_copy):
    """用项目统一的 DockerSandbox 跑一遍基线（与上面测试里的等价 docker 参数互相印证）。"""
    res = DockerSandbox(image=SANDBOX_IMAGE).run(mini_copy, "make test")
    assert res.rc == 0 and not res.timed_out, res.stdout + res.stderr
    assert re.search(r"\d+ checks, 0 failures", res.stdout)
    assert (mini_copy / "mini_test").is_file()      # 产物落在 tmp/ 下的副本里，夹具目录保持干净
    assert not (MINI / "mini_test").exists()


# --------------------------------------------------------------------------- cJSON 只读提取


def rough_function_count(text: str) -> tuple[int, int]:
    """grep 级别的粗估：返回 (函数定义头个数, 其中落在 #if defined(_MSC_VER) 分支里的个数)。

    定义头 = 行首 '{' 且上一行以 ')' 结尾（Allman 风格），或行首标识符开头、以 ') {' 结尾的单行头。
    """
    lines = text.splitlines()
    total = inactive = 0
    in_msc = False
    for i, ln in enumerate(lines):
        if ln.startswith("#if defined(_MSC_VER)"):
            in_msc = True
        elif in_msc and ln.startswith(("#else", "#endif")):
            in_msc = False
        allman = ln.startswith("{") and i > 0 and lines[i - 1].rstrip().endswith(")")
        same_line = re.match(r"^[A-Za-z_].*\) *\{ *$", ln) is not None
        if allman or same_line:
            total += 1
            inactive += in_msc
    return total, inactive


def test_cjson_readonly_extraction():
    CJSON = require_real_corpus("cjson")
    watched = [CJSON / "cJSON.c", CJSON / "cJSON.h"]
    digest = lambda: [hashlib.sha1(p.read_bytes()).hexdigest() for p in watched]  # noqa: E731
    before = digest()

    spec = CorpusSpec(
        name="cjson", project_files=["cJSON.c", "cJSON.h"], target_files=["cJSON.c"], cflags=["-std=c89", "-I."]
    )
    assert error_diagnostics(parse_file(spec, CJSON, "cJSON.c")) == []
    targets = extract_functions(spec, CJSON)
    names = {t.name for t in targets}

    text = (CJSON / "cJSON.c").read_text(encoding="utf-8")
    est, inactive = rough_function_count(text)
    print(f"\n[cJSON.c] libclang 函数数={len(targets)}  grep 粗估={est}（其中 _MSC_VER 分支里 {inactive} 个）")
    # 粗估 - 未启用分支里的定义 = libclang 的数；差异只有这一个来源
    assert len(targets) == est - inactive
    assert {"internal_malloc", "internal_free", "internal_realloc"}.isdisjoint(names)
    assert {"cJSON_Parse", "cJSON_Version", "print_number", "parse_value", "minify_string",
            "cJSON_CreateArrayReference"} <= names          # 后两个是「花括号在签名同一行」的写法，Allman 式 grep 会漏

    data = (CJSON / "cJSON.c").read_bytes()
    for t in targets:
        assert data[t.body_start : t.body_start + 1] == b"{" and data[t.end - 1 : t.end] == b"}"
        assert re.search(rf"\b{t.name}\b", t.signature)
    by = {t.name: t for t in targets}
    assert by["cJSON_Parse"].callees == ["cJSON_ParseWithOpts"]
    assert by["cJSON_Version"].signature == "CJSON_PUBLIC(const char*) cJSON_Version(void)"
    assert "#define CJSON_PUBLIC(type) type" in by["cJSON_Version"].deps.macros
    assert any(x.startswith("typedef struct cJSON") for x in by["get_array_item"].deps.types)
    assert "extern internal_hooks global_hooks;" in by["cJSON_InitHooks"].deps.globals
    assert before == digest(), "只读提取不应改动语料文件"


@needs_docker
def test_cjson_dependency_closure_compiles_standalone(workdir):
    CJSON = require_real_corpus("cjson")
    spec = CorpusSpec(
        name="cjson", project_files=["cJSON.c", "cJSON.h"], target_files=["cJSON.c"], cflags=["-std=c89", "-I."]
    )
    data = (CJSON / "cJSON.c").read_bytes()
    targets = extract_functions(spec, CJSON)
    for t in targets:
        (workdir / f"{t.name}.c").write_text(standalone_source(t, data), encoding="utf-8")
    ok = syntax_check_in_sandbox(workdir, "c89")
    assert len(ok) == len(targets)
    assert [f for f, passed in ok.items() if not passed] == []
