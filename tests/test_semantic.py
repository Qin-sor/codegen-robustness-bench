"""bench.obfuscate.semantic（L3 语义混淆）的测试。

分四块：
1. 单类改写的最小用例（不需要 docker，用 ``transform_source`` 直接改一小段 C 源码）：五类改写各自的正例，
   以及指针 / 调用 / 宏 / 非原子操作数等必须放过的反例；嵌套改写、行数与字节保持、确定性、回滚、跳过。
2. 目录级入口 ``transform_semantic``：vendored / files 过滤、统计字典的形状和可序列化、
   目标函数在 L3 副本里仍能被 ``extract.locate`` 重新定位。
3. 沙箱对拍（docker, gcc:13）：手写的覆盖各种类型 / 优先级 / 副作用的 C 程序、随机表达式程序，
   改写前后在 -O0 / -O2 下输出必须逐字节相同。
4. 沙箱测试：mini 夹具与 9 个真实语料（cjson、sds、mpc 为必选，其余同样跑）改写后项目自带测试全部通过；
   同时检查 L2 -> L3、L1 -> L2 -> L3 叠加（symbols / structure 模块存在时）。

改写和构建运行都只在项目 tmp/ 下的临时目录里做；语料代码的编译和运行只走 docker 沙箱。
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

import clang.cindex as ci
import pytest

from bench import clangcfg, corpus, extract
from bench.obfuscate import semantic
from bench.obfuscate import structure, symbols
from bench.obfuscate.semantic import KINDS, transform_bytes, transform_semantic, transform_source
from bench.sandbox import DockerSandbox
from tests.helpers import MINI, needs_docker, require_real_corpus

REQUIRED_CORPORA = ["cjson", "sds", "mpc"]
OTHER_CORPORA = ["parson", "tinyexpr", "jsmn", "inih", "tiny-regex-c", "vec"]
SEED = 2026
NOT_OPAQUE = tuple(k for k in KINDS if k != "opaque")


# --------------------------------------------------------------------------- 夹具与小工具


@pytest.fixture(scope="module")
def sandbox() -> DockerSandbox:
    return DockerSandbox(timeout_s=180, cpus=2.0)


def tx(src: str, kinds=NOT_OPAQUE, ratio: float = 1.0, seed: int = 1, opaque_ratio: float = 1.0,
       std: str = "c99") -> tuple[str, dict]:
    """改写一段源码，默认只做除 opaque 以外的四类、全部可改位置都改（ratio=1）。"""
    return transform_source(src, [f"-std={std}"], seed, ratio=ratio, opaque_ratio=opaque_ratio, kinds=kinds)


def parse_errors(src: str, std: str = "c99") -> list[str]:
    """用 libclang 解析一段源码，返回 error 级诊断。"""
    path = "__semantic_probe__.c"
    args = ["-x", "c", "-w", "-ferror-limit=0", *clangcfg.builtin_include_args(), f"-std={std}"]
    tu = clangcfg.index().parse(path, args=args, unsaved_files=[(path, src.encode("utf-8"))], options=0)
    return [d.spelling for d in tu.diagnostics if d.severity >= ci.Diagnostic.Error]


def body(text: str, name: str) -> str:
    """取出名为 name 的函数定义到下一个顶层 '}' 为止的文本，便于对单个函数断言。"""
    i = text.index(name + "(")
    j = text.index("\n}", i)
    return text[i:j + 2]


def make_spec(root: Path, files: list[str], vendored: list[str] | None = None, std: str = "c99") -> corpus.CorpusSpec:
    return corpus.CorpusSpec(name="sem", project_files=files, target_files=files, cflags=[f"-std={std}"],
                             build_cmd="true", vendored=vendored or [])


def write_tree(root: Path, files: dict[str, str]) -> None:
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(text.encode("latin-1"))


def corpus_source(name: str) -> Path:
    """真实语料的源码目录：corpus/src/<name>，或 CORPUS_MIRROR 指向的本地克隆；都没有就跳过（BENCH_REQUIRE_CORPUS=1 时失败）。"""
    return require_real_corpus(name)


def run_spec(sandbox: DockerSandbox, spec, work: Path):
    res = sandbox.run(work, spec.run_command(), timeout_s=max(spec.timeout_s, 60))
    return spec.passes(res.rc, res.stdout) and not res.timed_out, res


def check_passes(sandbox: DockerSandbox, spec, work: Path, what: str) -> None:
    ok, res = run_spec(sandbox, spec, work)
    assert ok, (f"{what} 在沙箱里没有通过 (rc={res.rc}, timed_out={res.timed_out})\n--- stdout 末尾\n"
                f"{res.stdout[-1200:]}\n--- stderr 末尾\n{res.stderr[-2000:]}")


# --------------------------------------------------------------------------- 1. 词法器与渲染器的单元测试


def test_lexer_matches_c_token_boundaries():
    src = "a->b <<= 'x' \"s//\" /* c */ 1.5e+3 0x1p-2 // end\n+ x++ ... '\\'' L\"w\" a\\\nb"
    t = semantic._Toks(src)
    spelled = [t.spell(i) for i in range(len(t.starts))]
    assert spelled == ["a", "->", "b", "<<=", "'x'", '"s//"', "1.5e+3", "0x1p-2", "+", "x", "++", "...",
                       "'\\''", 'L"w"', "a", "b"]
    # 字符串里的 // 和 /* 不会被当成注释；注释里的记号一概不出现
    assert t.kinds[spelled.index('"s//"')] == "str"
    assert t.aligned(t.starts[0], t.ends[2]) and not t.aligned(t.starts[0] + 2, t.ends[2])


def test_lexer_survives_unterminated_literals():
    t = semantic._Toks("int a; /* 没有结尾的注释\nchar c = 'x;\nint b;")
    # 无法成对的引号退化成单字符记号，不会吞掉后面的行
    assert "b" in [t.spell(i) for i in range(len(t.starts))]


# --------------------------------------------------------------------------- 2. relation_flip

REL_SRC = """
struct S { int x; int y; };
int rel(int a, int b, struct S s, struct S *p)
{
    int r = 0;
    r += a < b;
    r += a <= b;
    r += a > b;
    r += a >= b;
    r += s.x <= 3;
    r += 3 < p->y;
    r += p->x >= s.y;
    r += s.x /* c */ < p->y;
    r += a < b < r;
    return r;
}
"""

REL_REFUSED = """
#define LIMIT 10
#define LT(a, b) ((a) < (b))
int f(int, int);
int rel(int a, int b, int *p, int *q, double d, int arr[], float fl)
{
    int r = 0;
    r += p < q;
    r += arr < p;
    r += arr[0] < a;
    r += a + 1 < b;
    r += f(a, b) < a;
    r += a++ < b;
    r += (a = b) < 3;
    r += *p < 3;
    r += a < -1;
    r += a < LIMIT;
    r += LT(a, b);
    r += a < a;
    r += (a) < b;
    r += (int)d < a;
    r += sizeof(a) < b;
    return r;
}
"""


def test_relation_flip_reverses_operands_and_operator():
    out, st = tx(REL_SRC, ("relation_flip",))
    pairs = [("r += a < b;", "r += b > a;"), ("r += a <= b;", "r += b >= a;"),
             ("r += a > b;", "r += b < a;"), ("r += a >= b;", "r += b <= a;"),
             ("r += s.x <= 3;", "r += 3 >= s.x;"), ("r += 3 < p->y;", "r += p->y > 3;"),
             ("r += p->x >= s.y;", "r += s.y <= p->x;"),
             ("r += s.x /* c */ < p->y;", "r += p->y /* c */ > s.x;")]
    for old, new in pairs:
        assert old not in out, old
        assert new in out, new
    # a < b < r 是 (a < b) < r：只有内层的 a < b 满足条件
    assert "r += b > a < r;" in out
    assert st["relation_flip"] == len(pairs) + 1 and st["total"] == st["relation_flip"]
    assert parse_errors(out) == []


def test_relation_flip_leaves_unsafe_operands_alone():
    out, st = tx(REL_REFUSED, ("relation_flip",))
    assert out == REL_REFUSED
    assert st["relation_flip"] == 0
    reasons = set(st["skipped"]["relation_flip"])
    # 指针 / 数组、非原子操作数（调用、自增、赋值、解引用、下标、括号、强转）、宏、两侧相同，各有一个原因
    assert {semantic.R_NOT_ARITH, semantic.R_NOT_ATOMIC, semantic.R_MACRO, semantic.R_SAME} <= reasons


# --------------------------------------------------------------------------- 3. commute

COM_SRC = """
struct S { int x; int y; };
double com(int a, int b, double d, double e, struct S s, int *p, unsigned u)
{
    double r = 0;
    r += a + b;
    r += a * 2;
    r += 3 + d;
    r += d * e;
    r += s.x * s.y;
    r += a - b;
    r += a / 2;
    r += p[0] + a;
    r += (double)(p + 1 == p);
    r += a + b + u;
    r += a + a;
    r += a * b * u;
    return r;
}
"""


def test_commute_swaps_pure_arithmetic_operands():
    out, st = tx(COM_SRC, ("commute",))
    for old, new in [("r += a + b;", "r += b + a;"), ("r += a * 2;", "r += 2 * a;"),
                     ("r += 3 + d;", "r += d + 3;"), ("r += d * e;", "r += e * d;"),
                     ("r += s.x * s.y;", "r += s.y * s.x;")]:
        assert old not in out and new in out, (old, new)
    # a + b + u 是 (a + b) + u：只换内层
    assert "r += b + a + u;" in out and "r += b * a * u;" in out
    # 减、除、下标操作数、指针运算、两侧相同：原样
    for keep in ("r += a - b;", "r += a / 2;", "r += p[0] + a;", "(double)(p + 1 == p)", "r += a + a;"):
        assert keep in out, keep
    assert st["commute"] == 7
    assert semantic.R_NOT_ARITH in st["skipped"]["commute"]
    assert parse_errors(out) == []


# --------------------------------------------------------------------------- 4. eq_zero

EQZ_SRC = r"""
enum E { E0, E1 };
typedef unsigned char byte;
int g(int);
int eqz(int a, unsigned u, long long ll, char c, _Bool b, enum E e, byte y, const char *s)
{
    int r = 0;
    r += !a;
    r += !u;
    r += !ll;
    r += !c;
    r += !b;
    r += !e;
    r += !y;
    r += a == 0;
    r += 0 == a;
    r += a + 1 == 0;
    r += (a & 1) == 0;
    r += g(a) == 0;
    r += s[0] == 0;
    r += *s == 0;
    r += u == 0;
    r += (a < 3) == 0;
    r += !(a < 3);
    r += !!a;
    return r;
}
"""

EQZ_REFUSED = r"""
#define ZERO 0
int eqz2(int a, int *p, double d, const char *s, int arr[3], int (*fp)(int))
{
    int r = 0;
    r += !p;
    r += p == 0;
    r += !d;
    r += d == 0;
    r += !s;
    r += !arr;
    r += !fp;
    r += a != 0;
    r += a == 0u;
    r += a == 00;
    r += a == 0L;
    r += a == (0);
    r += a == '\0';
    r += a == ZERO;
    return r;
}
"""


def test_eq_zero_swaps_not_and_equals_zero_for_integer_operands():
    out, st = tx(EQZ_SRC, ("eq_zero",))
    expected = {
        "r += !a;": "r += (a == 0);", "r += !u;": "r += (u == 0);", "r += !ll;": "r += (ll == 0);",
        "r += !c;": "r += (c == 0);", "r += !b;": "r += (b == 0);", "r += !e;": "r += (e == 0);",
        "r += !y;": "r += (y == 0);",
        "r += a == 0;": "r += !a;", "r += 0 == a;": "r += !a;", "r += a + 1 == 0;": "r += !(a + 1);",
        "r += (a & 1) == 0;": "r += !(a & 1);", "r += g(a) == 0;": "r += !g(a);",
        "r += s[0] == 0;": "r += !s[0];", "r += *s == 0;": "r += !*s;", "r += u == 0;": "r += !u;",
        "r += (a < 3) == 0;": "r += !(a < 3);", "r += !(a < 3);": "r += ((a < 3) == 0);",
    }
    for old, new in expected.items():
        assert new in out, (old, new)
    # !!a：外层 !(!a) 变 ((!a) == 0)，内层 !a 变 (a == 0)，两处嵌套生效
    assert "r += ((a == 0) == 0);" in out
    assert out.count("r += !a;") == 2 and out.count("r += (a == 0);") == 1
    assert st["eq_zero"] == len(expected) + 2
    assert parse_errors(out) == []


def test_eq_zero_refuses_pointers_floats_and_inexact_zero_literals():
    out, st = tx(EQZ_REFUSED, ("eq_zero",))
    assert out == EQZ_REFUSED
    assert st["eq_zero"] == 0
    assert semantic.R_NOT_INT in st["skipped"]["eq_zero"]


# --------------------------------------------------------------------------- 5. de_morgan

DM_SRC = """
int f(int);
int dm(int a, int b, int c, int *p)
{
    int r = 0;
    r += !(a && b);
    r += !(a || b);
    r += !(a == 1 && b > 2);
    r += !(f(a) || *p);
    r += !(p && p[0] > 3);
    r += !(a && b && c);
    r += !(a ? b : c);
    r += !(a & b);
    r += !((a && b));
    r += !a && !b;
    r += !(a);
    r += !(a, b);
    return r;
}
"""

DM_ML = """int ml(int a, int b)
{
    int r = !(a /* x */ &&   // y
              b);
    int s = !(
        a || b);
    return r + s;
}
"""


def test_de_morgan_rewrites_explicit_negated_conjunctions_and_disjunctions():
    out, st = tx(DM_SRC, ("de_morgan",))
    expected = {
        "r += !(a && b);": "r += (!a || !b);",
        "r += !(a || b);": "r += (!a && !b);",
        "r += !(a == 1 && b > 2);": "r += (!(a == 1) || !(b > 2));",
        "r += !(f(a) || *p);": "r += (!f(a) && !*p);",
        "r += !(p && p[0] > 3);": "r += (!p || !(p[0] > 3));",
        "r += !(a && b && c);": "r += (!(a && b) || !c);",
    }
    for old, new in expected.items():
        assert old not in out and new in out, (old, new)
    for keep in ("r += !(a ? b : c);", "r += !(a & b);", "r += !((a && b));", "r += !a && !b;",
                 "r += !(a);", "r += !(a, b);"):
        assert keep in out, keep
    assert st["de_morgan"] == len(expected)
    assert parse_errors(out) == []


def test_de_morgan_composes_with_inner_rewrites():
    out, st = tx("int f(int a, int b, int c, int d) { return !(a < b && c > d); }\n", ("de_morgan", "relation_flip"))
    assert "return (!(b > a) || !(d < c));" in out
    assert (st["de_morgan"], st["relation_flip"]) == (1, 2)


def test_de_morgan_keeps_comments_and_line_count():
    out, st = tx(DM_ML, ("de_morgan",))
    assert st["de_morgan"] == 2
    assert "(!a /* x */ ||   // y\n" + " " * 14 + "!b)" in out
    assert out.count("\n") == DM_ML.count("\n")
    assert parse_errors(out) == []


def test_not_with_both_rewrites_available_prefers_de_morgan():
    out, st = tx("int f(int a, int b) { return !(a && b); }\n", ("de_morgan", "eq_zero"))
    assert "return (!a || !b);" in out
    assert (st["de_morgan"], st["eq_zero"]) == (1, 0)
    # 只开 eq_zero 时同一个节点按 eq_zero 改
    out, st = tx("int f(int a, int b) { return !(a && b); }\n", ("eq_zero",))
    assert "return ((a && b) == 0);" in out and st["eq_zero"] == 1


# --------------------------------------------------------------------------- 6. opaque

OPQ_SRC = """
int f(int x) { return x; }
void g(void)
{
    int y = 1;
    (void)y;
}
static int k(a, b) int a; int b; { return a + b; }
#define BODY { return 1; }
int m(void) BODY
"""

OPQ_RE = re.compile(r"\{ \{ volatile int (v_\d+) = 0; \(void\)\1; \}")


def test_opaque_statement_goes_right_after_the_function_body_brace():
    out, st = tx(OPQ_SRC, ("opaque",))
    names = [m.group(1) for m in OPQ_RE.finditer(out)]
    assert len(names) == 3 and len(set(names)) == 3
    assert st["opaque"] == 3
    f_open = "int f(int x) { { volatile int %s = 0; (void)%s; } return x; }" % (names[0], names[0])
    assert f_open in out
    assert "void g(void)\n{ { volatile int %s = 0; (void)%s; }\n    int y = 1;" % (names[1], names[1]) in out
    assert "int a; int b; { { volatile int %s = 0; (void)%s; } return a + b; }" % (names[2], names[2]) in out
    # 函数体花括号是宏生成的：不碰，并记一次跳过
    assert "int m(void) BODY" in out
    assert st["skipped"]["opaque"] == {semantic.R_BRACE: 1}
    assert parse_errors(out) == []


def test_opaque_ratio_zero_inserts_nothing():
    out, st = tx(OPQ_SRC, ("opaque",), opaque_ratio=0.0)
    assert out == OPQ_SRC and st["opaque"] == 0 and st["candidates"]["opaque"] == 3


def test_opaque_names_avoid_every_spelling_already_in_the_project():
    taken = {"v_1", "v_2", "keep"}
    names = semantic._opaque_names(1, "a.c", 3, taken)
    assert len(names) == len(set(names)) == 3 and all(re.fullmatch(r"v_\d+", n) for n in names)
    assert not ({"v_1", "v_2"} & set(names)) and taken == {"v_1", "v_2", "keep", *names}
    # 同样的输入得到同样的名字（由 seed、文件、函数个数决定）
    assert names == semantic._opaque_names(1, "a.c", 3, {"v_1", "v_2", "keep"})
    # 第二次改写时，第一次留下的名字也在已有拼写里，新名字不会重复
    once, _ = tx(OPQ_SRC, ("opaque",))
    twice, _ = tx(once, ("opaque",))
    found = re.findall(r"volatile int (v_\d+) = 0; \(void\)\1;", twice)
    assert len(found) == 6 and len(set(found)) == 6


def test_opaque_names_follow_the_l1_naming_scheme_not_a_fixed_pattern():
    """设计文档 §4.6 L3 第 5 条：opaque 语句的变量名也经过改名表，长得和 L1 改过的变量名一样（v_<n>），
    而不是一眼能认出来的固定模式（以前是 _k + 随机数字）。"""
    out, _ = tx(OPQ_SRC, ("opaque",))
    assert "_k" not in re.sub(r"v_\d+", "", " ".join(OPQ_RE.findall(out)))
    assert len(OPQ_RE.findall(out)) == 3
    assert not re.search(r"volatile int _k", out)
    # 与 L1 的编号方案一致：同一张改名表，不与已有的 v_<n> 重名
    l1_src = "int f(int v_1, int v_2) { return v_1 + v_2; }\n"
    out2, _ = tx(l1_src, ("opaque",))
    (name,) = OPQ_RE.findall(out2)
    assert name not in ("v_1", "v_2")


# --------------------------------------------------------------------------- 7. 必须原样放过的上下文

MACRO_SRC = r"""
#include <stdio.h>
#define CHECK(c) do { if (!(c)) puts(#c); } while (0)
#define LIMIT 10
#define MAXV(a, b) ((a) > (b) ? (a) : (b))
#define NOT(x) (!(x))
int mm(int a, int b)
{
    CHECK(a < b);
    CHECK(!(a && b));
    CHECK(a + b == 0);
    if (a < LIMIT) return 1;
    if (!NOT(a)) return 2;
    if (MAXV(a, b) == 0) return 4;
    if (a > b) return 3;
    return MAXV(a, b) + 0;
}
"""

DIRECTIVE_SRC = """
int dd(int a, int b)
{
    if (a <
#ifdef FOO
        b + 1
#else
        b
#endif
       ) return 1;
    return 0;
}
"""


def test_nodes_from_macros_are_never_touched():
    out, st = tx(MACRO_SRC)
    # 宏参数里的比较 / 宏体里的运算 / 含宏调用的表达式全部原样，只有普通代码里的 a > b 被改
    assert out == MACRO_SRC.replace("if (a > b) return 3;", "if (b < a) return 3;")
    assert st["relation_flip"] == 1 and st["total"] == 1
    assert semantic.R_MACRO in st["skipped"]["relation_flip"]


def test_expressions_split_by_preprocessor_directives_are_left_alone():
    out, st = tx(DIRECTIVE_SRC)
    assert out == DIRECTIVE_SRC and st["total"] == 0


def test_stringizing_and_builtin_line_macros_keep_their_text():
    src = r"""
#define SHOW(e) printf("%s => %d\n", #e, (int)(e))
int printf(const char *, ...);
void s(int a, int b)
{
    SHOW(a < b);
    SHOW(!(a && b));
    if (a == __LINE__ || b < __LINE__) a = 0;
}
"""
    out, st = tx(src)
    assert out == src and st["total"] == 0


def test_non_ascii_bytes_and_line_endings_survive():
    src = ("/* 中文注释 */\r\nint f(int a, int b)\r\n{\r\n    const char *s = \"数据\"; // 注释\r\n"
           "    return a < b && s[0] == 0;\r\n}\r\n")
    data = src.encode("utf-8")
    new, st = transform_bytes(data, ["-std=c99"], 1, ratio=1.0, kinds=NOT_OPAQUE)
    for piece in ("中文注释", "数据", "// 注释"):
        assert piece.encode("utf-8") in new
    assert new.count(b"\r\n") == data.count(b"\r\n") and new.count(b"\n") == data.count(b"\n")
    new.decode("utf-8")
    assert st["total"] == 2 and b"return b > a && !s[0];" in new


def test_transform_source_handles_non_ascii_text():
    src = "/* 比较 */\nint f(int a, int b) { const char *s = \"数据\"; return a < b && s[0] == 0; }\n"
    out, st = tx(src)
    assert "/* 比较 */" in out and '"数据"' in out and "return b > a && !s[0];" in out and st["total"] == 2


@pytest.mark.parametrize("src", [REL_SRC, COM_SRC, EQZ_SRC, DM_SRC, DM_ML, OPQ_SRC, MACRO_SRC])
def test_line_count_is_preserved_for_every_kind_combination(src):
    for kinds in (KINDS, NOT_OPAQUE, ("de_morgan",), ("eq_zero",)):
        out, _ = tx(src, kinds)
        assert out.count("\n") == src.count("\n"), kinds
        assert parse_errors(out) == [], kinds


# --------------------------------------------------------------------------- 8. 选择、确定性、回滚、跳过


def big_source(n: int = 40) -> str:
    parts = ["int ext(int);"]
    for i in range(n):
        parts.append(
            f"int fn{i}(int a, int b, int c)\n{{\n    int r = 0;\n    if (a < b) r += a + c;\n"
            f"    if (!(b > c || a == 0)) r += c * 2;\n    if (c <= {i}) r++;\n    return r + (b == 0);\n}}\n")
    return "\n".join(parts)


def test_same_seed_same_output_different_seed_different_output():
    src = big_source()
    a1, s1 = tx(src, KINDS, ratio=0.5, seed=7)
    a2, s2 = tx(src, KINDS, ratio=0.5, seed=7)
    b, _ = tx(src, KINDS, ratio=0.5, seed=8)
    assert a1 == a2 and s1 == s2
    assert a1 != b
    assert parse_errors(a1) == [] and parse_errors(b) == []


def test_ratio_controls_how_many_positions_are_rewritten():
    src = big_source()
    _, none = tx(src, NOT_OPAQUE, ratio=0.0)
    _, half = tx(src, NOT_OPAQUE, ratio=0.5)
    _, full = tx(src, NOT_OPAQUE, ratio=1.0)
    assert none["total"] == 0
    assert 0.3 < half["total"] / full["total"] < 0.7
    # 可用位置数与 ratio 无关
    assert none["candidates"] == half["candidates"] == full["candidates"]


def test_selection_depends_on_structure_not_on_identifier_names():
    src = big_source()
    renamed = re.sub(r"\b([abcr])\b", lambda m: "var_" + m.group(1) * 3, src)
    assert renamed != src
    _, s1 = tx(src, KINDS, ratio=0.5, seed=3)
    _, s2 = tx(renamed, KINDS, ratio=0.5, seed=3)
    assert {k: s1[k] for k in KINDS} == {k: s2[k] for k in KINDS}


def test_rewrite_failure_rolls_back_only_the_offending_edit(monkeypatch):
    src = "int f(int a, int b, int c, int d)\n{\n    int r = 0;\n    r += a < b;\n    r += c < d;\n    return r;\n}\n"
    real = semantic.Parser.valid

    def fake(self, unit, data):
        return (False, ["boom"]) if b"d > c" in data else real(self, unit, data)

    monkeypatch.setattr(semantic.Parser, "valid", fake)
    out, st = tx(src, ("relation_flip",))
    assert "r += b > a;" in out and "r += c < d;" in out and "d > c" not in out
    assert st["relation_flip"] == 1 and st["rolled_back"]["relation_flip"] == 1
    assert parse_errors(out) == []


def test_every_edit_failing_leaves_the_file_unchanged(monkeypatch):
    src = "int f(int a, int b, int c, int d)\n{\n    int r = 0;\n    r += a < b;\n    r += !c;\n    return r;\n}\n"
    monkeypatch.setattr(semantic.Parser, "valid", lambda self, unit, data: (False, ["boom"]))
    out, st = tx(src, KINDS)
    assert out == src
    assert st["total"] == 0 and sum(st["rolled_back"].values()) == 3     # relation_flip + eq_zero + opaque


def test_file_that_already_fails_to_parse_is_skipped():
    src = "int f(int a)\n{\n    return a < undeclared_name;\n}\n"
    out, st = tx(src)
    assert out == src and st["total"] == 0
    assert list(st["files_skipped"]) == ["snippet.c"] and "error" in st["files_skipped"]["snippet.c"]


def test_unknown_kind_is_rejected():
    with pytest.raises(ValueError):
        tx("int f(void) { return 0; }\n", ("no_such_kind",))


def test_only_the_selected_kinds_are_applied():
    out, st = tx(COM_SRC + EQZ_SRC, ("commute",))
    assert st["total"] == st["commute"] and st["relation_flip"] == st["eq_zero"] == st["opaque"] == 0
    assert "(a == 0)" not in out and "r += b > a;" not in out


def test_stats_have_the_same_shape_as_structure_stats_and_are_json_serializable():
    _, st = tx(REL_SRC, KINDS)
    for key in (*KINDS, "total", "candidates", "skipped", "rolled_back", "files_total", "files_changed",
                "files_skipped", "files", "seed", "ratio", "kinds"):
        assert key in st, key
    assert set(st["candidates"]) == set(st["skipped"]) == set(st["rolled_back"]) == set(KINDS)
    assert st["total"] == sum(st[k] for k in KINDS)
    assert json.loads(json.dumps(st)) == st


# --------------------------------------------------------------------------- 9. 目录级入口

DIR_SRC = "int d1(int a, int b) { return a < b; }\n"


def test_directory_entry_point_respects_vendored_and_files(scratch):
    write_tree(scratch, {"a.c": DIR_SRC, "b.c": DIR_SRC, "ext/lib.c": DIR_SRC, "notes.txt": "a < b\n"})
    spec = make_spec(scratch, ["a.c", "b.c", "ext/lib.c", "notes.txt"], vendored=["ext"])
    st = transform_semantic(spec, scratch, 1, ratio=1.0)
    assert (scratch / "a.c").read_text() != DIR_SRC and (scratch / "b.c").read_text() != DIR_SRC
    assert (scratch / "ext/lib.c").read_text() == DIR_SRC          # vendored 不动
    assert (scratch / "notes.txt").read_text() == "a < b\n"        # 非 C 文件不动
    assert set(st["files"]) == {"a.c", "b.c"} and st["files_total"] == 2 and st["files_changed"] == 2

    write_tree(scratch, {"a.c": DIR_SRC, "b.c": DIR_SRC})
    st = transform_semantic(spec, scratch, 1, ratio=1.0, files=["b.c"])
    assert (scratch / "a.c").read_text() == DIR_SRC and (scratch / "b.c").read_text() != DIR_SRC

    st = transform_semantic(spec, scratch, 1, ratio=1.0, include_vendored=True)
    assert (scratch / "ext/lib.c").read_text() != DIR_SRC and "ext/lib.c" in st["files"]


def test_symlink_is_not_written_through(scratch):
    """工作副本里的符号链接可能指向副本之外（例如原始语料目录），写下去会改坏原件：L2 有这个保护，L3 以前没有。"""
    outside = scratch / "outside"
    outside.mkdir()
    (outside / "real.c").write_text(DIR_SRC)
    root = scratch / "proj"
    root.mkdir()
    (root / "link.c").symlink_to(outside / "real.c")
    (root / "plain.c").write_text(DIR_SRC)
    spec = make_spec(root, ["link.c", "plain.c"])
    st = transform_semantic(spec, root, 1, ratio=1.0)
    assert (outside / "real.c").read_text() == DIR_SRC                 # 链接目标没被改
    assert (root / "plain.c").read_text() != DIR_SRC                   # 普通文件照常改
    assert "link.c" in st["files_skipped"] and st["files_changed"] == 1


def test_header_that_does_not_parse_alone_borrows_a_host_c_file(scratch):
    """头文件靠别的头才自洽（单独解析有 error）时，借一个包含它的 .c 当宿主解析，和 L2 一致；以前 L3 整份跳过。"""
    write_tree(scratch, {
        "t.h": "static int tlt(int a, int b, va_list ap) { return a < b; }\n",
        "m.c": '#include <stdarg.h>\n#include "t.h"\nint main(void) { return 0; }\n'})
    spec = make_spec(scratch, ["t.h", "m.c"])
    st = transform_semantic(spec, scratch, 1, ratio=1.0)
    assert "return b > a;" in (scratch / "t.h").read_text()
    assert st["files"]["t.h"]["host"] == "m.c" and "t.h" not in st["files_skipped"]


def test_functions_in_header_files_are_rewritten_too(scratch):
    write_tree(scratch, {
        "h.h": "#ifndef H_H\n#define H_H\nstatic int lt(int a, int b) { return a < b; }\n#endif\n",
        "m.c": '#include "h.h"\nint main(void) { return lt(1, 2) ? 0 : 1; }\n'})
    spec = make_spec(scratch, ["h.h", "m.c"])
    st = transform_semantic(spec, scratch, 1, ratio=1.0)
    assert "return b > a;" in (scratch / "h.h").read_text()
    assert st["files_changed"] == 2


def test_every_mini_target_can_be_relocated_in_the_l3_copy(scratch):
    """verify 靠 extract.locate 按名字重新定位目标：函数体里多了 opaque 语句，定位仍须成功，且函数体以它开头。"""
    spec = corpus.load_spec_file(MINI / "spec.json")
    targets = extract.extract_functions(spec, MINI)
    assert len(targets) == 40
    work = scratch / "mini"
    corpus.materialize(MINI, work)
    transform_semantic(spec, work, SEED, ratio=1.0)
    for t in targets:
        found = extract.locate(spec, work, t.name, t.file)
        assert found is not None, t.target_id
        data = (work / t.file).read_bytes()
        assert data[found.body_start:found.body_end].decode("latin-1").startswith("{ { volatile int v_"), t.target_id
        assert data[found.start:found.end].rstrip().endswith(b"}")


def test_transform_semantic_is_deterministic_on_a_directory(scratch):
    digests = []
    for name in ("one", "two"):
        d = scratch / name
        corpus.materialize(MINI, d)
        transform_semantic(corpus.load_spec_file(MINI / "spec.json"), d, SEED)
        digests.append(corpus.tree_digest(d))
    assert digests[0] == digests[1]
    pristine = scratch / "pristine"
    corpus.materialize(MINI, pristine)
    assert corpus.tree_digest(pristine) != digests[0]


# --------------------------------------------------------------------------- 10. 沙箱对拍：手写程序与随机表达式程序
#
# 同一份 C 程序改写前后分别在 -O0 / -O2 下编译运行，输出（含短路求值计数、__LINE__）必须逐字节相同。
# 原程序本身必须无未定义行为：-O0 与 -O2 输出一致，且能过 -fsanitize=undefined。

EQUIV_C = r"""#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>

#define LIMIT 10
#define SQ(x) ((x) * (x))
#define LT(a, b) ((a) < (b))
#define SHOW(e) printf("%s => %d\n", #e, (int)(e))

enum color { RED, GREEN = 3, BLUE };
typedef struct { int x, y; unsigned u; double d; signed char sc; unsigned char uc; short sh; long long ll; } rec_t;
struct node { int val; struct node *next; rec_t r; };
struct bits { unsigned a : 3; int b : 5; };

static int g_calls;
static int bump(int v) { g_calls++; return v; }
static int g_arr[4] = { 0, 1, 0, 2 };

static unsigned rel_ints(int a, int b, unsigned ua, unsigned ub, long la, long lb)
{
    unsigned r = 0;
    if (a < b) r |= 1u << 0;
    if (a <= b) r |= 1u << 1;
    if (a > b) r |= 1u << 2;
    if (a >= b) r |= 1u << 3;
    if (ua < ub) r |= 1u << 4;
    if (ua <= ub) r |= 1u << 5;
    if (la > lb) r |= 1u << 6;
    if (la >= lb) r |= 1u << 7;
    if (a < ua) r |= 1u << 8;
    if (ua >= a) r |= 1u << 9;
    if (a < 3) r |= 1u << 10;
    if (3 >= a) r |= 1u << 11;
    if (a /* c1 */ <
        b) r |= 1u << 12;
    if (ub > 0xFFFFFFF0u) r |= 1u << 13;
    return r;
}

static unsigned rel_fp(double a, double b, float f, float g)
{
    unsigned r = 0;
    if (a < b) r |= 1u << 0;
    if (a <= b) r |= 1u << 1;
    if (a > b) r |= 1u << 2;
    if (a >= b) r |= 1u << 3;
    if (f < g) r |= 1u << 4;
    if (f >= g) r |= 1u << 5;
    if (a < f) r |= 1u << 6;
    if (g >= b) r |= 1u << 7;
    if (a < 1.5) r |= 1u << 8;
    if (2.5f >= f) r |= 1u << 9;
    return r;
}

static unsigned rel_small(char c, unsigned char uc, signed char sc, short sh, _Bool bb)
{
    unsigned r = 0;
    if (c < 'a') r |= 1u << 0;
    if (c >= '0') r |= 1u << 1;
    if (uc > 200) r |= 1u << 2;
    if (uc <= sc) r |= 1u << 3;
    if (sc < sh) r |= 1u << 4;
    if (sh > c) r |= 1u << 5;
    if (bb > c) r |= 1u << 6;
    if (bb <= uc) r |= 1u << 7;
    return r;
}

static unsigned rel_members(struct node *n, rec_t v)
{
    unsigned r = 0;
    if (n->val < n->r.x) r |= 1u << 0;
    if (n->r.y <= n->next->r.y) r |= 1u << 1;
    if (v.u > v.sh) r |= 1u << 2;
    if (v.d >= n->r.d) r |= 1u << 3;
    if (v.ll < n->r.ll) r |= 1u << 4;
    if (v.uc < v.sc) r |= 1u << 5;
    if (n->next->val >= LIMIT) r |= 1u << 6;
    if (v.x < SQ(3)) r |= 1u << 7;
    if (LT(v.x, v.y)) r |= 1u << 8;
    return r;
}

static unsigned rel_enum(enum color e, int i)
{
    unsigned r = 0;
    if (e < BLUE) r |= 1u << 0;
    if (e >= GREEN) r |= 1u << 1;
    if (RED == e) r |= 1u << 2;
    if (!e) r |= 1u << 3;
    if (e == 0) r |= 1u << 4;
    if (i > e) r |= 1u << 5;
    if (!(e == BLUE || e == GREEN)) r |= 1u << 6;
    return r;
}

static double com(int a, int b, unsigned ua, unsigned ub, double d, double e, float f, long long ll, rec_t *p)
{
    double r = 0;
    r += a + b;
    r += (double)(ua * ub);
    r += (double)(ua + ub);
    r += d + e;
    r += d * e;
    r += f * f;
    r += f + a;
    r += (double)(ll * ll);
    r += (double)(ll + 7);
    r += p->x + p->y;
    r += p->d * p->u;
    r += a * 3 + b * 2;
    r += (a + b) * (a - b);
    r += a * b + ua * ub;
    r += (double)(a + p->sh);
    r += (double)(p->sc * p->uc);
    r += 2 + a;
    return r;
}

static unsigned ptrs_rel(int *p, int *q)
{
    unsigned r = 0;
    if (p < q) r |= 1u << 0;
    if (p >= q) r |= 1u << 1;
    if (p + 1 > q) r |= 1u << 2;
    if (!(p && q)) r |= 1u << 9;
    if (!(p || q)) r |= 1u << 10;
    if (!(p && *p)) r |= 1u << 11;
    return r;
}

static unsigned ptrs_null(int *p, const char *s)
{
    unsigned r = 0;
    if (!p) r |= 1u << 3;
    if (p == 0) r |= 1u << 4;
    if (s == NULL) r |= 1u << 5;
    if (!s) r |= 1u << 6;
    if (s != 0) { if (*s == 0) r |= 1u << 7; if (!*s) r |= 1u << 8; if (s[0] == '\0') r |= 1u << 12; }
    return r;
}

static unsigned eqz(int a, int b, int c, unsigned u, long long ll, char ch, _Bool bb, double d, int *gp)
{
    unsigned r = 0;
    int t = a;
    unsigned cu = (unsigned)c;
    double dc = (d == d && d > -1e9 && d < 1e9) ? d : 0.0;
    if (!a) r |= 1u << 0;
    if (b == 0) r |= 1u << 1;
    if (0 == c) r |= 1u << 2;
    if (u == 0) r |= 1u << 3;
    if (!ll) r |= 1u << 4;
    if (ch == 0) r |= 1u << 5;
    if (!bb) r |= 1u << 6;
    if (!d) r |= 1u << 7;
    if (d == 0) r |= 1u << 8;
    if ((long long)a + b == 0) r |= 1u << 9;
    if ((long long)a - b == 0) r |= 1u << 10;
    if ((a & b) == 0) r |= 1u << 11;
    if ((unsigned)a << 1 == 0) r |= 1u << 12;
    if (a % 3 == 0) r |= 1u << 13;
    if (-(long long)a == 0) r |= 1u << 14;
    if (~a == 0) r |= 1u << 15;
    if (*gp == 0) r |= 1u << 16;
    if (g_arr[a & 3] == 0) r |= 1u << 17;
    if (!g_arr[b & 3]) r |= 1u << 18;
    if ((a ? b : c) == 0) r |= 1u << 19;
    if ((int)dc == 0) r |= 1u << 20;
    if (((void)0, a) == 0) r |= 1u << 21;
    if (!((long long)a + b)) r |= 1u << 22;
    if (!(a & b)) r |= 1u << 23;
    if (!*gp) r |= 1u << 24;
    if (!(t = b)) r |= 1u << 25;
    if (!(cu++)) r |= 1u << 26;
    if (!bump(a)) r |= 1u << 27;
    if (bump(b) == 0) r |= 1u << 28;
    if (!-(long long)b) r |= 1u << 29;
    if (a < b == 0) r |= 1u << 30;
    if (!!a == 0) r |= 1u << 31;
    return r + (unsigned)t + cu;
}

static unsigned demorgan(int a, int b, int c, int d, int *p)
{
    unsigned r = 0;
    if (!(a && b)) r |= 1u << 0;
    if (!(a || b)) r |= 1u << 1;
    if (!(a && !(b || c))) r |= 1u << 2;
    if (!(bump(a) && bump(b))) r |= 1u << 3;
    if (!(bump(a) || bump(b))) r |= 1u << 4;
    if (!(p && p[0])) r |= 1u << 5;
    if (!(p == 0 || *p == 0)) r |= 1u << 6;
    if (!(a < b && c > d)) r |= 1u << 7;
    if (!(a == 1 || b == 2 || c == 3)) r |= 1u << 8;
    if (!(a & 1 && b | 2)) r |= 1u << 9;
    if (!((a = 0) || b)) r |= 1u << 10;
    if (!(a == b && (c, d))) r |= 1u << 11;
    if (!(a /* x */ && // y
          b)) r |= 1u << 12;
    if (!( a
           || b )) r |= 1u << 13;
    r += (unsigned)(!(a && b)) * 3u;
    return r + (unsigned)a;
}

static int sw(int k, int n)
{
    int r = 0;
    switch (k + n) {
    case 1 + 2: r = 10; break;
    case 2 * 3: r = 20; break;
    case 7: r = 30; break;
    default: r = k < n ? 1 : 2;
    }
    {
        int arr[2 + LIMIT];
        arr[0] = k;
        r += (int)sizeof(arr) + (int)sizeof(k < n);
    }
    return r;
}

static int loops(int n)
{
    int i, s = 0, j = 0;
    for (i = 0; i < n; i++) s += i;
    while (j <= n) { s += j; j++; }
    do { s++; } while (s < n);
    return s;
}

static long fib(long n)
{
    if (n < 2) return n;
    return fib(n - 1) + fib(n - 2);
}

static int kr(a, b)
    int a;
    int b;
{
    if (a < b) return b - a;
    return a - b;
}

static int bf(struct bits *s)
{
    return (s->a < s->b) + 2 * (s->b >= 0) + 4 * !s->a + 8 * (s->a == 0);
}

static int vol(volatile int *v, volatile int w)
{
    int r = 0;
    if (*v < 3) r += 1;
    if (w < 3) r += 2;
    if (w + w > 6) r += 4;
    return r;
}

static int lbl(int n)
{
    int r = 0;
again:
    if (r < n) { r += 2; goto again; }
    return r;
}

static int ternary_chain(int a, int b)
{
    return a < b ? (a == 0 ? 1 : 2) : (b == 0 ? !a : !b);
}

int main(void)
{
    int ivals[] = { INT_MIN, -7, -1, 0, 1, 2, 3, 10, 11, INT_MAX };
    unsigned uvals[] = { 0u, 1u, 3u, 0x7fffffffu, 0x80000000u, 0xffffffffu };
    double dvals[] = { 0.0, -0.0, -1.5, 1.5, 2.25, NAN, INFINITY, -INFINITY };
    size_t i, j, k;
    unsigned long long h = 14695981039346656037ULL;
    int gz = 0, gnz = 5;
    struct node n2 = { 4, 0, { 1, 2, 3u, 4.5, 5, 6, 7, 8 } };
    struct node n1 = { 3, &n2, { 0, 9, 8u, 1.5, -2, 250, -3, 123456789012LL } };
    struct bits bs = { 5, -4 };
    volatile int vv = 2;

    n2.next = &n1;

#define MIX(x) (h = (h ^ (unsigned long long)(unsigned)(x)) * 1099511628211ULL)
    for (i = 0; i < sizeof ivals / sizeof ivals[0]; i++)
        for (j = 0; j < sizeof ivals / sizeof ivals[0]; j++) {
            for (k = 0; k < 3; k++) {
                MIX(rel_ints(ivals[i], ivals[j], uvals[k], uvals[(k + i) % 6], (long)ivals[i] * 3, (long)ivals[j] * 3));
                MIX(eqz(ivals[i], ivals[j], ivals[(i + j) % 10], uvals[k], (long long)ivals[i] - ivals[j],
                        (char)ivals[i], (_Bool)(ivals[j] & 1), dvals[(i + j) % 8], (k & 1) ? &gz : &gnz));
                MIX(demorgan(ivals[i], ivals[j], ivals[(i + 1) % 10], ivals[(j + 3) % 10], (k == 0) ? 0 : (k == 1 ? &gz : &gnz)));
                MIX(g_calls);
            }
            MIX(sw(ivals[i] & 7, ivals[j] & 7));
            MIX(kr(ivals[i] / 2, ivals[j] / 2));
            MIX(ternary_chain(ivals[i], ivals[j]));
        }
    for (i = 0; i < 8; i++)
        for (j = 0; j < 8; j++) {
            MIX(rel_fp(dvals[i], dvals[j], (float)dvals[(i + 1) % 8], (float)dvals[(j + 2) % 8]));
            union { double d; unsigned long long u; } cv;
            cv.d = com((int)(i * 3 - 5), (int)(j + 1), (unsigned)i * 77u, (unsigned)j * 4000000000u, dvals[i], dvals[j],
                       (float)dvals[j], (long long)i * 1000007LL, &n1.r);
            MIX((unsigned)(cv.u >> 32));
            MIX((unsigned)cv.u);
        }
    for (i = 0; i < 256; i += 5) MIX(rel_small((char)i, (unsigned char)i, (signed char)i, (short)(i * 300), (_Bool)(i & 4)));
    MIX(rel_members(&n1, n2.r));
    MIX(rel_members(&n2, n1.r));
    for (i = 0; i < 6; i++) MIX(rel_enum((enum color)(i < 3 ? i : (i == 3 ? 3 : 4)), (int)i - 1));
    {
        int pa[2] = { 1, 2 };
        MIX(ptrs_rel(&pa[0], &pa[1]));
        MIX(ptrs_rel(&pa[1], &pa[0]));
        MIX(ptrs_rel(&pa[0], &pa[0]));
        MIX(ptrs_null(&pa[0], "hello"));
        MIX(ptrs_null(0, ""));
        MIX(ptrs_null(&pa[1], 0));
    }
    for (i = 0; i < 12; i++) { MIX(loops((int)i)); MIX(lbl((int)i)); MIX((unsigned long long)fib((long)i)); }
    MIX(bf(&bs));
    MIX(vol(&vv, vv + 2));
    MIX(vol(&vv, vv - 2));
    printf("hash %llu calls %d\n", h, g_calls);
    SHOW(gz < gnz);
    SHOW(!(gz && gnz));
    SHOW(gz == 0);
    SHOW(SQ(gnz) > LIMIT);
    printf("line %d\n", __LINE__);
    return 0;
}
"""


def compile_and_run(sandbox: DockerSandbox, work: Path, flags: str = "", ubsan: bool = False) -> dict[str, str]:
    """在沙箱里分别用 -O0 / -O2（可选再加一个 UBSan 版本）编译 prog.c 并运行，返回各版本的标准输出。"""
    steps = [f"gcc -std=c99 -O0 -w {flags} -o p0 prog.c -lm && ./p0 > out0.txt",
             f"gcc -std=c99 -O2 -w {flags} -o p2 prog.c -lm && ./p2 > out2.txt"]
    if ubsan:
        steps.append(f"gcc -std=c99 -O1 -w {flags} -fsanitize=undefined -fno-sanitize-recover=all -o pu prog.c -lm"
                     " && ./pu > outu.txt")
    res = sandbox.run(work, " ; ".join(steps), timeout_s=170)
    outs = {}
    for tag, fn in (("O0", "out0.txt"), ("O2", "out2.txt"), ("UBSAN", "outu.txt")):
        p = work / fn
        if p.exists():
            outs[tag] = p.read_text()
    need = {"O0", "O2"} | ({"UBSAN"} if ubsan else set())
    assert need <= set(outs) and all(outs[t] for t in need), (
        f"编译或运行失败: rc={res.rc}\n{res.stdout[-800:]}\n{res.stderr[-2500:]}")
    return outs


def rewrite_program(work: Path, text: str, ratio: float, seed: int) -> dict:
    (work / "prog.c").write_text(text)
    spec = make_spec(work, ["prog.c"])
    return transform_semantic(spec, work, seed, ratio=ratio)


@pytest.fixture(scope="module")
def equiv_base(sandbox, scratch_module):
    base = scratch_module / "equiv"
    base.mkdir()
    (base / "prog.c").write_text(EQUIV_C)
    outs = compile_and_run(sandbox, base, ubsan=True)
    assert outs["O0"] == outs["O2"] == outs["UBSAN"], "手写对拍程序在不同优化级别下输出不同，说明它自己有未定义行为"
    return outs["O0"]


@needs_docker
@pytest.mark.parametrize("ratio,seed", [(1.0, 1), (1.0, 2), (0.5, 3)])
def test_handwritten_program_behaves_identically_after_rewrite(scratch, sandbox, equiv_base, ratio, seed):
    st = rewrite_program(scratch, EQUIV_C, ratio, seed)
    assert all(st[k] > 0 for k in KINDS), st          # 五类改写在这个程序里都有机会
    assert sum(st["rolled_back"].values()) == 0
    outs = compile_and_run(sandbox, scratch)
    assert outs["O0"] == equiv_base and outs["O2"] == equiv_base


class ExprGen:
    """随机生成无未定义行为的整型表达式（只含 + - * & | ^ 比较 逻辑运算 ! ~ 三目 强转 和一个带副作用的调用）。

    有符号溢出靠 -fwrapv 兜底（改写不改变回绕语义）；没有除法、取模、移位，指针只做成员访问。
    """

    ATOM_RE = re.compile(r"^(?:[A-Za-z_]\w*(?:(?:\.|->)[A-Za-z_]\w*)*|\d+u?|0x[0-9A-Fa-f]+u)$")

    def __init__(self, seed: int) -> None:
        self.r = random.Random(seed)

    def atom(self) -> str:
        r = self.r
        k = r.randrange(11)
        if k < 4:
            return r.choice("abcd")
        if k == 4:
            return r.choice(["u", "v"])
        if k == 5:
            return "ch"
        if k == 6:
            return "ll"
        if k == 7:
            return r.choice(["p->x", "p->y", "p->q->x"])
        if k == 8:
            return str(r.choice([0, 0, 1, 2, 3]))
        if k == 9:
            return r.choice(["0u", "1u", "3u", "0xFFFFFFFFu"])
        return r.choice(["sg.a", "sg.b"])

    @classmethod
    def wrap(cls, t: str) -> str:
        if cls.ATOM_RE.match(t):
            return t
        depth = 0
        for i, ch in enumerate(t):
            depth += (ch == "(") - (ch == ")")
            if depth == 0 and i != len(t) - 1:
                break
        else:
            if t.startswith("(") and t.endswith(")"):
                return t
        return f"({t})"

    def gen(self, depth: int) -> str:
        r = self.r
        if depth <= 0 or r.random() < 0.18:
            return self.atom()
        k = r.randrange(100)
        sub = lambda: self.gen(depth - 1)          # noqa: E731
        if k < 22:
            return f"{sub()} {r.choice(['+', '+', '*', '-', '&', '|', '^'])} {sub()}"
        if k < 40:
            return f"{sub()} {r.choice(['<', '<=', '>', '>=', '==', '!='])} {sub()}"
        if k < 50:
            return f"{sub()} {r.choice(['&&', '||'])} {sub()}"
        if k < 60:
            return f"!({sub()} {r.choice(['&&', '||'])} {sub()})"
        if k < 68:
            return f"{sub()} == 0" if r.random() < 0.7 else f"0 == {sub()}"
        if k < 76:
            return f"!{self.wrap(sub())}"
        if k < 80:
            return f"{r.choice(['-', '~'])}{self.wrap(sub())}"
        if k < 86:
            return f"({sub()})"
        if k < 92:
            return f"({sub()}) ? {self.wrap(sub())} : {self.wrap(sub())}"
        if k < 97:
            return f"t(({sub()}) != 0)"
        return f"{r.choice(['(long long)', '(unsigned)', '(int)'])}{self.wrap(sub())}"


FUZZ_HEAD = """#include <stdio.h>
struct inner { int x, y; };
struct outer { int x, y; struct inner *q; };
struct bits { int a, b; };
static int g_calls;
static int t(int v) { g_calls++; return v; }
"""
FUZZ_SIG = ("int a, int b, int c, int d, unsigned u, unsigned v, signed char ch, long long ll, "
            "struct outer *p, struct bits sg")


def fuzz_program(seed: int, n: int = 40, depth: int = 5) -> str:
    g = ExprGen(seed)
    out = [FUZZ_HEAD]
    names = []
    for k in range(n):
        e1, e2, e3 = g.gen(depth), g.gen(depth), g.gen(depth)
        out.append(f"static int f{k}({FUZZ_SIG})\n{{\n    return (int)({e1});\n}}\n")
        out.append(f"static int g{k}({FUZZ_SIG})\n{{\n    int r = 0;\n    if ({e2}) r += 1;\n"
                   f"    if (!({e3})) r += 2;\n    r += ({e1}) ? 4 : 0;\n"
                   f"    while ({e2}) {{ r += 8; break; }}\n    return r;\n}}\n")
        names += [f"f{k}", f"g{k}"]
    out.append("int main(void)\n{\n    unsigned long long h = 1469598103934665603ULL;\n"
               "    int iv[] = { -2, 0, 1 };\n    unsigned uv[] = { 0u, 3u, 0xFFFFFFFFu };\n"
               "    signed char cv[] = { -1, 5 };\n    long long lv[] = { -2, 7 };\n"
               "    struct inner qi = { 2, -1 };\n    struct outer po = { 1, 0, &qi };\n    struct bits sg;\n"
               "    int ia, ib, ic, id, iu, iw, ie, il, ip;\n"
               "    for (ia = 0; ia < 3; ia++) for (ib = 0; ib < 3; ib++) for (ic = 0; ic < 3; ic++) "
               "for (id = 0; id < 3; id++)\n"
               "    for (iu = 0; iu < 3; iu++) for (iw = 0; iw < 2; iw++) for (ie = 0; ie < 2; ie++) "
               "for (il = 0; il < 2; il++) for (ip = 0; ip < 3; ip++) {\n"
               "        po.x = iv[ip]; po.y = iv[(ip + 1) % 3]; qi.x = iv[(ip + 2) % 3];\n"
               "        sg.a = iv[ia]; sg.b = iv[(ib + ip) % 3];\n")
    for nm in names:
        out.append(f"        h = (h ^ (unsigned)({nm}(iv[ia], iv[ib], iv[ic], iv[id], uv[iu], uv[(iu + iw) % 3], "
                   f"cv[ie], lv[il], &po, sg) + g_calls)) * 1099511628211ULL;\n")
    out.append('    }\n    printf("%llu %d\\n", h, g_calls);\n    return 0;\n}\n')
    return "".join(out)


@needs_docker
@pytest.mark.parametrize("gen_seed", [1, 2])
def test_random_expression_programs_behave_identically_after_rewrite(scratch, sandbox, gen_seed):
    text = fuzz_program(gen_seed)
    base_dir = scratch / "base"
    base_dir.mkdir()
    (base_dir / "prog.c").write_text(text)
    base = compile_and_run(sandbox, base_dir, flags="-fwrapv")
    assert base["O0"] == base["O2"]
    for ratio, seed in ((1.0, 1), (0.5, 2)):
        work = scratch / f"r{ratio}s{seed}"
        work.mkdir()
        st = rewrite_program(work, text, ratio, seed)
        assert st["de_morgan"] > 20 and st["eq_zero"] > 20 and st["relation_flip"] > 10 and st["commute"] > 10, st
        assert sum(st["rolled_back"].values()) == 0
        outs = compile_and_run(sandbox, work, flags="-fwrapv")
        assert outs["O0"] == base["O0"] and outs["O2"] == base["O0"], f"ratio={ratio} seed={seed}"


# --------------------------------------------------------------------------- 11. 沙箱：mini 夹具与真实语料
#
# 改写后的副本必须仍能通过项目自带测试（pass_criteria 与 oracle gate 同一口径）。
# 语料目录从 corpus/src 或本地镜像复制到 tmp/ 下的临时目录，原目录不动。

def l3_copy(src: Path, dest: Path, spec, ratio: float, seed: int, **kw) -> dict:
    corpus.materialize(src, dest)
    return transform_semantic(spec, dest, seed, ratio=ratio, **kw)


def summarize(name: str, ratio: float, st: dict) -> str:
    line = (f"[L3] {name:13s} ratio={ratio}: " + " ".join(f"{k}={st[k]}" for k in KINDS)
            + f" total={st['total']} functions={st['functions']} rolled_back={sum(st['rolled_back'].values())}")
    print(line)
    return line


@needs_docker
@pytest.mark.parametrize("ratio", [1.0, 0.5])
def test_mini_fixture_passes_its_own_tests_after_l3(scratch, sandbox, ratio):
    spec = corpus.load_spec_file(MINI / "spec.json")
    work = scratch / "mini"
    st = l3_copy(MINI, work, spec, ratio, SEED)
    summarize("mini", ratio, st)
    assert st["total"] > 0 and sum(st["rolled_back"].values()) == 0 and not st["files_skipped"]
    if ratio == 1.0:
        assert all(st[k] > 0 for k in KINDS), st            # mini 里五类改写都有用武之地
    assert corpus.tree_digest(work) != corpus.tree_digest(MINI)
    ok, res = run_spec(sandbox, spec, work)
    assert ok and "0 failures" in res.stdout, res.stdout[-800:] + res.stderr[-1500:]


@needs_docker
@pytest.mark.parametrize("name", REQUIRED_CORPORA + OTHER_CORPORA)
def test_real_corpus_passes_its_tests_after_l3_at_full_ratio(scratch, sandbox, name):
    spec = corpus.load_spec(name)
    src = corpus_source(name)
    work = scratch / name
    st = l3_copy(src, work, spec, 1.0, SEED)
    summarize(name, 1.0, st)
    assert st["total"] > 0
    assert sum(st["rolled_back"].values()) == 0, st["rolled_back"]
    assert st["files_skipped"] == {}, st["files_skipped"]          # jsmn 的 test/testutil.h 不自含，借 tests.c 当宿主解析
    assert st["opaque"] == st["candidates"]["opaque"]          # opaque_ratio=1：每个合格的函数体都插了
    assert st["candidates"]["opaque"] + sum(st["skipped"]["opaque"].values()) == st["functions"]
    assert st["overlap_dropped"] == 0
    assert corpus.tree_digest(work) != corpus.tree_digest(src)
    if name in REQUIRED_CORPORA:
        assert all(st[k] > 0 for k in KINDS if k != "de_morgan"), st    # 原始语料里显式 !( && ) 很少，不强求
    if name == "jsmn":
        assert st["files"]["test/testutil.h"]["host"] == "tests.c"
    check_passes(sandbox, spec, work, f"{name} 全量(ratio=1.0)语义混淆后的副本")


@needs_docker
@pytest.mark.parametrize("name", REQUIRED_CORPORA)
def test_real_corpus_passes_its_tests_after_l3_at_default_ratio(scratch, sandbox, name):
    spec = corpus.load_spec(name)
    work = scratch / name
    st = l3_copy(corpus_source(name), work, spec, semantic.DEFAULT_RATIO, SEED + 1)
    summarize(name, semantic.DEFAULT_RATIO, st)
    assert st["total"] > 0 and sum(st["rolled_back"].values()) == 0
    check_passes(sandbox, spec, work, f"{name} 默认比例(ratio=0.5)语义混淆后的副本")


@needs_docker
@pytest.mark.parametrize("name", ["mini"] + REQUIRED_CORPORA)
def test_l3_runs_on_top_of_l2_structure_obfuscation(scratch, sandbox, name):
    if name == "mini":
        src, spec = MINI, corpus.load_spec_file(MINI / "spec.json")
    else:
        src, spec = corpus_source(name), corpus.load_spec(name)
    work = scratch / name
    corpus.materialize(src, work)
    s2 = structure.transform_structure(spec, work, SEED)
    s3 = transform_semantic(spec, work, SEED, ratio=1.0)
    print(f"[L2+L3] {name:8s} L2={ {k: s2[k] for k in structure.KINDS} } L3={ {k: s3[k] for k in KINDS} }")
    assert s2["total"] > 0 and s3["total"] > 0
    assert sum(s3["rolled_back"].values()) == 0 and not s3["files_skipped"]
    check_passes(sandbox, spec, work, f"{name} 经 L2 + L3 之后的副本")


@needs_docker
@pytest.mark.parametrize("name", ["mini"] + REQUIRED_CORPORA)
def test_l3_runs_on_top_of_l1_symbols_and_l2_structure(scratch, sandbox, name):
    if name == "mini":
        src, spec = MINI, corpus.load_spec_file(MINI / "spec.json")
    else:
        src, spec = corpus_source(name), corpus.load_spec(name)
    work = scratch / name
    corpus.materialize(src, work)
    mapping = symbols.build_rename_map(symbols.collect_project_identifiers(spec, work), SEED)
    symbols.apply_renames(spec, work, mapping)
    structure.transform_structure(spec, work, SEED)
    st = transform_semantic(spec, work, SEED, ratio=1.0)
    summarize(f"{name} L1+L2", 1.0, st)
    assert st["total"] > 0 and sum(st["rolled_back"].values()) == 0 and not st["files_skipped"]
    check_passes(sandbox, spec, work, f"{name} 经 L1 + L2 + L3 之后的副本")


@needs_docker
def test_l3_is_safe_to_run_twice(scratch, sandbox):
    spec = corpus.load_spec_file(MINI / "spec.json")
    work = scratch / "mini"
    corpus.materialize(MINI, work)
    transform_semantic(spec, work, 1, ratio=1.0)
    st = transform_semantic(spec, work, 2, ratio=1.0)
    assert st["total"] > 0 and sum(st["rolled_back"].values()) == 0
    check_passes(sandbox, spec, work, "mini 连做两遍 L3 之后的副本")
