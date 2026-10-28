"""bench.mutate（缺陷修复任务的变异算子与变异体选择，附录 §3、§4、§8）的测试。

分三部分：

1. 算子（不需要 docker）：八个算子各自的最小用例——源码进、变异后源码出、重新解析零 error、位点记录的原文正确；
   宏、预处理指令、头文件目标等边界；排序与确定性。
2. 选择逻辑（假沙箱，不需要 docker）：假沙箱按候选内容给出预设的评测结果，核对试位点的顺序、杀死判定、
   等价 / 编译失败计数、``max_tries``、``prefer_operator``、跨级别复用算子、并发调用与重复运行的一致性。
3. 真沙箱（需要 docker）：自写的小项目上的等价变异体与编译失败；mini 夹具上每个入选目标、每一级都选出被杀死的变异体。
"""
from __future__ import annotations

import dataclasses
import random
import re
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from bench import corpus, extract, mutate
from bench.mutate import Mutant, Site
from bench.obfuscate import LEVELS, obfuscate
from bench.obfuscate._edits import Parser, Unit, libclang_args
from bench.runner import evaluate
from bench.sandbox import DockerSandbox, RunResult
from bench.store import Workspace
from bench.targets import TargetFilter, select
from bench.vocab import KILL_MODES, MUTATION_TITLES
from tests.helpers import MINI, needs_docker

SEED = 20261027


# ----------------------------------------------------------------------
# 小工具
# ----------------------------------------------------------------------
def write_files(root: Path, files: dict[str, str]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (root / name).write_text(text, encoding="utf-8")
    return root


def make_project(base: Path, source: str, *, name: str = "t", file: str = "t.c", **spec_kw):
    """只有一个源文件的小项目（不构建，只给 libclang 用）：返回 (spec, 项目根)。"""
    root = write_files(base / name, {file: source})
    kw = {"cflags": ["-std=c99"], **spec_kw}
    spec = corpus.CorpusSpec(name=name, project_files=[file], target_files=[file], **kw)
    return spec, root


def target_of(spec, root: Path, func: str = "f", file: str = "t.c"):
    return next(t for t in extract.extract_file(spec, root, file) if t.name == func)


def function_text(root: Path, target) -> str:
    return (root / target.file).read_bytes()[target.start:target.end].decode()


def reparse_errors(spec, root: Path, file: str, data: bytes) -> list[str]:
    """data 当作 root/file 的内容重新解析，返回 error 级诊断。"""
    parser = Parser(libclang_args(spec, root))
    return Parser.errors(parser.parse(Unit(root / file), data, detailed=False))


def mutate_site(spec, root: Path, target, site: Site, seed: int = 0) -> tuple[bytes, str]:
    """(变异后的整份文件, 变异后的函数文本)。"""
    data = (root / target.file).read_bytes()
    new = mutate.apply_site(data, site, random.Random(seed))
    return new, new[target.start:target.end + len(new) - len(data)].decode()


def sub(text: str, old: str, new: str) -> str:
    """text 里恰好出现一次的 old 换成 new（出现次数不对就是用例写错了）。"""
    assert text.count(old) == 1, (old, text.count(old))
    return text.replace(old, new)


def check_sites(scratch: Path, src: str, operator: str, expected: list[tuple[str, str, str]], *, func: str = "f",
                seed: int = 0, **kw) -> list[Site]:
    """operator 在 func 里的位点必须恰好是 expected 里的这些（按源码位置）：
    每项 (位点原文, 函数文本里被换掉的片段, 换成的片段)，变异后的函数必须等于「原函数里把片段换掉」，
    整份文件重新解析零 error，位点记录的原文与源码区间一致。"""
    spec, root = make_project(scratch, src, **kw)
    target = target_of(spec, root, func)
    original = function_text(root, target)
    data = (root / "t.c").read_bytes()
    sites = mutate.find_sites(spec, root, target, operator)
    assert [s.original for s in sites] == [e[0] for e in expected]
    assert [s.index for s in sites] == list(range(len(sites)))
    for site, (orig, old, new) in zip(sites, expected):
        assert site.operator == operator and data[site.start:site.end].decode() == site.original == orig
        new_file, new_func = mutate_site(spec, root, target, site, seed)
        assert new_func == sub(original, old, new), site
        assert reparse_errors(spec, root, "t.c", new_file) == [], site
    return sites


# ----------------------------------------------------------------------
# 1. 八个算子的最小用例
# ----------------------------------------------------------------------
class TestOperators:
    def test_ror_swaps_each_relational_operator(self, scratch):
        src = """int f(int a, int b)
{
    if (a < b) return 1;
    if (a <= b) return 2;
    if (a > b) return 3;
    if (a >= b) return 4;
    if (a == b) return 5;
    if (a != b) return 6;
    return 7;
}
"""
        sites = check_sites(scratch, src, "ROR", [
            ("<", "if (a < b)", "if (a <= b)"), ("<=", "if (a <= b)", "if (a < b)"),
            (">", "if (a > b)", "if (a >= b)"), (">=", "if (a >= b)", "if (a > b)"),
            ("==", "if (a == b)", "if (a != b)"), ("!=", "if (a != b)", "if (a == b)")])
        assert [s.detail for s in sites] == ["< -> <=", "<= -> <", "> -> >=", ">= -> >", "== -> !=", "!= -> =="]

    def test_neg_wraps_if_while_and_for_conditions(self, scratch):
        src = """int f(int n)
{
    int s = 0;
    int i;

    if (n > 0) {
        s = 1;
    } else if (n < -5) {
        s = 2;
    }
    while (s < 10)
        s++;
    for (i = 0; i < n; i++)
        s += i;
    for (;;) {
        break;
    }
    for (i = 0; ; i++) {
        break;
    }
    return s;
}
"""
        check_sites(scratch, src, "NEG", [
            ("n > 0", "if (n > 0)", "if (!(n > 0))"),
            ("n < -5", "else if (n < -5)", "else if (!(n < -5))"),
            ("s < 10", "while (s < 10)", "while (!(s < 10))"),
            ("i < n", "for (i = 0; i < n; i++)", "for (i = 0; !(i < n); i++)")])       # 没有条件的 for 不是位点

    def test_lor_swaps_and_or(self, scratch):
        src = """int f(int a, int b, int c)
{
    return (a && b) || c;
}
"""
        check_sites(scratch, src, "LOR", [("&&", "(a && b) || c", "(a || b) || c"), ("||", "(a && b) || c", "(a && b) && c")])

    def test_aor_swaps_binary_arithmetic_only(self, scratch):
        src = """int f(int a, int b)
{
    int x = a + b;
    int y = a - b;
    int z = a * b;
    int w = a / b;
    int m = a % b;
    int u = -a;
    int v = (a << 1) | *(&b);
    return x + y + z + w + m + u + v;
}
"""
        spec, root = make_project(scratch, src)
        target = target_of(spec, root)
        sites = mutate.find_sites(spec, root, target, "AOR")
        # 前四个是声明里的 + - * /；% 、一元负号、移位、按位或、解引用都不是位点；最后一行的 6 个加号都是
        assert [s.original for s in sites] == ["+", "-", "*", "/"] + ["+"] * 6
        new_file, new_func = mutate_site(spec, root, target, sites[3])
        assert new_func == sub(function_text(root, target), "int w = a / b;", "int w = a * b;")
        assert reparse_errors(spec, root, "t.c", new_file) == []

    def test_aor_allows_pointer_plus_or_minus_an_integer_but_not_other_pointer_arithmetic(self, scratch):
        src = """int f(const int *p, const int *q, int n)
{
    const int *r = p + n;
    const int *s = n + p;
    int d = (int)(q - p);
    return *(r - 1) + *s + d;
}
"""
        # 不是位点：n + p 改成 n - p、q - p 改成 q + p 在类型上就不合法；剩下的 p + n、r - 1 和最后一行两个整数加法都是
        check_sites(scratch, src, "AOR", [
            ("+", "p + n", "p - n"), ("-", "r - 1", "r + 1"),
            ("+", "*(r - 1) + *s", "*(r - 1) - *s"), ("+", "*s + d", "*s - d")])

    def test_obo_picks_decimal_literals_in_comparisons_and_subscripts(self, scratch):
        src = """int g(const int *, int);
int f(const int *a, int n)
{
    int t = 0;

    if (n > 10) t = a[3];
    while (t < n - 1) t = t + 7;
    if (n == 0x10) t = 1;
    if (a[n + 2] == 5u) t = 2;
    if (n == 'x') t = 3;
    return t + g(a, 4);
}
"""
        spec, root = make_project(scratch, src)
        target = target_of(spec, root)
        sites = mutate.find_sites(spec, root, target, "OBO")
        # 比较里的 10、n - 1 里的 1、下标里的 3、下标里 n + 2 的 2、5u；十六进制、字符、普通加法的 7、调用实参的 4 都不是
        assert [s.original for s in sites] == ["10", "3", "1", "2", "5u"]
        original = function_text(root, target)
        for site in sites:
            for seed in range(6):
                new_file, new_func = mutate_site(spec, root, target, site, seed)
                assert reparse_errors(spec, root, "t.c", new_file) == []
                text = mutate.mutation_text(site, random.Random(seed))
                assert text != site.original
                assert new_func == original[:site.start - target.start] + text + original[site.end - target.start:]

    def test_obo_goes_up_or_down_by_one_and_is_decided_by_the_rng(self):
        site = Site("OBO", 0, 2, "10", 0, "")
        texts = {mutate.mutation_text(site, random.Random(i)) for i in range(20)}
        assert texts == {"11", "9"}
        assert mutate.mutation_text(site, random.Random(3)) == mutate.mutation_text(site, random.Random(3))
        suffixed = Site("OBO", 0, 3, "10u", 0, "")
        assert {mutate.mutation_text(suffixed, random.Random(i)) for i in range(20)} == {"11u", "9u"}
        zero = Site("OBO", 0, 1, "0", 0, "")
        assert {mutate.mutation_text(zero, random.Random(i)) for i in range(20)} == {"1"}      # 0 不往负数改

    def test_dgs_deletes_single_statement_guards_and_leaves_no_blank_line(self, scratch):
        src = """int f(int *p, int n)
{
    int i;
    int s = 0;

    if (p == 0) return -1;
    if (n < 0) {
        return -2;
    }
    for (i = 0; i < n; i++) {
        if (p[i] < 0) break;
        if (p[i] == 7) continue;
        if (p[i] > 100) { s = 100; break; }
        if (p[i] == 8) return 8; else s += 1;
        s += p[i];
    }
    return s;
}
"""
        # 后面四种都不是：continue、两条语句的块、带 else 的 if
        check_sites(scratch, src, "DGS", [
            ("if (p == 0) return -1;", "    if (p == 0) return -1;\n", ""),
            ("if (n < 0) {\n        return -2;\n    }", "    if (n < 0) {\n        return -2;\n    }\n", ""),
            ("if (p[i] < 0) break;", "        if (p[i] < 0) break;\n", "")])

    def test_dgs_only_takes_statements_directly_inside_a_block(self, scratch):
        src = """int f(int a, int b)
{
    if (a) {
        return 1;
    } else if (b) return 2;
    while (a) if (b) return 3;
    return 0;
}
"""
        check_sites(scratch, src, "DGS", [])            # else if 链里的 if、没有花括号的循环体里的 if 都删不干净

    def test_rvr_replaces_the_returned_value_by_type(self, scratch):
        src = """#include <stddef.h>
int f(int a)
{
    if (a > 0) return a + 1;
    return 0;
}
const char *g(int a)
{
    if (a) return "x";
    return NULL;
}
double d(int a)
{
    return a * 1.5;
}
"""
        check_sites(scratch, src, "RVR", [("return a + 1;", "return a + 1;", "return 0;")])        # return 0; 本来就是 0
        check_sites(scratch, src, "RVR", [('return "x";', 'return "x";', "return NULL;")], func="g", name="t2")
        check_sites(scratch, src, "RVR", [("return a * 1.5;", "return a * 1.5;", "return 0;")], func="d", name="t3")

    def test_rvr_skips_void_struct_and_zero_literals_and_falls_back_to_0_without_null(self, scratch):
        src = """struct S { int a; };
void v(int *p) { if (p) return; *p = 1; }
struct S s(void) { struct S x; x.a = 1; return x; }
int z(int a) { if (a) return (0); return 0u; }
char *h(char *p) { return p; }
"""
        spec, root = make_project(scratch, src)
        names = {t.name: t for t in extract.extract_file(spec, root, "t.c")}
        for name in ("v", "s", "z"):
            assert mutate.find_sites(spec, root, names[name], "RVR") == [], name
        (site,) = mutate.find_sites(spec, root, names["h"], "RVR")
        assert site.detail == "return 0;"               # 这份源码里没有 NULL 宏，指针返回值也只能写 0

    def test_dst_deletes_assignments_increments_and_calls(self, scratch):
        src = """void g(int);
void f(int *p, int n)
{
    int i = 0;
    int k;

    k = n;
    *p += k;
    i++;
    --n;
    g(n);
    (void)g;
    if (k) k = 1; else g(k);
    p[0] = 1; p[1] = 2;
    return;
}
"""
        check_sites(scratch, src, "DST", [
            ("k = n;", "    k = n;\n", ""), ("*p += k;", "    *p += k;\n", ""), ("i++;", "    i++;\n", ""),
            ("--n;", "    --n;\n", ""), ("g(n);", "    g(n);\n", ""),
            ("p[0] = 1;", "p[0] = 1; p[1]", "p[1]"),            # 同一行还有别的语句：只删这一句，不动整行
            ("p[1] = 2;", "p[0] = 1; p[1] = 2;", "p[0] = 1;")])

    def test_dst_keeps_a_trailing_comment_and_the_line(self, scratch):
        src = """void f(int *p)
{
    *p = 1;  /* 保留 */
    *p = 2;
}
"""
        check_sites(scratch, src, "DST", [("*p = 1;", "*p = 1;  /* 保留 */", "/* 保留 */"), ("*p = 2;", "    *p = 2;\n", "")])

    def test_operators_are_exact_tokens_even_without_spaces_and_never_come_from_strings_or_comments(self, scratch):
        src = """int puts(const char *);
int f(int a,int b,int c){/* a < b && c */puts("a < b && c");return a<b&&b>=c;}
"""
        spec, root = make_project(scratch, src)
        target = target_of(spec, root)
        by_op = {op: mutate.find_sites(spec, root, target, op) for op in ("ROR", "LOR")}
        assert [s.original for s in by_op["ROR"]] == ["<", ">="] and [s.original for s in by_op["LOR"]] == ["&&"]
        original = function_text(root, target)
        for site, (old, new) in zip(by_op["ROR"] + by_op["LOR"], [("a<b&&", "a<=b&&"), ("b>=c", "b>c"), ("b&&b", "b||b")]):
            new_file, new_func = mutate_site(spec, root, target, site)
            assert new_func == original.replace(old, new) and reparse_errors(spec, root, "t.c", new_file) == []

    def test_crlf_sources_are_handled(self, scratch):
        src = "int g(int);\r\nint f(int a)\r\n{\r\n    int s = 0;\r\n\r\n    if (a < 0) return -1;\r\n    s = g(a);\r\n    return s + 1;\r\n}\r\n"
        spec, root = make_project(scratch, "")
        (root / "t.c").write_bytes(src.encode())
        target = target_of(spec, root)
        original = function_text(root, target)
        dst = mutate.find_sites(spec, root, target, "DST")
        assert [s.original for s in dst] == ["s = g(a);"]
        assert mutate_site(spec, root, target, dst[0])[1] == original.replace("    s = g(a);\r\n", "")
        (dgs,) = mutate.find_sites(spec, root, target, "DGS")
        new_file, new_func = mutate_site(spec, root, target, dgs)
        assert new_func == original.replace("    if (a < 0) return -1;\r\n", "") and reparse_errors(spec, root, "t.c", new_file) == []

    def test_sites_are_numbered_per_operator_in_source_order(self, scratch):
        src = """int f(int a, int b)
{
    if (a < b && b < 3) return a + b;
    return a - b;
}
"""
        spec, root = make_project(scratch, src)
        target = target_of(spec, root)
        sites = mutate.find_sites(spec, root, target)
        assert sites == mutate.find_sites(spec, root, target)          # 只依赖源码，重复调用结果相同
        assert [(s.operator, s.index) for s in sites] == [
            ("ROR", 0), ("ROR", 1), ("NEG", 0), ("LOR", 0), ("OBO", 0), ("DGS", 0), ("AOR", 0), ("AOR", 1), ("RVR", 0),
            ("RVR", 1)]
        for op in mutate.OPERATORS:
            assert mutate.find_sites(spec, root, target, op) == [s for s in sites if s.operator == op]
        assert [s.start for s in sites if s.operator == "ROR"] == sorted(s.start for s in sites if s.operator == "ROR")
        assert mutate.count_sites(sites) == {"ROR": 2, "NEG": 1, "LOR": 1, "OBO": 1, "DGS": 1, "AOR": 2, "RVR": 2, "DST": 0}
        assert list(mutate.count_sites(sites)) == list(mutate.OPERATORS)

    def test_unknown_operator_is_rejected(self, scratch):
        spec, root = make_project(scratch, "int f(void) { return 1; }\n")
        with pytest.raises(ValueError):
            mutate.find_sites(spec, root, target_of(spec, root), "XXX")


class TestMacrosDirectivesAndFiles:
    MACROS = """#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define CHECK(c) do { if (!(c)) return -1; } while (0)
#define INC(x) ((x)++)
#define LIMIT 10
int f(int a, int b)
{
    int m = MAX(a, b);
    CHECK(a < b);
    INC(m);
    if (MAX(a, 3) > b) {
        m = 1;
    }
    if (LIMIT) {
        m = a - LIMIT;
    }
    return m;
}
"""

    def test_sites_inside_macro_arguments_or_generated_by_macros_are_ignored(self, scratch):
        spec, root = make_project(scratch, self.MACROS)
        target = target_of(spec, root)
        by_op = {op: [s.original for s in mutate.find_sites(spec, root, target, op)] for op in mutate.OPERATORS}
        assert by_op["ROR"] == [">"]                    # MAX 的宏体和 CHECK 的实参里的比较都不动
        assert by_op["OBO"] == []                       # MAX(a, 3) 的 3 在宏实参里
        assert by_op["DST"] == ["m = 1;", "m = a - LIMIT;"]     # INC(m); 是宏生成的语句；语句里含有整个宏调用（LIMIT）可以删
        assert by_op["NEG"] == ["MAX(a, 3) > b", "LIMIT"]       # 整个宏调用在条件里面（或条件恰好就是一个宏）可以取反
        assert by_op["AOR"] == ["-"]
        assert by_op["RVR"] == ["return m;"]
        # 位点区间里整个包着宏调用的改写能用
        data = (root / "t.c").read_bytes()
        neg = mutate.find_sites(spec, root, target, "NEG")[0]
        new = mutate.apply_site(data, neg, random.Random(0))
        assert b"if (!(MAX(a, 3) > b)) {" in new and reparse_errors(spec, root, "t.c", new) == []

    def test_sites_whose_range_holds_a_preprocessor_directive_are_ignored(self, scratch):
        src = """int f(int a)
{
    if (a > 0)
#ifdef FOO
        return 1;
#else
        return 2;
#endif
    a = a + 1;
    return a;
}
"""
        spec, root = make_project(scratch, src)
        target = target_of(spec, root)
        assert mutate.find_sites(spec, root, target, "DGS") == []
        assert [s.original for s in mutate.find_sites(spec, root, target, "DST")] == ["a = a + 1;"]

    def test_header_only_targets_work(self, scratch):
        src = """static int clamp(int x, int lo, int hi)
{
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}
"""
        spec, root = make_project(scratch, src, name="hdr", file="lib.h")
        target = target_of(spec, root, "clamp", "lib.h")
        sites = mutate.find_sites(spec, root, target)
        assert mutate.count_sites(sites)["ROR"] == 2 and mutate.count_sites(sites)["DGS"] == 2
        data = (root / "lib.h").read_bytes()
        for site in sites:
            assert reparse_errors(spec, root, "lib.h", mutate.apply_site(data, site, random.Random(1))) == []

    def test_stale_offsets_are_refused(self, scratch):
        spec, root = make_project(scratch, "int f(int a) { return a + 1; }\n")
        target = target_of(spec, root)
        (root / "t.c").write_text("\n\n" + (root / "t.c").read_text(), encoding="utf-8")        # 文件变了，偏移作废
        with pytest.raises(Exception):
            mutate.find_sites(spec, root, target)

    def test_apply_site_refuses_a_site_that_does_not_match_the_source(self, scratch):
        spec, root = make_project(scratch, "int f(int a) { return a < 1; }\n")
        site = mutate.find_sites(spec, root, target_of(spec, root), "ROR")[0]
        data = (root / "t.c").read_bytes()
        with pytest.raises(mutate.MutateError):
            mutate.apply_site(data.replace(b"<", b">"), site, random.Random(0))
        with pytest.raises(mutate.MutateError):
            mutate.apply_site(data[:5], site, random.Random(0))


# ----------------------------------------------------------------------
# 2. 排序、确定性与选择逻辑（假沙箱）
# ----------------------------------------------------------------------
GCC_ERROR = '[{"kind": "error", "message": "boom: injected", "locations": []}]'

RICH = """int f(int a, int b, int *p)
{
    int s = 0;

    if (a < b && b != a) {
        s = a + b;
    }
    *p = s;
    if (s > b) return b;
    return s * 2;
}
"""
# RICH 里各算子的位点数：ROR 3（< != >）、NEG 2、LOR 1、OBO 0、DGS 1、AOR 2、RVR 2、DST 2，共 13 个
RICH_COUNTS = {"ROR": 3, "NEG": 2, "LOR": 1, "OBO": 0, "DGS": 1, "AOR": 2, "RVR": 2, "DST": 2}


class ScriptedSandbox:
    """假沙箱：evaluate 的两次沙箱运行（单文件语法检查、构建 + 测试）都按候选内容给出预设结果，不需要 docker。

    ``rule(text)`` 吃替换之后目标文件的全文，返回 pass / fail / crash / timeout / syntax / build 之一
    （fail：测试失败；crash：段错误；timeout：测试超时；syntax：语法检查没过；build：构建没过）。
    ``filename`` 为 None 时不读文件，``rule`` 收到空串。"""

    def __init__(self, rule, filename: str | None = "t.c") -> None:
        self.rule = rule
        self.filename = filename
        self.syntax_runs = 0
        self.build_runs = 0
        self._lock = threading.Lock()

    def run(self, workdir, cmd, env=None, timeout_s=None) -> RunResult:
        text = (Path(workdir) / self.filename).read_text(encoding="utf-8") if self.filename else ""
        verdict = self.rule(text)
        ok = dict(rc=0, stdout="", stderr="[]", duration_s=0.0, timed_out=False, signal=None)
        if "-fsyntax-only" in cmd:
            with self._lock:
                self.syntax_runs += 1
            return RunResult(**{**ok, "rc": 1, "stderr": GCC_ERROR}) if verdict == "syntax" else RunResult(**ok)
        with self._lock:
            self.build_runs += 1
        table = {
            "build": dict(rc=2, stderr=GCC_ERROR),
            "fail": dict(rc=1, stdout="FAIL\n"),
            "crash": dict(rc=139, stdout="partial\n", signal=11),
            "timeout": dict(rc=124, stdout="partial\n", timed_out=True, duration_s=30.0),
        }
        return RunResult(**{**ok, **table.get(verdict, dict(stdout="ok\n"))})


def scripted_for(spec, root: Path, target, verdicts: dict[int, str], *, seed: int, tid: str,
                 prefer: str | None = None, default: str = "pass"):
    """按「第 pos 个试的位点」给出预设结果的假沙箱（只能用于没有 OBO 位点的函数）。返回 (沙箱, 试位点的顺序)。"""
    data = (root / target.file).read_bytes()
    order = mutate.order_sites(mutate.find_sites(spec, root, target), seed, tid, prefer)
    table = {mutate.apply_site(data, s, random.Random(0)).decode(): verdicts.get(pos, default) for pos, s in enumerate(order)}
    assert len(table) == len(order)                      # 每个位点改出来的文件各不相同
    return ScriptedSandbox(lambda text: table.get(text, "pass"), target.file), order


def fake_corpus_spec(name: str = "t", file: str = "t.c", **kw) -> corpus.CorpusSpec:
    return corpus.CorpusSpec(name=name, project_files=[file], target_files=[file], cflags=["-std=c99"],
                             build_cmd="true", **kw)


@pytest.fixture()
def rich(scratch):
    spec, root = make_project(scratch, RICH)
    return spec, root, target_of(spec, root), scratch / "work"


class TestOrdering:
    def sites(self, rich):
        spec, root, target, _ = rich
        return mutate.find_sites(spec, root, target)

    def test_rich_function_has_the_expected_sites(self, rich):
        assert mutate.count_sites(self.sites(rich)) == RICH_COUNTS

    def test_order_is_a_deterministic_shuffle_of_the_sites(self, rich):
        sites = self.sites(rich)
        a = mutate.order_sites(sites, SEED, "t:t.c:f")
        assert sorted(a, key=lambda s: (mutate.OPERATORS.index(s.operator), s.index)) == sites
        assert a != sites                                           # 洗过牌
        assert a == mutate.order_sites(sites, SEED, "t:t.c:f")
        assert a == mutate.order_sites(list(reversed(sites)), SEED, "t:t.c:f")       # 输入顺序不影响结果

    def test_order_depends_on_seed_and_target_id_but_there_is_no_level_in_it(self, rich):
        """种子里没有级别：L1 相对 L0 位点表一一对应，同一个目标在各级走同一条试探路线（评审 #1：带上级别时 L0 / L1
        只有 116 / 208 个目标选中同一个位点）。"""
        sites = self.sites(rich)
        base = mutate.order_sites(sites, SEED, "t:t.c:f")
        assert mutate.order_sites(sites, SEED + 1, "t:t.c:f") != base
        assert mutate.order_sites(sites, SEED, "t:t.c:g") != base
        assert "level" not in mutate.order_sites.__code__.co_varnames[:mutate.order_sites.__code__.co_argcount]

    def test_preferred_operator_goes_first_and_the_rest_keep_their_shuffled_order(self, rich):
        sites = self.sites(rich)
        base = mutate.order_sites(sites, SEED, "t:t.c:f")
        for op in mutate.OPERATORS:
            got = mutate.order_sites(sites, SEED, "t:t.c:f", op)
            assert got == [s for s in base if s.operator == op] + [s for s in base if s.operator != op]
        assert mutate.order_sites(sites, SEED, "t:t.c:f", "OBO") == base                # 没有这个算子的位点：不变

    def test_the_anchor_goes_before_everything_including_the_preferred_operator(self, rich):
        sites = self.sites(rich)
        anchor = next(s for s in sites if s.operator == "DST" and s.index == 1)
        base = mutate.order_sites(sites, SEED, "t:t.c:f", "ROR")
        got = mutate.order_sites(sites, SEED, "t:t.c:f", "ROR", anchor)
        assert got[0] is anchor and got[1:] == [s for s in base if s is not anchor]       # 其余保持（偏好算子 + 洗牌）的相对顺序
        assert mutate.order_sites(sites, SEED, "t:t.c:f", None, anchor)[0] is anchor

    def test_unknown_preferred_operator_is_rejected(self, rich):
        with pytest.raises(ValueError):
            mutate.order_sites(self.sites(rich), SEED, "t:t.c:f", "XXX")

    def test_operator_priority_is_the_documented_order(self):
        assert mutate.OPERATORS == ("ROR", "NEG", "LOR", "OBO", "DGS", "AOR", "RVR", "DST")
        assert set(MUTATION_TITLES) == set(mutate.OPERATORS)
        assert KILL_MODES == ("test_fail", "runtime_crash", "runtime_timeout") and mutate.DEFAULT_MAX_TRIES == 12


class TestChooseWithAScriptedSandbox:
    TID = "t:t.c:f"

    def search(self, rich, verdicts, *, level="L0", seed=SEED, prefer=None, max_tries=12, default="pass", recheck=False):
        """recheck 默认关：这组用例数的是沙箱运行次数，复核会让每个被杀死的候选多跑一次（复核另有专门的用例）。"""
        spec, root, target, work = rich
        sb, order = scripted_for(fake_spec(spec), root, target, verdicts, seed=seed, tid=self.TID, prefer=prefer,
                                 default=default)
        res = mutate.search_mutant(fake_spec(spec), root, target, level, seed, sb, prefer, max_tries,
                                   target_id=self.TID, work_root=work, recheck=recheck)
        return res, sb, order

    def test_the_search_stops_at_the_first_killed_site_and_counts_the_rest(self, rich):
        res, sb, order = self.search(rich, {0: "pass", 1: "syntax", 2: "build", 3: "pass", 4: "fail"})
        m = res.mutant
        assert res.outcome == "killed" and m is not None and res.killed
        assert (m.tried_sites, m.equivalent_count, m.compile_fail_count, m.parse_fail_count) == (5, 2, 2, 0)
        assert (res.tried_sites, res.equivalent_count, res.compile_fail_count) == (5, 2, 2)
        assert [t["stage"] for t in res.tries] == ["pass", "syntax", "build", "pass", "test"]
        assert [(t["operator"], t["site_index"]) for t in res.tries] == [(s.operator, s.index) for s in order[:5]]
        assert (m.operator, m.site_index, m.kill_mode) == (order[4].operator, order[4].index, "test_fail")
        assert (m.target_id, m.level, m.original_text) == (self.TID, "L0", order[4].original)
        assert m.site_detail == order[4].detail and m.kill_detail
        assert sb.syntax_runs == 5 and sb.build_runs == 4          # 语法检查没过的那个没有进构建

    @pytest.mark.parametrize("verdict,mode", [("fail", "test_fail"), ("crash", "runtime_crash"), ("timeout", "runtime_timeout")])
    def test_kill_mode_follows_the_attribution_category(self, rich, verdict, mode):
        res, _, _ = self.search(rich, {0: verdict})
        assert res.mutant is not None and res.mutant.kill_mode == mode and res.tried_sites == 1
        if mode == "runtime_timeout":                       # 超时的耗时每次不同，不能进记录
            assert "after the time limit" in res.mutant.kill_detail
            assert not re.search(r"\d+(\.\d+)?s\b", res.mutant.kill_detail)

    @pytest.mark.parametrize("pos", range(13))
    def test_mutant_record_describes_exactly_what_was_changed(self, rich, pos):
        """试位点的顺序里第 pos 个被杀死：记录必须逐项对得上那个位点（RICH 的 13 个位点各走一遍，八个算子都覆盖到）。"""
        spec, root, target, _ = rich
        res, _, order = self.search(rich, {pos: "fail"}, max_tries=13)
        m = res.mutant
        data = (root / target.file).read_bytes()
        site = order[pos]
        original = function_text(root, target)
        before, after = original[:site.start - target.start], original[site.end - target.start:]
        assert (m.operator, m.site_index) == (site.operator, site.index)
        assert m.original_text == data[site.start:site.end].decode() and m.original_text != m.mutated_text
        assert m.code != original and m.code.startswith("int f(int a, int b, int *p)")
        if site.operator in ("DGS", "DST"):                      # 删整句：mutated_text 是空串，原文不再出现，也没有多出空行
            assert m.mutated_text == "" and m.original_text not in m.code
            assert m.code.count("\n") < original.count("\n") and m.code.count("\n\n") <= original.count("\n\n")
        else:                                                     # 行内改写 / 整句替换：只有位点区间变了，行数不变
            assert m.mutated_text != "" and m.code == before + m.mutated_text + after
            assert m.code.count("\n") == original.count("\n")
        assert reparse_errors(spec, root, "t.c", data[:target.start] + m.code.encode() + data[target.end:]) == []

    def test_compile_failures_and_passing_mutants_are_never_kills(self, rich):
        for verdict in ("syntax", "build", "pass"):
            res, _, _ = self.search(rich, {}, default=verdict, max_tries=20)
            assert res.mutant is None and res.outcome == "exhausted" and res.tried_sites == 13
            counts = (res.equivalent_count, res.compile_fail_count)
            assert counts == ((13, 0) if verdict == "pass" else (0, 13))

    def test_max_tries_caps_the_number_of_sandbox_evaluations(self, rich):
        res, sb, _ = self.search(rich, {}, max_tries=3)
        assert res.mutant is None and res.outcome == "gave_up" and res.tried_sites == 3 == res.equivalent_count
        assert sb.syntax_runs == 3 and sb.build_runs == 3 and res.n_sites == 13
        res, sb, _ = self.search(rich, {4: "fail"}, max_tries=4)               # 第 5 个才能杀死，预算只有 4 个
        assert res.mutant is None and res.outcome == "gave_up"
        res, sb, _ = self.search(rich, {4: "fail"}, max_tries=5)
        assert res.mutant is not None and res.tried_sites == 5
        res, sb, _ = self.search(rich, {}, max_tries=0)
        assert res.mutant is None and res.outcome == "no_sites" and sb.syntax_runs == 0

    def test_default_budget_is_twelve(self, rich):
        res, sb, _ = self.search(rich, {})
        assert res.outcome == "gave_up" and sb.syntax_runs == 12 == mutate.DEFAULT_MAX_TRIES

    def test_a_function_without_sites_is_given_up_without_running_the_sandbox(self, scratch):
        spec, root = make_project(scratch, "void f(void)\n{\n}\n")
        target = target_of(spec, root)
        sb = ScriptedSandbox(lambda text: "fail")
        res = mutate.search_mutant(fake_spec(spec), root, target, "L0", SEED, sb, work_root=scratch / "work")
        assert res.mutant is None and res.outcome == "no_sites" and res.n_sites == 0 and sb.syntax_runs == 0
        assert mutate.choose_mutant(fake_spec(spec), root, target, "L0", SEED, sb, None, work_root=scratch / "work") is None

    def test_candidates_that_do_not_reparse_cleanly_are_skipped_without_using_the_budget(self, scratch):
        src = """int f(int x)
{
    switch (x) {
    case 0: return 1;
    case 1 + 1: return 2;
    }
    return 3;
}
"""
        spec, root = make_project(scratch, src)
        target = target_of(spec, root)
        sb = ScriptedSandbox(lambda text: "fail")
        res = mutate.search_mutant(fake_spec(spec), root, target, "L1", SEED, sb, "AOR", work_root=scratch / "work",
                                   recheck=False)
        # 先试 AOR：case 1 + 1 改成 case 1 - 1，和 case 0 重复，libclang 报 error，不进沙箱也不占预算；接着试别的，被杀死
        assert res.parse_fail_count == 1 and res.tried_sites == 1 and sb.syntax_runs == 1
        assert res.mutant.operator != "AOR" and res.mutant.parse_fail_count == 1
        assert res.mutant.equivalent_count == 0 == res.mutant.compile_fail_count
        assert [s.original for s in mutate.find_sites(spec, root, target, "AOR")] == ["+"]       # 位点本身是存在的

    def test_preferred_operator_is_tried_first_and_decides_the_choice(self, rich):
        for op, n in RICH_COUNTS.items():
            res, _, order = self.search(rich, {0: "fail"}, prefer=op, level="L2")
            assert res.prefer_operator == op
            if n:
                assert res.mutant.operator == op and res.tried_sites == 1
            else:                                                # 没有这个算子的位点：和不指定一样
                plain, _, _ = self.search(rich, {0: "fail"}, level="L2")
                assert (res.mutant.operator, res.mutant.site_index) == (plain.mutant.operator, plain.mutant.site_index)

    def test_preferred_operator_that_never_kills_falls_back_to_the_others(self, rich):
        spec, root, target, work = rich
        data = (root / target.file).read_bytes()
        killable = {mutate.apply_site(data, s, random.Random(0)).decode() for s in mutate.find_sites(spec, root, target)
                    if s.operator != "ROR"}
        sb = ScriptedSandbox(lambda text: "fail" if text in killable else "pass")
        res = mutate.search_mutant(fake_spec(spec), root, target, "L1", SEED, sb, "ROR", target_id=self.TID, work_root=work)
        assert res.mutant is not None and res.mutant.operator != "ROR"
        assert res.equivalent_count == 3 and res.tried_sites == 4          # 3 个 ROR 位点都是等价变异体

    def test_same_inputs_give_the_same_mutant_and_other_seeds_may_not(self, rich):
        a, _, _ = self.search(rich, {0: "fail"})
        b, _, _ = self.search(rich, {0: "fail"})
        assert a == b and a.mutant == b.mutant
        picked = {self.search(rich, {0: "fail"}, seed=SEED + i)[0].mutant.code for i in range(8)}
        assert len(picked) > 1

    def test_target_id_enters_the_rng_but_the_level_does_not(self, rich):
        spec, root, target, work = rich
        sb = ScriptedSandbox(lambda text: "fail")
        a = mutate.search_mutant(fake_spec(spec), root, target, "L0", SEED, sb, target_id="x:y:z", work_root=work)
        b = mutate.search_mutant(fake_spec(spec), root, target, "L0", SEED, sb, work_root=work)      # 用 target.target_id
        c = mutate.search_mutant(fake_spec(spec), root, target, "L0", SEED, sb, target_id=target.target_id, work_root=work)
        assert a.mutant.target_id == "x:y:z" and b.mutant.target_id == c.mutant.target_id == target.target_id
        assert b.mutant == c.mutant
        # 同一份副本换个级别名重来：试位点的路线一样，选出同一个位点（只有记录里的 level 不同）
        d = mutate.search_mutant(fake_spec(spec), root, target, "L2", SEED, sb, work_root=work)
        assert d.mutant.level == "L2" and dataclasses.replace(d.mutant, level="L0") == b.mutant

    def test_choose_mutant_returns_the_mutant_or_none(self, rich):
        spec, root, target, work = rich
        assert mutate.choose_mutant(fake_spec(spec), root, target, "L0", SEED, ScriptedSandbox(lambda t: "fail"), None,
                                    target_id=self.TID, work_root=work).kill_mode == "test_fail"
        assert mutate.choose_mutant(fake_spec(spec), root, target, "L0", SEED, ScriptedSandbox(lambda t: "pass"), None,
                                    max_tries=5, target_id=self.TID, work_root=work) is None

    def test_arguments_are_validated(self, rich):
        spec, root, target, work = rich
        with pytest.raises(ValueError):
            mutate.choose_mutant(fake_spec(spec), root, target, "L9", SEED, ScriptedSandbox(lambda t: "fail"), None)
        with pytest.raises(ValueError):
            mutate.choose_mutant(fake_spec(spec), root, target, "L0", SEED, None, None)

    def test_mutant_round_trips_through_its_record(self, rich):
        res, _, _ = self.search(rich, {0: "fail"})
        m = res.mutant
        rec = m.to_dict()
        assert rec["mutant_code"] == m.code and "code" not in rec
        assert {"target_id", "level", "operator", "site_index", "original_text", "mutated_text", "kill_mode", "tried_sites",
                "equivalent_count", "compile_fail_count"} <= set(rec)
        assert Mutant.from_dict(rec) == m
        assert Mutant.from_dict({**rec, "extra": 1}) == m

    def test_search_record_keeps_the_counts_of_a_given_up_pair(self, rich):
        res, _, _ = self.search(rich, {}, max_tries=4)
        rec = res.to_dict()
        assert rec["mutant"] is None and rec["outcome"] == "gave_up"
        assert (rec["n_sites"], rec["tried_sites"], rec["equivalent_count"], rec["compile_fail_count"]) == (13, 4, 4, 0)
        assert len(rec["tries"]) == 4 and rec["target_id"] == self.TID and rec["level"] == "L0"
        killed, _, _ = self.search(rich, {0: "fail"})
        assert killed.to_dict()["mutant"] == killed.mutant.to_dict()


def fake_spec(spec: corpus.CorpusSpec) -> corpus.CorpusSpec:
    """同一个项目换成「构建命令随便写」的 spec（假沙箱不会真的执行它）。"""
    return fake_corpus_spec(spec.name, spec.target_files[0])


# ----------------------------------------------------------------------
# 去注释（pipeline 的 mutate 阶段用：L1 以上注释里仍是项目原名，不能进修复提示词）
# ----------------------------------------------------------------------
COMMENTED = """int f(int a, int b)
{
    /* f: pick the larger one, f(a, b) */
    int x = a; // keep a
    const char *s = "/* not a comment */ // nor this";


    if (a < b) {
        x = b; /* tail */
    }
    (void)s;
    return x;
}
"""


def passes_only_when_nothing_but_comments_changed(source: str):
    """假沙箱的判定：文件忽略注释与空白后与 source 相同（去了注释的原函数）→ 通过，其余（变异体）→ 测试失败。"""
    base = re.sub(r"\s+", "", extract.strip_c_comments(source))
    return lambda text: "pass" if re.sub(r"\s+", "", extract.strip_c_comments(text)) == base else "fail"


class TestStripComments:
    def test_strip_function_comments_keeps_strings_and_collapses_blank_lines(self):
        out = mutate.strip_function_comments(COMMENTED)
        assert out == """int f(int a, int b)
{

    int x = a;
    const char *s = "/* not a comment */ // nor this";

    if (a < b) {
        x = b;
    }
    (void)s;
    return x;
}
"""
        assert mutate.strip_function_comments(out) == out                        # 幂等
        assert mutate.strip_function_comments("int g(void) { return 1; }") == "int g(void) { return 1; }"

    def test_stripped_function_still_parses_to_the_same_tokens(self, scratch):
        spec, root = make_project(scratch, COMMENTED)
        target = target_of(spec, root)
        stripped = mutate.strip_function_comments(function_text(root, target))
        spec2, root2 = make_project(scratch, stripped, name="t2")
        assert extract.strip_c_comments(function_text(root, target)).split() == stripped.split()
        assert target_of(spec2, root2).n_statements == target.n_statements

    def test_search_with_strip_comments_checks_and_records_the_stripped_text(self, scratch):
        spec, root = make_project(scratch, COMMENTED)
        target = target_of(spec, root)
        seen: list[str] = []

        verdict = passes_only_when_nothing_but_comments_changed(COMMENTED)

        def rule(text: str) -> str:
            seen.append(text)
            return verdict(text)

        sb = ScriptedSandbox(rule, "t.c")
        res = mutate.search_mutant(fake_spec(spec), root, target, "L0", SEED, sb, target_id="t:t.c:f",
                                   work_root=scratch / "work")                # 默认就去注释
        m = res.mutant
        literal = '"/* not a comment */ // nor this"'
        assert res.has_comments and res.strip_checked and res.outcome == "killed"        # 先验证去了注释的原函数，再搜索
        assert m is not None and literal in m.code                                        # 字符串字面量里的 /* 与 // 原样保留
        assert "/*" not in m.code.replace(literal, "") and "//" not in m.code.replace(literal, "")
        assert m.code == mutate.strip_function_comments(m.code)
        assert "pick the larger one" not in seen[-1] and "keep a" not in seen[-1]       # kill 检查验证的就是去了注释的文本
        assert m.original_text and m.operator in mutate.OPERATORS
        # 显式关掉才保留注释：注释原样留在 Mutant.code 里（给想看原貌的调用方；pipeline 的 mutate 阶段不会这样调用）
        sb2 = ScriptedSandbox(verdict, "t.c")
        res2 = mutate.search_mutant(fake_spec(spec), root, target, "L0", SEED, sb2, target_id="t:t.c:f",
                                    work_root=scratch / "work", strip_comments=False)
        assert res2.mutant is not None and "pick the larger one" in res2.mutant.code
        assert not res2.strip_checked and res2.has_comments is False          # 没开 strip_comments 就不做注释检查
        assert (res.mutant.operator, res.mutant.site_index) == (res2.mutant.operator, res2.mutant.site_index)

    def test_choose_for_levels_and_choose_mutant_pass_the_option_through(self, scratch):
        spec, root = make_project(scratch, COMMENTED)
        target = target_of(spec, root)
        sb = ScriptedSandbox(passes_only_when_nothing_but_comments_changed(COMMENTED), "t.c")
        m = mutate.choose_mutant(fake_spec(spec), root, target, "L0", SEED, sb, target_id="t:t.c:f",
                                 work_root=scratch / "work")
        assert m is not None and "pick the larger one" not in m.code
        out = mutate.choose_for_levels(fake_spec(spec), {"L0": (root, target)}, SEED, sb, target_id="t:t.c:f",
                                       levels=["L0"], work_root=scratch / "work")
        assert out["L0"].mutant is not None and "pick the larger one" not in out["L0"].mutant.code
        kept = mutate.choose_mutant(fake_spec(spec), root, target, "L0", SEED, sb, target_id="t:t.c:f",
                                    work_root=scratch / "work", strip_comments=False)
        assert kept is not None and "pick the larger one" in kept.code


# ----------------------------------------------------------------------
# mini 夹具的四级副本（不跑 verify，不需要 docker）
# ----------------------------------------------------------------------
class MiniLevels:
    """mini 夹具的 L0–L3 副本，以及每个目标在各级副本里按映射名重新定位的结果。"""

    def __init__(self, base: Path) -> None:
        # 超时改短：死循环型的变异体（NEG 取反循环条件等）每个要等满超时，60 秒的上限会让整组测试白等；mini 的测试不到 1 秒
        self.spec = dataclasses.replace(corpus.load_spec_file(MINI / "spec.json"), timeout_s=20)
        self.ws = Workspace.under(base).ensure()
        corpus.materialize(MINI, self.ws.pristine("mini"))
        self.maps: dict[str, dict[str, str]] = {}
        for lv in LEVELS:
            res = obfuscate(self.spec, self.ws.pristine("mini"), lv, SEED, self.ws.obf_root("mini", lv), verify=False,
                            results_dir=self.ws.results)
            self.maps[lv] = res.rename_map
        # 与 TargetFilter 的静态过滤一致：语句数 >= 3、4 <= 行数 <= 120（stub-kill 另说，见 mini_search）
        self.targets = [t for t in extract.extract_functions(self.spec, self.ws.pristine("mini"))
                        if t.n_statements >= 3 and 4 <= t.n_lines <= 120]
        self.trees = {t.target_id: {lv: self.tree(t, lv) for lv in LEVELS} for t in self.targets}

    def tree(self, target, level: str):
        root = self.ws.obf_root("mini", level)
        located = extract.locate(self.spec, root, self.maps[level].get(target.name, target.name), target.file)
        assert located is not None, (target.target_id, level)
        return root, located

    def by_name(self, name: str):
        return next(t for t in self.targets if t.name == name)


@pytest.fixture(scope="module")
def mini_levels(scratch_module):
    return MiniLevels(scratch_module)


class TestAcrossLevels:
    def test_every_level_gives_sites_for_every_target(self, mini_levels):
        for t in mini_levels.targets:
            for lv in LEVELS:
                root, located = mini_levels.trees[t.target_id][lv]
                assert sum(mutate.count_sites(mutate.find_sites(mini_levels.spec, root, located)).values()) > 0, (t.name, lv)

    def test_l0_operator_is_reused_on_the_other_levels(self, mini_levels):
        ml = mini_levels
        sb = ScriptedSandbox(lambda text: "fail", filename=None)
        reused = 0
        for t in ml.targets[::2]:
            out = mutate.choose_for_levels(ml.spec, ml.trees[t.target_id], SEED, sb, target_id=t.target_id,
                                           work_root=ml.ws.work)
            assert list(out) == list(LEVELS)
            op0 = out["L0"].mutant.operator
            assert out["L0"].prefer_operator is None and out["L0"].mutant.same_operator_as_L0 is True
            for lv in ("L1", "L2", "L3"):
                res = out[lv]
                root, located = ml.trees[t.target_id][lv]
                here = {s.operator for s in mutate.find_sites(ml.spec, root, located)}
                assert res.prefer_operator == op0 and res.mutant.target_id == t.target_id and res.mutant.level == lv
                if op0 in here:                          # 这一级还有这个算子的位点：必须复用（假沙箱里第一个位点就杀死）
                    assert res.mutant.operator == op0 and res.mutant.same_operator_as_L0 is True
                    reused += 1
                else:
                    assert res.mutant.same_operator_as_L0 is False
        assert reused > 0

    def test_without_the_preference_the_operator_often_differs(self, mini_levels):
        """对照：不传 prefer_operator 时四级的算子并不总是一致，说明上一条测的确实是复用机制。"""
        ml = mini_levels
        sb = ScriptedSandbox(lambda text: "fail", filename=None)
        differing = 0
        for t in ml.targets[::2]:
            ops = set()
            for lv in LEVELS:
                root, located = ml.trees[t.target_id][lv]
                res = mutate.search_mutant(ml.spec, root, located, lv, SEED, sb, None, target_id=t.target_id,
                                           work_root=ml.ws.work)
                ops.add(res.mutant.operator)
            differing += len(ops) > 1
        assert differing > 0

    def test_a_reference_operator_stands_in_for_l0(self, mini_levels):
        ml = mini_levels
        t = ml.by_name("calc_median3")
        sb = ScriptedSandbox(lambda text: "fail", filename=None)
        out = mutate.choose_for_levels(ml.spec, ml.trees[t.target_id], SEED, sb, target_id=t.target_id,
                                       levels=["L2", "L3"], reference_operator="DST", work_root=ml.ws.work)
        assert list(out) == ["L2", "L3"]
        assert all(r.mutant.operator == "DST" and r.mutant.same_operator_as_L0 is True for r in out.values())

    def test_when_l0_has_no_killed_mutant_nothing_is_preferred(self, mini_levels):
        ml = mini_levels
        t = ml.by_name("calc_gcd")
        # calc.c 里还有原名 calc_apply 的只有 L0：L0 的变异体全是等价的，其余级别的第一个位点就被杀死
        sb = ScriptedSandbox(lambda text: "pass" if "calc_apply" in text else "fail", filename="calc.c")
        out = mutate.choose_for_levels(ml.spec, ml.trees[t.target_id], SEED, sb, target_id=t.target_id, max_tries=3,
                                       work_root=ml.ws.work)
        assert out["L0"].mutant is None and out["L0"].outcome == "gave_up"
        for lv in ("L1", "L2", "L3"):
            assert out[lv].prefer_operator is None and out[lv].mutant is not None
            assert out[lv].mutant.same_operator_as_L0 is None

    def test_concurrent_calls_give_the_same_answers_as_sequential_ones(self, mini_levels):
        ml = mini_levels
        sb = ScriptedSandbox(lambda text: "fail", filename=None)
        targets = ml.targets[:8]

        def run(t):
            return mutate.choose_for_levels(ml.spec, ml.trees[t.target_id], SEED, sb, target_id=t.target_id,
                                            work_root=ml.ws.work)

        sequential = [run(t) for t in targets]
        with ThreadPoolExecutor(8) as pool:
            parallel = list(pool.map(run, targets))
        assert parallel == sequential

    def test_sites_of_obfuscated_levels_stay_valid_after_reparse(self, mini_levels):
        """L1–L3 副本里的每个位点改出来的文件，重新解析都不比原文件多出 error（mini 里没有指针相减，应当全部有效）。"""
        ml = mini_levels
        for lv in LEVELS:
            for t in ml.targets[::3]:
                root, located = ml.trees[t.target_id][lv]
                data = (root / located.file).read_bytes()
                for site in mutate.find_sites(ml.spec, root, located):
                    new = mutate.apply_site(data, site, random.Random(7))
                    assert new != data
                    assert reparse_errors(ml.spec, root.resolve(), located.file, new) == [], (t.name, lv, site)


# ----------------------------------------------------------------------
# 位点表加记号表（analyze），以及跨级别的「同一个位点」（match_reference）
# ----------------------------------------------------------------------
SRC_L0 = """int f(int *a, int n)
{
    int s = 0;

    if (a[1] < n) {
        s = a[2] + 3;
    } else {
        s = n * 5;
    }
    return s;
}
"""


def renamed(src: str, table: dict[str, str]) -> str:
    """整词换名（模拟 L1 的符号混淆：只改标识符）。"""
    return re.sub(r"\b[A-Za-z_]\w*\b", lambda m: table.get(m.group(), m.group()), src)


def analysis_of(scratch: Path, src: str, name: str, func: str = "f"):
    spec, root = make_project(scratch, src, name=name)
    target = target_of(spec, root, func)
    return spec, root, target, mutate.analyze(spec, root, target)


class TestAnalysis:
    def test_analyze_gives_the_same_sites_as_find_sites_plus_normalized_tokens(self, scratch):
        spec, root, target, ana = analysis_of(scratch, SRC_L0, "a0")
        assert ana.sites == mutate.find_sites(spec, root, target)
        assert ana.tokens[:6] == ["int", "ID", "(", "int", "*", "ID"]                 # 标识符一律是 ID，关键字和标点原样
        assert len(ana.tokens) == len(ana.spans) and all(s < e for s, e in ana.spans)
        assert "a" not in ana.tokens and "n" not in ana.tokens

    def test_token_range_covers_exactly_the_tokens_inside_a_site(self, scratch):
        spec, root, target, ana = analysis_of(scratch, SRC_L0, "a1")
        ror = next(s for s in ana.sites if s.operator == "ROR")
        lo, hi = ana.token_range(ror.start, ror.end)
        assert (hi - lo, ana.tokens[lo]) == (1, "<")                                    # 运算符本身就是一个记号
        dst = next(s for s in ana.sites if s.operator == "DST" and s.original.startswith("s = a[2]"))
        lo, hi = ana.token_range(dst.start, dst.end)
        assert ana.tokens[lo:hi] == ["ID", "=", "ID", "[", "2", "]", "+", "3", ";"]
        assert ana.token_range(0, 0) is None

    def test_a_pure_renaming_keeps_the_token_table_and_the_site_table_aligned(self, scratch):
        names = {"f": "f_1", "a": "v_2", "n": "v_3", "s": "v_4"}
        _, _, _, l0 = analysis_of(scratch, SRC_L0, "b0")
        _, _, _, l1 = analysis_of(scratch, renamed(SRC_L0, names), "b1", "f_1")
        assert l0.tokens == l1.tokens
        assert [(s.operator, s.index) for s in l0.sites] == [(s.operator, s.index) for s in l1.sites]


class TestMatchReference:
    def test_every_site_finds_itself_after_a_pure_renaming(self, scratch):
        """L1：记号序列完全相同，每个位点都能对上自己（算子与序号都相同）。"""
        names = {"f": "f_1", "a": "v_2", "n": "v_3", "s": "v_4"}
        _, _, _, l0 = analysis_of(scratch, SRC_L0, "c0")
        _, _, _, l1 = analysis_of(scratch, renamed(SRC_L0, names), "c1", "f_1")
        assert len(l0.sites) >= 8
        for site in l0.sites:
            ref = mutate.reference_for(l0, site.operator, site.index)
            got = mutate.match_reference(ref, l1)
            assert got is not None and (got.operator, got.index) == (site.operator, site.index), site

    def test_unknown_reference_sites_give_none(self, scratch):
        _, _, _, l0 = analysis_of(scratch, SRC_L0, "d0")
        assert mutate.reference_for(l0, "LOR", 0) is None and mutate.reference_for(l0, "ROR", 99) is None

    def test_sites_survive_code_inserted_before_them_even_though_their_index_shifts(self, scratch):
        """L2 / L3 在前面加了带同类位点的代码：序号会漂，记号对齐照样对得上同一处代码。"""
        extra = "    if (a[7] > n) { s = 4; }\n"
        l2_src = SRC_L0.replace("    if (a[1] < n)", extra + "    if (a[1] < n)")
        _, _, _, l0 = analysis_of(scratch, SRC_L0, "e0")
        _, _, _, l2 = analysis_of(scratch, l2_src, "e2")
        shifted = 0
        for site in l0.sites:
            ref = mutate.reference_for(l0, site.operator, site.index)
            got = mutate.match_reference(ref, l2)
            if site.operator == "OBO" and site.original == "1":
                assert got is not None and got.original == "1" and got.index > site.index          # 序号漂了，位置没认错
                shifted += 1
            elif got is not None:
                assert got.original == site.original
        assert shifted == 1

    def test_a_site_whose_tokens_were_rewritten_has_no_image(self, scratch):
        l3_src = SRC_L0.replace("a[1] < n", "n > a[1]")                                    # L3 的 a < b -> b > a
        _, _, _, l0 = analysis_of(scratch, SRC_L0, "f0")
        _, _, _, l3 = analysis_of(scratch, l3_src, "f3")
        ror = next(s for s in l0.sites if s.operator == "ROR")
        assert mutate.match_reference(mutate.reference_for(l0, "ROR", ror.index), l3) is None

    def test_a_lone_operator_is_not_matched_to_a_lookalike_with_different_neighbours(self, scratch):
        """单个运算符的位点靠前后各一个记号确认；别处恰好有同样的 ``<`` 但上下文对不上，不能认成同一个位点。"""
        l0_src = "int f(int a, int b)\n{\n    return a < b;\n}\n"
        l3_src = "int f(int a, int b)\n{\n    return b > a;\n}\n"
        l3_with_lookalike = "int f(int a, int b)\n{\n    if (b > a) return a < 7;\n    return 0;\n}\n"
        _, _, _, l0 = analysis_of(scratch, l0_src, "g0")
        ref = mutate.reference_for(l0, "ROR", 0)
        for name, src in (("g3", l3_src), ("g4", l3_with_lookalike)):
            _, _, _, other = analysis_of(scratch, src, name)
            assert mutate.match_reference(ref, other) is None, src

    def test_a_negated_condition_is_not_the_same_site_as_the_plain_condition(self, scratch):
        """L2 的 if / else 交换把条件改成 ``!(c)``：NEG 位点的范围变大，对不上（保守地算「不是同一个位点」）。"""
        l2_src = SRC_L0.replace("if (a[1] < n) {\n        s = a[2] + 3;\n    } else {\n        s = n * 5;\n    }",
                                "if (!(a[1] < n)) {\n        s = n * 5;\n    } else {\n        s = a[2] + 3;\n    }")
        assert l2_src != SRC_L0
        _, _, _, l0 = analysis_of(scratch, SRC_L0, "h0")
        _, _, _, l2 = analysis_of(scratch, l2_src, "h2")
        neg = next(s for s in l0.sites if s.operator == "NEG")
        assert mutate.match_reference(mutate.reference_for(l0, "NEG", neg.index), l2) is None
        ror = next(s for s in l0.sites if s.operator == "ROR")                          # 条件里的 < 没动，照样对得上
        assert mutate.match_reference(mutate.reference_for(l0, "ROR", ror.index), l2) is not None


def normalized_tokens_equal(a: str, b: str) -> bool:
    from bench.similarity import normalized_tokens
    return normalized_tokens(a) == normalized_tokens(b)


def two_level_trees(scratch: Path, src: str, names: dict[str, str], l1_func: str):
    """L0 与「只改了名字」的 L1 两棵树（单文件小项目），返回 (spec, trees)；L1 的目标按改名后的函数名取。"""
    spec0, root0 = make_project(scratch, src, name="x0")
    spec1, root1 = make_project(scratch, renamed(src, names), name="x1")
    return fake_spec(spec0), {"L0": (root0, target_of(spec0, root0)), "L1": (root1, target_of(spec1, root1, l1_func))}


OBO_SRC = """int f(int *a, int n)
{
    int i = 0;

    while (i < 4 && a[i + 2] != 7) {
        if (a[1] > 3 || n == 5) {
            i = i + 1;
        }
        i++;
    }
    return a[3] + i;
}
"""


class TestAnchorAcrossLevels:
    NAMES = {"f": "f_1", "a": "v_2", "n": "v_3", "i": "v_4"}

    def test_l1_picks_exactly_the_site_l0_picked_whatever_the_seed(self, scratch):
        spec, trees = two_level_trees(scratch, OBO_SRC, self.NAMES, "f_1")
        sb = ScriptedSandbox(lambda text: "fail", filename=None)
        for seed in range(30):
            out = mutate.choose_for_levels(spec, trees, seed, sb, target_id="x:t.c:f", levels=["L0", "L1"],
                                           work_root=scratch / "work", recheck=False)
            m0, m1 = out["L0"].mutant, out["L1"].mutant
            assert (m1.operator, m1.site_index) == (m0.operator, m0.site_index), seed
            assert m1.same_site_as_L0 is True and m0.same_site_as_L0 is True and out["L1"].anchor == "chosen"
            assert normalized_tokens_equal(m0.code, m1.code), seed                  # 改名之外一个记号都不差：同一个缺陷

    def test_the_obo_direction_follows_the_l0_index_when_the_index_shifts(self, scratch):
        """L2 / L3 里同一个位点的序号会漂：OBO 的 ±1 要按 L0 的序号决定，不然同一个缺陷在各级改成相反方向。"""
        l2_src = OBO_SRC.replace("    int i = 0;\n", "    int i = 0;\n\n    if (a[8] == 6) { i = 9; }\n")
        spec0, root0 = make_project(scratch, OBO_SRC, name="y0")
        spec2, root2 = make_project(scratch, l2_src, name="y2")
        trees = {"L0": (root0, target_of(spec0, root0)), "L2": (root2, target_of(spec2, root2))}
        sb = ScriptedSandbox(lambda text: "fail", filename=None)
        seen = 0
        for seed in range(60):
            out = mutate.choose_for_levels(fake_spec(spec0), trees, seed, sb, target_id="x:t.c:f", levels=["L0", "L2"],
                                           work_root=scratch / "work", recheck=False)
            m0, m2 = out["L0"].mutant, out["L2"].mutant
            if m0.operator != "OBO":
                continue
            seen += 1
            assert m2.same_site_as_L0 is True and m2.original_text == m0.original_text
            assert m2.site_index > m0.site_index                                       # 序号漂了
            assert m2.mutated_text == m0.mutated_text, (seed, m0.mutated_text, m2.mutated_text)
        assert seen >= 5

    def test_a_missing_image_falls_back_to_the_operator_of_l0(self, scratch):
        spec0, root0 = make_project(scratch, "int f(int a, int b)\n{\n    a = b + 1;\n    return a < b;\n}\n", name="z0")
        spec1, root1 = make_project(scratch, "int f(int a, int b)\n{\n    a = b + 1;\n    return b > a;\n}\n", name="z1")
        trees = {"L0": (root0, target_of(spec0, root0)), "L3": (root1, target_of(spec1, root1))}
        sb = ScriptedSandbox(lambda text: "fail", filename=None)
        seen_ror = 0
        for seed in range(12):
            out = mutate.choose_for_levels(fake_spec(spec0), trees, seed, sb, target_id="x:t.c:f", levels=["L0", "L3"],
                                           work_root=scratch / "work", recheck=False)
            m0, m3 = out["L0"].mutant, out["L3"].mutant
            assert m3.operator == m0.operator and m3.same_operator_as_L0 is True
            if m0.operator == "ROR":        # L3 把 a < b 改成了 b > a：ROR 位点的记号没了，对不上，退回算子级复用
                seen_ror += 1
                assert out["L3"].anchor == "no_image" and m3.same_site_as_L0 is False
        assert seen_ror >= 1

    def test_the_anchor_that_is_not_killed_is_tried_first_and_then_the_operator(self, scratch):
        spec, trees = two_level_trees(scratch, OBO_SRC, self.NAMES, "f_1")
        sb0 = ScriptedSandbox(lambda text: "fail", filename=None)
        l0 = mutate.choose_for_levels(spec, {"L0": trees["L0"]}, 5, sb0, target_id="x:t.c:f", levels=["L0"],
                                      work_root=scratch / "work", recheck=False)["L0"]
        ref = mutate.reference_for(l0.analysis, l0.mutant.operator, l0.mutant.site_index)
        root1, target1 = trees["L1"]
        data = (root1 / target1.file).read_bytes()
        anchor_text = None
        # 对应位点改出来的文件测试仍通过（等价变异体），其余都被杀死
        for site in mutate.find_sites(spec, root1, target1, ref.operator):
            if site.index == ref.index:
                anchor_text = mutate.apply_site(data, site, random.Random(0)).decode()
        sb1 = ScriptedSandbox(lambda text: "pass" if text == anchor_text else "fail", target1.file)
        res = mutate.search_mutant(spec, root1, target1, "L1", 5, sb1, None, target_id="x:t.c:f", work_root=scratch / "work",
                                   recheck=False, reference=ref)
        assert res.anchor == "tried" and res.mutant is not None and res.equivalent_count == 1
        assert res.mutant.same_site_as_L0 is False and res.mutant.operator == ref.operator          # 退到同一个算子的别的位点
        assert res.tries[0] == {"operator": ref.operator, "site_index": ref.index, "stage": "pass", "category": "pass"}

    def test_without_a_reference_nothing_is_claimed_about_the_site(self, scratch):
        spec, trees = two_level_trees(scratch, OBO_SRC, self.NAMES, "f_1")
        sb = ScriptedSandbox(lambda text: "fail", filename=None)
        root1, target1 = trees["L1"]
        res = mutate.search_mutant(spec, root1, target1, "L1", 3, sb, None, target_id="x:t.c:f", work_root=scratch / "work")
        assert res.anchor == "none" and res.mutant.same_site_as_L0 is None
        out = mutate.choose_for_levels(spec, trees, 3, sb, target_id="x:t.c:f", levels=["L1"], work_root=scratch / "work")
        assert out["L1"].mutant.same_site_as_L0 is None and out["L1"].mutant.same_operator_as_L0 is None


# ----------------------------------------------------------------------
# 复核：判为杀死的候选再评测一次
# ----------------------------------------------------------------------
class FlakySandbox(ScriptedSandbox):
    """每个候选文本按评测次序给结果：``plan[text]`` 是 [第一次评测的结果, 复核的结果, ...]；没列出的文本一律 ``fail``。
    evaluate 对一个候选会调两次 run（语法检查、构建 + 测试），所以按文本累计 run 次数再除以 2 得到第几次评测。"""

    def __init__(self, plan: dict[str, list[str]], filename: str = "t.c") -> None:
        self.plan = plan
        self.runs_by_text: Counter = Counter()
        super().__init__(self.verdict, filename)

    def verdict(self, text: str) -> str:
        with self._lock:
            n = self.runs_by_text[text]
            self.runs_by_text[text] += 1
        seq = self.plan.get(text)
        return "fail" if seq is None else seq[min(n // 2, len(seq) - 1)]


class TestRecheck:
    TID = "t:t.c:f"

    def first_sites(self, rich):
        spec, root, target, work = rich
        data = (root / target.file).read_bytes()
        order = mutate.order_sites(mutate.find_sites(spec, root, target), SEED, self.TID)
        texts = [mutate.apply_site(data, s, random.Random(0)).decode() for s in order[:3]]
        return spec, root, target, work, order, texts

    def test_a_stable_kill_is_evaluated_twice_and_records_the_second_category(self, rich):
        spec, root, target, work, order, texts = self.first_sites(rich)
        sb = FlakySandbox({texts[0]: ["fail", "fail"]})
        res = mutate.search_mutant(fake_spec(spec), root, target, "L0", SEED, sb, None, target_id=self.TID, work_root=work)
        m = res.mutant
        assert m is not None and (m.operator, m.site_index) == (order[0].operator, order[0].index)
        assert (m.kill_mode, m.kill_recheck, m.unstable_count, res.rechecked) == ("test_fail", "test_fail", 0, 1)
        assert res.tries[0]["recheck"] == "test_fail" and sb.syntax_runs == 2 and sb.build_runs == 2

    def test_a_candidate_that_passes_on_the_recheck_is_not_a_kill_and_the_search_goes_on(self, rich):
        spec, root, target, work, order, texts = self.first_sites(rich)
        sb = FlakySandbox({texts[0]: ["fail", "pass"]})
        res = mutate.search_mutant(fake_spec(spec), root, target, "L0", SEED, sb, None, target_id=self.TID, work_root=work)
        m = res.mutant
        assert m is not None and (m.operator, m.site_index) == (order[1].operator, order[1].index)
        assert (res.unstable_count, res.equivalent_count, res.compile_fail_count, res.tried_sites) == (1, 0, 0, 2)
        assert m.unstable_count == 1 and m.tried_sites == 2 == m.equivalent_count + m.compile_fail_count + m.unstable_count + 1
        assert res.tries[0] == {"operator": order[0].operator, "site_index": order[0].index, "stage": "test",
                                "category": "test_fail", "recheck": "pass:pass"}

    def test_recheck_failing_differently_keeps_both_labels(self, rich):
        spec, root, target, work, order, texts = self.first_sites(rich)
        sb = FlakySandbox({texts[0]: ["timeout", "crash"]})
        res = mutate.search_mutant(fake_spec(spec), root, target, "L0", SEED, sb, None, target_id=self.TID, work_root=work)
        assert res.mutant.kill_mode == "runtime_timeout" and res.mutant.kill_recheck == "runtime_crash"
        assert res.mutant.unstable_count == 0
        assert "after the time limit" in res.mutant.kill_detail

    def test_the_unstable_candidates_use_up_the_budget(self, rich):
        spec, root, target, work, order, texts = self.first_sites(rich)
        sb = FlakySandbox({texts[0]: ["fail", "pass"], texts[1]: ["fail", "pass"]})
        res = mutate.search_mutant(fake_spec(spec), root, target, "L0", SEED, sb, None, 2, target_id=self.TID, work_root=work)
        assert res.mutant is None and res.outcome == "gave_up" and res.unstable_count == 2 and res.tried_sites == 2

    def test_recheck_can_be_turned_off(self, rich):
        spec, root, target, work, order, texts = self.first_sites(rich)
        sb = FlakySandbox({texts[0]: ["fail", "pass"]})
        res = mutate.search_mutant(fake_spec(spec), root, target, "L0", SEED, sb, None, target_id=self.TID, work_root=work,
                                   recheck=False)
        m = res.mutant
        assert (m.operator, m.site_index) == (order[0].operator, order[0].index) and m.kill_recheck is None
        assert res.rechecked == 0 and sb.build_runs == 1

    def test_non_kills_are_never_rechecked(self, rich):
        spec, root, target, work, order, texts = self.first_sites(rich)
        sb = FlakySandbox({texts[0]: ["pass"], texts[1]: ["build"], texts[2]: ["fail", "fail"]})
        res = mutate.search_mutant(fake_spec(spec), root, target, "L0", SEED, sb, None, target_id=self.TID, work_root=work)
        assert res.rechecked == 1 and res.equivalent_count == 1 and res.compile_fail_count == 1 and res.unstable_count == 0


# ----------------------------------------------------------------------
# 去注释的原函数必须先通过项目测试（strip_unsafe），失败的（目标, 级别）不搜索也不当参照
# ----------------------------------------------------------------------
class TestStripUnsafe:
    def test_the_stripped_original_that_fails_gives_up_the_pair_before_any_search(self, scratch):
        spec, root = make_project(scratch, COMMENTED)
        target = target_of(spec, root)
        sb = ScriptedSandbox(lambda text: "fail", "t.c")                # 去注释的原函数也没通过
        res = mutate.search_mutant(fake_spec(spec), root, target, "L0", SEED, sb, target_id="t:t.c:f", work_root=scratch / "work")
        assert res.mutant is None and res.outcome == "strip_unsafe" and res.has_comments and res.strip_checked
        assert res.tried_sites == 0 and sb.build_runs == 1 and res.n_sites > 0

    def test_a_function_without_comments_is_not_checked(self, rich):
        spec, root, target, work = rich
        sb = ScriptedSandbox(lambda text: "fail")
        res = mutate.search_mutant(fake_spec(spec), root, target, "L0", SEED, sb, target_id="t:t.c:f", work_root=work)
        assert res.outcome == "killed" and not res.has_comments and not res.strip_checked

    def test_l0_strip_unsafe_means_no_reference_for_the_other_levels(self, scratch):
        """L0 的去注释验证没过：L1 以上没有参照，不偏好算子、不找同一个位点（整体运行与只重做 L1 的结果一致，评审 #8）。"""
        names = {"f": "f_1", "a": "v_2", "b": "v_3", "x": "v_4"}
        spec0, root0 = make_project(scratch, COMMENTED, name="s0")
        spec1, root1 = make_project(scratch, renamed(COMMENTED, names).replace("keep v_2", "keep"), name="s1")
        trees = {"L0": (root0, target_of(spec0, root0)), "L1": (root1, target_of(spec1, root1, "f_1"))}

        def rule(text: str) -> str:
            # L0 的去注释原函数不通过；L1 的去注释原函数通过（f_1 只出现在 L1 的树里），L1 的变异体失败
            if "f_1" not in text:
                return "fail"
            return passes_only_when_nothing_but_comments_changed(renamed(COMMENTED, names))(text)

        out = mutate.choose_for_levels(fake_spec(spec0), trees, SEED, ScriptedSandbox(rule, "t.c"), target_id="t:t.c:f",
                                       levels=["L0", "L1"], work_root=scratch / "work")
        assert out["L0"].outcome == "strip_unsafe"
        assert out["L1"].prefer_operator is None and out["L1"].anchor == "none"
        assert out["L1"].mutant is not None and out["L1"].mutant.same_operator_as_L0 is None
        assert out["L1"].mutant.same_site_as_L0 is None


class TestWholeRunAndSplitRunAgree:
    def test_the_later_levels_get_the_same_mutants_whether_or_not_l0_was_run_in_the_same_call(self, mini_levels):
        ml = mini_levels
        sb = ScriptedSandbox(lambda text: "fail", filename=None)
        for t in ml.targets[:10]:
            trees = ml.trees[t.target_id]
            whole = mutate.choose_for_levels(ml.spec, trees, SEED, sb, target_id=t.target_id, work_root=ml.ws.work, recheck=False)
            l0 = whole["L0"]
            ref = mutate.reference_for(l0.analysis, l0.mutant.operator, l0.mutant.site_index)
            rest = {lv: trees[lv] for lv in ("L1", "L2", "L3")}
            split = mutate.choose_for_levels(ml.spec, rest, SEED, sb, target_id=t.target_id, levels=["L1", "L2", "L3"],
                                             reference=ref, work_root=ml.ws.work, recheck=False)
            assert {lv: whole[lv].mutant for lv in rest} == {lv: split[lv].mutant for lv in rest}, t.name


class TestAcrossLevelsSameSite:
    def test_l1_is_the_same_site_on_every_target_and_l2_l3_mostly(self, mini_levels):
        ml = mini_levels
        sb = ScriptedSandbox(lambda text: "fail", filename=None)
        n = same = 0
        for t in ml.targets:
            out = mutate.choose_for_levels(ml.spec, ml.trees[t.target_id], SEED, sb, target_id=t.target_id,
                                           work_root=ml.ws.work, recheck=False)
            m0, m1 = out["L0"].mutant, out["L1"].mutant
            assert (m1.operator, m1.site_index, m1.same_site_as_L0) == (m0.operator, m0.site_index, True), t.name
            assert normalized_tokens_equal(m0.code, m1.code), t.name                 # 改名之外一个记号都不差：同一个缺陷
            for lv in ("L2", "L3"):
                m = out[lv].mutant
                n += 1
                same += bool(m.same_site_as_L0)
                assert m.same_site_as_L0 in (True, False) and m.same_operator_as_L0 is not None
                assert (out[lv].anchor == "chosen") == bool(m.same_site_as_L0)
        assert same >= 0.6 * n                                                      # 实测 mini 上约八成对得上，下限放宽


# ----------------------------------------------------------------------
# 3. 真沙箱
# ----------------------------------------------------------------------
LIB_H = """#ifndef LIB_H
#define LIB_H
int classify(int x);
void note(int x);
extern int g_noted;
#endif
"""
EQV_C = """#include "lib.h"

int g_noted = 0;

int classify(int x)
{
    int r = x + 1;

    if (x > 1000) {
        if (x > 2000) {
            r = 0;
        }
    }
    return r;
}

void note(int x)
{
    if (x > 1000) {
        if (x > 2000) {
            g_noted = x;
        }
    }
}
"""
EQV_MAIN = """#include <stdio.h>
#include "lib.h"

int main(void)
{
    int bad = 0;

    if (classify(5) != 6) {
        printf("FAIL classify\\n");
        bad = 1;
    }
    note(3);
    if (!bad) {
        printf("ALL OK\\n");
    }
    return bad;
}
"""
CF_C = """static int twice(int x)
{
    return x * 2;
}

int f(int x)
{
    int r;

    r = twice(x);
    return r + 1;
}
"""
CF_MAIN = """#include <stdio.h>

int f(int x);

int main(void)
{
    if (f(3) != 7) {
        printf("FAIL f\\n");
        return 1;
    }
    printf("ALL OK\\n");
    return 0;
}
"""


def make_build_project(base: Path, name: str, files: dict[str, str], cc_flags: str, sources: str) -> tuple[corpus.CorpusSpec, Path]:
    """会真的构建并运行的小项目：make test 编译 sources 并运行，输出 ALL OK 且退出码为 0 才算通过。"""
    makefile = f"test: t_main\n\t./t_main\n\nt_main: {sources} lib.h\n\tgcc {cc_flags} $(CFLAGS) -o t_main {sources}\n"
    root = write_files(base / name, {**files, "Makefile": makefile})
    spec = corpus.CorpusSpec(
        name=name, project_files=[f for f in files if f.endswith((".c", ".h"))], target_files=["lib.c"],
        cflags=["-std=c99", "-I."], build_cmd="make test", pass_criteria={"exit_code": 0, "stdout_regex": "ALL OK"},
        timeout_s=60)
    return spec, root


@needs_docker
class TestRealSandboxOnSmallProjects:
    @pytest.fixture(scope="class")
    def sandbox(self):
        return DockerSandbox()

    @pytest.fixture(scope="class")
    def eqv(self, scratch_module):
        spec, root = make_build_project(scratch_module, "eqv", {"lib.h": LIB_H, "lib.c": EQV_C, "main.c": EQV_MAIN},
                                        "-std=c99 -Wall", "lib.c main.c")
        return spec, root, scratch_module / "work"

    def test_the_unmutated_project_passes(self, eqv, sandbox):
        spec, root, work = eqv
        ev = evaluate(spec, root, target_of(spec, root, "classify", "lib.c"), function_text(root, target_of(spec, root, "classify", "lib.c")),
                      sandbox, "m", "t", 0, work_root=work)
        assert ev.stage == "pass"

    def test_equivalent_mutants_are_counted_and_not_chosen(self, eqv, sandbox):
        """永远走不到的分支里的条件：ROR / NEG / OBO / DST 改了也不会让测试失败，是等价变异体。
        先试 NEG（两个都是等价的），再试别的算子，直到 AOR 或 RVR 这样真能被测试抓住的变异体。"""
        spec, root, work = eqv
        target = target_of(spec, root, "classify", "lib.c")
        assert mutate.count_sites(mutate.find_sites(spec, root, target)) == {
            "ROR": 2, "NEG": 2, "LOR": 0, "OBO": 2, "DGS": 0, "AOR": 1, "RVR": 1, "DST": 1}
        res = mutate.search_mutant(spec, root, target, "L0", SEED, sandbox, "NEG", target_id="eqv:lib.c:classify",
                                   work_root=work)
        m = res.mutant
        assert m is not None and res.outcome == "killed" and m.operator in ("AOR", "RVR") and m.kill_mode == "test_fail"
        assert [t["operator"] for t in res.tries[:2]] == ["NEG", "NEG"]
        assert all(t["stage"] == "pass" and t["category"] == "pass" for t in res.tries[:-1])
        assert res.tries[-1]["stage"] == "test"
        assert m.equivalent_count == m.tried_sites - 1 >= 2 and m.compile_fail_count == 0
        assert (res.equivalent_count, res.tried_sites) == (m.equivalent_count, m.tried_sites)

    def test_a_function_whose_mutants_are_all_equivalent_is_given_up(self, eqv, sandbox):
        spec, root, work = eqv
        target = target_of(spec, root, "note", "lib.c")
        assert sum(mutate.count_sites(mutate.find_sites(spec, root, target)).values()) == 7
        res = mutate.search_mutant(spec, root, target, "L0", SEED, sandbox, None, target_id="eqv:lib.c:note", work_root=work)
        assert res.mutant is None and res.outcome == "exhausted"
        assert (res.tried_sites, res.equivalent_count, res.compile_fail_count) == (7, 7, 0)
        assert mutate.choose_mutant(spec, root, target, "L0", SEED, sandbox, None, work_root=work) is None
        res = mutate.search_mutant(spec, root, target, "L0", SEED, sandbox, None, 3, target_id="eqv:lib.c:note", work_root=work)
        assert res.mutant is None and res.outcome == "gave_up" and res.tried_sites == res.equivalent_count == 3

    def test_a_mutant_that_does_not_build_counts_as_a_compile_failure_not_a_kill(self, scratch_module, sandbox):
        # -Werror 下，删掉 r = twice(x); 之后 twice 没人用，gcc 报错；libclang 不关心 unused-function，重新解析是干净的
        spec, root = make_build_project(scratch_module, "cf", {"lib.h": "", "lib.c": CF_C, "main.c": CF_MAIN},
                                        "-std=c99 -Wall -Werror", "lib.c main.c")
        target = target_of(spec, root, "f", "lib.c")
        res = mutate.search_mutant(spec, root, target, "L0", SEED, sandbox, "DST", target_id="cf:lib.c:f",
                                   work_root=scratch_module / "work")
        assert res.tries[0]["operator"] == "DST" and res.tries[0]["stage"] == "build"
        assert res.tries[0]["category"].startswith("compile_")
        m = res.mutant
        # 能被杀死的只有 AOR（RVR 把 r 变成「只赋值没使用」，-Werror 下同样编不过）；前面试过的全是构建失败
        assert m is not None and m.operator == "AOR" and m.kill_mode == "test_fail" and m.equivalent_count == 0
        assert [t["stage"] for t in res.tries[:-1]] == ["build"] * (len(res.tries) - 1)
        assert m.compile_fail_count == len(res.tries) - 1 >= 1 and res.parse_fail_count == 0


@pytest.fixture(scope="module")
def mini_search(mini_levels):
    """mini 夹具上真实的 stub-kill 选目标 + 每个入选目标在四级上的 choose_for_levels（8 路并发，约 1.5 分钟）。"""
    ml = mini_levels
    sandbox = DockerSandbox()
    sel = select(ml.spec, ml.ws.pristine("mini"), sandbox, TargetFilter(), SEED, jobs=8, scan_all=True,
                 work_root=ml.ws.work)

    def run(t):
        return t.target_id, mutate.choose_for_levels(ml.spec, ml.trees[t.target_id], SEED, sandbox, target_id=t.target_id,
                                                     work_root=ml.ws.work)

    with ThreadPoolExecutor(8) as pool:
        results = dict(pool.map(run, sel.targets))
    return {"sandbox": sandbox, "selected": sel.targets, "results": results}


@needs_docker
class TestMiniFixtureWithTheRealSandbox:
    def test_every_selected_target_has_a_killed_mutant_on_every_level(self, mini_levels, mini_search):
        sel, results = mini_search["selected"], mini_search["results"]
        assert len(sel) >= 20 and set(results) == {t.target_id for t in sel}
        for tid, per in results.items():
            assert list(per) == list(LEVELS)
            for lv, res in per.items():
                m = res.mutant
                assert m is not None and res.outcome == "killed", (tid, lv, res.outcome)
                assert m.kill_mode in KILL_MODES and m.target_id == tid and m.level == lv
                assert m.kill_recheck in KILL_MODES                              # 默认复核了：第二次也是失败
                assert 1 <= m.tried_sites <= mutate.DEFAULT_MAX_TRIES
                assert m.tried_sites == m.equivalent_count + m.compile_fail_count + m.unstable_count + 1

    def test_records_describe_exactly_the_mutated_site(self, mini_levels, mini_search):
        ml = mini_levels
        for tid, per in mini_search["results"].items():
            for lv, res in per.items():
                m = res.mutant
                root, located = ml.trees[tid][lv]
                data = (root / located.file).read_bytes()
                (site,) = [s for s in mutate.find_sites(ml.spec, root, located, m.operator) if s.index == m.site_index]
                assert site.original == m.original_text == data[site.start:site.end].decode()
                new, code = mutate_site(ml.spec, root, located, site, 0)
                assert m.mutated_text == mutate.mutation_text(site, random.Random(0)) or m.operator == "OBO"
                if m.operator != "OBO":                       # OBO 的 ±1 由位点自己的 rng 决定，这里的 rng 不同，不比文本
                    assert code == m.code
                assert m.code != function_text(root, located) and reparse_errors(ml.spec, root.resolve(), located.file, new) == []

    def test_mutants_really_fail_the_tests_and_the_originals_pass(self, mini_levels, mini_search):
        ml, sb = mini_levels, mini_search["sandbox"]
        picks = [(tid, lv) for tid in list(mini_search["results"])[::7] for lv in ("L0", "L3")]
        for tid, lv in picks:
            m = mini_search["results"][tid][lv].mutant
            root, located = ml.trees[tid][lv]
            killed = evaluate(ml.spec, root, located, m.code, sb, "check", f"{tid}@{lv}", 0, work_root=ml.ws.work)
            assert killed.stage == "test", (tid, lv)
            original = evaluate(ml.spec, root, located, function_text(root, located), sb, "check", f"{tid}@{lv}", 0,
                                work_root=ml.ws.work)
            assert original.stage == "pass", (tid, lv)

    def test_l0_operators_are_diverse_and_reused_across_levels(self, mini_search):
        results = mini_search["results"]
        ops0 = Counter(per["L0"].mutant.operator for per in results.values())
        assert len(ops0) >= 4                                  # 没有被某一个算子垄断
        same4 = sum(len({per[lv].mutant.operator for lv in LEVELS}) == 1 for per in results.values())
        assert same4 >= 0.7 * len(results)                     # 不复用的话四级同算子的比例只有三成左右
        for per in results.values():
            for lv, res in per.items():
                assert res.mutant.same_operator_as_L0 == (res.mutant.operator == per["L0"].mutant.operator)

    def test_l1_is_always_the_same_site_and_the_other_levels_mostly(self, mini_search):
        results = mini_search["results"]
        for per in results.values():
            m0, m1 = per["L0"].mutant, per["L1"].mutant
            assert m1.same_site_as_L0 is True and (m1.operator, m1.site_index) == (m0.operator, m0.site_index)
        flags = [per[lv].mutant.same_site_as_L0 for per in results.values() for lv in ("L2", "L3")]
        assert sum(flags) >= 0.6 * len(flags)

    def test_repeating_the_search_gives_the_same_mutants(self, mini_levels, mini_search):
        ml, sb = mini_levels, mini_search["sandbox"]
        for t in mini_search["selected"][::9]:
            again = mutate.choose_for_levels(ml.spec, ml.trees[t.target_id], SEED, sb, target_id=t.target_id,
                                             work_root=ml.ws.work)
            assert again == mini_search["results"][t.target_id]

    def test_choose_mutant_itself_returns_the_same_mutant_as_the_level_search(self, mini_levels, mini_search):
        ml, sb = mini_levels, mini_search["sandbox"]
        for t in mini_search["selected"][2::11]:
            root, located = ml.trees[t.target_id]["L0"]
            m = mutate.choose_mutant(ml.spec, root, located, "L0", SEED, sb, None, target_id=t.target_id, work_root=ml.ws.work)
            # same_operator_as_L0 是 choose_for_levels 事后填的，choose_mutant 单独调用时留空
            expected = dataclasses.replace(mini_search["results"][t.target_id]["L0"].mutant, same_operator_as_L0=None,
                                           same_site_as_L0=None)
            assert m is not None and m == expected
            # 指定另一个算子时优先用它（它在 L0 里有位点并且杀得死的话）
            other = next(op for op in ("DST", "RVR", "AOR") if op != m.operator)
            root1, located1 = ml.trees[t.target_id]["L1"]
            m1 = mutate.choose_mutant(ml.spec, root1, located1, "L1", SEED, sb, other, target_id=t.target_id,
                                      work_root=ml.ws.work)
            assert m1 is not None
            if mutate.find_sites(ml.spec, root1, located1, other):
                assert m1.operator == other or m1.equivalent_count + m1.compile_fail_count >= 1
