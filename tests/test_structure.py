"""结构混淆（L2）测试：bench/obfuscate/structure.py。

分四层：

1. 每条变换的最小用例：源码字符串进、字符串出，输出文本固定，并且用 libclang 重新解析零 error；
   同一个类里也放「必须不动」的边界（宏、条件编译、副作用、类型转换、复杂声明符），只看没变和跳过统计；
2. 基础设施：抽样比例、确定性、嵌套同时生效、回滚、头文件借宿主解析、vendored、符号链接、行数不变、
   不依赖标识符名字；
3. 沙箱（docker 或 gcc:13 镜像不可用时跳过）：mini 夹具 L2 后项目自带测试通过；语义对照程序变换前后输出逐字节一致；
   三目 return 的返回类型 x 分支类型矩阵、复合赋值的类型 x 运算符矩阵在边界值上结果一致；
4. 真实语料 cjson / parson / tinyexpr 变换后自带测试通过（语料没 prepare 时跳过）。

临时目录一律放在项目 tmp/ 下。
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import clang.cindex as ci
import pytest

from bench import clangcfg, corpus
from bench.obfuscate import _edits
from bench.obfuscate import structure as st
from bench.obfuscate import symbols
from bench.sandbox import DockerSandbox
from tests.helpers import MINI as FIXTURE_MINI, needs_docker, require_real_corpus


# ----------------------------------------------------------------------
# 公共工具
# ----------------------------------------------------------------------

def parse_errors(src: str | bytes, args=()) -> list[str]:
    """独立于被测模块的零 error 检查：libclang 解析源码，返回 error 级以上的诊断。"""
    clangcfg.configure()
    name = "t_structure_check.c"
    tu = clangcfg.index().parse(name, args=["-x", "c", "-w", *clangcfg.builtin_include_args(), *args],
                                unsaved_files=[(name, src)], options=0)
    return [f"{d.location.line}: {d.spelling}" for d in tu.diagnostics if d.severity >= ci.Diagnostic.Error]


def xf(src: str, kinds, args=(), seed: int = 0, ratio: float = 1.0) -> tuple[str, dict]:
    """只启用 kinds 里的变换（默认 100% 选中），断言输出能零 error 重新解析。"""
    kinds = [kinds] if isinstance(kinds, str) else list(kinds)
    out, stats = st.transform_source(src, args, seed, ratio=ratio, kinds=kinds)
    assert parse_errors(out, args) == [], out
    assert stats["files_skipped"] == {} and not any(stats["rolled_back"].values())
    return out, stats


def n_applied(stats: dict) -> dict:
    return {k: stats[k] for k in st.KINDS if stats[k]}


def mini_spec() -> corpus.CorpusSpec:
    return corpus.load_spec(str(FIXTURE_MINI / "spec.json"))


def sandbox_run(spec: corpus.CorpusSpec, root: Path):
    sb = DockerSandbox(timeout_s=spec.timeout_s + 60)
    res = sb.run(root, spec.run_command(), timeout_s=spec.timeout_s + 60)
    return spec.passes(res.rc, res.stdout) and not res.timed_out, res


def tree_errors(spec: corpus.CorpusSpec, root: Path, only_c: bool = False) -> dict[str, list[str]]:
    """工作副本里每个项目文件单独解析，返回有 error 的文件及诊断（only_c：只看 .c，头文件靠 .c 间接检查）。"""
    bad: dict[str, list[str]] = {}
    args = clangcfg.clang_args(spec.cflags, root)
    for p in corpus.expand_files(spec, root):
        if p.suffix not in _edits.SOURCE_SUFFIXES or (only_c and p.suffix != ".c"):
            continue
        tu = clangcfg.index().parse(str(p), args=args, options=0)
        errs = [d.spelling for d in tu.diagnostics if d.severity >= ci.Diagnostic.Error]
        if errs:
            bad[p.relative_to(root).as_posix()] = errs
    return bad


# ----------------------------------------------------------------------
# 1. 每条变换的最小用例
# ----------------------------------------------------------------------
class TestForToWhile:
    def test_expression_init(self):
        out, stats = xf("void g(int); void f(int n) { int i; for (i = 0; i < n; i++) g(i); }", "for_to_while")
        assert out == "void g(int); void f(int n) { int i; { i = 0; while (i < n) { g(i); i++; } } }"
        assert n_applied(stats) == {"for_to_while": 1}

    def test_braced_body_keeps_its_own_block(self):
        # body 自带的花括号要保留：body 里的声明可以遮蔽 step 用到的变量，step 不能被挪进 body 的作用域
        out, _ = xf("void g(int); void f(int n) { int i; for (i = 0; i < n; i++) { int i = 5; g(i); } }",
                    "for_to_while")
        assert out == ("void g(int); void f(int n) { int i; "
                       "{ i = 0; while (i < n) { { int i = 5; g(i); } i++; } } }")

    def test_declaration_init_scope_is_kept(self):
        # C99 的 for 头声明：整体包进花括号，同一函数里重复声明 i 才不会冲突
        src = "void g(int); void f(int n) { for (int i = 0, j = n; i < j; i++, j--) g(i); for (int i = 0; i < 3; i++) g(i); }"
        out, stats = xf(src, "for_to_while", args=["-std=c99"])
        assert out == ("void g(int); void f(int n) { { int i = 0, j = n; while (i < j) { g(i); i++, j--; } } "
                       "{ int i = 0; while (i < 3) { g(i); i++; } } }")
        assert stats["for_to_while"] == 2

    def test_missing_parts(self):
        out, _ = xf("void g(int); void f(int n) { for (;;) { g(n); } }", "for_to_while")
        assert out == "void g(int); void f(int n) { while (1) { g(n); } }"
        out, _ = xf("void f(int i) { for (; i < 9; ) i++; for (; i < 20; i++) ; }", "for_to_while")
        assert out == "void f(int i) { while (i < 9) i++; while (i < 20) { ; i++; } }"

    def test_empty_body(self):
        out, _ = xf("void f(int i, int n) { for (i = 0; i < n; i++); }", "for_to_while")
        assert out == "void f(int i, int n) { { i = 0; while (i < n) { ; i++; } } }"

    def test_continue_in_body_is_skipped(self):
        src = "void g(int); void f(int n) { int i; for (i = 0; i < n; i++) { if (i) continue; g(i); } }"
        out, stats = xf(src, "for_to_while")
        assert out == src
        assert stats["skipped"]["for_to_while"] == {"has_continue": 1}

    def test_continue_in_inner_loop_also_skips_outer(self):
        # 设计文档的口径：body 内任何位置有 continue 都不转
        src = ("void g(int); void f(int n) { int i, j; for (i = 0; i < n; i++) "
               "{ for (j = 0; j < n; j++) { if (j) continue; g(j); } } }")
        out, stats = xf(src, "for_to_while")
        assert out == src
        assert stats["skipped"]["for_to_while"] == {"has_continue": 2}

    def test_dangling_else_stays_with_its_if(self):
        src = "void g(int); void f(int a, int n, int i) { if (a) for (i = 0; i < n; i++) g(i); else g(-1); }"
        out, _ = xf(src, "for_to_while")
        assert out == "void g(int); void f(int a, int n, int i) { if (a) { i = 0; while (i < n) { g(i); i++; } } else g(-1); }"

    def test_macro_generated_for_is_skipped(self):
        src = "#define LOOP(i, n) for ((i) = 0; (i) < (n); (i)++)\nvoid g(int); void f(int n) { int i; LOOP(i, n) g(i); }\n"
        out, stats = xf(src, "for_to_while")
        assert out == src
        assert stats["skipped"]["for_to_while"] == {"macro_generated": 1}


class TestIfElseSwap:
    def test_braced_branches(self):
        out, stats = xf("int f(int a) { if (a) { return 1; } else { return 2; } }", "if_else_swap")
        assert out == "int f(int a) { if (!(a)) { return 2; } else { return 1; } }"
        assert n_applied(stats) == {"if_else_swap": 1}

    def test_condition_is_wrapped_as_a_whole(self):
        out, _ = xf("int f(int a) { if (a > 1 && a < 5) return 1; else return 2; }", "if_else_swap")
        assert out == "int f(int a) { if (!(a > 1 && a < 5)) { return 2; } else { return 1; } }"

    def test_non_block_branches_get_braces(self):
        # 不补花括号的话，else if 链交换后会出现悬空 else
        # 内层 if 没有 else：交换后它成了外层的 then 分支，必须带花括号，否则 else 会被它抢走
        out, _ = xf("int f(int a, int b) { if (a) return 1; else if (b) return 2; }", "if_else_swap")
        assert out == "int f(int a, int b) { if (!(a)) { if (b) return 2; } else { return 1; } }"
        out, _ = xf("int f(int a, int b) { if (a) if (b) return 1; else return 2; else return 3; }", "if_else_swap")
        assert out == ("int f(int a, int b) { if (!(a)) { return 3; } else "
                       "{ if (!(b)) { return 2; } else { return 1; } } }")

    def test_else_if_chain_nests_both_levels(self):
        out, stats = xf("int f(int a, int b) { if (a) { return 1; } else if (b) { return 2; } else { return 3; } }",
                        "if_else_swap")
        assert out == ("int f(int a, int b) { if (!(a)) { if (!(b)) { return 3; } else { return 2; } } "
                       "else { return 1; } }")
        assert stats["if_else_swap"] == 2

    def test_if_without_else_is_not_a_site(self):
        src = "int f(int a) { if (a) return 1; return 2; }"
        out, stats = xf(src, "if_else_swap")
        assert out == src
        assert stats["candidates"]["if_else_swap"] == 0 and stats["skipped"]["if_else_swap"] == {}

    def test_layout_is_kept_between_branches(self):
        src = "int f(int a) {\n  if (a)\n    return 1;\n  else\n    return 2;\n}\n"
        out, _ = xf(src, "if_else_swap")
        assert out == "int f(int a) {\n  if (!(a))\n    { return 2; }\n  else\n    { return 1; }\n}\n"

    def test_balanced_directives_inside_a_branch_move_intact(self):
        src = ("int f(int a) {\n  int r;\n  if (a) {\n#if 1\n    r = 1;\n#else\n    r = 2;\n#endif\n"
               "  } else {\n    r = 3;\n  }\n  return r;\n}\n")
        out, stats = xf(src, "if_else_swap")
        assert stats["if_else_swap"] == 1
        assert out.index("r = 3;") < out.index("#if 1") and out.count("#endif") == 1

    def test_define_before_the_statement_is_fine_but_inside_it_is_not(self):
        src = ("int f(int a) {\n  int r = 0;\n#define STEP(v) \\\n    ((v) + 1)\n"
               "  if (a) { r = STEP(a); } else { r = 2; }\n#undef STEP\n"
               "  if (r) {\n#define Q 1\n    r = Q;\n  } else { r = 2; }\n  return r;\n}\n")
        out, stats = xf(src, "if_else_swap")
        assert "if (!(a)) { r = 2; } else { r = STEP(a); }" in out        # 宏定义在语句之外：照常交换，续行的 #define 原样保留
        assert out.count("#define STEP(v) \\\n    ((v) + 1)") == 1
        assert "if (r) {\n#define Q 1" in out                            # 语句里面带 #define：不动
        assert stats["if_else_swap"] == 1 and stats["skipped"]["if_else_swap"] == {"directive": 1}

    def test_directive_between_branches_is_skipped(self):
        src = ("int f(int a) {\n  int r = 0;\n  if (a)\n    r = 1;\n#ifdef FOO\n  else\n    r = 2;\n#endif\n"
               "  return r;\n}\n")
        out, stats = xf(src, "if_else_swap", args=["-DFOO"])
        assert out == src
        assert stats["skipped"]["if_else_swap"] == {"directive_in_gap": 1}


class TestCompoundAssign:
    @pytest.mark.parametrize("op", ["+", "-", "*", "/", "%", "<<", ">>", "&", "^", "|"])
    def test_every_operator(self, op):
        out, stats = xf(f"void f(int x, int e) {{ x {op}= e; }}", "compound_assign")
        assert out == f"void f(int x, int e) {{ x = x {op} e; }}"
        assert stats["compound_assign"] == 1

    def test_member_chains_and_rhs_parentheses(self):
        src = ("struct S { int n; struct S *next; }; "
               "void f(struct S *s, struct S t, int e) { s->n += e + 1; t.n -= 2; s->next->n *= e; t.n |= s->n; }")
        out, stats = xf(src, "compound_assign")
        assert out == ("struct S { int n; struct S *next; }; "
                       "void f(struct S *s, struct S t, int e) { s->n = s->n + (e + 1); t.n = t.n - 2; "
                       "s->next->n = s->next->n * e; t.n = t.n | (s->n); }")
        assert stats["compound_assign"] == 4

    def test_rhs_conditional_needs_parentheses(self):
        out, _ = xf("void f(int x, int a) { x += a ? 1 : 2; }", "compound_assign")
        assert out == "void f(int x, int a) { x = x + (a ? 1 : 2); }"

    def test_subscript_deref_and_call_lvalues_are_skipped(self):
        src = "int *p(void); void f(int *q, int a[], int i, int e) { q[i] += e; *q -= 1; a[0] *= 2; *p() += 1; }"
        out, stats = xf(src, "compound_assign")
        assert out == src
        assert stats["skipped"]["compound_assign"] == {"lhs_not_simple": 4}

    def test_volatile_and_atomic_lvalues_are_skipped(self):
        # volatile 拆开会多读一次；_Atomic 的 x += 1 是单次原子读改写，拆开后就不是了
        src = "void f(volatile int x) { x += 1; }"
        out, stats = xf(src, "compound_assign")
        assert out == src and stats["skipped"]["compound_assign"] == {"lhs_not_simple": 1}
        src = "void f(void) { _Atomic int x = 0; x += 1; }"
        out, stats = xf(src, "compound_assign", args=["-std=c11"])
        assert out == src and stats["skipped"]["compound_assign"] == {"lhs_not_simple": 1}

    def test_rhs_with_calls_is_skipped(self):
        # x += f() 先读 x 还是先调 f 没有确定次序，f 又可能改 x，所以不拆
        src = "int g(void); void f(int x, int y) { x += g(); x -= (g() ? 1 : 2); x *= 2 * g(); }"
        out, stats = xf(src, "compound_assign")
        assert out == src
        assert stats["skipped"]["compound_assign"] == {"rhs_side_effect": 3}

    def test_rhs_call_hidden_in_a_macro_is_skipped(self):
        src = "int g(void);\n#define NOW g()\nvoid f(int x) { x += NOW; }\n"
        out, stats = xf(src, "compound_assign")
        assert out == src and stats["skipped"]["compound_assign"] == {"rhs_side_effect": 1}

    def test_rhs_increments_and_assignments_are_fine_and_nest(self):
        # 它们若真改到 x，原代码本身就是未定义行为；RHS 里的复合赋值作为嵌套位置同时被改写
        src = "void f(int x, int y, int z, int *p) { x += *p++; x += (y = 3); x *= (z += 2); x ^= y++ + ++z; }"
        out, stats = xf(src, "compound_assign")
        assert out == ("void f(int x, int y, int z, int *p) { x = x + (*p++); x = x + ((y = 3)); "
                       "x = x * ((z = z + 2)); x = x ^ (y++ + ++z); }")
        assert stats["compound_assign"] == 5

    def test_site_inside_macro_argument_is_skipped(self):
        # 参数里的改写会改变 #e 字符串化的结果
        src = '#define SHOW(e) printf("%s", #e), (e)\nint printf(const char *, ...);\nvoid f(int x) { SHOW(x += 1); }\n'
        out, stats = xf(src, "compound_assign")
        assert out == src and stats["compound_assign"] == 0

    def test_step_of_a_loop_and_nested_with_for(self):
        out, stats = xf("void g(int); void f(int i, int n) { for (; i < n; i += 2) g(i); }",
                        ["compound_assign", "for_to_while"])
        assert out == "void g(int); void f(int i, int n) { while (i < n) { g(i); i = i + 2; } }"
        assert n_applied(stats) == {"for_to_while": 1, "compound_assign": 1}


class TestTernaryReturn:
    def test_basic(self):
        out, stats = xf("int f(int a, int b) { return a < b ? b - a : a - b; }", "ternary_return")
        assert out == "int f(int a, int b) { if (a < b) return b - a; else return a - b; }"
        assert n_applied(stats) == {"ternary_return": 1}

    def test_parenthesized_ternary(self):
        out, _ = xf("int f(int a) { return (a ? 1 : 2); }", "ternary_return")
        assert out == "int f(int a) { if (a) return 1; else return 2; }"

    def test_non_block_parent_gets_braces(self):
        # return 是 if 的无花括号 then 分支时，不套花括号会把后面的 else 抢走
        out, _ = xf("int f(int a, int b) { if (a) return b > 5 ? 1 : 2; else return 0; }", "ternary_return")
        assert out == "int f(int a, int b) { if (a) { if (b > 5) return 1; else return 2; } else return 0; }"

    def test_case_and_label_parents_need_no_braces(self):
        out, _ = xf("int f(int a) { switch (a) { case 0: return a ? 1 : 2; default: ; } return 0; }", "ternary_return")
        assert out == "int f(int a) { switch (a) { case 0: if (a) return 1; else return 2; default: ; } return 0; }"

    def test_nested_ternary_keeps_the_inner_one(self):
        out, _ = xf("int f(int a, int b, int c) { return a ? b : c ? 1 : 2; }", "ternary_return")
        assert out == "int f(int a, int b, int c) { if (a) return b; else return c ? 1 : 2; }"

    def test_struct_pointer_and_small_constants(self):
        out, stats = xf("struct P { int x; }; struct P f(int a, struct P p, struct P q) { return a ? p : q; }",
                        "ternary_return")
        assert "if (a) return p; else return q;" in out
        out, _ = xf('const char *f(int a) { return a ? "y" : 0; }', "ternary_return")
        assert out == 'const char *f(int a) { if (a) return "y"; else return 0; }'
        out, _ = xf("unsigned long f(int a, unsigned long b) { return a ? b : 0; }", "ternary_return")
        assert out == "unsigned long f(int a, unsigned long b) { if (a) return b; else return 0; }"

    def test_enum_constants_and_object_macros_count_as_small_constants(self):
        src = "enum { A = 1, B = 2 };\n#define ZERO 0\nunsigned f(int c, unsigned u) { return c ? u : A; }\nunsigned g(int c, unsigned u) { return c ? u : ZERO; }\n"
        out, stats = xf(src, "ternary_return")
        assert "if (c) return u; else return A;" in out and "if (c) return u; else return ZERO;" in out
        assert stats["ternary_return"] == 2

    def test_value_changing_conversion_is_skipped(self):
        # -1 与 unsigned 做三目会先转成 unsigned，拆开后 return -1 转 long 得到 -1，值不同
        src = "long f(int a, unsigned b) { return a ? -1 : b; }"
        out, stats = xf(src, "ternary_return")
        assert out == src and stats["skipped"]["ternary_return"] == {"type_mismatch": 1}

    def test_ternary_from_macro_is_skipped(self):
        src = "#define PICK(c, a, b) ((c) ? (a) : (b))\nint f(int x) { return PICK(x > 0, x, -x); }\n"
        out, stats = xf(src, "ternary_return")
        assert out == src and stats["skipped"]["ternary_return"] == {"macro": 1}

    def test_plain_return_is_not_a_site(self):
        out, stats = xf("int f(int a) { return a; }", "ternary_return")
        assert out == "int f(int a) { return a; }"
        assert stats["candidates"]["ternary_return"] == 0 and stats["skipped"]["ternary_return"] == {}


class TestDeclSplit:
    def test_design_doc_examples(self):
        out, _ = xf("void f(void) { int a = 1, b; }", "decl_split")
        assert out == "void f(void) { int a = 1; int b; }"
        out, _ = xf("void f(void) { int *a, b; }", "decl_split")
        assert out == "void f(void) { int *a; int b; }"

    def test_arrays_initializers_and_pointers(self):
        out, _ = xf("void f(void) { int a[3] = {1, 2, 3}, b = (1, 2), *c[2], d = 4; }", "decl_split")
        assert out == "void f(void) { int a[3] = {1, 2, 3}; int b = (1, 2); int *c[2]; int d = 4; }"

    def test_storage_class_and_qualifiers_are_repeated(self):
        out, _ = xf('void f(void) { static const char *s = "a,b", *t; char *const p = 0, *const q = 0; }',
                    "decl_split")
        assert out == ('void f(void) { static const char *s = "a,b"; static const char *t; '
                       'char *const p = 0; char *const q = 0; }')

    def test_typedef_name_specifier(self):
        out, _ = xf("typedef unsigned long ul; void f(void) { ul a = 1, b = 2; }", "decl_split")
        assert out == "typedef unsigned long ul; void f(void) { ul a = 1; ul b = 2; }"

    def test_initializer_order_is_preserved_in_text(self):
        out, _ = xf("int g(void); void f(void) { int a = g(), b = g() + a; }", "decl_split")
        assert out == "int g(void); void f(void) { int a = g(); int b = g() + a; }"

    def test_unsupported_shapes_are_skipped_and_counted(self):
        for src, why in [
            ("void f(void) { int (*fp)(int), x; }", "declarator_shape"),
            ("void f(void) { struct { int a; } s1, s2; }", "not_plain_vars"),
            ("void f(void) { typedef int T1, *T2; }", "not_plain_vars"),
            ("void f(void) { int a __attribute__((unused)), b; }", "declarator_shape"),
        ]:
            out, stats = xf(src, "decl_split", args=["-std=gnu99"])
            assert out == src
            assert stats["skipped"]["decl_split"] == {why: 1}, src

    def test_single_declaration_and_for_init_are_not_sites(self):
        src = "void f(void) { int a = 1; for (int i = 0, j = 1; i < j; i++) ; }"
        out, stats = xf(src, "decl_split", args=["-std=c99"])
        assert out == src and stats["candidates"]["decl_split"] == 0

    def test_declaration_from_macro_is_skipped(self):
        src = "#define TWO(a, b) int a, b\nvoid f(void) { TWO(x, y); x = 1; y = 2; }\n"
        out, stats = xf(src, "decl_split")
        assert out == src and sum(stats["skipped"]["decl_split"].values()) == 1


# ----------------------------------------------------------------------
# 3. 基础设施
# ----------------------------------------------------------------------
MANY_FUNCS = "".join(f"int f{i}(int a) {{ if (a > {i}) return 1; else return 2; }}\n" for i in range(40))


class TestInfrastructure:
    def test_public_signature_matches_the_design_doc(self):
        params = list(inspect.signature(st.transform_structure).parameters)
        assert params[:3] == ["spec", "src_root", "seed"]
        assert st.KINDS == ("for_to_while", "if_else_swap", "compound_assign", "ternary_return", "decl_split")

    def test_ratio_zero_changes_nothing_and_one_changes_everything(self):
        out, stats = xf(MANY_FUNCS, "if_else_swap", ratio=0.0)
        assert out == MANY_FUNCS and stats["if_else_swap"] == 0 and stats["candidates"]["if_else_swap"] == 40
        out, stats = xf(MANY_FUNCS, "if_else_swap", ratio=1.0)
        assert stats["if_else_swap"] == 40 and out.count("!(a >") == 40

    def test_default_ratio_is_about_half(self):
        applied = []
        for seed in range(6):
            out, stats = st.transform_source(MANY_FUNCS, (), seed, kinds=["if_else_swap"])
            assert stats["ratio"] == st.DEFAULT_RATIO == 0.5
            applied.append(stats["if_else_swap"])
        assert 10 <= min(applied) and max(applied) <= 30
        assert 17 <= sum(applied) / len(applied) <= 23      # 每次约 20 个

    def test_same_seed_same_output_different_seed_different_output(self):
        a, _ = st.transform_source(MANY_FUNCS, (), 11, kinds=["if_else_swap"])
        b, _ = st.transform_source(MANY_FUNCS, (), 11, kinds=["if_else_swap"])
        c, _ = st.transform_source(MANY_FUNCS, (), 12, kinds=["if_else_swap"])
        assert a == b and a != c

    def test_selection_does_not_depend_on_identifier_names(self):
        tmpl = "int {f}(int {a}) {{ int {i}; for ({i} = 0; {i} < {a}; {i}++) {{ if ({i}) {a} += 2; else {a} -= 1; }} return {a}; }}\n"
        plain = "".join(tmpl.format(f=f"fn{n}", a="a", i="i") for n in range(30))
        renamed = "".join(tmpl.format(f=f"f_{n * 7 + 3}", a=f"v_{n}", i=f"loop_counter_{n}") for n in range(30))
        o1, s1 = st.transform_source(plain, (), 5)
        o2, s2 = st.transform_source(renamed, (), 5)
        assert n_applied(s1) == n_applied(s2) and s1["candidates"] == s2["candidates"]
        # 每个函数一行：被改动的行号集合必须一致
        changed = lambda src, out: [i for i, (x, y) in enumerate(zip(src.splitlines(), out.splitlines())) if x != y]
        assert changed(plain, o1) == changed(renamed, o2) and changed(plain, o1)

    def test_nested_sites_apply_together(self):
        src = ("void g(int); void f(int n) { int i, a = 0, b; "
               "for (i = 0; i < n; i += 2) { if (i > 3) { a += i; } else { b = i, g(b); } } }")
        out, stats = xf(src, st.KINDS)
        assert n_applied(stats) == {"for_to_while": 1, "if_else_swap": 1, "compound_assign": 2, "decl_split": 1}
        assert out.count("while (") == 1 and "i = i + 2;" in out and "a = a + i;" in out and "!(i > 3)" in out

    def test_line_count_is_preserved_per_file(self):
        src = ("int f(int n)\n{\n    int i, s = 0;\n    for (i = 0;\n         i < n;\n         i++)\n    {\n"
               "        s += i;\n    }\n    if (s)\n        return s ? 1 : 2;\n    else\n        return 3;\n}\n")
        out, stats = xf(src, st.KINDS)
        assert sum(n_applied(stats).values()) >= 4
        assert out.count("\n") == src.count("\n")

    def test_comments_and_non_ascii_bytes_survive(self):
        src = "/* 注释：中文 */\nint f(int a) { /* 内部 */ if (a) { return 1; } else { return 2; } } // 行尾\n"
        out, _ = xf(src, "if_else_swap")
        assert "/* 注释：中文 */" in out and "/* 内部 */" in out and out.endswith("// 行尾\n")

    def test_crlf_line_endings(self):
        src = "int f(int n)\r\n{\r\n    int i, s = 0;\r\n    for (i = 0; i < n;\r\n         i++) {\r\n        s += i;\r\n    }\r\n    return s;\r\n}\r\n"
        new, stats = st.transform_bytes(src.encode(), (), 0, ratio=1.0)
        assert parse_errors(new) == [] and new.count(b"\n") == src.count("\n") and new.count(b"\r\n") == new.count(b"\n")
        assert sum(n_applied(stats).values()) == 3

    def test_empty_comment_only_and_broken_sources(self):
        for src in ["", "\n\n", "/* nothing */\n// hello\n"]:
            out, stats = st.transform_source(src, (), 0, ratio=1.0)
            assert out == src and stats["total"] == 0
        broken = "int f( { return 1 }\n"
        out, stats = st.transform_source(broken, (), 0, ratio=1.0)
        assert out == broken and list(stats["files_skipped"]) == ["snippet.c"]
        assert "error" in stats["files_skipped"]["snippet.c"]

    def test_unknown_kind_is_rejected(self):
        with pytest.raises(ValueError, match="未知的变换种类"):
            st.transform_source("int x;", (), 0, kinds=["no_such_kind"])

    def test_header_with_static_inline_function_is_transformed(self, scratch):
        spec = corpus.CorpusSpec(name="hdr", project_files=["util.h", "main.c"], target_files=["util.h"],
                                 cflags=["-std=c99"], build_cmd="true")
        (scratch / "util.h").write_text("static inline int pick(int a) { if (a) return 1; else return 2; }\n")
        (scratch / "main.c").write_text('#include "util.h"\nint main(void) { return pick(0) - 2; }\n')
        stats = st.transform_structure(spec, scratch, 1, ratio=1.0)
        assert stats["if_else_swap"] == 1
        assert "if (!(a))" in (scratch / "util.h").read_text()
        assert tree_errors(spec, scratch) == {}

    def test_header_that_needs_its_includer_is_parsed_through_a_host(self, scratch):
        """util.h 用到了 va_list 却没自己包含 stdarg.h：单独解析不过，借包含它的 main.c 当宿主解析。"""
        spec = corpus.CorpusSpec(name="host", project_files=["util.h", "main.c"], target_files=["main.c"],
                                 cflags=["-std=c99"], build_cmd="true")
        header = ("static int vsum(int n, va_list ap) {\n  int i, s = 0;\n"
                  "  for (i = 0; i < n; i++) { if (i > 1) s += va_arg(ap, int); else s -= 1; }\n  return s;\n}\n")
        (scratch / "util.h").write_text(header)
        (scratch / "main.c").write_text('#include <stdarg.h>\n#include "util.h"\nint main(void) { return 0; }\n')
        stats = st.transform_structure(spec, scratch, 1, ratio=1.0)
        assert stats["files_skipped"] == {} and stats["if_else_swap"] == 1 and stats["for_to_while"] == 1
        assert "while (i < n)" in (scratch / "util.h").read_text() and "!(i > 1)" in (scratch / "util.h").read_text()
        assert tree_errors(spec, scratch, only_c=True) == {}      # 变换后的头文件经宿主 main.c 解析仍零 error

    def test_header_without_a_clean_host_is_skipped(self, scratch):
        spec = corpus.CorpusSpec(name="nohost", project_files=["util.h", "main.c"], target_files=["main.c"],
                                 cflags=["-std=c99"], build_cmd="true")
        header = "static int pick(int a) { if (a) return 1; else return 2; }\nstatic nosuch_t v;\n"
        (scratch / "util.h").write_text(header)
        (scratch / "main.c").write_text('#include "util.h"\nint main(void) { return 0; }\n')
        stats = st.transform_structure(spec, scratch, 1, ratio=1.0)
        assert (scratch / "util.h").read_text() == header and "util.h" in stats["files_skipped"]

    def test_vendored_directory_is_left_alone(self, scratch):
        spec = corpus.CorpusSpec(name="vend", project_files=["main.c", "vendor/*.c"], target_files=["main.c"],
                                 cflags=["-std=c99"], build_cmd="true", vendored=["vendor"])
        body = "int f(int a) { if (a) { return 1; } else { return 2; } }\n"
        (scratch / "vendor").mkdir()
        (scratch / "main.c").write_text(body)
        (scratch / "vendor" / "lib.c").write_text(body)
        stats = st.transform_structure(spec, scratch, 1, ratio=1.0)
        assert (scratch / "vendor" / "lib.c").read_text() == body
        assert (scratch / "main.c").read_text() != body
        assert stats["files_total"] == 1 and list(stats["files"]) == ["main.c"]

    def test_files_argument_restricts_the_run(self, scratch):
        spec = corpus.CorpusSpec(name="two", project_files=["a.c", "b.c"], target_files=["a.c"], cflags=["-std=c99"],
                                 build_cmd="true")
        body = "int f(int a) { if (a) { return 1; } else { return 2; } }\n"
        (scratch / "a.c").write_text(body)
        (scratch / "b.c").write_text(body)
        stats = st.transform_structure(spec, scratch, 1, ratio=1.0, files=["b.c"])
        assert (scratch / "a.c").read_text() == body and (scratch / "b.c").read_text() != body
        assert stats["files_total"] == 1 and list(stats["files"]) == ["b.c"]

    def test_symlink_is_not_written_through(self, scratch):
        outside = scratch / "outside"
        outside.mkdir()
        body = "int f(int a) { if (a) { return 1; } else { return 2; } }\n"
        (outside / "real.c").write_text(body)
        root = scratch / "proj"
        root.mkdir()
        (root / "link.c").symlink_to(outside / "real.c")
        spec = corpus.CorpusSpec(name="lnk", project_files=["link.c"], target_files=["link.c"], cflags=["-std=c99"],
                                 build_cmd="true")
        stats = st.transform_structure(spec, root, 1, ratio=1.0)
        assert (outside / "real.c").read_text() == body
        assert "link.c" in stats["files_skipped"] and stats["files_changed"] == 0

    def test_file_with_baseline_errors_is_skipped_not_half_done(self, scratch):
        spec = corpus.CorpusSpec(name="bad", project_files=["bad.c", "good.c"], target_files=["good.c"],
                                 cflags=["-std=c99"], build_cmd="true")
        good = "int f(int a) { if (a) { return 1; } else { return 2; } }\n"
        bad = "void_t g(int a) { if (a) { return 1; } else { return 2; } }\n"      # 未知类型：原文件就解析不过
        (scratch / "good.c").write_text(good)
        (scratch / "bad.c").write_text(bad)
        stats = st.transform_structure(spec, scratch, 1, ratio=1.0)
        assert (scratch / "bad.c").read_text() == bad and "bad.c" in stats["files_skipped"]
        assert (scratch / "good.c").read_text() != good

    def test_stats_are_json_serializable_and_complete(self, scratch):
        root = corpus.materialize(FIXTURE_MINI, scratch / "mini")
        stats = st.transform_structure(mini_spec(), root, 3, ratio=1.0)
        json.dumps(stats)
        assert stats["total"] == sum(stats[k] for k in st.KINDS) > 0
        assert set(stats["candidates"]) == set(stats["skipped"]) == set(stats["rolled_back"]) == set(st.KINDS)
        assert stats["files_total"] == 4 and stats["seed"] == 3 and stats["ratio"] == 1.0

    def test_rollback_drops_only_the_bad_edit(self, monkeypatch):
        """故意制造一个会让解析出错的变换：只能回滚它，其余位置照常生效。"""
        def poisoned(self, c, parent):
            s, e = self.ctx.extent(c)
            return _edits.Edit("decl_split", s, e, [b"int @@@ = ;"])

        monkeypatch.setattr(st._SiteFinder, "_decl", poisoned)
        src = ("int f(int a) { int x = 1, y = 2; if (a) { return x; } else { return y; } }\n"
               "int g(int a) { int p = 1, q = 2; return a ? p : q; }\n")
        out, stats = st.transform_source(src, (), 0, ratio=1.0)
        assert parse_errors(out) == [] and "@@@" not in out
        assert stats["rolled_back"]["decl_split"] == 2 and stats["decl_split"] == 0
        assert stats["if_else_swap"] == 1 and stats["ternary_return"] == 1

    def test_laminar_check_drops_partially_overlapping_edits(self):
        a = _edits.Edit("x", 0, 10, [(0, 10)])
        inside = _edits.Edit("x", 2, 5, [(2, 5)])
        crossing = _edits.Edit("x", 8, 14, [(8, 14)])
        outside = _edits.Edit("x", 20, 30, [(20, 30)])
        kept, dropped = _edits.check_laminar([a, inside, crossing, outside])
        assert kept == [a, inside, outside] and dropped == 1


# ----------------------------------------------------------------------
# 4. 沙箱
# ----------------------------------------------------------------------
@needs_docker
class TestMiniFixture:
    @pytest.mark.parametrize("seed,ratio", [(1, 0.5), (2, 0.5), (3, 1.0)])
    def test_project_tests_still_pass_after_l2(self, scratch, seed, ratio):
        spec = mini_spec()
        root = corpus.materialize(FIXTURE_MINI, scratch / "mini")
        stats = st.transform_structure(spec, root, seed, ratio=ratio)
        assert stats["total"] > 0 and stats["files_skipped"] == {} and not any(stats["rolled_back"].values())
        assert tree_errors(spec, root) == {}
        ok, res = sandbox_run(spec, root)
        assert ok, res.stdout[-800:] + res.stderr[-1500:]
        assert "0 failures" in res.stdout

    def test_every_kind_fires_on_mini_at_full_ratio(self, scratch):
        root = corpus.materialize(FIXTURE_MINI, scratch / "mini")
        stats = st.transform_structure(mini_spec(), root, 3, ratio=1.0)
        assert all(stats[k] > 0 for k in st.KINDS), n_applied(stats)
        # 全选时，通过全部检查的候选位置一个不剩
        assert all(stats[k] == stats["candidates"][k] for k in st.KINDS)

    def test_deterministic_and_line_counts_kept(self, scratch):
        spec = mini_spec()
        a = corpus.materialize(FIXTURE_MINI, scratch / "a")
        b = corpus.materialize(FIXTURE_MINI, scratch / "b")
        st.transform_structure(spec, a, 7)
        st.transform_structure(spec, b, 7)
        assert corpus.tree_digest(a) == corpus.tree_digest(b) != corpus.tree_digest(FIXTURE_MINI)
        for p in corpus.expand_files(spec, a):
            orig = FIXTURE_MINI / p.relative_to(a)
            assert p.read_bytes().count(b"\n") == orig.read_bytes().count(b"\n"), p.name

    def test_runs_on_l1_renamed_copy_with_same_selection(self, scratch):
        spec = mini_spec()
        plain = corpus.materialize(FIXTURE_MINI, scratch / "plain")
        renamed = corpus.materialize(FIXTURE_MINI, scratch / "renamed")
        names = symbols.collect_project_identifiers(spec, renamed)      # L1 出问题就让这个测试失败，不跳过
        symbols.apply_renames(spec, renamed, symbols.build_rename_map(names, 5))
        assert corpus.tree_digest(plain) != corpus.tree_digest(renamed)
        s_plain = st.transform_structure(spec, plain, 5)
        s_renamed = st.transform_structure(spec, renamed, 5)
        assert n_applied(s_plain) == n_applied(s_renamed) and s_plain["candidates"] == s_renamed["candidates"]
        ok, res = sandbox_run(spec, renamed)
        assert ok, res.stdout[-800:] + res.stderr[-1500:]


DIFF_C = r'''/* structure_diff.c：结构混淆的语义对照程序。
 *
 * 一个函数一类构造，main 里把所有结果折进校验和并逐行打印。测试把它原样编译一份、
 * 变换后再编译一份，两份输出必须逐字节相同（见 tests/test_structure.py）。
 * 刻意避开未定义 / 未指明求值次序的写法，这样输出只取决于语义是否等价。
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define STR_(x) #x
#define STR(x) STR_(x)
#define SHOW(e) printf("%s => %ld\n", #e, (long)(e))
#define LOOP_N(i, n) for ((i) = 0; (i) < (n); (i)++)
#define ADD_TO(v, e) ((v) += (e))
#define MIN2(a, b) ((a) < (b) ? (a) : (b))
#define PICK(c, a, b) ((c) ? (a) : (b))

static unsigned long acc = 1469598103UL;
static int counter = 0;

static void emit(long v)
{
    acc = (acc * 16777619UL) ^ (unsigned long)v;
}

static int next_val(void)
{
    counter = counter + 1;
    return counter;
}

struct pt { int x; int y; };
struct box {
    struct pt lo;
    struct pt hi;
    struct box *next;
    unsigned char flags;
    short small;
};
typedef struct { int a; double d; } mixed_t;

/* ------------------------------------------------------------------ for */
static int loop_basic(int n)
{
    int i, s = 0;

    for (i = 0; i < n; i++) {
        s += i;
    }
    return s;
}

static int loop_c99_decl(int n)
{
    int s = 0;

    for (int i = 0, j = n; i < j; i++, j--)
        s += i * j;
    return s;
}

static int loop_empty_parts(int n)
{
    int i = 0, s = 0;

    for (;;) {
        if (i >= n) {
            break;
        }
        s += i;
        i++;
    }
    for (; i < n + 3; ) {
        s += 2;
        i++;
    }
    for (i = 0; ; i++) {
        if (i > 4) {
            break;
        }
        s++;
    }
    return s;
}

static int loop_shadow(int n)
{
    int i, s = 0;

    for (i = 0; i < n; i++) {
        int i = 100;
        s += i;
    }
    return s;
}

static int loop_nested_break(int n)
{
    int i, j, s = 0;

    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++) {
            if (j > i) {
                break;
            }
            s += j;
        }
    return s;
}

static int loop_switch(int n)
{
    int i, s = 0;

    for (i = 0; i < n; i++) {
        switch (i % 3) {
        case 0: s += 1; break;
        case 1: s += 10; break;
        default: s += 100;
        }
    }
    return s;
}

static int loop_goto_out(int n)
{
    int i, s = 0;

    for (i = 0; i < n; i++) {
        if (i == 3) {
            goto done;
        }
        s += i;
    }
done:
    return s * 2;
}

static int loop_continue_kept(int n)
{
    int i, s = 0;

    for (i = 0; i < n; i++) {
        if (i % 2) {
            continue;
        }
        s += i;
    }
    return s;
}

static int loop_label(int n)
{
    int i = 0, s = 0;

again:
    for (i = 0; i < 3; i++)
        s += i;
    if (s < n) {
        goto again;
    }
    return s;
}

static int loop_as_then(int n, int flag)
{
    int i, s = 0;

    if (flag)
        for (i = 0; i < n; i++)
            s += i;
    else
        s = -1;
    return s;
}

static int loop_if_inside(int n)
{
    int i, s = 0;

    for (i = 0; i < n; i++)
        if (i % 2)
            s += i;
        else
            s -= i;
    return s;
}

static int loop_pointer(const int *a, int n)
{
    const int *p;
    int s = 0;

    for (p = a; p < a + n; p++)
        s += *p;
    return s;
}

static int loop_comma_step(int n)
{
    int i, j, s = 0;

    for (i = 0, j = n; i < j; i++, j--)
        s += j - i;
    return s;
}

static int loop_cond_compound(int n)
{
    int s = 0, k = n;

    for (; (k -= 2) > 0; )
        s++;
    return s + k;
}

static int loop_step_compound(int n)
{
    int i, s = 0;

    for (i = 0; i < n; i += 3)
        s += i;
    return s;
}

static int loop_do_inside(int n)
{
    int i, s = 0;

    for (i = 0; i < n; i++) {
        int k = 0;

        do {
            k++;
            s += k;
        } while (k < i);
    }
    return s;
}

static int loop_macro(int n)
{
    int i, s = 0;

    LOOP_N(i, n) {
        s += i;
    }
    for (i = 0; i < n; i++) {
        s += 2;
    }
    return s;
}

static int loop_empty_body(int n)
{
    int i;

    for (i = 0; i < n; i++)
        ;
    return i;
}

/* ------------------------------------------------------------------ if */
static int if_chain(int x)
{
    if (x < 0) return -1; else if (x == 0) return 0; else if (x < 10) return 1; else return 2;
}

static int if_dangling(int a, int b)
{
    int r = 0;

    if (a)
        if (b)
            r = 1;
        else
            r = 2;
    else
        r = 3;
    return r;
}

static int if_dangling_inner(int a, int b)
{
    int r = 0;

    if (a) {
        if (b)
            r = 1;
    } else
        r = 3;
    return r;
}

static int if_side_effect_cond(int n)
{
    counter = 0;
    if (next_val() > n) {
        return counter;
    } else {
        return -counter;
    }
}

static int if_in_loop(int n)
{
    int i, s = 0;

    for (i = 0; i < n; i++) {
        if (i & 1) {
            s += 3;
        } else {
            s -= 1;
        }
    }
    return s;
}

static int if_labels(int x)
{
    int r = 0;

    if (x) goto a; else goto b;
a:  r = 1; goto end;
b:  r = 2;
end:
    return r;
}

static int if_switch(int x)
{
    switch (x) {
    case 0:
        if (x == 0) return 5; else return 6;
    case 1: {
        if (x) { return 7; } else { return 8; }
    }
    default:
        break;
    }
    return 9;
}

static int if_float(double d)
{
    if (d) return 1; else return 0;
}

static double make_nan(void)
{
    volatile double z = 0.0;

    return z / z;
}

static int if_pointer(const char *p)
{
    if (p) { return (int)strlen(p); } else { return -1; }
}

static int if_assign_cond(int n)
{
    int c;

    if ((c = n * 2) > 10) return c; else return -c;
}

static int if_comma_cond(int n)
{
    int c = 0;

    if (c++, n > c) return 1; else return 0;
}

static int if_decl_in_branch(int x)
{
    if (x > 0) {
        int t = x * 2, u = t + 1;
        return u;
    } else {
        int t = -x;
        return t;
    }
}

static int if_empty_branches(int x)
{
    int r = 0;

    if (x) { } else { r = 1; }
    if (x > 5) ; else r += 10;
    return r;
}

/* ------------------------------------------------------------ 复合赋值 */
static long compound_ops(int a, unsigned b)
{
    int i = a;
    unsigned u = b;
    long l = a;
    char c = 10;
    short s = -5;
    unsigned char uc = 250;
    float f = 1.5f;
    double d = 2.5;

    i += 3; i -= 1; i *= 4; i /= 3; i %= 5; i <<= 2; i >>= 1; i &= 0xff; i ^= 5; i |= 2;
    u += i; u -= 1; u *= 3; u /= 2; u %= 1000; u <<= 3; u >>= 2; u &= 0xfff; u ^= 7; u |= 1;
    l += i * 1000; l -= u;
    c += 120; c *= 2; s -= 32767; s *= 3; uc += 10; uc <<= 1;
    f += 1; f *= 1.5; f /= 3; d -= f; d *= d; d /= 7;
    i += c ? 1 : 2;
    i -= a + b;
    i *= a - 1;
    i <<= 1 + 1;
    i |= a & 3;
    return (long)i + (long)u + l + c + s + uc + (long)(f * 100) + (long)d;
}

static long compound_members(void)
{
    struct box bx;
    struct box *p = &bx;
    mixed_t m;

    bx.lo.x = 1; bx.lo.y = 2; bx.hi.x = 3; bx.hi.y = 4;
    bx.next = &bx; bx.flags = 0; bx.small = 1;
    bx.lo.x += 5;
    p->hi.y *= 3;
    p->next->lo.y -= 1;
    p->next->hi.x <<= 2;
    bx.flags |= 4;
    p->flags ^= 1;
    bx.small += 30000;
    bx.small += 30000;
    m.a = 5; m.d = 1.0;
    m.a += 2;
    m.d /= 4;
    return bx.lo.x + bx.lo.y + bx.hi.x + bx.hi.y + bx.flags + bx.small + m.a + (long)(m.d * 8);
}

static long compound_pointer(void)
{
    int arr[5] = { 1, 2, 3, 4, 5 };
    int *p = arr;

    p += 3;
    p -= 1;
    arr[1] += 10;
    *p += 100;
    return *p + arr[1];
}

static long compound_value_used(int n)
{
    int x = n, y;

    y = (x += 2);
    return y * 100 + x;
}

static long compound_rhs_pure_macro(int n)
{
    int x = 1;

    x += MIN2(n, 7);
    x <<= 2;
    ADD_TO(x, 5);
    return x;
}

static long compound_in_cond(int n)
{
    int x = n, c = 0;

    while ((x -= 3) > 0) {
        c++;
    }
    return c * 10 + x;
}

static long compound_rhs_effects(int n)
{
    int a[4] = { 1, 2, 3, 4 };
    int *p = a;
    int x = n, y = 0, z = 0;

    x += *p++;
    x -= *p++;
    x += (y = 5);
    x *= (z += 2);
    x ^= y++ + ++z;
    x |= (y ? (z = 3) : 4);
    return x * 1000 + y * 100 + z * 10 + (p - a);
}

/* -------------------------------------------------------- return 三目 */
static int tern_int(int a, int b)
{
    return a < b ? b - a : a - b;
}

static long tern_mixed(int a, long b)
{
    return a ? b : 0;
}

static double tern_double(int a)
{
    return a ? 1.5 : 2;
}

static unsigned tern_unsigned(int a, unsigned b)
{
    return a ? b : 0u;
}

static long tern_signed_unsigned(int a, unsigned b)
{
    return a ? -1 : b;
}

static const char *tern_ptr(int a)
{
    return a ? "yes" : NULL;
}

static struct pt tern_struct(int a, struct pt p, struct pt q)
{
    return a ? p : q;
}

static char tern_char(int a)
{
    char x = 'x', y = 'y';

    return a ? x : y;
}

static int tern_nested(int a, int b)
{
    return a ? (b ? 1 : 2) : (b ? 3 : 4);
}

static int tern_chain(int a)
{
    return a < 0 ? -1 : a == 0 ? 0 : a < 10 ? 1 : 2;
}

static int tern_paren(int a)
{
    return (a ? 10 : 20);
}

static int tern_case(int a)
{
    switch (a) {
    case 0: return a ? 1 : 2;
    case 1: return a ? 3 : 4;
    default: ;
    }
    return 5;
}

static int tern_if(int a)
{
    if (a)
        return a > 5 ? 1 : 2;
    else
        return a < -5 ? 3 : 4;
}

static int tern_if_noelse(int a)
{
    if (a)
        return a > 5 ? 1 : 2;
    return 0;
}

static int tern_comma(int a)
{
    return a ? (a++, a) : 0;
}

static int tern_side(int a)
{
    counter = 0;
    return a ? next_val() : -next_val();
}

static short tern_short(int a)
{
    short s = 5, t = -5;

    return a ? s : t;
}

static float tern_float(int a, float f)
{
    return a ? f : 0;
}

static long tern_long_int(int a, int b, long c)
{
    return a ? b : c;
}

static int tern_macro(int a, int b)
{
    return PICK(a > b, a, b);
}

static int tern_label(int a)
{
    int r = 0;

    if (a > 100) goto out;
    r = 1;
out:
    return a ? r + 1 : r + 2;
}

/* ---------------------------------------------------------- 多声明拆分 */
static long decl_basic(void)
{
    int a = 1, b = 2, c;
    int *p = &a, *q = &b, r = 7;
    int arr[3] = { 1, 2, 3 }, n = 3, m[2][2] = { { 1, 2 }, { 3, 4 } };
    const char *s1 = "x", *s2 = "yz";
    static int st1 = 5, st2 = 6;
    unsigned long ul = 5UL, ul2 = ul * 2;
    char buf[4] = "ab", *bp = buf;
    int first = next_val(), second = next_val();
    struct pt pt1 = { 1, 2 }, pt2 = { 3, 4 }, *ptp = &pt1;
    double d1 = 1.5, d2 = d1 * 2;
    char *const cp = buf, *const cq = buf + 1;
    const int ci = 9, *const cpc = &ci;
    int sum = 0, i;

    c = a + b;
    st1 += 1;
    for (i = 0; i < 3; i++) {
        sum += arr[i] + m[i % 2][i % 2];
    }
    return c + *p + *q + r + n + (long)strlen(s1) + (long)strlen(s2) + st1 + st2 + (long)ul + (long)ul2 +
           *bp + first * 10 + second + pt1.x + pt2.y + ptp->y + (long)(d1 + d2) + *cp + *cq + ci + *cpc + sum;
}

static long decl_with_calls(int n)
{
    int x = n * 2, y = x + 1, z = MIN2(x, y);
    int u = (n > 1 ? 1 : 2), v = (n, 5);
    long big = (long)x * 1000, small = 3;

    return x + y + z + u + v + big + small;
}

static long decl_inside_blocks(int n)
{
    long total = 0;
    int i;

    for (i = 0; i < n; i++) {
        int a = i, b = i * i;

        total += a + b;
    }
    {
        int p = 1, q = 2;

        total += p * 10 + q;
    }
    return total;
}

/* ------------------------------------------------------ 宏参数与指令 */
static long macro_args(int n)
{
    int x = n;

    SHOW(x += 2);
    SHOW(x -= 1);
    SHOW(x ? 1 : 2);
    return x;
}

static int cond_directives(int x)
{
    int r = 0;

    if (x > 1) {
#if 1
        r = 1;
#else
        r = 2;
#endif
    } else {
        r = 3;
    }
#ifdef STRUCTURE_DIFF_NEVER
    if (x) r = 100;
    else
#endif
    r += 1;
    return r;
}

/* 全部混在一起 */
static long mixed(int n)
{
    long total = 0;
    int i, j;

    for (i = 0; i < n; i++) {
        int a = i * 2, b = i + 1;

        if (a > b) {
            total += a;
        } else {
            total -= b;
        }
        for (j = 0; j < i; j += 2) {
            total += (j & 1) ? 100 : 1;
        }
    }
    return total > 0 ? total : -total;
}

/* ---------------------------------------------------------------- main */
int main(void)
{
    int arr[6] = { 4, 8, 15, 16, 23, 42 };
    int n;
    struct pt p1 = { 1, 2 }, p2 = { 3, 4 };

    for (n = 0; n < 8; n++) {
        emit(loop_basic(n)); emit(loop_c99_decl(n)); emit(loop_empty_parts(n));
        emit(loop_shadow(n)); emit(loop_nested_break(n)); emit(loop_switch(n));
        emit(loop_goto_out(n)); emit(loop_continue_kept(n)); emit(loop_label(n + 5));
        emit(loop_as_then(n, 0)); emit(loop_as_then(n, 1)); emit(loop_if_inside(n));
        emit(loop_pointer(arr, n % 7)); emit(loop_comma_step(n)); emit(loop_cond_compound(n));
        emit(loop_step_compound(n)); emit(loop_do_inside(n)); emit(loop_macro(n));
        emit(loop_empty_body(n));
        emit(if_chain(n - 3)); emit(if_dangling(n & 1, n & 2)); emit(if_dangling_inner(n & 1, n & 2));
        emit(if_side_effect_cond(n)); emit(if_in_loop(n)); emit(if_labels(n & 1));
        emit(if_switch(n % 3)); emit(if_float(n * 0.5 - 1)); emit(if_float(make_nan())); emit(if_pointer(n & 1 ? "abc" : NULL));
        emit(if_assign_cond(n)); emit(if_comma_cond(n)); emit(if_decl_in_branch(n - 4));
        emit(if_empty_branches(n));
        emit(compound_ops(n, (unsigned)n * 7u)); emit(compound_members()); emit(compound_pointer());
        emit(compound_value_used(n)); emit(compound_rhs_effects(n)); emit(compound_rhs_pure_macro(n)); emit(compound_in_cond(n * 3));
        emit(tern_int(n, 4)); emit(tern_mixed(n & 1, 99L)); emit((long)(tern_double(n & 1) * 10));
        emit(tern_unsigned(n & 1, 77u)); emit(tern_signed_unsigned(n & 1, 5u));
        emit(tern_ptr(n & 1) ? 1 : 0); emit(tern_struct(n & 1, p1, p2).x);
        emit(tern_char(n & 1)); emit(tern_nested(n & 1, n & 2)); emit(tern_chain(n - 2));
        emit(tern_paren(n & 1)); emit(tern_case(n % 3)); emit(tern_if(n - 6));
        emit(tern_if_noelse(n - 1)); emit(tern_comma(n)); emit(tern_side(n & 1));
        emit(tern_short(n & 1)); emit((long)(tern_float(n & 1, 2.5f) * 4));
        emit(tern_long_int(n & 1, 3, 100000L)); emit(tern_macro(n, 4)); emit(tern_label(n * 30));
        emit(decl_basic()); emit(decl_with_calls(n)); emit(decl_inside_blocks(n));
        emit(macro_args(n)); emit(cond_directives(n)); emit(mixed(n));
        counter = 0;
    }
    printf("checksum %lu\n", acc);
    printf("STR %s\n", STR(a += b));
    return 0;
}
'''


@needs_docker
class TestSemanticEquivalence:
    def test_differential_program_prints_the_same_after_every_kind_of_transform(self, scratch):
        """一个函数一类构造的对照程序：原样编译一份、变换后再编译几份，输出必须逐字节相同。"""
        (scratch / "orig.c").write_text(DIFF_C, encoding="utf-8")
        names = []
        for seed, ratio in [(1, 1.0), (2, 0.5), (3, 0.5), (4, 0.7)]:
            new, stats = st.transform_source(DIFF_C, ["-std=c99"], seed, ratio=ratio, filename="structure_diff.c",
                                             workdir=scratch)
            assert parse_errors(new, ["-std=c99"]) == [] and not any(stats["rolled_back"].values())
            assert stats["files_skipped"] == {}
            if ratio == 1.0:
                assert all(stats[k] >= 10 for k in st.KINDS), n_applied(stats)
            names.append(f"v{seed}.c")
            (scratch / names[-1]).write_text(new, encoding="utf-8")
        cmd = ("set -e; gcc -std=c99 -O0 -w orig.c -o orig && ./orig > orig.out; "
               f"for v in {' '.join(names)}; do gcc -std=c99 -O0 -w $v -o ${{v%.c}} && ./${{v%.c}} > ${{v%.c}}.out "
               "&& cmp orig.out ${v%.c}.out && echo SAME $v; done")
        res = DockerSandbox().run(scratch, cmd, timeout_s=120)
        assert res.rc == 0, res.stdout + res.stderr[-2000:]
        assert res.stdout.count("SAME") == len(names)
        out = (scratch / "orig.out").read_text()
        # SHOW(x += 2) 用 #e 把参数原文打印出来：宏参数里的复合赋值不能被改写，否则两份输出早就不同了
        assert "checksum" in out and "STR a += b" in out and "x += 2 =>" in out and "x = x" not in out


# 类型 x 类型 / 类型 x 运算符矩阵：凡是被放行的改写，在边界值上与原代码输出必须逐字节一致
_MX_VALS = {
    "signed char": ["-1", "0", "127", "-128"], "unsigned char": ["0", "1", "255"],
    "short": ["-1", "0", "32767", "-32768"], "unsigned short": ["0", "1", "65535"],
    "int": ["-1", "0", "1", "2147483647", "(-2147483647-1)"], "unsigned": ["0", "1", "4294967295u", "2147483648u"],
    "long": ["-1L", "0L", "9223372036854775807L", "(-9223372036854775807L-1)"],
    "unsigned long": ["0UL", "1UL", "18446744073709551615UL"],
    "float": ["-1.5f", "0.0f", "3.9f", "100.25f"], "double": ["-1.5", "0.0", "3.9", "100.25"],
}
_MX_NONZERO = {
    "signed char": ["1", "3", "-2"], "unsigned char": ["1", "3", "255"], "short": ["1", "3", "-2"],
    "unsigned short": ["1", "3", "65535"], "int": ["1", "3", "-2"], "unsigned": ["1", "3", "4294967295u"],
    "long": ["1L", "3L", "-2L"], "unsigned long": ["1UL", "3UL"], "float": ["1.5f", "-2.25f", "0.5f"],
    "double": ["1.5", "-2.25", "0.5"],
}
_MX_TYPES = list(_MX_VALS)


def _shards(funcs: list[str], calls: list[list[str]], size: int = 100):
    """函数和它们各自的调用语句成对切片，每片生成一个独立的 C 文件（片太大 gcc 会吃光沙箱内存）。"""
    for i in range(0, len(funcs), size):
        body = "\n".join(funcs[i:i + size])
        main = "\n".join(c for cs in calls[i:i + size] for c in cs)
        yield "#include <stdio.h>\n" + body + "\nint main(void) {\n" + main + "\nreturn 0; }\n"


def _compare_in_sandbox(scratch: Path, pairs: list[tuple[str, str]], flags: str = "") -> str:
    """pairs 里每对 (原程序, 变换后程序) 各自编译运行，输出必须逐字节相同；返回沙箱 stdout。"""
    for k, (orig, new) in enumerate(pairs):
        (scratch / f"o{k}.c").write_text(orig)
        (scratch / f"n{k}.c").write_text(new)
    cmd = (f"set -e; for k in $(seq 0 {len(pairs) - 1}); do "
           f"gcc -std=c99 -O0 {flags} -w o$k.c -o o$k; gcc -std=c99 -O0 {flags} -w n$k.c -o n$k; "
           "./o$k > o$k.out; ./n$k > n$k.out; cmp o$k.out n$k.out || echo DIFF $k; done; echo FINISHED; cat o*.out | wc -l")
    res = DockerSandbox(memory="2g").run(scratch, cmd, timeout_s=600)
    assert res.rc == 0 and "FINISHED" in res.stdout, res.stdout + res.stderr[-2000:]
    assert "DIFF" not in res.stdout, res.stdout
    return res.stdout


@needs_docker
class TestConversionMatrices:
    def test_ternary_return_every_type_pair(self, scratch):
        """返回类型 x 两个分支的类型（含 0 / 1 / 255 / 65535 / -1 常量，两种顺序）：放行的组合改写前后值必须相同。"""
        rets = ["signed char", "short", "int", "unsigned", "long", "double"]
        consts = ["0", "1", "255", "65535", "-1"]
        funcs, calls = [], []

        def add(ret, params, ea, eb, arg_lists):
            name = f"f{len(funcs)}"
            funcs.append(f"static {ret} {name}({params}) {{ return c ? {ea} : {eb}; }}")
            calls.append([f'printf("%s %.17g\\n", "{name}", (double){name}({c}, {args}));'
                          for c in (0, 1) for args in arg_lists])

        for ret in rets:
            for ta in _MX_TYPES:
                for tb in _MX_TYPES:
                    add(ret, f"int c, {ta} a, {tb} b", "a", "b",
                        [f"({ta}){va}, ({tb}){vb}" for va in _MX_VALS[ta][:3] for vb in _MX_VALS[tb][:3]])
                for k in consts:
                    add(ret, f"int c, {ta} a", "a", k, [f"({ta}){va}" for va in _MX_VALS[ta][:3]])
                    add(ret, f"int c, {ta} a", k, "a", [f"({ta}){va}" for va in _MX_VALS[ta][:3]])
        pairs, done, skipped = [], 0, 0
        for src in _shards(funcs, calls):
            new, stats = st.transform_source(src, ["-std=c99"], 0, ratio=1.0, kinds=["ternary_return"])
            assert parse_errors(new, ["-std=c99"]) == [] and stats["files_skipped"] == {}
            done += stats["ternary_return"]
            skipped += stats["skipped"]["ternary_return"].get("type_mismatch", 0)
            pairs.append((src, new))
        assert done > 500 and skipped > 300       # 既有大量放行的组合，也有「-1 与 unsigned」这类被拦下的
        out = _compare_in_sandbox(scratch, pairs)
        assert int(out.split()[-1]) > 10000

    def test_compound_assign_every_type_and_operator(self, scratch):
        """左值类型 x 右值类型 x 十种运算符：x op= e 与 x = x op (e) 在边界值上结果相同（-fwrapv 避开有符号溢出）。"""
        floats = {"float", "double"}
        shifts = ["0", "1", "2", "7"]
        funcs, calls = [], []
        for op in ["+=", "-=", "*=", "/=", "%=", "<<=", ">>=", "&=", "^=", "|="]:
            for tl in _MX_TYPES:
                for tr in _MX_TYPES:
                    if op in ("%=", "<<=", ">>=", "&=", "^=", "|=") and (tl in floats or tr in floats):
                        continue
                    name = f"g{len(funcs)}"
                    funcs.append(f"static {tl} {name}({tl} x, {tr} e) {{ x {op} e; return x; }}")
                    rvals = shifts if op in ("<<=", ">>=") else (_MX_NONZERO[tr] if op in ("/=", "%=") else _MX_VALS[tr])
                    calls.append([f'printf("%s %.17g\\n", "{name}", (double){name}(({tl}){xv}, ({tr}){ev}));'
                                  for xv in _MX_VALS[tl] for ev in rvals])
        pairs, done = [], 0
        for src in _shards(funcs, calls):
            new, stats = st.transform_source(src, ["-std=c99"], 0, ratio=1.0, kinds=["compound_assign"])
            assert parse_errors(new, ["-std=c99"]) == [] and stats["files_skipped"] == {}
            done += stats["compound_assign"]
            pairs.append((src, new))
        assert done == len(funcs) > 500
        out = _compare_in_sandbox(scratch, pairs, flags="-fwrapv")
        assert int(out.split()[-1]) > 5000


def _real_corpus(name: str):
    return corpus.load_spec(name), require_real_corpus(name)


@needs_docker
class TestRealCorpora:
    """cjson / parson / tinyexpr：默认比例与全选两种强度，变换后项目自带测试必须通过。"""

    @pytest.mark.parametrize("name", ["cjson", "parson", "tinyexpr"])
    @pytest.mark.parametrize("seed,ratio", [(1, 0.5), (7, 1.0)])
    def test_corpus_tests_pass_after_l2(self, scratch, name, seed, ratio):
        spec, src = _real_corpus(name)
        root = corpus.materialize(src, scratch / name)
        stats = st.transform_structure(spec, root, seed, ratio=ratio)
        assert stats["total"] > 0 and stats["files_skipped"] == {}
        assert not any(stats["rolled_back"].values()) and stats["overlap_dropped"] == 0
        assert tree_errors(spec, root) == {}
        ok, res = sandbox_run(spec, root)
        assert ok, res.stdout[-1000:] + res.stderr[-2000:]
