"""bench.obfuscate.symbols（L1 符号混淆）与 bench.obfuscate.verify 的测试。

* 不需要 docker 的部分：收集（实体种类、系统头拼写 / 关键字 / 宏字符串化的排除）、改名表双射、
  token 级替换（字符串字面量、注释、#include 行不变，#define 行被改）、往返还原、回滚、verify 的各条失败路径。
* 需要 docker 的部分（gcc:13 镜像不在就整组跳过）：mini 夹具与「坑集合」小项目改名后在沙箱里编译并通过自带测试；
  strict 模式；obfuscate() 里「非 strict 失败后退回 strict」的自动回退；真实语料（corpus/src 里已 prepare 好的 9 个）L1 + verify。
  环境变量 ``BENCH_REAL_CORPORA``：默认跑全部九个语料，``quick`` 只跑 cjson、parson、sds、jsmn 四个，``0`` 全部跳过。

语料与改名后的副本只在 docker 沙箱里构建 / 运行；临时文件放在项目的 tmp/ 下，测试结束即清理。
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path

import pytest

from bench import corpus
from bench.corpus import CorpusSpec
from bench.extract import error_diagnostics, extract_functions, parse_file
from bench.obfuscate import obfuscate
from bench.obfuscate import symbols as S
from bench.obfuscate.symbols import (
    IdentifierSet,
    RenameError,
    apply_renames,
    build_rename_map,
    collect_project_identifiers,
    obfuscate_symbols,
)
from bench.obfuscate.verify import VerifyResult, relocate_targets, verify
from bench.sandbox import DockerSandbox, RunResult
from tests.helpers import MINI, ROOT, needs_docker, require_real_corpus

_REAL_MODE = os.environ.get("BENCH_REAL_CORPORA", "all")
REAL_QUICK = ["cjson", "parson", "sds", "jsmn"]
REAL_ALL = REAL_QUICK + ["tinyexpr", "mpc", "tiny-regex-c", "inih", "vec"]
REAL_CORPORA = [] if _REAL_MODE == "0" else (REAL_QUICK if _REAL_MODE == "quick" else REAL_ALL)

SEED = 20261027

# --------------------------------------------------------------------------- 「坑集合」小项目

TRICKY_FILES = {
    "tricky.h": r'''#ifndef TRICKY_H
#define TRICKY_H

#include <stddef.h>
#include <stdio.h>

#define TR_LIMIT 4
#define DECL_COUNTER(name) int name##_count; static int name##_next(void) { return ++name##_count; }
#define CALL_IMPL(f) f##_impl(3)
#define STR(x) #x
#define XSTR(x) STR(x)
#define TR_FIELD(n) tr_##n

struct counter {
    int tr_count;
};

struct node {
    int index;
    int stat;
    struct node *next;
};
typedef struct node node_t;

struct stamp {
    int tm_year;
    int tm_mon;
};

extern int tr_extern_only(int v);
int tr_sum(const node_t *list);
int tr_helper_impl(int v);
int tr_clock(void);
int tr_local_time(void);
int tr_format(const char *fmt, ...);
int tr_abs_user(int a);
int widget_total(void);
int tr_from_makefile(void);
int tr_stamp_sum(const struct stamp *st);
long tr_implicit_call(long v);
long tr_labs_shadow(void);
int tr_names_like_new(void);
const char *tr_label(void);
int tr_probe(void);
int tr_asm_out(int v);
int tr_attr_size(void);
int tr_pragma_size(void);
int tr_assert_noop(void);
const char *tr_strerr(int code);
int tr_counter_total(const struct counter *c);

#endif /* TRICKY_H */
''',
    "tricky.c": r'''/* tricky.c：专门给符号混淆挖坑的一个文件 */
#define _GNU_SOURCE
#define NDEBUG
#include <assert.h>
#include <string.h>
#include <time.h>
#include "tricky.h"

#ifndef TR_FROM_MAKEFILE
#define TR_FROM_MAKEFILE 0
#endif

#pragma GCC diagnostic ignored "-Wunused-function"
#pragma pack(push, 1)
struct packed_pair { char a; int b; };
#pragma pack(pop)

DECL_COUNTER(widget)

static int h = 5;
int stdio = 2;
static int push = 1;
static int packed = 1;
static int tr_ = 3;
static int unused __attribute__((unused)) = 3;
struct __attribute__((packed)) attr_pair { char a; int b; };

int tr_format(const char *fmt, ...) __attribute__((format(printf, 1, 2)));

int tr_format(const char *fmt, ...)
{
    return fmt != NULL ? (int)strlen(fmt) : -1;
}

int tr_sum(const node_t *list)
{
    int total = 0;
    int index = 0;

    while (list) {
        total += list->index + h + stdio * 0;
        list = list->next;
        index++;
    }
    return total + index;
}

int tr_helper_impl(int v)
{
    return v * TR_LIMIT + push;
}

int tr_clock(void)
{
    long now = (long)time(NULL);

    return now >= 0 ? 7 : 8;
}

int tr_local_time(void)
{
    int time = 7;

    return time;
}

int tr_abs_user(int a)
{
    int abs = 2;
    return a + abs;
}

int widget_total(void)
{
    int sum = 0;
    sum += widget_next();
    sum += widget_next();
    sum += widget_count;
    sum += CALL_IMPL(tr_helper);
    return sum;
}

int tr_from_makefile(void)
{
    return TR_FROM_MAKEFILE;
}

int tr_stamp_sum(const struct stamp *st)
{
    int fail_count = 0;

    if (st == NULL) {
        goto fail;
    }
    return st->tm_year + st->tm_mon;
fail:
    fail_count++;
    return -fail_count;
}

long tr_implicit_call(long v)
{
    return labs(v);            /* 没有 include <stdlib.h>：labs 是隐式声明 */
}

long tr_labs_shadow(void)
{
    long labs = 5;

    return labs;
}

int tr_names_like_new(void)
{
    int v_1 = 1;
    int f_1 = 2;

    return v_1 + f_1;
}

const char *tr_label(void)
{
    return "tr_sum is not tr_label";
}

int tr_attr_size(void)
{
    return (int)sizeof(struct attr_pair) + packed - 1;          /* packed 既是变量名又是属性名：5 */
}

int tr_pragma_size(void)
{
    return (int)sizeof(struct packed_pair) + push - 1;          /* push 既是变量名又是 pragma 参数：5 */
}

int tr_assert_noop(void)
{
    int x = 0;

    assert(x == 1);                                             /* NDEBUG 已定义，断言不生效 */
    return x + 7;
}

const char *tr_strerr(int code)
{
    static char buf[64];

    return strerror_r(code, buf, sizeof buf);                   /* _GNU_SOURCE 下返回 char *，否则（XSI 版）返回 int */
}

int tr_counter_total(const struct counter *c)
{
    return c->TR_FIELD(count) + tr_;                            /* tr_ ## count 拼出字段名 tr_count */
}

#if defined(__has_include)
#if __has_include(<stdint.h>)
#define TR_HAVE_STDINT 1
#endif
#endif
#if defined(__has_attribute)
#if __has_attribute(unused)
#define TR_ATTR_OK 1
#endif
#endif
#ifndef TR_HAVE_STDINT
#define TR_HAVE_STDINT 0
#endif
#ifndef TR_ATTR_OK
#define TR_ATTR_OK 0
#endif

int tr_probe(void)
{
    return TR_HAVE_STDINT + TR_ATTR_OK;
}

int tr_asm_out(int v)
{
    int out = v;

    __asm__ volatile ("/* %[out] */" : [out] "+r"(out));
    return out + 1;
}

#if 0
static int dead_branch(const node_t *list)
{
    return tr_sum(list) + TR_LIMIT;
}
#endif
''',
    "main.c": r'''#include <stdio.h>
#include <string.h>
#include "tricky.h"

int main(void)
{
    node_t b = { 10, 0, NULL };
    node_t a = { 1, 0, &b };
    struct stamp st = { 2020, 5 };
    int fails = 0;

    if (tr_sum(&a) != (1 + 5) + (10 + 5) + 2) { printf("tr_sum bad: %d\n", tr_sum(&a)); fails++; }
    if (tr_helper_impl(2) != 2 * 4 + 1) { printf("helper bad\n"); fails++; }
    if (tr_clock() != 7) { printf("clock bad\n"); fails++; }
    if (tr_local_time() != 7) { printf("local time bad\n"); fails++; }
    if (tr_abs_user(5) != 7) { printf("abs bad\n"); fails++; }
    if (widget_total() != 1 + 2 + 2 + 13) { printf("widget bad: %d\n", widget_total()); fails++; }
    if (tr_from_makefile() != 1) { printf("makefile macro bad\n"); fails++; }
    if (tr_stamp_sum(&st) != 2025) { printf("stamp bad\n"); fails++; }
    if (tr_stamp_sum(NULL) != -1) { printf("stamp null bad\n"); fails++; }
    if (tr_extern_only(4) != 8) { printf("extern bad\n"); fails++; }
    if (tr_format("abc") != 3) { printf("format bad\n"); fails++; }
    if (tr_implicit_call(-4) != 4) { printf("labs bad\n"); fails++; }
    if (tr_labs_shadow() != 5) { printf("labs shadow bad\n"); fails++; }
    if (tr_names_like_new() != 3) { printf("names bad\n"); fails++; }
    if (strcmp(tr_label(), "tr_sum is not tr_label") != 0) { printf("label bad\n"); fails++; }
    if (tr_probe() != 2) { printf("probe bad: %d\n", tr_probe()); fails++; }
    if (tr_asm_out(4) != 5) { printf("asm bad\n"); fails++; }
    if (tr_attr_size() != 5) { printf("attr size bad: %d\n", tr_attr_size()); fails++; }
    if (tr_pragma_size() != 5) { printf("pragma size bad: %d\n", tr_pragma_size()); fails++; }
    if (tr_assert_noop() != 7) { printf("assert bad\n"); fails++; }
    if (strlen(tr_strerr(2)) == 0) { printf("strerror bad\n"); fails++; }
    { struct counter cn = { 4 }; if (tr_counter_total(&cn) != 7) { printf("counter bad\n"); fails++; } }
    printf("%d failures\n", fails);
    return fails == 0 ? 0 : 1;
}
''',
    # 不在 project_files 里：tr_extern_only 的定义在项目外，链接时才出现
    "ext.c": '''int tr_extern_only(int v)
{
    return v * 2;
}
''',
    # Makefile 里用 -D 传入的宏，spec 的命令里看不到
    "Makefile": '''CC = gcc
CFLAGS += -std=c99 -Wall -DTR_FROM_MAKEFILE=1

test: tricky_test
\t./tricky_test

tricky_test: tricky.c main.c ext.c tricky.h
\t$(CC) $(CFLAGS) -o $@ tricky.c main.c ext.c

.PHONY: test
''',
}
TRICKY_SPEC = {
    "name": "tricky", "url": "", "commit": "", "license": "MIT",
    "project_files": ["tricky.h", "tricky.c", "main.c"], "target_files": ["tricky.c"],
    "cflags": ["-std=c99", "-I."], "build_cmd": "make test", "test_cmd": None,
    "pass_criteria": {"exit_code": 0, "stdout_regex": "0 failures"}, "coverage": {}, "timeout_s": 60, "notes": "",
}

# 字符串化比对：非 strict 会把 s_func 改名，STR(s_func) 就不再是 "s_func"
STRMINI_FILES = {
    "s.h": '''#ifndef S_H
#define S_H
#define STR(x) #x
int s_func(int v);
#endif
''',
    "s.c": '''#include "s.h"

int s_func(int v)
{
    int r = v;

    r += 1;
    return r;
}
''',
    "main.c": '''#include <stdio.h>
#include <string.h>
#include "s.h"

int main(void)
{
    if (strcmp(STR(s_func), "s_func") != 0) {
        printf("stringify mismatch: %s\\n", STR(s_func));
        return 1;
    }
    if (s_func(1) != 2) {
        printf("s_func bad\\n");
        return 1;
    }
    printf("0 failures\\n");
    return 0;
}
''',
}
STRMINI_SPEC = {
    "name": "strmini", "url": "", "commit": "", "license": "MIT",
    "project_files": ["s.h", "s.c", "main.c"], "target_files": ["s.c"],
    "cflags": ["-std=c99", "-I."],
    "build_cmd": "gcc -std=c99 -O0 -Wall -o strmini_test s.c main.c && ./strmini_test",
    "test_cmd": None, "pass_criteria": {"exit_code": 0, "stdout_regex": "0 failures"}, "coverage": {},
    "timeout_s": 60, "notes": "",
}

# 复杂语法压力项目：X-macro + ## 拼接枚举名、带名可变参数宏、语句表达式、标签取址、_Generic、K&R 定义、
# 指定初始化器、复合字面量、位域、匿名联合
STRESS_FILES = {
    "stress.h": r'''#ifndef STRESS_H
#define STRESS_H
#include <stddef.h>

#define COLOR_LIST(X) X(RED, 1) X(GREEN, 2) X(BLUE, 4)
#define AS_ENUM(name, value) COLOR_##name = value,
#define AS_NAME(name, value) { #name, COLOR_##name },

enum color_kind { COLOR_LIST(AS_ENUM) COLOR_END };

struct color_entry { const char *label; int code; };

typedef struct widget widget_t;
struct widget {
    int id;
    union { int as_int; float as_float; } payload;
    struct { unsigned flag_a : 1; unsigned flag_b : 3; } bits;
    int (*callback)(int arg);
    widget_t *next;
};

#define GET(obj, member) ((obj)->member)
#define SWAP_INT(p, q) do { int swap_tmp = *(p); *(p) = *(q); *(q) = swap_tmp; } while (0)
#define LOG_FMT(fmt, args...) stress_log(fmt, ##args)
#define CONCAT3(a, b, c) a##b##c
#define MAX_OF(a, b) ({ int ma = (a); int mb = (b); ma > mb ? ma : mb; })

int stress_log(const char *fmt, ...);
int stress_apply(widget_t *w, int v);
int stress_sum_ids(const widget_t *head);
int stress_color_code(const char *label);
int stress_old_style();
int stress_dispatch(int idx, int v);
int stress_generic(int v);
int stress_label_value(int n);
int stress_compound(int x, int y);
int stress_bits(void);
#endif
''',
    "stress.c": r'''#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include "stress.h"

static const struct color_entry color_table[] = { COLOR_LIST(AS_NAME) { 0, 0 } };

int stress_log(const char *fmt, ...)
{
    va_list ap;
    int n;

    va_start(ap, fmt);
    n = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    return n;
}

static int cb_double(int arg) { return arg * 2; }
static int cb_triple(int arg) { return arg * 3; }
static int cb_negate(int arg) { return -arg; }

static int (*const dispatch_table[])(int) = {
    [0] = cb_double,
    [2] = cb_triple,
    [1] = cb_negate,
};

int stress_apply(widget_t *w, int v)
{
    int r = GET(w, callback)(v);

    r += LOG_FMT("%d", r) * 0;
    return r + GET(w, id);
}

int stress_sum_ids(const widget_t *head)
{
    int total = 0;

    for (; head != NULL; head = head->next) {
        total += head->id;
    }
    return total;
}

int stress_color_code(const char *label)
{
    const struct color_entry *e;

    for (e = color_table; e->label != NULL; e++) {
        if (strcmp(e->label, label) == 0) {
            return e->code;
        }
    }
    return COLOR_END;
}

int stress_old_style(a, b)
    int a;
    int b;
{
    int swap_a = a, swap_b = b;

    SWAP_INT(&swap_a, &swap_b);
    return swap_a * 10 + swap_b;
}

int stress_dispatch(int idx, int v)
{
    return dispatch_table[idx](v);
}

int stress_generic(int v)
{
    return _Generic(v, int: v + 1, long: 0, default: -1);
}

int stress_label_value(int n)
{
    void *target = n > 0 ? &&positive : &&other;

    goto *target;
positive:
    return CONCAT3(1, 0, 0) + n;
other:
    return -1;
}

int stress_compound(int x, int y)
{
    widget_t w = (widget_t){ .id = x, .callback = cb_double, .next = NULL };
    int m = MAX_OF(x, y);

    return stress_apply(&w, m);
}

int stress_bits(void)
{
    widget_t w;
    int RED;                   /* 与 X-macro 列表项 RED 同名：列表项会被 AS_ENUM / AS_NAME 拼成 COLOR_RED */

    memset(&w, 0, sizeof w);
    w.bits.flag_a = 1;
    w.bits.flag_b = 5;
    w.payload.as_int = 7;
    RED = w.bits.flag_a;
    return RED * 100 + w.bits.flag_b * 10 + w.payload.as_int;
}
''',
    "main.c": r'''#include <stdio.h>
#include "stress.h"

static int cb_plus_one(int arg) { return arg + 1; }

int main(void)
{
    widget_t b = { 5, { 0 }, { 0, 0 }, 0, 0 };
    widget_t a = { 3, { 0 }, { 0, 0 }, cb_plus_one, &b };
    int fails = 0;

    if (stress_apply(&a, 4) != 5 + 3) { printf("apply bad\n"); fails++; }
    if (stress_sum_ids(&a) != 8) { printf("sum bad\n"); fails++; }
    if (stress_color_code("RED") != 1 || stress_color_code("GREEN") != 2 || stress_color_code("nope") != 5) {
        printf("color bad\n"); fails++;
    }
    if (stress_old_style(1, 2) != 21) { printf("old bad\n"); fails++; }
    if (stress_dispatch(0, 4) != 8 || stress_dispatch(1, 4) != -4 || stress_dispatch(2, 4) != 12) {
        printf("dispatch bad\n"); fails++;
    }
    if (stress_generic(4) != 5) { printf("generic bad\n"); fails++; }
    if (stress_label_value(5) != 105 || stress_label_value(-5) != -1) { printf("label bad\n"); fails++; }
    if (stress_compound(2, 7) != 14 + 2) { printf("compound bad\n"); fails++; }
    if (stress_bits() != 100 + 50 + 7) { printf("bits bad\n"); fails++; }
    printf("%d failures\n", fails);
    return fails == 0 ? 0 : 1;
}
''',
}
STRESS_SPEC = {
    "name": "stress", "url": "", "commit": "", "license": "MIT",
    "project_files": ["stress.h", "stress.c", "main.c"], "target_files": ["stress.c"],
    "cflags": ["-std=gnu99", "-I."],
    "build_cmd": "gcc -std=gnu99 -O0 -Wall -o stress_test stress.c main.c && ./stress_test",
    "test_cmd": None, "pass_criteria": {"exit_code": 0, "stdout_regex": "0 failures"}, "coverage": {},
    "timeout_s": 60, "notes": "",
}

# 条件编译：fx_func 只在 -DFEATURE_X 下才有定义，-D 写在 build_cmd 里（spec 能看到），基础配置解析看不见它的定义
UNION_FILES = {
    "u.h": '''#ifndef U_H
#define U_H
int fx_func(int v);
int plain(int v);
#endif
''',
    "u.c": '''#include "u.h"

#ifdef FEATURE_X
int fx_func(int v)
{
    int local_x = v * 2;

    return local_x;
}
#endif

int plain(int v)
{
    return v + 1;
}
''',
    "main.c": '''#include <stdio.h>
#include "u.h"

int main(void)
{
#ifdef FEATURE_X
    int r = fx_func(2);
#else
    int r = 4;
#endif
    printf("%d %d\\n", r, plain(1));
    return 0;
}
''',
}
UNION_SPEC = {
    "name": "unionmini", "url": "", "commit": "", "license": "MIT",
    "project_files": ["u.h", "u.c", "main.c"], "target_files": ["u.c"],
    "cflags": ["-std=c99", "-I."],
    "build_cmd": "gcc -std=c99 -O0 -DFEATURE_X=1 -o ut u.c main.c && ./ut",
    "test_cmd": None, "pass_criteria": {"exit_code": 0, "stdout_regex": "^4 2$"}, "coverage": {},
    "timeout_s": 60, "notes": "",
}

# vendored 目录：第三方头里声明的函数、被它引用的项目宏都不能改。
# vend.h 的内联函数在引用方（app.c）的翻译单元里编译，APP_FEATURE 在 app.h 里定义：改名就会让条件编译选错分支
VENDOR_FILES = {
    "vendor/vend.h": '''#ifndef VEND_H
#define VEND_H
#ifdef APP_FEATURE
#define VEND_MODE 1
#else
#define VEND_MODE 2
#endif
#define VEND_CALL(x) vendor_scale(x, app_factor)
static inline int vendor_scale(int v, int k)
{
    return v * k + VEND_MODE;
}
static inline int vendor_other(int v)
{
    return v - 1;
}
#endif
''',
    "app.h": '''#ifndef APP_H
#define APP_H
#define APP_FEATURE
extern int app_factor;
int app_run(int v);
int app_direct(int v);
int app_shadow(int vendor_other);
#endif
''',
    "app.c": '''#include "app.h"
#include "vendor/vend.h"

int app_factor = 3;

int app_run(int v)
{
    int local_value = v + 1;

    return VEND_CALL(local_value);
}

/* 直接按名字调用 vendored 里的函数；另一个函数的形参恰好同名 */
int app_direct(int v)
{
    return vendor_other(v);
}

int app_shadow(int vendor_other)
{
    return vendor_other + 1;
}
''',
    "main.c": '''#include <stdio.h>
#include "app.h"

int main(void)
{
    printf("%d\\n", app_run(2));
    return 0;
}
''',
}
VENDOR_SPEC = {
    "name": "vendtest", "url": "", "commit": "", "license": "MIT",
    "project_files": ["app.h", "app.c", "main.c", "vendor/*.h"], "target_files": ["app.c"],
    "cflags": ["-std=c99", "-I."],
    "build_cmd": "gcc -std=c99 -O0 -I. -o vt app.c main.c && ./vt",
    "test_cmd": None, "pass_criteria": {"exit_code": 0, "stdout_regex": "^10$"}, "coverage": {},
    "timeout_s": 60, "notes": "", "vendored": ["vendor"], "keep_names": [],
}


# --------------------------------------------------------------------------- 夹具与工具


@pytest.fixture(scope="module")
def scratch_root():
    base = ROOT / "tmp" / "pytest"
    base.mkdir(parents=True, exist_ok=True)
    d = base / uuid.uuid4().hex[:12]
    d.mkdir()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture()
def workdir():
    base = ROOT / "tmp" / "pytest"
    base.mkdir(parents=True, exist_ok=True)
    d = base / uuid.uuid4().hex[:12]
    d.mkdir()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture(scope="module")
def sandbox() -> DockerSandbox:
    return DockerSandbox(timeout_s=120)


@pytest.fixture(scope="module")
def mini_ids(mini_spec) -> IdentifierSet:
    return collect_project_identifiers(mini_spec, MINI)


@pytest.fixture(scope="module")
def mini_l1(scratch_root, mini_spec):
    """mini 做一次 L1（模块内复用）：返回 (副本目录, SymbolsResult)。"""
    dest = corpus.materialize(MINI, scratch_root / "mini_l1")
    res = obfuscate_symbols(mini_spec, dest, SEED)
    return dest, res


def write_project(dest: Path, files: dict[str, str]) -> Path:
    for rel, text in files.items():
        p = dest / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return dest


def make_spec(data: dict) -> CorpusSpec:
    return CorpusSpec.from_dict(dict(data))


@pytest.fixture(scope="module")
def tricky(scratch_root):
    src = write_project(scratch_root / "tricky_src", TRICKY_FILES)
    return make_spec(TRICKY_SPEC), src


@pytest.fixture(scope="module")
def tricky_ids(tricky) -> IdentifierSet:
    spec, src = tricky
    return collect_project_identifiers(spec, src)


def tree_digest(root: Path, rels: list[str]) -> str:
    h = hashlib.sha256()
    for rel in rels:
        h.update(rel.encode())
        h.update((Path(root) / rel).read_bytes())
    return h.hexdigest()


def lex_tree(spec, root: Path, rels: list[str]) -> dict[str, list[tuple[str, str, bytes]]]:
    """每个文件的 token 流：(kind, 拼写, 原始字节)。"""
    out = {}
    for rel in rels:
        fi = S.load_file_info(spec, root, rel)
        out[rel] = [(t.kind, t.text, fi.data[t.start:t.end]) for t in fi.toks]
    return out


def assert_token_equivalence(spec, src: Path, dest: Path, mapping: dict[str, str]) -> int:
    """改名前后的 token 流逐个对应：只有拼写在映射表里的标识符变了，其余字节完全一致。返回被改的 token 数。"""
    r_files, v_files = S.project_file_lists(spec, src)
    a = lex_tree(spec, src, r_files + v_files)
    b = lex_tree(spec, dest, r_files + v_files)
    changed = 0
    for rel in a:
        assert len(a[rel]) == len(b[rel]), f"{rel}: token 数变了"
        for (k1, t1, raw1), (k2, t2, raw2) in zip(a[rel], b[rel]):
            assert k1 == k2, rel
            if raw1 == raw2:
                continue
            assert k1 == "id" and t1 in mapping and t2 == mapping[t1], f"{rel}: {raw1!r} -> {raw2!r}"
            changed += 1
    return changed


class FakeSandbox:
    """不起 docker 的替身：按预设的 RunResult 返回，记录被调用时的工作目录内容。"""

    def __init__(self, result: RunResult) -> None:
        self.result = result
        self.calls: list[tuple[str, int]] = []

    def run(self, workdir, cmd, env=None, timeout_s=None):
        self.calls.append((cmd, len(list(Path(workdir).iterdir()))))
        return self.result


def ok_result(stdout: str = "117 checks, 0 failures\n") -> RunResult:
    return RunResult(rc=0, stdout=stdout, stderr="", duration_s=0.1, timed_out=False, signal=None)


# --------------------------------------------------------------------------- 收集


def test_collect_returns_set_subclass_with_metadata(mini_ids):
    assert isinstance(mini_ids, set)
    assert len(mini_ids) > 100
    assert set(mini_ids.kinds) == set(mini_ids)
    assert set(mini_ids.kinds.values()) <= {"func", "type", "var", "macro", "field"}
    assert mini_ids.stats["entities_renamed"] == len(mini_ids)
    assert mini_ids.reserved and mini_ids.excluded


def test_collect_covers_every_entity_kind(mini_ids):
    kinds = mini_ids.kinds
    expect = {
        "calc_apply": "func", "str_eq": "func",            # 函数（含 static）
        "g_calls": "var", "s_seed": "var",                  # 全局 / static 变量
        "a": "var", "total": "var",                         # 形参 / 局部变量
        "lo": "field", "items": "field",                    # 字段
        "point": "type", "rect": "type", "rounding": "type",  # struct / enum 标签
        "point_t": "type", "binop_fn": "type",              # typedef
        "OP_ADD": "var", "ROUND_UP": "var",                 # 枚举常量
        "fail": "var",                                      # 标签
        "MINI_MIN": "macro", "MINI_H": "macro",             # 宏（含头文件保护宏）
    }
    for name, kind in expect.items():
        assert name in mini_ids, name
        assert kinds[name] == kind, (name, kinds[name])


def test_system_header_spellings_are_excluded(mini_ids):
    # mini 的代码用到了 libc / 标准头里的函数、类型、宏：一个都不能进改名集合
    for name in ("strlen", "strcmp", "memset", "malloc", "free", "toupper", "printf", "size_t", "NULL", "INT_MAX",
                 "INT_MIN", "EOF"):
        assert name not in mini_ids, name


def test_keywords_main_and_reserved_prefix_excluded(mini_ids, tricky_ids):
    assert "main" not in mini_ids and "main" in mini_ids.excluded["entry"]
    assert not (set(mini_ids) & S.C_KEYWORDS)
    for name in ("_GNU_SOURCE", "__FILE__"):
        assert name not in tricky_ids
    assert "_GNU_SOURCE" in tricky_ids.excluded["reserved_prefix"]
    assert "NDEBUG" in tricky_ids.excluded["known_external"]


def test_stringify_macro_spellings_excluded_and_recorded(mini_ids):
    # test_main.c 的 CHECK(cond) 带 #cond：宏名、宏体里的全局计数器都不改，并记进 excluded_stringify
    assert set(mini_ids.stats["excluded_stringify"]) == {"CHECK", "g_checks", "g_failures"}
    for name in ("CHECK", "g_checks", "g_failures"):
        assert name not in mini_ids
    # 默认模式下调用实参里的函数照常改名，但要如实记录哪些拼写受字符串化影响
    assert "calc_apply" in mini_ids
    assert "calc_apply" in mini_ids.stats["stringified_names_renamed"]


def test_collect_stats_are_consistent(mini_ids):
    st = mini_ids.stats
    assert st["entities_declared"] == st["entities_renamed"] + st["entities_excluded"]
    assert sum(st["excluded_by_reason"].values()) == st["entities_excluded"]
    assert sum(st["renamed_by_kind"].values()) == st["entities_renamed"]
    json.dumps(st)                                          # 可序列化


def test_collect_does_not_depend_on_working_directory(mini_spec, mini_ids, workdir, monkeypatch):
    monkeypatch.chdir(workdir)
    assert set(collect_project_identifiers(mini_spec, MINI)) == set(mini_ids)
    assert set(collect_project_identifiers(mini_spec, os.path.relpath(MINI, workdir))) == set(mini_ids)


def test_tricky_exclusions(tricky_ids):
    ex = tricky_ids.excluded
    # 系统头里声明过的拼写：函数 time / index、struct tm 的字段 tm_year / tm_mon
    assert {"time", "index", "tm_year", "tm_mon"} <= set(ex["external_decl"])
    # ## 拼出来的名字与它的实参、包含 #/## 的宏本身
    assert {"widget_count", "widget_next", "tr_helper_impl", "DECL_COUNTER", "CALL_IMPL", "STR"} <= set(
        tricky_ids.stats["excluded_stringify"])
    assert "XSTR" in tricky_ids                              # XSTR 自己没有 #，只是包装，可以改
    assert "tr_extern_only" in ex["no_definition"]           # 只有声明、定义在项目外
    assert "labs" in ex["implicit_decl"]                     # 没 include 就调用的 libc 函数
    assert "TR_FROM_MAKEFILE" in ex["cmdline_macro"]         # Makefile 里 -D 传入
    assert "push" in ex["pragma"]                            # #pragma pack(push, 1)
    assert {"tr_", "TR_FIELD", "tr_count"} <= set(tricky_ids.stats["excluded_stringify"])     # tr_ ## n 拼出 tr_count
    # 与 #include <stdio.h> 的 stdio / h 同名的变量、与属性名同名的变量照常改名
    for name in ("stdio", "h", "unused", "tr_sum", "tr_format", "fail", "stat"):
        assert name in tricky_ids, name


# --------------------------------------------------------------------------- 改名表


def test_build_rename_map_is_bijective_with_kind_prefixes(mini_ids):
    mapping = build_rename_map(mini_ids, SEED)
    assert set(mapping) == set(mini_ids)
    assert len(set(mapping.values())) == len(mapping)                    # 双射
    inverse = {v: k for k, v in mapping.items()}
    assert len(inverse) == len(mapping) and all(inverse[mapping[k]] == k for k in mapping)
    prefix = {"func": "f_", "type": "t_", "var": "v_", "macro": "M_", "field": "m_"}
    for old, new in mapping.items():
        assert new.startswith(prefix[mini_ids.kinds[old]]), (old, new)
        assert new[2:].isdigit()
    # 新名字不与任何旧名字、系统头拼写或源码里已有的标识符相同
    assert not (set(mapping.values()) & (set(mapping) | set(mini_ids.reserved)))


def test_rename_map_numbers_are_a_shuffled_permutation(mini_ids):
    mapping = build_rename_map(mini_ids, SEED)
    for kind, pre in (("func", "f_"), ("macro", "M_")):
        names = sorted(n for n in mini_ids if mini_ids.kinds[n] == kind)
        nums = sorted(int(mapping[n][2:]) for n in names)
        assert nums == list(range(1, len(names) + 1))                    # 1..K 的一个排列（没有撞上保留拼写时）
        in_order = [int(mapping[n][2:]) for n in names]
        assert in_order != sorted(in_order)                              # 且确实被打乱了


def test_rename_map_is_deterministic_and_seed_sensitive(mini_ids):
    m1 = build_rename_map(mini_ids, 1)
    assert m1 == build_rename_map(mini_ids, 1)
    shuffled = IdentifierSet(sorted(mini_ids, reverse=True))              # 迭代顺序不同的同一批名字
    shuffled.kinds, shuffled.reserved = mini_ids.kinds, mini_ids.reserved
    assert m1 == build_rename_map(shuffled, 1)
    m2 = build_rename_map(mini_ids, 2)
    assert set(m1) == set(m2) and m1 != m2


def test_rename_map_plain_set_and_reserved_names():
    plain = build_rename_map({"alpha", "beta", "GAMMA"}, 3)
    assert plain["GAMMA"].startswith("M_") and plain["alpha"].startswith("v_")
    blocked = {f"v_{i}" for i in range(1, 4)}
    m = build_rename_map({"alpha", "beta"}, 3, kinds={"alpha": "var", "beta": "var"}, reserved=blocked)
    assert not (set(m.values()) & blocked)
    assert all(v.startswith("v_") for v in m.values())


def test_new_names_skip_project_names_that_look_like_new_names(tricky, tricky_ids):
    # 项目自己就有叫 v_1 / f_1 的局部变量和函数：新名字绕开它们，旧的 v_1 / f_1 也被正常改掉
    mapping = build_rename_map(tricky_ids, SEED)
    assert "v_1" in mapping and "f_1" in mapping
    assert mapping["v_1"] != "v_1" and mapping["f_1"] != "f_1"
    assert not (set(mapping.values()) & set(mapping))


# --------------------------------------------------------------------------- 替换（token 级）


def test_apply_renames_changes_only_mapped_identifier_tokens(mini_spec, mini_l1):
    dest, res = mini_l1
    changed = assert_token_equivalence(mini_spec, MINI, dest, res.mapping)
    assert changed == sum(res.per_file.values()) == res.stats["tokens_replaced"] > 500


def test_string_literals_and_comments_are_unchanged(mini_spec, mini_l1):
    dest, res = mini_l1
    rels, _ = S.project_file_lists(mini_spec, MINI)
    for rel in rels:
        a = S.load_file_info(mini_spec, MINI, rel)
        b = S.load_file_info(mini_spec, dest, rel)
        lits_a = [a.data[t.start:t.end] for t in a.toks if t.kind in ("lit", "cmt")]
        lits_b = [b.data[t.start:t.end] for t in b.toks if t.kind in ("lit", "cmt")]
        assert lits_a == lits_b, rel
    text = (dest / "calc.c").read_text(encoding="utf-8")
    assert '"division by zero"' in text and '"bad op"' in text
    # 注释里的原名保留（设计约定：不碰注释）：mini.h 的文档注释提到 g_last_error，而它的声明已经改名
    header = (dest / "mini.h").read_text(encoding="utf-8")
    assert "并设置 g_last_error。" in header
    assert f"extern const char *{res.mapping['g_last_error']};" in header


def test_define_lines_and_include_guard_are_renamed(mini_spec, mini_l1):
    dest, res = mini_l1
    header = (dest / "mini.h").read_text(encoding="utf-8")
    m = res.mapping
    assert f"#define {m['MINI_MIN']}(" in header and "#define MINI_MIN" not in header
    assert f"#define {m['MINI_VERSION']} 3" in header
    assert f"#ifndef {m['MINI_H']}" in header and f"#define {m['MINI_H']}\n" in header
    # 带 # 的断言宏本身保持原样；宏体里引用的计数器也不改
    tests = (dest / "test_main.c").read_text(encoding="utf-8")
    assert "#define CHECK(cond)" in tests and "g_checks++" in tests
    # #include 行逐行不变（原始词法器会把 <stdlib.h> 切成 stdlib 和 h 两个标识符，而局部变量 h 恰好被改名了）
    rels, _ = S.project_file_lists(mini_spec, MINI)

    def include_lines(root):
        return {rel: [ln for ln in (root / rel).read_text(encoding="utf-8").splitlines()
                      if ln.lstrip().startswith("#include")] for rel in rels}

    assert include_lines(dest) == include_lines(MINI)
    assert "#include <stdlib.h>" in include_lines(dest)["util.c"]
    assert "h" in res.mapping and res.mapping["h"].startswith("v_")


def test_angle_include_pragma_and_attribute_names_are_protected(tricky, tricky_ids, workdir):
    spec, src = tricky
    dest = corpus.materialize(src, workdir / "t")
    mapping = build_rename_map(tricky_ids, SEED)
    apply_renames(spec, dest, mapping)
    header = (dest / "tricky.h").read_text(encoding="utf-8")
    code = (dest / "tricky.c").read_text(encoding="utf-8")
    assert "#include <stdio.h>" in header and "#include <stddef.h>" in header
    assert "#pragma pack(push, 1)" in code and '#pragma GCC diagnostic ignored "-Wunused-function"' in code
    assert "__attribute__((unused))" in code and "__attribute__((format(printf, 1, 2)))" in code
    # stdio / h / unused / format 这类和上面那些位置同名的变量照常被改
    assert f"int {mapping['stdio']} = 2;" in code and f"static int {mapping['h']} = 5;" in code
    assert f"static int {mapping['unused']} __attribute__((unused))" in code
    # 预处理探测函数的括号（<stdint.h>、属性名）和 asm 操作数的符号名不是项目标识符
    assert "__has_include(<stdint.h>)" in code and "__has_attribute(unused)" in code
    assert '__asm__ volatile ("/* %[out] */" : [out] "+r"(' in code
    assert f"int {mapping['out']} = {mapping['v']};" in code
    # #if 0 里的死代码按拼写一并改名；字符串字面量里的函数名不动
    assert "tr_sum(" not in code.split("#if 0")[1]
    assert '"tr_sum is not tr_label"' in code


def test_pasted_names_and_their_arguments_keep_their_spelling(tricky, tricky_ids, workdir):
    spec, src = tricky
    dest = corpus.materialize(src, workdir / "t")
    apply_renames(spec, dest, build_rename_map(tricky_ids, SEED))
    code = (dest / "tricky.c").read_text(encoding="utf-8")
    assert "DECL_COUNTER(widget)" in code and "widget_next()" in code and "widget_count" in code
    assert "CALL_IMPL(tr_helper)" in code
    assert "int tr_helper_impl(" in code


def test_roundtrip_with_inverse_mapping_restores_every_byte(mini_spec, mini_l1):
    dest, res = mini_l1
    inverse = {v: k for k, v in res.mapping.items()}
    rels, _ = S.project_file_lists(mini_spec, MINI)
    for rel in rels:
        fi = S.load_file_info(mini_spec, dest, rel)
        back, _n = S.rewrite_bytes(fi, inverse)
        assert back == (MINI / rel).read_bytes(), rel


def test_same_seed_same_tree_and_different_seed_differs(mini_spec, workdir):
    rels, _ = S.project_file_lists(mini_spec, MINI)
    digests = []
    for i, seed in enumerate((11, 11, 12)):
        d = corpus.materialize(MINI, workdir / f"d{i}")
        obfuscate_symbols(mini_spec, d, seed)
        digests.append(tree_digest(d, rels))
    assert digests[0] == digests[1] != digests[2]


def test_target_files_reparse_without_errors_after_rename(mini_spec, mini_l1):
    dest, _res = mini_l1
    for rel in ("calc.c", "util.c"):
        assert error_diagnostics(parse_file(mini_spec, MINI, rel)) == []
        assert error_diagnostics(parse_file(mini_spec, dest, rel)) == []


def test_obfuscate_symbols_stats_shape(mini_l1):
    _dest, res = mini_l1
    st = res.stats
    assert st["level"] == "L1" and st["seed"] == SEED and st["reparse_errors"] == 0
    assert st["files_total"] == 4 and st["files_changed"] == 4
    assert st["entities_renamed"] == len(res.mapping)
    assert sum(st["renamed_by_prefix"].values()) == len(res.mapping)
    assert set(st["renamed_by_prefix"]) == {"M", "f", "m", "t", "v"}
    json.dumps(st)


def test_rename_error_restores_all_files(mini_spec, workdir):
    dest = corpus.materialize(MINI, workdir / "rb")
    rels, _ = S.project_file_lists(mini_spec, dest)
    before = tree_digest(dest, rels)
    # 把 size_t 改成没人声明的名字：util.c 里用到它的地方全部变成未知类型，目标文件出现新的 error
    with pytest.raises(RenameError) as exc:
        apply_renames(mini_spec, dest, {"size_t": "zzz_unknown_t"})
    assert exc.value.diagnostics
    assert tree_digest(dest, rels) == before


def test_mapping_validation(mini_spec, workdir):
    dest = corpus.materialize(MINI, workdir / "mv")
    rels, _ = S.project_file_lists(mini_spec, dest)
    before = tree_digest(dest, rels)
    with pytest.raises(ValueError):
        apply_renames(mini_spec, dest, {"calc_apply": "zz_1", "calc_gcd": "zz_1"})      # 不是双射
    with pytest.raises(ValueError):
        apply_renames(mini_spec, dest, {"calc_apply": "while"})                          # 关键字
    with pytest.raises(ValueError):
        apply_renames(mini_spec, dest, {"calc_apply": "calc_gcd", "calc_gcd": "zz_2"})  # 新旧名字重叠
    with pytest.raises(ValueError):
        apply_renames(mini_spec, dest, {"calc_apply": "sum_array"})                      # 与源码里已有的标识符撞名
    with pytest.raises(ValueError):
        apply_renames(mini_spec, dest, {"calc_apply": "not an identifier"})
    assert tree_digest(dest, rels) == before
    assert set(apply_renames(mini_spec, dest, {}).values()) == {0}


def test_symlinked_source_is_replaced_not_written_through(mini_spec, workdir):
    dest = corpus.materialize(MINI, workdir / "sl")
    outside = workdir / "outside_util.c"
    shutil.copy2(dest / "util.c", outside)
    (dest / "util.c").unlink()
    (dest / "util.c").symlink_to(outside)
    original = outside.read_bytes()
    ids = collect_project_identifiers(mini_spec, dest)
    apply_renames(mini_spec, dest, build_rename_map(ids, 5))
    assert outside.read_bytes() == original                              # 树外的文件没被改
    assert not (dest / "util.c").is_symlink()


def test_vendored_declarations_and_their_users_are_not_renamed(workdir):
    spec = make_spec(VENDOR_SPEC)
    src = write_project(workdir / "v", VENDOR_FILES)
    ids = collect_project_identifiers(spec, src)
    # vendored 里声明的函数 / 宏不改；vendored 的宏体和条件编译里用到的项目名字（app_factor、APP_FEATURE）也不能改
    for name in ("vendor_scale", "vendor_other", "VEND_CALL", "VEND_MODE", "VEND_H", "app_factor", "APP_FEATURE"):
        assert name not in ids, name
    assert "vendor_other" in ids.excluded["external_decl"]       # 项目里有同名形参，但它是 vendored 声明过的拼写
    assert {"app_factor", "APP_FEATURE"} <= set(ids.excluded["vendored_use"])   # vendored 宏体 / 条件编译里用到
    assert {"app_run", "app_direct", "local_value", "APP_H"} <= set(ids)
    before = {rel: (src / rel).read_bytes() for rel in ("vendor/vend.h",)}
    apply_renames(spec, src, build_rename_map(ids, 9))
    assert {rel: (src / rel).read_bytes() for rel in before} == before   # vendored 文件一字节都没动
    assert "app_run" not in (src / "app.c").read_text(encoding="utf-8").replace("/* app_run */", "")


@needs_docker
def test_vendored_project_still_builds_and_gives_the_same_answer(sandbox, workdir):
    spec = make_spec(VENDOR_SPEC)
    src = write_project(workdir / "v", VENDOR_FILES)
    targets = extract_functions(spec, src)
    assert verify(spec, src, targets, {}, sandbox, work_root=workdir / "work").ok
    res = obfuscate_symbols(spec, src, SEED)
    vr = verify(spec, src, targets, res.mapping, sandbox, work_root=workdir / "work")
    assert vr.ok, vr.stdout_tail                    # 输出仍是 10：APP_FEATURE / app_factor 没被改，vendored 代码照常解析


# --------------------------------------------------------------------------- verify（不用 docker）


@pytest.fixture()
def mini_obf(mini_spec, mini_l1, workdir):
    """每个测试一份独立的混淆副本（verify 的失败路径会改文件）。"""
    dest, res = mini_l1
    return corpus.materialize(dest, workdir / "obf"), res.mapping


def test_verify_ok_with_fake_sandbox_and_runs_in_a_copy(mini_spec, mini_obf, workdir):
    obf, mapping = mini_obf
    targets = extract_functions(mini_spec, MINI)
    rels, _ = S.project_file_lists(mini_spec, obf)
    before = tree_digest(obf, rels)
    fake = FakeSandbox(ok_result())
    vr = verify(mini_spec, obf, targets, mapping, fake, work_root=workdir / "work")
    assert isinstance(vr, VerifyResult)
    assert vr.ok and vr.build_rc == 0 and vr.tests_passed
    assert vr.relocated == len(targets) == 40 and vr.missing == []
    assert fake.calls and fake.calls[0][0] == mini_spec.run_command()
    assert tree_digest(obf, rels) == before                              # 构建在副本里跑，不碰 obf_root
    assert not (workdir / "work").exists() or not list((workdir / "work").iterdir())     # 副本已删
    json.dumps(vr.to_dict())


def test_verify_reports_missing_targets(mini_spec, mini_obf, workdir):
    obf, mapping = mini_obf
    targets = extract_functions(mini_spec, MINI)
    # 不用映射表：原名在改名后的副本里根本找不到
    vr = verify(mini_spec, obf, targets, {}, FakeSandbox(ok_result()), work_root=workdir / "work")
    renamed_targets = [t.target_id for t in targets if t.name in mapping]
    assert not vr.ok and vr.tests_passed
    assert sorted(vr.missing) == sorted(renamed_targets)
    assert vr.relocated == len(targets) - len(renamed_targets)
    # 映射表里没有的目标按原名找
    n, missing = relocate_targets(mini_spec, obf, targets[:3], mapping)
    assert n == 3 and missing == []
    n, missing = relocate_targets(mini_spec, obf, [{"name": "no_such_fn", "file": "calc.c", "target_id": "mini:calc.c:no_such_fn"}],
                                  mapping)
    assert n == 0 and missing == ["mini:calc.c:no_such_fn"]


def test_verify_failure_paths(mini_spec, mini_obf, workdir):
    obf, mapping = mini_obf
    targets = extract_functions(mini_spec, MINI)[:5]
    bad = RunResult(rc=1, stdout="FAIL somewhere\n", stderr="gcc: error: boom\n", duration_s=0.2, timed_out=False,
                    signal=None)
    vr = verify(mini_spec, obf, targets, mapping, FakeSandbox(bad), work_root=workdir / "work")
    assert not vr.ok and vr.build_rc == 1 and not vr.tests_passed
    assert "FAIL somewhere" in vr.stdout_tail and "gcc: error: boom" in vr.stdout_tail
    slow = RunResult(rc=137, stdout="", stderr="", duration_s=60.0, timed_out=True, signal=9)
    vr = verify(mini_spec, obf, targets, mapping, FakeSandbox(slow), work_root=workdir / "work")
    assert not vr.ok and vr.timed_out and vr.build_rc == 137
    # rc=0 但输出不满足 stdout_regex 也不算通过（mpc / parson 这类 main 永远返回 0 的语料靠它）
    spec = make_spec(TRICKY_SPEC)
    vr = verify(spec, obf, [], {}, FakeSandbox(ok_result("some other output\n")), work_root=workdir / "work")
    assert not vr.ok and vr.build_rc == 0 and not vr.tests_passed


# --------------------------------------------------------------------------- 需要 docker 的集成测试


@needs_docker
def test_mini_l1_passes_project_tests_in_sandbox(mini_spec, mini_obf, sandbox, workdir):
    obf, mapping = mini_obf
    targets = extract_functions(mini_spec, MINI)
    vr = verify(mini_spec, obf, targets, mapping, sandbox, work_root=workdir / "work")
    assert vr.ok, vr.stdout_tail
    assert "0 failures" in vr.stdout_tail
    assert vr.relocated == 40 and vr.missing == []


@needs_docker
def test_mini_original_tree_passes_too(mini_spec, sandbox, workdir):
    # 对照：未改名的原树本来就该通过，证明上面的通过不是因为判据太松
    vr = verify(mini_spec, MINI, extract_functions(mini_spec, MINI), {}, sandbox, work_root=workdir / "work")
    assert vr.ok, vr.stdout_tail


@needs_docker
def test_verify_catches_a_broken_function_in_the_renamed_copy(mini_spec, mini_obf, sandbox, workdir):
    obf, mapping = mini_obf
    targets = extract_functions(mini_spec, MINI)
    util = obf / "util.c"
    text = util.read_text(encoding="utf-8")
    head = text.index(f"int {mapping['sum_array']}(")
    ret = f"return {mapping['total']};"
    pos = text.index(ret, head)
    util.write_text(text[:pos] + f"return {mapping['total']} + 1;" + text[pos + len(ret):], encoding="utf-8")
    vr = verify(mini_spec, obf, targets, mapping, sandbox, work_root=workdir / "work")
    assert not vr.ok and not vr.tests_passed and vr.build_rc != 0
    assert "FAIL" in vr.stdout_tail
    assert vr.relocated == 40                                            # 函数都还在，只是行为错了


@needs_docker
def test_tricky_project_builds_and_passes_before_and_after(tricky, sandbox, workdir):
    spec, src = tricky
    targets = extract_functions(spec, src)
    vr0 = verify(spec, src, targets, {}, sandbox, work_root=workdir / "work")
    assert vr0.ok, vr0.stdout_tail
    dest = corpus.materialize(src, workdir / "t")
    res = obfuscate_symbols(spec, dest, SEED)
    vr = verify(spec, dest, targets, res.mapping, sandbox, work_root=workdir / "work")
    assert vr.ok, vr.stdout_tail
    assert vr.relocated == len(targets) and vr.missing == []
    # 目标里被改名的函数，原名在改名后的代码里一个都不剩（注释和字符串除外）
    renamed = [t.name for t in targets if t.name in res.mapping]
    assert renamed
    lex = lex_tree(spec, dest, ["tricky.c", "tricky.h", "main.c"])
    left = {text for toks in lex.values() for kind, text, _raw in toks if kind == "id"}
    assert not (set(renamed) & left)


@needs_docker
def test_union_config_sees_definitions_behind_build_command_defines(sandbox, workdir):
    spec = make_spec(UNION_SPEC)
    src = write_project(workdir / "u", UNION_FILES)
    ids = collect_project_identifiers(spec, src)
    assert ids.stats["configs"] == 2                        # 基础配置 + 加上 build_cmd 里 -D 的并集配置
    assert {"fx_func", "local_x", "plain"} <= set(ids)      # 并集配置下 fx_func 有定义，不会被当成「只有声明」排除
    assert "FEATURE_X" not in ids                           # -D 传入的宏不是项目声明的，更不能改
    targets = extract_functions(spec, src)                  # 基础配置下只看得见 plain
    assert [t.name for t in targets] == ["plain"]
    res = obfuscate_symbols(spec, src, SEED)
    vr = verify(spec, src, targets, res.mapping, sandbox, work_root=workdir / "work")
    assert vr.ok, vr.stdout_tail                            # 沙箱里带 -DFEATURE_X=1 编译，改名后输出仍是 "4 2"
    text = (src / "u.c").read_text(encoding="utf-8")
    assert "fx_func" not in text and res.mapping["fx_func"] in text         # #ifdef 里的定义也一并改名了


@needs_docker
def test_exotic_c_constructs_survive_renaming(sandbox, workdir):
    spec = make_spec(STRESS_SPEC)
    src = write_project(workdir / "stress_src", STRESS_FILES)
    targets = extract_functions(spec, src)
    assert verify(spec, src, targets, {}, sandbox, work_root=workdir / "work").ok      # 原树本身是对的
    ids = collect_project_identifiers(spec, src)
    # ## 拼接宏的实参、宏体里的标识符、拼出来的名字都不改；其余照常改
    assert {"AS_ENUM", "AS_NAME", "CONCAT3", "LOG_FMT"} <= set(ids.stats["excluded_stringify"])
    assert "RED" in ids.excluded["stringify"] and "RED" not in ids         # 高阶宏：列表项 RED 会被 ## 拼接
    assert ids.stats["dropped_unseen"] >= 3                                # COLOR_RED 等只由 ## 拼出，源码里没有这个 token
    assert not {"COLOR_RED", "COLOR_GREEN", "COLOR_BLUE"} & set(ids)
    assert {"stress_apply", "stress_bits", "callback", "payload", "flag_b", "COLOR_END", "color_table"} <= set(ids)
    dest = corpus.materialize(src, workdir / "stress")
    res = obfuscate_symbols(spec, dest, SEED)
    vr = verify(spec, dest, targets, res.mapping, sandbox, work_root=workdir / "work")
    assert vr.ok, vr.stdout_tail
    assert vr.relocated == len(targets)


@needs_docker
def test_strict_stringify_keeps_stringified_names(mini_spec, sandbox, workdir):
    default = collect_project_identifiers(mini_spec, MINI)
    strict = collect_project_identifiers(mini_spec, MINI, strict_stringify=True)
    assert "calc_apply" in default and "calc_apply" not in strict
    assert strict.stats["strict_stringify"] and strict.stats["stringified_names_renamed"] == []
    assert set(default.stats["stringified_names_renamed"]) <= set(strict.stats["excluded_stringify"])
    dest = corpus.materialize(MINI, workdir / "st")
    res = obfuscate_symbols(mini_spec, dest, SEED, strict_stringify=True)
    vr = verify(mini_spec, dest, extract_functions(mini_spec, MINI), res.mapping, sandbox, work_root=workdir / "work")
    assert vr.ok, vr.stdout_tail
    assert res.stats["entities_renamed"] < default.stats["entities_renamed"]


@needs_docker
def test_obfuscate_falls_back_to_strict_when_stringified_names_matter(sandbox, workdir):
    spec = make_spec(STRMINI_SPEC)
    src = write_project(workdir / "sm_src", STRMINI_FILES)
    targets = extract_functions(spec, src)
    assert [t.name for t in targets] == ["s_func"]
    ids = collect_project_identifiers(spec, src)
    # STR(s_func) 出现在测试文件 main.c 里：非 strict 照常改名，如实记下来
    assert "s_func" in ids and ids.stats["stringified_names_renamed"] == ["s_func"]
    out = obfuscate(spec, src, "L1", SEED, workdir / "sm_out", sandbox=sandbox, targets=targets, write_result=False,
                    work_root=workdir / "work")
    assert [a["strict_stringify"] for a in out.stats["attempts"]] == [False, True]
    assert [a["ok"] for a in out.stats["attempts"]] == [False, True]      # 非 strict 的字符串化比对失败，strict 通过
    assert out.strict_stringify and out.ok and out.verify["relocated"] == 1
    assert "s_func" not in out.rename_map
    assert out.stats["targets_not_renamed"] == ["strmini:s.c:s_func"]   # strict 保住了字符串化的输出，代价是这个目标没改名
    # 固定成非 strict 时，失败就是失败
    out2 = obfuscate(spec, src, "L1", SEED, workdir / "sm_out2", sandbox=sandbox, targets=targets,
                     strict_stringify=False, write_result=False, work_root=workdir / "work")
    assert len(out2.stats["attempts"]) == 1 and not out2.ok and not out2.strict_stringify


@needs_docker
def test_obfuscate_does_not_retry_when_strict_would_change_nothing(mini_spec, sandbox, workdir):
    # 没有名字受字符串化影响时，一次通过就结束，不会多跑第二遍
    targets = extract_functions(mini_spec, MINI)
    out = obfuscate(mini_spec, MINI, "L1", SEED, workdir / "o", sandbox=sandbox, targets=targets, write_result=False,
                    work_root=workdir / "work")
    assert out.ok and len(out.stats["attempts"]) == 1 and not out.strict_stringify
    assert out.stats["targets_not_renamed"] == []                       # mini 的 40 个目标名字全被改掉了


LIBSTR_FILES = {
    "lib.h": '''#ifndef LIB_H
#define LIB_H
#define STR(x) #x
int lib_helper(int v);
const char *lib_name(void);
#endif
''',
    "lib.c": '''#include "lib.h"

int lib_helper(int v)
{
    int r = v;

    r += 1;
    return r;
}

const char *lib_name(void)
{
    return STR(lib_helper);
}
''',
    "main.c": '''#include <stdio.h>
#include <string.h>
#include "lib.h"

int main(void)
{
    if (strcmp(lib_name(), "lib_helper") != 0) {
        printf("library name changed: %s\\n", lib_name());
        return 1;
    }
    if (lib_helper(1) != 2) {
        return 1;
    }
    printf("0 failures\\n");
    return 0;
}
''',
}
LIBSTR_SPEC = {
    "name": "libstr", "url": "", "commit": "", "license": "MIT",
    "project_files": ["lib.h", "lib.c", "main.c"], "target_files": ["lib.c"],
    "cflags": ["-std=c99", "-I."],
    "build_cmd": "gcc -std=c99 -O0 -Wall -o libstr_test lib.c main.c && ./libstr_test",
    "test_cmd": None, "pass_criteria": {"exit_code": 0, "stdout_regex": "0 failures"}, "coverage": {},
    "timeout_s": 60, "notes": "",
}


def test_stringified_arguments_in_library_files_are_never_renamed(workdir):
    """库自己的源码（target_files）里被 # 宏字符串化的名字会变成运行时输出，改名等于改了库的行为：不论 strict 与否都不改。
    只有测试文件里的 # 宏实参（失败提示之类）才在非 strict 时照常改名。"""
    spec = make_spec(LIBSTR_SPEC)
    src = write_project(workdir / "ls_src", LIBSTR_FILES)
    ids = collect_project_identifiers(spec, src)
    assert "lib_helper" not in ids and "lib_helper" in ids.stats["excluded_stringify"]
    assert ids.stats["stringified_names_renamed"] == []
    # 对照：同一个宏换到测试文件里用，非 strict 就照常改名并如实记录
    spec2 = make_spec(STRMINI_SPEC)
    src2 = write_project(workdir / "sm_src", STRMINI_FILES)
    ids2 = collect_project_identifiers(spec2, src2)
    assert "s_func" in ids2 and ids2.stats["stringified_names_renamed"] == ["s_func"]


@needs_docker
def test_library_stringify_needs_no_strict_fallback(sandbox, workdir):
    spec = make_spec(LIBSTR_SPEC)
    src = write_project(workdir / "ls_src", LIBSTR_FILES)
    targets = extract_functions(spec, src)
    out = obfuscate(spec, src, "L1", SEED, workdir / "ls_out", sandbox=sandbox, targets=targets, write_result=False,
                    work_root=workdir / "work")
    assert out.ok and [a["strict_stringify"] for a in out.stats["attempts"]] == [False]    # 一次通过，不需要退回 strict
    assert "lib_helper" not in out.rename_map


# --------------------------------------------------------------------------- 真实语料（集成检查）


def _real_corpus(name: str):
    return corpus.load_spec(name), require_real_corpus(name)


@needs_docker
@pytest.mark.parametrize("name", REAL_CORPORA)
def test_real_corpus_l1_passes_project_tests(name, sandbox, workdir):
    spec, src = _real_corpus(name)
    targets = extract_functions(spec, src)
    assert targets
    dest = corpus.materialize(src, workdir / name)
    res = obfuscate_symbols(spec, dest, SEED)
    st = res.stats
    assert st["entities_renamed"] > 0 and st["entities_renamed"] == len(res.mapping)
    assert st["entities_declared"] == st["entities_renamed"] + st["entities_excluded"]
    assert_token_equivalence(spec, src, dest, res.mapping)
    vr = verify(spec, dest, targets, res.mapping, sandbox, work_root=workdir / "work")
    assert vr.ok, vr.stdout_tail
    assert vr.relocated == len(targets) and vr.missing == []
    # 目标函数的原名在改名后的代码里不再以标识符出现（被排除的除外）
    r_files, _ = S.project_file_lists(spec, dest)
    left = {text for toks in lex_tree(spec, dest, r_files).values() for kind, text, _raw in toks if kind == "id"}
    assert sorted({t.name for t in targets if t.name in res.mapping} & left) == []
    print(f"{name}: targets={len(targets)} declared={st['entities_declared']} renamed={st['entities_renamed']} "
          f"excluded={st['entities_excluded']} by_reason={st['excluded_by_reason']}")
