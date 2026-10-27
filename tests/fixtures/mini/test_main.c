/* test_main.c：mini 项目的自带测试，退出码 0 表示全部通过。
 * 每个被测函数都至少有一个「返回零值的桩会被抓住」的断言（stub-kill）。 */
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mini.h"

static int g_checks = 0;
static int g_failures = 0;

/* 带 # 字符串化的断言宏：符号混淆时要特别处理这类宏 */
#define CHECK(cond)                                                      \
    do {                                                                 \
        g_checks++;                                                      \
        if (!(cond)) {                                                   \
            g_failures++;                                                \
            printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);       \
        }                                                                \
    } while (0)

static int str_eq(const char *a, const char *b)
{
    return a != NULL && b != NULL && strcmp(a, b) == 0;
}

static int is_even(int x) { return x % 2 == 0; }
static int is_positive(int x) { return x > 0; }

static void test_calc_apply(void)
{
    g_calls = 0;
    g_last_error = NULL;
    CHECK(calc_apply(OP_ADD, 2, 3) == 5);
    CHECK(calc_apply(OP_SUB, 2, 3) == -1);
    CHECK(calc_apply(OP_MUL, 4, 5) == 20);
    CHECK(calc_apply(OP_DIV, 7, 2) == 3);
    CHECK(g_calls == 4);
    CHECK(calc_apply(OP_COUNT, 1, 1) == -1);
    CHECK(str_eq(g_last_error, "bad op"));
    g_last_error = NULL;
    CHECK(calc_apply(OP_DIV, 1, 0) == 0);
    CHECK(str_eq(g_last_error, "division by zero"));
}

static void test_op_name(void)
{
    CHECK(str_eq(op_name(OP_ADD), "add"));
    CHECK(str_eq(op_name(OP_SUB), "sub"));
    CHECK(str_eq(op_name(OP_MUL), "mul"));
    CHECK(str_eq(op_name(OP_DIV), "div"));
    CHECK(str_eq(op_name(OP_COUNT), "unknown"));
}

static void test_calc_misc(void)
{
    CHECK(mini_version() == 3);
    CHECK(calc_abs_diff(3, 10) == 7);
    CHECK(calc_abs_diff(10, 3) == 7);
    CHECK(calc_abs_diff(4, 4) == 0);
    CHECK(calc_clamp(15, 0, 10) == 10);
    CHECK(calc_clamp(-5, 0, 10) == 0);
    CHECK(calc_clamp(5, 0, 10) == 5);
    CHECK(calc_add_sat(INT_MAX, 1) == INT_MAX);
    CHECK(calc_add_sat(INT_MIN, -1) == INT_MIN);
    CHECK(calc_add_sat(2, 3) == 5);
    CHECK(calc_add_sat(-2, -3) == -5);
    CHECK(calc_gcd(12, 18) == 6);
    CHECK(calc_gcd(17, 5) == 1);
    CHECK(calc_gcd(0, 9) == 9);
    CHECK(calc_gcd(-5, 0) == 5);
    CHECK(calc_median3(1, 2, 3) == 2);
    CHECK(calc_median3(3, 1, 2) == 2);
    CHECK(calc_median3(2, 3, 1) == 2);
    CHECK(calc_median3(1, 1, 5) == 1);
    CHECK(calc_median3(9, 5, 7) == 7);
    CHECK(calc_in_range(5, 1, 10) == 1);
    CHECK(calc_in_range(0, 1, 10) == 0);
    CHECK(calc_in_range(11, 1, 10) == 0);
    CHECK(calc_in_range(1, 1, 10) == 1);
    CHECK(calc_score(3) == 10);
    CHECK(calc_score(-3) == -10);
    CHECK(calc_score(0) == 1);
}

static void test_calc_div_round(void)
{
    g_last_error = NULL;
    CHECK(calc_div_round(7, 2, ROUND_DOWN) == 3);
    CHECK(calc_div_round(7, 2, ROUND_UP) == 4);
    CHECK(calc_div_round(-7, 2, ROUND_UP) == -4);
    CHECK(calc_div_round(-7, 2, ROUND_DOWN) == -3);
    CHECK(calc_div_round(6, 2, ROUND_UP) == 3);
    CHECK(calc_div_round(1, 0, ROUND_UP) == 0);
    CHECK(str_eq(g_last_error, "division by zero"));
}

static void test_geometry(void)
{
    point_t p = point_make(3, 4);
    point_t q = point_add(p, point_make(10, 20));
    struct rect r;
    struct rect inverted;

    CHECK(p.x == 3 && p.y == 4);
    CHECK(q.x == 13 && q.y == 24);
    CHECK(point_dist2(point_make(0, 0), p) == 25);

    r.lo = point_make(1, 1);
    r.hi = point_make(4, 5);
    CHECK(rect_area(&r) == 12);
    CHECK(rect_contains(&r, 2, 2) == 1);
    CHECK(rect_contains(&r, 5, 2) == 0);
    CHECK(rect_contains(&r, 2, 0) == 0);

    inverted.lo = point_make(5, 9);
    inverted.hi = point_make(1, 2);
    CHECK(rect_area(&inverted) == 0);
    rect_normalize(&inverted);
    CHECK(inverted.lo.x == 1 && inverted.lo.y == 2);
    CHECK(inverted.hi.x == 5 && inverted.hi.y == 9);
    CHECK(rect_area(&inverted) == 28);
}

static void test_arrays(void)
{
    int a[5] = { 1, 2, 3, 4, 6 };
    int b[4] = { -1, 5, 0, 7 };
    int c[1] = { 9 };
    int gap[5] = { 1, 4, 2, 9, 8 };

    CHECK(sum_array(a, 5) == 16);
    CHECK(sum_array(a, 0) == 0);
    CHECK(sum_positive(b, 4) == 12);
    CHECK(weighted_sum(a, 5) == 3 * 1 + 1 * 2 + 4 * 3 + 1 * 4 + 5 * 6);
    CHECK(weighted_sum(a, 2) == 5);
    CHECK(array_find(a, 5, 4) == 3);
    CHECK(array_find(a, 5, 9) == -1);
    CHECK(array_find(a, 0, 1) == -1);
    CHECK(array_count_if(a, 5, is_even) == 3);
    CHECK(array_count_if(b, 4, is_positive) == 2);
    CHECK(array_count_even(a, 5) == 3);
    CHECK(array_count_even(b, 4) == 1);

    CHECK(array_max_gap(gap, 5) == 7);
    CHECK(array_max_gap(gap, 1) == 0);

    array_reverse(a, 5);
    CHECK(a[0] == 6 && a[1] == 4 && a[2] == 3 && a[3] == 2 && a[4] == 1);
    array_reverse(a, 0);
    CHECK(a[0] == 6 && a[4] == 1);
    array_reverse(c, 1);
    CHECK(c[0] == 9);
}

static void test_strings(void)
{
    char *up = str_dup_upper("hello, World 42");
    char *empty = str_dup_upper("");
    int value = -1;

    CHECK(str_eq(up, "HELLO, WORLD 42"));
    CHECK(empty != NULL && empty[0] == '\0');
    CHECK(str_dup_upper(NULL) == NULL);
    free(up);
    free(empty);

    g_last_error = NULL;
    CHECK(parse_digits("123abc", &value) == 0);
    CHECK(value == 123);
    CHECK(parse_digits("abc", &value) == -1);
    CHECK(str_eq(g_last_error, "parse error"));
    CHECK(parse_digits(NULL, &value) == -1);
    CHECK(parse_digits("7", NULL) == -1);
}

static void test_stack(void)
{
    intstack_t st;
    int out = -1;
    int i;

    memset(&st, 0xAB, sizeof st);
    stack_init(&st);
    CHECK(st.len == 0);
    CHECK(st.items[0] == 0 && st.items[MINI_STACK_CAP - 1] == 0);

    g_last_error = NULL;
    CHECK(stack_pop(&st, &out) == -1);
    CHECK(str_eq(g_last_error, "stack empty"));
    for (i = 0; i < MINI_STACK_CAP; i++) {
        CHECK(stack_push(&st, i * 10) == 0);
    }
    CHECK(st.len == MINI_STACK_CAP);
    g_last_error = NULL;
    CHECK(stack_push(&st, 99) == -1);
    CHECK(str_eq(g_last_error, "stack full"));
    CHECK(stack_pop(&st, &out) == 0);
    CHECK(out == 70);
    CHECK(stack_pop(&st, &out) == 0);
    CHECK(out == 60);
    CHECK(st.len == MINI_STACK_CAP - 2);
}

static void test_globals(void)
{
    g_calls = 0;
    CHECK(counter_next() == 1);
    CHECK(counter_next() == 2);
    CHECK(counter_next() == 3);
    CHECK(g_calls == 3);

    /* 期望值由 Python 按同一线性同余公式算出（无符号 32 位回绕） */
    rand_seed(42);
    CHECK(rand_next() == 27);
    CHECK(rand_next() == 64);
    rand_seed(7);
    CHECK(rand_next() == 16);
    CHECK(rand_next() == 33);

    CHECK(prime_at(0) == 2);
    CHECK(prime_at(2) == 5);
    CHECK(prime_at(4) == 11);
    CHECK(prime_at(5) == -1);
}

int main(void)
{
    test_calc_apply();
    test_op_name();
    test_calc_misc();
    test_calc_div_round();
    test_geometry();
    test_arrays();
    test_strings();
    test_stack();
    test_globals();
    printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
