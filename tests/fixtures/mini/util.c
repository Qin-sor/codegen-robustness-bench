/* util.c：数组、字符串、栈、回调相关函数。 */
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include "mini.h"

static unsigned int s_seed = 12345u;
static const int s_weights[] = { 3, 1, 4, 1, 5 };
const int g_primes[5] = { 2, 3, 5, 7, 11 };

/* 对数组求和（中文注释：多字节字符会让字节偏移与字符偏移错开，用来验证偏移按字节算） */
int sum_array(const int *a, size_t n)
{
    int total = 0;
    size_t i;

    for (i = 0; i < n; i++) {
        total += a[i];
    }
    return total;
}

/* 加权求和：权重表是省略长度的 static 数组，函数体里用 sizeof 推长度 */
int weighted_sum(const int *a, size_t n)
{
    int total = 0;
    size_t i;

    for (i = 0; i < n && i < sizeof(s_weights) / sizeof(s_weights[0]); i++) {
        total += a[i] * s_weights[i];
    }
    return total;
}

int sum_positive(const int *a, size_t n)
{
    int total = 0;
    size_t i;

    for (i = 0; i < n; i++) {
        if (a[i] <= 0) {
            continue;
        }
        total += a[i];
    }
    return total;
}

int array_find(const int *a, size_t n, int v)
{
    size_t index;

    for (index = 0; index < n; index++) {
        if (a[index] == v) {
            return (int)index;
        }
    }
    return -1;
}

/* 相邻元素差的最大值：跨文件调用 calc.c 里的 calc_abs_diff（原型带返回类型包装宏） */
int array_max_gap(const int *a, size_t n)
{
    int best = 0;
    size_t i;

    for (i = 1; i < n; i++) {
        int gap = calc_abs_diff(a[i - 1], a[i]);

        if (gap > best) {
            best = gap;
        }
    }
    return best;
}

void array_reverse(int *a, size_t n)
{
    size_t i = 0, j;
    int tmp;

    if (!n) {
        return;
    }
    j = n - 1;
    while (i < j) {
        tmp = a[i];
        a[i] = a[j];
        a[j] = tmp;
        i++;
        j--;
    }
}

int array_count_if(const int *a, size_t n, pred_fn pred)
{
    int count = 0;
    size_t i;

    for (i = 0; i < n; i++) {
        if (pred(a[i])) {
            count++;
        }
    }
    return count;
}

/* 静态辅助函数：只通过函数指针（取地址）被引用，不会出现在任何函数的 callees 里 */
static int is_even_value(int x)
{
    return x % 2 == 0;
}

int array_count_even(const int *a, size_t n)
{
    int total = array_count_if(a, n, is_even_value);

    return total;
}

char *str_dup_upper(const char *s)
{
    size_t len, i;
    char *out;

    if (s == NULL) {
        return NULL;
    }
    len = strlen(s);
    out = (char *)malloc(len + 1);
    if (out == NULL) {
        return NULL;
    }
    for (i = 0; i < len; i++) {
        out[i] = (char)toupper((unsigned char)s[i]);
    }
    out[len] = '\0';
    return out;
}

int parse_digits(const char *s, int *out)
{
    int value = 0;
    int seen = 0;

    if (s == NULL || out == NULL) {
        goto fail;
    }
    while (*s >= '0' && *s <= '9') {
        value = value * 10 + (*s - '0');
        seen = 1;
        s++;
    }
    if (!seen) {
        goto fail;
    }
    *out = value;
    return 0;

fail:
    g_last_error = "parse error";
    return -1;
}

void stack_init(intstack_t *s)
{
    size_t i;

    s->len = 0;
    for (i = 0; i < MINI_STACK_CAP; i++) {
        s->items[i] = 0;
    }
}

int stack_push(intstack_t *s, int v)
{
    if (s->len >= MINI_STACK_CAP) {
        g_last_error = "stack full";
        return -1;
    }
    s->items[s->len++] = v;
    return 0;
}

int stack_pop(intstack_t *s, int *out)
{
    if (s->len == 0) {
        g_last_error = "stack empty";
        return -1;
    }
    s->len--;
    *out = s->items[s->len];
    return 0;
}

int counter_next(void)
{
    g_calls += 1;
    return g_calls;
}

void rand_seed(unsigned int seed)
{
    s_seed = seed;
}

int rand_next(void)
{
    s_seed = (s_seed * 1103515245u + 12345u) & 0x7fffffffu;
    return (int)(s_seed % 100u);
}

int prime_at(size_t i)
{
    if (i >= sizeof(g_primes) / sizeof(g_primes[0])) {
        return -1;
    }
    return g_primes[i];
}
