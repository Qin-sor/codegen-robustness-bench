/* calc.c：运算与几何相关函数。 */
#include <limits.h>
#include "mini.h"

int g_calls = 0;
const char *g_last_error = NULL;

/* 四个二元运算回调：被下面的函数指针表引用 */
static int op_add(int a, int b) { return a + b; }
static int op_sub(int a, int b) { return a - b; }
static int op_mul(int a, int b) { return a * b; }

static int op_div(int a, int b)
{
    if (b == 0) {
        g_last_error = "division by zero";
        return 0;
    }
    return a / b;
}

static const binop_fn g_ops[OP_COUNT] = { op_add, op_sub, op_mul, op_div };

/* 单语句的 static inline：extract_functions 会把它排除在目标之外 */
static inline int sq_plus_one(int x) { return MINI_SQUARE(x) + 1; }

/* 多语句的 static inline：不应被排除 */
static inline int sign_of(int x)
{
    if (x > 0) {
        return 1;
    }
    if (x < 0) {
        return -1;
    }
    return 0;
}

MINI_API int mini_version(void)
{
    return MINI_VERSION;
}

int calc_apply(op_kind_t op, int a, int b)
{
    binop_fn fn;
    int result;

    if (op >= OP_COUNT) {
        g_last_error = "bad op";
        return -1;
    }
    fn = g_ops[op];
    result = fn(a, b);
    g_calls++;
    return result;
}

// 运算名：switch + 指针返回
const char *op_name(op_kind_t op)
{
    switch (op) {
    case OP_ADD: return "add";
    case OP_SUB: return "sub";
    case OP_MUL: return "mul";
    case OP_DIV: return "div";
    default: break;
    }
    return "unknown";
}

MINI_RET(int) calc_abs_diff(int a, int b)
{
    return a < b ? b - a : a - b;
}

int calc_clamp(int x, int lo, int hi)
{
    return MINI_CLAMP(x, lo, hi);
}

/* 饱和加法：INT_MAX / INT_MIN 来自系统头 limits.h，不算项目宏 */
int calc_add_sat(int a, int b)
{
    if (b > 0 && a > INT_MAX - b) {
        return INT_MAX;
    }
    if (b < 0 && a < INT_MIN - b) {
        return INT_MIN;
    }
    return a + b;
}

int calc_div_round(int a, int b, enum rounding mode)
{
    int q, r;

    if (b == 0) {
        g_last_error = "division by zero";
        return 0;
    }
    q = a / b;
    r = a % b;
    if (mode == ROUND_UP && r != 0) {
        q += ((a < 0) == (b < 0)) ? 1 : -1;
    }
    return q;
}

int calc_gcd(int a, int b)
{
    if (b == 0) {
        return a < 0 ? -a : a;
    }
    return calc_gcd(b, a % b);
}

/* 三数中位数：if / else 以及 else if 链 */
int calc_median3(int a, int b, int c)
{
    int lo, hi;

    if (a < b) {
        lo = a;
        hi = b;
    } else {
        lo = b;
        hi = a;
    }
    if (c < lo) {
        return lo;
    } else if (c > hi) {
        return hi;
    } else {
        return c;
    }
}

int calc_in_range(int x, int lo, int hi)
{
    return !(x < lo || x > hi);
}

int calc_score(int x)
{
    int base = sq_plus_one(x);
    int s = sign_of(x);

    return s < 0 ? -base : base;
}

point_t point_make(int x, int y)
{
    point_t p;

    p.x = x;
    p.y = y;
    return p;
}

point_t point_add(point_t a, point_t b)
{
    point_t r = point_make(a.x + b.x, a.y + b.y);

    g_calls++;
    return r;
}

int point_dist2(point_t a, point_t b)
{
    int dx = a.x - b.x;
    int dy = a.y - b.y;

    return MINI_SQUARE(dx) + MINI_SQUARE(dy);
}

int rect_area(const struct rect *r)
{
    int w = r->hi.x - r->lo.x, h = r->hi.y - r->lo.y;

    if (!(w > 0 && h > 0)) {
        return 0;
    }
    return w * h;
}

void rect_normalize(struct rect *r)
{
    int x0 = MINI_MIN(r->lo.x, r->hi.x), x1 = MINI_MAX(r->lo.x, r->hi.x);
    int y0 = MINI_MIN(r->lo.y, r->hi.y), y1 = MINI_MAX(r->lo.y, r->hi.y);

    r->lo.x = x0;
    r->hi.x = x1;
    r->lo.y = y0;
    r->hi.y = y1;
}

int rect_contains(const struct rect *r,
                  int x,
                  int y)
{
    return x >= r->lo.x && x <= r->hi.x &&
           y >= r->lo.y && y <= r->hi.y;
}
