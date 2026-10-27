/* mini.h：extract / 混淆 / 评测链路自测用的小型 C 项目（公共头文件）。 */
#ifndef MINI_H
#define MINI_H

#include <stddef.h>

/* ---- 宏：无参、带参、嵌套、空宏、返回类型包装 ---- */
#define MINI_VERSION 3
#define MINI_STACK_CAP 8
#define MINI_API
#define MINI_RET(type) type
#define MINI_MIN(a, b) ((a) < (b) ? (a) : (b))
#define MINI_MAX(a, b) ((a) > (b) ? (a) : (b))
#define MINI_CLAMP(x, lo, hi) MINI_MIN(MINI_MAX((x), (lo)), (hi))
#define MINI_SQUARE(x) ((x) * (x))

/* ---- 类型：enum、struct、typedef、函数指针 ---- */
typedef enum { OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_COUNT } op_kind_t;

enum rounding { ROUND_DOWN, ROUND_UP };

typedef struct point {
    int x;
    int y;
} point_t;

struct rect {
    point_t lo;
    point_t hi;
};

typedef struct intstack {
    int items[MINI_STACK_CAP];
    size_t len;
} intstack_t;

typedef int (*binop_fn)(int, int);
typedef int (*pred_fn)(int);

/* ---- 全局变量 ---- */
extern int g_calls;
extern const char *g_last_error;
extern const int g_primes[5];

/* ---- calc.c ---- */
MINI_API int mini_version(void);

/** 按运算类型对两个整数做运算；运算类型非法时返回 -1 并设置 g_last_error。 */
int calc_apply(op_kind_t op, int a, int b);
const char *op_name(op_kind_t op);
MINI_RET(int) calc_abs_diff(int a, int b);
int calc_clamp(int x, int lo, int hi);
int calc_add_sat(int a, int b);
int calc_div_round(int a, int b, enum rounding mode);
int calc_gcd(int a, int b);
int calc_median3(int a, int b, int c);
int calc_in_range(int x, int lo, int hi);
int calc_score(int x);
point_t point_make(int x, int y);
point_t point_add(point_t a, point_t b);
int point_dist2(point_t a, point_t b);
int rect_area(const struct rect *r);
void rect_normalize(struct rect *r);
int rect_contains(const struct rect *r, int x, int y);

/* ---- util.c ---- */
int sum_array(const int *a, size_t n);
int weighted_sum(const int *a, size_t n);
int sum_positive(const int *a, size_t n);
int array_find(const int *a, size_t n, int v);
int array_max_gap(const int *a, size_t n);
void array_reverse(int *a, size_t n);
int array_count_if(const int *a, size_t n, pred_fn pred);
int array_count_even(const int *a, size_t n);
char *str_dup_upper(const char *s);
int parse_digits(const char *s, int *out);
void stack_init(intstack_t *s);
int stack_push(intstack_t *s, int v);
int stack_pop(intstack_t *s, int *out);
int counter_next(void);
void rand_seed(unsigned int seed);
int rand_next(void);
int prime_at(size_t i);

#endif /* MINI_H */
