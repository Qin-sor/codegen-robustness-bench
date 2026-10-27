# mini 项目：make test 编译并运行 test_main.c，退出码 0 表示全部通过。
# CFLAGS / LDFLAGS 可被环境变量覆盖（覆盖率构建会注入 --coverage）。
CC = gcc
CFLAGS ?= -std=c99 -Wall -Wextra -O0
LDFLAGS ?=
SRCS = calc.c util.c test_main.c

test: mini_test
	./mini_test

mini_test: $(SRCS) mini.h
	$(CC) $(CFLAGS) -o $@ $(SRCS) $(LDFLAGS)

clean:
	rm -f mini_test *.o *.gcda *.gcno

.PHONY: test clean
