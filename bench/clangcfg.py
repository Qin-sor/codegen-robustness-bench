"""libclang 配置：统一设置库文件路径，所有解析代码从这里拿 Index。

库文件路径优先取环境变量 LIBCLANG_PATH，否则在常见位置里找；
解析选项默认带 PARSE_DETAILED_PROCESSING_RECORD，这样宏定义与宏展开才会出现在 AST 里。
"""
from __future__ import annotations

import os
from pathlib import Path

import clang.cindex as ci

_CANDIDATES = [
    "/usr/lib/llvm-14/lib/libclang.so.1",
    "/usr/lib/x86_64-linux-gnu/libclang-14.so.1",
    "/usr/lib/llvm-14/lib/libclang.so",
]

DEFAULT_PARSE_OPTIONS = ci.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD

_configured = False


def configure() -> None:
    """只执行一次：定位 libclang 并告诉 cindex。"""
    global _configured
    if _configured:
        return
    path = os.environ.get("LIBCLANG_PATH")
    if not path:
        for cand in _CANDIDATES:
            if Path(cand).exists():
                path = cand
                break
    if path:
        ci.Config.set_library_file(path)
    _configured = True


def index() -> ci.Index:
    """返回配置好的 Index，调用方用 index().parse(...)。"""
    configure()
    return ci.Index.create()
