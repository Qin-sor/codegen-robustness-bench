"""大模型代码生成鲁棒性评测平台（codegen-robustness-bench）。

需要 Python >= 3.10：代码里用了 ``dataclass(slots=True)``、``bisect`` 的 ``key=``、``glob`` 的 ``root_dir=``，
低版本只会在 import 时报一个看不出原因的 TypeError，这里先给一个明确的提示。
"""
import sys

if sys.version_info < (3, 10):
    raise RuntimeError(f"codegen-robustness-bench 需要 Python >= 3.10，当前是 {sys.version.split()[0]}")
