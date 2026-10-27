"""全项目共用的词表：混淆级别、回复问题名、归因类别。

没有任何依赖（不拉起 libclang、不读文件），obfuscate / postprocess / attribute / metrics / report 都从这里取，
各处不再各抄一份；tests/test_e2e_mock.py 锁住它们和各模块导出值的一致性。
"""
from __future__ import annotations

__all__ = ["LEVELS", "ISSUES", "CATEGORIES"]

#: 混淆级别：L0 原样；L1 符号；L2 = L1 + 结构；L3 = L2 + 语义
LEVELS = ["L0", "L1", "L2", "L3"]

#: 回复后处理记下的问题（postprocess.Candidate.issues 的取值与顺序）
ISSUES = ("no_code_block", "multiple_functions", "signature_mismatch", "added_include", "truncated",
          "policy_violation")

#: 错误归因类别（attribute.attribute 的取值，也是报告表里的列顺序）
CATEGORIES = [
    "pass", "extract_fail", "policy_violation", "compile_syntax", "compile_undeclared", "compile_type",
    "compile_other", "link_error", "runtime_crash", "runtime_timeout", "test_fail",
]
