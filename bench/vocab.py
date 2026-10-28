"""全项目共用的词表：混淆级别、回复问题名、归因类别。

没有任何依赖（不拉起 libclang、不读文件），obfuscate / postprocess / attribute / metrics / report 都从这里取，
各处不再各抄一份；tests/test_e2e_mock.py 锁住它们和各模块导出值的一致性。
"""
from __future__ import annotations

__all__ = ["LEVELS", "ISSUES", "CATEGORIES", "MUTATION_OPERATORS", "MUTATION_TITLES", "KILL_MODES"]

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

#: 缺陷修复任务的变异算子编号：按 docs/设计-缺陷修复任务.md §4 的优先序（也是报告里各行的顺序），以及中文名
MUTATION_OPERATORS = ("ROR", "NEG", "LOR", "OBO", "DGS", "AOR", "RVR", "DST")
MUTATION_TITLES = {
    "ROR": "关系算子替换", "NEG": "条件取反", "LOR": "逻辑算子替换", "OBO": "差一", "DGS": "删守卫",
    "AOR": "算术算子替换", "RVR": "返回值替换", "DST": "删语句",
}

#: 桩 / 变异体被项目测试杀死的方式（attribute 的归因类别里属于「测试阶段失败」的三类）：断言失败、崩溃、超时
KILL_MODES = ("test_fail", "runtime_crash", "runtime_timeout")
