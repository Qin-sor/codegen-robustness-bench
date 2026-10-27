#!/usr/bin/env bash
# CI 脚本：把「环境不全就悄悄跳过」变成失败。
#
# 可信度措施（oracle gate、stub-kill、e2e 自测、沙箱评测）的自测全靠 docker 沙箱和真实语料；
# 在没有 docker、没有 gcc:13 镜像、没有语料的机器上，pytest 会把这些用例整组跳过却仍然显示通过。
# 所以这里：
#   1. 设 BENCH_REQUIRE_DOCKER=1 / BENCH_REQUIRE_CORPUS=1，缺 docker 或缺语料时相关用例直接失败；
#   2. 跑完再检查输出里没有「N skipped」，跳过数大于 0 即失败；
#   3. 最后跑一遍 e2e（python3 -m bench e2e），三种 MockLLM 模式的断言必须成立。
#
# 用法：bash scripts/ci.sh        （在项目任意位置运行都行）
# 前置：docker 与镜像 gcc:13；语料要么已在 corpus/src，要么设置 CORPUS_MIRROR（本地克隆）或能联网克隆。
set -euo pipefail
cd "$(dirname "$0")/.."

export BENCH_REQUIRE_DOCKER=1
export BENCH_REQUIRE_CORPUS=1

python3 -m pyflakes bench tests
python3 -m bench prepare --no-build-check            # 幂等：corpus/src 已有就复用

mkdir -p tmp
log="tmp/ci-pytest.log"
set +e
python3 -m pytest tests -q -rs -p no:cacheprovider 2>&1 | tee "$log"
status=${PIPESTATUS[0]}
set -e
if [ "$status" -ne 0 ]; then
    echo "CI 失败：pytest 退出码 $status" >&2
    exit "$status"
fi
if grep -Eq '[0-9]+ skipped' "$log"; then
    echo "CI 失败：有用例被跳过（环境不全？），详见 $log 里 -rs 列出的原因" >&2
    exit 1
fi

python3 -m bench e2e --backend mock
echo "CI 通过"
