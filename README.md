# codegen-robustness-bench · 大模型代码生成鲁棒性评测平台

用「同一份功能规格、不同强度的代码混淆」检验大模型写 C 函数的能力是靠理解，还是靠记忆。

本文是项目说明：定位、流程、用法、可信度措施和局限都如实写出；第 9 节的模型实验结果表留空（没有跑过真实模型，不编数据），
链路本身（prepare 到 verify，以及缺陷修复任务的 mutate）在 9 个真实语料上的集成检查结果见 [`docs/实验记录.md`](docs/实验记录.md)。
接口契约与设计细节见 [`docs/设计.md`](docs/设计.md)（v0.3，第 10 节是「实现偏差记录」）和缺陷修复任务的附录
[`docs/设计-缺陷修复任务.md`](docs/设计-缺陷修复任务.md)。

## 1. 定位

现有代码生成评测（HumanEval、MBPP，以及直接取自 GitHub 的函数级基准）容易受训练数据泄漏影响：被测函数和它所在的项目
很可能出现在模型的训练语料里，模型靠记忆而不是靠理解生成代码，分数反映不出真实的工程生成能力。

本平台的做法：

1. 从多个开源 C 项目里提取目标函数及其依赖上下文；
2. 对**整个项目**做三类可验证的代码混淆：符号（L1）、结构（L2）、语义（L3），每个混淆副本的项目自带测试必须全部通过；
3. 让模型在「同一份功能规格、不同混淆级别」下重新生成目标函数；
4. 把生成结果替换回对应的混淆副本，在 Docker 沙箱里编译并运行项目自带测试；
5. 统计各级别的通过率变化，对失败样本做错误归因，并度量生成结果与原函数的相似度（记忆痕迹）。

要回答的问题：

1. 同一模型在无混淆（L0）与各级混淆（L1–L3）下的编译通过率、功能测试通过率相差多少；
2. 失败是哪一类（语法、依赖缺失、类型、逻辑、崩溃、超时、违反约束），随混淆强度如何迁移；
3. 不同模型的横向对比（任何 OpenAI 兼容端点都能接入）；
4. L0 生成结果与原函数的相似度是否在 L1 后明显下降。

不做的事：不评测多文件新增功能，不评测 C 以外的语言，不做模型微调。

**两类并列的任务。** 上面说的是**功能规格生成任务**（generate）：模型的输入只有函数签名、依赖声明和功能规格，没有函数体；
L2 / L3 的变换发生在函数体内，模型看不到，所以这两级的模型输入与 L1 逐字相同，只对评测侧的项目副本有意义（第 8 节第 7 条）。
为了让三类混淆都作用在模型读到的代码上，另有**缺陷修复任务**（repair，第 3.4 节）：在每一级副本的目标函数里注入一个能被项目测试捕获的缺陷，
让模型读这段（混淆后的）函数、返回修正后的函数。两类任务并列报告：生成任务回答「写函数是否依赖记忆」，修复任务回答「读懂并修好函数是否依赖表面形态」。

## 2. 总体流程

```
prepare    取语料到 corpus/src/<name>，沙箱里确认构建与测试可用
coverage   沙箱里以 --coverage 编译并跑测试，gcov 得到每个函数的执行次数
targets    选目标函数：非测试文件、非 main、语句数 ≥ 3、执行次数 > 0、stub-kill 通过
obfuscate  对整个项目生成 L0 / L1 / L2 / L3 四个副本（含测试文件），记录映射与统计，并 verify
verify     oracle gate：用混淆后的原函数文本逐目标走一遍完整 evaluate，必须通过，否则该目标作废
mutate     （只给缺陷修复任务用）在每级副本的目标函数里注入一个缺陷，沙箱里验证「编译通过且项目测试稳定地失败」才出题
check-mutants  （只给缺陷修复任务用）把 mutate 选定的变异体在沙箱里回放复核，都要稳定地让项目测试失败
describe   用 LLM 为每个目标生成「不含任何项目标识符」的功能规格（一次生成、四级共用；修复任务不需要它）
generate   按（目标, 级别, 模型, 样本）出题，调模型，缓存回复；调用失败的样本单独记录，不静默丢弃（--task 选任务类型）
evaluate   后处理 → 违反约束的直接判定 → 替换回混淆副本 → 沙箱编译 → 跑项目测试 → 归因 → 相似度（--task 选任务类型）
report     汇总 model × level × corpus 的指标表、归因分布、相似度，输出 md / csv / 图（两类任务并列）
```

每一步幂等：输出已存在且输入哈希没变就跳过；所有随机性由 `--seed` 决定。
出题（`generate` / `evaluate` 的前置）时会再核对一遍上游产出：副本、oracle gate、功能规格对不上现状的目标直接作废并计数，
不会因为中间某个阶段被跳过就用上过期数据。

## 3. 快速开始

### 3.1 环境

* Python 3.10（`pyproject.toml` 的 `requires-python`；低版本 import `bench` 时会给出明确提示）；
  依赖：`pip install clang==14.0 pytest pyflakes`（运行时只要 `clang==14.0`；`pyproject.toml` 声明 Python 版本、依赖和 pytest 配置，
  项目不打包安装，直接在项目根目录运行）。libclang 库文件默认在
  `/usr/lib/llvm-14/lib/libclang.so.1`，不在那里时用环境变量 `LIBCLANG_PATH` 指定；
* Docker，并且本地有镜像 `gcc:13`（`docker pull gcc:13`）。语料仓库里的构建、测试，以及模型生成的 C 代码，
  只在沙箱里执行（`--network none --cap-drop ALL --pids-limit --memory --cpus`，非 root 用户），宿主机不直接 make；
* 不需要其它第三方依赖：HTTP 用标准库 urllib，图用标准库直接写 SVG；
* 下面的命令都在项目根目录下运行（`python3 -m bench` 要能 import `bench` 包）。

### 3.2 先跑离线自测

不需要 API key，不联网，用 `tests/fixtures/mini` 这个自写的小项目和 MockLLM 把全链路跑一遍（生成任务和缺陷修复任务都跑；
在 16 核 / 15 GB 的开发机上实测 5 到 7 分钟，结束时打印「用时 N s」；随机器和并发度而变，不是保证值）：

```bash
python3 -m bench e2e --backend mock
```

它会跑 L0–L3 × oracle / broken / wrong 三种 MockLLM 模式，并断言：oracle 100% pass；broken 100% compile_syntax；
wrong（返回零值 / 空函数体的桩）一个都不通过，且每个样本的归因恰好等于 stub-kill 记录的杀死方式
（绝大多数是 test_fail；个别目标的桩会让后面的测试代码读到未初始化的状态而崩溃，归 runtime_crash，那是 stub-kill 已经记下的事实）；
每个副本的 verify 与 oracle gate 全过；把所有阶段再跑一遍必须全部跳过。输出在 `tmp/e2e/`，不碰 `results/`。

缺陷修复任务在同一次 e2e 里跑 mutate → generate → evaluate → check-mutants，MockLLM 多三个只用于修复任务的模式并断言：
oracle 100% pass 且 restored（逐字还原）；unchanged（原样返回变异体）一个都不通过、stage 为 test、`unchanged` 为真，归因与变异体的 `kill_mode`
一致的比例不低于 95%；reformat（原函数换排版、加注释）100% pass、`restored` 为真、`unchanged` 为假，检验规范化比较只忽略排版与注释；
touched（变异体加一条无害语句）一个都不通过，`unchanged` 与 `restored` 都为假，检验「改了别处、缺陷还在」不会被算成什么也没改；
broken 100% compile_syntax；wrong 一个都不通过且归因等于 stub-kill 记录；每个被杀死了变异体的（目标, 级别）都出了题，出题率不低于 90%；
回放复核（`check-mutants --oracle`）里变异体都仍被杀死、没有一个通过，去注释的原函数都通过。

### 3.3 跑真实语料

```bash
export DEEPSEEK_API_KEY=...            # 只从环境变量读，不写进任何文件、日志或结果

python3 -m bench prepare   --corpus cjson sds
python3 -m bench coverage  --corpus cjson sds
python3 -m bench targets   --corpus cjson sds
python3 -m bench obfuscate --corpus cjson sds
python3 -m bench verify    --corpus cjson sds
python3 -m bench describe  --corpus cjson sds --model deepseek-v4-pro
python3 -m bench generate  --corpus cjson sds --model deepseek-flash --levels L0 L1 L2 L3
python3 -m bench evaluate  --corpus cjson sds --model deepseek-flash --tag pilot
python3 -m bench report    --tag pilot
```

* `--corpus` 可多选，省略则是 `corpus/specs/` 下的全部 9 个语料；`--seed` 默认 20261027；`--jobs` 是沙箱与 LLM 调用的并发度。
* 正式调用前先 `python3 -m bench generate --dry-run ...`：只估算还要发多少次请求（扣掉已有回复）、去重后有多少个不同的请求、
  提示 token 约多少，不构造模型、不写文件。L1–L3 的提示词逐字相同（见第 8 节），共用同一个缓存键，所以四级全开时真实请求数约为任务数的一半。
* `describe` 与 `generate` 只有真有活要干时才需要 API key：输出都已是最新的，纯离线续跑、复核不会因为没设 key 报错。
* 模型名：设计文档里的 `deepseek-chat` / `deepseek-reasoner` 在 DeepSeek 官方 Change Log 里有 2026-07-24 起弃用的通知，
  是否还能调用以官方当前文档为准；默认和预设改用 `bench/llm.py` 的 `PRESETS`（`deepseek-flash`、`deepseek-flash-thinking`、
  `deepseek-v4-pro`、`deepseek-v4-pro-thinking`），旧名字只作为 legacy 预设保留，使用时给出警告。
  `mock-oracle` / `mock-broken` / `mock-wrong` / `mock-empty` 是离线自测用的假模型（`mock-unchanged`、`mock-reformat`、`mock-touched` 只用于缺陷修复任务：
  原样返回变异体、返回换了排版加了注释的原函数、返回加了一条无害语句的变异体），只有 `--unsafe-debug` 才允许写进真实工作区。
  其它 OpenAI 兼容端点在 `PRESETS` 里加一项即可。
* 多样本：`--samples N` 需要 `--temperature > 0`；样本 i 把 i 当 seed 参与缓存键，所以每个样本单独请求、单独缓存。
  DeepSeek 预设不把 seed 发给 API（文档里没有这个参数），样本之间的差异来自 temperature 采样。
  `--no-cache` 不读 LLM 缓存（照常写入）：`--force` 只作废已有记录，缓存命中仍会原样复用旧回复，要重新采样得两个一起加。
* 语料来源：设置了环境变量 `CORPUS_MIRROR` 且那里有对应的本地克隆时，`prepare` 直接复制它、不联网；没设置就按 spec 的 url + commit
  克隆（不会去猜任何本机目录）。git 联网时，环境变量里配了代理就先探测：连得上照常用，一个都连不上才临时绕开
  （`CORPUS_KEEP_PROXY=1` 强制保留，`CORPUS_NO_PROXY=1` 强制清掉）；调模型的客户端默认走环境变量里的代理，`BENCH_LLM_NO_PROXY=1` 关掉。
* 评测中途被打断（Ctrl-C、进程被杀）可以直接续跑：已落盘的产出按输入哈希复用，结果文件里留下的半行会被识别并计数。

### 3.4 缺陷修复任务

```bash
python3 -m bench mutate        --corpus cjson sds                  # 出题：在每级副本的目标函数里注入缺陷（先要有 verify 的产出）
python3 -m bench check-mutants --corpus cjson sds --oracle         # 回放复核：变异体都仍被杀死，去注释的原函数都通过
python3 -m bench generate  --corpus cjson sds --model deepseek-flash --task repair
python3 -m bench evaluate  --corpus cjson sds --model deepseek-flash --task repair --tag pilot
python3 -m bench report    --tag pilot                             # 默认 --task all：生成与修复两张并列的表
```

* **题目。** 模型拿到依赖声明和该级副本里**注入了一处缺陷的完整函数**（已去掉注释），没有功能规格、没有测试输出、不说明缺陷在哪，
  要返回修正后的完整函数（签名不变、不新增 `#include`）；判定与生成任务完全相同（替换 → 语法检查 → 构建 → 项目测试 → 归因），通过 = 项目测试全过。
  提示词模板在 `bench/prompt.py`（`REPAIR_PROMPT_VERSION` / `REPAIR_PROMPT_DIGEST`，与生成任务的 `PROMPT_VERSION` / `PROMPT_DIGEST` 分开）。
* **变异算子。** 八个：ROR（关系算子替换）、NEG（条件取反）、LOR（逻辑算子替换）、OBO（差一）、DGS（删守卫）、AOR（算术算子替换）、
  RVR（返回值替换）、DST（删语句），在目标函数体上按 AST 找位点、按字节偏移改写，改写后整份文件必须能用 libclang 零新增 error 重新解析。
  宏实参里的、宏生成的、含预处理指令行的位点一律不动。
* **选择与 kill 检查。** 每个（目标, 级别）把全部位点按固定顺序洗牌（由 `--seed` 和目标决定，**种子里没有级别**），依次把变异体当作候选
  走一遍完整 evaluate：编译通过且在**测试阶段**失败（`test_fail` / `runtime_crash` / `runtime_timeout`，记为 `kill_mode`）才算杀死、才出题；
  测试仍全过的是等价变异体，编译失败的不算杀死；最多试 `--max-tries`（默认 12）个位点，试满仍没有杀死就放弃这个（目标, 级别）并计数。
  只处理 oracle gate 通过的（目标, 级别）。
* **复核。** 判为杀死的候选再评测一次，第二次仍然失败才出题（记 `kill_recheck`）；第二次通过了的（崩溃 / 超时 / OOM 的结局会随负载变，
  删赋值留下的未初始化变量也是）计入 `unstable_count`，不出题、继续试下一个位点。复核在同一并发度下做，只能挡掉一部分偶发情况；
  要在安静的机器上再确认一遍，用 `check-mutants`（`--jobs 1`）。
* **跨级别：同一个缺陷。** L1 只改标识符，位点表与序号一一对应，所以 L0 先选，L1–L3 先试 **L0 选定的那个位点在本级的对应位点**，再优先试同一个算子的位点
  （`same_operator_as_L0`）。L1 的对应是精确的；L2 / L3 改写了函数体，靠记号序列对齐（标识符一律当作同一个记号，
  位点的记号和前后各一个记号都对得上才算），记号被改掉的位点（`x == 0` 换成 `!x`、`a < b` 换成 `b > a`、`if` / `else` 交换后的条件）
  对不上，就退回到只复用算子。选中的变异体记 `same_site_as_L0`；OBO 的 ±1 也按 L0 位点的序号决定，同一个缺陷在各级改成同一个方向。
  **「四级同算子」子集恒等于「四级都出了题」，不构成对照**（复用算子保证了这一点）；报告里的**可比子集**是四级同一个位点、提示词上下文又没有被裁条目的目标
  （各自的个数由 `python3 -m bench.experiment_record` 从数据文件算出，写在 `docs/实验记录.md` 里）。
* **去注释。** 符号混淆只改标识符，L1 以上函数体注释里仍是项目原有的名字，原样放进提示词会泄漏本该隐藏的名字；
  所以四个级别的变异体都先去注释再做 kill 检查，函数里有注释的还要先把去注释后的原函数放回副本再跑一遍项目测试，不通过就放弃这个（目标, 级别）
  （不搜索，也不拿它当 L1–L3 的参照）。字符串字面量里的原名无法去掉（改了就改变行为）：目标函数自己的原名出现在字面量里的 L1 以上的题出题时作废
  （`name_in_literal`），其余只能如实统计。
* **提示词上下文。** 依赖声明原本取自原函数，缺陷删掉的调用、宏、全局还留在里面等于把答案线索写进提示词，所以出题时按注入缺陷后的函数重新裁剪
  （`prompt.prune_context`，只裁原型、宏和全局，类型定义不裁），被裁掉的名字记在任务记录的 `context_orphans` 里；这样的题修复它可能需要提示词里没有的名字，
  难度来自信息缺口而不是混淆，不进可比子集，报告里单列通过率。
* **不可靠的题。** `unchanged` 的候选（模型什么也没改）却通过了项目测试，说明变异体没有被测试稳定地杀死：`evaluate` 点名并让阶段失败，
  `report` 把这道题整题（所有模型、所有样本）剔除并在开头点名。
* **结果记录。** 每条多 `task`（`repair`）、`mutation`（算子、位点序号、原文、改成了什么）、`kill_mode`（变异体怎么死的）、`stub_kill_mode`（目标被零值桩杀死的方式）、
  `unchanged`（候选与变异体规范化后相同，即模型什么也没改）、`restored`（候选与该级原函数规范化后相同，即逐字还原）、`same_operator_as_L0`、`same_site_as_L0`、
  `context_orphans`。规范化只忽略排版与注释，标识符原样比较。修复任务里 `restored` 与相似度**不是记忆指标**（提示词里本来就是只差一处的原函数，
  单点缺陷的任何正确修复都逐字等于原函数），要看级别之间通过率的变化。
* **回放复核 `check-mutants`。** 把 mutate 选定的变异体当作候选再评测一遍（相当于 `mock-unchanged` 的评测，但不经过 generate / evaluate，不写评测结果）：
  都要在测试阶段失败，杀死方式有没有变也记下来（两种结局都算被杀死，只影响分层标签）；`--oracle` 另把去注释后的原函数放回副本，都要通过。
  结果写进 `results/mutants-<语料>.check.json`，只要有一个变异体回放通过了测试，命令就返回非零。出题与评测前核对 mutate 产出有没有过期，对不上直接报错。
* **`--task`。** `generate` / `evaluate` / `report` 都有 `--task {generate,repair,all}`，默认 `all`：生成任务照常处理，
  修复任务只在 mutate 已经有产出时一并处理（没有就给出提示并跳过，免得只想做生成任务的人被拦住）；`--task repair` 在没跑 mutate 时直接报错并提示先运行 mutate。
  `report` 里哪一类任务没有结果文件就只出另一类。生成任务的 `--tag` 与模型名不能以 `repair-` 开头（会与修复任务的 `results-repair-<tag>.jsonl` 等文件撞名）。
* **局限。** 见第 8 节最后几条：变异体是人工注入的单点缺陷，不给测试输出，崩溃型 / 超时型杀死的目标测试可能不检查返回内容，提示词上下文被裁剪过，
  修复任务的相似度和 restored 率不是记忆指标，NEG 缺陷的显眼程度随级别变化，字符串字面量里的项目词汇去不掉。

## 4. 命令参考

`python3 -m bench <cmd>`，所有命令都接受 `--corpus`、`--seed`、`--jobs`、`--force`（忽略已有产出）、`--workspace DIR`
（把 results、语料副本、混淆副本、临时目录都放进另一个目录）、`--image` / `--cpus` / `--memory`（沙箱参数）、`--unsafe-debug`、`-v` / `--verbose`。

`--task` 的取值：`generate`（功能规格生成任务）、`repair`（缺陷修复任务，先运行 `mutate`）、`all`（默认，两类都处理；修复任务在 mutate 还没有产出时给出提示并跳过）。
`--tag` 与 `--model` 的取值不能以 `repair-` 开头（会与修复任务的数据文件撞名，当作用法错误拒绝）。

退出码：0 成功；1 某阶段失败（含 evaluate / report 发现缺样本或坏行）或 e2e 断言不成立；2 用法错误或缺前置条件
（缺上游产出、没有 API key、docker 不可用等）；130 被 Ctrl-C 中断（已落盘的产出可以续跑）。
未预期的运行期故障（沙箱不可用、libclang 解析失败等）返回 1，完整堆栈追加写在工作区 `results/bench-error.log`。

`--unsafe-debug` 是唯一「放宽可信度措施」的开关（`targets` 可以跳过 stub-kill、允许 mock 描述、放行没做过 stub-kill 的目标），
只给调试用，产出不可用于正式报告，报告开头会写明。

| 命令 | 作用 | 主要选项 | 产出 |
|---|---|---|---|
| `prepare` | 取源码并在沙箱里确认构建与测试可用 | `--src DIR` 本地源码，`--no-build-check` | `corpus/src/<name>`，`results/prepare-<name>.json` |
| `coverage` | 带 `--coverage` 构建，gcov 取函数执行次数 | | `results/coverage-<name>.json` |
| `targets` | 选目标：静态过滤、执行次数、按 seed 抽样、stub-kill | `--max-per-corpus`，`--min-statements`，`--min-lines`，`--max-lines`，`--no-exec-filter`，`--no-stub-kill`（要同时加 `--unsafe-debug`），`--lazy` | `results/targets-<name>.jsonl`、`.summary.json` |
| `obfuscate` | 生成混淆副本并 verify | `--level` / `--levels L0 L1 L2 L3`，`--strict-stringify auto/on/off` | `results/obf-<name>-<level>.json`，`tmp/obf/<name>/<level>/` |
| `verify` | oracle gate：逐目标用混淆后的原函数走完整 evaluate | `--level` / `--levels` | 写回 `obf-*.json` 的 `oracle_gate` |
| `mutate` | 缺陷修复任务的出题：在每级副本的目标函数里注入缺陷并做 kill 检查，判为杀死的再复核一次（需要先 verify） | `--level` / `--levels`，`--max-tries`（默认 12） | `results/mutants-<name>-<level>.jsonl`、`results/mutants-<name>.summary.json` |
| `check-mutants` | 回放复核 mutate 选定的变异体：都要仍然在测试阶段失败，杀死方式有没有变 | `--level` / `--levels`，`--oracle`（另把去注释的原函数放回副本，都要通过） | `results/mutants-<name>.check.json` |
| `describe` | 为每个目标生成不含标识符的功能规格 | `--model` | `results/descriptions.jsonl` |
| `generate` | 出题并调模型 | `--model`（可多个），`--levels` / `--level`，`--samples`，`--temperature`，`--max-tokens`，`--dry-run`，`--no-cache`，`--task` | `results/tasks.jsonl`，`results/generations-<model>.jsonl`，`results/generation-failures-<model>.jsonl`（修复任务见第 6 节） |
| `evaluate` | 评测回复：后处理、替换、编译、测试、归因、相似度 | `--model`（可多个），`--levels` / `--level`，`--tag`，`--allow-missing`，`--task` | `results/results-<tag>.jsonl`，`results/completeness-<tag>.json`（修复任务见第 6 节） |
| `report` | 汇总指标（两类任务并列） | `--tag`，`--task`，`--format md csv svg`，`--allow-stale`，`--allow-missing` | `results/summary-<tag>.md` / `.csv`，`results/fig-<tag>.svg` |
| `e2e` | mini 夹具上的全链路自测 | `--backend mock`，`--keep`，`--levels` | `tmp/e2e/` |

### 环境变量

| 变量 | 用途 |
|---|---|
| `BENCH_ROOT` | 项目根（默认由 `bench/store.py` 的位置推出） |
| `BENCH_RESULTS_DIR` / `BENCH_CORPUS_SRC` / `BENCH_OBF_DIR` / `BENCH_WORK_DIR` | 结果目录、语料目录、混淆副本目录、临时工作目录（默认分别是 `results/`、`corpus/src/`、`tmp/obf/`、`tmp/work/`） |
| `BENCH_SPEC_DIR` | spec 目录（默认 `corpus/specs/`） |
| `LIBCLANG_PATH` | libclang 库文件 |
| `BENCH_CLANG_INCLUDE` | libclang 找不到编译器内置头（stddef.h 等）时，显式指定内置头目录（`os.pathsep` 分隔） |
| `CORPUS_MIRROR` | 本地克隆镜像目录，`prepare` 优先复制；没设置就不用镜像 |
| `CORPUS_GIT_TIMEOUT` | 联网 git 命令的超时秒数（默认 180） |
| `CORPUS_KEEP_PROXY` / `CORPUS_NO_PROXY` | git 联网时强制保留 / 强制清掉环境变量里的代理（都不设则探测：代理连不上才绕开） |
| `BENCH_LLM_NO_PROXY` | 调模型 API 时不走环境变量里的代理 |

模型的 API key 只认环境变量（`DEEPSEEK_API_KEY`，见 `PRESETS` 里每个预设的 `api_key_env`），不写进任何文件、日志、缓存或结果。
只给测试和 CI 用的变量见第 11 节。

## 5. 混淆级别

高一级包含低一级的全部变换。三类变换都由 `--seed` 决定，同种子同结果（符号混淆的改名表还依赖宿主机的系统头，见第 8 节最后一条）；
每个副本都必须通过项目自带测试（verify）。

| 级别 | 变换 | 要点 |
|---|---|---|
| L0 | 原样复制 | 对照组 |
| L1 | 符号混淆：项目内声明的函数、变量、参数、字段、类型、枚举常量、标签、宏的拼写整体改成不透露原意的 `f_<n>` `t_<n>` `v_<n>` `M_<n>` `m_<n>` | token 级替换，只改标识符 token，不碰字符串字面量与注释；系统头里出现过的拼写、`main`、关键字、第三方框架（`spec.vendored`）、被构建命令用 `-D` 指定的宏（`spec.keep_names`）等一律不改；遇到字符串化宏先非 strict 验证，不过再退回 strict，记录里写明用的哪种，以及非 strict 下被字符串化后仍被改名的拼写个数 |
| L2 | 结构混淆：`for` 改 `while`、`if/else` 交换分支、复合赋值展开、三目 `return` 改 `if/else`、多声明拆分 | 每个函数体内约 50% 的可用位置；宏生成的语句、宏参数里的位置不动；改完必须能零 error 重新解析 |
| L3 | 语义混淆：`a < b` 改 `b > a`、De Morgan、`!x` 与 `x == 0` 互换、`a + b` 交换律换序、函数体开头插入 opaque 语句 | 只对无副作用的简单操作数、整型与算术类型下手；同样要零 error 重新解析 |

## 6. 数据文件

JSON / JSONL，都带 `schema_version`。阶段产出整份原子重写，可断点续跑的（generations、results）按行追加。
每条产出带 `inputs`（各输入的内容哈希）与 `input_hash`，幂等判断与 `report` 的校验都靠它；评测相关代码的摘要
（`bench/provenance.py` 按阶段对源文件内容取哈希）、沙箱镜像 ID 和资源上限也在其中，换了镜像或改了评测代码，旧结果会被判为过期。

| 文件 | 内容 |
|---|---|
| `results/prepare-<corpus>.json` | 源码树哈希、沙箱里构建与测试的结果 |
| `results/coverage-<corpus>.json` | 每个函数的执行次数 |
| `results/targets-<corpus>.jsonl` | 入选目标：FunctionTarget 全字段 + `exec_count` + `stub_killed` + `stub_kill_mode`（test_fail / runtime_crash / runtime_timeout） |
| `results/targets-<corpus>.summary.json` | 提取数、候选数、各排除原因的计数与 target_id（含 stub-kill 存活数、各目标的 stub-kill 判定） |
| `results/obf-<corpus>-<level>.json` | `rename_map`、`stats`、verify 结果、oracle gate 结果（检查了哪些目标、过了哪些）、副本源码树哈希、宿主 libclang 环境摘要 |
| `results/descriptions.jsonl` | `target_id`、功能规格、生成模型、泄漏重试次数、`failed`、放行词命中、词组警告、泄漏检查器摘要 |
| `results/tasks.jsonl` | Task 全字段（提示词的全部输入）与消息哈希 |
| `results/generations-<model>.jsonl` | `task_id`、`sample_idx`、回复文本、token 用量、是否命中缓存、延迟、温度 / max_tokens / seed、模型指纹、输入哈希 |
| `results/generation-failures-<model>.jsonl` | 重试用尽仍失败的生成调用（task_id、sample_idx、原因）；之后补成功了会从这里删掉 |
| `results/results-<tag>.jsonl` | EvalResult + `category` + `detail` + `similarity` + 回复规范性问题 + 输入哈希 |
| `results/completeness-<tag>.json` | 每个（模型, 级别, 语料）应有多少样本、评测了多少、缺多少（生成失败 / 没生成 / 回复过期） |
| `results/mutants-<corpus>-<level>.jsonl` | mutate 选定的变异体，每个出了题的（目标, 级别）一行：`target_id`、`level`、`operator`、`site_index`、`original_text`、`mutated_text`、`mutant_code`（变异后的完整函数，已去注释）、`kill_mode`、`kill_recheck`（复核时的归因）、`tried_sites`、`equivalent_count`、`compile_fail_count`、`parse_fail_count`、`unstable_count`、`same_operator_as_L0`、`same_site_as_L0`、`anchor`（L0 位点在本级的去向）、`context_orphans`（提示词上下文里因缺陷而多余的名字）、输入哈希 |
| `results/mutants-<corpus>.summary.json` | mutate 的汇总：各级的输入哈希、出题数、放弃数及原因、算子与杀死方式分布、`same_operator_as_L0` / `same_site_as_L0` 比例、记账（含复核）、各类变异体占的沙箱时间、耗时，以及四级都出了题 / 四级同算子 / 四级同位点的目标数 |
| `results/mutants-<corpus>.check.json` | `check-mutants` 的回放结果：回放了多少、仍被杀死多少、杀死方式有没有变、有没有回放通过的（不合格）、去注释的原函数有没有通过 |
| `results/tasks-repair.jsonl`、`results/generations-repair-<model>.jsonl`、`results/generation-failures-repair-<model>.jsonl` | 缺陷修复任务的任务、回复和失败的调用：结构与生成任务的对应文件一致，多 `task` 与 `mutation` 字段 |
| `results/results-repair-<tag>.jsonl`、`results/completeness-repair-<tag>.json` | 缺陷修复任务的评测结果与完整性记录：多 `task`、`mutation`、`kill_mode`、`unchanged`、`restored`、`same_operator_as_L0`、`same_site_as_L0`、`context_orphans` |
| `results/usage.jsonl` | 每次真实 API 调用的 token 用量（命中缓存的不记） |
| `results/cache/` | LLM 回复缓存，键为 sha256(模型, 端点, messages, temperature, max_tokens, seed)，被 `.gitignore` 忽略 |
| `results/summary-<tag>.md` / `.csv`、`fig-<tag>.svg` | `report` 的输出 |
| `results/bench-error.log` | 未预期故障的完整堆栈（追加写） |

## 7. 可信度措施

| 措施 | 做法 | 自测 |
|---|---|---|
| oracle gate | 每个混淆副本先整体 verify；再用混淆后的原函数文本对每个目标走一遍完整 evaluate，必须 pass，不过的副本或目标作废并计数；结果记下检查过哪些目标，出题时逐个核对，gate 对着另一份目标集 / 副本跑的一律报错 | `tests/test_e2e_mock.py`、`tests/test_integrity.py` |
| stub-kill | 目标函数换成「返回零值 / 空函数体」的桩后项目测试必须失败，否则不入选；存活数写进报告；杀死方式（test_fail / runtime_crash / runtime_timeout）逐目标记录，报告按杀死方式分层给通过率 | `tests/test_targets.py` |
| 硬性约束 | 提示词禁止调用 exit / abort / system 等、禁止联网和汇编、禁止定义宏；`postprocess` 用 libclang 扫候选代码，这些以及会绕过判据的写法（构造函数属性、`#undef` / `#pragma`）一律直接记 `policy_violation`，不进沙箱；每个语料的通过判据除了退出码，还要求测试输出里出现完成标志（汇总行或固定行，必要时限定出现次数），提前 `exit(0)` 过不了 | `tests/test_postprocess.py`、`tests/test_corpus.py`、`tests/test_integrity.py` |
| MockLLM 三模式 e2e | oracle 100% pass、broken 100% compile_syntax、wrong 不通过且归因与 stub-kill 一致；缺陷修复任务另有 unchanged 模式（原样返回变异体，一个都不能通过、`unchanged` 为真）、reformat 模式（原函数换排版加注释，必须通过且 `restored`）和 touched 模式（变异体加一条无害语句，不能通过，`unchanged` 与 `restored` 都为假），oracle 模式还要 `restored`；再加 `check-mutants` 回放 | `tests/test_e2e_mock.py` |
| 变异体 kill 检查 | 缺陷修复任务的每个变异体当作候选走一遍完整 evaluate，编译通过且在测试阶段失败才出题，判为杀死的再评测一次、第二次仍失败才出题；改写后重新解析不能多出新 error；去注释后的原函数先验证（不通过就不搜索）；mutate 的输入哈希含 oracle gate、副本、种子、`--max-tries`、是否复核、相关代码、沙箱和宿主 libclang，使用前（出题、report）把这些再逐项重算核对一遍，对不上直接报错；`check-mutants` 可以随时回放复核；`unchanged` 的候选却通过了测试的题整题剔除并点名 | `tests/test_mutate.py`、`tests/test_repair_pipeline.py`、`tests/test_repair_report.py` |
| 描述泄漏检查 | 功能规格里不得出现任何项目标识符（含 L0 真名）：单词边界精确匹配 + 忽略大小写 / 分隔符的规范化匹配 + 词组警告；常见英文词表放行，放行命中如实记录；出题时用当前检查器重新检查一遍；泄漏则带着清单重新生成，最多 3 次，仍泄漏的目标排除 | `tests/test_describe.py` |
| 全链路确定性 | seed 固定则目标集、改名表（同一台宿主机上）、变换位置、提示词完全一致；LLM 调用 temperature=0 并带缓存 | 各模块的测试 |
| 输入哈希校验 | 结果文件自带源码树、spec（只含影响构建与测试的字段）、prompt 模板、回复、oracle gate、评测代码、沙箱镜像的哈希；`report` 对照 tasks / generations / 副本 / gate 逐条校验，对不上的记录排除并在报告里列出原因 | `tests/test_e2e_mock.py`、`tests/test_integrity.py` |
| 不静默丢样本 | 生成调用失败单独落盘，`evaluate` 写完整性记录，报告里标「不完整」，命令返回非零（`--allow-missing` 放行）；结果文件的坏行同样计数 | `tests/test_integrity.py` |
| 用语与路径检查 | 违例用语、本机绝对路径、README 与实现（环境变量、命令行选项、模块、局限、数据文件）是否一致，全仓库统一检查 | `tests/test_hygiene.py` |

## 8. 局限

如实写在这里，读结果时请对照（与报告的「局限」一节同一份内容，由 `bench/limitations.py` 单点维护，`tests/test_hygiene.py` 检查两边一致）：

1. **功能规格由 LLM 生成。** 规格歧义会抬高所有级别的失败率，所以 L0 通过率只是上限参考；规格里不含标识符是靠泄漏检查保证的，检查口径是单词边界精确匹配加忽略大小写 / 分隔符的规范化匹配，再加常见英文词表放行（放行命中如实记在描述记录里），不是语义层面的保证。
2. **语料是知名小库。** L0 下模型可能已经见过这些代码，这正是设置混淆对照的意义；但 9 个库的流行程度不同，不能把单个库上的差异直接当成记忆的证据，要看 L0 → L1 的相似度与通过率变化是否一致。
3. **只测 C。** 结论不外推到其它语言；构建与测试都是 gcc 13 + 各库自带的测试。
4. **单函数粒度。** 一次只替换一个函数，目标函数以外的代码是原样的（混淆过的）项目代码；不评测多文件新增功能。
5. **测试强度依赖各库自带测试。** stub-kill 只保证「零值桩会被测试抓住」，也就是测试不是空的，不保证测试能抓住所有逻辑错误：通过测试的生成结果仍可能有测试没覆盖到的错误，通过率会被高估。其中一部分目标是被「桩让测试程序崩溃或超时」杀死的（runtime_crash / runtime_timeout），它们的测试并没有断言函数返回了什么，任何不崩溃的错误实现都能通过；报告里按杀死方式分层给出通过率，读结果时优先看 test_fail 型目标。
6. **结构与语义变换覆盖的语法子集有限。** 宏生成的语句、宏参数里的位置、被 #if 劈开的位置、带函数调用的复合赋值等一律不动，宏用得重的库（parson、cJSON 的 Unity 断言宏、vec）实际改动的位置明显少于理论候选数；#if 里未生效的分支、spec.cflags 没有打开的条件编译块里的函数体也不会被改。
7. **L1 / L2 / L3 共用同一批回复。** L2 / L3 只改函数体，而提示词里没有函数体，所以 L1、L2、L3 的提示词逐字相同、temperature=0 时模型回复也相同、共用同一个缓存键。L2 / L3 衡量的是「生成的函数放回被结构、语义混淆过的项目后，是否仍能编译并通过测试」，以及相似度指标上的差异，不是模型对混淆过的函数体的理解力；只有 L0 与 L1 的差异能反映模型行为（提示词里的名字不同）。要让 L2 / L3 改变模型看到的内容，需要在设计上另做安排（例如把被调函数的函数体放进上下文）。缺陷修复任务就是这样一种安排：它的提示词里放的是混淆后的函数体，L2 / L3 的输入与 L1 不同。
8. **OOM 与超时的归因随负载抖动。** 符号混淆只改名字，注释和字符串里仍是原名，所以提示词里不放源码注释；并发评测受本机 CPU 与内存限制（每个沙箱容器 1 个 CPU、至多 1 GB），同时跑的语料不宜过多；会无限分配内存的生成结果，超时（各语料取 30 s）和 1 GB 内存上限谁先到取决于机器负载：前者归 runtime_timeout，后者被 OOM 杀死（rc=137，没有超时标记），按规则归 test_fail，所以这类样本的归因在不同负载下可能不同。
9. **符号混淆的改名表依赖宿主机的系统头。** 符号混淆用宿主机上的 libclang 解析，系统头里声明过的拼写一律不改名；宿主机的 libc 头与借来的编译器内置头目录与沙箱里 gcc:13 的并不相同，所以换一台机器或升级系统后，同一个 seed 的改名表可能不同。混淆结果的输入哈希里带了宿主 libclang 环境的摘要，环境变了旧副本会被判定为过期。
10. **缺陷修复任务的变异体是人工注入的单点缺陷。** 它们不等于真实缺陷的分布；不同级别的变异体不一定是同一个缺陷。mutate 让 L1–L3 先试 L0 选定的位点在本级的对应位点，L1 只改名字，位点一一对应，这一步是精确的；L2 / L3 改写了函数体，靠记号对齐，位点的记号被改掉的退回到只复用算子。报告里的「可比子集」只含四级是同一个位点、提示词上下文又没有被裁条目的目标，只有它上面的级别差异才不混入缺陷位置的差异；「四级同算子」的目标子集恒等于全集（复用算子保证了这一点），不是对照。
11. **缺陷修复任务不给测试输出。** 难度高于真实场景中的修复，通过率的绝对值不可与其它基准比较，只看级别间的差异。
12. **崩溃型与超时型杀死的修复任务，测试可能不检查返回内容。** 修复后只要不崩溃、不超时就可能通过，与生成任务相同的局限。报告给两个口径的分层：变异体怎么死的（kill_mode，这个具体缺陷让断言失败，还是让测试程序崩溃或挂起），和目标被零值桩杀死的方式（stub_kill_mode，测试有没有断言返回内容）；两者不等价，读结果时两张表都要看，优先看 test_fail 型。
13. **修复任务的提示词上下文被裁剪过。** 依赖上下文原本取自原函数，缺陷删掉的调用、宏、全局还留在里面会泄漏答案线索，所以出题时按注入缺陷后的函数重新裁剪（只裁原型、宏和全局变量，类型定义不裁）。被裁掉条目的题，修复它可能需要提示词里没有的名字，难度来自信息缺口而不是混淆，不进可比子集，通过率单独成表。
14. **修复任务的相似度和 restored 率不是记忆指标。** 提示词里本来就是只差一处的原函数，候选只要没大改，相似度天然接近 1；单点缺陷的任何正确修复都逐字等于原函数，restored 约等于通过，与 L0 凭记忆还原无法区分。修复任务回答的问题要看级别之间通过率的变化。
15. **NEG 缺陷的显眼程度随级别变化。** NEG 的改写固定是 !(c)：L0 / L1 的代码里这种写法几乎不出现，缺陷一眼可见；L2 的 if / else 交换会让代码自带这种写法，缺陷被掩盖。NEG 题的难度随级别变化有一部分与混淆想测的因素无关，读按算子分层的表时要单独看这一行。
16. **字符串字面量里的项目词汇去不掉。** 改字面量会改变行为，所以 L1 以上的变异体函数文本里，字符串和字符常量里的项目词汇（错误提示里的函数名之类）原样留着；目标函数自己的原名出现在字面量里的题出题时作废（name_in_literal），其余只能如实统计。

## 9. 实验结果

数据截至 2026-10-28。完整表格见 `results/summary-v2.md`（逐样本记录 `results/results-v2.jsonl` 与 `results/results-repair-v2.jsonl`），解读见 [`docs/实验结果.md`](docs/实验结果.md)。入选目标 208 个；规格生成任务里 11 个目标的功能规格重试后仍泄漏项目标识符而作废，进入统计 197 个；缺陷修复任务有 3 题因目标自己的原名出现在字符串字面量里作废，L0 208 题、L1–L3 各 207 题。temperature 0、pass@1。

### 规格生成任务：模型 × 混淆级别（通过率 = 替换回项目后项目自带测试全过）

| 模型 | L0 | L1 | L2 | L3 | L0→L1 配对差 | McNemar p |
|---|---:|---:|---:|---:|---:|---:|
| deepseek-flash | 85.3% | 49.2% | 48.7% | 49.2% | −36.0 pp | < 0.001 |
| deepseek-v4-pro | 80.2% | 51.3% | 50.8% | 50.8% | −28.9 pp | < 0.001 |
| deepseek-flash-thinking | 90.4% | 69.0% | 71.1% | 70.6% | −21.3 pp | < 0.001 |

L1 只把项目里的标识符（含测试文件）统一改成无意义符号，功能规格与签名类型不变；L2 / L3 的变换发生在模型看不到的函数体里，所以与 L1 几乎相同（见局限 7）。候选与原函数的相似度（标识符已归一）flash 0.825 → 0.701、v4-pro 0.776 → 0.714、thinking 0.845 → 0.802。thinking 在 L1 有 10.7% 的回复推理到 32k token 上限被截断，按失败计，69.0% 是下限。

### 缺陷修复任务：可比子集（150 个目标，四级是同一个缺陷且提示词上下文未被裁剪）

| 模型 | L0 | L1 | L2 | L3 | L0→L1 | L0→L3 | unchanged 率（L1） |
|---|---:|---:|---:|---:|---:|---:|---:|
| deepseek-flash | 92.7% | 70.7% | 70.7% | 72.0% | −22.0 pp | −20.7 pp | 6.0% |
| deepseek-v4-pro | 92.7% | 66.7% | 64.0% | 57.3% | −26.0 pp | −35.3 pp | 18.0% |

全集（L0 208 题、其余各 207 题）：flash 85.6% / 63.8% / 65.2% / 63.8%，v4-pro 83.2% / 59.9% / 58.5% / 52.7%。unchanged = 模型把带缺陷的函数原样返回。deepseek-flash-thinking 的修复任务结果跑完后补入。

报告开头逐条说明两类任务的输入差异：

* 生成任务（generate）：模型拿到依赖声明、目标函数的签名和 LLM 写的功能规格，没有函数体，要写出整个函数；L1 / L2 / L3 的提示词逐字相同，只有 L0 与 L1 的差异能反映模型行为。
* 缺陷修复任务（repair）：模型拿到依赖声明和该级副本里注入了一处缺陷的完整函数（已去掉注释），没有功能规格、没有测试输出、不说明缺陷在哪，要返回修正后的完整函数；混淆作用在模型读到的函数体上，L2 / L3 的输入与 L1 不同，四个级别都能反映模型行为。
* 两类任务的题型不同、测试对它们的灵敏度也不同，通过率的绝对值不能直接比较，要看各自在级别之间的变化：生成任务回答「写函数是否依赖记忆」，修复任务回答「读懂并修好函数是否依赖表面形态」。

### 读法

1. 两类任务上，去掉标识符后两到三个模型都显著下降；修复任务里 v4-pro 在 L2、L3 继续下降，对结构与语义改写比 flash 敏感。
2. flash 在原名下领先 v4-pro（生成任务 +5.1 pp），改名后反落后（−2.1 pp）：原名下的优势来自记忆。
3. 越知名的库下降越大（生成任务 flash：tiny-regex-c −62.5、mpc −55.1、cJSON −42.2 pp；vec、inih 不降）。
4. 成本：全部实验约 35 元，thinking 占一半以上。

## 10. 目录结构

```
bench/
  __init__.py __main__.py         包入口（Python 版本检查）、python3 -m bench
  vocab.py limitations.py provenance.py  共用词表（级别 / 回复问题 / 归因类别）、局限列表（README 与报告共用）、代码与沙箱摘要
  corpus.py sandbox.py clangcfg.py  语料 spec 与源码准备、Docker 沙箱、libclang 配置（解析参数的唯一出口）
  extract.py coverage.py targets.py  函数提取与依赖闭包、执行次数、目标选择与 stub-kill
  mutate.py                       缺陷修复任务的八个变异算子（找位点、改写）、跨级别的同一个位点（记号对齐）与变异体选择、kill 检查与复核
  obfuscate/                      __init__（总入口）symbols structure semantic verify _edits（结构 / 语义变换共用的改写基础设施，含括号配对与宏调用区间，mutate 也用）
  llm.py prompt.py describe.py postprocess.py  模型调用与缓存、提示词（含修复任务的上下文裁剪）、功能规格、回复后处理与硬性约束检查
  runner.py attribute.py similarity.py  替换与沙箱评测、错误归因、词法器与归一化相似度（含 unchanged / restored 用的规范化比较）
  metrics.py report.py            指标汇总与报告
  experiment_record.py            把 prepare 到 verify 以及 mutate、check-mutants 的产出汇总成 docs/实验记录.md 的数据部分
  store.py pipeline.py cli.py     工作区与数据文件读写、各阶段实现、命令行
corpus/specs/*.json               9 个语料的 spec（构建与测试命令、过滤、vendored、keep_names、exclude_targets、allow_words）
corpus/src/                       语料源码副本（prepare 生成，不入库）
tests/                            单元测试与 e2e（fixtures/mini 是自写的小项目；conftest.py / helpers.py 是共用夹具）
scripts/ci.sh                     CI 脚本：缺 docker 或语料就失败，不允许有跳过的用例
pyproject.toml                    打包与 pytest 配置（Python >= 3.10，运行时只依赖 clang==14.0）
results/                          数据文件（缓存目录不入库）
docs/                             设计文档（含缺陷修复任务的附录）、实验记录
docker/                           预留目录，目前为空（沙箱直接用官方 gcc:13 镜像）
tmp/                              中间产物（混淆副本、沙箱临时目录、e2e 输出），不入库
```

## 11. 开发与测试

```bash
python3 -m pytest tests -q                       # 全量，需要 docker 与 gcc:13；开发机上实测约 20 分钟（pytest 结束时打印总用时）
python3 -m pytest tests/test_e2e_mock.py -q      # 集成：指标、报告、命令行、mini 上的全链路 e2e（含缺陷修复任务）；同一台机器上实测约 11 分钟
BENCH_REAL_CORPORA=quick python3 -m pytest tests/test_symbols.py -q   # 符号混淆只在 4 个真实语料上跑；=0 全部跳过，默认 9 个都跑
python3 -m pyflakes bench tests                  # 静态检查，必须零告警
python3 -m bench.experiment_record --update docs/实验记录.md   # 用 results/ 里的产出重新生成实验记录的数据部分
bash scripts/ci.sh                               # CI：pyflakes + 全量 pytest（缺 docker / 语料即失败，有跳过即失败）+ e2e
```

沙箱用例（`needs_docker` 标记）在 docker 或镜像不可用时默认**跳过**，套件仍显示通过；
设 `BENCH_REQUIRE_DOCKER=1` 改为失败，`BENCH_REQUIRE_CORPUS=1` 让找不到真实语料时也失败。CI 必须同时设这两个（`scripts/ci.sh` 已设），
否则整组可信度措施的自测在没有 docker 的机器上悄悄跳过。

上面的耗时是 16 核 / 15 GB 的开发机上的实测（全量 20 分 05 秒，其中 `test_e2e_mock.py` 约 650 秒），随机器和负载变化。想自己量：
`python3 -m pytest tests -q --durations=15 --junitxml=tmp/pytest/junit.xml`，结束时打印总用时和最慢的 15 个测试，XML 里每个测试有 `time`，按文件加起来就是各文件的耗时。

只给测试用的环境变量：`BENCH_REQUIRE_DOCKER`、`BENCH_REQUIRE_CORPUS`、`BENCH_SANDBOX_IMAGE`（沙箱镜像，默认 `gcc:13`）、
`BENCH_REAL_CORPORA`（`quick` / `0`，见上）。临时文件都放在项目 `tmp/` 下，不碰系统临时目录。
