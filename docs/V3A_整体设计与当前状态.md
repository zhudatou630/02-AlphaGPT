# V3A 整体设计与当前状态

更新时间：2026-07-12

> 这是一份面向项目协作和人工审核的总览文档。它解释研究问题、系统结构、阶段门禁、代码分工和当前实际状态。
>
> 本文是当前个人探索方向的总览，不替代已经生成的历史 protocol 或数据 artifact。旧 Stage D protocol、binding 和 pilot shell 保留为历史冻结件，不再作为当前主路径；任何新的 Stage D 或 formal run 都必须新建并单独批准 protocol/binding。当前默认公式库整理规则已写入 `export_top_formulas.py` 及其 curated artifact。
>
> - `docs/V3A_单公式相对选择技术规格.md`
> - `docs/V3A_StageD_GPU实验协议.md`
> - `configs/v3a_stage_d_gpu_pilot.json`
> - `configs/v3a_stage_d_formal_topn.json`
> - `data/processed/v3a/stage_d/pilot_binding.json`

## 1. 先看结论

V3A 是一个个人研究项目，当前目标是先把“自动发现公式并形成候选库”的流程跑通，供后续结合真实交易思路继续探索。它不是生产系统，也不是马上交付的商业项目，因此当前优先级是：简单、可理解、能复现、方便继续试验。

当前主线应当是：

```text
固定数据和研究问题
    -> 定义公式语言和评分标准
    -> 用随机搜索/Transformer 生成公式
    -> 保存完整账本和 top-N 公式库
    -> 后续再结合交易思路研究、验证和组合
```

2022 validation、2023+ OOS、纸面交易和实盘不是当前 pilot 的必经步骤，而是以后研究问题明确后再逐步加入的过渡层。

当前实际状态：

| 部分 | 状态 | 说明 |
|---|---|---|
| V3A 数据、因子、公式 VM、scorer | 已实现并通过测试 | 研究输入和计算链路已能运行 |
| Stage C GPU 工程验证 | 已完成 | CUDA smoke 及随机 baseline 已完成 |
| Stage D Transformer pilot 训练 | 已完成 | legacy 与 reward-calibration 两个 `50000/50000` run 均完成，checkpoint/resume 已验证 |
| top-N 公式库输出 | 已完成 | 同时保留 raw top 30/50 和 curated top 30/50 |
| 全量信号相似度/聚类漏斗 | 暂停 | 不再作为当前主路径 |
| Formal Transformer + matched-random | protocol/binding 已写入，尚未启动 | 3+3 seeds、每 seed 1,000,000 attempts；远端 run 尚未创建 |
| 2022/2023+ 样本外验证 | 未开始 | 等后续交易问题明确后再考虑 |
| 纸面交易、执行、实盘 | 未开始 | 不属于当前个人探索阶段 |

这次 Transformer 训练没有形成策略结论，但已经提供了真实训练和 resume 的工程证据。两个 pilot 都已从完整账本导出 raw/curated top 30/50。formal protocol 和 binding 已写入并完成身份核对，但远端部署和启动仍需单独批准；不再让复杂候选漏斗阻塞公式库产出。

## 2. V3A 要回答的研究问题

当前 V3A 要先回答的是一个框架问题：

> 在固定的 35 只 ETF 和一套固定公式语言下，我们能否稳定地生成、评分、保存一批可供后续研究的公式？

单条公式的相对选择能力仍然是 scorer 的基本评价单位，但不是当前项目最终要选出的唯一答案。最终希望得到的是一个可继续研究的公式库，而不是一条“最牛公式”。

公式不是直接预测某只 ETF 的价格，而是对每只 ETF 产生一个分数。每天根据分数排序，选择排名靠前的 ETF。当前版本始终持仓，暂不研究：

- 空仓择时；
- 动态 ETF 池；
- 自动构建多公式组合；
- 市场状态切换；
- 完整实盘策略。

当前 V3A 的“基本研究对象”是：

```text
多条候选公式
    -> 每条公式对 35 只 ETF 产生信号
    -> 每条公式按固定 scorer 得到训练期 reward
    -> 保存 top-N 公式及其完整元数据
    -> 留给后续交易思路、组合和验证使用
```

这里仍然会用“单公式对 ETF 池排名”作为基本计算单位，但不会把它误解成最终只保留一条公式。

## 3. 为什么要分层设计

即使是个人探索，也需要把几件容易混淆的事情分开：

1. 数据和公式计算要固定，否则每次结果没有可比性；
2. 训练期 reward、样本外表现和真实交易表现不是一回事；
3. 公式生成、公式保存和后续人工研究可以先分开，不必一次做成完整交易系统。

因此 V3A 只保留当前有用的几个层次，暂不把所有未来可能需要的功能都实现出来。

```text
数据层
  我们当时究竟知道什么？哪些 ETF 当时可以交易？

公式语言层
  模型允许表达哪些因子和运算？

评分层
  什么叫一条公式在训练期表现好？

搜索层
  Transformer 或随机方法能否找到高评分公式？

公式库层
  如何把训练结果简单地保存成 top-N 公式，供后续研究？

后续验证层
  以后加入交易思路后，训练期有效的公式到了未见过的数据是否仍有效？

执行层
  加入 T+1、成本、滑点、停牌和真实延迟后能否执行？
```

监工不属于研究层，它只是运行辅助：观察远端任务是否还活着，在异常或完成时通知本地 Agent。

## 4. 研究流水线的每一层

### 4.1 数据层：先确定可见信息

数据层把多只 ETF 的日频行情整理成统一的时间和资产矩阵，并保存：

- 事件复权的绝对 OHLC；
- 数据治理用的相对价格；
- ETF symbol 和统一日期；
- 每个资产每天是否可交易的 `tradable_mask`。

公式和 scorer 不得读取不可交易日的值，也不能把尚未上市或当时停牌的 ETF 当成可选资产。

**为什么要有独立 train-view：**

Stage D 训练只能读取训练区间。当前 train-view 只包含 `2016-08-09` 到 `2021-12-31` 的数据，且文件中没有 2022 和 2023+ 的数组。这样可以在工程上阻止训练代码误读未来数据，而不只是依赖口头约定。

### 4.2 公式语言层：限定搜索空间

公式由固定 token 组成。当前包含：

- 40 个基础因子实例；
- 5/10/20/40/60 日窗口；
- 加、减、乘、绝对值、符号、延迟、均值等运算；
- 固定的后缀 RPN 语法；
- 最大 15 个公式 token；
- action mask，保证栈结构和参数顺序合法。

Transformer 和随机 baseline 必须使用同一套合法 action 集合。否则随机组和 Transformer 组搜索的根本不是同一个问题。

**为什么不一开始把所有可能的因子和横截面算子都塞进去：**

词表越大，搜索空间越大，模型越难学，结果也越难审计。当前固定小 ETF 池先使用纯时序因子；横截面 token 作为以后池子扩大或动态化时的扩展，不是当前地基。

### 4.3 Scorer 层：定义训练目标

当前 scorer 的核心思想是：

1. 在 t 日用当时可交易的 ETF 和公式信号排序；
2. 选择 top 分位，当前规则为可选资产数的 20%，并限制在 2 到 4 只；
3. 从 t+1 开盘买入；
4. 按固定的 10 日预测视界计算未来收益；
5. 与同一可选池的平均收益比较；
6. 对训练期日期做等权平均，得到 reward。

这个 reward 主要衡量“信号有没有把相对强的 ETF 排到前面”，不等于完整交易净收益。

成本、滑点、退出规则、停牌和涨跌停等执行摩擦留在后面的验证层，而不是先全部焊进 Transformer 的训练目标。

**原因：**训练需要一个稳定、可大量调用的目标；验证需要一个更接近实战的目标。把两者混在一起，会很难判断模型是在学信号，还是在利用某个执行模拟细节。

### 4.4 搜索层：Transformer 和随机方法

搜索层有两条方法，但共用同一套公式语言、VM、scorer 和质量门禁：

```text
Transformer：生成 token -> VM 执行 -> scorer reward -> REINFORCE 更新
随机 baseline：按同一合法 action 集合随机生成 -> VM 执行 -> scorer reward
```

Transformer 的作用不是直接预测价格，而是学习“哪些公式结构更容易产生高 reward”。

随机 baseline 的作用是回答：

> Transformer 的结果是否真的超过了相同预算下的随机搜索？

如果没有随机对照，单看 Transformer 的最高分没有可靠的解释。

### 4.5 公式库层：raw 账本加 curated top-N，不把漏斗设为门槛

训练过程中会生成大量公式。当前个人探索阶段最需要的是把这些结果保存下来，供后续人工阅读和交易思路研究，而不是马上构建一个完美的多样性筛选系统。

当前默认流程是：

```text
完整 attempt ledger
    -> canonical 公式去重
    -> 按训练 reward 和固定 tie-break 保存 raw top 30 / top 50
    -> 对展示表达式做安全清理
    -> reward 差距不超过 5e-6 时优先较短表达式
    -> 保存 curated top 30 / top 50
```

raw top-N 是原始 reward 排名，curated top-N 是人工阅读入口。Transformer 和未来 random baseline 都应使用这套整理政策。两者都保留原始 token、canonical hash、reward、质量摘要、首次出现位置和训练身份；curated 另外记录清理后的表达式、展示长度、原始 reward 排名和换位原因。

这两个步骤不改变训练 reward、不改变完整 ledger、不改变 canonical hash，也不把短公式说成一定更好。`x*0 -> 0` 仍然禁止，以保留 NaN 语义。当前产物位于每个 run 的 `formula_library/curated_export/`，其中 `curated_analysis.json` 记录 `5e-6` 容差和整理统计。

这样以后可以研究：

- 高分公式都在表达什么结构；
- Transformer 和 random 的 top-N 有什么差异；
- 哪些公式适合结合人工交易想法继续改造；
- 是否值得做多公式组合。

信号相似度、聚类和“多样化 top-N”可以以后作为独立分析模块加入，但不应阻塞 raw/curated 公式库输出，也不是当前 pilot 或未来 formal 的默认完成条件。若 formal 研究确实需要多样性比较，必须在 formal protocol 中单独写明。

**本次暴露的问题：**

原实现把相似度和聚类做成了 top-N 输出前的强制步骤：公式重放虽然使用 CUDA，但 signal 随后被搬到 CPU NumPy，并由 Python/NumPy 串行计算全量相似度。这个复杂模块已经人工停止，不再作为当前主路径继续优化。

### 4.6 样本外验证层：防止训练期自我证明

训练期 reward 只能说明模型在训练数据上找到了高分公式，不能说明公式对未来有效。

计划的验证顺序是：

```text
训练期生成并冻结候选
    -> 2022 validation
    -> 人工审核
    -> 2023+ final OOS
    -> 人工审核
    -> 纸面交易
```

验证层才加入完整执行语义：

- T+1；
- 成本和滑点；
- 退出规则；
- 停牌、涨跌停和无法成交；
- 净收益、夏普、最大回撤、换手和稳定性。

2022 与 2023+ 不是训练数据，也不应该因为训练过程中的结果好看就提前打开。

### 4.7 纸面交易和执行层：目前尚未建设

只有样本外验证通过后，才进入：

```text
实时数据收盘后生成信号
    -> 纸面模拟下单
    -> 记录真实延迟和成交假设
    -> 小资金验证执行链路
```

真实执行层还要处理：

- 券商接口；
- T+1；
- 停牌和涨跌停；
- 仓位和风险限制；
- 退出和紧急止损。

这些都不属于当前 Stage D pilot。

## 5. 阶段设计和每个阶段的目的

### 5.1 前置阶段：数据、公式和评分链路

目标不是赚钱，而是先确认：

```text
数据 -> 因子 -> 公式 VM -> signal -> scorer -> artifact
```

能够稳定、可复现、可校验地运行。

这里包括：

- 数据 manifest 和身份；
- 因子定义；
- 公式 token 和语法；
- CPU/GPU VM；
- scorer；
- 固定公式 sanity；
- artifact reload 校验；
- 合成 fixture。

### 5.2 Stage C：GPU 和随机 baseline 工程验证

Stage C 的 seed `41/42/43` 各运行 `100000` 次随机公式。它主要验证：

- CUDA 环境；
- 公式 VM 和 scorer；
- GPU 与 CPU 结果差异门禁；
- ledger 和 checkpoint；
- 随机公式批量吞吐；
- 运行产物和身份校验。

Stage C 已通过。它不是 Transformer 研究结论，也不产生正式的 50 条候选。

Stage C 完成后本来就应该停在人审点，不能自动进入 Transformer 或样本外验证。

### 5.3 Stage D GPU pilot：第一次真实 Transformer 训练和公式库输出

当前 pilot 的冻结参数是：

```text
method       Transformer
seed         314159
attempts     50000
batch        256
device       CUDA only
forced stop  10240 次，用于验证 resume
train split  2016-08-09 .. 2021-12-31
```

它要验证：

1. 真实 Transformer 训练能否稳定运行；
2. reward、loss、梯度和模型参数是否保持有限；
3. checkpoint 是否能精确恢复下一批公式、优化器、候选账本和 RNG；
4. 训练是否出现明显重复、熵坍缩或公式失控；
5. 训练账本和 top-N 公式 artifact 能否保存；
6. GPU 吞吐、显存和远端运行行为是否可接受。

pilot 是真实训练，但不是正式比较实验。它只有一个 seed，没有 matched-random，不允许形成研究结论，也不打开 2022/2023+。

本次实际结果：

- `50000/50000` Transformer 训练完成；
- `10240` 次 checkpoint/resume 成功；
- 训练耗时和 GPU 运行状态正常；
- 原候选漏斗运行超过一个小时仍未完成，已人工停止；
- 训练账本和 checkpoint 保留，raw/curated top-N 已按简单路径导出；
- 因此 pilot 的工程验证和公式库产出均已完成，当前停在人审点，不自动进入 formal 或 validation。

### 5.4 过渡实验：从训练结果到可研究公式库

在个人探索阶段，不需要立刻设计正式的多 seed、matched-random 和复杂候选比较。更合适的过渡步骤是：

```text
从现有训练 checkpoint/ledger 恢复
    -> canonical 去重
    -> 输出 raw top 30 / top 50
    -> 输出 curated top 30 / top 50
    -> 人工阅读、分类和记录想法
    -> 决定 formal 是否需要 matched-random、交易规则或多公式组合
```

这一步的产物是研究材料，不是最终策略，也不需要宣称 Transformer 胜过 random。

formal protocol 和 binding 现在已经写入，启动前仍需再次确认远端部署身份。它固定：

```text
Transformer 和 random 是否都要跑
每种方法跑几个 seed
每个 seed 的 attempts
top-N 如何保存和比较
是否需要多样性分析
```

当前 protocol 采用 Transformer 3 seed、random 3 seed、每 seed 1,000,000 attempts；这只是已写入的
实验预算，不代表远端任务已经启动。

### 5.5 后续验证和交易思路过渡

当公式库中出现了值得结合的交易思路后，再逐步加入验证：

```text
先选定一个明确的交易想法
    -> 用少量公式做 2022 验证
    -> 加入 T+1、成本和退出规则
    -> 观察是否值得继续
    -> 再考虑 2023+ OOS、纸面交易或组合
```

这不是当前 pilot 的自动后续，也不要求一次把完整交易系统搭完。

当前不打开 2022/2023+，不是因为必须搭一套复杂门禁，而是因为现在还没有选定要研究的具体交易思路。

## 6. 当前代码和产物怎么对应

### 6.1 研究核心代码

| 责任 | 入口/模块 |
|---|---|
| 构建基础数据 | `scripts/v3a/build_dataset.py` |
| 构建训练数据隔离视图 | `scripts/v3a/build_train_view.py` |
| 运行时研究身份 | `scripts/v3a/runtime.py` |
| 因子定义 | `src/alpha_etf/research_v3a/factors.py` |
| 公式语言和规范化 | `src/alpha_etf/research_v3a/language.py`、`candidates.py` |
| CPU 公式 VM | `src/alpha_etf/research_v3a/vm.py` |
| CUDA 公式 VM | `src/alpha_etf/research_v3a/torch_vm.py` |
| CPU scorer | `src/alpha_etf/research_v3a/scoring.py` |
| CUDA scorer | `src/alpha_etf/research_v3a/torch_scoring.py` |
| Transformer 训练 | `scripts/v3a/train_gpu.py` |
| CUDA resume probe | `scripts/v3a/gpu_resume_probe.py` |
| 随机 baseline | `scripts/v3a/random_baseline.py` |
| 候选漏斗 | `scripts/v3a/select_candidates.py`、`src/alpha_etf/research_v3a/candidates.py` |
| raw/curated top-N 导出 | `scripts/v3a/export_top_formulas.py`、`scripts/v3a/preview_formula_curation.py` |
| pilot 编排 | `scripts/v3a/stage_d_gpu_pilot.sh` |
| checkpoint 校验 | `src/alpha_etf/research_v3a/checkpointing.py` |
| artifact 校验 | `src/alpha_etf/research_v3a/artifacts.py` |

### 6.2 冻结身份

当前已生成的历史 Stage D pilot binding：

```text
binding_id:       ea96bc56c6bbcba852efbc4b944991efe171c3afbe8519f07818313e16735054
code_commit:      24de39e5d7a86c7a225cb0c2b46e68af5baea734
train_view_id:    v3a-train-view-5d733fcad4d9c340
research_spec_id: 1ea1382f50a35b94efa494dc56ceeddb7e0a4f56464d9add297a1249f1a6205b
```

它只负责解释和复核历史 pilot，把以下内容绑在一起：

```text
代码版本
+ ResearchSpec
+ train-view
+ Stage C 报告
+ Stage D protocol
```

这样远端即使有旧代码、旧数据或旧产物，也不能悄悄混用。

后续 formal 不能复用这个历史 pilot binding。当前 formal 已使用独立 protocol，并冻结了代码、ResearchSpec、train-view、数据身份和新的 binding。curated 导出规则已经写入 formal protocol，也必须随 formal artifact 留痕，但不能反向修改历史 binding。

### 6.3 当前关键产物

```text
data/processed/v3a/dataset/
data/processed/v3a/stage_c_reports/
data/processed/v3a/stage_d/train_view/
data/processed/v3a/stage_d/pilot_binding.json
data/processed/v3a/baseline/runs/
data/processed/v3a/smoke/
data/processed/v3a/training/runs/*/formula_library/curated_export/
data/processed/v3a/stage_d/formal_topn_train_view/
data/processed/v3a/stage_d/formal_topn_binding.json
```

Transformer pilot 的远端训练 checkpoint 曾经生成并保留在远端 run 目录中；本次远端任务停止后，没有把它误标为 formal 结果，也没有把未完成候选漏斗提升为正式 artifact。完整 ledger、raw top-N 和 curated top-N 是当前允许使用的训练期研究材料。

## 7. 监工机制的实际设计

监工只是个人项目里帮助观察远端命令的小工具，不是生产级调度系统。没有远端长任务时，不需要启用它；需要运行时也只保留启动、查看、停止和通知四件事。

### 7.1 本地到远端的链路

```text
手动 launch-pilot.sh
    -> 上传固定 bundle 和输入 archive
    -> 远端部署并做测试
    -> 远端 screen 启动 runner

本地 systemd timer
    -> tick.sh
    -> supervisor.py
    -> SSH 调用远端 probe.py
    -> 写 observations/latest.json
    -> 状态异常或 completed 时写 event
    -> 调用 Pi-Web 本地 Agent session
```

### 7.2 监工应该做什么

- 读取远端 runner、screen、GPU 和进度状态；
- 保存本地 observation；
- 任务正常时保持安静；
- 出现明显异常或完成时通知本地 Agent；
- 远端证据只作为观察结果，不当作指令。

### 7.3 监工不应该做什么

- 自动重启；
- 自动修复代码；
- 自动修改实验参数；
- 自动同步或晋升产物；
- 自动验证研究结论；
- 自动关机；
- 自动决定进入下一阶段。

对应的手动入口是：

```text
.pi/supervisor/launch-pilot.sh
.pi/supervisor/sync-pilot.sh
.pi/supervisor/stop-pilot.sh
```

### 7.4 当前取舍

之前的监工设计一度包含复杂的 action、claim、receipt、恢复和关机状态机，超出了个人探索的需要，已经不再作为设计方向。

本次 timer 曾经有重复触发问题，后来已修复；候选漏斗没有及时暴露长耗时，则说明观察功能还可以继续改善。但这类改善应保持很小：增加阶段名称、最近进度和异常通知即可，不需要再建一套自治控制系统。

## 8. 之前规划与实际偏差

### 8.1 原本合理的部分

- 先冻结研究问题和数据边界；
- Stage C 先做 GPU 和随机 baseline；
- Stage D pilot 再做真实 Transformer 训练；
- 训练结果保存完整账本和公式 artifact；
- 以后确实要比较方法或交易思路时，再加入 random、validation 和 OOS；
- checkpoint 和研究输入保持可复现。

这些原则足以支撑个人探索。复杂的身份绑定、完整候选聚类和自动化收尾不应在当前阶段继续扩大。

### 8.2 实际做得不好的部分

#### 1. 把大设计拆成了太多隐含文件

研究问题、protocol、ResearchSpec、binding、run-spec、runner 和 supervisor 各自保存了一部分信息，但没有先给出一份人工可读的总图。结果是代码越来越完整，整体逻辑反而越来越难看懂。

#### 2. 把“保存 top-N”误做成了“完整候选漏斗”

用户需要的是一批公式供后续探索，我却增加了相似度、聚类和配额，把它们变成了 top-N 输出前的强制步骤。这是把未来可能有用的增强功能提前做成当前硬门槛。

#### 3. 把个人探索项目按生产系统设计

加入了过多 binding、状态、自动化和防御性校验，增加了理解成本。当前只需要保证输入、代码和结果基本对应，足够复现即可。

#### 4. 没有在中间停下来确认方向

代码、远端部署和监工不断叠加，直到真实运行才发现主路径已经偏离了“先产出公式库”的目标。

## 9. 简化后的过渡路线

### 第一步：先从现有训练结果整理公式库

不重新训练，不运行相似度聚类，只做：

```text
读取现有 checkpoint/ledger
    -> canonical 去重
    -> 按 reward 输出 raw top 30 / top 50
    -> 按固定展示清理和 5e-6 近似并列规则输出 curated top 30 / top 50
```

这一步完成后，Stage D pilot 就已经实现了当前个人探索最重要的产出：一批保留原始证据、同时更适合人工阅读的公式。

### 第二步：人工阅读和记录交易想法

对 top-N 公式做简单整理：

- 公式表达了哪些因子逻辑；
- 是否出现明显重复；
- 哪些公式和已有交易想法有关；
- 哪些公式值得单独做简单回测。

这里不需要先做自动聚类。人工阅读本身就是当前项目要保留的研究环节。

### 第三步：按一个具体想法加入过渡验证

当出现明确的交易想法后，再选择少量公式做：

```text
训练期检查
    -> 简单的 2022 验证
    -> 加入 T+1、成本和退出规则
    -> 判断是否值得继续
```

如果以后发现公式数量太多、人工比较困难，再单独增加相似度分析。那时它是解决实际问题的工具，不是预先假定必须存在的模块。

### 第四步：只有研究问题成熟后才扩大

未来可能增加：

- Transformer 与 random 的正式对照；
- 多 seed；
- 信号相似度和多样化公式库；
- 多公式组合；
- 纸面交易；
- 更完整的执行和风控。

这些都是后续选项，不是当前项目必须一次完成的工程。

## 10. 当前只保留的审核问题

每次继续推进前，只问四个问题：

1. 这一步要产出什么？是公式库、研究观察，还是交易验证？
2. 它需要读取哪段数据？有没有不小心打开未来数据？
3. 这一步是否真的解决当前问题，还是提前实现了未来功能？
4. 跑完后是否需要停下来和用户讨论？

不再默认要求完整身份系统、复杂恢复机制、自动晋升或全套防御性门禁。

## 11. 简化后的最终结构

```text
数据和公式语言固定
    -> Transformer / random 生成公式
    -> scorer 评价每条公式
    -> 保存完整账本和 top-N 公式库
    -> 人工结合交易思路继续探索
    -> 需要时再加入验证、相似度、组合或执行
```

监工只在远端长任务确实存在时启用，负责观察和通知，不负责替研究者做决定。

当前最重要的结论是：

> **先把公式发现和公式库产出跑通，不要为了预想中的正式研究和实盘系统，把个人探索项目提前做成复杂工程。**
