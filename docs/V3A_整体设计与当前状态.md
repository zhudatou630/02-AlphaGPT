# V3A 整体设计与当前状态

更新时间：2026-07-12

> 这是一份面向项目协作和人工审核的总览文档。它解释研究问题、系统结构、阶段门禁、代码分工和当前实际状态。
>
> 本文不替代冻结的研究契约。涉及具体数据定义、scorer 数学口径、公式语法、候选阈值和身份校验时，以以下文件和已生成 artifact 为准：
>
> - `docs/V3A_单公式相对选择技术规格.md`
> - `docs/V3A_StageD_GPU实验协议.md`
> - `configs/v3a_stage_d_gpu_pilot.json`
> - `data/processed/v3a/stage_d/pilot_binding.json`

## 1. 先看结论

V3A 不是一个一次性“训练模型然后看收益”的大工程，而是一条逐层加真实度的研究流水线：

```text
固定数据和研究问题
    -> 定义公式语言和评分标准
    -> 用随机搜索/Transformer 搜索公式
    -> 在训练期去重并筛选候选
    -> 打开 2022 样本外验证
    -> 打开 2023+ 最终样本外审计
    -> 通过后才考虑纸面交易和实盘
```

当前实际状态：

| 部分 | 状态 | 说明 |
|---|---|---|
| V3A 数据、因子、公式 VM、scorer | 已实现并通过测试 | 研究输入和计算契约已冻结 |
| Stage C GPU 工程验证 | 已完成 | CUDA smoke 及随机 baseline 已完成 |
| Stage D Transformer pilot 训练 | 训练部分完成 | `50000/50000`，checkpoint/resume 已验证 |
| Stage D 候选漏斗 | 未完成 | 相似度计算长时间占用单核 CPU，已人工停止 |
| Stage D pilot 总体 | 未通过 | 候选漏斗未完成，不能生成 50 条完整候选 |
| Formal Transformer + matched-random | 未开始 | formal budget 仍未批准 |
| 2022 validation | 未开始 | 数据仍封存 |
| 2023+ final OOS | 未开始 | 数据仍封存 |
| 纸面交易、执行、实盘 | 未开始 | 不属于当前 V3A pilot |

这意味着：当前没有任何正式策略结论，也没有进入下一阶段的依据。

## 2. V3A 要回答的研究问题

V3A 只研究一个窄问题：

> 在固定的 35 只 ETF 中，一个公式能否持续把相对更强的 ETF 排在前面？

公式不是直接预测某只 ETF 的价格，而是对每只 ETF 产生一个分数。每天根据分数排序，选择排名靠前的 ETF。当前版本始终持仓，暂不研究：

- 空仓择时；
- 动态 ETF 池；
- 多公式组合；
- 市场状态切换；
- 完整实盘策略。

当前 V3A 的“策略对象”是：

```text
一个公式
    -> 对 35 只 ETF 产生信号
    -> 每天横截面排名
    -> 选择 top 分位
    -> 按固定预测视界评价相对收益
```

这里的“单公式”不是说只给一只 ETF 用，而是说同一条公式同时作用于整个 ETF 池，再进行横截面选择。

## 3. 为什么要分层设计

如果让模型直接生成公式、直接看未来收益、直接做完整回测并最后挑一个最高分结果，最容易出现三类问题：

1. 模型、数据和交易规则互相缠在一起，无法知道结果到底由什么造成；
2. 训练过程反复看同一段历史，最高分公式很容易过拟合；
3. 一旦把执行细节、未来数据或人工选择混入训练，后面的收益数字无法解释。

因此 V3A 把系统拆成以下层次，每层只回答一个问题。

```text
数据层
  我们当时究竟知道什么？哪些 ETF 当时可以交易？

公式语言层
  模型允许表达哪些因子和运算？

评分层
  什么叫一条公式在训练期表现好？

搜索层
  Transformer 或随机方法能否找到高评分公式？

候选层
  如何避免最后留下 50 条几乎相同的公式？

验证层
  训练期有效的公式，到了未见过的数据是否仍有效？

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

### 4.5 候选漏斗层：从很多公式变成少量代表

训练过程中会生成大量公式。即使公式写法不同，它们也可能：

- 规范化后完全相同；
- 对 ETF 的信号排名完全相同；
- 信号相关性极高；
- 只是同一类公式的微小变体。

所以正式比较不能简单取 reward 最高的 50 条。完整漏斗原本计划为：

```text
保留训练账本
    -> canonical 规范化去重
    -> 重新执行候选公式得到信号
    -> 检查覆盖率和重叠天数
    -> 计算训练期横截面信号相似度
    -> 去掉 rho >= 0.995 的重复信号
    -> 按 rho >= 0.90 聚类
    -> 按公式长度和多样性配额选出 50 条
```

这个步骤的研究目的，是让最终候选集合代表不同的信号方向，而不是同一信号的 50 个复制品。

**本次暴露的问题：**

当前实现中，公式重放使用了 CUDA，但重放后的 signal 被搬到 CPU NumPy；相似度和聚类部分使用 Python 外层循环及 NumPy 计算。对约 1110 条候选，它实际变成了大量串行 CPU 相似度计算。远端进程显示约 100% CPU，线程检查显示主要只有一个线程在工作。

因此候选漏斗的研究逻辑暂时保留，但当前工程实现不通过，不能继续用于正式实验。

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

### 5.3 Stage D GPU pilot：第一次真实 Transformer 训练

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
5. 候选漏斗在真实候选数量下能否完成；
6. GPU 吞吐、显存和远端运行行为是否可接受。

pilot 是真实训练，但不是正式比较实验。它只有一个 seed，没有 matched-random，不允许形成研究结论，也不打开 2022/2023+。

本次实际结果：

- `50000/50000` Transformer 训练完成；
- `10240` 次 checkpoint/resume 成功；
- 训练耗时和 GPU 运行状态正常；
- 候选漏斗运行超过一个小时仍未完成；
- 远端 pilot 已人工停止；
- 因候选漏斗未完成，Stage D pilot 整体未通过。

### 5.4 Formal Stage D：正式对照实验

只有 pilot 通过并经过人工审核后，才重新冻结 formal protocol。正式实验至少需要：

```text
Transformer 多个 seed
matched-random 多个 seed
两边相同的 attempt 预算
两边相同的合法 action 集合
两边相同的 scorer
两边相同的完整候选漏斗
两边以 50 条候选集合做比较
```

不能用“单条最高分”判断 Transformer 胜过随机。正式比较应看候选集合整体质量、稳定性、多样性和跨 seed 表现。

目前 `formal_budget_approved=false`，matched-random seed 为空，因此正式 Stage D 没有开始。

此前出现过的“Transformer 3 seed、random 3 seed、每 seed 1,000,000 attempts”只是候选预算建议，不是批准的正式协议。

### 5.5 2022 validation 和 2023+ final OOS

正式 Stage D 完成后，先冻结训练期候选和身份，再打开 2022。2022 结果经过人工审核后，才允许打开 2023+ final OOS。

这两个阶段之间也必须停下来，不能把 validation 结果直接用于修改候选后再声称是原始 OOS 结果。

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
| pilot 编排 | `scripts/v3a/stage_d_gpu_pilot.sh` |
| checkpoint 校验 | `src/alpha_etf/research_v3a/checkpointing.py` |
| artifact 校验 | `src/alpha_etf/research_v3a/artifacts.py` |

### 6.2 冻结身份

当前已生成的 Stage D pilot binding：

```text
binding_id:       ea96bc56c6bbcba852efbc4b944991efe171c3afbe8519f07818313e16735054
code_commit:      24de39e5d7a86c7a225cb0c2b46e68af5baea734
train_view_id:    v3a-train-view-5d733fcad4d9c340
research_spec_id: 1ea1382f50a35b94efa494dc56ceeddb7e0a4f56464d9add297a1249f1a6205b
```

它的作用是把以下内容绑在一起：

```text
代码版本
+ ResearchSpec
+ train-view
+ Stage C 报告
+ Stage D protocol
```

这样远端即使有旧代码、旧数据或旧产物，也不能悄悄混用。

### 6.3 当前关键产物

```text
data/processed/v3a/dataset/
data/processed/v3a/stage_c_reports/
data/processed/v3a/stage_d/train_view/
data/processed/v3a/stage_d/pilot_binding.json
data/processed/v3a/baseline/runs/
data/processed/v3a/smoke/
```

Transformer pilot 的远端训练 checkpoint 曾经生成并保留在远端 run 目录中；本次远端任务停止后，没有把它误标为完整 Stage D 结果，也没有把未完成候选漏斗提升为正式 artifact。

## 7. 监工机制的实际设计

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
- 对同一个异常或完成状态只生成一个事件；
- 通知本地 Agent；
- 远端证据只作为不可信的观察结果，不当作指令。

### 7.3 监工不应该做什么

- 自动重启；
- 自动修复代码；
- 自动修改实验参数；
- 自动同步或晋升产物；
- 自动验证正式结果；
- 自动关机；
- 自动决定进入下一阶段。

对应的手动入口是：

```text
.pi/supervisor/launch-pilot.sh
.pi/supervisor/sync-pilot.sh
.pi/supervisor/stop-pilot.sh
```

### 7.4 本次监工暴露的问题

之前的监工设计一度包含复杂的 action、claim、receipt、恢复和关机状态机，超出了“观察并通知”的需要，后来已经删减为只读观察器。

简化后又暴露出两个问题：

1. timer 的重复触发配置错误，启动后没有继续执行 tick；
2. 监工只把 `running` 视为静默状态，没有对阶段长时间无进度做简单提醒。

timer 配置后来已修复，但这说明监工也应该先有一个很小的、可测试的状态契约，而不是边运行边补。

## 8. 之前规划与实际偏差

### 8.1 原本合理的部分

- 先冻结研究问题和数据边界；
- Stage C 先做 GPU 和随机 baseline；
- Stage D pilot 再做真实 Transformer 训练；
- formal 实验要有 matched-random；
- validation 和 final OOS 必须封存并分阶段打开；
- checkpoint、binding 和 artifact 需要严格校验。

这些原则的目的是避免前视、身份漂移、单条最高分误导和中途改参数。

### 8.2 实际做得不好的部分

#### 1. 把大设计拆成了太多隐含文件

研究问题、protocol、ResearchSpec、binding、run-spec、runner 和 supervisor 各自保存了一部分信息，但没有先给出一份人工可读的总图。结果是代码越来越完整，整体逻辑反而越来越难看懂。

#### 2. 在性能基准前就跑了完整候选漏斗

候选漏斗中最重的相似度计算没有先用小规模数据测 10、100、1000 条候选的耗时和 CPU/GPU 利用率，就直接放到远端真实 run 中。这个顺序是错误的。

#### 3. 把训练完成误认为整个 pilot 接近完成

Transformer 训练只是 Stage D 的一个子阶段。候选漏斗、artifact、完整验收也是 pilot 的必要部分。以后必须按阶段分别报告，不用一个总的 `running` 掩盖内部阶段。

#### 4. 监工设计先复杂化，后简化

先做了过多自动决策，再退回只读观察；这消耗了实现和审核精力，也让职责边界不清晰。

#### 5. 没有在每个远端步骤前停下来人工审核

正确的停点应当是：

```text
代码/测试完成 -> 人工审核
GPU smoke 完成 -> 人工审核
pilot 训练完成 -> 人工审核
候选漏斗完成 -> 人工审核
formal protocol -> 明确批准
2022 validation -> 人工审核
2023+ OOS -> 人工审核
```

此前没有把这些停点作为实际操作流程固定下来。

## 9. 当前应采用的恢复方式

现在不应该继续在旧的远端运行上打补丁。建议按以下顺序恢复清晰度。

### 第一步：冻结当前现场

- 不重新启动 Stage D；
- 不启动 formal Transformer；
- 不启动 matched-random；
- 不打开 2022 或 2023+；
- 保留已有代码、checkpoint、ledger 和报告；
- 将本次候选漏斗标记为未完成工程试验，而不是正式结果。

### 第二步：重做候选漏斗的性能设计

先在本机合成 fixture 或 GPU 服务器上做小规模基准：

```text
10 条候选
100 条候选
1000 条候选
```

至少记录：

- 端到端耗时；
- GPU 利用率；
- CPU 每核利用率；
- 内存和显存；
- 中间进度；
- 中断后能否恢复；
- 与 CPU 参考实现的数值一致性。

实现方向优先考虑：

1. 将排名和相似度计算批量化到 CUDA；或
2. 明确拆成多核任务，并设计确定性的合并顺序；或
3. 如果完整漏斗本身过重，先重新审查算法复杂度，而不是只加线程数。

不能只设置 `OMP_NUM_THREADS` 就认为已经完成多核优化，因为当前瓶颈在 Python 外层串行流程。

### 第三步：重新定义阶段状态

runner 和 probe 至少要区分：

```text
preflight
smoke
resume_probe
training
candidate_replay
candidate_similarity
candidate_write
completed
failed
stopped
```

每个阶段必须有：

- 当前进度；
- 最近进度时间；
- 可恢复产物；
- 明确的成功标记；
- 明确的失败标记。

监工只报告这些状态，不替 Agent 决定处理方式。

### 第四步：重新审核 formal protocol

pilot 通过后，才重新讨论：

- Transformer seed 数；
- matched-random seed 数；
- 每个 seed 的 attempts；
- 候选集合的比较指标；
- formal 运行的停止和同步规则。

在 formal protocol 明确标记批准之前，formal 入口必须保持 fail closed。

## 10. 人工审核清单

后续任何远端启动前，至少要逐项回答：

### 研究问题

- 这一步具体在回答哪个问题？
- 它产生的是工程证据、训练期候选，还是正式研究结论？

### 数据边界

- 这一步能读取哪些日期？
- 2022 和 2023+ 是否仍然封存？
- 是否使用了与 binding 相同的代码、数据和 ResearchSpec？

### 计算可行性

- 这个步骤的复杂度是什么？
- 小规模基准跑过没有？
- 用的是 GPU、多核 CPU，还是串行 CPU？
- 预计多久？有没有阶段进度和恢复点？

### 结果边界

- 成功后生成什么 artifact？
- 哪些指标可以解释，哪些不能解释？
- 是否必须停下来人工审核？

### 监工边界

- 监工只观察什么？
- 它会不会自动改变实验？
- 异常时是通知本地 Agent，还是直接执行动作？

## 11. 最终结构的一句话版本

```text
先把“我们研究什么”固定下来，
再把“公式能表达什么”和“什么叫好公式”固定下来，
然后用随机搜索和 Transformer 找公式，
在训练期去重并保留多样候选，
经过人工审核后逐层打开未见过的数据，
最后才加入真实交易摩擦和执行。

监工只负责告诉本地 Agent 远端发生了什么，
不负责替研究者做决定。
```

当前最重要的结论不是“继续把 Stage D 跑完”，而是：

> **先恢复设计、状态和验收的可见性，再恢复计算。**
