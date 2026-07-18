# V3A Stage D Validation 实验协议

> 冻结日期：2026-07-18
> 协议状态：研究语义已冻结，候选 artifact 与运行 binding 尚未生成，**未授权读取 2022**。
> 数据边界：候选准备只能读取 `2016-08-09..2021-12-31`；正式 validation 只读
> `2022-01-01..2022-12-31`；`2023-01-01..数据末日`继续封存。

## 1. 目的与结论边界

本次 validation 只回答两个问题：

1. 训练期原始高分、训练期三子期都稳定增值的复杂公式、以及这些复杂公式的同源短公式，哪一种
   **候选规则**能在 2022 保留组级样本外增量；
2. 最终选中的 Transformer 候选规则，是否明确优于相同 canonical 搜索规模下的 matched Random。

2022 只选择一套候选规则，不在规则内部挑公式。选中规则在三个 seed 中预先冻结的完整公式集，
才有资格原样进入 2023+ final。本次不回答单条公式是否可实盘，不做多公式集成，不选择交易成本、
市场开关、阈值或风控参数，也不证明动态历史 ETF 池中的可投资业绩。

允许没有赢家。没有赢家、Transformer 未明确胜 Random，或任何正式门禁无法判定时，均停止并继续
封存 2023+。

## 2. 冻结依据

- Gate 1 三个机制 run：protocol `9f9cfbe379a7201fdc6156d3b6f866156d788fc54c6bce0c84dce4d4dbf0a3ca`，
  code commit `469aa8f07467dfed279cb400cb92a7b0456874b7`，Transformer seeds `101/102/103`，
  每个 run 共 2,000,000 attempts，其中 Transformer lane 1,500,000 attempts。
- 候选训练身份：train view `v3a-train-view-0dceecd129115f1b`，Gate 1 binding
  `dc37a173f2fa7daec5aeeb543883ddfca12d9dc0c1981a8209e2e90ca2b9f86c`。
- Random 来源：旧 formal matched-random seeds `101/102/103` 的冻结 ledger；每个 seed 只取达到
  对应 Transformer 最终 canonical 唯一数的最短 ledger 前缀。
- 训练子期沿用已冻结复杂度审计：

| 子期 | 日期 | scorer days |
|---|---|---:|
| S1 | 2016-08-09..2018-05-24 | 424 |
| S2 | 2018-05-25..2020-03-06 | 423 |
| S3 | 2020-03-09..2021-12-16 | 423 |

本协议取代早期“2022 最多筛 5 条公式”的约定，不改变 train / validation / final 的时间隔离。

## 3. 训练期候选构建

### 3.1 共同确定性规则

Transformer 每个 seed 只使用 Gate 1 最终 2m ledger 中的 Transformer lane，不混入该 run 的 25%
Random lane。原始候选沿用冻结的 `etf-v3a-display-simplifier-v1` curated top-50 规则和 `5e-6`
近似并列容差；展示清理只影响近似并列时的先后，不改变 canonical hash，validator 始终执行原始 exact
token 序列。

每条候选及其可能使用的短 subtree 必须先通过以下数值门：

1. CUDA 与 CPU 在完整训练 scorer 日期上选出的 top-k ETF 序列逐日一致；
2. CPU 重算 full-train reward 与冻结 CUDA 重评分的绝对差不超过 `1e-6`；源自 ledger 的完整公式
   还必须同时与 ledger reward 满足该容差；
3. CPU 和 CUDA 两端都通过冻结公式质量门：coverage 至少 95%、有限值标准差大于 `1e-12`、
   scorer 有效且 scorer days 至少 252。

不通过的公式在读取 2022 前剔除，并在同 seed、同方法的训练排名中确定性顺延补位。补位公式必须
完成同样的数值门、S1/S2/S3 重评分和 subtree 审计。所有排序并列依次按训练 reward 降序、token
长度升序、canonical hash 升序打破。

### 3.2 三套候选规则

每个 Transformer seed 独立生成三组，组内按 canonical 去重：

- **原始高分组 `original_top50`**：通过共同数值门后的最终 curated top 50。每 seed 固定 50 条。
  这里的“原始”指未经复杂度筛选，不是 Stage D 公式库中的 raw top-N artifact。
- **稳健复杂组 `stable_complex`**：从该 seed 的 `original_top50` 中保留满足以下条件的公式：先枚举其
  所有非 constant、raw token 长度不超过 10 的 proper subtree；以 full-train reward 最高者作为
  对应简化公式；复杂公式必须在 S1、S2、S3 的 reward 中都严格高于该简化公式。
- **对应简化组 `paired_simple`**：取 `stable_complex` 的对应简化公式，按 seed 做 canonical 去重。
  同一简化公式可以对应多条复杂公式；组级分布只计算一次，但复杂-简化配对检验保留每一条原始配对。

若 subtree 的 full-train reward 并列，依次选择 token 更短、canonical hash 更小者。没有合格 proper
subtree 的复杂公式不能进入 `stable_complex`。各组保留规则自然产生的数量，不裁成等量；不同 seed
之间不合并公式后统一计算，以免候选较多的 seed 获得更大权重。

### 3.3 Random 镜像候选

Random 前缀直接冻结为 Gate 1 同 canonical 分析已经产出的确定结果：

| Seed | Transformer canonical 唯一数 | Random 前缀 attempts |
|---:|---:|---:|
| 101 | 1,418,275 | 1,519,926 |
| 102 | 1,338,626 | 1,434,125 |
| 103 | 1,381,295 | 1,480,379 |

在该前缀上执行与 Transformer 完全相同的 top-50、数值门、补位、子期重评分和 subtree 规则，生成
`random_original_top50`、`random_stable_complex`、`random_paired_simple`。实现必须复核截止行的累计
canonical 唯一数与表中目标相等，但不得自行移动截止点。

Random 不是第四种待选规则，不参与 Transformer 三组的复杂度选择；它只在 Transformer 规则确定后，
用同名镜像组否证“学习搜索优于随机搜索”的主张。

### 3.4 解封前候选验收

候选 artifact 必须记录每条公式的 exact tokens、canonical hash、来源方法和 seed、训练排名、full/S1/S2/S3
reward、CPU/GPU 门结果、补位原因、复杂-简化配对，以及源 ledger SHA-256。另输出每 seed 各组数量、
组内和跨组 canonical 重合、Random 截止 attempt、全部 artifact SHA-256。

任一 seed 的 `stable_complex`、`paired_simple` 或对应 Random 镜像组为空，候选身份无法闭合，或数值门
无法复现时，不得读取 2022；先回到训练期事实和用户决策。

## 4. 单公式交易模拟

### 4.1 信号与排名

每条公式独立管理初始资金 1.0，不做公式间组合。正式 validation 使用 CPU float32 公式信号；横截面
中位数和 MAD 用这些有限值计算。每个决策日只使用当日收盘时已知的数据：

- 排名池为当日 tradable 且公式 signal 有限的 ETF；降序排名，同值按冻结 universe symbol 顺序打破；
- 排名池少于 10 只时，不做排名买入或排名退出，已有持仓只检查止损；
- `robust_z = (signal - median) / (1.4826 * MAD)`；MAD 为 0 或非有限时，当日不新买；
- 新买资格为 `rank <= 3` 且 `robust_z >= 1.5`；公式 signal 的原始正负不参与判断；
- 在排名池不少于 10 只的决策日，持仓排名退出条件为 `rank > 5`；持仓 ETF 不在有效排名池时视为
  `rank = +inf`。

公式只提供相对选择，不增加统一市场风险开关。

### 4.2 初始建仓与因果时序

- 初始现金和净值均为 1.0，不继承训练期持仓或盈亏；
- 使用 2021 最后一个有效收盘决策日的信号，在 2022 第一个可交易开盘执行初始买单；
- 此后均为 t 日收盘形成固定订单，t+1 开盘先卖后买；排名和候选不得查看 t+1 可交易性；
- 买单在次日开盘无法成交时立即作废，不向第 4 名以后补位，不把该预算追加给其他候选；
- 卖单在次日开盘无法成交时保留，直到该 ETF 首个可成交开盘执行；卖出前该仓位继续占槽；
- 2022 最后一个收盘不再生成需要 2023 执行的订单，年末按 2022 最后有效收盘盯市，不读取 2023 价格。

### 4.3 仓位、现金与退出

- 最多 3 个持仓；已有持仓完全不再平衡；允许分数份额；
- 每次开盘先完成可执行卖单，再把当时全部现金视为当前空槽的局部资金池，按空槽数等分；按上日排名
  给每个合格且未持有的候选分配一份；候选不足或买入失败时，相应资金留在现金；
- 主动退出只有排名退出和止损。止损在 t 日收盘按 `close / actual_entry_open - 1 <= -7%`确认，
  t+1 或之后首个可成交开盘卖出；无最长持有期、止盈或移动止盈；
- ETF 因止损卖出后进入禁止重入状态。只有在卖出后的某个收盘决策日先失去一次买入资格，禁止状态
  才解除；之后重新满足买入资格才能下单。排名退出不使用该禁止状态；
- 停牌或价格缺失时用最近一个有限收盘价盯市，不假设成交。

正式主口径不计手续费和滑点，现金收益为 0。

## 5. 基准与逐公式指标

### 5.1 市场基准

主基准为每日可交易 ETF 池等权复合净值，与策略同在 2022 第一个可交易开盘从 1.0 起步。首日对
开盘和收盘都可成交且价格有限的 ETF 计算 open-to-close 收益并等权平均；此后只使用前一收盘和当前
收盘都 tradable 且价格有限的 ETF，计算各自 close-to-close 收益后等权平均并逐日复合。当日无可计算
ETF 时收益记 0。该基准不继承个别 ETF 权重，也不使用某一决策日之后的可交易性选择当日成分。

辅助基准为 2022 第一个可交易开盘时，对当时可成交 ETF 等权买入并持有至年末；中途不补入、不
再平衡。辅助基准只报告，不参与正式门槛。

### 5.2 正式指标

每条公式只用以下两个正式指标：

- **总收益**：`2022年末净值 / 1.0 - 1`；
- **最大回撤**：`max_t(1 - equity_t / running_peak_t)`，按非负损失幅度报告。

每个方法、seed、候选规则先在组内计算总收益中位数、总收益 75% 分位数和最大回撤中位数。分位数
固定使用 NumPy `quantile(method="linear")`；三个 seed 始终等权，不按公式数量加权。单条最高收益、
Sharpe、季度结果和任何复合评分都不参与正式判断。

## 6. 正式判定算法

### 6.1 生存门

一套 Transformer 候选规则在某个 seed 生存，当且仅当同时满足：

1. 组内总收益中位数 `> 0`；
2. 组内总收益中位数 `> 主基准总收益`；
3. 组内最大回撤中位数 `<= 20%`；
4. 组内最大回撤中位数 `<= 主基准最大回撤`。

规则必须至少 2/3 seed 生存。`stable_complex` 还必须至少 2/3 seed 满足“复杂公式总收益减对应
简化公式总收益”的配对中位数 `> 0`；配对按复杂公式逐条计算，共用同一简化公式的多条配对分别保留。

### 6.2 复杂度升级门

规则简单度固定为：`paired_simple > stable_complex > original_top50`。先排除未通过生存门的规则；
`stable_complex` 未通过配对门也一并排除。默认选择仍合格的最简单规则。

更复杂或更少筛选的规则 A 只有同时满足下列条件，才能推翻一个仍合格的更简单规则 B：

1. 每 seed 计算 `A总收益Q75 - B总收益Q75`，三 seed 等权平均至少为 `0.03`；
2. 上述 Q75 差在至少 2/3 seed 严格为正；
3. A 的三个 seed“组内总收益中位数”的等权平均不低于 B；
4. A 的三个 seed“组内最大回撤中位数”的等权平均不高于 B。

若有多个更简单规则仍合格，A 必须逐一通过上述升级门才可成为赢家。没有形成明确升级优势时保持
简单规则。因此接近时的确定性顺序是 `paired_simple > stable_complex > original_top50`；若只有一个规则
合格，则直接成为暂定赢家。

### 6.3 Random 否证门

暂定 Transformer 赢家只与同名 Random 镜像规则比较。每 seed 计算 Transformer 减 Random 的组内
总收益 Q75；只有同时满足以下条件，Transformer 才成为正式赢家：

1. Q75 差三 seed 等权平均至少为 `0.03`；
2. Q75 差在至少 2/3 seed 严格为正；
3. Transformer 的三个 seed“组内总收益中位数”等权平均不低于 Random；
4. Transformer 的三个 seed“组内最大回撤中位数”等权平均不高于 Random。

Random 自身不参加三组选择，也不要求先通过生存门。Transformer 未通过否证门时，本次结论为
“未证明学习搜索有样本外增量”，没有公式进入 final。

### 6.4 唯一正式结果

正式结果只能是以下三类之一：

- `winner`：唯一规则通过全部门，冻结其三个 seed 的完整公式集进入 final 准备；
- `no_surviving_rule`：没有 Transformer 规则通过生存门和适用的配对门；
- `transformer_not_better_than_random`：有暂定规则，但未通过 Random 否证门。

不得按 2022 单条公式表现删减、补位、加权或组成公式集，不得事后改变阈值、基准、分位数、seed
权重或赢家顺序。

## 7. 运行身份、解封门与输出

正式 validator 实现完成后，先只用合成 fixture 或训练期切片验证成交路径，并在训练期检查
`robust_z >= 1.5`是否出现长期全满仓、全空仓、MAD 退化或持仓数异常。该 sanity 只判断规则是否
机械退化，不比较训练收益，不搜索新阈值。若退化，停下讨论，不读取 2022。

正式运行前必须生成独立 validation binding，至少固定：

- 本协议所在 commit 与机器可读 protocol SHA-256；
- 当前机器可读协议为`configs/v3a_stage_d_validation.json`，protocol ID
  `30d07d673ab9a7b787f0d44f58249ab48666bfa21cab6a7911a82eb4b62aa675`，其中
  `validation_run_approved=false`；
- validator code commit 和 code fingerprint；
- dataset / panel / validation-view 身份与日期边界；
- Transformer、Random 源 ledger SHA-256；
- 候选 artifact、配对 artifact 和候选清单 SHA-256；
- 公式 VM、因子、排名、分位数与交易配置版本；
- `final_metrics_read=false` 和显式 `validation_run_approved=true`。

在候选数量、重合、配对、数值门、占用率 sanity、smoke 和 binding 向用户展示并获得**新的明确授权**
前，入口必须 fail closed，禁止加载或检查任何 2022 signal、收益、净值或指标。

获批后一次性运行全部 Transformer 与 Random 候选，先生成不可变的机器正式结果，再生成报告和公式
明细。正式输出至少包括逐公式 summary、逐 seed 组级统计、全部门槛的输入值和布尔结果、唯一正式
结论、交易与每日净值审计文件、artifact SHA-256，以及是否读取 validation / final 的边界声明。

2022 明细只能在机器正式结论冻结后作探索性分析，且不得改变正式赢家。季度分析不预设、不作为
固定输出或门槛。若发生纯工程故障，只能在候选、协议和判定规则完全不变的前提下修复并重跑；故障
产物和原因必须保留，不能利用已见 2022 结果修改研究设计。

## 8. 明确不做

- 不读取 2023+，不提前设计或运行 final 结论；
- 不按 2022 选择 top 5、最佳单条公式或 seed；
- 不做多公式投票、加权、组合优化或全组合再平衡；
- 不加入手续费、滑点、市场择时、止盈、最长持有期或新风控；
- 不用训练收益调 `robust_z`，不用 validation 调训练机制；
- 不把固定当代 35 只 ETF 池的结果解释为无事后选池偏差的历史实盘表现。

## 9. 下一停点

本协议完成后仍不读取 2022。下一阶段只做训练期候选 artifact、19 条新增 top-50 成员及所需补位的
复杂度/数值审计、训练期占用率 sanity、validator 合成 smoke 和 validation binding。上述结果经人工
复核并再次授权后，才允许一次性解封 2022。