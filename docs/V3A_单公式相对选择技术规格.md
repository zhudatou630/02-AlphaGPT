# V3A 单公式 ETF 相对选择技术规格

## 1. 目标与边界

V3A 只验证一个问题：在固定的 35 只当代 ETF 池中，一个由 Transformer 搜索得到的单公式，
能否在必须投资时提供相对选择能力。

本轮不实现市场状态、空仓、多公式池、动态历史 ETF 池或完整实盘策略。固定当代池存在事后
选池偏差，最终结果不能解释为严格历史可投资业绩。

## 2. 版本身份

以下版本字符串是 ResearchSpec 的组成部分，修改语义时必须升级版本：

| 组件 | 版本 |
|---|---|
| dataset schema | `etf-v3a-dataset-v1` |
| ResearchSpec schema | `etf-v3a-research-spec-v1` |
| factor spec | `etf-v3a-factors-v1` |
| formula vocab | `etf-v3a-formula-v1` |
| formula grammar | `etf-v3a-grammar-v1` |
| scorer | `etf-v3a-scorer-v1` |
| canonicalizer | `etf-v3a-canonical-v1` |
| candidate funnel | `etf-v3a-candidate-funnel-v1` |
| artifact | `etf-v3a-formula-artifact-v1` |
| checkpoint | `etf-v3a-checkpoint-v1` |
| training funnel artifact | `etf-v3a-training-funnel-v1` |
| checkpoint candidate state | `etf-v3a-candidate-state-v1` |

## 3. 数据契约

### 3.1 数据分工

- 事件复权绝对 OHLC 只供基础因子内部计算和 scorer 成交。
- 相对 OHLC 只供数据治理和交叉核验。
- 公式词表不得包含裸 OHLC、volume 或 amount。
- `tradable_mask` 是 t 日当时可知状态；上市前、缺行、零成交和停牌均为 false。

V3A dataset 保存：

```text
absolute_ohlc   [asset, open/high/low/close, date]
relative_ohlc   [asset, open_rel/high_rel/low_rel/close_rel, date]
tradable_mask   [asset, date]
symbols
dates
```

研究输入在 `tradable_mask=False` 处为 NaN。治理长表可保留不可交易日的原始价格，但 VM 不得
读取这些值。

### 3.2 相对价格核验

相对 OHLC 必须在治理长表应用研究 mask 前计算：

```text
open_rel  = O_t / previous_source_row_close - 1
high_rel  = H_t / previous_source_row_close - 1
low_rel   = L_t / previous_source_row_close - 1
close_rel = C_t / previous_source_row_close - 1
```

`previous_source_row_close` 是同一 ETF 的上一条治理源数据行，不是 masked panel 上前一全市场
交易日。当前行计算完成后，才按当前行 `tradable` 写入 panel。禁止从 masked absolute panel
反推 V3 relative panel。

### 3.3 数据身份

manifest 必须记录并校验：panel SHA-256、上游 V3 dataset ID/SHA、治理 parquet SHA、universe
JSON SHA、symbols、特征、日期范围、shape、tradable 数量、mask 和复权语义。除
`dataset_id/dataset_fingerprint` 外的规范 JSON 生成 fingerprint：

```text
dataset_id = etf-v3a-<fingerprint前12位>
```

## 4. 因子定义

设事件复权绝对价格为 `O_t/H_t/L_t/C_t`。价格分母必须有限且大于 0，否则输出 NaN；不使用
固定 epsilon。区间宽度为 0 时，CLV/RSV 取 0。

### 4.1 固定因子

```text
DAYRET   = C_t / C_(t-1) - 1
GAP      = O_t / C_(t-1) - 1
INTRADAY = C_t / O_t - 1
RANGE    = (H_t - L_t) / C_(t-1)
CLV      = (2*C_t - H_t - L_t) / (H_t - L_t)
```

### 4.2 单窗口因子

`N in {5,10,20,40,60}`：

```text
ROC(N)      = C_t / C_(t-N) - 1
PRICE_MA(N) = C_t / Mean(C,N) - 1
VOL(N)      = Std(DAYRET,N,ddof=0)
TS_RANK(N)  = (average_rank(C_t)-1)/(N-1)
RSV(N)      = (C_t-Min(L,N))/(Max(H,N)-Min(L,N))
```

TS_RANK 的并列值用平均秩；窗口全相等时取 0.5。

### 4.3 双窗口因子

```text
MA_RATIO(short,long) = Mean(C,short) / Mean(C,long) - 1
short,long in {5,10,20,40,60}, short < long
```

共 `5 + 5*5 + C(5,2) = 40` 个基础实例。

### 4.4 窗口和缺失

- 所有位移按全市场统一交易日历，不按单只 ETF 有效交易日压缩。
- 不前值填充，也不跳过 NaN 拼成满窗口。
- REF 要求目标位移值有效；滚动算子要求连续 N 个交易日全部有效。
- 研究期前历史可用于 warm-up；任何算子不得读取 t 之后数据。

## 5. 公式语言

### 5.1 固定 token ID

```text
 0 DAYRET       1 GAP          2 INTRADAY    3 RANGE       4 CLV
 5 ROC          6 PRICE_MA     7 VOL         8 TS_RANK     9 RSV
10 MA_RATIO
11 WIN_1       12 WIN_5       13 WIN_10     14 WIN_20     15 WIN_40     16 WIN_60
17 ADD         18 SUB         19 MUL        20 NEG        21 ABS        22 SIGN
23 REF         24 MEAN
25 CONST_0     26 CONST_1
```

policy token 为 `PAD=0/BOS=1/EOS=2`，公式 token offset 为 3，总词表 30。

### 5.2 语法

基础因子采用因子族后跟窗口参数；组合表达式使用后缀 RPN：

```text
ROC(20)                 -> ROC WIN_20
MA_RATIO(5,20)          -> MA_RATIO WIN_5 WIN_20
ADD(DAYRET,ROC(20))     -> DAYRET ROC WIN_20 ADD
REF(PRICE_MA(20),5)     -> PRICE_MA WIN_20 WIN_5 REF
MEAN(DAYRET,10)         -> DAYRET WIN_10 MEAN
```

- 窗口基础因子只允许 5/10/20/40/60。
- MA_RATIO 的第二个窗口必须大于第一个。
- REF 允许 1/5/10/20/40/60；MEAN 只允许 5/10/20/40/60。
- EOS 只在无待填参数且表达式栈恰有一个值时开放。
- 最大 15 个公式 token，不含 BOS/EOS/PAD；合法单 token 公式可结束。
- action mask 必须保证类型合法且剩余 token 足够闭合。
- Transformer 与随机 baseline 调用同一合法 policy action 集合（含 EOS）；随机程序在该集合上
  均匀抽样，不使用第二套停止概率。

## 6. Scorer

### 6.1 分段

```text
train      2016-08-09 .. 2021-12-31
validation 2022-01-01 .. 2022-12-31
final OOS  2023-01-01 .. 数据末日
```

每段先生成所有公式共用的 `D_split`：t 日可选 ETF 至少 10 只、t+1 在段内，且 A_t 中每只
可买 ETF 的实际顺延卖出日在段内。标签不完整的日期对所有公式统一删除，不允许公式改变日期分母。

### 6.2 选择和收益

对 `t in D_split`：

1. `A_t=tradable_mask[:,t]`，排名时不得查看未来 mask。
2. `k=clip(ceil(0.2*|A_t|),2,4)`。
3. CPU/GPU 排名统一使用 float32 signal；同值按 universe symbol 顺序稳定破同分。
4. 任一共同日期有限 signal 少于 k，整条公式无效，不删除该日、不缩小 k。
5. t 日收盘排名，t+1 开盘买入；无法买入的槽位收益为 0，不补买。
6. 计划在 `t+1+10` 开盘卖出；不能成交则顺延至第一个可交易开盘。
7. A_t 中每只 ETF 以同一规则生成池基准收益。
8. 日 reward 为 top-k 平均收益减 A_t 平均收益；总 reward 是 D_split 等权平均。

绝对收益、最大回撤和 Rank IC 只作审计，不进入 reward。

## 7. 质量与候选漏斗

### 7.1 公式质量

- 训练期去掉最长 warm-up 后有限覆盖率至少 95%。
- 全局有限值标准差大于 `1e-12`。
- D_train 至少 252 日。
- 所有尝试在合法性、质量和去重之前计数。

### 7.2 规范化

- ADD/MUL 的直接子节点按 hash 排序，不跨层重排。
- 折叠纯常数子树；化简 `x+0/x-0/x*1`。
- 化简 `NEG(NEG(x))`、`ABS(ABS(x))`、`SIGN(SIGN(x))`。
- 不化简 `x*0`，以保留 NaN 语义。

### 7.3 相似度和聚类

逐训练日计算横截面 Spearman，将 rho 截断到 `[-1+1e-7,1-1e-7]` 后按 `n-3` 对 Fisher-z
加权。至少 252 个重叠日；不足者不得进入候选漏斗。

- `rho>=0.995` 为信号重复。
- `rho>=0.90` 为同一多样性簇。
- 不对 rho 取绝对值。

在读取 2022 前，按训练 reward 降序、token 长度升序、hash 升序执行固定代表贪心聚类并落盘
cluster_id。2022 选择器不得重新聚类。

### 7.4 候选数量

训练期流式候选桶：1..5、6..10、11..15 token，各保留 top 500 canonical unique。

- 第一轮每簇最多一条，短/中/长配额 10/10/5。
- 第二轮要求与已选项 rho<0.995，补到 20/20/10。
- 桶不足可转移空缺，但重复阈值不变。
- 训练期最终仅 50 条进入 2022；2022 最多选 5 条。
- 5 条和全部身份冻结后，才允许运行 2023+ 最终审计。
- canonical 账本按 hash 保留首次 attempt index、总出现次数、最高 reward 记录及其 attempt index。
- 训练漏斗 artifact 必须保存每个 cluster 的固定代表 hash、完整成员 hash、最终 50 条和排序规则；
  2022 选择器只读取并校验该 artifact。

### 7.5 工程 baseline 与正式漏斗边界

Stage C 的 `100,000 x 3 seed` 是预算前工程 baseline，不生成正式 50 条候选，也不执行
`rho>=0.90` 聚类和长度配额。每组必须保存全部 attempt 账本，并报告两种不同口径：

- 全部语义有效 attempt 中，训练期每日实际 top-k 选中索引张量完全相同的“选择序列重复率”；
- 每桶 reward heap 合并后的 raw-reward 前 50 条中，按第 7.3 节完整 Spearman/Fisher-z 定义计算的
  `rho>=0.995` 信号重复率，分母固定为该组实际可得的前 50 条数量。

这两个指标不得混称。Stage D 的 Transformer 与 matched random 才使用完全相同的完整候选漏斗，
并各自生成训练期 50 条候选及 training-funnel artifact。

## 8. 数值门禁

- NumPy float64 是 CPU 参考真值，GPU 缓存使用 float32。
- 因子/公式有限位置和 NaN 位置一致；`rtol=2e-5, atol=2e-6`。
- CPU/GPU 每日选中 symbol 一致；reward 绝对差不超过 `1e-6`。
- 每只 ETF 的整段绝对 OHLC 乘不同正常数后，40 个因子、公式信号和最终排名不变。
- V1/V2 dataset、vocab、artifact 和 checkpoint 均须被 V3A 硬拒绝。
- Torch VM 按最大实际栈深和显存预算自动分块；输出超过门禁时必须先切研究日期或减小公式 batch，
  不得尝试分配 `batch × max_token × asset × full_date` 的完整栈。
- resume 必须在加载权重前严格核对 run ID、model/scorer/train 配置、候选状态 schema 和完整
  ResearchSpec，并恢复 model、optimizer、候选账本、全局 RNG 与独立随机生成器状态。

## 9. 依据

- Qlib Alpha158/Alpha360：`microsoft/qlib@d5379c520f66a39953bad76234a7019a72796fd0`。
- AlphaGen：`ICT-FinD-Lab/alphagen@259687e8f316994426416c530a94842a2fe6405e`。
- V3A 的 ROC/PRICE_MA 使用正向定义，滚动窗口要求完整，不声称原样复刻 Qlib。
- AlphaGen 只作为独立窗口 token、后缀 builder、动作屏蔽和提前结束的实现参考。