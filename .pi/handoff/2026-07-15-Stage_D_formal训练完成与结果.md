# Stage D formal训练完成与结果

时间：2026-07-15

## 完成状态

- 905机上的六个formal run全部完成，每个8,000,000 attempts，`resume_count=0`。
- 六个completion marker、summary、retained candidates、checkpoint和ledger SHA全部闭合。
- Watchman已拉回收尾证据，停用timer并关闭905机；当前SSH拒绝连接。
- 本分析只读取2017-2021训练期产物，没有读取2022 validation或2023+ final。

固定身份：

```text
protocol_id       02cca48d1c8536f90a23a0e0361cb45e64a30e95623afc05ef58bd57a0dca5c1
code_commit       0df9404331b43bc8775d7ff17dab1d3172ecf186
code_fingerprint  f291520bf15d4d7f875e608e381386ac8c982ecc9b6e72a0826b345dc3d4f0ed
research_spec_id  1ab171f45467c6e8273a925a771988583c3ccecd640d6e240248dca436d7db07
train_view_id     v3a-train-view-8e48ba4ce6c2af91
binding_id        f545b622f5d8887419325cc0551483b6635e35998830e4df50c7cdb0d576da12
```

## Formal结论

冻结主指标是配对seed的`curated_top50_mean_reward`。验收要求Transformer配对均差为正，且至少赢2/3。

| Seed | Transformer | Matched random | T-R（基点） |
|---:|---:|---:|---:|
| 101 | 0.4436% | 0.5472% | -10.35 |
| 102 | 0.4664% | 0.5540% | -8.76 |
| 103 | 0.4777% | 0.5510% | -7.33 |
| 均值 | 0.4626% | 0.5507% | -8.81 |

Transformer赢0/3，配对均差为负，**冻结主判据失败；matched random胜出。**

这只回答训练期搜索比较，不代表random公式可以交易，也不是样本外策略结论。

## 根因判断

Transformer并非没有学习：

- 语义有效率99.65%，random为32.00%。
- 所有有效样本的平均reward为0.3996%，random为0.0885%。
- 吞吐9,934 attempts/s，random为5,661/s。

但Transformer的canonical唯一公式平均只有294,930，random为7,432,094；前者只有后者约4%。accepted唯一公式约19.9万对198.2万，相差约10倍。

因此当前REINFORCE目标成功提高了“下一次采样的平均reward”，却造成策略收缩和大量重复；formal指标考的是“最好50个不同公式的质量”。训练目标与研究目标错位，是Transformer输掉top50的主要原因。

## 公式观察

- matched random三个seed的150条curated top50中，150/150都包含`GAP`、`MEAN`、`WIN_20`，核心主题是20日跳空均值。
- 简单锚点`NEG(MEAN(GAP,20))`训练reward约0.5346%，在三个random run排61/78/63。
- Transformer seed 102/103找到该短公式并排第3，seed 101没有，说明策略收缩有seed依赖。
- curated top50的精确公式hash跨seed几乎不重合，但这不代表信号不相似；本轮未做信号相关性聚类。

## 工程观察

- 六run纯训练合计约110.9分钟，没有resume或报错。
- Transformer约比random快75.5%，但CUDA reserved峰值平均约93.9GiB，random约62.3GiB。
- 六run都成功完成；Transformer正式长跑的显存余量比阶段5短门禁观察更小，未来扩大batch或VM不能直接沿用当前余量假设。

## 证据与分析

```text
.pi/profile/results/v3a-formal-runs-20260715/
  formal-multicpu-closeout-20260715T111422Z.tar.gz
  analysis/formal_training_comparison.json
  analysis/formal_training_comparison.md
```

结果解释、第一性根因、AlphaGPT/AlphaGen参考、下一轮候选优化方向和研究边界已集中整理到：

```text
docs/V3A_StageD_formal结果与下一轮优化设计.md
```

后续session应优先读这份详细文档，不要仅凭本handoff中的摘要直接设计新训练。

收尾包SHA-256：

```text
7409c635f7661a25938da123174d7b7a8bb04cd412c1899f5984b8ee2e8703ef
```

## 下一研究门

当前必须先由用户选择两条路径之一：

1. **路径A：先完成当前研究闭环**。冻结两边curated top50与简单锚点，预注册判据后一次性打开2022 validation。代价是2022此后不再是未来新Transformer方案的干净门禁。
2. **路径B：先保留2022封存，做训练期机制诊断与一轮预注册的小预算修复实验**。最终冻结旧/新方案后，再统一打开2022一次。若当前优先目标是把Transformer机制研究明白，更倾向此路径。

在用户选择前，不连接远端、不打开2022、不修改训练代码。两条路径的完整取舍见`docs/V3A_StageD_formal结果与下一轮优化设计.md`第12节。

无论选择哪条路径，2023+ final继续封存。任何新的Transformer训练目标、熵策略、去重奖励或多样性机制都是下一轮研究设计，不能回头修改本次formal结论。