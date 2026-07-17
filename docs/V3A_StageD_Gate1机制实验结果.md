# V3A Stage D Gate 1 机制实验结果

> 时间：2026-07-16
> 数据边界：只使用`2016-08-09`至`2021-12-31`训练期；2022 validation和2023+ final未读取。
> 结论层级：新学习反馈通过训练期机制门，不构成样本外或可交易结论。

## 1. 一页结论

Gate 1把旧的“按出现次数优化平均reward”改成“奖励高质量新canonical、奖励Transformer自身top50改善、惩罚重复/无效attempt”，并保留25%不参与梯度的合法Random探索。

结果明确：

- 50-batch远端工程门全部通过，最终连续吞吐约5,340 attempts/s；
- 三个新机制run各完成2,000,000 attempts，无resume、无OOM；
- Transformer自主lane各1,500,000 attempts，发现约133.9万至141.8万个canonical；
- 末50万的新canonical产出不是衰减到零，而是达到首50万的103%至108%；
- 新公式top10%平均reward从首段到末段在3/3 seed提高；
- Transformer自身top50后半程在2/3 seed提高，三seed平均为正；
- 相同canonical唯一数下，Transformer在3/3 seed超过Random，平均领先12.03bp；
- 四项预注册机制门联合通过。

因此训练期最可信的判断是：

> 目标错位和重复反馈确实是旧Transformer失败的主要机制原因；前置历史canonical反馈、右尾学习和archive增量奖励能够让模型保持探索，并把新公式的高质量右尾继续向上推。

但新top50全部为15-token复杂公式，核心围绕`GAP / MEAN / WIN_20`叠加更多结构。训练期高分可能包含明显的多重检验和复杂度过拟合；现有结果不能证明这些公式在2022继续有效。

## 2. 冻结身份

```text
protocol_id       9f9cfbe379a7201fdc6156d3b6f866156d788fc54c6bce0c84dce4d4dbf0a3ca
code_commit       469aa8f07467dfed279cb400cb92a7b0456874b7
code_fingerprint  bc15ca3c73b08e43bc29398c108bde79566ca770bbb70fa4460ea03e2fb31e05
research_spec_id  536cb701bb25e9cf022e84c203e9eab5ec731d446e4a5d6f93d0096c5bf9a2ec
train_view_id     v3a-train-view-0dceecd129115f1b
binding_id        dc37a173f2fa7daec5aeeb543883ddfca12d9dc0c1981a8209e2e90ca2b9f86c
```

`469aa8f`包含对`rolling_mean_torch`的分行块执行优化；它仍使用原`unfold + finite + mean`计算，每行浮点结果与整块实现逐值一致，只限制瞬时工作区。

## 3. 工程门

第一版commit `91b809d`在25-batch停止后resume到第42批时，`MEAN`密集公式使旧rolling window一次申请30.28GiB并OOM。根因不是跨batch泄漏或teacher-forcing，而是同一MEAN指令行数增加后，window临时张量没有分行上限。

`469aa8f`将同一MEAN操作最多512行一块执行。最终工程证据：

| 指标 | 结果 | 门槛 |
|---|---:|---:|
| 连续50-batch吞吐 | 5,339.96 attempts/s | >=3,000 |
| canonical未隐藏等待占比均值 | 0.88% | <=25% |
| teacher-forcing占比均值 | 1.00% | <=20% |
| 外部GPU峰值显存 | 50,461 MiB | 至少留5GiB |
| 外部GPU最低显存余量 | 46,790 MiB | >=5GiB |
| GPU利用率中位数 | 83% | 报告项 |
| loss/gradient | 全部有限 | 必须有限 |

连续50批与“25批停止、resume到50批”的以下结果完全一致：

- `attempts.bin` SHA-256：`410b4b31ba5a24b003a2406f017229e9729976309fbeea2210c56695641bed17`；
- `candidate_snapshot_a409600.npz` SHA-256：`cab9b0bd0cba20c31b384c484d3868799aea87d97b7d602dfe7ce3a6937f473e`；
- 最终model参数逐tensor一致；
- optimizer state逐tensor一致。

## 4. 三个机制run

| Seed | 总attempts | 模型lane attempts | 模型canonical唯一 | 总吞吐 | Transformer top50均值 |
|---:|---:|---:|---:|---:|---:|
| 101 | 2,000,000 | 1,500,000 | 1,418,275 | 5,102/s | 0.6364% |
| 102 | 2,000,000 | 1,500,000 | 1,338,626 | 5,046/s | 0.6540% |
| 103 | 2,000,000 | 1,500,000 | 1,381,295 | 4,999/s | 0.6364% |

三份ledger SHA-256：

```text
seed 101  b1180ede7662dc143993bd36acca3c7608c457833db94e804a86257213610398
seed 102  d426c2d8d4fa1d8afffaf82009fe9abb6ad24761775ae1aed3335ff26f969c94
seed 103  3c512d636f5d8be22db338b360a558275b90ac4a525f56b2ed5d2ec4307b8d8a
```

作为直观对照，旧Transformer在总attempt 2m时只有约25.8万至33.9万个canonical；新机制即使只计算75%的模型lane，也提高到约133.9万至141.8万。

## 5. 四项联合门槛

| Seed | 末段/首段新canonical | 新公式右尾末-初 | top50 2m-1m | 同唯一数T-R |
|---:|---:|---:|---:|---:|
| 101 | 103.10% | +7.65bp | 0.00bp | +11.95bp |
| 102 | 105.17% | +5.52bp | +0.39bp | +12.49bp |
| 103 | 107.97% | +3.19bp | +0.02bp | +11.66bp |

联合判断：

1. 探索保持：平均105.41%，3/3通过；
2. 右尾提升：平均+5.45bp，3/3通过；
3. Archive持续：平均+0.14bp，2/3通过；
4. 同唯一数胜Random：平均+12.03bp，3/3通过。

四项全部通过预注册规则。seed 101的top50在1m后没有继续提高，但另外两个seed为正、三seed平均为正，因此按冻结规则通过；不得事后改成“3/3才算通过”。

## 6. 三套候选库

| Seed | Transformer | 本次Random lane | 两lane合并 |
|---:|---:|---:|---:|
| 101 | 0.6364% | 0.4939% | 0.6364% |
| 102 | 0.6540% | 0.4971% | 0.6540% |
| 103 | 0.6364% | 0.4772% | 0.6364% |

三个seed的合并top50均值与Transformer独立库完全相同，25% Random lane没有抬高最终top50；Random的作用是保留不可关闭的覆盖底线，不是替模型完成本次胜利。

## 7. 公式复杂度风险

三个seed的curated top50共150条，raw token长度全部为15；展示清理后平均长度仍为14.62 / 15.00 / 14.86。三组top50全部包含`GAP`、`MEAN`和`WIN_20`，多数还叠加`MUL/SUB`等结构。

这说明模型稳定抓住了formal已出现的20日跳空均值主题，同时继续在最大复杂度边界上搜索训练期增益。可能解释有两种，现有训练期数据无法分开：

1. 复杂组合确实提取了简单`MEAN(GAP,20)`之外的增量结构；
2. 复杂公式利用更多自由度拟合训练噪声。

因此“机制门通过”不能扩大为“复杂公式值得进入策略”或“应立即扩大训练预算”。

## 8. 证据位置

```text
.pi/profile/results/v3a-gate1-engineering-20260716/
  continuous/gate1_engineering_check.json
  continuous/gpu.csv
  resume/gate1_engineering_check.json
  identity/gate1_binding.json

.pi/profile/results/v3a-gate1-runs-20260716/
  runs/*/attempts.bin
  runs/*/training_summary.json
  runs/*/training_complete.json
  runs/*/retained_candidates.json
  runs/*/checkpoint_final.pt
  analysis/gate1_results.json
  analysis/gate1_report.md
  analysis/SHA256SUMS
```

远端和本地`gate1_results.json` SHA-256均为：

```text
d10465d0a27bc0ee12f1c13b11815c6086c99b912f7409a24f76ca78cec9d3a6
```

## 9. 下一步边界

当前只冻结以下结论：新反馈机制通过训练期Gate 1，值得进入下一轮研究决策。

后续训练期复杂度审计已完成，结果见`docs/V3A_StageD_训练期复杂度审计结果.md`。审计确认了稳定的`GAP + MEAN(20)`核心，也发现完整公式族跨seed分散、top500子期排名负相关和部分复杂外围结构不稳定。

尚未决定：

- 如何冻结简单核心、子期稳健复杂公式和原始极端top50三组候选；
- 是否先跑新的完整8m formal；
- 新formal是否继续25% Random lane；
- 旧formal、新2m机制和未来完整机制候选如何统一冻结后一次性进入2022。

这些选择会改变研究问题或validation使用方式，必须另行确认。2022 validation和2023+ final继续封存。