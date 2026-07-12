# V3A Stage D GPU 实验协议

## 1. 当前决策

- Stage C 通过，可以进入 Stage D 准备。
- 本机只允许编写代码、运行单元测试、静态检查和不接触真实 V3A panel 的合成 fixture。
- Stage D 的小规模训练 pilot 和正式训练都只能在 GPU 服务器执行；不得在本机 CPU 上运行
  具有研究含义的训练。
- 当前只批准实现 Stage D 和执行一次 GPU pilot。正式 attempts、seed 数和 matched-random
  预算仍未批准，必须在 pilot 报告后写入新的 formal protocol。
- Stage D 只读取训练区间 `2016-08-09..2021-12-31`。2022 validation 与 2023+ final
  OOS 继续封存。
- pilot protocol 绑定已通过的 Stage C 完成报告 SHA、dataset ID 和 panel SHA；报告或身份不一致
  时拒绝启动。
- Stage D 训练进程只接收独立的 train-view artifact：预计算因子、open、mask、symbols 和 dates
  均截止 2021-12-31，文件中不存在 2022/2023+ 数组。完整 panel 只由独立构建/身份门禁读取。
- 冻结分两层：本文件对应的 protocol 固定实验语义；代码提交及 train-view 生成后，再产生独立
  runtime binding，固定 commit、code fingerprint、ResearchSpec、train-view 和 Stage C 报告 SHA。
  GPU probe、训练和候选漏斗必须验证同一个由人工批准的 binding ID。

## 2. Stage D 要回答的问题

在相同公式 attempt 预算、相同合法 action 集合、相同 scorer 和相同完整候选漏斗下，经过
REINFORCE 训练的 Transformer，能否比均匀随机搜索找到整体更强且不过度重复的 ETF 排名公式。

不得用单条最高分作为胜负依据。正式比较对象是两种方法分别经过 canonical 去重、训练期信号
去重、`rho>=0.90` 聚类和 `20/20/10` 长度配额后得到的 50 条候选整体。

## 3. GPU pilot

pilot 是工程和训练稳定性验证，不形成正式研究结论，也不占用未来 formal seeds。

```text
mode                    pilot
method                  transformer
device                  cuda only
seed                    314159
attempts                50,000
batch size              256
model                   d_model=64, layers=2, heads=4, ff_dim=128, dropout=0
optimizer               AdamW, lr=1e-4, weight_decay=1e-5
advantage               leave-one-out batch baseline 后按标准差归一化，epsilon=1e-5
entropy coefficient     0.001，按含 EOS 的实际决策次数归一化
gradient clip           1.0
checkpoint              每 10 step 或 10 分钟，取更频繁者
candidate funnel        完整执行，但只标记为 pilot diagnostic
```

覆盖不足、恒定信号、VM 无效或 scorer 无效的公式统一使用冻结的 hard-invalid reward；不能让模型
通过输出不可用公式获得训练收益。训练日志必须同时记录 reward、有效率、各无效原因、entropy、
EOS/长度分布、canonical/选择序列重复率、梯度有限性、吞吐和显存。

pilot 通过条件：

1. 固定代码、dataset、panel、ResearchSpec 和 protocol 身份全部匹配；
2. 真实 CUDA，attempt 账本精确为 50,000，语法非法率为 0；
3. 全部 loss、梯度、reward 与模型参数保持有限；
4. batch 256 的 CUDA 工程 probe 使用真实 train-view、公式 VM、scorer 和质量门禁，证明恢复后的下一批公式、奖励、模型、优化器与 RNG 精确一致；pilot 另在 10240 attempts 强制中断，验证计数、候选状态与完整 ledger 前缀经恢复后可继续；
5. 完整候选漏斗能产生 50 条诊断候选，并保存全部 cluster 成员和排序依据；
6. 未读取 validation/final 指标，未把 pilot artifact 标成 formal；
7. 输出足够信息判断吞吐、模式坍缩和正式预算，但不根据 pilot 宣称 Transformer 胜过随机。

训练 checkpoint 还必须保存完整但紧凑的 canonical 账本，包括每个 quality-valid hash 的首次
attempt、累计出现次数、最高 reward 记录和对应 attempt；三个 top-500 bucket 只是该账本的有序
引用，不能代替完整账本。

## 4. Formal protocol 门禁

正式训练入口必须 fail closed：没有独立、带摘要的 formal protocol，或其中
`formal_budget_approved` 不是 `true`，不得启动 formal run。

formal protocol 至少冻结：

- Transformer seeds 与 matched-random seeds；
- 每个 seed 的相同 attempt 数；
- 模型、optimizer、REINFORCE、batch、checkpoint 配置；
- 两种方法共用的完整候选漏斗；
- Transformer 超过随机的成对比较指标和通过标准；
- 最大恢复次数、费用/时长边界与完成后关机策略；
- 代码、数据、ResearchSpec 和 protocol identity pins。

建议预算仍为候选值而非已批准值：Transformer 3 seeds、matched random 3 seeds、每 seed
1,000,000 attempts。最终值只在 GPU pilot 给出真实训练吞吐和稳定性后确认。

## 5. 后续停点

1. 本机实现和测试完成后停在 GPU pilot 前；
2. 用户手动开启普通 AutoDL 实例后，部署新冻结 commit 并重跑 CUDA 身份/smoke；
3. 执行 pilot，产物同步回本机并关机；
4. 向用户提交 pilot 报告和 formal protocol 建议，再次等待明确批准；
5. formal Stage D 完成后再次停止，不自动打开 2022。