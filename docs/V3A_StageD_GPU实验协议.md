# V3A Stage D GPU 实验协议

## 1. 当前决策

- Stage C 通过，可以进入 Stage D 准备。
- 本机只允许编写代码、运行单元测试、静态检查和不接触真实 V3A panel 的合成 fixture。
- Stage D 的小规模训练 pilot 和正式训练都只能在 GPU 服务器执行；不得在本机 CPU 上运行
  具有研究含义的训练。
- 历史上只批准过 Stage D pilot；当前 Transformer pilot 和 reward calibration pilot 均已完成。正式
  top-N protocol 已写入 `configs/v3a_stage_d_formal_topn.json`，formal train-view 和 binding 已生成，
  但尚未启动远端 run。
- 历史 pilot protocol、binding 和 runner 保留用于复核，不作为新的 Stage D 或 formal run 入口。
- 当前公式库主路径是完整 ledger、raw top 30/50 和 curated top 30/50；完整候选漏斗只作为未来
  可选分析，不再是公式库产出的硬门槛。
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

当前 Stage D 首先要回答：在固定公式语言、数据区间和 scorer 下，Transformer 能否稳定生成一批
可供后续研究的 ETF 排名公式，并保存完整、可复核的公式库。若之后批准 Transformer 与随机方法
的正式比较，再在新的 formal protocol 中明确是否加入相似度、聚类和多样性漏斗。

不得用单条最高分作为研究结论。当前输出优先保留 raw top-N 和 curated top-N；curated 只做训练后
展示整理，不改变 reward、ledger 或 canonical hash。

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
candidate funnel        历史 pilot 曾要求完整执行；当前默认导出不依赖它
```

覆盖不足、恒定信号、VM 无效或 scorer 无效的公式统一使用冻结的 scorer hard-invalid reward；不能让模型
通过输出不可用公式获得训练收益。当前训练候选配置使用 `training_invalid_reward=-0.01`，但 scorer
默认值仍为 `-5.0`。训练日志必须同时记录 reward、有效率、各无效原因、entropy、
EOS/长度分布、canonical/选择序列重复率、梯度有限性、吞吐和显存。

pilot 通过条件：

1. 固定代码、dataset、panel、ResearchSpec 和 protocol 身份全部匹配；
2. 真实 CUDA，attempt 账本精确为 50,000，语法非法率为 0；
3. 全部 loss、梯度、reward 与模型参数保持有限；
4. batch 256 的 CUDA 工程 probe 使用真实 train-view、公式 VM、scorer 和质量门禁，证明恢复后的下一批公式、奖励、模型、优化器与 RNG 精确一致；pilot 另在 10240 attempts 强制中断，验证计数、候选状态与完整 ledger 前缀经恢复后可继续；
5. 完整 ledger 能导出 raw top 30/50 和 curated top 30/50；curated 记录安全展示清理、`5e-6`
   reward 近似并列规则、原始 reward 排名和换位原因；候选漏斗只有在 protocol 明确启用时才验收；
6. 未读取 validation/final 指标，未把 pilot artifact 标成 formal；
7. 输出足够信息判断吞吐、模式坍缩和正式预算，但不根据 pilot 宣称 Transformer 胜过随机。

当前 curated 导出规则：展示层只做可证明不改变 VM 语义的清理，例如 `x+0`、`x-0`、`x*1`、
连续 unary 操作和常数 unary 组合；禁止 `x*0 -> 0`。当 reward 差距不超过 `5e-6` 时，优先展示
清理后更短的公式。原始 top-N 和完整 ledger 必须保留。

训练 checkpoint 还必须保存完整但紧凑的 canonical 账本，包括每个 quality-valid hash 的首次
attempt、累计出现次数、最高 reward 记录和对应 attempt；三个 top-500 bucket 只是该账本的有序
引用，不能代替完整账本。

## 4. Formal protocol 门禁

正式训练入口必须 fail closed：没有独立、带摘要的 formal protocol，或其中
`formal_budget_approved` 不是 `true`，不得启动 formal run。

当前已写入的 formal protocol 为 `configs/v3a_stage_d_formal_topn.json`，protocol ID 为
`b815644fedcb84a4b9d17ececf957801a1060b4d95c2dcbef5b615e66ea5873b`。formal binding 为
`cd9923cdfc301c0f086f96385c44bbae1a5bfc3b3314c8a25036faf82e77104e`，绑定代码 commit
`1793ee10808f1136589c65e408611a21469392ac`；它已通过身份门禁，但没有启动远端任务。

formal protocol 至少冻结：

- Transformer seeds 与 matched-random seeds；
- 每个 seed 的相同 attempt 数；
- 模型、optimizer、REINFORCE、batch、checkpoint 配置；
- 两种方法共用的 raw/curated top-N 导出政策；是否加入完整候选漏斗必须作为独立 formal 决策；
- Transformer 超过随机的成对比较指标和通过标准；
- 最大恢复次数、费用/时长边界与完成后关机策略；
- 代码、数据、ResearchSpec 和 protocol identity pins。

如果 formal 选择当前 top-N 主路径，protocol 还必须冻结以下 formula-library policy：raw/curated
尺寸为 `30/50`、整理版本为 `etf-v3a-display-simplifier-v1`、reward 容差为 `5e-6`，并明确
保留 raw artifact。代码会拒绝缺少这组政策的 top-N formal protocol。

本 protocol 已冻结：Transformer 与 matched random 各使用 seeds `101/102/103`，每个 seed
`1,000,000 attempts`。这是写入 protocol 的实验预算，不代表任务已经启动；部署前仍需基于最终
代码 commit 生成新的 binding，并由用户单独确认启动。

## 5. 后续停点

1. 历史 pilot 和 reward calibration 的 raw/curated 公式库先完成人工阅读；
2. `training_invalid_reward=-0.01` 和 curated `5e-6` 已写入 formal protocol；
3. formal binding 已生成，代码、数据、ResearchSpec 和 train-view 身份已完成核对；
4. 用户明确批准启动后，才在 GPU 服务器部署该 commit 并运行 CUDA 身份/smoke；
5. formal Stage D 完成后再次停止，不自动打开 2022。