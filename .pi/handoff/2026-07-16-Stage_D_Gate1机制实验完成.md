# Stage D Gate 1 机制实验完成

时间：2026-07-16

## 当前状态

- 用户通过grill逐项批准持续学习方案，记录在`.pi/grill/2026-07-16-0800——持续学习方案.md`。
- 实现commit：`91b809dc2ad5b446d50530e1edaadb3695eaac94`。
- rolling mean工作区修复commit：`469aa8f07467dfed279cb400cb92a7b0456874b7`。
- Gate 1 protocol：`9f9cfbe379a7201fdc6156d3b6f866156d788fc54c6bce0c84dce4d4dbf0a3ca`。
- 最终binding：`dc37a173f2fa7daec5aeeb543883ddfca12d9dc0c1981a8209e2e90ca2b9f86c`。
- 905工程门通过，3个2m机制run全部完成，四项联合机制门全部通过。
- 三份完整ledger和小型闭合产物已拉回本机；905已关机。
- 2022 validation和2023+ final始终未读取。

## 主要结果

```text
探索保持            103.10% / 105.17% / 107.97%，3/3通过
新公式右尾末-初      +7.65 / +5.52 / +3.19 bp，3/3通过
模型top50 2m-1m     0.00 / +0.39 / +0.02 bp，2/3通过
同canonical数T-R    +11.95 / +12.49 / +11.66 bp，3/3通过
```

新机制Transformer top50均值为0.6364% / 0.6540% / 0.6364%。训练期机制成功，但三seed curated top50全部为15-token复杂公式，不能跳过复杂度/过拟合讨论直接打开validation。

## 工程事实

- 初版在resume后第42批遇到MEAN window瞬时30.28GiB分配OOM；不是泄漏或teacher-forcing。
- `469aa8f`对同一rolling mean操作按最多512行分块，逐行浮点结果与原实现一致。
- 最终连续50批吞吐5,340 attempts/s，canonical等待0.88%，teacher-forcing 1.00%。
- 外部峰值显存50,461MiB，最低余量46,790MiB，GPU利用率中位数83%。
- 连续与resume ledger/snapshot/model/optimizer一致。

## 权威文档与证据

- 设计：`docs/V3A_StageD_Gate1持续学习机制设计.md`
- 结果：`docs/V3A_StageD_Gate1机制实验结果.md`
- 机器结果：`.pi/profile/results/v3a-gate1-runs-20260716/analysis/gate1_results.json`
- 工程证据：`.pi/profile/results/v3a-gate1-engineering-20260716/`
- 三run证据：`.pi/profile/results/v3a-gate1-runs-20260716/runs/`

## 下一步

不要继续修改训练代码、启动8m run或打开2022。先与用户讨论复杂度风险和下一研究门：复杂度约束/短公式对照、完整新formal、以及validation时点。2023+ final继续封存。