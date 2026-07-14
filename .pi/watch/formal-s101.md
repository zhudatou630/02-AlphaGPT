# formal-s101 监工授权

## 任务
监工 Stage D formal 训练第一个 transformer run（seed 101）。run_id
`v3a-stage-d-formal-transformer-s101-00185964276c`，目标 800 万 attempts，
预期产物是 `training_summary.json` + checkpoint + top 公式库。外推约 6.5 小时。

## 被唤醒后先读
- 本文件
- 项目 AGENTS.md（/home/zhujunshen/Quant/02-AlphaGPT/AGENTS.md）
- .pi/handoff/ 最新交接
- .pi/grill/2026-07-13-0628——formal训练方案.md

## 远端机器
西北B区 / 898机，RTX PRO 6000 96GB，AutoDL 实例。
- SSH：`ssh autodl-898`（~/.ssh/config 已配免密 key id_ed25519_autodl）
- 登录信息：.pi/local/autodl-login.md
- 代码目录：/root/02-AlphaGPT（git commit 54716bc）
- run 目录：/root/02-AlphaGPT/data/processed/v3a/training/runs/v3a-stage-d-formal-transformer-s101-00185964276c
- 检查脚本：/root/02-AlphaGPT/.pi-watch-check-s101.sh
- screen：`screen -ls` 看会话 `train-s101`
- 日志：/root/02-AlphaGPT/run-s101.log（stdout，可能为空；进度看 attempts.jsonl 行数）

## 检查命令退出码
- 0 = RUNNING（运行中，静默）
- 1 = COMPLETE（进程退出且有 training_summary.json）
- 2 = CRASHED（进程退出且无 summary）
- 超时/124 = SSH 不可达

## 动作授权（优先于通用原则）
- **COMPLETE（退出码 1）** → 核验 training_summary.json + checkpoint 齐全 → **关机**（远端 shutdown）→ 通知用户。产物留在远端 run 目录，先别拉回（等用户决定）。
- **CRASHED（退出码 2）** → 看 run-s101.log / attempts.jsonl 末尾诊断原因 → 保存已有 checkpoint（已在 run 目录）→ **关机** → 通知用户决定（调参 / resume / 换机器）。不要自行 resume 或改参数。
- **SSH 不可达（超时）** → 只通知用户，连不上无法操作。
- **改参数 / 删数据 / 改代码 / resume / 换机器** → 等用户确认，不自主做。

逻辑：保存状态和关机是可逆省钱动作，自主做；改变方案或增加成本的动作，等用户定。

## 关机方式
AutoDL 实例直接系统关机：
```
ssh autodl-898 'shutdown now'
```
关机后实例进入「已关机」状态，按存储费计费，不计 GPU 费。关机前确认 checkpoint/summary 已落盘。

## 关键身份（commit 54716bc）
- protocol_id: 00185964276cf9934870477656c81039252405b35e0e7a86f7194bff55d749a2
- binding_id: 53959fbb8b879317c7ae1fbee5be1414b8625c4248ec311080b9c02ceb1a82b0
- train_view_id: v3a-train-view-7380696637ca0678
- 部署中修了两个 grill 遗漏：gpu_class(4090d→rtx_pro_6000)、BatchTorchVM 内存门限(2GB→按显存比例)。详见 handoff。

## 后续 run（本监工不覆盖，用户启动后再搭）
第一个 run 没问题后，逐个跑 transformer seed 102/103 + matched_random seed 101/102/103。random baseline 用 scripts/v3a/random_baseline.py。
