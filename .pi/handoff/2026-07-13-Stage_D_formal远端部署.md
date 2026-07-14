# Stage D formal 远端部署交接

**时间**：2026-07-13（部署日更新）
**状态**：远端部署完成，第一个 formal run（transformer seed 101）正在 898 机训练，watchman 监工已上线。

## 必读

1. 本文件
2. `.pi/grill/2026-07-13-0628——formal训练方案.md` —— 完整决策记录（研究定位、训练参数、执行策略、监工授权）
3. 项目 `AGENTS.md`
4. `.pi/local/autodl-login.md` —— 机器登录信息
5. `.pi/watch/formal-s101.md` —— watchman 监工授权（监工 session 被唤醒时的唯一依据）

## 背景

Stage D formal 训练：用 Transformer + matched-random 各 3 个 seed（101/102/103），每 seed 800 万 attempts，比较"Transformer 搜索是否优于纯随机"。产出是搜索层比较结论 + top-50 公式库，不打开 2022/2023 样本外，不直接出交易策略。

## 部署中修了两个 grill 遗漏

grill 决定用 RTX PRO 6000 96GB 机器，但 protocol/代码停留在 4090D 假设，部署时暴露并修正（两次都重算了身份链）：

1. **protocol gpu_class**（commit 2219752）：`execution.gpu_class` 仍是 `rtx_4090d`，`require_stage_d_cuda` 在 PRO 6000 上校验失败。改为 `rtx_pro_6000`；`stage_d.py` 的 `APPROVED_FORMAL_PROTOCOL_IDS` 加新 protocol_id，`validate_stage_d_protocol` 的 execution 批准集合新增 `rtx_pro_6000`（保留 4090D 给已完成的 pilot）。
2. **BatchTorchVM 内存门限**（commit 54716bc）：默认 `max_output_bytes=2GB`（小显存保守值），batch 8192 的 VM 输出张量 4.7GB 被挡。`train_gpu.py` 改为按 `torch.cuda.get_device_properties(device).total_memory` 的比例设定（output/working 各 25%，total 50%）。

每次改 code → code_fingerprint 变 → 重建 train_view manifest（数据 npy 不变）→ 重新生成 binding → 同步远端。

## 当前身份链（commit 54716bc，已替代旧链）

旧身份链（protocol_id 12e2a009 / binding_id c746dd4 / train_view de903f2 / commit 3524144f）**已失效**。新链：

- protocol_id: `00185964276cf9934870477656c81039252405b35e0e7a86f7194bff55d749a2`
- binding_id: `53959fbb8b879317c7ae1fbee5be1414b8625c4248ec311080b9c02ceb1a82b0`
- code_commit: `54716bc9efeb3c43dd14c7aaa50bfbd117981282`（当前 HEAD）
- code_fingerprint: `d5cc0882ae57cd06fe1e9b3faf50148dfc0b0e0feac2fc31841b13cc9a9dee90`
- research_spec_id: `765fc68313d6f962d934863766b537259e0c6a7b09dc3bd282eb3f4dc0977b90`
- train_view_id: `v3a-train-view-7380696637ca0678`

## 远端环境（898机 / 西北B区）

- 机器：RTX PRO 6000 96GB，AutoDL，`ssh autodl-898`（~/.ssh/config 配免密 key `id_ed25519_autodl`）
- 代码：/root/02-AlphaGPT，HEAD=54716bc，工作区干净
- Python：/root/miniconda3/bin/python（torch 2.8.0+cu128, numpy 2.3.2, pandas 3.0.3；本机是 numpy 1.26.4/pandas 2.1.4，版本差异不影响身份校验，数值可能有微量差）
- 同步方式：rsync（排除 .venv/历史runs/references/tests/.pi），含 .git 供 `require_clean_v3a_code` 校验

## 第一个 run（seed 101）状态

- run_id: `v3a-stage-d-formal-transformer-s101-00185964276c`
- screen: `train-s101`
- 启动命令：
  ```
  cd /root/02-AlphaGPT && /root/miniconda3/bin/python scripts/v3a/train_gpu.py \
    --protocol-file configs/v3a_stage_d_formal_topn.json \
    --seed 101 \
    --train-view-dir data/processed/v3a/stage_d/formal_topn_train_view \
    --stage-c-report data/processed/v3a/stage_c_reports/v3a-stage-c-20260711-01.json \
    --binding-file data/processed/v3a/stage_d/formal_topn_binding.json \
    --out-dir data/processed/v3a/training/runs
  ```
- 观察到的指标（启动后约 30 分钟）：
  - 显存：起步 63GB，后涨到 86GB（candidate state/torch cache 增长，96GB 内仍在余量；若 OOM 崩了 watchman 按崩溃授权处理）
  - 吞吐：~341 attempts/s（pilot 是 153），外推 800 万 ≈ 6.5h
  - GPU 利用率：6%（CPU 后处理瓶颈，REINFORCE 典型，不影响正确性）
  - reward：step 1 均值 -0.0059 → step 11 +0.0007，accepted_unique 34%→87%，模型在学
  - checkpoint：step 10（81920 attempts）已写

## watchman 监工

- 配置：`.pi/watch/formal-s101.json`，授权：`.pi/watch/formal-s101.md`
- 检查命令：`ssh autodl-898 'bash /root/02-AlphaGPT/.pi-watch-check-s101.sh'`（exit 0=运行中 / 1=完成 / 2=崩溃 / 超时=不可达）
- timer：`pi-watch-formal-s101.timer`，每 10 分钟，linger 已开
- session：`019f5f0e-8341-7d49-923a-8535d580ab11`（监工｜AlphaGPT，GLM5.2 medium）
- 授权（grill 定，用户确认）：完成→核验产物→**关远端机**→通知；崩溃→保存 checkpoint→**关远端机**→通知用户决定；SSH 不可达→只通知；改参/resume/换机→等用户确认
- 关机：`ssh autodl-898 'shutdown now'`，关后只计存储费
- **本机 NUC 不能关**：Pi-Web 和 timer 在本机，关了监工失效

## 待执行（后续）

第一个 run 没问题（用户确认 reward/产物 OK）后，逐个跑其余 5 个：
- transformer seed 102 / 103（同 train_gpu.py，换 --seed）
- matched_random seed 101 / 102 / 103（scripts/v3a/random_baseline.py）

每换 run 需更新 watchman 检查命令里的 run_id（或改检查脚本支持参数化）。

## 待定 / 风险

- **显存上涨**：86GB 且在涨，关注是否 OOM。崩了有 checkpoint 可 resume。
- **entropy 0.005 是否合适**：盲调值，跑完看 entropy 坍缩程度再定。
- **numpy/pandas 版本差异**：远端 numpy 2.3.2/pandas 3.0.3 vs 本机 1.26.4/2.1.4，身份校验（SHA）不受影响，但若要严格复现数值结果需对齐版本。
