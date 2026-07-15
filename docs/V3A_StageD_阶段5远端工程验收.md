# V3A Stage D 阶段 5 远端工程验收

验收时间：2026-07-15

验收结论：**通过，可以进入 formal protocol 与身份链重建；尚未批准或启动 formal 训练。**

## 1. 验收目的

阶段 5 只回答新 Stage D 框架能否在目标硬件上稳定运行：

- 真实 CUDA 吞吐是否超过最低 1000 attempts/s、目标 2000 attempts/s；
- GPU/CPU 双流水线是否产生实际收益；
- 1/4/8/16 CPU worker 应选择哪个；
- 真实 `SIGTERM` 后能否保存一致状态并 resume；
- 连续运行和中断/resume 是否得到相同结果；
- CPU 内存、GPU 显存、磁盘、checkpoint、候选快照和恢复时间能否支持 800 万 attempts；
- Transformer 与 matched random 是否都能通过新统一入口运行。

本阶段不回答 Transformer 是否优于随机搜索，也不读取 2022/2023+ 数据。

## 2. 最终门槛结果

| 门槛 | 要求 | 实测或保守外推 | 结论 |
|---|---:|---:|---|
| Transformer 吞吐 | 最低 1000，目标 2000+ attempts/s | 连续 7124；中断/resume 6849 | 通过 |
| matched random CUDA smoke | 新统一入口可运行 | 4916 attempts/s（5 batch） | 通过 |
| CPU 总内存 | 800 万外推 < 70GB | 保守外推 < 10GiB | 通过 |
| GPU 显存余量 | 约 5GB 以上 | 峰值 80,975MiB，余量 16,912MiB | 通过 |
| 单 run 磁盘 | < 8GB | 约 1.60GiB | 通过 |
| 快 checkpoint | < 5 秒 | 0.029 秒 | 通过 |
| 800 万候选快照 | < 60 秒 | 线性外推约 2.91 秒 | 通过 |
| resume | < 10 分钟 | 409,600 时完整启动额外约 3.8 秒；800 万外推 < 1 分钟 | 通过 |
| 中断一致性 | checkpoint/snapshot/ledger 同边界 | `SIGTERM` 后停在 221,184，边界一致 | 通过 |
| 连续与 resume 一致 | ledger、索引、模型、optimizer 一致 | 全部逐字节或逐 tensor 一致 | 通过 |

## 3. 验收机器

用户提供的机器 3：

```text
GPU               NVIDIA RTX PRO 6000 Blackwell Server Edition
GPU memory        97,887 MiB
driver            580.82.09
cgroup CPU        22 vCPU（cpu.max = 2200000 / 100000）
cgroup memory     118,111,600,640 bytes，约 110GiB
Python            /root/miniconda3/bin/python
Torch             2.8.0+cu128
NumPy             2.3.2
Pandas            3.0.3
```

机器 3 上原有旧仓库存在历史 rsync 删除状态，因此没有复用。各轮代码都从本地 git bundle 克隆到独立目录，旧目录和旧 898 seed 101 均未修改。

## 4. 最终被测身份

```text
tested code commit      df223f60cdc0f78dccec2f80aafb8168b77a692e
code fingerprint        a66e6a091e95a4255c15b3a573a5562f03c8ed3c95cf4bef196fac0b3f5f72b8
research spec id        36f55718a2da7d5d090c2f86bc4f29cf91961a8d68e0445b344a8a6d9a1db6c0
train view id           v3a-train-view-3218d05780b999ec
train view fingerprint  3218d05780b999ec66f3208ad05432fcd21e83217e55df76705fa7551ea41bd2
engineering binding id  9872ae38ae6375a33ac320c28b90dafcd83530a826aa789ba37260553ad8de3c
```

这个 binding 只用于阶段 5 工程验收。它引用旧 formal protocol 作为 batch、模型和 reward 参数来源，但所有 run 都使用独立 `stage5-*` run ID、隔离目录和部分 `stop-after`，没有生成 formal completion marker，不能作为正式研究结果。

## 5. 部署与身份核对

最终隔离目录：

```text
/root/02-AlphaGPT-stage5-v4
```

部署步骤：

1. 本地提交新框架代码；
2. 用该 commit 重建 stage5 train-view manifest；
3. 生成独立工程 binding；
4. 通过 git bundle 克隆远端仓库；
5. 单独同步 46MB train-view、Stage C 报告和 binding；
6. 远端重新执行 `build_stage_d_binding.py`；
7. 确认远端重建 binding 与本地文件逐字节一致；
8. 确认代码工作区 clean 后才启动 runner。

## 6. CPU worker 预选

每个预选 case 使用同一 seed 101、同一前 4 batch、共 32,768 attempts。

| 配置 | 累计 attempts/s | cgroup current 峰值 | 判断 |
|---|---:|---:|---|
| 1 worker，同步 | 3431 | 3.37GiB | 基线 |
| 1 worker，重叠 | 4214 | 3.85GiB | 重叠有效 |
| 4 worker，重叠 | 5113 | 4.70GiB | 已基本隐藏 CPU |
| 8 worker，重叠 | 5128 | 5.67GiB | 最终选择 |
| 16 worker，重叠 | 4944 | 7.89GiB | 更慢且更占内存 |

五个 case 的 `attempts.bin` SHA-256 完全一致，最终 Transformer 模型参数逐 tensor 一致，reward、候选计数和公式序列也一致。

最终选择 8 worker。16 worker 虽然把单块 CPU 整理时间继续压低，但 GPU 已经成为主要节奏，额外进程只增加启动、IPC 和内存成本。

## 7. 50-batch 主门禁

最终 Transformer run：

```text
run id              stage5-gate-v4-transformer-s101-w8
batch size          8192
total batches       50
total attempts      409,600
CPU workers         8
CPU/GPU overlap     enabled
forced snapshot     enabled at stop
```

### 7.1 真实信号停止

runner 启动目标仍为 409,600 attempts。外部控制器观察 ledger 达到 204,800 后向主 Python 进程发送真实 `SIGTERM`。

信号到达时 GPU 正在处理后续 batch。runner 按设计完成当前 GPU batch、等待并提交全部 CPU future，将 ledger、日志、模型和索引推进到同一边界，保存非空候选快照和快 checkpoint 后正常退出。

```text
attempt_count             221,184
candidate snapshot        candidate_snapshot_a221184.npz
snapshot attempt_count    221,184
checkpoint attempt_count  221,184
process exit status       0
```

### 7.2 resume

使用相同 code identity、run identity、worker 和存储开关执行 `--resume`，最终达到：

```text
attempt_count  409,600
step           50
resume_count   1
ledger bytes   44,646,400 = 409,600 × 109
```

第二段完整墙钟约 29.90 秒，其中实际继续训练约 26.12 秒。约 3.8 秒差额包含 Python/CUDA 启动、train-view 加载、模型和 optimizer 构建、checkpoint/快照/索引恢复，以及最终保存。

## 8. 连续运行对照

同一最终代码、seed、8 worker 和 409,600 attempts 另跑一条不中断路径：

```text
continuous runner elapsed   57.50 秒
continuous throughput       7124 attempts/s
continuous resume_count     0
```

与 `SIGTERM + resume` 路径比较：

| 对象 | 结果 |
|---|---|
| `attempts.bin` | SHA-256 一致 |
| 最终候选快照 | SHA-256 一致 |
| Transformer model state | 逐 tensor 一致 |
| optimizer param groups | 一致 |
| optimizer state tensors | 逐 tensor 一致 |

这证明最终框架自身在真实 CUDA 上满足连续运行与 checkpoint/resume 一致，不依赖旧原型的随机序列。

## 9. 显存问题与最终处理

### 9.1 第一轮发现

最初50-batch运行吞吐达到7699 attempts/s，但外部采样捕捉到93,547MiB显存峰值，只剩4,340MiB，略低于约5GB余量门槛，因此没有直接算通过。

### 9.2 scorer 分块

scorer 和 quality 按公式维从8192分成两个4096块。逐公式排序和 reward 不依赖其他公式，因此可以按原顺序拼接。本地与远端均证明分块前后 ledger、候选快照和模型参数一致。

但50-batch峰值仍为93,547MiB，说明scorer不是缓存增长的根因。

### 9.3 缩小 VM 工作区

曾把VM working gate从25%降到18%。5-batch看似降低峰值，但50-batch仍达到94,787MiB，而且不同chunk形状带来允许范围内的浮点路径变化。该方案没有解决问题，已回退。

### 9.4 根因与最终方案

训练日志显示 batch 提交时，实际存活 CUDA tensors 只有约0.04GiB，但CUDA reserved cache最高约92GiB。大量VM/scorer临时张量已经释放，PyTorch CUDA缓存池仍保留它们，跨batch累积后让 `nvidia-smi` 接近满显存。

最终代码在 GPU batch 完成、CPU数组已经搬出、局部GPU对象已经释放后执行一次 `torch.cuda.empty_cache()`，并把释放耗时计入 `gpu_batch_seconds`。

```text
GPU memory peak               80,975 MiB
GPU headroom                  16,912 MiB
CUDA reserved at commit peak   1,846 MiB
GPU utilization mean             64.2%
GPU utilization median           85.0%
```

代价是吞吐从不释放缓存时约7699下降到6849 attempts/s，约11%；但显存余量从4.3GiB提高到16.5GiB，且结果逐字节/逐tensor不变。这个取舍符合正式长运行的稳定性目标。

## 10. CPU、流水线和提交

```text
GPU batch（含cache释放）  mean 1.072s, median 1.033s
CPU wait                  mean 0.012s
commit                    mean 0.102s
pending batches max       1
cgroup memory peak        6.76GiB
main process peak RSS     2.11GiB
```

CPU prepare计时从提交时回看 future 生命周期，会包含与下一批 GPU 重叠的时间，因此不能直接解释为CPU占用。真正能说明背压的是 `cpu_wait_seconds`：均值只有0.012秒，CPU整理已经基本被GPU隐藏。

## 11. checkpoint 与候选快照

最终409,600边界：

```text
checkpoint_latest.pt             919,819 bytes
candidate_snapshot_a409600.npz  43,068,596 bytes
candidate snapshot save + SHA        0.149 s
complete checkpoint barrier           0.029 s
```

候选快照加载实测：

```text
attempts              409,600
canonical rows        330,131
selection rows        218,148
load time               0.184 s
RSS increment       141,074,432 bytes，约134.5MiB
```

## 12. 800 万 attempts 容量预测

### 12.1 时间

Transformer按连续吞吐约18.7分钟/run，按中断路径6849/s保守估计约19.5分钟/run。

matched random 5-batch为4916 attempts/s。按这个包含启动影响的保守值约27.1分钟/run。3个Transformer加3个matched random的纯训练工程估计约2.3小时，不包含身份重建、部署、run切换、产物核验和候选导出时间。

### 12.2 磁盘

```text
binary ledger             872,000,000 bytes，约0.812GiB
final candidate snapshot  约841MB，0.783GiB
training log              约1MB
checkpoint                约1MB
正常单run总量             约1.72GB，1.60GiB
```

即使保留若干中断快照，也有较大空间低于8GB门槛。

### 12.3 CPU 内存

快照加载的索引RSS增量线性外推约2.57GiB。加上模型、train-view、8个worker和固定运行时，保守总量小于10GiB，远低于70GB门槛。

### 12.4 快照和 resume

```text
800万快照保存 + SHA  约2.91秒
800万快照加载        约3.60秒
```

Python大字典并不保证严格线性，因此正式运行仍应保留实际storage日志；但即使放大10倍，也低于60秒快照和10分钟resume门槛。

## 13. matched random

最终代码额外运行5个batch、40,960 attempts：

```text
throughput         4916 attempts/s
GPU peak           53,121 MiB
cgroup peak         6.36 GiB
snapshot bytes      4,494,764
checkpoint bytes       17,349
```

matched random没有模型和optimizer，其他 sampler、VM、scorer、CPU worker、ledger、索引、snapshot和checkpoint路径均与Transformer共用。

## 14. 证据产物

本地证据目录：

```text
.pi/profile/results/v3a-stage5-20260715/
```

```text
file     v3a-stage5-evidence-v4-20260715.tgz
SHA-256  ace1fed428127f5a6a8e26baa4c4af60b0fdd042d02efa90a6d52ecdb5b6a0da
```

压缩包约56MB，解压后约86MB，包含最终工程identity、机器信息、409,600条binary ledger、非空候选快照、resumed/continuous checkpoint、训练和storage日志、cgroup/GPU采样、worker预选、连续/resume比较和matched-random smoke。

本地解压后已执行 `sha256sum -c sha256sums.txt`，全部通过。

## 15. 阶段 5 之后还不能做什么

阶段5通过只说明新Stage D框架在目标机器上达到速度、容量、显存和恢复门槛。它不说明：

- Transformer优于matched random；
- 800万过程中一定不会出现新的训练分布变化；
- top-50公式有样本外价值；
- 可以直接续跑旧seed 101；
- 旧formal protocol/binding仍然有效。

## 16. 正式训练前剩余任务

1. 更新formal protocol中的旧checkpoint配置，写入30分钟快checkpoint和120分钟候选快照；
2. 将最终代码合并到选定正式分支；
3. 重新计算protocol ID和代码批准集合；
4. 用最终commit重建code fingerprint、ResearchSpec和train-view manifest；
5. 重建formal binding；
6. 人工核对六个run的method、seed、attempts、batch、storage和比较规则；
7. 用户再次明确批准后，Transformer和matched random各3个seed从头启动；
8. 不续跑旧seed 101，不混用阶段5工程产物。

阶段5到此停止，不自动进入formal训练。