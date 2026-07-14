# V3A Stage D 新训练框架与当前进度

更新时间：2026-07-14

## 1. 文档定位

本文记录 V3A Stage D 从旧训练原型迁移到新训练框架的背景、设计原则、代码结构、状态模型、实现进度和下一阶段任务。它主要回答：

1. 为什么旧框架在 formal 规模下变慢；
2. 新框架从宏观上怎样分配 GPU、CPU、内存和磁盘工作；
3. 新旧实现有哪些实质区别；
4. 关键文件分别负责什么；
5. 阶段 1 至 4 已经实现和验证了什么；
6. 进入远端工程验收前还缺什么；
7. 何时才能建立新的 formal 身份并开始正式训练。

本文是当前实现与交接文档，不替代研究 protocol。研究数据边界、比较问题和 formal 预算仍需由 protocol 和 binding 冻结。

当前必须区分三种状态：

- **工程实现事实**：阶段 1 至 4 的本地代码已经完成，并通过合成 fixture。
- **工程性能结论**：尚未形成，必须在 898 GPU 服务器运行阶段 5 后才能判断。
- **正式研究结论**：完全没有形成，六个新 formal run 尚未开始。

## 2. 当前结论

旧 Stage D 代码已经证明研究链路能够运行，但它本质上是从 pilot 放大的原型。formal 规模从 batch 256、5 万 attempts 扩大到 batch 8192、800 万 attempts 后，计算调度、内存状态和 checkpoint 设计没有同步扩容。

新框架采用以下总体结构：

```text
GPU 批量主路径
    公式语法状态机
    -> Transformer / uniform random 抽样
    -> VM 指令
    -> GPU VM
    -> scorer / quality
    -> reward
    -> Transformer backward / optimizer.step

CPU 后台路径
    GPU 批量结果一次搬到 CPU
    -> 多进程 compile / canonicalize / hash
    -> 主进程按 attempt 原顺序提交

状态与磁盘
    append-only binary ledger 作为 attempt 事实来源
    -> 紧凑候选索引作为可重建缓存
    -> 小型训练 checkpoint
    -> 低频候选索引快照
```

阶段 1 至 4 已经把这个结构落实到本地代码。当前尚未在真实 CUDA 上证明吞吐、worker 数量、显存余量和快照耗时。

## 3. 研究内核与工程边界

### 3.1 不允许改变的研究内核

新框架必须保持：

- 训练数据区间和 train-view 边界不变；
- 公式语言和公式数学含义不变；
- VM、scorer、reward 和质量门禁含义不变；
- Transformer 与 matched random 使用相同公式语言、有效性规则、attempt 预算和候选整理规则；
- Transformer 按模型概率抽样；matched random 在合法动作中均匀抽样；
- validation 和 final OOS 在 Stage D 训练期继续封存；
- 两种方法使用成对 seed 和同一正式比较口径。

### 3.2 不再冻结的旧实现细节

由于旧 formal 训练没有完整跑完，新框架不要求复现旧程序的：

- 相同 seed 下的逐条公式序列；
- 旧 `formula_sequence_digest`；
- JSONL ledger 字节；
- Python 对象结构；
- 完整候选字典 checkpoint；
- 每 10 batch 保存一次的行为；
- seed 101 旧 checkpoint 的连续 resume。

旧 seed 101 停在 737,280 attempts，只保留为性能诊断材料。新框架通过后，Transformer 和 matched random 的 3+3 个 formal run 都从头开始，不混用旧结果。

## 4. 旧框架为什么慢

898 机临时副本的性能体检显示：

- 实际 cgroup 配额是 22 vCPU、110GiB 内存，不是宿主机显示的 208 CPU 和 1TiB；
- GPU 为 RTX PRO 6000 96GB；
- 旧运行 GPU 利用率采样中位数为 0%，均值约 1.37%；
- 旧 `sample_formulas` 每个 token 位置逐条处理 8192 个 Python `GrammarState`；
- 每个 batch 最多 16 轮 CPU/GPU 来回交接；
- 公式在采样后先用 Python compile，再复制回 GPU VM；
- 候选入库重复 compile、canonicalize、hash 和结构校验；
- candidate state、reward 列表、canonical ledger 和表达式对象被重复保存在多个位置；
- 737,280 attempts 时 checkpoint 已约 556MB，保存和 resume 扫描成本继续增长。

本地语法穷举同时证明：

- 公式词表只有 27 个 token；
- 最大公式长度为 15；
- 可达 `GrammarState` 只有 121 个；
- 状态/长度组合只有 759 个；
- 合法转移只有 3343 条。

因此最主要的问题不是 GPU 算力不足，也不是 canonicalize 单函数特别慢，而是大量逐条 Python 控制让 GPU 等待 CPU。

## 5. 新旧框架对比

### 5.1 代码设计原则

新实现不是把旧循环机械拆成更多文件，而是按数据所有权划分边界：

1. **研究定义只有一份**：语法、因子、VM、scorer、canonicalize 和 REINFORCE 继续复用原权威实现；
2. **GPU 负责批量数值工作**：公式状态、抽样、VM、打分和模型更新不再逐公式回 CPU；
3. **worker 只做纯计算**：CPU 进程不能修改共享候选状态，不能直接写 ledger；
4. **全局状态只有一个写入者**：主进程严格按 attempt 顺序提交，避免并发锁和不可复现顺序；
5. **历史事实只保存一次**：attempt 事实进入 ledger，索引和 summary 都可重建；
6. **并发必须有界**：最多保留两批 CPU 工作，慢时显式背压，不用无界队列掩盖瓶颈；
7. **checkpoint 保存恢复边界，不保存全部历史对象**；
8. **开发期证明正确，训练热路径只保留少量必要检查**；
9. **先串行闭环，再增加多进程，最后增加重叠**：每个阶段只引入一种复杂度；
10. **吞吐优先于表面利用率**：最终指标是 attempts/s、恢复成本和单位租金，而不是强求 CPU/GPU 同时 100%。

### 5.2 对比表

| 维度 | 旧框架 | 新框架 |
|---|---|---|
| 公式语法 | 8192 个 Python 状态逐条更新 | 121 状态的 GPU 查表 |
| action mask | 每个 token 位置由 CPU 逐行构造 | GPU 按状态 ID 和长度批量索引 |
| token 状态转移 | action 搬回 CPU 后逐条执行 | GPU 批量状态转移 |
| VM 指令 | CPU AST compile 后重新复制到 GPU | 采样时由状态转移直接产生 GPU VM 指令 |
| GPU 主路径 | 被 CPU 语法和编译频繁打断 | 采样、VM、scorer、reward、backward 连续执行 |
| CPU 整理 | 主进程单线程逐公式 | spawn 多进程按块整理 |
| CPU/GPU 调度 | 完全串行 | GPU batch N+1 与 CPU batch N 重叠 |
| 全局提交 | 计算和状态更新混在逐条循环 | worker 只计算，主进程 FIFO 有序提交 |
| attempt ledger | 逐条 JSONL | 109 字节固定宽度二进制记录 |
| 候选状态 | 嵌套 Python dict 和完整表达式 | hash 到行号映射 + 连续 NumPy 数组 |
| reward 历史 | 全量 Python list 常驻 checkpoint | attempt reward 只在 ledger，完成时离线汇总 |
| checkpoint | 模型和全部候选历史一起 pickle | 模型/optimizer/RNG/持久化边界的小 checkpoint |
| 候选快照 | 每个 checkpoint 重复保存全部状态 | 每 120 分钟保存紧凑索引快照 |
| 保存频率 | 每 10 step 或旧时间门槛 | 快 checkpoint 30 分钟，索引快照 120 分钟 |
| resume | 加载巨大 checkpoint 并深扫历史 | 截断到 checkpoint，加载快照并重放增量 ledger |
| Transformer/random | 两套独立 runner 和状态逻辑 | 同一 sampler、VM、scorer、ledger、index、runner |

## 6. 新框架的计算流程

### 6.1 GPU 公式生成

启动时，CPU 根据 `language.py` 的权威语法枚举状态，并生成三张小表：

```text
legal_actions[length, state_id, policy_action]
next_states[state_id, formula_token]
emitted_codes[state_id, formula_token]
```

表随后常驻 GPU。训练时每个公式只保留一个整数 `state_id`，不再创建 Python `GrammarState`。

token 位置仍然必须顺序执行，因为后一个 token 依赖前面的公式；但每个 token 位置上的 8192 条公式由 GPU 批量处理。

采样器预分配：

- `token_ids[batch,max_len]`；
- `token_lengths[batch]`；
- `vm_codes[batch,max_len]`；
- `vm_lengths[batch]`；
- log-prob、entropy 和 normalized entropy。

公式生成结束时，GPU 已经得到紧凑 VM 指令，不需要 CPU 先构建完整 AST 才能执行 VM。

### 6.2 Transformer 与 matched random

两种方法共用 `TensorFormulaSampler`：

```text
Transformer：模型 logits + 合法动作 mask
matched random：全零 logits + 合法动作 mask
```

全零 logits 在合法动作集合上产生均匀分布。因此两种方法的差异只剩“token 概率来自模型还是均匀分布”。

matched random 不创建模型和 optimizer，也不做 backward；其他路径完全共用。

### 6.3 GPU VM、scorer 和训练

每个 batch 的 GPU 路径为：

```text
sample
-> BatchTorchVM.execute
-> score_signal_batch
-> signal_quality_batch
-> effective_training_rewards
-> reinforce_objective（仅 Transformer）
-> backward / gradient clip / optimizer.step（仅 Transformer）
```

VM 可直接使用语法表给出的最大栈深度 8，不再为了计算栈深度把整个指令矩阵复制回 CPU 扫描。

### 6.4 GPU 到 CPU 的批量交接

GPU 完成模型更新后，一次性搬运：

- token IDs 和长度；
- VM/scorer/quality 有效性；
- reward 和 training reward；
- coverage 和 finite std；
- selected asset indices。

不再逐公式执行 `.item()` 读取 training reward。

### 6.5 CPU 多进程整理

`AttemptBatchPreparer` 使用持久化 `ProcessPoolExecutor`。默认 16 个 worker，启动方式为 `spawn`：

- 避免已经初始化 CUDA 后使用不安全的 `fork`；
- 每个 worker 处理一个连续 chunk；
- batch 8192、16 worker 时每块约 512 条；
- worker 只运行纯函数式 compile、canonicalize、canonical hash 和 selection hash；
- worker 不访问 CUDA、模型、optimizer、RNG、候选索引或文件。

worker 可以乱序完成，但 future 按提交顺序收集，chunk 内 attempt 顺序不变，最终 batch 恢复原始顺序。

### 6.6 GPU/CPU 双流水线

开启重叠时：

```text
GPU：batch N+1 采样、执行、打分和训练
CPU：batch N compile、canonicalize 和 hash
主进程：在安全位置 FIFO 提交 batch N
```

runner 分开维护：

- `generated`：GPU 模型已经完成到哪个 attempt；
- `state["attempt_count"]`：CPU 索引和 ledger 已提交到哪个 attempt。

pending 队列最多包含两批。GPU 完成 N+1 后、主进程提交 N 前，模型可能短暂领先已提交 ledger 两批；提交 N 后通常只领先一批。任何 checkpoint、快照、停止或完成前都会排空 pending，使模型、RNG、索引、日志和 ledger 回到同一 attempt 边界。

CPU 过慢时，主进程会等待最早的 future，形成背压，不允许任务无限堆积。

## 7. attempt ledger

### 7.1 二进制记录

`attempts.bin` 每条固定 109 字节，字段包括：

```text
attempt_index          uint64
training_step          uint32
status                 uint8
token_len              uint8
token_ids[15]          uint8
canonical_hash[32]     uint8
selection_hash[32]     uint8
reward                 float32
training_reward        float32
coverage               float32
finite_std             float32
```

无效公式的研究 reward 使用 NaN 表示；training reward 仍保存实际训练使用的 hard-invalid 值。

token name、原始表达式和 canonical 表达式不重复保存，因为可以从 token IDs 重建。

800 万 attempts 的 ledger 理论大小约为：

```text
8,000,000 × 109 bytes
= 872,000,000 bytes
≈ 0.81 GiB
```

### 7.2 ledger 职责

ledger 是 attempt 历史的唯一事实来源。它负责重建：

- canonical 出现次数；
- selection 出现次数；
- status、长度和 token 分布；
- 最佳 reward 和最佳 attempt；
- 候选 top bucket；
- reward 分布。

候选索引、summary 和 retained candidates 都是 ledger 的派生结果。

### 7.3 顺序和恢复

ledger 只接受连续 attempt index，且不允许未解析的 semantic-valid 状态落盘。

正常打开要求文件大小是 109 字节整数倍。resume 允许看到崩溃留下的不完整尾记录，但只信任完整前缀，并立即按 checkpoint 的安全 record count 截断。截断后重新计算前缀 SHA-256，再与 checkpoint 比较。

## 8. 紧凑候选索引

候选索引不保存完整表达式对象。canonical 部分主要保存：

- canonical hash；
- 全部出现次数；
- semantic-valid 出现次数；
- 首次 attempt 和首次有效 attempt；
- 最佳 reward、attempt、training step；
- 最佳 token IDs、长度、coverage、std。

selection 部分保存：

- selection hash；
- 出现次数。

查找使用：

```text
Python dict: hash bytes -> array row
NumPy arrays: row -> compact fields
```

数组容量按需扩张。top-500 不在每个 attempt 上维护复杂 heap，而是在最终输出等真正需要 top 候选时，根据紧凑数组批量排序生成，避免每次候选变化扫描数百万条记录。

完整表达式只对最终保留的少量候选重新构建。

## 9. checkpoint 与候选快照

### 9.1 快 checkpoint

默认每 30 分钟保存，包含：

- model state（matched random 为 None）；
- optimizer state（matched random 为 None）；
- Python、NumPy、Torch、CUDA RNG；
- step 和 attempt count；
- binary ledger record count、byte offset 和 prefix SHA-256；
- training log offset 和 prefix SHA-256；
- 最近候选快照文件名、attempt、elapsed 和 SHA-256；
- elapsed seconds 和 resume count；
- run identity。

快 checkpoint 不包含候选索引数组，因此大小不随 attempts 线性增长。

### 9.2 候选索引快照

默认每 120 分钟保存为未压缩 NPZ，内容是紧凑索引数组和必要累计统计。

checkpoint 通过以下三项绑定快照：

- 快照文件名；
- 快照内部 attempt count；
- 快照 SHA-256。

resume 时任一项不一致都会停止，不会静默接受另一 run 的快照。

### 9.3 一致性屏障

保存前执行：

```text
停止产生新 batch
-> 等待全部 CPU future
-> FIFO 提交全部 pending batch
-> flush + fsync ledger
-> flush + fsync training log
-> 必要时保存并校验候选快照
-> 保存 model / optimizer / RNG checkpoint
-> 恢复训练
```

### 9.4 最终 checkpoint

训练完成时先原子写 `checkpoint_final.pt`，再更新 `checkpoint_latest.pt`。

如果两次写入之间停机，latest 仍指向旧安全点；恢复后最多重跑一段，不会出现 latest 已完成但 final 不存在的状态。

## 10. resume 流程

resume 按以下顺序执行：

1. 加载并核对 checkpoint schema、run ID 和 run identity；
2. 以允许 torn tail 的模式打开 `attempts.bin`；
3. 按 checkpoint record count 截断 ledger；
4. 按 checkpoint byte offset 截断 training log；
5. 核对 ledger 和 training log 前缀 SHA；
6. 核对候选快照文件存在和 SHA；
7. 加载快照并核对快照内部 attempt；
8. 从快照 attempt 到 checkpoint attempt 重放 ledger；
9. 恢复 model、optimizer 和所有 RNG；
10. 从统一边界继续训练。

不存在“索引比模型多一批”或“ledger 比 RNG 少一批”的合法 checkpoint。

## 11. 关键代码文件

### 11.1 新增文件

| 文件 | 主要职责 |
|---|---|
| `src/alpha_etf/research_v3a/gpu_sampling.py` | GPU 语法表、Transformer/random 抽样、直接 VM 指令生成 |
| `src/alpha_etf/research_v3a/attempts.py` | 109 字节记录 schema、CPU batch 准备、binary ledger |
| `src/alpha_etf/research_v3a/attempt_workers.py` | spawn 进程池、分块提交、同步/异步结果拼接 |
| `src/alpha_etf/research_v3a/candidate_index.py` | 紧凑候选索引、顺序提交、NPZ 快照、ledger 重放 |
| `src/alpha_etf/research_v3a/stage_d_checkpoint.py` | 小型 checkpoint 构建、校验、保存和恢复 |
| `src/alpha_etf/research_v3a/stage_d_runner.py` | Transformer/random 统一 runner、双流水线、屏障、summary |
| `scripts/v3a/run_stage_d.py` | 薄 CLI、protocol/binding 门禁、数据加载、模型构造、运行锁 |
| `tests/test_v3a_stage_d_components.py` | 阶段 1 至 4 聚合 fixture |

### 11.2 修改文件

| 文件 | 修改 |
|---|---|
| `src/alpha_etf/research_v3a/language.py` | 增加权威的单次状态转移 VM 指令定义 |
| `src/alpha_etf/research_v3a/torch_vm.py` | 允许使用语法表已知的最大栈深度，避免 CPU 扫描指令 |
| `src/alpha_etf/research_v3a/torch_scoring.py` | 增加 GPU 批量 coverage/std 质量计算 |

### 11.3 继续复用的研究模块

| 文件 | 保持的研究含义 |
|---|---|
| `language.py` | 公式词表、语法和 AST 权威定义 |
| `factors.py` | 因子数学定义 |
| `torch_vm.py` | GPU 公式执行语义 |
| `torch_scoring.py` | ETF 横截面选择和 reward |
| `stage_d.py` | protocol、train-view、binding、REINFORCE 目标 |
| `candidates.py` | canonicalize、expression hash 和最终 CandidateRecord |
| `scripts/v3a/runtime.py` | code fingerprint 和 ResearchSpec 构造 |

### 11.4 暂时保留的旧入口

- `scripts/v3a/train_gpu.py`；
- `scripts/v3a/random_baseline.py`。

它们只作为历史参考，不是新 formal 主线。新框架通过阶段 5 后，再决定删除、归档或明确标成 legacy，当前不提前清理。

### 11.5 关键类和函数

| 符号 | 作用 |
|---|---|
| `language.transition_instruction_code` | 定义一次合法语法转移是否产生 VM 指令以及指令代码 |
| `gpu_sampling.build_tensor_grammar_tables` | 从权威语法枚举 GPU 查表张量 |
| `gpu_sampling.TensorFormulaSampler` | 执行 policy/uniform 批量公式采样 |
| `attempts.ATTEMPT_DTYPE` | 冻结 109 字节二进制记录格式 |
| `attempts.prepare_attempt_batch` | 单个 CPU chunk 的 compile、canonicalize 和 hash 纯函数 |
| `attempts.AttemptLedger` | binary ledger 追加、摘要、截断和批量读取 |
| `attempt_workers.AttemptBatchPreparer` | 单进程或 spawn 多进程 batch 准备 |
| `attempt_workers.SubmittedAttemptBatch` | 双流水线中的异步 CPU future 集合 |
| `candidate_index.CompactCandidateIndex.commit` | 按顺序解析重复状态并更新紧凑索引 |
| `candidate_index.CompactCandidateIndex.restore` | 加载快照并重放增量 ledger |
| `stage_d_checkpoint.build_training_checkpoint` | 构造不含候选大对象的恢复状态 |
| `stage_d_runner.SerialStageDConfig` | 冻结单个 runner 的方法、预算、worker 和保存参数 |
| `stage_d_runner.SerialStageDRunner` | 统一调度 Transformer/random、流水线、提交和恢复 |
| `scripts/v3a/run_stage_d.main` | 执行 protocol/binding 门禁并启动 runner |

## 12. 统一 runner 的运行产物

每个新 run 目录预计包含：

```text
.run.lock
attempts.bin
training_log.jsonl
storage_metrics.jsonl
candidate_snapshot_a0.npz
candidate_snapshot_a<attempt>.npz
checkpoint_latest.pt
checkpoint_final.pt             # 完成后
retained_candidates.json        # 完成后
training_summary.json           # 完成后
training_complete.json          # 完成后
```

`training_log.jsonl` 每 batch 一行，记录：

- loss、reward、entropy 和有效率；
- GPU batch 时间；
- CPU prepare 总时间；
- 主进程等待 CPU 时间；
- commit 时间；
- batch 完整延迟；
- pending CPU batch 数量；
- CPU 主进程峰值 RSS；
- CUDA allocated/reserved memory；
- 累计 attempts/s。

其中主进程 `RUSAGE_SELF` 不包含全部 worker 内存。阶段 5 必须从 cgroup 外部采样总内存。

`storage_metrics.jsonl`单独记录候选快照、latest/final checkpoint 和完整 checkpoint barrier 的保存耗时与文件大小。阶段 5 可用`--candidate-snapshot-on-stop`在部分运行停止点强制生成非空候选快照；这个开关只用于工程验收，不改变公式和 reward。

## 13. 四个本地实施阶段

### 阶段 1：独立组件

已完成：

- GPU 语法状态表；
- Transformer/random 共用张量采样；
- 直接 VM 指令；
- fixed-width attempt schema；
- binary ledger；
- 紧凑候选索引和快照；
- 小型 checkpoint。

### 阶段 2：串行新主线

已完成：

- `SerialStageDRunner`；
- Transformer 和 matched random 统一入口；
- GPU -> CPU -> index -> ledger 串行闭环；
- checkpoint、resume、retained candidates、summary 和 completion marker；
- `SIGTERM/SIGINT` 在当前 batch 结束后保存快 checkpoint。

### 阶段 3：CPU 多进程

已完成：

- 持久 spawn ProcessPool；
- 单 worker 与多 worker 共用同一 `prepare_attempt_batch`；
- chunk 按原顺序拼接；
- worker 异常 fail fast；
- CLI `--cpu-workers`，当前默认 16。

### 阶段 4：GPU/CPU 双流水线

已完成：

- 异步提交 CPU prepare；
- GPU batch N+1 与 CPU batch N 重叠；
- pending 队列最多两批；
- FIFO 单提交器；
- checkpoint/snapshot/stop/final 屏障；
- CLI `--disable-cpu-gpu-overlap` 用于工程对照。

阶段 4 后又完成一次宏观和代码细节审查，并修复：

1. torn binary ledger 尾部导致 checkpoint 无法恢复；
2. final/latest checkpoint 双写窗口；
3. candidate snapshot 未与 checkpoint 绑定；
4. Transformer 最后一个 batch 只有 1 条时 leave-one-out 必然失败。

## 14. 本地验证

当前聚合 fixture 命令：

```bash
PYTHONPATH=src .venv/bin/python -m unittest tests.test_v3a_stage_d_components
```

最近结果：

```text
Ran 4 tests in 4.612s
OK
```

已经覆盖：

- 121 个语法状态和 3343 条合法转移；
- 张量采样 VM 指令与 `compile_formula` 一致；
- Transformer log-prob 可反向传播；
- binary ledger 写入、读取、digest 和截断；
- 候选快照 + ledger 重放与连续提交一致；
- 小型 checkpoint 恢复 model、optimizer 和 RNG；
- matched random 端到端完成；
- 单 worker、关闭 overlap、连续运行；
- 多 worker、开启 overlap、中断/resume；
- 两种路径的 `attempts.bin` 逐字节一致；
- 两种路径最终 Transformer 参数完全一致；
- overlap 日志实际出现 pending batch；
- torn ledger 尾部可按 checkpoint 恢复；
- 被篡改候选快照被拒绝；
- final 先于 latest 写入；
- Transformer 单条尾 batch 被提前拒绝。

这些都是本地合成工程验证，不是训练期观察，更不是正式研究结论。

## 15. 当前工作区状态

截至本文更新：

```text
worktree / branch: multi-cpu
base commit:       607d272
实现状态:          阶段 1 至 4 本地完成
git 状态:          新实现尚未提交
main 合并:         尚未进行
898 远端:          已关机，尚未部署新代码
formal 身份链:     尚未重建
formal 新 run:     0 个
```

当前 `.pi/grill/` 决策记录和 `.pi/profile/` 诊断材料也尚未作为代码提交处理。

旧 seed 101 正式目录仍停在 737,280 attempts，checkpoint 和 ledger 未被性能诊断修改。它不是新框架的 formal 结果。

## 16. 当前 protocol 与身份链的重要提醒

`configs/v3a_stage_d_formal_topn.json` 仍然包含旧 checkpoint 配置：

```json
"checkpoint": {
  "every_steps": 10,
  "max_seconds": 600
}
```

新 runner 实际使用：

```text
fast checkpoint:      1800 秒
candidate snapshot:   7200 秒
```

当前 CLI 会把新 storage schedule 写入 run identity，但正式 protocol 本身还没有更新。因此现有 formal protocol、ResearchSpec、train-view manifest 和 binding 都不能直接作为新 formal 身份继续使用。

阶段 5 工程验收通过后，需要：

1. 更新正式 protocol 的分层保存配置；
2. 重新计算 protocol ID；
3. 提交最终代码，获得新 commit；
4. 重新计算 code fingerprint 和 ResearchSpec；
5. 重建 train-view manifest 和 Stage D binding；
6. 人工核对新身份链；
7. 再决定是否启动六个 formal run。

现有 `docs/V3A_StageD_GPU实验协议.md` 中关于“checkpoint 保存完整 canonical 账本”和旧 formal attempts/身份的内容属于旧实现状态，不能作为新 runner 的实现说明。研究原则仍可参考，但 formal 启动前需要同步更新状态文档。

## 17. 阶段 5：远端工程验收

阶段 5 的目标不是训练出研究结论，而是证明 800 万规模在目标机器上可运行、可恢复、成本可接受。

### 17.1 上机前准备

1. 对当前 diff 做最终范围核对；
2. 将阶段 1 至 4 代码和本文档提交到 `multi-cpu`；
3. 生成只用于工程验收的新代码身份和 binding；
4. 增加强制执行一次非空 candidate snapshot 的工程开关；
5. 准备外部 cgroup CPU、内存、GPU、磁盘采样；
6. 明确验收目录与正式 run 目录隔离；
7. 保证测试完成后自动收集产物并关闭 898。

### 17.2 worker 数量预选

不应直接假定 16 worker 最快。先用少量 batch 比较：

```text
cpu_workers = 1 / 4 / 8 / 16
overlap     = off / on（至少保留一个对照）
```

预选只比较工程吞吐、CPU wait、总 cgroup RSS 和 IPC 代价，不形成研究结果。选择一个稳定配置后，再运行完整 50-batch 门禁。

### 17.3 50-batch 工程门禁

正式门禁长度：

```text
50 × 8192 = 409,600 attempts
```

最低通过条件：

- 正确性 smoke 和 resume 通过；
- 稳态吞吐不低于 1000 attempts/s；
- 目标吞吐为 2000 attempts/s 以上；
- GPU 不长期等待可避免的 Python 逐条控制；
- CPU pending 队列不持续满载；
- cgroup 总内存外推到 800 万后低于 70GB；
- GPU 至少保留约 5GB 余量；
- 单 run 总产物外推低于 8GB；
- 快 checkpoint 小于 5 秒；
- 非空 candidate snapshot 外推小于 60 秒；
- 正常 resume 小于 10 分钟；
- batch 时间不随 attempt 增长持续恶化。

50-batch测试中需要强制执行：

1. 一次快 checkpoint；
2. 一次非空候选快照；
3. 一次停止和 resume；
4. 一次 ledger、snapshot、checkpoint 边界核对；
5. 一次产物体积和 800 万容量外推。

### 17.4 阶段 5 需要采集的指标

训练内部日志：

- `gpu_batch_seconds`；
- `cpu_prepare_seconds`；
- `cpu_wait_seconds`；
- `commit_seconds`；
- `batch_latency_seconds`；
- `pending_cpu_batches`；
- attempts/s；
- CUDA allocated/reserved。

外部系统采样：

- cgroup `cpu.stat`；
- cgroup `memory.current` 和 `memory.peak`；
- 所有 worker 的总 RSS；
- GPU utilization 和显存；
- ledger、snapshot、checkpoint 写入时间和字节数；
- 进程池启动和首批预热时间。

## 18. 远端 profile 后再决定的问题

以下问题目前有可能成为下一瓶颈，但没有证据支持现在修改：

### 18.1 GPU VM 中的 Python/CUDA 同步

`torch_vm.py` 仍在 position/code 循环中使用 `.any().item()`。旧 profile 显示 VM 约 0.41 秒/batch，不是旧系统首要瓶颈。采样器优化后需要重新 profile，只有 VM 占比明显上升才考虑重写。

### 18.2 scorer 的稳定全量排序

当前 scorer 使用稳定 `argsort` 保持并列资产顺序。它会产生较大的 int64 排序结果，但直接改成 `topk` 可能改变并列语义。必须在真实 profile 和研究语义评估后决定。

### 18.3 scorer 与 quality 的重复决策信号扫描

两者都会从大 signal tensor 选择 decision dates。是否合并要看 GPU 时间和峰值显存，当前不静态优化。

### 18.4 ProcessPool IPC

selected indices 等 CPU 输入会按 chunk 通过 spawn IPC 复制。是否需要 shared memory 或固定 pinned 双缓冲，取决于：

- CPU prepare 是否仍为瓶颈；
- IPC 占比；
- 1/4/8/16 worker 的收益曲线；
- 总 cgroup RSS。

如果单进程 CPU 工作已经被 GPU 完全隐藏，应减少 worker，而不是保留复杂度追求 CPU 满载。

### 18.5 大索引恢复

加载候选快照后需要从 hash 数组重建两个 Python lookup dict。它不影响正常 batch，但后期 resume 可能出现线性成本。先测 800 万外推，再决定是否需要更紧凑哈希表、分片或数据库。

## 19. 明确不做的事情

当前不做：

- 不在本机进行有研究含义的训练；
- 不延续旧 seed 101；
- 不复刻旧程序逐条随机序列；
- 不打开 2022 validation 或 2023+ final；
- 不提前引入数据库；
- 不为了追求 CPU/GPU 100% 利用率增加无意义计算；
- 不在远端 profile 前重写 VM 或 scorer；
- 不建立无界任务队列；
- 不让 worker 修改全局候选状态或写 ledger；
- 不把大量逐公式深校验重新放回训练热路径；
- 不在阶段 5 工程结果出来前启动六个 formal run。

## 20. 新入口示意

新统一入口为：

```bash
python scripts/v3a/run_stage_d.py \
  --method transformer \
  --protocol-file <new_protocol.json> \
  --seed 101 \
  --train-view-dir <new_train_view> \
  --stage-c-report <stage_c_report.json> \
  --binding-file <new_binding.json> \
  --out-dir <engineering_or_formal_runs> \
  --cpu-workers 16
```

matched random 只替换：

```bash
--method matched_random
```

同步对照可增加：

```bash
--disable-cpu-gpu-overlap
```

当前不能直接用旧 formal binding 执行以上命令。代码尚未提交，新身份链也未生成。

## 21. 后续顺序

推荐严格按以下顺序继续：

```text
阶段 1-4 本地实现与验证（已完成）
-> 提交 multi-cpu 阶段实现
-> 准备阶段 5 强制快照和外部监控
-> 898 小批量 worker/overlap 预选
-> 898 50-batch 工程门禁
-> 拉回产物并做容量、吞吐和恢复结论
-> 898 关机
-> 用户确认是否接受工程结果
-> 更新正式 protocol 和文档
-> 重建 commit/code fingerprint/ResearchSpec/train-view/binding
-> 人工核对新身份链
-> 用户再次批准后，从头执行 3+3 个 formal run
```

阶段 5 通过之前，“新框架更快”仍是待验证工程假设；阶段 5 通过之后，也只能说明工程可运行，不能说明 Transformer 在研究上优于 matched random。

## 22. 相关记录

- `.pi/grill/2026-07-14-0316——StageD性能体检.md`
- `.pi/profile/2026-07-14——formal-s101性能体检结论.md`
- `.pi/grill/2026-07-14-0737——StageD新架构.md`
- `.pi/handoff/2026-07-13-Stage_D_formal远端部署.md`
- `docs/V3A_StageD_GPU实验协议.md`
- `docs/V3A_整体设计与当前状态.md`

其中旧 handoff 和旧实验协议记录历史部署状态；本文件记录新训练框架的当前实现事实。正式启动前必须以新的 protocol、binding 和最新 handoff 为准。