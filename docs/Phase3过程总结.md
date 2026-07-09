# Phase3 过程总结

## 一句话结论

Phase3 已经把 AlphaGPT 的核心骨架在多 ETF 项目里跑通：公式从 Python 手写函数变成可生成、可保存、可复现的 token 序列；StackVM 可以执行公式；Phase2 scorer 可以给 reward；小 Transformer + REINFORCE 可以完成最小训练闭环。但当前结果只能证明工程链路成立，不能证明模型已经稳定挖出有效投资因子。

## Phase3 的目标

Phase3 要验证的不是某条公式能否赚钱，而是这条链路是否成立：

```text
Transformer 生成公式 token
-> StackVM 执行成 signal [ETF, time]
-> Phase2 scorer 给 reward
-> REINFORCE 更新生成器
-> 保存 checkpoint / artifact
-> 重新加载后可复现
-> validator 做交易审计
```

这和原作者 AlphaGPT 的核心思想一致：模型不是直接预测价格，而是自动写可解释公式，再用历史数据给公式打分。

## Phase3-0：环境准备

本地完成了项目内虚拟环境和 CPU 版 PyTorch 准备：

```text
.venv/
torch 2.12.1+cpu
numpy 1.26.4
pandas 2.1.4
pyarrow 24.0.0
CUDA 不可用
```

`.gitignore` 增加了：

```text
.venv/
.pi-subagents/
.pi/local/
AGENTS.md
```

其中 `.pi/local/` 用于保存本地敏感信息，不进 git。

## Phase3a：不训练模型的公式测试台

Phase3a 先不接 Transformer，目标是验证公式表示和评分链路。

新增模块：

```text
src/alpha_etf/gpt/vocab.py
src/alpha_etf/gpt/ops.py
src/alpha_etf/gpt/vm.py
src/alpha_etf/gpt/random_formula.py
scripts/phase3a_random_formulas.py
```

Phase3a 做了这些事：

```text
1. 定义 token 词表：feature / constant / operator
2. 用 StackVM 执行 RPN 公式
3. 随机生成半受控公式
4. 用 Phase2 scorer 给公式打分
5. 对 best formulas 跑 validator 审计
6. 写出 CSV / JSONL artifact
7. 重新加载 artifact 并校验 reward 一致
```

默认运行结果：

```text
总公式：302 条，包括 300 条随机公式 + 2 条 sanity 公式
有效公式：214
无效公式：88
无效率：29.14%
best_formulas.jsonl 重新加载 reward 差异：0.0
validator summary：40 行
```

sanity 公式验证通过：

```text
sanity_mom_10
sanity_ma_gap_5_20
```

这说明 VM token 公式和 Phase2 Python 固定公式口径对齐。

## Phase3a 的关键发现

Phase3a 暴露出一个重要问题：生成公式的 raw signal 零轴未必有交易含义。

Phase2B 原 validator 使用：

```text
signal > 0 才允许买入
```

这适合手写动量类公式，但不一定适合任意生成公式。有些公式全为负，但横截面排序仍然有效；scorer 只看相对排序，validator 却因为没有正数而不开仓。

因此后续不能简单把 Phase2B validator 主口径改掉，而应该新增 generated-formula 专用审计口径，例如 rank_only。

## Phase3b：小 Transformer 训练闭环

Phase3b 方案先经过 grill 确认，记录在：

```text
.pi/grill/2026-07-08-0858——Phase3b训练方案.md
```

关键决策：

```text
模型：小 Transformer，不用 GRU
尺寸：d_model=64, layers=2, heads=4, ff_dim=128
max_len=16
batch_size=64
train_steps=500
BOS/EOS/PAD：显式加入
Action mask：只约束栈结构合法，不加金融语义规则
reward：沿用 Phase3a scorer_mean_return
validator：不进入 reward，只训练后审计
rank_only validator：新增 generated-formula 审计 variant
entropy：默认 0.0，保留参数
advantage：batch 内标准化
critic：第一版不使用，最多预留接口
optimizer：AdamW, lr=3e-4, weight_decay=1e-5, grad_clip=1.0
checkpoint：保存完整 model / optimizer / config / RNG state
resume：第一版不支持
输出：data/processed/phase3b/runs/<run_id>/
```

新增模块：

```text
src/alpha_etf/gpt/policy.py
src/alpha_etf/gpt/sampling.py
src/alpha_etf/gpt/evaluation.py
src/alpha_etf/gpt/checkpointing.py
scripts/phase3b_train_transformer.py
tests/test_phase3b.py
```

`src/alpha_etf/validation.py` 新增：

```text
rank_only_signal
run_rank_only_validator
```

原始 validator 主口径没有覆盖。

## Phase3b 默认运行结果

最终本地 CPU run：

```text
data/processed/phase3b/runs/20260708-190425_seed42_steps500/
```

核心结果：

```text
总采样公式：32000
有效公式：31584
无效公式：416
最终 valid_rate：100%
best_reward：0.006878246107346835
best_formulas：20 条
artifact reload reward_abs_diff：0.0
validator summary：80 行
checkpoint 默认 torch.load：通过
```

validator 口径各 20 行：

```text
no_max_holding
rank_only_no_max_holding
max_holding_10
rank_only_max_holding_10
```

验证命令已通过：

```text
.venv/bin/python -m compileall scripts src tests
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests
.venv/bin/python scripts/phase3a_random_formulas.py
.venv/bin/python scripts/phase3b_train_transformer.py
```

## 训练结果怎么解读

工程闭环成功，但训练质量还没有证明。

训练曲线显示：

```text
step 1:   avg_reward=-1.168869, valid_rate=71.9%, entropy=41.64
step 100: avg_reward=0.005000, valid_rate=100.0%, entropy=36.43
step 500: avg_reward=0.005944, valid_rate=100.0%, entropy=1.96
```

说明模型确实学会了少写坏公式，action mask 也有效降低了结构非法。但 best formula 出现在第 1 个 step 的第 21 条样本，后续没有突破。

这说明当前训练更像：

```text
合法公式生成器 + 弱 REINFORCE 训练
```

还不能证明模型已经超过随机搜索。

## 第一轮出现最优公式的可能原因

最优公式：

```text
train_000001_021
reward=0.006878246107346835
```

它不是第 1 条公式，而是第 1 个 batch 里的第 21 条。

可能原因：

```text
1. batch_size=64 太小，后续梯度估计噪声大，难以稳定改进早期偶然高分。
2. 早期 entropy 很高，探索范围最大，更容易撞到极端高分公式。
3. 后期 entropy 快速下降，模型收缩到稳定有效公式，极值探索变弱。
4. reward 分布很平，第一名与第二名差距很小，排名受随机扰动影响大。
5. scorer 没有惩罚公式复杂度、冗余 token、极端信号值。
6. 当前没有 elite replay，历史最优不会被反复学习。
```

## 最优公式的问题

最优公式 token：

```text
low ABS DECAY MA10 MA5 RET10 RET10 ABS CONST_1 DELAY1 SIGN RET5 DELAY10 ADD DELAY1 MA10
```

有效部分可简化为：

```text
MA10(DELAY1(ABS(RET10(RET10(MA5(MA10(DECAY(low))))))))
```

其中：

```text
CONST_1 -> DELAY1 -> SIGN -> RET5 -> DELAY10
```

基本是零项，删掉后 reward 不变。`ABS(low)` 中的 ABS 也没有实际意义，因为 low 价格本来为正。

这个公式本质上更像低价序列平滑后的二阶变化幅度，存在 `RET10(RET10(...))` 这类可能制造极端值的结构。信号分布最大值达到约 `1.56e13`，说明未来需要考虑极值裁剪或标准化。

对应 validator：

```text
no_max_holding total_return=1.4058, ann_return=9.88%, max_drawdown=-58.86%
max_holding_10 total_return=1.4941, ann_return=10.31%, max_drawdown=-61.31%
benchmark total_return=0.8742, ann_return=6.97%
```

历史内看起来跑赢 benchmark，但最大回撤很大，不能作为可用策略结论。

## EOS 的发现

Phase3b 加了 BOS/EOS/PAD，但训练后平均公式长度仍然到 16：

```text
avg_formula_len = 16.0
```

原因是 EOS 只是“允许提前结束”，不是“鼓励提前结束”。当前 reward 没有奖励短公式，也没有惩罚冗余复杂度，所以模型没有理由提前停。

这说明 BOS/EOS/PAD 在 artifact/schema 上更干净，但训练行为上还需要配套机制：

```text
1. 轻微长度惩罚
2. EOS 使用率日志
3. 公式重复度/token 分布日志
4. 或实验原作者式“结构完成后强制 EOS/PAD”
```

## CPU 训练性能瓶颈

本地 CPU 默认 500 步耗时约：

```text
7564 秒，约 2 小时 6 分钟
```

这说明本地 CPU 已经完成“证明闭环”的任务，不适合继续长时间调参。

慢的原因不只是 Transformer，而是：

```text
1. 每条公式逐条 VM 执行
2. 每条公式逐条 scorer 横截面逐日打分
3. numpy/pandas 逻辑主要在 CPU
4. 训练脚本还没有批量化 scorer/VM
```

因此 GPU 不是万能加速。GPU 可以加速 Transformer forward/backward，但当前主瓶颈可能仍有较大一部分在 CPU 评估链路。

## AutoDL GPU 环境检查

已租用 AutoDL 服务器，登录信息保存在本地忽略文件：

```text
.pi/local/autodl-login.md
```

不要提交口令。

服务器检查结果：

```text
地区：重庆A区 019机
GPU：RTX 4090D 24GB
CPU 限制：18 核 AMD EPYC 9754
内存限制：60GB
系统：Ubuntu 22.04.5
驱动：580.105.08
PyTorch：2.8.0+cu128
Python：3.12.3
CUDA 可用：True
CUDA matmul smoke：通过
```

注意：非交互 SSH 下默认没有 `python` 命令，实际路径是：

```text
/root/miniconda3/bin/python
/root/miniconda3/bin/pip
```

当前缺：

```text
pandas
pyarrow
pytest
```

后续部署时需要安装。

## 下一步建议

不要继续在本机 CPU 上盲目调参。下一步应该分两条线：

```text
1. 写 GPU 实验部署/运行脚本
2. 设计小规模参数对比实验
```

建议第一批实验：

```text
A. 同预算 random baseline：判断训练是否超过随机搜索。
B. batch_size=256/512/1024：判断小 batch 是否是第一轮出最优的主因。
C. entropy_coef=0.001/0.01：判断探索塌缩是否能缓解。
D. EOS/长度惩罚实验：判断公式写满 16 是否影响训练。
```

在 GPU 机上先跑 smoke，不要直接长训：

```bash
python scripts/phase3b_train_transformer.py --device cuda --smoke-steps 50
```

然后观察：

```text
耗时
nvidia-smi GPU 利用率
CPU 利用率
valid_rate
entropy
best_reward 是否后期刷新
```

如果 GPU 利用率低、CPU 满载，优先优化 VM/scorer，而不是租更贵 GPU。

## 当前 git 状态

Phase3 代码与产物已提交：

```text
1bfb3df Add Phase3 token formula training pipeline
8fa47b8 Stop tracking local agent instructions
```

`AGENTS.md` 已从 git 跟踪中移除并加入 `.gitignore`，本地仍保留。服务器登录文件 `.pi/local/autodl-login.md` 也被 `.gitignore` 排除。
