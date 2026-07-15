# V3A Stage D formal结果与下一轮优化设计

> 状态：2026-07-15 formal训练完成后的结果解释与设计讨论稿。
> 性质：记录已经发生的实验事实、当前最可信的根因判断和下一轮候选设计；**不是新的实验protocol，也不代表用户已经批准重训或打开validation**。
> 数据边界：只使用本次formal训练期产物。2022 validation和2023+ final仍未读取。

## 0. 一页结论

本次Stage D formal比较了Transformer与matched random在相同预算下的公式搜索能力：两种方法各跑seed 101/102/103，每个run 8,000,000 attempts，主指标是配对seed的`curated_top50_mean_reward`。

最终结果明确：

- Transformer在三个配对seed上全部落后，赢0/3；
- Transformer的curated top50平均reward为0.4626%，matched random为0.5507%；
- Transformer平均落后8.81个基点，相对低约16.0%；
- 冻结验收要求是Transformer配对均差为正且至少赢2/3，因此formal主判据失败，matched random胜出。

但Transformer并非没有学习。它把语义有效率提高到99.65%，所有有效采样的平均reward达到0.3996%，显著高于random的32.00%和0.0885%。问题在于，它只探索出约29.5万个canonical唯一公式，而random探索出约743.2万个，相差约25倍；accepted唯一状态约19.9万对198.2万，相差约10倍。

因此当前最可信的根因不是“Transformer不会生成公式”，而是：

> 当前REINFORCE优化“下一次采样的期望reward”，formal考核“最终最好50个不同公式的质量”。模型通过反复生成熟悉的中高分公式成功提高了前者，却因探索范围快速收缩而输掉后者。

下一轮若继续研究Transformer，优先级最高的不是增大模型、batch或attempts，而是：

1. 让训练反馈知道历史候选库和重复状态；
2. 将reward从单公式绝对分数转向高质量新候选对archive的增量价值；
3. 保留模型无法关闭的随机探索底线；
4. 让训练更关注高分尾部，而非全部样本平均值；
5. 同时监控attempt、唯一公式数、top50前沿和熵，不能再只看平均reward。

## 1. 文档目的

这份文档解决三个后续协作问题。

第一，防止把训练结果误读为“Transformer完全失败”或“random公式已经可以交易”。本次只回答训练期搜索比较，不回答样本外投资价值。

第二，防止后续session直接从一个表面参数开始调，例如单独提高entropy、增大模型或再跑800万次。当前问题首先是目标函数和反馈闭环错位，不是单一超参数不够大。

第三，明确参考项目中哪些设计真正相关。AlphaGPT原版没有解决重复探索问题；AlphaGen和其附带的DSO基线提供了archive、相关性过滤、PPO和risk-seeking训练等灵感，但它们的研究目标与本项目不同，不能直接照搬。

## 2. Formal实验身份与数据边界

### 2.1 冻结身份

```text
protocol_id       02cca48d1c8536f90a23a0e0361cb45e64a30e95623afc05ef58bd57a0dca5c1
code_commit       0df9404331b43bc8775d7ff17dab1d3172ecf186
code_fingerprint  f291520bf15d4d7f875e608e381386ac8c982ecc9b6e72a0826b345dc3d4f0ed
research_spec_id  1ab171f45467c6e8273a925a771988583c3ccecd640d6e240248dca436d7db07
train_view_id     v3a-train-view-8e48ba4ce6c2af91
binding_id        f545b622f5d8887419325cc0551483b6635e35998830e4df50c7cdb0d576da12
```

### 2.2 训练范围

- protocol训练区间：2016-08-09至2021-12-31；
- scorer实际包含1,303个可评分决策日；
- 预测/持有视界：未来10个交易日；
- 每个决策日从可用ETF中选信号最高的2至4只；
- reward：所选ETF未来10日平均收益减去当日可选ETF整体平均收益，再对训练决策日取均值；
- reward不能直接年化，未扣交易成本，也不是完整组合净收益。

早期人工分析简报曾用“2017-2021”作概括；正式边界一律以protocol中的`2016-08-09`至`2021-12-31`为准。旧分析artifact已有SHA留痕，不做事后覆盖。

### 2.3 六个run

```text
Transformer:    seeds 101 / 102 / 103，各8,000,000 attempts
Matched random: seeds 101 / 102 / 103，各8,000,000 attempts
Batch size:     8192
```

六个run全部完成，`resume_count=0`，completion marker、summary、retained candidates、checkpoint和ledger SHA均闭合。六run纯训练合计约110.9分钟，905机已由watchman关机。

### 2.4 本地证据

```text
.pi/profile/results/v3a-formal-runs-20260715/
  formal-multicpu-closeout-20260715T111422Z.tar.gz
  artifact_metadata.json
  runs/*/training_summary.json
  runs/*/retained_candidates.json
  analysis/formal_training_comparison.json
  analysis/formal_training_comparison.md
```

收尾包SHA-256：

```text
7409c635f7661a25938da123174d7b7a8bb04cd412c1899f5984b8ee2e8703ef
```

本地closeout包含已验证的summary、retained candidates、日志和大小/SHA元数据，不包含6份完整872MB ledger或完整checkpoint；完整产物仍保存在已关机的905机数据盘。

## 3. Formal结果

### 3.1 冻结主指标

主指标为`curated_top50_mean_reward`。curated过程只使用冻结的展示清理规则和`5e-6`近似并列容差，不改变原始公式语义和reward。

| Seed | Transformer top50均值 | Matched random top50均值 | T-R（基点） | 胜者 |
|---:|---:|---:|---:|---|
| 101 | 0.4436% | 0.5472% | -10.35 | Matched random |
| 102 | 0.4664% | 0.5540% | -8.76 | Matched random |
| 103 | 0.4777% | 0.5510% | -7.33 | Matched random |
| **均值** | **0.4626%** | **0.5507%** | **-8.81** | **Matched random** |

冻结验收要求：Transformer配对均差为正，且至少赢2/3个seed。实际为均差-8.81基点、赢0/3，因此formal判据明确失败。

这条结论的准确表述是：

> 在当前模型、reward、训练规则和每run 800万attempt预算下，Transformer没有提升训练期去重后top50公式的发现质量。

不应扩大为：

- Transformer作为一种方法永远不适合公式搜索；
- matched random找到了可交易策略；
- random的训练期top50一定会在样本外继续领先。

### 3.2 为什么“平均采样更好”与“top50更差”可以同时成立

| 指标（三seed均值） | Transformer | Matched random |
|---|---:|---:|
| 语义有效率 | 99.65% | 32.00% |
| 所有语义有效采样的平均reward | 0.3996% | 0.0885% |
| canonical唯一公式数 | 294,930 | 7,432,094 |
| canonical唯一率 | 3.69% | 92.90% |
| accepted唯一状态数 | 198,565 | 1,981,929 |
| 单run吞吐 | 9,934 attempts/s | 5,661 attempts/s |

“所有有效采样平均reward”按出现次数加权。同一条高分公式出现10万次，会进入平均值10万次。因此Transformer的0.3996%说明它成功把概率集中到了一个中高分区域，不说明它发现了大量不同的高分公式。

top50则先看不同候选，再从中取极端右尾。Random虽然大量公式低分或无效，但不同公式数量约为Transformer的25倍，因此有更多机会碰到稀有高分区域。

### 3.3 公式主题

Matched random三个seed共150条curated top50中：

- 150/150都包含`GAP`、`MEAN`和`WIN_20` token；
- 130/150的整理后文本明确包含`MEAN(GAP,20)`、`MEAN(NEG(GAP),20)`或`NEG(MEAN(GAP,20))`；
- 说明训练期高分尾部稳定集中于20日跳空均值及其正反向组合。

最值得保留作解释锚点的短公式是：

```text
NEG(MEAN(GAP,20))
```

它的训练reward约0.5346%，在三个random run分别排第61、78、63。Transformer seed 102/103也找到它并排第3，seed 101没有找到等效版本。

这说明Transformer具备表达和发现关键短公式的能力；问题更像早期搜索路径和策略收缩，而不是模型完全无法表示这一信号。

同时也要谨慎：复杂random top50可能只是围绕这个简单核心叠加训练期偶然项。只有validation能判断复杂结构是否真的带来可泛化增益。

### 3.4 工程观察

- Transformer吞吐约比random高75.5%；
- Transformer候选快照远小于random，原因正是唯一公式少；
- Transformer CUDA reserved峰值平均约93.9GiB，random约62.3GiB；
- 六run没有OOM，但正式长跑显存余量比阶段5短门禁观察更小；
- 后续不能在没有新门禁的情况下扩大batch、VM内存比例或并发run。

## 4. 第一性根因：训练目标与研究目标不是同一个数学问题

### 4.1 当前实际优化目标

当前REINFORCE近似优化：

```text
maximize E[f ~ pθ][R(f)]
```

其中`pθ`是Transformer生成公式的概率分布，`R(f)`是单条公式的训练reward。

这回答的是：

> 下一次从当前策略采一条公式，它平均有多好？

### 4.2 Formal实际考核目标

Formal考核接近：

```text
maximize Mean(Top50(Unique(f1, f2, ..., fN)))
```

这回答的是：

> 在固定预算内，最终发现的最好50个不同公式有多好？

期望值高不保证极端尾部高。一个分布可以把全部概率放在“稳定80分”的小区域，从而拥有很高的平均值；另一个分布大多数为0分，却因覆盖范围极广而出现足够多的95分和100分样本。前者赢平均，后者赢top50。

### 4.3 当前feedback不知道“以前发生过什么”

当前模型更新只看到本batch的：

```text
token序列 + 单公式reward + log probability + entropy
```

它看不到：

- 这条canonical公式历史上出现过多少次；
- 它是否与已有公式产生完全相同的选股结果；
- 它是否进入当前top50 archive；
- 它是否改善了top50最低分、平均分或结构多样性；
- 它是否只是已有信号的复杂等价改写。

更关键的是，当前流水线在GPU评分后立即计算REINFORCE并执行`optimizer.step()`，之后CPU worker才canonicalize，单线程提交器才知道公式是否重复。因此去重事实到达得太晚，无法影响产生该公式的梯度。

### 4.4 正反馈导致策略收缩

训练初期某个formula family偶然高于batch平均后：

```text
提高该token路径概率
→ 下一批出现更多同类公式
→ 同类公式贡献更多正梯度
→ 概率继续提高
→ 其他区域越来越难被采到
```

重复公式仍按出现次数参与loss，相当于同一个观点重复一千次就获得一千份投票权。

这解释了本次最反常但最有信息量的组合：

```text
语义有效率接近100%
平均采样reward显著提高
canonical唯一率却只有约3.7%
top50尾部输给random
```

### 4.5 大batch和更多attempts为什么不自动增加视野

batch size只决定一次从**当前分布**抽多少次，不决定分布覆盖多广。

一旦策略收缩：

```text
小batch = 一次抽少量相似公式
大batch = 一次抽8192条相似公式
```

大batch降低了梯度噪声，可能让模型更稳定、更快地走向当前局部最优。800万attempts对应约977次参数更新；若前期已经收缩，后面数百万attempts只是在高吞吐重复已知区域。

### 4.6 极值统计与金融噪声

Random从约743万个不同公式中选50个，Transformer从约29.5万个不同公式中选50个。前者能筛选更极端的分位数，因此训练期top50自然占有“独立抽样次数”优势。

但金融reward含有大量噪声。Random探索越广，也越容易找到“真实规律 + 恰好适配训练期噪声”的极端公式。因此random训练期top50更高可能同时包含：

1. 它覆盖了Transformer没有到达的真实高质量区域；
2. 它进行了更多独立检验，选中了更极端的训练期过拟合。

现有训练期结果无法分开两者。2022 validation的意义就在这里。

## 5. 证据强度分级

### 5.1 已由本次formal直接证明

- 当前Transformer的curated top50在3/3配对seed上低于matched random；
- Transformer有效采样平均reward显著高于random；
- Transformer canonical唯一数和accepted唯一数显著低于random；
- 当前配置在固定attempt预算下发生了严重的探索覆盖不足；
- 当前结果不是单个seed偶然翻转。

### 5.2 由代码机制和数据共同强支持

- 单公式期望reward与top50 archive目标错位；
- 重复公式在梯度中被重复计权；
- canonical/selection重复信息到达optimizer之后，无法形成新颖性反馈；
- 正反馈造成策略集中，是唯一公式数量崩塌的主要解释。

### 5.3 尚未被因果实验确认

- `entropy_coefficient=0.005`是否“太低”；
- d_model 64、2层Transformer是否容量不足；
- 收缩具体发生在第几个batch；
- 提高entropy是否足以恢复top50；
- random的top50优势有多少来自真实结构、有多少来自过拟合；
- 同样唯一公式预算下，Transformer的分布是否优于random。

这些问题需要曲线诊断或新消融实验，不能从六个最终summary中直接断言。

## 6. 两个参考项目真正提供了什么

### 6.1 AlphaGPT：原始结构并没有解决重复探索

AlphaGPT原版同样使用batch 8192、约1000个训练step。它对batch reward做标准化，然后让每次出现的公式直接贡献policy gradient；没有历史archive、canonical去重反馈或重复惩罚：

- [AlphaGPT engine.py：采样、reward标准化与更新](https://github.com/imbue-bit/AlphaGPT/blob/d851f2221dcaf4d53a707344f68ae6801e3e5af5/model_core/engine.py#L65-L120)
- [AlphaGPT config.py：batch 8192与1000 step](https://github.com/imbue-bit/AlphaGPT/blob/d851f2221dcaf4d53a707344f68ae6801e3e5af5/model_core/config.py#L5-L13)

AlphaGPT模型定义了critic head，但engine采样时把value输出丢弃，训练loss没有使用critic：

- [AlphaGPT alphagpt.py：critic定义与返回](https://github.com/imbue-bit/AlphaGPT/blob/d851f2221dcaf4d53a707344f68ae6801e3e5af5/model_core/alphagpt.py#L248-L271)

因此本次结果并不是“我们偏离AlphaGPT才失败”，反而更像用严格random对照暴露了原始骨架中长期未被检验的目标缺口。AlphaGPT的LoRD、RMSNorm、looped Transformer等模型技巧不直接解决archive新颖性问题。

### 6.2 AlphaGen：有状态alpha pool是最重要的启发

AlphaGen的环境在公式结束时调用`pool.try_new_expr(expr)`，reward来自候选加入后的pool状态，而不是一个与历史无关的纯单公式分数：

- [AlphaGen env/core.py：公式完成后交给pool评价](https://github.com/ICT-FinD-Lab/AlphaGen/blob/259687e8f316994426416c530a94842a2fe6405e/alphagen/rl/env/core.py#L48-L74)

其线性alpha pool具有固定容量，会尝试加入新公式、重新优化权重，并在超容量时移除最弱项：

- [AlphaGen linear_alpha_pool.py：候选加入和容量淘汰](https://github.com/ICT-FinD-Lab/AlphaGen/blob/259687e8f316994426416c530a94842a2fe6405e/alphagen/models/linear_alpha_pool.py#L61-L104)

它还计算候选与池内公式的mutual IC，并拒绝相关性过高的候选：

- [AlphaGen linear_alpha_pool.py：mutual IC过滤](https://github.com/ICT-FinD-Lab/AlphaGen/blob/259687e8f316994426416c530a94842a2fe6405e/alphagen/models/linear_alpha_pool.py#L173-L189)

训练器使用Maskable PPO、LSTM特征网络和entropy coefficient 0.01：

- [AlphaGen scripts/rl.py：PPO配置](https://github.com/ICT-FinD-Lab/AlphaGen/blob/259687e8f316994426416c530a94842a2fe6405e/scripts/rl.py#L252-L274)

真正值得借鉴的不是“PPO一定更好”，而是：

> 生成器被放进一个有记忆、有容量、有淘汰规则的候选池环境，候选价值取决于已有池子。

但AlphaGen优化的是互补的小型alpha组合，而本次formal优化的是单公式top50。若直接改成AlphaGen pool objective，就会改变研究问题，不能静默照搬。

### 6.3 AlphaGen附带的DSO：risk-seeking训练与top-N更接近

AlphaGen仓库附带的Deep Symbolic Optimization基线使用risk-seeking policy gradient。默认`epsilon=0.05`，即按每批reward的95%分位数过滤，只用最高约5%的表达式训练：

- [DSO配置：epsilon 0.05和分位数baseline](https://github.com/ICT-FinD-Lab/AlphaGen/blob/259687e8f316994426416c530a94842a2fe6405e/dso/config/config_common.json#L24-L38)
- [DSO训练：计算分位数、过滤低reward样本](https://github.com/ICT-FinD-Lab/AlphaGen/blob/259687e8f316994426416c530a94842a2fe6405e/dso/train.py#L321-L405)

它还提供不接收重复项的priority queue，只保留固定容量内的最高分程序：

- [DSO memory.py：UniquePriorityQueue](https://github.com/ICT-FinD-Lab/AlphaGen/blob/259687e8f316994426416c530a94842a2fe6405e/dso/memory.py#L221-L251)

这比全样本期望reward更接近本项目的top50目标。但risk-seeking本身仍可能围绕少量elite塌缩，因此必须与去重反馈和不可关闭的探索通道结合。

## 7. 开始优化前必须先决定研究目标

存在两个都合理、但不能混在一起的目标。

### 7.1 目标A：发现一批单独高分的公式

```text
目标：curated top30/top50单公式库
评价：各公式独立reward、复杂度、稳定性
用途：后续再做筛选或组合
```

这是本次formal已经冻结的问题。若继续沿这条线，archive质量可定义为top50平均分、最低分、长度分桶和新候选是否替换现有尾部。

### 7.2 目标B：直接发现互补的小型公式组合

```text
目标：一个容量固定、信号互补的alpha pool
评价：组合后的IC/收益及候选边际贡献
用途：直接形成组合信号
```

这更接近AlphaGen。它可能更贴近最终组合价值，但会把研究问题从“搜索器能否发现好单公式”改成“生成器能否维护好组合”。若要转向，必须新建研究路线和对照，不能把它当成本次formal的修补参数。

当前讨论的保守建议是：下一轮先保持目标A不变，只借鉴archive记忆、risk-seeking和探索机制。是否最终转向目标B，另行决策。

## 8. 下一轮可从哪些方向优化

### 8.1 方向一：archive-aware reward

令`A`表示当前已知的高质量候选库，训练reward不再只使用`R(f)`，而使用类似：

```text
R_train(f, A)
  = quality_component(f)
  + archive_improvement(f | A)
  + gated_novelty(f | A)
  - duplicate_cost(f | A)
```

其中：

- `quality_component`提供稠密学习信号，避免archive reward过于稀疏；
- `archive_improvement`奖励进入top50、替换尾部或提高分桶前沿；
- `gated_novelty`只在质量达到门槛后奖励新颖性；
- `duplicate_cost`让重复公式不再获得与首次出现相同的梯度权重。

不能简单采用“新公式一律加分”。否则最优策略会变成生成大量独特但无意义的公式。

### 8.2 方向二：让高分尾部而不是全体均值主导训练

可借鉴risk-seeking思路：

- 每批只让高于某个动态分位数的**唯一公式**贡献主要正梯度；
- baseline使用分位数或archive门槛，而不是batch均值；
- 低质量但新颖的公式不获得大奖励；
- elite内部仍按archive增量和重复情况加权。

直接只优化“是否进入全局top50”会非常稀疏和非平稳。较稳妥的设计是：批内高分分位数提供稠密方向，archive improvement负责与最终目标对齐。

### 8.3 方向三：保留模型无法关闭的探索底线

不能只依赖entropy。Entropy约束局部token概率，不保证全局canonical或信号新颖性。

候选方式包括：

```text
常温Transformer采样
+ 高温Transformer采样
+ 固定比例纯随机采样
```

随机比例应作为新protocol参数预先冻结。讨论阶段可把25%至50%视为候选区间，但当前没有证据支持某个具体值，不能直接写死。

还可以根据滚动canonical唯一率和archive增长自动提高温度或随机比例，但自适应规则必须事先确定，不能在训练中人工看结果调参。

### 8.4 方向四：三层重复反馈

应区分：

1. exact token重复：token序列完全相同；
2. canonical重复：交换律或安全等价清理后相同；
3. selection/signal重复：语法不同，但选出的ETF或信号排序高度相似。

前两层较便宜，可进入训练热路径。第三层最有研究意义但较贵，优先使用现有selection fingerprint做近似，信号相关性可周期性或后处理计算，不必恢复逐公式重型校验。

同一canonical在一个batch出现多次时，不能让它按次数无限放大梯度。可采用：

- 每个canonical每次更新最多一票；
- 按出现次数做`1/count`降权；
- 同类中只保留最有archive增量的一条。

### 8.5 方向五：调整反馈时序

当前optimizer在CPU canonicalize之前更新。若要使用canonical/selection新颖性，必须改变反馈闭环。

较清晰的候选方案：

```text
GPU生成与评分
→ CPU canonicalize / selection fingerprint / archive判定
→ 得到最终训练权重
→ 用保存的token重新前向计算log-prob
→ optimizer step
```

这样不必跨CPU等待长期保留GPU autograd graph，代价是每batch增加一次较小的Transformer前向。该代价是否可接受应先做工程smoke，不应直接假设。

只在GPU按exact token去重是更小的改动，但不能解决canonical和信号等价，只能作为第一层减损。

### 8.6 方向六：用多个预算口径看算法，而不是只看attempt

未来比较至少同时报告：

```text
固定attempt：衡量端到端搜索器在相同调用预算下的效果
固定canonical唯一数：衡量生成分布本身的质量
固定wall-clock：衡量实际工程效率
```

本次formal固定attempt是合理的，因为重复本身就是算法效率的一部分。但要定位根因，必须增加固定唯一数比较。

### 8.7 方向七：训练reward的稳健性

即使修复探索，直接追逐训练期极值仍可能放大多重检验。未来可讨论只在训练区间内部使用预先冻结的稳健度，例如：

- 不同训练子时期的一致性；
- 平均reward减去一定的不稳定性惩罚；
- 简单公式与复杂公式的透明复杂度约束；
- top50中不同长度或不同信号主题的最低覆盖。

但这些会改变训练研究问题，必须新建protocol，且不能利用2022 validation反向调权重。

### 8.8 模型大小、学习率、entropy和batch属于第二层

这些参数可能影响收缩速度，但没有解决目标错位：

- 更大模型可能表达更多模式，也可能更快记住局部高分模式；
- 更高entropy可能增加token随机性，但不保证canonical新颖；
- 更大batch可能让错误目标优化得更稳定；
- 更多attempts在重复率不变时只是扩大浪费。

因此不应把“换大Transformer”作为第一项实验。

## 9. 不建议的单点修补

### 9.1 只提高entropy coefficient

可能延缓概率极化，但无法告诉模型哪些公式已经出现过，也无法保证信号层多样性。

### 9.2 只增加训练次数

如果新增唯一公式率已经接近零，更多attempts只会增加重复。

### 9.3 只扩大batch

大batch不等于大视野；它只会更准确地估计当前狭窄分布的梯度。

### 9.4 所有新公式都奖励

会把目标从“好公式”改成“奇怪但不同的公式”。新颖性必须有质量门槛。

### 9.5 直接照搬AlphaGen

AlphaGen优化组合pool，本项目当前考单公式top50。直接复制会改变研究问题和公平对照。

### 9.6 用validation调探索参数

一旦根据2022表现调entropy、随机比例或archive reward，2022就变成训练信息，不能再作为干净validation。

## 10. 新训练前最值得做的现有数据诊断

以下诊断不需要产生新的研究训练，但部分需要重新启动905机只读拉回完整ledger和`training_log.jsonl`；机器当前已关机，任何开机动作仍需用户批准。

### 10.1 固定唯一公式数量比较

把matched random按attempt顺序截断到与Transformer相同的canonical唯一数，例如约29.5万，再比较两边top50。

解释：

- 若相同唯一数下Transformer更好，说明模型确有“把搜索引向高质量区域”的价值，主要失败来自后期重复；
- 若相同唯一数下仍输，说明模型学习到的分布本身也偏离高分区域。

这是区分“覆盖不足”和“搜索方向错误”最关键的诊断。

### 10.2 随attempt变化的四条曲线

```text
累计canonical唯一公式数
每10万attempt新增唯一公式数
累计raw/curated top50前沿
平均reward、normalized entropy与token集中度
```

它们可以回答策略何时收缩、top50何时停止改善，以及平均reward上涨是否与探索崩塌同步。

### 10.3 重复梯度放大量

按batch统计：

- exact重复数量；
- canonical重复数量；
- 每个canonical对loss的总权重；
- 去重后重新计算的有效batch size。

这能估算“重复样本多次投票”究竟贡献了多少收缩压力。

### 10.4 Random的极值优势分解

比较：

- 固定attempt；
- 固定语义有效样本数；
- 固定canonical唯一数；
- 固定accepted唯一数。

若random优势主要随唯一数增加而扩大，说明极值统计是主因；若很早就领先，说明其先验分布覆盖了Transformer容易错过的重要区域。

## 11. 候选的下一轮最小实验结构

以下只是讨论框架，不是已批准方案，也没有冻结参数。

### 11.1 Gate 0：先完成现有数据诊断

在不知道收缩曲线和固定唯一数结果前，不进入新800万attempt formal。

### 11.2 Gate 1：小预算机制实验

候选三臂：

```text
A：当前Transformer，原样机制对照
B：risk-seeking高分分位数训练 + 固定随机探索
C：archive-aware reward + canonical降权 + 固定随机探索
```

Matched random继续作为外部基线。研究训练仍只使用原train view，在远端GPU执行。

首轮只需足以观察机制的较小预算，例如100万至200万attempt候选范围，而不是立即跑满800万。具体预算、seed数和探索比例必须在新protocol中预先决定。

### 11.3 机制门槛

下一轮不能只看最终best reward，还应预先冻结：

- canonical唯一率是否显著高于当前约3.7%；
- top50前沿是否在后半程继续改善；
- 平均reward提高是否仍伴随新增唯一数接近零；
- 同一canonical的batch梯度权重是否受控；
- 相同唯一数下是否优于random；
- 固定attempt和固定wall-clock下是否仍有实际价值。

只有机制门槛通过，才值得再做完整3-seed formal。

## 12. Validation时点是当前最重要的开放决策

此前交接提出“先冻结当前两边top50，再打开2022 validation”。这仍是一条合法路径，但现在出现了新的取舍。

### 路径A：先完成当前研究闭环

```text
冻结当前Transformer/random公式库
→ 预注册validation判据
→ 打开2022一次
→ 判断random高分是否泛化
```

优点：尽快回答当前候选是否有样本外价值。
代价：2022从此已被看过，后续改进Transformer时不能再把它当完全干净的validation。

### 路径B：先保留validation，做训练期机制修复

```text
2022继续封存
→ 只用训练期ledger做诊断
→ 预注册一轮小预算机制实验
→ 最终冻结旧/新方法候选
→ 再统一打开2022一次
```

优点：保留2022对新旧搜索器的统一干净门禁。
代价：延后知道当前random top50是否只是训练期过拟合。

如果用户的当前优先目标是“把Transformer搜索机制研究明白”，更倾向路径B；如果目标是“尽快判断现有公式有没有样本外价值”，更倾向路径A。该选择会改变后续研究价值，必须由用户明确决定。

无论选择哪条路径，2023+ final继续封存。

## 13. 已冻结结论与尚未决定事项

### 13.1 已冻结，不能被下一轮改写

- 本次formal身份和六run产物；
- matched random在当前冻结主指标上3/3胜出；
- Transformer当前配置未通过top50搜索比较；
- 本次训练没有读取validation/final；
- 新设计必须建立新protocol、ResearchSpec和binding；
- 不能把下一轮改进结果回填成“本次formal其实通过”。

### 13.2 当前强判断，但仍可被诊断修订

- 目标错位和重复反馈缺失是主要根因；
- Transformer具有局部质量引导能力，但因覆盖不足没有转化为top50优势；
- archive-aware reward、risk-seeking和强制探索比扩大模型更值得优先研究。

### 13.3 尚未决定

- 先validation还是先训练期机制修复；
- 目标保持单公式top50还是转向互补pool；
- random探索比例；
- risk-seeking分位数；
- duplicate penalty与archive gain的具体形式；
- 是否使用固定或自适应temperature/entropy；
- 新小实验的attempt预算和seed数；
- 是否需要信号相关性进入训练热路径。

## 14. 后续session接手顺序

1. 先读本文和`.pi/handoff/2026-07-15-Stage_D_formal训练完成与结果.md`。
2. 不连接远端、不打开2022前，先确认用户选择路径A还是路径B。
3. 若选择训练期机制修复，先提出只读拉回完整ledger/log的开机计划和预计成本，等用户批准。
4. 用固定唯一数比较和累计曲线验证根因，不直接改代码。
5. 根因诊断后再讨论新protocol；任何参数仍需预注册。
6. 若选择validation，先冻结正式公式库artifact和validation判据，再一次性打开2022。
7. 2023+ final始终封存，直到整个方法和validation选择全部完成。

## 15. 相关文件

- Formal protocol：`configs/v3a_stage_d_formal_topn.json`
- 阶段5验收：`docs/V3A_StageD_阶段5远端工程验收.md`
- 新训练框架：`docs/V3A_StageD_新训练框架与当前进度.md`
- 训练期人工报告：`.pi/profile/results/v3a-formal-runs-20260715/analysis/formal_training_comparison.md`
- 机器可读配对和curation审计：`.pi/profile/results/v3a-formal-runs-20260715/analysis/formal_training_comparison.json`
- Formal完成交接：`.pi/handoff/2026-07-15-Stage_D_formal训练完成与结果.md`