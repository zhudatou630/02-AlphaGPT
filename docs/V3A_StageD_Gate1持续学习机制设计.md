# V3A Stage D Gate 1 持续学习机制设计

> 状态：2026-07-16经用户逐项确认，进入实现。
> 性质：训练期机制实验设计，不是新formal，也不授权读取2022 validation或2023+ final。
> 研究问题：Transformer自身能否在持续发现新公式的同时，让新公式的高质量右尾和自身top50随训练推进而改善。

## 1. 为什么要改

Stage D formal优化的是按出现次数加权的平均reward，最终比较的是固定attempt预算下最好50个不同公式。完整ledger证明Transformer在40万至50万attempts已发现最终canonical唯一数的95%，最后100万仅新增2/1/4个canonical；后续大量计算用于重复熟悉公式。

Gate 1不扩大模型，不改公式语言、scorer或训练数据。它只把学习反馈改成：

```text
奖励高质量的新发现
+ 奖励对Transformer历史top50的真实改善
- 惩罚重复或无效的浪费attempt
```

## 2. 冻结边界

保持不变：

- 训练区间`2016-08-09`至`2021-12-31`；
- 35只ETF train view、公式词表、最大长度、合法动作mask；
- VM、scorer、coverage/constant质量门禁和单公式reward；
- Transformer尺寸`64 / 2层 / 4头 / ff128`；
- AdamW、`lr=1e-3`、`weight_decay=1e-5`、gradient clip `1.0`；
- batch总attempt数8192、seed 101/102/103；
- 单公式canonical top50研究目标。

明确改变：

- 新增机制实验protocol mode，不回写formal身份；
- 每批75% Transformer、25%合法uniform random；
- entropy coefficient从`0.005`改为`0`；
- optimizer前完成Transformer历史canonical判断；
- 旧REINFORCE batch-zscore改为三组等权目标；
- 采样不保留autograd graph，标签完成后整序列teacher-forcing一次重算。

formal protocol实际已经使用`dropout=0.0`，Gate 1继续保持，不构成参数变化。

## 3. 每批数据归属

完整batch固定8192条，正常完整批次为：

```text
Transformer lane  6144
Random lane       2048
```

最后不足完整batch时仍按预先冻结的25%比例确定整数数量，不允许运行中自适应。attempt ledger使用固定顺序：每批Transformer lane在前、Random lane在后；因此无需改变109字节ledger schema即可从`attempt_index`和batch边界恢复来源。

Random lane：

- 使用相同合法动作、VM、scorer和质量门禁；
- 不进入Transformer seen集合；
- 不进入Transformer训练archive；
- 不参与模型loss；
- 进入Random独立候选统计和全局合并候选库。

Transformer lane：

- 参与模型历史seen判断、训练archive和loss；
- 单独报告新公式率、右尾质量和top50前沿；
- 与Random及合并库的结果分开，不把Random贡献解释成模型学习。

## 4. Transformer历史新颖性

batch开始时冻结Transformer历史canonical seen集合`S`。对本批Transformer lane按canonical分组：

1. canonical已在`S`中：全部进入浪费attempt组；
2. canonical不在`S`中：选择一个代表；同批其他副本进入浪费attempt组；
3. 同组存在语义有效且reward有限的记录时，代表必须从这些有效记录中选择：reward更高优先，reward相同则token更短，仍相同则attempt更早；全组均无效时取attempt更早的记录作为无效代表；
4. 代表语义无效：进入浪费attempt组；
5. 代表语义有效：成为本批“有效新公式”；
6. batch反馈计算后，将本批所有Transformer canonical加入`S`，包括首次出现但语义无效的canonical。

Random先发现某个canonical不会让它对Transformer变成历史重复；两个lane维护独立研究含义。

## 5. 三组等权loss

令`lp_i`为第`i`条Transformer公式整条序列的teacher-forced log-prob总和，不按公式长度归一化。

### 5.1 批内右尾组

只在本批有效新公式中按以下稳定键排序：

```text
reward降序 → token长度升序 → canonical hash升序
```

取前`ceil(10% × N)`条；`N>0`时至少取1条。组内统一权重：

```text
L_elite = -mean(lp_i)
```

其他有效新公式权重为零，不因暂时低分受到负反馈。

### 5.2 Archive改善组

训练archive`A`只保存此前Transformer有效新公式中reward最高的50个canonical。batch开始时冻结旧archive：

- `|A| < 50`：本批无archive改善loss；
- `|A| = 50`：旧第50名reward为门槛`r_floor`；
- 本批有效新公式若`r_i > r_floor`，则改善幅度`delta_i = r_i - r_floor`；
- 组内权重`w_i = delta_i / sum(delta)`；
- `L_archive = -sum(w_i × lp_i)`。

一个公式可同时属于右尾组和archive改善组，从而获得两份正反馈。整批计算完loss标签后，才用本批有效新公式统一更新archive；不按数组顺序逐条抬高门槛。

### 5.3 浪费attempt组

包含：

- Transformer历史canonical重复；
- 本批同canonical非代表副本；
- 首次canonical代表但语义无效。

普通但有效的新公式不在此组。浪费组为：

```text
L_waste = +mean(lp_i)
```

最小化该项会降低浪费公式的生成概率。组整体只占一个固定权重，不随浪费样本总数无限放大；组内高频公式因出现次数多而承担更大比例。

### 5.4 总loss

```text
L = present(L_elite) + present(L_archive) + present(L_waste)
```

三组分别归一化后等权。某组为空时该项为0，不动态补偿其他组。第一版不增加critic、entropy、低分新公式惩罚、重复次数分级或自动权重。

## 6. 计算顺序

目标调度：

```text
GPU: Transformer lane无梯度采样、VM和scorer
CPU: 立即开始Transformer canonical准备
GPU: CPU工作期间处理Random lane采样、VM和scorer
CPU: 完成Transformer历史seen、右尾和archive标签
GPU: 一次整序列teacher-forcing得到全部lp_i
GPU: 三组loss、backward、clip和optimizer.step
CPU: 提交完整ledger、全局候选库和日志
```

policy新增整序列logits接口；teacher-forcing根据权威grammar tables重建每个位置的合法动作mask，计算已生成动作在相同确定性策略下的log-prob。生成阶段使用`torch.no_grad()`，不跨CPU等待保留autograd graph。

不生成使用旧模型参数的下一批Transformer公式，不引入stale-policy流水线。Random ledger整理和不影响模型反馈的全局提交可继续使用有界异步路径，但同一checkpoint前必须排空。

## 7. 状态、恢复和产物

新训练状态必须随安全checkpoint恢复：

- Transformer seen canonical集合；
- Transformer独立top50 archive；
- model、optimizer和全部RNG；
- lane边界、step和attempt count；
- ledger、training log和候选快照绑定。

实现优先复用现有compact candidate snapshot/ledger replay能力，不建立数据库。第一版resume从已截断到checkpoint边界的ledger按固定lane顺序重建Transformer seen/archive；50-batch门同时记录恢复耗时，只有重放成为实际瓶颈才增加第二套状态snapshot。

`training_reward`字段保存该attempt最终参与新loss的有符号样本权重或零值；batch log至少新增：

```text
model/random lane count
model history duplicate / within-batch duplicate / semantic invalid count
model new valid count / elite count / archive improver count
L_elite / L_archive / L_waste / total loss
model new canonical rate
model new-valid reward mean / q90 / top10 mean
model archive top50 mean / floor
transformer GPU seconds / random overlap seconds
canonical label wait seconds / teacher-forcing seconds / backward seconds
```

runner最终summary明确标记标准候选索引为两lane合并库，并输出Transformer训练archive摘要；Gate 1只读分析从ledger按固定lane重建并分别输出Transformer、当前Random lane和合并候选库的raw/curated结果，不覆盖旧formal summary语义。

## 8. 本地与远端工程门

本地只做合成fixture，不做研究训练。最小覆盖：

- 历史重复、同批重复、无效、新普通、elite和archive improver分类；
- 三组loss符号、等权和空组；
- teacher-forcing log-prob与逐token确定性计算一致；
- Random lane不进入模型seen/archive/loss；
- 连续运行与checkpoint/resume的ledger、训练状态和模型参数一致；
- legacy formal路径在相同fixture下保持原行为。

905先运行50个完整batch，即409,600总attempts。通过条件：

```text
canonical label wait / batch total <= 25%
teacher-forcing / batch total <= 20%
throughput >= 3000 attempts/s
GPU显存余量 >= 5GiB
无非有限loss/gradient
产物边界与resume一致
```

未通过时只profile和优化计算调度，不改研究loss。工程门产物与机制研究目录隔离。

## 9. 机制实验

工程门通过后，新机制只跑三个run：

```text
seed:            101 / 102 / 103
attempts/run:    2,000,000 total
model lane/run:  约1,500,000
random lane/run: 约500,000
```

旧Transformer和matched random对照直接读取formal ledger前200万，不重跑。工程门后protocol和参数冻结；三个seed串行跑完，中途只因工程错误停止，不因研究表现修改参数。

## 10. 四项联合门槛

全部只看训练期。

### 10.1 探索保持

按总attempt分首段`0-500k`和末段`1.5m-2m`，只统计Transformer lane历史首次canonical：

```text
late_new_count / early_new_count >= 50%
```

要求三seed平均比例至少50%，且至少2/3 seed各自达到50%。

### 10.2 右尾提升

每段只看Transformer lane有效新公式，计算该段reward最高10%的平均值：

```text
late_top10_mean - early_top10_mean > 0
```

要求三seed配对平均差为正，且至少2/3 seed为正。中间两个50万段完整报告但不要求机械单调。

### 10.3 Archive持续改善

比较Transformer独立curated top50：

```text
top50_mean@2m - top50_mean@1m > 0
```

要求三seed配对平均差为正，且至少2/3 seed为正。

### 10.4 相同canonical唯一数搜索效率

以新机制Transformer lane截至2m的全部canonical唯一数为目标，将同seed旧matched-random ledger按attempt顺序截断到相同唯一数，比较curated top50：

```text
Transformer - Random > 0
```

要求三seed配对平均差为正，且至少赢2/3。

四项必须联合通过。固定attempt下与旧Transformer、Random及合并archive的结果均报告，但不替代联合门槛。

## 11. 明确不做

- 不打开2022 validation或2023+ final；
- 不增加模型层数、宽度、critic、PPO或模仿Random的loss；
- 不加入selection相似度、公式主题多样性或组合pool；
- 不用validation选择elite比例、Random比例或loss权重；
- 不做自适应Random比例、重复惩罚分级或自动性能降级；
- 不因单个seed中途结果修改已冻结方案；
- 机制门未通过前不设计新800万formal。

## 12. 实施顺序

```text
决策记录与本文档
→ policy整序列接口和teacher-forcing一致性
→ 训练反馈状态与三组loss
→ 75/25 runner调度、日志和checkpoint
→ 本地合成fixture
→ 新工程protocol/ResearchSpec/binding
→ 905 50-batch性能门
→ 人工核对工程结果和新机制身份
→ 3 × 2m机制实验
→ 只读联合门槛分析
```

相关决策记录：`.pi/grill/2026-07-16-0800——持续学习方案.md`。