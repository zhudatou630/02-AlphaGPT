# V3A Stage D Validation 候选冻结结果

> 时间：2026-07-18
> 数据边界：只使用`2016-08-09..2021-12-31`训练期；未读取2022 validation或2023+ final。
> 协议：`30d07d673ab9a7b787f0d44f58249ab48666bfa21cab6a7911a82eb4b62aa675`。
> 状态：候选与CPU/CUDA数值门完成；`validation_run_approved=false`。

## 1. 直接结论

启动905前的本机准备、905上的RTX PRO 6000 CUDA审计，以及拉回后的确定性重建均已完成。905在
产物拉回并通过SHA校验后已经关机。

最终候选三组均非空，Transformer每个seed的原始高分组都保持50条，且没有Transformer父公式因
CPU/CUDA门被剔除。数值差异主要发生在短subtree；它改变了少量Random简化锚点，但没有改变
Transformer三组数量。

## 2. 最终候选数量

| 方法 | Seed | 原始高分组 | 稳健复杂组 | 对应简化组 | 配对数 |
|---|---:|---:|---:|---:|---:|
| Transformer | 101 | 50 | 10 | 2 | 10 |
| Transformer | 102 | 50 | 39 | 5 | 39 |
| Transformer | 103 | 50 | 26 | 1 | 26 |
| Random | 101 | 50 | 18 | 11 | 18 |
| Random | 102 | 50 | 24 | 20 | 24 |
| Random | 103 | 50 | 20 | 19 | 20 |

`paired_simple`按seed做canonical去重，但配对检验按复杂公式逐条保留。因此Transformer seed 103虽只有
1条唯一简化公式，仍保留26条复杂-简化配对；这反映多个复杂公式共享同一个短核心，不是丢失配对。

## 3. 数值门结果

CUDA registry共2,162个exact token序列，其中1,936个通过完整数值门：

- 204个subtree在CPU和CUDA两端都不通过冻结质量门；
- 22个序列两端质量都有效，但CPU/GPU选择不一致；
- 上述22个中，20个同时出现full-train reward差超过`1e-6`，2个只出现top-k序列差异；
- 最大CPU/GPU full reward绝对差为`8.882791735231876e-5`；
- 父公式储备只有Random seed 101的第71和76名未通过门，均不属于最终top50；
- Transformer三个seed的父公式储备均通过数值门。

相对CPU provisional组，真实CUDA门只造成以下组级变化：

- Random seed 101：稳健复杂组`19 -> 18`，对应简化组`12 -> 11`；
- Random seed 102：稳健复杂组不变，对应简化组`21 -> 20`；
- 其余组数量不变。

## 4. `robust_z=1.5`训练期sanity

用最终候选组重跑后：

- 所有组的MAD退化日占比中位数均为0；
- Transformer各组平均持仓数中位数约为`2.33..2.46`；
- Transformer各组空仓占比中位数约为`0.62%..1.96%`；
- Transformer各组满仓占比中位数约为`51.33%..60.36%`；
- 无合格买入对象的日期占比中位数约为`15.38%..18.92%`。

因此`robust_z=1.5`没有退化成长期全空或长期满仓。该检查只观察资格和不含止损的3进5出粘性持仓，
不计算训练收益；训练view没有close，-7%止损由合成路径测试覆盖。

## 5. 工程验证

- 合成validator覆盖初始建仓、局部现金池、次日排名退出、停牌卖单保留、-7%止损和失去资格后重入；
- validation默认信号为CPU float32；
- 11项相关单元测试通过；
- CPU准备、CUDA产物、最终候选和最终sanity的SHA集合全部通过；
- 905 CUDA manifest明确记录`train_view_end=2021-12-31`和
  `validation_or_final_metrics_read=false`。

## 6. 权威产物

```text
.pi/profile/results/v3a-validation-prep-20260718/
.pi/profile/results/v3a-validation-cuda-20260718/
.pi/profile/results/v3a-validation-final-candidates-20260718/
.pi/profile/results/v3a-validation-final-sanity-20260718/
```

最终组清单SHA-256：

```text
12aa88da48942c11e23f02ec93d83a411e0e61e3cdcdcafaf9dc0a1ffa9f6860
```

## 7. 下一停点

当前仍不得读取2022。下一步只生成并展示预运行binding；用户再次明确授权后，才允许构建2022
validation view并一次性运行全部Transformer与Random候选。2023+继续封存。