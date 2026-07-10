# Phase1b ETF 事件复权数据治理

本阶段只生成独立数据产物，不替换现有 panel，也不改 scorer、validator 或训练入口。

## 问题

TDX 不复权行情本身记录的是交易所实际成交价格。ETF 份额拆分或合并后，每份价格会机械跳变，持有人份额同时反向变化。当前 TDX `ApplyQFQ` 只处理 `category=1` 的普通除权除息，没有处理 ETF 的 `category=11/12` 扩缩股事件，而且复权价会舍入到两位小数，不适合量化计算。

## 新口径

- `category=1`：按分红、配股、送转字段计算理论除权价。
- `category=11/12`：把 `c3` 作为新份额/旧份额比例。
- 价格：使用乘法事件调整，锚定最新份额口径，保持普通交易日收益率不变。
- flow 原始值：同时保留 TDX 与 Tushare 的 volume/amount。
- volume/amount 主口径：重合日期使用 Tushare `fund_daily`，无 Tushare 行时回退 TDX；Tushare amount 从千元换算为元。
- volume 事件调整：再按份额变化倍数调整，消除拆分/合并导致的单位台阶。
- 停牌/折算零成交日：保留在长表审计，但在 NPZ 中置为缺失且 `mask=False`。
- 全程使用 `float64`，不模拟通达信客户端的两位小数显示。

事件审计同时记录原始 volume、按事件后份额口径换算的前一日 volume，以及调整前后的 volume 比率。成交量本身受市场活跃度影响，因此比率只用于检查单位口径，不作为必须连续的价格类门禁。作为 flow 硬门禁，`amount / (volume * 100)` 推导的成交均价必须落在当日 low/high 附近。

价格事件比例：

```text
category=1:
share_step = (10 + 送转股 + 配股) / 10
cash_per_share = (分红 - 配股 * 配股价) / 10
theoretical_ex_close = (previous_close - cash_per_share) / share_step
price_step = theoretical_ex_close / previous_close

category=11/12:
share_step = c3
price_step = 1 / c3
```

事件日前所有历史价格乘 `price_step`，历史 volume 乘 `share_step`。多次事件按时间累计。

## 运行

```bash
PYTHONPATH=src .venv/bin/python scripts/phase1b_build_event_adjusted_data.py
```

默认输出到：

```text
data/processed/phase1b_event_adjusted/
```

主要文件：

```text
etf_daily_event_adjusted.parquet
etf_panel_event_adjusted.npz
etf_adjustment_events.csv
etf_adjustment_quality_summary.csv
etf_adjustment_anomalies.csv
build_summary.json
```

如果本地已有 Tushare 探针数据，还会输出逐日收益对照：

```text
tushare_return_comparison.csv
tushare_return_comparison_summary.csv
```

## 边界

该产物目前是 shadow 数据，不被任何研究模块默认读取。必须先完成事件审计、异常收益审计和 Tushare 对照，再决定是否替换现有数据入口。

构建门禁要求：支持的样本内事件不得意外跳过，复权 OHLC 必须为有限正数且结构合法，调整后 volume 与 amount 必须为有限非负数，flow 推导均价必须合理，调整后单日收益不得超过配置的异常阈值。门禁失败时不写出新产物。