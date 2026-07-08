# Phase 1 数据地基

Phase 1 只做数据地基和第一版词表，不碰模型训练。

## 前置条件

- Python 3.10+
- `pandas`, `numpy`, `pyarrow`
- Go 1.23+（当前环境暂未安装 Go，安装后才能运行 TDX exporter）
- 可访问通达信行情服务器

## 数据源口径

- 主源：`github.com/injoyai/tdx v0.0.82`
- 不静默混源；TDX 拉取异常时中止并人工决策
- 保存 TDX 不复权 OHLCV 和 gbbq 事件
- `qfq_latest` 是当前前复权派生数据，不是严格点时复权

## 运行顺序

先导出 TDX staging JSONL：

```bash
cd tools/tdx_exporter
go mod download
go run . -out ../../data/staging -start 2010-01-01
```

再构建 Python 研究数据：

```bash
python scripts/phase1_build_data.py
```

## 预期产物

```text
data/staging/tdx_daily_bfq.jsonl
data/staging/tdx_daily_qfq_latest.jsonl
data/staging/tdx_gbbq_events.jsonl
data/raw/etf_daily_bfq.parquet
data/raw/etf_gbbq_events.parquet
data/processed/etf_panel_raw.npz
data/processed/etf_panel_qfq_latest.npz
data/processed/etf_panel_report.csv
```

## 验收标准

- 17 只 ETF 全部成功拉取
- raw/qfq 日期对齐
- OHLC 合法
- `high >= max(open, close)`
- `low <= min(open, close)`
- `volume/amount` 无异常空值
- raw/qfq mask 一致
- `[N,F,T]` shape 正确
