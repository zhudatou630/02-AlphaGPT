"""Build Phase 1 parquet, panel npz, and audit reports from TDX staging JSONL."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_etf.config import ETF_UNIVERSE, RAW_FEATURES, START_DATE


ROOT = Path(__file__).resolve().parents[3]
STAGING_DIR = ROOT / "data" / "staging"
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"


DAILY_BFQ_JSONL = STAGING_DIR / "tdx_daily_bfq.jsonl"
DAILY_QFQ_JSONL = STAGING_DIR / "tdx_daily_qfq_latest.jsonl"
GBBQ_JSONL = STAGING_DIR / "tdx_gbbq_events.jsonl"

DAILY_BFQ_PARQUET = RAW_DIR / "etf_daily_bfq.parquet"
GBBQ_PARQUET = RAW_DIR / "etf_gbbq_events.parquet"
RAW_PANEL_NPZ = PROCESSED_DIR / "etf_panel_raw.npz"
QFQ_PANEL_NPZ = PROCESSED_DIR / "etf_panel_qfq_latest.npz"
REPORT_CSV = PROCESSED_DIR / "etf_panel_report.csv"


def read_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing staging file: {path}")

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    return pd.DataFrame(rows)


def normalize_daily(df: pd.DataFrame, source: str) -> pd.DataFrame:
    required = {
        "date",
        "symbol",
        "tdx_symbol",
        "name",
        "category",
        "bucket",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{source} daily data missing columns: {sorted(missing)}")

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out = out[out["date"] >= pd.Timestamp(START_DATE)].copy()
    out["source"] = source

    for col in RAW_FEATURES:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    cols = [
        "date",
        "symbol",
        "tdx_symbol",
        "name",
        "category",
        "bucket",
        "source",
        *RAW_FEATURES,
    ]
    return out[cols].sort_values(["symbol", "date"]).reset_index(drop=True)


def normalize_gbbq(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "symbol",
                "tdx_symbol",
                "category_code",
                "c1",
                "c2",
                "c3",
                "c4",
                "is_xrxd",
                "is_equity",
            ]
        )

    required = {"date", "symbol", "tdx_symbol", "category_code", "c1", "c2", "c3", "c4"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"gbbq data missing columns: {sorted(missing)}")

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["category_code"] = pd.to_numeric(out["category_code"], errors="coerce").astype("Int64")
    for col in ("c1", "c2", "c3", "c4"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["is_xrxd"] = out["category_code"].eq(1)
    out["is_equity"] = out["category_code"].isin([2, 3, 5, 7, 8, 9, 10])
    cols = ["date", "symbol", "tdx_symbol", "category_code", "c1", "c2", "c3", "c4", "is_xrxd", "is_equity"]
    return out[cols].sort_values(["symbol", "date", "category_code"]).reset_index(drop=True)


def assert_no_duplicate_dates(df: pd.DataFrame, label: str) -> None:
    dup = df.duplicated(["symbol", "date"])
    if dup.any():
        examples = df.loc[dup, ["symbol", "date"]].head(10).to_dict("records")
        raise ValueError(f"{label} has duplicate symbol/date rows: {examples}")


def check_ohlc(df: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    for symbol, g in df.groupby("symbol", sort=False):
        invalid_price = (g[["open", "high", "low", "close"]] <= 0).any(axis=1)
        invalid_high = g["high"] < g[["open", "close"]].max(axis=1)
        invalid_low = g["low"] > g[["open", "close"]].min(axis=1)
        missing_flow = g[["volume", "amount"]].isna().any(axis=1)
        rows.append(
            {
                "symbol": symbol,
                f"{label}_invalid_price_rows": int(invalid_price.sum()),
                f"{label}_invalid_high_rows": int(invalid_high.sum()),
                f"{label}_invalid_low_rows": int(invalid_low.sum()),
                f"{label}_missing_flow_rows": int(missing_flow.sum()),
            }
        )
    return pd.DataFrame(rows)


def build_panel(df: pd.DataFrame, dates: pd.DatetimeIndex, symbols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    values = np.full((len(symbols), len(RAW_FEATURES), len(dates)), np.nan, dtype=np.float64)
    mask = np.zeros((len(symbols), len(dates)), dtype=bool)

    date_index = {date: i for i, date in enumerate(dates)}
    symbol_index = {symbol: i for i, symbol in enumerate(symbols)}

    for row in df.itertuples(index=False):
        n = symbol_index[row.symbol]
        t = date_index[row.date]
        for f_idx, feature in enumerate(RAW_FEATURES):
            values[n, f_idx, t] = getattr(row, feature)
        mask[n, t] = not np.isnan(values[n, :, t]).any()

    return values, mask


def make_report(raw: pd.DataFrame, qfq: pd.DataFrame, gbbq: pd.DataFrame, mask: np.ndarray, dates: pd.DatetimeIndex) -> pd.DataFrame:
    ohlc_raw = check_ohlc(raw, "raw")
    ohlc_qfq = check_ohlc(qfq, "qfq")
    rows = []
    for i, etf in enumerate(ETF_UNIVERSE):
        rg = raw[raw["symbol"] == etf.symbol]
        qg = qfq[qfq["symbol"] == etf.symbol]
        eg = gbbq[gbbq["symbol"] == etf.symbol]
        raw_dates = set(rg["date"])
        qfq_dates = set(qg["date"])
        latest_raw_close = rg.iloc[-1]["close"] if len(rg) else np.nan
        latest_qfq_close = qg.iloc[-1]["close"] if len(qg) else np.nan
        rows.append(
            {
                "symbol": etf.symbol,
                "tdx_symbol": etf.tdx_symbol,
                "name": etf.name,
                "category": etf.category,
                "bucket": etf.bucket,
                "raw_first_date": rg["date"].min().date().isoformat() if len(rg) else "",
                "raw_last_date": rg["date"].max().date().isoformat() if len(rg) else "",
                "raw_rows": len(rg),
                "qfq_rows": len(qg),
                "gbbq_events": len(eg),
                "gbbq_xrxd_events": int(eg["is_xrxd"].sum()) if len(eg) else 0,
                "date_mismatch_count": len(raw_dates.symmetric_difference(qfq_dates)),
                "latest_close_abs_diff": float(abs(latest_raw_close - latest_qfq_close)) if len(rg) and len(qg) else np.nan,
                "mask_true_days": int(mask[i].sum()),
                "mask_coverage": float(mask[i].mean()) if len(dates) else 0.0,
            }
        )

    report = pd.DataFrame(rows)
    report = report.merge(ohlc_raw, on="symbol", how="left").merge(ohlc_qfq, on="symbol", how="left")
    return report


def run_sanity_checks(raw: pd.DataFrame, qfq: pd.DataFrame, report: pd.DataFrame, raw_mask: np.ndarray, qfq_mask: np.ndarray) -> None:
    expected = {etf.symbol for etf in ETF_UNIVERSE}
    raw_symbols = set(raw["symbol"].unique())
    qfq_symbols = set(qfq["symbol"].unique())
    if raw_symbols != expected:
        raise ValueError(f"raw symbols mismatch: missing={sorted(expected - raw_symbols)}, extra={sorted(raw_symbols - expected)}")
    if qfq_symbols != expected:
        raise ValueError(f"qfq symbols mismatch: missing={sorted(expected - qfq_symbols)}, extra={sorted(qfq_symbols - expected)}")
    if not np.array_equal(raw_mask, qfq_mask):
        raise ValueError("raw and qfq masks differ; date alignment must be investigated")

    bad_cols = [
        col
        for col in report.columns
        if col.endswith("invalid_price_rows") or col.endswith("invalid_high_rows") or col.endswith("invalid_low_rows")
    ]
    bad_total = int(report[bad_cols].fillna(0).to_numpy().sum()) if bad_cols else 0
    if bad_total:
        raise ValueError(f"OHLC sanity checks failed; see {REPORT_CSV}")

    mismatch = report[report["date_mismatch_count"] != 0]
    if len(mismatch):
        raise ValueError(f"raw/qfq date mismatch found: {mismatch[['symbol', 'date_mismatch_count']].to_dict('records')}")


def build() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    raw = normalize_daily(read_jsonl(DAILY_BFQ_JSONL), "tdx_bfq")
    qfq = normalize_daily(read_jsonl(DAILY_QFQ_JSONL), "tdx_qfq_latest")
    gbbq = normalize_gbbq(read_jsonl(GBBQ_JSONL))

    assert_no_duplicate_dates(raw, "raw")
    assert_no_duplicate_dates(qfq, "qfq")

    raw.to_parquet(DAILY_BFQ_PARQUET, index=False)
    gbbq.to_parquet(GBBQ_PARQUET, index=False)

    all_dates = pd.DatetimeIndex(sorted(set(raw["date"]).union(set(qfq["date"]))))
    symbols = [etf.symbol for etf in ETF_UNIVERSE]
    raw_values, raw_mask = build_panel(raw, all_dates, symbols)
    qfq_values, qfq_mask = build_panel(qfq, all_dates, symbols)

    report = make_report(raw, qfq, gbbq, raw_mask, all_dates)
    report.to_csv(REPORT_CSV, index=False)
    run_sanity_checks(raw, qfq, report, raw_mask, qfq_mask)

    np.savez_compressed(
        RAW_PANEL_NPZ,
        values=raw_values,
        mask=raw_mask,
        symbols=np.array(symbols),
        features=np.array(RAW_FEATURES),
        dates=all_dates.strftime("%Y-%m-%d").to_numpy(dtype=str),
    )
    np.savez_compressed(
        QFQ_PANEL_NPZ,
        values=qfq_values,
        mask=qfq_mask,
        symbols=np.array(symbols),
        features=np.array(RAW_FEATURES),
        dates=all_dates.strftime("%Y-%m-%d").to_numpy(dtype=str),
    )

    print(f"symbols: {len(symbols)}")
    print(f"features: {', '.join(RAW_FEATURES)}")
    print(f"dates: {all_dates.min().date()} ~ {all_dates.max().date()} ({len(all_dates)})")
    print(f"raw_values.shape: {raw_values.shape}")
    print(f"qfq_values.shape: {qfq_values.shape}")
    print(f"mask.shape: {raw_mask.shape}")
    print(f"report: {REPORT_CSV}")


if __name__ == "__main__":
    build()
