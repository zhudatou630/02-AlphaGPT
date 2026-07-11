#!/usr/bin/env python3
"""Validate V3 TDX staging data and write immutable raw parquet inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from alpha_etf.data.build_panel import assert_no_duplicate_dates, check_ohlc, normalize_gbbq, read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", type=Path, default=ROOT / "configs/research_v3_universe.json")
    parser.add_argument("--staging-dir", type=Path, default=ROOT / "data/staging/v3")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/raw/v3")
    return parser.parse_args()


def normalize_daily(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "date", "symbol", "tdx_symbol", "name", "category", "bucket",
        "open", "high", "low", "close", "volume", "amount",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"TDX daily data missing columns: {sorted(missing)}")
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["symbol"] = out["symbol"].astype(str)
    for column in ("open", "high", "low", "close", "volume", "amount"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out.sort_values(["symbol", "date"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    universe = json.loads(args.universe.read_text(encoding="utf-8"))
    symbols = [str(item["symbol"]) for item in universe]
    expected = set(symbols)

    raw = normalize_daily(read_jsonl(args.staging_dir / "tdx_daily_bfq.jsonl"))
    events = normalize_gbbq(read_jsonl(args.staging_dir / "tdx_gbbq_events.jsonl"))
    assert_no_duplicate_dates(raw, "V3 TDX raw")
    actual = set(raw["symbol"].unique())
    if actual != expected:
        raise ValueError(f"TDX symbol mismatch: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")
    report = check_ohlc(raw, "tdx_raw")
    gate_columns = [column for column in report if column.endswith("_rows")]
    failures = {column: int(report[column].sum()) for column in gate_columns if report[column].sum()}
    if failures:
        raise ValueError(f"V3 raw TDX quality gate failed: {failures}")

    first_dates = raw.groupby("symbol")["date"].min()
    last_dates = raw.groupby("symbol")["date"].max()
    report["first_date"] = report["symbol"].map(first_dates)
    report["last_date"] = report["symbol"].map(last_dates)
    report["rows"] = report["symbol"].map(raw.groupby("symbol").size())
    report["gbbq_events"] = report["symbol"].map(events.groupby("symbol").size()).fillna(0).astype(int)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(args.output_dir / "etf_daily_bfq.parquet", index=False)
    events.to_parquet(args.output_dir / "etf_gbbq_events.parquet", index=False)
    report.to_csv(args.output_dir / "tdx_raw_quality.csv", index=False)
    summary = {
        "symbols": len(expected),
        "daily_rows": len(raw),
        "event_rows": len(events),
        "first_date": raw["date"].min().date().isoformat(),
        "last_date": raw["date"].max().date().isoformat(),
        "quality_gate_failures": failures,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()