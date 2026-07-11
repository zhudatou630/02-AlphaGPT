#!/usr/bin/env python3
"""Download local-only Tushare quality probes for the frozen V3 universe."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import tushare as ts


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", type=Path, default=ROOT / "configs/research_v3_universe.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed/v3/data_quality")
    parser.add_argument("--end-date", default=pd.Timestamp.today().strftime("%Y%m%d"))
    return parser.parse_args()


def yearly_ranges(start: str, end: str) -> list[tuple[str, str]]:
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)
    ranges = []
    for year in range(start_ts.year, end_ts.year + 1):
        left = max(start_ts, pd.Timestamp(year=year, month=1, day=1))
        right = min(end_ts, pd.Timestamp(year=year, month=12, day=31))
        if left <= right:
            ranges.append((left.strftime("%Y%m%d"), right.strftime("%Y%m%d")))
    return ranges


def main() -> None:
    args = parse_args()
    if not os.environ.get("TUSHARE_TOKEN"):
        raise RuntimeError("TUSHARE_TOKEN is not set; source .venv/bin/activate first")
    universe = json.loads(args.universe.read_text(encoding="utf-8"))
    pro = ts.pro_api()
    basic_frames = [pro.fund_basic(market="E", status=status) for status in ("L", "D")]
    basic = pd.concat(basic_frames, ignore_index=True).drop_duplicates("ts_code")
    basic["symbol"] = basic["ts_code"].str[:6]
    basic = basic[basic["symbol"].isin({item["symbol"] for item in universe})].copy()
    if len(basic) != len(universe):
        raise ValueError(f"Tushare basic coverage mismatch: {len(basic)} != {len(universe)}")

    daily_frames = []
    adj_frames = []
    for row in basic.itertuples(index=False):
        start = str(row.list_date or row.found_date)
        for range_start, range_end in yearly_ranges(start, args.end_date):
            daily = pro.fund_daily(ts_code=row.ts_code, start_date=range_start, end_date=range_end)
            if not daily.empty:
                daily_frames.append(daily)
            adj = pro.fund_adj(ts_code=row.ts_code, start_date=range_start, end_date=range_end)
            if not adj.empty:
                adj_frames.append(adj)
        print(f"downloaded {row.ts_code} {row.name}")

    daily = pd.concat(daily_frames, ignore_index=True).drop_duplicates(["ts_code", "trade_date"])
    adj = pd.concat(adj_frames, ignore_index=True).drop_duplicates(["ts_code", "trade_date"])
    for frame in (daily, adj):
        frame["symbol"] = frame["ts_code"].str[:6]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    basic.to_parquet(args.output_dir / "tushare_fund_basic.parquet", index=False)
    daily.to_parquet(args.output_dir / "tushare_fund_daily.parquet", index=False)
    adj.to_parquet(args.output_dir / "tushare_fund_adj.parquet", index=False)
    summary = {
        "symbols": len(basic),
        "daily_rows": len(daily),
        "adj_rows": len(adj),
        "daily_first_date": str(daily["trade_date"].min()),
        "daily_last_date": str(daily["trade_date"].max()),
    }
    (args.output_dir / "download_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()