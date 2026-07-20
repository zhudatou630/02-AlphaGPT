#!/usr/bin/env python3
"""Download immutable source snapshots for expanded ETF universe auditing."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import tushare as ts


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "expanded-etf-audit-source-snapshot-v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "processed" / "expanded_etf_audit" / "sources",
    )
    parser.add_argument("--snapshot-as-of", default=pd.Timestamp.today().strftime("%Y%m%d"))
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nonempty(frames: list[pd.DataFrame], name: str) -> pd.DataFrame:
    values = [frame for frame in frames if not frame.empty]
    if not values:
        raise RuntimeError(f"Tushare returned no {name} rows")
    return pd.concat(values, ignore_index=True).drop_duplicates("ts_code", keep="first")


def main() -> None:
    args = _parse_args()
    if not os.environ.get("TUSHARE_TOKEN"):
        raise RuntimeError("TUSHARE_TOKEN is not set")
    pro = ts.pro_api()
    fund_basic = _nonempty(
        [pro.fund_basic(market="E", status=status) for status in ("L", "D")],
        "fund_basic",
    )
    etf_basic = _nonempty(
        [pro.etf_basic(list_status=status) for status in ("L", "D")],
        "etf_basic",
    )
    index_basic = _nonempty(
        [pro.index_basic(market=market) for market in ("SSE", "SZSE", "CSI", "SW")],
        "index_basic",
    )
    trade_calendar = pro.trade_cal(
        exchange="SSE",
        start_date="20050101",
        end_date=args.snapshot_as_of,
        is_open="1",
    )
    if trade_calendar.empty:
        raise RuntimeError("Tushare returned no open trade calendar rows")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "fund_basic": args.output_dir / "tushare_fund_basic_all.parquet",
        "etf_basic": args.output_dir / "tushare_etf_basic_all.parquet",
        "index_basic": args.output_dir / "tushare_index_basic_all.parquet",
        "trade_calendar": args.output_dir / "tushare_sse_open_calendar.parquet",
    }
    fund_basic.to_parquet(paths["fund_basic"], index=False)
    etf_basic.to_parquet(paths["etf_basic"], index=False)
    index_basic.to_parquet(paths["index_basic"], index=False)
    trade_calendar.to_parquet(paths["trade_calendar"], index=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "snapshot_as_of": args.snapshot_as_of,
        "tushare_version": ts.__version__,
        "sources": {
            name: {"file": path.name, "sha256": _sha256(path)}
            for name, path in paths.items()
        },
        "row_counts": {
            "fund_basic": len(fund_basic),
            "etf_basic": len(etf_basic),
            "index_basic": len(index_basic),
            "trade_calendar": len(trade_calendar),
        },
    }
    manifest_path = args.output_dir / "source_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()