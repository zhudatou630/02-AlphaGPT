#!/usr/bin/env python3
"""Download pre-termination market and NAV history for ended target ETFs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import tushare as ts


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "expanded-etf-ended-history-snapshot-v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase1-dir",
        type=Path,
        default=ROOT / "data" / "processed" / "expanded_etf_audit" / "phase1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "processed" / "expanded_etf_audit" / "ended_history",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _concat(frames: list[pd.DataFrame], keys: list[str]) -> pd.DataFrame:
    values = [frame for frame in frames if not frame.empty]
    if not values:
        return pd.DataFrame()
    return pd.concat(values, ignore_index=True).drop_duplicates(keys, keep="last")


def main() -> None:
    args = _parse_args()
    if not os.environ.get("TUSHARE_TOKEN"):
        raise RuntimeError("TUSHARE_TOKEN is not set")
    products = pd.read_csv(args.phase1_dir / "product_master.csv", dtype=str)
    ended = products[
        products["scope_decision"].eq("include_domestic_passive_equity")
        & products["status"].eq("D")
    ].copy()
    pro = ts.pro_api()
    product_dir = args.output_dir / "products"
    product_dir.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, str]] = []
    for row in ended.itertuples(index=False):
        daily_path = product_dir / f"{row.ts_code}_daily.parquet"
        nav_path = product_dir / f"{row.ts_code}_nav.parquet"
        share_path = product_dir / f"{row.ts_code}_share.parquet"
        if daily_path.exists() and nav_path.exists() and share_path.exists():
            print(f"cached ended history {row.ts_code} {row.name}", flush=True)
            continue
        try:
            end = str(row.delist_date or row.due_date)
            if not daily_path.exists():
                daily = pro.fund_daily(
                    ts_code=row.ts_code, start_date=str(row.list_date), end_date=end
                )
                daily.to_parquet(daily_path, index=False)
            if not nav_path.exists():
                nav = pro.fund_nav(ts_code=row.ts_code, market="E")
                nav.to_parquet(nav_path, index=False)
            if not share_path.exists():
                share = pro.fund_share(ts_code=row.ts_code)
                share.to_parquet(share_path, index=False)
            print(f"downloaded ended history {row.ts_code} {row.name}", flush=True)
        except Exception as exc:
            failures.append(
                {"ts_code": str(row.ts_code), "error": f"{type(exc).__name__}: {exc}"}
            )
            print(f"failed ended history {row.ts_code}: {exc}", flush=True)
    daily = _concat(
        [pd.read_parquet(path) for path in sorted(product_dir.glob("*_daily.parquet"))],
        ["ts_code", "trade_date"],
    )
    nav = _concat(
        [pd.read_parquet(path) for path in sorted(product_dir.glob("*_nav.parquet"))],
        ["ts_code", "nav_date"],
    )
    share = _concat(
        [pd.read_parquet(path) for path in sorted(product_dir.glob("*_share.parquet"))],
        ["ts_code", "trade_date"],
    )
    failure_frame = pd.DataFrame(failures, columns=["ts_code", "error"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "daily": args.output_dir / "tushare_ended_fund_daily.parquet",
        "nav": args.output_dir / "tushare_ended_fund_nav.parquet",
        "share": args.output_dir / "tushare_ended_fund_share.parquet",
        "failures": args.output_dir / "download_failures.csv",
    }
    daily.to_parquet(outputs["daily"], index=False)
    nav.to_parquet(outputs["nav"], index=False)
    share.to_parquet(outputs["share"], index=False)
    failure_frame.to_csv(outputs["failures"], index=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "target_ended_products": len(ended),
        "daily_products": int(daily["ts_code"].nunique()) if not daily.empty else 0,
        "nav_products": int(nav["ts_code"].nunique()) if not nav.empty else 0,
        "share_products": int(share["ts_code"].nunique()) if not share.empty else 0,
        "download_failures": len(failure_frame),
        "row_counts": {
            "daily": len(daily),
            "nav": len(nav),
            "share": len(share),
        },
        "outputs": {
            name: {"file": path.name, "sha256": _sha256(path)}
            for name, path in outputs.items()
        },
    }
    manifest_path = args.output_dir / "ended_history_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()