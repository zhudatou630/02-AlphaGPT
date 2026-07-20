#!/usr/bin/env python3
"""Cross-check expanded-universe price-jump candidates against Tushare OHLC."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import pandas as pd
import tushare as ts


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "expanded-etf-price-jump-cross-source-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase1-dir",
        type=Path,
        default=ROOT / "data" / "processed" / "expanded_etf_audit" / "phase1",
    )
    parser.add_argument("--request-interval", type=float, default=0.08)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not os.environ.get("TUSHARE_TOKEN"):
        raise RuntimeError("TUSHARE_TOKEN is not set")
    candidate_path = args.phase1_dir / "price_jump_audit.csv"
    master_path = args.phase1_dir / "product_master.csv"
    candidates = pd.read_csv(candidate_path, dtype={"symbol": str})
    master = (
        pd.read_csv(master_path, dtype=str)
        .drop_duplicates("symbol")
        .set_index("symbol")
    )
    pro = ts.pro_api()
    rows: list[dict[str, object]] = []
    for row in candidates.itertuples(index=False):
        ts_code = str(master.loc[row.symbol, "ts_code"])
        trade_date = str(row.date).replace("-", "")[:8]
        previous_date = str(row.previous_date).replace("-", "")[:8]
        if row.source != "tdx_bfq":
            rows.append(
                {
                    "symbol": row.symbol,
                    "date": row.date,
                    "previous_date": row.previous_date,
                    "ts_code": ts_code,
                    "classification": row.classification,
                    "primary_source": row.source,
                    "status": "same_source_not_independent",
                }
            )
            continue
        result = pd.DataFrame()
        error = ""
        for attempt in range(3):
            try:
                result = pro.fund_daily(
                    ts_code=ts_code, start_date=previous_date, end_date=trade_date
                )
                break
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                time.sleep(1.0 + attempt)
        current = result[result["trade_date"].astype(str).eq(trade_date)]
        previous = result[result["trade_date"].astype(str).eq(previous_date)]
        if current.empty:
            rows.append(
                {
                    "symbol": row.symbol,
                    "date": row.date,
                    "previous_date": row.previous_date,
                    "ts_code": ts_code,
                    "classification": row.classification,
                    "primary_source": row.source,
                    "status": error or "missing_current_row",
                }
            )
            continue
        value = current.iloc[0]
        previous_value = previous.iloc[0] if not previous.empty else None
        differences = [
            abs(float(getattr(row, price)) - float(value[price]))
            for price in ("open", "high", "low", "close")
        ]
        if previous_value is not None:
            differences.append(
                abs(float(row.previous_close) - float(previous_value["close"]))
            )
        ts_reference = float(value["pre_close"])
        ts_adjusted_gap = max(
            abs(float(value["open"]) / ts_reference - 1.0),
            abs(float(value["close"]) / ts_reference - 1.0),
        )
        ts_limit_bound = 0.20 + 0.0005 / ts_reference
        rows.append(
            {
                "symbol": row.symbol,
                "date": row.date,
                "previous_date": row.previous_date,
                "ts_code": ts_code,
                "classification": row.classification,
                "primary_source": row.source,
                "status": (
                    "ok_cross_source_full"
                    if previous_value is not None
                    else "ok_cross_source_adjusted_reference"
                ),
                "primary_previous_close": row.previous_close,
                "ts_previous_close": (
                    previous_value["close"] if previous_value is not None else pd.NA
                ),
                "event_reference_close": row.event_reference_close,
                "ts_current_pre_close": value["pre_close"],
                "event_reference_abs_diff": abs(
                    float(row.event_reference_close) - ts_reference
                ),
                "ts_max_abs_adjusted_gap": ts_adjusted_gap,
                "ts_limit_rounding_bound": ts_limit_bound,
                "ts_adjusted_limit_breach": ts_adjusted_gap > ts_limit_bound,
                **{
                    f"primary_{price}": getattr(row, price)
                    for price in ("open", "high", "low", "close")
                },
                **{
                    f"ts_{price}": value[price]
                    for price in ("open", "high", "low", "close")
                },
                "max_primary_price_abs_diff": max(differences),
            }
        )
        time.sleep(args.request_interval)
    output = pd.DataFrame(rows)
    output_path = args.phase1_dir / "price_jump_cross_source_check.csv"
    output.to_csv(output_path, index=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "query": {
            "api": "tushare.pro.fund_daily",
            "date_range": "candidate_previous_date_to_candidate_date",
            "compared_fields": [
                "previous_close_when_available",
                "event_reference_close",
                "open",
                "high",
                "low",
                "close",
            ],
        },
        "summary": {
            "candidates": len(candidates),
            "verified": int(output["status"].str.startswith("ok_cross_source").sum()),
            "full_previous_close_verified": int(
                output["status"].eq("ok_cross_source_full").sum()
            ),
            "adjusted_reference_only": int(
                output["status"].eq("ok_cross_source_adjusted_reference").sum()
            ),
            "same_source_unverified": int(
                output["status"].eq("same_source_not_independent").sum()
            ),
            "failed": int(
                (
                    ~output["status"].isin(
                        [
                            "ok_cross_source_full",
                            "ok_cross_source_adjusted_reference",
                            "same_source_not_independent",
                        ]
                    )
                ).sum()
            ),
            "mismatches": int(
                pd.to_numeric(output["max_primary_price_abs_diff"], errors="coerce")
                .gt(1e-12)
                .sum()
            ),
            "adjusted_limit_breaches": int(
                output["ts_adjusted_limit_breach"].fillna(False).astype(bool).sum()
            ),
        },
        "inputs": {
            "price_jump_audit": _sha256(candidate_path),
            "product_master": _sha256(master_path),
        },
        "output": {"file": output_path.name, "sha256": _sha256(output_path)},
    }
    manifest_path = args.phase1_dir / "price_jump_cross_source_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()