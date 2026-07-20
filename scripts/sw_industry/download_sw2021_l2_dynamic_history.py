#!/usr/bin/env python3
"""Download immutable SW2021 L2 history with per-index dynamic entry dates."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import numpy as np
import pandas as pd
import tushare as ts


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from download_sw2021_l2_backcast import (  # noqa: E402
    DEFAULT_IDENTITY_CONFIG,
    EXPECTED_PUBLISHED_INDUSTRIES,
    NUMERIC_FIELDS,
    QUOTE_FIELDS,
    ROOT,
    SW2021_EFFECTIVE_DATE,
    fetch_calendar,
    fetch_classification,
    fsync_directory,
    fsync_file,
    load_identity,
    project_path,
    sha256,
    write_text_synced,
)


DEFAULT_START_DATE = "20000315"
DOWNLOAD_CHUNKS = (
    ("20000315", "20091231"),
    ("20100101", "20191231"),
    ("20200101", None),
)
EXPECTED_INTERNAL_GAP = {
    "ts_code": "801193.SI",
    "first_date": "20150401",
    "last_date": "20150630",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity-config", type=Path, default=DEFAULT_IDENTITY_CONFIG)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=pd.Timestamp.today().strftime("%Y%m%d"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "data/raw/sw_industry_l2/sw2021_dynamic/snapshots",
    )
    parser.add_argument("--snapshot-id")
    return parser.parse_args()


def chunk_ranges(start_date: str, end_date: str) -> list[tuple[str, str]]:
    ranges = []
    for chunk_start, chunk_end in DOWNLOAD_CHUNKS:
        lower = max(start_date, chunk_start)
        upper = min(end_date, chunk_end or end_date)
        if lower <= upper:
            ranges.append((lower, upper))
    if not ranges or ranges[0][0] != start_date or ranges[-1][1] != end_date:
        raise ValueError("Requested range is outside the frozen chunk schedule")
    return ranges


def fetch_dynamic_quotes(
    pro: Any,
    expected: dict[str, str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    frames = []
    ranges = chunk_ranges(start_date, end_date)
    for code in sorted(expected):
        code_frames = []
        for chunk_start, chunk_end in ranges:
            frame = pro.sw_daily(
                ts_code=code,
                start_date=chunk_start,
                end_date=chunk_end,
                fields=",".join(QUOTE_FIELDS),
            ).copy()
            if len(frame) >= 4000:
                raise RuntimeError(
                    f"Tushare row limit may have truncated {code} {chunk_start}-{chunk_end}"
                )
            if not frame.empty:
                code_frames.append(frame)
        if not code_frames:
            raise RuntimeError(f"Tushare returned no history for published L2 code {code}")
        combined = pd.concat(code_frames, ignore_index=True)
        if combined.duplicated(["ts_code", "trade_date"]).any():
            raise RuntimeError(f"Chunk overlap produced duplicate rows for {code}")
        frames.append(combined)
        print(
            f"downloaded {code} {expected[code]}: {len(combined)} rows "
            f"{combined['trade_date'].astype(str).min()}-{combined['trade_date'].astype(str).max()}"
        )
    quotes = pd.concat(frames, ignore_index=True)
    quotes = quotes.rename(columns={"name": "source_name"})
    quotes["ts_code"] = quotes["ts_code"].astype(str)
    quotes["trade_date"] = quotes["trade_date"].astype(str)
    for field in NUMERIC_FIELDS:
        quotes[field] = pd.to_numeric(quotes[field], errors="coerce")
    quotes["classification_version"] = "SW2021"
    quotes["classification_level"] = "L2"
    quotes["industry_name"] = quotes["ts_code"].map(expected)
    return quotes.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)


def validate_dynamic_quotes(
    quotes: pd.DataFrame,
    expected: dict[str, str],
    calendar: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    expected_codes = set(expected)
    observed_codes = set(quotes["ts_code"])
    open_dates = calendar.loc[
        (pd.to_numeric(calendar["is_open"]) == 1)
        & (calendar["cal_date"] >= start_date)
        & (calendar["cal_date"] <= end_date),
        "cal_date",
    ].astype(str)
    open_date_set = set(open_dates)
    quote_date_set = set(quotes["trade_date"])
    prices = quotes[["open", "high", "low", "close"]].to_numpy(dtype=np.float64)
    flow = quotes[["vol", "amount"]].to_numpy(dtype=np.float64)
    invalid_ohlc = quotes[
        (quotes["high"] < quotes[["open", "low", "close"]].max(axis=1))
        | (quotes["low"] > quotes[["open", "high", "close"]].min(axis=1))
    ]
    availability_rows = []
    internal_gap_records = []
    internal_gap_dates_by_code: dict[str, set[str]] = {}
    terminal_gap_codes = []
    ordered_open_dates = sorted(open_date_set)
    for code in sorted(expected_codes):
        code_dates = set(quotes.loc[quotes["ts_code"] == code, "trade_date"])
        first_date = min(code_dates)
        last_date = max(code_dates)
        expected_dates = {
            date for date in ordered_open_dates if first_date <= date <= last_date
        }
        missing_dates = sorted(expected_dates - code_dates)
        if missing_dates:
            internal_gap_dates_by_code[code] = set(missing_dates)
            internal_gap_records.append(
                {
                    "ts_code": code,
                    "first_date": first_date,
                    "last_date": last_date,
                    "missing_dates": len(missing_dates),
                    "first_missing_dates": missing_dates[:10],
                }
            )
        if last_date != end_date:
            terminal_gap_codes.append(code)
        availability_rows.append(
            {
                "ts_code": code,
                "industry_name": expected[code],
                "first_quote_date": first_date,
                "last_quote_date": last_date,
                "quote_rows": len(code_dates),
                "missing_open_dates_within_span": len(missing_dates),
            }
        )
    availability = pd.DataFrame(availability_rows)
    expected_gap_dates = {
        date
        for date in ordered_open_dates
        if EXPECTED_INTERNAL_GAP["first_date"]
        <= date
        <= EXPECTED_INTERNAL_GAP["last_date"]
    }
    expected_gaps = {EXPECTED_INTERNAL_GAP["ts_code"]: expected_gap_dates}
    failures = {
        "duplicate_key_rows": int(quotes.duplicated(["ts_code", "trade_date"]).sum()),
        "missing_expected_codes": len(expected_codes - observed_codes),
        "unexpected_codes": len(observed_codes - expected_codes),
        "open_dates_without_any_industry_quotes": len(open_date_set - quote_date_set),
        "quotes_on_closed_dates": len(quote_date_set - open_date_set),
        "codes_missing_latest_open_date": len(terminal_gap_codes),
        "unexpected_internal_quote_gap_contract": int(
            internal_gap_dates_by_code != expected_gaps
        ),
        "nonfinite_price_rows": int((~np.isfinite(prices)).any(axis=1).sum()),
        "nonpositive_price_rows": int((prices <= 0).any(axis=1).sum()),
        "invalid_ohlc_structure_rows": len(invalid_ohlc),
    }
    blocking = {key: value for key, value in failures.items() if value}
    if blocking:
        raise ValueError(
            f"SW2021 L2 dynamic quote quality gate failed: {blocking}; "
            f"internal_gap_examples={internal_gap_records[:5]}; "
            f"terminal_gap_codes={terminal_gap_codes[:10]}"
        )
    if str(quotes["trade_date"].min()) != start_date:
        raise ValueError("Observed first quote date differs from frozen dynamic start")
    first_date_counts = {
        str(date): int(count)
        for date, count in availability.groupby("first_quote_date").size().items()
    }
    daily_counts = quotes.groupby("trade_date")["ts_code"].nunique()
    source_name_counts = quotes.groupby("ts_code")["source_name"].nunique(dropna=False)
    quality = {
        "rows": len(quotes),
        "industries": len(expected_codes),
        "calendar_trading_days": len(open_date_set),
        "quote_trading_days": len(quote_date_set),
        "first_date": str(quotes["trade_date"].min()),
        "last_date": str(quotes["trade_date"].max()),
        "minimum_daily_industries": int(daily_counts.min()),
        "maximum_daily_industries": int(daily_counts.max()),
        "first_quote_date_counts": first_date_counts,
        "codes_with_internal_quote_gaps": len(internal_gap_records),
        "internal_quote_gap_dates": int(
            sum(record["missing_dates"] for record in internal_gap_records)
        ),
        "internal_quote_gap_examples": internal_gap_records[:20],
        "internal_quote_gap_contract": {
            **EXPECTED_INTERNAL_GAP,
            "trading_days": len(expected_gap_dates),
        },
        "nonfinite_flow_rows": int((~np.isfinite(flow)).any(axis=1).sum()),
        "nonpositive_flow_rows": int((flow <= 0).any(axis=1).sum()),
        "flow_fields_used_by_research": False,
        "rows_before_SW2021_effective_date": int(
            (quotes["trade_date"] < SW2021_EFFECTIVE_DATE).sum()
        ),
        "rows_on_or_after_SW2021_effective_date": int(
            (quotes["trade_date"] >= SW2021_EFFECTIVE_DATE).sum()
        ),
        "codes_with_multiple_source_names": int((source_name_counts > 1).sum()),
        "missing_values": {
            column: int(count)
            for column, count in quotes.isna().sum().items()
            if count
        },
        "blocking_quality_failures": blocking,
        "dynamic_close_research_gate_passed": True,
    }
    return quality, availability.sort_values("ts_code").reset_index(drop=True)


def main() -> None:
    args = parse_args()
    captured_at = datetime.now(timezone.utc)
    frozen_rows, expected = load_identity(args.identity_config)
    pro = ts.pro_api()
    calendar, latest_open_date = fetch_calendar(pro, args.start_date, args.end_date)
    classification = fetch_classification(pro, frozen_rows, expected)
    quotes = fetch_dynamic_quotes(pro, expected, args.start_date, latest_open_date)
    quality, availability = validate_dynamic_quotes(
        quotes, expected, calendar, args.start_date, latest_open_date
    )

    timestamp = captured_at.strftime("%Y%m%dT%H%M%SZ")
    snapshot_id = args.snapshot_id or (
        f"sw2021-l2-dynamic-history-{args.start_date}-{latest_open_date}-{timestamp}"
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    final_dir = args.output_root / snapshot_id
    if final_dir.exists():
        raise FileExistsError(f"Snapshot already exists: {final_dir}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{snapshot_id}.", dir=args.output_root))
    try:
        quote_path = temporary / "sw2021_l2_dynamic_daily.parquet"
        classification_path = temporary / "sw2021_l2_classification.parquet"
        calendar_path = temporary / "sse_trade_calendar.parquet"
        availability_path = temporary / "availability.csv"
        quotes.to_parquet(quote_path, index=False)
        classification.to_parquet(classification_path, index=False)
        calendar.to_parquet(calendar_path, index=False)
        availability.to_csv(availability_path, index=False)
        data_files = (quote_path, classification_path, calendar_path, availability_path)
        for path in data_files:
            fsync_file(path)
        manifest = {
            "schema_version": "sw2021-l2-dynamic-history-snapshot-v1",
            "snapshot_id": snapshot_id,
            "captured_at_utc": captured_at.isoformat(),
            "source": "Tushare",
            "classification_endpoint": {
                "api": "index_classify",
                "params": {"level": "L2", "src": "SW2021"},
            },
            "quote_endpoint": {
                "api": "sw_daily",
                "request_strategy": "three date chunks per frozen published SW2021 L2 code",
                "chunks": chunk_ranges(args.start_date, latest_open_date),
                "fields": list(QUOTE_FIELDS),
            },
            "calendar_endpoint": {"api": "trade_cal", "params": {"exchange": "SSE"}},
            "requested_start_date": args.start_date,
            "requested_end_date": args.end_date,
            "data_as_of_trade_date": latest_open_date,
            "classification_version": "SW2021",
            "classification_level": "L2",
            "classification_effective_date": SW2021_EFFECTIVE_DATE,
            "identity_policy": "CURRENTLY_PUBLISHED_SW2021_L2_ONLY",
            "entry_policy": "FIRST_VENDOR_QUOTE_WITH_PER_ASSET_ROC40_WARMUP_DOWNSTREAM",
            "published_industry_count": len(expected),
            "series_semantics": "PUBLISHED_INDEX_LEVEL_VENDOR_CURRENT",
            "historical_vintage_proven": False,
            "identity_config": project_path(args.identity_config),
            "identity_config_sha256": sha256(args.identity_config),
            "quality": quality,
            "files": {
                path.name: {"sha256": sha256(path), "size_bytes": path.stat().st_size}
                for path in data_files
            },
            "package_versions": {
                package: metadata.version(package)
                for package in ("pandas", "pyarrow", "tushare")
            },
        }
        manifest_path = temporary / "manifest.json"
        write_text_synced(
            manifest_path,
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        )
        checksum_paths = (*data_files, manifest_path)
        write_text_synced(
            temporary / "SHA256SUMS",
            "\n".join(f"{sha256(path)}  {path.name}" for path in checksum_paths) + "\n",
        )
        fsync_directory(temporary)
        os.replace(temporary, final_dir)
        fsync_directory(args.output_root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2))
    print(f"snapshot: {final_dir}")


if __name__ == "__main__":
    main()