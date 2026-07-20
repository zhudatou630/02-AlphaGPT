#!/usr/bin/env python3
"""Download an immutable SW2021 level-2 current-history snapshot from Tushare."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np
import pandas as pd
import tushare as ts


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_START_DATE = "20160909"
DEFAULT_IDENTITY_CONFIG = ROOT / "configs/sw_industry_l2_sw2021_identity.json"
SW2021_EFFECTIVE_DATE = "20211213"
EXPECTED_CLASSIFICATION_ROWS = 134
EXPECTED_PUBLISHED_INDUSTRIES = 124
EXPECTED_PARENT_INDUSTRIES = 31
QUOTE_FIELDS = (
    "ts_code",
    "trade_date",
    "name",
    "open",
    "low",
    "high",
    "close",
    "change",
    "pct_change",
    "vol",
    "amount",
    "pe",
    "pb",
    "float_mv",
    "total_mv",
)
NUMERIC_FIELDS = tuple(
    field for field in QUOTE_FIELDS if field not in {"ts_code", "trade_date", "name"}
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity-config", type=Path, default=DEFAULT_IDENTITY_CONFIG)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=pd.Timestamp.today().strftime("%Y%m%d"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "data/raw/sw_industry_l2/sw2021_backcast/snapshots",
    )
    parser.add_argument("--snapshot-id")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_text_synced(path: Path, content: str) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def project_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def load_identity(path: Path) -> tuple[list[dict[str, object]], dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "sw2021-l2-identity-v1":
        raise ValueError("Unexpected SW2021 L2 identity schema")
    published = payload.get("published_industries", [])
    unpublished = payload.get("excluded_unpublished_industries", [])
    rows = [*published, *unpublished]
    failures = {
        "classification_rows": len(rows) != EXPECTED_CLASSIFICATION_ROWS,
        "published_industries": len(published) != EXPECTED_PUBLISHED_INDUSTRIES,
        "unpublished_industries": len(unpublished)
        != EXPECTED_CLASSIFICATION_ROWS - EXPECTED_PUBLISHED_INDUSTRIES,
        "published_parent_industries": len({str(row["parent_code"]) for row in published})
        != EXPECTED_PARENT_INDUSTRIES,
        "duplicate_codes": len({str(row["index_code"]) for row in rows}) != len(rows),
    }
    blocking = [name for name, failed in failures.items() if failed]
    if blocking:
        raise ValueError(f"Frozen SW2021 L2 identity contract differs: {blocking}")
    expected = {str(row["index_code"]): str(row["industry_name"]) for row in published}
    return rows, expected


def fetch_calendar(pro: Any, start_date: str, requested_end_date: str) -> tuple[pd.DataFrame, str]:
    calendar = pro.trade_cal(
        exchange="SSE",
        start_date=start_date,
        end_date=requested_end_date,
    ).copy()
    if calendar.empty:
        raise RuntimeError("Tushare returned an empty SSE trade calendar")
    calendar["cal_date"] = calendar["cal_date"].astype(str)
    open_calendar = calendar.loc[pd.to_numeric(calendar["is_open"]) == 1]
    if open_calendar.empty:
        raise RuntimeError("SSE trade calendar has no open dates in the requested range")
    return calendar.sort_values("cal_date").reset_index(drop=True), str(open_calendar["cal_date"].max())


def fetch_classification(
    pro: Any,
    frozen_rows: list[dict[str, object]],
    expected: dict[str, str],
) -> pd.DataFrame:
    classification = pro.index_classify(level="L2", src="SW2021").copy()
    required = {
        "index_code",
        "industry_name",
        "level",
        "industry_code",
        "is_pub",
        "parent_code",
        "src",
    }
    missing = required - set(classification.columns)
    if missing:
        raise ValueError(f"Tushare L2 classification missing columns: {sorted(missing)}")
    classification["index_code"] = classification["index_code"].astype(str)
    classification["is_pub"] = pd.to_numeric(classification["is_pub"], errors="raise").astype(int)
    failures = {
        "classification_rows": len(classification) != EXPECTED_CLASSIFICATION_ROWS,
        "duplicate_codes": bool(classification["index_code"].duplicated().any()),
        "unexpected_levels": set(classification["level"].astype(str)) != {"L2"},
        "unexpected_sources": set(classification["src"].astype(str)) != {"SW2021"},
        "published_industries": int((classification["is_pub"] == 1).sum())
        != EXPECTED_PUBLISHED_INDUSTRIES,
        "published_parent_industries": classification.loc[
            classification["is_pub"] == 1, "parent_code"
        ].nunique()
        != EXPECTED_PARENT_INDUSTRIES,
    }
    blocking = [name for name, failed in failures.items() if failed]
    if blocking:
        raise ValueError(f"Tushare SW2021 L2 classification contract differs: {blocking}")
    identity_columns = ("index_code", "industry_name", "industry_code", "parent_code", "is_pub")
    observed_identity = classification.loc[:, identity_columns].copy()
    for column in identity_columns[:-1]:
        observed_identity[column] = observed_identity[column].astype(str)
    frozen_identity = pd.DataFrame(frozen_rows).loc[:, identity_columns].copy()
    for column in identity_columns[:-1]:
        frozen_identity[column] = frozen_identity[column].astype(str)
    frozen_identity["is_pub"] = pd.to_numeric(frozen_identity["is_pub"]).astype(int)
    observed_records = observed_identity.sort_values("index_code").to_dict("records")
    frozen_records = frozen_identity.sort_values("index_code").to_dict("records")
    if observed_records != frozen_records:
        raise ValueError("Tushare SW2021 L2 classification differs from frozen identity config")
    observed_published = classification.loc[classification["is_pub"] == 1, "index_code"]
    if set(observed_published) != set(expected):
        raise ValueError("Tushare published L2 code set differs from frozen identity config")
    classification["requested_source"] = "SW2021"
    return classification.sort_values("index_code").reset_index(drop=True)


def fetch_quotes(
    pro: Any,
    expected: dict[str, str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    frames = []
    for code in sorted(expected):
        frame = pro.sw_daily(
            ts_code=code,
            start_date=start_date,
            end_date=end_date,
            fields=",".join(QUOTE_FIELDS),
        ).copy()
        if len(frame) >= 4000:
            raise RuntimeError(f"Tushare row limit may have truncated {code}; split the request")
        if frame.empty:
            raise RuntimeError(f"Tushare returned no rows for published L2 code {code}")
        frames.append(frame)
        print(f"downloaded {code} {expected[code]}: {len(frame)} rows")
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


def validate_quotes(
    quotes: pd.DataFrame,
    expected: dict[str, str],
    calendar: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> dict[str, Any]:
    expected_codes = set(expected)
    observed_codes = set(quotes["ts_code"])
    open_dates = sorted(
        calendar.loc[
            (pd.to_numeric(calendar["is_open"]) == 1)
            & (calendar["cal_date"] >= start_date)
            & (calendar["cal_date"] <= end_date),
            "cal_date",
        ].astype(str)
    )
    counts = quotes.groupby("trade_date")["ts_code"].nunique().reindex(open_dates)
    partial_dates = counts[(counts.notna()) & (counts != EXPECTED_PUBLISHED_INDUSTRIES)]
    missing_dates = counts[counts.isna()]
    extra_dates = sorted(set(quotes["trade_date"]) - set(open_dates))
    prices = quotes[["open", "high", "low", "close"]].to_numpy(dtype=np.float64)
    flow = quotes[["vol", "amount"]].to_numpy(dtype=np.float64)
    invalid_ohlc = quotes[
        (quotes["high"] < quotes[["open", "low", "close"]].max(axis=1))
        | (quotes["low"] > quotes[["open", "high", "close"]].min(axis=1))
    ]
    failures = {
        "duplicate_key_rows": int(quotes.duplicated(["ts_code", "trade_date"]).sum()),
        "missing_expected_codes": len(expected_codes - observed_codes),
        "unexpected_codes": len(observed_codes - expected_codes),
        "missing_open_dates": len(missing_dates),
        "partial_open_dates": len(partial_dates),
        "quotes_on_closed_dates": len(extra_dates),
        "nonfinite_price_rows": int((~np.isfinite(prices)).any(axis=1).sum()),
        "nonpositive_price_rows": int((prices <= 0).any(axis=1).sum()),
        "nonfinite_flow_rows": int((~np.isfinite(flow)).any(axis=1).sum()),
        "nonpositive_flow_rows": int((flow <= 0).any(axis=1).sum()),
        "invalid_ohlc_structure_rows": len(invalid_ohlc),
    }
    blocking = {key: value for key, value in failures.items() if value}
    if blocking:
        raise ValueError(f"SW2021 L2 quote quality gate failed: {blocking}")
    if str(quotes["trade_date"].min()) != start_date:
        raise ValueError("Observed first date differs from requested complete-panel start")
    if str(quotes["trade_date"].max()) != end_date:
        raise ValueError("Observed last date differs from latest SSE open date")
    source_name_counts = quotes.groupby("ts_code")["source_name"].nunique(dropna=False)
    return {
        "rows": len(quotes),
        "industries": len(expected_codes),
        "trading_days": len(open_dates),
        "first_date": str(quotes["trade_date"].min()),
        "last_date": str(quotes["trade_date"].max()),
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
        "close_only_research_gate_passed": True,
    }


def main() -> None:
    args = parse_args()
    captured_at = datetime.now(timezone.utc)
    frozen_rows, expected = load_identity(args.identity_config)
    pro = ts.pro_api()
    calendar, latest_open_date = fetch_calendar(pro, args.start_date, args.end_date)
    classification = fetch_classification(pro, frozen_rows, expected)
    quotes = fetch_quotes(pro, expected, args.start_date, latest_open_date)
    quality = validate_quotes(quotes, expected, calendar, args.start_date, latest_open_date)

    timestamp = captured_at.strftime("%Y%m%dT%H%M%SZ")
    snapshot_id = args.snapshot_id or (
        f"sw2021-l2-current-history-{args.start_date}-{latest_open_date}-{timestamp}"
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    final_dir = args.output_root / snapshot_id
    if final_dir.exists():
        raise FileExistsError(f"Snapshot already exists: {final_dir}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{snapshot_id}.", dir=args.output_root))
    try:
        quote_path = temporary / "sw2021_l2_daily.parquet"
        classification_path = temporary / "sw2021_l2_classification.parquet"
        calendar_path = temporary / "sse_trade_calendar.parquet"
        quotes.to_parquet(quote_path, index=False)
        classification.to_parquet(classification_path, index=False)
        calendar.to_parquet(calendar_path, index=False)
        data_files = (quote_path, classification_path, calendar_path)
        for path in data_files:
            fsync_file(path)

        unpublished = classification.loc[classification["is_pub"] == 0]
        manifest = {
            "schema_version": "sw2021-l2-current-history-snapshot-v1",
            "snapshot_id": snapshot_id,
            "captured_at_utc": captured_at.isoformat(),
            "source": "Tushare",
            "classification_endpoint": {
                "api": "index_classify",
                "params": {"level": "L2", "src": "SW2021"},
            },
            "quote_endpoint": {
                "api": "sw_daily",
                "request_strategy": "one request per currently published SW2021 L2 code",
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
            "classification_rows": len(classification),
            "published_industry_count": len(expected),
            "unpublished_industry_count": len(unpublished),
            "excluded_unpublished_codes": unpublished["index_code"].astype(str).sort_values().tolist(),
            "series_semantics": "PUBLISHED_INDEX_LEVEL_VENDOR_CURRENT",
            "historical_vintage_proven": False,
            "complete_cross_section_start": args.start_date,
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