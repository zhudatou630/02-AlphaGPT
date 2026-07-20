#!/usr/bin/env python3
"""Govern an immutable SW2021 L2 dynamic-history snapshot."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from build_sw2021_l2_governed import (  # noqa: E402
    DEFAULT_IDENTITY_CONFIG,
    ROOT,
    build_governed,
    fsync_directory,
    fsync_file,
    load_identity,
    project_path,
    sha256,
    verify_checksum_file,
    write_text_synced,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--identity-config", type=Path, default=DEFAULT_IDENTITY_CONFIG)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "data/processed/sw_industry_l2/sw2021_dynamic/governed_snapshots",
    )
    return parser.parse_args()


def verify_dynamic_source(
    snapshot: Path,
    identity_config: Path,
) -> tuple[dict[str, Any], Path, Path, set[str]]:
    manifest_path = snapshot / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing source manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "sw2021-l2-dynamic-history-snapshot-v1":
        raise RuntimeError("Unexpected dynamic source snapshot schema")
    verify_checksum_file(snapshot)
    if manifest.get("identity_config_sha256") != sha256(identity_config):
        raise RuntimeError("Dynamic source identity config SHA mismatch")
    quality = manifest.get("quality", {})
    if quality.get("blocking_quality_failures") or not quality.get(
        "dynamic_close_research_gate_passed"
    ):
        raise RuntimeError("Dynamic source quality gate is not clean")
    paths = {
        name: snapshot / name
        for name in (
            "sw2021_l2_dynamic_daily.parquet",
            "sw2021_l2_classification.parquet",
            "sse_trade_calendar.parquet",
            "availability.csv",
        )
    }
    for name, path in paths.items():
        if not path.exists() or sha256(path) != manifest["files"][name]["sha256"]:
            raise RuntimeError(f"Dynamic source file hash mismatch: {name}")
    frozen_rows, published_codes = load_identity(identity_config)
    identity_columns = ("index_code", "industry_name", "industry_code", "parent_code", "is_pub")
    observed = pd.read_parquet(paths["sw2021_l2_classification.parquet"])[
        list(identity_columns)
    ].copy()
    frozen = pd.DataFrame(frozen_rows)[list(identity_columns)].copy()
    for frame in (observed, frozen):
        for column in identity_columns[:-1]:
            frame[column] = frame[column].astype(str)
        frame["is_pub"] = pd.to_numeric(frame["is_pub"]).astype(int)
    if observed.sort_values("index_code").to_dict("records") != frozen.sort_values(
        "index_code"
    ).to_dict("records"):
        raise RuntimeError("Dynamic source classification differs from frozen identity")
    return manifest, paths["sw2021_l2_dynamic_daily.parquet"], paths["availability.csv"], published_codes


def verify_dynamic_table(
    source: pd.DataFrame,
    availability: pd.DataFrame,
    manifest: dict[str, Any],
    published_codes: set[str],
) -> None:
    source_codes = set(source["ts_code"].astype(str))
    source_dates = set(source["trade_date"].astype(str))
    availability_codes = set(availability["ts_code"].astype(str))
    observed = (
        source.assign(ts_code=source["ts_code"].astype(str), trade_date=source["trade_date"].astype(str))
        .groupby("ts_code")["trade_date"]
        .agg(first_quote_date="min", last_quote_date="max", quote_rows="size")
        .reset_index()
        .sort_values("ts_code")
        .reset_index(drop=True)
    )
    expected = availability[
        ["ts_code", "first_quote_date", "last_quote_date", "quote_rows"]
    ].copy()
    for column in ("ts_code", "first_quote_date", "last_quote_date"):
        expected[column] = expected[column].astype(str)
    expected["quote_rows"] = pd.to_numeric(expected["quote_rows"]).astype(int)
    expected = expected.sort_values("ts_code").reset_index(drop=True)
    failures = {
        "published_code_set": source_codes != published_codes,
        "availability_code_set": availability_codes != published_codes,
        "availability_summary": observed.to_dict("records") != expected.to_dict("records"),
        "duplicate_keys": bool(source.duplicated(["ts_code", "trade_date"]).any()),
        "manifest_rows": len(source) != manifest.get("quality", {}).get("rows"),
        "manifest_quote_dates": len(source_dates)
        != manifest.get("quality", {}).get("quote_trading_days"),
        "manifest_first_date": min(source_dates) != manifest.get("quality", {}).get("first_date"),
        "manifest_last_date": max(source_dates) != manifest.get("quality", {}).get("last_date"),
    }
    blocking = [name for name, failed in failures.items() if failed]
    if blocking:
        raise RuntimeError(f"Dynamic source table contract differs: {blocking}")


def main() -> None:
    args = parse_args()
    source_snapshot = args.source_snapshot.resolve()
    source_manifest, source_quote_path, availability_path, published_codes = (
        verify_dynamic_source(source_snapshot, args.identity_config)
    )
    source = pd.read_parquet(source_quote_path)
    availability = pd.read_csv(availability_path, dtype={"ts_code": str})
    verify_dynamic_table(source, availability, source_manifest, published_codes)
    governed, audit, quality = build_governed(
        source,
        source_manifest["snapshot_id"],
        require_rectangular=False,
        require_global_date_adjacency_for_returns=True,
    )

    governed_id = f"governed-v2-{source_manifest['snapshot_id']}"
    args.output_root.mkdir(parents=True, exist_ok=True)
    final_dir = args.output_root / governed_id
    if final_dir.exists():
        raise FileExistsError(f"Governed snapshot already exists: {final_dir}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{governed_id}.", dir=args.output_root))
    try:
        data_path = temporary / "sw2021_l2_dynamic_daily_governed.parquet"
        audit_path = temporary / "repair_audit.csv"
        quality_path = temporary / "quality_report.json"
        availability_output = temporary / "availability.csv"
        governed.to_parquet(data_path, index=False)
        audit.to_csv(audit_path, index=False)
        availability.to_csv(availability_output, index=False)
        for path in (data_path, audit_path, availability_output):
            fsync_file(path)
        write_text_synced(
            quality_path,
            json.dumps(quality, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        )
        output_files = (data_path, audit_path, quality_path, availability_output)
        manifest = {
            "schema_version": "sw2021-l2-dynamic-governed-history-v2",
            "governed_id": governed_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_snapshot": project_path(source_snapshot),
            "source_snapshot_id": source_manifest["snapshot_id"],
            "source_manifest_sha256": sha256(source_snapshot / "manifest.json"),
            "source_quote_sha256": sha256(source_quote_path),
            "identity_config": project_path(args.identity_config),
            "identity_config_sha256": sha256(args.identity_config),
            "identity_policy": source_manifest["identity_policy"],
            "entry_policy": source_manifest["entry_policy"],
            "series_semantics": source_manifest["series_semantics"],
            "historical_vintage_proven": source_manifest["historical_vintage_proven"],
            "transformations": [
                "repair source OHLC envelope violations while retaining an audit record",
                "compute return fields by code without filling unavailable dates",
                "require consecutive global exchange dates for computed return fields",
                "preserve original change and pct_change as source_change and source_pct_change",
                "replace materially inconsistent or missing source return fields",
            ],
            "quality": quality,
            "source_dynamic_quality": source_manifest["quality"],
            "files": {
                path.name: {"sha256": sha256(path), "size_bytes": path.stat().st_size}
                for path in output_files
            },
        }
        manifest_path = temporary / "manifest.json"
        write_text_synced(
            manifest_path,
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        )
        checksum_paths = (*output_files, manifest_path)
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
    print(f"governed snapshot: {final_dir}")


if __name__ == "__main__":
    main()