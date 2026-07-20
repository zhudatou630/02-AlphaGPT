#!/usr/bin/env python3
"""Build a corrected, audited research table from an immutable SW2021 snapshot."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "data/processed/sw_industry_l1/sw2021_backcast/governed_snapshots",
    )
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


def verify_source(snapshot: Path) -> tuple[dict[str, Any], Path]:
    manifest_path = snapshot / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing source manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "sw2021-l1-current-history-snapshot-v1":
        raise RuntimeError("Unexpected source snapshot schema")
    quote_path = snapshot / "sw2021_l1_daily.parquet"
    expected_hash = manifest.get("files", {}).get(quote_path.name, {}).get("sha256")
    actual_hash = sha256(quote_path)
    if actual_hash != expected_hash:
        raise RuntimeError(f"Source quote hash mismatch: {actual_hash} != {expected_hash}")
    return manifest, quote_path


def build_governed(
    source: pd.DataFrame,
    source_snapshot_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    frame = source.sort_values(["ts_code", "trade_date"]).reset_index(drop=True).copy()
    repairs: list[dict[str, Any]] = []

    high_floor = frame[["open", "low", "close"]].max(axis=1)
    high_bad = frame["high"] < high_floor
    for index in frame.index[high_bad]:
        repairs.append(
            {
                "source_snapshot_id": source_snapshot_id,
                "ts_code": frame.at[index, "ts_code"],
                "trade_date": frame.at[index, "trade_date"],
                "field": "high",
                "source_value": float(frame.at[index, "high"]),
                "governed_value": float(high_floor.at[index]),
                "reason": "high_below_open_low_or_close",
            }
        )
    frame.loc[high_bad, "high"] = high_floor[high_bad]

    low_ceiling = frame[["open", "high", "close"]].min(axis=1)
    low_bad = frame["low"] > low_ceiling
    for index in frame.index[low_bad]:
        repairs.append(
            {
                "source_snapshot_id": source_snapshot_id,
                "ts_code": frame.at[index, "ts_code"],
                "trade_date": frame.at[index, "trade_date"],
                "field": "low",
                "source_value": float(frame.at[index, "low"]),
                "governed_value": float(low_ceiling.at[index]),
                "reason": "low_above_open_high_or_close",
            }
        )
    frame.loc[low_bad, "low"] = low_ceiling[low_bad]

    frame["source_change"] = frame["change"]
    frame["source_pct_change"] = frame["pct_change"]
    previous_close = frame.groupby("ts_code")["close"].shift(1)
    frame["close_change_computed"] = frame["close"] - previous_close
    frame["close_pct_change_computed"] = (
        frame["close_change_computed"] / previous_close * 100.0
    )
    frame["ohlc_repaired"] = high_bad | low_bad

    valid_previous = previous_close.notna()
    source_change_diff = (
        frame.loc[valid_previous, "source_change"]
        - frame.loc[valid_previous, "close_change_computed"]
    ).abs()
    source_pct_diff = (
        frame.loc[valid_previous, "source_pct_change"]
        - frame.loc[valid_previous, "close_pct_change_computed"]
    ).abs()
    material_pct = frame.loc[valid_previous].loc[source_pct_diff > 0.01]
    material_dates = {
        str(date): int(count)
        for date, count in material_pct.groupby("trade_date").size().items()
    }
    known_return_bad = (frame["trade_date"] == "20171010") & (
        (frame["source_pct_change"] - frame["close_pct_change_computed"]).abs() > 0.01
    )
    if int(known_return_bad.sum()) != 27:
        raise ValueError(
            "Expected 27 known source return-field errors on 2017-10-10, found "
            f"{int(known_return_bad.sum())}"
        )
    for index in frame.index[known_return_bad]:
        repairs.extend(
            [
                {
                    "source_snapshot_id": source_snapshot_id,
                    "ts_code": frame.at[index, "ts_code"],
                    "trade_date": frame.at[index, "trade_date"],
                    "field": "change",
                    "source_value": float(frame.at[index, "source_change"]),
                    "governed_value": float(frame.at[index, "close_change_computed"]),
                    "reason": "source_change_used_stale_previous_close",
                },
                {
                    "source_snapshot_id": source_snapshot_id,
                    "ts_code": frame.at[index, "ts_code"],
                    "trade_date": frame.at[index, "trade_date"],
                    "field": "pct_change",
                    "source_value": float(frame.at[index, "source_pct_change"]),
                    "governed_value": float(frame.at[index, "close_pct_change_computed"]),
                    "reason": "source_pct_change_used_stale_previous_close",
                },
            ]
        )
    frame.loc[known_return_bad, "change"] = frame.loc[
        known_return_bad, "close_change_computed"
    ]
    frame.loc[known_return_bad, "pct_change"] = frame.loc[
        known_return_bad, "close_pct_change_computed"
    ]
    frame["return_fields_repaired"] = known_return_bad
    governed_change_diff = (
        frame.loc[valid_previous, "change"]
        - frame.loc[valid_previous, "close_change_computed"]
    ).abs()
    governed_pct_diff = (
        frame.loc[valid_previous, "pct_change"]
        - frame.loc[valid_previous, "close_pct_change_computed"]
    ).abs()

    prices = frame[["open", "high", "low", "close"]].to_numpy(dtype=np.float64)
    blocking_failures = {
        "duplicate_key_rows": int(frame.duplicated(["ts_code", "trade_date"]).sum()),
        "nonfinite_price_rows": int((~np.isfinite(prices)).any(axis=1).sum()),
        "nonpositive_price_rows": int((prices <= 0).any(axis=1).sum()),
        "high_envelope_violations_after_repair": int(
            (frame["high"] < frame[["open", "low", "close"]].max(axis=1)).sum()
        ),
        "low_envelope_violations_after_repair": int(
            (frame["low"] > frame[["open", "high", "close"]].min(axis=1)).sum()
        ),
    }
    blocking = {key: value for key, value in blocking_failures.items() if value}
    if blocking:
        raise ValueError(f"Governed SW2021 quality gate failed: {blocking}")

    close_returns = frame["close_pct_change_computed"] / 100.0
    boundary = frame[frame["trade_date"].isin(["20211210", "20211213"])].pivot(
        index="ts_code", columns="trade_date", values="close"
    ).dropna()
    boundary_returns = boundary["20211213"] / boundary["20211210"] - 1.0
    quality = {
        "rows": len(frame),
        "industries": int(frame["ts_code"].nunique()),
        "trading_days": int(frame["trade_date"].nunique()),
        "first_date": str(frame["trade_date"].min()),
        "last_date": str(frame["trade_date"].max()),
        "repairs": len(repairs),
        "high_repairs": int(high_bad.sum()),
        "low_repairs": int(low_bad.sum()),
        "return_field_rows_repaired": int(known_return_bad.sum()),
        "return_field_values_repaired": int(known_return_bad.sum()) * 2,
        "blocking_quality_failures": blocking,
        "source_change_abs_diff_gt_0_02_rows": int((source_change_diff > 0.02).sum()),
        "source_pct_change_abs_diff_gt_0_01pp_rows": int((source_pct_diff > 0.01).sum()),
        "source_pct_change_material_mismatch_dates": material_dates,
        "source_pct_change_policy": (
            "preserved in source_change and source_pct_change; known material errors are "
            "corrected in change and pct_change"
        ),
        "governed_change_abs_diff_gt_0_02_rows": int((governed_change_diff > 0.02).sum()),
        "governed_pct_change_abs_diff_gt_0_01pp_rows": int((governed_pct_diff > 0.01).sum()),
        "max_abs_computed_close_return": float(close_returns.abs().max()),
        "computed_close_returns_abs_gt_15pct": int((close_returns.abs() > 0.15).sum()),
        "SW2021_boundary_return_min": float(boundary_returns.min()),
        "SW2021_boundary_return_median": float(boundary_returns.median()),
        "SW2021_boundary_return_max": float(boundary_returns.max()),
        "missing_values": {
            column: int(count)
            for column, count in frame.isna().sum().items()
            if count
        },
        "close_research_gate_passed": True,
    }
    audit = pd.DataFrame(
        repairs,
        columns=(
            "source_snapshot_id",
            "ts_code",
            "trade_date",
            "field",
            "source_value",
            "governed_value",
            "reason",
        ),
    )
    return frame.sort_values(["trade_date", "ts_code"]).reset_index(drop=True), audit, quality


def main() -> None:
    args = parse_args()
    source_snapshot = args.source_snapshot.resolve()
    source_manifest, source_quote_path = verify_source(source_snapshot)
    source = pd.read_parquet(source_quote_path)
    governed, audit, quality = build_governed(source, source_manifest["snapshot_id"])

    governed_id = f"governed-v2-{source_manifest['snapshot_id']}"
    args.output_root.mkdir(parents=True, exist_ok=True)
    final_dir = args.output_root / governed_id
    if final_dir.exists():
        raise FileExistsError(f"Governed snapshot already exists: {final_dir}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{governed_id}.", dir=args.output_root))
    try:
        data_path = temporary / "sw2021_l1_daily_governed.parquet"
        audit_path = temporary / "repair_audit.csv"
        quality_path = temporary / "quality_report.json"
        governed.to_parquet(data_path, index=False)
        audit.to_csv(audit_path, index=False)
        fsync_file(data_path)
        fsync_file(audit_path)
        write_text_synced(
            quality_path,
            json.dumps(quality, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        )

        output_files = (data_path, audit_path, quality_path)
        manifest = {
            "schema_version": "sw2021-l1-governed-current-history-v2",
            "governed_id": governed_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_snapshot": project_path(source_snapshot),
            "source_snapshot_id": source_manifest["snapshot_id"],
            "source_manifest_sha256": sha256(source_snapshot / "manifest.json"),
            "source_quote_sha256": sha256(source_quote_path),
            "series_semantics": source_manifest["series_semantics"],
            "historical_vintage_proven": source_manifest["historical_vintage_proven"],
            "transformations": [
                "raise high to max(open, low, close) when the source high violates the OHLC envelope",
                "lower low to min(open, high, close) when the source low violates the OHLC envelope",
                "add close_change_computed and close_pct_change_computed from consecutive close levels",
                "preserve original change and pct_change as source_change and source_pct_change",
                "replace the 27 stale-baseline change and pct_change values on 2017-10-10",
            ],
            "quality": quality,
            "files": {
                path.name: {
                    "sha256": sha256(path),
                    "size_bytes": path.stat().st_size,
                }
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