#!/usr/bin/env python3
"""Build an audited research table from an immutable SW2021 L2 snapshot."""

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
DEFAULT_IDENTITY_CONFIG = ROOT / "configs/sw_industry_l2_sw2021_identity.json"
CHANGE_TOLERANCE = 0.02
PCT_CHANGE_TOLERANCE_PP = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--identity-config", type=Path, default=DEFAULT_IDENTITY_CONFIG)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "data/processed/sw_industry_l2/sw2021_backcast/governed_snapshots",
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


def verify_checksum_file(snapshot: Path) -> None:
    checksum_path = snapshot / "SHA256SUMS"
    if not checksum_path.exists():
        raise FileNotFoundError(f"Missing source checksums: {checksum_path}")
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        expected_hash, filename = line.split("  ", 1)
        path = snapshot / filename
        if not path.exists() or sha256(path) != expected_hash:
            raise RuntimeError(f"Source checksum mismatch: {filename}")


def load_identity(path: Path) -> tuple[list[dict[str, object]], set[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "sw2021-l2-identity-v1":
        raise RuntimeError("Unexpected SW2021 L2 identity schema")
    rows = [
        *payload.get("published_industries", []),
        *payload.get("excluded_unpublished_industries", []),
    ]
    published_codes = {
        str(row["index_code"]) for row in payload.get("published_industries", [])
    }
    if len(rows) != 134 or len(published_codes) != 124:
        raise RuntimeError("SW2021 L2 identity counts differ")
    return rows, published_codes


def verify_source(
    snapshot: Path,
    identity_config: Path,
) -> tuple[dict[str, Any], Path, Path, Path, set[str]]:
    manifest_path = snapshot / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing source manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "sw2021-l2-current-history-snapshot-v1":
        raise RuntimeError("Unexpected source snapshot schema")
    verify_checksum_file(snapshot)
    if manifest.get("identity_config_sha256") != sha256(identity_config):
        raise RuntimeError("Source snapshot identity config SHA mismatch")
    if manifest.get("identity_policy") != "CURRENTLY_PUBLISHED_SW2021_L2_ONLY":
        raise RuntimeError("Source snapshot identity policy differs")
    quality = manifest.get("quality", {})
    if quality.get("blocking_quality_failures") or not quality.get(
        "close_only_research_gate_passed"
    ):
        raise RuntimeError("Source snapshot quality gate is not clean")
    paths = {
        name: snapshot / name
        for name in (
            "sw2021_l2_daily.parquet",
            "sw2021_l2_classification.parquet",
            "sse_trade_calendar.parquet",
        )
    }
    for name, path in paths.items():
        expected_hash = manifest.get("files", {}).get(name, {}).get("sha256")
        if not path.exists() or sha256(path) != expected_hash:
            raise RuntimeError(f"Source file hash mismatch: {name}")
    frozen_rows, published_codes = load_identity(identity_config)
    classification = pd.read_parquet(paths["sw2021_l2_classification.parquet"])
    identity_columns = ("index_code", "industry_name", "industry_code", "parent_code", "is_pub")
    observed = classification.loc[:, identity_columns].copy()
    frozen = pd.DataFrame(frozen_rows).loc[:, identity_columns].copy()
    for frame in (observed, frozen):
        for column in identity_columns[:-1]:
            frame[column] = frame[column].astype(str)
        frame["is_pub"] = pd.to_numeric(frame["is_pub"]).astype(int)
    if observed.sort_values("index_code").to_dict("records") != frozen.sort_values(
        "index_code"
    ).to_dict("records"):
        raise RuntimeError("Source classification differs from frozen identity config")
    return (
        manifest,
        paths["sw2021_l2_daily.parquet"],
        paths["sw2021_l2_classification.parquet"],
        paths["sse_trade_calendar.parquet"],
        published_codes,
    )


def verify_source_table(
    source: pd.DataFrame,
    calendar: pd.DataFrame,
    manifest: dict[str, Any],
    published_codes: set[str],
) -> None:
    source_codes = set(source["ts_code"].astype(str))
    source_dates = set(source["trade_date"].astype(str))
    open_dates = set(
        calendar.loc[
            (pd.to_numeric(calendar["is_open"]) == 1)
            & (calendar["cal_date"].astype(str) >= manifest["requested_start_date"])
            & (calendar["cal_date"].astype(str) <= manifest["data_as_of_trade_date"]),
            "cal_date",
        ].astype(str)
    )
    failures = {
        "published_code_set": source_codes != published_codes,
        "SSE_open_date_set": source_dates != open_dates,
        "row_count": len(source) != len(published_codes) * len(open_dates),
        "duplicate_keys": bool(source.duplicated(["ts_code", "trade_date"]).any()),
        "manifest_rows": len(source) != manifest.get("quality", {}).get("rows"),
        "manifest_industries": len(source_codes)
        != manifest.get("quality", {}).get("industries"),
        "manifest_trading_days": len(source_dates)
        != manifest.get("quality", {}).get("trading_days"),
    }
    blocking = [name for name, failed in failures.items() if failed]
    if blocking:
        raise RuntimeError(f"Source snapshot table contract differs: {blocking}")


def build_governed(
    source: pd.DataFrame,
    source_snapshot_id: str,
    *,
    require_rectangular: bool = True,
    require_global_date_adjacency_for_returns: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    frame = source.sort_values(["ts_code", "trade_date"]).reset_index(drop=True).copy()
    repairs: list[dict[str, Any]] = []

    high_floor = frame[["open", "low", "close"]].max(axis=1)
    high_bad = frame["high"] < high_floor
    low_ceiling = frame[["open", "high", "close"]].min(axis=1)
    low_bad = frame["low"] > low_ceiling
    for field, mask, governed in (
        ("high", high_bad, high_floor),
        ("low", low_bad, low_ceiling),
    ):
        for index in frame.index[mask]:
            repairs.append(
                {
                    "source_snapshot_id": source_snapshot_id,
                    "ts_code": frame.at[index, "ts_code"],
                    "trade_date": frame.at[index, "trade_date"],
                    "field": field,
                    "source_value": float(frame.at[index, field]),
                    "governed_value": float(governed.at[index]),
                    "reason": f"{field}_violates_ohlc_envelope",
                }
            )
        frame.loc[mask, field] = governed[mask]

    frame["source_change"] = frame["change"]
    frame["source_pct_change"] = frame["pct_change"]
    previous_close = frame.groupby("ts_code")["close"].shift(1)
    if require_global_date_adjacency_for_returns:
        date_positions = {
            str(date): index for index, date in enumerate(sorted(frame["trade_date"].unique()))
        }
        current_position = frame["trade_date"].astype(str).map(date_positions)
        previous_position = current_position.groupby(frame["ts_code"]).shift(1)
        previous_close = previous_close.where(current_position - previous_position == 1)
    frame["close_change_computed"] = frame["close"] - previous_close
    frame["close_pct_change_computed"] = frame["close_change_computed"] / previous_close * 100.0
    valid_previous = previous_close.notna()
    change_diff = (frame["source_change"] - frame["close_change_computed"]).abs()
    pct_diff = (frame["source_pct_change"] - frame["close_pct_change_computed"]).abs()
    missing_source_return = frame["source_change"].isna() | frame["source_pct_change"].isna()
    return_bad = valid_previous & (
        missing_source_return
        | (change_diff > CHANGE_TOLERANCE)
        | (pct_diff > PCT_CHANGE_TOLERANCE_PP)
    )
    for index in frame.index[return_bad]:
        for field, source_field, computed_field in (
            ("change", "source_change", "close_change_computed"),
            ("pct_change", "source_pct_change", "close_pct_change_computed"),
        ):
            repairs.append(
                {
                    "source_snapshot_id": source_snapshot_id,
                    "ts_code": frame.at[index, "ts_code"],
                    "trade_date": frame.at[index, "trade_date"],
                    "field": field,
                    "source_value": (
                        np.nan
                        if pd.isna(frame.at[index, source_field])
                        else float(frame.at[index, source_field])
                    ),
                    "governed_value": float(frame.at[index, computed_field]),
                    "reason": (
                        "source_return_missing"
                        if pd.isna(frame.at[index, source_field])
                        else "source_return_differs_from_consecutive_close"
                    ),
                }
            )
    frame.loc[return_bad, "change"] = frame.loc[return_bad, "close_change_computed"]
    frame.loc[return_bad, "pct_change"] = frame.loc[return_bad, "close_pct_change_computed"]
    frame["ohlc_repaired"] = high_bad | low_bad
    frame["return_fields_repaired"] = return_bad

    prices = frame[["open", "high", "low", "close"]].to_numpy(dtype=np.float64)
    dates_per_code = frame.groupby("ts_code")["trade_date"].nunique()
    blocking_failures = {
        "duplicate_key_rows": int(frame.duplicated(["ts_code", "trade_date"]).sum()),
        "non_rectangular_codes": (
            int((dates_per_code != frame["trade_date"].nunique()).sum())
            if require_rectangular
            else 0
        ),
        "nonfinite_price_rows": int((~np.isfinite(prices)).any(axis=1).sum()),
        "nonpositive_price_rows": int((prices <= 0).any(axis=1).sum()),
        "high_envelope_violations_after_repair": int(
            (frame["high"] < frame[["open", "low", "close"]].max(axis=1)).sum()
        ),
        "low_envelope_violations_after_repair": int(
            (frame["low"] > frame[["open", "high", "close"]].min(axis=1)).sum()
        ),
        "missing_governed_return_fields_with_previous_close": int(
            (
                valid_previous
                & (frame["change"].isna() | frame["pct_change"].isna())
            ).sum()
        ),
    }
    blocking = {key: value for key, value in blocking_failures.items() if value}
    if blocking:
        raise ValueError(f"Governed SW2021 L2 quality gate failed: {blocking}")

    governed_change_diff = (frame["change"] - frame["close_change_computed"]).abs()
    governed_pct_diff = (frame["pct_change"] - frame["close_pct_change_computed"]).abs()
    source_name_counts = frame.groupby("ts_code")["source_name"].nunique(dropna=False)
    boundary = frame[frame["trade_date"].isin(["20211210", "20211213"])].pivot(
        index="ts_code", columns="trade_date", values="close"
    ).dropna()
    boundary_dates = {"20211210", "20211213"}
    if boundary_dates.issubset(boundary.columns):
        boundary_returns = boundary["20211213"] / boundary["20211210"] - 1.0
    else:
        boundary_returns = pd.Series(dtype=np.float64)
    close_returns = frame["close_pct_change_computed"] / 100.0
    quality = {
        "rows": len(frame),
        "industries": int(frame["ts_code"].nunique()),
        "trading_days": int(frame["trade_date"].nunique()),
        "first_date": str(frame["trade_date"].min()),
        "last_date": str(frame["trade_date"].max()),
        "rectangular_panel_required": require_rectangular,
        "global_date_adjacency_required_for_returns": (
            require_global_date_adjacency_for_returns
        ),
        "repairs": len(repairs),
        "high_repairs": int(high_bad.sum()),
        "low_repairs": int(low_bad.sum()),
        "return_field_rows_repaired": int(return_bad.sum()),
        "return_field_values_repaired": int(return_bad.sum()) * 2,
        "codes_with_multiple_source_names": int((source_name_counts > 1).sum()),
        "blocking_quality_failures": blocking,
        "source_change_abs_diff_gt_0_02_rows": int((valid_previous & (change_diff > CHANGE_TOLERANCE)).sum()),
        "source_pct_change_abs_diff_gt_0_01pp_rows": int(
            (valid_previous & (pct_diff > PCT_CHANGE_TOLERANCE_PP)).sum()
        ),
        "governed_change_abs_diff_gt_0_02_rows": int(
            (valid_previous & (governed_change_diff > CHANGE_TOLERANCE)).sum()
        ),
        "governed_pct_change_abs_diff_gt_0_01pp_rows": int(
            (valid_previous & (governed_pct_diff > PCT_CHANGE_TOLERANCE_PP)).sum()
        ),
        "max_abs_computed_close_return": float(close_returns.abs().max()),
        "computed_close_returns_abs_gt_15pct": int((close_returns.abs() > 0.15).sum()),
        "SW2021_boundary_code_count": len(boundary_returns),
        "SW2021_boundary_return_min": (
            float(boundary_returns.min()) if len(boundary_returns) else None
        ),
        "SW2021_boundary_return_median": (
            float(boundary_returns.median()) if len(boundary_returns) else None
        ),
        "SW2021_boundary_return_max": (
            float(boundary_returns.max()) if len(boundary_returns) else None
        ),
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
    (
        source_manifest,
        source_quote_path,
        _classification_path,
        calendar_path,
        published_codes,
    ) = verify_source(source_snapshot, args.identity_config)
    source = pd.read_parquet(source_quote_path)
    calendar = pd.read_parquet(calendar_path)
    verify_source_table(source, calendar, source_manifest, published_codes)
    governed, audit, quality = build_governed(source, source_manifest["snapshot_id"])

    governed_id = f"governed-v1-{source_manifest['snapshot_id']}"
    args.output_root.mkdir(parents=True, exist_ok=True)
    final_dir = args.output_root / governed_id
    if final_dir.exists():
        raise FileExistsError(f"Governed snapshot already exists: {final_dir}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{governed_id}.", dir=args.output_root))
    try:
        data_path = temporary / "sw2021_l2_daily_governed.parquet"
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
            "schema_version": "sw2021-l2-governed-current-history-v1",
            "governed_id": governed_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_snapshot": project_path(source_snapshot),
            "source_snapshot_id": source_manifest["snapshot_id"],
            "source_manifest_sha256": sha256(source_snapshot / "manifest.json"),
            "source_quote_sha256": sha256(source_quote_path),
            "identity_config": project_path(args.identity_config),
            "identity_config_sha256": sha256(args.identity_config),
            "identity_policy": source_manifest["identity_policy"],
            "series_semantics": source_manifest["series_semantics"],
            "historical_vintage_proven": source_manifest["historical_vintage_proven"],
            "transformations": [
                "repair source OHLC envelope violations while retaining an audit record",
                "add close_change_computed and close_pct_change_computed from consecutive close levels",
                "preserve original change and pct_change as source_change and source_pct_change",
                "replace materially inconsistent source return fields using fixed tolerances",
            ],
            "quality": quality,
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