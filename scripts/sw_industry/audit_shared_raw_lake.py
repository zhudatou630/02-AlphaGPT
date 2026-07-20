#!/usr/bin/env python3
"""Audit the shared market-data Shenwan lake without modifying it."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from glob import glob
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MARKET_DATA_ROOT = Path("/home/zhujunshen/market-data")
DEFAULT_SIDECAR = "v1_14_sw_levels_raw_20151225_20260604"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market-data-root", type=Path, default=DEFAULT_MARKET_DATA_ROOT)
    parser.add_argument("--sidecar-snapshot", default=DEFAULT_SIDECAR)
    parser.add_argument(
        "--identity-config",
        type=Path,
        default=ROOT / "configs/sw_industry_l1_versions.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--compare-tushare", action="store_true")
    parser.add_argument("--target-end-date", default="20260710")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_partitioned(paths: list[str], columns: list[str]) -> pd.DataFrame:
    return pd.concat(
        [pd.read_parquet(path, columns=columns) for path in sorted(paths)],
        ignore_index=True,
    )


def summarize_current_lake(market_root: Path) -> dict[str, Any]:
    paths = glob(str(market_root / "raw_lake/sw_daily/year=*/part.parquet"))
    daily = read_partitioned(
        paths, ["src", "level", "index_code", "trade_date", "open", "high", "low", "close"]
    )
    groups = []
    for (source, level), frame in daily.groupby(["src", "level"], dropna=False):
        groups.append(
            {
                "src": source,
                "level": level,
                "rows": len(frame),
                "codes": int(frame["index_code"].nunique()),
                "first_date": str(frame["trade_date"].min()),
                "last_date": str(frame["trade_date"].max()),
                "duplicate_key_rows": int(
                    frame.duplicated(["src", "level", "index_code", "trade_date"]).sum()
                ),
                "missing_ohlc_rows": int(
                    frame[["open", "high", "low", "close"]].isna().any(axis=1).sum()
                ),
            }
        )
    meta = pd.read_parquet(market_root / "raw_lake/sw_index_meta/part.parquet")
    meta_counts = {
        str(level): int(frame["index_code"].nunique())
        for level, frame in meta[meta["src"] == "SW2021"].groupby("level")
    }
    return {
        "files": len(paths),
        "rows": len(daily),
        "groups": groups,
        "contains_L1": bool((daily["level"] == "L1").any()),
        "SW2021_meta_code_counts": meta_counts,
    }


def expected_versions(identity: dict[str, Any]) -> dict[str, dict[str, str]]:
    return {
        segment["version"]: dict(segment["industries"])
        for segment in identity["segments"]
    }


def compare_tushare(sidecar_l1: pd.DataFrame) -> dict[str, Any]:
    if not os.environ.get("TUSHARE_TOKEN"):
        raise RuntimeError("TUSHARE_TOKEN is not set")
    import tushare as ts

    columns = ["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount"]
    frames = []
    pro = ts.pro_api()
    for code in sorted(sidecar_l1["index_code"].unique()):
        frames.append(
            pro.sw_daily(
                ts_code=code,
                start_date=str(sidecar_l1["trade_date"].min()),
                end_date=str(sidecar_l1["trade_date"].max()),
                fields=",".join(columns),
            )
        )
    api = pd.concat(frames, ignore_index=True)
    local = sidecar_l1.rename(columns={"index_code": "ts_code"})[columns]
    merged = local.merge(
        api,
        on=["ts_code", "trade_date"],
        how="outer",
        suffixes=("_lake", "_api"),
        indicator=True,
        validate="one_to_one",
    )
    comparisons = {}
    shared = merged[merged["_merge"] == "both"]
    for column in ("open", "high", "low", "close", "vol", "amount"):
        lake = shared[f"{column}_lake"]
        remote = shared[f"{column}_api"]
        delta = (lake - remote).abs()
        null_mismatches = lake.isna() ^ remote.isna()
        comparisons[column] = {
            "max_abs_diff": float(delta.max()),
            "mismatches_gt_1e_9": int(((delta > 1e-9) | null_mismatches).sum()),
            "null_mismatches": int(null_mismatches.sum()),
            "both_null": int((lake.isna() & remote.isna()).sum()),
        }
    return {
        "lake_rows": len(local),
        "api_rows": len(api),
        "shared_rows": len(shared),
        "only_lake": int((merged["_merge"] == "left_only").sum()),
        "only_api": int((merged["_merge"] == "right_only").sum()),
        "columns": comparisons,
        "exact": bool(
            len(local) == len(api) == len(shared)
            and all(item["mismatches_gt_1e_9"] == 0 for item in comparisons.values())
        ),
    }


def catalog_hash_check(
    market_root: Path, snapshot: str, paths: list[Path]
) -> dict[str, Any]:
    database = market_root / "catalog/metadata.sqlite"
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    rows = connection.execute(
        "SELECT file_path, content_sha256 FROM table_manifest WHERE snapshot_id = ? "
        "AND dataset LIKE 'sw_levels_%'",
        (snapshot,),
    ).fetchall()
    connection.close()
    expected = {str(Path(path)): digest for path, digest in rows}
    actual_paths = {str(path) for path in paths}
    results = []
    for path in paths:
        actual = sha256(path)
        results.append(
            {
                "path": str(path),
                "expected_sha256": expected.get(str(path)),
                "actual_sha256": actual,
                "matches_catalog": expected.get(str(path)) == actual,
            }
        )
    return {
        "files": len(results),
        "manifest_files": len(expected),
        "missing_physical_files": sorted(set(expected) - actual_paths),
        "unmanifested_physical_files": sorted(actual_paths - set(expected)),
        "all_match_catalog": bool(
            actual_paths == set(expected) and all(row["matches_catalog"] for row in results)
        ),
        "results": results,
    }


def audit_sidecar(
    market_root: Path,
    snapshot: str,
    identity: dict[str, Any],
    with_tushare: bool,
    target_end_date: str,
) -> dict[str, Any]:
    base = market_root / "raw_lake_versions" / snapshot
    daily_paths = [Path(path) for path in glob(str(base / "sw_levels_sector_daily/year=*/part.parquet"))]
    l1 = read_partitioned(
        [str(path) for path in daily_paths],
        [
            "src",
            "level",
            "index_code",
            "trade_date",
            "open",
            "high",
            "low",
            "close",
            "pct_chg",
            "vol",
            "amount",
        ],
    )
    l1 = l1[l1["level"] == "L1"].copy()
    meta_path = base / "sw_levels_sector_meta/part.parquet"
    member_path = base / "sw_levels_sector_member/part.parquet"
    hierarchy_path = base / "sw_levels_sector_hierarchy/part.parquet"
    meta = pd.read_parquet(meta_path)
    meta_l1 = meta[meta["level"] == "L1"].copy()
    members = pd.read_parquet(member_path)
    members_l1 = members[members["level"] == "L1"].copy()

    versions = expected_versions(identity)
    observed = set(l1["index_code"])
    daily_counts = l1.groupby("trade_date")["index_code"].nunique()
    sw2014 = versions["SW2014"]
    sw2021 = versions["SW2021"]
    sw2014_end = next(
        item["index_effective_to"] for item in identity["segments"] if item["version"] == "SW2014"
    ).replace("-", "")
    sw2021_start = next(
        item["index_effective_from"] for item in identity["segments"] if item["version"] == "SW2021"
    ).replace("-", "")
    pre_effective = l1[l1["trade_date"] < sw2021_start]
    native = l1[l1["trade_date"] >= sw2021_start]
    introduced_codes = set(sw2021) - set(sw2014)
    shared_codes = set(sw2021) & set(sw2014)
    confirmed_backcast = pre_effective[pre_effective["index_code"].isin(introduced_codes)]
    reused_code_pre_effective = pre_effective[pre_effective["index_code"].isin(shared_codes)]

    invalid_ohlc = l1[
        (l1[["open", "high", "low", "close"]] <= 0).any(axis=1)
        | (l1["high"] < l1[["open", "low", "close"]].max(axis=1))
        | (l1["low"] > l1[["open", "high", "close"]].min(axis=1))
    ]
    native_invalid_ohlc = invalid_ohlc[invalid_ohlc["trade_date"] >= sw2021_start]
    target_rows = native[native["trade_date"] == target_end_date]
    source_files = daily_paths + [meta_path, member_path, hierarchy_path]
    hash_check = catalog_hash_check(market_root, snapshot, source_files)
    result: dict[str, Any] = {
        "snapshot": snapshot,
        "source_files": len(source_files),
        "src_values": sorted(str(item) for item in l1["src"].dropna().unique()),
        "rows": len(l1),
        "codes": int(l1["index_code"].nunique()),
        "dates": int(l1["trade_date"].nunique()),
        "first_date": str(l1["trade_date"].min()),
        "last_date": str(l1["trade_date"].max()),
        "min_codes_per_date": int(daily_counts.min()),
        "max_codes_per_date": int(daily_counts.max()),
        "matches_SW2021_code_set": observed == set(sw2021),
        "matches_SW2014_code_set": observed == set(sw2014),
        "SW2014_missing_codes": sorted(set(sw2014) - observed),
        "SW2014_extra_codes": sorted(observed - set(sw2014)),
        "pre_SW2021_effective_rows": len(pre_effective),
        "pre_SW2021_effective_first_date": str(pre_effective["trade_date"].min()),
        "pre_SW2021_effective_last_date": str(pre_effective["trade_date"].max()),
        "confirmed_backcast_introduced_code_rows": len(confirmed_backcast),
        "confirmed_backcast_introduced_codes": sorted(introduced_codes),
        "reused_code_pre_effective_rows_identity_unproven": len(reused_code_pre_effective),
        "reused_code_count": len(shared_codes),
        "native_SW2021_rows": len(native),
        "native_SW2021_first_date": str(native["trade_date"].min()),
        "native_SW2021_last_date": str(native["trade_date"].max()),
        "target_end_date": target_end_date,
        "target_end_present": bool(len(target_rows) > 0),
        "target_end_code_count": int(target_rows["index_code"].nunique()),
        "target_end_covered": bool(
            len(target_rows) > 0 and target_rows["index_code"].nunique() == len(sw2021)
        ),
        "SW2014_overlap_last_date": sw2014_end,
        "duplicate_key_rows": int(
            l1.duplicated(["src", "level", "index_code", "trade_date"]).sum()
        ),
        "missing_ohlc_rows": int(l1[["open", "high", "low", "close"]].isna().any(axis=1).sum()),
        "invalid_ohlc_rows": len(invalid_ohlc),
        "native_SW2021_invalid_ohlc_rows": len(native_invalid_ohlc),
        "invalid_ohlc_examples": invalid_ohlc[
            ["index_code", "trade_date", "open", "high", "low", "close"]
        ].to_dict("records"),
        "nonnull_pct_chg_rows": int(l1["pct_chg"].notna().sum()),
        "meta_L1_rows": len(meta_l1),
        "meta_L1_codes": int(meta_l1["index_code"].nunique()),
        "member_contract": {
            "rows": len(members_l1),
            "active_rows": int((members_l1["is_active"] == 1).sum()),
            "nonnull_out_date_rows": int(members_l1["out_date"].notna().sum()),
            "contract_dates": sorted(
                str(item) for item in members_l1["contract_date"].dropna().unique()
            ),
            "identity": "current_active_snapshot_not_full_history",
        },
        "catalog_hash_check": hash_check,
    }
    if with_tushare:
        result["tushare_comparison"] = compare_tushare(l1)
    comparison_pass = with_tushare and result["tushare_comparison"]["exact"]
    result["quality_gates"] = {
        "SW2021_code_set": result["matches_SW2021_code_set"],
        "unique_keys": result["duplicate_key_rows"] == 0,
        "complete_ohlc": result["missing_ohlc_rows"] == 0,
        "native_SW2021_valid_ohlc_structure": result["native_SW2021_invalid_ohlc_rows"] == 0,
        "catalog_manifest_complete_and_matching": hash_check["all_match_catalog"],
        "tushare_comparison_if_requested": comparison_pass,
        "native_slice_quality_pass": bool(
            result["matches_SW2021_code_set"]
            and result["duplicate_key_rows"] == 0
            and result["missing_ohlc_rows"] == 0
            and result["native_SW2021_invalid_ohlc_rows"] == 0
            and hash_check["all_match_catalog"]
            and comparison_pass
        ),
    }
    return result


def main() -> None:
    args = parse_args()
    identity = json.loads(args.identity_config.read_text(encoding="utf-8"))
    sidecar = audit_sidecar(
        args.market_data_root,
        args.sidecar_snapshot,
        identity,
        args.compare_tushare,
        args.target_end_date,
    )
    result = {
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "market_data_root": str(args.market_data_root),
        "identity_config": str(args.identity_config.resolve().relative_to(ROOT)),
        "identity_config_sha256": sha256(args.identity_config),
        "target_end_date": args.target_end_date,
        "current_raw_lake": summarize_current_lake(args.market_data_root),
        "sidecar": sidecar,
        "decision": {
            "native_panel_primary_source": False,
            "SW2014_native_eligible": False,
            "current_SW2021_native_cache_eligible": bool(
                sidecar["quality_gates"]["native_slice_quality_pass"]
                and sidecar["target_end_covered"]
            ),
            "pre_effective_full_cross_section_native_eligible": False,
            "recommended_role": "local Tushare cache and cross-check, not project primary source",
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result_path = args.output_dir / "result.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "SHA256SUMS").write_text(
        f"{sha256(result_path)}  {result_path.name}\n", encoding="ascii"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()