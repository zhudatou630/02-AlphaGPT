#!/usr/bin/env python3
"""Probe Shenwan level-1 source identity without persisting full quote history."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import tushare as ts


ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_HISTORY_API = "https://www.swsresearch.com/institute-sw/api/index_publish/trend/"
USER_AGENT = "Mozilla/5.0 (compatible; AlphaGPT source-identity probe)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--identity-config",
        type=Path,
        default=ROOT / "configs/sw_industry_l1_versions.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--official-ca-bundle",
        type=Path,
        help="CA bundle that includes the intermediate certificate omitted by the official server",
    )
    return parser.parse_args()


def load_identity(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    versions = [segment["version"] for segment in payload["segments"]]
    if len(versions) != len(set(versions)):
        raise ValueError("Duplicate segment versions")
    for segment in payload["segments"]:
        industries = segment["industries"]
        if len(industries) != segment["industry_count"]:
            raise ValueError(f"Industry count mismatch for {segment['version']}")
        codes = [row[0] for row in industries]
        if len(codes) != len(set(codes)):
            raise ValueError(f"Duplicate codes in {segment['version']}")
    for left, right, boundary in zip(
        payload["segments"][:-1],
        payload["segments"][1:],
        payload["boundaries"],
        strict=True,
    ):
        if boundary["last_old_trade_date"] != left["index_effective_to"]:
            raise ValueError(f"Old boundary date mismatch for {boundary['boundary']}")
        if boundary["first_new_trade_date"] != right["index_effective_from"]:
            raise ValueError(f"New boundary date mismatch for {boundary['boundary']}")
        left_names = dict(left["industries"])
        right_names = dict(right["industries"])
        retired = set(left_names) - set(right_names)
        introduced = set(right_names) - set(left_names)
        renamed = {
            code: [left_names[code], right_names[code]]
            for code in set(left_names) & set(right_names)
            if left_names[code] != right_names[code]
        }
        if set(boundary["retired_codes"]) != retired:
            raise ValueError(f"Retired code mismatch for {boundary['boundary']}")
        if set(boundary["introduced_codes"]) != introduced:
            raise ValueError(f"Introduced code mismatch for {boundary['boundary']}")
        if boundary.get("renamed_same_code", {}) != renamed:
            raise ValueError(f"Renamed code mismatch for {boundary['boundary']}")
    return payload


def official_history_probe(
    session: requests.Session,
    code: str,
    target_dates: set[str],
    endpoint: str,
    verify: bool | str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    response = session.get(
        endpoint,
        params={"swindexcode": code.removesuffix(".SI"), "period": "DAY"},
        timeout=60,
        verify=verify,
    )
    response.raise_for_status()
    rows = response.json().get("data") or []
    dates = [str(row["bargaindate"]) for row in rows]
    summary = {
        "ts_code": code,
        "row_count": len(rows),
        "first_date": min(dates) if dates else None,
        "last_date": max(dates) if dates else None,
    }
    quotes = [
        {
            "ts_code": code,
            "boundary_date": str(row["bargaindate"]),
            "official_close": float(row["closeindex"]),
        }
        for row in rows
        if str(row["bargaindate"]) in target_dates
    ]
    return summary, quotes


def tushare_classifications(pro: Any) -> pd.DataFrame:
    frames = []
    for source in ("SW2014", "SW2021"):
        frame = pro.index_classify(level="L1", src=source).copy()
        frame["requested_source"] = source
        frames.append(frame)
    return pd.concat(frames, ignore_index=True).sort_values(
        ["requested_source", "index_code"]
    )


def verify_classifications(identity: dict[str, Any], frame: pd.DataFrame) -> None:
    expected = {
        segment["version"]: dict(segment["industries"])
        for segment in identity["segments"]
        if segment["version"] in {"SW2014", "SW2021"}
    }
    for source, identities in expected.items():
        source_frame = frame.loc[frame["requested_source"] == source]
        observed = dict(zip(source_frame["index_code"], source_frame["industry_name"]))
        if observed != identities:
            raise ValueError(
                f"Tushare {source} identity mismatch: expected={identities}, observed={observed}"
            )


def boundary_quotes(identity: dict[str, Any], pro: Any) -> pd.DataFrame:
    rows = []
    for segment in identity["segments"]:
        dates = [segment["index_effective_from"], segment["index_effective_to"]]
        active_codes = {row[0] for row in segment["industries"]}
        for date in (item for item in dates if item):
            trade_date = date.replace("-", "")
            frame = pro.sw_daily(trade_date=trade_date)
            observed = frame[frame["ts_code"].isin(active_codes)].copy()
            if observed["ts_code"].duplicated().any():
                raise ValueError(f"Duplicate boundary quotes for {segment['version']} on {date}")
            if set(observed["ts_code"]) != active_codes:
                raise ValueError(
                    f"Boundary code mismatch for {segment['version']} on {date}"
                )
            closes = pd.to_numeric(observed["close"], errors="coerce")
            if not closes.map(lambda value: math.isfinite(value) and value > 0).all():
                raise ValueError(
                    f"Invalid Tushare close for {segment['version']} on {date}"
                )
            for row in observed.itertuples(index=False):
                rows.append(
                    {
                        "version": segment["version"],
                        "boundary_date": date,
                        "ts_code": row.ts_code,
                        "daily_name": row.name,
                        "close": float(row.close),
                    }
                )
            if len(observed) != segment["industry_count"]:
                raise ValueError(
                    f"Boundary coverage mismatch for {segment['version']} on {date}: "
                    f"{len(observed)} != {segment['industry_count']}"
                )
    return pd.DataFrame(rows).sort_values(["boundary_date", "ts_code"])


def write_sha256s(output_dir: Path, paths: list[Path]) -> None:
    lines = []
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.name}")
    (output_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="ascii")


def main() -> None:
    args = parse_args()
    if not os.environ.get("TUSHARE_TOKEN"):
        raise RuntimeError("TUSHARE_TOKEN is not set")
    identity = load_identity(args.identity_config)
    pro = ts.pro_api()
    classifications = tushare_classifications(pro)
    verify_classifications(identity, classifications)
    quotes = boundary_quotes(identity, pro)

    all_codes = sorted(
        {row[0] for segment in identity["segments"] for row in segment["industries"]}
    )
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    endpoint = identity["sources"].get("official_history_api", OFFICIAL_HISTORY_API)
    verify: bool | str = str(args.official_ca_bundle) if args.official_ca_bundle else True
    target_dates = set(quotes["boundary_date"])
    history_rows = []
    official_quote_rows = []
    for code in all_codes:
        history_row, quote_rows = official_history_probe(
            session, code, target_dates, endpoint, verify
        )
        history_rows.append(history_row)
        official_quote_rows.extend(quote_rows)
    history = pd.DataFrame(history_rows).sort_values("ts_code")
    official_quotes = pd.DataFrame(official_quote_rows)
    comparison = quotes.merge(
        official_quotes, on=["ts_code", "boundary_date"], how="left", validate="one_to_one"
    )
    if comparison["official_close"].isna().any():
        missing = comparison.loc[
            comparison["official_close"].isna(), ["ts_code", "boundary_date"]
        ].to_dict("records")
        raise ValueError(f"Official boundary quotes missing: {missing}")
    official_closes = pd.to_numeric(comparison["official_close"], errors="coerce")
    if not official_closes.map(lambda value: math.isfinite(value) and value > 0).all():
        raise ValueError("Official boundary close is non-finite or non-positive")
    comparison["close_abs_diff"] = (
        comparison["close"] - comparison["official_close"]
    ).abs()
    tolerance = 0.01
    mismatches = comparison["close_abs_diff"] > tolerance
    if mismatches.any():
        examples = comparison.loc[
            mismatches, ["ts_code", "boundary_date", "close", "official_close"]
        ].head(10).to_dict("records")
        raise ValueError(f"Cross-source boundary close mismatch: {examples}")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    classification_path = args.output_dir / "tushare_l1_classifications.csv"
    boundary_path = args.output_dir / "tushare_boundary_quotes.csv"
    history_path = args.output_dir / "official_history_identity.csv"
    comparison_path = args.output_dir / "cross_source_boundary_comparison.csv"
    classifications.to_csv(classification_path, index=False)
    quotes.to_csv(boundary_path, index=False)
    history.to_csv(history_path, index=False)
    comparison.to_csv(comparison_path, index=False)

    summary = {
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "identity_config": str(args.identity_config.resolve().relative_to(ROOT)),
        "identity_config_sha256": hashlib.sha256(
            args.identity_config.read_bytes()
        ).hexdigest(),
        "tushare_classification_rows": len(classifications),
        "boundary_quote_rows": len(quotes),
        "official_history_codes": len(history),
        "cross_source_boundary_rows": len(comparison),
        "cross_source_max_close_abs_diff": float(comparison["close_abs_diff"].max()),
        "cross_source_close_tolerance": tolerance,
        "cross_source_close_mismatches_gt_tolerance": int(mismatches.sum()),
        "official_history_endpoint": endpoint,
        "official_history_params": {"period": "DAY", "swindexcode": "per-code"},
        "official_ca_bundle_sha256": (
            hashlib.sha256(args.official_ca_bundle.read_bytes()).hexdigest()
            if args.official_ca_bundle
            else None
        ),
        "tushare_endpoints": {
            "index_classify": {"level": "L1", "src": ["SW2014", "SW2021"]},
            "sw_daily": {"trade_date": sorted(target_dates)},
        },
        "package_versions": {
            name: metadata.version(name) for name in ("pandas", "requests", "tushare")
        },
        "response_columns": {
            "tushare_classification": list(classifications.columns),
            "tushare_boundary_quotes": list(quotes.columns),
            "official_history_summary": list(history.columns),
        },
        "full_quote_history_persisted": False,
        "checks": {
            "SW2014_code_set_exact": True,
            "SW2021_code_set_exact": True,
            "configured_boundary_expected_codes_complete": True,
            "cross_source_close_within_tolerance": True,
        },
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_sha256s(
        args.output_dir,
        [
            classification_path,
            boundary_path,
            history_path,
            comparison_path,
            summary_path,
        ],
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()