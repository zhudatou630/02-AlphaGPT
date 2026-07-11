#!/usr/bin/env python3
"""Build the governed, price-only V3 relative-OHLC research panel."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from alpha_etf.data.event_adjustment import (
    anomaly_rows,
    apply_event_adjustments,
    build_event_audit,
    quality_gate_failures,
    quality_summary,
    reconcile_flow_data,
)
from alpha_etf.data.relative_price import RELATIVE_FEATURES, add_relative_ohlc, relative_panel, relative_quality


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", type=Path, default=ROOT / "configs/research_v3_universe.json")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw/v3")
    parser.add_argument("--probe-dir", type=Path, default=ROOT / "data/processed/v3/data_quality")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed/v3/governance")
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "data/processed/v3/dataset")
    parser.add_argument("--jump-threshold", type=float, default=0.201)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tushare_comparison(adjusted: pd.DataFrame, daily: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    probe = daily[["symbol", "trade_date", "close"]].merge(
        factors[["symbol", "trade_date", "adj_factor"]], on=["symbol", "trade_date"], how="inner"
    )
    probe["date"] = pd.to_datetime(probe["trade_date"]).dt.normalize()
    probe = probe.sort_values(["symbol", "date"])
    probe["tushare_adjusted_close"] = probe["close"] * probe["adj_factor"]
    probe["tushare_adjusted_return"] = probe.groupby("symbol")["tushare_adjusted_close"].pct_change(
        fill_method=None
    )
    comparison = adjusted[["symbol", "date", "event_qfq_close_return", "applied_event_count"]].merge(
        probe[["symbol", "date", "tushare_adjusted_return"]], on=["symbol", "date"], how="inner"
    )
    comparison["adjusted_return_diff"] = (
        comparison["event_qfq_close_return"] - comparison["tushare_adjusted_return"]
    )
    return comparison


def main() -> None:
    args = parse_args()
    universe = json.loads(args.universe.read_text(encoding="utf-8"))
    symbols = [str(item["symbol"]) for item in universe]
    raw = pd.read_parquet(args.raw_dir / "etf_daily_bfq.parquet")
    events = pd.read_parquet(args.raw_dir / "etf_gbbq_events.parquet")
    tushare_daily = pd.read_parquet(args.probe_dir / "tushare_fund_daily.parquet")
    tushare_adj = pd.read_parquet(args.probe_dir / "tushare_fund_adj.parquet")
    tushare_basic = pd.read_parquet(args.probe_dir / "tushare_fund_basic.parquet")

    reconciled = reconcile_flow_data(raw, tushare_daily)
    event_audit = build_event_audit(reconciled, events)
    adjusted = apply_event_adjustments(reconciled, event_audit)
    relative = add_relative_ohlc(adjusted)
    values, mask, dates = relative_panel(relative, symbols)
    quality = quality_summary(adjusted, event_audit, threshold=args.jump_threshold)
    failures = quality_gate_failures(quality, event_audit)
    failures.update({key: value for key, value in relative_quality(relative).items() if value})

    comparison = tushare_comparison(adjusted, tushare_daily, tushare_adj)
    unexplained_cross_source = comparison[
        (comparison["adjusted_return_diff"].abs() > 0.01)
        & (comparison["applied_event_count"] == 0)
    ]
    if len(unexplained_cross_source):
        failures["unexplained_tdx_tushare_return_diff_gt_1pct"] = len(unexplained_cross_source)

    basic_dates = tushare_basic.set_index("symbol")["list_date"].dropna().astype(str)
    first_dates = raw.groupby("symbol")["date"].min().dt.strftime("%Y%m%d")
    pre_listing = [symbol for symbol in symbols if first_dates[symbol] < basic_dates[symbol]]
    if pre_listing:
        failures["symbols_with_pre_listing_rows"] = len(pre_listing)
    if failures:
        raise ValueError(f"V3 quality gate failed: {failures}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.dataset_dir.mkdir(parents=True, exist_ok=True)
    adjusted.to_parquet(args.output_dir / "etf_daily_event_adjusted.parquet", index=False)
    event_audit.to_csv(args.output_dir / "event_audit.csv", index=False)
    quality.to_csv(args.output_dir / "quality_summary.csv", index=False)
    anomaly_rows(adjusted, threshold=args.jump_threshold).to_csv(
        args.output_dir / "adjusted_return_anomalies.csv", index=False
    )
    comparison.to_csv(args.output_dir / "tushare_return_comparison.csv", index=False)
    np.savez_compressed(
        args.dataset_dir / "panel_relative_price_v3.npz",
        values=values,
        mask=mask,
        symbols=np.array(symbols),
        features=np.array(RELATIVE_FEATURES),
        dates=dates.strftime("%Y-%m-%d").to_numpy(dtype=str),
        price_adjustment=np.array("multiplicative_event_qfq"),
        feature_transform=np.array("ohlc_div_previous_adjusted_close_minus_one"),
    )
    panel_path = args.dataset_dir / "panel_relative_price_v3.npz"
    applied = event_audit[event_audit["applied"].astype(bool)]
    summary = {
        "research_version": "v3",
        "symbols": len(symbols),
        "features": list(RELATIVE_FEATURES),
        "flow_features_exposed": False,
        "dates": len(dates),
        "first_date": dates.min().date().isoformat(),
        "last_date": dates.max().date().isoformat(),
        "panel_shape": list(values.shape),
        "tradable_observations": int(mask.sum()),
        "applied_events": len(applied),
        "share_events": int(applied["category_code"].isin([11, 12]).sum()),
        "nontradable_rows": int((~relative["tradable"].astype(bool)).sum()),
        "cross_source_overlap": len(comparison),
        "cross_source_diff_gt_1pct": int((comparison["adjusted_return_diff"].abs() > 0.01).sum()),
        "unexplained_cross_source_diff_gt_1pct": len(unexplained_cross_source),
        "quality_gate_failures": failures,
        "panel_sha256": sha256(panel_path),
    }
    manifest_payload = json.dumps(summary, ensure_ascii=False, sort_keys=True).encode("utf-8")
    summary["dataset_id"] = "etf-relative-price-v3-" + hashlib.sha256(manifest_payload).hexdigest()[:12]
    (args.dataset_dir / "dataset_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()