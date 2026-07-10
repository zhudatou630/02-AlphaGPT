#!/usr/bin/env python3
"""Build independent ETF event-adjusted data and quality audit artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alpha_etf.config import ETF_UNIVERSE
from alpha_etf.data.event_adjustment import (
    PANEL_FEATURES,
    anomaly_rows,
    apply_event_adjustments,
    build_event_audit,
    panel_arrays,
    quality_gate_failures,
    quality_summary,
    reconcile_flow_data,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-daily",
        type=Path,
        default=ROOT / "data" / "raw" / "etf_daily_bfq.parquet",
    )
    parser.add_argument(
        "--events",
        type=Path,
        default=ROOT / "data" / "raw" / "etf_gbbq_events.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "processed" / "phase1b_event_adjusted",
    )
    parser.add_argument("--jump-threshold", type=float, default=0.20)
    parser.add_argument(
        "--tushare-daily-probe",
        type=Path,
        default=ROOT
        / "data"
        / "processed"
        / "data_quality"
        / "tushare_probe"
        / "tushare_fund_daily_probe.parquet",
    )
    parser.add_argument(
        "--tushare-probe",
        type=Path,
        default=ROOT
        / "data"
        / "processed"
        / "data_quality"
        / "tushare_probe"
        / "tushare_merged_adjusted_probe.parquet",
    )
    return parser.parse_args()


def _write_tushare_comparison(adjusted: pd.DataFrame, probe_path: Path, output_dir: Path) -> None:
    if not probe_path.exists():
        return
    probe = pd.read_parquet(probe_path)
    required = {"symbol", "trade_date", "adj_close_ret"}
    if not required.issubset(probe.columns):
        return
    probe = probe.copy()
    probe["symbol"] = probe["symbol"].astype(str)
    probe["date"] = pd.to_datetime(probe["trade_date"]).dt.normalize()
    comparison = adjusted[
        ["symbol", "date", "raw_close_return", "event_qfq_close_return", "applied_event_count"]
    ].merge(
        probe[["symbol", "date", "raw_close_ret", "adj_close_ret", "adj_factor"]],
        on=["symbol", "date"],
        how="inner",
    )
    comparison = comparison.rename(
        columns={
            "raw_close_ret": "tushare_raw_close_return",
            "adj_close_ret": "tushare_adjusted_close_return",
            "adj_factor": "tushare_adj_factor",
        }
    )
    comparison["tdx_tushare_adjusted_return_diff"] = (
        comparison["event_qfq_close_return"] - comparison["tushare_adjusted_close_return"]
    )
    comparison.to_csv(output_dir / "tushare_return_comparison.csv", index=False)
    valid = comparison.dropna(
        subset=["event_qfq_close_return", "tushare_adjusted_close_return"]
    ).copy()
    summary = (
        valid.groupby("symbol")
        .agg(
            overlap_rows=("symbol", "size"),
            mean_abs_adjusted_return_diff=(
                "tdx_tushare_adjusted_return_diff",
                lambda values: float(values.abs().mean()),
            ),
            max_abs_adjusted_return_diff=(
                "tdx_tushare_adjusted_return_diff",
                lambda values: float(values.abs().max()),
            ),
            diff_gt_1pct=(
                "tdx_tushare_adjusted_return_diff",
                lambda values: int((values.abs() > 0.01).sum()),
            ),
            diff_gt_5pct=(
                "tdx_tushare_adjusted_return_diff",
                lambda values: int((values.abs() > 0.05).sum()),
            ),
        )
        .reset_index()
    )
    summary.to_csv(output_dir / "tushare_return_comparison_summary.csv", index=False)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_parquet(args.raw_daily)
    events = pd.read_parquet(args.events)
    tushare_daily = pd.read_parquet(args.tushare_daily_probe) if args.tushare_daily_probe.exists() else None
    reconciled = reconcile_flow_data(raw, tushare_daily)

    event_audit = build_event_audit(reconciled, events)
    adjusted = apply_event_adjustments(reconciled, event_audit)
    quality = quality_summary(adjusted, event_audit, threshold=args.jump_threshold)
    anomalies = anomaly_rows(adjusted, threshold=args.jump_threshold)
    symbols = [etf.symbol for etf in ETF_UNIVERSE]
    values, mask, dates = panel_arrays(adjusted, symbols)
    gate_failures = quality_gate_failures(quality, event_audit)
    summary = {
        "raw_daily": str(args.raw_daily),
        "events": str(args.events),
        "symbols": len(symbols),
        "dates": len(dates),
        "rows": len(adjusted),
        "tradable_rows": int(adjusted["tradable"].astype(bool).sum()),
        "flow_source_counts": {
            str(source): int(count) for source, count in adjusted["flow_source"].value_counts().items()
        },
        "applied_events": int(event_audit["applied"].astype(bool).sum()),
        "share_events": int(
            event_audit.loc[event_audit["applied"].astype(bool), "category_code"].isin([11, 12]).sum()
        ),
        "jump_threshold": args.jump_threshold,
        "quality_gate_failures": gate_failures,
        "panel_shape": list(values.shape),
        "output_dir": str(args.output_dir),
    }
    if gate_failures:
        raise ValueError(f"Event-adjusted quality gate failed: {summary}")

    adjusted.to_parquet(args.output_dir / "etf_daily_event_adjusted.parquet", index=False)
    event_audit.to_csv(args.output_dir / "etf_adjustment_events.csv", index=False)
    quality.to_csv(args.output_dir / "etf_adjustment_quality_summary.csv", index=False)
    anomalies.to_csv(args.output_dir / "etf_adjustment_anomalies.csv", index=False)
    np.savez_compressed(
        args.output_dir / "etf_panel_event_adjusted.npz",
        values=values,
        mask=mask,
        symbols=np.array(symbols),
        features=np.array(PANEL_FEATURES),
        dates=dates.strftime("%Y-%m-%d").to_numpy(dtype=str),
        price_adjustment=np.array("multiplicative_event_qfq"),
        volume_adjustment=np.array("share_unit_adjusted"),
        amount_adjustment=np.array("raw_currency_turnover"),
    )
    _write_tushare_comparison(adjusted, args.tushare_probe, args.output_dir)
    (args.output_dir / "build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()