#!/usr/bin/env python3
"""Run Phase 2 fixed formula probes through scorer and validator."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alpha_etf.formulas import fixed_formulas
from alpha_etf.panel import load_market_panel
from alpha_etf.scoring import ScorerConfig, score_signal
from alpha_etf.validation import ValidatorConfig, run_validator


OUT_DIR = ROOT / "data" / "processed" / "phase2a"
HORIZONS = (10, 5)



def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    panel = load_market_panel()
    open_prices = panel.qfq("open")
    close_prices = panel.qfq("close")

    scorer_frames = []
    validator_frames = []
    trade_frames = []
    summaries = []

    for formula_name, formula_fn in fixed_formulas().items():
        signal = formula_fn(panel)
        for horizon in HORIZONS:
            scorer_daily, scorer_summary = score_signal(
                formula=formula_name,
                signal=signal,
                open_prices=open_prices,
                mask=panel.mask,
                dates=panel.dates,
                symbols=panel.symbols,
                config=ScorerConfig(horizon=horizon),
            )
            validator_daily, trades, validator_summary = run_validator(
                formula=formula_name,
                signal=signal,
                open_prices=open_prices,
                close_prices=close_prices,
                mask=panel.mask,
                dates=panel.dates,
                symbols=panel.symbols,
                config=ValidatorConfig(
                    horizon=horizon,
                    validator_variant=f"max_holding_{horizon}",
                    max_holding_days=horizon,
                    transaction_cost_bps=0.0,
                ),
            )

            scorer_frames.append(scorer_daily)
            validator_frames.append(validator_daily)
            trade_frames.append(trades)
            summaries.append({**scorer_summary, **validator_summary})

    summary_df = pd.DataFrame(summaries).sort_values(["formula", "horizon"])
    scorer_df = pd.concat(scorer_frames, ignore_index=True) if scorer_frames else pd.DataFrame()
    validator_df = pd.concat(validator_frames, ignore_index=True) if validator_frames else pd.DataFrame()
    trades_df = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()

    summary_path = OUT_DIR / "formula_summary.csv"
    scorer_path = OUT_DIR / "scorer_daily.csv"
    validator_path = OUT_DIR / "validator_daily.csv"
    trades_path = OUT_DIR / "trades.csv"

    summary_df.to_csv(summary_path, index=False)
    scorer_df.to_csv(scorer_path, index=False)
    validator_df.to_csv(validator_path, index=False)
    trades_df.to_csv(trades_path, index=False)

    print(f"summary: {summary_path}")
    print(f"scorer_daily: {scorer_path} ({len(scorer_df)} rows)")
    print(f"validator_daily: {validator_path} ({len(validator_df)} rows)")
    print(f"trades: {trades_path} ({len(trades_df)} rows)")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
