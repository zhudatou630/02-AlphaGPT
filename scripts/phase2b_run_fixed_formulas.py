#!/usr/bin/env python3
"""Run Phase 2B fixed formula probes with revised validator diagnostics."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alpha_etf.formulas import fixed_formulas
from alpha_etf.panel import load_market_panel
from alpha_etf.scoring import ScorerConfig, score_signal
from alpha_etf.validation import ValidatorConfig, run_validator


OUT_DIR = ROOT / "data" / "processed" / "phase2b"
HORIZONS = (10, 5)
TRANSACTION_COST_BPS = 5.0
GROUP_COLS = ["formula", "horizon", "validator_variant", "max_holding_days", "transaction_cost_bps"]


def _validator_configs(horizon: int) -> tuple[ValidatorConfig, ...]:
    return (
        ValidatorConfig(
            horizon=horizon,
            validator_variant="no_max_holding",
            max_holding_days=None,
            transaction_cost_bps=TRANSACTION_COST_BPS,
        ),
        ValidatorConfig(
            horizon=horizon,
            validator_variant=f"max_holding_{horizon}",
            max_holding_days=horizon,
            transaction_cost_bps=TRANSACTION_COST_BPS,
        ),
    )


def _annual_summary(validator_df: pd.DataFrame) -> pd.DataFrame:
    if validator_df.empty:
        return pd.DataFrame()

    df = validator_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year

    rows: list[dict[str, object]] = []
    for keys, group in df.groupby([*GROUP_COLS, "year"], dropna=False):
        group = group.sort_values("date")
        equity = group["equity"].astype(float)
        returns = equity.pct_change().fillna(0.0)
        ret_std = returns.std(ddof=0)
        benchmark = group["benchmark_equity"].astype(float)
        drawdown = equity / equity.cummax() - 1.0
        years = max(len(group) / 252.0, 1e-9)

        row = dict(zip([*GROUP_COLS, "year"], keys, strict=True))
        row.update(
            {
                "days": int(len(group)),
                "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1.0),
                "ann_return": float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0),
                "sharpe": np.nan if ret_std == 0 else float(returns.mean() / ret_std * np.sqrt(252.0)),
                "max_drawdown": float(drawdown.min()),
                "benchmark_total_return": float(benchmark.iloc[-1] / benchmark.iloc[0] - 1.0),
            }
        )
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["formula", "horizon", "validator_variant", "year"])


def _trade_distribution(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame()

    sells = trades_df.loc[trades_df["action"] == "SELL"].copy()
    if sells.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for keys, group in sells.groupby(GROUP_COLS, dropna=False):
        pnl = group["pnl_pct"].astype(float)
        holding_days = group["holding_days"].astype(float)
        row = dict(zip(GROUP_COLS, keys, strict=True))
        row.update(
            {
                "sell_count": int(len(group)),
                "win_rate": float((pnl > 0).mean()),
                "mean_pnl_pct": float(pnl.mean()),
                "median_pnl_pct": float(pnl.median()),
                "p05_pnl_pct": float(pnl.quantile(0.05)),
                "p95_pnl_pct": float(pnl.quantile(0.95)),
                "mean_holding_days": float(holding_days.mean()),
                "median_holding_days": float(holding_days.median()),
                "total_pnl_cash": float(group["pnl_cash"].astype(float).sum()),
            }
        )
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["formula", "horizon", "validator_variant"])


def _tail_attribution(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame()

    sells = trades_df.loc[trades_df["action"] == "SELL"].copy()
    if sells.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for keys, group in sells.groupby(GROUP_COLS, dropna=False):
        group = group.sort_values("pnl_pct")
        tail_size = max(1, int(np.ceil(len(group) * 0.10)))
        tails = (
            ("left_tail", group.head(tail_size)),
            ("right_tail", group.tail(tail_size)),
        )
        for tail_name, tail_group in tails:
            for (symbol, reason), item in tail_group.groupby(["symbol", "reason"], dropna=False):
                row = dict(zip(GROUP_COLS, keys, strict=True))
                row.update(
                    {
                        "tail": tail_name,
                        "symbol": symbol,
                        "reason": reason,
                        "trade_count": int(len(item)),
                        "mean_pnl_pct": float(item["pnl_pct"].astype(float).mean()),
                        "total_pnl_cash": float(item["pnl_cash"].astype(float).sum()),
                        "mean_holding_days": float(item["holding_days"].astype(float).mean()),
                    }
                )
                rows.append(row)

    return pd.DataFrame(rows).sort_values(["formula", "horizon", "validator_variant", "tail", "symbol"])


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
            scorer_frames.append(scorer_daily)

            for validator_config in _validator_configs(horizon):
                validator_daily, trades, validator_summary = run_validator(
                    formula=formula_name,
                    signal=signal,
                    open_prices=open_prices,
                    close_prices=close_prices,
                    mask=panel.mask,
                    dates=panel.dates,
                    symbols=panel.symbols,
                    config=validator_config,
                )

                validator_frames.append(validator_daily)
                trade_frames.append(trades)
                summaries.append({**scorer_summary, **validator_summary})

    summary_df = pd.DataFrame(summaries).sort_values(["formula", "horizon", "validator_variant"])
    scorer_df = pd.concat(scorer_frames, ignore_index=True) if scorer_frames else pd.DataFrame()
    validator_df = pd.concat(validator_frames, ignore_index=True) if validator_frames else pd.DataFrame()
    trades_df = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()

    annual_df = _annual_summary(validator_df)
    distribution_df = _trade_distribution(trades_df)
    tail_df = _tail_attribution(trades_df)

    paths = {
        "summary": OUT_DIR / "formula_summary.csv",
        "scorer_daily": OUT_DIR / "scorer_daily.csv",
        "validator_daily": OUT_DIR / "validator_daily.csv",
        "trades": OUT_DIR / "trades.csv",
        "annual_summary": OUT_DIR / "annual_summary.csv",
        "trade_distribution": OUT_DIR / "trade_distribution.csv",
        "tail_attribution": OUT_DIR / "tail_attribution.csv",
    }

    summary_df.to_csv(paths["summary"], index=False)
    scorer_df.to_csv(paths["scorer_daily"], index=False)
    validator_df.to_csv(paths["validator_daily"], index=False)
    trades_df.to_csv(paths["trades"], index=False)
    annual_df.to_csv(paths["annual_summary"], index=False)
    distribution_df.to_csv(paths["trade_distribution"], index=False)
    tail_df.to_csv(paths["tail_attribution"], index=False)

    print(f"summary: {paths['summary']}")
    print(f"scorer_daily: {paths['scorer_daily']} ({len(scorer_df)} rows)")
    print(f"validator_daily: {paths['validator_daily']} ({len(validator_df)} rows)")
    print(f"trades: {paths['trades']} ({len(trades_df)} rows)")
    print(f"annual_summary: {paths['annual_summary']} ({len(annual_df)} rows)")
    print(f"trade_distribution: {paths['trade_distribution']} ({len(distribution_df)} rows)")
    print(f"tail_attribution: {paths['tail_attribution']} ({len(tail_df)} rows)")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
