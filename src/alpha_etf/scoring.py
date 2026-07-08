"""Pure signal scorer for Phase 2."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ScorerConfig:
    horizon: int
    top_fraction: float = 0.20
    min_top_k: int = 2
    max_top_k: int = 4
    min_universe: int = 10



def _rank_desc(signal_t: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    ranks = np.full(signal_t.shape, np.nan, dtype=float)
    eligible_idx = np.where(eligible)[0]
    if len(eligible_idx) == 0:
        return ranks
    ordered = eligible_idx[np.argsort(-signal_t[eligible_idx], kind="mergesort")]
    ranks[ordered] = np.arange(1, len(ordered) + 1, dtype=float)
    return ranks



def _top_k(count: int, config: ScorerConfig) -> int:
    raw_k = int(np.ceil(count * config.top_fraction))
    return min(config.max_top_k, max(config.min_top_k, raw_k))



def score_signal(
    formula: str,
    signal: np.ndarray,
    open_prices: np.ndarray,
    mask: np.ndarray,
    dates: pd.DatetimeIndex,
    symbols: np.ndarray,
    config: ScorerConfig,
) -> tuple[pd.DataFrame, dict[str, float]]:
    rows: list[dict[str, object]] = []
    horizon = config.horizon
    t_max = signal.shape[1] - horizon - 2

    for t in range(max(t_max, -1) + 1):
        available = mask[:, t]
        available_count = int(available.sum())
        if available_count < config.min_universe:
            continue

        buy_t = t + 1
        sell_t = t + 1 + horizon
        executable = (
            available
            & mask[:, buy_t]
            & mask[:, sell_t]
            & np.isfinite(signal[:, t])
            & np.isfinite(open_prices[:, buy_t])
            & np.isfinite(open_prices[:, sell_t])
            & (open_prices[:, buy_t] > 0)
        )
        eligible_count = int(executable.sum())
        if eligible_count < config.min_top_k:
            continue

        k = min(_top_k(available_count, config), eligible_count)
        ranks = _rank_desc(signal[:, t], executable)
        selected = np.where(ranks <= k)[0]
        forward_returns = open_prices[selected, sell_t] / open_prices[selected, buy_t] - 1.0

        rows.append(
            {
                "formula": formula,
                "horizon": horizon,
                "date": dates[t].date().isoformat(),
                "buy_date": dates[buy_t].date().isoformat(),
                "sell_date": dates[sell_t].date().isoformat(),
                "available_count": available_count,
                "eligible_count": eligible_count,
                "top_k": int(k),
                "selected_symbols": ";".join(symbols[selected].astype(str)),
                "mean_forward_return": float(np.nanmean(forward_returns)),
                "median_forward_return": float(np.nanmedian(forward_returns)),
                "hit_rate": float(np.nanmean(forward_returns > 0)),
            }
        )

    daily = pd.DataFrame(rows)
    summary = summarize_scorer_daily(daily, formula, horizon)
    return daily, summary



def summarize_scorer_daily(daily: pd.DataFrame, formula: str, horizon: int) -> dict[str, float]:
    if daily.empty:
        return {
            "formula": formula,
            "horizon": horizon,
            "scorer_days": 0,
            "scorer_mean_return": np.nan,
            "scorer_median_return": np.nan,
            "scorer_hit_rate": np.nan,
            "scorer_ann_return_proxy": np.nan,
            "avg_top_k": np.nan,
        }

    returns = daily["mean_forward_return"].astype(float)
    return {
        "formula": formula,
        "horizon": horizon,
        "scorer_days": int(len(daily)),
        "scorer_mean_return": float(returns.mean()),
        "scorer_median_return": float(returns.median()),
        "scorer_hit_rate": float((returns > 0).mean()),
        "scorer_ann_return_proxy": float(returns.mean() * 252.0 / horizon),
        "avg_top_k": float(daily["top_k"].astype(float).mean()),
    }
