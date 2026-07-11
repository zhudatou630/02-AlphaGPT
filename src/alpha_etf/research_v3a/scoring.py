"""Causal reference scorer for the V3A single-formula objective."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


SCORER_VERSION = "etf-v3a-scorer-v1"


@dataclass(frozen=True)
class ScorerConfig:
    horizon: int = 10
    top_fraction: float = 0.20
    min_top_k: int = 2
    max_top_k: int = 4
    min_universe: int = 10
    hard_invalid_reward: float = -5.0
    ranking_dtype: str = "float32"

    def to_dict(self) -> dict[str, Any]:
        return {"version": SCORER_VERSION, **asdict(self)}


@dataclass(frozen=True)
class SplitSpec:
    name: str
    start: str
    end: str | None


@dataclass(frozen=True)
class ForwardTargets:
    split: SplitSpec
    decision_indices: np.ndarray
    available: np.ndarray
    available_count: np.ndarray
    top_k: np.ndarray
    buy_indices: np.ndarray
    planned_sell_indices: np.ndarray
    actual_sell_indices: np.ndarray
    forward_returns: np.ndarray
    baseline_returns: np.ndarray

    @property
    def days(self) -> int:
        return int(len(self.decision_indices))


@dataclass(frozen=True)
class ScoreResult:
    valid: bool
    reward: float
    invalid_reason: str
    daily: pd.DataFrame
    summary: dict[str, Any]
    audit: dict[str, Any]


def _top_k(count: int, config: ScorerConfig) -> int:
    raw = int(np.ceil(count * config.top_fraction))
    return min(config.max_top_k, max(config.min_top_k, raw))


def _date_bounds(dates: pd.DatetimeIndex, split: SplitSpec) -> tuple[int, int]:
    start = pd.Timestamp(split.start).normalize()
    end = dates[-1] if split.end is None else pd.Timestamp(split.end).normalize()
    candidates = np.where((dates >= start) & (dates <= end))[0]
    if len(candidates) == 0:
        raise ValueError(f"Split {split.name} has no dates in panel")
    return int(candidates[0]), int(candidates[-1])


def build_forward_targets(
    open_prices: np.ndarray,
    mask: np.ndarray,
    dates: pd.DatetimeIndex,
    split: SplitSpec,
    config: ScorerConfig,
) -> ForwardTargets:
    if open_prices.shape != mask.shape:
        raise ValueError(f"open_prices and mask differ: {open_prices.shape} != {mask.shape}")
    if open_prices.shape[1] != len(dates):
        raise ValueError("open_prices date dimension differs from dates")
    if config.horizon < 1:
        raise ValueError("horizon must be positive")

    start, end = _date_bounds(dates, split)
    decision_indices: list[int] = []
    available_rows: list[np.ndarray] = []
    available_counts: list[int] = []
    top_ks: list[int] = []
    buy_rows: list[np.ndarray] = []
    planned_sells: list[int] = []
    actual_sell_rows: list[np.ndarray] = []
    return_rows: list[np.ndarray] = []
    baseline_returns: list[float] = []
    assets = mask.shape[0]

    for decision in range(start, end + 1):
        available = mask[:, decision].astype(bool)
        count = int(available.sum())
        if count < config.min_universe:
            continue
        buy = decision + 1
        planned_sell = buy + config.horizon
        if buy > end:
            continue

        buy_indices = np.full(assets, -1, dtype=np.int64)
        actual_sells = np.full(assets, -1, dtype=np.int64)
        returns = np.full(assets, np.nan, dtype=np.float64)
        complete = True
        for asset in np.where(available)[0]:
            buyable = (
                mask[asset, buy]
                and np.isfinite(open_prices[asset, buy])
                and open_prices[asset, buy] > 0
            )
            if not buyable:
                returns[asset] = 0.0
                continue
            if planned_sell > end:
                complete = False
                break
            executable = np.where(
                mask[asset, planned_sell : end + 1]
                & np.isfinite(open_prices[asset, planned_sell : end + 1])
                & (open_prices[asset, planned_sell : end + 1] > 0)
            )[0]
            if len(executable) == 0:
                complete = False
                break
            sell = planned_sell + int(executable[0])
            buy_indices[asset] = buy
            actual_sells[asset] = sell
            returns[asset] = open_prices[asset, sell] / open_prices[asset, buy] - 1.0
        if not complete:
            continue
        if not np.isfinite(returns[available]).all():
            raise RuntimeError(f"Incomplete V3A forward returns at {dates[decision]}")

        decision_indices.append(decision)
        available_rows.append(available)
        available_counts.append(count)
        top_ks.append(_top_k(count, config))
        buy_rows.append(buy_indices)
        planned_sells.append(planned_sell)
        actual_sell_rows.append(actual_sells)
        return_rows.append(returns)
        baseline_returns.append(float(np.mean(returns[available])))

    if not decision_indices:
        raise ValueError(f"Split {split.name} has no complete V3A scorer dates")
    return ForwardTargets(
        split=split,
        decision_indices=np.asarray(decision_indices, dtype=np.int64),
        available=np.stack(available_rows),
        available_count=np.asarray(available_counts, dtype=np.int64),
        top_k=np.asarray(top_ks, dtype=np.int64),
        buy_indices=np.stack(buy_rows),
        planned_sell_indices=np.asarray(planned_sells, dtype=np.int64),
        actual_sell_indices=np.stack(actual_sell_rows),
        forward_returns=np.stack(return_rows),
        baseline_returns=np.asarray(baseline_returns, dtype=np.float64),
    )


def _rank_desc(signal: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    indices = np.where(eligible)[0]
    return indices[np.argsort(-signal[indices], kind="stable")]


def _average_ranks(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average").to_numpy(dtype=float)


def _rank_ic(signal: np.ndarray, returns: np.ndarray) -> float:
    if len(signal) < 2:
        return np.nan
    signal_rank = _average_ranks(signal)
    return_rank = _average_ranks(returns)
    signal_std = float(np.std(signal_rank, ddof=0))
    return_std = float(np.std(return_rank, ddof=0))
    if signal_std == 0.0 or return_std == 0.0:
        return np.nan
    return float(np.corrcoef(signal_rank, return_rank)[0, 1])


def _drawdown_audit(daily: pd.DataFrame, targets: ForwardTargets, horizon: int) -> dict[str, Any]:
    if daily.empty:
        return {
            "offset_total_return_median": np.nan,
            "offset_total_return_worst": np.nan,
            "offset_max_drawdown_median": np.nan,
            "offset_max_drawdown_worst": np.nan,
            "offset_count": 0,
        }
    totals: list[float] = []
    drawdowns: list[float] = []
    for offset in range(horizon):
        equity = 1.0
        peak = 1.0
        max_drawdown = 0.0
        previous_exit = -1
        used = 0
        for position in range(offset, len(daily), horizon):
            row = daily.iloc[position]
            target_row = int(row["target_row"])
            buy = int(targets.decision_indices[target_row]) + 1
            if buy <= previous_exit:
                continue
            selected = tuple(int(item) for item in row["selected_indices"])
            exits = targets.actual_sell_indices[target_row, list(selected)]
            actual_exits = exits[exits >= 0]
            previous_exit = int(actual_exits.max()) if len(actual_exits) else buy
            equity *= 1.0 + float(row["absolute_return"])
            peak = max(peak, equity)
            max_drawdown = min(max_drawdown, equity / peak - 1.0)
            used += 1
        if used:
            totals.append(equity - 1.0)
            drawdowns.append(max_drawdown)
    return {
        "offset_total_return_median": float(np.median(totals)) if totals else np.nan,
        "offset_total_return_worst": float(np.min(totals)) if totals else np.nan,
        "offset_max_drawdown_median": float(np.median(drawdowns)) if drawdowns else np.nan,
        "offset_max_drawdown_worst": float(np.min(drawdowns)) if drawdowns else np.nan,
        "offset_count": len(totals),
    }


def score_signal(
    formula_id: str,
    signal: np.ndarray,
    targets: ForwardTargets,
    dates: pd.DatetimeIndex,
    symbols: np.ndarray,
    config: ScorerConfig,
) -> ScoreResult:
    if signal.shape != (len(symbols), len(dates)):
        raise ValueError(f"Signal shape differs from panel: {signal.shape}")
    rows: list[dict[str, Any]] = []
    if config.ranking_dtype != "float32":
        raise ValueError(f"Unsupported V3A ranking dtype: {config.ranking_dtype}")
    for target_row, decision in enumerate(targets.decision_indices):
        available = targets.available[target_row]
        ranking_signal = signal[:, decision].astype(np.float32)
        eligible = available & np.isfinite(ranking_signal)
        k = int(targets.top_k[target_row])
        if int(eligible.sum()) < k:
            summary = {
                "formula_id": formula_id,
                "split": targets.split.name,
                "scorer_days": targets.days,
                "failed_decision_index": int(decision),
            }
            return ScoreResult(
                valid=False,
                reward=config.hard_invalid_reward,
                invalid_reason="insufficient_daily_signal",
                daily=pd.DataFrame(),
                summary=summary,
                audit={},
            )
        ordered = _rank_desc(ranking_signal, eligible)
        selected = ordered[:k]
        selected_returns = targets.forward_returns[target_row, selected]
        absolute_return = float(np.mean(selected_returns))
        baseline_return = float(targets.baseline_returns[target_row])
        excess_return = absolute_return - baseline_return
        ic_eligible = available & np.isfinite(ranking_signal)
        rank_ic = (
            _rank_ic(
                ranking_signal[ic_eligible], targets.forward_returns[target_row, ic_eligible]
            )
            if int(ic_eligible.sum()) >= config.min_universe
            else np.nan
        )
        rows.append(
            {
                "formula_id": formula_id,
                "split": targets.split.name,
                "target_row": target_row,
                "decision_index": int(decision),
                "date": dates[decision].date().isoformat(),
                "buy_date": dates[decision + 1].date().isoformat(),
                "planned_sell_date": (
                    dates[targets.planned_sell_indices[target_row]].date().isoformat()
                    if targets.planned_sell_indices[target_row] < len(dates)
                    else "cash"
                ),
                "available_count": int(targets.available_count[target_row]),
                "top_k": k,
                "selected_indices": tuple(int(item) for item in selected),
                "selected_symbols": ";".join(symbols[selected].astype(str)),
                "selected_returns": tuple(float(item) for item in selected_returns),
                "actual_sell_dates": tuple(
                    dates[index].date().isoformat() if index >= 0 else "cash"
                    for index in targets.actual_sell_indices[target_row, selected]
                ),
                "absolute_return": absolute_return,
                "baseline_return": baseline_return,
                "excess_return": excess_return,
                "rank_ic": rank_ic,
            }
        )

    daily = pd.DataFrame(rows)
    reward = float(daily["excess_return"].mean())
    rank_ic_values = daily["rank_ic"].astype(float).dropna()
    rank_ic_std = float(rank_ic_values.std(ddof=0)) if len(rank_ic_values) else np.nan
    summary = {
        "formula_id": formula_id,
        "split": targets.split.name,
        "scorer_days": int(len(daily)),
        "reward": reward,
        "mean_excess_return": reward,
        "mean_absolute_return": float(daily["absolute_return"].mean()),
        "median_absolute_return": float(daily["absolute_return"].median()),
        "absolute_hit_rate": float((daily["absolute_return"] > 0).mean()),
        "mean_baseline_return": float(daily["baseline_return"].mean()),
        "rank_ic_mean": float(rank_ic_values.mean()) if len(rank_ic_values) else np.nan,
        "rank_ic_std": rank_ic_std,
        "rank_ic_ir": (
            float(rank_ic_values.mean() / rank_ic_std)
            if len(rank_ic_values) and rank_ic_std > 0
            else np.nan
        ),
        "rank_ic_positive_rate": (
            float((rank_ic_values > 0).mean()) if len(rank_ic_values) else np.nan
        ),
    }
    audit = _drawdown_audit(daily, targets, config.horizon)
    return ScoreResult(True, reward, "", daily, summary, audit)