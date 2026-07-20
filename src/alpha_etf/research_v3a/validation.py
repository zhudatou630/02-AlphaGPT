"""Frozen Stage D validation trading and group-decision semantics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ValidationConfig:
    validation_start: str = "2022-01-01"
    validation_end: str = "2022-12-31"
    initial_cash: float = 1.0
    min_universe: int = 10
    slots: int = 3
    buy_rank: int = 3
    hold_rank: int = 5
    robust_z_threshold: float = 1.5
    stop_loss: float = -0.07
    metrics_scope: str = "validation"


@dataclass
class Position:
    symbol_idx: int
    qty: float
    entry_price: float
    entry_date: str
    entry_decision_date: str
    terminal_cash_received_per_share: float = 0.0


@dataclass(frozen=True)
class ExitOrder:
    slot: int
    reason: str
    decision_date: str


@dataclass(frozen=True)
class EntryOrder:
    symbol_idx: int
    rank: int
    decision_date: str


def rank_and_qualify(
    signal_t: np.ndarray,
    mask_t: np.ndarray,
    symbols: np.ndarray,
    config: ValidationConfig,
) -> tuple[np.ndarray, np.ndarray, bool]:
    ranks = np.full(len(signal_t), np.inf, dtype=np.float64)
    robust_z = np.full(len(signal_t), np.nan, dtype=np.float64)
    eligible = mask_t & np.isfinite(signal_t)
    indices = np.flatnonzero(eligible)
    if len(indices) < config.min_universe:
        return ranks, robust_z, False
    symbol_order = np.argsort(symbols[indices], kind="stable")
    symbol_sorted = indices[symbol_order]
    value_order = np.argsort(-signal_t[symbol_sorted], kind="stable")
    ordered = symbol_sorted[value_order]
    ranks[ordered] = np.arange(1, len(ordered) + 1, dtype=np.float64)
    values = signal_t[indices].astype(np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if np.isfinite(mad) and mad > 0.0:
        robust_z[indices] = (values - median) / (1.4826 * mad)
    return ranks, robust_z, True


def _entry_candidates(
    signal_t: np.ndarray,
    mask_t: np.ndarray,
    symbols: np.ndarray,
    positions: list[Position | None],
    blocked_symbols: set[int],
    config: ValidationConfig,
    *,
    decision_date: str,
) -> tuple[list[EntryOrder], np.ndarray, np.ndarray, bool]:
    ranks, robust_z, rankable = rank_and_qualify(signal_t, mask_t, symbols, config)
    if not rankable:
        return [], ranks, robust_z, False
    held = {position.symbol_idx for position in positions if position is not None}
    candidates = np.flatnonzero(
        (ranks <= config.buy_rank) & (robust_z >= config.robust_z_threshold)
    )
    ordered = sorted(candidates, key=lambda index: (ranks[index], str(symbols[index])))
    entries = [
        EntryOrder(int(index), int(ranks[index]), decision_date)
        for index in ordered
        if int(index) not in held and int(index) not in blocked_symbols
    ]
    return entries, ranks, robust_z, True


def _last_finite_closes_before(
    close_prices: np.ndarray, validation_start_index: int
) -> np.ndarray:
    output = np.full(close_prices.shape[0], np.nan, dtype=np.float64)
    for index in range(validation_start_index):
        values = close_prices[:, index]
        usable = np.isfinite(values) & (values > 0)
        output[usable] = values[usable]
    return output


def _portfolio_value(
    cash: float,
    positions: list[Position | None],
    last_closes: np.ndarray,
) -> float:
    value = float(cash)
    for position in positions:
        if position is None:
            continue
        price = last_closes[position.symbol_idx]
        if not (np.isfinite(price) and price > 0):
            price = position.entry_price
        residual_price = max(
            float(price) - position.terminal_cash_received_per_share, 0.0
        )
        value += position.qty * residual_price
    return float(value)


def _benchmark_path(
    open_prices: np.ndarray,
    close_prices: np.ndarray,
    mask: np.ndarray,
    validation_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    benchmark = np.empty(len(validation_indices), dtype=np.float64)
    buy_hold = np.empty(len(validation_indices), dtype=np.float64)
    first = int(validation_indices[0])
    first_eligible = (
        mask[:, first]
        & np.isfinite(open_prices[:, first])
        & np.isfinite(close_prices[:, first])
        & (open_prices[:, first] > 0)
        & (close_prices[:, first] > 0)
    )
    first_return = (
        float(np.mean(close_prices[first_eligible, first] / open_prices[first_eligible, first] - 1.0))
        if first_eligible.any()
        else 0.0
    )
    benchmark[0] = 1.0 + first_return
    quantities = np.zeros(open_prices.shape[0], dtype=np.float64)
    if first_eligible.any():
        allocation = 1.0 / int(first_eligible.sum())
        quantities[first_eligible] = allocation / open_prices[first_eligible, first]
    last_closes = np.full(open_prices.shape[0], np.nan, dtype=np.float64)
    last_closes[first_eligible] = close_prices[first_eligible, first]
    buy_hold[0] = float(np.nansum(quantities * last_closes))
    for output_index, current in enumerate(validation_indices[1:], start=1):
        previous = int(validation_indices[output_index - 1])
        current = int(current)
        eligible = (
            mask[:, previous]
            & mask[:, current]
            & np.isfinite(close_prices[:, previous])
            & np.isfinite(close_prices[:, current])
            & (close_prices[:, previous] > 0)
            & (close_prices[:, current] > 0)
        )
        daily_return = (
            float(np.mean(close_prices[eligible, current] / close_prices[eligible, previous] - 1.0))
            if eligible.any()
            else 0.0
        )
        benchmark[output_index] = benchmark[output_index - 1] * (1.0 + daily_return)
        current_close = close_prices[:, current]
        usable = np.isfinite(current_close) & (current_close > 0)
        last_closes[usable] = current_close[usable]
        buy_hold[output_index] = float(np.nansum(quantities * last_closes))
    return benchmark, buy_hold


def _max_drawdown(equity: Iterable[float], *, initial: float = 1.0) -> float:
    values = np.r_[float(initial), np.asarray(list(equity), dtype=np.float64)]
    peaks = np.maximum.accumulate(values)
    return float(np.max(1.0 - values / peaks))


def run_formula_validation(
    *,
    formula_id: str,
    signal: np.ndarray,
    open_prices: np.ndarray,
    close_prices: np.ndarray,
    tradable_mask: np.ndarray,
    dates: pd.DatetimeIndex,
    symbols: np.ndarray,
    config: ValidationConfig = ValidationConfig(),
    terminal_cashflows: pd.DataFrame | None = None,
    execution_mask: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run one frozen formula from the prior close through validation year-end."""

    shape = tradable_mask.shape
    execution = tradable_mask if execution_mask is None else execution_mask
    if (
        signal.shape != shape
        or open_prices.shape != shape
        or close_prices.shape != shape
        or execution.shape != shape
        or shape != (len(symbols), len(dates))
    ):
        raise ValueError("Validation arrays have inconsistent shapes")
    if signal.dtype != np.float32:
        raise ValueError("Formal V3A validation signal must be float32")
    if not dates.is_monotonic_increasing or not dates.is_unique:
        raise ValueError("Validation dates must be strictly ordered")
    validation_indices = np.flatnonzero(
        (dates >= pd.Timestamp(config.validation_start))
        & (dates <= pd.Timestamp(config.validation_end))
    )
    if not len(validation_indices):
        raise ValueError("Validation date range is empty")
    first = int(validation_indices[0])
    prior_indices = np.flatnonzero(dates < pd.Timestamp(config.validation_start))
    if not len(prior_indices):
        raise ValueError("Validation requires a prior close decision date")
    prior = int(prior_indices[-1])
    if dates[int(validation_indices[-1])] > pd.Timestamp(config.validation_end):
        raise RuntimeError("Validation runner exposed dates after the frozen boundary")

    terminal_by_day: dict[int, list[tuple[int, float, bool, str]]] = {}
    if terminal_cashflows is not None and not terminal_cashflows.empty:
        required_terminal = {
            "fund_code",
            "payment_date",
            "cash_per_share",
            "is_final_payment",
            "evidence_type",
        }
        missing_terminal = required_terminal - set(terminal_cashflows.columns)
        if missing_terminal:
            raise ValueError(
                f"Terminal cashflows missing columns: {sorted(missing_terminal)}"
            )
        symbol_lookup = {str(symbol): index for index, symbol in enumerate(symbols)}
        for row in terminal_cashflows.itertuples(index=False):
            symbol = str(row.fund_code)
            if symbol not in symbol_lookup:
                raise ValueError(f"Terminal cashflow symbol is absent from panel: {symbol}")
            candidates = np.flatnonzero(dates >= pd.Timestamp(row.payment_date))
            if not len(candidates):
                continue
            day = int(candidates[0])
            final_value = row.is_final_payment
            is_final = (
                bool(final_value)
                if isinstance(final_value, (bool, np.bool_))
                else str(final_value).strip().lower() == "true"
            )
            terminal_by_day.setdefault(day, []).append(
                (
                    symbol_lookup[symbol],
                    float(row.cash_per_share),
                    is_final,
                    str(row.evidence_type),
                )
            )

    positions: list[Position | None] = [None] * config.slots
    blocked_symbols: set[int] = set()
    pending_exits: dict[int, ExitOrder] = {}
    initial_entries, _, _, _ = _entry_candidates(
        signal[:, prior],
        tradable_mask[:, prior],
        symbols,
        positions,
        blocked_symbols,
        config,
        decision_date=dates[prior].date().isoformat(),
    )
    pending_entries = initial_entries
    cash = float(config.initial_cash)
    last_closes = _last_finite_closes_before(close_prices, first)
    benchmark, buy_hold = _benchmark_path(
        open_prices, close_prices, tradable_mask, validation_indices
    )
    daily_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    peak = float(config.initial_cash)

    for output_index, t_value in enumerate(validation_indices):
        t = int(t_value)
        date = dates[t].date().isoformat()
        open_t = open_prices[:, t]
        for slot, order in list(pending_exits.items()):
            position = positions[slot]
            if position is None:
                pending_exits.pop(slot)
                continue
            price = open_t[position.symbol_idx]
            if not (execution[position.symbol_idx, t] and np.isfinite(price) and price > 0):
                continue
            proceeds = position.qty * float(price)
            cash += proceeds
            trade_rows.append(
                {
                    "formula_id": formula_id,
                    "date": date,
                    "action": "SELL",
                    "symbol": str(symbols[position.symbol_idx]),
                    "slot": slot,
                    "price": float(price),
                    "qty": position.qty,
                    "notional": proceeds,
                    "reason": order.reason,
                    "decision_date": order.decision_date,
                    "entry_date": position.entry_date,
                    "pnl_pct": float(price / position.entry_price - 1.0),
                }
            )
            if order.reason == "stop_loss":
                blocked_symbols.add(position.symbol_idx)
            positions[slot] = None
            pending_exits.pop(slot)

        empty_slots = [slot for slot, position in enumerate(positions) if position is None]
        cash_slice = cash / len(empty_slots) if empty_slots else 0.0
        for order in pending_entries:
            empty_slots = [slot for slot, position in enumerate(positions) if position is None]
            if not empty_slots:
                break
            if order.symbol_idx in blocked_symbols or any(
                position is not None and position.symbol_idx == order.symbol_idx
                for position in positions
            ):
                continue
            price = open_t[order.symbol_idx]
            if not (
                execution[order.symbol_idx, t]
                and np.isfinite(price)
                and price > 0
                and cash_slice > 0
            ):
                continue
            slot = empty_slots[0]
            value = min(cash_slice, cash)
            qty = value / float(price)
            cash -= value
            positions[slot] = Position(
                symbol_idx=order.symbol_idx,
                qty=float(qty),
                entry_price=float(price),
                entry_date=date,
                entry_decision_date=order.decision_date,
            )
            trade_rows.append(
                {
                    "formula_id": formula_id,
                    "date": date,
                    "action": "BUY",
                    "symbol": str(symbols[order.symbol_idx]),
                    "slot": slot,
                    "price": float(price),
                    "qty": float(qty),
                    "notional": float(value),
                    "reason": "entry",
                    "decision_date": order.decision_date,
                    "entry_date": "",
                    "pnl_pct": np.nan,
                }
            )
        pending_entries = []

        for symbol_idx, cash_per_share, is_final, evidence_type in terminal_by_day.get(
            t, []
        ):
            for slot, position in enumerate(positions):
                if position is None or position.symbol_idx != symbol_idx:
                    continue
                cash_amount = position.qty * cash_per_share
                cash += cash_amount
                position.terminal_cash_received_per_share += cash_per_share
                trade_rows.append(
                    {
                        "formula_id": formula_id,
                        "date": date,
                        "action": "TERMINAL_CASH" if is_final else "DISTRIBUTION",
                        "symbol": str(symbols[symbol_idx]),
                        "slot": slot,
                        "price": cash_per_share,
                        "qty": position.qty,
                        "notional": cash_amount,
                        "reason": "terminal_final_payment"
                        if is_final
                        else "terminal_partial_payment",
                        "decision_date": "",
                        "entry_date": position.entry_date,
                        "pnl_pct": (
                            position.terminal_cash_received_per_share
                            / position.entry_price
                            - 1.0
                            if is_final
                            else np.nan
                        ),
                        "evidence_type": evidence_type,
                    }
                )
                if is_final:
                    positions[slot] = None
                    pending_exits.pop(slot, None)

        close_t = close_prices[:, t]
        usable_close = np.isfinite(close_t) & (close_t > 0)
        last_closes[usable_close] = close_t[usable_close]
        equity = _portfolio_value(cash, positions, last_closes)
        peak = max(peak, equity)
        held_symbols = [
            str(symbols[position.symbol_idx])
            for position in positions
            if position is not None
        ]
        daily_rows.append(
            {
                "formula_id": formula_id,
                "date": date,
                "equity": equity,
                "cash": cash,
                "cash_weight": cash / equity if equity > 0 else np.nan,
                "position_count": len(held_symbols),
                "held_symbols": ";".join(held_symbols),
                "drawdown": 1.0 - equity / peak,
                "benchmark_equity": float(benchmark[output_index]),
                "buy_hold_equity": float(buy_hold[output_index]),
                "available_count": int(tradable_mask[:, t].sum()),
            }
        )

        if output_index == len(validation_indices) - 1:
            continue
        entries, ranks, robust_z, rankable = _entry_candidates(
            signal[:, t],
            tradable_mask[:, t],
            symbols,
            positions,
            blocked_symbols,
            config,
            decision_date=date,
        )
        if rankable:
            buy_eligible = (ranks <= config.buy_rank) & (
                robust_z >= config.robust_z_threshold
            )
            lost_eligibility = {
                symbol for symbol in blocked_symbols if not buy_eligible[symbol]
            }
            blocked_symbols.difference_update(lost_eligibility)
            for slot, position in enumerate(positions):
                if position is None or slot in pending_exits:
                    continue
                rank_exit = ranks[position.symbol_idx] > config.hold_rank
                current_close = close_t[position.symbol_idx]
                stop = (
                    np.isfinite(current_close)
                    and current_close > 0
                    and current_close / position.entry_price - 1.0 <= config.stop_loss
                )
                if stop or rank_exit:
                    pending_exits[slot] = ExitOrder(
                        slot=slot,
                        reason="stop_loss" if stop else "rank_exit",
                        decision_date=date,
                    )
        else:
            for slot, position in enumerate(positions):
                if position is None or slot in pending_exits:
                    continue
                current_close = close_t[position.symbol_idx]
                if (
                    np.isfinite(current_close)
                    and current_close > 0
                    and current_close / position.entry_price - 1.0 <= config.stop_loss
                ):
                    pending_exits[slot] = ExitOrder(slot, "stop_loss", date)
        pending_entries = entries

    daily = pd.DataFrame(daily_rows)
    trades = pd.DataFrame(trade_rows)
    summary = {
        "formula_id": formula_id,
        "validation_start": config.validation_start,
        "validation_end": config.validation_end,
        "validation_days": len(daily),
        "total_return": float(daily.iloc[-1]["equity"] / config.initial_cash - 1.0),
        "max_drawdown": _max_drawdown(daily["equity"], initial=config.initial_cash),
        "trade_count": len(trades),
        "sell_count": int((trades["action"] == "SELL").sum()) if not trades.empty else 0,
        "average_cash_weight": float(daily["cash_weight"].mean()),
        "benchmark_total_return": float(benchmark[-1] - 1.0),
        "benchmark_max_drawdown": _max_drawdown(benchmark),
        "buy_hold_total_return": float(buy_hold[-1] - 1.0),
        "metrics_scope": config.metrics_scope,
        "validation_or_final_metrics_read": True,
        "validation_metrics_read": config.metrics_scope
        in {"validation", "full_history_diagnostic"},
        "post2023_metrics_read": config.metrics_scope
        in {"post2023_exploratory", "full_history_diagnostic"},
        "final_metrics_read": config.metrics_scope
        in {"post2023_exploratory", "full_history_diagnostic"},
    }
    return daily, trades, summary


def summarize_group(rows: Iterable[dict[str, float]]) -> dict[str, float]:
    values = list(rows)
    if not values:
        raise ValueError("Validation group must not be empty")
    returns = np.asarray([row["total_return"] for row in values], dtype=np.float64)
    drawdowns = np.asarray([row["max_drawdown"] for row in values], dtype=np.float64)
    return {
        "count": int(len(values)),
        "return_median": float(np.quantile(returns, 0.5, method="linear")),
        "return_q75": float(np.quantile(returns, 0.75, method="linear")),
        "max_drawdown_median": float(np.quantile(drawdowns, 0.5, method="linear")),
    }


def _comparison(left: dict[str, dict[str, float]], right: dict[str, dict[str, float]]) -> dict[str, Any]:
    seeds = ("101", "102", "103")
    q75 = [left[seed]["return_q75"] - right[seed]["return_q75"] for seed in seeds]
    median = [
        left[seed]["return_median"] - right[seed]["return_median"] for seed in seeds
    ]
    drawdown = [
        left[seed]["max_drawdown_median"] - right[seed]["max_drawdown_median"]
        for seed in seeds
    ]
    return {
        "q75_differences": q75,
        "q75_mean_difference": float(np.mean(q75)),
        "q75_winning_seed_count": int(sum(value > 0 for value in q75)),
        "median_mean_difference": float(np.mean(median)),
        "drawdown_mean_difference": float(np.mean(drawdown)),
    }


def decide_validation(
    *,
    transformer: dict[str, dict[str, list[dict[str, float]]]],
    random: dict[str, dict[str, list[dict[str, float]]]],
    benchmark: dict[str, float],
    stable_pair_differences: dict[str, list[float]],
) -> dict[str, Any]:
    """Apply the frozen survival, complexity-upgrade, and Random denial gates."""

    order = ("paired_simple", "stable_complex", "original_top50")
    transformer_stats = {
        rule: {seed: summarize_group(rows) for seed, rows in seeds.items()}
        for rule, seeds in transformer.items()
    }
    random_stats = {
        rule: {seed: summarize_group(rows) for seed, rows in seeds.items()}
        for rule, seeds in random.items()
    }
    survival: dict[str, Any] = {}
    eligible: list[str] = []
    for rule in order:
        by_seed: dict[str, bool] = {}
        for seed in ("101", "102", "103"):
            stats = transformer_stats[rule][seed]
            by_seed[seed] = (
                stats["return_median"] > 0.0
                and stats["return_median"] > benchmark["total_return"]
                and stats["max_drawdown_median"] <= 0.20
                and stats["max_drawdown_median"] <= benchmark["max_drawdown"]
            )
        pair_seed_pass: dict[str, bool] = {}
        pair_pass = True
        if rule == "stable_complex":
            pair_seed_pass = {
                seed: bool(values) and float(np.median(values)) > 0.0
                for seed, values in stable_pair_differences.items()
            }
            pair_pass = sum(pair_seed_pass.values()) >= 2
        passed = sum(by_seed.values()) >= 2 and pair_pass
        survival[rule] = {
            "by_seed": by_seed,
            "surviving_seed_count": sum(by_seed.values()),
            "pair_by_seed": pair_seed_pass,
            "passed": passed,
        }
        if passed:
            eligible.append(rule)
    if not eligible:
        return {
            "outcome": "no_surviving_rule",
            "winner": None,
            "transformer_stats": transformer_stats,
            "random_stats": random_stats,
            "survival": survival,
        }

    provisional = eligible[0]
    upgrades: dict[str, Any] = {}
    for rule in eligible[1:]:
        comparisons = {
            simpler: _comparison(transformer_stats[rule], transformer_stats[simpler])
            for simpler in eligible
            if order.index(simpler) < order.index(rule)
        }
        passed = all(
            values["q75_mean_difference"] >= 0.03
            and values["q75_winning_seed_count"] >= 2
            and values["median_mean_difference"] >= 0.0
            and values["drawdown_mean_difference"] <= 0.0
            for values in comparisons.values()
        )
        upgrades[rule] = {"comparisons": comparisons, "passed": passed}
        if passed:
            provisional = rule

    random_comparison = _comparison(
        transformer_stats[provisional], random_stats[provisional]
    )
    random_passed = (
        random_comparison["q75_mean_difference"] >= 0.03
        and random_comparison["q75_winning_seed_count"] >= 2
        and random_comparison["median_mean_difference"] >= 0.0
        and random_comparison["drawdown_mean_difference"] <= 0.0
    )
    return {
        "outcome": "winner" if random_passed else "transformer_not_better_than_random",
        "winner": provisional if random_passed else None,
        "provisional_winner": provisional,
        "transformer_stats": transformer_stats,
        "random_stats": random_stats,
        "survival": survival,
        "upgrades": upgrades,
        "random_gate": {"comparison": random_comparison, "passed": random_passed},
    }