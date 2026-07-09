"""Simplified trading validator for Phase 2."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ValidatorConfig:
    horizon: int
    validator_variant: str = ""
    max_holding_days: int | None = None
    transaction_cost_bps: float = 0.0
    min_universe: int = 10
    slots: int = 3
    buy_rank: int = 3
    hold_rank: int = 5
    stop_loss: float = -0.05
    initial_cash: float = 1.0


@dataclass
class Position:
    symbol_idx: int
    qty: float
    entry_price: float
    entry_t: int
    entry_decision_t: int
    entry_notional: float
    entry_cost: float
    slot: int


@dataclass(frozen=True)
class ExitOrder:
    slot: int
    reason: str
    exit_flags: str
    decision_t: int


@dataclass(frozen=True)
class EntryOrder:
    slot: int
    symbol_idx: int
    decision_t: int
    reason: str = "entry"



def _rank_desc(signal_t: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    ranks = np.full(signal_t.shape, np.nan, dtype=float)
    eligible_idx = np.where(eligible)[0]
    if len(eligible_idx) == 0:
        return ranks
    ordered = eligible_idx[np.argsort(-signal_t[eligible_idx], kind="mergesort")]
    ranks[ordered] = np.arange(1, len(ordered) + 1, dtype=float)
    return ranks


def rank_only_signal(signal: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Convert raw generated-formula values into positive cross-sectional rank scores."""

    out = np.full(signal.shape, np.nan, dtype=float)
    for t in range(signal.shape[1]):
        eligible = mask[:, t] & np.isfinite(signal[:, t])
        ranks = _rank_desc(signal[:, t], eligible)
        eligible_count = int(np.isfinite(ranks).sum())
        if eligible_count == 0:
            continue
        out[np.isfinite(ranks), t] = eligible_count - ranks[np.isfinite(ranks)] + 1.0
    return out


def run_rank_only_validator(
    formula: str,
    signal: np.ndarray,
    open_prices: np.ndarray,
    close_prices: np.ndarray,
    mask: np.ndarray,
    dates: pd.DatetimeIndex,
    symbols: np.ndarray,
    config: ValidatorConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Run validator on rank-only positive scores for generated formulas.

    The existing validator intentionally keeps Phase2B's `signal > 0` entry
    rule. Generated formulas often only have ranking meaning, so this wrapper
    replaces raw values with positive cross-sectional ranks and leaves the rest
    of the trading audit unchanged.
    """

    rank_signal = rank_only_signal(signal, mask)
    variant = config.validator_variant or _variant(config)
    rank_config = ValidatorConfig(
        horizon=config.horizon,
        validator_variant=f"rank_only_{variant}",
        max_holding_days=config.max_holding_days,
        transaction_cost_bps=config.transaction_cost_bps,
        min_universe=config.min_universe,
        slots=config.slots,
        buy_rank=config.buy_rank,
        hold_rank=config.hold_rank,
        stop_loss=config.stop_loss,
        initial_cash=config.initial_cash,
    )
    return run_validator(formula, rank_signal, open_prices, close_prices, mask, dates, symbols, rank_config)



def _first_start(mask: np.ndarray, min_universe: int) -> int:
    counts = mask.sum(axis=0)
    idx = np.where(counts >= min_universe)[0]
    if len(idx) == 0:
        raise ValueError(f"No date has available_count >= {min_universe}")
    return int(idx[0])



def _portfolio_value(cash: float, positions: list[Position | None], prices_t: np.ndarray) -> float:
    value = cash
    for pos in positions:
        if pos is None:
            continue
        price = prices_t[pos.symbol_idx]
        if np.isfinite(price) and price > 0:
            value += pos.qty * price
        else:
            value += pos.qty * pos.entry_price
    return float(value)



def _date_or_blank(dates: pd.DatetimeIndex, t: int) -> str:
    if t < 0:
        return ""
    return dates[t].date().isoformat()



def _cost_rate(config: ValidatorConfig) -> float:
    return float(config.transaction_cost_bps) / 10000.0



def _variant(config: ValidatorConfig) -> str:
    if config.validator_variant:
        return config.validator_variant
    if config.max_holding_days is None:
        return "no_max_holding"
    return f"max_holding_{config.max_holding_days}"



def _benchmark_equity(close_prices: np.ndarray, mask: np.ndarray, start_t: int) -> np.ndarray:
    equity = np.full(mask.shape[1], np.nan, dtype=float)
    equity[start_t] = 1.0
    for t in range(start_t + 1, mask.shape[1]):
        eligible = (
            mask[:, t - 1]
            & mask[:, t]
            & np.isfinite(close_prices[:, t - 1])
            & np.isfinite(close_prices[:, t])
            & (close_prices[:, t - 1] > 0)
        )
        if eligible.any():
            daily_ret = np.nanmean(close_prices[eligible, t] / close_prices[eligible, t - 1] - 1.0)
        else:
            daily_ret = 0.0
        equity[t] = equity[t - 1] * (1.0 + daily_ret)
    return equity



def _make_entry_orders(
    signal_t: np.ndarray,
    ranks: np.ndarray,
    positions: list[Position | None],
    exiting_slots: set[int],
    config: ValidatorConfig,
) -> list[EntryOrder]:
    free_slots = [i for i, pos in enumerate(positions) if pos is None]
    free_slots.extend(sorted(exiting_slots))
    if not free_slots:
        return []

    held_symbols = {pos.symbol_idx for pos in positions if pos is not None}
    candidates = np.where(
        (ranks <= config.buy_rank)
        & (signal_t > 0)
    )[0]
    candidates = [int(idx) for idx in candidates if int(idx) not in held_symbols]
    candidates.sort(key=lambda idx: ranks[idx])

    orders = []
    for slot, symbol_idx in zip(free_slots, candidates, strict=False):
        orders.append(EntryOrder(slot=slot, symbol_idx=symbol_idx, decision_t=-1))
    return orders



def run_validator(
    formula: str,
    signal: np.ndarray,
    open_prices: np.ndarray,
    close_prices: np.ndarray,
    mask: np.ndarray,
    dates: pd.DatetimeIndex,
    symbols: np.ndarray,
    config: ValidatorConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    start_t = _first_start(mask, config.min_universe)
    benchmark = _benchmark_equity(close_prices, mask, start_t)
    cost_rate = _cost_rate(config)
    validator_variant = _variant(config)

    cash = float(config.initial_cash)
    positions: list[Position | None] = [None] * config.slots
    pending_exits: list[ExitOrder] = []
    pending_entries: list[EntryOrder] = []
    daily_rows: list[dict[str, object]] = []
    trade_rows: list[dict[str, object]] = []
    peak = config.initial_cash

    for t in range(start_t, signal.shape[1]):
        open_t = open_prices[:, t]

        for order in pending_exits:
            pos = positions[order.slot]
            if pos is None:
                continue
            price = open_t[pos.symbol_idx]
            if not (np.isfinite(price) and price > 0):
                continue
            proceeds = pos.qty * price
            transaction_cost = proceeds * cost_rate
            net_proceeds = proceeds - transaction_cost
            pnl_cash = net_proceeds - pos.entry_notional - pos.entry_cost
            pnl_pct = pnl_cash / (pos.entry_notional + pos.entry_cost)
            cash += net_proceeds
            trade_rows.append(
                {
                    "formula": formula,
                    "horizon": config.horizon,
                    "validator_variant": validator_variant,
                    "max_holding_days": config.max_holding_days,
                    "transaction_cost_bps": float(config.transaction_cost_bps),
                    "date": dates[t].date().isoformat(),
                    "action": "SELL",
                    "symbol": str(symbols[pos.symbol_idx]),
                    "slot": order.slot,
                    "price": float(price),
                    "qty": float(pos.qty),
                    "notional": float(proceeds),
                    "transaction_cost": float(transaction_cost),
                    "net_cash_flow": float(net_proceeds),
                    "cash_after": float(cash),
                    "reason": order.reason,
                    "exit_flags": order.exit_flags,
                    "entry_date": dates[pos.entry_t].date().isoformat(),
                    "entry_decision_date": _date_or_blank(dates, pos.entry_decision_t),
                    "exit_decision_date": _date_or_blank(dates, order.decision_t),
                    "holding_days": int(t - pos.entry_t),
                    "pnl_pct": float(pnl_pct),
                    "pnl_cash": float(pnl_cash),
                }
            )
            positions[order.slot] = None
        pending_exits = []

        for order in pending_entries:
            if positions[order.slot] is not None:
                continue
            price = open_t[order.symbol_idx]
            if not (np.isfinite(price) and price > 0):
                continue
            target_value = _portfolio_value(cash, positions, open_t) / config.slots
            value = min(target_value, cash / (1.0 + cost_rate))
            if value <= 0:
                continue
            qty = value / price
            transaction_cost = value * cost_rate
            cash -= value + transaction_cost
            positions[order.slot] = Position(
                symbol_idx=order.symbol_idx,
                qty=float(qty),
                entry_price=float(price),
                entry_t=t,
                entry_decision_t=order.decision_t,
                entry_notional=float(value),
                entry_cost=float(transaction_cost),
                slot=order.slot,
            )
            trade_rows.append(
                {
                    "formula": formula,
                    "horizon": config.horizon,
                    "validator_variant": validator_variant,
                    "max_holding_days": config.max_holding_days,
                    "transaction_cost_bps": float(config.transaction_cost_bps),
                    "date": dates[t].date().isoformat(),
                    "action": "BUY",
                    "symbol": str(symbols[order.symbol_idx]),
                    "slot": order.slot,
                    "price": float(price),
                    "qty": float(qty),
                    "notional": float(value),
                    "transaction_cost": float(transaction_cost),
                    "net_cash_flow": float(-(value + transaction_cost)),
                    "cash_after": float(cash),
                    "reason": order.reason,
                    "exit_flags": "",
                    "entry_date": "",
                    "entry_decision_date": _date_or_blank(dates, order.decision_t),
                    "exit_decision_date": "",
                    "holding_days": 0,
                    "pnl_pct": np.nan,
                    "pnl_cash": np.nan,
                }
            )
        pending_entries = []

        equity = _portfolio_value(cash, positions, close_prices[:, t])
        peak = max(peak, equity)
        drawdown = equity / peak - 1.0 if peak > 0 else np.nan
        held = [str(symbols[pos.symbol_idx]) for pos in positions if pos is not None]
        daily_rows.append(
            {
                "formula": formula,
                "horizon": config.horizon,
                "validator_variant": validator_variant,
                "max_holding_days": config.max_holding_days,
                "transaction_cost_bps": float(config.transaction_cost_bps),
                "date": dates[t].date().isoformat(),
                "equity": equity,
                "cash": cash,
                "cash_weight": cash / equity if equity > 0 else np.nan,
                "position_count": len(held),
                "held_symbols": ";".join(held),
                "drawdown": drawdown,
                "benchmark_equity": float(benchmark[t]),
                "available_count": int(mask[:, t].sum()),
            }
        )

        if t >= signal.shape[1] - 1:
            continue
        if int(mask[:, t].sum()) < config.min_universe:
            continue

        eligible = mask[:, t] & np.isfinite(signal[:, t])
        ranks = _rank_desc(signal[:, t], eligible)

        exit_orders: list[ExitOrder] = []
        exiting_slots: set[int] = set()
        for slot, pos in enumerate(positions):
            if pos is None:
                continue
            sig = signal[pos.symbol_idx, t]
            rank = ranks[pos.symbol_idx]
            flags: list[str] = []
            if not np.isfinite(sig) or sig <= 0:
                flags.append("signal_nonpositive")
            if not np.isfinite(rank) or rank > config.hold_rank:
                flags.append("rank_exit")
            close_t = close_prices[pos.symbol_idx, t]
            if np.isfinite(close_t) and close_t > 0 and close_t / pos.entry_price - 1.0 <= config.stop_loss:
                flags.append("stop_loss")
            if config.max_holding_days is not None and t - pos.entry_t + 1 >= config.max_holding_days:
                flags.append("max_holding")
            if flags:
                exit_orders.append(
                    ExitOrder(
                        slot=slot,
                        reason=flags[0],
                        exit_flags=";".join(flags),
                        decision_t=t,
                    )
                )
                exiting_slots.add(slot)

        entry_orders = _make_entry_orders(
            signal_t=signal[:, t],
            ranks=ranks,
            positions=positions,
            exiting_slots=exiting_slots,
            config=config,
        )
        entry_orders = [
            EntryOrder(slot=o.slot, symbol_idx=o.symbol_idx, decision_t=t, reason=o.reason)
            for o in entry_orders
        ]

        pending_exits = exit_orders
        pending_entries = entry_orders

    daily = pd.DataFrame(daily_rows)
    trades = pd.DataFrame(trade_rows)
    summary = summarize_validator_daily(daily, trades, formula, config)
    return daily, trades, summary



def summarize_validator_daily(
    daily: pd.DataFrame,
    trades: pd.DataFrame,
    formula: str,
    config: ValidatorConfig,
) -> dict[str, float]:
    validator_variant = _variant(config)
    if daily.empty:
        return {
            "formula": formula,
            "horizon": config.horizon,
            "validator_variant": validator_variant,
            "max_holding_days": config.max_holding_days,
            "transaction_cost_bps": float(config.transaction_cost_bps),
            "validator_days": 0,
            "validator_total_return": np.nan,
            "validator_ann_return": np.nan,
            "validator_sharpe": np.nan,
            "validator_max_drawdown": np.nan,
            "validator_trade_count": 0,
            "validator_sell_count": 0,
            "validator_avg_cash_weight": np.nan,
            "benchmark_total_return": np.nan,
            "benchmark_ann_return": np.nan,
        }

    equity = daily["equity"].astype(float)
    returns = equity.pct_change().fillna(0.0)
    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0
    years = max(len(daily) / 252.0, 1e-9)
    ann_return = (equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0
    ret_std = returns.std(ddof=0)
    sharpe = np.nan if ret_std == 0 else returns.mean() / ret_std * np.sqrt(252.0)

    benchmark = daily["benchmark_equity"].astype(float)
    benchmark_total = benchmark.iloc[-1] / benchmark.iloc[0] - 1.0
    benchmark_ann = (benchmark.iloc[-1] / benchmark.iloc[0]) ** (1.0 / years) - 1.0

    return {
        "formula": formula,
        "horizon": config.horizon,
        "validator_variant": validator_variant,
        "max_holding_days": config.max_holding_days,
        "transaction_cost_bps": float(config.transaction_cost_bps),
        "validator_days": int(len(daily)),
        "validator_total_return": float(total_return),
        "validator_ann_return": float(ann_return),
        "validator_sharpe": float(sharpe) if np.isfinite(sharpe) else np.nan,
        "validator_max_drawdown": float(daily["drawdown"].min()),
        "validator_trade_count": int(len(trades)),
        "validator_sell_count": int((trades["action"] == "SELL").sum()) if not trades.empty else 0,
        "validator_avg_cash_weight": float(daily["cash_weight"].astype(float).mean()),
        "benchmark_total_return": float(benchmark_total),
        "benchmark_ann_return": float(benchmark_ann),
    }
