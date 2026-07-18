#!/usr/bin/env python3
"""Generate a self-contained interactive HTML report for the ABS(ROC(40)) backtest."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_PATH = ROOT / "scripts/v3a/templates/absroc40_report.html"
sys.path.insert(0, str(ROOT / "src"))

from alpha_etf.research_v3a.factors import (  # noqa: E402
    FACTOR_NAMES,
    build_factor_values_numpy,
)
from alpha_etf.research_v3a.spec import load_panel  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--daily", type=Path, required=True)
    parser.add_argument("--trades", type=Path, required=True)
    parser.add_argument("--universe", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--echarts-js", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def _drawdown(values: np.ndarray) -> np.ndarray:
    peaks = np.maximum.accumulate(np.r_[1.0, values])[1:]
    return values / peaks - 1.0


def _drawdown_episodes(
    dates: list[str], values: np.ndarray, *, initial_date: str
) -> list[dict[str, Any]]:
    all_dates = [initial_date, *dates]
    all_values = np.r_[1.0, values]
    peak_index = 0
    active: dict[str, int] | None = None
    episodes: list[dict[str, Any]] = []

    def finish(recovery_index: int | None) -> None:
        assert active is not None
        end_index = recovery_index if recovery_index is not None else len(all_values) - 1
        peak_date = pd.Timestamp(all_dates[active["peak"]])
        end_date = pd.Timestamp(all_dates[end_index])
        episodes.append(
            {
                "peak_date": all_dates[active["peak"]],
                "trough_date": all_dates[active["trough"]],
                "recovery_date": all_dates[recovery_index] if recovery_index is not None else None,
                "drawdown": float(
                    all_values[active["trough"]] / all_values[active["peak"]] - 1.0
                ),
                "duration_days": int((end_date - peak_date).days),
                "status": "recovered" if recovery_index is not None else "ongoing",
            }
        )

    for index in range(1, len(all_values)):
        if all_values[index] >= all_values[peak_index] - 1e-12:
            if active is not None:
                finish(index)
                active = None
            peak_index = index
            continue
        if active is None:
            active = {"peak": peak_index, "trough": index}
        elif all_values[index] < all_values[active["trough"]]:
            active["trough"] = index
    if active is not None:
        finish(None)
    return sorted(episodes, key=lambda row: (row["drawdown"], row["peak_date"]))[:5]


def _monthly_returns(daily: pd.DataFrame) -> tuple[list[str], list[list[Any]]]:
    month_end = daily.set_index("date")["equity"].resample("M").last()
    returns = month_end.pct_change()
    if not returns.empty:
        returns.iloc[0] = month_end.iloc[0] - 1.0
    years = sorted({int(index.year) for index in returns.index})
    year_index = {year: index for index, year in enumerate(years)}
    cells = [
        [int(index.month - 1), year_index[int(index.year)], _finite(value)]
        for index, value in returns.items()
    ]
    return [str(year) for year in years], cells


def _histogram(values: np.ndarray) -> list[dict[str, Any]]:
    edges = np.arange(-0.15, 1.0001, 0.05)
    counts, edges = np.histogram(values, bins=edges)
    return [
        {
            "label": f"{edges[index] * 100:.0f}%~{edges[index + 1] * 100:.0f}%",
            "count": int(count),
            "positive": bool(edges[index] >= 0),
        }
        for index, count in enumerate(counts)
    ]


def _annotate_entry_roc40(
    trades: pd.DataFrame, dataset_dir: Path, *, expected_end: str
) -> tuple[pd.DataFrame, dict[str, float]]:
    panel = load_panel(dataset_dir)
    if panel.dates[-1].date().isoformat() != expected_end:
        raise RuntimeError("Signed ROC attribution dataset end differs from report result")
    factor_values = build_factor_values_numpy(panel.absolute_ohlc, panel.tradable_mask)
    roc40 = factor_values[FACTOR_NAMES.index("ROC_40")].astype(np.float32)
    date_index = {date.date().isoformat(): index for index, date in enumerate(panel.dates)}
    symbol_index = {str(symbol): index for index, symbol in enumerate(panel.symbols)}
    positions: dict[int, tuple[float, str]] = {}
    entry_values: list[float] = []
    directions: list[str] = []
    for row in trades.itertuples():
        slot = int(row.slot)
        if row.action == "BUY":
            try:
                value = float(roc40[symbol_index[str(row.symbol)], date_index[str(row.decision_date)]])
            except KeyError as exc:
                raise RuntimeError("Trade decision date or symbol is absent from ROC panel") from exc
            if not np.isfinite(value) or value == 0.0:
                raise RuntimeError("ABS(ROC(40)) buy must have a finite nonzero signed ROC")
            direction = "positive" if value > 0.0 else "negative"
            positions[slot] = (value, direction)
        else:
            if slot not in positions:
                raise RuntimeError("Sell has no open position during signed ROC attribution")
            value, direction = positions.pop(slot)
        entry_values.append(value)
        directions.append(direction)
    output = trades.copy()
    output["entry_roc40"] = entry_values
    output["entry_direction"] = directions
    end_index = date_index[expected_end]
    last_closes: dict[str, float] = {}
    close = panel.absolute("close")
    for symbol, index in symbol_index.items():
        values = close[index, : end_index + 1]
        finite = values[np.isfinite(values) & (values > 0)]
        if len(finite):
            last_closes[symbol] = float(finite[-1])
    return output, last_closes


def _attribution_rows(
    sells: pd.DataFrame,
    column: str,
    *,
    order: list[str] | None = None,
    display: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, group in sells.groupby(column, sort=False):
        label = str(label)
        pnl = group["realized_pnl"]
        rows.append(
            {
                "key": label,
                "label": (display or {}).get(label, label),
                "net_pnl": float(pnl.sum()),
                "gross_profit": float(pnl[pnl > 0].sum()),
                "gross_loss": float(pnl[pnl < 0].sum()),
                "trade_count": int(len(group)),
                "win_rate": float((pnl > 0).mean()),
                "average_return": float(group["pnl_pct"].mean()),
            }
        )
    if order is not None:
        positions = {key: index for index, key in enumerate(order)}
        return sorted(rows, key=lambda row: (positions.get(row["key"], len(order)), row["key"]))
    return sorted(rows, key=lambda row: (-row["net_pnl"], row["key"]))


def _trade_contribution(
    daily: pd.DataFrame,
    trades: pd.DataFrame,
    names: dict[str, str],
    last_closes: dict[str, float],
) -> dict[str, Any]:
    sells = trades[trades["action"] == "SELL"].copy()
    sells["holding_days"] = (
        pd.to_datetime(sells["date"]) - pd.to_datetime(sells["entry_date"])
    ).dt.days
    sells["realized_pnl"] = (
        sells["notional"] * sells["pnl_pct"] / (1.0 + sells["pnl_pct"])
    )
    final = daily.iloc[-1]
    positions: dict[int, Any] = {}
    for row in trades.itertuples():
        if row.action == "BUY":
            positions[int(row.slot)] = row
        else:
            positions.pop(int(row.slot), None)
    open_positions = [
        {
            "symbol": str(row.symbol),
            "name": names.get(str(row.symbol), str(row.symbol)),
            "entry_date": str(row.date),
            "cost": float(row.notional),
            "entry_roc40": float(row.entry_roc40),
            "entry_direction": str(row.entry_direction),
            "unrealized_pnl": float(
                float(row.qty) * last_closes[str(row.symbol)] - float(row.notional)
            ),
        }
        for row in positions.values()
    ]
    unrealized_pnl = float(sum(row["unrealized_pnl"] for row in open_positions))
    realized_net = float(sells["realized_pnl"].sum())
    gross_profit = float(sells.loc[sells["realized_pnl"] > 0, "realized_pnl"].sum())
    gross_loss = float(sells.loc[sells["realized_pnl"] < 0, "realized_pnl"].sum())
    total_profit = float(final["equity"] - 1.0)
    reconciliation_error = realized_net + unrealized_pnl - total_profit
    if abs(reconciliation_error) > 1e-10:
        raise RuntimeError(f"Trade contribution does not reconcile: {reconciliation_error}")

    ranked = sells.sort_values("realized_pnl", ascending=False, kind="stable")
    cumulative = ranked["realized_pnl"].cumsum()
    concentration: list[dict[str, Any]] = []
    for fraction in (0.01, 0.05, 0.10, 0.20):
        count = max(1, int(np.ceil(len(ranked) * fraction)))
        contribution = float(ranked.iloc[:count]["realized_pnl"].sum())
        concentration.append(
            {
                "fraction": fraction,
                "count": count,
                "pnl": contribution,
                "gross_profit_share": contribution / gross_profit,
                "total_net_profit_share": contribution / total_profit,
            }
        )

    def return_bucket(value: float) -> str:
        if value <= -0.10:
            return "<=-10%"
        if value < -0.05:
            return "-10%~-5%"
        if value < 0.0:
            return "-5%~0%"
        if value == 0.0:
            return "0%"
        if value < 0.05:
            return "0%~5%"
        if value < 0.10:
            return "5%~10%"
        if value < 0.20:
            return "10%~20%"
        return ">=20%"

    return_order = [
        "<=-10%",
        "-10%~-5%",
        "-5%~0%",
        "0%",
        "0%~5%",
        "5%~10%",
        "10%~20%",
        ">=20%",
    ]
    sells["return_bucket"] = sells["pnl_pct"].map(return_bucket)
    return_bins = _attribution_rows(sells, "return_bucket", order=return_order)

    sells["exit_year"] = sells["date"].astype(str).str[:4]
    holding_order = ["1-5天", "6-10天", "11-20天", "21-60天", "60天以上"]
    sells["holding_bucket"] = pd.cut(
        sells["holding_days"],
        bins=[0, 5, 10, 20, 60, np.inf],
        labels=holding_order,
        include_lowest=True,
    ).astype(str)
    sells["symbol_label"] = sells["symbol"].astype(str).map(
        lambda symbol: f"{symbol} {names.get(symbol, symbol)}"
    )
    attribution = {
        "etf": _attribution_rows(sells, "symbol_label"),
        "year": _attribution_rows(
            sells,
            "exit_year",
            order=sorted(sells["exit_year"].unique().tolist()),
        ),
        "holding": _attribution_rows(sells, "holding_bucket", order=holding_order),
        "exit_reason": _attribution_rows(
            sells,
            "reason",
            order=["rank_exit", "stop_loss"],
            display={"rank_exit": "排名退出", "stop_loss": "止损"},
        ),
    }
    direction_order = ["positive", "negative"]
    direction_display = {"positive": "正ROC", "negative": "负ROC"}
    buys = trades[trades["action"] == "BUY"]
    direction_rows: list[dict[str, Any]] = []
    exit_rows: list[dict[str, Any]] = []
    open_frame = pd.DataFrame(open_positions)
    for direction in direction_order:
        direction_buys = buys[buys["entry_direction"] == direction]
        direction_sells = sells[sells["entry_direction"] == direction]
        direction_open = (
            open_frame[open_frame["entry_direction"] == direction]
            if not open_frame.empty
            else open_frame
        )
        direction_realized = float(direction_sells["realized_pnl"].sum())
        direction_unrealized = (
            float(direction_open["unrealized_pnl"].sum()) if not direction_open.empty else 0.0
        )
        rank_exits = direction_sells[direction_sells["reason"] == "rank_exit"]
        stop_losses = direction_sells[direction_sells["reason"] == "stop_loss"]
        direction_rows.append(
            {
                "key": direction,
                "label": direction_display[direction],
                "buy_count": int(len(direction_buys)),
                "buy_share": float(len(direction_buys) / len(buys)),
                "mean_entry_roc40": float(direction_buys["entry_roc40"].mean()),
                "median_entry_roc40": float(direction_buys["entry_roc40"].median()),
                "closed_count": int(len(direction_sells)),
                "win_rate": float((direction_sells["pnl_pct"] > 0).mean()),
                "average_return": float(direction_sells["pnl_pct"].mean()),
                "median_return": float(direction_sells["pnl_pct"].median()),
                "realized_pnl": direction_realized,
                "unrealized_pnl": direction_unrealized,
                "total_pnl": direction_realized + direction_unrealized,
                "total_profit_share": (direction_realized + direction_unrealized) / total_profit,
                "rank_exit_count": int(len(rank_exits)),
                "rank_exit_rate": float(len(rank_exits) / len(direction_sells)),
                "rank_exit_win_rate": float((rank_exits["pnl_pct"] > 0).mean()),
                "rank_exit_average_return": float(rank_exits["pnl_pct"].mean()),
                "rank_exit_median_return": float(rank_exits["pnl_pct"].median()),
                "rank_exit_pnl": float(rank_exits["realized_pnl"].sum()),
                "rank_exit_average_holding": float(rank_exits["holding_days"].mean()),
                "rank_exit_median_holding": float(rank_exits["holding_days"].median()),
                "rank_exit_p90": float(rank_exits["pnl_pct"].quantile(0.90)),
                "rank_exit_p95": float(rank_exits["pnl_pct"].quantile(0.95)),
                "rank_exit_p99": float(rank_exits["pnl_pct"].quantile(0.99)),
                "rank_exit_max": float(rank_exits["pnl_pct"].max()),
                "stop_loss_count": int(len(stop_losses)),
                "stop_loss_rate": float(len(stop_losses) / len(direction_sells)),
                "stop_loss_average_return": float(stop_losses["pnl_pct"].mean()),
                "stop_loss_median_return": float(stop_losses["pnl_pct"].median()),
                "stop_loss_pnl": float(stop_losses["realized_pnl"].sum()),
                "stop_loss_average_holding": float(stop_losses["holding_days"].mean()),
                "stop_loss_median_holding": float(stop_losses["holding_days"].median()),
            }
        )
        for reason in ("rank_exit", "stop_loss"):
            group = direction_sells[direction_sells["reason"] == reason]
            exit_rows.append(
                {
                    "direction": direction,
                    "direction_label": direction_display[direction],
                    "reason": reason,
                    "reason_label": "排名退出" if reason == "rank_exit" else "止损",
                    "count": int(len(group)),
                    "share_within_direction": float(len(group) / len(direction_sells)),
                    "win_rate": float((group["pnl_pct"] > 0).mean()),
                    "average_return": float(group["pnl_pct"].mean()),
                    "median_return": float(group["pnl_pct"].median()),
                    "realized_pnl": float(group["realized_pnl"].sum()),
                    "gross_profit": float(group.loc[group["realized_pnl"] > 0, "realized_pnl"].sum()),
                    "gross_loss": float(group.loc[group["realized_pnl"] < 0, "realized_pnl"].sum()),
                    "average_holding": float(group["holding_days"].mean()),
                    "median_holding": float(group["holding_days"].median()),
                }
            )
    direction_holding: list[dict[str, Any]] = []
    for bucket in holding_order:
        row: dict[str, Any] = {"bucket": bucket}
        for direction in direction_order:
            group = sells[
                (sells["holding_bucket"] == bucket)
                & (sells["entry_direction"] == direction)
            ]
            row[f"{direction}_count"] = int(len(group))
            row[f"{direction}_pnl"] = float(group["realized_pnl"].sum())
        direction_holding.append(row)
    top_tail: list[dict[str, Any]] = []
    for count in (6, 10, 18, 29, 58):
        top = ranked.iloc[:count]
        top_tail.append(
            {
                "count": count,
                "positive_count": int((top["entry_direction"] == "positive").sum()),
                "negative_count": int((top["entry_direction"] == "negative").sum()),
                "positive_pnl": float(
                    top.loc[top["entry_direction"] == "positive", "realized_pnl"].sum()
                ),
                "negative_pnl": float(
                    top.loc[top["entry_direction"] == "negative", "realized_pnl"].sum()
                ),
            }
        )
    return {
        "summary": {
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "realized_net": realized_net,
            "unrealized_pnl": unrealized_pnl,
            "total_profit": total_profit,
            "profit_factor": gross_profit / abs(gross_loss),
            "positive_count": int((sells["realized_pnl"] > 0).sum()),
            "negative_count": int((sells["realized_pnl"] < 0).sum()),
            "breakeven_count": int((sells["realized_pnl"] == 0).sum()),
            "reconciliation_error": reconciliation_error,
        },
        "open_positions": open_positions,
        "concentration": concentration,
        "curve": {
            "trade_fraction": [float((index + 1) / len(ranked)) for index in range(len(ranked))],
            "cumulative_pnl": [float(value) for value in cumulative],
        },
        "return_bins": return_bins,
        "attribution": attribution,
        "signed_roc": {
            "directions": direction_rows,
            "exit_rows": exit_rows,
            "holding": direction_holding,
            "top_tail": top_tail,
        },
    }


def _etf_stats(
    daily: pd.DataFrame, trades: pd.DataFrame, names: dict[str, str]
) -> list[dict[str, Any]]:
    held_days: Counter[str] = Counter()
    for value in daily["held_symbols"].fillna("").astype(str):
        held_days.update(symbol for symbol in value.split(";") if symbol)
    buys = Counter(trades.loc[trades["action"] == "BUY", "symbol"].astype(str))
    sells = trades[trades["action"] == "SELL"].copy()
    output: list[dict[str, Any]] = []
    for symbol in sorted(set(held_days) | set(buys)):
        symbol_sells = sells[sells["symbol"].astype(str) == symbol]
        output.append(
            {
                "symbol": symbol,
                "name": names.get(symbol, symbol),
                "held_days": int(held_days[symbol]),
                "buy_count": int(buys[symbol]),
                "closed_count": int(len(symbol_sells)),
                "average_closed_return": (
                    _finite(symbol_sells["pnl_pct"].mean()) if not symbol_sells.empty else None
                ),
            }
        )
    return sorted(output, key=lambda row: (-row["held_days"], row["symbol"]))


def _trade_records(trades: pd.DataFrame, names: dict[str, str]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in trades.sort_values(["date", "slot"], ascending=[False, True]).itertuples():
        symbol = str(row.symbol)
        output.append(
            {
                "date": str(row.date),
                "action": str(row.action),
                "symbol": symbol,
                "name": names.get(symbol, symbol),
                "reason": str(row.reason),
                "decision_date": str(row.decision_date),
                "entry_date": None if pd.isna(row.entry_date) else str(row.entry_date),
                "entry_roc40": _finite(row.entry_roc40),
                "entry_direction": str(row.entry_direction),
                "price": _finite(row.price),
                "notional": _finite(row.notional),
                "pnl_pct": _finite(row.pnl_pct),
                "realized_pnl": (
                    _finite(row.notional * row.pnl_pct / (1.0 + row.pnl_pct))
                    if row.action == "SELL" and pd.notna(row.pnl_pct)
                    else None
                ),
                "slot": int(row.slot),
            }
        )
    return output


def _build_payload(
    result: dict[str, Any],
    daily: pd.DataFrame,
    trades: pd.DataFrame,
    names: dict[str, str],
    source_hashes: dict[str, str],
    last_closes: dict[str, float],
) -> dict[str, Any]:
    dates = daily["date"].dt.strftime("%Y-%m-%d").tolist()
    strategy = daily["equity"].to_numpy(dtype=np.float64)
    benchmark = daily["benchmark_equity"].to_numpy(dtype=np.float64)
    buy_hold = daily["buy_hold_equity"].to_numpy(dtype=np.float64)
    strategy_daily = pd.Series(strategy).pct_change()
    benchmark_daily = pd.Series(benchmark).pct_change()
    rolling_strategy = (1.0 + strategy_daily).rolling(252).apply(np.prod, raw=True) - 1.0
    rolling_benchmark = (1.0 + benchmark_daily).rolling(252).apply(np.prod, raw=True) - 1.0
    month_years, monthly = _monthly_returns(daily)
    strategy_years = result["strategy"]["calendar_year_returns"]
    benchmark_years = result["benchmark"]["calendar_year_returns"]
    years = list(strategy_years)

    sells = trades[trades["action"] == "SELL"].copy()
    sells["holding_days"] = (
        pd.to_datetime(sells["date"]) - pd.to_datetime(sells["entry_date"])
    ).dt.days
    scatter = [
        {
            "holding_days": int(row.holding_days),
            "pnl_pct": _finite(row.pnl_pct),
            "symbol": str(row.symbol),
            "name": names.get(str(row.symbol), str(row.symbol)),
            "date": str(row.date),
            "reason": str(row.reason),
        }
        for row in sells.itertuples()
    ]

    return {
        "identity": {
            "formula": result["formula"]["text"],
            "date_start": result["date_start"],
            "date_end": result["date_end"],
            "prior_signal_date": result["prior_signal_date"],
            "protocol_id": result["protocol_id"],
            "dataset_id": result["dataset_id"],
            "source_hashes": source_hashes,
        },
        "metrics": {
            "strategy": result["strategy"],
            "benchmark": result["benchmark"],
        },
        "path": {
            "dates": dates,
            "strategy": [_finite(value) for value in strategy],
            "benchmark": [_finite(value) for value in benchmark],
            "buy_hold": [_finite(value) for value in buy_hold],
            "strategy_drawdown": [_finite(value) for value in _drawdown(strategy)],
            "benchmark_drawdown": [_finite(value) for value in _drawdown(benchmark)],
            "rolling_strategy": [_finite(value) for value in rolling_strategy],
            "rolling_benchmark": [_finite(value) for value in rolling_benchmark],
            "cash_weight": [_finite(value) for value in daily["cash_weight"]],
            "position_count": [int(value) for value in daily["position_count"]],
            "available_count": [int(value) for value in daily["available_count"]],
        },
        "annual": {
            "years": years,
            "strategy": [_finite(strategy_years[year]) for year in years],
            "benchmark": [_finite(benchmark_years[year]) for year in years],
        },
        "monthly": {"years": month_years, "cells": monthly},
        "drawdowns": _drawdown_episodes(
            dates, strategy, initial_date=result["prior_signal_date"]
        ),
        "exit_reasons": {
            "rank_exit": int((sells["reason"] == "rank_exit").sum()),
            "stop_loss": int((sells["reason"] == "stop_loss").sum()),
        },
        "trade_histogram": _histogram(sells["pnl_pct"].to_numpy(dtype=np.float64)),
        "trade_scatter": scatter,
        "trade_contribution": _trade_contribution(daily, trades, names, last_closes),
        "etf_stats": _etf_stats(daily, trades, names),
        "trades": _trade_records(trades, names),
    }


def main() -> None:
    args = _parse_args()
    paths = {
        "result": args.result.resolve(),
        "daily": args.daily.resolve(),
        "trades": args.trades.resolve(),
    }
    result = json.loads(paths["result"].read_text(encoding="utf-8"))
    if (
        result.get("formula", {}).get("text") != "ABS(ROC(40))"
        or result.get("interpretation")
        != "exploratory_long_history_diagnostic_not_oos_proof"
    ):
        raise RuntimeError("Report source is not the frozen ABS(ROC(40)) diagnostic")
    daily = pd.read_csv(paths["daily"])
    daily["date"] = pd.to_datetime(daily["date"])
    trades = pd.read_csv(paths["trades"], dtype={"symbol": str})
    if (
        daily.iloc[0]["date"].date().isoformat() != result["date_start"]
        or daily.iloc[-1]["date"].date().isoformat() != result["date_end"]
        or not (daily["formula_id"] == "abs_roc_40").all()
        or not (trades["formula_id"] == "abs_roc_40").all()
    ):
        raise RuntimeError("Report source boundaries or formula identity mismatch")
    universe = json.loads(args.universe.resolve().read_text(encoding="utf-8"))
    names = {str(row["symbol"]): str(row["name"]) for row in universe}
    trades, last_closes = _annotate_entry_roc40(
        trades,
        args.dataset_dir.resolve(),
        expected_end=result["date_end"],
    )
    source_hashes = {name: _sha256(path) for name, path in paths.items()}
    payload = _build_payload(result, daily, trades, names, source_hashes, last_closes)

    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    echarts_js = args.echarts_js.resolve().read_text(encoding="utf-8")
    if "Apache Software Foundation" not in echarts_js[:2_000] or len(echarts_js) < 500_000:
        raise RuntimeError("ECharts bundle is missing or invalid")
    html = template.replace("/*__ECHARTS_BUNDLE__*/", echarts_js).replace(
        "/*__REPORT_DATA__*/", json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )
    if "/*__ECHARTS_BUNDLE__*/" in html or "/*__REPORT_DATA__*/" in html:
        raise RuntimeError("HTML template placeholders were not replaced")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "bytes": output.stat().st_size,
                "sha256": _sha256(output),
                "source_hashes": source_hashes,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()