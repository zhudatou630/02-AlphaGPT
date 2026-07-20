#!/usr/bin/env python3
"""Run the frozen expanded-universe ABS(ROC(40)) phase-two backtests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from alpha_etf.research_v3a.spec import canonical_sha256  # noqa: E402
from alpha_etf.research_v3a.validation import (  # noqa: E402
    ValidationConfig,
    run_formula_validation,
)
from scripts.v3a.runtime import code_fingerprint, git_commit  # noqa: E402


SCHEMA_VERSION = "expanded-absroc40-phase2-result-v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    base = ROOT / "data" / "processed" / "expanded_etf_audit"
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs" / "v3a_expanded_absroc40_phase2.json",
    )
    parser.add_argument("--dataset-dir", type=Path, default=base / "phase2_dataset")
    parser.add_argument("--output-dir", type=Path, default=base / "phase2_results")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    payload = dict(protocol)
    expected = payload.pop("protocol_id")
    if canonical_sha256(payload) != expected:
        raise RuntimeError("expanded phase-two protocol ID mismatch")
    return protocol


def _terminal_schedule(
    terminal: pd.DataFrame, dates: pd.DatetimeIndex, symbols: np.ndarray
) -> dict[int, list[tuple[int, float, bool]]]:
    symbol_index = {str(symbol): index for index, symbol in enumerate(symbols)}
    schedule: dict[int, list[tuple[int, float, bool]]] = {}
    for row in terminal.itertuples(index=False):
        candidates = np.flatnonzero(dates >= pd.Timestamp(row.payment_date))
        if not len(candidates):
            continue
        value = row.is_final_payment
        is_final = (
            bool(value)
            if isinstance(value, (bool, np.bool_))
            else str(value).strip().lower() == "true"
        )
        schedule.setdefault(int(candidates[0]), []).append(
            (symbol_index[str(row.fund_code)], float(row.cash_per_share), is_final)
        )
    return schedule


def _monthly_equal_weight(
    *,
    open_prices: np.ndarray,
    close_prices: np.ndarray,
    price_mask: np.ndarray,
    eligibility_mask: np.ndarray,
    dates: pd.DatetimeIndex,
    start: str,
    end: str,
    min_universe: int,
    symbols: np.ndarray,
    terminal: pd.DataFrame,
) -> pd.DataFrame:
    indices = np.flatnonzero(
        (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    )
    terminal_by_day = _terminal_schedule(terminal, dates, symbols)
    quantities = np.zeros(len(symbols), dtype=np.float64)
    cash_received = np.zeros(len(symbols), dtype=np.float64)
    last_closes = np.full(len(symbols), np.nan, dtype=np.float64)
    for t in np.flatnonzero(dates < pd.Timestamp(start)):
        valid = price_mask[:, t] & np.isfinite(close_prices[:, t])
        last_closes[valid] = close_prices[valid, t]
    cash = 1.0
    initialized = False
    last_rebalance_month: tuple[int, int] | None = None
    rows: list[dict[str, object]] = []
    for t_value in indices:
        t = int(t_value)
        previous = t - 1
        eligible = (
            eligibility_mask[:, previous]
            & price_mask[:, t]
            & np.isfinite(open_prices[:, t])
            & (open_prices[:, t] > 0)
        )
        month = (dates[t].year, dates[t].month)
        initialize = not initialized and int(eligible.sum()) >= min_universe
        monthly_rebalance = initialized and month != last_rebalance_month
        if initialize or monthly_rebalance:
            tradable_holdings = (
                (quantities > 0)
                & price_mask[:, t]
                & np.isfinite(open_prices[:, t])
                & (open_prices[:, t] > 0)
            )
            cash += float(np.sum(quantities[tradable_holdings] * open_prices[tradable_holdings, t]))
            quantities[tradable_holdings] = 0.0
            cash_received[tradable_holdings] = 0.0
            targets = np.flatnonzero(eligible & (quantities == 0))
            if len(targets):
                allocation = cash / len(targets)
                quantities[targets] = allocation / open_prices[targets, t]
                cash = 0.0
                initialized = True
                last_rebalance_month = month

        for asset, cash_per_share, is_final in terminal_by_day.get(t, []):
            if quantities[asset] <= 0:
                continue
            cash += quantities[asset] * cash_per_share
            cash_received[asset] += cash_per_share
            if is_final:
                quantities[asset] = 0.0
                cash_received[asset] = 0.0

        valid_close = price_mask[:, t] & np.isfinite(close_prices[:, t])
        last_closes[valid_close] = close_prices[valid_close, t]
        residual_prices = np.maximum(last_closes - cash_received, 0.0)
        equity = cash + float(
            np.nansum(np.where(quantities > 0, quantities * residual_prices, 0.0))
        )
        rows.append(
            {
                "date": dates[t].date().isoformat(),
                "benchmark_equity": equity,
                "benchmark_cash": cash,
                "benchmark_position_count": int((quantities > 0).sum()),
                "benchmark_eligible_count": int(eligible.sum()),
                "benchmark_rebalanced": bool(initialize or monthly_rebalance),
            }
        )
    return pd.DataFrame(rows)


def _path_metrics(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    values = np.r_[1.0, frame[column].to_numpy(dtype=np.float64)]
    peaks = np.maximum.accumulate(values)
    drawdowns = 1.0 - values / peaks
    trough = int(np.argmax(drawdowns))
    peak = int(np.argmax(values[: trough + 1]))
    dates = ["initial", *frame["date"].astype(str).tolist()]
    total = float(values[-1] - 1.0)
    return {
        "total_return": total,
        "annualized_return": float((1.0 + total) ** (252.0 / (len(values) - 1)) - 1.0),
        "max_drawdown": float(drawdowns[trough]),
        "max_drawdown_peak_date": dates[peak],
        "max_drawdown_trough_date": dates[trough],
    }


def _year_returns(frame: pd.DataFrame, column: str) -> dict[str, float]:
    values = frame.set_index(pd.to_datetime(frame["date"]))[column].astype(float)
    previous = 1.0
    output: dict[str, float] = {}
    for year, part in values.groupby(values.index.year):
        last = float(part.iloc[-1])
        output[str(year)] = last / previous - 1.0
        previous = last
    return output


def _trade_metrics(trades: pd.DataFrame) -> dict[str, Any]:
    exits = trades[trades["action"].isin(["SELL", "TERMINAL_CASH"])]
    return {
        "action_count": len(trades),
        "buy_count": int(trades["action"].eq("BUY").sum()),
        "market_sell_count": int(trades["action"].eq("SELL").sum()),
        "terminal_close_count": int(trades["action"].eq("TERMINAL_CASH").sum()),
        "partial_distribution_count": int(trades["action"].eq("DISTRIBUTION").sum()),
        "closed_trade_win_rate": float((exits["pnl_pct"] > 0).mean())
        if len(exits)
        else None,
        "closed_trade_average_pnl": float(exits["pnl_pct"].mean())
        if len(exits)
        else None,
        "stop_loss_sell_count": int(trades["reason"].eq("stop_loss").sum()),
        "rank_exit_sell_count": int(trades["reason"].eq("rank_exit").sum()),
    }


def _run_one(
    *,
    strategy_id: str,
    slots: int,
    buy_rank: int,
    hold_rank: int,
    signal: np.ndarray,
    absolute: np.ndarray,
    mask: np.ndarray,
    execution_mask: np.ndarray,
    dates: pd.DatetimeIndex,
    symbols: np.ndarray,
    terminal: pd.DataFrame,
    protocol: dict[str, Any],
    benchmark: pd.DataFrame,
    start: str | None = None,
    end: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    shared = protocol["shared_trading"]
    split = protocol["split"]
    run_start = start or split["start"]
    run_end = end or split["end"]
    config = ValidationConfig(
        validation_start=run_start,
        validation_end=run_end,
        initial_cash=float(shared["initial_cash"]),
        min_universe=int(protocol["universe"]["min_universe"]),
        slots=slots,
        buy_rank=buy_rank,
        hold_rank=hold_rank,
        robust_z_threshold=float(shared["robust_z_threshold"]),
        stop_loss=float(shared["stop_loss"]),
        metrics_scope="expanded_phase2_exploratory",
    )
    daily, trades, _ = run_formula_validation(
        formula_id="abs_roc_40",
        signal=signal,
        open_prices=absolute[:, 0, :],
        close_prices=absolute[:, 3, :],
        tradable_mask=mask,
        dates=dates,
        symbols=symbols,
        config=config,
        terminal_cashflows=terminal,
        execution_mask=execution_mask,
    )
    daily = daily.drop(columns=["benchmark_equity", "buy_hold_equity"]).merge(
        benchmark, on="date", how="left", validate="one_to_one"
    )
    if daily["benchmark_equity"].isna().any():
        raise RuntimeError("monthly benchmark does not cover all strategy dates")
    summary = {
        "strategy_id": strategy_id,
        "trading": {
            "slots": slots,
            "buy_rank": buy_rank,
            "hold_rank": hold_rank,
            **shared,
        },
        "strategy": {
            **_path_metrics(daily, "equity"),
            "calendar_year_returns": _year_returns(daily, "equity"),
            "average_cash_weight": float(daily["cash_weight"].mean()),
            "average_position_count": float(daily["position_count"].mean()),
            **_trade_metrics(trades),
        },
        "monthly_equal_weight_benchmark": {
            **_path_metrics(daily, "benchmark_equity"),
            "calendar_year_returns": _year_returns(daily, "benchmark_equity"),
        },
    }
    return daily, trades, summary


def main() -> None:
    args = _parse_args()
    protocol = _load_protocol(args.protocol)
    manifest_path = args.dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "expanded-absroc40-product-panel-v1":
        raise RuntimeError("expanded dataset manifest schema mismatch")
    if manifest["protocol_id"] != protocol["protocol_id"]:
        raise RuntimeError("dataset protocol differs from runner protocol")
    identity = dict(manifest)
    dataset_id = identity.pop("dataset_id")
    if dataset_id != "expanded-absroc40-" + canonical_sha256(identity)[:12]:
        raise RuntimeError("expanded dataset ID does not bind the complete manifest")
    if manifest["code_fingerprint"] != code_fingerprint():
        raise RuntimeError("runtime code differs from dataset build code")
    panel_path = args.dataset_dir / manifest["outputs"]["panel"]["file"]
    if _sha256(panel_path) != manifest["outputs"]["panel"]["sha256"]:
        raise RuntimeError("expanded panel hash mismatch")
    with np.load(panel_path, allow_pickle=False) as panel:
        absolute = panel["absolute_ohlc"].astype(np.float64)
        price_mask = panel["price_mask"].astype(bool)
        eligibility = panel["eligibility_mask"].astype(bool)
        signal = panel["abs_roc40"].astype(np.float32)
        symbols = panel["symbols"].astype(str)
        dates = pd.DatetimeIndex(pd.to_datetime(panel["dates"].astype(str)))
    control_panel_path = args.dataset_dir / manifest["outputs"]["original35_panel"]["file"]
    if _sha256(control_panel_path) != manifest["outputs"]["original35_panel"]["sha256"]:
        raise RuntimeError("original35 control panel hash mismatch")
    with np.load(control_panel_path, allow_pickle=False) as panel:
        control_absolute = panel["absolute_ohlc"].astype(np.float64)
        control_price_mask = panel["price_mask"].astype(bool)
        control_mask = panel["eligibility_mask"].astype(bool)
        control_signal = panel["abs_roc40"].astype(np.float32)
        control_symbols = panel["symbols"].astype(str)
        control_dates = pd.DatetimeIndex(pd.to_datetime(panel["dates"].astype(str)))
    terminal_path = args.dataset_dir / manifest["outputs"]["terminal_cashflows"]["file"]
    if _sha256(terminal_path) != manifest["outputs"]["terminal_cashflows"]["sha256"]:
        raise RuntimeError("terminal cashflow hash mismatch")
    terminal = pd.read_csv(
        terminal_path,
        dtype={"fund_code": str},
        parse_dates=["payment_date"],
    )

    expanded_benchmark = _monthly_equal_weight(
        open_prices=absolute[:, 0, :],
        close_prices=absolute[:, 3, :],
        price_mask=price_mask,
        eligibility_mask=eligibility,
        dates=dates,
        start=protocol["split"]["start"],
        end=protocol["split"]["end"],
        min_universe=int(protocol["universe"]["min_universe"]),
        symbols=symbols,
        terminal=terminal,
    )
    outputs: dict[str, tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]] = {}
    for strategy in protocol["strategies"]:
        outputs[strategy["id"]] = _run_one(
            strategy_id=strategy["id"],
            slots=int(strategy["slots"]),
            buy_rank=int(strategy["buy_rank"]),
            hold_rank=int(strategy["hold_rank"]),
            signal=signal,
            absolute=absolute,
            mask=eligibility,
            execution_mask=price_mask,
            dates=dates,
            symbols=symbols,
            terminal=terminal,
            protocol=protocol,
            benchmark=expanded_benchmark,
        )

    control_start = str(protocol["controls"]["original_35_start"])
    control_benchmark = _monthly_equal_weight(
        open_prices=control_absolute[:, 0, :],
        close_prices=control_absolute[:, 3, :],
        price_mask=control_price_mask,
        eligibility_mask=control_mask,
        dates=control_dates,
        start=control_start,
        end=protocol["split"]["end"],
        min_universe=int(protocol["universe"]["min_universe"]),
        symbols=control_symbols,
        terminal=terminal.iloc[0:0],
    )
    outputs["original35_updated"] = _run_one(
        strategy_id="original35_updated",
        slots=3,
        buy_rank=3,
        hold_rank=5,
        signal=control_signal,
        absolute=control_absolute,
        mask=control_mask,
        execution_mask=control_price_mask,
        dates=control_dates,
        symbols=control_symbols,
        terminal=terminal.iloc[0:0],
        protocol=protocol,
        benchmark=control_benchmark,
        start=control_start,
        end=protocol["split"]["end"],
    )

    if args.output_dir.exists():
        raise RuntimeError("phase-two output exists; refusing to overwrite")
    args.output_dir.mkdir(parents=True)
    result = {
        "schema_version": SCHEMA_VERSION,
        "interpretation": "exploratory_selection_sensitivity_not_new_oos_proof",
        "protocol_id": protocol["protocol_id"],
        "dataset_id": manifest["dataset_id"],
        "dataset_manifest_sha256": _sha256(manifest_path),
        "panel_sha256": manifest["outputs"]["panel"]["sha256"],
        "terminal_cashflow_sha256": manifest["outputs"]["terminal_cashflows"]["sha256"],
        "code_commit": git_commit(),
        "code_fingerprint": code_fingerprint(),
        "date_start": protocol["split"]["start"],
        "date_end": protocol["split"]["end"],
        "expanded_universe": {
            "products": manifest["symbols"],
            "eligible_at_start": manifest["eligible_at_start"],
            "eligible_at_end": manifest["eligible_at_end"],
            "max_eligible": manifest["max_eligible"],
            "excluded_terminal_products": manifest["excluded_terminal_products"],
        },
        "runs": {name: value[2] for name, value in outputs.items()},
        "published_original35_reference": {
            "period": "2017-01-03_to_2026-07-10",
            "total_return": 5.6218,
            "annualized_return": 0.2290,
            "max_drawdown": 0.3117,
            "source": "docs/V3A_ABSROC40_2017至今回测结果.md",
        },
    }
    result_path = args.output_dir / "result.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    written = [result_path]
    for name, (daily, trades, _) in outputs.items():
        daily_path = args.output_dir / f"daily_{name}.csv"
        trades_path = args.output_dir / f"trades_{name}.csv"
        daily.to_csv(daily_path, index=False)
        trades.to_csv(trades_path, index=False)
        written.extend([daily_path, trades_path])
    sums = args.output_dir / "SHA256SUMS"
    sums.write_text(
        "\n".join(f"{_sha256(path)}  {path.name}" for path in written) + "\n",
        encoding="ascii",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()