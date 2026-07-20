#!/usr/bin/env python3
"""Run the one-shot SW2021 ABS(ROC40) cross-universe validation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from alpha_etf.research_v3a.validation import ValidationConfig, run_formula_validation  # noqa: E402
from alpha_etf.sw_industry.spec import (  # noqa: E402
    load_dataset_manifest,
    load_panel,
    sha256_file,
)
from alpha_etf.sw_industry.validation import (  # noqa: E402
    build_absroc40_signal,
    load_protocol,
    validate_protocol_dataset,
)


RESULT_SCHEMA_VERSION = "sw2021-l1-absroc40-cross-universe-result-v1"
FORMULA_ID = "abs_roc_40"
CLAIM_ROOT = ROOT / ".pi/profile/results/sw-industry-validation-claims"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def code_fingerprint() -> str:
    paths = [
        ROOT / "src/alpha_etf/research_v3a/factors.py",
        ROOT / "src/alpha_etf/research_v3a/validation.py",
        ROOT / "src/alpha_etf/sw_industry/spec.py",
        ROOT / "src/alpha_etf/sw_industry/validation.py",
        Path(__file__).resolve(),
    ]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def reserve_one_shot_run(
    protocol: dict[str, Any],
    manifest: dict[str, Any],
    output_dir: Path,
    *,
    claim_root: Path = CLAIM_ROOT,
) -> Path:
    claim_root.mkdir(parents=True, exist_ok=True)
    claim_path = claim_root / f"{protocol['protocol_id']}.json"
    payload = {
        "schema_version": "sw2021-l1-validation-one-shot-claim-v1",
        "claimed_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_id": protocol["protocol_id"],
        "dataset_id": manifest["dataset_id"],
        "panel_sha256": manifest["panel_sha256"],
        "code_fingerprint": code_fingerprint(),
        "output_dir": str(output_dir),
        "status": "reserved_before_result_computation",
    }
    try:
        descriptor = os.open(claim_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(
            f"One-shot validation protocol has already been claimed: {claim_path}"
        ) from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    descriptor = os.open(claim_root, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return claim_path


def path_metrics(daily: pd.DataFrame, column: str) -> dict[str, Any]:
    equity = daily[column].to_numpy(dtype=np.float64)
    values = np.r_[1.0, equity]
    peaks = np.maximum.accumulate(values)
    drawdowns = 1.0 - values / peaks
    trough = int(np.argmax(drawdowns))
    peak = int(np.argmax(values[: trough + 1]))
    labels = ["initial"] + daily["date"].astype(str).tolist()
    total_return = float(values[-1] - 1.0)
    return {
        "total_return": total_return,
        "annualized_return": float((1.0 + total_return) ** (252.0 / len(daily)) - 1.0),
        "max_drawdown": float(drawdowns[trough]),
        "max_drawdown_peak_date": labels[peak],
        "max_drawdown_trough_date": labels[trough],
    }


def annual_returns(daily: pd.DataFrame, column: str) -> dict[str, float]:
    values = daily.set_index(pd.to_datetime(daily["date"]))[column].astype(float)
    previous = 1.0
    output: dict[str, float] = {}
    for year in sorted(values.index.year.unique()):
        last = float(values[values.index.year == year].iloc[-1])
        output[str(year)] = last / previous - 1.0
        previous = last
    return output


def enrich_and_reconcile_trades(
    trades: pd.DataFrame,
    *,
    roc40: np.ndarray,
    dates: pd.DatetimeIndex,
    symbols: np.ndarray,
    names: np.ndarray,
    final_close: np.ndarray,
    final_cash: float,
    final_equity: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    symbol_index = {str(symbol): index for index, symbol in enumerate(symbols)}
    name_by_symbol = dict(zip(symbols.astype(str), names.astype(str)))
    date_index = {date.date().isoformat(): index for index, date in enumerate(dates)}
    active: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    realized_pnl = 0.0
    for source in trades.to_dict("records"):
        row = dict(source)
        symbol = str(row["symbol"])
        row["name"] = name_by_symbol[symbol]
        if row["action"] == "BUY":
            asset = symbol_index[symbol]
            decision = str(row["decision_date"])
            entry_roc40 = float(roc40[asset, date_index[decision]])
            metadata = {
                "entry_roc40": entry_roc40,
                "entry_direction": "positive" if entry_roc40 >= 0 else "negative",
                "cost": float(row["notional"]),
                "qty": float(row["qty"]),
                "entry_date": str(row["date"]),
            }
            if symbol in active:
                raise RuntimeError(f"Duplicate active position for {symbol}")
            active[symbol] = metadata
            row.update(metadata)
            row["realized_pnl"] = np.nan
        else:
            if symbol not in active:
                raise RuntimeError(f"Sell without active position for {symbol}")
            metadata = active.pop(symbol)
            pnl = float(row["notional"]) - float(metadata["cost"])
            realized_pnl += pnl
            row.update(metadata)
            row["realized_pnl"] = pnl
        records.append(row)
    enriched = pd.DataFrame(records)
    open_positions = []
    unrealized_pnl = 0.0
    market_value = 0.0
    for symbol, metadata in sorted(active.items()):
        asset = symbol_index[symbol]
        value = float(metadata["qty"] * final_close[asset])
        pnl = value - float(metadata["cost"])
        market_value += value
        unrealized_pnl += pnl
        open_positions.append(
            {
                "symbol": symbol,
                "name": name_by_symbol[symbol],
                "entry_date": metadata["entry_date"],
                "entry_direction": metadata["entry_direction"],
                "entry_roc40": metadata["entry_roc40"],
                "cost": metadata["cost"],
                "market_value": value,
                "unrealized_pnl": pnl,
            }
        )
    reconstructed_equity = float(final_cash + market_value)
    pnl_equity = float(1.0 + realized_pnl + unrealized_pnl)
    if abs(reconstructed_equity - final_equity) > 1e-10:
        raise RuntimeError("Final position market value does not reconcile to equity")
    if abs(pnl_equity - final_equity) > 1e-10:
        raise RuntimeError("Realized and unrealized PnL do not reconcile to equity")
    accounting = {
        "realized_pnl": realized_pnl,
        "unrealized_pnl": unrealized_pnl,
        "final_cash": final_cash,
        "final_position_market_value": market_value,
        "final_equity": final_equity,
        "reconstructed_equity": reconstructed_equity,
        "pnl_reconstructed_equity": pnl_equity,
        "open_positions": open_positions,
    }
    return enriched, accounting


def trade_metrics(trades: pd.DataFrame) -> dict[str, Any]:
    if trades.empty:
        return {
            "trade_count": 0,
            "buy_count": 0,
            "sell_count": 0,
            "closed_trade_win_rate": None,
            "closed_trade_average_return": None,
            "closed_trade_median_return": None,
            "stop_loss_sell_count": 0,
            "rank_exit_sell_count": 0,
            "positive_roc_buy_count": 0,
            "negative_roc_buy_count": 0,
        }
    sells = trades[trades["action"] == "SELL"].copy()
    buys = trades[trades["action"] == "BUY"].copy()
    return {
        "trade_count": len(trades),
        "buy_count": len(buys),
        "sell_count": len(sells),
        "closed_trade_win_rate": float((sells["pnl_pct"] > 0).mean()) if len(sells) else None,
        "closed_trade_average_return": float(sells["pnl_pct"].mean()) if len(sells) else None,
        "closed_trade_median_return": float(sells["pnl_pct"].median()) if len(sells) else None,
        "stop_loss_sell_count": int((sells["reason"] == "stop_loss").sum()),
        "rank_exit_sell_count": int((sells["reason"] == "rank_exit").sum()),
        "positive_roc_buy_count": int((buys["entry_direction"] == "positive").sum()),
        "negative_roc_buy_count": int((buys["entry_direction"] == "negative").sum()),
    }


def write_text_synced(path: Path, content: str) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    args = parse_args()
    protocol_path = args.protocol.resolve()
    dataset_dir = args.dataset_dir.resolve()
    protocol = load_protocol(protocol_path)
    manifest = load_dataset_manifest(dataset_dir)
    validate_protocol_dataset(protocol, manifest)
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Validation output already exists: {output_dir}")
    claim_path = reserve_one_shot_run(protocol, manifest, output_dir)
    panel = load_panel(dataset_dir)
    roc40, signal = build_absroc40_signal(panel)
    split = protocol["split"]
    finite_all = np.isfinite(signal).all(axis=0)
    first_signal = int(np.flatnonzero(finite_all)[0])
    if panel.dates[first_signal].date().isoformat() != split["prior_signal_date"]:
        raise RuntimeError("First complete ROC40 signal date differs from protocol")
    if panel.dates[first_signal + 1].date().isoformat() != split["validation_start"]:
        raise RuntimeError("First execution date differs from protocol")
    if panel.dates[-1].date().isoformat() != split["validation_end"]:
        raise RuntimeError("Panel end differs from validation protocol")

    trading = protocol["trading"]
    config = ValidationConfig(
        validation_start=split["validation_start"],
        validation_end=split["validation_end"],
        initial_cash=float(trading["initial_cash"]),
        min_universe=int(trading["min_universe"]),
        slots=int(trading["slots"]),
        buy_rank=int(trading["buy_rank"]),
        hold_rank=int(trading["hold_rank"]),
        robust_z_threshold=float(trading["robust_z_threshold"]),
        stop_loss=float(trading["stop_loss"]),
        metrics_scope="validation",
    )
    daily, raw_trades, base_summary = run_formula_validation(
        formula_id=FORMULA_ID,
        signal=signal,
        open_prices=panel.absolute("open"),
        close_prices=panel.absolute("close"),
        tradable_mask=panel.tradable_mask,
        dates=panel.dates,
        symbols=panel.symbols,
        config=config,
    )
    if str(daily.iloc[0]["date"]) != split["validation_start"]:
        raise RuntimeError("Validation output starts on the wrong date")

    trades, accounting = enrich_and_reconcile_trades(
        raw_trades,
        roc40=roc40,
        dates=panel.dates,
        symbols=panel.symbols,
        names=panel.names,
        final_close=panel.absolute("close")[:, -1],
        final_cash=float(daily.iloc[-1]["cash"]),
        final_equity=float(daily.iloc[-1]["equity"]),
    )
    strategy = {
        **path_metrics(daily, "equity"),
        "calendar_year_returns": annual_returns(daily, "equity"),
        "average_cash_weight": float(daily["cash_weight"].mean()),
        "average_position_count": float(daily["position_count"].mean()),
        **trade_metrics(trades),
    }
    benchmark = {
        **path_metrics(daily, "benchmark_equity"),
        "calendar_year_returns": annual_returns(daily, "benchmark_equity"),
    }
    buy_hold = path_metrics(daily, "buy_hold_equity")
    annual = pd.DataFrame(
        {
            "year": list(strategy["calendar_year_returns"]),
            "strategy_return": list(strategy["calendar_year_returns"].values()),
            "benchmark_return": list(benchmark["calendar_year_returns"].values()),
        }
    )
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "interpretation": "cross_universe_replication_not_executable_etf_backtest",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256_file(protocol_path),
        "dataset_id": manifest["dataset_id"],
        "panel_sha256": manifest["panel_sha256"],
        "code_commit": git_commit(),
        "code_fingerprint": code_fingerprint(),
        "one_shot_claim": str(claim_path),
        "one_shot_claim_sha256": sha256_file(claim_path),
        "formula": protocol["formula"],
        "date_start": str(daily.iloc[0]["date"]),
        "date_end": str(daily.iloc[-1]["date"]),
        "prior_signal_date": split["prior_signal_date"],
        "trading": trading,
        "strategy": strategy,
        "benchmark": benchmark,
        "buy_hold": buy_hold,
        "accounting": accounting,
        "base_summary": base_summary,
        "parameter_tuning_performed": False,
        "result_dependent_protocol_change_performed": False,
        "run_sequence": 1,
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        result_path = temporary / "result.json"
        daily_path = temporary / "daily.csv"
        trades_path = temporary / "trades.csv"
        annual_path = temporary / "annual_summary.csv"
        write_text_synced(
            result_path,
            json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        )
        daily.to_csv(daily_path, index=False)
        trades.to_csv(trades_path, index=False)
        annual.to_csv(annual_path, index=False)
        for path in (daily_path, trades_path, annual_path):
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
        outputs = (result_path, daily_path, trades_path, annual_path)
        write_text_synced(
            temporary / "SHA256SUMS",
            "\n".join(f"{sha256_file(path)}  {path.name}" for path in outputs) + "\n",
        )
        descriptor = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, output_dir)
        descriptor = os.open(output_dir.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    print(f"result: {output_dir}")


if __name__ == "__main__":
    main()
