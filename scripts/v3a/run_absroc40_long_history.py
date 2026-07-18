#!/usr/bin/env python3
"""Run the fixed ABS(ROC(40)) strategy continuously from 2017 to current data."""

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

from alpha_etf.research_v3a.factors import (  # noqa: E402
    FACTOR_NAMES,
    build_factor_values_numpy,
)
from alpha_etf.research_v3a.spec import (  # noqa: E402
    canonical_sha256,
    load_dataset_manifest,
    load_panel,
)
from alpha_etf.research_v3a.validation import (  # noqa: E402
    ValidationConfig,
    run_formula_validation,
)
from scripts.v3a.runtime import code_fingerprint, git_commit  # noqa: E402


SCHEMA_VERSION = "etf-v3a-absroc40-long-history-result-v1"
FORMULA_ID = "abs_roc_40"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    payload = dict(protocol)
    expected = str(payload.pop("protocol_id", ""))
    if canonical_sha256(payload) != expected:
        raise RuntimeError("ABSROC40 long-history protocol ID mismatch")
    if (
        protocol.get("schema_version") != "etf-v3a-absroc40-long-history-v1"
        or protocol["interpretation"] != {
            "exploratory_only": True,
            "formal_winner_allowed": False,
            "window_selection_allowed": False,
            "changes_stage_d_outcome": False,
        }
        or protocol["formula"] != {
            "text": "ABS(ROC(40))",
            "token_ids": [5, 15, 21],
            "token_names": ["ROC", "WIN_40", "ABS"],
        }
        or protocol["split"]["continuous_path"] is not True
        or protocol["split"]["annual_reset"] is not False
    ):
        raise RuntimeError("ABSROC40 long-history protocol semantics drifted")
    return protocol


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _year_returns(daily: pd.DataFrame, column: str, years: list[int]) -> dict[str, float]:
    equity = daily.set_index(pd.to_datetime(daily["date"]))[column].astype(float)
    previous = 1.0
    output: dict[str, float] = {}
    for year in years:
        part = equity[equity.index.year == year]
        if part.empty:
            raise RuntimeError(f"Long-history result has no rows for {year}")
        last = float(part.iloc[-1])
        output[str(year)] = last / previous - 1.0
        previous = last
    return output


def _path_metrics(daily: pd.DataFrame, column: str) -> dict[str, Any]:
    equity = daily[column].to_numpy(dtype=np.float64)
    values = np.r_[1.0, equity]
    peaks = np.maximum.accumulate(values)
    drawdowns = 1.0 - values / peaks
    trough = int(np.argmax(drawdowns))
    peak = int(np.argmax(values[: trough + 1]))
    date_values = ["initial"] + daily["date"].astype(str).tolist()
    total_return = float(values[-1] - 1.0)
    return {
        "total_return": total_return,
        "annualized_return": float((1.0 + total_return) ** (252.0 / len(daily)) - 1.0),
        "max_drawdown": float(drawdowns[trough]),
        "max_drawdown_peak_date": date_values[peak],
        "max_drawdown_trough_date": date_values[trough],
    }


def _trade_metrics(trades: pd.DataFrame) -> dict[str, Any]:
    if trades.empty:
        return {"trade_count": 0, "buy_count": 0, "sell_count": 0}
    sells = trades[trades["action"] == "SELL"].copy()
    if not sells.empty:
        entry = pd.to_datetime(sells["entry_date"])
        exit_ = pd.to_datetime(sells["date"])
        holding_days = (exit_ - entry).dt.days.astype(float)
    else:
        holding_days = pd.Series(dtype=float)
    return {
        "trade_count": int(len(trades)),
        "buy_count": int((trades["action"] == "BUY").sum()),
        "sell_count": int(len(sells)),
        "closed_trade_win_rate": (
            float((sells["pnl_pct"] > 0).mean()) if not sells.empty else None
        ),
        "closed_trade_average_pnl": (
            float(sells["pnl_pct"].mean()) if not sells.empty else None
        ),
        "closed_trade_median_pnl": (
            float(sells["pnl_pct"].median()) if not sells.empty else None
        ),
        "average_holding_calendar_days": (
            float(holding_days.mean()) if not holding_days.empty else None
        ),
        "median_holding_calendar_days": (
            float(holding_days.median()) if not holding_days.empty else None
        ),
        "stop_loss_sell_count": int((sells["reason"] == "stop_loss").sum()),
        "rank_exit_sell_count": int((sells["reason"] == "rank_exit").sum()),
    }


def main() -> None:
    args = _parse_args()
    protocol_path = args.protocol.resolve()
    protocol = _load_protocol(protocol_path)
    dataset_dir = args.dataset_dir.resolve()
    manifest = load_dataset_manifest(dataset_dir)
    panel = load_panel(dataset_dir)
    split = protocol["split"]
    if panel.dates[-1].date().isoformat() != split["end"]:
        raise RuntimeError("Dataset end differs from long-history protocol")
    prior_dates = panel.dates[panel.dates < pd.Timestamp(split["start"])]
    if prior_dates.empty or prior_dates[-1].date().isoformat() != split["prior_signal_date"]:
        raise RuntimeError("Prior signal date differs from long-history protocol")

    factor_values = build_factor_values_numpy(panel.absolute_ohlc, panel.tradable_mask)
    roc40 = factor_values[FACTOR_NAMES.index("ROC_40")]
    signal = np.abs(roc40).astype(np.float32)
    trading = protocol["trading"]
    config = ValidationConfig(
        validation_start=split["start"],
        validation_end=split["end"],
        initial_cash=float(trading["initial_cash"]),
        min_universe=int(trading["min_universe"]),
        slots=int(trading["slots"]),
        buy_rank=int(trading["buy_rank"]),
        hold_rank=int(trading["hold_rank"]),
        robust_z_threshold=float(trading["robust_z_threshold"]),
        stop_loss=float(trading["stop_loss"]),
        metrics_scope="full_history_diagnostic",
    )
    daily, trades, base_summary = run_formula_validation(
        formula_id=FORMULA_ID,
        signal=signal,
        open_prices=panel.absolute("open"),
        close_prices=panel.absolute("close"),
        tradable_mask=panel.tradable_mask,
        dates=panel.dates,
        symbols=panel.symbols,
        config=config,
    )
    if str(daily.iloc[0]["date"]) != split["start"]:
        raise RuntimeError("Long-history result did not start on the frozen date")

    years = [int(year) for year in split["calendar_years"]]
    strategy = {
        **_path_metrics(daily, "equity"),
        "calendar_year_returns": _year_returns(daily, "equity", years),
        "average_cash_weight": float(daily["cash_weight"].mean()),
        "average_position_count": float(daily["position_count"].mean()),
        **_trade_metrics(trades),
    }
    benchmark = {
        **_path_metrics(daily, "benchmark_equity"),
        "calendar_year_returns": _year_returns(daily, "benchmark_equity", years),
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "interpretation": "exploratory_long_history_diagnostic_not_oos_proof",
        "protocol_id": protocol["protocol_id"],
        "code_commit": git_commit(),
        "code_fingerprint": code_fingerprint(),
        "dataset_id": manifest["dataset_id"],
        "panel_sha256": manifest["panel_sha256"],
        "formula": protocol["formula"],
        "date_start": str(daily.iloc[0]["date"]),
        "date_end": str(daily.iloc[-1]["date"]),
        "prior_signal_date": split["prior_signal_date"],
        "trading": trading,
        "strategy": strategy,
        "benchmark": benchmark,
        "base_summary": base_summary,
        "formal_winner_generated": False,
        "window_selection_performed": False,
        "changes_stage_d_outcome": False,
    }

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise RuntimeError("Long-history output exists; refusing to overwrite it")
    output_dir.mkdir(parents=True)
    result_path = output_dir / "result.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    daily.to_csv(output_dir / "daily.csv", index=False)
    trades.to_csv(output_dir / "trades.csv", index=False)
    names = ("result.json", "daily.csv", "trades.csv")
    (output_dir / "SHA256SUMS").write_text(
        "\n".join(f"{_sha256(output_dir / name)}  {name}" for name in names) + "\n",
        encoding="ascii",
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()