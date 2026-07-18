#!/usr/bin/env python3
"""Compare ROC(40), -ROC(40), and ABS(ROC(40)) under identical trading rules."""

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


SCHEMA_VERSION = "etf-v3a-roc40-directional-comparison-result-v1"
EXPECTED_FORMULAS = [
    {"id": "roc_40", "text": "ROC(40)", "transform": "identity"},
    {"id": "neg_roc_40", "text": "NEG(ROC(40))", "transform": "negative"},
    {
        "id": "abs_roc_40",
        "text": "ABS(ROC(40))",
        "transform": "absolute",
        "role": "existing_control",
    },
]


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
    interpretation = protocol.get("interpretation", {})
    if canonical_sha256(payload) != expected:
        raise RuntimeError("ROC40 directional comparison protocol ID mismatch")
    if (
        protocol.get("schema_version") != "etf-v3a-roc40-directional-comparison-v1"
        or protocol.get("formulas") != EXPECTED_FORMULAS
        or interpretation.get("exploratory_only") is not True
        or interpretation.get("designed_after_absroc40_results_read") is not True
        or interpretation.get("formal_winner_allowed") is not False
        or interpretation.get("parameter_selection_allowed") is not False
        or protocol["split"]["continuous_path"] is not True
        or protocol["split"]["annual_reset"] is not False
    ):
        raise RuntimeError("ROC40 directional comparison protocol semantics drifted")
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
            raise RuntimeError(f"Directional comparison has no rows for {year}")
        last = float(part.iloc[-1])
        output[str(year)] = last / previous - 1.0
        previous = last
    return output


def _path_metrics(daily: pd.DataFrame, column: str) -> dict[str, Any]:
    values = np.r_[1.0, daily[column].to_numpy(dtype=np.float64)]
    peaks = np.maximum.accumulate(values)
    drawdowns = 1.0 - values / peaks
    trough = int(np.argmax(drawdowns))
    peak = int(np.argmax(values[: trough + 1]))
    dates = ["initial", *daily["date"].astype(str).tolist()]
    total = float(values[-1] - 1.0)
    return {
        "total_return": total,
        "annualized_return": float((1.0 + total) ** (252.0 / len(daily)) - 1.0),
        "max_drawdown": float(drawdowns[trough]),
        "max_drawdown_peak_date": dates[peak],
        "max_drawdown_trough_date": dates[trough],
    }


def _buy_direction_counts(
    trades: pd.DataFrame,
    roc40: np.ndarray,
    dates: pd.DatetimeIndex,
    symbols: np.ndarray,
) -> dict[str, Any]:
    date_index = {date.date().isoformat(): index for index, date in enumerate(dates)}
    symbol_index = {str(symbol): index for index, symbol in enumerate(symbols)}
    values = np.asarray(
        [
            roc40[symbol_index[str(row.symbol)], date_index[str(row.decision_date)]]
            for row in trades[trades["action"] == "BUY"].itertuples()
        ],
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        raise RuntimeError("Directional comparison buy has invalid original ROC(40)")
    return {
        "positive_count": int((values > 0).sum()),
        "negative_count": int((values < 0).sum()),
        "zero_count": int((values == 0).sum()),
        "positive_share": float((values > 0).mean()),
        "negative_share": float((values < 0).mean()),
        "mean_original_roc40": float(values.mean()),
        "median_original_roc40": float(np.median(values)),
    }


def main() -> None:
    args = _parse_args()
    protocol = _load_protocol(args.protocol.resolve())
    dataset_dir = args.dataset_dir.resolve()
    manifest = load_dataset_manifest(dataset_dir)
    panel = load_panel(dataset_dir)
    split = protocol["split"]
    if panel.dates[-1].date().isoformat() != split["end"]:
        raise RuntimeError("Directional comparison dataset end differs from protocol")
    prior = panel.dates[panel.dates < pd.Timestamp(split["start"])]
    if prior.empty or prior[-1].date().isoformat() != split["prior_signal_date"]:
        raise RuntimeError("Directional comparison prior signal date mismatch")

    factors = build_factor_values_numpy(panel.absolute_ohlc, panel.tradable_mask)
    roc40 = factors[FACTOR_NAMES.index("ROC_40")].astype(np.float32)
    signals = {
        "roc_40": roc40,
        "neg_roc_40": (-roc40).astype(np.float32),
        "abs_roc_40": np.abs(roc40).astype(np.float32),
    }
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
    years = [int(year) for year in split["calendar_years"]]
    strategies: dict[str, Any] = {}
    daily_frames: list[pd.DataFrame] = []
    trade_frames: list[pd.DataFrame] = []
    first_daily: pd.DataFrame | None = None
    for formula in protocol["formulas"]:
        formula_id = formula["id"]
        daily, trades, base = run_formula_validation(
            formula_id=formula_id,
            signal=signals[formula_id],
            open_prices=panel.absolute("open"),
            close_prices=panel.absolute("close"),
            tradable_mask=panel.tradable_mask,
            dates=panel.dates,
            symbols=panel.symbols,
            config=config,
        )
        if first_daily is None:
            first_daily = daily
        path = _path_metrics(daily, "equity")
        strategies[formula_id] = {
            "formula": formula,
            **path,
            "calendar_year_returns": _year_returns(daily, "equity", years),
            "average_cash_weight": float(daily["cash_weight"].mean()),
            "average_position_count": float(daily["position_count"].mean()),
            "trade_count": int(len(trades)),
            "buy_count": int((trades["action"] == "BUY").sum()),
            "sell_count": int((trades["action"] == "SELL").sum()),
            "stop_loss_sell_count": int(
                ((trades["action"] == "SELL") & (trades["reason"] == "stop_loss")).sum()
            ),
            "rank_exit_sell_count": int(
                ((trades["action"] == "SELL") & (trades["reason"] == "rank_exit")).sum()
            ),
            "buy_original_roc40": _buy_direction_counts(
                trades, roc40, panel.dates, panel.symbols
            ),
            "base_summary": base,
        }
        daily_frames.append(daily)
        trade_frames.append(trades)
    assert first_daily is not None
    benchmark = {
        **_path_metrics(first_daily, "benchmark_equity"),
        "calendar_year_returns": _year_returns(
            first_daily, "benchmark_equity", years
        ),
    }
    abs_control = strategies["abs_roc_40"]
    result = {
        "schema_version": SCHEMA_VERSION,
        "interpretation": "post_abs_result_exploratory_directional_comparison",
        "protocol_id": protocol["protocol_id"],
        "code_commit": git_commit(),
        "code_fingerprint": code_fingerprint(),
        "dataset_id": manifest["dataset_id"],
        "panel_sha256": manifest["panel_sha256"],
        "date_start": str(first_daily.iloc[0]["date"]),
        "date_end": str(first_daily.iloc[-1]["date"]),
        "prior_signal_date": split["prior_signal_date"],
        "trading": trading,
        "benchmark": benchmark,
        "strategies": strategies,
        "abs_differences": {
            formula_id: {
                "total_return": abs_control["total_return"] - strategies[formula_id]["total_return"],
                "annualized_return": abs_control["annualized_return"] - strategies[formula_id]["annualized_return"],
                "max_drawdown": abs_control["max_drawdown"] - strategies[formula_id]["max_drawdown"],
            }
            for formula_id in ("roc_40", "neg_roc_40")
        },
        "formal_winner_generated": False,
        "parameter_selection_performed": False,
        "changes_stage_d_outcome": False,
    }
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise RuntimeError("Directional comparison output exists; refusing to overwrite it")
    output_dir.mkdir(parents=True)
    result_path = output_dir / "result.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    pd.concat(daily_frames, ignore_index=True).to_csv(output_dir / "daily.csv", index=False)
    pd.concat(trade_frames, ignore_index=True).to_csv(output_dir / "trades.csv", index=False)
    names = ("result.json", "daily.csv", "trades.csv")
    (output_dir / "SHA256SUMS").write_text(
        "\n".join(f"{_sha256(output_dir / name)}  {name}" for name in names) + "\n",
        encoding="ascii",
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()