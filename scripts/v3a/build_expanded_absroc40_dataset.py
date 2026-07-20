#!/usr/bin/env python3
"""Build the frozen product-level dataset for expanded ABS(ROC(40)) backtests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from alpha_etf.research_v3a.spec import canonical_sha256  # noqa: E402
from alpha_etf.data.event_adjustment import (  # noqa: E402
    apply_event_adjustments,
    build_event_audit,
    quality_summary,
)
from scripts.v3a.runtime import code_fingerprint, git_commit  # noqa: E402


SCHEMA_VERSION = "expanded-absroc40-product-panel-v1"
FEATURES = ("open", "high", "low", "close")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    base = ROOT / "data" / "processed" / "expanded_etf_audit"
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs" / "v3a_expanded_absroc40_phase2.json",
    )
    parser.add_argument("--phase1-dir", type=Path, default=base / "phase1")
    parser.add_argument("--adjusted-dir", type=Path, default=base / "event_adjusted")
    parser.add_argument("--source-dir", type=Path, default=base / "sources")
    parser.add_argument("--output-dir", type=Path, default=base / "phase2_dataset")
    parser.add_argument(
        "--control-tdx-dir", type=Path, default=base / "original35_control" / "tdx"
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_protocol(path: Path) -> dict[str, object]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    payload = dict(protocol)
    expected = str(payload.pop("protocol_id"))
    if canonical_sha256(payload) != expected:
        raise RuntimeError("expanded phase-two protocol ID mismatch")
    if protocol["schema_version"] != "etf-v3a-expanded-absroc40-phase2-v1":
        raise RuntimeError("expanded phase-two protocol schema mismatch")
    return protocol


def _terminal_events(
    phase1_dir: Path, excluded: set[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    coverage = pd.read_csv(
        phase1_dir / "terminal_cashflow_coverage.csv", dtype={"fund_code": str}
    )
    ended_representatives = set(coverage["fund_code"])
    expected_excluded = set(
        coverage.loc[
            coverage["cashflow_status"].isin(
                ["missing_distribution_announcement", "non_cash_lifecycle_event"]
            )
            | pd.to_numeric(
                coverage["derived_per_share_count"], errors="coerce"
            ).fillna(0).gt(0),
            "fund_code",
        ]
    )
    if expected_excluded != excluded:
        raise RuntimeError("protocol terminal exclusion set differs from evidence table")

    candidates = pd.read_csv(
        phase1_dir / "terminal_cashflow_candidates.csv", dtype={"fund_code": str}
    )
    direct = candidates[
        candidates["fund_code"].isin(ended_representatives - excluded)
        & candidates["cash_value_source"].eq("announcement_per_share")
    ][["fund_code", "payment_date", "cash_per_share", "attachment_url"]].copy()
    direct["evidence_type"] = "announcement_per_share"
    overrides = pd.read_csv(
        ROOT / "configs" / "expanded_etf_terminal_cashflow_overrides.csv",
        dtype={"ts_code": str},
    ).rename(
        columns={
            "ts_code": "fund_code",
            "source_url": "attachment_url",
        }
    )
    overrides["fund_code"] = overrides["fund_code"].str[:6]
    overrides = overrides[
        overrides["fund_code"].isin(ended_representatives - excluded)
    ][["fund_code", "payment_date", "cash_per_share", "attachment_url", "evidence_type"]]
    terminal = pd.concat([direct, overrides], ignore_index=True)
    terminal["cash_per_share"] = pd.to_numeric(terminal["cash_per_share"])
    terminal["payment_date"] = pd.to_datetime(terminal["payment_date"])
    terminal = terminal.sort_values(["payment_date", "fund_code"]).reset_index(drop=True)
    terminal["is_final_payment"] = False
    terminal.loc[terminal.groupby("fund_code").tail(1).index, "is_final_payment"] = True

    included = coverage[~coverage["fund_code"].isin(excluded)].copy()
    actual = terminal.groupby("fund_code")["cash_per_share"].sum()
    expected = pd.to_numeric(included.set_index("fund_code")["total_cash_per_share"])
    missing = set(expected.index) - set(actual.index)
    if missing:
        raise RuntimeError(f"included ended representatives lack terminal cash: {sorted(missing)}")
    if not np.allclose(
        actual.reindex(expected.index).to_numpy(),
        expected.to_numpy(),
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError("terminal event sums differ from terminal coverage totals")
    excluded_rows = coverage[coverage["fund_code"].isin(excluded)].sort_values(
        "fund_code"
    )
    return terminal, excluded_rows


def _roc40_signal(
    absolute: np.ndarray, price_mask: np.ndarray
) -> np.ndarray:
    close = absolute[:, FEATURES.index("close"), :]
    roc40 = np.full_like(close, np.nan, dtype=np.float64)
    valid = (
        price_mask[:, 40:]
        & price_mask[:, :-40]
        & np.isfinite(close[:, 40:])
        & np.isfinite(close[:, :-40])
        & (close[:, :-40] > 0)
    )
    np.divide(close[:, 40:], close[:, :-40], out=roc40[:, 40:], where=valid)
    roc40[:, 40:] -= np.where(valid, 1.0, np.nan)
    return np.abs(roc40).astype(np.float32)


def _build_original35_control(
    control_dir: Path, dates: pd.DatetimeIndex
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    universe = json.loads(
        (ROOT / "configs" / "research_v3_universe.json").read_text(encoding="utf-8")
    )
    symbols = np.asarray([str(row["symbol"]) for row in universe], dtype=str)
    symbol_index = {symbol: index for index, symbol in enumerate(symbols)}
    raw = pd.read_json(control_dir / "tdx_daily_bfq.jsonl", lines=True)
    raw["symbol"] = raw["symbol"].astype(str).str.zfill(6)
    raw["date"] = pd.to_datetime(raw["date"])
    raw["tradable"] = raw["volume"].gt(0) & raw["amount"].gt(0)
    events = pd.read_json(control_dir / "tdx_gbbq_events.jsonl", lines=True)
    events["symbol"] = events["symbol"].astype(str).str.zfill(6)
    events["date"] = pd.to_datetime(events["date"])
    audit = build_event_audit(raw, events)
    adjusted = apply_event_adjustments(raw, audit)
    quality = quality_summary(adjusted, audit, threshold=0.201)
    failures = {
        "adjusted_return_breaches": int(quality["adjusted_abs_return_gt_threshold"].sum()),
        "invalid_adjusted_prices": int(quality["invalid_adjusted_price_rows"].sum()),
        "invalid_adjusted_high": int(quality["invalid_adjusted_high_rows"].sum()),
        "invalid_adjusted_low": int(quality["invalid_adjusted_low_rows"].sum()),
    }
    failures = {key: value for key, value in failures.items() if value}
    if failures:
        raise RuntimeError(f"original35 control price gate failed: {failures}")

    date_index = {date: index for index, date in enumerate(dates)}
    absolute = np.full(
        (len(symbols), len(FEATURES), len(dates)), np.nan, dtype=np.float64
    )
    price_mask = np.zeros((len(symbols), len(dates)), dtype=bool)
    for row in adjusted.itertuples(index=False):
        if row.symbol not in symbol_index or pd.Timestamp(row.date) not in date_index:
            continue
        asset = symbol_index[row.symbol]
        day = date_index[pd.Timestamp(row.date)]
        values = np.asarray(
            [getattr(row, f"event_qfq_{feature}") for feature in FEATURES],
            dtype=np.float64,
        )
        if bool(row.tradable) and np.isfinite(values).all() and (values > 0).all():
            absolute[asset, :, day] = values
            price_mask[asset, day] = True
    signal = _roc40_signal(absolute, price_mask)
    eligibility = price_mask & np.isfinite(signal)
    summary = {
        "products": len(symbols),
        "price_observations": int(price_mask.sum()),
        "event_rows": len(audit),
        "applied_events": int(audit["applied"].astype(bool).sum()),
    }
    return absolute, price_mask, eligibility, signal, summary


def main() -> None:
    args = _parse_args()
    protocol = _load_protocol(args.protocol)
    excluded = set(protocol["terminal_events"]["excluded_products"])
    adjusted_path = args.adjusted_dir / "etf_daily_event_adjusted.parquet"
    adjusted_manifest_path = args.adjusted_dir / "manifest.json"
    adjusted_manifest = json.loads(
        adjusted_manifest_path.read_text(encoding="utf-8")
    )
    adjusted = pd.read_parquet(adjusted_path)
    if {"open", "high", "low", "close"} & set(adjusted.columns):
        raise RuntimeError("adjusted source exposes ambiguous bare OHLC columns")
    required = {
        "symbol",
        "date",
        "tradable",
        *(f"event_qfq_{feature}" for feature in FEATURES),
    }
    missing = required - set(adjusted.columns)
    if missing:
        raise RuntimeError(f"adjusted source missing fields: {sorted(missing)}")

    master = pd.read_csv(args.phase1_dir / "product_master.csv", dtype=str)
    target = master[
        master["scope_decision"].eq("include_domestic_passive_equity")
    ].drop_duplicates("symbol")
    symbols = np.asarray(sorted(target["symbol"].astype(str)), dtype=str)
    symbol_index = {symbol: index for index, symbol in enumerate(symbols)}
    calendar = pd.read_parquet(args.source_dir / "tushare_sse_open_calendar.parquet")
    dates = pd.DatetimeIndex(
        pd.to_datetime(calendar["cal_date"], format="%Y%m%d").sort_values().unique()
    )
    dates = dates[dates <= pd.Timestamp(protocol["split"]["end"])]
    date_index = {date: index for index, date in enumerate(dates)}

    absolute = np.full(
        (len(symbols), len(FEATURES), len(dates)), np.nan, dtype=np.float64
    )
    price_mask = np.zeros((len(symbols), len(dates)), dtype=bool)
    adjusted = adjusted[adjusted["symbol"].astype(str).isin(symbol_index)].copy()
    adjusted["symbol"] = adjusted["symbol"].astype(str)
    adjusted["date"] = pd.to_datetime(adjusted["date"]).dt.normalize()
    adjusted = adjusted[adjusted["date"].isin(date_index)]
    for row in adjusted.itertuples(index=False):
        asset = symbol_index[row.symbol]
        day = date_index[pd.Timestamp(row.date)]
        values = np.asarray(
            [getattr(row, f"event_qfq_{feature}") for feature in FEATURES],
            dtype=np.float64,
        )
        usable = bool(row.tradable) and np.isfinite(values).all() and (values > 0).all()
        if usable:
            absolute[asset, :, day] = values
            price_mask[asset, day] = True

    abs_roc40 = _roc40_signal(absolute, price_mask)

    eligibility = pd.read_csv(
        args.phase1_dir / "representative_eligibility.csv",
        parse_dates=[
            "representative_from",
            "representative_to",
            "warmup_date",
        ],
        dtype={"representative_symbol": str},
    )
    eligibility_mask = np.zeros_like(price_mask)
    variety_assignments: list[dict[str, object]] = []
    for row in eligibility.itertuples(index=False):
        symbol = str(row.representative_symbol)
        if symbol in excluded or pd.isna(row.warmup_date):
            continue
        start = max(pd.Timestamp(row.representative_from), pd.Timestamp(row.warmup_date))
        end = min(pd.Timestamp(row.representative_to), dates[-1])
        selected_dates = dates[(dates >= start) & (dates <= end)]
        asset = symbol_index[symbol]
        indices = np.asarray([date_index[date] for date in selected_dates], dtype=int)
        eligibility_mask[asset, indices] = True
        variety_assignments.append(
            {
                "variety_id": row.variety_id,
                "representative_symbol": symbol,
                "eligible_from": start,
                "eligible_to": end,
            }
        )
    eligibility_mask &= price_mask & np.isfinite(abs_roc40)

    assignment = pd.DataFrame(variety_assignments)
    duplicate_variety_days = 0
    for _, group in assignment.groupby("variety_id"):
        ordered = group.sort_values("eligible_from")
        previous_end = pd.Timestamp.min
        for row in ordered.itertuples(index=False):
            if pd.Timestamp(row.eligible_from) <= previous_end:
                duplicate_variety_days += 1
            previous_end = max(previous_end, pd.Timestamp(row.eligible_to))
    if duplicate_variety_days:
        raise RuntimeError("multiple representatives are eligible for one variety on one date")

    start_index = date_index[pd.Timestamp(protocol["split"]["start"])]
    end_index = date_index[pd.Timestamp(protocol["split"]["end"])]
    eligible_counts = eligibility_mask.sum(axis=0)
    if eligible_counts[start_index] < int(protocol["universe"]["min_universe"]):
        raise RuntimeError("frozen start has fewer than the required eligible varieties")
    if (eligible_counts[start_index : end_index + 1] < 10).any():
        raise RuntimeError("eligible variety count falls below 10 in the frozen period")

    terminal, excluded_rows = _terminal_events(args.phase1_dir, excluded)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel_path = args.output_dir / "panel.npz"
    np.savez_compressed(
        panel_path,
        absolute_ohlc=absolute,
        price_mask=price_mask,
        eligibility_mask=eligibility_mask,
        abs_roc40=abs_roc40,
        symbols=symbols,
        dates=dates.strftime("%Y-%m-%d").to_numpy(dtype=str),
        features=np.asarray(FEATURES, dtype=str),
    )
    control_absolute, control_price_mask, control_eligibility, control_signal, control_summary = (
        _build_original35_control(args.control_tdx_dir, dates)
    )
    control_panel_path = args.output_dir / "original35_panel.npz"
    np.savez_compressed(
        control_panel_path,
        absolute_ohlc=control_absolute,
        price_mask=control_price_mask,
        eligibility_mask=control_eligibility,
        abs_roc40=control_signal,
        symbols=np.asarray(
            [
                str(row["symbol"])
                for row in json.loads(
                    (ROOT / "configs" / "research_v3_universe.json").read_text(
                        encoding="utf-8"
                    )
                )
            ],
            dtype=str,
        ),
        dates=dates.strftime("%Y-%m-%d").to_numpy(dtype=str),
        features=np.asarray(FEATURES, dtype=str),
    )
    terminal_path = args.output_dir / "terminal_cashflows.csv"
    excluded_path = args.output_dir / "excluded_terminal_products.csv"
    assignment_path = args.output_dir / "representative_assignments.csv"
    count_path = args.output_dir / "eligible_counts.csv"
    terminal.to_csv(terminal_path, index=False)
    excluded_rows.to_csv(excluded_path, index=False)
    assignment.to_csv(assignment_path, index=False)
    pd.DataFrame(
        {"date": dates, "eligible_variety_count": eligible_counts}
    ).to_csv(count_path, index=False)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "interpretation": "frozen_phase_two_dataset_no_strategy_results",
        "protocol_id": protocol["protocol_id"],
        "code_commit": git_commit(),
        "code_fingerprint": code_fingerprint(),
        "symbols": len(symbols),
        "dates": len(dates),
        "date_start": dates[0].date().isoformat(),
        "date_end": dates[-1].date().isoformat(),
        "backtest_start": protocol["split"]["start"],
        "backtest_end": protocol["split"]["end"],
        "price_features": list(FEATURES),
        "price_fields": [f"event_qfq_{feature}" for feature in FEATURES],
        "raw_ohlc_exposed": False,
        "price_observations": int(price_mask.sum()),
        "eligible_observations": int(eligibility_mask.sum()),
        "eligible_at_start": int(eligible_counts[start_index]),
        "eligible_at_end": int(eligible_counts[end_index]),
        "max_eligible": int(eligible_counts[start_index : end_index + 1].max()),
        "excluded_terminal_products": len(excluded),
        "terminal_cashflow_products": int(terminal["fund_code"].nunique()),
        "terminal_cashflow_rows": len(terminal),
        "original35_control": control_summary,
        "inputs": {
            "protocol": _sha256(args.protocol),
            "adjusted_daily": _sha256(adjusted_path),
            "adjusted_manifest": _sha256(adjusted_manifest_path),
            "adjusted_dataset_id": adjusted_manifest["schema_version"],
            "representative_eligibility": _sha256(
                args.phase1_dir / "representative_eligibility.csv"
            ),
            "terminal_coverage": _sha256(
                args.phase1_dir / "terminal_cashflow_coverage.csv"
            ),
            "trade_calendar": _sha256(
                args.source_dir / "tushare_sse_open_calendar.parquet"
            ),
            "original35_universe": _sha256(
                ROOT / "configs" / "research_v3_universe.json"
            ),
            "original35_tdx_bfq": _sha256(
                args.control_tdx_dir / "tdx_daily_bfq.jsonl"
            ),
            "original35_tdx_events": _sha256(
                args.control_tdx_dir / "tdx_gbbq_events.jsonl"
            ),
        },
        "outputs": {
            "panel": {"file": panel_path.name, "sha256": _sha256(panel_path)},
            "original35_panel": {
                "file": control_panel_path.name,
                "sha256": _sha256(control_panel_path),
            },
            "terminal_cashflows": {
                "file": terminal_path.name,
                "sha256": _sha256(terminal_path),
            },
            "excluded_products": {
                "file": excluded_path.name,
                "sha256": _sha256(excluded_path),
            },
            "assignments": {
                "file": assignment_path.name,
                "sha256": _sha256(assignment_path),
            },
            "eligible_counts": {
                "file": count_path.name,
                "sha256": _sha256(count_path),
            },
        },
    }
    manifest["dataset_id"] = "expanded-absroc40-" + canonical_sha256(manifest)[:12]
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()