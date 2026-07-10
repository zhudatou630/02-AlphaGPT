#!/usr/bin/env python3
"""Build the V2 price-only panel and immutable dataset manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from alpha_etf.config import ETF_UNIVERSE
from alpha_etf.research_v2.spec import (
    DATASET_SCHEMA_VERSION,
    PRICE_FEATURES,
    canonical_sha256,
    sha256_file,
)


def _project_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT
        / "data"
        / "processed"
        / "phase1b_event_adjusted"
        / "etf_daily_event_adjusted.parquet",
    )
    parser.add_argument(
        "--phase1b-summary",
        type=Path,
        default=ROOT / "data" / "processed" / "phase1b_event_adjusted" / "build_summary.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "processed" / "v2" / "dataset",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = pd.read_parquet(args.source)
    required = {
        "date",
        "symbol",
        "tradable",
        "event_qfq_open",
        "event_qfq_high",
        "event_qfq_low",
        "event_qfq_close",
    }
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"Phase1b source missing columns: {sorted(missing)}")
    phase1b_summary = json.loads(args.phase1b_summary.read_text(encoding="utf-8"))
    if phase1b_summary.get("quality_gate_failures"):
        raise RuntimeError(f"Phase1b quality gate is not clean: {phase1b_summary['quality_gate_failures']}")

    source = source.copy()
    source["date"] = pd.to_datetime(source["date"]).dt.normalize()
    source["symbol"] = source["symbol"].astype(str)
    duplicate = source.duplicated(["symbol", "date"], keep=False)
    if duplicate.any():
        examples = source.loc[duplicate, ["symbol", "date"]].head(10).to_dict("records")
        raise ValueError(f"Phase1b source has duplicate symbol/date rows: {examples}")

    symbols = [etf.symbol for etf in ETF_UNIVERSE]
    dates = pd.DatetimeIndex(sorted(source["date"].unique()))
    values = np.full((len(symbols), len(PRICE_FEATURES), len(dates)), np.nan, dtype=np.float64)
    mask = np.zeros((len(symbols), len(dates)), dtype=bool)
    symbol_index = {symbol: idx for idx, symbol in enumerate(symbols)}
    date_index = {date: idx for idx, date in enumerate(dates)}
    source_columns = {feature: f"event_qfq_{feature}" for feature in PRICE_FEATURES}
    for row in source.itertuples(index=False):
        symbol = str(row.symbol)
        if symbol not in symbol_index or not bool(row.tradable):
            continue
        n = symbol_index[symbol]
        t = date_index[pd.Timestamp(row.date)]
        for feature_idx, feature in enumerate(PRICE_FEATURES):
            values[n, feature_idx, t] = float(getattr(row, source_columns[feature]))
        mask[n, t] = bool(np.isfinite(values[n, :, t]).all())

    sample_view = values.transpose(0, 2, 1)
    if not np.isfinite(sample_view[mask]).all() or not np.isnan(sample_view[~mask]).all():
        raise ValueError("V2 panel finite/NaN contract failed")
    if not (sample_view[mask] > 0).all():
        raise ValueError("V2 tradable prices must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel_file = "panel_price_event_v2.npz"
    panel_path = args.output_dir / panel_file
    np.savez_compressed(
        panel_path,
        values=values,
        mask=mask,
        symbols=np.asarray(symbols, dtype=str),
        features=np.asarray(PRICE_FEATURES, dtype=str),
        dates=dates.strftime("%Y-%m-%d").to_numpy(dtype=str),
    )
    identity = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "panel_file": panel_file,
        "panel_sha256": sha256_file(panel_path),
        "source_file": _project_path(args.source),
        "source_sha256": sha256_file(args.source),
        "phase1b_summary_file": _project_path(args.phase1b_summary),
        "phase1b_summary_sha256": sha256_file(args.phase1b_summary),
        "features": list(PRICE_FEATURES),
        "price_adjustment": "multiplicative_event_qfq",
        "flow_features_exposed": False,
        "tradable_mask": "positive reconciled volume and amount; nontradable values are NaN",
        "symbols": symbols,
        "date_start": dates.min().date().isoformat(),
        "date_end": dates.max().date().isoformat(),
        "date_count": int(len(dates)),
        "tradable_observations": int(mask.sum()),
        "panel_shape": list(values.shape),
    }
    fingerprint = canonical_sha256(identity)
    manifest = {
        **identity,
        "dataset_id": f"etf-price-event-v2-{fingerprint[:12]}",
        "dataset_fingerprint": fingerprint,
    }
    manifest_path = args.output_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()