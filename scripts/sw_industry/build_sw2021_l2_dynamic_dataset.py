#!/usr/bin/env python3
"""Build the immutable SW2021 L2 dynamic-history research panel."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from alpha_etf.sw_industry.l2_spec import (  # noqa: E402
    ABSOLUTE_FEATURES,
    DATASET_SCHEMA_VERSION,
    build_dataset_manifest,
    build_dynamic_panel_arrays,
    load_panel,
)
from alpha_etf.sw_industry.spec import sha256_file  # noqa: E402


DEFAULT_GOVERNED = ROOT / (
    "data/processed/sw_industry_l2/sw2021_dynamic/governed_snapshots/"
    "governed-v2-sw2021-l2-dynamic-history-20000315-20260717-20260719T083826Z"
)
DEFAULT_OUTPUT = ROOT / "data/processed/sw_industry_l2/sw2021_dynamic/dataset-v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--governed-dir", type=Path, default=DEFAULT_GOVERNED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def project_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def write_json_synced(path: Path, payload: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    args = parse_args()
    governed_dir = args.governed_dir.resolve()
    manifest_path = governed_dir / "manifest.json"
    data_path = governed_dir / "sw2021_l2_dynamic_daily_governed.parquet"
    availability_source = governed_dir / "availability.csv"
    governed_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if governed_manifest.get("schema_version") != "sw2021-l2-dynamic-governed-history-v2":
        raise RuntimeError("Dataset build requires the dynamic governed-v2 snapshot")
    for path in (data_path, availability_source):
        if sha256_file(path) != governed_manifest["files"][path.name]["sha256"]:
            raise RuntimeError(f"Governed SW2021 L2 file SHA mismatch: {path.name}")
    quality = governed_manifest["quality"]
    if quality.get("blocking_quality_failures") or not quality.get(
        "close_research_gate_passed"
    ):
        raise RuntimeError("Governed SW2021 L2 quality gate is not clean")

    frame = pd.read_parquet(data_path)
    absolute, mask, symbols, names, dates = build_dynamic_panel_arrays(frame)
    signal_mask = np.zeros_like(mask)
    signal_mask[:, 40:] = mask[:, 40:] & mask[:, :-40]
    quote_counts = mask.sum(axis=0)
    signal_counts = signal_mask.sum(axis=0)
    first_rankable_index = int(np.flatnonzero(signal_counts >= 10)[0])
    if args.output_dir.exists():
        raise FileExistsError(f"Dataset output already exists: {args.output_dir}")
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{args.output_dir.name}.", dir=args.output_dir.parent)
    )
    try:
        panel_path = temporary / "panel_sw2021_l2_dynamic.npz"
        np.savez_compressed(
            panel_path,
            absolute_ohlc=absolute,
            tradable_mask=mask,
            symbols=symbols,
            names=names,
            dates=dates.strftime("%Y-%m-%d").to_numpy(dtype=str),
            features=np.asarray(ABSOLUTE_FEATURES, dtype=str),
        )
        availability = pd.read_csv(availability_source, dtype={"ts_code": str})
        first_signal_by_symbol = {}
        for asset, symbol in enumerate(symbols):
            indices = np.flatnonzero(signal_mask[asset])
            first_signal_by_symbol[str(symbol)] = (
                dates[int(indices[0])].date().isoformat() if len(indices) else None
            )
        availability["first_roc40_signal_date"] = availability["ts_code"].map(
            first_signal_by_symbol
        )
        availability_path = temporary / "availability.csv"
        availability.to_csv(availability_path, index=False)
        with availability_path.open("rb") as handle:
            os.fsync(handle.fileno())
        identity = {
            "schema_version": DATASET_SCHEMA_VERSION,
            "panel_file": panel_path.name,
            "panel_sha256": sha256_file(panel_path),
            "availability_file": availability_path.name,
            "availability_sha256": sha256_file(availability_path),
            "source_governed_dir": project_path(governed_dir),
            "source_governed_manifest_sha256": sha256_file(manifest_path),
            "source_governed_parquet_sha256": sha256_file(data_path),
            "source_snapshot_id": governed_manifest["source_snapshot_id"],
            "identity_config_sha256": governed_manifest["identity_config_sha256"],
            "identity_policy": governed_manifest["identity_policy"],
            "entry_policy": governed_manifest["entry_policy"],
            "series_semantics": governed_manifest["series_semantics"],
            "historical_vintage_proven": governed_manifest["historical_vintage_proven"],
            "symbols": symbols.tolist(),
            "names": names.tolist(),
            "features": list(ABSOLUTE_FEATURES),
            "panel_shape": list(absolute.shape),
            "mask_shape": list(mask.shape),
            "date_start": dates[0].date().isoformat(),
            "date_end": dates[-1].date().isoformat(),
            "date_count": len(dates),
            "tradable_observations": int(mask.sum()),
            "first_day_quote_universe": int(quote_counts[0]),
            "final_day_quote_universe": int(quote_counts[-1]),
            "first_rankable_signal_date": dates[first_rankable_index].date().isoformat(),
            "first_execution_date": dates[first_rankable_index + 1].date().isoformat(),
            "roc40_warmup": "global_exchange_date_t_minus_40_no_valid_row_compression",
            "price_adjustment": "none_published_index_levels",
            "return_policy": "compute from consecutive global-date close; source returns unused",
            "internal_quote_gaps": governed_manifest["source_dynamic_quality"][
                "internal_quote_gap_examples"
            ],
            "repairs": {
                "ohlc_rows": int(quality["high_repairs"] + quality["low_repairs"]),
                "source_return_field_rows": int(quality["return_field_rows_repaired"]),
            },
        }
        manifest = build_dataset_manifest(identity)
        dataset_manifest_path = temporary / "dataset_manifest.json"
        write_json_synced(dataset_manifest_path, manifest)
        os.replace(temporary, args.output_dir)
        descriptor = os.open(args.output_dir.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    loaded = load_panel(args.output_dir)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2))
    print(f"panel shape: {loaded.absolute_ohlc.shape}")
    print(f"dataset: {args.output_dir}")


if __name__ == "__main__":
    main()