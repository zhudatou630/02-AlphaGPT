#!/usr/bin/env python3
"""Build the immutable dual-layer dataset used by the V3A research line."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from alpha_etf.research_v3a.spec import (
    ABSOLUTE_FEATURES,
    DATASET_SCHEMA_VERSION,
    EFFECTIVE_START,
    EXPECTED_UPSTREAM_V3_ID,
    MIN_UNIVERSE,
    PRICE_ADJUSTMENT,
    RELATIVE_TRANSFORM,
    TRADABLE_MASK_SEMANTICS,
    build_dataset_manifest,
    build_panel_arrays,
    sha256_file,
)
from alpha_etf.data.relative_price import RELATIVE_FEATURES


def _project_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--governance",
        type=Path,
        default=ROOT / "data/processed/v3/governance/etf_daily_event_adjusted.parquet",
    )
    parser.add_argument(
        "--upstream-dir", type=Path, default=ROOT / "data/processed/v3/dataset"
    )
    parser.add_argument(
        "--universe", type=Path, default=ROOT / "configs/research_v3_universe.json"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "data/processed/v3a/dataset"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    upstream_manifest_path = args.upstream_dir / "dataset_manifest.json"
    upstream_panel_path = args.upstream_dir / "panel_relative_price_v3.npz"
    upstream_manifest = json.loads(upstream_manifest_path.read_text(encoding="utf-8"))
    if upstream_manifest.get("dataset_id") != EXPECTED_UPSTREAM_V3_ID:
        raise RuntimeError(
            f"Upstream V3 dataset mismatch: {upstream_manifest.get('dataset_id')}"
        )
    actual_upstream_hash = sha256_file(upstream_panel_path)
    if actual_upstream_hash != upstream_manifest.get("panel_sha256"):
        raise RuntimeError(
            f"Upstream V3 panel hash mismatch: {actual_upstream_hash} != "
            f"{upstream_manifest.get('panel_sha256')}"
        )
    if upstream_manifest.get("quality_gate_failures"):
        raise RuntimeError(
            f"Upstream V3 quality gate is not clean: "
            f"{upstream_manifest['quality_gate_failures']}"
        )

    universe = json.loads(args.universe.read_text(encoding="utf-8"))
    symbols = [str(item["symbol"]) for item in universe]
    governance = pd.read_parquet(args.governance)
    with np.load(upstream_panel_path, allow_pickle=False) as upstream:
        required = {"values", "mask", "symbols", "features", "dates"}
        missing = required - set(upstream.files)
        if missing:
            raise ValueError(f"Upstream V3 panel missing keys: {sorted(missing)}")
        upstream_relative = upstream["values"].astype(np.float64)
        upstream_mask = upstream["mask"].astype(bool)
        upstream_symbols = upstream["symbols"].astype(str)
        upstream_features = tuple(upstream["features"].astype(str))
        upstream_dates = upstream["dates"].astype(str)

    if list(upstream_symbols) != symbols:
        raise ValueError("Universe symbols differ from upstream V3 panel")
    if upstream_features != RELATIVE_FEATURES:
        raise ValueError(f"Upstream relative features differ: {upstream_features}")

    absolute, relative, mask, dates = build_panel_arrays(
        governance,
        symbols=symbols,
        upstream_relative=upstream_relative,
        upstream_mask=upstream_mask,
        upstream_dates=upstream_dates,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel_file = "panel_v3a.npz"
    panel_path = args.output_dir / panel_file
    np.savez_compressed(
        panel_path,
        absolute_ohlc=absolute,
        relative_ohlc=relative,
        tradable_mask=mask,
        symbols=np.asarray(symbols, dtype=str),
        dates=dates.strftime("%Y-%m-%d").to_numpy(dtype=str),
        absolute_features=np.asarray(ABSOLUTE_FEATURES, dtype=str),
        relative_features=np.asarray(RELATIVE_FEATURES, dtype=str),
    )

    identity = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "panel_file": panel_file,
        "panel_sha256": sha256_file(panel_path),
        "upstream_v3_manifest_file": _project_path(upstream_manifest_path),
        "upstream_v3_manifest_sha256": sha256_file(upstream_manifest_path),
        "upstream_v3_dataset_id": upstream_manifest["dataset_id"],
        "upstream_v3_panel_sha256": upstream_manifest["panel_sha256"],
        "governance_file": _project_path(args.governance),
        "governance_sha256": sha256_file(args.governance),
        "universe_file": _project_path(args.universe),
        "universe_sha256": sha256_file(args.universe),
        "symbols": symbols,
        "absolute_features": list(ABSOLUTE_FEATURES),
        "relative_features": list(RELATIVE_FEATURES),
        "absolute_shape": list(absolute.shape),
        "relative_shape": list(relative.shape),
        "mask_shape": list(mask.shape),
        "date_start": dates.min().date().isoformat(),
        "date_end": dates.max().date().isoformat(),
        "date_count": int(len(dates)),
        "tradable_observations": int(mask.sum()),
        "effective_start": EFFECTIVE_START,
        "min_universe": MIN_UNIVERSE,
        "price_adjustment": PRICE_ADJUSTMENT,
        "relative_transform": RELATIVE_TRANSFORM,
        "tradable_mask": TRADABLE_MASK_SEMANTICS,
        "flow_features_exposed": False,
    }
    manifest = build_dataset_manifest(identity)
    manifest_path = args.output_dir / "dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()