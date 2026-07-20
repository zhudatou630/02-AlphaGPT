#!/usr/bin/env python3
"""Build the immutable SW2021 level-1 industry research panel."""

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

from alpha_etf.sw_industry.spec import (  # noqa: E402
    ABSOLUTE_FEATURES,
    DATASET_SCHEMA_VERSION,
    build_dataset_manifest,
    build_panel_arrays,
    load_panel,
    sha256_file,
)


DEFAULT_GOVERNED = ROOT / (
    "data/processed/sw_industry_l1/sw2021_backcast/governed_snapshots/"
    "governed-v2-sw2021-current-history-20140221-20260717-20260719T041322Z"
)
DEFAULT_OUTPUT = ROOT / "data/processed/sw_industry_l1/sw2021_backcast/dataset"


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
    governed_manifest_path = governed_dir / "manifest.json"
    governed_path = governed_dir / "sw2021_l1_daily_governed.parquet"
    governed_manifest = json.loads(governed_manifest_path.read_text(encoding="utf-8"))
    if governed_manifest.get("schema_version") != "sw2021-l1-governed-current-history-v2":
        raise RuntimeError("Dataset build requires the governed-v2 snapshot")
    expected_hash = governed_manifest["files"][governed_path.name]["sha256"]
    if sha256_file(governed_path) != expected_hash:
        raise RuntimeError("Governed SW2021 parquet SHA mismatch")
    quality = governed_manifest["quality"]
    if quality.get("blocking_quality_failures") or not quality.get("close_research_gate_passed"):
        raise RuntimeError("Governed SW2021 quality gate is not clean")
    if quality.get("high_repairs") != 2 or quality.get("return_field_rows_repaired") != 27:
        raise RuntimeError("Governed SW2021 repair contract differs")

    frame = pd.read_parquet(governed_path)
    absolute, mask, symbols, names, dates = build_panel_arrays(frame)
    if args.output_dir.exists():
        raise FileExistsError(f"Dataset output already exists: {args.output_dir}")
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{args.output_dir.name}.", dir=args.output_dir.parent)
    )
    try:
        panel_path = temporary / "panel_sw2021_l1.npz"
        np.savez_compressed(
            panel_path,
            absolute_ohlc=absolute,
            tradable_mask=mask,
            symbols=symbols,
            names=names,
            dates=dates.strftime("%Y-%m-%d").to_numpy(dtype=str),
            features=np.asarray(ABSOLUTE_FEATURES, dtype=str),
        )
        identity = {
            "schema_version": DATASET_SCHEMA_VERSION,
            "panel_file": panel_path.name,
            "panel_sha256": sha256_file(panel_path),
            "source_governed_dir": project_path(governed_dir),
            "source_governed_manifest_sha256": sha256_file(governed_manifest_path),
            "source_governed_parquet_sha256": sha256_file(governed_path),
            "source_snapshot_id": governed_manifest["source_snapshot_id"],
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
            "price_adjustment": "none_published_index_levels",
            "return_policy": "compute from consecutive close; do not use source return fields",
            "repairs": {
                "ohlc_high_rows": int(quality["high_repairs"]),
                "source_return_field_rows": int(quality["return_field_rows_repaired"]),
            },
        }
        manifest = build_dataset_manifest(identity)
        manifest_path = temporary / "dataset_manifest.json"
        write_json_synced(manifest_path, manifest)
        os.replace(temporary, args.output_dir)
        directory = os.open(args.output_dir.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    loaded = load_panel(args.output_dir)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2))
    print(f"panel shape: {loaded.absolute_ohlc.shape}")
    print(f"dataset: {args.output_dir}")


if __name__ == "__main__":
    main()