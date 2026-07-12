#!/usr/bin/env python3
"""Build the immutable train-only array view consumed by Stage D GPU runs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from alpha_etf.research_v3a.candidates import CandidateConfig
from alpha_etf.research_v3a.factors import FACTOR_NAMES, build_factor_values_numpy
from alpha_etf.research_v3a.scoring import ScorerConfig
from alpha_etf.research_v3a.spec import load_dataset_manifest, load_panel, sha256_file
from alpha_etf.research_v3a.stage_d import (
    TRAIN_VIEW_SCHEMA_VERSION,
    build_train_view_manifest,
    load_stage_d_train_view,
)
from scripts.v3a.runtime import (
    DATASET_DIR,
    build_runtime_research_spec,
    require_clean_v3a_code,
)


DEFAULT_OUT_DIR = ROOT / "data/processed/v3a/stage_d/train_view"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def _write_npy(path: Path, value: np.ndarray) -> None:
    with path.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    args = parse_args()
    require_clean_v3a_code()
    dataset_manifest = load_dataset_manifest(args.dataset_dir)
    research_spec = build_runtime_research_spec(
        dataset_manifest,
        scorer_config=ScorerConfig(),
        candidate_config=CandidateConfig(),
    )
    if args.out_dir.exists():
        view = load_stage_d_train_view(
            args.out_dir,
            research_spec=research_spec,
        )
        print(json.dumps(view.manifest, ensure_ascii=False, indent=2))
        print(f"output: {args.out_dir}")
        return

    panel = load_panel(args.dataset_dir)
    train_end = int(
        np.searchsorted(panel.dates.values, np.datetime64("2021-12-31"), side="right")
    )
    dates = panel.dates[:train_end].strftime("%Y-%m-%d").to_numpy(dtype="U10")
    mask = panel.tradable_mask[:, :train_end].astype(bool, copy=True)
    absolute = panel.absolute_ohlc[:, :, :train_end]
    absolute_open = absolute[:, 0, :].astype(np.float64, copy=True)
    factor_values = build_factor_values_numpy(absolute, mask).astype(
        np.float64, copy=False
    )
    symbols = panel.symbols.astype(str)

    args.out_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{args.out_dir.name}.", dir=args.out_dir.parent)
    )
    try:
        arrays = {
            "factor_values": factor_values,
            "absolute_open": absolute_open,
            "tradable_mask": mask,
            "symbols": symbols,
            "dates": dates,
        }
        files: dict[str, Any] = {}
        for name, value in arrays.items():
            filename = f"{name}.npy"
            path = temporary / filename
            _write_npy(path, value)
            files[name] = {
                "path": filename,
                "sha256": sha256_file(path),
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
        manifest = build_train_view_manifest(
            {
                "schema_version": TRAIN_VIEW_SCHEMA_VERSION,
                "source_dataset_id": dataset_manifest["dataset_id"],
                "source_panel_sha256": dataset_manifest["panel_sha256"],
                "source_dataset_manifest": dataset_manifest,
                "research_spec_id": research_spec["research_spec_id"],
                "code_commit": research_spec["code_commit"],
                "code_fingerprint": research_spec["code_fingerprint"],
                "split": {
                    "signal_start": "2016-08-09",
                    "data_end": "2021-12-31",
                    "validation_columns_present": False,
                    "final_columns_present": False,
                },
                "factor_names": list(FACTOR_NAMES),
                "factor_shape": list(factor_values.shape),
                "mask_shape": list(mask.shape),
                "date_start": str(dates[0]),
                "date_end": str(dates[-1]),
                "files": files,
            }
        )
        _write_json(temporary / "train_view_manifest.json", manifest)
        os.replace(temporary, args.out_dir)
        directory = os.open(args.out_dir.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    view = load_stage_d_train_view(
        args.out_dir,
        research_spec=research_spec,
    )
    print(json.dumps(view.manifest, ensure_ascii=False, indent=2))
    print(f"output: {args.out_dir}")


if __name__ == "__main__":
    main()