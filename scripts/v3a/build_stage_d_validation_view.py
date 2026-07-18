#!/usr/bin/env python3
"""Build the authorized, physically final-free 2022 validation view."""

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

from alpha_etf.research_v3a.factors import FACTOR_NAMES, build_factor_values_numpy  # noqa: E402
from alpha_etf.research_v3a.spec import (  # noqa: E402
    canonical_sha256,
    load_dataset_manifest,
    load_panel,
    sha256_file,
)
from alpha_etf.research_v3a.validation_view import (  # noqa: E402
    VALIDATION_VIEW_SCHEMA_VERSION,
    build_validation_view_manifest,
    load_validation_view,
)
from scripts.v3a.prepare_stage_d_validation import _load_protocol  # noqa: E402
from scripts.v3a.runtime import code_fingerprint, git_commit, require_clean_v3a_code  # noqa: E402


APPROVAL_SCHEMA_VERSION = "etf-v3a-validation-approval-v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--preflight-binding", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_approval(
    approval: dict[str, Any], protocol: dict[str, Any], binding: dict[str, Any]
) -> None:
    payload = dict(approval)
    expected = str(payload.pop("approval_id", ""))
    if approval.get("schema_version") != APPROVAL_SCHEMA_VERSION:
        raise RuntimeError("Validation approval schema mismatch")
    if canonical_sha256(payload) != expected:
        raise RuntimeError("Validation approval ID mismatch")
    if (
        approval.get("protocol_id") != protocol["protocol_id"]
        or approval.get("preflight_binding_id") != binding["binding_id"]
        or approval.get("validation_run_approved") is not True
        or approval.get("final_metrics_approved") is not False
        or approval.get("scope")
        != "build_2022_validation_view_and_run_frozen_validation_once"
    ):
        raise RuntimeError("Validation approval scope or identity mismatch")


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
    args = _parse_args()
    protocol_path = args.protocol.resolve()
    require_clean_v3a_code(extra_paths=(protocol_path,))
    protocol = _load_protocol(protocol_path)
    approval = _load_json(args.approval.resolve())
    binding = _load_json(args.preflight_binding.resolve())
    _validate_approval(approval, protocol, binding)
    if (
        binding["code_commit"] != git_commit()
        or binding["code_fingerprint"] != code_fingerprint()
        or binding["validation_view_built"] is not False
        or binding["validation_run_approved"] is not False
    ):
        raise RuntimeError("Validation preflight binding differs from committed runtime")
    dataset_dir = args.dataset_dir.resolve()
    dataset_manifest = load_dataset_manifest(dataset_dir)
    if (
        dataset_manifest["dataset_id"] != binding["dataset_id"]
        or dataset_manifest["panel_sha256"] != binding["panel_sha256"]
    ):
        raise RuntimeError("Validation source dataset differs from preflight binding")
    if args.output_dir.exists():
        view = load_validation_view(args.output_dir.resolve())
        if view.manifest["approval_id"] != approval["approval_id"]:
            raise RuntimeError("Existing validation view used another approval")
        print(json.dumps(view.manifest, ensure_ascii=False, indent=2))
        return

    # The full source artifact is loaded only inside this authorized boundary builder;
    # every array passed to factor computation is sliced before 2023.
    panel = load_panel(dataset_dir)
    cutoff = int(
        np.searchsorted(panel.dates.values, np.datetime64("2022-12-31"), side="right")
    )
    dates = panel.dates[:cutoff].strftime("%Y-%m-%d").to_numpy(dtype="U10")
    if dates[-1] != "2022-12-30" or np.any(dates >= "2023-01-01"):
        raise RuntimeError("Validation source slice crossed the final boundary")
    mask = panel.tradable_mask[:, :cutoff].astype(bool, copy=True)
    absolute = panel.absolute_ohlc[:, :, :cutoff].astype(np.float64, copy=True)
    factor_values = build_factor_values_numpy(absolute, mask)
    arrays = {
        "factor_values": factor_values.astype(np.float64, copy=False),
        "absolute_open": absolute[:, 0, :].copy(),
        "absolute_close": absolute[:, 3, :].copy(),
        "tradable_mask": mask,
        "symbols": panel.symbols.astype(str),
        "dates": dates,
    }
    output_dir = args.output_dir.resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
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
        manifest = build_validation_view_manifest(
            {
                "schema_version": VALIDATION_VIEW_SCHEMA_VERSION,
                "protocol_id": protocol["protocol_id"],
                "approval_id": approval["approval_id"],
                "preflight_binding_id": binding["binding_id"],
                "source_dataset_id": dataset_manifest["dataset_id"],
                "source_panel_sha256": dataset_manifest["panel_sha256"],
                "source_dataset_manifest": dataset_manifest,
                "code_commit": git_commit(),
                "code_fingerprint": code_fingerprint(),
                "split": {
                    "signal_start": "2016-08-09",
                    "validation_start": "2022-01-01",
                    "validation_end": "2022-12-31",
                    "data_end": "2022-12-31",
                    "validation_columns_present": True,
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
        _write_json(temporary / "validation_view_manifest.json", manifest)
        os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    view = load_validation_view(output_dir)
    print(json.dumps(view.manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()