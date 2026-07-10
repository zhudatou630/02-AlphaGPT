#!/usr/bin/env python3
"""Catalog legacy V1 code and contaminated research artifacts without moving them."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

ARCHIVE_DIR = ROOT / "archive" / "v1_contaminated"
LEGACY_COMMIT = "2a9ac25e971494a17f6e45a553e2cb040c743b76"
LEGACY_PATHS = (
    ROOT / "data" / "processed" / "etf_panel_raw.npz",
    ROOT / "data" / "processed" / "etf_panel_qfq_latest.npz",
    ROOT / "data" / "processed" / "phase2a",
    ROOT / "data" / "processed" / "phase2b",
    ROOT / "data" / "processed" / "phase3a",
    ROOT / "data" / "processed" / "phase3b",
    ROOT / "data" / "processed" / "phase3c",
    ROOT / "scripts" / "phase1_build_data.py",
    ROOT / "scripts" / "phase2_run_fixed_formulas.py",
    ROOT / "scripts" / "phase2b_run_fixed_formulas.py",
    ROOT / "scripts" / "phase3a_random_formulas.py",
    ROOT / "scripts" / "phase3b_train_transformer.py",
    ROOT / "scripts" / "phase3c_train_gpu.py",
)


def _git_commit() -> str:
    return LEGACY_COMMIT


def _files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(item for item in path.rglob("*") if item.is_file())
    return []


def _git_blob(path: Path) -> bytes | None:
    relative = str(path.relative_to(ROOT))
    try:
        return subprocess.check_output(
            ["git", "show", f"{LEGACY_COMMIT}:{relative}"],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    records = []
    for root_path in LEGACY_PATHS:
        for path in _files(root_path):
            blob = _git_blob(path)
            if blob is None:
                continue
            records.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "size_bytes": len(blob),
                    "sha256": hashlib.sha256(blob).hexdigest(),
                    "snapshot_source": f"git:{LEGACY_COMMIT}",
                }
            )
    manifest = {
        "schema_version": "legacy-research-manifest-v1",
        "legacy_id": "phase1-phase3c-contaminated-v1",
        "status": "frozen_invalid_for_research_conclusions",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "physical_files_moved": False,
        "resume_allowed_in_v2": False,
        "known_issues": [
            "TDX qfq ignored ETF category 11/12 share split and merge events",
            "TDX qfq rounded adjusted prices to two decimals",
            "formula vocabulary exposed raw volume and amount",
            "some TDX historical volume rows were inconsistent with amount and price",
            "legacy checkpoints and artifacts did not record dataset identity",
        ],
        "replacement": {
            "dataset_dir": "data/processed/v2/dataset",
            "training_root": "data/processed/v2/training",
            "vocab_version": "price-event-v2",
        },
        "files": records,
    }
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    output = ARCHIVE_DIR / "manifest.json"
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"manifest: {output}")
    print(f"files: {len(records)}")


if __name__ == "__main__":
    main()
