#!/usr/bin/env python3
"""Bind committed validation code to frozen candidates without reading validation data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from alpha_etf.research_v3a.spec import canonical_sha256, sha256_file  # noqa: E402
from scripts.v3a.prepare_stage_d_validation import _load_protocol  # noqa: E402
from scripts.v3a.runtime import (  # noqa: E402
    code_fingerprint,
    git_commit,
    require_clean_v3a_code,
)


SCHEMA_VERSION = "etf-v3a-validation-preflight-binding-v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "configs/v3a_stage_d_validation.json"
    )
    parser.add_argument("--train-view-manifest", type=Path, required=True)
    parser.add_argument("--cuda-manifest", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--sanity-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_sha256s(path: Path) -> None:
    for line in (path / "SHA256SUMS").read_text(encoding="ascii").splitlines():
        expected, name = line.split("  ", 1)
        if sha256_file(path / name) != expected:
            raise RuntimeError(f"Validation preflight SHA mismatch: {path / name}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    args = _parse_args()
    protocol_path = args.protocol.resolve()
    protocol = _load_protocol(protocol_path)
    require_clean_v3a_code(extra_paths=(protocol_path,))
    candidate_dir = args.candidate_dir.resolve()
    sanity_dir = args.sanity_dir.resolve()
    _verify_sha256s(candidate_dir)
    _verify_sha256s(sanity_dir)
    train_manifest = _load_json(args.train_view_manifest.resolve())
    cuda_manifest = _load_json(args.cuda_manifest.resolve())
    candidate_manifest = _load_json(candidate_dir / "final_candidate_manifest.json")
    sanity = _load_json(sanity_dir / "sanity_summary.json")
    identities = (candidate_manifest, sanity, cuda_manifest)
    if any(item["protocol_id"] != protocol["protocol_id"] for item in identities):
        raise RuntimeError("Validation preflight protocol identities differ")
    if any(item["validation_or_final_metrics_read"] is not False for item in identities):
        raise RuntimeError("Validation preflight artifact read sealed metrics")
    if (
        candidate_manifest["cuda_audit_complete"] is not True
        or candidate_manifest["validation_run_approved"] is not False
        or train_manifest["split"]["validation_columns_present"] is not False
        or train_manifest["split"]["final_columns_present"] is not False
        or train_manifest["date_end"] != "2021-12-31"
    ):
        raise RuntimeError("Validation preflight boundary is not closed")
    source = train_manifest["source_dataset_manifest"]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256_file(protocol_path),
        "code_commit": git_commit(),
        "code_fingerprint": code_fingerprint(),
        "dataset_id": source["dataset_id"],
        "panel_sha256": source["panel_sha256"],
        "train_view_id": train_manifest["train_view_id"],
        "train_view_end": train_manifest["date_end"],
        "cuda_device": cuda_manifest["cuda_device"],
        "cuda_manifest_sha256": sha256_file(args.cuda_manifest.resolve()),
        "candidate_manifest_sha256": sha256_file(
            candidate_dir / "final_candidate_manifest.json"
        ),
        "final_groups_sha256": sha256_file(candidate_dir / "final_groups.json"),
        "numerical_gates_sha256": sha256_file(
            candidate_dir / "numerical_gates.jsonl"
        ),
        "sanity_summary_sha256": sha256_file(sanity_dir / "sanity_summary.json"),
        "validation_view_built": False,
        "validation_run_approved": False,
        "validation_metrics_read": False,
        "final_metrics_read": False,
    }
    binding = {**payload, "binding_id": canonical_sha256(payload)}
    _write_json(args.output.resolve(), binding)
    print(json.dumps(binding, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()