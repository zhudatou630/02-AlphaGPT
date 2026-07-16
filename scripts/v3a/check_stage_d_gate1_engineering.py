#!/usr/bin/env python3
"""Check the frozen 50-batch Stage D Gate 1 engineering thresholds."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--protocol-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_dir = args.run_dir.resolve()
    protocol = json.loads(args.protocol_file.read_text(encoding="utf-8"))
    gate = protocol["engineering_gate"]
    expected_batches = int(gate["batches"])
    expected_attempts = expected_batches * int(protocol["batch_size"])
    rows = [
        json.loads(line)
        for line in (run_dir / "training_log.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != expected_batches:
        raise RuntimeError(f"Gate 1 engineering log has {len(rows)} batches")
    if int(rows[-1]["attempt_count"]) != expected_attempts:
        raise RuntimeError("Gate 1 engineering run stopped at the wrong attempt")
    checkpoint = torch.load(
        run_dir / "checkpoint_latest.pt", map_location="cpu", weights_only=False
    )
    if int(checkpoint["attempt_count"]) != expected_attempts:
        raise RuntimeError("Gate 1 engineering checkpoint boundary mismatch")

    wait_mean = float(np.mean([row["canonical_label_wait_fraction"] for row in rows]))
    teacher_mean = float(np.mean([row["teacher_forcing_fraction"] for row in rows]))
    final_throughput = float(rows[-1]["attempts_per_second"])
    min_cuda_free = min(int(row["cuda_free_bytes"]) for row in rows)
    finite_training = all(
        math.isfinite(float(row["loss"]))
        and math.isfinite(float(row["gradient_norm"]))
        for row in rows
    )
    checks = {
        "canonical_label_wait": {
            "value": wait_mean,
            "threshold": float(gate["max_canonical_label_wait_fraction"]),
            "passed": wait_mean <= float(gate["max_canonical_label_wait_fraction"]),
        },
        "teacher_forcing": {
            "value": teacher_mean,
            "threshold": float(gate["max_teacher_forcing_fraction"]),
            "passed": teacher_mean <= float(gate["max_teacher_forcing_fraction"]),
        },
        "throughput": {
            "value": final_throughput,
            "threshold": float(gate["min_attempts_per_second"]),
            "passed": final_throughput >= float(gate["min_attempts_per_second"]),
        },
        "cuda_free_bytes": {
            "value": min_cuda_free,
            "threshold": int(gate["min_cuda_free_bytes"]),
            "passed": min_cuda_free >= int(gate["min_cuda_free_bytes"]),
        },
        "finite_training": {
            "value": finite_training,
            "threshold": True,
            "passed": finite_training,
        },
    }
    payload = {
        "schema_version": "v3a-stage-d-gate1-engineering-check-v1",
        "run_id": checkpoint["run_id"],
        "attempt_count": expected_attempts,
        "batch_count": expected_batches,
        "checks": checks,
        "passed": all(check["passed"] for check in checks.values()),
        "resume_count": int(checkpoint["resume_count"]),
        "cuda_peak_reserved_bytes": max(
            int(row["cuda_reserved_bytes"]) for row in rows
        ),
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "gate1_engineering_check.json"
    output.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()