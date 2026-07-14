"""Small Stage D training checkpoints bound to durable ledger positions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from alpha_etf.research_v3a.attempts import (
    ATTEMPT_DTYPE,
    ATTEMPT_LEDGER_SCHEMA_VERSION,
)
from alpha_etf.research_v3a.checkpointing import (
    atomic_save,
    restore_rng_state,
    rng_state_dict,
)


STAGE_D_CHECKPOINT_SCHEMA_VERSION = "etf-v3a-stage-d-checkpoint-v2"


def build_training_checkpoint(
    *,
    run_id: str,
    run_identity: dict[str, Any],
    step: int,
    attempt_count: int,
    model: torch.nn.Module | None,
    optimizer: torch.optim.Optimizer | None,
    ledger_prefix_sha256: str,
    training_log_offset: int,
    training_log_prefix_sha256: str,
    candidate_snapshot_name: str | None,
    candidate_snapshot_attempt: int,
    candidate_snapshot_elapsed_seconds: float,
    candidate_snapshot_sha256: str,
    elapsed_seconds: float,
    resume_count: int,
) -> dict[str, Any]:
    if (model is None) != (optimizer is None):
        raise ValueError("Stage D model and optimizer checkpoint states must be paired")
    if step < 0 or attempt_count < 0 or candidate_snapshot_attempt < 0:
        raise ValueError("Stage D checkpoint counters must be non-negative")
    if candidate_snapshot_attempt > attempt_count:
        raise ValueError("Stage D candidate snapshot is ahead of the checkpoint")
    if len(candidate_snapshot_sha256) != 64:
        raise ValueError("Stage D candidate snapshot SHA-256 is invalid")
    return {
        "schema_version": STAGE_D_CHECKPOINT_SCHEMA_VERSION,
        "run_id": run_id,
        "run_identity": dict(run_identity),
        "step": int(step),
        "attempt_count": int(attempt_count),
        "model_state_dict": model.state_dict() if model is not None else None,
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "rng_state": rng_state_dict(),
        "ledger": {
            "schema_version": ATTEMPT_LEDGER_SCHEMA_VERSION,
            "record_size": ATTEMPT_DTYPE.itemsize,
            "record_count": int(attempt_count),
            "byte_offset": int(attempt_count * ATTEMPT_DTYPE.itemsize),
            "prefix_sha256": ledger_prefix_sha256,
        },
        "training_log": {
            "byte_offset": int(training_log_offset),
            "prefix_sha256": training_log_prefix_sha256,
        },
        "candidate_snapshot": {
            "name": candidate_snapshot_name,
            "attempt_count": int(candidate_snapshot_attempt),
            "elapsed_seconds": float(candidate_snapshot_elapsed_seconds),
            "sha256": candidate_snapshot_sha256,
        },
        "elapsed_seconds": float(elapsed_seconds),
        "resume_count": int(resume_count),
    }


def save_training_checkpoint(checkpoint: dict[str, Any], path: Path) -> None:
    atomic_save(checkpoint, path)


def load_training_checkpoint(
    path: Path,
    *,
    expected_run_id: str,
    expected_run_identity: dict[str, Any],
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != STAGE_D_CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError("Stage D training checkpoint schema mismatch")
    if checkpoint.get("run_id") != expected_run_id:
        raise RuntimeError("Stage D training checkpoint run id mismatch")
    if checkpoint.get("run_identity") != expected_run_identity:
        raise RuntimeError("Stage D training checkpoint identity mismatch")
    ledger = checkpoint.get("ledger", {})
    attempt_count = int(checkpoint.get("attempt_count", -1))
    if (
        int(checkpoint.get("step", -1)) < 0
        or attempt_count < 0
        or ledger.get("schema_version") != ATTEMPT_LEDGER_SCHEMA_VERSION
        or int(ledger.get("record_size", -1)) != ATTEMPT_DTYPE.itemsize
        or int(ledger.get("record_count", -1)) != attempt_count
        or int(ledger.get("byte_offset", -1))
        != attempt_count * ATTEMPT_DTYPE.itemsize
    ):
        raise RuntimeError("Stage D training checkpoint ledger boundary is invalid")
    snapshot = checkpoint.get("candidate_snapshot", {})
    snapshot_attempt = int(snapshot.get("attempt_count", -1))
    training_log = checkpoint.get("training_log", {})
    if snapshot_attempt < 0 or snapshot_attempt > attempt_count:
        raise RuntimeError("Stage D training checkpoint snapshot boundary is invalid")
    if not isinstance(snapshot.get("sha256"), str) or len(snapshot["sha256"]) != 64:
        raise RuntimeError("Stage D training checkpoint snapshot SHA-256 is invalid")
    if int(training_log.get("byte_offset", -1)) < 0:
        raise RuntimeError("Stage D training checkpoint log boundary is invalid")
    return checkpoint


def restore_training_checkpoint(
    checkpoint: dict[str, Any],
    *,
    model: torch.nn.Module | None,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, Any]:
    saved_model = checkpoint.get("model_state_dict")
    saved_optimizer = checkpoint.get("optimizer_state_dict")
    if (model is None) != (optimizer is None):
        raise ValueError("Stage D model and optimizer restore targets must be paired")
    if model is None:
        if saved_model is not None or saved_optimizer is not None:
            raise RuntimeError("Stage D random run checkpoint unexpectedly contains a model")
    else:
        if saved_model is None or saved_optimizer is None:
            raise RuntimeError("Stage D transformer checkpoint lacks training state")
        model.load_state_dict(saved_model)
        optimizer.load_state_dict(saved_optimizer)
        device = next(model.parameters()).device
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)
    restore_rng_state(checkpoint["rng_state"])
    return {
        "step": int(checkpoint["step"]),
        "attempt_count": int(checkpoint["attempt_count"]),
        "ledger": dict(checkpoint["ledger"]),
        "training_log": dict(checkpoint["training_log"]),
        "candidate_snapshot": dict(checkpoint["candidate_snapshot"]),
        "elapsed_seconds": float(checkpoint["elapsed_seconds"]),
        "resume_count": int(checkpoint["resume_count"]),
    }