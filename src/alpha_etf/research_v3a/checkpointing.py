"""Checkpoint schema and compatibility gates for V3A."""

from __future__ import annotations

import copy
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from alpha_etf.research_v3a.language import FORMULA_VOCAB
from alpha_etf.research_v3a.sampling import PolicyVocab
from alpha_etf.research_v3a.spec import validate_research_spec


CHECKPOINT_SCHEMA_VERSION = "etf-v3a-checkpoint-v1"
CANDIDATE_STATE_SCHEMA_VERSION = "etf-v3a-candidate-state-v1"


def empty_candidate_state(*, attempt_count: int = 0) -> dict[str, Any]:
    return {
        "schema_version": CANDIDATE_STATE_SCHEMA_VERSION,
        "attempt_count": int(attempt_count),
        "canonical_state": {},
        "bucket_state": [[], [], []],
        "funnel_artifact": None,
    }


def validate_candidate_state(
    state: dict[str, Any], *, attempt_count: int, research_spec: dict[str, Any]
) -> None:
    if state.get("schema_version") != CANDIDATE_STATE_SCHEMA_VERSION:
        raise RuntimeError("V3A checkpoint candidate-state schema mismatch")
    if int(state.get("attempt_count", -1)) != int(attempt_count):
        raise RuntimeError("V3A checkpoint candidate-state attempt count mismatch")
    canonical = state.get("canonical_state")
    if not isinstance(canonical, dict):
        raise RuntimeError("V3A checkpoint canonical state is invalid")
    buckets = state.get("bucket_state")
    if not isinstance(buckets, list) or len(buckets) != 3 or not all(
        isinstance(bucket, list) for bucket in buckets
    ):
        raise RuntimeError("V3A checkpoint bucket state is invalid")
    if state.get("funnel_artifact") is not None and not isinstance(
        state.get("funnel_artifact"), dict
    ):
        raise RuntimeError("V3A checkpoint funnel artifact is invalid")
    from alpha_etf.research_v3a.candidates import (
        CandidateConfig,
        candidate_record_from_dict,
    )

    canonical_attempts = 0
    token_lengths: dict[str, int] = {}
    for formula_hash, entry in canonical.items():
        if not isinstance(entry, dict):
            raise RuntimeError("V3A checkpoint canonical ledger entry is invalid")
        required = {
            "formula_hash",
            "first_attempt_index",
            "attempt_count",
            "best_attempt_index",
            "best_reward",
            "best_record",
        }
        if required - set(entry) or str(entry.get("formula_hash")) != str(formula_hash):
            raise RuntimeError("V3A checkpoint canonical ledger fields mismatch")
        record = candidate_record_from_dict(entry["best_record"])
        if record.formula_hash != formula_hash:
            raise RuntimeError("V3A checkpoint canonical ledger hash mismatch")
        entry_attempts = int(entry["attempt_count"])
        first_attempt = int(entry["first_attempt_index"])
        best_attempt = int(entry["best_attempt_index"])
        best_reward = float(entry["best_reward"])
        if (
            entry_attempts < 1
            or not 0 <= first_attempt <= best_attempt < attempt_count
            or not np.isfinite(best_reward)
            or best_reward != record.reward
        ):
            raise RuntimeError("V3A checkpoint canonical ledger values are invalid")
        if (
            record.attempt_count != entry_attempts
            or record.first_attempt_index != first_attempt
            or record.best_attempt_index != best_attempt
        ):
            raise RuntimeError("V3A checkpoint candidate record ledger mismatch")
        canonical_attempts += entry_attempts
        token_lengths[formula_hash] = record.token_len
    if canonical_attempts > attempt_count:
        raise RuntimeError("V3A checkpoint canonical attempts exceed total attempts")

    seen_bucket_hashes: set[str] = set()
    for bucket_index, bucket in enumerate(buckets):
        for formula_hash in bucket:
            if not isinstance(formula_hash, str) or formula_hash not in canonical:
                raise RuntimeError("V3A checkpoint bucket references an unknown formula")
            if formula_hash in seen_bucket_hashes:
                raise RuntimeError("V3A checkpoint formula appears in multiple buckets")
            token_len = token_lengths[formula_hash]
            expected_bucket = 0 if token_len <= 5 else 1 if token_len <= 10 else 2
            if expected_bucket != bucket_index:
                raise RuntimeError("V3A checkpoint formula is in the wrong length bucket")
            seen_bucket_hashes.add(formula_hash)

    funnel = state.get("funnel_artifact")
    if funnel is not None:
        from alpha_etf.research_v3a.artifacts import validate_training_funnel_artifact

        validate_training_funnel_artifact(
            funnel,
            research_spec=research_spec,
            candidate_config=CandidateConfig(),
        )


def _serialize_random_state(state: tuple[Any, ...]) -> dict[str, Any]:
    return {"version": int(state[0]), "state": list(state[1]), "gauss": state[2]}


def _restore_random_instance(rng: random.Random, state: dict[str, Any]) -> None:
    rng.setstate(
        (
            int(state["version"]),
            tuple(int(item) for item in state["state"]),
            state.get("gauss"),
        )
    )


def rng_state_dict() -> dict[str, Any]:
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    return {
        "python_random_state": {
            "version": int(python_state[0]),
            "state": list(python_state[1]),
            "gauss": python_state[2],
        },
        "numpy_random_state": {
            "bit_generator": str(numpy_state[0]),
            "state": numpy_state[1].tolist(),
            "pos": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_random_state": torch.get_rng_state(),
        "torch_cuda_random_state_all": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else [],
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    python_state = state.get("python_random_state")
    if python_state:
        random.setstate(
            (
                int(python_state["version"]),
                tuple(int(item) for item in python_state["state"]),
                python_state.get("gauss"),
            )
        )
    numpy_state = state.get("numpy_random_state")
    if numpy_state:
        np.random.set_state(
            (
                str(numpy_state["bit_generator"]),
                np.asarray(numpy_state["state"], dtype=np.uint32),
                int(numpy_state["pos"]),
                int(numpy_state["has_gauss"]),
                float(numpy_state["cached_gaussian"]),
            )
        )
    torch_state = state.get("torch_random_state")
    if torch_state is not None:
        torch.set_rng_state(torch_state.detach().cpu())
    cuda_states = state.get("torch_cuda_random_state_all") or []
    if cuda_states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.detach().cpu() for item in cuda_states])


def build_checkpoint(
    *,
    run_id: str,
    step: int,
    attempt_count: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    model_config: dict[str, Any],
    train_config: dict[str, Any],
    scorer_config: dict[str, Any],
    candidate_state: dict[str, Any],
    research_spec: dict[str, Any],
    random_generators: dict[str, random.Random] | None = None,
    torch_generators: dict[str, torch.Generator] | None = None,
) -> dict[str, Any]:
    validate_research_spec(research_spec)
    validate_candidate_state(
        candidate_state, attempt_count=attempt_count, research_spec=research_spec
    )
    policy_vocab = PolicyVocab()
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "run_id": run_id,
        "step": int(step),
        "attempt_count": int(attempt_count),
        "model_state_dict": copy.deepcopy(model.state_dict()),
        "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
        "model_config": model_config,
        "train_config": train_config,
        "scorer_config": scorer_config,
        "formula_vocab": {
            "version": FORMULA_VOCAB.version,
            "token_names": list(FORMULA_VOCAB.token_names),
        },
        "policy_vocab": {
            "token_names": list(policy_vocab.token_names),
            "special_tokens": policy_vocab.special_tokens,
        },
        "candidate_state": candidate_state,
        "research_spec_id": research_spec["research_spec_id"],
        "research_spec": research_spec,
        "rng_state": rng_state_dict(),
        "generator_states": {
            "python": {
                name: _serialize_random_state(generator.getstate())
                for name, generator in sorted((random_generators or {}).items())
            },
            "torch": {
                name: generator.get_state()
                for name, generator in sorted((torch_generators or {}).items())
            },
        },
    }


def validate_checkpoint(
    checkpoint: dict[str, Any],
    *,
    research_spec: dict[str, Any],
    run_id: str,
    model_config: dict[str, Any],
    train_config: dict[str, Any],
    scorer_config: dict[str, Any],
) -> None:
    validate_research_spec(research_spec)
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError(
            f"V3A checkpoint schema mismatch: {checkpoint.get('schema_version')} != "
            f"{CHECKPOINT_SCHEMA_VERSION}"
        )
    if checkpoint.get("research_spec_id") != research_spec.get("research_spec_id"):
        raise RuntimeError("V3A checkpoint ResearchSpec id mismatch")
    if checkpoint.get("research_spec") != research_spec:
        raise RuntimeError("V3A checkpoint ResearchSpec mismatch")
    expected_formula_vocab = {
        "version": FORMULA_VOCAB.version,
        "token_names": list(FORMULA_VOCAB.token_names),
    }
    if checkpoint.get("formula_vocab") != expected_formula_vocab:
        raise RuntimeError("V3A checkpoint formula vocab mismatch")
    policy_vocab = PolicyVocab()
    expected_policy_vocab = {
        "token_names": list(policy_vocab.token_names),
        "special_tokens": policy_vocab.special_tokens,
    }
    if checkpoint.get("policy_vocab") != expected_policy_vocab:
        raise RuntimeError("V3A checkpoint policy vocab mismatch")
    if checkpoint.get("run_id") != run_id:
        raise RuntimeError("V3A checkpoint run id mismatch")
    if checkpoint.get("model_config") != model_config:
        raise RuntimeError("V3A checkpoint model config mismatch")
    if checkpoint.get("scorer_config") != scorer_config:
        raise RuntimeError("V3A checkpoint scorer config mismatch")
    if checkpoint.get("train_config") != train_config:
        raise RuntimeError("V3A checkpoint train config mismatch")
    if int(checkpoint.get("step", -1)) < 0 or int(checkpoint.get("attempt_count", -1)) < 0:
        raise RuntimeError("V3A checkpoint counters are invalid")
    validate_candidate_state(
        checkpoint.get("candidate_state", {}),
        attempt_count=int(checkpoint["attempt_count"]),
        research_spec=research_spec,
    )
    generator_states = checkpoint.get("generator_states")
    if not isinstance(generator_states, dict) or not isinstance(
        generator_states.get("python"), dict
    ) or not isinstance(generator_states.get("torch"), dict):
        raise RuntimeError("V3A checkpoint generator states are invalid")


def atomic_save(checkpoint: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(checkpoint, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(
    path: Path,
    *,
    research_spec: dict[str, Any],
    run_id: str,
    model_config: dict[str, Any],
    train_config: dict[str, Any],
    scorer_config: dict[str, Any],
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    validate_checkpoint(
        checkpoint,
        research_spec=research_spec,
        run_id=run_id,
        model_config=model_config,
        train_config=train_config,
        scorer_config=scorer_config,
    )
    return checkpoint


def restore_training_state(
    checkpoint: dict[str, Any],
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    research_spec: dict[str, Any],
    run_id: str,
    model_config: dict[str, Any],
    train_config: dict[str, Any],
    scorer_config: dict[str, Any],
    random_generators: dict[str, random.Random] | None = None,
    torch_generators: dict[str, torch.Generator] | None = None,
) -> dict[str, Any]:
    validate_checkpoint(
        checkpoint,
        research_spec=research_spec,
        run_id=run_id,
        model_config=model_config,
        train_config=train_config,
        scorer_config=scorer_config,
    )
    saved_generators = checkpoint["generator_states"]
    expected_python = set(saved_generators["python"])
    actual_python = set((random_generators or {}))
    expected_torch = set(saved_generators["torch"])
    actual_torch = set((torch_generators or {}))
    if expected_python != actual_python or expected_torch != actual_torch:
        raise RuntimeError("V3A checkpoint generator inventory mismatch")

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    restore_rng_state(checkpoint["rng_state"])
    for name, generator in (random_generators or {}).items():
        _restore_random_instance(generator, saved_generators["python"][name])
    for name, generator in (torch_generators or {}).items():
        generator.set_state(saved_generators["torch"][name].detach().cpu())
    return {
        "step": int(checkpoint["step"]),
        "attempt_count": int(checkpoint["attempt_count"]),
        "candidate_state": checkpoint["candidate_state"],
    }