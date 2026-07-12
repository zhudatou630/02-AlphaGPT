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


def assert_checkpoint_state_equal(
    expected: Any, actual: Any, *, label: str = "checkpoint"
) -> None:
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor) or not torch.equal(
            expected.cpu(), actual.cpu()
        ):
            raise RuntimeError(f"V3A {label} state mismatch")
        return
    if isinstance(expected, np.ndarray):
        if not isinstance(actual, np.ndarray) or not np.array_equal(expected, actual):
            raise RuntimeError(f"V3A {label} state mismatch")
        return
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(expected) != set(actual):
            raise RuntimeError(f"V3A {label} state keys mismatch")
        for key in expected:
            assert_checkpoint_state_equal(
                expected[key], actual[key], label=f"{label}.{key}"
            )
        return
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, type(expected)) or len(expected) != len(actual):
            raise RuntimeError(f"V3A {label} state shape mismatch")
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            assert_checkpoint_state_equal(
                left, right, label=f"{label}[{index}]"
            )
        return
    if expected != actual:
        raise RuntimeError(f"V3A {label} state mismatch")


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
    model_state = checkpoint.get("model_state_dict")
    if (
        not isinstance(model_state, dict)
        or not model_state
        or not all(
            isinstance(key, str)
            and isinstance(value, torch.Tensor)
            and value.numel() > 0
            and not value.is_complex()
            and (
                not value.is_floating_point()
                or bool(torch.isfinite(value).all().item())
            )
            for key, value in model_state.items()
        )
    ):
        raise RuntimeError("V3A checkpoint model state is invalid")
    optimizer_state = checkpoint.get("optimizer_state_dict")
    is_stage_d_checkpoint = str(train_config.get("schema_version", "")).startswith(
        "etf-v3a-stage-d-"
    )
    if (
        not isinstance(optimizer_state, dict)
        or not isinstance(optimizer_state.get("state"), dict)
        or not isinstance(optimizer_state.get("param_groups"), list)
        or not optimizer_state["param_groups"]
        or not all(
            isinstance(group, dict)
            and isinstance(group.get("params"), list)
            and bool(group["params"])
            for group in optimizer_state["param_groups"]
        )
    ):
        raise RuntimeError("V3A checkpoint optimizer state is invalid")
    optimizer_parameter_ids = [
        parameter_id
        for group in optimizer_state["param_groups"]
        for parameter_id in group["params"]
    ]
    if (
        not optimizer_parameter_ids
        or len(optimizer_parameter_ids) != len(set(optimizer_parameter_ids))
        or (
            is_stage_d_checkpoint
            and int(checkpoint["attempt_count"]) > 0
            and (
                set(optimizer_state["state"]) != set(optimizer_parameter_ids)
                or len(optimizer_parameter_ids) != len(model_state)
            )
        )
    ):
        raise RuntimeError("V3A checkpoint optimizer state is incomplete")
    if is_stage_d_checkpoint and int(checkpoint["attempt_count"]) > 0:
        for parameter_id, parameter in zip(
            optimizer_parameter_ids, model_state.values(), strict=True
        ):
            payload = optimizer_state["state"][parameter_id]
            required_adamw = {"step", "exp_avg", "exp_avg_sq"}
            if not isinstance(payload, dict) or required_adamw - set(payload):
                raise RuntimeError("V3A checkpoint AdamW state is incomplete")
            step_value = payload["step"]
            exp_avg = payload["exp_avg"]
            exp_avg_sq = payload["exp_avg_sq"]
            if (
                not isinstance(step_value, torch.Tensor)
                or step_value.numel() != 1
                or step_value.is_complex()
                or not bool(torch.isfinite(step_value).all().item())
                or not isinstance(exp_avg, torch.Tensor)
                or not isinstance(exp_avg_sq, torch.Tensor)
                or exp_avg.shape != parameter.shape
                or exp_avg_sq.shape != parameter.shape
                or exp_avg.dtype != parameter.dtype
                or exp_avg_sq.dtype != parameter.dtype
                or exp_avg.is_complex()
                or exp_avg_sq.is_complex()
                or not bool(torch.isfinite(exp_avg).all().item())
                or not bool(torch.isfinite(exp_avg_sq).all().item())
            ):
                raise RuntimeError("V3A checkpoint AdamW tensor state is invalid")
    rng_state = checkpoint.get("rng_state")
    required_rng = {
        "python_random_state",
        "numpy_random_state",
        "torch_random_state",
        "torch_cuda_random_state_all",
    }
    if not isinstance(rng_state, dict) or required_rng - set(rng_state):
        raise RuntimeError("V3A checkpoint RNG state is invalid")
    if (
        not isinstance(rng_state["python_random_state"], dict)
        or not isinstance(rng_state["numpy_random_state"], dict)
        or not isinstance(rng_state["torch_random_state"], torch.Tensor)
        or rng_state["torch_random_state"].numel() == 0
        or not isinstance(rng_state["torch_cuda_random_state_all"], list)
        or not all(
            isinstance(value, torch.Tensor) and value.numel() > 0
            for value in rng_state["torch_cuda_random_state_all"]
        )
    ):
        raise RuntimeError("V3A checkpoint RNG state is invalid")
    python_state = rng_state["python_random_state"]
    numpy_state = rng_state["numpy_random_state"]
    if (
        not isinstance(python_state.get("version"), int)
        or not isinstance(python_state.get("state"), list)
        or not python_state["state"]
        or not isinstance(numpy_state.get("bit_generator"), str)
        or not isinstance(numpy_state.get("state"), list)
        or not numpy_state["state"]
        or not all(
            key in numpy_state
            for key in ("pos", "has_gauss", "cached_gaussian")
        )
    ):
        raise RuntimeError("V3A checkpoint RNG state is incomplete")
    if is_stage_d_checkpoint and not rng_state["torch_cuda_random_state_all"]:
        raise RuntimeError("V3A Stage D checkpoint lacks CUDA RNG state")
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