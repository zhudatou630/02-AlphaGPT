"""Checkpoint schema helpers for Phase 3b."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Any

import numpy as np
import torch

from alpha_etf.gpt.vocab import FormulaVocab


CHECKPOINT_SCHEMA_VERSION = "phase3b-checkpoint-v1"
CHECKPOINT_SCHEMA_VERSION_V2 = "etf-price-checkpoint-v2"


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    batch_size: int = 64
    train_steps: int = 500
    max_len: int = 16
    min_formula_len: int = 3
    top_n: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 1.0
    entropy_coef: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
        "torch_cuda_random_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def build_checkpoint(
    *,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    model_config: dict[str, Any],
    policy_vocab: dict[str, Any],
    formula_vocab: dict[str, Any],
    scorer_config: dict[str, Any],
    train_config: dict[str, Any],
    best_formulas: list[dict[str, Any]],
    best_reward: float | None,
    run_id: str,
    research_spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    checkpoint = {
        "schema_version": (
            CHECKPOINT_SCHEMA_VERSION_V2 if research_spec is not None else CHECKPOINT_SCHEMA_VERSION
        ),
        "run_id": run_id,
        "step": int(step),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": model_config,
        "policy_vocab": policy_vocab,
        "formula_vocab": formula_vocab,
        "scorer_config": scorer_config,
        "train_config": train_config,
        "best_formulas": best_formulas,
        "best_formula": best_formulas[0] if best_formulas else None,
        "best_reward": best_reward,
        "rng_state": rng_state_dict(),
    }
    if research_spec is not None:
        checkpoint["research_spec"] = research_spec
    return checkpoint


def validate_checkpoint_contract(
    checkpoint: dict[str, Any],
    *,
    vocab: FormulaVocab,
    research_spec: dict[str, Any],
) -> None:
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION_V2:
        raise RuntimeError(
            f"Checkpoint schema mismatch: {checkpoint.get('schema_version')} != {CHECKPOINT_SCHEMA_VERSION_V2}"
        )
    expected_vocab = {"vocab_version": vocab.version, "token_names": vocab.token_names}
    actual_vocab = checkpoint.get("formula_vocab", {})
    actual_vocab = {
        "vocab_version": actual_vocab.get("vocab_version"),
        "token_names": tuple(actual_vocab.get("token_names", ())),
    }
    if actual_vocab != expected_vocab:
        raise RuntimeError(f"Checkpoint formula vocab mismatch: {actual_vocab} != {expected_vocab}")
    if checkpoint.get("research_spec") != research_spec:
        raise RuntimeError("Checkpoint research spec mismatch")