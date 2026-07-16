"""Archive-aware, right-tail learning state for Stage D mechanism runs."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import torch

from alpha_etf.research_v3a.attempts import (
    ATTEMPT_DTYPE,
    AttemptStatus,
    read_attempt_batches,
)


ARCHIVE_TAIL_OBJECTIVE_VERSION = "etf-v3a-archive-tail-v1"
_REPLAY_VALID_STATUSES = {
    int(AttemptStatus.CANONICAL_DUPLICATE),
    int(AttemptStatus.SELECTION_DUPLICATE),
    int(AttemptStatus.ACCEPTED_UNIQUE),
}


@dataclass(frozen=True)
class ArchiveTailConfig:
    elite_fraction: float = 0.10
    archive_size: int = 50
    model_fraction_numerator: int = 3
    model_fraction_denominator: int = 4

    def __post_init__(self) -> None:
        if not 0.0 < self.elite_fraction <= 1.0:
            raise ValueError("Archive-tail elite fraction must be in (0,1]")
        if self.archive_size < 1:
            raise ValueError("Archive-tail size must be positive")
        if not 0 < self.model_fraction_numerator < self.model_fraction_denominator:
            raise ValueError("Archive-tail model fraction is invalid")

    def model_count(self, batch_count: int) -> int:
        if batch_count < 1 or batch_count % self.model_fraction_denominator:
            raise ValueError("Archive-tail batch count must preserve the fixed lane ratio")
        return batch_count * self.model_fraction_numerator // self.model_fraction_denominator

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "objective": ARCHIVE_TAIL_OBJECTIVE_VERSION,
            "elite_fraction": self.elite_fraction,
            "archive_size": self.archive_size,
            "model_fraction_numerator": self.model_fraction_numerator,
            "model_fraction_denominator": self.model_fraction_denominator,
        }


@dataclass(frozen=True)
class ArchiveEntry:
    canonical_hash: bytes
    reward: float
    token_len: int
    attempt_index: int


@dataclass(frozen=True)
class ArchiveTailLabels:
    elite_weights: np.ndarray
    archive_weights: np.ndarray
    waste_weights: np.ndarray
    combined_weights: np.ndarray
    metrics: dict[str, float | int | None]
    new_canonical_indices: tuple[int, ...]
    new_valid_indices: tuple[int, ...]
    elite_indices: tuple[int, ...]
    archive_improver_indices: tuple[int, ...]


@dataclass(frozen=True)
class ArchiveTailObjective:
    loss: torch.Tensor
    elite_loss: torch.Tensor
    archive_loss: torch.Tensor
    waste_loss: torch.Tensor


def archive_tail_objective(
    *,
    log_prob_sums: torch.Tensor,
    elite_weights: torch.Tensor,
    archive_weights: torch.Tensor,
    waste_weights: torch.Tensor,
) -> ArchiveTailObjective:
    if log_prob_sums.ndim != 1:
        raise ValueError("Archive-tail log-probabilities must be one-dimensional")
    for name, weights in (
        ("elite", elite_weights),
        ("archive", archive_weights),
        ("waste", waste_weights),
    ):
        if weights.shape != log_prob_sums.shape:
            raise ValueError(f"Archive-tail {name} weights differ from log-probabilities")
        if not bool(torch.isfinite(weights).all().item()) or bool((weights < 0).any().item()):
            raise ValueError(f"Archive-tail {name} weights are invalid")
    elite_loss = -(log_prob_sums * elite_weights.detach()).sum()
    archive_loss = -(log_prob_sums * archive_weights.detach()).sum()
    waste_loss = (log_prob_sums * waste_weights.detach()).sum()
    loss = elite_loss + archive_loss + waste_loss
    if not bool(torch.isfinite(loss).item()):
        raise RuntimeError("Archive-tail objective is non-finite")
    return ArchiveTailObjective(
        loss=loss,
        elite_loss=elite_loss,
        archive_loss=archive_loss,
        waste_loss=waste_loss,
    )


class ArchiveTailState:
    """Transformer-only seen set and top-reward archive."""

    def __init__(self, config: ArchiveTailConfig = ArchiveTailConfig()):
        self.config = config
        self.seen: set[bytes] = set()
        self.archive: dict[bytes, ArchiveEntry] = {}
        self.model_attempt_count = 0

    @staticmethod
    def _semantic_valid(record: np.void, *, prepared: bool) -> bool:
        status = int(record["status"])
        valid_status = (
            status == int(AttemptStatus.PENDING_SEMANTIC_VALID)
            if prepared
            else status in _REPLAY_VALID_STATUSES
        )
        return valid_status and np.isfinite(float(record["reward"]))

    @staticmethod
    def _candidate_key(record: np.void) -> tuple[float, int, int]:
        return (
            float(record["reward"]),
            -int(record["token_len"]),
            -int(record["attempt_index"]),
        )

    @staticmethod
    def _archive_key(entry: ArchiveEntry) -> tuple[float, int, bytes]:
        return (-entry.reward, entry.token_len, entry.canonical_hash)

    def apply(self, records: np.ndarray, *, prepared: bool) -> ArchiveTailLabels:
        values = np.asarray(records)
        if values.ndim != 1 or values.dtype != ATTEMPT_DTYPE:
            raise ValueError("Archive-tail records must use the Stage D ledger dtype")
        count = int(values.size)
        elite = np.zeros(count, dtype=np.float32)
        archive = np.zeros(count, dtype=np.float32)
        waste = np.zeros(count, dtype=np.float32)
        groups: dict[bytes, list[int]] = {}
        for index, record in enumerate(values):
            groups.setdefault(record["canonical_hash"].tobytes(), []).append(index)

        history_duplicate_count = 0
        within_batch_duplicate_count = 0
        semantic_invalid_count = 0
        new_canonical: list[int] = []
        new_valid: list[int] = []
        for canonical_hash, indices in groups.items():
            if canonical_hash in self.seen:
                history_duplicate_count += len(indices)
                waste[indices] = 1.0
                continue
            valid_indices = [
                index
                for index in indices
                if self._semantic_valid(values[index], prepared=prepared)
            ]
            representative = (
                max(valid_indices, key=lambda index: self._candidate_key(values[index]))
                if valid_indices
                else min(indices, key=lambda index: int(values[index]["attempt_index"]))
            )
            new_canonical.append(representative)
            duplicates = [index for index in indices if index != representative]
            if duplicates:
                within_batch_duplicate_count += len(duplicates)
                waste[duplicates] = 1.0
            if valid_indices:
                new_valid.append(representative)
            else:
                semantic_invalid_count += 1
                waste[representative] = 1.0

        if new_valid:
            elite_count = max(1, math.ceil(len(new_valid) * self.config.elite_fraction))
            ordered = sorted(
                new_valid,
                key=lambda index: (
                    -float(values[index]["reward"]),
                    int(values[index]["token_len"]),
                    values[index]["canonical_hash"].tobytes(),
                ),
            )
            elite_indices = ordered[:elite_count]
            elite[elite_indices] = np.float32(1.0 / elite_count)
        else:
            elite_indices = []

        archive_floor: float | None = None
        archive_indices: list[int] = []
        if len(self.archive) == self.config.archive_size:
            archive_floor = max(self.archive.values(), key=self._archive_key).reward
            archive_indices = [
                index
                for index in new_valid
                if float(values[index]["reward"]) > archive_floor
            ]
            if archive_indices:
                deltas = np.asarray(
                    [float(values[index]["reward"]) - archive_floor for index in archive_indices],
                    dtype=np.float64,
                )
                deltas /= deltas.sum()
                archive[archive_indices] = deltas.astype(np.float32)

        if waste.any():
            waste /= waste.sum(dtype=np.float64)
        combined = elite + archive - waste

        for canonical_hash in groups:
            self.seen.add(canonical_hash)
        for index in new_valid:
            record = values[index]
            canonical_hash = record["canonical_hash"].tobytes()
            self.archive[canonical_hash] = ArchiveEntry(
                canonical_hash=canonical_hash,
                reward=float(record["reward"]),
                token_len=int(record["token_len"]),
                attempt_index=int(record["attempt_index"]),
            )
        if len(self.archive) > self.config.archive_size:
            retained = sorted(self.archive.values(), key=self._archive_key)[
                : self.config.archive_size
            ]
            self.archive = {entry.canonical_hash: entry for entry in retained}
        self.model_attempt_count += count

        new_rewards = np.asarray(
            [float(values[index]["reward"]) for index in new_valid], dtype=np.float64
        )
        archive_rewards = np.asarray(
            [entry.reward for entry in self.archive.values()], dtype=np.float64
        )
        metrics: dict[str, float | int | None] = {
            "model_attempt_count": count,
            "history_duplicate_count": history_duplicate_count,
            "within_batch_duplicate_count": within_batch_duplicate_count,
            "semantic_invalid_count": semantic_invalid_count,
            "new_valid_count": len(new_valid),
            "elite_count": len(elite_indices),
            "archive_improver_count": len(archive_indices),
            "new_valid_reward_mean": float(new_rewards.mean()) if new_rewards.size else None,
            "new_valid_reward_q90": (
                float(np.quantile(new_rewards, 0.90)) if new_rewards.size else None
            ),
            "new_valid_top10_mean": (
                float(np.mean([float(values[index]["reward"]) for index in elite_indices]))
                if elite_indices
                else None
            ),
            "archive_floor_before": archive_floor,
            "archive_count": len(self.archive),
            "archive_mean_reward": (
                float(archive_rewards.mean()) if archive_rewards.size else None
            ),
            "archive_floor_after": (
                float(archive_rewards.min())
                if archive_rewards.size == self.config.archive_size
                else None
            ),
        }
        return ArchiveTailLabels(
            elite_weights=elite,
            archive_weights=archive,
            waste_weights=waste,
            combined_weights=combined,
            metrics=metrics,
            new_canonical_indices=tuple(new_canonical),
            new_valid_indices=tuple(new_valid),
            elite_indices=tuple(elite_indices),
            archive_improver_indices=tuple(archive_indices),
        )

    @classmethod
    def restore_from_ledger(
        cls,
        *,
        ledger_path: Path,
        stop: int,
        batch_size: int,
        config: ArchiveTailConfig = ArchiveTailConfig(),
    ) -> "ArchiveTailState":
        if stop < 0 or stop % config.model_fraction_denominator:
            raise ValueError("Archive-tail restore boundary is invalid")
        state = cls(config)
        for start in range(0, stop, batch_size):
            batch_stop = min(start + batch_size, stop)
            batch_count = batch_stop - start
            model_count = config.model_count(batch_count)
            chunks = list(
                read_attempt_batches(
                    ledger_path,
                    start=start,
                    stop=start + model_count,
                    batch_size=model_count,
                )
            )
            model_records = np.concatenate(chunks) if chunks else np.empty(0, dtype=ATTEMPT_DTYPE)
            labels = state.apply(model_records, prepared=False)
            if not np.allclose(
                model_records["training_reward"],
                labels.combined_weights,
                rtol=0.0,
                atol=1e-7,
            ):
                raise RuntimeError("Archive-tail ledger labels differ during restore")
        return state