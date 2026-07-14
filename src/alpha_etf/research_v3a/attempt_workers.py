"""CPU process-pool preparation for Stage D attempt batches."""

from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
import multiprocessing

import numpy as np

from alpha_etf.research_v3a.attempts import (
    PreparedAttemptBatch,
    prepare_attempt_batch,
)


@dataclass(frozen=True)
class SubmittedAttemptBatch:
    futures: tuple[Future[PreparedAttemptBatch], ...]

    def result(self) -> PreparedAttemptBatch:
        records = np.concatenate([future.result().records for future in self.futures])
        return PreparedAttemptBatch(records)


class AttemptBatchPreparer:
    """Prepare one complete batch, optionally using isolated CPU workers."""

    def __init__(self, worker_count: int):
        if worker_count < 1:
            raise ValueError("Stage D CPU worker count must be positive")
        self.worker_count = int(worker_count)
        self._executor = ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=multiprocessing.get_context("spawn"),
        )

    def prepare(
        self,
        *,
        attempt_start: int,
        training_step: int,
        token_ids: np.ndarray,
        token_lengths: np.ndarray,
        vm_valid: np.ndarray,
        score_valid: np.ndarray,
        quality_valid: np.ndarray,
        variation_valid: np.ndarray,
        rewards: np.ndarray,
        training_rewards: np.ndarray,
        coverage: np.ndarray,
        finite_std: np.ndarray,
        selected_indices: np.ndarray,
        top_k: np.ndarray,
    ) -> PreparedAttemptBatch:
        batch_size = int(np.asarray(token_ids).shape[0])
        arguments = {
            "training_step": training_step,
            "token_ids": token_ids,
            "token_lengths": token_lengths,
            "vm_valid": vm_valid,
            "score_valid": score_valid,
            "quality_valid": quality_valid,
            "variation_valid": variation_valid,
            "rewards": rewards,
            "training_rewards": training_rewards,
            "coverage": coverage,
            "finite_std": finite_std,
            "selected_indices": selected_indices,
            "top_k": top_k,
        }
        if self.worker_count == 1 or batch_size == 1:
            return prepare_attempt_batch(attempt_start=attempt_start, **arguments)

        return self._submit(
            attempt_start=attempt_start,
            batch_size=batch_size,
            arguments=arguments,
        ).result()

    def submit(
        self,
        *,
        attempt_start: int,
        training_step: int,
        token_ids: np.ndarray,
        token_lengths: np.ndarray,
        vm_valid: np.ndarray,
        score_valid: np.ndarray,
        quality_valid: np.ndarray,
        variation_valid: np.ndarray,
        rewards: np.ndarray,
        training_rewards: np.ndarray,
        coverage: np.ndarray,
        finite_std: np.ndarray,
        selected_indices: np.ndarray,
        top_k: np.ndarray,
    ) -> SubmittedAttemptBatch:
        batch_size = int(np.asarray(token_ids).shape[0])
        return self._submit(
            attempt_start=attempt_start,
            batch_size=batch_size,
            arguments={
                "training_step": training_step,
                "token_ids": token_ids,
                "token_lengths": token_lengths,
                "vm_valid": vm_valid,
                "score_valid": score_valid,
                "quality_valid": quality_valid,
                "variation_valid": variation_valid,
                "rewards": rewards,
                "training_rewards": training_rewards,
                "coverage": coverage,
                "finite_std": finite_std,
                "selected_indices": selected_indices,
                "top_k": top_k,
            },
        )

    def _submit(
        self,
        *,
        attempt_start: int,
        batch_size: int,
        arguments: dict[str, object],
    ) -> SubmittedAttemptBatch:

        chunk_size = (batch_size + self.worker_count - 1) // self.worker_count
        batch_keys = {
            "token_ids",
            "token_lengths",
            "vm_valid",
            "score_valid",
            "quality_valid",
            "variation_valid",
            "rewards",
            "training_rewards",
            "coverage",
            "finite_std",
            "selected_indices",
        }
        futures: list[Future[PreparedAttemptBatch]] = []
        for start in range(0, batch_size, chunk_size):
            stop = min(start + chunk_size, batch_size)
            chunk = {
                key: value[start:stop] if key in batch_keys else value
                for key, value in arguments.items()
            }
            futures.append(
                self._executor.submit(
                    prepare_attempt_batch,
                    attempt_start=attempt_start + start,
                    **chunk,
                )
            )
        return SubmittedAttemptBatch(tuple(futures))

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)

    def __enter__(self) -> "AttemptBatchPreparer":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()