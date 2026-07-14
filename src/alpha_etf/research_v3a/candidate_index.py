"""Compact, replayable candidate index for Stage D attempts."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from alpha_etf.research_v3a.attempts import (
    ATTEMPT_DTYPE,
    HASH_BYTES,
    TOKEN_PAD,
    AttemptStatus,
    PreparedAttemptBatch,
    read_attempt_batches,
)
from alpha_etf.research_v3a.candidates import CandidateConfig
from alpha_etf.research_v3a.language import FORMULA_VOCAB, MAX_FORMULA_TOKENS


CANDIDATE_INDEX_SCHEMA_VERSION = "etf-v3a-candidate-index-v2"
_SEMANTIC_STATUSES = {
    int(AttemptStatus.CANONICAL_DUPLICATE),
    int(AttemptStatus.SELECTION_DUPLICATE),
    int(AttemptStatus.ACCEPTED_UNIQUE),
}


class CompactCandidateIndex:
    """Keep duplicate counts and best formula payloads without expression objects."""

    def __init__(
        self,
        *,
        candidate_config: CandidateConfig = CandidateConfig(),
        initial_best_reward: float = -5.0,
        initial_capacity: int = 1024,
    ):
        if initial_capacity < 1:
            raise ValueError("Stage D candidate index capacity must be positive")
        self.candidate_config = candidate_config
        self.initial_best_reward = float(initial_best_reward)
        self.attempt_count = 0
        self.canonical_size = 0
        self.selection_size = 0
        self._canonical_lookup: dict[bytes, int] = {}
        self._selection_lookup: dict[bytes, int] = {}
        self.status_counts = np.zeros(len(AttemptStatus), dtype=np.uint64)
        self.length_counts = np.zeros(MAX_FORMULA_TOKENS + 1, dtype=np.uint64)
        self.token_counts = np.zeros(FORMULA_VOCAB.size, dtype=np.uint64)
        self.structural_duplicate_count = 0
        self.selection_duplicate_count = 0
        self.semantic_valid_count = 0
        self.best_semantic_reward = self.initial_best_reward
        self.best_reward_attempt_auc_numerator = 0.0
        self._allocate_canonical(initial_capacity)
        self._allocate_selection(initial_capacity)

    def _allocate_canonical(self, capacity: int) -> None:
        self.canonical_hashes = np.zeros((capacity, HASH_BYTES), dtype=np.uint8)
        self.canonical_counts = np.zeros(capacity, dtype=np.uint64)
        self.canonical_valid_counts = np.zeros(capacity, dtype=np.uint64)
        self.canonical_first_attempts = np.zeros(capacity, dtype=np.uint64)
        self.canonical_first_valid_attempts = np.zeros(capacity, dtype=np.uint64)
        self.canonical_best_rewards = np.full(capacity, np.nan, dtype=np.float32)
        self.canonical_best_attempts = np.zeros(capacity, dtype=np.uint64)
        self.canonical_best_steps = np.zeros(capacity, dtype=np.uint32)
        self.canonical_best_token_lengths = np.zeros(capacity, dtype=np.uint8)
        self.canonical_best_tokens = np.full(
            (capacity, MAX_FORMULA_TOKENS), TOKEN_PAD, dtype=np.uint8
        )
        self.canonical_best_coverage = np.full(capacity, np.nan, dtype=np.float32)
        self.canonical_best_std = np.full(capacity, np.nan, dtype=np.float32)

    def _allocate_selection(self, capacity: int) -> None:
        self.selection_hashes = np.zeros((capacity, HASH_BYTES), dtype=np.uint8)
        self.selection_counts = np.zeros(capacity, dtype=np.uint64)

    @staticmethod
    def _grow(array: np.ndarray, capacity: int, *, fill: int | float = 0) -> np.ndarray:
        shape = (capacity, *array.shape[1:])
        grown = np.full(shape, fill, dtype=array.dtype)
        grown[: array.shape[0]] = array
        return grown

    def _ensure_canonical_capacity(self) -> None:
        if self.canonical_size < self.canonical_hashes.shape[0]:
            return
        capacity = self.canonical_hashes.shape[0] * 2
        self.canonical_hashes = self._grow(self.canonical_hashes, capacity)
        self.canonical_counts = self._grow(self.canonical_counts, capacity)
        self.canonical_valid_counts = self._grow(self.canonical_valid_counts, capacity)
        self.canonical_first_attempts = self._grow(
            self.canonical_first_attempts, capacity
        )
        self.canonical_first_valid_attempts = self._grow(
            self.canonical_first_valid_attempts, capacity
        )
        self.canonical_best_rewards = self._grow(
            self.canonical_best_rewards, capacity, fill=np.nan
        )
        self.canonical_best_attempts = self._grow(
            self.canonical_best_attempts, capacity
        )
        self.canonical_best_steps = self._grow(self.canonical_best_steps, capacity)
        self.canonical_best_token_lengths = self._grow(
            self.canonical_best_token_lengths, capacity
        )
        self.canonical_best_tokens = self._grow(
            self.canonical_best_tokens, capacity, fill=TOKEN_PAD
        )
        self.canonical_best_coverage = self._grow(
            self.canonical_best_coverage, capacity, fill=np.nan
        )
        self.canonical_best_std = self._grow(
            self.canonical_best_std, capacity, fill=np.nan
        )

    def _ensure_selection_capacity(self) -> None:
        if self.selection_size < self.selection_hashes.shape[0]:
            return
        capacity = self.selection_hashes.shape[0] * 2
        self.selection_hashes = self._grow(self.selection_hashes, capacity)
        self.selection_counts = self._grow(self.selection_counts, capacity)

    def commit(self, batch: PreparedAttemptBatch) -> np.ndarray:
        records = batch.records.copy()
        self._apply(records, resolve_status=True)
        return records

    def replay(self, records: np.ndarray) -> None:
        values = np.asarray(records)
        if values.ndim != 1 or values.dtype != ATTEMPT_DTYPE:
            raise ValueError("Stage D candidate replay requires frozen attempt records")
        self._apply(values, resolve_status=False)

    def _apply(self, records: np.ndarray, *, resolve_status: bool) -> None:
        if records.size and int(records["attempt_index"][0]) != self.attempt_count:
            raise RuntimeError("Stage D candidate index attempt order is not contiguous")
        for record in records:
            attempt_index = int(record["attempt_index"])
            if attempt_index != self.attempt_count:
                raise RuntimeError("Stage D candidate index attempt order is not contiguous")
            token_len = int(record["token_len"])
            tokens = record["token_ids"][:token_len]
            self.length_counts[token_len] += 1
            np.add.at(self.token_counts, tokens, 1)

            canonical_key = record["canonical_hash"].tobytes()
            canonical_row = self._canonical_lookup.get(canonical_key)
            canonical_duplicate = canonical_row is not None
            if canonical_row is None:
                self._ensure_canonical_capacity()
                canonical_row = self.canonical_size
                self.canonical_size += 1
                self._canonical_lookup[canonical_key] = canonical_row
                self.canonical_hashes[canonical_row] = record["canonical_hash"]
                self.canonical_first_attempts[canonical_row] = attempt_index
            else:
                self.structural_duplicate_count += 1
            self.canonical_counts[canonical_row] += 1

            status = int(record["status"])
            semantic_valid = (
                status == int(AttemptStatus.PENDING_SEMANTIC_VALID)
                if resolve_status
                else status in _SEMANTIC_STATUSES
            )
            selection_duplicate = False
            if semantic_valid:
                selection_key = record["selection_hash"].tobytes()
                selection_row = self._selection_lookup.get(selection_key)
                selection_duplicate = selection_row is not None
                if selection_row is None:
                    self._ensure_selection_capacity()
                    selection_row = self.selection_size
                    self.selection_size += 1
                    self._selection_lookup[selection_key] = selection_row
                    self.selection_hashes[selection_row] = record["selection_hash"]
                else:
                    self.selection_duplicate_count += 1
                self.selection_counts[selection_row] += 1

                if resolve_status:
                    if canonical_duplicate:
                        status = int(AttemptStatus.CANONICAL_DUPLICATE)
                    elif selection_duplicate:
                        status = int(AttemptStatus.SELECTION_DUPLICATE)
                    else:
                        status = int(AttemptStatus.ACCEPTED_UNIQUE)
                    record["status"] = np.uint8(status)

                self.semantic_valid_count += 1
                self.canonical_valid_counts[canonical_row] += 1
                if self.canonical_valid_counts[canonical_row] == 1:
                    self.canonical_first_valid_attempts[canonical_row] = attempt_index
                if self._is_better(record, canonical_row):
                    self._set_best(record, canonical_row)
                self.best_semantic_reward = max(
                    self.best_semantic_reward, float(record["reward"])
                )

            if status == int(AttemptStatus.PENDING_SEMANTIC_VALID):
                raise RuntimeError("Stage D candidate index left an attempt status unresolved")
            self.status_counts[status] += 1
            self.best_reward_attempt_auc_numerator += self.best_semantic_reward
            self.attempt_count += 1

    def _is_better(self, record: np.void, row: int) -> bool:
        previous = float(self.canonical_best_rewards[row])
        if not np.isfinite(previous):
            return True
        reward = float(record["reward"])
        if reward != previous:
            return reward > previous
        return int(record["token_len"]) < int(self.canonical_best_token_lengths[row])

    def _set_best(self, record: np.void, row: int) -> None:
        self.canonical_best_rewards[row] = record["reward"]
        self.canonical_best_attempts[row] = record["attempt_index"]
        self.canonical_best_steps[row] = record["training_step"]
        self.canonical_best_token_lengths[row] = record["token_len"]
        self.canonical_best_tokens[row] = record["token_ids"]
        self.canonical_best_coverage[row] = record["coverage"]
        self.canonical_best_std[row] = record["finite_std"]

    def retained_rows(self) -> list[int]:
        rows: list[int] = []
        valid_rows = np.flatnonzero(self.canonical_valid_counts[: self.canonical_size] > 0)
        valid_lengths = self.canonical_best_token_lengths[valid_rows]
        for bucket_index in range(3):
            if bucket_index == 0:
                in_bucket = valid_lengths <= 5
            elif bucket_index == 1:
                in_bucket = (valid_lengths > 5) & (valid_lengths <= 10)
            else:
                in_bucket = valid_lengths > 10
            bucket_rows = valid_rows[in_bucket]
            if not bucket_rows.size:
                continue
            hash_keys = tuple(
                self.canonical_hashes[bucket_rows, index]
                for index in reversed(range(HASH_BYTES))
            )
            order = np.lexsort(
                (
                    *hash_keys,
                    self.canonical_best_token_lengths[bucket_rows],
                    -self.canonical_best_rewards[bucket_rows],
                )
            )
            rows.extend(
                int(value)
                for value in bucket_rows[order[: self.candidate_config.heap_per_bucket]]
            )
        return rows

    def save_snapshot(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        payload = {
            "schema_version": np.asarray(CANDIDATE_INDEX_SCHEMA_VERSION),
            "attempt_count": np.asarray(self.attempt_count, dtype=np.uint64),
            "initial_best_reward": np.asarray(self.initial_best_reward, dtype=np.float64),
            "candidate_config": np.asarray(
                json.dumps(
                    self.candidate_config.to_dict(),
                    sort_keys=True,
                    separators=(",", ":"),
                )
            ),
            "status_counts": self.status_counts,
            "length_counts": self.length_counts,
            "token_counts": self.token_counts,
            "structural_duplicate_count": np.asarray(
                self.structural_duplicate_count, dtype=np.uint64
            ),
            "selection_duplicate_count": np.asarray(
                self.selection_duplicate_count, dtype=np.uint64
            ),
            "semantic_valid_count": np.asarray(self.semantic_valid_count, dtype=np.uint64),
            "best_semantic_reward": np.asarray(
                self.best_semantic_reward, dtype=np.float64
            ),
            "best_reward_attempt_auc_numerator": np.asarray(
                self.best_reward_attempt_auc_numerator, dtype=np.float64
            ),
            "canonical_hashes": self.canonical_hashes[: self.canonical_size],
            "canonical_counts": self.canonical_counts[: self.canonical_size],
            "canonical_valid_counts": self.canonical_valid_counts[: self.canonical_size],
            "canonical_first_attempts": self.canonical_first_attempts[
                : self.canonical_size
            ],
            "canonical_first_valid_attempts": self.canonical_first_valid_attempts[
                : self.canonical_size
            ],
            "canonical_best_rewards": self.canonical_best_rewards[: self.canonical_size],
            "canonical_best_attempts": self.canonical_best_attempts[
                : self.canonical_size
            ],
            "canonical_best_steps": self.canonical_best_steps[: self.canonical_size],
            "canonical_best_token_lengths": self.canonical_best_token_lengths[
                : self.canonical_size
            ],
            "canonical_best_tokens": self.canonical_best_tokens[: self.canonical_size],
            "canonical_best_coverage": self.canonical_best_coverage[
                : self.canonical_size
            ],
            "canonical_best_std": self.canonical_best_std[: self.canonical_size],
            "selection_hashes": self.selection_hashes[: self.selection_size],
            "selection_counts": self.selection_counts[: self.selection_size],
        }
        try:
            with temporary.open("xb") as handle:
                np.savez(handle, **payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def load_snapshot(cls, path: Path) -> "CompactCandidateIndex":
        with np.load(path, allow_pickle=False) as payload:
            if str(payload["schema_version"].item()) != CANDIDATE_INDEX_SCHEMA_VERSION:
                raise RuntimeError("Stage D candidate snapshot schema mismatch")
            config = CandidateConfig(
                **{
                    key: tuple(value) if key in {"first_pass_quotas", "final_quotas"} else value
                    for key, value in json.loads(payload["candidate_config"].item()).items()
                    if key not in {"version", "canonicalizer_version"}
                }
            )
            canonical_size = int(payload["canonical_hashes"].shape[0])
            selection_size = int(payload["selection_hashes"].shape[0])
            index = cls(
                candidate_config=config,
                initial_best_reward=float(payload["initial_best_reward"].item()),
                initial_capacity=max(1, canonical_size, selection_size),
            )
            index.attempt_count = int(payload["attempt_count"].item())
            index.status_counts[:] = payload["status_counts"]
            index.length_counts[:] = payload["length_counts"]
            index.token_counts[:] = payload["token_counts"]
            index.structural_duplicate_count = int(
                payload["structural_duplicate_count"].item()
            )
            index.selection_duplicate_count = int(
                payload["selection_duplicate_count"].item()
            )
            index.semantic_valid_count = int(payload["semantic_valid_count"].item())
            index.best_semantic_reward = float(payload["best_semantic_reward"].item())
            index.best_reward_attempt_auc_numerator = float(
                payload["best_reward_attempt_auc_numerator"].item()
            )
            index.canonical_size = canonical_size
            index.selection_size = selection_size
            for name in (
                "canonical_hashes",
                "canonical_counts",
                "canonical_valid_counts",
                "canonical_first_attempts",
                "canonical_first_valid_attempts",
                "canonical_best_rewards",
                "canonical_best_attempts",
                "canonical_best_steps",
                "canonical_best_token_lengths",
                "canonical_best_tokens",
                "canonical_best_coverage",
                "canonical_best_std",
            ):
                target = getattr(index, name)
                target[:canonical_size] = payload[name]
            index.selection_hashes[:selection_size] = payload["selection_hashes"]
            index.selection_counts[:selection_size] = payload["selection_counts"]
        index._canonical_lookup = {
            row.tobytes(): offset
            for offset, row in enumerate(index.canonical_hashes[:canonical_size])
        }
        index._selection_lookup = {
            row.tobytes(): offset
            for offset, row in enumerate(index.selection_hashes[:selection_size])
        }
        return index

    @classmethod
    def restore(
        cls,
        *,
        snapshot_path: Path | None,
        ledger_path: Path,
        stop: int,
        expected_snapshot_attempt: int | None = None,
        candidate_config: CandidateConfig = CandidateConfig(),
        initial_best_reward: float = -5.0,
    ) -> "CompactCandidateIndex":
        if snapshot_path is None:
            index = cls(
                candidate_config=candidate_config,
                initial_best_reward=initial_best_reward,
            )
        else:
            index = cls.load_snapshot(snapshot_path)
            if (
                expected_snapshot_attempt is not None
                and index.attempt_count != expected_snapshot_attempt
            ):
                raise RuntimeError("Stage D candidate snapshot attempt boundary mismatch")
        for records in read_attempt_batches(
            ledger_path, start=index.attempt_count, stop=stop
        ):
            index.replay(records)
        return index