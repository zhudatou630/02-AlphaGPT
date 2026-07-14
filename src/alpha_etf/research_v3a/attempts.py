"""Compact Stage D attempt records and append-only binary ledger."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import hashlib
import os
from pathlib import Path
from typing import Iterator

import numpy as np

from alpha_etf.research_v3a.candidates import canonicalize_expression, expression_hash
from alpha_etf.research_v3a.language import MAX_FORMULA_TOKENS, compile_formula


ATTEMPT_LEDGER_SCHEMA_VERSION = "etf-v3a-attempt-ledger-v2"
HASH_BYTES = 32
TOKEN_PAD = 255


class AttemptStatus(IntEnum):
    PENDING_SEMANTIC_VALID = 0
    GRAMMAR_INVALID = 1
    VM_INVALID = 2
    INSUFFICIENT_DAILY_SIGNAL = 3
    LOW_COVERAGE = 4
    CONSTANT_SIGNAL = 5
    CANONICAL_DUPLICATE = 6
    SELECTION_DUPLICATE = 7
    ACCEPTED_UNIQUE = 8


ATTEMPT_DTYPE = np.dtype(
    [
        ("attempt_index", "<u8"),
        ("training_step", "<u4"),
        ("status", "u1"),
        ("token_len", "u1"),
        ("token_ids", "u1", (MAX_FORMULA_TOKENS,)),
        ("canonical_hash", "u1", (HASH_BYTES,)),
        ("selection_hash", "u1", (HASH_BYTES,)),
        ("reward", "<f4"),
        ("training_reward", "<f4"),
        ("coverage", "<f4"),
        ("finite_std", "<f4"),
    ],
    align=False,
)

if ATTEMPT_DTYPE.itemsize != 109:  # pragma: no cover - frozen schema assertion.
    raise AssertionError(f"Unexpected Stage D attempt record size: {ATTEMPT_DTYPE.itemsize}")


@dataclass(frozen=True)
class PreparedAttemptBatch:
    records: np.ndarray

    def __post_init__(self) -> None:
        if self.records.ndim != 1 or self.records.dtype != ATTEMPT_DTYPE:
            raise ValueError("Prepared Stage D attempts must use the frozen ledger dtype")

    @property
    def count(self) -> int:
        return int(self.records.shape[0])


def _array(value: np.ndarray, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError(f"Stage D {name} shape differs: {array.shape} != {shape}")
    return array


def prepare_attempt_batch(
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
    """Build one ordered CPU batch without consulting global candidate state."""

    tokens = np.asarray(token_ids)
    if tokens.ndim != 2 or tokens.shape[1] != MAX_FORMULA_TOKENS:
        raise ValueError("Stage D token_ids must be [batch,max_formula_tokens]")
    batch_size = int(tokens.shape[0])
    lengths = _array(token_lengths, shape=(batch_size,), name="token_lengths")
    vm_ok = _array(vm_valid, shape=(batch_size,), name="vm_valid").astype(bool, copy=False)
    score_ok = _array(score_valid, shape=(batch_size,), name="score_valid").astype(
        bool, copy=False
    )
    quality_ok = _array(
        quality_valid, shape=(batch_size,), name="quality_valid"
    ).astype(bool, copy=False)
    variation_ok = _array(
        variation_valid, shape=(batch_size,), name="variation_valid"
    ).astype(bool, copy=False)
    reward_values = _array(rewards, shape=(batch_size,), name="rewards")
    training_values = _array(
        training_rewards, shape=(batch_size,), name="training_rewards"
    )
    coverage_values = _array(coverage, shape=(batch_size,), name="coverage")
    std_values = _array(finite_std, shape=(batch_size,), name="finite_std")
    selected = np.asarray(selected_indices)
    day_top_k = np.asarray(top_k)
    if selected.ndim != 3 or selected.shape[0] != batch_size:
        raise ValueError("Stage D selected_indices must be [batch,day,position]")
    if day_top_k.shape != (selected.shape[1],):
        raise ValueError("Stage D top_k differs from selected_indices days")
    if np.any(lengths < 1) or np.any(lengths > MAX_FORMULA_TOKENS):
        raise ValueError("Stage D formula lengths are outside the frozen grammar limit")

    records = np.zeros(batch_size, dtype=ATTEMPT_DTYPE)
    records["attempt_index"] = np.arange(
        attempt_start, attempt_start + batch_size, dtype=np.uint64
    )
    records["training_step"] = np.uint32(training_step)
    records["status"] = np.uint8(AttemptStatus.PENDING_SEMANTIC_VALID)
    records["token_len"] = lengths.astype(np.uint8, copy=False)
    records["token_ids"] = np.uint8(TOKEN_PAD)
    records["reward"] = np.float32(np.nan)
    records["training_reward"] = training_values.astype(np.float32, copy=False)
    records["coverage"] = coverage_values.astype(np.float32, copy=False)
    records["finite_std"] = std_values.astype(np.float32, copy=False)

    semantic_valid = vm_ok & score_ok & quality_ok & np.isfinite(reward_values)
    if np.any(vm_ok & score_ok & quality_ok & ~np.isfinite(reward_values)):
        raise RuntimeError("Stage D scorer marked a non-finite reward as otherwise valid")
    records["status"][~vm_ok] = np.uint8(AttemptStatus.VM_INVALID)
    records["status"][vm_ok & ~score_ok] = np.uint8(
        AttemptStatus.INSUFFICIENT_DAILY_SIGNAL
    )
    records["status"][vm_ok & score_ok & ~quality_ok & ~variation_ok] = np.uint8(
        AttemptStatus.CONSTANT_SIGNAL
    )
    records["status"][vm_ok & score_ok & ~quality_ok & variation_ok] = np.uint8(
        AttemptStatus.LOW_COVERAGE
    )
    records["reward"][semantic_valid] = reward_values[semantic_valid].astype(
        np.float32, copy=False
    )

    normalized_selection = selected.astype(np.int16, copy=True)
    positions = np.arange(selected.shape[2], dtype=np.int64)
    normalized_selection = np.where(
        positions[None, None, :] < day_top_k[None, :, None],
        normalized_selection,
        np.int16(-1),
    )

    for row in range(batch_size):
        length = int(lengths[row])
        formula_tokens = [int(value) for value in tokens[row, :length]]
        records["token_ids"][row, :length] = np.asarray(formula_tokens, dtype=np.uint8)
        compiled = compile_formula(formula_tokens)
        canonical = canonicalize_expression(compiled.expression)
        records["canonical_hash"][row] = np.frombuffer(
            bytes.fromhex(expression_hash(canonical)), dtype=np.uint8
        )
        if semantic_valid[row]:
            records["selection_hash"][row] = np.frombuffer(
                hashlib.sha256(normalized_selection[row].tobytes()).digest(),
                dtype=np.uint8,
            )

    return PreparedAttemptBatch(records)


class AttemptLedger:
    """Append fixed records and maintain the digest of the durable prefix."""

    def __init__(
        self, path: Path, *, create: bool, allow_incomplete_tail: bool = False
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if create:
            self._handle = self.path.open("x+b")
        else:
            self._handle = self.path.open("r+b")
        size = self.path.stat().st_size
        if size % ATTEMPT_DTYPE.itemsize and not allow_incomplete_tail:
            self._handle.close()
            raise RuntimeError("Stage D attempt ledger ends inside a binary record")
        self._record_count = size // ATTEMPT_DTYPE.itemsize
        self._handle.seek(0)
        self._digest = hashlib.sha256()
        remaining = self.byte_offset
        while remaining:
            chunk = self._handle.read(min(1024 * 1024, remaining))
            if not chunk:
                self._handle.close()
                raise RuntimeError("Stage D attempt ledger ended inside its complete prefix")
            self._digest.update(chunk)
            remaining -= len(chunk)
        self._handle.seek(0, os.SEEK_END)

    @property
    def record_count(self) -> int:
        return self._record_count

    @property
    def byte_offset(self) -> int:
        return self._record_count * ATTEMPT_DTYPE.itemsize

    @property
    def prefix_sha256(self) -> str:
        return self._digest.hexdigest()

    def append(self, records: np.ndarray) -> None:
        values = np.asarray(records)
        if values.ndim != 1 or values.dtype != ATTEMPT_DTYPE:
            raise ValueError("Stage D ledger append requires the frozen record dtype")
        if values.size and int(values["attempt_index"][0]) != self._record_count:
            raise RuntimeError("Stage D ledger attempt order is not contiguous")
        if values.size and not np.array_equal(
            values["attempt_index"],
            np.arange(self._record_count, self._record_count + values.size, dtype=np.uint64),
        ):
            raise RuntimeError("Stage D ledger batch attempt order is not contiguous")
        if np.any(values["status"] == np.uint8(AttemptStatus.PENDING_SEMANTIC_VALID)):
            raise RuntimeError("Stage D ledger cannot persist unresolved semantic-valid attempts")
        encoded = values.tobytes(order="C")
        self._handle.write(encoded)
        self._digest.update(encoded)
        self._record_count += int(values.size)

    def flush(self, *, durable: bool = False) -> None:
        self._handle.flush()
        if durable:
            os.fsync(self._handle.fileno())

    def truncate(self, record_count: int) -> None:
        if record_count < 0 or record_count > self._record_count:
            raise ValueError("Stage D ledger truncate target is outside the file")
        self._handle.truncate(record_count * ATTEMPT_DTYPE.itemsize)
        self._handle.flush()
        self._record_count = int(record_count)
        self._handle.seek(0)
        self._digest = hashlib.sha256()
        remaining = self.byte_offset
        while remaining:
            chunk = self._handle.read(min(1024 * 1024, remaining))
            if not chunk:
                raise RuntimeError("Stage D ledger ended before its truncate target")
            self._digest.update(chunk)
            remaining -= len(chunk)
        self._handle.seek(0, os.SEEK_END)

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "AttemptLedger":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def read_attempt_batches(
    path: Path, *, start: int = 0, stop: int | None = None, batch_size: int = 65_536
) -> Iterator[np.ndarray]:
    if start < 0 or batch_size < 1:
        raise ValueError("Stage D ledger read bounds are invalid")
    size = Path(path).stat().st_size
    if size % ATTEMPT_DTYPE.itemsize:
        raise RuntimeError("Stage D attempt ledger ends inside a binary record")
    record_count = size // ATTEMPT_DTYPE.itemsize
    resolved_stop = record_count if stop is None else int(stop)
    if resolved_stop < start or resolved_stop > record_count:
        raise ValueError("Stage D ledger read range is outside the file")
    with Path(path).open("rb") as handle:
        handle.seek(start * ATTEMPT_DTYPE.itemsize)
        remaining = resolved_stop - start
        while remaining:
            count = min(batch_size, remaining)
            encoded = handle.read(count * ATTEMPT_DTYPE.itemsize)
            if len(encoded) != count * ATTEMPT_DTYPE.itemsize:
                raise RuntimeError("Stage D attempt ledger ended during batch read")
            yield np.frombuffer(encoded, dtype=ATTEMPT_DTYPE).copy()
            remaining -= count