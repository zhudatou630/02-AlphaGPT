"""Serial Stage D runner built on the compact V3A training components."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import time
from typing import Any

import numpy as np
import torch

from alpha_etf.research_v3a.archive_tail import (
    ARCHIVE_TAIL_OBJECTIVE_VERSION,
    ArchiveTailConfig,
    ArchiveTailState,
    archive_tail_objective,
)
from alpha_etf.research_v3a.attempts import (
    AttemptLedger,
    AttemptStatus,
    PreparedAttemptBatch,
    read_attempt_batches,
)
from alpha_etf.research_v3a.attempt_workers import (
    AttemptBatchPreparer,
    SubmittedAttemptBatch,
)
from alpha_etf.research_v3a.candidate_index import CompactCandidateIndex
from alpha_etf.research_v3a.candidates import (
    CandidateConfig,
    CandidateRecord,
    build_candidate_record,
)
from alpha_etf.research_v3a.gpu_sampling import TensorFormulaSampler
from alpha_etf.research_v3a.language import FORMULA_VOCAB
from alpha_etf.research_v3a.scoring import ScorerConfig
from alpha_etf.research_v3a.spec import sha256_file
from alpha_etf.research_v3a.stage_d import (
    effective_training_rewards,
    reinforce_objective,
)
from alpha_etf.research_v3a.stage_d_checkpoint import (
    build_training_checkpoint,
    load_training_checkpoint,
    restore_training_checkpoint,
    save_training_checkpoint,
)
from alpha_etf.research_v3a.torch_scoring import (
    TorchForwardTargets,
    score_signal_batch_chunked,
    signal_quality_batch_chunked,
)
from alpha_etf.research_v3a.torch_vm import BatchTorchVM


SERIAL_RUNNER_SCHEMA_VERSION = "etf-v3a-stage-d-serial-runner-v4"
TRAINING_SUMMARY_SCHEMA_VERSION = "etf-v3a-stage-d-training-summary-v2"
TRAINING_COMPLETE_SCHEMA_VERSION = "etf-v3a-stage-d-training-complete-v2"
FAST_CHECKPOINT_SECONDS = 30 * 60
CANDIDATE_SNAPSHOT_SECONDS = 120 * 60
SCORER_BATCH_CHUNK_SIZE = 4096


@dataclass(frozen=True)
class SerialStageDConfig:
    run_id: str
    run_identity: dict[str, Any]
    method: str
    seed: int
    attempts: int
    batch_size: int
    cpu_worker_count: int
    cpu_gpu_overlap: bool
    training_invalid_reward: float
    advantage_epsilon: float
    entropy_coefficient: float
    gradient_clip_norm: float
    scorer_batch_chunk_size: int = SCORER_BATCH_CHUNK_SIZE
    release_cuda_cache_after_batch: bool = True
    checkpoint_seconds: float = FAST_CHECKPOINT_SECONDS
    candidate_snapshot_seconds: float = CANDIDATE_SNAPSHOT_SECONDS
    candidate_snapshot_on_stop: bool = False
    learning_objective: str = "legacy_reinforce"
    archive_tail_config: ArchiveTailConfig | None = None

    def __post_init__(self) -> None:
        if self.method not in {"transformer", "matched_random"}:
            raise ValueError(f"Unsupported Stage D method: {self.method}")
        if self.attempts < 1 or self.batch_size < 2:
            raise ValueError("Stage D attempts and batch size are invalid")
        if self.cpu_worker_count < 1:
            raise ValueError("Stage D CPU worker count must be positive")
        if self.scorer_batch_chunk_size < 1:
            raise ValueError("Stage D scorer batch chunk size must be positive")
        if self.method == "transformer" and self.attempts % self.batch_size == 1:
            raise ValueError("Stage D Transformer cannot end with a one-formula batch")
        if self.checkpoint_seconds <= 0 or self.candidate_snapshot_seconds <= 0:
            raise ValueError("Stage D checkpoint intervals must be positive")
        if self.learning_objective not in {
            "legacy_reinforce",
            ARCHIVE_TAIL_OBJECTIVE_VERSION,
        }:
            raise ValueError("Unsupported Stage D learning objective")
        if self.learning_objective == ARCHIVE_TAIL_OBJECTIVE_VERSION:
            if self.method != "transformer" or self.archive_tail_config is None:
                raise ValueError("Archive-tail learning requires Transformer configuration")
            if self.entropy_coefficient != 0.0:
                raise ValueError("Archive-tail learning requires zero entropy coefficient")
            if self.batch_size % self.archive_tail_config.model_fraction_denominator:
                raise ValueError("Archive-tail batch size must preserve its lane ratio")
            if self.attempts % self.archive_tail_config.model_fraction_denominator:
                raise ValueError("Archive-tail attempts must preserve its lane ratio")
        elif self.archive_tail_config is not None:
            raise ValueError("Legacy Stage D cannot receive archive-tail configuration")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _PendingBatch:
    step: int
    attempt_start: int
    batch_count: int
    submission: SubmittedAttemptBatch | None
    prepared: PreparedAttemptBatch | None
    prepare_started: float
    gpu_seconds: float
    batch_started: float
    training_metrics: dict[str, Any]


@dataclass(frozen=True)
class _LaneBatch:
    arguments: dict[str, np.ndarray | int]
    gpu_seconds: float
    entropy: float
    normalized_entropy: float
    average_allowed_actions: float


class _BatchLog:
    def __init__(self, path: Path, *, create: bool):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("x+b" if create else "r+b")
        self._digest = hashlib.sha256()
        self._handle.seek(0)
        while chunk := self._handle.read(1024 * 1024):
            self._digest.update(chunk)
        self._handle.seek(0, os.SEEK_END)

    @property
    def byte_offset(self) -> int:
        return int(self._handle.tell())

    @property
    def prefix_sha256(self) -> str:
        return self._digest.hexdigest()

    def append(self, payload: dict[str, Any]) -> None:
        encoded = (
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        self._handle.write(encoded)
        self._digest.update(encoded)

    def flush(self, *, durable: bool = False) -> None:
        self._handle.flush()
        if durable:
            os.fsync(self._handle.fileno())

    def truncate(self, byte_offset: int) -> None:
        if byte_offset < 0 or byte_offset > self.byte_offset:
            raise ValueError("Stage D training log truncate target is outside the file")
        self._handle.truncate(byte_offset)
        self._handle.flush()
        self._handle.seek(0)
        self._digest = hashlib.sha256()
        remaining = byte_offset
        while remaining:
            chunk = self._handle.read(min(1024 * 1024, remaining))
            if not chunk:
                raise RuntimeError("Stage D training log ended before its checkpoint")
            self._digest.update(chunk)
            remaining -= len(chunk)
        self._handle.seek(0, os.SEEK_END)

    def close(self) -> None:
        self._handle.close()


def _write_json_atomic(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(encoded)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _reward_summary(ledger_path: Path) -> dict[str, float | int | None]:
    chunks: list[np.ndarray] = []
    for records in read_attempt_batches(ledger_path):
        valid = np.isfinite(records["reward"])
        if np.any(valid):
            chunks.append(records["reward"][valid].astype(np.float64))
    if not chunks:
        return {"count": 0, "mean": None, "std": None, "min": None, "median": None, "max": None}
    values = np.concatenate(chunks)
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "median": float(np.median(values)),
        "max": float(values.max()),
    }


class SerialStageDRunner:
    """Run one method serially; later stages may replace only its scheduling."""

    def __init__(
        self,
        *,
        config: SerialStageDConfig,
        run_dir: Path,
        sampler: TensorFormulaSampler,
        vm: BatchTorchVM,
        factors: torch.Tensor,
        tradable_mask: torch.Tensor,
        targets: TorchForwardTargets,
        scorer_config: ScorerConfig,
        candidate_config: CandidateConfig,
        model: torch.nn.Module | None,
        optimizer: torch.optim.Optimizer | None,
    ):
        if (config.method == "transformer") != (model is not None and optimizer is not None):
            raise ValueError("Stage D method differs from its model and optimizer")
        self.config = config
        self.run_dir = Path(run_dir)
        self.sampler = sampler
        self.vm = vm
        self.factors = factors
        self.tradable_mask = tradable_mask
        self.targets = targets
        self.scorer_config = scorer_config
        self.candidate_config = candidate_config
        self.model = model
        self.optimizer = optimizer
        self.device = factors.device
        self._top_k_cpu = targets.top_k.detach().cpu().numpy()
        self.checkpoint_path = self.run_dir / "checkpoint_latest.pt"
        self.final_checkpoint_path = self.run_dir / "checkpoint_final.pt"
        self.ledger_path = self.run_dir / "attempts.bin"
        self.log_path = self.run_dir / "training_log.jsonl"
        self._stop_requested = False

    def request_stop(self) -> None:
        self._stop_requested = True

    def run(self, *, resume: bool = False, stop_after: int | None = None) -> dict[str, Any]:
        run_until = self._run_until(stop_after)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        existing = [path for path in self.run_dir.iterdir() if path.name != ".run.lock"]
        if not resume and existing:
            raise FileExistsError(f"Stage D run directory is not empty: {self.run_dir}")

        if resume:
            state = self._resume()
            ledger = state.pop("ledger_handle")
            batch_log = state.pop("log_handle")
            index = state.pop("candidate_index")
            archive_tail_state = state.pop("archive_tail_state", None)
        else:
            ledger = AttemptLedger(self.ledger_path, create=True)
            batch_log = _BatchLog(self.log_path, create=True)
            index = CompactCandidateIndex(
                candidate_config=self.candidate_config,
                initial_best_reward=self.scorer_config.hard_invalid_reward,
            )
            state = {
                "step": 0,
                "attempt_count": 0,
                "elapsed_seconds": 0.0,
                "resume_count": 0,
            }
            archive_tail_state = (
                ArchiveTailState(self.config.archive_tail_config)
                if self.config.learning_objective == ARCHIVE_TAIL_OBJECTIVE_VERSION
                else None
            )
            self._save_candidate_snapshot(index, state)
            ledger.flush(durable=True)
            batch_log.flush(durable=True)
            self._save_checkpoint(state, ledger, batch_log)

        committed = int(state["attempt_count"])
        if committed > run_until:
            ledger.close()
            batch_log.close()
            raise ValueError("Stage D stop target is behind the checkpoint")

        elapsed_before = float(state["elapsed_seconds"])
        session_started = time.monotonic()
        last_checkpoint_elapsed = elapsed_before
        last_checkpoint_attempt = committed
        generated = committed
        generated_step = int(state["step"])
        pending: list[_PendingBatch] = []
        preparer = AttemptBatchPreparer(self.config.cpu_worker_count)
        try:
            while generated < run_until and not self._stop_requested:
                generated_step += 1
                batch_count = min(self.config.batch_size, run_until - generated)
                produced = self._produce_batch(
                    attempt_start=generated,
                    step=generated_step,
                    batch_count=batch_count,
                    preparer=preparer,
                    archive_tail_state=archive_tail_state,
                )
                if self.device.type == "cuda" and self.config.release_cuda_cache_after_batch:
                    cache_started = time.perf_counter()
                    torch.cuda.empty_cache()
                    produced = replace(
                        produced,
                        gpu_seconds=(
                            produced.gpu_seconds + time.perf_counter() - cache_started
                        ),
                    )
                pending.append(produced)
                generated += batch_count

                if not self.config.cpu_gpu_overlap or len(pending) >= 2:
                    self._commit_pending(
                        pending.pop(0),
                        state=state,
                        index=index,
                        ledger=ledger,
                        batch_log=batch_log,
                        elapsed_before=elapsed_before,
                        session_started=session_started,
                        pending_batches=len(pending),
                    )

                elapsed_now = elapsed_before + time.monotonic() - session_started
                snapshot_due = (
                    elapsed_now - float(state["snapshot_elapsed_seconds"])
                    >= self.config.candidate_snapshot_seconds
                    or generated == self.config.attempts
                    or (
                        self.config.candidate_snapshot_on_stop
                        and (generated == run_until or self._stop_requested)
                    )
                )
                checkpoint_due = (
                    elapsed_now - last_checkpoint_elapsed
                    >= self.config.checkpoint_seconds
                    or generated == run_until
                    or self._stop_requested
                )
                if snapshot_due or checkpoint_due:
                    while pending:
                        self._commit_pending(
                            pending.pop(0),
                            state=state,
                            index=index,
                            ledger=ledger,
                            batch_log=batch_log,
                            elapsed_before=elapsed_before,
                            session_started=session_started,
                            pending_batches=len(pending),
                        )
                    elapsed_now = elapsed_before + time.monotonic() - session_started
                    state["elapsed_seconds"] = elapsed_now
                    if snapshot_due:
                        self._save_candidate_snapshot(index, state)
                    self._save_checkpoint(state, ledger, batch_log)
                    last_checkpoint_elapsed = elapsed_now
                    last_checkpoint_attempt = int(state["attempt_count"])

            while pending:
                self._commit_pending(
                    pending.pop(0),
                    state=state,
                    index=index,
                    ledger=ledger,
                    batch_log=batch_log,
                    elapsed_before=elapsed_before,
                    session_started=session_started,
                    pending_batches=len(pending),
                )
            if int(state["attempt_count"]) != last_checkpoint_attempt:
                elapsed_now = elapsed_before + time.monotonic() - session_started
                state["elapsed_seconds"] = elapsed_now
                if (
                    int(state["attempt_count"]) == self.config.attempts
                    or self.config.candidate_snapshot_on_stop
                ):
                    self._save_candidate_snapshot(index, state)
                self._save_checkpoint(state, ledger, batch_log)
        finally:
            preparer.close()
            ledger.close()
            batch_log.close()

        completed = int(state["attempt_count"])
        if completed < self.config.attempts:
            return {
                "status": "checkpointed",
                "run_id": self.config.run_id,
                "method": self.config.method,
                "attempt_count": completed,
                "target_attempts": self.config.attempts,
            }
        return self._finalize(index, state, archive_tail_state=archive_tail_state)

    def _produce_batch(
        self,
        *,
        attempt_start: int,
        step: int,
        batch_count: int,
        preparer: AttemptBatchPreparer,
        archive_tail_state: ArchiveTailState | None,
    ) -> _PendingBatch:
        if self.config.learning_objective == ARCHIVE_TAIL_OBJECTIVE_VERSION:
            if archive_tail_state is None:
                raise RuntimeError("Archive-tail runner lacks its learning state")
            return self._produce_archive_tail_batch(
                attempt_start=attempt_start,
                step=step,
                batch_count=batch_count,
                preparer=preparer,
                archive_tail_state=archive_tail_state,
            )
        batch_started = time.perf_counter()
        gpu_started = time.perf_counter()
        if self.config.method == "transformer":
            assert self.model is not None
            self.model.train()
            sample = self.sampler.sample_policy(self.model, batch_count)
        else:
            sample = self.sampler.sample_uniform(batch_count)

        vm_result = self.vm.execute(
            sample.vm_codes,
            sample.vm_lengths,
            self.factors,
            self.tradable_mask,
            max_stack_depth=self.sampler.tables.max_vm_stack_depth,
        )
        scored = score_signal_batch_chunked(
            vm_result.signal,
            vm_result.valid,
            self.targets,
            self.scorer_config,
            chunk_size=self.config.scorer_batch_chunk_size,
        )
        quality_valid, coverage, finite_std, variation_valid = (
            signal_quality_batch_chunked(
                vm_result.signal,
                self.targets,
                min_coverage=self.candidate_config.min_coverage,
                constant_std_eps=self.candidate_config.constant_std_eps,
                chunk_size=self.config.scorer_batch_chunk_size,
            )
        )
        training_rewards = effective_training_rewards(
            scored.reward,
            scorer_valid=scored.valid,
            quality_valid=quality_valid,
            hard_invalid_reward=self.config.training_invalid_reward,
        )

        objective = None
        gradient_norm: float | None = None
        if self.config.method == "transformer":
            assert self.optimizer is not None
            objective = reinforce_objective(
                log_prob_sums=sample.log_prob_sums,
                entropy_sums=sample.entropy_sums,
                decision_counts=sample.token_lengths + 1,
                rewards=training_rewards,
                advantage_epsilon=self.config.advantage_epsilon,
                entropy_coefficient=self.config.entropy_coefficient,
            )
            self.optimizer.zero_grad(set_to_none=True)
            objective.loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.gradient_clip_norm
            )
            if not bool(torch.isfinite(norm).item()):
                raise RuntimeError("Stage D gradient norm is non-finite")
            gradient_norm = float(norm.detach().cpu().item())
            self.optimizer.step()

        decision_counts = sample.token_lengths + 1
        training_metrics = {
            "loss": float(objective.loss.detach().cpu().item()) if objective else None,
            "reward_mean": (
                float(objective.reward_mean.detach().cpu().item()) if objective else None
            ),
            "reward_std": (
                float(objective.reward_std.detach().cpu().item()) if objective else None
            ),
            "entropy": float(
                (sample.entropy_sums / decision_counts).mean().detach().cpu().item()
            ),
            "normalized_entropy": float(
                (
                    sample.normalized_entropy_sums
                    / decision_counts.to(torch.float32)
                )
                .mean()
                .detach()
                .cpu()
                .item()
            ),
            "average_allowed_actions": float(
                sample.average_allowed_actions.detach().cpu().item()
            ),
            "gradient_norm": gradient_norm,
        }
        token_ids_cpu = sample.token_ids.detach().cpu().numpy()
        token_lengths_cpu = sample.token_lengths.detach().cpu().numpy()
        vm_valid_cpu = vm_result.valid.detach().cpu().numpy()
        score_valid_cpu = scored.valid.detach().cpu().numpy()
        quality_valid_cpu = quality_valid.detach().cpu().numpy()
        variation_valid_cpu = variation_valid.detach().cpu().numpy()
        rewards_cpu = scored.reward.detach().cpu().numpy()
        training_rewards_cpu = training_rewards.detach().cpu().numpy()
        coverage_cpu = coverage.detach().cpu().numpy()
        finite_std_cpu = finite_std.detach().cpu().numpy()
        selected_cpu = scored.selected_indices.detach().to(torch.int16).cpu().numpy()
        gpu_seconds = time.perf_counter() - gpu_started
        prepare_started = time.perf_counter()
        arguments = {
            "attempt_start": attempt_start,
            "training_step": step,
            "token_ids": token_ids_cpu,
            "token_lengths": token_lengths_cpu,
            "vm_valid": vm_valid_cpu,
            "score_valid": score_valid_cpu,
            "quality_valid": quality_valid_cpu,
            "variation_valid": variation_valid_cpu,
            "rewards": rewards_cpu,
            "training_rewards": training_rewards_cpu,
            "coverage": coverage_cpu,
            "finite_std": finite_std_cpu,
            "selected_indices": selected_cpu,
            "top_k": self._top_k_cpu,
        }
        if self.config.cpu_gpu_overlap:
            submission = preparer.submit(**arguments)
            prepared = None
        else:
            submission = None
            prepared = preparer.prepare(**arguments)
        return _PendingBatch(
            step=step,
            attempt_start=attempt_start,
            batch_count=batch_count,
            submission=submission,
            prepared=prepared,
            prepare_started=prepare_started,
            gpu_seconds=gpu_seconds,
            batch_started=batch_started,
            training_metrics=training_metrics,
        )

    def _evaluate_archive_tail_lane(
        self,
        *,
        sample: Any,
        attempt_start: int,
        step: int,
        gpu_started: float,
    ) -> _LaneBatch:
        vm_result = self.vm.execute(
            sample.vm_codes,
            sample.vm_lengths,
            self.factors,
            self.tradable_mask,
            max_stack_depth=self.sampler.tables.max_vm_stack_depth,
        )
        scored = score_signal_batch_chunked(
            vm_result.signal,
            vm_result.valid,
            self.targets,
            self.scorer_config,
            chunk_size=self.config.scorer_batch_chunk_size,
        )
        quality_valid, coverage, finite_std, variation_valid = (
            signal_quality_batch_chunked(
                vm_result.signal,
                self.targets,
                min_coverage=self.candidate_config.min_coverage,
                constant_std_eps=self.candidate_config.constant_std_eps,
                chunk_size=self.config.scorer_batch_chunk_size,
            )
        )
        decision_counts = sample.token_lengths + 1
        arguments: dict[str, np.ndarray | int] = {
            "attempt_start": attempt_start,
            "training_step": step,
            "token_ids": sample.token_ids.detach().cpu().numpy(),
            "token_lengths": sample.token_lengths.detach().cpu().numpy(),
            "vm_valid": vm_result.valid.detach().cpu().numpy(),
            "score_valid": scored.valid.detach().cpu().numpy(),
            "quality_valid": quality_valid.detach().cpu().numpy(),
            "variation_valid": variation_valid.detach().cpu().numpy(),
            "rewards": scored.reward.detach().cpu().numpy(),
            "training_rewards": np.zeros(
                int(sample.token_ids.shape[0]), dtype=np.float32
            ),
            "coverage": coverage.detach().cpu().numpy(),
            "finite_std": finite_std.detach().cpu().numpy(),
            "selected_indices": scored.selected_indices.detach()
            .to(torch.int16)
            .cpu()
            .numpy(),
            "top_k": self._top_k_cpu,
        }
        return _LaneBatch(
            arguments=arguments,
            gpu_seconds=time.perf_counter() - gpu_started,
            entropy=float(
                (sample.entropy_sums / decision_counts).mean().detach().cpu().item()
            ),
            normalized_entropy=float(
                (
                    sample.normalized_entropy_sums
                    / decision_counts.to(torch.float32)
                )
                .mean()
                .detach()
                .cpu()
                .item()
            ),
            average_allowed_actions=float(
                sample.average_allowed_actions.detach().cpu().item()
            ),
        )

    def _produce_archive_tail_batch(
        self,
        *,
        attempt_start: int,
        step: int,
        batch_count: int,
        preparer: AttemptBatchPreparer,
        archive_tail_state: ArchiveTailState,
    ) -> _PendingBatch:
        assert self.model is not None and self.optimizer is not None
        learning = archive_tail_state.config
        model_count = learning.model_count(batch_count)
        random_count = batch_count - model_count
        batch_started = time.perf_counter()
        self.model.train()

        model_gpu_started = time.perf_counter()
        with torch.no_grad():
            model_sample = self.sampler.sample_policy(self.model, model_count)
            model_lane = self._evaluate_archive_tail_lane(
                sample=model_sample,
                attempt_start=attempt_start,
                step=step,
                gpu_started=model_gpu_started,
            )
        model_prepare_started = time.perf_counter()
        model_submission = preparer.submit(**model_lane.arguments)

        random_gpu_started = time.perf_counter()
        with torch.no_grad():
            random_sample = self.sampler.sample_uniform(random_count)
            random_lane = self._evaluate_archive_tail_lane(
                sample=random_sample,
                attempt_start=attempt_start + model_count,
                step=step,
                gpu_started=random_gpu_started,
            )
        random_submission = preparer.submit(**random_lane.arguments)

        label_wait_started = time.perf_counter()
        model_prepared = model_submission.result()
        canonical_label_wait_seconds = time.perf_counter() - label_wait_started
        model_cpu_prepare_seconds = time.perf_counter() - model_prepare_started
        labels = archive_tail_state.apply(model_prepared.records, prepared=True)
        model_records = model_prepared.records.copy()
        model_records["training_reward"] = labels.combined_weights
        model_prepared = PreparedAttemptBatch(model_records)

        teacher_started = time.perf_counter()
        log_prob_sums = self.sampler.score_policy_sequences(
            self.model,
            torch.as_tensor(model_lane.arguments["token_ids"], device=self.device),
            torch.as_tensor(model_lane.arguments["token_lengths"], device=self.device),
        )
        elite_weights = torch.as_tensor(
            labels.elite_weights, dtype=log_prob_sums.dtype, device=self.device
        )
        archive_weights = torch.as_tensor(
            labels.archive_weights, dtype=log_prob_sums.dtype, device=self.device
        )
        waste_weights = torch.as_tensor(
            labels.waste_weights, dtype=log_prob_sums.dtype, device=self.device
        )
        objective = archive_tail_objective(
            log_prob_sums=log_prob_sums,
            elite_weights=elite_weights,
            archive_weights=archive_weights,
            waste_weights=waste_weights,
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        teacher_forcing_seconds = time.perf_counter() - teacher_started

        backward_started = time.perf_counter()
        self.optimizer.zero_grad(set_to_none=True)
        objective.loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.gradient_clip_norm
        )
        if not bool(torch.isfinite(norm).item()):
            raise RuntimeError("Stage D archive-tail gradient norm is non-finite")
        gradient_norm = float(norm.detach().cpu().item())
        self.optimizer.step()
        backward_seconds = time.perf_counter() - backward_started

        random_wait_started = time.perf_counter()
        random_prepared = random_submission.result()
        random_cpu_wait_seconds = time.perf_counter() - random_wait_started
        prepared = PreparedAttemptBatch(
            np.concatenate([model_prepared.records, random_prepared.records])
        )
        semantic_rewards = model_prepared.records["reward"][
            np.isfinite(model_prepared.records["reward"])
        ].astype(np.float64)
        training_metrics: dict[str, Any] = {
            "loss": float(objective.loss.detach().cpu().item()),
            "elite_loss": float(objective.elite_loss.detach().cpu().item()),
            "archive_loss": float(objective.archive_loss.detach().cpu().item()),
            "waste_loss": float(objective.waste_loss.detach().cpu().item()),
            "reward_mean": (
                float(semantic_rewards.mean()) if semantic_rewards.size else None
            ),
            "reward_std": (
                float(semantic_rewards.std()) if semantic_rewards.size else None
            ),
            "entropy": model_lane.entropy,
            "normalized_entropy": model_lane.normalized_entropy,
            "average_allowed_actions": model_lane.average_allowed_actions,
            "gradient_norm": gradient_norm,
            "model_lane_count": model_count,
            "random_lane_count": random_count,
            "model_gpu_seconds": model_lane.gpu_seconds,
            "random_gpu_seconds": random_lane.gpu_seconds,
            "model_cpu_prepare_seconds": model_cpu_prepare_seconds,
            "canonical_label_wait_seconds": canonical_label_wait_seconds,
            "teacher_forcing_seconds": teacher_forcing_seconds,
            "backward_seconds": backward_seconds,
            "random_cpu_wait_seconds": random_cpu_wait_seconds,
            "mechanism_critical_path_seconds": time.perf_counter() - batch_started,
            **labels.metrics,
        }
        return _PendingBatch(
            step=step,
            attempt_start=attempt_start,
            batch_count=batch_count,
            submission=None,
            prepared=prepared,
            prepare_started=model_prepare_started,
            gpu_seconds=(
                model_lane.gpu_seconds
                + random_lane.gpu_seconds
                + teacher_forcing_seconds
                + backward_seconds
            ),
            batch_started=batch_started,
            training_metrics=training_metrics,
        )

    def _commit_pending(
        self,
        pending: _PendingBatch,
        *,
        state: dict[str, Any],
        index: CompactCandidateIndex,
        ledger: AttemptLedger,
        batch_log: _BatchLog,
        elapsed_before: float,
        session_started: float,
        pending_batches: int,
    ) -> None:
        wait_started = time.perf_counter()
        if pending.prepared is not None:
            prepared = pending.prepared
        else:
            assert pending.submission is not None
            prepared = pending.submission.result()
        cpu_wait_seconds = time.perf_counter() - wait_started
        cpu_prepare_seconds = time.perf_counter() - pending.prepare_started
        commit_started = time.perf_counter()
        resolved = index.commit(prepared)
        ledger.append(resolved)
        commit_seconds = time.perf_counter() - commit_started
        completed = pending.attempt_start + pending.batch_count
        elapsed = elapsed_before + time.monotonic() - session_started
        state.update(
            {
                "step": pending.step,
                "attempt_count": completed,
                "elapsed_seconds": elapsed,
            }
        )
        semantic = np.isfinite(resolved["reward"])
        batch_log.append(
            self._batch_metrics(
                pending=pending,
                completed=completed,
                resolved=resolved,
                semantic=semantic,
                cpu_prepare_seconds=cpu_prepare_seconds,
                cpu_wait_seconds=cpu_wait_seconds,
                commit_seconds=commit_seconds,
                pending_batches=pending_batches,
                index=index,
                elapsed=elapsed,
            )
        )

    def _run_until(self, stop_after: int | None) -> int:
        target = self.config.attempts if stop_after is None else int(stop_after)
        if target < 1 or target > self.config.attempts:
            raise ValueError("Stage D stop-after is outside the run budget")
        if target != self.config.attempts and target % self.config.batch_size:
            raise ValueError("Stage D stop-after must be a complete batch boundary")
        if (
            self.config.learning_objective == ARCHIVE_TAIL_OBJECTIVE_VERSION
            and self.config.archive_tail_config is not None
            and target % self.config.archive_tail_config.model_fraction_denominator
        ):
            raise ValueError("Archive-tail stop target must preserve its lane ratio")
        return target

    def _resume(self) -> dict[str, Any]:
        checkpoint = load_training_checkpoint(
            self.checkpoint_path,
            expected_run_id=self.config.run_id,
            expected_run_identity=self.config.run_identity,
        )
        ledger = AttemptLedger(
            self.ledger_path, create=False, allow_incomplete_tail=True
        )
        log = _BatchLog(self.log_path, create=False)
        ledger.truncate(int(checkpoint["ledger"]["record_count"]))
        log.truncate(int(checkpoint["training_log"]["byte_offset"]))
        if ledger.prefix_sha256 != checkpoint["ledger"]["prefix_sha256"]:
            ledger.close()
            log.close()
            raise RuntimeError("Stage D ledger prefix differs from its checkpoint")
        if log.prefix_sha256 != checkpoint["training_log"]["prefix_sha256"]:
            ledger.close()
            log.close()
            raise RuntimeError("Stage D training log prefix differs from its checkpoint")
        snapshot = checkpoint["candidate_snapshot"]
        snapshot_path = self.run_dir / str(snapshot["name"])
        if not snapshot_path.is_file():
            ledger.close()
            log.close()
            raise RuntimeError("Stage D candidate snapshot is missing")
        if sha256_file(snapshot_path) != snapshot["sha256"]:
            ledger.close()
            log.close()
            raise RuntimeError("Stage D candidate snapshot differs from its checkpoint")
        try:
            index = CompactCandidateIndex.restore(
                snapshot_path=snapshot_path,
                ledger_path=self.ledger_path,
                stop=int(checkpoint["attempt_count"]),
                expected_snapshot_attempt=int(snapshot["attempt_count"]),
            )
        except Exception:
            ledger.close()
            log.close()
            raise
        restored = restore_training_checkpoint(
            checkpoint, model=self.model, optimizer=self.optimizer
        )
        archive_tail_state = None
        if self.config.learning_objective == ARCHIVE_TAIL_OBJECTIVE_VERSION:
            assert self.config.archive_tail_config is not None
            archive_tail_state = ArchiveTailState.restore_from_ledger(
                ledger_path=self.ledger_path,
                stop=int(checkpoint["attempt_count"]),
                batch_size=self.config.batch_size,
                config=self.config.archive_tail_config,
            )
        return {
            "step": restored["step"],
            "attempt_count": restored["attempt_count"],
            "elapsed_seconds": restored["elapsed_seconds"],
            "resume_count": restored["resume_count"] + 1,
            "snapshot_name": snapshot["name"],
            "snapshot_attempt": snapshot["attempt_count"],
            "snapshot_elapsed_seconds": snapshot["elapsed_seconds"],
            "snapshot_sha256": snapshot["sha256"],
            "ledger_handle": ledger,
            "log_handle": log,
            "candidate_index": index,
            "archive_tail_state": archive_tail_state,
        }

    def _save_checkpoint(
        self,
        state: dict[str, Any],
        ledger: AttemptLedger,
        batch_log: _BatchLog,
    ) -> dict[str, Any]:
        checkpoint_started = time.perf_counter()
        ledger.flush(durable=True)
        batch_log.flush(durable=True)
        checkpoint = build_training_checkpoint(
            run_id=self.config.run_id,
            run_identity=self.config.run_identity,
            step=int(state["step"]),
            attempt_count=int(state["attempt_count"]),
            model=self.model,
            optimizer=self.optimizer,
            ledger_prefix_sha256=ledger.prefix_sha256,
            training_log_offset=batch_log.byte_offset,
            training_log_prefix_sha256=batch_log.prefix_sha256,
            candidate_snapshot_name=str(state["snapshot_name"]),
            candidate_snapshot_attempt=int(state["snapshot_attempt"]),
            candidate_snapshot_elapsed_seconds=float(
                state["snapshot_elapsed_seconds"]
            ),
            candidate_snapshot_sha256=str(state["snapshot_sha256"]),
            elapsed_seconds=float(state["elapsed_seconds"]),
            resume_count=int(state["resume_count"]),
        )
        if int(state["attempt_count"]) == self.config.attempts:
            started = time.perf_counter()
            save_training_checkpoint(checkpoint, self.final_checkpoint_path)
            self._record_storage_metric(
                event="final_checkpoint",
                path=self.final_checkpoint_path,
                attempt_count=int(state["attempt_count"]),
                seconds=time.perf_counter() - started,
            )
        started = time.perf_counter()
        save_training_checkpoint(checkpoint, self.checkpoint_path)
        self._record_storage_metric(
            event="latest_checkpoint",
            path=self.checkpoint_path,
            attempt_count=int(state["attempt_count"]),
            seconds=time.perf_counter() - started,
        )
        self._record_storage_metric(
            event="checkpoint_barrier",
            path=self.checkpoint_path,
            attempt_count=int(state["attempt_count"]),
            seconds=time.perf_counter() - checkpoint_started,
        )
        return checkpoint

    def _save_candidate_snapshot(
        self, index: CompactCandidateIndex, state: dict[str, Any]
    ) -> None:
        attempt_count = int(state["attempt_count"])
        snapshot_name = self._snapshot_name(attempt_count)
        snapshot_path = self.run_dir / snapshot_name
        started = time.perf_counter()
        index.save_snapshot(snapshot_path)
        snapshot_sha256 = sha256_file(snapshot_path)
        seconds = time.perf_counter() - started
        state.update(
            {
                "snapshot_name": snapshot_name,
                "snapshot_attempt": attempt_count,
                "snapshot_elapsed_seconds": float(state["elapsed_seconds"]),
                "snapshot_sha256": snapshot_sha256,
            }
        )
        self._record_storage_metric(
            event="candidate_snapshot",
            path=snapshot_path,
            attempt_count=attempt_count,
            seconds=seconds,
        )

    def _record_storage_metric(
        self, *, event: str, path: Path, attempt_count: int, seconds: float
    ) -> None:
        payload = {
            "event": event,
            "attempt_count": attempt_count,
            "seconds": seconds,
            "bytes": path.stat().st_size,
            "path": path.name,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        with (self.run_dir / "storage_metrics.jsonl").open(
            "a", encoding="utf-8"
        ) as handle:
            handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")))
            handle.write("\n")

    def _batch_metrics(
        self,
        *,
        pending: _PendingBatch,
        completed: int,
        resolved: np.ndarray,
        semantic: np.ndarray,
        cpu_prepare_seconds: float,
        cpu_wait_seconds: float,
        commit_seconds: float,
        pending_batches: int,
        index: CompactCandidateIndex,
        elapsed: float,
    ) -> dict[str, Any]:
        semantic_rewards = resolved["reward"][semantic]
        batch_latency_seconds = time.perf_counter() - pending.batch_started
        payload = {
            "step": pending.step,
            "attempt_count": completed,
            "batch_count": pending.batch_count,
            "method": self.config.method,
            "cpu_worker_count": self.config.cpu_worker_count,
            "cpu_gpu_overlap": self.config.cpu_gpu_overlap,
            "loss": pending.training_metrics["loss"],
            "reward_mean": pending.training_metrics["reward_mean"],
            "reward_std": pending.training_metrics["reward_std"],
            "semantic_valid_count": int(semantic.sum()),
            "semantic_reward_mean": (
                float(semantic_rewards.mean()) if semantic_rewards.size else None
            ),
            "valid_rate": float(semantic.mean()),
            "entropy": pending.training_metrics["entropy"],
            "normalized_entropy": pending.training_metrics["normalized_entropy"],
            "average_formula_length": float(resolved["token_len"].mean()),
            "average_allowed_actions": pending.training_metrics[
                "average_allowed_actions"
            ],
            "gradient_norm": pending.training_metrics["gradient_norm"],
            "canonical_unique_count": index.canonical_size,
            "selection_unique_count": index.selection_size,
            "best_semantic_reward": index.best_semantic_reward,
            "gpu_batch_seconds": pending.gpu_seconds,
            "cpu_prepare_seconds": cpu_prepare_seconds,
            "cpu_wait_seconds": cpu_wait_seconds,
            "commit_seconds": commit_seconds,
            "batch_latency_seconds": batch_latency_seconds,
            "pending_cpu_batches": pending_batches,
            "elapsed_seconds": elapsed,
            "attempts_per_second": completed / elapsed if elapsed > 0 else None,
            "cpu_peak_rss_bytes": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            ),
            "cuda_allocated_bytes": (
                torch.cuda.memory_allocated(self.device)
                if self.device.type == "cuda"
                else None
            ),
            "cuda_reserved_bytes": (
                torch.cuda.memory_reserved(self.device)
                if self.device.type == "cuda"
                else None
            ),
        }
        if self.config.learning_objective == ARCHIVE_TAIL_OBJECTIVE_VERSION:
            mechanism_keys = (
                "elite_loss",
                "archive_loss",
                "waste_loss",
                "model_lane_count",
                "random_lane_count",
                "model_gpu_seconds",
                "random_gpu_seconds",
                "model_cpu_prepare_seconds",
                "canonical_label_wait_seconds",
                "teacher_forcing_seconds",
                "backward_seconds",
                "random_cpu_wait_seconds",
                "mechanism_critical_path_seconds",
                "model_attempt_count",
                "history_duplicate_count",
                "within_batch_duplicate_count",
                "semantic_invalid_count",
                "new_valid_count",
                "elite_count",
                "archive_improver_count",
                "new_valid_reward_mean",
                "new_valid_reward_q90",
                "new_valid_top10_mean",
                "archive_floor_before",
                "archive_count",
                "archive_mean_reward",
                "archive_floor_after",
            )
            payload["learning_objective"] = self.config.learning_objective
            for key in mechanism_keys:
                payload[key] = pending.training_metrics[key]
            critical = float(pending.training_metrics["mechanism_critical_path_seconds"])
            payload["canonical_label_wait_fraction"] = (
                float(pending.training_metrics["canonical_label_wait_seconds"])
                / critical
            )
            payload["teacher_forcing_fraction"] = (
                float(pending.training_metrics["teacher_forcing_seconds"])
                / critical
            )
            if self.device.type == "cuda":
                cuda_free, cuda_total = torch.cuda.mem_get_info(self.device)
                payload["cuda_free_bytes"] = int(cuda_free)
                payload["cuda_total_bytes"] = int(cuda_total)
        return payload

    def _retained_candidates(self, index: CompactCandidateIndex) -> list[CandidateRecord]:
        records: list[CandidateRecord] = []
        for row in index.retained_rows():
            token_len = int(index.canonical_best_token_lengths[row])
            tokens = [int(value) for value in index.canonical_best_tokens[row, :token_len]]
            best_attempt = int(index.canonical_best_attempts[row])
            record = build_candidate_record(
                formula_id=(
                    f"{self.config.method}_s{self.config.seed}_a{best_attempt}"
                ),
                source=f"v3a_stage_d_{self.config.method}",
                token_ids=tokens,
                reward=float(index.canonical_best_rewards[row]),
                train_summary={
                    "scorer_days": int(self.targets.decision_indices.numel()),
                    "coverage": float(index.canonical_best_coverage[row]),
                    "finite_std": float(index.canonical_best_std[row]),
                    "training_step": int(index.canonical_best_steps[row]),
                },
                attempt_index=best_attempt,
            )
            records.append(
                replace(
                    record,
                    first_attempt_index=int(index.canonical_first_valid_attempts[row]),
                    attempt_count=int(index.canonical_valid_counts[row]),
                    best_attempt_index=best_attempt,
                )
            )
        return sorted(records, key=lambda item: (-item.reward, item.token_len, item.formula_hash))

    def _finalize(
        self,
        index: CompactCandidateIndex,
        state: dict[str, Any],
        *,
        archive_tail_state: ArchiveTailState | None = None,
    ) -> dict[str, Any]:
        retained = self._retained_candidates(index)
        retained_path = self.run_dir / "retained_candidates.json"
        _write_json_atomic(retained_path, [record.to_dict() for record in retained])
        status_counts = {
            status.name.lower(): int(index.status_counts[int(status)])
            for status in AttemptStatus
            if status is not AttemptStatus.PENDING_SEMANTIC_VALID
        }
        elapsed = float(state["elapsed_seconds"])
        summary = {
            "schema_version": TRAINING_SUMMARY_SCHEMA_VERSION,
            "runner_schema_version": SERIAL_RUNNER_SCHEMA_VERSION,
            "status": "trained_awaiting_curated_library",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "run_id": self.config.run_id,
            "run_identity": self.config.run_identity,
            "method": self.config.method,
            "seed": self.config.seed,
            "attempt_count": self.config.attempts,
            "step_count": int(state["step"]),
            "batch_size": self.config.batch_size,
            "cpu_worker_count": self.config.cpu_worker_count,
            "cpu_gpu_overlap": self.config.cpu_gpu_overlap,
            "status_counts": status_counts,
            "canonical_unique_count": index.canonical_size,
            "structural_duplicate_count": index.structural_duplicate_count,
            "selection_unique_count": index.selection_size,
            "selection_duplicate_count": index.selection_duplicate_count,
            "semantic_valid_count": index.semantic_valid_count,
            "accepted_unique_count": status_counts["accepted_unique"],
            "length_distribution": {
                str(length): int(count)
                for length, count in enumerate(index.length_counts)
                if count
            },
            "token_distribution": {
                FORMULA_VOCAB.id_to_token(token_id).name: int(count)
                for token_id, count in enumerate(index.token_counts)
                if count
            },
            "reward_distribution": _reward_summary(self.ledger_path),
            "retained_candidate_count": len(retained),
            "retained_bucket_counts": [
                sum(record.token_len <= 5 for record in retained),
                sum(5 < record.token_len <= 10 for record in retained),
                sum(record.token_len > 10 for record in retained),
            ],
            "best_retained_reward": retained[0].reward if retained else None,
            "best_reward_attempt_auc": (
                index.best_reward_attempt_auc_numerator / self.config.attempts
            ),
            "attempt_ledger_prefix_sha256": sha256_file(self.ledger_path),
            "resume_count": int(state["resume_count"]),
            "elapsed_seconds": elapsed,
            "attempts_per_second": self.config.attempts / elapsed if elapsed > 0 else None,
            "checkpoint_latest_bytes": self.checkpoint_path.stat().st_size,
            "checkpoint_final_bytes": self.final_checkpoint_path.stat().st_size,
            "candidate_snapshot": str(state["snapshot_name"]),
            "candidate_snapshot_bytes": (self.run_dir / str(state["snapshot_name"])).stat().st_size,
            "cuda_peak_allocated_bytes": (
                torch.cuda.max_memory_allocated(self.device)
                if self.device.type == "cuda"
                else None
            ),
            "cuda_peak_reserved_bytes": (
                torch.cuda.max_memory_reserved(self.device)
                if self.device.type == "cuda"
                else None
            ),
            "validation_or_final_metrics_read": False,
        }
        if archive_tail_state is not None:
            archive_rewards = np.asarray(
                [entry.reward for entry in archive_tail_state.archive.values()],
                dtype=np.float64,
            )
            summary["archive_tail"] = {
                "model_attempt_count": archive_tail_state.model_attempt_count,
                "model_seen_canonical_count": len(archive_tail_state.seen),
                "model_archive_count": len(archive_tail_state.archive),
                "model_archive_mean_reward": (
                    float(archive_rewards.mean()) if archive_rewards.size else None
                ),
                "model_archive_floor_reward": (
                    float(archive_rewards.min())
                    if archive_rewards.size == archive_tail_state.config.archive_size
                    else None
                ),
                "config": archive_tail_state.config.to_dict(),
                "standard_candidate_index_scope": "combined_transformer_and_random_lanes",
                "separate_lane_libraries": "derived_by_gate1_read_only_analysis",
            }
            summary["learning_objective"] = self.config.learning_objective
        summary_path = self.run_dir / "training_summary.json"
        _write_json_atomic(summary_path, summary)
        marker = {
            "schema_version": TRAINING_COMPLETE_SCHEMA_VERSION,
            "run_id": self.config.run_id,
            "attempt_count": self.config.attempts,
            "summary_sha256": sha256_file(summary_path),
            "retained_candidates_sha256": sha256_file(retained_path),
            "checkpoint_final_sha256": sha256_file(self.final_checkpoint_path),
            "attempt_ledger_sha256": summary["attempt_ledger_prefix_sha256"],
        }
        _write_json_atomic(self.run_dir / "training_complete.json", marker)
        return summary

    @staticmethod
    def _snapshot_name(attempt_count: int) -> str:
        return f"candidate_snapshot_a{attempt_count}.npz"