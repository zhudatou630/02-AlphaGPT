#!/usr/bin/env python3
"""Run a protocol-gated, resumable V3A Stage D Transformer on CUDA."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from alpha_etf.gpt.policy import TransformerFormulaPolicy, TransformerPolicyConfig
from alpha_etf.research_v3a.candidates import (
    CandidateConfig,
    build_candidate_record,
    canonicalize_expression,
    expression_hash,
)
from alpha_etf.research_v3a.checkpointing import (
    assert_checkpoint_state_equal,
    atomic_save,
    build_checkpoint,
    load_checkpoint,
    restore_training_state,
)
from alpha_etf.research_v3a.language import FORMULA_VOCAB, compile_formula
from alpha_etf.research_v3a.sampling import PolicyVocab, SamplingConfig, sample_formulas
from alpha_etf.research_v3a.scoring import ScorerConfig, SplitSpec, build_forward_targets
from alpha_etf.research_v3a.spec import sha256_file
from alpha_etf.research_v3a.stage_d import (
    STATUS_KEYS,
    effective_training_rewards,
    empty_training_candidate_state,
    load_stage_d_binding,
    load_stage_d_protocol,
    load_stage_d_train_view,
    load_train_view_source_manifest,
    method_run_config,
    reinforce_objective,
    require_stage_d_cuda,
    retain_candidate,
    retained_candidates,
    sequence_digest,
    validate_training_candidate_state,
    verify_stage_c_prerequisite,
)
from alpha_etf.research_v3a.torch_scoring import TorchForwardTargets, score_signal_batch
from alpha_etf.research_v3a.torch_vm import BatchTorchVM, compiled_to_tensor
from scripts.v3a.random_baseline import (
    _quality_metrics,
    _reward_summary,
    _validate_ledger_prefix,
    _write_json_atomic,
    _write_jsonl_atomic,
)
from scripts.v3a.runtime import build_runtime_research_spec, require_clean_v3a_code


TRAINING_SCHEMA_VERSION = "etf-v3a-stage-d-transformer-v1"
TRAINING_COMPLETE_SCHEMA_VERSION = "etf-v3a-stage-d-transformer-complete-v1"
DEFAULT_PROTOCOL = ROOT / "configs/v3a_stage_d_gpu_pilot.json"
DEFAULT_OUT_DIR = ROOT / "data/processed/v3a/training/runs"
DEFAULT_TRAIN_VIEW = ROOT / "data/processed/v3a/stage_d/train_view"
DEFAULT_STAGE_C_REPORT = (
    ROOT / "data/processed/v3a/stage_c_reports/v3a-stage-c-20260711-01.json"
)
DEFAULT_BINDING = ROOT / "data/processed/v3a/stage_d/pilot_binding.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-file", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--train-view-dir", type=Path, default=DEFAULT_TRAIN_VIEW)
    parser.add_argument("--stage-c-report", type=Path, default=DEFAULT_STAGE_C_REPORT)
    parser.add_argument("--binding-file", type=Path, default=DEFAULT_BINDING)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--stop-after",
        type=int,
        help="Checkpoint after this cumulative attempt count; used for GPU resume validation.",
    )
    return parser.parse_args()


def _resolved_run_until(
    *, attempts: int, batch_size: int, stop_after: int | None
) -> int:
    run_until = attempts if stop_after is None else int(stop_after)
    if not 1 <= run_until <= attempts:
        raise ValueError("stop-after must be positive and no larger than protocol attempts")
    if run_until != attempts and run_until % batch_size != 0:
        raise ValueError("stop-after must be an exact training batch boundary")
    return run_until


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _model_config(protocol: dict[str, Any], policy_vocab: PolicyVocab) -> TransformerPolicyConfig:
    model = protocol["model"]
    return TransformerPolicyConfig(
        model_vocab_size=policy_vocab.size,
        max_sequence_len=SamplingConfig().max_len + 1,
        d_model=int(model["d_model"]),
        num_layers=int(model["num_layers"]),
        num_heads=int(model["num_heads"]),
        ff_dim=int(model["ff_dim"]),
        dropout=float(model["dropout"]),
        use_critic_head=bool(model["use_critic_head"]),
    )


def _move_optimizer_state(
    optimizer: torch.optim.Optimizer, device: torch.device
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _selection_hashes(
    selected_indices: torch.Tensor, top_k: np.ndarray
) -> list[bytes]:
    selected = selected_indices.detach().cpu().numpy()
    normalized = np.full_like(selected, -1)
    for day, count in enumerate(top_k):
        normalized[:, day, : int(count)] = selected[:, day, : int(count)]
    return [
        hashlib.sha256(row.astype(np.int16, copy=False).tobytes()).digest()
        for row in normalized
    ]


def _ledger_row_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _ledger_digest_at(path: Path, offset: int) -> Any:
    digest = hashlib.sha256()
    consumed = 0
    with path.open("rb") as handle:
        while consumed < offset:
            chunk = handle.read(min(1024 * 1024, offset - consumed))
            if not chunk:
                raise RuntimeError("V3A Stage D ledger ended before its checkpoint offset")
            digest.update(chunk)
            consumed += len(chunk)
    return digest


def _checkpoint(
    *,
    path: Path,
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
    ledger: Any,
    ledger_digest: Any,
    ledger_path: Path,
) -> None:
    ledger.flush()
    os.fsync(ledger.fileno())
    offset = ledger.tell()
    candidate_state["attempt_count"] = attempt_count
    candidate_state["attempt_ledger_offset"] = offset
    candidate_state["attempt_ledger_line_count"] = attempt_count
    candidate_state["attempt_ledger_prefix_sha256"] = ledger_digest.hexdigest()
    validate_training_candidate_state(
        candidate_state,
        attempt_count=attempt_count,
        research_spec=research_spec,
    )
    checkpoint = build_checkpoint(
        run_id=run_id,
        step=step,
        attempt_count=attempt_count,
        model=model,
        optimizer=optimizer,
        model_config=model_config,
        train_config=train_config,
        scorer_config=scorer_config,
        candidate_state=candidate_state,
        research_spec=research_spec,
    )
    atomic_save(checkpoint, path)


def _training_marker(
    *, run_dir: Path, summary: dict[str, Any], checkpoint_path: Path
) -> dict[str, Any]:
    summary_path = run_dir / "training_summary.json"
    log_path = run_dir / "training_log.jsonl"
    return {
        "schema_version": TRAINING_COMPLETE_SCHEMA_VERSION,
        "run_id": summary["run_id"],
        "protocol_id": summary["protocol_id"],
        "binding_id": summary["binding_id"],
        "research_spec_id": summary["research_spec_id"],
        "attempt_count": summary["attempt_count"],
        "formula_sequence_digest": summary["formula_sequence_digest"],
        "attempt_ledger_prefix_sha256": summary["attempt_ledger_prefix_sha256"],
        "summary_sha256": sha256_file(summary_path),
        "training_log_sha256": sha256_file(log_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }


def _recover_completed(
    *,
    run_dir: Path,
    checkpoint_path: Path,
    candidate_state: dict[str, Any],
    run_config: dict[str, Any],
    research_spec: dict[str, Any],
    model_config: dict[str, Any],
    train_config: dict[str, Any],
    scorer_config: dict[str, Any],
    latest_checkpoint: dict[str, Any],
) -> dict[str, Any] | None:
    summary_path = run_dir / "training_summary.json"
    marker_path = run_dir / "training_complete.json"
    log_path = run_dir / "training_log.jsonl"
    if not summary_path.exists() and not marker_path.exists():
        return None
    if not summary_path.exists() or not log_path.exists() or not checkpoint_path.exists():
        raise RuntimeError("V3A Stage D completed training artifacts are incomplete")
    final_checkpoint = load_checkpoint(
        checkpoint_path,
        research_spec=research_spec,
        run_id=str(run_config["run_id"]),
        model_config=model_config,
        train_config=train_config,
        scorer_config=scorer_config,
    )
    if int(final_checkpoint["attempt_count"]) != int(run_config["attempts"]):
        raise RuntimeError("V3A Stage D final checkpoint attempt count mismatch")
    assert_checkpoint_state_equal(
        latest_checkpoint,
        final_checkpoint,
        label="Stage D latest/final checkpoint",
    )
    final_candidate_state = final_checkpoint["candidate_state"]
    validate_training_candidate_state(
        final_candidate_state,
        attempt_count=int(run_config["attempts"]),
        research_spec=research_spec,
        deep=True,
    )
    if final_candidate_state != candidate_state:
        raise RuntimeError("V3A Stage D final and latest candidate states differ")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("run_id") != run_config["run_id"]
        or summary.get("protocol_id") != run_config["protocol_id"]
        or summary.get("binding_id") != run_config["binding_id"]
        or summary.get("research_spec_id") != research_spec["research_spec_id"]
        or int(summary.get("attempt_count", -1)) != int(run_config["attempts"])
        or summary.get("formula_sequence_digest")
        != candidate_state["formula_sequence_digest"]
        or summary.get("attempt_ledger_prefix_sha256")
        != candidate_state["attempt_ledger_prefix_sha256"]
    ):
        raise RuntimeError("V3A Stage D completed training identity mismatch")
    marker = _training_marker(
        run_dir=run_dir, summary=summary, checkpoint_path=checkpoint_path
    )
    _write_json_atomic(marker_path, marker)
    return summary


def main() -> None:
    args = parse_args()
    protocol = load_stage_d_protocol(args.protocol_file)
    run_config = method_run_config(
        protocol, method="transformer", seed=args.seed
    )
    verify_stage_c_prerequisite(protocol, args.stage_c_report)
    require_clean_v3a_code(extra_paths=(args.protocol_file,))
    device = require_stage_d_cuda(protocol)
    manifest = load_train_view_source_manifest(args.train_view_dir)
    scorer_config = ScorerConfig()
    candidate_config = CandidateConfig()
    research_spec = build_runtime_research_spec(
        manifest,
        scorer_config=scorer_config,
        candidate_config=candidate_config,
    )
    train_view = load_stage_d_train_view(
        args.train_view_dir,
        research_spec=research_spec,
    )
    binding = load_stage_d_binding(
        args.binding_file,
        protocol=protocol,
        research_spec=research_spec,
        train_view_manifest=train_view.manifest,
        stage_c_report_sha256=sha256_file(args.stage_c_report),
    )

    run_id = args.run_id or (
        f"v3a-stage-d-{protocol['mode']}-transformer-s{args.seed}-"
        f"{protocol['protocol_id'][:12]}"
    )
    run_config = {
        **run_config,
        "run_id": run_id,
        "protocol": protocol,
        "binding_id": binding["binding_id"],
        "train_view_id": train_view.manifest["train_view_id"],
        "train_view_fingerprint": train_view.manifest["train_view_fingerprint"],
    }
    attempts = int(run_config["attempts"])
    batch_size = int(run_config["batch_size"])
    run_until = _resolved_run_until(
        attempts=attempts,
        batch_size=batch_size,
        stop_after=args.stop_after,
    )

    run_dir = args.out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_handle = (run_dir / ".run.lock").open("a+")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"V3A Stage D run is already active: {run_id}") from exc
    existing = [path for path in run_dir.iterdir() if path.name != ".run.lock"]
    if not args.resume and existing:
        raise FileExistsError(f"V3A Stage D run already exists: {run_dir}")

    policy_vocab = PolicyVocab()
    sampling_config = SamplingConfig()
    model_config_object = _model_config(protocol, policy_vocab)
    model_config = model_config_object.to_dict()
    train_config = {
        "schema_version": TRAINING_SCHEMA_VERSION,
        "run_config": run_config,
        "sampling_config": sampling_config.to_dict(),
        "candidate_config": candidate_config.to_dict(),
    }
    optimizer_config = protocol["optimizer"]
    _seed_everything(args.seed)
    model = TransformerFormulaPolicy(model_config_object).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_config["learning_rate"]),
        weight_decay=float(optimizer_config["weight_decay"]),
    )
    checkpoint_path = run_dir / "checkpoint_latest.pt"
    final_checkpoint_path = run_dir / "checkpoint_final.pt"
    ledger_path = run_dir / "attempts.jsonl"

    if args.resume:
        if not checkpoint_path.exists() or not ledger_path.exists():
            raise FileNotFoundError("V3A Stage D resume files are incomplete")
        checkpoint = load_checkpoint(
            checkpoint_path,
            research_spec=research_spec,
            run_id=run_id,
            model_config=model_config,
            train_config=train_config,
            scorer_config=scorer_config.to_dict(),
        )
        restored = restore_training_state(
            checkpoint,
            model=model,
            optimizer=optimizer,
            research_spec=research_spec,
            run_id=run_id,
            model_config=model_config,
            train_config=train_config,
            scorer_config=scorer_config.to_dict(),
        )
        _move_optimizer_state(optimizer, device)
        candidate_state = restored["candidate_state"]
        completed_attempts = int(restored["attempt_count"])
        step = int(restored["step"])
        validate_training_candidate_state(
            candidate_state,
            attempt_count=completed_attempts,
            research_spec=research_spec,
            deep=True,
        )
        _validate_ledger_prefix(
            ledger_path,
            offset=int(candidate_state["attempt_ledger_offset"]),
            expected_lines=completed_attempts,
            expected_sha256=str(candidate_state["attempt_ledger_prefix_sha256"]),
        )
        with ledger_path.open("r+b") as handle:
            handle.truncate(int(candidate_state["attempt_ledger_offset"]))
            handle.flush()
            os.fsync(handle.fileno())
        if completed_attempts < attempts:
            candidate_state["resume_count"] = int(candidate_state["resume_count"]) + 1
        ledger = ledger_path.open("ab")
        ledger_digest = _ledger_digest_at(
            ledger_path, int(candidate_state["attempt_ledger_offset"])
        )
        latest_checkpoint = checkpoint
    else:
        created_at = datetime.now(timezone.utc).isoformat()
        candidate_state = empty_training_candidate_state(created_at=created_at)
        completed_attempts = 0
        step = 0
        ledger = ledger_path.open("xb")
        latest_checkpoint = None
        ledger_digest = hashlib.sha256()
        ledger.flush()
        os.fsync(ledger.fileno())
        validate_training_candidate_state(
            candidate_state, attempt_count=0, research_spec=research_spec
        )
        _checkpoint(
            path=checkpoint_path,
            run_id=run_id,
            step=0,
            attempt_count=0,
            model=model,
            optimizer=optimizer,
            model_config=model_config,
            train_config=train_config,
            scorer_config=scorer_config.to_dict(),
            candidate_state=candidate_state,
            research_spec=research_spec,
            ledger=ledger,
            ledger_digest=ledger_digest,
            ledger_path=ledger_path,
        )

    if completed_attempts > attempts:
        ledger.close()
        raise RuntimeError("V3A Stage D checkpoint exceeds protocol attempts")
    needs_finalization = completed_attempts == attempts and args.stop_after is None
    if needs_finalization:
        if latest_checkpoint is None:
            raise RuntimeError("V3A Stage D completed recovery lacks a latest checkpoint")
        recovered = _recover_completed(
            run_dir=run_dir,
            checkpoint_path=final_checkpoint_path,
            candidate_state=candidate_state,
            run_config=run_config,
            research_spec=research_spec,
            model_config=model_config,
            train_config=train_config,
            scorer_config=scorer_config.to_dict(),
            latest_checkpoint=latest_checkpoint,
        )
        if recovered is not None:
            ledger.close()
            print(json.dumps(recovered, ensure_ascii=False, indent=2))
            print(f"output: {run_dir}")
            return
    if not needs_finalization and not completed_attempts < run_until:
        ledger.close()
        raise ValueError("stop-after must be above the checkpoint attempt count")

    dates = train_view.dates
    mask_numpy = train_view.tradable_mask
    factor_numpy = train_view.factor_values
    targets = build_forward_targets(
        train_view.absolute_open,
        mask_numpy,
        dates,
        SplitSpec("train", "2016-08-09", "2021-12-31"),
        scorer_config,
    )
    factors = torch.as_tensor(factor_numpy, dtype=torch.float32, device=device)
    mask = torch.as_tensor(mask_numpy, dtype=torch.bool, device=device)
    torch_targets = TorchForwardTargets.from_numpy(
        targets, device=device, dtype=torch.float32
    )
    vm = BatchTorchVM()
    torch.cuda.reset_peak_memory_stats(device)

    counts = Counter(
        {key: int(value) for key, value in candidate_state["status_counts"].items()}
    )
    invalid_reasons = Counter(
        {
            str(key): int(value)
            for key, value in candidate_state["invalid_reason_counts"].items()
        }
    )
    length_counts = Counter(
        {
            int(key): int(value)
            for key, value in candidate_state["length_counts"].items()
        }
    )
    canonical_attempt_counts: dict[bytes, int] = candidate_state[
        "canonical_attempt_counts"
    ]
    selection_attempt_counts: dict[bytes, int] = candidate_state[
        "selection_attempt_counts"
    ]
    token_counts = Counter(
        {int(key): int(value) for key, value in candidate_state["token_counts"].items()}
    )
    rewards = [float(value) for value in candidate_state["semantic_valid_rewards"]]
    elapsed_before = float(candidate_state["elapsed_seconds"])
    session_started = time.monotonic()
    last_checkpoint_at = session_started
    checkpoint_config = protocol["checkpoint"]
    reinforce_config = protocol["reinforce"]

    try:
        while completed_attempts < run_until:
            step += 1
            batch_start = completed_attempts
            batch_count = min(batch_size, run_until - batch_start)
            model.train()
            sample = sample_formulas(
                model, batch_count, policy_vocab, sampling_config, device
            )
            compiled = [compile_formula(tokens) for tokens in sample.formulas]
            codes, lengths = compiled_to_tensor(compiled, device=device)
            vm_result = vm.execute(codes, lengths, factors, mask)
            scored = score_signal_batch(
                vm_result.signal, vm_result.valid, torch_targets, scorer_config
            )
            quality_valid, coverage, std, variation_valid = _quality_metrics(
                vm_result.signal, torch_targets, candidate_config
            )
            training_rewards = effective_training_rewards(
                scored.reward,
                scorer_valid=scored.valid,
                quality_valid=quality_valid,
                hard_invalid_reward=scorer_config.hard_invalid_reward,
            )
            objective = reinforce_objective(
                log_prob_sums=sample.log_prob_sums,
                entropy_sums=sample.entropy_sums,
                decision_counts=sample.formula_lengths.to(device) + 1,
                rewards=training_rewards,
                advantage_epsilon=float(reinforce_config["advantage_epsilon"]),
                entropy_coefficient=float(reinforce_config["entropy_coefficient"]),
            )
            optimizer.zero_grad(set_to_none=True)
            objective.loss.backward()
            gradients_finite = all(
                parameter.grad is None
                or bool(torch.isfinite(parameter.grad).all().item())
                for parameter in model.parameters()
            )
            if not gradients_finite:
                raise RuntimeError("V3A Stage D produced non-finite gradients")
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(optimizer_config["gradient_clip_norm"])
            )
            if not bool(torch.isfinite(gradient_norm).item()):
                raise RuntimeError("V3A Stage D gradient norm is non-finite")
            optimizer.step()
            if not all(
                bool(torch.isfinite(parameter).all().item())
                for parameter in model.parameters()
            ):
                raise RuntimeError("V3A Stage D produced non-finite model parameters")

            vm_valid_cpu = vm_result.valid.detach().cpu().numpy()
            score_valid_cpu = scored.valid.detach().cpu().numpy()
            quality_valid_cpu = quality_valid.detach().cpu().numpy()
            variation_valid_cpu = variation_valid.detach().cpu().numpy()
            coverage_cpu = coverage.detach().cpu().numpy()
            std_cpu = std.detach().cpu().numpy()
            reward_cpu = scored.reward.detach().cpu().numpy()
            selection_hashes = _selection_hashes(scored.selected_indices, targets.top_k)

            for row, (tokens, formula) in enumerate(
                zip(sample.formulas, compiled, strict=True)
            ):
                attempt_index = batch_start + row
                token_len = len(tokens)
                length_counts[token_len] += 1
                token_counts.update(tokens)
                candidate_state["formula_sequence_digest"] = sequence_digest(
                    candidate_state["formula_sequence_digest"], tokens
                )
                canonical_hash = expression_hash(
                    canonicalize_expression(formula.expression)
                )
                canonical_key = bytes.fromhex(canonical_hash)
                canonical_duplicate = canonical_key in canonical_attempt_counts
                if canonical_duplicate:
                    candidate_state["structural_duplicate_count"] = int(
                        candidate_state["structural_duplicate_count"]
                    ) + 1
                canonical_attempt_counts[canonical_key] = (
                    int(canonical_attempt_counts.get(canonical_key, 0)) + 1
                )
                selection_hash: str | None = None
                reward: float | None = None
                invalid_reason = ""
                if not bool(vm_valid_cpu[row]):
                    status = "vm_invalid"
                    invalid_reason = "vm_invalid"
                elif not bool(score_valid_cpu[row]):
                    status = "insufficient_daily_signal"
                    invalid_reason = "insufficient_daily_signal"
                elif not bool(quality_valid_cpu[row]):
                    if not bool(variation_valid_cpu[row]):
                        status = "constant_signal"
                        invalid_reason = "constant_signal"
                    else:
                        status = "low_coverage"
                        invalid_reason = "low_coverage"
                else:
                    reward = float(reward_cpu[row])
                    rewards.append(reward)
                    selection_key = selection_hashes[row]
                    selection_hash = selection_key.hex()
                    selection_duplicate = selection_key in selection_attempt_counts
                    if selection_duplicate:
                        candidate_state["selection_duplicate_observations"] = int(
                            candidate_state["selection_duplicate_observations"]
                        ) + 1
                    selection_attempt_counts[selection_key] = (
                        int(selection_attempt_counts.get(selection_key, 0)) + 1
                    )
                    record = build_candidate_record(
                        formula_id=f"transformer_s{args.seed}_a{attempt_index}",
                        source="v3a_stage_d_transformer",
                        token_ids=tokens,
                        reward=reward,
                        train_summary={
                            "scorer_days": targets.days,
                            "coverage": float(coverage_cpu[row]),
                            "finite_std": float(std_cpu[row]),
                            "training_step": step,
                        },
                        attempt_index=attempt_index,
                    )
                    if record.formula_hash != canonical_hash:
                        raise RuntimeError("V3A Stage D canonical hash drift")
                    retain_candidate(candidate_state, record, candidate_config)
                    if canonical_duplicate:
                        status = "canonical_duplicate"
                    elif selection_duplicate:
                        status = "selection_duplicate"
                    else:
                        status = "accepted_unique"
                counts[status] += 1
                if invalid_reason:
                    invalid_reasons[invalid_reason] += 1
                encoded_row = _ledger_row_bytes(
                    {
                        "attempt_index": attempt_index,
                        "training_step": step,
                        "sample_index": row,
                        "token_ids": tokens,
                        "token_names": FORMULA_VOCAB.decode(tokens),
                        "token_len": token_len,
                        "canonical_hash": canonical_hash,
                        "selection_hash": selection_hash,
                        "status": status,
                        "invalid_reason": invalid_reason,
                        "reward": reward,
                        "training_reward": float(
                            training_rewards[row].detach().cpu().item()
                        ),
                    }
                )
                ledger.write(encoded_row)
                ledger_digest.update(encoded_row)

            completed_attempts = batch_start + batch_count
            if rewards:
                candidate_state["best_semantic_reward"] = max(
                    float(candidate_state["best_semantic_reward"]), max(rewards)
                )
            candidate_state["best_reward_attempt_auc_numerator"] = float(
                candidate_state["best_reward_attempt_auc_numerator"]
            ) + float(candidate_state["best_semantic_reward"]) * batch_count
            elapsed = elapsed_before + time.monotonic() - session_started
            candidate_state.update(
                {
                    "attempt_count": completed_attempts,
                    "status_counts": {key: int(counts[key]) for key in STATUS_KEYS},
                    "invalid_reason_counts": dict(invalid_reasons),
                    "length_counts": {
                        str(key): int(value) for key, value in length_counts.items()
                    },
                    "canonical_attempt_counts": canonical_attempt_counts,
                    "selection_attempt_counts": selection_attempt_counts,
                    "semantic_valid_rewards": rewards,
                    "token_counts": {
                        str(key): int(value) for key, value in token_counts.items()
                    },
                    "elapsed_seconds": elapsed,
                }
            )
            candidate_state["training_log"].append(
                {
                    "step": step,
                    "attempt_count": completed_attempts,
                    "batch_count": batch_count,
                    "loss": float(objective.loss.detach().cpu().item()),
                    "reinforce_loss": float(
                        objective.reinforce_loss.detach().cpu().item()
                    ),
                    "reward_mean": float(objective.reward_mean.detach().cpu().item()),
                    "reward_std": float(objective.reward_std.detach().cpu().item()),
                    "valid_rate": float(
                        (scored.valid & quality_valid).float().mean().detach().cpu().item()
                    ),
                    "entropy": float(objective.entropy_mean.detach().cpu().item()),
                    "normalized_entropy": float(
                        (
                            sample.normalized_entropy_sums
                            / (sample.formula_lengths.to(device) + 1)
                        )
                        .mean()
                        .detach()
                        .cpu()
                        .item()
                    ),
                    "average_formula_length": float(
                        sample.formula_lengths.float().mean().item()
                    ),
                    "eos_before_limit_rate": float(
                        (sample.formula_lengths < sampling_config.max_len)
                        .float()
                        .mean()
                        .item()
                    ),
                    "average_allowed_actions": sample.avg_allowed_actions,
                    "gradient_norm": float(gradient_norm.detach().cpu().item()),
                    "retained_candidate_count": len(
                        candidate_state["canonical_state"]
                    ),
                    "retained_bucket_counts": [
                        len(bucket) for bucket in candidate_state["bucket_state"]
                    ],
                    "canonical_unique_rate": len(canonical_attempt_counts)
                    / completed_attempts,
                    "max_canonical_share": max(canonical_attempt_counts.values())
                    / completed_attempts,
                    "selection_unique_rate": len(selection_attempt_counts)
                    / len(rewards)
                    if rewards
                    else 0.0,
                    "max_selection_share": max(selection_attempt_counts.values())
                    / len(rewards)
                    if rewards
                    else 0.0,
                    "best_semantic_reward": candidate_state[
                        "best_semantic_reward"
                    ],
                    "best_reward_attempt_auc": float(
                        candidate_state["best_reward_attempt_auc_numerator"]
                    )
                    / completed_attempts,
                    "elapsed_seconds": elapsed,
                    "attempts_per_second": completed_attempts / elapsed
                    if elapsed > 0
                    else None,
                }
            )

            now = time.monotonic()
            should_checkpoint = (
                step % int(checkpoint_config["every_steps"]) == 0
                or now - last_checkpoint_at >= int(checkpoint_config["max_seconds"])
                or completed_attempts == run_until
            )
            if should_checkpoint:
                _checkpoint(
                    path=checkpoint_path,
                    run_id=run_id,
                    step=step,
                    attempt_count=completed_attempts,
                    model=model,
                    optimizer=optimizer,
                    model_config=model_config,
                    train_config=train_config,
                    scorer_config=scorer_config.to_dict(),
                    candidate_state=candidate_state,
                    research_spec=research_spec,
                    ledger=ledger,
                    ledger_digest=ledger_digest,
                    ledger_path=ledger_path,
                )
                last_checkpoint_at = now
                print(
                    f"attempts={completed_attempts}/{attempts} "
                    f"valid={candidate_state['training_log'][-1]['valid_rate']:.1%} "
                    f"entropy={candidate_state['training_log'][-1]['entropy']:.3f}",
                    flush=True,
                )
    finally:
        ledger.close()

    if completed_attempts < attempts:
        print(
            json.dumps(
                {
                    "status": "checkpointed",
                    "run_id": run_id,
                    "protocol_id": protocol["protocol_id"],
                    "completed_attempts": completed_attempts,
                    "target_attempts": attempts,
                    "formula_sequence_digest": candidate_state[
                        "formula_sequence_digest"
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        print(f"output: {run_dir}")
        return

    validate_training_candidate_state(
        candidate_state,
        attempt_count=attempts,
        research_spec=research_spec,
        deep=True,
    )
    _validate_ledger_prefix(
        ledger_path,
        offset=int(candidate_state["attempt_ledger_offset"]),
        expected_lines=attempts,
        expected_sha256=str(candidate_state["attempt_ledger_prefix_sha256"]),
    )
    checkpoint = build_checkpoint(
        run_id=run_id,
        step=step,
        attempt_count=attempts,
        model=model,
        optimizer=optimizer,
        model_config=model_config,
        train_config=train_config,
        scorer_config=scorer_config.to_dict(),
        candidate_state=candidate_state,
        research_spec=research_spec,
    )
    atomic_save(checkpoint, final_checkpoint_path)
    training_log = list(candidate_state["training_log"])
    _write_jsonl_atomic(run_dir / "training_log.jsonl", training_log)
    elapsed = float(candidate_state["elapsed_seconds"])
    retained = retained_candidates(candidate_state)
    summary: dict[str, Any] = {
        "schema_version": TRAINING_SCHEMA_VERSION,
        "status": f"{protocol['mode']}_trained_awaiting_funnel",
        "created_at": candidate_state["artifact_created_at"],
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "protocol_id": protocol["protocol_id"],
        "binding_id": binding["binding_id"],
        "protocol_mode": protocol["mode"],
        "formal_budget_approved": protocol["formal_budget_approved"],
        "method": "transformer",
        "seed": args.seed,
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(device),
        "dataset_id": manifest["dataset_id"],
        "panel_sha256": manifest["panel_sha256"],
        "research_spec_id": research_spec["research_spec_id"],
        "code_commit": research_spec["code_commit"],
        "code_fingerprint": research_spec["code_fingerprint"],
        "split": protocol["split"],
        "train_view_id": train_view.manifest["train_view_id"],
        "train_view_fingerprint": train_view.manifest["train_view_fingerprint"],
        "attempt_count": attempts,
        "step_count": step,
        "batch_size": batch_size,
        "attempt_ledger_reconciled": True,
        "attempt_ledger_line_count": candidate_state[
            "attempt_ledger_line_count"
        ],
        "attempt_ledger_prefix_sha256": candidate_state[
            "attempt_ledger_prefix_sha256"
        ],
        "status_counts": {key: int(counts[key]) for key in STATUS_KEYS},
        "invalid_reason_counts": dict(sorted(invalid_reasons.items())),
        "grammar_invalid_rate": counts["grammar_invalid"] / attempts,
        "canonical_unique_count": len(canonical_attempt_counts),
        "structural_duplicate_count": candidate_state[
            "structural_duplicate_count"
        ],
        "selection_unique_count": len(selection_attempt_counts),
        "selection_duplicate_count": candidate_state[
            "selection_duplicate_observations"
        ],
        "semantic_valid_count": len(rewards),
        "accepted_unique_count": counts["accepted_unique"],
        "length_distribution": {
            str(key): int(value) for key, value in sorted(length_counts.items())
        },
        "token_distribution": {
            FORMULA_VOCAB.id_to_token(key).name: int(value)
            for key, value in sorted(token_counts.items())
        },
        "reward_distribution": _reward_summary(rewards),
        "retained_candidate_count": len(retained),
        "retained_bucket_counts": [
            len(candidate_state["bucket_state"][index]) for index in range(3)
        ],
        "best_retained_reward": retained[0].reward if retained else None,
        "best_reward_attempt_auc": float(
            candidate_state["best_reward_attempt_auc_numerator"]
        )
        / attempts,
        "formula_sequence_digest": candidate_state["formula_sequence_digest"],
        "resume_count": candidate_state["resume_count"],
        "elapsed_seconds": elapsed,
        "attempts_per_second": attempts / elapsed if elapsed > 0 else None,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "first_training_log": training_log[0] if training_log else None,
        "last_training_log": training_log[-1] if training_log else None,
        "checkpoint_latest": checkpoint_path.name,
        "checkpoint_final": final_checkpoint_path.name,
        "checkpoint_latest_bytes": checkpoint_path.stat().st_size,
        "checkpoint_final_bytes": final_checkpoint_path.stat().st_size,
        "full_candidate_funnel_applied": False,
        "candidate_conclusion_allowed": protocol["candidate_output"][
            "research_conclusion_allowed"
        ],
        "validation_or_final_metrics_read": False,
    }
    _write_json_atomic(run_dir / "training_summary.json", summary)
    marker = _training_marker(
        run_dir=run_dir,
        summary=summary,
        checkpoint_path=final_checkpoint_path,
    )
    _write_json_atomic(run_dir / "training_complete.json", marker)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"output: {run_dir}")


if __name__ == "__main__":
    main()