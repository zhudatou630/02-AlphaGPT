"""Frozen protocol, objective, and candidate-state helpers for V3A Stage D."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from alpha_etf.research_v3a.candidates import (
    CandidateConfig,
    CandidateRecord,
    candidate_record_from_dict,
)
from alpha_etf.research_v3a.checkpointing import (
    empty_candidate_state,
    validate_candidate_state,
)
from alpha_etf.research_v3a.factors import FACTOR_NAMES
from alpha_etf.research_v3a.spec import canonical_sha256


STAGE_D_PROTOCOL_SCHEMA_VERSION = "etf-v3a-stage-d-protocol-v1"
STAGE_D_BINDING_SCHEMA_VERSION = "etf-v3a-stage-d-binding-v1"
TRAINING_TRACKER_SCHEMA_VERSION = "etf-v3a-training-tracker-v1"
TRAIN_VIEW_SCHEMA_VERSION = "etf-v3a-stage-d-train-view-v1"
APPROVED_PILOT_PROTOCOL_IDS = {
    "078af5f6466bd9661443d26f5722a107f67e69298bb0e593c730e7f6f042cc8b"
}
APPROVED_FORMAL_PROTOCOL_IDS: set[str] = set()
STATUS_KEYS = (
    "grammar_invalid",
    "vm_invalid",
    "insufficient_daily_signal",
    "low_coverage",
    "constant_signal",
    "canonical_duplicate",
    "selection_duplicate",
    "accepted_unique",
)


@dataclass(frozen=True)
class StageDTrainView:
    factor_values: np.ndarray
    absolute_open: np.ndarray
    tradable_mask: np.ndarray
    symbols: np.ndarray
    dates: pd.DatetimeIndex
    manifest: dict[str, Any]


def train_view_fingerprint(payload: dict[str, Any]) -> str:
    identity = dict(payload)
    identity.pop("train_view_id", None)
    identity.pop("train_view_fingerprint", None)
    return canonical_sha256(identity)


def build_train_view_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema_version") != TRAIN_VIEW_SCHEMA_VERSION:
        raise ValueError("V3A Stage D train-view schema mismatch")
    fingerprint = train_view_fingerprint(payload)
    return {
        **payload,
        "train_view_fingerprint": fingerprint,
        "train_view_id": f"v3a-train-view-{fingerprint[:16]}",
    }


def stage_d_binding_id(payload: dict[str, Any]) -> str:
    identity = dict(payload)
    identity.pop("binding_id", None)
    return canonical_sha256(identity)


def build_stage_d_binding(
    *,
    protocol: dict[str, Any],
    research_spec: dict[str, Any],
    train_view_manifest: dict[str, Any],
    stage_c_report_sha256: str,
) -> dict[str, Any]:
    payload = {
        "schema_version": STAGE_D_BINDING_SCHEMA_VERSION,
        "protocol_id": protocol["protocol_id"],
        "protocol_mode": protocol["mode"],
        "formal_budget_approved": protocol["formal_budget_approved"],
        "stage_c_report_sha256": stage_c_report_sha256,
        "dataset_id": research_spec["dataset"]["dataset_id"],
        "panel_sha256": research_spec["dataset"]["panel_sha256"],
        "code_commit": research_spec["code_commit"],
        "code_fingerprint": research_spec["code_fingerprint"],
        "research_spec_id": research_spec["research_spec_id"],
        "train_view_id": train_view_manifest["train_view_id"],
        "train_view_fingerprint": train_view_manifest["train_view_fingerprint"],
        "split": protocol["split"],
    }
    return {**payload, "binding_id": stage_d_binding_id(payload)}


def validate_stage_d_binding(
    binding: dict[str, Any],
    *,
    protocol: dict[str, Any],
    research_spec: dict[str, Any],
    train_view_manifest: dict[str, Any],
    stage_c_report_sha256: str,
) -> None:
    expected = build_stage_d_binding(
        protocol=protocol,
        research_spec=research_spec,
        train_view_manifest=train_view_manifest,
        stage_c_report_sha256=stage_c_report_sha256,
    )
    if binding != expected:
        raise RuntimeError("V3A Stage D binding differs from the frozen runtime identity")


def load_stage_d_binding(
    path: Path,
    *,
    protocol: dict[str, Any],
    research_spec: dict[str, Any],
    train_view_manifest: dict[str, Any],
    stage_c_report_sha256: str,
) -> dict[str, Any]:
    binding = json.loads(path.read_text(encoding="utf-8"))
    validate_stage_d_binding(
        binding,
        protocol=protocol,
        research_spec=research_spec,
        train_view_manifest=train_view_manifest,
        stage_c_report_sha256=stage_c_report_sha256,
    )
    return binding


def load_train_view_source_manifest(path: Path) -> dict[str, Any]:
    manifest_path = path / "train_view_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing V3A Stage D train view: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = manifest.get("source_dataset_manifest")
    if not isinstance(source, dict):
        raise RuntimeError("V3A Stage D train view lacks its source dataset manifest")
    if (
        source.get("dataset_id") != manifest.get("source_dataset_id")
        or source.get("panel_sha256") != manifest.get("source_panel_sha256")
    ):
        raise RuntimeError("V3A Stage D embedded dataset identity mismatch")
    return source


def load_stage_d_train_view(
    path: Path,
    *,
    research_spec: dict[str, Any],
) -> StageDTrainView:
    manifest_path = path / "train_view_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing V3A Stage D train view: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != TRAIN_VIEW_SCHEMA_VERSION:
        raise RuntimeError("V3A Stage D train-view schema mismatch")
    dataset_manifest = load_train_view_source_manifest(path)
    expected_fingerprint = train_view_fingerprint(manifest)
    if (
        manifest.get("train_view_fingerprint") != expected_fingerprint
        or manifest.get("train_view_id")
        != f"v3a-train-view-{expected_fingerprint[:16]}"
    ):
        raise RuntimeError("V3A Stage D train-view identity mismatch")
    if (
        manifest.get("source_dataset_id") != dataset_manifest.get("dataset_id")
        or manifest.get("source_panel_sha256") != dataset_manifest.get("panel_sha256")
        or manifest.get("research_spec_id") != research_spec.get("research_spec_id")
        or manifest.get("code_fingerprint") != research_spec.get("code_fingerprint")
    ):
        raise RuntimeError("V3A Stage D train view differs from the frozen research identity")
    if manifest.get("split") != {
        "signal_start": "2016-08-09",
        "data_end": "2021-12-31",
        "validation_columns_present": False,
        "final_columns_present": False,
    }:
        raise RuntimeError("V3A Stage D train-view split mismatch")

    required_files = {
        "factor_values": "factor_values.npy",
        "absolute_open": "absolute_open.npy",
        "tradable_mask": "tradable_mask.npy",
        "symbols": "symbols.npy",
        "dates": "dates.npy",
    }
    arrays: dict[str, np.ndarray] = {}
    file_manifest = manifest.get("files", {})
    for key, filename in required_files.items():
        entry = file_manifest.get(key, {})
        if entry.get("path") != filename:
            raise RuntimeError(f"V3A Stage D train-view file mapping drifted: {key}")
        file_path = path / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Missing V3A Stage D train-view array: {file_path}")
        digest = hashlib.sha256(file_path.read_bytes()).hexdigest()
        if digest != entry.get("sha256"):
            raise RuntimeError(f"V3A Stage D train-view SHA mismatch: {key}")
        arrays[key] = np.load(file_path, allow_pickle=False)

    factor_values = arrays["factor_values"]
    absolute_open = arrays["absolute_open"]
    tradable_mask = arrays["tradable_mask"]
    symbols = arrays["symbols"].astype(str)
    date_values = arrays["dates"].astype(str)
    if (
        factor_values.dtype != np.float64
        or absolute_open.dtype != np.float64
        or tradable_mask.dtype != np.bool_
        or factor_values.shape[1:] != tradable_mask.shape
        or absolute_open.shape != tradable_mask.shape
        or tradable_mask.shape != (len(symbols), len(date_values))
    ):
        raise RuntimeError("V3A Stage D train-view array contract mismatch")
    if list(factor_values.shape) != manifest.get("factor_shape") or list(
        tradable_mask.shape
    ) != manifest.get("mask_shape"):
        raise RuntimeError("V3A Stage D train-view shape differs from its manifest")
    if (
        manifest.get("factor_names") != list(FACTOR_NAMES)
        or factor_values.shape[0] != len(FACTOR_NAMES)
    ):
        raise RuntimeError("V3A Stage D train-view factor inventory mismatch")
    if list(symbols) != list(dataset_manifest.get("symbols", [])):
        raise RuntimeError("V3A Stage D train-view symbols differ from the dataset")
    if (
        not len(date_values)
        or manifest.get("date_start") != date_values[0]
        or manifest.get("date_end") != date_values[-1]
        or date_values[-1] != "2021-12-31"
        or any(
        value > "2021-12-31" for value in date_values
        )
    ):
        raise RuntimeError("V3A Stage D train view exposes future dates")
    date_index = pd.DatetimeIndex(pd.to_datetime(date_values))
    if not date_index.is_monotonic_increasing or not date_index.is_unique:
        raise RuntimeError("V3A Stage D train-view dates are not strictly ordered")
    if np.isnan(absolute_open[tradable_mask]).any() or not (
        absolute_open[tradable_mask] > 0
    ).all():
        raise RuntimeError("V3A Stage D train-view tradable opens are invalid")
    return StageDTrainView(
        factor_values=factor_values,
        absolute_open=absolute_open,
        tradable_mask=tradable_mask,
        symbols=symbols,
        dates=date_index,
        manifest=manifest,
    )


def protocol_payload(protocol: dict[str, Any]) -> dict[str, Any]:
    payload = dict(protocol)
    payload.pop("protocol_id", None)
    return payload


def stage_d_protocol_id(protocol: dict[str, Any]) -> str:
    return canonical_sha256(protocol_payload(protocol))


def validate_stage_d_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("schema_version") != STAGE_D_PROTOCOL_SCHEMA_VERSION:
        raise RuntimeError("V3A Stage D protocol schema mismatch")
    actual_id = stage_d_protocol_id(protocol)
    if protocol.get("protocol_id") != actual_id:
        raise RuntimeError("V3A Stage D protocol id mismatch")
    mode = protocol.get("mode")
    if mode == "pilot":
        if actual_id not in APPROVED_PILOT_PROTOCOL_IDS:
            raise RuntimeError("V3A Stage D pilot protocol is not approved by code")
        if protocol.get("formal_budget_approved") is not False:
            raise RuntimeError("V3A Stage D pilot cannot approve a formal budget")
    elif mode == "formal":
        if actual_id not in APPROVED_FORMAL_PROTOCOL_IDS:
            raise RuntimeError("V3A Stage D formal protocol has not been approved")
        if protocol.get("formal_budget_approved") is not True:
            raise RuntimeError("V3A Stage D formal budget is not approved")
    else:
        raise RuntimeError(f"Unsupported V3A Stage D protocol mode: {mode!r}")

    execution = protocol.get("execution", {})
    if execution != {
        "device": "cuda",
        "gpu_class": "rtx_4090d",
        "local_research_training_forbidden": True,
    }:
        raise RuntimeError("V3A Stage D protocol must require the approved CUDA host class")
    if protocol.get("split") != {
        "name": "train",
        "start": "2016-08-09",
        "end": "2021-12-31",
        "validation_metrics_read": False,
        "final_metrics_read": False,
    }:
        raise RuntimeError("V3A Stage D protocol split mismatch")
    if int(protocol.get("batch_size", -1)) != 256:
        raise RuntimeError("V3A Stage D batch size must remain 256")
    if protocol.get("prerequisite") != {
        "stage_c_run_id": "v3a-stage-c-20260711-01",
        "stage_c_report_sha256": "3d742928cbae05c30acb8fef01eac27dfb458f058ca216ec3b2a8c48bc2947a4",
        "dataset_id": "etf-v3a-dbdc6d0b7741",
        "panel_sha256": "fce56f8b3aa5bfc13f7848c47eff5d1f3b449a379c048e25262c7f560494a85f",
    }:
        raise RuntimeError("V3A Stage D prerequisite evidence mismatch")

    runs = protocol.get("runs", {})
    if set(runs) != {"transformer", "matched_random"}:
        raise RuntimeError("V3A Stage D protocol method inventory mismatch")
    for method, config in runs.items():
        seeds = config.get("seeds")
        attempts = config.get("attempts")
        if not isinstance(seeds, list) or len(seeds) != len(set(seeds)) or not all(
            isinstance(seed, int) and seed >= 0 for seed in seeds
        ):
            raise RuntimeError(f"V3A Stage D {method} seeds are invalid")
        if seeds and (not isinstance(attempts, int) or attempts < 1):
            raise RuntimeError(f"V3A Stage D {method} attempts are invalid")
        if not seeds and attempts is not None:
            raise RuntimeError(f"V3A Stage D {method} has attempts without seeds")
    if mode == "pilot":
        if runs != {
            "transformer": {"seeds": [314159], "attempts": 50_000},
            "matched_random": {"seeds": [], "attempts": None},
        }:
            raise RuntimeError("V3A Stage D pilot budget differs from the approved pilot")
    else:
        transformer = runs["transformer"]
        random_run = runs["matched_random"]
        if (
            len(transformer["seeds"]) < 2
            or len(transformer["seeds"]) != len(random_run["seeds"])
            or transformer["attempts"] != random_run["attempts"]
        ):
            raise RuntimeError("V3A Stage D formal methods must use matched seed counts and budgets")

    model = protocol.get("model", {})
    if model != {
        "d_model": 64,
        "num_layers": 2,
        "num_heads": 4,
        "ff_dim": 128,
        "dropout": 0.0,
        "use_critic_head": False,
    }:
        raise RuntimeError("V3A Stage D model config mismatch")
    optimizer = protocol.get("optimizer", {})
    if optimizer != {
        "name": "AdamW",
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "gradient_clip_norm": 1.0,
    }:
        raise RuntimeError("V3A Stage D optimizer config mismatch")
    reinforce = protocol.get("reinforce", {})
    if reinforce != {
        "advantage": "leave_one_out_batch_zscore",
        "advantage_epsilon": 1e-5,
        "entropy_coefficient": 1e-3,
        "entropy_normalization": "per_decision_including_eos",
        "quality_invalid_uses_hard_invalid_reward": True,
    }:
        raise RuntimeError("V3A Stage D REINFORCE config mismatch")
    checkpoint = protocol.get("checkpoint", {})
    if checkpoint != {"every_steps": 10, "max_seconds": 600}:
        raise RuntimeError("V3A Stage D checkpoint config mismatch")
    candidate = protocol.get("candidate_output", {})
    if (
        candidate.get("apply_full_funnel") is not True
        or int(candidate.get("required_selected_count", -1)) != 50
        or candidate.get("research_conclusion_allowed") is not (mode == "formal")
    ):
        raise RuntimeError("V3A Stage D candidate-output protocol mismatch")


def load_stage_d_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    validate_stage_d_protocol(protocol)
    return protocol


def verify_stage_c_prerequisite(
    protocol: dict[str, Any], report_path: Path
) -> None:
    validate_stage_d_protocol(protocol)
    prerequisite = protocol["prerequisite"]
    if not report_path.exists():
        raise FileNotFoundError(f"Missing V3A Stage C completion report: {report_path}")
    digest = hashlib.sha256(report_path.read_bytes()).hexdigest()
    if digest != prerequisite["stage_c_report_sha256"]:
        raise RuntimeError("V3A Stage C completion report SHA mismatch")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("status") != "passed"
        or report.get("run_id") != prerequisite["stage_c_run_id"]
        or report.get("frozen_identity", {}).get("dataset_id")
        != prerequisite["dataset_id"]
        or report.get("frozen_identity", {}).get("panel_sha256")
        != prerequisite["panel_sha256"]
        or report.get("stage_c_stop_required") is not True
        or report.get("transformer_started") is not False
        or len(report.get("baselines", [])) != 3
        or {int(item.get("seed", -1)) for item in report.get("baselines", [])}
        != {41, 42, 43}
    ):
        raise RuntimeError("V3A Stage C prerequisite did not pass its frozen gate")


def method_run_config(
    protocol: dict[str, Any], *, method: str, seed: int
) -> dict[str, Any]:
    validate_stage_d_protocol(protocol)
    if method not in {"transformer", "matched_random"}:
        raise RuntimeError(f"Unsupported V3A Stage D method: {method}")
    method_config = protocol["runs"][method]
    if seed not in method_config["seeds"]:
        raise RuntimeError(f"Seed {seed} is not registered for V3A Stage D {method}")
    return {
        "schema_version": "etf-v3a-stage-d-run-config-v1",
        "protocol_id": protocol["protocol_id"],
        "protocol_mode": protocol["mode"],
        "method": method,
        "seed": seed,
        "attempts": int(method_config["attempts"]),
        "batch_size": int(protocol["batch_size"]),
        "model": protocol["model"],
        "optimizer": protocol["optimizer"],
        "reinforce": protocol["reinforce"],
        "checkpoint": protocol["checkpoint"],
        "candidate_output": protocol["candidate_output"],
        "split": protocol["split"],
        "device_type": "cuda",
    }


@dataclass(frozen=True)
class ReinforceObjective:
    loss: torch.Tensor
    reinforce_loss: torch.Tensor
    entropy_mean: torch.Tensor
    reward_mean: torch.Tensor
    reward_std: torch.Tensor
    advantages: torch.Tensor


def effective_training_rewards(
    scored_rewards: torch.Tensor,
    *,
    scorer_valid: torch.Tensor,
    quality_valid: torch.Tensor,
    hard_invalid_reward: float,
) -> torch.Tensor:
    if (
        scored_rewards.ndim != 1
        or scorer_valid.shape != scored_rewards.shape
        or quality_valid.shape != scored_rewards.shape
    ):
        raise ValueError("V3A Stage D reward masks differ from the reward batch")
    valid = scorer_valid & quality_valid & torch.isfinite(scored_rewards)
    return torch.where(
        valid,
        scored_rewards,
        torch.full_like(scored_rewards, float(hard_invalid_reward)),
    )


def reinforce_objective(
    *,
    log_prob_sums: torch.Tensor,
    entropy_sums: torch.Tensor,
    decision_counts: torch.Tensor,
    rewards: torch.Tensor,
    advantage_epsilon: float,
    entropy_coefficient: float,
) -> ReinforceObjective:
    if (
        log_prob_sums.ndim != 1
        or entropy_sums.shape != log_prob_sums.shape
        or decision_counts.shape != log_prob_sums.shape
        or rewards.shape != log_prob_sums.shape
    ):
        raise ValueError("V3A Stage D objective batch shapes differ")
    if not bool(torch.isfinite(rewards).all().item()):
        raise RuntimeError("V3A Stage D objective received non-finite rewards")
    if len(rewards) < 2:
        raise ValueError("V3A Stage D leave-one-out objective requires batch size > 1")
    if not bool((decision_counts > 0).all().item()):
        raise ValueError("V3A Stage D decision counts must be positive")
    reward_mean = rewards.mean()
    reward_std = rewards.std(unbiased=False)
    leave_one_out = (rewards.sum() - rewards) / float(len(rewards) - 1)
    advantages = (rewards - leave_one_out) / (
        reward_std + float(advantage_epsilon)
    )
    reinforce_loss = -(log_prob_sums * advantages.detach()).mean()
    entropy_mean = (entropy_sums / decision_counts.to(entropy_sums.dtype)).mean()
    loss = reinforce_loss - float(entropy_coefficient) * entropy_mean
    if not bool(torch.isfinite(loss).item()):
        raise RuntimeError("V3A Stage D objective is non-finite")
    return ReinforceObjective(
        loss=loss,
        reinforce_loss=reinforce_loss,
        entropy_mean=entropy_mean,
        reward_mean=reward_mean,
        reward_std=reward_std,
        advantages=advantages,
    )


def _bucket(token_len: int) -> int:
    return 0 if token_len <= 5 else 1 if token_len <= 10 else 2


def _record_sort_key(record: CandidateRecord) -> tuple[float, int, str]:
    return (-record.reward, record.token_len, record.formula_hash)


def _ledger_sort_key(entry: dict[str, Any]) -> tuple[float, int, str]:
    record = entry["best_record"]
    return (
        -float(entry["best_reward"]),
        int(record["token_len"]),
        str(entry["formula_hash"]),
    )


def sequence_digest(previous: str, token_ids: list[int]) -> str:
    digest = hashlib.sha256()
    digest.update(bytes.fromhex(previous))
    digest.update(json.dumps(token_ids, separators=(",", ":")).encode("ascii"))
    return digest.hexdigest()


def empty_training_candidate_state(*, created_at: str) -> dict[str, Any]:
    return {
        **empty_candidate_state(attempt_count=0),
        "tracker_schema_version": TRAINING_TRACKER_SCHEMA_VERSION,
        "status_counts": {key: 0 for key in STATUS_KEYS},
        "invalid_reason_counts": {},
        "length_counts": {},
        "canonical_attempt_counts": {},
        "structural_duplicate_count": 0,
        "selection_attempt_counts": {},
        "selection_duplicate_observations": 0,
        "canonical_ledger": {},
        "canonical_ledger_attempt_count": 0,
        "semantic_valid_rewards": [],
        "formula_sequence_digest": "00" * 32,
        "attempt_ledger_offset": 0,
        "attempt_ledger_line_count": 0,
        "attempt_ledger_prefix_sha256": hashlib.sha256(b"").hexdigest(),
        "artifact_created_at": created_at,
        "elapsed_seconds": 0.0,
        "resume_count": 0,
        "training_log": [],
        "token_counts": {},
        "best_semantic_reward": -5.0,
        "best_reward_attempt_auc_numerator": 0.0,
    }


def retain_candidate(
    state: dict[str, Any], record: CandidateRecord, config: CandidateConfig
) -> None:
    formula_hash = record.formula_hash
    canonical = state["canonical_state"]
    ledger = state["canonical_ledger"]
    existing_ledger = ledger.get(formula_hash)
    previous: CandidateRecord | None = None
    if existing_ledger is not None:
        count = int(existing_ledger["attempt_count"]) + 1
        previous = candidate_record_from_dict(existing_ledger["best_record"])
        best = record if _record_sort_key(record) < _record_sort_key(previous) else previous
        best_record = replace(
            best,
            attempt_count=count,
            first_attempt_index=int(existing_ledger["first_attempt_index"]),
        )
        existing_ledger.update(
            {
                "attempt_count": count,
                "best_attempt_index": best_record.best_attempt_index,
                "best_reward": best_record.reward,
                "best_record": best_record.to_dict(),
            }
        )
    else:
        ledger[formula_hash] = {
            "formula_hash": formula_hash,
            "first_attempt_index": record.first_attempt_index,
            "attempt_count": 1,
            "best_attempt_index": record.best_attempt_index,
            "best_reward": record.reward,
            "best_record": record.to_dict(),
        }
    state["canonical_ledger_attempt_count"] = int(
        state["canonical_ledger_attempt_count"]
    ) + 1

    best_entry = ledger[formula_hash]
    best_record = candidate_record_from_dict(best_entry["best_record"])

    changed_buckets: set[int] = set()
    previous_bucket_members: dict[int, set[str]] = {}

    def mark_changed(bucket_index: int) -> None:
        if bucket_index not in previous_bucket_members:
            previous_bucket_members[bucket_index] = set(
                state["bucket_state"][bucket_index]
            )
        changed_buckets.add(bucket_index)

    def consider(bucket_index: int, candidate_hash: str) -> None:
        mark_changed(bucket_index)
        bucket = state["bucket_state"][bucket_index]
        if candidate_hash not in bucket:
            bucket.append(candidate_hash)
        bucket.sort(key=lambda value: _ledger_sort_key(ledger[value]))
        del bucket[config.heap_per_bucket :]

    def refill(bucket_index: int) -> None:
        bucket = state["bucket_state"][bucket_index]
        if len(bucket) >= config.heap_per_bucket:
            return
        mark_changed(bucket_index)
        candidates = sorted(
            (
                entry
                for entry in ledger.values()
                if _bucket(int(entry["best_record"]["token_len"])) == bucket_index
            ),
            key=_ledger_sort_key,
        )[: config.heap_per_bucket]
        state["bucket_state"][bucket_index] = [
            str(item["formula_hash"]) for item in candidates
        ]

    new_bucket = _bucket(best_record.token_len)
    if previous is None:
        consider(new_bucket, formula_hash)
    else:
        old_bucket = _bucket(previous.token_len)
        was_retained = formula_hash in state["bucket_state"][old_bucket]
        if old_bucket != new_bucket and was_retained:
            mark_changed(old_bucket)
            state["bucket_state"][old_bucket].remove(formula_hash)
            refill(old_bucket)
        if was_retained or _record_sort_key(best_record) < _record_sort_key(previous):
            consider(new_bucket, formula_hash)

    for bucket_index in changed_buckets:
        current = set(state["bucket_state"][bucket_index])
        previous_members = previous_bucket_members[bucket_index]
        for removed in previous_members - current:
            canonical.pop(removed, None)
        for added in current - previous_members:
            canonical[added] = dict(ledger[added])
    if any(formula_hash in bucket for bucket in state["bucket_state"]):
        canonical[formula_hash] = dict(ledger[formula_hash])


def retained_candidates(state: dict[str, Any]) -> list[CandidateRecord]:
    records: list[CandidateRecord] = []
    for bucket in state["bucket_state"]:
        records.extend(
            candidate_record_from_dict(state["canonical_state"][formula_hash]["best_record"])
            for formula_hash in bucket
        )
    return sorted(records, key=_record_sort_key)


def validate_training_candidate_state(
    state: dict[str, Any],
    *,
    attempt_count: int,
    research_spec: dict[str, Any],
    deep: bool = False,
) -> None:
    validate_candidate_state(
        state, attempt_count=attempt_count, research_spec=research_spec
    )
    if state.get("tracker_schema_version") != TRAINING_TRACKER_SCHEMA_VERSION:
        raise RuntimeError("V3A Stage D training tracker schema mismatch")
    counts = state.get("status_counts")
    if not isinstance(counts, dict) or set(counts) != set(STATUS_KEYS):
        raise RuntimeError("V3A Stage D status ledger mismatch")
    if sum(int(value) for value in counts.values()) != attempt_count:
        raise RuntimeError("V3A Stage D status ledger does not reconcile")
    if int(state.get("attempt_ledger_line_count", -1)) != attempt_count:
        raise RuntimeError("V3A Stage D attempt ledger line count mismatch")
    if sum(int(value) for value in state.get("length_counts", {}).values()) != attempt_count:
        raise RuntimeError("V3A Stage D formula-length ledger does not reconcile")
    canonical_counts = state.get("canonical_attempt_counts")
    selection_counts = state.get("selection_attempt_counts")
    if not isinstance(canonical_counts, dict) or not isinstance(selection_counts, dict):
        raise RuntimeError("V3A Stage D duplicate state is invalid")
    if (
        sum(int(value) for value in canonical_counts.values()) != attempt_count
        or len(canonical_counts) + int(state.get("structural_duplicate_count", -1))
        != attempt_count
    ):
        raise RuntimeError("V3A Stage D canonical duplicate ledger does not reconcile")
    semantic_count = sum(
        int(counts[key])
        for key in ("canonical_duplicate", "selection_duplicate", "accepted_unique")
    )
    if len(state.get("semantic_valid_rewards", [])) != semantic_count:
        raise RuntimeError("V3A Stage D semantic reward ledger does not reconcile")
    ledger = state.get("canonical_ledger")
    if not isinstance(ledger, dict):
        raise RuntimeError("V3A Stage D canonical ledger is invalid")
    if int(state.get("canonical_ledger_attempt_count", -1)) != semantic_count:
        raise RuntimeError("V3A Stage D canonical ledger count does not reconcile")
    if deep:
        ledger_attempts = 0
        for formula_hash, entry in ledger.items():
            if (
                not isinstance(entry, dict)
                or entry.get("formula_hash") != formula_hash
                or int(entry.get("attempt_count", 0)) < 1
            ):
                raise RuntimeError("V3A Stage D canonical ledger entry is invalid")
            record = candidate_record_from_dict(entry.get("best_record", {}))
            if (
                record.formula_hash != formula_hash
                or record.attempt_count != int(entry["attempt_count"])
                or record.first_attempt_index
                != int(entry["first_attempt_index"])
                or record.best_attempt_index != int(entry["best_attempt_index"])
                or record.reward != float(entry["best_reward"])
            ):
                raise RuntimeError("V3A Stage D canonical ledger record drifted")
            ledger_attempts += int(entry["attempt_count"])
        if ledger_attempts != semantic_count:
            raise RuntimeError("V3A Stage D deep canonical ledger does not reconcile")
    if not set(state["canonical_state"]).issubset(ledger):
        raise RuntimeError("V3A Stage D retained heap is not in the canonical ledger")
    for formula_hash, entry in state["canonical_state"].items():
        if entry != ledger[formula_hash]:
            raise RuntimeError("V3A Stage D retained canonical entry drifted")
    if (
        sum(int(value) for value in selection_counts.values()) != semantic_count
        or semantic_count - len(selection_counts) != int(
        state.get("selection_duplicate_observations", -1)
        )
    ):
        raise RuntimeError("V3A Stage D selection duplicate ledger does not reconcile")
    expected_token_count = sum(
        int(length) * int(count)
        for length, count in state.get("length_counts", {}).items()
    )
    if sum(int(value) for value in state.get("token_counts", {}).values()) != expected_token_count:
        raise RuntimeError("V3A Stage D token ledger does not reconcile")
    sha = str(state.get("attempt_ledger_prefix_sha256", ""))
    if len(sha) != 64 or any(character not in "0123456789abcdef" for character in sha):
        raise RuntimeError("V3A Stage D attempt ledger SHA is invalid")
    for bucket_index, bucket in enumerate(state["bucket_state"]):
        if len(bucket) > CandidateConfig().heap_per_bucket:
            raise RuntimeError("V3A Stage D candidate heap exceeds its frozen limit")
        records = [
            candidate_record_from_dict(state["canonical_state"][value]["best_record"])
            for value in bucket
        ]
        if any(_bucket(record.token_len) != bucket_index for record in records):
            raise RuntimeError("V3A Stage D candidate is in the wrong length bucket")
        if records != sorted(records, key=_record_sort_key):
            raise RuntimeError("V3A Stage D candidate heap ordering drifted")