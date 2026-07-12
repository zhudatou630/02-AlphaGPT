#!/usr/bin/env python3
"""Build the frozen Stage D training-only candidate funnel on CUDA."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from alpha_etf.research_v3a.artifacts import (
    build_formula_artifact,
    build_training_funnel_artifact,
    load_jsonl,
    validate_formula_artifact,
    validate_training_funnel_artifact,
)
from alpha_etf.research_v3a.candidates import (
    CandidateConfig,
    CandidateRecord,
    SimilarityContext,
    candidate_record_from_dict,
    select_training_candidates,
)
from alpha_etf.research_v3a.checkpointing import (
    atomic_save,
    validate_checkpoint,
)
from alpha_etf.research_v3a.scoring import ScorerConfig, SplitSpec, build_forward_targets
from alpha_etf.research_v3a.spec import sha256_file
from alpha_etf.research_v3a.stage_d import (
    load_stage_d_binding,
    load_stage_d_protocol,
    load_stage_d_train_view,
    load_train_view_source_manifest,
    method_run_config,
    retained_candidates,
    validate_training_candidate_state,
    verify_stage_c_prerequisite,
)
from alpha_etf.research_v3a.torch_scoring import TorchForwardTargets
from scripts.v3a.random_baseline import (
    _replay_candidate_signals,
    _reward_summary,
    _write_json_atomic,
    _write_jsonl_atomic,
)
from scripts.v3a.runtime import build_runtime_research_spec, require_clean_v3a_code
from scripts.v3a.train_gpu import (
    TRAINING_COMPLETE_SCHEMA_VERSION,
    _model_config,
    _training_marker,
)
from alpha_etf.research_v3a.sampling import PolicyVocab, SamplingConfig


FUNNEL_SUMMARY_SCHEMA_VERSION = "etf-v3a-stage-d-funnel-summary-v1"
FUNNEL_COMPLETE_SCHEMA_VERSION = "etf-v3a-stage-d-funnel-complete-v1"
DEFAULT_TRAIN_VIEW = ROOT / "data/processed/v3a/stage_d/train_view"
DEFAULT_STAGE_C_REPORT = (
    ROOT / "data/processed/v3a/stage_c_reports/v3a-stage-c-20260711-01.json"
)
DEFAULT_BINDING = ROOT / "data/processed/v3a/stage_d/pilot_binding.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-file", type=Path, required=True)
    parser.add_argument("--method", choices=("transformer",), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--train-view-dir", type=Path, default=DEFAULT_TRAIN_VIEW)
    parser.add_argument("--stage-c-report", type=Path, default=DEFAULT_STAGE_C_REPORT)
    parser.add_argument("--binding-file", type=Path, default=DEFAULT_BINDING)
    return parser.parse_args()


def _require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("V3A Stage D candidate replay is CUDA-only")
    return torch.device("cuda")


def _load_transformer_source(
    *,
    run_dir: Path,
    expected_run_config: dict[str, Any],
    protocol: dict[str, Any],
    research_spec: dict[str, Any],
    scorer_config: ScorerConfig,
) -> tuple[
    str,
    list[CandidateRecord],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    checkpoint_path = run_dir / "checkpoint_final.pt"
    summary_path = run_dir / "training_summary.json"
    marker_path = run_dir / "training_complete.json"
    if not checkpoint_path.exists() or not summary_path.exists() or not marker_path.exists():
        raise FileNotFoundError("V3A Stage D Transformer source is incomplete")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("schema_version") != TRAINING_COMPLETE_SCHEMA_VERSION:
        raise RuntimeError("V3A Stage D Transformer completion schema mismatch")
    expected_marker = _training_marker(
        run_dir=run_dir, summary=summary, checkpoint_path=checkpoint_path
    )
    if marker != expected_marker:
        raise RuntimeError("V3A Stage D Transformer completion marker mismatch")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    run_id = str(checkpoint.get("run_id", ""))
    expected_run_config = {
        **expected_run_config,
        "run_id": run_id,
        "protocol": protocol,
    }
    policy_vocab = PolicyVocab()
    expected_model_config = _model_config(protocol, policy_vocab).to_dict()
    expected_train_config = {
        "schema_version": "etf-v3a-stage-d-transformer-v1",
        "run_config": expected_run_config,
        "sampling_config": SamplingConfig().to_dict(),
        "candidate_config": CandidateConfig().to_dict(),
    }
    validate_checkpoint(
        checkpoint,
        research_spec=research_spec,
        run_id=run_id,
        model_config=expected_model_config,
        train_config=expected_train_config,
        scorer_config=scorer_config.to_dict(),
    )
    candidate_state = checkpoint["candidate_state"]
    attempts = int(expected_run_config["attempts"])
    validate_training_candidate_state(
        candidate_state,
        attempt_count=attempts,
        research_spec=research_spec,
        deep=True,
    )
    if (
        int(checkpoint["attempt_count"]) != attempts
        or int(summary.get("attempt_count", -1)) != attempts
        or summary.get("run_id") != run_id
        or summary.get("protocol_id") != protocol["protocol_id"]
        or summary.get("binding_id") != expected_run_config["binding_id"]
        or summary.get("research_spec_id") != research_spec["research_spec_id"]
        or summary.get("code_commit") != research_spec["code_commit"]
        or summary.get("code_fingerprint") != research_spec["code_fingerprint"]
        or summary.get("train_view_id") != expected_run_config["train_view_id"]
        or summary.get("train_view_fingerprint")
        != expected_run_config["train_view_fingerprint"]
        or summary.get("formula_sequence_digest")
        != candidate_state["formula_sequence_digest"]
        or summary.get("attempt_ledger_prefix_sha256")
        != candidate_state["attempt_ledger_prefix_sha256"]
        or summary.get("status")
        != f"{protocol['mode']}_trained_awaiting_funnel"
        or summary.get("validation_or_final_metrics_read") is not False
    ):
        raise RuntimeError("V3A Stage D Transformer source did not finish its protocol")
    return (
        run_id,
        retained_candidates(candidate_state),
        candidate_state,
        summary,
        checkpoint,
    )


def _funnel_marker(
    *, output_dir: Path, summary: dict[str, Any], final_state_path: Path
) -> dict[str, Any]:
    return {
        "schema_version": FUNNEL_COMPLETE_SCHEMA_VERSION,
        "run_id": summary["run_id"],
        "protocol_id": summary["protocol_id"],
        "binding_id": summary["binding_id"],
        "research_spec_id": summary["research_spec_id"],
        "method": summary["method"],
        "seed": summary["seed"],
        "selected_count": summary["selected_count"],
        "summary_sha256": sha256_file(output_dir / "summary.json"),
        "training_funnel_sha256": sha256_file(
            output_dir / "training_funnel.json"
        ),
        "selected_formulas_sha256": sha256_file(
            output_dir / "selected_formulas.jsonl"
        ),
        "final_state_sha256": sha256_file(final_state_path),
    }


def _assert_nested_equal(expected: Any, actual: Any, label: str) -> None:
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor) or not torch.equal(
            expected.cpu(), actual.cpu()
        ):
            raise RuntimeError(f"V3A Stage D final state changed {label}")
        return
    if isinstance(expected, np.ndarray):
        if not isinstance(actual, np.ndarray) or not np.array_equal(expected, actual):
            raise RuntimeError(f"V3A Stage D final state changed {label}")
        return
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(expected) != set(actual):
            raise RuntimeError(f"V3A Stage D final state changed {label} keys")
        for key in expected:
            _assert_nested_equal(expected[key], actual[key], f"{label}.{key}")
        return
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, type(expected)) or len(expected) != len(actual):
            raise RuntimeError(f"V3A Stage D final state changed {label} shape")
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            _assert_nested_equal(left, right, f"{label}[{index}]")
        return
    if expected != actual:
        raise RuntimeError(f"V3A Stage D final state changed {label}")


def _assert_final_state_derived(
    source_checkpoint: dict[str, Any],
    final_checkpoint: dict[str, Any],
    funnel_artifact: dict[str, Any],
) -> None:
    expected = copy.deepcopy(source_checkpoint)
    expected["candidate_state"]["funnel_artifact"] = funnel_artifact
    _assert_nested_equal(expected, final_checkpoint, "checkpoint")


def main() -> None:
    args = parse_args()
    protocol = load_stage_d_protocol(args.protocol_file)
    expected_run_config = method_run_config(
        protocol, method=args.method, seed=args.seed
    )
    verify_stage_c_prerequisite(protocol, args.stage_c_report)
    require_clean_v3a_code(extra_paths=(args.protocol_file,))
    device = _require_cuda()
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
    expected_run_config = {
        **expected_run_config,
        "binding_id": binding["binding_id"],
        "train_view_id": train_view.manifest["train_view_id"],
        "train_view_fingerprint": train_view.manifest["train_view_fingerprint"],
    }

    output_dir = args.run_dir / "funnel"
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_handle = (output_dir / ".funnel.lock").open("a+")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("V3A Stage D funnel is already active") from exc

    run_id, records, state, source_summary, source_checkpoint = (
        _load_transformer_source(
            run_dir=args.run_dir,
            expected_run_config=expected_run_config,
            protocol=protocol,
            research_spec=research_spec,
            scorer_config=scorer_config,
        )
    )
    if not records:
        raise RuntimeError("V3A Stage D source retained no candidate records")

    final_state_path = output_dir / "final_state.pt"
    summary_path = output_dir / "summary.json"
    marker_path = output_dir / "complete.json"
    if summary_path.exists():
        required_paths = (
            final_state_path,
            output_dir / "training_funnel.json",
            output_dir / "selected_formulas.jsonl",
        )
        if any(not path.exists() for path in required_paths):
            raise RuntimeError("V3A Stage D existing funnel output is incomplete")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        expected_status = (
            "pilot_diagnostic_passed"
            if protocol["mode"] == "pilot"
            else "formal_candidates_passed"
        )
        if (
            summary.get("status") != expected_status
            or summary.get("run_id") != run_id
            or summary.get("protocol_id") != protocol["protocol_id"]
            or summary.get("binding_id") != binding["binding_id"]
            or summary.get("method") != args.method
            or int(summary.get("seed", -1)) != args.seed
            or summary.get("research_spec_id") != research_spec["research_spec_id"]
            or summary.get("train_view_id") != train_view.manifest["train_view_id"]
            or int(summary.get("selected_count", -1))
            != int(protocol["candidate_output"]["required_selected_count"])
        ):
            raise RuntimeError("V3A Stage D existing funnel summary identity mismatch")
        existing_funnel = json.loads(
            (output_dir / "training_funnel.json").read_text(encoding="utf-8")
        )
        validate_training_funnel_artifact(
            existing_funnel,
            research_spec=research_spec,
            candidate_config=candidate_config,
        )
        existing_formulas = load_jsonl(output_dir / "selected_formulas.jsonl")
        if len(existing_formulas) != int(summary["selected_count"]):
            raise RuntimeError("V3A Stage D existing selected-formula count mismatch")
        validated_formulas = [
            validate_formula_artifact(artifact, research_spec=research_spec)
            for artifact in existing_formulas
        ]
        expected_hashes = list(existing_funnel["selected_formula_hashes"])
        actual_hashes = [str(artifact["formula_hash"]) for artifact in existing_formulas]
        if actual_hashes != expected_hashes:
            raise RuntimeError("V3A Stage D selected formulas differ from the funnel")
        funnel_records = {
            str(payload["formula_hash"]): candidate_record_from_dict(payload)
            for payload in existing_funnel["signal_unique_records"]
        }
        if any(
            record != funnel_records.get(record.formula_hash)
            for record in validated_formulas
        ):
            raise RuntimeError("V3A Stage D selected formula records differ from funnel")
        funnel_audit = existing_funnel["audit"]
        selected_rewards = [record.reward for record in validated_formulas]
        if (
            summary.get("code_commit") != research_spec["code_commit"]
            or summary.get("code_fingerprint") != research_spec["code_fingerprint"]
            or summary.get("split") != protocol["split"]
            or summary.get("train_view_fingerprint")
            != train_view.manifest["train_view_fingerprint"]
            or int(summary.get("attempt_count", -1))
            != int(expected_run_config["attempts"])
            or summary.get("source_status") != source_summary["status"]
            or int(summary.get("retained_input_count", -1)) != len(records)
            or summary.get("selected_bucket_counts")
            != funnel_audit["selected_bucket_counts"]
            or int(summary.get("cluster_count", -1))
            != int(funnel_audit["cluster_count"])
            or int(summary.get("signal_duplicate_count", -1))
            != int(funnel_audit["signal_duplicate_count"])
            or int(summary.get("insufficient_similarity_overlap_count", -1))
            != int(funnel_audit["insufficient_similarity_overlap_count"])
            or summary.get("selected_reward_distribution")
            != _reward_summary(selected_rewards)
            or summary.get("funnel_artifact_id")
            != existing_funnel["funnel_artifact_id"]
            or summary.get("validation_or_final_metrics_read") is not False
        ):
            raise RuntimeError("V3A Stage D existing funnel summary does not reconcile")
        recovered_state = torch.load(
            final_state_path, map_location="cpu", weights_only=False
        )
        validate_checkpoint(
            recovered_state,
            research_spec=research_spec,
            run_id=run_id,
            model_config=source_checkpoint["model_config"],
            train_config=source_checkpoint["train_config"],
            scorer_config=scorer_config.to_dict(),
        )
        _assert_final_state_derived(
            source_checkpoint, recovered_state, existing_funnel
        )
        expected_marker = _funnel_marker(
            output_dir=output_dir,
            summary=summary,
            final_state_path=final_state_path,
        )
        if marker_path.exists() and json.loads(
            marker_path.read_text(encoding="utf-8")
        ) != expected_marker:
            raise RuntimeError("V3A Stage D existing funnel marker mismatch")
        _write_json_atomic(marker_path, expected_marker)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"output: {output_dir}")
        return
    if marker_path.exists():
        raise RuntimeError("V3A Stage D funnel marker exists without its summary")

    finalization_started = time.monotonic()
    dates = train_view.dates
    mask_numpy = train_view.tradable_mask
    factors_numpy = train_view.factor_values
    targets = build_forward_targets(
        train_view.absolute_open,
        mask_numpy,
        dates,
        SplitSpec("train", "2016-08-09", "2021-12-31"),
        scorer_config,
    )
    factors = torch.as_tensor(factors_numpy, dtype=torch.float32, device=device)
    mask = torch.as_tensor(mask_numpy, dtype=torch.bool, device=device)
    torch_targets = TorchForwardTargets.from_numpy(
        targets, device=device, dtype=torch.float32
    )
    torch.cuda.reset_peak_memory_stats(device)
    signals = _replay_candidate_signals(
        records,
        factors=factors,
        mask=mask,
        targets=torch_targets,
        device=device,
        batch_size=int(expected_run_config["batch_size"]),
    )
    context = SimilarityContext(
        decision_indices=np.arange(targets.days, dtype=np.int64),
        available=targets.available,
    )
    selection = select_training_candidates(
        records, signals, context, candidate_config
    )
    required_count = int(protocol["candidate_output"]["required_selected_count"])
    if len(selection.selected) != required_count:
        raise RuntimeError(
            f"V3A Stage D funnel selected {len(selection.selected)} candidates; "
            f"required {required_count}"
        )
    created_at = str(state["artifact_created_at"])
    funnel_artifact = build_training_funnel_artifact(
        selection,
        research_spec=research_spec,
        candidate_config=candidate_config,
        created_at=created_at,
    )
    validate_training_funnel_artifact(
        funnel_artifact,
        research_spec=research_spec,
        candidate_config=candidate_config,
    )
    formula_artifacts = [
        build_formula_artifact(
            record,
            research_spec=research_spec,
            created_at=created_at,
        )
        for record in selection.selected
    ]
    _write_json_atomic(output_dir / "training_funnel.json", funnel_artifact)
    _write_jsonl_atomic(output_dir / "selected_formulas.jsonl", formula_artifacts)

    final_state = copy.deepcopy(source_checkpoint)
    final_state["candidate_state"]["funnel_artifact"] = funnel_artifact
    validate_checkpoint(
        final_state,
        research_spec=research_spec,
        run_id=run_id,
        model_config=final_state["model_config"],
        train_config=final_state["train_config"],
        scorer_config=scorer_config.to_dict(),
    )
    _assert_final_state_derived(source_checkpoint, final_state, funnel_artifact)
    atomic_save(final_state, final_state_path)

    selected_rewards = [record.reward for record in selection.selected]
    summary: dict[str, Any] = {
        "schema_version": FUNNEL_SUMMARY_SCHEMA_VERSION,
        "status": (
            "pilot_diagnostic_passed"
            if protocol["mode"] == "pilot"
            else "formal_candidates_passed"
        ),
        "created_at": created_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "protocol_id": protocol["protocol_id"],
        "binding_id": binding["binding_id"],
        "protocol_mode": protocol["mode"],
        "formal_budget_approved": protocol["formal_budget_approved"],
        "research_conclusion_allowed": protocol["candidate_output"][
            "research_conclusion_allowed"
        ],
        "method": args.method,
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
        "attempt_count": expected_run_config["attempts"],
        "source_status": source_summary["status"],
        "retained_input_count": len(records),
        "selected_count": len(selection.selected),
        "selected_bucket_counts": selection.audit["selected_bucket_counts"],
        "cluster_count": selection.audit["cluster_count"],
        "signal_duplicate_count": selection.audit["signal_duplicate_count"],
        "insufficient_similarity_overlap_count": selection.audit[
            "insufficient_similarity_overlap_count"
        ],
        "selected_reward_distribution": _reward_summary(selected_rewards),
        "funnel_artifact_id": funnel_artifact["funnel_artifact_id"],
        "finalization_seconds": time.monotonic() - finalization_started,
        "final_state_bytes": final_state_path.stat().st_size,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "validation_or_final_metrics_read": False,
    }
    _write_json_atomic(summary_path, summary)
    marker = _funnel_marker(
        output_dir=output_dir,
        summary=summary,
        final_state_path=final_state_path,
    )
    _write_json_atomic(marker_path, marker)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"output: {output_dir}")


if __name__ == "__main__":
    main()