#!/usr/bin/env python3
"""Verify the next Stage D optimizer update is exact across a CUDA resume."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from alpha_etf.gpt.policy import TransformerFormulaPolicy
from alpha_etf.research_v3a.candidates import CandidateConfig
from alpha_etf.research_v3a.checkpointing import (
    atomic_save,
    build_checkpoint,
    empty_candidate_state,
    load_checkpoint,
    restore_training_state,
)
from alpha_etf.research_v3a.sampling import PolicyVocab, SamplingConfig, sample_formulas
from alpha_etf.research_v3a.language import compile_formula
from alpha_etf.research_v3a.scoring import ScorerConfig, SplitSpec, build_forward_targets
from alpha_etf.research_v3a.spec import sha256_file
from alpha_etf.research_v3a.stage_d import (
    effective_training_rewards,
    load_stage_d_binding,
    load_stage_d_protocol,
    load_stage_d_train_view,
    load_train_view_source_manifest,
    reinforce_objective,
    require_stage_d_cuda,
    verify_stage_c_prerequisite,
)
from alpha_etf.research_v3a.torch_scoring import (
    TorchForwardTargets,
    score_signal_batch,
)
from alpha_etf.research_v3a.torch_vm import BatchTorchVM, compiled_to_tensor
from scripts.v3a.random_baseline import _quality_metrics
from scripts.v3a.random_baseline import _write_json_atomic
from scripts.v3a.runtime import build_runtime_research_spec, require_clean_v3a_code
from scripts.v3a.train_gpu import _model_config, _move_optimizer_state, _seed_everything


DEFAULT_PROTOCOL = ROOT / "configs/v3a_stage_d_gpu_pilot.json"
DEFAULT_TRAIN_VIEW = ROOT / "data/processed/v3a/stage_d/train_view"
DEFAULT_STAGE_C_REPORT = (
    ROOT / "data/processed/v3a/stage_c_reports/v3a-stage-c-20260711-01.json"
)
DEFAULT_OUTPUT = ROOT / "data/processed/v3a/stage_d/gpu_resume_probe.json"
DEFAULT_BINDING = ROOT / "data/processed/v3a/stage_d/pilot_binding.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-file", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--train-view-dir", type=Path, default=DEFAULT_TRAIN_VIEW)
    parser.add_argument("--stage-c-report", type=Path, default=DEFAULT_STAGE_C_REPORT)
    parser.add_argument("--binding-file", type=Path, default=DEFAULT_BINDING)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _optimizer(
    model: torch.nn.Module, protocol: dict[str, Any]
) -> torch.optim.Optimizer:
    config = protocol["optimizer"]
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )


def _update(
    model: TransformerFormulaPolicy,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    protocol: dict[str, Any],
    batch_size: int,
    factors: torch.Tensor,
    mask: torch.Tensor,
    targets: TorchForwardTargets,
    scorer_config: ScorerConfig,
    candidate_config: CandidateConfig,
) -> tuple[list[list[int]], list[float], float]:
    policy_vocab = PolicyVocab()
    sampling = SamplingConfig()
    sample = sample_formulas(model, batch_size, policy_vocab, sampling, device)
    compiled = [compile_formula(tokens) for tokens in sample.formulas]
    codes, lengths = compiled_to_tensor(compiled, device=device)
    vm_result = BatchTorchVM().execute(codes, lengths, factors, mask)
    scored = score_signal_batch(
        vm_result.signal, vm_result.valid, targets, scorer_config
    )
    quality_valid, _, _, _ = _quality_metrics(
        vm_result.signal, targets, candidate_config
    )
    rewards = effective_training_rewards(
        scored.reward,
        scorer_valid=scored.valid,
        quality_valid=quality_valid,
        hard_invalid_reward=scorer_config.hard_invalid_reward,
    )
    objective = reinforce_objective(
        log_prob_sums=sample.log_prob_sums,
        entropy_sums=sample.entropy_sums,
        decision_counts=lengths + 1,
        rewards=rewards,
        advantage_epsilon=float(protocol["reinforce"]["advantage_epsilon"]),
        entropy_coefficient=float(protocol["reinforce"]["entropy_coefficient"]),
    )
    optimizer.zero_grad(set_to_none=True)
    objective.loss.backward()
    torch.nn.utils.clip_grad_norm_(
        model.parameters(), float(protocol["optimizer"]["gradient_clip_norm"])
    )
    optimizer.step()
    return (
        sample.formulas,
        rewards.detach().cpu().tolist(),
        float(objective.loss.detach().cpu().item()),
    )


def _assert_nested_equal(left: Any, right: Any, label: str) -> None:
    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor) or not torch.equal(left.cpu(), right.cpu()):
            raise RuntimeError(f"CUDA resume changed {label}")
        return
    if isinstance(left, dict):
        if not isinstance(right, dict) or set(left) != set(right):
            raise RuntimeError(f"CUDA resume changed {label} keys")
        for key in left:
            _assert_nested_equal(left[key], right[key], f"{label}.{key}")
        return
    if isinstance(left, (list, tuple)):
        if not isinstance(right, type(left)) or len(left) != len(right):
            raise RuntimeError(f"CUDA resume changed {label} shape")
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            _assert_nested_equal(left_item, right_item, f"{label}[{index}]")
        return
    if left != right:
        raise RuntimeError(f"CUDA resume changed {label}")


def main() -> None:
    args = parse_args()
    protocol = load_stage_d_protocol(args.protocol_file)
    verify_stage_c_prerequisite(protocol, args.stage_c_report)
    require_clean_v3a_code(extra_paths=(args.protocol_file,))
    device = require_stage_d_cuda(protocol)
    source_manifest = load_train_view_source_manifest(args.train_view_dir)
    scorer_config_object = ScorerConfig()
    candidate_config = CandidateConfig()
    research_spec = build_runtime_research_spec(
        source_manifest,
        scorer_config=scorer_config_object,
        candidate_config=candidate_config,
    )
    view_manifest = json.loads(
        (args.train_view_dir / "train_view_manifest.json").read_text(encoding="utf-8")
    )
    binding = load_stage_d_binding(
        args.binding_file,
        protocol=protocol,
        research_spec=research_spec,
        train_view_manifest=view_manifest,
        stage_c_report_sha256=sha256_file(args.stage_c_report),
    )

    train_view = load_stage_d_train_view(
        args.train_view_dir, research_spec=research_spec
    )
    targets_numpy = build_forward_targets(
        train_view.absolute_open,
        train_view.tradable_mask,
        train_view.dates,
        SplitSpec("train", "2016-08-09", "2021-12-31"),
        scorer_config_object,
    )
    factors = torch.as_tensor(
        train_view.factor_values, dtype=torch.float32, device=device
    )
    mask = torch.as_tensor(train_view.tradable_mask, dtype=torch.bool, device=device)
    targets = TorchForwardTargets.from_numpy(
        targets_numpy, device=device, dtype=torch.float32
    )

    seed = 271828
    batch_size = int(protocol["batch_size"])
    policy_vocab = PolicyVocab()
    model_config_object = _model_config(protocol, policy_vocab)
    model_config = model_config_object.to_dict()
    train_config = {
        "schema_version": "etf-v3a-stage-d-cuda-resume-probe-v1",
        "protocol_id": protocol["protocol_id"],
        "binding_id": binding["binding_id"],
        "seed": seed,
        "batch_size": batch_size,
    }
    scorer_config = scorer_config_object.to_dict()

    _seed_everything(seed)
    continuous_model = TransformerFormulaPolicy(model_config_object).to(device)
    continuous_optimizer = _optimizer(continuous_model, protocol)
    _update(
        continuous_model,
        continuous_optimizer,
        device=device,
        protocol=protocol,
        batch_size=batch_size,
        factors=factors,
        mask=mask,
        targets=targets,
        scorer_config=scorer_config_object,
        candidate_config=candidate_config,
    )
    checkpoint = build_checkpoint(
        run_id="v3a-stage-d-cuda-resume-probe",
        step=1,
        attempt_count=batch_size,
        model=continuous_model,
        optimizer=continuous_optimizer,
        model_config=model_config,
        train_config=train_config,
        scorer_config=scorer_config,
        candidate_state=empty_candidate_state(attempt_count=batch_size),
        research_spec=research_spec,
    )
    expected_formulas, expected_rewards, expected_loss = _update(
        continuous_model,
        continuous_optimizer,
        device=device,
        protocol=protocol,
        batch_size=batch_size,
        factors=factors,
        mask=mask,
        targets=targets,
        scorer_config=scorer_config_object,
        candidate_config=candidate_config,
    )
    expected_model = continuous_model.state_dict()
    expected_optimizer = continuous_optimizer.state_dict()
    expected_cpu_rng = torch.get_rng_state()
    expected_cuda_rng = torch.cuda.get_rng_state_all()

    with tempfile.TemporaryDirectory() as temporary:
        checkpoint_path = Path(temporary) / "checkpoint.pt"
        atomic_save(checkpoint, checkpoint_path)
        restored_model = TransformerFormulaPolicy(model_config_object).to(device)
        restored_optimizer = _optimizer(restored_model, protocol)
        loaded = load_checkpoint(
            checkpoint_path,
            research_spec=research_spec,
            run_id="v3a-stage-d-cuda-resume-probe",
            model_config=model_config,
            train_config=train_config,
            scorer_config=scorer_config,
        )
        restore_training_state(
            loaded,
            model=restored_model,
            optimizer=restored_optimizer,
            research_spec=research_spec,
            run_id="v3a-stage-d-cuda-resume-probe",
            model_config=model_config,
            train_config=train_config,
            scorer_config=scorer_config,
        )
        _move_optimizer_state(restored_optimizer, device)
        actual_formulas, actual_rewards, actual_loss = _update(
            restored_model,
            restored_optimizer,
            device=device,
            protocol=protocol,
            batch_size=batch_size,
            factors=factors,
            mask=mask,
            targets=targets,
            scorer_config=scorer_config_object,
            candidate_config=candidate_config,
        )

    if (
        actual_formulas != expected_formulas
        or actual_rewards != expected_rewards
        or actual_loss != expected_loss
    ):
        raise RuntimeError("CUDA resume changed the next formulas, rewards, or loss")
    _assert_nested_equal(expected_model, restored_model.state_dict(), "model")
    _assert_nested_equal(
        expected_optimizer, restored_optimizer.state_dict(), "optimizer"
    )
    _assert_nested_equal(expected_cpu_rng, torch.get_rng_state(), "CPU RNG")
    _assert_nested_equal(expected_cuda_rng, torch.cuda.get_rng_state_all(), "CUDA RNG")
    summary = {
        "schema_version": "etf-v3a-stage-d-cuda-resume-probe-v1",
        "status": "passed",
        "protocol_id": protocol["protocol_id"],
        "binding_id": binding["binding_id"],
        "research_spec_id": research_spec["research_spec_id"],
        "code_fingerprint": research_spec["code_fingerprint"],
        "train_view_id": view_manifest["train_view_id"],
        "cuda_device_name": torch.cuda.get_device_name(device),
        "batch_size": batch_size,
        "actual_vm_and_scorer_used": True,
        "next_rewards_equal": True,
        "next_optimizer_update_equal": True,
        "validation_or_final_metrics_read": False,
    }
    _write_json_atomic(args.output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()