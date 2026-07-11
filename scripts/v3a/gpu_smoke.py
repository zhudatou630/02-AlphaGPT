#!/usr/bin/env python3
"""Run the frozen V3A CUDA, Transformer, and checkpoint smoke gates."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import subprocess
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from alpha_etf.gpt.policy import TransformerFormulaPolicy, TransformerPolicyConfig
from alpha_etf.research_v3a.candidates import CandidateConfig
from alpha_etf.research_v3a.checkpointing import (
    atomic_save,
    build_checkpoint,
    empty_candidate_state,
    load_checkpoint,
    restore_training_state,
)
from alpha_etf.research_v3a.factors import build_factor_values_numpy
from alpha_etf.research_v3a.language import compile_formula
from alpha_etf.research_v3a.sampling import (
    PolicyVocab,
    SamplingConfig,
    sample_formulas,
)
from alpha_etf.research_v3a.scoring import ScorerConfig, SplitSpec, build_forward_targets
from alpha_etf.research_v3a.spec import load_dataset_manifest, load_panel
from alpha_etf.research_v3a.torch_scoring import TorchForwardTargets, score_signal_batch
from alpha_etf.research_v3a.torch_vm import BatchTorchVM, compiled_to_tensor
from scripts.v3a.runtime import DATASET_DIR, build_runtime_research_spec


DEFAULT_OUT_DIR = ROOT / "data/processed/v3a/smoke"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--run-id")
    parser.add_argument("--expected-dataset-id")
    parser.add_argument("--expected-panel-sha256")
    parser.add_argument("--expected-research-spec-id")
    parser.add_argument("--expected-code-fingerprint")
    parser.add_argument(
        "--allow-unpinned-identity",
        action="store_true",
        help="Permit a local non-gating test without externally pinned identities.",
    )
    return parser.parse_args()


def _device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(name)


def _require_identity(actual: str, expected: str | None, name: str) -> None:
    if expected is not None and actual != expected:
        raise RuntimeError(f"V3A {name} mismatch: {actual} != {expected}")


def _identity_is_pinned(args: argparse.Namespace) -> bool:
    values = (
        args.expected_dataset_id,
        args.expected_panel_sha256,
        args.expected_research_spec_id,
        args.expected_code_fingerprint,
    )
    if all(value is not None for value in values):
        return True
    if any(value is not None for value in values):
        raise RuntimeError("V3A identity pins must be supplied together")
    if not args.allow_unpinned_identity:
        raise RuntimeError(
            "V3A formal smoke requires all four expected identities; "
            "use --allow-unpinned-identity only for a local non-gating test"
        )
    return False


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.batch_size < 2:
        raise ValueError("batch-size must be at least 2")
    device = _device(args.device)
    identity_pinned = _identity_is_pinned(args)
    gpu_gate_satisfied = device.type == "cuda" and identity_pinned
    manifest = load_dataset_manifest(args.dataset_dir)
    scorer_config = ScorerConfig()
    candidate_config = CandidateConfig()
    research_spec = build_runtime_research_spec(
        manifest, scorer_config=scorer_config, candidate_config=candidate_config
    )
    _require_identity(manifest["dataset_id"], args.expected_dataset_id, "dataset id")
    _require_identity(manifest["panel_sha256"], args.expected_panel_sha256, "panel SHA")
    _require_identity(
        research_spec["research_spec_id"],
        args.expected_research_spec_id,
        "ResearchSpec id",
    )
    _require_identity(
        research_spec["code_fingerprint"],
        args.expected_code_fingerprint,
        "code fingerprint",
    )

    run_id = args.run_id or (
        f"v3a-smoke-{args.device}-s{args.seed}-"
        f"{research_spec['research_spec_id'][:12]}"
    )
    run_dir = args.out_dir / run_id
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"V3A GPU-smoke run already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    fixed_dir = run_dir / "fixed_formula"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/v3a/fixed_formula_sanity.py"),
            "--dataset-dir",
            str(args.dataset_dir),
            "--out-dir",
            str(fixed_dir),
            "--device",
            args.device,
        ],
        cwd=ROOT,
        check=True,
    )
    fixed_summary = json.loads(
        (fixed_dir / "fixed_formula_sanity.json").read_text(encoding="utf-8")
    )
    if (
        fixed_summary.get("status") != "passed"
        or fixed_summary.get("research_spec_id") != research_spec["research_spec_id"]
    ):
        raise RuntimeError("V3A fixed-formula smoke identity or status mismatch")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(device)

    panel = load_panel(args.dataset_dir)
    train_end = int(
        np.searchsorted(panel.dates.values, np.datetime64("2021-12-31"), side="right")
    )
    dates = panel.dates[:train_end]
    mask_numpy = panel.tradable_mask[:, :train_end]
    absolute = panel.absolute_ohlc[:, :, :train_end]
    factor_numpy = build_factor_values_numpy(absolute, mask_numpy)
    targets = build_forward_targets(
        absolute[:, 0, :],
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

    policy_vocab = PolicyVocab()
    sampling_config = SamplingConfig()
    model_config = TransformerPolicyConfig(
        model_vocab_size=policy_vocab.size,
        max_sequence_len=sampling_config.max_len + 1,
        d_model=64,
        num_layers=2,
        num_heads=4,
        ff_dim=128,
        dropout=0.0,
    )
    model = TransformerFormulaPolicy(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    sample = sample_formulas(
        model, args.batch_size, policy_vocab, sampling_config, device
    )
    compiled = [compile_formula(tokens) for tokens in sample.formulas]
    codes, lengths = compiled_to_tensor(compiled, device=device)
    vm_result = BatchTorchVM().execute(codes, lengths, factors, mask)
    score_result = score_signal_batch(vm_result.signal, vm_result.valid, torch_targets, scorer_config)
    loss = -(sample.log_prob_sums * score_result.reward.detach()).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    finite_gradients = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item())
        for parameter in model.parameters()
    )
    if not finite_gradients:
        raise RuntimeError("V3A Transformer smoke produced non-finite gradients")
    optimizer.step()

    train_config = {
        "schema_version": "etf-v3a-gpu-smoke-v1",
        "random_initialization": True,
        "source_checkpoint": None,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "sampling_config": sampling_config.to_dict(),
    }
    checkpoint = build_checkpoint(
        run_id=run_id,
        step=1,
        attempt_count=args.batch_size,
        model=model,
        optimizer=optimizer,
        model_config=model_config.to_dict(),
        train_config=train_config,
        scorer_config=scorer_config.to_dict(),
        candidate_state=empty_candidate_state(attempt_count=args.batch_size),
        research_spec=research_spec,
    )
    checkpoint_path = run_dir / "checkpoint.pt"
    atomic_save(checkpoint, checkpoint_path)
    expected_next = sample_formulas(
        model, args.batch_size, policy_vocab, sampling_config, device
    ).formulas

    restored_model = TransformerFormulaPolicy(model_config).to(device)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-4)
    loaded = load_checkpoint(
        checkpoint_path,
        research_spec=research_spec,
        run_id=run_id,
        model_config=model_config.to_dict(),
        train_config=train_config,
        scorer_config=scorer_config.to_dict(),
    )
    restored = restore_training_state(
        loaded,
        model=restored_model,
        optimizer=restored_optimizer,
        research_spec=research_spec,
        run_id=run_id,
        model_config=model_config.to_dict(),
        train_config=train_config,
        scorer_config=scorer_config.to_dict(),
    )
    resumed_next = sample_formulas(
        restored_model, args.batch_size, policy_vocab, sampling_config, device
    ).formulas
    if resumed_next != expected_next:
        raise RuntimeError("V3A checkpoint resume changed the next sampled formula batch")

    summary: dict[str, Any] = {
        "schema_version": "etf-v3a-gpu-smoke-result-v1",
        "status": "passed" if gpu_gate_satisfied else "local_test_passed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "device": str(device),
        "identity_pinned": identity_pinned,
        "gpu_gate_satisfied": gpu_gate_satisfied,
        "cuda_device_name": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else None,
        "dataset_id": manifest["dataset_id"],
        "panel_sha256": manifest["panel_sha256"],
        "research_spec_id": research_spec["research_spec_id"],
        "code_commit": research_spec["code_commit"],
        "code_fingerprint": research_spec["code_fingerprint"],
        "random_initialization": True,
        "source_checkpoint": None,
        "batch_size": args.batch_size,
        "formula_lengths": lengths.detach().cpu().tolist(),
        "vm_valid_count": int(vm_result.valid.sum().item()),
        "scorer_valid_count": int(score_result.valid.sum().item()),
        "loss": float(loss.detach().cpu().item()),
        "finite_gradients": finite_gradients,
        "checkpoint_step": restored["step"],
        "checkpoint_attempt_count": restored["attempt_count"],
        "resume_next_batch_equal": True,
        "fixed_formula_max_signal_abs_diff": fixed_summary[
            "max_cpu_torch_signal_abs_diff"
        ],
        "fixed_formula_max_reward_abs_diff": fixed_summary[
            "max_cpu_torch_reward_abs_diff"
        ],
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else None,
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device)
        if device.type == "cuda"
        else None,
        "validation_or_final_metrics_read": False,
    }
    _write_json_atomic(run_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"output: {run_dir}")


if __name__ == "__main__":
    main()