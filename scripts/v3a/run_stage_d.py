#!/usr/bin/env python3
"""Run the serial V3A Stage D Transformer or matched-random search."""

from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import random
import signal
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from alpha_etf.gpt.policy import TransformerFormulaPolicy, TransformerPolicyConfig
from alpha_etf.research_v3a.candidates import CandidateConfig
from alpha_etf.research_v3a.gpu_sampling import TensorFormulaSampler
from alpha_etf.research_v3a.sampling import PolicyVocab, SamplingConfig
from alpha_etf.research_v3a.scoring import ScorerConfig, SplitSpec, build_forward_targets
from alpha_etf.research_v3a.spec import sha256_file
from alpha_etf.research_v3a.stage_d import (
    load_stage_d_binding,
    load_stage_d_protocol,
    load_stage_d_train_view,
    load_train_view_source_manifest,
    method_run_config,
    require_stage_d_cuda,
    verify_stage_c_prerequisite,
)
from alpha_etf.research_v3a.stage_d_runner import (
    CANDIDATE_SNAPSHOT_SECONDS,
    FAST_CHECKPOINT_SECONDS,
    SCORER_BATCH_CHUNK_SIZE,
    SERIAL_RUNNER_SCHEMA_VERSION,
    SerialStageDConfig,
    SerialStageDRunner,
)
from alpha_etf.research_v3a.torch_scoring import TorchForwardTargets
from alpha_etf.research_v3a.torch_vm import BatchTorchVM
from scripts.v3a.runtime import build_runtime_research_spec, require_clean_v3a_code


DEFAULT_PROTOCOL = ROOT / "configs/v3a_stage_d_gpu_pilot.json"
DEFAULT_OUT_DIR = ROOT / "data/processed/v3a/stage_d/runs"
DEFAULT_TRAIN_VIEW = ROOT / "data/processed/v3a/stage_d/train_view"
DEFAULT_STAGE_C_REPORT = (
    ROOT / "data/processed/v3a/stage_c_reports/v3a-stage-c-20260711-01.json"
)
DEFAULT_BINDING = ROOT / "data/processed/v3a/stage_d/pilot_binding.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("transformer", "matched_random"), required=True)
    parser.add_argument("--protocol-file", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--train-view-dir", type=Path, default=DEFAULT_TRAIN_VIEW)
    parser.add_argument("--stage-c-report", type=Path, default=DEFAULT_STAGE_C_REPORT)
    parser.add_argument("--binding-file", type=Path, default=DEFAULT_BINDING)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--cpu-workers", type=int, default=16)
    parser.add_argument("--disable-cpu-gpu-overlap", action="store_true")
    parser.add_argument("--candidate-snapshot-on-stop", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int)
    return parser.parse_args()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _model_config(
    protocol: dict[str, object], policy_vocab: PolicyVocab
) -> TransformerPolicyConfig:
    model = protocol["model"]
    assert isinstance(model, dict)
    return TransformerPolicyConfig(
        model_vocab_size=policy_vocab.size,
        max_sequence_len=SamplingConfig().max_len + 1,
        d_model=int(model["d_model"]),
        num_layers=int(model["num_layers"]),
        num_heads=int(model["num_heads"]),
        ff_dim=int(model["ff_dim"]),
        dropout=float(model["dropout"]),
        use_critic_head=bool(model["use_critic_head"]),
        use_rmsnorm=bool(model.get("use_rmsnorm", False)),
        use_swiglu=bool(model.get("use_swiglu", False)),
    )


def main() -> None:
    args = parse_args()
    protocol = load_stage_d_protocol(args.protocol_file)
    method_config = method_run_config(protocol, method=args.method, seed=args.seed)
    verify_stage_c_prerequisite(protocol, args.stage_c_report)
    require_clean_v3a_code(extra_paths=(args.protocol_file,))
    device = require_stage_d_cuda(protocol)

    scorer_config = ScorerConfig()
    candidate_config = CandidateConfig()
    source_manifest = load_train_view_source_manifest(args.train_view_dir)
    research_spec = build_runtime_research_spec(
        source_manifest,
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
        f"v3a-stage-d-{protocol['mode']}-{args.method}-s{args.seed}-"
        f"{protocol['protocol_id'][:12]}"
    )
    policy_vocab = PolicyVocab()
    model_config = _model_config(protocol, policy_vocab)
    run_identity = {
        "runner_schema_version": SERIAL_RUNNER_SCHEMA_VERSION,
        "protocol_id": protocol["protocol_id"],
        "binding_id": binding["binding_id"],
        "research_spec_id": research_spec["research_spec_id"],
        "code_commit": research_spec["code_commit"],
        "code_fingerprint": research_spec["code_fingerprint"],
        "train_view_id": train_view.manifest["train_view_id"],
        "train_view_fingerprint": train_view.manifest["train_view_fingerprint"],
        "method": args.method,
        "seed": args.seed,
        "attempts": int(method_config["attempts"]),
        "batch_size": int(method_config["batch_size"]),
        "model": model_config.to_dict() if args.method == "transformer" else None,
        "optimizer": protocol["optimizer"] if args.method == "transformer" else None,
        "reinforce": protocol["reinforce"] if args.method == "transformer" else None,
        "scorer": scorer_config.to_dict(),
        "candidate": candidate_config.to_dict(),
        "scorer_batch_chunk_size": SCORER_BATCH_CHUNK_SIZE,
        "storage_schedule": {
            "fast_checkpoint_seconds": FAST_CHECKPOINT_SECONDS,
            "candidate_snapshot_seconds": CANDIDATE_SNAPSHOT_SECONDS,
            "candidate_snapshot_on_stop": args.candidate_snapshot_on_stop,
        },
    }

    _seed_everything(args.seed)
    model: torch.nn.Module | None = None
    optimizer: torch.optim.Optimizer | None = None
    if args.method == "transformer":
        model = TransformerFormulaPolicy(model_config).to(device)
        optimizer_config = protocol["optimizer"]
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(optimizer_config["learning_rate"]),
            weight_decay=float(optimizer_config["weight_decay"]),
        )

    split = protocol["split"]
    split_end = None if split["end"] is None else str(split["end"])
    targets = build_forward_targets(
        train_view.absolute_open,
        train_view.tradable_mask,
        train_view.dates,
        SplitSpec(str(split["name"]), str(split["start"]), split_end),
        scorer_config,
    )
    factors = torch.as_tensor(
        train_view.factor_values, dtype=torch.float32, device=device
    )
    tradable_mask = torch.as_tensor(
        train_view.tradable_mask, dtype=torch.bool, device=device
    )
    torch_targets = TorchForwardTargets.from_numpy(
        targets, device=device, dtype=torch.float32
    )
    sampler = TensorFormulaSampler(device=device, policy_vocab=policy_vocab)
    gpu_bytes = torch.cuda.get_device_properties(device).total_memory
    vm = BatchTorchVM(
        max_output_bytes=int(gpu_bytes * 0.25),
        max_working_bytes=int(gpu_bytes * 0.25),
        max_total_bytes=int(gpu_bytes * 0.5),
    )
    torch.cuda.reset_peak_memory_stats(device)

    reinforce = protocol["reinforce"]
    optimizer_config = protocol["optimizer"]
    serial_config = SerialStageDConfig(
        run_id=run_id,
        run_identity=run_identity,
        method=args.method,
        seed=args.seed,
        attempts=int(method_config["attempts"]),
        batch_size=int(method_config["batch_size"]),
        cpu_worker_count=args.cpu_workers,
        cpu_gpu_overlap=not args.disable_cpu_gpu_overlap,
        training_invalid_reward=float(
            reinforce.get("training_invalid_reward", scorer_config.hard_invalid_reward)
        ),
        advantage_epsilon=float(reinforce["advantage_epsilon"]),
        entropy_coefficient=float(reinforce["entropy_coefficient"]),
        gradient_clip_norm=float(optimizer_config["gradient_clip_norm"]),
        candidate_snapshot_on_stop=args.candidate_snapshot_on_stop,
    )
    run_dir = args.out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_handle = (run_dir / ".run.lock").open("a+")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"Stage D run is already active: {run_id}") from exc
    try:
        runner = SerialStageDRunner(
            config=serial_config,
            run_dir=run_dir,
            sampler=sampler,
            vm=vm,
            factors=factors,
            tradable_mask=tradable_mask,
            targets=torch_targets,
            scorer_config=scorer_config,
            candidate_config=candidate_config,
            model=model,
            optimizer=optimizer,
        )
        signal.signal(signal.SIGTERM, lambda _signum, _frame: runner.request_stop())
        signal.signal(signal.SIGINT, lambda _signum, _frame: runner.request_stop())
        result = runner.run(resume=args.resume, stop_after=args.stop_after)
    finally:
        lock_handle.close()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"output: {run_dir}")


if __name__ == "__main__":
    main()