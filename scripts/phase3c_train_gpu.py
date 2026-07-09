#!/usr/bin/env python3
"""Train the Phase 3c GPU-first Transformer formula generator."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alpha_etf.gpt.checkpointing import TrainConfig, build_checkpoint
from alpha_etf.gpt.evaluation import (
    FormulaScoreConfig,
    artifact_from_row,
    load_jsonl,
    rescore_loaded_artifacts,
    score_token_formula,
    validate_artifact,
    write_jsonl,
)
from alpha_etf.gpt.policy import TransformerFormulaPolicy, TransformerPolicyConfig
from alpha_etf.gpt.sampling import PolicyVocab, SamplingConfig, sample_formulas
from alpha_etf.gpt.torch_scoring import INVALID_CODE_TO_REASON as SCORE_REASON
from alpha_etf.gpt.torch_scoring import BatchScoreResult, score_vm_batch
from alpha_etf.gpt.torch_vm import BatchTorchVM, TorchMarketPanel, formulas_to_tensor
from alpha_etf.gpt.vocab import FORMULA_VOCAB, VOCAB_VERSION
from alpha_etf.panel import load_market_panel
from alpha_etf.validation import ValidatorConfig, run_rank_only_validator, run_validator


OUT_ROOT = ROOT / "data" / "processed" / "phase3c"
HORIZON = 10
TRANSACTION_COST_BPS = 5.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preset", choices=("", "4090d-smoke", "4090d", "4090d-rms-swiglu-smoke", "4090d-rms-swiglu"), default="")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--train-steps", type=int)
    parser.add_argument("--max-len", type=int)
    parser.add_argument("--min-formula-len", type=int, default=3)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--cpu-audit-candidates", type=int, default=100)
    parser.add_argument("--d-model", type=int)
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--num-heads", type=int)
    parser.add_argument("--ff-dim", type=int)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--entropy-coef", type=float)
    parser.add_argument("--use-rmsnorm", action="store_true")
    parser.add_argument("--use-swiglu", action="store_true")
    parser.add_argument("--min-coverage", type=float, default=0.20)
    parser.add_argument("--constant-std-eps", type=float, default=1e-12)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument("--device", type=str, default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--skip-validator", action="store_true")
    parser.add_argument("--save-sampled-formulas", action="store_true")
    parser.add_argument("--resume-from", type=Path, help="resume Phase3c training from checkpoint_latest.pt or checkpoint.pt")
    parser.add_argument("--checkpoint-every-steps", type=int, default=10)
    parser.add_argument("--eval-chunk-size", type=int, help="number of formulas per GPU VM/scorer chunk")
    return parser.parse_args()


def _resolved_batch_size(args: argparse.Namespace) -> int:
    if args.batch_size is not None:
        return int(args.batch_size)
    if args.preset in {"4090d-rms-swiglu-smoke", "4090d-rms-swiglu"}:
        return 4096
    if args.preset == "4090d":
        return 2048
    return 1024


def _resolved_train_steps(args: argparse.Namespace) -> int:
    if args.train_steps is not None:
        return int(args.train_steps)
    if args.preset == "4090d-rms-swiglu-smoke":
        return 2
    if args.preset == "4090d-rms-swiglu":
        return 1000
    if args.preset == "4090d-smoke":
        return 50
    return 100


def _uses_rms_swiglu_preset(args: argparse.Namespace) -> bool:
    return args.preset in {"4090d-rms-swiglu-smoke", "4090d-rms-swiglu"}


def _resolved_max_len(args: argparse.Namespace) -> int:
    if args.max_len is not None:
        return int(args.max_len)
    if _uses_rms_swiglu_preset(args):
        return 12
    return 16


def _resolved_d_model(args: argparse.Namespace) -> int:
    if args.d_model is not None:
        return int(args.d_model)
    if _uses_rms_swiglu_preset(args):
        return 128
    return 64


def _resolved_num_layers(args: argparse.Namespace) -> int:
    if args.num_layers is not None:
        return int(args.num_layers)
    if _uses_rms_swiglu_preset(args):
        return 4
    return 2


def _resolved_num_heads(args: argparse.Namespace) -> int:
    if args.num_heads is not None:
        return int(args.num_heads)
    return 4


def _resolved_ff_dim(args: argparse.Namespace) -> int:
    if args.ff_dim is not None:
        return int(args.ff_dim)
    if _uses_rms_swiglu_preset(args):
        return 512
    return 128


def _resolved_entropy_coef(args: argparse.Namespace) -> float:
    if args.entropy_coef is not None:
        return float(args.entropy_coef)
    if _uses_rms_swiglu_preset(args):
        return 0.001
    return 0.0


def _resolved_use_rmsnorm(args: argparse.Namespace) -> bool:
    return bool(args.use_rmsnorm or _uses_rms_swiglu_preset(args))


def _resolved_use_swiglu(args: argparse.Namespace) -> bool:
    return bool(args.use_swiglu or _uses_rms_swiglu_preset(args))


def _resolved_eval_chunk_size(args: argparse.Namespace, batch_size: int) -> int:
    if args.eval_chunk_size is not None:
        return int(args.eval_chunk_size)
    if _uses_rms_swiglu_preset(args):
        return min(batch_size, 256)
    return batch_size


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _restore_rng_state(state: dict[str, Any]) -> None:
    python_state = state.get("python_random_state")
    if python_state:
        random.setstate((int(python_state["version"]), tuple(python_state["state"]), python_state.get("gauss")))

    numpy_state = state.get("numpy_random_state")
    if numpy_state:
        np.random.set_state(
            (
                str(numpy_state["bit_generator"]),
                np.asarray(numpy_state["state"], dtype=np.uint32),
                int(numpy_state["pos"]),
                int(numpy_state["has_gauss"]),
                float(numpy_state["cached_gaussian"]),
            )
        )

    torch_state = state.get("torch_random_state")
    if torch_state is not None:
        torch.set_rng_state(torch_state.detach().cpu())

    cuda_states = state.get("torch_cuda_random_state_all") or []
    if cuda_states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.detach().cpu() for item in cuda_states])


def _atomic_torch_save(obj: dict[str, Any], path: Path) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp_path)
    tmp_path.replace(path)


def _move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")
    return torch.device(name)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _cuda_memory(device: torch.device) -> tuple[float, float]:
    if device.type != "cuda":
        return np.nan, np.nan
    return float(torch.cuda.memory_allocated(device) / 1024**2), float(torch.cuda.memory_reserved(device) / 1024**2)


def _device_info(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        return {"device_name": str(device), "cuda_total_memory_mb": None}
    props = torch.cuda.get_device_properties(device)
    return {"device_name": props.name, "cuda_total_memory_mb": int(props.total_memory / 1024**2)}


def _run_id(args: argparse.Namespace, train_steps: int) -> str:
    if args.run_id:
        return args.run_id
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    preset = f"_{args.preset}" if args.preset else ""
    return f"{stamp}{preset}_seed{args.seed}_steps{train_steps}"


def _validator_configs() -> tuple[ValidatorConfig, ...]:
    return (
        ValidatorConfig(
            horizon=HORIZON,
            validator_variant="no_max_holding",
            max_holding_days=None,
            transaction_cost_bps=TRANSACTION_COST_BPS,
        ),
        ValidatorConfig(
            horizon=HORIZON,
            validator_variant="max_holding_10",
            max_holding_days=HORIZON,
            transaction_cost_bps=TRANSACTION_COST_BPS,
        ),
    )


def _validator_audit(artifacts: list[dict[str, Any]], panel, score_config: FormulaScoreConfig) -> pd.DataFrame:
    from alpha_etf.gpt.vm import StackVM

    vm = StackVM()
    rows = []
    for artifact in artifacts:
        formula_id = str(artifact["formula_id"])
        token_ids = validate_artifact(artifact, score_config)
        result = vm.execute(token_ids, panel)
        if not result.valid or result.signal is None:
            raise RuntimeError(f"Validator artifact invalid: {formula_id} {result.invalid_reason}")
        for config in _validator_configs():
            _, _, summary = run_validator(
                formula=formula_id,
                signal=result.signal,
                open_prices=panel.qfq("open"),
                close_prices=panel.qfq("close"),
                mask=panel.mask,
                dates=panel.dates,
                symbols=panel.symbols,
                config=config,
            )
            rows.append(summary)
            _, _, rank_summary = run_rank_only_validator(
                formula=formula_id,
                signal=result.signal,
                open_prices=panel.qfq("open"),
                close_prices=panel.qfq("close"),
                mask=panel.mask,
                dates=panel.dates,
                symbols=panel.symbols,
                config=config,
            )
            rows.append(rank_summary)
    return pd.DataFrame(rows)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _token_text(token_ids: list[int]) -> tuple[str, str, str]:
    from alpha_etf.gpt.evaluation import decode_rpn

    token_ids_text = " ".join(str(token_id) for token_id in token_ids)
    token_names = FORMULA_VOCAB.decode(token_ids)
    return token_ids_text, " ".join(token_names), decode_rpn(token_ids)


def _row_from_gpu_score(
    *,
    formula_id: str,
    source: str,
    token_ids: list[int],
    step: int,
    sample_idx: int,
    score: BatchScoreResult,
    score_idx: int,
) -> dict[str, Any]:
    token_ids_text, token_names_text, decoded_formula = _token_text(token_ids)
    invalid_code = int(score.invalid_code[score_idx].detach().cpu().item())
    return {
        "formula_id": formula_id,
        "source": source,
        "vocab_version": VOCAB_VERSION,
        "token_ids": token_ids_text,
        "token_names": token_names_text,
        "decoded_formula": decoded_formula,
        "token_len": len(token_ids),
        "horizon": HORIZON,
        "valid": bool(score.valid[score_idx].detach().cpu().item()),
        "invalid_reason": SCORE_REASON.get(invalid_code, f"unknown_{invalid_code}"),
        "reward": float(score.reward[score_idx].detach().cpu().item()),
        "finite_count": int(score.finite_count[score_idx].detach().cpu().item()),
        "coverage": float(score.coverage[score_idx].detach().cpu().item()),
        "finite_std": float(score.finite_std[score_idx].detach().cpu().item()),
        "scorer_days": int(score.scorer_days[score_idx].detach().cpu().item()),
        "scorer_mean_return": float(score.scorer_mean_return[score_idx].detach().cpu().item()),
        "scorer_median_return": np.nan,
        "scorer_hit_rate": float(score.scorer_hit_rate[score_idx].detach().cpu().item()),
        "scorer_ann_return_proxy": float(score.scorer_mean_return[score_idx].detach().cpu().item()) * 252.0 / HORIZON,
        "avg_top_k": float(score.avg_top_k[score_idx].detach().cpu().item()),
        "max_abs_signal": float(score.max_abs_signal[score_idx].detach().cpu().item()),
        "step": int(step),
        "sample_idx": int(sample_idx),
    }


def _update_best_candidates(
    candidates: dict[str, dict[str, Any]],
    formulas: list[list[int]],
    score: BatchScoreResult,
    step: int,
    keep_n: int,
) -> None:
    valid = score.valid.detach()
    if not bool(valid.any().item()):
        return
    rewards = torch.where(valid, score.reward.detach(), torch.full_like(score.reward, -torch.inf))
    k = min(int(keep_n), int(valid.sum().item()))
    _, indices = torch.topk(rewards, k=k)
    for idx in indices.detach().cpu().tolist():
        token_ids = formulas[int(idx)]
        token_key = " ".join(str(item) for item in token_ids)
        row = _row_from_gpu_score(
            formula_id=f"phase3c_gpu_{step:06d}_{int(idx):04d}",
            source="phase3c_gpu_train",
            token_ids=token_ids,
            step=step,
            sample_idx=int(idx),
            score=score,
            score_idx=int(idx),
        )
        old = candidates.get(token_key)
        if old is None or float(row["reward"]) > float(old["reward"]):
            candidates[token_key] = row
        if len(candidates) > keep_n * 4:
            best = sorted(candidates.values(), key=lambda item: float(item["reward"]), reverse=True)[: keep_n * 2]
            candidates.clear()
            candidates.update({str(item["token_ids"]): item for item in best})


def _audit_candidates_cpu(
    gpu_candidates: list[dict[str, Any]],
    panel,
    score_config: FormulaScoreConfig,
    top_n: int,
) -> pd.DataFrame:
    from alpha_etf.gpt.vm import StackVM

    vm = StackVM()
    rows = []
    for rank, candidate in enumerate(gpu_candidates):
        token_ids = [int(item) for item in str(candidate["token_ids"]).split()]
        row, _ = score_token_formula(
            formula_id=str(candidate["formula_id"]),
            source="phase3c_cpu_audit",
            token_ids=token_ids,
            panel=panel,
            vm=vm,
            config=score_config,
        )
        row.update(
            {
                "gpu_reward": float(candidate["reward"]),
                "gpu_rank": int(rank + 1),
                "step": int(candidate.get("step", -1)),
                "sample_idx": int(candidate.get("sample_idx", -1)),
            }
        )
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    valid = df.loc[df["valid"]].copy()
    if valid.empty:
        return valid
    valid = valid.sort_values("reward", ascending=False)
    valid = valid.drop_duplicates("token_ids", keep="first")
    return valid.head(top_n)


def _save_phase3c_checkpoint(
    *,
    path: Path,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    model_config: TransformerPolicyConfig,
    policy_vocab: PolicyVocab,
    score_config: FormulaScoreConfig,
    train_config: TrainConfig,
    best_candidates: dict[str, dict[str, Any]],
    log_rows: list[dict[str, Any]],
    run_id: str,
    best_formulas: list[dict[str, Any]] | None = None,
    best_reward: float | None = None,
) -> None:
    if best_reward is None:
        best_reward = max((float(row["reward"]) for row in best_candidates.values()), default=None)
    checkpoint = build_checkpoint(
        step=step,
        model=model,
        optimizer=optimizer,
        model_config=model_config.to_dict(),
        policy_vocab={"token_names": policy_vocab.token_names, "special_tokens": policy_vocab.special_tokens},
        formula_vocab={"vocab_version": VOCAB_VERSION, "token_names": FORMULA_VOCAB.token_names},
        scorer_config=score_config.to_dict(),
        train_config=train_config.to_dict(),
        best_formulas=best_formulas or [],
        best_reward=best_reward,
        run_id=run_id,
    )
    checkpoint["phase3c_state"] = {
        "best_candidates": best_candidates,
        "log_rows": log_rows,
    }
    _atomic_torch_save(checkpoint, path)


def _cat_score_results(parts: list[BatchScoreResult]) -> BatchScoreResult:
    if not parts:
        raise ValueError("score result parts must not be empty")
    return BatchScoreResult(
        reward=torch.cat([part.reward for part in parts], dim=0),
        valid=torch.cat([part.valid for part in parts], dim=0),
        invalid_code=torch.cat([part.invalid_code for part in parts], dim=0),
        finite_count=torch.cat([part.finite_count for part in parts], dim=0),
        coverage=torch.cat([part.coverage for part in parts], dim=0),
        finite_std=torch.cat([part.finite_std for part in parts], dim=0),
        scorer_days=torch.cat([part.scorer_days for part in parts], dim=0),
        scorer_mean_return=torch.cat([part.scorer_mean_return for part in parts], dim=0),
        scorer_hit_rate=torch.cat([part.scorer_hit_rate for part in parts], dim=0),
        avg_top_k=torch.cat([part.avg_top_k for part in parts], dim=0),
        max_abs_signal=torch.cat([part.max_abs_signal for part in parts], dim=0),
    )


def _score_formulas_in_chunks(
    *,
    formulas: list[list[int]],
    device: torch.device,
    vm: BatchTorchVM,
    panel: TorchMarketPanel,
    score_config: FormulaScoreConfig,
    chunk_size: int,
) -> BatchScoreResult:
    if chunk_size < 1:
        raise ValueError("eval chunk size must be positive")
    parts: list[BatchScoreResult] = []
    for start in range(0, len(formulas), chunk_size):
        chunk = formulas[start : start + chunk_size]
        token_tensor, lengths = formulas_to_tensor(chunk, device=device)
        vm_result = vm.execute(token_tensor, lengths, panel)
        parts.append(score_vm_batch(vm_result, panel, score_config))
        del token_tensor, lengths, vm_result
    return _cat_score_results(parts)


def main() -> None:
    args = _parse_args()
    resume_checkpoint: dict[str, Any] | None = None
    if args.resume_from is not None:
        if not args.resume_from.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {args.resume_from}")
        resume_checkpoint = torch.load(args.resume_from, map_location="cpu")
        train_dict = dict(resume_checkpoint["train_config"])
        model_dict = dict(resume_checkpoint["model_config"])
        batch_size = int(train_dict["batch_size"])
        train_steps = int(args.train_steps if args.train_steps is not None else train_dict["train_steps"])
        if train_steps < int(resume_checkpoint["step"]):
            raise ValueError("train_steps must be >= checkpoint step when resuming")
        train_dict["train_steps"] = train_steps
        max_len = int(train_dict["max_len"])
        entropy_coef = float(train_dict.get("entropy_coef", 0.0))
        args.min_formula_len = int(train_dict["min_formula_len"])
    else:
        batch_size = _resolved_batch_size(args)
        train_steps = _resolved_train_steps(args)
        max_len = _resolved_max_len(args)
        d_model = _resolved_d_model(args)
        num_layers = _resolved_num_layers(args)
        num_heads = _resolved_num_heads(args)
        ff_dim = _resolved_ff_dim(args)
        entropy_coef = _resolved_entropy_coef(args)
        use_rmsnorm = _resolved_use_rmsnorm(args)
        use_swiglu = _resolved_use_swiglu(args)
        train_dict = {}
        model_dict = {
            "model_vocab_size": 0,
            "max_sequence_len": max_len + 2,
            "d_model": d_model,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "ff_dim": ff_dim,
            "dropout": args.dropout,
            "use_critic_head": False,
            "use_rmsnorm": use_rmsnorm,
            "use_swiglu": use_swiglu,
        }
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    if train_steps < 1:
        raise ValueError("train steps must be positive")
    if max_len < args.min_formula_len:
        raise ValueError("max_len must be >= min_formula_len")
    eval_chunk_size = _resolved_eval_chunk_size(args, batch_size)
    if eval_chunk_size < 1:
        raise ValueError("eval chunk size must be positive")
    if args.top_n < 1:
        raise ValueError("top_n must be positive")
    if args.cpu_audit_candidates < args.top_n:
        raise ValueError("cpu audit candidates must be >= top_n")
    if args.checkpoint_every_steps < 1:
        raise ValueError("checkpoint_every_steps must be positive")

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    _set_seeds(args.seed)
    device = _device(args.device)
    if resume_checkpoint is not None:
        _restore_rng_state(resume_checkpoint.get("rng_state", {}))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    if resume_checkpoint is not None:
        run_id = str(resume_checkpoint["run_id"])
        if args.run_id and args.run_id != run_id:
            raise ValueError(f"--run-id {args.run_id!r} does not match checkpoint run_id {run_id!r}")
        run_dir = args.resume_from.resolve().parent
        run_dir.mkdir(parents=True, exist_ok=True)
        args.out_root = run_dir.parents[1]
    else:
        run_id = _run_id(args, train_steps)
        run_dir = args.out_root / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        args.out_root.mkdir(parents=True, exist_ok=True)

    panel = load_market_panel()
    torch_panel = TorchMarketPanel.from_market_panel(panel, device=device, dtype=torch.float32)
    vm = BatchTorchVM()
    policy_vocab = PolicyVocab(FORMULA_VOCAB)
    sampling_config = SamplingConfig(max_len=max_len, min_formula_len=args.min_formula_len)
    score_config = FormulaScoreConfig(
        horizon=HORIZON,
        min_coverage=args.min_coverage,
        constant_std_eps=args.constant_std_eps,
    )
    train_config = TrainConfig(
        seed=args.seed,
        batch_size=batch_size,
        train_steps=train_steps,
        max_len=max_len,
        min_formula_len=args.min_formula_len,
        top_n=args.top_n,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.grad_clip,
        entropy_coef=entropy_coef,
    )
    if resume_checkpoint is not None:
        train_config = TrainConfig(**{**train_config.to_dict(), **train_dict})
        train_config = TrainConfig(**{**train_config.to_dict(), "train_steps": train_steps})
        model_config = TransformerPolicyConfig(**model_dict)
    else:
        model_dict["model_vocab_size"] = policy_vocab.size
        model_config = TransformerPolicyConfig(**model_dict)
    model = TransformerFormulaPolicy(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_step = 1
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model_state_dict"])
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        _move_optimizer_state(optimizer, device)
        start_step = int(resume_checkpoint["step"]) + 1

    candidate_keep_n = max(args.cpu_audit_candidates, args.top_n)
    phase3c_state = resume_checkpoint.get("phase3c_state", {}) if resume_checkpoint is not None else {}
    best_candidates: dict[str, dict[str, Any]] = dict(phase3c_state.get("best_candidates", {}))
    sampled_rows: list[dict[str, Any]] = []
    log_rows: list[dict[str, Any]] = list(phase3c_state.get("log_rows", []))
    run_start = time.perf_counter()

    if start_step > train_steps:
        print(f"checkpoint already reached step {start_step - 1}; finalizing artifacts")

    latest_checkpoint_path = run_dir / "checkpoint_latest.pt"
    for step in range(start_step, train_steps + 1):
        step_start = time.perf_counter()
        model.train()

        sample_start = time.perf_counter()
        sample = sample_formulas(model, batch_size, policy_vocab, sampling_config, device)
        _sync(device)
        sample_seconds = time.perf_counter() - sample_start

        with torch.no_grad():
            score_start = time.perf_counter()
            score = _score_formulas_in_chunks(
                formulas=sample.formulas,
                device=device,
                vm=vm,
                panel=torch_panel,
                score_config=score_config,
                chunk_size=eval_chunk_size,
            )
            _sync(device)
            vm_seconds = float("nan")
            score_seconds = time.perf_counter() - score_start

        rewards = score.reward.detach()
        reward_std = rewards.std(unbiased=False)
        adv = (rewards - rewards.mean()) / (reward_std + 1e-5)
        reinforce_loss = -(sample.log_prob_sums * adv.detach()).mean()
        entropy_mean = sample.entropy_sums.mean()
        loss = reinforce_loss - float(entropy_coef) * entropy_mean

        backward_start = time.perf_counter()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        _sync(device)
        backward_seconds = time.perf_counter() - backward_start

        _update_best_candidates(best_candidates, sample.formulas, score, step, candidate_keep_n)
        if args.save_sampled_formulas:
            for i, token_ids in enumerate(sample.formulas):
                sampled_rows.append(
                    _row_from_gpu_score(
                        formula_id=f"phase3c_gpu_{step:06d}_{i:04d}",
                        source="phase3c_gpu_train",
                        token_ids=token_ids,
                        step=step,
                        sample_idx=i,
                        score=score,
                        score_idx=i,
                    )
                )

        step_seconds = time.perf_counter() - step_start
        valid_count = int(score.valid.sum().item())
        invalid_count = batch_size - valid_count
        best_reward = max((float(row["reward"]) for row in best_candidates.values()), default=np.nan)
        mem_alloc, mem_reserved = _cuda_memory(device)
        log_rows.append(
            {
                "step": step,
                "avg_reward": float(rewards.mean().item()),
                "std_reward": float(reward_std.item()),
                "step_max_reward": float(rewards.max().item()),
                "best_gpu_reward": float(best_reward),
                "valid_rate": float(valid_count / batch_size),
                "invalid_rate": float(invalid_count / batch_size),
                "avg_formula_len": float(np.mean([len(tokens) for tokens in sample.formulas])),
                "entropy": float(entropy_mean.item()),
                "loss": float(loss.item()),
                "sample_seconds": float(sample_seconds),
                "vm_seconds": float(vm_seconds),
                "score_seconds": float(score_seconds),
                "backward_seconds": float(backward_seconds),
                "step_seconds": float(step_seconds),
                "formulas_per_second": float(batch_size / step_seconds),
                "cuda_memory_allocated_mb": mem_alloc,
                "cuda_memory_reserved_mb": mem_reserved,
                "avg_allowed_actions": sample.avg_allowed_actions,
                "eval_chunk_size": int(eval_chunk_size),
            }
        )
        if step == 1 or step == train_steps or step % max(train_steps // 10, 1) == 0:
            latest = log_rows[-1]
            print(
                f"step={step}/{train_steps} avg_reward={latest['avg_reward']:.6f} "
                f"valid={latest['valid_rate']:.1%} best_gpu={latest['best_gpu_reward']:.6f} "
                f"fps={latest['formulas_per_second']:.1f}",
                flush=True,
            )
        if step == train_steps or step % args.checkpoint_every_steps == 0:
            pd.DataFrame(log_rows).to_csv(run_dir / "training_log.csv", index=False)
            gpu_candidates = sorted(best_candidates.values(), key=lambda item: float(item["reward"]), reverse=True)[:candidate_keep_n]
            pd.DataFrame(gpu_candidates).to_csv(run_dir / "gpu_candidate_formulas.csv", index=False)
            _save_phase3c_checkpoint(
                path=latest_checkpoint_path,
                step=step,
                model=model,
                optimizer=optimizer,
                model_config=model_config,
                policy_vocab=policy_vocab,
                score_config=score_config,
                train_config=train_config,
                best_candidates=best_candidates,
                log_rows=log_rows,
                run_id=run_id,
            )

    log_df = pd.DataFrame(log_rows)
    training_log_path = run_dir / "training_log.csv"
    log_df.to_csv(training_log_path, index=False)

    sampled_path = None
    if args.save_sampled_formulas:
        sampled_path = run_dir / "sampled_formulas.csv"
        pd.DataFrame(sampled_rows).to_csv(sampled_path, index=False)

    gpu_candidates = sorted(best_candidates.values(), key=lambda item: float(item["reward"]), reverse=True)[:candidate_keep_n]
    gpu_candidates_path = run_dir / "gpu_candidate_formulas.csv"
    pd.DataFrame(gpu_candidates).to_csv(gpu_candidates_path, index=False)

    best_df = _audit_candidates_cpu(gpu_candidates, panel, score_config, args.top_n)
    best_csv_path = run_dir / "best_formulas.csv"
    best_df.to_csv(best_csv_path, index=False)

    created_at = datetime.now(timezone.utc).isoformat()
    best_records = best_df.to_dict("records") if not best_df.empty else []
    artifacts = [artifact_from_row(record, score_config, created_at) for record in best_records]
    best_jsonl_path = run_dir / "best_formulas.jsonl"
    write_jsonl(best_jsonl_path, artifacts)

    loaded_artifacts = load_jsonl(best_jsonl_path)
    from alpha_etf.gpt.vm import StackVM

    reloaded_rewards = rescore_loaded_artifacts(loaded_artifacts, panel, StackVM(), score_config)
    if not best_df.empty:
        best_df["reloaded_reward"] = best_df["formula_id"].map(reloaded_rewards)
        best_df["reward_abs_diff"] = (best_df["reward"].astype(float) - best_df["reloaded_reward"].astype(float)).abs()
        max_reload_diff = float(best_df["reward_abs_diff"].max())
        if max_reload_diff > 1e-12:
            raise RuntimeError(f"Artifact reload reward mismatch: max_diff={max_reload_diff}")
        best_df.to_csv(best_csv_path, index=False)
    else:
        max_reload_diff = np.nan

    if args.skip_validator or not loaded_artifacts:
        validator_df = pd.DataFrame()
    else:
        validator_df = _validator_audit(loaded_artifacts, panel, score_config)
    validator_path = run_dir / "phase3c_validator_summary.csv"
    validator_df.to_csv(validator_path, index=False)

    best_reward = float(best_records[0]["reward"]) if best_records else None
    checkpoint_path = run_dir / "checkpoint.pt"
    _save_phase3c_checkpoint(
        path=checkpoint_path,
        step=train_steps,
        model=model,
        optimizer=optimizer,
        model_config=model_config,
        policy_vocab=policy_vocab,
        score_config=score_config,
        train_config=train_config,
        best_candidates=best_candidates,
        log_rows=log_rows,
        best_formulas=loaded_artifacts,
        best_reward=best_reward,
        run_id=run_id,
    )
    _save_phase3c_checkpoint(
        path=latest_checkpoint_path,
        step=train_steps,
        model=model,
        optimizer=optimizer,
        model_config=model_config,
        policy_vocab=policy_vocab,
        score_config=score_config,
        train_config=train_config,
        best_candidates=best_candidates,
        log_rows=log_rows,
        best_formulas=loaded_artifacts,
        best_reward=best_reward,
        run_id=run_id,
    )

    total_seconds = time.perf_counter() - run_start
    outputs = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_latest": str(latest_checkpoint_path),
        "training_log": str(training_log_path),
        "gpu_candidate_formulas": str(gpu_candidates_path),
        "best_formulas_jsonl": str(best_jsonl_path),
        "best_formulas_csv": str(best_csv_path),
        "validator_summary": str(validator_path),
    }
    if sampled_path is not None:
        outputs["sampled_formulas"] = str(sampled_path)
    summary = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "device": str(device),
        "preset": args.preset,
        "device_info": _device_info(device),
        "total_sampled": int(batch_size * train_steps),
        "best_reward": best_reward,
        "reload_max_abs_diff": max_reload_diff,
        "total_seconds": float(total_seconds),
        "avg_step_seconds": float(log_df["step_seconds"].mean()) if not log_df.empty else np.nan,
        "avg_formulas_per_second": float(log_df["formulas_per_second"].mean()) if not log_df.empty else np.nan,
        "train_config": train_config.to_dict(),
        "model_config": model_config.to_dict(),
        "score_config": score_config.to_dict(),
        "eval_chunk_size": int(eval_chunk_size),
        "outputs": outputs,
    }
    summary_path = run_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    (args.out_root / "latest_run.txt").write_text(str(run_dir), encoding="utf-8")

    print(f"run_dir: {run_dir}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"training_log: {training_log_path}")
    print(f"gpu_candidate_formulas: {gpu_candidates_path} ({len(gpu_candidates)} rows)")
    print(f"best_formulas_jsonl: {best_jsonl_path} ({len(loaded_artifacts)} rows)")
    print(f"phase3c_validator_summary: {validator_path} ({len(validator_df)} rows)")


if __name__ == "__main__":
    main()
