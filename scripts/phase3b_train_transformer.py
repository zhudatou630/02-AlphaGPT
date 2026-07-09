#!/usr/bin/env python3
"""Train the Phase 3b small Transformer formula generator."""

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
from alpha_etf.gpt.vm import StackVM
from alpha_etf.gpt.vocab import FORMULA_VOCAB, VOCAB_VERSION
from alpha_etf.panel import load_market_panel
from alpha_etf.validation import ValidatorConfig, run_rank_only_validator, run_validator


OUT_ROOT = ROOT / "data" / "processed" / "phase3b"
HORIZON = 10
TRANSACTION_COST_BPS = 5.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-steps", type=int, default=500)
    parser.add_argument("--smoke-steps", type=int, choices=(50, 100), help="override train steps for a smoke run")
    parser.add_argument("--max-len", type=int, default=16)
    parser.add_argument("--min-formula-len", type=int, default=3)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--ff-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--min-coverage", type=float, default=0.20)
    parser.add_argument("--constant-std-eps", type=float, default=1e-12)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument("--device", type=str, default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--skip-validator", action="store_true")
    return parser.parse_args()


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")
    return torch.device(name)


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


def _run_id(args: argparse.Namespace) -> str:
    if args.run_id:
        return args.run_id
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    steps = int(args.smoke_steps or args.train_steps)
    return f"{stamp}_seed{args.seed}_steps{steps}"


def _best_records(sampled_df: pd.DataFrame, top_n: int) -> list[dict[str, Any]]:
    valid = sampled_df.loc[sampled_df["valid"]].copy()
    if valid.empty:
        return []
    valid = valid.sort_values("reward", ascending=False)
    valid = valid.drop_duplicates("token_ids", keep="first")
    return valid.head(top_n).to_dict("records")


def _invalid_rate(rows: list[dict[str, Any]], reason: str) -> float:
    if not rows:
        return np.nan
    return float(sum(row.get("invalid_reason") == reason for row in rows) / len(rows))


def _validator_audit(artifacts: list[dict[str, Any]], panel, vm: StackVM, score_config: FormulaScoreConfig) -> pd.DataFrame:
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


def main() -> None:
    args = _parse_args()
    train_steps = int(args.smoke_steps or args.train_steps)
    if train_steps < 1:
        raise ValueError("train steps must be positive")
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    if args.top_n < 1:
        raise ValueError("top_n must be positive")

    _set_seeds(args.seed)
    device = _device(args.device)
    run_id = _run_id(args)
    run_dir = args.out_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    args.out_root.mkdir(parents=True, exist_ok=True)

    panel = load_market_panel()
    vm = StackVM()
    policy_vocab = PolicyVocab(FORMULA_VOCAB)
    sampling_config = SamplingConfig(max_len=args.max_len, min_formula_len=args.min_formula_len)
    score_config = FormulaScoreConfig(
        horizon=HORIZON,
        min_coverage=args.min_coverage,
        constant_std_eps=args.constant_std_eps,
    )
    train_config = TrainConfig(
        seed=args.seed,
        batch_size=args.batch_size,
        train_steps=train_steps,
        max_len=args.max_len,
        min_formula_len=args.min_formula_len,
        top_n=args.top_n,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.grad_clip,
        entropy_coef=args.entropy_coef,
    )
    model_config = TransformerPolicyConfig(
        model_vocab_size=policy_vocab.size,
        max_sequence_len=args.max_len + 2,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        use_critic_head=False,
    )
    model = TransformerFormulaPolicy(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    sampled_rows: list[dict[str, Any]] = []
    log_rows: list[dict[str, Any]] = []
    start = time.perf_counter()

    for step in range(1, train_steps + 1):
        model.train()
        sample = sample_formulas(model, args.batch_size, policy_vocab, sampling_config, device)

        rows = []
        for i, token_ids in enumerate(sample.formulas):
            row, _ = score_token_formula(
                formula_id=f"train_{step:06d}_{i:03d}",
                source="phase3b_train",
                token_ids=token_ids,
                panel=panel,
                vm=vm,
                config=score_config,
            )
            row.update({"step": step, "sample_idx": i})
            rows.append(row)
        sampled_rows.extend(rows)

        rewards_np = np.array([float(row["reward"]) for row in rows], dtype=np.float32)
        rewards = torch.tensor(rewards_np, dtype=torch.float32, device=device)
        reward_std = rewards.std(unbiased=False)
        adv = (rewards - rewards.mean()) / (reward_std + 1e-5)
        reinforce_loss = -(sample.log_prob_sums * adv.detach()).mean()
        entropy_mean = sample.entropy_sums.mean()
        loss = reinforce_loss - float(args.entropy_coef) * entropy_mean

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        valid_count = sum(bool(row["valid"]) for row in rows)
        invalid_count = len(rows) - valid_count
        best_so_far = _best_records(pd.DataFrame(sampled_rows), args.top_n)
        best_reward = float(best_so_far[0]["reward"]) if best_so_far else np.nan
        log_rows.append(
            {
                "step": step,
                "avg_reward": float(rewards.mean().item()),
                "std_reward": float(reward_std.item()),
                "best_reward": best_reward,
                "valid_rate": float(valid_count / len(rows)),
                "invalid_rate": float(invalid_count / len(rows)),
                "invalid_vm_rate": _invalid_rate(rows, "stack_underflow") + _invalid_rate(rows, "stack_not_single"),
                "non_finite_rate": _invalid_rate(rows, "non_finite_result"),
                "constant_rate": _invalid_rate(rows, "constant_signal"),
                "low_coverage_rate": _invalid_rate(rows, "low_coverage"),
                "avg_formula_len": float(np.mean([len(tokens) for tokens in sample.formulas])),
                "entropy": float(entropy_mean.item()),
                "loss": float(loss.item()),
                "elapsed_seconds": float(time.perf_counter() - start),
                "avg_allowed_actions": sample.avg_allowed_actions,
            }
        )
        if step == 1 or step == train_steps or step % max(train_steps // 10, 1) == 0:
            latest = log_rows[-1]
            print(
                f"step={step}/{train_steps} avg_reward={latest['avg_reward']:.6f} "
                f"valid={latest['valid_rate']:.1%} best={latest['best_reward']:.6f}",
                flush=True,
            )

    sampled_df = pd.DataFrame(sampled_rows)
    sampled_path = run_dir / "sampled_formulas.csv"
    sampled_df.to_csv(sampled_path, index=False)

    log_df = pd.DataFrame(log_rows)
    training_log_path = run_dir / "training_log.csv"
    log_df.to_csv(training_log_path, index=False)

    created_at = datetime.now(timezone.utc).isoformat()
    best_records = _best_records(sampled_df, args.top_n)
    artifacts = [artifact_from_row(record, score_config, created_at) for record in best_records]
    best_jsonl_path = run_dir / "best_formulas.jsonl"
    write_jsonl(best_jsonl_path, artifacts)

    loaded_artifacts = load_jsonl(best_jsonl_path)
    reloaded_rewards = rescore_loaded_artifacts(loaded_artifacts, panel, vm, score_config)
    best_df = pd.DataFrame(best_records)
    if not best_df.empty:
        best_df["reloaded_reward"] = best_df["formula_id"].map(reloaded_rewards)
        best_df["reward_abs_diff"] = (best_df["reward"].astype(float) - best_df["reloaded_reward"].astype(float)).abs()
        max_reload_diff = float(best_df["reward_abs_diff"].max())
        if max_reload_diff > 1e-12:
            raise RuntimeError(f"Artifact reload reward mismatch: max_diff={max_reload_diff}")
    else:
        max_reload_diff = np.nan
    best_csv_path = run_dir / "best_formulas.csv"
    best_df.to_csv(best_csv_path, index=False)

    if args.skip_validator or not loaded_artifacts:
        validator_df = pd.DataFrame()
    else:
        validator_df = _validator_audit(loaded_artifacts, panel, vm, score_config)
    validator_path = run_dir / "phase3b_validator_summary.csv"
    validator_df.to_csv(validator_path, index=False)

    best_reward = float(best_records[0]["reward"]) if best_records else None
    checkpoint = build_checkpoint(
        step=train_steps,
        model=model,
        optimizer=optimizer,
        model_config=model_config.to_dict(),
        policy_vocab={"token_names": policy_vocab.token_names, "special_tokens": policy_vocab.special_tokens},
        formula_vocab={"vocab_version": VOCAB_VERSION, "token_names": FORMULA_VOCAB.token_names},
        scorer_config=score_config.to_dict(),
        train_config=train_config.to_dict(),
        best_formulas=loaded_artifacts,
        best_reward=best_reward,
        run_id=run_id,
    )
    checkpoint_path = run_dir / "checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)

    summary = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "device": str(device),
        "total_sampled": int(len(sampled_df)),
        "valid_count": int(sampled_df["valid"].sum()) if not sampled_df.empty else 0,
        "invalid_count": int((~sampled_df["valid"]).sum()) if not sampled_df.empty else 0,
        "best_reward": best_reward,
        "reload_max_abs_diff": max_reload_diff,
        "train_config": train_config.to_dict(),
        "model_config": model_config.to_dict(),
        "score_config": score_config.to_dict(),
        "outputs": {
            "checkpoint": str(checkpoint_path),
            "training_log": str(training_log_path),
            "sampled_formulas": str(sampled_path),
            "best_formulas_jsonl": str(best_jsonl_path),
            "best_formulas_csv": str(best_csv_path),
            "validator_summary": str(validator_path),
        },
    }
    summary_path = run_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    (args.out_root / "latest_run.txt").write_text(str(run_dir), encoding="utf-8")

    print(f"run_dir: {run_dir}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"training_log: {training_log_path}")
    print(f"best_formulas_jsonl: {best_jsonl_path} ({len(loaded_artifacts)} rows)")
    print(f"phase3b_validator_summary: {validator_path} ({len(validator_df)} rows)")


if __name__ == "__main__":
    main()