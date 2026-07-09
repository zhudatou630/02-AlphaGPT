#!/usr/bin/env python3
"""Run Phase 3a random token formulas through VM, scorer, and validator audit."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alpha_etf.formulas import ma_gap_5_20, mom_10
from alpha_etf.gpt.random_formula import RandomFormulaConfig, generate_random_formula
from alpha_etf.gpt.vm import StackVM, check_signal_quality
from alpha_etf.gpt.vocab import FORMULA_VOCAB, VOCAB_VERSION
from alpha_etf.panel import MarketPanel, load_market_panel
from alpha_etf.scoring import ScorerConfig, score_signal
from alpha_etf.validation import ValidatorConfig, run_validator


OUT_DIR = ROOT / "data" / "processed" / "phase3a"
HORIZON = 10
TRANSACTION_COST_BPS = 5.0
SANITY_FORMULAS = {
    "sanity_mom_10": ["close", "close", "DELAY10", "DIV", "CONST_1", "SUB"],
    "sanity_ma_gap_5_20": ["close", "MA5", "close", "MA20", "DIV", "CONST_1", "SUB"],
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=300, help="number of random formulas")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--top-n", type=int, default=20, help="number of valid best formulas to audit")
    parser.add_argument("--min-len", type=int, default=3, help="minimum random formula length")
    parser.add_argument("--max-len", type=int, default=12, help="maximum random formula length before final reductions")
    parser.add_argument("--max-stack-depth", type=int, default=4, help="maximum preferred stack depth")
    parser.add_argument("--min-coverage", type=float, default=0.20, help="minimum finite coverage over mask")
    parser.add_argument("--constant-std-eps", type=float, default=1e-12, help="std threshold for constant signals")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR, help="output directory")
    parser.add_argument("--skip-validator", action="store_true", help="skip best formula validator audit")
    return parser.parse_args()


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


def _empty_scorer_fields() -> dict[str, float]:
    return {
        "scorer_days": np.nan,
        "scorer_mean_return": np.nan,
        "scorer_median_return": np.nan,
        "scorer_hit_rate": np.nan,
        "scorer_ann_return_proxy": np.nan,
        "avg_top_k": np.nan,
    }


def _formula_text(token_ids: list[int]) -> tuple[str, str]:
    token_names = FORMULA_VOCAB.decode(token_ids)
    return " ".join(str(token_id) for token_id in token_ids), " ".join(token_names)


def _score_token_formula(
    formula_id: str,
    source: str,
    token_ids: list[int],
    panel: MarketPanel,
    vm: StackVM,
    min_coverage: float,
    constant_std_eps: float,
) -> tuple[dict[str, Any], np.ndarray | None]:
    token_ids_text, token_names_text = _formula_text(token_ids)
    base_row: dict[str, Any] = {
        "formula_id": formula_id,
        "source": source,
        "vocab_version": VOCAB_VERSION,
        "token_ids": token_ids_text,
        "token_names": token_names_text,
        "token_len": len(token_ids),
        "horizon": HORIZON,
        "valid": False,
        "invalid_reason": "",
        "reward": np.nan,
        "finite_count": 0,
        "coverage": 0.0,
        "finite_std": np.nan,
        **_empty_scorer_fields(),
    }

    result = vm.execute(token_ids, panel)
    if not result.valid or result.signal is None:
        base_row.update({"invalid_reason": result.invalid_reason, "reward": -5.0})
        return base_row, None

    quality = check_signal_quality(result.signal, panel.mask, min_coverage, constant_std_eps)
    base_row.update(
        {
            "finite_count": quality.finite_count,
            "coverage": quality.coverage,
            "finite_std": quality.finite_std,
        }
    )
    if not quality.valid:
        reward = -2.0 if quality.invalid_reason in {"constant_signal", "low_coverage"} else -5.0
        base_row.update({"invalid_reason": quality.invalid_reason, "reward": reward})
        return base_row, result.signal

    _, scorer_summary = score_signal(
        formula=formula_id,
        signal=result.signal,
        open_prices=panel.qfq("open"),
        mask=panel.mask,
        dates=panel.dates,
        symbols=panel.symbols,
        config=ScorerConfig(horizon=HORIZON),
    )
    base_row.update(scorer_summary)

    reward = scorer_summary["scorer_mean_return"]
    scorer_days = scorer_summary["scorer_days"]
    if scorer_days <= 0 or not np.isfinite(reward):
        base_row.update({"invalid_reason": "scorer_no_valid_days", "reward": -5.0})
        return base_row, result.signal

    base_row.update({"valid": True, "invalid_reason": "", "reward": float(reward)})
    return base_row, result.signal


def _sanity_check(panel: MarketPanel, vm: StackVM) -> None:
    expected = {
        "sanity_mom_10": mom_10(panel),
        "sanity_ma_gap_5_20": ma_gap_5_20(panel),
    }
    for formula_id, token_names in SANITY_FORMULAS.items():
        token_ids = FORMULA_VOCAB.encode(token_names)
        result = vm.execute(token_ids, panel)
        if not result.valid or result.signal is None:
            raise RuntimeError(f"{formula_id} VM sanity failed: {result.invalid_reason}")
        finite = np.isfinite(expected[formula_id]) & np.isfinite(result.signal)
        if not finite.any():
            raise RuntimeError(f"{formula_id} sanity has no comparable finite values")
        max_abs_diff = float(np.nanmax(np.abs(expected[formula_id][finite] - result.signal[finite])))
        if max_abs_diff > 1e-12:
            raise RuntimeError(f"{formula_id} sanity mismatch: max_abs_diff={max_abs_diff}")


def _write_jsonl(path: Path, records: list[dict[str, Any]], created_at: str) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            token_ids = [int(item) for item in str(record["token_ids"]).split()]
            artifact = {
                "formula_id": record["formula_id"],
                "token_ids": token_ids,
                "token_names": FORMULA_VOCAB.decode(token_ids),
                "vocab_version": VOCAB_VERSION,
                "scorer_config": {"horizon": HORIZON, "reward": "scorer_mean_return"},
                "reward": float(record["reward"]),
                "created_at": created_at,
            }
            f.write(json.dumps(artifact, ensure_ascii=False, sort_keys=True) + "\n")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def _validate_artifact(artifact: dict[str, Any]) -> list[int]:
    if artifact.get("vocab_version") != VOCAB_VERSION:
        raise RuntimeError(
            f"Artifact vocab mismatch for {artifact.get('formula_id')}: "
            f"{artifact.get('vocab_version')} != {VOCAB_VERSION}"
        )

    scorer_config = artifact.get("scorer_config", {})
    if scorer_config.get("horizon") != HORIZON or scorer_config.get("reward") != "scorer_mean_return":
        raise RuntimeError(f"Artifact scorer config mismatch for {artifact.get('formula_id')}: {scorer_config}")

    token_ids = [int(token_id) for token_id in artifact["token_ids"]]
    expected_names = FORMULA_VOCAB.decode(token_ids)
    if artifact.get("token_names") != expected_names:
        raise RuntimeError(
            f"Artifact token name mismatch for {artifact.get('formula_id')}: "
            f"{artifact.get('token_names')} != {expected_names}"
        )
    return token_ids


def _build_best_records(generated_df: pd.DataFrame, top_n: int) -> list[dict[str, Any]]:
    valid = generated_df.loc[generated_df["valid"]].copy()
    if valid.empty:
        return []
    valid = valid.sort_values("reward", ascending=False)
    valid = valid.drop_duplicates("token_ids", keep="first")
    return valid.head(top_n).to_dict("records")


def _rescore_loaded_artifacts(
    artifacts: list[dict[str, Any]],
    panel: MarketPanel,
    vm: StackVM,
    min_coverage: float,
    constant_std_eps: float,
) -> dict[str, float]:
    rewards = {}
    for artifact in artifacts:
        token_ids = _validate_artifact(artifact)
        row, _ = _score_token_formula(
            formula_id=str(artifact["formula_id"]),
            source="reloaded",
            token_ids=token_ids,
            panel=panel,
            vm=vm,
            min_coverage=min_coverage,
            constant_std_eps=constant_std_eps,
        )
        if not row["valid"]:
            raise RuntimeError(f"Reloaded artifact became invalid: {artifact['formula_id']} {row['invalid_reason']}")
        rewards[str(artifact["formula_id"])] = float(row["reward"])
    return rewards


def _validator_audit(artifacts: list[dict[str, Any]], panel: MarketPanel, vm: StackVM) -> pd.DataFrame:
    rows = []
    for artifact in artifacts:
        formula_id = str(artifact["formula_id"])
        token_ids = _validate_artifact(artifact)
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
    return pd.DataFrame(rows)


def _run_summary(generated_df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    valid = generated_df.loc[generated_df["valid"]]
    invalid = generated_df.loc[~generated_df["valid"]]
    rows: list[dict[str, Any]] = [
        {
            "count": int(args.count),
            "seed": int(args.seed),
            "total_rows": int(len(generated_df)),
            "valid_count": int(len(valid)),
            "invalid_count": int(len(invalid)),
            "invalid_rate": float(len(invalid) / len(generated_df)) if len(generated_df) else np.nan,
            "best_reward": float(valid["reward"].max()) if not valid.empty else np.nan,
            "mean_valid_reward": float(valid["reward"].mean()) if not valid.empty else np.nan,
            "top_n": int(args.top_n),
            "min_coverage": float(args.min_coverage),
            "constant_std_eps": float(args.constant_std_eps),
        }
    ]
    for reason, group in invalid.groupby("invalid_reason", dropna=False):
        rows.append(
            {
                "count": int(args.count),
                "seed": int(args.seed),
                "invalid_reason": reason,
                "invalid_reason_count": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = _parse_args()
    if args.count < 0:
        raise ValueError("--count must be non-negative")
    if args.top_n < 1:
        raise ValueError("--top-n must be positive")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    panel = load_market_panel()
    vm = StackVM()
    rng = random.Random(args.seed)

    _sanity_check(panel, vm)

    formula_config = RandomFormulaConfig(
        min_len=args.min_len,
        max_len=args.max_len,
        max_stack_depth=args.max_stack_depth,
    )
    formula_specs: list[tuple[str, str, list[int]]] = []
    for formula_id, token_names in SANITY_FORMULAS.items():
        formula_specs.append((formula_id, "sanity", FORMULA_VOCAB.encode(token_names)))
    for i in range(args.count):
        token_ids = generate_random_formula(rng, FORMULA_VOCAB, formula_config)
        formula_specs.append((f"random_{i:06d}", "random", token_ids))

    rows = []
    for formula_id, source, token_ids in formula_specs:
        row, _ = _score_token_formula(
            formula_id=formula_id,
            source=source,
            token_ids=token_ids,
            panel=panel,
            vm=vm,
            min_coverage=args.min_coverage,
            constant_std_eps=args.constant_std_eps,
        )
        rows.append(row)

    generated_df = pd.DataFrame(rows)
    generated_path = args.out_dir / "generated_formulas.csv"
    generated_df.to_csv(generated_path, index=False)

    created_at = datetime.now(timezone.utc).isoformat()
    best_records = _build_best_records(generated_df, args.top_n)
    best_jsonl_path = args.out_dir / "best_formulas.jsonl"
    _write_jsonl(best_jsonl_path, best_records, created_at)

    artifacts = _load_jsonl(best_jsonl_path)
    reloaded_rewards = _rescore_loaded_artifacts(artifacts, panel, vm, args.min_coverage, args.constant_std_eps)

    best_df = pd.DataFrame(best_records)
    if not best_df.empty:
        best_df["reloaded_reward"] = best_df["formula_id"].map(reloaded_rewards)
        best_df["reward_abs_diff"] = (best_df["reward"].astype(float) - best_df["reloaded_reward"].astype(float)).abs()
        max_diff = float(best_df["reward_abs_diff"].max())
        if max_diff > 1e-12:
            raise RuntimeError(f"Artifact reload reward mismatch: max_diff={max_diff}")
    best_csv_path = args.out_dir / "best_formulas.csv"
    best_df.to_csv(best_csv_path, index=False)

    if args.skip_validator or not artifacts:
        validator_df = pd.DataFrame()
    else:
        validator_df = _validator_audit(artifacts, panel, vm)
    validator_path = args.out_dir / "phase3a_validator_summary.csv"
    validator_df.to_csv(validator_path, index=False)

    summary_df = _run_summary(generated_df, args)
    summary_path = args.out_dir / "random_run_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"generated_formulas: {generated_path} ({len(generated_df)} rows)")
    print(f"best_formulas_csv: {best_csv_path} ({len(best_df)} rows)")
    print(f"best_formulas_jsonl: {best_jsonl_path} ({len(artifacts)} rows)")
    print(f"phase3a_validator_summary: {validator_path} ({len(validator_df)} rows)")
    print(f"random_run_summary: {summary_path}")
    print(summary_df.head(1).to_string(index=False))
    if not best_df.empty:
        print(best_df[["formula_id", "reward", "token_names", "reloaded_reward", "reward_abs_diff"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
