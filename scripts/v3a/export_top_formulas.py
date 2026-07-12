#!/usr/bin/env python3
"""Export a simple training-reward top-N formula library from a Stage D run."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from alpha_etf.research_v3a.artifacts import (  # noqa: E402
    build_formula_artifact,
    validate_formula_artifact,
)
from alpha_etf.research_v3a.candidates import (  # noqa: E402
    CandidateRecord,
    candidate_record_from_dict,
)
from alpha_etf.research_v3a.language import (  # noqa: E402
    FORMULA_VOCAB,
    Expression,
    compile_formula,
)


TOP_LIBRARY_SCHEMA_VERSION = "etf-v3a-top-formula-library-v1"
DEFAULT_OUTPUT_DIR_NAME = "formula_library"
SEMANTIC_STATUSES = {"canonical_duplicate", "selection_duplicate", "accepted_unique"}
BASE_FACTORS = {
    "DAYRET",
    "GAP",
    "INTRADAY",
    "RANGE",
    "CLV",
    "ROC",
    "PRICE_MA",
    "VOL",
    "TS_RANK",
    "RSV",
    "MA_RATIO",
}
OPERATOR_NAMES = {"ADD", "SUB", "MUL", "NEG", "ABS", "SIGN", "REF", "MEAN"}
UNARY_NAMES = {"NEG", "ABS", "SIGN"}
WINDOW_NAMES = {name for name in FORMULA_VOCAB.token_names if name.startswith("WIN_")}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, help="Defaults to <run-dir>/formula_library."
    )
    return parser.parse_args()


def _sort_key(record: CandidateRecord) -> tuple[float, int, str]:
    return (-float(record.reward), int(record.token_len), str(record.formula_hash))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, payloads: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for payload in payloads:
            handle.write(
                json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
            )
    temporary.replace(path)


def _load_source(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any], list[CandidateRecord]]:
    summary = json.loads((run_dir / "training_summary.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(
        run_dir / "checkpoint_final.pt", map_location="cpu", weights_only=False
    )
    state = checkpoint["candidate_state"]
    ledger = state.get("canonical_ledger")
    if not isinstance(ledger, dict) or not ledger:
        raise RuntimeError("Stage D checkpoint has no canonical ledger")

    records: list[CandidateRecord] = []
    ledger_attempts = 0
    for formula_hash, entry in ledger.items():
        if str(entry.get("formula_hash")) != str(formula_hash):
            raise RuntimeError("Canonical ledger hash key mismatch")
        record = candidate_record_from_dict(entry["best_record"])
        if record.formula_hash != str(formula_hash):
            raise RuntimeError("Canonical ledger record hash mismatch")
        if float(entry["best_reward"]) != record.reward:
            raise RuntimeError("Canonical ledger best reward mismatch")
        if int(entry["attempt_count"]) != record.attempt_count:
            raise RuntimeError("Canonical ledger attempt count mismatch")
        ledger_attempts += int(entry["attempt_count"])
        records.append(record)

    expected_semantic = sum(
        int(summary["status_counts"][status]) for status in SEMANTIC_STATUSES
    )
    if ledger_attempts != expected_semantic:
        raise RuntimeError(
            f"Canonical ledger attempts {ledger_attempts} != semantic attempts {expected_semantic}"
        )
    records.sort(key=_sort_key)
    return summary, checkpoint, records


def _walk_expression(
    expression: Expression, *, depth: int = 1
) -> tuple[list[str], list[str], list[str], int, int]:
    factors: list[str] = []
    constants: list[str] = []
    operators: list[str] = []
    max_depth = depth
    node_count = 1
    if expression.kind == "factor":
        factors.append(expression.name)
    elif expression.kind == "constant":
        constants.append(expression.name)
    else:
        operators.append(expression.name)
    for child in expression.children:
        child_factors, child_constants, child_operators, child_depth, child_nodes = (
            _walk_expression(child, depth=depth + 1)
        )
        factors.extend(child_factors)
        constants.extend(child_constants)
        operators.extend(child_operators)
        max_depth = max(max_depth, child_depth)
        node_count += child_nodes
    return factors, constants, operators, max_depth, node_count


def _expression_token_len(payload: dict[str, Any]) -> int:
    kind = str(payload["kind"])
    if kind in {"factor", "constant"}:
        return 1 + (len(payload.get("params", [])) if kind == "factor" else 0)
    return 1 + len(payload.get("params", [])) + sum(
        _expression_token_len(child) for child in payload.get("children", [])
    )


def _record_features(record: CandidateRecord) -> dict[str, Any]:
    compiled = compile_formula(record.token_ids)
    expression = compiled.expression
    factors, constants, operators, max_depth, node_count = _walk_expression(expression)
    token_names = list(record.token_names)
    repeated_factor_count = len(factors) - len(set(factors))
    canonical_token_len = _expression_token_len(record.canonical_expression)
    return {
        "formula_hash": record.formula_hash,
        "formula_text": expression.text(),
        "token_len": int(record.token_len),
        "canonical_token_len": canonical_token_len,
        "canonical_token_reduction": int(record.token_len - canonical_token_len),
        "reward": float(record.reward),
        "best_attempt_index": int(record.best_attempt_index),
        "factors": factors,
        "factor_count": len(factors),
        "unique_factor_count": len(set(factors)),
        "repeated_factor_count": repeated_factor_count,
        "constants": constants,
        "constant_count": len(constants),
        "operators": operators,
        "operator_count": len(operators),
        "unary_operator_count": sum(name in UNARY_NAMES for name in operators),
        "max_depth": max_depth,
        "expression_node_count": node_count,
        "token_counts": dict(Counter(token_names)),
        "base_factor_tokens": [name for name in token_names if name in BASE_FACTORS],
        "window_tokens": [name for name in token_names if name in WINDOW_NAMES],
        "has_constants": bool(constants),
        "has_repeated_factors": repeated_factor_count > 0,
        "canonical_changed": record.expression != record.canonical_expression,
        "is_max_length": record.token_len == 15,
    }


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _reward_distribution(rewards: list[float]) -> dict[str, Any]:
    values = np.asarray(rewards, dtype=np.float64)
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p99": float(np.quantile(values, 0.99)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _factor_summary(features: list[dict[str, Any]]) -> dict[str, Any]:
    formula_counts = Counter()
    token_counts = Counter()
    for feature in features:
        formula_counts.update(set(feature["base_factor_tokens"]))
        token_counts.update(feature["base_factor_tokens"])
    return {
        "formula_occurrence": [
            {"name": name, "count": int(count), "share": float(count / len(features))}
            for name, count in formula_counts.most_common()
        ],
        "token_occurrence": [
            {"name": name, "count": int(count)}
            for name, count in token_counts.most_common()
        ],
    }


def _length_summary(features: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = Counter(int(feature["token_len"]) for feature in features)
    return {str(length): int(lengths[length]) for length in sorted(lengths)}


def _complexity_summary(features: list[dict[str, Any]]) -> dict[str, Any]:
    rewards = np.asarray([feature["reward"] for feature in features], dtype=np.float64)
    lengths = np.asarray([feature["token_len"] for feature in features], dtype=np.float64)
    return {
        "max_length_share": float(np.mean(lengths == 15)),
        "short_le_6_share": float(np.mean(lengths <= 6)),
        "mean_token_len": float(lengths.mean()),
        "mean_operator_count": float(
            np.mean([feature["operator_count"] for feature in features])
        ),
        "mean_max_depth": float(
            np.mean([feature["max_depth"] for feature in features])
        ),
        "with_constants_share": float(
            np.mean([feature["has_constants"] for feature in features])
        ),
        "with_repeated_factors_share": float(
            np.mean([feature["has_repeated_factors"] for feature in features])
        ),
        "canonical_changed_share": float(
            np.mean([feature["canonical_changed"] for feature in features])
        ),
        "canonical_reduction_share": float(
            np.mean([feature["canonical_token_reduction"] > 0 for feature in features])
        ),
        "mean_canonical_token_reduction": float(
            np.mean([feature["canonical_token_reduction"] for feature in features])
        ),
        "reward_length_pearson": _pearson(rewards, lengths),
    }


def _reward_by_length(features: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[int, list[float]] = {}
    for feature in features:
        grouped.setdefault(int(feature["token_len"]), []).append(float(feature["reward"]))
    return {
        str(length): _reward_distribution(rewards)
        for length, rewards in sorted(grouped.items())
    }


def _rank_of_best(
    records: list[CandidateRecord], predicate: Any
) -> dict[str, Any]:
    matches = [record for record in records if predicate(record)]
    if not matches:
        return {"count": 0, "best_reward": None, "best_rank": None}
    best = min(matches, key=_sort_key)
    return {
        "count": len(matches),
        "best_reward": float(best.reward),
        "best_rank": records.index(best) + 1,
        "best_token_len": int(best.token_len),
        "best_formula_hash": best.formula_hash,
    }


def _library_summary(
    *,
    records: list[CandidateRecord],
    features: list[dict[str, Any]],
    top_features: dict[int, list[dict[str, Any]]],
    summary: dict[str, Any],
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    by_hash = {feature["formula_hash"]: feature for feature in features}
    all_reward = [float(record.reward) for record in records]
    output: dict[str, Any] = {
        "schema_version": TOP_LIBRARY_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "run_id": summary["run_id"],
            "protocol_id": summary["protocol_id"],
            "research_spec_id": checkpoint["research_spec_id"],
            "attempt_count": int(summary["attempt_count"]),
            "canonical_valid_formula_count": len(records),
            "validation_or_final_metrics_read": False,
        },
        "ordering": ["reward_desc", "token_len_asc", "formula_hash_asc"],
        "all_canonical_reward_distribution": _reward_distribution(all_reward),
        "all_canonical_length_distribution": _length_summary(features),
        "all_canonical_factor_summary": _factor_summary(features),
        "all_canonical_reward_by_length": _reward_by_length(features),
        "all_canonical_complexity": _complexity_summary(features),
        "simple_formula_reference": {
            "token_len_le_6": _rank_of_best(records, lambda record: record.token_len <= 6),
            "token_len_le_10": _rank_of_best(records, lambda record: record.token_len <= 10),
        },
        "top": {},
    }
    for size, selected in top_features.items():
        selected_features = selected
        selected_records = records[:size]
        selected_rewards = [float(record.reward) for record in selected_records]
        selected_lengths = np.asarray(
            [feature["token_len"] for feature in selected_features], dtype=np.float64
        )
        top_payload = {
            "count": size,
            "reward_distribution": _reward_distribution(selected_rewards),
            "length_distribution": _length_summary(selected_features),
            "complexity": _complexity_summary(selected_features),
            "factor_summary": _factor_summary(selected_features),
            "simple_formula_count_le_6": int(np.sum(selected_lengths <= 6)),
            "simple_formula_count_le_10": int(np.sum(selected_lengths <= 10)),
            "max_length_count": int(np.sum(selected_lengths == 15)),
            "formula_previews": selected_features,
        }
        output["top"][str(size)] = top_payload
    output["source"]["summary_status"] = summary.get("status")
    output["source"]["source_candidate_status"] = summary.get("status")
    return output


def _load_attempts(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "attempts.jsonl"
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _batch_advantages(rewards: np.ndarray, hard_invalid_reward: float) -> tuple[np.ndarray, np.ndarray]:
    effective = np.where(np.isfinite(rewards), rewards, hard_invalid_reward)
    reward_std = float(np.std(effective))
    leave_one_out = (float(effective.sum()) - effective) / float(len(effective) - 1)
    advantages = (effective - leave_one_out) / (reward_std + 1e-5)
    return effective, advantages


def _hard_invalid_sensitivity(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    if not attempts:
        raise RuntimeError("Attempt ledger is empty")
    observed_rewards = np.asarray(
        [float(row["reward"]) if row.get("reward") is not None else np.nan for row in attempts],
        dtype=np.float64,
    )
    invalid = ~np.isfinite(observed_rewards)
    values: dict[str, Any] = {}
    for hard_invalid in (-5.0, -0.1, -0.01, -0.005):
        per_batch: list[dict[str, float]] = []
        for start in range(0, len(attempts), 256):
            batch_rewards = observed_rewards[start : start + 256]
            batch_invalid = ~np.isfinite(batch_rewards)
            if not np.any(batch_invalid):
                continue
            if len(batch_rewards) < 2:
                raise RuntimeError("Cannot standardize a one-row Stage D batch")
            effective, advantages = _batch_advantages(batch_rewards, hard_invalid)
            valid_advantages = advantages[~batch_invalid]
            invalid_advantages = advantages[batch_invalid]
            valid_rewards = batch_rewards[~batch_invalid]
            total_abs = float(np.abs(advantages).sum())
            per_batch.append(
                {
                    "invalid_count": int(batch_invalid.sum()),
                    "invalid_advantage_abs_mean": float(np.abs(invalid_advantages).mean()),
                    "valid_advantage_abs_mean": float(np.abs(valid_advantages).mean()),
                    "invalid_advantage_abs_share": float(
                        np.abs(invalid_advantages).sum() / total_abs if total_abs else 0.0
                    ),
                    "valid_reward_std_over_batch_std": float(
                        np.std(valid_rewards) / (np.std(effective) + 1e-5)
                    ),
                }
            )
        recent = per_batch[-20:]
        values[str(hard_invalid)] = {
            "batches_with_invalid": len(per_batch),
            "all_batches": {
                key: float(np.mean([item[key] for item in per_batch]))
                for key in per_batch[0]
                if key != "invalid_count"
            },
            "last_20_batches": {
                key: float(np.mean([item[key] for item in recent]))
                for key in recent[0]
                if key != "invalid_count"
            },
            "last_20_invalid_count_mean": float(
                np.mean([item["invalid_count"] for item in recent])
            ),
        }
    values["observed_invalid_count"] = int(invalid.sum())
    values["observed_invalid_rate"] = float(invalid.mean())
    return values


def main() -> None:
    args = _parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = (args.output_dir or run_dir / DEFAULT_OUTPUT_DIR_NAME).resolve()
    summary, checkpoint, records = _load_source(run_dir)
    features = [_record_features(record) for record in records]
    feature_by_hash = {feature["formula_hash"]: feature for feature in features}
    created_at = str(checkpoint["candidate_state"]["artifact_created_at"])
    research_spec = checkpoint["research_spec"]

    top_artifacts: dict[int, list[dict[str, Any]]] = {}
    top_features: dict[int, list[dict[str, Any]]] = {}
    for size in (30, 50):
        artifacts: list[dict[str, Any]] = []
        selected_features: list[dict[str, Any]] = []
        for record in records[:size]:
            artifact = build_formula_artifact(
                record, research_spec=research_spec, created_at=created_at
            )
            validate_formula_artifact(artifact, research_spec=research_spec)
            artifacts.append(artifact)
            selected_features.append(feature_by_hash[record.formula_hash])
        top_artifacts[size] = artifacts
        top_features[size] = selected_features
        _write_jsonl(output_dir / f"top{size}_formulas.jsonl", artifacts)

    attempts = _load_attempts(run_dir)
    analysis = _library_summary(
        records=records,
        features=features,
        top_features=top_features,
        summary=summary,
        checkpoint=checkpoint,
    )
    analysis["hard_invalid_sensitivity"] = _hard_invalid_sensitivity(attempts)
    analysis["top_preview"] = {
        str(size): top_features[size][:size] for size in (30, 50)
    }
    _write_json(output_dir / "analysis.json", analysis)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "canonical_valid_formula_count": len(records),
                "top30": str(output_dir / "top30_formulas.jsonl"),
                "top50": str(output_dir / "top50_formulas.jsonl"),
                "analysis": str(output_dir / "analysis.json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()