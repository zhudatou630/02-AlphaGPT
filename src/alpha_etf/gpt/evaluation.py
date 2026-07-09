"""Formula scoring and artifact helpers shared by Phase 3 training scripts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from alpha_etf.gpt.vm import StackVM, check_signal_quality
from alpha_etf.gpt.vocab import FORMULA_VOCAB, VOCAB_VERSION
from alpha_etf.panel import MarketPanel
from alpha_etf.scoring import ScorerConfig, score_signal


@dataclass(frozen=True)
class FormulaScoreConfig:
    horizon: int = 10
    min_coverage: float = 0.20
    constant_std_eps: float = 1e-12
    hard_invalid_reward: float = -5.0
    weak_invalid_reward: float = -2.0
    reward_name: str = "scorer_mean_return"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def empty_scorer_fields() -> dict[str, float]:
    return {
        "scorer_days": np.nan,
        "scorer_mean_return": np.nan,
        "scorer_median_return": np.nan,
        "scorer_hit_rate": np.nan,
        "scorer_ann_return_proxy": np.nan,
        "avg_top_k": np.nan,
    }


def formula_text(token_ids: list[int]) -> tuple[str, str, str]:
    token_names = FORMULA_VOCAB.decode(token_ids)
    return " ".join(str(token_id) for token_id in token_ids), " ".join(token_names), decode_rpn(token_ids)


def decode_rpn(token_ids: list[int]) -> str:
    stack: list[str] = []
    try:
        for token_id in token_ids:
            token = FORMULA_VOCAB.id_to_token(int(token_id))
            if token.kind in {"feature", "constant"}:
                stack.append(token.name)
            elif token.kind == "operator" and token.arity == 1:
                arg = stack.pop()
                stack.append(f"{token.name}({arg})")
            elif token.kind == "operator" and token.arity == 2:
                right = stack.pop()
                left = stack.pop()
                stack.append(f"({left} {token.name} {right})")
            else:
                return " ".join(FORMULA_VOCAB.decode(token_ids))
        return stack[0] if len(stack) == 1 else " ".join(FORMULA_VOCAB.decode(token_ids))
    except Exception:
        return " ".join(FORMULA_VOCAB.decode(token_ids))


def score_token_formula(
    formula_id: str,
    source: str,
    token_ids: list[int],
    panel: MarketPanel,
    vm: StackVM,
    config: FormulaScoreConfig,
) -> tuple[dict[str, Any], np.ndarray | None]:
    token_ids_text, token_names_text, decoded_formula = formula_text(token_ids)
    base_row: dict[str, Any] = {
        "formula_id": formula_id,
        "source": source,
        "vocab_version": VOCAB_VERSION,
        "token_ids": token_ids_text,
        "token_names": token_names_text,
        "decoded_formula": decoded_formula,
        "token_len": len(token_ids),
        "horizon": config.horizon,
        "valid": False,
        "invalid_reason": "",
        "reward": np.nan,
        "finite_count": 0,
        "coverage": 0.0,
        "finite_std": np.nan,
        **empty_scorer_fields(),
    }

    result = vm.execute(token_ids, panel)
    if not result.valid or result.signal is None:
        base_row.update({"invalid_reason": result.invalid_reason, "reward": config.hard_invalid_reward})
        return base_row, None

    quality = check_signal_quality(result.signal, panel.mask, config.min_coverage, config.constant_std_eps)
    base_row.update(
        {
            "finite_count": quality.finite_count,
            "coverage": quality.coverage,
            "finite_std": quality.finite_std,
        }
    )
    if not quality.valid:
        reward = config.weak_invalid_reward if quality.invalid_reason in {"constant_signal", "low_coverage"} else config.hard_invalid_reward
        base_row.update({"invalid_reason": quality.invalid_reason, "reward": reward})
        return base_row, result.signal

    _, scorer_summary = score_signal(
        formula=formula_id,
        signal=result.signal,
        open_prices=panel.qfq("open"),
        mask=panel.mask,
        dates=panel.dates,
        symbols=panel.symbols,
        config=ScorerConfig(horizon=config.horizon),
    )
    base_row.update(scorer_summary)

    reward = scorer_summary[config.reward_name]
    scorer_days = scorer_summary["scorer_days"]
    if scorer_days <= 0 or not np.isfinite(reward):
        base_row.update({"invalid_reason": "scorer_no_valid_days", "reward": config.hard_invalid_reward})
        return base_row, result.signal

    base_row.update({"valid": True, "invalid_reason": "", "reward": float(reward)})
    return base_row, result.signal


def artifact_from_row(record: dict[str, Any], score_config: FormulaScoreConfig, created_at: str) -> dict[str, Any]:
    token_ids = [int(item) for item in str(record["token_ids"]).split()]
    return {
        "formula_id": record["formula_id"],
        "token_ids": token_ids,
        "token_names": FORMULA_VOCAB.decode(token_ids),
        "decoded_formula": decode_rpn(token_ids),
        "vocab_version": VOCAB_VERSION,
        "scorer_config": score_config.to_dict(),
        "reward": float(record["reward"]),
        "valid": bool(record["valid"]),
        "created_at": created_at,
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def validate_artifact(artifact: dict[str, Any], score_config: FormulaScoreConfig) -> list[int]:
    if artifact.get("vocab_version") != VOCAB_VERSION:
        raise RuntimeError(f"Artifact vocab mismatch for {artifact.get('formula_id')}")
    scorer_config = artifact.get("scorer_config", {})
    expected_scorer = score_config.to_dict()
    if scorer_config != expected_scorer:
        raise RuntimeError(f"Artifact scorer config mismatch for {artifact.get('formula_id')}: {scorer_config}")
    token_ids = [int(token_id) for token_id in artifact["token_ids"]]
    expected_names = FORMULA_VOCAB.decode(token_ids)
    if artifact.get("token_names") != expected_names:
        raise RuntimeError(f"Artifact token name mismatch for {artifact.get('formula_id')}")
    return token_ids


def rescore_loaded_artifacts(
    artifacts: list[dict[str, Any]],
    panel: MarketPanel,
    vm: StackVM,
    score_config: FormulaScoreConfig,
) -> dict[str, float]:
    rewards = {}
    for artifact in artifacts:
        token_ids = validate_artifact(artifact, score_config)
        row, _ = score_token_formula(
            formula_id=str(artifact["formula_id"]),
            source="reloaded",
            token_ids=token_ids,
            panel=panel,
            vm=vm,
            config=score_config,
        )
        if not row["valid"]:
            raise RuntimeError(f"Reloaded artifact became invalid: {artifact['formula_id']} {row['invalid_reason']}")
        rewards[str(artifact["formula_id"])] = float(row["reward"])
    return rewards