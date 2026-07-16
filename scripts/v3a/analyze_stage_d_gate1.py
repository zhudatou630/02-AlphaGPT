#!/usr/bin/env python3
"""Evaluate the four frozen Gate 1 archive-tail mechanism criteria."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from alpha_etf.research_v3a.archive_tail import (  # noqa: E402
    ArchiveTailConfig,
    ArchiveTailState,
)
from alpha_etf.research_v3a.attempts import (  # noqa: E402
    ATTEMPT_DTYPE,
    AttemptStatus,
)
from alpha_etf.research_v3a.candidates import build_candidate_record  # noqa: E402
from scripts.v3a.export_top_formulas import select_curated_records  # noqa: E402
from scripts.v3a.preview_formula_curation import (  # noqa: E402
    CURATED_REWARD_TOLERANCE,
    display_formula_metrics,
)


PROTOCOL_PREFIX = "9f9cfbe379a7"
TOTAL_ATTEMPTS = 2_000_000
BATCH_SIZE = 8192
EARLY_STOP = 500_000
LATE_START = 1_500_000
ARCHIVE_CHECKPOINT = 1_000_000
TOP_SIZE = 50
RETAINED_PER_BUCKET = 500


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mechanism-runs-dir", type=Path, required=True)
    parser.add_argument(
        "--formal-ledgers-dir",
        type=Path,
        default=ROOT / ".pi/profile/results/v3a-gate0-20260715/inputs",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_complete_ledger(run_dir: Path, expected_attempts: int) -> np.memmap:
    marker = json.loads((run_dir / "training_complete.json").read_text(encoding="utf-8"))
    if int(marker["attempt_count"]) != expected_attempts:
        raise RuntimeError(f"Incomplete Gate 1 run: {run_dir}")
    path = run_dir / "attempts.bin"
    if path.stat().st_size != expected_attempts * ATTEMPT_DTYPE.itemsize:
        raise RuntimeError(f"Gate 1 ledger size mismatch: {path}")
    if _sha256(path) != marker["attempt_ledger_sha256"]:
        raise RuntimeError(f"Gate 1 ledger SHA mismatch: {path}")
    records = np.memmap(path, dtype=ATTEMPT_DTYPE, mode="r")
    if not np.array_equal(
        records["attempt_index"], np.arange(expected_attempts, dtype=np.uint64)
    ):
        raise RuntimeError(f"Gate 1 ledger attempt order mismatch: {path}")
    return records


def _model_batches(records: np.ndarray, config: ArchiveTailConfig):
    for start in range(0, len(records), BATCH_SIZE):
        stop = min(start + BATCH_SIZE, len(records))
        model_count = config.model_count(stop - start)
        yield records[start : start + model_count]


def _model_records(records: np.ndarray, config: ArchiveTailConfig) -> np.ndarray:
    return np.concatenate([batch.copy() for batch in _model_batches(records, config)])


def _random_records(records: np.ndarray, config: ArchiveTailConfig) -> np.ndarray:
    chunks = []
    for start in range(0, len(records), BATCH_SIZE):
        stop = min(start + BATCH_SIZE, len(records))
        model_count = config.model_count(stop - start)
        chunks.append(records[start + model_count : stop].copy())
    return np.concatenate(chunks)


def _bucket(lengths: np.ndarray, bucket: int) -> np.ndarray:
    if bucket == 0:
        return lengths <= 5
    if bucket == 1:
        return (lengths > 5) & (lengths <= 10)
    return lengths > 10


def _curated_top50(records: np.ndarray) -> dict[str, float]:
    valid = np.flatnonzero(
        (records["status"] >= int(AttemptStatus.CANONICAL_DUPLICATE))
        & np.isfinite(records["reward"])
    )
    if not valid.size:
        raise RuntimeError("No valid formulas available for Gate 1 top50")
    hashes = np.ascontiguousarray(records["canonical_hash"][valid]).view("S32").reshape(-1)
    order = np.lexsort(
        (
            records["attempt_index"][valid],
            records["token_len"][valid],
            -records["reward"][valid],
            hashes,
        )
    )
    sorted_hashes = hashes[order]
    starts = np.r_[0, np.flatnonzero(sorted_hashes[1:] != sorted_hashes[:-1]) + 1]
    best = valid[order[starts]]
    retained: list[int] = []
    for bucket_index in range(3):
        bucket_rows = best[_bucket(records["token_len"][best], bucket_index)]
        bucket_hashes = np.ascontiguousarray(
            records["canonical_hash"][bucket_rows]
        ).view("S32").reshape(-1)
        bucket_order = np.lexsort(
            (
                bucket_hashes,
                records["token_len"][bucket_rows],
                -records["reward"][bucket_rows],
            )
        )
        retained.extend(int(value) for value in bucket_rows[bucket_order[:RETAINED_PER_BUCKET]])
    candidates = []
    for index in retained:
        row = records[index]
        token_len = int(row["token_len"])
        candidate = build_candidate_record(
            formula_id=f"gate1_a{int(row['attempt_index'])}",
            source="v3a_stage_d_gate1",
            token_ids=[int(value) for value in row["token_ids"][:token_len]],
            reward=float(row["reward"]),
            train_summary={
                "coverage": float(row["coverage"]),
                "finite_std": float(row["finite_std"]),
                "training_step": int(row["training_step"]),
            },
            attempt_index=int(row["attempt_index"]),
        )
        if candidate.formula_hash != row["canonical_hash"].tobytes().hex():
            raise RuntimeError("Gate 1 rebuilt formula hash mismatch")
        candidates.append(candidate)
    candidates.sort(key=lambda item: (-item.reward, item.token_len, item.formula_hash))
    display = {
        candidate.formula_hash: display_formula_metrics(candidate.token_names)
        for candidate in candidates
    }
    curated, _ = select_curated_records(
        candidates,
        size=TOP_SIZE,
        display_metrics=display,
        tolerance=CURATED_REWARD_TOLERANCE,
    )
    rewards = np.asarray([candidate.reward for candidate in curated], dtype=np.float64)
    return {
        "mean_reward": float(rewards.mean()),
        "median_reward": float(np.median(rewards)),
        "rank50_reward": float(rewards[-1]),
    }


def _top_fraction_mean(values: list[float], fraction: float = 0.10) -> float:
    if not values:
        raise RuntimeError("Gate 1 segment has no valid new formulas")
    ordered = np.sort(np.asarray(values, dtype=np.float64))[::-1]
    count = max(1, math.ceil(len(ordered) * fraction))
    return float(ordered[:count].mean())


def _random_cutoff(records: np.ndarray, target_unique: int) -> int:
    hashes = np.ascontiguousarray(records["canonical_hash"]).view("S32").reshape(-1)
    _, first = np.unique(hashes, return_index=True)
    first.sort()
    if target_unique > len(first):
        raise RuntimeError("Random ledger does not reach Gate 1 unique target")
    return int(first[target_unique - 1]) + 1


def _analyze_seed(
    *,
    seed: int,
    mechanism_dir: Path,
    formal_dir: Path,
) -> dict[str, Any]:
    run_id = f"v3a-stage-d-mechanism-transformer-s{seed}-{PROTOCOL_PREFIX}"
    mixed = _load_complete_ledger(mechanism_dir / run_id, TOTAL_ATTEMPTS)
    config = ArchiveTailConfig()
    state = ArchiveTailState(config)
    new_canonical_attempts: list[int] = []
    new_valid: list[tuple[int, float]] = []
    for batch in _model_batches(mixed, config):
        labels = state.apply(batch, prepared=False)
        if not np.allclose(
            batch["training_reward"],
            labels.combined_weights,
            rtol=0.0,
            atol=1e-7,
        ):
            raise RuntimeError("Gate 1 model-lane training labels do not replay")
        new_canonical_attempts.extend(
            int(batch[index]["attempt_index"]) for index in labels.new_canonical_indices
        )
        new_valid.extend(
            (int(batch[index]["attempt_index"]), float(batch[index]["reward"]))
            for index in labels.new_valid_indices
        )
    model = _model_records(mixed, config)
    random_lane = _random_records(mixed, config)
    if np.any(random_lane["training_reward"] != 0.0):
        raise RuntimeError("Gate 1 Random lane unexpectedly affected model training")
    model_1m = model[model["attempt_index"] < ARCHIVE_CHECKPOINT]
    model_2m = model
    top_1m = _curated_top50(model_1m)
    top_2m = _curated_top50(model_2m)

    early_new = sum(attempt < EARLY_STOP for attempt in new_canonical_attempts)
    late_new = sum(attempt >= LATE_START for attempt in new_canonical_attempts)
    early_rewards = [reward for attempt, reward in new_valid if attempt < EARLY_STOP]
    late_rewards = [reward for attempt, reward in new_valid if attempt >= LATE_START]
    early_tail = _top_fraction_mean(early_rewards)
    late_tail = _top_fraction_mean(late_rewards)

    random_run = formal_dir / f"v3a-stage-d-formal-matched_random-s{seed}-02cca48d1c85"
    random = np.memmap(random_run / "attempts.bin", dtype=ATTEMPT_DTYPE, mode="r")
    cutoff = _random_cutoff(random, len(state.seen))
    random_top = _curated_top50(random[:cutoff])
    return {
        "seed": seed,
        "model_attempt_count": len(model),
        "model_canonical_unique_count": len(state.seen),
        "early_new_canonical_count": early_new,
        "late_new_canonical_count": late_new,
        "late_to_early_new_ratio": late_new / early_new,
        "early_new_valid_top10_mean_reward": early_tail,
        "late_new_valid_top10_mean_reward": late_tail,
        "right_tail_difference_reward": late_tail - early_tail,
        "transformer_top50_at_1m": top_1m,
        "transformer_top50_at_2m": top_2m,
        "mechanism_random_lane_top50": _curated_top50(random_lane),
        "combined_lane_top50": _curated_top50(mixed),
        "archive_difference_reward": top_2m["mean_reward"] - top_1m["mean_reward"],
        "random_same_unique_cutoff_attempt": cutoff,
        "random_same_unique_top50": random_top,
        "same_unique_difference_reward": top_2m["mean_reward"] - random_top["mean_reward"],
    }


def _paired_gate(
    results: list[dict[str, Any]], key: str, *, threshold: float = 0.0
) -> dict[str, Any]:
    values = [float(item[key]) for item in results]
    wins = sum(value >= threshold if threshold else value > 0.0 for value in values)
    mean = float(np.mean(values))
    passed = mean >= threshold and wins >= 2 if threshold else mean > 0.0 and wins >= 2
    return {"values": values, "mean": mean, "wins": wins, "passed": passed}


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# V3A Stage D Gate 1 机制结果",
        "",
        "> 仅使用训练期；2022 validation和2023+ final未读取。",
        "",
        f"联合门槛：{'通过' if payload['decision']['all_passed'] else '未通过'}。",
        "",
        "| Seed | 末/初新公式 | 右尾末-初(bp) | top50 2m-1m(bp) | 同唯一数T-R(bp) |",
        "|---:|---:|---:|---:|---:|",
    ]
    for item in payload["results"]:
        lines.append(
            f"| {item['seed']} | {item['late_to_early_new_ratio']:.2%} | "
            f"{item['right_tail_difference_reward'] * 10000:.2f} | "
            f"{item['archive_difference_reward'] * 10000:.2f} | "
            f"{item['same_unique_difference_reward'] * 10000:.2f} |"
        )
    lines.extend(["", "## 四项门槛", ""])
    for name, gate in payload["decision"]["gates"].items():
        lines.append(
            f"- `{name}`：{'通过' if gate['passed'] else '未通过'}；"
            f"均值 `{gate['mean']:.6g}`，达标 `{gate['wins']}/3`。"
        )
    lines.extend(
        [
            "",
            "## 三套候选库",
            "",
            "| Seed | Transformer | 当前Random lane | 两lane合并 |",
            "|---:|---:|---:|---:|",
        ]
    )
    for item in payload["results"]:
        lines.append(
            f"| {item['seed']} | {item['transformer_top50_at_2m']['mean_reward']:.6%} | "
            f"{item['mechanism_random_lane_top50']['mean_reward']:.6%} | "
            f"{item['combined_lane_top50']['mean_reward']:.6%} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    results = [
        _analyze_seed(
            seed=seed,
            mechanism_dir=args.mechanism_runs_dir.resolve(),
            formal_dir=args.formal_ledgers_dir.resolve(),
        )
        for seed in (101, 102, 103)
    ]
    gates = {
        "exploration_retention": _paired_gate(
            results, "late_to_early_new_ratio", threshold=0.50
        ),
        "right_tail_improvement": _paired_gate(results, "right_tail_difference_reward"),
        "archive_continuation": _paired_gate(results, "archive_difference_reward"),
        "same_unique_vs_random": _paired_gate(results, "same_unique_difference_reward"),
    }
    payload = {
        "schema_version": "v3a-stage-d-gate1-analysis-v1",
        "validation_or_final_metrics_read": False,
        "results": results,
        "decision": {
            "gates": gates,
            "all_passed": all(gate["passed"] for gate in gates.values()),
        },
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "gate1_results.json"
    report_path = output_dir / "gate1_report.md"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_report(report_path, payload)
    checksums = [
        f"{_sha256(json_path)}  {json_path.name}",
        f"{_sha256(report_path)}  {report_path.name}",
    ]
    (output_dir / "SHA256SUMS").write_text(
        "\n".join(checksums) + "\n", encoding="ascii"
    )
    print(json.dumps(payload["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()