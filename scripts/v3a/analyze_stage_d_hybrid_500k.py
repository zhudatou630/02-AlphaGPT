#!/usr/bin/env python3
"""Test a fixed 500k Transformer + 7.5m random Stage D search budget."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import heapq
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from alpha_etf.research_v3a.candidates import CandidateRecord  # noqa: E402
from scripts.v3a.analyze_stage_d_gate0 import (  # noqa: E402
    RETAINED_PER_BUCKET,
    RUN_SPECS,
    _best_valid_rows_in_segment,
    _bucket,
    _candidate_record,
    _run_id,
    _validate_ledger,
)
from scripts.v3a.export_top_formulas import select_curated_records  # noqa: E402
from scripts.v3a.preview_formula_curation import (  # noqa: E402
    CURATED_REWARD_TOLERANCE,
    CURATION_RULE_VERSION,
    display_formula_metrics,
)


SCHEMA_VERSION = "v3a-stage-d-hybrid-500k-diagnostics-v1"
TRANSFORMER_ATTEMPTS = 500_000
RANDOM_ATTEMPTS = 7_500_000
TOTAL_ATTEMPTS = TRANSFORMER_ATTEMPTS + RANDOM_ATTEMPTS
TOP_SIZE = 50


@dataclass(frozen=True)
class MergedCandidate:
    method: str
    index: int
    provenance: str
    canonical_hash: bytes


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / ".pi/profile/results/v3a-gate0-20260715/inputs",
    )
    parser.add_argument(
        "--gate0-results",
        type=Path,
        default=ROOT / ".pi/profile/results/v3a-gate0-20260715/gate0_results.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / ".pi/profile/results/v3a-gate0-20260715/hybrid_500k",
    )
    return parser.parse_args()


def _row(
    candidate: MergedCandidate, transformer: np.memmap, random: np.memmap
) -> np.void:
    records = transformer if candidate.method == "transformer" else random
    return records[candidate.index]


def _formal_key(
    candidate: MergedCandidate, transformer: np.memmap, random: np.memmap
) -> tuple[float, int, bytes]:
    row = _row(candidate, transformer, random)
    return (
        -float(row["reward"]),
        int(row["token_len"]),
        candidate.canonical_hash,
    )


def _heap_quality(
    candidate: MergedCandidate, transformer: np.memmap, random: np.memmap
) -> tuple[float, int, int]:
    row = _row(candidate, transformer, random)
    return (
        float(row["reward"]),
        -int(row["token_len"]),
        -int.from_bytes(candidate.canonical_hash, "big"),
    )


def _add_retained(
    heaps: list[list[tuple[tuple[float, int, int], int, MergedCandidate]]],
    candidate: MergedCandidate,
    transformer: np.memmap,
    random: np.memmap,
    serial: int,
) -> None:
    row = _row(candidate, transformer, random)
    bucket_index = _bucket(int(row["token_len"]))
    entry = (_heap_quality(candidate, transformer, random), serial, candidate)
    heap = heaps[bucket_index]
    if len(heap) < RETAINED_PER_BUCKET:
        heapq.heappush(heap, entry)
    elif entry[0] > heap[0][0]:
        heapq.heapreplace(heap, entry)


def _choose_cross_source(
    transformer: np.memmap,
    transformer_index: int,
    random: np.memmap,
    random_index: int,
) -> tuple[str, int]:
    transformer_row = transformer[transformer_index]
    random_row = random[random_index]
    transformer_reward = float(transformer_row["reward"])
    random_reward = float(random_row["reward"])
    if random_reward > transformer_reward:
        return "matched_random", random_index
    if random_reward < transformer_reward:
        return "transformer", transformer_index
    if int(random_row["token_len"]) < int(transformer_row["token_len"]):
        return "matched_random", random_index
    return "transformer", transformer_index


def _build_candidate(
    candidate: MergedCandidate,
    transformer: np.memmap,
    random: np.memmap,
    *,
    seed: int,
) -> CandidateRecord:
    records = transformer if candidate.method == "transformer" else random
    return _candidate_record(
        records,
        candidate.index,
        method=candidate.method,
        seed=seed,
    )


def _top50(
    retained: list[MergedCandidate],
    transformer: np.memmap,
    random: np.memmap,
    *,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    retained.sort(key=lambda item: _formal_key(item, transformer, random))
    records: list[CandidateRecord] = []
    provenance: dict[str, str] = {}
    for candidate in retained:
        record = _build_candidate(candidate, transformer, random, seed=seed)
        records.append(record)
        provenance[record.formula_hash] = candidate.provenance
    display = {
        record.formula_hash: display_formula_metrics(record.token_names)
        for record in records
    }
    curated, _ = select_curated_records(
        records,
        size=TOP_SIZE,
        display_metrics=display,
        tolerance=CURATED_REWARD_TOLERANCE,
    )
    rewards = np.asarray([record.reward for record in curated], dtype=np.float64)
    provenance_counts = {
        label: sum(provenance[record.formula_hash] == label for record in curated)
        for label in ("transformer_only", "random_only", "both")
    }
    formulas = [
        {
            "rank": rank,
            "formula_hash": record.formula_hash,
            "reward": record.reward,
            "token_names": list(record.token_names),
            "simplified_formula": display[record.formula_hash]["simplified_formula_text"],
            "provenance": provenance[record.formula_hash],
        }
        for rank, record in enumerate(curated, start=1)
    ]
    return (
        {
            "curated_top50_mean_reward": float(rewards.mean()),
            "curated_top50_median_reward": float(np.median(rewards)),
            "curated_top50_rank50_reward": float(rewards[-1]),
            "curated_top50_provenance_counts": provenance_counts,
        },
        formulas,
    )


def _gate0_run(gate0: dict[str, Any], method: str, seed: int) -> dict[str, Any]:
    return next(
        run
        for run in gate0["runs"]
        if run["method"] == method and int(run["seed"]) == seed
    )


def _analyze_seed(
    *,
    input_dir: Path,
    gate0: dict[str, Any],
    seed: int,
    expected: dict[tuple[str, int], str],
) -> dict[str, Any]:
    transformer_path = input_dir / _run_id("transformer", seed) / "attempts.bin"
    random_path = input_dir / _run_id("matched_random", seed) / "attempts.bin"
    print(f"[seed {seed}] validating ledgers", flush=True)
    transformer = _validate_ledger(transformer_path, expected[("transformer", seed)])
    random = _validate_ledger(random_path, expected[("matched_random", seed)])

    print(f"[seed {seed}] deduplicating fixed prefixes", flush=True)
    transformer_best = _best_valid_rows_in_segment(
        transformer, 0, TRANSFORMER_ATTEMPTS
    )
    random_best = _best_valid_rows_in_segment(random, 0, RANDOM_ATTEMPTS)
    transformer_by_hash = {
        transformer[index]["canonical_hash"].tobytes(): int(index)
        for index in transformer_best
    }
    heaps: list[
        list[tuple[tuple[float, int, int], int, MergedCandidate]]
    ] = [[], [], []]
    overlap_count = 0
    serial = 0
    for value in random_best:
        random_index = int(value)
        canonical_hash = random[random_index]["canonical_hash"].tobytes()
        transformer_index = transformer_by_hash.pop(canonical_hash, None)
        if transformer_index is None:
            candidate = MergedCandidate(
                "matched_random", random_index, "random_only", canonical_hash
            )
        else:
            overlap_count += 1
            method, index = _choose_cross_source(
                transformer, transformer_index, random, random_index
            )
            candidate = MergedCandidate(method, index, "both", canonical_hash)
        _add_retained(heaps, candidate, transformer, random, serial)
        serial += 1
    transformer_only_count = len(transformer_by_hash)
    for canonical_hash, transformer_index in transformer_by_hash.items():
        candidate = MergedCandidate(
            "transformer", transformer_index, "transformer_only", canonical_hash
        )
        _add_retained(heaps, candidate, transformer, random, serial)
        serial += 1

    retained = [entry[2] for heap in heaps for entry in heap]
    hybrid, formulas = _top50(retained, transformer, random, seed=seed)
    random_gate0 = _gate0_run(gate0, "matched_random", seed)
    random_7_5m = next(
        row
        for row in random_gate0["curves"]
        if row["attempt_count"] == RANDOM_ATTEMPTS
    )
    random_8m = random_gate0["final"]
    hybrid_reward = float(hybrid["curated_top50_mean_reward"])
    random_7_5m_reward = float(random_7_5m["curated_top50_mean_reward"])
    random_8m_reward = float(random_8m["curated_top50_mean_reward"])
    result = {
        "seed": seed,
        "budget": {
            "transformer_attempts": TRANSFORMER_ATTEMPTS,
            "matched_random_attempts": RANDOM_ATTEMPTS,
            "total_attempts": TOTAL_ATTEMPTS,
        },
        "semantic_valid_canonical": {
            "transformer_prefix": int(transformer_best.size),
            "random_prefix": int(random_best.size),
            "overlap": overlap_count,
            "transformer_only": transformer_only_count,
            "hybrid_union": int(random_best.size + transformer_only_count),
        },
        "random_7_5m_curated_top50_mean_reward": random_7_5m_reward,
        "hybrid": hybrid,
        "random_8m_curated_top50_mean_reward": random_8m_reward,
        "transformer_increment_over_random_7_5m_bps": (
            hybrid_reward - random_7_5m_reward
        )
        * 10_000.0,
        "random_last_500k_increment_bps": (random_8m_reward - random_7_5m_reward)
        * 10_000.0,
        "hybrid_minus_random_8m_bps": (hybrid_reward - random_8m_reward)
        * 10_000.0,
        "hybrid_beats_random_8m": hybrid_reward > random_8m_reward,
        "hybrid_curated_top50": formulas,
    }
    del transformer
    del random
    return result


def _decision(results: list[dict[str, Any]]) -> dict[str, Any]:
    differences = [float(item["hybrid_minus_random_8m_bps"]) for item in results]
    wins = sum(bool(item["hybrid_beats_random_8m"]) for item in results)
    mean_difference = float(np.mean(differences))
    if mean_difference > 0.0 and wins >= 2:
        classification = "transformer_prefix_has_practical_search_value"
        conclusion = (
            "Transformer前50万次带来的候选价值，足以弥补少跑50万次Random；"
            "当前Transformer值得作为混合搜索的前期通道保留。"
        )
    elif mean_difference < 0.0 and wins <= 1:
        classification = "transformer_prefix_not_worth_fixed_budget"
        conclusion = (
            "Transformer前50万次带来的候选价值，不足以弥补少跑50万次Random；"
            "当前Transformer不值得直接占用固定搜索预算，应先修改学习方式。"
        )
    else:
        classification = "mixed_seed_dependent"
        conclusion = (
            "Transformer前50万次的价值依赖seed，尚不能作为稳定的混合搜索通道。"
        )
    return {
        "classification": classification,
        "hybrid_wins": wins,
        "pair_count": len(results),
        "mean_hybrid_minus_random_8m_bps": mean_difference,
        "conclusion": conclusion,
        "validation_or_final_metrics_read": False,
    }


def _pct(value: float) -> str:
    return f"{100.0 * value:.4f}%"


def _write_report(
    path: Path, results: list[dict[str, Any]], decision: dict[str, Any]
) -> None:
    lines = [
        "# Stage D 固定混合搜索诊断",
        "",
        f"生成时间：{datetime.now(timezone.utc).isoformat()}",
        "",
        "> 每个seed固定使用Transformer前50万次和matched random前750万次，总预算800万；只读取训练期ledger。",
        "",
        "## 结论",
        "",
        decision["conclusion"],
        f"混合方案相对纯Random 800万次的三seed平均差为 {decision['mean_hybrid_minus_random_8m_bps']:.2f} bp，赢 {decision['hybrid_wins']}/3。",
        "",
        "## 配对结果",
        "",
        "| Seed | Random 750万 | 加入Transformer 50万 | Random 800万 | T带来的增益 | Random最后50万增益 | 混合-Random 800万 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in results:
        lines.append(
            f"| {item['seed']} | "
            f"{_pct(item['random_7_5m_curated_top50_mean_reward'])} | "
            f"{_pct(item['hybrid']['curated_top50_mean_reward'])} | "
            f"{_pct(item['random_8m_curated_top50_mean_reward'])} | "
            f"{item['transformer_increment_over_random_7_5m_bps']:.2f} bp | "
            f"{item['random_last_500k_increment_bps']:.2f} bp | "
            f"{item['hybrid_minus_random_8m_bps']:.2f} bp |"
        )
    lines.extend(
        [
            "",
            "## 混合top50来源",
            "",
            "| Seed | 仅Transformer发现 | 仅Random发现 | 双方都发现 |",
            "|---:|---:|---:|---:|",
        ]
    )
    for item in results:
        counts = item["hybrid"]["curated_top50_provenance_counts"]
        lines.append(
            f"| {item['seed']} | {counts['transformer_only']} | "
            f"{counts['random_only']} | {counts['both']} |"
        )
    lines.extend(
        [
            "",
            "## 判断含义",
            "",
            "- 这不是寻找最佳混合比例，只检验一个固定问题：当前Transformer前50万次是否值得替代Random的50万次预算。",
            "- 混合库先跨来源按canonical去重，再复用formal的top500长度桶与curated top50规则。",
            "- 结果只说明训练期搜索价值，不说明公式可交易，也不授权打开validation。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = _parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    gate0 = json.loads(args.gate0_results.read_text(encoding="utf-8"))
    if gate0["formal_identity"]["validation_or_final_metrics_read"]:
        raise RuntimeError("Gate 0 artifact indicates validation/final data was read")
    expected = {(method, seed): sha for method, seed, sha in RUN_SPECS}
    results = [
        _analyze_seed(
            input_dir=input_dir,
            gate0=gate0,
            seed=seed,
            expected=expected,
        )
        for seed in (101, 102, 103)
    ]
    decision = _decision(results)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "design": {
            "transformer_attempts": TRANSFORMER_ATTEMPTS,
            "matched_random_attempts": RANDOM_ATTEMPTS,
            "total_attempts": TOTAL_ATTEMPTS,
            "curation_rule_version": CURATION_RULE_VERSION,
            "curation_reward_tolerance": CURATED_REWARD_TOLERANCE,
            "ratio_was_not_optimized": True,
        },
        "results": results,
        "decision": decision,
    }
    json_path = output_dir / "hybrid_500k_results.json"
    report_path = output_dir / "hybrid_500k_report.md"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_report(report_path, results, decision)
    checksums = [
        f"{_sha256(json_path)}  {json_path.name}",
        f"{_sha256(report_path)}  {report_path.name}",
    ]
    (output_dir / "SHA256SUMS").write_text(
        "\n".join(checksums) + "\n", encoding="ascii"
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()