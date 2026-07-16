#!/usr/bin/env python3
"""Read-only Gate 0 diagnostics for the six frozen Stage D formal ledgers."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
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

from alpha_etf.research_v3a.attempts import (  # noqa: E402
    ATTEMPT_DTYPE,
    AttemptStatus,
)
from alpha_etf.research_v3a.candidates import (  # noqa: E402
    CandidateRecord,
    build_candidate_record,
)
from scripts.v3a.export_top_formulas import (  # noqa: E402
    select_curated_records,
)
from scripts.v3a.preview_formula_curation import (  # noqa: E402
    CURATED_REWARD_TOLERANCE,
    CURATION_RULE_VERSION,
    display_formula_metrics,
)


SCHEMA_VERSION = "v3a-stage-d-gate0-diagnostics-v1"
ATTEMPTS = 8_000_000
CHECKPOINT_INTERVAL = 100_000
RETAINED_PER_BUCKET = 500
TOP_SIZE = 50
RUN_PREFIX = "v3a-stage-d-formal"
RUN_SPECS = (
    ("transformer", 101, "0df6c3885b9207f02401bfb6ab589e0fa5eb34b6de1831b15dfad91500c7e6a8"),
    ("transformer", 102, "d43fe7e4b3a8b69dd9ceace834eebe64e7091aec20097ca1331827065ec3c11e"),
    ("transformer", 103, "fc54abb3cd254c4320f75760fce52ba1114b888d4a60c22ce90e7e8368b01b1d"),
    ("matched_random", 101, "c7126e328d16ace125efddb8cd556b2270dce60a87232748faca39563767af08"),
    ("matched_random", 102, "f543847c84a22cf9c35527f0c43c63a765ddf6dadfb3f1578e2eb876b33c823e"),
    ("matched_random", 103, "24ae2c3e8e28458d457ca27fe5332d394657def9a7050cdf9fc4aa663fe3c5b1"),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / ".pi/profile/results/v3a-gate0-20260715/inputs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / ".pi/profile/results/v3a-gate0-20260715",
    )
    return parser.parse_args()


def _run_id(method: str, seed: int) -> str:
    return f"{RUN_PREFIX}-{method}-s{seed}-02cca48d1c85"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_ledger(path: Path, expected_sha256: str) -> np.memmap:
    expected_size = ATTEMPTS * ATTEMPT_DTYPE.itemsize
    if path.stat().st_size != expected_size:
        raise RuntimeError(f"{path} size differs from {expected_size}")
    actual_sha256 = _sha256(path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(f"{path} SHA-256 differs: {actual_sha256}")
    records = np.memmap(path, dtype=ATTEMPT_DTYPE, mode="r")
    for start in range(0, ATTEMPTS, 1_000_000):
        stop = min(start + 1_000_000, ATTEMPTS)
        expected = np.arange(start, stop, dtype=np.uint64)
        if not np.array_equal(records["attempt_index"][start:stop], expected):
            raise RuntimeError(f"{path} attempt indices are not contiguous at {start}")
    if np.any(records["status"] == int(AttemptStatus.PENDING_SEMANTIC_VALID)):
        raise RuntimeError(f"{path} contains unresolved attempt statuses")
    return records


def _first_occurrences(values: np.ndarray) -> np.ndarray:
    _, first = np.unique(values, return_index=True)
    return np.sort(first.astype(np.int64, copy=False))


def _canonical_first_occurrences(records: np.memmap) -> np.ndarray:
    hashes = np.ascontiguousarray(records["canonical_hash"]).view("S32").reshape(-1)
    return _first_occurrences(hashes)


def _exact_first_occurrences(records: np.memmap) -> np.ndarray:
    keys = np.empty((records.shape[0], 16), dtype=np.uint8)
    keys[:, 0] = records["token_len"]
    keys[:, 1:] = records["token_ids"]
    return _first_occurrences(keys.view("S16").reshape(-1))


def _bucket(token_len: int) -> int:
    if token_len <= 5:
        return 0
    if token_len <= 10:
        return 1
    return 2


def _record_sort_key(records: np.memmap, index: int) -> tuple[float, int, bytes]:
    row = records[index]
    return (
        -float(row["reward"]),
        int(row["token_len"]),
        row["canonical_hash"].tobytes(),
    )


def _is_better(records: np.memmap, new_index: int, old_index: int) -> bool:
    new = records[new_index]
    old = records[old_index]
    new_reward = float(new["reward"])
    old_reward = float(old["reward"])
    if new_reward != old_reward:
        return new_reward > old_reward
    return int(new["token_len"]) < int(old["token_len"])


def _best_valid_rows_in_segment(
    records: np.memmap, start: int, stop: int
) -> np.ndarray:
    statuses = records["status"][start:stop]
    local = np.flatnonzero(statuses >= int(AttemptStatus.CANONICAL_DUPLICATE))
    if not local.size:
        return np.empty(0, dtype=np.int64)
    indices = local.astype(np.int64, copy=False) + start
    hashes = np.ascontiguousarray(records["canonical_hash"][indices]).view("S32").reshape(-1)
    rewards = records["reward"][indices]
    lengths = records["token_len"][indices]
    order = np.lexsort((indices, lengths, -rewards, hashes))
    sorted_hashes = hashes[order]
    starts = np.r_[0, np.flatnonzero(sorted_hashes[1:] != sorted_hashes[:-1]) + 1]
    return indices[order[starts]]


def _update_candidate_heaps(
    records: np.memmap,
    states: dict[bytes, int],
    heaps: list[list[tuple[tuple[float, int, bytes], bytes, int]]],
    candidates: np.ndarray,
) -> None:
    for value in candidates:
        index = int(value)
        row = records[index]
        key = row["canonical_hash"].tobytes()
        old_index = states.get(key)
        if old_index is not None and not _is_better(records, index, old_index):
            continue
        states[key] = index
        bucket_index = _bucket(int(row["token_len"]))
        heapq.heappush(
            heaps[bucket_index],
            (_record_sort_key(records, index), key, index),
        )


def _retained_indices(
    records: np.memmap,
    states: dict[bytes, int],
    heaps: list[list[tuple[tuple[float, int, bytes], bytes, int]]],
) -> list[int]:
    retained: list[int] = []
    for bucket_index, heap in enumerate(heaps):
        selected: list[tuple[tuple[float, int, bytes], bytes, int]] = []
        while heap and len(selected) < RETAINED_PER_BUCKET:
            entry = heapq.heappop(heap)
            _, key, index = entry
            if states.get(key) != index:
                continue
            if _bucket(int(records["token_len"][index])) != bucket_index:
                continue
            selected.append(entry)
            retained.append(index)
        for entry in selected:
            heapq.heappush(heap, entry)
    return retained


def _candidate_record(
    records: np.memmap,
    index: int,
    *,
    method: str,
    seed: int,
) -> CandidateRecord:
    row = records[index]
    token_len = int(row["token_len"])
    token_ids = [int(value) for value in row["token_ids"][:token_len]]
    candidate = build_candidate_record(
        formula_id=f"gate0_{method}_s{seed}_a{index}",
        source=f"v3a_stage_d_{method}",
        token_ids=token_ids,
        reward=float(row["reward"]),
        train_summary={
            "coverage": float(row["coverage"]),
            "finite_std": float(row["finite_std"]),
            "training_step": int(row["training_step"]),
        },
        attempt_index=index,
    )
    ledger_hash = row["canonical_hash"].tobytes().hex()
    if candidate.formula_hash != ledger_hash:
        raise RuntimeError(f"Rebuilt candidate hash differs at attempt {index}")
    return replace(candidate, best_attempt_index=index)


def _top50_metrics(
    records: np.memmap,
    states: dict[bytes, int],
    heaps: list[list[tuple[tuple[float, int, bytes], bytes, int]]],
    cache: dict[int, CandidateRecord],
    *,
    method: str,
    seed: int,
) -> dict[str, float]:
    indices = _retained_indices(records, states, heaps)
    for index in indices:
        if index not in cache:
            cache[index] = _candidate_record(records, index, method=method, seed=seed)
    candidates = sorted(
        (cache[index] for index in indices),
        key=lambda item: (-item.reward, item.token_len, item.formula_hash),
    )
    if len(candidates) < TOP_SIZE:
        raise RuntimeError(f"Only {len(candidates)} retained candidates at checkpoint")
    raw = candidates[:TOP_SIZE]
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

    def summary(values: list[CandidateRecord], prefix: str) -> dict[str, float]:
        rewards = np.asarray([item.reward for item in values], dtype=np.float64)
        return {
            f"{prefix}_mean_reward": float(rewards.mean()),
            f"{prefix}_median_reward": float(np.median(rewards)),
            f"{prefix}_rank50_reward": float(rewards[-1]),
        }

    return {**summary(raw, "raw_top50"), **summary(curated, "curated_top50")}


def _count_before(first_occurrences: np.ndarray, attempt_count: int) -> int:
    return int(np.searchsorted(first_occurrences, attempt_count, side="left"))


def _analyze_run(
    *,
    path: Path,
    method: str,
    seed: int,
    expected_sha256: str,
    target_unique: int | None,
) -> dict[str, Any]:
    print(f"[{method} s{seed}] validating ledger", flush=True)
    records = _validate_ledger(path, expected_sha256)
    print(f"[{method} s{seed}] computing canonical growth", flush=True)
    canonical_first = _canonical_first_occurrences(records)
    canonical_unique = int(canonical_first.size)
    cutoff = None
    if target_unique is not None:
        if target_unique > canonical_unique:
            raise RuntimeError(f"Random s{seed} does not reach target canonical unique count")
        cutoff = int(canonical_first[target_unique - 1]) + 1
    print(f"[{method} s{seed}] computing exact-token growth", flush=True)
    exact_first = _exact_first_occurrences(records)

    regular = list(range(CHECKPOINT_INTERVAL, ATTEMPTS + 1, CHECKPOINT_INTERVAL))
    endpoints = sorted(set(regular + ([cutoff] if cutoff is not None else [])))
    valid_prefix = np.cumsum(
        records["status"] >= int(AttemptStatus.CANONICAL_DUPLICATE), dtype=np.uint64
    )
    accepted_prefix = np.cumsum(
        records["status"] == int(AttemptStatus.ACCEPTED_UNIQUE), dtype=np.uint64
    )
    states: dict[bytes, int] = {}
    heaps: list[list[tuple[tuple[float, int, bytes], bytes, int]]] = [[], [], []]
    cache: dict[int, CandidateRecord] = {}
    curves: list[dict[str, Any]] = []
    previous_endpoint = 0
    previous_canonical = 0
    previous_exact = 0
    previous_raw_mean: float | None = None
    last_raw_improvement_attempt: int | None = None

    print(f"[{method} s{seed}] building cumulative top50 frontier", flush=True)
    for endpoint in endpoints:
        candidates = _best_valid_rows_in_segment(records, previous_endpoint, endpoint)
        _update_candidate_heaps(records, states, heaps, candidates)
        top = _top50_metrics(
            records, states, heaps, cache, method=method, seed=seed
        )
        current_canonical = _count_before(canonical_first, endpoint)
        current_exact = _count_before(exact_first, endpoint)
        interval_attempts = endpoint - previous_endpoint
        interval_canonical = current_canonical - previous_canonical
        interval_exact = current_exact - previous_exact
        raw_mean = float(top["raw_top50_mean_reward"])
        if previous_raw_mean is None or raw_mean > previous_raw_mean:
            last_raw_improvement_attempt = endpoint
        previous_raw_mean = raw_mean
        curves.append(
            {
                "method": method,
                "seed": seed,
                "attempt_count": endpoint,
                "checkpoint_kind": "same_unique_cutoff" if endpoint == cutoff else "regular",
                "canonical_unique_count": current_canonical,
                "canonical_unique_rate": current_canonical / endpoint,
                "canonical_duplicate_rate": 1.0 - current_canonical / endpoint,
                "interval_canonical_unique_count": interval_canonical,
                "interval_canonical_unique_rate": interval_canonical / interval_attempts,
                "interval_canonical_duplicate_rate": 1.0 - interval_canonical / interval_attempts,
                "exact_unique_count": current_exact,
                "exact_unique_rate": current_exact / endpoint,
                "exact_duplicate_rate": 1.0 - current_exact / endpoint,
                "interval_exact_unique_count": interval_exact,
                "interval_exact_unique_rate": interval_exact / interval_attempts,
                "semantic_valid_count": int(valid_prefix[endpoint - 1]),
                "accepted_unique_count": int(accepted_prefix[endpoint - 1]),
                **top,
            }
        )
        previous_endpoint = endpoint
        previous_canonical = current_canonical
        previous_exact = current_exact

    final = next(item for item in curves if item["attempt_count"] == ATTEMPTS)
    cutoff_metrics = (
        next(item for item in curves if item["attempt_count"] == cutoff)
        if cutoff is not None
        else None
    )
    regular_curves = [item for item in curves if item["checkpoint_kind"] == "regular"]
    at_one_million = next(
        item for item in regular_curves if item["attempt_count"] == 1_000_000
    )
    at_seven_million = next(
        item for item in regular_curves if item["attempt_count"] == 7_000_000
    )

    def attempts_to_fraction(fraction: float) -> int:
        target = float(final["canonical_unique_count"]) * fraction
        return int(
            next(
                item["attempt_count"]
                for item in regular_curves
                if item["canonical_unique_count"] >= target
            )
        )

    growth_summary = {
        "canonical_unique_first_1m": int(at_one_million["canonical_unique_count"]),
        "canonical_unique_last_1m": int(
            final["canonical_unique_count"] - at_seven_million["canonical_unique_count"]
        ),
        "attempts_to_50pct_final_canonical_unique": attempts_to_fraction(0.50),
        "attempts_to_90pct_final_canonical_unique": attempts_to_fraction(0.90),
        "attempts_to_95pct_final_canonical_unique": attempts_to_fraction(0.95),
        "last_100k_canonical_unique_count": int(
            final["interval_canonical_unique_count"]
        ),
        "last_100k_canonical_duplicate_rate": float(
            final["interval_canonical_duplicate_rate"]
        ),
        "curated_top50_mean_reward_at_1m": float(
            at_one_million["curated_top50_mean_reward"]
        ),
        "curated_top50_gain_after_1m_bps": float(
            (
                final["curated_top50_mean_reward"]
                - at_one_million["curated_top50_mean_reward"]
            )
            * 10_000.0
        ),
    }
    del records
    return {
        "run_id": _run_id(method, seed),
        "method": method,
        "seed": seed,
        "input": {
            "path": str(path.resolve()),
            "bytes": ATTEMPTS * ATTEMPT_DTYPE.itemsize,
            "sha256": expected_sha256,
        },
        "final": final,
        "same_unique_target": target_unique,
        "same_unique_cutoff_attempt": cutoff,
        "same_unique_cutoff": cutoff_metrics,
        "sampled_last_raw_top50_improvement_attempt": last_raw_improvement_attempt,
        "growth_summary": growth_summary,
        "curves": curves,
    }


def _paired_results(runs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    lookup = {(item["method"], item["seed"]): item for item in runs}
    paired: list[dict[str, Any]] = []
    for seed in (101, 102, 103):
        transformer = lookup[("transformer", seed)]
        random = lookup[("matched_random", seed)]
        transformer_metric = float(transformer["final"]["curated_top50_mean_reward"])
        random_metric = float(random["same_unique_cutoff"]["curated_top50_mean_reward"])
        random_full_metric = float(random["final"]["curated_top50_mean_reward"])
        difference = transformer_metric - random_metric
        paired.append(
            {
                "seed": seed,
                "canonical_unique_target": int(transformer["final"]["canonical_unique_count"]),
                "random_cutoff_attempt": int(random["same_unique_cutoff_attempt"]),
                "random_cutoff_semantic_valid_count": int(
                    random["same_unique_cutoff"]["semantic_valid_count"]
                ),
                "random_cutoff_accepted_unique_count": int(
                    random["same_unique_cutoff"]["accepted_unique_count"]
                ),
                "transformer_curated_top50_mean_reward": transformer_metric,
                "random_curated_top50_mean_reward": random_metric,
                "difference_reward": difference,
                "difference_bps": difference * 10_000.0,
                "full_budget_difference_bps": (
                    transformer_metric - random_full_metric
                )
                * 10_000.0,
                "transformer_win": difference > 0.0,
            }
        )
    mean_difference = float(np.mean([item["difference_reward"] for item in paired]))
    full_budget_mean_difference_bps = float(
        np.mean([item["full_budget_difference_bps"] for item in paired])
    )
    wins = sum(bool(item["transformer_win"]) for item in paired)
    if mean_difference > 0.0 and wins >= 2:
        classification = "coverage_repetition_primary"
        conclusion = (
            "同 canonical 唯一数下 Transformer 平均领先且至少赢 2/3；"
            "搜索方向有正价值，固定 attempt 失败主要来自覆盖收缩和重复浪费。"
        )
    elif mean_difference < 0.0 and wins <= 1:
        classification = "distribution_quality_also_wrong"
        conclusion = (
            "同 canonical 唯一数下 Transformer 平均仍落后且最多赢 1/3；"
            "覆盖不足不是唯一原因，学习后的分布本身也偏离高分尾部。"
        )
    else:
        classification = "mixed_seed_dependent"
        conclusion = "平均方向与胜场规则不一致；证据混合，不能把根因简化为单一问题。"
    return paired, {
        "classification": classification,
        "paired_mean_difference_reward": mean_difference,
        "paired_mean_difference_bps": mean_difference * 10_000.0,
        "full_budget_paired_mean_difference_bps": full_budget_mean_difference_bps,
        "difference_reduction_bps": (
            mean_difference * 10_000.0 - full_budget_mean_difference_bps
        ),
        "transformer_wins": wins,
        "pair_count": len(paired),
        "conclusion": conclusion,
        "scope": "training_only; no validation or final data read",
    }


def _write_csv(path: Path, runs: list[dict[str, Any]]) -> None:
    rows = [row for run in runs for row in run["curves"]]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(path: Path, runs: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {101: "#0072B2", 102: "#D55E00", 103: "#009E73"}
    styles = {"transformer": "-", "matched_random": "--"}
    labels = {"transformer": "Transformer", "matched_random": "Random"}
    figure, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True)
    for run in runs:
        regular = [row for row in run["curves"] if row["checkpoint_kind"] == "regular"]
        x = np.asarray([row["attempt_count"] for row in regular]) / 1_000_000.0
        method = str(run["method"])
        seed = int(run["seed"])
        plot_args = {
            "color": colors[seed],
            "linestyle": styles[method],
            "linewidth": 1.7,
            "label": f"{labels[method]} s{seed}",
        }
        axes[0].plot(x, [row["canonical_unique_count"] for row in regular], **plot_args)
        axes[1].plot(
            x,
            [100.0 * row["interval_canonical_duplicate_rate"] for row in regular],
            **plot_args,
        )
        axes[2].plot(
            x,
            [100.0 * row["curated_top50_mean_reward"] for row in regular],
            **plot_args,
        )
    axes[0].set_ylabel("Cumulative canonical unique")
    axes[1].set_ylabel("Window duplicate rate (%)")
    axes[2].set_ylabel("Curated top50 mean reward (%)")
    axes[2].set_xlabel("Attempts (million)")
    axes[0].legend(ncol=2, fontsize=9)
    for axis in axes:
        axis.grid(True, color="#d9d9d9", linewidth=0.6)
    figure.suptitle("V3A Stage D Gate 0 training-ledger diagnostics")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _pct(value: float) -> str:
    return f"{100.0 * value:.4f}%"


def _write_report(
    path: Path,
    runs: list[dict[str, Any]],
    paired: list[dict[str, Any]],
    decision: dict[str, Any],
) -> None:
    lines = [
        "# V3A Stage D Gate 0 只读诊断",
        "",
        f"生成时间：{datetime.now(timezone.utc).isoformat()}",
        "",
        "> 仅使用六个 formal run 的 2016-08-09 至 2021-12-31 训练期 attempts.bin；未读取 2022 validation 或 2023+ final。",
        "",
        "## 结论",
        "",
        decision["conclusion"],
        f"完整800万attempt的三组平均差为 {decision['full_budget_paired_mean_difference_bps']:.2f} bp；"
        f"匹配canonical唯一数后缩小到 {decision['paired_mean_difference_bps']:.2f} bp，Transformer 胜 {decision['transformer_wins']}/3。",
        "",
        "## 完整预算终点",
        "",
        "| 方法 | Seed | exact唯一数 | exact重复率 | canonical唯一数 | canonical重复率 | curated top50均值 | 最后一次采样前沿改善 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in runs:
        final = run["final"]
        lines.append(
            f"| {run['method']} | {run['seed']} | {final['exact_unique_count']:,} | "
            f"{100.0 * final['exact_duplicate_rate']:.2f}% | "
            f"{final['canonical_unique_count']:,} | "
            f"{100.0 * final['canonical_duplicate_rate']:.2f}% | "
            f"{_pct(final['curated_top50_mean_reward'])} | "
            f"{run['sampled_last_raw_top50_improvement_attempt']:,} |"
        )
    lines.extend(
        [
            "",
            "## 唯一增长与前沿停滞",
            "",
            "| 方法 | Seed | 前100万新增canonical | 后100万新增canonical | 达到最终95%的attempt | 100万attempt后的top50增益 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for run in runs:
        growth = run["growth_summary"]
        lines.append(
            f"| {run['method']} | {run['seed']} | "
            f"{growth['canonical_unique_first_1m']:,} | "
            f"{growth['canonical_unique_last_1m']:,} | "
            f"{growth['attempts_to_95pct_final_canonical_unique']:,} | "
            f"{growth['curated_top50_gain_after_1m_bps']:.2f} bp |"
        )
    lines.extend(
        [
            "",
            "## 同 canonical 唯一数配对",
            "",
            "| Seed | 唯一数目标 | Random截断attempt | Random有效样本 | Random accepted唯一 | Transformer top50 | Random top50 | T-R (bp) |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in paired:
        lines.append(
            f"| {item['seed']} | {item['canonical_unique_target']:,} | "
            f"{item['random_cutoff_attempt']:,} | {item['random_cutoff_semantic_valid_count']:,} | "
            f"{item['random_cutoff_accepted_unique_count']:,} | "
            f"{_pct(item['transformer_curated_top50_mean_reward'])} | "
            f"{_pct(item['random_curated_top50_mean_reward'])} | {item['difference_bps']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## 口径",
            "",
            "- canonical唯一数覆盖全部attempt；top50只使用语义有效状态，并对canonical取截至当时最佳reward。",
            "- 每10万attempt输出累计唯一数、累计/区间重复率，以及raw/curated top50前沿。",
            f"- curated复用 `{CURATION_RULE_VERSION}` 和 `{CURATED_REWARD_TOLERANCE:g}` reward容差。",
            "- Random按attempt顺序在首次达到同seed Transformer最终canonical唯一数时截断。",
            "- 同canonical唯一数匹配包含无效公式；因此另外列出Random截断时的语义有效数和accepted唯一数，避免误读为有效候选预算也相同。",
            "- 三个seed只做预注册式配对方向判断，不做显著性检验。",
            "",
            "## 数据边界",
            "",
            "Gate 0是训练期机制诊断，不授权新训练，也不构成样本外投资结论。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_checksums(output_dir: Path, names: list[str]) -> None:
    lines = [f"{_sha256(output_dir / name)}  {name}" for name in names]
    (output_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="ascii")


def main() -> None:
    args = _parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    specs = {(method, seed): sha for method, seed, sha in RUN_SPECS}
    runs: list[dict[str, Any]] = []
    transformer_targets: dict[int, int] = {}
    for method in ("transformer", "matched_random"):
        for seed in (101, 102, 103):
            run_id = _run_id(method, seed)
            run = _analyze_run(
                path=input_dir / run_id / "attempts.bin",
                method=method,
                seed=seed,
                expected_sha256=specs[(method, seed)],
                target_unique=(
                    transformer_targets[seed] if method == "matched_random" else None
                ),
            )
            runs.append(run)
            if method == "transformer":
                transformer_targets[seed] = int(run["final"]["canonical_unique_count"])

    paired, decision = _paired_results(runs)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "formal_identity": {
            "protocol_id": "02cca48d1c8536f90a23a0e0361cb45e64a30e95623afc05ef58bd57a0dca5c1",
            "code_commit": "0df9404331b43bc8775d7ff17dab1d3172ecf186",
            "validation_or_final_metrics_read": False,
        },
        "metric_policy": {
            "checkpoint_interval": CHECKPOINT_INTERVAL,
            "canonical_unique_includes_invalid_attempts": True,
            "semantic_statuses": [
                "canonical_duplicate",
                "selection_duplicate",
                "accepted_unique",
            ],
            "retained_per_length_bucket": RETAINED_PER_BUCKET,
            "curation_rule_version": CURATION_RULE_VERSION,
            "curation_reward_tolerance": CURATED_REWARD_TOLERANCE,
        },
        "runs": runs,
        "paired_same_canonical_unique": paired,
        "decision": decision,
    }
    json_path = output_dir / "gate0_results.json"
    csv_path = output_dir / "gate0_curves.csv"
    plot_path = output_dir / "gate0_curves.png"
    report_path = output_dir / "gate0_report.md"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(csv_path, runs)
    _plot(plot_path, runs)
    _write_report(report_path, runs, paired, decision)
    names = [json_path.name, csv_path.name, plot_path.name, report_path.name]
    _write_checksums(output_dir, names)
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()