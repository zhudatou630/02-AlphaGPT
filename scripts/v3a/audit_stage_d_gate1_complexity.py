#!/usr/bin/env python3
"""Audit Gate 1 formula complexity and within-train stability without future data."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import torch


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
from alpha_etf.research_v3a.candidates import CandidateConfig  # noqa: E402
from alpha_etf.research_v3a.language import (  # noqa: E402
    Expression,
    FORMULA_VOCAB,
    compile_formula,
)
from alpha_etf.research_v3a.scoring import (  # noqa: E402
    ScorerConfig,
    SplitSpec,
    build_forward_targets,
)
from alpha_etf.research_v3a.stage_d import load_stage_d_train_view  # noqa: E402
from alpha_etf.research_v3a.torch_scoring import (  # noqa: E402
    TorchForwardTargets,
    score_signal_batch,
    signal_quality_batch,
)
from alpha_etf.research_v3a.torch_vm import BatchTorchVM  # noqa: E402
from scripts.v3a.export_top_formulas import select_curated_records  # noqa: E402
from scripts.v3a.preview_formula_curation import (  # noqa: E402
    CURATED_REWARD_TOLERANCE,
    display_formula_metrics,
)


SCHEMA_VERSION = "v3a-stage-d-gate1-complexity-audit-v1"
PROTOCOL_PREFIX = "9f9cfbe379a7"
SEEDS = (101, 102, 103)
TOTAL_ATTEMPTS = 2_000_000
BATCH_SIZE = 8192
MODEL_BATCH_SIZE = 6144
EARLY_STOP = 500_000
LATE_START = 1_500_000
ARCHIVE_CHECKPOINT = 1_000_000
TAIL_FRACTION = 0.10
TAIL_SAMPLE_SIZE = 200
COMPLEX_POOL_SIZE = 500
TOP_SIZE = 50
FULL_REWARD_TOLERANCE = 1e-6
VALIDATION_END = "2021-12-31"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mechanism-runs-dir", type=Path, required=True)
    parser.add_argument("--train-view-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_ledger(run_dir: Path) -> np.memmap:
    marker = json.loads((run_dir / "training_complete.json").read_text(encoding="utf-8"))
    if int(marker["attempt_count"]) != TOTAL_ATTEMPTS:
        raise RuntimeError(f"Incomplete mechanism run: {run_dir}")
    path = run_dir / "attempts.bin"
    if path.stat().st_size != TOTAL_ATTEMPTS * ATTEMPT_DTYPE.itemsize:
        raise RuntimeError(f"Attempt ledger size mismatch: {path}")
    if _sha256(path) != marker["attempt_ledger_sha256"]:
        raise RuntimeError(f"Attempt ledger SHA mismatch: {path}")
    records = np.memmap(path, dtype=ATTEMPT_DTYPE, mode="r")
    if not np.array_equal(
        records["attempt_index"], np.arange(TOTAL_ATTEMPTS, dtype=np.uint64)
    ):
        raise RuntimeError(f"Attempt ledger order mismatch: {path}")
    return records


def _model_batches(records: np.ndarray):
    for start in range(0, len(records), BATCH_SIZE):
        stop = min(start + BATCH_SIZE, len(records))
        model_count = (stop - start) * 3 // 4
        yield records[start : start + model_count]


def _model_indices(stop: int = TOTAL_ATTEMPTS) -> np.ndarray:
    indices = np.arange(stop, dtype=np.int64)
    return indices[(indices % BATCH_SIZE) < MODEL_BATCH_SIZE]


def _valid_mask(records: np.ndarray) -> np.ndarray:
    return (
        (records["status"] >= int(AttemptStatus.CANONICAL_DUPLICATE))
        & np.isfinite(records["reward"])
    )


def _hash_view(records: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(records["canonical_hash"][indices]).view("S32").reshape(-1)


def _representative_indices(records: np.ndarray, indices: np.ndarray) -> np.ndarray:
    selected = np.asarray(indices, dtype=np.int64)
    selected = selected[_valid_mask(records[selected])]
    hashes = _hash_view(records, selected)
    order = np.lexsort(
        (
            records["attempt_index"][selected],
            records["token_len"][selected],
            -records["reward"][selected],
            hashes,
        )
    )
    ordered_hashes = hashes[order]
    starts = np.r_[0, np.flatnonzero(ordered_hashes[1:] != ordered_hashes[:-1]) + 1]
    return selected[order[starts]]


def _reward_order(records: np.ndarray, indices: np.ndarray) -> np.ndarray:
    selected = np.asarray(indices, dtype=np.int64)
    hashes = _hash_view(records, selected)
    order = np.lexsort(
        (hashes, records["token_len"][selected], -records["reward"][selected])
    )
    return selected[order]


def _row_token_ids(row: np.void) -> tuple[int, ...]:
    return tuple(int(value) for value in row["token_ids"][: int(row["token_len"])])


def _row_hash(row: np.void) -> str:
    return row["canonical_hash"].tobytes().hex()


def _curated_top_indices(records: np.ndarray, representatives: np.ndarray) -> np.ndarray:
    retained: list[int] = []
    lengths = records["token_len"][representatives]
    buckets = (lengths <= 5, (lengths > 5) & (lengths <= 10), lengths > 10)
    for mask in buckets:
        retained.extend(int(index) for index in _reward_order(records, representatives[mask])[:500])
    candidates = []
    index_by_hash: dict[str, int] = {}
    for index in retained:
        row = records[index]
        candidate = build_candidate_record(
            formula_id=f"audit_a{int(row['attempt_index'])}",
            source="v3a_gate1_complexity_audit",
            token_ids=_row_token_ids(row),
            reward=float(row["reward"]),
            train_summary={},
            attempt_index=int(row["attempt_index"]),
        )
        if candidate.formula_hash != _row_hash(row):
            raise RuntimeError("Rebuilt candidate hash differs from ledger")
        candidates.append(candidate)
        index_by_hash[candidate.formula_hash] = int(index)
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
    return np.asarray([index_by_hash[item.formula_hash] for item in curated], dtype=np.int64)


def _new_valid_tail_indices(records: np.ndarray) -> dict[str, np.ndarray]:
    state = ArchiveTailState(ArchiveTailConfig())
    by_stage: dict[str, list[int]] = {"early": [], "late": []}
    for batch in _model_batches(records):
        labels = state.apply(batch, prepared=False)
        if not np.allclose(
            batch["training_reward"], labels.combined_weights, rtol=0.0, atol=1e-7
        ):
            raise RuntimeError("Archive-tail labels do not replay during audit")
        for local_index in labels.new_valid_indices:
            attempt = int(batch[local_index]["attempt_index"])
            if attempt < EARLY_STOP:
                by_stage["early"].append(attempt)
            elif attempt >= LATE_START:
                by_stage["late"].append(attempt)
    result: dict[str, np.ndarray] = {}
    for stage, values in by_stage.items():
        indices = np.asarray(values, dtype=np.int64)
        ordered = _reward_order(records, indices)
        count = max(1, math.ceil(len(ordered) * TAIL_FRACTION))
        result[stage] = ordered[:count]
    return result


def _quantile_sample(indices: np.ndarray, size: int) -> np.ndarray:
    if len(indices) <= size:
        return indices.copy()
    positions = np.rint(np.linspace(0, len(indices) - 1, size)).astype(np.int64)
    return indices[positions]


def _expression_token_names(expression: Expression) -> list[str]:
    if expression.kind == "factor":
        return [expression.name, *(f"WIN_{int(value)}" for value in expression.params)]
    if expression.kind == "constant":
        return [expression.name]
    names: list[str] = []
    for child in expression.children:
        names.extend(_expression_token_names(child))
    names.extend(f"WIN_{int(value)}" for value in expression.params)
    names.append(expression.name)
    return names


def _proper_subexpressions(expression: Expression) -> Iterable[Expression]:
    for child in expression.children:
        if child.kind != "constant":
            yield child
        yield from _proper_subexpressions(child)


def _expression_features(expression: Expression) -> tuple[set[str], str]:
    features: set[str] = set()
    factors: set[str] = set()
    rolling: set[str] = set()

    def visit(node: Expression) -> None:
        if node.kind == "factor":
            instance = node.name + "".join(f"_{int(value)}" for value in node.params)
            factors.add(instance)
            features.add(f"factor:{instance}")
            for value in node.params:
                features.add(f"window:{int(value)}")
        elif node.kind == "constant":
            features.add("constant:any")
        else:
            features.add(f"operator:{node.name}")
            if node.name in {"MEAN", "REF"}:
                item = f"{node.name}_{int(node.params[0])}"
                rolling.add(item)
                features.add(f"rolling:{item}")
                features.add(f"window:{int(node.params[0])}")
        for child in node.children:
            visit(child)

    visit(expression)
    signature = "f=" + ",".join(sorted(factors)) + "|r=" + ",".join(sorted(rolling))
    return features, signature


def _feature_summary(records: np.ndarray, indices: np.ndarray) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    families: Counter[str] = Counter()
    for index in indices:
        expression = compile_formula(_row_token_ids(records[int(index)])).expression
        features, family = _expression_features(expression)
        counts.update(features)
        families[family] += 1
    total = len(indices)
    return {
        "count": total,
        "prevalence": {key: value / total for key, value in sorted(counts.items())},
        "top20_families": [
            {"signature": key, "count": count, "fraction": count / total}
            for key, count in families.most_common(20)
        ],
    }


def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
    keys = sorted(set(left) | set(right))
    a = np.asarray([left.get(key, 0.0) for key in keys], dtype=np.float64)
    b = np.asarray([right.get(key, 0.0) for key in keys], dtype=np.float64)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator else float("nan")


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else float("nan")


def _pairwise(values: list[Any], function) -> list[float]:
    return [
        float(function(values[left], values[right]))
        for left, right in ((0, 1), (0, 2), (1, 2))
    ]


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    def ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="stable")
        result = np.empty(len(values), dtype=np.float64)
        result[order] = np.arange(len(values), dtype=np.float64)
        return result

    a = ranks(np.asarray(left, dtype=np.float64))
    b = ranks(np.asarray(right, dtype=np.float64))
    return float(np.corrcoef(a, b)[0, 1])


def _register_formula(
    registry: dict[str, dict[str, Any]],
    *,
    token_ids: tuple[int, ...],
    expected_reward: float | None,
) -> str:
    candidate = build_candidate_record(
        formula_id="audit",
        source="v3a_gate1_complexity_audit",
        token_ids=token_ids,
        reward=0.0 if expected_reward is None else expected_reward,
        train_summary={},
        attempt_index=0,
    )
    key = hashlib.sha256(bytes(token_ids)).hexdigest()
    entry = registry.setdefault(
        key,
        {
            "canonical_hash": candidate.formula_hash,
            "token_ids": tuple(int(value) for value in token_ids),
            "token_names": list(FORMULA_VOCAB.decode(token_ids)),
            "formula_text": compile_formula(token_ids).expression.text(),
            "expected_rewards": [],
        },
    )
    if entry["token_ids"] != tuple(token_ids):
        raise RuntimeError("Formula sequence hash collision")
    if expected_reward is not None:
        entry["expected_rewards"].append(float(expected_reward))
    return key


def _register_rows(
    registry: dict[str, dict[str, Any]], records: np.ndarray, indices: np.ndarray
) -> list[str]:
    return [
        _register_formula(
            registry,
            token_ids=_row_token_ids(records[int(index)]),
            expected_reward=float(records[int(index)]["reward"]),
        )
        for index in indices
    ]


def _build_targets(train_view) -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    config = ScorerConfig()
    full = build_forward_targets(
        train_view.absolute_open,
        train_view.tradable_mask,
        train_view.dates,
        SplitSpec("full", "2016-08-09", VALIDATION_END),
        config,
    )
    chunks = np.array_split(full.decision_indices, 3)
    targets: dict[str, Any] = {"full": full}
    boundaries: dict[str, dict[str, str]] = {}
    for number, chunk in enumerate(chunks, start=1):
        name = f"S{number}"
        start = train_view.dates[int(chunk[0])].date().isoformat()
        end = train_view.dates[int(chunk[-1])].date().isoformat()
        if end > VALIDATION_END:
            raise RuntimeError("Complexity audit subperiod exposes future dates")
        target = build_forward_targets(
            train_view.absolute_open,
            train_view.tradable_mask,
            train_view.dates,
            SplitSpec(name, start, end),
            config,
        )
        targets[name] = target
        boundaries[name] = {
            "start": start,
            "end": end,
            "scorer_days": int(target.days),
        }
    return targets, boundaries


def _score_registry(
    registry: dict[str, dict[str, Any]], train_view, targets: dict[str, Any]
) -> tuple[dict[str, dict[str, float | bool]], float, list[dict[str, Any]]]:
    device = torch.device("cpu")
    factors = torch.as_tensor(train_view.factor_values, dtype=torch.float32, device=device)
    mask = torch.as_tensor(train_view.tradable_mask, dtype=torch.bool, device=device)
    torch_targets = {
        name: TorchForwardTargets.from_numpy(target, device=device, dtype=torch.float32)
        for name, target in targets.items()
    }
    vm = BatchTorchVM(
        max_output_bytes=2 * 1024**3,
        max_working_bytes=2 * 1024**3,
        max_total_bytes=3 * 1024**3,
    )
    keys = sorted(registry)
    output: dict[str, dict[str, float | bool]] = {}
    max_difference = 0.0
    max_difference_detail: dict[str, Any] | None = None
    sensitive: list[dict[str, Any]] = []
    candidate_config = CandidateConfig()
    for start in range(0, len(keys), 16):
        batch_keys = keys[start : start + 16]
        compiled = [compile_formula(registry[key]["token_ids"]) for key in batch_keys]
        max_length = max(len(item.instructions) for item in compiled)
        codes = torch.full((len(compiled), max_length), -1, dtype=torch.long)
        lengths = torch.as_tensor([len(item.instructions) for item in compiled], dtype=torch.long)
        for row, item in enumerate(compiled):
            codes[row, : len(item.instructions)] = torch.as_tensor(item.instructions)
        vm_result = vm.execute(codes, lengths, factors, mask)
        if not bool(vm_result.valid.all().item()):
            raise RuntimeError("Registered audit formula is VM-invalid")
        quality_valid, coverage, finite_std, _ = signal_quality_batch(
            vm_result.signal,
            torch_targets["full"],
            min_coverage=candidate_config.min_coverage,
            constant_std_eps=candidate_config.constant_std_eps,
        )
        batch_scores: dict[str, torch.Tensor] = {}
        for name, target in torch_targets.items():
            scored = score_signal_batch(
                vm_result.signal, vm_result.valid, target, ScorerConfig()
            )
            if not bool(scored.valid.all().item()):
                raise RuntimeError(f"Registered audit formula is scorer-invalid in {name}")
            batch_scores[name] = scored.reward.detach().cpu()
        for row, key in enumerate(batch_keys):
            output[key] = {
                **{
                    name: float(values[row].item())
                    for name, values in batch_scores.items()
                },
                "cpu_quality_valid": bool(quality_valid[row].item()),
                "cpu_coverage": float(coverage[row].item()),
                "cpu_finite_std": float(finite_std[row].item()),
                "device_sensitive": False,
            }
            for expected in registry[key]["expected_rewards"]:
                difference = abs(float(output[key]["full"]) - expected)
                if difference > max_difference:
                    max_difference = difference
                    max_difference_detail = {
                        "sequence_hash": key,
                        "token_names": registry[key]["token_names"],
                        "rescored_reward": output[key]["full"],
                        "expected_reward": expected,
                    }
            differences = [
                abs(float(output[key]["full"]) - expected)
                for expected in registry[key]["expected_rewards"]
            ]
            if differences and max(differences) > FULL_REWARD_TOLERANCE:
                output[key]["device_sensitive"] = True
                sensitive.append(
                    {
                        "sequence_hash": key,
                        "token_names": registry[key]["token_names"],
                        "max_abs_error": max(differences),
                        "rescored_reward": output[key]["full"],
                        "expected_rewards": registry[key]["expected_rewards"],
                    }
                )
    sensitive.sort(key=lambda item: -item["max_abs_error"])
    return output, max_difference, sensitive


def _eligible_keys(
    keys: Iterable[str], scores: dict[str, dict[str, float | bool]]
) -> list[str]:
    return [
        key
        for key in keys
        if not bool(scores[key]["device_sensitive"])
        and bool(scores[key]["cpu_quality_valid"])
    ]


def _group_metrics(
    keys: list[str], scores: dict[str, dict[str, float | bool]]
) -> dict[str, Any]:
    eligible = _eligible_keys(keys, scores)
    if not eligible:
        raise RuntimeError("Complexity audit group has no CPU-reproducible formulas")
    matrix = np.asarray(
        [
            [float(scores[key][period]) for period in ("S1", "S2", "S3")]
            for key in eligible
        ],
        dtype=np.float64,
    )
    full = np.asarray([float(scores[key]["full"]) for key in eligible], dtype=np.float64)
    worst = matrix.min(axis=1)
    ranges = matrix.max(axis=1) - matrix.min(axis=1)
    return {
        "selected_count": len(keys),
        "eligible_count": len(eligible),
        "excluded_count": len(keys) - len(eligible),
        "full_median": float(np.median(full)),
        "subperiod_medians": {
            f"S{index + 1}": float(np.median(matrix[:, index])) for index in range(3)
        },
        "worst_subperiod_median": float(np.median(worst)),
        "all_subperiods_positive_rate": float(np.mean((matrix > 0.0).all(axis=1))),
        "subperiod_range_median": float(np.median(ranges)),
    }


def _stable_region_summary(
    per_seed: list[dict[str, Any]], feature_payload: dict[str, Any]
) -> dict[str, Any]:
    worst_wins = sum(
        item["late_metrics"]["worst_subperiod_median"]
        > item["early_metrics"]["worst_subperiod_median"]
        for item in per_seed
    )
    positive_wins = sum(
        item["late_metrics"]["all_subperiods_positive_rate"]
        >= item["early_metrics"]["all_subperiods_positive_rate"]
        for item in per_seed
    )
    similarity_passed = (
        feature_payload["late_pairwise_cosine_mean"]
        >= feature_payload["early_pairwise_cosine_mean"]
    )
    replicated = feature_payload["replicated_feature_changes"]
    passed = worst_wins >= 2 and positive_wins >= 2 and similarity_passed and bool(replicated)
    return {
        "status": "within_train_supported" if passed else "mixed_or_not_supported",
        "worst_subperiod_wins": worst_wins,
        "all_positive_noninferiority_wins": positive_wins,
        "late_feature_similarity_noninferior": similarity_passed,
        "replicated_feature_change_count": len(replicated),
    }


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    stable = payload["stable_region"]
    lines = [
        "# V3A Stage D 训练期复杂度审计",
        "",
        "> 只使用2016-08-09至2021-12-31训练期；2022 validation和2023+ final未读取。",
        "> 训练子期仍参与过模型训练；本报告不是样本外验证。",
        "",
        "## 结果摘要",
        "",
        f"后期稳定区域判读：`{stable['decision']['status']}`。",
        "",
        "| Seed | early最差子期中位数 | late最差子期中位数 | early三期全正 | late三期全正 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for item in stable["seeds"]:
        lines.append(
            f"| {item['seed']} | {item['early_metrics']['worst_subperiod_median']:.6%} | "
            f"{item['late_metrics']['worst_subperiod_median']:.6%} | "
            f"{item['early_metrics']['all_subperiods_positive_rate']:.1%} | "
            f"{item['late_metrics']['all_subperiods_positive_rate']:.1%} |"
        )
    lines.extend(
        [
            "",
            "## 极端Top50",
            "",
            "| Seed | Top50三期全正 | Top50最差子期 | 短公式最差子期 | Top500平均rank rho | 三期都胜短subtree |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in payload["extreme_top50"]["seeds"]:
        lines.append(
            f"| {item['seed']} | {item['top50_metrics']['all_subperiods_positive_rate']:.1%} | "
            f"{item['top50_metrics']['worst_subperiod_median']:.6%} | "
            f"{item['simple_metrics']['worst_subperiod_median']:.6%} | "
            f"{item['complex_pool_pairwise_spearman_mean']:.3f} | "
            f"{item['subtree_comparison']['all_three_positive_rate']:.1%} |"
        )
    lines.extend(
        [
            "",
            "## 子期边界",
            "",
            "```json",
            json.dumps(payload["subperiods"], ensure_ascii=False, indent=2),
            "```",
            "",
            "## 数据边界",
            "",
            f"- 全期重评分最大绝对误差：`{payload['full_reward_rescore_max_abs_error']:.3g}`；",
            "- `validation_or_final_metrics_read=false`；",
            "- 详细公式分数见`formula_scores.jsonl`。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    torch.set_num_threads(min(8, torch.get_num_threads()))
    runs_dir = args.mechanism_runs_dir.resolve()
    train_view_dir = args.train_view_dir.resolve()
    output_dir = args.output_dir.resolve()

    manifest = json.loads(
        (train_view_dir / "train_view_manifest.json").read_text(encoding="utf-8")
    )
    train_view = load_stage_d_train_view(
        train_view_dir,
        research_spec={
            "research_spec_id": manifest["research_spec_id"],
            "code_fingerprint": manifest["code_fingerprint"],
        },
    )
    if train_view.dates[-1].date().isoformat() != VALIDATION_END:
        raise RuntimeError("Complexity audit train view end differs from frozen boundary")
    targets, boundaries = _build_targets(train_view)

    registry: dict[str, dict[str, Any]] = {}
    seed_inputs: list[dict[str, Any]] = []
    feature_summaries: dict[str, dict[int, dict[str, Any]]] = {
        "early": {},
        "late": {},
    }
    exact_top50: dict[int, set[str]] = {}

    for seed in SEEDS:
        run_id = f"v3a-stage-d-mechanism-transformer-s{seed}-{PROTOCOL_PREFIX}"
        records = _load_ledger(runs_dir / run_id)
        tails = _new_valid_tail_indices(records)
        early_sample = _quantile_sample(tails["early"], TAIL_SAMPLE_SIZE)
        late_sample = _quantile_sample(tails["late"], TAIL_SAMPLE_SIZE)
        feature_summaries["early"][seed] = _feature_summary(records, tails["early"])
        feature_summaries["late"][seed] = _feature_summary(records, tails["late"])

        first_1m = _model_indices(ARCHIVE_CHECKPOINT)
        representatives = _representative_indices(records, first_1m)
        complex_indices = _reward_order(
            records, representatives[records["token_len"][representatives] == 15]
        )[:COMPLEX_POOL_SIZE]
        top50_indices = _curated_top_indices(records, complex_indices)
        short_indices = _reward_order(
            records, representatives[records["token_len"][representatives] <= 5]
        )[:25]
        medium_indices = _reward_order(
            records,
            representatives[
                (records["token_len"][representatives] >= 6)
                & (records["token_len"][representatives] <= 10)
            ],
        )[:25]
        simple_indices = np.concatenate((short_indices, medium_indices))

        early_keys = _register_rows(registry, records, early_sample)
        late_keys = _register_rows(registry, records, late_sample)
        top50_keys = _register_rows(registry, records, top50_indices)
        complex_keys = _register_rows(registry, records, complex_indices)
        simple_keys = _register_rows(registry, records, simple_indices)
        exact_top50[seed] = {
            str(registry[key]["canonical_hash"]) for key in top50_keys
        }

        subtree_keys: dict[str, list[str]] = {}
        for top_key in top50_keys:
            expression = compile_formula(registry[top_key]["token_ids"]).expression
            descendants: list[str] = []
            for subtree in _proper_subexpressions(expression):
                names = _expression_token_names(subtree)
                if len(names) > 10:
                    continue
                token_ids = tuple(FORMULA_VOCAB.encode(names))
                descendants.append(
                    _register_formula(registry, token_ids=token_ids, expected_reward=None)
                )
            subtree_keys[top_key] = sorted(set(descendants))

        seed_inputs.append(
            {
                "seed": seed,
                "early_keys": early_keys,
                "late_keys": late_keys,
                "top50_keys": top50_keys,
                "complex_keys": complex_keys,
                "simple_keys": simple_keys,
                "subtree_keys": subtree_keys,
            }
        )

    scores, max_difference, sensitive = _score_registry(registry, train_view, targets)

    stable_seeds: list[dict[str, Any]] = []
    for item in seed_inputs:
        stable_seeds.append(
            {
                "seed": item["seed"],
                "early_metrics": _group_metrics(item["early_keys"], scores),
                "late_metrics": _group_metrics(item["late_keys"], scores),
            }
        )

    stage_vectors: dict[str, list[dict[str, float]]] = {}
    stage_families: dict[str, list[set[str]]] = {}
    for stage in ("early", "late"):
        stage_vectors[stage] = [
            feature_summaries[stage][seed]["prevalence"] for seed in SEEDS
        ]
        stage_families[stage] = [
            {
                item["signature"]
                for item in feature_summaries[stage][seed]["top20_families"]
            }
            for seed in SEEDS
        ]
    replicated_changes: list[dict[str, Any]] = []
    all_features = sorted(
        set().union(*(set(vector) for stage in stage_vectors.values() for vector in stage))
    )
    for feature in all_features:
        changes = [
            stage_vectors["late"][index].get(feature, 0.0)
            - stage_vectors["early"][index].get(feature, 0.0)
            for index in range(3)
        ]
        mean_change = float(np.mean(changes))
        if (all(value > 0 for value in changes) or all(value < 0 for value in changes)) and abs(
            mean_change
        ) >= 0.05:
            replicated_changes.append(
                {"feature": feature, "changes": changes, "mean_change": mean_change}
            )
    replicated_changes.sort(key=lambda item: -abs(item["mean_change"]))
    feature_payload = {
        "early_pairwise_cosine": _pairwise(stage_vectors["early"], _cosine),
        "late_pairwise_cosine": _pairwise(stage_vectors["late"], _cosine),
        "early_top20_family_jaccard": _pairwise(stage_families["early"], _jaccard),
        "late_top20_family_jaccard": _pairwise(stage_families["late"], _jaccard),
        "replicated_feature_changes": replicated_changes,
        "by_stage_seed": feature_summaries,
    }
    feature_payload["early_pairwise_cosine_mean"] = float(
        np.mean(feature_payload["early_pairwise_cosine"])
    )
    feature_payload["late_pairwise_cosine_mean"] = float(
        np.mean(feature_payload["late_pairwise_cosine"])
    )
    feature_payload["early_top20_family_jaccard_mean"] = float(
        np.mean(feature_payload["early_top20_family_jaccard"])
    )
    feature_payload["late_top20_family_jaccard_mean"] = float(
        np.mean(feature_payload["late_top20_family_jaccard"])
    )

    extreme_seeds: list[dict[str, Any]] = []
    for item in seed_inputs:
        pool = _eligible_keys(item["complex_keys"], scores)
        top50_eligible = _eligible_keys(item["top50_keys"], scores)
        if len(top50_eligible) != TOP_SIZE:
            raise RuntimeError(
                f"Seed {item['seed']} top50 contains a device-sensitive or CPU-invalid formula"
            )
        period_values = {
            period: np.asarray(
                [float(scores[key][period]) for key in pool], dtype=np.float64
            )
            for period in ("S1", "S2", "S3")
        }
        correlations = [
            _spearman(period_values[left], period_values[right])
            for left, right in (("S1", "S2"), ("S1", "S3"), ("S2", "S3"))
        ]
        top50_set = set(item["top50_keys"])
        overlaps = {
            period: len(
                top50_set
                & {
                    pool[index]
                    for index in np.argsort(-period_values[period], kind="stable")[:TOP_SIZE]
                }
            )
            for period in ("S1", "S2", "S3")
        }
        deltas: list[list[float]] = []
        comparable = 0
        anchors: dict[str, str] = {}
        for top_key, descendant_keys in item["subtree_keys"].items():
            eligible_descendants = _eligible_keys(descendant_keys, scores)
            if not eligible_descendants:
                continue
            anchor = max(
                eligible_descendants, key=lambda key: float(scores[key]["full"])
            )
            anchors[top_key] = anchor
            comparable += 1
            deltas.append(
                [
                    float(scores[top_key][period]) - float(scores[anchor][period])
                    for period in ("S1", "S2", "S3")
                ]
            )
        delta_matrix = np.asarray(deltas, dtype=np.float64)
        subtree = {
            "comparable_count": comparable,
            "all_three_positive_rate": float(np.mean((delta_matrix > 0.0).all(axis=1))),
            "period_median_deltas": {
                f"S{index + 1}": float(np.median(delta_matrix[:, index]))
                for index in range(3)
            },
            "anchors": anchors,
        }
        extreme_seeds.append(
            {
                "seed": item["seed"],
                "top50_metrics": _group_metrics(item["top50_keys"], scores),
                "complex_rank51_500_metrics": _group_metrics(
                    [key for key in pool if key not in top50_set], scores
                ),
                "simple_metrics": _group_metrics(item["simple_keys"], scores),
                "complex_pool_pairwise_spearman": correlations,
                "complex_pool_pairwise_spearman_mean": float(np.mean(correlations)),
                "full_top50_overlap_with_subperiod_top50": overlaps,
                "subtree_comparison": subtree,
            }
        )

    exact_overlap = {
        "pairwise_counts": [
            len(exact_top50[left] & exact_top50[right])
            for left, right in ((101, 102), (101, 103), (102, 103))
        ],
        "all_three_count": len(set.intersection(*(exact_top50[seed] for seed in SEEDS))),
    }
    stable_decision = _stable_region_summary(stable_seeds, feature_payload)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "validation_or_final_metrics_read": False,
        "train_view_id": manifest["train_view_id"],
        "train_view_end": train_view.dates[-1].date().isoformat(),
        "subperiods": boundaries,
        "selection": {
            "tail_fraction": TAIL_FRACTION,
            "tail_sample_size_per_stage_seed": TAIL_SAMPLE_SIZE,
            "complex_pool_size_per_seed": COMPLEX_POOL_SIZE,
            "top_size_per_seed": TOP_SIZE,
            "complex_pool_attempt_stop": ARCHIVE_CHECKPOINT,
        },
        "registered_formula_count": len(registry),
        "full_reward_rescore_tolerance": FULL_REWARD_TOLERANCE,
        "full_reward_rescore_max_abs_error": max_difference,
        "device_sensitive_formula_count": len(sensitive),
        "device_sensitive_formulas": sensitive,
        "stable_region": {
            "seeds": stable_seeds,
            "features": feature_payload,
            "decision": stable_decision,
        },
        "extreme_top50": {
            "seeds": extreme_seeds,
            "exact_cross_seed_overlap": exact_overlap,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "complexity_audit.json"
    report_path = output_dir / "complexity_audit_report.md"
    score_path = output_dir / "formula_scores.jsonl"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    with score_path.open("w", encoding="utf-8") as handle:
        for key in sorted(registry):
            handle.write(
                json.dumps(
                    {"sequence_hash": key, **registry[key], "scores": scores[key]},
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    _write_report(report_path, payload)
    checksum_path = output_dir / "SHA256SUMS"
    checksum_path.write_text(
        "\n".join(
            f"{_sha256(path)}  {path.name}" for path in (json_path, report_path, score_path)
        )
        + "\n",
        encoding="ascii",
    )
    print(
        json.dumps(
            {
                "registered_formula_count": len(registry),
                "full_reward_rescore_max_abs_error": max_difference,
                "stable_region_decision": stable_decision,
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()