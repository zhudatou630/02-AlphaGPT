"""Canonicalization, signal similarity, and the frozen V3A candidate funnel."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from typing import Any

import numpy as np

from alpha_etf.research_v3a.language import Expression, compile_formula


CANONICALIZER_VERSION = "etf-v3a-canonical-v1"
CANDIDATE_FUNNEL_VERSION = "etf-v3a-candidate-funnel-v1"


def canonicalizer_config() -> dict[str, Any]:
    return {
        "version": CANONICALIZER_VERSION,
        "commutative_direct_child_sort": ["ADD", "MUL"],
        "nan_preserving": True,
        "constant_rewrites": ["x+0", "x-0", "x*1"],
        "forbidden_rewrites": ["x*0"],
    }


@dataclass(frozen=True)
class CandidateConfig:
    min_coverage: float = 0.95
    constant_std_eps: float = 1e-12
    min_scorer_days: int = 252
    min_similarity_days: int = 252
    min_similarity_assets: int = 10
    duplicate_rho: float = 0.995
    cluster_rho: float = 0.90
    fisher_clip: float = 1e-7
    heap_per_bucket: int = 500
    first_pass_quotas: tuple[int, int, int] = (10, 10, 5)
    final_quotas: tuple[int, int, int] = (20, 20, 10)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["first_pass_quotas"] = list(self.first_pass_quotas)
        value["final_quotas"] = list(self.final_quotas)
        value["version"] = CANDIDATE_FUNNEL_VERSION
        value["canonicalizer_version"] = CANONICALIZER_VERSION
        return value


@dataclass(frozen=True)
class SimilarityContext:
    decision_indices: np.ndarray
    available: np.ndarray

    def __post_init__(self) -> None:
        if self.available.shape[0] != len(self.decision_indices):
            raise ValueError("Similarity availability differs from decision indices")


@dataclass(frozen=True)
class SimilarityResult:
    rho: float
    days: int
    sufficient: bool


@dataclass(frozen=True)
class _PreparedRanks:
    eligible: np.ndarray
    counts: np.ndarray
    centered: np.ndarray
    norm: np.ndarray


@dataclass(frozen=True)
class CandidateRecord:
    formula_id: str
    source: str
    token_ids: tuple[int, ...]
    token_names: tuple[str, ...]
    token_len: int
    expression: dict[str, Any]
    canonical_expression: dict[str, Any]
    formula_hash: str
    reward: float
    train_summary: dict[str, Any]
    cluster_id: str = ""
    first_attempt_index: int = -1
    attempt_count: int = 1
    best_attempt_index: int = -1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateSelection:
    selected: tuple[CandidateRecord, ...]
    clustered: tuple[CandidateRecord, ...]
    audit: dict[str, Any]


def _canonical_json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def expression_hash(expression: Expression) -> str:
    return hashlib.sha256(_canonical_json(expression.to_dict()).encode("utf-8")).hexdigest()


def _constant_value(expression: Expression) -> float | None:
    if expression.kind != "constant":
        return None
    if expression.name == "CONST_0":
        return 0.0
    if expression.name == "CONST_1":
        return 1.0
    if expression.name == "CONST" and expression.params:
        return float(expression.params[0])
    return None


def _constant(value: float) -> Expression:
    if value == 0.0:
        return Expression("constant", "CONST_0", (0.0,))
    if value == 1.0:
        return Expression("constant", "CONST_1", (1.0,))
    return Expression("constant", "CONST", (float(value),))


def canonicalize_expression(expression: Expression) -> Expression:
    if not expression.children:
        return expression
    children = tuple(canonicalize_expression(child) for child in expression.children)
    name = expression.name

    if name == "NEG" and children[0].name == "NEG":
        return children[0].children[0]
    if name == "ABS" and children[0].name == "ABS":
        return children[0]
    if name == "SIGN" and children[0].name == "SIGN":
        return children[0]

    if name in {"ADD", "SUB", "MUL"}:
        left, right = children
        left_constant = _constant_value(left)
        right_constant = _constant_value(right)
        if left_constant is not None and right_constant is not None:
            if name == "ADD":
                return _constant(left_constant + right_constant)
            if name == "SUB":
                return _constant(left_constant - right_constant)
            return _constant(left_constant * right_constant)
        if name == "ADD":
            if left_constant == 0.0:
                return right
            if right_constant == 0.0:
                return left
        elif name == "SUB" and right_constant == 0.0:
            return left
        elif name == "MUL":
            if left_constant == 1.0:
                return right
            if right_constant == 1.0:
                return left
        if name in {"ADD", "MUL"} and expression_hash(right) < expression_hash(left):
            left, right = right, left
        return Expression("operator", name, expression.params, (left, right))
    return Expression(expression.kind, name, expression.params, children)


def build_candidate_record(
    *,
    formula_id: str,
    source: str,
    token_ids: list[int] | tuple[int, ...],
    reward: float,
    train_summary: dict[str, Any],
    attempt_index: int = -1,
) -> CandidateRecord:
    compiled = compile_formula(token_ids)
    canonical = canonicalize_expression(compiled.expression)
    from alpha_etf.research_v3a.language import FORMULA_VOCAB

    ids = tuple(int(item) for item in token_ids)
    return CandidateRecord(
        formula_id=formula_id,
        source=source,
        token_ids=ids,
        token_names=tuple(FORMULA_VOCAB.decode(ids)),
        token_len=len(ids),
        expression=compiled.expression.to_dict(),
        canonical_expression=canonical.to_dict(),
        formula_hash=expression_hash(canonical),
        reward=float(reward),
        train_summary=dict(train_summary),
        first_attempt_index=int(attempt_index),
        attempt_count=1,
        best_attempt_index=int(attempt_index),
    )


def candidate_record_from_dict(payload: dict[str, Any]) -> CandidateRecord:
    required = {
        "formula_id",
        "source",
        "token_ids",
        "token_names",
        "token_len",
        "expression",
        "canonical_expression",
        "formula_hash",
        "reward",
        "train_summary",
        "cluster_id",
        "first_attempt_index",
        "attempt_count",
        "best_attempt_index",
    }
    missing = required - set(payload)
    if missing:
        raise RuntimeError(f"V3A candidate record missing fields: {sorted(missing)}")
    from alpha_etf.research_v3a.language import FORMULA_VOCAB

    token_ids = tuple(int(item) for item in payload["token_ids"])
    if int(payload["token_len"]) != len(token_ids) or not 1 <= len(token_ids) <= 15:
        raise RuntimeError("V3A candidate token length mismatch")
    token_names = tuple(FORMULA_VOCAB.decode(token_ids))
    if tuple(payload["token_names"]) != token_names:
        raise RuntimeError("V3A candidate token names mismatch")
    compiled = compile_formula(token_ids)
    canonical = canonicalize_expression(compiled.expression)
    formula_hash = expression_hash(canonical)
    if payload["expression"] != compiled.expression.to_dict():
        raise RuntimeError("V3A candidate expression mismatch")
    if payload["canonical_expression"] != canonical.to_dict():
        raise RuntimeError("V3A candidate canonical expression mismatch")
    if str(payload["formula_hash"]) != formula_hash:
        raise RuntimeError("V3A candidate formula hash mismatch")
    attempt_count = int(payload["attempt_count"])
    first_attempt = int(payload["first_attempt_index"])
    best_attempt = int(payload["best_attempt_index"])
    if attempt_count < 1 or first_attempt < -1 or best_attempt < -1:
        raise RuntimeError("V3A candidate attempt ledger is invalid")
    reward = float(payload["reward"])
    if not np.isfinite(reward):
        raise RuntimeError("V3A candidate reward must be finite")
    return CandidateRecord(
        formula_id=str(payload["formula_id"]),
        source=str(payload["source"]),
        token_ids=token_ids,
        token_names=token_names,
        token_len=len(token_ids),
        expression=compiled.expression.to_dict(),
        canonical_expression=canonical.to_dict(),
        formula_hash=formula_hash,
        reward=reward,
        train_summary=dict(payload["train_summary"]),
        cluster_id=str(payload["cluster_id"]),
        first_attempt_index=first_attempt,
        attempt_count=attempt_count,
        best_attempt_index=best_attempt,
    )


def signal_quality(
    signal: np.ndarray,
    mask: np.ndarray,
    *,
    min_coverage: float,
    constant_std_eps: float,
) -> dict[str, Any]:
    usable = mask & np.isfinite(signal)
    total = int(mask.sum())
    finite_count = int(usable.sum())
    coverage = finite_count / total if total else 0.0
    values = signal[usable]
    std = float(np.std(values, ddof=0)) if len(values) else np.nan
    if finite_count == 0:
        reason = "non_finite_result"
    elif coverage < min_coverage:
        reason = "low_coverage"
    elif not np.isfinite(std) or std <= constant_std_eps:
        reason = "constant_signal"
    else:
        reason = ""
    return {
        "valid": not reason,
        "invalid_reason": reason,
        "finite_count": finite_count,
        "coverage": coverage,
        "finite_std": std,
    }


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        average_rank = (start + 1 + stop) / 2.0
        ranks[order[start:stop]] = average_rank
        start = stop
    return ranks


def signal_similarity(
    left: np.ndarray,
    right: np.ndarray,
    context: SimilarityContext,
    config: CandidateConfig,
) -> SimilarityResult:
    left_values = left[:, context.decision_indices].T
    right_values = right[:, context.decision_indices].T
    eligible = (
        context.available
        & np.isfinite(left_values)
        & np.isfinite(right_values)
    )
    counts = eligible.sum(axis=1)
    usable_days = counts >= config.min_similarity_assets
    if not usable_days.any():
        return SimilarityResult(rho=np.nan, days=0, sufficient=False)
    left_values = left_values[usable_days]
    right_values = right_values[usable_days]
    eligible = eligible[usable_days]
    counts = counts[usable_days]

    def row_ranks(values: np.ndarray) -> np.ndarray:
        less = (
            (values[:, None, :] < values[:, :, None])
            & eligible[:, None, :]
        ).sum(axis=-1, dtype=np.float64)
        equal = (
            (values[:, None, :] == values[:, :, None])
            & eligible[:, None, :]
        ).sum(axis=-1, dtype=np.float64)
        ranks = less + (equal + 1.0) / 2.0
        return np.where(eligible, ranks, 0.0)

    left_rank = row_ranks(left_values)
    right_rank = row_ranks(right_values)
    rank_mean = (counts.astype(np.float64) + 1.0) / 2.0
    left_centered = np.where(
        eligible, left_rank - rank_mean[:, None], 0.0
    )
    right_centered = np.where(
        eligible, right_rank - rank_mean[:, None], 0.0
    )
    numerator = np.sum(left_centered * right_centered, axis=1)
    denominator = np.sqrt(
        np.sum(left_centered * left_centered, axis=1)
        * np.sum(right_centered * right_centered, axis=1)
    )
    nonconstant = denominator > 0
    rho_by_day = numerator[nonconstant] / denominator[nonconstant]
    valid_counts = counts[nonconstant]
    lower = -1.0 + config.fisher_clip
    upper = 1.0 - config.fisher_clip
    clipped = np.clip(rho_by_day, lower, upper)
    weights = np.maximum(valid_counts - 3, 1).astype(np.float64)
    weighted_z = float(np.sum(np.arctanh(clipped) * weights))
    total_weight = float(np.sum(weights))
    days = int(nonconstant.sum())
    sufficient = days >= config.min_similarity_days and total_weight > 0
    rho = float(np.tanh(weighted_z / total_weight)) if sufficient else np.nan
    return SimilarityResult(rho=rho, days=days, sufficient=sufficient)


class SignalSimilarityCache:
    """Cache exact daily ranks when two signals share the same finite mask."""

    def __init__(
        self,
        signals: dict[str, np.ndarray],
        context: SimilarityContext,
        config: CandidateConfig,
    ) -> None:
        self.signals = signals
        self.context = context
        self.config = config
        self._prepared: dict[str, _PreparedRanks] = {}

    def _prepare(self, formula_hash: str) -> _PreparedRanks:
        cached = self._prepared.get(formula_hash)
        if cached is not None:
            return cached
        signal = self.signals[formula_hash]
        values = signal[:, self.context.decision_indices].T
        eligible = self.context.available & np.isfinite(values)
        counts = eligible.sum(axis=1)
        less = (
            (values[:, None, :] < values[:, :, None])
            & eligible[:, None, :]
        ).sum(axis=-1, dtype=np.float64)
        equal = (
            (values[:, None, :] == values[:, :, None])
            & eligible[:, None, :]
        ).sum(axis=-1, dtype=np.float64)
        ranks = less + (equal + 1.0) / 2.0
        rank_mean = (counts.astype(np.float64) + 1.0) / 2.0
        centered = np.where(eligible, ranks - rank_mean[:, None], 0.0)
        norm = np.sqrt(np.sum(centered * centered, axis=1))
        cached = _PreparedRanks(eligible, counts, centered, norm)
        self._prepared[formula_hash] = cached
        return cached

    def compare(self, left_hash: str, right_hash: str) -> SimilarityResult:
        left = self._prepare(left_hash)
        right = self._prepare(right_hash)
        if not np.array_equal(left.eligible, right.eligible):
            return signal_similarity(
                self.signals[left_hash],
                self.signals[right_hash],
                self.context,
                self.config,
            )
        denominator = left.norm * right.norm
        usable = (
            (left.counts >= self.config.min_similarity_assets)
            & (denominator > 0)
        )
        rho_by_day = (
            np.sum(left.centered[usable] * right.centered[usable], axis=1)
            / denominator[usable]
        )
        valid_counts = left.counts[usable]
        lower = -1.0 + self.config.fisher_clip
        upper = 1.0 - self.config.fisher_clip
        clipped = np.clip(rho_by_day, lower, upper)
        weights = np.maximum(valid_counts - 3, 1).astype(np.float64)
        total_weight = float(np.sum(weights))
        days = int(usable.sum())
        sufficient = days >= self.config.min_similarity_days and total_weight > 0
        rho = (
            float(np.tanh(np.sum(np.arctanh(clipped) * weights) / total_weight))
            if sufficient
            else np.nan
        )
        return SimilarityResult(rho=rho, days=days, sufficient=sufficient)


def _candidate_sort_key(candidate: CandidateRecord) -> tuple[float, int, str]:
    return (-candidate.reward, candidate.token_len, candidate.formula_hash)


def _bucket(token_len: int) -> int:
    if 1 <= token_len <= 5:
        return 0
    if 6 <= token_len <= 10:
        return 1
    if 11 <= token_len <= 15:
        return 2
    raise ValueError(f"Candidate token length outside V3A range: {token_len}")


def _signal_has_similarity_coverage(
    signal: np.ndarray, context: SimilarityContext, config: CandidateConfig
) -> bool:
    days = 0
    for row, decision in enumerate(context.decision_indices):
        count = int((context.available[row] & np.isfinite(signal[:, decision])).sum())
        if count >= config.min_similarity_assets:
            days += 1
    return days >= config.min_similarity_days


def select_training_candidates(
    records: list[CandidateRecord],
    signals: dict[str, np.ndarray],
    context: SimilarityContext,
    config: CandidateConfig = CandidateConfig(),
) -> CandidateSelection:
    by_hash: dict[str, CandidateRecord] = {}
    for record in records:
        old = by_hash.get(record.formula_hash)
        if old is None:
            by_hash[record.formula_hash] = record
            continue
        first_indices = [
            index
            for index in (old.first_attempt_index, record.first_attempt_index)
            if index >= 0
        ]
        first_attempt = min(first_indices) if first_indices else -1
        best = record if _candidate_sort_key(record) < _candidate_sort_key(old) else old
        best_attempt = best.best_attempt_index
        if best_attempt < 0:
            best_attempt = best.first_attempt_index
        by_hash[record.formula_hash] = replace(
            best,
            first_attempt_index=first_attempt,
            attempt_count=old.attempt_count + record.attempt_count,
            best_attempt_index=best_attempt,
        )
    canonical_records = list(by_hash.values())

    bucketed: list[CandidateRecord] = []
    for bucket in range(3):
        values = sorted(
            (record for record in canonical_records if _bucket(record.token_len) == bucket),
            key=_candidate_sort_key,
        )
        bucketed.extend(values[: config.heap_per_bucket])
    ordered = sorted(bucketed, key=_candidate_sort_key)

    eligible: list[CandidateRecord] = []
    insufficient_overlap = 0
    for record in ordered:
        signal = signals.get(record.formula_hash)
        if signal is None:
            raise KeyError(f"Missing signal for candidate {record.formula_hash}")
        if _signal_has_similarity_coverage(signal, context, config):
            eligible.append(record)
        else:
            insufficient_overlap += 1

    unique: list[CandidateRecord] = []
    signal_duplicates = 0
    similarity_cache: dict[tuple[str, str], SimilarityResult] = {}
    rank_cache = SignalSimilarityCache(signals, context, config)

    def similarity(left: CandidateRecord, right: CandidateRecord) -> SimilarityResult:
        key = tuple(sorted((left.formula_hash, right.formula_hash)))
        if key not in similarity_cache:
            similarity_cache[key] = rank_cache.compare(
                left.formula_hash, right.formula_hash
            )
        return similarity_cache[key]

    for record in eligible:
        duplicate = False
        for kept in unique:
            result = similarity(record, kept)
            if not result.sufficient:
                duplicate = True
                insufficient_overlap += 1
                break
            if result.rho >= config.duplicate_rho:
                duplicate = True
                signal_duplicates += 1
                break
        if not duplicate:
            unique.append(record)

    representatives: list[CandidateRecord] = []
    clustered: list[CandidateRecord] = []
    for record in unique:
        cluster_index = -1
        for index, representative in enumerate(representatives):
            result = similarity(record, representative)
            if result.sufficient and result.rho >= config.cluster_rho:
                cluster_index = index
                break
        if cluster_index < 0:
            representatives.append(record)
            cluster_index = len(representatives) - 1
        clustered.append(replace(record, cluster_id=f"cluster_{cluster_index:04d}"))

    selected: list[CandidateRecord] = []
    selected_hashes: set[str] = set()
    used_clusters: set[str] = set()
    first_counts = [0, 0, 0]
    for record in clustered:
        bucket = _bucket(record.token_len)
        if first_counts[bucket] >= config.first_pass_quotas[bucket]:
            continue
        if record.cluster_id in used_clusters:
            continue
        selected.append(record)
        selected_hashes.add(record.formula_hash)
        used_clusters.add(record.cluster_id)
        first_counts[bucket] += 1

    final_counts = [sum(_bucket(item.token_len) == bucket for item in selected) for bucket in range(3)]
    for record in clustered:
        if record.formula_hash in selected_hashes:
            continue
        bucket = _bucket(record.token_len)
        if final_counts[bucket] >= config.final_quotas[bucket]:
            continue
        selected.append(record)
        selected_hashes.add(record.formula_hash)
        final_counts[bucket] += 1

    target_total = sum(config.final_quotas)
    if len(selected) < target_total:
        for record in clustered:
            if record.formula_hash in selected_hashes:
                continue
            selected.append(record)
            selected_hashes.add(record.formula_hash)
            if len(selected) == target_total:
                break

    audit = {
        "input_count": len(records),
        "canonical_unique_count": len(canonical_records),
        "canonical_attempt_count": sum(record.attempt_count for record in canonical_records),
        "heap_count": len(bucketed),
        "similarity_eligible_count": len(eligible),
        "signal_unique_count": len(unique),
        "cluster_count": len(representatives),
        "selected_count": len(selected),
        "signal_duplicate_count": signal_duplicates,
        "insufficient_similarity_overlap_count": insufficient_overlap,
        "selected_bucket_counts": [
            sum(_bucket(item.token_len) == bucket for item in selected) for bucket in range(3)
        ],
    }
    return CandidateSelection(tuple(selected), tuple(clustered), audit)