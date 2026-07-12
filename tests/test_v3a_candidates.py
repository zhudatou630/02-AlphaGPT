from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

from alpha_etf.research_v3a.candidates import (
    CandidateConfig,
    CandidateRecord,
    SignalSimilarityCache,
    SimilarityContext,
    build_candidate_record,
    canonicalize_expression,
    expression_hash,
    select_training_candidates,
    signal_quality,
    signal_similarity,
)
from alpha_etf.research_v3a.language import FORMULA_VOCAB, compile_formula


class V3ACandidateTests(unittest.TestCase):
    def test_canonicalizer_handles_proven_equivalences_only(self) -> None:
        left = compile_formula(FORMULA_VOCAB.encode(["DAYRET", "GAP", "ADD"])).expression
        right = compile_formula(FORMULA_VOCAB.encode(["GAP", "DAYRET", "ADD"])).expression
        self.assertEqual(
            expression_hash(canonicalize_expression(left)),
            expression_hash(canonicalize_expression(right)),
        )
        double_negative = compile_formula(
            FORMULA_VOCAB.encode(["DAYRET", "NEG", "NEG"])
        ).expression
        dayret = compile_formula(FORMULA_VOCAB.encode(["DAYRET"])).expression
        self.assertEqual(
            expression_hash(canonicalize_expression(double_negative)),
            expression_hash(canonicalize_expression(dayret)),
        )
        times_zero = compile_formula(
            FORMULA_VOCAB.encode(["DAYRET", "CONST_0", "MUL"])
        ).expression
        zero = compile_formula(FORMULA_VOCAB.encode(["CONST_0"])).expression
        self.assertNotEqual(
            expression_hash(canonicalize_expression(times_zero)),
            expression_hash(canonicalize_expression(zero)),
        )

    def test_signal_quality_rejects_sparse_and_constant_values(self) -> None:
        mask = np.ones((4, 10), dtype=bool)
        constant = signal_quality(
            np.ones_like(mask, dtype=float),
            mask,
            min_coverage=0.95,
            constant_std_eps=1e-12,
        )
        self.assertEqual(constant["invalid_reason"], "constant_signal")
        sparse_signal = np.full(mask.shape, np.nan)
        sparse_signal[:, :5] = np.arange(4)[:, None]
        sparse = signal_quality(
            sparse_signal, mask, min_coverage=0.95, constant_std_eps=1e-12
        )
        self.assertEqual(sparse["invalid_reason"], "low_coverage")

    def test_similarity_keeps_direction_and_handles_exact_endpoints(self) -> None:
        rng = np.random.default_rng(5)
        signal = rng.normal(size=(12, 40))
        context = SimilarityContext(
            decision_indices=np.arange(40), available=np.ones((40, 12), dtype=bool)
        )
        config = CandidateConfig(min_similarity_days=20, min_similarity_assets=10)
        same = signal_similarity(signal, signal * 3.0, context, config)
        inverse = signal_similarity(signal, -signal, context, config)
        self.assertTrue(same.sufficient)
        self.assertGreater(same.rho, 0.999)
        self.assertLess(inverse.rho, -0.999)

    def test_vectorized_similarity_matches_daywise_reference_with_ties_and_missing(self) -> None:
        rng = np.random.default_rng(27)
        assets = 12
        dates = 40
        left = rng.integers(-2, 3, size=(assets, dates)).astype(float)
        right = rng.integers(-2, 3, size=(assets, dates)).astype(float)
        left[0, ::5] = np.nan
        right[1, ::7] = np.nan
        available = rng.random((dates, assets)) > 0.08
        context = SimilarityContext(np.arange(dates), available)
        config = CandidateConfig(min_similarity_days=5, min_similarity_assets=8)

        weighted_z = 0.0
        total_weight = 0.0
        used_days = 0
        for row, decision in enumerate(context.decision_indices):
            eligible = (
                context.available[row]
                & np.isfinite(left[:, decision])
                & np.isfinite(right[:, decision])
            )
            count = int(eligible.sum())
            if count < config.min_similarity_assets:
                continue
            left_rank = np.empty(count)
            right_rank = np.empty(count)
            for values, ranks in (
                (left[eligible, decision], left_rank),
                (right[eligible, decision], right_rank),
            ):
                order = np.argsort(values, kind="stable")
                sorted_values = values[order]
                start = 0
                while start < count:
                    stop = start + 1
                    while stop < count and sorted_values[stop] == sorted_values[start]:
                        stop += 1
                    ranks[order[start:stop]] = (start + 1 + stop) / 2.0
                    start = stop
            if np.std(left_rank) == 0.0 or np.std(right_rank) == 0.0:
                continue
            rho = float(np.corrcoef(left_rank, right_rank)[0, 1])
            weight = float(max(count - 3, 1))
            weighted_z += np.arctanh(
                np.clip(rho, -1.0 + config.fisher_clip, 1.0 - config.fisher_clip)
            ) * weight
            total_weight += weight
            used_days += 1
        expected = float(np.tanh(weighted_z / total_weight))
        actual = signal_similarity(left, right, context, config)
        self.assertEqual(actual.days, used_days)
        self.assertAlmostEqual(actual.rho, expected, places=12)

        cached = SignalSimilarityCache(
            {"left": left, "right": right}, context, config
        ).compare("left", "right")
        self.assertEqual(cached.days, actual.days)
        self.assertEqual(cached.sufficient, actual.sufficient)
        self.assertAlmostEqual(cached.rho, actual.rho, places=12)

    def test_similarity_cache_matches_reference_for_identical_finite_masks(self) -> None:
        rng = np.random.default_rng(91)
        left = rng.normal(size=(12, 50))
        right = rng.normal(size=(12, 50))
        left[0, ::4] = np.nan
        right[0, ::4] = np.nan
        context = SimilarityContext(
            np.arange(50), np.ones((50, 12), dtype=bool)
        )
        config = CandidateConfig(min_similarity_days=20, min_similarity_assets=10)
        reference = signal_similarity(left, right, context, config)
        cached = SignalSimilarityCache(
            {"left": left, "right": right}, context, config
        ).compare("left", "right")
        self.assertEqual(cached.days, reference.days)
        self.assertEqual(cached.sufficient, reference.sufficient)
        self.assertAlmostEqual(cached.rho, reference.rho, places=12)

    def test_similarity_cache_excludes_days_below_asset_minimum(self) -> None:
        rng = np.random.default_rng(101)
        left = rng.normal(size=(12, 300))
        right = rng.normal(size=(12, 300))
        left[2:, 252:] = np.nan
        right[2:, 252:] = np.nan
        context = SimilarityContext(
            np.arange(300), np.ones((300, 12), dtype=bool)
        )
        config = CandidateConfig(min_similarity_days=200, min_similarity_assets=10)
        reference = signal_similarity(left, right, context, config)
        cached = SignalSimilarityCache(
            {"left": left, "right": right}, context, config
        ).compare("left", "right")
        self.assertEqual(reference.days, 252)
        self.assertEqual(cached.days, reference.days)
        self.assertEqual(cached.sufficient, reference.sufficient)
        self.assertAlmostEqual(cached.rho, reference.rho, places=12)

    def test_candidate_funnel_is_deterministic_and_respects_length_quotas(self) -> None:
        rng = np.random.default_rng(17)
        assets = 6
        dates = 30
        context = SimilarityContext(
            decision_indices=np.arange(dates),
            available=np.ones((dates, assets), dtype=bool),
        )
        config = CandidateConfig(
            min_similarity_days=20,
            min_similarity_assets=4,
            heap_per_bucket=10,
            first_pass_quotas=(1, 1, 1),
            final_quotas=(2, 2, 1),
        )
        lengths = [2, 3, 5, 6, 8, 10, 11, 14]
        records: list[CandidateRecord] = []
        signals: dict[str, np.ndarray] = {}
        for index, length in enumerate(lengths):
            formula_hash = f"hash_{index:02d}"
            record = CandidateRecord(
                formula_id=f"formula_{index}",
                source="unit",
                token_ids=(0,),
                token_names=("DAYRET",),
                token_len=length,
                expression={"id": index},
                canonical_expression={"id": index},
                formula_hash=formula_hash,
                reward=1.0 - index * 0.01,
                train_summary={},
            )
            records.append(record)
            signals[formula_hash] = rng.normal(size=(assets, dates))

        duplicate = replace(
            records[0],
            formula_id="duplicate",
            formula_hash="hash_duplicate",
            reward=0.5,
        )
        records.append(duplicate)
        signals[duplicate.formula_hash] = signals[records[0].formula_hash].copy()
        first = select_training_candidates(records, signals, context, config)
        second = select_training_candidates(records, signals, context, config)
        self.assertEqual(
            [record.formula_hash for record in first.selected],
            [record.formula_hash for record in second.selected],
        )
        self.assertEqual(first.audit["selected_count"], 5)
        self.assertEqual(first.audit["selected_bucket_counts"], [2, 2, 1])
        self.assertEqual(first.audit["signal_duplicate_count"], 1)
        self.assertNotIn("hash_duplicate", {record.formula_hash for record in first.clustered})
        self.assertTrue(all(record.cluster_id for record in first.clustered))

    def test_canonical_ledger_preserves_first_attempt_and_best_record(self) -> None:
        context = SimilarityContext(
            decision_indices=np.arange(5), available=np.ones((5, 4), dtype=bool)
        )
        config = CandidateConfig(
            min_similarity_days=5,
            min_similarity_assets=4,
            heap_per_bucket=5,
            first_pass_quotas=(1, 0, 0),
            final_quotas=(1, 0, 0),
        )
        base = build_candidate_record(
            formula_id="first",
            source="unit",
            token_ids=FORMULA_VOCAB.encode(["DAYRET"]),
            reward=0.1,
            train_summary={},
            attempt_index=3,
        )
        better = replace(
            base,
            formula_id="better",
            reward=0.2,
            first_attempt_index=9,
            best_attempt_index=9,
        )
        signal = np.tile(np.arange(4, dtype=float)[:, None], (1, 5))
        selection = select_training_candidates(
            [base, better], {base.formula_hash: signal}, context, config
        )
        record = selection.clustered[0]
        self.assertEqual(record.formula_id, "better")
        self.assertEqual(record.first_attempt_index, 3)
        self.assertEqual(record.best_attempt_index, 9)
        self.assertEqual(record.attempt_count, 2)
        self.assertEqual(selection.audit["canonical_attempt_count"], 2)

    def test_build_candidate_record_binds_source_and_canonical_formula(self) -> None:
        record = build_candidate_record(
            formula_id="unit",
            source="test",
            token_ids=FORMULA_VOCAB.encode(["GAP", "DAYRET", "ADD"]),
            reward=0.01,
            train_summary={"days": 300},
        )
        self.assertEqual(record.token_len, 3)
        self.assertEqual(record.token_names, ("GAP", "DAYRET", "ADD"))
        self.assertEqual(len(record.formula_hash), 64)


if __name__ == "__main__":
    unittest.main()