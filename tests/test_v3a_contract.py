from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch

from alpha_etf.gpt.policy import TransformerFormulaPolicy, TransformerPolicyConfig
from alpha_etf.research_v3a.artifacts import (
    build_formula_artifact,
    build_training_funnel_artifact,
    load_jsonl,
    validate_formula_artifact,
    validate_training_funnel_artifact,
    write_jsonl,
)
from alpha_etf.research_v3a.candidates import (
    CandidateConfig,
    CandidateSelection,
    build_candidate_record,
    canonicalizer_config,
)
from alpha_etf.research_v3a.checkpointing import (
    atomic_save,
    build_checkpoint,
    empty_candidate_state,
    load_checkpoint,
    restore_training_state,
    validate_candidate_state,
    validate_checkpoint,
)
from alpha_etf.research_v3a.language import FORMULA_VOCAB
from alpha_etf.research_v3a.factors import factor_config
from alpha_etf.research_v3a.scoring import ScorerConfig
from alpha_etf.research_v3a.sampling import (
    PolicyVocab,
    SamplingConfig,
    generate_random_formula,
    sample_formulas,
)
from alpha_etf.research_v3a.spec import build_research_spec


def _research_spec() -> dict[str, object]:
    manifest = {
        "dataset_id": "etf-v3a-unit",
        "dataset_fingerprint": "dataset-fingerprint",
        "panel_sha256": "panel-sha",
        "upstream_v3_dataset_id": "etf-relative-price-v3-989c5d80a4e0",
        "upstream_v3_panel_sha256": "upstream-sha",
        "symbols": [f"ETF{i:02d}" for i in range(35)],
        "date_start": "2016-01-01",
        "date_end": "2026-01-01",
    }
    return build_research_spec(
        dataset_manifest=manifest,
        factor_config=factor_config(),
        vocab_config={
            **FORMULA_VOCAB.to_config(),
            "sampling": SamplingConfig().to_dict(),
        },
        scorer_config=ScorerConfig().to_dict(),
        canonicalizer_config=canonicalizer_config(),
        candidate_config=CandidateConfig().to_dict(),
        code_commit="commit",
        code_fingerprint="fingerprint",
    )


class V3AContractTests(unittest.TestCase):
    def test_artifact_roundtrip_and_tamper_rejection(self) -> None:
        spec = _research_spec()
        record = build_candidate_record(
            formula_id="unit",
            source="test",
            token_ids=FORMULA_VOCAB.encode(["ROC", "WIN_20"]),
            reward=0.012,
            train_summary={"scorer_days": 300},
        )
        artifact = build_formula_artifact(
            record, research_spec=spec, created_at="2026-07-11T00:00:00Z"
        )
        restored = validate_formula_artifact(artifact, research_spec=spec)
        self.assertEqual(restored.formula_hash, record.formula_hash)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "formulas.jsonl"
            write_jsonl(path, [artifact])
            loaded = load_jsonl(path)
        self.assertEqual(len(loaded), 1)
        validate_formula_artifact(loaded[0], research_spec=spec)

        tampered = json.loads(json.dumps(artifact))
        tampered["formula_hash"] = "bad"
        with self.assertRaises(RuntimeError):
            validate_formula_artifact(tampered, research_spec=spec)
        legacy = {**artifact, "schema_version": "etf-price-artifact-v2"}
        with self.assertRaises(RuntimeError):
            validate_formula_artifact(legacy, research_spec=spec)
        with self.assertRaisesRegex(RuntimeError, "finite"):
            build_formula_artifact(
                replace(record, reward=float("nan")),
                research_spec=spec,
                created_at="2026-07-11T00:00:00Z",
            )

    def test_training_funnel_artifact_freezes_representatives_and_members(self) -> None:
        spec = _research_spec()
        config = CandidateConfig()
        first = build_candidate_record(
            formula_id="first",
            source="test",
            token_ids=FORMULA_VOCAB.encode(["DAYRET"]),
            reward=0.2,
            train_summary={},
            attempt_index=0,
        )
        second = build_candidate_record(
            formula_id="second",
            source="test",
            token_ids=FORMULA_VOCAB.encode(["GAP"]),
            reward=0.1,
            train_summary={},
            attempt_index=1,
        )
        first = replace(first, cluster_id="cluster_0000")
        second = replace(second, cluster_id="cluster_0000")
        selection = CandidateSelection(
            (first,),
            (first, second),
            {
                "signal_unique_count": 2,
                "cluster_count": 1,
                "selected_count": 1,
                "selected_bucket_counts": [1, 0, 0],
            },
        )
        artifact = build_training_funnel_artifact(
            selection,
            research_spec=spec,
            candidate_config=config,
            created_at="2026-07-11T00:00:00Z",
        )
        validate_training_funnel_artifact(
            artifact, research_spec=spec, candidate_config=config
        )
        self.assertEqual(artifact["clusters"][0]["representative_hash"], first.formula_hash)
        self.assertEqual(
            artifact["clusters"][0]["member_hashes"],
            [first.formula_hash, second.formula_hash],
        )
        tampered = json.loads(json.dumps(artifact))
        tampered["clusters"][0]["representative_hash"] = second.formula_hash
        with self.assertRaises(RuntimeError):
            validate_training_funnel_artifact(
                tampered, research_spec=spec, candidate_config=config
            )
        duplicate_selected = json.loads(json.dumps(artifact))
        duplicate_selected["selected_formula_hashes"] = [
            first.formula_hash,
            first.formula_hash,
        ]
        with self.assertRaisesRegex(RuntimeError, "selected formulas"):
            validate_training_funnel_artifact(
                duplicate_selected, research_spec=spec, candidate_config=config
            )

        candidate_state = {
            "schema_version": "etf-v3a-candidate-state-v1",
            "attempt_count": 2,
            "canonical_state": {
                first.formula_hash: {
                    "formula_hash": first.formula_hash,
                    "first_attempt_index": first.first_attempt_index,
                    "attempt_count": first.attempt_count,
                    "best_attempt_index": first.best_attempt_index,
                    "best_reward": first.reward,
                    "best_record": first.to_dict(),
                },
                second.formula_hash: {
                    "formula_hash": second.formula_hash,
                    "first_attempt_index": second.first_attempt_index,
                    "attempt_count": second.attempt_count,
                    "best_attempt_index": second.best_attempt_index,
                    "best_reward": second.reward,
                    "best_record": second.to_dict(),
                },
            },
            "bucket_state": [[first.formula_hash, second.formula_hash], [], []],
            "funnel_artifact": artifact,
        }
        validate_candidate_state(candidate_state, attempt_count=2, research_spec=spec)
        wrong_bucket = json.loads(json.dumps(candidate_state))
        wrong_bucket["bucket_state"] = [[], [], [first.formula_hash, second.formula_hash]]
        with self.assertRaises(RuntimeError):
            validate_candidate_state(wrong_bucket, attempt_count=2, research_spec=spec)
        for field, value in (
            ("best_reward", 999.0),
            ("best_attempt_index", 999),
            ("first_attempt_index", 1),
        ):
            invalid_ledger = json.loads(json.dumps(candidate_state))
            invalid_ledger["canonical_state"][first.formula_hash][field] = value
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                validate_candidate_state(
                    invalid_ledger, attempt_count=2, research_spec=spec
                )

    def test_checkpoint_roundtrip_and_v1_v2_rejection(self) -> None:
        spec = _research_spec()
        policy_vocab = PolicyVocab()
        model_config = TransformerPolicyConfig(
            model_vocab_size=policy_vocab.size,
            max_sequence_len=17,
            d_model=16,
            num_layers=1,
            num_heads=4,
            ff_dim=32,
            dropout=0.0,
        )
        model = TransformerFormulaPolicy(model_config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        train_config = {"seed": 42, "batch_size": 8, "max_len": 15}
        scorer_config = ScorerConfig().to_dict()
        model_config_dict = model_config.to_dict()
        checkpoint = build_checkpoint(
            run_id="unit",
            step=3,
            attempt_count=24,
            model=model,
            optimizer=optimizer,
            model_config=model_config_dict,
            train_config=train_config,
            scorer_config=scorer_config,
            candidate_state=empty_candidate_state(attempt_count=24),
            research_spec=spec,
        )
        validate_checkpoint(
            checkpoint,
            research_spec=spec,
            run_id="unit",
            model_config=model_config_dict,
            train_config=train_config,
            scorer_config=scorer_config,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pt"
            atomic_save(checkpoint, path)
            loaded = load_checkpoint(
                path,
                research_spec=spec,
                run_id="unit",
                model_config=model_config_dict,
                train_config=train_config,
                scorer_config=scorer_config,
            )
        self.assertEqual(loaded["attempt_count"], 24)
        self.assertIn("torch_random_state", loaded["rng_state"])

        changed_spec = _research_spec()
        changed_spec["code_commit"] = "different"
        with self.assertRaises(RuntimeError):
            validate_checkpoint(
                checkpoint,
                research_spec=changed_spec,
                run_id="unit",
                model_config=model_config_dict,
                train_config=train_config,
                scorer_config=scorer_config,
            )
        for legacy_schema in ("phase3b-checkpoint-v1", "etf-price-checkpoint-v2"):
            legacy = {**checkpoint, "schema_version": legacy_schema}
            with self.assertRaises(RuntimeError):
                validate_checkpoint(
                    legacy,
                    research_spec=spec,
                    run_id="unit",
                    model_config=model_config_dict,
                    train_config=train_config,
                    scorer_config=scorer_config,
                )

    def test_checkpoint_rejects_vocab_or_training_config_changes(self) -> None:
        spec = _research_spec()
        policy_vocab = PolicyVocab()
        model = TransformerFormulaPolicy(
            TransformerPolicyConfig(
                model_vocab_size=policy_vocab.size,
                max_sequence_len=17,
                d_model=16,
                num_layers=1,
                num_heads=4,
                ff_dim=32,
            )
        )
        optimizer = torch.optim.AdamW(model.parameters())
        model_config = model.config.to_dict()
        train_config = {"seed": 1}
        scorer_config = ScorerConfig().to_dict()
        checkpoint = build_checkpoint(
            run_id="unit",
            step=0,
            attempt_count=0,
            model=model,
            optimizer=optimizer,
            model_config=model_config,
            train_config=train_config,
            scorer_config=scorer_config,
            candidate_state=empty_candidate_state(),
            research_spec=spec,
        )
        checkpoint["formula_vocab"] = {"version": "old", "token_names": []}
        with self.assertRaises(RuntimeError):
            validate_checkpoint(
                checkpoint,
                research_spec=spec,
                run_id="unit",
                model_config=model_config,
                train_config=train_config,
                scorer_config=scorer_config,
            )
        checkpoint = build_checkpoint(
            run_id="unit",
            step=0,
            attempt_count=0,
            model=model,
            optimizer=optimizer,
            model_config=model_config,
            train_config=train_config,
            scorer_config=scorer_config,
            candidate_state=empty_candidate_state(),
            research_spec=spec,
        )
        for field, value in (
            ("run_id", "other-run"),
            ("model_config", {"changed": True}),
            ("scorer_config", {"changed": True}),
            ("candidate_state", {"schema_version": "legacy"}),
            (
                "candidate_state",
                {
                    **empty_candidate_state(),
                    "bucket_state": [["unknown_hash"], [], []],
                },
            ),
            (
                "candidate_state",
                {
                    **empty_candidate_state(),
                    "canonical_state": {"bad_hash": {"formula_hash": "bad_hash"}},
                },
            ),
            (
                "candidate_state",
                {
                    **empty_candidate_state(),
                    "funnel_artifact": {"schema_version": "legacy-funnel"},
                },
            ),
        ):
            tampered = {**checkpoint, field: value}
            with self.subTest(field=field):
                with self.assertRaises(RuntimeError):
                    validate_checkpoint(
                        tampered,
                        research_spec=spec,
                        run_id="unit",
                        model_config=model_config,
                        train_config=train_config,
                        scorer_config=scorer_config,
                    )
        checkpoint = build_checkpoint(
            run_id="unit",
            step=0,
            attempt_count=0,
            model=model,
            optimizer=optimizer,
            model_config=model_config,
            train_config=train_config,
            scorer_config=scorer_config,
            candidate_state=empty_candidate_state(),
            research_spec=spec,
        )
        with self.assertRaises(RuntimeError):
            validate_checkpoint(
                checkpoint,
                research_spec=spec,
                run_id="unit",
                model_config=model_config,
                train_config={"seed": 2},
                scorer_config=scorer_config,
            )

    def test_checkpoint_rng_state_resumes_exactly(self) -> None:
        random.seed(19)
        np.random.seed(19)
        torch.manual_seed(19)
        spec = _research_spec()
        policy_vocab = PolicyVocab()
        model = TransformerFormulaPolicy(
            TransformerPolicyConfig(
                model_vocab_size=policy_vocab.size,
                max_sequence_len=17,
                d_model=16,
                num_layers=1,
                num_heads=4,
                ff_dim=32,
            )
        )
        optimizer = torch.optim.AdamW(model.parameters())
        independent_rng = random.Random(31)
        model_config = model.config.to_dict()
        train_config = {"seed": 19}
        scorer_config = ScorerConfig().to_dict()
        checkpoint = build_checkpoint(
            run_id="rng",
            step=1,
            attempt_count=8,
            model=model,
            optimizer=optimizer,
            model_config=model_config,
            train_config=train_config,
            scorer_config=scorer_config,
            candidate_state=empty_candidate_state(attempt_count=8),
            research_spec=spec,
            random_generators={"random_baseline": independent_rng},
        )
        expected_batch = sample_formulas(
            model,
            4,
            policy_vocab,
            SamplingConfig(),
            torch.device("cpu"),
        )
        expected = (
            expected_batch.formulas,
            random.random(),
            float(np.random.random()),
            float(torch.rand(())),
            generate_random_formula(independent_rng),
        )
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(1.0)
        random.seed(99)
        np.random.seed(99)
        torch.manual_seed(99)
        restored_rng = random.Random(99)
        restored = restore_training_state(
            checkpoint,
            model=model,
            optimizer=optimizer,
            research_spec=spec,
            run_id="rng",
            model_config=model_config,
            train_config=train_config,
            scorer_config=scorer_config,
            random_generators={"random_baseline": restored_rng},
        )
        actual_batch = sample_formulas(
            model,
            4,
            policy_vocab,
            SamplingConfig(),
            torch.device("cpu"),
        )
        actual = (
            actual_batch.formulas,
            random.random(),
            float(np.random.random()),
            float(torch.rand(())),
            generate_random_formula(restored_rng),
        )
        self.assertEqual(actual, expected)
        self.assertEqual(restored["attempt_count"], 8)
        self.assertEqual(
            restored["candidate_state"], empty_candidate_state(attempt_count=8)
        )


if __name__ == "__main__":
    unittest.main()