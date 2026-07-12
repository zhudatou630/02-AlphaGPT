from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
from unittest import mock
import unittest

import numpy as np
import torch

from alpha_etf.gpt.policy import TransformerFormulaPolicy, TransformerPolicyConfig
from alpha_etf.research_v3a.candidates import CandidateConfig, build_candidate_record
from alpha_etf.research_v3a.checkpointing import (
    atomic_save,
    build_checkpoint,
    empty_candidate_state,
    load_checkpoint,
    restore_training_state,
    validate_checkpoint,
)
from alpha_etf.research_v3a.language import FORMULA_VOCAB
from alpha_etf.research_v3a.sampling import PolicyVocab, SamplingConfig, sample_formulas
from alpha_etf.research_v3a.stage_d import (
    TRAIN_VIEW_SCHEMA_VERSION,
    build_stage_d_binding,
    build_train_view_manifest,
    effective_training_rewards,
    empty_training_candidate_state,
    load_stage_d_protocol,
    load_stage_d_train_view,
    method_run_config,
    reinforce_objective,
    require_stage_d_cuda,
    retain_candidate,
    sequence_digest,
    stage_d_protocol_id,
    validate_stage_d_protocol,
    validate_stage_d_binding,
    validate_training_candidate_state,
)
from scripts.v3a import select_candidates, train_gpu
from tests.test_v3a_contract import _research_spec


ROOT = Path(__file__).resolve().parents[1]
PILOT_PROTOCOL = ROOT / "configs/v3a_stage_d_gpu_pilot.json"


class V3AStageDTests(unittest.TestCase):
    def test_runtime_binding_freezes_post_commit_identities(self) -> None:
        protocol = load_stage_d_protocol(PILOT_PROTOCOL)
        spec = _research_spec()
        view_manifest = {
            "train_view_id": "v3a-train-view-fixture",
            "train_view_fingerprint": "ef" * 32,
        }
        binding = build_stage_d_binding(
            protocol=protocol,
            research_spec=spec,
            train_view_manifest=view_manifest,
            stage_c_report_sha256="12" * 32,
        )
        validate_stage_d_binding(
            binding,
            protocol=protocol,
            research_spec=spec,
            train_view_manifest=view_manifest,
            stage_c_report_sha256="12" * 32,
        )
        changed = copy.deepcopy(binding)
        changed["code_commit"] = "different"
        with self.assertRaisesRegex(RuntimeError, "frozen runtime identity"):
            validate_stage_d_binding(
                changed,
                protocol=protocol,
                research_spec=spec,
                train_view_manifest=view_manifest,
                stage_c_report_sha256="12" * 32,
            )

    def test_approved_pilot_protocol_is_exact_and_formal_is_fail_closed(self) -> None:
        protocol = load_stage_d_protocol(PILOT_PROTOCOL)
        self.assertEqual(protocol["mode"], "pilot")
        self.assertFalse(protocol["formal_budget_approved"])
        self.assertEqual(
            method_run_config(protocol, method="transformer", seed=314159)[
                "attempts"
            ],
            50_000,
        )
        with self.assertRaises(RuntimeError):
            method_run_config(protocol, method="matched_random", seed=314159)

        changed = copy.deepcopy(protocol)
        changed["runs"]["transformer"]["attempts"] = 50_001
        changed["protocol_id"] = stage_d_protocol_id(changed)
        with self.assertRaisesRegex(RuntimeError, "not approved by code"):
            validate_stage_d_protocol(changed)

        formal = copy.deepcopy(protocol)
        formal["mode"] = "formal"
        formal["formal_budget_approved"] = True
        formal["candidate_output"]["research_conclusion_allowed"] = True
        formal["runs"] = {
            "transformer": {"seeds": [1, 2], "attempts": 1_000_000},
            "matched_random": {"seeds": [3, 4], "attempts": 1_000_000},
        }
        formal["protocol_id"] = stage_d_protocol_id(formal)
        with self.assertRaisesRegex(RuntimeError, "has not been approved"):
            validate_stage_d_protocol(formal)

    def test_quality_invalid_formula_receives_hard_invalid_reward(self) -> None:
        effective = effective_training_rewards(
            torch.tensor([0.1, 0.2, float("nan")]),
            scorer_valid=torch.tensor([True, True, True]),
            quality_valid=torch.tensor([True, False, True]),
            hard_invalid_reward=-5.0,
        )
        torch.testing.assert_close(effective, torch.tensor([0.1, -5.0, -5.0]))

    def test_reinforce_objective_standardizes_advantage_and_preserves_gradients(self) -> None:
        log_prob = torch.tensor([-2.0, -3.0, -4.0], requires_grad=True)
        objective = reinforce_objective(
            log_prob_sums=log_prob,
            entropy_sums=torch.tensor([1.0, 2.0, 3.0]),
            decision_counts=torch.tensor([1, 2, 3]),
            rewards=torch.tensor([-5.0, 0.0, 1.0]),
            advantage_epsilon=1e-5,
            entropy_coefficient=1e-3,
        )
        self.assertAlmostEqual(float(objective.advantages.mean()), 0.0, places=6)
        self.assertTrue(bool(torch.isfinite(objective.loss).item()))
        objective.loss.backward()
        self.assertTrue(bool(torch.isfinite(log_prob.grad).all().item()))

        equal = reinforce_objective(
            log_prob_sums=torch.tensor([-1.0, -2.0]),
            entropy_sums=torch.tensor([2.0, 4.0]),
            decision_counts=torch.tensor([2, 4]),
            rewards=torch.tensor([0.5, 0.5]),
            advantage_epsilon=1e-5,
            entropy_coefficient=1e-3,
        )
        torch.testing.assert_close(equal.advantages, torch.zeros(2))
        self.assertAlmostEqual(float(equal.entropy_mean), 1.0)

    def test_training_candidate_state_reconciles_attempts_and_retained_duplicates(self) -> None:
        spec = _research_spec()
        state = empty_training_candidate_state(created_at="2026-07-11T00:00:00Z")
        tokens = FORMULA_VOCAB.encode(["DAYRET"])
        first = build_candidate_record(
            formula_id="first",
            source="unit",
            token_ids=tokens,
            reward=0.1,
            train_summary={"scorer_days": 300},
            attempt_index=0,
        )
        retain_candidate(state, first, CandidateConfig())
        canonical_key = bytes.fromhex(first.formula_hash)
        selection_key = hashlib.sha256(b"selection").digest()
        state.update(
            {
                "attempt_count": 1,
                "status_counts": {
                    "grammar_invalid": 0,
                    "vm_invalid": 0,
                    "insufficient_daily_signal": 0,
                    "low_coverage": 0,
                    "constant_signal": 0,
                    "canonical_duplicate": 0,
                    "selection_duplicate": 0,
                    "accepted_unique": 1,
                },
                "length_counts": {"1": 1},
                "canonical_attempt_counts": {canonical_key: 1},
                "selection_attempt_counts": {selection_key: 1},
                "semantic_valid_rewards": [0.1],
                "token_counts": {str(tokens[0]): 1},
                "formula_sequence_digest": sequence_digest("00" * 32, tokens),
                "attempt_ledger_offset": 10,
                "attempt_ledger_line_count": 1,
                "attempt_ledger_prefix_sha256": hashlib.sha256(b"row-1").hexdigest(),
            }
        )
        validate_training_candidate_state(
            state, attempt_count=1, research_spec=spec
        )

        duplicate = build_candidate_record(
            formula_id="second",
            source="unit",
            token_ids=tokens,
            reward=0.2,
            train_summary={"scorer_days": 300},
            attempt_index=1,
        )
        retain_candidate(state, duplicate, CandidateConfig())
        state["attempt_count"] = 2
        state["status_counts"]["canonical_duplicate"] = 1
        state["length_counts"]["1"] = 2
        state["structural_duplicate_count"] = 1
        state["canonical_attempt_counts"][canonical_key] = 2
        state["selection_duplicate_observations"] = 1
        state["selection_attempt_counts"][selection_key] = 2
        state["semantic_valid_rewards"].append(0.2)
        state["token_counts"][str(tokens[0])] = 2
        state["formula_sequence_digest"] = sequence_digest(
            state["formula_sequence_digest"], tokens
        )
        state["attempt_ledger_offset"] = 20
        state["attempt_ledger_line_count"] = 2
        state["attempt_ledger_prefix_sha256"] = hashlib.sha256(b"row-1row-2").hexdigest()
        validate_training_candidate_state(
            state, attempt_count=2, research_spec=spec
        )
        entry = state["canonical_state"][first.formula_hash]
        self.assertEqual(entry["attempt_count"], 2)
        self.assertEqual(entry["best_reward"], 0.2)
        self.assertEqual(entry["best_attempt_index"], 1)

        other_tokens = FORMULA_VOCAB.encode(["GAP"])
        other = build_candidate_record(
            formula_id="third",
            source="unit",
            token_ids=other_tokens,
            reward=0.3,
            train_summary={"scorer_days": 300},
            attempt_index=2,
        )
        retain_candidate(state, other, CandidateConfig(heap_per_bucket=1))
        other_key = bytes.fromhex(other.formula_hash)
        other_selection = hashlib.sha256(b"other-selection").digest()
        state["attempt_count"] = 3
        state["status_counts"]["accepted_unique"] = 2
        state["length_counts"]["1"] = 3
        state["canonical_attempt_counts"][other_key] = 1
        state["selection_attempt_counts"][other_selection] = 1
        state["semantic_valid_rewards"].append(0.3)
        state["token_counts"][str(other_tokens[0])] = 1
        state["formula_sequence_digest"] = sequence_digest(
            state["formula_sequence_digest"], other_tokens
        )
        state["attempt_ledger_offset"] = 30
        state["attempt_ledger_line_count"] = 3
        state["attempt_ledger_prefix_sha256"] = hashlib.sha256(
            b"row-1row-2row-3"
        ).hexdigest()
        validate_training_candidate_state(
            state, attempt_count=3, research_spec=spec, deep=True
        )
        self.assertNotIn(first.formula_hash, state["canonical_state"])
        removed = state["canonical_ledger"].pop(first.formula_hash)
        with self.assertRaisesRegex(RuntimeError, "deep canonical ledger"):
            validate_training_candidate_state(
                state, attempt_count=3, research_spec=spec, deep=True
            )
        state["canonical_ledger"][first.formula_hash] = removed

        state["attempt_ledger_line_count"] = 1
        with self.assertRaisesRegex(RuntimeError, "line count"):
            validate_training_candidate_state(
                state, attempt_count=3, research_spec=spec
            )

    def test_training_entry_has_no_cpu_fallback(self) -> None:
        protocol = load_stage_d_protocol(PILOT_PROTOCOL)
        with mock.patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "no CPU fallback"):
                require_stage_d_cuda(protocol)
        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.get_device_name", return_value="NVIDIA A100"),
        ):
            with self.assertRaisesRegex(RuntimeError, "rtx_4090d"):
                require_stage_d_cuda(protocol)
        self.assertEqual(
            train_gpu._resolved_run_until(
                attempts=50_000, batch_size=256, stop_after=10_240
            ),
            10_240,
        )
        with self.assertRaisesRegex(ValueError, "batch boundary"):
            train_gpu._resolved_run_until(
                attempts=50_000, batch_size=256, stop_after=10_000
            )

    def test_train_view_contains_no_validation_or_final_columns(self) -> None:
        spec = _research_spec()
        dataset = {
            "dataset_id": spec["dataset"]["dataset_id"],
            "panel_sha256": spec["dataset"]["panel_sha256"],
            "symbols": spec["dataset"]["symbols"],
        }
        assets = len(dataset["symbols"])
        dates = np.asarray(["2021-12-30", "2021-12-31"], dtype="U10")
        arrays = {
            "factor_values": np.zeros((40, assets, 2), dtype=np.float64),
            "absolute_open": np.ones((assets, 2), dtype=np.float64),
            "tradable_mask": np.ones((assets, 2), dtype=bool),
            "symbols": np.asarray(dataset["symbols"], dtype="U16"),
            "dates": dates,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            files = {}
            for name, value in arrays.items():
                path = root / f"{name}.npy"
                with path.open("wb") as handle:
                    np.save(handle, value, allow_pickle=False)
                files[name] = {
                    "path": path.name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                }
            from alpha_etf.research_v3a.factors import FACTOR_NAMES

            payload = {
                "schema_version": TRAIN_VIEW_SCHEMA_VERSION,
                "source_dataset_id": dataset["dataset_id"],
                "source_panel_sha256": dataset["panel_sha256"],
                "source_dataset_manifest": dataset,
                "research_spec_id": spec["research_spec_id"],
                "code_commit": spec["code_commit"],
                "code_fingerprint": spec["code_fingerprint"],
                "split": {
                    "signal_start": "2016-08-09",
                    "data_end": "2021-12-31",
                    "validation_columns_present": False,
                    "final_columns_present": False,
                },
                "factor_names": list(FACTOR_NAMES),
                "factor_shape": list(arrays["factor_values"].shape),
                "mask_shape": list(arrays["tradable_mask"].shape),
                "date_start": str(dates[0]),
                "date_end": str(dates[-1]),
                "files": files,
            }
            manifest = build_train_view_manifest(payload)
            (root / "train_view_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            view = load_stage_d_train_view(
                root, research_spec=spec
            )
            self.assertEqual(view.dates.max().strftime("%Y-%m-%d"), "2021-12-31")
            self.assertFalse(view.manifest["split"]["validation_columns_present"])

            arrays["dates"][-1] = "2022-01-03"
            with (root / "dates.npy").open("wb") as handle:
                np.save(handle, arrays["dates"], allow_pickle=False)
            payload["files"]["dates"]["sha256"] = hashlib.sha256(
                (root / "dates.npy").read_bytes()
            ).hexdigest()
            changed = build_train_view_manifest(payload)
            (root / "train_view_manifest.json").write_text(
                json.dumps(changed), encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "future dates"):
                load_stage_d_train_view(
                    root, research_spec=spec
                )

    def test_resume_reproduces_next_optimizer_update(self) -> None:
        spec = _research_spec()
        policy_vocab = PolicyVocab()
        sampling = SamplingConfig()
        model_config = TransformerPolicyConfig(
            model_vocab_size=policy_vocab.size,
            max_sequence_len=sampling.max_len + 1,
            d_model=16,
            num_layers=1,
            num_heads=4,
            ff_dim=32,
            dropout=0.0,
        )
        train_config = {"fixture": "stage-d-resume-update-v1"}
        scorer_config = {"fixture": True}

        def update(
            model: TransformerFormulaPolicy,
            optimizer: torch.optim.Optimizer,
        ) -> tuple[list[list[int]], float]:
            sampled = sample_formulas(
                model,
                4,
                policy_vocab,
                sampling,
                torch.device("cpu"),
            )
            rewards = torch.tensor(
                [sum(tokens) * 0.001 for tokens in sampled.formulas],
                dtype=torch.float32,
            )
            objective = reinforce_objective(
                log_prob_sums=sampled.log_prob_sums,
                entropy_sums=sampled.entropy_sums,
                decision_counts=sampled.formula_lengths + 1,
                rewards=rewards,
                advantage_epsilon=1e-5,
                entropy_coefficient=1e-3,
            )
            optimizer.zero_grad(set_to_none=True)
            objective.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            return sampled.formulas, float(objective.loss.detach())

        torch.manual_seed(1234)
        continuous_model = TransformerFormulaPolicy(model_config)
        continuous_optimizer = torch.optim.AdamW(
            continuous_model.parameters(), lr=1e-4, weight_decay=1e-5
        )
        update(continuous_model, continuous_optimizer)
        checkpoint = build_checkpoint(
            run_id="synthetic-stage-d",
            step=1,
            attempt_count=4,
            model=continuous_model,
            optimizer=continuous_optimizer,
            model_config=model_config.to_dict(),
            train_config=train_config,
            scorer_config=scorer_config,
            candidate_state=empty_candidate_state(attempt_count=4),
            research_spec=spec,
        )
        expected_formulas, expected_loss = update(
            continuous_model, continuous_optimizer
        )
        expected_parameters = {
            key: value.detach().clone()
            for key, value in continuous_model.state_dict().items()
        }

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.pt"
            atomic_save(checkpoint, path)
            restored_model = TransformerFormulaPolicy(model_config)
            restored_optimizer = torch.optim.AdamW(
                restored_model.parameters(), lr=1e-4, weight_decay=1e-5
            )
            loaded = load_checkpoint(
                path,
                research_spec=spec,
                run_id="synthetic-stage-d",
                model_config=model_config.to_dict(),
                train_config=train_config,
                scorer_config=scorer_config,
            )
            restore_training_state(
                loaded,
                model=restored_model,
                optimizer=restored_optimizer,
                research_spec=spec,
                run_id="synthetic-stage-d",
                model_config=model_config.to_dict(),
                train_config=train_config,
                scorer_config=scorer_config,
            )
            actual_formulas, actual_loss = update(
                restored_model, restored_optimizer
            )

        self.assertEqual(actual_formulas, expected_formulas)
        self.assertEqual(actual_loss, expected_loss)
        for key, expected in expected_parameters.items():
            torch.testing.assert_close(
                restored_model.state_dict()[key], expected, rtol=0.0, atol=0.0
            )

    def test_completed_recovery_binds_final_checkpoint(self) -> None:
        spec = _research_spec()
        candidate_state = empty_training_candidate_state(
            created_at="2026-07-11T00:00:00+00:00"
        )
        run_config = {
            "run_id": "completed-stage-d",
            "protocol_id": "protocol",
            "binding_id": "binding",
            "attempts": 0,
        }
        policy_vocab = PolicyVocab()
        sampling = SamplingConfig()
        model_config_object = TransformerPolicyConfig(
            model_vocab_size=policy_vocab.size,
            max_sequence_len=sampling.max_len + 1,
            d_model=16,
            num_layers=1,
            num_heads=4,
            ff_dim=32,
            dropout=0.0,
        )
        model_config = model_config_object.to_dict()
        train_config = {"fixture": "completed-recovery-v1"}
        scorer_config = {"fixture": True}
        model = TransformerFormulaPolicy(model_config_object)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        valid_checkpoint = build_checkpoint(
            run_id=run_config["run_id"],
            step=0,
            attempt_count=0,
            model=model,
            optimizer=optimizer,
            model_config=model_config,
            train_config=train_config,
            scorer_config=scorer_config,
            candidate_state=candidate_state,
            research_spec=spec,
        )
        for missing_key in (
            "model_state_dict",
            "optimizer_state_dict",
            "rng_state",
        ):
            damaged = copy.deepcopy(valid_checkpoint)
            damaged.pop(missing_key)
            with self.assertRaisesRegex(RuntimeError, "state is invalid"):
                validate_checkpoint(
                    damaged,
                    research_spec=spec,
                    run_id=run_config["run_id"],
                    model_config=model_config,
                    train_config=train_config,
                    scorer_config=scorer_config,
                )
        for rng_key, empty_value in (
            ("python_random_state", {}),
            ("numpy_random_state", {}),
            ("torch_random_state", torch.empty(0, dtype=torch.uint8)),
        ):
            damaged = copy.deepcopy(valid_checkpoint)
            damaged["rng_state"][rng_key] = empty_value
            with self.assertRaisesRegex(RuntimeError, "RNG state"):
                validate_checkpoint(
                    damaged,
                    research_spec=spec,
                    run_id=run_config["run_id"],
                    model_config=model_config,
                    train_config=train_config,
                    scorer_config=scorer_config,
                )
        stage_d_train_config = {
            "schema_version": "etf-v3a-stage-d-transformer-v1"
        }
        missing_optimizer_state = copy.deepcopy(valid_checkpoint)
        missing_optimizer_state["attempt_count"] = 1
        missing_optimizer_state["candidate_state"]["attempt_count"] = 1
        missing_optimizer_state["train_config"] = stage_d_train_config
        with self.assertRaisesRegex(RuntimeError, "optimizer state is incomplete"):
            validate_checkpoint(
                missing_optimizer_state,
                research_spec=spec,
                run_id=run_config["run_id"],
                model_config=model_config,
                train_config=stage_d_train_config,
                scorer_config=scorer_config,
            )
        stage_model = TransformerFormulaPolicy(model_config_object)
        stage_optimizer = torch.optim.AdamW(stage_model.parameters(), lr=1e-3)
        for parameter in stage_model.parameters():
            parameter.grad = torch.ones_like(parameter)
        stage_optimizer.step()
        stage_candidate_state = empty_training_candidate_state(
            created_at="2026-07-11T00:00:00+00:00"
        )
        stage_candidate_state["attempt_count"] = 1
        complete_stage_d_checkpoint = build_checkpoint(
            run_id=run_config["run_id"],
            step=1,
            attempt_count=1,
            model=stage_model,
            optimizer=stage_optimizer,
            model_config=model_config,
            train_config=stage_d_train_config,
            scorer_config=scorer_config,
            candidate_state=stage_candidate_state,
            research_spec=spec,
        )
        complete_stage_d_checkpoint["rng_state"][
            "torch_cuda_random_state_all"
        ] = [torch.ones(4, dtype=torch.uint8)]
        validate_checkpoint(
            complete_stage_d_checkpoint,
            research_spec=spec,
            run_id=run_config["run_id"],
            model_config=model_config,
            train_config=stage_d_train_config,
            scorer_config=scorer_config,
        )
        missing_moments = copy.deepcopy(complete_stage_d_checkpoint)
        for payload in missing_moments["optimizer_state_dict"]["state"].values():
            payload.pop("exp_avg")
            payload.pop("exp_avg_sq")
        with self.assertRaisesRegex(RuntimeError, "AdamW state is incomplete"):
            validate_checkpoint(
                missing_moments,
                research_spec=spec,
                run_id=run_config["run_id"],
                model_config=model_config,
                train_config=stage_d_train_config,
                scorer_config=scorer_config,
            )
        complex_model = copy.deepcopy(complete_stage_d_checkpoint)
        model_key = next(iter(complex_model["model_state_dict"]))
        complex_model["model_state_dict"][model_key] = complex_model[
            "model_state_dict"
        ][model_key].to(torch.complex64)
        with self.assertRaisesRegex(RuntimeError, "model state is invalid"):
            validate_checkpoint(
                complex_model,
                research_spec=spec,
                run_id=run_config["run_id"],
                model_config=model_config,
                train_config=stage_d_train_config,
                scorer_config=scorer_config,
            )
        complex_optimizer = copy.deepcopy(complete_stage_d_checkpoint)
        first_payload = next(
            iter(complex_optimizer["optimizer_state_dict"]["state"].values())
        )
        first_payload["exp_avg"] = first_payload["exp_avg"].to(torch.complex64)
        with self.assertRaisesRegex(RuntimeError, "AdamW tensor state is invalid"):
            validate_checkpoint(
                complex_optimizer,
                research_spec=spec,
                run_id=run_config["run_id"],
                model_config=model_config,
                train_config=stage_d_train_config,
                scorer_config=scorer_config,
            )
        summary = {
            "run_id": run_config["run_id"],
            "protocol_id": run_config["protocol_id"],
            "binding_id": run_config["binding_id"],
            "research_spec_id": spec["research_spec_id"],
            "attempt_count": run_config["attempts"],
            "formula_sequence_digest": candidate_state["formula_sequence_digest"],
            "attempt_ledger_prefix_sha256": candidate_state[
                "attempt_ledger_prefix_sha256"
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "training_summary.json").write_text(
                json.dumps(summary), encoding="utf-8"
            )
            (run_dir / "training_log.jsonl").write_text("{}\n", encoding="utf-8")
            latest = run_dir / "checkpoint_latest.pt"
            final = run_dir / "checkpoint_final.pt"
            latest.write_bytes(b"latest")
            atomic_save(valid_checkpoint, final)
            final_sha256 = hashlib.sha256(final.read_bytes()).hexdigest()
            recovered = train_gpu._recover_completed(
                run_dir=run_dir,
                checkpoint_path=final,
                candidate_state=candidate_state,
                run_config=run_config,
                research_spec=spec,
                model_config=model_config,
                train_config=train_config,
                scorer_config=scorer_config,
                latest_checkpoint=valid_checkpoint,
            )
            marker = json.loads(
                (run_dir / "training_complete.json").read_text(encoding="utf-8")
            )
            tampered = copy.deepcopy(valid_checkpoint)
            first_key = next(iter(tampered["model_state_dict"]))
            tampered["model_state_dict"][first_key].view(-1)[0] += 1.0
            atomic_save(tampered, final)
            with self.assertRaisesRegex(RuntimeError, "latest/final"):
                train_gpu._recover_completed(
                    run_dir=run_dir,
                    checkpoint_path=final,
                    candidate_state=candidate_state,
                    run_config=run_config,
                    research_spec=spec,
                    model_config=model_config,
                    train_config=train_config,
                    scorer_config=scorer_config,
                    latest_checkpoint=valid_checkpoint,
                )
        self.assertEqual(recovered, summary)
        self.assertEqual(marker["checkpoint_sha256"], final_sha256)
        self.assertNotEqual(
            marker["checkpoint_sha256"], hashlib.sha256(b"latest").hexdigest()
        )

    def test_final_funnel_state_must_derive_from_source_checkpoint(self) -> None:
        funnel = {"funnel_artifact_id": "fixture"}
        source = {
            "model_state_dict": {"weight": torch.tensor([1.0, 2.0])},
            "optimizer_state_dict": {"step": 3},
            "candidate_state": {"funnel_artifact": None, "attempt_count": 4},
            "generator_states": {"numpy": np.asarray([1, 2], dtype=np.uint32)},
        }
        final = copy.deepcopy(source)
        final["candidate_state"]["funnel_artifact"] = funnel
        select_candidates._assert_final_state_derived(source, final, funnel)
        final["model_state_dict"]["weight"][0] = 9.0
        with self.assertRaisesRegex(RuntimeError, "model_state_dict"):
            select_candidates._assert_final_state_derived(source, final, funnel)


if __name__ == "__main__":
    unittest.main()