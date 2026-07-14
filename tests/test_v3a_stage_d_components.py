from __future__ import annotations

import json
import random
from pathlib import Path
import tempfile
from unittest import mock
import unittest

import numpy as np
import torch

from alpha_etf.gpt.policy import TransformerFormulaPolicy, TransformerPolicyConfig
from alpha_etf.research_v3a.attempts import (
    ATTEMPT_DTYPE,
    AttemptLedger,
    AttemptStatus,
    PreparedAttemptBatch,
    prepare_attempt_batch,
    read_attempt_batches,
)
from alpha_etf.research_v3a.candidate_index import CompactCandidateIndex
from alpha_etf.research_v3a.candidates import CandidateConfig
from alpha_etf.research_v3a.factors import FACTOR_NAMES
from alpha_etf.research_v3a.gpu_sampling import (
    TensorFormulaSampler,
    build_tensor_grammar_tables,
)
from alpha_etf.research_v3a.language import (
    FORMULA_VOCAB,
    MAX_FORMULA_TOKENS,
    GrammarState,
    allowed_formula_token_ids,
    compile_formula,
    transition_state,
)
from alpha_etf.research_v3a.sampling import PolicyVocab, SamplingConfig
from alpha_etf.research_v3a.scoring import ScorerConfig
from alpha_etf.research_v3a.stage_d_checkpoint import (
    build_training_checkpoint,
    load_training_checkpoint,
    restore_training_checkpoint,
    save_training_checkpoint,
)
from alpha_etf.research_v3a.stage_d_runner import (
    SerialStageDConfig,
    SerialStageDRunner,
)
from alpha_etf.research_v3a import stage_d_runner as stage_d_runner_module
from alpha_etf.research_v3a.torch_scoring import TorchForwardTargets
from alpha_etf.research_v3a.torch_vm import BatchTorchVM


class V3AStageDComponentTests(unittest.TestCase):
    def test_tensor_grammar_and_vm_codes_match_reference(self) -> None:
        device = torch.device("cpu")
        policy_vocab = PolicyVocab()
        config = SamplingConfig()
        tables = build_tensor_grammar_tables(
            device=device, policy_vocab=policy_vocab, config=config
        )
        self.assertEqual(len(tables.states), 121)
        self.assertEqual(tables.max_vm_stack_depth, 8)

        state_ids = {state: index for index, state in enumerate(tables.states)}
        states_by_length = {GrammarState()}
        legal_transition_count = 0
        for length in range(config.max_len + 1):
            next_states: set[GrammarState] = set()
            for state in states_by_length:
                expected = {
                    policy_vocab.to_model_id(token_id)
                    for token_id in allowed_formula_token_ids(
                        state,
                        formula_length=length,
                        max_length=config.max_len,
                        vocab=FORMULA_VOCAB,
                    )
                }
                if state.valid:
                    expected.add(policy_vocab.eos_id)
                actual = set(
                    torch.where(tables.legal_actions[length, state_ids[state]])[0].tolist()
                )
                self.assertEqual(actual, expected)
                legal_transition_count += len(expected - {policy_vocab.eos_id})
                if length < config.max_len:
                    for model_id in expected:
                        if model_id == policy_vocab.eos_id:
                            continue
                        token_id = policy_vocab.to_formula_id(model_id)
                        next_states.add(
                            transition_state(state, FORMULA_VOCAB.id_to_token(token_id))
                        )
            states_by_length = next_states
        self.assertEqual(legal_transition_count, 3343)

        torch.manual_seed(7)
        sampler = TensorFormulaSampler(device=device)
        batch = sampler.sample_uniform(128)
        for row in range(128):
            token_len = int(batch.token_lengths[row])
            tokens = batch.token_ids[row, :token_len].tolist()
            expected_codes = compile_formula(tokens).instructions
            vm_len = int(batch.vm_lengths[row])
            self.assertEqual(batch.vm_codes[row, :vm_len].tolist(), list(expected_codes))

        model = TransformerFormulaPolicy(
            TransformerPolicyConfig(
                model_vocab_size=policy_vocab.size,
                max_sequence_len=MAX_FORMULA_TOKENS + 2,
                d_model=8,
                num_layers=1,
                num_heads=2,
                ff_dim=16,
                dropout=0.0,
            )
        )
        policy_batch = sampler.sample_policy(model, 8)
        (-policy_batch.log_prob_sums.mean()).backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_binary_ledger_snapshot_and_replay(self) -> None:
        token_ids = np.full((4, MAX_FORMULA_TOKENS), -1, dtype=np.int64)
        token_ids[:, 0] = np.asarray(
            [
                FORMULA_VOCAB.name_to_id("DAYRET"),
                FORMULA_VOCAB.name_to_id("GAP"),
                FORMULA_VOCAB.name_to_id("DAYRET"),
                FORMULA_VOCAB.name_to_id("INTRADAY"),
            ]
        )
        selected = np.asarray(
            [
                [[0, 9], [1, 2]],
                [[0, 8], [1, 2]],
                [[3, 7], [4, 5]],
                [[6, 7], [8, 9]],
            ],
            dtype=np.int16,
        )
        prepared = prepare_attempt_batch(
            attempt_start=0,
            training_step=1,
            token_ids=token_ids,
            token_lengths=np.ones(4, dtype=np.int64),
            vm_valid=np.asarray([True, True, True, False]),
            score_valid=np.ones(4, dtype=bool),
            quality_valid=np.ones(4, dtype=bool),
            variation_valid=np.ones(4, dtype=bool),
            rewards=np.asarray([0.1, 0.2, 0.3, -5.0], dtype=np.float32),
            training_rewards=np.asarray([0.1, 0.2, 0.3, -5.0], dtype=np.float32),
            coverage=np.asarray([1.0, 1.0, 1.0, 0.0], dtype=np.float32),
            finite_std=np.asarray([0.2, 0.3, 0.4, 0.0], dtype=np.float32),
            selected_indices=selected,
            top_k=np.asarray([1, 2], dtype=np.int64),
        )

        full = CompactCandidateIndex(initial_capacity=2)
        resolved_full = full.commit(PreparedAttemptBatch(prepared.records.copy()))
        self.assertEqual(
            resolved_full["status"].tolist(),
            [
                int(AttemptStatus.ACCEPTED_UNIQUE),
                int(AttemptStatus.SELECTION_DUPLICATE),
                int(AttemptStatus.CANONICAL_DUPLICATE),
                int(AttemptStatus.VM_INVALID),
            ],
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ledger_path = root / "attempts.bin"
            snapshot_path = root / "candidate_snapshot_a2.npz"
            partial = CompactCandidateIndex(initial_capacity=2)
            first = partial.commit(PreparedAttemptBatch(prepared.records[:2].copy()))
            with AttemptLedger(ledger_path, create=True) as ledger:
                ledger.append(first)
                partial.save_snapshot(snapshot_path)
                second = partial.commit(PreparedAttemptBatch(prepared.records[2:].copy()))
                ledger.append(second)
                ledger.flush(durable=True)
                expected_digest = ledger.prefix_sha256
            restored = CompactCandidateIndex.restore(
                snapshot_path=snapshot_path,
                ledger_path=ledger_path,
                stop=4,
            )
            stored = np.concatenate(list(read_attempt_batches(ledger_path)))
            self.assertEqual(ledger_path.stat().st_size, 4 * ATTEMPT_DTYPE.itemsize)
            self.assertEqual(stored.tobytes(), resolved_full.tobytes())
            self.assertEqual(partial.attempt_count, restored.attempt_count)
            self.assertEqual(partial._canonical_lookup, restored._canonical_lookup)
            self.assertEqual(partial._selection_lookup, restored._selection_lookup)
            self.assertTrue(
                np.array_equal(partial.status_counts, restored.status_counts)
            )
            self.assertEqual(partial.retained_rows(), restored.retained_rows())
            with AttemptLedger(ledger_path, create=False) as ledger:
                self.assertEqual(ledger.prefix_sha256, expected_digest)
                ledger.truncate(2)
                self.assertEqual(ledger.record_count, 2)

    def test_small_checkpoint_restores_training_and_rng_state(self) -> None:
        random.seed(13)
        np.random.seed(13)
        torch.manual_seed(13)
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        loss = model(torch.ones(1, 2)).sum()
        loss.backward()
        optimizer.step()
        saved_parameters = {
            name: value.detach().clone() for name, value in model.state_dict().items()
        }
        identity = {"research_spec_id": "spec", "method": "transformer", "seed": 13}
        checkpoint = build_training_checkpoint(
            run_id="fixture-run",
            run_identity=identity,
            step=2,
            attempt_count=16,
            model=model,
            optimizer=optimizer,
            ledger_prefix_sha256="ab" * 32,
            training_log_offset=123,
            training_log_prefix_sha256="cd" * 32,
            candidate_snapshot_name="candidate_snapshot_a0.npz",
            candidate_snapshot_attempt=0,
            candidate_snapshot_elapsed_seconds=0.0,
            candidate_snapshot_sha256="ef" * 32,
            elapsed_seconds=4.5,
            resume_count=0,
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint_latest.pt"
            save_training_checkpoint(checkpoint, path)
            expected_random = random.random()
            expected_numpy = float(np.random.random())
            expected_torch = torch.rand(3)
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.add_(10.0)
            random.random()
            np.random.random()
            torch.rand(3)
            loaded = load_training_checkpoint(
                path,
                expected_run_id="fixture-run",
                expected_run_identity=identity,
            )
            restored = restore_training_checkpoint(
                loaded, model=model, optimizer=optimizer
            )
        self.assertEqual(restored["attempt_count"], 16)
        self.assertEqual(restored["ledger"]["record_count"], 16)
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, saved_parameters[name]))
        self.assertEqual(random.random(), expected_random)
        self.assertEqual(float(np.random.random()), expected_numpy)
        self.assertTrue(torch.equal(torch.rand(3), expected_torch))

    def test_serial_runner_resume_and_matched_random_complete(self) -> None:
        with self.assertRaisesRegex(ValueError, "one-formula batch"):
            SerialStageDConfig(
                run_id="invalid-tail",
                run_identity={},
                method="transformer",
                seed=1,
                attempts=5,
                batch_size=4,
                cpu_worker_count=1,
                cpu_gpu_overlap=False,
                training_invalid_reward=-0.01,
                advantage_epsilon=1e-5,
                entropy_coefficient=0.005,
                gradient_clip_norm=1.0,
            )
        device = torch.device("cpu")
        generator = np.random.default_rng(123)
        factors = torch.as_tensor(
            generator.normal(size=(len(FACTOR_NAMES), 10, 80)), dtype=torch.float32
        )
        mask = torch.ones((10, 80), dtype=torch.bool)
        targets = TorchForwardTargets(
            decision_indices=torch.arange(65, 75, dtype=torch.long),
            available=torch.ones((10, 10), dtype=torch.bool),
            top_k=torch.full((10,), 2, dtype=torch.long),
            forward_returns=torch.as_tensor(
                generator.normal(scale=0.01, size=(10, 10)), dtype=torch.float32
            ),
            baseline_returns=torch.zeros(10, dtype=torch.float32),
        )
        scorer_config = ScorerConfig()
        candidate_config = CandidateConfig(min_coverage=0.5)

        def build_runner(
            run_dir: Path,
            *,
            method: str,
            seed: int,
            cpu_workers: int = 1,
            cpu_gpu_overlap: bool = False,
            snapshot_on_stop: bool = False,
        ) -> SerialStageDRunner:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            policy_vocab = PolicyVocab()
            sampler = TensorFormulaSampler(device=device, policy_vocab=policy_vocab)
            model = None
            optimizer = None
            if method == "transformer":
                model = TransformerFormulaPolicy(
                    TransformerPolicyConfig(
                        model_vocab_size=policy_vocab.size,
                        max_sequence_len=MAX_FORMULA_TOKENS + 1,
                        d_model=8,
                        num_layers=1,
                        num_heads=2,
                        ff_dim=16,
                        dropout=0.0,
                    )
                )
                optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
            config = SerialStageDConfig(
                run_id=run_dir.name,
                run_identity={"fixture": "serial-v2", "method": method, "seed": seed},
                method=method,
                seed=seed,
                attempts=12,
                batch_size=4,
                cpu_worker_count=cpu_workers,
                cpu_gpu_overlap=cpu_gpu_overlap,
                training_invalid_reward=-0.01,
                advantage_epsilon=1e-5,
                entropy_coefficient=0.005,
                gradient_clip_norm=1.0,
                candidate_snapshot_on_stop=snapshot_on_stop,
                checkpoint_seconds=10_000,
                candidate_snapshot_seconds=10_000,
            )
            return SerialStageDRunner(
                config=config,
                run_dir=run_dir,
                sampler=sampler,
                vm=BatchTorchVM(
                    max_working_bytes=1024**2,
                    max_output_bytes=1024**2,
                    max_total_bytes=2 * 1024**2,
                ),
                factors=factors,
                tradable_mask=mask,
                targets=targets,
                scorer_config=scorer_config,
                candidate_config=candidate_config,
                model=model,
                optimizer=optimizer,
            )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            continuous_dir = root / "continuous"
            resumed_dir = root / "resumed"
            random_dir = root / "random"

            continuous = build_runner(continuous_dir, method="transformer", seed=17)
            checkpoint_writes: list[str] = []
            real_save_checkpoint = stage_d_runner_module.save_training_checkpoint

            def record_checkpoint_write(checkpoint: dict, path: Path) -> None:
                checkpoint_writes.append(path.name)
                real_save_checkpoint(checkpoint, path)

            with mock.patch.object(
                stage_d_runner_module,
                "save_training_checkpoint",
                side_effect=record_checkpoint_write,
            ):
                continuous_summary = continuous.run()
            self.assertEqual(
                checkpoint_writes[-2:],
                ["checkpoint_final.pt", "checkpoint_latest.pt"],
            )
            partial = build_runner(
                resumed_dir,
                method="transformer",
                seed=17,
                cpu_workers=2,
                cpu_gpu_overlap=True,
                snapshot_on_stop=True,
            )
            partial_result = partial.run(stop_after=8)
            self.assertEqual(partial_result["status"], "checkpointed")
            ledger_path = resumed_dir / "attempts.bin"
            ledger_path.write_bytes(ledger_path.read_bytes() + b"torn-tail")
            partial_checkpoint = torch.load(
                resumed_dir / "checkpoint_latest.pt",
                map_location="cpu",
                weights_only=False,
            )
            snapshot_path = resumed_dir / partial_checkpoint["candidate_snapshot"]["name"]
            self.assertEqual(partial_checkpoint["candidate_snapshot"]["attempt_count"], 8)
            snapshot_bytes = snapshot_path.read_bytes()
            snapshot_path.write_bytes(snapshot_bytes + b"wrong-snapshot")
            rejected = build_runner(
                resumed_dir,
                method="transformer",
                seed=17,
                cpu_workers=2,
                cpu_gpu_overlap=True,
            )
            with self.assertRaisesRegex(RuntimeError, "snapshot differs"):
                rejected.run(resume=True)
            self.assertEqual(ledger_path.stat().st_size, 8 * ATTEMPT_DTYPE.itemsize)
            snapshot_path.write_bytes(snapshot_bytes)
            resumed = build_runner(
                resumed_dir,
                method="transformer",
                seed=17,
                cpu_workers=2,
                cpu_gpu_overlap=True,
            )
            resumed_summary = resumed.run(resume=True)
            resumed_logs = [
                json.loads(line)
                for line in (resumed_dir / "training_log.jsonl").read_text().splitlines()
            ]
            self.assertTrue(all(row["cpu_gpu_overlap"] for row in resumed_logs))
            self.assertEqual(resumed_logs[0]["pending_cpu_batches"], 1)
            self.assertEqual(
                (continuous_dir / "attempts.bin").read_bytes(),
                (resumed_dir / "attempts.bin").read_bytes(),
            )
            continuous_checkpoint = torch.load(
                continuous_dir / "checkpoint_final.pt",
                map_location="cpu",
                weights_only=False,
            )
            resumed_checkpoint = torch.load(
                resumed_dir / "checkpoint_final.pt",
                map_location="cpu",
                weights_only=False,
            )
            for name, value in continuous_checkpoint["model_state_dict"].items():
                self.assertTrue(
                    torch.equal(value, resumed_checkpoint["model_state_dict"][name])
                )
            self.assertEqual(continuous_summary["attempt_count"], 12)
            self.assertEqual(resumed_summary["resume_count"], 1)

            random_runner = build_runner(random_dir, method="matched_random", seed=23)
            random_summary = random_runner.run()
            self.assertEqual(random_summary["method"], "matched_random")
            self.assertEqual(random_summary["attempt_count"], 12)
            self.assertTrue((random_dir / "training_complete.json").exists())


if __name__ == "__main__":
    unittest.main()