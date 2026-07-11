from __future__ import annotations

import random
from argparse import Namespace
import json
from pathlib import Path
import tempfile
import unittest

import torch

from alpha_etf.research_v3a.artifacts import build_formula_artifact
from alpha_etf.research_v3a.candidates import CandidateConfig
from alpha_etf.research_v3a.candidates import build_candidate_record
from alpha_etf.research_v3a.language import FORMULA_VOCAB
from alpha_etf.research_v3a.sampling import generate_random_formula
from alpha_etf.research_v3a.torch_scoring import TorchForwardTargets
from scripts.v3a import random_baseline
from scripts.v3a import gpu_smoke
from tests.test_v3a_contract import _research_spec


class V3AStageCScriptTests(unittest.TestCase):
    def test_random_sequence_digest_is_reproducible(self) -> None:
        def generate(seed: int) -> tuple[list[list[int]], str]:
            rng = random.Random(seed)
            formulas: list[list[int]] = []
            digest = "00" * 32
            for _ in range(100):
                formula = generate_random_formula(rng)
                formulas.append(formula)
                digest = random_baseline._sequence_digest(digest, formula)
            return formulas, digest

        self.assertEqual(generate(41), generate(41))
        self.assertNotEqual(generate(41), generate(42))

    def test_quality_gate_separates_coverage_and_constant_signals(self) -> None:
        signals = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[1.0, 1.0], [1.0, 1.0]],
                [[1.0, float("nan")], [2.0, 3.0]],
            ],
            dtype=torch.float32,
        )
        targets = TorchForwardTargets(
            decision_indices=torch.tensor([0, 1]),
            available=torch.ones((2, 2), dtype=torch.bool),
            top_k=torch.ones(2, dtype=torch.long),
            forward_returns=torch.zeros((2, 2)),
            baseline_returns=torch.zeros(2),
        )
        valid, coverage, _, variation = random_baseline._quality_metrics(
            signals, targets, CandidateConfig()
        )
        self.assertEqual(valid.tolist(), [True, False, False])
        self.assertEqual(variation.tolist(), [True, False, True])
        self.assertEqual(coverage.tolist(), [1.0, 1.0, 0.75])

    def test_random_state_requires_exact_attempt_reconciliation(self) -> None:
        spec = _research_spec()
        config = {"test": True}
        state = random_baseline._empty_state(
            run_id="unit", config=config, research_spec=spec, seed=41
        )
        random_baseline._validate_state(
            state, run_id="unit", config=config, research_spec=spec
        )
        state["completed_attempts"] = 1
        with self.assertRaises(RuntimeError):
            random_baseline._validate_state(
                state, run_id="unit", config=config, research_spec=spec
            )

    def test_formal_identity_requires_all_four_pins(self) -> None:
        values = {
            "expected_dataset_id": None,
            "expected_panel_sha256": None,
            "expected_research_spec_id": None,
            "expected_code_fingerprint": None,
            "allow_unpinned_identity": False,
        }
        for module in (gpu_smoke, random_baseline):
            with self.subTest(module=module.__name__), self.assertRaises(RuntimeError):
                module._identity_is_pinned(Namespace(**values))
            self.assertFalse(
                module._identity_is_pinned(
                    Namespace(**{**values, "allow_unpinned_identity": True})
                )
            )
            partial = {**values, "expected_dataset_id": "dataset"}
            with self.assertRaises(RuntimeError):
                module._identity_is_pinned(Namespace(**partial))
            pinned = {
                key: "frozen" if key.startswith("expected_") else value
                for key, value in values.items()
            }
            self.assertTrue(module._identity_is_pinned(Namespace(**pinned)))

    def test_ledger_prefix_rejects_truncation_and_index_gaps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "attempts.jsonl"
            content = b"".join(
                (json.dumps({"attempt_index": index}) + "\n").encode("utf-8")
                for index in range(2)
            )
            path.write_bytes(content)
            digest = random_baseline._validate_ledger_prefix(
                path, offset=len(content), expected_lines=2, expected_sha256=None
            )
            path.write_bytes(content[:-1])
            with self.assertRaises(RuntimeError):
                random_baseline._validate_ledger_prefix(
                    path,
                    offset=len(content),
                    expected_lines=2,
                    expected_sha256=digest,
                )
            bad = (
                json.dumps({"attempt_index": 0})
                + "\n"
                + json.dumps({"attempt_index": 2})
                + "\n"
            ).encode("utf-8")
            path.write_bytes(bad)
            with self.assertRaises(RuntimeError):
                random_baseline._validate_ledger_prefix(
                    path, offset=len(bad), expected_lines=2, expected_sha256=None
                )
            invalid_json = b"{not-json}\n"
            path.write_bytes(invalid_json)
            with self.assertRaises(RuntimeError):
                random_baseline._validate_ledger_prefix(
                    path,
                    offset=len(invalid_json),
                    expected_lines=1,
                    expected_sha256=None,
                )
            path.write_bytes(content + b'{"attempt_index":')
            self.assertEqual(
                random_baseline._validate_ledger_prefix(
                    path,
                    offset=len(content),
                    expected_lines=2,
                    expected_sha256=digest,
                ),
                digest,
            )
            corrupt = bytearray(content)
            corrupt[corrupt.index(ord(" "))] = ord("\t")
            path.write_bytes(corrupt)
            with self.assertRaises(RuntimeError):
                random_baseline._validate_ledger_prefix(
                    path,
                    offset=len(content),
                    expected_lines=2,
                    expected_sha256=digest,
                )

    def test_only_registered_cuda_baselines_can_pass_formal_gate(self) -> None:
        self.assertTrue(
            random_baseline._is_formal_baseline_run(
                device=torch.device("cuda"),
                identity_pinned=True,
                seed=41,
                attempts=100_000,
            )
        )
        for device, pinned, seed, attempts in (
            ("cpu", True, 41, 100_000),
            ("cuda", False, 41, 100_000),
            ("cuda", True, 99, 100_000),
            ("cuda", True, 41, 99_999),
        ):
            with self.subTest(
                device=device, pinned=pinned, seed=seed, attempts=attempts
            ):
                self.assertFalse(
                    random_baseline._is_formal_baseline_run(
                        device=torch.device(device),
                        identity_pinned=pinned,
                        seed=seed,
                        attempts=attempts,
                    )
                )

    def test_immutable_artifact_writer_rejects_conflicting_content(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "artifact.json"
            random_baseline._write_bytes_idempotent(path, b"first\n")
            random_baseline._write_bytes_idempotent(path, b"first\n")
            with self.assertRaises(RuntimeError):
                random_baseline._write_bytes_idempotent(path, b"second\n")

    def test_completed_summary_without_marker_is_finalized_idempotently(self) -> None:
        spec = _research_spec()
        state = random_baseline._empty_state(
            run_id="completed", config={"attempts": 0}, research_spec=spec, seed=41
        )
        record = build_candidate_record(
            formula_id="unit",
            source="unit",
            token_ids=FORMULA_VOCAB.encode(["DAYRET"]),
            reward=0.1,
            train_summary={},
        )
        artifact = build_formula_artifact(
            record, research_spec=spec, created_at=state["artifact_created_at"]
        )
        summary = {
            "run_id": "completed",
            "research_spec_id": spec["research_spec_id"],
            "attempt_count": 0,
            "formula_sequence_digest": state["formula_sequence_digest"],
            "attempt_ledger_prefix_sha256": state[
                "attempt_ledger_prefix_sha256"
            ],
            "attempt_ledger_reconciled": True,
            "status_counts": {key: 0 for key in random_baseline.STATUS_KEYS},
        }
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            random_baseline._write_jsonl_atomic(
                run_dir / "top_formulas.jsonl", [artifact]
            )
            random_baseline._write_json_atomic(run_dir / "summary.json", summary)
            first = random_baseline._recover_completed_run(
                run_dir=run_dir, state=state, research_spec=spec
            )
            marker_hash = (run_dir / "complete.json").read_bytes()
            second = random_baseline._recover_completed_run(
                run_dir=run_dir, state=state, research_spec=spec
            )
            self.assertEqual(first, summary)
            self.assertEqual(second, summary)
            self.assertEqual((run_dir / "complete.json").read_bytes(), marker_hash)


if __name__ == "__main__":
    unittest.main()