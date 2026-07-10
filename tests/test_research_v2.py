from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from alpha_etf.gpt.checkpointing import build_checkpoint, validate_checkpoint_contract
from alpha_etf.gpt.evaluation import FormulaScoreConfig, artifact_from_row, validate_artifact
from alpha_etf.gpt.torch_vm import BatchTorchVM, TorchMarketPanel, formulas_to_tensor
from alpha_etf.gpt.vm import StackVM
from alpha_etf.gpt.vocab import FORMULA_VOCAB
from alpha_etf.panel import MarketPanel
from alpha_etf.research_v2.spec import (
    DATASET_SCHEMA_VERSION,
    PRICE_FEATURES,
    canonical_sha256,
    load_price_panel,
    sha256_file,
)
from alpha_etf.research_v2.vocab import FORMULA_VOCAB_V2, VOCAB_VERSION_V2


class ResearchV2Tests(unittest.TestCase):
    def test_v1_token_ids_are_frozen_and_v2_has_no_flow_tokens(self) -> None:
        expected_v1 = (
            ("open", "feature", 0, "qfq", None),
            ("high", "feature", 0, "qfq", None),
            ("low", "feature", 0, "qfq", None),
            ("close", "feature", 0, "qfq", None),
            ("volume", "feature", 0, "raw", None),
            ("amount", "feature", 0, "raw", None),
            ("CONST_0", "constant", 0, "", 0.0),
            ("CONST_1", "constant", 0, "", 1.0),
            ("ADD", "operator", 2, "", None),
            ("SUB", "operator", 2, "", None),
            ("MUL", "operator", 2, "", None),
            ("DIV", "operator", 2, "", None),
            ("NEG", "operator", 1, "", None),
            ("ABS", "operator", 1, "", None),
            ("SIGN", "operator", 1, "", None),
            ("DELAY1", "operator", 1, "", None),
            ("DELAY5", "operator", 1, "", None),
            ("DELAY10", "operator", 1, "", None),
            ("MA5", "operator", 1, "", None),
            ("MA10", "operator", 1, "", None),
            ("MA20", "operator", 1, "", None),
            ("STD10", "operator", 1, "", None),
            ("RET5", "operator", 1, "", None),
            ("RET10", "operator", 1, "", None),
            ("DECAY", "operator", 1, "", None),
        )
        actual_v1 = tuple(
            (token.name, token.kind, token.arity, token.source, token.value)
            for token in FORMULA_VOCAB.tokens
        )
        self.assertEqual(actual_v1, expected_v1)
        self.assertEqual(FORMULA_VOCAB.name_to_id("volume"), 4)
        self.assertEqual(FORMULA_VOCAB.name_to_id("amount"), 5)
        self.assertEqual(FORMULA_VOCAB.name_to_id("CONST_0"), 6)
        self.assertEqual(FORMULA_VOCAB.name_to_id("ADD"), 8)
        self.assertEqual(FORMULA_VOCAB_V2.version, VOCAB_VERSION_V2)
        self.assertNotIn("volume", FORMULA_VOCAB_V2.token_names)
        self.assertNotIn("amount", FORMULA_VOCAB_V2.token_names)
        self.assertEqual(FORMULA_VOCAB_V2.name_to_id("CONST_0"), 4)

    def test_v2_artifact_requires_v2_vocab_and_exact_research_spec(self) -> None:
        score_config = FormulaScoreConfig()
        token_ids = FORMULA_VOCAB_V2.encode(["close", "RET5"])
        row = {
            "formula_id": "v2_test",
            "token_ids": " ".join(str(item) for item in token_ids),
            "reward": 0.01,
            "valid": True,
        }
        research_spec = {"schema_version": "etf-research-spec-v2", "dataset": {"id": "x"}}
        artifact = artifact_from_row(
            row,
            score_config,
            "2026-01-01T00:00:00Z",
            FORMULA_VOCAB_V2,
            research_spec,
        )
        with self.assertRaises(RuntimeError):
            artifact_from_row(
                row,
                score_config,
                "2026-01-01T00:00:00Z",
                FORMULA_VOCAB_V2,
            )
        self.assertEqual(
            validate_artifact(artifact, score_config, FORMULA_VOCAB_V2, research_spec), token_ids
        )
        with self.assertRaises(RuntimeError):
            validate_artifact(artifact, score_config, FORMULA_VOCAB)
        with self.assertRaises(RuntimeError):
            validate_artifact(artifact, score_config, FORMULA_VOCAB_V2)
        with self.assertRaises(RuntimeError):
            validate_artifact(artifact, score_config, FORMULA_VOCAB_V2, {"dataset": {"id": "y"}})

    def test_v2_checkpoint_rejects_v1_or_different_research_contract(self) -> None:
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters())
        research_spec = {"schema_version": "etf-research-spec-v2", "dataset": {"id": "x"}}
        common = {
            "step": 1,
            "model": model,
            "optimizer": optimizer,
            "model_config": {},
            "policy_vocab": {},
            "formula_vocab": {
                "vocab_version": FORMULA_VOCAB_V2.version,
                "token_names": FORMULA_VOCAB_V2.token_names,
            },
            "scorer_config": {},
            "train_config": {},
            "best_formulas": [],
            "best_reward": None,
            "run_id": "test",
        }
        checkpoint = build_checkpoint(**common, research_spec=research_spec)
        validate_checkpoint_contract(
            checkpoint, vocab=FORMULA_VOCAB_V2, research_spec=research_spec
        )
        with self.assertRaises(RuntimeError):
            validate_checkpoint_contract(
                checkpoint,
                vocab=FORMULA_VOCAB_V2,
                research_spec={"schema_version": "etf-research-spec-v2", "dataset": {"id": "y"}},
            )
        legacy = build_checkpoint(**common)
        with self.assertRaises(RuntimeError):
            validate_checkpoint_contract(
                legacy, vocab=FORMULA_VOCAB_V2, research_spec=research_spec
            )

    def test_v2_dataset_manifest_and_panel_hash_are_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            panel_path = root / "panel.npz"
            values = np.array(
                [[
                    [1.0, np.nan],
                    [1.1, np.nan],
                    [0.9, np.nan],
                    [1.0, np.nan],
                ]],
                dtype=float,
            )
            mask = np.array([[True, False]])
            np.savez_compressed(
                panel_path,
                values=values,
                mask=mask,
                symbols=np.array(["510000"]),
                features=np.array(PRICE_FEATURES),
                dates=np.array(["2026-01-05", "2026-01-06"]),
            )
            identity = {
                "schema_version": DATASET_SCHEMA_VERSION,
                "panel_file": panel_path.name,
                "panel_sha256": sha256_file(panel_path),
                "source_file": "synthetic",
                "source_sha256": "x",
                "phase1b_summary_file": "synthetic",
                "phase1b_summary_sha256": "y",
                "features": list(PRICE_FEATURES),
                "price_adjustment": "multiplicative_event_qfq",
                "flow_features_exposed": False,
                "tradable_mask": "test",
                "symbols": ["510000"],
                "date_start": "2026-01-05",
                "date_end": "2026-01-06",
                "date_count": 2,
                "tradable_observations": 1,
                "panel_shape": [1, 4, 2],
            }
            fingerprint = canonical_sha256(identity)
            manifest = {
                **identity,
                "dataset_id": f"etf-price-event-v2-{fingerprint[:12]}",
                "dataset_fingerprint": fingerprint,
            }
            (root / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            panel = load_price_panel(root)
            self.assertEqual(panel.qfq_values.shape, (1, 4, 2))
            self.assertIsInstance(panel.dates, pd.DatetimeIndex)
            manifest["panel_sha256"] = "bad"
            (root / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                load_price_panel(root)

    def test_cpu_and_torch_vm_match_under_v2_vocab(self) -> None:
        rng = np.random.default_rng(7)
        values = rng.uniform(1.0, 3.0, size=(3, 4, 40)).astype(float)
        mask = np.ones((3, 40), dtype=bool)
        panel = MarketPanel(
            raw_values=values.copy(),
            qfq_values=values,
            mask=mask,
            symbols=np.array(["a", "b", "c"]),
            features=np.array(PRICE_FEATURES),
            dates=pd.date_range("2026-01-01", periods=40),
        )
        formula = FORMULA_VOCAB_V2.encode(["close", "RET5", "MA5"])
        cpu = StackVM(FORMULA_VOCAB_V2).execute(formula, panel)
        torch_panel = TorchMarketPanel.from_market_panel(
            panel, device=torch.device("cpu"), dtype=torch.float32
        )
        tokens, lengths = formulas_to_tensor([formula], device=torch.device("cpu"))
        gpu_style = BatchTorchVM(FORMULA_VOCAB_V2).execute(tokens, lengths, torch_panel)
        self.assertTrue(cpu.valid)
        self.assertTrue(bool(gpu_style.valid[0].item()))
        np.testing.assert_allclose(
            cpu.signal,
            gpu_style.signal[0].numpy(),
            rtol=1e-5,
            atol=1e-6,
            equal_nan=True,
        )


if __name__ == "__main__":
    unittest.main()