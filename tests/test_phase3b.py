from __future__ import annotations

import unittest
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import torch

from alpha_etf.gpt.checkpointing import CHECKPOINT_SCHEMA_VERSION, build_checkpoint
from alpha_etf.gpt.evaluation import FormulaScoreConfig, artifact_from_row, rescore_loaded_artifacts, score_token_formula
from alpha_etf.gpt.policy import TransformerFormulaPolicy, TransformerPolicyConfig
from alpha_etf.gpt.sampling import PolicyVocab, SamplingConfig, build_action_mask, sample_formulas
from alpha_etf.gpt.vm import StackVM
from alpha_etf.gpt.vocab import FORMULA_VOCAB, VOCAB_VERSION
from alpha_etf.panel import MarketPanel


def _stack_depth(token_ids: list[int]) -> int:
    depth = 0
    for token_id in token_ids:
        token = FORMULA_VOCAB.id_to_token(token_id)
        if token.kind in {"feature", "constant"}:
            depth += 1
        elif token.kind == "operator":
            if depth < token.arity:
                return -1
            depth += 1 - token.arity
    return depth


def _fake_panel() -> MarketPanel:
    symbols = np.array([f"ETF{i:02d}" for i in range(10)])
    features = np.array(["open", "high", "low", "close", "volume", "amount"])
    dates = pd.date_range("2020-01-01", periods=40, freq="D")
    base = np.linspace(1.0, 1.4, len(dates))
    values = np.zeros((len(symbols), len(features), len(dates)), dtype=float)
    for i in range(len(symbols)):
        close = base * (1.0 + i * 0.01)
        values[i, 0, :] = close * 0.99
        values[i, 1, :] = close * 1.01
        values[i, 2, :] = close * 0.98
        values[i, 3, :] = close
        values[i, 4, :] = 1000.0 + i
        values[i, 5, :] = values[i, 3, :] * values[i, 4, :]
    return MarketPanel(
        raw_values=values.copy(),
        qfq_values=values.copy(),
        mask=np.ones((len(symbols), len(dates)), dtype=bool),
        symbols=symbols,
        features=features,
        dates=dates,
    )


class Phase3bTests(unittest.TestCase):
    def test_action_mask_disallows_underflow_and_allows_eos_when_complete(self) -> None:
        policy_vocab = PolicyVocab(FORMULA_VOCAB)
        config = SamplingConfig(max_len=5, min_formula_len=1)
        mask = build_action_mask(
            stack_depths=torch.tensor([0, 1, 2]),
            formula_lengths=torch.tensor([0, 1, 1]),
            done=torch.tensor([False, False, False]),
            policy_vocab=policy_vocab,
            config=config,
        )
        add_id = policy_vocab.to_model_id(FORMULA_VOCAB.name_to_id("ADD"))
        close_id = policy_vocab.to_model_id(FORMULA_VOCAB.name_to_id("close"))
        self.assertLess(mask[0, add_id].item(), -1e8)
        self.assertEqual(mask[0, close_id].item(), 0.0)
        self.assertEqual(mask[1, policy_vocab.eos_id].item(), 0.0)
        self.assertEqual(mask[2, add_id].item(), 0.0)

    def test_sampler_outputs_valid_rpn_prefixes(self) -> None:
        torch.manual_seed(1)
        policy_vocab = PolicyVocab(FORMULA_VOCAB)
        model_config = TransformerPolicyConfig(
            model_vocab_size=policy_vocab.size,
            max_sequence_len=7,
            d_model=16,
            num_layers=1,
            num_heads=4,
            ff_dim=32,
            dropout=0.0,
        )
        model = TransformerFormulaPolicy(model_config)
        sample = sample_formulas(
            model=model,
            batch_size=4,
            policy_vocab=policy_vocab,
            config=SamplingConfig(max_len=5, min_formula_len=1),
            device=torch.device("cpu"),
        )
        self.assertEqual(sample.log_prob_sums.shape, (4,))
        self.assertEqual(len(sample.formulas), 4)
        for formula in sample.formulas:
            self.assertLessEqual(len(formula), 5)
            self.assertEqual(_stack_depth(formula), 1)

    def test_checkpoint_schema_contains_required_sections(self) -> None:
        policy_vocab = PolicyVocab(FORMULA_VOCAB)
        model_config = TransformerPolicyConfig(
            model_vocab_size=policy_vocab.size,
            max_sequence_len=7,
            d_model=16,
            num_layers=1,
            num_heads=4,
            ff_dim=32,
            dropout=0.0,
        )
        model = TransformerFormulaPolicy(model_config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        checkpoint = build_checkpoint(
            step=1,
            model=model,
            optimizer=optimizer,
            model_config=model_config.to_dict(),
            policy_vocab={"token_names": policy_vocab.token_names, "special_tokens": policy_vocab.special_tokens},
            formula_vocab={"vocab_version": VOCAB_VERSION, "token_names": FORMULA_VOCAB.token_names},
            scorer_config={"horizon": 10, "reward_name": "scorer_mean_return"},
            train_config={"batch_size": 4},
            best_formulas=[],
            best_reward=None,
            run_id="unit",
        )
        self.assertEqual(checkpoint["schema_version"], CHECKPOINT_SCHEMA_VERSION)
        for key in ("model_state_dict", "optimizer_state_dict", "model_config", "policy_vocab", "rng_state"):
            self.assertIn(key, checkpoint)
        self.assertIn("torch_cuda_random_state_all", checkpoint["rng_state"])
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/checkpoint.pt"
            torch.save(checkpoint, path)
            loaded = torch.load(path, map_location="cpu")
        self.assertEqual(loaded["schema_version"], CHECKPOINT_SCHEMA_VERSION)

    def test_artifact_reload_reward_matches(self) -> None:
        panel = _fake_panel()
        vm = StackVM()
        score_config = FormulaScoreConfig(horizon=5, min_coverage=0.2)
        token_ids = FORMULA_VOCAB.encode(["close", "close", "DELAY1", "DIV", "CONST_1", "SUB"])
        row, _ = score_token_formula("unit_formula", "unit", token_ids, panel, vm, score_config)
        self.assertTrue(row["valid"])
        artifact = artifact_from_row(row, score_config, "2026-01-01T00:00:00+00:00")
        rewards = rescore_loaded_artifacts([artifact], panel, vm, score_config)
        self.assertAlmostEqual(float(row["reward"]), rewards["unit_formula"], places=12)
        changed_config = FormulaScoreConfig(horizon=5, min_coverage=0.99)
        with self.assertRaises(RuntimeError):
            rescore_loaded_artifacts([artifact], panel, vm, changed_config)


if __name__ == "__main__":
    unittest.main()