from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import torch

from alpha_etf.gpt.evaluation import FormulaScoreConfig, score_token_formula
from alpha_etf.gpt.policy import TransformerFormulaPolicy, TransformerPolicyConfig
from alpha_etf.gpt.sampling import PolicyVocab, SamplingConfig, sample_formulas
from alpha_etf.gpt.torch_scoring import score_vm_batch
from alpha_etf.gpt.torch_vm import BatchTorchVM, TorchMarketPanel, formulas_to_tensor
from alpha_etf.gpt.vm import StackVM
from alpha_etf.gpt.vocab import FORMULA_VOCAB
from alpha_etf.panel import MarketPanel


def _fake_panel() -> MarketPanel:
    symbols = np.array([f"ETF{i:02d}" for i in range(12)])
    features = np.array(["open", "high", "low", "close", "volume", "amount"])
    dates = pd.date_range("2020-01-01", periods=80, freq="D")
    trend = np.linspace(1.0, 1.8, len(dates))
    values = np.zeros((len(symbols), len(features), len(dates)), dtype=float)
    for i in range(len(symbols)):
        asset_scale = 1.0 + i * 0.025
        seasonal = 1.0 + 0.01 * np.sin(np.linspace(0.0, 8.0, len(dates)) + i * 0.2)
        close = trend * asset_scale * seasonal
        values[i, 0, :] = close * 0.995
        values[i, 1, :] = close * 1.015
        values[i, 2, :] = close * 0.985
        values[i, 3, :] = close
        values[i, 4, :] = 1000.0 + i * 20.0 + np.arange(len(dates)) * (1.0 + i * 0.05)
        values[i, 5, :] = values[i, 3, :] * values[i, 4, :]
    return MarketPanel(
        raw_values=values.copy(),
        qfq_values=values.copy(),
        mask=np.ones((len(symbols), len(dates)), dtype=bool),
        symbols=symbols,
        features=features,
        dates=dates,
    )


class Phase3cGpuTests(unittest.TestCase):
    def test_rmsnorm_swiglu_policy_samples_and_backprops(self) -> None:
        torch.manual_seed(7)
        policy_vocab = PolicyVocab(FORMULA_VOCAB)
        model = TransformerFormulaPolicy(
            TransformerPolicyConfig(
                model_vocab_size=policy_vocab.size,
                max_sequence_len=8,
                d_model=32,
                num_layers=2,
                num_heads=4,
                ff_dim=64,
                dropout=0.0,
                use_rmsnorm=True,
                use_swiglu=True,
            )
        )
        sample = sample_formulas(
            model=model,
            batch_size=4,
            policy_vocab=policy_vocab,
            config=SamplingConfig(max_len=6, min_formula_len=1),
            device=torch.device("cpu"),
        )
        loss = -sample.log_prob_sums.mean()
        loss.backward()
        grad_norm = sum(float(p.grad.abs().sum().item()) for p in model.parameters() if p.grad is not None)
        self.assertGreater(grad_norm, 0.0)

    def test_torch_vm_matches_cpu_vm_on_fixed_formulas(self) -> None:
        panel = _fake_panel()
        torch_panel = TorchMarketPanel.from_market_panel(panel, device=torch.device("cpu"), dtype=torch.float32)
        formulas = [
            FORMULA_VOCAB.encode(["close"]),
            FORMULA_VOCAB.encode(["low", "ABS", "DECAY", "MA10"]),
            FORMULA_VOCAB.encode(["close", "MA5", "open", "MA5", "SUB", "STD10"]),
            FORMULA_VOCAB.encode(["close", "close", "DELAY5", "DIV", "CONST_1", "SUB"]),
        ]
        token_tensor, lengths = formulas_to_tensor(formulas, device=torch.device("cpu"))
        gpu_result = BatchTorchVM().execute(token_tensor, lengths, torch_panel)
        cpu_vm = StackVM()
        self.assertTrue(bool(gpu_result.valid.all().item()))
        for i, formula in enumerate(formulas):
            cpu_result = cpu_vm.execute(formula, panel)
            self.assertTrue(cpu_result.valid, cpu_result.invalid_reason)
            np.testing.assert_allclose(
                gpu_result.signal[i].numpy(),
                cpu_result.signal,
                rtol=1e-5,
                atol=1e-6,
                equal_nan=True,
            )

    def test_torch_scorer_is_close_to_cpu_scorer_on_small_formula_set(self) -> None:
        panel = _fake_panel()
        torch_panel = TorchMarketPanel.from_market_panel(panel, device=torch.device("cpu"), dtype=torch.float32)
        formulas = [
            FORMULA_VOCAB.encode(["close"]),
            FORMULA_VOCAB.encode(["volume"]),
            FORMULA_VOCAB.encode(["low", "ABS", "DECAY", "MA10"]),
            FORMULA_VOCAB.encode(["close", "close", "DELAY5", "DIV", "CONST_1", "SUB"]),
        ]
        token_tensor, lengths = formulas_to_tensor(formulas, device=torch.device("cpu"))
        gpu_vm = BatchTorchVM().execute(token_tensor, lengths, torch_panel)
        score_config = FormulaScoreConfig(horizon=5, min_coverage=0.2)
        gpu_score = score_vm_batch(gpu_vm, torch_panel, score_config)

        cpu_vm = StackVM()
        for i, formula in enumerate(formulas):
            cpu_row, _ = score_token_formula(f"formula_{i}", "unit", formula, panel, cpu_vm, score_config)
            self.assertEqual(bool(cpu_row["valid"]), bool(gpu_score.valid[i].item()))
            if cpu_row["valid"]:
                self.assertAlmostEqual(float(cpu_row["reward"]), float(gpu_score.reward[i].item()), places=5)


if __name__ == "__main__":
    unittest.main()
