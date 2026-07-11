from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import torch

from alpha_etf.research_v3a.scoring import (
    ScorerConfig,
    SplitSpec,
    build_forward_targets,
    score_signal,
)
from alpha_etf.research_v3a.torch_scoring import (
    SCORE_INSUFFICIENT_DAILY_SIGNAL,
    TorchForwardTargets,
    score_signal_batch,
)


def _market() -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, np.ndarray]:
    assets = 4
    dates = pd.date_range("2026-01-01", periods=16, freq="D")
    opens = np.empty((assets, len(dates)), dtype=float)
    for asset in range(assets):
        opens[asset] = 10.0 + asset + np.arange(len(dates)) * (0.10 + asset * 0.01)
    mask = np.ones_like(opens, dtype=bool)
    symbols = np.array(["A", "B", "C", "D"])
    return opens, mask, dates, symbols


class V3AScoringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ScorerConfig(
            horizon=2,
            top_fraction=0.5,
            min_top_k=2,
            max_top_k=2,
            min_universe=4,
        )
        self.split = SplitSpec("unit", "2026-01-01", "2026-01-16")

    def test_selection_uses_only_decision_day_and_unbuyable_slot_is_cash(self) -> None:
        opens, mask, dates, symbols = _market()
        decision = 2
        mask[0, decision + 1] = False
        mask[1, decision + 1 + self.config.horizon] = False
        targets = build_forward_targets(opens, mask, dates, self.split, self.config)
        target_row = int(np.where(targets.decision_indices == decision)[0][0])

        signal = np.tile(np.array([[4.0], [3.0], [2.0], [1.0]]), (1, len(dates)))
        result = score_signal("unit", signal, targets, dates, symbols, self.config)
        self.assertTrue(result.valid)
        row = result.daily.loc[result.daily["decision_index"] == decision].iloc[0]
        self.assertEqual(row["selected_indices"], (0, 1))
        self.assertEqual(row["selected_returns"][0], 0.0)
        self.assertEqual(row["actual_sell_dates"][0], "cash")
        self.assertEqual(row["actual_sell_dates"][1], dates[decision + 4].date().isoformat())
        self.assertAlmostEqual(
            row["baseline_return"],
            float(np.mean(targets.forward_returns[target_row, targets.available[target_row]])),
        )

    def test_common_scorer_dates_cannot_be_dropped_by_formula(self) -> None:
        opens, mask, dates, symbols = _market()
        targets = build_forward_targets(opens, mask, dates, self.split, self.config)
        signal = np.tile(np.array([[4.0], [3.0], [2.0], [1.0]]), (1, len(dates)))
        valid = score_signal("valid", signal, targets, dates, symbols, self.config)
        self.assertEqual(len(valid.daily), targets.days)
        failed_decision = int(targets.decision_indices[3])
        signal[:3, failed_decision] = np.nan
        invalid = score_signal("missing", signal, targets, dates, symbols, self.config)
        self.assertFalse(invalid.valid)
        self.assertEqual(invalid.invalid_reason, "insufficient_daily_signal")
        self.assertEqual(invalid.summary["failed_decision_index"], failed_decision)

    def test_split_boundary_labels_do_not_cross(self) -> None:
        opens, mask, dates, _ = _market()
        targets = build_forward_targets(opens, mask, dates, self.split, self.config)
        self.assertLessEqual(
            int(targets.planned_sell_indices.max()), int(np.where(dates <= pd.Timestamp(self.split.end))[0][-1])
        )
        self.assertEqual(int(targets.decision_indices.max()), len(dates) - self.config.horizon - 2)

    def test_open_ended_final_split_uses_panel_end(self) -> None:
        opens, mask, dates, _ = _market()
        targets = build_forward_targets(
            opens, mask, dates, SplitSpec("final", "2026-01-01", None), self.config
        )
        self.assertGreater(targets.days, 0)

    def test_all_cash_boundary_day_remains_in_common_dates(self) -> None:
        opens, mask, dates, _ = _market()
        decision = len(dates) - 2
        mask[:, decision + 1] = False
        targets = build_forward_targets(opens, mask, dates, self.split, self.config)
        row = int(np.where(targets.decision_indices == decision)[0][0])
        self.assertTrue(np.all(targets.forward_returns[row, targets.available[row]] == 0.0))
        signal = np.tile(np.arange(4, 0, -1, dtype=float)[:, None], (1, len(dates)))
        result = score_signal("all_cash", signal, targets, dates, np.array(list("ABCD")), self.config)
        self.assertTrue(result.valid)
        daily = result.daily.loc[result.daily["decision_index"] == decision].iloc[0]
        self.assertEqual(daily["planned_sell_date"], "cash")
        self.assertEqual(daily["absolute_return"], 0.0)

    def test_torch_batch_matches_cpu_and_stable_ties(self) -> None:
        opens, mask, dates, symbols = _market()
        targets = build_forward_targets(opens, mask, dates, self.split, self.config)
        signal_a = np.tile(np.array([[2.0], [2.0], [1.0], [0.0]]), (1, len(dates)))
        signal_b = np.tile(np.array([[1.0], [2.0], [3.0], [4.0]]), (1, len(dates)))
        cpu_a = score_signal("a", signal_a, targets, dates, symbols, self.config)
        cpu_b = score_signal("b", signal_b, targets, dates, symbols, self.config)
        torch_targets = TorchForwardTargets.from_numpy(
            targets, device=torch.device("cpu"), dtype=torch.float32
        )
        batch = score_signal_batch(
            torch.as_tensor(np.stack([signal_a, signal_b]), dtype=torch.float32),
            torch.tensor([True, True]),
            torch_targets,
            self.config,
        )
        self.assertTrue(bool(batch.valid.all().item()))
        self.assertEqual(tuple(batch.selected_indices[0, 0, :2].tolist()), (0, 1))
        self.assertAlmostEqual(float(batch.reward[0]), cpu_a.reward, places=6)
        self.assertAlmostEqual(float(batch.reward[1]), cpu_b.reward, places=6)
        np.testing.assert_allclose(
            batch.daily_excess_return[0].numpy(),
            cpu_a.daily["excess_return"].to_numpy(),
            rtol=0.0,
            atol=1e-6,
        )

    def test_torch_marks_formula_with_missing_common_day_invalid(self) -> None:
        opens, mask, dates, _ = _market()
        targets = build_forward_targets(opens, mask, dates, self.split, self.config)
        signal = np.ones((1, 4, len(dates)), dtype=float)
        signal[0, :3, targets.decision_indices[0]] = np.nan
        torch_targets = TorchForwardTargets.from_numpy(
            targets, device=torch.device("cpu"), dtype=torch.float32
        )
        result = score_signal_batch(
            torch.as_tensor(signal, dtype=torch.float32),
            torch.tensor([True]),
            torch_targets,
            self.config,
        )
        self.assertFalse(bool(result.valid[0].item()))
        self.assertEqual(
            int(result.invalid_code[0].item()), SCORE_INSUFFICIENT_DAILY_SIGNAL
        )


if __name__ == "__main__":
    unittest.main()