from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from alpha_etf.research_v3a.validation import (
    ValidationConfig,
    decide_validation,
    run_formula_validation,
)


class V3AValidationTests(unittest.TestCase):
    def _fixture(self):
        dates = pd.DatetimeIndex(
            pd.to_datetime(["2021-12-31", "2022-01-04", "2022-01-05", "2022-01-06"])
        )
        symbols = np.asarray(["A", "B", "C", "D", "E"])
        mask = np.ones((5, 4), dtype=bool)
        opens = np.full((5, 4), 10.0)
        closes = opens.copy()
        signal = np.asarray(
            [
                [10.0, 10.0, -5.0, -5.0],
                [8.0, 8.0, 8.0, 8.0],
                [6.0, 6.0, 6.0, 6.0],
                [0.0, 0.0, 10.0, 10.0],
                [-2.0, -2.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        config = ValidationConfig(
            validation_start="2022-01-01",
            validation_end="2022-12-31",
            min_universe=5,
            robust_z_threshold=0.0,
            hold_rank=3,
        )
        return dates, symbols, mask, opens, closes, signal, config

    def test_initial_entry_rank_exit_and_local_cash_pool(self) -> None:
        dates, symbols, mask, opens, closes, signal, config = self._fixture()
        opens[2, 1] = np.nan
        mask[2, 1] = False
        daily, trades, summary = run_formula_validation(
            formula_id="fixture",
            signal=signal,
            open_prices=opens,
            close_prices=closes,
            tradable_mask=mask,
            dates=dates,
            symbols=symbols,
            config=config,
        )
        first_buys = trades[(trades["date"] == "2022-01-04") & (trades["action"] == "BUY")]
        self.assertEqual(list(first_buys["symbol"]), ["A", "B"])
        self.assertTrue(np.allclose(first_buys["notional"], [1 / 3, 1 / 3]))
        self.assertAlmostEqual(float(daily.iloc[0]["cash"]), 1 / 3)
        sells = trades[trades["action"] == "SELL"]
        self.assertEqual(list(sells["symbol"]), ["A"])
        self.assertEqual(list(sells["date"]), ["2022-01-06"])
        self.assertEqual(list(sells["reason"]), ["rank_exit"])
        self.assertEqual(summary["validation_days"], 3)
        self.assertFalse(summary["final_metrics_read"])

    def test_stop_loss_waits_for_open_and_reentry_requires_lost_eligibility(self) -> None:
        dates = pd.DatetimeIndex(
            pd.to_datetime(
                [
                    "2021-12-31",
                    "2022-01-04",
                    "2022-01-05",
                    "2022-01-06",
                    "2022-01-07",
                    "2022-01-10",
                ]
            )
        )
        symbols = np.asarray(["A", "B", "C", "D", "E"])
        mask = np.ones((5, len(dates)), dtype=bool)
        opens = np.full(mask.shape, 10.0)
        closes = opens.copy()
        closes[0, 1] = 9.0
        mask[0, 2] = False
        opens[0, 2] = np.nan
        signal = np.asarray(
            [
                [10.0, 10.0, 10.0, -5.0, 10.0, 10.0],
                [8.0, 8.0, 8.0, 10.0, 8.0, 8.0],
                [6.0, 6.0, 6.0, 8.0, 6.0, 6.0],
                [0.0, 0.0, 0.0, 6.0, 0.0, 0.0],
                [-2.0, -2.0, -2.0, 0.0, -2.0, -2.0],
            ],
            dtype=np.float32,
        )
        config = ValidationConfig(
            validation_start="2022-01-01",
            validation_end="2022-12-31",
            min_universe=5,
            robust_z_threshold=0.0,
            hold_rank=3,
        )
        _, trades, _ = run_formula_validation(
            formula_id="stop",
            signal=signal,
            open_prices=opens,
            close_prices=closes,
            tradable_mask=mask,
            dates=dates,
            symbols=symbols,
            config=config,
        )
        a_trades = trades[trades["symbol"] == "A"]
        self.assertEqual(list(a_trades["action"]), ["BUY", "SELL", "BUY"])
        self.assertEqual(
            list(a_trades["date"]), ["2022-01-04", "2022-01-06", "2022-01-10"]
        )
        self.assertEqual(a_trades.iloc[1]["reason"], "stop_loss")

    def test_decision_defaults_simple_and_requires_random_gate(self) -> None:
        def rows(total_return: float, drawdown: float = 0.1):
            return [
                {"total_return": total_return + offset, "max_drawdown": drawdown}
                for offset in (-0.01, 0.0, 0.01, 0.02)
            ]

        transformer = {
            "paired_simple": {seed: rows(0.10) for seed in ("101", "102", "103")},
            "stable_complex": {seed: rows(0.11) for seed in ("101", "102", "103")},
            "original_top50": {seed: rows(0.12) for seed in ("101", "102", "103")},
        }
        random = {
            rule: {seed: rows(0.05) for seed in ("101", "102", "103")}
            for rule in transformer
        }
        decision = decide_validation(
            transformer=transformer,
            random=random,
            benchmark={"total_return": 0.02, "max_drawdown": 0.15},
            stable_pair_differences={seed: [0.01] for seed in ("101", "102", "103")},
        )
        self.assertEqual(decision["outcome"], "winner")
        self.assertEqual(decision["winner"], "paired_simple")
        self.assertFalse(decision["upgrades"]["stable_complex"]["passed"])


if __name__ == "__main__":
    unittest.main()