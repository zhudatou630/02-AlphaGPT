import unittest

import numpy as np
import pandas as pd

from alpha_etf.research_v3a.validation import ValidationConfig, run_formula_validation
from scripts.v3a.run_expanded_absroc40_phase2 import _monthly_equal_weight


class ExpandedPhaseTwoBacktestTest(unittest.TestCase):
    def test_prior_close_order_uses_execution_price_mask_not_same_day_signal_mask(self) -> None:
        dates = pd.DatetimeIndex(
            pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
        )
        symbols = np.asarray(["510000", "510001", "510002"])
        signal = np.asarray(
            [[3.0, np.nan, np.nan], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        prices = np.ones((3, 3), dtype=np.float64)
        eligibility = np.asarray(
            [[True, False, False], [True, True, True], [True, True, True]]
        )
        execution = np.ones((3, 3), dtype=bool)
        config = ValidationConfig(
            validation_start="2024-01-03",
            validation_end="2024-01-04",
            min_universe=1,
            slots=1,
            buy_rank=1,
            hold_rank=1,
            robust_z_threshold=0.0,
        )

        _, trades, _ = run_formula_validation(
            formula_id="test",
            signal=signal,
            open_prices=prices,
            close_prices=prices,
            tradable_mask=eligibility,
            execution_mask=execution,
            dates=dates,
            symbols=symbols,
            config=config,
        )

        self.assertEqual(trades["action"].tolist()[:2], ["BUY", "SELL"])
        self.assertEqual(trades.iloc[0]["date"], "2024-01-03")
        self.assertEqual(trades.iloc[0]["decision_date"], "2024-01-02")

    def test_monthly_benchmark_cannot_reuse_terminal_cash_at_same_open(self) -> None:
        dates = pd.DatetimeIndex(
            pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-02-01"])
        )
        symbols = np.asarray(["510000", "510001"])
        prices = np.asarray([[1.0, 1.0, 1.0, np.nan], [1.0, 1.0, 1.0, 1.0]])
        price_mask = np.isfinite(prices)
        eligibility = np.asarray(
            [[True, True, False, False], [False, False, True, True]]
        )
        terminal = pd.DataFrame(
            [
                {
                    "fund_code": "510000",
                    "payment_date": "2024-02-01",
                    "cash_per_share": 1.0,
                    "is_final_payment": True,
                }
            ]
        )

        result = _monthly_equal_weight(
            open_prices=prices,
            close_prices=prices,
            price_mask=price_mask,
            eligibility_mask=eligibility,
            dates=dates,
            start="2024-01-03",
            end="2024-02-01",
            min_universe=1,
            symbols=symbols,
            terminal=terminal,
        )

        self.assertEqual(int(result.iloc[-1]["benchmark_position_count"]), 0)
        self.assertAlmostEqual(float(result.iloc[-1]["benchmark_cash"]), 1.0)

    def test_final_terminal_cash_closes_position_at_announced_value(self) -> None:
        dates = pd.DatetimeIndex(pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]))
        symbols = np.asarray(["510000", "510001", "510002"])
        signal = np.asarray(
            [[3.0, 3.0, np.nan], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        prices = np.ones((3, 3), dtype=np.float64)
        mask = np.asarray(
            [[True, True, False], [True, True, True], [True, True, True]]
        )
        terminal = pd.DataFrame(
            [
                {
                    "fund_code": "510000",
                    "payment_date": "2024-01-04",
                    "cash_per_share": 1.1,
                    "is_final_payment": True,
                    "evidence_type": "announcement_per_share",
                }
            ]
        )
        config = ValidationConfig(
            validation_start="2024-01-03",
            validation_end="2024-01-04",
            min_universe=1,
            slots=1,
            buy_rank=1,
            hold_rank=1,
            robust_z_threshold=0.0,
        )

        daily, trades, _ = run_formula_validation(
            formula_id="test",
            signal=signal,
            open_prices=prices,
            close_prices=prices,
            tradable_mask=mask,
            dates=dates,
            symbols=symbols,
            config=config,
            terminal_cashflows=terminal,
        )

        self.assertEqual(trades["action"].tolist(), ["BUY", "TERMINAL_CASH"])
        self.assertAlmostEqual(float(daily.iloc[-1]["equity"]), 1.1)
        self.assertAlmostEqual(float(trades.iloc[-1]["pnl_pct"]), 0.1)

    def test_partial_terminal_cash_reduces_residual_mark_until_final_payment(self) -> None:
        dates = pd.DatetimeIndex(
            pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
        )
        symbols = np.asarray(["510000", "510001", "510002"])
        signal = np.asarray(
            [
                [3.0, 3.0, np.nan, np.nan],
                [1.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        prices = np.ones((3, 4), dtype=np.float64)
        mask = np.asarray(
            [
                [True, True, False, False],
                [True, True, True, True],
                [True, True, True, True],
            ]
        )
        terminal = pd.DataFrame(
            [
                {
                    "fund_code": "510000",
                    "payment_date": "2024-01-04",
                    "cash_per_share": 0.4,
                    "is_final_payment": False,
                    "evidence_type": "announcement_per_share",
                },
                {
                    "fund_code": "510000",
                    "payment_date": "2024-01-05",
                    "cash_per_share": 0.7,
                    "is_final_payment": True,
                    "evidence_type": "announcement_per_share",
                },
            ]
        )
        config = ValidationConfig(
            validation_start="2024-01-03",
            validation_end="2024-01-05",
            min_universe=1,
            slots=1,
            buy_rank=1,
            hold_rank=1,
            robust_z_threshold=0.0,
        )

        daily, trades, _ = run_formula_validation(
            formula_id="test",
            signal=signal,
            open_prices=prices,
            close_prices=prices,
            tradable_mask=mask,
            dates=dates,
            symbols=symbols,
            config=config,
            terminal_cashflows=terminal,
        )

        self.assertEqual(
            trades["action"].tolist(), ["BUY", "DISTRIBUTION", "TERMINAL_CASH"]
        )
        self.assertAlmostEqual(float(daily.iloc[1]["equity"]), 1.0)
        self.assertAlmostEqual(float(daily.iloc[2]["equity"]), 1.1)
        self.assertAlmostEqual(float(trades.iloc[-1]["pnl_pct"]), 0.1)


if __name__ == "__main__":
    unittest.main()