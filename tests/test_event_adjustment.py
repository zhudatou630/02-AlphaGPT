from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from alpha_etf.data.event_adjustment import (
    apply_event_adjustments,
    build_event_audit,
    panel_arrays,
    quality_gate_failures,
    quality_summary,
    reconcile_flow_data,
)


def _daily(closes: list[float], volumes: list[float] | None = None) -> pd.DataFrame:
    volumes = volumes or [100.0] * len(closes)
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-02", periods=len(closes), freq="B"),
            "symbol": "510000",
            "name": "test ETF",
            "open": closes,
            "high": np.array(closes) + 0.1,
            "low": np.array(closes) - 0.1,
            "close": closes,
            "volume": volumes,
            "amount": np.array(closes) * np.array(volumes),
        }
    )


def _event(date: str, category: int, c1: float = 0.0, c2: float = 0.0, c3: float = 0.0, c4: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [date],
            "symbol": ["510000"],
            "category_code": [category],
            "c1": [c1],
            "c2": [c2],
            "c3": [c3],
            "c4": [c4],
        }
    )


class EventAdjustmentTests(unittest.TestCase):
    def test_share_split_adjusts_price_and_volume(self) -> None:
        daily = _daily([100.0, 102.0, 51.0], [10.0, 12.0, 24.0])
        events = _event("2024-01-04", 11, c3=2.0)
        audit = build_event_audit(daily, events)
        adjusted = apply_event_adjustments(daily, audit)

        self.assertAlmostEqual(audit.iloc[0]["adjusted_event_return"], 0.0)
        self.assertAlmostEqual(audit.iloc[0]["previous_volume_on_post_event_units"], 24.0)
        self.assertAlmostEqual(audit.iloc[0]["adjusted_volume_event_ratio"], 1.0)
        np.testing.assert_allclose(adjusted["event_qfq_close"], [50.0, 51.0, 51.0])
        np.testing.assert_allclose(adjusted["event_adjusted_volume"], [20.0, 24.0, 24.0])
        self.assertAlmostEqual(adjusted.iloc[1]["event_qfq_close_return"], 0.02)

    def test_share_merge_uses_fractional_share_multiplier(self) -> None:
        daily = _daily([10.0, 10.0, 20.0], [100.0, 120.0, 60.0])
        events = _event("2024-01-04", 11, c3=0.5)
        adjusted = apply_event_adjustments(daily, build_event_audit(daily, events))

        np.testing.assert_allclose(adjusted["event_qfq_close"], [20.0, 20.0, 20.0])
        np.testing.assert_allclose(adjusted["event_adjusted_volume"], [50.0, 60.0, 60.0])

    def test_cash_dividend_preserves_pre_event_returns(self) -> None:
        daily = _daily([9.0, 10.0, 9.0])
        events = _event("2024-01-04", 1, c1=10.0)
        adjusted = apply_event_adjustments(daily, build_event_audit(daily, events))

        np.testing.assert_allclose(adjusted["event_qfq_close"], [8.1, 9.0, 9.0])
        self.assertAlmostEqual(adjusted.iloc[1]["event_qfq_close_return"], 10.0 / 9.0 - 1.0)
        self.assertAlmostEqual(adjusted.iloc[2]["event_qfq_close_return"], 0.0)
        np.testing.assert_allclose(adjusted["event_adjusted_volume"], [100.0, 100.0, 100.0])

    def test_bonus_share_event_adjusts_volume(self) -> None:
        daily = _daily([9.0, 10.0, 5.0], [20.0, 20.0, 40.0])
        events = _event("2024-01-04", 1, c3=10.0)
        adjusted = apply_event_adjustments(daily, build_event_audit(daily, events))

        np.testing.assert_allclose(adjusted["event_qfq_close"], [4.5, 5.0, 5.0])
        np.testing.assert_allclose(adjusted["event_adjusted_volume"], [40.0, 40.0, 40.0])

    def test_panel_uses_adjusted_prices_and_volume_but_raw_amount(self) -> None:
        daily = _daily([10.0, 5.0], [20.0, 40.0])
        events = _event("2024-01-03", 11, c3=2.0)
        adjusted = apply_event_adjustments(daily, build_event_audit(daily, events))
        values, mask, dates = panel_arrays(adjusted, ["510000"])

        self.assertEqual(values.shape, (1, 6, 2))
        np.testing.assert_allclose(values[0, 3], [5.0, 5.0])
        np.testing.assert_allclose(values[0, 4], [40.0, 40.0])
        np.testing.assert_allclose(values[0, 5], [200.0, 200.0])
        self.assertTrue(mask.all())
        self.assertEqual(len(dates), 2)

    def test_non_trading_day_event_marks_effective_trade_date(self) -> None:
        daily = _daily([100.0, 50.0])
        daily.loc[1, "date"] = pd.Timestamp("2024-01-08")
        events = _event("2024-01-06", 11, c3=2.0)
        audit = build_event_audit(daily, events)
        adjusted = apply_event_adjustments(daily, audit)

        self.assertEqual(audit.iloc[0]["effective_trade_date"], pd.Timestamp("2024-01-08"))
        self.assertEqual(adjusted.iloc[1]["applied_event_count"], 1)

    def test_quality_gate_rejects_negative_volume(self) -> None:
        daily = _daily([10.0, 10.1], [100.0, -1.0])
        events = _event("2023-01-01", 11, c3=2.0)
        audit = build_event_audit(daily, events)
        adjusted = apply_event_adjustments(daily, audit)
        quality = quality_summary(adjusted, audit)

        self.assertEqual(quality_gate_failures(quality, audit)["invalid_adjusted_volume_rows"], 1)

    def test_tushare_flow_replaces_bad_tdx_volume_and_converts_amount_units(self) -> None:
        daily = _daily([1.13, 1.13], [1288.0, 0.0])
        daily.loc[0, "amount"] = 1470.3
        daily.loc[1, "amount"] = 0.0
        tushare = pd.DataFrame(
            {
                "trade_date": [daily.loc[0, "date"]],
                "symbol": ["510000"],
                "vol": [13.0],
                "amount": [1.4703],
            }
        )
        reconciled = reconcile_flow_data(daily, tushare)

        self.assertEqual(reconciled.loc[0, "tdx_volume"], 1288.0)
        self.assertEqual(reconciled.loc[0, "volume"], 13.0)
        self.assertAlmostEqual(reconciled.loc[0, "amount"], 1470.3)
        self.assertEqual(reconciled.loc[0, "flow_source"], "tushare_fund_daily")
        self.assertTrue(reconciled.loc[0, "tradable"])
        self.assertFalse(reconciled.loc[1, "tradable"])

    def test_panel_excludes_nontradable_zero_flow_row(self) -> None:
        daily = reconcile_flow_data(_daily([10.0, 10.0], [100.0, 0.0]))
        daily.loc[1, "amount"] = 0.0
        daily.loc[1, "tradable"] = False
        events = _event("2023-01-01", 11, c3=2.0)
        adjusted = apply_event_adjustments(daily, build_event_audit(daily, events))
        values, mask, _ = panel_arrays(adjusted, ["510000"])

        self.assertTrue(mask[0, 0])
        self.assertFalse(mask[0, 1])
        self.assertTrue(np.isnan(values[0, :, 1]).all())


if __name__ == "__main__":
    unittest.main()