import unittest

import pandas as pd

from scripts.v3a.audit_expanded_universe_price_jumps import (
    _classify,
    _event_reference,
)


class ExpandedUniversePriceJumpTest(unittest.TestCase):
    def test_share_consolidation_restates_previous_close(self) -> None:
        events = pd.DataFrame(
            [
                {
                    "symbol": "510020",
                    "date": pd.Timestamp("2012-12-14"),
                    "category_code": 11,
                    "c1": 0.0,
                    "c2": 0.0,
                    "c3": 0.1,
                    "c4": 0.0,
                }
            ]
        )
        self.assertAlmostEqual(_event_reference(0.174, events), 1.74)

    def test_same_day_cash_is_applied_after_share_change(self) -> None:
        events = pd.DataFrame(
            [
                {
                    "symbol": "159922",
                    "date": pd.Timestamp("2024-12-02"),
                    "category_code": 11,
                    "c1": 0.0,
                    "c2": 0.0,
                    "c3": 2.5,
                    "c4": 0.0,
                },
                {
                    "symbol": "159922",
                    "date": pd.Timestamp("2024-12-02"),
                    "category_code": 1,
                    "c1": 1.292,
                    "c2": 0.0,
                    "c3": 0.0,
                    "c4": 0.0,
                },
            ]
        )
        self.assertAlmostEqual(_event_reference(6.159, events), 2.3344)

    def test_classification_keeps_rounding_separate_from_events(self) -> None:
        market = pd.Series(
            {
                "event_categories": "",
                "adjusted_open_gap": 0.20094,
                "adjusted_close_jump": 0.198,
                "limit_rounding_bound": 0.20138,
            }
        )
        unexplained = market.copy()
        unexplained["adjusted_open_gap"] = 0.21
        share = unexplained.copy()
        share["event_categories"] = "11"
        explained_share = share.copy()
        explained_share["adjusted_open_gap"] = 0.10
        self.assertEqual(
            _classify(market), "market_move_within_20pct_limit_rounding"
        )
        self.assertEqual(_classify(unexplained), "unexplained_large_gap")
        self.assertEqual(_classify(share), "unexplained_large_gap")
        self.assertEqual(_classify(explained_share), "explained_share_adjustment")


if __name__ == "__main__":
    unittest.main()