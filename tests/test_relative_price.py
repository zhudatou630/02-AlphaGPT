import unittest

import numpy as np
import pandas as pd

from alpha_etf.data.relative_price import add_relative_ohlc, relative_panel, relative_quality


class RelativePriceTest(unittest.TestCase):
    def test_relative_ohlc_removes_common_scale(self) -> None:
        frame = pd.DataFrame(
            {
                "symbol": ["A", "A"],
                "date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
                "tradable": [True, True],
                "event_qfq_open": [10.0, 10.5],
                "event_qfq_high": [11.0, 11.0],
                "event_qfq_low": [9.0, 10.0],
                "event_qfq_close": [10.0, 10.8],
            }
        )
        relative = add_relative_ohlc(frame)
        self.assertAlmostEqual(relative.iloc[1]["open_rel"], 0.05)
        self.assertAlmostEqual(relative.iloc[1]["close_rel"], 0.08)
        scaled = frame.copy()
        for column in ("open", "high", "low", "close"):
            scaled[f"event_qfq_{column}"] *= 7.0
        scaled_relative = add_relative_ohlc(scaled)
        np.testing.assert_allclose(
            relative.iloc[1][["open_rel", "high_rel", "low_rel", "close_rel"]].astype(float),
            scaled_relative.iloc[1][["open_rel", "high_rel", "low_rel", "close_rel"]].astype(float),
        )

    def test_panel_masks_first_and_nontradable_rows(self) -> None:
        frame = pd.DataFrame(
            {
                "symbol": ["A", "A", "A"],
                "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
                "tradable": [True, False, True],
                "event_qfq_open": [10.0, 10.1, 10.2],
                "event_qfq_high": [10.2, 10.3, 10.4],
                "event_qfq_low": [9.9, 10.0, 10.1],
                "event_qfq_close": [10.0, 10.2, 10.3],
            }
        )
        values, mask, _ = relative_panel(add_relative_ohlc(frame), ["A"])
        self.assertEqual(values.shape, (1, 4, 3))
        self.assertEqual(mask.tolist(), [[False, False, True]])

    def test_quality_detects_bad_bar(self) -> None:
        relative = pd.DataFrame(
            {"open_rel": [0.0], "high_rel": [-0.1], "low_rel": [-0.2], "close_rel": [0.1]}
        )
        self.assertEqual(relative_quality(relative)["invalid_relative_high_rows"], 1)


if __name__ == "__main__":
    unittest.main()