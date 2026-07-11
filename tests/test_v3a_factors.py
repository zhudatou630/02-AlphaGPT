from __future__ import annotations

import unittest

import numpy as np
import torch

from alpha_etf.research_v3a.factors import (
    FACTOR_NAMES,
    MA_RATIO_WINDOWS,
    WINDOWS,
    build_factor_values_numpy,
    build_factor_values_torch,
)


def _panel(assets: int = 3, dates: int = 90) -> tuple[np.ndarray, np.ndarray]:
    time = np.arange(dates, dtype=float)
    values = np.empty((assets, 4, dates), dtype=float)
    for asset in range(assets):
        close = (100.0 + time * (1.0 + asset * 0.1)) * (1.0 + asset * 0.25)
        open_ = close - 0.5
        high = close + 1.0
        low = close - 1.0
        values[asset, 0] = open_
        values[asset, 1] = high
        values[asset, 2] = low
        values[asset, 3] = close
    return values, np.ones((assets, dates), dtype=bool)


class V3AFactorTests(unittest.TestCase):
    def test_factor_inventory_and_hand_calculated_values(self) -> None:
        absolute, mask = _panel(assets=1)
        factors = build_factor_values_numpy(absolute, mask)
        self.assertEqual(len(FACTOR_NAMES), 40)
        self.assertEqual(len(MA_RATIO_WINDOWS), 10)
        self.assertEqual(factors.shape, (40, 1, 90))

        lookup = {name: factors[index, 0] for index, name in enumerate(FACTOR_NAMES)}
        self.assertAlmostEqual(lookup["DAYRET"][1], 101.0 / 100.0 - 1.0)
        self.assertAlmostEqual(lookup["GAP"][1], 100.5 / 100.0 - 1.0)
        self.assertAlmostEqual(lookup["INTRADAY"][1], 101.0 / 100.5 - 1.0)
        self.assertAlmostEqual(lookup["RANGE"][1], 2.0 / 100.0)
        self.assertAlmostEqual(lookup["CLV"][1], 0.0)
        self.assertAlmostEqual(lookup["ROC_5"][5], 105.0 / 100.0 - 1.0)
        self.assertAlmostEqual(lookup["PRICE_MA_5"][4], 104.0 / 102.0 - 1.0)
        self.assertAlmostEqual(lookup["TS_RANK_5"][4], 1.0)
        self.assertAlmostEqual(lookup["RSV_5"][4], 5.0 / 6.0)
        self.assertAlmostEqual(
            lookup["MA_RATIO_5_10"][9], np.mean(np.arange(105.0, 110.0)) / 104.5 - 1.0
        )
        self.assertTrue(np.isnan(lookup["PRICE_MA_5"][3]))
        self.assertTrue(np.isnan(lookup["VOL_5"][4]))
        self.assertTrue(np.isfinite(lookup["VOL_5"][5]))

    def test_zero_range_and_tied_rank_are_neutral(self) -> None:
        absolute = np.full((1, 4, 10), 10.0)
        mask = np.ones((1, 10), dtype=bool)
        factors = build_factor_values_numpy(absolute, mask)
        lookup = {name: factors[index, 0] for index, name in enumerate(FACTOR_NAMES)}
        self.assertEqual(lookup["CLV"][5], 0.0)
        self.assertEqual(lookup["RSV_5"][5], 0.0)
        self.assertEqual(lookup["TS_RANK_5"][5], 0.5)

    def test_missing_day_invalidates_full_windows_without_compression(self) -> None:
        absolute, mask = _panel(assets=1)
        mask[0, 20] = False
        absolute[0, :, 20] = np.nan
        factors = build_factor_values_numpy(absolute, mask)
        lookup = {name: factors[index, 0] for index, name in enumerate(FACTOR_NAMES)}
        self.assertTrue(np.isnan(lookup["PRICE_MA_5"][24]))
        self.assertTrue(np.isfinite(lookup["PRICE_MA_5"][25]))
        self.assertTrue(np.isnan(lookup["ROC_5"][25]))
        self.assertTrue(np.isfinite(lookup["ROC_5"][26]))

    def test_asset_specific_scaling_leaves_all_factors_unchanged(self) -> None:
        absolute, mask = _panel()
        original = build_factor_values_numpy(absolute, mask)
        scales = np.array([0.17, 3.5, 19.0])[:, None, None]
        scaled = build_factor_values_numpy(absolute * scales, mask)
        np.testing.assert_allclose(original, scaled, rtol=1e-12, atol=1e-12, equal_nan=True)

    def test_future_changes_do_not_affect_past_factors(self) -> None:
        absolute, mask = _panel()
        original = build_factor_values_numpy(absolute, mask)
        changed = absolute.copy()
        changed[:, :, 61:] *= np.linspace(1.1, 1.8, changed.shape[-1] - 61)
        modified = build_factor_values_numpy(changed, mask)
        np.testing.assert_allclose(
            original[..., :61], modified[..., :61], rtol=0.0, atol=0.0, equal_nan=True
        )

    def test_torch_matches_numpy(self) -> None:
        absolute, mask = _panel()
        mask[1, 37] = False
        absolute[1, :, 37] = np.nan
        expected = build_factor_values_numpy(absolute, mask)
        actual = build_factor_values_torch(
            torch.as_tensor(absolute, dtype=torch.float64), torch.as_tensor(mask)
        ).numpy()
        np.testing.assert_allclose(expected, actual, rtol=1e-12, atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(
            expected.astype(np.float32),
            actual.astype(np.float32),
            rtol=1e-7,
            atol=1e-8,
            equal_nan=True,
        )
        with self.assertRaises(ValueError):
            build_factor_values_torch(
                torch.as_tensor(absolute, dtype=torch.float32), torch.as_tensor(mask)
            )
        self.assertEqual(tuple(WINDOWS), (5, 10, 20, 40, 60))


if __name__ == "__main__":
    unittest.main()