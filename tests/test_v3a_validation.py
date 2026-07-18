from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from alpha_etf.research_v3a.validation import (
    ValidationConfig,
    decide_validation,
    run_formula_validation,
)
from alpha_etf.research_v3a.factors import FACTOR_NAMES
from alpha_etf.research_v3a.spec import sha256_file
from alpha_etf.research_v3a.validation_view import (
    VALIDATION_VIEW_SCHEMA_VERSION,
    build_validation_view_manifest,
    load_validation_view,
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

    def test_validation_view_is_physically_final_free_and_tamper_checked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            arrays = {
                "factor_values": np.ones((len(FACTOR_NAMES), 2, 3), dtype=np.float64),
                "absolute_open": np.ones((2, 3), dtype=np.float64),
                "absolute_close": np.ones((2, 3), dtype=np.float64),
                "tradable_mask": np.ones((2, 3), dtype=bool),
                "symbols": np.asarray(["A", "B"]),
                "dates": np.asarray(["2021-12-31", "2022-01-04", "2022-12-30"]),
            }
            files = {}
            for name, value in arrays.items():
                path = root / f"{name}.npy"
                np.save(path, value, allow_pickle=False)
                files[name] = {
                    "path": path.name,
                    "sha256": sha256_file(path),
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                }
            manifest = build_validation_view_manifest(
                {
                    "schema_version": VALIDATION_VIEW_SCHEMA_VERSION,
                    "protocol_id": "protocol",
                    "approval_id": "approval",
                    "preflight_binding_id": "binding",
                    "source_dataset_id": "dataset",
                    "source_panel_sha256": "panel",
                    "source_dataset_manifest": {"symbols": ["A", "B"]},
                    "code_commit": "commit",
                    "code_fingerprint": "fingerprint",
                    "split": {
                        "signal_start": "2016-08-09",
                        "validation_start": "2022-01-01",
                        "validation_end": "2022-12-31",
                        "data_end": "2022-12-31",
                        "validation_columns_present": True,
                        "final_columns_present": False,
                    },
                    "factor_names": list(FACTOR_NAMES),
                    "factor_shape": list(arrays["factor_values"].shape),
                    "mask_shape": list(arrays["tradable_mask"].shape),
                    "date_start": "2021-12-31",
                    "date_end": "2022-12-30",
                    "files": files,
                }
            )
            manifest_path = root / "validation_view_manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            view = load_validation_view(root)
            self.assertEqual(view.dates[-1], pd.Timestamp("2022-12-30"))
            dates_path = root / "dates.npy"
            np.save(
                dates_path,
                np.asarray(["2021-12-31", "2022-01-04", "2023-01-03"]),
                allow_pickle=False,
            )
            with self.assertRaisesRegex(RuntimeError, "SHA mismatch"):
                load_validation_view(root)


if __name__ == "__main__":
    unittest.main()