from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from alpha_etf.sw_industry.spec import (
    ABSOLUTE_FEATURES,
    SWIndustryPanel,
    build_dataset_manifest,
    build_panel_arrays,
    canonical_sha256,
)
from alpha_etf.sw_industry.validation import (
    EXPECTED_FORMULA,
    EXPECTED_TRADING,
    FROZEN_PROTOCOL_ID,
    PROTOCOL_SCHEMA_VERSION,
    build_absroc40_signal,
    load_protocol,
)
from scripts.sw_industry.run_absroc40_validation import reserve_one_shot_run, trade_metrics


class SWIndustryTests(unittest.TestCase):
    def test_build_panel_requires_complete_31_industry_rectangle(self) -> None:
        rows = []
        for asset in range(31):
            for date, value in (("2026-01-05", 100.0 + asset), ("2026-01-06", 101.0 + asset)):
                rows.append(
                    {
                        "ts_code": f"I{asset:02d}",
                        "industry_name": f"Industry {asset:02d}",
                        "trade_date": date,
                        "open": value,
                        "high": value + 1.0,
                        "low": value - 1.0,
                        "close": value + 0.5,
                    }
                )
        frame = pd.DataFrame(rows)
        values, mask, symbols, names, dates = build_panel_arrays(frame)
        self.assertEqual(values.shape, (31, 4, 2))
        self.assertTrue(mask.all())
        self.assertEqual(symbols[0], "I00")
        self.assertEqual(names[-1], "Industry 30")
        self.assertEqual(dates[-1], pd.Timestamp("2026-01-06"))
        with self.assertRaisesRegex(ValueError, "not rectangular"):
            build_panel_arrays(frame.iloc[:-1])

    def test_absroc40_signal_has_exact_40_day_delay(self) -> None:
        dates = pd.date_range("2026-01-01", periods=45, freq="D")
        close = np.tile(np.arange(100.0, 145.0), (31, 1))
        absolute = np.stack([close, close, close, close], axis=1)
        panel = SWIndustryPanel(
            absolute_ohlc=absolute,
            tradable_mask=np.ones((31, 45), dtype=bool),
            symbols=np.asarray([f"I{i:02d}" for i in range(31)]),
            names=np.asarray([f"Industry {i:02d}" for i in range(31)]),
            dates=dates,
        )
        roc40, signal = build_absroc40_signal(panel)
        self.assertTrue(np.isnan(signal[:, :40]).all())
        expected = 140.0 / 100.0 - 1.0
        self.assertTrue(np.allclose(roc40[:, 40], expected))
        self.assertEqual(signal.dtype, np.float32)

    def test_protocol_hash_and_frozen_trading_rules(self) -> None:
        payload = {
            "schema_version": PROTOCOL_SCHEMA_VERSION,
            "name": "unit",
            "interpretation": {
                "test_type": "cross_universe_replication",
                "industry_data_used_for_formula_selection": False,
                "parameter_tuning_allowed": False,
                "result_dependent_protocol_changes_allowed": False,
                "executable_etf_strategy_claim": False,
                "planned_run_count": 1,
            },
            "dataset": {},
            "formula": EXPECTED_FORMULA,
            "split": {},
            "trading": dict(EXPECTED_TRADING),
        }
        protocol = {**payload, "protocol_id": canonical_sha256(payload)}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "protocol.json"
            path.write_text(json.dumps(protocol), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "uniquely frozen"):
                load_protocol(path)
            protocol["trading"]["buy_rank"] = 4
            path.write_text(json.dumps(protocol), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                load_protocol(path)

    def test_real_frozen_protocol_and_dates(self) -> None:
        root = Path(__file__).resolve().parents[1]
        protocol = load_protocol(
            root / "configs/sw2021_absroc40_cross_universe_validation.json"
        )
        self.assertEqual(protocol["protocol_id"], FROZEN_PROTOCOL_ID)
        self.assertEqual(protocol["split"]["prior_signal_date"], "2014-04-21")
        self.assertEqual(protocol["split"]["validation_start"], "2014-04-22")
        self.assertEqual(protocol["trading"], EXPECTED_TRADING)

    def test_one_shot_claim_rejects_second_output_and_zero_trade_metrics(self) -> None:
        protocol = {"protocol_id": FROZEN_PROTOCOL_ID}
        manifest = {"dataset_id": "dataset", "panel_sha256": "panel"}
        with tempfile.TemporaryDirectory() as tmp:
            claim_root = Path(tmp) / "claims"
            reserve_one_shot_run(
                protocol, manifest, Path(tmp) / "first-output", claim_root=claim_root
            )
            with self.assertRaisesRegex(RuntimeError, "already been claimed"):
                reserve_one_shot_run(
                    protocol, manifest, Path(tmp) / "different-output", claim_root=claim_root
                )
        self.assertEqual(trade_metrics(pd.DataFrame())["trade_count"], 0)

    def test_dataset_id_changes_with_identity(self) -> None:
        identity = {
            "schema_version": "sw2021-l1-industry-panel-v1",
            "panel_sha256": "a" * 64,
        }
        first = build_dataset_manifest(identity)
        second = build_dataset_manifest({**identity, "panel_sha256": "b" * 64})
        self.assertNotEqual(first["dataset_id"], second["dataset_id"])


if __name__ == "__main__":
    unittest.main()