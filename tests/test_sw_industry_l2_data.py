from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load script: {relative_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DOWNLOAD = load_script(
    "download_sw2021_l2_backcast",
    "scripts/sw_industry/download_sw2021_l2_backcast.py",
)
GOVERN = load_script(
    "build_sw2021_l2_governed",
    "scripts/sw_industry/build_sw2021_l2_governed.py",
)
IDENTITY_PATH = ROOT / "configs/sw_industry_l2_sw2021_identity.json"


class FakePro:
    def __init__(self, classification: pd.DataFrame):
        self.classification = classification

    def index_classify(self, **_: object) -> pd.DataFrame:
        return self.classification.copy()


class SWIndustryL2DataTests(unittest.TestCase):
    def test_frozen_identity_has_exact_published_contract(self) -> None:
        rows, published = DOWNLOAD.load_identity(IDENTITY_PATH)
        self.assertEqual(len(rows), 134)
        self.assertEqual(len(published), 124)
        self.assertEqual(len({str(row["parent_code"]) for row in rows if row["is_pub"] == 1}), 31)

    def test_classification_parent_drift_is_rejected(self) -> None:
        rows, published = DOWNLOAD.load_identity(IDENTITY_PATH)
        classification = pd.DataFrame(rows)
        classification["level"] = "L2"
        classification["src"] = "SW2021"
        classification.loc[classification.index[0], "parent_code"] = "changed"
        with self.assertRaisesRegex(ValueError, "contract differs|differs from frozen identity"):
            DOWNLOAD.fetch_classification(FakePro(classification), rows, published)

    def test_missing_return_fields_are_recomputed(self) -> None:
        rows = []
        for code, closes in (("A", (100.0, 101.0)), ("B", (200.0, 198.0))):
            for date, close in zip(("20210104", "20210105"), closes, strict=True):
                rows.append(
                    {
                        "ts_code": code,
                        "trade_date": date,
                        "source_name": code,
                        "industry_name": code,
                        "open": close,
                        "high": close,
                        "low": close,
                        "close": close,
                        "change": None,
                        "pct_change": None,
                    }
                )
        governed, audit, quality = GOVERN.build_governed(pd.DataFrame(rows), "synthetic")
        repaired = governed[governed["trade_date"] == "20210105"]
        self.assertTrue(repaired["return_fields_repaired"].all())
        self.assertTrue(repaired[["change", "pct_change"]].notna().all().all())
        self.assertEqual(quality["return_field_rows_repaired"], 2)
        self.assertEqual(quality["SW2021_boundary_code_count"], 0)
        self.assertEqual(len(audit), 4)

    def test_source_table_rejects_smaller_rectangular_panel(self) -> None:
        identity = json.loads(IDENTITY_PATH.read_text(encoding="utf-8"))
        published_codes = {
            str(row["index_code"]) for row in identity["published_industries"]
        }
        source = pd.DataFrame(
            [{"ts_code": next(iter(published_codes)), "trade_date": "20210104"}]
        )
        calendar = pd.DataFrame([{"cal_date": "20210104", "is_open": 1}])
        manifest = {
            "requested_start_date": "20210104",
            "data_as_of_trade_date": "20210104",
            "quality": {"rows": 1, "industries": 1, "trading_days": 1},
        }
        with self.assertRaisesRegex(RuntimeError, "published_code_set"):
            GOVERN.verify_source_table(source, calendar, manifest, published_codes)


if __name__ == "__main__":
    unittest.main()