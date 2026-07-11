from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from alpha_etf.data.relative_price import add_relative_ohlc, relative_panel
from alpha_etf.research_v3a.spec import (
    ABSOLUTE_FEATURES,
    DATASET_SCHEMA_VERSION,
    EFFECTIVE_START,
    EXPECTED_UPSTREAM_V3_ID,
    MIN_UNIVERSE,
    PRICE_ADJUSTMENT,
    RELATIVE_TRANSFORM,
    RELATIVE_FEATURES,
    TRADABLE_MASK_SEMANTICS,
    build_dataset_manifest,
    build_panel_arrays,
    build_research_spec,
    canonical_sha256,
    load_panel,
    sha256_file,
    validate_research_spec,
)
from alpha_etf.research_v3a.candidates import CandidateConfig, canonicalizer_config
from alpha_etf.research_v3a.factors import factor_config
from alpha_etf.research_v3a.language import FORMULA_VOCAB
from alpha_etf.research_v3a.scoring import ScorerConfig
from alpha_etf.research_v3a.sampling import SamplingConfig


def _governance_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["A", "A", "A", "B", "B", "B"],
            "date": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-03",
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-03",
                ]
            ),
            "tradable": [True, False, True, True, True, True],
            "event_qfq_open": [10.0, 10.1, 10.4, 20.0, 20.1, 20.2],
            "event_qfq_high": [10.2, 10.3, 10.6, 20.2, 20.3, 20.4],
            "event_qfq_low": [9.9, 10.0, 10.3, 19.9, 20.0, 20.1],
            "event_qfq_close": [10.0, 10.2, 10.5, 20.0, 20.2, 20.3],
        }
    )


def _write_dataset(root: Path) -> dict[str, object]:
    absolute = np.array(
        [
            [[np.nan, 10.1], [np.nan, 10.3], [np.nan, 10.0], [np.nan, 10.2]],
        ],
        dtype=float,
    )
    relative = np.array(
        [
            [[np.nan, 0.01], [np.nan, 0.03], [np.nan, 0.0], [np.nan, 0.02]],
        ],
        dtype=float,
    )
    mask = np.array([[False, True]])
    panel_path = root / "panel_v3a.npz"
    np.savez_compressed(
        panel_path,
        absolute_ohlc=absolute,
        relative_ohlc=relative,
        tradable_mask=mask,
        symbols=np.array(["A"]),
        dates=np.array(["2026-01-01", "2026-01-02"]),
        absolute_features=np.array(ABSOLUTE_FEATURES),
        relative_features=np.array(RELATIVE_FEATURES),
    )
    identity = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "panel_file": panel_path.name,
        "panel_sha256": sha256_file(panel_path),
        "upstream_v3_manifest_file": "upstream.json",
        "upstream_v3_manifest_sha256": "1" * 64,
        "upstream_v3_dataset_id": EXPECTED_UPSTREAM_V3_ID,
        "upstream_v3_panel_sha256": "2" * 64,
        "governance_file": "governance.parquet",
        "governance_sha256": "3" * 64,
        "universe_file": "universe.json",
        "universe_sha256": "4" * 64,
        "symbols": ["A"],
        "absolute_features": list(ABSOLUTE_FEATURES),
        "relative_features": list(RELATIVE_FEATURES),
        "absolute_shape": list(absolute.shape),
        "relative_shape": list(relative.shape),
        "mask_shape": list(mask.shape),
        "date_start": "2026-01-01",
        "date_end": "2026-01-02",
        "date_count": 2,
        "tradable_observations": 1,
        "effective_start": EFFECTIVE_START,
        "min_universe": MIN_UNIVERSE,
        "price_adjustment": PRICE_ADJUSTMENT,
        "relative_transform": RELATIVE_TRANSFORM,
        "tradable_mask": TRADABLE_MASK_SEMANTICS,
        "flow_features_exposed": False,
    }
    manifest = build_dataset_manifest(identity)
    (root / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


class V3ADatasetTests(unittest.TestCase):
    def test_build_arrays_preserves_source_row_relative_semantics(self) -> None:
        frame = _governance_frame()
        relative_frame = add_relative_ohlc(frame)
        expected_relative, expected_mask, dates = relative_panel(relative_frame, ["A", "B"])
        absolute, relative, mask, actual_dates = build_panel_arrays(
            frame,
            symbols=["A", "B"],
            upstream_relative=expected_relative,
            upstream_mask=expected_mask,
            upstream_dates=dates.strftime("%Y-%m-%d").to_numpy(dtype=str),
        )
        self.assertTrue(actual_dates.equals(dates))
        self.assertTrue(np.isnan(absolute[0, :, 1]).all())
        self.assertTrue(mask[0, 2])
        self.assertAlmostEqual(relative[0, 3, 2], 10.5 / 10.2 - 1.0)

    def test_missing_global_date_uses_previous_source_row_for_relative(self) -> None:
        frame = _governance_frame().query("not (symbol == 'A' and date == '2026-01-02')")
        relative_frame = add_relative_ohlc(frame)
        expected_relative, expected_mask, dates = relative_panel(relative_frame, ["A", "B"])
        absolute, relative, _, _ = build_panel_arrays(
            frame,
            symbols=["A", "B"],
            upstream_relative=expected_relative,
            upstream_mask=expected_mask,
            upstream_dates=dates.strftime("%Y-%m-%d").to_numpy(dtype=str),
        )
        self.assertTrue(np.isnan(absolute[0, :, 1]).all())
        self.assertAlmostEqual(relative[0, 3, 2], 10.5 / 10.0 - 1.0)

    def test_loader_enforces_hash_schema_and_mask_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = _write_dataset(root)
            panel = load_panel(root)
            self.assertEqual(panel.absolute_ohlc.shape, (1, 4, 2))
            self.assertEqual(panel.tradable_mask.tolist(), [[False, True]])

            manifest["panel_sha256"] = "bad"
            (root / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                load_panel(root)

    def test_loader_rejects_missing_identity_fields_and_wrong_mask_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = _write_dataset(root)
            missing = dict(manifest)
            del missing["governance_sha256"]
            identity = {key: value for key, value in missing.items() if key not in {"dataset_id", "dataset_fingerprint"}}
            rebuilt = build_dataset_manifest(identity)
            (root / "dataset_manifest.json").write_text(json.dumps(rebuilt), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                load_panel(root)

            manifest = _write_dataset(root)
            identity = {
                key: value
                for key, value in manifest.items()
                if key not in {"dataset_id", "dataset_fingerprint"}
            }
            identity["mask_shape"] = [1, 3]
            wrong_shape = build_dataset_manifest(identity)
            (root / "dataset_manifest.json").write_text(
                json.dumps(wrong_shape), encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                load_panel(root)

    def test_research_spec_identity_changes_with_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = _write_dataset(Path(tmp))
        manifest = {
            **manifest,
            "symbols": [f"ETF{i:02d}" for i in range(35)],
        }
        common = {
            "dataset_manifest": manifest,
            "factor_config": factor_config(),
            "vocab_config": {
                **FORMULA_VOCAB.to_config(),
                "sampling": SamplingConfig().to_dict(),
            },
            "scorer_config": ScorerConfig().to_dict(),
            "canonicalizer_config": canonicalizer_config(),
            "candidate_config": CandidateConfig().to_dict(),
            "code_commit": "abc",
            "code_fingerprint": "def",
        }
        spec = build_research_spec(**common)
        validate_research_spec(spec)
        changed = build_research_spec(
            **{**common, "scorer_config": {**ScorerConfig().to_dict(), "version": "changed"}}
        )
        self.assertNotEqual(spec["research_spec_id"], changed["research_spec_id"])
        spec["code_commit"] = "tampered"
        with self.assertRaises(RuntimeError):
            validate_research_spec(spec)

    def test_research_spec_rejects_rehashed_legacy_component_versions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = _write_dataset(Path(tmp))
        manifest = {**manifest, "symbols": [f"ETF{i:02d}" for i in range(35)]}
        spec = build_research_spec(
            dataset_manifest=manifest,
            factor_config=factor_config(),
            vocab_config={
                **FORMULA_VOCAB.to_config(),
                "sampling": SamplingConfig().to_dict(),
            },
            scorer_config=ScorerConfig().to_dict(),
            canonicalizer_config=canonicalizer_config(),
            candidate_config=CandidateConfig().to_dict(),
            code_commit="abc",
            code_fingerprint="def",
        )
        mutations = (
            ("factor_config", "version", "price-event-v2"),
            ("vocab_config", "version", "price-event-v2"),
            ("vocab_config", "grammar_version", "legacy-grammar"),
            ("scorer_config", "version", "legacy-scorer"),
            ("canonicalizer_config", "version", "legacy-canonicalizer"),
            ("candidate_config", "version", "legacy-funnel"),
            ("factor_config", "missing", "forward_fill"),
            ("canonicalizer_config", "nan_preserving", False),
        )
        for section, field, value in mutations:
            changed = json.loads(json.dumps(spec))
            changed[section][field] = value
            payload = dict(changed)
            payload.pop("research_spec_id")
            changed["research_spec_id"] = canonical_sha256(payload)
            with self.subTest(section=section, field=field):
                with self.assertRaises(RuntimeError):
                    validate_research_spec(changed)


if __name__ == "__main__":
    unittest.main()