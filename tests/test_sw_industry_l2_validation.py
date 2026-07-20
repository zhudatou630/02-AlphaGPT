from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from alpha_etf.sw_industry.l2_spec import load_dataset_manifest, load_panel
from alpha_etf.sw_industry.l2_validation import (
    EXPECTED_TRADING,
    FROZEN_PROTOCOL_ID,
    build_absroc40_signal,
    load_protocol,
    validate_protocol_dataset,
)
from scripts.sw_industry.run_l2_dynamic_absroc40_validation import (
    CODE_FINGERPRINT_PATHS,
    code_fingerprint,
    reserve_one_shot_run,
    verify_receipt,
)


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "data/processed/sw_industry_l2/sw2021_dynamic/dataset-v2"
PROTOCOL_PATH = ROOT / "configs/sw2021_l2_absroc40_dynamic_long_history.json"
RECEIPT_PATH = ROOT / "configs/sw2021_l2_dynamic_long_history_receipt.json"


class SWIndustryL2ValidationTests(unittest.TestCase):
    def test_real_dataset_and_protocol_are_bound(self) -> None:
        manifest = load_dataset_manifest(DATASET_DIR)
        protocol = load_protocol(PROTOCOL_PATH)
        validate_protocol_dataset(protocol, manifest)
        self.assertEqual(protocol["protocol_id"], FROZEN_PROTOCOL_ID)
        self.assertEqual(protocol["trading"], EXPECTED_TRADING)
        self.assertEqual(protocol["split"]["prior_signal_date"], "2000-05-17")
        self.assertEqual(protocol["split"]["validation_start"], "2000-05-18")

    def test_every_asset_has_exact_global_date_roc40_delay(self) -> None:
        panel = load_panel(DATASET_DIR)
        _, signal = build_absroc40_signal(panel)
        eligible = panel.tradable_mask & np.isfinite(signal)
        for asset in range(len(panel.symbols)):
            first_quote = int(np.flatnonzero(panel.tradable_mask[asset])[0])
            first_signal = int(np.flatnonzero(eligible[asset])[0])
            self.assertEqual(first_signal - first_quote, 40, str(panel.symbols[asset]))

    def test_frozen_receipt_binds_code_protocol_and_dataset(self) -> None:
        manifest = load_dataset_manifest(DATASET_DIR)
        protocol = load_protocol(PROTOCOL_PATH)
        receipt = verify_receipt(
            RECEIPT_PATH,
            PROTOCOL_PATH,
            DATASET_DIR,
            protocol,
            manifest,
        )
        self.assertEqual(receipt["code_fingerprint"], code_fingerprint())
        self.assertEqual(receipt["protocol"]["protocol_id"], FROZEN_PROTOCOL_ID)
        self.assertIn(ROOT / "src/alpha_etf/sw_industry/spec.py", CODE_FINGERPRINT_PATHS)

    def test_receipt_semantic_mutations_are_rejected(self) -> None:
        manifest = load_dataset_manifest(DATASET_DIR)
        protocol = load_protocol(PROTOCOL_PATH)
        original = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
        mutations = (
            ("raw_id", lambda value: value["raw_snapshot"].update(snapshot_id="changed")),
            ("governed_id", lambda value: value["governed_snapshot"].update(governed_id="changed")),
            ("window", lambda value: value["study_window"].update(end_date="2001-01-01")),
            ("claim", lambda value: value.update(claim_policy="changed")),
            ("code", lambda value: value.update(code_fingerprint="0" * 64)),
        )
        for label, mutate in mutations:
            candidate = json.loads(json.dumps(original))
            mutate(candidate)
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "receipt.json"
                path.write_text(json.dumps(candidate), encoding="utf-8")
                with self.assertRaises(RuntimeError):
                    verify_receipt(path, PROTOCOL_PATH, DATASET_DIR, protocol, manifest)

    def test_internal_gap_is_not_filled_and_invalidates_roc_endpoints(self) -> None:
        panel = load_panel(DATASET_DIR)
        _, signal = build_absroc40_signal(panel)
        asset = int(np.flatnonzero(panel.symbols == "801193.SI")[0])
        gap = ~panel.tradable_mask[asset]
        first_quote = int(np.flatnonzero(panel.tradable_mask[asset])[0])
        gap[:first_quote] = False
        self.assertEqual(int(gap.sum()), 62)
        shifted_gap = np.zeros_like(gap)
        shifted_gap[40:] = gap[:-40]
        self.assertTrue(np.isnan(signal[asset, gap | shifted_gap]).all())

    def test_one_shot_claim_is_protocol_global(self) -> None:
        protocol = load_protocol(PROTOCOL_PATH)
        manifest = load_dataset_manifest(DATASET_DIR)
        with tempfile.TemporaryDirectory() as tmp:
            claim_root = Path(tmp) / "claims"
            reserve_one_shot_run(
                protocol,
                manifest,
                Path(tmp) / "first",
                claim_root=claim_root,
                receipt_sha256="a" * 64,
                protocol_sha256="b" * 64,
                dataset_manifest_sha256="c" * 64,
            )
            with self.assertRaisesRegex(RuntimeError, "already been claimed"):
                reserve_one_shot_run(
                    protocol,
                    manifest,
                    Path(tmp) / "second",
                    claim_root=claim_root,
                    receipt_sha256="a" * 64,
                    protocol_sha256="b" * 64,
                    dataset_manifest_sha256="c" * 64,
                )

    def test_protocol_self_hash_rejects_mutation(self) -> None:
        protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        protocol["universe"]["warmup"] = "changed"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "protocol.json"
            path.write_text(json.dumps(protocol), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "protocol ID mismatch"):
                load_protocol(path)


if __name__ == "__main__":
    unittest.main()