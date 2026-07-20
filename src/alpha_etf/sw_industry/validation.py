"""Frozen ABS(ROC40) cross-universe validation helpers for SW2021."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from alpha_etf.research_v3a.factors import FACTOR_NAMES, build_factor_values_numpy
from alpha_etf.sw_industry.spec import SWIndustryPanel, canonical_sha256


PROTOCOL_SCHEMA_VERSION = "sw2021-l1-absroc40-cross-universe-validation-v1"
FROZEN_PROTOCOL_ID = "be4e2eb7166bccd384037fb140bcad8ef4453ead824b314866482d5548717b2c"
EXPECTED_FORMULA = {
    "text": "ABS(ROC(40))",
    "factor": "ROC_40",
    "absolute_value": True,
}
EXPECTED_TRADING = {
    "initial_cash": 1.0,
    "min_universe": 10,
    "slots": 3,
    "buy_rank": 3,
    "hold_rank": 5,
    "robust_z_threshold": 1.5,
    "stop_loss": -0.07,
    "transaction_cost_bps": 0.0,
    "cash_return": 0.0,
    "max_holding_days": None,
    "allow_fractional_shares": True,
    "signal_dtype": "float32",
    "execution": "t_close_decision_t_plus_1_open",
    "annual_reset": False,
}


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    payload = dict(protocol)
    expected_id = str(payload.pop("protocol_id", ""))
    if canonical_sha256(payload) != expected_id:
        raise RuntimeError("SW2021 validation protocol ID mismatch")
    if expected_id != FROZEN_PROTOCOL_ID:
        raise RuntimeError("SW2021 validation protocol is not the uniquely frozen protocol")
    if protocol.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        raise RuntimeError("SW2021 validation protocol schema mismatch")
    if protocol.get("formula") != EXPECTED_FORMULA:
        raise RuntimeError("SW2021 validation formula drifted")
    if protocol.get("trading") != EXPECTED_TRADING:
        raise RuntimeError("SW2021 validation trading rules drifted from ETF validation")
    interpretation = protocol.get("interpretation", {})
    expected_interpretation = {
        "test_type": "cross_universe_replication",
        "industry_data_used_for_formula_selection": False,
        "parameter_tuning_allowed": False,
        "result_dependent_protocol_changes_allowed": False,
        "executable_etf_strategy_claim": False,
        "planned_run_count": 1,
    }
    if interpretation != expected_interpretation:
        raise RuntimeError("SW2021 validation interpretation drifted")
    return protocol


def validate_protocol_dataset(protocol: dict[str, Any], manifest: dict[str, Any]) -> None:
    expected = protocol["dataset"]
    observed = {
        "dataset_id": manifest["dataset_id"],
        "panel_sha256": manifest["panel_sha256"],
        "date_start": manifest["date_start"],
        "date_end": manifest["date_end"],
        "symbols": manifest["symbols"],
        "series_semantics": manifest["series_semantics"],
        "historical_vintage_proven": manifest["historical_vintage_proven"],
    }
    if observed != expected:
        raise RuntimeError("SW2021 validation dataset binding mismatch")


def build_absroc40_signal(panel: SWIndustryPanel) -> tuple[np.ndarray, np.ndarray]:
    factors = build_factor_values_numpy(panel.absolute_ohlc, panel.tradable_mask)
    roc40 = factors[FACTOR_NAMES.index("ROC_40")]
    signal = np.abs(roc40).astype(np.float32)
    return roc40, signal
