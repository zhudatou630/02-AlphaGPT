"""Frozen ABS(ROC40) helpers for the SW2021 L2 dynamic-history test."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from alpha_etf.research_v3a.factors import FACTOR_NAMES, build_factor_values_numpy
from alpha_etf.sw_industry.l2_spec import SWIndustryL2Panel
from alpha_etf.sw_industry.spec import canonical_sha256


PROTOCOL_SCHEMA_VERSION = "sw2021-l2-absroc40-dynamic-long-history-v1"
FROZEN_PROTOCOL_ID = "81a77d785f756628d0504bce970a8c9d3067567d117f09f5cc2e2cda4d04034f"
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
EXPECTED_INTERPRETATION = {
    "test_type": "sequential_exploratory_cross_granularity_long_history",
    "hypothesis_informed_by_level1_result": True,
    "current_identity_survivorship_and_backcast_acknowledged": True,
    "industry_data_used_for_formula_selection": False,
    "parameter_tuning_allowed": False,
    "result_dependent_protocol_changes_allowed": False,
    "executable_etf_strategy_claim": False,
    "planned_run_count": 1,
}


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    payload = dict(protocol)
    expected_id = str(payload.pop("protocol_id", ""))
    if canonical_sha256(payload) != expected_id:
        raise RuntimeError("SW2021 L2 protocol ID mismatch")
    if expected_id != FROZEN_PROTOCOL_ID:
        raise RuntimeError("SW2021 L2 protocol is not the uniquely frozen protocol")
    if protocol.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        raise RuntimeError("SW2021 L2 protocol schema mismatch")
    if protocol.get("formula") != EXPECTED_FORMULA:
        raise RuntimeError("SW2021 L2 formula drifted")
    if protocol.get("trading") != EXPECTED_TRADING:
        raise RuntimeError("SW2021 L2 trading rules drifted from ETF validation")
    if protocol.get("interpretation") != EXPECTED_INTERPRETATION:
        raise RuntimeError("SW2021 L2 interpretation drifted")
    return protocol


def validate_protocol_dataset(protocol: dict[str, Any], manifest: dict[str, Any]) -> None:
    expected = protocol["dataset"]
    observed = {
        "dataset_id": manifest["dataset_id"],
        "panel_sha256": manifest["panel_sha256"],
        "availability_sha256": manifest["availability_sha256"],
        "date_start": manifest["date_start"],
        "date_end": manifest["date_end"],
        "symbol_count": len(manifest["symbols"]),
        "identity_policy": manifest["identity_policy"],
        "entry_policy": manifest["entry_policy"],
        "series_semantics": manifest["series_semantics"],
        "historical_vintage_proven": manifest["historical_vintage_proven"],
    }
    if observed != expected:
        raise RuntimeError("SW2021 L2 protocol dataset binding mismatch")


def build_absroc40_signal(panel: SWIndustryL2Panel) -> tuple[np.ndarray, np.ndarray]:
    factors = build_factor_values_numpy(panel.absolute_ohlc, panel.tradable_mask)
    roc40 = factors[FACTOR_NAMES.index("ROC_40")]
    signal = np.abs(roc40).astype(np.float32)
    return roc40, signal