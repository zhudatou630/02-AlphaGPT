"""Identity and loading for the physically 2023-free V3A validation view."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from alpha_etf.research_v3a.factors import FACTOR_NAMES
from alpha_etf.research_v3a.spec import canonical_sha256


VALIDATION_VIEW_SCHEMA_VERSION = "etf-v3a-validation-view-v1"


@dataclass(frozen=True)
class ValidationView:
    factor_values: np.ndarray
    absolute_open: np.ndarray
    absolute_close: np.ndarray
    tradable_mask: np.ndarray
    symbols: np.ndarray
    dates: pd.DatetimeIndex
    manifest: dict[str, Any]


def validation_view_fingerprint(payload: dict[str, Any]) -> str:
    identity = dict(payload)
    identity.pop("validation_view_id", None)
    identity.pop("validation_view_fingerprint", None)
    return canonical_sha256(identity)


def build_validation_view_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema_version") != VALIDATION_VIEW_SCHEMA_VERSION:
        raise ValueError("V3A validation-view schema mismatch")
    fingerprint = validation_view_fingerprint(payload)
    return {
        **payload,
        "validation_view_fingerprint": fingerprint,
        "validation_view_id": f"v3a-validation-view-{fingerprint[:16]}",
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_validation_view(path: Path) -> ValidationView:
    manifest_path = path / "validation_view_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing validation view manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != VALIDATION_VIEW_SCHEMA_VERSION:
        raise RuntimeError("V3A validation-view schema mismatch")
    fingerprint = validation_view_fingerprint(manifest)
    if (
        manifest.get("validation_view_fingerprint") != fingerprint
        or manifest.get("validation_view_id")
        != f"v3a-validation-view-{fingerprint[:16]}"
    ):
        raise RuntimeError("V3A validation-view identity mismatch")
    if manifest.get("split") != {
        "signal_start": "2016-08-09",
        "validation_start": "2022-01-01",
        "validation_end": "2022-12-31",
        "data_end": "2022-12-31",
        "validation_columns_present": True,
        "final_columns_present": False,
    }:
        raise RuntimeError("V3A validation-view split mismatch")
    required = {
        "factor_values": "factor_values.npy",
        "absolute_open": "absolute_open.npy",
        "absolute_close": "absolute_close.npy",
        "tradable_mask": "tradable_mask.npy",
        "symbols": "symbols.npy",
        "dates": "dates.npy",
    }
    arrays: dict[str, np.ndarray] = {}
    for name, filename in required.items():
        entry = manifest.get("files", {}).get(name, {})
        file_path = path / filename
        if entry.get("path") != filename or not file_path.exists():
            raise RuntimeError(f"V3A validation-view file mapping mismatch: {name}")
        if _sha256(file_path) != entry.get("sha256"):
            raise RuntimeError(f"V3A validation-view SHA mismatch: {name}")
        arrays[name] = np.load(file_path, allow_pickle=False)
    factors = arrays["factor_values"]
    open_prices = arrays["absolute_open"]
    close_prices = arrays["absolute_close"]
    mask = arrays["tradable_mask"]
    symbols = arrays["symbols"].astype(str)
    date_values = arrays["dates"].astype(str)
    if (
        factors.dtype != np.float64
        or open_prices.dtype != np.float64
        or close_prices.dtype != np.float64
        or mask.dtype != np.bool_
        or factors.shape != (len(FACTOR_NAMES), len(symbols), len(date_values))
        or open_prices.shape != mask.shape
        or close_prices.shape != mask.shape
        or mask.shape != (len(symbols), len(date_values))
    ):
        raise RuntimeError("V3A validation-view array contract mismatch")
    if (
        manifest.get("factor_names") != list(FACTOR_NAMES)
        or list(factors.shape) != manifest.get("factor_shape")
        or list(mask.shape) != manifest.get("mask_shape")
        or not len(date_values)
        or date_values[-1] != "2022-12-30"
        or manifest.get("date_end") != date_values[-1]
        or any(value >= "2023-01-01" for value in date_values)
    ):
        raise RuntimeError("V3A validation-view exposes final dates or drifted metadata")
    dates = pd.DatetimeIndex(pd.to_datetime(date_values))
    if not dates.is_monotonic_increasing or not dates.is_unique:
        raise RuntimeError("V3A validation-view dates are not strictly ordered")
    if list(symbols) != list(manifest["source_dataset_manifest"]["symbols"]):
        raise RuntimeError("V3A validation-view symbols differ from dataset manifest")
    for prices, label in ((open_prices, "open"), (close_prices, "close")):
        if np.isnan(prices[mask]).any() or not (prices[mask] > 0).all():
            raise RuntimeError(f"V3A validation-view tradable {label} is invalid")
        if not np.isnan(prices[~mask]).all():
            raise RuntimeError(f"V3A validation-view nontradable {label} must be NaN")
    return ValidationView(
        factor_values=factors,
        absolute_open=open_prices,
        absolute_close=close_prices,
        tradable_mask=mask,
        symbols=symbols,
        dates=dates,
        manifest=manifest,
    )