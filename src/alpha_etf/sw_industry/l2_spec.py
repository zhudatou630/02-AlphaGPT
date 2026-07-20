"""Immutable SW2021 level-2 dynamic industry-panel helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from alpha_etf.sw_industry.spec import canonical_sha256, sha256_file


DATASET_SCHEMA_VERSION = "sw2021-l2-dynamic-industry-panel-v1"
ABSOLUTE_FEATURES = ("open", "high", "low", "close")
EXPECTED_INDUSTRIES = 124


@dataclass(frozen=True)
class SWIndustryL2Panel:
    absolute_ohlc: np.ndarray
    tradable_mask: np.ndarray
    symbols: np.ndarray
    names: np.ndarray
    dates: pd.DatetimeIndex

    def absolute(self, feature: str) -> np.ndarray:
        try:
            index = ABSOLUTE_FEATURES.index(feature)
        except ValueError as exc:
            raise KeyError(f"Unknown absolute feature: {feature}") from exc
        return self.absolute_ohlc[:, index, :]


def build_dynamic_panel_arrays(
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    required = {"ts_code", "industry_name", "trade_date", *ABSOLUTE_FEATURES}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Governed SW2021 L2 data missing columns: {sorted(missing)}")
    data = frame.copy()
    data["ts_code"] = data["ts_code"].astype(str)
    data["industry_name"] = data["industry_name"].astype(str)
    data["trade_date"] = pd.to_datetime(data["trade_date"].astype(str)).dt.normalize()
    duplicate = data.duplicated(["ts_code", "trade_date"], keep=False)
    if duplicate.any():
        examples = data.loc[duplicate, ["ts_code", "trade_date"]].head(10).to_dict("records")
        raise ValueError(f"Governed SW2021 L2 data has duplicate keys: {examples}")

    symbols = np.asarray(sorted(data["ts_code"].unique()), dtype=str)
    if len(symbols) != EXPECTED_INDUSTRIES:
        raise ValueError(f"SW2021 L2 industry count differs: {len(symbols)}")
    name_counts = data.groupby("ts_code")["industry_name"].nunique()
    if (name_counts != 1).any():
        raise ValueError("Frozen L2 industry_name must be unique within each ts_code")
    names_by_symbol = data.groupby("ts_code")["industry_name"].first()
    names = np.asarray([names_by_symbol[symbol] for symbol in symbols], dtype=str)
    dates = pd.DatetimeIndex(sorted(data["trade_date"].unique()))

    indexed = data.set_index(["ts_code", "trade_date"])
    target = pd.MultiIndex.from_product([symbols, dates], names=["ts_code", "trade_date"])
    indexed = indexed.reindex(target)
    values = np.stack(
        [
            indexed[feature].to_numpy(dtype=np.float64).reshape(len(symbols), len(dates))
            for feature in ABSOLUTE_FEATURES
        ],
        axis=1,
    )
    mask = np.isfinite(values).all(axis=1) & (values > 0).all(axis=1)
    partial_rows = np.isfinite(values).any(axis=1) & ~mask
    if partial_rows.any():
        raise ValueError("SW2021 L2 panel contains partial or non-positive OHLC rows")
    if not mask[:, -1].all():
        raise ValueError("Every frozen L2 industry must be available on the final date")
    open_, high, low, close = (values[:, index, :] for index in range(4))
    high_bad = mask & (high < np.maximum.reduce([open_, low, close]))
    low_bad = mask & (low > np.minimum.reduce([open_, high, close]))
    if high_bad.any() or low_bad.any():
        raise ValueError("SW2021 L2 panel violates the OHLC envelope")
    return values, mask, symbols, names, dates


def build_dataset_manifest(identity: dict[str, Any]) -> dict[str, Any]:
    if identity.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise ValueError("Unexpected SW2021 L2 dataset schema")
    fingerprint = canonical_sha256(identity)
    return {
        **identity,
        "dataset_fingerprint": fingerprint,
        "dataset_id": f"sw2021-l2-dynamic-panel-{fingerprint[:12]}",
    }


def load_dataset_manifest(dataset_dir: Path) -> dict[str, Any]:
    manifest_path = dataset_dir / "dataset_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing SW2021 L2 dataset manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise RuntimeError("SW2021 L2 dataset schema mismatch")
    identity = dict(manifest)
    expected_id = str(identity.pop("dataset_id", ""))
    expected_fingerprint = str(identity.pop("dataset_fingerprint", ""))
    actual_fingerprint = canonical_sha256(identity)
    if actual_fingerprint != expected_fingerprint:
        raise RuntimeError("SW2021 L2 dataset fingerprint mismatch")
    if expected_id != f"sw2021-l2-dynamic-panel-{actual_fingerprint[:12]}":
        raise RuntimeError("SW2021 L2 dataset ID mismatch")
    panel_path = dataset_dir / str(manifest.get("panel_file", ""))
    if sha256_file(panel_path) != manifest.get("panel_sha256"):
        raise RuntimeError("SW2021 L2 panel SHA mismatch")
    availability_path = dataset_dir / str(manifest.get("availability_file", ""))
    if sha256_file(availability_path) != manifest.get("availability_sha256"):
        raise RuntimeError("SW2021 L2 availability SHA mismatch")
    return manifest


def load_panel(dataset_dir: Path) -> SWIndustryL2Panel:
    manifest = load_dataset_manifest(dataset_dir)
    panel_path = dataset_dir / str(manifest["panel_file"])
    with np.load(panel_path, allow_pickle=False) as payload:
        required = {
            "absolute_ohlc",
            "tradable_mask",
            "symbols",
            "names",
            "dates",
            "features",
        }
        missing = required - set(payload.files)
        if missing:
            raise ValueError(f"SW2021 L2 panel missing arrays: {sorted(missing)}")
        absolute = payload["absolute_ohlc"].astype(np.float64)
        mask = payload["tradable_mask"].astype(bool)
        symbols = payload["symbols"].astype(str)
        names = payload["names"].astype(str)
        dates = payload["dates"].astype(str)
        features = tuple(payload["features"].astype(str))
    expected_shape = (len(symbols), len(ABSOLUTE_FEATURES), len(dates))
    if features != ABSOLUTE_FEATURES or absolute.shape != expected_shape:
        raise ValueError("SW2021 L2 panel feature schema or shape mismatch")
    if mask.shape != (len(symbols), len(dates)):
        raise ValueError("SW2021 L2 panel mask shape mismatch")
    if list(mask.shape) != manifest.get("mask_shape"):
        raise ValueError("SW2021 L2 panel mask shape differs from manifest")
    parsed_dates = pd.DatetimeIndex(pd.to_datetime(dates))
    if not parsed_dates.is_unique or not parsed_dates.is_monotonic_increasing:
        raise ValueError("SW2021 L2 panel dates must be unique and strictly increasing")
    if len(dates) != manifest.get("date_count"):
        raise ValueError("SW2021 L2 panel date count differs from manifest")
    finite_positive = np.isfinite(absolute).all(axis=1) & (absolute > 0).all(axis=1)
    if not np.array_equal(mask, finite_positive):
        raise ValueError("SW2021 L2 panel mask differs from finite positive OHLC")
    if np.isfinite(absolute[~np.broadcast_to(mask[:, None, :], absolute.shape)]).any():
        raise ValueError("Unavailable SW2021 L2 panel values must remain NaN")
    if list(symbols) != manifest.get("symbols") or list(names) != manifest.get("names"):
        raise ValueError("SW2021 L2 panel identity differs from manifest")
    if list(absolute.shape) != manifest.get("panel_shape"):
        raise ValueError("SW2021 L2 panel shape differs from manifest")
    if int(mask.sum()) != manifest.get("tradable_observations"):
        raise ValueError("SW2021 L2 tradable observation count differs from manifest")
    if int(mask[:, 0].sum()) != manifest.get("first_day_quote_universe"):
        raise ValueError("SW2021 L2 first-day universe differs from manifest")
    if int(mask[:, -1].sum()) != manifest.get("final_day_quote_universe"):
        raise ValueError("SW2021 L2 final-day universe differs from manifest")
    if dates[0] != manifest.get("date_start") or dates[-1] != manifest.get("date_end"):
        raise ValueError("SW2021 L2 panel date range differs from manifest")
    return SWIndustryL2Panel(
        absolute_ohlc=absolute,
        tradable_mask=mask,
        symbols=symbols,
        names=names,
        dates=parsed_dates,
    )