"""Immutable SW2021 industry-panel identity and loading helpers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DATASET_SCHEMA_VERSION = "sw2021-l1-industry-panel-v1"
ABSOLUTE_FEATURES = ("open", "high", "low", "close")
EXPECTED_INDUSTRIES = 31


@dataclass(frozen=True)
class SWIndustryPanel:
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: dict[str, Any]) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_panel_arrays(
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    required = {"ts_code", "industry_name", "trade_date", *ABSOLUTE_FEATURES}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Governed SW2021 data missing columns: {sorted(missing)}")
    data = frame.copy()
    data["ts_code"] = data["ts_code"].astype(str)
    data["industry_name"] = data["industry_name"].astype(str)
    data["trade_date"] = pd.to_datetime(data["trade_date"].astype(str)).dt.normalize()
    duplicate = data.duplicated(["ts_code", "trade_date"], keep=False)
    if duplicate.any():
        examples = data.loc[duplicate, ["ts_code", "trade_date"]].head(10).to_dict("records")
        raise ValueError(f"Governed SW2021 data has duplicate keys: {examples}")

    symbols = np.asarray(sorted(data["ts_code"].unique()), dtype=str)
    if len(symbols) != EXPECTED_INDUSTRIES:
        raise ValueError(f"SW2021 industry count differs: {len(symbols)}")
    name_counts = data.groupby("ts_code")["industry_name"].nunique()
    if (name_counts != 1).any():
        raise ValueError("Frozen industry_name must be unique within each ts_code")
    names_by_symbol = data.groupby("ts_code")["industry_name"].first()
    names = np.asarray([names_by_symbol[symbol] for symbol in symbols], dtype=str)
    dates = pd.DatetimeIndex(sorted(data["trade_date"].unique()))
    expected_rows = len(symbols) * len(dates)
    if len(data) != expected_rows:
        raise ValueError(f"SW2021 panel is not rectangular: {len(data)} != {expected_rows}")

    indexed = data.set_index(["ts_code", "trade_date"])
    target = pd.MultiIndex.from_product([symbols, dates], names=["ts_code", "trade_date"])
    indexed = indexed.reindex(target)
    values = np.stack(
        [indexed[feature].to_numpy(dtype=np.float64).reshape(len(symbols), len(dates))
         for feature in ABSOLUTE_FEATURES],
        axis=1,
    )
    finite_positive = np.isfinite(values).all(axis=1) & (values > 0).all(axis=1)
    if not finite_positive.all():
        raise ValueError("SW2021 panel contains non-finite or non-positive OHLC")
    open_, high, low, close = (values[:, index, :] for index in range(4))
    if (high < np.maximum.reduce([open_, low, close])).any():
        raise ValueError("SW2021 panel high violates the OHLC envelope")
    if (low > np.minimum.reduce([open_, high, close])).any():
        raise ValueError("SW2021 panel low violates the OHLC envelope")
    mask = np.ones((len(symbols), len(dates)), dtype=bool)
    return values, mask, symbols, names, dates


def build_dataset_manifest(identity: dict[str, Any]) -> dict[str, Any]:
    if identity.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise ValueError("Unexpected SW2021 dataset schema")
    fingerprint = canonical_sha256(identity)
    return {
        **identity,
        "dataset_fingerprint": fingerprint,
        "dataset_id": f"sw2021-l1-panel-{fingerprint[:12]}",
    }


def load_dataset_manifest(dataset_dir: Path) -> dict[str, Any]:
    manifest_path = dataset_dir / "dataset_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing SW2021 dataset manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise RuntimeError("SW2021 dataset schema mismatch")
    identity = dict(manifest)
    expected_id = str(identity.pop("dataset_id", ""))
    expected_fingerprint = str(identity.pop("dataset_fingerprint", ""))
    actual_fingerprint = canonical_sha256(identity)
    if actual_fingerprint != expected_fingerprint:
        raise RuntimeError("SW2021 dataset fingerprint mismatch")
    if expected_id != f"sw2021-l1-panel-{actual_fingerprint[:12]}":
        raise RuntimeError("SW2021 dataset ID mismatch")
    panel_path = dataset_dir / str(manifest.get("panel_file", ""))
    if sha256_file(panel_path) != manifest.get("panel_sha256"):
        raise RuntimeError("SW2021 panel SHA mismatch")
    return manifest


def load_panel(dataset_dir: Path) -> SWIndustryPanel:
    manifest = load_dataset_manifest(dataset_dir)
    panel_path = dataset_dir / str(manifest["panel_file"])
    with np.load(panel_path, allow_pickle=False) as payload:
        required = {
            "absolute_ohlc", "tradable_mask", "symbols", "names", "dates", "features"
        }
        missing = required - set(payload.files)
        if missing:
            raise ValueError(f"SW2021 panel missing arrays: {sorted(missing)}")
        absolute = payload["absolute_ohlc"].astype(np.float64)
        mask = payload["tradable_mask"].astype(bool)
        symbols = payload["symbols"].astype(str)
        names = payload["names"].astype(str)
        dates = payload["dates"].astype(str)
        features = tuple(payload["features"].astype(str))
    expected_shape = (len(symbols), len(ABSOLUTE_FEATURES), len(dates))
    if features != ABSOLUTE_FEATURES or absolute.shape != expected_shape:
        raise ValueError("SW2021 panel feature schema or shape mismatch")
    if mask.shape != (len(symbols), len(dates)) or not mask.all():
        raise ValueError("SW2021 panel mask must be a complete rectangle")
    if list(symbols) != manifest.get("symbols") or list(names) != manifest.get("names"):
        raise ValueError("SW2021 panel identity differs from manifest")
    if list(absolute.shape) != manifest.get("panel_shape"):
        raise ValueError("SW2021 panel shape differs from manifest")
    if dates[0] != manifest.get("date_start") or dates[-1] != manifest.get("date_end"):
        raise ValueError("SW2021 panel date range differs from manifest")
    if not np.isfinite(absolute).all() or not (absolute > 0).all():
        raise ValueError("SW2021 panel prices must be finite and positive")
    return SWIndustryPanel(
        absolute_ohlc=absolute,
        tradable_mask=mask,
        symbols=symbols,
        names=names,
        dates=pd.DatetimeIndex(pd.to_datetime(dates)),
    )
