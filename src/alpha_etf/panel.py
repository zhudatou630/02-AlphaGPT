"""Panel loading helpers for Phase 2 research."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
RAW_PANEL_NPZ = PROCESSED_DIR / "etf_panel_raw.npz"
QFQ_PANEL_NPZ = PROCESSED_DIR / "etf_panel_qfq_latest.npz"


@dataclass(frozen=True)
class MarketPanel:
    raw_values: np.ndarray
    qfq_values: np.ndarray
    mask: np.ndarray
    symbols: np.ndarray
    features: np.ndarray
    dates: pd.DatetimeIndex

    def feature_index(self, feature: str) -> int:
        matches = np.where(self.features == feature)[0]
        if len(matches) != 1:
            raise KeyError(f"Feature not found: {feature}")
        return int(matches[0])

    def raw(self, feature: str) -> np.ndarray:
        return self.raw_values[:, self.feature_index(feature), :]

    def qfq(self, feature: str) -> np.ndarray:
        return self.qfq_values[:, self.feature_index(feature), :]



def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing panel file: {path}")
    # Phase 1 originally saved dates as object dtype. allow_pickle keeps old
    # artifacts readable; regenerated artifacts no longer need it.
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}



def load_market_panel(raw_path: Path = RAW_PANEL_NPZ, qfq_path: Path = QFQ_PANEL_NPZ) -> MarketPanel:
    raw = _load_npz(raw_path)
    qfq = _load_npz(qfq_path)

    for key in ("values", "mask", "symbols", "features", "dates"):
        if key not in raw:
            raise ValueError(f"raw panel missing key: {key}")
        if key not in qfq:
            raise ValueError(f"qfq panel missing key: {key}")

    if not np.array_equal(raw["symbols"], qfq["symbols"]):
        raise ValueError("raw/qfq symbols differ")
    if not np.array_equal(raw["features"], qfq["features"]):
        raise ValueError("raw/qfq features differ")
    if not np.array_equal(raw["dates"].astype(str), qfq["dates"].astype(str)):
        raise ValueError("raw/qfq dates differ")

    mask = raw["mask"] & qfq["mask"]
    if not np.array_equal(raw["mask"], qfq["mask"]):
        raise ValueError("raw/qfq masks differ")

    return MarketPanel(
        raw_values=raw["values"].astype(float),
        qfq_values=qfq["values"].astype(float),
        mask=mask.astype(bool),
        symbols=raw["symbols"].astype(str),
        features=raw["features"].astype(str),
        dates=pd.DatetimeIndex(pd.to_datetime(raw["dates"].astype(str))),
    )
