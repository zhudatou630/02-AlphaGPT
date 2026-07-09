"""Numpy operators used by the Phase 3a StackVM."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd


ArrayOp = Callable[..., np.ndarray]


@dataclass(frozen=True)
class OperatorSpec:
    name: str
    arity: int
    func: ArrayOp


def delay(values: np.ndarray, periods: int) -> np.ndarray:
    out = np.full_like(values, np.nan, dtype=float)
    if periods <= 0:
        return values.astype(float).copy()
    out[:, periods:] = values[:, :-periods]
    return out


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    return pd.DataFrame(values.T).rolling(window, min_periods=window).mean().to_numpy().T


def rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    return pd.DataFrame(values.T).rolling(window, min_periods=window).std(ddof=0).to_numpy().T


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = numerator / denominator
    return finite_or_nan(out)


def ret(values: np.ndarray, periods: int) -> np.ndarray:
    return safe_divide(values, delay(values, periods)) - 1.0


def decay(values: np.ndarray) -> np.ndarray:
    return (values + 0.8 * delay(values, 1) + 0.6 * delay(values, 2)) / 2.4


def finite_or_nan(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=float).copy()
    out[~np.isfinite(out)] = np.nan
    return out


OPS: dict[str, OperatorSpec] = {
    "ADD": OperatorSpec("ADD", 2, lambda x, y: x + y),
    "SUB": OperatorSpec("SUB", 2, lambda x, y: x - y),
    "MUL": OperatorSpec("MUL", 2, lambda x, y: x * y),
    "DIV": OperatorSpec("DIV", 2, safe_divide),
    "NEG": OperatorSpec("NEG", 1, lambda x: -x),
    "ABS": OperatorSpec("ABS", 1, np.abs),
    "SIGN": OperatorSpec("SIGN", 1, np.sign),
    "DELAY1": OperatorSpec("DELAY1", 1, lambda x: delay(x, 1)),
    "DELAY5": OperatorSpec("DELAY5", 1, lambda x: delay(x, 5)),
    "DELAY10": OperatorSpec("DELAY10", 1, lambda x: delay(x, 10)),
    "MA5": OperatorSpec("MA5", 1, lambda x: rolling_mean(x, 5)),
    "MA10": OperatorSpec("MA10", 1, lambda x: rolling_mean(x, 10)),
    "MA20": OperatorSpec("MA20", 1, lambda x: rolling_mean(x, 20)),
    "STD10": OperatorSpec("STD10", 1, lambda x: rolling_std(x, 10)),
    "RET5": OperatorSpec("RET5", 1, lambda x: ret(x, 5)),
    "RET10": OperatorSpec("RET10", 1, lambda x: ret(x, 10)),
    "DECAY": OperatorSpec("DECAY", 1, decay),
}
