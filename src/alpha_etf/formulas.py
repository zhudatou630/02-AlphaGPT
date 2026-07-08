"""Fixed Phase 2 formula probes."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from alpha_etf.panel import MarketPanel


Formula = Callable[[MarketPanel], np.ndarray]



def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    return pd.DataFrame(values.T).rolling(window, min_periods=window).mean().to_numpy().T



def _rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    return pd.DataFrame(values.T).rolling(window, min_periods=window).std(ddof=0).to_numpy().T



def _delay(values: np.ndarray, periods: int) -> np.ndarray:
    out = np.full_like(values, np.nan, dtype=float)
    if periods <= 0:
        return values.astype(float).copy()
    out[:, periods:] = values[:, :-periods]
    return out



def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = numerator / denominator
    out[~np.isfinite(out)] = np.nan
    return out



def mom_10(panel: MarketPanel) -> np.ndarray:
    close = panel.qfq("close")
    return _safe_divide(close, _delay(close, 10)) - 1.0



def ma_gap_5_20(panel: MarketPanel) -> np.ndarray:
    close = panel.qfq("close")
    ma5 = _rolling_mean(close, 5)
    ma20 = _rolling_mean(close, 20)
    return _safe_divide(ma5, ma20) - 1.0



def volume_confirmed_mom_10(panel: MarketPanel) -> np.ndarray:
    amount = panel.raw("amount")
    amount_mean = _rolling_mean(amount, 20)
    amount_std = _rolling_std(amount, 20)
    amount_z = _safe_divide(amount - amount_mean, amount_std)
    return mom_10(panel) * np.maximum(amount_z, 0.0)



def reversal_5(panel: MarketPanel) -> np.ndarray:
    close = panel.qfq("close")
    return -(_safe_divide(close, _delay(close, 5)) - 1.0)



def fixed_formulas() -> dict[str, Formula]:
    return {
        "mom_10": mom_10,
        "ma_gap_5_20": ma_gap_5_20,
        "volume_confirmed_mom_10": volume_confirmed_mom_10,
        "reversal_5": reversal_5,
    }
