"""Scale-independent OHLC features for the V3 research panel."""

from __future__ import annotations

import numpy as np
import pandas as pd


RELATIVE_FEATURES = ("open_rel", "high_rel", "low_rel", "close_rel")


def add_relative_ohlc(adjusted: pd.DataFrame) -> pd.DataFrame:
    required = {"symbol", "date", "tradable", *(f"event_qfq_{x}" for x in ("open", "high", "low", "close"))}
    missing = required - set(adjusted.columns)
    if missing:
        raise ValueError(f"Adjusted data missing columns: {sorted(missing)}")
    out = adjusted.sort_values(["symbol", "date"]).copy()
    out["previous_adjusted_close"] = out.groupby("symbol")["event_qfq_close"].shift(1)
    denominator = out["previous_adjusted_close"].where(out["previous_adjusted_close"] > 0)
    for price in ("open", "high", "low", "close"):
        out[f"{price}_rel"] = out[f"event_qfq_{price}"] / denominator - 1.0
    return out


def relative_panel(
    relative: pd.DataFrame,
    symbols: list[str],
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    dates = pd.DatetimeIndex(sorted(relative["date"].unique()))
    values = np.full((len(symbols), len(RELATIVE_FEATURES), len(dates)), np.nan, dtype=np.float64)
    mask = np.zeros((len(symbols), len(dates)), dtype=bool)
    symbol_index = {symbol: index for index, symbol in enumerate(symbols)}
    date_index = {date: index for index, date in enumerate(dates)}
    for row in relative.itertuples(index=False):
        symbol = str(row.symbol)
        if symbol not in symbol_index or not bool(row.tradable):
            continue
        feature_values = np.array([getattr(row, feature) for feature in RELATIVE_FEATURES], dtype=float)
        if not np.isfinite(feature_values).all():
            continue
        n = symbol_index[symbol]
        t = date_index[pd.Timestamp(row.date)]
        values[n, :, t] = feature_values
        mask[n, t] = True
    return values, mask, dates


def relative_quality(relative: pd.DataFrame) -> dict[str, int]:
    finite = np.isfinite(relative[list(RELATIVE_FEATURES)]).all(axis=1)
    invalid_high = finite & (
        relative["high_rel"] < relative[["open_rel", "close_rel"]].max(axis=1)
    )
    invalid_low = finite & (
        relative["low_rel"] > relative[["open_rel", "close_rel"]].min(axis=1)
    )
    impossible_price = finite & (relative[list(RELATIVE_FEATURES)] <= -1.0).any(axis=1)
    return {
        "invalid_relative_high_rows": int(invalid_high.sum()),
        "invalid_relative_low_rows": int(invalid_low.sum()),
        "impossible_relative_price_rows": int(impossible_price.sum()),
    }