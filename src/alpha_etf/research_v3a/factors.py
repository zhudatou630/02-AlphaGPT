"""Scale-independent V3A base factors for NumPy and Torch."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import torch

from alpha_etf.research_v3a.spec import V3APanel


FACTOR_SPEC_VERSION = "etf-v3a-factors-v1"
WINDOWS = (5, 10, 20, 40, 60)
FIXED_FACTOR_NAMES = ("DAYRET", "GAP", "INTRADAY", "RANGE", "CLV")
WINDOW_FACTOR_FAMILIES = ("ROC", "PRICE_MA", "VOL", "TS_RANK", "RSV")
MA_RATIO_WINDOWS = tuple(combinations(WINDOWS, 2))
FACTOR_NAMES = (
    FIXED_FACTOR_NAMES
    + tuple(f"{family}_{window}" for family in WINDOW_FACTOR_FAMILIES for window in WINDOWS)
    + tuple(f"MA_RATIO_{short}_{long}" for short, long in MA_RATIO_WINDOWS)
)


def factor_config() -> dict[str, Any]:
    return {
        "version": FACTOR_SPEC_VERSION,
        "windows": list(WINDOWS),
        "factor_names": list(FACTOR_NAMES),
        "rolling_min_periods": "full_window",
        "calendar": "global_exchange_trading_dates",
        "missing": "NaN, no forward fill, no valid-row compression",
        "volatility_ddof": 0,
        "zero_range": 0.0,
    }


def _delay_numpy(values: np.ndarray, periods: int) -> np.ndarray:
    out = np.full_like(values, np.nan, dtype=np.float64)
    if periods <= 0:
        return values.astype(np.float64, copy=True)
    if periods < values.shape[-1]:
        out[..., periods:] = values[..., :-periods]
    return out


def _ratio_numpy(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator > 0)
    out = np.full(np.broadcast_shapes(numerator.shape, denominator.shape), np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=out, where=valid)
    return out


def _rolling_sum_numpy(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(values)
    cleaned = np.where(finite, values, 0.0)
    padded_values = np.pad(cleaned, ((0, 0), (1, 0)), constant_values=0.0)
    padded_counts = np.pad(finite.astype(np.int64), ((0, 0), (1, 0)), constant_values=0)
    sums = np.cumsum(padded_values, axis=-1)
    counts = np.cumsum(padded_counts, axis=-1)
    rolling_sums = sums[:, window:] - sums[:, :-window]
    rolling_counts = counts[:, window:] - counts[:, :-window]
    out_sums = np.full_like(values, np.nan, dtype=np.float64)
    out_counts = np.zeros_like(values, dtype=np.int64)
    out_sums[:, window - 1 :] = rolling_sums
    out_counts[:, window - 1 :] = rolling_counts
    return out_sums, out_counts


def rolling_mean_numpy(values: np.ndarray, window: int) -> np.ndarray:
    sums, counts = _rolling_sum_numpy(values, window)
    out = sums / float(window)
    return np.where(counts == window, out, np.nan)


def rolling_std_numpy(values: np.ndarray, window: int) -> np.ndarray:
    sums, counts = _rolling_sum_numpy(values, window)
    sumsq, _ = _rolling_sum_numpy(values * values, window)
    mean = sums / float(window)
    variance = np.maximum(sumsq / float(window) - mean * mean, 0.0)
    return np.where(counts == window, np.sqrt(variance), np.nan)


def _rolling_extreme_numpy(values: np.ndarray, window: int, *, maximum: bool) -> np.ndarray:
    out = np.full_like(values, np.nan, dtype=np.float64)
    if window > values.shape[-1]:
        return out
    windows = np.lib.stride_tricks.sliding_window_view(values, window, axis=-1)
    valid = np.isfinite(windows).all(axis=-1)
    with np.errstate(invalid="ignore"):
        reduced = np.max(windows, axis=-1) if maximum else np.min(windows, axis=-1)
    out[:, window - 1 :] = np.where(valid, reduced, np.nan)
    return out


def rolling_rank_numpy(values: np.ndarray, window: int) -> np.ndarray:
    out = np.full_like(values, np.nan, dtype=np.float64)
    if window > values.shape[-1]:
        return out
    windows = np.lib.stride_tricks.sliding_window_view(values, window, axis=-1)
    valid = np.isfinite(windows).all(axis=-1)
    current = windows[..., -1:]
    less = np.sum(windows < current, axis=-1, dtype=np.float64)
    equal = np.sum(windows == current, axis=-1, dtype=np.float64)
    average_rank = less + (equal + 1.0) / 2.0
    rank = (average_rank - 1.0) / float(window - 1)
    rank = np.where(equal == window, 0.5, rank)
    out[:, window - 1 :] = np.where(valid, rank, np.nan)
    return out


def build_factor_values_numpy(absolute_ohlc: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if absolute_ohlc.ndim != 3 or absolute_ohlc.shape[1] != 4:
        raise ValueError(f"absolute_ohlc must have shape [asset,4,date], got {absolute_ohlc.shape}")
    if mask.shape != (absolute_ohlc.shape[0], absolute_ohlc.shape[2]):
        raise ValueError(f"mask shape differs from absolute OHLC: {mask.shape}")

    values = absolute_ohlc.astype(np.float64, copy=True)
    values = np.where(mask[:, None, :], values, np.nan)
    open_, high, low, close = (values[:, index, :] for index in range(4))
    previous_close = _delay_numpy(close, 1)

    dayret = _ratio_numpy(close, previous_close) - 1.0
    gap = _ratio_numpy(open_, previous_close) - 1.0
    intraday = _ratio_numpy(close, open_) - 1.0
    price_range = _ratio_numpy(high - low, previous_close)
    width = high - low
    clv_valid = np.isfinite(close) & np.isfinite(high) & np.isfinite(low) & (width >= 0)
    clv = np.full_like(close, np.nan)
    np.divide(2.0 * close - high - low, width, out=clv, where=clv_valid & (width > 0))
    clv = np.where(clv_valid & (width == 0), 0.0, clv)

    factors: list[np.ndarray] = [dayret, gap, intraday, price_range, clv]
    mean_cache = {window: rolling_mean_numpy(close, window) for window in WINDOWS}
    min_low_cache = {
        window: _rolling_extreme_numpy(low, window, maximum=False) for window in WINDOWS
    }
    max_high_cache = {
        window: _rolling_extreme_numpy(high, window, maximum=True) for window in WINDOWS
    }

    for family in WINDOW_FACTOR_FAMILIES:
        for window in WINDOWS:
            if family == "ROC":
                factor = _ratio_numpy(close, _delay_numpy(close, window)) - 1.0
            elif family == "PRICE_MA":
                factor = _ratio_numpy(close, mean_cache[window]) - 1.0
            elif family == "VOL":
                factor = rolling_std_numpy(dayret, window)
            elif family == "TS_RANK":
                factor = rolling_rank_numpy(close, window)
            elif family == "RSV":
                lower = min_low_cache[window]
                upper = max_high_cache[window]
                span = upper - lower
                valid = np.isfinite(close) & np.isfinite(lower) & np.isfinite(upper) & (span >= 0)
                factor = np.full_like(close, np.nan)
                np.divide(close - lower, span, out=factor, where=valid & (span > 0))
                factor = np.where(valid & (span == 0), 0.0, factor)
            else:  # pragma: no cover - frozen family list.
                raise AssertionError(f"Unknown factor family: {family}")
            factors.append(factor)

    for short, long in MA_RATIO_WINDOWS:
        factors.append(_ratio_numpy(mean_cache[short], mean_cache[long]) - 1.0)

    output = np.stack(factors, axis=0)
    if output.shape != (len(FACTOR_NAMES), mask.shape[0], mask.shape[1]):
        raise RuntimeError(f"V3A factor cache shape mismatch: {output.shape}")
    return np.where(mask[None, :, :], output, np.nan)


def attach_factor_cache(panel: V3APanel) -> V3APanel:
    values = build_factor_values_numpy(panel.absolute_ohlc, panel.tradable_mask)
    return panel.with_factor_cache(values, FACTOR_NAMES)


def _delay_torch(values: torch.Tensor, periods: int) -> torch.Tensor:
    out = torch.full_like(values, float("nan"))
    if periods <= 0:
        return values.clone()
    if periods < values.shape[-1]:
        out[..., periods:] = values[..., :-periods]
    return out


def _ratio_torch(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    valid = torch.isfinite(numerator) & torch.isfinite(denominator) & (denominator > 0)
    return torch.where(valid, numerator / denominator, torch.full_like(numerator, float("nan")))


def _rolling_sum_torch(values: torch.Tensor, window: int) -> tuple[torch.Tensor, torch.Tensor]:
    finite = torch.isfinite(values)
    cleaned = torch.where(finite, values, torch.zeros_like(values))
    padded_values = torch.nn.functional.pad(cleaned, (1, 0), value=0.0)
    padded_counts = torch.nn.functional.pad(finite.to(values.dtype), (1, 0), value=0.0)
    sums = padded_values.cumsum(dim=-1)
    counts = padded_counts.cumsum(dim=-1)
    rolling_sums = sums[..., window:] - sums[..., :-window]
    rolling_counts = counts[..., window:] - counts[..., :-window]
    out_sums = torch.full_like(values, float("nan"))
    out_counts = torch.zeros_like(values)
    out_sums[..., window - 1 :] = rolling_sums
    out_counts[..., window - 1 :] = rolling_counts
    return out_sums, out_counts


def rolling_mean_torch(values: torch.Tensor, window: int) -> torch.Tensor:
    out = torch.full_like(values, float("nan"))
    if window > values.shape[-1]:
        return out
    windows = values.unfold(-1, window, 1)
    valid = torch.isfinite(windows).all(dim=-1)
    cleaned = torch.where(torch.isfinite(windows), windows, torch.zeros_like(windows))
    mean = cleaned.mean(dim=-1)
    out[..., window - 1 :] = torch.where(valid, mean, torch.full_like(mean, float("nan")))
    return out


def rolling_std_torch(values: torch.Tensor, window: int) -> torch.Tensor:
    out = torch.full_like(values, float("nan"))
    if window > values.shape[-1]:
        return out
    windows = values.unfold(-1, window, 1)
    valid = torch.isfinite(windows).all(dim=-1)
    cleaned = torch.where(torch.isfinite(windows), windows, torch.zeros_like(windows))
    mean = cleaned.mean(dim=-1, keepdim=True)
    variance = torch.mean((cleaned - mean) ** 2, dim=-1)
    std = torch.sqrt(torch.clamp(variance, min=0.0))
    out[..., window - 1 :] = torch.where(valid, std, torch.full_like(std, float("nan")))
    return out


def _rolling_extreme_torch(values: torch.Tensor, window: int, *, maximum: bool) -> torch.Tensor:
    out = torch.full_like(values, float("nan"))
    if window > values.shape[-1]:
        return out
    windows = values.unfold(-1, window, 1)
    valid = torch.isfinite(windows).all(dim=-1)
    reduced = windows.amax(dim=-1) if maximum else windows.amin(dim=-1)
    out[..., window - 1 :] = torch.where(valid, reduced, torch.full_like(reduced, float("nan")))
    return out


def rolling_rank_torch(values: torch.Tensor, window: int) -> torch.Tensor:
    out = torch.full_like(values, float("nan"))
    if window > values.shape[-1]:
        return out
    windows = values.unfold(-1, window, 1)
    valid = torch.isfinite(windows).all(dim=-1)
    current = windows[..., -1:]
    less = (windows < current).sum(dim=-1).to(values.dtype)
    equal = (windows == current).sum(dim=-1).to(values.dtype)
    average_rank = less + (equal + 1.0) / 2.0
    rank = (average_rank - 1.0) / float(window - 1)
    rank = torch.where(equal == float(window), torch.full_like(rank, 0.5), rank)
    out[..., window - 1 :] = torch.where(valid, rank, torch.full_like(rank, float("nan")))
    return out


def build_factor_values_torch(absolute_ohlc: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if absolute_ohlc.ndim != 3 or absolute_ohlc.shape[1] != 4:
        raise ValueError(f"absolute_ohlc must have shape [asset,4,date], got {absolute_ohlc.shape}")
    if absolute_ohlc.dtype != torch.float64:
        raise ValueError(
            "V3A factors must be computed from float64 absolute OHLC; cast only the finished "
            "factor cache to float32"
        )
    if mask.shape != (absolute_ohlc.shape[0], absolute_ohlc.shape[2]):
        raise ValueError(f"mask shape differs from absolute OHLC: {tuple(mask.shape)}")

    values = torch.where(
        mask[:, None, :], absolute_ohlc, torch.full_like(absolute_ohlc, float("nan"))
    )
    open_, high, low, close = (values[:, index, :] for index in range(4))
    previous_close = _delay_torch(close, 1)
    dayret = _ratio_torch(close, previous_close) - 1.0
    gap = _ratio_torch(open_, previous_close) - 1.0
    intraday = _ratio_torch(close, open_) - 1.0
    price_range = _ratio_torch(high - low, previous_close)
    width = high - low
    clv_valid = torch.isfinite(close) & torch.isfinite(high) & torch.isfinite(low) & (width >= 0)
    clv = torch.where(
        clv_valid & (width > 0),
        (2.0 * close - high - low) / width,
        torch.full_like(close, float("nan")),
    )
    clv = torch.where(clv_valid & (width == 0), torch.zeros_like(clv), clv)

    factors: list[torch.Tensor] = [dayret, gap, intraday, price_range, clv]
    mean_cache = {window: rolling_mean_torch(close, window) for window in WINDOWS}
    min_low_cache = {
        window: _rolling_extreme_torch(low, window, maximum=False) for window in WINDOWS
    }
    max_high_cache = {
        window: _rolling_extreme_torch(high, window, maximum=True) for window in WINDOWS
    }
    for family in WINDOW_FACTOR_FAMILIES:
        for window in WINDOWS:
            if family == "ROC":
                factor = _ratio_torch(close, _delay_torch(close, window)) - 1.0
            elif family == "PRICE_MA":
                factor = _ratio_torch(close, mean_cache[window]) - 1.0
            elif family == "VOL":
                factor = rolling_std_torch(dayret, window)
            elif family == "TS_RANK":
                factor = rolling_rank_torch(close, window)
            elif family == "RSV":
                lower = min_low_cache[window]
                upper = max_high_cache[window]
                span = upper - lower
                valid = torch.isfinite(close) & torch.isfinite(lower) & torch.isfinite(upper) & (span >= 0)
                factor = torch.where(
                    valid & (span > 0),
                    (close - lower) / span,
                    torch.full_like(close, float("nan")),
                )
                factor = torch.where(valid & (span == 0), torch.zeros_like(factor), factor)
            else:  # pragma: no cover - frozen family list.
                raise AssertionError(f"Unknown factor family: {family}")
            factors.append(factor)
    for short, long in MA_RATIO_WINDOWS:
        factors.append(_ratio_torch(mean_cache[short], mean_cache[long]) - 1.0)

    output = torch.stack(factors, dim=0)
    return torch.where(mask[None, :, :], output, torch.full_like(output, float("nan")))