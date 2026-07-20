"""Build event-complete, high-precision adjusted ETF market data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


PRICE_COLUMNS = ("open", "high", "low", "close")
PANEL_FEATURES = ("open", "high", "low", "close", "volume", "amount")
SUPPORTED_EVENT_CATEGORIES = (1, 11, 12)


@dataclass(frozen=True)
class EventAdjustment:
    price_step: float
    share_step: float
    theoretical_ex_close: float
    event_kind: str


def _required(df: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = columns - set(df.columns)
    if missing:
        raise ValueError(f"{label} missing columns: {sorted(missing)}")


def _event_adjustment(event: pd.Series, previous_close: float) -> EventAdjustment:
    category = int(event["category_code"])
    if category == 1:
        dividend = float(event["c1"])
        rights_price = float(event["c2"])
        bonus_shares = float(event["c3"])
        rights_shares = float(event["c4"])
        share_step = (10.0 + bonus_shares + rights_shares) / 10.0
        cash_per_share = (dividend - rights_shares * rights_price) / 10.0
        theoretical_ex_close = (previous_close - cash_per_share) / share_step
        event_kind = "cash_bonus_rights"
    elif category in (11, 12):
        share_step = float(event["c3"])
        theoretical_ex_close = previous_close / share_step
        event_kind = "share_split_merge"
    else:
        raise ValueError(f"Unsupported adjustment event category: {category}")

    if not np.isfinite(share_step) or share_step <= 0:
        raise ValueError(f"Invalid share multiplier: {share_step}")
    if not np.isfinite(theoretical_ex_close) or theoretical_ex_close <= 0:
        raise ValueError(f"Invalid theoretical ex close: {theoretical_ex_close}")
    return EventAdjustment(
        price_step=float(theoretical_ex_close / previous_close),
        share_step=share_step,
        theoretical_ex_close=float(theoretical_ex_close),
        event_kind=event_kind,
    )


def reconcile_flow_data(raw: pd.DataFrame, tushare_daily: pd.DataFrame | None = None) -> pd.DataFrame:
    """Reconcile volume/amount while preserving both source values.

    TDX and Tushare use lots for ETF volume. TDX amount is yuan, while Tushare
    fund_daily amount is thousand yuan. Tushare flow replaces TDX on overlapping
    dates because a small number of low-volume TDX rows decode incorrectly.
    """

    _required(
        raw,
        {"date", "symbol", "low", "high", "volume", "amount"},
        "raw daily data",
    )
    daily = raw.copy()
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    daily["symbol"] = daily["symbol"].astype(str)
    daily = daily.rename(columns={"volume": "tdx_volume", "amount": "tdx_amount"})

    if tushare_daily is not None:
        _required(
            tushare_daily,
            {"trade_date", "symbol", "vol", "amount"},
            "Tushare fund daily data",
        )
        tushare = tushare_daily.copy()
        tushare["date"] = pd.to_datetime(tushare["trade_date"]).dt.normalize()
        tushare["symbol"] = tushare["symbol"].astype(str)
        duplicate_tushare = tushare.duplicated(["symbol", "date"], keep=False)
        if duplicate_tushare.any():
            examples = tushare.loc[duplicate_tushare, ["symbol", "date"]].head(10).to_dict("records")
            raise ValueError(f"Tushare flow data has duplicate symbol/date rows: {examples}")
        tushare = tushare[["symbol", "date", "vol", "amount"]].rename(
            columns={"vol": "tushare_volume", "amount": "tushare_amount_thousand_yuan"}
        )
        daily = daily.merge(tushare, on=["symbol", "date"], how="left")
    else:
        daily["tushare_volume"] = np.nan
        daily["tushare_amount_thousand_yuan"] = np.nan

    has_tushare = daily["tushare_volume"].notna() & daily["tushare_amount_thousand_yuan"].notna()
    daily["volume"] = daily["tushare_volume"].where(has_tushare, daily["tdx_volume"]).astype(float)
    daily["amount"] = (
        (daily["tushare_amount_thousand_yuan"] * 1000.0).where(has_tushare, daily["tdx_amount"])
    ).astype(float)
    daily["flow_source"] = np.where(has_tushare, "tushare_fund_daily", "tdx_fallback")
    daily["tradable"] = (daily["volume"] > 0) & (daily["amount"] > 0)
    daily["flow_implied_price"] = np.where(
        daily["tradable"], daily["amount"] / (daily["volume"] * 100.0), np.nan
    )
    return daily


def build_event_audit(raw: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Resolve TDX events against previous/effective trading rows.

    Category 1 is ordinary dividend/rights/bonus-share data. Categories 11 and
    12 are ETF share split/merge events. Other event categories remain in the
    audit inventory but do not change prices.
    """

    _required(
        raw,
        {"date", "symbol", "close", "volume"},
        "raw daily data",
    )
    _required(
        events,
        {"date", "symbol", "category_code", "c1", "c2", "c3", "c4"},
        "event data",
    )

    daily = raw.copy()
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    daily["symbol"] = daily["symbol"].astype(str)
    daily = daily.sort_values(["symbol", "date"]).reset_index(drop=True)
    duplicate_daily = daily.duplicated(["symbol", "date"], keep=False)
    if duplicate_daily.any():
        examples = daily.loc[duplicate_daily, ["symbol", "date"]].head(10).to_dict("records")
        raise ValueError(f"Raw daily data has duplicate symbol/date rows: {examples}")

    event_data = events.copy()
    event_data["date"] = pd.to_datetime(event_data["date"]).dt.normalize()
    event_data["symbol"] = event_data["symbol"].astype(str)
    event_data = event_data.sort_values(["symbol", "date", "category_code"]).reset_index(drop=True)

    daily_by_symbol = {symbol: group.reset_index(drop=True) for symbol, group in daily.groupby("symbol")}
    rows: list[dict[str, object]] = []
    for event in event_data.to_dict("records"):
        symbol = str(event["symbol"])
        event_date = pd.Timestamp(event["date"])
        category = int(event["category_code"])
        row: dict[str, object] = {
            **event,
            "event_kind": "non_adjustment_event",
            "applied": False,
            "skip_reason": "unsupported_category",
            "previous_trade_date": pd.NaT,
            "effective_trade_date": pd.NaT,
            "previous_close": np.nan,
            "effective_close": np.nan,
            "previous_raw_volume": np.nan,
            "effective_raw_volume": np.nan,
            "previous_volume_on_post_event_units": np.nan,
            "raw_volume_event_ratio": np.nan,
            "adjusted_volume_event_ratio": np.nan,
            "price_step": np.nan,
            "share_step": np.nan,
            "theoretical_ex_close": np.nan,
            "raw_event_return": np.nan,
            "adjusted_event_return": np.nan,
        }
        if category not in SUPPORTED_EVENT_CATEGORIES:
            rows.append(row)
            continue

        symbol_daily = daily_by_symbol.get(symbol)
        if symbol_daily is None:
            row["skip_reason"] = "symbol_missing_from_daily"
            rows.append(row)
            continue

        previous = symbol_daily[symbol_daily["date"] < event_date].tail(1)
        effective = symbol_daily[symbol_daily["date"] >= event_date].head(1)
        if previous.empty:
            row["skip_reason"] = "no_previous_daily_row"
            rows.append(row)
            continue
        if effective.empty:
            row["skip_reason"] = "future_event"
            rows.append(row)
            continue

        previous_row = previous.iloc[0]
        effective_row = effective.iloc[0]
        previous_close = float(previous_row["close"])
        effective_close = float(effective_row["close"])
        adjustment = _event_adjustment(pd.Series(event), previous_close)
        previous_raw_volume = float(previous_row["volume"])
        effective_raw_volume = float(effective_row["volume"])
        previous_volume_on_post_event_units = previous_raw_volume * adjustment.share_step
        row.update(
            {
                "event_kind": adjustment.event_kind,
                "applied": True,
                "skip_reason": "",
                "previous_trade_date": previous_row["date"],
                "effective_trade_date": effective_row["date"],
                "previous_close": previous_close,
                "effective_close": effective_close,
                "previous_raw_volume": previous_raw_volume,
                "effective_raw_volume": effective_raw_volume,
                "previous_volume_on_post_event_units": previous_volume_on_post_event_units,
                "raw_volume_event_ratio": (
                    effective_raw_volume / previous_raw_volume if previous_raw_volume > 0 else np.nan
                ),
                "adjusted_volume_event_ratio": (
                    effective_raw_volume / previous_volume_on_post_event_units
                    if previous_volume_on_post_event_units > 0
                    else np.nan
                ),
                "price_step": adjustment.price_step,
                "share_step": adjustment.share_step,
                "theoretical_ex_close": adjustment.theoretical_ex_close,
                "raw_event_return": effective_close / previous_close - 1.0,
                "adjusted_event_return": effective_close / adjustment.theoretical_ex_close - 1.0,
            }
        )
        rows.append(row)

    audit = pd.DataFrame(rows)
    audit["event_sequence_index"] = 1
    audit["event_sequence_count"] = 1
    audit["cumulative_share_step"] = audit["share_step"]
    applied = audit[audit["applied"].astype(bool)]
    for _, group in applied.groupby(["symbol", "effective_trade_date"], sort=False):
        order = group.assign(
            event_order=group["category_code"].map({11: 0, 12: 0, 1: 1}).fillna(2)
        ).sort_values(["date", "event_order"])
        reference = float(order.iloc[0]["previous_close"])
        previous_raw_volume = float(order.iloc[0]["previous_raw_volume"])
        effective_close = float(order.iloc[0]["effective_close"])
        effective_raw_volume = float(order.iloc[0]["effective_raw_volume"])
        cumulative_share_step = 1.0
        count = len(order)
        for sequence, (index, event) in enumerate(order.iterrows(), start=1):
            adjustment = _event_adjustment(event, reference)
            cumulative_share_step *= adjustment.share_step
            previous_volume_on_post_event_units = (
                previous_raw_volume * cumulative_share_step
            )
            audit.loc[index, [
                "event_sequence_index",
                "event_sequence_count",
                "previous_close",
                "price_step",
                "share_step",
                "cumulative_share_step",
                "theoretical_ex_close",
                "previous_volume_on_post_event_units",
                "adjusted_volume_event_ratio",
                "adjusted_event_return",
            ]] = [
                sequence,
                count,
                reference,
                adjustment.price_step,
                adjustment.share_step,
                cumulative_share_step,
                adjustment.theoretical_ex_close,
                previous_volume_on_post_event_units,
                (
                    effective_raw_volume / previous_volume_on_post_event_units
                    if previous_volume_on_post_event_units > 0
                    else np.nan
                ),
                (
                    effective_close / adjustment.theoretical_ex_close - 1.0
                    if sequence == count
                    else np.nan
                ),
            ]
            reference = adjustment.theoretical_ex_close
    return audit


def apply_event_adjustments(raw: pd.DataFrame, event_audit: pd.DataFrame) -> pd.DataFrame:
    """Apply multiplicative price and share-unit adjustments to raw ETF data.

    Price factors are anchored to the latest share/price basis. This preserves
    ordinary within-regime returns and removes mechanical event gaps. Volume is
    adjusted only by share-count changes; amount remains raw currency turnover.
    """

    _required(
        raw,
        {"date", "symbol", *PRICE_COLUMNS, "volume", "amount"},
        "raw daily data",
    )
    _required(
        event_audit,
        {"date", "symbol", "applied", "price_step", "share_step"},
        "event audit",
    )

    daily = raw.copy()
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    daily["symbol"] = daily["symbol"].astype(str)
    daily = daily.sort_values(["symbol", "date"]).reset_index(drop=True)

    adjusted_groups = []
    applied_events = event_audit[event_audit["applied"].astype(bool)].copy()
    applied_events["date"] = pd.to_datetime(applied_events["date"]).dt.normalize()
    for symbol, group in daily.groupby("symbol", sort=False):
        group = group.copy()
        price_factor = np.ones(len(group), dtype=np.float64)
        share_factor = np.ones(len(group), dtype=np.float64)
        symbol_events = applied_events[applied_events["symbol"].astype(str) == str(symbol)]
        for event in symbol_events.itertuples(index=False):
            before_event = group["date"].to_numpy() < np.datetime64(event.date)
            price_factor[before_event] *= float(event.price_step)
            share_factor[before_event] *= float(event.share_step)

        group["price_factor"] = price_factor
        group["share_factor"] = share_factor
        for column in PRICE_COLUMNS:
            group[f"event_qfq_{column}"] = group[column].astype(float) * price_factor
        group["event_adjusted_volume"] = group["volume"].astype(float) * share_factor
        adjusted_groups.append(group)

    adjusted = pd.concat(adjusted_groups, ignore_index=True)
    adjusted["raw_close_return"] = adjusted.groupby("symbol")["close"].pct_change(fill_method=None)
    adjusted["event_qfq_close_return"] = adjusted.groupby("symbol")["event_qfq_close"].pct_change(
        fill_method=None
    )
    event_counts = applied_events.copy()
    event_counts["date"] = pd.to_datetime(event_counts["effective_trade_date"]).dt.normalize()
    event_counts = event_counts.groupby(["symbol", "date"]).size().rename("applied_event_count").reset_index()
    adjusted = adjusted.merge(event_counts, on=["symbol", "date"], how="left")
    adjusted["applied_event_count"] = adjusted["applied_event_count"].fillna(0).astype(int)
    return adjusted


def quality_summary(adjusted: pd.DataFrame, event_audit: pd.DataFrame, threshold: float = 0.20) -> pd.DataFrame:
    _required(
        adjusted,
        {
            "symbol",
            "date",
            "raw_close_return",
            "event_qfq_close_return",
            "event_qfq_open",
            "event_qfq_high",
            "event_qfq_low",
            "event_qfq_close",
        },
        "adjusted daily data",
    )
    applied = event_audit[event_audit["applied"].astype(bool)].copy()
    rows = []
    for symbol, group in adjusted.groupby("symbol", sort=False):
        prices = group[[f"event_qfq_{column}" for column in PRICE_COLUMNS]]
        invalid_price = (~np.isfinite(prices)).any(axis=1) | (prices <= 0).any(axis=1)
        invalid_high = group["event_qfq_high"] < group[["event_qfq_open", "event_qfq_close"]].max(axis=1)
        invalid_low = group["event_qfq_low"] > group[["event_qfq_open", "event_qfq_close"]].min(axis=1)
        adjusted_volume = group["event_adjusted_volume"].astype(float)
        amount = group["amount"].astype(float)
        tradable = (
            group["tradable"].astype(bool)
            if "tradable" in group
            else (group["volume"].astype(float) > 0) & (amount > 0)
        )
        implied_price = amount / group["volume"].astype(float).replace(0, np.nan) / 100.0
        invalid_implied_price = tradable & (
            (implied_price < group["low"].astype(float) * 0.98)
            | (implied_price > group["high"].astype(float) * 1.02)
        )
        flow_zero_mismatch = (group["volume"].astype(float) == 0) != (amount == 0)
        symbol_events = applied[applied["symbol"].astype(str) == str(symbol)]
        rows.append(
            {
                "symbol": str(symbol),
                "name": group["name"].iloc[0] if "name" in group else "",
                "rows": int(len(group)),
                "first_date": group["date"].min().date().isoformat(),
                "last_date": group["date"].max().date().isoformat(),
                "applied_event_count": int(len(symbol_events)),
                "share_event_count": int(symbol_events["category_code"].isin([11, 12]).sum()),
                "raw_abs_return_gt_threshold": int((group["raw_close_return"].abs() > threshold).sum()),
                "adjusted_abs_return_gt_threshold": int(
                    (group["event_qfq_close_return"].abs() > threshold).sum()
                ),
                "max_abs_raw_return": float(group["raw_close_return"].abs().max(skipna=True)),
                "max_abs_adjusted_return": float(
                    group["event_qfq_close_return"].abs().max(skipna=True)
                ),
                "invalid_adjusted_price_rows": int(invalid_price.sum()),
                "invalid_adjusted_high_rows": int(invalid_high.sum()),
                "invalid_adjusted_low_rows": int(invalid_low.sum()),
                "invalid_adjusted_volume_rows": int(
                    ((~np.isfinite(adjusted_volume)) | (adjusted_volume < 0)).sum()
                ),
                "invalid_raw_amount_rows": int(((~np.isfinite(amount)) | (amount < 0)).sum()),
                "invalid_flow_implied_price_rows": int(invalid_implied_price.sum()),
                "flow_zero_mismatch_rows": int(flow_zero_mismatch.sum()),
                "nontradable_rows": int((~tradable).sum()),
            }
        )
    return pd.DataFrame(rows)


def quality_gate_failures(quality: pd.DataFrame, event_audit: pd.DataFrame) -> dict[str, int]:
    expected_skip_reasons = {"no_previous_daily_row", "future_event"}
    supported = event_audit[event_audit["category_code"].isin(SUPPORTED_EVENT_CATEGORIES)]
    unexpected_skips = supported[
        (~supported["applied"].astype(bool)) & (~supported["skip_reason"].isin(expected_skip_reasons))
    ]
    gate_columns = (
        "adjusted_abs_return_gt_threshold",
        "invalid_adjusted_price_rows",
        "invalid_adjusted_high_rows",
        "invalid_adjusted_low_rows",
        "invalid_adjusted_volume_rows",
        "invalid_raw_amount_rows",
        "invalid_flow_implied_price_rows",
        "flow_zero_mismatch_rows",
    )
    failures = {column: int(quality[column].sum()) for column in gate_columns}
    failures["unexpected_supported_event_skips"] = int(len(unexpected_skips))
    return {name: count for name, count in failures.items() if count != 0}


def anomaly_rows(adjusted: pd.DataFrame, threshold: float = 0.20) -> pd.DataFrame:
    mask = (adjusted["raw_close_return"].abs() > threshold) | (
        adjusted["event_qfq_close_return"].abs() > threshold
    )
    columns = [
        "symbol",
        "name",
        "date",
        "close",
        "event_qfq_close",
        "raw_close_return",
        "event_qfq_close_return",
        "price_factor",
        "share_factor",
        "volume",
        "event_adjusted_volume",
        "amount",
        "applied_event_count",
    ]
    return adjusted.loc[mask, [column for column in columns if column in adjusted]].copy()


def panel_arrays(
    adjusted: pd.DataFrame,
    symbols: list[str],
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    dates = pd.DatetimeIndex(sorted(adjusted["date"].unique()))
    values = np.full((len(symbols), len(PANEL_FEATURES), len(dates)), np.nan, dtype=np.float64)
    mask = np.zeros((len(symbols), len(dates)), dtype=bool)
    date_index = {date: idx for idx, date in enumerate(dates)}
    symbol_index = {symbol: idx for idx, symbol in enumerate(symbols)}
    source_columns = {
        "open": "event_qfq_open",
        "high": "event_qfq_high",
        "low": "event_qfq_low",
        "close": "event_qfq_close",
        "volume": "event_adjusted_volume",
        "amount": "amount",
    }
    for row in adjusted.itertuples(index=False):
        symbol = str(row.symbol)
        if symbol not in symbol_index:
            continue
        n = symbol_index[symbol]
        t = date_index[pd.Timestamp(row.date)]
        if hasattr(row, "tradable") and not bool(row.tradable):
            continue
        for feature_idx, feature in enumerate(PANEL_FEATURES):
            values[n, feature_idx, t] = float(getattr(row, source_columns[feature]))
        mask[n, t] = bool(np.isfinite(values[n, :, t]).all())
    return values, mask, dates