"""Point-in-time identity governance for the expanded historical ETF universe."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd


AUDIT_SCHEMA_VERSION = "expanded-etf-universe-audit-v1"
IDENTITY_INTERVAL_SCHEMA_VERSION = "expanded-etf-identity-interval-v1"
REPRESENTATIVE_SCHEMA_VERSION = "expanded-etf-representative-lifecycle-v1"
METADATA_EVENT_COLUMNS = (
    "ts_code",
    "event_date",
    "event_type",
    "old_index_code",
    "new_index_code",
    "old_index_name",
    "new_index_name",
    "canonical_variety_id",
    "evidence_url",
    "requires_roc_reset",
    "notes",
)

NON_DOMESTIC_PATTERNS = (
    "QDII",
    "港股",
    "港股通",
    "沪港深",
    "沪深港",
    "香港",
    "恒生",
    "纳斯达克",
    "标普",
    "道琼斯",
    "日经",
    "东证",
    "德国",
    "法国",
    "美国",
    "日本",
    "韩国",
    "印度",
    "沙特",
    "新加坡",
    "东南亚",
    "海外",
    "中概",
    "中国互联网50",
    "中国互联网30",
)
DOMESTIC_ALLOW_PATTERNS = ("MSCI中国A50互联互通",)

OVERRIDE_COLUMNS = (
    "ts_code",
    "index_code",
    "index_name",
    "effective_from",
    "effective_to",
    "evidence_url",
    "evidence_type",
    "change_type",
    "notes",
)


def normalize_benchmark(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = re.sub(r"[（(].*?人民币.*?[）)]", "", text)
    text = text.replace("指数P", "指数")
    text = re.sub(r"(?:收益率)?[×xX*]?100(?:\.0+)?%", "", text)
    text = re.sub(r"[\s（）()]+", "", text)
    return text.strip("+；;,，")


def identity_name_key(value: object) -> str:
    text = normalize_benchmark(value)
    return re.sub(r"指数$", "", text)


def classify_product(row: pd.Series) -> tuple[str, str]:
    if str(row.get("fund_type", "")) != "股票型":
        return "exclude_non_equity", "fund_type"
    if str(row.get("invest_type", "")) != "被动指数型":
        return "exclude_non_passive", "invest_type"
    product_text = f"{row.get('name', '')}|{row.get('type', '')}".upper()
    if "联接" in product_text or "LOF" in product_text:
        return "exclude_non_etf_product", "feeder_or_lof"
    if pd.isna(row.get("list_status")):
        return "review_missing_etf_master", "not_found_in_etf_basic"
    etf_type = "" if pd.isna(row.get("etf_type")) else str(row.get("etf_type"))
    if etf_type and etf_type != "纯境内":
        return "separate_cross_border", f"etf_type={etf_type}"
    text = f"{row.get('name', '')}|{row.get('benchmark', '')}".upper()
    if any(pattern.upper() in text for pattern in DOMESTIC_ALLOW_PATTERNS):
        return "include_domestic_passive_equity", "explicit_domestic_allow"
    matches = [pattern for pattern in NON_DOMESTIC_PATTERNS if pattern.upper() in text]
    if matches:
        return "separate_cross_border", ";".join(matches)
    return "include_domestic_passive_equity", "mechanical_scope"


def _parse_date_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    values = frame[column].replace({"": pd.NA, "nan": pd.NA})
    return pd.to_datetime(values, format="%Y%m%d", errors="coerce")


def _unique_mapping(values: pd.Series) -> object:
    unique = tuple(sorted(set(values.dropna().astype(str))))
    return unique[0] if len(unique) == 1 else pd.NA


def build_product_master(
    fund_basic: pd.DataFrame,
    etf_basic: pd.DataFrame,
    index_basic: pd.DataFrame | None = None,
    identity_candidates: pd.DataFrame | None = None,
) -> pd.DataFrame:
    required = {"ts_code", "name", "fund_type", "invest_type", "list_date", "status"}
    missing = required - set(fund_basic.columns)
    if missing:
        raise ValueError(f"fund_basic missing columns: {sorted(missing)}")
    if fund_basic["ts_code"].duplicated().any():
        raise ValueError("fund_basic contains duplicate ts_code")
    etf_required = {"ts_code", "index_code", "index_name", "etf_type", "list_status"}
    missing = etf_required - set(etf_basic.columns)
    if missing:
        raise ValueError(f"etf_basic missing columns: {sorted(missing)}")
    if etf_basic["ts_code"].duplicated().any():
        raise ValueError("etf_basic contains duplicate ts_code")

    funds = fund_basic.copy()
    is_etf = funds["name"].fillna("").str.contains("ETF", case=False)
    funds = funds.loc[is_etf].copy()
    etfs = etf_basic[
        [
            "ts_code",
            "index_code",
            "index_name",
            "etf_type",
            "list_status",
            "exchange",
            "extname",
        ]
    ].copy()
    master = funds.merge(etfs, on="ts_code", how="left", validate="one_to_one")
    master["symbol"] = master["ts_code"].str[:6]
    master["list_date_parsed"] = _parse_date_column(master, "list_date")
    master["delist_date_parsed"] = _parse_date_column(master, "delist_date")
    master["benchmark_group"] = master["benchmark"].map(normalize_benchmark)
    master["benchmark_identity_key"] = master["benchmark"].map(identity_name_key)

    classifications = master.apply(classify_product, axis=1, result_type="expand")
    classifications.columns = ["scope_decision", "scope_reason"]
    master = pd.concat([master, classifications], axis=1)

    direct = master.dropna(subset=["index_code"])
    group_map = direct.groupby("benchmark_group")["index_code"].agg(_unique_mapping)
    group_inferred = master["benchmark_group"].map(group_map)
    index_name_inferred = pd.Series(pd.NA, index=master.index, dtype="object")
    index_master_names = pd.Series(dtype="object")
    if index_basic is not None:
        index_required = {"ts_code", "name"}
        missing = index_required - set(index_basic.columns)
        if missing:
            raise ValueError(f"index_basic missing columns: {sorted(missing)}")
        index_values = index_basic.dropna(subset=["ts_code", "name"]).copy()
        index_values["identity_key"] = index_values["name"].map(identity_name_key)
        index_map = index_values.groupby("identity_key")["ts_code"].agg(_unique_mapping)
        index_name_inferred = master["benchmark_identity_key"].map(index_map)
        index_master_names = index_values.groupby("ts_code")["name"].agg(_unique_mapping)
    master["resolved_index_code"] = master["index_code"]
    master["resolved_index_code"] = master["resolved_index_code"].where(
        master["resolved_index_code"].notna(), group_inferred
    )
    master["resolved_index_code"] = master["resolved_index_code"].where(
        master["resolved_index_code"].notna(), index_name_inferred
    )
    direct_names = direct.groupby("index_code")["index_name"].agg(_unique_mapping)
    master["resolved_index_name"] = master["index_name"].where(
        master["index_name"].notna(), master["resolved_index_code"].map(direct_names)
    )
    master["resolved_index_name"] = master["resolved_index_name"].where(
        master["resolved_index_name"].notna(), master["resolved_index_code"].map(index_master_names)
    )
    master["identity_resolution"] = "unresolved"
    master["identity_candidate_evidence_url"] = pd.NA
    master["identity_candidate_confidence"] = pd.NA
    master.loc[index_name_inferred.notna(), "identity_resolution"] = "index_basic_unique_name"
    master.loc[master["resolved_index_code"].notna(), "identity_resolution"] = (
        "same_benchmark_unique_current_index"
    )
    master.loc[
        master["index_code"].isna() & group_inferred.isna() & index_name_inferred.notna(),
        "identity_resolution",
    ] = "index_basic_unique_name"
    if identity_candidates is not None:
        candidate_required = {
            "ts_code",
            "index_code",
            "index_name",
            "evidence_url",
            "confidence",
        }
        missing = candidate_required - set(identity_candidates.columns)
        if missing:
            raise ValueError(f"identity candidates missing columns: {sorted(missing)}")
        if identity_candidates["ts_code"].duplicated().any():
            raise ValueError("identity candidates contain duplicate ts_code")
        candidates = identity_candidates.set_index("ts_code")
        candidate_codes = master["ts_code"].map(candidates["index_code"])
        candidate_names = master["ts_code"].map(candidates["index_name"])
        candidate_evidence = master["ts_code"].map(candidates["evidence_url"])
        candidate_confidence = master["ts_code"].map(candidates["confidence"])
        use_candidate = master["resolved_index_code"].isna() & candidate_codes.notna()
        master.loc[use_candidate, "resolved_index_code"] = candidate_codes[use_candidate]
        master.loc[use_candidate, "resolved_index_name"] = candidate_names[use_candidate]
        master.loc[use_candidate, "identity_resolution"] = "provider_identity_candidate"
        master.loc[use_candidate, "identity_candidate_evidence_url"] = candidate_evidence[
            use_candidate
        ]
        master.loc[use_candidate, "identity_candidate_confidence"] = candidate_confidence[
            use_candidate
        ]
    master.loc[master["index_code"].notna(), "identity_resolution"] = "etf_basic_current"
    master["history_identity_status"] = "current_identity_requires_history_review"
    master.loc[master["resolved_index_code"].isna(), "history_identity_status"] = (
        "missing_official_index_code"
    )
    return master.sort_values(["scope_decision", "list_date_parsed", "ts_code"]).reset_index(
        drop=True
    )


def validate_overrides(overrides: pd.DataFrame) -> pd.DataFrame:
    missing = set(OVERRIDE_COLUMNS) - set(overrides.columns)
    if missing:
        raise ValueError(f"identity overrides missing columns: {sorted(missing)}")
    values = overrides.loc[:, OVERRIDE_COLUMNS].copy()
    if values.empty:
        return values
    for column in ("ts_code", "index_code", "index_name", "effective_from", "evidence_url"):
        if values[column].isna().any() or values[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"identity overrides require non-empty {column}")
    values["effective_from_parsed"] = _parse_date_column(values, "effective_from")
    values["effective_to_parsed"] = _parse_date_column(values, "effective_to")
    if values["effective_from_parsed"].isna().any():
        raise ValueError("identity overrides contain invalid effective_from")
    bad = values["effective_to_parsed"].notna() & (
        values["effective_to_parsed"] < values["effective_from_parsed"]
    )
    if bad.any():
        raise ValueError("identity overrides contain effective_to before effective_from")
    return values


def build_identity_intervals(
    master: pd.DataFrame,
    overrides: pd.DataFrame,
    *,
    as_of: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = pd.Timestamp(as_of)
    override_values = validate_overrides(overrides)
    target = master[master["scope_decision"] == "include_domestic_passive_equity"].copy()
    rows: list[dict[str, Any]] = []
    exceptions: list[dict[str, Any]] = []
    override_codes = set(override_values["ts_code"].astype(str)) if not override_values.empty else set()

    for product in target.itertuples(index=False):
        if pd.isna(product.list_date_parsed):
            exceptions.append(_exception(product.ts_code, "missing_list_date", "error"))
            continue
        if not pd.isna(product.delist_date_parsed) and product.delist_date_parsed < product.list_date_parsed:
            exceptions.append(_exception(product.ts_code, "delist_before_list", "error"))
            continue
        product_overrides = override_values[override_values["ts_code"].astype(str) == product.ts_code]
        if not product_overrides.empty:
            for value in product_overrides.itertuples(index=False):
                confidence = (
                    "official_historical_interval"
                    if str(value.evidence_type).startswith("official_")
                    else "reviewed_historical_interval"
                )
                rows.append(
                    _interval_row(
                        product,
                        value.index_code,
                        value.index_name,
                        value.effective_from_parsed,
                        value.effective_to_parsed,
                        "official_override",
                        confidence,
                        value.evidence_url,
                    )
                )
            continue
        if pd.isna(product.resolved_index_code):
            exceptions.append(_exception(product.ts_code, "missing_official_index_code", "error"))
            continue
        effective_to = product.delist_date_parsed
        if pd.isna(effective_to):
            effective_to = cutoff
        identity_confidence = "provisional_current_identity_projected_back"
        evidence = "tushare.etf_basic/fund_basic snapshot"
        if product.identity_resolution == "provider_identity_candidate":
            identity_confidence = "provider_identity_candidate_history_unverified"
            evidence = str(product.identity_candidate_evidence_url)
        rows.append(
            _interval_row(
                product,
                product.resolved_index_code,
                product.resolved_index_name,
                product.list_date_parsed,
                effective_to,
                product.identity_resolution,
                identity_confidence,
                evidence,
            )
        )
        exceptions.append(
            _exception(product.ts_code, "history_identity_unverified", "warning")
        )

    unknown_overrides = override_codes - set(master["ts_code"].astype(str))
    if unknown_overrides:
        raise ValueError(f"identity overrides contain unknown products: {sorted(unknown_overrides)}")
    intervals = pd.DataFrame(rows)
    if not intervals.empty:
        intervals = intervals.sort_values(["index_code", "effective_from", "ts_code"]).reset_index(
            drop=True
        )
        _validate_non_overlapping_product_intervals(intervals)
    exception_frame = pd.DataFrame(
        exceptions, columns=["ts_code", "exception_code", "severity", "message"]
    ).sort_values(["severity", "exception_code", "ts_code"])
    return intervals, exception_frame.reset_index(drop=True)


def _interval_row(
    product: Any,
    index_code: object,
    index_name: object,
    effective_from: pd.Timestamp,
    effective_to: pd.Timestamp | pd.NaT,
    source: str,
    confidence: str,
    evidence: str,
) -> dict[str, Any]:
    return {
        "schema_version": IDENTITY_INTERVAL_SCHEMA_VERSION,
        "ts_code": str(product.ts_code),
        "symbol": str(product.symbol),
        "product_name": str(product.name),
        "product_list_date": pd.Timestamp(product.list_date_parsed),
        "index_code": str(index_code),
        "variety_id": str(index_code),
        "index_name": "" if pd.isna(index_name) else str(index_name),
        "effective_from": pd.Timestamp(effective_from),
        "effective_to": pd.NaT if pd.isna(effective_to) else pd.Timestamp(effective_to),
        "warmup_from": pd.Timestamp(effective_from),
        "identity_source": source,
        "identity_confidence": confidence,
        "evidence": evidence,
    }


def _exception(ts_code: str, code: str, severity: str) -> dict[str, str]:
    return {
        "ts_code": str(ts_code),
        "exception_code": code,
        "severity": severity,
        "message": code.replace("_", " "),
    }


def _validate_non_overlapping_product_intervals(intervals: pd.DataFrame) -> None:
    for ts_code, group in intervals.groupby("ts_code"):
        ordered = group.sort_values("effective_from")
        previous_end: pd.Timestamp | None = None
        for row in ordered.itertuples(index=False):
            if previous_end is not None and row.effective_from <= previous_end:
                raise ValueError(f"overlapping identity intervals for {ts_code}")
            previous_end = None if pd.isna(row.effective_to) else pd.Timestamp(row.effective_to)


def apply_index_metadata_events(
    intervals: pd.DataFrame, events: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = set(METADATA_EVENT_COLUMNS) - set(events.columns)
    if missing:
        raise ValueError(f"index metadata events missing columns: {sorted(missing)}")
    values = events.loc[:, METADATA_EVENT_COLUMNS].copy()
    if values.empty:
        return intervals.copy(), values
    values["event_date_parsed"] = pd.to_datetime(
        values["event_date"], format="%Y%m%d", errors="coerce"
    )
    if values["event_date_parsed"].isna().any():
        raise ValueError("index metadata events contain invalid event_date")
    result = intervals.copy()
    for event in values.itertuples(index=False):
        date = pd.Timestamp(event.event_date_parsed)
        matches = result[
            result["ts_code"].eq(event.ts_code)
            & (result["effective_from"] <= date)
            & (result["effective_to"].isna() | (result["effective_to"] >= date))
        ]
        if len(matches) != 1:
            raise ValueError(
                "metadata event does not map to exactly one identity interval: "
                f"{event.ts_code} {event.event_date}"
            )
        original = matches.iloc[0].copy()
        if str(original["index_code"]) != str(event.new_index_code):
            raise ValueError(f"metadata event new index differs from current identity: {event.ts_code}")
        result = result.drop(index=matches.index)
        before = original.copy()
        before["index_code"] = str(event.old_index_code)
        before["index_name"] = str(event.old_index_name)
        before["variety_id"] = str(event.canonical_variety_id)
        before["effective_to"] = date - pd.Timedelta(days=1)
        before["identity_source"] = "official_index_metadata_event"
        after = original.copy()
        after["index_code"] = str(event.new_index_code)
        after["index_name"] = str(event.new_index_name)
        after["variety_id"] = str(event.canonical_variety_id)
        after["effective_from"] = date
        after["identity_source"] = "official_index_metadata_event"
        if str(event.requires_roc_reset).strip().lower() == "true":
            after["warmup_from"] = date
        result = pd.concat([result, before.to_frame().T, after.to_frame().T], ignore_index=True)
    result = result.sort_values(["variety_id", "effective_from", "ts_code"]).reset_index(drop=True)
    _validate_non_overlapping_product_intervals(result)
    return result, values.drop(columns=["event_date_parsed"])


def apply_variety_aliases(intervals: pd.DataFrame, aliases: pd.DataFrame) -> pd.DataFrame:
    required = {"alias_index_code", "canonical_variety_id", "evidence_url"}
    missing = required - set(aliases.columns)
    if missing:
        raise ValueError(f"index aliases missing columns: {sorted(missing)}")
    if aliases["alias_index_code"].duplicated().any():
        raise ValueError("index aliases contain duplicate alias_index_code")
    mapping = aliases.set_index("alias_index_code")["canonical_variety_id"]
    result = intervals.copy()
    result["variety_id"] = result["index_code"].map(mapping).where(
        result["index_code"].map(mapping).notna(), result["variety_id"]
    )
    return result.sort_values(["variety_id", "effective_from", "ts_code"]).reset_index(drop=True)


def build_representative_lifecycle(intervals: pd.DataFrame) -> pd.DataFrame:
    if intervals.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for variety_id, group in intervals.groupby("variety_id"):
        boundaries = set(group["effective_from"].map(pd.Timestamp))
        boundaries.update(
            pd.Timestamp(value) + pd.Timedelta(days=1)
            for value in group["effective_to"].dropna()
        )
        ordered_boundaries = sorted(boundaries)
        current_rep: tuple[str, str, str] | None = None
        current_product: str | None = None
        current_row: dict[str, Any] | None = None
        for i, start in enumerate(ordered_boundaries):
            active = group[
                (group["effective_from"] <= start)
                & (group["effective_to"].isna() | (group["effective_to"] >= start))
            ].sort_values(["product_list_date", "ts_code"])
            if active.empty:
                representative = None
                current_product = None
            elif current_product is not None and active["ts_code"].eq(current_product).any():
                representative = active[active["ts_code"].eq(current_product)].iloc[0]
            else:
                representative = active.iloc[0]
                current_product = str(representative["ts_code"])
            representative_code = None if representative is None else str(representative["ts_code"])
            representative_key = (
                None
                if representative is None
                else (
                    representative_code,
                    str(representative["index_code"]),
                    str(representative["index_name"]),
                )
            )
            segment_end = (
                ordered_boundaries[i + 1] - pd.Timedelta(days=1)
                if i + 1 < len(ordered_boundaries)
                else group["effective_to"].max()
            )
            if representative_key == current_rep and current_row is not None:
                current_row["representative_to"] = segment_end
                continue
            if current_row is not None:
                rows.append(current_row)
                current_row = None
            current_rep = representative_key
            if representative is not None:
                current_row = {
                    "schema_version": REPRESENTATIVE_SCHEMA_VERSION,
                    "variety_id": str(variety_id),
                    "index_code": str(representative["index_code"]),
                    "index_name": str(representative["index_name"]),
                    "representative_ts_code": representative_code,
                    "representative_symbol": str(representative["symbol"]),
                    "representative_name": str(representative["product_name"]),
                    "representative_from": start,
                    "representative_to": segment_end,
                    "selection_rule": "earliest_listed_then_ts_code_successor_on_termination",
                    "identity_confidence": str(representative["identity_confidence"]),
                }
        if current_row is not None:
            rows.append(current_row)
    return pd.DataFrame(rows).sort_values(
        ["variety_id", "representative_from", "representative_ts_code"]
    ).reset_index(drop=True)


def index_change_report(intervals: pd.DataFrame) -> pd.DataFrame:
    if intervals.empty:
        return pd.DataFrame()
    counts = intervals.groupby("ts_code")["variety_id"].nunique()
    changed = set(counts[counts > 1].index.astype(str))
    return intervals[intervals["ts_code"].isin(changed)].sort_values(
        ["ts_code", "effective_from"]
    ).reset_index(drop=True)


def build_eligible_variety_counts(
    intervals: pd.DataFrame,
    lifecycle: pd.DataFrame,
    valid_dates_by_symbol: dict[str, pd.DatetimeIndex],
    *,
    warmup_observations: int = 41,
    calendar: pd.DatetimeIndex | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    memberships: dict[pd.Timestamp, set[str]] = (
        {pd.Timestamp(date): set() for date in pd.DatetimeIndex(calendar)}
        if calendar is not None
        else {}
    )
    eligibility_rows: list[dict[str, Any]] = []
    for representative in lifecycle.itertuples(index=False):
        identity = intervals[
            intervals["ts_code"].eq(representative.representative_ts_code)
            & intervals["index_code"].eq(representative.index_code)
            & (intervals["effective_from"] <= representative.representative_from)
            & (
                intervals["effective_to"].isna()
                | (intervals["effective_to"] >= representative.representative_from)
            )
        ]
        if len(identity) != 1:
            raise ValueError(
                "representative lifecycle does not map to exactly one identity interval: "
                f"{representative.index_code} {representative.representative_ts_code}"
            )
        identity_row = identity.iloc[0]
        valid_dates = pd.DatetimeIndex(
            valid_dates_by_symbol.get(str(representative.representative_symbol), pd.DatetimeIndex([]))
        ).sort_values()
        identity_dates = valid_dates[valid_dates >= pd.Timestamp(identity_row["warmup_from"])]
        if not pd.isna(identity_row["effective_to"]):
            identity_dates = identity_dates[
                identity_dates <= pd.Timestamp(identity_row["effective_to"])
            ]
        warmup_date = (
            identity_dates[warmup_observations - 1]
            if len(identity_dates) >= warmup_observations
            else pd.NaT
        )
        eligible_dates = pd.DatetimeIndex([])
        if not pd.isna(warmup_date):
            eligible_dates = identity_dates[
                identity_dates >= max(pd.Timestamp(warmup_date), pd.Timestamp(representative.representative_from))
            ]
            if not pd.isna(representative.representative_to):
                eligible_dates = eligible_dates[
                    eligible_dates <= pd.Timestamp(representative.representative_to)
                ]
            for date in eligible_dates:
                memberships.setdefault(pd.Timestamp(date), set()).add(str(representative.variety_id))
        eligibility_rows.append(
            {
                "variety_id": str(representative.variety_id),
                "index_code": str(representative.index_code),
                "representative_ts_code": str(representative.representative_ts_code),
                "representative_symbol": str(representative.representative_symbol),
                "representative_from": representative.representative_from,
                "representative_to": representative.representative_to,
                "identity_from": identity_row["effective_from"],
                "identity_to": identity_row["effective_to"],
                "warmup_from": identity_row["warmup_from"],
                "valid_observations_in_identity": len(identity_dates),
                "warmup_date": warmup_date,
                "eligible_observations_in_segment": len(eligible_dates),
            }
        )
    timeline = pd.DataFrame(
        [
            {"date": date, "eligible_variety_count": len(index_codes)}
            for date, index_codes in sorted(memberships.items())
        ]
    )
    return timeline, pd.DataFrame(eligibility_rows)