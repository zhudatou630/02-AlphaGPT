#!/usr/bin/env python3
"""Audit daily-data and termination coverage without computing any returns."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from alpha_etf.data.expanded_universe_audit import build_eligible_variety_counts  # noqa: E402


SCHEMA_VERSION = "expanded-etf-phase1-coverage-audit-v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase1-dir",
        type=Path,
        default=ROOT / "data" / "processed" / "expanded_etf_audit" / "phase1",
    )
    parser.add_argument(
        "--ended-history-dir",
        type=Path,
        default=ROOT / "data" / "processed" / "expanded_etf_audit" / "ended_history",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=ROOT / "data" / "processed" / "expanded_etf_audit" / "sources",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tdx_rows(path: Path) -> pd.DataFrame:
    with path.open(encoding="utf-8") as handle:
        return pd.DataFrame(json.loads(line) for line in handle if line.strip())


def _valid_tushare_dates(daily: pd.DataFrame) -> dict[str, pd.DatetimeIndex]:
    numeric = ["open", "high", "low", "close", "vol", "amount"]
    values = daily.copy()
    for column in numeric:
        values[column] = pd.to_numeric(values[column], errors="coerce")
    valid = values[numeric].notna().all(axis=1) & values[numeric].gt(0).all(axis=1)
    values = values[valid].copy()
    values["date"] = pd.to_datetime(values["trade_date"], format="%Y%m%d")
    return {
        str(code)[:6]: pd.DatetimeIndex(group["date"].sort_values().unique())
        for code, group in values.groupby("ts_code")
    }


def _termination_audit(master: pd.DataFrame, daily: pd.DataFrame, nav: pd.DataFrame) -> pd.DataFrame:
    ended = master[
        master["scope_decision"].eq("include_domestic_passive_equity")
        & master["status"].eq("D")
    ][["ts_code", "symbol", "name", "due_date", "list_date", "delist_date"]].copy()
    daily_dates = daily.groupby("ts_code")["trade_date"].agg(["min", "max", "count"]).reset_index()
    daily_dates.columns = ["ts_code", "first_daily_date", "last_daily_date", "daily_rows"]
    nav_values = nav.copy()
    nav_values["nav_date_value"] = pd.to_datetime(nav_values["nav_date"], format="%Y%m%d")
    latest = nav_values.sort_values("nav_date_value").groupby("ts_code").tail(1)
    latest = latest[["ts_code", "nav_date", "unit_nav", "accum_nav", "adj_nav"]].rename(
        columns={"nav_date": "last_nav_date"}
    )
    result = ended.merge(daily_dates, on="ts_code", how="left").merge(
        latest, on="ts_code", how="left"
    )
    for column in ("due_date", "delist_date", "last_daily_date", "last_nav_date"):
        result[f"{column}_parsed"] = pd.to_datetime(
            result[column], format="%Y%m%d", errors="coerce"
        )
    result["last_market_to_due_days"] = (
        result["due_date_parsed"] - result["last_daily_date_parsed"]
    ).dt.days
    result["last_nav_to_due_days"] = (
        result["due_date_parsed"] - result["last_nav_date_parsed"]
    ).dt.days
    result["due_to_delist_days"] = (
        result["delist_date_parsed"] - result["due_date_parsed"]
    ).dt.days
    result["last_nav_minus_last_market_days"] = (
        result["last_nav_date_parsed"] - result["last_daily_date_parsed"]
    ).dt.days
    result["termination_coverage_status"] = "daily_and_nav_present_cashflow_unverified"
    result.loc[result["daily_rows"].isna(), "termination_coverage_status"] = "missing_daily"
    result.loc[result["last_nav_date"].isna(), "termination_coverage_status"] = "missing_nav"
    result["date_sequence_anomaly"] = (
        result["due_date_parsed"].notna()
        & result["delist_date_parsed"].notna()
        & (result["due_date_parsed"] > result["delist_date_parsed"])
    )
    return result.drop(
        columns=[column for column in result.columns if column.endswith("_parsed")]
    ).sort_values(["termination_coverage_status", "due_date", "ts_code"])


def _report(summary: dict[str, object]) -> str:
    return "\n".join(
        [
            "# 扩大ETF池第一阶段行情覆盖审计",
            "",
            "> 本报告只包含数据可得性和有效品种数量，不包含价格收益、公式信号或策略结果。",
            "",
            "## 日线覆盖",
            "",
            f"- 目标产品：{summary['target_products']:,} 只。",
            f"- TDX有效产品：{summary['tdx_covered_products']:,} 只。",
            f"- Tushare补充退市产品：{summary['tushare_fallback_products']:,} 只。",
            f"- 无任何有效日线：{summary['missing_daily_products']:,} 只。",
            f"- 达到41个有效观察：{summary['products_with_41_observations']:,} 只。",
            "",
            "## 终止产品",
            "",
            f"- 已终止目标产品：{summary['ended_products']:,} 只。",
            f"- 有日线：{summary['ended_daily_covered']:,} 只。",
            f"- 有净值：{summary['ended_nav_covered']:,} 只。",
            f"- 日期顺序异常：{summary['termination_date_sequence_anomalies']:,} 只。",
            f"- 清算公告已扫描：{summary['terminal_cashflow_scan_successes']:,}/{summary['ended_representative_products']:,} 只退市代表。",
            f"- 现金终止路径已闭合：{summary['resolved_terminal_cashflow_products']:,} 只。",
            f"- 非现金合并或转型：{summary['non_cash_lifecycle_products']:,} 只。",
            f"- 终止路径仍未闭合：{summary['unresolved_terminal_products']:,} 只。",
            f"- 官方原件核验现金流：{summary['verified_terminal_cashflow_products']:,} 只。",
            f"- 官方原件核验非现金事件：{summary['verified_non_cash_lifecycle_products']:,} 只。",
            f"- 尚未扫描：{summary['terminal_cashflow_unscanned_products']:,} 只。",
            "- `due_date`、最后交易日、最后净值日和退市日语义并不统一；净值覆盖不能替代清算款到账公告。",
            "",
            "## 暂定历史横截面",
            "",
            f"- 暂可识别指数品种：{summary['provisional_index_varieties']:,} 个。",
            f"- 首次达到10个有效品种：{summary['first_date_at_least_10'] or '未达到'}。",
            f"- 此后持续不少于10个有效品种：{summary['first_stable_date_at_least_10'] or '未达到'}。",
            f"- 覆盖时间线末日：{summary['timeline_last_date'] or '无'}。",
            "- 上述日期基于尚未完成历史身份核验的临时区间，只用于选择后续审计范围，不得冻结为回测起点。",
            "",
        ]
    )


def main() -> None:
    args = _parse_args()
    master = pd.read_csv(args.phase1_dir / "product_master.csv", dtype=str)
    intervals = pd.read_csv(
        args.phase1_dir / "identity_intervals.csv",
        parse_dates=["effective_from", "effective_to"],
    )
    lifecycle = pd.read_csv(
        args.phase1_dir / "representative_lifecycle.csv",
        parse_dates=["representative_from", "representative_to"],
    )
    tdx = _tdx_rows(args.phase1_dir / "tdx_coverage.jsonl")
    ended_daily = pd.read_parquet(args.ended_history_dir / "tushare_ended_fund_daily.parquet")
    ended_nav = pd.read_parquet(args.ended_history_dir / "tushare_ended_fund_nav.parquet")
    trade_calendar = pd.read_parquet(args.source_dir / "tushare_sse_open_calendar.parquet")
    calendar = pd.DatetimeIndex(
        pd.to_datetime(trade_calendar["cal_date"], format="%Y%m%d").sort_values().unique()
    )
    target = master[master["scope_decision"].eq("include_domestic_passive_equity")].copy()
    target_symbols = set(target["symbol"].astype(str))
    dates_by_symbol: dict[str, pd.DatetimeIndex] = {}
    coverage_rows: list[dict[str, object]] = []
    tushare_dates = _valid_tushare_dates(ended_daily)
    for row in tdx.itertuples(index=False):
        symbol = str(row.symbol)
        if symbol not in target_symbols:
            continue
        tdx_dates = (
            pd.DatetimeIndex(pd.to_datetime(row.valid_dates))
            if isinstance(row.valid_dates, list) and row.valid_dates
            else pd.DatetimeIndex([])
        )
        source = "tdx"
        dates = tdx_dates
        if dates.empty and symbol in tushare_dates:
            source = "tushare_ended_fallback"
            dates = tushare_dates[symbol]
        dates_by_symbol[symbol] = dates
        coverage_rows.append(
            {
                "symbol": symbol,
                "source": source if len(dates) else "missing",
                "valid_observations": len(dates),
                "first_valid_date": dates.min() if len(dates) else pd.NaT,
                "last_valid_date": dates.max() if len(dates) else pd.NaT,
                "observation_41_date": dates[40] if len(dates) >= 41 else pd.NaT,
                "tdx_daily_status": row.daily_status,
                "tdx_event_status": row.event_status,
                "tdx_event_count": row.event_count,
            }
        )
    coverage = pd.DataFrame(coverage_rows).sort_values("symbol")
    timeline, representative_eligibility = build_eligible_variety_counts(
        intervals, lifecycle, dates_by_symbol, calendar=calendar
    )
    termination = _termination_audit(master, ended_daily, ended_nav)
    first_at_least_10 = None
    first_stable_10 = None
    if not timeline.empty:
        qualifying = timeline[timeline["eligible_variety_count"] >= 10]
        if not qualifying.empty:
            first_at_least_10 = qualifying.iloc[0]["date"].date().isoformat()
        stable = timeline["eligible_variety_count"][::-1].cummin()[::-1] >= 10
        if stable.any():
            first_stable_10 = timeline.loc[stable, "date"].iloc[0].date().isoformat()
    summary = {
        "target_products": len(target),
        "tdx_covered_products": int(coverage["source"].eq("tdx").sum()),
        "tushare_fallback_products": int(
            coverage["source"].eq("tushare_ended_fallback").sum()
        ),
        "missing_daily_products": int(coverage["source"].eq("missing").sum()),
        "products_with_41_observations": int(coverage["valid_observations"].ge(41).sum()),
        "ended_products": len(termination),
        "ended_daily_covered": int(termination["daily_rows"].notna().sum()),
        "ended_nav_covered": int(termination["last_nav_date"].notna().sum()),
        "termination_date_sequence_anomalies": int(termination["date_sequence_anomaly"].sum()),
        "provisional_index_varieties": int(intervals["variety_id"].nunique()),
        "first_date_at_least_10": first_at_least_10,
        "first_stable_date_at_least_10": first_stable_10,
        "timeline_last_date": (
            timeline.iloc[-1]["date"].date().isoformat() if not timeline.empty else None
        ),
    }
    terminal_manifest_path = args.phase1_dir / "terminal_cashflow_manifest.json"
    terminal_candidates_path = args.phase1_dir / "terminal_cashflow_candidates.csv"
    terminal_coverage_path = args.phase1_dir / "terminal_cashflow_coverage.csv"
    missing_terminal_inputs = [
        path
        for path in (terminal_manifest_path, terminal_candidates_path, terminal_coverage_path)
        if not path.exists()
    ]
    if missing_terminal_inputs:
        raise RuntimeError(
            "termination cashflow audit is required before coverage analysis: "
            + ", ".join(str(path) for path in missing_terminal_inputs)
        )
    terminal_manifest = json.loads(terminal_manifest_path.read_text(encoding="utf-8"))
    terminal_coverage = pd.read_csv(terminal_coverage_path, dtype=str)
    summary.update(
        {
            "ended_representative_products": int(
                terminal_manifest["ended_representative_products"]
            ),
            "terminal_cashflow_scan_successes": int(terminal_manifest["scan_successes"]),
            "terminal_cashflow_unscanned_products": int(
                terminal_manifest.get(
                    "unscanned_products",
                    terminal_manifest["ended_representative_products"]
                    - terminal_manifest["scan_successes"],
                )
            ),
            "resolved_terminal_cashflow_products": int(
                terminal_manifest["cashflow_resolved_products"]
            ),
            "non_cash_lifecycle_products": int(
                terminal_manifest["non_cash_lifecycle_products"]
            ),
            "unresolved_terminal_products": int(
                terminal_coverage["cashflow_status"]
                .eq("missing_distribution_announcement")
                .sum()
            ),
            "verified_terminal_cashflow_products": int(
                terminal_manifest.get("official_cashflow_verified_products", 0)
            ),
            "verified_non_cash_lifecycle_products": int(
                terminal_manifest.get("official_lifecycle_event_products", 0)
            ),
        }
    )
    outputs = {
        "daily_coverage": args.phase1_dir / "daily_coverage.csv",
        "termination_coverage": args.phase1_dir / "termination_coverage.csv",
        "representative_eligibility": args.phase1_dir / "representative_eligibility.csv",
        "eligible_variety_counts": args.phase1_dir / "eligible_variety_counts.csv",
    }
    coverage.to_csv(outputs["daily_coverage"], index=False)
    termination.to_csv(outputs["termination_coverage"], index=False)
    representative_eligibility.to_csv(outputs["representative_eligibility"], index=False)
    timeline.to_csv(outputs["eligible_variety_counts"], index=False)
    report_path = args.phase1_dir / "coverage_report.md"
    report_path.write_text(_report(summary), encoding="utf-8")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "interpretation": "phase_one_coverage_only_no_returns_or_strategy_outputs",
        "summary": summary,
        "inputs": {
            "tdx_coverage": _sha256(args.phase1_dir / "tdx_coverage.jsonl"),
            "ended_history_manifest": _sha256(
                args.ended_history_dir / "ended_history_manifest.json"
            ),
            "identity_manifest": _sha256(args.phase1_dir / "audit_manifest.json"),
            "trade_calendar": _sha256(args.source_dir / "tushare_sse_open_calendar.parquet"),
            "terminal_cashflow_manifest": _sha256(terminal_manifest_path),
            "terminal_cashflow_candidates": _sha256(terminal_candidates_path),
            "terminal_cashflow_coverage": _sha256(terminal_coverage_path),
        },
        "outputs": {
            name: {"file": path.name, "sha256": _sha256(path)}
            for name, path in outputs.items()
        },
        "report": {"file": report_path.name, "sha256": _sha256(report_path)},
    }
    manifest_path = args.phase1_dir / "coverage_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()