#!/usr/bin/env python3
"""Build governed event-adjusted OHLC for the expanded domestic ETF universe."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from alpha_etf.data.event_adjustment import (  # noqa: E402
    anomaly_rows,
    apply_event_adjustments,
    build_event_audit,
    quality_summary,
)


SCHEMA_VERSION = "expanded-etf-event-adjusted-daily-v1"
ADJUSTED_RETURN_LIMIT = 0.201


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    base = ROOT / "data" / "processed" / "expanded_etf_audit"
    parser.add_argument("--phase1-dir", type=Path, default=base / "phase1")
    parser.add_argument("--tdx-dir", type=Path, default=base / "price_audit" / "tdx")
    parser.add_argument(
        "--fallback-event-dir",
        type=Path,
        default=base / "price_audit" / "fallback_events",
    )
    parser.add_argument("--ended-history-dir", type=Path, default=base / "ended_history")
    parser.add_argument("--output-dir", type=Path, default=base / "event_adjusted")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_raw(args: argparse.Namespace) -> pd.DataFrame:
    coverage = pd.read_csv(args.phase1_dir / "daily_coverage.csv", dtype={"symbol": str})
    target_symbols = set(coverage["symbol"].astype(str))
    tdx = pd.read_json(args.tdx_dir / "tdx_daily_bfq.jsonl", lines=True)
    tdx["symbol"] = tdx["symbol"].astype(str).str.zfill(6)
    tdx = tdx[tdx["symbol"].isin(target_symbols)].copy()
    tdx["date"] = pd.to_datetime(tdx["date"])
    tdx["price_source"] = "tdx_bfq"
    tdx["flow_source"] = "tdx"

    fallback_symbols = set(
        coverage.loc[
            coverage["source"].eq("tushare_ended_fallback"), "symbol"
        ].astype(str)
    )
    ended = pd.read_parquet(
        args.ended_history_dir / "tushare_ended_fund_daily.parquet"
    )
    ended["symbol"] = ended["ts_code"].str[:6]
    ended = ended[ended["symbol"].isin(fallback_symbols)].copy()
    ended["date"] = pd.to_datetime(ended["trade_date"], format="%Y%m%d")
    ended = ended.rename(columns={"vol": "volume"})
    ended["amount"] = pd.to_numeric(ended["amount"]) * 1000.0
    ended["price_source"] = "tushare_ended_fallback"
    ended["flow_source"] = "tushare_fund_daily"

    columns = [
        "symbol",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "price_source",
        "flow_source",
    ]
    raw = pd.concat([tdx[columns], ended[columns]], ignore_index=True)
    numeric = ["open", "high", "low", "close", "volume", "amount"]
    raw[numeric] = raw[numeric].apply(pd.to_numeric, errors="coerce")
    raw["tradable"] = raw["volume"].gt(0) & raw["amount"].gt(0)
    master = (
        pd.read_csv(args.phase1_dir / "product_master.csv", dtype=str)
        .drop_duplicates("symbol")
        .set_index("symbol")
    )
    raw["name"] = raw["symbol"].map(master["name"])
    raw_symbols = set(raw["symbol"])
    if raw_symbols != target_symbols:
        raise RuntimeError(
            f"raw product set differs from coverage: "
            f"missing={sorted(target_symbols - raw_symbols)[:20]} "
            f"extra={sorted(raw_symbols - target_symbols)[:20]}"
        )
    return raw.sort_values(["symbol", "date"]).reset_index(drop=True)


def _load_events(args: argparse.Namespace) -> pd.DataFrame:
    paths = [
        args.tdx_dir / "tdx_gbbq_events.jsonl",
        args.fallback_event_dir / "tdx_gbbq_events.jsonl",
    ]
    frames = [pd.read_json(path, lines=True) for path in paths]
    events = pd.concat(frames, ignore_index=True)
    events["symbol"] = events["symbol"].astype(str).str.zfill(6)
    events["date"] = pd.to_datetime(events["date"])
    if events.duplicated(["symbol", "date", "category_code"]).any():
        raise RuntimeError("duplicate TDX adjustment events")
    coverage = pd.read_csv(args.phase1_dir / "daily_coverage.csv", dtype={"symbol": str})
    if not coverage["tdx_event_status"].eq("ok").all():
        raise RuntimeError("not all target products have successful TDX event queries")
    expected = coverage.set_index("symbol")["tdx_event_count"].astype(int).sort_index()
    extra_symbols = set(events["symbol"]) - set(expected.index)
    if extra_symbols:
        raise RuntimeError(f"event files contain products outside coverage: {sorted(extra_symbols)[:20]}")
    actual = events.groupby("symbol").size().reindex(expected.index, fill_value=0)
    mismatches = expected.ne(actual)
    if mismatches.any():
        examples = pd.DataFrame(
            {"expected": expected[mismatches], "actual": actual[mismatches]}
        ).head(20)
        raise RuntimeError(f"TDX event count mismatch by product:\n{examples}")
    return events.sort_values(["symbol", "date", "category_code"]).reset_index(drop=True)


def _report(summary: dict[str, object]) -> str:
    return "\n".join(
        [
            "# 扩大ETF池事件复权数据报告",
            "",
            "> 本产物只生成复权行情，不计算公式信号、策略收益、基准收益或PnL。",
            "",
            f"- 产品：{summary['products']:,} 只。",
            f"- 日线：{summary['daily_rows']:,} 行。",
            f"- TDX主源：{summary['tdx_products']:,} 只。",
            f"- Tushare退市补源：{summary['tushare_fallback_products']:,} 只。",
            f"- 事件总数：{summary['event_rows']:,} 条。",
            f"- 已应用事件：{summary['applied_events']:,} 条。",
            f"- 已应用份额折算：{summary['applied_share_events']:,} 条。",
            f"- 首行前事件：{summary['events_before_first_price']:,} 条。",
            f"- 末行后事件：{summary['events_after_last_price']:,} 条。",
            f"- 意外跳过事件：{summary['unexpected_event_skips']:,} 条。",
            f"- 事件复权后超过20.1%的收盘跳变：{summary['adjusted_return_breaches']:,} 行。",
            f"- 事件复权后最大绝对收盘跳变：{summary['max_abs_adjusted_close_return']:.4%}。",
            f"- 无效复权OHLC：{summary['invalid_adjusted_ohlc_rows']:,} 行。",
            f"- flow口径告警：{summary['flow_warning_rows']:,} 行。",
            "",
            "价格门禁与flow告警分离：本研究不使用成交量或成交额特征，但保留原始flow字段和告警。",
            "第二阶段的ROC、开盘执行和每日估值必须读取`event_qfq_*`字段，禁止读取原始OHLC或TDX自带QFQ。",
            "",
        ]
    )


def main() -> None:
    args = _parse_args()
    raw = _load_raw(args)
    events = _load_events(args)
    event_audit = build_event_audit(raw, events)
    adjusted = apply_event_adjustments(raw, event_audit)
    quality = quality_summary(adjusted, event_audit, threshold=ADJUSTED_RETURN_LIMIT)
    anomalies = anomaly_rows(adjusted, threshold=ADJUSTED_RETURN_LIMIT)

    applied = event_audit[event_audit["applied"].astype(bool)]
    supported = event_audit[event_audit["category_code"].isin([1, 11, 12])]
    unexpected = supported[
        ~supported["applied"].astype(bool)
        & ~supported["skip_reason"].isin(["no_previous_daily_row", "future_event"])
    ]
    price_gate_columns = [
        "adjusted_abs_return_gt_threshold",
        "invalid_adjusted_price_rows",
        "invalid_adjusted_high_rows",
        "invalid_adjusted_low_rows",
        "invalid_adjusted_volume_rows",
        "invalid_raw_amount_rows",
    ]
    price_failures = {column: int(quality[column].sum()) for column in price_gate_columns}
    price_failures["unexpected_event_skips"] = len(unexpected)
    price_failures = {key: value for key, value in price_failures.items() if value}
    if price_failures:
        raise RuntimeError(f"expanded event-adjusted price gate failed: {price_failures}")

    flow_warning_rows = int(
        quality["invalid_flow_implied_price_rows"].sum()
        + quality["flow_zero_mismatch_rows"].sum()
    )
    summary = {
        "products": int(adjusted["symbol"].nunique()),
        "daily_rows": len(adjusted),
        "tdx_products": int(
            adjusted.loc[adjusted["price_source"].eq("tdx_bfq"), "symbol"].nunique()
        ),
        "tushare_fallback_products": int(
            adjusted.loc[
                adjusted["price_source"].eq("tushare_ended_fallback"), "symbol"
            ].nunique()
        ),
        "event_rows": len(event_audit),
        "applied_events": len(applied),
        "applied_share_events": int(applied["category_code"].isin([11, 12]).sum()),
        "events_before_first_price": int(event_audit["skip_reason"].eq("no_previous_daily_row").sum()),
        "events_after_last_price": int(event_audit["skip_reason"].eq("future_event").sum()),
        "unexpected_event_skips": len(unexpected),
        "adjusted_return_breaches": int(
            quality["adjusted_abs_return_gt_threshold"].sum()
        ),
        "max_abs_adjusted_close_return": float(
            quality["max_abs_adjusted_return"].max()
        ),
        "invalid_adjusted_ohlc_rows": int(
            quality[
                [
                    "invalid_adjusted_price_rows",
                    "invalid_adjusted_high_rows",
                    "invalid_adjusted_low_rows",
                ]
            ].to_numpy().sum()
        ),
        "flow_warning_rows": flow_warning_rows,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "daily": args.output_dir / "etf_daily_event_adjusted.parquet",
        "event_audit": args.output_dir / "event_audit.csv",
        "quality": args.output_dir / "quality_summary.csv",
        "anomalies": args.output_dir / "adjusted_return_anomalies.csv",
        "report": args.output_dir / "report.md",
    }
    adjusted = adjusted.rename(
        columns={
            "open": "raw_open",
            "high": "raw_high",
            "low": "raw_low",
            "close": "raw_close",
        }
    )
    adjusted.to_parquet(outputs["daily"], index=False)
    event_audit.to_csv(outputs["event_audit"], index=False)
    quality.to_csv(outputs["quality"], index=False)
    anomalies.to_csv(outputs["anomalies"], index=False)
    outputs["report"].write_text(_report(summary), encoding="utf-8")

    price_jump_manifest = args.phase1_dir / "price_jump_audit_manifest.json"
    if not price_jump_manifest.exists():
        raise RuntimeError("price-jump audit manifest is required")
    jump = json.loads(price_jump_manifest.read_text(encoding="utf-8"))
    if jump["summary"]["unexplained_large_gap_rows"] != 0:
        raise RuntimeError("price-jump audit contains unexplained large gaps")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "interpretation": "event_adjusted_price_data_only_no_strategy_outputs",
        "price_adjustment": "multiplicative_event_qfq_category_1_11_12",
        "required_backtest_fields": [
            "event_qfq_open",
            "event_qfq_high",
            "event_qfq_low",
            "event_qfq_close",
        ],
        "summary": summary,
        "inputs": {
            "tdx_bfq": _sha256(args.tdx_dir / "tdx_daily_bfq.jsonl"),
            "tdx_primary_events": _sha256(
                args.tdx_dir / "tdx_gbbq_events.jsonl"
            ),
            "tdx_fallback_events": _sha256(
                args.fallback_event_dir / "tdx_gbbq_events.jsonl"
            ),
            "tushare_ended_daily": _sha256(
                args.ended_history_dir / "tushare_ended_fund_daily.parquet"
            ),
            "daily_coverage": _sha256(args.phase1_dir / "daily_coverage.csv"),
            "product_master": _sha256(args.phase1_dir / "product_master.csv"),
            "price_jump_manifest": _sha256(price_jump_manifest),
        },
        "outputs": {
            name: {"file": path.name, "sha256": _sha256(path)}
            for name, path in outputs.items()
        },
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()