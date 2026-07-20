#!/usr/bin/env python3
"""Audit large raw ETF price gaps and separate market moves from unit events."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "expanded-etf-price-jump-audit-v1"
RAW_GAP_THRESHOLD = 0.20
MARKET_LIMIT = 0.20
PRICE_TICK = 0.001


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase1-dir",
        type=Path,
        default=ROOT / "data" / "processed" / "expanded_etf_audit" / "phase1",
    )
    parser.add_argument(
        "--tdx-dir",
        type=Path,
        default=ROOT
        / "data"
        / "processed"
        / "expanded_etf_audit"
        / "price_audit"
        / "tdx",
    )
    parser.add_argument(
        "--ended-history-dir",
        type=Path,
        default=ROOT
        / "data"
        / "processed"
        / "expanded_etf_audit"
        / "ended_history",
    )
    parser.add_argument("--require-cross-source", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_prices(args: argparse.Namespace) -> pd.DataFrame:
    tdx = pd.read_json(args.tdx_dir / "tdx_daily_bfq.jsonl", lines=True)
    tdx["symbol"] = tdx["symbol"].astype(str).str.zfill(6)
    tdx["date"] = pd.to_datetime(tdx["date"])
    tdx["source"] = "tdx_bfq"

    coverage = pd.read_csv(args.phase1_dir / "daily_coverage.csv", dtype={"symbol": str})
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
    ended["source"] = "tushare_ended_fallback"

    columns = [
        "symbol",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "source",
    ]
    prices = pd.concat([tdx[columns], ended[columns]], ignore_index=True)
    numeric = ["open", "high", "low", "close", "volume", "amount"]
    prices[numeric] = prices[numeric].apply(pd.to_numeric, errors="coerce")
    return prices.sort_values(["symbol", "date"]).reset_index(drop=True)


def _load_events(path: Path) -> pd.DataFrame:
    events = pd.read_json(path, lines=True)
    events["symbol"] = events["symbol"].astype(str).str.zfill(6)
    events["date"] = pd.to_datetime(events["date"])
    return events.sort_values(["symbol", "date", "category_code"])


def _event_reference(previous_close: float, events: pd.DataFrame) -> float:
    reference = float(previous_close)
    # Unit changes take effect before same-day cash distributions.
    ordered = events.assign(
        event_order=events["category_code"].map({11: 0, 12: 0, 1: 1}).fillna(2)
    ).sort_values(["date", "event_order"])
    for event in ordered.itertuples(index=False):
        category = int(event.category_code)
        if category in (11, 12):
            share_step = float(event.c3)
            if not np.isfinite(share_step) or share_step <= 0:
                raise ValueError(
                    f"invalid share step for {event.symbol} on {event.date}: {share_step}"
                )
            reference /= share_step
        elif category == 1:
            share_step = (10.0 + float(event.c3) + float(event.c4)) / 10.0
            cash_per_share = (float(event.c1) - float(event.c4) * float(event.c2)) / 10.0
            reference = (reference - cash_per_share) / share_step
        else:
            raise ValueError(f"unsupported TDX event category: {category}")
    return reference


def _classify(row: pd.Series) -> str:
    adjusted = max(abs(row["adjusted_open_gap"]), abs(row["adjusted_close_jump"]))
    if adjusted > row["limit_rounding_bound"]:
        return "unexplained_large_gap"
    categories = set(str(row["event_categories"]).split("|"))
    if "11" in categories or "12" in categories:
        return "explained_share_adjustment"
    if "1" in categories:
        return "explained_cash_distribution"
    return "market_move_within_20pct_limit_rounding"


def _report(summary: dict[str, object]) -> str:
    return "\n".join(
        [
            "# 扩大ETF池大幅跳空审计",
            "",
            "> 本报告只审计价格质量，不包含策略收益或公式输出。",
            "",
            "## 覆盖",
            "",
            f"- 产品：{summary['products']:,} 只。",
            f"- 日线：{summary['price_rows']:,} 行。",
            f"- 重复产品日期：{summary['duplicate_rows']:,} 行。",
            f"- OHLC结构错误：{summary['invalid_ohlc_rows']:,} 行。",
            "",
            "## 隔夜跳空",
            "",
            f"- 原始开盘或收盘相对前收盘超过20%：{summary['raw_gap_gt_20pct_rows']:,} 行。",
            f"- 其中份额折算：{summary['share_adjustment_rows']:,} 行。",
            f"- 其中现金分配：{summary['cash_distribution_rows']:,} 行。",
            f"- 其中20%涨跌幅边界及取整：{summary['market_limit_rows']:,} 行。",
            f"- 事件调整后仍超过逐行20%报价取整上界：{summary['unexplained_large_gap_rows']:,} 行。",
            f"- 原始超过30%且无份额事件：{summary['raw_gap_gt_30pct_without_share_event']:,} 行。",
            f"- 最大原始绝对跳变：{summary['max_abs_raw_gap']:.2%}。",
            f"- 最大事件调整后绝对跳变：{summary['max_abs_adjusted_gap']:.2%}。",
            "",
            "## 交叉验证",
            "",
            f"- 原始跳变超过100%的事件：{summary['extreme_gap_gt_100pct_rows']:,} 行。",
            f"- 全部候选获得Tushare调整后参考价及同日OHLC：{summary['cross_source_verified_rows']:,} 行。",
            f"- 同时获得前一交易日原始收盘：{summary['full_previous_close_verified_rows']:,} 行。",
            f"- 仅能核验调整后参考价：{summary['adjusted_reference_only_rows']:,} 行。",
            f"- TDX与Tushare价格不一致：{summary['cross_source_mismatch_rows']:,} 行。",
            f"- 按Tushare复权前收仍越过涨跌幅取整上界：{summary['cross_source_adjusted_limit_breach_rows']:,} 行。",
            f"- 退市补源同源复查、不能算独立核验：{summary['same_source_unverified_rows']:,} 行。",
            "",
            "结论：不存在事件调整后仍无法解释的大幅隔夜跳空。原始不复权价格不能直接用于ROC计算；",
            "份额折算和现金分配必须先进入事件复权链。",
            "",
        ]
    )


def main() -> None:
    args = _parse_args()
    prices = _load_prices(args)
    event_path = args.tdx_dir / "tdx_gbbq_events.jsonl"
    events = _load_events(event_path)

    duplicate_rows = int(prices.duplicated(["symbol", "date"], keep=False).sum())
    invalid_ohlc = (
        prices[["open", "high", "low", "close"]].isna().any(axis=1)
        | prices[["open", "high", "low", "close"]].le(0).any(axis=1)
        | prices["high"].lt(prices[["open", "close"]].max(axis=1))
        | prices["low"].gt(prices[["open", "close"]].min(axis=1))
    )

    grouped = prices.groupby("symbol", sort=False)
    prices["previous_date"] = grouped["date"].shift()
    prices["previous_close"] = grouped["close"].shift()
    prices["raw_open_gap"] = prices["open"] / prices["previous_close"] - 1.0
    prices["raw_close_jump"] = prices["close"] / prices["previous_close"] - 1.0
    raw_gap = prices[["raw_open_gap", "raw_close_jump"]].abs().max(axis=1)
    candidates = prices[raw_gap.gt(RAW_GAP_THRESHOLD)].copy()

    rows: list[dict[str, object]] = []
    for row in candidates.itertuples(index=False):
        mapped = events[
            events["symbol"].eq(row.symbol)
            & events["date"].gt(row.previous_date)
            & events["date"].le(row.date)
        ]
        reference = _event_reference(row.previous_close, mapped)
        categories = "|".join(str(int(value)) for value in mapped["category_code"])
        rows.append(
            {
                **row._asdict(),
                "event_categories": categories,
                "event_reference_close": reference,
                "adjusted_open_gap": row.open / reference - 1.0,
                "adjusted_close_jump": row.close / reference - 1.0,
                "limit_rounding_bound": MARKET_LIMIT
                + (PRICE_TICK / 2.0) / reference,
            }
        )
    audited = pd.DataFrame(rows)
    audited["classification"] = audited.apply(_classify, axis=1)
    audited["max_abs_raw_gap"] = audited[["raw_open_gap", "raw_close_jump"]].abs().max(axis=1)
    audited["max_abs_adjusted_gap"] = audited[
        ["adjusted_open_gap", "adjusted_close_jump"]
    ].abs().max(axis=1)

    output_path = args.phase1_dir / "price_jump_audit.csv"
    audited.to_csv(output_path, index=False)
    cross_path = args.phase1_dir / "price_jump_cross_source_check.csv"
    cross_manifest_path = args.phase1_dir / "price_jump_cross_source_manifest.json"
    has_cross_evidence = cross_path.exists() and cross_manifest_path.exists()
    if args.require_cross_source and not has_cross_evidence:
        raise RuntimeError("complete cross-source CSV and manifest are required")
    if has_cross_evidence:
        cross = pd.read_csv(cross_path, dtype={"symbol": str})
        cross["date"] = pd.to_datetime(cross["date"])
        if cross.duplicated(["symbol", "date"]).any():
            raise RuntimeError("cross-source evidence has duplicate symbol/date rows")
        candidate_keys = set(zip(audited["symbol"], audited["date"], strict=True))
        cross_keys = set(zip(cross["symbol"], cross["date"], strict=True))
        if candidate_keys != cross_keys:
            raise RuntimeError("cross-source evidence keys do not match current candidates")
        cross_manifest = json.loads(cross_manifest_path.read_text(encoding="utf-8"))
        if cross_manifest["inputs"]["price_jump_audit"] != _sha256(output_path):
            raise RuntimeError("cross-source evidence was built from a different audit CSV")
        if cross_manifest["inputs"]["product_master"] != _sha256(
            args.phase1_dir / "product_master.csv"
        ):
            raise RuntimeError("cross-source evidence used a different product master")
        if cross_manifest["output"]["sha256"] != _sha256(cross_path):
            raise RuntimeError("cross-source CSV hash does not match its manifest")
        cross_full = cross["status"].eq("ok_cross_source_full")
        cross_reference_only = cross["status"].eq(
            "ok_cross_source_adjusted_reference"
        )
        cross_ok = cross_full | cross_reference_only
        same_source = cross["status"].eq("same_source_not_independent")
        expected_source = audited.set_index(["symbol", "date"])["source"]
        cross_source = pd.MultiIndex.from_frame(cross[["symbol", "date"]]).map(
            expected_source
        )
        invalid_source_status = (
            pd.Series(cross_source, index=cross.index).eq("tdx_bfq") & ~cross_ok
        ) | (
            pd.Series(cross_source, index=cross.index)
            .ne("tdx_bfq")
            & ~same_source
        )
        if invalid_source_status.any():
            raise RuntimeError("cross-source status does not match the primary data source")
        unexpected_status = ~(cross_ok | same_source)
        if unexpected_status.any():
            raise RuntimeError("cross-source evidence contains failed or missing queries")
        cross_mismatch = cross_ok & pd.to_numeric(
            cross["max_primary_price_abs_diff"], errors="coerce"
        ).gt(1e-12)
        cross_limit_breach = cross_ok & cross["ts_adjusted_limit_breach"].fillna(
            False
        ).astype(bool)
    else:
        cross_ok = pd.Series(dtype=bool)
        cross_full = pd.Series(dtype=bool)
        cross_reference_only = pd.Series(dtype=bool)
        same_source = pd.Series(dtype=bool)
        cross_mismatch = pd.Series(dtype=bool)
        cross_limit_breach = pd.Series(dtype=bool)
    counts = audited["classification"].value_counts()
    share_event = audited["event_categories"].str.contains(r"(?:^|\|)1[12](?:\||$)")
    summary = {
        "products": int(prices["symbol"].nunique()),
        "price_rows": len(prices),
        "duplicate_rows": duplicate_rows,
        "invalid_ohlc_rows": int(invalid_ohlc.sum()),
        "raw_gap_gt_20pct_rows": len(audited),
        "share_adjustment_rows": int(counts.get("explained_share_adjustment", 0)),
        "cash_distribution_rows": int(counts.get("explained_cash_distribution", 0)),
        "market_limit_rows": int(
            counts.get("market_move_within_20pct_limit_rounding", 0)
        ),
        "unexplained_large_gap_rows": int(counts.get("unexplained_large_gap", 0)),
        "raw_gap_gt_30pct_without_share_event": int(
            (audited["max_abs_raw_gap"].gt(0.30) & ~share_event).sum()
        ),
        "max_abs_raw_gap": float(audited["max_abs_raw_gap"].max()),
        "max_abs_adjusted_gap": float(audited["max_abs_adjusted_gap"].max()),
        "extreme_gap_gt_100pct_rows": int(audited["max_abs_raw_gap"].gt(1.0).sum()),
        "cross_source_verified_rows": int(cross_ok.sum()),
        "full_previous_close_verified_rows": int(cross_full.sum()),
        "adjusted_reference_only_rows": int(cross_reference_only.sum()),
        "cross_source_mismatch_rows": int(cross_mismatch.sum()),
        "cross_source_adjusted_limit_breach_rows": int(cross_limit_breach.sum()),
        "same_source_unverified_rows": int(same_source.sum()),
    }
    report_path = args.phase1_dir / "price_jump_audit_report.md"
    report_path.write_text(_report(summary), encoding="utf-8")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "interpretation": "price_quality_only_no_returns_or_strategy_outputs",
        "thresholds": {
            "raw_gap": RAW_GAP_THRESHOLD,
            "market_limit": MARKET_LIMIT,
            "price_tick": PRICE_TICK,
            "rounding_bound": "market_limit + half_price_tick / event_reference_close",
        },
        "cross_source_required": args.require_cross_source,
        "summary": summary,
        "inputs": {
            "tdx_bfq": _sha256(args.tdx_dir / "tdx_daily_bfq.jsonl"),
            "tdx_events": _sha256(event_path),
            "ended_fund_daily": _sha256(
                args.ended_history_dir / "tushare_ended_fund_daily.parquet"
            ),
            "daily_coverage": _sha256(args.phase1_dir / "daily_coverage.csv"),
            **(
                {
                    "cross_source_check": _sha256(cross_path),
                    "cross_source_manifest": _sha256(cross_manifest_path),
                }
                if has_cross_evidence
                else {}
            ),
        },
        "outputs": {
            "audit": {"file": output_path.name, "sha256": _sha256(output_path)},
            "report": {"file": report_path.name, "sha256": _sha256(report_path)},
        },
    }
    manifest_path = args.phase1_dir / "price_jump_audit_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()