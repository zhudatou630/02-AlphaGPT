#!/usr/bin/env python3
"""Build a reviewable inventory of historically listed domestic-equity ETFs."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd
import tushare as ts


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

REVIEW_PATTERNS = (
    "全球",
    "亚太",
    "跨境",
    "互联互通",
)

DOMESTIC_ALLOW_PATTERNS = ("MSCI中国A50互联互通",)

CURRENT_V2_CODES = {
    "510050.SH", "510300.SH", "510500.SH", "512100.SH", "159915.SZ",
    "588000.SH", "510880.SH", "512800.SH", "512880.SH", "512760.SH",
    "512660.SH", "512010.SH", "159928.SZ", "512400.SH", "515220.SH",
    "515790.SH", "515030.SH",
}

BENCHMARK_ALIASES = {
    "上海证券交易所50成份指数": "上证50指数",
    "上海证券交易所上证红利指数": "上证红利指数",
    "中证小盘500指数": "中证500指数",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/etf_market_inventory"),
    )
    parser.add_argument("--liquidity-days", type=int, default=20)
    parser.add_argument("--as-of", default=None, help="YYYYMMDD; defaults to latest open day")
    return parser.parse_args()


def normalize_benchmark(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = re.sub(r"[（(].*?人民币.*?[）)]", "", text)
    text = text.replace("指数P", "指数")
    text = re.sub(r"(?:收益率)?[×xX*]?100(?:\.0+)?%", "", text)
    text = re.sub(r"[\s（）()]+", "", text)
    text = text.strip("+；;,，")
    return text


def canonical_benchmark(value: object) -> str:
    normalized = normalize_benchmark(value)
    normalized = normalized.replace("价格指数", "指数")
    return BENCHMARK_ALIASES.get(normalized, normalized)


def market_state(row: pd.Series, as_of: str) -> str:
    list_date = "" if pd.isna(row.get("list_date")) else str(row["list_date"])
    delist_date = "" if pd.isna(row.get("delist_date")) else str(row["delist_date"])
    if delist_date and delist_date <= as_of:
        return "ended"
    if not list_date or list_date > as_of:
        return "not_yet_listed"
    return "active"


def classify(row: pd.Series) -> tuple[str, str]:
    text = f"{row.get('name', '')}|{row.get('benchmark', '')}".upper()
    if any(pattern.upper() in text for pattern in DOMESTIC_ALLOW_PATTERNS):
        return "candidate_domestic_equity", "explicit_domestic_allow"
    matches = [pattern for pattern in NON_DOMESTIC_PATTERNS if pattern.upper() in text]
    if matches:
        return "exclude_non_domestic", ";".join(matches)
    review = [pattern for pattern in REVIEW_PATTERNS if pattern.upper() in text]
    if review:
        return "review", ";".join(review)
    return "candidate_domestic_equity", ""


def fetch_basic(pro: object) -> pd.DataFrame:
    frames = [
        frame
        for status in ("L", "D")
        if not (frame := pro.fund_basic(market="E", status=status)).empty
    ]
    basic = pd.concat(frames, ignore_index=True).drop_duplicates("ts_code", keep="first")
    is_etf = basic["name"].fillna("").str.contains("ETF", case=False)
    return basic.loc[is_etf & basic["fund_type"].eq("股票型")].copy()


def trading_dates(pro: object, as_of: str | None, count: int) -> list[str]:
    end = as_of or pd.Timestamp.today().strftime("%Y%m%d")
    start = (pd.to_datetime(end) - pd.Timedelta(days=max(60, count * 4))).strftime("%Y%m%d")
    calendar = pro.trade_cal(exchange="SSE", start_date=start, end_date=end, is_open="1")
    dates = sorted(calendar["cal_date"].astype(str).tolist())
    if not dates:
        raise RuntimeError(f"No trading dates found through {end}")
    return dates[-count:]


def fetch_liquidity(pro: object, dates: list[str]) -> pd.DataFrame:
    frames = []
    for date in dates:
        daily = pro.fund_daily(trade_date=date)
        frames.append(daily[["ts_code", "trade_date", "amount", "vol", "close"]])
    values = pd.concat(frames, ignore_index=True)
    grouped = values.groupby("ts_code", as_index=False).agg(
        liquidity_observations=("trade_date", "count"),
        avg_amount_20d=("amount", "mean"),
        median_amount_20d=("amount", "median"),
        avg_volume_20d=("vol", "mean"),
        latest_close=("close", "last"),
    )
    return grouped


def fetch_latest_share(pro: object, dates: list[str]) -> pd.DataFrame:
    for date in reversed(dates):
        share = pro.fund_share(trade_date=date)
        if not share.empty:
            return share[["ts_code", "trade_date", "fd_share"]].rename(
                columns={"trade_date": "share_date", "fd_share": "latest_share"}
            )
    return pd.DataFrame(columns=["ts_code", "share_date", "latest_share"])


def build_report(inventory: pd.DataFrame, dates: list[str]) -> str:
    candidates = inventory[inventory["scope_class"] == "candidate_domestic_equity"]
    current = candidates[candidates["market_state_as_of"] == "active"]
    ended = candidates[candidates["market_state_as_of"] == "ended"]
    pending = candidates[candidates["market_state_as_of"] == "not_yet_listed"]
    groups = (
        candidates.groupby("benchmark_group", dropna=False)
        .agg(
            product_count=("ts_code", "count"),
            current_count=("market_state_as_of", lambda x: int((x == "active").sum())),
            ended_count=("market_state_as_of", lambda x: int((x == "ended").sum())),
        )
        .sort_values(["product_count", "current_count"], ascending=False)
    )
    duplicated = groups[groups["product_count"] > 1]
    top = groups.head(30).reset_index()
    top_lines = [
        f"| {row.benchmark_group} | {row.product_count} | {row.current_count} | {row.ended_count} |"
        for row in top.itertuples(index=False)
    ]
    return "\n".join(
        [
            "# 境内股票 ETF 市场初步梳理",
            "",
            f"数据截至：{dates[-1]}；近期流动性窗口：{dates[0]} 至 {dates[-1]}。",
            "",
            "## 口径",
            "",
            "- 母样本为 Tushare 交易所基金中名称含 ETF 且基金类型为股票型的全部存续及已终止产品。",
            "- 境内股票 ETF 候选通过名称和业绩比较基准排除 QDII、港股通、沪港深及其他明确境外暴露。",
            "- 名称规则不能替代成分股核验；`review` 和最终入池产品必须人工复核。",
            "- 同类产品按规范化后的业绩比较基准分组，不自动决定保留哪只产品。",
            "- Tushare `amount` 原始单位保持不变，本报告不将其直接换算为人民币金额。",
            "",
            "## 总览",
            "",
            f"- 历史股票型 ETF 母样本：{len(inventory):,} 只。",
            f"- 境内股票 ETF 候选：{len(candidates):,} 只，其中已上市存续 {len(current):,} 只、已终止 {len(ended):,} 只、待上市或上市日缺失 {len(pending):,} 只。",
            f"- 明确非境内股票暴露：{int((inventory.scope_class == 'exclude_non_domestic').sum()):,} 只。",
            f"- 边界待人工复核：{int((inventory.scope_class == 'review').sum()):,} 只。",
            f"- 境内候选跟踪基准组：{len(groups):,} 组，其中存在多个产品的基准组 {len(duplicated):,} 组。",
            "",
            "## 产品最多的跟踪基准",
            "",
            "| 跟踪基准 | 历史产品数 | 当前存续 | 已终止 |",
            "|---|---:|---:|---:|",
            *top_lines,
            "",
            "## 人工选池建议顺序",
            "",
            "1. 在 `benchmark_groups.csv` 中先选希望覆盖的指数或暴露。",
            "2. 在 `domestic_equity_candidates.csv` 中比较同组产品的上市历史、状态、近期成交额和份额。",
            "3. 对名称或基准接近但分组不同的产品核对指数编制方案，避免仅按名称合并。",
            "4. 对最终候选核查完整历史行情、事件数据和停牌覆盖后，再形成研究池。",
            "",
            "## 文件",
            "",
            "- `all_stock_etfs.csv`：历史股票型 ETF 母样本及分类理由。",
            "- `domestic_equity_candidates.csv`：境内股票 ETF 候选，供人工选池。",
            "- `active_selection_view.csv`：当前已上市产品的精简人工选池视图。",
            "- `excluded_and_review.csv`：明确排除及边界待复核产品。",
            "- `benchmark_groups.csv`：按跟踪基准汇总的同类产品组。",
        ]
    ) + "\n"


def main() -> None:
    args = parse_args()
    if not os.environ.get("TUSHARE_TOKEN"):
        raise RuntimeError("TUSHARE_TOKEN is not set; source .venv/bin/activate first")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pro = ts.pro_api()
    basic = fetch_basic(pro)
    dates = trading_dates(pro, args.as_of, args.liquidity_days)
    liquidity = fetch_liquidity(pro, dates)
    share = fetch_latest_share(pro, dates)

    classifications = basic.apply(classify, axis=1, result_type="expand")
    classifications.columns = ["scope_class", "scope_reason"]
    basic = pd.concat([basic, classifications], axis=1)
    basic["benchmark_group_raw"] = basic["benchmark"].map(normalize_benchmark)
    basic["benchmark_group"] = basic["benchmark"].map(canonical_benchmark)
    basic["market_state_as_of"] = basic.apply(market_state, axis=1, as_of=dates[-1])
    inventory = basic.merge(liquidity, on="ts_code", how="left").merge(share, on="ts_code", how="left")
    inventory["in_current_v2_pool"] = inventory["ts_code"].isin(CURRENT_V2_CODES)
    inventory["liquidity_rank_in_group"] = (
        inventory.groupby("benchmark_group")["avg_amount_20d"]
        .rank(method="min", ascending=False, na_option="bottom")
        .astype("Int64")
    )
    inventory = inventory.sort_values(["scope_class", "benchmark_group", "status", "ts_code"])

    candidates = inventory[inventory["scope_class"] == "candidate_domestic_equity"].copy()
    excluded = inventory[inventory["scope_class"] != "candidate_domestic_equity"].copy()
    groups = (
        candidates.groupby("benchmark_group", as_index=False)
        .agg(
            product_count=("ts_code", "count"),
            current_count=("market_state_as_of", lambda x: int((x == "active").sum())),
            ended_count=("market_state_as_of", lambda x: int((x == "ended").sum())),
            pending_count=("market_state_as_of", lambda x: int((x == "not_yet_listed").sum())),
            products=("ts_code", lambda x: ";".join(x.astype(str))),
            product_names=("name", lambda x: ";".join(x.astype(str))),
            max_avg_amount_20d=("avg_amount_20d", "max"),
        )
        .sort_values(["product_count", "current_count", "benchmark_group"], ascending=[False, False, True])
    )
    group_counts = groups.set_index("benchmark_group")["current_count"]
    selection = candidates[candidates["market_state_as_of"] == "active"].copy()
    selection["active_products_in_group"] = selection["benchmark_group"].map(group_counts)
    selection_columns = [
        "benchmark_group",
        "active_products_in_group",
        "liquidity_rank_in_group",
        "ts_code",
        "name",
        "management",
        "found_date",
        "list_date",
        "avg_amount_20d",
        "median_amount_20d",
        "latest_share",
        "share_date",
        "in_current_v2_pool",
        "benchmark",
    ]
    selection = selection[selection_columns].sort_values(
        ["benchmark_group", "liquidity_rank_in_group", "list_date", "ts_code"]
    )

    inventory.to_csv(args.output_dir / "all_stock_etfs.csv", index=False)
    candidates.to_csv(args.output_dir / "domestic_equity_candidates.csv", index=False)
    selection.to_csv(args.output_dir / "active_selection_view.csv", index=False)
    excluded.to_csv(args.output_dir / "excluded_and_review.csv", index=False)
    groups.to_csv(args.output_dir / "benchmark_groups.csv", index=False)
    (args.output_dir / "report.md").write_text(build_report(inventory, dates), encoding="utf-8")
    summary = {
        "as_of": dates[-1],
        "liquidity_start": dates[0],
        "liquidity_days": len(dates),
        "all_stock_etfs": len(inventory),
        "domestic_candidates": len(candidates),
        "current_domestic_candidates": int((candidates["market_state_as_of"] == "active").sum()),
        "ended_domestic_candidates": int((candidates["market_state_as_of"] == "ended").sum()),
        "pending_domestic_candidates": int((candidates["market_state_as_of"] == "not_yet_listed").sum()),
        "excluded_non_domestic": int((inventory["scope_class"] == "exclude_non_domestic").sum()),
        "review": int((inventory["scope_class"] == "review").sum()),
        "benchmark_groups": len(groups),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()