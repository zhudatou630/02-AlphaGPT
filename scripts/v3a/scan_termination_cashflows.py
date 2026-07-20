#!/usr/bin/env python3
"""Extract announced terminal cash distributions for ended representative ETFs."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
from pathlib import Path
import re
import subprocess
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "expanded-etf-terminal-cashflow-scan-v1"
LIST_API = "https://api.fund.eastmoney.com/f10/JJGG"
CONTENT_API = "https://np-cnotice-fund.eastmoney.com/api/content/ann"
TITLE_PATTERN = re.compile(r"清算资金.*(?:发放|分配)|剩余财产.*分配|清算款.*发放")
PAYMENT_DATE_PATTERNS = (
    re.compile(r"资金发放日为\s*(\d{4})年\s*(\d{1,2})月\s*(\d{1,2})日"),
    re.compile(r"清盘发放日为\s*(\d{4})年\s*(\d{1,2})月\s*(\d{1,2})日"),
    re.compile(r"发放日(?:为|[:：])?\s*(\d{4})年\s*(\d{1,2})月\s*(\d{1,2})日"),
)
CASH_PATTERNS = (
    re.compile(r"每份基金份额(?:可获分配)?清算资金为人民币\s*([0-9.]+)\s*元"),
    re.compile(r"每份基金份额发放资金为人民币\s*([0-9.]+)\s*元"),
    re.compile(r"每份基金份额(?:可获分配)?(?:的)?清算款为人民币\s*([0-9.]+)\s*元"),
    re.compile(r"每份场内基金份额实际发放资金为\s*([0-9.]+)\s*元"),
    re.compile(r"每份基金份额实际发放资金为\s*([0-9.]+)\s*元"),
    re.compile(r"每份基金份额(?:可获分配)?\s*([0-9.]+)\s*元"),
)
PER_HUNDRED_CASH_PATTERN = re.compile(r"每百份基金份额发放资金为(?:人民币)?\s*([0-9.]+)\s*元")
NAMED_PER_HUNDRED_CASH_PATTERN = re.compile(
    r"每百份[^，。；]*?份额实际发放资金为(?:人民币)?\s*([0-9.]+)\s*元"
)
NAMED_PER_SHARE_CASH_PATTERN = re.compile(
    r"每份[^，。；]*?基金份额实际发放资金为(?:人民币)?\s*([0-9.]+)\s*元"
)
TOTAL_CASH_PATTERNS = (
    re.compile(r"应分配剩余财产共计人民币\s*([0-9,.]+)\s*元"),
    re.compile(r"(?:剩余财产)?清盘总金额为人民币\s*([0-9,.]+)\s*元"),
    re.compile(r"本次清盘总金额为\s*([0-9,.]+)\s*元"),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase1-dir",
        type=Path,
        default=ROOT / "data" / "processed" / "expanded_etf_audit" / "phase1",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--cache-only", action="store_true")
    return parser.parse_args()


def _json_request(url: str) -> dict[str, object]:
    for attempt in range(4):
        try:
            completed = subprocess.run(
                [
                    "curl",
                    "-L",
                    "-sS",
                    "--max-time",
                    "30",
                    "-H",
                    "Referer: https://fundf10.eastmoney.com/",
                    "-H",
                    "User-Agent: Mozilla/5.0",
                    url,
                ],
                check=True,
                capture_output=True,
            )
            return json.loads(completed.stdout.decode("utf-8"))
        except Exception:
            if attempt == 3:
                raise
            time.sleep(1.5 * (attempt + 1))
    raise AssertionError("unreachable")


def _list_page(code: str, page: int, page_size: int) -> dict[str, object]:
    query = urlencode(
        {"fundcode": code, "pageIndex": page, "pageSize": page_size, "type": 6}
    )
    return _json_request(f"{LIST_API}?{query}")


def _parse_date(text: str) -> str | None:
    text = re.sub(r"\s+", "", text)
    for pattern in PAYMENT_DATE_PATTERNS:
        match = pattern.search(text)
        if match:
            year, month, day = (int(value) for value in match.groups())
            return f"{year:04d}-{month:02d}-{day:02d}"
    return None


def _parse_cash(text: str) -> float | None:
    text = re.sub(r"\s+", "", text)
    for pattern in CASH_PATTERNS:
        match = pattern.search(text)
        if match:
            return float(match.group(1))
    per_hundred = PER_HUNDRED_CASH_PATTERN.search(text)
    if per_hundred:
        return float(per_hundred.group(1)) / 100.0
    named_per_hundred = NAMED_PER_HUNDRED_CASH_PATTERN.search(text)
    if named_per_hundred:
        return float(named_per_hundred.group(1)) / 100.0
    named_per_share = NAMED_PER_SHARE_CASH_PATTERN.search(text)
    if named_per_share:
        return float(named_per_share.group(1))
    return None


def _parse_total_cash(text: str) -> float | None:
    text = re.sub(r"\s+", "", text)
    for pattern in TOTAL_CASH_PATTERNS:
        match = pattern.search(text)
        if match:
            return float(match.group(1).replace(",", ""))
    return None


def _pdf_text(url: str, announcement_id: str, cache_dir: Path) -> str:
    pdf_dir = cache_dir / "pdf"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = pdf_dir / f"{announcement_id}.pdf"
    text_path = pdf_dir / f"{announcement_id}.txt"
    if not pdf_path.exists():
        request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request, timeout=60) as response:
            pdf_path.write_bytes(response.read())
    if not text_path.exists():
        subprocess.run(
            ["pdftotext", "-layout", str(pdf_path), str(text_path)],
            check=True,
            capture_output=True,
        )
    return text_path.read_text(encoding="utf-8", errors="replace")


def _scan_one(code: str, page_size: int, cache_dir: Path) -> dict[str, object]:
    cache_path = cache_dir / f"{code}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    first = _list_page(code, 1, page_size)
    total = int(first.get("TotalCount", 0))
    pages = max(1, math.ceil(total / page_size))
    records = list(first.get("Data") or [])
    for page in range(2, pages + 1):
        records.extend(_list_page(code, page, page_size).get("Data") or [])
    candidates = [value for value in records if TITLE_PATTERN.search(str(value.get("TITLE") or ""))]
    parsed = []
    for candidate in candidates:
        announcement_id = str(candidate.get("ID"))
        try:
            content = _json_request(
                f"{CONTENT_API}?{urlencode({'art_code': announcement_id, 'client_source': 'web_fund'})}"
            ).get("data") or {}
        except Exception:
            content = {}
        text = str(content.get("notice_content") or "").replace("\n", " ")
        attachment_url = content.get("attach_url_web") or content.get("attach_url")
        if not attachment_url:
            attachment_url = f"https://pdf.dfcfw.com/pdf/H2_{announcement_id}_1.pdf"
        parsed.append(
            {
                "fund_code": code,
                "announcement_id": announcement_id,
                "publish_date": candidate.get("PUBLISHDATEDesc"),
                "title": candidate.get("TITLE"),
                "payment_date": _parse_date(text),
                "cash_per_share": _parse_cash(text),
                "total_distribution_amount": _parse_total_cash(text),
                "attachment_url": attachment_url,
                "content_chars": len(text),
            }
        )
    result = {
        "fund_code": code,
        "records_scanned": len(records),
        "candidate_count": len(parsed),
        "candidates": parsed,
    }
    cache_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    args = _parse_args()
    lifecycle = pd.read_csv(args.phase1_dir / "representative_lifecycle.csv", dtype=str)
    products = pd.read_csv(args.phase1_dir / "product_master.csv", dtype=str)
    status = products.set_index("symbol")["status"]
    ended_codes = sorted(
        {
            str(symbol)
            for symbol in lifecycle["representative_symbol"].astype(str)
            if status.get(str(symbol)) == "D"
        }
    )
    cache_dir = args.phase1_dir / "terminal_cashflow_scan_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        scan_codes = [
            code
            for code in ended_codes
            if not args.cache_only or (cache_dir / f"{code}.json").exists()
        ]
        futures = {
            executor.submit(_scan_one, code, args.page_size, cache_dir): code
            for code in scan_codes
        }
        for future in as_completed(futures):
            code = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(f"scanned terminal cashflow {code} candidates={result['candidate_count']}", flush=True)
            except Exception as exc:
                errors.append({"fund_code": code, "error": f"{type(exc).__name__}: {exc}"})
                print(f"failed terminal cashflow {code}: {exc}", flush=True)
    candidates = pd.DataFrame(
        [value for result in results for value in result["candidates"]],
        columns=[
            "fund_code",
            "announcement_id",
            "publish_date",
            "title",
            "payment_date",
            "cash_per_share",
            "total_distribution_amount",
            "attachment_url",
            "content_chars",
        ],
    ).sort_values(["fund_code", "publish_date"])
    for index, row in candidates.iterrows():
        if pd.notna(row["payment_date"]) and pd.notna(row["cash_per_share"]):
            continue
        attachment = row["attachment_url"]
        if pd.isna(attachment) or not str(attachment):
            continue
        try:
            text = _pdf_text(str(attachment), str(row["announcement_id"]), cache_dir)
            if pd.isna(row["payment_date"]):
                candidates.at[index, "payment_date"] = _parse_date(text)
            if pd.isna(row["cash_per_share"]):
                candidates.at[index, "cash_per_share"] = _parse_cash(text)
            if pd.isna(row["total_distribution_amount"]):
                candidates.at[index, "total_distribution_amount"] = _parse_total_cash(text)
        except Exception as exc:
            errors.append(
                {
                    "fund_code": str(row["fund_code"]),
                    "error": f"pdf {row['announcement_id']}: {type(exc).__name__}: {exc}",
                }
            )
    share = pd.read_parquet(
        ROOT
        / "data"
        / "processed"
        / "expanded_etf_audit"
        / "ended_history"
        / "tushare_ended_fund_share.parquet"
    )
    share["fund_code"] = share["ts_code"].astype(str).str[:6]
    final_share = share.sort_values("trade_date").groupby("fund_code").tail(1)[
        ["fund_code", "trade_date", "fd_share"]
    ].rename(columns={"trade_date": "final_share_date", "fd_share": "final_share_10k"})
    candidates = candidates.merge(final_share, on="fund_code", how="left")
    candidates["cash_value_source"] = pd.NA
    candidates.loc[candidates["cash_per_share"].notna(), "cash_value_source"] = (
        "announcement_per_share"
    )
    can_derive = (
        candidates["cash_per_share"].isna()
        & candidates["total_distribution_amount"].notna()
        & candidates["final_share_10k"].notna()
        & (pd.to_numeric(candidates["final_share_10k"], errors="coerce") > 0)
    )
    candidates.loc[can_derive, "cash_per_share"] = (
        pd.to_numeric(candidates.loc[can_derive, "total_distribution_amount"])
        / (pd.to_numeric(candidates.loc[can_derive, "final_share_10k"]) * 10000.0)
    )
    candidates.loc[can_derive, "cash_value_source"] = (
        "derived_total_distribution_over_final_share"
    )
    coverage_rows = []
    scanned_codes = {str(result["fund_code"]) for result in results}
    for code in ended_codes:
        group = candidates[candidates["fund_code"].eq(code)].sort_values("publish_date")
        parsed = group[group["payment_date"].notna() & group["cash_per_share"].notna()]
        explicit = parsed[parsed["cash_value_source"].eq("announcement_per_share")]
        derived = parsed[
            parsed["cash_value_source"].eq("derived_total_distribution_over_final_share")
        ]
        if code not in scanned_codes:
            status = "unscanned_source_error"
        elif group.empty:
            status = "missing_distribution_announcement"
        elif len(parsed) == len(group):
            status = "all_distribution_announcements_parsed"
        elif not parsed.empty:
            status = "partially_parsed_distribution_announcements"
        else:
            status = "announcement_found_unparsed"
        coverage_rows.append(
            {
                "fund_code": code,
                "distribution_announcement_count": len(group),
                "parsed_distribution_count": len(parsed),
                "explicit_per_share_count": len(explicit),
                "derived_per_share_count": len(derived),
                "cashflow_status": status,
                "first_payment_date": parsed["payment_date"].min() if not parsed.empty else pd.NA,
                "last_payment_date": parsed["payment_date"].max() if not parsed.empty else pd.NA,
                "total_cash_per_share": (
                    pd.to_numeric(parsed["cash_per_share"]).sum()
                    if len(parsed) == len(group) and not parsed.empty
                    else pd.NA
                ),
                "official_cashflow_verified": False,
            }
        )
    coverage = pd.DataFrame(coverage_rows)
    coverage["evidence_source_url"] = pd.NA
    coverage["lifecycle_event_type"] = pd.NA
    coverage["successor_code"] = pd.NA
    coverage["lifecycle_event_verified"] = False
    overrides = pd.read_csv(
        ROOT / "configs" / "expanded_etf_terminal_cashflow_overrides.csv", dtype=str
    )
    for row in overrides.itertuples(index=False):
        code = str(row.ts_code)[:6]
        mask = coverage["fund_code"].eq(code)
        if not mask.any():
            continue
        coverage.loc[mask, "distribution_announcement_count"] = 1
        coverage.loc[mask, "parsed_distribution_count"] = 1
        coverage.loc[mask, "explicit_per_share_count"] = 1
        coverage.loc[mask, "derived_per_share_count"] = 0
        coverage.loc[mask, "cashflow_status"] = "official_cashflow_override"
        coverage.loc[mask, "first_payment_date"] = row.payment_date
        coverage.loc[mask, "last_payment_date"] = row.payment_date
        coverage.loc[mask, "total_cash_per_share"] = row.cash_per_share
        coverage.loc[mask, "official_cashflow_verified"] = True
        coverage.loc[mask, "evidence_source_url"] = row.source_url
    lifecycle = pd.read_csv(
        ROOT / "configs" / "expanded_etf_lifecycle_events.csv", dtype=str
    )
    for row in lifecycle.itertuples(index=False):
        code = str(row.ts_code)[:6]
        mask = coverage["fund_code"].eq(code)
        if not mask.any():
            continue
        coverage.loc[mask, "cashflow_status"] = "non_cash_lifecycle_event"
        coverage.loc[mask, "official_cashflow_verified"] = False
        coverage.loc[mask, "lifecycle_event_verified"] = True
        coverage.loc[mask, "evidence_source_url"] = row.source_url
        coverage.loc[mask, "lifecycle_event_type"] = row.event_type
        coverage.loc[mask, "successor_code"] = row.successor_code
    error_frame = pd.DataFrame(errors, columns=["fund_code", "error"])
    candidates.to_csv(args.phase1_dir / "terminal_cashflow_candidates.csv", index=False)
    coverage.to_csv(args.phase1_dir / "terminal_cashflow_coverage.csv", index=False)
    error_frame.to_csv(args.phase1_dir / "terminal_cashflow_scan_errors.csv", index=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "ended_representative_products": len(ended_codes),
        "scan_successes": len(results),
        "unscanned_products": len(ended_codes) - len(results),
        "scan_errors": len(errors),
        "distribution_announcements": len(candidates),
        "cashflow_resolved_products": int(
            coverage["cashflow_status"].isin(
                ["all_distribution_announcements_parsed", "official_cashflow_override"]
            ).sum()
        ),
        "explicit_per_share_candidate_products": int(
            coverage["explicit_per_share_count"].gt(0).sum()
        ),
        "derived_candidate_products": int(coverage["derived_per_share_count"].gt(0).sum()),
        "official_cashflow_verified_products": int(
            coverage["official_cashflow_verified"].sum()
        ),
        "official_lifecycle_event_products": int(
            coverage["lifecycle_event_verified"].sum()
        ),
        "non_cash_lifecycle_products": int(
            coverage["cashflow_status"].eq("non_cash_lifecycle_event").sum()
        ),
        "partially_parsed_products": int(
            coverage["cashflow_status"].eq("partially_parsed_distribution_announcements").sum()
        ),
        "announcement_found_unparsed": int(
            coverage["cashflow_status"].eq("announcement_found_unparsed").sum()
        ),
        "missing_distribution_announcement": int(
            coverage["cashflow_status"].eq("missing_distribution_announcement").sum()
        ),
        "evidence_boundary": "mirror_title_and_content_scan_requires_final_official_review",
    }
    (args.phase1_dir / "terminal_cashflow_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()