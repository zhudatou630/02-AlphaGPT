#!/usr/bin/env python3
"""Scan representative ETF announcement titles for identity and termination evidence."""

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
SCHEMA_VERSION = "expanded-etf-announcement-title-scan-v1"
API = "https://api.fund.eastmoney.com/f10/JJGG"
TITLE_PATTERN = re.compile(
    r"(?:变更|更换|更名|调整|转换).*(?:标的指数|跟踪指数|基准指数)|"
    r"(?:标的指数|跟踪指数|基准指数).*(?:变更|更换|更名|调整)|"
    r"终止上市|基金财产清算|清算报告|终止基金合同|清算款|"
    r"吸收合并|基金合并|合并方案|转换运作方式|基金转型|持有人大会"
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
    parser.add_argument("--ended-only", action="store_true")
    return parser.parse_args()


def _request(code: str, page: int, page_size: int) -> dict[str, object]:
    params = urlencode(
        {
            "fundcode": code,
            "pageIndex": page,
            "pageSize": page_size,
            "type": 6,
        }
    )
    url = f"{API}?{params}"
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
            payload = json.loads(completed.stdout.decode("utf-8"))
            if int(payload.get("ErrCode", -1)) != 0:
                raise RuntimeError(str(payload.get("ErrMsg")))
            return payload
        except Exception:
            if attempt == 3:
                raise
            time.sleep(1.5 * (attempt + 1))
    raise AssertionError("unreachable")


def _scan_one(code: str, list_date: str, cache_dir: Path, page_size: int) -> dict[str, object]:
    cache_path = cache_dir / f"{code}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    first = _request(code, 1, page_size)
    total = int(first.get("TotalCount", 0))
    pages = max(1, math.ceil(total / page_size))
    records = list(first.get("Data") or [])
    for page in range(2, pages + 1):
        records.extend(_request(code, page, page_size).get("Data") or [])
    candidates = []
    for value in records:
        publish_date = str(value.get("PUBLISHDATEDesc") or "").replace("-", "")
        title = str(value.get("TITLE") or "")
        if publish_date and publish_date < list_date:
            continue
        if TITLE_PATTERN.search(title):
            candidates.append(
                {
                    "fund_code": code,
                    "announcement_id": value.get("ID"),
                    "publish_date": value.get("PUBLISHDATEDesc"),
                    "title": title,
                    "category": value.get("NEWCATEGORY"),
                }
            )
    result = {
        "fund_code": code,
        "list_date": list_date,
        "total_other_announcements": total,
        "pages_scanned": pages,
        "records_scanned": len(records),
        "candidate_count": len(candidates),
        "candidates": candidates,
    }
    cache_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    args = _parse_args()
    products = pd.read_csv(args.phase1_dir / "product_master.csv", dtype=str)
    targets = products[
        products["scope_decision"].eq("include_domestic_passive_equity")
    ]
    if args.ended_only:
        targets = products[
            products["scope_decision"].eq("include_domestic_passive_equity")
            & products["status"].eq("D")
        ]
    targets = targets[["ts_code", "name", "list_date", "delist_date"]].drop_duplicates(
        "ts_code"
    )
    prefix = "ended_announcement" if args.ended_only else "announcement"
    cache_name = "announcement_scan_cache_v2" if args.ended_only else "announcement_scan_cache_v3_all_targets"
    cache_dir = args.phase1_dir / cache_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _scan_one,
                str(row.ts_code)[:6],
                str(row.list_date),
                cache_dir,
                args.page_size,
            ): row
            for row in targets.itertuples(index=False)
        }
        for future in as_completed(futures):
            row = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(
                    f"scanned {row.ts_code} candidates={result['candidate_count']}",
                    flush=True,
                )
            except Exception as exc:
                errors.append(
                    {
                        "ts_code": str(row.ts_code),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                print(f"failed {row.ts_code}: {exc}", flush=True)
    summaries = pd.DataFrame(
        [
            {key: value for key, value in result.items() if key != "candidates"}
            for result in results
        ]
    ).sort_values("fund_code")
    candidates = pd.DataFrame(
        [candidate for result in results for candidate in result["candidates"]],
        columns=["fund_code", "announcement_id", "publish_date", "title", "category"],
    ).sort_values(["publish_date", "fund_code"], ascending=[False, True])
    error_frame = pd.DataFrame(errors, columns=["ts_code", "error"])
    summaries.to_csv(args.phase1_dir / f"{prefix}_scan_summary.csv", index=False)
    candidates.to_csv(args.phase1_dir / f"{prefix}_candidates.csv", index=False)
    error_frame.to_csv(args.phase1_dir / f"{prefix}_scan_errors.csv", index=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "target_products": len(targets),
        "successfully_scanned": len(summaries),
        "scan_errors": len(error_frame),
        "records_scanned": int(summaries["records_scanned"].sum()) if not summaries.empty else 0,
        "candidate_announcements": len(candidates),
        "scope": "title_candidates_only_requires_official_document_review",
    }
    (args.phase1_dir / f"{prefix}_scan_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()