#!/usr/bin/env python3
"""Classify ended-ETF holder-meeting resolutions for non-cash lifecycle actions."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PHASE1 = ROOT / "data" / "processed" / "expanded_etf_audit" / "phase1"
RESOLUTION_PATTERN = re.compile(r"表决结果暨决议生效")
ACTION_PATTERN = re.compile(r"吸收合并|基金合并|合并方案|转换运作方式|基金转型")


def main() -> None:
    announcements = pd.read_csv(PHASE1 / "ended_announcement_candidates.csv", dtype=str)
    resolutions = announcements[
        announcements["title"].str.contains(RESOLUTION_PATTERN, na=False)
    ].copy()
    cache_dir = PHASE1 / "lifecycle_action_scan_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    errors = []
    for row in resolutions.itertuples(index=False):
        pdf_path = cache_dir / f"{row.announcement_id}.pdf"
        text_path = cache_dir / f"{row.announcement_id}.txt"
        try:
            if not text_path.exists():
                subprocess.run(
                    [
                        "curl",
                        "-L",
                        "-sS",
                        "--max-time",
                        "60",
                        "-o",
                        str(pdf_path),
                        f"https://pdf.dfcfw.com/pdf/H2_{row.announcement_id}_1.pdf",
                    ],
                    check=True,
                )
                subprocess.run(
                    ["pdftotext", "-layout", str(pdf_path), str(text_path)], check=True
                )
            text = re.sub(r"\s+", "", text_path.read_text(encoding="utf-8", errors="replace"))
            matches = sorted(set(ACTION_PATTERN.findall(text)))
            if matches:
                first = min(text.find(match) for match in matches)
                rows.append(
                    {
                        "fund_code": row.fund_code,
                        "announcement_id": row.announcement_id,
                        "publish_date": row.publish_date,
                        "title": row.title,
                        "matched_actions": "|".join(matches),
                        "evidence_excerpt": text[max(0, first - 120) : first + 320],
                        "source_url": f"https://pdf.dfcfw.com/pdf/H2_{row.announcement_id}_1.pdf",
                    }
                )
        except Exception as exc:
            errors.append(
                {
                    "fund_code": row.fund_code,
                    "announcement_id": row.announcement_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    candidates = pd.DataFrame(rows)
    error_frame = pd.DataFrame(errors)
    candidates.to_csv(PHASE1 / "lifecycle_action_candidates.csv", index=False)
    error_frame.to_csv(PHASE1 / "lifecycle_action_scan_errors.csv", index=False)
    manifest = {
        "resolution_announcements": len(resolutions),
        "action_candidates": len(candidates),
        "candidate_products": int(candidates["fund_code"].nunique()) if not candidates.empty else 0,
        "scan_errors": len(error_frame),
        "evidence_boundary": "mirror_pdf_text_scan_requires_final_official_review",
    }
    (PHASE1 / "lifecycle_action_scan_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()