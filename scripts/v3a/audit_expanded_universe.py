#!/usr/bin/env python3
"""Build phase-one identity and lifecycle audits without computing strategy returns."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from alpha_etf.data.expanded_universe_audit import (  # noqa: E402
    AUDIT_SCHEMA_VERSION,
    apply_index_metadata_events,
    apply_variety_aliases,
    build_identity_intervals,
    build_product_master,
    build_representative_lifecycle,
    index_change_report,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=ROOT / "data" / "processed" / "expanded_etf_audit" / "sources",
    )
    parser.add_argument(
        "--overrides",
        type=Path,
        default=ROOT / "configs" / "expanded_etf_identity_overrides.csv",
    )
    parser.add_argument(
        "--metadata-events",
        type=Path,
        default=ROOT / "configs" / "expanded_etf_index_metadata_events.csv",
    )
    parser.add_argument(
        "--identity-candidates",
        type=Path,
        default=ROOT / "configs" / "expanded_etf_identity_candidates.csv",
    )
    parser.add_argument(
        "--index-aliases",
        type=Path,
        default=ROOT / "configs" / "expanded_etf_index_aliases.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "processed" / "expanded_etf_audit" / "phase1",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _report(summary: dict[str, object]) -> str:
    return "\n".join(
        [
            "# 扩大ETF池第一阶段身份审计",
            "",
            "> 本报告不包含公式、策略或收益结果。",
            "",
            "## 范围",
            "",
            f"- 产品快照：{summary['product_rows']:,} 只交易所ETF。",
            f"- 境内普通被动股票ETF：{summary['target_product_rows']:,} 只。",
            f"- 当前直接官方指数代码覆盖：{summary['direct_identity_rows']:,} 只。",
            f"- 同基准组唯一代码推断：{summary['inferred_identity_rows']:,} 只，仅作待核验线索。",
            f"- 供应商目录候选映射：{summary['candidate_identity_rows']:,} 只，仅作待核验线索。",
            f"- `etf_basic` 当前指数代码缺失：{summary['raw_current_index_code_missing_rows']:,} 只。",
            f"- 经分组与证据候选后代码仍缺失：{summary['remaining_index_code_unresolved_rows']:,} 只。",
            "",
            "## 历史身份",
            "",
            f"- 身份区间：{summary['identity_interval_rows']:,} 条。",
            f"- 已有官方历史区间覆盖的产品：{summary['official_history_product_rows']:,} 只。",
            f"- 已复核历史映射但仍需闸门确认：{summary['reviewed_history_product_rows']:,} 只。",
            f"- 历史身份待核验产品：{summary['history_review_product_rows']:,} 只。",
            f"- 历史身份尚未解决产品：{summary['historical_identity_unresolved_rows']:,} 只。",
            f"- 已确认跟踪指数变更产品：{summary['changed_product_rows']:,} 只。",
            f"- 同一指数代码/名称元数据调整：{summary['index_metadata_event_rows']:,} 条。",
            "",
            "## 生命周期",
            "",
            f"- 可暂时识别的指数品种：{summary['index_variety_rows']:,} 个。",
            f"- 代表ETF生命周期区间：{summary['representative_interval_rows']:,} 条。",
            f"- 已终止目标产品：{summary['ended_target_product_rows']:,} 只。",
            "",
            "## 阻塞项",
            "",
            f"- 错误级异常：{summary['error_exception_rows']:,} 条。",
            f"- 警告级异常：{summary['warning_exception_rows']:,} 条。",
            "- 当前身份快照不能证明历史上未更换指数；未有官方区间证据的记录不得进入第二阶段正式协议。",
            "- 行情、复权事件、终止现金流和逐日有效品种覆盖尚未在本报告中验证。",
            "",
            "## 产物",
            "",
            "- `product_master.csv`：产品范围、当前身份及解析状态。",
            "- `identity_intervals.csv`：带来源和置信状态的指数身份区间。",
            "- `representative_lifecycle.csv`：最早上市优先、终止后接续的代表区间。",
            "- `index_changes.csv`：有官方覆盖记录的指数变更清单。",
            "- `index_metadata_events.csv`：不改变经济品种的代码或名称调整。",
            "- `exceptions.csv`：所有缺失与待核验项。",
            "- `audit_manifest.json`：输入输出哈希及审计计数。",
            "",
        ]
    )


def main() -> None:
    args = _parse_args()
    source_manifest_path = args.source_dir / "source_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    fund_basic = pd.read_parquet(args.source_dir / "tushare_fund_basic_all.parquet")
    etf_basic = pd.read_parquet(args.source_dir / "tushare_etf_basic_all.parquet")
    index_basic = pd.read_parquet(args.source_dir / "tushare_index_basic_all.parquet")
    overrides = pd.read_csv(args.overrides, dtype=str)
    metadata_events = pd.read_csv(args.metadata_events, dtype=str)
    identity_candidates = pd.read_csv(args.identity_candidates, dtype=str)
    index_aliases = pd.read_csv(args.index_aliases, dtype=str)
    master = build_product_master(fund_basic, etf_basic, index_basic, identity_candidates)
    intervals, exceptions = build_identity_intervals(
        master,
        overrides,
        as_of=str(source_manifest["snapshot_as_of"]),
    )
    intervals, reviewed_metadata_events = apply_index_metadata_events(
        intervals, metadata_events
    )
    intervals = apply_variety_aliases(intervals, index_aliases)
    lifecycle = build_representative_lifecycle(intervals)
    changes = index_change_report(intervals)
    target = master[master["scope_decision"] == "include_domestic_passive_equity"]
    summary = {
        "product_rows": len(master),
        "target_product_rows": len(target),
        "direct_identity_rows": int(target["identity_resolution"].eq("etf_basic_current").sum()),
        "inferred_identity_rows": int(
            target["identity_resolution"].eq("same_benchmark_unique_current_index").sum()
        ),
        "candidate_identity_rows": int(
            target["identity_resolution"].eq("provider_identity_candidate").sum()
        ),
        "raw_current_index_code_missing_rows": int(target["index_code"].isna().sum()),
        "identity_interval_rows": len(intervals),
        "official_history_product_rows": int(
            intervals.loc[
                intervals["identity_confidence"].eq("official_historical_interval"), "ts_code"
            ].nunique()
            if not intervals.empty
            else 0
        ),
        "reviewed_history_product_rows": int(
            intervals.loc[
                intervals["identity_confidence"].eq("reviewed_historical_interval"), "ts_code"
            ].nunique()
            if not intervals.empty
            else 0
        ),
        "history_review_product_rows": int(
            exceptions.loc[
                exceptions["exception_code"].eq("history_identity_unverified"), "ts_code"
            ].nunique()
        ),
        "changed_product_rows": int(changes["ts_code"].nunique()) if not changes.empty else 0,
        "index_metadata_event_rows": len(reviewed_metadata_events),
        "remaining_index_code_unresolved_rows": int(
            len(target) - intervals["ts_code"].nunique()
        ),
        "historical_identity_unresolved_rows": int(
            len(target)
            - intervals.loc[
                intervals["identity_confidence"].isin(
                    ["official_historical_interval", "reviewed_historical_interval"]
                ),
                "ts_code",
            ].nunique()
        ),
        "index_variety_rows": int(intervals["variety_id"].nunique()) if not intervals.empty else 0,
        "representative_interval_rows": len(lifecycle),
        "ended_target_product_rows": int(target["status"].eq("D").sum()),
        "error_exception_rows": int(exceptions["severity"].eq("error").sum()),
        "warning_exception_rows": int(exceptions["severity"].eq("warning").sum()),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "product_master": args.output_dir / "product_master.csv",
        "identity_intervals": args.output_dir / "identity_intervals.csv",
        "representative_lifecycle": args.output_dir / "representative_lifecycle.csv",
        "index_changes": args.output_dir / "index_changes.csv",
        "index_metadata_events": args.output_dir / "index_metadata_events.csv",
        "exceptions": args.output_dir / "exceptions.csv",
        "tdx_audit_universe": args.output_dir / "tdx_audit_universe.json",
    }
    master.to_csv(outputs["product_master"], index=False)
    intervals.to_csv(outputs["identity_intervals"], index=False)
    lifecycle.to_csv(outputs["representative_lifecycle"], index=False)
    changes.to_csv(outputs["index_changes"], index=False)
    reviewed_metadata_events.to_csv(outputs["index_metadata_events"], index=False)
    exceptions.to_csv(outputs["exceptions"], index=False)
    tdx_universe = [
        {
            "symbol": str(row.symbol),
            "name": str(row.name),
            "category": "audit",
            "bucket": (
                str(row.resolved_index_code)
                if not pd.isna(row.resolved_index_code)
                else "pending_identity"
            ),
        }
        for row in target.sort_values("ts_code").itertuples(index=False)
    ]
    outputs["tdx_audit_universe"].write_text(
        json.dumps(tdx_universe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report_path = args.output_dir / "report.md"
    report_path.write_text(_report(summary), encoding="utf-8")
    manifest = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "interpretation": "phase_one_identity_audit_without_strategy_or_return_outputs",
        "source_manifest_sha256": _sha256(source_manifest_path),
        "identity_overrides_sha256": _sha256(args.overrides),
        "identity_candidates_sha256": _sha256(args.identity_candidates),
        "index_metadata_events_sha256": _sha256(args.metadata_events),
        "index_aliases_sha256": _sha256(args.index_aliases),
        "summary": summary,
        "outputs": {
            name: {"file": path.name, "sha256": _sha256(path)}
            for name, path in outputs.items()
        },
        "report": {"file": report_path.name, "sha256": _sha256(report_path)},
    }
    manifest_path = args.output_dir / "audit_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()