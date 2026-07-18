#!/usr/bin/env python3
"""Run frozen Stage D groups on permanently unsealed 2023+ data for diagnosis only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from alpha_etf.research_v3a.factors import build_factor_values_numpy  # noqa: E402
from alpha_etf.research_v3a.language import compile_formula  # noqa: E402
from alpha_etf.research_v3a.spec import (  # noqa: E402
    canonical_sha256,
    load_dataset_manifest,
    load_panel,
    sha256_file,
)
from alpha_etf.research_v3a.torch_vm import BatchTorchVM, compiled_to_tensor  # noqa: E402
from alpha_etf.research_v3a.validation import (  # noqa: E402
    ValidationConfig,
    run_formula_validation,
)
from scripts.v3a.prepare_stage_d_validation import _load_jsonl  # noqa: E402
from scripts.v3a.runtime import code_fingerprint, git_commit, require_clean_v3a_code  # noqa: E402


SCHEMA_VERSION = "etf-v3a-post2023-exploratory-result-v1"
APPROVAL_SCHEMA_VERSION = "etf-v3a-post2023-exploratory-approval-v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--formal-2022-result", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_protocol(path: Path) -> dict[str, Any]:
    protocol = _load_json(path)
    payload = dict(protocol)
    expected = str(payload.pop("protocol_id", ""))
    if canonical_sha256(payload) != expected:
        raise RuntimeError("Post-2023 exploratory protocol ID mismatch")
    if (
        protocol["exploratory_run_approved"] is not False
        or protocol["interpretation"]["formal_final"] is not False
        or protocol["interpretation"]["winner_selection_allowed"] is not False
        or protocol["interpretation"]["formula_selection_allowed"] is not False
        or protocol["interpretation"]["changes_2022_outcome"] is not False
    ):
        raise RuntimeError("Post-2023 protocol is not frozen as exploratory-only")
    return protocol


def _validate_approval(approval: dict[str, Any], protocol: dict[str, Any]) -> None:
    payload = dict(approval)
    expected = str(payload.pop("approval_id", ""))
    if (
        approval.get("schema_version") != APPROVAL_SCHEMA_VERSION
        or canonical_sha256(payload) != expected
        or approval.get("protocol_id") != protocol["protocol_id"]
        or approval.get("scope")
        != "permanently_unseal_post2023_and_run_frozen_exploratory_once"
        or approval.get("exploratory_run_approved") is not True
        or approval.get("formal_final_approved") is not False
    ):
        raise RuntimeError("Post-2023 exploratory approval mismatch")


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _formula_year_returns(daily: pd.DataFrame) -> dict[str, float]:
    equity = daily.set_index(pd.to_datetime(daily["date"]))["equity"].astype(float)
    previous = 1.0
    output: dict[str, float] = {}
    for year in (2023, 2024, 2025, 2026):
        part = equity[equity.index.year == year]
        if part.empty:
            raise RuntimeError(f"Post-2023 path has no rows for {year}")
        last = float(part.iloc[-1])
        output[str(year)] = last / previous - 1.0
        previous = last
    return output


def _benchmark_year_returns(daily: pd.DataFrame) -> dict[str, float]:
    equity = daily.set_index(pd.to_datetime(daily["date"]))["benchmark_equity"].astype(float)
    previous = 1.0
    output: dict[str, float] = {}
    for year in (2023, 2024, 2025, 2026):
        part = equity[equity.index.year == year]
        last = float(part.iloc[-1])
        output[str(year)] = last / previous - 1.0
        previous = last
    return output


def _group_metrics(keys: list[str], summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    totals = np.asarray([summaries[key]["total_return"] for key in keys], dtype=np.float64)
    annualized = np.asarray(
        [summaries[key]["annualized_return"] for key in keys], dtype=np.float64
    )
    drawdowns = np.asarray(
        [summaries[key]["max_drawdown"] for key in keys], dtype=np.float64
    )
    years = {
        year: np.asarray(
            [summaries[key]["calendar_year_returns"][year] for key in keys],
            dtype=np.float64,
        )
        for year in ("2023", "2024", "2025", "2026")
    }
    return {
        "count": len(keys),
        "total_return_median": float(np.quantile(totals, 0.5, method="linear")),
        "total_return_q75": float(np.quantile(totals, 0.75, method="linear")),
        "annualized_return_median": float(
            np.quantile(annualized, 0.5, method="linear")
        ),
        "max_drawdown_median": float(np.quantile(drawdowns, 0.5, method="linear")),
        "calendar_years": {
            year: {
                "return_median": float(np.quantile(values, 0.5, method="linear")),
                "return_q75": float(np.quantile(values, 0.75, method="linear")),
            }
            for year, values in years.items()
        },
    }


def _group_payload(
    groups: dict[str, Any], summaries: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    return {
        method: {
            seed: {
                rule: _group_metrics(seed_groups[rule], summaries)
                for rule in ("original_top50", "stable_complex", "paired_simple")
            }
            for seed, seed_groups in seeds.items()
        }
        for method, seeds in groups.items()
    }


def _write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# V3A Stage D 2023+ 探索性诊断结果",
        "",
        "> 身份：探索性诊断，不是正式final，不改变2022 `no_surviving_rule`。",
        "",
        f"基准整段收益：{result['benchmark']['total_return']:.2%}；",
        f"最大回撤：{result['benchmark']['max_drawdown']:.2%}。",
        "",
        "| 方法 | Seed | 规则 | 数量 | 总收益中位数 | 总收益Q75 | 年化中位数 | 回撤中位数 |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for method, seeds in result["groups"].items():
        for seed, rules in seeds.items():
            for rule, values in rules.items():
                lines.append(
                    f"| {method} | {seed} | {rule} | {values['count']} | "
                    f"{values['total_return_median']:.2%} | {values['total_return_q75']:.2%} | "
                    f"{values['annualized_return_median']:.2%} | "
                    f"{values['max_drawdown_median']:.2%} |"
                )
    lines.extend(["", "## 逐年基准", ""])
    for year, value in result["benchmark"]["calendar_year_returns"].items():
        lines.append(f"- {year}: {value:.2%}")
    lines.extend(
        [
            "",
            "本报告不生成winner、pass/fail或公式筛选结果。详细逐年组级分布见机器结果。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    protocol_path = args.protocol.resolve()
    require_clean_v3a_code(extra_paths=(protocol_path,))
    protocol = _load_protocol(protocol_path)
    approval = _load_json(args.approval.resolve())
    _validate_approval(approval, protocol)
    formal_2022_path = args.formal_2022_result.resolve()
    formal_2022 = _load_json(formal_2022_path)
    if (
        _sha256(formal_2022_path)
        != protocol["interpretation"]["frozen_2022_result_sha256"]
        or formal_2022["decision"]["outcome"] != "no_surviving_rule"
        or formal_2022["final_metrics_read"] is not False
    ):
        raise RuntimeError("Frozen 2022 result identity or outcome mismatch")
    candidate_dir = args.candidate_dir.resolve()
    groups_path = candidate_dir / "final_groups.json"
    if _sha256(groups_path) != protocol["candidates"]["final_groups_sha256"]:
        raise RuntimeError("Post-2023 candidate groups differ from frozen validation groups")
    groups = _load_json(groups_path)
    registry_rows = _load_jsonl(args.registry.resolve())
    registry = {row["sequence_hash"]: row for row in registry_rows}
    selected_keys = sorted(
        {
            key
            for methods in groups.values()
            for seed_groups in methods.values()
            for name, keys in seed_groups.items()
            if name in {"original_top50", "stable_complex", "paired_simple"}
            for key in keys
        }
    )
    if set(selected_keys) - set(registry):
        raise RuntimeError("Post-2023 groups reference formulas missing from registry")
    dataset_dir = args.dataset_dir.resolve()
    dataset_manifest = load_dataset_manifest(dataset_dir)
    panel = load_panel(dataset_dir)
    if panel.dates[-1].date().isoformat() != protocol["split"]["end"]:
        raise RuntimeError("Post-2023 source data end differs from frozen protocol")
    factor_values = build_factor_values_numpy(
        panel.absolute_ohlc, panel.tradable_mask
    ).astype(np.float64, copy=False)
    config = ValidationConfig(
        validation_start=protocol["split"]["start"],
        validation_end=protocol["split"]["end"],
        initial_cash=float(protocol["trading"]["initial_cash"]),
        min_universe=int(protocol["trading"]["min_universe"]),
        slots=int(protocol["trading"]["slots"]),
        buy_rank=int(protocol["trading"]["buy_rank"]),
        hold_rank=int(protocol["trading"]["hold_rank"]),
        robust_z_threshold=float(protocol["trading"]["robust_z_threshold"]),
        stop_loss=float(protocol["trading"]["stop_loss"]),
        metrics_scope="post2023_exploratory",
    )
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise RuntimeError("Post-2023 exploratory output exists; refusing a second run")
    output_dir.mkdir(parents=True)
    factors = torch.as_tensor(factor_values, dtype=torch.float32)
    mask = torch.as_tensor(panel.tradable_mask, dtype=torch.bool)
    vm = BatchTorchVM(
        max_output_bytes=2 * 1024**3,
        max_working_bytes=2 * 1024**3,
        max_total_bytes=3 * 1024**3,
    )
    summaries: dict[str, dict[str, Any]] = {}
    daily_frames: list[pd.DataFrame] = []
    trade_frames: list[pd.DataFrame] = []
    first_daily: pd.DataFrame | None = None
    for start in range(0, len(selected_keys), 32):
        keys = selected_keys[start : start + 32]
        formulas = [compile_formula(registry[key]["token_ids"]) for key in keys]
        codes, lengths = compiled_to_tensor(formulas, device=torch.device("cpu"))
        vm_result = vm.execute(codes, lengths, factors, mask)
        if not bool(vm_result.valid.all().item()):
            raise RuntimeError("Frozen post-2023 formula is VM-invalid")
        signals = vm_result.signal.detach().cpu().numpy()
        for index, key in enumerate(keys):
            daily, trades, summary = run_formula_validation(
                formula_id=key,
                signal=signals[index],
                open_prices=panel.absolute("open"),
                close_prices=panel.absolute("close"),
                tradable_mask=panel.tradable_mask,
                dates=panel.dates,
                symbols=panel.symbols,
                config=config,
            )
            if first_daily is None:
                first_daily = daily.copy()
            years = _formula_year_returns(daily)
            days = len(daily)
            total = float(summary["total_return"])
            summary["annualized_return"] = float((1.0 + total) ** (252.0 / days) - 1.0)
            summary["calendar_year_returns"] = years
            summaries[key] = summary
            daily_frames.append(daily)
            if not trades.empty:
                trade_frames.append(trades)
    assert first_daily is not None
    benchmark_values = np.r_[
        1.0, first_daily["benchmark_equity"].to_numpy(dtype=np.float64)
    ]
    benchmark = {
        "total_return": float(first_daily.iloc[-1]["benchmark_equity"] - 1.0),
        "max_drawdown": float(
            np.max(
                1.0
                - benchmark_values / np.maximum.accumulate(benchmark_values)
            )
        ),
        "calendar_year_returns": _benchmark_year_returns(first_daily),
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "interpretation": "exploratory_only_not_formal_final",
        "protocol_id": protocol["protocol_id"],
        "approval_id": approval["approval_id"],
        "code_commit": git_commit(),
        "code_fingerprint": code_fingerprint(),
        "dataset_id": dataset_manifest["dataset_id"],
        "panel_sha256": dataset_manifest["panel_sha256"],
        "frozen_2022_outcome": "no_surviving_rule",
        "frozen_2022_result_sha256": _sha256(formal_2022_path),
        "final_groups_sha256": _sha256(groups_path),
        "formula_count": len(selected_keys),
        "date_start": str(first_daily.iloc[0]["date"]),
        "date_end": str(first_daily.iloc[-1]["date"]),
        "benchmark": benchmark,
        "groups": _group_payload(groups, summaries),
        "formal_winner_generated": False,
        "formula_selection_performed": False,
        "post2023_metrics_read": True,
        "final_metrics_read": True,
    }
    result_path = output_dir / "exploratory_result.json"
    _write_json(result_path, result)
    (output_dir / "EXPLORATORY_RESULT_SHA256").write_text(
        f"{_sha256(result_path)}  exploratory_result.json\n", encoding="ascii"
    )
    _write_jsonl(
        output_dir / "formula_summaries.jsonl",
        [
            {
                "sequence_hash": key,
                "canonical_hash": registry[key]["canonical_hash"],
                **summaries[key],
            }
            for key in selected_keys
        ],
    )
    pd.concat(daily_frames, ignore_index=True).to_csv(output_dir / "daily.csv", index=False)
    (
        pd.concat(trade_frames, ignore_index=True)
        if trade_frames
        else pd.DataFrame()
    ).to_csv(output_dir / "trades.csv", index=False)
    _write_report(output_dir / "exploratory_report.md", result)
    names = (
        "exploratory_result.json",
        "EXPLORATORY_RESULT_SHA256",
        "formula_summaries.jsonl",
        "daily.csv",
        "trades.csv",
        "exploratory_report.md",
    )
    (output_dir / "SHA256SUMS").write_text(
        "\n".join(f"{_sha256(output_dir / name)}  {name}" for name in names) + "\n",
        encoding="ascii",
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()