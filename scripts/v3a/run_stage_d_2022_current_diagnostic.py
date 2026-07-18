#!/usr/bin/env python3
"""Run frozen Stage D groups on one continuous 2022-to-current capital path."""

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
)
from alpha_etf.research_v3a.torch_vm import BatchTorchVM, compiled_to_tensor  # noqa: E402
from alpha_etf.research_v3a.validation import ValidationConfig, run_formula_validation  # noqa: E402
from scripts.v3a.prepare_stage_d_validation import _load_jsonl  # noqa: E402
from scripts.v3a.runtime import code_fingerprint, git_commit, require_clean_v3a_code  # noqa: E402


SCHEMA_VERSION = "etf-v3a-2022-current-continuous-diagnostic-v1"
APPROVAL_SCHEMA_VERSION = "etf-v3a-2022-current-diagnostic-approval-v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--formal-2022-result", type=Path, required=True)
    parser.add_argument("--post2023-result", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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


def _load_protocol(path: Path) -> dict[str, Any]:
    protocol = _load_json(path)
    payload = dict(protocol)
    expected = str(payload.pop("protocol_id", ""))
    if canonical_sha256(payload) != expected:
        raise RuntimeError("2022-current diagnostic protocol ID mismatch")
    interpretation = protocol["interpretation"]
    if (
        protocol["run_approved"] is not False
        or interpretation["exploratory_only"] is not True
        or interpretation["formal_winner_allowed"] is not False
        or interpretation["formula_selection_allowed"] is not False
        or interpretation["changes_2022_outcome"] is not False
        or protocol["split"]["continuous_path"] is not True
        or protocol["split"]["reset_at_2023"] is not False
        or protocol["split"]["annual_reset"] is not False
    ):
        raise RuntimeError("2022-current diagnostic protocol semantics drifted")
    return protocol


def _validate_approval(approval: dict[str, Any], protocol: dict[str, Any]) -> None:
    payload = dict(approval)
    expected = str(payload.pop("approval_id", ""))
    if (
        approval.get("schema_version") != APPROVAL_SCHEMA_VERSION
        or canonical_sha256(payload) != expected
        or approval.get("protocol_id") != protocol["protocol_id"]
        or approval.get("code_commit") != git_commit()
        or approval.get("scope") != "run_frozen_2022_to_current_continuous_diagnostic_once"
        or approval.get("run_approved") is not True
        or approval.get("formal_winner_approved") is not False
    ):
        raise RuntimeError("2022-current diagnostic approval mismatch")


def _year_returns(daily: pd.DataFrame, column: str, years: list[int]) -> dict[str, float]:
    equity = daily.set_index(pd.to_datetime(daily["date"]))[column].astype(float)
    previous = 1.0
    output: dict[str, float] = {}
    for year in years:
        part = equity[equity.index.year == year]
        if part.empty:
            raise RuntimeError(f"Continuous diagnostic has no rows for {year}")
        last = float(part.iloc[-1])
        output[str(year)] = last / previous - 1.0
        previous = last
    return output


def _group_metrics(
    keys: list[str], summaries: dict[str, dict[str, Any]], years: list[int]
) -> dict[str, Any]:
    def values(field: str) -> np.ndarray:
        return np.asarray([summaries[key][field] for key in keys], dtype=np.float64)

    totals = values("total_return")
    annualized = values("annualized_return")
    drawdowns = values("max_drawdown")
    cash = values("average_cash_weight")
    return {
        "count": len(keys),
        "total_return_median": float(np.quantile(totals, 0.5, method="linear")),
        "total_return_q75": float(np.quantile(totals, 0.75, method="linear")),
        "annualized_return_median": float(
            np.quantile(annualized, 0.5, method="linear")
        ),
        "max_drawdown_median": float(np.quantile(drawdowns, 0.5, method="linear")),
        "average_cash_weight_median": float(np.quantile(cash, 0.5, method="linear")),
        "calendar_years": {
            str(year): {
                "return_median": float(
                    np.quantile(
                        [summaries[key]["calendar_year_returns"][str(year)] for key in keys],
                        0.5,
                        method="linear",
                    )
                ),
                "return_q75": float(
                    np.quantile(
                        [summaries[key]["calendar_year_returns"][str(year)] for key in keys],
                        0.75,
                        method="linear",
                    )
                ),
            }
            for year in years
        },
    }


def _typical_formula(
    keys: list[str],
    summaries: dict[str, dict[str, Any]],
    registry: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    median = float(np.median([summaries[key]["total_return"] for key in keys]))
    key = min(keys, key=lambda item: (abs(summaries[item]["total_return"] - median), item))
    return {
        "sequence_hash": key,
        "canonical_hash": registry[key]["canonical_hash"],
        "formula_text": registry[key]["formula_text"],
        "token_names": registry[key]["token_names"],
        "summary": summaries[key],
        "selection_rule": "closest_to_group_total_return_median_then_sequence_hash",
    }


def _group_payload(
    groups: dict[str, Any],
    summaries: dict[str, dict[str, Any]],
    registry: dict[str, dict[str, Any]],
    years: list[int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    metrics: dict[str, Any] = {}
    typical: dict[str, Any] = {}
    for method, seeds in groups.items():
        metrics[method] = {}
        typical[method] = {}
        for seed, seed_groups in seeds.items():
            metrics[method][seed] = {}
            typical[method][seed] = {}
            for rule in ("original_top50", "stable_complex", "paired_simple"):
                keys = seed_groups[rule]
                metrics[method][seed][rule] = _group_metrics(keys, summaries, years)
                typical[method][seed][rule] = _typical_formula(keys, summaries, registry)
    return metrics, typical


def main() -> None:
    args = _parse_args()
    protocol_path = args.protocol.resolve()
    require_clean_v3a_code(extra_paths=(protocol_path,))
    protocol = _load_protocol(protocol_path)
    approval = _load_json(args.approval.resolve())
    _validate_approval(approval, protocol)
    formal_2022_path = args.formal_2022_result.resolve()
    post2023_path = args.post2023_result.resolve()
    if (
        _sha256(formal_2022_path)
        != protocol["interpretation"]["frozen_2022_result_sha256"]
        or _sha256(post2023_path)
        != protocol["interpretation"]["frozen_post2023_result_sha256"]
    ):
        raise RuntimeError("Continuous diagnostic source result SHA mismatch")
    formal_2022 = _load_json(formal_2022_path)
    post2023 = _load_json(post2023_path)
    if (
        formal_2022["decision"]["outcome"] != "no_surviving_rule"
        or post2023["formal_winner_generated"] is not False
    ):
        raise RuntimeError("Continuous diagnostic source interpretation mismatch")
    candidate_dir = args.candidate_dir.resolve()
    groups_path = candidate_dir / "final_groups.json"
    if _sha256(groups_path) != protocol["candidates"]["final_groups_sha256"]:
        raise RuntimeError("Continuous diagnostic candidate groups changed")
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
        raise RuntimeError("Continuous diagnostic groups reference missing formulas")
    dataset_dir = args.dataset_dir.resolve()
    dataset_manifest = load_dataset_manifest(dataset_dir)
    panel = load_panel(dataset_dir)
    if panel.dates[-1].date().isoformat() != protocol["split"]["end"]:
        raise RuntimeError("Continuous diagnostic data end differs from protocol")
    factor_values = build_factor_values_numpy(panel.absolute_ohlc, panel.tradable_mask)
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
        metrics_scope="full_history_diagnostic",
    )
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise RuntimeError("Continuous diagnostic output exists; refusing a second run")
    output_dir.mkdir(parents=True)
    factors = torch.as_tensor(factor_values, dtype=torch.float32)
    mask = torch.as_tensor(panel.tradable_mask, dtype=torch.bool)
    vm = BatchTorchVM(
        max_output_bytes=2 * 1024**3,
        max_working_bytes=2 * 1024**3,
        max_total_bytes=3 * 1024**3,
    )
    years = [int(year) for year in protocol["split"]["calendar_years"]]
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
            raise RuntimeError("Frozen continuous diagnostic formula is VM-invalid")
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
            total = float(summary["total_return"])
            summary["annualized_return"] = float(
                (1.0 + total) ** (252.0 / len(daily)) - 1.0
            )
            summary["calendar_year_returns"] = _year_returns(daily, "equity", years)
            summaries[key] = summary
            daily_frames.append(daily)
            if not trades.empty:
                trade_frames.append(trades)
    assert first_daily is not None
    benchmark_values = np.r_[
        1.0, first_daily["benchmark_equity"].to_numpy(dtype=np.float64)
    ]
    benchmark = {
        "total_return": float(benchmark_values[-1] - 1.0),
        "annualized_return": float(
            benchmark_values[-1] ** (252.0 / len(first_daily)) - 1.0
        ),
        "max_drawdown": float(
            np.max(1.0 - benchmark_values / np.maximum.accumulate(benchmark_values))
        ),
        "calendar_year_returns": _year_returns(
            first_daily, "benchmark_equity", years
        ),
    }
    group_metrics, typical = _group_payload(
        groups, summaries, registry, years
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "interpretation": "exploratory_continuous_synthesis_not_formal_final",
        "protocol_id": protocol["protocol_id"],
        "approval_id": approval["approval_id"],
        "code_commit": git_commit(),
        "code_fingerprint": code_fingerprint(),
        "dataset_id": dataset_manifest["dataset_id"],
        "panel_sha256": dataset_manifest["panel_sha256"],
        "formula_count": len(selected_keys),
        "date_start": str(first_daily.iloc[0]["date"]),
        "date_end": str(first_daily.iloc[-1]["date"]),
        "continuous_path": True,
        "reset_at_2023": False,
        "benchmark": benchmark,
        "groups": group_metrics,
        "typical_formulas": typical,
        "frozen_2022_outcome": "no_surviving_rule",
        "formal_winner_generated": False,
        "formula_selection_performed": False,
        "validation_metrics_read": True,
        "post2023_metrics_read": True,
        "final_metrics_read": True,
    }
    result_path = output_dir / "continuous_result.json"
    _write_json(result_path, result)
    (output_dir / "CONTINUOUS_RESULT_SHA256").write_text(
        f"{_sha256(result_path)}  continuous_result.json\n", encoding="ascii"
    )
    _write_jsonl(
        output_dir / "formula_summaries.jsonl",
        [
            {
                "sequence_hash": key,
                "canonical_hash": registry[key]["canonical_hash"],
                "formula_text": registry[key]["formula_text"],
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
    names = (
        "continuous_result.json",
        "CONTINUOUS_RESULT_SHA256",
        "formula_summaries.jsonl",
        "daily.csv",
        "trades.csv",
    )
    (output_dir / "SHA256SUMS").write_text(
        "\n".join(f"{_sha256(output_dir / name)}  {name}" for name in names) + "\n",
        encoding="ascii",
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()