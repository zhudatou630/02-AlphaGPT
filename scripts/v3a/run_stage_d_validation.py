#!/usr/bin/env python3
"""Run the one-shot frozen 2022 Stage D validation and machine decision."""

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

from alpha_etf.research_v3a.language import compile_formula  # noqa: E402
from alpha_etf.research_v3a.spec import canonical_sha256, sha256_file  # noqa: E402
from alpha_etf.research_v3a.torch_vm import BatchTorchVM, compiled_to_tensor  # noqa: E402
from alpha_etf.research_v3a.validation import (  # noqa: E402
    ValidationConfig,
    decide_validation,
    run_formula_validation,
)
from alpha_etf.research_v3a.validation_view import load_validation_view  # noqa: E402
from scripts.v3a.build_stage_d_validation_view import _validate_approval  # noqa: E402
from scripts.v3a.prepare_stage_d_validation import _load_jsonl, _load_protocol  # noqa: E402
from scripts.v3a.runtime import code_fingerprint, git_commit, require_clean_v3a_code  # noqa: E402


SCHEMA_VERSION = "etf-v3a-validation-formal-result-v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--preflight-binding", type=Path, required=True)
    parser.add_argument("--validation-view-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _group_rows(
    groups: dict[str, Any], summaries: dict[str, dict[str, Any]], method: str
) -> dict[str, dict[str, list[dict[str, float]]]]:
    output: dict[str, dict[str, list[dict[str, float]]]] = {}
    for rule in ("paired_simple", "stable_complex", "original_top50"):
        output[rule] = {}
        for seed in ("101", "102", "103"):
            output[rule][seed] = [
                {
                    "total_return": float(summaries[key]["total_return"]),
                    "max_drawdown": float(summaries[key]["max_drawdown"]),
                }
                for key in groups[method][seed][rule]
            ]
    return output


def _write_report(path: Path, result: dict[str, Any]) -> None:
    decision = result["decision"]
    lines = [
        "# V3A Stage D 2022 Validation 正式结果",
        "",
        f"> 正式结论：`{decision['outcome']}`",
        f"> 正式赢家：`{decision.get('winner')}`",
        "> 2023+ final未读取。",
        "",
        "## 生存门",
        "",
        "| 规则 | 生存seed数 | 配对门 | 通过 |",
        "|---|---:|---|---|",
    ]
    for rule, gate in decision["survival"].items():
        pair = gate.get("pair_by_seed") or {}
        pair_text = ", ".join(f"{seed}:{value}" for seed, value in pair.items()) or "不适用"
        lines.append(
            f"| {rule} | {gate['surviving_seed_count']} | {pair_text} | {gate['passed']} |"
        )
    lines.extend(["", "## 正式判定", "", "```json"])
    lines.extend(json.dumps(decision, ensure_ascii=False, sort_keys=True, indent=2).splitlines())
    lines.extend(["```", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    protocol_path = args.protocol.resolve()
    require_clean_v3a_code(extra_paths=(protocol_path,))
    protocol = _load_protocol(protocol_path)
    approval = _load_json(args.approval.resolve())
    binding = _load_json(args.preflight_binding.resolve())
    _validate_approval(approval, protocol, binding)
    if binding["code_commit"] != git_commit() or binding["code_fingerprint"] != code_fingerprint():
        raise RuntimeError("Validation runner code differs from preflight binding")
    view = load_validation_view(args.validation_view_dir.resolve())
    if (
        view.manifest["approval_id"] != approval["approval_id"]
        or view.manifest["preflight_binding_id"] != binding["binding_id"]
        or view.manifest["protocol_id"] != protocol["protocol_id"]
        or view.manifest["code_commit"] != git_commit()
        or view.manifest["code_fingerprint"] != code_fingerprint()
    ):
        raise RuntimeError("Validation view runtime identity mismatch")
    candidate_dir = args.candidate_dir.resolve()
    candidate_manifest = _load_json(candidate_dir / "final_candidate_manifest.json")
    groups = _load_json(candidate_dir / "final_groups.json")
    if (
        candidate_manifest["protocol_id"] != protocol["protocol_id"]
        or candidate_manifest["validation_run_approved"] is not False
        or candidate_manifest["validation_or_final_metrics_read"] is not False
        or sha256_file(candidate_dir / "final_groups.json") != binding["final_groups_sha256"]
    ):
        raise RuntimeError("Validation candidate identity mismatch")
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
        raise RuntimeError("Validation groups reference formulas missing from registry")
    config = ValidationConfig(
        validation_start=protocol["split"]["validation_start"],
        validation_end=protocol["split"]["validation_end"],
        initial_cash=float(protocol["trading"]["initial_cash"]),
        min_universe=int(protocol["trading"]["min_universe"]),
        slots=int(protocol["trading"]["slots"]),
        buy_rank=int(protocol["trading"]["buy_rank"]),
        hold_rank=int(protocol["trading"]["hold_rank"]),
        robust_z_threshold=float(protocol["trading"]["robust_z_threshold"]),
        stop_loss=float(protocol["trading"]["stop_loss"]),
    )
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise RuntimeError("Formal validation output already exists; refusing a second run")
    output_dir.mkdir(parents=True)
    factors = torch.as_tensor(view.factor_values, dtype=torch.float32)
    mask = torch.as_tensor(view.tradable_mask, dtype=torch.bool)
    vm = BatchTorchVM(
        max_output_bytes=2 * 1024**3,
        max_working_bytes=2 * 1024**3,
        max_total_bytes=3 * 1024**3,
    )
    summaries: dict[str, dict[str, Any]] = {}
    daily_frames: list[pd.DataFrame] = []
    trade_frames: list[pd.DataFrame] = []
    for start in range(0, len(selected_keys), 32):
        keys = selected_keys[start : start + 32]
        formulas = [compile_formula(registry[key]["token_ids"]) for key in keys]
        codes, lengths = compiled_to_tensor(formulas, device=torch.device("cpu"))
        vm_result = vm.execute(codes, lengths, factors, mask)
        if not bool(vm_result.valid.all().item()):
            raise RuntimeError("Frozen validation formula is VM-invalid")
        signals = vm_result.signal.detach().cpu().numpy()
        for index, key in enumerate(keys):
            daily, trades, summary = run_formula_validation(
                formula_id=key,
                signal=signals[index],
                open_prices=view.absolute_open,
                close_prices=view.absolute_close,
                tradable_mask=view.tradable_mask,
                dates=view.dates,
                symbols=view.symbols,
                config=config,
            )
            summaries[key] = summary
            daily_frames.append(daily)
            if not trades.empty:
                trade_frames.append(trades)
    benchmark_pairs = {
        (summary["benchmark_total_return"], summary["benchmark_max_drawdown"])
        for summary in summaries.values()
    }
    if len(benchmark_pairs) != 1:
        raise RuntimeError("Validation formulas produced different market benchmarks")
    benchmark_return, benchmark_drawdown = next(iter(benchmark_pairs))
    stable_pair_differences = {
        seed: [
            summaries[pair["complex"]]["total_return"]
            - summaries[pair["simple"]]["total_return"]
            for pair in groups["transformer"][seed]["pairs"]
        ]
        for seed in ("101", "102", "103")
    }
    decision = decide_validation(
        transformer=_group_rows(groups, summaries, "transformer"),
        random=_group_rows(groups, summaries, "random"),
        benchmark={
            "total_return": float(benchmark_return),
            "max_drawdown": float(benchmark_drawdown),
        },
        stable_pair_differences=stable_pair_differences,
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": protocol["protocol_id"],
        "approval_id": approval["approval_id"],
        "preflight_binding_id": binding["binding_id"],
        "code_commit": git_commit(),
        "code_fingerprint": code_fingerprint(),
        "validation_view_id": view.manifest["validation_view_id"],
        "validation_view_manifest_sha256": sha256_file(
            args.validation_view_dir.resolve() / "validation_view_manifest.json"
        ),
        "final_groups_sha256": binding["final_groups_sha256"],
        "formula_count": len(selected_keys),
        "benchmark": {
            "total_return": float(benchmark_return),
            "max_drawdown": float(benchmark_drawdown),
        },
        "decision": decision,
        "validation_metrics_read": True,
        "final_metrics_read": False,
    }
    formal_result = output_dir / "formal_result.json"
    _write_json(formal_result, result)
    (output_dir / "FORMAL_RESULT_SHA256").write_text(
        f"{_sha256(formal_result)}  formal_result.json\n", encoding="ascii"
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
    _write_report(output_dir / "formal_report.md", result)
    artifact_names = (
        "formal_result.json",
        "FORMAL_RESULT_SHA256",
        "formula_summaries.jsonl",
        "daily.csv",
        "trades.csv",
        "formal_report.md",
    )
    lines = [f"{_sha256(output_dir / name)}  {name}" for name in artifact_names]
    (output_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="ascii")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()