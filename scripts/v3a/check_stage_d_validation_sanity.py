#!/usr/bin/env python3
"""Check train-only robust-z qualification and sticky occupancy for frozen candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from alpha_etf.research_v3a.language import compile_formula  # noqa: E402
from alpha_etf.research_v3a.torch_vm import BatchTorchVM, compiled_to_tensor  # noqa: E402
from alpha_etf.research_v3a.validation import (  # noqa: E402
    ValidationConfig,
    rank_and_qualify,
)
from scripts.v3a.prepare_stage_d_validation import (  # noqa: E402
    _load_jsonl,
    _load_protocol,
    _load_train_view,
    _sha256,
    _write_json,
    _write_jsonl,
)


SCHEMA_VERSION = "etf-v3a-validation-train-occupancy-sanity-v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "configs/v3a_stage_d_validation.json"
    )
    parser.add_argument("--train-view-dir", type=Path, required=True)
    parser.add_argument("--preparation-dir", type=Path, required=True)
    parser.add_argument("--groups-file", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _verify_preparation(path: Path) -> None:
    for line in (path / "SHA256SUMS").read_text(encoding="ascii").splitlines():
        expected, name = line.split("  ", 1)
        if _sha256(path / name) != expected:
            raise RuntimeError(f"Validation preparation SHA mismatch: {name}")
    manifest = json.loads((path / "preparation_manifest.json").read_text(encoding="utf-8"))
    if (
        manifest["validation_or_final_metrics_read"] is not False
        or manifest["train_view_end"] != "2021-12-31"
    ):
        raise RuntimeError("Validation preparation manifest crossed the sealed boundary")


def _sticky_metrics(
    signal: np.ndarray,
    mask: np.ndarray,
    symbols: np.ndarray,
    decision_indices: np.ndarray,
    config: ValidationConfig,
) -> dict[str, Any]:
    held: set[int] = set()
    position_counts: list[int] = []
    qualified_counts: list[int] = []
    rankable_days = 0
    mad_degenerate_days = 0
    for t_value in decision_indices:
        t = int(t_value)
        ranks, robust_z, rankable = rank_and_qualify(
            signal[:, t], mask[:, t], symbols, config
        )
        if not rankable:
            continue
        rankable_days += 1
        if not np.isfinite(robust_z).any():
            mad_degenerate_days += 1
            qualified: list[int] = []
        else:
            qualified = [
                int(index)
                for index in np.flatnonzero(
                    (ranks <= config.buy_rank)
                    & (robust_z >= config.robust_z_threshold)
                )
            ]
            qualified.sort(key=lambda index: (ranks[index], str(symbols[index])))
        held = {index for index in held if ranks[index] <= config.hold_rank}
        for index in qualified:
            if len(held) >= config.slots:
                break
            held.add(index)
        qualified_counts.append(len(qualified))
        position_counts.append(len(held))
    if not rankable_days:
        raise RuntimeError("Validation sanity formula has no rankable training days")
    positions = np.asarray(position_counts, dtype=np.float64)
    qualified = np.asarray(qualified_counts, dtype=np.float64)
    return {
        "rankable_days": rankable_days,
        "mad_degenerate_days": mad_degenerate_days,
        "mad_degenerate_fraction": mad_degenerate_days / rankable_days,
        "mean_qualified_count": float(np.mean(qualified)),
        "no_qualified_fraction": float(np.mean(qualified == 0)),
        "three_qualified_fraction": float(np.mean(qualified >= config.slots)),
        "mean_position_count": float(np.mean(positions)),
        "empty_position_fraction": float(np.mean(positions == 0)),
        "full_position_fraction": float(np.mean(positions == config.slots)),
        "min_position_count": int(np.min(positions)),
        "max_position_count": int(np.max(positions)),
    }


def _group_summary(keys: list[str], metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "mad_degenerate_fraction",
        "mean_qualified_count",
        "no_qualified_fraction",
        "three_qualified_fraction",
        "mean_position_count",
        "empty_position_fraction",
        "full_position_fraction",
    )
    return {
        "formula_count": len(keys),
        **{
            f"median_{field}": float(np.median([metrics[key][field] for key in keys]))
            for field in fields
        },
    }


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# V3A Validation 训练期占用率 sanity",
        "",
        "> 只使用截止2021-12-31的train-view；不计算收益，不读取2022或2023+。",
        "",
        "该检查只观察`robust_z>=1.5`与3进5出是否机械退化。训练view没有close，",
        "因此粘性持仓代理不含-7%止损；止损路径由合成交易测试覆盖。",
        "",
        "| 方法 | Seed | 组 | 公式数 | MAD退化 | 无资格日 | 满仓占用 | 平均持仓数 |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for method, seeds in payload["groups"].items():
        for seed, groups in seeds.items():
            for group, values in groups.items():
                lines.append(
                    f"| {method} | {seed} | {group} | {values['formula_count']} | "
                    f"{values['median_mad_degenerate_fraction']:.2%} | "
                    f"{values['median_no_qualified_fraction']:.2%} | "
                    f"{values['median_full_position_fraction']:.2%} | "
                    f"{values['median_mean_position_count']:.2f} |"
                )
    lines.extend(
        [
            "",
            "## 边界",
            "",
            "- 本结果不是训练收益或样本外结论。",
            "- CUDA门完成后，如最终候选发生变化，必须用冻结后的最终组重跑本检查。",
            "- 本检查不搜索或比较其他robust-z阈值。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    torch.set_num_threads(min(8, torch.get_num_threads()))
    protocol = _load_protocol(args.protocol.resolve())
    preparation_dir = args.preparation_dir.resolve()
    _verify_preparation(preparation_dir)
    view = _load_train_view(args.train_view_dir.resolve(), protocol)
    registry_rows = _load_jsonl(preparation_dir / "cuda_registry.jsonl")
    registry = {row["sequence_hash"]: row for row in registry_rows}
    groups_path = (
        args.groups_file.resolve()
        if args.groups_file is not None
        else preparation_dir / "provisional_groups.json"
    )
    groups = json.loads(groups_path.read_text(encoding="utf-8"))
    selected_keys = sorted(
        {
            key
            for seeds in groups.values()
            for seed_groups in seeds.values()
            for name, keys in seed_groups.items()
            if name in {"original_top50", "stable_complex", "paired_simple"}
            for key in keys
        }
    )
    missing = set(selected_keys) - set(registry)
    if missing:
        raise RuntimeError("Validation sanity groups reference missing registry formulas")
    config = ValidationConfig(
        min_universe=int(protocol["trading"]["min_universe"]),
        slots=int(protocol["trading"]["slots"]),
        buy_rank=int(protocol["trading"]["buy_rank"]),
        hold_rank=int(protocol["trading"]["hold_rank"]),
        robust_z_threshold=float(protocol["trading"]["robust_z_threshold"]),
        stop_loss=float(protocol["trading"]["stop_loss"]),
    )
    factors = torch.as_tensor(view.factor_values, dtype=torch.float32)
    mask = torch.as_tensor(view.tradable_mask, dtype=torch.bool)
    vm = BatchTorchVM(
        max_output_bytes=2 * 1024**3,
        max_working_bytes=2 * 1024**3,
        max_total_bytes=3 * 1024**3,
    )
    decision_indices = np.flatnonzero(
        (view.dates >= np.datetime64("2016-08-09"))
        & (view.dates <= np.datetime64("2021-12-31"))
    )
    metrics: dict[str, dict[str, Any]] = {}
    for start in range(0, len(selected_keys), 32):
        keys = selected_keys[start : start + 32]
        formulas = [compile_formula(registry[key]["token_ids"]) for key in keys]
        codes, lengths = compiled_to_tensor(formulas, device=torch.device("cpu"))
        result = vm.execute(codes, lengths, factors, mask)
        if not bool(result.valid.all().item()):
            raise RuntimeError("Validation sanity selected formula is VM-invalid")
        signals = result.signal.detach().cpu().numpy()
        for index, key in enumerate(keys):
            metrics[key] = {
                "sequence_hash": key,
                "canonical_hash": registry[key]["canonical_hash"],
                **_sticky_metrics(
                    signals[index],
                    view.tradable_mask,
                    view.symbols,
                    decision_indices,
                    config,
                ),
            }
    group_summary: dict[str, Any] = {}
    for method, seeds in groups.items():
        group_summary[method] = {}
        for seed, seed_groups in seeds.items():
            group_summary[method][seed] = {
                name: _group_summary(keys, metrics)
                for name, keys in seed_groups.items()
                if name in {"original_top50", "stable_complex", "paired_simple"}
            }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": protocol["protocol_id"],
        "validation_or_final_metrics_read": False,
        "train_view_id": view.manifest["train_view_id"],
        "train_view_end": view.dates[-1].date().isoformat(),
        "robust_z_threshold": config.robust_z_threshold,
        "stop_loss_included": False,
        "return_metrics_computed": False,
        "selected_formula_count": len(selected_keys),
        "groups": group_summary,
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "formula_sanity.jsonl", [metrics[key] for key in selected_keys])
    _write_json(output_dir / "sanity_summary.json", payload)
    _write_report(output_dir / "sanity_report.md", payload)
    names = ("formula_sanity.jsonl", "sanity_summary.json", "sanity_report.md")
    lines = [f"{_sha256(output_dir / name)}  {name}" for name in names]
    (output_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="ascii")
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()