#!/usr/bin/env python3
"""Run V3A fixed-formula correctness gates without starting training."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from alpha_etf.research_v3a.artifacts import build_formula_artifact, validate_formula_artifact
from alpha_etf.research_v3a.candidates import CandidateConfig, build_candidate_record
from alpha_etf.research_v3a.factors import (
    FACTOR_NAMES,
    attach_factor_cache,
    build_factor_values_numpy,
)
from alpha_etf.research_v3a.language import FORMULA_VOCAB, compile_formula
from alpha_etf.research_v3a.scoring import (
    ScorerConfig,
    SplitSpec,
    build_forward_targets,
    score_signal,
)
from alpha_etf.research_v3a.spec import V3APanel, load_dataset_manifest, load_panel
from alpha_etf.research_v3a.torch_scoring import TorchForwardTargets, score_signal_batch
from alpha_etf.research_v3a.torch_vm import BatchTorchVM, compiled_to_tensor
from alpha_etf.research_v3a.vm import StackVM
from scripts.v3a.runtime import DATASET_DIR, build_runtime_research_spec


OUT_DIR = ROOT / "data/processed/v3a/sanity"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def _device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(name)


def _factor_source_tokens(name: str) -> list[int]:
    if name in {"DAYRET", "GAP", "INTRADAY", "RANGE", "CLV"}:
        names = [name]
    elif name.startswith("MA_RATIO_"):
        _, _, short, long = name.split("_")
        names = ["MA_RATIO", f"WIN_{short}", f"WIN_{long}"]
    else:
        family, window = name.rsplit("_", 1)
        names = [family, f"WIN_{window}"]
    return FORMULA_VOCAB.encode(names)


def _fixed_formulas() -> dict[str, list[int]]:
    formulas = {name: _factor_source_tokens(name) for name in FACTOR_NAMES}
    generic = {
        "GENERIC_ADD": ["DAYRET", "GAP", "ADD"],
        "GENERIC_SUB": ["ROC", "WIN_20", "PRICE_MA", "WIN_20", "SUB"],
        "GENERIC_MUL": ["DAYRET", "CLV", "MUL"],
        "GENERIC_NEG": ["DAYRET", "NEG"],
        "GENERIC_ABS": ["GAP", "ABS"],
        "GENERIC_SIGN": ["MA_RATIO", "WIN_5", "WIN_20", "SIGN"],
        "GENERIC_REF": ["PRICE_MA", "WIN_20", "WIN_5", "REF"],
        "GENERIC_MEAN": ["DAYRET", "WIN_10", "MEAN"],
        "GENERIC_LEN15": [
            "DAYRET",
            "ROC",
            "WIN_20",
            "ADD",
            "PRICE_MA",
            "WIN_20",
            "SUB",
            "WIN_5",
            "MEAN",
            "CLV",
            "ABS",
            "MUL",
            "WIN_1",
            "REF",
            "NEG",
        ],
    }
    formulas.update({name: FORMULA_VOCAB.encode(tokens) for name, tokens in generic.items()})
    return formulas


def _finite_max_abs(left: np.ndarray, right: np.ndarray) -> float:
    finite = np.isfinite(left) & np.isfinite(right)
    return float(np.max(np.abs(left[finite] - right[finite]))) if finite.any() else 0.0


def _stable_order(signal: np.ndarray, available: np.ndarray) -> tuple[int, ...]:
    eligible = available & np.isfinite(signal)
    indices = np.where(eligible)[0]
    return tuple(indices[np.argsort(-signal[indices], kind="stable")])


def main() -> None:
    args = parse_args()
    device = _device(args.device)
    manifest = load_dataset_manifest(args.dataset_dir)
    full_panel = load_panel(args.dataset_dir)
    train_end = int(
        np.searchsorted(
            full_panel.dates.values, np.datetime64("2021-12-31"), side="right"
        )
    )
    panel = attach_factor_cache(
        V3APanel(
            absolute_ohlc=full_panel.absolute_ohlc[:, :, :train_end],
            relative_ohlc=full_panel.relative_ohlc[:, :, :train_end],
            tradable_mask=full_panel.tradable_mask[:, :train_end],
            symbols=full_panel.symbols,
            dates=full_panel.dates[:train_end],
        )
    )
    assert panel.factor_values is not None
    formulas = _fixed_formulas()
    compiled = [compile_formula(tokens) for tokens in formulas.values()]
    cpu_vm = StackVM()
    cpu_results = [
        cpu_vm.execute_compiled(formula, panel.factor_values, panel.tradable_mask)
        for formula in compiled
    ]
    for name, result in zip(formulas, cpu_results, strict=True):
        if not result.valid or result.signal is None:
            raise RuntimeError(f"Fixed formula {name} failed CPU VM: {result.invalid_reason}")

    factor_cache = torch.as_tensor(panel.factor_values, dtype=torch.float32, device=device)
    torch_mask = torch.as_tensor(panel.tradable_mask, dtype=torch.bool, device=device)
    codes, lengths = compiled_to_tensor(compiled, device=device)
    torch_result = BatchTorchVM().execute(codes, lengths, factor_cache, torch_mask)
    if not bool(torch_result.valid.all().item()):
        raise RuntimeError(f"Fixed Torch VM invalid codes: {torch_result.invalid_code.tolist()}")
    torch_signals = torch_result.signal.detach().cpu().numpy()
    max_vm_diff = 0.0
    for index, result in enumerate(cpu_results):
        assert result.signal is not None
        if not np.array_equal(np.isfinite(result.signal), np.isfinite(torch_signals[index])):
            raise RuntimeError(f"CPU/Torch NaN mask differs for {list(formulas)[index]}")
        np.testing.assert_allclose(
            result.signal,
            torch_signals[index],
            rtol=2e-5,
            atol=2e-6,
            equal_nan=True,
        )
        max_vm_diff = max(max_vm_diff, _finite_max_abs(result.signal, torch_signals[index]))

    scales = np.ldexp(np.ones(len(panel.symbols), dtype=np.float64), np.arange(len(panel.symbols)) - 17)
    scaled_factors = build_factor_values_numpy(
        panel.absolute_ohlc * scales[:, None, None], panel.tradable_mask
    )
    np.testing.assert_allclose(
        panel.factor_values, scaled_factors, rtol=1e-10, atol=1e-12, equal_nan=True
    )
    max_scale_factor_diff = _finite_max_abs(panel.factor_values, scaled_factors)
    scaled_results = [
        cpu_vm.execute_compiled(formula, scaled_factors, panel.tradable_mask)
        for formula in compiled
    ]
    max_scale_formula_diff = 0.0
    for name, original, scaled in zip(formulas, cpu_results, scaled_results, strict=True):
        assert original.signal is not None and scaled.signal is not None
        np.testing.assert_allclose(
            original.signal, scaled.signal, rtol=1e-10, atol=1e-12, equal_nan=True
        )
        max_scale_formula_diff = max(
            max_scale_formula_diff, _finite_max_abs(original.signal, scaled.signal)
        )
        for date in range(panel.tradable_mask.shape[1]):
            if _stable_order(original.signal[:, date], panel.tradable_mask[:, date]) != _stable_order(
                scaled.signal[:, date], panel.tradable_mask[:, date]
            ):
                raise RuntimeError(f"Scale changed ranking for {name} at {panel.dates[date]}")

    cutoff = int(np.searchsorted(panel.dates.values, np.datetime64("2020-06-30"), side="right") - 1)
    changed_absolute = panel.absolute_ohlc.copy()
    future_multiplier = np.linspace(1.1, 2.0, changed_absolute.shape[-1] - cutoff - 1)
    changed_absolute[:, :, cutoff + 1 :] *= future_multiplier[None, None, :]
    changed_factors = build_factor_values_numpy(changed_absolute, panel.tradable_mask)
    np.testing.assert_allclose(
        panel.factor_values[..., : cutoff + 1],
        changed_factors[..., : cutoff + 1],
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )

    scorer_config = ScorerConfig()
    candidate_config = CandidateConfig()
    targets = build_forward_targets(
        panel.absolute("open"),
        panel.tradable_mask,
        panel.dates,
        SplitSpec("train", "2016-08-09", "2021-12-31"),
        scorer_config,
    )
    audit_names = (
        "DAYRET",
        "ROC_20",
        "PRICE_MA_20",
        "VOL_20",
        "TS_RANK_20",
        "RSV_20",
        "MA_RATIO_5_20",
        "GENERIC_LEN15",
    )
    formula_names = list(formulas)
    audit_indices = [formula_names.index(name) for name in audit_names]
    cpu_scores = [
        score_signal(
            name,
            cpu_results[index].signal,
            targets,
            panel.dates,
            panel.symbols,
            scorer_config,
        )
        for name, index in zip(audit_names, audit_indices, strict=True)
    ]
    for name, result in zip(audit_names, cpu_scores, strict=True):
        if not result.valid:
            raise RuntimeError(f"Fixed formula {name} failed scorer: {result.invalid_reason}")
    torch_targets = TorchForwardTargets.from_numpy(
        targets, device=device, dtype=torch.float32
    )
    batch_score = score_signal_batch(
        torch_result.signal[audit_indices],
        torch.ones(len(audit_indices), dtype=torch.bool, device=device),
        torch_targets,
        scorer_config,
    )
    if not bool(batch_score.valid.all().item()):
        raise RuntimeError(f"Fixed Torch scorer invalid codes: {batch_score.invalid_code.tolist()}")
    max_reward_diff = 0.0
    for row, result in enumerate(cpu_scores):
        reward_diff = abs(float(batch_score.reward[row].item()) - result.reward)
        max_reward_diff = max(max_reward_diff, reward_diff)
        if reward_diff > 1e-6:
            raise RuntimeError(f"CPU/Torch reward mismatch for {audit_names[row]}: {reward_diff}")
        for day, selected in enumerate(result.daily["selected_indices"]):
            k = int(targets.top_k[day])
            torch_selected = tuple(
                int(item) for item in batch_score.selected_indices[row, day, :k].tolist()
            )
            if tuple(selected) != torch_selected:
                raise RuntimeError(
                    f"CPU/Torch selection mismatch for {audit_names[row]} at target row {day}"
                )

    research_spec = build_runtime_research_spec(
        manifest, scorer_config=scorer_config, candidate_config=candidate_config
    )
    first_score = cpu_scores[0]
    record = build_candidate_record(
        formula_id="sanity_DAYRET",
        source="v3a_fixed_sanity",
        token_ids=formulas["DAYRET"],
        reward=first_score.reward,
        train_summary=first_score.summary,
    )
    artifact = build_formula_artifact(
        record,
        research_spec=research_spec,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    validate_formula_artifact(artifact, research_spec=research_spec)

    summary: dict[str, Any] = {
        "status": "passed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "dataset_id": manifest["dataset_id"],
        "panel_sha256": manifest["panel_sha256"],
        "research_spec_id": research_spec["research_spec_id"],
        "code_commit": research_spec["code_commit"],
        "code_fingerprint": research_spec["code_fingerprint"],
        "formula_count": len(formulas),
        "base_factor_count": len(FACTOR_NAMES),
        "train_scorer_days": targets.days,
        "scored_formula_count": len(audit_names),
        "max_cpu_torch_signal_abs_diff": max_vm_diff,
        "max_cpu_torch_reward_abs_diff": max_reward_diff,
        "max_scale_factor_abs_diff": max_scale_factor_diff,
        "max_scale_formula_abs_diff": max_scale_formula_diff,
        "future_cutoff": panel.dates[cutoff].date().isoformat(),
        "data_cutoff": panel.dates[-1].date().isoformat(),
        "scale_rank_invariant": True,
        "artifact_roundtrip": True,
        "research_spec": research_spec,
        "scorer_results": [result.summary for result in cpu_scores],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.out_dir / "fixed_formula_sanity.json"
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "research_spec"}, ensure_ascii=False, indent=2))
    print(f"output: {output_path}")


if __name__ == "__main__":
    main()