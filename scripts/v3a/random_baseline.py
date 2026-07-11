#!/usr/bin/env python3
"""Run a resumable, training-only V3A uniform-random formula baseline."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from alpha_etf.research_v3a.artifacts import (
    build_formula_artifact,
    validate_formula_artifact,
)
from alpha_etf.research_v3a.candidates import (
    CandidateConfig,
    CandidateRecord,
    SimilarityContext,
    build_candidate_record,
    candidate_record_from_dict,
    canonicalize_expression,
    expression_hash,
    signal_similarity,
)
from alpha_etf.research_v3a.checkpointing import atomic_save
from alpha_etf.research_v3a.factors import build_factor_values_numpy
from alpha_etf.research_v3a.language import FORMULA_VOCAB, compile_formula
from alpha_etf.research_v3a.sampling import SamplingConfig, generate_random_formula
from alpha_etf.research_v3a.scoring import (
    ScorerConfig,
    SplitSpec,
    build_forward_targets,
    score_signal as score_signal_v3a,
)
from alpha_etf.research_v3a.spec import load_dataset_manifest, load_panel
from alpha_etf.research_v3a.torch_scoring import TorchForwardTargets, score_signal_batch
from alpha_etf.research_v3a.torch_vm import BatchTorchVM, compiled_to_tensor
from alpha_etf.research_v3a.vm import StackVM
from alpha_etf.scoring import ScorerConfig as LegacyScorerConfig
from alpha_etf.scoring import score_signal as score_signal_legacy
from scripts.v3a.runtime import DATASET_DIR, build_runtime_research_spec


BASELINE_SCHEMA_VERSION = "etf-v3a-random-baseline-v1"
STATE_SCHEMA_VERSION = "etf-v3a-random-baseline-state-v1"
DEFAULT_OUT_DIR = ROOT / "data/processed/v3a/baseline/runs"
STATUS_KEYS = (
    "grammar_invalid",
    "vm_invalid",
    "insufficient_daily_signal",
    "low_coverage",
    "constant_signal",
    "canonical_duplicate",
    "selection_duplicate",
    "accepted_unique",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempts", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--checkpoint-every", type=int, default=10_000)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--run-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--stop-after",
        type=int,
        help="Stop after this cumulative attempt for checkpoint/resume verification.",
    )
    parser.add_argument("--expected-dataset-id")
    parser.add_argument("--expected-panel-sha256")
    parser.add_argument("--expected-research-spec-id")
    parser.add_argument("--expected-code-fingerprint")
    parser.add_argument(
        "--allow-unpinned-identity",
        action="store_true",
        help="Permit a local non-gating test without externally pinned identities.",
    )
    return parser.parse_args()


def _device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(name)


def _require_identity(actual: str, expected: str | None, name: str) -> None:
    if expected is not None and actual != expected:
        raise RuntimeError(f"V3A {name} mismatch: {actual} != {expected}")


def _identity_is_pinned(args: argparse.Namespace) -> bool:
    values = (
        args.expected_dataset_id,
        args.expected_panel_sha256,
        args.expected_research_spec_id,
        args.expected_code_fingerprint,
    )
    if all(value is not None for value in values):
        return True
    if any(value is not None for value in values):
        raise RuntimeError("V3A identity pins must be supplied together")
    if not args.allow_unpinned_identity:
        raise RuntimeError(
            "V3A formal baseline requires all four expected identities; "
            "use --allow-unpinned-identity only for a local non-gating test"
        )
    return False


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    _write_bytes_idempotent(path, encoded)


def _write_bytes_idempotent(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"V3A immutable artifact differs: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)


def _write_jsonl_atomic(path: Path, payloads: list[dict[str, Any]]) -> None:
    content = b"".join(
        (
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        for payload in payloads
    )
    _write_bytes_idempotent(path, content)


def _bucket(token_len: int) -> int:
    return 0 if token_len <= 5 else 1 if token_len <= 10 else 2


def _record_sort_key(record: CandidateRecord) -> tuple[float, int, str]:
    return (-record.reward, record.token_len, record.formula_hash)


def _sequence_digest(previous: str, tokens: list[int]) -> str:
    digest = hashlib.sha256()
    digest.update(bytes.fromhex(previous))
    digest.update(json.dumps(tokens, separators=(",", ":")).encode("ascii"))
    return digest.hexdigest()


def _quality_metrics(
    signals: torch.Tensor,
    targets: TorchForwardTargets,
    config: CandidateConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    decision = signals.index_select(2, targets.decision_indices).permute(0, 2, 1)
    usable = targets.available.unsqueeze(0) & torch.isfinite(decision)
    finite_count = usable.sum(dim=(1, 2))
    total = int(targets.available.sum().item())
    coverage = finite_count.to(signals.dtype) / float(total)
    cleaned = torch.where(usable, decision, torch.zeros_like(decision))
    denominator = finite_count.clamp_min(1).to(signals.dtype)
    mean = cleaned.sum(dim=(1, 2)) / denominator
    variance = (
        torch.where(
            usable,
            (decision - mean[:, None, None]) ** 2,
            torch.zeros_like(decision),
        ).sum(dim=(1, 2))
        / denominator
    )
    std = torch.sqrt(torch.clamp(variance, min=0.0))
    coverage_valid = coverage >= config.min_coverage
    variation_valid = torch.isfinite(std) & (std > config.constant_std_eps)
    return coverage_valid & variation_valid, coverage, std, variation_valid


def _empty_state(
    *, run_id: str, config: dict[str, Any], research_spec: dict[str, Any], seed: int
) -> dict[str, Any]:
    rng = random.Random(seed)
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "run_id": run_id,
        "config": config,
        "research_spec_id": research_spec["research_spec_id"],
        "research_spec": research_spec,
        "completed_attempts": 0,
        "rng_state": rng.getstate(),
        "status_counts": {key: 0 for key in STATUS_KEYS},
        "invalid_reason_counts": {},
        "length_counts": {},
        "canonical_hashes": set(),
        "structural_duplicate_count": 0,
        "selection_hashes": set(),
        "selection_duplicate_observations": 0,
        "semantic_valid_rewards": [],
        "top_records": [[], [], []],
        "formula_sequence_digest": "00" * 32,
        "attempt_ledger_offset": 0,
        "attempt_ledger_line_count": 0,
        "attempt_ledger_prefix_sha256": hashlib.sha256(b"").hexdigest(),
        "artifact_created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": 0.0,
        "resume_count": 0,
    }


def _validate_state(
    state: dict[str, Any], *, run_id: str, config: dict[str, Any], research_spec: dict[str, Any]
) -> None:
    if state.get("schema_version") != STATE_SCHEMA_VERSION:
        raise RuntimeError("V3A random-baseline state schema mismatch")
    if state.get("run_id") != run_id or state.get("config") != config:
        raise RuntimeError("V3A random-baseline run/config mismatch")
    if (
        state.get("research_spec_id") != research_spec["research_spec_id"]
        or state.get("research_spec") != research_spec
    ):
        raise RuntimeError("V3A random-baseline ResearchSpec mismatch")
    if set(state.get("status_counts", {})) != set(STATUS_KEYS):
        raise RuntimeError("V3A random-baseline status ledger mismatch")
    completed = int(state.get("completed_attempts", -1))
    if completed < 0 or sum(int(value) for value in state["status_counts"].values()) != completed:
        raise RuntimeError("V3A random-baseline attempt ledger does not reconcile")
    if int(state.get("attempt_ledger_line_count", -1)) != completed:
        raise RuntimeError("V3A random-baseline ledger line count does not reconcile")
    ledger_sha = str(state.get("attempt_ledger_prefix_sha256", ""))
    if len(ledger_sha) != 64 or any(character not in "0123456789abcdef" for character in ledger_sha):
        raise RuntimeError("V3A random-baseline ledger prefix SHA is invalid")
    if not isinstance(state.get("top_records"), list) or len(state["top_records"]) != 3:
        raise RuntimeError("V3A random-baseline top-record buckets are invalid")
    selection_duplicate_observations = int(
        state.get("selection_duplicate_observations", -1)
    )
    semantic_valid_count = len(state.get("semantic_valid_rewards", []))
    if (
        selection_duplicate_observations < 0
        or semantic_valid_count > completed
        or selection_duplicate_observations > semantic_valid_count
        or semantic_valid_count - len(state.get("selection_hashes", set()))
        != selection_duplicate_observations
    ):
        raise RuntimeError("V3A random-baseline selection-duplicate audit is invalid")
    for bucket_index, records in enumerate(state["top_records"]):
        for payload in records:
            record = candidate_record_from_dict(payload)
            if _bucket(record.token_len) != bucket_index:
                raise RuntimeError("V3A random-baseline record is in the wrong bucket")


def _save_state(state: dict[str, Any], path: Path) -> None:
    atomic_save(state, path)


def _validate_ledger_prefix(
    path: Path, *, offset: int, expected_lines: int, expected_sha256: str | None
) -> str:
    if not path.exists() or path.stat().st_size < offset:
        raise RuntimeError("V3A random-baseline ledger is shorter than checkpoint offset")
    digest = hashlib.sha256()
    consumed = 0
    lines = 0
    with path.open("rb") as handle:
        while consumed < offset:
            line = handle.readline()
            if not line or consumed + len(line) > offset or not line.endswith(b"\n"):
                raise RuntimeError("V3A random-baseline checkpoint offset is not a JSONL boundary")
            digest.update(line)
            consumed += len(line)
            try:
                payload = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RuntimeError("V3A random-baseline ledger contains invalid JSON") from exc
            if int(payload.get("attempt_index", -1)) != lines:
                raise RuntimeError("V3A random-baseline ledger attempt indices are not contiguous")
            lines += 1
    if consumed != offset or lines != expected_lines:
        raise RuntimeError("V3A random-baseline ledger prefix does not match checkpoint counts")
    actual_sha256 = digest.hexdigest()
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise RuntimeError("V3A random-baseline ledger prefix SHA mismatch")
    return actual_sha256


def _load_or_create_state(
    *,
    run_dir: Path,
    run_id: str,
    config: dict[str, Any],
    research_spec: dict[str, Any],
    seed: int,
    resume: bool,
) -> tuple[dict[str, Any], Any]:
    state_path = run_dir / "state.pt"
    ledger_path = run_dir / "attempts.jsonl"
    if resume:
        if not state_path.exists() or not ledger_path.exists():
            raise FileNotFoundError("V3A random-baseline resume files are incomplete")
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        _validate_state(state, run_id=run_id, config=config, research_spec=research_spec)
        _validate_ledger_prefix(
            ledger_path,
            offset=int(state["attempt_ledger_offset"]),
            expected_lines=int(state["attempt_ledger_line_count"]),
            expected_sha256=str(state["attempt_ledger_prefix_sha256"]),
        )
        with ledger_path.open("r+b") as handle:
            handle.truncate(int(state["attempt_ledger_offset"]))
        if int(state["completed_attempts"]) < int(config["attempts"]):
            state["resume_count"] = int(state["resume_count"]) + 1
        ledger = ledger_path.open("ab")
        return state, ledger
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"V3A random-baseline run already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    state = _empty_state(
        run_id=run_id, config=config, research_spec=research_spec, seed=seed
    )
    ledger = ledger_path.open("xb")
    ledger.flush()
    os.fsync(ledger.fileno())
    _validate_state(state, run_id=run_id, config=config, research_spec=research_spec)
    _save_state(state, state_path)
    return state, ledger


def _reward_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "std": None, "quantiles": {}}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "quantiles": {
            str(quantile): float(np.quantile(array, quantile))
            for quantile in (0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0)
        },
    }


def _is_formal_baseline_run(
    *, device: torch.device, identity_pinned: bool, seed: int, attempts: int
) -> bool:
    return (
        device.type == "cuda"
        and identity_pinned
        and seed in {41, 42, 43}
        and attempts == 100_000
    )


def _replay_candidate_signals(
    records: list[CandidateRecord],
    *,
    factors: torch.Tensor,
    mask: torch.Tensor,
    targets: TorchForwardTargets,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    signals: dict[str, np.ndarray] = {}
    vm = BatchTorchVM()
    replay_batch_size = min(batch_size, 256)
    with torch.no_grad():
        for start in range(0, len(records), replay_batch_size):
            batch = records[start : start + replay_batch_size]
            compiled = [compile_formula(record.token_ids) for record in batch]
            codes, lengths = compiled_to_tensor(compiled, device=device)
            result = vm.execute(codes, lengths, factors, mask)
            if not bool(result.valid.all().item()):
                raise RuntimeError("A retained V3A baseline candidate failed VM replay")
            compressed = (
                result.signal.index_select(2, targets.decision_indices)
                .detach()
                .cpu()
                .numpy()
            )
            for record, signal in zip(batch, compressed, strict=True):
                signals[record.formula_hash] = signal
    return signals


def _completion_marker(
    *, run_dir: Path, summary: dict[str, Any], research_spec: dict[str, Any]
) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    formulas_path = run_dir / "top_formulas.jsonl"
    return {
        "schema_version": "etf-v3a-random-baseline-complete-v1",
        "run_id": summary["run_id"],
        "research_spec_id": research_spec["research_spec_id"],
        "attempt_count": summary["attempt_count"],
        "formula_sequence_digest": summary["formula_sequence_digest"],
        "attempt_ledger_prefix_sha256": summary["attempt_ledger_prefix_sha256"],
        "summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        "top_formulas_sha256": hashlib.sha256(formulas_path.read_bytes()).hexdigest(),
    }


def _recover_completed_run(
    *, run_dir: Path, state: dict[str, Any], research_spec: dict[str, Any]
) -> dict[str, Any] | None:
    summary_path = run_dir / "summary.json"
    marker_path = run_dir / "complete.json"
    formulas_path = run_dir / "top_formulas.jsonl"
    if not summary_path.exists() and not marker_path.exists():
        return None
    if not summary_path.exists() or not formulas_path.exists():
        raise RuntimeError("V3A completed baseline artifacts are incomplete")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("run_id") != state["run_id"]
        or summary.get("research_spec_id") != research_spec["research_spec_id"]
        or int(summary.get("attempt_count", -1)) != int(state["completed_attempts"])
        or summary.get("formula_sequence_digest") != state["formula_sequence_digest"]
        or summary.get("attempt_ledger_prefix_sha256")
        != state["attempt_ledger_prefix_sha256"]
        or not summary.get("attempt_ledger_reconciled")
        or sum(int(value) for value in summary.get("status_counts", {}).values())
        != int(state["completed_attempts"])
    ):
        raise RuntimeError("V3A completed baseline summary identity mismatch")
    for line in formulas_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            validate_formula_artifact(json.loads(line), research_spec=research_spec)
    marker = _completion_marker(
        run_dir=run_dir, summary=summary, research_spec=research_spec
    )
    _write_json_atomic(marker_path, marker)
    return summary


def _top_signal_duplicate_audit(
    records: list[CandidateRecord],
    signals: dict[str, np.ndarray],
    context: SimilarityContext,
    config: CandidateConfig,
) -> dict[str, Any]:
    unique: list[CandidateRecord] = []
    duplicate_count = 0
    comparison_count = 0
    insufficient_overlap_count = 0
    for record in records:
        duplicate = False
        for kept in unique:
            comparison_count += 1
            result = signal_similarity(
                signals[record.formula_hash],
                signals[kept.formula_hash],
                context,
                config,
            )
            if not result.sufficient:
                insufficient_overlap_count += 1
                continue
            if result.rho >= config.duplicate_rho:
                duplicate = True
                break
        if duplicate:
            duplicate_count += 1
        else:
            unique.append(record)
    denominator = len(records)
    return {
        "scope": "raw_reward_top_50_heap_candidates",
        "definition": "training_daily_spearman_fisher_z",
        "duplicate_rho": config.duplicate_rho,
        "formula_count": denominator,
        "duplicate_formula_count": duplicate_count,
        "duplicate_rate": duplicate_count / denominator if denominator else 0.0,
        "comparison_count": comparison_count,
        "insufficient_overlap_count": insufficient_overlap_count,
    }


def _legacy_comparison(
    records: list[CandidateRecord],
    *,
    factor_values: np.ndarray,
    absolute_open: np.ndarray,
    mask: np.ndarray,
    dates: Any,
    symbols: np.ndarray,
    targets: Any,
    scorer_config: ScorerConfig,
) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    vm = StackVM()
    split_start = int(
        np.searchsorted(dates.values, np.datetime64("2016-08-09"), side="left")
    )
    legacy_config = LegacyScorerConfig(
        horizon=scorer_config.horizon,
        top_fraction=scorer_config.top_fraction,
        min_top_k=scorer_config.min_top_k,
        max_top_k=scorer_config.max_top_k,
        min_universe=scorer_config.min_universe,
    )
    for record in records[:10]:
        vm_result = vm.execute_compiled(
            compile_formula(record.token_ids), factor_values, mask
        )
        if not vm_result.valid or vm_result.signal is None:
            raise RuntimeError("A retained random-baseline formula failed CPU replay")
        corrected = score_signal_v3a(
            record.formula_id,
            vm_result.signal,
            targets,
            dates,
            symbols,
            scorer_config,
        )
        _, legacy = score_signal_legacy(
            record.formula_id,
            vm_result.signal[:, split_start:],
            absolute_open[:, split_start:],
            mask[:, split_start:],
            dates[split_start:],
            symbols,
            legacy_config,
        )
        legacy_mean = float(legacy["scorer_mean_return"])
        comparisons.append(
            {
                "formula_hash": record.formula_hash,
                "corrected_scorer_days": corrected.summary["scorer_days"],
                "legacy_scorer_days": int(legacy["scorer_days"]),
                "corrected_mean_absolute_return": corrected.summary[
                    "mean_absolute_return"
                ],
                "corrected_mean_excess_return": corrected.reward,
                "legacy_mean_absolute_return": legacy_mean,
                "absolute_return_mean_delta": corrected.summary[
                    "mean_absolute_return"
                ]
                - legacy_mean,
            }
        )
    return comparisons


def main() -> None:
    args = parse_args()
    if args.attempts < 1 or args.batch_size < 1 or args.checkpoint_every < 1:
        raise ValueError("attempts, batch-size, and checkpoint-every must be positive")
    device = _device(args.device)
    identity_pinned = _identity_is_pinned(args)
    formal_gate_run = _is_formal_baseline_run(
        device=device,
        identity_pinned=identity_pinned,
        seed=args.seed,
        attempts=args.attempts,
    )
    manifest = load_dataset_manifest(args.dataset_dir)
    scorer_config = ScorerConfig()
    candidate_config = CandidateConfig()
    sampling_config = SamplingConfig()
    research_spec = build_runtime_research_spec(
        manifest, scorer_config=scorer_config, candidate_config=candidate_config
    )
    _require_identity(manifest["dataset_id"], args.expected_dataset_id, "dataset id")
    _require_identity(manifest["panel_sha256"], args.expected_panel_sha256, "panel SHA")
    _require_identity(
        research_spec["research_spec_id"],
        args.expected_research_spec_id,
        "ResearchSpec id",
    )
    _require_identity(
        research_spec["code_fingerprint"],
        args.expected_code_fingerprint,
        "code fingerprint",
    )
    run_id = args.run_id or (
        f"v3a-random-s{args.seed}-n{args.attempts}-"
        f"{research_spec['research_spec_id'][:12]}"
    )
    config = {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "seed": args.seed,
        "attempts": args.attempts,
        "batch_size": args.batch_size,
        "checkpoint_every": args.checkpoint_every,
        "device_type": args.device,
        "identity_pinned": identity_pinned,
        "sampling_config": sampling_config.to_dict(),
        "candidate_config": candidate_config.to_dict(),
        "split": {"name": "train", "start": "2016-08-09", "end": "2021-12-31"},
    }
    run_dir = args.out_dir / run_id
    state, ledger = _load_or_create_state(
        run_dir=run_dir,
        run_id=run_id,
        config=config,
        research_spec=research_spec,
        seed=args.seed,
        resume=args.resume,
    )
    if int(state["completed_attempts"]) > args.attempts:
        ledger.close()
        raise RuntimeError("V3A random-baseline state exceeds requested attempts")
    completed_attempts = int(state["completed_attempts"])
    run_until = args.attempts if args.stop_after is None else int(args.stop_after)
    completed_finalization = completed_attempts == args.attempts and args.stop_after is None
    if not completed_finalization and not completed_attempts < run_until <= args.attempts:
        ledger.close()
        raise ValueError("stop-after must be above completed attempts and no larger than attempts")
    if completed_finalization:
        recovered_summary = _recover_completed_run(
            run_dir=run_dir, state=state, research_spec=research_spec
        )
        if recovered_summary is not None:
            ledger.close()
            print(json.dumps(recovered_summary, ensure_ascii=False, indent=2))
            print(f"output: {run_dir}")
            return
    rng = random.Random()
    rng.setstate(state["rng_state"])
    status_counts = Counter({key: int(value) for key, value in state["status_counts"].items()})
    invalid_reasons = Counter(
        {str(key): int(value) for key, value in state["invalid_reason_counts"].items()}
    )
    length_counts = Counter(
        {int(key): int(value) for key, value in state["length_counts"].items()}
    )
    canonical_hashes: set[str] = set(state["canonical_hashes"])
    structural_duplicate_count = int(state["structural_duplicate_count"])
    selection_hashes: set[str] = set(state["selection_hashes"])
    selection_duplicate_observations = int(
        state["selection_duplicate_observations"]
    )
    rewards = [float(value) for value in state["semantic_valid_rewards"]]
    top_records = [
        [candidate_record_from_dict(payload) for payload in bucket]
        for bucket in state["top_records"]
    ]

    panel = load_panel(args.dataset_dir)
    train_end = int(
        np.searchsorted(panel.dates.values, np.datetime64("2021-12-31"), side="right")
    )
    dates = panel.dates[:train_end]
    mask_numpy = panel.tradable_mask[:, :train_end]
    absolute = panel.absolute_ohlc[:, :, :train_end]
    factor_numpy = build_factor_values_numpy(absolute, mask_numpy)
    targets = build_forward_targets(
        absolute[:, 0, :],
        mask_numpy,
        dates,
        SplitSpec("train", "2016-08-09", "2021-12-31"),
        scorer_config,
    )
    factors = torch.as_tensor(factor_numpy, dtype=torch.float32, device=device)
    mask = torch.as_tensor(mask_numpy, dtype=torch.bool, device=device)
    torch_targets = TorchForwardTargets.from_numpy(
        targets, device=device, dtype=torch.float32
    )
    vm = BatchTorchVM()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    started = time.monotonic()
    last_checkpoint = int(state["completed_attempts"])
    try:
        while int(state["completed_attempts"]) < run_until:
            batch_start = int(state["completed_attempts"])
            batch_count = min(args.batch_size, run_until - batch_start)
            formulas = [
                generate_random_formula(rng, config=sampling_config)
                for _ in range(batch_count)
            ]
            compiled = [compile_formula(tokens) for tokens in formulas]
            codes, lengths = compiled_to_tensor(compiled, device=device)
            with torch.no_grad():
                vm_result = vm.execute(codes, lengths, factors, mask)
                scored = score_signal_batch(
                    vm_result.signal, vm_result.valid, torch_targets, scorer_config
                )
                quality_valid, coverage, std, variation_valid = _quality_metrics(
                    vm_result.signal, torch_targets, candidate_config
                )
            vm_valid_cpu = vm_result.valid.detach().cpu().numpy()
            score_valid_cpu = scored.valid.detach().cpu().numpy()
            quality_valid_cpu = quality_valid.detach().cpu().numpy()
            variation_valid_cpu = variation_valid.detach().cpu().numpy()
            coverage_cpu = coverage.detach().cpu().numpy()
            std_cpu = std.detach().cpu().numpy()
            reward_cpu = scored.reward.detach().cpu().numpy()
            selected_cpu = scored.selected_indices.detach().cpu().numpy()
            selected_for_hash = np.full_like(selected_cpu, -1)
            for day, top_k in enumerate(targets.top_k):
                selected_for_hash[:, day, : int(top_k)] = selected_cpu[
                    :, day, : int(top_k)
                ]

            for row, (tokens, formula) in enumerate(zip(formulas, compiled, strict=True)):
                attempt_index = batch_start + row
                token_len = len(tokens)
                length_counts[token_len] += 1
                state["formula_sequence_digest"] = _sequence_digest(
                    state["formula_sequence_digest"], tokens
                )
                canonical_hash = expression_hash(
                    canonicalize_expression(formula.expression)
                )
                is_canonical_duplicate = canonical_hash in canonical_hashes
                if is_canonical_duplicate:
                    structural_duplicate_count += 1
                canonical_hashes.add(canonical_hash)
                selection_hash: str | None = None
                reward: float | None = None
                invalid_reason = ""
                if not bool(vm_valid_cpu[row]):
                    status = "vm_invalid"
                    invalid_reason = "vm_invalid"
                elif not bool(score_valid_cpu[row]):
                    status = "insufficient_daily_signal"
                    invalid_reason = "insufficient_daily_signal"
                elif not bool(quality_valid_cpu[row]):
                    if not bool(variation_valid_cpu[row]):
                        status = "constant_signal"
                        invalid_reason = "constant_signal"
                    else:
                        status = "low_coverage"
                        invalid_reason = "low_coverage"
                else:
                    reward = float(reward_cpu[row])
                    rewards.append(reward)
                    selection_hash = hashlib.sha256(
                        selected_for_hash[row].astype(np.int16, copy=False).tobytes()
                    ).hexdigest()
                    is_selection_duplicate = selection_hash in selection_hashes
                    if is_selection_duplicate:
                        selection_duplicate_observations += 1
                    selection_hashes.add(selection_hash)
                    record = build_candidate_record(
                        formula_id=f"random_s{args.seed}_a{attempt_index}",
                        source="v3a_uniform_random_baseline",
                        token_ids=tokens,
                        reward=reward,
                        train_summary={
                            "scorer_days": targets.days,
                            "coverage": float(coverage_cpu[row]),
                            "finite_std": float(std_cpu[row]),
                        },
                        attempt_index=attempt_index,
                    )
                    if record.formula_hash != canonical_hash:
                        raise RuntimeError("V3A random-baseline canonical hash drift")
                    bucket = _bucket(token_len)
                    top_records[bucket].append(record)
                    top_records[bucket] = sorted(
                        top_records[bucket], key=_record_sort_key
                    )[: candidate_config.heap_per_bucket]
                    if is_canonical_duplicate:
                        status = "canonical_duplicate"
                    elif is_selection_duplicate:
                        status = "selection_duplicate"
                    else:
                        status = "accepted_unique"
                status_counts[status] += 1
                if invalid_reason:
                    invalid_reasons[invalid_reason] += 1
                ledger_row = {
                    "attempt_index": attempt_index,
                    "token_ids": tokens,
                    "token_names": FORMULA_VOCAB.decode(tokens),
                    "token_len": token_len,
                    "canonical_hash": canonical_hash,
                    "selection_hash": selection_hash,
                    "status": status,
                    "invalid_reason": invalid_reason,
                    "reward": reward,
                }
                ledger.write(
                    (
                        json.dumps(
                            ledger_row,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                            allow_nan=False,
                        )
                        + "\n"
                    ).encode("utf-8")
                )
            state["completed_attempts"] = batch_start + batch_count
            should_checkpoint = (
                int(state["completed_attempts"]) - last_checkpoint
                >= args.checkpoint_every
                or int(state["completed_attempts"]) == run_until
            )
            if should_checkpoint:
                ledger.flush()
                os.fsync(ledger.fileno())
                ledger_offset = ledger.tell()
                ledger_sha256 = _validate_ledger_prefix(
                    run_dir / "attempts.jsonl",
                    offset=ledger_offset,
                    expected_lines=int(state["completed_attempts"]),
                    expected_sha256=None,
                )
                state.update(
                    {
                        "rng_state": rng.getstate(),
                        "status_counts": dict(status_counts),
                        "invalid_reason_counts": dict(invalid_reasons),
                        "length_counts": {str(key): value for key, value in length_counts.items()},
                        "canonical_hashes": canonical_hashes,
                        "structural_duplicate_count": structural_duplicate_count,
                        "selection_hashes": selection_hashes,
                        "selection_duplicate_observations": selection_duplicate_observations,
                        "semantic_valid_rewards": rewards,
                        "top_records": [
                            [record.to_dict() for record in bucket]
                            for bucket in top_records
                        ],
                        "attempt_ledger_offset": ledger_offset,
                        "attempt_ledger_line_count": int(state["completed_attempts"]),
                        "attempt_ledger_prefix_sha256": ledger_sha256,
                        "elapsed_seconds": float(state["elapsed_seconds"])
                        + time.monotonic()
                        - started,
                    }
                )
                _validate_state(
                    state, run_id=run_id, config=config, research_spec=research_spec
                )
                _save_state(state, run_dir / "state.pt")
                last_checkpoint = int(state["completed_attempts"])
                started = time.monotonic()
                print(
                    f"attempts={state['completed_attempts']}/{args.attempts} "
                    f"accepted={status_counts['accepted_unique']}",
                    flush=True,
                )
    finally:
        ledger.close()

    if int(state["completed_attempts"]) < args.attempts:
        print(
            json.dumps(
                {
                    "status": "checkpointed",
                    "run_id": run_id,
                    "completed_attempts": state["completed_attempts"],
                    "target_attempts": args.attempts,
                    "formula_sequence_digest": state["formula_sequence_digest"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        print(f"output: {run_dir}")
        return

    if sum(status_counts.values()) != args.attempts:
        raise RuntimeError("V3A random-baseline final attempt ledger does not reconcile")
    _validate_ledger_prefix(
        run_dir / "attempts.jsonl",
        offset=int(state["attempt_ledger_offset"]),
        expected_lines=args.attempts,
        expected_sha256=str(state["attempt_ledger_prefix_sha256"]),
    )
    ordered_records = sorted(
        [record for bucket in top_records for record in bucket], key=_record_sort_key
    )
    finalization_started = time.monotonic()
    top50 = ordered_records[:50]
    top_signals = _replay_candidate_signals(
        top50,
        factors=factors,
        mask=mask,
        targets=torch_targets,
        device=device,
        batch_size=args.batch_size,
    )
    similarity_context = SimilarityContext(
        decision_indices=np.arange(targets.days, dtype=np.int64),
        available=targets.available,
    )
    signal_duplicate_audit = _top_signal_duplicate_audit(
        top50,
        top_signals,
        similarity_context,
        candidate_config,
    )
    created_at = str(state["artifact_created_at"])
    artifacts = [
        build_formula_artifact(
            record, research_spec=research_spec, created_at=created_at
        )
        for record in top50
    ]
    _write_jsonl_atomic(run_dir / "top_formulas.jsonl", artifacts)
    legacy_comparison = _legacy_comparison(
        top50,
        factor_values=factor_numpy,
        absolute_open=absolute[:, 0, :],
        mask=mask_numpy,
        dates=dates,
        symbols=panel.symbols,
        targets=targets,
        scorer_config=scorer_config,
    )
    elapsed = float(state["elapsed_seconds"])
    finalization_seconds = time.monotonic() - finalization_started
    summary: dict[str, Any] = {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "status": "passed" if formal_gate_run else "local_test_passed",
        "created_at": created_at,
        "run_id": run_id,
        "seed": args.seed,
        "device": str(device),
        "identity_pinned": identity_pinned,
        "gpu_gate_run": formal_gate_run,
        "stage_c_registered_seed": args.seed in {41, 42, 43},
        "stage_c_attempt_budget_satisfied": args.attempts == 100_000,
        "cuda_device_name": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else None,
        "dataset_id": manifest["dataset_id"],
        "panel_sha256": manifest["panel_sha256"],
        "research_spec_id": research_spec["research_spec_id"],
        "code_commit": research_spec["code_commit"],
        "code_fingerprint": research_spec["code_fingerprint"],
        "split": config["split"],
        "attempt_count": args.attempts,
        "attempt_ledger_reconciled": True,
        "attempt_ledger_line_count": int(state["attempt_ledger_line_count"]),
        "attempt_ledger_prefix_sha256": state["attempt_ledger_prefix_sha256"],
        "status_counts": {key: int(status_counts[key]) for key in STATUS_KEYS},
        "invalid_reason_counts": dict(sorted(invalid_reasons.items())),
        "grammar_invalid_rate": status_counts["grammar_invalid"] / args.attempts,
        "structural_duplicate_count": structural_duplicate_count,
        "canonical_duplicate_rate": structural_duplicate_count / args.attempts,
        "selection_duplicate_count": selection_duplicate_observations,
        "selection_duplicate_rate": (
            selection_duplicate_observations / len(rewards) if rewards else 0.0
        ),
        "semantic_valid_count": len(rewards),
        "accepted_unique_count": status_counts["accepted_unique"],
        "canonical_unique_count": len(canonical_hashes),
        "selection_unique_count": len(selection_hashes),
        "length_distribution": {
            str(key): int(value) for key, value in sorted(length_counts.items())
        },
        "reward_distribution": _reward_summary(rewards),
        "top_rewards": [record.reward for record in top50],
        "top_formula_hashes": [record.formula_hash for record in top50],
        "top_bucket_counts": [
            sum(_bucket(record.token_len) == bucket for record in top50)
            for bucket in range(3)
        ],
        "formula_sequence_digest": state["formula_sequence_digest"],
        "resume_count": int(state["resume_count"]),
        "formula_evaluation_seconds": elapsed,
        "finalization_seconds": finalization_seconds,
        "elapsed_seconds": elapsed + finalization_seconds,
        "attempts_per_second": args.attempts / elapsed if elapsed > 0 else None,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else None,
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device)
        if device.type == "cuda"
        else None,
        "legacy_scorer_comparison": legacy_comparison,
        "signal_duplicate_audit": signal_duplicate_audit,
        "stage_d_full_candidate_funnel_applied": False,
        "selection_hash_semantics": "full_train_daily_top-k_index_tensor",
        "random_action_semantics": sampling_config.to_dict()[
            "random_action_distribution"
        ],
        "random_program_trained": False,
        "transformer_output_read": False,
        "validation_or_final_metrics_read": False,
    }
    _write_json_atomic(run_dir / "summary.json", summary)
    marker = _completion_marker(
        run_dir=run_dir, summary=summary, research_spec=research_spec
    )
    _write_json_atomic(run_dir / "complete.json", marker)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"output: {run_dir}")


if __name__ == "__main__":
    main()