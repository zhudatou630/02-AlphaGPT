#!/usr/bin/env python3
"""Prepare frozen train-only candidates and CPU/CUDA numerical audit results."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from alpha_etf.research_v3a.attempts import ATTEMPT_DTYPE, AttemptStatus  # noqa: E402
from alpha_etf.research_v3a.candidates import (  # noqa: E402
    CandidateConfig,
    CandidateRecord,
    build_candidate_record,
)
from alpha_etf.research_v3a.language import (  # noqa: E402
    Expression,
    FORMULA_VOCAB,
    compile_formula,
)
from alpha_etf.research_v3a.scoring import ScorerConfig, SplitSpec, build_forward_targets  # noqa: E402
from alpha_etf.research_v3a.spec import canonical_sha256  # noqa: E402
from alpha_etf.research_v3a.stage_d import load_stage_d_train_view  # noqa: E402
from alpha_etf.research_v3a.torch_scoring import (  # noqa: E402
    TorchForwardTargets,
    score_signal_batch,
    signal_quality_batch,
)
from alpha_etf.research_v3a.torch_vm import BatchTorchVM, compiled_to_tensor  # noqa: E402
from scripts.v3a.export_top_formulas import select_curated_records  # noqa: E402
from scripts.v3a.preview_formula_curation import display_formula_metrics  # noqa: E402


SCHEMA_VERSION = "etf-v3a-validation-candidate-preparation-v1"
REGISTRY_SCHEMA_VERSION = "etf-v3a-validation-cuda-registry-v1"
SEEDS = (101, 102, 103)
GATE1_PROTOCOL_PREFIX = "9f9cfbe379a7"
GATE1_TOTAL_ATTEMPTS = 2_000_000
GATE1_BATCH_SIZE = 8192
FORMAL_TOTAL_ATTEMPTS = 8_000_000


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("prepare-cpu", "audit-cuda", "finalize"),
        default="prepare-cpu",
    )
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "configs/v3a_stage_d_validation.json"
    )
    parser.add_argument("--mechanism-runs-dir", type=Path)
    parser.add_argument("--random-runs-dir", type=Path)
    parser.add_argument("--train-view-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--preparation-dir", type=Path)
    parser.add_argument("--cuda-dir", type=Path)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    payload = dict(protocol)
    protocol_id = str(payload.pop("protocol_id"))
    actual = canonical_sha256(payload)
    if protocol_id != actual:
        raise RuntimeError(f"Validation protocol ID mismatch: {protocol_id} != {actual}")
    if protocol["validation_run_approved"] is not False:
        raise RuntimeError("Candidate preparation requires validation_run_approved=false")
    split = protocol["split"]
    if (
        split["candidate_end"] != "2021-12-31"
        or split["validation_metrics_read"] is not False
        or split["final_metrics_read"] is not False
    ):
        raise RuntimeError("Validation candidate protocol exposes sealed data")
    return protocol


def _load_train_view(path: Path, protocol: dict[str, Any]):
    manifest = json.loads((path / "train_view_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("train_view_id") != protocol["sources"]["train_view_id"]:
        raise RuntimeError("Validation candidate train-view identity mismatch")
    view = load_stage_d_train_view(
        path,
        research_spec={
            "research_spec_id": manifest["research_spec_id"],
            "code_fingerprint": manifest["code_fingerprint"],
        },
    )
    if (
        view.dates[-1].date().isoformat() != "2021-12-31"
        or view.manifest["split"]["validation_columns_present"] is not False
        or view.manifest["split"]["final_columns_present"] is not False
    ):
        raise RuntimeError("Validation preparation train view crosses the sealed boundary")
    return view


def _valid_mask(records: np.ndarray) -> np.ndarray:
    return (
        (records["status"] >= int(AttemptStatus.CANONICAL_DUPLICATE))
        & np.isfinite(records["reward"])
    )


def _hash_view(records: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(records["canonical_hash"][indices]).view("S32").reshape(-1)


def _representative_indices(records: np.ndarray, indices: np.ndarray) -> np.ndarray:
    selected = np.asarray(indices, dtype=np.int64)
    selected = selected[_valid_mask(records[selected])]
    hashes = _hash_view(records, selected)
    order = np.lexsort(
        (
            records["attempt_index"][selected],
            records["token_len"][selected],
            -records["reward"][selected],
            hashes,
        )
    )
    ordered_hashes = hashes[order]
    starts = np.r_[0, np.flatnonzero(ordered_hashes[1:] != ordered_hashes[:-1]) + 1]
    return selected[order[starts]]


def _reward_order(records: np.ndarray, indices: np.ndarray) -> np.ndarray:
    selected = np.asarray(indices, dtype=np.int64)
    hashes = _hash_view(records, selected)
    order = np.lexsort(
        (hashes, records["token_len"][selected], -records["reward"][selected])
    )
    return selected[order]


def _row_tokens(row: np.void) -> tuple[int, ...]:
    return tuple(int(value) for value in row["token_ids"][: int(row["token_len"])])


def _model_indices() -> np.ndarray:
    chunks: list[np.ndarray] = []
    for start in range(0, GATE1_TOTAL_ATTEMPTS, GATE1_BATCH_SIZE):
        stop = min(start + GATE1_BATCH_SIZE, GATE1_TOTAL_ATTEMPTS)
        model_count = (stop - start) * 3 // 4
        chunks.append(np.arange(start, start + model_count, dtype=np.int64))
    indices = np.concatenate(chunks)
    if len(indices) != GATE1_TOTAL_ATTEMPTS * 3 // 4:
        raise RuntimeError("Gate 1 Transformer lane count differs from frozen 75% split")
    return indices


def _load_transformer_ledger(runs_dir: Path, seed: int) -> tuple[np.memmap, Path]:
    run = runs_dir / f"v3a-stage-d-mechanism-transformer-s{seed}-{GATE1_PROTOCOL_PREFIX}"
    marker = json.loads((run / "training_complete.json").read_text(encoding="utf-8"))
    path = run / "attempts.bin"
    if (
        int(marker["attempt_count"]) != GATE1_TOTAL_ATTEMPTS
        or path.stat().st_size != GATE1_TOTAL_ATTEMPTS * ATTEMPT_DTYPE.itemsize
        or _sha256(path) != marker["attempt_ledger_sha256"]
    ):
        raise RuntimeError(f"Incomplete Gate 1 ledger: {path}")
    records = np.memmap(path, dtype=ATTEMPT_DTYPE, mode="r")
    if not np.array_equal(records["attempt_index"], np.arange(len(records), dtype=np.uint64)):
        raise RuntimeError(f"Gate 1 ledger attempt order mismatch: {path}")
    return records, path


def _load_random_ledger(
    runs_dir: Path, seed: int, cutoff: int, canonical_target: int
) -> tuple[np.memmap, Path]:
    run = runs_dir / f"v3a-stage-d-formal-matched_random-s{seed}-02cca48d1c85"
    path = run / "attempts.bin"
    if path.stat().st_size != FORMAL_TOTAL_ATTEMPTS * ATTEMPT_DTYPE.itemsize:
        raise RuntimeError(f"Random ledger size mismatch: {path}")
    records = np.memmap(path, dtype=ATTEMPT_DTYPE, mode="r")
    if not np.array_equal(
        records["attempt_index"][:cutoff], np.arange(cutoff, dtype=np.uint64)
    ):
        raise RuntimeError(f"Random ledger prefix order mismatch: {path}")
    prefix_hashes = (
        np.ascontiguousarray(records["canonical_hash"][:cutoff])
        .view("S32")
        .reshape(-1)
    )
    if len(np.unique(prefix_hashes)) != canonical_target:
        raise RuntimeError(f"Random ledger canonical target mismatch: {path}")
    return records, path


def _candidate_record(row: np.void, *, method: str, seed: int) -> CandidateRecord:
    record = build_candidate_record(
        formula_id=f"validation_{method}_s{seed}_a{int(row['attempt_index'])}",
        source=f"v3a_validation_{method}",
        token_ids=_row_tokens(row),
        reward=float(row["reward"]),
        train_summary={
            "coverage": float(row["coverage"]),
            "finite_std": float(row["finite_std"]),
        },
        attempt_index=int(row["attempt_index"]),
    )
    if record.formula_hash != row["canonical_hash"].tobytes().hex():
        raise RuntimeError("Validation candidate canonical hash differs from ledger")
    return record


def _curated_reserve(
    records: np.ndarray,
    representatives: np.ndarray,
    *,
    method: str,
    seed: int,
    protocol: dict[str, Any],
) -> list[CandidateRecord]:
    retained: list[int] = []
    lengths = records["token_len"][representatives]
    per_bucket = int(protocol["candidate_policy"]["retained_per_length_bucket"])
    for mask in (lengths <= 5, (lengths > 5) & (lengths <= 10), lengths > 10):
        retained.extend(
            int(index)
            for index in _reward_order(records, representatives[mask])[:per_bucket]
        )
    candidates = [_candidate_record(records[index], method=method, seed=seed) for index in retained]
    candidates.sort(key=lambda item: (-item.reward, item.token_len, item.formula_hash))
    display = {
        item.formula_hash: display_formula_metrics(item.token_names) for item in candidates
    }
    selected, _ = select_curated_records(
        candidates,
        size=int(protocol["candidate_policy"]["cuda_reserve_count_per_seed"]),
        display_metrics=display,
        tolerance=float(protocol["candidate_policy"]["curation_reward_tolerance"]),
    )
    return selected


def _expression_names(expression: Expression) -> list[str]:
    if expression.kind == "factor":
        return [expression.name, *(f"WIN_{int(value)}" for value in expression.params)]
    if expression.kind == "constant":
        return [expression.name]
    names: list[str] = []
    for child in expression.children:
        names.extend(_expression_names(child))
    names.extend(f"WIN_{int(value)}" for value in expression.params)
    names.append(expression.name)
    return names


def _proper_subtrees(expression: Expression) -> Iterable[Expression]:
    for child in expression.children:
        if child.kind != "constant":
            yield child
        yield from _proper_subtrees(child)


def _sequence_key(token_ids: Iterable[int]) -> str:
    return hashlib.sha256(bytes(int(value) for value in token_ids)).hexdigest()


def _register(
    registry: dict[str, dict[str, Any]],
    token_ids: tuple[int, ...],
    *,
    reference: dict[str, Any],
) -> str:
    key = _sequence_key(token_ids)
    candidate = build_candidate_record(
        formula_id="validation_registry",
        source="v3a_validation_registry",
        token_ids=token_ids,
        reward=float(reference.get("expected_reward", 0.0)),
        train_summary={},
    )
    entry = registry.setdefault(
        key,
        {
            "schema_version": REGISTRY_SCHEMA_VERSION,
            "sequence_hash": key,
            "canonical_hash": candidate.formula_hash,
            "token_ids": list(token_ids),
            "token_names": list(FORMULA_VOCAB.decode(token_ids)),
            "formula_text": compile_formula(token_ids).expression.text(),
            "references": [],
        },
    )
    if entry["token_ids"] != list(token_ids):
        raise RuntimeError("Validation registry sequence hash collision")
    entry["references"].append(reference)
    return key


def _build_targets(view, protocol: dict[str, Any]):
    scorer = ScorerConfig()
    full = build_forward_targets(
        view.absolute_open,
        view.tradable_mask,
        view.dates,
        SplitSpec("full", "2016-08-09", "2021-12-31"),
        scorer,
    )
    targets = {"full": full}
    for name, (start, end) in protocol["candidate_policy"]["subperiods"].items():
        targets[name] = build_forward_targets(
            view.absolute_open,
            view.tradable_mask,
            view.dates,
            SplitSpec(name, start, end),
            scorer,
        )
    return targets


def _selection_sha(selected: torch.Tensor, top_k: torch.Tensor) -> str:
    values = selected.detach().cpu().numpy()
    counts = top_k.detach().cpu().numpy()
    digest = hashlib.sha256()
    for row, count in zip(values, counts, strict=True):
        chosen = np.asarray(row[: int(count)], dtype="<i2")
        digest.update(chosen.tobytes())
    return digest.hexdigest()


def _score_registry(
    registry: list[dict[str, Any]],
    view,
    targets: dict[str, Any],
    *,
    device: torch.device,
    protocol: dict[str, Any],
) -> list[dict[str, Any]]:
    factors = torch.as_tensor(view.factor_values, dtype=torch.float32, device=device)
    mask = torch.as_tensor(view.tradable_mask, dtype=torch.bool, device=device)
    torch_targets = {
        name: TorchForwardTargets.from_numpy(target, device=device, dtype=torch.float32)
        for name, target in targets.items()
    }
    quality_config = CandidateConfig()
    gate = protocol["numerical_gate"]
    if (
        quality_config.min_coverage != float(gate["min_coverage"])
        or quality_config.constant_std_eps != float(gate["constant_std_eps"])
    ):
        raise RuntimeError("Validation numerical gate differs from CandidateConfig")
    vm = BatchTorchVM(
        max_output_bytes=2 * 1024**3,
        max_working_bytes=2 * 1024**3,
        max_total_bytes=3 * 1024**3,
    )
    output: list[dict[str, Any]] = []
    batch_size = 16 if device.type == "cpu" else 256
    scorer = ScorerConfig()
    for start in range(0, len(registry), batch_size):
        rows = registry[start : start + batch_size]
        formulas = [compile_formula(row["token_ids"]) for row in rows]
        codes, lengths = compiled_to_tensor(formulas, device=device)
        vm_result = vm.execute(codes, lengths, factors, mask)
        quality, coverage, finite_std, _ = signal_quality_batch(
            vm_result.signal,
            torch_targets["full"],
            min_coverage=float(gate["min_coverage"]),
            constant_std_eps=float(gate["constant_std_eps"]),
        )
        scores = {
            name: score_signal_batch(vm_result.signal, vm_result.valid, target, scorer)
            for name, target in torch_targets.items()
        }
        for index, row in enumerate(rows):
            result = {
                "sequence_hash": row["sequence_hash"],
                "device": device.type,
                "vm_valid": bool(vm_result.valid[index].item()),
                "quality_valid": bool(quality[index].item()),
                "coverage": float(coverage[index].item()),
                "finite_std": float(finite_std[index].item()),
                "scores": {
                    name: {
                        "valid": bool(score.valid[index].item()),
                        "reward": float(score.reward[index].item()),
                    }
                    for name, score in scores.items()
                },
                "full_topk_sha256": _selection_sha(
                    scores["full"].selected_indices[index],
                    torch_targets["full"].top_k,
                ),
            }
            output.append(result)
    return output


def _numerically_eligible(
    registry_row: dict[str, Any],
    cpu_score: dict[str, Any],
    protocol: dict[str, Any],
    cuda_score: dict[str, Any] | None = None,
) -> bool:
    if not cpu_score["vm_valid"] or not cpu_score["quality_valid"]:
        return False
    if not all(item["valid"] for item in cpu_score["scores"].values()):
        return False
    expected = [
        float(ref["expected_reward"])
        for ref in registry_row["references"]
        if "expected_reward" in ref
    ]
    tolerance = float(protocol["numerical_gate"]["full_reward_abs_tolerance"])
    cpu_full = float(cpu_score["scores"]["full"]["reward"])
    if expected and max(abs(cpu_full - value) for value in expected) > tolerance:
        return False
    if cuda_score is None:
        return True
    if not cuda_score["vm_valid"] or not cuda_score["quality_valid"]:
        return False
    if not all(item["valid"] for item in cuda_score["scores"].values()):
        return False
    cuda_full = float(cuda_score["scores"]["full"]["reward"])
    if abs(cpu_full - cuda_full) > tolerance:
        return False
    if expected and max(abs(cuda_full - value) for value in expected) > tolerance:
        return False
    return cpu_score["full_topk_sha256"] == cuda_score["full_topk_sha256"]


def _provisional_groups(
    parents: list[dict[str, Any]],
    registry: dict[str, dict[str, Any]],
    scores: dict[str, dict[str, Any]],
    protocol: dict[str, Any],
    cuda_scores: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    target_count = int(protocol["candidate_policy"]["original_count_per_seed"])
    tolerance = float(protocol["candidate_policy"]["curation_reward_tolerance"])
    for method in ("transformer", "random"):
        result[method] = {}
        for seed in SEEDS:
            rows = [
                row
                for row in parents
                if row["method"] == method
                and row["seed"] == seed
                and _numerically_eligible(
                    registry[row["sequence_hash"]],
                    scores[row["sequence_hash"]],
                    protocol,
                    None if cuda_scores is None else cuda_scores[row["sequence_hash"]],
                )
            ]
            records = [
                build_candidate_record(
                    formula_id=f"provisional_{method}_{seed}_{row['reserve_rank']}",
                    source="v3a_validation_provisional",
                    token_ids=registry[row["sequence_hash"]]["token_ids"],
                    reward=float(row["expected_reward"]),
                    train_summary={},
                )
                for row in rows
            ]
            records.sort(key=lambda item: (-item.reward, item.token_len, item.formula_hash))
            display = {
                item.formula_hash: display_formula_metrics(item.token_names) for item in records
            }
            selected, _ = select_curated_records(
                records, size=target_count, display_metrics=display, tolerance=tolerance
            )
            by_canonical = {
                registry[row["sequence_hash"]]["canonical_hash"]: row for row in rows
            }
            original = [by_canonical[item.formula_hash]["sequence_hash"] for item in selected]
            stable: list[str] = []
            pairs: list[dict[str, str]] = []
            for parent_key in original:
                parent = next(row for row in parents if row["sequence_hash"] == parent_key)
                eligible_subtrees = [
                    key
                    for key in parent["subtree_sequence_hashes"]
                    if _numerically_eligible(
                        registry[key],
                        scores[key],
                        protocol,
                        None if cuda_scores is None else cuda_scores[key],
                    )
                ]
                if not eligible_subtrees:
                    continue
                anchor = min(
                    eligible_subtrees,
                    key=lambda key: (
                        -float(scores[key]["scores"]["full"]["reward"]),
                        len(registry[key]["token_ids"]),
                        registry[key]["canonical_hash"],
                    ),
                )
                if all(
                    float(scores[parent_key]["scores"][period]["reward"])
                    > float(scores[anchor]["scores"][period]["reward"])
                    for period in ("S1", "S2", "S3")
                ):
                    stable.append(parent_key)
                    pairs.append({"complex": parent_key, "simple": anchor})
            simple_by_canonical: dict[str, str] = {}
            for pair in pairs:
                key = pair["simple"]
                simple_by_canonical.setdefault(registry[key]["canonical_hash"], key)
            result[method][str(seed)] = {
                "original_top50": original,
                "stable_complex": stable,
                "paired_simple": list(simple_by_canonical.values()),
                "pairs": pairs,
                "cpu_excluded_reserve_count": 100 - len(rows),
            }
    return result


def _write_sha256s(output_dir: Path, names: Iterable[str]) -> None:
    lines = [f"{_sha256(output_dir / name)}  {name}" for name in names]
    (output_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="ascii")


def _prepare_cpu(args: argparse.Namespace, protocol: dict[str, Any]) -> None:
    if (
        args.mechanism_runs_dir is None
        or args.random_runs_dir is None
        or args.train_view_dir is None
    ):
        raise ValueError(
            "prepare-cpu requires --mechanism-runs-dir, --random-runs-dir, and --train-view-dir"
        )
    view = _load_train_view(args.train_view_dir.resolve(), protocol)
    targets = _build_targets(view, protocol)
    registry: dict[str, dict[str, Any]] = {}
    parents: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    max_subtree_tokens = int(protocol["candidate_policy"]["short_subtree_max_tokens"])
    for seed in SEEDS:
        transformer, transformer_path = _load_transformer_ledger(
            args.mechanism_runs_dir.resolve(), seed
        )
        random_cutoff = int(protocol["sources"]["random_prefix_attempts"][str(seed)])
        canonical_target = int(
            protocol["sources"]["canonical_unique_targets"][str(seed)]
        )
        random, random_path = _load_random_ledger(
            args.random_runs_dir.resolve(), seed, random_cutoff, canonical_target
        )
        inputs = (
            ("transformer", transformer, _model_indices(), transformer_path),
            ("random", random, np.arange(random_cutoff, dtype=np.int64), random_path),
        )
        for method, records, indices, source_path in inputs:
            representatives = _representative_indices(records, indices)
            reserve = _curated_reserve(
                records,
                representatives,
                method=method,
                seed=seed,
                protocol=protocol,
            )
            sources.append(
                {
                    "method": method,
                    "seed": seed,
                    "path": str(source_path),
                    "sha256": _sha256(source_path),
                    "attempt_limit": int(len(indices)),
                }
            )
            for reserve_rank, candidate in enumerate(reserve, start=1):
                parent_key = _register(
                    registry,
                    tuple(candidate.token_ids),
                    reference={
                        "role": "parent",
                        "method": method,
                        "seed": seed,
                        "reserve_rank": reserve_rank,
                        "expected_reward": float(candidate.reward),
                    },
                )
                subtree_keys: list[str] = []
                for subtree in _proper_subtrees(compile_formula(candidate.token_ids).expression):
                    names = _expression_names(subtree)
                    if len(names) > max_subtree_tokens:
                        continue
                    token_ids = tuple(FORMULA_VOCAB.encode(names))
                    subtree_keys.append(
                        _register(
                            registry,
                            token_ids,
                            reference={
                                "role": "subtree",
                                "method": method,
                                "seed": seed,
                                "parent_sequence_hash": parent_key,
                            },
                        )
                    )
                parents.append(
                    {
                        "method": method,
                        "seed": seed,
                        "reserve_rank": reserve_rank,
                        "sequence_hash": parent_key,
                        "expected_reward": float(candidate.reward),
                        "subtree_sequence_hashes": sorted(set(subtree_keys)),
                    }
                )
    registry_rows = [registry[key] for key in sorted(registry)]
    cpu_rows = _score_registry(
        registry_rows, view, targets, device=torch.device("cpu"), protocol=protocol
    )
    scores = {row["sequence_hash"]: row for row in cpu_rows}
    groups = _provisional_groups(parents, registry, scores, protocol)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol_copy = output_dir / "validation_protocol.json"
    shutil.copy2(args.protocol.resolve(), protocol_copy)
    _write_jsonl(output_dir / "cuda_registry.jsonl", registry_rows)
    _write_jsonl(output_dir / "cpu_scores.jsonl", cpu_rows)
    _write_json(output_dir / "parent_reserve.json", parents)
    _write_json(output_dir / "provisional_groups.json", groups)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": protocol["protocol_id"],
        "validation_run_approved": False,
        "validation_or_final_metrics_read": False,
        "train_view_id": view.manifest["train_view_id"],
        "train_view_end": view.dates[-1].date().isoformat(),
        "registry_count": len(registry_rows),
        "parent_reserve_count": len(parents),
        "source_ledgers": sources,
        "subperiod_scorer_days": {
            name: int(target.days) for name, target in targets.items()
        },
        "cuda_audit_complete": False,
    }
    _write_json(output_dir / "preparation_manifest.json", manifest)
    names = (
        "validation_protocol.json",
        "cuda_registry.jsonl",
        "cpu_scores.jsonl",
        "parent_reserve.json",
        "provisional_groups.json",
        "preparation_manifest.json",
    )
    _write_sha256s(output_dir, names)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2))


def _audit_cuda(args: argparse.Namespace, protocol: dict[str, Any]) -> None:
    if args.registry is None or args.train_view_dir is None:
        raise ValueError("audit-cuda requires --registry and --train-view-dir")
    if not torch.cuda.is_available():
        raise RuntimeError("Validation CUDA audit requires a CUDA device")
    actual_device = torch.cuda.get_device_name(torch.device("cuda"))
    expected_device = str(protocol["numerical_gate"]["cuda_audit_gpu_class"])
    normalized_expected = "".join(
        character for character in expected_device.lower() if character.isalnum()
    )
    normalized_actual = "".join(
        character for character in actual_device.lower() if character.isalnum()
    )
    if normalized_expected not in normalized_actual:
        raise RuntimeError(
            f"Validation CUDA audit requires {expected_device}; found {actual_device}"
        )
    view = _load_train_view(args.train_view_dir.resolve(), protocol)
    targets = _build_targets(view, protocol)
    registry = _load_jsonl(args.registry.resolve())
    rows = _score_registry(
        registry, view, targets, device=torch.device("cuda"), protocol=protocol
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "cuda_scores.jsonl", rows)
    manifest = {
        "schema_version": "etf-v3a-validation-cuda-audit-v1",
        "protocol_id": protocol["protocol_id"],
        "validation_or_final_metrics_read": False,
        "train_view_id": view.manifest["train_view_id"],
        "train_view_end": view.dates[-1].date().isoformat(),
        "registry_sha256": _sha256(args.registry.resolve()),
        "registry_count": len(registry),
        "cuda_device": actual_device,
    }
    _write_json(output_dir / "cuda_audit_manifest.json", manifest)
    _write_sha256s(output_dir, ("cuda_scores.jsonl", "cuda_audit_manifest.json"))
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2))


def _verify_sha256s(path: Path) -> None:
    for line in (path / "SHA256SUMS").read_text(encoding="ascii").splitlines():
        expected, name = line.split("  ", 1)
        if _sha256(path / name) != expected:
            raise RuntimeError(f"Validation artifact SHA mismatch: {path / name}")


def _finalize(args: argparse.Namespace, protocol: dict[str, Any]) -> None:
    if args.preparation_dir is None or args.cuda_dir is None:
        raise ValueError("finalize requires --preparation-dir and --cuda-dir")
    preparation_dir = args.preparation_dir.resolve()
    cuda_dir = args.cuda_dir.resolve()
    _verify_sha256s(preparation_dir)
    _verify_sha256s(cuda_dir)
    preparation_manifest = json.loads(
        (preparation_dir / "preparation_manifest.json").read_text(encoding="utf-8")
    )
    cuda_manifest = json.loads(
        (cuda_dir / "cuda_audit_manifest.json").read_text(encoding="utf-8")
    )
    if (
        preparation_manifest["protocol_id"] != protocol["protocol_id"]
        or cuda_manifest["protocol_id"] != protocol["protocol_id"]
        or preparation_manifest["validation_or_final_metrics_read"] is not False
        or cuda_manifest["validation_or_final_metrics_read"] is not False
    ):
        raise RuntimeError("Validation finalization identity or data boundary mismatch")
    registry_rows = _load_jsonl(preparation_dir / "cuda_registry.jsonl")
    if cuda_manifest["registry_sha256"] != _sha256(
        preparation_dir / "cuda_registry.jsonl"
    ):
        raise RuntimeError("Validation CUDA audit used a different registry")
    cpu_rows = _load_jsonl(preparation_dir / "cpu_scores.jsonl")
    cuda_rows = _load_jsonl(cuda_dir / "cuda_scores.jsonl")
    registry = {row["sequence_hash"]: row for row in registry_rows}
    cpu = {row["sequence_hash"]: row for row in cpu_rows}
    cuda = {row["sequence_hash"]: row for row in cuda_rows}
    if set(registry) != set(cpu) or set(registry) != set(cuda):
        raise RuntimeError("Validation CPU/CUDA registry keys differ")
    parents = json.loads(
        (preparation_dir / "parent_reserve.json").read_text(encoding="utf-8")
    )
    gates = []
    tolerance = float(protocol["numerical_gate"]["full_reward_abs_tolerance"])
    for key in sorted(registry):
        cpu_full = float(cpu[key]["scores"]["full"]["reward"])
        cuda_full = float(cuda[key]["scores"]["full"]["reward"])
        gates.append(
            {
                "sequence_hash": key,
                "cpu_quality_valid": bool(cpu[key]["quality_valid"]),
                "cuda_quality_valid": bool(cuda[key]["quality_valid"]),
                "full_reward_abs_difference": abs(cpu_full - cuda_full),
                "full_reward_within_tolerance": abs(cpu_full - cuda_full) <= tolerance,
                "topk_exact": cpu[key]["full_topk_sha256"]
                == cuda[key]["full_topk_sha256"],
                "eligible": _numerically_eligible(
                    registry[key], cpu[key], protocol, cuda[key]
                ),
            }
        )
    groups = _provisional_groups(
        parents, registry, cpu, protocol, cuda_scores=cuda
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "numerical_gates.jsonl", gates)
    _write_json(output_dir / "final_groups.json", groups)
    manifest = {
        "schema_version": "etf-v3a-validation-candidate-final-v1",
        "protocol_id": protocol["protocol_id"],
        "validation_run_approved": False,
        "validation_or_final_metrics_read": False,
        "train_view_id": preparation_manifest["train_view_id"],
        "train_view_end": preparation_manifest["train_view_end"],
        "registry_sha256": cuda_manifest["registry_sha256"],
        "registry_count": len(registry),
        "numerically_eligible_count": sum(row["eligible"] for row in gates),
        "cuda_device": cuda_manifest["cuda_device"],
        "cuda_audit_complete": True,
    }
    _write_json(output_dir / "final_candidate_manifest.json", manifest)
    _write_sha256s(
        output_dir,
        ("numerical_gates.jsonl", "final_groups.json", "final_candidate_manifest.json"),
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2))


def main() -> None:
    args = _parse_args()
    torch.set_num_threads(min(8, torch.get_num_threads()))
    protocol = _load_protocol(args.protocol.resolve())
    if args.mode == "prepare-cpu":
        _prepare_cpu(args, protocol)
    elif args.mode == "audit-cuda":
        _audit_cuda(args, protocol)
    else:
        _finalize(args, protocol)


if __name__ == "__main__":
    main()