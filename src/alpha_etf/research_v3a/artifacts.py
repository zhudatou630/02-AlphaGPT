"""Strict formula artifact serialization for V3A."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from alpha_etf.research_v3a.candidates import (
    CANONICALIZER_VERSION,
    CANDIDATE_FUNNEL_VERSION,
    CandidateConfig,
    CandidateRecord,
    CandidateSelection,
    candidate_record_from_dict,
    canonicalize_expression,
    expression_hash,
)
from alpha_etf.research_v3a.language import FORMULA_VOCAB, compile_formula
from alpha_etf.research_v3a.spec import validate_research_spec


ARTIFACT_SCHEMA_VERSION = "etf-v3a-formula-artifact-v1"
FUNNEL_ARTIFACT_SCHEMA_VERSION = "etf-v3a-training-funnel-v1"


def _artifact_id(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "v3a-formula-" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def build_formula_artifact(
    record: CandidateRecord, *, research_spec: dict[str, Any], created_at: str
) -> dict[str, Any]:
    validate_research_spec(research_spec)
    if not math.isfinite(record.reward):
        raise RuntimeError("V3A artifact reward must be finite")
    payload = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "formula_id": record.formula_id,
        "source": record.source,
        "token_ids": list(record.token_ids),
        "token_names": list(record.token_names),
        "token_len": record.token_len,
        "vocab_version": FORMULA_VOCAB.version,
        "expression": record.expression,
        "canonicalizer_version": CANONICALIZER_VERSION,
        "canonical_expression": record.canonical_expression,
        "formula_hash": record.formula_hash,
        "reward": record.reward,
        "train_summary": record.train_summary,
        "cluster_id": record.cluster_id,
        "first_attempt_index": record.first_attempt_index,
        "attempt_count": record.attempt_count,
        "best_attempt_index": record.best_attempt_index,
        "research_spec_id": research_spec["research_spec_id"],
        "research_spec": research_spec,
        "created_at": created_at,
    }
    return {**payload, "artifact_id": _artifact_id(payload)}


def validate_formula_artifact(
    artifact: dict[str, Any], *, research_spec: dict[str, Any]
) -> CandidateRecord:
    validate_research_spec(research_spec)
    if artifact.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise RuntimeError(f"V3A artifact schema mismatch: {artifact.get('schema_version')}")
    if artifact.get("research_spec_id") != research_spec.get("research_spec_id"):
        raise RuntimeError("V3A artifact ResearchSpec id mismatch")
    if artifact.get("research_spec") != research_spec:
        raise RuntimeError("V3A artifact ResearchSpec mismatch")
    if artifact.get("vocab_version") != FORMULA_VOCAB.version:
        raise RuntimeError("V3A artifact vocab mismatch")
    if artifact.get("canonicalizer_version") != CANONICALIZER_VERSION:
        raise RuntimeError("V3A artifact canonicalizer mismatch")
    token_ids = tuple(int(item) for item in artifact.get("token_ids", []))
    if int(artifact.get("token_len", -1)) != len(token_ids):
        raise RuntimeError("V3A artifact token length mismatch")
    token_names = tuple(FORMULA_VOCAB.decode(token_ids))
    if tuple(artifact.get("token_names", ())) != token_names:
        raise RuntimeError("V3A artifact token names mismatch")
    compiled = compile_formula(token_ids)
    canonical = canonicalize_expression(compiled.expression)
    formula_hash = expression_hash(canonical)
    if artifact.get("expression") != compiled.expression.to_dict():
        raise RuntimeError("V3A artifact expression mismatch")
    if artifact.get("canonical_expression") != canonical.to_dict():
        raise RuntimeError("V3A artifact canonical expression mismatch")
    if artifact.get("formula_hash") != formula_hash:
        raise RuntimeError("V3A artifact formula hash mismatch")
    payload = dict(artifact)
    actual_artifact_id = str(payload.pop("artifact_id", ""))
    if actual_artifact_id != _artifact_id(payload):
        raise RuntimeError("V3A artifact id mismatch")
    reward = float(artifact["reward"])
    if not math.isfinite(reward):
        raise RuntimeError("V3A artifact reward must be finite")
    return CandidateRecord(
        formula_id=str(artifact["formula_id"]),
        source=str(artifact["source"]),
        token_ids=token_ids,
        token_names=token_names,
        token_len=len(token_ids),
        expression=compiled.expression.to_dict(),
        canonical_expression=canonical.to_dict(),
        formula_hash=formula_hash,
        reward=reward,
        train_summary=dict(artifact.get("train_summary", {})),
        cluster_id=str(artifact.get("cluster_id", "")),
        first_attempt_index=int(artifact.get("first_attempt_index", -1)),
        attempt_count=int(artifact.get("attempt_count", 1)),
        best_attempt_index=int(artifact.get("best_attempt_index", -1)),
    )


def build_training_funnel_artifact(
    selection: CandidateSelection,
    *,
    research_spec: dict[str, Any],
    candidate_config: CandidateConfig,
    created_at: str,
) -> dict[str, Any]:
    validate_research_spec(research_spec)
    cluster_order: list[str] = []
    cluster_members: dict[str, list[str]] = {}
    representatives: dict[str, str] = {}
    for record in selection.clustered:
        if record.cluster_id not in cluster_members:
            cluster_order.append(record.cluster_id)
            cluster_members[record.cluster_id] = []
            representatives[record.cluster_id] = record.formula_hash
        cluster_members[record.cluster_id].append(record.formula_hash)
    clusters = [
        {
            "cluster_id": cluster_id,
            "representative_hash": representatives[cluster_id],
            "member_hashes": cluster_members[cluster_id],
        }
        for cluster_id in cluster_order
    ]
    records = [json.loads(json.dumps(record.to_dict(), ensure_ascii=False)) for record in selection.clustered]
    payload = {
        "schema_version": FUNNEL_ARTIFACT_SCHEMA_VERSION,
        "candidate_funnel_version": CANDIDATE_FUNNEL_VERSION,
        "candidate_config": candidate_config.to_dict(),
        "ordering": ["reward_desc", "token_len_asc", "formula_hash_asc"],
        "records": records,
        "clusters": clusters,
        "selected_formula_hashes": [record.formula_hash for record in selection.selected],
        "audit": selection.audit,
        "research_spec_id": research_spec["research_spec_id"],
        "research_spec": research_spec,
        "created_at": created_at,
    }
    return {**payload, "funnel_artifact_id": _artifact_id(payload).replace("v3a-formula", "v3a-funnel")}


def validate_training_funnel_artifact(
    artifact: dict[str, Any],
    *,
    research_spec: dict[str, Any],
    candidate_config: CandidateConfig,
) -> None:
    validate_research_spec(research_spec)
    if artifact.get("schema_version") != FUNNEL_ARTIFACT_SCHEMA_VERSION:
        raise RuntimeError("V3A training-funnel artifact schema mismatch")
    if artifact.get("candidate_funnel_version") != CANDIDATE_FUNNEL_VERSION:
        raise RuntimeError("V3A training-funnel version mismatch")
    if artifact.get("candidate_config") != candidate_config.to_dict():
        raise RuntimeError("V3A training-funnel config mismatch")
    if artifact.get("research_spec_id") != research_spec.get("research_spec_id") or artifact.get(
        "research_spec"
    ) != research_spec:
        raise RuntimeError("V3A training-funnel ResearchSpec mismatch")
    records = artifact.get("records")
    clusters = artifact.get("clusters")
    selected = artifact.get("selected_formula_hashes")
    if not isinstance(records, list) or not isinstance(clusters, list) or not isinstance(selected, list):
        raise RuntimeError("V3A training-funnel payload is incomplete")
    record_hashes = [str(record.get("formula_hash", "")) for record in records]
    if len(record_hashes) != len(set(record_hashes)) or any(not value for value in record_hashes):
        raise RuntimeError("V3A training-funnel records are not unique")
    expected_clusters: list[dict[str, Any]] = []
    by_cluster: dict[str, list[str]] = {}
    for record in records:
        candidate_record_from_dict(record)
        cluster_id = str(record.get("cluster_id", ""))
        if not cluster_id:
            raise RuntimeError("V3A training-funnel record lacks cluster id")
        by_cluster.setdefault(cluster_id, []).append(str(record["formula_hash"]))
    for cluster_id, members in by_cluster.items():
        expected_clusters.append(
            {
                "cluster_id": cluster_id,
                "representative_hash": members[0],
                "member_hashes": members,
            }
        )
    if clusters != expected_clusters:
        raise RuntimeError("V3A training-funnel cluster membership mismatch")
    if (
        len(selected) != len(set(selected))
        or len(selected) > sum(candidate_config.final_quotas)
        or not set(selected).issubset(record_hashes)
    ):
        raise RuntimeError("V3A training-funnel selected formulas are invalid")
    audit = artifact.get("audit")
    if not isinstance(audit, dict):
        raise RuntimeError("V3A training-funnel audit is missing")
    records_by_hash = {
        str(record["formula_hash"]): candidate_record_from_dict(record)
        for record in records
    }
    selected_bucket_counts = [0, 0, 0]
    for formula_hash in selected:
        token_len = records_by_hash[formula_hash].token_len
        selected_bucket_counts[0 if token_len <= 5 else 1 if token_len <= 10 else 2] += 1
    if (
        int(audit.get("signal_unique_count", -1)) != len(records)
        or int(audit.get("cluster_count", -1)) != len(clusters)
        or int(audit.get("selected_count", -1)) != len(selected)
        or audit.get("selected_bucket_counts") != selected_bucket_counts
    ):
        raise RuntimeError("V3A training-funnel audit does not reconcile")
    payload = dict(artifact)
    actual_id = str(payload.pop("funnel_artifact_id", ""))
    expected_id = _artifact_id(payload).replace("v3a-formula", "v3a-funnel")
    if actual_id != expected_id:
        raise RuntimeError("V3A training-funnel artifact id mismatch")


def write_jsonl(path: Path, artifacts: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for artifact in artifacts:
            handle.write(json.dumps(artifact, ensure_ascii=False, sort_keys=True) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records