"""Dataset identity and loading helpers for the V2 research line."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from alpha_etf.gpt.vocab import FormulaVocab
from alpha_etf.panel import MarketPanel


DATASET_SCHEMA_VERSION = "etf-price-dataset-v2"
RESEARCH_SCHEMA_VERSION = "etf-research-spec-v2"
PRICE_FEATURES = ("open", "high", "low", "close")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: dict[str, Any]) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_dataset_manifest(dataset_dir: Path) -> dict[str, Any]:
    manifest_path = dataset_dir / "dataset_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing V2 dataset manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise RuntimeError(f"V2 dataset schema mismatch: {manifest.get('schema_version')}")
    if tuple(manifest.get("features", ())) != PRICE_FEATURES:
        raise RuntimeError(f"V2 feature schema mismatch: {manifest.get('features')}")
    panel_path = dataset_dir / str(manifest["panel_file"])
    actual_hash = sha256_file(panel_path)
    if actual_hash != manifest.get("panel_sha256"):
        raise RuntimeError(f"V2 panel hash mismatch: {actual_hash} != {manifest.get('panel_sha256')}")
    identity = dict(manifest)
    expected_fingerprint = str(identity.pop("dataset_fingerprint"))
    identity.pop("dataset_id", None)
    actual_fingerprint = canonical_sha256(identity)
    if actual_fingerprint != expected_fingerprint:
        raise RuntimeError(
            f"V2 dataset fingerprint mismatch: {actual_fingerprint} != {expected_fingerprint}"
        )
    expected_id = f"etf-price-event-v2-{expected_fingerprint[:12]}"
    if manifest.get("dataset_id") != expected_id:
        raise RuntimeError(f"V2 dataset id mismatch: {manifest.get('dataset_id')} != {expected_id}")
    return manifest


def load_price_panel(dataset_dir: Path) -> MarketPanel:
    manifest = load_dataset_manifest(dataset_dir)
    panel_path = dataset_dir / str(manifest["panel_file"])
    with np.load(panel_path, allow_pickle=False) as data:
        required = {"values", "mask", "symbols", "features", "dates"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"V2 panel missing keys: {sorted(missing)}")
        values = data["values"].astype(float)
        mask = data["mask"].astype(bool)
        symbols = data["symbols"].astype(str)
        features = data["features"].astype(str)
        dates = data["dates"].astype(str)
    if tuple(features) != PRICE_FEATURES:
        raise ValueError(f"V2 panel features differ from contract: {tuple(features)}")
    if values.shape != (len(symbols), len(features), len(dates)):
        raise ValueError(f"V2 panel shape mismatch: {values.shape}")
    if mask.shape != (len(symbols), len(dates)):
        raise ValueError(f"V2 mask shape mismatch: {mask.shape}")
    if list(symbols) != list(manifest["symbols"]):
        raise ValueError("V2 panel symbols differ from manifest")
    if len(dates) != int(manifest["date_count"]):
        raise ValueError("V2 panel date count differs from manifest")
    if list(values.shape) != list(manifest["panel_shape"]):
        raise ValueError("V2 panel shape differs from manifest")
    if dates[0] != str(manifest["date_start"]) or dates[-1] != str(manifest["date_end"]):
        raise ValueError("V2 panel date range differs from manifest")
    if int(mask.sum()) != int(manifest["tradable_observations"]):
        raise ValueError("V2 panel tradable count differs from manifest")
    sample_view = values.transpose(0, 2, 1)
    if not np.isfinite(sample_view[mask]).all():
        raise ValueError("V2 tradable panel values must be finite")
    if not np.isnan(sample_view[~mask]).all():
        raise ValueError("V2 nontradable panel values must be NaN")
    return MarketPanel(
        raw_values=values.copy(),
        qfq_values=values,
        mask=mask,
        symbols=symbols,
        features=features,
        dates=pd.DatetimeIndex(pd.to_datetime(dates)),
    )


def build_research_spec(
    *,
    dataset_manifest: dict[str, Any],
    vocab: FormulaVocab,
    scorer_config: dict[str, Any],
    code_commit: str,
    code_fingerprint: str,
) -> dict[str, Any]:
    return {
        "schema_version": RESEARCH_SCHEMA_VERSION,
        "research_id": "etf-price-only-v2",
        "dataset": {
            "dataset_id": dataset_manifest["dataset_id"],
            "dataset_fingerprint": dataset_manifest["dataset_fingerprint"],
            "panel_sha256": dataset_manifest["panel_sha256"],
            "features": list(dataset_manifest["features"]),
            "price_adjustment": dataset_manifest["price_adjustment"],
            "tradable_mask": dataset_manifest["tradable_mask"],
            "date_start": dataset_manifest["date_start"],
            "date_end": dataset_manifest["date_end"],
            "symbols": list(dataset_manifest["symbols"]),
        },
        "vocab": {
            "version": vocab.version,
            "token_names": list(vocab.token_names),
        },
        "scorer_config": scorer_config,
        "code_commit": code_commit,
        "code_fingerprint": code_fingerprint,
    }
