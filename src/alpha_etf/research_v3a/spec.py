"""Dataset identity, loading, and ResearchSpec helpers for V3A."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from alpha_etf.data.relative_price import RELATIVE_FEATURES, add_relative_ohlc, relative_panel


DATASET_SCHEMA_VERSION = "etf-v3a-dataset-v1"
RESEARCH_SCHEMA_VERSION = "etf-v3a-research-spec-v1"
EXPECTED_UPSTREAM_V3_ID = "etf-relative-price-v3-989c5d80a4e0"
ABSOLUTE_FEATURES = ("open", "high", "low", "close")
MIN_UNIVERSE = 10
EFFECTIVE_START = "2016-08-09"
TRAIN_RANGE = ("2016-08-09", "2021-12-31")
VALIDATION_RANGE = ("2022-01-01", "2022-12-31")
FINAL_OOS_RANGE = ("2023-01-01", None)
PRICE_ADJUSTMENT = "multiplicative_event_qfq"
RELATIVE_TRANSFORM = "ohlc_div_previous_source_row_adjusted_close_minus_one"
TRADABLE_MASK_SEMANTICS = (
    "positive reconciled volume and amount, finite relative OHLC, and not first source row; "
    "nontradable research values are NaN"
)
MANIFEST_REQUIRED_FIELDS = {
    "schema_version",
    "panel_file",
    "panel_sha256",
    "upstream_v3_manifest_file",
    "upstream_v3_manifest_sha256",
    "upstream_v3_dataset_id",
    "upstream_v3_panel_sha256",
    "governance_file",
    "governance_sha256",
    "universe_file",
    "universe_sha256",
    "symbols",
    "absolute_features",
    "relative_features",
    "absolute_shape",
    "relative_shape",
    "mask_shape",
    "date_start",
    "date_end",
    "date_count",
    "tradable_observations",
    "effective_start",
    "min_universe",
    "price_adjustment",
    "relative_transform",
    "tradable_mask",
    "flow_features_exposed",
    "dataset_fingerprint",
    "dataset_id",
}


@dataclass(frozen=True)
class V3APanel:
    absolute_ohlc: np.ndarray
    relative_ohlc: np.ndarray
    tradable_mask: np.ndarray
    symbols: np.ndarray
    dates: pd.DatetimeIndex
    factor_values: np.ndarray | None = None
    factor_names: tuple[str, ...] = ()

    def symbol_index(self, symbol: str) -> int:
        matches = np.where(self.symbols == str(symbol))[0]
        if len(matches) != 1:
            raise KeyError(f"Symbol not found: {symbol}")
        return int(matches[0])

    def absolute(self, feature: str) -> np.ndarray:
        try:
            index = ABSOLUTE_FEATURES.index(feature)
        except ValueError as exc:
            raise KeyError(f"Absolute feature not found: {feature}") from exc
        return self.absolute_ohlc[:, index, :]

    def relative(self, feature: str) -> np.ndarray:
        try:
            index = RELATIVE_FEATURES.index(feature)
        except ValueError as exc:
            raise KeyError(f"Relative feature not found: {feature}") from exc
        return self.relative_ohlc[:, index, :]

    def with_factor_cache(self, values: np.ndarray, names: tuple[str, ...]) -> "V3APanel":
        if values.shape[1:] != self.tradable_mask.shape:
            raise ValueError(
                f"Factor cache shape mismatch: {values.shape} vs (*,{self.tradable_mask.shape})"
            )
        if values.shape[0] != len(names):
            raise ValueError(f"Factor names differ from values: {len(names)} != {values.shape[0]}")
        return replace(self, factor_values=values, factor_names=tuple(names))

    def factor(self, name: str) -> np.ndarray:
        if self.factor_values is None:
            raise RuntimeError("Factor cache has not been attached")
        try:
            index = self.factor_names.index(name)
        except ValueError as exc:
            raise KeyError(f"Factor not found: {name}") from exc
        return self.factor_values[index]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: dict[str, Any]) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_panel_arrays(
    governance: pd.DataFrame,
    *,
    symbols: list[str],
    upstream_relative: np.ndarray,
    upstream_mask: np.ndarray,
    upstream_dates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    required = {
        "symbol",
        "date",
        "tradable",
        *(f"event_qfq_{feature}" for feature in ABSOLUTE_FEATURES),
    }
    missing = required - set(governance.columns)
    if missing:
        raise ValueError(f"V3 governance data missing columns: {sorted(missing)}")

    frame = governance.copy()
    frame["symbol"] = frame["symbol"].astype(str)
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    duplicates = frame.duplicated(["symbol", "date"], keep=False)
    if duplicates.any():
        examples = frame.loc[duplicates, ["symbol", "date"]].head(10).to_dict("records")
        raise ValueError(f"V3 governance data has duplicate symbol/date rows: {examples}")

    computed_relative_frame = add_relative_ohlc(frame)
    computed_relative, computed_mask, computed_dates = relative_panel(
        computed_relative_frame, symbols
    )
    expected_dates = pd.DatetimeIndex(pd.to_datetime(upstream_dates.astype(str)))
    if not computed_dates.equals(expected_dates):
        raise ValueError("V3A governance dates differ from upstream V3 panel")
    if computed_relative.shape != upstream_relative.shape:
        raise ValueError(
            f"V3A relative shape differs from upstream: {computed_relative.shape} != "
            f"{upstream_relative.shape}"
        )
    if not np.array_equal(computed_mask, upstream_mask):
        mismatch = int(np.count_nonzero(computed_mask != upstream_mask))
        raise ValueError(f"V3A relative mask differs from upstream in {mismatch} cells")
    np.testing.assert_allclose(
        computed_relative,
        upstream_relative,
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
        err_msg="V3A relative values differ from upstream V3 panel",
    )

    assets = len(symbols)
    dates = len(computed_dates)
    absolute = np.full((assets, len(ABSOLUTE_FEATURES), dates), np.nan, dtype=np.float64)
    symbol_index = {symbol: index for index, symbol in enumerate(symbols)}
    date_index = {date: index for index, date in enumerate(computed_dates)}
    for row in frame.itertuples(index=False):
        symbol = str(row.symbol)
        if symbol not in symbol_index:
            continue
        asset = symbol_index[symbol]
        date = date_index[pd.Timestamp(row.date)]
        if not bool(upstream_mask[asset, date]):
            continue
        values = np.asarray(
            [getattr(row, f"event_qfq_{feature}") for feature in ABSOLUTE_FEATURES],
            dtype=np.float64,
        )
        if not np.isfinite(values).all() or not (values > 0).all():
            raise ValueError(f"Invalid V3A absolute OHLC for {symbol} at {row.date}")
        absolute[asset, :, date] = values

    sample_view = absolute.transpose(0, 2, 1)
    if not np.isfinite(sample_view[upstream_mask]).all():
        raise ValueError("V3A tradable absolute OHLC must be finite")
    if not np.isnan(sample_view[~upstream_mask]).all():
        raise ValueError("V3A nontradable absolute OHLC must be NaN")
    return absolute, computed_relative, upstream_mask.astype(bool).copy(), computed_dates


def build_dataset_manifest(identity: dict[str, Any]) -> dict[str, Any]:
    if identity.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise ValueError(f"V3A dataset schema mismatch: {identity.get('schema_version')}")
    fingerprint = canonical_sha256(identity)
    return {
        **identity,
        "dataset_fingerprint": fingerprint,
        "dataset_id": f"etf-v3a-{fingerprint[:12]}",
    }


def load_dataset_manifest(dataset_dir: Path) -> dict[str, Any]:
    manifest_path = dataset_dir / "dataset_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing V3A dataset manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    missing_fields = MANIFEST_REQUIRED_FIELDS - set(manifest)
    if missing_fields:
        raise RuntimeError(f"V3A manifest missing fields: {sorted(missing_fields)}")
    if manifest.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise RuntimeError(f"V3A dataset schema mismatch: {manifest.get('schema_version')}")
    if manifest.get("upstream_v3_dataset_id") != EXPECTED_UPSTREAM_V3_ID:
        raise RuntimeError(
            f"V3A upstream dataset mismatch: {manifest.get('upstream_v3_dataset_id')}"
        )
    if tuple(manifest.get("absolute_features", ())) != ABSOLUTE_FEATURES:
        raise RuntimeError(f"V3A absolute feature mismatch: {manifest.get('absolute_features')}")
    if tuple(manifest.get("relative_features", ())) != RELATIVE_FEATURES:
        raise RuntimeError(f"V3A relative feature mismatch: {manifest.get('relative_features')}")
    if manifest.get("flow_features_exposed") is not False:
        raise RuntimeError("V3A dataset must not expose flow features")
    frozen_values = {
        "effective_start": EFFECTIVE_START,
        "min_universe": MIN_UNIVERSE,
        "price_adjustment": PRICE_ADJUSTMENT,
        "relative_transform": RELATIVE_TRANSFORM,
        "tradable_mask": TRADABLE_MASK_SEMANTICS,
    }
    for field, expected in frozen_values.items():
        if manifest.get(field) != expected:
            raise RuntimeError(
                f"V3A manifest {field} mismatch: {manifest.get(field)!r} != {expected!r}"
            )
    hash_fields = (
        "panel_sha256",
        "upstream_v3_manifest_sha256",
        "upstream_v3_panel_sha256",
        "governance_sha256",
        "universe_sha256",
        "dataset_fingerprint",
    )
    for field in hash_fields:
        value = str(manifest.get(field, ""))
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise RuntimeError(f"V3A manifest {field} is not a lowercase SHA-256")
    symbols = manifest.get("symbols")
    if not isinstance(symbols, list) or not symbols or len(symbols) != len(set(symbols)):
        raise RuntimeError("V3A manifest symbols must be a nonempty unique list")
    for field, expected_length in (
        ("absolute_shape", 3),
        ("relative_shape", 3),
        ("mask_shape", 2),
    ):
        value = manifest.get(field)
        if not isinstance(value, list) or len(value) != expected_length or not all(
            isinstance(item, int) and item > 0 for item in value
        ):
            raise RuntimeError(f"V3A manifest {field} is invalid: {value}")

    panel_path = dataset_dir / str(manifest.get("panel_file", ""))
    if not panel_path.exists():
        raise FileNotFoundError(f"Missing V3A panel: {panel_path}")
    actual_panel_hash = sha256_file(panel_path)
    if actual_panel_hash != manifest.get("panel_sha256"):
        raise RuntimeError(
            f"V3A panel hash mismatch: {actual_panel_hash} != {manifest.get('panel_sha256')}"
        )

    identity = dict(manifest)
    expected_fingerprint = str(identity.pop("dataset_fingerprint", ""))
    expected_id = str(identity.pop("dataset_id", ""))
    actual_fingerprint = canonical_sha256(identity)
    if actual_fingerprint != expected_fingerprint:
        raise RuntimeError(
            f"V3A dataset fingerprint mismatch: {actual_fingerprint} != {expected_fingerprint}"
        )
    actual_id = f"etf-v3a-{actual_fingerprint[:12]}"
    if expected_id != actual_id:
        raise RuntimeError(f"V3A dataset id mismatch: {expected_id} != {actual_id}")
    return manifest


def load_panel(dataset_dir: Path) -> V3APanel:
    manifest = load_dataset_manifest(dataset_dir)
    panel_path = dataset_dir / str(manifest["panel_file"])
    with np.load(panel_path, allow_pickle=False) as data:
        required = {
            "absolute_ohlc",
            "relative_ohlc",
            "tradable_mask",
            "symbols",
            "dates",
            "absolute_features",
            "relative_features",
        }
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"V3A panel missing keys: {sorted(missing)}")
        absolute = data["absolute_ohlc"].astype(np.float64)
        relative = data["relative_ohlc"].astype(np.float64)
        mask = data["tradable_mask"].astype(bool)
        symbols = data["symbols"].astype(str)
        dates = data["dates"].astype(str)
        absolute_features = tuple(data["absolute_features"].astype(str))
        relative_features = tuple(data["relative_features"].astype(str))

    if absolute_features != ABSOLUTE_FEATURES:
        raise ValueError(f"V3A panel absolute features differ: {absolute_features}")
    if relative_features != RELATIVE_FEATURES:
        raise ValueError(f"V3A panel relative features differ: {relative_features}")
    expected_shape = (len(symbols), len(ABSOLUTE_FEATURES), len(dates))
    if absolute.shape != expected_shape or relative.shape != expected_shape:
        raise ValueError(
            f"V3A panel shape mismatch: absolute={absolute.shape}, relative={relative.shape}, "
            f"expected={expected_shape}"
        )
    if mask.shape != (len(symbols), len(dates)):
        raise ValueError(f"V3A mask shape mismatch: {mask.shape}")
    if list(mask.shape) != list(manifest.get("mask_shape", [])):
        raise ValueError("V3A mask shape differs from manifest")
    if list(symbols) != list(manifest.get("symbols", [])):
        raise ValueError("V3A panel symbols differ from manifest")
    if list(absolute.shape) != list(manifest.get("absolute_shape", [])):
        raise ValueError("V3A absolute shape differs from manifest")
    if list(relative.shape) != list(manifest.get("relative_shape", [])):
        raise ValueError("V3A relative shape differs from manifest")
    if len(dates) != int(manifest.get("date_count", -1)):
        raise ValueError("V3A date count differs from manifest")
    if dates[0] != manifest.get("date_start") or dates[-1] != manifest.get("date_end"):
        raise ValueError("V3A date range differs from manifest")
    if int(mask.sum()) != int(manifest.get("tradable_observations", -1)):
        raise ValueError("V3A tradable count differs from manifest")

    absolute_view = absolute.transpose(0, 2, 1)
    relative_view = relative.transpose(0, 2, 1)
    if not np.isfinite(absolute_view[mask]).all() or not np.isfinite(relative_view[mask]).all():
        raise ValueError("V3A tradable values must be finite")
    if not np.isnan(absolute_view[~mask]).all() or not np.isnan(relative_view[~mask]).all():
        raise ValueError("V3A nontradable values must be NaN")
    if not (absolute_view[mask] > 0).all():
        raise ValueError("V3A tradable absolute prices must be positive")

    return V3APanel(
        absolute_ohlc=absolute,
        relative_ohlc=relative,
        tradable_mask=mask,
        symbols=symbols,
        dates=pd.DatetimeIndex(pd.to_datetime(dates)),
    )


def build_research_spec(
    *,
    dataset_manifest: dict[str, Any],
    factor_config: dict[str, Any],
    vocab_config: dict[str, Any],
    scorer_config: dict[str, Any],
    canonicalizer_config: dict[str, Any],
    candidate_config: dict[str, Any],
    code_commit: str,
    code_fingerprint: str,
) -> dict[str, Any]:
    base = {
        "schema_version": RESEARCH_SCHEMA_VERSION,
        "research_id": "etf-v3a-single-formula-selection",
        "dataset": {
            "dataset_id": dataset_manifest["dataset_id"],
            "dataset_fingerprint": dataset_manifest["dataset_fingerprint"],
            "panel_sha256": dataset_manifest["panel_sha256"],
            "upstream_v3_dataset_id": dataset_manifest["upstream_v3_dataset_id"],
            "upstream_v3_panel_sha256": dataset_manifest["upstream_v3_panel_sha256"],
            "symbols": list(dataset_manifest["symbols"]),
            "date_start": dataset_manifest["date_start"],
            "date_end": dataset_manifest["date_end"],
        },
        "splits": {
            "train": list(TRAIN_RANGE),
            "validation": list(VALIDATION_RANGE),
            "final_oos": list(FINAL_OOS_RANGE),
            "effective_start": EFFECTIVE_START,
            "min_universe": MIN_UNIVERSE,
        },
        "factor_config": factor_config,
        "vocab_config": vocab_config,
        "scorer_config": scorer_config,
        "canonicalizer_config": canonicalizer_config,
        "candidate_config": candidate_config,
        "code_commit": code_commit,
        "code_fingerprint": code_fingerprint,
    }
    return {**base, "research_spec_id": canonical_sha256(base)}


def validate_research_spec(spec: dict[str, Any]) -> None:
    if spec.get("schema_version") != RESEARCH_SCHEMA_VERSION:
        raise RuntimeError(f"V3A ResearchSpec schema mismatch: {spec.get('schema_version')}")
    payload = dict(spec)
    expected = str(payload.pop("research_spec_id", ""))
    actual = canonical_sha256(payload)
    if expected != actual:
        raise RuntimeError(f"V3A ResearchSpec id mismatch: {expected} != {actual}")
    from alpha_etf.research_v3a.candidates import (
        CandidateConfig,
        canonicalizer_config,
    )
    from alpha_etf.research_v3a.factors import FACTOR_NAMES, FACTOR_SPEC_VERSION, WINDOWS
    from alpha_etf.research_v3a.language import (
        FORMULA_VOCAB,
        GRAMMAR_VERSION,
        MAX_FORMULA_TOKENS,
    )
    from alpha_etf.research_v3a.scoring import ScorerConfig
    from alpha_etf.research_v3a.sampling import SamplingConfig

    if spec.get("research_id") != "etf-v3a-single-formula-selection":
        raise RuntimeError("V3A ResearchSpec research id mismatch")
    if spec.get("splits") != {
        "train": list(TRAIN_RANGE),
        "validation": list(VALIDATION_RANGE),
        "final_oos": list(FINAL_OOS_RANGE),
        "effective_start": EFFECTIVE_START,
        "min_universe": MIN_UNIVERSE,
    }:
        raise RuntimeError("V3A ResearchSpec split mismatch")
    factor = spec.get("factor_config", {})
    from alpha_etf.research_v3a.factors import factor_config as expected_factor_config

    if factor != expected_factor_config():
        raise RuntimeError("V3A ResearchSpec factor config mismatch")
    vocab = spec.get("vocab_config", {})
    expected_vocab = FORMULA_VOCAB.to_config()
    if any(vocab.get(key) != value for key, value in expected_vocab.items()):
        raise RuntimeError("V3A ResearchSpec vocab config mismatch")
    if vocab.get("grammar_version") != GRAMMAR_VERSION or vocab.get(
        "max_formula_tokens"
    ) != MAX_FORMULA_TOKENS:
        raise RuntimeError("V3A ResearchSpec grammar mismatch")
    if vocab.get("sampling") != SamplingConfig().to_dict():
        raise RuntimeError("V3A ResearchSpec sampling protocol mismatch")
    if spec.get("scorer_config") != ScorerConfig().to_dict():
        raise RuntimeError("V3A ResearchSpec scorer config mismatch")
    canonicalizer = spec.get("canonicalizer_config", {})
    if canonicalizer != canonicalizer_config():
        raise RuntimeError("V3A ResearchSpec canonicalizer mismatch")
    if spec.get("candidate_config") != CandidateConfig().to_dict():
        raise RuntimeError("V3A ResearchSpec candidate config mismatch")
    dataset = spec.get("dataset", {})
    if (
        not str(dataset.get("dataset_id", "")).startswith("etf-v3a-")
        or dataset.get("upstream_v3_dataset_id") != EXPECTED_UPSTREAM_V3_ID
        or len(dataset.get("symbols", [])) != 35
    ):
        raise RuntimeError("V3A ResearchSpec dataset identity mismatch")