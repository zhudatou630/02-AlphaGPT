#!/usr/bin/env python3
"""Run the one-shot SW2021 L2 dynamic long-history ABS(ROC40) test."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from alpha_etf.research_v3a.validation import ValidationConfig, run_formula_validation  # noqa: E402
from alpha_etf.sw_industry.l2_spec import (  # noqa: E402
    load_dataset_manifest,
    load_panel,
)
from alpha_etf.sw_industry.l2_validation import (  # noqa: E402
    build_absroc40_signal,
    load_protocol,
    validate_protocol_dataset,
)
from alpha_etf.sw_industry.spec import sha256_file  # noqa: E402
from scripts.sw_industry.run_absroc40_validation import (  # noqa: E402
    annual_returns,
    enrich_and_reconcile_trades,
    path_metrics,
    trade_metrics,
    write_text_synced,
)


RESULT_SCHEMA_VERSION = "sw2021-l2-absroc40-dynamic-long-history-result-v1"
FORMULA_ID = "abs_roc_40"
CLAIM_ROOT = ROOT / "configs/validation_claims"
CLAIM_POLICY = (
    "versionable O_CREAT|O_EXCL claim under configs/validation_claims before panel load "
    "and result computation"
)
CODE_FINGERPRINT_PATHS = (
    ROOT / "src/alpha_etf/research_v3a/factors.py",
    ROOT / "src/alpha_etf/research_v3a/validation.py",
    ROOT / "src/alpha_etf/sw_industry/spec.py",
    ROOT / "src/alpha_etf/sw_industry/l2_spec.py",
    ROOT / "src/alpha_etf/sw_industry/l2_validation.py",
    ROOT / "scripts/sw_industry/run_absroc40_validation.py",
    Path(__file__).resolve(),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def code_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in CODE_FINGERPRINT_PATHS:
        digest.update(str(path.relative_to(ROOT)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def project_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def verify_receipt(
    receipt_path: Path,
    protocol_path: Path,
    dataset_dir: Path,
    protocol: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema_version") != "sw2021-l2-dynamic-long-history-receipt-v1":
        raise RuntimeError("Unexpected L2 frozen receipt schema")
    if receipt.get("research_status") != "FROZEN_BEFORE_ONE_SHOT_RESULT":
        raise RuntimeError("L2 receipt is not frozen before the result")
    if receipt.get("claim_policy") != CLAIM_POLICY:
        raise RuntimeError("Frozen L2 receipt claim policy mismatch")
    bindings = (
        (ROOT / receipt["identity_config"], receipt["identity_config_sha256"]),
        (
            ROOT / receipt["raw_snapshot"]["local_path"] / "manifest.json",
            receipt["raw_snapshot"]["manifest_sha256"],
        ),
        (
            ROOT / receipt["governed_snapshot"]["local_path"] / "manifest.json",
            receipt["governed_snapshot"]["manifest_sha256"],
        ),
        (
            ROOT / receipt["dataset"]["local_path"] / "dataset_manifest.json",
            receipt["dataset"]["manifest_sha256"],
        ),
        (
            ROOT / receipt["dataset"]["local_path"] / manifest["panel_file"],
            receipt["dataset"]["panel_sha256"],
        ),
        (
            ROOT / receipt["dataset"]["local_path"] / manifest["availability_file"],
            receipt["dataset"]["availability_sha256"],
        ),
        (ROOT / receipt["protocol"]["path"], receipt["protocol"]["sha256"]),
    )
    for path, expected_hash in bindings:
        if not path.exists() or sha256_file(path) != expected_hash:
            raise RuntimeError(f"Frozen L2 receipt binding mismatch: {path}")
    raw_manifest = json.loads(
        (ROOT / receipt["raw_snapshot"]["local_path"] / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    governed_manifest = json.loads(
        (
            ROOT / receipt["governed_snapshot"]["local_path"] / "manifest.json"
        ).read_text(encoding="utf-8")
    )
    if (ROOT / receipt["protocol"]["path"]).resolve() != protocol_path:
        raise RuntimeError("Frozen L2 receipt points to a different protocol path")
    if (ROOT / receipt["dataset"]["local_path"]).resolve() != dataset_dir:
        raise RuntimeError("Frozen L2 receipt points to a different dataset path")
    if receipt["protocol"]["protocol_id"] != protocol["protocol_id"]:
        raise RuntimeError("Frozen L2 receipt protocol ID mismatch")
    if receipt["dataset"]["dataset_id"] != manifest["dataset_id"]:
        raise RuntimeError("Frozen L2 receipt dataset ID mismatch")
    if receipt["dataset"]["panel_sha256"] != manifest["panel_sha256"]:
        raise RuntimeError("Frozen L2 receipt panel SHA mismatch")
    if receipt["dataset"]["availability_sha256"] != manifest["availability_sha256"]:
        raise RuntimeError("Frozen L2 receipt availability SHA mismatch")
    if receipt["raw_snapshot"]["snapshot_id"] != raw_manifest.get("snapshot_id"):
        raise RuntimeError("Frozen L2 receipt raw snapshot ID mismatch")
    if receipt["governed_snapshot"]["governed_id"] != governed_manifest.get(
        "governed_id"
    ):
        raise RuntimeError("Frozen L2 receipt governed snapshot ID mismatch")
    if governed_manifest.get("source_snapshot_id") != raw_manifest.get("snapshot_id"):
        raise RuntimeError("Frozen L2 governed-to-raw identity link mismatch")
    if manifest.get("source_snapshot_id") != raw_manifest.get("snapshot_id"):
        raise RuntimeError("Frozen L2 dataset-to-raw identity link mismatch")
    if governed_manifest.get("identity_config_sha256") != receipt.get(
        "identity_config_sha256"
    ):
        raise RuntimeError("Frozen L2 governed identity SHA mismatch")
    expected_window = {
        "data_start": manifest["date_start"],
        "first_rankable_signal_date": manifest["first_rankable_signal_date"],
        "first_execution_date": manifest["first_execution_date"],
        "end_date": manifest["date_end"],
        "per_asset_global_trading_date_warmup": 40,
    }
    if receipt.get("study_window") != expected_window:
        raise RuntimeError("Frozen L2 receipt study window mismatch")
    if receipt.get("code_fingerprint") != code_fingerprint():
        raise RuntimeError("Frozen L2 code fingerprint mismatch")
    return receipt


def reserve_one_shot_run(
    protocol: dict[str, Any],
    manifest: dict[str, Any],
    output_dir: Path,
    *,
    claim_root: Path = CLAIM_ROOT,
    receipt_sha256: str,
    protocol_sha256: str,
    dataset_manifest_sha256: str,
) -> Path:
    claim_root.mkdir(parents=True, exist_ok=True)
    claim_path = claim_root / f"{protocol['protocol_id']}.json"
    payload = {
        "schema_version": "sw2021-l2-dynamic-validation-one-shot-claim-v1",
        "claimed_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_id": protocol["protocol_id"],
        "dataset_id": manifest["dataset_id"],
        "panel_sha256": manifest["panel_sha256"],
        "availability_sha256": manifest["availability_sha256"],
        "code_fingerprint": code_fingerprint(),
        "receipt_sha256": receipt_sha256,
        "protocol_sha256": protocol_sha256,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "output_dir": str(output_dir),
        "status": "reserved_before_result_computation",
    }
    try:
        descriptor = os.open(claim_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(
            f"One-shot L2 protocol has already been claimed: {claim_path}"
        ) from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    descriptor = os.open(claim_root, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return claim_path


def main() -> None:
    args = parse_args()
    protocol_path = args.protocol.resolve()
    receipt_path = args.receipt.resolve()
    dataset_dir = args.dataset_dir.resolve()
    protocol = load_protocol(protocol_path)
    manifest = load_dataset_manifest(dataset_dir)
    validate_protocol_dataset(protocol, manifest)
    receipt = verify_receipt(
        receipt_path,
        protocol_path,
        dataset_dir,
        protocol,
        manifest,
    )
    receipt_sha256 = sha256_file(receipt_path)
    protocol_sha256 = sha256_file(protocol_path)
    dataset_manifest_sha256 = sha256_file(dataset_dir / "dataset_manifest.json")
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Validation output already exists: {output_dir}")
    claim_path = reserve_one_shot_run(
        protocol,
        manifest,
        output_dir,
        receipt_sha256=receipt_sha256,
        protocol_sha256=protocol_sha256,
        dataset_manifest_sha256=dataset_manifest_sha256,
    )

    panel = load_panel(dataset_dir)
    roc40, signal = build_absroc40_signal(panel)
    signal_eligible = panel.tradable_mask & np.isfinite(signal)
    signal_counts = signal_eligible.sum(axis=0)
    quote_counts = panel.tradable_mask.sum(axis=0)
    split = protocol["split"]
    trading = protocol["trading"]
    rankable = signal_counts >= int(trading["min_universe"])
    first_rankable = int(np.flatnonzero(rankable)[0])
    if panel.dates[first_rankable].date().isoformat() != split["prior_signal_date"]:
        raise RuntimeError("First rankable L2 signal date differs from protocol")
    if panel.dates[first_rankable + 1].date().isoformat() != split["validation_start"]:
        raise RuntimeError("First L2 execution date differs from protocol")
    if panel.dates[-1].date().isoformat() != split["validation_end"]:
        raise RuntimeError("L2 panel end differs from protocol")
    warmup_delays = []
    for asset in range(len(panel.symbols)):
        first_quote = int(np.flatnonzero(panel.tradable_mask[asset])[0])
        first_signal = int(np.flatnonzero(signal_eligible[asset])[0])
        warmup_delays.append(first_signal - first_quote)
    if set(warmup_delays) != {40}:
        raise RuntimeError(f"Per-asset ROC40 warmup differs: {sorted(set(warmup_delays))}")

    config = ValidationConfig(
        validation_start=split["validation_start"],
        validation_end=split["validation_end"],
        initial_cash=float(trading["initial_cash"]),
        min_universe=int(trading["min_universe"]),
        slots=int(trading["slots"]),
        buy_rank=int(trading["buy_rank"]),
        hold_rank=int(trading["hold_rank"]),
        robust_z_threshold=float(trading["robust_z_threshold"]),
        stop_loss=float(trading["stop_loss"]),
        metrics_scope="validation",
    )
    daily, raw_trades, base_summary = run_formula_validation(
        formula_id=FORMULA_ID,
        signal=signal,
        open_prices=panel.absolute("open"),
        close_prices=panel.absolute("close"),
        tradable_mask=panel.tradable_mask,
        dates=panel.dates,
        symbols=panel.symbols,
        config=config,
    )
    daily_date_indices = {
        date.date().isoformat(): index for index, date in enumerate(panel.dates)
    }
    daily_indices = np.asarray([daily_date_indices[str(date)] for date in daily["date"]])
    daily["quote_universe_count"] = quote_counts[daily_indices]
    daily["signal_universe_count"] = signal_counts[daily_indices]
    trades, accounting = enrich_and_reconcile_trades(
        raw_trades,
        roc40=roc40,
        dates=panel.dates,
        symbols=panel.symbols,
        names=panel.names,
        final_close=panel.absolute("close")[:, -1],
        final_cash=float(daily.iloc[-1]["cash"]),
        final_equity=float(daily.iloc[-1]["equity"]),
    )
    strategy = {
        **path_metrics(daily, "equity"),
        "calendar_year_returns": annual_returns(daily, "equity"),
        "average_cash_weight": float(daily["cash_weight"].mean()),
        "average_position_count": float(daily["position_count"].mean()),
        **trade_metrics(trades),
    }
    benchmark = {
        **path_metrics(daily, "benchmark_equity"),
        "calendar_year_returns": annual_returns(daily, "benchmark_equity"),
    }
    buy_hold = path_metrics(daily, "buy_hold_equity")
    annual = pd.DataFrame(
        {
            "year": list(strategy["calendar_year_returns"]),
            "strategy_return": list(strategy["calendar_year_returns"].values()),
            "benchmark_return": list(benchmark["calendar_year_returns"].values()),
        }
    )
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "interpretation": "sequential_exploratory_industry_mechanism_test_not_tradable_backtest",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256,
        "receipt": project_path(receipt_path),
        "receipt_sha256": receipt_sha256,
        "dataset_id": manifest["dataset_id"],
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "panel_sha256": manifest["panel_sha256"],
        "availability_sha256": manifest["availability_sha256"],
        "code_commit": git_commit(),
        "code_fingerprint": code_fingerprint(),
        "one_shot_claim": str(claim_path),
        "one_shot_claim_sha256": sha256_file(claim_path),
        "formula": protocol["formula"],
        "date_start": str(daily.iloc[0]["date"]),
        "date_end": str(daily.iloc[-1]["date"]),
        "prior_signal_date": split["prior_signal_date"],
        "trading": trading,
        "universe": {
            "first_quote_count": int(quote_counts[0]),
            "first_signal_count": int(signal_counts[first_rankable]),
            "final_quote_count": int(quote_counts[-1]),
            "final_signal_count": int(signal_counts[-1]),
            "minimum_signal_count_during_validation": int(
                daily["signal_universe_count"].min()
            ),
            "maximum_signal_count_during_validation": int(
                daily["signal_universe_count"].max()
            ),
            "per_asset_first_signal_global_date_delay": 40,
        },
        "strategy": strategy,
        "benchmark": benchmark,
        "buy_hold": buy_hold,
        "accounting": accounting,
        "base_summary": base_summary,
        "parameter_tuning_performed": False,
        "result_dependent_protocol_change_performed": False,
        "run_sequence": 1,
        "receipt_research_status": receipt["research_status"],
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        result_path = temporary / "result.json"
        daily_path = temporary / "daily.csv"
        trades_path = temporary / "trades.csv"
        annual_path = temporary / "annual_summary.csv"
        write_text_synced(
            result_path,
            json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        )
        daily.to_csv(daily_path, index=False)
        trades.to_csv(trades_path, index=False)
        annual.to_csv(annual_path, index=False)
        for path in (daily_path, trades_path, annual_path):
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
        outputs = (result_path, daily_path, trades_path, annual_path)
        write_text_synced(
            temporary / "SHA256SUMS",
            "\n".join(f"{sha256_file(path)}  {path.name}" for path in outputs) + "\n",
        )
        descriptor = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, output_dir)
        descriptor = os.open(output_dir.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    print(f"result: {output_dir}")


if __name__ == "__main__":
    main()