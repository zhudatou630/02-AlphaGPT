#!/usr/bin/env python3
"""Run the shared Phase3c trainer under the clean V2 research contract."""

from __future__ import annotations

from functools import partial
import hashlib
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from alpha_etf.research_v2.spec import build_research_spec, load_dataset_manifest, load_price_panel
from alpha_etf.research_v2.vocab import FORMULA_VOCAB_V2
from scripts.phase3c_train_gpu import TrainingRuntime, main


DATASET_DIR = ROOT / "data" / "processed" / "v2" / "dataset"
OUT_ROOT = ROOT / "data" / "processed" / "v2" / "training"


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _code_fingerprint() -> str:
    digest = hashlib.sha256()
    paths = sorted((ROOT / "src" / "alpha_etf").rglob("*.py"))
    paths.extend([ROOT / "scripts" / "phase3c_train_gpu.py", Path(__file__)])
    for path in sorted(set(paths)):
        digest.update(str(path.relative_to(ROOT)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _runtime() -> TrainingRuntime:
    dataset_manifest = load_dataset_manifest(DATASET_DIR)
    code_commit = _git_commit()
    code_fingerprint = _code_fingerprint()

    def research_spec_factory(scorer_config: dict[str, object]) -> dict[str, object]:
        return build_research_spec(
            dataset_manifest=dataset_manifest,
            vocab=FORMULA_VOCAB_V2,
            scorer_config=scorer_config,
            code_commit=code_commit,
            code_fingerprint=code_fingerprint,
        )

    return TrainingRuntime(
        vocab=FORMULA_VOCAB_V2,
        panel_loader=partial(load_price_panel, DATASET_DIR),
        default_out_root=OUT_ROOT,
        research_spec_factory=research_spec_factory,
        formula_prefix="v2_gpu",
        train_source="v2_gpu_train",
        audit_source="v2_cpu_audit",
        validator_filename="v2_validator_summary.csv",
    )


if __name__ == "__main__":
    main(_runtime())