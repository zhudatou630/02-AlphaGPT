"""Runtime identity helpers shared by V3A command-line entry points."""

from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
from typing import Any

from alpha_etf.research_v3a.candidates import (
    CANONICALIZER_VERSION,
    CandidateConfig,
    canonicalizer_config,
)
from alpha_etf.research_v3a.factors import factor_config
from alpha_etf.research_v3a.language import FORMULA_VOCAB
from alpha_etf.research_v3a.scoring import ScorerConfig
from alpha_etf.research_v3a.sampling import SamplingConfig
from alpha_etf.research_v3a.spec import build_research_spec


ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "data/processed/v3a/dataset"


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def code_fingerprint() -> str:
    paths = sorted((ROOT / "src/alpha_etf/research_v3a").glob("*.py"))
    paths.append(ROOT / "src/alpha_etf/gpt/policy.py")
    paths.extend(sorted((ROOT / "scripts/v3a").glob("*.py")))
    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        digest.update(str(path.relative_to(ROOT)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def build_runtime_research_spec(
    dataset_manifest: dict[str, Any],
    *,
    scorer_config: ScorerConfig,
    candidate_config: CandidateConfig,
) -> dict[str, Any]:
    return build_research_spec(
        dataset_manifest=dataset_manifest,
        factor_config=factor_config(),
        vocab_config={
            **FORMULA_VOCAB.to_config(),
            "sampling": SamplingConfig().to_dict(),
        },
        scorer_config=scorer_config.to_dict(),
        canonicalizer_config=canonicalizer_config(),
        candidate_config=candidate_config.to_dict(),
        code_commit=git_commit(),
        code_fingerprint=code_fingerprint(),
    )