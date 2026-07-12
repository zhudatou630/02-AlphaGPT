#!/usr/bin/env python3
"""Build the immutable post-commit runtime identity for a Stage D GPU run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from alpha_etf.research_v3a.candidates import CandidateConfig
from alpha_etf.research_v3a.scoring import ScorerConfig
from alpha_etf.research_v3a.spec import sha256_file
from alpha_etf.research_v3a.stage_d import (
    build_stage_d_binding,
    load_stage_d_protocol,
    load_stage_d_train_view,
    load_train_view_source_manifest,
    verify_stage_c_prerequisite,
)
from scripts.v3a.random_baseline import _write_json_atomic
from scripts.v3a.runtime import build_runtime_research_spec, require_clean_v3a_code


DEFAULT_PROTOCOL = ROOT / "configs/v3a_stage_d_gpu_pilot.json"
DEFAULT_TRAIN_VIEW = ROOT / "data/processed/v3a/stage_d/train_view"
DEFAULT_STAGE_C_REPORT = (
    ROOT / "data/processed/v3a/stage_c_reports/v3a-stage-c-20260711-01.json"
)
DEFAULT_OUTPUT = ROOT / "data/processed/v3a/stage_d/pilot_binding.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-file", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--train-view-dir", type=Path, default=DEFAULT_TRAIN_VIEW)
    parser.add_argument("--stage-c-report", type=Path, default=DEFAULT_STAGE_C_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = load_stage_d_protocol(args.protocol_file)
    verify_stage_c_prerequisite(protocol, args.stage_c_report)
    require_clean_v3a_code(extra_paths=(args.protocol_file,))
    source_manifest = load_train_view_source_manifest(args.train_view_dir)
    research_spec = build_runtime_research_spec(
        source_manifest,
        scorer_config=ScorerConfig(),
        candidate_config=CandidateConfig(),
    )
    train_view = load_stage_d_train_view(
        args.train_view_dir, research_spec=research_spec
    )
    binding = build_stage_d_binding(
        protocol=protocol,
        research_spec=research_spec,
        train_view_manifest=train_view.manifest,
        stage_c_report_sha256=sha256_file(args.stage_c_report),
    )
    _write_json_atomic(args.output, binding)
    print(json.dumps(binding, ensure_ascii=False, indent=2))
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()