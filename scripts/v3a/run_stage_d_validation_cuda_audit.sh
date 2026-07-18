#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON=${V3A_PYTHON:-python}
TRAIN_VIEW_DIR=${V3A_VALIDATION_TRAIN_VIEW_DIR:-$ROOT/.pi/validation-cuda-input/train_view}
REGISTRY=${V3A_VALIDATION_REGISTRY:-$ROOT/.pi/validation-cuda-input/cuda_registry.jsonl}
OUTPUT_DIR=${V3A_VALIDATION_CUDA_OUTPUT:-$ROOT/.pi/validation-cuda-output}

cd "$ROOT"
PYTHONPATH=src "$PYTHON" scripts/v3a/prepare_stage_d_validation.py \
  --mode audit-cuda \
  --protocol configs/v3a_stage_d_validation.json \
  --train-view-dir "$TRAIN_VIEW_DIR" \
  --registry "$REGISTRY" \
  --output-dir "$OUTPUT_DIR"

cd "$OUTPUT_DIR"
sha256sum -c SHA256SUMS