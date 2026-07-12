#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"

PYTHON_BIN=${V3A_PYTHON_BIN:-$ROOT/.venv/bin/python}
PROTOCOL=${V3A_STAGE_D_PROTOCOL:-$ROOT/configs/v3a_stage_d_reward_calibration.json}
DATASET_DIR=${V3A_DATASET_DIR:-$ROOT/data/processed/v3a/dataset}
TRAIN_VIEW_DIR=${V3A_TRAIN_VIEW_DIR:-$ROOT/data/processed/v3a/stage_d/reward_calibration_train_view}
BINDING=${V3A_STAGE_D_BINDING:-$ROOT/data/processed/v3a/stage_d/reward_calibration_binding.json}
RUN_ROOT=${V3A_STAGE_D_RUN_ROOT:-$ROOT/data/processed/v3a/training/runs}
SEED=314159
STOP_AFTER=10240
PROTOCOL_PREFIX=b6f34bf5e0f7
RUN_ID="v3a-stage-d-reward-calibration-transformer-s${SEED}-${PROTOCOL_PREFIX}"
RUN_DIR="$RUN_ROOT/$RUN_ID"

: "${V3A_EXPECTED_BINDING_ID:?missing V3A_EXPECTED_BINDING_ID}"

mapfile -t BINDING_VALUES < <(
  PYTHONPATH=src "$PYTHON_BIN" - "$PROTOCOL" "$BINDING" "$V3A_EXPECTED_BINDING_ID" <<'PY'
import json
import sys
from alpha_etf.research_v3a.stage_d import stage_d_binding_id, stage_d_protocol_id

protocol = json.load(open(sys.argv[1], encoding="utf-8"))
binding = json.load(open(sys.argv[2], encoding="utf-8"))

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

require(protocol["protocol_id"] == stage_d_protocol_id(protocol), "invalid calibration protocol id")
require(binding["binding_id"] == sys.argv[3], "unapproved calibration binding")
require(binding["binding_id"] == stage_d_binding_id(binding), "invalid calibration binding id")
require(binding["protocol_id"] == protocol["protocol_id"], "calibration protocol mismatch")
for key in (
    "dataset_id", "panel_sha256", "research_spec_id", "code_fingerprint",
    "train_view_id", "code_commit", "binding_id",
):
    print(binding[key])
PY
)
if [[ ${#BINDING_VALUES[@]} -ne 7 ]]; then
  printf 'invalid reward-calibration binding output\n' >&2
  exit 1
fi
EXPECTED_DATASET_ID=${BINDING_VALUES[0]}
EXPECTED_PANEL_SHA256=${BINDING_VALUES[1]}
EXPECTED_RESEARCH_SPEC_ID=${BINDING_VALUES[2]}
EXPECTED_CODE_FINGERPRINT=${BINDING_VALUES[3]}
EXPECTED_COMMIT=${BINDING_VALUES[5]}
if [[ $(git rev-parse HEAD) != "$EXPECTED_COMMIT" ]]; then
  printf 'reward-calibration binding commit does not match HEAD\n' >&2
  exit 1
fi

PIN_ARGS=(
  --expected-dataset-id "$EXPECTED_DATASET_ID"
  --expected-panel-sha256 "$EXPECTED_PANEL_SHA256"
  --expected-research-spec-id "$EXPECTED_RESEARCH_SPEC_ID"
  --expected-code-fingerprint "$EXPECTED_CODE_FINGERPRINT"
)

SMOKE_ID="v3a-stage-d-reward-calibration-smoke-${PROTOCOL_PREFIX}"
PYTHONPATH=src "$PYTHON_BIN" scripts/v3a/gpu_smoke.py \
  --device cuda \
  --run-id "$SMOKE_ID" \
  --dataset-dir "$DATASET_DIR" \
  "${PIN_ARGS[@]}"

PYTHONPATH=src "$PYTHON_BIN" scripts/v3a/gpu_resume_probe.py \
  --protocol-file "$PROTOCOL" \
  --train-view-dir "$TRAIN_VIEW_DIR" \
  --binding-file "$BINDING" \
  --output "$ROOT/data/processed/v3a/stage_d/gpu-resume-probe-${PROTOCOL_PREFIX}.json"

checkpoint_attempts() {
  "$PYTHON_BIN" - "$RUN_DIR/checkpoint_latest.pt" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(int(checkpoint["attempt_count"]))
PY
}

TRAIN_ARGS=(
  --protocol-file "$PROTOCOL"
  --seed "$SEED"
  --run-id "$RUN_ID"
  --train-view-dir "$TRAIN_VIEW_DIR"
  --binding-file "$BINDING"
  --out-dir "$RUN_ROOT"
)

if [[ ! -f "$RUN_DIR/training_complete.json" ]]; then
  if [[ -f "$RUN_DIR/checkpoint_latest.pt" ]]; then
    completed=$(checkpoint_attempts)
    if (( completed < STOP_AFTER )); then
      PYTHONPATH=src "$PYTHON_BIN" scripts/v3a/train_gpu.py \
        "${TRAIN_ARGS[@]}" --resume --stop-after "$STOP_AFTER"
    fi
  else
    PYTHONPATH=src "$PYTHON_BIN" scripts/v3a/train_gpu.py \
      "${TRAIN_ARGS[@]}" --stop-after "$STOP_AFTER"
  fi

  completed=$(checkpoint_attempts)
  if (( completed < STOP_AFTER )); then
    printf 'calibration resume checkpoint did not reach %s: %s\n' "$STOP_AFTER" "$completed" >&2
    exit 1
  fi
  PYTHONPATH=src "$PYTHON_BIN" scripts/v3a/train_gpu.py \
    "${TRAIN_ARGS[@]}" --resume
fi

PYTHONPATH=src "$PYTHON_BIN" scripts/v3a/export_top_formulas.py \
  --run-dir "$RUN_DIR"

"$PYTHON_BIN" - "$RUN_DIR/training_summary.json" "$RUN_ID" "$V3A_EXPECTED_BINDING_ID" <<'PY'
import json
import sys

training = json.load(open(sys.argv[1], encoding="utf-8"))

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

require(training["status"] == "pilot_calibration_trained", "calibration training status mismatch")
require(training["run_id"] == sys.argv[2], "calibration run id mismatch")
require(training["binding_id"] == sys.argv[3], "calibration binding mismatch")
require(training["attempt_count"] == 50_000, "calibration attempts mismatch")
require(training["resume_count"] >= 1, "calibration forced resume did not occur")
require(training["grammar_invalid_rate"] == 0.0, "calibration grammar invalid rate is nonzero")
require(training["training_invalid_reward"] == -0.01, "calibration invalid reward mismatch")
require(training["full_candidate_funnel_applied"] is False, "calibration unexpectedly ran funnel")
require(training["candidate_conclusion_allowed"] is False, "calibration allowed a research conclusion")
require(training["validation_or_final_metrics_read"] is False, "calibration read sealed data")
require("semantic_reward_mean" in training["last_training_log"], "semantic reward log is missing")
print(json.dumps({
    "status": "reward_calibration_pilot_passed",
    "run_id": training["run_id"],
    "attempts": training["attempt_count"],
    "training_invalid_reward": training["training_invalid_reward"],
    "last_semantic_reward_mean": training["last_training_log"]["semantic_reward_mean"],
}, ensure_ascii=False))
PY