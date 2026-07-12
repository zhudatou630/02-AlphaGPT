#!/usr/bin/env bash
set -euo pipefail

# Historical runner: this entry point is bound to the legacy full-funnel pilot
# protocol and must not be reused for new Stage D or formal runs. New runs need
# a separately approved protocol/binding; export_top_formulas.py now emits both
# raw and curated top-N artifacts without requiring the funnel.

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"

PYTHON_BIN=${V3A_PYTHON_BIN:-$ROOT/.venv/bin/python}
PROTOCOL=${V3A_STAGE_D_PROTOCOL:-$ROOT/configs/v3a_stage_d_gpu_pilot.json}
DATASET_DIR=${V3A_DATASET_DIR:-$ROOT/data/processed/v3a/dataset}
TRAIN_VIEW_DIR=${V3A_TRAIN_VIEW_DIR:-$ROOT/data/processed/v3a/stage_d/train_view}
BINDING=${V3A_STAGE_D_BINDING:-$ROOT/data/processed/v3a/stage_d/pilot_binding.json}
RUN_ROOT=${V3A_STAGE_D_RUN_ROOT:-$ROOT/data/processed/v3a/training/runs}
SEED=314159
STOP_AFTER=10240
PROTOCOL_PREFIX=078af5f6466b
RUN_ID="v3a-stage-d-pilot-transformer-s${SEED}-${PROTOCOL_PREFIX}"
RUN_DIR="$RUN_ROOT/$RUN_ID"
SMOKE_ID="v3a-stage-d-pilot-smoke-${PROTOCOL_PREFIX}"
SMOKE_DIR="$ROOT/data/processed/v3a/smoke/$SMOKE_ID"

: "${V3A_EXPECTED_BINDING_ID:?missing V3A_EXPECTED_BINDING_ID}"

mapfile -t BINDING_VALUES < <(
  PYTHONPATH=src "$PYTHON_BIN" - "$BINDING" "$V3A_EXPECTED_BINDING_ID" <<'PY'
import json, sys
from alpha_etf.research_v3a.stage_d import stage_d_binding_id
binding=json.load(open(sys.argv[1], encoding="utf-8"))
def require(condition, message):
    if not condition:
        raise RuntimeError(message)
require(binding["binding_id"] == sys.argv[2], "unapproved Stage D binding")
require(binding["binding_id"] == stage_d_binding_id(binding), "invalid Stage D binding id")
require(
    binding["protocol_id"] == "078af5f6466bd9661443d26f5722a107f67e69298bb0e593c730e7f6f042cc8b",
    "wrong Stage D pilot protocol",
)
for key in (
    "dataset_id", "panel_sha256", "research_spec_id", "code_fingerprint",
    "train_view_id", "code_commit", "binding_id",
):
    print(binding[key])
PY
)
if [[ ${#BINDING_VALUES[@]} -ne 7 ]]; then
  printf 'invalid Stage D binding output\n' >&2
  exit 1
fi
V3A_EXPECTED_DATASET_ID=${BINDING_VALUES[0]}
V3A_EXPECTED_PANEL_SHA256=${BINDING_VALUES[1]}
V3A_EXPECTED_RESEARCH_SPEC_ID=${BINDING_VALUES[2]}
V3A_EXPECTED_CODE_FINGERPRINT=${BINDING_VALUES[3]}
V3A_EXPECTED_TRAIN_VIEW_ID=${BINDING_VALUES[4]}
V3A_EXPECTED_CODE_COMMIT=${BINDING_VALUES[5]}
if [[ $(git rev-parse HEAD) != "$V3A_EXPECTED_CODE_COMMIT" ]]; then
  printf 'Stage D binding commit does not match HEAD\n' >&2
  exit 1
fi

PIN_ARGS=(
  --expected-dataset-id "$V3A_EXPECTED_DATASET_ID"
  --expected-panel-sha256 "$V3A_EXPECTED_PANEL_SHA256"
  --expected-research-spec-id "$V3A_EXPECTED_RESEARCH_SPEC_ID"
  --expected-code-fingerprint "$V3A_EXPECTED_CODE_FINGERPRINT"
)

PYTHONPATH=src "$PYTHON_BIN" scripts/v3a/gpu_resume_probe.py \
  --protocol-file "$PROTOCOL" \
  --train-view-dir "$TRAIN_VIEW_DIR" \
  --binding-file "$BINDING" \
  --output "$ROOT/data/processed/v3a/stage_d/gpu_resume_probe-${PROTOCOL_PREFIX}.json"

if [[ -f "$SMOKE_DIR/summary.json" ]]; then
  "$PYTHON_BIN" - \
    "$SMOKE_DIR/summary.json" \
    "$SMOKE_DIR/checkpoint.pt" \
    "$SMOKE_ID" \
    "$V3A_EXPECTED_DATASET_ID" \
    "$V3A_EXPECTED_PANEL_SHA256" \
    "$V3A_EXPECTED_RESEARCH_SPEC_ID" \
    "$V3A_EXPECTED_CODE_FINGERPRINT" <<'PY'
import hashlib, json, sys
summary=json.load(open(sys.argv[1], encoding="utf-8"))
checkpoint=__import__("pathlib").Path(sys.argv[2])
def require(condition, message):
    if not condition:
        raise RuntimeError(message)
require(checkpoint.is_file(), "missing smoke checkpoint")
require(summary["status"] == "passed", "smoke did not pass")
require(summary["run_id"] == sys.argv[3], "smoke run id mismatch")
require(summary["device"] == "cuda", "smoke did not use CUDA")
require(summary["finite_gradients"] is True, "smoke gradients are not finite")
require(summary["resume_next_batch_equal"] is True, "smoke resume mismatch")
require(summary["validation_or_final_metrics_read"] is False, "smoke read sealed data")
require(summary["dataset_id"] == sys.argv[4], "smoke dataset mismatch")
require(summary["panel_sha256"] == sys.argv[5], "smoke panel mismatch")
require(summary["research_spec_id"] == sys.argv[6], "smoke ResearchSpec mismatch")
require(summary["code_fingerprint"] == sys.argv[7], "smoke code mismatch")
require(
    summary["checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    "smoke checkpoint mismatch",
)
PY
else
  PYTHONPATH=src "$PYTHON_BIN" scripts/v3a/gpu_smoke.py \
    --device cuda \
    --run-id "$SMOKE_ID" \
    --dataset-dir "$DATASET_DIR" \
    "${PIN_ARGS[@]}"
fi

checkpoint_attempts() {
  "$PYTHON_BIN" - "$RUN_DIR/checkpoint_latest.pt" <<'PY'
import sys, torch
checkpoint=torch.load(sys.argv[1], map_location="cpu", weights_only=False)
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
    printf 'pilot resume checkpoint did not reach %s: %s\n' "$STOP_AFTER" "$completed" >&2
    exit 1
  fi
  PYTHONPATH=src "$PYTHON_BIN" scripts/v3a/train_gpu.py \
    "${TRAIN_ARGS[@]}" --resume
fi

PYTHONPATH=src "$PYTHON_BIN" scripts/v3a/select_candidates.py \
  --protocol-file "$PROTOCOL" \
  --method transformer \
  --seed "$SEED" \
  --run-dir "$RUN_DIR" \
  --train-view-dir "$TRAIN_VIEW_DIR" \
  --binding-file "$BINDING"

"$PYTHON_BIN" - "$RUN_DIR/training_summary.json" "$RUN_DIR/funnel/summary.json" "$V3A_EXPECTED_BINDING_ID" <<'PY'
import json, sys
training=json.load(open(sys.argv[1], encoding="utf-8"))
funnel=json.load(open(sys.argv[2], encoding="utf-8"))
def require(condition, message):
    if not condition:
        raise RuntimeError(message)
require(training["status"] == "pilot_trained_awaiting_funnel", "training status mismatch")
require(training["binding_id"] == sys.argv[3], "training binding mismatch")
require(training["attempt_count"] == 50_000, "training attempts mismatch")
require(training["resume_count"] >= 1, "forced resume did not occur")
require(training["grammar_invalid_rate"] == 0.0, "grammar invalid rate is nonzero")
require(training["validation_or_final_metrics_read"] is False, "training read sealed data")
require(funnel["status"] == "pilot_diagnostic_passed", "funnel status mismatch")
require(funnel["binding_id"] == sys.argv[3], "funnel binding mismatch")
require(funnel["selected_count"] == 50, "funnel selected count mismatch")
require(funnel["research_conclusion_allowed"] is False, "pilot allowed a research conclusion")
require(funnel["validation_or_final_metrics_read"] is False, "funnel read sealed data")
print(json.dumps({
    "status": "pilot_passed",
    "run_id": training["run_id"],
    "attempts": training["attempt_count"],
    "attempts_per_second": training["attempts_per_second"],
    "selected_count": funnel["selected_count"],
}, ensure_ascii=False))
PY