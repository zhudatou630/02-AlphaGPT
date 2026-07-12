#!/usr/bin/env bash
set -euo pipefail
umask 077

CONFIG=${1:?runtime config is required}
SYSTEM_PYTHON=/root/miniconda3/bin/python

config_get() {
  "$SYSTEM_PYTHON" - "$CONFIG" "$1" <<'PY'
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
value = data
for key in sys.argv[2].split("."):
    value = value[key]
print(value)
PY
}

RUN_ID=$(config_get run_id)
REPO=$(config_get repo)
RUNTIME_DIR=$(config_get runtime_dir)
COMMIT=$(config_get expected.commit)
BINDING_ID=$(config_get expected.binding_id)
BINDING_SHA=$(config_get expected.binding_sha256)
RUN_SPEC=$(config_get run_spec_path)
RUN_SPEC_SHA=$(config_get run_spec_sha256)
RUNNER_SHA=$(config_get runner_sha256)
STATE_WRITER="$RUNTIME_DIR/write_state.py"
PYTHON="$REPO/.venv/bin/python"
LOCK="$RUNTIME_DIR/runner.lock"
CURRENT_PHASE=preflight
FINISHED=0

write_state() {
  "$SYSTEM_PYTHON" "$STATE_WRITER" --config "$CONFIG" "$@" >/dev/null
}

on_exit() {
  rc=$?
  if [[ $FINISHED -eq 0 ]]; then
    write_state --status failed --phase "$CURRENT_PHASE" \
      --detail "Runner exited before Stage D pilot completion" --exit-code "$rc" \
      --runner-pid "$$"
  fi
  exit "$rc"
}
trap on_exit EXIT

exec 9>"$LOCK"
if ! flock -n 9; then
  exit 75
fi

mkdir -p "$RUNTIME_DIR/logs"
exec >>"$RUNTIME_DIR/runner.log" 2>&1
printf 'runner_start=%s pid=%s\n' "$(date -u +%FT%TZ)" "$$"
write_state --status running --phase preflight \
  --detail "Checking Stage D frozen identities" --runner-pid "$$"

[[ $(git -C "$REPO" rev-parse HEAD) == "$COMMIT" ]]
git -C "$REPO" diff --quiet
git -C "$REPO" diff --cached --quiet
[[ $(sha256sum "$RUN_SPEC" | awk '{print $1}') == "$RUN_SPEC_SHA" ]]
[[ $(sha256sum "$RUNTIME_DIR/runner.sh" | awk '{print $1}') == "$RUNNER_SHA" ]]
[[ $(sha256sum "$REPO/data/processed/v3a/stage_d/pilot_binding.json" | awk '{print $1}') == "$BINDING_SHA" ]]

CURRENT_PHASE=pilot
write_state --status running --phase pilot \
  --detail "Running the frozen 50,000-attempt Stage D CUDA pilot" --runner-pid "$$"
V3A_EXPECTED_BINDING_ID="$BINDING_ID" \
V3A_PYTHON_BIN="$PYTHON" \
V3A_STAGE_D_PROTOCOL="$REPO/configs/v3a_stage_d_gpu_pilot.json" \
V3A_DATASET_DIR="$REPO/data/processed/v3a/dataset" \
V3A_TRAIN_VIEW_DIR="$REPO/data/processed/v3a/stage_d/train_view" \
V3A_STAGE_D_BINDING="$REPO/data/processed/v3a/stage_d/pilot_binding.json" \
V3A_STAGE_D_RUN_ROOT="$REPO/data/processed/v3a/training/runs" \
PYTHONUNBUFFERED=1 PYTHONPATH="$REPO/src" \
  "$REPO/scripts/v3a/stage_d_gpu_pilot.sh" \
  >>"$RUNTIME_DIR/logs/stage-d-pilot.log" 2>&1

"$SYSTEM_PYTHON" - \
  "$REPO/data/processed/v3a/training/runs/$RUN_ID/training_summary.json" \
  "$REPO/data/processed/v3a/training/runs/$RUN_ID/funnel/summary.json" \
  "$RUN_ID" "$BINDING_ID" <<'PY'
import json, sys
training = json.load(open(sys.argv[1], encoding="utf-8"))
funnel = json.load(open(sys.argv[2], encoding="utf-8"))
if training.get("run_id") != sys.argv[3] or funnel.get("run_id") != sys.argv[3]:
    raise RuntimeError("Stage D runner output run id mismatch")
if training.get("binding_id") != sys.argv[4] or funnel.get("binding_id") != sys.argv[4]:
    raise RuntimeError("Stage D runner output binding mismatch")
if training.get("attempt_count") != 50_000 or funnel.get("selected_count") != 50:
    raise RuntimeError("Stage D runner output budget mismatch")
PY

CURRENT_PHASE=complete
write_state --status awaiting_validation --phase complete \
  --detail "Stage D pilot finished; local artifact validation is pending" \
  --exit-code 0 --runner-pid "$$"
FINISHED=1
printf 'runner_complete=%s\n' "$(date -u +%FT%TZ)"