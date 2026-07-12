#!/usr/bin/env bash
set -euo pipefail
umask 077

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
source "$ROOT/.pi/supervisor/ssh.sh"
RUNTIME="$ROOT/.pi/supervisor/runtime.json"
RUN_ID=$(jq -er '.run_id' "$RUNTIME")
REMOTE_PROJECT=$(jq -er '.remote_project' "$RUNTIME")
OUTPUT=${1:?local output directory is required}
SMOKE_ID=$(jq -er '.smoke.run_id' "$ROOT/.pi/supervisor/run-spec.json")

rsync_from_remote "$REMOTE_PROJECT/data/processed/v3a/training/runs/$RUN_ID" "$OUTPUT/training/runs/"
rsync_from_remote "$REMOTE_PROJECT/data/processed/v3a/smoke/$SMOKE_ID" "$OUTPUT/smoke/"
rsync_from_remote "$REMOTE_PROJECT/data/processed/v3a/stage_d/" "$OUTPUT/stage_d/"
rsync_from_remote "$(jq -er '.remote_runtime' "$RUNTIME")/" "$OUTPUT/remote-runtime/"
printf '%s\n' "pilot artifacts synced to $OUTPUT"