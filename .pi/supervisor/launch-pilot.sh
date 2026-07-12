#!/usr/bin/env bash
set -euo pipefail
umask 077

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
source "$ROOT/.pi/supervisor/ssh.sh"
RUNTIME="$ROOT/.pi/supervisor/runtime.json"
RUN_SPEC="$ROOT/.pi/supervisor/run-spec.json"
REMOTE_RUNTIME=$(jq -er '.remote_runtime' "$RUNTIME")
REMOTE_PROJECT=$(jq -er '.remote_project' "$RUNTIME")
BUNDLE=$(jq -er '.local_bundle' "$RUNTIME")
INPUTS=$(jq -er '.local_input_archive' "$RUNTIME")
RUN_ID=$(jq -er '.run_id' "$RUNTIME")

[[ $(sha256sum "$BUNDLE" | awk '{print $1}') == "$(jq -er '.expected.bundle_sha256' "$RUNTIME")" ]]
[[ $(sha256sum "$INPUTS" | awk '{print $1}') == "$(jq -er '.expected.input_archive_sha256' "$RUNTIME")" ]]
RUN_SPEC_SHA=$(sha256sum "$RUN_SPEC" | awk '{print $1}')
RUNNER_SHA=$(sha256sum "$ROOT/.pi/supervisor/remote/runner.sh" | awk '{print $1}')

remote_exec mkdir -p "$REMOTE_RUNTIME"
for file in deploy.sh probe.py runner.sh start.sh stop.sh write_state.py; do
  scp_to_remote "$ROOT/.pi/supervisor/remote/$file" "$REMOTE_RUNTIME/$file"
done
scp_to_remote "$BUNDLE" "$REMOTE_RUNTIME/v3a.bundle"
scp_to_remote "$INPUTS" "$REMOTE_RUNTIME/v3a-stage-d-inputs.tar.gz"
scp_to_remote "$RUN_SPEC" "$REMOTE_RUNTIME/run-spec.json"

temporary=$(mktemp)
trap 'rm -f "$temporary"' EXIT
jq -n \
  --arg run_id "$RUN_ID" \
  --arg repo "$REMOTE_PROJECT" \
  --arg runtime_dir "$REMOTE_RUNTIME" \
  --arg screen_name "$(jq -er '.remote_screen' "$RUNTIME")" \
  --arg state_file "$REMOTE_RUNTIME/runner-state.json" \
  --arg bundle_path "$REMOTE_RUNTIME/v3a.bundle" \
  --arg input_archive_path "$REMOTE_RUNTIME/v3a-stage-d-inputs.tar.gz" \
  --arg run_spec_path "$REMOTE_RUNTIME/run-spec.json" \
  --arg run_spec_sha256 "$RUN_SPEC_SHA" \
  --arg runner_sha256 "$RUNNER_SHA" \
  --argjson expected "$(jq -ec '.expected' "$RUNTIME")" \
  '{run_id:$run_id,repo:$repo,runtime_dir:$runtime_dir,screen_name:$screen_name,state_file:$state_file,bundle_path:$bundle_path,input_archive_path:$input_archive_path,run_spec_path:$run_spec_path,run_spec_sha256:$run_spec_sha256,runner_sha256:$runner_sha256,expected:$expected}' \
  >"$temporary"
scp_to_remote "$temporary" "$REMOTE_RUNTIME/runtime.json"
remote_exec chmod 700 \
  "$REMOTE_RUNTIME/deploy.sh" "$REMOTE_RUNTIME/probe.py" \
  "$REMOTE_RUNTIME/runner.sh" "$REMOTE_RUNTIME/start.sh" \
  "$REMOTE_RUNTIME/stop.sh" "$REMOTE_RUNTIME/write_state.py"
remote_exec "$REMOTE_RUNTIME/deploy.sh" "$REMOTE_RUNTIME/runtime.json"
remote_exec "$REMOTE_RUNTIME/start.sh" "$REMOTE_RUNTIME/runtime.json"
printf '%s\n' "pilot launched: $RUN_ID"