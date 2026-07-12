#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
source "$ROOT/.pi/supervisor/ssh.sh"
REMOTE_RUNTIME=$(jq -er '.remote_runtime' "$ROOT/.pi/supervisor/runtime.json")
remote_exec "$REMOTE_RUNTIME/stop.sh" "$REMOTE_RUNTIME/runtime.json"