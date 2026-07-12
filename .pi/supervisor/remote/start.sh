#!/usr/bin/env bash
set -euo pipefail
umask 077

CONFIG=${1:?runtime config is required}
PYTHON=/root/miniconda3/bin/python
get() {
  "$PYTHON" - "$CONFIG" "$1" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
for key in sys.argv[2].split("."):
    value = value[key]
print(value)
PY
}

RUNTIME=$(get runtime_dir)
SCREEN=$(get screen_name)
[[ -f "$RUNTIME/deploy-receipt.json" ]]
if /usr/bin/screen -ls 2>/dev/null | grep -Fq ".$SCREEN"; then
  printf '%s\n' '{"status":"already_running"}'
  exit 0
fi
/usr/bin/screen -L -Logfile "$RUNTIME/screen.log" -dmS "$SCREEN" \
  /usr/bin/bash "$RUNTIME/runner.sh" "$RUNTIME/runtime.json"
printf '%s\n' '{"status":"started"}'