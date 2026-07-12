#!/usr/bin/env bash
set -euo pipefail
CONFIG=${1:?runtime config is required}
PYTHON=/root/miniconda3/bin/python
read_config() {
  "$PYTHON" - "$CONFIG" "$1" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
for key in sys.argv[2].split("."):
    value = value[key]
print(value)
PY
}
SCREEN=$(read_config screen_name)
if /usr/bin/screen -ls 2>/dev/null | grep -Fq ".$SCREEN"; then
  /usr/bin/screen -S "$SCREEN" -X quit
fi
printf '%s\n' '{"status":"stopped"}'