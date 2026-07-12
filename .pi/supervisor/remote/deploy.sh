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

REPO=$(get repo)
RUNTIME=$(get runtime_dir)
BUNDLE=$(get bundle_path)
INPUTS=$(get input_archive_path)
COMMIT=$(get expected.commit)
EXPECTED_BUNDLE=$(get expected.bundle_sha256)
EXPECTED_INPUTS=$(get expected.input_archive_sha256)

mkdir -p "$RUNTIME/logs"
[[ $(sha256sum "$BUNDLE" | awk '{print $1}') == "$EXPECTED_BUNDLE" ]]
[[ $(sha256sum "$INPUTS" | awk '{print $1}') == "$EXPECTED_INPUTS" ]]

if [[ ! -d "$REPO/.git" ]]; then
  git clone --no-checkout "$BUNDLE" "$REPO" >>"$RUNTIME/logs/deploy.log" 2>&1
fi
git -C "$REPO" checkout --detach "$COMMIT" >>"$RUNTIME/logs/deploy.log" 2>&1
git -C "$REPO" diff --quiet
git -C "$REPO" diff --cached --quiet
mkdir -p "$REPO/data/processed/v3a"
tar -C "$REPO/data/processed/v3a" -xzf "$INPUTS"

if [[ ! -x "$REPO/.venv/bin/python" ]]; then
  "$PYTHON" -m venv --system-site-packages "$REPO/.venv"
fi
"$REPO/.venv/bin/python" -m pip install --disable-pip-version-check \
  'numpy==1.26.4' 'pandas==2.1.4' >>"$RUNTIME/logs/environment.log" 2>&1
(
  cd "$REPO"
  PYTHONPATH=src .venv/bin/python -m compileall -q src scripts tests
  PYTHONPATH=src .venv/bin/python -m unittest discover -s tests
)
printf '%s\n' "deployed commit=$COMMIT" >"$RUNTIME/deploy-receipt.json"
printf '%s\n' '{"status":"deployed"}'