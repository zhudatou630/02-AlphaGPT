#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
REMOTE=autodl-898
REMOTE_SCRIPT=/root/profile-formal-s101.sh
LOCAL_SCRIPT="$ROOT/.pi/profile/profile-formal-s101.sh"
RESULTS_DIR="$ROOT/.pi/profile/results"

mkdir -p "$RESULTS_DIR"

ssh -o ConnectTimeout=15 "$REMOTE" 'test -d /root/02-AlphaGPT && test -d /root/autodl-tmp'
scp "$LOCAL_SCRIPT" "$REMOTE:$REMOTE_SCRIPT"
ssh "$REMOTE" "chmod 700 '$REMOTE_SCRIPT'"

set +e
ssh "$REMOTE" "bash '$REMOTE_SCRIPT'"
profile_status=$?
set -e

profile_root=$(ssh "$REMOTE" 'cat /root/autodl-tmp/v3a-profile-s101-latest.txt')
profile_name=$(basename "$profile_root")
remote_archive="/tmp/$profile_name.tgz"
local_archive="$RESULTS_DIR/$profile_name.tgz"

ssh "$REMOTE" "tar -C '$profile_root' --exclude='./runs' -czf '$remote_archive' ."
scp "$REMOTE:$remote_archive" "$local_archive"
tar -tzf "$local_archive" >/dev/null
printf 'profile_status=%s\nresult=%s\n' "$profile_status" "$local_archive"

set +e
ssh -o ConnectTimeout=15 "$REMOTE" 'sync && shutdown now'
shutdown_status=$?
set -e
if (( shutdown_status != 0 )); then
    echo "Shutdown command disconnected or returned $shutdown_status; verify instance state in AutoDL." >&2
fi

exit "$profile_status"