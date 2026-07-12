#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RUNTIME="$ROOT/.pi/supervisor/runtime.json"

load_ssh() {
  local login_file
  login_file=$(jq -er '.ssh_login_file' "$RUNTIME")
  SSH_PORT=$(awk '/^ssh -p / {print $3; exit}' "$login_file")
  SSH_TARGET=$(awk '/^ssh -p / {print $4; exit}' "$login_file")
  SSH_PASSWORD=$(awk -F'：' '/^密码：/ {print $2; exit}' "$login_file")
  [[ -n "$SSH_PORT" && -n "$SSH_TARGET" && -n "$SSH_PASSWORD" ]]
  export SSH_PORT SSH_TARGET SSH_PASSWORD
}

remote_exec() {
  load_ssh
  local command
  printf -v command '%q ' "$@"
  SSHPASS="$SSH_PASSWORD" sshpass -e ssh \
    -o ConnectTimeout=12 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 \
    -o StrictHostKeyChecking=yes -p "$SSH_PORT" "$SSH_TARGET" "$command"
}

scp_to_remote() {
  load_ssh
  SSHPASS="$SSH_PASSWORD" sshpass -e scp \
    -o ConnectTimeout=12 -o StrictHostKeyChecking=yes -P "$SSH_PORT" \
    "$1" "$SSH_TARGET:$2"
}

rsync_from_remote() {
  load_ssh
  mkdir -p "$2"
  SSHPASS="$SSH_PASSWORD" sshpass -e rsync -a --partial \
    -e "ssh -o ConnectTimeout=12 -o StrictHostKeyChecking=yes -p $SSH_PORT" \
    "$SSH_TARGET:$1" "$2"
}