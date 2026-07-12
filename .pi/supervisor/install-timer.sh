#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
UNIT_DIR="$HOME/.config/systemd/user"
SERVICE=alphagpt-supervisor-3227c7e8e4.service
TIMER=alphagpt-supervisor-3227c7e8e4.timer

mkdir -p "$UNIT_DIR"
"$ROOT/.pi/supervisor/supervisor.py" create-session
install -m 0644 "$ROOT/.pi/supervisor/systemd/$SERVICE" "$UNIT_DIR/$SERVICE"
install -m 0644 "$ROOT/.pi/supervisor/systemd/$TIMER" "$UNIT_DIR/$TIMER"
systemctl --user daemon-reload
systemctl --user enable --now "$TIMER"
printf '%s\n' "monitor timer enabled"