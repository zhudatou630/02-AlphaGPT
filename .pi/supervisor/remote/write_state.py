#!/usr/bin/env python3
"""Atomically update the remote runner state."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--status")
    parser.add_argument("--phase")
    parser.add_argument("--detail")
    parser.add_argument("--exit-code", type=int)
    parser.add_argument("--runner-pid", type=int)
    parser.add_argument("--launch-action-id")
    parser.add_argument("--increment-launch", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    path = Path(config["state_file"])
    state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    timestamp = now()
    state.setdefault("version", 1)
    state.setdefault("run_id", config["run_id"])
    state.setdefault("created_at", timestamp)
    state.setdefault("launch_count", 0)
    state["sequence"] = int(state.get("sequence", 0)) + 1
    state["updated_at"] = timestamp
    if args.increment_launch:
        state["launch_count"] = int(state.get("launch_count", 0)) + 1
        state["last_launch_at"] = timestamp
    for key, value in (
        ("status", args.status),
        ("phase", args.phase),
        ("detail", args.detail),
        ("exit_code", args.exit_code),
        ("runner_pid", args.runner_pid),
        ("launch_action_id", args.launch_action_id),
    ):
        if value is not None:
            state[key] = value
    if state.get("status") == "running":
        state.setdefault("started_at", timestamp)
    if state.get("status") in {"awaiting_validation", "completed", "failed", "stopped"}:
        state["finished_at"] = timestamp
    atomic_write(path, state)
    print(json.dumps(state, ensure_ascii=False))


if __name__ == "__main__":
    main()