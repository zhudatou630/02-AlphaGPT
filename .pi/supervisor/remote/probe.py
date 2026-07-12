#!/usr/bin/env python3
"""Read-only remote probe for one V3A Stage D pilot."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
from typing import Any


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def process_alive(pid: Any) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def line_count(path: Path) -> int:
    if not path.exists():
        return 0
    result = subprocess.run(
        ["/usr/bin/wc", "-l", str(path)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return int(result.stdout.split()[0]) if result.returncode == 0 else 0


def screen_alive(name: str) -> bool:
    try:
        result = subprocess.run(
            ["/usr/bin/screen", "-ls"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    except OSError:
        return False
    return f".{name}\t" in result.stdout or f".{name} " in result.stdout


def gpu_evidence() -> dict[str, Any]:
    try:
        result = subprocess.run(
            [
                "/usr/bin/nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return {"available": False}
    if result.returncode != 0 or not result.stdout.strip():
        return {"available": False}
    fields = [item.strip() for item in result.stdout.strip().split(",")]
    if len(fields) != 4:
        return {"available": False}
    return {
        "available": True,
        "name": fields[0],
        "memory_used_mib": int(fields[1]),
        "memory_total_mib": int(fields[2]),
        "temperature_c": int(fields[3]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    state_path = Path(config["state_file"])
    state = load_json(state_path, {})
    repo = Path(config["repo"])
    runtime_dir = Path(config["runtime_dir"])
    run_dir = repo / "data/processed/v3a/training/runs" / config["run_id"]
    ledger = run_dir / "attempts.jsonl"
    training = load_json(run_dir / "training_summary.json", {})
    funnel = load_json(run_dir / "funnel/summary.json", {})
    screen = screen_alive(config["screen_name"])
    runner = process_alive(state.get("runner_pid"))
    phase = str(state.get("phase", "not-deployed"))
    status = str(state.get("status", "not-deployed"))
    attempts = line_count(ledger)

    if status == "awaiting_validation":
        observed_state = "completed"
        summary_text = "Stage D pilot finished; local Agent should inspect the artifacts"
    elif status == "failed":
        observed_state = "needs_attention"
        summary_text = f"Remote Stage D runner failed during {phase} with exit {state.get('exit_code')}"
    elif status == "running" and screen and runner:
        observed_state = "running"
        summary_text = f"Remote Stage D runner is active in {phase}"
    elif status == "running":
        observed_state = "stopped"
        summary_text = f"Runner says running in {phase}, but screen/PID is absent"
    elif status in {"deployed", "not-deployed"}:
        observed_state = "created"
        summary_text = f"Remote Stage D run is {status}"
    else:
        observed_state = "stopped"
        summary_text = f"Remote Stage D runner status is {status}"

    progress_key = (
        f"{phase}:attempts={attempts}:training={training.get('status')}:"
        f"funnel={funnel.get('status')}"
    )
    payload = {
        "run_id": config["run_id"],
        "state": observed_state,
        "summary": summary_text,
        "progress_key": progress_key,
        "fingerprint": f"{observed_state}:{progress_key}",
        "incident_id": state.get("updated_at") or f"{status}:{phase}",
        "evidence": {
            "runner_state_file": str(state_path),
            "runner_status": status,
            "runner_phase": phase,
            "runner_exit_code": state.get("exit_code"),
            "runner_pid_alive": runner,
            "screen_alive": screen,
            "launch_count": state.get("launch_count", 0),
            "attempt_ledger_lines": attempts,
            "training_status": training.get("status"),
            "training_resume_count": training.get("resume_count"),
            "funnel_status": funnel.get("status"),
            "selected_count": funnel.get("selected_count"),
            "gpu": gpu_evidence(),
            "runner_log": str(runtime_dir / "runner.log"),
        },
    }
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()