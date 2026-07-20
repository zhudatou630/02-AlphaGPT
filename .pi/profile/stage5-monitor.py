#!/usr/bin/env python3
"""Sample container cgroup, Stage D process, GPU, and run-directory resources."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time


CGROUP = Path("/sys/fs/cgroup")


def read_int(path: Path) -> int | None:
    try:
        value = path.read_text(encoding="utf-8").strip()
        return None if value == "max" else int(value)
    except (FileNotFoundError, ValueError):
        return None


def read_cpu_stat() -> dict[str, int]:
    result: dict[str, int] = {}
    for line in (CGROUP / "cpu.stat").read_text(encoding="utf-8").splitlines():
        key, value = line.split()
        result[f"cpu_{key}"] = int(value)
    return result


def stage_d_process_rss() -> tuple[int, int]:
    total = 0
    count = 0
    for raw_pid in (CGROUP / "cgroup.procs").read_text(encoding="utf-8").split():
        proc = Path("/proc") / raw_pid
        try:
            command = (proc / "cmdline").read_bytes().replace(b"\0", b" ")
            if not any(
                marker in command
                for marker in (
                    b"run_stage_d.py",
                    b"multiprocessing.spawn",
                    b"multiprocessing.resource_tracker",
                )
            ):
                continue
            for line in (proc / "status").read_text(encoding="utf-8").splitlines():
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1]) * 1024
                    count += 1
                    break
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return total, count


def gpu_metrics() -> dict[str, float | int | str | None]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, text=True, timeout=5).strip()
        name, utilization, used, total, power = [part.strip() for part in output.split(",")]
        return {
            "gpu_name": name,
            "gpu_utilization_percent": int(utilization),
            "gpu_memory_used_mib": int(used),
            "gpu_memory_total_mib": int(total),
            "gpu_power_watts": float(power),
        }
    except (OSError, subprocess.SubprocessError, ValueError):
        return {
            "gpu_name": None,
            "gpu_utilization_percent": None,
            "gpu_memory_used_mib": None,
            "gpu_memory_total_mib": None,
            "gpu_power_watts": None,
        }


def directory_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return total
    for root, _directories, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except FileNotFoundError:
                continue
    return total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.stop_file.unlink(missing_ok=True)
    with args.output.open("a", encoding="utf-8") as handle:
        while not args.stop_file.exists():
            rss, process_count = stage_d_process_rss()
            payload = {
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "monotonic_seconds": time.monotonic(),
                "cgroup_memory_current_bytes": read_int(CGROUP / "memory.current"),
                "cgroup_memory_peak_bytes": read_int(CGROUP / "memory.peak"),
                "stage_d_process_rss_bytes": rss,
                "stage_d_process_count": process_count,
                "run_root_bytes": directory_bytes(args.run_root),
                **read_cpu_stat(),
                **gpu_metrics(),
            }
            handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
            handle.flush()
            time.sleep(args.interval)


if __name__ == "__main__":
    main()