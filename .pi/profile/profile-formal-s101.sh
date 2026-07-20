#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/02-AlphaGPT
PYTHON=/root/miniconda3/bin/python
RUN_ID=v3a-stage-d-formal-transformer-s101-00185964276c
SOURCE_RUN="$ROOT/data/processed/v3a/training/runs/$RUN_ID"
DATA_ROOT=/root/autodl-tmp
EXPECTED_HEAD=54716bc9efeb3c43dd14c7aaa50bfbd117981282
BATCH_SIZE=8192
PROFILE_BATCHES=3

timestamp=$(date -u +%Y%m%dT%H%M%SZ)
PROFILE_ROOT="$DATA_ROOT/v3a-profile-s101-$timestamp"
PROFILE_RUNS="$PROFILE_ROOT/runs"
PROFILE_RUN="$PROFILE_RUNS/$RUN_ID"
mkdir -p "$PROFILE_ROOT"
printf '%s\n' "$PROFILE_ROOT" > "$DATA_ROOT/v3a-profile-s101-latest.txt"

exec > >(tee -a "$PROFILE_ROOT/controller.log") 2>&1

echo "profile_root=$PROFILE_ROOT"
echo "started_at=$(date -Iseconds)"

cd "$ROOT"
actual_head=$(git rev-parse HEAD)
if [[ "$actual_head" != "$EXPECTED_HEAD" ]]; then
    echo "Refusing profile: remote HEAD $actual_head != $EXPECTED_HEAD" >&2
    exit 10
fi
if pgrep -f '[s]cripts/v3a/train_gpu.py' >/dev/null; then
    echo "Refusing profile: a train_gpu.py process is already running" >&2
    exit 11
fi
if [[ ! -f "$SOURCE_RUN/checkpoint_latest.pt" || ! -f "$SOURCE_RUN/attempts.jsonl" ]]; then
    echo "Refusing profile: formal seed 101 checkpoint or ledger is missing" >&2
    exit 12
fi
if [[ ! -d "$DATA_ROOT" ]]; then
    echo "Refusing profile: data disk path $DATA_ROOT is missing" >&2
    exit 13
fi

git status --short > "$PROFILE_ROOT/git-status-before.txt"
git diff --quiet HEAD -- src/alpha_etf/research_v3a scripts/v3a configs/v3a_stage_d_formal_topn.json || {
    echo "Refusing profile: tracked V3A code or protocol is dirty" >&2
    exit 14
}

{
    uname -a
    echo "nproc=$(nproc)"
    echo "nproc_all=$(nproc --all)"
    echo -n "cgroup_cpu_max="
    cat /sys/fs/cgroup/cpu.max 2>/dev/null || true
    echo -n "cgroup_cpuset="
    cat /sys/fs/cgroup/cpuset.cpus.effective 2>/dev/null || true
    echo -n "cgroup_memory_max="
    cat /sys/fs/cgroup/memory.max 2>/dev/null || true
    lscpu
    free -h
    df -h "$ROOT" "$DATA_ROOT"
    nvidia-smi
    "$PYTHON" --version
} > "$PROFILE_ROOT/machine.txt" 2>&1

du -sb "$SOURCE_RUN" > "$PROFILE_ROOT/formal-source-size-before.txt"
sha256sum "$SOURCE_RUN/checkpoint_latest.pt" "$SOURCE_RUN/attempts.jsonl" \
    > "$PROFILE_ROOT/formal-source-before.sha256"

source_kb=$(du -sk "$SOURCE_RUN" | awk '{print $1}')
free_kb=$(df -Pk "$DATA_ROOT" | awk 'NR==2 {print $4}')
reserve_kb=$((5 * 1024 * 1024))
if (( free_kb < source_kb + reserve_kb )); then
    echo "Refusing profile: insufficient data-disk space" >&2
    exit 15
fi

mkdir -p "$PROFILE_RUNS"
cp -a --reflink=auto "$SOURCE_RUN" "$PROFILE_RUN"

checkpoint_attempts=$(
    "$PYTHON" - "$PROFILE_RUN/checkpoint_latest.pt" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(int(checkpoint["attempt_count"]))
PY
)
stop_after=$((checkpoint_attempts + BATCH_SIZE * PROFILE_BATCHES))
if (( stop_after > 8000000 )); then
    echo "Refusing profile: three batches would exceed the formal attempt budget" >&2
    exit 16
fi
printf 'checkpoint_attempts=%s\nstop_after=%s\n' \
    "$checkpoint_attempts" "$stop_after" > "$PROFILE_ROOT/profile-range.txt"

monitor_system() {
    local target_pid=$1
    while kill -0 "$target_pid" 2>/dev/null; do
        echo "timestamp=$(date -Iseconds)"
        ps -eo pid,ppid,pcpu,pmem,rss,vsz,stat,comm,args --sort=-pcpu | sed -n '1,30p'
        free -b
        df -B1 "$ROOT" "$DATA_ROOT"
        sleep 1
    done
}

monitor_gpu() {
    while true; do
        nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.free,power.draw \
            --format=csv,noheader,nounits
        sleep 1
    done
}

set +e
profile_started_ns=$(date +%s%N)
"$PYTHON" -m cProfile -o "$PROFILE_ROOT/profile.pstats" \
    scripts/v3a/train_gpu.py \
    --protocol-file configs/v3a_stage_d_formal_topn.json \
    --seed 101 \
    --train-view-dir data/processed/v3a/stage_d/formal_topn_train_view \
    --stage-c-report data/processed/v3a/stage_c_reports/v3a-stage-c-20260711-01.json \
    --binding-file data/processed/v3a/stage_d/formal_topn_binding.json \
    --out-dir "$PROFILE_RUNS" \
    --resume \
    --stop-after "$stop_after" \
    > "$PROFILE_ROOT/train.log" 2>&1 &
profile_pid=$!
monitor_system "$profile_pid" > "$PROFILE_ROOT/system.log" 2>&1 &
system_pid=$!
monitor_gpu > "$PROFILE_ROOT/gpu.csv" 2> "$PROFILE_ROOT/gpu-monitor.err" &
gpu_pid=$!

wait "$profile_pid"
profile_status=$?
profile_finished_ns=$(date +%s%N)
printf 'started_ns=%s\nfinished_ns=%s\nelapsed_seconds=%s\n' \
    "$profile_started_ns" \
    "$profile_finished_ns" \
    "$(( (profile_finished_ns - profile_started_ns) / 1000000000 ))" \
    > "$PROFILE_ROOT/time.txt"
kill "$system_pid" "$gpu_pid" 2>/dev/null
wait "$system_pid" "$gpu_pid" 2>/dev/null
set -e

if [[ -s "$PROFILE_ROOT/profile.pstats" ]]; then
    "$PYTHON" - "$PROFILE_ROOT/profile.pstats" > "$PROFILE_ROOT/profile-top.txt" <<'PY'
import pstats
import sys

stats = pstats.Stats(sys.argv[1])
stats.strip_dirs().sort_stats("cumulative").print_stats(120)
PY
fi

du -sb "$PROFILE_RUN" > "$PROFILE_ROOT/profile-run-size-after.txt"
sha256sum "$SOURCE_RUN/checkpoint_latest.pt" "$SOURCE_RUN/attempts.jsonl" \
    > "$PROFILE_ROOT/formal-source-after.sha256"
if ! cmp -s "$PROFILE_ROOT/formal-source-before.sha256" "$PROFILE_ROOT/formal-source-after.sha256"; then
    echo "ERROR: formal source hashes changed during profiling" >&2
    profile_status=20
fi

{
    echo "profile_status=$profile_status"
    echo "finished_at=$(date -Iseconds)"
    echo "profile_root=$PROFILE_ROOT"
} | tee "$PROFILE_ROOT/PROFILE_STATUS.txt"

exit "$profile_status"