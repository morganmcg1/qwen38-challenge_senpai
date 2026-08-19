#!/usr/bin/env bash
# Positive and negative control for research/gpu_busy_check.py.
#
#   research/validate_gpu_busy_gate.sh HARNESS_BINARY
#
# The gate decides whether an E44 timing session may run, and relaxing it from a
# single-peak rule to a consecutive-run rule could in principle make it blind. A
# gate that never fires is worse than no gate, because it manufactures
# coordination evidence that does not exist. So assert both directions: BUSY while
# a real Metal workload occupies the GPU, IDLE once that workload is gone.
set -euo pipefail

harness="${1:?usage: validate_gpu_busy_gate.sh HARNESS_BINARY}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
check="${repo_root}/research/gpu_busy_check.py"

echo "== negative control: no load =="
idle_status=0
python3 "${check}" --seconds 8 || idle_status=$?
echo "idle_exit=${idle_status}"

echo "== positive control: sustained Metal load =="
"${harness}" --base --cand --widths 8 --pairs 40 --reps 60 --inner 20 \
  >/dev/null 2>&1 &
load_pid=$!
# Let the load reach steady state so the sampler is not racing library compilation.
sleep 4
busy_status=0
python3 "${check}" --seconds 10 || busy_status=$?
echo "busy_exit=${busy_status}"
kill "${load_pid}" 2>/dev/null || true
wait "${load_pid}" 2>/dev/null || true

echo "== recovery: load gone =="
sleep 3
recovered_status=0
python3 "${check}" --seconds 8 || recovered_status=$?
echo "recovered_exit=${recovered_status}"

echo
if [[ "${idle_status}" == "0" && "${busy_status}" == "1" && "${recovered_status}" == "0" ]]; then
  echo "GPU BUSY GATE VALIDATED: idle=IDLE load=BUSY recovered=IDLE"
  exit 0
fi
echo "GPU BUSY GATE NOT VALIDATED: idle=${idle_status} load=${busy_status} recovered=${recovered_status}"
echo "expected idle=0 load=1 recovered=0"
exit 1
