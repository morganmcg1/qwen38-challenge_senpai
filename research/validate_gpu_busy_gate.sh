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

harness="${1:?usage: validate_gpu_busy_gate.sh HARNESS_BINARY ARM_DIR}"
arm_dir="${2:?usage: validate_gpu_busy_gate.sh HARNESS_BINARY ARM_DIR}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
check="${repo_root}/research/gpu_busy_check.py"

for required in "${harness}" "${arm_dir}/base.metal" "${arm_dir}/cand.metal"; do
  if [[ ! -e "${required}" ]]; then
    echo "validate_gpu_busy_gate.sh: missing ${required}" >&2
    exit 2
  fi
done

echo "== negative control: no load =="
idle_status=0
python3 "${check}" --seconds 8 || idle_status=$?
echo "idle_exit=${idle_status}"

echo "== positive control: sustained Metal load =="
load_log="$(mktemp)"
"${harness}" --base "${arm_dir}/base.metal" --cand "${arm_dir}/cand.metal" \
  --out /dev/null --widths 8 --pairs 40 --reps 60 --inner 20 \
  >"${load_log}" 2>&1 &
load_pid=$!
# Let the load reach steady state so the sampler is not racing library compilation,
# then confirm it is still alive: a harness that exited early would make an IDLE
# verdict look like gate blindness when there was simply no load to detect.
sleep 12
if ! kill -0 "${load_pid}" 2>/dev/null; then
  echo "validate_gpu_busy_gate.sh: load exited before sampling; no positive control" >&2
  tail -5 "${load_log}" >&2
  rm -f "${load_log}"
  exit 2
fi
# If the gate reads IDLE under real load, the next question is whether any driver
# counter moved at all, so record both accelerator classes while the load is live.
for probe_class in AGXAccelerator IOAccelerator; do
  echo "-- ${probe_class} under load --"
  ioreg -r -d 1 -c "${probe_class}" \
    | grep -oE '"(Device|Renderer|Tiler) Utilization %"=[0-9]+' | sort -u
done
busy_status=0
python3 "${check}" --seconds 10 || busy_status=$?
echo "busy_exit=${busy_status}"
load_alive_after="yes"
kill -0 "${load_pid}" 2>/dev/null || load_alive_after="no"
echo "load_alive_after_sampling=${load_alive_after}"
kill "${load_pid}" 2>/dev/null || true
wait "${load_pid}" 2>/dev/null || true
rm -f "${load_log}"

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
