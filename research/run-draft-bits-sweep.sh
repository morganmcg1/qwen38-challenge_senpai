#!/usr/bin/env bash
# Research-only driver: cost of the compact draft readout (M=1, K=5120,
# N=98336, affine g64) as a function of quantization bits, through the
# vendored MLX the scored worker links.
#
#   research/run-draft-bits-sweep.sh TAG
#
# This is the cheap gate for the draft-head precision experiment. This
# checkout carries a crossrow specialization for
# `!batched && group_size == 64 && bits == 4 && out_vec_size >= 1024`, but it
# is an in-kernel branch on `ntg.x` with cases 2...9 only, and the host sets
# `ntg.x == M`. The shipped draft readout is M=1, so BOTH the 4-bit and 3-bit
# arms fall through to `qmv_fast_impl` and 3 bits is a pure byte saving. The
# M=2 arms in the sweep are the empirical check on that reading. If the 3-bit
# arm still loses, no model-side plumbing can rescue it.
#
# Holds benchmark.sh's own local run lock, so this never overlaps a
# model-holding run. The 40C cool gate is attempted and its outcome plus the
# start/end temperature are recorded with the samples.
set -euo pipefail

tag="${1:?usage: run-draft-bits-sweep.sh TAG [--skip-build]}"
skip_build="${2:-}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

LOCAL_RUN_LOCK_OWNED=""
local_run_guard_enabled() {
  [[ "${MLXFAST_LOCAL_RUN_GUARD:-1}" != "0" ]]
}
run_lock_definitions="$(
  awk '/^readonly RESIDENT_MODEL_PROCESS_PATTERN=/' benchmark.sh
  awk '/^local_run_lock_path\(\) \{/,/^\}/' benchmark.sh
  awk '/^acquire_local_run_lock\(\) \{/,/^\}/' benchmark.sh
  awk '/^release_local_run_lock\(\) \{/,/^\}/' benchmark.sh
  awk '/^list_resident_model_processes\(\) \{/,/^\}/' benchmark.sh
  awk '/^abort_if_model_already_resident\(\) \{/,/^\}/' benchmark.sh
)"
if ! eval "${run_lock_definitions}"; then
  echo "run-draft-bits-sweep.sh: could not evaluate benchmark.sh's local run guard; refusing to run unguarded" >&2
  exit 1
fi
for reused in \
  local_run_lock_path acquire_local_run_lock release_local_run_lock \
  list_resident_model_processes abort_if_model_already_resident
do
  if ! declare -F "${reused}" >/dev/null 2>&1; then
    echo "run-draft-bits-sweep.sh: could not reuse benchmark.sh's ${reused}(); refusing to run unguarded" >&2
    exit 1
  fi
done
if [[ -z "${RESIDENT_MODEL_PROCESS_PATTERN:-}" ]]; then
  echo "run-draft-bits-sweep.sh: benchmark.sh's RESIDENT_MODEL_PROCESS_PATTERN is empty; refusing to run unguarded" >&2
  exit 1
fi

cleanup() {
  release_local_run_lock
}
trap cleanup EXIT

acquire_local_run_lock
abort_if_model_already_resident

out_dir="${repo_root}/.mlxfast-private/draft-bits-sweep/${tag}"
rm -rf -- "${out_dir}"
mkdir -p "${out_dir}"

{
  echo "run-draft-bits-sweep: tag=${tag}"
  echo "run-draft-bits-sweep: head=$(git rev-parse HEAD) dirty=$(git status --porcelain | wc -l | tr -d ' ')"
  echo "run-draft-bits-sweep: host=$(sysctl -n machdep.cpu.brand_string) mem=$(sysctl -n hw.memsize)"
  date -u '+run-draft-bits-sweep: started_utc=%Y-%m-%dT%H:%M:%SZ'
} | tee "${out_dir}/identity.txt" >&2

# A full build ahead of the gate leaves the SoC hot enough that the gate's
# stall detector aborts before 40C. Build separately, then measure.
if [[ "${skip_build}" != "--skip-build" ]]; then
  swift build -c release --build-tests --force-resolved-versions -Xswiftc -enable-testing
  tools/build-mlx-metallib.sh --all-build-roots
fi

eval "$(
  awk '/^find_macmon\(\) \{/,/^\}/' benchmark.sh
  awk '/^local_gpu_temp\(\) \{/,/^\}/' benchmark.sh
)"
COOL_GATE_MACMON_BIN="$(find_macmon || true)"

# The arms are timed round-robin inside one process, so a thermal floor above
# the 40C gate biases every arm equally and cancels in the comparison the sweep
# exists to make. The gate still runs, but a stalled cool-down downgrades to a
# recorded start/end temperature instead of discarding the measurement.
echo "run-draft-bits-sweep: cool gate before sweep" >&2
if ./benchmark.sh --local-cool-gate-only; then
  echo "cool_gate=passed" | tee -a "${out_dir}/identity.txt" >&2
else
  echo "cool_gate=stalled_above_40C" | tee -a "${out_dir}/identity.txt" >&2
fi
echo "gpu_temp_c_before=$(local_gpu_temp || true)" | tee -a "${out_dir}/identity.txt" >&2

MLXFAST_RUN_QMV_BITS_SWEEP=1 \
MLXFAST_QMV_BITS_SWEEP_OUT="${out_dir}/bits.json" \
  swift test -c release --force-resolved-versions -Xswiftc -enable-testing \
  --filter sweepCompactDraftReadoutOverBits 2>&1 | tee "${out_dir}/bits.log"

echo "gpu_temp_c_after=$(local_gpu_temp || true)" | tee -a "${out_dir}/identity.txt" >&2
date -u '+run-draft-bits-sweep: finished_utc=%Y-%m-%dT%H:%M:%SZ' >&2
echo "run-draft-bits-sweep: artifacts in ${out_dir}" >&2
