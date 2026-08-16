#!/usr/bin/env bash
# Research-only driver: cost of the compact draft readout (M=1, K=5120,
# N=98336, affine g64) as a function of quantization bits, through the
# vendored MLX the scored worker links.
#
#   research/run-draft-bits-sweep.sh TAG
#
# This is the cheap gate for the draft-head precision experiment. The 4-bit arm
# is not stock here: this checkout routes
# `!batched && group_size == 64 && bits == 4 && out_vec_size >= 1024` into
# `qmv_fast_crossrow_affine4_g64*`, so 3-bit/2-bit trade less weight traffic
# for the stock `qmv_fast_impl`. If that trade is a loss, no model-side
# plumbing can rescue it.
#
# Holds benchmark.sh's own local run lock and passes its 40C cool gate before
# the timed process, so this never overlaps a model-holding run.
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

echo "run-draft-bits-sweep: cool gate before sweep" >&2
./benchmark.sh --local-cool-gate-only

MLXFAST_RUN_QMV_BITS_SWEEP=1 \
MLXFAST_QMV_BITS_SWEEP_OUT="${out_dir}/bits.json" \
  swift test -c release --force-resolved-versions -Xswiftc -enable-testing \
  --filter sweepCompactDraftReadoutOverBits 2>&1 | tee "${out_dir}/bits.log"

date -u '+run-draft-bits-sweep: finished_utc=%Y-%m-%dT%H:%M:%SZ' >&2
echo "run-draft-bits-sweep: artifacts in ${out_dir}" >&2
