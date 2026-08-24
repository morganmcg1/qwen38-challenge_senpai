#!/usr/bin/env bash
# Research-only driver for E170: halve the register block of the wide
# affine-4/group-64 QMV at the deficient widths.
#
#   research/run-e170-rows-per-simd.sh TAG [--exactness-only] [--screen-only]
#                                      [--widths L] [--reps N] [--inner N]
#
# Rung 1 is the bit-exactness gate and is untimed. Rung 2 is the matched
# directional screen: both arms run in ONE process against ONE build, so the
# only difference between them is the `ROWS_PER_SIMD` template argument.
#
# Holds benchmark.sh's own local run lock for the whole measurement, so this
# never overlaps a model-holding run, and passes benchmark.sh's cool gate
# before the timed rung.
set -euo pipefail

tag="${1:?usage: run-e170-rows-per-simd.sh TAG}"
shift

run_exactness=1
run_screen=1
screen_widths=""
screen_reps=""
screen_inner=""
while [[ $# -gt 0 ]]; do
  case "${1}" in
    --exactness-only) run_screen=""; shift ;;
    --screen-only) run_exactness=""; shift ;;
    --widths) screen_widths="${2:?--widths needs a comma-separated list}"; shift 2 ;;
    --reps) screen_reps="${2:?--reps needs a count}"; shift 2 ;;
    --inner) screen_inner="${2:?--inner needs a count}"; shift 2 ;;
    *) echo "run-e170-rows-per-simd.sh: unknown argument ${1}" >&2; exit 2 ;;
  esac
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

# Reuse benchmark.sh's local run guard, then verify each name really got
# defined, so a refactor of benchmark.sh fails closed here instead of running
# unguarded.
LOCAL_RUN_LOCK_OWNED=""
local_run_guard_enabled() { [[ "${MLXFAST_LOCAL_RUN_GUARD:-1}" != "0" ]]; }
run_lock_definitions="$(
  awk '/^readonly RESIDENT_MODEL_PROCESS_PATTERN=/' benchmark.sh
  awk '/^local_run_lock_path\(\) \{/,/^\}/' benchmark.sh
  awk '/^acquire_local_run_lock\(\) \{/,/^\}/' benchmark.sh
  awk '/^release_local_run_lock\(\) \{/,/^\}/' benchmark.sh
  awk '/^list_resident_model_processes\(\) \{/,/^\}/' benchmark.sh
  awk '/^abort_if_model_already_resident\(\) \{/,/^\}/' benchmark.sh
)"
eval "${run_lock_definitions}"
for reused in local_run_lock_path acquire_local_run_lock release_local_run_lock \
              list_resident_model_processes abort_if_model_already_resident; do
  declare -F "${reused}" >/dev/null 2>&1 || {
    echo "run-e170-rows-per-simd.sh: could not reuse benchmark.sh's ${reused}(); refusing to run unguarded" >&2
    exit 1
  }
done

cleanup() { release_local_run_lock; }
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

acquire_local_run_lock
abort_if_model_already_resident

out_dir="${repo_root}/.mlxfast-private/e170/${tag}"
rm -rf -- "${out_dir}"
mkdir -p "${out_dir}"

{
  echo "e170: tag=${tag}"
  echo "e170: head=$(git rev-parse HEAD) dirty=$(git status --porcelain | wc -l | tr -d ' ')"
  echo "e170: host=$(sysctl -n machdep.cpu.brand_string) mem=$(sysctl -n hw.memsize)"
  echo "e170: widths=${screen_widths:-default} reps=${screen_reps:-default} inner=${screen_inner:-default}"
  date -u '+e170: started_utc=%Y-%m-%dT%H:%M:%SZ'
} | tee "${out_dir}/identity.txt" >&2

swift build -c release --build-tests --force-resolved-versions -Xswiftc -enable-testing
# The xctest bundle is a location Cmlx does not search by default; without this
# the first MLXArray fails to load the default metallib.
tools/build-mlx-metallib.sh --all-build-roots

eval "$(
  awk '/^find_macmon\(\) \{/,/^\}/' benchmark.sh
  awk '/^local_gpu_temp\(\) \{/,/^\}/' benchmark.sh
)"
COOL_GATE_MACMON_BIN="$(find_macmon || true)"

if [[ -n "${run_exactness}" ]]; then
  MLXFAST_RUN_E170_EXACTNESS=1 \
  MLXFAST_E170_EXACTNESS_OUT="${out_dir}/exactness.json" \
    swift test -c release --force-resolved-versions -Xswiftc -enable-testing \
    --filter E170RowsPerSimdExactnessTests 2>&1 | tee "${out_dir}/exactness.log"
  [[ -s "${out_dir}/exactness.json" ]] || {
    echo "run-e170-rows-per-simd.sh: the exactness rung produced no digests" >&2
    exit 1
  }
fi

if [[ -n "${run_screen}" ]]; then
  echo "e170: cool gate before the timed screen" >&2
  if ./benchmark.sh --local-cool-gate-only; then
    echo "cool_gate_screen=passed" | tee -a "${out_dir}/identity.txt" >&2
  else
    echo "cool_gate_screen=stalled_above_40C" | tee -a "${out_dir}/identity.txt" >&2
  fi
  echo "gpu_temp_c_before_screen=$(local_gpu_temp || true)" \
    | tee -a "${out_dir}/identity.txt" >&2

  MLXFAST_RUN_E170_SCREEN=1 \
  MLXFAST_E170_SCREEN_OUT="${out_dir}/screen.json" \
  MLXFAST_E170_SCREEN_WIDTHS="${screen_widths}" \
  MLXFAST_E170_SCREEN_REPS="${screen_reps}" \
  MLXFAST_E170_SCREEN_INNER="${screen_inner}" \
    swift test -c release --force-resolved-versions -Xswiftc -enable-testing \
    --filter E170RowsPerSimdScreenTests 2>&1 | tee "${out_dir}/screen.log"
  [[ -s "${out_dir}/screen.json" ]] || {
    echo "run-e170-rows-per-simd.sh: the screen rung produced no timings" >&2
    exit 1
  }
  echo "gpu_temp_c_after_screen=$(local_gpu_temp || true)" \
    | tee -a "${out_dir}/identity.txt" >&2
fi

date -u '+e170: finished_utc=%Y-%m-%dT%H:%M:%SZ' >&2
echo "e170: artifacts in ${out_dir}" >&2
