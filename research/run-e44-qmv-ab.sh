#!/usr/bin/env bash
# E44 section 7.3 driver: paired base-vs-candidate microbenchmark of the scored
# affine-4/g64 `qmv_fast` width tree.
#
#   research/run-e44-qmv-ab.sh TAG BASE_SHA [--widths L] [--pairs N]
#                              [--reps N] [--inner N]
#
# Emits both runtime-effective JIT arm sources, builds the harness, and runs one
# session in which the arms alternate ABBA. Holds benchmark.sh's own local run
# lock so this never overlaps a model-holding run, and records the real cool-gate
# outcome plus entry and exit GPU temperature verbatim.
set -euo pipefail

tag="${1:?usage: run-e44-qmv-ab.sh TAG BASE_SHA}"
base_sha="${2:?usage: run-e44-qmv-ab.sh TAG BASE_SHA}"
shift 2

widths="1,2,3,4,5,6,7,8,9"
pairs=5
reps=25
inner=20
probe=0
coverage=0
while [[ $# -gt 0 ]]; do
  case "${1}" in
    --widths) widths="${2:?--widths needs a list}"; shift 2 ;;
    --pairs) pairs="${2:?--pairs needs a count}"; shift 2 ;;
    --reps) reps="${2:?--reps needs a count}"; shift 2 ;;
    --inner) inner="${2:?--inner needs a count}"; shift 2 ;;
    --probe) probe="${2:?--probe needs a k count}"; shift 2 ;;
    --coverage) coverage="${2:?--coverage needs a word stride}"; shift 2 ;;
    *) echo "run-e44-qmv-ab.sh: unknown argument ${1}" >&2; exit 2 ;;
  esac
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

# --- reuse benchmark.sh's local run guard -------------------------------------
# Same fail-closed extraction contract as research/run-qmv-curve.sh: take the
# definitions, then verify each name really got defined, so a refactor of
# benchmark.sh stops this script instead of letting it run unguarded.
LOCAL_RUN_LOCK_OWNED=""
run_lock_definitions="$(
  awk '/^readonly RESIDENT_MODEL_PROCESS_PATTERN=/' benchmark.sh
  awk '/^local_run_lock_path\(\) \{/,/^\}/' benchmark.sh
  awk '/^acquire_local_run_lock\(\) \{/,/^\}/' benchmark.sh
  awk '/^release_local_run_lock\(\) \{/,/^\}/' benchmark.sh
  awk '/^list_resident_model_processes\(\) \{/,/^\}/' benchmark.sh
  awk '/^abort_if_model_already_resident\(\) \{/,/^\}/' benchmark.sh
)"
if ! eval "${run_lock_definitions}"; then
  echo "run-e44-qmv-ab.sh: could not evaluate benchmark.sh's local run guard; refusing to run unguarded" >&2
  exit 1
fi
for reused in \
  local_run_lock_path acquire_local_run_lock release_local_run_lock \
  list_resident_model_processes abort_if_model_already_resident
do
  if ! declare -F "${reused}" >/dev/null 2>&1; then
    echo "run-e44-qmv-ab.sh: could not reuse benchmark.sh's ${reused}(); refusing to run unguarded" >&2
    exit 1
  fi
done
if [[ -z "${RESIDENT_MODEL_PROCESS_PATTERN:-}" ]]; then
  echo "run-e44-qmv-ab.sh: benchmark.sh's RESIDENT_MODEL_PROCESS_PATTERN is empty; refusing to run unguarded" >&2
  exit 1
fi

cleanup() { release_local_run_lock; }
trap cleanup EXIT

acquire_local_run_lock
abort_if_model_already_resident

out_dir="${repo_root}/.mlxfast-private/e44-qmv-ab/${tag}"
rm -rf -- "${out_dir}"
mkdir -p "${out_dir}"

eval "$(
  awk '/^find_macmon\(\) \{/,/^\}/' benchmark.sh
  awk '/^local_gpu_temp\(\) \{/,/^\}/' benchmark.sh
)"
COOL_GATE_MACMON_BIN="$(find_macmon || true)"

{
  echo "tag=${tag}"
  echo "base_sha=${base_sha}"
  echo "head=$(git rev-parse HEAD)"
  echo "dirty_files=$(git status --porcelain | wc -l | tr -d ' ')"
  echo "host_cpu=$(sysctl -n machdep.cpu.brand_string)"
  echo "host_mem=$(sysctl -n hw.memsize)"
  echo "metal_toolchain=$(xcrun metal --version 2>&1 | head -1)"
  echo "widths=${widths} pairs=${pairs} reps=${reps} inner=${inner}"
  date -u '+started_utc=%Y-%m-%dT%H:%M:%SZ'
} | tee "${out_dir}/identity.txt" >&2

# --- arm sources --------------------------------------------------------------
# Both arms are the RUNTIME-EFFECTIVE JIT string, not the readable .metal, so the
# measured kernels are the ones the scored worker would actually compile.
cell='affine_qmv_fast<bfloat16_t, 64, 4, false>'
python3 research/jit_string_compile.py --emit "${out_dir}/base.metal" \
  --rev "${base_sha}" "${cell}" | tee -a "${out_dir}/identity.txt" >&2
python3 research/jit_string_compile.py --emit "${out_dir}/cand.metal" \
  "${cell}" | tee -a "${out_dir}/identity.txt" >&2
{
  echo "base_metal_sha256=$(shasum -a 256 "${out_dir}/base.metal" | cut -d' ' -f1)"
  echo "cand_metal_sha256=$(shasum -a 256 "${out_dir}/cand.metal" | cut -d' ' -f1)"
  echo "arm_source_diff_hunks=$(diff "${out_dir}/base.metal" "${out_dir}/cand.metal" | grep -cE '^[0-9]' || true)"
} | tee -a "${out_dir}/identity.txt" >&2

clang -fobjc-arc -O2 -framework Metal -framework Foundation \
  -o "${out_dir}/e44_qmv_ab" research/e44_qmv_ab.m

# --- thermal record -----------------------------------------------------------
# This host's idle GPU floor sits above benchmark.sh's 40C target, so the real
# gate cannot pass here. program.md permits an ungated local timed arm only when
# the arms are ABBA-counterbalanced in one session, entry and exit temperature
# are recorded, and the two false flags are preserved verbatim. All three hold:
# the ABBA alternation is INSIDE the harness process, at roughly 100ms
# granularity, so drift between the paired arms is far smaller than between
# separate sessions.
if ./benchmark.sh --local-cool-gate-only; then
  cool_gate_real="passed"
  gate_qualified="true"
else
  cool_gate_real="stalled_above_40C"
  gate_qualified="false"
fi
entry_temp="$(local_gpu_temp || true)"
{
  echo "cool_gate_real_outcome=${cool_gate_real}"
  echo "cool_gate_passed_real_gate=$([[ "${cool_gate_real}" == passed ]] && echo true || echo false)"
  echo "gate_qualified_for_timing=${gate_qualified}"
  echo "gpu_temp_c_entry=${entry_temp:-unknown}"
} | tee -a "${out_dir}/identity.txt" >&2

set +e
"${out_dir}/e44_qmv_ab" \
  --base "${out_dir}/base.metal" \
  --cand "${out_dir}/cand.metal" \
  --out "${out_dir}/ab.json" \
  --widths "${widths}" --pairs "${pairs}" --reps "${reps}" --inner "${inner}" \
  --probe "${probe}" --coverage "${coverage}" \
  2>&1 | tee "${out_dir}/ab.log"
harness_status="${PIPESTATUS[0]}"
set -e

exit_temp="$(local_gpu_temp || true)"
{
  echo "gpu_temp_c_exit=${exit_temp:-unknown}"
  echo "harness_exit=${harness_status}"
  date -u '+finished_utc=%Y-%m-%dT%H:%M:%SZ'
} | tee -a "${out_dir}/identity.txt" >&2

echo "run-e44-qmv-ab.sh: results in ${out_dir}" >&2
exit "${harness_status}"
