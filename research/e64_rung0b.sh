#!/usr/bin/env bash
# E64 rung 0b: the dose. Does forcing `acc` into private memory at NA=5, where
# the ladder is on-model, reproduce the measured NA=5 -> 6 step?
#
# One session: regenerate the arms from the shipped kernel, certify them from
# AIR, emit the JIT-style source, build the harness, pass the real 40C cool gate,
# then time the arms in palindrome order inside one process.
#
#   research/e64_rung0b.sh [--na N] [--reps N] [--shape NAME] [--skip-gate]
#                          [--target-bytes X] [--out PATH] [--log PATH]
set -uo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo}"

na="5"
reps="21"
target_bytes="24e9"
artifacts="research/e64-artifacts"
out="${artifacts}/rung0b-timing.json"
log="${artifacts}/rung0b.log"
build="/tmp/e64-build"
shape=""
skip_gate=""

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --na) na="$2"; shift 2 ;;
    --reps) reps="$2"; shift 2 ;;
    --shape) shape="$2"; shift 2 ;;
    --target-bytes) target_bytes="$2"; shift 2 ;;
    --out) out="$2"; shift 2 ;;
    --log) log="$2"; shift 2 ;;
    --skip-gate) skip_gate="1"; shift ;;
    *) echo "e64_rung0b: unknown argument $1" >&2; exit 2 ;;
  esac
done
mkdir -p "${artifacts}" "${build}" "$(dirname "${out}")"

macmon_bin="${MLXFAST_MACMON_BIN:-${HOME}/bin/macmon}"
sample_thermal() {
  [[ -x "${macmon_bin}" ]] || { echo "unavailable"; return 0; }
  "${macmon_bin}" pipe -s1 2>/dev/null \
    | jq -r '"gpu_temp=\(.temp.gpu_temp_avg // "?")C cpu_temp=\(.temp.cpu_temp_avg // "?")C gpu_power=\(.gpu_power // "?")W"' \
      2>/dev/null || echo "unreadable"
}

{
  echo "e64_rung0b head=$(git rev-parse HEAD) dirty=$(git status --porcelain | wc -l | tr -d ' ')"
  echo "e64_rung0b host=$(hostname -s) chip=$(sysctl -n machdep.cpu.brand_string)"
  echo "e64_rung0b na=${na} reps=${reps} target_bytes=${target_bytes}"
  echo "e64_rung0b started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} | tee -a "${log}"

python3 research/e64_wide_gen.py --check | tee -a "${log}" || exit 1
python3 research/e64_air_census.py --na 5 6 \
  --out "${artifacts}/rung0b-air.json" | tee -a "${log}" || exit 1
python3 research/e64_emit_arms.py --na "${na}" \
  --out "${build}/arms_na${na}.metal" | tee -a "${log}" || exit 1
clang -fobjc-arc -O2 -framework Metal -framework Foundation \
  -o "${build}/e64_cell_ab" research/e64_cell_ab.m 2>&1 \
  | grep -v 'deprecated\|folding-constant\|warnings generated\|^ \|^$\|\^' \
  | tee -a "${log}"
[[ -x "${build}/e64_cell_ab" ]] || { echo "e64_rung0b: harness build failed" | tee -a "${log}"; exit 1; }

if [[ -z "${skip_gate}" ]]; then
  echo "e64_rung0b cool_gate before=$(sample_thermal)" | tee -a "${log}"
  if ./benchmark.sh --local-cool-gate-only >>"${log}" 2>&1; then
    echo "e64_rung0b cool_gate_passed_real_gate=true" | tee -a "${log}"
  else
    echo "e64_rung0b cool_gate_passed_real_gate=false gate_qualified_for_timing=false" \
      | tee -a "${log}"
  fi
else
  echo "e64_rung0b cool_gate=skipped_by_request gate_qualified_for_timing=false" \
    | tee -a "${log}"
fi
echo "e64_rung0b entry_thermal $(sample_thermal)" | tee -a "${log}"

shape_args=()
[[ -n "${shape}" ]] && shape_args=(--shape "${shape}")
MLXFAST_MACMON_BIN="${macmon_bin}" "${build}/e64_cell_ab" \
  --source "${build}/arms_na${na}.metal" \
  --na "${na}" --reps "${reps}" --warmup-reps 1 \
  --target-bytes "${target_bytes}" \
  ${shape_args[@]+"${shape_args[@]}"} \
  --out "${out}" 2>&1 | tee -a "${log}"
status="${PIPESTATUS[0]}"

{
  echo "e64_rung0b exit_thermal $(sample_thermal)"
  echo "e64_rung0b finished=$(date -u +%Y-%m-%dT%H:%M:%SZ) status=${status}"
} | tee -a "${log}"
exit "${status}"
