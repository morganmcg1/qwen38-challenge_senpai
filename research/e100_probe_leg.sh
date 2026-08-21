#!/usr/bin/env bash
# E100 -- run one isolated stream-collapse probe leg and record the identity
# tuple around it.
#
#   usage: research/e100_probe_leg.sh TAG ARM
#
#   TAG   output directory name under research/out/
#   ARM   `base` or `collapse`, recorded verbatim so an ABBA session can be
#         reconstructed from the artifacts alone.
#
# The session is a within-session ABBA comparison across two BUILDS, because
# the dispatch table is a compile-time template selection and cannot be
# switched by an environment variable. Each leg therefore records its own build
# witness: the dispatch entries for M = 5 and M = 9 read back out of the
# runtime-effective JIT twin.
#
# No thermal gate, no score. Entry and exit GPU temperature are recorded so the
# entry spread can be reported with the effect.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tag="${1:?usage: e100_probe_leg.sh TAG ARM}"
arm="${2:?usage: e100_probe_leg.sh TAG ARM}"
out_dir="research/out/${tag}"
mkdir -p "${out_dir}"

twin="Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp"

gpu_temp() {
  local macmon
  for macmon in "${MLXFAST_MACMON_BIN:-}" "${HOME}/bin/macmon" \
                /opt/homebrew/bin/macmon /usr/local/bin/macmon; do
    [[ -n "${macmon}" && -x "${macmon}" ]] || continue
    "${macmon}" pipe -s1 2>/dev/null | jq -r '.temp.gpu_temp_avg // empty'
    return 0
  done
  echo ""
}

entry_c="$(gpu_temp)"
start_iso="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

MLXFAST_RUN_E100_PROBE=1 \
MLXFAST_E100_OUT="${PWD}/${out_dir}/cells.json" \
swift test --force-resolved-versions \
  --filter E100StreamCollapseProbeTests 2>&1 | tee "${out_dir}/probe.log"
status="${PIPESTATUS[0]}"

exit_c="$(gpu_temp)"

{
  echo "experiment=e100-fewer-weight-streams-per-round"
  echo "arm=${arm}"
  echo "harness=local"
  echo "started_utc=${start_iso}"
  echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "git_head=$(git rev-parse HEAD)"
  echo "git_dirty=$(git status --porcelain | wc -l | tr -d ' ')"
  echo "twin_m5=$(grep -c 'qmv_fast_crossrow_affine4_g64_m<T, 5, 5, true>' "${twin}")"
  echo "twin_m9=$(grep -c 'qmv_fast_crossrow_affine4_g64_m<T, 9, 5, true>' "${twin}")"
  echo "twin_na_bound=$(grep -o 'wide multi-row QMV supports NA in \[2, [0-9]\]' "${twin}" | head -1)"
  echo "host=$(hostname)"
  echo "chip=$(sysctl -n machdep.cpu.brand_string)"
  echo "memory_gib=$(( $(sysctl -n hw.memsize) / 1073741824 ))"
  echo "toolchain=$(swift --version 2>&1 | head -1)"
  echo "gpu_temp_entry_c=${entry_c}"
  echo "gpu_temp_exit_c=${exit_c}"
  echo "cool_gate_passed_real_gate=false"
  echo "gate_qualified_for_timing=false"
  echo "timing_valid=false"
  echo "status=${status}"
} > "${out_dir}/meta.txt"

cat "${out_dir}/meta.txt"
exit "${status}"
