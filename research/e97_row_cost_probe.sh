#!/usr/bin/env bash
# E97 -- run one isolated per-row cost probe rung and record the identity
# tuple around it.
#
#   usage: research/e97_row_cost_probe.sh TAG
#
# The probe is a within-session counterbalanced comparison of slopes, so it runs
# under no thermal gate and reports no score. Entry and exit GPU temperature are
# recorded because the slope of a drifting session is only trustworthy when the
# drift is bounded and the ABBA block order can cancel it.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tag="${1:?usage: e97_row_cost_probe.sh TAG [row|peak|shape]}"
mode="${2:-row}"
out_dir="research/out/${tag}"
mkdir -p "${out_dir}"

run_row=0; run_peak=0; run_shape=0
case "${mode}" in
  row)   run_row=1 ;;
  peak)  run_peak=1 ;;
  shape) run_shape=1 ;;
  *) echo "unknown mode ${mode}" >&2; exit 2 ;;
esac

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

MLXFAST_RUN_E97_ROW_COST="${run_row}" \
MLXFAST_RUN_E97_PEAK="${run_peak}" \
MLXFAST_RUN_E97_SHAPE="${run_shape}" \
MLXFAST_E97_ROW_OUT="${PWD}/${out_dir}/row-cost.json" \
MLXFAST_E97_PEAK_OUT="${PWD}/${out_dir}/peak.json" \
MLXFAST_E97_SHAPE_OUT="${PWD}/${out_dir}/shape.json" \
swift test --force-resolved-versions \
  --filter E97VerifyRowCostTests 2>&1 | tee "${out_dir}/probe.log"
status="${PIPESTATUS[0]}"

exit_c="$(gpu_temp)"

{
  echo "experiment=e97-ranked-per-row-verify-cost"
  echo "rung=${mode}"
  echo "harness=local"
  echo "started_utc=${start_iso}"
  echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "git_head=$(git rev-parse HEAD)"
  echo "git_dirty=$(git status --porcelain | wc -l | tr -d ' ')"
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
