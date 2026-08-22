#!/usr/bin/env bash
# E117 -- run one shipped-width-frame probe session and record the identity
# tuple around it.
#
#   usage: research/e117_probe.sh TAG [shapes] [widths] [blocks]
#
#   TAG      output directory name under research/out/
#   shapes   comma list of preset names or `name:N:K` triples,
#            default `mlp.gate_up,control.small`
#   widths   comma list of row counts M, default `1,2,3,4,5,6,7,8,9`
#   blocks   palindromic blocks per cell, default 6
#
# Every arm is timed forward and then in reverse inside one block, so monotone
# thermal drift cancels to first order. Temperature is sampled only at block
# boundaries and every sample is followed by a discarded fixed-duration ramp
# burst, which is the harness-defect-16 fix. No thermal gate, no score: entry
# and exit GPU temperature are recorded per block by the probe itself, and
# `cool_gate_passed_real_gate=false` plus `gate_qualified_for_timing=false` are
# preserved verbatim.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tag="${1:?usage: e117_probe.sh TAG [shapes] [widths] [blocks]}"
shapes="${2:-mlp.gate_up,control.small}"
widths="${3:-1,2,3,4,5,6,7,8,9}"
blocks="${4:-6}"
out_dir="research/out/${tag}"
mkdir -p "${out_dir}"

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

config="${MLXFAST_E117_CONFIG:-release}"

MLXFAST_RUN_E117_PROBE=1 \
MLXFAST_E117_SHAPES="${shapes}" \
MLXFAST_E117_WIDTHS="${widths}" \
MLXFAST_E117_BLOCKS="${blocks}" \
MLXFAST_E117_OUT="${PWD}/${out_dir}/cells.json" \
swift test -c "${config}" --force-resolved-versions \
  --filter E117WidthFrameProbeTests 2>&1 | tee "${out_dir}/probe.log"
status="${PIPESTATUS[0]}"

exit_c="$(gpu_temp)"

{
  echo "experiment=qwen38-r1-e117-gate-up-na4-rate-dip-and-the-serialised-n-split"
  echo "harness=local"
  echo "shapes=${shapes}"
  echo "widths=${widths}"
  echo "blocks=${blocks}"
  echo "swift_config=${config}"
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
