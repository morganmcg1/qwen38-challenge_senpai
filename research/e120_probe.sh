#!/usr/bin/env bash
# E120 -- run one candidate-owned QMV dispatch probe session and record the
# identity tuple around it.
#
#   usage: research/e120_probe.sh TAG [shapes] [widths] [blocks] [filter]
#
#   TAG      output directory name under research/out/
#   shapes   comma list of preset names or `name:N:K` triples,
#            default `mlp.gate_up`
#   widths   comma list of row counts M, default `3,4,5,6,7,8,9`
#   blocks   palindromic blocks per cell, default 6
#   filter   swift-testing filter, default `E120CustomQMVProbeTests`
#
# Every arm is timed forward and then in reverse inside one block, so monotone
# thermal drift cancels to first order. Temperature is sampled only at block
# boundaries and every sample is followed by a discarded fixed-duration ramp
# burst. No thermal gate, no score: `cool_gate_passed_real_gate=false` and
# `gate_qualified_for_timing=false` are preserved verbatim.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tag="${1:?usage: e120_probe.sh TAG [shapes] [widths] [blocks] [filter]}"
shapes="${2:-mlp.gate_up}"
widths="${3:-3,4,5,6,7,8,9}"
blocks="${4:-6}"
filter="${5:-E120CustomQMVProbeTests}"
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

config="${MLXFAST_E120_CONFIG:-release}"

MLXFAST_RUN_E120_PROBE="${MLXFAST_RUN_E120_PROBE:-1}" \
MLXFAST_RUN_MLX_RUNTIME_TESTS="${MLXFAST_RUN_MLX_RUNTIME_TESTS:-1}" \
MLXFAST_E120_SHAPES="${shapes}" \
MLXFAST_E120_WIDTHS="${widths}" \
MLXFAST_E120_BLOCKS="${blocks}" \
MLXFAST_E120_OUT="${PWD}/${out_dir}/cells.json" \
swift test -c "${config}" --force-resolved-versions \
  --filter "${filter}" 2>&1 | tee "${out_dir}/probe.log"
status="${PIPESTATUS[0]}"

exit_c="$(gpu_temp)"

{
  echo "experiment=qwen38-r1-e120-own-the-qmv-dispatch"
  echo "harness=local"
  echo "shapes=${shapes}"
  echo "widths=${widths}"
  echo "blocks=${blocks}"
  echo "filter=${filter}"
  echo "target_us=${MLXFAST_E120_TARGET_US:-100000}"
  echo "ramp_s=${MLXFAST_E120_RAMP_S:-0.30}"
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
