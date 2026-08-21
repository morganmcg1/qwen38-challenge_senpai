#!/usr/bin/env bash
# E115 rung 1 -- run one isolated concurrent-dispatch probe session and record
# the identity tuple around it.
#
#   usage: research/e115_probe.sh TAG [shapes] [widths] [blocks]
#
#   TAG      output directory name under research/out/
#   shapes   comma list, default `mlp.gate_up,lm_head,gdn.in_proj,fa.qkv`
#   widths   comma list, default `2,3,4,5`
#   blocks   palindromic blocks per cell, default 6
#
# Every arm inside a cell is timed forward and then in reverse inside the same
# block, so monotone thermal drift cancels to first order. The first block is
# discarded by the analysis. No thermal gate, no score: entry and exit GPU
# temperature are recorded per block by the probe itself, and
# `cool_gate_passed_real_gate=false` plus `gate_qualified_for_timing=false` are
# preserved verbatim.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tag="${1:?usage: e115_probe.sh TAG [shapes] [widths] [blocks]}"
shapes="${2:-mlp.gate_up,lm_head,gdn.in_proj,fa.qkv}"
widths="${3:-2,3,4,5}"
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

# A debug test build leaves about 124 us of host cost on every blocking eval,
# which is a quarter of one gate_up dispatch. Release cuts it, and the arm
# structure control cell measures whatever is left.
config="${MLXFAST_E115_CONFIG:-release}"

MLXFAST_RUN_E115_PROBE=1 \
MLXFAST_E115_SHAPES="${shapes}" \
MLXFAST_E115_WIDTHS="${widths}" \
MLXFAST_E115_BLOCKS="${blocks}" \
MLXFAST_E115_OUT="${PWD}/${out_dir}/cells.json" \
swift test -c "${config}" --force-resolved-versions \
  --filter E115ConcurrentDispatchProbeTests 2>&1 | tee "${out_dir}/probe.log"
status="${PIPESTATUS[0]}"

exit_c="$(gpu_temp)"

{
  echo "experiment=qwen38-r1-e115-concurrency-discriminator-and-the-n-split"
  echo "rung=1"
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
