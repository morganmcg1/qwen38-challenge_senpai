#!/usr/bin/env bash
# E98 rung 1a -- run the metadata byte ladder and record the identity tuple
# around it.
#
#   usage: research/e98_bytes_probe.sh TAG
#
# The probe is a within-session counterbalanced comparison, so it runs under no
# thermal gate and reports no score. Entry and exit GPU temperature are
# recorded because a drifting session only supports a relative claim.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tag="${1:?usage: e98_bytes_probe.sh TAG}"
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

MLXFAST_RUN_E98_BYTES=1 \
MLXFAST_E98_BYTES_OUT="${PWD}/${out_dir}/bytes.json" \
swift test --force-resolved-versions \
  --filter E98MetadataByteProbeTests 2>&1 | tee "${out_dir}/probe.log"
status="${PIPESTATUS[0]}"

exit_c="$(gpu_temp)"

{
  echo "experiment=e98-transform-owned-weight-metadata-index"
  echo "rung=1a-byte-ladder"
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
