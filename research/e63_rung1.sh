#!/usr/bin/env bash
# E63 rung 1: replicate the QMV weight-stream bandwidth ladder on a SECOND host.
#
# `Tests/MLXFastTests/QwenQMVCostCurveTests.swift` has never been run. It emits
# the roofline, the per-shape width sweep, the dispatch boundary probes, the
# fast-path probes and the crossrow gate in one command, with no resident model.
#
# askeladd's E61 ladder is the sole source of the six bandwidth numbers the E63
# brief rests on, and it has one host behind it. This gives a second host and a
# roofline measured here instead of an inherited 227.9 GB/s.
#
#   research/e63_rung1.sh
set -uo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo}"

out="${E63_OUT:-research/e63-artifacts/e63-cost-curve.json}"
log="${E63_LOG:-research/e63-artifacts/e63-rung1.log}"
mkdir -p "$(dirname "${out}")"

macmon_bin="${MLXFAST_MACMON_BIN:-${HOME}/bin/macmon}"
sample_thermal() {
  [[ -x "${macmon_bin}" ]] || { echo "unavailable"; return 0; }
  "${macmon_bin}" pipe -s1 2>/dev/null \
    | jq -r '"gpu_temp=\(.temp.gpu_temp_avg // "?")C cpu_temp=\(.temp.cpu_temp_avg // "?")C gpu_power=\(.gpu_power // "?")W all_power=\(.all_power // "?")W"' \
      2>/dev/null || echo "unreadable"
}

{
  echo "e63_rung1 head=$(git rev-parse HEAD) dirty=$(git status --porcelain | wc -l | tr -d ' ')"
  echo "e63_rung1 host=$(hostname -s) chip=$(sysctl -n machdep.cpu.brand_string)"
  echo "e63_rung1 entry_thermal $(sample_thermal)"
  echo "e63_rung1 started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} | tee -a "${log}"

MLXFAST_RUN_QMV_COST_CURVE=1 \
MLXFAST_QMV_COST_CURVE_OUT="${out}" \
MLXFAST_QMV_COST_CURVE_REPS="${E63_REPS:-15}" \
MLXFAST_QMV_COST_CURVE_INNER="${E63_INNER:-10}" \
  swift test --force-resolved-versions --filter QwenQMVCostCurveTests 2>&1 | tee -a "${log}"
status="${PIPESTATUS[0]}"

{
  echo "e63_rung1 exit_thermal $(sample_thermal)"
  echo "e63_rung1 finished=$(date -u +%Y-%m-%dT%H:%M:%SZ) status=${status}"
} | tee -a "${log}"
exit "${status}"
