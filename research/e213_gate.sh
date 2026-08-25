#!/usr/bin/env bash
# E213 Stage-0 numerics falsification gate.
#
#   research/e213_gate.sh TAG
#
# Compares each E213 arm against the shipped plan, at every width that arm
# moves, at all seven fused decode cells and at one n = 4104 edge cell, on real
# packed 4-bit weights and real bfloat16 activations, as actual floating-point
# values. The expectation is bit-exact; a positive control under both geometries
# proves the comparison can fail.
#
#   g1     (7,4,rows4)->(7,7,rows2), (8,4,rows4)->(8,8,rows1),
#          (9,5,rows4)->(9,9,rows1): G = 1 at the three widths, held under the
#          128-register boundary by a lower rows_per_simd.
#   probe  (9,5,rows4)->(9,5,rows2): the attribution control at fixed G = 2.
#
# It also compares the shipped rows = 4 path against the pre-E213 header at
# every width, so the ROWS parameterization cannot have moved untouched code,
# and it asserts that the launched y extent covers every output row once.
#
# No timing. No thermal gate. No score.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tag="${1:?usage: research/e213_gate.sh TAG}"
out="research/out/${tag}"
rm -rf "${out}"
mkdir -p "${out}"

export MLXFAST_MACMON_BIN="${MLXFAST_MACMON_BIN:-${HOME}/bin/macmon}"
gpu_temp() {
  "${MLXFAST_MACMON_BIN}" pipe -s1 2>/dev/null \
    | jq -r '.temp.gpu_temp_avg // empty' 2>/dev/null
}

{
  echo "tag=${tag}"
  echo "section=stage0-numerics-gate"
  echo "experiment=e213-rows-per-simd-g1"
  echo "harness=local"
  echo "cool_gate_passed_real_gate=false"
  echo "gate_qualified_for_timing=false"
  echo "official_or_ranked_score=false"
  echo "base_sha=$(git rev-parse HEAD)"
  echo "dirty_candidate_paths=$(
    git status --porcelain -- Sources Vendor Package.swift | wc -l | tr -d ' ')"
  echo "host=$(hostname)"
  echo "chip=$(sysctl -n machdep.cpu.brand_string 2>/dev/null)"
  echo "gpu_cores=$(ioreg -l 2>/dev/null \
    | LC_ALL=C sed -n 's/.*"gpu-core-count" = \([0-9][0-9]*\).*/\1/p' | head -1)"
  echo "memory_bytes=$(sysctl -n hw.memsize)"
  echo "os=$(sw_vers -productVersion)"
  echo "swift=$(swift --version 2>&1 | head -1)"
  echo "metallib_source_fingerprint=$(tools/build-mlx-metallib.sh --print-fingerprint)"
  echo "cells=${MLXFAST_E213_CELLS:-<all seven>}"
  echo "gpu_temp_entry_c=$(gpu_temp)"
  echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${out}/meta.txt"

# SwiftPM copies `mlx.metallib` beside the debug test executable but not beside
# the release one, and MLX resolves the default metallib from the executable's
# own directory. Build first, place the metallib, then run without rebuilding.
swift build -c release --force-resolved-versions -Xswiftc -enable-testing \
  --build-tests 2>&1 | tee "${out}/build.log"
status=${PIPESTATUS[0]}

bundle=".build/arm64-apple-macosx/release/mlxfast-challenge-devPackageTests.xctest/Contents/MacOS"
if [[ "${status}" -eq 0 ]]; then
  cp .build/arm64-apple-macosx/release/mlx.metallib "${bundle}/" || status=1
fi

if [[ "${status}" -eq 0 ]]; then
  MLXFAST_RUN_E213_GATE=1 \
  MLXFAST_E213_GATE_OUT="${PWD}/${out}/gate.json" \
  MLXFAST_E213_PLAIN_OUT="${PWD}/${out}/plain-vs-table.json" \
    swift test -c release --force-resolved-versions -Xswiftc -enable-testing \
      --skip-build --filter E213RowsPerSimdQMVTests 2>&1 | tee "${out}/gate.log"
  status=${PIPESTATUS[0]}
fi

{
  echo "gpu_temp_exit_c=$(gpu_temp)"
  echo "exit=${status}"
  echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >> "${out}/meta.txt"

exit "${status}"
