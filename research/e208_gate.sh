#!/usr/bin/env bash
# E208 Stage-0 numerics falsification gate.
#
#   research/e208_gate.sh TAG
#
# Compares the compiled `(9, 5)` group partition against the shipped `(9, 3)`
# one at all seven fused decode cells, on real packed 4-bit weights and real
# bfloat16 activations, as actual floating-point values. The expectation is
# bit-exact; a positive control proves the comparison can fail.
#
# No timing. No thermal gate. No score.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tag="${1:?usage: research/e208_gate.sh TAG}"
out="research/out/${tag}"
rm -rf "${out}"
mkdir -p "${out}"

export MLXFAST_MACMON_BIN="${MLXFAST_MACMON_BIN:-/opt/homebrew/bin/macmon}"
gpu_temp() {
  "${MLXFAST_MACMON_BIN}" pipe -s1 2>/dev/null \
    | jq -r '.temp.gpu_temp_avg // empty' 2>/dev/null
}

{
  echo "tag=${tag}"
  echo "section=stage0-numerics-gate"
  echo "experiment=e208-qmv-9row-ipg5"
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
  echo "cells=${MLXFAST_E208_CELLS:-<all seven>}"
  echo "gpu_temp_entry_c=$(gpu_temp)"
  echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${out}/meta.txt"

MLXFAST_RUN_E208_GATE=1 \
MLXFAST_E208_GATE_OUT="${PWD}/${out}/gate.json" \
  swift test -c release --force-resolved-versions -Xswiftc -enable-testing \
    --filter E208Wide9QMVTests 2>&1 | tee "${out}/gate.log"
status=${PIPESTATUS[0]}

{
  echo "gpu_temp_exit_c=$(gpu_temp)"
  echo "exit=${status}"
  echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >> "${out}/meta.txt"

exit "${status}"
