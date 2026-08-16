#!/usr/bin/env bash
# Research-only driver: one arm of the draft-head readout precision experiment.
#
#   research/run-draft-bits-arm.sh BITS TAG [TOKENS] [BASE_SHA]
#
# `MLX_QWEN_MTP_DRAFT_BITS` requantizes the compact draft readout for the DRAFT
# path only (Qwen35.swift makeCompactDraftHead); verify keeps the pinned 4-bit
# lm_head. Bits 4 is the control and must reproduce the shipped build.
#
# Delegates every gate to research/run-amdahl-measurement.sh, which delegates in
# turn to benchmark-qwen-mtp.sh: drift tripwire, orphan scan, run lock, 40C cool
# gate, and report seals all run unmodified.
set -euo pipefail

bits="${1:?usage: run-draft-bits-arm.sh BITS TAG [TOKENS] [BASE_SHA]}"
tag="${2:?usage: run-draft-bits-arm.sh BITS TAG [TOKENS] [BASE_SHA]}"
tokens="${3:-512}"
base_sha="${4:-}"

case "${bits}" in
  2 | 3 | 4) ;;
  *)
    echo "run-draft-bits-arm.sh: BITS must be 2, 3, or 4 (got ${bits})" >&2
    exit 1
    ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

# Not under .mlxfast-private/amdahl/<tag>: run-amdahl-measurement.sh clears
# that directory, which would delete these before the run starts.
out_dir="${repo_root}/.mlxfast-private/draft-bits/${tag}"
rm -rf -- "${out_dir}"
mkdir -p "${out_dir}"

export MLX_QWEN_MTP_DRAFT_BITS="${bits}"
# Provenance for which head the worker actually built. A file, not stderr:
# the worker only forwards stderr under MLX_QWEN_MTP_TRACE=1, which also writes
# per-round trace rows inside the timed window.
export MLX_QWEN_MTP_DRAFT_HEAD_REPORT="${out_dir}/draft-head.txt"
: >"${MLX_QWEN_MTP_DRAFT_HEAD_REPORT}"
export MLX_QWEN_MTP_TRACE="${MLX_QWEN_MTP_TRACE:-0}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-qwen38-r1-e6-draft-head-precision}"

# This host idles at a ~40.7C GPU floor (racked Mac: cpu_temp 36.8C and 0.22W
# package power while the GPU sensor reads 40.7C+, so the floor is inlet-air
# bound, not load bound), and the 40C gate therefore always hits its stalled
# abort. Acceptance, parity and the emitted token stream are thermally
# invariant, which is what this arm exists to measure, so the arms run ungated
# and every second they report is recorded as NOT gate-qualified. Arms stay
# comparable to each other -- same host, same hot state, back to back -- but not
# to any gated baseline. Authoritative timing comes from the interleaved
# QwenQMVCostCurveTests sweep instead.
export MLXFAST_LOCAL_COOL_GATE="${MLXFAST_LOCAL_COOL_GATE:-0}"

gpu_temp_now() {
  local macmon
  macmon="$(command -v macmon || true)"
  [[ -n "${macmon}" ]] || return 0
  "${macmon}" pipe -s1 2>/dev/null | jq -r '.temp.gpu_temp_avg // empty' 2>/dev/null
}

identity="${out_dir}/identity.txt"
{
  echo "run-draft-bits-arm: tag=${tag} bits=${bits} tokens=${tokens}"
  echo "run-draft-bits-arm: head=$(git rev-parse HEAD) dirty=$(git status --porcelain | wc -l | tr -d ' ')"
  echo "run-draft-bits-arm: base_sha=${base_sha}"
  echo "run-draft-bits-arm: host=$(sysctl -n machdep.cpu.brand_string) mem=$(sysctl -n hw.memsize)"
  echo "run-draft-bits-arm: started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "cool_gate=disabled_ambient_floor"
  echo "gpu_temp_c_before=$(gpu_temp_now)"
} >"${identity}"

# The MTP local-iterate report carries no memory field, so peak comes from
# `ru_maxrss`, which XNU folds across waited descendants as a max (in bytes).
{
  /usr/bin/time -l research/run-amdahl-measurement.sh \
    "${tag}" --local-iterate "${tokens}" "${base_sha}" \
    "draft-head readout precision: MLX_QWEN_MTP_DRAFT_BITS=${bits}"
} 2>&1 | tee "${out_dir}/rusage.txt"

# Collect the captured legs so the arm directory is self-contained: the next
# arm's run-amdahl-measurement.sh clears its own tag directory, and
# research/draft_bits_arms.py compares arms after every arm has run.
cp -R "${repo_root}/.mlxfast-private/amdahl/${tag}/reports" "${out_dir}/reports"
cp "${repo_root}/.mlxfast-private/amdahl/${tag}/amdahl.json" "${out_dir}/amdahl.json"

echo "gpu_temp_c_after=$(gpu_temp_now)" >>"${identity}"

echo "run-draft-bits-arm: arm summary"
cat "${MLX_QWEN_MTP_DRAFT_HEAD_REPORT}"
grep -i "maximum resident set size" "${out_dir}/rusage.txt" || true
