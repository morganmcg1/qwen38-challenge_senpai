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

out_dir=".mlxfast-private/amdahl/${tag}"
mkdir -p "${out_dir}"

export MLX_QWEN_MTP_DRAFT_BITS="${bits}"
# Provenance for which head the worker actually built. A file, not stderr:
# the worker only forwards stderr under MLX_QWEN_MTP_TRACE=1, which also writes
# per-round trace rows inside the timed window.
export MLX_QWEN_MTP_DRAFT_HEAD_REPORT="${repo_root}/${out_dir}/draft-head.txt"
: >"${MLX_QWEN_MTP_DRAFT_HEAD_REPORT}"
export MLX_QWEN_MTP_TRACE="${MLX_QWEN_MTP_TRACE:-0}"

# The MTP local-iterate report carries no memory field, so peak comes from
# `ru_maxrss`, which XNU folds across waited descendants as a max (in bytes).
{
  /usr/bin/time -l research/run-amdahl-measurement.sh \
    "${tag}" --local-iterate "${tokens}" "${base_sha}" \
    "draft-head readout precision: MLX_QWEN_MTP_DRAFT_BITS=${bits}"
} 2>&1 | tee "${out_dir}/rusage.txt"

echo "run-draft-bits-arm: arm summary"
cat "${MLX_QWEN_MTP_DRAFT_HEAD_REPORT}"
grep -i "maximum resident set size" "${out_dir}/rusage.txt" || true
