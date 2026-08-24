#!/usr/bin/env bash
# E196 Stage 1 -- chain-slope pricing session for the qL 6..9 split SDPA calls.
#
# One ABBA-counterbalanced session. Every arm is a direct dispatch measurement,
# not a decode leg, so RULE 388's leg-level floor does not bind it.
# harness=local, no thermal gate: cool_gate_passed_real_gate=false and
# gate_qualified_for_timing=false are written verbatim into the report.
#
# Usage: research/e196_session.sh [blocks] [reps] [warmup] [out]
set -uo pipefail

cd "$(dirname "$0")/.."

export MLX_E196_TIMING=1
export MLX_E196_BLOCKS="${1:-6}"
export MLX_E196_REPS="${2:-20}"
export MLX_E196_WARMUP="${3:-6}"
export MLX_E196_TIMING_OUT="${4:-research/e196-timing.json}"
export MLX_E196_KV="${MLX_E196_KV:-512,768,1024,2048}"
export MLX_E196_M="${MLX_E196_M:-6,7,8,9}"
export MLX_E196_CHAINS="${MLX_E196_CHAINS:-1,2,4,8,16}"
export MLX_E196_ARM_CHAINS="${MLX_E196_ARM_CHAINS:-1,8}"

echo "E196 timing session"
echo "  kv=${MLX_E196_KV} m=${MLX_E196_M}"
echo "  chains=${MLX_E196_CHAINS} arm_chains=${MLX_E196_ARM_CHAINS}"
echo "  blocks=${MLX_E196_BLOCKS} reps=${MLX_E196_REPS} warmup=${MLX_E196_WARMUP}"
echo "  out=${MLX_E196_TIMING_OUT}"

swift test --force-resolved-versions \
    --filter E196ChainSlopePricingTests 2>&1 | tail -25
status=${PIPESTATUS[0]}
echo "swift test exit: ${status}"
exit "${status}"
