#!/bin/bash
# E198 Stage 2 pricing: fused dispatch against the shipped pair and the
# one-row floor, ABBA-counterbalanced, ungated, with per-arm GPU temperature.
#   research/e198_pricing.sh [widths] [kv lengths] [blocks] [reps] [out]
set -uo pipefail
cd "$(dirname "$0")/.."

WIDTHS="${1:-6,8}"
KV="${2:-512,1014}"
BLOCKS="${3:-6}"
REPS="${4:-20}"
OUT="${5:-research/e198-timing.json}"

export MLX_E198_TIMING=1
export MLX_E198_M="$WIDTHS"
export MLX_E198_KV="$KV"
export MLX_E198_BLOCKS="$BLOCKS"
export MLX_E198_REPS="$REPS"
export MLX_E198_TIMING_OUT="$OUT"

echo "--- pricing widths=$WIDTHS kv=$KV blocks=$BLOCKS reps=$REPS"
swift test --force-resolved-versions --filter E198DispatchPricingTests
status=$?
echo "exit: $status"
exit "$status"
