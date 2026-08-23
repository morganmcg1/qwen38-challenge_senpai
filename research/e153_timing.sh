#!/usr/bin/env bash
# E153 R2 mechanism attribution: measured split vs merged per full-attention
# layer, against the E149 C1 fit. Directional, ungated, within-session.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

OUT=research/out
mkdir -p "$OUT"

export MLXFAST_RUN_E153_PROBE=1
export MLXFAST_E153_TIMING_OUT="$PWD/$OUT/e153-r2-timing.json"

swift test --force-resolved-versions \
    --filter 'E153MergedSdpaKernelTests/recoveredCost' \
    2>&1 | tee "$OUT/e153-r2-timing.log"
code=${PIPESTATUS[0]}
echo "swift test exit $code"
exit "$code"
