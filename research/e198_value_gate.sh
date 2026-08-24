#!/bin/bash
# E198 Stage 3 value gate: fused row-amortized SDPA vs the shipped two-call split,
# element-wise in bf16, with a firing positive control.
#   research/e198_value_gate.sh [widths] [kv lengths]
set -uo pipefail
cd "$(dirname "$0")/.."

WIDTHS="${1:-6,7,8,9}"
KV="${2:-512,1024}"
OUT="research/e198-exactness.json"

export MLXFAST_RUN_MLX_RUNTIME_TESTS=1
export MLX_E198_M="$WIDTHS"
export MLX_E198_KV="$KV"
export MLX_E198_EXACTNESS_OUT="$OUT"

echo "--- value gate widths=$WIDTHS kv=$KV"
swift test --force-resolved-versions --filter E198FusedExactnessTests
status=$?
echo "exit: $status"
echo "--- $OUT"
cat "$OUT" 2>/dev/null
exit "$status"
