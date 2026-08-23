#!/usr/bin/env bash
# E153 R2 exactness gate: the merged wide-decode SDPA kernel against the
# shipped two-call split, plus the Rule 101 positive control and the 2-pass
# band refusal. Stop the experiment on any nonzero deviation.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

OUT=research/out
mkdir -p "$OUT"

export MLXFAST_RUN_MLX_RUNTIME_TESTS=1
export MLXFAST_E153_EXACTNESS_OUT="$PWD/$OUT/e153-r2-exactness.json"

swift test --force-resolved-versions --filter E153MergedSdpaKernelTests \
    2>&1 | tee "$OUT/e153-r2-exactness.log"
code=${PIPESTATUS[0]}
echo "swift test exit $code"
exit "$code"
