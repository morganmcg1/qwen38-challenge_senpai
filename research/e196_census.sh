#!/usr/bin/env bash
# E196 Stage 0 -- per-call kernel identity census for the qL 6..9 split.
# No timing. Names the kernel that fires for SDPA call A and call B at every
# cell of m x kv, plus the R = 1..5 ladder the pricing model is fitted on.
set -uo pipefail

cd "$(dirname "$0")/.."

export MLX_E196_CENSUS=1
export MLX_E196_CENSUS_OUT="${MLX_E196_CENSUS_OUT:-research/e196-census.json}"
export MLX_E196_KV="${MLX_E196_KV:-512,768,1024,2048}"
export MLX_E196_M="${MLX_E196_M:-6,7,8,9}"

echo "E196 census: kv=${MLX_E196_KV} m=${MLX_E196_M} out=${MLX_E196_CENSUS_OUT}"
swift test --force-resolved-versions \
    --filter E196KernelIdentityCensusTests 2>&1 | tail -40
status=${PIPESTATUS[0]}
echo "swift test exit: ${status}"
exit "${status}"
