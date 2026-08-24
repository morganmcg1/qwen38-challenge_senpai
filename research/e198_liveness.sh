#!/usr/bin/env bash
# Research-only (e198-route1-row-amortized-kernel): find out whether the fused
# kernel actually runs inside the scored worker.
#
#   research/e198_liveness.sh [rows] [tokens]
#
# Stage 4 measured no effect. `attend` declines silently, so a null timing
# result alone cannot tell "the kernel ran and did not help" from "the kernel
# never ran". This arms the liveness counter, runs one short --local-iterate
# leg, and reports served calls by width and declines by reason.
#
# This leg is NOT a timing arm.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

rows="${1:-6,7,8,9}"
tokens="${2:-64}"
out="research/out/e198/liveness"
mkdir -p "${out}"

export MLXFAST_QWEN_MTP_HEAD_DIR="${MLXFAST_QWEN_MTP_HEAD_DIR:-${HOME}/.cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head-declared-run}"
export DARKBLOOM_QWEN_FUSED_SDPA_ROWS="${rows}"
export DARKBLOOM_E198_LIVENESS_OUT="${PWD}/${out}/counts.json"
export MLXFAST_LOCAL_COOL_GATE=0
export MLXFAST_QWEN_MTP_LOCAL_ITERATE_TOKENS="${tokens}"
export MLXFAST_SCORE_PATH="${PWD}/${out}/score.json"
rm -f "${DARKBLOOM_E198_LIVENESS_OUT}"

echo "e198_liveness: rows=${rows} tokens=${tokens}"
./benchmark-qwen-mtp.sh --local-iterate > "${out}/run.log" 2>&1
rc=$?
echo "exit: ${rc}"
echo "--- counts ---"
cat "${DARKBLOOM_E198_LIVENESS_OUT}" 2>/dev/null || echo "NO COUNTS FILE WRITTEN"
echo
exit "${rc}"
