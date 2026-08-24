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
# The runtime-worker sandbox in benchmark.sh's write_runtime_worker_sandbox_profile
# denies every file write except /dev/null, including TMPDIR, and the mtp-timed
# parent discards worker stderr. A probe inside the worker therefore has no way
# to report anything until the sandbox is lifted. This is the same local-only
# escape the MLX_QWEN_MTP_TRACE sink uses; it is refused when
# MLXFAST_OFFICIAL_BENCHMARK_RUN=1, so it cannot reach an official run.
export MLXFAST_NO_SANDBOX=1
export MLXFAST_QWEN_MTP_LOCAL_ITERATE_TOKENS="${tokens}"
export MLXFAST_SCORE_PATH="${PWD}/${out}/score.json"
rm -f "${DARKBLOOM_E198_LIVENESS_OUT}"
# Clear stale fallback files so this leg cannot read an earlier run's evidence.
rm -f "${HOME}/e198-liveness.json"

echo "e198_liveness: rows=${rows} tokens=${tokens}"
./benchmark-qwen-mtp.sh --local-iterate > "${out}/run.log" 2>&1
rc=$?
echo "exit: ${rc}"
# The probe reports every write on stderr, which the run log captures. This is
# the witness that the probe ran at all, independent of where it wrote.
echo "--- probe stderr witness ---"
grep -c "e198-liveness: wrote" "${out}/run.log" || echo "PROBE NEVER WROTE"
grep -m 2 "e198-liveness: FAILED" "${out}/run.log"
echo "--- counts (requested path) ---"
cat "${DARKBLOOM_E198_LIVENESS_OUT}" 2>/dev/null || echo "NO COUNTS FILE AT REQUESTED PATH"
echo
# The probe falls back to HOME when the requested path did not arrive, so a
# file here and not above isolates environment delivery.
echo "--- counts (HOME fallback) ---"
cat "${HOME}/e198-liveness.json" 2>/dev/null || echo "NO COUNTS FILE AT HOME FALLBACK"
echo
exit "${rc}"
