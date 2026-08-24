#!/usr/bin/env bash
# E198 Stage 1 -- static register / occupancy curve for the fused
# row-amortized SDPA kernel, one width per process.
#
# Bit-exactness requires the full 1024-thread threadgroup, so a width whose
# register pressure drops the pipeline's maxTotalThreadsPerThreadgroup below
# 1024 makes MLX throw at dispatch and can end the process. Each width
# therefore runs alone, and the ledger writes every pipeline record to disk at
# CREATION time, before any dispatch can fail.
#
# Usage: research/e198_register_gate.sh [widths] [kv]
set -uo pipefail

cd "$(dirname "$0")/.."

WIDTHS="${1:-6,7,8,9}"
KV="${2:-512}"
OUT="research/e198-pipeline.jsonl"

: > "${OUT}"

echo "E198 Stage 1 register curve: widths=${WIDTHS} kv=${KV}"
for m in ${WIDTHS//,/ }; do
    echo "--- m=${m}"
    MLX_E198_REGISTER=1 MLX_E198_M="${m}" MLX_E198_KV="${KV}" \
        MLX_E198_PIPELINE_OUT="${OUT}" \
        swift test --force-resolved-versions \
            --filter E198RegisterGateTests 2>&1 | tail -12
    echo "exit(m=${m}): ${PIPESTATUS[0]}"
done

echo "--- pipeline records"
cat "${OUT}"
