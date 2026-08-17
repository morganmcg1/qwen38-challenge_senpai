#!/usr/bin/env bash
# E9 r3 / A1 runtime half: prove which draft-readout path the scored worker
# takes on the promoted-frontier base.
#
# This is NOT a timed arm. It runs the shortest legal decode window with the
# stderr trace enabled, so every number it prints is a path counter, never a
# speed. Two env vars are both required:
#
#   MLX_QWEN_MTP_TRACE=1        flips forwardsWorkerStderr on the mtp-timed
#                               verb (Sources/MLXFastCLI/main.swift:1805),
#                               without which worker stderr is discarded
#   MLX_QWEN_MTP_TRACE_DRAFT=1  arms the Qwen35DraftTrace probes themselves
#
# MLX_QWEN_MTP_TRACE=1 also turns on the session's per-position row dump, so
# the interesting lines are extracted by grepping for E9R3-TRACE.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

tokens="${1:-16}"
out_dir=".mlxfast-private/e9r3-liveness"
mkdir -p "${out_dir}"

# Same build commands and scratch paths benchmark.sh uses. The scored binary is
# the .build-worker twin; a plain `swift build` refreshes only .build/release
# and leaves a stale worker behind, which is the defect this run must not repeat.
mkdir -p .build/clang-module-cache .build-worker/clang-module-cache
CLANG_MODULE_CACHE_PATH="${PWD}/.build/clang-module-cache" \
  swift build -c release --force-resolved-versions --product mlxfast-swift
CLANG_MODULE_CACHE_PATH="${PWD}/.build-worker/clang-module-cache" \
  swift build -c release --force-resolved-versions \
  --scratch-path .build-worker --product mlxfast-runtime-worker

worker_bin=".build-worker/release/mlxfast-runtime-worker"
# grep -c, not grep -q: under pipefail an early-exiting grep SIGPIPEs strings.
if [[ "$(strings -a "${worker_bin}" | grep -c "E9R3-TRACE" || true)" -eq 0 ]]; then
  echo "run-e9r3-liveness: ${worker_bin} carries no E9R3-TRACE probe; refusing to report a silent run as evidence" >&2
  exit 1
fi

{
  echo "run-e9r3-liveness: head=$(git rev-parse HEAD) dirty=$(git status --porcelain | wc -l | tr -d ' ')"
  echo "run-e9r3-liveness: tokens=${tokens} started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "worker_sha256=$(shasum -a 256 "${worker_bin}" | cut -d' ' -f1)"
  echo "cli_sha256=$(shasum -a 256 .build/release/mlxfast-swift | cut -d' ' -f1)"
  echo "untimed=true reason=trace_enabled_stderr_forwarding"
} | tee "${out_dir}/identity.txt"

export MLX_QWEN_MTP_TRACE=1
export MLX_QWEN_MTP_TRACE_DRAFT=1
export MLXFAST_QWEN_MTP_LOCAL_ITERATE_TOKENS="${tokens}"
export MLXFAST_LOCAL_COOL_GATE=0
export WANDB_MODE=disabled

set +e
./benchmark-qwen-mtp.sh --local-iterate >"${out_dir}/run.log" 2>&1
bench_rc=$?
set -e
echo "run-e9r3-liveness: benchmark exit=${bench_rc}"

echo "=== E9R3-TRACE lines ==="
grep -a "E9R3-TRACE" "${out_dir}/run.log" | sort -u | tee "${out_dir}/trace.txt"
echo "=== end trace ($(grep -ac 'E9R3-TRACE' "${out_dir}/run.log" || true) raw lines) ==="
exit "${bench_rc}"
