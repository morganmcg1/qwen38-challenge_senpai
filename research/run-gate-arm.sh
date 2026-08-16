#!/usr/bin/env bash
# Research-only driver for one traced local Qwen-MTP gate arm.
#
#   research/run-gate-arm.sh TAG [TOKENS] [MODE]
#
# Rebuilds both products exactly as benchmark.sh would (the wrapper itself does
# not build), runs one gated ./benchmark-qwen-mtp.sh leg with the stderr round
# trace enabled, and leaves three artifacts behind:
#
#   research/trace-TAG.log          filtered worker trace + phase markers
#   research/score-TAG.json         wrapper score payload
#   research/capture-TAG/           per-verb CLI reports the wrapper would delete
#
# Every gate the wrapper owns (drift tripwire, orphan scan, run lock, 40C cool
# gate, report seals) still runs unmodified; MLXFAST_SWIFT_BIN only points at a
# passthrough that tees each stdout report.
set -uo pipefail

tag="${1:?usage: run-gate-arm.sh TAG [TOKENS] [MODE]}"
tokens="${2:-512}"
mode="${3:---local-iterate}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

capture_dir="${repo_root}/research/capture-${tag}"
raw_log="${repo_root}/research/raw-${tag}.log"
trace_log="${repo_root}/research/trace-${tag}.log"
rm -rf -- "${capture_dir}"
mkdir -p "${capture_dir}"

{
  echo "run-gate-arm: tag=${tag} mode=${mode} tokens=${tokens}"
  echo "run-gate-arm: head=$(git rev-parse HEAD) dirty=$(git status --porcelain | wc -l | tr -d ' ')"
  echo "run-gate-arm: $(grep -hoE 'segmented(VerifyDepthCap|StreakGate) = [0-9]+|sdpaWidthWallDepthCap = [0-9]+' Sources/MLXFastModel/Qwen36MTPBlockSession.swift | tr '\n' ' ')"
  date -u '+run-gate-arm: started_utc=%Y-%m-%dT%H:%M:%SZ'
} 2>&1 | tee "${raw_log}"

mkdir -p .build/clang-module-cache .build-worker/clang-module-cache
CLANG_MODULE_CACHE_PATH="${PWD}/.build/clang-module-cache" \
  swift build -c release --force-resolved-versions --product mlxfast-swift 2>&1 | tee -a "${raw_log}"
build_a=${PIPESTATUS[0]}
CLANG_MODULE_CACHE_PATH="${PWD}/.build-worker/clang-module-cache" \
  swift build -c release --force-resolved-versions --scratch-path .build-worker --product mlxfast-runtime-worker 2>&1 | tee -a "${raw_log}"
build_b=${PIPESTATUS[0]}
if (( build_a != 0 || build_b != 0 )); then
  echo "run-gate-arm: build failed (cli=${build_a} worker=${build_b})" | tee -a "${raw_log}"
  exit 1
fi

export MLX_QWEN_MTP_TRACE=1
export MLXFAST_CAPTURE_REAL_BIN="${repo_root}/.build/release/mlxfast-swift"
export MLXFAST_CAPTURE_DIR="${capture_dir}"
export MLXFAST_SWIFT_BIN="${repo_root}/research/capture-cli.sh"
export MLXFAST_SCORE_PATH="${repo_root}/research/score-${tag}.json"
export MLXFAST_QWEN_MTP_LOCAL_ITERATE_TOKENS="${tokens}"
export MLXFAST_QWEN_MTP_LOCAL_SUBMIT_TOKENS="${tokens}"

./benchmark-qwen-mtp.sh "${mode}" 2>&1 | tee -a "${raw_log}"
status=${PIPESTATUS[0]}

date -u '+run-gate-arm: benchmark_finished_utc=%Y-%m-%dT%H:%M:%SZ' | tee -a "${raw_log}"

grep -E 'mtp-trace|mtp-row|generating the MTP reference rows|measuring the TRUE serial control|measuring native-MTP decode' \
  "${raw_log}" > "${trace_log}"

echo "run-gate-arm: status=${status} trace_lines=$(wc -l < "${trace_log}" | tr -d ' ')"
exit "${status}"
