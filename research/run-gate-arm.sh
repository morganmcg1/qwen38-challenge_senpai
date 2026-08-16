#!/usr/bin/env bash
# Research-only driver for one traced local Qwen-MTP gate arm.
#
#   research/run-gate-arm.sh TAG EXPECT_CAP EXPECT_GATE [TOKENS] [MODE]
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

tag="${1:?usage: run-gate-arm.sh TAG EXPECT_CAP EXPECT_GATE [TOKENS] [MODE]}"
expect_cap="${2:?usage: run-gate-arm.sh TAG EXPECT_CAP EXPECT_GATE [TOKENS] [MODE]}"
expect_gate="${3:?usage: run-gate-arm.sh TAG EXPECT_CAP EXPECT_GATE [TOKENS] [MODE]}"
tokens="${4:-512}"
mode="${5:---local-iterate}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

session_src="Sources/MLXFastModel/Qwen36MTPBlockSession.swift"

# Two conversations of this role share one checkout, and a concurrent write to
# session_src between the constant read and the compile silently retags an arm
# (observed 2026-08-16T18:10Z: cap 8 -> 7 landed mid-build). Assert the two
# constants that define an arm rather than trusting the pre-build read.
log() { echo "run-gate-arm: $*" | tee -a "${raw_log}"; }

assert_arm_config() {
  local when="$1" cap gate
  cap="$(sed -n 's/.*segmentedVerifyDepthCap = \([0-9]*\).*/\1/p' "${session_src}")"
  gate="$(sed -n 's/.*segmentedStreakGate = \([0-9]*\).*/\1/p' "${session_src}")"
  log "${when} cap=${cap} gate=${gate} (expected cap=${expect_cap} gate=${expect_gate})"
  [[ "${cap}" == "${expect_cap}" && "${gate}" == "${expect_gate}" ]]
}

apply_arm_config() {
  sed -i '' \
    -e "s/segmentedVerifyDepthCap = [0-9]*/segmentedVerifyDepthCap = ${expect_cap}/" \
    -e "s/segmentedStreakGate = [0-9]*/segmentedStreakGate = ${expect_gate}/" \
    "${session_src}"
}

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

build_products() {
  mkdir -p .build/clang-module-cache .build-worker/clang-module-cache
  CLANG_MODULE_CACHE_PATH="${PWD}/.build/clang-module-cache" \
    swift build -c release --force-resolved-versions --product mlxfast-swift 2>&1 | tee -a "${raw_log}"
  local build_a=${PIPESTATUS[0]}
  CLANG_MODULE_CACHE_PATH="${PWD}/.build-worker/clang-module-cache" \
    swift build -c release --force-resolved-versions --scratch-path .build-worker --product mlxfast-runtime-worker 2>&1 | tee -a "${raw_log}"
  local build_b=${PIPESTATUS[0]}
  if (( build_a != 0 || build_b != 0 )); then
    echo "run-gate-arm: build failed (cli=${build_a} worker=${build_b})" | tee -a "${raw_log}"
    return 1
  fi
}

if ! assert_arm_config pre-build; then
  log "pre-build mismatch, applying expected constants"
  apply_arm_config
  assert_arm_config pre-build-reapplied || { log "aborting, cannot set arm constants"; exit 2; }
fi

build_products || exit 1

# A concurrent write can land after the pre-build read and still reach the
# compiler, so the binary is only trusted once the post-build source matches.
if ! assert_arm_config post-build; then
  log "post-build mismatch, reapplying and rebuilding once"
  apply_arm_config
  build_products || exit 1
  assert_arm_config post-rebuild || { log "aborting, arm constants unstable under concurrent writes"; exit 2; }
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

grep -E 'run-gate-arm:|mtp-trace|mtp-row|generating the MTP reference rows|measuring the TRUE serial control|measuring native-MTP decode' \
  "${raw_log}" > "${trace_log}"

echo "run-gate-arm: status=${status} trace_lines=$(wc -l < "${trace_log}" | tr -d ' ')"
exit "${status}"
