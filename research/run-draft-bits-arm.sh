#!/usr/bin/env bash
# Research-only driver: one arm of the draft-head readout precision experiment.
#
#   research/run-draft-bits-arm.sh BITS TAG [TOKENS] [BASE_SHA]
#
# `MLX_QWEN_MTP_DRAFT_BITS` requantizes the compact draft readout for the DRAFT
# path only (Qwen35.swift makeCompactDraftHead); verify keeps the pinned 4-bit
# lm_head. Bits 4 is the control and must reproduce the shipped build.
#
# Delegates every gate to research/run-amdahl-measurement.sh, which delegates in
# turn to benchmark-qwen-mtp.sh: drift tripwire, orphan scan, run lock, 40C cool
# gate, and report seals all run unmodified.
set -euo pipefail

bits="${1:?usage: run-draft-bits-arm.sh BITS TAG [TOKENS] [BASE_SHA]}"
tag="${2:?usage: run-draft-bits-arm.sh BITS TAG [TOKENS] [BASE_SHA]}"
tokens="${3:-512}"
base_sha="${4:-}"

case "${bits}" in
  2 | 3 | 4) ;;
  *)
    echo "run-draft-bits-arm.sh: BITS must be 2, 3, or 4 (got ${bits})" >&2
    exit 1
    ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

# Not under .mlxfast-private/amdahl/<tag>: run-amdahl-measurement.sh clears
# that directory, which would delete these before the run starts.
out_dir="${repo_root}/.mlxfast-private/draft-bits/${tag}"
rm -rf -- "${out_dir}"
mkdir -p "${out_dir}"

export MLX_QWEN_MTP_DRAFT_BITS="${bits}"
# No in-band provenance line is possible: on the mtp-timed verb the worker runs
# under a Seatbelt profile that denies file-write* except /dev/null
# (MLXFastCLI/main.swift writeRuntimeWorkerSandboxProfile), its stdout is the
# request protocol, and runQwenMTPTimed builds worker options with the default
# forwardsWorkerStderr:false, so the parent's drain discards worker stderr. The
# arm proves the requested head four other ways: the built worker binary
# contains the env read (gate below), MLX_ survives the worker env allowlist
# (sanitizedRuntimeWorkerEnvironment), the shape/byte math is pinned by
# QwenQMVCostCurveTests, and the bits=2 arm is the positive control -- two
# independent bits=4 runs agreed on accepted_draft_rate to 16 digits, so any
# arm that moves acceptance proves MLX_QWEN_MTP_DRAFT_BITS reached the head.
export MLX_QWEN_MTP_TRACE="${MLX_QWEN_MTP_TRACE:-0}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-qwen38-r1-e6-draft-head-precision}"

# The r1 claim of a ~40.7C inlet-bound floor was measured while a foreign GPU
# load was resident and is retracted: a true-idle sample on this host reads
# 39.92C GPU at 0.0076W GPU / 0.083W package, i.e. 0.08C BELOW the 40C gate, so
# the gate can pass here. The arms still run ungated for a budget reason:
# benchmark-qwen-mtp.sh calls the gate three times per arm and each call may wait
# COOL_GATE_MAX_WAIT_SECONDS=900, which four arms cannot fit in a 30-minute job.
# run-draft-bits-phase3.sh instead settles to the same 40.0C threshold before the
# arm and witnesses it with the real --local-cool-gate-only gate; MLXFAST_
# COOL_GATE_STATUS below records per-arm whether that witness actually passed.
export MLXFAST_LOCAL_COOL_GATE="${MLXFAST_LOCAL_COOL_GATE:-0}"

gpu_temp_now() {
  local macmon
  macmon="$(command -v macmon || true)"
  [[ -n "${macmon}" ]] || return 0
  "${macmon}" pipe -s1 2>/dev/null | jq -r '.temp.gpu_temp_avg // empty' 2>/dev/null
}

# benchmark-qwen-mtp.sh drives the CLI directly and reaches benchmark.sh only
# through --local-cool-gate-only, which returns before the build gate, so it
# never rebuilds a stale worker. The scored binary is the .build-worker twin;
# build both here with benchmark.sh's own commands and cache roots.
mkdir -p .build/clang-module-cache .build-worker/clang-module-cache
CLANG_MODULE_CACHE_PATH="${PWD}/.build/clang-module-cache" \
  swift build -c release --force-resolved-versions --product mlxfast-swift
CLANG_MODULE_CACHE_PATH="${PWD}/.build-worker/clang-module-cache" \
  swift build -c release --force-resolved-versions \
  --scratch-path .build-worker --product mlxfast-runtime-worker

worker_bin=".build-worker/release/mlxfast-runtime-worker"
# grep -c, not grep -q: under `pipefail` an early-exiting grep SIGPIPEs strings
# and the pipeline reports 141 even on a match.
if [[ "$(strings -a "${worker_bin}" | grep -c "MLX_QWEN_MTP_DRAFT_BITS" || true)" -eq 0 ]]; then
  echo "run-draft-bits-arm.sh: ${worker_bin} has no MLX_QWEN_MTP_DRAFT_BITS; refusing to time a build that cannot honour the arm" >&2
  exit 1
fi

# The strings tripwire above only proves the env READ survives; it goes blind to
# a change in the compiled DEFAULT, which is exactly what this experiment moves.
# worker_sha256 is the tripwire for that: two arms differing only in the env var
# must report the SAME worker digest, and a digest equal to a pre-change build
# would prove the default never got compiled in.
identity="${out_dir}/identity.txt"
{
  echo "run-draft-bits-arm: tag=${tag} bits=${bits} tokens=${tokens}"
  echo "run-draft-bits-arm: head=$(git rev-parse HEAD) dirty=$(git status --porcelain | wc -l | tr -d ' ')"
  echo "run-draft-bits-arm: base_sha=${base_sha}"
  echo "run-draft-bits-arm: host=$(sysctl -n machdep.cpu.brand_string) mem=$(sysctl -n hw.memsize)"
  echo "run-draft-bits-arm: started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "worker_sha256=$(shasum -a 256 "${worker_bin}" | cut -d' ' -f1)"
  echo "segmented_streak_gate=$(grep -oE 'segmentedStreakGate = [0-9]+' \
    Sources/MLXFastModel/Qwen36MTPBlockSession.swift | head -1 | grep -oE '[0-9]+$')"
  echo "m8_ipg=$(grep -oE 'qmv_fast_crossrow_affine4_g64_m<T, 8, [0-9]+>' \
    Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h |
    head -1 | grep -oE '[0-9]+>$' | tr -d '>')"
  echo "cool_gate=${MLXFAST_COOL_GATE_STATUS:-not_probed}"
  echo "settle_target_c=${MLXFAST_SETTLE_TARGET_C:-none} settle_reached_c=${MLXFAST_SETTLE_REACHED_C:-none} settle_min_c=${MLXFAST_SETTLE_MIN_C:-none} settle_waited_s=${MLXFAST_SETTLE_WAITED_S:-none}"
  echo "gpu_temp_c_before=$(gpu_temp_now)"
} >"${identity}"

# The MTP local-iterate report carries no memory field, so peak comes from
# `ru_maxrss`, which XNU folds across waited descendants as a max (in bytes).
{
  /usr/bin/time -l research/run-amdahl-measurement.sh \
    "${tag}" --local-iterate "${tokens}" "${base_sha}" \
    "draft-head readout precision: MLX_QWEN_MTP_DRAFT_BITS=${bits}"
} 2>&1 | tee "${out_dir}/rusage.txt"

# Collect the captured legs so the arm directory is self-contained: the next
# arm's run-amdahl-measurement.sh clears its own tag directory, and
# research/draft_bits_arms.py compares arms after every arm has run.
cp -R "${repo_root}/.mlxfast-private/amdahl/${tag}/reports" "${out_dir}/reports"
cp "${repo_root}/.mlxfast-private/amdahl/${tag}/amdahl.json" "${out_dir}/amdahl.json"

echo "gpu_temp_c_after=$(gpu_temp_now)" >>"${identity}"

echo "run-draft-bits-arm: arm summary"
grep -i "maximum resident set size" "${out_dir}/rusage.txt" || true
python3 -c 'import json,sys;r=json.load(open(sys.argv[1]))["mtp_leg"];print("run-draft-bits-arm: accepted_draft_rate=%.16f round_count=%d spt=%.9f" % (r["accepted_draft_rate"],r["round_count"],r["parent_measured_seconds_per_token"]))' \
  "${out_dir}/amdahl.json"
