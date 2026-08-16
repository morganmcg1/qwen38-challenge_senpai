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

bits="${1:?usage: run-draft-bits-arm.sh BITS TAG [TOKENS] [BASE_SHA] [MODE]}"
tag="${2:?usage: run-draft-bits-arm.sh BITS TAG [TOKENS] [BASE_SHA] [MODE]}"
tokens="${3:-512}"
base_sha="${4:-}"
# --local-submit swaps the drift tripwire onto the 1024-step public golden,
# which is the only local way to check fidelity past key_len 1024. Decode
# tokens stay capped at 512 by benchmark-qwen-mtp.sh either way.
mode="${5:---local-iterate}"

case "${bits}" in
  2 | 3 | 4 | default) ;;
  *)
    echo "run-draft-bits-arm.sh: BITS must be 2, 3, 4, or default (got ${bits})" >&2
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

if [[ "${bits}" == "default" ]]; then
  # The whole point of the `default` arm: prove what the ranked worker gets when
  # nothing sets the knob. The ranked workflow exports only names it defines
  # itself, so it can never set MLX_QWEN_MTP_DRAFT_BITS.
  unset MLX_QWEN_MTP_DRAFT_BITS
else
  export MLX_QWEN_MTP_DRAFT_BITS="${bits}"
fi
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

# This host idles at a ~40.7C GPU floor (racked Mac: cpu_temp 36.8C and 0.22W
# package power while the GPU sensor reads 40.7C+, so the floor is inlet-air
# bound, not load bound), and the 40C gate therefore always hits its stalled
# abort. Acceptance, parity and the emitted token stream are thermally
# invariant, which is what this arm exists to measure, so the arms run ungated
# and every second they report is recorded as NOT gate-qualified. Arms stay
# comparable to each other -- same host, same hot state, back to back -- but not
# to any gated baseline. Authoritative timing comes from the interleaved
# QwenQMVCostCurveTests sweep instead.
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
# The `strings` gate above goes blind the moment the in-tree DEFAULT changes:
# both a stale and a fresh binary still contain that symbol. The binary digest
# is the gate that cannot go blind -- an arm that shares another arm's worker
# hash measured another arm's code, whatever the env said.
worker_sha="$(shasum -a 256 "${worker_bin}" | cut -d' ' -f1)"
cli_sha="$(shasum -a 256 .build/release/mlxfast-swift | cut -d' ' -f1)"
golden_fixture="${MLXFAST_QWEN_MTP_LOCAL_GOLDEN_FIXTURE:-<harness default>}"

identity="${out_dir}/identity.txt"
{
  echo "run-draft-bits-arm: tag=${tag} bits=${bits} tokens=${tokens} mode=${mode}"
  echo "run-draft-bits-arm: head=$(git rev-parse HEAD) dirty=$(git status --porcelain | wc -l | tr -d ' ')"
  echo "run-draft-bits-arm: base_sha=${base_sha}"
  echo "run-draft-bits-arm: host=$(sysctl -n machdep.cpu.brand_string) mem=$(sysctl -n hw.memsize)"
  echo "run-draft-bits-arm: started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "run-draft-bits-arm: env_draft_bits=${MLX_QWEN_MTP_DRAFT_BITS:-<unset>}"
  echo "run-draft-bits-arm: golden_fixture=${golden_fixture}"
  echo "worker_sha256=${worker_sha}"
  echo "cli_sha256=${cli_sha}"
  echo "cool_gate=disabled_ambient_floor"
  echo "gpu_temp_c_before=$(gpu_temp_now)"
} >"${identity}"

export MLXFAST_AMDAHL_EXTRA_CONFIG="$(
  jq -cn \
    --arg bits "${bits}" \
    --arg env_bits "${MLX_QWEN_MTP_DRAFT_BITS:-unset}" \
    --arg fixture "${golden_fixture}" \
    --arg worker_sha "${worker_sha}" \
    --arg cli_sha "${cli_sha}" \
    --arg tokens "${tokens}" \
    --arg mode "${mode}" \
    '{draft_bits_arm: $bits, env_draft_bits: $env_bits,
      golden_fixture: $fixture, worker_sha256: $worker_sha,
      cli_sha256: $cli_sha, decode_tokens: ($tokens | tonumber),
      bench_mode: $mode, cool_gate: "disabled_ambient_floor"}'
)"

# The MTP local-iterate report carries no memory field, so peak comes from
# `ru_maxrss`, which XNU folds across waited descendants as a max (in bytes).
{
  /usr/bin/time -l research/run-amdahl-measurement.sh \
    "${tag}" "${mode}" "${tokens}" "${base_sha}" \
    "draft-head readout precision: MLX_QWEN_MTP_DRAFT_BITS=${bits} mode=${mode}"
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
