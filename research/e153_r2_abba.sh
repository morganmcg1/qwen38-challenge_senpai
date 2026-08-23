#!/usr/bin/env bash
# E153 R2: does the merged wide-decode SDPA kernel beat the shipped two-call
# split end to end, on top of the leaf16 surface this branch already ships?
#
#   usage: research/e153_r2_abba.sh [REPLICATES] [LABEL] [FIRST]
#   env:   E153_PHASES=0,1,2,3   which phases to run (default all)
#          E153_NO_BUILD=1       skip the compile in phase 0, keep the
#                                selector assertion
#
# A = split    MLX_E153_MERGED_SDPA=0, the shipped two sdpa calls at 6 <= M <= 9
# B = merged   the compiled default, one custom Metal kernel over the same KV
#
# BOTH ARMS RUN THE SAME BINARY and the same leaf16 default, so this session
# prices the R2 increment ALONE. R1 is not a variable here: it is compiled in
# on both sides.
#
# WHY THE SCHEDULE WITNESS CANNOT SEPARATE THESE ARMS, AND WHAT DOES.
# `Tests/MLXFastTests/E153MergedSdpaKernelTests` measured the merged kernel
# BITWISE identical to the split at every scored cell, so both arms must decode
# the same tokens, take the same accept/reject decisions and report the same
# `round_count` and `effective_mean_draft_len`. That identity is the exactness
# result; it also means the usual Rule 114 schedule signature is blind here.
# The arm witness is therefore the kernel's own line, written to
# `MLX_QWEN_MTP_TRACE_PATH` by `Qwen35MergedSdpaVector.recordWitness`. Setting
# only `MLX_QWEN_MTP_TRACE_PATH`, and NOT `MLX_QWEN_MTP_TRACE=1`, gives the arm
# witness without the per-round trace that would void the timing.
#
# EQUAL SCHEDULES ARE ALSO THE CLEANEST POSSIBLE COMPARISON. E149 arm A had to
# argue about a leaf-width change that moved acceptance. Here any difference in
# seconds per token is attributable to the kernel and nothing else, because the
# two arms perform the identical sequence of rounds at the identical widths.
#
# REAL 40 C GATE. `benchmark.sh --local-cool-gate-only` runs before every timed
# leg and returns before the resident-model lock is taken, so it does not fight
# the lock `research/e128_session.sh` takes for the leg.
#
# ORDER. Inside one replicate the order per prompt is A B B A, so both arms sit
# at mean position 2.5 and monotone drift cancels to first order.
#
# NO REBUILD BETWEEN LEGS. Phase 0 builds once and asserts both selectors are
# in the binary that is about to run (HARNESS DEFECT 36).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

replicates="${1:-1}"
label="${2:-s1}"
first="${3:-1}"
phases="${E153_PHASES:-0,1,2,3}"

tokens=512
depth=8
prompts=(benchfixture beagle_a essays_montaigne)
runs_parent=".mlxfast-private/e128/runs-e153r2"

has_phase() { [[ ",${phases}," == *",$1,"* ]]; }

dirty="$(git status --porcelain -- Sources Vendor Package.swift \
  Package.resolved mtp-head.manifest.json)"
if [[ -n "${dirty}" ]]; then
  echo "e153_r2: scored surface is dirty; refusing to time over uncommitted" \
       "work" >&2
  echo "${dirty}" >&2
  exit 1
fi
session_commit="$(git rev-parse HEAD)"

if has_phase 0; then
  echo "=== e153_r2 phase 0: worker build and selector assertion ==="
  # `--require` reads the string table, which is where the arm selector, the
  # kernel name and the Metal source literal live. `--require-symbol` reads
  # `nm -a`, which is the only place a Swift identifier appears.
  build_args=(--require MLX_E153_MERGED_SDPA
    --require MLX_E141_ROWS_PER_LEAF
    --require qwen35_merged_sdpa_vector
    --require 'const int causal_limit = N - qL + q_seq_idx;'
    --require-symbol qwen35MergedSdpaVector
    --require-symbol warmQwen35MergedSdpaVector
    --require-symbol activeClusterRowsPerLeaf
    --require-symbol qwen35E141RowsPerLeafOverride)
  [[ "${E153_NO_BUILD:-0}" == "1" ]] && build_args+=(--no-build)
  senpai/rebuild-and-assert-worker.sh "${build_args[@]}" || {
    echo "e153_r2: the worker does not carry the merged kernel; not timing" >&2
    exit 3; }
  CLANG_MODULE_CACHE_PATH="${PWD}/.build/clang-module-cache" \
    swift build -c release --force-resolved-versions --product mlxfast-swift \
    || { echo "e153_r2: CLI build failed" >&2; exit 3; }
fi

session_worker="$(
  shasum -a 256 .build-worker/release/mlxfast-runtime-worker | awk '{print $1}')"
session_cli="$(
  shasum -a 256 .build/release/mlxfast-swift | awk '{print $1}')"
echo "e153_r2: session_commit=${session_commit}"
echo "e153_r2: session_worker_sha256=${session_worker}"
echo "e153_r2: session_cli_sha256=${session_cli}"

# Phase 1 -- Rule 137 warmup and reference-row generation. Reference rows must
# exist BEFORE the gated phase: generating a golden inside a gated leg would
# heat the GPU between the gate and the clock. These legs are TRACED, so
# research/e128_session.sh writes timing_valid=false on them.
if has_phase 1; then
  for id in "${prompts[@]}"; do
    echo "=== e153_r2 phase 1: warmup and goldens for ${id} ==="
    env E128_FORCE=1 \
        E128_TOKENS="${tokens}" \
        E128_DEPTH="${depth}" \
        E128_RUNS_DIR="runs-e153r2/warm-${id}" \
      research/e128_session.sh "${id}"
    out="${runs_parent}/warm-${id}/${id}"
    {
      echo "e153_leg_role=warmup-discarded"
      echo "e153_session_commit=${session_commit}"
    } >> "${out}/meta.txt"
  done
fi

failures=0
voided=0
if has_phase 2; then
 for ((rep = first; rep < first + replicates; rep++)); do
  for id in "${prompts[@]}"; do
    position=0
    for arm in split merged merged split; do
      position=$((position + 1))
      slot="${label}k${rep}p${position}${arm}"
      out="${runs_parent}/${slot}/${id}"

      echo "=== e153_r2 cool gate before ${slot} ${id} (real 40C) ==="
      if ./benchmark.sh --local-cool-gate-only; then
        gate_status=passed
      else
        gate_status=stalled_above_40C
      fi

      echo "=== e153_r2 ${slot}: prompt=${id} arm=${arm}" \
           "replicate=${rep} position=${position} gate=${gate_status} ==="
      # MLX_QWEN_MTP_TRACE_PATH without MLX_QWEN_MTP_TRACE=1: the arm witness
      # lands in the leg's own trace.txt and the per-round trace stays off.
      env ${arm:+MLX_E153_MERGED_SDPA=$([[ "${arm}" == split ]] && echo 0 || echo 1)} \
          MLX_QWEN_MTP_TRACE_PATH="${PWD}/${out}/trace.txt" \
          E128_FORCE=1 \
          E128_NO_TRACE=1 \
          E128_TOKENS="${tokens}" \
          E128_DEPTH="${depth}" \
          E128_RUNS_DIR="runs-e153r2/${slot}" \
        research/e128_session.sh "${id}"
      status=$?

      observed_arm="$(sed -n 's/^qwen35-merged-sdpa: arm=\([a-z]*\).*/\1/p' \
        "${out}/trace.txt" 2>/dev/null | sort -u | tr '\n' ',' | sed 's/,$//')"
      witness=ok
      if [[ "${observed_arm}" != "${arm}" ]]; then
        witness=ARM_MISMATCH
        voided=$((voided + 1))
        echo "e153_r2: ${slot} ${id} asked for ${arm} but the kernel" \
             "witnessed '${observed_arm:-none}'" >&2
      fi
      {
        echo "e153_leg_role=timed"
        echo "e153_arm_requested=${arm}"
        echo "e153_arm_witnessed=${observed_arm:-none}"
        echo "e153_merged_sdpa_exported=$(
          [[ "${arm}" == split ]] && echo 0 || echo 1)"
        echo "e153_rows_per_leaf=16-compiled-default"
        echo "e153_cool_gate_real_gate_invoked=1"
        echo "e153_cool_gate_status=${gate_status}"
        echo "e153_gate_qualified_for_timing=$(
          [[ "${gate_status}" == passed ]] && echo true || echo false)"
        echo "e153_witness=${witness}"
        echo "e153_replicate=${rep}"
        echo "e153_position=${position}"
        echo "e153_session_commit=${session_commit}"
        echo "e153_session_worker_sha256=${session_worker}"
        echo "e153_session_cli_sha256=${session_cli}"
      } >> "${out}/meta.txt"

      if ((status != 0)); then
        echo "e153_r2: ${slot} ${id} exited ${status}" >&2
        failures=$((failures + 1))
      fi
    done
  done
 done
 echo "e153_r2: ${failures} failed legs, ${voided} voided witnesses"
fi

if has_phase 3; then
  python3 research/e153_r2_report.py --label "${label}" \
    --runs "${runs_parent}" --out research/e153-r2-abba.json
fi
exit $(( failures > 0 ))
