#!/usr/bin/env bash
# Research-only (e198-route1-row-amortized-kernel): build the Stage 4 tree and
# certify what is inside the worker that the session will time.
#
#   research/e198_build.sh
#
# E198 needs ONE built tree, not one tree per arm. The arm is
# DARKBLOOM_QWEN_FUSED_SDPA_ROWS, which `FusedRowAmortizedSDPA.enabledRows` reads
# once per worker process, so every leg runs a byte-identical binary and the
# session asserts that one worker digest before and after every leg. RULE 384.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

worker=".build-worker/release/mlxfast-runtime-worker"
metallib=".build-worker/release/mlx.metallib"

mkdir -p .build/clang-module-cache .build-worker/clang-module-cache
CLANG_MODULE_CACHE_PATH="${PWD}/.build/clang-module-cache" \
  swift build -c release --force-resolved-versions --product mlxfast-swift || exit 1
CLANG_MODULE_CACHE_PATH="${PWD}/.build-worker/clang-module-cache" \
  swift build -c release --force-resolved-versions \
  --scratch-path .build-worker --product mlxfast-runtime-worker || exit 1
tools/build-mlx-metallib.sh || exit 1

count_in_worker() { strings -a "${worker}" | grep -c -F -- "$1" || true; }

# The kernel body is a Swift string literal, so it survives into the binary.
# The env name is the arm switch. MLX_QWEN_MTP_TRACE is the positive control
# that proves the string probe can see into this binary at all.
kernel_needle='qwen_mtp_fused_row_amortized_sdpa'
causal_needle='if (i <= N - M + q_row) {'
arm_needle='DARKBLOOM_QWEN_FUSED_SDPA_ROWS'
control_needle='MLX_QWEN_MTP_TRACE'

kernel_count="$(count_in_worker "${kernel_needle}")"
causal_count="$(count_in_worker "${causal_needle}")"
arm_count="$(count_in_worker "${arm_needle}")"
control_count="$(count_in_worker "${control_needle}")"

# Read the worktree state BEFORE `tee` creates the record, which would
# otherwise report itself as an untracked file.
worktree_clean="$([[ -z "$(git status --porcelain)" ]] && echo true || echo false)"

status=0
if [[ "${control_count}" -eq 0 ]]; then
  echo "e198_build: positive control '${control_needle}' absent; the string probe is broken" >&2
  status=1
fi
for pair in "kernel:${kernel_count}" "causal:${causal_count}" "arm:${arm_count}"; do
  if [[ "${pair##*:}" -eq 0 ]]; then
    echo "e198_build: worker carries no ${pair%%:*} needle" >&2
    status=1
  fi
done

{
  echo "experiment=e198-route1-row-amortized-kernel"
  echo "tree=${PWD}"
  echo "organizer_sha=$(git rev-parse upstream/main)"
  echo "assignment_head=$(git rev-parse HEAD)"
  echo "worktree_clean=${worktree_clean}"
  echo "scored_diff_vs_organizer=$(git diff --numstat upstream/main -- Sources Vendor | awk '{a+=$1; d+=$2; n+=1} END {printf "%d files +%d -%d", n, a, d}')"
  echo "worker_sha256=$(shasum -a 256 "${worker}" | cut -d' ' -f1)"
  echo "cli_sha256=$(shasum -a 256 .build/release/mlxfast-swift | cut -d' ' -f1)"
  echo "metallib_sha256=$(shasum -a 256 "${metallib}" | cut -d' ' -f1)"
  echo "metallib_fingerprint=$(tools/build-mlx-metallib.sh --print-fingerprint 2>/dev/null | tail -1)"
  echo "needle_kernel_name=${kernel_count}"
  echo "needle_causal_rule=${causal_count}"
  echo "needle_arm_switch=${arm_count}"
  echo "needle_control_trace=${control_count}"
  echo "twin_audit<<EOF"
  python3 research/twin_audit.py 2>&1 | tail -5
  echo "EOF"
} | tee "research/e198-build.txt"

exit "${status}"
