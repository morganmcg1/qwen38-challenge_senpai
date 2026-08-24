#!/usr/bin/env bash
# Research-only (E202): build the barrier-arm tree and certify what is inside
# the worker that the session will time.
#
#   research/e202_build.sh
#
# E202 needs ONE built tree. The arm rotates IN PROCESS at every round boundary
# (RULE 388), selected by DARKBLOOM_E202_BARRIER_OFFSET, so every leg runs a
# byte-identical binary and the session asserts that one worker digest before
# and after every leg. RULE 384.
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

# The arm switch name and the witness field names are Swift string literals, so
# they survive into the binary. MLX_QWEN_MTP_TRACE is the positive control that
# proves the string probe can see into this binary at all.
arm_needle='DARKBLOOM_E202_BARRIER_OFFSET'
witness_needle='e202_barriers='
control_needle='MLX_QWEN_MTP_TRACE'

arm_count="$(count_in_worker "${arm_needle}")"
witness_count="$(count_in_worker "${witness_needle}")"
control_count="$(count_in_worker "${control_needle}")"

worktree_clean="$([[ -z "$(git status --porcelain)" ]] && echo true || echo false)"

status=0
if [[ "${control_count}" -eq 0 ]]; then
  echo "e202_build: positive control '${control_needle}' absent; the string probe is broken" >&2
  status=1
fi
for pair in "arm:${arm_count}" "witness:${witness_count}"; do
  if [[ "${pair##*:}" -eq 0 ]]; then
    echo "e202_build: worker carries no ${pair%%:*} needle" >&2
    status=1
  fi
done

{
  echo "experiment=e202-eval-barrier-dispatch-settlement"
  echo "tree=${PWD}"
  echo "organizer_sha=$(git rev-parse upstream/main)"
  echo "assignment_head=$(git rev-parse HEAD)"
  echo "worktree_clean=${worktree_clean}"
  echo "scored_diff_vs_organizer=$(git diff --numstat upstream/main -- Sources Vendor | awk '{a+=$1; d+=$2; n+=1} END {printf "%d files +%d -%d", n, a, d}')"
  echo "worker_sha256=$(shasum -a 256 "${worker}" | cut -d' ' -f1)"
  echo "cli_sha256=$(shasum -a 256 .build/release/mlxfast-swift | cut -d' ' -f1)"
  echo "metallib_sha256=$(shasum -a 256 "${metallib}" | cut -d' ' -f1)"
  echo "metallib_fingerprint=$(tools/build-mlx-metallib.sh --print-fingerprint 2>/dev/null | tail -1)"
  echo "needle_arm_switch=${arm_count}"
  echo "needle_witness_field=${witness_count}"
  echo "needle_control_trace=${control_count}"
  echo "twin_audit<<EOF"
  python3 research/twin_audit.py 2>&1 | tail -5
  echo "EOF"
} | tee "research/e202-build.txt"

exit "${status}"
