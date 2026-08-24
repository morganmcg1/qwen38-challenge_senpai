#!/usr/bin/env bash
# E163: run the exactness gate once per width-plan arm and diff the arms.
#
#   usage: research/e163_exactness_run.sh
#
# One `swift test` process per arm, because the plan is read once at process
# start. Each process writes the per-cell digests, the actual output floats it
# compared, and the positive-control result for its own arm. The comparison is
# then a pure file diff with no GPU in it.
#
# This gate must pass before any timed leg. It is the only thing that proves a
# regrouping moves no output bit; the timing session assumes that and cannot
# detect it.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

out="research/e163-artifacts"
mkdir -p "${out}"

arms=(shipped minna5 minna456)
files=()
failures=0

for arm in "${arms[@]}"; do
  path="${out}/exactness-${arm}.json"
  files+=("${path}")
  echo "=== exactness gate: MLX_E163_IPG_PLAN=${arm} ==="
  MLXFAST_RUN_E163_EXACTNESS=1 \
  MLXFAST_E163_OUT="${PWD}/${path}" \
  MLX_E163_IPG_PLAN="${arm}" \
    swift test --force-resolved-versions --filter E163IPGPlanExactnessTests
  status=$?
  if ((status != 0)); then
    echo "e163_exactness_run: arm ${arm} exited ${status}" >&2
    failures=$((failures + 1))
  fi
done

((failures == 0)) || exit 1
python3 research/e163_exactness_compare.py "${files[@]}"
