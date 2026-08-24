#!/usr/bin/env bash
# E163: measure the routed quantized matvec cost of one verify round, per arm.
#
#   usage: research/e163_qmv_timing_run.sh [WIDTHS]
#
# One `swift test` process per arm, because the width plan is read once at
# process start, exactly as in the exactness gate.
#
# This is NOT a timing leg. It is an isolated kernel microbenchmark with no
# model resident, no cool gate and no score. It exists to supply the one factor
# the advisor's published-point chain is missing: the share of a per-width round
# that the routed matvec occupies.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

widths="${1:-5,6}"
out="research/e163-artifacts"
mkdir -p "${out}"

arms=(shipped minna5)
files=()
failures=0

for arm in "${arms[@]}"; do
  path="${out}/qmv-timing-${arm}.json"
  files+=("${path}")
  echo "=== isolated routed QMV timing: MLX_E163_IPG_PLAN=${arm} widths=${widths} ==="
  MLXFAST_RUN_E163_QMV_TIMING=1 \
  MLXFAST_E163_TIMING_OUT="${PWD}/${path}" \
  MLXFAST_E163_TIMING_WIDTHS="${widths}" \
  MLX_E163_IPG_PLAN="${arm}" \
    swift test --force-resolved-versions --filter E163QMVIsolatedTimingTests
  status=$?
  if ((status != 0)); then
    echo "e163_qmv_timing_run: arm ${arm} exited ${status}" >&2
    failures=$((failures + 1))
  fi
done

((failures == 0)) || exit 1
python3 research/e163_qmv_share.py "${files[@]}"
