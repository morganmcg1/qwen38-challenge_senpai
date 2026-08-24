#!/usr/bin/env bash
# E163 F4 check 2: measure this host's streaming memory bandwidth.
#
#   usage: research/e163_bandwidth_run.sh [SIZES_GIB] [REPS]
#
# No model is resident, so this holds no model lock and is not a timing leg.
# It supplies the MEASURED denominator for the round's achieved bandwidth,
# replacing the quoted 273 GB/s datasheet figure.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

sizes="${1:-0.25,1,4,14}"
reps="${2:-9}"
out="research/e163-artifacts"
mkdir -p "${out}"
path="${out}/bandwidth-peak.json"

echo "=== streaming bandwidth peak: sizes=${sizes} GiB reps=${reps} ==="
MLXFAST_RUN_E163_BANDWIDTH=1 \
MLXFAST_E163_BW_OUT="${PWD}/${path}" \
MLXFAST_E163_BW_SIZES_GIB="${sizes}" \
MLXFAST_E163_BW_REPS="${reps}" \
  swift test --force-resolved-versions --filter E163MemoryBandwidthTests
status=$?
if ((status != 0)); then
  echo "e163_bandwidth_run: swift test exited ${status}" >&2
  exit "${status}"
fi

python3 research/e163_round_bandwidth.py "${path}"
