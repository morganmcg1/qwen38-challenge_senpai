#!/usr/bin/env bash
# Research-only launcher for the E58 dispatch-storm microbenchmark.
#
# The storm holds no model, so it needs neither the run lock's residency
# protection nor the thermal gate: one full sweep moves the GPU by well under a
# degree. It DOES need its own process per MLX command-buffer limit, because MLX
# reads MLX_MAX_OPS_PER_BUFFER once, when its device is constructed.
#
# usage:
#   research/e58_run_storm.sh TAG [MLX_MAX_OPS_PER_BUFFER] [MLX_MAX_MB_PER_BUFFER]
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tag="${1:?usage: e58_run_storm.sh TAG [ops_per_buffer] [mb_per_buffer]}"
ops="${2:-}"
mb="${3:-}"

out="research/out/${tag}"
mkdir -p "${out}"

export MLXFAST_E58_DISPATCH_STORM=1
export MLXFAST_E58_DISPATCH_STORM_OUT="${PWD}/${out}/storm.json"
[[ -n "${ops}" ]] && export MLX_MAX_OPS_PER_BUFFER="${ops}"
[[ -n "${mb}" ]] && export MLX_MAX_MB_PER_BUFFER="${mb}"

swift test --force-resolved-versions --filter E58DispatchStormTests \
  > "${out}/storm.out" 2> "${out}/storm.err"
status=$?
echo "exit=${status}" >> "${out}/storm.out"
exit "${status}"
