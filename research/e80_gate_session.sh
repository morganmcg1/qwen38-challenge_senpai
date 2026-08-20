#!/usr/bin/env bash
# E80 rung 1 -- run the E71 width-tax census with the GPU-time instrument on.
#
#   usage: research/e80_gate_session.sh TAG [smoke|full]
#
# The E71 driver owns the local run lock, the ABBA order and the per-block
# thermal record. This wrapper only adds the E80 environment, because run_job
# takes an argv list with no environment field.
#
# The GPU ledger writes JSONL next to the E71 census JSON. Snapshots are
# emitted per BLOCK (`endWindow`), so `MLX_E80_SNAPSHOT_ROUNDS` is set far above
# the per-block rep count on purpose: a rep-triggered snapshot would split one
# block across two records.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tag="${1:?usage: e80_gate_session.sh TAG [smoke|full]}"
profile="${2:-full}"

gpu_out="research/out/${tag}-gpu"
rm -rf "${gpu_out}"
mkdir -p "${gpu_out}"

export MLX_E80_GPU_TIME=1
export MLX_E80_SNAPSHOT_ROUNDS=1000000
export MLX_E58_DISPATCH_CENSUS=1
export MLX_E58_DISPATCH_CENSUS_SHAPES=1
export MLX_E58_DISPATCH_CENSUS_PATH="${PWD}/${gpu_out}/census.jsonl"
: > "${MLX_E58_DISPATCH_CENSUS_PATH}"

# Isolated mode forces one MLX op per command buffer, so a command-buffer GPU
# interval is one kernel's GPU time. Leave unset for the in-situ arm.
if [[ -n "${E80_OPS_PER_BUFFER:-}" ]]; then
  export MLX_E58_BUFFER_LIMIT_OPS="${E80_OPS_PER_BUFFER}"
fi

exec research/e71_census.sh "${tag}" "${profile}"
