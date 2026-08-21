#!/usr/bin/env bash
# E93 rung 2 -- run one decode leg with Metal command-buffer GPU timestamps on.
#
#   usage: research/e93_gputime_leg.sh TAG DRAFTS TOKENS [OPS_PER_BUFFER]
#
# `research/e85_census_leg.sh` owns the head, the lock, the forced draft width
# and the census environment. This wrapper only adds the E80 GPU clock, because
# run_job takes an argv list with no environment field.
#
# Two geometries answer two different questions from the same binary:
#
#   no OPS_PER_BUFFER   MLX packs many ops into one command buffer, so a buffer
#                       interval is an IN-SITU total. `by_phase` then gives the
#                       head pass per draft, which must land in the 2261-2381 us
#                       band E82, E85, E90 and the composed tree established.
#   OPS_PER_BUFFER=1    one MLX op per command buffer, so a buffer interval is
#                       ONE kernel's GPU time. `exclusive_kernels` then resolves
#                       per-shape cost. This removes intra-buffer concurrency,
#                       so it over-states each kernel; the ratio of the two
#                       totals is the concurrency discount.
#
# Host wall clock is invalid in both geometries: the census lock serialises
# every dispatch. GPU nanoseconds come from Metal's own clock and stay valid,
# which is the same footing E85 used for its 2285.283 us head pass. Host
# `dispatch_ns` is encode cost and is never a duration.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tag="${1:?usage: e93_gputime_leg.sh TAG DRAFTS TOKENS [OPS_PER_BUFFER]}"
drafts="${2:?usage: e93_gputime_leg.sh TAG DRAFTS TOKENS [OPS_PER_BUFFER]}"
tokens="${3:?usage: e93_gputime_leg.sh TAG DRAFTS TOKENS [OPS_PER_BUFFER]}"
ops_per_buffer="${4:-}"

export MLX_E80_GPU_TIME=1
# One snapshot per round keeps warmup rounds droppable offline.
export MLX_E80_SNAPSHOT_ROUNDS="${MLX_E80_SNAPSHOT_ROUNDS:-1}"
if [[ -n "${ops_per_buffer}" ]]; then
  export MLX_E58_BUFFER_LIMIT_OPS="${ops_per_buffer}"
fi

research/e85_census_leg.sh "${tag}" "${drafts}" "${tokens}" ab
status=$?

{
  echo "gputime=1"
  echo "ops_per_buffer=${ops_per_buffer:-default}"
  echo "snapshot_rounds=${MLX_E80_SNAPSHOT_ROUNDS}"
} >> "research/out/${tag}/meta.txt"

exit "${status}"
