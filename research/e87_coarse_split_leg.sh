#!/usr/bin/env bash
# E87 section 4 -- isolate the two arm C coarse-readout matmuls.
#
#   usage: research/e87_coarse_split_leg.sh TAG DRAFTS TOKENS
#
# `MLX_E58_BUFFER_LIMIT_OPS=1` alone does NOT isolate one kernel per command
# buffer. MLX commits on `buffer_ops_ > max_ops`
# (Vendor/mlx-swift/.../backend/metal/device.cpp:484), so a limit of 1 packs
# TWO ops into every buffer. The arm C stage matmuls therefore share a buffer
# with the `argPartition` mbsort chain that follows them, and the buffer
# interval prices both together.
#
# The same predicate also commits on `(buffer_sizes_ >> 20) > max_mb`. Both
# stage matmuls touch far more than 2 MiB (19.67 MB and 39.33 MB), so adding
# `MLX_E58_BUFFER_LIMIT_MB=1` commits each of them on its own, while the small
# mbsort and copy ops keep pairing under the op limit. That gives an ISOLATED
# GPU interval for each stage without any kernel change.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tag="${1:?usage: e87_coarse_split_leg.sh TAG DRAFTS TOKENS}"
drafts="${2:?usage: e87_coarse_split_leg.sh TAG DRAFTS TOKENS}"
tokens="${3:?usage: e87_coarse_split_leg.sh TAG DRAFTS TOKENS}"

export MLX_E58_BUFFER_LIMIT_MB=1

research/e93_gputime_leg.sh "${tag}" "${drafts}" "${tokens}" 1
status=$?

echo "buffer_limit_mb=1" >> "research/out/${tag}/meta.txt"
exit "${status}"
