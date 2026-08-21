#!/usr/bin/env bash
# E96 rung 3 census leg: one isolated dispatch per command buffer.
#
#   usage: research/e96_census_leg.sh TAG DRAFTS TOKENS [OPS_PER_BUFFER]
#
# `research/e93_gputime_leg.sh` sets `MLX_E58_BUFFER_LIMIT_OPS` only. The rung 3
# question needs the byte limit pinned as well, so a large dispatch cannot share
# a command buffer with its neighbour and hide behind it. This wrapper sets both
# limits, because run_job takes an argv list with no environment field.
#
# The ops limit must be 0, not 1. MLX commits on `buffer_ops_ > max_ops`
# (`device.cpp:485-486`), so `max_ops = 1` admits a second op before it commits
# and every small kernel stays paired with its neighbour. Measured on this host
# with `OPS=1` and `MB=1` together: signature `...,custom_kernel_qwen35_fused
# _residual` reported 6 dispatches over 3 buffers. `max_ops = 0` commits after
# the first op, which is the only setting that isolates one kernel per buffer.
#
# A census leg is NEVER a timing leg. The census swizzle serialises every
# dispatch, so host wall clock is invalid and only Metal's own GPU clock counts.
# It also requires the research-only census instrument:
#
#   git apply -3 research/e95-artifacts/e95-census-instrument.patch
#
# Revert that patch before any submission scope check.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tag="${1:?usage: e96_census_leg.sh TAG DRAFTS TOKENS [OPS_PER_BUFFER]}"
drafts="${2:?usage: e96_census_leg.sh TAG DRAFTS TOKENS [OPS_PER_BUFFER]}"
tokens="${3:?usage: e96_census_leg.sh TAG DRAFTS TOKENS [OPS_PER_BUFFER]}"
ops="${4:-0}"

export MLX_E58_BUFFER_LIMIT_MB=1
research/e93_gputime_leg.sh "${tag}" "${drafts}" "${tokens}" "${ops}"
status=$?

echo "buffer_limit_mb=1" >> "research/out/${tag}/meta.txt"
exit "${status}"
