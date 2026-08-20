#!/usr/bin/env bash
# E79 rung 1 + rung 2 session: complete the ABBA order started by
# `e79-r1-a1-census512`.
#
#   A = plain phase trace. The head chain stays asynchronous, so head-chain
#       GPU time hides inside verify_build_us.
#   B = MLX_QWEN_MTP_TRACE_SYNC_HEAD=1. The chain is drained before the verify
#       window, so that same GPU time moves into draft_build_us.
#
# The A-B difference in draft_build_us is the IN-SITU head chain cost. Running
# A1 B1 B2 A2 makes monotone thermal drift cancel to first order in that
# difference.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

research/e79_trace_leg.sh e79-r2-b1-synchead512 512 --sync-head || exit $?
research/e79_trace_leg.sh e79-r2-b2-synchead512 512 --sync-head || exit $?
research/e79_trace_leg.sh e79-r1-a2-census512 512 || exit $?
