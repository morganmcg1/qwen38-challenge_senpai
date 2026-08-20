#!/usr/bin/env bash
# E79 rung 1 + rung 2 session: one ABBA-counterbalanced block per head.
#
#   A = plain phase trace. The head chain stays asynchronous, so head-chain
#       GPU time hides inside the trailing asyncEval and verify_build_us.
#   B = MLX_QWEN_MTP_TRACE_SYNC_HEAD=1. The chain is drained before the verify
#       window, so that same GPU time moves into draft_build_us.
#
# The A-B difference is the IN-SITU head chain cost. Running A1 B1 B2 A2 makes
# monotone thermal drift cancel to first order in that difference.
#
#   usage: research/e79_session.sh PREFIX [--pinned-head]
#
# Without `--pinned-head` the block runs the head `mtp-head.manifest.json`
# declares, which is the head the ranked candidate leg executes. With it, the
# block runs the organizer-pinned bf16 head. That is a genuine head VARIANT,
# so the pinned block is the rung-3a natural experiment, not a control.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

prefix="${1:?usage: e79_session.sh PREFIX [--pinned-head]}"
shift
if [[ "${1:-}" == "--pinned-head" ]]; then
  export E79_HEAD_DIR="${HOME}/.cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head"
fi

research/e79_trace_leg.sh "${prefix}-a1-census512" 512 || exit $?
research/e79_trace_leg.sh "${prefix}-b1-synchead512" 512 --sync-head || exit $?
research/e79_trace_leg.sh "${prefix}-b2-synchead512" 512 --sync-head || exit $?
research/e79_trace_leg.sh "${prefix}-a2-census512" 512 || exit $?
