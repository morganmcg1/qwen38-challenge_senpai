#!/usr/bin/env bash
# Measure the candidate leg's cost as a function of a PINNED verify width.
#
#   research/e56_width_sweep.sh [--tokens N]
#
# The first E56 session showed that the stream-aware price never permits a
# 4->5 crossing at any acceptance probability, so it behaves as an
# unconditional verify-width-4 cap rather than as a cost-aware rule. That makes
# the timing result ambiguous: width 4 may win because it is the widest
# single-weight-stream verify, or simply because this fixture is faster when it
# drafts less.
#
# This sweep separates those two explanations without touching the candidate.
# `MLXFAST_QWEN_MTP_DEPTH` is the parent's offered per-round draft ceiling, and
# at this fixture's acceptance the unmodified base walk always wants depth 8,
# so an offer of k pins every round to verify width k+1. Timing the base binary
# at offers 2..5 therefore traces the round cost across widths 3, 4, 5 and 6.
#
# The staircase hypothesis makes a sharp prediction: the cost per emitted token
# must show a KINK between width 4 and width 5, where the second weight stream
# is charged. A smooth curve refutes it and leaves "shallower is faster here",
# which is a level question, not a cost-shape one.
#
# The offers run in palindrome order, so each width sits at two positions that
# are symmetric about the middle of the session and linear thermal or power
# drift cancels exactly for every point.
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

tokens=256
while (($#)); do
  case "$1" in
    --tokens) tokens="$2"; shift 2 ;;
    *) echo "e56_width_sweep: unknown argument $1" >&2; exit 2 ;;
  esac
done

export MLXFAST_LOCAL_RUN_LOCK_DIR="${MLXFAST_LOCAL_RUN_LOCK_DIR:-/tmp/mlxfast-shared}"

echo "e56_width_sweep: GPU observation before the first leg"
python3 research/e49_gpu_gate.py || true

rc=0
for spec in 2:w3a 3:w4a 4:w5a 5:w6a 5:w6b 4:w5b 3:w4b 2:w3b; do
  offer="${spec%%:*}"
  tag="${spec##*:}"
  research/e56_run_leg.sh base "${tag}" --tokens "${tokens}" --depth "${offer}" \
    || { rc=1; echo "e56_width_sweep: leg ${tag} FAILED"; }
done
date -u "+e56_width_sweep: === %Y-%m-%dT%H:%M:%SZ sweep complete ==="
exit "${rc}"
