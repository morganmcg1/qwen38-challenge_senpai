#!/usr/bin/env bash
# E65 rung 1b: which statement inside the draft section holds the round-1 cost?
#
# Arm r1probe splits draft_build into d_pre, d_flush, d_head1, d_submit1,
# d_chain and d_submit2, which tile the interval exactly. Two traced legs name
# the statement instead of the section. The trace field d_pre_us only exists in
# this arm, so the parsed census is itself the content witness.
#
# This session attributes the cost. It does not time the arm.
set -uo pipefail
cd "$(dirname "$0")/.."

session="${1:-r1b}"
status=0
run() {
  local tag="$1"; shift
  echo "=== ${tag}: $* ==="
  if research/e65_run_leg.sh "${tag}" "$@"; then
    echo "=== ${tag} finished ==="
  else
    echo "=== ${tag} FAILED; continuing ==="
    status=1
  fi
}

run "e65-${session}-01-probe512a" r1probe 512 --label r1probe-census-512-a
run "e65-${session}-02-probe512b" r1probe 512 --label r1probe-census-512-b

exit "${status}"
