#!/usr/bin/env bash
# E65 rung 1: does warming the live priming expression remove the round-1
# host-build spike the rung-0 census localised?
#
# Two traced legs of arm r1warm mirror rung-0 legs 01 and 02 exactly, so the
# round-1 cell comparison is like-for-like. This session decides the mechanism;
# it does not time the arm. Rung 3 owns counterbalanced timing.
set -uo pipefail
cd "$(dirname "$0")/.."

session="${1:-r1}"
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

run "e65-${session}-01-census512a" r1warm 512 --label r1warm-census-512-a
run "e65-${session}-02-census512b" r1warm 512 --label r1warm-census-512-b

exit "${status}"
