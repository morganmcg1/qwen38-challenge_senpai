#!/usr/bin/env bash
# E116 rung 1 -- prove the dose is real, bit exact, and measured.
#
#   usage: research/e116_rung1.sh
#
# Four legs, in this order, and the order matters: every cheap falsification
# runs before the expensive one.
#
#   1. e116r1-iso-k1      64 tokens, dose 1, ONE dispatch per command buffer.
#                         The clean isolated M=1 rate for one dose unit, which
#                         is the number E107 measured at 410.93 us for this
#                         exact cell. A dose of 1 removes any concurrency
#                         between dose dispatches from the rate.
#   2. e116r1-insitu-k0   64 tokens, dose 0, default buffer geometry.
#   3. e116r1-insitu-k4   64 tokens, dose 4, default buffer geometry.
#                         (3) minus (2) is the GPU time the dose adds to a
#                         round under the SAME command-buffer geometry a timed
#                         leg has. That difference, not the isolated rate, is
#                         the denominator of the rung 2 absorption coefficient.
#   4. e116r1-neg-force1  64 tokens, dose 0, verify width pinned to 1.
#                         The negative control for the row digest.
#
# The 512-token exactness pair is run separately by
# `research/e116_rung1_exact512.sh`, because it costs about ten minutes and
# nothing above needs to wait for it.
#
# THERMAL. Every leg here is ungated (MLXFAST_LOCAL_COOL_GATE=0 inside the leg
# scripts). None of them is a timing leg: a census leg is invalid for wall
# clock by construction, and a trace leg here exists only to emit rows.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ -n "$(git status --porcelain)" ]]; then
  echo "e116_rung1: worktree is dirty; refusing to measure over uncommitted work" >&2
  exit 1
fi

failures=0
run() {
  echo
  echo "=== $* ==="
  "$@" || { echo "e116_rung1: FAILED: $*" >&2; failures=$((failures + 1)); }
}

run research/e116_census_leg.sh e116r1-iso-k1 realised 64 1 0
run research/e116_census_leg.sh e116r1-insitu-k0 realised 64 0
run research/e116_census_leg.sh e116r1-insitu-k4 realised 64 4
run research/e116_exactness_leg.sh e116r1-neg-force1 64 0 1

echo
echo "e116_rung1: ${failures} failed legs"
exit "${failures}"
