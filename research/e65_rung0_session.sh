#!/usr/bin/env bash
# E65 rung 0: the cold first-touch census on the unmodified base.
#
#   usage: research/e65_rung0_session.sh SESSION
#
# Four legs, all on the same arm, so no arm contrast is confounded with
# position:
#
#   1. 512 tokens, traced      - the census leg
#   2. 512 tokens, traced      - replication. A spike at the SAME round index
#                                in both legs is structural; a spike that moves
#                                is a scheduler artifact.
#   3. 640 tokens, traced      - deconfounding. At exactly 512 decode tokens the
#                                kL >= 1024 crossing round is ALSO the last
#                                round of the leg. At 640 tokens the crossing
#                                lands mid-leg with same-width peers after it,
#                                so "crossing" and "last round" separate.
#   4. 512 tokens, sync-head   - attribution. verify_build_us overlaps the
#                                asynchronously submitted head chain, so it
#                                cannot tell host graph build from a head-chain
#                                GPU stall. Draining the chain first moves head
#                                GPU time into draft_build_us. Diagnostic only:
#                                this leg is never a timed contrast.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

session="${1:?usage: e65_rung0_session.sh SESSION}"

run() {
  local tag="$1" tokens="$2"
  shift 2
  echo "=== ${tag}: ${tokens} tokens $* ==="
  if research/e65_run_leg.sh "${tag}" base "${tokens}" "$@"; then
    echo "=== ${tag} finished ==="
  else
    echo "=== ${tag} FAILED; continuing ===" >&2
    tail -30 "research/out/${tag}/wrapper.err" >&2 || true
  fi
}

run "e65-${session}-01-census512a" 512 --label census-512-a
run "e65-${session}-02-census512b" 512 --label census-512-b
run "e65-${session}-03-census640" 640 --label census-640-deconfound
run "e65-${session}-04-synchead512" 512 --sync-head --label synchead-512-attrib

echo "=== rung 0 session ${session} complete ==="
