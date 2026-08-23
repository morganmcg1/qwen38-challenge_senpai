#!/usr/bin/env bash
# E145 R1 -- the regime sweep. Does `pb6` pay in deep regimes only?
#
#   usage: research/e145_r1_session.sh [TOKENS] [FIXTURE ...]
#
# THE QUESTION. Three measurements of the same arm disagree by 5.9 points.
# The E134 item 5b same-binary ABBA on `benchfixture` made `pb6` 2.2467 %
# FASTER. The ranked board receipt `572b2cc4 -> e003a86d` made beagle 3.65 %
# SLOWER. The offline replay predicted beagle 2.3 % faster. `benchfixture`
# drafts 6.36 tokens per round and lives above the width-6 cliff; ranked beagle
# drafts 4.38 and mostly sits below it. If `pb6` pays only where the schedule
# would otherwise spend real mass at width 6 and above, all three readings are
# compatible and the difference is a REGIME, not a contradiction.
#
# THE DESIGN. Per fixture, five legs in one palindrome:
#
#     ship, pb6, serial, pb6, ship
#
# `ship` sits at positions 1 and 5 and `pb6` at 2 and 4, so both arms have mean
# position 3 and monotone thermal drift cancels to first order (RULE 102). The
# depth-0 serial control sits at position 3, also mean position 3, and it reads
# no depth price at all, so one serial leg per fixture is the denominator for
# both arms' local ratio.
#
# UNLIKE E134's SESSION, EVERY LEG TAKES THE REAL 40 C GATE. See the header of
# `research/e145_lib.sh`.
#
# ONE BINARY, TWO ARMS. The arm is chosen at run time through
# `MLX_E134_DEPTH_PRICE_ARM`, so no leg differs from another by a single
# compiled byte. Phase 0 proves the selector is inside the worker before any
# leg runs. The compiled default is NOT touched by this session.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

source research/e145_lib.sh

tokens="${1:-512}"
shift || true
fixtures=("$@")
# `benchfixture` runs FIRST because it is the positive control. It is the only
# fixture with a published prior same-binary arm effect (E134 item 5b, `pb6`
# 2.2467 % faster). If this rig cannot reproduce that number, no other fixture
# it measures can be trusted, so the control must be able to stop the session
# before the other fixtures spend GPU time.
((${#fixtures[@]})) || fixtures=(benchfixture beagle_a essays_montaigne)

for fixture in "${fixtures[@]}"; do
  golden=".mlxfast-private/e128/goldens/${fixture}-rows-$((tokens + 1)).json"
  [[ -s "${golden}" ]] || {
    echo "e145_r1: no reference rows at ${golden}; generate them before the" \
         "session so no gated leg pays for a golden" >&2
    exit 2; }
done

e145_prepare_session || exit $?

failures=0
for fixture in "${fixtures[@]}"; do
  position=0
  for arm in ship pb6 serial pb6 ship; do
    position=$((position + 1))
    slot="r1-${fixture}-p${position}-${arm}"
    e145_leg "${slot}" "${fixture}" "${arm}" none "${tokens}"
    status=$?
    echo "e145_r1_position=${position}" >> "${E145_LEG_OUT}/meta.txt"
    ((status == 0)) || {
      echo "e145_r1: ${slot} exited ${status}" >&2
      failures=$((failures + 1)); }
  done
done

echo "e145_r1: ${failures} failed legs"
exit $(( failures > 0 ))
