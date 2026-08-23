#!/usr/bin/env bash
# E145 R2b -- the width-1 anchor the R2 palindrome could not reach.
#
#   usage: research/e145_r2b_session.sh [TOKENS] [FIXTURE]
#
# THE PROBLEM. R2 swept pins 1 to 7, so it measured verify widths 2 to 8. The
# shipped price table is
#
#     cumulative[d] = round_us(d + 1) / round_us(1)
#
# so EVERY entry is divided by the width-1 round cost. R3 re-prices the E140
# and E134 decision cells on the measured curve, and without a measured width 1
# the whole table would rest on a back-extrapolation across the region where
# the measured curve is most clearly convex. R2 measured the 2 to 3 step at
# 4768 us, the 3 to 4 step at 7991 us and the 4 to 5 step at 12114 us, so a
# straight line through the low widths is already falsified; extrapolating one
# more step below width 2 would be the least defensible number in R3.
#
# THE DESIGN. Pin 0. `chooseDepth` returns 0 before it reads the estimator, so
# the round drafts nothing and realises verify width 1. This is the same
# non-drafting round the shipped schedule produces when its walk declines the
# first step, which is exactly the round the price table normalises by.
#
# It is NOT the same thing as the `--mtp-depth 0` serial control. That control
# is measured for the score denominator and runs a verify-only row with no head
# step and no session bookkeeping. Reporting the serial round cost as width 1
# would understate the width-1 cost and therefore overstate every ratio in the
# table. Both are collected here so the gap between them is measured rather
# than assumed.
#
# Pin 4 is repeated as a CROSS-SESSION TIE POINT. R2 measured width 5 inside
# its own session; measuring it again here places the width-1 anchor on the R2
# level instead of assuming two sessions share a level. The palindrome
#
#     0 4 4 0
#
# gives every pin mean position 2.5, so monotone drift cancels to first order,
# and each leg takes the real 40 C gate.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

source research/e145_lib.sh

tokens="${1:-512}"
fixture="${2:-beagle_a}"

golden=".mlxfast-private/e128/goldens/${fixture}-rows-$((tokens + 1)).json"
[[ -s "${golden}" ]] || {
  echo "e145_r2b: no reference rows at ${golden}" >&2; exit 2; }

e145_prepare_session || exit $?

failures=0
position=0
for pin in 0 4 4 0; do
  position=$((position + 1))
  slot="r2b-${fixture}-p${position}-pin${pin}"
  e145_leg "${slot}" "${fixture}" ship "${pin}" "${tokens}"
  status=$?
  echo "e145_r2b_position=${position}" >> "${E145_LEG_OUT}/meta.txt"
  ((status == 0)) || {
    echo "e145_r2b: ${slot} exited ${status}" >&2
    failures=$((failures + 1)); }
done

echo "e145_r2b: ${failures} failed legs"
exit $(( failures > 0 ))
