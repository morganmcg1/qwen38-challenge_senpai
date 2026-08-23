#!/usr/bin/env bash
# E145 R1b -- the Rule 128 warm-phase telemetry probe.
#
#   usage: research/e145_r1b_session.sh [TOKENS] [FIXTURE]
#
# THE QUESTION. CAMPAIGN RULE 128 says two arms are comparable only when their
# warm phase left the process in the same residency and cache state. The R1 and
# R2 sessions launched before `e145_lib.sh` recorded that telemetry, so every
# leg already collected reports `warm_telemetry_present=false`. This probe adds
# the missing evidence without repeating a timed session.
#
# It also decides between the two live explanations of the six-point board
# disagreement. R1 killed the regime hypothesis: `pb6` is faster on both local
# fixtures. FINDING 220's interaction hypothesis is what remains. If the `ship`
# and `pb6` arms report identical `wired-zh` and `warm` fields, then the arm
# cannot be moving residency state on its own and the interaction, if it
# exists, needs a second mechanism present to appear.
#
# THE DESIGN. Four legs, palindrome `ship, pb6, pb6, ship`, at the same 512
# tokens R1 used. Both telemetry lines are written before the timed window, so
# a shorter leg would answer the Rule 128 question just as well, but only the
# 513-row golden exists and generating a shorter one costs its own GPU run. At
# 512 tokens the probe therefore doubles as a SECOND SESSION replicate of R1's
# decisive `beagle_a` arm effect, which is what Rule 119 asks for: the session
# is the unit that drifts, so a result seen once in one session is not yet
# reproduced.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

source research/e145_lib.sh

tokens="${1:-512}"
fixture="${2:-beagle_a}"

golden=".mlxfast-private/e128/goldens/${fixture}-rows-$((tokens + 1)).json"
[[ -s "${golden}" ]] || {
  echo "e145_r1b: no reference rows at ${golden}; generate them before the" \
       "session so no gated leg pays for a golden" >&2
  exit 2; }

e145_prepare_session || exit $?

failures=0
position=0
for arm in ship pb6 pb6 ship; do
  position=$((position + 1))
  slot="r1b-${fixture}-p${position}-${arm}"
  e145_leg "${slot}" "${fixture}" "${arm}" none "${tokens}"
  status=$?
  echo "e145_r1b_position=${position}" >> "${E145_LEG_OUT}/meta.txt"
  ((status == 0)) || {
    echo "e145_r1b: ${slot} exited ${status}" >&2
    failures=$((failures + 1)); }
done

echo "e145_r1b: ${failures} failed legs"
exit $(( failures > 0 ))
