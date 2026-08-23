#!/usr/bin/env bash
# E145 R2 -- the live per-width round cost curve.
#
#   usage: research/e145_r2_session.sh [TOKENS] [FIXTURE]
#
# THE QUESTION. Everything this campaign believes about verify width descends
# from a curve that was REBUILT from isolated kernel timings and transfer
# factors:
#
#   rows 1..9, per_drafting_round, us
#   31173  34619  38065  41511  44958  61199  62825  70315  75639
#                                      ^^^^^ the 5 -> 6 cliff, 16,241 us
#
# `pb6`, F22, the E137 transfer factor, the E138 plan surface and the whole
# depth walk are priced through that one step. It has never been read off a
# live decode.
#
# THE DESIGN. `MLX_E145_PIN_DEPTH` pins the drafted depth to a constant, so
# every drafting round realises verify width `M = depth + 1`. Widths 2 to 8 are
# swept as one palindrome inside one session, on one worker binary:
#
#     1 2 3 4 5 6 7 7 6 5 4 3 2 1       (pinned depth; width is depth + 1)
#
# Every pin has mean position 7.5, so monotone drift across the session cancels
# to first order, and every leg takes the real 40 C gate.
#
# WHY A PIN AND NOT A SHIPPED LEG. The shipped schedule chooses depth from the
# round's own state, so the rounds that reach width 6 are the rounds the
# estimator already believed were hot. A per-width cost read off a shipped leg
# is selection biased by construction. The pin removes the selection.
#
# PHASE 2 collects one UNTIMED traced leg per pin. Decoding is deterministic
# for a fixed fixture, token budget, build and pin, so the traced leg replays
# the timed leg's exact round sequence and supplies the per-round accepted
# count that the timed report does not carry. Joining them by round index
# separates the width WORK term from the rejection REPAIR term. A traced leg
# writes inside the round, so it is never used for timing.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

source research/e145_lib.sh

tokens="${1:-512}"
fixture="${2:-beagle_a}"

# The assignment asks for widths 3 to 7, which is what decides the 5 to 6 step.
# This sweeps widths 2 to 8 instead, a strict superset, for two reasons.
#
# R3 re-prices the E140 and E134 decision cells on the measured curve, and
# those cells read the curve at every width from 1 to 9. Measuring only 3 to 7
# would force widths 2 and 8 to be extrapolated across the very boundary the
# experiment exists to test, and width 8 carries most of the realised mass on
# the deep prompts. Second, the QMV grouping boundary sits at width 8, so it is
# the one other place the source predicts a non-linear step, and measuring it
# costs four gated legs.
#
# Depth 8 is not swept: the shipped envelope clamps the pin, and the deepest
# width observed from any shipped leg is 8, so pin 7 is the deepest pin that
# realises the width it asks for.
readonly PINS=(1 2 3 4 5 6 7)

golden=".mlxfast-private/e128/goldens/${fixture}-rows-$((tokens + 1)).json"
[[ -s "${golden}" ]] || {
  echo "e145_r2: no reference rows at ${golden}" >&2; exit 2; }

e145_prepare_session || exit $?

failures=0
position=0
for pin in "${PINS[@]}" $(printf '%s\n' "${PINS[@]}" | sort -rn); do
  position=$((position + 1))
  slot="r2-${fixture}-p${position}-pin${pin}"
  e145_leg "${slot}" "${fixture}" ship "${pin}" "${tokens}"
  status=$?
  echo "e145_r2_position=${position}" >> "${E145_LEG_OUT}/meta.txt"
  ((status == 0)) || {
    echo "e145_r2: ${slot} exited ${status}" >&2
    failures=$((failures + 1)); }
done

# PHASE 2: the untimed traced legs. No gate is taken and none is claimed: these
# legs answer an accounting question, not a timing question.
for pin in "${PINS[@]}"; do
  slot="r2trace-${fixture}-pin${pin}"
  out=".mlxfast-private/e128/e145/${slot}/${fixture}"
  echo "=== e145 ${slot}: untimed traced leg, pin=${pin} ==="
  env E128_FORCE=1 "E128_TOKENS=${tokens}" E128_DEPTH=8 \
      MLX_E134_DEPTH_PRICE_ARM=ship "MLX_E145_PIN_DEPTH=${pin}" \
      "E128_RUNS_DIR=e145/${slot}" \
    research/e128_session.sh "${fixture}"
  status=$?
  {
    echo "e145_slot=${slot}"
    echo "e145_pin_requested=${pin}"
    echo "e145_real_cool_gate_taken=false"
    echo "e145_leg_kind=untimed-traced-accounting"
    echo "e145_leg_exit=${status}"
  } >> "${out}/meta.txt"
  ((status == 0)) || {
    echo "e145_r2: ${slot} exited ${status}" >&2
    failures=$((failures + 1)); }
done

echo "e145_r2: ${failures} failed legs"
exit $(( failures > 0 ))
