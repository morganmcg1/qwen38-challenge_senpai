#!/usr/bin/env bash
# E134 item 0 -- the warm-parity arms, ABBA-counterbalanced inside one session.
#
#   usage: research/e134_item0_matrix.sh [PROMPT_ID ...]
#
# ORDER. Each prompt runs the arm list forward and then backward, so every arm
# occupies one early and one late session slot and a monotone thermal trend
# cancels to first order in the arm difference. `E134_REP` separates the two
# passes on disk; the forward pass is rep 1 and the mirrored pass is rep 2.
#
# PROMPTS. `beagle_a` is the highest-weight drafting prompt in the F83 set.
# `plutarch_lives` carries 0.0000 F83 weight and drafts on a small minority of
# its rounds, which is what makes it the discriminating control the advisor
# asked for in F8 item 3: a first-touch deletion is a fixed millisecond saving
# over a leg and must read LARGER in per cent on plutarch, while a residency
# effect scales with drafting work and must read near zero there.
#
# W-REFILL IS NOT IN THIS MATRIX. Advisor F8 has a ranked receipt for it at
# +5.28 % F83-weighted slower. Run it separately through
# `research/e134_warm_session.sh refill ...` only as a labelled negative
# control, and never on this host as evidence about the ranked runner.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

arms=(base wnorm wprefetch all)
prompts=("$@")
((${#prompts[@]})) || prompts=(beagle_a plutarch_lives)

status=0
for prompt in "${prompts[@]}"; do
  for ((i = 0; i < ${#arms[@]}; i++)); do
    E134_REP=1 research/e134_warm_session.sh "${arms[i]}" "${prompt}" || status=1
  done
  for ((i = ${#arms[@]} - 1; i >= 0; i--)); do
    E134_REP=2 research/e134_warm_session.sh "${arms[i]}" "${prompt}" || status=1
  done
done
exit "${status}"
