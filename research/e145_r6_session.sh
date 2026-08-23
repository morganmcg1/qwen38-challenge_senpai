#!/usr/bin/env bash
# E145 R6-1 -- does the residency wiring path change decode time, and is it
# bimodal?
#
#   usage: research/e145_r6_session.sh [TOKENS] [FIXTURE]
#
# THE QUESTION. FINDING 235 prices the ranked instrument's one-in-three
# +903 us/drafting-round failure mode at 1.65 % of the published median, which
# is larger than every mechanism now in hand. F8's hypothesis is that the
# failure mode is the wired-residency lottery: `wireResidentWeightsIfEnabled()`
# sizes the wired limit at the post-warm live footprint plus 64 MiB, and which
# buffers win that 64 MiB is decided once, greedily, in allocation order.
#
# R6-0 read the byte counts and found the slack cannot hold the decode state:
# the full-attention KV cache at 1024 tokens is 64 MiB EXACTLY, but the 48
# float32 GDN recurrent states are another 144 MiB and the per-round snapshot
# is 147 MiB more, so the persistent decode state is 3.29x the slack and the
# round peak is 5.59x. The lottery is real; the question is what winning it is
# worth.
#
# THE OBSTACLE. The guard refuses below 96 GiB, so this 48 GiB host has never
# executed the path. `MLX_E145_WIRED_MIN_GIB=32` lowers the floor for the
# candidate legs only. The knob cannot change the ranked M5 runner, which
# clears 96 and 32 alike.
#
# THE DESIGN. Twelve 512-token `beagle_a` legs on the compiled default arm,
# under the real 40 C gate, in the position-balanced palindrome
#
#   W U U W  W U U W  W U U W
#
# so the mean position of the wired and unwired legs is 6.5 in both arms and
# monotone thermal drift cancels to first order. Six legs per arm is the
# smallest sample that can show a two-cluster split at all.
#
# THE WITNESS COMES FREE. Every leg appends its own `wired-zh` line to
# `E145_R6_WIRED_LOG`. A wired leg must write `request=... applied=...`; an
# unwired leg must write `skipped=guard min_gib=96`. That pair is the Rule 114
# witness and the Rule 101 failing control in one file, at no extra GPU cost:
# if the override did nothing, both arms would write the same line and the
# session is invalid on its face.
#
# WHAT THIS CANNOT SHOW. The effect size will not transfer. This host has
# 48 GiB and therefore also runs the LOW-memory startup profile
# (`RuntimeStartupMemoryPolicy` puts the boundary at 64 GiB) and never sets the
# ranked box's 512 MiB post-wire command-buffer budget. A local number is
# evidence that the mechanism exists and roughly how large it is, not a
# prediction of the M5 delta.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

source research/e145_lib.sh

tokens="${1:-512}"
fixture="${2:-beagle_a}"
arm="${E145_R6_ARM:-pb6}"

golden=".mlxfast-private/e128/goldens/${fixture}-rows-$((tokens + 1)).json"
[[ -s "${golden}" ]] || {
  echo "e145_r6: no reference rows at ${golden}; generate them before the" \
       "session so no gated leg pays for a golden" >&2
  exit 2; }

export E145_R6_WIRED_LOG="${PWD}/.mlxfast-private/e128/e145/r6-wired.log"
mkdir -p "$(dirname "${E145_R6_WIRED_LOG}")"
: > "${E145_R6_WIRED_LOG}"

e145_prepare_session || exit $?

failures=0
position=0
for residency in wired unwired unwired wired \
                 wired unwired unwired wired \
                 wired unwired unwired wired; do
  position=$((position + 1))
  slot="r6-${fixture}-p${position}-${residency}"
  echo "--- e145_r6 position ${position}: ${residency} ---" \
    >> "${E145_R6_WIRED_LOG}"
  if [[ "${residency}" == "wired" ]]; then
    export MLX_E145_WIRED_MIN_GIB="${E145_R6_MIN_GIB:-32}"
  else
    unset MLX_E145_WIRED_MIN_GIB
  fi
  export MLX_E145_WIRED_LOG="${E145_R6_WIRED_LOG}"

  e145_leg "${slot}" "${fixture}" "${arm}" none "${tokens}"
  status=$?
  {
    echo "e145_r6_position=${position}"
    echo "e145_r6_residency=${residency}"
    echo "e145_r6_min_gib=${MLX_E145_WIRED_MIN_GIB:-unset}"
    echo "e145_r6_wired_log=${E145_R6_WIRED_LOG}"
  } >> "${E145_LEG_OUT}/meta.txt"
  ((status == 0)) || {
    echo "e145_r6: ${slot} exited ${status}" >&2
    failures=$((failures + 1)); }
done

unset MLX_E145_WIRED_MIN_GIB
echo "e145_r6: ${failures} failed legs"
echo "e145_r6: witness log ${E145_R6_WIRED_LOG}"
grep -c 'wired-zh' "${E145_R6_WIRED_LOG}" 2>/dev/null \
  | sed 's/^/e145_r6: witness lines /'
exit $(( failures > 0 ))
