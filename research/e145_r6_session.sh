#!/usr/bin/env bash
# E145 R6-1 -- with wiring ACTIVE, is local decode time bimodal at the scale of
# the ranked instrument's 879 us/round state step?
#
#   usage: research/e145_r6_session.sh [TOKENS] [FIXTURE]
#
# THE QUESTION, REFRAMED AFTER F9. F8's mechanism -- that the 64 MiB slack
# holds the full-attention KV cache at 1024 tokens EXACTLY, so which buffers
# win the slack decides the state step -- is refuted at source by E130. The
# wired set at sizing is byte-identical at every slack rung and steady-state
# headroom is 0.023-0.117 MiB even at s2048, so the slack is 98.3-99.9 percent
# consumed at every rung and its size protects no particular buffer. R6-0's
# `64 MiB == 65536 B/token x 1024 tokens` equality is a coincidence.
#
# What survives is narrower and still worth twelve legs: NO CLEAN WIRED LEG
# EXISTS anywhere in the campaign. E130 rung 12 (13 legs) and E146 R-C (8 legs)
# both ran unwired; rung 11 ran wired but with a 23.27 C entry spread and no
# warmup leg. Every local statement about run-to-run reproducibility therefore
# describes a machine that never executed the residency path. This session
# supplies the missing arm and asks one falsifiable question: with wiring on,
# does the leg-to-leg spread stay at the 0.065-0.109 percent seen unwired, or
# does it open into two clusters separated by something near 879 us/round?
#
# THE OBSTACLE AND THE REUSED KEY. The guard refuses below 96 GiB, so this
# 48 GiB host has never executed the path. `MLX_E130_WIRED_GATE_GIB=32` is
# E130's existing research override (rungs 10a and 11, revert `69a6d26e`); this
# session reuses it rather than adding a second key for the same job. The knob
# cannot change the ranked M5 runner, which clears 96 and 32 alike.
#
# THE DESIGN. One discarded warmup leg -- E130 rung 12 cut the entry-temperature
# spread from 23.27 C to 1.757 C with exactly this device -- then twelve timed
# 512-token `beagle_a` legs on the compiled default arm, one binary, under the
# real 40 C gate, in the position-balanced palindrome
#
#   W U U W  W U U W  W U U W
#
# so the mean position of the wired and unwired legs is 6.5 in both arms and
# monotone thermal drift cancels to first order. Six legs per arm is the
# smallest sample that can show a two-cluster split at all.
#
# THE WITNESS COMES FREE. Every leg gets its own
# `MLX_E130_RESIDENCY_PROBE_PATH`, and each of the three workers in a leg
# appends its own `wired-zh` line there. A wired leg must write
# `request=... applied=<positive> ... gate_gib=32`; an unwired leg must write
# `skipped=gate gate_gib=96`. That pair is the Rule 114 witness and the Rule 101
# failing control in one file, at no extra GPU cost: if the override did
# nothing, both arms would write the same line and the session is invalid on
# its face. The unwired arm doubles as proof that the COMPILED default is still
# the shipped 96 GiB, because it reaches the refusal with the override unset.
#
# WHAT THIS CANNOT SHOW. The effect size will not transfer. This host has
# 48 GiB and therefore also runs the LOW-memory startup profile
# (`RuntimeStartupMemoryPolicy` puts the boundary at 64 GiB) and never sets the
# ranked box's 512 MiB post-wire command-buffer budget. A local number is
# evidence that the mechanism exists and roughly how large it is, not a
# prediction of the M5 delta.
#
# PRE-REGISTERED KILL. If the wired arm's within-arm standard deviation is
# below 0.30 percent and no pair of wired legs differs by more than 0.80
# percent, the ranked state is not locally reproducible with wiring on. Say so
# and stop; do not open a second residency rung.
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

probe_dir="${PWD}/.mlxfast-private/e128/e145/r6-probe"
mkdir -p "${probe_dir}"

e145_prepare_session || exit $?

echo "=== e145_r6 leg 00 WARMUP (discarded, unwired, absorbs the cold start) ==="
MLX_E130_RESIDENCY_PROBE_PATH="${probe_dir}/00-warmup.log" \
  e145_leg "r6-${fixture}-p00-warmup" "${fixture}" "${arm}" none "${tokens}"

failures=0
position=0
for residency in wired unwired unwired wired \
                 wired unwired unwired wired \
                 wired unwired unwired wired; do
  position=$((position + 1))
  slot="r6-${fixture}-p${position}-${residency}"
  probe="${probe_dir}/$(printf '%02d' "${position}")-${residency}.log"
  : > "${probe}"
  if [[ "${residency}" == "wired" ]]; then
    export MLX_E130_WIRED_GATE_GIB="${E145_R6_GATE_GIB:-32}"
  else
    unset MLX_E130_WIRED_GATE_GIB
  fi
  export MLX_E130_RESIDENCY_PROBE_PATH="${probe}"

  e145_leg "${slot}" "${fixture}" "${arm}" none "${tokens}"
  status=$?
  {
    echo "e145_r6_position=${position}"
    echo "e145_r6_residency=${residency}"
    echo "e145_r6_gate_gib=${MLX_E130_WIRED_GATE_GIB:-unset}"
    echo "e145_r6_probe_path=${probe}"
    echo "e145_r6_probe_lines=$(grep -c 'wired-zh' "${probe}" 2>/dev/null || echo 0)"
    echo "e145_r6_probe_applied=$(grep -c 'applied=' "${probe}" 2>/dev/null || echo 0)"
    echo "e145_r6_probe_refused=$(grep -c 'skipped=gate' "${probe}" 2>/dev/null || echo 0)"
  } >> "${E145_LEG_OUT}/meta.txt"
  ((status == 0)) || {
    echo "e145_r6: ${slot} exited ${status}" >&2
    failures=$((failures + 1)); }
done

unset MLX_E130_WIRED_GATE_GIB MLX_E130_RESIDENCY_PROBE_PATH
echo "e145_r6: ${failures} failed legs"
echo "e145_r6: witness logs under ${probe_dir}"
grep -h 'wired-zh' "${probe_dir}"/*.log 2>/dev/null | sort | uniq -c
exit $(( failures > 0 ))
