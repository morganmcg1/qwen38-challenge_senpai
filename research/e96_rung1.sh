#!/usr/bin/env bash
# E96 rung 1 + 1b: measure the GDN recurrent step by removing it.
#
#   usage: research/e96_rung1.sh [TOKENS] [FORCE_DRAFTS]
#
# Nine legs per direction, counterbalanced inside one ungated session:
#
#   vendor    the unmodified kernel and its (32, 4, 1) threadgroup
#   clone     a byte-identical clone dispatched from Qwen35.swift, same geometry
#   rep R=1   the clone plus the repetition scaffold, one repetition
#   rep R=2,4,8,16  the same scan repeated, bit-identical output
#   t1        the clone with the t-loop forced to one iteration
#   off       no dispatch: y = v, state_out = state_in
#
# vendor - clone isolates any cost of moving the dispatch into the editable
# file. clone - rep R=1 isolates the scaffold. The slope of round cost against
# R measures one step directly, over arms that share one token stream and one
# (d, acc) distribution. off and t1 are the removal bracket, and they are NOT
# bucket matched: they emit different tokens, so their rounds land at different
# acceptance counts.
#
# Every arm runs through research/e96_direct_leg.sh, including the two
# unablated controls. The wrapper cannot time an ablated arm at all: its step 1
# public drift tripwire compares against the M5 golden and exits first, and an
# ablation changes the tokens by construction. Running the controls through the
# wrapper and the ablations through the direct CLI would confound the arm with
# the harness, so all eight legs take the identical path.
#
# The session builds nothing. Build and witness the worker before it starts.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tokens="${1:-128}"
drafts="${2:-4}"
failures=0

# slot:mode:repeat. The order is a palindrome, so every arm's two legs sit
# symmetrically about the session midpoint and monotone thermal drift cancels
# to first order.
for spec in a1:vendor:1 a2:clone:1 a3:rep:1 a4:rep:2 a5:rep:4 a6:rep:8 \
            a7:rep:16 a8:t1:1 a9:off:1 \
            b9:off:1 b8:t1:1 b7:rep:16 b6:rep:8 b5:rep:4 b4:rep:2 b3:rep:1 \
            b2:clone:1 b1:vendor:1
do
  IFS=: read -r slot mode reps <<<"${spec}"
  MLX_E96_REPEAT="${reps}" \
    research/e96_direct_leg.sh "e96r1-${slot}-${mode}${reps}" "${tokens}" \
      "${mode}" 4 "${drafts}"
  status=$?
  failures=$((failures + (status != 0)))
  echo "e96_rung1: leg ${slot} mode=${mode} R=${reps} exit=${status}"
done

echo "e96_rung1: ${failures} failed legs"
exit "${failures}"
