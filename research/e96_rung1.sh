#!/usr/bin/env bash
# E96 rung 1 + 1b: measure the GDN recurrent step by removing it.
#
#   usage: research/e96_rung1.sh [TOKENS] [FORCE_DRAFTS]
#
# Four arms, ABBA-counterbalanced inside one ungated session:
#
#   vendor  the unmodified kernel and its (32, 4, 1) threadgroup
#   clone   a byte-identical clone dispatched from Qwen35.swift, same geometry
#   t1      the clone with the t-loop forced to one iteration      (rung 1b)
#   off     no dispatch: y = v, state_out = state_in               (rung 1)
#
# vendor - clone isolates any cost of moving the dispatch into the editable
# file. clone - off is the true cost of the step. clone - t1 divided by
# (T - 1) is the per-timestep cost, and t1 - per_timestep is launch plus
# state traffic.
#
# The session builds nothing. Build and witness the worker before it starts.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tokens="${1:-128}"
drafts="${2:-4}"
failures=0

for spec in v1:vendor:4 c1:clone:4 t1:t1:4 o1:off:4 \
            o2:off:4 t2:t1:4 c2:clone:4 v2:vendor:4; do
  IFS=: read -r slot mode y <<<"${spec}"
  research/e96_leg.sh "e96r1-${slot}-${mode}" "${tokens}" "${mode}" "${y}" "${drafts}"
  status=$?
  failures=$((failures + (status != 0)))
  echo "e96_rung1: leg ${slot} mode=${mode} exit=${status}"
done

echo "e96_rung1: ${failures} failed legs"
exit "${failures}"
