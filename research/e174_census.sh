#!/usr/bin/env bash
# E174 step-0 census: read the live chunk-sum fill surface on this base.
#
#   usage: research/e174_census.sh [TOKENS] [LABEL]
#
# Ledger 332.3 asked for this number and nobody has read it: the block session
# already prints `xs_hit` and `xs_fill` every round, and the pair says how much
# of the 257-site table-paying fill surface the inherited fusion covers. The
# E173 count of 130 unserved cells is source-derived; this is the run.
#
# TWO ARMS, one worker, no rebuild between legs, both untimed:
#
#   on    shipped sidecar        expect xs_hit ~127, xs_fill ~130 per round
#   off   MLX_E174_XSUMS_SIDECAR=off, producer publishes nothing
#                                expect xs_hit 0, xs_fill ~257 per round
#
# The `off` arm is the positive control for the `on` arm and the reverse: each
# arm's census must fail the other arm's expectation, so a switch that never
# reached the worker cannot pass. Arithmetic is identical in both arms, so no
# exactness gate is needed beyond the accept ledger the legs already report.
#
# UNGATED ON PURPOSE. These legs are a counter census, not a timing arm. No
# duration from them is quotable: `gate_qualified_for_timing=false`.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tokens="${1:-32}"
label="${2:-c1}"

if [[ -n "$(git status --porcelain -- Sources Vendor Package.swift)" ]]; then
  echo "e174_census: scored surface is dirty; refusing to census over" \
       "uncommitted work" >&2
  exit 1
fi

for arm in on off; do
  tag="e174${label}${arm}"
  echo "=== ${tag}: arm=${arm} tokens=${tokens} untimed ==="
  if [[ "${arm}" == "off" ]]; then
    export MLX_E174_XSUMS_SIDECAR=off
  else
    unset MLX_E174_XSUMS_SIDECAR
  fi
  research/e79_trace_leg.sh "${tag}" "${tokens}"
  status=$?
  unset MLX_E174_XSUMS_SIDECAR
  echo "e174_arm=${arm}" >> "research/out/${tag}/meta.txt"
  if ((status != 0)); then
    echo "e174_census: ${tag} exited ${status}" >&2
    exit 5
  fi
done

python3 research/e174_census.py --label "${label}"
