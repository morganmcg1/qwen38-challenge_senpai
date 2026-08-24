#!/usr/bin/env bash
# E174 step-1 dedup census: how many of the standalone chunk-sum table fills
# in one MTP round describe an activation some OTHER fill in the same round
# already described?
#
#   usage: research/e174_dedup_census.sh [TOKENS] [LABEL]
#
# `Qwen35CustomQMV.xsumsTable(x)` reads its `k` and `m` out of `x`, so the
# table it returns is a pure function of the activation. Two routed cells that
# read the same normed activation -- the ordinary transformer shape, where one
# norm feeds several projections -- therefore launch two dispatches that
# compute the SAME numbers. The sidecar's `take` dedupes producer to consumer
# only. Nothing dedupes consumer to consumer.
#
# TWO ARMS, one worker, no rebuild between legs, both untimed and traced:
#
#   s   shipped tree. 127 cells served by the sidecar, ~130 standalone fills.
#       This is the surface a dedup arm could actually remove today.
#   o   MLX_E174_XSUMS_SIDECAR=off. The producer publishes nothing, so all
#       ~257 table-paying cells go standalone. This is the full fill surface
#       the advisor named, and it is the positive control for arm `s`: the two
#       arms must disagree on `xs_fill` in the direction the census predicts,
#       so a switch that never reached the worker cannot pass both.
#
# The instrument carries its own positive control. `Qwen35XSumsDedupCensus`
# keys duplicates on `ObjectIdentifier`, and every round's summary is prefixed
# with `ok` only when two references to one array and one separately allocated
# array key as 2 distinct identities with exactly one 2x bucket. An all-1x
# census behind a `FAIL` prefix is an instrument fault, not a finding.
#
# UNGATED ON PURPOSE. These legs are a counter census, not a timing arm. No
# duration from them is quotable: `gate_qualified_for_timing=false`.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tokens="${1:-512}"
label="${2:-d1}"

if [[ -n "$(git status --porcelain -- Sources Vendor Package.swift)" ]]; then
  echo "e174_dedup_census: scored surface is dirty; refusing to census over" \
       "uncommitted work" >&2
  exit 1
fi

export MLX_E174_DEDUP_CENSUS=1
export MLX_QWEN_MTP_TRACE=1

for arm in s o; do
  tag="e174dedup${label}${arm}"
  echo "=== ${tag}: arm=${arm} tokens=${tokens} untimed ==="
  if [[ "${arm}" == "o" ]]; then
    export MLX_E174_XSUMS_SIDECAR=off
  else
    unset MLX_E174_XSUMS_SIDECAR
  fi
  research/e79_trace_leg.sh "${tag}" "${tokens}"
  status=$?
  unset MLX_E174_XSUMS_SIDECAR
  echo "e174_arm=${arm}" >> "research/out/${tag}/meta.txt"
  if ((status != 0)); then
    echo "e174_dedup_census: ${tag} exited ${status}" >&2
    exit 5
  fi
done

python3 research/e174_dedup_census.py --label "${label}"
