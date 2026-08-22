#!/usr/bin/env bash
# E137 F2 step 1: does every scored linear cell reach Route B at every realised
# verify width, or do some widths fall into the generic `quantized.h` gate?
#
# F2 section 5 states the hypothesis: if some scored cells decline `routable`
# at M >= 6 they land on `case 6: <T, 6, 3, true>`, ipg 3, two weight passes,
# which would explain both the M=5 to M=6 step and the FINDING 156 shortfall.
#
# `Qwen35CustomQMV.notePipeline` (`Qwen35.swift:1998-2013`) counts one routed
# dispatch per width, so `by_width[M]` over a whole leg is the number of scored
# cells that Route B CLAIMED at width M. The same leg's trace gives the number
# of forwards run at width M. 257 linear cells per forward is the closure test.
#
# THIS LEG IS NOT A TIMING LEG. The pipeline log adds a host dictionary update
# to every routed dispatch, so its seconds per token are not comparable with
# any timed arm. It is a routing census only.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tag="${1:-e137pipe512}"
tokens="${2:-512}"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "e137_pipeline_census: worktree is dirty; refusing to measure over" \
       "uncommitted work" >&2
  git status --porcelain >&2
  exit 1
fi

log="${PWD}/research/out/${tag}-pipelines.json"
mkdir -p "$(dirname "${log}")"
rm -f "${log}"
export MLX_E120_QMV_PIPELINE_LOG="${log}"

research/e79_trace_leg.sh "${tag}" "${tokens}" || exit 1

python3 research/e137_pipeline_census.py --tag "${tag}"
