#!/usr/bin/env bash
# E80 rung 2 -- the per-kernel GPU-time census.
#
#   usage: research/e80_census_session.sh TAG [WIDTHS] [TOKENS]
#
# Runs one 512-token leg per verify width, in the order the assignment fixes:
# 6, 5, 1, 4, 9. Width 6 carries the largest ranked mass (33.4 %), so it runs
# first and is the least exposed if the session is cut short. Width 1 is the
# serial control that anchors F(1).
#
# Each width runs TWICE:
#
#   default   many dispatches per command buffer. This is the true cost of the
#             round, and the only configuration whose absolute milliseconds mean
#             anything. It cannot price an individual kernel, because a buffer
#             interval covers everything inside it.
#
#   isolated  MLX_MAX_OPS_PER_BUFFER=1, so a command-buffer interval IS one
#             dispatch's GPU time. This is the only configuration that resolves
#             per-kernel cost. It also removes intra-buffer concurrency, so its
#             total overstates the round.
#
# in-situ / isolated is the concurrency discount, which rung 0c asks for per
# family. The two modes of one width run back to back so that ratio is not
# contaminated by drift between distant parts of the session.
#
# The real 40 C cool gate stays ON. e58_run_arm.sh reads entry and exit GPU
# temperature per leg, and the census is a timing measurement: an ungated leg
# would need ABBA counterbalancing that a one-pass width sweep does not have.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tag="${1:?usage: e80_census_session.sh TAG [WIDTHS] [TOKENS]}"
widths="${2:-6,5,1,4,9}"
tokens="${3:-512}"

# The shipped draftPolicy is an operator k-test pinned to k = 1, so an unforced
# leg only ever measures width 1. Every other width needs both a forced draft
# count and an offered depth wide enough to hold it.
depth=8

summary="research/out/${tag}-summary.txt"
mkdir -p "$(dirname "${summary}")"
: > "${summary}"

started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "session=${tag} started=${started} widths=${widths} tokens=${tokens}" \
  | tee -a "${summary}"

status=0
IFS=',' read -r -a width_list <<< "${widths}"
for width in "${width_list[@]}"; do
  drafts=$((width - 1))
  if ((drafts < 0 || drafts > depth)); then
    echo "e80_census_session.sh: width ${width} needs ${drafts} drafts, outside 0..${depth}" \
      | tee -a "${summary}"
    status=2
    continue
  fi
  for mode in default isolated; do
    leg="${tag}-w${width}-${mode}"
    args=("${leg}" --gpu-time --force-drafts "${drafts}" --depth "${depth}"
          --tokens "${tokens}")
    [[ "${mode}" == "isolated" ]] && args+=(--buffer-ops 1)

    echo "--- leg=${leg} width=${width} drafts=${drafts} mode=${mode} start=$(date -u +%H:%M:%S)" \
      | tee -a "${summary}"
    research/e58_run_arm.sh "${args[@]}"
    rc=$?
    lines=0
    [[ -f "research/out/${leg}/census.jsonl" ]] \
      && lines="$(wc -l < "research/out/${leg}/census.jsonl" | tr -d ' ')"
    echo "--- leg=${leg} exit=${rc} census_lines=${lines} end=$(date -u +%H:%M:%S)" \
      | tee -a "${summary}"
    # A failed leg must not abort the sweep: the widths are independent, and a
    # partial census still answers the ranked-weighting question for the widths
    # that did land. The non-zero exit is preserved for the caller.
    ((rc != 0)) && status="${rc}"
  done
done

echo "session=${tag} finished=$(date -u +%Y-%m-%dT%H:%M:%SZ) exit=${status}" \
  | tee -a "${summary}"
exit "${status}"
