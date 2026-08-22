#!/usr/bin/env bash
# E134 item 1 -- run the four warm arms over four prompts in a counterbalanced
# order, so a monotone thermal trend across the session cannot align with one
# arm. Each row is one prompt and each arm appears once in every session
# position.
#
#   usage: research/e134_item1_matrix.sh
#
# Every leg is one process, one model load, one trace file. The session script
# owns the run lock, so the legs are serial by construction.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

matrix=(
  "beagle_a         base clear clearrefill refill"
  "botany_andrews   refill clearrefill clear base"
  "essays_bacon     clear base refill clearrefill"
  "medicine_hippoc  clearrefill refill base clear"
)

status=0
for row in "${matrix[@]}"; do
  read -r prompt arms <<< "${row}"
  for arm in ${arms}; do
    research/e134_warm_session.sh "${arm}" "${prompt}" || status=1
  done
done
exit "${status}"
