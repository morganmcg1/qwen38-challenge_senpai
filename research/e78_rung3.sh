#!/usr/bin/env bash
# E78 rung 3: whole-leg timing of `e_kdown`, the only table the rung 2a cells
# score faster than the base.
#
#   research/e78_rung3.sh [--legs N] [--ungated]
#
# Order is A E E A. With N legs per group both arms hold the same mean leg
# position, so a monotone session drift cancels to first order. One discarded
# `warm` group runs first so neither arm pays the first-load cost.
#
# The primary metric is ABSOLUTE candidate seconds per token against the
# `a_ship` group measured in this same session.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

base_sha="8d938c911df52b6a324f259a55dbaa75e508c822"
legs="3"
gate="1"

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --legs) legs="$2"; shift 2 ;;
    --ungated) gate="0"; shift ;;
    *) echo "e78_rung3: unknown argument $1" >&2; exit 2 ;;
  esac
done

export E66_BASE_SHA="${E66_BASE_SHA:-${base_sha}}"
export E42_BASE_SHA="${E42_BASE_SHA:-${base_sha}}"
export E66_ARMS_MODULE="research/e78_arms.py"
export E66_MANIFEST_DIR="${PWD}/.mlxfast-private/e78/arms"
export E66_LEG_RUNNER="research/e78-run.sh"
export E66_WANDB_LOGGER="research/e78_wandb_leg.py"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-e78-rung3-kdown}"
export MLXFAST_LOCAL_RUN_LOCK_DIR="${MLXFAST_LOCAL_RUN_LOCK_DIR:-/tmp/mlxfast-shared}"
export E78_COOL_GATE="${gate}"

echo "e78_rung3: legs_per_group=${legs} cool_gate=${gate} base=${base_sha}"

research/e66_whole_leg_session.sh \
  "a_ship:warm:1" \
  "a_ship:a1:${legs}" \
  "e_kdown:e1:${legs}" \
  "e_kdown:e2:${legs}" \
  "a_ship:a2:${legs}"
status=$?

python3 research/e78_analyze.py --rung 3 \
  --out research/e78-artifacts/rung3.json
analysis=$?

echo "e78_rung3: session_status=${status} analysis_status=${analysis}"
((status == 0)) || exit "${status}"
exit "${analysis}"
