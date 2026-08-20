#!/usr/bin/env bash
# E78 rung 2b: whole-leg timing of the four dispatch tables in ONE session.
#
#   research/e78_rung2b.sh [--legs N] [--ungated]
#
# Order is A B C D D C B A. With N legs per group every arm holds the same mean
# leg position, so a monotone session drift cancels to first order in every
# pairwise arm contrast. One discarded `warm` group runs first so no arm pays
# the first-load cost.
#
# The primary metric is ABSOLUTE candidate seconds per token against the
# `a_ship` group measured in this same session. The local serial-to-MTP ratio is
# secondary: both local legs run the same candidate build, so a change that
# speeds the target path can cancel in that ratio.
#
# The arms differ only in the QMV dispatch table, so every arm must present the
# identical rows-per-round histogram. research/e78_analyze.py fails the session
# if any arm does not.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

base_sha="8d938c911df52b6a324f259a55dbaa75e508c822"
legs="2"
gate="1"

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --legs) legs="$2"; shift 2 ;;
    --ungated) gate="0"; shift ;;
    *) echo "e78_rung2b: unknown argument $1" >&2; exit 2 ;;
  esac
done

export E66_BASE_SHA="${E66_BASE_SHA:-${base_sha}}"
export E42_BASE_SHA="${E42_BASE_SHA:-${base_sha}}"
export E66_ARMS_MODULE="research/e78_arms.py"
export E66_MANIFEST_DIR="${PWD}/.mlxfast-private/e78/arms"
export E66_LEG_RUNNER="research/e78-run.sh"
export E66_WANDB_LOGGER="research/e78_wandb_leg.py"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-e78-width-dependent-inner-group-count}"
export MLXFAST_LOCAL_RUN_LOCK_DIR="${MLXFAST_LOCAL_RUN_LOCK_DIR:-/tmp/mlxfast-shared}"
export E78_COOL_GATE="${gate}"

echo "e78_rung2b: legs_per_group=${legs} cool_gate=${gate} base=${base_sha}"

research/e66_whole_leg_session.sh \
  "a_ship:warm:1" \
  "a_ship:a1:${legs}" \
  "b_crown:b1:${legs}" \
  "c_hybrid24928:c1:${legs}" \
  "d_hybrid8192:d1:${legs}" \
  "d_hybrid8192:d2:${legs}" \
  "c_hybrid24928:c2:${legs}" \
  "b_crown:b2:${legs}" \
  "a_ship:a2:${legs}"
status=$?

python3 research/e78_analyze.py --out research/e78-artifacts/rung2b.json
analysis=$?

echo "e78_rung2b: session_status=${status} analysis_status=${analysis}"
((status == 0)) || exit "${status}"
exit "${analysis}"
