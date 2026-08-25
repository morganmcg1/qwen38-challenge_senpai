#!/usr/bin/env bash
# E210 session driver: one traced leg with the E90 command-buffer interval
# ledger, optionally with a planted CPU stall at the top of every round.
#
#   usage: research/e210_session.sh TAG TOKENS [STALL_US]
#
# STALL_US is the Stage-0 positive control. The host sleeps for that long at
# the top of every round and submits nothing while it sleeps, so device busy
# time cannot change: the census must report the leg's GPU idle growing by
# STALL_US per round, inside `d_pre`. An instrument that cannot see a planted
# idle cannot carry the census.
#
# Research-only. The armed worker must be built before the session.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tag="${1:?usage: research/e210_session.sh TAG TOKENS [STALL_US]}"
tokens="${2:?usage: research/e210_session.sh TAG TOKENS [STALL_US]}"
stall="${3:-0}"

if [[ "${stall}" != "0" ]]; then
  export MLX_E210_STALL_US="${stall}"
fi

research/e90_leg.sh "${tag}" "${tokens}" --intervals
status=$?

{
  echo "stall_us=${stall}"
  echo "experiment=e210-gpu-idle-census"
} >> "research/out/${tag}/meta.txt"

exit "${status}"
