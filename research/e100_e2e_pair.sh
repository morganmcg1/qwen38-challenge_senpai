#!/usr/bin/env bash
# E100 -- run every leg of one ABBA slot on ONE build.
#
#   usage: research/e100_e2e_pair.sh SLOT ARM [PLAN]
#
#   SLOT  a1 | b1 | b2 | a2, the ABBA position
#   ARM   base | collapse
#   PLAN  dual64 (default) | w512 | dual64+w512
#
# Two builds and four slots give the ABBA session. Running every leg of a slot
# back to back keeps the number of build switches at three while still
# counterbalancing each session on its own.
#
#   dual64  the 64-token depth-8 and depth-4 sessions.
#   w512    the ranked decode window. The timed leg always carries the same
#           512-token seed prefill, so at 64 decode tokens the prefill is most
#           of the leg and dilutes any decode-side effect by about 6x. 512
#           decode tokens is the window program.md requires for a credible
#           candidate, and it also multiplies the round count by 8.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

slot="${1:?usage: e100_e2e_pair.sh SLOT ARM [PLAN]}"
arm="${2:?usage: e100_e2e_pair.sh SLOT ARM [PLAN]}"
plan="${3:-dual64}"

rc=0
case "${plan}" in *dual64*)
  for depth in 8 4; do
    MLXFAST_QWEN_MTP_DEPTH="${depth}" \
      research/e100_e2e_leg.sh "e100-e2e-d${depth}-${slot}" "${arm}" || rc=$?
  done
esac
case "${plan}" in *w512*)
  MLXFAST_QWEN_MTP_DEPTH=8 MLXFAST_QWEN_MTP_LOCAL_ITERATE_TOKENS=512 \
    research/e100_e2e_leg.sh "e100-e2e-w512-${slot}" "${arm}" || rc=$?
esac
exit "${rc}"
