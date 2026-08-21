#!/usr/bin/env bash
# E100 -- run every leg of one ABBA slot on ONE build.
#
#   usage: research/e100_e2e_pair.sh SLOTS ARM [SESSIONS]
#
#   SLOTS     comma-separated ABBA positions, e.g. `a1` or `b1,b2`
#   ARM       base | collapse, recorded verbatim
#   SESSIONS  comma-separated session names, default `d8,d4`
#
#     d8      64 decode tokens, offered depth 8   -- the shipped schedule
#     d4      64 decode tokens, offered depth 4   -- the reach control
#     w512    512 decode tokens, offered depth 8  -- the ranked decode window
#     w512d4  512 decode tokens, offered depth 4  -- the decisive leg. It is
#             the ranked decode window with M = 5 on nearly every round, so
#             reach stops confounding conversion.
#
# The timed leg always carries the same 512-token seed prefill, so at 64 decode
# tokens the prefill is most of the leg and dilutes any decode-side effect by
# about 6x. Adjacent slots that share an arm run in one invocation, which keeps
# a four-slot A B B A session down to two build switches.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

slots="${1:?usage: e100_e2e_pair.sh SLOTS ARM [SESSIONS]}"
arm="${2:?usage: e100_e2e_pair.sh SLOTS ARM [SESSIONS]}"
sessions="${3:-d8,d4}"

session_depth() {
  case "$1" in
    d8|w512) echo 8 ;;
    d4|w512d4) echo 4 ;;
    *) echo "unknown session $1" >&2; return 1 ;;
  esac
}

session_tokens() {
  case "$1" in
    d8|d4) echo 64 ;;
    w512|w512d4) echo 512 ;;
    *) echo "unknown session $1" >&2; return 1 ;;
  esac
}

rc=0
IFS=',' read -r -a slot_list <<< "${slots}"
IFS=',' read -r -a session_list <<< "${sessions}"
for slot in "${slot_list[@]}"; do
  for session in "${session_list[@]}"; do
    depth="$(session_depth "${session}")" || exit 2
    tokens="$(session_tokens "${session}")" || exit 2
    MLXFAST_QWEN_MTP_DEPTH="${depth}" \
    MLXFAST_QWEN_MTP_LOCAL_ITERATE_TOKENS="${tokens}" \
      research/e100_e2e_leg.sh "e100-e2e-${session}-${slot}" "${arm}" || rc=$?
  done
done
exit "${rc}"
