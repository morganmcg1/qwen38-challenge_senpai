#!/usr/bin/env bash
# E92 rung 2: one ordered sequence of width-pinned legs.
#
#   usage: research/e92_width_session.sh SUFFIX TOKENS PINS [--production]
#
#   SUFFIX        one letter appended to each tag, for example `a` or `b`.
#                 Run the reverse pin order under the other letter so the
#                 session is ABBA-counterbalanced against thermal drift.
#   PINS          comma-separated draft counts. Verify width is M = pin + 1,
#                 so `0,1,4,5,8` measures M in {1, 2, 5, 6, 9}.
#   --production  drop `--sync-head`. The head chain then overlaps the verify
#                 window again, which is the shipped behaviour and the control
#                 for how much the attribution flag perturbs device time.
#
# Every leg runs with the GPU interval ledger on, the declared head, the local
# cool gate disabled and 512 decode tokens by default. Tags are
# `e92w<M><SUFFIX>` and `e92p<M><SUFFIX>` for the production form.
#
# The session builds nothing. Build and witness the worker before it starts.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

suffix="${1:?usage: e92_width_session.sh SUFFIX TOKENS PINS [--production]}"
tokens="${2:?usage: e92_width_session.sh SUFFIX TOKENS PINS [--production]}"
pins="${3:?usage: e92_width_session.sh SUFFIX TOKENS PINS [--production]}"
shift 3

production=0
declare -a legargs=(--intervals)
while (($#)); do
  case "$1" in
    --production) production=1; shift ;;
    *) echo "e92_width_session.sh: unknown argument $1" >&2; exit 2 ;;
  esac
done
((production)) || legargs+=(--sync-head)

prefix="e92w"
((production)) && prefix="e92p"

failures=0
IFS=',' read -r -a pin_list <<< "${pins}"
for pin in "${pin_list[@]}"; do
  width=$((pin + 1))
  tag="${prefix}${width}${suffix}"
  echo "=== ${tag}: pinned drafts=${pin}, verify width M=${width} ==="
  MLX_E92_PIN_DRAFTS="${pin}" research/e90_leg.sh "${tag}" "${tokens}" "${legargs[@]}"
  status=$?
  {
    echo "e92_pinned_drafts=${pin}"
    echo "e92_verify_width=${width}"
    echo "e92_form=$( ((production)) && echo production || echo sync_head)"
    echo "experiment=e92-rung2"
  } >> "research/out/${tag}/meta.txt"
  if ((status != 0)); then
    echo "e92_width_session: ${tag} exited ${status}" >&2
    failures=$((failures + 1))
  fi
done

exit "${failures}"
