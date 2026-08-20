#!/usr/bin/env bash
# One counterbalanced E62 session.
#
#   usage: research/e62_session.sh SESSION TOKENS ARMBIN SPEC [SPEC ...]
#
#   SESSION  tag prefix, e.g. r1-ops
#   TOKENS   decode window for every leg in the session
#   ARMBIN   which prebuilt binary every leg uses: stock | wired | census
#   SPEC     label:mb:ops:wired  e.g. ops50:4096:50:off
#
# The caller supplies the leg order and is responsible for making it
# position-balanced, so monotone thermal drift cancels to first order. Every leg
# logs to W&B as soon as it closes, never at session end.
#
# A failed leg does not stop the session: an arm that cannot complete the window
# is itself a result, and the remaining legs keep their planned positions.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

session="${1:?usage: e62_session.sh SESSION TOKENS ARMBIN SPEC [SPEC ...]}"
tokens="${2:?usage: e62_session.sh SESSION TOKENS ARMBIN SPEC [SPEC ...]}"
armbin="${3:?usage: e62_session.sh SESSION TOKENS ARMBIN SPEC [SPEC ...]}"
shift 3

index=0
failures=0
for spec in "$@"; do
  IFS=: read -r label mb ops wired <<< "${spec}"
  index=$((index + 1))
  tag="e62-${session}-$(printf '%02d' "${index}")-${label}"
  echo "=== leg ${index}: ${label} (mb=${mb} ops=${ops} wired=${wired}), ${tokens} tokens ==="
  if research/e62_run_leg.sh "${tag}" "${armbin}" "${tokens}" \
      --mb "${mb}" --ops "${ops}" --wired "${wired}" --label "${label}"; then
    echo "=== leg ${index} (${tag}) finished ==="
  else
    echo "=== leg ${index} (${tag}) FAILED; continuing the session ===" >&2
    failures=$((failures + 1))
    tail -20 "research/out/${tag}/wrapper.err" >&2 || true
  fi
  research/e62_wandb_log.py --leg "research/out/${tag}" --position "${index}" \
      --session "${session}" \
    > "research/out/${tag}/wandb.log" 2>&1 \
    || echo "=== leg ${index}: W&B logging failed ===" >&2
done

echo "=== session ${session} complete: ${index} legs, ${failures} failed ==="
exit 0
