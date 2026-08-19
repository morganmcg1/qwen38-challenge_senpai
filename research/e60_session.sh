#!/usr/bin/env bash
# One counterbalanced E60 session.
#
#   usage: research/e60_session.sh TOKENS ARM [ARM ...]
#   e.g.   research/e60_session.sh 300 A B C C B A
#          research/e60_session.sh 512 B C C B
#
# The arm order is given by the caller and is a palindrome, so monotone thermal
# drift cancels to first order across all arms inside one session. Every leg
# runs at the ranked command-buffer geometry and logs to W&B as soon as it
# closes.
#
# A failed leg does not stop the session: an arm that cannot complete the window
# is a result, and the remaining legs still carry the palindrome.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tokens="${1:?usage: e60_session.sh TOKENS ARM [ARM ...]}"
shift

index=0
failures=0
for arm in "$@"; do
  index=$((index + 1))
  tag="e60-t${tokens}-${index}${arm}"
  echo "=== leg ${index}: arm ${arm}, ${tokens} tokens, tag ${tag} ==="
  if research/e60_run_leg.sh "${tag}" "${arm}" "${tokens}"; then
    echo "=== leg ${index} (${tag}) finished ==="
  else
    echo "=== leg ${index} (${tag}) FAILED; continuing the session ===" >&2
    failures=$((failures + 1))
    tail -20 "research/out/${tag}/wrapper.err" >&2 || true
  fi
  research/e60_wandb_log.py --leg "research/out/${tag}" \
    > "research/out/${tag}/wandb.log" 2>&1 \
    || echo "=== leg ${index}: W&B logging failed ===" >&2
done

echo "=== session complete: ${index} legs, ${failures} failed ==="
exit 0
