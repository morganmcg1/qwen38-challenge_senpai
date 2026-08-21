#!/usr/bin/env bash
# E110 rung 2: `xv4` against the unchanged base, ABBA-counterbalanced inside
# one session.
#
#   usage: research/e110_rung2_abba.sh [REPLICATES] [TOKENS] [LABEL]
#
# Each replicate runs base, xv4, xv4, base in that order. The arm effect is
# therefore orthogonal to monotone thermal drift to first order, and the two
# same-arm pairs give the session null. Every leg rebuilds the worker and
# witnesses its own arm in the JIT string before and after it is timed.
#
# Rung 2 was run as `research/e110_rung2_abba.sh 2 512 r2`.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

replicates="${1:-2}"
tokens="${2:-512}"
label="${3:-r2}"

export E110_EXPERIMENT="e110-rung2"

failures=0
for ((rep = 1; rep <= replicates; rep++)); do
  position=0
  for arm in base xv4 xv4 base; do
    position=$((position + 1))
    tag="e110${label}k${rep}p${position}${arm}"
    echo "=== ${tag}: arm=${arm} replicate=${rep} tokens=${tokens} ==="
    research/e110_rung2_leg.sh "${arm}" "${tag}" "${tokens}"
    status=$?
    {
      echo "e110_replicate=${rep}"
      echo "e110_position=${position}"
    } >> "research/out/${tag}/meta.txt"
    if ((status != 0)); then
      echo "e110_rung2_abba: ${tag} exited ${status}" >&2
      failures=$((failures + 1))
    fi
  done
done

echo "e110_rung2_abba: ${failures} failed legs"
exit "${failures}"
