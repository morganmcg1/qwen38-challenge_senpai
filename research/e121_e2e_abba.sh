#!/usr/bin/env bash
# E121 rung 3: the gated cross-simdgroup chunk-sum share against the unchanged
# base, ABBA-counterbalanced inside one session.
#
#   usage: research/e121_e2e_abba.sh [REPLICATES] [TOKENS] [LABEL] [FIRST]
#
# Each replicate runs base, share, share, base in that order. Both arms have
# mean position 2.5, so monotone thermal drift inside the replicate cancels to
# first order, and the two base legs bracket the replicate and give the session
# null directly.
#
# FIRST numbers the first replicate, so a later session EXTENDS an existing
# estimate instead of restarting it.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

replicates="${1:-3}"
tokens="${2:-512}"
label="${3:-r3}"
first="${4:-1}"

export E121_EXPERIMENT="e121-rung3"

failures=0
for ((rep = first; rep < first + replicates; rep++)); do
  position=0
  for arm in base share share base; do
    position=$((position + 1))
    tag="e121${label}k${rep}p${position}${arm}"
    echo "=== ${tag}: arm=${arm} replicate=${rep} tokens=${tokens} ==="
    research/e121_e2e_leg.sh "${arm}" "${tag}" "${tokens}"
    status=$?
    {
      echo "e121_replicate=${rep}"
      echo "e121_position=${position}"
    } >> "research/out/${tag}/meta.txt"
    if ((status != 0)); then
      echo "e121_e2e_abba: ${tag} exited ${status}" >&2
      failures=$((failures + 1))
    fi
  done
done

echo "e121_e2e_abba: ${failures} failed legs"
exit "${failures}"
