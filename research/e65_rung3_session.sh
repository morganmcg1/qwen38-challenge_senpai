#!/usr/bin/env bash
# E65 rung 3: leg-level timing of base vs r1warm, counterbalanced.
#
# One declared warm-up leg is discarded, then ABBA BAAB. The palindrome pairs
# make monotone thermal drift cancel to first order, and each arm holds the
# same set of leg positions. Legs run WITHOUT tracing: rung 0 and rung 1 scores
# are traced and are not comparable with these.
#
# This arm is underpowered on purpose-honesty grounds. The mechanism is worth
# about 0.15 % of a leg, and the largest same-arm score spread measured in
# rung 0 was 0.36 %. The round-latency census is the primary instrument; this
# session only checks that the leg-level sign is not contradicted.
set -uo pipefail
cd "$(dirname "$0")/.."

session="${1:-r3}"
status=0
i=0
run() {
  local arm="$1"
  i=$((i + 1))
  local tag
  tag=$(printf "e65-%s-%02d-%s" "${session}" "${i}" "${arm}")
  echo "=== ${tag}: ${arm} ==="
  if research/e65_run_leg.sh "${tag}" "${arm}" 512 --no-trace \
      --label "${session}-pos${i}-${arm}"; then
    echo "=== ${tag} finished ==="
  else
    echo "=== ${tag} FAILED; continuing ==="
    status=1
  fi
}

# Position 1 is a declared discard: it absorbs the cold page cache and the
# coldest GPU entry temperature of the session.
run base

for arm in base r1warm r1warm base r1warm base base r1warm; do
  run "${arm}"
done

exit "${status}"
