#!/usr/bin/env bash
# E110 rung 2 exactness: prove the `xv4` vector activation load changes no bit
# of the scored decode.
#
#   usage: research/e110_rung2_exact.sh [TOKENS]
#
# Two traced in-situ legs, base then xv4, at the same token window. `mtp-row:`
# carries the target row position, the top-two token ids, and the top-two
# values as HEX float literals, so a digest over the ordered row lines compares
# exact bits and not a rounded print. Three claims come out of it:
#
#   1. arm-relative bit equality  base rows == xv4 rows;
#   2. campaign continuity        both digests equal the digests the merged
#                                 base recorded in research/e101-results.md;
#   3. contract closure           all_tokens_matched and
#                                 residual_divergence_count from score.json.
#
# The digest tool is proven able to fail by a NEGATIVE CONTROL: one row line of
# the base stream is perturbed and re-digested. A gate that cannot fail proves
# nothing about the two passes above.
#
# Untimed evidence. These legs run with the cool gate off and are never a
# gated, official, or ranked score.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tokens="${1:-512}"

export E110_EXPERIMENT="e110-rung2-exact"
export E110_TRACE=1

failures=0
for arm in base xv4; do
  tag="e110r2x${tokens}${arm}"
  echo "=== ${tag}: arm=${arm} tokens=${tokens} (traced) ==="
  research/e110_rung2_leg.sh "${arm}" "${tag}" "${tokens}"
  status=$?
  if ((status != 0)); then
    echo "e110_rung2_exact: ${tag} exited ${status}" >&2
    failures=$((failures + 1))
  fi
done

((failures == 0)) || exit "${failures}"

python3 research/e110_rung2_exact_report.py \
  --base-tag "e110r2x${tokens}base" \
  --cand-tag "e110r2x${tokens}xv4" \
  --tokens "${tokens}" \
  --output research/out/e110/rung2-exactness.json
exit $?
