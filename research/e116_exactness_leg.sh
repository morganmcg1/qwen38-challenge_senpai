#!/usr/bin/env bash
# E116 rung 1 -- ONE traced leg whose only product is the exact row evidence.
#
#   usage: research/e116_exactness_leg.sh TAG TOKENS DOSE [FORCE_DRAFTS]
#
# The dose runs after every token, row-ledger entry and draft decision of a
# round is already fixed, so it is bit exact BY CONSTRUCTION. This leg proves
# it anyway: `research/e101_row_digest.py` compares the ordered `mtp-row:`
# lines, whose top-two values are hex float literals, so the comparison is over
# exact bits and not over a rounded print.
#
# FORCE_DRAFTS is the negative control. Pinning the verify width changes which
# rows a round declares, so the digest MUST move. A check that cannot fail is
# not a check.
#
# This is a trace leg, not a timing leg. It exists to produce rows.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tag="${1:?usage: e116_exactness_leg.sh TAG TOKENS DOSE [FORCE_DRAFTS]}"
tokens="${2:?usage: e116_exactness_leg.sh TAG TOKENS DOSE [FORCE_DRAFTS]}"
dose="${3:?usage: e116_exactness_leg.sh TAG TOKENS DOSE [FORCE_DRAFTS]}"
force_drafts="${4:-}"

export MLX_E116_DOSE="${dose}"
if [[ -n "${force_drafts}" ]]; then
  export MLX_E80_FORCE_DRAFTS="${force_drafts}"
fi

research/e79_trace_leg.sh "${tag}" "${tokens}"
status=$?

{
  echo "experiment=e116-measured-transfer-from-kernel-percent-to-leg-seconds"
  echo "leg_kind=e116-exactness-rows"
  echo "MLX_E116_DOSE=${dose}"
  echo "MLX_E80_FORCE_DRAFTS=${force_drafts:-<unset>}"
  echo "mtp_rows=$(
    grep -c '^mtp-row: ' "research/out/${tag}/trace.txt" 2>/dev/null || echo 0)"
  echo "dose_trace_lines=$(
    grep -c '^mtp-trace: e116 dose ' "research/out/${tag}/trace.txt" \
      2>/dev/null || echo 0)"
} >> "research/out/${tag}/meta.txt"

exit "${status}"
