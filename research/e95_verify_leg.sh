#!/usr/bin/env bash
# E95 rung 2 -- one target_verify census leg at a pinned verify width.
#
#   usage: research/e95_verify_leg.sh TAG DRAFTS TOKENS
#
# The verify width is M = DRAFTS + 1, because the target checks the pending
# primary token plus every draft in the same forward pass.
#
# `research/e93_gputime_leg.sh` owns the E80 Metal command-buffer clock and
# `research/e85_census_leg.sh` owns the lock, the head, the forced width and the
# census environment. This wrapper only adds the entry and exit GPU temperature,
# because a census leg reports GPU nanoseconds and those move with temperature
# even though the census lock makes host wall clock meaningless.
#
# A census leg is NEVER a gate-qualified timing leg: MLXFAST_LOCAL_COOL_GATE=0
# and the census swizzle serialises every dispatch.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tag="${1:?usage: e95_verify_leg.sh TAG DRAFTS TOKENS}"
drafts="${2:?usage: e95_verify_leg.sh TAG DRAFTS TOKENS}"
tokens="${3:?usage: e95_verify_leg.sh TAG DRAFTS TOKENS}"

gpu_temp() {
  local macmon
  for macmon in "${MLXFAST_MACMON_BIN:-}" "${HOME}/bin/macmon" \
                /opt/homebrew/bin/macmon /usr/local/bin/macmon; do
    [[ -n "${macmon}" && -x "${macmon}" ]] || continue
    "${macmon}" pipe -s1 2>/dev/null | jq -r '.temp.gpu_temp_avg // empty'
    return 0
  done
  echo ""
}

entry_c="$(gpu_temp)"
research/e93_gputime_leg.sh "${tag}" "${drafts}" "${tokens}"
status=$?
exit_c="$(gpu_temp)"

{
  echo "experiment=e95-target-verify-census"
  echo "verify_width_m=$((drafts + 1))"
  echo "gpu_temp_entry_c=${entry_c}"
  echo "gpu_temp_exit_c=${exit_c}"
  echo "sandbox=off"
} >> "research/out/${tag}/meta.txt"

exit "${status}"
