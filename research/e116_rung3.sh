#!/usr/bin/env bash
# E116 rung 3 -- beta: how much of a round-time change reaches leg seconds?
#
#   usage: research/e116_rung3.sh [DOSE_UNIT_US ALPHA]
#
# FOUR ARMS, a dose ladder with NO alternation, so every round of a leg carries
# the same dose and the leg endpoint moves:
#
#   k0   MLX_E116_DOSE=0      k4   MLX_E116_DOSE=4
#   k8   MLX_E116_DOSE=8      k12  MLX_E116_DOSE=12
#
# All four arms are armed, so all four allocate and hold the same 100.27 MB.
# Only the dispatch count differs.
#
# WHY A LADDER AND NOT ONE DOSE. A single dose point gives one difference and
# no way to tell a real response from an arm offset. Four points give a slope
# with a residual, and the residual is the falsification: if the leg response
# is not linear in the dose, the transfer coefficient this experiment reports
# does not exist and the ladder says so. k = 0 is on the ladder, so the fitted
# line must pass through the null arm as well.
#
# THE KILL RULE. The assignment's stop condition is that the k = 12 dose must
# resolve on the round response at 2 sigma. k = 12 injects about 12 dose units
# per round; rung 1 measures what that is worth in microseconds. If the k = 12
# arm cannot be separated from k = 0, the ladder is under-powered and the
# experiment reports that, together with the observed noise, instead of a
# number the data cannot support.
#
# BLOCKS. `E109_BLOCKS=3` runs three estimate blocks plus one conditioning
# block: 16 legs, 12 of which enter the estimate, three per arm. Positions are
# counterbalanced by rotation plus mirror inside `e109_ab_session.sh`.
#
# THERMAL. Ungated, counterbalanced within one session, entry and exit GPU
# temperature per leg, honesty flags preserved verbatim. Not gate qualified and
# not a ranked score.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

dose_unit_us="${1:-}"
alpha="${2:-}"
tag="${E116_RUNG3_TAG:-e116r3-ladder}"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "e116_rung3: worktree is dirty; refusing to measure over uncommitted" \
       "work" >&2
  exit 1
fi

E109_BLOCKS="${E109_BLOCKS:-3}" E109_TOKENS="${E109_TOKENS:-512}" \
  research/e109_ab_session.sh "${tag}" \
    "k0=MLX_E116_DOSE=0" \
    "k4=MLX_E116_DOSE=4" \
    "k8=MLX_E116_DOSE=8" \
    "k12=MLX_E116_DOSE=12"
status=$?

if [[ -z "${dose_unit_us}" || -z "${alpha}" ]]; then
  echo
  echo "e116_rung3: legs done; rerun the reducer with rung 1 and rung 2:"
  echo "  python3 research/e116_transfer_report.py \\"
  echo "    research/out/${tag}/b[123]-* --dose-unit-us US --alpha A \\"
  echo "    --json research/e116-artifacts/rung3-transfer.json"
  exit "${status}"
fi

legs=()
for dir in "research/out/${tag}"/b*-*; do
  [[ "$(basename "${dir}")" == b0-* ]] && continue
  legs+=("${dir}")
done

python3 research/e116_transfer_report.py "${legs[@]}" \
  --dose-unit-us "${dose_unit_us}" --alpha "${alpha}" \
  --json "research/e116-artifacts/rung3-transfer.json"
