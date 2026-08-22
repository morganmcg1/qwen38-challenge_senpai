#!/usr/bin/env bash
# E116 rung 2 -- alpha: how much of an injected GPU dose reaches the round wall?
#
#   usage: research/e116_rung2.sh [DOSE_UNIT_US]
#
# TWO ARMS, both alternating round by round inside every leg:
#
#   null  MLX_E116_DOSE=0   MLX_E116_DOSE_ALTERNATE=1
#   k4    MLX_E116_DOSE=4   MLX_E116_DOSE_ALTERNATE=1
#
# The `null` arm is armed exactly like the dosed arm -- the environment
# variable is PRESENT, so the 100.27 MB dose weight is allocated and resident
# in both arms -- and it applies zero units. It therefore carries the identical
# memory footprint, the identical allocation work and the identical
# hypothetical odd/even round assignment, and differs only in whether four
# quantized matrix-vector dispatches run. Any period-2 structure a decode round
# has on its own appears in BOTH arms and cancels in the difference. Without
# this arm the alternating estimator would report the round's own parity as
# absorption.
#
# WHY ALTERNATION AND NOT AN ARM CONTRAST. E109 v1 measured the whole leg
# against another leg and 97.9 % of its pair variance was a per-leg offset.
# Alternating the dose inside one leg puts the contrast between neighbouring
# rounds, where the offset is shared. The dosed and undosed members of a pair
# are also matched on realised verify width, so a change in draft acceptance
# cannot leak into the difference.
#
# DOSE_UNIT_US is the measured M=1 rate for one dose unit and it must come from
# rung 1, not from the E107 ledger row. Rung 1 measured this exact cell in this
# exact build and disagreed with the ledger's 410.93 us by more than 10 %, so
# the ledger number is not usable as the denominator here.
#
# THERMAL. Ungated, counterbalanced within one session, entry and exit GPU
# temperature per leg, honesty flags preserved verbatim. Not gate qualified and
# not a ranked score.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

dose_unit_us="${1:-}"
tag="${E116_RUNG2_TAG:-e116r2-absorption}"
trace='MLX_QWEN_MTP_TRACE=1,MLX_QWEN_MTP_TRACE_PATH=@LEG@/trace.txt'

if [[ -n "$(git status --porcelain)" ]]; then
  echo "e116_rung2: worktree is dirty; refusing to measure over uncommitted" \
       "work" >&2
  exit 1
fi

E109_BLOCKS="${E109_BLOCKS:-2}" E109_TOKENS="${E109_TOKENS:-512}" \
  research/e109_ab_session.sh "${tag}" \
    "null=MLX_E116_DOSE=0,MLX_E116_DOSE_ALTERNATE=1,${trace}" \
    "k4=MLX_E116_DOSE=4,MLX_E116_DOSE_ALTERNATE=1,${trace}"
status=$?

if [[ -z "${dose_unit_us}" ]]; then
  echo
  echo "e116_rung2: legs done; rerun the reducer with the rung 1 dose unit:"
  echo "  python3 research/e116_absorption_report.py \\"
  echo "    research/out/${tag}/b[12]-* --dose-unit-us US \\"
  echo "    --json research/out/e116-artifacts/rung2-absorption.json"
  exit "${status}"
fi

# Block 0 conditions the machine and is excluded from every estimate.
legs=()
for dir in "research/out/${tag}"/b*-*; do
  [[ "$(basename "${dir}")" == b0-* ]] && continue
  legs+=("${dir}")
done

python3 research/e116_absorption_report.py "${legs[@]}" \
  --dose-unit-us "${dose_unit_us}" \
  --json "research/out/e116-artifacts/rung2-absorption.json"
