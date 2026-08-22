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
# exact build at 411.86 us isolated, which is +0.23 % against E107's 410.93 us,
# so the two agree; the rung 1 number is still the one used, because it is the
# rate of the binary that ran.
#
# BOTH ARMS ARE TRACED, ON PURPOSE. `MLX_QWEN_MTP_TRACE=1` makes the worker
# write one `mtp-trace: e116 dose` line per round, which is the in-process
# witness that the dose really alternated in the leg that was timed rather than
# in a census leg run separately. Tracing costs a few hundred bytes of host
# work per round against a round near 160,000 us, and BOTH arms pay it, so it
# cancels in the contrast. The round frame this session reports is therefore a
# TRACED round frame and must not be compared with an untraced one. The rung 3
# ladder is deliberately untraced, because its endpoint is an absolute leg
# time and has no second arm to cancel against.
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
  echo "    --json research/e116-artifacts/rung2-absorption.json"
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
  --json "research/e116-artifacts/rung2-absorption.json"
