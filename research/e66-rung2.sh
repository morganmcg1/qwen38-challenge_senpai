#!/usr/bin/env bash
# E66 rung 2: exactness of the composed QMV surface on the merged base.
#
#   research/e66-rung2.sh
#
# Four untimed 512-token depth-8 ledger legs: the three timed arms plus the
# `c_lane_perturb` positive control. All four replay the same golden, which is
# arm A leg 1 of the rung 3 timed session, so a ledger difference between arms
# is a difference in the kernel and not in the prompt or the schedule.
#
# The base pin is the advisor branch that carries `t55` and `t6` together. This
# branch's candidate surface is byte-identical to it, so the guard asserts that
# the arms are applied to an unpatched tree.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

merged_base="b7b6589a9b319c1176c737b5698b915740df0937"
export E66_BASE_SHA="${E66_BASE_SHA:-${merged_base}}"
export E42_BASE_SHA="${E42_BASE_SHA:-${merged_base}}"
export E66_LEG_RUNNER="research/e66_ledger_leg.sh"
# The ledger legs are untimed replays, so there is no leg metric to log.
export E66_WANDB=0

exec research/e66_whole_leg_session.sh \
  a_neither:a:1 \
  b_t6:b:1 \
  c_t55_t6:c:1 \
  c_lane_perturb:c_perturb:1
