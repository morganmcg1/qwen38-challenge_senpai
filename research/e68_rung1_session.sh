#!/usr/bin/env bash
# E68 rung 1: measure the per-width cost of the CURRENT SHIPPED dispatch table.
#
#   research/e68_rung1_session.sh ARM:TAG [ARM:TAG ...] [--widths L --reps N --inner N]
#
# The advisor's marginal-cost table for the post-`t55`+`t6` table is modelled,
# not measured: it is `C(M) = C1(NA_0) + 0.80 * sum_{g>0} C1(NA_g)` over the
# E61 single-stream ladder (ledger 199(B), 200(E)). Rung 1 replaces every
# modelled entry with a measured one, because the whole E68 brief rests on the
# claim that the 4->5 and 5->6 steps INVERTED.
#
# One leg times every width round-robin inside one process, so the curve is
# internally matched by construction; the replicates exist to bound the
# between-leg null, not to build the curve.
#
# `run_job` takes an argv list with no environment field, so the environment is
# set here, exactly as `research/e59_session.sh` does for E59.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

export LEG_ARMS_MODULE="${LEG_ARMS_MODULE:-research/e68_arms.py}"
export LEG_MANIFEST_DIR="${LEG_MANIFEST_DIR:-.mlxfast-private/e68-legs}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-e68-schedule-against-the-new-cost-curve}"

exec research/e59_session.sh "$@"
