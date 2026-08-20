#!/usr/bin/env bash
# E75 rung B: measure the per-width cost curve of the CROWN dispatch table.
#
#   research/e75_rungB_session.sh ARM:TAG [ARM:TAG ...] [--widths L --reps N --inner N]
#
# This is the E68 rung-1 instrument with one thing changed: the arm module. The
# widths, reps and inner counts are passed by the caller and must stay at
# `--widths 1,2,3,4,5,6,7,8,9,10 --reps 21 --inner 10 --skip-stock`, which is
# what E68 rung 1 ran, so the two curves are comparable cell by cell.
#
# The `ours` legs are the in-session control. They are not decoration: they are
# the only way to tell a real crown-table effect from between-session drift,
# and they also replay E68 rung 1 on the current base, which nothing else in
# E75 does.
#
# `run_job` takes an argv list with no environment field, so the environment is
# set here, exactly as `research/e68_rung1_session.sh` does for E68.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

export LEG_ARMS_MODULE="${LEG_ARMS_MODULE:-research/e75_arms.py}"
export LEG_MANIFEST_DIR="${LEG_MANIFEST_DIR:-.mlxfast-private/e75-legs}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-e75-bank-pbfit-and-price-it-on-the-crown-table}"

# E75 is pinned to the assignment base. `e59_session.sh` reads
# `origin/senpai/qwen38-mtp-r1` live, and that ref moved to dd60d0b while this
# assignment was running, which would fail the leg runner's no-stacked-patches
# guard against a base this experiment never measured.
export E49_BASE_SHA="${E49_BASE_SHA:-432eba00db0b194731a68202059ce5bfb158c1e8}"

exec research/e59_session.sh "$@"
