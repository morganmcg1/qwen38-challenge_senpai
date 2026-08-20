#!/usr/bin/env bash
# Run several E61 legs back to back inside one session.
#
#   research/e61_session.sh ARM:TAG [ARM:TAG ...] [--widths L --reps N --inner N]
#
# E61 reuses the E49 leg runner and only repoints the arm module, the manifest
# directory and the growth base. The wrapper exists because run_job takes an
# argv list with no environment field, so the environment must be set here.
#
# These are lone-group microbenchmark legs, not whole-leg decode timing. They
# deliberately do NOT export the E61 geometry lever
# (DARKBLOOM_STARTUP_MEMORY_PROFILE / MLX_MAX_MB_PER_BUFFER /
# MLX_MAX_OPS_PER_BUFFER), because the rung 1 ladder has to stay comparable
# with the E54 NA=2..5 anchors that were measured without it. The whole-leg
# arms in research/e61-run.sh do export it.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

export MLXFAST_LOCAL_RUN_LOCK_DIR="${MLXFAST_LOCAL_RUN_LOCK_DIR:-/tmp/mlxfast-shared}"
export E49_BASE_SHA="${E49_BASE_SHA:-d2139c924c7a7d98ca6026eea63867c2776abbca}"
export LEG_ARMS_MODULE="${LEG_ARMS_MODULE:-research/e61_arms.py}"
export LEG_MANIFEST_DIR="${LEG_MANIFEST_DIR:-.mlxfast-private/e61-legs}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-e61-single-stream-qmv-m6}"

exec research/e49_session.sh "$@"
