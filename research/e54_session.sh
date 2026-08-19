#!/usr/bin/env bash
# Run several E54 legs back to back inside one session.
#
#   research/e54_session.sh ARM:TAG [ARM:TAG ...] [--widths L --reps N --inner N]
#
# E54 reuses the E49 leg runner and only repoints the arm module, the manifest
# directory and the growth base.  The wrapper exists because run_job takes an
# argv list with no environment field, so the environment must be set here.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

export MLXFAST_LOCAL_RUN_LOCK_DIR="${MLXFAST_LOCAL_RUN_LOCK_DIR:-/tmp/mlxfast-shared}"
export E49_BASE_SHA="${E49_BASE_SHA:-a35bb006fd47785dc916241df63ec8780bda8e5c}"
export LEG_ARMS_MODULE="${LEG_ARMS_MODULE:-research/e54_arms.py}"
export LEG_MANIFEST_DIR="${LEG_MANIFEST_DIR:-.mlxfast-private/e54-legs}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-e54-lone-vs-sibling-na5}"

exec research/e49_session.sh "$@"
