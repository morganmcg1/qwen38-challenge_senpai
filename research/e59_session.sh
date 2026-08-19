#!/usr/bin/env bash
# Run the E59 rung 3 isolated-cell legs back to back inside one session.
#
#   research/e59_session.sh ARM:TAG [ARM:TAG ...] [--widths L --reps N --inner N]
#
# E59 reuses the E49 leg runner and only repoints the arm module, the manifest
# directory and the growth base. The wrapper exists because run_job takes an
# argv list with no environment field, so the environment must be set here.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

export MLXFAST_LOCAL_RUN_LOCK_DIR="${MLXFAST_LOCAL_RUN_LOCK_DIR:-/tmp/mlxfast-shared}"
export E49_BASE_SHA="${E49_BASE_SHA:-989596895b7c8f889443dac0c87e024a428e6e9e}"
export LEG_ARMS_MODULE="${LEG_ARMS_MODULE:-research/e59_arms.py}"
export LEG_MANIFEST_DIR="${LEG_MANIFEST_DIR:-.mlxfast-private/e59-legs}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-e59-m5-rowblock-r2}"

exec research/e49_session.sh "$@"
