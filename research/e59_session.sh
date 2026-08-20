#!/usr/bin/env bash
# Run E59 cell legs back to back inside one session.
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
# E55 landed on the campaign base after rung 3, and it changed both scored
# files, so a pinned SHA here now fails the leg runner's "no stacked patches"
# guard. Read the live base instead.
export E49_BASE_SHA="${E49_BASE_SHA:-$(git rev-parse origin/senpai/qwen38-mtp-r1)}"
export LEG_ARMS_MODULE="${LEG_ARMS_MODULE:-research/e59_arms.py}"
export LEG_MANIFEST_DIR="${LEG_MANIFEST_DIR:-.mlxfast-private/e59-legs}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-e59-m5-rowblock-r2}"

# Ranked command-buffer geometry, required for every cell leg.
# A cell leg times the kernel inside `swift test`, which never starts the MTP
# worker, so `applyQwenMTPStartupMemoryProfile` never runs and these two names
# reach MLX unmodified. The profile name is exported anyway so that every leg
# carries one identity tuple.
export DARKBLOOM_STARTUP_MEMORY_PROFILE="${DARKBLOOM_STARTUP_MEMORY_PROFILE:-full}"
export MLX_MAX_MB_PER_BUFFER="${MLX_MAX_MB_PER_BUFFER:-512}"
export MLX_MAX_OPS_PER_BUFFER="${MLX_MAX_OPS_PER_BUFFER:-50}"

exec research/e49_session.sh "$@"
