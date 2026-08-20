#!/usr/bin/env bash
# Research-only (qwen38-r1-e66-composition-certification): drive one E66 arm
# through the E42 driver, pinned to THIS base.
#
#   research/e66-run.sh TAG [--legs N] [--curve]
#
# TAG names the leg group (warm, a1, b1, c1, c2, b2, a2). The twins must already
# hold that arm's source and be COMMITTED, so every timed leg records dirty=0
# and names an exact commit.
#
# Timing is NOT gate-qualified: the permitted local-only ungated protocol applies
# (program.md "Local Measurement"). meta.txt preserves
# cool_gate_passed_real_gate=false, gate_qualified_for_timing=false and
# official_or_ranked_score=false verbatim. Arms are position balanced across the
# session, so all three have the same mean leg position.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
repo_root="${PWD}"

export E42_ROOT="${repo_root}/.mlxfast-private/e66"
export E42_BASE_SHA="${E42_BASE_SHA:-45b4f3a800f879e3579ca27ef0b1c0ef40e4473d}"
export E42_CURVE_PREFIX="e66"
export MLXFAST_LOCAL_RUN_LOCK_DIR="${MLXFAST_LOCAL_RUN_LOCK_DIR:-/tmp/mlxfast-shared}"
# E66 edits the quantized section, which de-pins the campaign audit's whole-body
# comment waiver for every arm including the base arm. research/e55_twin_gate.py
# is arm-independent: it asserts the same non-comment-identity guard pinned to
# the known comment divergence. research/twin_audit.py remains the promotion
# gate and runs in rung 5.
export E42_TWIN_GATE="research/e55_twin_gate.py"
export E42_BINARY_ASSERT="research/e66_binary_assert.sh"
export E42_LEG_ASSERT="research/e66_leg_assert.sh"

# --- ranked command-buffer geometry, required on every E66 timed leg ----------
# MLX reads both buffer limits through `static` locals in mlx/utils.h, so they
# must be set before the process starts. On this 48 GiB host the startup policy
# would otherwise select the low-memory profile, which halves the referenced-byte
# and op budgets and clears the allocator cache after warmup -- a different
# command-buffer geometry from the ranked box. E61 verified the lever governs the
# worker with research/e61_geometry_proof.sh; E62 then measured the per-commit
# cost directly. Same values, so the E66 session stays comparable with E61.
export DARKBLOOM_STARTUP_MEMORY_PROFILE="${DARKBLOOM_STARTUP_MEMORY_PROFILE:-full}"
export MLX_MAX_MB_PER_BUFFER="${MLX_MAX_MB_PER_BUFFER:-512}"
export MLX_MAX_OPS_PER_BUFFER="${MLX_MAX_OPS_PER_BUFFER:-50}"

tag="${1:?usage: research/e66-run.sh TAG [--legs N] [--curve]}"

log_dir="${repo_root}/.mlxfast-private/e66/logs"
mkdir -p "${log_dir}"
log="${log_dir}/${tag}.log"

echo "e66-run: DARKBLOOM_STARTUP_MEMORY_PROFILE=${DARKBLOOM_STARTUP_MEMORY_PROFILE}" \
     "MLX_MAX_MB_PER_BUFFER=${MLX_MAX_MB_PER_BUFFER}" \
     "MLX_MAX_OPS_PER_BUFFER=${MLX_MAX_OPS_PER_BUFFER}"

research/e42-run.sh "$@" 2>&1 | tee "${log}"
rc="${PIPESTATUS[0]}"

# wired_residency_active: Qwen36MTPBlockSession.wireResidentWeightsIfEnabled
# returns early below 96 GiB of physical memory, so resident weights are never
# wired on this 48 GiB host. The ranked M5 wires them.
{
  echo "darkbloom_startup_memory_profile=${DARKBLOOM_STARTUP_MEMORY_PROFILE}"
  echo "mlx_max_mb_per_buffer=${MLX_MAX_MB_PER_BUFFER}"
  echo "mlx_max_ops_per_buffer=${MLX_MAX_OPS_PER_BUFFER}"
  echo "geometry_lever_verified_by=research/e61_geometry_proof.sh"
  echo "wired_residency_active=false"
  echo "run_log=${log}"
} >> "${E42_ROOT}/runs/${tag}/meta.txt" 2>/dev/null || true

exit "${rc}"
