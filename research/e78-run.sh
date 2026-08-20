#!/usr/bin/env bash
# Research-only (qwen38-r1-e78-width-dependent-inner-group-count): drive one E78
# arm through the E42 driver, pinned to THIS base.
#
#   research/e78-run.sh TAG [--legs N]
#
# TAG names the leg group. The twins must already hold that arm's source and be
# COMMITTED, so every timed leg records dirty=0 and names an exact commit.
#
# Unlike E66, this experiment asks for the REAL 40 C cool gate, so
# MLXFAST_LOCAL_COOL_GATE is left at its default of 1 and e42-run.sh records the
# gate state it actually used. Set E78_COOL_GATE=0 to fall back to the permitted
# ungated ABBA protocol; the meta then carries
# cool_gate_passed_real_gate=false verbatim.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
repo_root="${PWD}"

export E42_ROOT="${repo_root}/.mlxfast-private/e78"
export E42_BASE_SHA="${E42_BASE_SHA:-8d938c911df52b6a324f259a55dbaa75e508c822}"
export E42_CURVE_PREFIX="e78"
export MLXFAST_LOCAL_RUN_LOCK_DIR="${MLXFAST_LOCAL_RUN_LOCK_DIR:-/tmp/mlxfast-shared}"
# E78 edits the quantized section, which de-pins the campaign audit's whole-body
# comment waiver for every arm. research/e55_twin_gate.py is arm-independent and
# asserts the same non-comment-identity guard pinned to the known comment
# divergence. research/twin_audit.py stays the promotion gate and is run
# separately in rung 3; the shipped patch carries no comment, so it passes there
# unmodified.
export E42_TWIN_GATE="research/e55_twin_gate.py"
export E42_BINARY_ASSERT="research/e78_binary_assert.sh"
export E42_LEG_ASSERT="research/e78_leg_assert.sh"
export E78_LEG_ASSERT_STATE="${repo_root}/.mlxfast-private/e78/leg-assert"

export MLXFAST_LOCAL_COOL_GATE="${E78_COOL_GATE:-1}"

# --- ranked command-buffer geometry, required on every timed leg -------------
# MLX reads both buffer limits through `static` locals in mlx/utils.h, so they
# must be set before the process starts. On this 48 GiB host the startup policy
# would otherwise select the low-memory profile, which halves the
# referenced-byte and op budgets and clears the allocator cache after warmup --
# a different command-buffer geometry from the ranked box. Same values as E61
# and E66, so this session stays comparable with them.
export DARKBLOOM_STARTUP_MEMORY_PROFILE="${DARKBLOOM_STARTUP_MEMORY_PROFILE:-full}"
export MLX_MAX_MB_PER_BUFFER="${MLX_MAX_MB_PER_BUFFER:-512}"
export MLX_MAX_OPS_PER_BUFFER="${MLX_MAX_OPS_PER_BUFFER:-50}"

tag="${1:?usage: research/e78-run.sh TAG [--legs N]}"

log_dir="${repo_root}/.mlxfast-private/e78/logs"
mkdir -p "${log_dir}"
log="${log_dir}/${tag}.log"

echo "e78-run: MLXFAST_LOCAL_COOL_GATE=${MLXFAST_LOCAL_COOL_GATE}" \
     "DARKBLOOM_STARTUP_MEMORY_PROFILE=${DARKBLOOM_STARTUP_MEMORY_PROFILE}" \
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
