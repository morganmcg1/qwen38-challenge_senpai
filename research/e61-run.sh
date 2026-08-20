#!/usr/bin/env bash
# Research-only (qwen38-r1-e61-single-weight-stream-qmv-m6): drive one E61 arm
# through the E42 driver, pinned to THIS base.
#
#   research/e61-run.sh TAG [--legs N] [--curve]
#
# TAG names the arm (base, m6, base2). The twins must already hold that arm's
# source and be COMMITTED, so every timed leg records dirty=0 and names an exact
# commit.
#
# Timing is NOT gate-qualified: the permitted local-only ungated protocol applies
# (program.md "Local Measurement"). meta.txt preserves
# cool_gate_passed_real_gate=false, gate_qualified_for_timing=false and
# official_or_ranked_score=false verbatim. base and base2 are the null arm and
# the drift control; arms are ABBA-counterbalanced across the session.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
repo_root="${PWD}"

export E42_ROOT="${repo_root}/.mlxfast-private/e61"
export E42_BASE_SHA="d2139c924c7a7d98ca6026eea63867c2776abbca"
export E42_CURVE_PREFIX="e61"
export MLXFAST_LOCAL_RUN_LOCK_DIR="${MLXFAST_LOCAL_RUN_LOCK_DIR:-/tmp/mlxfast-shared}"
# E61 edits the quantized section, which de-pins the campaign audit's whole-body
# comment waiver for EVERY arm including base. research/e55_twin_gate.py is
# arm-independent: it asserts the same non-comment-identity guard pinned to the
# known comment divergence, so it holds for base, m6 and base2 alike.
# research/twin_audit.py remains the promotion gate.
export E42_TWIN_GATE="research/e55_twin_gate.py"
export E42_BINARY_ASSERT="research/e61_binary_assert.sh"

# --- ranked command-buffer geometry, required on every E61 timed leg ----------
# MLX reads both buffer limits through `static` locals in mlx/utils.h, so they
# must be set before the process starts. On this 48 GiB host the startup policy
# would otherwise select the low-memory profile, which halves the referenced-byte
# and op budgets and clears the allocator cache after warmup -- a different
# command-buffer geometry from the ranked box, which is exactly the confound the
# assignment's lever removes.
# Overridable so research/e61_geometry_proof.sh can starve the ops budget on a
# throwaway leg and show that the value governs the worker. meta.txt records the
# values a leg actually ran under, so a timed leg still names its own geometry.
export DARKBLOOM_STARTUP_MEMORY_PROFILE="${DARKBLOOM_STARTUP_MEMORY_PROFILE:-full}"
export MLX_MAX_MB_PER_BUFFER="${MLX_MAX_MB_PER_BUFFER:-512}"
export MLX_MAX_OPS_PER_BUFFER="${MLX_MAX_OPS_PER_BUFFER:-50}"

tag="${1:?usage: research/e61-run.sh TAG [--legs N] [--curve]}"

log_dir="${repo_root}/.mlxfast-private/e61/logs"
mkdir -p "${log_dir}"
log="${log_dir}/${tag}.log"

echo "e61-run: DARKBLOOM_STARTUP_MEMORY_PROFILE=${DARKBLOOM_STARTUP_MEMORY_PROFILE}" \
     "MLX_MAX_MB_PER_BUFFER=${MLX_MAX_MB_PER_BUFFER}" \
     "MLX_MAX_OPS_PER_BUFFER=${MLX_MAX_OPS_PER_BUFFER}"

research/e42-run.sh "$@" 2>&1 | tee "${log}"
rc="${PIPESTATUS[0]}"

# DEFECT REMOVED: this used to grep the leg log for the worker's low-memory
# startup notice. That check cannot fail. The notice is written to the worker's
# stderr (QwenRuntimeMTPWorker.swift:493-497) but `mtp-timed` builds worker
# options without `forwardsWorkerStderr` (MLXFastCLI/main.swift, default false
# at QwenRuntime.swift), and the drain installs a swallowing emitter
# (QwenRuntimeWorker.swift), so the line never reaches this log under any
# profile. A grep that can only pass is not an instrument, and it reported a
# false `geometry_lever_verified=true` on the E61 smoke leg.
#
# research/e61_geometry_proof.sh carries the two falsifiable session controls
# instead: a `DARKBLOOM_STARTUP_MEMORY_PROFILE=bogus` launch that must crash on
# `RuntimeStartupMemoryPolicy.resolve`'s `preconditionFailure`, and a starved
# `MLX_MAX_OPS_PER_BUFFER` leg that must run measurably slower. This leg only
# records the geometry it exported.
#
# wired_residency_active: Qwen36MTPBlockSession.wireResidentWeightsIfEnabled
# returns early below 96 GiB of physical memory, so resident weights are never
# wired on this host. The ranked M5 wires them. Recorded per leg on the
# advisor's instruction; E62 tests whether it moves a leg.
{
  echo "darkbloom_startup_memory_profile=${DARKBLOOM_STARTUP_MEMORY_PROFILE}"
  echo "mlx_max_mb_per_buffer=${MLX_MAX_MB_PER_BUFFER}"
  echo "mlx_max_ops_per_buffer=${MLX_MAX_OPS_PER_BUFFER}"
  echo "geometry_lever_verified_by=research/e61_geometry_proof.sh"
  echo "wired_residency_active=false"
  echo "run_log=${log}"
} >> "${E42_ROOT}/runs/${tag}/meta.txt" 2>/dev/null || true

exit "${rc}"
