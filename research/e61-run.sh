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
export DARKBLOOM_STARTUP_MEMORY_PROFILE=full
export MLX_MAX_MB_PER_BUFFER=512
export MLX_MAX_OPS_PER_BUFFER=50

tag="${1:?usage: research/e61-run.sh TAG [--legs N] [--curve]}"

log_dir="${repo_root}/.mlxfast-private/e61/logs"
mkdir -p "${log_dir}"
log="${log_dir}/${tag}.log"

echo "e61-run: DARKBLOOM_STARTUP_MEMORY_PROFILE=${DARKBLOOM_STARTUP_MEMORY_PROFILE}" \
     "MLX_MAX_MB_PER_BUFFER=${MLX_MAX_MB_PER_BUFFER}" \
     "MLX_MAX_OPS_PER_BUFFER=${MLX_MAX_OPS_PER_BUFFER}"

research/e42-run.sh "$@" 2>&1 | tee "${log}"
rc="${PIPESTATUS[0]}"

# The lever is only worth setting if it is also verified. The worker announces a
# low-memory startup on stderr, and QwenRuntimeWorker forwards worker stderr with
# the `mlxfast-worker: ` prefix, so the announcement reaches this log. Fail the
# leg rather than report a number measured under the wrong geometry.
low_memory="Sources/MLXFastTrustedHarness/QwenRuntimeMTPWorker.swift:495"
if grep -q "mlxfast-worker: low-memory startup profile engaged" "${log}"; then
  echo "e61-run: ${tag} ran under the LOW-MEMORY startup profile (${low_memory});" \
       "the geometry lever did not take. Failing the leg." >&2
  echo "geometry_lever_verified=false" >> "${E42_ROOT}/runs/${tag}/meta.txt" 2>/dev/null || true
  exit 6
fi
{
  echo "geometry_lever_verified=true"
  echo "darkbloom_startup_memory_profile=${DARKBLOOM_STARTUP_MEMORY_PROFILE}"
  echo "mlx_max_mb_per_buffer=${MLX_MAX_MB_PER_BUFFER}"
  echo "mlx_max_ops_per_buffer=${MLX_MAX_OPS_PER_BUFFER}"
  echo "run_log=${log}"
} >> "${E42_ROOT}/runs/${tag}/meta.txt" 2>/dev/null || true

exit "${rc}"
