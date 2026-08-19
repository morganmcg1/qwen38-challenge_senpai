#!/usr/bin/env bash
# Research-only (qwen38-r1-e55-compose-m9-two-stream-on-shipped-table): drive one
# E55 arm through the E42/E48 driver, pinned to THIS base.
#
#   research/e55-run.sh TAG [--legs N]
#
# TAG names the arm (base, m9two, base2). The twins must already hold that arm's
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

export E42_ROOT="${PWD}/.mlxfast-private/e55"
export E42_BASE_SHA="a35bb006fd47785dc916241df63ec8780bda8e5c"
export E42_CURVE_PREFIX="e55"
export MLXFAST_LOCAL_RUN_LOCK_DIR="${MLXFAST_LOCAL_RUN_LOCK_DIR:-/tmp/mlxfast-shared}"
# E55 edits the quantized section, which de-pins the campaign audit's whole-body
# comment waiver for EVERY arm including base. research/e55_twin_gate.py asserts
# the same non-comment-identity guard pinned to the divergence, so it holds for
# base, m9two and base2 alike. research/twin_audit.py remains the promotion gate.
export E42_TWIN_GATE="research/e55_twin_gate.py"

exec research/e42-run.sh "$@"
