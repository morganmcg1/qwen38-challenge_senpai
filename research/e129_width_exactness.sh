#!/usr/bin/env bash
# E129 — is every routed width bit exact under each dispatch table?
#
#   usage: research/e129_width_exactness.sh [TAG]
#
# THE QUESTION. The one-pass tables send M=6, M=7 and M=8 through `wide<6>`,
# `wide<7>` and `wide<8>` at `rps = 4`, bodies the shipped table never
# instantiates. The change is
# bit-exact by construction, so the only real risks are compiler-level:
# contraction at a new vector width, and the loop-unroll miscompile E65 found
# at NA=5. `research/e129_contraction_check.py` falsifies the first one at the
# AIR layer for zero GPU seconds. This script settles the second, on the GPU.
#
# WHY THIS IS THE RIGHT GATE BEFORE THE 512-TOKEN DIGEST. The reference here is
# MLX's own `quantizedMM`, not a candidate-generated row set, so the comparison
# is not circular. It reaches every routed width and both `USE_TABLE` arms in
# about a minute, against roughly an hour for the full digest. Its positive
# controls (`x_hit`, `meta_hit`, `table_hit`, `restored_diff`) prove on every
# row that the comparison is able to fail.
#
# WHAT IT CANNOT SHOW. One QMV call is not one decode round. The digest still
# owns exact tokens, post-EOS continuation and row-ledger closure.
# A failing leg must not hide the legs after it, so this driver records every
# leg's exit status and reports them together at the end.
set -uo pipefail

cd "$(dirname "$0")/.."
TAG="${1:-e129-width-exactness}"
OUT="research/out/${TAG}"
mkdir -p "${OUT}"

# Four routed cells with four different k-block counts: 10, 12, 34 and 1. The
# k loop is where E65's miscompile lived, so the k-block count is the axis that
# must vary.
SHAPES="${MLXFAST_E129_SHAPES:-mlp.gate_up,fa.o_proj,mlp.down}"

echo "base_sha        $(git rev-parse HEAD)"
echo "dirty           $(git status --porcelain | wc -l | tr -d ' ')"
echo "shapes          ${SHAPES}"
echo "out             ${OUT}"

run_leg () {
    local name="$1" entry="$2" table="$3" grid="${4:-wide}"
    echo
    echo "=== leg ${name}: entry=${entry} table=${table} grid=${grid} ==="
    MLXFAST_RUN_MLX_RUNTIME_TESTS=1 \
    MLX_E120_QMV_ENTRY="${entry}" \
    MLX_E120_QMV_TABLE="${table}" \
    MLX_E120_QMV_GRID="${grid}" \
    MLXFAST_E120_SHAPES="${SHAPES}" \
    MLXFAST_E120_EXACT_OUT="${OUT}/${name}.json" \
    MLX_E120_QMV_PIPELINE_LOG="${OUT}/${name}-pipelines.json" \
        swift test -c release --force-resolved-versions \
            --filter "replicaExactness" 2>&1 \
        | grep -Ev "^\[[0-9]+/[0-9]+\]|warning:|^ *[0-9]+ \||^ *\||^$" \
        | tail -40
    local status="${PIPESTATUS[0]}"
    LEG_STATUS+=("${name}=${status}")
}

LEG_STATUS=()

# The shared switch on the shipped plan is the control: it is the built worker
# today, so a failure here indicts the harness and not the new widths.
run_leg base shared_switch shipped wide
run_leg tier tiered_switch shipped wide
run_leg one6 tiered_switch onepass6 wide
run_leg one67 tiered_switch onepass67 wide
run_leg one678 tiered_switch onepass678 wide
# `tight` drops the x threadgroups that return before doing any work. It must
# leave every row bit identical, on both tables and on the shared switch.
run_leg basetight shared_switch shipped tight
run_leg tiertight tiered_switch shipped tight
run_leg one678tight tiered_switch onepass678 tight

echo
echo "leg exit status: ${LEG_STATUS[*]}"
python3 research/e129_exactness_report.py "${OUT}"
