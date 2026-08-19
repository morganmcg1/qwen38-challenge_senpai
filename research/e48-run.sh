#!/usr/bin/env bash
# Research-only (qwen38-r1-e48-score-weighted-qmv-and-uniform-sign): drive one
# E48 arm through the E42 driver, pinned to THIS base.
#
#   research/e48-run.sh TAG [--curve] [--legs N]
#
# TAG names the arm (base, ulo, uhi, base2). The twins must already hold that
# arm's source and be COMMITTED. Timing is NOT gate-qualified: the permitted
# local-only ungated protocol applies, and the base brackets at session start
# and end are the drift control.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export E42_ROOT="${PWD}/.mlxfast-private/e48"
export E42_BASE_SHA="fb0a09d3912477d94ed631bdb90fd04172d7b4cf"
export E42_CURVE_PREFIX="e48"
export MLXFAST_LOCAL_RUN_LOCK_DIR="${MLXFAST_LOCAL_RUN_LOCK_DIR:-/tmp/mlxfast-shared}"

exec research/e42-run.sh "$@"
