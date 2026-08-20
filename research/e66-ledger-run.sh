#!/usr/bin/env bash
# Rung 2: capture the E66 arms' WIDE-dispatch per-row top-2 evidence, pinned to
# THIS experiment's run tree.
#
#   research/e66-ledger-run.sh ARM
#
# `mtp-verify --golden` (no `--generate`) is the only verb that runs the
# candidate session with `retainLedger: true`, so it is the only producer of
# `row_ledger`, which carries `top2_tokens` and `top2_logits` for every row the
# wide multi-row dispatch actually evaluated. The benchmark pipeline never
# invokes that mode, so without this step there is no DIRECT bitwise reading of
# the M=5 and M=6 rows this experiment changes.
#
# The golden is pinned to arm A's leg 1 from the rung 3 session, so all arms are
# judged against one byte-identical reference and any ledger difference is
# attributable to the wide dispatch rather than to the reference.
#
# Not timed and produces no score, so it needs no thermal gate and is not a
# replicate of the timed measurement.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
repo_root="${PWD}"

export E55_LEDGER_ROOT="${repo_root}/.mlxfast-private/e66"
export E55_LEDGER_GOLDEN="${E66_LEDGER_GOLDEN:-${E55_LEDGER_ROOT}/runs/a1/reports/leg-1/02-mtp-verify-output.json}"
export E55_LEDGER_BINARY_ASSERT="research/e66_binary_assert.sh"
export E55_LEDGER_TOKENS="${E66_LEDGER_TOKENS:-512}"
export E55_LEDGER_DEPTH="${E66_LEDGER_DEPTH:-8}"
export MLXFAST_LOCAL_RUN_LOCK_DIR="${MLXFAST_LOCAL_RUN_LOCK_DIR:-/tmp/mlxfast-shared}"

# The ledger run is not timed, but it must still execute the same command-buffer
# geometry as the arms whose exactness it certifies: a different geometry can
# change evaluation boundaries and therefore which rows are batched together.
export DARKBLOOM_STARTUP_MEMORY_PROFILE=full
export MLX_MAX_MB_PER_BUFFER=512
export MLX_MAX_OPS_PER_BUFFER=50

exec research/e55-ledger-run.sh "$@"
