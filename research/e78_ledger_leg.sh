#!/usr/bin/env bash
# Adapter so research/e66_whole_leg_session.sh can drive the E78 rung 1 ledger
# legs.
#
#   research/e78_ledger_leg.sh TAG [--legs N]
#
# The session driver owns the arm patch, the transient commit and the unwind. It
# calls its leg runner as `RUNNER TAG --legs N`. The ledger runner takes only an
# arm name and always runs exactly one untimed pass, so this adapter drops the
# leg count and forwards the tag.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
repo_root="${PWD}"

tag="${1:?usage: research/e78_ledger_leg.sh TAG [--legs N]}"

export E55_LEDGER_ROOT="${repo_root}/.mlxfast-private/e78"
export E55_LEDGER_GOLDEN="${E78_LEDGER_GOLDEN:-${E55_LEDGER_ROOT}/runs/golden/reports/leg-1/02-mtp-verify-output.json}"
export E55_LEDGER_BINARY_ASSERT="research/e78_binary_assert.sh"
export E55_LEDGER_TOKENS="${E78_LEDGER_TOKENS:-512}"
export E55_LEDGER_DEPTH="${E78_LEDGER_DEPTH:-8}"
export MLXFAST_LOCAL_RUN_LOCK_DIR="${MLXFAST_LOCAL_RUN_LOCK_DIR:-/tmp/mlxfast-shared}"

# Not timed, but it must execute the same command-buffer geometry as the arms
# whose exactness it certifies: a different geometry can change evaluation
# boundaries and therefore which rows are batched together.
export DARKBLOOM_STARTUP_MEMORY_PROFILE=full
export MLX_MAX_MB_PER_BUFFER=512
export MLX_MAX_OPS_PER_BUFFER=50

exec research/e55-ledger-run.sh "${tag}"
