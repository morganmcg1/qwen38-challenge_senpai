#!/usr/bin/env bash
# One E75 rung D session: the 2x2 over {kernel table} x {depth price}.
#
#   research/e75_rungD_session.sh CELL:TAG [CELL:TAG ...] [--tokens N] [--hot]
#
# CELL is `TABLE-PRICE`, one of the four directories left by
# research/e75_rungD_prebuild.sh. Every leg installs that cell's prebuilt
# binaries rather than rebuilding, so the kernel table can change between
# consecutive legs and the leg order is free.
#
# `run_job` takes an argv list with no environment field, so the environment is
# set here, exactly as research/e75_session.sh does for a single-table session.
# The difference is that this wrapper must NOT pin one table: each leg carries
# its own, applied and unwound by the leg runner.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

store="${E75_CELL_STORE:-${repo_root}/.mlxfast-private/e75-rungD/cells}"

legs=()
passthrough=()
while (($#)); do
  case "$1" in
    --tokens) passthrough+=("$1" "$2"); shift 2 ;;
    --hot|--warmup) passthrough+=("$1"); shift ;;
    *) legs+=("$1"); shift ;;
  esac
done
((${#legs[@]})) || { echo "e75_rungD_session: no legs given" >&2; exit 2; }

for spec in "${legs[@]}"; do
  cell="${spec%%:*}"
  [[ -s "${store}/${cell}/cell.json" ]] \
    || { echo "e75_rungD_session: no prebuilt cell at ${store}/${cell}" >&2; exit 2; }
done

export MLXFAST_LOCAL_RUN_LOCK_DIR="${MLXFAST_LOCAL_RUN_LOCK_DIR:-/tmp/mlxfast-shared}"
export E68_BASE_SHA="${E68_BASE_SHA:-$(git rev-parse HEAD)}"
export E68_E2E_ROOT="${E68_E2E_ROOT:-${repo_root}/.mlxfast-private/e75-e2e}"
export E68_VERIFY_FORWARD_KEY="${E68_VERIFY_FORWARD_KEY:-0.060300}"

export LEG_TIP_ARM="${LEG_TIP_ARM:-pbfit}"
export LEG_WANDB_LOGGER="research/e75_wandb_log.py"
export LEG_WANDB_ARGS="--rung D"
export LEG_TABLE_ARMS_MODULE="research/e75_arms.py"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-e75-bank-pbfit-and-price-it-on-the-crown-table}"

export DARKBLOOM_STARTUP_MEMORY_PROFILE="${DARKBLOOM_STARTUP_MEMORY_PROFILE:-full}"
export MLX_MAX_MB_PER_BUFFER="${MLX_MAX_MB_PER_BUFFER:-512}"
export MLX_MAX_OPS_PER_BUFFER="${MLX_MAX_OPS_PER_BUFFER:-50}"

failed=0
for spec in "${legs[@]}"; do
  cell="${spec%%:*}"
  tag="${spec#*:}"
  [[ "${cell}" != "${spec}" && -n "${tag}" ]] \
    || { echo "e75_rungD_session: bad leg spec ${spec}" >&2; exit 2; }
  table="${cell%%-*}"
  price="${cell##*-}"
  extra=()
  [[ "${tag}" == *warmup* ]] && extra+=(--warmup)
  echo "e75_rungD_session: === $(date -u +%Y-%m-%dT%H:%M:%SZ) leg ${tag} (${cell}) ===" >&2
  rc=0
  LEG_TABLE_ARM="${table}" \
  LEG_PREBUILT_CELL="${store}/${cell}" \
  LEG_EXTRA_META="kernel_table=${table}
cell=${cell}
rung=D" \
  research/e68_run_leg.sh "${price}" "${tag}" \
    "${passthrough[@]+"${passthrough[@]}"}" "${extra[@]+"${extra[@]}"}" || rc=$?
  if ((rc != 0)); then
    failed=$((failed + 1))
    echo "e75_rungD_session: leg ${tag} failed with ${rc}; continuing" >&2
  fi
done

echo "e75_rungD_session: done, ${failed} failed leg(s)" >&2
exit 0
