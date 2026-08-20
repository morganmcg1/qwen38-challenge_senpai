#!/usr/bin/env bash
# One E68 rung 3 session: every depth-price arm, counterbalanced, back to back.
#
#   research/e68_rung3_session.sh ARM:TAG [ARM:TAG ...] [--tokens N] [--hot]
#
# The wrapper exists because run_job takes an argv list with no environment
# field, so the environment must be set here.
#
# Design of the arm order. The four arms are `ship`, `pb5`, `pb7` and `pbfit`,
# and a session runs them as one mirrored palindrome after a declared warm-up
# leg that is discarded:
#
#   warmup  ship pb5 pb7 pbfit  pbfit pb7 pb5 ship
#
# Every arm then appears once in the first half and once in the second, at
# mirrored positions, so a monotone drift over the session cancels to first
# order in the arm contrast. The null bar is the largest same-arm spread the
# session itself produced, not an assumed noise figure.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

export MLXFAST_LOCAL_RUN_LOCK_DIR="${MLXFAST_LOCAL_RUN_LOCK_DIR:-/tmp/mlxfast-shared}"
export E68_BASE_SHA="${E68_BASE_SHA:-$(git rev-parse origin/senpai/qwen38-mtp-r1)}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-e68-schedule-against-the-new-cost-curve}"

# Which verify-forward normaliser `pbfit` divides the measured width steps by.
# The rung-1 artifact carries a sweep, so the arm must name one rather than
# guess. 0.060300 is the fit closest to the measured C(1) = 60.372 ms, which
# keeps numerator and denominator inside the same measurement: the in-situ
# curve is a near constant 0.50 of the isolated one at every width, so that
# factor cancels only when both come from the same source. The depth decision
# is invariant across the whole swept range anyway.
export E68_VERIFY_FORWARD_KEY="${E68_VERIFY_FORWARD_KEY:-0.060300}"

# Ranked command-buffer geometry. Unlike a rung-1 cell leg, a rung-3 leg starts
# the MTP worker, so `DARKBLOOM_STARTUP_MEMORY_PROFILE=full` is load bearing:
# without it the worker force-sets 128 MiB / 64 ops on this 48 GiB host and the
# other two exports never reach MLX.
export DARKBLOOM_STARTUP_MEMORY_PROFILE="${DARKBLOOM_STARTUP_MEMORY_PROFILE:-full}"
export MLX_MAX_MB_PER_BUFFER="${MLX_MAX_MB_PER_BUFFER:-512}"
export MLX_MAX_OPS_PER_BUFFER="${MLX_MAX_OPS_PER_BUFFER:-50}"

legs=()
passthrough=()
while (($#)); do
  case "$1" in
    --tokens|--hot|--warmup) passthrough+=("$1")
      [[ "$1" == "--tokens" ]] && { passthrough+=("$2"); shift; }
      shift ;;
    *) legs+=("$1"); shift ;;
  esac
done
((${#legs[@]})) || { echo "e68_rung3_session: no legs given" >&2; exit 2; }

# The Metal library is not part of any E68 arm, but a tree whose metallib was
# never built emits a stale-source warning that discards the leg. Build it once
# here rather than paying for it, or tripping over it, on every leg.
tools/build-mlx-metallib.sh --all-build-roots \
  > .mlxfast-private/e68-metallib-build.log 2>&1 \
  || { echo "e68_rung3_session: metallib build failed" >&2; exit 2; }

failed=0
for spec in "${legs[@]}"; do
  arm="${spec%%:*}"
  tag="${spec#*:}"
  [[ "${arm}" != "${spec}" && -n "${tag}" ]] \
    || { echo "e68_rung3_session: bad leg spec ${spec}" >&2; exit 2; }
  extra=()
  # The first leg of a session is cold. Declare it discarded here rather than
  # dropping it from the analysis later.
  [[ "${tag}" == *warmup* ]] && extra+=(--warmup)
  echo "e68_rung3_session: === $(date -u +%Y-%m-%dT%H:%M:%SZ) leg ${tag} (${arm}) ===" >&2
  rc=0
  research/e68_run_leg.sh "${arm}" "${tag}" \
    "${passthrough[@]+"${passthrough[@]}"}" "${extra[@]+"${extra[@]}"}" || rc=$?
  if ((rc != 0)); then
    failed=$((failed + 1))
    echo "e68_rung3_session: leg ${tag} failed with ${rc}; continuing" >&2
  fi
done

echo "e68_rung3_session: done, ${failed} failed leg(s)" >&2
exit 0
