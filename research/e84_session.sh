#!/usr/bin/env bash
# One E84 counterbalanced timing session.
#
#   research/e84_session.sh ARM:TAG [ARM:TAG ...] [--tokens N] [--hot]
#
# `run_job` takes an argv list with no environment field, so the session sets
# the environment once here and every leg inherits it. Run the arms as a
# palindrome inside one session: monotone thermal drift then cancels to first
# order between the two halves.
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

legs=()
passthrough=()
while (($#)); do
  case "$1" in
    --tokens) passthrough+=("$1" "$2"); shift 2 ;;
    --hot|--warmup|--build-only) passthrough+=("$1"); shift ;;
    *) legs+=("$1"); shift ;;
  esac
done
((${#legs[@]})) || { echo "e84_session: no legs given" >&2; exit 2; }

export MLXFAST_LOCAL_RUN_LOCK_DIR="${MLXFAST_LOCAL_RUN_LOCK_DIR:-/tmp/mlxfast-shared}"
export E84_BASE_SHA="${E84_BASE_SHA:-5ea174c50b98407bc463c463cc7c7a85d32960a7}"
export E84_ROOT="${E84_ROOT:-${repo_root}/.mlxfast-private/e84}"
export DARKBLOOM_STARTUP_MEMORY_PROFILE="${DARKBLOOM_STARTUP_MEMORY_PROFILE:-full}"
export MLX_MAX_MB_PER_BUFFER="${MLX_MAX_MB_PER_BUFFER:-512}"
export MLX_MAX_OPS_PER_BUFFER="${MLX_MAX_OPS_PER_BUFFER:-50}"

failed=0
for spec in "${legs[@]}"; do
  arm="${spec%%:*}"
  tag="${spec#*:}"
  [[ "${arm}" != "${spec}" && -n "${tag}" ]] \
    || { echo "e84_session: bad leg spec ${spec}" >&2; exit 2; }
  extra=()
  [[ "${tag}" == *warmup* ]] && extra+=(--warmup)
  echo "e84_session: === $(date -u +%Y-%m-%dT%H:%M:%SZ) leg ${tag} (${arm}) ===" >&2
  rc=0
  research/e84_run_leg.sh "${arm}" "${tag}" \
    "${passthrough[@]+"${passthrough[@]}"}" "${extra[@]+"${extra[@]}"}" || rc=$?
  if ((rc != 0)); then
    failed=$((failed + 1))
    echo "e84_session: leg ${tag} failed with ${rc}; continuing" >&2
  fi
done

echo "e84_session: done, ${failed} failed leg(s)" >&2
exit 0
