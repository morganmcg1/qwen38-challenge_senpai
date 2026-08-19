#!/usr/bin/env bash
# Run the E59 rung 4 end-to-end legs back to back inside one session.
#
#   research/e59_e2e_session.sh ARM:TAG [ARM:TAG ...] [--tokens N] [--hot]
#
# run_job takes an argv list with no environment field, so the session
# environment has to be established inside a script. A leg failure stops the
# session: a half-counterbalanced order is not analysable, and continuing would
# spend the remaining allocation on a sequence the analysis must throw away.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export MLXFAST_LOCAL_RUN_LOCK_DIR="${MLXFAST_LOCAL_RUN_LOCK_DIR:-/tmp/mlxfast-shared}"
export E59_BASE_SHA="${E59_BASE_SHA:-989596895b7c8f889443dac0c87e024a428e6e9e}"
export E59_E2E_ROOT="${E59_E2E_ROOT:-${PWD}/.mlxfast-private/e59-e2e}"

legs=()
passthrough=()
while (($#)); do
  case "$1" in
    --tokens) passthrough+=("--tokens" "$2"); shift 2 ;;
    --hot) passthrough+=("--hot"); shift ;;
    *:*) legs+=("$1"); shift ;;
    *) echo "e59_e2e_session: unknown argument $1" >&2; exit 2 ;;
  esac
done
((${#legs[@]})) || { echo "e59_e2e_session: no ARM:TAG legs given" >&2; exit 2; }

echo "e59_e2e_session: === $(date -u +%Y-%m-%dT%H:%M:%SZ) ${#legs[@]} legs ==="
i=0
for spec in "${legs[@]}"; do
  i=$((i + 1))
  arm="${spec%%:*}"
  tag="${spec#*:}"
  echo "e59_e2e_session: --- leg ${i}/${#legs[@]} arm=${arm} tag=${tag} $(date -u +%Y-%m-%dT%H:%M:%SZ) ---"
  research/e59_e2e_run.sh "${arm}" "${tag}" "${passthrough[@]}"
  rc=$?
  if ((rc != 0)); then
    echo "e59_e2e_session: leg ${tag} exited ${rc}; stopping the session" >&2
    exit "${rc}"
  fi
done
echo "e59_e2e_session: === done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
