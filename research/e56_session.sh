#!/usr/bin/env bash
# One E56 ABBA session: base, sched, sched, base.
#
#   research/e56_session.sh [--tokens N]
#
# ABBA order cancels monotone thermal or power drift to first order, and the
# second `base` leg is the null arm: two byte-identical builds measured in the
# same session bound what this instrument calls a difference when there is none.
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

tokens=256
while (($#)); do
  case "$1" in
    --tokens) tokens="$2"; shift 2 ;;
    *) echo "e56_session: unknown argument $1" >&2; exit 2 ;;
  esac
done

export MLXFAST_LOCAL_RUN_LOCK_DIR="${MLXFAST_LOCAL_RUN_LOCK_DIR:-/tmp/mlxfast-shared}"

echo "e56_session: GPU observation before the first leg"
python3 research/e49_gpu_gate.py || true

rc=0
for spec in base:baseA sched:schedB sched:schedB2 base:baseA2; do
  arm="${spec%%:*}"
  tag="${spec##*:}"
  research/e56_run_leg.sh "${arm}" "${tag}" --tokens "${tokens}" \
    || { rc=1; echo "e56_session: leg ${tag} FAILED"; }
done
date -u "+e56_session: === %Y-%m-%dT%H:%M:%SZ session complete ==="
exit "${rc}"
