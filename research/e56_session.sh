#!/usr/bin/env bash
# One E56 session: four arms, each twice, in palindrome order.
#
#   research/e56_session.sh [--tokens N]
#
# base, s45, s89, sfull, sfull, s89, s45, base. A palindrome cancels monotone
# thermal or power drift to first order for EVERY arm, not only for the two an
# ABBA pair balances, and the outer `base` pair is the null arm: two
# byte-identical builds measured in the same session bound what this instrument
# calls a difference when there is none.
#
# The session stops if the first leg fails, because every later leg would fail
# the same way and spend an hour proving it.
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

tokens=512
while (($#)); do
  case "$1" in
    --tokens) tokens="$2"; shift 2 ;;
    *) echo "e56_session: unknown argument $1" >&2; exit 2 ;;
  esac
done

export MLXFAST_LOCAL_RUN_LOCK_DIR="${MLXFAST_LOCAL_RUN_LOCK_DIR:-/tmp/mlxfast-shared}"

echo "e56_session: ${tokens} decode tokens per leg"
echo "e56_session: GPU observation before the first leg"
python3 research/e49_gpu_gate.py || true

rc=0
first=1
for spec in base:baseA s45:s45B s89:s89C sfull:sfullD \
            sfull:sfullD2 s89:s89C2 s45:s45B2 base:baseA2; do
  arm="${spec%%:*}"
  tag="${spec##*:}"
  research/e56_run_leg.sh "${arm}" "${tag}" --tokens "${tokens}"
  status=$?
  if ((status != 0)); then
    rc=1
    echo "e56_session: leg ${tag} FAILED (exit ${status})"
    if ((first)); then
      echo "e56_session: first leg failed; stopping before the session spends the rest"
      break
    fi
  fi
  first=0
done
date -u "+e56_session: === %Y-%m-%dT%H:%M:%SZ session complete ==="
exit "${rc}"
