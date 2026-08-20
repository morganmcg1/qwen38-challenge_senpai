#!/usr/bin/env bash
# One E56 session: five arms, each twice, in palindrome order.
#
#   research/e56_session.sh [--tokens N]
#
# base, s45, s89, h224, s45h224, s45h224, h224, s89, s45, base. A palindrome
# cancels monotone thermal or power drift to first order for EVERY arm, not
# only for the two an ABBA pair balances, and the outer `base` pair is the null
# arm: two byte-identical builds measured in the same session bound what this
# instrument calls a difference when there is none.
#
# Session 3 lost a leg because the real cool gate refused to release the GPU
# inside its wait, which cost the pair that leg belonged to. A failed leg is
# therefore retried once under a suffixed tag before the session gives up on it.
#
# The session stops if the first leg fails twice, because every later leg would
# fail the same way and spend two hours proving it.
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
# Session label, so W&B and research/out keep the post-E55 five-arm session
# distinguishable from the pre-E55 four-arm session 3.
export E56_SESSION="${E56_SESSION:-s4}"

echo "e56_session: ${tokens} decode tokens per leg"
echo "e56_session: GPU observation before the first leg"
python3 research/e49_gpu_gate.py || true

rc=0
first=1
for spec in base:s4base1 s45:s4s45a s89:s4s89a h224:s4h224a s45h224:s4mixa \
            s45h224:s4mixb h224:s4h224b s89:s4s89b s45:s4s45b base:s4base2; do
  arm="${spec%%:*}"
  tag="${spec##*:}"
  research/e56_run_leg.sh "${arm}" "${tag}" --tokens "${tokens}"
  status=$?
  if ((status != 0)); then
    echo "e56_session: leg ${tag} FAILED (exit ${status}); retrying once"
    research/e56_run_leg.sh "${arm}" "${tag}r" --tokens "${tokens}"
    status=$?
  fi
  if ((status != 0)); then
    rc=1
    echo "e56_session: leg ${tag} FAILED on the retry too (exit ${status})"
    if ((first)); then
      echo "e56_session: first leg failed twice; stopping before the session spends the rest"
      break
    fi
  fi
  first=0
done
date -u "+e56_session: === %Y-%m-%dT%H:%M:%SZ session complete ==="
exit "${rc}"
