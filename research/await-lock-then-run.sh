#!/usr/bin/env bash
# Wait for the local benchmark lock to clear, then exec a measurement command.
#
# benchmark.sh takes the lock fail-fast, so a run launched while an unrelated
# local benchmark is resident dies immediately.  This wrapper lets run_job own
# the waiting instead of a polling loop in an agent terminal.  It never sets
# MLXFAST_LOCAL_RUN_GUARD=0: contention is resolved by waiting, never bypassed.
#
# Usage: research/await-lock-then-run.sh MAX_WAIT_SECONDS CMD [ARGS...]

set -uo pipefail

if [ "$#" -lt 2 ]; then
  echo "usage: $0 MAX_WAIT_SECONDS CMD [ARGS...]" >&2
  exit 2
fi

max_wait="$1"
shift

lock_root="${MLXFAST_CACHE_ROOT:-${HOME}/.cache/mlxfast}"
lock_path="${lock_root%/}/mlxfast-local-benchmark-$(id -u).lock"

deadline=$(( $(date +%s) + max_wait ))
waited=0
while [ -d "$lock_path" ]; do
  holder="$(cat "${lock_path}/pid" 2>/dev/null || echo '?')"
  if [ "$holder" != "?" ] && ! kill -0 "$holder" 2>/dev/null; then
    echo "await-lock: lock holder pid ${holder} is gone but the lock remains; leaving it for a human" >&2
    exit 1
  fi
  now=$(date +%s)
  if [ "$now" -ge "$deadline" ]; then
    echo "await-lock: still held by pid ${holder} after ${waited}s; giving up without running" >&2
    exit 75
  fi
  if [ $(( waited % 60 )) -eq 0 ]; then
    echo "await-lock: held by pid ${holder}, waited ${waited}s of ${max_wait}s"
  fi
  sleep 15
  waited=$(( waited + 15 ))
done

echo "await-lock: lock free after ${waited}s; starting: $*"
exec "$@"
