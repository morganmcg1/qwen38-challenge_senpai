#!/usr/bin/env bash
# Run several forced-depth arms back to back inside one supervised job.
#
# One `--local-iterate` invocation costs four model-loading phases (tripwire,
# reference generation, serial control, MTP leg) plus two cool gates, so
# batching arms is the only way to fit a depth sweep inside the job wall clock.
# A failing arm does not abort the batch: a partial sweep still fits a curve.
#
# usage: research/run-arms.sh --tokens N --head-dir DIR ARM [ARM ...]
# where ARM is a depth 0..8 (forced) or `NAME=adaptive` (schedule decides).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tokens=256
head_dir=""
while (($#)); do
  case "$1" in
    --tokens) tokens="$2"; shift 2 ;;
    --head-dir) head_dir="$2"; shift 2 ;;
    *) break ;;
  esac
done

rc=0
for spec in "$@"; do
  echo "=== run-arms.sh: ${spec} (tokens ${tokens}) at $(date -u +%H:%M:%SZ) ==="
  if [[ "${spec}" == *=adaptive ]]; then
    args=(research/run-arm.sh "${spec%=adaptive}" --trace --tokens "${tokens}")
  else
    args=(research/run-arm.sh "d${spec}" --force-depth "${spec}" --trace --tokens "${tokens}")
  fi
  [[ -n "${head_dir}" ]] && args+=(--head-dir "${head_dir}")
  "${args[@]}" || { rc=1; echo "=== run-arms.sh: ${spec} FAILED ==="; }
done
exit "${rc}"
