#!/usr/bin/env bash
# Run several forced-depth arms back to back inside one supervised job.
#
# One `--local-iterate` invocation costs four model-loading phases (tripwire,
# reference generation, serial control, MTP leg) plus two cool gates, so
# batching arms is the only way to fit a depth sweep inside the job wall clock.
# A failing arm does not abort the batch: a partial sweep still fits a curve.
#
# usage: research/run-arms.sh [--rebuild] --tokens N --head-dir DIR ARM [ARM ...]
# where ARM is a depth 0..8 (forced), `NAME=adaptive` (schedule decides), or
# `NAME=adaptive@V` to substitute the cost curve -- V is either eight
# comma-separated ratios or one scalar repeated eight times. A scalar makes
# the schedule identical to the pre-change greedy policy, so a baseline arm
# and a candidate arm can share one binary and one thermal window.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tokens=256
head_dir=""
rebuild=0
while (($#)); do
  case "$1" in
    --tokens) tokens="$2"; shift 2 ;;
    --head-dir) head_dir="$2"; shift 2 ;;
    --rebuild) rebuild=1; shift ;;
    *) break ;;
  esac
done

# Rebuilding here rather than letting the first arm absorb it keeps the build
# off the far side of that arm's cool gate, so every arm in the batch is timed
# against the same thermal preparation.
if ((rebuild)); then
  echo "=== run-arms.sh: rebuild at $(date -u +%H:%M:%SZ) ==="
  research/rebuild.sh || exit 1
fi

rc=0
for spec in "$@"; do
  echo "=== run-arms.sh: ${spec} (tokens ${tokens}) at $(date -u +%H:%M:%SZ) ==="
  if [[ "${spec}" == *=adaptive* ]]; then
    name="${spec%%=adaptive*}"
    hvec="${spec#*=adaptive}"; hvec="${hvec#@}"
    args=(research/run-arm.sh "${name}" --trace --tokens "${tokens}")
    if [[ -n "${hvec}" ]]; then
      [[ "${hvec}" == *,* ]] || hvec="${hvec},${hvec},${hvec},${hvec},${hvec},${hvec},${hvec},${hvec}"
      args+=(--h-vector "${hvec}")
    fi
  else
    args=(research/run-arm.sh "d${spec}" --force-depth "${spec}" --trace --tokens "${tokens}")
  fi
  [[ -n "${head_dir}" ]] && args+=(--head-dir "${head_dir}")
  "${args[@]}" || { rc=1; echo "=== run-arms.sh: ${spec} FAILED ==="; }
done
exit "${rc}"
