#!/usr/bin/env bash
# Bring a fresh role checkout back to a runnable state without re-downloading
# the 14 GB target checkpoint or the two head artifacts.
#
# The launcher created a new role directory for this generation. The transformed
# target checkpoint, the organizer-pinned head and the declared head already
# exist in the previous generation's role directory on the same APFS volume, so
# `cp -c` clones them at zero extra disk cost. Everything else is rebuilt from
# source in this checkout.
#
# Research-only. Not a submitted path.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PREV_ROLE="${E158_PREV_ROLE:-/Users/ec2-user/.senpai/native/qwen38-mlx-senpai-r2/roles/student-qwen-askeladd}"
CACHE_ROOT="${HOME}/.cache/mlxfast/qwen3.8-27b-mtp-v1"
PREV_CACHE="${PREV_ROLE}/home/.cache/mlxfast/qwen3.8-27b-mtp-v1"

step() { printf '\n=== %s === %s\n' "$1" "$(date -u +%H:%M:%SZ)"; }

step "target checkpoint"
if [[ -f weights/config.json ]]; then
  echo "weights/ already present"
else
  [[ -f "${PREV_ROLE}/workspace/target/weights/config.json" ]] || {
    echo "FATAL: no previous transformed checkpoint at ${PREV_ROLE}" >&2
    exit 1
  }
  cp -Rc "${PREV_ROLE}/workspace/target/weights/." weights/ || exit 1
  ls -1 weights | head
fi

step "head artifacts"
mkdir -p "${CACHE_ROOT}"
for d in mtp-head mtp-head-declared mtp-head-declared-run; do
  if [[ -e "${CACHE_ROOT}/${d}" ]]; then
    echo "${d} already present"
  elif [[ -e "${PREV_CACHE}/${d}" ]]; then
    cp -Rc "${PREV_CACHE}/${d}" "${CACHE_ROOT}/${d}" || exit 1
    echo "${d} cloned"
  else
    echo "WARNING: ${d} missing in previous role cache" >&2
  fi
done
du -sh "${CACHE_ROOT}"/* 2>/dev/null

step "head digests"
python3 research/e158_head_census.py --digest-only 2>/dev/null || true

step "swift build"
# The rebuild guard refuses to run without an assertion. `installExactQKVRows`
# is the island selector this branch reads at warm-up. `installExactQKVRows` is
# fully inlined by the release compiler and leaves no symbol, so it cannot be
# used as a witness.
senpai/rebuild-and-assert-worker.sh --require-symbol Qwen35IslandArm || exit 1

step "done"
ls -l .build/release/mlxfast-swift .build-worker/release/mlxfast-runtime-worker
