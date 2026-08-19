#!/usr/bin/env bash
# Build one immutable worker binary per E56 arm, OUTSIDE the checkout.
#
#   research/e56_build_arms.sh
#
# The two arms differ in exactly one file, so an earlier version of this
# harness selected an arm by checking that file out at the base commit for the
# duration of a leg. That left the branch dirty for as long as a base leg ran,
# which is both a reporting hazard (a hard kill skips the restore and leaves
# base bytes staged on the branch) and unusable here, because a turn may not
# end while the work tree is dirty.
#
# The scored artifact is the worker binary, and both `benchmark.sh` and the
# trusted CLI resolve it through MLXFAST_RUNTIME_WORKER_EXECUTABLE. So build
# each arm ONCE here, publish it outside the repository, and let every timed
# leg run against a clean checkout. Two further properties come free:
# replicates of one arm now run a byte-identical binary, and both arms share
# one trusted CLI, so the instrument is the same on both sides.
#
# This script is the only place that touches the work tree, and it restores it
# on every exit path.
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

readonly SCHEDULE_FILE="Sources/MLXFastModel/Qwen36MTPBlockSession.swift"
readonly E56_BASE_SHA="${E56_BASE_SHA:-a2c3dbc497fd76b3e4f99c529a3eb5e8b2090abf}"
ARM_DIR="${E56_ARM_DIR:-${HOME}/e56-arms}"

head_sha="$(git rev-parse HEAD)"
patched=0

unwind() {
  if ((patched)); then
    git checkout -q "${head_sha}" -- "${SCHEDULE_FILE}" || true
    patched=0
  fi
}
trap unwind EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

if [[ -n "$(git status --porcelain)" ]]; then
  echo "e56_build_arms: work tree is dirty; refusing to build arms over uncommitted work" >&2
  exit 1
fi

publish() {
  local arm="$1" dest="${ARM_DIR}/${arm}"
  echo "=== e56 build arm ${arm} (schedule blob $(git hash-object "${SCHEDULE_FILE}")) ==="
  mkdir -p "${dest}" || return 1
  research/rebuild.sh || return 1
  tools/build-mlx-metallib.sh --all-build-roots || return 1
  # The CLI verifies the metallib that sits BESIDE the worker executable, so
  # the sidecar has to travel with the binary.
  cp -f .build-worker/release/mlxfast-runtime-worker "${dest}/" || return 1
  cp -f .build-worker/release/mlx.metallib "${dest}/" || return 1
  cp -f .build-worker/release/mlx.metallib.fingerprint "${dest}/" || return 1
  {
    echo "arm=${arm}"
    echo "schedule_blob=$(git hash-object "${SCHEDULE_FILE}")"
    echo "worker_sha256=$(shasum -a 256 "${dest}/mlxfast-runtime-worker" | cut -d' ' -f1)"
    echo "metallib_sha256=$(shasum -a 256 "${dest}/mlx.metallib" | cut -d' ' -f1)"
    echo "built=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "${dest}/arm.txt"
  cat "${dest}/arm.txt"
}

# `base` first, `sched` last, so the in-tree build products this script leaves
# behind were compiled from HEAD. benchmark.sh rebuilds whenever a source file
# is newer than the build product, and restoring the schedule file after the
# last build would make that true on the session's first leg.
git checkout -q "${E56_BASE_SHA}" -- "${SCHEDULE_FILE}" || exit 1
patched=1
publish base || exit 1
unwind
publish sched || exit 1

if [[ -n "$(git status --porcelain)" ]]; then
  echo "e56_build_arms: work tree did not come back clean" >&2
  exit 1
fi

# benchmark.sh treats the arm binary as a build product and would rebuild if a
# source looked newer. The base arm was copied before the schedule file came
# back, so republish both mtimes now that the tree is final.
touch "${ARM_DIR}"/base/* "${ARM_DIR}"/sched/*

sched_worker="$(shasum -a 256 "${ARM_DIR}/sched/mlxfast-runtime-worker" | cut -d' ' -f1)"
base_worker="$(shasum -a 256 "${ARM_DIR}/base/mlxfast-runtime-worker" | cut -d' ' -f1)"
sched_metallib="$(shasum -a 256 "${ARM_DIR}/sched/mlx.metallib" | cut -d' ' -f1)"
base_metallib="$(shasum -a 256 "${ARM_DIR}/base/mlx.metallib" | cut -d' ' -f1)"

# If the two arms produced the same binary, the experiment has no treatment.
if [[ "${sched_worker}" == "${base_worker}" ]]; then
  echo "e56_build_arms: both arms produced the same worker; there is no treatment to measure" >&2
  exit 1
fi
# The change touches no Metal source, so a metallib difference would mean the
# arms differ by something this experiment did not intend to test.
if [[ "${sched_metallib}" != "${base_metallib}" ]]; then
  echo "e56_build_arms: the arms carry different metallibs; the treatment is not confined to the schedule" >&2
  exit 1
fi

echo "e56_build_arms: arms ready in ${ARM_DIR}"
echo "  sched worker ${sched_worker}"
echo "  base  worker ${base_worker}"
echo "  shared metallib ${sched_metallib}"
