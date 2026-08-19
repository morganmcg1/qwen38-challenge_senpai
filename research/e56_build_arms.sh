#!/usr/bin/env bash
# Build one immutable worker binary per E56 arm, OUTSIDE the checkout.
#
#   research/e56_build_arms.sh
#
# Four arms, because the two weight-stream crossings do not pay off in the same
# place: `base` is the shipped scalar price, `s45` prices only the 4 -> 5
# crossing, `s89` prices only the 8 -> 9 crossing, and `sfull` prices both.
# Every arm differs from HEAD by at most the two declaration lines
# `pricedBoundaryWidths` and `declaredClosedDepthSteps`, and every patched arm
# is re-read by `e56_walk_probe.py --check` before it is built, so an arm whose
# price silently closes a depth step cannot reach the GPU.
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
readonly E56_BASE_SHA="${E56_BASE_SHA:-aded0f57429169262d3d2bdc11be7acbdef261d1}"
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
  local arm="$1"
  local dest="${ARM_DIR}/${arm}"
  echo "=== e56 build arm ${arm} (schedule blob $(git hash-object "${SCHEDULE_FILE}")) ==="
  mkdir -p "${dest}" || return 1
  if [[ "${arm}" != "base" ]]; then
    python3 research/e56_walk_probe.py --check || return 1
  fi
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
    echo "priced_boundary_widths=$(sed -n 's/.*pricedBoundaryWidths: Set<Int> = \[\(.*\)\]/\1/p' "${SCHEDULE_FILE}" | tr -d ' ')"
    echo "declared_closed_depth_steps=$(sed -n 's/.*declaredClosedDepthSteps: \[Int\] = \[\(.*\)\]/\1/p' "${SCHEDULE_FILE}" | tr -d ' ')"
    echo "built=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "${dest}/arm.txt"
  cat "${dest}/arm.txt"
}

# Patch one declaration line in place and prove the substitution took.
set_declaration() {
  local pattern="$1" replacement="$2"
  sed -i '' -E "s/${pattern}/${replacement}/" "${SCHEDULE_FILE}" || return 1
  grep -qF "${replacement}" "${SCHEDULE_FILE}" || {
    echo "e56_build_arms: substitution ${replacement} did not take" >&2
    return 1
  }
}

# `sfull` closes depth step 7 at every acceptance rate and `s89` does too, so
# each arm has to declare its own closures or the probe rejects it.
select_arm() {
  local arm="$1"
  git checkout -q "${head_sha}" -- "${SCHEDULE_FILE}" || return 1
  patched=1
  case "${arm}" in
    s45) ;;   # HEAD already prices only the 4 -> 5 crossing
    s89)
      set_declaration \
        "pricedBoundaryWidths: Set<Int> = \[5\]" \
        "pricedBoundaryWidths: Set<Int> = [9]" || return 1
      set_declaration \
        "declaredClosedDepthSteps: \[Int\] = \[\]" \
        "declaredClosedDepthSteps: [Int] = [7]" || return 1
      ;;
    sfull)
      set_declaration \
        "pricedBoundaryWidths: Set<Int> = \[5\]" \
        "pricedBoundaryWidths: Set<Int> = [5, 9]" || return 1
      set_declaration \
        "declaredClosedDepthSteps: \[Int\] = \[\]" \
        "declaredClosedDepthSteps: [Int] = [7]" || return 1
      ;;
    *) echo "e56_build_arms: unknown arm ${arm}" >&2; return 1 ;;
  esac
}

# `base` first and the HEAD arm last, so the in-tree build products this script
# leaves behind were compiled from HEAD. benchmark.sh rebuilds whenever a source
# file is newer than the build product, and restoring the schedule file after
# the last build would make that true on the session's first leg.
git checkout -q "${E56_BASE_SHA}" -- "${SCHEDULE_FILE}" || exit 1
patched=1
publish base || exit 1
for arm in s89 sfull s45; do
  select_arm "${arm}" || exit 1
  publish "${arm}" || exit 1
done
unwind

if [[ -n "$(git status --porcelain)" ]]; then
  echo "e56_build_arms: work tree did not come back clean" >&2
  exit 1
fi

# benchmark.sh treats the arm binary as a build product and would rebuild if a
# source looked newer, so republish every mtime now that the tree is final.
for arm in base s45 s89 sfull; do
  touch "${ARM_DIR}/${arm}"/*
done

declare -a workers=()
metallib=""
for arm in base s45 s89 sfull; do
  worker_digest="$(shasum -a 256 "${ARM_DIR}/${arm}/mlxfast-runtime-worker" | cut -d' ' -f1)"
  metallib_digest="$(shasum -a 256 "${ARM_DIR}/${arm}/mlx.metallib" | cut -d' ' -f1)"
  echo "  ${arm} worker ${worker_digest}"
  # Two arms with the same binary carry the same treatment, so a contrast
  # between them would measure nothing.
  for seen in ${workers[@]+"${workers[@]}"}; do
    if [[ "${seen}" == "${worker_digest}" ]]; then
      echo "e56_build_arms: arm ${arm} duplicates another arm's worker" >&2
      exit 1
    fi
  done
  workers+=("${worker_digest}")
  # The change touches no Metal source, so a metallib difference would mean the
  # arms differ by something this experiment did not intend to test.
  if [[ -z "${metallib}" ]]; then
    metallib="${metallib_digest}"
  elif [[ "${metallib}" != "${metallib_digest}" ]]; then
    echo "e56_build_arms: arm ${arm} carries a different metallib; the treatment is not confined to the schedule" >&2
    exit 1
  fi
done

echo "e56_build_arms: arms ready in ${ARM_DIR}"
echo "  shared metallib ${metallib}"
