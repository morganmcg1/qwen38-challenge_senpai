#!/usr/bin/env bash
# Research-only (qwen38-r1-e214): build and stage the two census arms.
#
#   usage: research/e214_stage_arms.sh
#
# The census pair needs a BASE binary and a CANDIDATE binary on this host. Only
# this build step ever patches the worktree, and it restores the tree before it
# exits, so the measurement step can run against staged binaries with a clean
# checkout throughout.
#
# The only submitted file this candidate changes is the session file, so
# checking that one file out at BASE_SHA makes the whole submitted surface
# byte-identical to the base. The script asserts that surface, asserts the
# built symbol for each arm, and records every digest in a manifest the
# measurement step re-checks before it runs a leg.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

base_sha="e0c7a026aebc9dbdd579577964ad413a11ede6fc"
session="Sources/MLXFastModel/Qwen36MTPBlockSession.swift"
surface=(Sources Vendor Package.swift mtp-head.manifest.json)
stage="${MLXFAST_E214_WORKERS:-$(cd ../.. && pwd)/e214-workers}"
worker=".build-worker/release/mlxfast-runtime-worker"
metallib=".build-worker/release/mlx.metallib"
cli=".build/release/mlxfast-swift"

[[ -z "$(git status --porcelain)" ]] || {
  echo "e214_stage_arms.sh: worktree is dirty; commit or discard first" >&2
  exit 2; }

changed="$(git diff --name-only "${base_sha}" -- "${surface[@]}")"
[[ "${changed}" == "${session}" ]] || {
  echo "e214_stage_arms.sh: submitted surface differs from ${base_sha} in more" >&2
  echo "  than the session file, so one checkout cannot make the base arm:" >&2
  echo "${changed}" >&2
  exit 2; }

restore() {
  # From HEAD, not the index: staging the base arm also writes the base blob
  # into the index, so an index-side restore would lose the branch file.
  git checkout HEAD -- "${session}" 2>/dev/null || true
}
trap restore EXIT

stage_arm() {
  local arm="$1" dir="${stage}/$1"
  mkdir -p "${dir}"
  cp "${worker}" "${dir}/mlxfast-runtime-worker"
  cp "${metallib}" "${dir}/mlx.metallib"
  cp "${cli}" "${dir}/mlxfast-swift"
  {
    echo "arm=${arm}"
    echo "session_blob=$(git hash-object "${session}")"
    echo "surface_equals_base=$(
      git diff --quiet "${base_sha}" -- "${surface[@]}" && echo true || echo false)"
    echo "worker_sha256=$(shasum -a 256 "${dir}/mlxfast-runtime-worker" | awk '{print $1}')"
    echo "metallib_sha256=$(shasum -a 256 "${dir}/mlx.metallib" | awk '{print $1}')"
    echo "cli_sha256=$(shasum -a 256 "${dir}/mlxfast-swift" | awk '{print $1}')"
    echo "staged=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "${dir}/manifest.txt"
  cat "${dir}/manifest.txt"
}

mkdir -p "${stage}"

echo "=== cand: building the branch tree"
senpai/rebuild-and-assert-worker.sh \
  --require-symbol makeThresholdDepthPrice \
  --require-symbol snapshotScheduleSignal > "${stage}/cand-build.log" 2>&1 || {
  echo "e214_stage_arms.sh: candidate build failed; see ${stage}/cand-build.log" >&2
  exit 3; }
stage_arm cand

echo "=== base: checking ${session} out at ${base_sha}"
git checkout "${base_sha}" -- "${session}"
git diff --quiet "${base_sha}" -- "${surface[@]}" || {
  echo "e214_stage_arms.sh: base arm surface is not byte-identical to the base" >&2
  exit 3; }
senpai/rebuild-and-assert-worker.sh \
  --forbid-symbol makeThresholdDepthPrice \
  --require-symbol snapshotScheduleSignal > "${stage}/base-build.log" 2>&1 || {
  echo "e214_stage_arms.sh: base build failed; see ${stage}/base-build.log" >&2
  exit 3; }
stage_arm base

echo "=== restoring the branch tree and reinstalling the candidate binaries"
restore
[[ -z "$(git status --porcelain)" ]] || {
  echo "e214_stage_arms.sh: tree not restored" >&2; exit 3; }
cp "${stage}/cand/mlxfast-runtime-worker" "${worker}"
cp "${stage}/cand/mlx.metallib" "${metallib}"
cp "${stage}/cand/mlxfast-swift" "${cli}"

echo "=== staged in ${stage}"
