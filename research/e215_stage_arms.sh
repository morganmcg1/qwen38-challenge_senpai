#!/usr/bin/env bash
# Research-only (qwen38-r1-e215): build and stage the two schedule arms.
#
#   usage: research/e215_stage_arms.sh
#
# ARMS. `stepq` is the composed base exactly as merged (E214, FINDING 554).
# `ship` is the control: the SAME tree with one line changed,
#
#     internal static let depthPriceArm: DepthPriceArm = .stepq   ->   .ship
#
# so the walk reads the uniform price the step table replaced. Every other
# byte of the submitted surface, including the inert threshold table and the
# trace witness, is identical between the arms. The census delta therefore
# carries the schedule and nothing else.
#
# The build step is the only step that ever patches the worktree, and it
# restores the tree before it exits, so every measured leg runs against a
# staged binary with a clean checkout. The staged binaries are copied back into
# place AFTER the restore, which also keeps them newer than the sources and
# stops the wrapper's staleness check from rebuilding an arm mid-leg.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

session="Sources/MLXFastModel/Qwen36MTPBlockSession.swift"
surface=(Sources Vendor Package.swift mtp-head.manifest.json)
stage="${MLXFAST_E215_WORKERS:-$(cd ../.. && pwd)/e215-workers}"
worker=".build-worker/release/mlxfast-runtime-worker"
metallib=".build-worker/release/mlx.metallib"
cli=".build/release/mlxfast-swift"
shipped_line='internal static let depthPriceArm: DepthPriceArm = .stepq'
control_line='internal static let depthPriceArm: DepthPriceArm = .ship'

[[ -z "$(git status --porcelain)" ]] || {
  echo "e215_stage_arms.sh: worktree is dirty; commit or discard first" >&2
  exit 2; }

grep -qF "${shipped_line}" "${session}" || {
  echo "e215_stage_arms.sh: the shipped arm line is not in ${session}" >&2
  exit 2; }

restore() { git checkout HEAD -- "${session}" 2>/dev/null || true; }
trap restore EXIT

stage_arm() {
  local arm="$1" dir="${stage}/$1"
  mkdir -p "${dir}"
  cp "${worker}" "${dir}/mlxfast-runtime-worker"
  cp "${metallib}" "${dir}/mlx.metallib"
  cp "${cli}" "${dir}/mlxfast-swift"
  {
    echo "arm=${arm}"
    echo "branch_sha=$(git rev-parse HEAD)"
    echo "session_blob=$(git hash-object "${session}")"
    echo "surface_files_changed_vs_head=$(
      git diff --name-only HEAD -- "${surface[@]}" | wc -l | tr -d ' ')"
    echo "worker_sha256=$(shasum -a 256 "${dir}/mlxfast-runtime-worker" | awk '{print $1}')"
    echo "metallib_sha256=$(shasum -a 256 "${dir}/mlx.metallib" | awk '{print $1}')"
    echo "cli_sha256=$(shasum -a 256 "${dir}/mlxfast-swift" | awk '{print $1}')"
    echo "staged=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "${dir}/manifest.txt"
  cat "${dir}/manifest.txt"
}

mkdir -p "${stage}"

echo "=== stepq: building the composed base tree"
senpai/rebuild-and-assert-worker.sh \
  --require-symbol makeThresholdDepthPrice \
  --require-symbol snapshotScheduleSignal > "${stage}/stepq-build.log" 2>&1 || {
  echo "e215_stage_arms.sh: stepq build failed; see ${stage}/stepq-build.log" >&2
  exit 3; }
stage_arm stepq

echo "=== ship: patching the one arm line"
python3 - "${session}" "${shipped_line}" "${control_line}" <<'PY'
import sys
path, old, new = sys.argv[1:4]
text = open(path).read()
if text.count(old) != 1:
    raise SystemExit(f"e215_stage_arms.sh: {text.count(old)} matches for the arm line")
open(path, "w").write(text.replace(old, new))
PY
changed="$(git diff --name-only -- "${surface[@]}")"
added="$(git diff --numstat -- "${session}" | awk '{print $1}')"
removed="$(git diff --numstat -- "${session}" | awk '{print $2}')"
[[ "${changed}" == "${session}" && "${added}" == "1" && "${removed}" == "1" ]] || {
  echo "e215_stage_arms.sh: the ship patch is not one line of ${session}:" >&2
  git diff --numstat -- "${surface[@]}" >&2
  exit 3; }
senpai/rebuild-and-assert-worker.sh \
  --require-symbol snapshotScheduleSignal > "${stage}/ship-build.log" 2>&1 || {
  echo "e215_stage_arms.sh: ship build failed; see ${stage}/ship-build.log" >&2
  exit 3; }
stage_arm ship

echo "=== restoring the branch tree and reinstalling the stepq binaries"
restore
[[ -z "$(git status --porcelain)" ]] || {
  echo "e215_stage_arms.sh: tree not restored" >&2; exit 3; }
cp "${stage}/stepq/mlxfast-runtime-worker" "${worker}"
cp "${stage}/stepq/mlx.metallib" "${metallib}"
cp "${stage}/stepq/mlxfast-swift" "${cli}"

echo "=== staged in ${stage}"
