#!/usr/bin/env bash
# Research-only (qwen38-r1-e168-margin-clamp-calibration): freeze the current
# checkout's release products under a named arm, so a later session can time
# two builds against each other without rebuilding between legs.
#
#   research/e168_cap_stage.sh ARM
#
# Usage for the depth-cap confirmation, from a CLEAN worktree:
#
#   research/e168_cap_stage.sh cap                 # on the candidate commit
#   git checkout -B e168-base origin/senpai/qwen38-mtp-r1
#   research/e168_cap_stage.sh base                # on the campaign base
#   git checkout qwen-askeladd/e168-margin-clamp-calibration
#   research/e168_cap_confirm.sh 1
#
# The staged commit is recorded next to the binaries. A staged pair whose
# `stage.txt` does not name the commit you think you measured is not evidence.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

arm="${1:?usage: research/e168_cap_stage.sh ARM}"
bin_root="${E168_BIN:-${PWD}/research/out/e172/bin}"
dest="${bin_root}/${arm}"

[[ -z "$(git status --porcelain)" ]] || {
  echo "e168_cap_stage: worktree is dirty; a staged binary must name one commit" >&2
  exit 2
}

research/e168_build.sh || exit 1

mkdir -p "${dest}"
cp .build/release/mlxfast-swift "${dest}/mlxfast-swift"
cp .build-worker/release/mlxfast-runtime-worker "${dest}/mlxfast-runtime-worker"

{
  echo "arm=${arm}"
  echo "commit=$(git rev-parse HEAD)"
  echo "commit_subject=$(git log -1 --pretty=%s)"
  echo "staged_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "cli_sha256=$(shasum -a 256 "${dest}/mlxfast-swift" | cut -d' ' -f1)"
  echo "worker_sha256=$(shasum -a 256 "${dest}/mlxfast-runtime-worker" | cut -d' ' -f1)"
} > "${dest}/stage.txt"

cat "${dest}/stage.txt"
