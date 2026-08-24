#!/usr/bin/env bash
# Research-only (qwen38-r1-e168-margin-clamp-calibration): freeze the CURRENT
# HEAD's release products under a named arm, so a later session can time two
# builds against each other without rebuilding between legs.
#
#   research/e168_cap_stage.sh ARM EXPECTED_COMMIT
#
# EXPECTED_COMMIT is mandatory and is compared against HEAD before and after
# the build. It is not ceremony: an earlier staging attempt found the job
# runner had restored the candidate branch after the shell had checked out the
# base, which would have staged candidate binaries under the name `base`.
#
# This script stages ONE arm and never changes the checkout. The campaign base
# does not contain research/, so a script that checked out the base would
# delete itself and its build helper while running. The base arm is therefore
# staged by the out-of-tree companion, which inlines the same build and the
# same stale-worker tripwire:
#
#   ../e168_stage_base.sh REPO base <base-sha>
#
# Full sequence for the depth-cap confirmation, from a CLEAN worktree:
#
#   research/e168_cap_stage.sh cap "$(git rev-parse HEAD)"
#   ../e168_stage_base.sh "${PWD}" base "$(git rev-parse origin/senpai/qwen38-mtp-r1)"
#   research/e168_cap_confirm.sh 1
#
# A staged pair whose `stage.txt` does not name the commit you think you
# measured is not evidence.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

arm="${1:?usage: research/e168_cap_stage.sh ARM EXPECTED_COMMIT}"
want_ref="${2:?usage: research/e168_cap_stage.sh ARM EXPECTED_COMMIT}"
bin_root="${E168_BIN:-${PWD}/research/out/e172/bin}"
dest="${bin_root}/${arm}"

[[ -z "$(git status --porcelain)" ]] || {
  echo "e168_cap_stage: worktree is dirty; a staged binary must name one commit" >&2
  exit 2
}

want="$(git rev-parse --verify "${want_ref}^{commit}")" || exit 2
[[ "$(git rev-parse HEAD)" == "${want}" ]] || {
  echo "e168_cap_stage: HEAD is $(git rev-parse HEAD), expected ${want}" >&2
  exit 2
}

research/e168_build.sh || exit 1

have="$(git rev-parse HEAD)"
[[ "${have}" == "${want}" ]] || {
  echo "e168_cap_stage: HEAD moved during the build: ${have} != ${want}" >&2
  exit 2
}

mkdir -p "${dest}"
cp .build/release/mlxfast-swift "${dest}/mlxfast-swift"
cp .build-worker/release/mlxfast-runtime-worker "${dest}/mlxfast-runtime-worker"
# benchmark-qwen-mtp.sh resolves the metallib as
# `$(dirname "${RUNTIME_WORKER_BIN}")/mlx.metallib`, so a staged worker without
# one beside it cannot run. Neither arm of this experiment changes Metal
# source, so both arms stage the same library; its digest is recorded so a
# reader can check that rather than take it on trust.
cp .build-worker/arm64-apple-macosx/release/mlx.metallib "${dest}/mlx.metallib"

{
  echo "arm=${arm}"
  echo "commit=${have}"
  echo "metallib_sha256=$(shasum -a 256 "${dest}/mlx.metallib" | cut -d' ' -f1)"
  echo "commit_subject=$(git log -1 --pretty=%s)"
  echo "staged_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "cli_sha256=$(shasum -a 256 "${dest}/mlxfast-swift" | cut -d' ' -f1)"
  echo "worker_sha256=$(shasum -a 256 "${dest}/mlxfast-runtime-worker" | cut -d' ' -f1)"
} > "${dest}/stage.txt"

cat "${dest}/stage.txt"
