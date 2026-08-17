#!/bin/bash
# Capture the swift test failure SET with the candidate's one source change
# reverted, so the comparison isolates the change rather than the base.
set -uo pipefail
cd "$(dirname "$0")/.."

FILE="Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift"
BASE="${1:?usage: run-swift-test-control.sh BASE_SHA TAG}"
TAG="${2:?usage: run-swift-test-control.sh BASE_SHA TAG}"

if [ -n "$(git status --porcelain)" ]; then
  echo "refusing to run: worktree is dirty" >&2
  git status --porcelain >&2
  exit 2
fi

restore() { git checkout -- "$FILE"; }
trap restore EXIT

git checkout "$BASE" -- "$FILE"
echo "control: $FILE reverted to $BASE"
git --no-pager diff --stat -- "$FILE"
research/run-swift-test-set.sh "$TAG"
