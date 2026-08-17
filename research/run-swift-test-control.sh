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

# Revert through the worktree only. `git checkout $BASE -- $FILE` would also
# stage the base blob, which makes a plain `git checkout -- $FILE` restore from
# that poisoned index instead of from HEAD and silently leave the candidate
# change reverted. Verify the restore rather than assuming it.
restore() {
  local now expect
  git checkout HEAD -- "$FILE"
  now="$(git hash-object "$FILE")"
  expect="$(git rev-parse "HEAD:$FILE")"
  if [ "$now" != "$expect" ]; then
    echo "RESTORE FAILED: $FILE is $now, expected $expect" >&2
    return 1
  fi
  echo "control: $FILE restored to HEAD ($expect)"
}
trap restore EXIT

git show "$BASE:$FILE" >"$FILE" || exit 3
echo "control: $FILE reverted to $BASE"
git --no-pager diff --stat -- "$FILE"
research/run-swift-test-set.sh "$TAG"
