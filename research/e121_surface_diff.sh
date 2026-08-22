#!/usr/bin/env bash
# Rule 55: prove the submitted surface equals the exact measured tree.
#
#   usage: research/e121_surface_diff.sh SUBMIT_BASE_SHA
#
# `submit-official.sh` packs the CURRENT content of every path in
# `benchmark.json:editablePaths`. This script diffs those paths only, from the
# merge base of HEAD and the submit base up to HEAD, so the output is exactly
# the scored-surface change the submission carries. Any path outside
# `editablePaths` is invisible here on purpose: research files and notes are
# not submitted and must not appear.
#
# The worktree must be clean, because a dirty scored path would mean the packed
# bytes are not the bytes that were measured.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

submit_base="${1:?usage: research/e121_surface_diff.sh SUBMIT_BASE_SHA}"
out="research/e121-artifacts/rung3-surface-diff.txt"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "e121_surface_diff: worktree is dirty; packed bytes would not be measured bytes" >&2
  git status --porcelain >&2
  exit 1
fi

mapfile -t paths < <(python3 -c "
import json
print('\n'.join(json.load(open('benchmark.json'))['editablePaths']))
")
[[ "${#paths[@]}" -gt 0 ]] || { echo "e121_surface_diff: no editablePaths" >&2; exit 2; }

merge_base="$(git merge-base HEAD "${submit_base}")" || exit 3
head_sha="$(git rev-parse HEAD)"

{
  echo "submit_base=${submit_base}"
  echo "merge_base=${merge_base}"
  echo "head=${head_sha}"
  echo "editable_path_count=${#paths[@]}"
  echo "=== git diff --stat over editablePaths only ==="
  git diff --stat "${merge_base}" "${head_sha}" -- "${paths[@]}"
  echo "=== changed scored paths ==="
  git diff --name-only "${merge_base}" "${head_sha}" -- "${paths[@]}"
  echo "=== full diff ==="
  git diff "${merge_base}" "${head_sha}" -- "${paths[@]}"
} > "${out}" 2>&1

echo "wrote ${out}"
sed -n '1,40p' "${out}"
