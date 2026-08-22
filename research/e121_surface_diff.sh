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

# `mapfile` is bash 4 and macOS ships bash 3.2, and this git has neither
# `git status --pathspec-from-file` nor `git diff --pathspec-from-file`, so the
# path list goes through a NUL-delimited file and `xargs -0` expands it onto
# each command line.
spec="$(mktemp)"
trap 'rm -f "${spec}"' EXIT
count="$(python3 -c "
import json, sys
paths = json.load(open('benchmark.json'))['editablePaths']
sys.stderr.write('\0'.join(paths))
print(len(paths))
" 2> "${spec}")"
[[ "${count}" -gt 0 ]] || { echo "e121_surface_diff: no editablePaths" >&2; exit 2; }

git_over_paths() { xargs -0 git "$@" -- < "${spec}"; }

# Only a dirty SCORED path breaks the guarantee. Research files are never
# packed, so a dirty write-up must not block the check.
dirty="$(git_over_paths status --porcelain)"
if [[ -n "${dirty}" ]]; then
  echo "e121_surface_diff: a scored path is dirty; packed bytes would not be measured bytes" >&2
  echo "${dirty}" >&2
  exit 1
fi

merge_base="$(git merge-base HEAD "${submit_base}")" || exit 3
head_sha="$(git rev-parse HEAD)"

{
  echo "submit_base=${submit_base}"
  echo "merge_base=${merge_base}"
  echo "head=${head_sha}"
  echo "editable_path_count=${count}"
  echo "=== git diff --stat over editablePaths only ==="
  git_over_paths diff --stat "${merge_base}" "${head_sha}"
  echo "=== changed scored paths ==="
  git_over_paths diff --name-only "${merge_base}" "${head_sha}"
  echo "=== full diff ==="
  git_over_paths diff "${merge_base}" "${head_sha}"
} > "${out}" 2>&1

echo "wrote ${out}"
sed -n '1,40p' "${out}"
