#!/usr/bin/env bash
# Run `swift test` on this branch and on the assignment base, so the failing
# set can be compared instead of asserted.
#
# The base arm builds in a throwaway git worktree outside the repository, so
# neither `.build` tree contaminates the other.
#
#   research/e70_test_control.sh <BASE_SHA>
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

(($# == 1)) || { echo "usage: e70_test_control.sh <BASE_SHA>" >&2; exit 2; }
base_sha="$1"

out="research/out/e70-tests"
mkdir -p "${out}"

echo "e70_test_control: head arm $(git rev-parse HEAD)"
swift test --force-resolved-versions > "${out}/head.log" 2>&1
echo "e70_test_control: head exit=$?"

control_dir="${TMPDIR:-/tmp}/e70-base-control"
git worktree remove --force "${control_dir}" 2>/dev/null
rm -rf "${control_dir}"

echo "e70_test_control: base arm ${base_sha}"
if git worktree add --detach "${control_dir}" "${base_sha}" \
  > "${out}/base-worktree.log" 2>&1
then
  ( cd "${control_dir}" \
    && swift test --force-resolved-versions ) > "${out}/base.log" 2>&1
  echo "e70_test_control: base exit=$?"
  git worktree remove --force "${control_dir}"
else
  echo "e70_test_control: base worktree FAILED"
  tail -20 "${out}/base-worktree.log"
fi

echo "e70_test_control: logs in ${out}"
exit 0
