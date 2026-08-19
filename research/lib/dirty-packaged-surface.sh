#!/usr/bin/env bash
# Shared refusal: never certify a packaged surface you have not committed.
#
# WHY THIS EXISTS (ledger 162, found TWICE).
#
# Every surface gate in research/ answers its question with a git comparison of
# the form `<pinned baseline> .. <rev>`, and `<rev>` defaults to HEAD. That means
# an UNCOMMITTED edit to a packaged file is invisible to the gate. The state in
# which that matters is exactly the state you are in while reconciling with the
# organizer's tip: I reverted E27 in the worktree, re-ran the gates, and they
# certified the OLD tree while printing the new tree's repo root.
#
# I fixed it inline in research/frontier-revert-gate.sh and assumed the class was
# closed. It was not. Writing a MUTATION CONTROL for the shipped-surface gate's
# twin check -- append one line of code to quantized.h, demand the gate fail --
# showed the control could not fire at all, because the gate was still reading
# the commit. The control found the defect the fix had missed in a sibling.
#
# So the refusal lives here once, and the surface gates share it. A gate that
# grades a commit while you are about to submit a worktree is not a gate, it is
# a decoration.
#
# Usage, after cd'ing to the repository root:
#     . research/lib/dirty-packaged-surface.sh
#     refuse_if_packaged_surface_dirty "${REV}" "shipped-surface gate" || exit 1

# Args: $1 = rev under test, $2 = gate name for the message.
# Returns 0 if it is safe to proceed, 1 if the gate must refuse.
refuse_if_packaged_surface_dirty() {
  local rev="${1:-HEAD}" gate="${2:-gate}"
  local rev_sha head_sha paths dirty

  # Dirtiness only matters when the gate is grading the CURRENT checkout. A gate
  # deliberately pointed at some historical rev is answering a question about
  # that rev, and the worktree is irrelevant to it.
  rev_sha="$(git rev-parse "${rev}" 2>/dev/null || true)"
  head_sha="$(git rev-parse HEAD 2>/dev/null || true)"
  if [ -z "${rev_sha}" ] || [ "${rev_sha}" != "${head_sha}" ]; then
    return 0
  fi

  # The packaged set is whatever benchmark.json says it is. Deriving it rather
  # than hardcoding it is the point: the set has changed twice this campaign, and
  # a hardcoded list would have quietly stopped covering the files that moved.
  paths="$(python3 -c '
import json, shlex
paths = json.load(open("benchmark.json")).get("editablePaths") or []
print(" ".join(shlex.quote(p) for p in paths))
' 2>/dev/null || true)"
  if [ -z "${paths}" ]; then
    printf 'FAIL: %s: cannot read editablePaths from benchmark.json, so the packaged\n' "${gate}" >&2
    printf '      surface is unknown and "clean" cannot be established. Failing closed.\n' >&2
    return 1
  fi

  # Intentionally unquoted: python emitted a shell-quoted, space-joined list.
  dirty="$(git status --porcelain -- ${paths} 2>/dev/null || true)"
  if [ -n "${dirty}" ]; then
    printf 'FAIL: %s: packaged paths have UNCOMMITTED changes, and every comparison\n' "${gate}" >&2
    printf '      this gate makes is against %s, so those changes are NOT examined:\n' "${rev}" >&2
    printf '%s\n' "${dirty}" | sed 's/^/        /' >&2
    printf '      Commit or stash them and re-run. Do not read the report as\n' >&2
    printf '      covering the tree you are holding.\n' >&2
    return 1
  fi
  return 0
}
