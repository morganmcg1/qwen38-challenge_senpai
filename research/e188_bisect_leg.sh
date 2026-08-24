#!/usr/bin/env bash
# E188 stage 2: one commit-phase bisection point.
#
#   usage: research/e188_bisect_leg.sh REV TAG TOKENS
#
# E185 measured the per-round commit phase growing about 2.5x over the
# campaign with the GPU 100.00 % idle throughout. `e188_commit_phase_bisect.py`
# showed the window body and every function it calls live in ONE file,
# `Qwen36MTPBlockSession.swift`, and change at only three merges. So a
# bisection point does not need a whole historical tree: it needs that one file
# at the historical revision, on top of the current base.
#
# That is also the better contrast. A full historical checkout would move ~200
# unrelated changes at the same time; swapping one file holds everything else
# at the current base and isolates the host-side commit phase.
#
# The tree is restored on every exit path, including failure, so no
# scored-surface edit ever survives this script. Nothing here is committed.
#
# The arm certificate is the `prefetchHeadStep` symbol: revisions from the E165
# merge b51f893a onward carry it and earlier ones do not, so the built worker
# proves which arm actually ran. `restoreAfterPrefixReject` is the positive
# control that the symbol table was read at all.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

rev="${1:?usage: e188_bisect_leg.sh REV TAG TOKENS}"
tag="${2:?usage: e188_bisect_leg.sh REV TAG TOKENS}"
tokens="${3:?usage: e188_bisect_leg.sh REV TAG TOKENS}"

FILE="Sources/MLXFastModel/Qwen36MTPBlockSession.swift"

# `git checkout REV -- FILE` writes the historical blob to the index as well as
# the worktree, so the restore must name HEAD as the source for both. A plain
# `git checkout -- FILE` only refills the worktree from the index and would
# leave the scored file staged.
restore() {
  git restore --source=HEAD --staged --worktree -- "${FILE}"
  echo "e188_bisect_leg: restored ${FILE} to HEAD content"
  if [[ "$(git rev-parse ":${FILE}")" != "$(git rev-parse "HEAD:${FILE}")" ]]; then
    echo "e188_bisect_leg: FATAL restore failed, ${FILE} still differs from HEAD" >&2
  fi
  git status --short -- "${FILE}"
}
trap restore EXIT

blob_head="$(git rev-parse "HEAD:${FILE}")"
blob_rev="$(git rev-parse "${rev}:${FILE}")"
echo "e188_bisect_leg: rev=${rev} tag=${tag} tokens=${tokens}"
echo "e188_bisect_leg: HEAD blob=${blob_head}"
echo "e188_bisect_leg: arm  blob=${blob_rev}"

# The E165 merge introduced the prefetch; before it the symbol must be absent.
if git show "${rev}:${FILE}" | grep -q "func prefetchHeadStep"; then
  arm_assert=(--require-symbol prefetchHeadStep)
  echo "e188_bisect_leg: arm carries the prefetch (post-E165)"
else
  arm_assert=(--forbid-symbol prefetchHeadStep)
  echo "e188_bisect_leg: arm has no prefetch (pre-E165)"
fi

if [[ "${blob_rev}" != "${blob_head}" ]]; then
  git checkout "${rev}" -- "${FILE}" || exit 1
fi

senpai/rebuild-and-assert-worker.sh \
  "${arm_assert[@]}" --require-symbol restoreAfterPrefixReject || exit 1

research/e90_leg.sh "${tag}" "${tokens}"
status=$?

out="research/out/${tag}"
{
  echo "e188_bisect_rev=${rev}"
  echo "e188_bisect_blob=${blob_rev}"
  echo "e188_bisect_head_blob=${blob_head}"
} >> "${out}/meta.txt"

exit "${status}"
