#!/usr/bin/env bash
# E188 stage 2, advisor-requested diagnostic: time `clearRecurrentRollback`
# directly and count how much recurrent rollback state it releases.
#
#   usage: research/e188_clear_timer_leg.sh TAG TOKENS
#
# The advisor asked to separate two explanations of the full-acceptance commit
# phase rising from ~180 us in rounds 1-2 to ~440 us from round 4:
#
#   (a) the commit phase pays to RELEASE populated recurrent rollback state
#       that does not exist before the first prefix rejection; or
#   (b) the commit phase became more expensive for an unrelated reason, such
#       as an allocator or free-list change, and the release is incidental.
#
# The cheapest discriminator is pure instrumentation, so this arm changes no
# behaviour at all. It adds a `DispatchTime` pair around the single
# `clearRecurrentRollback` call site and makes that function return the number
# of MLXArray references it dropped. Under (a), `clear_release_us` must carry most of
# the ~260 us step and `clear_release_count` must be 0 in rounds 1-2 and positive later.
# Under (b), `clear_release_us` stays small while `commit_us` still steps up.
#
# The patch is applied to the worktree, built, measured, and then reverted on
# every exit path. Nothing is committed. The build certificate requires the
# new `clear_us` trace key, so the measured worker provably carries the arm.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tag="${1:?usage: e188_clear_timer_leg.sh TAG TOKENS}"
tokens="${2:?usage: e188_clear_timer_leg.sh TAG TOKENS}"

FILE="Sources/MLXFastModel/Qwen36MTPBlockSession.swift"

restore() {
  git restore --source=HEAD --staged --worktree -- "${FILE}"
  echo "e188_clear_timer_leg: restored ${FILE} to HEAD content"
  if [[ "$(git rev-parse ":${FILE}")" != "$(git rev-parse "HEAD:${FILE}")" ]]; then
    echo "e188_clear_timer_leg: FATAL restore failed" >&2
  fi
  git status --short -- "${FILE}"
}
trap restore EXIT

python3 - "${FILE}" <<'PY' || exit 1
import sys

path = sys.argv[1]
src = open(path).read()

edits = [
    # 1. a slot for the measured clear duration and the released-reference count
    ("        var tCommitDone: UInt64 = 0\n",
     "        var tCommitDone: UInt64 = 0\n"
     "        var tClearNs: UInt64 = 0\n"
     "        var nClearReleased: Int = 0\n"),

    # 2. time the single call site inside the full-acceptance branch
    ("            Self.clearRecurrentRollback(cache)\n",
     "            let tClear0 = Self.traceRounds\n"
     "                ? DispatchTime.now().uptimeNanoseconds : 0\n"
     "            nClearReleased = Self.clearRecurrentRollback(cache)\n"
     "            if Self.traceRounds {\n"
     "                tClearNs = DispatchTime.now().uptimeNanoseconds - tClear0\n"
     "            }\n"),

    # 3. count the references dropped, without changing what is dropped
    ("    private static func clearRecurrentRollback(_ cache: [any KVCache]) {\n"
     "        for entry in cache {\n"
     "            if let arrays = entry as? ArraysCache {\n"
     "                arrays.rollbackState = nil\n"
     "                arrays.rollbackCheckpoints = []\n"
     "                arrays.prefixReplayTape = nil\n"
     "            }\n"
     "        }\n"
     "    }\n",
     "    @discardableResult\n"
     "    private static func clearRecurrentRollback(_ cache: [any KVCache]) -> Int {\n"
     "        var released = 0\n"
     "        for entry in cache {\n"
     "            if let arrays = entry as? ArraysCache {\n"
     "                if arrays.rollbackState != nil { released += 1 }\n"
     "                released += arrays.rollbackCheckpoints.count\n"
     "                if arrays.prefixReplayTape != nil { released += 1 }\n"
     "                arrays.rollbackState = nil\n"
     "                arrays.rollbackCheckpoints = []\n"
     "                arrays.prefixReplayTape = nil\n"
     "            }\n"
     "        }\n"
     "        return released\n"
     "    }\n"),

    # 4. publish both in the existing trace line
    ('                + "commit_us=\\((tCommitDone - tReadDone) / 1000) "\n',
     '                + "commit_us=\\((tCommitDone - tReadDone) / 1000) "\n'
     '                + "clear_release_us=\\(tClearNs / 1000) "\n'
     '                + "clear_release_count=\\(nClearReleased) "\n'),
]

for old, new in edits:
    if src.count(old) != 1:
        print(f"e188_clear_timer_leg: expected exactly one match for:\n{old}",
              file=sys.stderr)
        sys.exit(1)
    src = src.replace(old, new)

open(path, "w").write(src)
print("e188_clear_timer_leg: applied 4 instrumentation edits")
PY

senpai/rebuild-and-assert-worker.sh --require "clear_release_us=" --require "clear_release_count=" || exit 1

research/e90_leg.sh "${tag}" "${tokens}"
status=$?

{
  echo "e188_arm=clear-timer-instrumented"
  echo "e188_arm_behaviour_change=none"
} >> "research/out/${tag}/meta.txt"

exit "${status}"
