#!/usr/bin/env bash
# E205 stage-A instrumentation: manufacture rejection rounds on demand and
# price them at the round endpoint (RULE 394).
#
#   usage: research/e205_trunc_patch.sh apply
#          research/e205_trunc_patch.sh revert [SOURCE_REF]
#
# WHY A FORCED ARM. Every cost law this campaign owns was fit on rounds where
# `acc == d` (FINDING 484's declared limitation). Natural rejections on the
# public fixture are ~3 in 18 rounds, which cannot power a paired contrast in
# one 256-token leg. The arm below TRUNCATES the accepted prefix by `j` drafts
# on designated rounds: the walk still computes the true longest common prefix,
# and the round then commits `acc_true - j` of it.
#
# WHY IT STAYS EXACT. Truncation drops CORRECT drafts. It never changes which
# token the target produces at any position, because
#   * `pendingPrimary` is row `acceptedCount` of the verify output, which is the
#     target's own greedy continuation of the committed prefix -- with the
#     truncation applied, that row IS the first truncated draft, so the next
#     round commits it as its primary;
#   * `pendingHidden` is the trunk hidden of the same row; and
#   * the parent owns the decode window and counts emitted tokens, so a round
#     that commits fewer tokens only makes the leg use more rounds.
# `all_tokens_matched` must therefore stay true. This is a diagnostic arm and
# NEVER a candidate: it deliberately throws away accepted tokens.
#
# WHY IT IS THE SAME CODE PATH A GENUINE REJECTION DRIVES. The repair site
# `Qwen36MTPBlockSession.swift:1778-1821` branches on `acceptedCount ==
# drafts.count` only. `restoreAfterPrefixReject` (:2059-2126) reads
# `acceptedCount`, `draftCount` and `committedOffset` and has no access to which
# drafts were correct, so a forced truncation and a genuine rejection with the
# same `(acc, d)` execute byte-identical work: the same `replayRecurrentPrefix`
# over `acc + 1` rows, the same attention-cache trims, the same E020 async
# prefetch, and the same generic-fallback branch when the preflight fails.
# Rejected state stays inaccessible exactly as today -- the tape, the rollback
# state and the checkpoints are all cleared inside the same restore.
#
# WHY THE POLICY STATE STAYS ON THE TRUE ACCEPTANCE. `recordAcceptOutcome` and
# `fullAcceptStreak` feed the depth schedule. Feeding them the truncated count
# would shorten drafts in the truncation arm and confound width with the effect
# under test, so both are fed `acceptedTrue`. The two arms therefore differ in
# the commit/repair path and in nothing else the schedule can see.
#
# INSTRUMENTS (all compiled into every arm at identical per-call cost, RULE
# 391(c); all read under the existing `MLX_QWEN_MTP_TRACE` gate):
#   repair_us / repair_cpu_us   wall and thread-CPU around the whole repair
#                               site. FINDING 528's discriminator: a site that
#                               waits on device work reads wall >> cpu.
#   repair_replay_us            `replayRecurrentPrefix` alone.
#   repair_trim_us              the attention-cache trim loop alone.
#   repair_prefetch_us          the E020 boundary-state compactMap + asyncEval.
#   repair_generic_us           the generic K>1 fallback, when it fires.
#   repair_path                 0 full acceptance, 1 fast restore, 2 generic.
#   clear_release_us / _cpu_us  the full-acceptance branch's own release, so
#                               the two branches are priced against each other.
#   acc_true / e205_trunc_req / e205_trunc_applied
#                               RULE 391(b) per-round arm witnesses, on the
#                               drafting line and on the serial line.
#
# The decision statistic is `round_us` (RULE 394); every timer above only
# attributes.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

mode="${1:?usage: e205_trunc_patch.sh apply|revert}"

SESSION_FILE="Sources/MLXFastModel/Qwen36MTPBlockSession.swift"

if [[ "${mode}" == "revert" ]]; then
  # `run_job` refuses a dirty worktree, so the instrumentation has to be
  # committed before a session. Revert restores from the assignment base, not
  # from HEAD; HEAD carries the instrumentation commit.
  source_ref="${2:-${BASE_SHA:-2a748cf054e495fe087815636424c7bf754acd24}}"
  git restore --source="${source_ref}" --staged --worktree -- "${SESSION_FILE}"
  echo "e205_trunc_patch: restored ${SESSION_FILE} to ${source_ref}"
  git --no-pager diff --stat "${source_ref}" -- "${SESSION_FILE}"
  exit 0
fi

if [[ "${mode}" != "apply" ]]; then
  echo "e205_trunc_patch: unknown mode '${mode}'" >&2
  exit 2
fi

python3 - "${SESSION_FILE}" <<'PY' || exit 1
import sys

session_path = sys.argv[1]

edits = [
    # 1. per-round slots for every instrument
    ("        var tCommitDone: UInt64 = 0\n",
     "        var tCommitDone: UInt64 = 0\n"
     "        var tRepairNs: UInt64 = 0\n"
     "        var cRepairNs: UInt64 = 0\n"
     "        var tGenericNs: UInt64 = 0\n"
     "        var tClearNs: UInt64 = 0\n"
     "        var cClearNs: UInt64 = 0\n"
     "        var e205RepairPath = 0\n"),

    # 2. the forced-truncation arm, immediately after the true accept walk
    ("        var perRowTop2Tokens: [[Int]] = []\n",
     "        // E205 forced-truncation arm. `acceptedTrue` is the real longest\n"
     "        // common prefix; the arm treats the last `j` accepted drafts as\n"
     "        // rejected so the repair path runs on demand. Correct drafts are\n"
     "        // dropped, never changed, so the emitted trajectory is identical\n"
     "        // and the leg simply uses more rounds. Diagnostic only.\n"
     "        let acceptedTrue = acceptedCount\n"
     "        let e205Requested = Self.e205TruncRequested(round: roundCount)\n"
     "        var e205Applied = 0\n"
     "        if e205Requested > 0, acceptedCount > 0 {\n"
     "            e205Applied = Swift.min(e205Requested, acceptedCount)\n"
     "            acceptedCount -= e205Applied\n"
     "        }\n"
     "\n"
     "        var perRowTop2Tokens: [[Int]] = []\n"),

    # 3. price the full-acceptance branch's own release against the repair
    ("            Self.clearRecurrentRollback(cache)\n"
     "            committed.append(contentsOf: drafts)\n",
     "            let tClear0 = DispatchTime.now().uptimeNanoseconds\n"
     "            let cClear0 = Self.threadCPUNanoseconds()\n"
     "            Self.clearRecurrentRollback(cache)\n"
     "            tClearNs = DispatchTime.now().uptimeNanoseconds - tClear0\n"
     "            cClearNs = Self.threadCPUNanoseconds() &- cClear0\n"
     "            committed.append(contentsOf: drafts)\n"),

    # 4. wall and thread-CPU clock around the whole repair site
    ("            let committedOffset = base + committed.count\n"
     "            if !Self.restoreAfterPrefixReject(\n"
     "                model, cache,\n"
     "                acceptedCount: acceptedCount, draftCount: draftCount,\n"
     "                to: committedOffset)\n"
     "            {\n",
     "            let committedOffset = base + committed.count\n"
     "            Self.e205ReplayNs = 0\n"
     "            Self.e205TrimNs = 0\n"
     "            Self.e205PrefetchNs = 0\n"
     "            let tRepair0 = DispatchTime.now().uptimeNanoseconds\n"
     "            let cRepair0 = Self.threadCPUNanoseconds()\n"
     "            let e205Restored = Self.restoreAfterPrefixReject(\n"
     "                model, cache,\n"
     "                acceptedCount: acceptedCount, draftCount: draftCount,\n"
     "                to: committedOffset)\n"
     "            tRepairNs = DispatchTime.now().uptimeNanoseconds - tRepair0\n"
     "            cRepairNs = Self.threadCPUNanoseconds() &- cRepair0\n"
     "            e205RepairPath = e205Restored ? 1 : 2\n"
     "            let tGeneric0 = DispatchTime.now().uptimeNanoseconds\n"
     "            if !e205Restored {\n"),

    # 5. close the generic-fallback window
    ("                perRowTop2Tokens[perRowTop2Tokens.count - 1] = ids\n"
     "                perRowTop2Logits[perRowTop2Logits.count - 1] = values\n"
     "            }\n",
     "                perRowTop2Tokens[perRowTop2Tokens.count - 1] = ids\n"
     "                perRowTop2Logits[perRowTop2Logits.count - 1] = values\n"
     "                tGenericNs =\n"
     "                    DispatchTime.now().uptimeNanoseconds - tGeneric0\n"
     "            }\n"),

    # 6. the depth schedule sees the TRUE acceptance in both arms
    ("        fullAcceptStreak =\n"
     "            acceptedCount == drafts.count ? fullAcceptStreak + 1 : 0\n"
     "        recordAcceptOutcome(acceptedCount: acceptedCount, drafts: drafts)\n",
     "        // Both fed the true walk outcome: the schedule must not see the\n"
     "        // forced truncation, or width would move with the arm.\n"
     "        fullAcceptStreak =\n"
     "            acceptedTrue == drafts.count ? fullAcceptStreak + 1 : 0\n"
     "        recordAcceptOutcome(acceptedCount: acceptedTrue, drafts: drafts)\n"),

    # 7. site sub-timers inside the fast restore
    ("            guard model.replayRecurrentPrefix(\n"
     "                cache: cache, committedRows: acceptedCount + 1)\n"
     "            else { return false }\n",
     "            let tReplay0 = DispatchTime.now().uptimeNanoseconds\n"
     "            let replayed = model.replayRecurrentPrefix(\n"
     "                cache: cache, committedRows: acceptedCount + 1)\n"
     "            e205ReplayNs = DispatchTime.now().uptimeNanoseconds - tReplay0\n"
     "            guard replayed else { return false }\n"
     "            let tTrim0 = DispatchTime.now().uptimeNanoseconds\n"),

    ("            for entry in cache where !(entry is ArraysCache) {\n"
     "                if entry.isTrimmable, entry.offset > committedOffset {\n"
     "                    _ = entry.trim(entry.offset - committedOffset)\n"
     "                }\n"
     "            }\n",
     "            for entry in cache where !(entry is ArraysCache) {\n"
     "                if entry.isTrimmable, entry.offset > committedOffset {\n"
     "                    _ = entry.trim(entry.offset - committedOffset)\n"
     "                }\n"
     "            }\n"
     "            e205TrimNs = DispatchTime.now().uptimeNanoseconds - tTrim0\n"
     "            let tPrefetch0 = DispatchTime.now().uptimeNanoseconds\n"),

    ("            asyncEval(replayedRecurrentStates)\n"
     "            return true\n",
     "            asyncEval(replayedRecurrentStates)\n"
     "            e205PrefetchNs =\n"
     "                DispatchTime.now().uptimeNanoseconds - tPrefetch0\n"
     "            return true\n"),

    # 8. the arm switch and the site-timer slots
    ("    private static func clearRecurrentRollback(_ cache: [any KVCache]) {\n",
     "    /// E205 forced-truncation arm. Open-loop by construction: the\n"
     "    /// schedule reads the round index and nothing else, so no measured\n"
     "    /// quantity can steer which rounds are truncated (RULE 392).\n"
     "    ///\n"
     "    ///   off      never truncate (the shipped behaviour)\n"
     "    ///   alt1     truncate 1 draft on odd rounds\n"
     "    ///   alt2     truncate 2 drafts on odd rounds\n"
     "    ///   alt4     truncate 4 drafts on odd rounds\n"
     "    ///   cycle8   1 / 2 / 4 drafts on rounds 1 / 3 / 5 mod 8\n"
     "    private static let e205TruncMode =\n"
     "        ProcessInfo.processInfo.environment[\"MLX_E205_TRUNC\"] ?? \"off\"\n"
     "\n"
     "    @inline(__always)\n"
     "    private static func e205TruncRequested(round: Int) -> Int {\n"
     "        switch e205TruncMode {\n"
     "        case \"alt1\": return round % 2 == 1 ? 1 : 0\n"
     "        case \"alt2\": return round % 2 == 1 ? 2 : 0\n"
     "        case \"alt4\": return round % 2 == 1 ? 4 : 0\n"
     "        case \"cycle8\":\n"
     "            switch round % 8 {\n"
     "            case 1: return 1\n"
     "            case 3: return 2\n"
     "            case 5: return 4\n"
     "            default: return 0\n"
     "            }\n"
     "        default: return 0\n"
     "        }\n"
     "    }\n"
     "\n"
     "    /// Site timers for the fast restore. Written by the static repair\n"
     "    /// helper, read once per round by the trace line.\n"
     "    nonisolated(unsafe) private static var e205ReplayNs: UInt64 = 0\n"
     "    nonisolated(unsafe) private static var e205TrimNs: UInt64 = 0\n"
     "    nonisolated(unsafe) private static var e205PrefetchNs: UInt64 = 0\n"
     "\n"
     "    private static func clearRecurrentRollback(_ cache: [any KVCache]) {\n"),

    # 9. publish every instrument and both witnesses on the drafting line
    ('                + "commit_us=\\((tCommitDone - tReadDone) / 1000) "\n',
     '                + "commit_us=\\((tCommitDone - tReadDone) / 1000) "\n'
     '                + "acc_true=\\(acceptedTrue) "\n'
     '                + "e205_trunc_req=\\(e205Requested) "\n'
     '                + "e205_trunc_applied=\\(e205Applied) "\n'
     '                + "repair_path=\\(e205RepairPath) "\n'
     '                + "repair_us=\\(tRepairNs / 1000) "\n'
     '                + "repair_cpu_us=\\(cRepairNs / 1000) "\n'
     '                + "repair_replay_us=\\(Self.e205ReplayNs / 1000) "\n'
     '                + "repair_trim_us=\\(Self.e205TrimNs / 1000) "\n'
     '                + "repair_prefetch_us=\\(Self.e205PrefetchNs / 1000) "\n'
     '                + "repair_generic_us=\\(tGenericNs / 1000) "\n'
     '                + "clear_release_us=\\(tClearNs / 1000) "\n'
     '                + "clear_release_cpu_us=\\(cClearNs / 1000) "\n'),

    # 10. the serial line carries the same witnesses, so one parser reads both
    ('                        + "serial_body=1\\n")\n',
     '                        + "acc_true=0 "\n'
     '                        + "e205_trunc_req='
     '\\(Self.e205TruncRequested(round: roundCount)) "\n'
     '                        + "e205_trunc_applied=0 repair_path=0 "\n'
     '                        + "repair_us=0 repair_cpu_us=0 "\n'
     '                        + "repair_replay_us=0 repair_trim_us=0 "\n'
     '                        + "repair_prefetch_us=0 repair_generic_us=0 "\n'
     '                        + "clear_release_us=0 clear_release_cpu_us=0 "\n'
     '                        + "serial_body=1\\n")\n'),
]

src = open(session_path).read()
for old, new in edits:
    if src.count(old) != 1:
        print(f"e205_trunc_patch: expected exactly one match in {session_path} "
              f"for:\n{old}", file=sys.stderr)
        sys.exit(1)
    src = src.replace(old, new)
open(session_path, "w").write(src)
print(f"e205_trunc_patch: applied {len(edits)} edits to {session_path}")
PY

git --no-pager diff --stat -- "${SESSION_FILE}"
