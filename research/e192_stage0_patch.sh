#!/usr/bin/env bash
# E192 stage 0 instrumentation: apply or revert the diagnostic edits that
# decide WHY `clearRecurrentRollback` costs 397 us per full-acceptance round
# (FINDING 495) before any fix is built.
#
#   usage: research/e192_stage0_patch.sh apply
#          research/e192_stage0_patch.sh revert
#
# Three accounts are on the table. The first two are the pair E188 pre-designed
# a barrier leg for; the third is a structural account this session adds after
# reading the tape source.
#
#   (a) IN-FLIGHT REFERENCE. Dropping the last host reference waits on device
#       work still in flight over the recurrent boundary. A persistent in-place
#       slot inherits the same dependency, so it removes nothing.
#   (b) ALLOCATOR FREE. The cost is the free itself. `MetalAllocator::free`
#       (Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/allocator.cpp:233)
#       recycles to the buffer cache below `max_pool_size_` and otherwise pays
#       `residency_set_.erase` plus a real `buf->release()`. That branch is a
#       one-time regime step, which matches FINDING 495's 2.53x jump.
#   (c) RELOCATION, NOT REMOVAL. The 48 references are the GDN verify forward's
#       OWN intermediates. `ArraysCache.PrefixReplayTape` holds convInput, q, k,
#       v, a, b, g, beta and the pre-verify fp32 SSM state, and the width-2 mid
#       kernel stores slices of `convInput` and of the mid output. Nothing is
#       allocated separately for rollback. Retaining the tape therefore only
#       DEFERS a free that the verify eval would otherwise pay. If that is the
#       whole story, releasing earlier moves the cost into the verify window
#       instead of deleting it, and no rollback-slot design can win the 397 us.
#
# Three instruments, one build, no behaviour change in the primary arm:
#
#   1. `clear_release_cpu_us` beside `clear_release_us`: thread CPU time and
#      wall time around the SAME release. Account (a) blocks, so wall >> cpu.
#      Account (b) is real host work, so wall is about cpu. This is a pure
#      read and changes nothing.
#   2. `MLX_E192_BARRIER=alt`, E188's pre-designed leg, as confirmation:
#      evaluate every array the clear is about to drop, timed separately, on
#      odd rounds only. Under (a) `clear_release_us` collapses after the
#      barrier; under (b) it is unchanged.
#   3. `MLX_E192_TAPE=alt`, the account-(c) probe: suppress the replay tape on
#      odd rounds so the same arrays are released inside the verify eval. The
#      reject path then falls back to the exact generic repair, which is why
#      this arm is a diagnostic and never a candidate. Endpoints are `round_us`
#      and `host_thread_cpu_ns` on full-acceptance rounds: a cost that is
#      removed lowers both, a cost that is relocated leaves the round flat.
#
# Both arms alternate WITHIN one process by round parity (RULE 388), so leg
# level thermal drift cannot produce the contrast. RULE 391(b) witnesses:
# `e192_barrier_arm` and `e192_tape_suppressed_arm` per round, plus the
# `clear_release_count` delta that only the tape arm can produce. RULE 391(c):
# both clocks and both counters are compiled into every arm at identical
# per-call cost; only the env value differs.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

mode="${1:?usage: e192_stage0_patch.sh apply|revert}"

SESSION_FILE="Sources/MLXFastModel/Qwen36MTPBlockSession.swift"
MODEL_FILE="Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift"

if [[ "${mode}" == "revert" ]]; then
  git restore --source=HEAD --staged --worktree -- "${SESSION_FILE}" "${MODEL_FILE}"
  echo "e192_stage0_patch: restored both files to HEAD content"
  git status --short -- "${SESSION_FILE}" "${MODEL_FILE}"
  exit 0
fi

if [[ "${mode}" != "apply" ]]; then
  echo "e192_stage0_patch: unknown mode '${mode}'" >&2
  exit 2
fi

python3 - "${SESSION_FILE}" "${MODEL_FILE}" <<'PY' || exit 1
import sys

session_path, model_path = sys.argv[1], sys.argv[2]

session_edits = [
    # 1. per-round slots for every instrument
    ("        var tCommitDone: UInt64 = 0\n",
     "        var tCommitDone: UInt64 = 0\n"
     "        var tClearNs: UInt64 = 0\n"
     "        var cClearNs: UInt64 = 0\n"
     "        var nClearReleased: Int = 0\n"
     "        var tClearBarrierNs: UInt64 = 0\n"
     "        var e192Barrier: Int = 0\n"),

    # 2. arm the tape probe for this round, beside the existing band-timer arm
    ("        if Qwen35BandTimer.enabled { Qwen35BandTimer.reset() }\n",
     "        if Qwen35BandTimer.enabled { Qwen35BandTimer.reset() }\n"
     "        Qwen35E192TapeArm.suppress =\n"
     "            Qwen35E192TapeArm.suppressed(round: roundCount)\n"),

    # 3. wall AND thread-CPU clock around the release, optional barrier first
    ("            Self.clearRecurrentRollback(cache)\n",
     "            e192Barrier = Self.e192BarrierOn(round: roundCount) ? 1 : 0\n"
     "            if e192Barrier == 1 {\n"
     "                let tBarrier0 = DispatchTime.now().uptimeNanoseconds\n"
     "                Self.e192EvalRollbackState(cache)\n"
     "                tClearBarrierNs =\n"
     "                    DispatchTime.now().uptimeNanoseconds - tBarrier0\n"
     "            }\n"
     "            let tClear0 = DispatchTime.now().uptimeNanoseconds\n"
     "            let cClear0 = Self.threadCPUNanoseconds()\n"
     "            nClearReleased = Self.clearRecurrentRollback(cache)\n"
     "            tClearNs = DispatchTime.now().uptimeNanoseconds - tClear0\n"
     "            cClearNs = Self.threadCPUNanoseconds() &- cClear0\n"),

    # 4. count the dropped references and add the barrier helpers
    ("    private static func clearRecurrentRollback(_ cache: [any KVCache]) {\n"
     "        for entry in cache {\n"
     "            if let arrays = entry as? ArraysCache {\n"
     "                arrays.rollbackState = nil\n"
     "                arrays.rollbackCheckpoints = []\n"
     "                arrays.prefixReplayTape = nil\n"
     "            }\n"
     "        }\n"
     "    }\n",
     "    private static let e192BarrierMode =\n"
     "        ProcessInfo.processInfo.environment[\"MLX_E192_BARRIER\"] ?? \"0\"\n"
     "\n"
     "    private static func e192BarrierOn(round: Int) -> Bool {\n"
     "        switch e192BarrierMode {\n"
     "        case \"1\": return true\n"
     "        case \"alt\": return round % 2 == 1\n"
     "        default: return false\n"
     "        }\n"
     "    }\n"
     "\n"
     "    /// Bring every array the clear is about to drop to completed status,\n"
     "    /// so a release that waits on in-flight device work pays inside this\n"
     "    /// timed window instead of inside the clear.\n"
     "    private static func e192EvalRollbackState(_ cache: [any KVCache]) {\n"
     "        var pending: [MLXArray] = []\n"
     "        for entry in cache {\n"
     "            guard let arrays = entry as? ArraysCache else { continue }\n"
     "            if let state = arrays.rollbackState {\n"
     "                pending.append(state.0)\n"
     "                pending.append(state.1)\n"
     "            }\n"
     "            for checkpoint in arrays.rollbackCheckpoints {\n"
     "                pending.append(checkpoint.0)\n"
     "                pending.append(checkpoint.1)\n"
     "            }\n"
     "            if let tape = arrays.prefixReplayTape {\n"
     "                pending.append(contentsOf: [\n"
     "                    tape.convInput, tape.q, tape.k, tape.v,\n"
     "                    tape.a, tape.b, tape.g, tape.beta,\n"
     "                ])\n"
     "                if let ssmPre = tape.ssmPre { pending.append(ssmPre) }\n"
     "                if let mask = tape.mask { pending.append(mask) }\n"
     "            }\n"
     "        }\n"
     "        if !pending.isEmpty { eval(pending) }\n"
     "    }\n"
     "\n"
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

    # 5. publish every instrument and both arm witnesses in the trace line
    ('                + "commit_us=\\((tCommitDone - tReadDone) / 1000) "\n',
     '                + "commit_us=\\((tCommitDone - tReadDone) / 1000) "\n'
     '                + "clear_release_us=\\(tClearNs / 1000) "\n'
     '                + "clear_release_cpu_us=\\(cClearNs / 1000) "\n'
     '                + "clear_release_count=\\(nClearReleased) "\n'
     '                + "clear_barrier_us=\\(tClearBarrierNs / 1000) "\n'
     '                + "e192_barrier_arm=\\(e192Barrier) "\n'
     '                + "e192_tape_suppressed_arm='
     '\\(Qwen35E192TapeArm.suppress ? 1 : 0) "\n'),
]

model_edits = [
    ("public enum Qwen35BandTimer {\n",
     "/// E192 stage-0 diagnostic arm. `alt` suppresses the compact replay tape\n"
     "/// on odd rounds, so the verify forward's own intermediates are released\n"
     "/// inside the verify eval instead of in the next commit phase. That\n"
     "/// separates a RELOCATED release cost from a REMOVED one. The reject path\n"
     "/// then falls back to the exact generic repair, so this is a diagnostic\n"
     "/// arm and never a candidate.\n"
     "public enum Qwen35E192TapeArm {\n"
     "    public static let mode =\n"
     "        ProcessInfo.processInfo.environment[\"MLX_E192_TAPE\"] ?? \"on\"\n"
     "\n"
     "    public nonisolated(unsafe) static var suppress = false\n"
     "\n"
     "    public static func suppressed(round: Int) -> Bool {\n"
     "        switch mode {\n"
     "        case \"off\": return true\n"
     "        case \"alt\": return round % 2 == 1\n"
     "        default: return false\n"
     "        }\n"
     "    }\n"
     "}\n"
     "\n"
     "public enum Qwen35BandTimer {\n"),

    ("            pendingPrefixTape = tape\n",
     "            pendingPrefixTape = Qwen35E192TapeArm.suppress ? nil : tape\n"),
]

for path, edits in ((session_path, session_edits), (model_path, model_edits)):
    src = open(path).read()
    for old, new in edits:
        if src.count(old) != 1:
            print(f"e192_stage0_patch: expected exactly one match in {path} for:\n{old}",
                  file=sys.stderr)
            sys.exit(1)
        src = src.replace(old, new)
    open(path, "w").write(src)
    print(f"e192_stage0_patch: applied {len(edits)} edits to {path}")
PY

git --no-pager diff --stat -- "${SESSION_FILE}" "${MODEL_FILE}"
