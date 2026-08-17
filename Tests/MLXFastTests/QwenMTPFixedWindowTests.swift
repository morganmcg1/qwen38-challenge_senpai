import Foundation
import Testing

@testable import MLXFastModel

// THE FIXED DECODE WINDOW, AND WHY THIS FILE KEEPS COMING BACK.
//
// The ranked leg decodes a FIXED count of parent-counted tokens (512). A stop
// token inside that window is DATA, not a terminator: the parent owns the
// window length, so emitting EOS at position 300 does not end the leg, it just
// puts token 151645 at position 300 and keeps going. The organizer's own
// trusted driver has always worked this way -- see the guard suite below, which
// pins the two lines in Sources/MLXFastTrustedHarness/QwenRuntimeMTPDriver.swift
// that own the length -- and benchmark.json states it in
// /scoring/mtpEmptyDraftRoundsLegalNote.
//
// THE OVERLAY IS GONE, AND WE DID NOT RESTORE IT.
//
// Campaign commit c8dceb9 imported the promoted submitted surface from
// d1530a409848b82a0a1890141c1483875d1e0173 -- the frontier that scores
// 3.13098700135133, the top of the board. That surface truncates at EOS:
// `reachedStopToken` is a tracked mutable flag, `stopTokens` is stored and
// consulted, and the pure prefix helper `acceptedDraftPrefixCount` no longer
// exists. All fourteen call sites in this file broke at once.
//
// We deliberately did NOT restore the overlay, because the frontier settled the
// question against it. A candidate scoring 3.131 on the official 512-token
// exact-match runner WITH truncation is decisive evidence that truncation
// passes the token-fidelity gate; post-EOS continuation was therefore never a
// correctness fix, only a performance-shaped bet that no ranked run has ever
// supported. Restoring it to keep a test compiling would be shipping an
// unvalidated perturbation to defend our own test file, which is backwards.
//
// So the accept RULE survives here as an executable specification (every depth,
// cost-curve and row-ledger argument the campaign makes is denominated in it),
// and the source guard below flips from an alignment claim into a TRIPWIRE that
// records what the shipped session actually does today. That turns a silent
// frontier removal into a checkable campaign fact: if a later sync re-adds the
// overlay, this file fails and the question is re-adjudicated on its merits
// instead of being inherited by accident.
//
// REVERT HISTORY. This literal has now been deleted four times, each time by a
// frontier-sync merge rather than by a decision:
//
//   f1a874d  qwen: continue fixed decode windows past EOS   (added)
//   330b44e  reverted f1a874d
//   b219009  qwen: continue fixed decode windows past EOS   (re-added)
//   bc552e5  "Retire the orphaned fixed-window EOS guard test" (deleted)
//
// The same merge that produced the current base ALSO silently deleted
// Tests/MLXFastTests/QwenQMVCostCurveTests.swift (722 lines at ef16dea4,
// absent afterwards) -- a delete/modify conflict resolved the wrong way. Test
// files are the campaign's memory; a sync that drops them is a defect, and the
// cheapest defence is that the deletion has to break something. Everything
// here is CPU-only: no model, no weights, no network, no GPU.

/// THE FIXED-WINDOW ACCEPT RULE, as an executable specification.
///
/// This used to be `Qwen36MTPBlockSession.acceptedDraftPrefixCount`. The rule
/// is unchanged and still describes what the shipped accept loop does for every
/// draft prefix that contains no stop token -- which, on the 512-token long-copy
/// goldens, is essentially all of them. Keeping it here means the properties
/// below still fail loudly if our MODEL of acceptance drifts, even though the
/// product no longer exposes the helper.
private func fixedWindowAcceptedPrefixCount(
    drafts: [Int], verifyArgmax: [Int]
) -> Int {
    precondition(verifyArgmax.count >= drafts.count)
    for index in drafts.indices where verifyArgmax[index] != drafts[index] {
        return index
    }
    return drafts.count
}

@Suite
struct QwenMTPFixedWindowTests {
    /// The original pair, restored verbatim from bc552e5^ so that a diff
    /// against any earlier incarnation of this file is empty.
    @Test
    func eosInsideAnAcceptedPrefixDoesNotEndTheWindow() {
        let eos = 151_645
        let drafts = [41, eos, 73, 89]
        let targetTokens = [41, eos, 73, 97, 101]

        let accepted = fixedWindowAcceptedPrefixCount(
            drafts: drafts, verifyArgmax: targetTokens)
        let committed = [13] + Array(drafts.prefix(accepted))

        #expect(accepted == 3)
        #expect(Array(committed) == [13, 41, eos, 73])
        #expect(accepted + (drafts.count - accepted) + 1 == drafts.count + 1)
    }

    @Test
    func onlyATargetMismatchEndsTheAcceptedPrefix() {
        let eos = 151_645
        let drafts = [41, 67, eos]
        let targetTokens = [41, 71, eos, 101]

        #expect(
            fixedWindowAcceptedPrefixCount(
                drafts: drafts, verifyArgmax: targetTokens) == 1)
    }

    /// An INDEPENDENT reference implementation of longest-common-prefix,
    /// written so it cannot share a bug with the shipped loop, plus the EOS
    /// invariant stated as an experiment: inject a stop token at EVERY position
    /// of both sequences and the count must not move. Deterministic generator,
    /// so a failure is reproducible from the seed alone.
    @Test
    func theCountIsTheLongestCommonPrefixAndEosIsJustAToken() {
        func reference(_ drafts: [Int], _ verify: [Int]) -> Int {
            var count = 0
            while count < drafts.count, drafts[count] == verify[count] {
                count += 1
            }
            return count
        }
        // Small alphabet on purpose: it makes long agreeing prefixes common,
        // which is the regime the accept loop actually runs in.
        var state: UInt64 = 0x9E37_79B9_7F4A_7C15
        func next(_ bound: Int) -> Int {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Int((state >> 33) % UInt64(bound))
        }

        let eos = 151_645
        for _ in 0..<400 {
            let width = 1 + next(8)
            let drafts = (0..<width).map { _ in next(5) }
            // verifyArgmax carries one extra row: the next parent token.
            var verify = (0..<(width + 1)).map { _ in next(5) }
            // Agree on a random prefix so the interesting cases dominate.
            let agree = next(width + 1)
            for index in 0..<agree { verify[index] = drafts[index] }

            let expected = reference(drafts, verify)
            #expect(
                fixedWindowAcceptedPrefixCount(
                    drafts: drafts, verifyArgmax: verify) == expected)

            // Token IDENTITY is irrelevant: substituting EOS at any single
            // position, on either side or both, changes the count only through
            // the equality it changes -- never because it is EOS.
            for position in 0..<width {
                var draftsWithEOS = drafts
                var verifyWithEOS = verify
                draftsWithEOS[position] = eos
                verifyWithEOS[position] = eos
                #expect(
                    fixedWindowAcceptedPrefixCount(
                        drafts: draftsWithEOS, verifyArgmax: verifyWithEOS)
                        == reference(draftsWithEOS, verifyWithEOS))

                var draftsOnly = drafts
                draftsOnly[position] = eos
                #expect(
                    fixedWindowAcceptedPrefixCount(
                        drafts: draftsOnly, verifyArgmax: verify)
                        == reference(draftsOnly, verify))
            }
        }
    }

    /// Both extremes are reachable, including the empty draft round that
    /// /scoring/mtpEmptyDraftRoundsLegal declares legal.
    @Test
    func fullAcceptanceAndZeroAcceptanceAreBothReachable() {
        let eos = 151_645

        let all = [7, 8, eos, 9]
        #expect(
            fixedWindowAcceptedPrefixCount(
                drafts: all, verifyArgmax: all + [10]) == all.count)

        // An EOS-only draft that the target agrees with is FULLY accepted.
        #expect(
            fixedWindowAcceptedPrefixCount(
                drafts: [eos], verifyArgmax: [eos, 11]) == 1)

        // A mismatch at position 0 accepts nothing, even when the draft is EOS.
        #expect(
            fixedWindowAcceptedPrefixCount(
                drafts: [eos, 8], verifyArgmax: [7, 8, 9]) == 0)
        #expect(
            fixedWindowAcceptedPrefixCount(
                drafts: [7, 8], verifyArgmax: [eos, 8, 9]) == 0)

        // A non-drafting round: zero drafts, zero accepted, and the round still
        // commits the parent token that the caller adds.
        #expect(
            fixedWindowAcceptedPrefixCount(
                drafts: [], verifyArgmax: [12]) == 0)
        #expect(
            fixedWindowAcceptedPrefixCount(
                drafts: [], verifyArgmax: []) == 0)
    }

    /// The normal shape is `verifyArgmax.count == drafts.count + 1`: the extra
    /// row is the parent's own next token and can never be accepted as a draft,
    /// so it must not influence the count -- not even when it is EOS.
    @Test
    func theExtraVerifyRowNeverChangesTheCount() {
        let eos = 151_645
        let drafts = [21, 22, 23]
        for tail in [24, eos, 21, 0] {
            #expect(
                fixedWindowAcceptedPrefixCount(
                    drafts: drafts, verifyArgmax: drafts + [tail]) == 3)
        }
        // Longer-than-normal verify rows are also harmless.
        #expect(
            fixedWindowAcceptedPrefixCount(
                drafts: drafts, verifyArgmax: drafts + [eos, eos, eos]) == 3)
    }

    /// THE ROW LEDGER. Every round commits `accepted + 1` parent-counted rows
    /// (the accepted draft prefix plus the parent token the verify pass proves),
    /// and the rejected tail is `drafts.count - accepted`. Those three numbers
    /// must close over the whole window for any draft/verify pair, because the
    /// ranked reverifier checks declared rows against emitted tokens exactly.
    @Test
    func theRowLedgerClosesOverAWholeWindow() {
        let eos = 151_645
        let windowTokens = 512
        var state: UInt64 = 0xD1B5_4A32_D192_ED03
        func next(_ bound: Int) -> Int {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Int((state >> 33) % UInt64(bound))
        }

        var committed = 0
        var rounds = 0
        var proposed = 0
        var accepted = 0
        var sawEOS = false

        while committed < windowTokens {
            let depth = next(9)  // 0...8, matching /scoring/mtpMaxDraftDepth
            let drafts = (0..<depth).map { _ in next(4) == 0 ? eos : next(50) }
            var verify = (0..<(depth + 1)).map { _ in next(4) == 0 ? eos : next(50) }
            let agree = next(depth + 1)
            for index in 0..<agree { verify[index] = drafts[index] }

            let take = fixedWindowAcceptedPrefixCount(
                drafts: drafts, verifyArgmax: verify)
            #expect(take <= depth)
            let round = Array(drafts.prefix(take)) + [verify[take]]
            #expect(round.count == take + 1)
            if round.contains(eos) { sawEOS = true }

            committed += round.count
            rounds += 1
            proposed += depth
            accepted += take
        }

        // A round always commits at least the parent token, so the window
        // terminates in at most `windowTokens` rounds and never stalls.
        #expect(rounds >= 1)
        #expect(rounds <= windowTokens)
        // Overshoot is bounded by the deepest round and is truncated by the
        // caller; it is never resolved by stopping early on a token value.
        #expect(committed >= windowTokens)
        #expect(committed <= windowTokens + 8)
        // The ledger identity.
        #expect(committed == accepted + rounds)
        #expect(proposed >= accepted)
        // The window really did contain stop tokens and ran past them.
        #expect(sawEOS)
    }
}

/// SOURCE-TEXT GUARDS. These assert the shape of the checked-in code rather
/// than its behaviour, which is the only way to catch a re-introduced early
/// return: a `break` on EOS would still pass every behavioural test above,
/// because it lives in the accept LOOP, not in the prefix helper.
@Suite
struct QwenMTPFixedWindowSourceGuardTests {
    private typealias S = DFlashGateTextSupport

    static let sessionPath = "Sources/MLXFastModel/Qwen36MTPBlockSession.swift"
    static let driverPath =
        "Sources/MLXFastTrustedHarness/QwenRuntimeMTPDriver.swift"

    /// TRIPWIRE, NOT AN ALIGNMENT CLAIM.
    ///
    /// Asserts what the shipped session does TODAY: it truncates at EOS, the
    /// same behaviour carried by the promoted 3.13098700135133 frontier. A
    /// failure here is not a defect -- it means a sync re-introduced the
    /// fixed-window overlay, so the post-EOS continuation question is live
    /// again and has to be argued from evidence rather than inherited.
    @Test
    func theEditableSessionTruncatesAtEosToday() throws {
        let session = try S.text(Self.sessionPath)

        // `reachedStopToken` is a tracked mutable flag, not a stored `false`.
        #expect(
            session.contains("public private(set) var reachedStopToken = false"))
        #expect(
            session.contains("reachedStopToken = true"),
            "the shipped session has a state in which a stop token was reached")
        #expect(
            !session.contains("public var reachedStopToken: Bool { false }"),
            "the overlay's constant-false accessor must stay absent")

        // `stopTokens` is stored and consulted, not accepted and dropped.
        #expect(
            !session.contains("stopTokens _: Set<Int>,"),
            "stopTokens must not be an ignored initialiser parameter")
        #expect(
            session.components(separatedBy: "stopTokens").count - 1 >= 2,
            "stopTokens must be consulted, not merely accepted")

        // And the overlay's identity-blind prefix helper went with it.
        #expect(
            !session.contains("static func acceptedDraftPrefixCount("),
            "the overlay's pure prefix helper must stay absent")
    }

    /// THE ALIGNMENT INVARIANT, and the reason this file is worth keeping. The
    /// TRUSTED driver -- which we may not edit -- owns the window length by
    /// token COUNT and has no stop-token concept, so the 512 parent-counted
    /// tokens are demanded regardless of what the editable session believes
    /// about EOS. This is the NON-EDITABLE enforcer, which is why it, and not
    /// any overlay, is what this suite pins.
    @Test
    func theTrustedDriverStillOwnsTheWindowLengthByCount() throws {
        let driver = try S.text(Self.driverPath)
        #expect(driver.contains("while emitted.count < options.totalTokenCount"))
        #expect(
            driver.contains(
                "let remaining = options.totalTokenCount - emitted.count"))
        #expect(
            driver.components(separatedBy: "reachedStopToken").count - 1 == 0,
            """
            the trusted driver has never had stop-token logic; if it grows \
            some, the editable overlay is no longer alignment
            """)
    }

    /// The manifest is the authority for the window itself. Read the numbers
    /// rather than restating them, so an organizer change fails loudly here
    /// instead of silently invalidating every result in research/.
    @Test
    func theManifestStillDeclaresTheFixedWindow() throws {
        let manifest = try S.json("benchmark.json")
        let scoring = try #require(manifest["scoring"] as? [String: Any])

        #expect((scoring["mode"] as? String) == "qwen-mtp-paired-decode-only")
        #expect((scoring["decodeTokens"] as? Int) == 512)
        #expect((scoring["mtpMaxDraftDepth"] as? Int) == 8)
        #expect((scoring["mtpEmptyDraftRoundsLegal"] as? Bool) == true)
        #expect((scoring["decodeSpeedupFloor"] as? Double) == 0.9)
        #expect((scoring["pairsPerPrompt"] as? Int) == 1)

        // The note is the organizer's own statement of the rule the overlay
        // implements; losing it would leave the overlay unattributed.
        let note = try #require(
            scoring["mtpEmptyDraftRoundsLegalNote"] as? String)
        #expect(!note.isEmpty)
    }

    /// THE DOCUMENTED CEILING MATCHES THE MANIFEST. The plausibility gate moved
    /// from 3.0 to 5.0 on 2026-08-17 and the campaign docs had to be corrected
    /// by hand; several research notes were still stale afterwards. Derive the
    /// number from benchmark.json and require the two governing documents to
    /// agree, so the next move corrects them or fails here.
    @Test
    func theDocumentedCeilingMatchesTheManifest() throws {
        let manifest = try S.json("benchmark.json")
        let scoring = try #require(manifest["scoring"] as? [String: Any])
        let ceiling = try #require(scoring["decodeSpeedupCeiling"] as? Double)
        #expect(ceiling > 1.0)

        // Accept either spelling of an integral ceiling: the docs may write
        // `5.0` or `5`, and which one is not the invariant.
        var spellings = [String(ceiling)]
        if ceiling == ceiling.rounded() {
            spellings.append(String(Int(ceiling)))
        }

        let agents = try S.text("AGENTS.md")
        #expect(
            spellings.contains(where: {
                agents.contains("`\($0)` plausibility ceiling")
            }),
            "AGENTS.md states a plausibility ceiling other than \(ceiling)")

        let program = try S.text("senpai/program.md")
        #expect(
            spellings.contains(where: {
                program.contains("plausibility gate at `\($0)`")
            }),
            "senpai/program.md states a plausibility gate other than \(ceiling)")
        #expect(
            spellings.contains(where: {
                program.contains("The `\($0)` gate is not an optimization target")
            }),
            "senpai/program.md's not-a-target sentence names a stale ceiling")
    }
}
