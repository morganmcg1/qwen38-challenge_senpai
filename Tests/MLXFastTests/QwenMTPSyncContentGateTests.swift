import Foundation
import Testing

/// THE SYNC CONTENT GATE (E26, PR #33 item v).
///
/// `QwenMTPFixedWindowSourceGuardTests` already asserts the shape of the
/// WORKING TREE, and that is necessary but not sufficient. Two gaps remain, and
/// this suite exists for exactly those two:
///
///  1. A working-tree assertion can only speak about the tree it is run in. The
///     fixed-window fix has now been discarded by a content-mirror overlay five
///     times (f1a874d, 330b44e, b219009, bc552e5, 006a369) and every one of
///     those regressions was invisible to branch bookkeeping: the merge commit
///     and the branch name both survived while the file content reverted.
///     `research/e26-content-gate.sh` inspects an arbitrary revision, raw blob,
///     or candidate file, so an incoming sync can be judged BEFORE it is
///     merged, and a blob argument still resolves when the ref that carried it
///     was never fetched -- the normal situation in a student checkout.
///
///  2. A guard that has quietly stopped detecting anything passes forever. The
///     previous revision of the fixed-window guard recorded the DEFECT's shape
///     and therefore passed at the base by design; it was documentation, not a
///     detector. So the gate here is exercised in both polarities, and the
///     negative polarity is synthesised rather than borrowed, so it cannot
///     decay into a no-op.
///
/// Writing this suite immediately earned its keep: the gate's first normaliser
/// squeezed whitespace RUNS, and `stopTokens .contains( primary )` walked
/// straight through it. It now deletes whitespace outright.
@Suite
struct QwenMTPSyncContentGateTests {
    static let gatePath = "research/e26-content-gate.sh"
    static let sessionPath = "Sources/MLXFastModel/Qwen36MTPBlockSession.swift"

    /// The overlay blob that shipped the truncating session. Spelled as a
    /// literal because it is the only stable name for content that no live ref
    /// points at any more.
    static let overlayBlob = "0f41bbf904d09c28e93736217fd90729ba0636e7"

    /// Each of these, present, must fail the gate on its own. They are appended
    /// to an otherwise-correct session: the gate is a content check, so position
    /// is irrelevant and the mutated text never has to compile. The spacing is
    /// deliberately hostile, and the third is wrapped across lines, because a
    /// reformat is a realistic way for the defect to return.
    static let defectMarkers = [
        "if  self.stopTokens .contains( primary ) {",
        "public private(set) var reachedStopToken = false",
        "if let i = committed.firstIndex(where: {\n  stopTokens.contains($0)\n}) {",
    ]

    /// ...and each of these, absent, must also fail it. The gate has to notice
    /// the fix being REMOVED, not only the defect being added: truncating the
    /// PROPOSAL at a stop token is correct and required, and `pendingPrimary` is
    /// the carry that lets a committed stop token span rounds instead of ending
    /// the window.
    static let requiredMarkers = [
        "stopTokens.contains(drafts[",
        "pendingPrimary",
    ]

    // MARK: - Positive polarity

    /// The committed session passes. This is the claim a frontier sync breaks.
    @Test
    func theGatePassesOnTheCommittedSession() throws {
        let run = try Self.runGate(["HEAD"])
        #expect(
            run.status == 0,
            "gate rejected HEAD\n\(run.stdout)\n\(run.stderr)")
        #expect(run.stdout.contains("PASS"))
        #expect(!run.stdout.contains("FAIL"))
    }

    /// ...and so does the working tree, which is what an uncommitted overlay
    /// application would damage first.
    @Test
    func theGatePassesOnTheWorkingTree() throws {
        let run = try Self.runGate([Self.sessionPath])
        #expect(
            run.status == 0,
            "gate rejected the working tree\n\(run.stdout)\n\(run.stderr)")
    }

    // MARK: - Negative polarity

    /// THE HISTORICAL CONTROL. Skipped rather than failed when the object is
    /// not in this store: a checkout that never fetched the overlay is legal,
    /// and the synthesised controls below already guarantee polarity coverage.
    /// A skip here is not a pass.
    ///
    /// This assertion is expected to keep failing the overlay forever. If
    /// campaign main is ever repaired the blob does not change -- content is
    /// immutable -- so this stays valid regardless.
    @Test(.enabled(if: QwenMTPSyncContentGateTests.overlayBlobIsPresent))
    func theGateRejectsTheHistoricalOverlayBlob() throws {
        let run = try Self.runGate([Self.overlayBlob])
        #expect(
            run.status == 1,
            "expected the overlay blob to be rejected\n\(run.stdout)\n\(run.stderr)")
        #expect(run.stdout.contains("FAIL"))
        #expect(
            run.stderr.contains("E26 / PR #33"),
            "a failing gate must name the campaign item that owns the rule")
    }

    /// THE SYNTHESISED CONTROLS. Each case first proves the unmutated copy
    /// passes through the same temporary-file path, so a failure is attributable
    /// to the mutation and not to the mechanism.
    @Test(arguments: defectMarkers)
    func theGateRejectsAReintroducedDefectMarker(_ marker: String) throws {
        let source = try Self.committedSession()
        try Self.expectGatePasses(on: source, "the unmutated control")
        try Self.expectGateFails(on: source + "\n" + marker + "\n", marker)
    }

    @Test(arguments: requiredMarkers)
    func theGateRejectsADeletedFixMarker(_ marker: String) throws {
        let source = try Self.committedSession()
        try Self.expectGatePasses(on: source, "the unmutated control")
        try #require(
            source.contains(marker),
            "the session no longer contains \(marker); the gate's premise moved")
        try Self.expectGateFails(
            on: source.replacingOccurrences(of: marker, with: "XX_DELETED_XX"),
            "deletion of \(marker)")
    }

    // MARK: - Support

    private struct Run {
        var status: Int32
        var stdout: String
        var stderr: String
    }

    private static let repositoryRoot = FileManager.default.currentDirectoryPath

    static let overlayBlobIsPresent: Bool = {
        guard
            let run = try? shell(
                "/usr/bin/git", ["cat-file", "-t", overlayBlob])
        else { return false }
        return run.status == 0
            && run.stdout.trimmingCharacters(in: .whitespacesAndNewlines) == "blob"
    }()

    private static func committedSession() throws -> String {
        let run = try shell(
            "/usr/bin/git", ["cat-file", "-p", "HEAD:\(sessionPath)"])
        try #require(run.status == 0, "cannot read HEAD:\(sessionPath)")
        return run.stdout
    }

    private static func expectGatePasses(on source: String, _ what: String) throws {
        let run = try runGate(on: source)
        #expect(
            run.status == 0,
            "\(what) should have passed\n\(run.stdout)\n\(run.stderr)")
    }

    private static func expectGateFails(on source: String, _ what: String) throws {
        let run = try runGate(on: source)
        #expect(
            run.status == 1,
            "\(what) slipped past the gate\n\(run.stdout)\n\(run.stderr)")
        #expect(run.stdout.contains("FAIL"))
    }

    private static func runGate(on source: String) throws -> Run {
        let file = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("e26-gate-\(UUID().uuidString).swift")
        try source.write(to: file, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: file) }
        return try runGate([file.path])
    }

    private static func runGate(_ arguments: [String]) throws -> Run {
        try shell("/bin/bash", ["\(repositoryRoot)/\(gatePath)"] + arguments)
    }

    private static func shell(_ tool: String, _ arguments: [String]) throws -> Run {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: tool)
        process.arguments = arguments
        process.currentDirectoryURL = URL(fileURLWithPath: repositoryRoot)
        let out = Pipe()
        let err = Pipe()
        process.standardOutput = out
        process.standardError = err
        try process.run()
        // Both streams are a few hundred bytes, well inside one pipe buffer, so
        // draining them in sequence cannot deadlock.
        let stdoutData = out.fileHandleForReading.readDataToEndOfFile()
        let stderrData = err.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        return Run(
            status: process.terminationStatus,
            stdout: String(decoding: stdoutData, as: UTF8.self),
            stderr: String(decoding: stderrData, as: UTF8.self))
    }
}
