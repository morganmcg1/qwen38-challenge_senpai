import Foundation
import Testing

// WHY THIS FILE EXISTS.
//
// `Qwen36MTPBlockSession` sizes its wired-residency ticket once, from the
// allocator's active byte count at one instant after the shape warm, and then
// never resizes it. Everything the scored window allocates afterwards has to
// fit inside a fixed slack allowance or the driver leaves it on the unwired
// path. The allowance is one integer literal, and the value that is correct is
// decided by three measured bounds that no compiler and no other test checks.
//
// E130 rung 9 measured the failure directly: the shipped 64 MiB allowance was
// short of the persistent post-sizing growth by 154.71 MiB, so part of the
// tower could not stay resident and which part missed varied between runs.
// The bounds below are the arithmetic that replaced 64 with 512. They are
// cheap text and integer checks, so this class of regression does not need a
// model, a GPU, or a timed leg to catch — which is the whole point.
//
// Every bound has a POSITIVE CONTROL: a slack value that must fail it. A bound
// that cannot fail is not a bound.

private let sessionSourcePath = "Sources/MLXFastModel/Qwen36MTPBlockSession.swift"

private func repositoryRoot() -> URL {
    var url = URL(fileURLWithPath: #filePath)
    for _ in 0 ..< 3 { url = url.deletingLastPathComponent() }
    return url
}

private func sessionSource() throws -> String {
    try String(
        contentsOf: repositoryRoot().appendingPathComponent(sessionSourcePath),
        encoding: .utf8)
}

private struct MissingDeclaration: Error, CustomStringConvertible {
    let name: String
    var description: String {
        "\(sessionSourcePath) no longer declares \(name)"
    }
}

/// Read the value assigned to one `private static let` in the session source.
///
/// The constants are private and `Int`/`Double` literals, so Swift inlines
/// them and they carry no symbol. Reading the declaration text is the only way
/// to pin them from a test.
private func declaredConstant(_ name: String) throws -> String {
    let needle = "static let \(name) = "
    let value = try sessionSource()
        .split(separator: "\n", omittingEmptySubsequences: false)
        .first { $0.contains(needle) }?
        .components(separatedBy: needle).last
    guard let value else { throw MissingDeclaration(name: name) }
    return value.trimmingCharacters(in: .whitespaces)
}

// MARK: - measured inputs

private let mib = 1 << 20
private let gib = 1 << 30

/// E130 rung 9, run `e130rung9`, artifact
/// `research/e130-artifacts/rung9-allocation-growth.json`. Persistent active
/// growth after the sizing instant over a 1024-token window, measured on seven
/// worker processes. MEASURED, not derived.
private let measuredPersistentGrowthMB = 218.71

/// E130 rung 10. `Memory.numResources` at the sizing instant, identical in
/// every probed process. MEASURED.
private let measuredResourceCountAtSizing = 4454

/// `ResidencySet` counts page-rounded allocation sizes, while the sizing input
/// counts buffer lengths, so every live buffer charges up to one short page of
/// rounding against the allowance. Apple Silicon uses 16 KiB pages for the
/// application and 4 KiB for some driver mappings; the expected case charges
/// half a 16 KiB page per buffer and the worst case charges a whole one.
/// DERIVED from the measured resource count, per Rule 89.
private let derivedPageTaxExpectedMB =
    Double(measuredResourceCountAtSizing) * 8.0 / 1024.0
private let derivedPageTaxWorstCaseMB =
    Double(measuredResourceCountAtSizing) * 16.0 / 1024.0

/// Live scratch peaks near this during the scored window. The allowance must
/// stay structurally below it: scratch that fails the fit test takes the
/// commit-free unwired path, and an allowance large enough to admit scratch
/// would put per-round allocation churn on the committing path instead.
/// MEASURED.
private let observedScratchFloorMB = 2370.0

/// This 48 GiB development host. Both MEASURED with `sysctl hw.memsize` and
/// `GPU.maxRecommendedWorkingSetBytes()`.
private let localPhysicalMemoryBytes = 51_539_607_552.0
private let localRecommendedWorkingSetBytes = 40_200_896_512.0

/// The ranked M5 host. Its memory size is a contract value; its recommended
/// working set is DERIVED by transferring the locally measured ratio, per
/// Rule 89. It has never been measured on the ranked host.
private let rankedPhysicalMemoryBytes = 128.0 * Double(gib)
private let derivedRankedRecommendedWorkingSetBytes =
    rankedPhysicalMemoryBytes
        * (localRecommendedWorkingSetBytes / localPhysicalMemoryBytes)

/// `Memory.activeMemory` at the sizing instant with the ranked
/// command-buffer geometry in force. MEASURED on the local host, and used here
/// as a PROXY for the ranked tower: the backbone and head byte counts are
/// fixed by the checkpoint, but a 128 GiB host may hold more runtime state, so
/// the clamp check below demands a large headroom rather than a tight fit.
private let measuredActiveAtSizingBytes = 26_146_704_372.0

/// The source keeps this margin under the recommended working set for system
/// bookkeeping before applying the clamp.
private let clampMarginBytes = 256.0 * Double(mib)

// MARK: - the arithmetic under test

/// Mirrors `wireResidentWeightsIfEnabled`: scale the active count by the
/// fraction, add the slack allowance, then clamp to the recommended working
/// set less the bookkeeping margin.
private func wiredTargetBytes(
    activeBytes: Double,
    fraction: Double,
    slackMB: Int,
    recommendedBytes: Double
) -> Double {
    let scaled = (activeBytes * min(max(fraction, 0.0), 1.0)).rounded(.down)
    let requested = scaled + Double(max(0, slackMB) * mib)
    return min(requested, max(0.0, recommendedBytes - clampMarginBytes))
}

private func satisfiesBoundA(slackMB: Int) -> Bool {
    Double(slackMB) >= measuredPersistentGrowthMB + derivedPageTaxWorstCaseMB
}

private func satisfiesBoundC(slackMB: Int) -> Bool {
    Double(slackMB) <= observedScratchFloorMB / 1.1
}

@Suite
struct E130WiredResidencySlackTests {
    private func shippedSlackMB() throws -> Int {
        let name = "wiredZHDefaultSlackMB"
        guard let slackMB = Int(try declaredConstant(name)) else {
            throw MissingDeclaration(name: name)
        }
        return slackMB
    }

    /// BOUND A, the floor. The allowance must cover what the scored window
    /// allocates and keeps live after sizing, plus the page rounding the
    /// residency set charges and the sizing input does not.
    @Test
    func theWiredSlackCoversTheMeasuredGrowthAndItsPageRoundingTax() throws {
        let slackMB = try shippedSlackMB()

        #expect(
            satisfiesBoundA(slackMB: slackMB),
            """
            bound A fails: slack \(slackMB) MiB is below the measured \
            persistent growth \(measuredPersistentGrowthMB) MiB plus the \
            worst-case page tax \(derivedPageTaxWorstCaseMB) MiB. Part of the \
            tower cannot stay resident, and which part misses is decided by \
            unordered-set iteration, so the run-to-run split is a lottery.
            """
        )

        // POSITIVE CONTROLS. 64 MiB is the value E130 rung 9 falsified: it is
        // short of the growth alone, before any page tax. 256 MiB clears the
        // expected tax but not the worst case, which is why it was rejected as
        // marginal rather than adopted as the fix.
        #expect(satisfiesBoundA(slackMB: 64) == false)
        #expect(Double(256) >= measuredPersistentGrowthMB + derivedPageTaxExpectedMB)
        #expect(satisfiesBoundA(slackMB: 256) == false)
    }

    /// BOUND C, the ceiling, and the binding one. Keep the allowance well
    /// under the live scratch peak so scratch keeps failing the fit test.
    @Test
    func theWiredSlackStaysStructurallyBelowTheLiveScratchFloor() throws {
        let slackMB = try shippedSlackMB()

        #expect(
            satisfiesBoundC(slackMB: slackMB),
            """
            bound C fails: slack \(slackMB) MiB is not structurally below the \
            observed live scratch floor \(observedScratchFloorMB) MiB, so \
            per-round scratch can enter the residency set and pay a commit on \
            every allocation.
            """
        )

        // POSITIVE CONTROL: an allowance at the scratch floor must fail.
        #expect(satisfiesBoundC(slackMB: Int(observedScratchFloorMB)) == false)
    }

    /// BOUND B, which must not bind. The clamp to the recommended working set
    /// has to leave the request untouched on the ranked host by a wide margin,
    /// because the ranked active count is a transferred proxy and not a
    /// measurement.
    @Test
    func theWiredTargetStaysFarUnderTheRankedWorkingSetClamp() throws {
        let slackMB = try shippedSlackMB()
        let clamp = derivedRankedRecommendedWorkingSetBytes - clampMarginBytes
        let target = wiredTargetBytes(
            activeBytes: measuredActiveAtSizingBytes,
            fraction: 1.0,
            slackMB: slackMB,
            recommendedBytes: derivedRankedRecommendedWorkingSetBytes)

        #expect(
            target == measuredActiveAtSizingBytes + Double(slackMB * mib),
            "bound B fails: the clamp reduced the request on the ranked host")
        #expect(
            clamp - target >= 32.0 * Double(gib),
            """
            bound B is close to binding: only \
            \(((clamp - target) / Double(gib)).rounded()) GiB separate the \
            request from the clamp, and the ranked active count is a proxy.
            """
        )

        // POSITIVE CONTROL: an allowance beyond the remaining working set must
        // be clamped, which proves the check above can fail.
        let absurd = wiredTargetBytes(
            activeBytes: measuredActiveAtSizingBytes,
            fraction: 1.0,
            slackMB: 80 * 1024,
            recommendedBytes: derivedRankedRecommendedWorkingSetBytes)
        #expect(absurd == clamp)
    }

    /// The fix is the absolute term, not the fraction. A fraction below 1.0
    /// would leave part of the tower outside the ticket at every size, which
    /// is a different failure from the one E130 measured.
    @Test
    func theWiredFractionStillWiresTheWholeTower() throws {
        let fraction = try declaredConstant("wiredZHDefaultFraction")
        #expect(fraction == "1.0")
    }

    /// The measurement path depends on this exact override name: the leg
    /// scripts select an arm with it, and `sanitizedRuntimeWorkerEnvironment`
    /// forwards it only because of the `DARKBLOOM_` prefix. A rename would
    /// silently make every arm run the compiled default.
    @Test
    func theSlackOverrideKeepsTheForwardedEnvironmentName() throws {
        let source = try sessionSource()
        #expect(source.contains("DARKBLOOM_QWEN_MTP_WIRED_ZH_SLACK_MB"))
    }
}
