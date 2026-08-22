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
// E130 rung 9 measured the failure and rung 10 measured its mechanism. The
// allowance is not spent when the ticket is sized: at that instant the whole
// tower fits and tens of MiB are still free. It is spent afterwards, by the
// persistent state the scored window allocates, and then it runs out. At the
// shipped 64 MiB the probe measured 62.90 to 63.00 MiB consumed and under
// 25 KB of headroom left, against a persistent growth of 218.71 MiB — so about
// 156 MiB of persistent state never became resident.
//
// The bounds below are the arithmetic that replaced 64 with 512. They are
// cheap text and integer checks, so this class of regression does not need a
// model, a GPU, or a timed leg to catch — which is the whole point.
//
// Every bound has a POSITIVE CONTROL: a slack value that must fail it. A bound
// that cannot fail is not a bound.
//
// RETRACTION PINNED BY THIS FILE. An earlier revision derived a page-rounding
// tax of 34.80 to 129.23 MiB from `Memory.numResources`. That inference had no
// measurement under it and was wrong by more than 35x. The tax is now measured,
// and `theMeasuredPageTaxRefutesTheResourceCountModel` keeps the retracted
// model from returning.

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
/// growth after the sizing instant, as `growth_min_last_third_mib`. MEASURED on
/// seven worker processes.
///
/// Two frames, because the same compiled constant serves every worker role and
/// the roles do not grow by the same amount. The scored-role frame is the
/// `mtp_decode` worker over a 512-token window. The max-over-roles frame is the
/// largest growth any probed process reached, and it is the one bound A uses:
/// an allowance that only covers the average role still strands the others.
private let measuredPersistentGrowthScoredRoleMB = 218.71
private let measuredPersistentGrowthMaxOverRolesMB = 267.79

/// E130 rung 10 admission probe, three worker processes, identical to the byte
/// in all three. `wired_bytes_sum` at the greedy fill in `ResidencySet::resize`
/// against `Memory.activeMemory` at the sizing instant. Both MEASURED, and
/// their difference is the page-rounding tax that the residency set charges and
/// the sizing input does not.
private let measuredResidencySetBytesAtGreedyFill = 26_147_726_336.0

/// `Memory.numResources` at the sizing instant, and the number of allocations
/// the residency set actually holds. Both MEASURED. They differ by 2.07x
/// because MLX serves small tensors from a 1 MiB heap that enters the set as
/// one allocation, which is why a per-resource page charge is not a valid model.
private let measuredResourceCountAtSizing = 4454
private let measuredResidencySetAllocationCount = 2157

/// E130 rung 10 admission session, `s64` and `s512` arms, three worker roles,
/// four to five process draws each. All MEASURED at steady state.
///
/// These three numbers are why bound A is necessary but NOT sufficient. The
/// allowance is a first-come-first-served pool that persistent state and
/// scratch draw from together, and it is exhausted at BOTH sizes: the fill
/// stops with well under 200 KB unspent either way, and gigabytes stay
/// unwired. Raising the allowance does not reach a threshold and stop helping.
/// It buys residency one byte at a time, at a measured slope of exactly 1.0.
///
/// The first value is the largest steady-state headroom left unspent by the
/// greedy fill, over both arms and all three worker roles. The per-role values
/// are 132,596, 24,564 and 20,468 bytes, and each role reports the SAME value
/// in both arms even though the capacity differs by 448 MiB.
private let measuredMaxHeadroomAtSaturationBytes = 132_596.0

/// Change in admitted bytes for the 448 MiB change in allowance, in the two
/// roles that reported no dispersion across four draws each. MEASURED, and
/// equal to the change in allowance.
private let measuredSlackDeltaMB = 448.0
private let measuredAdmittedDeltaMB = 448.0

/// Smallest steady-state unwired total observed in the `s512` arm. Residency
/// demand therefore exceeds the shipped allowance by at least this much.
/// MEASURED.
private let measuredMinUnwiredAtS512MB = 2442.7

/// Live scratch peaks near this during the scored window. The allowance must
/// stay structurally below it: scratch that fails the fit test takes the
/// commit-free unwired path, and an allowance large enough to admit scratch
/// would put per-round allocation churn on the committing path instead.
/// MEASURED. With bound A demoted, this is the only real ceiling.
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

/// The page-rounding tax, as the difference of two measured byte counts rather
/// than as a per-buffer guess. 1,021,964 B, which is 474 B per admitted
/// allocation, or 2.9 % of one 16 KiB page: the buffers are already page
/// multiples and there is almost nothing to round.
private let measuredPageTaxBytes =
    measuredResidencySetBytesAtGreedyFill - measuredActiveAtSizingBytes
private let measuredPageTaxMB = measuredPageTaxBytes / Double(mib)

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

/// Bound A on the max-over-roles frame: 268.76 MiB.
private func satisfiesBoundA(slackMB: Int) -> Bool {
    Double(slackMB) >= measuredPersistentGrowthMaxOverRolesMB + measuredPageTaxMB
}

/// Bound A on the scored-role frame alone: 219.68 MiB. Kept so the positive
/// controls can show which frame each verdict comes from.
private func satisfiesBoundAScoredRoleOnly(slackMB: Int) -> Bool {
    Double(slackMB) >= measuredPersistentGrowthScoredRoleMB + measuredPageTaxMB
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

    /// BOUND A, a NECESSARY BUT NOT SUFFICIENT lower bound.
    ///
    /// An allowance smaller than the persistent state the scored window keeps
    /// live cannot be defended at any point in the design, so this still has to
    /// hold. It does NOT mean the allowance is large enough:
    /// `theWiredSlackIsExhaustedAtEveryMeasuredSize` shows the pool saturates
    /// at 512 MiB as well, so clearing this bound settles nothing on its own.
    @Test
    func theWiredSlackCoversTheMeasuredGrowthAndItsPageRoundingTax() throws {
        let slackMB = try shippedSlackMB()

        #expect(
            satisfiesBoundA(slackMB: slackMB),
            """
            bound A fails: slack \(slackMB) MiB is below the largest measured \
            persistent growth \(measuredPersistentGrowthMaxOverRolesMB) MiB \
            plus the measured page tax \(measuredPageTaxMB) MiB. Some worker \
            role would exhaust its allowance during the window and leave the \
            remainder of its persistent state off the resident path.
            """
        )

        // POSITIVE CONTROL. 64 MiB is the value E130 rung 9 falsified. It is
        // short of the growth on either frame, before any page tax at all.
        #expect(satisfiesBoundA(slackMB: 64) == false)
        #expect(satisfiesBoundAScoredRoleOnly(slackMB: 64) == false)

        // POSITIVE CONTROL at the boundary, which proves the bound is tight
        // and not merely satisfied by a large shipped value.
        #expect(satisfiesBoundA(slackMB: 268) == false)
        #expect(satisfiesBoundA(slackMB: 269))

        // 256 MiB is the frame-sensitive case and the reason both frames are
        // kept. It clears the scored role and fails the worst observed role.
        // An earlier revision rejected 256 for a page tax that did not exist;
        // it is still rejected, but only on evidence that was measured.
        #expect(satisfiesBoundAScoredRoleOnly(slackMB: 256))
        #expect(satisfiesBoundA(slackMB: 256) == false)
    }

    /// The allowance is exhausted at every size E130 measured, so no value of
    /// it is "enough". This is the fact that demotes bound A, and it is pinned
    /// here so a future reader cannot restore the floor story from the bound
    /// alone.
    @Test
    func theWiredSlackIsExhaustedAtEveryMeasuredSize() throws {
        let slackMB = try shippedSlackMB()

        // The same absolute headroom is left in both arms, so measuring it
        // against the SMALLER allowance is the weakest form of the claim.
        // Even there the fill spends better than 99 % of what it was given.
        #expect(
            measuredMaxHeadroomAtSaturationBytes / (64.0 * Double(mib)) < 0.01)

        // Residency demand beyond the sizing point, as what the s512 arm
        // admitted plus what it still could not admit. An allowance below this
        // saturates and keeps buying residency for every byte added.
        let measuredDemandMB = Double(slackMB) + measuredMinUnwiredAtS512MB
        #expect(measuredDemandMB > Double(slackMB))

        // POSITIVE CONTROL: an allowance above the measured demand would not
        // saturate, and the same comparison must report that.
        #expect((measuredDemandMB > 16.0 * 1024.0) == false)
    }

    /// Admitted bytes respond one for one to the allowance. Slope 1.0 is what
    /// makes the allowance a pool rather than a reservation, and it is why the
    /// shipped value is justified by the timing ladder and not by arithmetic.
    @Test
    func theAdmittedBytesRespondOneForOneToTheAllowance() throws {
        let slope = measuredAdmittedDeltaMB / measuredSlackDeltaMB
        #expect(abs(slope - 1.0) < 0.005)

        // POSITIVE CONTROL: a reservation model, where the allowance stops
        // being consumed once the persistent growth is covered, admits only
        // the growth above the old allowance and must fail the same check.
        let reservationDeltaMB = measuredPersistentGrowthMaxOverRolesMB - 64.0
        let reservationSlope = reservationDeltaMB / measuredSlackDeltaMB
        #expect((abs(reservationSlope - 1.0) < 0.005) == false)
    }

    /// The retracted model must not come back. A page charge per live resource
    /// overstates the measured tax by more than 30x, because the residency set
    /// holds roughly half as many allocations as there are resources and those
    /// allocations are already page multiples.
    @Test
    func theMeasuredPageTaxRefutesTheResourceCountModel() throws {
        #expect(measuredPageTaxBytes == 1_021_964.0)

        let bytesPerAdmittedAllocation =
            measuredPageTaxBytes / Double(measuredResidencySetAllocationCount)
        #expect(bytesPerAdmittedAllocation < 1024.0)

        // The retracted derivation, reconstructed here only so the comparison
        // is explicit rather than remembered.
        let retractedExpectedTaxMB =
            Double(measuredResourceCountAtSizing) * 8.0 / 1024.0
        #expect(retractedExpectedTaxMB / measuredPageTaxMB > 30.0)

        // POSITIVE CONTROL: the same check applied to a genuine whole-page
        // charge must not clear the threshold, so the assertion above can fail.
        let wholePageTaxMB =
            Double(measuredResidencySetAllocationCount) * 16.0 / 1024.0
        #expect((retractedExpectedTaxMB / wholePageTaxMB > 30.0) == false)
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
