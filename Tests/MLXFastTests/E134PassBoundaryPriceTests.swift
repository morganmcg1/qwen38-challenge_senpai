import Foundation
@testable import MLXFastModel
import Testing

// E134: the depth price at the QMV pass boundary.
//
// E56 priced one boundary at a time and fitted a single tier factor for verify
// widths 5 and 7. It never priced width 6, which is the width where this
// stack's QMV dispatch table stops covering the rows in one group. E134's
// replayer scores a `pb6` arm at that width at `+2.34 %` replayed ranked
// median, leave-one-prompt-out, against `0.00 %` for the shipped flat price,
// `-2.23 %` for the same boundary at E56's tier, and `-2.78 %` for the tier
// that equals the measured cost ratio.
//
// These tests own three things the arm depends on and nothing else: that the
// dispatch table still puts the pass boundary where the constant says it is,
// that the built table is exactly the table the replayer priced, and that
// adding a tier parameter left the E56 arms bit-identical.

private let h = 0.18
private let maxDepth = Qwen36MTPLimits.maxDepth

/// The `(M, IPG)` QMV dispatch pairs, parsed from the file that owns them.
///
/// `Qwen35.swift` belongs to another agent and this stack must not copy its
/// table into a constant that can drift. Reading the literal makes a table
/// change fail here, which is the signal to refit the tier rather than to
/// relax the test.
private func dispatchTable(file: StaticString = #filePath) throws -> [(m: Int, ipg: Int)] {
    var root = URL(fileURLWithPath: "\(file)")
    for _ in 0 ..< 3 { root.deleteLastPathComponent() }
    let source = root
        .appendingPathComponent("Vendor/mlx-swift-lm/Libraries/MLXLLM")
        .appendingPathComponent("Models/Qwen35.swift")
    let text = try String(contentsOf: source, encoding: .utf8)
    guard let line = text.split(separator: "\n", omittingEmptySubsequences: false)
        .first(where: { $0.contains("let cases = [") })
    else {
        throw PassBoundaryFailure.tableNotFound
    }
    let pattern = try NSRegularExpression(pattern: "\\((\\d+),\\s*(\\d+)\\)")
    let subject = String(line)
    let range = NSRange(subject.startIndex ..< subject.endIndex, in: subject)
    return pattern.matches(in: subject, range: range).compactMap { match in
        guard let mRange = Range(match.range(at: 1), in: subject),
              let ipgRange = Range(match.range(at: 2), in: subject),
              let m = Int(subject[mRange]), let ipg = Int(subject[ipgRange])
        else { return nil }
        return (m: m, ipg: ipg)
    }
}

private enum PassBoundaryFailure: Error {
    case tableNotFound
}

/// Replay of the shipped walk under a flat acceptance rate, matching
/// `QwenMTPDepthPriceTests`.
private func walkDepth(
    _ price: Qwen36MTPBlockSession.DepthPrice, cap: Int, p: Double
) -> Int {
    var reach = 1.0
    var expected = 0.0
    var depth = 0
    while depth < cap {
        reach *= p
        let threshold = price.marginal[depth] * (1.0 + expected) /
            price.cumulative[depth]
        guard reach > threshold else { break }
        expected += reach
        depth += 1
    }
    return depth
}

@Suite("E134 pass boundary depth price")
struct E134PassBoundaryPriceTests {
    @Test("the QMV dispatch table still puts the pass boundary at width 6")
    func dispatchTablePutsPassBoundaryAtSix() throws {
        let table = try dispatchTable().sorted { $0.m < $1.m }
        #expect(table.count >= 6)
        let passes = table.map { ($0.m + $0.ipg - 1) / $0.ipg }
        // A single boundary: one pass below it, more than one at and above it.
        let boundary = try #require(zip(table, passes).first { $0.1 > 1 }?.0.m)
        #expect(boundary == Qwen36MTPBlockSession.passBoundaryVerifyWidth)
        for (entry, count) in zip(table, passes) {
            #expect(entry.m < boundary ? count == 1 : count > 1)
        }
    }

    @Test("the pb6 table is exactly the table the replayer priced")
    func pb6TableMatchesTheReplayer() {
        let tier = Qwen36MTPBlockSession.passBoundaryTierFactor
        let width = Qwen36MTPBlockSession.passBoundaryVerifyWidth
        let within = Double(maxDepth) * h / (Double(maxDepth - 1) + tier)
        let price = Qwen36MTPBlockSession.makeBoundaryDepthPrice(
            enteringVerifyWidth: width, tier: tier)
        for depth in 0 ..< maxDepth {
            let expected = depth == width - 2 ? within * tier : within
            #expect(abs(price.marginal[depth] - expected) <= expected.ulp)
        }
        let total = price.marginal.reduce(0.0, +)
        #expect(abs(total - Double(maxDepth) * h) < 1e-12)
        var running = 1.0
        #expect(price.cumulative[0] == 1.0)
        for depth in 0 ..< maxDepth {
            running += price.marginal[depth]
            #expect(price.cumulative[depth + 1] == running)
        }
    }

    // The tier parameter carries E56's constant as its default, so the arms
    // that experiment published must not move by one ulp.
    @Test("the tier parameter leaves the E56 arms bit-identical")
    func tierParameterLeavesE56ArmsUnchanged() {
        let e56 = Qwen36MTPBlockSession.boundaryTierFactor
        let within = Double(maxDepth) * h / (Double(maxDepth - 1) + e56)
        for width in [5, 7] {
            let price = Qwen36MTPBlockSession.makeBoundaryDepthPrice(
                enteringVerifyWidth: width)
            for depth in 0 ..< maxDepth {
                let expected = depth == width - 2 ? within * e56 : within
                #expect(price.marginal[depth] == expected)
            }
        }
    }

    @Test("the shipped arm is still ship")
    func shippedArmIsStillShip() {
        #expect(Qwen36MTPBlockSession.depthPriceArm == .ship)
    }

    // The mechanism, in two halves. Holding the total means every step below
    // the boundary gets CHEAPER as the one priced step rises, so pb6 is not a
    // level change. A level change is the `flat` control, which the replayer
    // scored at `-0.30 %` against pb6's `+2.34 %`.
    @Test("pb6 charges less below the pass boundary and more across it")
    func pb6ChargesLessBelowAndMoreAcross() {
        let tier = Qwen36MTPBlockSession.passBoundaryTierFactor
        let width = Qwen36MTPBlockSession.passBoundaryVerifyWidth
        let ship = Qwen36MTPBlockSession.makeUniformDepthPrice()
        let pb6 = Qwen36MTPBlockSession.makeBoundaryDepthPrice(
            enteringVerifyWidth: width, tier: tier)
        for depth in 0 ..< maxDepth {
            if depth == width - 2 {
                #expect(pb6.marginal[depth] > ship.marginal[depth])
            } else {
                #expect(pb6.marginal[depth] < ship.marginal[depth])
            }
        }
    }

    // The second half: at both ranked acceptance rates the flat-rate walk
    // stops with `width - 2` drafts taken, which is exactly the last depth
    // that stays inside one QMV pass, while the shipped price crosses. The
    // flat-rate walk cannot separate one tier from another, so it pins the
    // direction only; the tier itself comes from the replayer.
    @Test("pb6 stops the flat-rate walk at the pass boundary")
    func pb6StopsTheWalkAtThePassBoundary() {
        let tier = Qwen36MTPBlockSession.passBoundaryTierFactor
        let width = Qwen36MTPBlockSession.passBoundaryVerifyWidth
        let ship = Qwen36MTPBlockSession.makeUniformDepthPrice()
        let pb6 = Qwen36MTPBlockSession.makeBoundaryDepthPrice(
            enteringVerifyWidth: width, tier: tier)
        for p in [0.8351, 0.8750] {
            #expect(walkDepth(pb6, cap: 7, p: p) == width - 2)
            #expect(walkDepth(ship, cap: 7, p: p) > width - 2)
        }
    }
}
