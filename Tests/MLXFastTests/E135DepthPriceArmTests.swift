import Foundation
@testable import MLXFastModel
import Testing

// E135 F39 item 1: the compiled depth-price table must be `.ship`, the
// uniform price.
//
// This file previously pinned `.pb6`, on advisor F33's estimate that `.ship`
// cost about -0.58 % of the ranked timed leg. F39 withdrew that estimate as
// advisor error 174: the state corrector behind it was retired, and the one
// ranked receipt pair that isolates the arm, `572b2cc4 -> e003a86d`, prices
// `pb6` at `-2.3800 %` on the published median. `pb6` buys drafting on
// plutarch, which carries zero median weight, and pays for it on beagle,
// which carries about half. The sign of the pin is reversed; the pin itself
// is not weakened.
//
// A timing leg cannot prove which table shipped: the two arms differ only in
// the numbers the scheduler reads, and the schedule they produce is what the
// leg measures. This file proves the table itself, element by element, from
// literals rather than from the source constants. If `headStepCostRatio`,
// `passBoundaryVerifyWidth`, `passBoundaryTierFactor` or `Qwen36MTPLimits`
// drift, every expectation below fails, which is the point.
//
// Rule 101: each predicate is stated with a control that the same predicate
// rejects. A table match that cannot fail is not evidence.

/// The shipped uniform arm's parameters, written out rather than read back.
private let armH = 0.18
private let armCount = 8

/// The retired `pb6` research arm, kept here as the failing polarity.
private let armTier = 1.45
private let armPricedWidth = 6

/// `count * h / (count - 1 + tier)`, the level `pb6` uses to hold the total at
/// `count * h` once one step is multiplied by `tier`.
private let armWithin = Double(armCount) * armH
    / (Double(armCount - 1) + armTier)

private func expectedShipMarginal() -> [Double] {
    [Double](repeating: armH, count: armCount)
}

private func expectedPB6Marginal() -> [Double] {
    var marginal = [Double](repeating: armWithin, count: armCount)
    marginal[armPricedWidth - 2] = armWithin * armTier
    return marginal
}

/// The uniform arm's own accumulation order: `1 + i*h`, a multiply, NOT a
/// running sum. The two orders disagree by one ulp from index 3 upward
/// (`1.54` against `1.5399999999999998`), which is exactly the drift this
/// element-by-element test exists to catch.
private func expectedShipCumulative() -> [Double] {
    (0 ... armCount).map { 1.0 + Double($0) * armH }
}

/// The boundary arm's accumulation order, which IS a running sum. A different
/// order differs by one ulp and an arm that is not bit-identical to the tip is
/// not a match.
private func expectedCumulative(_ marginal: [Double]) -> [Double] {
    var out = [1.0]
    var running = 1.0
    for value in marginal {
        running += value
        out.append(running)
    }
    return out
}

@Suite("E135 compiled depth-price arm")
struct E135DepthPriceArmTests {
    @Test("the pb6 parameters are the ones F33 named")
    func pb6ParametersMatchTheStatedArithmetic() throws {
        #expect(Qwen36MTPLimits.maxDepth == armCount)
        #expect(Qwen36MTPBlockSession.passBoundaryVerifyWidth == armPricedWidth)
        #expect(Qwen36MTPBlockSession.passBoundaryTierFactor == armTier)

        // The two values F33 quoted, to seven places. The arm is retired, but
        // its parameters stay pinned so a future reopening compares like with
        // like against the ranked receipt that refuted it.
        #expect(abs(armWithin - 0.1704142) < 5e-8)
        #expect(abs(expectedPB6Marginal()[4] - 0.2471006) < 5e-8)

        // Control: the E56 tier the `pb5` and `pb7` arms still use produces a
        // different level, so the tier really does select these numbers.
        let e56 = Double(armCount) * armH
            / (Double(armCount - 1) + Qwen36MTPBlockSession.boundaryTierFactor)
        #expect(abs(e56 - 0.1704142) > 5e-8)
    }

    @Test("the compiled table is ship element by element")
    func compiledTableMatchesTheUniformShipTable() throws {
        // The arm is env-selectable, so an arm session compiles a different
        // table on purpose. Only the env-free build is the shipped one.
        try #require(ProcessInfo.processInfo
            .environment["MLX_E134_DEPTH_PRICE_ARM"] == nil,
            "run this test without MLX_E134_DEPTH_PRICE_ARM set")

        #expect(Qwen36MTPBlockSession.depthPriceArm == .ship)

        let want = expectedShipMarginal()
        let wantCumulative = expectedShipCumulative()
        let live = Qwen36MTPBlockSession.depthPrice

        #expect(live.marginal.count == want.count)
        #expect(live.cumulative.count == wantCumulative.count)
        for index in want.indices {
            #expect(live.marginal[index] == want[index],
                    "marginal[\(index)]")
        }
        for index in wantCumulative.indices {
            #expect(live.cumulative[index] == wantCumulative[index],
                    "cumulative[\(index)]")
        }

        // The total is the invariant every arm holds: an arm changes the SHAPE
        // of the price, never its level.
        #expect(abs(live.cumulative[armCount] - (1.0 + Double(armCount) * armH))
                < 1e-12)
    }

    @Test("the same comparison rejects the pb6 arm")
    func theTableComparisonRejectsTheBoundaryArm() throws {
        let want = expectedShipMarginal()
        let pb6 = Qwen36MTPBlockSession.makeBoundaryDepthPrice(
            enteringVerifyWidth: Qwen36MTPBlockSession
                .passBoundaryVerifyWidth,
            tier: Qwen36MTPBlockSession.passBoundaryTierFactor)

        // Positive control. `pb6` holds the same total, so the level test
        // alone cannot separate the arms; only the element-wise shape can.
        #expect(abs(pb6.cumulative[armCount] - (1.0 + Double(armCount) * armH))
                < 1e-12)
        #expect(pb6.marginal.count == want.count)
        #expect(pb6.marginal == expectedPB6Marginal())
        #expect(pb6.marginal != want)
        #expect(pb6.marginal[armPricedWidth - 2] != want[armPricedWidth - 2])
        #expect(pb6.cumulative != expectedShipCumulative())
        #expect(pb6.cumulative == expectedCumulative(expectedPB6Marginal()))
    }
}
