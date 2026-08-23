import Foundation
@testable import MLXFastModel
import Testing

// E135 F33 item 1: the compiled depth-price table must be the base's `pb6`.
//
// This branch briefly compiled `.ship`. Advisor F33 prices that regression at
// about -0.58 % of the ranked timed leg, roughly three times the +0.1912 %
// the one-pass table buys, so the arm the worker compiles is load-bearing for
// the submission and not a research knob.
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

/// The base's `pb6` parameters, written out rather than read back.
private let armH = 0.18
private let armTier = 1.45
private let armCount = 8
private let armPricedWidth = 6

/// `count * h / (count - 1 + tier)`, the level that holds the total at
/// `count * h` once one step is multiplied by `tier`.
private let armWithin = Double(armCount) * armH
    / (Double(armCount - 1) + armTier)

private func expectedMarginal() -> [Double] {
    var marginal = [Double](repeating: armWithin, count: armCount)
    marginal[armPricedWidth - 2] = armWithin * armTier
    return marginal
}

/// The tip's own accumulation order. A different order differs by one ulp and
/// an arm that is not bit-identical to the tip is not a match.
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

        // The two values F33 quoted, to seven places.
        #expect(abs(armWithin - 0.1704142) < 5e-8)
        #expect(abs(expectedMarginal()[4] - 0.2471006) < 5e-8)

        // Control: the E56 tier the `pb5` and `pb7` arms still use produces a
        // different level, so the tier really does select these numbers.
        let e56 = Double(armCount) * armH
            / (Double(armCount - 1) + Qwen36MTPBlockSession.boundaryTierFactor)
        #expect(abs(e56 - 0.1704142) > 5e-8)
    }

    @Test("the compiled table is pb6 element by element")
    func compiledTableMatchesTheBasePb6Table() throws {
        // The arm is env-selectable, so an arm session compiles a different
        // table on purpose. Only the env-free build is the shipped one.
        try #require(ProcessInfo.processInfo
            .environment["MLX_E134_DEPTH_PRICE_ARM"] == nil,
            "run this test without MLX_E134_DEPTH_PRICE_ARM set")

        #expect(Qwen36MTPBlockSession.depthPriceArm == .pb6)

        let want = expectedMarginal()
        let wantCumulative = expectedCumulative(want)
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

    @Test("the same comparison rejects the ship arm")
    func theTableComparisonRejectsTheUniformArm() throws {
        let want = expectedMarginal()
        let ship = Qwen36MTPBlockSession.makeUniformDepthPrice()

        // Positive control. `ship` holds the same total, so the level test
        // alone cannot separate the arms; only the element-wise shape can.
        #expect(abs(ship.cumulative[armCount] - (1.0 + Double(armCount) * armH))
                < 1e-12)
        #expect(ship.marginal.count == want.count)
        #expect(ship.marginal != want)
        #expect(ship.marginal[armPricedWidth - 2] != want[armPricedWidth - 2])
        #expect(ship.cumulative != expectedCumulative(want))
    }
}
