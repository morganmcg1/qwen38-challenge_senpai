import Foundation
import Testing

@testable import MLXFastModel

// E200 -- the width-aware depth price, as a MEASUREMENT ARM only.
//
// These tests protect the two properties a leg session depends on. First, an
// unset `MLX_E200_DEPTH_PRICE` must leave the shipped price bit-identical, or
// every earlier leg silently re-times against a different schedule. Second,
// the ranked table must be the E197 smooth-step law and nothing else, so a
// leg cannot report an arm it did not run.
//
// The arm exists to answer one causal question and is deleted before any
// freeze (RULE 198). It is not a candidate for the submitted surface.

@Suite("E200 ranked step depth price")
struct E200DepthPriceTableTests {

    @Test("the compile-time default is still the shipped uniform price")
    func shippedDefaultUnchanged() {
        #expect(Qwen36MTPBlockSession.depthPriceArm == .ship)
        let uniform = Qwen36MTPBlockSession.makeUniformDepthPrice()
        #expect(uniform.marginal.count == Qwen36MTPLimits.maxDepth)
        #expect(uniform.cumulative.count == Qwen36MTPLimits.maxDepth + 1)
        for value in uniform.marginal {
            #expect(value == 0.18)
        }
        // The shipped constructor uses a closed form, not an accumulation.
        // A control arm that is not bit-identical to the tip is not a control.
        for (index, value) in uniform.cumulative.enumerated() {
            #expect(value == 1.0 + Double(index) * 0.18)
        }
    }

    @Test("the unset environment selects the compile-time arm")
    func unsetEnvironmentSelectsShip() {
        // The leg scripts set the variable per leg. Inside `swift test` it is
        // unset, which is the case that has to stay bit-identical.
        #expect(ProcessInfo.processInfo
            .environment["MLX_E200_DEPTH_PRICE"] == nil)
        #expect(Qwen36MTPBlockSession.depthPriceArmEffective
            == Qwen36MTPBlockSession.depthPriceArm)
        let live = Qwen36MTPBlockSession.depthPrice
        let uniform = Qwen36MTPBlockSession.makeUniformDepthPrice()
        #expect(live.marginal == uniform.marginal)
        #expect(live.cumulative == uniform.cumulative)
    }

    @Test("the ranked cost table is R(1...9) and is strictly increasing")
    func rankedCostTableShape() {
        let cost = Qwen36MTPBlockSession.rankedRoundCostMs
        #expect(cost.count == Qwen36MTPLimits.maxDepth + 1)
        for index in 1 ..< cost.count {
            #expect(cost[index] > cost[index - 1])
        }
        // FINDING 505 / 514: the step into verify width 6 is the large one.
        let step6 = cost[5] - cost[4]
        let step2 = cost[1] - cost[0]
        #expect(step6 > 7.9 && step6 < 8.3)
        #expect(step2 > 1.3 && step2 < 1.5)
        #expect(step6 > 5.0 * step2)
    }

    @Test("the held-level arm keeps the reachable total and changes the shape")
    func heldLevelArm() {
        let price = Qwen36MTPBlockSession
            .makeRankedStepDepthPrice(holdLevel: true)
        #expect(price.marginal.count == Qwen36MTPLimits.maxDepth)
        #expect(price.cumulative.count == Qwen36MTPLimits.maxDepth + 1)
        #expect(price.cumulative[0] == 1.0)
        let reachable = 7                       // segmentedVerifyDepthCap
        let total = price.marginal.prefix(reachable).reduce(0.0, +)
        #expect(abs(total - Double(reachable) * 0.18) < 1e-12)
        // The shape, not the level, is the mechanism: the width-6 step must
        // cost several times a shallow step after the rescale.
        #expect(price.marginal[4] > 3.0 * price.marginal[0])
        for index in 0 ..< Qwen36MTPLimits.maxDepth {
            #expect(price.cumulative[index + 1]
                == price.cumulative[index] + price.marginal[index])
        }
    }

    @Test("the measured-level arm is the ranked law divided by the forward")
    func measuredLevelArm() {
        let cost = Qwen36MTPBlockSession.rankedRoundCostMs
        let price = Qwen36MTPBlockSession
            .makeRankedStepDepthPrice(holdLevel: false)
        for index in 0 ..< Qwen36MTPLimits.maxDepth {
            let expected = (cost[index + 1] - cost[index]) / cost[0]
            #expect(abs(price.marginal[index] - expected) < 1e-15)
        }
        // The honest ranked level is BELOW the shipped one over the reachable
        // steps. That is the E200 finding: `headStepCostRatio` is not a cost
        // estimate, it is a compensator for a pessimistic `reach`.
        let total = price.marginal.prefix(7).reduce(0.0, +)
        #expect(total < 7.0 * 0.18)
    }

    @Test("a width-aware price closes the width-6 step for a strong round")
    func widthSixStepClosesUnderTheRankedPrice() {
        // The mechanism the price side cannot deliver. At depth 4 the guard is
        // `reach > marginal[4] * (1 + expected) / cumulative[4]`, and
        // `1 + expected` is LARGER for a strong round, so a large marginal[4]
        // blocks strong rounds hardest rather than only marginal ones.
        let price = Qwen36MTPBlockSession
            .makeRankedStepDepthPrice(holdLevel: true)
        let q = 0.95                            // a very strong round
        var reach = 1.0
        var expected = 0.0
        for depth in 0 ..< 4 {
            reach *= q
            expected += reach
            _ = depth
        }
        reach *= q
        let threshold = price.marginal[4] * (1.0 + expected)
            / price.cumulative[4]
        #expect(threshold > 1.0)
        #expect(reach < threshold)

        // The shipped uniform price leaves the same step open.
        let uniform = Qwen36MTPBlockSession.makeUniformDepthPrice()
        let shipThreshold = uniform.marginal[4] * (1.0 + expected)
            / uniform.cumulative[4]
        #expect(reach > shipThreshold)
    }
}
