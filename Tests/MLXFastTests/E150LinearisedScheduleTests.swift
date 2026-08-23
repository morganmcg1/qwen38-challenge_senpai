import Foundation
@testable import MLXFastModel
import Testing

// E150 R4: the linearised depth schedule, its price table, and its gate.
//
// The candidate is ONE mechanism with three parts that only make sense
// together: the ranked measured cost curve as the price table, a global
// argmax over depth in place of the shipped first-break walk, and the two
// margin clamps removed. These tests own the parts a build can check without
// a GPU: the table is derived from the curve rather than transcribed, the
// rule finds a maximum the first-break walk provably cannot reach, the gate
// has both polarities, and `shipped` restores the previous schedule exactly.
//
// Rule 79: nothing here is timing evidence. The effect sizes quoted in the
// source comments are offline replay prices and only a ranked receipt can
// confirm them.
@Suite("E150 linearised schedule")
struct E150LinearisedScheduleTests {
    typealias Session = Qwen36MTPBlockSession

    /// The Python replay this candidate was priced in carries the same curve.
    /// If the two ever disagree the replay is no longer pricing the shipped
    /// candidate, so the whole R0.5 and R1 evidence chain detaches.
    /// `research/e150_r4_constants.py` prints these from `env.measured`.
    static let pythonMarginal: [Double] = [
        0.077933895188272206,
        0.080439059893738341,
        0.12432227344783353,
        0.17400606530406293,
        0.43230963612879036,
        0.40356482908000624,
        0.048073362350579352,
        0.048073362350578908,
    ]

    @Test("the price table is derived from the measured curve, not retyped")
    func priceTableMatchesTheReplay() {
        let price = Session.makeRankedMeasuredDepthPrice()
        #expect(price.marginal.count == Self.pythonMarginal.count)
        var worst = 0.0
        for (mine, theirs) in zip(price.marginal, Self.pythonMarginal) {
            worst = Swift.max(worst, abs(mine - theirs))
        }
        #expect(worst < 1e-12)
    }

    /// `cumulative[0]` is the verify forward on its own and every later entry
    /// accumulates one marginal. A table that fails this is not comparable
    /// with `lambda`, which is measured in the same normalised units.
    @Test("the cumulative table starts at one and accumulates")
    func cumulativeIsAPrefixSum() {
        let price = Session.makeRankedMeasuredDepthPrice()
        #expect(price.cumulative.count == price.marginal.count + 1)
        #expect(price.cumulative[0] == 1.0)
        var running = 1.0
        for (index, step) in price.marginal.enumerated() {
            running += step
            #expect(abs(price.cumulative[index + 1] - running) < 1e-12)
        }
    }

    /// The measured curve is NOT concave. This is the whole reason a
    /// first-break walk loses: it stops at the first step that fails a local
    /// test, and a cheaper step sits behind the expensive one.
    @Test("the measured curve is non-concave, so a local rule can be trapped")
    func theCurveIsNonConcave() {
        let price = Session.makeRankedMeasuredDepthPrice()
        // Step into width 6 is index 4, step into width 8 is index 6.
        #expect(price.marginal[4] > price.marginal[6] * 8.0)
        var increases = 0
        for index in 1 ..< price.marginal.count
        where price.marginal[index] > price.marginal[index - 1] {
            increases += 1
        }
        #expect(increases >= 1)
    }

    /// CAMPAIGN RULE 101 applied to the gate: a witness that cannot fail is
    /// worse than no witness, so prove both polarities of the parse.
    @Test("the schedule gate has both polarities and defaults to linearised")
    func gateHasBothPolarities() {
        func parse(_ raw: String?) -> Session.ScheduleArm {
            Session.ScheduleArm(rawValue: raw ?? "") ?? .linearised
        }
        #expect(parse(nil) == .linearised)
        #expect(parse("") == .linearised)
        #expect(parse("nonsense") == .linearised)
        #expect(parse("shipped") == .shipped)
        #expect(parse("linearised") == .linearised)
        #expect(ProcessInfo.processInfo
            .environment["MLX_E150_SCHEDULE_ARM"] == nil)
        #expect(Session.scheduleArm == .linearised)
    }

    /// `mu*` is a fixed point solved by bisection, and its eight
    /// leave-one-prompt-out folds bracket it. A constant that fell outside its
    /// own folds would be a selection artefact rather than a fixed point.
    @Test("lambda star sits inside its own leave-one-prompt-out folds")
    func lambdaStarIsInsideItsFolds() {
        let (low, high) = Session.linearisedLambdaStarFoldRange
        #expect(Session.linearisedLambdaStar > low)
        #expect(Session.linearisedLambdaStar < high)
    }

    // ------------------------------------------------------------- the rules
    //
    // Both walks are reproduced here as free functions over an explicit state
    // vector. The session's own methods read instance state that only a live
    // decode can build, and the property under test is the RULE, not the
    // plumbing that feeds it.

    static func firstBreak(ema: [Double], cap: Int,
                           price: Qwen36MTPBlockSession.DepthPrice) -> Int {
        var reach = 1.0
        var expected = 0.0
        var depth = 0
        while depth < cap {
            reach *= ema[depth]
            let threshold = price.marginal[depth] * (1.0 + expected)
                / price.cumulative[depth]
            guard reach > threshold else { break }
            expected += reach
            depth += 1
        }
        return depth
    }

    static func linearised(ema: [Double], cap: Int, lambda: Double,
                           price: Qwen36MTPBlockSession.DepthPrice) -> Int {
        var best = 0
        var bestValue = lambda - price.cumulative[0]
        var reach = 1.0
        var expected = 0.0
        var depth = 0
        while depth < cap {
            reach *= ema[depth]
            expected += reach
            depth += 1
            let value = lambda * (1.0 + expected) - price.cumulative[depth]
            if value > bestValue {
                bestValue = value
                best = depth
            }
        }
        return best
    }

    /// The decisive property. On a hot prompt the two rules disagree, and the
    /// argmax goes DEEPER, because it looks past the width-6 step to the cheap
    /// width-8 step behind it. This is a positive control: if the two rules
    /// ever agree everywhere, the mechanism is inert and the replay evidence
    /// is describing something else.
    @Test("the argmax reaches past a step the first-break walk stops at")
    func argmaxLooksPastTheExpensiveStep() {
        let price = Session.makeRankedMeasuredDepthPrice()
        let lambda = Session.linearisedLambdaStar
        let hot = [Double](repeating: 0.97, count: 8)
        let greedy = Self.firstBreak(ema: hot, cap: 7, price: price)
        let argmax = Self.linearised(ema: hot, cap: 7, lambda: lambda,
                                     price: price)
        #expect(argmax > greedy)
        // The first-break walk stops at depth 4, which is verify width 5,
        // exactly at the foot of the width-6 step. The argmax takes the whole
        // cap, which is verify width 8, because the two cheap steps behind
        // that cliff repay it.
        #expect(greedy == 4)
        #expect(argmax == 7)
    }

    /// The argmax is not simply deeper. On a cold prompt it must refuse to
    /// draft at all, which is the safety property the removed clamps used to
    /// carry under the first-break rule.
    @Test("the argmax refuses to draft on a cold prompt")
    func argmaxStopsWhenReachCollapses() {
        let price = Session.makeRankedMeasuredDepthPrice()
        let lambda = Session.linearisedLambdaStar
        let cold = [Double](repeating: 0.05, count: 8)
        #expect(Self.linearised(ema: cold, cap: 7, lambda: lambda,
                                price: price) == 0)
    }

    /// The objective is monotone in acceptance, so a strictly better head can
    /// never make the schedule shallower. A rule without this property would
    /// punish head improvements, which is the opposite of the campaign.
    @Test("depth is monotone in the acceptance estimate")
    func depthIsMonotoneInAcceptance() {
        let price = Session.makeRankedMeasuredDepthPrice()
        let lambda = Session.linearisedLambdaStar
        var previous = -1
        for step in 0 ... 20 {
            let p = 0.05 * Double(step)
            let ema = [Double](repeating: Swift.min(p, 1.0), count: 8)
            let depth = Self.linearised(ema: ema, cap: 7, lambda: lambda,
                                        price: price)
            #expect(depth >= previous)
            previous = depth
        }
        #expect(previous == 7)
    }

    /// The cap is the only width authority. `segmentedVerifyDepthCap` is 7, so
    /// even a perfect head cannot ask for a verify width the campaign has not
    /// proven bit-exact.
    @Test("the cap still bounds the argmax")
    func capBoundsTheArgmax() {
        let price = Session.makeRankedMeasuredDepthPrice()
        let lambda = Session.linearisedLambdaStar
        let perfect = [Double](repeating: 1.0, count: 8)
        for cap in 0 ... 7 {
            #expect(Self.linearised(ema: perfect, cap: cap, lambda: lambda,
                                    price: price) == cap)
        }
    }

    /// The Dinkelbach property this whole rung rests on: AT THE FIXED POINT,
    /// maximising `lambda * T_d - C_d` is the same as minimising `C_d / T_d`.
    /// That is what makes a linear objective a legitimate stand-in for a ratio
    /// of sums, and it is why `mu*` had to be solved rather than chosen.
    ///
    /// The test sets `lambda` to the rate the argmax itself achieves, which is
    /// the fixed-point condition on ONE round, then checks that the same depth
    /// also minimises cost per token. It runs over several acceptance profiles
    /// so a single lucky vector cannot carry it.
    @Test("at the fixed point the argmax also minimises cost per token")
    func argmaxIsTheRatioMinimiserAtItsOwnRate() {
        let price = Session.makeRankedMeasuredDepthPrice()
        let profiles: [[Double]] = [
            [0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5, 0.4],
            [0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92],
            [0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6],
            [0.3, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
        ]
        for ema in profiles {
            var tokens = [1.0]
            var reach = 1.0
            var expected = 0.0
            for depth in 0 ..< 7 {
                reach *= ema[depth]
                expected += reach
                tokens.append(1.0 + expected)
            }
            let ratios = (0 ... 7).map { price.cumulative[$0] / tokens[$0] }
            let ratioBest = ratios.enumerated()
                .min(by: { $0.element < $1.element })!.offset
            let lambda = ratios[ratioBest]
            let chosen = Self.linearised(ema: ema, cap: 7, lambda: lambda,
                                         price: price)
            #expect(ratios[chosen] <= ratios[ratioBest] + 1e-12)
        }
    }

    /// A NEGATIVE CONTROL for the test above. The equivalence is a property of
    /// the fixed point, not of any `lambda`, so a badly wrong `lambda` must be
    /// able to pick a depth that is NOT the ratio minimiser. If this never
    /// fires, the previous test proves nothing.
    @Test("away from the fixed point the two objectives can disagree")
    func awayFromTheFixedPointTheyDisagree() {
        let price = Session.makeRankedMeasuredDepthPrice()
        let ema = [0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5, 0.4]
        var tokens = [1.0]
        var reach = 1.0
        var expected = 0.0
        for depth in 0 ..< 7 {
            reach *= ema[depth]
            expected += reach
            tokens.append(1.0 + expected)
        }
        let ratios = (0 ... 7).map { price.cumulative[$0] / tokens[$0] }
        let ratioBest = ratios.enumerated()
            .min(by: { $0.element < $1.element })!.offset
        var disagreements = 0
        for step in 1 ... 40 {
            let lambda = 0.05 * Double(step)
            let chosen = Self.linearised(ema: ema, cap: 7, lambda: lambda,
                                         price: price)
            if chosen != ratioBest { disagreements += 1 }
        }
        #expect(disagreements > 0)
    }
}
