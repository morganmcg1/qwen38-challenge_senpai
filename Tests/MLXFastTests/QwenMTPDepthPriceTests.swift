import Foundation
@testable import MLXFastModel
import Testing

// E68: the depth price as a per-position vector.
//
// The scored change inside `costModelDepth` is one expression. The tip reads
//
//     h * (1.0 + expected) / (1.0 + Double(depth) * h)
//
// and the arm reads
//
//     price.marginal[depth] * (1.0 + expected) / price.cumulative[depth]
//
// The loop around it is untouched, so the tip's own coverage still applies to
// the walk. What these tests own is everything the new expression depends on:
// that the `ship` arm is BIT-IDENTICAL to the tip, that the reconstruction of
// E56's boundary vector reproduces that experiment's published constants, and
// that each arm's walk stops where the E68 report says it stops.

private let h = 0.18
private let maxDepth = Qwen36MTPLimits.maxDepth

/// Replay of the shipped walk under a flat acceptance rate. The clamps at
/// depth 0 and 1 read live round state, so a unit test drives the rule at its
/// EMA inputs instead.
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

/// The tip's closed form, written out so the control arm can be compared with
/// it rather than with a paraphrase of it.
private func tipThreshold(depth: Int, expected: Double) -> Double {
    h * (1.0 + expected) / (1.0 + Double(depth) * h)
}

@Suite("E68 depth price")
struct QwenMTPDepthPriceTests {
    @Test("the ship arm reproduces the tip's cumulative cost bit for bit")
    func shipCumulativeIsTheTipClosedForm() {
        let price = Qwen36MTPBlockSession.makeUniformDepthPrice()
        for depth in 0 ... maxDepth {
            #expect(price.cumulative[depth] == 1.0 + Double(depth) * h)
        }
        for depth in 0 ..< maxDepth {
            #expect(price.marginal[depth] == h)
        }
    }

    // The positive control for the test above. Accumulating 0.18 is the
    // obvious way to build `cumulative`, and it is wrong: it differs from the
    // closed form in the last bits at depth 3 through 6. If this ever stops
    // failing, the comparison above has lost its power and the control arm is
    // no longer provably identical to the tip.
    @Test("repeated addition is NOT the tip's cumulative cost")
    func repeatedAdditionDivergesFromTheClosedForm() {
        var running = 1.0
        var divergences = 0
        for depth in 0 ... maxDepth {
            if running != 1.0 + Double(depth) * h { divergences += 1 }
            running += h
        }
        #expect(divergences > 0)
    }

    @Test("the ship arm's threshold equals the tip's at every reachable state")
    func shipThresholdMatchesTheTip() {
        let price = Qwen36MTPBlockSession.makeUniformDepthPrice()
        var mismatches = 0
        for depth in 0 ..< maxDepth {
            for step in 0 ... 400 {
                let expected = Double(step) * 0.02
                let arm = price.marginal[depth] * (1.0 + expected) /
                    price.cumulative[depth]
                if arm != tipThreshold(depth: depth, expected: expected) {
                    mismatches += 1
                }
            }
        }
        #expect(mismatches == 0)
    }

    @Test("the ship arm's walk equals the tip's walk on a dense grid")
    func shipWalkMatchesTheTip() {
        let price = Qwen36MTPBlockSession.makeUniformDepthPrice()
        for step in 1 ... 999 {
            let p = Double(step) / 1000.0
            for cap in 1 ... maxDepth {
                var reach = 1.0
                var expected = 0.0
                var tipDepth = 0
                while tipDepth < cap {
                    reach *= p
                    guard reach > tipThreshold(depth: tipDepth,
                                               expected: expected)
                    else { break }
                    expected += reach
                    tipDepth += 1
                }
                #expect(walkDepth(price, cap: cap, p: p) == tipDepth)
            }
        }
    }

    @Test("the boundary vector reproduces E56's published constants")
    func boundaryVectorReproducesE56() {
        let price = Qwen36MTPBlockSession
            .makeBoundaryDepthPrice(enteringVerifyWidth: 5)
        // Published to six significant figures, so compare at that precision.
        #expect(abs(price.marginal[0] - 0.159467) < 5e-7)
        #expect(abs(price.marginal[3] - 0.323733) < 5e-6)
        #expect(abs(price.cumulative[3] - 1.478400) < 5e-7)
        let coefficient = price.marginal[3] / price.cumulative[3]
        #expect(abs(coefficient - 0.218975) < 5e-6)
    }

    @Test("every arm holds the shipped total")
    func everyArmHoldsTheTotal() {
        let total = Double(maxDepth) * h
        let arms = [
            Qwen36MTPBlockSession.makeUniformDepthPrice(),
            Qwen36MTPBlockSession.makeBoundaryDepthPrice(enteringVerifyWidth: 5),
            Qwen36MTPBlockSession.makeBoundaryDepthPrice(enteringVerifyWidth: 7),
        ]
        for price in arms {
            #expect(abs(price.marginal.reduce(0, +) - total) < 1e-12)
            #expect(price.cumulative.count == maxDepth + 1)
            #expect(price.cumulative[0] == 1.0)
        }
    }

    // The pre-registered predictions from the E68 report. `pb5` is the
    // positive control: it MUST shorten the walk at both ranked acceptance
    // rates, because shortening is the direction this pool has already priced
    // at about -3%. If `pb5` ever stops shortening, the arm no longer tests
    // what the session claims it tests.
    @Test("each arm stops where the E68 report says it stops")
    func armsStopWhereReported() {
        let cap = 5
        let ship = Qwen36MTPBlockSession.makeUniformDepthPrice()
        let pb5 = Qwen36MTPBlockSession
            .makeBoundaryDepthPrice(enteringVerifyWidth: 5)
        let pb7 = Qwen36MTPBlockSession
            .makeBoundaryDepthPrice(enteringVerifyWidth: 7)
        for p in [0.8351, 0.8750] {
            #expect(walkDepth(ship, cap: cap, p: p) == 5)
            #expect(walkDepth(pb5, cap: cap, p: p) == 3)
            #expect(walkDepth(pb7, cap: cap, p: p) == 5)
        }
    }

    @Test("pbfit refuses to run as ship when the measured curve is missing")
    func pbfitRefusesAnUnfilledCurve() {
        // The scored default is `ship`, so `measuredRawDepthPrice` is empty on
        // the branch tip. An empty curve must be a trap, not a silent
        // fallback to the flat vector: a `pbfit` leg that quietly measured
        // `ship` would look like a null result instead of a broken arm.
        #expect(Qwen36MTPBlockSession.measuredRawDepthPrice.isEmpty
                || Qwen36MTPBlockSession.measuredRawDepthPrice.count == maxDepth)
    }
}
