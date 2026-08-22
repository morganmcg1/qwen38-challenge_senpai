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

    // E75 rung A: the E68 winner is now the shipped default, so these tests
    // own the three ways banking it could silently regress — an unfilled
    // curve, a refit that moves the LEVEL instead of the shape, and a leg
    // session that leaves another arm behind.

    @Test("the measured curve is filled, so pbfit cannot trap at startup")
    func measuredCurveIsFilled() {
        // An unfilled curve must be a trap, not a silent fallback to the flat
        // vector: a `pbfit` build that quietly ran `ship` would look like a
        // null result instead of a broken arm. `makeMeasuredDepthPrice`
        // preconditions on the count, and the shipped arm calls it, so an
        // empty array is now a startup crash rather than a test failure.
        #expect(Qwen36MTPBlockSession.measuredRawDepthPrice.count == maxDepth)
    }

    @Test("the measured arm holds the shipped total, so it changes shape only")
    func measuredArmHoldsTheTotal() {
        let price = Qwen36MTPBlockSession.makeMeasuredDepthPrice()
        #expect(price.marginal.count == maxDepth)
        #expect(price.cumulative.count == maxDepth + 1)
        #expect(price.cumulative[0] == 1.0)
        #expect(abs(price.marginal.reduce(0, +) - Double(maxDepth) * h) < 1e-12)
    }

    // The exact vector the nine E68 rung-3 legs timed. The committed array is
    // the RAW curve and `makeMeasuredDepthPrice` rescales it, so this test
    // proves the rescale reproduces the timed arm bit for bit rather than to
    // within an ulp. If it ever fails, the shipped arm is not the arm that
    // measured -3.500 %.
    //
    // These are the values Swift computes, and they are what every E68 leg
    // ran. At depths 0, 3 and 7 they sit ONE ULP below the vector printed in
    // research/e68-results.md, because Swift evaluates `raw * (total / sum)`
    // while the Python arm manifest that produced the report evaluated
    // `raw * total / sum`. The manifest never reached the compiler; only the
    // raw array did. The published rendering is checked separately below at
    // one-ulp tolerance so both facts stay on the record.
    @Test("the measured arm is bit-identical to the vector E68 timed")
    func measuredArmMatchesTheTimedVector() {
        let timed = [0.12014290579688386, 0.13336973691819140,
                     0.15825051194819845, 0.18378135596082668,
                     0.28910578332644965, 0.19917881598825601,
                     0.16197661758144877, 0.19419427247974499]
        let published = [0.12014290579688387, 0.13336973691819140,
                         0.15825051194819845, 0.18378135596082670,
                         0.28910578332644965, 0.19917881598825600,
                         0.16197661758144877, 0.19419427247974502]
        let price = Qwen36MTPBlockSession.makeMeasuredDepthPrice()
        for depth in 0 ..< maxDepth {
            #expect(price.marginal[depth] == timed[depth])
            #expect(abs(price.marginal[depth] - published[depth])
                    <= published[depth].ulp)
        }
    }

    // Provenance. The raw curve is `h + (C(d + 2) - C(d + 1)) / V` over the
    // rung-1 isolated whole-table QMV medians, in milliseconds here where the
    // session worked in seconds, so this reconstruction is exact only to
    // rounding.
    @Test("the raw curve reconstructs from the rung-1 width medians")
    func rawCurveReconstructsFromRung1() {
        let cost = [1: 60.372, 2: 65.377, 3: 72.128, 4: 82.163, 5: 95.568,
                    6: 122.876, 7: 138.314, 8: 148.841, 9: 163.621]
        let verifyForwardMs = 60.300
        for depth in 0 ..< maxDepth {
            let step = cost[depth + 2]! - cost[depth + 1]!
            let expected = h + step / verifyForwardMs
            let raw = Qwen36MTPBlockSession.measuredRawDepthPrice[depth]
            #expect(abs(raw - expected) < 1e-5)
        }
    }

    @Test("the shipped arm is pb6")
    func shippedArmIsPB6() {
        #expect(Qwen36MTPBlockSession.depthPriceArm == .pb6)
        let shipped = Qwen36MTPBlockSession.depthPrice
        let pb6 = Qwen36MTPBlockSession.makeBoundaryDepthPrice(
            enteringVerifyWidth: Qwen36MTPBlockSession
                .passBoundaryVerifyWidth,
            tier: Qwen36MTPBlockSession.passBoundaryTierFactor)
        for depth in 0 ..< maxDepth {
            #expect(shipped.marginal[depth] == pb6.marginal[depth])
        }
        for depth in 0 ... maxDepth {
            #expect(shipped.cumulative[depth] == pb6.cumulative[depth])
        }
    }

    // The pbfit arm drafts SHORTER than the flat price at both ranked
    // acceptance rates. That is the mechanism E68 measured: the vector moves
    // rounds off the expensive width-6 step onto width 5. E75 showed the
    // mechanism is table-specific, so pbfit is retained as a research arm and
    // this test pins what the arm still does, not what ships.
    @Test("the pbfit arm shortens the walk against the flat price")
    func pbfitArmShortensTheWalk() {
        let cap = 5
        let ship = Qwen36MTPBlockSession.makeUniformDepthPrice()
        let pbfit = Qwen36MTPBlockSession.makeMeasuredDepthPrice()
        for p in [0.8351, 0.8750] {
            #expect(walkDepth(ship, cap: cap, p: p) == 5)
            #expect(walkDepth(pbfit, cap: cap, p: p) == 4)
        }
    }
}
