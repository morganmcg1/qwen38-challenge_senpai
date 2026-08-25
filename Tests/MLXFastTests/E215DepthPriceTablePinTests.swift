import Foundation
import Testing

@testable import MLXFastModel

// E215 Q3 -- pin the shipped depth-price schedule so a silent table regression
// fails the suite.
//
// WHAT SHIPS. E214 promoted the `stepq` arm: `depthPriceArm == .stepq`, and
// the walk reads `makeThresholdDepthPrice(shippedDepthThresholds)`. Those eight
// doubles ARE the schedule. Nothing else in the tree asserts either fact, so an
// edit that flips the arm back to `.ship`, or that rounds one threshold while
// refitting, changes the scored draft depth on every round and still passes a
// green suite.
//
// WHY HEXFLOAT. The table came out of a fit, not out of a person, and the
// fitted artifact is compared as doubles. A decimal literal in a test can round
// to a different double than the source literal it is supposed to pin, so the
// pin is written as the exact bit pattern of each value.
//
// FALSIFICATION. `pinCanFail` is the positive control: it perturbs one
// threshold by a single ulp and shows that both the value pin and the derived
// price table notice. Without it the pin would be a guard whose failure mode is
// untested.
//
// Research instrument. `Tests/` is never packaged into a submission.
@Suite("E215 shipped depth-price table pin")
struct E215DepthPriceTablePinTests {
    /// The exact doubles behind `research/e211-artifacts/step-price.json`,
    /// key `receipt_proof.forward_guarded` (FINDING 554, shipped by E214).
    static let pinnedThresholdsHex = [
        "0x1.eb65abab49018p-4",
        "0x1.3cd94a284f571p-3",
        "0x1.0d1de16a67c83p-1",
        "0x1.4d3a8b9cf3e83p-1",
        "0x1.a836c532de064p-1",
        "0x1.a836c532de064p-1",
        "0x1.a836c532de064p-1",
        "0x1.a836c532de064p-1",
    ]

    static func hex(_ values: [Double]) -> [String] {
        values.map { String(format: "%a", $0) }
    }

    @Test("the shipped arm is stepq")
    func armIsStepq() {
        #expect(Qwen36MTPBlockSession.depthPriceArm == .stepq)
        #expect(Qwen36MTPBlockSession.depthPriceArm.rawValue == "stepq")
    }

    @Test("the shipped threshold table is bit-identical to the fitted table")
    func tableIsPinned() {
        let shipped = Qwen36MTPBlockSession.shippedDepthThresholds
        #expect(shipped.count == Qwen36MTPLimits.maxDepth)
        #expect(Self.hex(shipped) == Self.pinnedThresholdsHex)
        #expect(zip(shipped, shipped.dropFirst()).allSatisfy { $0 <= $1 })
    }

    @Test("the walk reads the shipped table and nothing else")
    func walkReadsShippedTable() {
        let expected = Qwen36MTPBlockSession.makeThresholdDepthPrice(
            Qwen36MTPBlockSession.shippedDepthThresholds)
        let live = Qwen36MTPBlockSession.depthPrice
        #expect(Self.hex(live.marginal) == Self.hex(expected.marginal))
        #expect(Self.hex(live.cumulative) == Self.hex(expected.cumulative))
    }

    @Test("the pin can fail: one ulp moves both the table and the price")
    func pinCanFail() {
        var perturbed = Qwen36MTPBlockSession.shippedDepthThresholds
        perturbed[0] = perturbed[0].nextUp
        #expect(Self.hex(perturbed) != Self.pinnedThresholdsHex)
        let perturbedPrice = Qwen36MTPBlockSession.makeThresholdDepthPrice(perturbed)
        #expect(
            Self.hex(perturbedPrice.marginal)
                != Self.hex(Qwen36MTPBlockSession.depthPrice.marginal))
    }
}
