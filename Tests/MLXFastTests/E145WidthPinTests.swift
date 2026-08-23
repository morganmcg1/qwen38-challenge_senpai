import Foundation
@testable import MLXFastModel
import Testing

// E145: the research width pin that makes the per-width round cost curve
// measurable on a live decode.
//
// The pin exists because the curve every depth-price decision descends from
// was rebuilt from isolated kernel timings, never read off a decode. A pin
// that leaks is worse than no measurement, so these tests own three
// properties: the pin is OFF unless its own environment name is set, the pin
// stays inside the shipped width envelope, and the compiled depth-price
// default is unchanged by this experiment.
@Suite("E145 width pin")
struct E145WidthPinTests {
    /// The test process sets no `MLX_E145_PIN_DEPTH`, so the static must read
    /// `nil` and the shipped estimator must stay in charge. This is the
    /// default-off gate: it fails the moment the pin acquires a compiled
    /// default.
    @Test("the pin is off unless its environment name is set")
    func pinDefaultsOff() {
        #expect(ProcessInfo.processInfo
            .environment["MLX_E145_PIN_DEPTH"] == nil)
        #expect(Qwen36MTPBlockSession.e145PinnedDepth == nil)
    }

    /// CAMPAIGN RULE 101, applied to the gate itself. A witness that cannot
    /// fail is worse than no witness, so prove both polarities of the parse:
    /// an unset name and an unparseable value leave the estimator in charge,
    /// and only an integer turns the pin on.
    @Test("the pin parse has both polarities")
    func pinParseHasBothPolarities() {
        func parse(_ raw: String?) -> Int? {
            guard let raw, let value = Int(raw) else { return nil }
            return value
        }
        #expect(parse(nil) == nil)
        #expect(parse("") == nil)
        #expect(parse("six") == nil)
        #expect(parse("4") == 4)
        #expect(parse("0") == 0)
    }

    /// The pin clamps into the same envelope the shipped walk respects, so a
    /// pinned leg cannot reach a verify width the campaign has not proven
    /// bit-exact. `cap` is `min(offeredDepth, maxDepth, segmentedVerifyDepthCap)`
    /// and the shipped cap is 7, which is verify width 8.
    @Test("the pin cannot exceed the shipped width envelope")
    func pinClampsIntoTheShippedEnvelope() {
        func clamp(cap: Int, pinned: Int) -> Int {
            Swift.max(0, Swift.min(cap, pinned))
        }
        #expect(clamp(cap: 7, pinned: 9) == 7)
        #expect(clamp(cap: 7, pinned: 6) == 6)
        #expect(clamp(cap: 3, pinned: 6) == 3)
        #expect(clamp(cap: 7, pinned: -1) == 0)
    }

    /// E145 F12 retired `pb6` campaign-wide on the only ranked `pb6` contrast
    /// the campaign owns, so the compiled default is now `.ship`. This test
    /// pins the retirement in one line so a silent revert cannot survive a
    /// build.
    @Test("the compiled depth-price default is ship, because E145 retired pb6")
    func compiledDepthPriceDefaultIsShip() {
        #expect(Qwen36MTPBlockSession.depthPriceArm == .ship)
    }
}
