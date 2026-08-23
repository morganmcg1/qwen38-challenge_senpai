import Testing

@testable import MLXFastModel

/// E150 R3. The readback arm is a cost gate: it must be invisible to the
/// candidate build and it must not be able to move a depth decision. These
/// tests pin both properties at compile time so a leg session cannot ship an
/// arm by accident, the way `QwenMTPDepthPriceTests` pins the depth price.
@Suite("E150 readback arm")
struct E150ReadbackArmTests {
    /// The submitted build must pay nothing. `MLX_E150_READBACK_ARM` is unset
    /// in the ranked worker, and an unset or misspelled value must fall to
    /// `.off` rather than to the most expensive arm.
    @Test("default is off")
    func defaultIsOff() {
        #expect(Qwen36MTPBlockSession.readbackArm == .off)
    }

    /// The raw values are the exact strings an ABBA leg writes into the
    /// environment. A rename here silently turns every arm leg into an `.off`
    /// leg and the contrast would read zero for the wrong reason.
    @Test("arm names are the strings a leg sets")
    func armNamesAreStable() {
        #expect(Qwen36MTPBlockSession.ReadbackArm(rawValue: "off") == .off)
        #expect(
            Qwen36MTPBlockSession.ReadbackArm(rawValue: "firstOnly")
                == .firstOnly)
        #expect(
            Qwen36MTPBlockSession.ReadbackArm(rawValue: "perStep") == .perStep)
        #expect(Qwen36MTPBlockSession.ReadbackArm(rawValue: "perstep") == nil)
        #expect(Qwen36MTPBlockSession.ReadbackArm(rawValue: "1") == nil)
    }

    /// RULE 79. The contrast is only a cost measurement if the arm cannot
    /// reach the schedule. `readbackArm` is read at exactly two call sites,
    /// both after the width is already fixed, so the two inputs that choose a
    /// width must still read at their shipped values while an arm leg runs.
    /// This is the compiled half of that claim; the run-time half is the
    /// `d=` histogram equality the witness legs check against `rb=`.
    @Test("the schedule inputs an arm leg must not move")
    func scheduleInputsAreUnmoved() {
        #expect(Qwen36MTPBlockSession.scheduleArm == .linearised)
        let price = Qwen36MTPBlockSession.linearisedDepthPrice
        #expect(price.cumulative.first == 1.0)
        #expect(price.marginal.count == Qwen36MTPLimits.maxDepth)
    }
}
