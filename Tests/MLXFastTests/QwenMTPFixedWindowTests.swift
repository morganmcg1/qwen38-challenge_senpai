@testable import MLXFastModel
import Testing

@Suite
struct QwenMTPFixedWindowTests {
    @Test
    func eosInsideAnAcceptedPrefixDoesNotEndTheWindow() {
        let eos = 151_645
        let drafts = [41, eos, 73, 89]
        let targetTokens = [41, eos, 73, 97, 101]

        let accepted = Qwen36MTPBlockSession.acceptedDraftPrefixCount(
            drafts: drafts, verifyArgmax: targetTokens)
        let committed = [13] + Array(drafts.prefix(accepted))

        #expect(accepted == 3)
        #expect(Array(committed) == [13, 41, eos, 73])
        #expect(accepted + (drafts.count - accepted) + 1 == drafts.count + 1)
    }

    @Test
    func onlyATargetMismatchEndsTheAcceptedPrefix() {
        let eos = 151_645
        let drafts = [41, 67, eos]
        let targetTokens = [41, 71, eos, 101]

        #expect(
            Qwen36MTPBlockSession.acceptedDraftPrefixCount(
                drafts: drafts, verifyArgmax: targetTokens) == 1)
    }
}
