import Foundation
import MLX
import MLXLLM
import Testing

/// E141: the compact draft vocabulary cannot propose one target token in two
/// hundred, and widening it needs no new shipped data.
///
/// Two properties are checked here.
///
/// **The count family.** One prefix bound drives the control block, the real
/// count and the padded count (`Qwen35.swift:4224-4239`). Widening the prefix
/// must never duplicate a row, must keep the padded count a multiple of eight,
/// and must collapse the appended control block once the prefix already covers
/// it. This part needs no GPU and no weights.
///
/// **The licensing gate, rung 1.** `Qwen35.swift:5940-5943` claims the shipped
/// `draft_lm_head.*` is exactly `quantize(dequantize(exact compact lm_head,
/// 64, 4), 64, 2)`. If that holds bit for bit, the coarse shortlist table is a
/// pure function of the target's own fixed lm_head, so
/// `deriveCompactCoarseTable` can produce it at ANY prefix with zero shipped
/// bytes and no head re-declaration. This part reads the two artifacts the
/// process already loads and applies the same MLX ops the derivation calls, so
/// enable it with `MLXFAST_RUN_MLX_RUNTIME_TESTS=1`.
@Suite
struct E141CompactDraftVocabularyTests {
    private static let vocabulary = 248_320
    private static let shippedPrefix = 98_304
    private static let controlStart = 248_044
    private static let controlEnd = 248_070

    private static var runtimeEnabled: Bool {
        ProcessInfo.processInfo
            .environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1"
    }

    /// The test bundle's working directory is not the checkout, so resolve the
    /// weights against this source file instead.
    private static var repositoryRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    /// Candidate bounds the census names, plus both corners of the control
    /// block, which are the only places the collapse rule can be wrong.
    private static let bounds = [
        1_024, 98_304, 114_688, 131_072, 147_456, 163_840, 196_608,
        248_043, 248_044, 248_050, 248_069, 248_070, 248_320,
    ]

    @Test
    func theShippedBoundReproducesTheShippedCounts() {
        let counts = qwen35CompactDraftCounts(prefix: Self.shippedPrefix)
        #expect(counts.controlStart == Self.controlStart)
        #expect(counts.controlEnd == Self.controlEnd)
        #expect(counts.real == 98_330)
        #expect(counts.padded == 98_336)
    }

    /// The default build must stay the shipped arm. A leg that exports nothing
    /// reads `source=unset`, which is the bare-leg proof that widening is
    /// opt-in rather than the compiled default.
    @Test
    func theCompiledDefaultIsTheShippedArm() {
        let live = qwen35CompactDraftGeometry
        guard live.source == "unset" else { return }
        #expect(live.prefix == Self.shippedPrefix)
        #expect(live.real == 98_330)
        #expect(live.padded == 98_336)
    }

    @Test
    func everyBoundKeepsTheLeafGroupingAndTheRowAccounting() {
        for prefix in Self.bounds {
            let c = qwen35CompactDraftCounts(prefix: prefix)
            #expect(c.padded % 8 == 0, "padded count not a leaf multiple at \(prefix)")
            #expect(c.padded >= c.real, "padding is negative at \(prefix)")
            #expect(c.padded - c.real < 8, "over-padded at \(prefix)")
            #expect(c.real <= Self.vocabulary, "real count exceeds the vocabulary at \(prefix)")
            #expect(c.controlEnd >= c.controlStart, "inverted control block at \(prefix)")
            #expect(c.real == prefix + c.controlEnd - c.controlStart)
        }
    }

    /// The appended control block must never repeat a prefix row, and it must
    /// vanish once the prefix reaches it. A duplicated row that survives to the
    /// argmax is a proposal for the wrong token.
    @Test
    func theControlBlockNeverDuplicatesAPrefixRow() {
        for prefix in Self.bounds {
            let c = qwen35CompactDraftCounts(prefix: prefix)
            #expect(c.controlStart >= prefix, "control block overlaps the prefix at \(prefix)")
            if prefix >= Self.controlEnd {
                #expect(c.controlEnd == c.controlStart, "control block survives at \(prefix)")
                #expect(c.real == prefix, "real count carries a stale block at \(prefix)")
            }
        }
        // The full-vocabulary arm is the one rung 3 measures: every id
        // proposable, no control block, and no padding row at all.
        let full = qwen35CompactDraftCounts(prefix: Self.vocabulary)
        #expect(full.real == Self.vocabulary)
        #expect(full.padded == Self.vocabulary)
    }

    /// Compact-to-full id mapping, as `mapDraftTokenIds` computes it. Every
    /// compact row must map to a distinct, in-range target id.
    @Test
    func theIdMappingIsInjectiveAndInRangeAtEveryBound() {
        for prefix in Self.bounds {
            let c = qwen35CompactDraftCounts(prefix: prefix)
            let offset = c.controlStart - prefix
            var seen = Set<Int>()
            for id in 0 ..< c.real {
                let mapped = id < prefix ? id : id + offset
                #expect(mapped >= 0 && mapped < Self.vocabulary,
                        "id \(id) maps out of range at prefix \(prefix)")
                #expect(seen.insert(mapped).inserted,
                        "id \(id) collides at prefix \(prefix)")
            }
            #expect(seen.count == c.real)
        }
    }

    /// Rung 1. The declared coarse table is `quantize(dequantize(exact compact
    /// lm_head, 64, 4), 64, 2)` bit for bit, so deriving it at a wider prefix
    /// creates no parameter and ships no bytes.
    @Test
    func theDeclaredCoarseTableIsAPureFunctionOfTheTargetLMHead() throws {
        guard Self.runtimeEnabled else { return }
        let targetPath = ProcessInfo.processInfo
            .environment["MLXFAST_E141_TARGET_SHARD"]
            ?? Self.repositoryRoot
            .appendingPathComponent("weights/model-00003-of-00003.safetensors").path
        let headPath = ProcessInfo.processInfo
            .environment["MLXFAST_E141_DECLARED_HEAD"]
            ?? FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent(
                    ".cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head-declared/model.safetensors"
                ).path
        // A silently skipped licensing gate reads exactly like a passing one,
        // so demand the artifacts once the runtime flag asks for the check.
        try #require(FileManager.default.fileExists(atPath: targetPath),
                     "missing target shard \(targetPath)")
        try #require(FileManager.default.fileExists(atPath: headPath),
                     "missing declared head \(headPath)")

        let target = try MLX.loadArrays(url: URL(fileURLWithPath: targetPath))
        let head = try MLX.loadArrays(url: URL(fileURLWithPath: headPath))
        let counts = qwen35CompactDraftCounts(prefix: Self.shippedPrefix)

        func compactRows(_ a: MLXArray) -> MLXArray {
            concatenated(
                [
                    a[0 ..< Self.shippedPrefix],
                    a[counts.controlStart ..< counts.controlEnd],
                    a[0 ..< (counts.padded - counts.real)],
                ], axis: 0)
        }

        let exactWeight = compactRows(try #require(target["language_model.lm_head.weight"]))
        let exactScales = compactRows(try #require(target["language_model.lm_head.scales"]))
        let exactBiases = compactRows(try #require(target["language_model.lm_head.biases"]))
        let declaredWeight = try #require(head["draft_lm_head.weight"])
        let declaredScales = try #require(head["draft_lm_head.scales"])
        let declaredBiases = try #require(head["draft_lm_head.biases"])

        func derive(_ scales: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
            let rows = dequantized(
                exactWeight, scales: scales, biases: exactBiases,
                groupSize: 64, bits: 4, mode: .affine)
            let coarse = quantized(rows, groupSize: 64, bits: 2, mode: .affine)
            let biases = coarse.biases ?? MLXArray(Float(0))
            eval(coarse.wq, coarse.scales, biases)
            return (coarse.wq, coarse.scales, biases)
        }

        func differingElements(_ got: MLXArray, _ want: MLXArray) -> Int {
            #expect(got.shape == want.shape)
            #expect(got.dtype == want.dtype)
            let n = MLX.sum((got .!= want).asType(.int32))
            eval(n)
            return n.item(Int.self)
        }

        let (w, s, z) = derive(exactScales)
        #expect(differingElements(w, declaredWeight) == 0,
                "the derived 2-bit rows differ from the declared table")
        #expect(differingElements(s, declaredScales) == 0,
                "the derived 2-bit scales differ from the declared table")
        #expect(differingElements(z, declaredBiases) == 0,
                "the derived 2-bit biases differ from the declared table")

        // Rule 101 positive control: perturb one input row and require the
        // comparison to fail. A check that cannot fail is not a check.
        let perturbed = concatenated(
            [
                exactScales[0 ..< 7],
                (exactScales[7 ..< 8].asType(.float32) * 1.5)
                    .asType(exactScales.dtype),
                exactScales[8...],
            ], axis: 0)
        eval(perturbed)
        let (cw, cs, cz) = derive(perturbed)
        let controlDiffered =
            differingElements(cw, declaredWeight) > 0
            || differingElements(cs, declaredScales) > 0
            || differingElements(cz, declaredBiases) > 0
        #expect(controlDiffered, "the rung 1 comparison cannot fail")
    }
}
