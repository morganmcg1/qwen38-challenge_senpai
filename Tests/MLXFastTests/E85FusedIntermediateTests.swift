import Foundation
import MLX
import MLXNN
import MLXRandom
import Testing

@testable import MLXLLM

/// E85 numerical guards for the two per-draft fusions.
///
/// Both arms remove a materialised intermediate from the proposal-head path.
/// The head is proposal-only, so neither arm has to be bit-exact against the
/// eager path, but a silent arithmetic error would show up only as a slow
/// acceptance-rate collapse in a full run. These checks cost a few seconds.
struct E85FusedIntermediateTests {

    /// Arm (a): the fused quantized-embedding dual-norm-concat kernel must
    /// reproduce `qwen35DualRMSNormConcat(a: embedTokens(ids), b: hidden)`
    /// exactly. The kernel dequantizes with the same `scale * d + bias`
    /// expression in `bfloat` that `affine_dequantize` uses, so every output
    /// bit must match, not merely agree to a tolerance.
    @Test func fusedEmbedDualRMSNormConcatMatchesEagerBitwise() throws {
        MLXRandom.seed(85)
        let dimensions = 5120
        let vocabulary = 96
        let rows = 3

        let table = (MLXRandom.normal([vocabulary, dimensions]) * 0.02)
            .asType(.bfloat16)
        let embedding = QuantizedEmbedding(
            weight: table, groupSize: 64, bits: 4, mode: .affine)
        let zeroPoints = try #require(embedding.biases)

        let ids = MLXArray([3, 17, 95] as [Int32]).reshaped([1, rows])
        let hidden = MLXRandom.normal([1, rows, dimensions]).asType(.bfloat16)
        let aWeight = MLXRandom.normal([dimensions]).asType(.bfloat16)
        let bWeight = MLXRandom.normal([dimensions]).asType(.bfloat16)
        let eps: Float = 1e-6

        let eager = qwen35DualRMSNormConcat(
            a: embedding(ids), b: hidden,
            aWeight: aWeight, bWeight: bWeight, eps: eps)
        let fused = qwen35EmbedDualRMSNormConcat(
            ids: ids,
            embedWeight: embedding.weight,
            embedScales: embedding.scales,
            embedBiases: zeroPoints,
            b: hidden,
            aWeight: aWeight, bWeight: bWeight, eps: eps)

        #expect(fused.shape == eager.shape)
        #expect(fused.shape == [1, rows, dimensions * 2])

        let differing = MLX.sum(MLX.notEqual(fused, eager)).item(Int.self)
        #expect(differing == 0, "fused concat differs in \(differing) elements")
    }

    /// The embedding half must actually depend on the id. A kernel that read
    /// row 0 for every row would still match on a single-row case.
    @Test func fusedEmbedDualRMSNormConcatIsIdDependent() throws {
        MLXRandom.seed(86)
        let dimensions = 5120
        let table = (MLXRandom.normal([64, dimensions]) * 0.02).asType(.bfloat16)
        let embedding = QuantizedEmbedding(
            weight: table, groupSize: 64, bits: 4, mode: .affine)
        let zeroPoints = try #require(embedding.biases)

        let hidden = MLXRandom.normal([1, 1, dimensions]).asType(.bfloat16)
        let aWeight = MLXRandom.normal([dimensions]).asType(.bfloat16)
        let bWeight = MLXRandom.normal([dimensions]).asType(.bfloat16)

        func run(_ id: Int32) -> MLXArray {
            qwen35EmbedDualRMSNormConcat(
                ids: MLXArray([id]).reshaped([1, 1]),
                embedWeight: embedding.weight,
                embedScales: embedding.scales,
                embedBiases: zeroPoints,
                b: hidden,
                aWeight: aWeight, bWeight: bWeight, eps: 1e-6)
        }

        let first = run(0)
        let second = run(63)
        let differing = MLX.sum(MLX.notEqual(first, second)).item(Int.self)
        #expect(differing > 0, "fused concat ignored the token id")
    }

    /// Arm (b): `gatherQuantizedMM` over batch-dimension views must rank the
    /// 32 rerank candidates the same way as three `take` calls followed by
    /// `quantizedMM`. The rerank kernel only reads the ordering, so an exact
    /// bit match is not required, but the winner must not move.
    @Test func gatherQuantizedMMRerankMatchesTakeThenQuantizedMM() throws {
        MLXRandom.seed(87)
        let inDimensions = 5120
        let rows = 512
        let candidates = 32

        let dense = (MLXRandom.normal([rows, inDimensions]) * 0.05)
            .asType(.bfloat16)
        let (weight, scales, zeroPoints) = MLX.quantized(
            dense, groupSize: 64, bits: 4, mode: .affine)
        let biases = try #require(zeroPoints)

        let x = MLXRandom.normal([1, 1, inDimensions]).asType(.bfloat16)
        // Deliberately unsorted, matching what `qwen35DraftTop32` returns.
        let ids = MLXArray(
            (0 ..< candidates).map { UInt32((rows - 1) - $0 * 7) })

        let eager = quantizedMM(
            x,
            MLX.take(weight, ids, axis: 0),
            scales: MLX.take(scales, ids, axis: 0),
            biases: MLX.take(biases, ids, axis: 0),
            transpose: true, groupSize: 64, bits: 4, mode: .affine
        ).reshaped([candidates])

        let gathered = gatherQuantizedMM(
            x,
            weight.reshaped([rows, 1, inDimensions / 8]),
            scales: scales.reshaped([rows, 1, inDimensions / 64]),
            biases: biases.reshaped([rows, 1, inDimensions / 64]),
            rhsIndices: ids,
            transpose: true, groupSize: 64, bits: 4, mode: .affine
        ).reshaped([candidates])

        #expect(gathered.shape == eager.shape)

        let eagerBest = MLX.argMax(eager).item(Int.self)
        let gatheredBest = MLX.argMax(gathered).item(Int.self)
        #expect(eagerBest == gatheredBest)

        let spread = (MLX.max(eager) - MLX.min(eager)).item(Float.self)
        let worst = MLX.max(MLX.abs(gathered - eager)).item(Float.self)
        #expect(
            worst <= spread * 1e-3,
            "gather_qmm logits drifted by \(worst) against a spread of \(spread)")

        // E90 follow-up: a cached `uint32` left-hand index must be BIT-identical
        // to the array `gather_qmm` synthesises when the argument is omitted,
        // because the only difference is that the `arange` dispatch disappears.
        let cachedLhs = gatherQuantizedMM(
            x,
            weight.reshaped([rows, 1, inDimensions / 8]),
            scales: scales.reshaped([rows, 1, inDimensions / 64]),
            biases: biases.reshaped([rows, 1, inDimensions / 64]),
            lhsIndices: MLXArray([UInt32(0)]), rhsIndices: ids,
            transpose: true, groupSize: 64, bits: 4, mode: .affine
        ).reshaped([candidates])
        #expect(cachedLhs.shape == gathered.shape)
        #expect(MLX.all(MLX.equal(cachedLhs, gathered)).item(Bool.self))
    }
}
