import Foundation
import MLX
import MLXNN
import MLXRandom
import Metal
import ObjectiveC
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

/// The probe below swizzles Metal dispatch entry points for the whole process,
/// so it is opt-in for the same reason `E57SdpaChunkDispatchCountTests` is: a
/// plain `swift test` would otherwise count the dispatches of every other GPU
/// suite running beside it.
private func e90ProbeEnabled() -> Bool {
    ProcessInfo.processInfo.environment["MLXFAST_E90_DISPATCH_COUNT"] == "1"
}

@Suite(.serialized)
struct E90GatherLhsDispatchCountTests {

    /// E90 follow-up, the price of the change rather than its correctness.
    /// Omitting `lhsIndices` makes `gather_qmm` synthesise `arange(1, uint32)`,
    /// which is a real GPU dispatch on every draft step. Counting dispatches is
    /// exact even though the counting hook perturbs timing, so this measures
    /// the mechanism directly instead of hunting for it under timing noise.
    @Test(
        .enabled(
            if: e90ProbeEnabled(),
            "set MLXFAST_E90_DISPATCH_COUNT=1 to run the GPU probe"))
    func cachedGatherLhsRemovesOneDispatchPerCall() throws {
        try #require(E90TestDispatchCounter.install(), "no compute encoder hook")
        MLXRandom.seed(90)
        let inDimensions = 5120
        let rows = 512
        let candidates = 32

        let dense = (MLXRandom.normal([rows, inDimensions]) * 0.05)
            .asType(.bfloat16)
        let (weight, scales, zeroPoints) = MLX.quantized(
            dense, groupSize: 64, bits: 4, mode: .affine)
        let biases = try #require(zeroPoints)
        let w = weight.reshaped([rows, 1, inDimensions / 8])
        let s = scales.reshaped([rows, 1, inDimensions / 64])
        let b = biases.reshaped([rows, 1, inDimensions / 64])
        let x = MLXRandom.normal([1, 1, inDimensions]).asType(.bfloat16)
        let ids = MLXArray(
            (0 ..< candidates).map { UInt32((rows - 1) - $0 * 7) })
        let cachedLhs = MLXArray([UInt32(0)])

        func run(lhs: MLXArray?) {
            let out = gatherQuantizedMM(
                x, w, scales: s, biases: b,
                lhsIndices: lhs, rhsIndices: ids,
                transpose: true, groupSize: 64, bits: 4, mode: .affine)
            MLX.eval(out)
        }

        // Warm both forms so pipeline creation and buffer allocation are not
        // inside either measured window.
        run(lhs: nil)
        run(lhs: cachedLhs)

        let repeats = 8
        let synthesised = E90TestDispatchCounter.measure {
            for _ in 0 ..< repeats { run(lhs: nil) }
        }
        let cached = E90TestDispatchCounter.measure {
            for _ in 0 ..< repeats { run(lhs: cachedLhs) }
        }

        #expect(
            synthesised % repeats == 0 && cached % repeats == 0,
            "dispatch counts \(synthesised) and \(cached) are not per-call stable")
        let perCall = Double(synthesised - cached) / Double(repeats)
        #expect(
            perCall == 1.0,
            "expected one dispatch removed per call, measured \(perCall) from \(synthesised) synthesised against \(cached) cached over \(repeats) calls")
    }
}

/// Counts Metal compute dispatches by swizzling the two selectors MLX issues
/// work through. Test-only: it never reaches the submitted surface, and it is
/// a counter, not a timer, so the lock it takes cannot bias the result.
private enum E90TestDispatchCounter {
    private static let lock = NSLock()
    private nonisolated(unsafe) static var count = 0
    private nonisolated(unsafe) static var installed = false

    static func install() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        if installed { return true }
        guard let device = MTLCreateSystemDefaultDevice(),
            let queue = device.makeCommandQueue(),
            let buffer = queue.makeCommandBuffer(),
            let encoder = buffer.makeComputeCommandEncoder()
        else { return false }
        let encoderClass: AnyClass = type(of: encoder as AnyObject)
        encoder.endEncoding()
        let hooked = [
            "dispatchThreadgroups:threadsPerThreadgroup:",
            "dispatchThreads:threadsPerThreadgroup:",
        ].map { swizzle(encoderClass, $0) }
        installed = hooked.contains(true)
        return installed
    }

    static func measure(_ body: () -> Void) -> Int {
        lock.lock()
        count = 0
        lock.unlock()
        body()
        lock.lock()
        defer { lock.unlock() }
        return count
    }

    fileprivate static func note() {
        lock.lock()
        count += 1
        lock.unlock()
    }

    private typealias DispatchIMP =
        @convention(c) (AnyObject, Selector, MTLSize, MTLSize) -> Void

    private static func swizzle(_ cls: AnyClass, _ name: String) -> Bool {
        let selector = NSSelectorFromString(name)
        guard let method = class_getInstanceMethod(cls, selector) else {
            return false
        }
        let original = unsafeBitCast(
            method_getImplementation(method), to: DispatchIMP.self)
        let replacement: @convention(block) (AnyObject, MTLSize, MTLSize) -> Void = {
            encoder, grid, group in
            E90TestDispatchCounter.note()
            original(encoder, selector, grid, group)
        }
        method_setImplementation(method, imp_implementationWithBlock(replacement))
        return true
    }
}
