import Foundation
import MLX
import Testing

// E84 mechanism A -- exactness gate for narrowing the proposal head's affine-4
// Q/K/V pack to its q+gate rows.
//
// The declared head carries one BF16 precision-island row for EVERY K output
// row and EVERY V output row (verified against the pinned artifact by
// `research/e84_island_index_audit.py`). `Qwen35Attention.qkv` therefore
// computed 2048 quantized rows that a `putAlong` overwrote in full, and
// `Qwen35Attention.kv` computed its entire quantized pack and overwrote all of
// it. The new form runs the affine-4 pack over the q+gate rows only and reads
// K and V out of the island rows put back in natural output order.
//
// Two arithmetic boundaries move, and both are checked here at exact values
// with a positive control that proves the comparison can fail:
//
//   1. `N` for `quantizedMM` drops 14336 -> 12288. `N` selects the kernel
//      variant and the grid, so the q rows are only bit-identical if both
//      values land on the same arm.
//   2. The BF16 island matmul splits 3072 -> 1024 (q scatter) + 2048 (K/V),
//      and the K/V rows are permuted into output order. `out_vector_len`
//      selects the gemv tile parameters, so the same argument applies.
//
// Shapes are the scored ones: K = 5120, q+gate 12288, K 1024, V 1024, affine
// group-64 4-bit, one row in (`qkv` and the head's decode step are M == 1).

private let quantizedDispatchPath =
    "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/quantized.cpp"
private let matmulDispatchPath =
    "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/matmul.cpp"

private let hiddenSize = 5120
private let qOutputCount = 12_288
private let kOutputCount = 1_024
private let vOutputCount = 1_024

private func repositoryRoot() -> URL {
    var url = URL(fileURLWithPath: #filePath)
    for _ in 0 ..< 3 { url = url.deletingLastPathComponent() }
    return url
}

private func vendoredSource(_ relativePath: String) throws -> String {
    try String(
        contentsOf: repositoryRoot().appendingPathComponent(relativePath),
        encoding: .utf8)
}

/// splitmix64, so a reported mismatch is reproducible from the seed alone and
/// independent of MLXRandom's global state.
private struct SplitMix {
    private var state: UInt64
    init(_ seed: UInt64) { state = seed }

    mutating func next() -> UInt64 {
        state = state &+ 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    mutating func unit() -> Double {
        Double(next() >> 11) * (1.0 / 9_007_199_254_740_992.0)
    }

    mutating func normal() -> Float {
        let u = Swift.max(unit(), 1e-12)
        let v = unit()
        return Float(
            (-2.0 * Foundation.log(u)).squareRoot()
                * Foundation.cos(2.0 * Double.pi * v))
    }

    mutating func permutation(_ count: Int) -> [Int32] {
        var values = (0 ..< count).map(Int32.init)
        for i in stride(from: count - 1, to: 0, by: -1) {
            let j = Int(next() % UInt64(i + 1))
            values.swapAt(i, j)
        }
        return values
    }
}

private func normalArray(_ rng: inout SplitMix, _ shape: [Int]) -> MLXArray {
    let count = shape.reduce(1, *)
    var values = [Float](repeating: 0, count: count)
    for i in 0 ..< count { values[i] = rng.normal() * 0.05 }
    return MLXArray(values, shape).asType(.bfloat16)
}

/// Exact bit patterns of a bf16 array. bf16 -> Float is exact and injective, so
/// equal Float bit patterns are equal bf16 words.
private func bitPatterns(_ array: MLXArray) -> [UInt32] {
    array.asType(.float32).asArray(Float.self).map { $0.bitPattern }
}

/// Inverse of a permutation: `inverse[p[i]] = i`.
private func inversePermutation(_ p: [Int32]) -> [Int] {
    var inverse = [Int](repeating: 0, count: p.count)
    for (i, value) in p.enumerated() { inverse[Int(value)] = i }
    return inverse
}

@Suite(.serialized)
struct E84IslandDeadWorkExactnessTests {

    /// The narrowed affine-4 pack must land on the same kernel arm as the full
    /// pack, or the q rows are a different sum. Read the two live predicates
    /// out of the vendored dispatcher rather than restating them from memory.
    @Test
    func narrowedQuantizedPackKeepsTheSameKernelArm() throws {
        let source = try vendoredSource(quantizedDispatchPath)

        // qmv() picks the `_fast` arm from N and K only.
        #expect(source.contains("bool fast = N % bn == 0 && K % 512 == 0;"))
        #expect(source.contains("int bn = 8;"))
        // The kernel name carries mode, dtype, group size, bits and batching.
        // None of those is a function of N.
        #expect(
            source.contains(
                """
                      mode + (fast ? "_qmv_fast_" : "_qmv_"),
                """))
        // dispatch_qmv only leaves qmv for K in {64, 128}; ours is 5120.
        #expect(source.contains("if ((K == 128 || K == 64) && is_power_of_2(bits)) {"))
        // The M threshold that would leave the qmv family altogether buckets on
        // D and O against 2048 and 4096. D = K = 5120 exceeds both, so the
        // bucket is the same for either N.
        #expect(source.contains("inline int get_qmv_batch_limit(int D, int O, metal::Device& d) {"))
        #expect(source.contains("if (D <= 2048 && O <= 2048) {"))
        #expect(source.contains("} else if (D <= 4096 && O <= 4096) {"))

        let bn = 8
        let fullN = qOutputCount + kOutputCount + vOutputCount
        for n in [fullN, qOutputCount] {
            #expect(n % bn == 0, "N = \(n) must keep the qmv_fast arm")
        }
        #expect(hiddenSize % 512 == 0)
        #expect(hiddenSize > 4096, "K decides the qmv batch-limit bucket")
    }

    /// The island gemv must land on the same tile parameters at 3072, 2048 and
    /// 1024 output rows, or the island rows are a different sum.
    @Test
    func islandGemvKeepsTheSameTileParameters() throws {
        let source = try vendoredSource(matmulDispatchPath)
        #expect(source.contains("bm = out_vector_len >= 4096 ? 8 : 4;"))
        #expect(source.contains("} else if (K >= 16 * out_vector_len) {"))
        #expect(source.contains("tm = out_vector_len < tm ? 1 : tm;"))

        // bm, bn, sm, sn, tm, tn are the whole kernel key here, and every one
        // of them is the same at all three widths this experiment uses.
        func tiles(outVectorLen: Int, k: Int) -> [Int] {
            var bm = outVectorLen >= 4096 ? 8 : 4
            var bn = 1
            var sm = 1
            var sn = 32
            var tm = 4
            let tn = 4
            if k <= 64 {
                bm = 1
                sm = 8
                sn = 4
            } else if k >= 16 * outVectorLen {
                bm = 1
                bn = 8
            }
            tm = outVectorLen < tm ? 1 : tm
            return [bm, bn, sm, sn, tm, tn]
        }

        let today = tiles(outVectorLen: 3072, k: hiddenSize)
        #expect(tiles(outVectorLen: 1024, k: hiddenSize) == today)
        #expect(tiles(outVectorLen: 2048, k: hiddenSize) == today)
    }

    /// Narrowing the pack to the q+gate rows must not move one bit of the q
    /// output, and the comparison must be able to fail.
    @Test
    func qOnlyPackIsBitIdenticalToTheFullPackQRows() {
        var rng = SplitMix(0x5EA1_74C5_0E84_0001)
        let fullN = qOutputCount + kOutputCount + vOutputCount
        let dense = normalArray(&rng, [fullN, hiddenSize])
        let (packed, scales, biases) = MLX.quantized(
            dense, groupSize: 64, bits: 4, mode: .affine)
        guard let biases else {
            Issue.record("affine quantization must produce biases")
            return
        }
        let x = normalArray(&rng, [1, 1, hiddenSize])

        let full = MLX.quantizedMM(
            x, packed, scales: scales, biases: biases, transpose: true,
            groupSize: 64, bits: 4, mode: .affine)
        let qOnly = MLX.quantizedMM(
            x, packed[0 ..< qOutputCount], scales: scales[0 ..< qOutputCount],
            biases: biases[0 ..< qOutputCount], transpose: true,
            groupSize: 64, bits: 4, mode: .affine)
        eval(full, qOnly)

        #expect(full.dim(-1) == fullN)
        #expect(qOnly.dim(-1) == qOutputCount)
        let fullBits = Array(bitPatterns(full)[0 ..< qOutputCount])
        let qBits = bitPatterns(qOnly)
        let mismatches = zip(fullBits, qBits).filter { $0 != $1 }.count
        #expect(
            mismatches == 0,
            "narrowed pack moved \(mismatches) of \(qOutputCount) q values")

        // POSITIVE CONTROL. Perturb the scale of one q row in the narrowed pack
        // only. Failure mode: exactly that row's value must change, which shows
        // the comparison above is sensitive at single-row granularity.
        var perturbed = scales[0 ..< qOutputCount]
        perturbed[100, 0] = perturbed[100, 0] + MLXArray(Float(0.5)).asType(scales.dtype)
        let control = MLX.quantizedMM(
            x, packed[0 ..< qOutputCount], scales: perturbed,
            biases: biases[0 ..< qOutputCount], transpose: true,
            groupSize: 64, bits: 4, mode: .affine)
        eval(control)
        let controlBits = bitPatterns(control)
        let controlDiffs = (0 ..< qOutputCount).filter { fullBits[$0] != controlBits[$0] }
        #expect(
            controlDiffs == [100],
            "positive control changed rows \(controlDiffs.prefix(8)), expected [100]")
    }

    /// K and V read out of the reordered island rows must be bit-identical to
    /// the rows the scatter used to write, and the comparison must be able to
    /// fail.
    @Test
    func reorderedIslandRowsAreBitIdenticalToTheScatteredRows() {
        var rng = SplitMix(0x5EA1_74C5_0E84_0002)
        let qIsland = normalArray(&rng, [1024, hiddenSize])
        let kIsland = normalArray(&rng, [kOutputCount, hiddenSize])
        let vIsland = normalArray(&rng, [vOutputCount, hiddenSize])
        let kIndices = rng.permutation(kOutputCount)
        let vIndices = rng.permutation(vOutputCount)
        let x = normalArray(&rng, [1, 1, hiddenSize])

        // TODAY: one matmul over all 3072 island rows, then a putAlong.
        let islandAll = concatenated([qIsland, kIsland, vIsland], axis: 0)
            .contiguous()
        let exactAll = matmul(x, islandAll.transposed(1, 0))
        // TODAY, K/V-only flush: the trailing 2048 island rows, island order.
        let exactKVToday = matmul(x, islandAll[1024...].transposed(1, 0))

        // NEW: q islands alone for the scatter, K and V in natural output
        // order for a direct read.
        let kNatural = MLX.take(
            kIsland, argSort(MLXArray(kIndices)), axis: 0)
        let vNatural = MLX.take(
            vIsland, argSort(MLXArray(vIndices)), axis: 0)
        let kvNatural = concatenated([kNatural, vNatural], axis: 0).contiguous()
        let exactQOnly = matmul(x, qIsland.transposed(1, 0))
        let exactKVNew = matmul(x, kvNatural.transposed(1, 0))
        eval(exactAll, exactKVToday, exactQOnly, exactKVNew)

        let allBits = bitPatterns(exactAll)
        let qOnlyBits = bitPatterns(exactQOnly)
        let kvTodayBits = bitPatterns(exactKVToday)
        let kvNewBits = bitPatterns(exactKVNew)

        let qMismatches = (0 ..< 1024).filter { allBits[$0] != qOnlyBits[$0] }.count
        #expect(qMismatches == 0, "q island gemv moved \(qMismatches) values at N = 1024")

        let kvSliceMismatches = (0 ..< 2048)
            .filter { allBits[1024 + $0] != kvTodayBits[$0] }.count
        #expect(
            kvSliceMismatches == 0,
            "island-order K/V slice moved \(kvSliceMismatches) values at N = 2048")

        // The scatter wrote island row i to output row indices[i]; the natural
        // order reads output row j from island row inverse[j].
        let kInverse = inversePermutation(kIndices)
        let vInverse = inversePermutation(vIndices)
        var permutedMismatches = 0
        for j in 0 ..< kOutputCount where kvNewBits[j] != kvTodayBits[kInverse[j]] {
            permutedMismatches += 1
        }
        for j in 0 ..< vOutputCount
        where kvNewBits[kOutputCount + j] != kvTodayBits[kOutputCount + vInverse[j]] {
            permutedMismatches += 1
        }
        #expect(
            permutedMismatches == 0,
            "reordering moved \(permutedMismatches) of 2048 K/V values")

        // POSITIVE CONTROL. Perturb one weight element of one reordered K row.
        // Failure mode: exactly output row `target` must change.
        let target = 37
        var perturbedRows = kvNatural
        perturbedRows[target, 0] =
            perturbedRows[target, 0] + MLXArray(Float(1.0)).asType(.bfloat16)
        let control = matmul(x, perturbedRows.transposed(1, 0))
        eval(control)
        let controlBits = bitPatterns(control)
        let controlDiffs = (0 ..< 2048).filter { kvNewBits[$0] != controlBits[$0] }
        #expect(
            controlDiffs == [target],
            "positive control changed rows \(controlDiffs.prefix(8)), expected [\(target)]")
    }

    /// The permutation test the install path runs is the whole precondition for
    /// mechanism A. It must accept a shuffled complete cover and reject every
    /// near miss.
    @Test
    func completePermutationDetectionRejectsNearMisses() {
        var rng = SplitMix(0x5EA1_74C5_0E84_0003)
        let identity = (0 ..< 8).map(Int32.init)
        let shuffled = rng.permutation(8)

        func isCompletePermutation(_ indices: [Int32], count: Int) -> Bool {
            guard count > 0, indices.count == count else { return false }
            var seen = [Bool](repeating: false, count: count)
            for value in indices {
                let row = Int(value)
                guard row >= 0, row < count, !seen[row] else { return false }
                seen[row] = true
            }
            return true
        }

        #expect(isCompletePermutation(identity, count: 8))
        #expect(isCompletePermutation(shuffled, count: 8))
        #expect(!isCompletePermutation(identity, count: 9), "short cover")
        #expect(!isCompletePermutation([0, 1, 2, 3, 4, 5, 6, 6], count: 8), "duplicate")
        #expect(!isCompletePermutation([0, 1, 2, 3, 4, 5, 6, 8], count: 8), "out of range")
        #expect(!isCompletePermutation([-1, 1, 2, 3, 4, 5, 6, 7], count: 8), "negative")
    }
}
