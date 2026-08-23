import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXRandom
import Testing

// E153 R2 -- is the merged wide-decode SDPA kernel BITWISE identical to the
// shipped two-call split at the scored widths?
//
// WHAT IS COMPARED, AND WHY IT IS THE SPLIT AND NOT THE FALLBACK.
// `AttentionUtils.swift` guards `B == 1, 6 <= qL <= 9, kL >= qL, .causal` and
// issues TWO fused vector calls, `5` rows over `keys[..<kL-(qL-5)]` and
// `qL-5` rows over the full keys, then concatenates. That pair is what the
// scored worker runs today, so it is the reference. E149 C1 measured the
// composed `sdpa_full`-less fallback at 0.001953125-0.00390625 max abs from
// the split; the fallback is therefore NOT a valid reference and is not used
// here.
//
// WHY A MERGED VECTOR CALL CAN BE BIT-EXACT.
// `sdpa_vector.h:98` walks the KV axis as `for (i = simd_gid; i < N; i += BN)`
// with `BN = 32`, so simdgroup g accumulates exactly the KV indices congruent
// to g modulo 32, ascending, for ANY N. The causal predicate `:101` is
// `i <= N - tpg.y + q_seq_idx`. For chunk A (N = kL-(qL-5), tpg.y = 5, row r)
// that is `i <= kL-qL+r`; for the merged call (N = kL, tpg.y = qL, row r) it
// is `i <= kL-qL+r`. Identical. For chunk B (N = kL, tpg.y = qL-5, row j) it
// is `i <= kL-qL+5+j`, which is the merged predicate at row 5+j. Identical.
// Masked keys are skipped whole, so they touch neither the running max nor the
// running sum. Same visited set, same order, same arithmetic expressions =>
// same bits.
//
// WHAT WOULD BREAK IT, AND WHAT THIS TEST PINS.
//   * the 2-pass route. MLX picks `sdpa_vector_2pass` at `kL >= 1024` on
//     architectures ending in `d`/`s`, which includes the ranked M5. That form
//     combines partial softmax states and is a DIFFERENT accumulation. The
//     kernel refuses `kL >= 1024` outright; `refusesTheTwoPassBand` pins the
//     refusal rather than trusting it.
//   * query layout. The vendored kernel hard-codes a row-contiguous query
//     offset and MLX copies a non-contiguous query before the call; this
//     kernel reads strides instead. Both layouts the scored path can produce
//     are compared.
//   * the KV layout. Keys arrive as a row slice of one preallocated
//     `KVCacheSimple` buffer, which `kv_copy_unless` accepts uncopied. The
//     cells below build the KV through a real `KVCacheSimple.update` loop so
//     the measured strides are the production strides, not a synthetic
//     contiguous array.
//
// RULE 101. A comparison that cannot fail proves nothing. Each cell also runs
// the split against a recomputation of itself, which must be exactly 0, and
// against a perturbed input, which must be strictly positive.
//
// Runtime gate: the standard `MLXFAST_RUN_MLX_RUNTIME_TESTS=1` sweep. `Tests/`
// is never packaged into a submission.

private struct E153Cell: Codable {
    var qL: Int
    var kL: Int
    var queryLayout: String
    var mergedRan: Bool
    var mergedVsSplitMaxAbs: Double
    var mergedVsSplitMismatchCount: Int
    var splitVsSplitMaxAbs: Double
    var positiveControlMaxAbs: Double
}

enum E153MergedSdpaSupport {
    static let queryHeads = 24
    static let kvHeads = 4
    static let headDim = 256

    static let runtimeEnabled =
        ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1"

    static var keyLengths: [Int] {
        guard let raw = ProcessInfo.processInfo.environment["MLXFAST_E153_KL"],
            !raw.isEmpty
        else { return [513, 769, 1023] }
        return raw.split(separator: ",").compactMap { Int($0) }
    }

    static var widths: [Int] {
        guard let raw = ProcessInfo.processInfo.environment["MLXFAST_E153_WIDTHS"],
            !raw.isEmpty
        else { return [6, 7, 8, 9] }
        return raw.split(separator: ",").compactMap { Int($0) }
    }

    static let scale = 1.0 / Float(Double(headDim).squareRoot())

    static func bf16(_ shape: [Int], _ seed: UInt64) -> MLXArray {
        MLXRandom.normal(shape, key: MLXRandom.key(seed)).asType(.bfloat16)
    }

    /// Drive a real `KVCacheSimple` to `length` rows so the keys and values the
    /// kernel reads carry the production strides: a row view of one
    /// preallocated buffer whose capacity is rounded up to the cache step.
    static func cacheKV(length: Int, seed: UInt64) -> (MLXArray, MLXArray) {
        let cache = KVCacheSimple()
        var lastKeys: MLXArray?
        var lastValues: MLXArray?
        var written = 0
        // One prefill-shaped update then one-row decode updates, which is the
        // exact call sequence the scored session makes.
        let chunks = [length - min(length - 1, 7)] + Array(
            repeating: 1, count: min(length - 1, 7))
        for (index, rows) in chunks.enumerated() where rows > 0 {
            let k = bf16([1, kvHeads, rows, headDim], seed &+ UInt64(index) &* 2)
            let v = bf16(
                [1, kvHeads, rows, headDim], seed &+ UInt64(index) &* 2 &+ 1)
            let (ck, cv) = cache.update(keys: k, values: v)
            lastKeys = ck
            lastValues = cv
            written += rows
        }
        precondition(written == length, "cache fill produced \(written) rows")
        return (lastKeys!, lastValues!)
    }

    /// `contiguous` is a fresh RoPE result. `headTransposed` is the
    /// `[B, L, H, D] -> transposed(0, 2, 1, 3)` view an attention layer builds;
    /// MLX's own `q_copy_unless` accepts it without a copy, so the scored path
    /// can hand either layout to the kernel.
    static func queries(qL: Int, layout: String, seed: UInt64) -> MLXArray {
        switch layout {
        case "contiguous":
            return bf16([1, queryHeads, qL, headDim], seed)
        default:
            return bf16([1, qL, queryHeads, headDim], seed).transposed(0, 2, 1, 3)
        }
    }

    /// The shipped two-call form, transcribed from `AttentionUtils.swift`.
    static func split(
        queries: MLXArray, keys: MLXArray, values: MLXArray
    ) -> MLXArray {
        let qL = queries.dim(2)
        let kL = keys.dim(2)
        let splitRow = 5
        let kSplit = kL - (qL - splitRow)
        let outA = MLXFast.scaledDotProductAttention(
            queries: queries[0..., 0..., 0 ..< splitRow, 0...],
            keys: keys[0..., 0..., 0 ..< kSplit, 0...],
            values: values[0..., 0..., 0 ..< kSplit, 0...],
            scale: scale, mask: .causal)
        let outB = MLXFast.scaledDotProductAttention(
            queries: queries[0..., 0..., splitRow..., 0...],
            keys: keys, values: values, scale: scale, mask: .causal)
        return concatenated([outA, outB], axis: 2)
    }

    /// Bitwise on bf16: compare the raw 16-bit patterns, not a float tolerance.
    /// `asType(.uint16)` would round the value, so reinterpret through
    /// `asData`.
    static func bitMismatchCount(_ a: MLXArray, _ b: MLXArray) -> Int {
        eval(a, b)
        let da = a.asData(access: .copy).data
        let db = b.asData(access: .copy).data
        precondition(da.count == db.count, "shape mismatch in bit comparison")
        var mismatches = 0
        for i in da.indices where da[i] != db[i] { mismatches += 1 }
        return mismatches
    }

    static func maxAbsDeviation(_ a: MLXArray, _ b: MLXArray) -> Double {
        let d = MLX.abs(a.asType(.float32) - b.asType(.float32)).max()
        eval(d)
        return Double(d.item(Float.self))
    }
}

@Suite(.serialized)
struct E153MergedSdpaKernelTests {
    typealias S = E153MergedSdpaSupport

    @Test(
        "E153 R2: the merged kernel is bitwise identical to the shipped split",
        .enabled(if: E153MergedSdpaSupport.runtimeEnabled))
    func mergedMatchesSplitBitwise() throws {
        var cells: [E153Cell] = []

        for layout in ["contiguous", "headTransposed"] {
            for kL in S.keyLengths {
                let (keys, values) = S.cacheKV(length: kL, seed: 0x5153_0100)
                eval(keys, values)
                for qL in S.widths where qL <= kL {
                    let q = S.queries(qL: qL, layout: layout, seed: 0x5153_0001)
                    eval(q)

                    let splitOut = S.split(queries: q, keys: keys, values: values)
                    let splitAgain = S.split(queries: q, keys: keys, values: values)
                    let perturbed = S.split(
                        queries: q, keys: keys,
                        values: (values.asType(.float32) + MLXArray(Float(1e-2)))
                            .asType(.bfloat16))
                    eval(splitOut, splitAgain, perturbed)

                    let merged = qwen35MergedSdpaVector(
                        queries: q, keys: keys, values: values,
                        scale: S.scale, qL: qL, kL: kL)
                    if let merged { eval(merged) }

                    let selfDelta = S.maxAbsDeviation(splitOut, splitAgain)
                    let controlDelta = S.maxAbsDeviation(splitOut, perturbed)
                    #expect(selfDelta == 0)
                    #expect(controlDelta > 0)

                    var mergedDelta = Double.nan
                    var mismatches = -1
                    if let merged {
                        #expect(merged.shape == splitOut.shape)
                        #expect(merged.dtype == splitOut.dtype)
                        mergedDelta = S.maxAbsDeviation(splitOut, merged)
                        mismatches = S.bitMismatchCount(splitOut, merged)
                        #expect(mergedDelta == 0)
                        #expect(mismatches == 0)
                    }

                    cells.append(
                        E153Cell(
                            qL: qL, kL: kL, queryLayout: layout,
                            mergedRan: merged != nil,
                            mergedVsSplitMaxAbs: mergedDelta,
                            mergedVsSplitMismatchCount: mismatches,
                            splitVsSplitMaxAbs: selfDelta,
                            positiveControlMaxAbs: controlDelta))
                }
            }
        }

        // Every scored cell must have actually run the merged kernel, or the
        // zero deviations above would be vacuous.
        #expect(cells.filter(\.mergedRan).count == cells.count)
        #expect(cells.count == 2 * S.keyLengths.count * S.widths.count)

        let report: [String: Any] = [
            "probe": "e153_r2_merged_sdpa_exactness",
            "harness": "local",
            "reference_arm": "shipped_two_call_split",
            "query_heads": S.queryHeads,
            "kv_heads": S.kvHeads,
            "head_dim": S.headDim,
            "kv_built_through": "KVCacheSimple.update",
            "cells": cells.map {
                [
                    "qL": $0.qL, "kL": $0.kL, "query_layout": $0.queryLayout,
                    "merged_ran": $0.mergedRan,
                    "merged_vs_split_max_abs": $0.mergedVsSplitMaxAbs,
                    "merged_vs_split_bit_mismatch_bytes":
                        $0.mergedVsSplitMismatchCount,
                    "split_vs_split_max_abs": $0.splitVsSplitMaxAbs,
                    "positive_control_max_abs": $0.positiveControlMaxAbs,
                ] as [String: Any]
            },
        ]
        let text = String(
            decoding: try JSONSerialization.data(
                withJSONObject: report, options: [.prettyPrinted, .sortedKeys]),
            as: UTF8.self)
        print("E153_R2_EXACTNESS_JSON_BEGIN")
        print(text)
        print("E153_R2_EXACTNESS_JSON_END")
        if let path = ProcessInfo.processInfo
            .environment["MLXFAST_E153_EXACTNESS_OUT"], !path.isEmpty
        {
            try text.write(toFile: path, atomically: true, encoding: .utf8)
        }
    }

    /// The kernel reproduces the single-pass accumulation only. At `kL >= 1024`
    /// MLX may route the split to `sdpa_vector_2pass`, whose partial-state
    /// combination is a different accumulation, so the merged form must refuse
    /// the whole band instead of depending on the host architecture string.
    @Test(
        "E153 R2: the merged kernel refuses the 2-pass key band",
        .enabled(if: E153MergedSdpaSupport.runtimeEnabled))
    func refusesTheTwoPassBand() throws {
        let (keys, values) = S.cacheKV(length: 1024, seed: 0x5153_0200)
        eval(keys, values)
        let q = S.queries(qL: 6, layout: "contiguous", seed: 0x5153_0002)
        eval(q)
        #expect(
            qwen35MergedSdpaVector(
                queries: q, keys: keys, values: values, scale: S.scale,
                qL: 6, kL: 1024) == nil)

        // One row below the band it must still run, so the refusal is a band
        // edge and not a blanket disable.
        let (keys1023, values1023) = S.cacheKV(length: 1023, seed: 0x5153_0201)
        eval(keys1023, values1023)
        #expect(
            qwen35MergedSdpaVector(
                queries: q, keys: keys1023, values: values1023, scale: S.scale,
                qL: 6, kL: 1023) != nil)
    }

    /// Rule 110: a warm that never ran is not a warm. The scored session calls
    /// `warmQwen35MergedSdpaVector` outside the timed window to move the JIT
    /// Metal compile out of the leg; this pins that it reports success on the
    /// shapes the session hands it.
    @Test(
        "E153 R2: the warm entry point actually runs the kernel",
        .enabled(if: E153MergedSdpaSupport.runtimeEnabled))
    func warmRuns() throws {
        let (keys, values) = S.cacheKV(length: 1030, seed: 0x5153_0300)
        eval(keys, values)
        #expect(
            warmQwen35MergedSdpaVector(
                keys: keys, values: values, queryHeads: S.queryHeads,
                scale: S.scale))
    }
}
