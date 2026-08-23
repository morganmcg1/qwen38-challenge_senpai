import Foundation
import MLX
import MLXRandom
import Testing

// E149 arm C1 -- what does the SDPA query split actually cost per round, and
// what is the largest saving any merged form could return?
//
// THE MECHANISM. `AttentionUtils.swift:120-143` guards on
// `queries.dim(0) == 1, qL >= 6, qL <= 9, kL >= qL, .causal` and then sets
// `let split = 5`. For a verify width M in 6...9 it issues TWO sdpa calls,
// `5` rows over `keys[..<kL-(M-5)]` and `M-5` rows over the full keys, and
// concatenates. Both halves stream the KV cache, so a width-M round reads the
// KV of every full-attention layer about twice.
//
// WHY 5, AND WHY IT CANNOT BE RAISED.
// `backend/metal/scaled_dot_product_attention.cpp:633-637` reads
// `supports_sdpa_vector = (qL <= 8) && (qL <= kL)
//  && head_dim in {64,96,128,256} && (qL * gqa_factor) <= 32`
// and `:632` reads `supports_sdpa_full = qL > 8 && ... && head_dim in
// {64,80,128}`. Qwen 3.8 has head dim 256 and gqa 24/4 = 6, so sdpa_full is
// NEVER available and the vector path holds only to `floor(32/6) = 5`. At
// qL >= 6 the op falls back to composed matmul plus softmax. That dispatcher
// file is NOT in `benchmark.json` `editablePaths`, so the bound is not ours to
// move.
//
// WHY A MERGED FORM WOULD BE BIT-EXACT, AND WHY IT IS STILL UNREACHABLE.
// `sdpa_vector.h:98` walks the KV axis as `for (i = simd_gid; i < N; i += BN)`
// with `BN = 32` (`:43`), so simdgroup g always accumulates exactly the KV
// indices congruent to g modulo 32, in ascending order, for ANY N. The causal
// predicate at `:101` is `i <= (N - tpg.y + q_seq_idx)`, which for chunk A
// (N = kSplit, tpg.y = 5, row r) and for a merged call (N = kL, tpg.y = M,
// row r) both reduce to `i <= kL - M + r`. Masked keys are skipped whole, so
// they touch neither the running max nor the running sum. The visited set and
// the visit order per row are therefore IDENTICAL, and a merged vector call
// would be bit-exact. The obstruction is not arithmetic; it is that the host
// gate above lives in a file we may not edit.
//
// WHAT THIS PROBE MEASURES. At each scored kL:
//
//   single_qN  ONE vector call at N = 1...5 rows over the full kL. The cost
//              of a merged M-row call is read off this scaling curve, because
//              a merged call is one vector kernel over the same KV.
//   split      the shipped pair, 5 rows over kSplit + (M-5) rows over kL,
//              concatenated -- what the scored path runs today
//   fallback   ONE call at M rows over kL -- the composed path the guard
//              exists to avoid, reported for reference only
//
// `split - extrapolated single_qM` is an UPPER BOUND on the saving any merged
// kernel could return. An upper bound is the right statistic for a kill
// decision: if the bound sits under the detection floor, no implementation of
// the idea can clear it.
//
// A tiny-KV control of the same structure runs first, so host encode cost can
// be subtracted. A serial probe does not overlap encode with execution; the
// scored path does.
//
// EXACTNESS EVIDENCE, live rather than argued. The probe reports the max
// absolute deviation between `split` and `fallback` at the touched cells. A
// nonzero deviation is direct evidence that a single merged call is not
// bit-exact on this base. Two controls keep that comparison honest: `split`
// against a recomputation of itself must be exactly 0, and `split` against a
// perturbed input must be nonzero, so the comparison is shown able to fail.
//
// Research instrument. Off unless `MLXFAST_RUN_E149_PROBE=1`. `Tests/` is
// never packaged into a submission. Within-session relative measurement:
// `cool_gate_passed_real_gate=false` and `gate_qualified_for_timing=false` are
// recorded in the output and this probe never produces a score.

private struct E149Cell: Codable {
    var kL: Int
    var arm: String
    var microseconds: Double
    var forwardMicroseconds: Double
    var reverseMicroseconds: Double
    var replicates: Int
}

struct E149SdpaSplitCostTests {
    static let enabled =
        ProcessInfo.processInfo.environment["MLXFAST_RUN_E149_PROBE"] == "1"

    // The scored full-attention shape: 24 query heads over 4 KV heads at head
    // dim 256, bf16, B == 1. `Qwen35Config` pins 64 layers of which 16 are
    // `full_attention`.
    static let queryHeads = 24
    static let kvHeads = 4
    static let headDim = 256
    static let fullAttentionLayers = 16
    static let splitRow = 5

    /// A 512-token seed plus 512 decoded tokens walks kL from 513 to exactly
    /// 1024. E57 measured `kL >= 1025` at zero occurrences in every leg, and
    /// measured only 1 round of 76 reaching 1024, at width 4. These three
    /// points bracket the reachable walk; 769 is its mean.
    static var keyLengths: [Int] {
        guard let raw = ProcessInfo.processInfo.environment["MLXFAST_E149_KL"],
              !raw.isEmpty
        else { return [513, 769, 1024] }
        return raw.split(separator: ",").compactMap { Int($0) }
    }

    static var widths: [Int] {
        guard let raw = ProcessInfo.processInfo.environment["MLXFAST_E149_WIDTHS"],
              !raw.isEmpty
        else { return [6, 7, 8, 9] }
        return raw.split(separator: ",").compactMap { Int($0) }
    }

    static var rampSeconds: Double {
        Double(ProcessInfo.processInfo.environment["MLXFAST_E149_RAMP_S"] ?? "")
            ?? 0.30
    }

    static var targetMicroseconds: Double {
        Double(ProcessInfo.processInfo.environment["MLXFAST_E149_TARGET_US"] ?? "")
            ?? 60_000
    }

    private static func bf16(_ shape: [Int], _ seed: UInt64) -> MLXArray {
        MLXRandom.normal(shape, key: MLXRandom.key(seed)).asType(.bfloat16)
    }

    private static func timed(_ count: Int, _ body: () -> MLXArray) -> Double {
        let start = DispatchTime.now().uptimeNanoseconds
        for _ in 0 ..< count { eval(body()) }
        return Double(DispatchTime.now().uptimeNanoseconds - start) / 1e3
            / Double(count)
    }

    /// HARNESS DEFECT 16: a cold GPU pays a fixed 30-80 ms DVFS ramp that a
    /// palindrome cannot cancel, so burn a fixed wall-clock duration first and
    /// discard it.
    private static func ramp(_ body: () -> MLXArray, seconds: Double) {
        let start = DispatchTime.now().uptimeNanoseconds
        while Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9 < seconds {
            eval(body())
        }
    }

    private static func maxAbsDeviation(_ a: MLXArray, _ b: MLXArray) -> Double {
        let d = MLX.abs(a.asType(.float32) - b.asType(.float32)).max()
        eval(d)
        return Double(d.item(Float.self))
    }

    @Test(
        "E149 C1: isolate the SDPA query split cost at the scored shapes",
        .enabled(if: E149SdpaSplitCostTests.enabled))
    func splitCost() throws {
        let scale = 1.0 / Float(Double(Self.headDim).squareRoot())
        var cells: [E149Cell] = []
        var exactness: [[String: Double]] = []
        var controlCost: [String: Double] = [:]

        for kL in [8] + Self.keyLengths {
            let isControl = kL == 8
            let keys = Self.bf16([1, Self.kvHeads, kL, Self.headDim], 0x5149_0002)
            let values = Self.bf16([1, Self.kvHeads, kL, Self.headDim], 0x5149_0003)
            eval(keys, values)

            var arms: [(String, () -> MLXArray)] = []

            // The merged-form cost curve: one vector call at 1...5 rows over
            // the same KV. A merged M-row call is one such kernel, so its cost
            // is read off this curve instead of being guessed.
            for rows in 1 ... Self.splitRow where rows <= kL {
                let q = Self.bf16(
                    [1, Self.queryHeads, rows, Self.headDim], 0x5149_0001)
                eval(q)
                arms.append(
                    ("single_q\(rows)",
                     {
                         MLXFast.scaledDotProductAttention(
                             queries: q, keys: keys, values: values,
                             scale: scale, mask: .causal)
                     }))
            }

            for width in Self.widths where width <= kL {
                let kSplit = kL - (width - Self.splitRow)
                let q = Self.bf16(
                    [1, Self.queryHeads, width, Self.headDim], 0x5149_0001)
                eval(q)
                arms.append(
                    ("split_m\(width)",
                     {
                         let outA = MLXFast.scaledDotProductAttention(
                             queries: q[0..., 0..., 0 ..< Self.splitRow, 0...],
                             keys: keys[0..., 0..., 0 ..< kSplit, 0...],
                             values: values[0..., 0..., 0 ..< kSplit, 0...],
                             scale: scale, mask: .causal)
                         let outB = MLXFast.scaledDotProductAttention(
                             queries: q[0..., 0..., Self.splitRow..., 0...],
                             keys: keys, values: values, scale: scale,
                             mask: .causal)
                         return concatenated([outA, outB], axis: 2)
                     }))
                arms.append(
                    ("fallback_m\(width)",
                     {
                         MLXFast.scaledDotProductAttention(
                             queries: q, keys: keys, values: values,
                             scale: scale, mask: .causal)
                     }))
            }

            for (_, body) in arms { Self.ramp(body, seconds: Self.rampSeconds) }
            let probe = Self.timed(8, arms[arms.count - 1].1)
            let count = max(
                8, min(4000, Int(Self.targetMicroseconds / max(probe, 1))))

            var forward: [String: Double] = [:]
            for (name, body) in arms { forward[name] = Self.timed(count, body) }
            var reverse: [String: Double] = [:]
            for (name, body) in arms.reversed() {
                reverse[name] = Self.timed(count, body)
            }
            for (name, _) in arms {
                let mean = (forward[name]! + reverse[name]!) / 2
                if isControl {
                    controlCost[name] = mean
                } else {
                    cells.append(
                        E149Cell(
                            kL: kL, arm: name, microseconds: mean,
                            forwardMicroseconds: forward[name]!,
                            reverseMicroseconds: reverse[name]!,
                            replicates: count))
                }
            }

            if isControl { continue }

            for width in Self.widths where width <= kL {
                let kSplit = kL - (width - Self.splitRow)
                let q = Self.bf16(
                    [1, Self.queryHeads, width, Self.headDim], 0x5149_0001)
                func splitOnce() -> MLXArray {
                    concatenated(
                        [
                            MLXFast.scaledDotProductAttention(
                                queries: q[0..., 0..., 0 ..< Self.splitRow, 0...],
                                keys: keys[0..., 0..., 0 ..< kSplit, 0...],
                                values: values[0..., 0..., 0 ..< kSplit, 0...],
                                scale: scale, mask: .causal),
                            MLXFast.scaledDotProductAttention(
                                queries: q[0..., 0..., Self.splitRow..., 0...],
                                keys: keys, values: values, scale: scale,
                                mask: .causal),
                        ], axis: 2)
                }
                let splitOut = splitOnce()
                let splitAgain = splitOnce()
                let fallbackOut = MLXFast.scaledDotProductAttention(
                    queries: q, keys: keys, values: values, scale: scale,
                    mask: .causal)
                let perturbed = MLXFast.scaledDotProductAttention(
                    queries: q, keys: keys,
                    values: (values + MLXArray(Float(1e-2))).asType(.bfloat16),
                    scale: scale, mask: .causal)
                eval(splitOut, splitAgain, fallbackOut, perturbed)

                let selfDelta = Self.maxAbsDeviation(splitOut, splitAgain)
                let fallbackDelta = Self.maxAbsDeviation(splitOut, fallbackOut)
                let controlDelta = Self.maxAbsDeviation(fallbackOut, perturbed)

                // The comparison must be shown able to fail before a zero from
                // it means anything.
                #expect(selfDelta == 0)
                #expect(controlDelta > 0)
                exactness.append([
                    "kL": Double(kL), "width": Double(width),
                    "split_vs_split_max_abs": selfDelta,
                    "split_vs_fallback_max_abs": fallbackDelta,
                    "positive_control_max_abs": controlDelta,
                ])
            }
        }

        func cell(_ kL: Int, _ arm: String) -> Double? {
            cells.first { $0.kL == kL && $0.arm == arm }?.microseconds
        }

        var perRound: [[String: Double]] = []
        for kL in Self.keyLengths {
            // A least-squares line through single_q1...single_q5 gives the cost
            // of one vector call as a function of query rows at this kL.
            let points = (1 ... Self.splitRow).compactMap {
                (r) -> (Double, Double)? in
                guard let y = cell(kL, "single_q\(r)") else { return nil }
                return (Double(r), y)
            }
            guard points.count >= 2 else { continue }
            let n = Double(points.count)
            let mx = points.map { $0.0 }.reduce(0, +) / n
            let my = points.map { $0.1 }.reduce(0, +) / n
            let sxy = points.map { ($0.0 - mx) * ($0.1 - my) }.reduce(0, +)
            let sxx = points.map { ($0.0 - mx) * ($0.0 - mx) }.reduce(0, +)
            let slope = sxx == 0 ? 0 : sxy / sxx
            let intercept = my - slope * mx

            for width in Self.widths {
                guard let shipped = cell(kL, "split_m\(width)") else { continue }
                let merged = intercept + slope * Double(width)
                let hostDelta =
                    (controlCost["split_m\(width)"] ?? 0)
                    - (controlCost["single_q\(Self.splitRow)"] ?? 0)
                let perLayer = shipped - merged
                perRound.append([
                    "kL": Double(kL),
                    "width": Double(width),
                    "split_us_per_layer": shipped,
                    "merged_forecast_us_per_layer": merged,
                    "single_q5_us_per_layer":
                        cell(kL, "single_q\(Self.splitRow)") ?? 0,
                    "fallback_us_per_layer": cell(kL, "fallback_m\(width)") ?? 0,
                    "vector_slope_us_per_query_row": slope,
                    "vector_intercept_us": intercept,
                    "upper_bound_saving_us_per_layer": perLayer,
                    "host_structure_us_per_layer": hostDelta,
                    "gpu_attributable_us_per_layer": perLayer - hostDelta,
                    "upper_bound_saving_us_per_round":
                        perLayer * Double(Self.fullAttentionLayers),
                    "gpu_attributable_us_per_round":
                        (perLayer - hostDelta) * Double(Self.fullAttentionLayers),
                ])
            }
        }

        let summary: [String: Any] = [
            "probe": "e149_c1_sdpa_split_cost",
            "harness": "local",
            "cool_gate_passed_real_gate": false,
            "gate_qualified_for_timing": false,
            "query_heads": Self.queryHeads,
            "kv_heads": Self.kvHeads,
            "gqa_factor": Self.queryHeads / Self.kvHeads,
            "head_dim": Self.headDim,
            "full_attention_layers": Self.fullAttentionLayers,
            "split_row": Self.splitRow,
            "control_host_microseconds": controlCost,
            "exactness": exactness,
            "per_round": perRound,
            "cells": cells.map {
                [
                    "kL": $0.kL, "arm": $0.arm, "microseconds": $0.microseconds,
                    "forward_microseconds": $0.forwardMicroseconds,
                    "reverse_microseconds": $0.reverseMicroseconds,
                    "replicates": $0.replicates,
                ] as [String: Any]
            },
        ]

        let text = String(
            data: try JSONSerialization.data(
                withJSONObject: summary, options: [.prettyPrinted, .sortedKeys]),
            encoding: .utf8)!
        print("E149_C1_JSON_BEGIN")
        print(text)
        print("E149_C1_JSON_END")
        if let path = ProcessInfo.processInfo.environment["MLXFAST_E149_OUT"],
           !path.isEmpty
        {
            try text.write(toFile: path, atomically: true, encoding: .utf8)
        }
    }
}
