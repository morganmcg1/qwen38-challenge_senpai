import Foundation
import MLX
import MLXLLM
import Testing

// E163: how much of one verify round is the ROUTED QUANTIZED MATVEC?
//
// The advisor can price a per-verified-row change in published points, but the
// chain has one unmeasured factor: the share of a per-width round cost that the
// routed matvec occupies. Every future kernel experiment at every width is
// priced through that constant, so it is worth more than any single arm.
//
// The campaign's own instrument for it, the E58/E80 dispatch census that
// `research/e116_qmv_share.py` reduces, is VOID on this base. Neither
// `MLX_E58_DISPATCH_CENSUS` nor `MLX_E80_GPU_TIME` exists in the current tree
// or in the built worker, so no per-kernel exclusive GPU time can be read from
// a leg. This suite measures the numerator directly instead.
//
// WHAT IT MEASURES. `Qwen35CustomQMV.matmul` at the exact scored shapes and
// widths, through the same routed entry point the scored worker takes and the
// same live width plan, with `inner` calls chained into one graph before a
// single `eval`. A verify round builds its whole graph and evaluates once, so
// chained-then-evaluated is closer to the scored dispatch pattern than a
// per-call synchronise would be.
//
// WHAT IT DOES NOT MEASURE. This is an ISOLATED number. No other kernel of the
// round is resident, nothing overlaps it, and the weights are synthetic. The
// campaign's isolated-to-in-situ factor is 0.7858. The reducer applies it and
// labels both forms. An isolated microbenchmark is never a leg and never a
// score.
//
// Enable with `MLXFAST_RUN_E163_QMV_TIMING=1` and point
// `MLXFAST_E163_TIMING_OUT` at the JSON destination. The arm comes from
// `MLX_E163_IPG_PLAN`, which the routed entry point reads once at process
// start, so one process measures one arm.

private struct E163TimingShape {
    let name: String
    let k: Int
    let n: Int
    let callsPerVerify: Int
}

/// The same routing census the exactness suite uses: the affine 4-bit
/// group-64 transposed projections one target verify pass reaches, with their
/// per-pass dispatch counts.
private let e163TimingShapes: [E163TimingShape] = [
    .init(name: "linear_attn.in_proj_fused_qkvzba", k: 5120, n: 16480, callsPerVerify: 48),
    .init(name: "linear_attn.out_proj", k: 6144, n: 5120, callsPerVerify: 48),
    .init(name: "full_attn.qkv_proj_fused", k: 5120, n: 14336, callsPerVerify: 16),
    .init(name: "full_attn.o_proj", k: 6144, n: 5120, callsPerVerify: 16),
    .init(name: "mlp.gate_up_fused", k: 5120, n: 34816, callsPerVerify: 64),
    .init(name: "mlp.down", k: 17408, n: 5120, callsPerVerify: 64),
    .init(name: "head.lm_head", k: 5120, n: 248320, callsPerVerify: 1),
]

private struct E163TimingWeight {
    let w: MLXArray
    let scales: MLXArray
    let biases: MLXArray
}

private func e163TimingWeight(k: Int, n: Int) -> E163TimingWeight {
    let words = k / 8
    let tile = (0..<words).map { index -> UInt32 in
        UInt32(truncatingIfNeeded: index &* 2_654_435_761) ^ 0x9E37_79B9
    }
    let w = MLXArray(tile).reshaped([1, words])
        + arange(0, n, dtype: .uint32).reshaped([n, 1])

    let groups = k / 64
    let rowJitter = arange(0, n, dtype: .float32).reshaped([n, 1]) * 1.7e-6
    let scaleTile: [Float] = (0..<groups).map { index in
        0.00713 + 0.00391 * Float((index &* 37) % 61) / 61.0
    }
    let biasTile: [Float] = (0..<groups).map { index in
        -0.0517 - 0.0233 * Float((index &* 23) % 53) / 53.0
    }
    let weight = E163TimingWeight(
        w: w,
        scales: (MLXArray(scaleTile).reshaped([1, groups]) + rowJitter).asType(.bfloat16),
        biases: (MLXArray(biasTile).reshaped([1, groups]) + rowJitter).asType(.bfloat16))
    eval(weight.w, weight.scales, weight.biases)
    return weight
}

private func e163TimingActivations(m: Int, k: Int) -> MLXArray {
    var values = [Float](repeating: 0, count: m * k)
    for row in 0..<m {
        for col in 0..<k {
            let hashed = (row &* 7_919 &+ col &* 104_729 &+ 15_485_863) % 1_000_003
            values[row * k + col] = (Float(hashed) / 1_000_003.0 - 0.5) * 1.7320508
        }
    }
    let x = MLXArray(values, [m, k]).asType(.bfloat16)
    eval(x)
    return x
}

@Suite
struct E163QMVIsolatedTimingTests {
    private static var enabled: Bool {
        ProcessInfo.processInfo.environment["MLXFAST_RUN_E163_QMV_TIMING"] == "1"
    }

    @Test(.enabled(if: E163QMVIsolatedTimingTests.enabled))
    func routedMatvecCostPerVerifyRound() throws {
        let env = ProcessInfo.processInfo.environment
        let outPath = try #require(
            env["MLXFAST_E163_TIMING_OUT"],
            "MLXFAST_E163_TIMING_OUT must name the JSON destination")
        let arm = env["MLX_E163_IPG_PLAN"] ?? "shipped"
        let plan = Qwen35CustomQMV.inputsPerGroupPlan
        let reps = Int(env["MLXFAST_E163_TIMING_REPS"] ?? "") ?? 12
        let inner = Int(env["MLXFAST_E163_TIMING_INNER"] ?? "") ?? 8
        let widths =
            (env["MLXFAST_E163_TIMING_WIDTHS"]?
                .split(separator: ",")
                .compactMap { Int($0.trimmingCharacters(in: .whitespaces)) })
            .flatMap { $0.isEmpty ? nil : $0 } ?? [5, 6]

        var cells: [[String: Any]] = []
        for shape in e163TimingShapes {
            let weight = e163TimingWeight(k: shape.k, n: shape.n)
            for m in widths {
                let x = e163TimingActivations(m: m, k: shape.k)

                let warm = try #require(
                    Qwen35CustomQMV.matmul(
                        x, weight.w, scales: weight.scales, biases: weight.biases,
                        groupSize: 64, bits: 4, mode: .affine),
                    Comment(
                        rawValue: "the candidate route declined \(shape.name) at M=\(m), "
                            + "so this cell is not on the routed path at all"))
                eval(warm)

                var perCallUs: [Double] = []
                for _ in 0..<reps {
                    var outputs: [MLXArray] = []
                    outputs.reserveCapacity(inner)
                    let start = DispatchTime.now().uptimeNanoseconds
                    for _ in 0..<inner {
                        outputs.append(
                            Qwen35CustomQMV.matmul(
                                x, weight.w, scales: weight.scales, biases: weight.biases,
                                groupSize: 64, bits: 4, mode: .affine)!)
                    }
                    eval(outputs)
                    let elapsed = DispatchTime.now().uptimeNanoseconds - start
                    perCallUs.append(Double(elapsed) / 1e3 / Double(inner))
                }
                let sorted = perCallUs.sorted()
                let ipg = try #require(
                    plan[m],
                    Comment(rawValue: "the live plan has no entry for width \(m)"))
                // Affine 4-bit group-64: packed nibbles, plus one bf16 scale
                // and one bf16 bias per group of 64 input elements.
                let packedBytes = shape.n * shape.k / 2
                let metaBytes = 2 * shape.n * (shape.k / 64) * 2
                let activationBytes = m * shape.k * 2
                let outputBytes = m * shape.n * 2
                let bytesPerCall =
                    packedBytes + metaBytes + activationBytes + outputBytes
                cells.append([
                    "shape": shape.name,
                    "k": shape.k,
                    "n": shape.n,
                    "m": m,
                    "calls_per_verify": shape.callsPerVerify,
                    "inputs_per_group": ipg,
                    "accumulator_lanes_na": ipg,
                    "active_groups": (m + ipg - 1) / ipg,
                    "weight_packed_bytes": packedBytes,
                    "weight_scale_bias_bytes": metaBytes,
                    "activation_bytes": activationBytes,
                    "output_bytes": outputBytes,
                    "bytes_per_call": bytesPerCall,
                    "per_call_us_min": sorted.first!,
                    "per_call_us_median": sorted[sorted.count / 2],
                    "per_call_us_max": sorted.last!,
                    "per_call_us_all": perCallUs,
                ])
            }
        }

        let payload: [String: Any] = [
            "arm": arm,
            "harness": "local",
            "measurement": "isolated routed Qwen35CustomQMV.matmul, no other kernel resident",
            "timing_valid_as_leg": false,
            "official_or_ranked_score": false,
            "reps": reps,
            "inner_calls_per_rep": inner,
            "widths": widths,
            "plan": plan.keys.sorted().map { ["m": $0, "inputs_per_group": plan[$0]!] },
            "cells": cells,
        ]
        let data = try JSONSerialization.data(
            withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: URL(fileURLWithPath: outPath))
    }
}
