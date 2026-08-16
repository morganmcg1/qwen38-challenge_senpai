import Foundation
import MLX
import Testing

/// Cost curve of `quantized_matmul` at the exact scored Qwen 3.8 27B shapes,
/// swept over verify width M, through the VENDORED MLX the scored worker links.
///
/// This lives in the test target on purpose. Yukon never submits tests, so the
/// authoritative kernels can be timed without touching `Package.swift` or the
/// submitted surface. `research/qmv_cost_curve.py` runs the same sweep against
/// stock pip MLX 0.32.0; the two curves diverge wherever this checkout carries
/// a kernel the public release does not.
///
/// Enable with `MLXFAST_RUN_QMV_COST_CURVE=1` and point
/// `MLXFAST_QMV_COST_CURVE_OUT` at the JSON destination.
@Suite
struct QwenQMVCostCurveTests {
    private static var enabled: Bool {
        ProcessInfo.processInfo
            .environment["MLXFAST_RUN_QMV_COST_CURVE"] == "1"
    }

    @Test(.enabled(if: QwenQMVCostCurveTests.enabled))
    func sweepQuantizedMatmulOverVerifyWidth() throws {
        let env = ProcessInfo.processInfo.environment
        let outPath = try #require(
            env["MLXFAST_QMV_COST_CURVE_OUT"],
            "MLXFAST_QMV_COST_CURVE_OUT must name the JSON destination")
        let reps = Int(env["MLXFAST_QMV_COST_CURVE_REPS"] ?? "") ?? 15
        let inner = Int(env["MLXFAST_QMV_COST_CURVE_INNER"] ?? "") ?? 10

        var payload: [String: Any] = [
            "source": "vendored-mlx-swift",
            "reps": reps,
            "inner_calls_per_rep": inner,
            "widths": scoredWidths,
        ]
        payload["roofline"] = measureRoofline(reps: reps)
        payload["shapes"] = scoredShapes.map { shape in
            sweep(shape: shape, widths: scoredWidths, reps: reps, inner: inner)
        }
        payload["dispatch_boundary_probes"] = dispatchBoundaryProbes.map { probe in
            sweep(
                shape: probe.shape, widths: probe.widths, reps: reps, inner: inner,
                extra: ["predicted_vector_limit": probe.predictedVectorLimit])
        }
        payload["fast_path_probes"] = fastPathProbes.map { shape in
            sweep(shape: shape, widths: [1, 4, 8, 9], reps: reps, inner: inner)
        }

        let data = try JSONSerialization.data(
            withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: URL(fileURLWithPath: outPath))
        print("QMV_COST_CURVE_OUT \(outPath)")
    }
}

// MARK: - what the scored verify actually calls

/// One affine 4-bit group-64 transposed projection the target runs per verify,
/// with the number of times a single verify pass reaches it.
private struct ScoredShape {
    let name: String
    let k: Int
    let n: Int
    let callsPerVerify: Int
}

/// Shapes read off the live target: `Qwen35.swift` (48 Gated DeltaNet layers,
/// 16 full-attention layers, fused gate/up MLP) plus the unconditional full
/// vocabulary readout. `out_proj`/`o_proj` share a shape and are listed
/// separately so the call mix stays legible.
/// `M = 1...12` brackets `vector_limit`; the decade points anchor `FLOPS_eff`
/// empirically and expose the bandwidth/compute knee instead of borrowing it
/// from the prefill measurement.
let scoredWidths: [Int] = Array(1...12) + [16, 32, 64, 128, 256, 512]

private let scoredShapes: [ScoredShape] = [
    .init(name: "linear_attn.in_proj_fused_qkvzba", k: 5120, n: 16480, callsPerVerify: 48),
    .init(name: "linear_attn.out_proj", k: 6144, n: 5120, callsPerVerify: 48),
    .init(name: "full_attn.qkv_proj_fused", k: 5120, n: 14336, callsPerVerify: 16),
    .init(name: "full_attn.o_proj", k: 6144, n: 5120, callsPerVerify: 16),
    .init(name: "mlp.gate_up_fused", k: 5120, n: 34816, callsPerVerify: 64),
    .init(name: "mlp.down", k: 17408, n: 5120, callsPerVerify: 64),
    .init(name: "head.lm_head", k: 5120, n: 248320, callsPerVerify: 1),
    // Draft-side, not part of the verify mix: the padded compact draft
    // vocabulary (`Qwen35.swift:2061-2066`, `applyDraftLMHead:2336`) runs once
    // per draft step at M=1. Carried here to settle whether its cost is the
    // ~0.6ms in the advisor's notes or the ~1.25ms its 283MB implies.
    .init(name: "head.compact_draft_vocab", k: 5120, n: 98336, callsPerVerify: 0),
]

/// `get_qmv_batch_limit` claims three different limits by (K, N) size class on
/// this architecture. Sweeping past each predicted limit turns that read of the
/// host dispatcher into a measurement.
private struct BoundaryProbe {
    let shape: ScoredShape
    let widths: [Int]
    let predictedVectorLimit: Int
}

private let dispatchBoundaryProbes: [BoundaryProbe] = [
    .init(
        shape: .init(name: "probe.k2048_n2048", k: 2048, n: 2048, callsPerVerify: 0),
        widths: Array(1...22), predictedVectorLimit: 18),
    .init(
        shape: .init(name: "probe.k4096_n4096", k: 4096, n: 4096, callsPerVerify: 0),
        widths: Array(1...16), predictedVectorLimit: 12),
    .init(
        shape: .init(name: "probe.k5120_n5120", k: 5120, n: 5120, callsPerVerify: 0),
        widths: Array(1...14), predictedVectorLimit: 10),
]

/// `qmv` picks its fast variant on `N % 8 == 0 && K % 512 == 0`. 5120 and 5632
/// are multiples of 512; 5184 is not. A step between them at fixed N is the
/// fast-path selection becoming visible.
private let fastPathProbes: [ScoredShape] = [
    .init(name: "fastprobe.k5120_n16480", k: 5120, n: 16480, callsPerVerify: 0),
    .init(name: "fastprobe.k5184_n16480", k: 5184, n: 16480, callsPerVerify: 0),
    .init(name: "fastprobe.k5632_n16480", k: 5632, n: 16480, callsPerVerify: 0),
]

// MARK: - measurement

private func sweep(
    shape: ScoredShape,
    widths: [Int],
    reps: Int,
    inner: Int,
    extra: [String: Any] = [:]
) -> [String: Any] {
    let weight = syntheticAffine4Weight(k: shape.k, n: shape.n)
    var rows: [[String: Any]] = []
    var referenceRow0: [Float]? = nil

    for m in widths {
        // Wide widths are pure GPU work, so batching many of them per timed
        // region buys nothing and costs minutes. Shrink the batch and the
        // repetition count once the per-call cost dwarfs launch overhead.
        let innerForM = m <= 12 ? inner : max(2, (inner * 12) / m)
        let repsForM = m <= 12 ? reps : max(5, reps / 3)
        let xs = (0..<innerForM).map { syntheticActivations(m: m, k: shape.k, salt: $0) }
        eval(xs)

        // The scored verify pass is a strictly dependent chain: every layer's
        // quantized matmul consumes the previous layer's output. MLX only emits
        // a Metal barrier between kernels that actually share a buffer
        // (`CommandEncoder::maybeInsertBarrier`, device.cpp:324-364) and encodes
        // with `MTL::DispatchTypeConcurrent` (device.cpp:548), so independent
        // calls batched into one `eval` overlap and understate small-M cost --
        // exactly the regime this experiment is trying to measure. Threading a
        // 1x1 tap of each output into the next input restores that dependency.
        // The tap is scaled to 1e-30 so it vanishes in bf16 rounding: the graph
        // edge is real, the activations are bitwise unchanged.
        let chained = {
            var outs: [MLXArray] = []
            outs.reserveCapacity(innerForM)
            var x = xs[0]
            for i in 0..<innerForM {
                let o = quantizedMM(
                    x, weight.w, scales: weight.scales, biases: weight.biases,
                    transpose: true, groupSize: 64, bits: 4)
                outs.append(o)
                if i + 1 < innerForM { x = xs[i + 1] + o[0..<1, 0..<1] * 1e-30 }
            }
            return outs
        }
        // Same graph minus the matmuls: isolates the tap scaffolding so the
        // chained number can be reported with its own overhead quantified.
        let tapsOnly = {
            var outs: [MLXArray] = []
            outs.reserveCapacity(innerForM)
            var x = xs[0]
            for i in 0..<innerForM {
                outs.append(x)
                if i + 1 < innerForM { x = xs[i + 1] + x[0..<1, 0..<1] * 1e-30 }
            }
            return outs
        }
        let concurrent = {
            xs.map {
                quantizedMM(
                    $0, weight.w, scales: weight.scales, biases: weight.biases,
                    transpose: true, groupSize: 64, bits: 4)
            }
        }

        let samples = medianSpread(reps: repsForM, inner: innerForM, warmup: 3, body: chained)
        let conc = medianSpread(reps: repsForM, inner: innerForM, warmup: 3, body: concurrent)
        let taps = medianSpread(reps: repsForM, inner: innerForM, warmup: 3, body: tapsOnly)

        let row0 = quantizedMM(
            xs[0], weight.w, scales: weight.scales, biases: weight.biases,
            transpose: true, groupSize: 64, bits: 4)[0]
            .asType(.float32).asArray(Float.self)
        if referenceRow0 == nil { referenceRow0 = row0 }

        rows.append([
            "m": m,
            "seconds_per_call": samples[samples.count / 2],
            "seconds_per_call_min": samples[0],
            "seconds_per_call_max": samples[samples.count - 1],
            "seconds_per_call_concurrent": conc[conc.count / 2],
            "tap_overhead_seconds_per_call": taps[taps.count / 2],
            "calls_per_timed_region": innerForM,
            "timed_regions": repsForM,
            "row0_bitwise_matches_m1": row0 == referenceRow0!,
            "row0_max_abs_delta_vs_m1": maxAbsDelta(row0, referenceRow0!),
        ])
    }

    var out: [String: Any] = [
        "name": shape.name,
        "k": shape.k,
        "n": shape.n,
        "calls_per_verify": shape.callsPerVerify,
        "weight_bytes": affine4WeightBytes(k: shape.k, n: shape.n),
        "flops_per_row": 2 * shape.k * shape.n,
        "rows": rows,
    ]
    out.merge(extra) { a, _ in a }
    return out
}

/// Affine 4-bit group-64: packed nibbles plus a bf16 scale and bias per group.
private func affine4WeightBytes(k: Int, n: Int) -> Int {
    n * (k / 2) + 2 * (n * (k / 64) * 2)
}

private func maxAbsDelta(_ a: [Float], _ b: [Float]) -> Double {
    guard a.count == b.count else { return .infinity }
    var worst = Float(0)
    for i in a.indices { worst = max(worst, abs(a[i] - b[i])) }
    return Double(worst)
}

private struct Affine4Weight {
    let w: MLXArray
    let scales: MLXArray
    let biases: MLXArray
}

/// Builds a contiguous quantized weight without materializing a dense source.
/// A per-row broadcast add both varies the data across rows and forces the
/// broadcast tile into real row-contiguous storage, so `quantized_matmul` never
/// pays a hidden contiguity copy inside the timed region.
private func syntheticAffine4Weight(k: Int, n: Int) -> Affine4Weight {
    let words = k / 8
    let tile = (0..<words).map { index -> UInt32 in
        UInt32(truncatingIfNeeded: index &* 2_654_435_761) ^ 0x9E37_79B9
    }
    let w = MLXArray(tile).reshaped([1, words]) + arange(0, n, dtype: .uint32).reshaped([n, 1])

    let groups = k / 64
    let rowJitter = arange(0, n, dtype: .float32).reshaped([n, 1]) * 1e-6
    let scaleTile: [Float] = (0..<groups).map { index -> Float in
        let step = Float((index &* 37) % 61)
        return 0.006 + 0.004 * step / 61.0
    }
    let biasTile: [Float] = (0..<groups).map { index -> Float in
        let step = Float((index &* 23) % 53)
        return -0.05 - 0.02 * step / 53.0
    }
    let scales = (MLXArray(scaleTile).reshaped([1, groups]) + rowJitter).asType(.bfloat16)
    let biases = (MLXArray(biasTile).reshaped([1, groups]) + rowJitter).asType(.bfloat16)

    let weight = Affine4Weight(w: w, scales: scales, biases: biases)
    eval(weight.w, weight.scales, weight.biases)
    return weight
}

private func syntheticActivations(m: Int, k: Int, salt: Int) -> MLXArray {
    let tile: [Float] = (0..<k).map { index -> Float in
        Float((index &* 131 &+ salt &* 7919) % 251) / 251.0 - 0.5
    }
    let rowJitter = arange(0, m, dtype: .float32).reshaped([m, 1]) * 0.01
    return (MLXArray(tile).reshaped([1, k]) + rowJitter).asType(.bfloat16)
}

/// Roofline constants for `qmv_tax`, measured on this host rather than quoted:
/// a large streaming elementwise op for bandwidth and a dense bf16 GEMM for the
/// arithmetic ceiling.
private func measureRoofline(reps: Int) -> [String: Any] {
    let streamElements = 256 * 1024 * 1024
    let stream = zeros([streamElements], dtype: .bfloat16)
    eval(stream)
    let streamSeconds = median(reps: reps) { eval(stream + Float(1)) }

    let gemmDim = 4096
    let a = zeros([gemmDim, gemmDim], dtype: .bfloat16)
    let b = zeros([gemmDim, gemmDim], dtype: .bfloat16)
    eval(a, b)
    let gemmSeconds = median(reps: reps) { eval(a.matmul(b)) }

    return [
        "stream_bytes": 2 * 2 * streamElements,
        "stream_seconds": streamSeconds,
        "peak_bandwidth_bytes_per_second": Double(2 * 2 * streamElements) / streamSeconds,
        "gemm_flops": 2 * gemmDim * gemmDim * gemmDim,
        "gemm_seconds": gemmSeconds,
        "peak_flops_per_second": Double(2 * gemmDim * gemmDim * gemmDim) / gemmSeconds,
    ]
}

/// Sorted per-call seconds over `reps` timed regions, each of which builds and
/// evaluates `inner` calls. Graph construction is inside the timed region for
/// every closure, so the taps-only closure subtracts it honestly.
private func medianSpread(
    reps: Int,
    inner: Int,
    warmup: Int,
    body: () -> [MLXArray]
) -> [Double] {
    for _ in 0..<warmup { eval(body()) }
    var samples: [Double] = []
    samples.reserveCapacity(reps)
    for _ in 0..<reps {
        let start = DispatchTime.now().uptimeNanoseconds
        eval(body())
        let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
        samples.append(elapsed / Double(inner))
    }
    samples.sort()
    return samples
}

private func median(reps: Int, _ body: () -> Void) -> Double {
    for _ in 0..<3 { body() }
    var samples: [Double] = []
    for _ in 0..<reps {
        let start = DispatchTime.now().uptimeNanoseconds
        body()
        samples.append(Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9)
    }
    samples.sort()
    return samples[samples.count / 2]
}
