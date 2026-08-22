import Foundation
import MLX
import MLXLLM
import Testing

/// E137 item 2, corrected: the isolated QMV width curve on the path the scored
/// worker actually dispatches.
///
/// `QwenQMVCostCurveTests` calls `quantizedMM` directly, which bypasses
/// `Qwen35CustomQMV.matmul` and therefore measures the FALLBACK path. The
/// scored call site is `qwen35RoutedQuantizedMM` (`Qwen35.swift:2252-2270`),
/// which tries Route B first and only falls back when `routable`
/// (`Qwen35.swift:2109-2135`) declines the cell. Every contributing scored
/// shape satisfies `routable`, so the fallback curve cannot price the step.
///
/// This suite sweeps both paths over the same shapes, widths, weights and
/// session so the two curves are directly comparable, and checks that Route B
/// reproduces the fallback bit for bit at every cell it claims.
///
/// Enable with `MLXFAST_RUN_E137_ROUTEB_CURVE=1` and point
/// `MLXFAST_E137_ROUTEB_CURVE_OUT` at the JSON destination.
@Suite
struct E137RouteBCostCurveTests {
    private static var enabled: Bool {
        ProcessInfo.processInfo
            .environment["MLXFAST_RUN_E137_ROUTEB_CURVE"] == "1"
    }

    @Test(.enabled(if: E137RouteBCostCurveTests.enabled))
    func sweepRoutedAndFallbackOverVerifyWidth() throws {
        let env = ProcessInfo.processInfo.environment
        let outPath = try #require(
            env["MLXFAST_E137_ROUTEB_CURVE_OUT"],
            "MLXFAST_E137_ROUTEB_CURVE_OUT must name the JSON destination")
        let reps = Int(env["MLXFAST_E137_ROUTEB_CURVE_REPS"] ?? "") ?? 15
        let inner = Int(env["MLXFAST_E137_ROUTEB_CURVE_INNER"] ?? "") ?? 10
        let widths =
            (env["MLXFAST_E137_ROUTEB_CURVE_WIDTHS"]?
                .split(separator: ",")
                .compactMap { Int($0.trimmingCharacters(in: .whitespaces)) })
            .flatMap { $0.isEmpty ? nil : $0 } ?? Array(3...9)

        var payload: [String: Any] = [
            "experiment": "e137-item2-routeb",
            "source": "vendored-mlx-swift + Qwen35CustomQMV",
            "reps": reps,
            "inner_calls_per_rep": inner,
            "widths": widths,
            "arm": Qwen35CustomQMV.arm.rawValue,
            "entry": Qwen35CustomQMV.entry.rawValue,
            "routed_widths": [
                Qwen35CustomQMV.widths.lowerBound,
                Qwen35CustomQMV.widths.upperBound,
            ],
            "note":
                "routed = Qwen35CustomQMV.matmul, the scored path; fallback = "
                + "quantizedMM, what QwenQMVCostCurveTests measured. Arms are "
                + "ABBA-interleaved inside each timed cell.",
        ]
        payload["shapes"] = e137ScoredShapes.map { shape in
            e137Sweep(shape: shape, widths: widths, reps: reps, inner: inner)
        }

        let data = try JSONSerialization.data(
            withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: URL(fileURLWithPath: outPath))
    }
}

// MARK: - shapes

private struct E137Shape {
    let name: String
    let k: Int
    let n: Int
    let callsPerVerify: Int
}

/// The seven shapes the verify forward actually calls, with the per-verify call
/// count the 64-layer target implies: 48 Gated DeltaNet layers, 16 full
/// attention layers, 64 MLP blocks, one vocabulary readout.
private let e137ScoredShapes: [E137Shape] = [
    .init(name: "linear_attn.in_proj_fused_qkvzba", k: 5120, n: 16480, callsPerVerify: 48),
    .init(name: "linear_attn.out_proj", k: 6144, n: 5120, callsPerVerify: 48),
    .init(name: "full_attn.qkv_proj_fused", k: 5120, n: 14336, callsPerVerify: 16),
    .init(name: "full_attn.o_proj", k: 6144, n: 5120, callsPerVerify: 16),
    .init(name: "mlp.gate_up_fused", k: 5120, n: 34816, callsPerVerify: 64),
    .init(name: "mlp.down", k: 17408, n: 5120, callsPerVerify: 64),
    .init(name: "head.lm_head", k: 5120, n: 248320, callsPerVerify: 1),
]

// MARK: - measurement

private func e137Sweep(
    shape: E137Shape, widths: [Int], reps: Int, inner: Int
) -> [String: Any] {
    let weight = e137QuantWeight(k: shape.k, n: shape.n)
    var rows: [[String: Any]] = []

    for m in widths {
        let xs = (0..<inner).map { e137Activations(m: m, k: shape.k, salt: $0) }
        eval(xs)

        // The verify forward is a dependent chain: each layer consumes the
        // previous layer's output. Independent calls batched into one `eval`
        // overlap and understate cost, so a 1e-30 tap of each output is
        // threaded into the next input. The tap vanishes in bf16 rounding, so
        // the graph edge is real and the activations are bitwise unchanged.
        let routed: () -> [MLXArray] = {
            var outs: [MLXArray] = []
            outs.reserveCapacity(inner)
            var x = xs[0]
            for i in 0..<inner {
                guard
                    let o = Qwen35CustomQMV.matmul(
                        x, weight.w, scales: weight.scales, biases: weight.biases,
                        groupSize: 64, bits: 4, mode: .affine)
                else { return [] }
                outs.append(o)
                if i + 1 < inner { x = xs[i + 1] + o[0..<1, 0..<1] * 1e-30 }
            }
            return outs
        }
        let fallback = {
            var outs: [MLXArray] = []
            outs.reserveCapacity(inner)
            var x = xs[0]
            for i in 0..<inner {
                let o = quantizedMM(
                    x, weight.w, scales: weight.scales, biases: weight.biases,
                    transpose: true, groupSize: 64, bits: 4)
                outs.append(o)
                if i + 1 < inner { x = xs[i + 1] + o[0..<1, 0..<1] * 1e-30 }
            }
            return outs
        }
        let tapsOnly = {
            var outs: [MLXArray] = []
            outs.reserveCapacity(inner)
            var x = xs[0]
            for i in 0..<inner {
                outs.append(x)
                if i + 1 < inner { x = xs[i + 1] + x[0..<1, 0..<1] * 1e-30 }
            }
            return outs
        }

        // A cell Route B declines is the answer to the routing question, not a
        // failure. Record it and move on rather than timing an empty body.
        let claimed = !routed().isEmpty
        var row: [String: Any] = [
            "m": m,
            "routed_claimed": claimed,
            "tier_ipg": Qwen35CustomQMV.widths.contains(m)
                ? Qwen35CustomQMV.tier(m: m) : -1,
        ]
        if claimed, Qwen35CustomQMV.widths.contains(m) {
            let ipg = Qwen35CustomQMV.tier(m: m)
            row["weight_passes"] = (m + ipg - 1) / ipg
        }

        let taps = e137MedianSpread(reps: reps, inner: inner, body: tapsOnly)
        // ABBA inside the timed cell so monotone thermal drift cancels to
        // first order between the two arms.
        var routedBody: (() -> [MLXArray])? = nil
        if claimed { routedBody = routed }
        let (routedSamples, fallbackSamples) = e137ABBA(
            reps: reps, inner: inner, a: routedBody, b: fallback)

        if claimed {
            row["routed_seconds_per_call"] = routedSamples[routedSamples.count / 2]
            row["routed_seconds_per_call_min"] = routedSamples[0]
            row["routed_seconds_per_call_max"] = routedSamples[routedSamples.count - 1]
            // Every replicate is kept so the reader can interval the median and
            // the width step itself. A min-to-max envelope over 15 draws is an
            // extreme-value range, not an uncertainty on the median, and it is
            // far too wide to decide a 30 % boundary.
            row["routed_samples"] = routedSamples
        }
        row["fallback_seconds_per_call"] = fallbackSamples[fallbackSamples.count / 2]
        row["fallback_seconds_per_call_min"] = fallbackSamples[0]
        row["fallback_seconds_per_call_max"] = fallbackSamples[fallbackSamples.count - 1]
        row["fallback_samples"] = fallbackSamples
        row["tap_overhead_seconds_per_call"] = taps[taps.count / 2]
        row["tap_samples"] = taps

        // Route B advertises a bit-exact replica of the incumbent. Checking it
        // here also proves both arms really dispatched their own kernel.
        if claimed {
            let a = Qwen35CustomQMV.matmul(
                xs[0], weight.w, scales: weight.scales, biases: weight.biases,
                groupSize: 64, bits: 4, mode: .affine)!
            let b = quantizedMM(
                xs[0], weight.w, scales: weight.scales, biases: weight.biases,
                transpose: true, groupSize: 64, bits: 4)
            let av = a.asType(.float32).asArray(Float.self)
            let bv = b.asType(.float32).asArray(Float.self)
            row["routed_matches_fallback_bitwise"] = av == bv
            row["routed_vs_fallback_max_abs_delta"] = zip(av, bv)
                .map { Double(abs($0 - $1)) }.max() ?? 0
        }
        rows.append(row)
    }

    return [
        "name": shape.name,
        "k": shape.k,
        "n": shape.n,
        "calls_per_verify": shape.callsPerVerify,
        "rows": rows,
    ]
}

/// Alternates A/B/B/A across replicate pairs so a monotone drift over the timed
/// cell contributes equally to both arms.
private func e137ABBA(
    reps: Int, inner: Int, a: (() -> [MLXArray])?, b: @escaping () -> [MLXArray]
) -> (aSamples: [Double], bSamples: [Double]) {
    for _ in 0..<3 {
        if let a { eval(a()) }
        eval(b())
    }
    var aSamples: [Double] = []
    var bSamples: [Double] = []
    for rep in 0..<reps {
        let aFirst = rep % 2 == 0
        if aFirst {
            if let a { aSamples.append(e137Time(inner: inner, a)) }
            bSamples.append(e137Time(inner: inner, b))
        } else {
            bSamples.append(e137Time(inner: inner, b))
            if let a { aSamples.append(e137Time(inner: inner, a)) }
        }
    }
    aSamples.sort()
    bSamples.sort()
    return (aSamples, bSamples)
}

private func e137Time(inner: Int, _ body: () -> [MLXArray]) -> Double {
    let start = DispatchTime.now().uptimeNanoseconds
    eval(body())
    let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
    return elapsed / Double(inner)
}

private func e137MedianSpread(
    reps: Int, inner: Int, body: () -> [MLXArray]
) -> [Double] {
    for _ in 0..<3 { eval(body()) }
    var samples: [Double] = []
    samples.reserveCapacity(reps)
    for _ in 0..<reps { samples.append(e137Time(inner: inner, body)) }
    samples.sort()
    return samples
}

private struct E137QuantWeight {
    let w: MLXArray
    let scales: MLXArray
    let biases: MLXArray
}

private func e137QuantWeight(k: Int, n: Int) -> E137QuantWeight {
    let words = k / 8
    let tile = (0..<words).map { index -> UInt32 in
        UInt32(truncatingIfNeeded: index &* 2_654_435_761) ^ 0x9E37_79B9
    }
    let w = MLXArray(tile).reshaped([1, words])
        + arange(0, n, dtype: .uint32).reshaped([n, 1])

    let groups = k / 64
    let rowJitter = arange(0, n, dtype: .float32).reshaped([n, 1]) * 1e-6
    let scaleTile: [Float] = (0..<groups).map { index -> Float in
        0.006 + 0.004 * Float((index &* 37) % 61) / 61.0
    }
    let biasTile: [Float] = (0..<groups).map { index -> Float in
        -0.05 - 0.02 * Float((index &* 23) % 53) / 53.0
    }
    let scales = (MLXArray(scaleTile).reshaped([1, groups]) + rowJitter).asType(.bfloat16)
    let biases = (MLXArray(biasTile).reshaped([1, groups]) + rowJitter).asType(.bfloat16)

    let weight = E137QuantWeight(w: w, scales: scales, biases: biases)
    eval(weight.w, weight.scales, weight.biases)
    return weight
}

private func e137Activations(m: Int, k: Int, salt: Int) -> MLXArray {
    let tile: [Float] = (0..<k).map { index -> Float in
        Float((index &* 131 &+ salt &* 7919) % 251) / 251.0 - 0.5
    }
    let rowJitter = arange(0, m, dtype: .float32).reshaped([m, 1]) * 0.01
    return (MLXArray(tile).reshaped([1, k]) + rowJitter).asType(.bfloat16)
}
