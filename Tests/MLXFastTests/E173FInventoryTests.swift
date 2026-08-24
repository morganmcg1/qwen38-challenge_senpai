import Foundation
import MLX
import MLXFast
@testable import MLXFastModel
@testable import MLXLLM
import MLXNN
import Testing

// E173 -- what is inside `F`, the M-independent per-round fixed cost.
//
// TWO INSTRUMENTS, both bottom-up, both `harness=local`.
//
// 1. `E173AdmissionCensusTests` prices the ADMISSION PATH of the candidate's
//    wide-QMV router on the HOST ONLY. Nothing is evaluated: every weight is a
//    lazily declared `zeros` of the scored shape, so `routable` reads dtype,
//    rank, shape and strides exactly as it does in situ while no buffer is ever
//    allocated and no kernel ever runs. The suite therefore needs no thermal
//    gate and holds no model: it measures host instructions, not GPU time.
//
// 2. `E173GPULineItemTests` prices the per-round GPU work that is NOT the
//    quantized weight pass, at the checkpoint's real geometry: the 48 gated
//    delta recurrences, the 16 full-attention SDPA calls at a decode-length
//    cache, the small-operator dispatch floor, the proposal head's BF16
//    precision-island GEMMs, and the vocabulary top-two readout. These are the
//    named GPU line items of `F`.
//
// NOTHING HERE IS ON THE SUBMITTED SURFACE. Yukon never packages `Tests/`.

// MARK: - shared helpers

/// The seven wide-QMV shapes that make up all 257 routed cells of one decode
/// round. Counts are the target's own layer geometry: 48 GDN layers x 2, 16
/// full-attention layers x 2, 64 MLP blocks x 2, and one `lm_head`.
private struct E173Shape {
    let name: String
    let k: Int
    let n: Int
    let callsPerRound: Int
}

private let e173ScoredShapes: [E173Shape] = [
    .init(name: "gdn.in_proj", k: 5120, n: 16480, callsPerRound: 48),
    .init(name: "gdn.out_proj", k: 6144, n: 5120, callsPerRound: 48),
    .init(name: "fa.qkv", k: 5120, n: 14336, callsPerRound: 16),
    .init(name: "fa.o_proj", k: 6144, n: 5120, callsPerRound: 16),
    .init(name: "mlp.gate_up", k: 5120, n: 34816, callsPerRound: 64),
    .init(name: "mlp.down", k: 17408, n: 5120, callsPerRound: 64),
    .init(name: "lm_head", k: 5120, n: 248_320, callsPerRound: 1),
]

/// Median and spread of `samples`, in seconds.
private func e173Stats(_ samples: [Double]) -> (median: Double, min: Double, max: Double) {
    let sorted = samples.sorted()
    return (sorted[sorted.count / 2], sorted[0], sorted[sorted.count - 1])
}

private func e173Device() -> [String: Any] {
    let device = MLX.GPU.deviceInfo()
    return [
        "architecture": device.architecture,
        "memory_size": device.memorySize,
        "max_recommended_working_set_size": Int(device.maxRecommendedWorkingSetSize),
    ]
}

private func e173GPUTemperature() -> Double? {
    let binary =
        ProcessInfo.processInfo.environment["MLXFAST_E173_MACMON"] ?? "/opt/homebrew/bin/macmon"
    guard FileManager.default.isExecutableFile(atPath: binary) else { return nil }
    let process = Process()
    process.executableURL = URL(fileURLWithPath: binary)
    process.arguments = ["pipe", "-s1"]
    let pipe = Pipe()
    process.standardOutput = pipe
    process.standardError = FileHandle.nullDevice
    do { try process.run() } catch { return nil }
    let data = pipe.fileHandleForReading.readDataToEndOfFile()
    process.waitUntilExit()
    for line in String(decoding: data, as: UTF8.self).split(separator: "\n") {
        guard let object = try? JSONSerialization.jsonObject(with: Data(line.utf8)),
            let root = object as? [String: Any],
            let temp = root["temp"] as? [String: Any],
            let gpu = temp["gpu_temp_avg"] as? Double
        else { continue }
        return gpu
    }
    return nil
}

private func e173Write(_ payload: [String: Any], to path: String) throws {
    let data = try JSONSerialization.data(
        withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
    try data.write(to: URL(fileURLWithPath: path))
}

// MARK: - 1. host-side admission census

/// A cell's tensors, declared at the scored shape and never evaluated.
private struct E173LazyCell {
    let x: MLXArray
    let w: MLXArray
    let scales: MLXArray
    let biases: MLXArray
}

private func e173LazyCell(k: Int, n: Int, m: Int) -> E173LazyCell {
    E173LazyCell(
        x: zeros([1, m, k], dtype: .bfloat16),
        w: zeros([n, k / 8], dtype: .uint32),
        scales: zeros([n, k / 64], dtype: .bfloat16),
        biases: zeros([n, k / 64], dtype: .bfloat16)
    )
}

/// Median host nanoseconds per call of `body`, which must not evaluate.
private func e173MedianHostNanos(
    reps: Int, inner: Int, warmup: Int = 2, _ body: () -> Int
) -> [Double] {
    var sink = 0
    for _ in 0..<warmup { sink &+= body() }
    var samples: [Double] = []
    samples.reserveCapacity(reps)
    for _ in 0..<reps {
        let start = DispatchTime.now().uptimeNanoseconds
        sink &+= body()
        let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start)
        samples.append(elapsed / Double(inner))
    }
    // Report the sink so the optimiser cannot delete the measured work.
    e173HostSink &+= sink
    return samples
}

nonisolated(unsafe) private var e173HostSink = 0

@Suite(.serialized)
struct E173AdmissionCensusTests {
    private static var enabled: Bool {
        ProcessInfo.processInfo.environment["MLXFAST_RUN_E173_ADMISSION"] == "1"
    }

    @Test(.enabled(if: E173AdmissionCensusTests.enabled))
    func priceTheAdmissionPath() throws {
        let env = ProcessInfo.processInfo.environment
        let outPath = try #require(
            env["MLXFAST_E173_ADMISSION_OUT"],
            "MLXFAST_E173_ADMISSION_OUT must name the JSON destination")
        let reps = Int(env["MLXFAST_E173_REPS"] ?? "") ?? 11
        let inner = Int(env["MLXFAST_E173_INNER"] ?? "") ?? 4096

        var payload: [String: Any] = [
            "experiment": "e173-admission-path-host-census",
            "harness": "local",
            "evaluates_anything": false,
            "holds_model": false,
            "reps": reps,
            "inner_calls_per_timed_region": inner,
            "custom_qmv_arm": String(describing: Qwen35CustomQMV.arm),
            "minimum_table_width": Qwen35CustomQMV.minimumTableWidth,
            "routed_widths": [Qwen35CustomQMV.widths.lowerBound, Qwen35CustomQMV.widths.upperBound],
            "device": e173Device(),
        ]

        // --- 1a. `routable` at every width the round can take.
        var routableRows: [[String: Any]] = []
        for shape in e173ScoredShapes {
            for m in [1, 2, 3, 4, 5, 9, 512] {
                let cell = e173LazyCell(k: shape.k, n: shape.n, m: m)
                let accepted =
                    Qwen35CustomQMV.routable(
                        cell.x, cell.w, scales: cell.scales, biases: cell.biases,
                        groupSize: 64, bits: 4, mode: .affine) != nil
                let samples = e173MedianHostNanos(reps: reps, inner: inner) {
                    var local = 0
                    for _ in 0..<inner {
                        if let hit = Qwen35CustomQMV.routable(
                            cell.x, cell.w, scales: cell.scales, biases: cell.biases,
                            groupSize: 64, bits: 4, mode: .affine)
                        {
                            local &+= hit.m
                        } else {
                            local &+= 1
                        }
                    }
                    return local
                }
                let stats = e173Stats(samples)
                routableRows.append([
                    "shape": shape.name,
                    "k": shape.k,
                    "n": shape.n,
                    "m": m,
                    "calls_per_round": shape.callsPerRound,
                    "accepted": accepted,
                    "refusing_guard": accepted
                        ? "none"
                        : (m < Qwen35CustomQMV.widths.lowerBound
                            || m > Qwen35CustomQMV.widths.upperBound
                            ? "widths.contains(m)" : "unexpected"),
                    "host_ns_per_call": stats.median,
                    "host_ns_per_call_min": stats.min,
                    "host_ns_per_call_max": stats.max,
                ])
            }
        }
        payload["routable"] = routableRows

        // --- 1b. guards that no target cell reaches, priced as counterfactuals.
        // `n = 1024` is the unfused k_proj/v_proj width and `n = 2048` is the
        // proposal head's KV pack: both fail `n >= 4096`.
        var refusalRows: [[String: Any]] = []
        for (label, k, n, bits, groupSize) in [
            ("n1024_narrow_kv", 5120, 1024, 4, 64),
            ("n2048_head_kv_pack", 5120, 2048, 4, 64),
            ("bits8_first_guard", 5120, 16480, 8, 64),
            ("group32_first_guard", 5120, 16480, 4, 32),
        ] {
            let cell = e173LazyCell(k: k, n: n, m: 4)
            let accepted =
                Qwen35CustomQMV.routable(
                    cell.x, cell.w, scales: cell.scales, biases: cell.biases,
                    groupSize: groupSize, bits: bits, mode: .affine) != nil
            let samples = e173MedianHostNanos(reps: reps, inner: inner) {
                var local = 0
                for _ in 0..<inner {
                    if Qwen35CustomQMV.routable(
                        cell.x, cell.w, scales: cell.scales, biases: cell.biases,
                        groupSize: groupSize, bits: bits, mode: .affine) != nil
                    {
                        local &+= 2
                    } else {
                        local &+= 1
                    }
                }
                return local
            }
            let stats = e173Stats(samples)
            refusalRows.append([
                "case": label,
                "k": k,
                "n": n,
                "bits": bits,
                "group_size": groupSize,
                "m": 4,
                "accepted": accepted,
                "host_ns_per_call": stats.median,
                "host_ns_per_call_min": stats.min,
                "host_ns_per_call_max": stats.max,
            ])
        }
        payload["refusal_counterfactuals"] = refusalRows

        // --- 1c. the chunk-sum sidecar lookup, which every routed cell at
        // `m >= minimumTableWidth` performs before it dispatches.
        var sidecarRows: [[String: Any]] = []
        let sidecarCell = e173LazyCell(k: 5120, n: 16480, m: 4)
        let sidecarTable = zeros([(5120 / 512) * 32 * Qwen35CustomQMV.sumsStride(4)], dtype: .float32)
        for state in ["miss_empty_slots", "hit_published"] {
            Qwen35XSumsSidecar.slots = [Qwen35XSumsSidecar.Slot](
                repeating: Qwen35XSumsSidecar.Slot(), count: Qwen35XSumsSidecar.slotCount)
            if state == "hit_published" {
                Qwen35XSumsSidecar.publish(x: sidecarCell.x, table: sidecarTable)
            }
            let hit =
                Qwen35XSumsSidecar.take(sidecarCell.x, k: 5120, m: 4) != nil
            let samples = e173MedianHostNanos(reps: reps, inner: inner) {
                var local = 0
                for _ in 0..<inner {
                    if Qwen35XSumsSidecar.take(sidecarCell.x, k: 5120, m: 4) != nil {
                        local &+= 2
                    } else {
                        local &+= 1
                    }
                }
                return local
            }
            let stats = e173Stats(samples)
            sidecarRows.append([
                "state": state,
                "returned_table": hit,
                "slot_count": Qwen35XSumsSidecar.slotCount,
                "host_ns_per_call": stats.median,
                "host_ns_per_call_min": stats.min,
                "host_ns_per_call_max": stats.max,
            ])
        }
        Qwen35XSumsSidecar.slots = [Qwen35XSumsSidecar.Slot](
            repeating: Qwen35XSumsSidecar.Slot(), count: Qwen35XSumsSidecar.slotCount)
        payload["xsums_sidecar"] = sidecarRows

        // --- 1d. the two counter increments FINDING 413 isolated. This is the
        // arithmetic alone, with no code-layout or specialisation change.
        let counterSamples = e173MedianHostNanos(reps: reps, inner: inner) {
            for _ in 0..<inner {
                qwen35XSumsStandaloneFills &+= 1
                qwen35XSumsSidecarHits &+= 1
            }
            return qwen35XSumsStandaloneFills & 1
        }
        let counterStats = e173Stats(counterSamples)
        payload["counter_increment_pair"] = [
            "host_ns_per_call": counterStats.median,
            "host_ns_per_call_min": counterStats.min,
            "host_ns_per_call_max": counterStats.max,
        ]
        qwen35XSumsStandaloneFills = 0
        qwen35XSumsSidecarHits = 0

        // --- 1e. host graph-build cost of the whole routed entry point, which
        // is admission plus the dispatch record MLX will later encode. The
        // `m = 3` and `m = 4` arms differ exactly by the `tablePays` branch.
        var buildRows: [[String: Any]] = []
        let buildInner = max(16, inner / 128)
        for shape in e173ScoredShapes {
            for m in [1, 3, 4] {
                let cell = e173LazyCell(k: shape.k, n: shape.n, m: m)
                let samples = e173MedianHostNanos(reps: reps, inner: buildInner) {
                    var local = 0
                    for _ in 0..<buildInner {
                        autoreleasepool {
                            let y = qwen35RoutedQuantizedMM(
                                cell.x, cell.w, scales: cell.scales, biases: cell.biases,
                                groupSize: 64, bits: 4, mode: .affine)
                            local &+= y.ndim
                        }
                    }
                    return local
                }
                let stats = e173Stats(samples)
                buildRows.append([
                    "shape": shape.name,
                    "m": m,
                    "table_path": Qwen35CustomQMV.tablePays(m: m)
                        && Qwen35CustomQMV.widths.contains(m),
                    "routed": Qwen35CustomQMV.widths.contains(m),
                    "calls_per_round": shape.callsPerRound,
                    "host_ns_per_call": stats.median,
                    "host_ns_per_call_min": stats.min,
                    "host_ns_per_call_max": stats.max,
                ])
            }
        }
        payload["routed_entry_point_graph_build"] = buildRows
        payload["host_sink"] = e173HostSink

        try e173Write(payload, to: outPath)
        print("E173_ADMISSION_OUT \(outPath)")
    }
}

// MARK: - 2. GPU line items of `F`

private func e173Fill(_ shape: [Int], salt: Int, dtype: DType = .bfloat16) -> MLXArray {
    let count = shape.reduce(1, *)
    let values = (0..<count).map { index -> Float in
        Float(((index &* 1_664_525) &+ salt &* 1_013_904_223) % 2_003) / 2_003.0 - 0.5
    }
    let array = MLXArray(values).reshaped(shape).asType(dtype)
    eval(array)
    return array
}

/// Median seconds per call over `reps` timed regions of `inner` chained calls.
/// Chaining matters: MLX encodes independent kernels concurrently, which would
/// understate the cost of the strictly dependent per-round chain.
private func e173MedianSecondsPerCall(
    reps: Int, inner: Int, warmup: Int = 3, _ body: () -> [MLXArray]
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
    return samples
}

@Suite(.serialized)
struct E173GPULineItemTests {
    private static var enabled: Bool {
        ProcessInfo.processInfo.environment["MLXFAST_RUN_E173_GPU"] == "1"
    }

    @Test(.enabled(if: E173GPULineItemTests.enabled))
    func priceNonWeightPassGPUWork() throws {
        let env = ProcessInfo.processInfo.environment
        let outPath = try #require(
            env["MLXFAST_E173_GPU_OUT"],
            "MLXFAST_E173_GPU_OUT must name the JSON destination")
        let reps = Int(env["MLXFAST_E173_GPU_REPS"] ?? "") ?? 9
        let inner = Int(env["MLXFAST_E173_GPU_INNER"] ?? "") ?? 8
        let widths = [1, 3, 4, 5]
        let cacheLength = Int(env["MLXFAST_E173_CACHE_LEN"] ?? "") ?? 768

        var payload: [String: Any] = [
            "experiment": "e173-non-weight-pass-gpu-line-items",
            "harness": "local",
            "cool_gate_passed_real_gate": false,
            "gate_qualified_for_timing": false,
            "reps": reps,
            "inner_calls_per_timed_region": inner,
            "cache_length": cacheLength,
            "device": e173Device(),
        ]
        if let temp = e173GPUTemperature() { payload["gpu_temp_entry_c"] = temp }

        var items: [[String: Any]] = []

        func record(
            _ name: String, family: String, callsPerRound: Int, m: Int,
            bytes: Int, samples: [Double], note: String
        ) {
            let stats = e173Stats(samples)
            items.append([
                "name": name,
                "family": family,
                "calls_per_round": callsPerRound,
                "m": m,
                "bytes_per_call": bytes,
                "seconds_per_call": stats.median,
                "seconds_per_call_min": stats.min,
                "seconds_per_call_max": stats.max,
                "ms_per_round": stats.median * 1e3 * Double(callsPerRound),
                "note": note,
            ])
        }

        // --- 2a. the 48 gated delta recurrences. State is FP32
        // [1, 48, 128, 128] per layer and is read and written every round, so
        // its traffic is M-independent: exactly the shape of `F`.
        let hk = 16, dk = 128, hv = 48, dv = 128
        let aLog = (zeros([hv], dtype: .float32) + Float(-0.5)).asType(.bfloat16)
        let dtBias = (zeros([hv], dtype: .float32) + Float(0.1)).asType(.bfloat16)
        let state0 = zeros([1, hv, dk, dv], dtype: .float32)
        eval(aLog, dtBias, state0)
        for m in widths {
            let q = e173Fill([1, m, hk, dk], salt: 1)
            let k = e173Fill([1, m, hk, dk], salt: 2)
            let v = e173Fill([1, m, hv, dv], salt: 3)
            let a = e173Fill([1, m, hv], salt: 4)
            let b = e173Fill([1, m, hv], salt: 5)
            let samples = e173MedianSecondsPerCall(reps: reps, inner: inner) {
                var outs: [MLXArray] = []
                var state = state0
                for _ in 0..<inner {
                    let (y, next) = gatedDeltaUpdate(
                        q: q, k: k, v: v, a: a, b: b,
                        aLog: aLog, dtBias: dtBias, state: state)
                    outs.append(y)
                    state = next
                }
                outs.append(state)
                return outs
            }
            record(
                "gdn.recurrence", family: "gdn", callsPerRound: 48, m: m,
                bytes: hv * dk * dv * 4 * 2, samples: samples,
                note: "state read + write per call, FP32")
        }
        Memory.clearCache()

        // --- 2b. the 16 full-attention SDPA calls against a decode-length
        // cache. 24 query heads, 4 KV heads, head dim 256.
        for m in widths {
            let q = e173Fill([1, 24, m, 256], salt: 11)
            let kCache = e173Fill([1, 4, cacheLength, 256], salt: 12)
            let vCache = e173Fill([1, 4, cacheLength, 256], salt: 13)
            let samples = e173MedianSecondsPerCall(reps: reps, inner: inner) {
                var outs: [MLXArray] = []
                var query = q
                for _ in 0..<inner {
                    let out = MLXFast.scaledDotProductAttention(
                        queries: query, keys: kCache, values: vCache,
                        scale: 0.0625, mask: .none)
                    outs.append(out)
                    query = q + out[0..., 0..., 0..., 0..<1] * 1e-30
                }
                return outs
            }
            record(
                "fa.sdpa", family: "attn", callsPerRound: 16, m: m,
                bytes: 2 * 4 * cacheLength * 256 * 2, samples: samples,
                note: "K and V cache read at cache_length, mask none")
        }
        Memory.clearCache()

        // --- 2c. the small-operator dispatch floor. A row-local RMSNorm over
        // [1, M, 5120] moves at most 92 KB at M = 9, so its wall time is the
        // per-dispatch floor of this host rather than its own arithmetic.
        for m in widths {
            let x = e173Fill([1, m, 5120], salt: 21)
            let weight = e173Fill([5120], salt: 22)
            let samples = e173MedianSecondsPerCall(reps: reps, inner: inner) {
                var outs: [MLXArray] = []
                var value = x
                for _ in 0..<inner {
                    value = MLXFast.rmsNorm(value, weight: weight, eps: 1e-6)
                    outs.append(value)
                }
                return outs
            }
            record(
                "envelope.rms_norm_5120", family: "dispatch_floor", callsPerRound: 127, m: m,
                bytes: m * 5120 * 2 * 2, samples: samples,
                note: "127 = 63 boundary-fused entry norms + 64 post-attention norms")
        }
        Memory.clearCache()

        // --- 2d. the MLP element-wise stage over the 17408-wide intermediate.
        for m in widths {
            let gate = e173Fill([1, m, 17408], salt: 31)
            let up = e173Fill([1, m, 17408], salt: 32)
            let samples = e173MedianSecondsPerCall(reps: reps, inner: inner) {
                var outs: [MLXArray] = []
                var value = gate
                for _ in 0..<inner {
                    value = silu(value) * up
                    outs.append(value)
                }
                return outs
            }
            record(
                "mlp.swiglu_elementwise", family: "mlp", callsPerRound: 64, m: m,
                bytes: m * 17408 * 2 * 3, samples: samples,
                note: "gate and up read, product written")
        }
        Memory.clearCache()

        // --- 2e. the proposal head's BF16 precision-island GEMMs. One KV pack
        // of [2048, 5120] and one Q island of [1024, 5120] per head step, plus
        // the scatter back into the 12288-wide Q output.
        let kvIsland = e173Fill([2048, 5120], salt: 41)
        let qIsland = e173Fill([1024, 5120], salt: 42)
        let qIndices = arange(0, 1024, dtype: .int32)
        eval(qIndices)
        for m in widths {
            let x = e173Fill([1, m, 5120], salt: 43)
            let base = e173Fill([1, m, 12288], salt: 44)
            let kvSamples = e173MedianSecondsPerCall(reps: reps, inner: inner) {
                var outs: [MLXArray] = []
                var value = x
                for _ in 0..<inner {
                    let y = matmul(value, kvIsland.transposed(1, 0))
                    outs.append(y)
                    value = x + y[0..., 0..., 0..<1] * 1e-30
                }
                return outs
            }
            record(
                "head.island_kv_matmul", family: "head", callsPerRound: 1, m: m,
                bytes: 2048 * 5120 * 2, samples: kvSamples,
                note: "BF16 KV pack read once per round by appendHistoryKV")

            let qSamples = e173MedianSecondsPerCall(reps: reps, inner: inner) {
                var outs: [MLXArray] = []
                var value = x
                for _ in 0..<inner {
                    let exact = matmul(value, qIsland.transposed(1, 0))
                    let scattered = putAlong(
                        base, qIndices.reshaped([1, 1, -1]), values: exact, axis: -1)
                    outs.append(scattered)
                    value = x + scattered[0..., 0..., 0..<1] * 1e-30
                }
                return outs
            }
            record(
                "head.island_q_matmul_scatter", family: "head", callsPerRound: 1, m: m,
                bytes: 1024 * 5120 * 2 + m * 12288 * 2 * 2, samples: qSamples,
                note: "Q island GEMM plus putAlong into the 12288-wide output")
        }
        Memory.clearCache()

        // --- 2f. the vocabulary top-two readout over [1, M, 248320].
        for m in widths {
            let logits = e173Fill([1, m, 248_320], salt: 51)
            let samples = e173MedianSecondsPerCall(reps: reps, inner: max(2, inner / 2)) {
                var outs: [MLXArray] = []
                for _ in 0..<max(2, inner / 2) {
                    let (ids, values) = Qwen36MTPBlockSession.linearTopTwoRows(logits)
                    outs.append(ids)
                    outs.append(values)
                }
                return outs
            }
            record(
                "head.top_two_readout", family: "readout", callsPerRound: 1, m: m,
                bytes: m * 248_320 * 2, samples: samples,
                note: "logits read by the two-pass reducer")
        }
        Memory.clearCache()

        payload["items"] = items
        if let temp = e173GPUTemperature() { payload["gpu_temp_exit_c"] = temp }
        try e173Write(payload, to: outPath)
        print("E173_GPU_OUT \(outPath)")
    }
}
