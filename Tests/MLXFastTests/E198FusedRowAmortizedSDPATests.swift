import Foundation
import MLX
import MLXLMCommon
import MLXRandom
import Metal
import ObjectiveC
import Testing

// E198 -- build the row-amortized one-pass fused SDPA kernel (Route 1) and
// gate it.
//
// FINDING 507 priced the opportunity: the vendored `sdpa_vector` kernel gives
// one threadgroup to every (query head, query row) pair, so the KV window is
// streamed `gqa * rows` times per layer. At the scored cell that is x30
// redundant load traffic at ~9% ALU and ~9% DRAM, and redundant load
// throughput is the binding resource. `FusedRowAmortizedSDPA` holds the query
// rows of one head in registers so a KV tile is read once per head.
//
// Three gates live here:
//   Stage 1  static pipeline limits per width M. Bit-exactness needs the full
//            1024-thread threadgroup, so a pipeline whose
//            `maxTotalThreadsPerThreadgroup` falls below 1024 IS the register
//            wall for that width.
//   Stage 3  bf16 value gate against the shipped two-call split, with a
//            positive control built from the shipped source by one
//            substitution, which must fire.
//   Stage 2  isolated dispatch pricing against today's pair and the
//            single-row floor.
//
// Research instrument. `Tests/` is never packaged into a submission.
// harness=local, within-session relative measurement, no thermal gate,
// no score.

/// Live-path certification needle (RULE 384/387): a dedicated >= 16-byte
/// literal on the executed path of this probe, not on any failure path.
let e198LivePathNeedle = "E198-ROUTE1-ROW-AMORTIZED-FUSED-SDPA-LIVE-PATH-2026-08-24"

private func e198IntList(_ key: String, _ fallback: [Int]) -> [Int] {
    guard let raw = ProcessInfo.processInfo.environment[key] else { return fallback }
    let parsed = raw.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
    return parsed.isEmpty ? fallback : parsed
}

private func e198WriteReport(_ payload: [String: Any], envKey: String) throws {
    guard let path = ProcessInfo.processInfo.environment[envKey] else { return }
    let data = try JSONSerialization.data(
        withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
    try data.write(to: URL(fileURLWithPath: path))
}

// MARK: - The scored geometry

private enum E198Probe {
    static let heads = 24
    static let kvHeads = 4
    static let headDim = 256
    static let fullAttentionLayers = 16
    static let split = 5
    static var gqa: Int { heads / kvHeads }
    static var scale: Float { 1.0 / Float(headDim).squareRoot() }
    /// Every width this probe can drive, so `attend` never refuses on the
    /// shipped default while a research arm is being priced.
    static let allWidths: Set<Int> = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    static func rows(_ length: Int, heads count: Int) -> MLXArray {
        MLXRandom.normal([1, count, length, headDim]).asType(.bfloat16)
    }

    static func queries(rows length: Int) -> MLXArray { rows(length, heads: heads) }
    static func newKV(rows length: Int) -> MLXArray { rows(length, heads: kvHeads) }

    /// A cache holding `length` rows with headroom, so every timed or compared
    /// window carries the production strides: `k_head_stride` is the CAPACITY
    /// stride, not the length stride.
    static func seededCache(length: Int) -> KVCacheSimple {
        let cache = KVCacheSimple()
        _ = cache.update(keys: newKV(rows: length), values: newKV(rows: length))
        _ = cache.update(keys: newKV(rows: 16), values: newKV(rows: 16))
        cache.offset = length
        eval(cache.innerState())
        return cache
    }

    /// The committed window a round of `m` rows produces, exactly what
    /// `attentionWithCacheUpdate` passes to the split.
    static func committedWindow(cache: KVCacheSimple, kvLength: Int, m: Int) -> (
        MLXArray, MLXArray
    ) {
        cache.offset = kvLength
        let kv = newKV(rows: m)
        let (keys, values) = cache.update(keys: kv, values: kv)
        eval(keys, values)
        return (keys, values)
    }

    static func sdpa(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray) -> MLXArray {
        MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale,
            mask: q.dim(2) == 1 ? .none : .causal)
    }

    /// The shipped two-call split, reproduced on an already committed window.
    static func today(queries q: MLXArray, keys: MLXArray, values: MLXArray) -> MLXArray {
        let qL = q.dim(2)
        let kL = keys.dim(2)
        let kSplit = kL - (qL - split)
        let outA = MLXFast.scaledDotProductAttention(
            queries: q[0..., 0..., 0 ..< split, 0...],
            keys: keys[0..., 0..., 0 ..< kSplit, 0...],
            values: values[0..., 0..., 0 ..< kSplit, 0...],
            scale: scale, mask: .causal)
        let outB = MLXFast.scaledDotProductAttention(
            queries: q[0..., 0..., split..., 0...],
            keys: keys, values: values, scale: scale, mask: .causal)
        return concatenated([outA, outB], axis: 2)
    }

    static func fused(queries q: MLXArray, keys: MLXArray, values: MLXArray) -> MLXArray? {
        FusedRowAmortizedSDPA.attend(
            queries: q, keys: keys, values: values, scale: scale, serving: allWidths)
    }
}

// MARK: - Metal pipeline ledger (Stage 1 register gate)

/// Records the static limits of every compute pipeline Metal builds, and
/// appends them to `MLX_E198_PIPELINE_OUT` as JSON lines AS THEY ARE CREATED.
/// Writing on creation matters: a width whose register pressure drops
/// `maxTotalThreadsPerThreadgroup` below the 1024 threads this kernel requires
/// makes MLX throw at dispatch, which can end the process before any test
/// assertion runs. The evidence is on disk before that can happen.
private final class E198PipelineLedger: @unchecked Sendable {
    static let shared = E198PipelineLedger()

    private let lock = NSLock()
    private var seen: Set<String> = []

    func note(pipeline: AnyObject, name: String) {
        guard name.contains("qwen_mtp_fused_row_amortized_sdpa") else { return }
        guard let state = pipeline as? MTLComputePipelineState else { return }
        lock.lock()
        defer { lock.unlock() }
        guard !seen.contains(name) else { return }
        seen.insert(name)
        let record: [String: Any] = [
            "kernel": name,
            "max_total_threads_per_threadgroup": state.maxTotalThreadsPerThreadgroup,
            "thread_execution_width": state.threadExecutionWidth,
            "static_threadgroup_memory_bytes": state.staticThreadgroupMemoryLength,
            "required_threads_per_threadgroup": 1024,
            "fits_required_threadgroup": state.maxTotalThreadsPerThreadgroup >= 1024,
        ]
        guard let path = ProcessInfo.processInfo.environment["MLX_E198_PIPELINE_OUT"],
            let data = try? JSONSerialization.data(withJSONObject: record, options: [.sortedKeys])
        else { return }
        var line = String(data: data, encoding: .utf8) ?? ""
        line += "\n"
        if let handle = FileHandle(forWritingAtPath: path) {
            handle.seekToEndOfFile()
            handle.write(Data(line.utf8))
            try? handle.close()
        } else {
            try? Data(line.utf8).write(to: URL(fileURLWithPath: path))
        }
    }
}

private typealias E198NewPipelineIMP = @convention(c) (
    AnyObject, Selector, AnyObject, UnsafeMutableRawPointer?
) -> UnsafeMutableRawPointer?

private func e198SwizzleNewPipeline(_ cls: AnyClass) -> Bool {
    let selector = NSSelectorFromString("newComputePipelineStateWithFunction:error:")
    guard let method = class_getInstanceMethod(cls, selector) else { return false }
    let original = unsafeBitCast(method_getImplementation(method), to: E198NewPipelineIMP.self)
    let replacement: @convention(block) (AnyObject, AnyObject, UnsafeMutableRawPointer?)
        -> UnsafeMutableRawPointer? = { device, function, errorOut in
            let result = original(device, selector, function, errorOut)
            if let result {
                let pipeline = Unmanaged<AnyObject>.fromOpaque(result).takeUnretainedValue()
                E198PipelineLedger.shared.note(
                    pipeline: pipeline, name: (function as? MTLFunction)?.name ?? "<unnamed>")
            }
            return result
        }
    method_setImplementation(method, imp_implementationWithBlock(replacement))
    return true
}

// MARK: - Stage 1: register / occupancy gate

@Suite(.serialized)
struct E198RegisterGateTests {
    @Test(
        .enabled(
            if: ProcessInfo.processInfo.environment["MLX_E198_REGISTER"] == "1",
            "set MLX_E198_REGISTER=1 to run the fused-kernel register gate"))
    func reportsStaticPipelineLimitsPerWidth() throws {
        _ = e198LivePathNeedle
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("no Metal device")
            return
        }
        #expect(e198SwizzleNewPipeline(type(of: device as AnyObject)))

        let widths = e198IntList("MLX_E198_M", [6])
        let kvLength = e198IntList("MLX_E198_KV", [512])[0]
        for m in widths {
            let cache = E198Probe.seededCache(length: kvLength)
            let (keys, values) = E198Probe.committedWindow(
                cache: cache, kvLength: kvLength, m: m)
            let q = E198Probe.queries(rows: m)
            eval(q)
            guard let out = E198Probe.fused(queries: q, keys: keys, values: values) else {
                Issue.record("fused kernel refused width \(m)")
                continue
            }
            eval(out)
            #expect(out.dim(2) == m)
        }
    }
}

// MARK: - Stage 3: bf16 value gate against the shipped split

@Suite(.serialized)
struct E198FusedExactnessTests {
    /// The positive control: the shipped source with the causal window of every
    /// row widened by one key. It must differ from the split on thousands of
    /// elements, which proves the comparison below can fail.
    private static let controlKernel: MLXFast.MLXFastKernel = {
        let perturbed = FusedRowAmortizedSDPA.kernelSource.replacingOccurrences(
            of: "if (i <= N - M + q_row) {", with: "if (i <= N - M + q_row + 1) {")
        precondition(
            perturbed != FusedRowAmortizedSDPA.kernelSource,
            "positive control substitution did not apply")
        return MLXFast.metalKernel(
            name: "qwen_mtp_fused_row_amortized_sdpa_e198_control",
            inputNames: ["queries", "keys", "values", "scale"],
            outputNames: ["out"],
            source: perturbed,
            header: "#include <metal_simdgroup>\n",
            ensureRowContiguous: false)
    }()

    private static func control(
        queries q: MLXArray, keys: MLXArray, values: MLXArray
    ) -> MLXArray {
        controlKernel(
            [q, keys, values, MLXArray(E198Probe.scale)],
            template: [("M", q.dim(2))],
            grid: (32, 32, q.dim(1) * q.dim(2)),
            threadGroup: (32, 32, 1),
            outputShapes: [[1, q.dim(1), q.dim(2), E198Probe.headDim]],
            outputDTypes: [.bfloat16])[0]
    }

    private static func compare(_ a: MLXArray, _ b: MLXArray) -> (Int, Double) {
        let left = a.asArray(Float.self)
        let right = b.asArray(Float.self)
        var differing = 0
        var maxAbsolute = 0.0
        for index in left.indices where left[index] != right[index] {
            differing += 1
            maxAbsolute = Swift.max(maxAbsolute, Double(Swift.abs(left[index] - right[index])))
        }
        return (differing, maxAbsolute)
    }

    @Test(
        .enabled(
            if: ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1",
            "set MLXFAST_RUN_MLX_RUNTIME_TESTS=1 to run the fused SDPA value gate"))
    func fusedMatchesTheSplitBitForBit() throws {
        _ = e198LivePathNeedle
        let widths = e198IntList("MLX_E198_M", [6])
        // A scored 512-token leg runs the cache offset from 512 to 1024, so
        // both weighted cells of the one-pass family are covered. The
        // kL = 1024 family boundary is never crossed by interpolation here:
        // both cells are measured.
        let kvLengths = e198IntList("MLX_E198_KV", [512, 1024])
        var cells: [[String: Any]] = []

        for kvLength in kvLengths {
            for m in widths {
                let cache = E198Probe.seededCache(length: kvLength)
                let (keys, values) = E198Probe.committedWindow(
                    cache: cache, kvLength: kvLength, m: m)
                let q = E198Probe.queries(rows: m)
                eval(q)

                let reference = E198Probe.today(queries: q, keys: keys, values: values)
                guard let candidate = E198Probe.fused(queries: q, keys: keys, values: values)
                else {
                    // A refusal is correct only at the vendored two-pass key
                    // length, where the split changes reduction order.
                    let boundary = keys.dim(2) >= FusedRowAmortizedSDPA.twoPassKeyLength
                    if !boundary {
                        Issue.record("fused kernel refused served cell m=\(m) kL=\(keys.dim(2))")
                    }
                    cells.append([
                        "m": m, "kv": kvLength, "kL": keys.dim(2), "served": false,
                        "declined_at_two_pass_boundary": boundary,
                    ])
                    continue
                }
                let controlOut = Self.control(queries: q, keys: keys, values: values)
                eval(reference, candidate, controlOut)

                let (differing, maxAbsolute) = Self.compare(reference, candidate)
                let (controlDiffering, controlMax) = Self.compare(reference, controlOut)

                #expect(candidate.shape == reference.shape)
                #expect(
                    differing == 0,
                    "fused differs from split in \(differing) elements at m=\(m) kv=\(kvLength), max abs \(maxAbsolute)"
                )
                #expect(
                    controlDiffering > 1000,
                    "positive control fired on only \(controlDiffering) elements at m=\(m) kv=\(kvLength)"
                )

                cells.append([
                    "m": m, "kv": kvLength, "kL": keys.dim(2), "served": true,
                    "elements": reference.size,
                    "differing_elements": differing,
                    "max_absolute_difference": maxAbsolute,
                    "control_differing_elements": controlDiffering,
                    "control_max_absolute_difference": controlMax,
                ])
            }
        }

        try e198WriteReport(
            [
                "probe": "e198-fused-value-gate",
                "harness": "local",
                "probe_needle": e198LivePathNeedle,
                "heads": E198Probe.heads, "kv_heads": E198Probe.kvHeads,
                "head_dim": E198Probe.headDim, "split_row": E198Probe.split,
                "cells": cells,
            ], envKey: "MLX_E198_EXACTNESS_OUT")
    }
}

// MARK: - Stage 2: isolated dispatch pricing

/// A streaming macmon sampler. The pricing session runs ABBA-counterbalanced
/// under no thermal gate, so the entry and exit temperature of every arm is the
/// required thermal record. One `macmon pipe` process feeds every reading, so a
/// reading costs nothing at the arm boundary and is at most one interval stale.
/// A single blocking `macmon pipe -s1` call costs about 2.7 s here and would
/// insert more idle time between arms than the arms themselves take.
private final class E198TemperatureSampler: @unchecked Sendable {
    private let process = Process()
    private let pipe = Pipe()
    private let lock = NSLock()
    private var latest: Double?
    private var pending = Data()

    static func binaryPath() -> String? {
        var candidates: [String] = []
        if let explicit = ProcessInfo.processInfo.environment["MLXFAST_E198_MACMON"] {
            candidates.append(explicit)
        }
        candidates.append("/opt/homebrew/bin/macmon")
        candidates.append("/usr/local/bin/macmon")
        // The role home, read from the environment: `NSHomeDirectory()`
        // resolves to the account home, not to `$HOME`, in this test process.
        if let home = ProcessInfo.processInfo.environment["HOME"] {
            candidates.append(home + "/bin/macmon")
        }
        return candidates.first { FileManager.default.isExecutableFile(atPath: $0) }
    }

    init?(intervalMilliseconds: Int = 500) {
        guard let binary = Self.binaryPath() else { return nil }
        process.executableURL = URL(fileURLWithPath: binary)
        process.arguments = ["pipe", "-s0", "-i", String(intervalMilliseconds)]
        process.standardOutput = pipe
        process.standardError = FileHandle.nullDevice
        pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            self?.consume(handle.availableData)
        }
        do { try process.run() } catch { return nil }
    }

    private func consume(_ data: Data) {
        guard !data.isEmpty else { return }
        lock.lock()
        defer { lock.unlock() }
        pending.append(data)
        while let newline = pending.firstIndex(of: 0x0A) {
            let line = pending[pending.startIndex ..< newline]
            pending = pending[pending.index(after: newline)...]
            if let object = try? JSONSerialization.jsonObject(with: Data(line)),
                let root = object as? [String: Any],
                let temp = root["temp"] as? [String: Any],
                let gpu = temp["gpu_temp_avg"] as? Double
            {
                latest = gpu
            }
        }
    }

    func read() -> Double? {
        lock.lock()
        defer { lock.unlock() }
        return latest
    }

    func stop() {
        pipe.fileHandleForReading.readabilityHandler = nil
        if process.isRunning { process.terminate() }
    }
}

private enum E198Mode: String {
    case indep
    case serial
}

@Suite(.serialized)
struct E198DispatchPricingTests {
    @Test(
        .enabled(
            if: ProcessInfo.processInfo.environment["MLX_E198_TIMING"] == "1",
            "set MLX_E198_TIMING=1 to run the fused-dispatch pricing session"))
    func pricesTheFusedDispatchAgainstTodaysPair() throws {
        _ = e198LivePathNeedle
        let widths = e198IntList("MLX_E198_M", [6])
        let kvLengths = e198IntList("MLX_E198_KV", [512, 1024])
        let blocks = e198IntList("MLX_E198_BLOCKS", [6])[0]
        let reps = e198IntList("MLX_E198_REPS", [20])[0]
        let warmup = e198IntList("MLX_E198_WARMUP", [6])[0]
        let chain = e198IntList("MLX_E198_CHAIN", [8])[0]

        struct Cell {
            var arm: String
            var mode: E198Mode
            var m: Int
            var kv: Int
            var kL: Int
            var run: () -> Void
        }

        var cells: [Cell] = []
        let poolSize = 4

        for kvLength in kvLengths {
            for m in widths {
                let cache = E198Probe.seededCache(length: kvLength)
                let (keys, values) = E198Probe.committedWindow(
                    cache: cache, kvLength: kvLength, m: m)
                let kL = keys.dim(2)
                let kSplit = kL - (m - E198Probe.split)
                let keysA = keys[0..., 0..., 0 ..< kSplit, 0...]
                let valuesA = values[0..., 0..., 0 ..< kSplit, 0...]
                eval(keysA, valuesA)

                let wide = (0 ..< poolSize).map { _ in E198Probe.queries(rows: m) }
                let headRows = (0 ..< poolSize).map { _ in
                    E198Probe.queries(rows: E198Probe.split)
                }
                let tailRows = (0 ..< poolSize).map { _ in
                    E198Probe.queries(rows: m - E198Probe.split)
                }
                let singleRows = (0 ..< poolSize).map { _ in E198Probe.queries(rows: 1) }
                eval(wide + headRows + tailRows + singleRows)

                // today: the shipped pair, call A on the shortened window and
                // call B on the full window, as one FA layer issues it.
                for mode in [E198Mode.indep, .serial] {
                    let pair: () -> Void = {
                        switch mode {
                        case .indep:
                            var outputs: [MLXArray] = []
                            outputs.reserveCapacity(chain * 2)
                            for index in 0 ..< chain {
                                outputs.append(
                                    E198Probe.sdpa(
                                        headRows[index % poolSize], keysA, valuesA))
                                outputs.append(
                                    E198Probe.sdpa(tailRows[index % poolSize], keys, values))
                            }
                            eval(outputs)
                        case .serial:
                            var a = headRows[0]
                            var b = tailRows[0]
                            for _ in 0 ..< chain {
                                a = E198Probe.sdpa(a, keysA, valuesA)
                                b = E198Probe.sdpa(b, keys, values)
                            }
                            eval(a, b)
                        }
                    }
                    cells.append(
                        Cell(arm: "today_pair", mode: mode, m: m, kv: kvLength, kL: kL,
                            run: pair))

                    let fused: () -> Void = {
                        switch mode {
                        case .indep:
                            var outputs: [MLXArray] = []
                            outputs.reserveCapacity(chain)
                            for index in 0 ..< chain {
                                if let out = E198Probe.fused(
                                    queries: wide[index % poolSize], keys: keys,
                                    values: values)
                                {
                                    outputs.append(out)
                                }
                            }
                            eval(outputs)
                        case .serial:
                            var current = wide[0]
                            for _ in 0 ..< chain {
                                guard let out = E198Probe.fused(
                                    queries: current, keys: keys, values: values)
                                else { return }
                                current = out
                            }
                            eval(current)
                        }
                    }
                    cells.append(
                        Cell(arm: "fused", mode: mode, m: m, kv: kvLength, kL: kL, run: fused))

                    // floor: one single-row one-pass dispatch over the same
                    // window. FINDING 507's floor identity is
                    // F(m, kL) = T(1, kL) + (m - 1) * b0.
                    let floor: () -> Void = {
                        switch mode {
                        case .indep:
                            var outputs: [MLXArray] = []
                            outputs.reserveCapacity(chain)
                            for index in 0 ..< chain {
                                outputs.append(
                                    E198Probe.sdpa(singleRows[index % poolSize], keys, values))
                            }
                            eval(outputs)
                        case .serial:
                            var current = singleRows[0]
                            for _ in 0 ..< chain {
                                current = E198Probe.sdpa(current, keys, values)
                            }
                            eval(current)
                        }
                    }
                    cells.append(
                        Cell(arm: "one_row_floor", mode: mode, m: m, kv: kvLength, kL: kL,
                            run: floor))
                }
            }
        }

        let settleWeights = MLXRandom.normal([2048, 2048]).asType(.bfloat16)
        let settleInput = MLXRandom.normal([64, 2048]).asType(.bfloat16)
        eval(settleWeights, settleInput)
        let settle = { eval(matmul(settleInput, settleWeights)) }

        // Started before warmup so the first block already has a reading.
        let sampler = E198TemperatureSampler()
        defer { sampler?.stop() }

        for _ in 0 ..< 40 { settle() }
        for cell in cells {
            for _ in 0 ..< warmup { cell.run() }
        }

        var samples: [[String: Any]] = []
        for block in 0 ..< blocks {
            let ascending = block % 2 == 0
            let order = ascending ? Array(cells.indices) : Array(cells.indices.reversed())
            for _ in 0 ..< 20 { settle() }
            for (position, index) in order.enumerated() {
                let cell = cells[index]
                let entryTemperature = sampler?.read()
                let start = DispatchTime.now().uptimeNanoseconds
                for _ in 0 ..< reps { cell.run() }
                let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start)
                let exitTemperature = sampler?.read()
                samples.append([
                    "arm": cell.arm, "mode": cell.mode.rawValue, "m": cell.m,
                    "kv": cell.kv, "kL": cell.kL, "chain": chain,
                    "block": block, "ascending": ascending, "position": position,
                    "microseconds": elapsed / 1e3 / Double(reps) / Double(chain),
                    "reps": reps,
                    "entry_gpu_temperature_c": entryTemperature ?? -1,
                    "exit_gpu_temperature_c": exitTemperature ?? -1,
                ])
            }
        }

        try e198WriteReport(
            [
                "probe": "e198-fused-dispatch-pricing",
                "harness": "local",
                "probe_needle": e198LivePathNeedle,
                "cool_gate_passed_real_gate": false,
                "gate_qualified_for_timing": false,
                "abba_counterbalanced": true,
                "temperature_source": E198TemperatureSampler.binaryPath() ?? "unavailable",
                "blocks": blocks, "reps": reps, "warmup": warmup, "chain": chain,
                "full_attention_layers": E198Probe.fullAttentionLayers,
                "gqa_factor": E198Probe.gqa, "split_row": E198Probe.split,
                "samples": samples,
            ], envKey: "MLX_E198_TIMING_OUT")
    }
}
