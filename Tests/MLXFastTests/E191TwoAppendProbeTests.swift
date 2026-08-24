import Foundation
import MLX
import MLXLMCommon
import MLXRandom
import Metal
import ObjectiveC
import Testing

// E191 Stage 0 -- does the m >= 6 split-SDPA path pay a cache-COPY (mechanism
// M1), and would the recorded two-append fix remove it?
//
// FINDING 489 priced a +87.7 us (kv=512) to +173.3 us (kv=1024) step at m = 6
// in E186's isolated `sdpa` family and read the KV-length growth as the
// signature of M1: "axis-2 strided slices of cachedKeys/cachedValues
// materialize full copies". E186 family 5 calls
// `MLXFast.scaledDotProductAttention` DIRECTLY on full contiguous keys, so it
// measures the UNSPLIT call. This probe measures the three forms side by side
// on the same host, in the same session:
//
//   today     the shipped production form: one `cache.update`, then two sdpa
//             calls over axis-2 SLICES of the returned arrays
//   twoAppend the recorded candidateFix: two `cache.update` calls, each sdpa
//             call reading its returned array in full
//   unsplit   one direct sdpa call at width m (E186 family 5; the composed
//             fallback at m >= 6)
//
// Two sections, each opt-in:
//
//   MLXFAST_E191_DISPATCH=1  counts the kernels Metal actually receives per
//                            form, by swizzling the same three selectors E57
//                            used. A materialized KV copy MUST appear here as
//                            a `copy_*` dispatch; its absence refutes M1
//                            without any timing noise.
//   MLXFAST_E191_TIMING=1    ABBA-counterbalanced absolute time per form at
//                            widths 5..9 and kv 512/1024, which prices the
//                            production m = 5 -> 6 step and the fix.
//
// Research instrument. `Tests/` is never packaged into a submission.
// harness=local, within-session relative measurement, no thermal gate,
// no score.

// MARK: - Metal dispatch ledger (same technique as E57, kept file-private)

private struct E191DispatchRecord {
    var kernel: String
    var grid: MTLSize
    var threadgroup: MTLSize
}

/// `@unchecked Sendable` because every mutable field is reached only under
/// `lock`, and the swizzled Metal selectors can be called from any thread.
private final class E191Ledger: @unchecked Sendable {
    static let shared = E191Ledger()

    private let lock = NSLock()
    private var pipelineNames: [ObjectIdentifier: String] = [:]
    private var encoderBinding: [ObjectIdentifier: String] = [:]
    private var records: [E191DispatchRecord] = []
    private var recording = false

    func note(pipeline: AnyObject, name: String) {
        lock.lock()
        pipelineNames[ObjectIdentifier(pipeline)] = name
        lock.unlock()
    }

    func bind(encoder: AnyObject, pipeline: AnyObject) {
        lock.lock()
        encoderBinding[ObjectIdentifier(encoder)] =
            pipelineNames[ObjectIdentifier(pipeline)] ?? "<unmapped>"
        lock.unlock()
    }

    func dispatch(encoder: AnyObject, grid: MTLSize, threadgroup: MTLSize) {
        lock.lock()
        if recording {
            records.append(
                E191DispatchRecord(
                    kernel: encoderBinding[ObjectIdentifier(encoder)] ?? "<unbound>",
                    grid: grid, threadgroup: threadgroup))
        }
        lock.unlock()
    }

    func start() {
        lock.lock()
        records = []
        recording = true
        lock.unlock()
    }

    func stop() -> [E191DispatchRecord] {
        lock.lock()
        recording = false
        let taken = records
        records = []
        lock.unlock()
        return taken
    }
}

private typealias E191DispatchIMP = @convention(c) (AnyObject, Selector, MTLSize, MTLSize)
    -> Void
private typealias E191SetPipelineIMP = @convention(c) (AnyObject, Selector, AnyObject) -> Void
private typealias E191NewPipelineIMP = @convention(c) (
    AnyObject, Selector, AnyObject, UnsafeMutableRawPointer?
) -> UnsafeMutableRawPointer?

private func e191SwizzleDispatch(_ cls: AnyClass, _ name: String) -> Bool {
    let selector = NSSelectorFromString(name)
    guard let method = class_getInstanceMethod(cls, selector) else { return false }
    let original = unsafeBitCast(method_getImplementation(method), to: E191DispatchIMP.self)
    let replacement: @convention(block) (AnyObject, MTLSize, MTLSize) -> Void = {
        encoder, grid, threadgroup in
        E191Ledger.shared.dispatch(encoder: encoder, grid: grid, threadgroup: threadgroup)
        original(encoder, selector, grid, threadgroup)
    }
    method_setImplementation(method, imp_implementationWithBlock(replacement))
    return true
}

private func e191SwizzleSetPipeline(_ cls: AnyClass) -> Bool {
    let selector = NSSelectorFromString("setComputePipelineState:")
    guard let method = class_getInstanceMethod(cls, selector) else { return false }
    let original = unsafeBitCast(method_getImplementation(method), to: E191SetPipelineIMP.self)
    let replacement: @convention(block) (AnyObject, AnyObject) -> Void = { encoder, pipeline in
        E191Ledger.shared.bind(encoder: encoder, pipeline: pipeline)
        original(encoder, selector, pipeline)
    }
    method_setImplementation(method, imp_implementationWithBlock(replacement))
    return true
}

private func e191SwizzleNewPipeline(_ cls: AnyClass) -> Bool {
    let selector = NSSelectorFromString("newComputePipelineStateWithFunction:error:")
    guard let method = class_getInstanceMethod(cls, selector) else { return false }
    let original = unsafeBitCast(method_getImplementation(method), to: E191NewPipelineIMP.self)
    let replacement: @convention(block) (AnyObject, AnyObject, UnsafeMutableRawPointer?)
        -> UnsafeMutableRawPointer? = { device, function, errorOut in
            let result = original(device, selector, function, errorOut)
            if let result {
                let pipeline = Unmanaged<AnyObject>.fromOpaque(result).takeUnretainedValue()
                E191Ledger.shared.note(
                    pipeline: pipeline, name: (function as? MTLFunction)?.name ?? "<unnamed>")
            }
            return result
        }
    method_setImplementation(method, imp_implementationWithBlock(replacement))
    return true
}

// MARK: - The three attention forms at the scored geometry

private enum E191Form: String, CaseIterable {
    case today
    case twoAppend
    case unsplit
}

private enum E191Probe {
    static let heads = 24
    static let kvHeads = 4
    static let headDim = 256
    static let fullAttentionLayers = 16
    static let split = 5
    static var scale: Float { 1.0 / Float(headDim).squareRoot() }

    /// The layout the scored attention layer hands to `attentionWithCacheUpdate`:
    /// `[B, L, H, D]` from the projection, transposed to `[B, H, L, D]`.
    static func queries(width: Int) -> MLXArray {
        MLXRandom.normal([1, width, heads, headDim]).asType(.bfloat16)
            .transposed(0, 2, 1, 3)
    }

    static func newKV(width: Int) -> MLXArray {
        MLXRandom.normal([1, width, kvHeads, headDim]).asType(.bfloat16)
            .transposed(0, 2, 1, 3)
    }

    /// A `KVCacheSimple` already holding the given rows, with capacity for the
    /// widest probe width so no arm pays a reallocation inside a timed call.
    static func seededCache(keys: MLXArray, values: MLXArray) -> KVCacheSimple {
        let length = keys.dim(2)
        let cache = KVCacheSimple()
        _ = cache.update(keys: keys, values: values)
        // Force the growth that the first over-capacity append would otherwise
        // charge to whichever arm ran first.
        _ = cache.update(keys: newKV(width: 16), values: newKV(width: 16))
        cache.offset = length
        eval(cache.innerState())
        return cache
    }

    static func seededCache(length: Int) -> KVCacheSimple {
        seededCache(keys: newKV(width: length), values: newKV(width: length))
    }

    /// The shipped production form, reproduced exactly:
    /// `AttentionUtils.swift:103-141`.
    static func today(queries q: MLXArray, cache: KVCacheSimple, kv: MLXArray) -> MLXArray {
        let (cachedKeys, cachedValues) = cache.update(keys: kv, values: kv)
        let qL = q.dim(2)
        let kL = cachedKeys.dim(2)
        let kSplit = kL - (qL - split)
        let outA = MLXFast.scaledDotProductAttention(
            queries: q[0..., 0..., 0 ..< split, 0...],
            keys: cachedKeys[0..., 0..., 0 ..< kSplit, 0...],
            values: cachedValues[0..., 0..., 0 ..< kSplit, 0...],
            scale: scale, mask: .causal)
        let outB = MLXFast.scaledDotProductAttention(
            queries: q[0..., 0..., split..., 0...],
            keys: cachedKeys, values: cachedValues,
            scale: scale, mask: .causal)
        return concatenated([outA, outB], axis: 2)
    }

    /// `frontier-state.widthSixWall.candidateFix`: append the first window, run
    /// sdpa A over the returned arrays in full, append the remainder, run
    /// sdpa B over the returned arrays in full.
    static func twoAppend(queries q: MLXArray, cache: KVCacheSimple, kv: MLXArray) -> MLXArray {
        let qL = q.dim(2)
        let (keysA, valuesA) = cache.update(
            keys: kv[0..., 0..., 0 ..< split, 0...],
            values: kv[0..., 0..., 0 ..< split, 0...])
        let outA = MLXFast.scaledDotProductAttention(
            queries: q[0..., 0..., 0 ..< split, 0...],
            keys: keysA, values: valuesA, scale: scale, mask: .causal)
        let (keysB, valuesB) = cache.update(
            keys: kv[0..., 0..., split ..< qL, 0...],
            values: kv[0..., 0..., split ..< qL, 0...])
        let outB = MLXFast.scaledDotProductAttention(
            queries: q[0..., 0..., split..., 0...],
            keys: keysB, values: valuesB, scale: scale, mask: .causal)
        return concatenated([outA, outB], axis: 2)
    }

    /// E186 family 5: one direct call at width m. This is the form FINDING 489
    /// measured.
    static func unsplit(queries q: MLXArray, cache: KVCacheSimple, kv: MLXArray) -> MLXArray {
        let (cachedKeys, cachedValues) = cache.update(keys: kv, values: kv)
        return MLXFast.scaledDotProductAttention(
            queries: q, keys: cachedKeys, values: cachedValues,
            scale: scale, mask: q.dim(2) == 1 ? .none : .causal)
    }

    /// One call of `form` at width `m` against a cache holding `kvLength` rows.
    /// The cache offset is restored so repeated calls measure the same shape.
    static func call(
        _ form: E191Form, q: MLXArray, kv: MLXArray, cache: KVCacheSimple, kvLength: Int
    ) -> MLXArray {
        cache.offset = kvLength
        switch form {
        case .today: return today(queries: q, cache: cache, kv: kv)
        case .twoAppend: return twoAppend(queries: q, cache: cache, kv: kv)
        case .unsplit: return unsplit(queries: q, cache: cache, kv: kv)
        }
    }

    /// `today` and `twoAppend` must produce the same numbers; a split form is
    /// only interesting if it does. Reported as max |delta| over the output.
    static func maxAbsoluteDelta(_ a: MLXArray, _ b: MLXArray) -> Float {
        eval(a, b)
        return abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }
}

private func e191GpuTemperature() -> Double? {
    for path in [
        ProcessInfo.processInfo.environment["MLXFAST_MACMON_BIN"] ?? "",
        "\(FileManager.default.homeDirectoryForCurrentUser.path)/bin/macmon",
        "/opt/homebrew/bin/macmon", "/usr/local/bin/macmon",
    ] where !path.isEmpty && FileManager.default.isExecutableFile(atPath: path) {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: path)
        process.arguments = ["pipe", "-s1"]
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = FileHandle.nullDevice
        guard (try? process.run()) != nil else { continue }
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        guard
            let line = String(decoding: data, as: UTF8.self)
                .split(separator: "\n").first,
            let object = try? JSONSerialization.jsonObject(
                with: Data(line.utf8)) as? [String: Any],
            let temp = object["temp"] as? [String: Any],
            let gpu = temp["gpu_temp_avg"] as? Double
        else { continue }
        return gpu
    }
    return nil
}

private func e191WriteReport(_ report: [String: Any], envKey: String) throws {
    let json = try JSONSerialization.data(
        withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
    print(String(decoding: json, as: UTF8.self))
    if let path = ProcessInfo.processInfo.environment[envKey], !path.isEmpty {
        try json.write(to: URL(fileURLWithPath: path))
    }
}

// MARK: - Section 1: kernel dispatch census

@Suite(.serialized)
struct E191DispatchCensusTests {
    @Test(
        .enabled(
            if: ProcessInfo.processInfo.environment["MLXFAST_E191_DISPATCH"] == "1",
            "set MLXFAST_E191_DISPATCH=1 to run the GPU dispatch census"))
    func countsEveryKernelPerForm() throws {
        guard let device = MTLCreateSystemDefaultDevice(),
            let queue = device.makeCommandQueue(),
            let buffer = queue.makeCommandBuffer(),
            let encoder = buffer.makeComputeCommandEncoder()
        else {
            Issue.record("no Metal compute encoder to swizzle")
            return
        }
        let encoderClass: AnyClass = type(of: encoder as AnyObject)
        let deviceClass: AnyClass = type(of: device as AnyObject)
        encoder.endEncoding()

        #expect(e191SwizzleNewPipeline(deviceClass))
        #expect(e191SwizzleSetPipeline(encoderClass))
        #expect(e191SwizzleDispatch(encoderClass, "dispatchThreadgroups:threadsPerThreadgroup:"))
        #expect(e191SwizzleDispatch(encoderClass, "dispatchThreads:threadsPerThreadgroup:"))

        var cells: [[String: Any]] = []
        for kvLength in [512, 1024] {
            for width in [5, 6, 7, 8, 9] {
                let q = E191Probe.queries(width: width)
                let kv = E191Probe.newKV(width: width)
                eval(q, kv)
                for form in E191Form.allCases {
                    if form != .unsplit, width < 6 { continue }
                    let cache = E191Probe.seededCache(length: kvLength)
                    // Warm the kernels and the pipeline map before counting so a
                    // first-use JIT compile cannot appear as a dispatch.
                    for _ in 0 ..< 3 {
                        eval(
                            E191Probe.call(
                                form, q: q, kv: kv, cache: cache, kvLength: kvLength))
                    }
                    E191Ledger.shared.start()
                    eval(E191Probe.call(form, q: q, kv: kv, cache: cache, kvLength: kvLength))
                    let records = E191Ledger.shared.stop()

                    var counts: [String: Int] = [:]
                    for record in records { counts[record.kernel, default: 0] += 1 }
                    let copies = records.filter { $0.kernel.lowercased().contains("copy") }
                    cells.append([
                        "form": form.rawValue, "qL": width, "kv": kvLength,
                        "dispatches": records.count,
                        "kernel_counts": counts,
                        "kernel_sequence": records.map(\.kernel),
                        "copy_dispatches": copies.count,
                        "copy_grids": copies.map {
                            "\($0.kernel) grid=(\($0.grid.width),\($0.grid.height),"
                                + "\($0.grid.depth))"
                        },
                    ])
                }
            }
        }

        try e191WriteReport(
            [
                "probe": "e191-dispatch-census",
                "host_architecture": device.architecture.name,
                "full_attention_layers": E191Probe.fullAttentionLayers,
                "gqa_factor": E191Probe.heads / E191Probe.kvHeads,
                "cells": cells,
            ], envKey: "MLXFAST_E191_DISPATCH_OUT")
    }
}

// MARK: - Section 2: ABBA-counterbalanced absolute time

@Suite(.serialized)
struct E191TwoAppendTimingTests {
    @Test(
        .enabled(
            if: ProcessInfo.processInfo.environment["MLXFAST_E191_TIMING"] == "1",
            "set MLXFAST_E191_TIMING=1 to run the GPU timing probe"))
    func pricesTheSplitAndTheTwoAppendFix() throws {
        let environment = ProcessInfo.processInfo.environment
        let blocks = Int(environment["MLXFAST_E191_BLOCKS"] ?? "") ?? 8
        let reps = Int(environment["MLXFAST_E191_REPS"] ?? "") ?? 40
        let warmup = Int(environment["MLXFAST_E191_WARMUP"] ?? "") ?? 20
        let widths = [5, 6, 7, 8, 9]
        let kvLengths = [512, 1024]

        var temperatures: [String: Double] = [:]
        func recordTemperature(_ label: String) {
            if let gpu = e191GpuTemperature() { temperatures[label] = gpu }
        }

        struct Cell {
            var form: E191Form
            var width: Int
            var kvLength: Int
            var q: MLXArray
            var kv: MLXArray
            var cache: KVCacheSimple
        }

        var cells: [Cell] = []
        for kvLength in kvLengths {
            for width in widths {
                let q = E191Probe.queries(width: width)
                let kv = E191Probe.newKV(width: width)
                eval(q, kv)
                for form in E191Form.allCases {
                    // Below the wall there is nothing to split; `today` and
                    // `twoAppend` are only defined at qL >= 6. Width 5 is the
                    // control that prices one fused vector call.
                    if form != .unsplit, width < 6 { continue }
                    cells.append(
                        Cell(
                            form: form, width: width, kvLength: kvLength, q: q, kv: kv,
                            cache: E191Probe.seededCache(length: kvLength)))
                }
            }
        }

        // Agreement gate: the two split forms must produce the same output
        // before either timing is worth reading.
        var agreement: [[String: Any]] = []
        for kvLength in kvLengths {
            for width in [6, 7, 8, 9] {
                let q = E191Probe.queries(width: width)
                let kv = E191Probe.newKV(width: width)
                eval(q, kv)
                let seedKeys = E191Probe.newKV(width: kvLength)
                let seedValues = E191Probe.newKV(width: kvLength)
                eval(seedKeys, seedValues)
                let cacheA = E191Probe.seededCache(keys: seedKeys, values: seedValues)
                let cacheB = E191Probe.seededCache(keys: seedKeys, values: seedValues)
                let a = E191Probe.call(
                    .today, q: q, kv: kv, cache: cacheA, kvLength: kvLength)
                let b = E191Probe.call(
                    .twoAppend, q: q, kv: kv, cache: cacheB, kvLength: kvLength)
                agreement.append([
                    "qL": width, "kv": kvLength,
                    "max_abs_delta": Double(E191Probe.maxAbsoluteDelta(a, b)),
                    "today_offset_after": cacheA.offset,
                    "two_append_offset_after": cacheB.offset,
                ])
            }
        }

        // Settle work that re-establishes GPU clocks after each temperature read.
        let settleWeights = MLXRandom.normal([2048, 2048]).asType(.bfloat16)
        let settleInput = MLXRandom.normal([64, 2048]).asType(.bfloat16)
        eval(settleWeights, settleInput)
        let settle = { eval(matmul(settleInput, settleWeights)) }

        recordTemperature("session_entry")
        for _ in 0 ..< 40 { settle() }
        for cell in cells {
            for _ in 0 ..< warmup {
                eval(
                    E191Probe.call(
                        cell.form, q: cell.q, kv: cell.kv, cache: cell.cache,
                        kvLength: cell.kvLength))
            }
        }
        recordTemperature("after_warmup")

        var samples: [[String: Any]] = []
        for block in 0 ..< blocks {
            let ascending = block % 2 == 0
            let order = ascending
                ? Array(cells.indices) : Array(cells.indices.reversed())
            recordTemperature("block_\(block)_entry")
            for _ in 0 ..< 20 { settle() }
            for (position, index) in order.enumerated() {
                let cell = cells[index]
                let start = DispatchTime.now().uptimeNanoseconds
                for _ in 0 ..< reps {
                    eval(
                        E191Probe.call(
                            cell.form, q: cell.q, kv: cell.kv, cache: cell.cache,
                            kvLength: cell.kvLength))
                }
                let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start)
                samples.append([
                    "form": cell.form.rawValue, "qL": cell.width, "kv": cell.kvLength,
                    "block": block, "ascending": ascending, "position": position,
                    "microseconds": elapsed / 1e3 / Double(reps), "reps": reps,
                ])
            }
        }
        recordTemperature("session_exit")

        try e191WriteReport(
            [
                "probe": "e191-two-append-timing",
                "harness": "local",
                "cool_gate_passed_real_gate": false,
                "gate_qualified_for_timing": false,
                "abba_counterbalanced": true,
                "blocks": blocks, "reps": reps, "warmup": warmup,
                "full_attention_layers": E191Probe.fullAttentionLayers,
                "gpu_temperature_c": temperatures,
                "agreement": agreement,
                "samples": samples,
            ], envKey: "MLXFAST_E191_TIMING_OUT")
    }
}
