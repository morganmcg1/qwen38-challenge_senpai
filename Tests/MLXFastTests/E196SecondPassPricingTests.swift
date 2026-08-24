import Foundation
import MLX
import MLXLMCommon
import MLXRandom
import Metal
import ObjectiveC
import Testing

// E196 -- price the SECOND SDPA call of the qL 6..9 exactness split directly,
// and decide Route 1 (a fused one-pass form) go/no-go.
//
// FINDING 497 priced the production m5 -> m step at +1.58..+3.04 ms/round.
// FINDING 502 showed that dispatch and barrier surgery recovers <= ~0.15
// ms/round, so the residual must sit in the second SDPA call's own passes.
// This probe measures the vector-family SDPA kernel cost as a function of
// (query rows, KV length) so the split can be decomposed into
//
//     T(rows R, keys N) = a(N) + b(N) * R
//
// where `a` is the per-CALL fixed cost and `b` the per-ROW marginal cost.
// Under that model the split pays
//
//     T(5, kL-(m-5)) + T(m-5, kL) - T_fused(m, kL)
//
// and a Route-1 fused kernel with the SAME per-row efficiency recovers
// exactly `a`, while a Route-1 kernel that amortises every KV read across
// rows (b' -> 0) recovers at most `a + b*m`. Those two numbers bracket the
// whole Route-1 prize, and both are measurable from the R = 1..5 ladder that
// the existing MLX dispatch already serves.
//
// Two instruments, because FINDING 497's probe charged one blocking `eval()`
// (~250 us) to every measured call and in-path decode has no such barrier:
//
//   chain slope   K SDPA calls behind ONE eval, K in {1,2,4,8,16}. The
//                 regression slope is the marginal cost of one more call in a
//                 PIPELINED regime, which is what in-path decode pays; the
//                 intercept absorbs the eval barrier.
//   per-eval      K = 1, i.e. FINDING 497's own instrument, kept as the
//                 bridge that reproduces the published step.
//
// Two chain modes, because independent calls may overlap on the GPU:
//   indep   distinct query tensors, calls independent (throughput view)
//   serial  each call's OUTPUT is the next call's query, so the chain is a
//           true dependency chain (latency view). SDPA returns
//           [B, H, L, D] -- exactly the query shape -- so this needs no
//           reshaping and no extra dispatch.
//
// Research instrument. `Tests/` is never packaged into a submission.
// harness=local, within-session relative measurement, no thermal gate,
// no score.

/// Live-path certification needle (RULE 384/387): a dedicated >= 16-byte
/// literal on the executed path of this probe, not on any failure path and
/// not shared with a `require`/`precondition` message.
let e196LivePathNeedle = "E196-SECOND-SDPA-PASS-PRICING-LIVE-PATH-2026-08-24"

// MARK: - Metal dispatch ledger (same technique as E57/E191, file-private)

private struct E196DispatchRecord {
    var kernel: String
    var grid: MTLSize
    var threadgroup: MTLSize
}

/// `@unchecked Sendable` because every mutable field is reached only under
/// `lock`, and the swizzled Metal selectors can be called from any thread.
private final class E196Ledger: @unchecked Sendable {
    static let shared = E196Ledger()

    private let lock = NSLock()
    private var pipelineNames: [ObjectIdentifier: String] = [:]
    private var encoderBinding: [ObjectIdentifier: String] = [:]
    private var records: [E196DispatchRecord] = []
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
                E196DispatchRecord(
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

    func stop() -> [E196DispatchRecord] {
        lock.lock()
        recording = false
        let taken = records
        records = []
        lock.unlock()
        return taken
    }
}

private typealias E196DispatchIMP = @convention(c) (AnyObject, Selector, MTLSize, MTLSize)
    -> Void
private typealias E196SetPipelineIMP = @convention(c) (AnyObject, Selector, AnyObject) -> Void
private typealias E196NewPipelineIMP = @convention(c) (
    AnyObject, Selector, AnyObject, UnsafeMutableRawPointer?
) -> UnsafeMutableRawPointer?

private func e196SwizzleDispatch(_ cls: AnyClass, _ name: String) -> Bool {
    let selector = NSSelectorFromString(name)
    guard let method = class_getInstanceMethod(cls, selector) else { return false }
    let original = unsafeBitCast(method_getImplementation(method), to: E196DispatchIMP.self)
    let replacement: @convention(block) (AnyObject, MTLSize, MTLSize) -> Void = {
        encoder, grid, threadgroup in
        E196Ledger.shared.dispatch(encoder: encoder, grid: grid, threadgroup: threadgroup)
        original(encoder, selector, grid, threadgroup)
    }
    method_setImplementation(method, imp_implementationWithBlock(replacement))
    return true
}

private func e196SwizzleSetPipeline(_ cls: AnyClass) -> Bool {
    let selector = NSSelectorFromString("setComputePipelineState:")
    guard let method = class_getInstanceMethod(cls, selector) else { return false }
    let original = unsafeBitCast(method_getImplementation(method), to: E196SetPipelineIMP.self)
    let replacement: @convention(block) (AnyObject, AnyObject) -> Void = { encoder, pipeline in
        E196Ledger.shared.bind(encoder: encoder, pipeline: pipeline)
        original(encoder, selector, pipeline)
    }
    method_setImplementation(method, imp_implementationWithBlock(replacement))
    return true
}

private func e196SwizzleNewPipeline(_ cls: AnyClass) -> Bool {
    let selector = NSSelectorFromString("newComputePipelineStateWithFunction:error:")
    guard let method = class_getInstanceMethod(cls, selector) else { return false }
    let original = unsafeBitCast(method_getImplementation(method), to: E196NewPipelineIMP.self)
    let replacement: @convention(block) (AnyObject, AnyObject, UnsafeMutableRawPointer?)
        -> UnsafeMutableRawPointer? = { device, function, errorOut in
            let result = original(device, selector, function, errorOut)
            if let result {
                let pipeline = Unmanaged<AnyObject>.fromOpaque(result).takeUnretainedValue()
                E196Ledger.shared.note(
                    pipeline: pipeline, name: (function as? MTLFunction)?.name ?? "<unnamed>")
            }
            return result
        }
    method_setImplementation(method, imp_implementationWithBlock(replacement))
    return true
}

// MARK: - The scored geometry

private enum E196Probe {
    static let heads = 24
    static let kvHeads = 4
    static let headDim = 256
    static let fullAttentionLayers = 16
    static let split = 5
    static var gqa: Int { heads / kvHeads }
    static var scale: Float { 1.0 / Float(headDim).squareRoot() }

    /// Row-contiguous `[1, H, L, D]`, the layout the fused
    /// `qwen35AttentionQKRMSRoPE` path hands to `attentionWithCacheUpdate`
    /// and the layout MLX's own query copy produces before the kernel runs.
    static func rows(_ length: Int, heads count: Int) -> MLXArray {
        MLXRandom.normal([1, count, length, headDim]).asType(.bfloat16)
    }

    static func queries(rows length: Int) -> MLXArray { rows(length, heads: heads) }
    static func newKV(rows length: Int) -> MLXArray { rows(length, heads: kvHeads) }

    /// A `KVCacheSimple` holding `length` rows with headroom for the widest
    /// probe width, so no timed call pays a reallocation and every timed
    /// slice carries the production strides (`k_head_stride` is the CAPACITY
    /// stride, not the length stride).
    static func seededCache(length: Int) -> KVCacheSimple {
        let cache = KVCacheSimple()
        _ = cache.update(keys: newKV(rows: length), values: newKV(rows: length))
        _ = cache.update(keys: newKV(rows: 16), values: newKV(rows: 16))
        cache.offset = length
        eval(cache.innerState())
        return cache
    }

    /// The committed window a round of `m` rows produces: one `cache.update`
    /// of `m` rows on a cache already holding `kvLength`. Returned arrays are
    /// axis-2 views of the cache buffer, which is exactly what
    /// `AttentionUtils.swift:103` passes to the two split calls.
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
}

private func e196GpuTemperature() -> Double? {
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

private func e196WriteReport(_ report: [String: Any], envKey: String) throws {
    var stamped = report
    stamped["probe_needle"] = e196LivePathNeedle
    let json = try JSONSerialization.data(
        withJSONObject: stamped, options: [.prettyPrinted, .sortedKeys])
    print(String(decoding: json, as: UTF8.self))
    if let path = ProcessInfo.processInfo.environment[envKey], !path.isEmpty {
        try json.write(to: URL(fileURLWithPath: path))
    }
}

private func e196IntList(_ key: String, _ fallback: [Int]) -> [Int] {
    guard let raw = ProcessInfo.processInfo.environment[key], !raw.isEmpty else {
        return fallback
    }
    return raw.split(separator: ",").compactMap { Int($0) }
}

// MARK: - Section 1: per-call kernel identity census

@Suite(.serialized)
struct E196KernelIdentityCensusTests {
    @Test(
        .enabled(
            if: ProcessInfo.processInfo.environment["MLX_E196_CENSUS"] == "1",
            "set MLX_E196_CENSUS=1 to run the per-call kernel identity census"))
    func namesTheKernelsOfCallAAndCallB() throws {
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

        #expect(e196SwizzleNewPipeline(deviceClass))
        #expect(e196SwizzleSetPipeline(encoderClass))
        #expect(e196SwizzleDispatch(encoderClass, "dispatchThreadgroups:threadsPerThreadgroup:"))
        #expect(e196SwizzleDispatch(encoderClass, "dispatchThreads:threadsPerThreadgroup:"))

        let kvLengths = e196IntList("MLX_E196_KV", [512, 768, 1024, 2048])
        let widths = e196IntList("MLX_E196_M", [6, 7, 8, 9])

        func census(_ label: String, _ body: () -> MLXArray) -> [String: Any] {
            for _ in 0 ..< 3 { eval(body()) }
            E196Ledger.shared.start()
            eval(body())
            let records = E196Ledger.shared.stop()
            var counts: [String: Int] = [:]
            for record in records { counts[record.kernel, default: 0] += 1 }
            return [
                "call": label,
                "dispatches": records.count,
                "kernel_counts": counts,
                "kernel_sequence": records.map(\.kernel),
                "grids": records.map {
                    "\($0.kernel) grid=(\($0.grid.width),\($0.grid.height),\($0.grid.depth))"
                        + " tg=(\($0.threadgroup.width),\($0.threadgroup.height),"
                        + "\($0.threadgroup.depth))"
                },
            ]
        }

        var cells: [[String: Any]] = []
        for kvLength in kvLengths {
            for m in widths {
                let cache = E196Probe.seededCache(length: kvLength)
                let (keys, values) = E196Probe.committedWindow(
                    cache: cache, kvLength: kvLength, m: m)
                let kL = keys.dim(2)
                let kSplit = kL - (m - E196Probe.split)
                let qFull = E196Probe.queries(rows: m)
                let qA = E196Probe.queries(rows: E196Probe.split)
                let qB = E196Probe.queries(rows: m - E196Probe.split)
                eval(qFull, qA, qB)

                let keysA = keys[0..., 0..., 0 ..< kSplit, 0...]
                let valuesA = values[0..., 0..., 0 ..< kSplit, 0...]

                var calls: [[String: Any]] = []
                // Call A and call B exactly as the shipped split issues them,
                // each evaluated alone so its kernels are attributable.
                calls.append(
                    census("A_shipped_slices") {
                        MLXFast.scaledDotProductAttention(
                            queries: qFull[0..., 0..., 0 ..< E196Probe.split, 0...],
                            keys: keysA, values: valuesA,
                            scale: E196Probe.scale, mask: .causal)
                    })
                calls.append(
                    census("B_shipped_slices") {
                        MLXFast.scaledDotProductAttention(
                            queries: qFull[0..., 0..., E196Probe.split..., 0...],
                            keys: keys, values: values,
                            scale: E196Probe.scale, mask: .causal)
                    })
                // The same two calls with a pre-contiguous query, which is
                // what the timing arms drive: the kernel work without MLX's
                // query copy.
                calls.append(
                    census("A_isolated") {
                        E196Probe.sdpa(qA, keysA, valuesA)
                    })
                calls.append(
                    census("B_isolated") {
                        E196Probe.sdpa(qB, keys, values)
                    })
                calls.append(
                    census("today_full_form") {
                        E196Probe.today(
                            queries: qFull, cache: E196Probe.seededCache(length: kvLength),
                            kv: E196Probe.newKV(rows: m))
                    })

                cells.append([
                    "kv": kvLength, "m": m, "kL": kL, "kSplit": kSplit,
                    "gqa": E196Probe.gqa,
                    "n_simds_A": E196Probe.gqa * E196Probe.split,
                    "n_simds_B": E196Probe.gqa * (m - E196Probe.split),
                    "calls": calls,
                ])
            }
        }

        // The R = 1..5 ladder the pricing model is fitted on, so its kernel
        // identity is on the record next to the identity of A and B.
        var ladder: [[String: Any]] = []
        for kvLength in kvLengths {
            let cache = E196Probe.seededCache(length: kvLength)
            let (keys, values) = E196Probe.committedWindow(
                cache: cache, kvLength: kvLength, m: 5)
            for rowCount in 1 ... 5 {
                let q = E196Probe.queries(rows: rowCount)
                eval(q)
                var record = census("ladder_R\(rowCount)") {
                    E196Probe.sdpa(q, keys, values)
                }
                record["kv"] = kvLength
                record["R"] = rowCount
                record["kL"] = keys.dim(2)
                ladder.append(record)
            }
        }

        try e196WriteReport(
            [
                "probe": "e196-kernel-identity-census",
                "host_architecture": device.architecture.name,
                "full_attention_layers": E196Probe.fullAttentionLayers,
                "gqa_factor": E196Probe.gqa,
                "split_row": E196Probe.split,
                "cells": cells,
                "ladder": ladder,
            ], envKey: "MLX_E196_CENSUS_OUT")
    }
}

// MARK: - Section 2: chain-slope pricing of the vector SDPA family

private enum E196Mode: String { case indep, serial }

@Suite(.serialized)
struct E196ChainSlopePricingTests {
    @Test(
        .enabled(
            if: ProcessInfo.processInfo.environment["MLX_E196_TIMING"] == "1",
            "set MLX_E196_TIMING=1 to run the chain-slope pricing probe"))
    func pricesTheSecondSdpaCall() throws {
        let environment = ProcessInfo.processInfo.environment
        let blocks = Int(environment["MLX_E196_BLOCKS"] ?? "") ?? 6
        let reps = Int(environment["MLX_E196_REPS"] ?? "") ?? 20
        let warmup = Int(environment["MLX_E196_WARMUP"] ?? "") ?? 6
        let kvLengths = e196IntList("MLX_E196_KV", [512, 768, 1024, 2048])
        let widths = e196IntList("MLX_E196_M", [6, 7, 8, 9])
        let ladderChains = e196IntList("MLX_E196_CHAINS", [1, 2, 4, 8, 16])
        let armChains = e196IntList("MLX_E196_ARM_CHAINS", [1, 8])

        var temperatures: [String: Double] = [:]
        func recordTemperature(_ label: String) {
            if let gpu = e196GpuTemperature() { temperatures[label] = gpu }
        }

        /// One timed cell: `chain` SDPA calls behind a single `eval`.
        struct Cell {
            var arm: String
            var mode: E196Mode
            var chain: Int
            var kv: Int
            var m: Int  // 0 for the ladder
            var rows: Int
            var keys: Int
            var run: () -> Void
        }

        var cells: [Cell] = []

        /// `indep`: distinct query tensors, so MLX cannot share a result and
        /// the calls stay independent. `serial`: the output of each call is
        /// the query of the next, which is a true dependency chain.
        func makeRunner(
            mode: E196Mode, chain: Int, queryPool: [MLXArray], keys: MLXArray, values: MLXArray
        ) -> () -> Void {
            switch mode {
            case .indep:
                return {
                    var outputs: [MLXArray] = []
                    outputs.reserveCapacity(chain)
                    for index in 0 ..< chain {
                        outputs.append(
                            E196Probe.sdpa(queryPool[index % queryPool.count], keys, values))
                    }
                    eval(outputs)
                }
            case .serial:
                return {
                    var current = queryPool[0]
                    for _ in 0 ..< chain {
                        current = E196Probe.sdpa(current, keys, values)
                    }
                    eval(current)
                }
            }
        }

        /// A pair runner: call A then call B in the same chain step, which is
        /// what one FA layer of a split round issues.
        func makePairRunner(
            mode: E196Mode, chain: Int, queryA: [MLXArray], queryB: [MLXArray],
            keysA: MLXArray, valuesA: MLXArray, keys: MLXArray, values: MLXArray
        ) -> () -> Void {
            switch mode {
            case .indep:
                return {
                    var outputs: [MLXArray] = []
                    outputs.reserveCapacity(chain * 2)
                    for index in 0 ..< chain {
                        outputs.append(
                            E196Probe.sdpa(queryA[index % queryA.count], keysA, valuesA))
                        outputs.append(
                            E196Probe.sdpa(queryB[index % queryB.count], keys, values))
                    }
                    eval(outputs)
                }
            case .serial:
                return {
                    var a = queryA[0]
                    var b = queryB[0]
                    for _ in 0 ..< chain {
                        a = E196Probe.sdpa(a, keysA, valuesA)
                        b = E196Probe.sdpa(b, keys, values)
                    }
                    eval(a, b)
                }
            }
        }

        let queryPoolSize = 4

        // Ladder cells: R = 1..5 at the m = 5 production window, the fit that
        // yields a(N) and b(N).
        for kvLength in kvLengths {
            let cache = E196Probe.seededCache(length: kvLength)
            let (keys, values) = E196Probe.committedWindow(
                cache: cache, kvLength: kvLength, m: 5)
            for rowCount in 1 ... 5 {
                let pool = (0 ..< queryPoolSize).map { _ in E196Probe.queries(rows: rowCount) }
                eval(pool)
                for mode in [E196Mode.indep, .serial] {
                    for chain in ladderChains {
                        cells.append(
                            Cell(
                                arm: "ladder", mode: mode, chain: chain, kv: kvLength, m: 0,
                                rows: rowCount, keys: keys.dim(2),
                                run: makeRunner(
                                    mode: mode, chain: chain, queryPool: pool, keys: keys,
                                    values: values)))
                    }
                }
            }
        }

        // Split-arm cells: call A, call B and the pair at the exact shapes the
        // shipped split issues.
        for kvLength in kvLengths {
            for m in widths {
                let cache = E196Probe.seededCache(length: kvLength)
                let (keys, values) = E196Probe.committedWindow(
                    cache: cache, kvLength: kvLength, m: m)
                let kL = keys.dim(2)
                let kSplit = kL - (m - E196Probe.split)
                let keysA = keys[0..., 0..., 0 ..< kSplit, 0...]
                let valuesA = values[0..., 0..., 0 ..< kSplit, 0...]
                let poolA = (0 ..< queryPoolSize).map { _ in
                    E196Probe.queries(rows: E196Probe.split)
                }
                let poolB = (0 ..< queryPoolSize).map { _ in
                    E196Probe.queries(rows: m - E196Probe.split)
                }
                eval(poolA)
                eval(poolB)
                for mode in [E196Mode.indep, .serial] {
                    for chain in armChains {
                        cells.append(
                            Cell(
                                arm: "callA", mode: mode, chain: chain, kv: kvLength, m: m,
                                rows: E196Probe.split, keys: kSplit,
                                run: makeRunner(
                                    mode: mode, chain: chain, queryPool: poolA, keys: keysA,
                                    values: valuesA)))
                        cells.append(
                            Cell(
                                arm: "callB", mode: mode, chain: chain, kv: kvLength, m: m,
                                rows: m - E196Probe.split, keys: kL,
                                run: makeRunner(
                                    mode: mode, chain: chain, queryPool: poolB, keys: keys,
                                    values: values)))
                        cells.append(
                            Cell(
                                arm: "pair", mode: mode, chain: chain, kv: kvLength, m: m,
                                rows: m, keys: kL,
                                run: makePairRunner(
                                    mode: mode, chain: chain, queryA: poolA, queryB: poolB,
                                    keysA: keysA, valuesA: valuesA, keys: keys, values: values)))
                    }
                }
            }
        }

        // Bridge cells: the whole shipped `today` form and the m = 5
        // production control, one call per eval -- FINDING 497's instrument,
        // so this probe can be checked against the published step.
        for kvLength in kvLengths {
            let cache5 = E196Probe.seededCache(length: kvLength)
            let q5 = E196Probe.queries(rows: 5)
            let kv5 = E196Probe.newKV(rows: 5)
            eval(q5, kv5)
            cells.append(
                Cell(
                    arm: "today", mode: .indep, chain: 1, kv: kvLength, m: 5, rows: 5,
                    keys: kvLength + 5,
                    run: {
                        cache5.offset = kvLength
                        let (keys, values) = cache5.update(keys: kv5, values: kv5)
                        eval(
                            MLXFast.scaledDotProductAttention(
                                queries: q5, keys: keys, values: values,
                                scale: E196Probe.scale, mask: .causal))
                    }))
            for m in widths {
                let cache = E196Probe.seededCache(length: kvLength)
                let q = E196Probe.queries(rows: m)
                let kv = E196Probe.newKV(rows: m)
                eval(q, kv)
                cells.append(
                    Cell(
                        arm: "today", mode: .indep, chain: 1, kv: kvLength, m: m, rows: m,
                        keys: kvLength + m,
                        run: {
                            cache.offset = kvLength
                            eval(E196Probe.today(queries: q, cache: cache, kv: kv))
                        }))
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
            for _ in 0 ..< warmup { cell.run() }
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
                for _ in 0 ..< reps { cell.run() }
                let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start)
                samples.append([
                    "arm": cell.arm, "mode": cell.mode.rawValue, "chain": cell.chain,
                    "kv": cell.kv, "m": cell.m, "rows": cell.rows, "keys": cell.keys,
                    "block": block, "ascending": ascending, "position": position,
                    "microseconds": elapsed / 1e3 / Double(reps), "reps": reps,
                ])
            }
        }
        recordTemperature("session_exit")

        try e196WriteReport(
            [
                "probe": "e196-chain-slope-pricing",
                "harness": "local",
                "cool_gate_passed_real_gate": false,
                "gate_qualified_for_timing": false,
                "abba_counterbalanced": true,
                "blocks": blocks, "reps": reps, "warmup": warmup,
                "query_pool_size": queryPoolSize,
                "full_attention_layers": E196Probe.fullAttentionLayers,
                "gqa_factor": E196Probe.gqa,
                "split_row": E196Probe.split,
                "gpu_temperature_c": temperatures,
                "samples": samples,
            ], envKey: "MLX_E196_TIMING_OUT")
    }
}
