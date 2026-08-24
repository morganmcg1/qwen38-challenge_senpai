import Foundation
import MLX
import MLXLMCommon
import MLXRandom
import Metal
import ObjectiveC
import Testing

// E194 Stage 1 -- can the m >= 6 SDPA split step lose its two per-chunk query
// copies AND the memory barrier between its two SDPA calls, with no change to
// any per-row accumulation?
//
// Route 2' materializes the queries ONCE in sequence-major layout and takes
// both chunk views from that single buffer:
//
//   today     the shipped production form (AttentionUtils.swift:106-142):
//             two axis-2 query slices of a row-contiguous [1, H, qL, D]
//             buffer. Each slice fails `q_copy_unless`
//             (scaled_dot_product_attention.cpp:686-699), so MLX copies it
//             immediately before its own SDPA call. That copy is a
//             read-after-write dependency, so `maybeInsertBarrier`
//             (device.cpp:315-375) separates the two SDPA dispatches.
//   seqMajor  Route 2': one `transposed(0,2,1,3).contiguous()`, then both
//             chunk views. The views satisfy `q_copy_unless`'s second branch
//             (strides[3]==1, strides[2]==shape[3]*shape[1],
//             strides[1]==shape[3]), so no per-chunk copy is issued and the
//             two SDPA dispatches share one already-written query buffer.
//
// Two sections, each opt-in:
//
//   MLX_E194_CENSUS=1  counts the kernels AND the memory barriers Metal
//                      actually receives per form. The barrier slot is the
//                      one unmeasured part of FINDING 497's mechanism. The
//                      named failure mode is allocator recycling:
//                      `set_output_array` also runs the `prev_outputs_` check
//                      on the OUTPUT buffer (device.cpp:329-334), so a
//                      recycled output buffer can force a barrier back. This
//                      section measures that instead of assuming it away.
//   MLX_E194_GATE=1    the bf16 value gate: full elementwise diff of the two
//                      forms at m = 6,7,8,9 and kv 512/1024, plus a positive
//                      control that proves the comparison can fail.
//
// Research instrument. `Tests/` is never packaged into a submission.
// harness=local, no thermal gate, no score.

// MARK: - Metal dispatch and barrier ledger (E191 technique, plus barriers)

private struct E194DispatchRecord {
    var kernel: String
    var grid: MTLSize
    var threadgroup: MTLSize
}

/// One ordered event: either a dispatch or a memory barrier. The ORDER is the
/// evidence -- a barrier between the two SDPA dispatches is what Route 2' must
/// remove.
private enum E194Event {
    case dispatch(E194DispatchRecord)
    case barrier(String)
}

/// `@unchecked Sendable` because every mutable field is reached only under
/// `lock`, and the swizzled Metal selectors can be called from any thread.
private final class E194Ledger: @unchecked Sendable {
    static let shared = E194Ledger()

    private let lock = NSLock()
    private var pipelineNames: [ObjectIdentifier: String] = [:]
    private var encoderBinding: [ObjectIdentifier: String] = [:]
    private var events: [E194Event] = []
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
            events.append(
                .dispatch(
                    E194DispatchRecord(
                        kernel: encoderBinding[ObjectIdentifier(encoder)] ?? "<unbound>",
                        grid: grid, threadgroup: threadgroup)))
        }
        lock.unlock()
    }

    func barrier(_ kind: String) {
        lock.lock()
        if recording { events.append(.barrier(kind)) }
        lock.unlock()
    }

    func start() {
        lock.lock()
        events = []
        recording = true
        lock.unlock()
    }

    func stop() -> [E194Event] {
        lock.lock()
        recording = false
        let taken = events
        events = []
        lock.unlock()
        return taken
    }
}

private typealias E194DispatchIMP = @convention(c) (AnyObject, Selector, MTLSize, MTLSize)
    -> Void
private typealias E194ScopeBarrierIMP = @convention(c) (AnyObject, Selector, UInt) -> Void
private typealias E194ResourceBarrierIMP = @convention(c) (
    AnyObject, Selector, UnsafeRawPointer?, Int
) -> Void
private typealias E194SetPipelineIMP = @convention(c) (AnyObject, Selector, AnyObject) -> Void
private typealias E194NewPipelineIMP = @convention(c) (
    AnyObject, Selector, AnyObject, UnsafeMutableRawPointer?
) -> UnsafeMutableRawPointer?

private func e194SwizzleDispatch(_ cls: AnyClass, _ name: String) -> Bool {
    let selector = NSSelectorFromString(name)
    guard let method = class_getInstanceMethod(cls, selector) else { return false }
    let original = unsafeBitCast(method_getImplementation(method), to: E194DispatchIMP.self)
    let replacement: @convention(block) (AnyObject, MTLSize, MTLSize) -> Void = {
        encoder, grid, threadgroup in
        E194Ledger.shared.dispatch(encoder: encoder, grid: grid, threadgroup: threadgroup)
        original(encoder, selector, grid, threadgroup)
    }
    method_setImplementation(method, imp_implementationWithBlock(replacement))
    return true
}

/// `memoryBarrierWithScope:` is the selector MLX reaches through
/// `CommandEncoder::maybeInsertBarrier` -> `memoryBarrier(BarrierScopeBuffers)`.
private func e194SwizzleScopeBarrier(_ cls: AnyClass) -> Bool {
    let selector = NSSelectorFromString("memoryBarrierWithScope:")
    guard let method = class_getInstanceMethod(cls, selector) else { return false }
    let original = unsafeBitCast(method_getImplementation(method), to: E194ScopeBarrierIMP.self)
    let replacement: @convention(block) (AnyObject, UInt) -> Void = { encoder, scope in
        E194Ledger.shared.barrier("scope:\(scope)")
        original(encoder, selector, scope)
    }
    method_setImplementation(method, imp_implementationWithBlock(replacement))
    return true
}

/// The resource-list form is not on MLX's Metal path today. It is recorded so
/// a future MLX change cannot make a barrier invisible to this census.
private func e194SwizzleResourceBarrier(_ cls: AnyClass) -> Bool {
    let selector = NSSelectorFromString("memoryBarrierWithResources:count:")
    guard let method = class_getInstanceMethod(cls, selector) else { return false }
    let original = unsafeBitCast(
        method_getImplementation(method), to: E194ResourceBarrierIMP.self)
    let replacement: @convention(block) (AnyObject, UnsafeRawPointer?, Int) -> Void = {
        encoder, resources, count in
        E194Ledger.shared.barrier("resources:\(count)")
        original(encoder, selector, resources, count)
    }
    method_setImplementation(method, imp_implementationWithBlock(replacement))
    return true
}

private func e194SwizzleSetPipeline(_ cls: AnyClass) -> Bool {
    let selector = NSSelectorFromString("setComputePipelineState:")
    guard let method = class_getInstanceMethod(cls, selector) else { return false }
    let original = unsafeBitCast(method_getImplementation(method), to: E194SetPipelineIMP.self)
    let replacement: @convention(block) (AnyObject, AnyObject) -> Void = { encoder, pipeline in
        E194Ledger.shared.bind(encoder: encoder, pipeline: pipeline)
        original(encoder, selector, pipeline)
    }
    method_setImplementation(method, imp_implementationWithBlock(replacement))
    return true
}

private func e194SwizzleNewPipeline(_ cls: AnyClass) -> Bool {
    let selector = NSSelectorFromString("newComputePipelineStateWithFunction:error:")
    guard let method = class_getInstanceMethod(cls, selector) else { return false }
    let original = unsafeBitCast(method_getImplementation(method), to: E194NewPipelineIMP.self)
    let replacement: @convention(block) (AnyObject, AnyObject, UnsafeMutableRawPointer?)
        -> UnsafeMutableRawPointer? = { device, function, errorOut in
            let result = original(device, selector, function, errorOut)
            if let result {
                let pipeline = Unmanaged<AnyObject>.fromOpaque(result).takeUnretainedValue()
                E194Ledger.shared.note(
                    pipeline: pipeline, name: (function as? MTLFunction)?.name ?? "<unnamed>")
            }
            return result
        }
    method_setImplementation(method, imp_implementationWithBlock(replacement))
    return true
}

/// Install every swizzle EXACTLY once per process. `method_setImplementation`
/// captures the previous IMP, so a second installation would chain the
/// counting block onto itself and double every count.
private enum E194Swizzle {
    static let installed: Bool = install()

    private static func install() -> Bool {
        guard let device = MTLCreateSystemDefaultDevice(),
            let queue = device.makeCommandQueue(),
            let buffer = queue.makeCommandBuffer(),
            let encoder = buffer.makeComputeCommandEncoder()
        else { return false }
        let encoderClass: AnyClass = type(of: encoder as AnyObject)
        let deviceClass: AnyClass = type(of: device as AnyObject)
        encoder.endEncoding()

        var ok = e194SwizzleNewPipeline(deviceClass)
        ok = e194SwizzleSetPipeline(encoderClass) && ok
        ok = e194SwizzleDispatch(encoderClass, "dispatchThreadgroups:threadsPerThreadgroup:") && ok
        ok = e194SwizzleDispatch(encoderClass, "dispatchThreads:threadsPerThreadgroup:") && ok
        ok = e194SwizzleScopeBarrier(encoderClass) && ok
        _ = e194SwizzleResourceBarrier(encoderClass)
        return ok
    }
}

/// Reduce one recorded event list to the counters the route argues about.
private func e194Summarize(_ events: [E194Event]) -> [String: Any] {
    var sequence: [String] = []
    var counts: [String: Int] = [:]
    var dispatches = 0
    var barriers = 0
    var copies = 0
    var sdpaIndices: [Int] = []
    for event in events {
        switch event {
        case .dispatch(let record):
            counts[record.kernel, default: 0] += 1
            dispatches += 1
            if record.kernel.lowercased().contains("copy") { copies += 1 }
            if record.kernel.contains("sdpa") { sdpaIndices.append(sequence.count) }
            sequence.append(record.kernel)
        case .barrier(let kind):
            barriers += 1
            sequence.append("BARRIER(\(kind))")
        }
    }
    // The decisive counter: barriers strictly between the FIRST and LAST sdpa
    // dispatch of the round. Route 2' must drive this to zero; anything else
    // falsifies mechanism 2.
    var barriersBetweenSdpa = 0
    if let first = sdpaIndices.first, let last = sdpaIndices.last, first < last {
        for index in (first + 1) ..< last where sequence[index].hasPrefix("BARRIER") {
            barriersBetweenSdpa += 1
        }
    }
    return [
        "dispatches": dispatches,
        "barriers": barriers,
        "barriers_between_sdpa": barriersBetweenSdpa,
        "copy_dispatches": copies,
        "sdpa_calls": sdpaIndices.count,
        "kernel_counts": counts,
        "event_sequence": sequence,
    ]
}

// MARK: - The two query-layout forms at the scored geometry

private enum E194Form: String, CaseIterable {
    case today
    case seqMajor
}

private enum E194Probe {
    static let heads = 24
    static let kvHeads = 4
    static let headDim = 256
    static let fullAttentionLayers = 16
    static let split = 5
    static var scale: Float { 1.0 / Float(headDim).squareRoot() }

    /// The scored query layout: the fused `qwen35AttentionQKRMSRoPE` kernel
    /// writes ROW-CONTIGUOUS `[B, H, L, D]` (Qwen35.swift:2362-2367, with
    /// `output_base = local_row * axis_size` enumerating batch, head,
    /// sequence).
    static func rows(width: Int, heads count: Int) -> MLXArray {
        MLXRandom.normal([1, count, width, headDim]).asType(.bfloat16)
    }

    static func queries(width: Int) -> MLXArray { rows(width: width, heads: heads) }
    static func newKV(width: Int) -> MLXArray { rows(width: width, heads: kvHeads) }

    static func seededCache(length: Int) -> KVCacheSimple {
        let keys = newKV(width: length)
        let values = newKV(width: length)
        let cache = KVCacheSimple()
        _ = cache.update(keys: keys, values: values)
        // Force the growth that the first over-capacity append would otherwise
        // charge to whichever arm ran first.
        _ = cache.update(keys: newKV(width: 16), values: newKV(width: 16))
        cache.offset = length
        eval(cache.innerState())
        return cache
    }

    /// The shipped production form, reproduced exactly:
    /// `AttentionUtils.swift:106-142` as of BASE_SHA 80abd5ac. This copy is the
    /// reference arm and stays fixed even after the shipped file changes.
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

    /// Route 2': identical windows, identical kernel family, one shared
    /// sequence-major query buffer.
    static func seqMajor(queries q: MLXArray, cache: KVCacheSimple, kv: MLXArray) -> MLXArray {
        let (cachedKeys, cachedValues) = cache.update(keys: kv, values: kv)
        let qL = q.dim(2)
        let kL = cachedKeys.dim(2)
        let kSplit = kL - (qL - split)
        let shared = q.transposed(0, 2, 1, 3).contiguous().transposed(0, 2, 1, 3)
        let outA = MLXFast.scaledDotProductAttention(
            queries: shared[0..., 0..., 0 ..< split, 0...],
            keys: cachedKeys[0..., 0..., 0 ..< kSplit, 0...],
            values: cachedValues[0..., 0..., 0 ..< kSplit, 0...],
            scale: scale, mask: .causal)
        let outB = MLXFast.scaledDotProductAttention(
            queries: shared[0..., 0..., split..., 0...],
            keys: cachedKeys, values: cachedValues,
            scale: scale, mask: .causal)
        return concatenated([outA, outB], axis: 2)
    }

    /// POSITIVE CONTROL: Route 2' with the split moved by one row. Every window
    /// is still legal and the output shape is unchanged, but rows `split` and
    /// `split+1` now see a different key window, so a correct comparison MUST
    /// report a difference. If this arm matches `today`, the gate is blind.
    static func shiftedSplit(queries q: MLXArray, cache: KVCacheSimple, kv: MLXArray)
        -> MLXArray
    {
        let (cachedKeys, cachedValues) = cache.update(keys: kv, values: kv)
        let qL = q.dim(2)
        let kL = cachedKeys.dim(2)
        let shifted = split + 1
        let kSplit = kL - (qL - shifted)
        let shared = q.transposed(0, 2, 1, 3).contiguous().transposed(0, 2, 1, 3)
        let outA = MLXFast.scaledDotProductAttention(
            queries: shared[0..., 0..., 0 ..< shifted, 0...],
            keys: cachedKeys[0..., 0..., 0 ..< kSplit, 0...],
            values: cachedValues[0..., 0..., 0 ..< kSplit, 0...],
            scale: scale, mask: .causal)
        let outB = MLXFast.scaledDotProductAttention(
            queries: shared[0..., 0..., shifted..., 0...],
            keys: cachedKeys, values: cachedValues,
            scale: scale, mask: .causal)
        return concatenated([outA, outB], axis: 2)
    }

    static func call(
        _ form: E194Form, q: MLXArray, kv: MLXArray, cache: KVCacheSimple, kvLength: Int
    ) -> MLXArray {
        cache.offset = kvLength
        switch form {
        case .today: return today(queries: q, cache: cache, kv: kv)
        case .seqMajor: return seqMajor(queries: q, cache: cache, kv: kv)
        }
    }

    /// Exact bf16 comparison: count differing elements and the worst absolute
    /// delta over the whole output. An argmax match is not accepted as
    /// evidence (program.md step 7).
    static func exactDiff(_ a: MLXArray, _ b: MLXArray) -> (differing: Int, maxAbs: Float) {
        eval(a, b)
        let left = a.asType(.float32)
        let right = b.asType(.float32)
        let delta = abs(left - right)
        let differing = (delta .> 0).sum().item(Int.self)
        return (differing, delta.max().item(Float.self))
    }
}

private func e194WriteReport(_ report: [String: Any], envKey: String) throws {
    let json = try JSONSerialization.data(
        withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
    print(String(decoding: json, as: UTF8.self))
    if let path = ProcessInfo.processInfo.environment[envKey], !path.isEmpty {
        try json.write(to: URL(fileURLWithPath: path))
    }
}

// MARK: - Section 1: dispatch AND barrier census

@Suite(.serialized)
struct E194CensusTests {
    @Test(
        .enabled(
            if: ProcessInfo.processInfo.environment["MLX_E194_CENSUS"] == "1",
            "set MLX_E194_CENSUS=1 to run the GPU dispatch and barrier census"))
    func countsDispatchesAndBarriersPerForm() throws {
        #expect(E194Swizzle.installed)
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("no Metal device")
            return
        }

        var cells: [[String: Any]] = []
        for kvLength in [512, 1024] {
            for width in [6, 7, 8, 9] {
                for form in E194Form.allCases {
                    let q = E194Probe.queries(width: width)
                    let kv = E194Probe.newKV(width: width)
                    eval(q, kv)
                    let cache = E194Probe.seededCache(length: kvLength)
                    // Warm the kernels and the pipeline map before counting so
                    // a first-use JIT compile cannot appear as a dispatch.
                    for _ in 0 ..< 3 {
                        eval(E194Probe.call(form, q: q, kv: kv, cache: cache, kvLength: kvLength))
                    }
                    E194Ledger.shared.start()
                    eval(E194Probe.call(form, q: q, kv: kv, cache: cache, kvLength: kvLength))
                    let events = E194Ledger.shared.stop()

                    var cell = e194Summarize(events)
                    cell["form"] = form.rawValue
                    cell["qL"] = width
                    cell["kv"] = kvLength
                    cells.append(cell)
                }
            }
        }

        try e194WriteReport(
            [
                "probe": "e194-dispatch-barrier-census",
                "harness": "local",
                "host_architecture": device.architecture.name,
                "full_attention_layers": E194Probe.fullAttentionLayers,
                "gqa_factor": E194Probe.heads / E194Probe.kvHeads,
                "cells": cells,
            ], envKey: "MLX_E194_CENSUS_OUT")
    }
}

// MARK: - Section 2: bf16 value gate with a positive control

@Suite(.serialized)
struct E194ValueGateTests {
    @Test(
        .enabled(
            if: ProcessInfo.processInfo.environment["MLX_E194_GATE"] == "1",
            "set MLX_E194_GATE=1 to run the bf16 value gate"))
    func routeTwoPrimeIsBitExact() throws {
        var cells: [[String: Any]] = []
        var worstDiffering = 0
        var controlFired = true

        for kvLength in [512, 1024, 2048] {
            for width in [6, 7, 8, 9] {
                let q = E194Probe.queries(width: width)
                let kv = E194Probe.newKV(width: width)
                eval(q, kv)
                let cache = E194Probe.seededCache(length: kvLength)

                cache.offset = kvLength
                let reference = E194Probe.today(queries: q, cache: cache, kv: kv)
                cache.offset = kvLength
                let candidate = E194Probe.seqMajor(queries: q, cache: cache, kv: kv)
                cache.offset = kvLength
                let control = E194Probe.shiftedSplit(queries: q, cache: cache, kv: kv)

                let compared = E194Probe.exactDiff(reference, candidate)
                let controlCompared = E194Probe.exactDiff(reference, control)
                worstDiffering = max(worstDiffering, compared.differing)
                if controlCompared.differing == 0 { controlFired = false }

                cells.append([
                    "qL": width, "kv": kvLength,
                    "elements": reference.size,
                    "differing_elements": compared.differing,
                    "max_abs_delta": Double(compared.maxAbs),
                    "control_differing_elements": controlCompared.differing,
                    "control_max_abs_delta": Double(controlCompared.maxAbs),
                ])

                #expect(
                    compared.differing == 0,
                    "Route 2' changed \(compared.differing) values at qL=\(width) kv=\(kvLength)")
                #expect(
                    controlCompared.differing > 0,
                    "positive control did not fire at qL=\(width) kv=\(kvLength)")
            }
        }

        try e194WriteReport(
            [
                "probe": "e194-value-gate",
                "harness": "local",
                "dtype": "bfloat16",
                "worst_differing_elements": worstDiffering,
                "positive_control_fired_everywhere": controlFired,
                "cells": cells,
            ], envKey: "MLX_E194_GATE_OUT")
    }
}

// MARK: - Section 3: arm liveness on the SHIPPED function

/// Sections 1 and 2 measure private copies of the two forms. This section
/// measures the function the worker actually calls,
/// `MLXLMCommon.attentionWithCacheUpdate`, and proves that
/// `MLX_E194_SEQ_MAJOR_Q` selects which form runs there.
///
/// Without it the ABBA session has a guard that cannot fail: if the flag never
/// reached the branch, both timing arms would execute the same code and the
/// experiment would report a null that means "no switch", not "no effect".
/// Run this file TWICE, once per value of the flag — the flag is read once per
/// process by design, so one process can only witness one arm.
@Suite(.serialized)
struct E194ArmLivenessTests {
    @Test(
        .enabled(
            if: ProcessInfo.processInfo.environment["MLX_E194_ARM"] == "1",
            "set MLX_E194_ARM=1 to census the shipped attentionWithCacheUpdate"))
    func shippedPathFollowsTheArmFlag() throws {
        #expect(E194Swizzle.installed)
        let raw = ProcessInfo.processInfo.environment["MLX_E194_SEQ_MAJOR_Q"]
        let expectedOn = raw != "0"
        #expect(mlxE194SequenceMajorQueryShare == expectedOn)

        let kvLength = 512
        var cells: [[String: Any]] = []
        for width in [6, 7, 8, 9] {
            let q = E194Probe.queries(width: width)
            let kv = E194Probe.newKV(width: width)
            eval(q, kv)

            func census(_ body: (KVCacheSimple) -> MLXArray) -> [String: Any] {
                let cache = E194Probe.seededCache(length: kvLength)
                for _ in 0 ..< 3 {
                    cache.offset = kvLength
                    eval(body(cache))
                }
                cache.offset = kvLength
                E194Ledger.shared.start()
                eval(body(cache))
                return e194Summarize(E194Ledger.shared.stop())
            }

            let shipped = census { cache in
                attentionWithCacheUpdate(
                    queries: q, keys: kv, values: kv, cache: cache,
                    scale: E194Probe.scale, mask: .causal)
            }
            let today = census { cache in
                E194Probe.today(queries: q, cache: cache, kv: kv)
            }
            let seqMajor = census { cache in
                E194Probe.seqMajor(queries: q, cache: cache, kv: kv)
            }

            let reference = expectedOn ? seqMajor : today
            let other = expectedOn ? today : seqMajor
            for key in ["dispatches", "barriers", "barriers_between_sdpa", "copy_dispatches"] {
                #expect(
                    shipped[key] as? Int == reference[key] as? Int,
                    """
                    shipped \(key)=\(shipped[key] ?? -1) at qL=\(width) does not match the \
                    \(expectedOn ? "seqMajor" : "today") form (\(reference[key] ?? -1))
                    """)
            }
            // The forms must differ, or matching one of them proves nothing.
            #expect(reference["dispatches"] as? Int != other["dispatches"] as? Int)

            cells.append([
                "qL": width, "kv": kvLength,
                "shipped": shipped, "today": today, "seq_major": seqMajor,
            ])
        }

        try e194WriteReport(
            [
                "probe": "e194-arm-liveness",
                "harness": "local",
                "env_MLX_E194_SEQ_MAJOR_Q": raw ?? "<unset>",
                "arm_flag_value": mlxE194SequenceMajorQueryShare,
                "cells": cells,
            ], envKey: "MLX_E194_ARM_OUT")
    }
}

