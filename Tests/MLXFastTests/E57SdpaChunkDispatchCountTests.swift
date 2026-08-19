import Foundation
import MLX
import Metal
import ObjectiveC
import Testing

// E57 rung 1 -- COUNT the kernels the wide-decode exactness chunk costs, and
// MEASURE which SDPA route each verify width takes. Research only.
//
// The chunk in `attentionWithCacheUpdate` replaces one SDPA call with two, and
// the trusted host dispatcher may add hidden work around each of them: it
// copies a query whose layout fails `q_copy_unless`
// (scaled_dot_product_attention.cpp:686-698) and MLX rejoins the two outputs
// with `concatenated`. None of that is visible from Swift, so this suite counts
// the kernels Metal actually receives.
//
// Method: swizzle three selectors on the concrete Metal classes of this
// process.
//
//   newComputePipelineStateWithFunction:error:  maps a pipeline to its kernel
//                                              name (device.cpp:714 is the
//                                              only creation path MLX uses)
//   setComputePipelineState:                   binds encoder -> kernel name
//   dispatchThreadgroups:threadsPerThreadgroup:  and dispatchThreads:...
//                                              record one line per dispatch
//
// The pipeline-creation hook returns the original +1 pointer unchanged, so the
// ownership contract of a `new` method is preserved.
//
// The threadgroup size of each SDPA dispatch is the measured route evidence:
// `sdpa_vector` 1-pass is a fixed (1024, 1, 1) (:358), `sdpa_vector_2pass`
// requests (32, gqa_factor, qL) (:484), and the steel path uses (32, wm, wn).
//
// Enable with MLXFAST_E57_DISPATCH_COUNT=1. MLXFAST_E57_DISPATCH_COUNT_OUT
// names an optional JSON output path. MLXFAST_E57_DISPATCH_COUNT_THROW=1 runs
// the illegal cell -- one unchunked wide call at kL >= 1024 -- which is
// expected to abort the process from an uncaught C++ runtime_error, so it must
// be run in its own process.

private struct DispatchRecord {
    var kernel: String
    var grid: MTLSize
    var threadgroup: MTLSize
}

/// `@unchecked Sendable` because every mutable field is reached only under
/// `lock`, and the swizzled Metal selectors can be called from any thread.
private final class DispatchLedger: @unchecked Sendable {
    static let shared = DispatchLedger()

    private let lock = NSLock()
    private var pipelineNames: [ObjectIdentifier: String] = [:]
    private var encoderBinding: [ObjectIdentifier: String] = [:]
    private var records: [DispatchRecord] = []
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
                DispatchRecord(
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

    func stop() -> [DispatchRecord] {
        lock.lock()
        recording = false
        let taken = records
        records = []
        lock.unlock()
        return taken
    }
}

private typealias DispatchIMP = @convention(c) (AnyObject, Selector, MTLSize, MTLSize) -> Void
private typealias SetPipelineIMP = @convention(c) (AnyObject, Selector, AnyObject) -> Void
private typealias NewPipelineIMP = @convention(c) (
    AnyObject, Selector, AnyObject, UnsafeMutableRawPointer?
) -> UnsafeMutableRawPointer?

private func swizzleDispatch(_ cls: AnyClass, _ name: String) -> Bool {
    let selector = NSSelectorFromString(name)
    guard let method = class_getInstanceMethod(cls, selector) else { return false }
    let original = unsafeBitCast(method_getImplementation(method), to: DispatchIMP.self)
    let replacement: @convention(block) (AnyObject, MTLSize, MTLSize) -> Void = {
        encoder, grid, threadgroup in
        DispatchLedger.shared.dispatch(
            encoder: encoder, grid: grid, threadgroup: threadgroup)
        original(encoder, selector, grid, threadgroup)
    }
    method_setImplementation(method, imp_implementationWithBlock(replacement))
    return true
}

private func swizzleSetPipeline(_ cls: AnyClass) -> Bool {
    let selector = NSSelectorFromString("setComputePipelineState:")
    guard let method = class_getInstanceMethod(cls, selector) else { return false }
    let original = unsafeBitCast(method_getImplementation(method), to: SetPipelineIMP.self)
    let replacement: @convention(block) (AnyObject, AnyObject) -> Void = { encoder, pipeline in
        DispatchLedger.shared.bind(encoder: encoder, pipeline: pipeline)
        original(encoder, selector, pipeline)
    }
    method_setImplementation(method, imp_implementationWithBlock(replacement))
    return true
}

private func swizzleNewPipeline(_ cls: AnyClass) -> Bool {
    let selector = NSSelectorFromString("newComputePipelineStateWithFunction:error:")
    guard let method = class_getInstanceMethod(cls, selector) else { return false }
    let original = unsafeBitCast(method_getImplementation(method), to: NewPipelineIMP.self)
    let replacement: @convention(block) (AnyObject, AnyObject, UnsafeMutableRawPointer?)
        -> UnsafeMutableRawPointer? = { device, function, errorOut in
            let result = original(device, selector, function, errorOut)
            if let result {
                let pipeline = Unmanaged<AnyObject>.fromOpaque(result).takeUnretainedValue()
                let name = (function as? MTLFunction)?.name ?? "<unnamed>"
                DispatchLedger.shared.note(pipeline: pipeline, name: name)
            }
            return result
        }
    method_setImplementation(method, imp_implementationWithBlock(replacement))
    return true
}

/// One counted attention form at one scored shape.
private struct ProbeCell: Encodable {
    var form: String
    var queryLayout: String
    var qL: Int
    var kL: Int
    var dispatches: Int
    var kernelCounts: [String: Int]
    var sdpaThreadgroups: [String]
    var kernelSequence: [String]
}

private enum SdpaChunkProbe {
    static let heads = 24
    static let kvHeads = 4
    static let headDim = 256
    static let fullAttentionLayers = 16

    /// A KV cache hands attention a row slice of one preallocated buffer, and
    /// `kv_copy_unless` (:702-714) accepts it without a copy because the batch
    /// dimension is 1. Building the keys the same way keeps the measured cell
    /// faithful to the scored call.
    static func cacheSlice(length: Int) -> MLXArray {
        let capacity = length + 64
        let buffer = MLXRandom.normal([1, kvHeads, capacity, headDim]).asType(.bfloat16)
        return buffer[0..., 0..., 0 ..< length, 0...]
    }

    /// `contiguous` is what a fresh RoPE result looks like. `headTransposed` is
    /// the `[B, L, H, D] -> transposed(0, 2, 1, 3)` view the attention layer
    /// builds, which `q_copy_unless` accepts without a copy. The two layouts
    /// bracket the query-copy cost of the chunk.
    static func queries(qL: Int, layout: String) -> MLXArray {
        let contiguous = MLXRandom.normal([1, heads, qL, headDim]).asType(.bfloat16)
        switch layout {
        case "contiguous":
            return contiguous
        default:
            let rowMajor = MLXRandom.normal([1, qL, heads, headDim]).asType(.bfloat16)
            return rowMajor.transposed(0, 2, 1, 3)
        }
    }

    static func whole(queries: MLXArray, keys: MLXArray, values: MLXArray) -> MLXArray {
        MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values,
            scale: 1.0 / Float(headDim).squareRoot(), mask: .causal)
    }

    static func chunked(queries: MLXArray, keys: MLXArray, values: MLXArray) -> MLXArray {
        let qL = queries.dim(2)
        let kL = keys.dim(2)
        let split = 5
        let kSplit = kL - (qL - split)
        let scale = 1.0 / Float(headDim).squareRoot()
        let outA = MLXFast.scaledDotProductAttention(
            queries: queries[0..., 0..., 0 ..< split, 0...],
            keys: keys[0..., 0..., 0 ..< kSplit, 0...],
            values: values[0..., 0..., 0 ..< kSplit, 0...],
            scale: scale, mask: .causal)
        let outB = MLXFast.scaledDotProductAttention(
            queries: queries[0..., 0..., split..., 0...],
            keys: keys, values: values, scale: scale, mask: .causal)
        return concatenated([outA, outB], axis: 2)
    }

    static func measure(
        form: String, layout: String, qL: Int, kL: Int
    ) -> ProbeCell {
        let queries = SdpaChunkProbe.queries(qL: qL, layout: layout)
        let keys = cacheSlice(length: kL)
        let values = cacheSlice(length: kL)
        eval(queries, keys, values)

        // Warm the kernels and the pipeline map before counting, so a first-use
        // JIT compile cannot appear as a dispatch.
        let warm = form == "whole"
            ? whole(queries: queries, keys: keys, values: values)
            : chunked(queries: queries, keys: keys, values: values)
        eval(warm)

        DispatchLedger.shared.start()
        let out = form == "whole"
            ? whole(queries: queries, keys: keys, values: values)
            : chunked(queries: queries, keys: keys, values: values)
        eval(out)
        let records = DispatchLedger.shared.stop()

        var counts: [String: Int] = [:]
        var sdpaThreadgroups: [String] = []
        for record in records {
            counts[record.kernel, default: 0] += 1
            if record.kernel.hasPrefix("sdpa") || record.kernel.hasPrefix("steel") {
                sdpaThreadgroups.append(
                    "\(record.kernel) grid=(\(record.grid.width),\(record.grid.height),"
                        + "\(record.grid.depth)) tg=(\(record.threadgroup.width),"
                        + "\(record.threadgroup.height),\(record.threadgroup.depth))")
            }
        }
        return ProbeCell(
            form: form, queryLayout: layout, qL: qL, kL: kL,
            dispatches: records.count, kernelCounts: counts,
            sdpaThreadgroups: sdpaThreadgroups,
            kernelSequence: records.map(\.kernel))
    }
}

@Suite(.serialized)
struct E57SdpaChunkDispatchCountTests {
    private var enabled: Bool {
        ProcessInfo.processInfo.environment["MLXFAST_E57_DISPATCH_COUNT"] == "1"
    }

    private var runsThrowingCell: Bool {
        ProcessInfo.processInfo.environment["MLXFAST_E57_DISPATCH_COUNT_THROW"] == "1"
    }

    @Test func countsTheChunkSurcharge() throws {
        try #require(enabled, "set MLXFAST_E57_DISPATCH_COUNT=1 to run the GPU probe")

        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("no Metal device")
            return
        }
        guard let queue = device.makeCommandQueue(),
            let buffer = queue.makeCommandBuffer(),
            let encoder = buffer.makeComputeCommandEncoder()
        else {
            Issue.record("could not build a compute encoder to swizzle")
            return
        }
        let encoderClass: AnyClass = type(of: encoder as AnyObject)
        let deviceClass: AnyClass = type(of: device as AnyObject)
        encoder.endEncoding()

        #expect(swizzleNewPipeline(deviceClass))
        #expect(swizzleSetPipeline(encoderClass))
        #expect(
            swizzleDispatch(encoderClass, "dispatchThreadgroups:threadsPerThreadgroup:"))
        #expect(swizzleDispatch(encoderClass, "dispatchThreads:threadsPerThreadgroup:"))

        let widths = [5, 6, 7, 8, 9]
        let coldLength = 768
        let hotLength = 1030
        var cells: [ProbeCell] = []
        for layout in ["contiguous", "headTransposed"] {
            for qL in widths {
                for kL in [coldLength, hotLength] {
                    // The unchunked wide call at kL >= 1024 is the illegal cell.
                    // It aborts the process, so it only runs in the opt-in
                    // single-cell mode below.
                    if kL >= 1024, qL >= 6, qL <= 8 { continue }
                    cells.append(
                        SdpaChunkProbe.measure(
                            form: "whole", layout: layout, qL: qL, kL: kL))
                }
            }
            for qL in widths where qL >= 6 {
                for kL in [coldLength, hotLength] {
                    cells.append(
                        SdpaChunkProbe.measure(
                            form: "chunked", layout: layout, qL: qL, kL: kL))
                }
            }
        }

        let report: [String: Any] = [
            "host_architecture": device.architecture.name,
            "full_attention_layers": SdpaChunkProbe.fullAttentionLayers,
            "gqa_factor": SdpaChunkProbe.heads / SdpaChunkProbe.kvHeads,
            "cells": cells.map { cell -> [String: Any] in
                [
                    "form": cell.form, "query_layout": cell.queryLayout,
                    "qL": cell.qL, "kL": cell.kL, "dispatches": cell.dispatches,
                    "kernel_counts": cell.kernelCounts,
                    "sdpa_threadgroups": cell.sdpaThreadgroups,
                    "kernel_sequence": cell.kernelSequence,
                ]
            },
        ]
        let json = try JSONSerialization.data(
            withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
        print(String(decoding: json, as: UTF8.self))
        if let path = ProcessInfo.processInfo
            .environment["MLXFAST_E57_DISPATCH_COUNT_OUT"], !path.isEmpty
        {
            try json.write(to: URL(fileURLWithPath: path))
        }

        if runsThrowingCell {
            // Route-3 reachability. The pre-registered reading was that an
            // unsplit qL = 6 call at kL >= 1024 asks sdpa_vector_2pass for
            // 32 * 6 * 6 = 1152 threads per threadgroup and that utils.h then
            // throws, ending the process. The legal cells above falsify the
            // premise: at qL >= 6 the dispatcher leaves the vector family for
            // steel_gemm_fused, whose threadgroup is a width-independent
            // (32, 2, 2). This cell is therefore now expected to RETURN, and it
            // still runs in its own process because the pre-registered
            // prediction was an abort.
            print("e57-probe: entering the illegal cell qL=6 kL=1030 unchunked")
            let cell = SdpaChunkProbe.measure(
                form: "whole", layout: "contiguous", qL: 6, kL: 1030)
            print("e57-probe: the illegal cell RETURNED, dispatches=\(cell.dispatches) "
                + "threadgroups=\(cell.sdpaThreadgroups)")
        }
    }

    /// The cheapest decisive check on the boundary the arms move: given ONE set
    /// of queries, keys and values, does the chunked form return the same bits
    /// as the unsplit call? The dispatch counter shows the two forms run
    /// different kernel families at `qL >= 6`, so this test says whether that
    /// difference is observable in the attention output itself, before any
    /// 512-token allocation is spent on the end-to-end row digest.
    ///
    /// Each width also carries an A/A control: the unsplit call is evaluated
    /// twice, and any difference there would mean the comparison is measuring
    /// nondeterminism instead of the chunk.
    @Test func chunkChangesTheAttentionOutputBitwise() throws {
        try #require(enabled, "set MLXFAST_E57_DISPATCH_COUNT=1 to run the GPU probe")

        // kL stays below the 2-pass boundary: the only cell that could abort is
        // an unsplit wide call at kL >= 1024, and that one belongs to --throw.
        let kL = 768
        var cells: [[String: Any]] = []
        for layout in ["contiguous", "headTransposed"] {
            for qL in [5, 6, 7, 8, 9] {
                MLXRandom.seed(UInt64(20_570_000 + qL))
                let queries = SdpaChunkProbe.queries(qL: qL, layout: layout)
                let keys = SdpaChunkProbe.cacheSlice(length: kL)
                let values = SdpaChunkProbe.cacheSlice(length: kL)
                eval(queries, keys, values)

                let whole = SdpaChunkProbe.whole(
                    queries: queries, keys: keys, values: values)
                let wholeAgain = SdpaChunkProbe.whole(
                    queries: queries, keys: keys, values: values)
                eval(whole, wholeAgain)
                let reference = whole.asType(.float32).asArray(Float.self)
                let control = wholeAgain.asType(.float32).asArray(Float.self)

                var candidate = reference
                if qL >= 6 {
                    let chunked = SdpaChunkProbe.chunked(
                        queries: queries, keys: keys, values: values)
                    eval(chunked)
                    #expect(chunked.shape == whole.shape)
                    candidate = chunked.asType(.float32).asArray(Float.self)
                }

                func compare(_ other: [Float]) -> (Int, Float, Float) {
                    var differing = 0
                    var maxAbsolute: Float = 0
                    var maxRelative: Float = 0
                    for index in reference.indices where reference[index] != other[index] {
                        differing += 1
                        let absolute = abs(reference[index] - other[index])
                        maxAbsolute = max(maxAbsolute, absolute)
                        let scale = max(abs(reference[index]), abs(other[index]))
                        if scale > 0 {
                            maxRelative = max(maxRelative, absolute / scale)
                        }
                    }
                    return (differing, maxAbsolute, maxRelative)
                }

                let (controlDiffering, _, _) = compare(control)
                let (differing, maxAbsolute, maxRelative) = compare(candidate)
                cells.append([
                    "query_layout": layout,
                    "qL": qL,
                    "kL": kL,
                    "elements": reference.count,
                    "aa_control_differing_elements": controlDiffering,
                    "chunk_differing_elements": differing,
                    "chunk_differing_fraction":
                        Double(differing) / Double(reference.count),
                    "chunk_max_absolute_difference": Double(maxAbsolute),
                    "chunk_max_relative_difference": Double(maxRelative),
                ])
                // A repeated unsplit call must be bit-identical, otherwise the
                // chunk comparison above measures nondeterminism.
                #expect(controlDiffering == 0)
            }
        }

        let json = try JSONSerialization.data(
            withJSONObject: ["cells": cells],
            options: [.prettyPrinted, .sortedKeys])
        print(String(decoding: json, as: UTF8.self))
        if let path = ProcessInfo.processInfo
            .environment["MLXFAST_E57_BITWISE_OUT"], !path.isEmpty
        {
            try json.write(to: URL(fileURLWithPath: path))
        }
    }
}
