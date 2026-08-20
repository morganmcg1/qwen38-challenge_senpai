import Foundation
import MLX
import Metal
import ObjectiveC
import Testing

// E70 rung 1 -- CAPTURE the kernel name MLX actually selects at each scored
// shape, under the real architecture and under a forced `MLX_METAL_GPU_ARCH`.
// Research only. This suite makes no timing claim and no correctness claim.
//
// Why a Swift test and not a script: the selection happens inside the vendored
// C++ Metal backend, which builds a kernel name string and hands it to
// `d.get_kernel`. Nothing in that path is observable from Python. The only
// vantage point is the Metal API itself, so this suite swizzles the same three
// selectors that `E57SdpaChunkDispatchCountTests` uses and reads the name off
// the `MTLFunction`.
//
// `Tests/` is not in `benchmark.json` `editablePaths`, so this file changes no
// candidate file and is not part of any submission archive.
//
// 🔴 Forced-architecture runs are kernel-selection evidence ONLY. Setting
// `MLX_METAL_GPU_ARCH=applegpu_g17s` makes `is_nax_available()` return true on
// silicon that has no neural accelerators. The `_nax` kernels may fail to
// compile, produce wrong numbers, or abort the process. All three are valid
// outcomes of the probe and none of them says anything about performance.
//
// Switches:
//   MLXFAST_E70_ARCH_PROBE=1  enables the suite
//   MLXFAST_E70_CELL=<id>     runs one cell; `all` runs every cell in order
//   MLXFAST_E70_OUT=<path>    optional JSON output path
//
// The forced arm runs one cell per process, because an uncaught C++ throw from
// a failed `_nax` pipeline ends the process and would hide every later cell.

private struct DispatchRecord {
    var kernel: String
    var grid: MTLSize
    var threadgroup: MTLSize
}

/// `@unchecked Sendable` because every mutable field is reached only under
/// `lock`, and the swizzled Metal selectors can be called from any thread.
private final class ArchProbeLedger: @unchecked Sendable {
    static let shared = ArchProbeLedger()

    private let lock = NSLock()
    private var pipelineNames: [ObjectIdentifier: String] = [:]
    private var encoderBinding: [ObjectIdentifier: String] = [:]
    private var records: [DispatchRecord] = []
    private var created: [String] = []
    private var recording = false

    func note(pipeline: AnyObject, name: String) {
        lock.lock()
        pipelineNames[ObjectIdentifier(pipeline)] = name
        if recording { created.append(name) }
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
        created = []
        recording = true
        lock.unlock()
    }

    func stop() -> (dispatched: [DispatchRecord], created: [String]) {
        lock.lock()
        recording = false
        let takenRecords = records
        let takenCreated = created
        records = []
        created = []
        lock.unlock()
        return (takenRecords, takenCreated)
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
        ArchProbeLedger.shared.dispatch(
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
        ArchProbeLedger.shared.bind(encoder: encoder, pipeline: pipeline)
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
                ArchProbeLedger.shared.note(pipeline: pipeline, name: name)
            }
            return result
        }
    method_setImplementation(method, imp_implementationWithBlock(replacement))
    return true
}

/// Unbuffered, so a process abort inside a forced-arch cell still leaves the
/// name of the cell that killed it in the log.
private func trace(_ message: String) {
    FileHandle.standardError.write(Data("e70-probe: \(message)\n".utf8))
}

private enum ArchProbe {
    // The scored geometry: hidden 5120, head_dim 256, 24 query heads, 4 KV
    // heads, affine 4-bit group 64. `fa.qkv` packed is N = 14336.
    static let hidden = 5120
    static let packedQKV = 14336
    static let heads = 24
    static let kvHeads = 4
    static let headDim = 256
    static let groupSize = 64
    static let bits = 4

    struct QuantizedWeight {
        var wq: MLXArray
        var scales: MLXArray
        var biases: MLXArray?
    }

    static func quantizedWeight(rows: Int, columns: Int) -> QuantizedWeight {
        let dense = MLXRandom.normal([rows, columns]).asType(.bfloat16)
        let (wq, scales, biases) = quantized(dense, groupSize: groupSize, bits: bits)
        eval(wq, scales)
        if let biases { eval(biases) }
        return QuantizedWeight(wq: wq, scales: scales, biases: biases)
    }

    /// The packed form built directly, without the dense matrix it would come
    /// from. `lm_head` is N = 248320 by K = 5120, and quantizing that honestly
    /// needs a 5 GB float32 temporary. Kernel selection reads only the shapes,
    /// so random packed bits answer the routing question at 1/8 of the memory.
    static func syntheticQuantizedWeight(rows: Int, columns: Int) -> QuantizedWeight {
        let packed = MLXRandom.randInt(Int32(0) ..< 65536, [rows, columns * bits / 32])
            .asType(.uint32)
        let scales = MLXRandom.normal([rows, columns / groupSize]).asType(.bfloat16)
        let biases = MLXRandom.normal([rows, columns / groupSize]).asType(.bfloat16)
        eval(packed, scales, biases)
        return QuantizedWeight(wq: packed, scales: scales, biases: biases)
    }

    /// `quantized.cpp:776-810`, transcribed. Returns the route a transposed
    /// non-batched quantized matmul takes once `M >= vector_limit`.
    static func splitKRoute(m: Int, n: Int, k: Int) -> (splitK: Int, route: String) {
        let bm = 32, bn = 32
        let nTiles = (n + bn - 1) / bn
        let mTiles = (m + bm - 1) / bm
        let kAlign = max(groupSize, 32)
        var splitK = min(max(1, 512 / (nTiles * mTiles)), k / kAlign)
        while splitK > 1 && k % (splitK * kAlign) != 0 {
            splitK -= 1
        }
        return (splitK, splitK <= 1 ? "qmm" : "qmm_t_splitk")
    }

    /// A KV cache hands attention a row slice of one preallocated buffer, which
    /// `kv_copy_unless` accepts without a copy. Building the keys the same way
    /// keeps the probed call faithful to the scored one.
    static func cacheSlice(length: Int) -> MLXArray {
        let buffer = MLXRandom.normal([1, kvHeads, length + 64, headDim]).asType(.bfloat16)
        return buffer[0..., 0..., 0 ..< length, 0...]
    }

    static func attention(qL: Int, kL: Int) -> () -> MLXArray {
        let queries = MLXRandom.normal([1, heads, qL, headDim]).asType(.bfloat16)
        let keys = cacheSlice(length: kL)
        let values = cacheSlice(length: kL)
        eval(queries, keys, values)
        return {
            MLXFast.scaledDotProductAttention(
                queries: queries, keys: keys, values: values,
                scale: 1.0 / Float(headDim).squareRoot(), mask: .causal)
        }
    }
}

/// One probed shape: what it is, which audited site it exercises, and what
/// rung 0 predicted for each arm.
///
/// `body` is built by a factory that has already generated and evaluated its
/// inputs, so the captured region contains the audited operation alone. With
/// the random inputs built inside the capture instead, every cell also reports
/// the `rbitsc` and `v_copy*` kernels of the RNG, which obscures the selection.
private struct ProbeCellSpec {
    var id: String
    var site: String
    var shape: String
    var rung0Local: String
    var rung0Forced: String
    var makeBody: () -> () -> MLXArray
}

private func probeCells() -> [ProbeCellSpec] {
    func quantizedCell(m: Int) -> () -> () -> MLXArray {
        {
            // The packed QKV projection, N = 14336, is the widest scored
            // affine-4 linear and it appears in both the target layers and the
            // MTP head.
            let w = ArchProbe.quantizedWeight(
                rows: ArchProbe.packedQKV, columns: ArchProbe.hidden)
            let x = MLXRandom.normal([m, ArchProbe.hidden]).asType(.bfloat16)
            eval(x)
            return {
                quantizedMatmul(
                    x, w.wq, scales: w.scales, biases: w.biases,
                    transpose: true, groupSize: ArchProbe.groupSize,
                    bits: ArchProbe.bits)
            }
        }
    }

    func denseCell(m: Int) -> () -> () -> MLXArray {
        {
            let x = MLXRandom.normal([m, ArchProbe.hidden]).asType(.bfloat16)
            let w = MLXRandom.normal([2048, ArchProbe.hidden]).asType(.bfloat16)
            let wt = w.transposed(1, 0)
            eval(x, wt)
            return { matmul(x, wt) }
        }
    }

    var cells: [ProbeCellSpec] = []
    for m in [1, 5, 9] {
        cells.append(
            ProbeCellSpec(
                id: "qmv_m\(m)",
                site: "S3 quantized.cpp:84 get_qmv_batch_limit",
                shape: "quantizedMatmul M=\(m) K=5120 N=14336 affine g64 b4 transposed",
                rung0Local: "M < vector_limit 10 -> dispatch_qmv -> affine_qmv_fast*",
                rung0Forced: "identical: vector_limit is 10 on gen 16 and gen 17",
                makeBody: quantizedCell(m: m)))
    }
    for m in [10, 511, 512] {
        cells.append(
            ProbeCellSpec(
                id: "qmm_m\(m)",
                site: "S4 quantized.cpp:697 qmm nax gate",
                shape: "quantizedMatmul M=\(m) K=5120 N=14336 affine g64 b4 transposed",
                rung0Local: "M >= 10 -> qmm_splitk -> split_k 1 -> qmm() -> affine_qmm_t*",
                rung0Forced: "is_nax_available() true -> affine_qmm_t_nax*",
                makeBody: quantizedCell(m: m)))
    }
    cells.append(
        ProbeCellSpec(
            id: "sdpa_prefill_512",
            site: "S13 sdpa.cpp:177 unreachable, S9/S7/S8 dense fallback",
            shape: "SDPA qL=512 kL=512 head_dim=256 heads=24 kv=4 causal",
            rung0Local:
                "head_dim 256 excludes sdpa_full and qL 512 exceeds the vector cap, "
                + "so use_fallback is true -> two dense steel_gemm_fused_* GEMMs",
            rung0Forced: "same fallback, but steel_gemm_fused_nax_*",
            makeBody: { ArchProbe.attention(qL: 512, kL: 512) }))
    for qL in [1, 5] {
        for kL in [768, 1030] {
            cells.append(
                ProbeCellSpec(
                    id: "sdpa_vector_q\(qL)_k\(kL)",
                    site: "S14 sdpa.cpp:443 and S15 sdpa.cpp:748, both devc-keyed",
                    shape: "SDPA qL=\(qL) kL=\(kL) head_dim=256 heads=24 kv=4 causal",
                    rung0Local:
                        kL >= 1024
                        ? "devc 's' and kL >= 1024 -> sdpa_vector_2pass_*"
                        : "devc 's' but kL < 1024 -> sdpa_vector_*",
                    rung0Forced: "identical: the forced arch keeps devc == 's'",
                    makeBody: { ArchProbe.attention(qL: qL, kL: kL) }))
        }
    }
    cells.append(
        ProbeCellSpec(
            id: "dense_gemv_m1",
            site: "S9 matmul.cpp:915, not reached",
            shape: "matmul M=1 K=5120 N=2048 bfloat16",
            rung0Local: "min(M, N) == 1 returns to gemv before any architecture read",
            rung0Forced: "identical: the early return is upstream of the nax gate",
            makeBody: denseCell(m: 1)))
    cells.append(
        ProbeCellSpec(
            id: "dense_matmul_m511",
            site: "S9 matmul.cpp:915 with S7 vs S8 tile parameters",
            shape: "matmul M=511 K=5120 N=2048 bfloat16 (the MTP head prime island patch)",
            rung0Local: "use_nax false -> steel_matmul_regular_axpby -> steel_gemm_fused_nt*",
            rung0Forced: "use_nax true -> steel_matmul_regular_axpby_NAX -> steel_gemm_fused_nax_nt*",
            makeBody: denseCell(m: 511)))
    cells.append(contentsOf: routingLadderCells())
    return cells
}

/// The advisor's `qmm_splitk` table, turned into live dispatcher evidence.
///
/// At M = 10 the verify route leaves `dispatch_qmv`, and the claim under test
/// is that it then forks: wide-output families fall to `qmm()` and reach the
/// `quantized.cpp:697` nax gate, while narrow-output families take
/// `qmm_t_splitk`, which is not a nax kernel on any host. M = 9 is the control
/// below the cliff and M = 12 the control above it.
private func routingLadderCells() -> [ProbeCellSpec] {
    // Every scored affine-4 linear, named as it appears in the checkpoint.
    let families: [(name: String, k: Int, n: Int)] = [
        ("gdn.out_proj and fa.o_proj share this shape", 6144, 5120),
        ("mlp.down", 17408, 5120),
        ("square control", 5120, 5120),
        ("fa.qkv packed", 5120, 14336),
        ("gdn.in_proj fused", 5120, 16480),
        ("mlp.gate_up fused", 5120, 34816),
        ("lm_head", 5120, 248320),
    ]
    var cells: [ProbeCellSpec] = []
    for family in families {
        for m in [9, 10, 12] {
            let route = ArchProbe.splitKRoute(m: m, n: family.n, k: family.k)
            let belowLimit = m < 10  // vector_limit is 10 on every audited arm
            let local: String
            let forced: String
            if belowLimit {
                local = "M < vector_limit 10 -> dispatch_qmv -> affine_qmv_fast*"
                forced = "identical: vector_limit is 10 on gen 16 and gen 17"
            } else if route.route == "qmm" {
                local = "M >= 10 -> qmm_splitk -> split_k \(route.splitK) -> qmm() "
                    + "-> affine_qmm_t*"
                forced = "same route, but the :697 gate is open -> affine_qmm_t_nax*"
            } else {
                local = "M >= 10 -> qmm_splitk -> split_k \(route.splitK) "
                    + "-> affine_qmm_t_splitk*"
                forced = "identical: qmm_t_splitk never reads is_nax_available()"
            }
            cells.append(
                ProbeCellSpec(
                    id: "route_k\(family.k)_n\(family.n)_m\(m)",
                    site: "S4 quantized.cpp:697 via the qmm_splitk fork at :776-810",
                    shape: "quantizedMatmul M=\(m) K=\(family.k) N=\(family.n) "
                        + "affine g64 b4 transposed -- \(family.name)",
                    rung0Local: local,
                    rung0Forced: forced,
                    makeBody: {
                        let w = ArchProbe.syntheticQuantizedWeight(
                            rows: family.n, columns: family.k)
                        let x = MLXRandom.normal([m, family.k]).asType(.bfloat16)
                        eval(x)
                        return {
                            quantizedMatmul(
                                x, w.wq, scales: w.scales, biases: w.biases,
                                transpose: true, groupSize: ArchProbe.groupSize,
                                bits: ArchProbe.bits)
                        }
                    }))
        }
    }
    return cells
}

private func probeEnabled() -> Bool {
    ProcessInfo.processInfo.environment["MLXFAST_E70_ARCH_PROBE"] == "1"
}

@Suite(.serialized)
struct E70ArchDispatchProbeTests {

    @Test(
        .enabled(
            if: probeEnabled(),
            "set MLXFAST_E70_ARCH_PROBE=1 to run the architecture probe"))
    func capturesSelectedKernelNames() throws {
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
        #expect(swizzleDispatch(encoderClass, "dispatchThreadgroups:threadsPerThreadgroup:"))
        #expect(swizzleDispatch(encoderClass, "dispatchThreads:threadsPerThreadgroup:"))

        let environment = ProcessInfo.processInfo.environment
        let forcedArchitecture = environment["MLX_METAL_GPU_ARCH"] ?? ""
        let selected = environment["MLXFAST_E70_CELL"] ?? "all"
        trace("real_architecture=\(device.architecture.name) "
            + "forced_architecture=\(forcedArchitecture.isEmpty ? "<none>" : forcedArchitecture) "
            + "cell=\(selected)")

        let all = probeCells()
        let cells = selected == "all" ? all : all.filter { $0.id == selected }
        #expect(!cells.isEmpty, "MLXFAST_E70_CELL did not name a known cell")

        var results: [[String: Any]] = []
        for cell in cells {
            trace("begin cell=\(cell.id) shape=\(cell.shape)")

            // The inputs are built and evaluated here, outside the capture.
            let body = cell.makeBody()

            // Warm first, so a first-use JIT compile does not appear as a
            // dispatch. Under a forced architecture this warm call is also the
            // most likely place for the process to die, which is why the trace
            // line above is already flushed.
            eval(body())

            ArchProbeLedger.shared.start()
            eval(body())
            let (dispatched, created) = ArchProbeLedger.shared.stop()

            var counts: [String: Int] = [:]
            var geometry: [String] = []
            for record in dispatched {
                counts[record.kernel, default: 0] += 1
                geometry.append(
                    "\(record.kernel) grid=(\(record.grid.width),\(record.grid.height),"
                        + "\(record.grid.depth)) tg=(\(record.threadgroup.width),"
                        + "\(record.threadgroup.height),\(record.threadgroup.depth))")
            }
            let names = counts.keys.sorted()
            trace("end cell=\(cell.id) dispatches=\(dispatched.count) kernels=\(names)")
            results.append([
                "cell": cell.id,
                "site": cell.site,
                "shape": cell.shape,
                "rung0_local_prediction": cell.rung0Local,
                "rung0_forced_prediction": cell.rung0Forced,
                "dispatches": dispatched.count,
                "kernel_counts": counts,
                "kernel_names": names,
                "kernel_sequence": dispatched.map(\.kernel),
                "pipelines_created_during_capture": created,
                "dispatch_geometry": geometry,
            ])
        }

        let report: [String: Any] = [
            "harness": "arch-probe",
            "warning":
                "kernel-selection evidence only; the silicon is unchanged and any "
                + "number produced under a forced architecture is meaningless",
            "real_architecture": device.architecture.name,
            "forced_architecture": forcedArchitecture,
            "requested_cell": selected,
            "cells": results,
        ]
        let json = try JSONSerialization.data(
            withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
        print(String(decoding: json, as: UTF8.self))
        if let path = environment["MLXFAST_E70_OUT"], !path.isEmpty {
            try json.write(to: URL(fileURLWithPath: path))
        }
    }
}
