import Darwin
import Foundation
import MLX
import Metal
import ObjectiveC

// E58 RESEARCH INSTRUMENT -- COUNT every GPU dispatch a decode round issues, and
// PRICE one dispatch in situ. Reverted out of the submitted surface before the
// experiment closes; every entry point is off unless its `MLX_E58_*` variable is
// set, and the shipped default is a no-op.
//
// Two independent instruments live here because they answer two different
// questions and must never run in the same process:
//
//   census (MLX_E58_DISPATCH_CENSUS=1)
//     Swizzles pipeline creation, pipeline binding, both dispatch selectors,
//     the buffer barrier and command-buffer commit on this process's Metal
//     classes, then attributes every dispatch to the round, verify width and
//     round phase that issued it. Counting is exact; the swizzle's own lock
//     cost makes the process unfit for timing.
//
//   tax (MLX_E58_DISPATCH_TAX=N)
//     Adds N deliberately trivial dispatches per round so the round's marginal
//     cost per dispatch can be regressed from matched timed runs. It adds no
//     model arithmetic: `metal` mode fires a one-thread kernel that touches one
//     float, `mlx` mode chains N one-element MLX adds. Nothing it computes
//     reaches the model, the head, the caches or the ledger.
//
// The `MLX_` prefix is load-bearing: `sanitizedRuntimeWorkerEnvironment` admits
// `MLX_` and drops `MLXFAST_`, so a `MLXFAST_`-spelled gate would never reach
// the worker process that owns the dispatch path.
public enum E58DispatchCensus {
    private static let environment = ProcessInfo.processInfo.environment

    static let censusEnabled = environment["MLX_E58_DISPATCH_CENSUS"] == "1"
    static let shapesEnabled = environment["MLX_E58_DISPATCH_CENSUS_SHAPES"] == "1"
    static let taxPerRound = Int(environment["MLX_E58_DISPATCH_TAX"] ?? "") ?? 0
    static let taxMode = environment["MLX_E58_DISPATCH_TAX_MODE"] ?? "metal"
    static let taxOpsPerBuffer =
        Int(environment["MLX_E58_DISPATCH_TAX_OPS_PER_BUFFER"] ?? "") ?? 64
    /// A blocking tax prices a SERIALISED dispatch: encode, submit and wait. A
    /// non-blocking tax prices a PIPELINED dispatch, which is what a real round
    /// issues, so its slope answers whether one more dispatch in the stream
    /// costs anything at all. The pair brackets the marginal price.
    static let taxWaits = (environment["MLX_E58_DISPATCH_TAX_WAIT"] ?? "1") != "0"

    /// RESEARCH PROBE ONLY. MLX caches both command-buffer limits once, when it
    /// constructs its device. On this 48 GiB host the trusted worker force-sets
    /// 128 MiB and 64 ops with overwrite=1 AFTER `resolve()` returns, so no
    /// editable writer can change the limits by `setenv` alone. Setting these
    /// variables writes the requested limits and then touches MLX, which pins
    /// the device before the trusted override runs. It exists only to show
    /// whether the round's command-buffer geometry responds to the limits at
    /// all, it is never part of a candidate, and it is reverted with the rest of
    /// this instrument.
    static let probeBufferMegabytes = environment["MLX_E58_BUFFER_LIMIT_MB"]
    static let probeBufferOps = environment["MLX_E58_BUFFER_LIMIT_OPS"]

    /// Called from the earliest editable startup hook, before the backbone and
    /// head loads create any Metal pipeline. Installing later would leave the
    /// pipelines built during weight loading and warmup unmapped, and their
    /// dispatches would be counted as `<unmapped>`.
    public static func installIfRequested() {
        if censusEnabled { DispatchLedger.shared.install() }
        pinBufferLimitsIfRequested()
        if taxPerRound > 0 { DispatchTax.shared.prepare() }
    }

    private static func pinBufferLimitsIfRequested() {
        guard probeBufferMegabytes != nil || probeBufferOps != nil else { return }
        if let megabytes = probeBufferMegabytes {
            setenv("MLX_MAX_MB_PER_BUFFER", megabytes, 1)
        }
        if let ops = probeBufferOps {
            setenv("MLX_MAX_OPS_PER_BUFFER", ops, 1)
        }
        // Touching MLX here constructs its device, which reads and caches both
        // limits, so the trusted worker's later overwrite cannot move them.
        let pin = MLXArray([Float(0)]) + Float(1)
        eval(pin)
        DispatchLedger.shared.noteBufferLimitProbe(
            megabytes: probeBufferMegabytes, ops: probeBufferOps)
    }

    public static func beginRound(round: Int, width: Int, depth: Int) {
        guard censusEnabled else { return }
        DispatchLedger.shared.beginRound(round: round, width: width, depth: depth)
    }

    public static func phase(_ name: String) {
        guard censusEnabled else { return }
        DispatchLedger.shared.setPhase(name)
    }

    public static func endRound(accepted: Int) {
        guard censusEnabled else { return }
        DispatchLedger.shared.endRound(accepted: accepted)
    }

    public static func fireTax() {
        guard taxPerRound > 0 else { return }
        DispatchTax.shared.fire(count: taxPerRound)
    }
}

// MARK: - census

private struct PhaseCounters {
    var dispatches = 0
    var barriers = 0
    var commits = 0
    /// Wall nanoseconds spent inside the ORIGINAL Metal implementations, so the
    /// ledger's own bookkeeping is excluded. `dispatchNs` is host encode cost;
    /// `commitNs` is host submit cost. Neither includes the GPU wait, which MLX
    /// performs elsewhere, so their sum is a direct lower bound on the round's
    /// host-side dispatch price and cannot be inflated by GPU time.
    var dispatchNs = 0
    var commitNs = 0
    var ledgerNs = 0
    var kernels: [String: Int] = [:]
    var shapes: [String: Int] = [:]

    mutating func add(kernel: String, shape: String, recordShape: Bool) {
        dispatches += 1
        kernels[kernel, default: 0] += 1
        if recordShape { shapes[shape, default: 0] += 1 }
    }
}

@inline(__always)
private func nowNs() -> UInt64 { clock_gettime_nsec_np(CLOCK_UPTIME_RAW) }

/// `@unchecked Sendable` because every mutable field is reached only under
/// `lock`. The swizzled Metal selectors run on whichever thread MLX's stream
/// happens to encode from, while the phase markers run on the session thread.
private final class DispatchLedger: @unchecked Sendable {
    static let shared = DispatchLedger()

    private let lock = NSLock()
    private var pipelineNames: [ObjectIdentifier: String] = [:]
    private var encoderBinding: [ObjectIdentifier: String] = [:]
    private var phases: [String: PhaseCounters] = [:]
    private var currentPhase = "outside"
    private var round = 0
    private var width = 0
    private var depth = 0
    private var installed = false
    private var segmentStartNs: UInt64 = 0

    private static let sink: FileHandle = {
        guard let path = ProcessInfo.processInfo
            .environment["MLX_E58_DISPATCH_CENSUS_PATH"], !path.isEmpty
        else { return FileHandle.standardError }
        let fd = open(path, O_WRONLY | O_CREAT | O_APPEND, 0o644)
        guard fd >= 0 else { return FileHandle.standardError }
        return FileHandle(fileDescriptor: fd, closeOnDealloc: false)
    }()

    func install() {
        lock.lock()
        let already = installed
        installed = true
        lock.unlock()
        guard !already else { return }
        guard let device = MTLCreateSystemDefaultDevice(),
            let queue = device.makeCommandQueue(),
            let buffer = queue.makeCommandBuffer(),
            let encoder = buffer.makeComputeCommandEncoder()
        else {
            write(["event": "census_install_failed"])
            return
        }
        let encoderClass: AnyClass = type(of: encoder as AnyObject)
        let bufferClass: AnyClass = type(of: buffer as AnyObject)
        let deviceClass: AnyClass = type(of: device as AnyObject)
        encoder.endEncoding()

        let results: [String: Bool] = [
            "new_pipeline": swizzleNewPipeline(deviceClass),
            "set_pipeline": swizzleSetPipeline(encoderClass),
            "dispatch_threadgroups": swizzleDispatch(
                encoderClass, "dispatchThreadgroups:threadsPerThreadgroup:"),
            "dispatch_threads": swizzleDispatch(
                encoderClass, "dispatchThreads:threadsPerThreadgroup:"),
            "barrier": swizzleBarrier(encoderClass),
            "commit": swizzleCommit(bufferClass),
        ]
        write([
            "event": "census_installed",
            "encoder_class": String(describing: encoderClass),
            "buffer_class": String(describing: bufferClass),
            "device_class": String(describing: deviceClass),
            "hooks": results.map { "\($0.key)=\($0.value ? 1 : 0)" }
                .sorted().joined(separator: ","),
        ])
    }

    func noteBufferLimitProbe(megabytes: String?, ops: String?) {
        write([
            "event": "buffer_limit_probe",
            "requested_mb": megabytes ?? "<unset>",
            "requested_ops": ops ?? "<unset>",
        ])
    }

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

    func dispatch(
        encoder: AnyObject, grid: MTLSize, threadgroup: MTLSize,
        ledgerNs: Int, originalNs: Int
    ) {
        lock.lock()
        let kernel = encoderBinding[ObjectIdentifier(encoder)] ?? "<unbound>"
        let shape = "\(kernel) grid=\(grid.width)x\(grid.height)x\(grid.depth)"
            + " tg=\(threadgroup.width)x\(threadgroup.height)x\(threadgroup.depth)"
        phases[currentPhase, default: PhaseCounters()].add(
            kernel: kernel, shape: shape,
            recordShape: E58DispatchCensus.shapesEnabled)
        phases[currentPhase, default: PhaseCounters()].dispatchNs += originalNs
        phases[currentPhase, default: PhaseCounters()].ledgerNs += ledgerNs
        lock.unlock()
    }

    func barrier() {
        lock.lock()
        phases[currentPhase, default: PhaseCounters()].barriers += 1
        lock.unlock()
    }

    func commit(originalNs: Int) {
        lock.lock()
        phases[currentPhase, default: PhaseCounters()].commits += 1
        phases[currentPhase, default: PhaseCounters()].commitNs += originalNs
        lock.unlock()
    }

    func setPhase(_ name: String) {
        lock.lock()
        currentPhase = name
        lock.unlock()
    }

    /// Flushes whatever accumulated outside a round -- warmup, the seed
    /// prefill, and any dispatch MLX encoded between rounds -- then opens the
    /// new round's buckets.
    func beginRound(round: Int, width: Int, depth: Int) {
        flush(kind: "gap")
        lock.lock()
        self.round = round
        self.width = width
        self.depth = depth
        currentPhase = "round_open"
        segmentStartNs = nowNs()
        lock.unlock()
    }

    func endRound(accepted: Int) {
        flush(kind: "round", accepted: accepted)
        lock.lock()
        currentPhase = "outside"
        lock.unlock()
    }

    private func flush(kind: String, accepted: Int = -1) {
        let closed = nowNs()
        lock.lock()
        let taken = phases
        let takenRound = round
        let takenWidth = width
        let takenDepth = depth
        let wallNs = segmentStartNs == 0 ? 0 : Int(closed - segmentStartNs)
        segmentStartNs = closed
        phases = [:]
        lock.unlock()
        guard !taken.isEmpty else { return }
        var payload: [String: Any] = [
            "event": kind,
            "round": takenRound,
            "width": takenWidth,
            "depth": takenDepth,
            "wall_ns": wallNs,
        ]
        if accepted >= 0 { payload["accepted"] = accepted }
        var phasePayload: [String: Any] = [:]
        for (name, counters) in taken {
            var entry: [String: Any] = [
                "dispatches": counters.dispatches,
                "barriers": counters.barriers,
                "commits": counters.commits,
                "dispatch_ns": counters.dispatchNs,
                "commit_ns": counters.commitNs,
                "clock_bias_ns": counters.ledgerNs,
                "kernels": counters.kernels,
            ]
            if !counters.shapes.isEmpty { entry["shapes"] = counters.shapes }
            phasePayload[name] = entry
        }
        payload["phases"] = phasePayload
        write(payload)
    }

    private func write(_ payload: [String: Any]) {
        var enriched = payload
        enriched["pid"] = Int(getpid())
        enriched["t_ns"] = Int(DispatchTime.now().uptimeNanoseconds)
        guard let data = try? JSONSerialization.data(
            withJSONObject: enriched, options: [.sortedKeys])
        else { return }
        Self.sink.write(data)
        Self.sink.write(Data("\n".utf8))
    }
}

private typealias DispatchIMP = @convention(c) (AnyObject, Selector, MTLSize, MTLSize) -> Void
private typealias SetPipelineIMP = @convention(c) (AnyObject, Selector, AnyObject) -> Void
private typealias BarrierIMP = @convention(c) (AnyObject, Selector, UInt) -> Void
private typealias CommitIMP = @convention(c) (AnyObject, Selector) -> Void
private typealias NewPipelineIMP = @convention(c) (
    AnyObject, Selector, AnyObject, UnsafeMutableRawPointer?
) -> UnsafeMutableRawPointer?

private func swizzleDispatch(_ cls: AnyClass, _ name: String) -> Bool {
    let selector = NSSelectorFromString(name)
    guard let method = class_getInstanceMethod(cls, selector) else { return false }
    let original = unsafeBitCast(method_getImplementation(method), to: DispatchIMP.self)
    let replacement: @convention(block) (AnyObject, MTLSize, MTLSize) -> Void = {
        encoder, grid, threadgroup in
        // Time the original call only, so neither the swizzle nor the ledger's
        // lock is charged to Metal. The back-to-back clock reads price one clock
        // read, which is the measurement's own bias on `originalNs`.
        let entered = nowNs()
        let started = nowNs()
        original(encoder, selector, grid, threadgroup)
        let ended = nowNs()
        DispatchLedger.shared.dispatch(
            encoder: encoder, grid: grid, threadgroup: threadgroup,
            ledgerNs: Int(started - entered), originalNs: Int(ended - started))
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

private func swizzleBarrier(_ cls: AnyClass) -> Bool {
    let selector = NSSelectorFromString("memoryBarrierWithScope:")
    guard let method = class_getInstanceMethod(cls, selector) else { return false }
    let original = unsafeBitCast(method_getImplementation(method), to: BarrierIMP.self)
    let replacement: @convention(block) (AnyObject, UInt) -> Void = { encoder, scope in
        DispatchLedger.shared.barrier()
        original(encoder, selector, scope)
    }
    method_setImplementation(method, imp_implementationWithBlock(replacement))
    return true
}

private func swizzleCommit(_ cls: AnyClass) -> Bool {
    let selector = NSSelectorFromString("commit")
    guard let method = class_getInstanceMethod(cls, selector) else { return false }
    let original = unsafeBitCast(method_getImplementation(method), to: CommitIMP.self)
    let replacement: @convention(block) (AnyObject) -> Void = { buffer in
        let started = nowNs()
        original(buffer, selector)
        let ended = nowNs()
        DispatchLedger.shared.commit(originalNs: Int(ended - started))
    }
    method_setImplementation(method, imp_implementationWithBlock(replacement))
    return true
}

/// The pipeline-creation hook returns the original +1 pointer unchanged, so the
/// ownership contract of a `new` method is preserved.
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

// MARK: - in-situ dispatch tax

/// Fires N trivial dispatches per round on its own queue and waits for them, so
/// the added cost lands on the round's critical path exactly like a real
/// dispatch does. `metal` mode is the pure Metal encode/submit/launch cost;
/// `mlx` mode routes the same count through MLX's own op and command-buffer
/// machinery, so the pair brackets how much of a dispatch's price is Metal's
/// and how much is MLX's.
private final class DispatchTax: @unchecked Sendable {
    static let shared = DispatchTax()

    private var queue: MTLCommandQueue?
    private var pipeline: MTLComputePipelineState?
    private var scratch: MTLBuffer?
    private var mlxScratch: MLXArray?

    private static let source = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void e58_tax(device float *out [[buffer(0)]],
                            uint tid [[thread_position_in_grid]]) {
            out[0] = out[0] + 1.0f;
        }
        """

    func prepare() {
        switch E58DispatchCensus.taxMode {
        case "mlx":
            mlxScratch = MLXArray([Float(0)])
            eval(mlxScratch!)
        default:
            guard let device = MTLCreateSystemDefaultDevice(),
                let queue = device.makeCommandQueue(),
                let library = try? device.makeLibrary(
                    source: Self.source, options: nil),
                let function = library.makeFunction(name: "e58_tax"),
                let pipeline = try? device.makeComputePipelineState(
                    function: function),
                let scratch = device.makeBuffer(
                    length: 4, options: .storageModeShared)
            else {
                fputs("e58: dispatch tax failed to prepare its metal path\n", stderr)
                return
            }
            self.queue = queue
            self.pipeline = pipeline
            self.scratch = scratch
        }
    }

    func fire(count: Int) {
        if E58DispatchCensus.taxMode == "mlx" {
            guard let scratch = mlxScratch else { return }
            var chained = scratch
            for _ in 0 ..< count { chained = chained + Float(1) }
            eval(chained)
            return
        }
        guard let queue, let pipeline, let scratch else { return }
        let opsPerBuffer = Swift.max(1, E58DispatchCensus.taxOpsPerBuffer)
        var remaining = count
        var last: MTLCommandBuffer?
        while remaining > 0 {
            let batch = Swift.min(remaining, opsPerBuffer)
            remaining -= batch
            guard let buffer = queue.makeCommandBuffer(),
                let encoder = buffer.makeComputeCommandEncoder()
            else { return }
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(scratch, offset: 0, index: 0)
            for _ in 0 ..< batch {
                encoder.dispatchThreads(
                    MTLSize(width: 1, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
            }
            encoder.endEncoding()
            buffer.commit()
            last = buffer
        }
        if E58DispatchCensus.taxWaits { last?.waitUntilCompleted() }
    }
}
