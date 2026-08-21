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

    /// E80 extension. GPU MILLISECONDS, not dispatch counts. `dispatch_ns`
    /// below is host encode time and mis-ranks the surface by about 1000x, so
    /// this gate adds a command-buffer completion handler that reads
    /// `GPUStartTime` / `GPUEndTime`. It implies the census, because the GPU
    /// clock alone cannot name a kernel: the pipeline map does that.
    static let gpuTimeEnabled = environment["MLX_E80_GPU_TIME"] == "1"
    /// Rounds per delta snapshot. Snapshots are DELTAS, so an offline reader
    /// can drop warmup rounds by dropping the first snapshots.
    static let gpuSnapshotRounds =
        Int(environment["MLX_E80_SNAPSHOT_ROUNDS"] ?? "") ?? 16

    /// E85 extension. DEVICE BUFFER ALLOCATIONS, not dispatches. The frontier
    /// pair priced one eliminated materialised intermediate at 13-16 us per
    /// draft, which is far above the 0.66-1.55 us dispatch price E80 measured,
    /// so the candidate mechanism is the allocation and its write-then-read
    /// round trip rather than the launch. This gate swizzles the two device
    /// buffer-creation selectors and attributes every allocation, its byte
    /// count and its host cost to the round phase that requested it. It
    /// implies the census, because only the phase machine can name the phase.
    static let allocCensusEnabled = environment["MLX_E85_ALLOC_CENSUS"] == "1"

    static let censusEnabled =
        environment["MLX_E58_DISPATCH_CENSUS"] == "1" || gpuTimeEnabled
        || allocCensusEnabled
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
        if gpuTimeEnabled { GPUTimeLedger.shared.armExitFlush() }
        pinBufferLimitsIfRequested()
        if taxPerRound > 0 { DispatchTax.shared.prepare() }
    }

    /// E80 rung 2. Pins the round's proposed draft count so one leg measures one
    /// verify width. Research-only: the emitted stream stays the same greedy
    /// target chain at any width, but a pinned width is not a schedule anyone
    /// would ship.
    public static let forcedDrafts = Int(environment["MLX_E80_FORCE_DRAFTS"] ?? "")

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

    /// E80. Names a measured window that is NOT a decode round -- an E71 census
    /// block, for example -- so the GPU ledger can emit one delta snapshot per
    /// window. `label` becomes the snapshot's `window` field.
    public static func beginWindow(_ label: String, width: Int) {
        guard gpuTimeEnabled else { return }
        GPUTimeLedger.shared.beginWindow(label: label, width: width)
    }

    public static func endWindow() {
        guard gpuTimeEnabled else { return }
        GPUTimeLedger.shared.endWindow()
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
    /// Host nanoseconds blocked inside `waitUntilCompleted`. MLX calls it from
    /// `CommandEncoder::synchronize()`, so this is the only place in a round
    /// where the host stops and waits for the GPU.
    var waits = 0
    var waitNs = 0
    var ledgerNs = 0
    /// E85. Device buffer creations that reached Metal. MLX serves most array
    /// outputs from its own buffer cache, so this counts only the requests the
    /// cache missed, which is exactly the population that can carry a
    /// per-allocation price.
    var allocations = 0
    var allocBytes = 0
    var allocNs = 0
    var kernels: [String: Int] = [:]
    var shapes: [String: Int] = [:]
    /// Byte count -> creation count, so an allocation can be matched to the
    /// intermediate that requested it by size.
    var allocSizes: [String: Int] = [:]

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

        var results: [String: Bool] = [
            "new_pipeline": swizzleNewPipeline(deviceClass),
            "set_pipeline": swizzleSetPipeline(encoderClass),
            "dispatch_threadgroups": swizzleDispatch(
                encoderClass, "dispatchThreadgroups:threadsPerThreadgroup:"),
            "dispatch_threads": swizzleDispatch(
                encoderClass, "dispatchThreads:threadsPerThreadgroup:"),
            "barrier": swizzleBarrier(encoderClass),
        ]
        // E80. The encoder-creation hooks must be installed BEFORE `commit`,
        // because the commit hook reads the per-buffer dispatch record that the
        // encoder map builds. Every dispatch whose encoder has no buffer is
        // reported as `unmapped_encoder_dispatches` rather than dropped.
        if E58DispatchCensus.gpuTimeEnabled {
            results["encoder_plain"] = swizzleEncoderPlain(
                bufferClass, "computeCommandEncoder")
            results["encoder_dispatch_type"] = swizzleEncoderDispatchType(
                bufferClass, "computeCommandEncoderWithDispatchType:")
            results["encoder_descriptor"] = swizzleEncoderDescriptor(
                bufferClass, "computeCommandEncoderWithDescriptor:")
        }
        if E58DispatchCensus.allocCensusEnabled {
            results["new_buffer_length"] = swizzleNewBufferLength(deviceClass)
            results["new_buffer_bytes"] = swizzleNewBufferBytes(deviceClass)
        }
        results["commit"] = swizzleCommit(bufferClass)
        results["wait"] = swizzleWait(bufferClass)
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
        let phase = currentPhase
        let observedWidth = width
        let observedRound = round
        lock.unlock()
        if E58DispatchCensus.gpuTimeEnabled {
            GPUTimeLedger.shared.noteDispatch(
                encoder: encoder, kernel: kernel, shape: shape, phase: phase,
                width: observedWidth, round: observedRound, encodeNs: originalNs)
        }
    }

    func barrier() {
        lock.lock()
        phases[currentPhase, default: PhaseCounters()].barriers += 1
        lock.unlock()
    }

    func allocate(bytes: Int, originalNs: Int) {
        lock.lock()
        phases[currentPhase, default: PhaseCounters()].allocations += 1
        phases[currentPhase, default: PhaseCounters()].allocBytes += bytes
        phases[currentPhase, default: PhaseCounters()].allocNs += originalNs
        phases[currentPhase, default: PhaseCounters()]
            .allocSizes["\(bytes)", default: 0] += 1
        lock.unlock()
    }

    func commit(originalNs: Int) {
        lock.lock()
        phases[currentPhase, default: PhaseCounters()].commits += 1
        phases[currentPhase, default: PhaseCounters()].commitNs += originalNs
        lock.unlock()
    }

    func wait(originalNs: Int) {
        lock.lock()
        phases[currentPhase, default: PhaseCounters()].waits += 1
        phases[currentPhase, default: PhaseCounters()].waitNs += originalNs
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
        let closedRound = round
        lock.unlock()
        if E58DispatchCensus.gpuTimeEnabled {
            GPUTimeLedger.shared.roundClosed(round: closedRound)
        }
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
                "waits": counters.waits,
                "wait_ns": counters.waitNs,
                "clock_bias_ns": counters.ledgerNs,
                "allocations": counters.allocations,
                "alloc_bytes": counters.allocBytes,
                "alloc_ns": counters.allocNs,
                "kernels": counters.kernels,
            ]
            if !counters.allocSizes.isEmpty {
                entry["alloc_sizes"] = counters.allocSizes
            }
            if !counters.shapes.isEmpty { entry["shapes"] = counters.shapes }
            phasePayload[name] = entry
        }
        payload["phases"] = phasePayload
        write(payload)
    }

    private func write(_ payload: [String: Any]) { E58CensusSink.write(payload) }
}

/// One O_APPEND sink shared by the dispatch census and the E80 GPU-time ledger,
/// so the reference, verify and timed workers can all write the same JSONL file
/// without one truncating another's records.
enum E58CensusSink {
    private static let handle: FileHandle = {
        guard let path = ProcessInfo.processInfo
            .environment["MLX_E58_DISPATCH_CENSUS_PATH"], !path.isEmpty
        else { return FileHandle.standardError }
        let fd = open(path, O_WRONLY | O_CREAT | O_APPEND, 0o644)
        guard fd >= 0 else { return FileHandle.standardError }
        return FileHandle(fileDescriptor: fd, closeOnDealloc: false)
    }()
    private static let lock = NSLock()

    static func write(_ payload: [String: Any]) {
        var enriched = payload
        enriched["pid"] = Int(getpid())
        enriched["t_ns"] = Int(DispatchTime.now().uptimeNanoseconds)
        guard let data = try? JSONSerialization.data(
            withJSONObject: enriched, options: [.sortedKeys])
        else { return }
        lock.lock()
        handle.write(data)
        handle.write(Data("\n".utf8))
        lock.unlock()
    }
}

private typealias DispatchIMP = @convention(c) (AnyObject, Selector, MTLSize, MTLSize) -> Void
private typealias SetPipelineIMP = @convention(c) (AnyObject, Selector, AnyObject) -> Void
private typealias BarrierIMP = @convention(c) (AnyObject, Selector, UInt) -> Void
private typealias CommitIMP = @convention(c) (AnyObject, Selector) -> Void
private typealias NewPipelineIMP = @convention(c) (
    AnyObject, Selector, AnyObject, UnsafeMutableRawPointer?
) -> UnsafeMutableRawPointer?
private typealias NewBufferLengthIMP = @convention(c) (
    AnyObject, Selector, UInt, UInt
) -> UnsafeMutableRawPointer?
private typealias NewBufferBytesIMP = @convention(c) (
    AnyObject, Selector, UnsafeRawPointer?, UInt, UInt
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
        // The completion handler must be attached BEFORE the original commit,
        // and the record must be taken out of the live map at the same point,
        // so a buffer that Metal recycles cannot alias a later one.
        if E58DispatchCensus.gpuTimeEnabled {
            GPUTimeLedger.shared.willCommit(buffer: buffer)
        }
        let started = nowNs()
        original(buffer, selector)
        let ended = nowNs()
        DispatchLedger.shared.commit(originalNs: Int(ended - started))
    }
    method_setImplementation(method, imp_implementationWithBlock(replacement))
    return true
}

private func swizzleWait(_ cls: AnyClass) -> Bool {
    let selector = NSSelectorFromString("waitUntilCompleted")
    guard let method = class_getInstanceMethod(cls, selector) else { return false }
    let original = unsafeBitCast(method_getImplementation(method), to: CommitIMP.self)
    let replacement: @convention(block) (AnyObject) -> Void = { buffer in
        let started = nowNs()
        original(buffer, selector)
        let ended = nowNs()
        DispatchLedger.shared.wait(originalNs: Int(ended - started))
    }
    method_setImplementation(method, imp_implementationWithBlock(replacement))
    return true
}

/// E85. Both device buffer-creation hooks return the original +1 pointer
/// unchanged, for the same ownership reason as the pipeline hook below. MLX's
/// Metal allocator calls `newBufferWithLength:options:`; the `bytes` variant is
/// hooked as well so a host-initialised array cannot leave the census silently
/// short.
private func swizzleNewBufferLength(_ cls: AnyClass) -> Bool {
    let selector = NSSelectorFromString("newBufferWithLength:options:")
    guard let method = class_getInstanceMethod(cls, selector) else { return false }
    let original = unsafeBitCast(method_getImplementation(method), to: NewBufferLengthIMP.self)
    let replacement: @convention(block) (AnyObject, UInt, UInt)
        -> UnsafeMutableRawPointer? = { device, length, options in
            let started = nowNs()
            let result = original(device, selector, length, options)
            let ended = nowNs()
            DispatchLedger.shared.allocate(
                bytes: Int(length), originalNs: Int(ended - started))
            return result
        }
    method_setImplementation(method, imp_implementationWithBlock(replacement))
    return true
}

private func swizzleNewBufferBytes(_ cls: AnyClass) -> Bool {
    let selector = NSSelectorFromString("newBufferWithBytes:length:options:")
    guard let method = class_getInstanceMethod(cls, selector) else { return false }
    let original = unsafeBitCast(method_getImplementation(method), to: NewBufferBytesIMP.self)
    let replacement: @convention(block) (AnyObject, UnsafeRawPointer?, UInt, UInt)
        -> UnsafeMutableRawPointer? = { device, bytes, length, options in
            let started = nowNs()
            let result = original(device, selector, bytes, length, options)
            let ended = nowNs()
            DispatchLedger.shared.allocate(
                bytes: Int(length), originalNs: Int(ended - started))
            return result
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

// MARK: - E80 GPU-time ledger

/// One dispatch, as the encoder saw it.
private struct DispatchNote {
    let kernel: String
    /// `kernel` plus the dispatch grid and threadgroup. One Metal function name
    /// covers wildly different work here: all 257 `affine_qmv_fast` dispatches in
    /// a verify round share a name while their output width runs from 5,120 to
    /// 248,320, so a per-NAME GPU time would average the projections with the
    /// vocabulary readout and mean nothing. Per-SHAPE is the resolvable unit.
    let shape: String
    let phase: String
    let width: Int
    let round: Int
    let encodeNs: Int
}

/// One bucket of GPU nanoseconds. `gpuNs` is the SUM of command-buffer
/// durations, so it double-counts when two buffers overlap on the device;
/// `busyNs` is the union of those intervals and does not.
private struct GPUBucket {
    var buffers = 0
    var dispatches = 0
    var gpuNs = 0
    var driverNs = 0
    var encodeNs = 0

    mutating func add(gpuNs: Int, driverNs: Int, encodeNs: Int, dispatches: Int) {
        self.buffers += 1
        self.dispatches += dispatches
        self.gpuNs += gpuNs
        self.driverNs += driverNs
        self.encodeNs += encodeNs
    }
}

private struct KernelBucket {
    var buffers = 0
    var gpuNs = 0
    var minNs = Int.max
    var maxNs = 0
    var sumSq = 0.0

    mutating func add(_ ns: Int) {
        buffers += 1
        gpuNs += ns
        minNs = Swift.min(minNs, ns)
        maxNs = Swift.max(maxNs, ns)
        sumSq += Double(ns) * Double(ns)
    }
}

/// E80. Per-command-buffer GPU time, keyed to the kernels that buffer carried.
///
/// `atDispatchBoundary` counter sampling is NOT supported on this host
/// (`research/e80-artifacts/rung0c-counter-capability.json`), so per-dispatch
/// GPU timestamps do not exist here. The only exact GPU clock available is
/// `MTLCommandBuffer.GPUStartTime` / `GPUEndTime`. That gives:
///
///   * exact per-kernel GPU time whenever a buffer carries ONE dispatch, which
///     is what `MLX_E58_BUFFER_LIMIT_OPS=1` forces. That is the ISOLATED cost;
///   * exact total GPU time, per phase and per verify width, in the default
///     configuration, where buffers carry many dispatches. That is the IN-SITU
///     cost, and its composition is recorded per buffer signature so an offline
///     regression can price a kernel inside a concurrent stream.
///
/// The ratio of the two is the concurrency discount, which is a result in its
/// own right rather than a nuisance parameter.
private final class GPUTimeLedger: @unchecked Sendable {
    static let shared = GPUTimeLedger()

    private let lock = NSLock()
    private var encoderToBuffer: [ObjectIdentifier: ObjectIdentifier] = [:]
    private var bufferNotes: [ObjectIdentifier: [DispatchNote]] = [:]
    private var bufferEncoders: [ObjectIdentifier: [ObjectIdentifier]] = [:]

    private var byPhase: [String: GPUBucket] = [:]
    private var byWidthPhase: [String: GPUBucket] = [:]
    private var exclusiveKernels: [String: KernelBucket] = [:]
    private var signatures: [String: GPUBucket] = [:]
    private var unmappedDispatches = 0
    private var untrackedBuffers = 0
    private var zeroTimeBuffers = 0
    private var completedBuffers = 0
    /// Coverage of the NNLS fit: how many in-phase buffers became equations,
    /// and how many were dropped for spanning phases.
    private var signatureBuffers = 0
    private var mixedPhaseBuffers = 0
    /// Never reset. `drain` compares the two to know when every committed
    /// buffer has reported its GPU interval.
    private var committedTotal = 0
    private var completedTotal = 0

    /// Device-clock union of every completed buffer interval, and the idle time
    /// between them. Command buffers on one queue complete in submission order,
    /// so the union can be accumulated online.
    private var busyNs = 0.0
    private var idleNs = 0.0
    private var spanStart = 0.0
    private var spanEnd = 0.0
    private var lastEnd = 0.0

    private var snapshotIndex = 0
    private var roundsInSnapshot = 0
    private var firstRoundInSnapshot = -1
    private var lastRoundInSnapshot = -1
    private var window: String?
    private var windowWidth = 0
    private var atexitArmed = false

    func armExitFlush() {
        lock.lock()
        let already = atexitArmed
        atexitArmed = true
        lock.unlock()
        guard !already else { return }
        atexit {
            GPUTimeLedger.shared.drain(timeoutMs: 1000)
            GPUTimeLedger.shared.snapshot(reason: "exit")
        }
    }

    func noteEncoder(_ encoder: AnyObject, buffer: AnyObject) {
        let encoderID = ObjectIdentifier(encoder)
        let bufferID = ObjectIdentifier(buffer)
        lock.lock()
        encoderToBuffer[encoderID] = bufferID
        bufferEncoders[bufferID, default: []].append(encoderID)
        lock.unlock()
    }

    func noteDispatch(
        encoder: AnyObject, kernel: String, shape: String, phase: String,
        width: Int, round: Int, encodeNs: Int
    ) {
        let note = DispatchNote(
            kernel: kernel, shape: shape, phase: phase, width: width,
            round: round, encodeNs: encodeNs)
        let encoderID = ObjectIdentifier(encoder)
        lock.lock()
        if let bufferID = encoderToBuffer[encoderID] {
            bufferNotes[bufferID, default: []].append(note)
        } else {
            unmappedDispatches += 1
        }
        lock.unlock()
    }

    func willCommit(buffer: AnyObject) {
        let bufferID = ObjectIdentifier(buffer)
        lock.lock()
        let notes = bufferNotes.removeValue(forKey: bufferID) ?? []
        for encoderID in bufferEncoders.removeValue(forKey: bufferID) ?? [] {
            encoderToBuffer.removeValue(forKey: encoderID)
        }
        if notes.isEmpty { untrackedBuffers += 1 } else { committedTotal += 1 }
        lock.unlock()
        guard !notes.isEmpty, let commandBuffer = buffer as? MTLCommandBuffer
        else { return }
        commandBuffer.addCompletedHandler { [weak self] finished in
            self?.complete(buffer: finished, notes: notes)
        }
    }

    private func complete(buffer: MTLCommandBuffer, notes: [DispatchNote]) {
        let start = buffer.gpuStartTime
        let end = buffer.gpuEndTime
        let gpuNs = Int(((end - start) * 1e9).rounded())
        let driverNs = Int(
            ((buffer.kernelEndTime - buffer.kernelStartTime) * 1e9).rounded())
        let encodeNs = notes.reduce(0) { $0 + $1.encodeNs }

        var perPhase: [String: Int] = [:]
        for note in notes { perPhase[note.phase, default: 0] += 1 }
        let width = notes[0].width
        // Composed from SHAPE, not kernel name. Every projection in the model
        // dispatches the same `affine_qmv_fast` function and is told apart only
        // by its grid, so a name-keyed signature pools the 248320-wide lm_head
        // readout with the 5120-wide down projections. Measured on the rung-2
        // debug leg that pooling put the fitted qmv time 7x under the directly
        // measured lm_head buffers.
        let signature = notes
            .reduce(into: [String: Int]()) { $0[$1.shape, default: 0] += 1 }
            .sorted { $0.key < $1.key }
            .map { "\($0.key)*\($0.value)" }
            .joined(separator: ",")

        lock.lock()
        completedBuffers += 1
        completedTotal += 1
        if gpuNs <= 0 { zeroTimeBuffers += 1 }
        if spanStart == 0 { spanStart = start }
        spanEnd = Swift.max(spanEnd, end)
        if start >= lastEnd {
            busyNs += (end - start) * 1e9
            if lastEnd > 0 { idleNs += (start - lastEnd) * 1e9 }
        } else {
            busyNs += Swift.max(0, end - lastEnd) * 1e9
        }
        lastEnd = Swift.max(lastEnd, end)

        // A buffer that spans phases is split by dispatch count, which is the
        // only split this instrument can defend. Single-phase buffers, which
        // are the overwhelming majority, are exact.
        for (phase, count) in perPhase {
            let share = Double(count) / Double(notes.count)
            let phaseNs = Int(Double(gpuNs) * share)
            byPhase[phase, default: GPUBucket()].add(
                gpuNs: phaseNs, driverNs: Int(Double(driverNs) * share),
                encodeNs: Int(Double(encodeNs) * share), dispatches: count)
            byWidthPhase["w\(width)|\(phase)", default: GPUBucket()].add(
                gpuNs: phaseNs, driverNs: Int(Double(driverNs) * share),
                encodeNs: Int(Double(encodeNs) * share), dispatches: count)
        }
        if notes.count == 1 {
            exclusiveKernels["w\(width)|\(notes[0].phase)|\(notes[0].shape)",
                default: KernelBucket()].add(gpuNs)
        }
        // Only single-phase buffers become NNLS equations. A buffer that spans
        // phases would need its GPU interval split before it could be an
        // equation, and the dispatch-count split used for `byPhase` is an
        // approximation this fit must not inherit. `outside` is excluded
        // because it holds weight loading, warmup and teardown, whose shape
        // variety would dominate the record for no analytical gain.
        if perPhase.count == 1, notes[0].phase != "outside" {
            signatureBuffers += 1
            signatures["w\(width)|\(notes[0].phase)|\(signature)",
                default: GPUBucket()].add(
                    gpuNs: gpuNs, driverNs: driverNs, encodeNs: encodeNs,
                    dispatches: notes.count)
        } else if notes[0].phase != "outside" || perPhase.count > 1 {
            mixedPhaseBuffers += 1
        }
        lock.unlock()
    }

    /// A snapshot is a DELTA since the previous snapshot, so opening a window
    /// without flushing first would charge it with everything that ran between
    /// the windows. In the E71 harness that is the 768-token seed prefill, the
    /// warmup reps and the cache teardown: measured at 6.4 s against a 0.24 s
    /// window, which is 26x the signal. The per-phase buckets survive that
    /// pollution because the inter-block work carries the `outside` phase, but
    /// `gpu_busy_ns` and `gpu_idle_ns` are unions over the whole delta and do
    /// not. Flushing here puts that work in its own record and leaves the window
    /// record holding only the window.
    func beginWindow(label: String, width: Int) {
        drain(timeoutMs: 500)
        snapshot(reason: "pre_window")
        lock.lock()
        window = label
        windowWidth = width
        lock.unlock()
    }

    func endWindow() {
        drain(timeoutMs: 500)
        snapshot(reason: "window")
        lock.lock()
        window = nil
        windowWidth = 0
        lock.unlock()
    }

    /// Waits for the completion handlers of every committed buffer. `eval` can
    /// return before the handler for its own buffer has run, so a window closed
    /// without this drain would leak its last buffers into the next window.
    func drain(timeoutMs: Int) {
        let deadline = nowNs() + UInt64(timeoutMs) * 1_000_000
        while nowNs() < deadline {
            lock.lock()
            let outstanding = committedTotal - completedTotal
            lock.unlock()
            if outstanding <= 0 { return }
            usleep(200)
        }
    }

    func roundClosed(round: Int) {
        lock.lock()
        if firstRoundInSnapshot < 0 { firstRoundInSnapshot = round }
        lastRoundInSnapshot = round
        roundsInSnapshot += 1
        let due = roundsInSnapshot >= E58DispatchCensus.gpuSnapshotRounds
        lock.unlock()
        if due { snapshot(reason: "rounds") }
    }

    /// Emits the accumulated DELTA and resets every bucket. `lastEnd` is kept,
    /// so the idle accounting stays continuous across snapshots.
    func snapshot(reason: String) {
        lock.lock()
        let payload: [String: Any] = [
            "event": "gputime",
            "reason": reason,
            "snapshot": snapshotIndex,
            "window": window ?? "",
            "window_width": windowWidth,
            "rounds": roundsInSnapshot,
            "round_first": firstRoundInSnapshot,
            "round_last": lastRoundInSnapshot,
            "completed_buffers": completedBuffers,
            "committed_total": committedTotal,
            "completed_total": completedTotal,
            "untracked_buffers": untrackedBuffers,
            "unmapped_encoder_dispatches": unmappedDispatches,
            "zero_time_buffers": zeroTimeBuffers,
            "signature_buffers": signatureBuffers,
            "mixed_phase_buffers": mixedPhaseBuffers,
            "gpu_busy_ns": Int(busyNs),
            "gpu_idle_ns": Int(idleNs),
            "gpu_span_ns": Int(Swift.max(0, spanEnd - spanStart) * 1e9),
            "by_phase": byPhase.mapValues(Self.encode),
            "by_width_phase": byWidthPhase.mapValues(Self.encode),
            "exclusive_kernels": exclusiveKernels.mapValues { bucket in
                [
                    "buffers": bucket.buffers, "gpu_ns": bucket.gpuNs,
                    "min_ns": bucket.minNs == Int.max ? 0 : bucket.minNs,
                    "max_ns": bucket.maxNs, "sum_sq_ns": bucket.sumSq,
                ] as [String: Any]
            },
            "signatures": signatures.mapValues(Self.encode),
        ]
        snapshotIndex += 1
        roundsInSnapshot = 0
        firstRoundInSnapshot = -1
        lastRoundInSnapshot = -1
        completedBuffers = 0
        untrackedBuffers = 0
        unmappedDispatches = 0
        zeroTimeBuffers = 0
        signatureBuffers = 0
        mixedPhaseBuffers = 0
        busyNs = 0
        idleNs = 0
        spanStart = 0
        spanEnd = 0
        byPhase = [:]
        byWidthPhase = [:]
        exclusiveKernels = [:]
        signatures = [:]
        lock.unlock()
        E58CensusSink.write(payload)
    }

    private static func encode(_ bucket: GPUBucket) -> [String: Any] {
        [
            "buffers": bucket.buffers, "dispatches": bucket.dispatches,
            "gpu_ns": bucket.gpuNs, "driver_ns": bucket.driverNs,
            "encode_ns": bucket.encodeNs,
        ]
    }
}

private typealias EncoderPlainIMP = @convention(c) (AnyObject, Selector)
    -> UnsafeMutableRawPointer?
private typealias EncoderDispatchTypeIMP = @convention(c) (
    AnyObject, Selector, UInt
) -> UnsafeMutableRawPointer?
private typealias EncoderDescriptorIMP = @convention(c) (
    AnyObject, Selector, AnyObject
) -> UnsafeMutableRawPointer?

private func swizzleEncoderPlain(_ cls: AnyClass, _ name: String) -> Bool {
    let selector = NSSelectorFromString(name)
    guard let method = class_getInstanceMethod(cls, selector) else { return false }
    let original = unsafeBitCast(
        method_getImplementation(method), to: EncoderPlainIMP.self)
    let replacement: @convention(block) (AnyObject) -> UnsafeMutableRawPointer? = {
        buffer in
        let result = original(buffer, selector)
        if let result {
            GPUTimeLedger.shared.noteEncoder(
                Unmanaged<AnyObject>.fromOpaque(result).takeUnretainedValue(),
                buffer: buffer)
        }
        return result
    }
    method_setImplementation(method, imp_implementationWithBlock(replacement))
    return true
}

private func swizzleEncoderDispatchType(_ cls: AnyClass, _ name: String) -> Bool {
    let selector = NSSelectorFromString(name)
    guard let method = class_getInstanceMethod(cls, selector) else { return false }
    let original = unsafeBitCast(
        method_getImplementation(method), to: EncoderDispatchTypeIMP.self)
    let replacement:
        @convention(block) (AnyObject, UInt) -> UnsafeMutableRawPointer? = {
            buffer, dispatchType in
            let result = original(buffer, selector, dispatchType)
            if let result {
                GPUTimeLedger.shared.noteEncoder(
                    Unmanaged<AnyObject>.fromOpaque(result).takeUnretainedValue(),
                    buffer: buffer)
            }
            return result
        }
    method_setImplementation(method, imp_implementationWithBlock(replacement))
    return true
}

private func swizzleEncoderDescriptor(_ cls: AnyClass, _ name: String) -> Bool {
    let selector = NSSelectorFromString(name)
    guard let method = class_getInstanceMethod(cls, selector) else { return false }
    let original = unsafeBitCast(
        method_getImplementation(method), to: EncoderDescriptorIMP.self)
    let replacement:
        @convention(block) (AnyObject, AnyObject) -> UnsafeMutableRawPointer? = {
            buffer, descriptor in
            let result = original(buffer, selector, descriptor)
            if let result {
                GPUTimeLedger.shared.noteEncoder(
                    Unmanaged<AnyObject>.fromOpaque(result).takeUnretainedValue(),
                    buffer: buffer)
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
