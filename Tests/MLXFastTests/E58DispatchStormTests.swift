import Foundation
import MLX
import Metal
import Testing

// E58 -- what does ONE GPU dispatch cost, with the model removed?
//
// The census counts dispatches exactly. This suite supplies the other half of
// the price: the marginal wall time of one dispatch that performs no useful
// arithmetic. It is a FLOOR on the real per-dispatch cost, never an estimate of
// it: a real dispatch also pays argument encoding proportional to its buffer
// count, a memory barrier, and whatever the kernel itself executes.
//
// Three storms, because they isolate three different costs:
//
//   metal   raw Metal encode + commit + launch, with the dispatch count per
//           command buffer swept. The slope against ops-per-buffer separates
//           per-dispatch encode cost from per-command-buffer submission cost,
//           which is exactly the quantity a larger MLX command buffer would
//           amortise.
//   mlx     the same count routed through MLX's own primitive, graph and
//           command-buffer machinery, so the difference against `metal` is
//           MLX's host-side cost per op.
//   barrier the `metal` storm with a buffer memory barrier before every
//           dispatch, which is what MLX's `maybeInsertBarrier` does on a
//           dependent op chain.
//
// Enable with MLXFAST_E58_DISPATCH_STORM=1 and read the JSON at
// MLXFAST_E58_DISPATCH_STORM_OUT. Set MLX_MAX_OPS_PER_BUFFER before the process
// starts to move MLX's own command-buffer limit: MLX reads it once, when its
// device is constructed.

private func stormEnabled() -> Bool {
    ProcessInfo.processInfo.environment["MLXFAST_E58_DISPATCH_STORM"] == "1"
}

private struct StormPoint: Encodable {
    var mode: String
    var opsPerBuffer: Int
    var dispatches: Int
    var repeats: Int
    var meanSeconds: Double
    var minSeconds: Double
    var microsecondsPerDispatch: Double
    var commandBuffers: Int
}

private struct StormReport: Encodable {
    var host: String
    var device: String
    var mlxMaxOpsPerBuffer: String
    var mlxMaxMBPerBuffer: String
    var points: [StormPoint]
    var metalFixedCostMicroseconds: Double?
    var metalPerDispatchMicroseconds: Double?
    var metalPerBufferMicroseconds: Double?
}

private let stormSource = """
    #include <metal_stdlib>
    using namespace metal;
    kernel void e58_storm(device float *out [[buffer(0)]],
                          uint tid [[thread_position_in_grid]]) {
        out[0] = out[0] + 1.0f;
    }
    """

@Suite(.serialized)
struct E58DispatchStormTests {
    @Test(
        .enabled(
            if: stormEnabled(),
            "set MLXFAST_E58_DISPATCH_STORM=1 to run the GPU dispatch storm"))
    func pricesOneDispatch() throws {
        guard let device = MTLCreateSystemDefaultDevice(),
            let queue = device.makeCommandQueue(),
            let library = try? device.makeLibrary(source: stormSource, options: nil),
            let function = library.makeFunction(name: "e58_storm"),
            let scratch = device.makeBuffer(length: 4, options: .storageModeShared)
        else {
            Issue.record("no Metal device, library or buffer for the storm")
            return
        }
        let pipeline = try device.makeComputePipelineState(function: function)

        /// Fires `count` one-thread dispatches, packed `opsPerBuffer` per
        /// command buffer, and waits for the last buffer. The wait is what puts
        /// the whole storm on the caller's critical path, the way a dependent
        /// MLX chain is.
        func metalStorm(count: Int, opsPerBuffer: Int, barrier: Bool) -> Int {
            var remaining = count
            var buffers = 0
            var last: MTLCommandBuffer?
            while remaining > 0 {
                let batch = min(remaining, opsPerBuffer)
                remaining -= batch
                guard let buffer = queue.makeCommandBuffer(),
                    let encoder = buffer.makeComputeCommandEncoder()
                else { return buffers }
                encoder.setComputePipelineState(pipeline)
                encoder.setBuffer(scratch, offset: 0, index: 0)
                for _ in 0 ..< batch {
                    if barrier { encoder.memoryBarrier(scope: .buffers) }
                    encoder.dispatchThreads(
                        MTLSize(width: 1, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
                }
                encoder.endEncoding()
                buffer.commit()
                buffers += 1
                last = buffer
            }
            last?.waitUntilCompleted()
            return buffers
        }

        func mlxStorm(count: Int) {
            var chained = MLXArray([Float(0)])
            eval(chained)
            for _ in 0 ..< count { chained = chained + Float(1) }
            eval(chained)
        }

        let repeats = 7
        var points: [StormPoint] = []

        func measure(
            mode: String, opsPerBuffer: Int, dispatches: Int,
            body: () -> Int
        ) {
            _ = body()  // warm the pipeline, the queue and the allocator
            var samples: [Double] = []
            var buffers = 0
            for _ in 0 ..< repeats {
                let started = DispatchTime.now().uptimeNanoseconds
                buffers = body()
                let finished = DispatchTime.now().uptimeNanoseconds
                samples.append(Double(finished - started) / 1e9)
            }
            let mean = samples.reduce(0, +) / Double(samples.count)
            points.append(
                StormPoint(
                    mode: mode, opsPerBuffer: opsPerBuffer,
                    dispatches: dispatches, repeats: repeats,
                    meanSeconds: mean, minSeconds: samples.min() ?? mean,
                    microsecondsPerDispatch: mean * 1e6 / Double(dispatches),
                    commandBuffers: buffers))
        }

        // Sweep the dispatch count at a fixed pack size to separate the fixed
        // cost of one storm from its per-dispatch slope, then sweep the pack
        // size at a fixed count to price a command-buffer submission.
        for dispatches in [64, 256, 1024, 4096] {
            measure(mode: "metal", opsPerBuffer: 64, dispatches: dispatches) {
                metalStorm(count: dispatches, opsPerBuffer: 64, barrier: false)
            }
        }
        for opsPerBuffer in [1, 8, 32, 50, 64, 128, 256, 1024, 4096] {
            measure(mode: "metal_pack", opsPerBuffer: opsPerBuffer, dispatches: 4096) {
                metalStorm(count: 4096, opsPerBuffer: opsPerBuffer, barrier: false)
            }
        }
        for opsPerBuffer in [64, 4096] {
            measure(mode: "metal_barrier", opsPerBuffer: opsPerBuffer, dispatches: 4096) {
                metalStorm(count: 4096, opsPerBuffer: opsPerBuffer, barrier: true)
            }
        }
        for dispatches in [64, 256, 1024, 4096] {
            measure(mode: "mlx", opsPerBuffer: -1, dispatches: dispatches) {
                mlxStorm(count: dispatches)
                return -1
            }
        }

        // Two-parameter fit over the pack sweep: seconds = a * dispatches +
        // b * buffers. With `dispatches` fixed at 4096 the sweep isolates `b`.
        var perBuffer: Double?
        var perDispatch: Double?
        let pack = points.filter { $0.mode == "metal_pack" && $0.commandBuffers > 0 }
        if pack.count >= 2 {
            let xs = pack.map { Double($0.commandBuffers) }
            let ys = pack.map(\.meanSeconds)
            let n = Double(xs.count)
            let meanX = xs.reduce(0, +) / n
            let meanY = ys.reduce(0, +) / n
            var sxy = 0.0
            var sxx = 0.0
            for (x, y) in zip(xs, ys) {
                sxy += (x - meanX) * (y - meanY)
                sxx += (x - meanX) * (x - meanX)
            }
            if sxx > 0 {
                let slope = sxy / sxx
                let intercept = meanY - slope * meanX
                perBuffer = slope * 1e6
                perDispatch = intercept * 1e6 / 4096.0
            }
        }

        let report = StormReport(
            host: ProcessInfo.processInfo.hostName,
            device: device.name,
            mlxMaxOpsPerBuffer: ProcessInfo.processInfo
                .environment["MLX_MAX_OPS_PER_BUFFER"] ?? "<unset>",
            mlxMaxMBPerBuffer: ProcessInfo.processInfo
                .environment["MLX_MAX_MB_PER_BUFFER"] ?? "<unset>",
            points: points,
            metalFixedCostMicroseconds: nil,
            metalPerDispatchMicroseconds: perDispatch,
            metalPerBufferMicroseconds: perBuffer)

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(report)
        if let path = ProcessInfo.processInfo
            .environment["MLXFAST_E58_DISPATCH_STORM_OUT"], !path.isEmpty
        {
            try data.write(to: URL(fileURLWithPath: path))
        }
        FileHandle.standardError.write(data)
        FileHandle.standardError.write(Data("\n".utf8))

        #expect(!points.isEmpty)
    }
}
