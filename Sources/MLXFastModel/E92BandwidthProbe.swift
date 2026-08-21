import Foundation
import MLX

// E92 RESEARCH INSTRUMENT -- in-session read-bandwidth residency sweep. It runs
// ONCE per process, inside the worker that owns the decode path, after the seed
// prefill has been evaluated. The model is therefore resident and the MLX
// allocator is in steady state, which is the whole point: E87 measured a
// standalone MLX bench inflating the dispatch boundary by 25 times against the
// in-session value, so a standalone roofline cannot price a production read.
//
// The probe does not time itself with a wall clock. It writes the absolute
// mach-uptime window of each repetition and leaves attribution to the E90
// command-buffer ledger, which records the GPU execution interval of every
// buffer this process submits. GPU busy inside the window is device time; the
// wall figure beside it is only a sanity bound.
//
// Off unless `MLX_E92_BANDWIDTH=1`. The `MLX_` prefix is load-bearing:
// `sanitizedRuntimeWorkerEnvironment` admits `MLX_` and drops `MLXFAST_`.
//
// RESEARCH ONLY. This file must not reach a submission.
public enum E92BandwidthProbe {
    private static let environment = ProcessInfo.processInfo.environment

    public static let enabled = environment["MLX_E92_BANDWIDTH"] == "1"

    /// Megabyte sizes of the streamed read, smallest first.
    private static let sizesMB: [Int] = {
        let raw = environment["MLX_E92_BANDWIDTH_SIZES_MB"] ?? "16,64,157,330,428,1024"
        return raw.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
    }()

    /// Size of the single bf16 comparison read, in megabytes.
    private static let bf16MB = Int(environment["MLX_E92_BANDWIDTH_BF16_MB"] ?? "428")

    private static let reps = Int(environment["MLX_E92_BANDWIDTH_REPS"] ?? "7") ?? 7

    private static let sink: FileHandle = {
        guard let path = environment["MLX_E92_BANDWIDTH_PATH"], !path.isEmpty
        else { return FileHandle.standardError }
        let fd = open(path, O_WRONLY | O_CREAT | O_APPEND, 0o644)
        guard fd >= 0 else { return FileHandle.standardError }
        return FileHandle(fileDescriptor: fd, closeOnDealloc: false)
    }()

    private static let lock = NSLock()
    nonisolated(unsafe) private static var fired = false

    /// Runs the sweep once. Safe to call from every round; later calls return.
    public static func runOnceIfRequested() {
        guard enabled else { return }
        lock.lock()
        if fired {
            lock.unlock()
            return
        }
        fired = true
        lock.unlock()

        for megabytes in sizesMB {
            sweepOne(kind: "int32", dtype: .int32, elementBytes: 4, megabytes: megabytes)
        }
        if let bf16MB {
            sweepOne(kind: "bfloat16", dtype: .bfloat16, elementBytes: 2, megabytes: bf16MB)
        }
    }

    /// One buffer size: allocate, materialise, warm the reduce kernel for this
    /// shape, then repeat a full-tensor reduction and record each window.
    private static func sweepOne(
        kind: String, dtype: DType, elementBytes: Int, megabytes: Int
    ) {
        let bytes = megabytes * 1024 * 1024
        let elements = bytes / elementBytes
        // `zeros + 1` materialises a real buffer rather than a lazy fill, so
        // the reduction below reads DRAM instead of a broadcast scalar.
        let buffer = zeros([elements], dtype: dtype) + 1
        eval(buffer)
        // Warm: the first reduction at a new shape pays a JIT and a first-call
        // allocation that no later repetition pays.
        eval(sum(buffer))

        for rep in 0 ..< reps {
            let t0 = DispatchTime.now().uptimeNanoseconds
            eval(sum(buffer))
            let t1 = DispatchTime.now().uptimeNanoseconds
            write([
                "\"event\":\"e92_bw\"",
                "\"pid\":\(ProcessInfo.processInfo.processIdentifier)",
                "\"kind\":\"\(kind)\"",
                "\"megabytes\":\(megabytes)",
                "\"bytes\":\(bytes)",
                "\"elements\":\(elements)",
                "\"rep\":\(rep)",
                "\"t0_ns\":\(t0)",
                "\"t1_ns\":\(t1)",
                "\"wall_us\":\((t1 - t0) / 1000)",
            ])
        }
    }

    private static func write(_ fields: [String]) {
        let line = "{" + fields.joined(separator: ",") + "}\n"
        guard let data = line.data(using: .utf8) else { return }
        lock.lock()
        defer { lock.unlock() }
        try? sink.write(contentsOf: data)
    }
}
