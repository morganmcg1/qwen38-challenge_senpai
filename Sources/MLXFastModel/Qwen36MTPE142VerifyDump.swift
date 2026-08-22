import Foundation
import MLX

/// E142 RESEARCH INSTRUMENT -- streams every post-norm hidden row that reaches
/// the SCORED verify readout, together with the verify input tokens and the
/// exact top-2 evidence the shipped dense readout produced for that row, so an
/// offline harness can measure certified survival fractions for a screened
/// readout without a timed decode.
///
/// This file carries a new symbol only. It is reverted out of the submitted
/// surface before the experiment closes; the shipped default is a no-op.
///
/// The `MLX_` prefix is load-bearing: `sanitizedRuntimeWorkerEnvironment`
/// admits `MLX_` and drops `MLXFAST_`, so a `MLXFAST_`-spelled gate would never
/// reach the worker process that owns the verify path.
///
/// `asArray` forces a host readback per round. That is why this instrument may
/// only run on the untimed `mtp-verify` path.
enum E142VerifyDump {
    private static let prefix =
        ProcessInfo.processInfo.environment["MLX_E142_VERIFY_DUMP"]
    private static let lock = NSLock()
    // `lock` is the external synchronization the concurrency checker asks for.
    nonisolated(unsafe) private static var metaFile: FileHandle?
    nonisolated(unsafe) private static var tokenFile: FileHandle?
    nonisolated(unsafe) private static var top2IDFile: FileHandle?
    nonisolated(unsafe) private static var top2ValueFile: FileHandle?
    nonisolated(unsafe) private static var hiddenFile: FileHandle?

    static func record(
        verifyTokens: MLXArray,
        normed: MLXArray?,
        top2IDs: MLXArray,
        top2Values: MLXArray
    ) {
        guard let prefix, let normed else { return }
        lock.lock()
        defer { lock.unlock() }
        if metaFile == nil {
            // The CLI and the worker it spawns both inherit the gate, and both
            // reach this readout. One shared path means whichever opens second
            // truncates the other's samples, so shard by process.
            let shard = "\(prefix).pid\(ProcessInfo.processInfo.processIdentifier)"
            let manager = FileManager.default
            for suffix in [".meta.i32", ".tok.i32", ".top2.i32", ".top2.f32", ".x.f32"] {
                manager.createFile(atPath: shard + suffix, contents: nil)
            }
            metaFile = FileHandle(forWritingAtPath: shard + ".meta.i32")
            tokenFile = FileHandle(forWritingAtPath: shard + ".tok.i32")
            top2IDFile = FileHandle(forWritingAtPath: shard + ".top2.i32")
            top2ValueFile = FileHandle(forWritingAtPath: shard + ".top2.f32")
            hiddenFile = FileHandle(forWritingAtPath: shard + ".x.f32")
        }
        let rows = normed.dim(normed.ndim - 2)
        write(metaFile, [Int32(rows)])
        write(tokenFile, flatInt32(verifyTokens))
        write(top2IDFile, flatInt32(top2IDs))
        write(top2ValueFile, flatFloat(top2Values))
        write(hiddenFile, flatFloat(normed))
    }

    private static func flatInt32(_ array: MLXArray) -> [Int32] {
        array.reshaped([array.size]).asType(.int32).asArray(Int32.self)
    }

    private static func flatFloat(_ array: MLXArray) -> [Float] {
        array.reshaped([array.size]).asType(.float32).asArray(Float.self)
    }

    private static func write<T>(_ handle: FileHandle?, _ values: [T]) {
        values.withUnsafeBufferPointer { try? handle?.write(contentsOf: Data(buffer: $0)) }
    }
}
