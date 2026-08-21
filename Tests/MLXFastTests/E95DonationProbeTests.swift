import Cmlx
import Foundation
import MLX
import MLXLMCommon
import Testing

// E95 rung 1a -- WHY does `KVCacheSimple.update`'s `slice_update` fail to
// donate its input buffer in the proposal-head chain, when the same class and
// the same line donate in the target's full-attention layers?
//
// E93 measured two `vn_copy` dispatches over the whole capacity-sized head K
// and V per MARGINAL draft step, and none on the first head call of a round.
// This probe reproduces that pattern outside the worker with no model, no
// weights and no head: one `KVCacheSimple`, N one-row updates, and a consumer
// that reads the returned window exactly as SDPA does.
//
// Detectors, both independent of wall-clock noise:
//   * buffer identity -- `mlx_array_data_float32` on the cache's own array
//     before and after the chain. A donated `slice_update` writes in place and
//     the pointer never moves.
//   * peak active memory -- a copy needs the input and the output resident at
//     the same time, so peak rises by one capacity per copy still in flight.
//
// The four arms isolate the blocking reference:
//   lazy_read       chained, read consumed, one eval at the end (head chain)
//   lazy_no_read    chained, returned window DISCARDED  (appendHistoryKV)
//   async_read      chained, read consumed, asyncEval per step
//   sync_read       chained, read consumed, blocking eval per step
//
// Research instrument. Off unless `MLXFAST_RUN_E95_DONATION_PROBE=1`.
// `Tests/` is never packaged into a submission.

private enum E95Submission {
    case lazy
    case asyncEach
    case syncEach
}

private struct E95ProbeResult {
    var arm: String
    var bufferMoved: Bool
    var peakBytes: Int
    var seconds: Double
}

@Suite("E95 KVCacheSimple donation probe")
struct E95DonationProbeTests {
    static let enabled = ProcessInfo.processInfo.environment[
        "MLXFAST_RUN_E95_DONATION_PROBE"] == "1"

    /// Data pointer of an evaluated array. Equal pointers across a
    /// `slice_update` mean the input buffer was donated and written in place.
    private static func dataPointer(_ array: MLXArray) -> UInt {
        eval(array)
        guard let raw = mlx_array_data_float32(array.ctx) else { return 0 }
        return UInt(bitPattern: UnsafeRawPointer(raw))
    }

    private static func run(
        arm: String, steps: Int, capacityRows: Int, holdRead: Bool,
        submission: E95Submission
    ) -> E95ProbeResult {
        let cache = KVCacheSimple()
        cache.step = capacityRows

        var carry = MLXArray.zeros([1, 4, 1, 256], dtype: .float32)
        eval(carry)

        // First update allocates the capacity array. Every later update is the
        // one under test, so the reference pointer is taken after it.
        let (k0, v0) = (carry + Float(1), carry + Float(2))
        _ = cache.update(keys: k0, values: v0)
        eval(cache.innerState())
        let pointerBefore = dataPointer(cache.innerState()[0])

        Memory.peakMemory = 0
        let start = DispatchTime.now().uptimeNanoseconds
        for step in 1 ... steps {
            let k = carry + Float(step)
            let v = carry * Float(2)
            let (readKeys, readValues) = cache.update(keys: k, values: v)
            if holdRead {
                carry =
                    (readKeys.sum(axis: 2, keepDims: true)
                        + readValues.sum(axis: 2, keepDims: true)) * Float(1e-6)
            } else {
                carry = (k + v) * Float(1e-6)
            }
            switch submission {
            case .lazy: break
            case .asyncEach: asyncEval(carry)
            case .syncEach: eval(carry)
            }
        }
        eval(carry)
        let seconds = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
        let peak = Memory.peakMemory
        let pointerAfter = dataPointer(cache.innerState()[0])

        return E95ProbeResult(
            arm: arm, bufferMoved: pointerAfter != pointerBefore,
            peakBytes: peak, seconds: seconds)
    }

    @Test("head chain and target pattern donate differently")
    func donationPattern() throws {
        try #require(Self.enabled)

        let capacityRows = 4096
        let steps = 16
        let capacityBytes = 1 * 4 * capacityRows * 256 * 4

        let arms: [E95ProbeResult] = [
            Self.run(
                arm: "lazy_read", steps: steps, capacityRows: capacityRows,
                holdRead: true, submission: .lazy),
            Self.run(
                arm: "lazy_no_read", steps: steps, capacityRows: capacityRows,
                holdRead: false, submission: .lazy),
            Self.run(
                arm: "async_read", steps: steps, capacityRows: capacityRows,
                holdRead: true, submission: .asyncEach),
            Self.run(
                arm: "sync_read", steps: steps, capacityRows: capacityRows,
                holdRead: true, submission: .syncEach),
            Self.run(
                arm: "sync_no_read", steps: steps, capacityRows: capacityRows,
                holdRead: false, submission: .syncEach),
        ]

        print("E95_PROBE capacity_bytes=\(capacityBytes) steps=\(steps)")
        for arm in arms {
            let excess = Double(arm.peakBytes) / Double(2 * capacityBytes)
            print(
                "E95_PROBE arm=\(arm.arm) buffer_moved=\(arm.bufferMoved) "
                    + "peak_bytes=\(arm.peakBytes) peak_over_kv=\(String(format: "%.2f", excess)) "
                    + "seconds=\(String(format: "%.4f", arm.seconds))")
        }
    }
}
