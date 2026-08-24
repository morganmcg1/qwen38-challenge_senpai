import Foundation
import MLX
import Testing

// E163 F4 check 2: MEASURE the host's streaming memory bandwidth. Do not quote
// the datasheet.
//
// The advisor prices the scored decode round against a 273 GB/s M4 Pro peak
// taken from Apple's specification. A specification number is a pin-count
// ceiling, not an attainable rate: no real kernel reaches it, and the fraction
// a kernel does reach is the whole question. This suite measures the attainable
// rate on THIS host with the simplest kernels that exist, so the round's
// achieved bandwidth can be reported against a measured denominator.
//
// WHAT IT MEASURES.
//   read   - `sum` and `max` over one large contiguous buffer. Each touches
//            every byte once and writes one scalar, so traffic is `bytes`.
//   copy   - `buffer * 2`, materialised. Traffic is `2 * bytes`.
// Sizes run past every cache level on the part, so the largest sizes are pure
// DRAM rates.
//
// WHAT IT DOES NOT MEASURE. Nothing is resident, nothing overlaps, and no
// model is loaded. This is an upper bound for a single streaming kernel, which
// is exactly the denominator the round's achieved rate needs. It is not a leg,
// not a score, and not thermally gated.
//
// Enable with `MLXFAST_RUN_E163_BANDWIDTH=1` and point
// `MLXFAST_E163_BW_OUT` at the JSON destination.

private struct E163BandwidthCell {
    let kernel: String
    let dtype: String
    let bytesTouched: Int
    let bufferBytes: Int
    let perCallUs: [Double]
}

@Suite
struct E163MemoryBandwidthTests {
    private static var enabled: Bool {
        ProcessInfo.processInfo.environment["MLXFAST_RUN_E163_BANDWIDTH"] == "1"
    }

    /// 4 MiB of float32 per row keeps every dimension well inside Int32 even
    /// for a 16 GiB buffer, which `mlx_full` requires.
    private static let rowElements = 1 << 20

    @Test(.enabled(if: E163MemoryBandwidthTests.enabled))
    func streamingBandwidthPeak() throws {
        let env = ProcessInfo.processInfo.environment
        let outPath = try #require(
            env["MLXFAST_E163_BW_OUT"],
            "MLXFAST_E163_BW_OUT must name the JSON destination")
        let reps = Int(env["MLXFAST_E163_BW_REPS"] ?? "") ?? 9
        let sizesGiB =
            (env["MLXFAST_E163_BW_SIZES_GIB"]?
                .split(separator: ",")
                .compactMap { Double($0.trimmingCharacters(in: .whitespaces)) })
            .flatMap { $0.isEmpty ? nil : $0 } ?? [0.25, 1, 4, 14]

        var cells: [E163BandwidthCell] = []

        for gib in sizesGiB {
            let rows = max(
                1, Int((gib * 1_073_741_824.0) / (Double(Self.rowElements) * 4.0)))

            for dtype in [DType.float32, DType.bfloat16] {
                let elementSize = dtype == .float32 ? 4 : 2
                let elements = rows * Self.rowElements
                let bufferBytes = elements * elementSize
                let buffer = full(
                    [rows, Self.rowElements], values: MLXArray(Float(1)),
                    dtype: dtype)
                eval(buffer)

                // The read really happens: every element is 1, so the maximum
                // is 1 only if the kernel visited the whole buffer.
                let sentinel = MLX.max(buffer)
                eval(sentinel)
                #expect(sentinel.asType(.float32).item(Float.self) == Float(1))

                func time(_ name: String, _ traffic: Int, _ body: () -> MLXArray) {
                    let warm = body()
                    eval(warm)
                    var perCallUs: [Double] = []
                    for _ in 0..<reps {
                        let start = DispatchTime.now().uptimeNanoseconds
                        let out = body()
                        eval(out)
                        let elapsed = DispatchTime.now().uptimeNanoseconds - start
                        perCallUs.append(Double(elapsed) / 1e3)
                    }
                    cells.append(
                        E163BandwidthCell(
                            kernel: name, dtype: "\(dtype)",
                            bytesTouched: traffic, bufferBytes: bufferBytes,
                            perCallUs: perCallUs))
                }

                time("sum", bufferBytes) { MLX.sum(buffer) }
                time("max", bufferBytes) { MLX.max(buffer) }
                // The copy doubles resident memory, so keep it off the sizes
                // that would approach the working-set limit.
                if bufferBytes <= 4 * 1_073_741_824 {
                    time("scale_copy", 2 * bufferBytes) { buffer * 2 }
                }
            }
            Memory.clearCache()
        }

        let info = GPU.deviceInfo()
        var cellPayload: [[String: Any]] = []
        for cell in cells {
            let sorted = cell.perCallUs.sorted()
            let best = sorted.first!
            let median = sorted[sorted.count / 2]
            cellPayload.append([
                "kernel": cell.kernel,
                "dtype": cell.dtype,
                "buffer_bytes": cell.bufferBytes,
                "buffer_gib": Double(cell.bufferBytes) / 1_073_741_824.0,
                "bytes_touched": cell.bytesTouched,
                "us_min": best,
                "us_median": median,
                "us_max": sorted.last!,
                "gb_per_s_best": Double(cell.bytesTouched) / (best * 1e3),
                "gb_per_s_median": Double(cell.bytesTouched) / (median * 1e3),
                "us_all": cell.perCallUs,
            ])
        }
        let bestOverall = cellPayload.compactMap { $0["gb_per_s_best"] as? Double }.max() ?? 0

        let payload: [String: Any] = [
            "harness": "local",
            "measurement": "isolated streaming bandwidth, no model resident",
            "timing_valid_as_leg": false,
            "official_or_ranked_score": false,
            "cool_gate_passed_real_gate": false,
            "gate_qualified_for_timing": false,
            "reps": reps,
            "sizes_gib": sizesGiB,
            "device_architecture": info.architecture,
            "device_memory_size_bytes": info.memorySize,
            "device_max_recommended_working_set_bytes":
                Int(info.maxRecommendedWorkingSetSize),
            "measured_peak_gb_per_s": bestOverall,
            "cells": cellPayload,
        ]
        let data = try JSONSerialization.data(
            withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: URL(fileURLWithPath: outPath))
    }
}
