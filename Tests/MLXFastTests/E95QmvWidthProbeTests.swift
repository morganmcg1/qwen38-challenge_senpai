import Foundation
import MLX
import Testing

// E95 -- is the per-weight-pass term `b` of the verify width model DRAM
// traffic, or replicated dequantisation work?
//
// The width model `verify_us = a + b*G + c*M` reads `b = 27,377 us` as one
// pass over 14,412 MB, which is 526 GB/s. That is about twice the DRAM read
// rate this host reaches for working sets of 64 MB to 1024 MB, so `b` cannot
// be pure DRAM traffic.
//
// This probe removes the model, the fixture and the worker. It calls the
// affine-4 group-64 quantized matmul directly on ONE tensor and sweeps the
// input width M, so `G = ceil(M / IPG)` steps from 1 to 2 while the bytes on
// disk never change. Two tensors answer the question together:
//
//   * K=5120, O=34816, 100.3 MB -- the MLP gate_up shape, far above cache.
//   * K=5120, O= 4096,  11.8 MB -- cache resident on this host.
//
// If the G step costs a full second DRAM pass on the large tensor, `b` is
// bytes. If the per-byte cost of the G step is the same on both tensors, `b`
// cannot be DRAM traffic, because the two working sets run at very different
// DRAM rates.
//
// A read-only reduction over the same packed tensor gives the achieved read
// rate for that exact working set on this host, so the comparison needs no
// external bandwidth figure.
//
// Research instrument. Off unless `MLXFAST_RUN_E95_QMV_PROBE=1`.
// `Tests/` is never packaged into a submission. Untimed by the campaign's
// thermal gate: this is a within-session relative measurement and it reports
// no score.

private struct E95QmvCell {
    var outputs: Int
    var width: Int
    var microseconds: Double
    var replicates: Int
}

@Suite("E95 qmv width probe")
struct E95QmvWidthProbeTests {
    static let enabled = ProcessInfo.processInfo.environment[
        "MLXFAST_RUN_E95_QMV_PROBE"] == "1"

    static let hidden = 5120
    static let groupSize = 64
    static let bits = 4

    /// affine 4-bit group-64: 4 bits per weight plus a bf16 scale and bias.
    private static func packedBytes(outputs: Int) -> Int {
        outputs * hidden / 2 + 4 * (outputs * hidden / groupSize)
    }

    private static func timed(_ replicates: Int, _ body: (Int) -> MLXArray)
        -> Double
    {
        let start = DispatchTime.now().uptimeNanoseconds
        for index in 0 ..< replicates {
            eval(body(index))
        }
        return Double(DispatchTime.now().uptimeNanoseconds - start) / 1e3
            / Double(replicates)
    }

    @Test("the G step costs the same per byte on a cached and an uncached pack")
    func widthStep() throws {
        try #require(Self.enabled)

        let widths = [1, 2, 3, 4, 5, 6, 8, 9]
        var cells: [E95QmvCell] = []
        var readRates: [Int: Double] = [:]

        for outputs in [34816, 4096] {
            let dense = MLXRandom.normal([outputs, Self.hidden]).asType(.bfloat16)
            let (packed, scales, biases) = quantized(
                dense, groupSize: Self.groupSize, bits: Self.bits, mode: .affine)
            eval(packed, scales, biases)
            let bytes = Self.packedBytes(outputs: outputs)
            let replicates = outputs > 10000 ? 200 : 800

            // Achieved read rate for this exact working set. A reduction over
            // the packed words touches every byte the matmul must also read.
            _ = Self.timed(5) { _ in packed.sum() }
            let readUs = Self.timed(replicates) { _ in packed.sum() }
            readRates[outputs] = Double(bytes) / readUs / 1e3
            print(
                "E95_QMV read outputs=\(outputs) bytes=\(bytes) "
                    + "us=\(String(format: "%.3f", readUs)) "
                    + "gb_s=\(String(format: "%.1f", Double(bytes) / readUs / 1e3))")

            for width in widths {
                let x = MLXRandom.normal([width, Self.hidden]).asType(.bfloat16)
                eval(x)
                _ = Self.timed(5) { _ in
                    quantizedMM(
                        x, packed, scales: scales, biases: biases,
                        transpose: true, groupSize: Self.groupSize,
                        bits: Self.bits, mode: .affine)
                }
                let us = Self.timed(replicates) { _ in
                    quantizedMM(
                        x, packed, scales: scales, biases: biases,
                        transpose: true, groupSize: Self.groupSize,
                        bits: Self.bits, mode: .affine)
                }
                cells.append(
                    E95QmvCell(
                        outputs: outputs, width: width, microseconds: us,
                        replicates: replicates))
                print(
                    "E95_QMV cell outputs=\(outputs) m=\(width) "
                        + "us=\(String(format: "%.3f", us)) "
                        + "replicates=\(replicates) "
                        + "gb_s_one_pass="
                        + String(format: "%.1f", Double(bytes) / us / 1e3))
            }
        }

        if let path = ProcessInfo.processInfo.environment["MLXFAST_E95_QMV_OUT"],
            !path.isEmpty
        {
            var rows: [[String: Any]] = []
            for cell in cells {
                rows.append([
                    "outputs": cell.outputs,
                    "packed_bytes": Self.packedBytes(outputs: cell.outputs),
                    "m": cell.width,
                    "microseconds": cell.microseconds,
                    "replicates": cell.replicates,
                ])
            }
            let payload: [String: Any] = [
                "hidden": Self.hidden,
                "group_size": Self.groupSize,
                "bits": Self.bits,
                "read_gb_s": readRates.reduce(into: [String: Double]()) {
                    $0[String($1.key)] = $1.value
                },
                "cells": rows,
            ]
            let data = try JSONSerialization.data(
                withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
            try data.write(to: URL(fileURLWithPath: path))
        }

        #expect(!cells.isEmpty)
    }
}
