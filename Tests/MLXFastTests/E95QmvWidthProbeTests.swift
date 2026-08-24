import Foundation
import MLX
import Testing

// E95 -- is the per-weight-pass term `b` of the verify width model DRAM
// traffic, or replicated dequantisation work?
//
// The width model `verify_us = a + b*G + c*M` reads `b = 27,377 us` as one
// pass over 14,412 MB, which is 526 GB/s. That is about twice the DRAM read
// rate this chip is reported to reach, so `b` cannot be one pure DRAM pass.
//
// This probe removes the model, the fixture and the worker. It calls the
// affine-4 group-64 quantized matmul directly on ONE tensor and sweeps the
// input width M, so `G = ceil(M / IPG)` steps from 1 to 2 and then to 3 while
// the bytes on disk never change.
//
// Every cell is one blocking `eval`, so every cell carries the same fixed
// host-plus-launch overhead. Three measurements make the arithmetic
// identified instead of merely consistent:
//
//   * `evalOverheadMicroseconds` times a 16-element elementwise op, so the
//     fixed per-eval cost is measured directly and is not solved for. A
//     two-point solve over two tensors cannot separate that overhead from a
//     cache-resident read rate.
//   * A reduction over each packed tensor gives the achieved read rate of
//     that exact working set, so the comparison needs no external bandwidth
//     figure.
//   * Four working sets from 2.9 MB to 100.3 MB show whether a small set is
//     cache resident on this host at all.
//
// The width sweep runs forward and then in reverse within one session, so
// monotone drift shows up as a forward-to-reverse gap instead of hiding in
// the slope.
//
// Research instrument. Off unless `MLXFAST_RUN_E95_QMV_PROBE=1`.
// `Tests/` is never packaged into a submission. This is a within-session
// relative measurement, it runs under no thermal gate and it reports no
// score.

private struct E95QmvCell {
    var outputs: Int
    var width: Int
    var forwardMicroseconds: Double
    var reverseMicroseconds: Double
    var replicates: Int
}

@Suite("E95 qmv width probe")
struct E95QmvWidthProbeTests {
    static let enabled = ProcessInfo.processInfo.environment[
        "MLXFAST_RUN_E95_QMV_PROBE"] == "1"

    static let hidden = 5120
    static let groupSize = 64
    static let bits = 4
    static let readOutputs = [34816, 16384, 4096, 1024]
    static let sweepOutputs = [34816, 4096]
    static let widths = [1, 2, 3, 4, 5, 6, 8, 9]

    /// affine 4-bit group-64: 4 bits per weight plus a bf16 scale and bias.
    private static func packedBytes(outputs: Int) -> Int {
        outputs * hidden / 2 + 4 * (outputs * hidden / groupSize)
    }

    private static func replicates(outputs: Int) -> Int {
        outputs > 10000 ? 200 : 800
    }

    private static func timed(_ count: Int, _ body: () -> MLXArray) -> Double {
        let start = DispatchTime.now().uptimeNanoseconds
        for _ in 0 ..< count {
            eval(body())
        }
        return Double(DispatchTime.now().uptimeNanoseconds - start) / 1e3
            / Double(count)
    }

    /// The floor every other cell in this probe also pays: one blocking
    /// `eval` of an op whose GPU work is negligible.
    private static func evalOverheadMicroseconds() -> Double {
        let tiny = MLXArray(Array(repeating: Float(1), count: 16))
        eval(tiny)
        _ = timed(200) { tiny + 1 }
        return timed(4000) { tiny + 1 }
    }

    @Test(
        "the G step costs far less than a second pass over the same bytes",
        .enabled(if: E95QmvWidthProbeTests.enabled))
    func widthStep() throws {
        let overhead = Self.evalOverheadMicroseconds()
        print(String(format: "E95_QMV overhead us=%.3f", overhead))

        var readMicroseconds: [Int: Double] = [:]
        for outputs in Self.readOutputs {
            let dense = MLXRandom.normal([outputs, Self.hidden]).asType(.bfloat16)
            let (packed, _, _) = quantized(
                dense, groupSize: Self.groupSize, bits: Self.bits, mode: .affine)
            eval(packed)
            let bytes = Self.packedBytes(outputs: outputs)
            let count = Self.replicates(outputs: outputs)
            _ = Self.timed(5) { packed.sum() }
            let raw = Self.timed(count) { packed.sum() }
            readMicroseconds[outputs] = raw
            print(
                "E95_QMV read outputs=\(outputs) bytes=\(bytes) "
                    + String(
                        format: "raw_us=%.3f net_us=%.3f net_gb_s=%.1f",
                        raw, raw - overhead,
                        Double(bytes) / (raw - overhead) / 1e3))
        }

        var cells: [E95QmvCell] = []
        for outputs in Self.sweepOutputs {
            let dense = MLXRandom.normal([outputs, Self.hidden]).asType(.bfloat16)
            let (packed, scales, biases) = quantized(
                dense, groupSize: Self.groupSize, bits: Self.bits, mode: .affine)
            eval(packed, scales, biases)
            let count = Self.replicates(outputs: outputs)
            let onePass = readMicroseconds[outputs]! - overhead

            var inputs: [Int: MLXArray] = [:]
            for width in Self.widths {
                let x = MLXRandom.normal([width, Self.hidden]).asType(.bfloat16)
                eval(x)
                inputs[width] = x
            }

            func measure(_ width: Int) -> Double {
                let x = inputs[width]!
                let call = {
                    quantizedMM(
                        x, packed, scales: scales, biases: biases,
                        transpose: true, groupSize: Self.groupSize,
                        bits: Self.bits, mode: .affine)
                }
                _ = Self.timed(5, call)
                return Self.timed(count, call)
            }

            let forward = Self.widths.map(measure)
            let reverse = Array(Self.widths.reversed().map(measure).reversed())

            for (index, width) in Self.widths.enumerated() {
                let cell = E95QmvCell(
                    outputs: outputs, width: width,
                    forwardMicroseconds: forward[index],
                    reverseMicroseconds: reverse[index],
                    replicates: count)
                cells.append(cell)
                let mean = (cell.forwardMicroseconds + cell.reverseMicroseconds) / 2
                print(
                    "E95_QMV cell outputs=\(outputs) m=\(width) "
                        + String(
                            format: "fwd_us=%.3f rev_us=%.3f net_us=%.3f "
                                + "net_over_one_pass=%.3f",
                            cell.forwardMicroseconds, cell.reverseMicroseconds,
                            mean - overhead, (mean - overhead) / onePass)
                        + " replicates=\(count)")
            }
        }

        if let path = ProcessInfo.processInfo.environment["MLXFAST_E95_QMV_OUT"],
            !path.isEmpty
        {
            let payload: [String: Any] = [
                "hidden": Self.hidden,
                "group_size": Self.groupSize,
                "bits": Self.bits,
                "eval_overhead_us": overhead,
                "reads": Self.readOutputs.map {
                    [
                        "outputs": $0,
                        "packed_bytes": Self.packedBytes(outputs: $0),
                        "raw_us": readMicroseconds[$0]!,
                    ]
                },
                "cells": cells.map {
                    [
                        "outputs": $0.outputs,
                        "packed_bytes": Self.packedBytes(outputs: $0.outputs),
                        "m": $0.width,
                        "forward_us": $0.forwardMicroseconds,
                        "reverse_us": $0.reverseMicroseconds,
                        "replicates": $0.replicates,
                    ]
                },
            ]
            let data = try JSONSerialization.data(
                withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
            try data.write(to: URL(fileURLWithPath: path))
        }

        #expect(!cells.isEmpty)
    }
}
