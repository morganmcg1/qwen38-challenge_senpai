import Foundation
import MLX
import Testing

// E97 rung 1 -- what does a marginal verify row cost, and is that cost
// dequantisation?
//
// The E95 width model prices a verify row at 10,268 us over the whole trunk.
// The ranked cost curve prices the same row at 6,705 us on M5. Neither number
// says WHAT the row buys. This probe isolates one weight tensor, removes the
// model, the fixture and the worker, and measures the marginal cost of one
// input row three ways at the same [K, N]:
//
//   * affine-4 group-64 `quantizedMM`, the scored form;
//   * unquantized bf16 `matmul`, which does the same multiply-accumulates and
//     reads 4x the weight bytes but unpacks nothing;
//   * both again above `get_qmv_batch_limit` (10 at K = 5120), where MLX leaves
//     the cross-row vector kernel and enters the split-K matrix kernel.
//
// Weight traffic is amortised across the input rows of one dispatch in every
// arm, so the SLOPE in M isolates per-row work and the intercept carries the
// weight stream. Intercepts are not comparable between the arms; slopes are.
//
// Design:
//   * every cell is one blocking `eval`, so the fixed per-eval host cost is a
//     constant in M and cannot bias a slope;
//   * `evalOverheadMicroseconds` measures that constant anyway, because the
//     implied FLOP rate needs the intercept;
//   * replicate counts are chosen per cell from a calibration pass so every
//     cell integrates a similar wall time;
//   * blocks alternate ascending and descending M, so monotone thermal drift
//     enters the slope with cancelling sign;
//   * the two kernels are measured back to back at each width inside a block,
//     so any residual drift is common to both arms;
//   * an A/A null repeats one cell at the start and the end of the session.
//
// Research instrument. Off unless `MLXFAST_RUN_E97_ROW_COST=1`.
// `Tests/` is never packaged into a submission. Within-session relative
// measurement, no thermal gate, no score.

private struct E97Sample {
    var kernel: String
    var outputs: Int
    var width: Int
    var block: Int
    var ascending: Bool
    var microseconds: Double
    var replicates: Int
}

@Suite("E97 verify row cost")
struct E97VerifyRowCostTests {
    static let enabled = ProcessInfo.processInfo.environment[
        "MLXFAST_RUN_E97_ROW_COST"] == "1"
    static let peakEnabled = ProcessInfo.processInfo.environment[
        "MLXFAST_RUN_E97_PEAK"] == "1"

    static let hidden = 5120
    static let groupSize = 64
    static let bits = 4
    /// `mlp.gate_up` and the `lm_head` readout: both take the WIDE affine-4
    /// branch at every M <= 9, and both are scored shapes.
    static let shapes = [34816, 248320]
    /// `get_qmv_batch_limit(5120, N) == 10` on gen 16 and 17, so 1...9 is the
    /// cross-row vector regime and 10... is the split-K matrix regime.
    static let vectorWidths = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    static let matrixWidths = [10, 12, 16, 24, 32]
    static let blocks = 8
    /// Wall time each cell should integrate, before the fixed per-eval cost.
    static let targetCellMicroseconds = 120_000.0

    private static func packedBytes(outputs: Int) -> Int {
        outputs * hidden / 2 + 4 * (outputs * hidden / groupSize)
    }

    private static func denseBytes(outputs: Int) -> Int {
        outputs * hidden * 2
    }

    private static func timed(_ count: Int, _ body: () -> MLXArray) -> Double {
        let start = DispatchTime.now().uptimeNanoseconds
        for _ in 0 ..< count {
            eval(body())
        }
        return Double(DispatchTime.now().uptimeNanoseconds - start) / 1e3
            / Double(count)
    }

    /// The floor every other cell also pays: one blocking `eval` of an op whose
    /// GPU work is negligible.
    private static func evalOverheadMicroseconds() -> Double {
        let tiny = MLXArray(Array(repeating: Float(1), count: 16))
        eval(tiny)
        _ = timed(200) { tiny + 1 }
        return timed(4000) { tiny + 1 }
    }

    private static func replicates(for body: () -> MLXArray) -> Int {
        _ = timed(2, body)
        let rough = timed(3, body)
        let count = Int((targetCellMicroseconds / max(rough, 1.0)).rounded())
        return min(max(count, 4), 500)
    }

    /// Rung 0. The achievable arithmetic ceiling of this GPU through the same
    /// library the scored path uses, so the per-row slope can be read as a
    /// fraction of something measured rather than of a specification number.
    @Test(
        "achievable arithmetic peak of this GPU",
        .enabled(if: E97VerifyRowCostTests.peakEnabled))
    func achievablePeak() throws {
        let overhead = Self.evalOverheadMicroseconds()
        print(String(format: "E97_PEAK overhead us=%.3f", overhead))

        var records: [[String: Any]] = []

        func record(
            _ label: String, _ dtype: String, _ flops: Double,
            _ shape: [Int], _ body: () -> MLXArray
        ) {
            _ = Self.timed(2, body)
            let rough = Self.timed(2, body)
            let reps = min(max(Int((Self.targetCellMicroseconds / rough).rounded()), 3), 200)
            let us = Self.timed(reps, body)
            let net = us - overhead
            let tflops = flops / (net * 1e-6) / 1e12
            records.append([
                "label": label, "dtype": dtype, "shape": shape,
                "flops": flops, "us": us, "net_us": net,
                "tflop_per_s": tflops, "replicates": reps,
            ])
            print(
                "E97_PEAK \(label) dtype=\(dtype) shape=\(shape) "
                    + String(
                        format: "net_us=%.1f tflop_s=%.3f", net, tflops)
                    + " replicates=\(reps)")
        }

        for size in [2048, 4096, 8192] {
            for (name, dtype) in [
                ("bfloat16", DType.bfloat16), ("float16", DType.float16),
                ("float32", DType.float32),
            ] {
                let a = MLXRandom.normal([size, size], dtype: dtype)
                let b = MLXRandom.normal([size, size], dtype: dtype)
                eval(a, b)
                record(
                    "square_matmul", name, 2.0 * Double(size) * Double(size)
                        * Double(size), [size, size, size]
                ) { matmul(a, b) }
            }
        }

        // The same ceiling question for the scored quantized shape: a wide
        // batch through the split-K matrix kernel, where the weight stream is
        // amortised over many rows.
        for outputs in Self.shapes {
            let dense = MLXRandom.normal([outputs, Self.hidden], dtype: .bfloat16)
            eval(dense)
            let (packed, scales, biases) = quantized(
                dense, groupSize: Self.groupSize, bits: Self.bits, mode: .affine)
            eval(packed, scales, biases)
            for rows in [256, 1024] {
                let x = MLXRandom.normal([rows, Self.hidden], dtype: .bfloat16)
                eval(x)
                let flops = 2.0 * Double(rows) * Double(Self.hidden)
                    * Double(outputs)
                record(
                    "affine4_batch", "bfloat16", flops,
                    [rows, Self.hidden, outputs]
                ) {
                    quantizedMM(
                        x, packed, scales: scales, biases: biases,
                        transpose: true, groupSize: Self.groupSize,
                        bits: Self.bits, mode: .affine)
                }
                record(
                    "bf16_batch", "bfloat16", flops, [rows, Self.hidden, outputs]
                ) { matmul(x, dense.T) }
            }
        }

        if let path = ProcessInfo.processInfo.environment["MLXFAST_E97_PEAK_OUT"],
            !path.isEmpty
        {
            let data = try JSONSerialization.data(
                withJSONObject: [
                    "eval_overhead_us": overhead, "records": records,
                ], options: [.prettyPrinted, .sortedKeys])
            try data.write(to: URL(fileURLWithPath: path))
        }

        #expect(!records.isEmpty)
    }

    @Test(
        "the marginal verify row costs the same quantized or bf16",
        .enabled(if: E97VerifyRowCostTests.enabled))
    func perRowSlope() throws {
        let overhead = Self.evalOverheadMicroseconds()
        print(String(format: "E97_ROW overhead us=%.3f", overhead))

        var samples: [E97Sample] = []
        var nulls: [[String: Any]] = []
        var reads: [[String: Any]] = []

        for outputs in Self.shapes {
            let dense = MLXRandom.normal([outputs, Self.hidden], dtype: .bfloat16)
            eval(dense)
            let (packed, scales, biases) = quantized(
                dense, groupSize: Self.groupSize, bits: Self.bits, mode: .affine)
            eval(packed, scales, biases)

            // The achieved read rate of this exact working set, so no external
            // bandwidth constant is needed to read the intercepts.
            for (name, tensor, bytes) in [
                ("affine4", packed, Self.packedBytes(outputs: outputs)),
                ("bf16", dense, Self.denseBytes(outputs: outputs)),
            ] as [(String, MLXArray, Int)] {
                _ = Self.timed(3) { tensor.sum() }
                let raw = Self.timed(30) { tensor.sum() }
                reads.append([
                    "kernel": name, "outputs": outputs, "bytes": bytes,
                    "raw_us": raw, "net_us": raw - overhead,
                    "gb_per_s": Double(bytes) / (raw - overhead) / 1e3,
                ])
                print(
                    "E97_ROW read kernel=\(name) outputs=\(outputs) bytes=\(bytes) "
                        + String(
                            format: "raw_us=%.3f net_gb_s=%.1f", raw,
                            Double(bytes) / (raw - overhead) / 1e3))
            }

            let widths = Self.vectorWidths + Self.matrixWidths
            var inputs: [Int: MLXArray] = [:]
            for width in widths {
                let x = MLXRandom.normal([width, Self.hidden], dtype: .bfloat16)
                eval(x)
                inputs[width] = x
            }

            func affine4(_ width: Int) -> () -> MLXArray {
                let x = inputs[width]!
                return {
                    quantizedMM(
                        x, packed, scales: scales, biases: biases,
                        transpose: true, groupSize: Self.groupSize,
                        bits: Self.bits, mode: .affine)
                }
            }

            func bf16(_ width: Int) -> () -> MLXArray {
                let x = inputs[width]!
                return { matmul(x, dense.T) }
            }

            var counts: [String: Int] = [:]
            for width in widths {
                counts["affine4-\(width)"] = Self.replicates(for: affine4(width))
                counts["bf16-\(width)"] = Self.replicates(for: bf16(width))
            }

            func measure(_ kernel: String, _ width: Int, _ block: Int, _ up: Bool)
            {
                let body = kernel == "affine4" ? affine4(width) : bf16(width)
                let count = counts["\(kernel)-\(width)"]!
                _ = Self.timed(2, body)
                let us = Self.timed(count, body)
                samples.append(
                    E97Sample(
                        kernel: kernel, outputs: outputs, width: width,
                        block: block, ascending: up, microseconds: us,
                        replicates: count))
                print(
                    "E97_ROW cell kernel=\(kernel) outputs=\(outputs) m=\(width) "
                        + "block=\(block) up=\(up) "
                        + String(format: "us=%.3f net_us=%.3f", us, us - overhead)
                        + " replicates=\(count)")
            }

            func nullCell(_ label: String) {
                let body = affine4(4)
                let count = counts["affine4-4"]!
                _ = Self.timed(2, body)
                let us = Self.timed(count, body)
                nulls.append([
                    "label": label, "outputs": outputs, "kernel": "affine4",
                    "m": 4, "us": us, "replicates": count,
                ])
                print(
                    "E97_ROW null label=\(label) outputs=\(outputs) "
                        + String(format: "us=%.3f", us))
            }

            nullCell("session_open")
            for block in 0 ..< Self.blocks {
                let up = block % 2 == 0
                let order = up ? widths : widths.reversed()
                for width in order {
                    measure("affine4", width, block, up)
                    measure("bf16", width, block, up)
                }
            }
            nullCell("session_close")
        }

        let overheadClose = Self.evalOverheadMicroseconds()
        print(String(format: "E97_ROW overhead_close us=%.3f", overheadClose))

        if let path = ProcessInfo.processInfo.environment["MLXFAST_E97_ROW_OUT"],
            !path.isEmpty
        {
            let payload: [String: Any] = [
                "hidden": Self.hidden,
                "group_size": Self.groupSize,
                "bits": Self.bits,
                "vector_limit": 10,
                "blocks": Self.blocks,
                "eval_overhead_us": overhead,
                "eval_overhead_close_us": overheadClose,
                "reads": reads,
                "nulls": nulls,
                "cells": samples.map {
                    [
                        "kernel": $0.kernel,
                        "outputs": $0.outputs,
                        "packed_bytes": Self.packedBytes(outputs: $0.outputs),
                        "dense_bytes": Self.denseBytes(outputs: $0.outputs),
                        "m": $0.width,
                        "block": $0.block,
                        "ascending": $0.ascending,
                        "us": $0.microseconds,
                        "replicates": $0.replicates,
                    ]
                },
            ]
            let data = try JSONSerialization.data(
                withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
            try data.write(to: URL(fileURLWithPath: path))
        }

        #expect(!samples.isEmpty)
    }
}
