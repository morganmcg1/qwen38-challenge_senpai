import Foundation
import MLX
import Testing

// E98 rung 1a -- does removing metadata bytes remove proportional time?
//
// An affine-4 group-64 projection reads 36 B per 64 weight elements: 32 B of
// nibbles, 2 B of scale and 2 B of bias. A lossless uint16 (scale, bias) index
// would read 34 B. The 5.56 % byte cut is arithmetic. Whether it is also a
// 5.56 % time cut depends on whether the dispatch is DRAM limited on that
// stream, and that is a measurement.
//
// The probe needs no new kernel. `affine_qmv_fast` enters the campaign WIDE
// switch only when `group_size == 64 && bits == 4`
// (kernels/quantized.h:1917), and case `ntg.x == 1` inside that switch calls
// `qmv_fast_impl<T, 64, 4>` anyway. Group sizes 32 and 128 fail the gate and
// call `qmv_fast_impl<T, group_size, 4>`. So at M = 1 all three group sizes run
// ONE kernel and differ only in metadata bytes per element:
//
//     group 32   40 B per 64 elements   1.1111 x g64
//     group 64   36 B per 64 elements   1.0000
//     group 128  34 B per 64 elements   0.9444 x g64
//
// The nibble stream is byte-identical in all three, and g128 reads exactly the
// byte count the indexed form targets. `g64 -> g128` at M = 1 is therefore the
// E98 prize measured through the shipped kernel, with the LUT lookup removed.
//
// At M > 1 group 64 leaves for the cross-row kernels while 32 and 128 stay on
// `qmv_fast_impl`, so the g32-to-g128 contrast stays clean at every M and the
// g64 cells are reported with their kernel family recorded, not silently mixed.
//
// Research instrument. Off unless `MLXFAST_RUN_E98_BYTES=1`. `Tests/` is never
// packaged into a submission. Within-session counterbalanced measurement, no
// thermal gate, no score.

private struct E98Cell {
    var shape: String
    var hidden: Int
    var outputs: Int
    var groupSize: Int
    var width: Int
    var block: Int
    var ascending: Bool
    var microseconds: Double
    var replicates: Int
}

@Suite("E98 metadata byte probe")
struct E98MetadataByteProbeTests {
    static let enabled = ProcessInfo.processInfo.environment[
        "MLXFAST_RUN_E98_BYTES"] == "1"

    static let bits = 4
    static let groupSizes = [32, 64, 128]
    /// `get_qmv_batch_limit(K, N) == 10` at every scored shape, so 1...8 stays
    /// inside the vector regime for every group size.
    static let widths = [1, 5, 6, 7, 8]
    static let blocks = 6
    static let targetCellMicroseconds = 120_000.0

    /// The three scored affine-4 g64 shapes named in the assignment, as
    /// (hidden K, outputs N, label). `qmv_fast` needs `K % 512 == 0` and
    /// `N % 8 == 0`; all three satisfy both.
    static let shapes: [(Int, Int, String)] = [
        (5120, 34816, "mlp_gate_up_k5120_n34816"),
        (17408, 5120, "mlp_down_k17408_n5120"),
        (5120, 248320, "lm_head_k5120_n248320"),
    ]

    /// Bytes the dispatch must read: nibbles, then metadata, then activations.
    private static func readBytes(
        hidden: Int, outputs: Int, groupSize: Int, width: Int
    ) -> Int {
        outputs * hidden / 2
            + 4 * (outputs * hidden / groupSize)
            + width * hidden * 2
    }

    private static func writeBytes(outputs: Int, width: Int) -> Int {
        width * outputs * 2
    }

    /// Which kernel the WIDE switch selects, so a cross-group-size comparison
    /// can never silently compare two different kernels.
    private static func kernelFamily(groupSize: Int, width: Int) -> String {
        if groupSize != 64 || width == 1 {
            return "qmv_fast_impl"
        }
        return width == 2 ? "crossrow" : "crossrow_m"
    }

    private static func timed(_ count: Int, _ body: () -> MLXArray) -> Double {
        let start = DispatchTime.now().uptimeNanoseconds
        for _ in 0 ..< count {
            eval(body())
        }
        return Double(DispatchTime.now().uptimeNanoseconds - start) / 1e3
            / Double(count)
    }

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

    @Test(
        "metadata bytes convert to time at the achieved read rate",
        .enabled(if: E98MetadataByteProbeTests.enabled))
    func metadataByteLadder() throws {
        let overhead = Self.evalOverheadMicroseconds()
        print(String(format: "E98_BYTES overhead us=%.3f", overhead))

        var cells: [E98Cell] = []
        var nulls: [[String: Any]] = []
        var reads: [[String: Any]] = []

        for (hidden, outputs, label) in Self.shapes {
            // One dense source for every group size, so the nibble stream and
            // the value distribution are shared and only the metadata rate
            // changes.
            var packedByGroup: [Int: (MLXArray, MLXArray, MLXArray)] = [:]
            do {
                let dense = MLXRandom.normal([outputs, hidden], dtype: .bfloat16)
                eval(dense)
                for groupSize in Self.groupSizes {
                    let (packed, scales, biases) = quantized(
                        dense, groupSize: groupSize, bits: Self.bits,
                        mode: .affine)
                    eval(packed, scales, biases)
                    packedByGroup[groupSize] = (packed, scales, biases)
                }
            }

            // The achieved read rate of this working set, so every cell can be
            // read against a rate this GPU actually reaches rather than the
            // DRAM specification.
            for groupSize in Self.groupSizes {
                let (packed, _, _) = packedByGroup[groupSize]!
                let bytes = outputs * hidden / 2
                _ = Self.timed(3) { packed.sum() }
                let raw = Self.timed(30) { packed.sum() }
                reads.append([
                    "shape": label, "group_size": groupSize, "bytes": bytes,
                    "raw_us": raw, "net_us": raw - overhead,
                    "gb_per_s": Double(bytes) / (raw - overhead) / 1e3,
                ])
                print(
                    "E98_BYTES read shape=\(label) gs=\(groupSize) "
                        + String(
                            format: "raw_us=%.3f net_gb_s=%.1f", raw,
                            Double(bytes) / (raw - overhead) / 1e3))
            }

            var inputs: [Int: MLXArray] = [:]
            for width in Self.widths {
                let x = MLXRandom.normal([width, hidden], dtype: .bfloat16)
                eval(x)
                inputs[width] = x
            }

            func cell(_ groupSize: Int, _ width: Int) -> () -> MLXArray {
                let x = inputs[width]!
                let (packed, scales, biases) = packedByGroup[groupSize]!
                return {
                    quantizedMM(
                        x, packed, scales: scales, biases: biases,
                        transpose: true, groupSize: groupSize,
                        bits: Self.bits, mode: .affine)
                }
            }

            var counts: [String: Int] = [:]
            for groupSize in Self.groupSizes {
                for width in Self.widths {
                    counts["\(groupSize)-\(width)"] = Self.replicates(
                        for: cell(groupSize, width))
                }
            }

            func measure(_ groupSize: Int, _ width: Int, _ block: Int, _ up: Bool)
            {
                let body = cell(groupSize, width)
                let count = counts["\(groupSize)-\(width)"]!
                _ = Self.timed(2, body)
                let us = Self.timed(count, body)
                cells.append(
                    E98Cell(
                        shape: label, hidden: hidden, outputs: outputs,
                        groupSize: groupSize, width: width, block: block,
                        ascending: up, microseconds: us, replicates: count))
                let bytes = Self.readBytes(
                    hidden: hidden, outputs: outputs, groupSize: groupSize,
                    width: width)
                print(
                    "E98_BYTES cell shape=\(label) gs=\(groupSize) m=\(width) "
                        + "block=\(block) up=\(up) "
                        + String(
                            format: "us=%.3f net_us=%.3f read_gb_s=%.1f",
                            us, us - overhead,
                            Double(bytes) / (us - overhead) / 1e3)
                        + " replicates=\(count)")
            }

            func nullCell(_ tag: String) {
                let body = cell(64, 5)
                let count = counts["64-5"]!
                _ = Self.timed(2, body)
                let us = Self.timed(count, body)
                nulls.append([
                    "shape": label, "tag": tag, "group_size": 64, "width": 5,
                    "microseconds": us, "replicates": count,
                ])
                print(
                    "E98_BYTES null shape=\(label) tag=\(tag) "
                        + String(format: "us=%.3f", us))
            }

            nullCell("session_open")
            for block in 0 ..< Self.blocks {
                let up = block % 2 == 0
                let order = up ? Self.groupSizes : Self.groupSizes.reversed()
                for groupSize in order {
                    for width in Self.widths {
                        measure(groupSize, width, block, up)
                    }
                }
            }
            nullCell("session_close")
        }

        let records: [[String: Any]] = cells.map { cell in
            let read = Self.readBytes(
                hidden: cell.hidden, outputs: cell.outputs,
                groupSize: cell.groupSize, width: cell.width)
            let write = Self.writeBytes(outputs: cell.outputs, width: cell.width)
            let net = cell.microseconds - overhead
            return [
                "shape": cell.shape, "hidden": cell.hidden,
                "outputs": cell.outputs, "group_size": cell.groupSize,
                "width": cell.width, "block": cell.block,
                "ascending": cell.ascending,
                "kernel_family": Self.kernelFamily(
                    groupSize: cell.groupSize, width: cell.width),
                "microseconds": cell.microseconds, "net_us": net,
                "replicates": cell.replicates,
                "read_bytes": read, "write_bytes": write,
                "read_gb_per_s": Double(read) / net / 1e3,
                "total_gb_per_s": Double(read + write) / net / 1e3,
            ]
        }

        let path =
            ProcessInfo.processInfo.environment["MLXFAST_E98_BYTES_OUT"]
            ?? "research/out/e98-bytes/bytes.json"
        try FileManager.default.createDirectory(
            at: URL(fileURLWithPath: path).deletingLastPathComponent(),
            withIntermediateDirectories: true)
        let data = try JSONSerialization.data(
            withJSONObject: [
                "eval_overhead_us": overhead, "bits": Self.bits,
                "blocks": Self.blocks, "group_sizes": Self.groupSizes,
                "widths": Self.widths, "cells": records, "nulls": nulls,
                "reads": reads,
            ], options: [.prettyPrinted, .sortedKeys])
        try data.write(to: URL(fileURLWithPath: path))

        #expect(!records.isEmpty)
    }
}
