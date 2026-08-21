import Foundation
import MLX
import MLXRandom
import Testing

// E115 rung 1 -- does a SECOND concurrent QMV dispatch buy overlap, and does
// that gain require the two dispatches to read the SAME weights?
//
// Every concurrent pair this campaign has ever timed was an M-partition of one
// tensor, so both dispatches read the identical weight matrix. Two rival
// mechanisms explain every such measurement equally well:
//
//   H1 request-level overlap    a second independent instruction stream hides
//                               the first stream's load-to-use stalls. The gain
//                               does NOT need shared weights.
//   H2 shared-weight caching    the second group's reads hit the system level
//                               cache, so LOGICAL bytes per second rises while
//                               DRAM bytes per second does not. The gain
//                               disappears on disjoint memory.
//   H3 slicing                  a smaller N per dispatch changes grid shape and
//                               cache blocking. Not concurrency at all.
//
// The arms separate them. All arms hold NA and total logical weight bytes fixed
// except `b_msplit`, which reads the weights twice by construction:
//
//   a_one            1 dispatch,  NA rows, full N                    bytes B
//   b_msplit         2 concurrent, disjoint M rows, full N each      bytes 2B
//   c_nsplit         2 concurrent, NA rows each, disjoint N halves   bytes B
//   c_nsplit_pre     c_nsplit with the halves hoisted out of the body bytes B
//   d_indep          2 concurrent, two DIFFERENT weight buffers      bytes B
//   e_nsplit_serial  c_nsplit with the two dispatches in separate evals
//   f_nsplit4        4 concurrent, disjoint N quarters               bytes B
//
// `c_nsplit` slices inside the timed body, which is what a naive call-site
// implementation would do; `c_nsplit_pre` hoists the slices. The pair prices the
// slice op. `Memory.activeMemory` around the hoisted slices says whether MLX
// aliases the parent buffer or materialises a copy.
//
// `e_nsplit_serial` removes concurrency by putting each dispatch in its own
// blocking eval, so it pays one extra eval overhead. That overhead is measured
// directly, not solved for, and the analysis reports raw and net.
//
// EXACTNESS. Output column n depends only on weight row n and the activation
// block, so an N-split must be bit-identical to one dispatch. The probe checks
// that rather than asserting it, and a deliberately WRONG split (the halves
// concatenated in swapped order) must change the digest. Without that positive
// control the exactness check cannot fail and proves nothing.
//
// Weights are synthesised directly in packed affine-4 group-64 layout. The
// kernel has no data-dependent control flow, so timing does not depend on the
// bit patterns, and every comparison here is between arms that read the very
// same buffers.
//
// Research instrument. Off unless `MLXFAST_RUN_E115_PROBE=1`. `Tests/` is never
// packaged into a submission. Within-session relative measurement: no thermal
// gate, no score. `cool_gate_passed_real_gate=false` and
// `gate_qualified_for_timing=false` are recorded in the output.

private struct E115Weights {
    var packed: MLXArray
    var scales: MLXArray
    var biases: MLXArray

    func rows(_ range: Range<Int>, groupSize: Int) -> E115Weights {
        E115Weights(
            packed: packed[range], scales: scales[range], biases: biases[range])
    }
}

private struct E115Arm {
    var name: String
    var dispatches: Int
    /// Logical weight passes, in units of one full pass over the tensor.
    var weightPasses: Double
    var evalsPerReplicate: Int
    var bodies: [() -> [MLXArray]]
}

private struct E115Cell {
    var shape: String
    var width: Int
    var arm: String
    var dispatches: Int
    var weightPasses: Double
    var evalsPerReplicate: Int
    var block: Int
    var forwardMicroseconds: Double
    var reverseMicroseconds: Double
    var replicates: Int
}

@Suite("E115 concurrent dispatch probe")
struct E115ConcurrentDispatchProbeTests {
    static let enabled = ProcessInfo.processInfo.environment[
        "MLXFAST_RUN_E115_PROBE"] == "1"

    static let groupSize = 64
    static let bits = 4

    /// Only tensors with N >= 8192 qualify: both halves must stay above the
    /// `out_vec_size >= 4096` branch at `quantized.h:1917` so the split keeps
    /// the same kernel. N=5120 tensors (`gdn.out_proj`, `fa.o_proj`,
    /// `mlp.down`) are excluded by construction.
    ///
    /// `control.small` is not a scored shape. It runs every arm with the same
    /// graph structure on a tensor whose GPU work is about 1.5 us, so its cell
    /// time is the host cost of that arm structure. The analysis subtracts it,
    /// which is the right model for the scored path: there MLX encodes the next
    /// dispatch while the GPU runs the current one, so host cost is hidden and
    /// only GPU time transfers.
    static let allShapes: [(name: String, hidden: Int, outputs: Int)] = [
        ("mlp.gate_up", 5120, 34816),
        ("lm_head", 5120, 248320),
        ("gdn.in_proj", 5120, 16480),
        ("fa.qkv", 5120, 14336),
        ("control.small", 64, 8192),
    ]

    static var shapes: [(name: String, hidden: Int, outputs: Int)] {
        guard
            let requested = ProcessInfo.processInfo.environment[
                "MLXFAST_E115_SHAPES"], !requested.isEmpty
        else { return allShapes }
        let names = requested.split(separator: ",").map {
            $0.trimmingCharacters(in: .whitespaces)
        }
        return names.compactMap { name in allShapes.first { $0.name == name } }
    }

    static var widths: [Int] {
        guard
            let requested = ProcessInfo.processInfo.environment[
                "MLXFAST_E115_WIDTHS"], !requested.isEmpty
        else { return [2, 3, 4, 5] }
        return requested.split(separator: ",").compactMap { Int($0) }
    }

    static var blocks: Int {
        Int(ProcessInfo.processInfo.environment["MLXFAST_E115_BLOCKS"] ?? "") ?? 6
    }

    /// affine 4-bit group-64: 4 bits per weight plus a bf16 scale and bias.
    private static func packedBytes(hidden: Int, outputs: Int) -> Int {
        outputs * hidden / 2 + 4 * (outputs * hidden / groupSize)
    }

    private static func makeWeights(hidden: Int, outputs: Int, seed: UInt64)
        -> E115Weights
    {
        MLXRandom.seed(seed)
        let packed = MLXRandom.randInt(
            low: Int32(0), high: Int32(1) << 30, [outputs, hidden / 8],
            type: Int32.self
        ).asType(.uint32)
        let scales = MLXRandom.uniform(
            low: Float(0.004), high: Float(0.02),
            [outputs, hidden / groupSize]
        ).asType(.bfloat16)
        let biases = MLXRandom.uniform(
            low: Float(-0.06), high: Float(0.06),
            [outputs, hidden / groupSize]
        ).asType(.bfloat16)
        eval(packed, scales, biases)
        return E115Weights(packed: packed, scales: scales, biases: biases)
    }

    private static func call(_ x: MLXArray, _ w: E115Weights) -> MLXArray {
        quantizedMM(
            x, w.packed, scales: w.scales, biases: w.biases, transpose: true,
            groupSize: groupSize, bits: bits, mode: .affine)
    }

    private static func timed(_ count: Int, _ bodies: [() -> [MLXArray]])
        -> Double
    {
        let start = DispatchTime.now().uptimeNanoseconds
        for _ in 0 ..< count {
            for body in bodies { eval(body()) }
        }
        return Double(DispatchTime.now().uptimeNanoseconds - start) / 1e3
            / Double(count)
    }

    /// The floor every cell pays: one blocking `eval` of negligible GPU work.
    private static func evalOverheadMicroseconds() -> Double {
        let tiny = MLXArray(Array(repeating: Float(1), count: 16))
        eval(tiny)
        _ = timed(200, [{ [tiny + 1] }])
        return timed(4000, [{ [tiny + 1] }])
    }

    /// FNV-1a over the raw float32 bit patterns of a whole output tensor.
    private static func digest(_ out: MLXArray) -> UInt64 {
        let values: [Float] = out.asType(.float32).asArray(Float.self)
        var h: UInt64 = 0xcbf2_9ce4_8422_2325
        for v in values {
            var bits = UInt64(v.bitPattern)
            for _ in 0 ..< 4 {
                h = (h ^ (bits & 0xff)) &* 0x0000_0100_0000_01b3
                bits >>= 8
            }
        }
        return h
    }

    @Test(
        "a second concurrent dispatch, with and without shared weights",
        .enabled(if: E115ConcurrentDispatchProbeTests.enabled))
    func concurrentDispatch() throws {
        let overhead = Self.evalOverheadMicroseconds()
        print(String(format: "E115 eval_overhead_us=%.3f", overhead))
        fflush(stdout)

        var cells: [E115Cell] = []
        var exactness: [[String: Any]] = []
        var sliceAliasing: [[String: Any]] = []
        var failures: [String] = []

        for shape in Self.shapes {
            let half = shape.outputs / 2
            let quarter = shape.outputs / 4
            // A half or quarter below 4096 would leave the wide branch and
            // change kernel, which voids bit-exactness by construction.
            guard half >= 4096, shape.outputs % 2 == 0 else {
                failures.append("\(shape.name): half \(half) below wide branch")
                continue
            }
            let splitFour = quarter >= 4096 && shape.outputs % 4 == 0

            let w = Self.makeWeights(
                hidden: shape.hidden, outputs: shape.outputs,
                seed: UInt64(0xE115) &+ UInt64(shape.outputs))
            // Two independent buffers of N/2 rows each. Same total logical
            // bytes as one full tensor, but no shared weight memory at all.
            let indepA = Self.makeWeights(
                hidden: shape.hidden, outputs: half,
                seed: UInt64(0xA115) &+ UInt64(shape.outputs))
            let indepB = Self.makeWeights(
                hidden: shape.hidden, outputs: half,
                seed: UInt64(0xB115) &+ UInt64(shape.outputs))

            // Does a first-dim slice alias the parent buffer or copy it?
            // Settle first: MLX releases a temporary's buffer lazily, so an
            // unsettled reading can fall while the slices are created and hide
            // a copy.
            eval(MLXArray(0))
            Memory.clearCache()
            let activeSettle = Memory.activeMemory
            eval(MLXArray(0))
            Memory.clearCache()
            let activeBefore = Memory.activeMemory
            let topPre = w.rows(0 ..< half, groupSize: Self.groupSize)
            let bottomPre = w.rows(half ..< shape.outputs, groupSize: Self.groupSize)
            eval(
                topPre.packed, topPre.scales, topPre.biases, bottomPre.packed,
                bottomPre.scales, bottomPre.biases)
            let activeAfter = Memory.activeMemory
            let fullBytes = Self.packedBytes(
                hidden: shape.hidden, outputs: shape.outputs)
            sliceAliasing.append([
                "shape": shape.name,
                "active_settle": activeSettle,
                "active_before": activeBefore,
                "active_after": activeAfter,
                "settle_drift_bytes": activeBefore - activeSettle,
                "delta_bytes": activeAfter - activeBefore,
                "full_tensor_bytes": fullBytes,
                "copy_would_add_bytes": fullBytes,
            ])
            print(
                "E115 slice_aliasing shape=\(shape.name) "
                    + "delta_bytes=\(activeAfter - activeBefore) "
                    + "full_tensor_bytes=\(fullBytes)")
            fflush(stdout)

            let quarters: [E115Weights] = splitFour
                ? (0 ..< 4).map {
                    w.rows(($0 * quarter) ..< (($0 + 1) * quarter),
                           groupSize: Self.groupSize)
                }
                : []
            if splitFour { eval(quarters.flatMap { [$0.packed, $0.scales, $0.biases] }) }

            let maxWidth = Self.widths.max() ?? 5
            MLXRandom.seed(UInt64(0xBEEF) &+ UInt64(shape.outputs))
            let block = MLXRandom.normal([2 * maxWidth, shape.hidden])
                .asType(.bfloat16)
            eval(block)

            // Roughly 100 ms of integrated GPU work per cell, as in E100.
            let count = max(12, min(400, Int(1.0e11) / (fullBytes * 4)))

            for width in Self.widths {
                let x = contiguous(block[0 ..< width])
                let x2 = contiguous(block[width ..< (2 * width)])
                eval(x, x2)

                var arms: [E115Arm] = [
                    E115Arm(
                        name: "a_one", dispatches: 1, weightPasses: 1,
                        evalsPerReplicate: 1,
                        bodies: [{ [Self.call(x, w)] }]),
                    E115Arm(
                        name: "b_msplit", dispatches: 2, weightPasses: 2,
                        evalsPerReplicate: 1,
                        bodies: [{ [Self.call(x, w), Self.call(x2, w)] }]),
                    E115Arm(
                        name: "c_nsplit", dispatches: 2, weightPasses: 1,
                        evalsPerReplicate: 1,
                        bodies: [{
                            [
                                Self.call(
                                    x, w.rows(0 ..< half, groupSize: Self.groupSize)),
                                Self.call(
                                    x,
                                    w.rows(half ..< shape.outputs,
                                           groupSize: Self.groupSize)),
                            ]
                        }]),
                    E115Arm(
                        name: "c_nsplit_pre", dispatches: 2, weightPasses: 1,
                        evalsPerReplicate: 1,
                        bodies: [{
                            [Self.call(x, topPre), Self.call(x, bottomPre)]
                        }]),
                    E115Arm(
                        name: "d_indep", dispatches: 2, weightPasses: 1,
                        evalsPerReplicate: 1,
                        bodies: [{
                            [Self.call(x, indepA), Self.call(x, indepB)]
                        }]),
                    E115Arm(
                        name: "e_nsplit_serial", dispatches: 2, weightPasses: 1,
                        evalsPerReplicate: 2,
                        bodies: [
                            { [Self.call(x, topPre)] },
                            { [Self.call(x, bottomPre)] },
                        ]),
                ]
                if splitFour {
                    arms.append(
                        E115Arm(
                            name: "f_nsplit4", dispatches: 4, weightPasses: 1,
                            evalsPerReplicate: 1,
                            bodies: [{ quarters.map { Self.call(x, $0) } }]))
                }

                for arm in arms { _ = Self.timed(3, arm.bodies) }

                for blockIndex in 0 ..< Self.blocks {
                    let entryTemp = e115GPUTemperature()
                    let forward = arms.map { Self.timed(count, $0.bodies) }
                    let reverse = Array(
                        arms.reversed().map { Self.timed(count, $0.bodies) }
                            .reversed())
                    let exitTemp = e115GPUTemperature()

                    for (index, arm) in arms.enumerated() {
                        cells.append(
                            E115Cell(
                                shape: shape.name, width: width, arm: arm.name,
                                dispatches: arm.dispatches,
                                weightPasses: arm.weightPasses,
                                evalsPerReplicate: arm.evalsPerReplicate,
                                block: blockIndex,
                                forwardMicroseconds: forward[index],
                                reverseMicroseconds: reverse[index],
                                replicates: count))
                    }
                    var record: [String: Any] = [
                        "shape": shape.name,
                        "width": width,
                        "block": blockIndex,
                        "replicates": count,
                        "arms": arms.enumerated().map { index, arm in
                            [
                                "arm": arm.name,
                                "forward_us": forward[index],
                                "reverse_us": reverse[index],
                                "mean_us": (forward[index] + reverse[index]) / 2,
                            ] as [String: Any]
                        },
                    ]
                    if let entryTemp { record["gpu_temp_entry_c"] = entryTemp }
                    if let exitTemp { record["gpu_temp_exit_c"] = exitTemp }
                    print(
                        "E115_BLOCK "
                            + (String(
                                data: (try? JSONSerialization.data(
                                    withJSONObject: record,
                                    options: [.sortedKeys])) ?? Data(),
                                encoding: .utf8) ?? "{}"))
                    fflush(stdout)
                }

                // Exactness, once per cell, outside the timed region.
                let one = Self.call(x, w)
                let top = Self.call(x, topPre)
                let bottom = Self.call(x, bottomPre)
                eval(one, top, bottom)
                let oneDigest = Self.digest(one)
                let splitDigest = Self.digest(concatenated([top, bottom], axis: 1))
                // Positive control: the same two halves in the wrong order MUST
                // change the digest, or the check above cannot fail.
                let wrongDigest = Self.digest(concatenated([bottom, top], axis: 1))
                var record: [String: Any] = [
                    "shape": shape.name,
                    "width": width,
                    "one_digest": String(oneDigest),
                    "nsplit_digest": String(splitDigest),
                    "wrong_split_digest": String(wrongDigest),
                    "nsplit_bit_exact": splitDigest == oneDigest,
                    "positive_control_differs": wrongDigest != oneDigest,
                ]
                if splitFour {
                    let fourDigest = Self.digest(
                        concatenated(quarters.map { Self.call(x, $0) }, axis: 1))
                    record["nsplit4_digest"] = String(fourDigest)
                    record["nsplit4_bit_exact"] = fourDigest == oneDigest
                }
                exactness.append(record)
                print("E115 exactness \(record)")
                fflush(stdout)

                eval(MLXArray(0))
            }

            Memory.clearCache()
        }

        for failure in failures { print("E115 SKIP \(failure)") }

        if let path = ProcessInfo.processInfo.environment["MLXFAST_E115_OUT"],
            !path.isEmpty
        {
            let payload: [String: Any] = [
                "group_size": Self.groupSize,
                "bits": Self.bits,
                "eval_overhead_us": overhead,
                "cool_gate_passed_real_gate": false,
                "gate_qualified_for_timing": false,
                "blocks": Self.blocks,
                "widths": Self.widths,
                "skipped": failures,
                "slice_aliasing": sliceAliasing,
                "exactness": exactness,
                "cells": cells.map {
                    [
                        "shape": $0.shape,
                        "width": $0.width,
                        "arm": $0.arm,
                        "dispatches": $0.dispatches,
                        "weight_passes": $0.weightPasses,
                        "evals_per_replicate": $0.evalsPerReplicate,
                        "block": $0.block,
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
        #expect(
            exactness.allSatisfy { ($0["nsplit_bit_exact"] as? Bool) == true })
        #expect(
            exactness.allSatisfy {
                ($0["positive_control_differs"] as? Bool) == true
            })
        #expect(
            exactness.allSatisfy {
                ($0["nsplit4_bit_exact"] as? Bool) ?? true
            })
    }
}

/// One macmon sample. The probe runs under no thermal gate, so the entry and
/// exit temperature of every block is the thermal record.
private func e115GPUTemperature() -> Double? {
    let binary =
        ProcessInfo.processInfo.environment["MLXFAST_E115_MACMON"]
        ?? "/opt/homebrew/bin/macmon"
    guard FileManager.default.isExecutableFile(atPath: binary) else { return nil }
    let process = Process()
    process.executableURL = URL(fileURLWithPath: binary)
    process.arguments = ["pipe", "-s1"]
    let pipe = Pipe()
    process.standardOutput = pipe
    process.standardError = FileHandle.nullDevice
    do { try process.run() } catch { return nil }
    let data = pipe.fileHandleForReading.readDataToEndOfFile()
    process.waitUntilExit()
    for line in String(decoding: data, as: UTF8.self).split(separator: "\n") {
        guard let object = try? JSONSerialization.jsonObject(with: Data(line.utf8)),
            let root = object as? [String: Any],
            let temp = root["temp"] as? [String: Any],
            let gpu = temp["gpu_temp_avg"] as? Double
        else { continue }
        return gpu
    }
    return nil
}
