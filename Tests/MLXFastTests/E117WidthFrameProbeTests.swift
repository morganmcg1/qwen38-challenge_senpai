import Foundation
import MLX
import MLXRandom
import Testing

// E117 rung 0 -- does the `mlp.gate_up` rate dip exist at the SHIPPED widths?
//
// E115 timed one x-group of NA rows in one dispatch and found that
// `mlp.gate_up` alone loses about 23 % of its rate between NA=3 and NA=4, and
// that a serialised two-way N-split recovers it. That probe was in the NA
// frame. The shipped kernel is in the M frame, and the two are not the same
// object at the width that matters.
//
// `quantized.cpp:251-254` dispatches `grid_dims(M, N/8, B)` threadgroups, so
// `ntg.x == M`. `quantized.h:1922-1979` then selects an inputs-per-group IPG
// and `qmv_fast_crossrow_affine4_g64_m` early-returns every threadgroup whose
// `first_m = tid.x * IPG` is at or past M. The realised partition is therefore
//
//   M     1    2    3    4    5      6      7      8        9
//   IPG   -    2    3    4    5      3      4      4        3
//   part  [1]  [2]  [3]  [4]  [5]  [3+3]  [4+3]  [4+4]  [3+3+3]
//   grps  1    1    1    1    1      2      2      2        3
//
// E115's "NA=4" cell is the `[4]` partition at `ntg.x = 4`: one working group
// in the whole dispatch. The width that carries 84 % of local rounds is M=8,
// which is `[4+4]`: two working groups inside ONE dispatch, sharing one weight
// stream. The dip may be a property of the group, in which case it survives
// grouping, or a property of the isolated single-group dispatch, in which case
// it is not claimable at all.
//
// This probe drives the real `quantizedMM` entry point at M = 1 .. 9 so the
// partition table above is exercised as shipped, and reports `a_one` rate in
// the M frame. The discriminator is arithmetic on the resulting table:
//
//   M=8 `[4+4]` close to 2 x M=4 `[4]`   -> the dip is per group and survives
//   M=8 `[4+4]` much better than 2 x [4] -> the dip is an isolated artefact
//
// Arms, all reading the very same buffers:
//
//   a_one            1 dispatch,  M rows, full N
//   c_nsplit         2 concurrent dispatches, M rows each, disjoint N halves
//   e_nsplit_serial  the same two dispatches in separate blocking evals
//
// `control.small` is not a scored shape. It runs every arm structure on a
// tensor whose GPU work is about 1.5 us, so its cell time is that structure's
// host cost. The analysis subtracts it, which is the right model for the scored
// path: there MLX encodes the next dispatch while the GPU runs the current one.
//
// EXACTNESS. Output column n depends only on weight row n and the activation
// block, so an N-split must be bit-identical to one dispatch. The probe checks
// that, and a deliberately WRONG split (halves concatenated in swapped order)
// must change the digest, or the check cannot fail.
//
// HARNESS DEFECT 16, which is mine and must not come back. Sampling `macmon`
// runs a subprocess and leaves the GPU idle long enough to drop its clocks. The
// DVFS ramp costs a fixed 30 to 80 ms of wall clock and is paid entirely by
// whichever arm is timed first, so it is not monotone drift and a palindrome
// does not cancel it. E115 inserted a ramp burst of `count / 4` replicates,
// which is a FIXED REPLICATE COUNT, so on `mlp.gate_up` it was only about 43 ms
// and absorbed the ramp only partly. Here the burst runs for a fixed WALL-CLOCK
// duration instead, and both passes are reported separately so the residual
// forward-versus-reverse gap per arm is publishable as proof it worked.
//
// Research instrument. Off unless `MLXFAST_RUN_E117_PROBE=1`. `Tests/` is never
// packaged into a submission. Within-session relative measurement: no thermal
// gate, no score. `cool_gate_passed_real_gate=false` and
// `gate_qualified_for_timing=false` are recorded in the output.

private struct E117Weights {
    var packed: MLXArray
    var scales: MLXArray
    var biases: MLXArray

    func rows(_ range: Range<Int>) -> E117Weights {
        E117Weights(packed: packed[range], scales: scales[range], biases: biases[range])
    }
}

private struct E117Arm {
    var name: String
    var dispatches: Int
    var evalsPerReplicate: Int
    var bodies: [() -> [MLXArray]]
}

private struct E117Cell {
    var shape: String
    var outputs: Int
    var hidden: Int
    var width: Int
    var arm: String
    var dispatches: Int
    var evalsPerReplicate: Int
    var block: Int
    var forwardMicroseconds: Double
    var reverseMicroseconds: Double
    var replicates: Int
}

@Suite("E117 shipped-width frame probe")
struct E117WidthFrameProbeTests {
    static let enabled = ProcessInfo.processInfo.environment["MLXFAST_RUN_E117_PROBE"] == "1"

    static let groupSize = 64
    static let bits = 4

    /// Named presets. `MLXFAST_E117_SHAPES` also accepts `name:N:K` triples so
    /// the rung-0b N sweep can ask for widths that are not scored shapes.
    static let presets: [String: (hidden: Int, outputs: Int)] = [
        "mlp.gate_up": (5120, 34816),
        "lm_head": (5120, 248_320),
        "gdn.in_proj": (5120, 16480),
        "fa.qkv": (5120, 14336),
        "control.small": (64, 8192),
    ]

    static var shapes: [(name: String, hidden: Int, outputs: Int)] {
        let requested =
            ProcessInfo.processInfo.environment["MLXFAST_E117_SHAPES"]
            ?? "mlp.gate_up,control.small"
        return requested.split(separator: ",").compactMap { token in
            let text = token.trimmingCharacters(in: .whitespaces)
            let parts = text.split(separator: ":")
            if parts.count == 3, let n = Int(parts[1]), let k = Int(parts[2]) {
                return (String(parts[0]), k, n)
            }
            guard let preset = presets[text] else { return nil }
            return (text, preset.hidden, preset.outputs)
        }
    }

    static var widths: [Int] {
        guard let requested = ProcessInfo.processInfo.environment["MLXFAST_E117_WIDTHS"],
            !requested.isEmpty
        else { return Array(1 ... 9) }
        return requested.split(separator: ",").compactMap { Int($0) }
    }

    static var blocks: Int {
        Int(ProcessInfo.processInfo.environment["MLXFAST_E117_BLOCKS"] ?? "") ?? 6
    }

    /// Fixed wall-clock seconds of discarded work after every temperature
    /// sample. Must exceed the 30 to 80 ms DVFS ramp with margin.
    static var rampSeconds: Double {
        Double(ProcessInfo.processInfo.environment["MLXFAST_E117_RAMP_S"] ?? "") ?? 0.30
    }

    /// Target integrated GPU microseconds per timed cell.
    static var targetMicroseconds: Double {
        Double(ProcessInfo.processInfo.environment["MLXFAST_E117_TARGET_US"] ?? "") ?? 100_000
    }

    /// affine 4-bit group-64: 4 bits per weight plus a bf16 scale and bias.
    private static func packedBytes(hidden: Int, outputs: Int) -> Int {
        outputs * hidden / 2 + 4 * (outputs * hidden / groupSize)
    }

    private static func makeWeights(hidden: Int, outputs: Int, seed: UInt64) -> E117Weights {
        MLXRandom.seed(seed)
        let packed = MLXRandom.randInt(
            low: Int32(0), high: Int32(1) << 30, [outputs, hidden / 8], type: Int32.self
        ).asType(.uint32)
        let scales = MLXRandom.uniform(
            low: Float(0.004), high: Float(0.02), [outputs, hidden / groupSize]
        ).asType(.bfloat16)
        let biases = MLXRandom.uniform(
            low: Float(-0.06), high: Float(0.06), [outputs, hidden / groupSize]
        ).asType(.bfloat16)
        eval(packed, scales, biases)
        return E117Weights(packed: packed, scales: scales, biases: biases)
    }

    private static func call(_ x: MLXArray, _ w: E117Weights) -> MLXArray {
        quantizedMM(
            x, w.packed, scales: w.scales, biases: w.biases, transpose: true,
            groupSize: groupSize, bits: bits, mode: .affine)
    }

    private static func timed(_ count: Int, _ bodies: [() -> [MLXArray]]) -> Double {
        let start = DispatchTime.now().uptimeNanoseconds
        for _ in 0 ..< count {
            for body in bodies { eval(body()) }
        }
        return Double(DispatchTime.now().uptimeNanoseconds - start) / 1e3 / Double(count)
    }

    /// Discarded work of fixed wall-clock duration. This is the harness-defect-16
    /// fix: it absorbs the whole DVFS ramp regardless of how expensive one
    /// replicate happens to be at this shape and width.
    private static func rampBurst(_ bodies: [() -> [MLXArray]], seconds: Double) {
        let start = DispatchTime.now().uptimeNanoseconds
        while Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9 < seconds {
            for body in bodies { eval(body()) }
        }
    }

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
        "a_one and the serial N-split across the shipped M partition table",
        .enabled(if: E117WidthFrameProbeTests.enabled))
    func widthFrame() throws {
        let overhead = Self.evalOverheadMicroseconds()
        print(String(format: "E117 eval_overhead_us=%.3f", overhead))
        fflush(stdout)

        var cells: [E117Cell] = []
        var exactness: [[String: Any]] = []
        var sliceAliasing: [[String: Any]] = []
        var failures: [String] = []

        for shape in Self.shapes {
            let half = shape.outputs / 2
            // Both halves must stay at or above the `out_vec_size >= 4096`
            // branch at `quantized.h:1917`, or the split changes kernel and
            // bit-exactness is void by construction.
            guard shape.outputs % 2 == 0, half >= 4096 else {
                failures.append("\(shape.name): half \(half) below the wide branch")
                continue
            }
            // `fast = N % 8 == 0 && K % 512 == 0` at `quantized.cpp:260`.
            let fastPath = shape.outputs % 8 == 0 && shape.hidden % 512 == 0
            let halfFast = half % 8 == 0 && shape.hidden % 512 == 0
            if fastPath != halfFast {
                failures.append("\(shape.name): split leaves the qmv_fast path")
                continue
            }

            let w = Self.makeWeights(
                hidden: shape.hidden, outputs: shape.outputs,
                seed: UInt64(0xE117) &+ UInt64(shape.outputs))

            eval(MLXArray(0))
            Memory.clearCache()
            let activeSettle = Memory.activeMemory
            eval(MLXArray(0))
            Memory.clearCache()
            let activeBefore = Memory.activeMemory
            let topPre = w.rows(0 ..< half)
            let bottomPre = w.rows(half ..< shape.outputs)
            eval(
                topPre.packed, topPre.scales, topPre.biases, bottomPre.packed,
                bottomPre.scales, bottomPre.biases)
            let activeAfter = Memory.activeMemory
            let fullBytes = Self.packedBytes(hidden: shape.hidden, outputs: shape.outputs)
            sliceAliasing.append([
                "shape": shape.name,
                "outputs": shape.outputs,
                "hidden": shape.hidden,
                "active_settle": activeSettle,
                "active_before": activeBefore,
                "active_after": activeAfter,
                "settle_drift_bytes": activeBefore - activeSettle,
                "delta_bytes": activeAfter - activeBefore,
                "full_tensor_bytes": fullBytes,
                "copy_would_add_bytes": fullBytes,
            ])
            print(
                "E117 slice_aliasing shape=\(shape.name) n=\(shape.outputs) "
                    + "delta_bytes=\(activeAfter - activeBefore) "
                    + "full_tensor_bytes=\(fullBytes)")
            fflush(stdout)

            let maxWidth = Self.widths.max() ?? 9
            MLXRandom.seed(UInt64(0xBEEF) &+ UInt64(shape.outputs))
            let block = MLXRandom.normal([maxWidth, shape.hidden]).asType(.bfloat16)
            eval(block)

            for width in Self.widths {
                let x = contiguous(block[0 ..< width])
                eval(x)

                let arms: [E117Arm] = [
                    E117Arm(
                        name: "a_one", dispatches: 1, evalsPerReplicate: 1,
                        bodies: [{ [Self.call(x, w)] }]),
                    E117Arm(
                        name: "c_nsplit", dispatches: 2, evalsPerReplicate: 1,
                        bodies: [{ [Self.call(x, topPre), Self.call(x, bottomPre)] }]),
                    E117Arm(
                        name: "e_nsplit_serial", dispatches: 2, evalsPerReplicate: 2,
                        bodies: [
                            { [Self.call(x, topPre)] },
                            { [Self.call(x, bottomPre)] },
                        ]),
                ]

                // Calibrate the replicate count from a measured `a_one`, not
                // from a byte estimate: the working group count varies with M,
                // so a fixed formula would integrate very different amounts of
                // GPU time at M=4 and M=9. Ramp first, or the calibration reads
                // a low-clock time and undercounts the replicates.
                Self.rampBurst(arms[0].bodies, seconds: Self.rampSeconds)
                let probeUs = Self.timed(8, arms[0].bodies)
                let count = max(8, min(600, Int(Self.targetMicroseconds / max(probeUs, 1.0))))
                for arm in arms { _ = Self.timed(3, arm.bodies) }

                for blockIndex in 0 ..< Self.blocks {
                    let entryTemp = e117GPUTemperature()
                    Self.rampBurst(arms[0].bodies, seconds: Self.rampSeconds)
                    let forward = arms.map { Self.timed(count, $0.bodies) }
                    let reverse = Array(
                        arms.reversed().map { Self.timed(count, $0.bodies) }.reversed())
                    let exitTemp = e117GPUTemperature()

                    for (index, arm) in arms.enumerated() {
                        cells.append(
                            E117Cell(
                                shape: shape.name, outputs: shape.outputs,
                                hidden: shape.hidden, width: width, arm: arm.name,
                                dispatches: arm.dispatches,
                                evalsPerReplicate: arm.evalsPerReplicate,
                                block: blockIndex, forwardMicroseconds: forward[index],
                                reverseMicroseconds: reverse[index], replicates: count))
                    }
                    var record: [String: Any] = [
                        "shape": shape.name,
                        "outputs": shape.outputs,
                        "width": width,
                        "block": blockIndex,
                        "replicates": count,
                        "arms": arms.enumerated().map { index, arm in
                            [
                                "arm": arm.name,
                                "forward_us": forward[index],
                                "reverse_us": reverse[index],
                            ] as [String: Any]
                        },
                    ]
                    if let entryTemp { record["gpu_temp_entry_c"] = entryTemp }
                    if let exitTemp { record["gpu_temp_exit_c"] = exitTemp }
                    print(
                        "E117_BLOCK "
                            + (String(
                                data: (try? JSONSerialization.data(
                                    withJSONObject: record, options: [.sortedKeys])) ?? Data(),
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
                let wrongDigest = Self.digest(concatenated([bottom, top], axis: 1))
                let record: [String: Any] = [
                    "shape": shape.name,
                    "outputs": shape.outputs,
                    "width": width,
                    "one_digest": String(oneDigest),
                    "nsplit_digest": String(splitDigest),
                    "wrong_split_digest": String(wrongDigest),
                    "nsplit_bit_exact": splitDigest == oneDigest,
                    "positive_control_differs": wrongDigest != oneDigest,
                ]
                exactness.append(record)
                print("E117 exactness \(record)")
                fflush(stdout)

                eval(MLXArray(0))
            }

            Memory.clearCache()
        }

        for failure in failures { print("E117 SKIP \(failure)") }

        if let path = ProcessInfo.processInfo.environment["MLXFAST_E117_OUT"], !path.isEmpty {
            let payload: [String: Any] = [
                "group_size": Self.groupSize,
                "bits": Self.bits,
                "eval_overhead_us": overhead,
                "ramp_seconds": Self.rampSeconds,
                "target_us": Self.targetMicroseconds,
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
                        "outputs": $0.outputs,
                        "hidden": $0.hidden,
                        "width": $0.width,
                        "arm": $0.arm,
                        "dispatches": $0.dispatches,
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
        #expect(exactness.allSatisfy { ($0["nsplit_bit_exact"] as? Bool) == true })
        #expect(exactness.allSatisfy { ($0["positive_control_differs"] as? Bool) == true })
    }
}

/// One macmon sample. The probe runs under no thermal gate, so the entry and
/// exit temperature of every block is the thermal record. Every sample is
/// followed by a discarded fixed-duration ramp burst, or is the last thing in
/// the block, so no timed arm pays the subprocess idle.
private func e117GPUTemperature() -> Double? {
    let binary =
        ProcessInfo.processInfo.environment["MLXFAST_E117_MACMON"]
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
