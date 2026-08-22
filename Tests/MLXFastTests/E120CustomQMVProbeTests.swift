import Foundation
import MLX
import MLXLLM
import MLXRandom
import Testing

// E120 -- own the wide affine-4/group-64 QMV dispatch.
//
// `quantized.cpp` is outside the editable surface, so the shipped wide kernel
// can never be given a tenth buffer and can never be launched on a grid the
// frozen launcher does not choose. `Qwen35CustomQMV` replaces the launcher, not
// the arithmetic: `MLXFast.metalKernel` binds exactly the buffers we name and
// `custom_kernel.cpp:113-117` dispatches exactly the grid we name.
//
// RUNG 1 GATE. Before any table or grid work is built on that capability, the
// replica must (1) be bit exact against `quantizedMM` on the same buffers and
// (2) match MLX end to end within about 1 %. A replica that cannot match MLX on
// identical arithmetic and identical geometry kills Route A outright.
//
// The third gate number is the per-dispatch host cost of the custom path. It is
// NOT only the `it->second != source_` comparison at `custom_kernel.cpp:57-71`:
// `metal_kernel.cpp:317-329` rebuilds the whole generated kernel source string
// on EVERY call, at graph-construction time, before any of it reaches the
// encoder. `hostBuild` below times graph construction with no `eval` at all, so
// it separates that host cost from every GPU effect, and the padded-source pair
// isolates the part of it that scales with source length.
//
// Research instrument. Timing is off unless `MLXFAST_RUN_E120_PROBE=1`;
// exactness runs with the standard `MLXFAST_RUN_MLX_RUNTIME_TESTS=1` sweep.
// `Tests/` is never packaged into a submission. Within-session relative
// measurement: no thermal gate, no score. `cool_gate_passed_real_gate=false`
// and `gate_qualified_for_timing=false` are recorded in the output.

private struct E120Weights {
    var packed: MLXArray
    var scales: MLXArray
    var biases: MLXArray
}

private struct SplitMix64 {
    var state: UInt64
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

private struct E120Arm {
    var name: String
    var body: () -> [MLXArray]
}

@Suite("E120 candidate-owned QMV dispatch")
struct E120CustomQMVProbeTests {
    static let runtimeEnabled =
        ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1"
    static let timingEnabled =
        ProcessInfo.processInfo.environment["MLXFAST_RUN_E120_PROBE"] == "1"

    static let groupSize = 64
    static let bits = 4

    static let presets: [String: (hidden: Int, outputs: Int)] = [
        "mlp.gate_up": (5120, 34816),
        "lm_head": (5120, 248_320),
        "gdn.in_proj": (5120, 16480),
        "fa.qkv": (5120, 14336),
        "control.small": (512, 4096),
    ]

    static var shapes: [(name: String, hidden: Int, outputs: Int)] {
        let requested =
            ProcessInfo.processInfo.environment["MLXFAST_E120_SHAPES"] ?? "mlp.gate_up"
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
        guard let requested = ProcessInfo.processInfo.environment["MLXFAST_E120_WIDTHS"],
            !requested.isEmpty
        else { return Array(3 ... 9) }
        return requested.split(separator: ",").compactMap { Int($0) }
    }

    static var blocks: Int {
        Int(ProcessInfo.processInfo.environment["MLXFAST_E120_BLOCKS"] ?? "") ?? 6
    }

    static var rampSeconds: Double {
        Double(ProcessInfo.processInfo.environment["MLXFAST_E120_RAMP_S"] ?? "") ?? 0.30
    }

    static var targetMicroseconds: Double {
        Double(ProcessInfo.processInfo.environment["MLXFAST_E120_TARGET_US"] ?? "") ?? 100_000
    }

    // MARK: fixtures

    /// Full 32-bit packed nibbles, so every nibble position sees the whole
    /// 0...15 range. `MLXRandom.randInt` tops out below 2^31 and would leave the
    /// high nibble permanently small.
    static func makeWeights(hidden: Int, outputs: Int, seed: UInt64) -> E120Weights {
        let words = outputs * hidden / 8
        var rng = SplitMix64(state: seed)
        var raw = [UInt32]()
        raw.reserveCapacity(words)
        for _ in 0 ..< words { raw.append(UInt32(truncatingIfNeeded: rng.next())) }
        let packed = MLXArray(raw).reshaped([outputs, hidden / 8])
        MLXRandom.seed(seed)
        let scales = MLXRandom.uniform(
            low: Float(0.004), high: Float(0.02), [outputs, hidden / groupSize]
        ).asType(.bfloat16)
        let biases = MLXRandom.uniform(
            low: Float(-0.06), high: Float(0.06), [outputs, hidden / groupSize]
        ).asType(.bfloat16)
        eval(packed, scales, biases)
        return E120Weights(packed: packed, scales: scales, biases: biases)
    }

    static func mlx(_ x: MLXArray, _ w: E120Weights) -> MLXArray {
        quantizedMM(
            x, w.packed, scales: w.scales, biases: w.biases, transpose: true,
            groupSize: groupSize, bits: bits, mode: .affine)
    }

    static func custom(_ x: MLXArray, _ w: E120Weights, _ arm: Qwen35CustomQMV.Arm)
        -> MLXArray?
    {
        Qwen35CustomQMV.matmul(
            x, w.packed, scales: w.scales, biases: w.biases,
            groupSize: groupSize, bits: bits, mode: .affine, arm: arm)
    }

    /// Number of output elements that differ. BF16 to float32 is exact, so a
    /// float32 inequality count is an exact bit comparison for finite values.
    static func mismatches(_ a: MLXArray, _ b: MLXArray) -> Int {
        (a .!= b).asType(.int32).sum().item(Int.self)
    }

    static func maxAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }

    // MARK: timing helpers

    static func timed(_ count: Int, _ body: () -> [MLXArray]) -> Double {
        let start = DispatchTime.now().uptimeNanoseconds
        for _ in 0 ..< count { eval(body()) }
        return Double(DispatchTime.now().uptimeNanoseconds - start) / 1e3 / Double(count)
    }

    static func rampBurst(_ body: () -> [MLXArray], seconds: Double) {
        let start = DispatchTime.now().uptimeNanoseconds
        while Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9 < seconds {
            eval(body())
        }
    }

    /// Graph construction only: build the op and drop it without evaluating.
    /// This is the whole host-side price of a dispatch, including the source
    /// string that `write_signature` rebuilds on every call.
    static func hostBuild(_ count: Int, _ body: () -> MLXArray) -> Double {
        for _ in 0 ..< 200 { _ = body() }
        let start = DispatchTime.now().uptimeNanoseconds
        for _ in 0 ..< count { _ = body() }
        return Double(DispatchTime.now().uptimeNanoseconds - start) / 1e3 / Double(count)
    }

    // MARK: rung 1 exactness

    @Test(
        "replica is bit exact against quantizedMM at every routed width",
        .enabled(if: E120CustomQMVProbeTests.runtimeEnabled))
    func replicaExactness() throws {
        var records: [[String: Any]] = []

        for shape in Self.shapes + [("control.small", 512, 4096)] {
            let w = Self.makeWeights(
                hidden: shape.hidden, outputs: shape.outputs,
                seed: UInt64(0xE120) &+ UInt64(shape.outputs))
            MLXRandom.seed(UInt64(0xB1F5) &+ UInt64(shape.outputs))
            let block = MLXRandom.normal([9, shape.hidden]).asType(.bfloat16)
            eval(block)

            for width in 3 ... 9 {
                let x = contiguous(block[0 ..< width])
                eval(x)
                let reference = Self.mlx(x, w)
                guard let replica = Self.custom(x, w, .replica) else {
                    Issue.record("\(shape.name) M=\(width): replica declined a routed cell")
                    continue
                }
                eval(reference, replica)

                // Positive control: one activation moved by one BF16 step must
                // change the replica's output, or the comparison cannot fail.
                let bumped = x + MLXArray(Float(0.5)).asType(.bfloat16)
                eval(bumped)
                guard let perturbed = Self.custom(bumped, w, .replica) else {
                    Issue.record("\(shape.name) M=\(width): control declined")
                    continue
                }
                eval(perturbed)

                let differing = Self.mismatches(reference, replica)
                let controlDiffering = Self.mismatches(reference, perturbed)
                let record: [String: Any] = [
                    "shape": shape.name,
                    "outputs": shape.outputs,
                    "hidden": shape.hidden,
                    "width": width,
                    "elements": reference.size,
                    "differing_elements": differing,
                    "max_abs_diff": Self.maxAbsDiff(reference, replica),
                    "bit_exact": differing == 0,
                    "positive_control_differing": controlDiffering,
                    "positive_control_can_fail": controlDiffering > 0,
                ]
                records.append(record)
                print("E120 exactness \(record)")
                fflush(stdout)
                #expect(differing == 0)
                #expect(controlDiffering > 0)
            }

            // The guard must decline every width the incumbent owns.
            for width in [1, 2, 10, 12] {
                let x = contiguous(
                    MLXRandom.normal([width, shape.hidden]).asType(.bfloat16))
                eval(x)
                #expect(Self.custom(x, w, .replica) == nil)
            }
        }

        #expect(!records.isEmpty)
    }

    // MARK: rung 1 host cost

    @Test(
        "per-dispatch host cost of the custom path against MLX",
        .enabled(if: E120CustomQMVProbeTests.timingEnabled))
    func hostDispatchCost() throws {
        let hidden = 5120
        let outputs = 34816
        let w = Self.makeWeights(hidden: hidden, outputs: outputs, seed: 0xE120)
        let x = contiguous(MLXRandom.normal([8, hidden]).asType(.bfloat16))
        eval(x)
        _ = Self.custom(x, w, .replica).map { eval($0) }
        eval(Self.mlx(x, w))

        let count = 4000
        let mlxBuild = Self.hostBuild(count) { Self.mlx(x, w) }
        let customBuild = Self.hostBuild(count) { Self.custom(x, w, .replica)! }

        // Source-length slope. Two trivial kernels with identical bodies and
        // different source lengths; the difference is the part of the host cost
        // that scales with the generated source string.
        let padding = String(repeating: "// e120 source length probe padding\n", count: 220)
        let tiny = MLXArray(Array(repeating: Float(1), count: 32))
        eval(tiny)
        let shortKernel = MLXFast.metalKernel(
            name: "e120_len_probe_short", inputNames: ["a"], outputNames: ["o"],
            source: "o[thread_position_in_grid.x] = a[thread_position_in_grid.x];",
            header: "", ensureRowContiguous: false)
        let longKernel = MLXFast.metalKernel(
            name: "e120_len_probe_long", inputNames: ["a"], outputNames: ["o"],
            source: "o[thread_position_in_grid.x] = a[thread_position_in_grid.x];",
            header: padding, ensureRowContiguous: false)
        let shortBuild = Self.hostBuild(count) {
            shortKernel(
                [tiny], grid: (32, 1, 1), threadGroup: (32, 1, 1),
                outputShapes: [[32]], outputDTypes: [.float32])[0]
        }
        let longBuild = Self.hostBuild(count) {
            longKernel(
                [tiny], grid: (32, 1, 1), threadGroup: (32, 1, 1),
                outputShapes: [[32]], outputDTypes: [.float32])[0]
        }

        let record: [String: Any] = [
            "mlx_graph_build_us": mlxBuild,
            "custom_graph_build_us": customBuild,
            "custom_minus_mlx_us": customBuild - mlxBuild,
            "len_probe_short_us": shortBuild,
            "len_probe_long_us": longBuild,
            "len_probe_padding_bytes": padding.utf8.count,
            "len_probe_us_per_kib": (longBuild - shortBuild) / (Double(padding.utf8.count) / 1024.0),
            "replicates": count,
        ]
        print(
            "E120_HOSTCOST "
                + (String(
                    data: (try? JSONSerialization.data(
                        withJSONObject: record, options: [.sortedKeys])) ?? Data(),
                    encoding: .utf8) ?? "{}"))
        fflush(stdout)
        #expect(customBuild > 0)
    }

    // MARK: rung 1 matched timing

    @Test(
        "matched ABBA timing of the replica against MLX",
        .enabled(if: E120CustomQMVProbeTests.timingEnabled))
    func matchedTiming() throws {
        var cells: [[String: Any]] = []

        for shape in Self.shapes {
            let w = Self.makeWeights(
                hidden: shape.hidden, outputs: shape.outputs,
                seed: UInt64(0xE120) &+ UInt64(shape.outputs))
            let maxWidth = Self.widths.max() ?? 9
            MLXRandom.seed(UInt64(0xB1F5) &+ UInt64(shape.outputs))
            let block = MLXRandom.normal([maxWidth, shape.hidden]).asType(.bfloat16)
            eval(block)

            for width in Self.widths {
                let x = contiguous(block[0 ..< width])
                eval(x)

                var arms: [E120Arm] = [
                    E120Arm(name: "a_mlx", body: { [Self.mlx(x, w)] })
                ]
                if Self.custom(x, w, .replica) != nil {
                    arms.append(
                        E120Arm(
                            name: "b_replica",
                            body: { [Self.custom(x, w, .replica)!] }))
                }

                Self.rampBurst(arms[0].body, seconds: Self.rampSeconds)
                let probeUs = Self.timed(8, arms[0].body)
                let count = max(8, min(600, Int(Self.targetMicroseconds / max(probeUs, 1.0))))
                for arm in arms { _ = Self.timed(3, arm.body) }

                for blockIndex in 0 ..< Self.blocks {
                    let entryTemp = e120GPUTemperature()
                    Self.rampBurst(arms[0].body, seconds: Self.rampSeconds)
                    let forward = arms.map { Self.timed(count, $0.body) }
                    let reverse = Array(
                        arms.reversed().map { Self.timed(count, $0.body) }.reversed())
                    let exitTemp = e120GPUTemperature()

                    var record: [String: Any] = [
                        "shape": shape.name,
                        "outputs": shape.outputs,
                        "hidden": shape.hidden,
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
                    cells.append(record)
                    print(
                        "E120_BLOCK "
                            + (String(
                                data: (try? JSONSerialization.data(
                                    withJSONObject: record, options: [.sortedKeys])) ?? Data(),
                                encoding: .utf8) ?? "{}"))
                    fflush(stdout)
                }
                eval(MLXArray(0))
            }
            Memory.clearCache()
        }

        if let path = ProcessInfo.processInfo.environment["MLXFAST_E120_OUT"], !path.isEmpty {
            let payload: [String: Any] = [
                "group_size": Self.groupSize,
                "bits": Self.bits,
                "ramp_seconds": Self.rampSeconds,
                "target_us": Self.targetMicroseconds,
                "cool_gate_passed_real_gate": false,
                "gate_qualified_for_timing": false,
                "blocks": Self.blocks,
                "widths": Self.widths,
                "cells": cells,
            ]
            let data = try JSONSerialization.data(
                withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
            try data.write(to: URL(fileURLWithPath: path))
        }

        #expect(!cells.isEmpty)
    }
}

/// One macmon sample. The probe runs under no thermal gate, so the entry and
/// exit temperature of every block is the thermal record.
private func e120GPUTemperature() -> Double? {
    let binary =
        ProcessInfo.processInfo.environment["MLXFAST_E120_MACMON"]
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
