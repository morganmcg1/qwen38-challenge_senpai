import CryptoKit
import Foundation
import MLX
@testable import MLXLLM
import Testing

// E170 -- halve the register block of the wide affine-4/group-64 QMV at the
// widths whose streaming efficiency is deficient.
//
// THE MECHANISM. `qwen_e120_qmv_wide` holds `acc[rows_per_simd]` and
// `partial[rows_per_simd]`, each a `vec<float, NA>`, plus four staged
// activation vectors of the same width. Per-thread register state therefore
// scales as `rows_per_simd * NA`. The E169 pass-cost model measured the kernel
// at 96.0 % and 98.1 % of the 273 GB/s stream roof at NA = 2 and NA = 3, then
// 86.1 % and 73.6 % at NA = 4 and NA = 5, with compute never above 47 % of the
// FLOP roof. `ROWS_PER_SIMD` is now a template parameter, so the launcher can
// choose 2 rows per simdgroup at the deficient widths and keep 4 elsewhere.
//
// TWO SUITES.
//
// `E170RowsPerSimdExactnessTests` is the gate. It digests the kernel bits at
// every scored shape and every routed width for `ROWS_PER_SIMD` 4 and 2, and
// against MLX's own launcher, then runs two positive controls that must fail.
// Untimed on purpose.
//
// `E170RowsPerSimdScreenTests` is the matched directional screen. Both arms run
// in ONE process against ONE build, ABBA-counterbalanced inside every
// (shape, width) cell, so nothing but `ROWS_PER_SIMD` differs between them.
//
// Nothing here is on the submitted surface: Yukon never packages `Tests/`.

private struct E170Shape {
    let name: String
    let k: Int
    let n: Int
    let callsPerVerify: Int
}

/// The seven affine-4/group-64 transposed projections one verify forward runs,
/// with the number of times it reaches each. Read off the live target the same
/// way `e169ScoredShapes` was, and identical to it.
private let e170ScoredShapes: [E170Shape] = [
    .init(name: "linear_attn.in_proj_fused_qkvzba", k: 5120, n: 16480, callsPerVerify: 48),
    .init(name: "linear_attn.out_proj", k: 6144, n: 5120, callsPerVerify: 48),
    .init(name: "full_attn.qkv_proj_fused", k: 5120, n: 14336, callsPerVerify: 16),
    .init(name: "full_attn.o_proj", k: 6144, n: 5120, callsPerVerify: 16),
    .init(name: "mlp.gate_up_fused", k: 5120, n: 34816, callsPerVerify: 64),
    .init(name: "mlp.down", k: 17408, n: 5120, callsPerVerify: 64),
    .init(name: "head.lm_head", k: 5120, n: 248320, callsPerVerify: 1),
]

// MARK: - exactness

/// Bit-for-bit gate on the row-blocking change.
///
/// For every scored shape and every routed width the suite digests three
/// outputs: MLX's own launcher, the candidate dispatch at 4 rows per simdgroup,
/// and the candidate dispatch at 2. All three must carry the same bits.
///
/// The digest is over the bf16 outputs widened to float32. That widening is
/// exact and injective, so equal digests mean equal bits, and it separates
/// `-0.0` from `+0.0` where a value comparison would not.
///
/// Enable with `MLXFAST_RUN_E170_EXACTNESS=1`. Point `MLXFAST_E170_EXACTNESS_OUT`
/// at a JSON destination to keep the per-cell digests.
@Suite(.serialized)
struct E170RowsPerSimdExactnessTests {
    private static var enabled: Bool {
        ProcessInfo.processInfo.environment["MLXFAST_RUN_E170_EXACTNESS"] == "1"
    }

    @Test(.enabled(if: E170RowsPerSimdExactnessTests.enabled))
    func rowBlockingDoesNotChangeOneBit() throws {
        let env = ProcessInfo.processInfo.environment
        let widths = e170ParseWidths(env["MLXFAST_E170_EXACTNESS_WIDTHS"]) ?? Array(2...9)

        var cells: [[String: Any]] = []
        var mismatchesAgainstRows4 = 0
        var mismatchesAgainstMLX = 0

        for shape in e170ScoredShapes {
            let weight = e170QuantWeight(k: shape.k, n: shape.n)
            for m in widths {
                let x = e170Activations(m: m, k: shape.k, salt: m)
                let incumbent = quantizedMM(
                    x, weight.w, scales: weight.scales, biases: weight.biases,
                    transpose: true, groupSize: 64, bits: 4)
                let rows4 = try #require(
                    Qwen35CustomQMV.matmul(
                        x, weight.w, scales: weight.scales, biases: weight.biases,
                        groupSize: 64, bits: 4, mode: .affine,
                        arm: .sumTable, rowsPerSimd: 4),
                    "the candidate dispatch must own \(shape.name) at m=\(m)")
                let rows2 = try #require(
                    Qwen35CustomQMV.matmul(
                        x, weight.w, scales: weight.scales, biases: weight.biases,
                        groupSize: 64, bits: 4, mode: .affine,
                        arm: .sumTable, rowsPerSimd: 2))

                let dIncumbent = e170Digest(incumbent)
                let d4 = e170Digest(rows4)
                let d2 = e170Digest(rows2)
                if d2 != d4 { mismatchesAgainstRows4 += 1 }
                if d4 != dIncumbent { mismatchesAgainstMLX += 1 }
                #expect(
                    d2 == d4,
                    "\(shape.name) m=\(m): 2 rows per simdgroup changed the bits")
                #expect(
                    d4 == dIncumbent,
                    "\(shape.name) m=\(m): the replica no longer matches MLX")

                cells.append([
                    "shape": shape.name, "m": m, "k": shape.k, "n": shape.n,
                    "digest_mlx": dIncumbent, "digest_rows4": d4, "digest_rows2": d2,
                    "rows_per_simd_shipped": Qwen35CustomQMV.rowsPerSimd(m),
                ])
                Memory.clearCache()
            }
        }

        // POSITIVE CONTROL 1 -- the digest must move when the arithmetic does.
        // One activation is nudged by a single bf16 ulp at a cell the touched
        // output rows read. Real floating-point values, not integers: the
        // activation tile is in [-0.5, 0.5) and the perturbed entry stays
        // there. If this control ever passes silently, the whole comparison
        // above is worthless.
        let control = e170ScoredShapes[1]
        let controlWeight = e170QuantWeight(k: control.k, n: control.n)
        let clean = e170Activations(m: 5, k: control.k, salt: 5)
        let perturbed = e170PerturbOneULP(clean, row: 3, column: 1_777)
        #expect(
            e170Digest(clean) != e170Digest(perturbed),
            "the one-ulp perturbation did not change the activations")
        let cleanOut = try #require(
            Qwen35CustomQMV.matmul(
                clean, controlWeight.w, scales: controlWeight.scales,
                biases: controlWeight.biases, groupSize: 64, bits: 4,
                mode: .affine, arm: .sumTable, rowsPerSimd: 2))
        let perturbedOut = try #require(
            Qwen35CustomQMV.matmul(
                perturbed, controlWeight.w, scales: controlWeight.scales,
                biases: controlWeight.biases, groupSize: 64, bits: 4,
                mode: .affine, arm: .sumTable, rowsPerSimd: 2))
        let ulpControlFails = e170Digest(cleanOut) != e170Digest(perturbedOut)
        #expect(ulpControlFails, "a one-ulp activation change left the output bits alone")

        // POSITIVE CONTROL 2 -- the row-coverage check must be able to fail.
        // The kernel compiles for 2 rows per simdgroup while the grid is sized
        // for 4, so only the leading half of the output rows is ever written.
        // A comparison that cannot see this cannot see a dropped row.
        let starvedGrid = try #require(
            Qwen35CustomQMV.matmulWithTable(
                clean, controlWeight.w, scales: controlWeight.scales,
                biases: controlWeight.biases,
                xsums: Qwen35CustomQMV.xsumsTable(clean),
                groupSize: 64, bits: 4, mode: .affine,
                rowsPerSimd: 2, gridRowsPerSimd: 4))
        let coverageControlFails = e170Digest(starvedGrid) != e170Digest(cleanOut)
        #expect(
            coverageControlFails,
            "a grid that covers half the output rows produced the correct bits")

        let payload: [String: Any] = [
            "experiment": "e170-rows-per-simd-exactness",
            "harness": "local",
            "widths": widths,
            "cells": cells,
            "mismatches_rows2_vs_rows4": mismatchesAgainstRows4,
            "mismatches_rows4_vs_mlx": mismatchesAgainstMLX,
            "control_one_ulp_activation_changes_output": ulpControlFails,
            "control_starved_grid_changes_output": coverageControlFails,
            "device": e170Device(),
        ]
        if let out = env["MLXFAST_E170_EXACTNESS_OUT"] {
            let data = try JSONSerialization.data(
                withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
            try data.write(to: URL(fileURLWithPath: out))
            print("E170_EXACTNESS_OUT \(out)")
        }
        print(
            "E170_EXACTNESS cells=\(cells.count) "
                + "mismatch_rows2_vs_rows4=\(mismatchesAgainstRows4) "
                + "mismatch_rows4_vs_mlx=\(mismatchesAgainstMLX) "
                + "control_ulp=\(ulpControlFails) control_grid=\(coverageControlFails)")
    }
}

// MARK: - timing screen

/// One matched directional measurement of the row-blocking change.
///
/// Both arms are the same build, the same process and the same resident
/// weights; the only difference is the `ROWS_PER_SIMD` template argument and
/// the grid extent it implies. Every (shape, width) cell runs ABBA, so
/// monotone thermal drift cancels to first order and the two samples of one
/// arm bound the within-cell noise without spending a separate null block.
///
/// Enable with `MLXFAST_RUN_E170_SCREEN=1` and point `MLXFAST_E170_SCREEN_OUT`
/// at the JSON destination.
@Suite(.serialized)
struct E170RowsPerSimdScreenTests {
    private static var enabled: Bool {
        ProcessInfo.processInfo.environment["MLXFAST_RUN_E170_SCREEN"] == "1"
    }

    @Test(.enabled(if: E170RowsPerSimdScreenTests.enabled))
    func priceTheRowBlockAtEveryDeficientWidth() throws {
        let env = ProcessInfo.processInfo.environment
        let outPath = try #require(
            env["MLXFAST_E170_SCREEN_OUT"],
            "MLXFAST_E170_SCREEN_OUT must name the JSON destination")
        // NA = 5 first. Advisor F1 makes M = 5 the cell the candidate would
        // spend all of its time in if fixed depth 4 promotes. M = 4 is the
        // NA = 4 target, and M = 3 is the near-roof control that should not
        // gain, because NA = 3 already streams at 98.1 % of roof.
        let widths = e170ParseWidths(env["MLXFAST_E170_SCREEN_WIDTHS"]) ?? [5, 4, 3]
        let reps = Int(env["MLXFAST_E170_SCREEN_REPS"] ?? "") ?? 9
        let inner = Int(env["MLXFAST_E170_SCREEN_INNER"] ?? "") ?? 8

        var payload: [String: Any] = [
            "experiment": "e170-rows-per-simd-screen",
            "harness": "local",
            "entry_point": "Qwen35CustomQMV.matmul",
            "cool_gate_passed_real_gate": false,
            "gate_qualified_for_timing": false,
            "official_or_ranked_score": false,
            "widths": widths,
            "reps": reps,
            "inner_calls_per_timed_region": inner,
            "arms_rows_per_simd": [4, 2],
            "block_order": "ABBA",
            "device": e170Device(),
            "custom_qmv_arm": String(describing: Qwen35CustomQMV.arm),
        ]
        if let temp = e170GPUTemperature() { payload["gpu_temp_entry_c"] = temp }

        // The first shape in a cold process carries the JIT of both template
        // instantiations plus the ramp from idle clocks. Spend it on a
        // discarded pass so it lands on nobody's slope.
        _ = e170SweepShape(e170ScoredShapes[0], widths: widths, reps: 2, inner: inner)
        Memory.clearCache()

        var records: [[String: Any]] = []
        for shape in e170ScoredShapes {
            var record = e170SweepShape(shape, widths: widths, reps: reps, inner: inner)
            if let temp = e170GPUTemperature() { record["gpu_temp_exit_c"] = temp }
            records.append(record)
            Memory.clearCache()
        }
        payload["shapes"] = records
        if let temp = e170GPUTemperature() { payload["gpu_temp_exit_c"] = temp }

        let data = try JSONSerialization.data(
            withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: URL(fileURLWithPath: outPath))
        print("E170_SCREEN_OUT \(outPath)")
    }
}

private func e170SweepShape(
    _ shape: E170Shape, widths: [Int], reps: Int, inner: Int
) -> [String: Any] {
    let weight = e170QuantWeight(k: shape.k, n: shape.n)
    var rows: [[String: Any]] = []
    for m in widths {
        let xs = (0..<inner).map { e170Activations(m: m, k: shape.k, salt: $0) }
        // ABBA within the cell: 4, 2, 2, 4.
        for (block, arm) in [4, 2, 2, 4].enumerated() {
            let samples = e170MedianSecondsPerCall(reps: reps, inner: inner) {
                var outs: [MLXArray] = []
                outs.reserveCapacity(inner)
                var x = xs[0]
                for index in 0..<inner {
                    guard
                        let out = Qwen35CustomQMV.matmul(
                            x, weight.w, scales: weight.scales, biases: weight.biases,
                            groupSize: 64, bits: 4, mode: .affine,
                            arm: .sumTable, rowsPerSimd: arm)
                    else { preconditionFailure("E170: \(shape.name) m=\(m) is not routable") }
                    outs.append(out)
                    // Chain the calls: MLX would otherwise encode independent
                    // matvecs concurrently and understate the dependent cost.
                    if index + 1 < inner { x = xs[index + 1] + out[0..., 0..<1, 0..<1] * 1e-30 }
                }
                return outs
            }
            rows.append([
                "m": m,
                "rows_per_simd": arm,
                "block": block,
                "seconds_per_call": samples[samples.count / 2],
                "seconds_per_call_min": samples[0],
                "seconds_per_call_max": samples[samples.count - 1],
            ])
        }
    }
    return [
        "name": shape.name,
        "k": shape.k,
        "n": shape.n,
        "calls_per_verify": shape.callsPerVerify,
        "rows": rows,
    ]
}

// MARK: - helpers

private struct E170QuantWeight {
    let w: MLXArray
    let scales: MLXArray
    let biases: MLXArray
}

/// Deterministic packed affine-4 g64 weights at the scored `[n, k]`, identical
/// in construction to `e169QuantWeight`. Values do not change kernel time;
/// shape, dtype and contiguity do.
private func e170QuantWeight(k: Int, n: Int) -> E170QuantWeight {
    let words = k / 8
    let tile = (0..<words).map { index -> UInt32 in
        UInt32(truncatingIfNeeded: index &* 2_654_435_761) ^ 0x9E37_79B9
    }
    let w = MLXArray(tile).reshaped([1, words]) + arange(0, n, dtype: .uint32).reshaped([n, 1])
    let groups = k / 64
    let jitter = arange(0, n, dtype: .float32).reshaped([n, 1]) * 1e-6
    let scaleTile: [Float] = (0..<groups).map { 0.006 + 0.004 * Float(($0 &* 37) % 61) / 61.0 }
    let biasTile: [Float] = (0..<groups).map { -0.05 - 0.02 * Float(($0 &* 23) % 53) / 53.0 }
    let scales = (MLXArray(scaleTile).reshaped([1, groups]) + jitter).asType(.bfloat16)
    let biases = (MLXArray(biasTile).reshaped([1, groups]) + jitter).asType(.bfloat16)
    eval(w, scales, biases)
    return E170QuantWeight(w: w, scales: scales, biases: biases)
}

private func e170Activations(m: Int, k: Int, salt: Int) -> MLXArray {
    let row = (0..<k).map { index -> Float in
        Float(((index &* 2_246_822_519) &+ salt &* 374_761_393) % 1_009) / 1_009.0 - 0.5
    }
    let base = MLXArray(row).reshaped([1, 1, k])
    let ramp = (arange(0, m, dtype: .float32) * 1e-3).reshaped([1, m, 1])
    let x = (base + ramp).asType(.bfloat16)
    eval(x)
    return x
}

/// Moves one bf16 activation by a single ulp, keeping the tensor contiguous and
/// the value a normal float. The row and column pick a cell the wide kernel
/// really reads: any `k` a lane owns feeds every output row of the call.
private func e170PerturbOneULP(_ x: MLXArray, row: Int, column: Int) -> MLXArray {
    let bits = x.asType(.bfloat16).view(dtype: .uint16)
    var flat = bits.reshaped([-1]).asArray(UInt16.self)
    let k = x.dim(-1)
    flat[row * k + column] &+= 1
    let moved = MLXArray(flat).reshaped(x.shape).view(dtype: .bfloat16)
    eval(moved)
    return moved
}

/// SHA-256 over the float32 widening of a bf16 tensor. bf16 to float32 is
/// exact and injective, so equal digests mean equal bits.
private func e170Digest(_ a: MLXArray) -> String {
    eval(a)
    let values = a.asType(.float32).asArray(Float.self)
    var hasher = SHA256()
    values.withUnsafeBytes { hasher.update(bufferPointer: $0) }
    return hasher.finalize().map { String(format: "%02x", $0) }.joined()
}

/// Median seconds per call over `reps` timed regions of `inner` chained calls.
private func e170MedianSecondsPerCall(
    reps: Int, inner: Int, warmup: Int = 3, _ body: () -> [MLXArray]
) -> [Double] {
    for _ in 0..<warmup { eval(body()) }
    var samples: [Double] = []
    samples.reserveCapacity(reps)
    for _ in 0..<reps {
        let start = DispatchTime.now().uptimeNanoseconds
        eval(body())
        let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
        samples.append(elapsed / Double(inner))
    }
    return samples.sorted()
}

private func e170ParseWidths(_ raw: String?) -> [Int]? {
    guard let raw, !raw.isEmpty else { return nil }
    let widths = raw.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
    return widths.isEmpty ? nil : widths
}

private func e170Device() -> [String: Any] {
    let device = MLX.GPU.deviceInfo()
    return [
        "architecture": device.architecture,
        "memory_size": device.memorySize,
        "max_recommended_working_set_size": Int(device.maxRecommendedWorkingSetSize),
    ]
}

private func e170GPUTemperature() -> Double? {
    let binary =
        ProcessInfo.processInfo.environment["MLXFAST_E170_MACMON"] ?? "/opt/homebrew/bin/macmon"
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
