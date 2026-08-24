import Foundation
import MLX
import MLXFast
import MLXRandom
import Testing

@testable import MLXLLM

// E189 -- can the wide affine-4/group-64 QMV serve m = 6, 7, 8 in ONE
// weight-streaming pass?
//
// The shipped `staged` width plan keeps IPG <= 5, so m = 6, 7 and 8 launch two
// input-row groups and each group streams the whole weight matrix. The
// `singlepass` plan sets IPG = m at those widths, so the weights stream once.
//
// Two sections, both opt-in:
//
//   * `gate` compares the two plans cell by cell with ACTUAL bfloat16
//     activations and packed 4-bit weights, reports per-cell max ULP, and
//     runs a positive control that proves the comparison can fail;
//   * `probe` times both plans per cell and width with ABBA-counterbalanced
//     block ordering and writes raw samples as JSON.
//
// Research instrument. `Tests/` is never packaged into a submission.
// Within-session relative measurement, harness=local, no thermal gate,
// no score.

private struct E189Cell {
    var name: String
    var k: Int
    var n: Int
    /// Wide-QMV invocations of this cell in one decode round (FINDING 487).
    var invocations: Int
}

private let e189AllCells = [
    E189Cell(name: "mlp.gate_up", k: 5120, n: 34816, invocations: 64),
    E189Cell(name: "mlp.down", k: 17408, n: 5120, invocations: 64),
    E189Cell(name: "gdn.in_proj", k: 5120, n: 16480, invocations: 48),
    E189Cell(name: "gdn.out_proj", k: 6144, n: 5120, invocations: 48),
    E189Cell(name: "fa.qkv", k: 5120, n: 14336, invocations: 16),
    E189Cell(name: "fa.o_proj", k: 6144, n: 5120, invocations: 16),
    E189Cell(name: "lm_head", k: 5120, n: 248_320, invocations: 1),
]

/// `MLXFAST_E189_CELLS` selects a subset for a smoke run; the full seven-cell
/// list is the default and the only list a reported result may use.
private let e189Cells: [E189Cell] = {
    guard
        let raw = ProcessInfo.processInfo.environment["MLXFAST_E189_CELLS"],
        !raw.isEmpty
    else { return e189AllCells }
    let wanted = Set(raw.split(separator: ",").map(String.init))
    return e189AllCells.filter { wanted.contains($0.name) }
}()

/// A JIT kernel pair for one width plan, built from the production source so
/// the instrument cannot drift from the shipped text. The shipped decode path
/// launches through `Qwen35CachedKernel`, so the instrument does too: a
/// `MLXFastKernel` call would add a per-call config build that the scored path
/// does not pay.
private struct E189Pipeline {
    var plan: Qwen35QMVWidthPlan
    var label: String
    var table: Qwen35CachedKernel
    var plain: Qwen35CachedKernel

    init(
        plan: Qwen35QMVWidthPlan, label: String,
        header: String? = nil, nameSuffix: String = ""
    ) {
        self.plan = plan
        self.label = label
        let head = header ?? qwen35E120QMVHeader
        self.table = Qwen35CachedKernel(
            name: "e189_qmv_table_\(label)\(nameSuffix)",
            inputNames: ["w", "scales", "biases", "x", "xsums"],
            outputNames: ["y"],
            source: qwen35E120QMVSource(table: true, plan: plan),
            header: head)
        self.plain = Qwen35CachedKernel(
            name: "e189_qmv_plain_\(label)\(nameSuffix)",
            inputNames: ["w", "scales", "biases", "x"],
            outputNames: ["y"],
            source: qwen35E120QMVSource(table: false, plan: plan),
            header: head)
    }

    func call(
        x: MLXArray, w: MLXArray, scales: MLXArray, biases: MLXArray,
        xsums: MLXArray, m: Int, n: Int, useTable: Bool
    ) -> MLXArray {
        let groups = Qwen35CustomQMV.activeInputGroups(m, plan: plan)
        var outShape = x.shape
        outShape[outShape.count - 1] = n
        let launch = Qwen35KernelLaunch(
            grid: (groups * 32, (n / 8) * 2, 1),
            threadGroup: (32, 2, 1),
            outputShape: outShape.map(Int32.init),
            outputDType: .bfloat16,
            useTable: useTable ? true : nil)
        if useTable {
            return table([w, scales, biases, x, xsums], launch)
        }
        return plain([w, scales, biases, x], launch)
    }
}

/// affine 4-bit group-64 packed weights and real bfloat16 activations at a
/// scored cell shape. The gate compares floating-point results, so the packed
/// nibbles and the scales carry the value range a decode round really sees.
private func e189PackedCell(k: Int, n: Int, seed: UInt64)
    -> (MLXArray, MLXArray, MLXArray)
{
    MLXRandom.seed(seed)
    let w = MLXRandom.uniform(low: 0.0, high: 4.2e9, [n, k / 8]).asType(.uint32)
    let scales = MLXRandom.uniform(low: 0.005, high: 0.03, [n, k / 64])
        .asType(.bfloat16)
    let biases = MLXRandom.uniform(low: -0.05, high: 0.05, [n, k / 64])
        .asType(.bfloat16)
    eval(w, scales, biases)
    return (w, scales, biases)
}

private func e189Activations(m: Int, k: Int, seed: UInt64) -> MLXArray {
    MLXRandom.seed(seed)
    let x = MLXRandom.normal([1, m, k]).asType(.bfloat16)
    eval(x)
    return x
}

/// bfloat16 total-order key. bfloat16 is the top half of the float32 bit
/// pattern, so the key orders the finite values monotonically and the
/// difference of two keys is the ULP distance.
private func e189Key(_ value: Float) -> Int {
    let bits = Int(value.bitPattern >> 16)
    let magnitude = bits & 0x7fff
    return (bits & 0x8000) != 0 ? -magnitude : magnitude
}

private struct E189Diff {
    var count: Int
    var maxUlp: Int
    var maxAbsDiff: Double
    var nanOrInf: Int
}

private func e189Compare(_ a: MLXArray, _ b: MLXArray) -> E189Diff {
    let av = a.asType(.float32).asArray(Float.self)
    let bv = b.asType(.float32).asArray(Float.self)
    var diff = E189Diff(count: 0, maxUlp: 0, maxAbsDiff: 0, nanOrInf: 0)
    for i in 0 ..< min(av.count, bv.count) {
        let x = av[i]
        let y = bv[i]
        if !x.isFinite || !y.isFinite { diff.nanOrInf += 1 }
        if x.bitPattern == y.bitPattern { continue }
        diff.count += 1
        diff.maxUlp = max(diff.maxUlp, abs(e189Key(x) - e189Key(y)))
        diff.maxAbsDiff = max(diff.maxAbsDiff, Double(abs(x - y)))
    }
    return diff
}

/// Microseconds per kernel call, from `chain` dispatches behind one blocking
/// `eval`. One sync per call would measure the host round trip, which is the
/// same for both arms and several times the cell's own device time; chaining
/// lets the device time dominate the sample.
private func e189Timed(reps: Int, chain: Int, _ body: (Int) -> MLXArray)
    -> Double
{
    let start = DispatchTime.now().uptimeNanoseconds
    for rep in 0 ..< reps {
        var outputs: [MLXArray] = []
        outputs.reserveCapacity(chain)
        for index in 0 ..< chain {
            outputs.append(body(rep &* chain &+ index))
        }
        eval(outputs)
    }
    return Double(DispatchTime.now().uptimeNanoseconds - start) / 1e3
        / Double(reps * chain)
}

private func e189GpuTemperature() -> Double? {
    for path in [
        ProcessInfo.processInfo.environment["MLXFAST_MACMON_BIN"] ?? "",
        "\(FileManager.default.homeDirectoryForCurrentUser.path)/bin/macmon",
        "/opt/homebrew/bin/macmon", "/usr/local/bin/macmon",
    ] where !path.isEmpty && FileManager.default.isExecutableFile(atPath: path) {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: path)
        process.arguments = ["pipe", "-s1"]
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = FileHandle.nullDevice
        guard (try? process.run()) != nil else { continue }
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        guard
            let object = try? JSONSerialization.jsonObject(with: data)
                as? [String: Any],
            let temp = object["temp"] as? [String: Any],
            let gpu = temp["gpu_temp_avg"] as? Double
        else { continue }
        return gpu
    }
    return nil
}

private func e189Write(_ payload: [String: Any], to key: String) throws {
    let path = try #require(
        ProcessInfo.processInfo.environment[key],
        "\(key) must name the JSON destination")
    let data = try JSONSerialization.data(
        withJSONObject: payload,
        options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
    try data.write(to: URL(fileURLWithPath: path))
}

@Suite("E189 single-pass QMV")
struct E189SinglePassQMVTests {
    static let gateEnabled =
        ProcessInfo.processInfo.environment["MLXFAST_RUN_E189_GATE"] == "1"
    static let probeEnabled =
        ProcessInfo.processInfo.environment["MLXFAST_RUN_E189_PROBE"] == "1"

    static let gateWidths =
        (ProcessInfo.processInfo.environment["MLXFAST_E189_GATE_WIDTHS"]?
            .split(separator: ",").compactMap { Int($0) }).flatMap {
            $0.isEmpty ? nil : $0
        } ?? [6, 7, 8]
    static let probeWidths =
        (ProcessInfo.processInfo.environment["MLXFAST_E189_PROBE_WIDTHS"]?
            .split(separator: ",").compactMap { Int($0) }).flatMap {
            $0.isEmpty ? nil : $0
        } ?? [5, 6, 7, 8]

    // MARK: - numerical gate

    @Test(.enabled(if: E189SinglePassQMVTests.gateEnabled))
    func numericalGate() throws {
        let staged = E189Pipeline(plan: .staged, label: "staged")
        let single = E189Pipeline(plan: .singlePass, label: "singlepass")

        // Positive control: the same single-pass source with one accumulation
        // perturbed. The comparison must report differences for this build,
        // otherwise a pass proves nothing.
        let perturbedHeader = qwen35E120QMVHeader.replacingOccurrences(
            of: "acc[r] += scale_local[r] * partial[r] + sums * bias_local[r];",
            with:
                "acc[r] += scale_local[r] * partial[r] * 1.0000305f "
                + "+ sums * bias_local[r];")
        #expect(perturbedHeader != qwen35E120QMVHeader)
        let perturbed = E189Pipeline(
            plan: .singlePass, label: "perturbed", header: perturbedHeader,
            nameSuffix: "_ctl")

        var rows: [[String: Any]] = []
        var control: [[String: Any]] = []
        var worstUlp = 0
        var totalDiffering = 0

        for cell in e189Cells {
            try autoreleasepool {
                let (w, scales, biases) = e189PackedCell(
                    k: cell.k, n: cell.n, seed: 0xE189)
                for m in Self.gateWidths {
                    let x = e189Activations(
                        m: m, k: cell.k, seed: UInt64(0xE189_0000 + m))
                    let xsums = Qwen35CustomQMV.xsumsTable(x)
                    eval(xsums)
                    for useTable in [true, false] {
                        let a = single.call(
                            x: x, w: w, scales: scales, biases: biases,
                            xsums: xsums, m: m, n: cell.n, useTable: useTable)
                        let b = staged.call(
                            x: x, w: w, scales: scales, biases: biases,
                            xsums: xsums, m: m, n: cell.n, useTable: useTable)
                        eval(a, b)
                        let diff = e189Compare(a, b)
                        worstUlp = max(worstUlp, diff.maxUlp)
                        totalDiffering += diff.count
                        rows.append([
                            "cell": cell.name, "k": cell.k, "n": cell.n, "m": m,
                            "use_table": useTable,
                            "elements": m * cell.n,
                            "differing": diff.count,
                            "max_ulp": diff.maxUlp,
                            "max_abs_diff": diff.maxAbsDiff,
                            "non_finite": diff.nanOrInf,
                            "staged_groups": Qwen35CustomQMV.activeInputGroups(
                                m, plan: .staged),
                            "single_pass_groups": Qwen35CustomQMV
                                .activeInputGroups(m, plan: .singlePass),
                        ])
                        #expect(
                            diff.count == 0,
                            """
                            \(cell.name) m=\(m) table=\(useTable): \
                            \(diff.count) differing outputs, \
                            max_ulp=\(diff.maxUlp)
                            """)
                        #expect(diff.nanOrInf == 0)
                    }

                    if cell.name == "gdn.out_proj" {
                        for useTable in [true, false] {
                            let bad = perturbed.call(
                                x: x, w: w, scales: scales, biases: biases,
                                xsums: xsums, m: m, n: cell.n,
                                useTable: useTable)
                            let good = staged.call(
                                x: x, w: w, scales: scales, biases: biases,
                                xsums: xsums, m: m, n: cell.n,
                                useTable: useTable)
                            eval(bad, good)
                            let diff = e189Compare(bad, good)
                            control.append([
                                "cell": cell.name, "m": m,
                                "use_table": useTable,
                                "differing": diff.count,
                                "max_ulp": diff.maxUlp,
                                "max_abs_diff": diff.maxAbsDiff,
                            ])
                            #expect(
                                diff.count > 0,
                                "positive control did not trip at m=\(m)")
                        }
                    }
                }
            }
        }

        try e189Write(
            [
                "harness": "local",
                "experiment": "e189",
                "section": "numerical_gate",
                "cool_gate_passed_real_gate": false,
                "gate_qualified_for_timing": false,
                "official_or_ranked_score": false,
                "active_plan": Qwen35QMVWidthPlan.active.rawValue,
                "qmv_arm": Qwen35CustomQMV.arm.rawValue,
                "worst_max_ulp": worstUlp,
                "total_differing": totalDiffering,
                "cells": rows,
                "positive_control": control,
            ], to: "MLXFAST_E189_GATE_OUT")
    }

    // MARK: - per-cell timing screen

    @Test(.enabled(if: E189SinglePassQMVTests.probeEnabled))
    func widthProbe() throws {
        let env = ProcessInfo.processInfo.environment
        let blocks = Int(env["MLXFAST_E189_BLOCKS"] ?? "") ?? 8
        let warmup = Int(env["MLXFAST_E189_WARMUP"] ?? "") ?? 6
        let reps = Int(env["MLXFAST_E189_REPS"] ?? "") ?? 4
        let chain = Int(env["MLXFAST_E189_CHAIN"] ?? "") ?? 8
        let variants = Int(env["MLXFAST_E189_VARIANTS"] ?? "") ?? 4

        let staged = E189Pipeline(plan: .staged, label: "staged")
        let single = E189Pipeline(plan: .singlePass, label: "singlepass")

        var samples: [[String: Any]] = []
        var temperatures: [[String: Any]] = []
        func recordTemperature(_ label: String) {
            temperatures.append([
                "label": label,
                "gpu_temp_c": e189GpuTemperature() ?? -1,
                "seconds": Date().timeIntervalSince1970,
            ])
        }

        recordTemperature("session_entry")
        for cell in e189Cells {
            try autoreleasepool {
                let (w, scales, biases) = e189PackedCell(
                    k: cell.k, n: cell.n, seed: 0xE189)
                for m in Self.probeWidths {
                    // Distinct activation tensors, cycled inside a chain, so no
                    // graph-level reuse can serve a repeat from an earlier
                    // result instead of relaunching the kernel.
                    let inputs: [(MLXArray, MLXArray)] = (0 ..< variants).map {
                        variant in
                        let x = e189Activations(
                            m: m, k: cell.k,
                            seed: UInt64(0xE189_0000 + m * 16 + variant))
                        let xsums = Qwen35CustomQMV.xsumsTable(x)
                        eval(xsums)
                        return (x, xsums)
                    }
                    let arms: [(String, E189Pipeline)] = [
                        ("staged", staged), ("singlepass", single),
                    ]
                    for (_, pipeline) in arms {
                        for index in 0 ..< warmup {
                            let (x, xsums) = inputs[index % inputs.count]
                            eval(
                                pipeline.call(
                                    x: x, w: w, scales: scales, biases: biases,
                                    xsums: xsums, m: m, n: cell.n,
                                    useTable: true))
                        }
                    }
                    for block in 0 ..< blocks {
                        // ABBA: the arm order alternates so that monotone
                        // drift across the block pair cancels to first order.
                        let ascending = block % 2 == 0
                        let order = ascending ? arms : arms.reversed()
                        for (position, entry) in order.enumerated() {
                            let us = e189Timed(reps: reps, chain: chain) {
                                index in
                                let (x, xsums) = inputs[index % inputs.count]
                                return entry.1.call(
                                    x: x, w: w, scales: scales, biases: biases,
                                    xsums: xsums, m: m, n: cell.n,
                                    useTable: true)
                            }
                            samples.append([
                                "cell": cell.name, "k": cell.k, "n": cell.n,
                                "m": m, "arm": entry.0, "block": block,
                                "ascending": ascending, "position": position,
                                "microseconds": us, "reps": reps,
                                "chain": chain,
                                "invocations_per_round": cell.invocations,
                                "groups": Qwen35CustomQMV.activeInputGroups(
                                    m, plan: entry.1.plan),
                            ])
                        }
                    }
                    recordTemperature("\(cell.name)_m\(m)")
                }
            }
        }
        recordTemperature("session_exit")

        try e189Write(
            [
                "harness": "local",
                "experiment": "e189",
                "section": "width_probe",
                "cool_gate_passed_real_gate": false,
                "gate_qualified_for_timing": false,
                "official_or_ranked_score": false,
                "active_plan": Qwen35QMVWidthPlan.active.rawValue,
                "blocks": blocks, "warmup": warmup, "reps": reps,
                "chain": chain, "variants": variants,
                "samples": samples,
                "temperatures": temperatures,
            ], to: "MLXFAST_E189_PROBE_OUT")
    }
}
