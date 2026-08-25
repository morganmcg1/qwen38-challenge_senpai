import Foundation
import MLX
import MLXFast
import MLXRandom
import Testing

@testable import MLXLLM

// E195 -- does a per-(cell, width) QMV plan beat both staged and all-cells
// single-pass?
//
// E189 proved the `_sp` single-pass kernels bit-exact and measured them end to
// end: a WIN at m = 6 and a large loss at m = 7 and m = 8. It also measured a
// per-cell split at m = 6, where six of the seven fused cells win and mlp.down
// loses. `qwen35QMVVariant` serves single-pass exactly where it wins.
//
// Three sections:
//
//   * `planTable` is a pure-function test of the plan itself and always runs.
//     It proves that every width outside m = 6 selects the shipped staged
//     kernel, so those widths keep byte-identical dispatch behavior;
//   * `numericalGate` compares the two compiled variants cell by cell with
//     ACTUAL bfloat16 activations and packed 4-bit weights, reports per-cell
//     max ULP, and runs a positive control that proves the comparison can fail;
//   * `lmHeadProbe` replicates the flagged staged lm_head column at m = 6, 7
//     and 8 and times both variants there.
//
// Research instrument. `Tests/` is never packaged into a submission.
// Within-session relative measurement, harness=local, no thermal gate,
// no score.

private struct E195Cell {
    var name: String
    var k: Int
    var n: Int
    /// Wide-QMV invocations of this cell in one decode round (FINDING 487).
    var invocations: Int
}

private let e195AllCells = [
    E195Cell(name: "mlp.gate_up", k: 5120, n: 34816, invocations: 64),
    E195Cell(name: "mlp.down", k: 17408, n: 5120, invocations: 64),
    E195Cell(name: "gdn.in_proj", k: 5120, n: 16480, invocations: 48),
    E195Cell(name: "gdn.out_proj", k: 6144, n: 5120, invocations: 48),
    E195Cell(name: "fa.qkv", k: 5120, n: 14336, invocations: 16),
    E195Cell(name: "fa.o_proj", k: 6144, n: 5120, invocations: 16),
    E195Cell(name: "lm_head", k: 5120, n: 248_320, invocations: 1),
]

/// `MLXFAST_E195_CELLS` selects a subset for a smoke run; the full seven-cell
/// list is the default and the only list a reported result may use.
private let e195Cells: [E195Cell] = {
    guard
        let raw = ProcessInfo.processInfo.environment["MLXFAST_E195_CELLS"],
        !raw.isEmpty
    else { return e195AllCells }
    let wanted = Set(raw.split(separator: ",").map(String.init))
    return e195AllCells.filter { wanted.contains($0.name) }
}()

/// A JIT kernel pair for one compiled variant, built from the production source
/// so the instrument cannot drift from the shipped text. The shipped decode path
/// launches through `Qwen35CachedKernel`, so the instrument does too: an
/// `MLXFastKernel` call would add a per-call config build that the scored path
/// does not pay.
private struct E195Pipeline {
    var variant: Qwen35QMVKernelVariant
    var label: String
    var table: Qwen35CachedKernel
    var plain: Qwen35CachedKernel

    init(
        variant: Qwen35QMVKernelVariant, label: String,
        header: String? = nil, nameSuffix: String = ""
    ) {
        self.variant = variant
        self.label = label
        let head = header ?? qwen35E120QMVHeader
        self.table = Qwen35CachedKernel(
            name: "e195_qmv_table_\(label)\(nameSuffix)",
            inputNames: ["w", "scales", "biases", "x", "xsums"],
            outputNames: ["y"],
            source: qwen35E120QMVSource(table: true, variant: variant),
            header: head)
        self.plain = Qwen35CachedKernel(
            name: "e195_qmv_plain_\(label)\(nameSuffix)",
            inputNames: ["w", "scales", "biases", "x"],
            outputNames: ["y"],
            source: qwen35E120QMVSource(table: false, variant: variant),
            header: head)
    }

    func call(
        x: MLXArray, w: MLXArray, scales: MLXArray, biases: MLXArray,
        xsums: MLXArray, m: Int, n: Int, useTable: Bool
    ) -> MLXArray {
        let groups = Qwen35CustomQMV.activeInputGroups(m, variant: variant)
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
private func e195PackedCell(k: Int, n: Int, seed: UInt64)
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

private func e195Activations(m: Int, k: Int, seed: UInt64) -> MLXArray {
    MLXRandom.seed(seed)
    let x = MLXRandom.normal([1, m, k]).asType(.bfloat16)
    eval(x)
    return x
}

/// bfloat16 total-order key. bfloat16 is the top half of the float32 bit
/// pattern, so the key orders the finite values monotonically and the
/// difference of two keys is the ULP distance.
private func e195Key(_ value: Float) -> Int {
    let bits = Int(value.bitPattern >> 16)
    let magnitude = bits & 0x7fff
    return (bits & 0x8000) != 0 ? -magnitude : magnitude
}

private struct E195Diff {
    var count: Int
    var maxUlp: Int
    var maxAbsDiff: Double
    var nanOrInf: Int
}

private func e195Compare(_ a: MLXArray, _ b: MLXArray) -> E195Diff {
    let av = a.asType(.float32).asArray(Float.self)
    let bv = b.asType(.float32).asArray(Float.self)
    var diff = E195Diff(count: 0, maxUlp: 0, maxAbsDiff: 0, nanOrInf: 0)
    for i in 0 ..< min(av.count, bv.count) {
        let x = av[i]
        let y = bv[i]
        if !x.isFinite || !y.isFinite { diff.nanOrInf += 1 }
        if x.bitPattern == y.bitPattern { continue }
        diff.count += 1
        diff.maxUlp = max(diff.maxUlp, abs(e195Key(x) - e195Key(y)))
        diff.maxAbsDiff = max(diff.maxAbsDiff, Double(abs(x - y)))
    }
    return diff
}

/// Microseconds per kernel call, from `chain` dispatches behind one blocking
/// `eval`. One sync per call would measure the host round trip, which is the
/// same for both arms and several times the cell's own device time; chaining
/// lets the device time dominate the sample.
private func e195Timed(reps: Int, chain: Int, _ body: (Int) -> MLXArray)
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

private func e195GpuTemperature() -> Double? {
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

private func e195Write(_ payload: [String: Any], to key: String) throws {
    let path = try #require(
        ProcessInfo.processInfo.environment[key],
        "\(key) must name the JSON destination")
    let data = try JSONSerialization.data(
        withJSONObject: payload,
        options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
    try data.write(to: URL(fileURLWithPath: path))
}

@Suite("E195 cell-selective QMV width plan")
struct E195CellSelectiveQMVTests {
    static let gateEnabled =
        ProcessInfo.processInfo.environment["MLXFAST_RUN_E195_GATE"] == "1"
    static let probeEnabled =
        ProcessInfo.processInfo.environment["MLXFAST_RUN_E195_PROBE"] == "1"

    static let gateWidths =
        (ProcessInfo.processInfo.environment["MLXFAST_E195_GATE_WIDTHS"]?
            .split(separator: ",").compactMap { Int($0) }).flatMap {
            $0.isEmpty ? nil : $0
        } ?? [6, 7, 8]

    // MARK: - plan table

    /// The shipped plan must leave every width it does not name on the staged
    /// kernel. This is the byte-identical-dispatch proof for m != 6, and it
    /// needs no GPU.
    @Test
    func planTable() {
        let cells: [Qwen35QMVCell] = [
            .mlpGateUp, .mlpDown, .gdnInProj, .outProj, .faQKV, .lmHead,
            .unlisted,
        ]
        for m in Qwen35CustomQMV.widths {
            for cell in cells {
                let variant = qwen35QMVVariant(m: m, cell: cell)
                if m == 6 {
                    let expected: Qwen35QMVKernelVariant =
                        (cell == .mlpDown || cell == .unlisted)
                        ? qwen35E208StagedVariant : .singlePass
                    #expect(variant == expected, "m=6 cell=\(cell)")
                } else {
                    // E208 put the m = 9 group partition behind a runtime arm,
                    // so the staged family is named by the arm rather than by
                    // the case. Single-pass must still never appear here.
                    #expect(
                        variant == qwen35E208StagedVariant, "m=\(m) cell=\(cell)"
                    )
                    #expect(variant != .singlePass, "m=\(m) cell=\(cell)")
                }
            }
        }

        // The dispatch routing must agree with the plan at the one width the
        // plan names, and stay staged everywhere else.
        #expect(
            Qwen35CustomQMV.kernelVariant((m: 6, k: 5120, n: 248_320))
                == .singlePass)
        #expect(
            Qwen35CustomQMV.kernelVariant((m: 6, k: 17408, n: 5120))
                == .staged)
        #expect(
            Qwen35CustomQMV.kernelVariant((m: 7, k: 5120, n: 248_320))
                == .staged)
        #expect(qwen35QMVWidthPlanWitness == "selective-m6+ipg9-5")

        // The launch witness must follow the compiled variant, not the width
        // alone.
        #expect(Qwen35CustomQMV.activeInputGroups(6, variant: .staged) == 2)
        #expect(Qwen35CustomQMV.activeInputGroups(6, variant: .singlePass) == 1)
        #expect(Qwen35CustomQMV.activeInputGroups(5, variant: .staged) == 1)
        #expect(Qwen35CustomQMV.activeInputGroups(5, variant: .singlePass) == 1)

        // The scored cell shapes must resolve to the entries the plan names.
        #expect(Qwen35QMVCell.identify(k: 5120, n: 34816) == .mlpGateUp)
        #expect(Qwen35QMVCell.identify(k: 17408, n: 5120) == .mlpDown)
        #expect(Qwen35QMVCell.identify(k: 5120, n: 16480) == .gdnInProj)
        #expect(Qwen35QMVCell.identify(k: 6144, n: 5120) == .outProj)
        #expect(Qwen35QMVCell.identify(k: 5120, n: 14336) == .faQKV)
        #expect(Qwen35QMVCell.identify(k: 5120, n: 248_320) == .lmHead)
        #expect(Qwen35QMVCell.identify(k: 5120, n: 4096) == .unlisted)

        // m <= 5 compiles the identical case in both variants, so the m = 5
        // null control really is a null.
        for m in 2 ... 5 {
            #expect(
                Qwen35QMVKernelVariant.staged.pairs.first { $0.m == m }?.ipg
                    == Qwen35QMVKernelVariant.singlePass.pairs.first { $0.m == m }?
                        .ipg)
        }
    }

    // MARK: - numerical gate

    @Test(.enabled(if: E195CellSelectiveQMVTests.gateEnabled))
    func numericalGate() throws {
        let staged = E195Pipeline(variant: .staged, label: "staged")
        let single = E195Pipeline(variant: .singlePass, label: "singlepass")

        // Positive control: the same single-pass source with one accumulation
        // perturbed. The comparison must report differences for this build,
        // otherwise a pass proves nothing.
        let perturbedHeader = qwen35E120QMVHeader.replacingOccurrences(
            of: "acc[r] += scale_local[r] * partial[r] + sums * bias_local[r];",
            with:
                "acc[r] += scale_local[r] * partial[r] * 1.0000305f "
                + "+ sums * bias_local[r];")
        #expect(perturbedHeader != qwen35E120QMVHeader)
        let perturbed = E195Pipeline(
            variant: .singlePass, label: "perturbed", header: perturbedHeader,
            nameSuffix: "_ctl")

        var rows: [[String: Any]] = []
        var control: [[String: Any]] = []
        var worstUlp = 0
        var totalDiffering = 0
        var totalElements = 0

        for cell in e195Cells {
            try autoreleasepool {
                let (w, scales, biases) = e195PackedCell(
                    k: cell.k, n: cell.n, seed: 0xE195)
                for m in Self.gateWidths {
                    let x = e195Activations(
                        m: m, k: cell.k, seed: UInt64(0xE195_0000 + m))
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
                        let diff = e195Compare(a, b)
                        worstUlp = max(worstUlp, diff.maxUlp)
                        totalDiffering += diff.count
                        totalElements += m * cell.n
                        rows.append([
                            "cell": cell.name, "k": cell.k, "n": cell.n, "m": m,
                            "use_table": useTable,
                            "elements": m * cell.n,
                            "differing": diff.count,
                            "max_ulp": diff.maxUlp,
                            "max_abs_diff": diff.maxAbsDiff,
                            "non_finite": diff.nanOrInf,
                            "staged_groups": Qwen35CustomQMV.activeInputGroups(
                                m, variant: .staged),
                            "single_pass_groups": Qwen35CustomQMV
                                .activeInputGroups(m, variant: .singlePass),
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
                            let diff = e195Compare(bad, good)
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

        try e195Write(
            [
                "harness": "local",
                "experiment": "e195",
                "section": "numerical_gate",
                "cool_gate_passed_real_gate": false,
                "gate_qualified_for_timing": false,
                "official_or_ranked_score": false,
                "active_plan": qwen35QMVWidthPlanWitness,
                "qmv_arm": Qwen35CustomQMV.arm.rawValue,
                "worst_max_ulp": worstUlp,
                "total_differing": totalDiffering,
                "total_elements": totalElements,
                "cells": rows,
                "positive_control": control,
            ], to: "MLXFAST_E195_GATE_OUT")
    }

    // MARK: - lm_head m=7 replication

    /// FINDING 501 flagged the staged lm_head column as non-monotone: 8541 us
    /// at m = 7 against 6175 us at m = 8, from a single invocation per round in
    /// a 69.6 C session. lm_head runs once per round, so the E189 probe had the
    /// fewest samples exactly where the anomaly sits. This section times only
    /// lm_head, at m = 6, 7 and 8, with more ABBA blocks.
    @Test(.enabled(if: E195CellSelectiveQMVTests.probeEnabled))
    func lmHeadProbe() throws {
        let env = ProcessInfo.processInfo.environment
        let blocks = Int(env["MLXFAST_E195_BLOCKS"] ?? "") ?? 12
        let warmup = Int(env["MLXFAST_E195_WARMUP"] ?? "") ?? 6
        let reps = Int(env["MLXFAST_E195_REPS"] ?? "") ?? 4
        let chain = Int(env["MLXFAST_E195_CHAIN"] ?? "") ?? 8
        let variants = Int(env["MLXFAST_E195_VARIANTS"] ?? "") ?? 4
        let widths =
            (env["MLXFAST_E195_PROBE_WIDTHS"]?
                .split(separator: ",").compactMap { Int($0) }).flatMap {
                $0.isEmpty ? nil : $0
            } ?? [5, 6, 7, 8]
        let cells =
            (env["MLXFAST_E195_PROBE_CELLS"]?
                .split(separator: ",").map(String.init)).flatMap {
                $0.isEmpty ? nil : Set($0)
            } ?? ["lm_head"]

        let staged = E195Pipeline(variant: .staged, label: "staged")
        let single = E195Pipeline(variant: .singlePass, label: "singlepass")

        var samples: [[String: Any]] = []
        var temperatures: [[String: Any]] = []
        func recordTemperature(_ label: String) {
            temperatures.append([
                "label": label,
                "gpu_temp_c": e195GpuTemperature() ?? -1,
                "seconds": Date().timeIntervalSince1970,
            ])
        }

        recordTemperature("session_entry")
        for cell in e195AllCells where cells.contains(cell.name) {
            try autoreleasepool {
                let (w, scales, biases) = e195PackedCell(
                    k: cell.k, n: cell.n, seed: 0xE195)
                for m in widths {
                    // Distinct activation tensors, cycled inside a chain, so no
                    // graph-level reuse can serve a repeat from an earlier
                    // result instead of relaunching the kernel.
                    let inputs: [(MLXArray, MLXArray)] = (0 ..< variants).map {
                        variant in
                        let x = e195Activations(
                            m: m, k: cell.k,
                            seed: UInt64(0xE195_0000 + m * 16 + variant))
                        let xsums = Qwen35CustomQMV.xsumsTable(x)
                        eval(xsums)
                        return (x, xsums)
                    }
                    let arms: [(String, E195Pipeline)] = [
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
                            let us = e195Timed(reps: reps, chain: chain) {
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
                                    m, variant: entry.1.variant),
                            ])
                        }
                    }
                    recordTemperature("\(cell.name)_m\(m)")
                }
            }
        }
        recordTemperature("session_exit")

        try e195Write(
            [
                "harness": "local",
                "experiment": "e195",
                "section": "lm_head_probe",
                "cool_gate_passed_real_gate": false,
                "gate_qualified_for_timing": false,
                "official_or_ranked_score": false,
                "active_plan": qwen35QMVWidthPlanWitness,
                "blocks": blocks, "warmup": warmup, "reps": reps,
                "chain": chain, "variants": variants,
                "samples": samples,
                "temperatures": temperatures,
            ], to: "MLXFAST_E195_PROBE_OUT")
    }
}
