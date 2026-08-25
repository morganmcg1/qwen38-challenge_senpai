import Foundation
import MLX
import MLXFast
import MLXRandom
import Testing

@testable import MLXLLM

// E208 -- does the staged QMV plan entry (9, 3) -> (9, 5) remove one of the
// three full weight passes a width-9 round pays, without changing a single
// output bit?
//
// FINDING 535: at m = 9 all seven fused decode cells take `.staged` with
// IPG = 3, so one threadgroup streams each weight matrix three times.
// FINDING 536 priced that third pass at 40.051 ms per m = 9 round locally.
// IPG = 5 splits the nine rows into a group of 5 and a tail group of 4, which
// the template's existing `TAIL` branch already builds, and cuts G from 3 to 2.
//
// This file is the Stage-0 falsification gate that must pass before any timing.
// Two sections:
//
//   * `planTable` is a pure-function test of the two tables. It proves that
//     `stagedWide9` differs from `staged` at m = 9 and nowhere else, that the
//     launched group count follows the table, and that the arm resolves the
//     witness the trace will carry;
//   * `numericalGate` compares the two compiled group partitions cell by cell
//     at m = 9 with real packed 4-bit weights and real bfloat16 activations. It
//     compares actual floating-point values, not argmax, records hexfloat
//     samples on both sides of the (9, 5) group boundary, and runs a positive
//     control that proves the comparison can fail.
//
// The expectation is bit-exact. `qwen_e120_qmv_wide` keeps row `m` in lane `m`
// of every `vec<float, NA>` it touches, and no operation mixes lanes: the
// activation load, the nibble products, the chunk sum, the scale/bias fold and
// the closing `simd_sum` are all per-lane. Regrouping rows therefore cannot
// reorder any row's own reduction. Anything other than zero differing outputs
// falsifies that reading and stops the experiment before Stage 1.
//
// Research instrument. `Tests/` is never packaged into a submission.
// harness=local, no thermal gate, no score.

private struct E208Cell {
    var name: String
    var k: Int
    var n: Int
    /// Wide-QMV invocations of this cell in one decode round (FINDING 487).
    var invocations: Int
}

private let e208AllCells = [
    E208Cell(name: "mlp.gate_up", k: 5120, n: 34816, invocations: 64),
    E208Cell(name: "mlp.down", k: 17408, n: 5120, invocations: 64),
    E208Cell(name: "gdn.in_proj", k: 5120, n: 16480, invocations: 48),
    E208Cell(name: "gdn.out_proj", k: 6144, n: 5120, invocations: 48),
    E208Cell(name: "fa.qkv", k: 5120, n: 14336, invocations: 16),
    E208Cell(name: "fa.o_proj", k: 6144, n: 5120, invocations: 16),
    E208Cell(name: "lm_head", k: 5120, n: 248_320, invocations: 1),
]

/// `MLXFAST_E208_CELLS` selects a subset for a smoke run. The full seven-cell
/// list is the default and the only list a reported result may use.
private let e208Cells: [E208Cell] = {
    guard
        let raw = ProcessInfo.processInfo.environment["MLXFAST_E208_CELLS"],
        !raw.isEmpty
    else { return e208AllCells }
    let wanted = Set(raw.split(separator: ",").map(String.init))
    return e208AllCells.filter { wanted.contains($0.name) }
}()

/// A JIT kernel pair for one compiled group partition, built from the
/// production source so the instrument cannot drift from the shipped text. The
/// scored decode path launches through `Qwen35CachedKernel`, so this does too.
private struct E208Pipeline {
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
            name: "e208_qmv_table_\(label)\(nameSuffix)",
            inputNames: ["w", "scales", "biases", "x", "xsums"],
            outputNames: ["y"],
            source: qwen35E120QMVSource(table: true, variant: variant),
            header: head)
        self.plain = Qwen35CachedKernel(
            name: "e208_qmv_plain_\(label)\(nameSuffix)",
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
private func e208PackedCell(k: Int, n: Int, seed: UInt64)
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

private func e208Activations(m: Int, k: Int, seed: UInt64) -> MLXArray {
    MLXRandom.seed(seed)
    let x = MLXRandom.normal([1, m, k]).asType(.bfloat16)
    eval(x)
    return x
}

/// bfloat16 total-order key. bfloat16 is the top half of the float32 bit
/// pattern, so the key orders the finite values monotonically and the
/// difference of two keys is the ULP distance.
private func e208Key(_ value: Float) -> Int {
    let bits = Int(value.bitPattern >> 16)
    let magnitude = bits & 0x7fff
    return (bits & 0x8000) != 0 ? -magnitude : magnitude
}

private struct E208Diff {
    var count: Int
    var maxUlp: Int
    var maxAbsDiff: Double
    var nanOrInf: Int
    var firstDifferingIndex: Int?
}

private func e208Compare(_ a: [Float], _ b: [Float]) -> E208Diff {
    var diff = E208Diff(
        count: 0, maxUlp: 0, maxAbsDiff: 0, nanOrInf: 0,
        firstDifferingIndex: nil)
    for i in 0 ..< min(a.count, b.count) {
        let x = a[i]
        let y = b[i]
        if !x.isFinite || !y.isFinite { diff.nanOrInf += 1 }
        if x.bitPattern == y.bitPattern { continue }
        if diff.firstDifferingIndex == nil { diff.firstDifferingIndex = i }
        diff.count += 1
        diff.maxUlp = max(diff.maxUlp, abs(e208Key(x) - e208Key(y)))
        diff.maxAbsDiff = max(diff.maxAbsDiff, Double(abs(x - y)))
    }
    return diff
}

/// The C99 hexadecimal float literal, plus the raw bits. The literal is the
/// exact value; the bits make an off-by-one-ULP report readable.
private func e208Hex(_ value: Float) -> String {
    String(format: "%a|0x%08x", value, value.bitPattern)
}

/// Hexfloat samples for one output tensor, taken at the rows that the two
/// partitions group differently. `(9, 3)` cuts after rows 2 and 5; `(9, 5)`
/// cuts after row 4. Rows 2, 4, 5 and 8 therefore sit on or beside every cut,
/// and row 0 is the common first row.
private func e208RowSamples(_ values: [Float], n: Int, columns: Int)
    -> [[String: Any]]
{
    var samples: [[String: Any]] = []
    for row in [0, 2, 4, 5, 8] {
        var hex: [String] = []
        for column in 0 ..< columns {
            let index = row * n + column
            guard index < values.count else { break }
            hex.append(e208Hex(values[index]))
        }
        samples.append(["row": row, "hexfloat": hex])
    }
    return samples
}

private func e208Write(_ payload: [String: Any], to key: String) throws {
    let path = try #require(
        ProcessInfo.processInfo.environment[key],
        "\(key) must name the JSON destination")
    let data = try JSONSerialization.data(
        withJSONObject: payload,
        options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
    try data.write(to: URL(fileURLWithPath: path))
}

@Suite("E208 width-9 staged QMV group partition")
struct E208Wide9QMVTests {
    static let gateEnabled =
        ProcessInfo.processInfo.environment["MLXFAST_RUN_E208_GATE"] == "1"

    // MARK: - plan table

    /// The wide-9 table must differ from the shipped table at m = 9 and at no
    /// other width, and the launched group count must follow the table. This is
    /// the "changes nothing else" proof and it needs no GPU.
    @Test
    func planTable() {
        let staged = Qwen35QMVKernelVariant.staged.pairs
        let wide9 = Qwen35QMVKernelVariant.stagedWide9.pairs
        #expect(staged.count == wide9.count)
        for (a, b) in zip(staged, wide9) {
            #expect(a.m == b.m)
            if a.m == 9 {
                #expect(a.ipg == 3)
                #expect(b.ipg == 5)
            } else {
                #expect(a.ipg == b.ipg, "width \(a.m) must not move")
            }
        }

        // `M % IPG == 1` is the one partition the template refuses to build.
        #expect(9 % 5 == 4)

        // The third weight pass is what this removes.
        #expect(Qwen35CustomQMV.activeInputGroups(9, variant: .staged) == 3)
        #expect(Qwen35CustomQMV.activeInputGroups(9, variant: .stagedWide9) == 2)
        for m in 2 ... 8 {
            #expect(
                Qwen35CustomQMV.activeInputGroups(m, variant: .staged)
                    == Qwen35CustomQMV.activeInputGroups(
                        m, variant: .stagedWide9),
                "width \(m) group count must not move")
        }

        // The compiled sources may differ only in the m = 9 case.
        for table in [true, false] {
            let a = qwen35E120QMVSource(table: table, variant: .staged)
            let b = qwen35E120QMVSource(table: table, variant: .stagedWide9)
            #expect(a != b)
            #expect(
                a.replacingOccurrences(
                    of: "qwen_e120_qmv_m<9, 3,", with: "qwen_e120_qmv_m<9, 5,")
                    == b)
        }

        // Distinct JIT names, so one arm cannot serve the other's compiled
        // kernel inside one process.
        #expect(Qwen35QMVKernelVariant.staged.kernelNameSuffix == "")
        #expect(Qwen35QMVKernelVariant.stagedWide9.kernelNameSuffix == "_w9")

        // The arm and the witness the trace will carry.
        let armOn =
            ProcessInfo.processInfo.environment["DARKBLOOM_E208_QMV_ARM"] == "on"
        #expect(qwen35E208StagedVariant == (armOn ? .stagedWide9 : .staged))
        #expect(
            qwen35QMVWidthPlanWitness
                == (armOn ? "selective-m6+ipg9-5" : "selective-m6+ipg9-3"))

        // The arm may not disturb the m = 6 single-pass selection E195 shipped.
        #expect(
            Qwen35CustomQMV.kernelVariant((m: 6, k: 5120, n: 248_320))
                == .singlePass)
        // Every fused cell at m = 9 takes the arm's staged family.
        for cell in e208AllCells {
            #expect(
                Qwen35CustomQMV.kernelVariant((m: 9, k: cell.k, n: cell.n))
                    == qwen35E208StagedVariant, "\(cell.name)")
        }
    }

    // MARK: - numerical gate

    @Test(.enabled(if: E208Wide9QMVTests.gateEnabled))
    func numericalGate() throws {
        let m = 9
        let staged = E208Pipeline(variant: .staged, label: "staged")
        let wide9 = E208Pipeline(variant: .stagedWide9, label: "wide9")

        // Positive control: the wide-9 source with one accumulation perturbed
        // by a single bfloat16 ULP of relative scale. The comparison must
        // report differences for this build, otherwise a pass proves nothing.
        let perturbedHeader = qwen35E120QMVHeader.replacingOccurrences(
            of: "acc[r] += scale_local[r] * partial[r] + sums * bias_local[r];",
            with:
                "acc[r] += scale_local[r] * partial[r] * 1.0000305f "
                + "+ sums * bias_local[r];")
        #expect(perturbedHeader != qwen35E120QMVHeader)
        let perturbed = E208Pipeline(
            variant: .stagedWide9, label: "perturbed", header: perturbedHeader,
            nameSuffix: "_ctl")

        var rows: [[String: Any]] = []
        var control: [[String: Any]] = []
        var worstUlp = 0
        var totalDiffering = 0
        var totalElements = 0

        for cell in e208Cells {
            autoreleasepool {
                let (w, scales, biases) = e208PackedCell(
                    k: cell.k, n: cell.n, seed: 0xE208)
                let x = e208Activations(
                    m: m, k: cell.k, seed: UInt64(0xE208_0000 + m))
                let xsums = Qwen35CustomQMV.xsumsTable(x)
                eval(xsums)
                for useTable in [true, false] {
                    let a = wide9.call(
                        x: x, w: w, scales: scales, biases: biases,
                        xsums: xsums, m: m, n: cell.n, useTable: useTable)
                    let b = staged.call(
                        x: x, w: w, scales: scales, biases: biases,
                        xsums: xsums, m: m, n: cell.n, useTable: useTable)
                    eval(a, b)
                    let av = a.asType(.float32).asArray(Float.self)
                    let bv = b.asType(.float32).asArray(Float.self)
                    let diff = e208Compare(av, bv)
                    worstUlp = max(worstUlp, diff.maxUlp)
                    totalDiffering += diff.count
                    totalElements += m * cell.n
                    rows.append([
                        "cell": cell.name, "k": cell.k, "n": cell.n, "m": m,
                        "use_table": useTable,
                        "invocations_per_round": cell.invocations,
                        "elements": m * cell.n,
                        "differing": diff.count,
                        "max_ulp": diff.maxUlp,
                        "max_abs_diff": diff.maxAbsDiff,
                        "non_finite": diff.nanOrInf,
                        "first_differing_index":
                            diff.firstDifferingIndex ?? -1,
                        "staged_groups": Qwen35CustomQMV.activeInputGroups(
                            m, variant: .staged),
                        "wide9_groups": Qwen35CustomQMV.activeInputGroups(
                            m, variant: .stagedWide9),
                        "hexfloat_staged": e208RowSamples(
                            bv, n: cell.n, columns: 4),
                        "hexfloat_wide9": e208RowSamples(
                            av, n: cell.n, columns: 4),
                    ])
                    #expect(
                        diff.count == 0,
                        """
                        \(cell.name) m=\(m) table=\(useTable): \
                        \(diff.count) differing outputs, \
                        max_ulp=\(diff.maxUlp), \
                        max_abs_diff=\(diff.maxAbsDiff)
                        """)
                    #expect(diff.nanOrInf == 0)
                }

                if cell.name == "gdn.out_proj" {
                    for useTable in [true, false] {
                        let bad = perturbed.call(
                            x: x, w: w, scales: scales, biases: biases,
                            xsums: xsums, m: m, n: cell.n, useTable: useTable)
                        let good = staged.call(
                            x: x, w: w, scales: scales, biases: biases,
                            xsums: xsums, m: m, n: cell.n, useTable: useTable)
                        eval(bad, good)
                        let diff = e208Compare(
                            bad.asType(.float32).asArray(Float.self),
                            good.asType(.float32).asArray(Float.self))
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

        try e208Write(
            [
                "harness": "local",
                "experiment": "e208",
                "section": "numerical_gate",
                "cool_gate_passed_real_gate": false,
                "gate_qualified_for_timing": false,
                "official_or_ranked_score": false,
                "width": m,
                "tolerance": "bit-exact: zero differing outputs, zero ULP",
                "active_plan": qwen35QMVWidthPlanWitness,
                "qmv_arm": Qwen35CustomQMV.arm.rawValue,
                "worst_max_ulp": worstUlp,
                "total_differing": totalDiffering,
                "total_elements": totalElements,
                "cells": rows,
                "positive_control": control,
            ], to: "MLXFAST_E208_GATE_OUT")
    }
}
