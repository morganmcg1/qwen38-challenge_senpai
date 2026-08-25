import Foundation
import MLX
import MLXFast
import MLXRandom
import Testing

@testable import MLXLLM

// E213 -- does raising rows per group at m = 6, 7 and 8 recover time, and is
// the IPG cliff a register boundary?
//
// Stage-0 desk result, before any timing:
//
//   * `(6,5)` is not a partition this template builds. `qwen_e120_qmv_m`
//     asserts `M % IPG != 1`, and `6 % 5 == 1` makes a one-input tail group.
//     The widest legal first group at m = 6 is 4, so the arm takes `(6,4)`.
//   * `G(m) = ceil(m / IPG)` does not move at any of the three widths:
//     (6,3) and (6,4) both pay 2 passes, (7,4) and (7,5) both pay 2, (8,4) and
//     (8,5) both pay 2. E208 won at m = 9 because `(9,3) -> (9,5)` cut G from
//     3 to 2. That reduction is not available below m = 9 without IPG = m,
//     which is the single-pass form E195 rejected at m = 7 and m = 8.
//   * The AIR register-pressure proxy puts `qwen_e120_qmv_wide<NA>` at 125
//     lane-weighted live values for NA = 5 and 144 for NA = 6, so the Apple
//     128-register boundary falls between them. That is the IPG cliff.
//
// The `retuned` arm therefore changes the row split alone, and `na6` is the
// control that can fail: `(9,5) -> (9,6)` holds G = 2 and crosses the register
// boundary, so it isolates the register term at fixed weight traffic.
//
// Two sections:
//
//   * `planTable` is a pure-function test. It proves the legality rule, that
//     each arm moves exactly the widths it claims and no others, that no arm
//     changes any group count, and that the witness a leg emits names its own
//     table;
//   * `numericalGate` compares each arm's compiled partition against the
//     shipped one, cell by cell, at the widths that arm touches, with real
//     packed 4-bit weights and real bfloat16 activations. It compares actual
//     floating-point values, records hexfloat samples at the group cuts, and
//     runs a positive control that proves the comparison can fail.
//
// The expectation is bit-exact. `qwen_e120_qmv_wide` keeps row `m` in lane `m`
// of every `vec<float, NA>` it touches and no operation mixes lanes, so
// regrouping rows cannot reorder any row's own reduction.
//
// Research instrument. `Tests/` is never packaged into a submission.
// harness=local, no thermal gate, no score.

private struct E213Cell {
    var name: String
    var k: Int
    var n: Int
    /// Wide-QMV invocations of this cell in one decode round (FINDING 487).
    var invocations: Int
}

private let e213AllCells = [
    E213Cell(name: "mlp.gate_up", k: 5120, n: 34816, invocations: 64),
    E213Cell(name: "mlp.down", k: 17408, n: 5120, invocations: 64),
    E213Cell(name: "gdn.in_proj", k: 5120, n: 16480, invocations: 48),
    E213Cell(name: "gdn.out_proj", k: 6144, n: 5120, invocations: 48),
    E213Cell(name: "fa.qkv", k: 5120, n: 14336, invocations: 16),
    E213Cell(name: "fa.o_proj", k: 6144, n: 5120, invocations: 16),
    E213Cell(name: "lm_head", k: 5120, n: 248_320, invocations: 1),
]

/// `MLXFAST_E213_CELLS` selects a subset for a smoke run. The full seven-cell
/// list is the default and the only list a reported result may use.
private let e213Cells: [E213Cell] = {
    guard
        let raw = ProcessInfo.processInfo.environment["MLXFAST_E213_CELLS"],
        !raw.isEmpty
    else { return e213AllCells }
    let wanted = Set(raw.split(separator: ",").map(String.init))
    return e213AllCells.filter { wanted.contains($0.name) }
}()

/// A JIT kernel pair for one compiled plan, built from the production source so
/// the instrument cannot drift from the shipped text. The scored decode path
/// launches through `Qwen35CachedKernel`, so this does too.
private struct E213Pipeline {
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
            name: "e213_qmv_table_\(label)\(nameSuffix)",
            inputNames: ["w", "scales", "biases", "x", "xsums"],
            outputNames: ["y"],
            source: qwen35E120QMVSource(table: true, variant: variant),
            header: head)
        self.plain = Qwen35CachedKernel(
            name: "e213_qmv_plain_\(label)\(nameSuffix)",
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
private func e213PackedCell(k: Int, n: Int, seed: UInt64)
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

private func e213Activations(m: Int, k: Int, seed: UInt64) -> MLXArray {
    MLXRandom.seed(seed)
    let x = MLXRandom.normal([1, m, k]).asType(.bfloat16)
    eval(x)
    return x
}

/// bfloat16 total-order key. bfloat16 is the top half of the float32 bit
/// pattern, so the key orders the finite values monotonically and the
/// difference of two keys is the ULP distance.
private func e213Key(_ value: Float) -> Int {
    let bits = Int(value.bitPattern >> 16)
    let magnitude = bits & 0x7fff
    return (bits & 0x8000) != 0 ? -magnitude : magnitude
}

private struct E213Diff {
    var count: Int
    var maxUlp: Int
    var maxAbsDiff: Double
    var nanOrInf: Int
    var firstDifferingIndex: Int?
}

private func e213Compare(_ a: [Float], _ b: [Float]) -> E213Diff {
    var diff = E213Diff(
        count: 0, maxUlp: 0, maxAbsDiff: 0, nanOrInf: 0,
        firstDifferingIndex: nil)
    for i in 0 ..< min(a.count, b.count) {
        let x = a[i]
        let y = b[i]
        if !x.isFinite || !y.isFinite { diff.nanOrInf += 1 }
        if x.bitPattern == y.bitPattern { continue }
        if diff.firstDifferingIndex == nil { diff.firstDifferingIndex = i }
        diff.count += 1
        diff.maxUlp = max(diff.maxUlp, abs(e213Key(x) - e213Key(y)))
        diff.maxAbsDiff = max(diff.maxAbsDiff, Double(abs(x - y)))
    }
    return diff
}

/// The C99 hexadecimal float literal, plus the raw bits. The literal is the
/// exact value; the bits make an off-by-one-ULP report readable.
private func e213Hex(_ value: Float) -> String {
    String(format: "%a|0x%08x", value, value.bitPattern)
}

/// The input rows where two partitions of the same width cut differently, plus
/// the first and last row. A sample taken away from a cut cannot witness a
/// regrouping defect, so the cuts of both plans are always included.
private func e213SampleRows(m: Int, a: Int, b: Int) -> [Int] {
    var rows: Set<Int> = [0, m - 1]
    for ipg in [a, b] {
        var first = ipg
        while first < m {
            rows.insert(first - 1)
            rows.insert(first)
            first += ipg
        }
    }
    return rows.sorted()
}

private func e213RowSamples(
    _ values: [Float], n: Int, columns: Int, rows: [Int]
) -> [[String: Any]] {
    var samples: [[String: Any]] = []
    for row in rows {
        var hex: [String] = []
        for column in 0 ..< columns {
            let index = row * n + column
            guard index < values.count else { break }
            hex.append(e213Hex(values[index]))
        }
        samples.append(["row": row, "hexfloat": hex])
    }
    return samples
}

private func e213Write(_ payload: [String: Any], to key: String) throws {
    let path = try #require(
        ProcessInfo.processInfo.environment[key],
        "\(key) must name the JSON destination")
    let data = try JSONSerialization.data(
        withJSONObject: payload,
        options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
    try data.write(to: URL(fileURLWithPath: path))
}

@Suite("E213 staged QMV rows-per-group retune")
struct E213RetuneQMVTests {
    static let gateEnabled =
        ProcessInfo.processInfo.environment["MLXFAST_RUN_E213_GATE"] == "1"

    /// Widths each arm moves, and the shipped IPG it moves away from.
    static let armWidths: [(variant: Qwen35QMVKernelVariant, moved: [Int])] = [
        (.stagedRetuned, [6, 7, 8]),
        (.stagedNA6, [9]),
    ]

    // MARK: - plan table

    /// Each arm moves exactly the widths it claims, and no arm changes a group
    /// count. This is the "the retune cannot remove a weight pass" proof and it
    /// needs no GPU.
    @Test
    func planTable() {
        let shipped = Qwen35QMVKernelVariant.staged.pairs
        for (variant, moved) in Self.armWidths {
            let arm = variant.pairs
            #expect(arm.count == shipped.count)
            for (armPair, shippedPair) in zip(arm, shipped) {
                #expect(armPair.m == shippedPair.m)
                if moved.contains(armPair.m) {
                    #expect(
                        armPair.ipg != shippedPair.ipg,
                        "width \(armPair.m) must move in \(variant.rawValue)")
                } else {
                    #expect(
                        armPair.ipg == shippedPair.ipg,
                        "width \(armPair.m) must not move in \(variant.rawValue)")
                }
                // The whole point: the row split moves, the pass count does not.
                #expect(
                    Qwen35CustomQMV.activeInputGroups(armPair.m, variant: variant)
                        == Qwen35CustomQMV.activeInputGroups(
                            shippedPair.m, variant: .staged),
                    "width \(armPair.m) group count must not move")
            }
        }

        // `M % IPG == 1` is the one partition the template refuses to build, so
        // (6,5) does not exist and m = 6 takes the widest legal group, 4.
        #expect(6 % 5 == 1)
        #expect(6 % 4 == 2)
        #expect(7 % 5 == 2)
        #expect(8 % 5 == 3)
        #expect(9 % 6 == 3)
        for (_, ipg) in Qwen35QMVKernelVariant.stagedRetuned.pairs {
            #expect(ipg <= 5, "the retune must stay inside the IPG <= 5 regime")
        }

        // Each arm's source is the shipped source with only its own
        // instantiations substituted.
        for table in [true, false] {
            let base = qwen35E120QMVSource(table: table, variant: .staged)
            for (variant, moved) in Self.armWidths {
                var expected = base
                for m in moved {
                    let from = Qwen35CustomQMV.inputsPerGroup(m, variant: .staged)
                    let to = Qwen35CustomQMV.inputsPerGroup(m, variant: variant)
                    expected = expected.replacingOccurrences(
                        of: "qwen_e120_qmv_m<\(m), \(from),",
                        with: "qwen_e120_qmv_m<\(m), \(to),")
                }
                #expect(expected != base)
                #expect(
                    expected == qwen35E120QMVSource(table: table, variant: variant),
                    "\(variant.rawValue) source must differ only at \(moved)")
            }
        }

        // The witness each leg emits, derived from its own table.
        #expect(
            qwen35QMVWidthPlanWitnessText(for: .staged) == "selective-m6+ipg9-5")
        #expect(
            qwen35QMVWidthPlanWitnessText(for: .stagedRetuned)
                == "selective-m6+ipg9-5+e213-4-5-5-5")
        #expect(
            qwen35QMVWidthPlanWitnessText(for: .stagedNA6)
                == "selective-m6+ipg9-6+e213-3-4-4-6")
        // No arm selected means the shipped plan, so a stray environment cannot
        // publish a research table as the shipped one.
        #expect(
            qwen35QMVWidthPlanWitness
                == qwen35QMVWidthPlanWitnessText(for: qwen35E213StagedVariant))

        // The arms may not disturb the m = 6 single-pass selection E195 shipped.
        // At m = 6 the staged table governs `mlp.down` alone.
        #expect(
            Qwen35CustomQMV.kernelVariant((m: 6, k: 5120, n: 248_320))
                == .singlePass)
        #expect(
            Qwen35CustomQMV.kernelVariant((m: 6, k: 17408, n: 5120))
                == qwen35E213StagedVariant)
    }

    // MARK: - numerical gate

    @Test(.enabled(if: E213RetuneQMVTests.gateEnabled))
    func numericalGate() throws {
        let shippedPlan = E213Pipeline(variant: .staged, label: "shipped")

        // Positive control: the shipped source with one accumulation perturbed
        // by a single bfloat16 ULP of relative scale. The comparison must
        // report differences for this build, otherwise a pass proves nothing.
        let perturbedHeader = qwen35E120QMVHeader.replacingOccurrences(
            of: "acc[r] += scale_local[r] * partial[r] + sums * bias_local[r];",
            with:
                "acc[r] += scale_local[r] * partial[r] * 1.0000305f "
                + "+ sums * bias_local[r];")
        #expect(perturbedHeader != qwen35E120QMVHeader)
        let perturbed = E213Pipeline(
            variant: .staged, label: "perturbed", header: perturbedHeader,
            nameSuffix: "_ctl")

        var rows: [[String: Any]] = []
        var control: [[String: Any]] = []
        var worstUlp = 0
        var totalDiffering = 0
        var totalElements = 0

        for (variant, widths) in Self.armWidths {
            let armPlan = E213Pipeline(variant: variant, label: variant.rawValue)
            for m in widths {
                let shippedIPG = Qwen35CustomQMV.inputsPerGroup(m, variant: .staged)
                let armIPG = Qwen35CustomQMV.inputsPerGroup(m, variant: variant)
                let sampleRows = e213SampleRows(m: m, a: shippedIPG, b: armIPG)
                for cell in e213Cells {
                    autoreleasepool {
                        let (w, scales, biases) = e213PackedCell(
                            k: cell.k, n: cell.n, seed: 0xE213)
                        let x = e213Activations(
                            m: m, k: cell.k, seed: UInt64(0xE213_0000 + m))
                        let xsums = Qwen35CustomQMV.xsumsTable(x)
                        eval(xsums)
                        for useTable in [true, false] {
                            let a = armPlan.call(
                                x: x, w: w, scales: scales, biases: biases,
                                xsums: xsums, m: m, n: cell.n, useTable: useTable)
                            let b = shippedPlan.call(
                                x: x, w: w, scales: scales, biases: biases,
                                xsums: xsums, m: m, n: cell.n, useTable: useTable)
                            eval(a, b)
                            let av = a.asType(.float32).asArray(Float.self)
                            let bv = b.asType(.float32).asArray(Float.self)
                            let diff = e213Compare(av, bv)
                            worstUlp = max(worstUlp, diff.maxUlp)
                            totalDiffering += diff.count
                            totalElements += m * cell.n
                            rows.append([
                                "arm": variant.rawValue,
                                "cell": cell.name, "k": cell.k, "n": cell.n,
                                "m": m, "use_table": useTable,
                                "shipped_ipg": shippedIPG, "arm_ipg": armIPG,
                                "shipped_groups": Qwen35CustomQMV
                                    .activeInputGroups(m, variant: .staged),
                                "arm_groups": Qwen35CustomQMV
                                    .activeInputGroups(m, variant: variant),
                                "invocations_per_round": cell.invocations,
                                "elements": m * cell.n,
                                "differing": diff.count,
                                "max_ulp": diff.maxUlp,
                                "max_abs_diff": diff.maxAbsDiff,
                                "non_finite": diff.nanOrInf,
                                "first_differing_index":
                                    diff.firstDifferingIndex ?? -1,
                                "sample_rows": sampleRows,
                                "hexfloat_shipped": e213RowSamples(
                                    bv, n: cell.n, columns: 4, rows: sampleRows),
                                "hexfloat_arm": e213RowSamples(
                                    av, n: cell.n, columns: 4, rows: sampleRows),
                            ])
                            #expect(
                                diff.count == 0,
                                """
                                \(variant.rawValue) \(cell.name) m=\(m) \
                                table=\(useTable): \(diff.count) differing \
                                outputs, max_ulp=\(diff.maxUlp), \
                                max_abs_diff=\(diff.maxAbsDiff)
                                """)
                            #expect(diff.nanOrInf == 0)
                        }

                        if cell.name == "gdn.out_proj" {
                            for useTable in [true, false] {
                                let bad = perturbed.call(
                                    x: x, w: w, scales: scales, biases: biases,
                                    xsums: xsums, m: m, n: cell.n,
                                    useTable: useTable)
                                let good = shippedPlan.call(
                                    x: x, w: w, scales: scales, biases: biases,
                                    xsums: xsums, m: m, n: cell.n,
                                    useTable: useTable)
                                eval(bad, good)
                                let diff = e213Compare(
                                    bad.asType(.float32).asArray(Float.self),
                                    good.asType(.float32).asArray(Float.self))
                                control.append([
                                    "arm": variant.rawValue,
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
        }

        try e213Write(
            [
                "harness": "local",
                "experiment": "e213",
                "section": "numerical_gate",
                "cool_gate_passed_real_gate": false,
                "gate_qualified_for_timing": false,
                "official_or_ranked_score": false,
                "tolerance": "bit-exact: zero differing outputs, zero ULP",
                "active_plan": qwen35QMVWidthPlanWitness,
                "qmv_arm": Qwen35CustomQMV.arm.rawValue,
                "worst_max_ulp": worstUlp,
                "total_differing": totalDiffering,
                "total_elements": totalElements,
                "cells": rows,
                "positive_control": control,
            ], to: "MLXFAST_E213_GATE_OUT")
    }
}
