import Foundation
import MLX
import MLXFast
import MLXRandom
import Testing

@testable import MLXLLM

// E216 -- does co-locating the two token-column groups of one output-row slice
// in ONE threadgroup change any output bit?
//
// The shipped `split` mapping gives threadgroup `(g, y)` token group `g` over
// output rows `8y ..< 8y+8`. The two groups of a row slice therefore sit in
// different threadgroups and each streams the same weight rows from device
// memory. E208 (FINDING 543) priced that second stream at +20.774 ms per m = 9
// round; FINDING 556 confirmed -21.37 ms cross-host.
//
// The `coop` mapping gives threadgroup `y` BOTH token groups over output rows
// `4y ..< 4y+4`: simdgroup 0 runs group 0, simdgroup 1 runs group 1. Same
// threadgroup count, same threadgroup shape, same per-simdgroup geometry
// (`NA <= 5`, `rows_per_simd = 4`, the E208-winning form), same set of
// `(first_m, out_row)` calls over the whole grid.
//
// This file is the Stage-0 falsification gate that must pass before any timing.
// Three sections:
//
//   * `geometry` is a pure-function test. It proves that `coop` moves exactly
//     the two-group widths, that the two mappings launch the same number of
//     threadgroups, that each covers every output row exactly once, and that
//     the compiled source differs only at the moved widths;
//   * `numericalGate` compares the two mappings as actual floating-point values
//     over every legal instantiation the plan builds -- all widths 2...9, all
//     seven fused decode cells, the TAIL forms at m = 6, 7 and 9, both the
//     table and the recompute pipelines, and one n = 4104 edge cell;
//   * two positive controls prove the comparison can fail under the paired
//     geometry: an arithmetic perturbation of one ULP of relative scale, and
//     the `coop` source launched on the `split` grid, which is exactly the
//     failure a desynchronised launcher would produce.
//
// The expectation is bit-exact. `qwen_e120_qmv_wide` keeps row `m` in lane `m`
// of every `vec<float, NA>` it touches and closes each row with its own
// `simd_sum`; no operation mixes lanes or simdgroups. Which threadgroup runs a
// given `(first_m, out_row)` call therefore cannot reorder any row's own
// reduction. Anything other than zero differing outputs falsifies that reading
// and stops the experiment before Stage 1.
//
// Research instrument. `Tests/` is never packaged into a submission.
// harness=local, no thermal gate, no score.

private struct E216Cell {
    var name: String
    var k: Int
    var n: Int
    /// Wide-QMV invocations of this cell in one decode round (FINDING 487).
    var invocations: Int
}

private let e216AllCells = [
    E216Cell(name: "mlp.gate_up", k: 5120, n: 34816, invocations: 64),
    E216Cell(name: "mlp.down", k: 17408, n: 5120, invocations: 64),
    E216Cell(name: "gdn.in_proj", k: 5120, n: 16480, invocations: 48),
    E216Cell(name: "gdn.out_proj", k: 6144, n: 5120, invocations: 48),
    E216Cell(name: "fa.qkv", k: 5120, n: 14336, invocations: 16),
    E216Cell(name: "fa.o_proj", k: 6144, n: 5120, invocations: 16),
    E216Cell(name: "lm_head", k: 5120, n: 248_320, invocations: 1),
    // Routable but outside the decode round's fused cells: `n` is a multiple of
    // 8 and not of 64, so it exercises the shortest legal y extent.
    E216Cell(name: "edge.n4104", k: 5120, n: 4104, invocations: 0),
]

/// `MLXFAST_E216_CELLS` selects a subset for a smoke run. The full list is the
/// default and the only list a reported result may use.
private let e216Cells: [E216Cell] = {
    guard
        let raw = ProcessInfo.processInfo.environment["MLXFAST_E216_CELLS"],
        !raw.isEmpty
    else { return e216AllCells }
    let wanted = Set(raw.split(separator: ",").map(String.init))
    return e216AllCells.filter { wanted.contains($0.name) }
}()

private let e216Widths: [Int] = {
    guard
        let raw = ProcessInfo.processInfo.environment["MLXFAST_E216_WIDTHS"],
        !raw.isEmpty
    else { return Array(Qwen35CustomQMV.widths) }
    return raw.split(separator: ",").compactMap { Int($0) }
}()

/// A JIT kernel pair for one compiled mapping, built from the production source
/// so the instrument cannot drift from the shipped text. The scored decode path
/// launches through `Qwen35CachedKernel`, so this does too.
private struct E216Pipeline {
    var stream: Qwen35CustomQMV.WeightStream
    var label: String
    var table: Qwen35CachedKernel
    var plain: Qwen35CachedKernel

    init(
        stream: Qwen35CustomQMV.WeightStream, label: String,
        header: String? = nil
    ) {
        self.stream = stream
        self.label = label
        let head = header ?? qwen35E120QMVHeader
        self.table = Qwen35CachedKernel(
            name: "e216_qmv_table_\(label)",
            inputNames: ["w", "scales", "biases", "x", "xsums"],
            outputNames: ["y"],
            source: qwen35E120QMVSource(
                table: true, variant: .staged, paired: stream == .coop),
            header: head)
        self.plain = Qwen35CachedKernel(
            name: "e216_qmv_plain_\(label)",
            inputNames: ["w", "scales", "biases", "x"],
            outputNames: ["y"],
            source: qwen35E120QMVSource(
                table: false, variant: .staged, paired: stream == .coop),
            header: head)
    }

    /// `geometryStream` defaults to this pipeline's own mapping. Overriding it
    /// launches the compiled source on the other mapping's grid, which is the
    /// desynchronised-launcher positive control.
    func call(
        x: MLXArray, w: MLXArray, scales: MLXArray, biases: MLXArray,
        xsums: MLXArray, m: Int, n: Int, useTable: Bool,
        geometryStream: Qwen35CustomQMV.WeightStream? = nil
    ) -> MLXArray {
        var outShape = x.shape
        outShape[outShape.count - 1] = n
        let geometry = Qwen35CustomQMV.launchGeometry(
            m: m, n: n, variant: .staged,
            stream: geometryStream ?? stream)
        let launch = Qwen35KernelLaunch(
            grid: geometry.grid,
            threadGroup: geometry.threadGroup,
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
private func e216PackedCell(k: Int, n: Int, seed: UInt64)
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

private func e216Activations(m: Int, k: Int, seed: UInt64) -> MLXArray {
    MLXRandom.seed(seed)
    let x = MLXRandom.normal([1, m, k]).asType(.bfloat16)
    eval(x)
    return x
}

/// bfloat16 total-order key. bfloat16 is the top half of the float32 bit
/// pattern, so the key orders the finite values monotonically and the
/// difference of two keys is the ULP distance.
private func e216Key(_ value: Float) -> Int {
    let bits = Int(value.bitPattern >> 16)
    let magnitude = bits & 0x7fff
    return (bits & 0x8000) != 0 ? -magnitude : magnitude
}

private struct E216Diff {
    var count: Int
    var maxUlp: Int
    var maxAbsDiff: Double
    var nanOrInf: Int
    var firstDifferingIndex: Int?
}

private func e216Compare(_ a: [Float], _ b: [Float]) -> E216Diff {
    var diff = E216Diff(
        count: 0, maxUlp: 0, maxAbsDiff: 0, nanOrInf: 0,
        firstDifferingIndex: nil)
    for i in 0 ..< min(a.count, b.count) {
        let x = a[i]
        let y = b[i]
        if !x.isFinite || !y.isFinite { diff.nanOrInf += 1 }
        if x.bitPattern == y.bitPattern { continue }
        if diff.firstDifferingIndex == nil { diff.firstDifferingIndex = i }
        diff.count += 1
        diff.maxUlp = max(diff.maxUlp, abs(e216Key(x) - e216Key(y)))
        diff.maxAbsDiff = max(diff.maxAbsDiff, Double(abs(x - y)))
    }
    return diff
}

/// The C99 hexadecimal float literal, plus the raw bits. The literal is the
/// exact value; the bits make an off-by-one-ULP report readable.
private func e216Hex(_ value: Float) -> String {
    String(format: "%a|0x%08x", value, value.bitPattern)
}

/// Hexfloat samples at the rows the paired mapping re-homes. `coop` cuts the
/// output-row axis every 4 rows where `split` cuts every 8, so rows 0, 3, 4 and
/// 7 sit on or beside every boundary the change moves.
private func e216RowSamples(_ values: [Float], m: Int, n: Int, columns: Int)
    -> [[String: Any]]
{
    var samples: [[String: Any]] = []
    for row in 0 ..< m {
        var hex: [String] = []
        for column in [0, 3, 4, 7, n - 1].prefix(columns) {
            let index = row * n + column
            guard column < n, index < values.count else { continue }
            hex.append("c\(column)=\(e216Hex(values[index]))")
        }
        samples.append(["input_row": row, "hexfloat": hex])
    }
    return samples
}

private func e216Write(_ payload: [String: Any], to key: String) throws {
    let path = try #require(
        ProcessInfo.processInfo.environment[key],
        "\(key) must name the JSON destination")
    let data = try JSONSerialization.data(
        withJSONObject: payload,
        options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
    try data.write(to: URL(fileURLWithPath: path))
}

@Suite("E216 cooperative weight stream")
struct E216CoopWeightStreamQMVTests {
    static let gateEnabled =
        ProcessInfo.processInfo.environment["MLXFAST_RUN_E216_GATE"] == "1"

    /// The staged plan entries that stream the weights exactly twice. Every
    /// other entry is a one-group width the paired mapping must leave alone.
    static let twoGroupWidths = [6, 7, 8, 9]

    // MARK: - geometry

    @Test
    func geometry() {
        // `coop` applies at exactly the two-group widths of the staged plan.
        for (m, ipg) in Qwen35QMVKernelVariant.staged.pairs {
            let groups = Qwen35CustomQMV.activeInputGroups(m, variant: .staged)
            #expect(
                Qwen35CustomQMV.cooperativeWeightStream(m: m, ipg: ipg)
                    == (groups == 2), "width \(m)")
            #expect(
                Self.twoGroupWidths.contains(m) == (groups == 2), "width \(m)")
        }
        // `singlePass` has no two-group entry, so the paired mapping cannot
        // reach the m = 6 cells E195 shipped on that plan.
        for (m, ipg) in Qwen35QMVKernelVariant.singlePass.pairs {
            #expect(!Qwen35CustomQMV.cooperativeWeightStream(m: m, ipg: ipg))
        }

        // Same threadgroup count and the same threadgroup shape in both
        // mappings: `coop` trades the x extent for twice the y extent.
        for cell in e216AllCells {
            for m in Qwen35CustomQMV.widths {
                let split = Qwen35CustomQMV.launchGeometry(
                    m: m, n: cell.n, variant: .staged, stream: .split)
                let coop = Qwen35CustomQMV.launchGeometry(
                    m: m, n: cell.n, variant: .staged, stream: .coop)
                #expect(split.threadGroup == (32, 2, 1))
                #expect(coop.threadGroup == (32, 2, 1))
                let splitGroups =
                    (split.grid.0 / 32) * (split.grid.1 / split.threadGroup.1)
                let coopGroups =
                    (coop.grid.0 / 32) * (coop.grid.1 / coop.threadGroup.1)
                #expect(
                    splitGroups == coopGroups,
                    "\(cell.name) m=\(m): \(splitGroups) vs \(coopGroups)")

                // Output-row coverage: every row is written exactly once.
                #expect(split.grid.1 / 2 * 8 == cell.n)
                if Self.twoGroupWidths.contains(m) {
                    #expect(coop.grid.0 == 32, "coop m=\(m) needs one x group")
                    #expect(coop.grid.1 / 2 * 4 == cell.n)
                } else {
                    #expect(coop.grid == split.grid, "unmoved width \(m)")
                }
            }
        }

        // The paired source differs from the shipped source at the two-group
        // widths and nowhere else.
        for table in [true, false] {
            let split = qwen35E120QMVSource(table: table, variant: .staged)
            let coop = qwen35E120QMVSource(
                table: table, variant: .staged, paired: true)
            #expect(split != coop)
            #expect(
                split.replacingOccurrences(
                    of: "int(qmv_tid.x), int(qmv_tid.y) * 8 + int(qmv_sgid) * 4",
                    with: "int(qmv_sgid), int(qmv_tid.y) * 4")
                    != coop,
                "a blanket substitution must not reproduce the paired source")
            let splitLines = split.split(separator: "\n")
            let coopLines = coop.split(separator: "\n")
            #expect(splitLines.count == coopLines.count)
            var movedCases = 0
            for (a, b) in zip(splitLines, coopLines) where a != b {
                #expect(a.contains("qmv_tid.x"))
                #expect(b.contains("qmv_sgid), int(qmv_tid.y) * 4"))
                movedCases += 1
            }
            #expect(movedCases == Self.twoGroupWidths.count)
        }

        // Default build carries the shipped mapping and the unchanged witness.
        #expect(Qwen35CustomQMV.weightStream == .split)
        #expect(qwen35QMVWidthPlanWitness == "selective-m6+ipg9-5")
        #expect(
            Qwen35CustomQMV.WeightStream.coop.witnessSuffix == "+e216-coop-g2")
    }

    // MARK: - numerical gate

    @Test(.enabled(if: E216CoopWeightStreamQMVTests.gateEnabled))
    func numericalGate() throws {
        let split = E216Pipeline(stream: .split, label: "split")
        let coop = E216Pipeline(stream: .coop, label: "coop")

        // Positive control: the paired source with one accumulation perturbed
        // by a single bfloat16 ULP of relative scale. The comparison must
        // report differences for this build, otherwise a pass proves nothing.
        let perturbedHeader = qwen35E120QMVHeader.replacingOccurrences(
            of: "acc[r] += scale_local[r] * partial[r] + sums * bias_local[r];",
            with:
                "acc[r] += scale_local[r] * partial[r] * 1.0000305f "
                + "+ sums * bias_local[r];")
        #expect(perturbedHeader != qwen35E120QMVHeader)
        let perturbed = E216Pipeline(
            stream: .coop, label: "coop_perturbed", header: perturbedHeader)

        var rows: [[String: Any]] = []
        var control: [[String: Any]] = []
        var worstUlp = 0
        var totalDiffering = 0
        var totalElements = 0

        for cell in e216Cells {
            autoreleasepool {
                let (w, scales, biases) = e216PackedCell(
                    k: cell.k, n: cell.n, seed: 0xE216)
                for m in e216Widths {
                    autoreleasepool {
                        let x = e216Activations(
                            m: m, k: cell.k, seed: UInt64(0xE216_0000 + m))
                        let xsums = Qwen35CustomQMV.xsumsTable(x)
                        eval(xsums)
                        let ipg = Qwen35CustomQMV.inputsPerGroup(
                            m, variant: .staged)
                        for useTable in [true, false] {
                            let a = split.call(
                                x: x, w: w, scales: scales, biases: biases,
                                xsums: xsums, m: m, n: cell.n,
                                useTable: useTable)
                            let b = coop.call(
                                x: x, w: w, scales: scales, biases: biases,
                                xsums: xsums, m: m, n: cell.n,
                                useTable: useTable)
                            eval(a, b)
                            let av = a.asType(.float32).asArray(Float.self)
                            let bv = b.asType(.float32).asArray(Float.self)
                            let diff = e216Compare(av, bv)
                            worstUlp = max(worstUlp, diff.maxUlp)
                            totalDiffering += diff.count
                            totalElements += m * cell.n
                            rows.append([
                                "cell": cell.name, "k": cell.k, "n": cell.n,
                                "m": m, "ipg": ipg,
                                "groups": Qwen35CustomQMV.activeInputGroups(
                                    m, variant: .staged),
                                "tail": m % ipg,
                                "moved_by_coop":
                                    Qwen35CustomQMV.cooperativeWeightStream(
                                        m: m, ipg: ipg),
                                "use_table": useTable,
                                "invocations_per_round": cell.invocations,
                                "elements": m * cell.n,
                                "differing": diff.count,
                                "max_ulp": diff.maxUlp,
                                "max_abs_diff": diff.maxAbsDiff,
                                "non_finite": diff.nanOrInf,
                                "first_differing_index":
                                    diff.firstDifferingIndex ?? -1,
                                "hexfloat_split": e216RowSamples(
                                    av, m: m, n: cell.n, columns: 5),
                                "hexfloat_coop": e216RowSamples(
                                    bv, m: m, n: cell.n, columns: 5),
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

                        // Positive controls at the widest moved width of a
                        // small cell, so both run on every cell that has one.
                        guard m == 9, cell.n <= 16480 else { return }
                        for useTable in [true, false] {
                            let good = split.call(
                                x: x, w: w, scales: scales, biases: biases,
                                xsums: xsums, m: m, n: cell.n,
                                useTable: useTable)
                            let bad = perturbed.call(
                                x: x, w: w, scales: scales, biases: biases,
                                xsums: xsums, m: m, n: cell.n,
                                useTable: useTable)
                            // The paired source on the split grid: one x group
                            // is dropped and half of every row slice is never
                            // written. This is the exact failure a launcher
                            // that desynchronised from the source would cause.
                            let misgridded = coop.call(
                                x: x, w: w, scales: scales, biases: biases,
                                xsums: xsums, m: m, n: cell.n,
                                useTable: useTable, geometryStream: .split)
                            eval(good, bad, misgridded)
                            let goodValues = good.asType(.float32)
                                .asArray(Float.self)
                            let arithmetic = e216Compare(
                                bad.asType(.float32).asArray(Float.self),
                                goodValues)
                            let mapping = e216Compare(
                                misgridded.asType(.float32)
                                    .asArray(Float.self),
                                goodValues)
                            control.append([
                                "cell": cell.name, "m": m,
                                "use_table": useTable,
                                "arithmetic_differing": arithmetic.count,
                                "arithmetic_max_ulp": arithmetic.maxUlp,
                                "arithmetic_max_abs_diff":
                                    arithmetic.maxAbsDiff,
                                "misgridded_differing": mapping.count,
                                "misgridded_max_abs_diff": mapping.maxAbsDiff,
                            ])
                            #expect(
                                arithmetic.count > 0,
                                "arithmetic control did not trip at \(cell.name)")
                            #expect(
                                mapping.count > 0,
                                "grid control did not trip at \(cell.name)")
                        }
                    }
                }
            }
        }

        try e216Write(
            [
                "harness": "local",
                "experiment": "e216-coop-weight-stream",
                "section": "numerical_gate",
                "cool_gate_passed_real_gate": false,
                "gate_qualified_for_timing": false,
                "official_or_ranked_score": false,
                "widths": e216Widths,
                "two_group_widths": Self.twoGroupWidths,
                "tolerance": "bit-exact: zero differing outputs, zero ULP",
                "active_plan": qwen35QMVWidthPlanWitness,
                "qmv_arm": Qwen35CustomQMV.arm.rawValue,
                "weight_stream": Qwen35CustomQMV.weightStream.rawValue,
                "worst_max_ulp": worstUlp,
                "total_differing": totalDiffering,
                "total_elements": totalElements,
                "cells": rows,
                "positive_control": control,
            ], to: "MLXFAST_E216_GATE_OUT")
    }
}
