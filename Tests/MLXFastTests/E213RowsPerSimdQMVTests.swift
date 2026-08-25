import Foundation
import MLX
import MLXFast
import MLXRandom
import Testing

@testable import MLXLLM

// E213 -- can a lower `rows_per_simd` buy the single-pass form at m = 7, 8
// and 9?
//
// Stage-0 desk result, recorded before this arm existed:
//
//   * `(6,5)` is not a partition this template builds. `qwen_e120_qmv_m`
//     asserts `M % IPG != 1`, and `6 % 5 == 1` makes a one-input tail group.
//   * `G(m) = ceil(m / IPG)` does not move for any rows-per-group retune below
//     m = 9, so the assigned family was null by construction and was withdrawn.
//   * The AIR register-pressure proxy puts `qwen_e120_qmv_wide<NA>` at
//     `rows_per_simd = 4` at 125 lane-weighted live values for NA = 5 and 144
//     for NA = 6, so the Apple 128-register boundary falls between them. That
//     boundary explains E208's win at NA = 5 and E195's single-pass losses at
//     NA = 7 and NA = 8.
//
// `rows_per_simd` multiplies the live accumulator set, so lowering it moves the
// boundary. The same proxy over the grid:
//
//     rows \ NA      5      6      7      8      9
//     4 (shipped)  125    144    155    175    195
//     3            108    124    140    166    185
//     2             91    104    117    132    147
//     1             74     84     94    104    114
//
// `(rows 1, NA 7)` = 94, `(rows 1, NA 8)` = 104 and `(rows 1, NA 9)` = 114 all
// sit under today's shipped 125 while giving `G = 1`. That is the `g1` arm: it
// removes the second weight pass at the three widths that carry 76 % of cap-8
// rounds. `probeRows2` is the attribution control for a loss: `(9, 5, rows 2)`
// holds `G = 2` and lowers `rows` alone.
//
// m = 7 uses `rows = 1` rather than the cheaper-looking `rows = 2`: the first
// `g1` gate found `<NA = 7, USE_TABLE = false, ROWS = 2>` numerically wrong at
// every cell, while `<NA = 7, ROWS = 4>` and `<NA = 5, ROWS = 2>` are both bit
// exact. Only that pair is wrong, and no plan may compile it.
//
// Four sections:
//
//   * `planTable` is a pure-function test. It proves the legality rule, the
//     group counts each arm claims, that no arm moves a width it does not
//     claim, that the emitted instantiation text is unchanged wherever
//     `rows_per_simd` stays 4, and that the witness a leg emits names its own
//     table;
//   * `launchGeometry` proves the launched y extent covers every output row of
//     every cell exactly once at every plan entry, so no row is dropped and no
//     thread addresses a row past the end;
//   * `plainAgreesWithTable` sweeps the `(NA, ROWS)` grid every plan compiles,
//     plus a research-only `(NA 7, ROWS 2)` witness, and reports where the two
//     pipelines disagree. It asserts that no `rows = 4` instantiation and no
//     scored plain width disagrees;
//   * `numericalGate` compares actual floating-point values, cell by cell, with
//     real packed 4-bit weights and real bfloat16 activations. It gates three
//     claims: the `ROWS` parameterization leaves the shipped `rows = 4` path
//     bit-identical to the pre-E213 header at every width; each arm is
//     bit-exact against the shipped partition at every width it moves; and a
//     perturbed build proves the comparison can fail under both geometries.
//
// The expectation is bit-exact. `qwen_e120_qmv_wide` keeps input row `m` in
// lane `m` of every `vec<float, NA>` it touches, and output row `out_row + r`
// in accumulator `r`; no operation mixes lanes or rows, so regrouping input
// rows or re-splitting output rows across simdgroups cannot reorder any
// output's own reduction.
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
    // Not a decode cell. `routable` admits any `n % 8 == 0, n >= 4096`, and
    // 4104 = 8 * 513 is the tightest y extent the plan can meet: at rows = 1 it
    // launches an odd number of threadgroups, so a y-extent defect that rounds
    // or drops the last threadgroup shows here and nowhere else.
    E213Cell(name: "edge.n4104", k: 5120, n: 4104, invocations: 0),
]

/// `MLXFAST_E213_CELLS` selects a subset for a smoke run. The full list is the
/// default and the only list a reported result may use.
private let e213Cells: [E213Cell] = {
    guard
        let raw = ProcessInfo.processInfo.environment["MLXFAST_E213_CELLS"],
        !raw.isEmpty
    else { return e213AllCells }
    let wanted = Set(raw.split(separator: ",").map(String.init))
    return e213AllCells.filter { wanted.contains($0.name) }
}()

/// The header exactly as it stood before E213 added the `ROWS` template
/// argument: `rows_per_simd` is the literal 4 again and the inner calls take
/// two template arguments. Derived from the shipped text by substitution, so it
/// tracks any later edit to the body instead of freezing a stale copy.
private let e213LegacyHeader = qwen35E120QMVHeader
    .replacingOccurrences(
        of: "template <int NA, bool USE_TABLE, int ROWS>",
        with: "template <int NA, bool USE_TABLE>")
    .replacingOccurrences(
        of: "constexpr int rows_per_simd = ROWS;",
        with: "constexpr int rows_per_simd = 4;")
    .replacingOccurrences(
        of: "template <int M, int IPG, bool USE_TABLE, int ROWS = 4>",
        with: "template <int M, int IPG, bool USE_TABLE>")
    .replacingOccurrences(
        of: "qwen_e120_qmv_wide<IPG, USE_TABLE, ROWS>",
        with: "qwen_e120_qmv_wide<IPG, USE_TABLE>")
    .replacingOccurrences(
        of: "qwen_e120_qmv_wide<(TAIL >= 2 ? TAIL : 2), USE_TABLE, ROWS>",
        with: "qwen_e120_qmv_wide<(TAIL >= 2 ? TAIL : 2), USE_TABLE>")

/// `<NA = 7, ROWS = 2>` at m = 7, built here instead of in a shipped plan.
///
/// The first `g1` gate compiled that instantiation through `stagedG1` and found
/// it numerically wrong on the plain pipeline at every cell. No shipped or
/// research plan may carry a known-wrong instantiation into a submitted
/// snapshot, so the plan now uses `(7, 7, 1)` and this text-level witness keeps
/// the defect reproducible from research-only code.
///
/// Derived from the `singlePass` source, which emits `<7, 7, flag>` at the
/// default `ROWS`, by rewriting that one case to the `ROWS = 2` template
/// argument and the matching output-row expression.
private func e213Rows2NA7Source(table: Bool) -> String {
    let source = qwen35E120QMVSource(table: table, variant: .singlePass)
    let flag = table ? "USE_TABLE" : "false"
    let original = e213CaseText(source, m: 7)
    let rewritten = original
        .replacingOccurrences(
            of: "qwen_e120_qmv_m<7, 7, \(flag)>",
            with: "qwen_e120_qmv_m<7, 7, \(flag), 2>")
        .replacingOccurrences(
            of: "qmv_out_row",
            with: "int(qmv_tid.y) * 4 + int(qmv_sgid) * 2")
    precondition(!original.isEmpty && rewritten != original)
    return source.replacingOccurrences(of: original, with: rewritten)
}

/// The `case <m>:` block of a compiled switch, so a test can compare one
/// width's emitted text without matching the whole source.
private func e213CaseText(_ source: String, m: Int) -> String {
    guard let start = source.range(of: "case \(m):") else { return "" }
    let rest = source[start.lowerBound...]
    guard let end = rest.range(of: "break;") else { return String(rest) }
    return String(rest[..<end.upperBound])
}

/// A JIT kernel pair for one compiled plan, built from the production source so
/// the instrument cannot drift from the shipped text. The scored decode path
/// launches through `Qwen35CachedKernel`, so this does too.
private struct E213Pipeline {
    var variant: Qwen35QMVKernelVariant
    var label: String
    var table: Qwen35CachedKernel
    var plain: Qwen35CachedKernel
    /// Rows per simdgroup this build really compiled, when it is not the value
    /// `variant` would give. The launcher must match the compiled body.
    var rowsOverride: Int?

    init(
        variant: Qwen35QMVKernelVariant, label: String,
        header: String? = nil, nameSuffix: String = "",
        source: ((Bool) -> String)? = nil, rowsOverride: Int? = nil
    ) {
        self.variant = variant
        self.label = label
        self.rowsOverride = rowsOverride
        let head = header ?? qwen35E120QMVHeader
        let body = source ?? { qwen35E120QMVSource(table: $0, variant: variant) }
        self.table = Qwen35CachedKernel(
            name: "e213_qmv_table_\(label)\(nameSuffix)",
            inputNames: ["w", "scales", "biases", "x", "xsums"],
            outputNames: ["y"],
            source: body(true),
            header: head)
        self.plain = Qwen35CachedKernel(
            name: "e213_qmv_plain_\(label)\(nameSuffix)",
            inputNames: ["w", "scales", "biases", "x"],
            outputNames: ["y"],
            source: body(false),
            header: head)
    }

    func call(
        x: MLXArray, w: MLXArray, scales: MLXArray, biases: MLXArray,
        xsums: MLXArray, m: Int, n: Int, useTable: Bool
    ) -> MLXArray {
        var outShape = x.shape
        outShape[outShape.count - 1] = n
        var grid = Qwen35CustomQMV.launchGrid(m: m, n: n, variant: variant)
        if let rows = rowsOverride { grid.1 = (n / (2 * rows)) * 2 }
        let launch = Qwen35KernelLaunch(
            grid: grid,
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

/// The output columns two plans cut differently, plus the ends of the y extent.
/// A `rows_per_simd` change re-splits output rows across simdgroups, so the
/// samples sit at the simdgroup and threadgroup cuts of both plans and at the
/// last threadgroup, where a dropped or truncated y extent would show.
private func e213SampleColumns(n: Int, rowsA: Int, rowsB: Int) -> [Int] {
    var columns: Set<Int> = [0, n - 2, n - 1]
    for rows in [rowsA, rowsB] {
        for cut in [rows, 2 * rows, n - 2 * rows, n - rows] {
            columns.insert(max(0, min(n - 1, cut - 1)))
            columns.insert(max(0, min(n - 1, cut)))
        }
    }
    return columns.sorted()
}

private func e213Samples(
    _ values: [Float], n: Int, m: Int, columns: [Int]
) -> [[String: Any]] {
    var samples: [[String: Any]] = []
    for row in 0 ..< m {
        var hex: [String] = []
        for column in columns {
            let index = row * n + column
            guard index < values.count else { break }
            hex.append(e213Hex(values[index]))
        }
        samples.append(["input_row": row, "hexfloat": hex])
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

@Suite("E213 QMV rows-per-simdgroup single-pass plan")
struct E213RowsPerSimdQMVTests {
    static let gateEnabled =
        ProcessInfo.processInfo.environment["MLXFAST_RUN_E213_GATE"] == "1"

    /// Widths each arm moves. An arm compiles the shipped instantiation at
    /// every other width.
    static let armWidths: [(variant: Qwen35QMVKernelVariant, moved: [Int])] = [
        (.stagedG1, [7, 8, 9]),
        (.probeRows2, [9]),
    ]

    /// Lane-weighted peak live values that `research/e213_regs.py` counts in
    /// `qwen_e120_qmv_wide`, keyed by `rows_per_simd x NA`. The gate reports
    /// them next to the plan, so the register claim travels with the evidence.
    static let registerProxy: [String: Int] = [
        "4x5": 125, "4x6": 144, "4x7": 155, "4x8": 175, "4x9": 195,
        "1x7": 94, "1x8": 104, "1x9": 114, "2x5": 91, "2x7": 117,
    ]

    // MARK: - plan table

    @Test
    func planTable() {
        let shipped = Qwen35QMVKernelVariant.staged.pairs

        // The shipped plan is unchanged: every width keeps E120's
        // `rows_per_simd = 4`, and E208's `(9,5)` still stands.
        for entry in shipped {
            #expect(entry.rows == 4, "width \(entry.m) must keep rows = 4")
        }
        #expect(shipped.first { $0.m == 9 }?.ipg == 5)

        for (variant, moved) in Self.armWidths {
            let arm = variant.pairs
            #expect(arm.count == shipped.count)
            for (armEntry, shippedEntry) in zip(arm, shipped) {
                #expect(armEntry.m == shippedEntry.m)
                // `M % IPG == 1` is the one partition the template refuses to
                // build, so every entry must clear it.
                #expect(armEntry.m % armEntry.ipg != 1)
                let same =
                    armEntry.ipg == shippedEntry.ipg
                    && armEntry.rows == shippedEntry.rows
                #expect(
                    same != moved.contains(armEntry.m),
                    "width \(armEntry.m) moves in \(variant.rawValue) only if claimed"
                )
            }
        }

        // What the arms claim about weight passes. `g1` removes the second pass
        // at m = 7, 8 and 9; `probeRows2` holds every group count, which is what
        // makes it an attribution control rather than a second mechanism.
        for m in Qwen35CustomQMV.widths {
            let expected =
                m <= 6
                ? Qwen35CustomQMV.activeInputGroups(m, variant: .staged) : 1
            #expect(
                Qwen35CustomQMV.activeInputGroups(m, variant: .stagedG1)
                    == expected,
                "g1 group count at m = \(m)")
            #expect(
                Qwen35CustomQMV.activeInputGroups(m, variant: .probeRows2)
                    == Qwen35CustomQMV.activeInputGroups(m, variant: .staged),
                "probe must not move a group count at m = \(m)")
        }
        for m in [7, 8, 9] {
            #expect(Qwen35CustomQMV.activeInputGroups(m, variant: .staged) == 2)
        }

        // Every moved entry must sit under the 128-register boundary the proxy
        // located, and the shipped `rows = 4` single-pass forms must not.
        for (variant, moved) in Self.armWidths {
            for m in moved {
                let rows = Qwen35CustomQMV.rowsPerSimd(m, variant: variant)
                let na = Qwen35CustomQMV.inputsPerGroup(m, variant: variant)
                let peak = Self.registerProxy["\(rows)x\(na)"]
                #expect(peak != nil, "no register census for \(rows)x\(na)")
                #expect(
                    (peak ?? 999) < 128,
                    "\(variant.rawValue) m = \(m) is over the boundary")
            }
        }
        #expect((Self.registerProxy["4x7"] ?? 0) > 128)
        #expect((Self.registerProxy["4x8"] ?? 0) > 128)

        // Each arm's source is the shipped source with only its own cases
        // substituted, and an untouched width emits byte-identical text.
        for table in [true, false] {
            let base = qwen35E120QMVSource(table: table, variant: .staged)
            for (variant, moved) in Self.armWidths {
                let armSource = qwen35E120QMVSource(table: table, variant: variant)
                #expect(armSource != base)
                for m in Qwen35CustomQMV.widths {
                    #expect(
                        (e213CaseText(base, m: m) == e213CaseText(armSource, m: m))
                            != moved.contains(m),
                        "\(variant.rawValue) case \(m) text")
                }
                for m in moved {
                    let rows = Qwen35CustomQMV.rowsPerSimd(m, variant: variant)
                    let ipg = Qwen35CustomQMV.inputsPerGroup(m, variant: variant)
                    let text = e213CaseText(armSource, m: m)
                    #expect(text.contains("qwen_e120_qmv_m<\(m), \(ipg),"))
                    #expect(
                        text.contains(
                            "int(qmv_tid.y) * \(2 * rows) + int(qmv_sgid) * \(rows)"
                        ))
                }
            }
            // The shipped plan keeps `rows_per_simd` at its default, so every
            // case passes the shared `qmv_out_row` and carries no `ROWS`
            // argument or per-case row expression.
            #expect(base.contains("int(qmv_tid.y) * 8 + int(qmv_sgid) * 4;"))
            for m in Qwen35CustomQMV.widths {
                let text = e213CaseText(base, m: m)
                #expect(text.contains("qmv_out_row, qmv_lid"))
                #expect(!text.contains("qmv_sgid"))
            }
        }

        // `ROWS` is confined to the template argument, the one constant it
        // feeds, and the two inner calls.
        #expect(!e213LegacyHeader.contains("ROWS"))
        #expect(e213LegacyHeader.contains("constexpr int rows_per_simd = 4;"))
        #expect(qwen35E120QMVHeader.contains("constexpr int rows_per_simd = ROWS;"))

        // The witness each leg emits, derived from its own table.
        #expect(
            qwen35QMVWidthPlanWitnessText(for: .staged) == "selective-m6+ipg9-5")
        #expect(
            qwen35QMVWidthPlanWitnessText(for: .stagedG1)
                == "selective-m6+ipg9-9+e213-3x4-7x1-8x1-9x1")
        #expect(
            qwen35QMVWidthPlanWitnessText(for: .probeRows2)
                == "selective-m6+ipg9-5+e213-3x4-4x4-4x4-5x2")
        // No arm selected means the shipped plan, so a stray environment cannot
        // publish a research table as the shipped one.
        #expect(
            qwen35QMVWidthPlanWitness
                == qwen35QMVWidthPlanWitnessText(for: qwen35E213StagedVariant))

        // The arms may not disturb the m = 6 single-pass selection E195
        // shipped. At m = 6 the staged table governs `mlp.down` alone.
        #expect(
            Qwen35CustomQMV.kernelVariant((m: 6, k: 5120, n: 248_320))
                == .singlePass)
        #expect(
            Qwen35CustomQMV.kernelVariant((m: 6, k: 17408, n: 5120))
                == qwen35E213StagedVariant)
        #expect(
            Qwen35QMVKernelVariant.singlePass.pairs.allSatisfy { $0.rows == 4 })
    }

    // MARK: - launch geometry

    /// The launched y extent must cover every output row exactly once at every
    /// plan entry: no row twice, no row dropped, no thread past the end. The
    /// threadgroup is `(32, 2, 1)`, so a threadgroup covers `2 * rows` output
    /// rows, and `dispatch_threads` counts the y extent in threads.
    @Test
    func launchGeometry() {
        for variant in Qwen35QMVKernelVariant.allCases {
            for m in Qwen35CustomQMV.widths {
                let rows = Qwen35CustomQMV.rowsPerSimd(m, variant: variant)
                let ipg = Qwen35CustomQMV.inputsPerGroup(m, variant: variant)
                #expect([1, 2, 4].contains(rows))
                for cell in e213AllCells {
                    let grid = Qwen35CustomQMV.launchGrid(
                        m: m, n: cell.n, variant: variant)
                    #expect(grid.2 == 1)
                    // y is counted in threads and the threadgroup is 2 deep.
                    #expect(grid.1 % 2 == 0)
                    let threadgroups = grid.1 / 2
                    #expect(
                        threadgroups * 2 * rows == cell.n,
                        """
                        \(variant.rawValue) m=\(m) \(cell.name): \
                        \(threadgroups) threadgroups * \(2 * rows) rows \
                        != n=\(cell.n)
                        """)
                    // The last simdgroup's last accumulator is the last output
                    // row, so nothing addresses a row past the end.
                    let lastOutRow = (threadgroups - 1) * 2 * rows + rows
                    #expect(lastOutRow + rows - 1 == cell.n - 1)
                    // Every input row is served by exactly one x group.
                    let groups = grid.0 / 32
                    #expect(
                        groups
                            == Qwen35CustomQMV.activeInputGroups(
                                m, variant: variant))
                    #expect((groups - 1) * ipg < m)
                    #expect(groups * ipg >= m)
                }
            }
        }

        // The shipped plan keeps E120's launch arithmetic exactly.
        for cell in e213AllCells {
            for m in Qwen35CustomQMV.widths {
                let grid = Qwen35CustomQMV.launchGrid(
                    m: m, n: cell.n, variant: .staged)
                #expect(grid.1 == (cell.n / 8) * 2)
            }
        }
    }

    // MARK: - numerical gate

    /// Where the plain pipeline stops agreeing with the table pipeline.
    ///
    /// The two paths are documented to agree bit for bit: the table entry holds
    /// the same float accumulation of the same BF16 expression tree in the same
    /// `i` order that `sums[m] += xv[0] + xv[1] + xv[2] + xv[3]` builds. The
    /// first `g1` gate found that they disagree at `<NA = 7, USE_TABLE = false,
    /// ROWS = 2>`, and that input row 0 is the only correct row.
    ///
    /// The sweep separates the candidate causes over the `(NA, ROWS)` grid:
    /// `singlePass` at m = 7 is `NA = 7` at the shipped `ROWS = 4`, and
    /// `probeRows2` at m = 9 is `ROWS = 2` at `NA = 5`. Both agree, so neither
    /// `NA = 7` nor `ROWS = 2` is wrong on its own and only the pair is.
    ///
    /// Two claims this section asserts rather than reports:
    ///
    ///   * every `rows = 4` instantiation agrees, so the base compiles no wrong
    ///     plain instantiation and the defect is reachable only through E213's
    ///     new `ROWS` values;
    ///   * the widths the scored `sumTable` arm really routes to the plain
    ///     kernel agree. `tablePays(m) = m >= 4`, so those are m = 2 and m = 3,
    ///     and both keep `rows = 4` in every plan.
    @Test(.enabled(if: E213RowsPerSimdQMVTests.gateEnabled))
    func plainAgreesWithTable() throws {
        struct Case {
            var label: String
            var widths: [Int]
            var rows: (Int) -> Int
            var pipeline: E213Pipeline
        }

        let sweep: [(variant: Qwen35QMVKernelVariant, widths: [Int])] = [
            (.staged, Array(Qwen35CustomQMV.widths)),
            (.singlePass, [6, 7, 8, 9]),
            (.stagedG1, [7, 8, 9]),
            (.probeRows2, [9]),
        ]
        var pipelines = sweep.map { entry in
            Case(
                label: entry.variant.rawValue, widths: entry.widths,
                rows: { Qwen35CustomQMV.rowsPerSimd($0, variant: entry.variant) },
                pipeline: E213Pipeline(
                    variant: entry.variant,
                    label: "pvt_\(entry.variant.rawValue)",
                    nameSuffix: "_pvt"))
        }
        // The withdrawn instantiation, kept reproducible outside every plan.
        pipelines.append(
            Case(
                label: "research_na7_rows2", widths: [7], rows: { _ in 2 },
                pipeline: E213Pipeline(
                    variant: .singlePass, label: "pvt_na7_rows2",
                    nameSuffix: "_pvt", source: e213Rows2NA7Source(table:),
                    rowsOverride: 2)))

        var rows: [[String: Any]] = []
        for cell in e213Cells {
            autoreleasepool {
                let (w, scales, biases) = e213PackedCell(
                    k: cell.k, n: cell.n, seed: 0xE213)
                for m in Qwen35CustomQMV.widths {
                    autoreleasepool {
                        let x = e213Activations(
                            m: m, k: cell.k, seed: UInt64(0xE213_0000 + m))
                        let xsums = Qwen35CustomQMV.xsumsTable(x)
                        eval(xsums)
                        for entry in pipelines where entry.widths.contains(m) {
                            let plain = entry.pipeline.call(
                                x: x, w: w, scales: scales, biases: biases,
                                xsums: xsums, m: m, n: cell.n, useTable: false)
                            let table = entry.pipeline.call(
                                x: x, w: w, scales: scales, biases: biases,
                                xsums: xsums, m: m, n: cell.n, useTable: true)
                            eval(plain, table)
                            let plainValues = plain.asType(.float32)
                                .asArray(Float.self)
                            let tableValues = table.asType(.float32)
                                .asArray(Float.self)
                            let diff = e213Compare(plainValues, tableValues)
                            // Which input rows the plain path gets wrong. The
                            // failing signature is "every row except row 0".
                            var wrongRows: [Int] = []
                            for row in 0 ..< m {
                                let lo = row * cell.n
                                let hi = min(lo + cell.n, plainValues.count)
                                for i in lo ..< hi
                                where plainValues[i].bitPattern
                                    != tableValues[i].bitPattern
                                {
                                    wrongRows.append(row)
                                    break
                                }
                            }
                            rows.append([
                                "variant": entry.label,
                                "cell": cell.name,
                                "m": m,
                                "na_first_group": Qwen35CustomQMV
                                    .inputsPerGroup(
                                        m, variant: entry.pipeline.variant),
                                "rows_per_simd": entry.rows(m),
                                "scored_plain_width":
                                    !Qwen35CustomQMV.tablePays(m: m),
                                "elements": m * cell.n,
                                "differing": diff.count,
                                "max_ulp": diff.maxUlp,
                                "max_abs_diff": diff.maxAbsDiff,
                                "wrong_input_rows": wrongRows,
                            ])
                        }
                    }
                }
            }
        }

        try e213Write(
            [
                "harness": "local",
                "experiment": "e213",
                "section": "plain_vs_table",
                "cool_gate_passed_real_gate": false,
                "gate_qualified_for_timing": false,
                "official_or_ranked_score": false,
                "claim": "the plain and table pipelines agree bit for bit",
                "rows": rows,
            ], to: "MLXFAST_E213_PLAIN_OUT")

        let broken = rows.filter { ($0["differing"] as? Int ?? 0) > 0 }

        // The base compiles no wrong plain instantiation: every `rows = 4`
        // entry, at every width and cell, agrees bit for bit.
        for row in broken where (row["rows_per_simd"] as? Int) == 4 {
            Issue.record(
                """
                base defect: rows=4 \(row["variant"] ?? "") \
                m=\(row["m"] ?? "") \(row["cell"] ?? "") differs, \
                \(row["differing"] ?? "") elements
                """)
        }

        // Affirmative scored-surface check. The `sumTable` arm reaches the
        // plain kernel only where `tablePays` is false, so those widths must
        // agree by measurement, not by a routing argument alone.
        let scored = rows.filter { $0["scored_plain_width"] as? Bool == true }
        #expect(!scored.isEmpty)
        #expect(scored.allSatisfy { ($0["differing"] as? Int ?? 0) == 0 })

        // The rest is reported, not asserted: this section exists to localize
        // a known defect, so it must produce its whole table.
        Issue.record(
            "plain vs table: \(broken.count) of \(rows.count) instantiations differ")
    }

    @Test(.enabled(if: E213RowsPerSimdQMVTests.gateEnabled))
    func numericalGate() throws {
        let shippedPlan = E213Pipeline(variant: .staged, label: "shipped")
        // The pre-E213 header, compiled under its own kernel name. The shipped
        // `rows = 4` widths must be bit-identical against it, otherwise the
        // parameterization moved code it was not supposed to touch.
        let legacyPlan = E213Pipeline(
            variant: .staged, label: "legacy", header: e213LegacyHeader,
            nameSuffix: "_lg")

        // Positive control: the shipped source with one accumulation perturbed
        // by a single bfloat16 ULP of relative scale. The comparison must
        // report differences for this build, otherwise a pass proves nothing.
        // It is compiled under both geometries, so the control also proves the
        // arm's own launch reads every output the comparison claims to check.
        let perturbedHeader = qwen35E120QMVHeader.replacingOccurrences(
            of: "acc[r] += scale_local[r] * partial[r] + sums * bias_local[r];",
            with:
                "acc[r] += scale_local[r] * partial[r] * 1.0000305f "
                + "+ sums * bias_local[r];")
        #expect(perturbedHeader != qwen35E120QMVHeader)
        let perturbed = E213Pipeline(
            variant: .staged, label: "perturbed", header: perturbedHeader,
            nameSuffix: "_ctl")
        let perturbedG1 = E213Pipeline(
            variant: .stagedG1, label: "perturbed_g1", header: perturbedHeader,
            nameSuffix: "_ctl")

        let arms = Self.armWidths.map {
            (
                variant: $0.variant, moved: $0.moved,
                pipeline: E213Pipeline(
                    variant: $0.variant, label: $0.variant.rawValue)
            )
        }

        var rows: [[String: Any]] = []
        var legacyRows: [[String: Any]] = []
        var control: [[String: Any]] = []
        var worstUlp = 0
        var totalDiffering = 0
        var totalElements = 0

        for cell in e213Cells {
            autoreleasepool {
                let (w, scales, biases) = e213PackedCell(
                    k: cell.k, n: cell.n, seed: 0xE213)

                for m in Qwen35CustomQMV.widths {
                    autoreleasepool {
                        let x = e213Activations(
                            m: m, k: cell.k, seed: UInt64(0xE213_0000 + m))
                        let xsums = Qwen35CustomQMV.xsumsTable(x)
                        eval(xsums)

                        // 1. The parameterization is inert at rows = 4.
                        for useTable in [true, false] {
                            let new = shippedPlan.call(
                                x: x, w: w, scales: scales, biases: biases,
                                xsums: xsums, m: m, n: cell.n, useTable: useTable)
                            let old = legacyPlan.call(
                                x: x, w: w, scales: scales, biases: biases,
                                xsums: xsums, m: m, n: cell.n, useTable: useTable)
                            eval(new, old)
                            let diff = e213Compare(
                                new.asType(.float32).asArray(Float.self),
                                old.asType(.float32).asArray(Float.self))
                            worstUlp = max(worstUlp, diff.maxUlp)
                            totalDiffering += diff.count
                            totalElements += m * cell.n
                            legacyRows.append([
                                "cell": cell.name, "m": m, "use_table": useTable,
                                "elements": m * cell.n,
                                "differing": diff.count,
                                "max_ulp": diff.maxUlp,
                                "max_abs_diff": diff.maxAbsDiff,
                                "non_finite": diff.nanOrInf,
                            ])
                            #expect(
                                diff.count == 0,
                                """
                                ROWS parameterization moved \(cell.name) \
                                m=\(m) table=\(useTable): \(diff.count) \
                                differing, max_ulp=\(diff.maxUlp)
                                """)
                            #expect(diff.nanOrInf == 0)
                        }

                        // 2. Each arm against the shipped partition, at every
                        //    width it moves.
                        for arm in arms where arm.moved.contains(m) {
                            let armRows = Qwen35CustomQMV.rowsPerSimd(
                                m, variant: arm.variant)
                            let armIPG = Qwen35CustomQMV.inputsPerGroup(
                                m, variant: arm.variant)
                            let grid = Qwen35CustomQMV.launchGrid(
                                m: m, n: cell.n, variant: arm.variant)
                            let columns = e213SampleColumns(
                                n: cell.n, rowsA: 4, rowsB: armRows)
                            for useTable in [true, false] {
                                let a = arm.pipeline.call(
                                    x: x, w: w, scales: scales, biases: biases,
                                    xsums: xsums, m: m, n: cell.n,
                                    useTable: useTable)
                                let b = shippedPlan.call(
                                    x: x, w: w, scales: scales, biases: biases,
                                    xsums: xsums, m: m, n: cell.n,
                                    useTable: useTable)
                                eval(a, b)
                                let av = a.asType(.float32).asArray(Float.self)
                                let bv = b.asType(.float32).asArray(Float.self)
                                let diff = e213Compare(av, bv)
                                worstUlp = max(worstUlp, diff.maxUlp)
                                totalDiffering += diff.count
                                totalElements += m * cell.n
                                rows.append([
                                    "arm": arm.variant.rawValue,
                                    "cell": cell.name, "k": cell.k, "n": cell.n,
                                    "m": m, "use_table": useTable,
                                    "shipped_ipg": Qwen35CustomQMV
                                        .inputsPerGroup(m, variant: .staged),
                                    "arm_ipg": armIPG,
                                    "shipped_rows_per_simd": Qwen35CustomQMV
                                        .rowsPerSimd(m, variant: .staged),
                                    "arm_rows_per_simd": armRows,
                                    "shipped_groups": Qwen35CustomQMV
                                        .activeInputGroups(m, variant: .staged),
                                    "arm_groups": Qwen35CustomQMV
                                        .activeInputGroups(m, variant: arm.variant),
                                    "arm_register_proxy":
                                        Self.registerProxy["\(armRows)x\(armIPG)"]
                                        ?? -1,
                                    "arm_grid_x": grid.0,
                                    "arm_grid_y": grid.1,
                                    "invocations_per_round": cell.invocations,
                                    "elements": m * cell.n,
                                    "differing": diff.count,
                                    "max_ulp": diff.maxUlp,
                                    "max_abs_diff": diff.maxAbsDiff,
                                    "non_finite": diff.nanOrInf,
                                    "first_differing_index":
                                        diff.firstDifferingIndex ?? -1,
                                    "sample_columns": columns,
                                    "hexfloat_shipped": e213Samples(
                                        bv, n: cell.n, m: m, columns: columns),
                                    "hexfloat_arm": e213Samples(
                                        av, n: cell.n, m: m, columns: columns),
                                ])
                                #expect(
                                    diff.count == 0,
                                    """
                                    \(arm.variant.rawValue) \(cell.name) \
                                    m=\(m) table=\(useTable): \(diff.count) \
                                    differing outputs, max_ulp=\(diff.maxUlp), \
                                    max_abs_diff=\(diff.maxAbsDiff)
                                    """)
                                #expect(diff.nanOrInf == 0)
                            }
                        }

                        // 3. Positive control, on one scored cell, under the
                        //    shipped geometry at every width and under the g1
                        //    geometry at the widths it moves.
                        guard cell.name == "gdn.out_proj" else { return }
                        for useTable in [true, false] {
                            let good = shippedPlan.call(
                                x: x, w: w, scales: scales, biases: biases,
                                xsums: xsums, m: m, n: cell.n, useTable: useTable)
                            let bad = perturbed.call(
                                x: x, w: w, scales: scales, biases: biases,
                                xsums: xsums, m: m, n: cell.n, useTable: useTable)
                            eval(good, bad)
                            let goodValues = good.asType(.float32).asArray(
                                Float.self)
                            let shippedDiff = e213Compare(
                                bad.asType(.float32).asArray(Float.self),
                                goodValues)
                            var g1Diff: E213Diff?
                            if [7, 8, 9].contains(m) {
                                let badG1 = perturbedG1.call(
                                    x: x, w: w, scales: scales, biases: biases,
                                    xsums: xsums, m: m, n: cell.n,
                                    useTable: useTable)
                                eval(badG1)
                                g1Diff = e213Compare(
                                    badG1.asType(.float32).asArray(Float.self),
                                    goodValues)
                            }
                            control.append([
                                "cell": cell.name, "m": m,
                                "use_table": useTable,
                                "shipped_geometry_differing": shippedDiff.count,
                                "shipped_geometry_max_ulp": shippedDiff.maxUlp,
                                "g1_geometry_differing": g1Diff?.count ?? -1,
                                "g1_geometry_max_ulp": g1Diff?.maxUlp ?? -1,
                            ])
                            #expect(
                                shippedDiff.count > 0,
                                "positive control did not trip at m=\(m)")
                            if let g1Diff {
                                #expect(
                                    g1Diff.count > 0,
                                    "g1 positive control did not trip at m=\(m)")
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
                "mechanism": "rows_per_simd co-tuned with IPG for G = 1",
                "cool_gate_passed_real_gate": false,
                "gate_qualified_for_timing": false,
                "official_or_ranked_score": false,
                "tolerance": "bit-exact: zero differing outputs, zero ULP",
                "active_plan": qwen35QMVWidthPlanWitness,
                "qmv_arm": Qwen35CustomQMV.arm.rawValue,
                "register_proxy_peak_live": Self.registerProxy,
                "worst_max_ulp": worstUlp,
                "total_differing": totalDiffering,
                "total_elements": totalElements,
                "parameterization_identity": legacyRows,
                "cells": rows,
                "positive_control": control,
            ], to: "MLXFAST_E213_GATE_OUT")
    }
}
