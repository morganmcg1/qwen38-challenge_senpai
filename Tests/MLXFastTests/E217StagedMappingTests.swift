import Foundation
import MLX
import MLXFast
import MLXRandom
import Testing

@testable import MLXLLM

// E217 -- can a `G == 2` width pay the per-pass fixed cost of the wide QMV
// kernel once instead of twice, without changing a single output bit?
//
// The shipped `split` mapping gives one threadgroup eight output rows and ONE
// token-column group, so a two-pass width launches the same weight bytes twice
// in two different threadgroups. Two research mappings try to remove that
// duplication:
//
//   * `coop` puts both column groups in one threadgroup over the same four
//     output rows and shares nothing explicitly. E216 measured this mapping and
//     it LOST (FINDING 565). It stays here only as a cross-host replication
//     arm;
//   * `staged` keeps the shipped eight output rows, spends four simdgroups
//     (two row halves x two column groups), and stages one raw packed eight-row
//     weight tile in threadgroup memory. RULE 404 requires a named physical
//     dedup resource; threadgroup memory is that resource.
//
// This file is the Stage-0 falsification gate that must pass before any timing.
// Two sections:
//
//   * `mappingGeometry` is a pure-function test of the launch geometry. It
//     proves each mapping covers every output row exactly once, that only a
//     `G == 2` width can leave the shipped mapping, and that the constants the
//     Python coverage walk (`research/e217_coverage.py`) modelled are the
//     constants Swift really launches;
//   * `numericalGate` compares both research mappings against the shipped one,
//     cell by cell and width by width, with real packed 4-bit weights and real
//     bfloat16 activations. It compares actual floating-point values, not
//     argmax, and runs one positive control per mapping that proves the
//     comparison can fail.
//
// The expectation is bit-exact. `staged` reads the SAME packed device word for
// a given (output row, column) that `split` would read, only through
// threadgroup memory, and it keeps every reduction inside one row and one
// token lane. The one real restructuring risk is the shared `vec<float, IPG>`
// accumulator: `qwen_e217_consume_block` must serve both an `NA == IPG` and an
// `NA == TAIL` branch, so the shipped vector statement
// `acc[r] += scale_local[r] * partial[r] + sums * bias_local[r]` becomes a
// per-component statement. That rewrite is what this gate is really for.
// Anything other than zero differing outputs stops the experiment before
// Stage 1.
//
// Research instrument. `Tests/` is never packaged into a submission.
// harness=local, no thermal gate, no score.

private struct E217Cell {
    var name: String
    var k: Int
    var n: Int
    /// Wide-QMV invocations of this cell in one decode round (FINDING 487).
    var invocations: Int
}

private let e217AllCells = [
    E217Cell(name: "mlp.gate_up", k: 5120, n: 34816, invocations: 64),
    E217Cell(name: "mlp.down", k: 17408, n: 5120, invocations: 64),
    E217Cell(name: "gdn.in_proj", k: 5120, n: 16480, invocations: 48),
    E217Cell(name: "gdn.out_proj", k: 6144, n: 5120, invocations: 48),
    E217Cell(name: "fa.qkv", k: 5120, n: 14336, invocations: 16),
    E217Cell(name: "fa.o_proj", k: 6144, n: 5120, invocations: 16),
    E217Cell(name: "lm_head", k: 5120, n: 248_320, invocations: 1),
    // Not a decode cell. `n / 8` is odd here, so the staged grid runs an odd
    // number of threadgroups and the last one is not aligned to any wider
    // stride. It costs little and it exercises grid arithmetic the seven
    // scored shapes never reach.
    E217Cell(name: "synthetic.n4104", k: 5120, n: 4104, invocations: 0),
]

/// `MLXFAST_E217_CELLS` selects a subset for a smoke run. The full list is the
/// default and the only list a reported result may use.
private let e217Cells: [E217Cell] = {
    guard
        let raw = ProcessInfo.processInfo.environment["MLXFAST_E217_CELLS"],
        !raw.isEmpty
    else { return e217AllCells }
    let wanted = Set(raw.split(separator: ",").map(String.init))
    return e217AllCells.filter { wanted.contains($0.name) }
}()

/// The widths this experiment moves: exactly the `G == 2` staged pairs, which
/// are the only widths `qwen35E217QMVSource` compiles.
private let e217Widths: [Int] = Qwen35QMVKernelVariant.staged.pairs
    .filter { ($0.m + $0.ipg - 1) / $0.ipg == 2 }
    .map(\.m)
    .sorted()

/// A JIT kernel pair for one grid mapping, built from the production source and
/// launched through the production geometry helpers so the instrument cannot
/// drift from the shipped text.
private struct E217Pipeline {
    var mapping: Qwen35QMVGroupMapping
    var label: String
    var table: Qwen35CachedKernel
    var plain: Qwen35CachedKernel

    init(
        mapping: Qwen35QMVGroupMapping, label: String,
        headerTransform: (String) -> String = { $0 }
    ) {
        self.mapping = mapping
        self.label = label
        let header =
            mapping == .split ? qwen35E120QMVHeader : qwen35E217Header
        func source(_ table: Bool) -> String {
            mapping == .split
                ? qwen35E120QMVSource(table: table, variant: .staged)
                : qwen35E217QMVSource(table: table, mapping: mapping)
        }
        self.table = Qwen35CachedKernel(
            name: "e217_qmv_table_\(label)",
            inputNames: ["w", "scales", "biases", "x", "xsums"],
            outputNames: ["y"],
            source: source(true),
            header: headerTransform(header))
        self.plain = Qwen35CachedKernel(
            name: "e217_qmv_plain_\(label)",
            inputNames: ["w", "scales", "biases", "x"],
            outputNames: ["y"],
            source: source(false),
            header: headerTransform(header))
    }

    func call(
        x: MLXArray, w: MLXArray, scales: MLXArray, biases: MLXArray,
        xsums: MLXArray, m: Int, n: Int, useTable: Bool
    ) -> MLXArray {
        let groups = Qwen35CustomQMV.activeInputGroups(m, variant: .staged)
        var outShape = x.shape
        outShape[outShape.count - 1] = n
        let launch = Qwen35KernelLaunch(
            grid: Qwen35CustomQMV.launchGrid(
                n: n, groups: groups, mapping: mapping),
            threadGroup: Qwen35CustomQMV.launchThreadgroup(mapping: mapping),
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
private func e217PackedCell(k: Int, n: Int, seed: UInt64)
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

private func e217Activations(m: Int, k: Int, seed: UInt64) -> MLXArray {
    MLXRandom.seed(seed)
    let x = MLXRandom.normal([1, m, k]).asType(.bfloat16)
    eval(x)
    return x
}

/// bfloat16 total-order key. bfloat16 is the top half of the float32 bit
/// pattern, so the key orders the finite values monotonically and the
/// difference of two keys is the ULP distance.
private func e217Key(_ value: Float) -> Int {
    let bits = Int(value.bitPattern >> 16)
    let magnitude = bits & 0x7fff
    return (bits & 0x8000) != 0 ? -magnitude : magnitude
}

private struct E217Diff {
    var count: Int
    var maxUlp: Int
    var maxAbsDiff: Double
    var nanOrInf: Int
    var firstDifferingIndex: Int?
}

private func e217Compare(_ a: [Float], _ b: [Float]) -> E217Diff {
    var diff = E217Diff(
        count: 0, maxUlp: 0, maxAbsDiff: 0, nanOrInf: 0,
        firstDifferingIndex: nil)
    for i in 0 ..< min(a.count, b.count) {
        let x = a[i]
        let y = b[i]
        if !x.isFinite || !y.isFinite { diff.nanOrInf += 1 }
        if x.bitPattern == y.bitPattern { continue }
        if diff.firstDifferingIndex == nil { diff.firstDifferingIndex = i }
        diff.count += 1
        diff.maxUlp = max(diff.maxUlp, abs(e217Key(x) - e217Key(y)))
        diff.maxAbsDiff = max(diff.maxAbsDiff, Double(abs(x - y)))
    }
    return diff
}

/// The C99 hexadecimal float literal, plus the raw bits. The literal is the
/// exact value; the bits make an off-by-one-ULP report readable.
private func e217Hex(_ value: Float) -> String {
    String(format: "%a|0x%08x", value, value.bitPattern)
}

/// Hexfloat samples at the rows the mappings treat differently. `staged` cuts
/// the eight output rows of a threadgroup after row 3, and every mapping cuts
/// the token rows after `IPG`. Rows 0, 3, 4 and 7 therefore sit on or beside
/// every seam.
private func e217RowSamples(_ values: [Float], m: Int, n: Int, columns: Int)
    -> [[String: Any]]
{
    var samples: [[String: Any]] = []
    for row in 0 ..< m {
        var hex: [String] = []
        for column in [0, 3, 4, 7] where column < columns * 8 {
            let index = row * n + column
            guard index < values.count else { break }
            hex.append(e217Hex(values[index]))
        }
        samples.append(["row": row, "hexfloat": hex])
    }
    return samples
}

private func e217Write(_ payload: [String: Any], to key: String) throws {
    let path = try #require(
        ProcessInfo.processInfo.environment[key],
        "\(key) must name the JSON destination")
    let data = try JSONSerialization.data(
        withJSONObject: payload,
        options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
    try data.write(to: URL(fileURLWithPath: path))
}

@Suite("E217 staged threadgroup-memory QMV grid mapping")
struct E217StagedMappingTests {
    static let gateEnabled =
        ProcessInfo.processInfo.environment["MLXFAST_RUN_E217_GATE"] == "1"

    // MARK: - mapping geometry

    /// The launch geometry of every mapping, as a pure function. This is the
    /// Swift half of the RULE 403 write-coverage walk: the Python walk enumerates
    /// thread coordinates against these constants, and this test proves the
    /// constants are the ones the production code launches.
    @Test
    func mappingGeometry() throws {
        // Rows and simdgroups per threadgroup.
        #expect(Qwen35QMVGroupMapping.split.rowsPerThreadgroup == 8)
        #expect(Qwen35QMVGroupMapping.coop.rowsPerThreadgroup == 4)
        #expect(Qwen35QMVGroupMapping.staged.rowsPerThreadgroup == 8)
        #expect(Qwen35QMVGroupMapping.split.simdgroupsPerThreadgroup == 2)
        #expect(Qwen35QMVGroupMapping.coop.simdgroupsPerThreadgroup == 2)
        #expect(Qwen35QMVGroupMapping.staged.simdgroupsPerThreadgroup == 4)

        // Only the four `G == 2` widths move, and the source compiles exactly
        // those and nothing else.
        #expect(e217Widths == [6, 7, 8, 9])
        for mapping in [Qwen35QMVGroupMapping.coop, .staged] {
            for table in [true, false] {
                let source = qwen35E217QMVSource(table: table, mapping: mapping)
                for m in Qwen35CustomQMV.widths {
                    let present = source.contains("case \(m):")
                    #expect(
                        present == e217Widths.contains(m),
                        "\(mapping) table=\(table) width \(m)")
                }
            }
        }

        // Every mapping covers every output row of every cell exactly once,
        // over exactly one thread per (output row, token group, lane).
        for mapping in Qwen35QMVGroupMapping.allCases {
            for cell in e217AllCells {
                let grid = Qwen35CustomQMV.launchGrid(
                    n: cell.n, groups: 2, mapping: mapping)
                let group = Qwen35CustomQMV.launchThreadgroup(mapping: mapping)
                #expect(group.0 == 32, "\(mapping) simd width")
                #expect(grid.0 % group.0 == 0, "\(mapping) x")
                #expect(grid.1 % group.1 == 0, "\(mapping) y")
                let threadgroups = (grid.0 / group.0) * (grid.1 / group.1)
                #expect(
                    threadgroups * mapping.rowsPerThreadgroup
                        == cell.n * (mapping == .split ? 2 : 1),
                    "\(mapping) \(cell.name) row cover")
                // A `split` threadgroup owns one token column group, so it takes
                // two of them to cover a row; `coop` and `staged` own both.
                let threads = grid.0 * grid.1 * grid.2
                #expect(
                    threads == threadgroups * group.0 * group.1,
                    "\(mapping) \(cell.name) thread count")
            }
        }

        // A width that is not two-pass can never leave the shipped mapping,
        // whatever the environment selected.
        for groups in [1, 3] {
            #expect(Qwen35CustomQMV.mappingForLaunch(groups: groups) == .split)
        }
        #expect(
            Qwen35CustomQMV.mappingForLaunch(groups: 2)
                == Qwen35CustomQMV.groupMapping)

        // The staged source declares the tile; the coop source does not.
        for table in [true, false] {
            #expect(
                qwen35E217QMVSource(table: table, mapping: .staged)
                    .contains("threadgroup ushort qmv_tile[1024]"))
            #expect(
                !qwen35E217QMVSource(table: table, mapping: .coop)
                    .contains("qmv_tile"))
        }
        // 8 rows x 512 columns of 4-bit weights, as 16-bit packed words.
        #expect(1024 * MemoryLayout<UInt16>.size == 2048)
    }

    // MARK: - numerical gate

    @Test(.enabled(if: E217StagedMappingTests.gateEnabled))
    func numericalGate() throws {
        let shipped = E217Pipeline(mapping: .split, label: "split")
        let candidates = [
            E217Pipeline(mapping: .coop, label: "coop"),
            E217Pipeline(mapping: .staged, label: "staged"),
        ]

        // Positive controls: one per mapping, each perturbing the accumulation
        // that mapping really executes by a single bfloat16 ULP of relative
        // scale. `coop` runs the shipped `qwen_e120_qmv_wide` body; `staged`
        // runs `qwen_e217_consume_block`. The comparison must report differences
        // for both, otherwise a pass proves nothing.
        // Both targets are single lines, so neither substitution depends on
        // how the surrounding Swift literal is indented.
        let coopAccumulate =
            "acc[r] += scale_local[r] * partial[r] + sums * bias_local[r];"
        let coopPerturbed =
            "acc[r] += scale_local[r] * partial[r] * 1.0000305f "
            + "+ sums * bias_local[r];"
        let stagedAccumulate =
            "scale_local[r] * partial[r][m] + sums[m] * bias_local[r];"
        let stagedPerturbed =
            "scale_local[r] * partial[r][m] * 1.0000305f "
            + "+ sums[m] * bias_local[r];"
        // A substitution that silently matched nothing would make the control
        // identical to the candidate and the control would prove nothing.
        #expect(qwen35E217Header.contains(coopAccumulate))
        #expect(qwen35E217Header.contains(stagedAccumulate))
        let coopControlHeader = qwen35E217Header.replacingOccurrences(
            of: coopAccumulate, with: coopPerturbed)
        let stagedControlHeader = qwen35E217Header.replacingOccurrences(
            of: stagedAccumulate, with: stagedPerturbed)
        #expect(coopControlHeader != qwen35E217Header)
        #expect(stagedControlHeader != qwen35E217Header)

        let controls = [
            E217Pipeline(
                mapping: .coop, label: "coop_ctl",
                headerTransform: { _ in coopControlHeader }),
            E217Pipeline(
                mapping: .staged, label: "staged_ctl",
                headerTransform: { _ in stagedControlHeader }),
        ]

        var rows: [[String: Any]] = []
        var controlRows: [[String: Any]] = []
        var worstUlp = 0
        var totalDiffering = 0
        var totalElements = 0

        for cell in e217Cells {
            autoreleasepool {
                let (w, scales, biases) = e217PackedCell(
                    k: cell.k, n: cell.n, seed: 0xE217)
                for m in e217Widths {
                    autoreleasepool {
                        let x = e217Activations(
                            m: m, k: cell.k, seed: UInt64(0xE217_0000 + m))
                        let xsums = Qwen35CustomQMV.xsumsTable(x)
                        eval(xsums)
                        for useTable in [true, false] {
                            let reference = shipped.call(
                                x: x, w: w, scales: scales, biases: biases,
                                xsums: xsums, m: m, n: cell.n,
                                useTable: useTable)
                            eval(reference)
                            let rv = reference.asType(.float32)
                                .asArray(Float.self)
                            for pipeline in candidates {
                                let y = pipeline.call(
                                    x: x, w: w, scales: scales, biases: biases,
                                    xsums: xsums, m: m, n: cell.n,
                                    useTable: useTable)
                                eval(y)
                                let yv = y.asType(.float32).asArray(Float.self)
                                let diff = e217Compare(rv, yv)
                                worstUlp = max(worstUlp, diff.maxUlp)
                                totalDiffering += diff.count
                                totalElements += m * cell.n
                                rows.append([
                                    "cell": cell.name, "k": cell.k, "n": cell.n,
                                    "m": m, "use_table": useTable,
                                    "mapping": pipeline.mapping.rawValue,
                                    "invocations_per_round": cell.invocations,
                                    "elements": m * cell.n,
                                    "differing": diff.count,
                                    "max_ulp": diff.maxUlp,
                                    "max_abs_diff": diff.maxAbsDiff,
                                    "non_finite": diff.nanOrInf,
                                    "first_differing_index":
                                        diff.firstDifferingIndex ?? -1,
                                    "hexfloat_split": e217RowSamples(
                                        rv, m: m, n: cell.n, columns: 1),
                                    "hexfloat_candidate": e217RowSamples(
                                        yv, m: m, n: cell.n, columns: 1),
                                ])
                                #expect(
                                    diff.count == 0,
                                    """
                                    \(pipeline.label) \(cell.name) m=\(m) \
                                    table=\(useTable): \(diff.count) differing \
                                    outputs, max_ulp=\(diff.maxUlp), \
                                    max_abs_diff=\(diff.maxAbsDiff)
                                    """)
                                #expect(diff.nanOrInf == 0)
                            }

                            if cell.name == "gdn.out_proj" {
                                for pipeline in controls {
                                    let bad = pipeline.call(
                                        x: x, w: w, scales: scales,
                                        biases: biases, xsums: xsums, m: m,
                                        n: cell.n, useTable: useTable)
                                    eval(bad)
                                    let diff = e217Compare(
                                        rv,
                                        bad.asType(.float32)
                                            .asArray(Float.self))
                                    controlRows.append([
                                        "cell": cell.name, "m": m,
                                        "use_table": useTable,
                                        "mapping": pipeline.label,
                                        "differing": diff.count,
                                        "max_ulp": diff.maxUlp,
                                        "max_abs_diff": diff.maxAbsDiff,
                                    ])
                                    #expect(
                                        diff.count > 0,
                                        """
                                        positive control \(pipeline.label) did \
                                        not trip at m=\(m) table=\(useTable)
                                        """)
                                }
                            }
                        }
                    }
                }
            }
        }

        try e217Write(
            [
                "harness": "local",
                "experiment": "e217",
                "section": "numerical_gate",
                "cool_gate_passed_real_gate": false,
                "gate_qualified_for_timing": false,
                "official_or_ranked_score": false,
                "widths": e217Widths,
                "tolerance": "bit-exact: zero differing outputs, zero ULP",
                "reference_mapping": "split",
                "candidate_mappings": candidates.map(\.mapping.rawValue),
                "active_plan": qwen35QMVWidthPlanWitness,
                "qmv_arm": Qwen35CustomQMV.arm.rawValue,
                "worst_max_ulp": worstUlp,
                "total_differing": totalDiffering,
                "total_elements": totalElements,
                "comparisons": rows,
                "positive_control": controlRows,
            ], to: "MLXFAST_E217_GATE_OUT")
    }
}
