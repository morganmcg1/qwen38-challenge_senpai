import Foundation
import MLX
import MLXRandom
import Testing

@testable import MLXLLM

// E222 stage 1 -- does sharing DEQUANTISED values across the two token groups
// of a G >= 2 staged QMV entry pay for its geometry?
//
// FINDING 573 measured that the second token group of a G = 2 entry re-pays the
// whole `b` term (the packed load, the nibble extraction and the nibble-to-float
// conversion), worth 11.614 ms/round pooled if the consuming group paid nothing.
// E216 deduped the FETCH and lost 1.323 pooled; E217 staged raw BYTES and lost
// 5.866 pooled. Neither shared dequantised VALUES.
//
// The only geometry that shares a value with zero staging traffic is to give one
// thread both token groups: the nibble becomes a float once and feeds two fma
// chains from a register. That is what the shipped `singlePass` variant already
// does at NA = m -- and the shipped plan selects it ONLY at m = 6 for cells
// other than `mlp.down`, because at m = 7 and m = 8 an all-cells switch cost
// 23.62 ms/round. The recorded reason is register/occupancy pressure.
//
// E222 stage 0 measured that wall with real AGX register counts instead of
// inferring it. Two facts decide this experiment:
//
//   * The register wall is 96 on the local `g16s` generation and 126 on the
//     ranked `g17s` M5. `rows_per_simd = 4` single-pass spills on BOTH at
//     m = 8 and m = 9, which is the recorded failure.
//   * Halving `rows_per_simd` to 2 and doubling the simdgroups to 4 keeps the
//     threadgroup's 8-row span -- so no E216 coalescing loss -- and brings
//     m = 9 single-pass to 90 (g16s) / 95 (g17s) registers with no spill.
//
// A second stage-0 fact shapes the candidate: `vec<float, NA>` pads to the next
// admissible width, so a single `vec<float, 9>` accumulator costs far more
// registers than a `vec<float, 5>` plus a `vec<float, 4>` pair. The fused
// template therefore keeps the two token groups in SEPARATE accumulators. At
// m = 9, rows = 2 that is 90 registers against 114 for the unsplit form.
//
// This file times those geometries against the shipped dispatch on the E219
// chain-slope instrument. It reuses E219's replica rings, palindromic block
// order, calibration and temperature record unchanged.
//
// Research instrument. `Tests/` is never packaged into a submission.
// harness=local-microbench. These are mechanism prices for one dispatch, never
// whole-leg or ranked numbers (RULE 79). No thermal gate, no score.

/// Live-path certification needle (RULE 384/387): a dedicated >= 16-byte
/// literal on the executed path of this probe, not on any failure path.
let e222LivePathNeedle = "E222-QMV-DEQUANT-VALUE-SHARING-LIVE-PATH-2026-08-25"

// MARK: - the shared-value template

/// One thread owns BOTH token groups of a `G = 2` entry for its weight rows.
///
/// Every arithmetic property of `qwen_e120_qmv_wide` is preserved, so each
/// output's result is bit-identical to the shipped dispatch:
///
///   * the `k` blocks, the per-lane `values_per_thread = 16` slice and the four
///     `i` sub-steps run in the shipped order;
///   * each output's fma chain is still `a0*n0 + a1*n1 + a2*n2 + a3*n3`
///     accumulated into `partial`, then closed by
///     `acc += scale * partial + sums * bias` and one `simd_sum`;
///   * `sums`, the activation offsets and the `y` offsets use the same absolute
///     column indices the two staged passes would use at `first_m = 0` and
///     `first_m = NA0`.
///
/// The three knobs are pure scheduling and cannot move a value:
///
///   `LAZY`     load the packed word inside the `i` loop instead of prefetching
///              all four words per row before it. Trades four live registers
///              per row for a load on the critical path.
///   `LATE_SB`  read `scales`/`biases` after the `i` loop instead of before it.
///   `SEQ`      run the two token groups as two phases of each `i` step, so
///              only one group's activation vectors are live at a time.
private let e222FusedTemplate: String = {
    // `research/e222_fused.msl` is the single source of truth for this
    // geometry. `research/e222_geometry.py` compiles the same text for the
    // register and spill screen, so a screened geometry and a timed geometry
    // cannot diverge.
    let root = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()   // Tests/MLXFastTests
        .deletingLastPathComponent()   // Tests
        .deletingLastPathComponent()   // repository root
    let path = root.appendingPathComponent("research/e222_fused.msl")
    guard let text = try? String(contentsOf: path, encoding: .utf8),
        text.contains("inline void qwen_e222_fused(")
    else {
        preconditionFailure("[e222] missing template at \(path.path)")
    }
    return text
}()

/// The shipped wide body with `rows_per_simd` promoted to a template argument,
/// derived textually from `qwen35E120QMVHeader` at run time.
///
/// The substitution is textual and the `R = 4` instance is the shipped source
/// character for character, so this ladder cannot drift from the scored kernel.
/// Only the wide template is copied; the `qwen_e120_qmv_m` dispatcher below it
/// still refers to the original symbol and is supplied unmodified alongside.
private let e222RowsTemplate: String = {
    let live = qwen35E120QMVHeader
    guard let cut = live.range(of: "template <int M, int IPG, bool USE_TABLE>")
    else { preconditionFailure("[e222] the shipped header lost its dispatcher") }
    var rows = String(live[live.startIndex ..< cut.lowerBound])
    rows = rows.replacingOccurrences(
        of: "template <int NA, bool USE_TABLE>",
        with: "template <int NA, int R, bool USE_TABLE>")
    rows = rows.replacingOccurrences(
        of: "qwen_e120_qmv_wide(", with: "qwen_e222_rows(")
    rows = rows.replacingOccurrences(
        of: "constexpr int rows_per_simd = 4;",
        with: "constexpr int rows_per_simd = R;")
    precondition(
        rows.contains("inline void qwen_e222_rows(")
            && rows.contains("constexpr int rows_per_simd = R;")
            && !rows.contains("qwen_e120_qmv_wide"),
        "[e222] the shipped body no longer matches the rows rewrite")
    return rows
}()

/// The shipped header verbatim (so the baseline arms launch the real
/// `qwen_e120_qmv_m` dispatcher, tail groups included), plus the two templates
/// this probe adds.
private let e222Header =
    qwen35E120QMVHeader + "\n" + e222RowsTemplate + "\n" + e222FusedTemplate

// MARK: - arms

/// The staged plan's `G = 2` widths and the token-group split each one uses.
/// `(m, IPG)` comes from `Qwen35QMVKernelVariant.staged.pairs`; the split is
/// `(IPG, m - IPG)`.
private func e222Split(_ m: Int) -> (na0: Int, na1: Int) {
    let ipg = Qwen35CustomQMV.inputsPerGroup(m, variant: .staged)
    return (na0: ipg, na1: m - ipg)
}

private enum E222Body {
    /// The shipped dispatcher at `(M, IPG)`, tail group included.
    case shipped(ipg: Int)
    /// The shipped wide body with `rows_per_simd = R`, one group of `NA = M`.
    case rows
    /// One thread, both token groups, separate accumulators.
    case fused(lazyW: Bool, lateSB: Bool, seq: Bool)
}

private struct E222Arm {
    var label: String
    var body: E222Body
    var rows: Int
    var simdgroups: Int
    /// What this arm is in the campaign record.
    var role: String

    var rowsPerThreadgroup: Int { rows * simdgroups }
}

private func e222Arm(_ name: String, m: Int) -> E222Arm? {
    let ipg = Qwen35CustomQMV.inputsPerGroup(m, variant: .staged)
    switch name {
    case "staged_r4":
        return E222Arm(
            label: name, body: .shipped(ipg: ipg), rows: 4, simdgroups: 2,
            role: "shipped staged plan, G = ceil(m/IPG)")
    case "single_r4":
        // One pass at NA = m, rows = 4: the register-wall control. It is the
        // shipped `singlePass` variant only for m <= 8; at m = 9 that variant
        // is IPG = 3, so this arm is the unshipped full single pass instead.
        return E222Arm(
            label: name, body: .shipped(ipg: m), rows: 4, simdgroups: 2,
            role: m <= 8
                ? "shipped singlePass variant, the recorded m>=7 loser"
                : "one pass at NA=m, rows=4; not a shipped form at m=9")
    case "single_r2":
        return E222Arm(
            label: name, body: .rows, rows: 2, simdgroups: 4,
            role: "single pass at rows=2, ONE vec<float,m> accumulator")
    case "fused_r4":
        // PRIMARY. rows = 4 keeps the shipped activation amortization exactly:
        // 36*NA0/4 + 36*NA1/4 issued activation bytes per output row per
        // k-block is identical to 36*(NA0+NA1)/4, so the arm's whole issued-
        // byte delta against the shipped staged pair is the deduped weight
        // word, -12 B per output row per k-block. It spills on g16s, so its
        // local time is a ranked-only lower bound (RULE 407).
        return E222Arm(
            label: name,
            body: .fused(lazyW: false, lateSB: false, seq: false), rows: 4,
            simdgroups: 2,
            role: "primary: fused at the shipped rows and schedule")
    case "fused_r4_seq":
        return E222Arm(
            label: name,
            body: .fused(lazyW: false, lateSB: false, seq: true), rows: 4,
            simdgroups: 2,
            role: "primary with group-sequenced activation liveness")
    case "fused_r4_lazyw":
        return E222Arm(
            label: name,
            body: .fused(lazyW: true, lateSB: false, seq: false), rows: 4,
            simdgroups: 2, role: "fused rows=4 with the lazy packed load")
    case "fused_r2":
        return E222Arm(
            label: name,
            body: .fused(lazyW: true, lateSB: false, seq: false), rows: 2,
            simdgroups: 4,
            role: "rows=2 fused: clean on both generations, pays +66% issued "
                + "activation bytes")
    case "fused_r2_both":
        return E222Arm(
            label: name,
            body: .fused(lazyW: true, lateSB: true, seq: false), rows: 2,
            simdgroups: 4, role: "rows=2 fused, register-minimal")
    default:
        return nil
    }
}

// MARK: - the instrument kernel

private struct E222Pipeline {
    let arm: E222Arm
    let m: Int
    let stride: Int
    /// Threadgroups along grid x: one per weight-group pass.
    let xGroups: Int
    let kernel: Qwen35CachedKernel

    init(arm: E222Arm, m: Int, stride: Int) {
        self.arm = arm
        self.m = m
        self.stride = stride
        let split = e222Split(m)
        let rows = arm.rows
        let span = arm.rowsPerThreadgroup
        let call: String
        switch arm.body {
        case .shipped(let ipg):
            xGroups = (m + ipg - 1) / ipg
            call = """
                qwen_e120_qmv_m<\(m), \(ipg), USE_TABLE>(
                            w, scales, biases, x, xsums, y,
                            qmv_k, qmv_n, \(stride),
                            int(qmv_tid.x), qmv_out_row, qmv_lid);
                """
        case .rows:
            xGroups = 1
            call = """
                qwen_e222_rows<\(m), \(rows), USE_TABLE>(
                            w, scales, biases, x, xsums, y,
                            qmv_k, qmv_n, \(stride),
                            int(qmv_tid.x) * \(m), qmv_out_row, qmv_lid);
                """
        case .fused(let lazyW, let lateSB, let seq):
            xGroups = 1
            let args =
                "\(split.na0), \(split.na1), \(rows), \(lazyW), \(lateSB), \(seq)"
            call = """
                qwen_e222_fused<\(args), USE_TABLE, false>(
                            w, scales, biases, x, xsums, y, nullptr,
                            qmv_k, qmv_n, \(stride),
                            qmv_out_row, qmv_lid);
                """
        }
        let source = """
                const int qmv_k = x_shape[x_ndim - 1];
                const int qmv_n = w_shape[0];
                const uint3 qmv_tid = threadgroup_position_in_grid;
                const uint qmv_lid = thread_index_in_simdgroup;
                const uint qmv_sgid = simdgroup_index_in_threadgroup;
                const int qmv_out_row =
                    int(qmv_tid.y) * \(span) + int(qmv_sgid) * \(rows);
                \(call)
            """
        self.kernel = Qwen35CachedKernel(
            name: "e222_\(arm.label)_m\(m)_s\(stride)",
            inputNames: ["w", "scales", "biases", "x", "xsums"],
            outputNames: ["y"],
            source: source,
            header: e222Header)
    }

    func call(x: MLXArray, xsums: MLXArray, set: E219WeightSet) -> MLXArray {
        let launch = Qwen35KernelLaunch(
            grid: (
                xGroups * 32, (set.n / arm.rowsPerThreadgroup) * arm.simdgroups,
                1
            ),
            threadGroup: (32, arm.simdgroups, 1),
            outputShape: [1, Int32(m), Int32(set.n)],
            outputDType: .bfloat16,
            useTable: true)
        return kernel([set.w, set.scales, set.biases, x, xsums], launch)
    }
}

private struct E222PipelineCache {
    private var pipelines: [String: E222Pipeline] = [:]

    mutating func get(arm: E222Arm, m: Int, stride: Int) -> E222Pipeline {
        let key = "\(arm.label)/\(m)/\(stride)"
        if let hit = pipelines[key] { return hit }
        let made = E222Pipeline(arm: arm, m: m, stride: stride)
        pipelines[key] = made
        return made
    }
}

// MARK: - reporting

private func e222PhaseEnabled(_ name: String) -> Bool {
    guard let raw = ProcessInfo.processInfo.environment["MLX_E222_PHASE"] else {
        return false
    }
    return raw == "all"
        || raw.split(separator: ",").map(String.init).contains(name)
}

private func e222Names(_ key: String, _ fallback: [String]) -> [String] {
    guard let raw = ProcessInfo.processInfo.environment[key], !raw.isEmpty
    else { return fallback }
    return raw.split(separator: ",").map(String.init)
}

private func e222Write(
    _ phase: String, _ body: [String: Any], session: E219Session?
) throws {
    var report: [String: Any] = [
        "probe": "e222-dequant-value-sharing",
        "phase": phase,
        "probe_needle": e222LivePathNeedle,
        "harness": "local-microbench",
        "cool_gate_passed_real_gate": false,
        "gate_qualified_for_timing": false,
        "official_or_ranked_score": false,
        "whole_leg_or_ranked_number": false,
        "active_plan_witness": qwen35QMVWidthPlanWitness,
        "qmv_arm": Qwen35CustomQMV.arm.rawValue,
        "config_cache_enabled": Qwen35KernelConfigCache.enabled,
        "shipped_rows_per_simd": 4,
        "shipped_threadgroup": [32, 2, 1],
    ]
    for (key, value) in body { report[key] = value }
    if let session {
        report["blocks"] = session.blocks
        report["chains"] = session.chains
        report["warmup"] = session.warmup
        report["pretouch_dispatches"] = session.touch
        report["minimum_reps"] = session.minimumReps
        report["target_microseconds"] = session.targetMicroseconds
        report["gpu_temperature_c"] = session.temperatures.mapValues {
            $0 ?? -1.0
        }
        report["samples"] = session.samples
    }
    let json = try JSONSerialization.data(
        withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
    if let path = ProcessInfo.processInfo.environment["MLX_E222_OUT"],
        !path.isEmpty
    {
        try json.write(to: URL(fileURLWithPath: path))
        print("[e222] wrote \(path) (\(json.count) bytes)")
    } else {
        print(String(decoding: json, as: UTF8.self))
    }
}

// MARK: - Section 0: every arm must be bit-exact

@Suite(.serialized)
struct E222ExactnessTests {
    /// Every arm must reproduce `Qwen35CustomQMV.matmulWithTable` bit for bit
    /// at every timed `(cell, width)`. A geometry change may move which thread
    /// computes a row; it may never move a row's own reduction.
    ///
    /// Write coverage is proved with the arm's own output: the reference is
    /// dense and non-zero, so an output row no arm thread wrote could only
    /// match by accident. The check therefore also asserts the reference has no
    /// exact zero, which is what makes "no differing element" a coverage proof
    /// (RULE 403).
    @Test(.enabled(if: e222PhaseEnabled("exact")))
    func armsAreBitExact() throws {
        let widths = e219IntList("MLX_E222_WIDTHS", [6, 7, 8, 9])
        let armNames = e222Names(
            "MLX_E222_ARMS",
            [
                "staged_r4", "single_r4", "single_r2", "fused_r4",
                "fused_r4_seq", "fused_r4_lazyw", "fused_r2", "fused_r2_both",
            ])
        var cache = E222PipelineCache()
        var rows: [[String: Any]] = []

        for cell in e219ScoredCells {
            try autoreleasepool {
                let set = e219RandomSet(k: cell.k, n: cell.n, seed: 0xE222)
                for m in widths {
                    try autoreleasepool {
                        let x = e219Activations(
                            m: m, k: cell.k, seed: UInt64(0xE222_0000 + m))
                        let xsums = Qwen35CustomQMV.xsumsTable(x)
                        eval(xsums)
                        guard
                            let shipped = Qwen35CustomQMV.matmulWithTable(
                                x, set.w, scales: set.scales,
                                biases: set.biases, xsums: xsums,
                                groupSize: 64, bits: 4, mode: .affine)
                        else {
                            Issue.record(
                                "shipped path declined \(cell.name) m=\(m)")
                            return
                        }
                        eval(shipped)
                        let reference = shipped.asType(.float32)
                            .asArray(Float.self)
                        let zeros = reference.filter { $0 == 0 }.count
                        #expect(
                            zeros == 0,
                            """
                            \(cell.name) m=\(m): the reference has \(zeros) \
                            exact zeros, so a match cannot prove write coverage
                            """)
                        for name in armNames {
                            guard let arm = e222Arm(name, m: m) else {
                                Issue.record("unknown E222 arm \(name)")
                                continue
                            }
                            guard cell.n % arm.rowsPerThreadgroup == 0 else {
                                continue
                            }
                            let pipeline = cache.get(
                                arm: arm, m: m,
                                stride: Qwen35CustomQMV.sumsStride(m))
                            let mine = pipeline.call(
                                x: x, xsums: xsums, set: set)
                            eval(mine)
                            let b = mine.asType(.float32).asArray(Float.self)
                            var differing = 0
                            var nonFinite = 0
                            for i in 0 ..< min(reference.count, b.count) {
                                if reference[i].bitPattern != b[i].bitPattern {
                                    differing += 1
                                }
                                if !b[i].isFinite { nonFinite += 1 }
                            }
                            rows.append([
                                "cell": cell.name, "k": cell.k, "n": cell.n,
                                "m": m, "arm": name, "rows_per_simd": arm.rows,
                                "simdgroups": arm.simdgroups,
                                "rows_per_threadgroup":
                                    arm.rowsPerThreadgroup,
                                "x_groups": pipeline.xGroups,
                                "elements": reference.count,
                                "reference_zeros": zeros,
                                "differing": differing,
                                "non_finite": nonFinite,
                            ])
                            #expect(
                                differing == 0,
                                """
                                \(cell.name) m=\(m) \(name): \(differing) of \
                                \(reference.count) outputs differ from the \
                                shipped dispatch
                                """)
                            #expect(nonFinite == 0)
                        }
                    }
                }
            }
        }

        #expect(!rows.isEmpty)
        try e222Write(
            "exact",
            [
                "deliverable":
                    "every E222 arm is bit-exact against "
                    + "Qwen35CustomQMV.matmulWithTable, with write coverage",
                "comparisons": rows,
            ], session: nil)
    }

    /// RULE 408 write coverage, for every fused instantiation.
    ///
    /// The exactness check above is a coverage proof only through the RULE 403
    /// argument that a dense non-zero reference cannot be matched at an output
    /// nobody wrote. That argument is sound but indirect, and it cannot see a
    /// DOUBLE write at all: two threads that write the same correct value to
    /// one output leave no trace in the values.
    ///
    /// This check reads the written key SET instead. `E222Census` reuses the
    /// fused body itself with `CENSUS = true`, which swaps the value store for
    /// one atomic increment at the same key expression, so the census cannot
    /// drift from the address arithmetic it is auditing. The gate then asserts
    /// four properties of the key set over `0 ..< m * n`: the count, the
    /// minimum, the maximum, and that every key was written exactly once.
    ///
    /// Two positive controls prove the gate can fail. `shortGrid` launches half
    /// the row slices, which must leave keys unwritten. `doubledGrid` launches
    /// the y extent twice, which must write every key twice. A gate that passes
    /// both of those is not measuring anything.
    @Test(.enabled(if: e222PhaseEnabled("exact")))
    func fusedWriteCoverage() throws {
        let widths = e219IntList("MLX_E222_WIDTHS", [6, 7, 8, 9])
        let armNames = e222Names(
            "MLX_E222_ARMS",
            [
                "fused_r4", "fused_r4_seq", "fused_r4_lazyw", "fused_r2",
                "fused_r2_both",
            ])
        var rows: [[String: Any]] = []

        for cell in e219ScoredCells {
            for m in widths {
                for name in armNames {
                    guard let arm = e222Arm(name, m: m),
                        case .fused = arm.body,
                        cell.n % arm.rowsPerThreadgroup == 0
                    else { continue }
                    try autoreleasepool {
                        let census = E222Census(arm: arm, m: m, cell: cell)
                        let good = try census.keySet(.exact)
                        let short = try census.keySet(.shortGrid)
                        let doubled = try census.keySet(.doubledGrid)
                        let expected = m * cell.n

                        rows.append([
                            "cell": cell.name, "m": m, "arm": name,
                            "rows_per_simd": arm.rows,
                            "keys_expected": expected,
                            "keys_written": good.written,
                            "min_key": good.minKey, "max_key": good.maxKey,
                            "min_writes": good.minWrites,
                            "max_writes": good.maxWrites,
                            "control_short_grid_keys_written": short.written,
                            "control_doubled_grid_max_writes":
                                doubled.maxWrites,
                        ])

                        let where_ = "\(cell.name) m=\(m) \(name)"
                        #expect(
                            good.written == expected,
                            "\(where_): wrote \(good.written) of \(expected) keys"
                        )
                        #expect(good.minKey == 0, "\(where_): min key")
                        #expect(
                            good.maxKey == expected - 1, "\(where_): max key")
                        #expect(
                            good.minWrites == 1 && good.maxWrites == 1,
                            """
                            \(where_): write counts span \(good.minWrites) to \
                            \(good.maxWrites), so some output is written twice \
                            or not at all
                            """)

                        #expect(
                            short.written < expected,
                            """
                            \(where_): the short-grid positive control wrote \
                            every key, so this gate cannot detect a gap
                            """)
                        #expect(
                            doubled.maxWrites == 2,
                            """
                            \(where_): the doubled-grid positive control did \
                            not double any write, so this gate cannot detect a \
                            duplicate
                            """)
                    }
                }
            }
        }

        #expect(!rows.isEmpty)
        try e222Write(
            "write-coverage",
            [
                "deliverable":
                    "RULE 408: the written key set of every fused "
                    + "instantiation, checked against 0 ..< m * n with two "
                    + "positive controls that trip",
                "census": rows,
            ], session: nil)
    }
}

// MARK: - RULE 408 write census

/// How the census kernel is launched. `exact` is the arm's real geometry; the
/// other two are the positive controls that must break the gate.
private enum E222CensusGrid {
    case exact
    /// Half the row slices in y, so the upper half of the output is unwritten.
    case shortGrid
    /// Two threadgroups along x instead of one. The fused body ignores
    /// `qmv_tid.x`, so both compute the same `qmv_out_row` and every key is
    /// written exactly twice. Doubling y instead would run `qmv_out_row` past
    /// `n` and write out of bounds, which is a different defect.
    case doubledGrid
}

private struct E222CensusResult {
    var written: Int
    var minKey: Int
    var maxKey: Int
    var minWrites: Int
    var maxWrites: Int
}

/// The fused body compiled with `CENSUS = true`: same key arithmetic, atomic
/// increment instead of a value store.
private struct E222Census {
    let arm: E222Arm
    let m: Int
    let cell: E219Cell
    let kernel: MLXFast.MLXFastKernel

    init(arm: E222Arm, m: Int, cell: E219Cell) {
        self.arm = arm
        self.m = m
        self.cell = cell
        let split = e222Split(m)
        guard case .fused(let lazyW, let lateSB, let seq) = arm.body else {
            preconditionFailure("E222Census is only defined for fused arms")
        }
        let args =
            "\(split.na0), \(split.na1), \(arm.rows), \(lazyW), \(lateSB), "
            + "\(seq), true, true"
        // `y` is the atomic census buffer here, so the value-store pointer is
        // null and `CENSUS = true` guarantees it is never dereferenced.
        let source = """
                const int qmv_k = x_shape[x_ndim - 1];
                const int qmv_n = w_shape[0];
                const uint3 qmv_tid = threadgroup_position_in_grid;
                const uint qmv_lid = thread_index_in_simdgroup;
                const uint qmv_sgid = simdgroup_index_in_threadgroup;
                const int qmv_out_row =
                    int(qmv_tid.y) * \(arm.rowsPerThreadgroup)
                    + int(qmv_sgid) * \(arm.rows);
                qwen_e222_fused<\(args)>(
                    w, scales, biases, x, xsums, nullptr, y,
                    qmv_k, qmv_n, \(Qwen35CustomQMV.sumsStride(m)),
                    qmv_out_row, qmv_lid);
            """
        self.kernel = MLXFast.metalKernel(
            name: "e222_census_\(arm.label)_m\(m)",
            inputNames: ["w", "scales", "biases", "x", "xsums"],
            outputNames: ["y"],
            source: source,
            header: e222Header,
            atomicOutputs: true)
    }

    func keySet(_ grid: E222CensusGrid) throws -> E222CensusResult {
        let set = e219RandomSet(k: cell.k, n: cell.n, seed: 0xE222)
        let x = e219Activations(m: m, k: cell.k, seed: 0xE222_C0DE)
        let xsums = Qwen35CustomQMV.xsumsTable(x)
        eval(xsums)

        let slices = cell.n / arm.rowsPerThreadgroup
        let yGroups = grid == .shortGrid ? slices / 2 : slices
        let xThreads = grid == .doubledGrid ? 64 : 32

        let counts = kernel(
            [set.w, set.scales, set.biases, x, xsums],
            grid: (xThreads, yGroups * arm.simdgroups, 1),
            threadGroup: (32, arm.simdgroups, 1),
            outputShapes: [[m * cell.n]],
            outputDTypes: [.uint32],
            initValue: 0)[0]
        eval(counts)
        let written = counts.asArray(UInt32.self)

        var result = E222CensusResult(
            written: 0, minKey: Int.max, maxKey: -1,
            minWrites: Int.max, maxWrites: 0)
        for (key, count) in written.enumerated() where count > 0 {
            result.written += 1
            result.minKey = min(result.minKey, key)
            result.maxKey = max(result.maxKey, key)
            result.minWrites = min(result.minWrites, Int(count))
            result.maxWrites = max(result.maxWrites, Int(count))
        }
        if result.written == 0 {
            result.minKey = -1
            result.minWrites = 0
        }
        return result
    }
}

// MARK: - Section 1: the timed screen

@Suite(.serialized)
struct E222ValueSharingTests {
    private static var replicaTarget: Int {
        e219Int("MLX_E222_REPLICA_TARGET_MB", 192) * 1_048_576
    }
    private static var replicaCap: Int {
        e219Int("MLX_E222_REPLICA_CAP_MB", 768) * 1_048_576
    }

    private func sweep(
        phase: String, widths: [Int], armNames: [String], deliverable: String
    ) throws {
        var cache = E222PipelineCache()
        var session = E219Session(prefix: "MLX_E222")
        var units: [E219Unit] = []
        var ringFacts: [[String: Any]] = []
        let ringBox = E219RingBox()

        for cell in e219ScoredCells {
            let ring = E219Replicas(
                k: cell.k, n: cell.n, seed: 0xE222_5EED,
                targetBytes: Self.replicaTarget, hardCapBytes: Self.replicaCap)
            ringFacts.append([
                "cell": cell.name, "k": cell.k, "n": cell.n,
                "replicas": ring.sets.count,
                "replica_set_bytes": ring.totalBytes,
                "bytes_per_replica": ring.sets[0].bytes,
            ])
            ringBox.install(cell.name, ring)
        }

        for cell in e219ScoredCells {
            for m in widths {
                let stride = Qwen35CustomQMV.sumsStride(m)
                let split = e222Split(m)
                let x = e219Activations(
                    m: m, k: cell.k, seed: UInt64(0xE222_1000 + m))
                let xsums = Qwen35CustomQMV.xsumsTable(x)
                eval(xsums)
                // The shipped plan's own choice for this cell at this width, so
                // each arm is compared against the baseline it must actually
                // beat rather than against `staged` everywhere.
                let shippedVariant = qwen35QMVVariant(
                    m: m, cell: Qwen35QMVCell.identify(k: cell.k, n: cell.n))
                for name in armNames {
                    guard let arm = e222Arm(name, m: m) else {
                        Issue.record("unknown E222 arm \(name)")
                        continue
                    }
                    guard cell.n % arm.rowsPerThreadgroup == 0 else { continue }
                    let pipeline = cache.get(arm: arm, m: m, stride: stride)
                    let bytes = E219Bytes(
                        k: cell.k, n: cell.n, na: m, stride: stride)
                    let key = cell.name
                    var fields: [String: Any] = [
                        "cell": cell.name, "k": cell.k, "n": cell.n,
                        "m": m, "arm": name, "arm_role": arm.role,
                        "rows_per_simd": arm.rows,
                        "simdgroups": arm.simdgroups,
                        "rows_per_threadgroup": arm.rowsPerThreadgroup,
                        "x_groups": pipeline.xGroups,
                        "na0": split.na0, "na1": split.na1,
                        "staged_ipg": Qwen35CustomQMV.inputsPerGroup(
                            m, variant: .staged),
                        "shipped_variant": shippedVariant.rawValue,
                        "stride": stride, "cold": true, "dispatches": 1,
                        "invocations_per_round": cell.invocations,
                        "replicas": ringBox.count(key),
                        "replica_set_bytes": ringBox.bytes(key),
                    ]
                    for (bk, bv) in bytes.dictionary { fields[bk] = bv }
                    // Every arm covers all `m` columns once, so the device
                    // weight stream is `x_groups` copies of the cell.
                    fields["pass_stream_bytes"] =
                        (bytes.weight + bytes.scaleBias) * pipeline.xGroups
                    units.append(
                        E219Unit(
                            label: "\(cell.name)/m\(m)/\(name)",
                            fields: fields,
                            enqueue: {
                                let set = ringBox.next(key, cold: true)
                                return [
                                    pipeline.call(x: x, xsums: xsums, set: set)
                                ]
                            }))
                }
            }
        }

        session.run(units, settle: e219Settle())
        try e222Write(
            phase,
            [
                "deliverable": deliverable,
                "widths": widths,
                "arms": armNames,
                "replica_rings": ringFacts,
                "cells": e219ScoredCells.map {
                    [
                        "name": $0.name, "k": $0.k, "n": $0.n,
                        "invocations_per_round": $0.invocations,
                    ]
                },
            ], session: session)
    }

    /// The decisive screen: every `G = 2` width against the shipped plan, with
    /// the two candidate geometries and the fusion-saving calibration in one
    /// counterbalanced session.
    ///
    /// The arms answer three separate questions at once.
    ///
    /// `single_r4` at `m = 6` is the calibration. The shipped router already
    /// picks `singlePass` there for `mlp.gate_up` and `gdn.in_proj`, and both
    /// geometries are register-legal, so the `staged_r4` to `single_r4` delta
    /// at `m = 6` measures the real fusion saving with no register confound.
    /// The desk model prices the whole experiment off that saving, so this is
    /// the measurement that validates or refutes the model.
    ///
    /// `single_r4` at `m = 7, 8, 9` is the register-wall control: it is the
    /// fusion the shipped router cannot take, and it spills on both devices.
    ///
    /// `fused_r2` is the candidate. It is the only arm that is register-clean
    /// on both `g16s` (local) and `g17s` (ranked) at every one of these widths,
    /// which is what lets it reach `m = 9` and the 257 of 637 pooled rounds
    /// that route there. Its local time is therefore fully comparable.
    ///
    /// `fused_r4` is the geometry the advisor named primary. It spills on
    /// `g16s` at every width and on `g17s` above `m = 7`, so its local time is
    /// a one-sided lower bound only (RULE 407).
    @Test(.enabled(if: e222PhaseEnabled("screen")))
    func screen() throws {
        try sweep(
            phase: "screen",
            widths: e219IntList("MLX_E222_WIDTHS", [6, 7, 8, 9]),
            armNames: e222Names(
                "MLX_E222_ARMS",
                ["staged_r4", "single_r4", "fused_r2", "fused_r4"]),
            deliverable:
                "the fusion saving, measured where it is already legal, and "
                + "what each candidate geometry keeps of it at the widths the "
                + "shipped router cannot reach")
    }

    /// Whether the `SEQ` group-sequenced liveness variant buys enough registers
    /// to run `rows = 4` where the plain fused body spills. Stage 0 prices the
    /// register saving; this prices what the re-derived dequantisation costs.
    @Test(.enabled(if: e222PhaseEnabled("widths")))
    func widths() throws {
        try sweep(
            phase: "widths",
            widths: e219IntList("MLX_E222_WIDTHS", [8, 9]),
            armNames: e222Names(
                "MLX_E222_ARMS",
                ["staged_r4", "fused_r2", "fused_r4", "fused_r4_seq"]),
            deliverable:
                "whether group-sequenced activation liveness reaches rows = 4 "
                + "inside the register budget, and what its re-derived "
                + "dequantisation costs")
    }

    /// Accumulator-shape and load-schedule controls at the dominant width.
    /// `single_r2` isolates what the split `VF0`/`VF1` accumulator buys over
    /// one `vec<float, m>` at the same geometry; `fused_r2_both` isolates the
    /// late scale/bias read on top of the lazy packed load.
    @Test(.enabled(if: e222PhaseEnabled("shape")))
    func shape() throws {
        try sweep(
            phase: "shape",
            widths: e219IntList("MLX_E222_WIDTHS", [9]),
            armNames: e222Names(
                "MLX_E222_ARMS",
                [
                    "staged_r4", "fused_r2", "single_r2", "fused_r2_both",
                    "fused_r4_lazyw",
                ]),
            deliverable:
                "accumulator-shape and load-schedule controls: what the split "
                + "accumulator and the lazy packed load each buy")
    }
}
