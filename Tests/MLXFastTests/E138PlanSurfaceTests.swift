import Foundation
import MLX
import MLXLLM
import Testing

/// E138: the isolated `(IPG, RPS)` plan surface of the wide affine-4/group-64
/// QMV, measured at the shipped tight launch column count.
///
/// E137 proved that this kernel family carries the M=5 to M=6 round step and
/// that no single shape owns it. This suite asks the next question: is there a
/// bit-exact `(IPG, RPS)` plan that makes width 6 cheap?
///
/// Every cell is the SAME kernel body. `qwen_e120_qmv_wide` accumulates each
/// output element independently in lane `m` of `acc[r]`
/// (`Qwen35.swift:1454-1527`), so the `K` accumulation order of one output
/// element does not depend on `IPG` or on `RPS`: those two only decide which
/// threadgroup and which simdgroup computes it. A plan change is therefore
/// bit-exact by construction, and every cell here checks that against the
/// incumbent gate rather than assuming it.
///
/// The Metal body is taken from `Qwen35CustomQMV.generatedSource`, the live
/// shipped generator, and only the dispatch cases are replaced. `caseTemplate`
/// pins the replacement against that generator with both polarities, so a case
/// template that drifts from the shipped one fails the build instead of
/// measuring a kernel nobody ships.
///
/// Enable with `MLXFAST_RUN_E138_PLAN_SURFACE=1` and point
/// `MLXFAST_E138_PLAN_SURFACE_OUT` at the JSON destination.
@Suite
struct E138PlanSurfaceTests {
    private static var enabled: Bool {
        ProcessInfo.processInfo
            .environment["MLXFAST_RUN_E138_PLAN_SURFACE"] == "1"
    }

    /// The case template this suite substitutes into the live generator must
    /// be the generator's own. Both polarities are checked: the shipped plan
    /// must be found verbatim, and a plan with one changed field must not.
    @Test
    func caseTemplateMatchesTheShippedGenerator() throws {
        let live = Qwen35CustomQMV.generatedSource(table: true, tier: nil)
        for plan in Qwen35CustomQMV.widthPlan {
            let cell = E138Cell(m: plan.m, ipg: plan.ipg, rps: plan.rps)
            #expect(
                live.contains(e138CaseText(cell)),
                "the E138 case template has drifted at m=\(plan.m)")
            let wrong = E138Cell(m: plan.m, ipg: plan.ipg, rps: plan.rps + 1)
            #expect(
                !live.contains(e138CaseText(wrong)),
                "the E138 case check cannot fail at m=\(plan.m)")
        }
    }

    /// A plan is only legal when no group carries a one-row tail
    /// (`Qwen35.swift:1545`). `(6, 5)`, the cell the assignment names as the
    /// discriminator, is prohibited by that rule.
    @Test
    func oneRowTailPlansAreRejectedBeforeTheyReachMetal() {
        #expect(E138Cell(m: 6, ipg: 6, rps: 4).legal)
        #expect(E138Cell(m: 6, ipg: 4, rps: 4).legal)
        #expect(E138Cell(m: 6, ipg: 3, rps: 4).legal)
        #expect(!E138Cell(m: 6, ipg: 5, rps: 4).legal)
        #expect(!E138Cell(m: 5, ipg: 4, rps: 4).legal)
        #expect(!E138Cell(m: 9, ipg: 8, rps: 4).legal)

        // The stock cell is not a plan, so the tail rule does not apply to it.
        let stock = E138Cell(m: 6, ipg: 0, rps: 0)
        #expect(stock.isStock)
        #expect(stock.legal)
        #expect(stock.label == "6:stock")
        #expect(!E138Cell(m: 6, ipg: 6, rps: 4).isStock)
    }

    /// An unparsed `@grid` suffix would silently fall back to the session
    /// default and make both arms of the factorial the same measurement, so
    /// the parse and the launch decision are checked rather than assumed.
    @Test
    func perCellGridSuffixSelectsTheLaunchColumnCount() {
        let cells = e138ParseCells("6:3:4@wide,6:3:4@tight,6:6:4,6:stock@wide")
        #expect(cells.map(\.label) == [
            "6:3:4@wide", "6:3:4@tight", "6:6:4", "6:stock@wide",
        ])
        #expect(cells[0].grid == .wide)
        #expect(cells[1].grid == .tight)
        #expect(cells[2].grid == nil)
        #expect(cells[3].isStock)

        // wide launches `m` columns, tight launches `passes`, and an unsuffixed
        // cell follows the session default in both directions.
        #expect(!cells[0].tight(default: true))
        #expect(cells[1].tight(default: false))
        #expect(cells[2].tight(default: true))
        #expect(!cells[2].tight(default: false))

        // The kernel identity must not carry the grid, or the two arms would
        // compile two pipelines and confound the contrast they exist to make.
        #expect(cells[0].name == cells[1].name)
        #expect(cells[0].name == E138Cell(m: 6, ipg: 3, rps: 4).name)
    }

    @Test(.enabled(if: E138PlanSurfaceTests.enabled))
    func sweepPlanSurface() throws {
        let env = ProcessInfo.processInfo.environment
        let outPath = try #require(
            env["MLXFAST_E138_PLAN_SURFACE_OUT"],
            "MLXFAST_E138_PLAN_SURFACE_OUT must name the JSON destination")
        let reps = Int(env["MLXFAST_E138_REPS"] ?? "") ?? 15
        let inner = Int(env["MLXFAST_E138_INNER"] ?? "") ?? 10
        let tight = (env["MLXFAST_E138_GRID"] ?? "tight") != "wide"
        let reference = e138ParseCells(env["MLXFAST_E138_REFERENCE"] ?? "6:6:4")
        let cells = e138ParseCells(
            env["MLXFAST_E138_CELLS"] ?? "5:5:4,6:6:4,6:4:4,6:3:4")
        let shapes = e138SelectedShapes(env["MLXFAST_E138_SHAPES"])
        let ref = try #require(reference.first, "no reference cell")

        for cell in cells {
            try #require(
                cell.legal, "cell \(cell.label) has a one-row tail group")
        }

        var payload: [String: Any] = [
            "experiment": "e138-plan-surface",
            "source": "vendored-mlx-swift + Qwen35CustomQMV generator",
            "reps": reps,
            "inner_calls_per_rep": inner,
            "grid": tight ? "tight" : "wide",
            "reference_cell": ref.label,
            "cells": cells.map(\.label),
            "arm": Qwen35CustomQMV.arm.rawValue,
            "shipped_plan": Qwen35CustomQMV.renderPlan(
                Qwen35CustomQMV.widthPlan),
            "note":
                "every cell dispatches the shipped kernel body at another "
                + "(M, IPG, RPS). Each cell is ABBA-interleaved against the "
                + "reference cell inside one timed block, so the pair shares "
                + "its thermal drift.",
        ]
        payload["shapes"] = shapes.map { shape in
            e138Sweep(
                shape: shape, cells: cells, reference: ref, tight: tight,
                reps: reps, inner: inner)
        }

        let data = try JSONSerialization.data(
            withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: URL(fileURLWithPath: outPath))
    }
}

// MARK: - cells

/// Which launch grid one cell is dispatched under. `nil` takes the session
/// default. Naming the grid per cell puts both grids of one plan inside the
/// same ABBA block. That is the only way to keep between-session drift off the
/// grid axis: an identical `6:6:4` cell moved by up to 10.4 % between two
/// sessions that were matched in every other respect.
enum E138Grid: String {
    case wide
    case tight
}

struct E138Cell: Hashable {
    let m: Int
    let ipg: Int
    let rps: Int
    var grid: E138Grid? = nil

    func tight(default sessionTight: Bool) -> Bool {
        guard let grid else { return sessionTight }
        return grid == .tight
    }

    /// `ipg == 0` is not a plan. It selects the stock MLX quantized matmul at
    /// this width, so the surface can separate a plan effect from a
    /// replica-versus-stock code difference in one interleaved block.
    var isStock: Bool { ipg == 0 }

    /// `Qwen35.swift:1545`, `static_assert(M % IPG != 1)`. A one-row tail group
    /// would take the clamped `TAIL >= 2 ? TAIL : 2` branch at
    /// `Qwen35.swift:1556` and read and write one row past `M`.
    var legal: Bool { isStock || (ipg >= 1 && ipg <= m && m % ipg != 1) }
    var passes: Int { isStock ? 0 : (m + ipg - 1) / ipg }
    var label: String {
        let plan = isStock ? "\(m):stock" : "\(m):\(ipg):\(rps)"
        return grid.map { "\(plan)@\($0.rawValue)" } ?? plan
    }
    var name: String { "e138_qmv_m\(m)_ipg\(ipg)_rps\(rps)_v1" }
}

private func e138ParseCells(_ raw: String) -> [E138Cell] {
    raw.split(separator: ",").compactMap { entry in
        let split = entry.split(separator: "@", maxSplits: 1)
        let grid =
            split.count == 2
            ? E138Grid(
                rawValue: split[1].trimmingCharacters(in: .whitespaces))
            : nil
        let fields = split[0].split(separator: ":").map {
            $0.trimmingCharacters(in: .whitespaces)
        }
        if fields.count == 2, fields[1] == "stock", let m = Int(fields[0]) {
            return E138Cell(m: m, ipg: 0, rps: 0, grid: grid)
        }
        let parts = fields.compactMap { Int($0) }
        guard parts.count == 3 else { return nil }
        return E138Cell(
            m: parts[0], ipg: parts[1], rps: parts[2], grid: grid)
    }
}

/// The dispatch case for one cell, in the exact form
/// `qwen35E120QMVSource` emits at `table = true`.
func e138CaseText(_ cell: E138Cell) -> String {
    """
            case \(cell.m):
                qwen_e120_qmv_m<\(cell.m), \(cell.ipg), \(cell.rps), USE_TABLE>(
                    w, scales, biases, x, xsums, y,
                    qmv_k, qmv_n, qmv_stride,
                    qmv_gx,
                    int(qmv_tid.y) * \(2 * cell.rps) + int(qmv_sgid) * \(cell.rps),
                    qmv_lid);
                break;
    """
}

/// The live shipped body with one hypothetical dispatch case in it.
///
/// The witness comment is re-rendered for the hypothetical plan, so a compiled
/// research kernel never carries the shipped route's witness string.
private func e138Source(_ cell: E138Cell) -> String {
    let live = Qwen35CustomQMV.generatedSource(table: true, tier: nil)
    guard let open = live.range(of: "switch (qmv_m) {"),
        let fallthroughAt = live.range(of: "default:")
    else {
        preconditionFailure("the shipped QMV generator has no width switch")
    }
    let lineStart = live[..<fallthroughAt.lowerBound]
        .lastIndex(of: "\n")
        .map { live.index(after: $0) } ?? fallthroughAt.lowerBound
    let head = String(live[..<open.upperBound])
        .replacingOccurrences(
            of: Qwen35CustomQMV.planWitness,
            with: Qwen35CustomQMV.renderPlan(
                [(m: cell.m, ipg: cell.ipg, rps: cell.rps)]))
    return head + "\n" + e138CaseText(cell) + "\n" + String(live[lineStart...])
}

private nonisolated(unsafe) var e138Kernels: [String: MLXFast.MLXFastKernel] =
    [:]

private func e138Kernel(_ cell: E138Cell) -> MLXFast.MLXFastKernel {
    if let cached = e138Kernels[cell.name] { return cached }
    let kernel = MLXFast.metalKernel(
        name: cell.name,
        inputNames: ["w", "scales", "biases", "x", "xsums"],
        outputNames: ["y"],
        source: e138Source(cell),
        header: Qwen35CustomQMV.generatedHeader,
        ensureRowContiguous: true
    )
    e138Kernels[cell.name] = kernel
    return kernel
}

/// One routed matvec under a hypothetical plan, including the chunk-sum fill
/// the shipped `sumtable` arm pays on every call (`Qwen35.swift:2215-2220`).
private func e138Matmul(
    _ cell: E138Cell, _ x: MLXArray, _ weight: E138QuantWeight, tight: Bool
) -> MLXArray {
    if cell.isStock {
        return quantizedMM(
            x, weight.w, scales: weight.scales, biases: weight.biases,
            transpose: true, groupSize: 64, bits: 4)
    }
    var outShape = x.shape
    outShape[outShape.count - 1] = weight.n
    let columns = cell.tight(default: tight) ? cell.passes : cell.m
    return e138Kernel(cell)(
        [weight.w, weight.scales, weight.biases, x,
         Qwen35CustomQMV.xsumsTable(x)],
        template: [("USE_TABLE", true)],
        grid: (columns * 32, weight.n / cell.rps, 1),
        threadGroup: (32, 2, 1),
        outputShapes: [outShape],
        outputDTypes: [.bfloat16]
    )[0]
}

// MARK: - shapes

struct E138Shape {
    let name: String
    let k: Int
    let n: Int
    let callsPerVerify: Int
}

/// The seven shapes of one target verify forward, with the call counts the
/// 64-layer target implies. Identical to the E137 table so the two curves are
/// directly comparable.
let e138ScoredShapes: [E138Shape] = [
    .init(name: "linear_attn.in_proj_fused_qkvzba", k: 5120, n: 16480, callsPerVerify: 48),
    .init(name: "linear_attn.out_proj", k: 6144, n: 5120, callsPerVerify: 48),
    .init(name: "full_attn.qkv_proj_fused", k: 5120, n: 14336, callsPerVerify: 16),
    .init(name: "full_attn.o_proj", k: 6144, n: 5120, callsPerVerify: 16),
    .init(name: "mlp.gate_up_fused", k: 5120, n: 34816, callsPerVerify: 64),
    .init(name: "mlp.down", k: 17408, n: 5120, callsPerVerify: 64),
    .init(name: "head.lm_head", k: 5120, n: 248320, callsPerVerify: 1),
]

private func e138SelectedShapes(_ raw: String?) -> [E138Shape] {
    guard let raw, !raw.isEmpty else { return e138ScoredShapes }
    let wanted = Set(raw.split(separator: ",").map {
        $0.trimmingCharacters(in: .whitespaces)
    })
    return e138ScoredShapes.filter { wanted.contains($0.name) }
}

// MARK: - measurement

private func e138Sweep(
    shape: E138Shape, cells: [E138Cell], reference: E138Cell, tight: Bool,
    reps: Int, inner: Int
) -> [String: Any] {
    let weight = e138QuantWeight(k: shape.k, n: shape.n)
    let refInputs = (0..<inner).map {
        e138Activations(m: reference.m, k: shape.k, salt: $0)
    }
    eval(refInputs)
    let refBody = e138Chain(
        cell: reference, xs: refInputs, weight: weight, tight: tight,
        inner: inner)

    var rows: [[String: Any]] = []
    for cell in cells {
        let xs = (0..<inner).map {
            e138Activations(m: cell.m, k: shape.k, salt: $0)
        }
        eval(xs)
        let body = e138Chain(
            cell: cell, xs: xs, weight: weight, tight: tight, inner: inner)
        let taps = e138Median(reps: 3, inner: inner,
                              body: e138TapChain(xs: xs, inner: inner))
        let (cellSamples, refSamples) = e138ABBA(
            reps: reps, inner: inner, a: body, b: refBody)

        // The plan decides which threadgroup computes an output element, never
        // how that element is accumulated, so every cell must reproduce the
        // incumbent gate bit for bit. This is the check, not the assumption.
        let mine = e138Matmul(cell, xs[0], weight, tight: tight)
        let stock = quantizedMM(
            xs[0], weight.w, scales: weight.scales, biases: weight.biases,
            transpose: true, groupSize: 64, bits: 4)
        let mineValues = mine.asType(.float32).asArray(Float.self)
        let stockValues = stock.asType(.float32).asArray(Float.self)

        let launchedColumns: Int
        let threadgroupsPerColumn: Int
        if cell.isStock {
            launchedColumns = 0
            threadgroupsPerColumn = 0
        } else {
            launchedColumns = cell.tight(default: tight)
                ? cell.passes : cell.m
            threadgroupsPerColumn = shape.n / (2 * cell.rps)
        }

        rows.append([
            "cell": cell.label,
            "grid": cell.tight(default: tight) ? "tight" : "wide",
            "m": cell.m,
            "ipg": cell.ipg,
            "rps": cell.rps,
            "passes": cell.passes,
            "is_stock": cell.isStock,
            "launched_columns": launchedColumns,
            "threadgroups_per_column": threadgroupsPerColumn,
            "seconds_per_call": cellSamples[cellSamples.count / 2],
            "samples": cellSamples,
            "reference_seconds_per_call": refSamples[refSamples.count / 2],
            "reference_samples": refSamples,
            "tap_overhead_seconds_per_call": taps[taps.count / 2],
            "matches_incumbent_bitwise": mineValues == stockValues,
            "max_abs_delta_vs_incumbent": zip(mineValues, stockValues)
                .map { Double(abs($0 - $1)) }.max() ?? 0,
        ])
    }

    // RULE 101: a comparator that cannot fail proves nothing. One activation
    // element is moved by one bf16 step and the same comparison must reject.
    let control = e138Activations(m: reference.m, k: shape.k, salt: 0)
    let perturbed = control + e138OneStepPerturbation(m: reference.m, k: shape.k)
    let clean = e138Matmul(reference, control, weight, tight: tight)
        .asType(.float32).asArray(Float.self)
    let moved = quantizedMM(
        perturbed, weight.w, scales: weight.scales, biases: weight.biases,
        transpose: true, groupSize: 64, bits: 4
    ).asType(.float32).asArray(Float.self)

    return [
        "name": shape.name,
        "k": shape.k,
        "n": shape.n,
        "calls_per_verify": shape.callsPerVerify,
        "exactness_positive_control_rejects": clean != moved,
        "rows": rows,
    ]
}

/// The verify forward is a dependent chain, so independent calls batched into
/// one `eval` would overlap and understate cost. A 1e-30 tap of each output is
/// threaded into the next input; it vanishes in bf16 rounding, so the graph
/// edge is real and the activations are bitwise unchanged.
private func e138Chain(
    cell: E138Cell, xs: [MLXArray], weight: E138QuantWeight, tight: Bool,
    inner: Int
) -> () -> [MLXArray] {
    {
        var outs: [MLXArray] = []
        outs.reserveCapacity(inner)
        var x = xs[0]
        for i in 0..<inner {
            let o = e138Matmul(cell, x, weight, tight: tight)
            outs.append(o)
            if i + 1 < inner { x = xs[i + 1] + o[0..<1, 0..<1] * 1e-30 }
        }
        return outs
    }
}

private func e138TapChain(xs: [MLXArray], inner: Int) -> () -> [MLXArray] {
    {
        var outs: [MLXArray] = []
        outs.reserveCapacity(inner)
        var x = xs[0]
        for i in 0..<inner {
            outs.append(x)
            if i + 1 < inner { x = xs[i + 1] + x[0..<1, 0..<1] * 1e-30 }
        }
        return outs
    }
}

/// Alternates A/B/B/A across replicate pairs so a monotone drift over the
/// timed block contributes equally to both arms.
private func e138ABBA(
    reps: Int, inner: Int, a: @escaping () -> [MLXArray],
    b: @escaping () -> [MLXArray]
) -> (aSamples: [Double], bSamples: [Double]) {
    for _ in 0..<3 {
        eval(a())
        eval(b())
    }
    var aSamples: [Double] = []
    var bSamples: [Double] = []
    for rep in 0..<reps {
        if rep % 2 == 0 {
            aSamples.append(e138Time(inner: inner, a))
            bSamples.append(e138Time(inner: inner, b))
        } else {
            bSamples.append(e138Time(inner: inner, b))
            aSamples.append(e138Time(inner: inner, a))
        }
    }
    aSamples.sort()
    bSamples.sort()
    return (aSamples, bSamples)
}

private func e138Time(inner: Int, _ body: () -> [MLXArray]) -> Double {
    let start = DispatchTime.now().uptimeNanoseconds
    eval(body())
    let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
    return elapsed / Double(inner)
}

private func e138Median(
    reps: Int, inner: Int, body: () -> [MLXArray]
) -> [Double] {
    for _ in 0..<2 { eval(body()) }
    var samples: [Double] = []
    for _ in 0..<reps { samples.append(e138Time(inner: inner, body)) }
    samples.sort()
    return samples
}

// MARK: - fixtures

struct E138QuantWeight {
    let w: MLXArray
    let scales: MLXArray
    let biases: MLXArray
    let n: Int
}

private func e138QuantWeight(k: Int, n: Int) -> E138QuantWeight {
    let words = k / 8
    let tile = (0..<words).map { index -> UInt32 in
        UInt32(truncatingIfNeeded: index &* 2_654_435_761) ^ 0x9E37_79B9
    }
    let w = MLXArray(tile).reshaped([1, words])
        + arange(0, n, dtype: .uint32).reshaped([n, 1])

    let groups = k / 64
    let rowJitter = arange(0, n, dtype: .float32).reshaped([n, 1]) * 1e-6
    let scaleTile: [Float] = (0..<groups).map { index -> Float in
        0.006 + 0.004 * Float((index &* 37) % 61) / 61.0
    }
    let biasTile: [Float] = (0..<groups).map { index -> Float in
        -0.05 - 0.02 * Float((index &* 23) % 53) / 53.0
    }
    let scales = (MLXArray(scaleTile).reshaped([1, groups]) + rowJitter)
        .asType(.bfloat16)
    let biases = (MLXArray(biasTile).reshaped([1, groups]) + rowJitter)
        .asType(.bfloat16)

    let weight = E138QuantWeight(w: w, scales: scales, biases: biases, n: n)
    eval(weight.w, weight.scales, weight.biases)
    return weight
}

private func e138Activations(m: Int, k: Int, salt: Int) -> MLXArray {
    let tile: [Float] = (0..<k).map { index -> Float in
        Float((index &* 131 &+ salt &* 7919) % 251) / 251.0 - 0.5
    }
    let rowJitter = arange(0, m, dtype: .float32).reshaped([m, 1]) * 0.01
    return (MLXArray(tile).reshaped([1, k]) + rowJitter).asType(.bfloat16)
}

/// One bf16 step at a single element, everywhere else exactly zero.
private func e138OneStepPerturbation(m: Int, k: Int) -> MLXArray {
    var values = [Float](repeating: 0, count: m * k)
    values[0] = 0.01
    return MLXArray(values).reshaped([m, k]).asType(.bfloat16)
}
