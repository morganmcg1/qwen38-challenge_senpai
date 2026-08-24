import Foundation
import MLX
import MLXLLM
import Testing

// E179 step 1 -- CPU-side probe that prices the launch-config cache.
//
// E174 measured one host kernel record at 5.566 us/call and counted 387 records
// in a decode round (257 table-paying QMV cells plus 130 standalone chunk-sum
// fills). E179 caches the `mlx_fast_metal_kernel_config` those records rebuild.
// The cache cannot touch the rest of the record: `mlx_fast_metal_kernel_apply`
// still regenerates the kernel source, names the kernel and builds the
// primitive on every call. This probe measures what the cache actually removes,
// which is the ceiling of the end-to-end effect and nothing more.
//
// The window is the same one E174 used: MLX is lazy, so building the node pays
// for the record and nothing else, and `eval` pays for encode, dispatch and
// compute afterwards. The arm comes from `MLX_E179_CFG_CACHE_ARM`, which
// `Qwen35KernelConfigCache` reads once per process, so one process measures one
// arm and `research/e179_record_probe.sh` runs the arms in ABBA order.
//
// RESEARCH-ONLY. Nothing here is on the scored surface, nothing here runs in a
// timed leg, and it is not thermally gated: it reports host time for a
// host-only phase and makes no ranked claim.

private struct ProbeCell {
    let name: String
    let k: Int
    let m: Int
    /// `nil` runs the standalone chunk-sum fill; a value runs the wide QMV
    /// against a prebuilt table at that output width.
    let n: Int?
}

/// Three live decode geometries: the fill of the post-norm hidden state, the
/// fused gate/up projection and the MLP down projection. `m = 6` is the
/// draft-plus-primary row count of a decode round.
private let probeCells = [
    ProbeCell(name: "xsums_hidden_5120", k: 5120, m: 6, n: nil),
    ProbeCell(name: "qmv_gate_up_5120", k: 5120, m: 6, n: 8192),
    ProbeCell(name: "qmv_down_17408", k: 17408, m: 6, n: 5120),
]

private func nowNanoseconds() -> UInt64 {
    DispatchTime.now().uptimeNanoseconds
}

private func median(_ values: [Double]) -> Double {
    let sorted = values.sorted()
    let mid = sorted.count / 2
    return sorted.count % 2 == 1
        ? sorted[mid] : 0.5 * (sorted[mid - 1] + sorted[mid])
}

private struct Weights {
    let w: MLXArray
    let scales: MLXArray
    let biases: MLXArray
}

/// Affine 4-bit group-64 weights of the shape the routed cell requires. The
/// values are arbitrary: this probe times host record construction and never
/// reads a result.
private func makeWeights(k: Int, n: Int) -> Weights {
    let packed = k / 8
    let groups = k / 64
    let w = MLXArray(
        (0 ..< (n * packed)).map { UInt32(truncatingIfNeeded: $0 &* 2_654_435_761) }
    ).reshaped([n, packed])
    let scales = MLXArray((0 ..< (n * groups)).map { Float($0 % 7) * 0.01 + 0.01 })
        .reshaped([n, groups]).asType(.bfloat16)
    let biases = MLXArray((0 ..< (n * groups)).map { Float($0 % 5) * -0.01 })
        .reshaped([n, groups]).asType(.bfloat16)
    eval(w, scales, biases)
    return Weights(w: w, scales: scales, biases: biases)
}

/// `batch` distinct activations, materialised and evaluated before any timing,
/// so neither their allocation nor their own graph work lands in the window.
private func makeActivations(k: Int, m: Int, batch: Int) -> [MLXArray] {
    var arrays: [MLXArray] = []
    arrays.reserveCapacity(batch)
    for i in 0 ..< batch {
        let a = MLXArray((0 ..< (m * k)).map { Float(($0 &+ i) % 17) * 0.01 })
            .reshaped([m, k])
            .asType(.bfloat16)
        arrays.append(a)
    }
    eval(arrays)
    return arrays
}

private struct Phase {
    var recordUsPerCall: Double
    var evalUsPerCall: Double
}

private func timeOneBatch(
    _ cell: ProbeCell, activations: [MLXArray], weights: Weights?,
    tables: [MLXArray]
) -> Phase {
    let batch = activations.count

    let tBuild0 = nowNanoseconds()
    var nodes: [MLXArray] = []
    nodes.reserveCapacity(batch)
    if let weights {
        for (i, a) in activations.enumerated() {
            let y = Qwen35CustomQMV.matmulWithTable(
                a, weights.w, scales: weights.scales, biases: weights.biases,
                xsums: tables[i], groupSize: 64, bits: 4, mode: .affine)
            nodes.append(y!)
        }
    } else {
        for a in activations {
            nodes.append(Qwen35CustomQMV.xsumsTable(a))
        }
    }
    let tBuild1 = nowNanoseconds()
    eval(nodes)
    let tEval1 = nowNanoseconds()

    let n = Double(batch)
    return Phase(
        recordUsPerCall: Double(tBuild1 - tBuild0) / 1000.0 / n,
        evalUsPerCall: Double(tEval1 - tBuild1) / 1000.0 / n
    )
}

@Test
func kernelConfigCacheRecordCostSplitsTheRecordBracket() {
    guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1"
    else {
        return
    }

    let environment = ProcessInfo.processInfo.environment
    let arm = environment["MLX_E179_CFG_CACHE_ARM"] ?? "on"
    let tag = environment["E179_PROBE_TAG"] ?? arm
    let repeats = 9
    // Two batch sizes are the positive control, as in the E174 probe. Record
    // construction is per-call work, so its per-call cost must be near-constant
    // in the batch size.
    let batches = [32, 256]

    var report: [String: Any] = [
        "experiment": "E179-step1-config-cache-record-probe",
        "harness": "local",
        "timed_leg": false,
        "gate_qualified_for_timing": false,
        "arm_env": arm,
        "arm_resolved_enabled": Qwen35KernelConfigCache.enabled,
        "tag": tag,
        "repeats": repeats,
        "batches": batches,
        "records_per_round": 387,
        "e174_record_us_per_call": 5.566,
    ]
    var cellReports: [String: Any] = [:]
    var controlFailures: [String] = []
    var allRecordUs: [Double] = []

    for cell in probeCells {
        var perBatch: [String: Any] = [:]
        var recordByBatch: [Int: Double] = [:]
        let weights = cell.n.map { makeWeights(k: cell.k, n: $0) }

        for batch in batches {
            let activations = makeActivations(k: cell.k, m: cell.m, batch: batch)
            let tables =
                cell.n == nil
                ? []
                : activations.map { Qwen35CustomQMV.xsumsTable($0) }
            if !tables.isEmpty { eval(tables) }

            // Warm: the kernel JIT-compiles and the pipeline state is cached on
            // first use, and the cache arm takes its first miss here. A compile
            // inside the window would be reported as a record cost hundreds of
            // times too large.
            _ = timeOneBatch(cell, activations: activations, weights: weights, tables: tables)

            var record: [Double] = []
            var evals: [Double] = []
            for _ in 0 ..< repeats {
                let phase = timeOneBatch(
                    cell, activations: activations, weights: weights, tables: tables)
                record.append(phase.recordUsPerCall)
                evals.append(phase.evalUsPerCall)
            }
            let recordMedian = median(record)
            recordByBatch[batch] = recordMedian
            allRecordUs.append(recordMedian)
            perBatch["batch_\(batch)"] = [
                "record_us_per_call_median": recordMedian,
                "record_us_per_call_min": record.min() ?? 0,
                "record_us_per_call_max": record.max() ?? 0,
                "eval_us_per_call_median": median(evals),
            ]
        }

        if let small = recordByBatch[batches[0]], let large = recordByBatch[batches[1]],
            small > 0, large > 0
        {
            let ratio = small / large
            perBatch["batch_scaling_ratio"] = ratio
            perBatch["batch_scaling_control_ok"] = ratio < 2.0 && ratio > 0.5
            if !(ratio < 2.0 && ratio > 0.5) {
                controlFailures.append("\(cell.name):ratio=\(ratio)")
            }
        } else {
            controlFailures.append("\(cell.name):missing")
        }

        perBatch["k"] = cell.k
        perBatch["m"] = cell.m
        perBatch["n"] = cell.n ?? 0
        cellReports[cell.name] = perBatch
    }

    report["cells"] = cellReports
    report["record_us_per_call_median_over_cells"] = median(allRecordUs)
    report["batch_scaling_control_failures"] = controlFailures
    // Hit-rate witness. Every call in this process routes through the same
    // three geometries, so the on arm must show a handful of misses and
    // thousands of hits, and the off arm must show zero of both.
    report["cache_hits"] = qwen35KernelConfigCacheHits
    report["cache_misses"] = qwen35KernelConfigCacheMisses

    let out = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        .appendingPathComponent("research/out/e179-record-probe-\(tag).json")
    try? FileManager.default.createDirectory(
        at: out.deletingLastPathComponent(), withIntermediateDirectories: true)
    if let data = try? JSONSerialization.data(
        withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
    {
        try? data.write(to: out)
    }

    #expect(controlFailures.isEmpty)
    if Qwen35KernelConfigCache.enabled {
        #expect(qwen35KernelConfigCacheHits > 0)
    } else {
        #expect(qwen35KernelConfigCacheHits == 0)
        #expect(qwen35KernelConfigCacheMisses == 0)
    }
}
