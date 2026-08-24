import Foundation
import MLX
import MLXLLM
import Testing

// E174 step 2b -- CPU-side probe that splits the standalone chunk-sum fill's
// measured cost into host record construction and dispatch plus compute.
//
// The E174 screen (W&B run `7ueick4f`, `harness=local`) measured one standalone
// fill dispatch PLUS its host kernel record at 4.0 to 5.3 us of candidate MTP
// wall time. Item (2) of the E173 F-inventory proposes to cache the host record
// and keep the dispatch, so its ceiling is the record's share of that bracket
// and nothing more. The advisor's rule is arithmetic: if record construction is
// under 1 us, item (2) is dead and no cache gets built.
//
// `MLXFastKernel.callAsFunction` is the record. It builds an
// `mlx_fast_metal_kernel_config`, the template arguments, the grid, the output
// arguments and an input `vector_array`, then calls
// `mlx_fast_metal_kernel_apply`. MLX is lazy, so that call returns an
// unevaluated node: the host has paid for the record and nothing else. `eval`
// then pays for the encode, the dispatch and the compute. Timing the two
// phases separately therefore splits the bracket along exactly the line item
// (2) proposes to cut.
//
// This probe replaces the E173 `admission.kernel_record_construction` row,
// which was a RESIDUAL of an entry-point total after three other measured rows
// and so inherited all of their errors. This is a direct measurement.
//
// RESEARCH-ONLY. Nothing here is on the scored surface and nothing here runs in
// a timed leg. It is not thermally gated, so it reports host CPU time for a
// host-only phase and wall time for the device phase, and it makes no ranked
// claim.

private struct Cell {
    let name: String
    let k: Int
    let m: Int
}

/// The three activation widths the live tree actually fills, from the E174
/// dedup census: the post-norm hidden state, the attention output, and the
/// SwiGLU output. `m` is the draft-plus-primary row count of a decode round.
private let probeCells = [
    Cell(name: "hidden_5120", k: 5120, m: 6),
    Cell(name: "attn_out_6144", k: 6144, m: 6),
    Cell(name: "mlp_act_17408", k: 17408, m: 6),
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

/// `batch` distinct activations, materialised and evaluated BEFORE any timing,
/// so neither their allocation nor their own graph work lands in the window.
/// Distinct rather than shared because the dedup census measured zero duplicate
/// activations per round: every real fill sees a tensor it has not seen before.
private func makeActivations(_ cell: Cell, batch: Int) -> [MLXArray] {
    var arrays: [MLXArray] = []
    arrays.reserveCapacity(batch)
    for i in 0 ..< batch {
        let a = MLXArray(
            (0 ..< (cell.m * cell.k)).map { Float(($0 &+ i) % 17) * 0.01 }
        )
        .reshaped([cell.m, cell.k])
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

private func timeOneBatch(_ cell: Cell, activations: [MLXArray]) -> Phase {
    let batch = activations.count

    let tBuild0 = nowNanoseconds()
    var nodes: [MLXArray] = []
    nodes.reserveCapacity(batch)
    for a in activations {
        nodes.append(Qwen35CustomQMV.xsumsTable(a))
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
func xsumsRecordConstructionCostSplitsTheFillBracket() {
    guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1"
    else {
        return
    }

    let repeats = 9
    // Two batch sizes are the probe's positive control. Record construction is
    // per-call work, so its per-call cost must be near-constant in the batch
    // size. A timer that is really measuring some fixed per-batch overhead
    // would report a per-call cost that falls roughly in proportion to the
    // batch, and the check below fails instead of reporting that number as a
    // record cost.
    let batches = [32, 256]

    var report: [String: Any] = [
        "experiment": "E174-step2b-record-probe",
        "harness": "local",
        "timed_leg": false,
        "gate_qualified_for_timing": false,
        "screen_bracket_us_per_dispatch_plus_record": [4.0, 5.3],
        "screen_source": "E174 screen, wandb run 7ueick4f",
        "repeats": repeats,
        "batches": batches,
    ]
    var cellReports: [String: Any] = [:]
    var controlFailures: [String] = []
    var allRecordUs: [Double] = []

    for cell in probeCells {
        var perBatch: [String: Any] = [:]
        var recordByBatch: [Int: Double] = [:]

        for batch in batches {
            let activations = makeActivations(cell, batch: batch)
            // Warm: the kernel JIT-compiles and the pipeline state is cached on
            // first use. A compile inside the window would be reported as a
            // record cost hundreds of times too large.
            _ = timeOneBatch(cell, activations: activations)

            var record: [Double] = []
            var evals: [Double] = []
            for _ in 0 ..< repeats {
                let phase = timeOneBatch(cell, activations: activations)
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
                "eval_us_per_call_min": evals.min() ?? 0,
                "eval_us_per_call_max": evals.max() ?? 0,
            ]
        }

        // The control: per-call record cost must not track the batch size.
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
        cellReports[cell.name] = perBatch
    }

    let recordUs = median(allRecordUs)
    report["cells"] = cellReports
    report["record_us_per_call_median_over_cells"] = recordUs
    report["batch_scaling_control_failures"] = controlFailures
    // The advisor's arithmetic stop rule, evaluated here rather than by hand.
    report["record_under_1us"] = recordUs < 1.0
    report["item2_dead_by_arithmetic"] = recordUs < 1.0 && controlFailures.isEmpty
    report["record_share_of_bracket_lo"] = recordUs / 5.3
    report["record_share_of_bracket_hi"] = recordUs / 4.0

    let out = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        .appendingPathComponent("research/out/e174-record-probe.json")
    try? FileManager.default.createDirectory(
        at: out.deletingLastPathComponent(), withIntermediateDirectories: true)
    if let data = try? JSONSerialization.data(
        withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
    {
        try? data.write(to: out)
        print("e174-record-probe: wrote \(out.path)")
    }
    print(report)

    // The probe reports a number; it does not assert an outcome. The only hard
    // failure is a broken control, because then no number is readable.
    #expect(controlFailures.isEmpty)
}
