import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXFastModel

// E91 -- the 512-token seed prefill, censused and priced.
//
// `begin()` is 8.59 % of the ranked candidate leg and 0 % of the ranked serial
// numerator, so every second removed from it is a pure ranked gain. E83 showed
// the seed leg is essentially all quantized GEMM; E91 named every kernel in it
// and closed the schedule axis.
//
//   RUNG 0, the census. Kernel names, dispatch counts, per-kernel grid and
//   threadgroup shapes and command-buffer commits, per phase of one `begin()`.
//   Untimed by construction: the selector swizzle takes a lock on every
//   dispatch, so it perturbs any clock in the same session.
//
//   RUNG 2, the kernel ceiling. The probe measures each prefill cell three ways
//   -- shipped `affine_qmm_t`, dense bf16 `matmul` at the same shape, and the
//   machine's own measured read bandwidth and bf16 GEMM peak -- and weights the
//   gap by the cell's measured share of the seed leg.
//
// RUNG 1, the `asyncEval` stride sweep, is CLOSED and its knob is deleted. 108
// timed blocks over nine schedules put the best arm at 0.94 sigma: the host
// enqueues the whole 64-layer graph in 118.7 ms of a 4043 ms block, so prefill
// has no host component to recover. See `research/e91-results.md`.
//
// Nothing here is on the submitted surface: Yukon never packages `Tests/`. The
// ceiling probe runs synthetic weights and makes no token claim.
//
// Enable rung 0 with `MLXFAST_RUN_E91_CENSUS=1` and rung 2 with
// `MLXFAST_RUN_E91_CEILING=1`.

@Suite(.serialized)
struct E91PrefillLadderTests {

    // MARK: - rung 0

    private static var censusEnabled: Bool {
        ProcessInfo.processInfo.environment["MLXFAST_RUN_E91_CENSUS"] == "1"
    }

    /// Every dispatch of one seed `begin()` on the shipped schedule, resolved to
    /// a kernel name and a launch shape.
    @Test(.enabled(if: E91PrefillLadderTests.censusEnabled))
    func censusThePrefillBlock() throws {
        let env = ProcessInfo.processInfo.environment
        let outPath = try #require(
            env["MLXFAST_E91_OUT"], "MLXFAST_E91_OUT must name the JSON destination")
        let weightsPath = env["MLXFAST_E91_WEIGHTS"] ?? "weights"
        let promptPath =
            env["MLXFAST_E91_PROMPT"]
            ?? "correctness_prompts/public_longcopy_gate_english_512_1024.json"
        let seedLength = Int(env["MLXFAST_E91_SEED_LEN"] ?? "") ?? 512
        let warmup = Int(env["MLXFAST_E91_WARMUP"] ?? "") ?? 1

        // The swizzle must be installed before MLX builds its first pipeline;
        // that is the only ordering in which a dispatch resolves to a real
        // kernel name rather than to `<unbound>`.
        let swizzleInstalled = e83InstallSwizzles()
        #expect(swizzleInstalled, "selector swizzle install failed")

        let tokens = try e83LoadPromptTokens(promptPath)
        #expect(tokens.count >= seedLength)

        let config = try Qwen35Config.load(from: weightsPath)
        let loader = try Qwen35WeightLoader(weightsPath: weightsPath)
        let loadStart = DispatchTime.now().uptimeNanoseconds
        let runtime = Qwen35RuntimeWeightCache(loader: loader, config: config)
        let model = try runtime.requireLibraryModel()
        let loadSeconds = Double(DispatchTime.now().uptimeNanoseconds - loadStart) / 1e9

        let harness = E83Harness(model: model, tokens: tokens, seedLength: seedLength)
        harness.warmLikeTheSession()
        for _ in 0..<warmup { _ = harness.begin(arm: .baseline, phased: false) }

        var census = harness.censusBoundaries()
        census["order"] = 0
        let blocks = [e83Emit(census)]

        let payload: [String: Any] = [
            "schema": 1,
            "experiment": env["MLXFAST_CENSUS_EXPERIMENT"] ?? "e91-prefill-census",
            "harness": "local",
            "cool_gate_passed_real_gate": false,
            "gate_qualified_for_timing": false,
            "official_or_ranked_score": false,
            "identity": [
                "weights_path": weightsPath,
                "prompt_path": promptPath,
                "seed_length": seedLength,
                "warmup": warmup,
                "swizzle_installed": swizzleInstalled,
                "model_load_seconds": loadSeconds,
                "num_hidden_layers": config.numHiddenLayers,
                "hidden_size": config.hiddenSize,
                "intermediate_size": config.intermediateSize,
                "vocab_size": config.vocabSize,
                "device": e83Device(),
                "host": ProcessInfo.processInfo.hostName,
            ],
            "blocks": blocks,
        ]
        let data = try JSONSerialization.data(
            withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: URL(fileURLWithPath: outPath))
        print("E91_OUT \(outPath)")
    }

    // MARK: - rung 2

    private static var ceilingEnabled: Bool {
        ProcessInfo.processInfo.environment["MLXFAST_RUN_E91_CEILING"] == "1"
    }

    /// How far the shipped quantized GEMM is from this machine's own measured
    /// limits at the exact prefill shapes. Runs synthetic weights, so the 15 GB
    /// checkpoint is not resident and the probe is cheap.
    @Test(.enabled(if: E91PrefillLadderTests.ceilingEnabled))
    func priceTheQuantizedGemmCeiling() throws {
        let env = ProcessInfo.processInfo.environment
        let outPath = try #require(
            env["MLXFAST_E91_CEILING_OUT"],
            "MLXFAST_E91_CEILING_OUT must name the JSON destination")
        let seedLength = Int(env["MLXFAST_E91_SEED_LEN"] ?? "") ?? 512
        let reps = Int(env["MLXFAST_E91_CEILING_REPS"] ?? "") ?? 9

        var blocks: [[String: Any]] = []

        var peaks = e91MachinePeaks(reps: reps)
        peaks["order"] = blocks.count
        blocks.append(e83Emit(peaks))
        let readBandwidth = peaks["read_gb_per_second"] as? Double ?? 0
        let bf16Peak = peaks["bf16_tflop_per_second"] as? Double ?? 0

        let executed = Set([
            "gdn.in_proj_qkv", "gdn.in_proj_z", "gdn.in_proj_b", "gdn.in_proj_a",
            "gdn.out_proj", "fa.qkv_packed", "fa.o_proj",
            "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
        ])
        let cells = e83IsolatedShapes(seed: seedLength).filter { executed.contains($0.family) }

        var modelledTotal = 0.0
        var cellBlocks: [[String: Any]] = []
        for cell in cells {
            var block = e83MeasureQuantizedShape(cell, reps: reps)
            let dense = e91MeasureDenseBF16(cell, reps: reps)
            let shipped = block["seconds_median"] as? Double ?? .nan
            let flop = block["flop_per_call"] as? Double ?? .nan
            let weightBytes = block["weight_bytes_per_call"] as? Double ?? .nan

            let bandwidthFloor = readBandwidth > 0 ? weightBytes / (readBandwidth * 1e9) : .nan
            let flopFloor = bf16Peak > 0 ? flop / (bf16Peak * 1e12) : .nan
            // Both a memory read and the arithmetic must happen, and neither
            // hides the other completely, so the larger is the honest floor.
            let hardwareFloor = max(bandwidthFloor, flopFloor)
            let denseSeconds = dense["seconds_median"] as? Double ?? .nan
            // The reachable ceiling is the faster of "a dense bf16 GEMM this
            // machine really achieves at this shape" and the hardware floor.
            let reachable = min(denseSeconds, max(hardwareFloor, 0))

            block["kind"] = "e91_ceiling_cell"
            block["dense_bf16_seconds_median"] = denseSeconds
            block["dense_bf16_tflop_per_second"] = flop / denseSeconds / 1e12
            block["bandwidth_floor_seconds"] = bandwidthFloor
            block["flop_floor_seconds"] = flopFloor
            block["hardware_floor_seconds"] = hardwareFloor
            block["reachable_ceiling_seconds"] = reachable
            block["arithmetic_intensity_flop_per_byte"] = flop / weightBytes
            block["headroom_fraction"] = (shipped - reachable) / shipped
            block["headroom_prefill_seconds"] = (shipped - reachable) * Double(cell.layers)
            modelledTotal += shipped * Double(cell.layers)
            cellBlocks.append(block)
        }

        var totalHeadroom = 0.0
        for var block in cellBlocks {
            let modelled = block["modelled_prefill_seconds"] as? Double ?? 0
            block["share_of_modelled_prefill"] =
                modelledTotal > 0 ? modelled / modelledTotal : .nan
            totalHeadroom += block["headroom_prefill_seconds"] as? Double ?? 0
            block["order"] = blocks.count
            blocks.append(e83Emit(block))
        }

        var summary: [String: Any] = [
            "kind": "e91_ceiling_summary",
            "modelled_prefill_seconds": modelledTotal,
            "headroom_prefill_seconds": totalHeadroom,
            "headroom_fraction_of_prefill":
                modelledTotal > 0 ? totalHeadroom / modelledTotal : .nan,
            // `begin()` is 8.59 % of the ranked candidate leg and 0 % of the
            // ranked serial numerator, so the leg gain is the prefill fraction
            // scaled by that share.
            "prefill_share_of_candidate_leg": 0.0859,
            "implied_candidate_leg_gain":
                modelledTotal > 0 ? 0.0859 * totalHeadroom / modelledTotal : .nan,
            "order": blocks.count,
        ]
        summary["read_gb_per_second"] = readBandwidth
        summary["bf16_tflop_per_second"] = bf16Peak
        blocks.append(e83Emit(summary))

        let payload: [String: Any] = [
            "schema": 1,
            "experiment": env["MLXFAST_CENSUS_EXPERIMENT"] ?? "e91-prefill-ceiling",
            "harness": "local",
            "cool_gate_passed_real_gate": false,
            "gate_qualified_for_timing": false,
            "official_or_ranked_score": false,
            "identity": [
                "seed_length": seedLength,
                "reps": reps,
                "device": e83Device(),
                "host": ProcessInfo.processInfo.hostName,
            ],
            "blocks": blocks,
        ]
        let data = try JSONSerialization.data(
            withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: URL(fileURLWithPath: outPath))
        print("E91_CEILING_OUT \(outPath)")
    }
}

// MARK: - helpers

/// A dense bf16 `matmul` at the same shape as the quantized cell: what this
/// machine achieves when the dequantization work is removed but the arithmetic
/// is not. At M = 512 every prefill cell is arithmetic-bound, so this is the
/// tighter of the two ceilings.
func e91MeasureDenseBF16(_ cell: E83Shape, reps: Int) -> [String: Any] {
    let x = e83Activations(m: cell.m, k: cell.k)
    let w = e83Activations(m: cell.n, k: cell.k)
    func call() -> MLXArray { matmul(x, w.transposed(1, 0)) }
    for _ in 0..<3 { eval(call()) }
    var samples: [Double] = []
    for _ in 0..<reps {
        let start = DispatchTime.now().uptimeNanoseconds
        eval(call())
        samples.append(Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9)
    }
    samples.sort()
    Memory.clearCache()
    return [
        "seconds_median": samples[samples.count / 2],
        "seconds_min": samples[0],
        "seconds_max": samples[samples.count - 1],
    ]
}

/// This machine's own limits, measured rather than quoted from a datasheet:
/// a large streaming read, a large streaming copy, and a large square bf16
/// GEMM. Every ceiling in rung 2 is expressed against these three numbers.
func e91MachinePeaks(reps: Int) -> [String: Any] {
    let entryTemp = e83GPUTemperature()
    let elements = 256 * 1024 * 1024 / 2  // 256 MiB of bf16
    let buffer = (arange(0, elements, dtype: .float32) * 1e-6).asType(.bfloat16)
    eval(buffer)
    let bytes = Double(elements * 2)

    func time(_ body: () -> MLXArray) -> Double {
        for _ in 0..<3 { eval(body()) }
        var samples: [Double] = []
        for _ in 0..<reps {
            let start = DispatchTime.now().uptimeNanoseconds
            eval(body())
            samples.append(Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9)
        }
        samples.sort()
        return samples[samples.count / 2]
    }

    let readSeconds = time { buffer.sum() }
    let copySeconds = time { buffer + MLXArray(bfloat16: 1) }

    let side = 4096
    let a = e83Activations(m: side, k: side)
    let b = e83Activations(m: side, k: side)
    let gemmSeconds = time { matmul(a, b.transposed(1, 0)) }
    let gemmFlop = 2.0 * Double(side) * Double(side) * Double(side)

    Memory.clearCache()
    var record: [String: Any] = [
        "kind": "e91_machine_peaks",
        "reps": reps,
        "buffer_bytes": bytes,
        "read_seconds": readSeconds,
        "read_gb_per_second": bytes / readSeconds / 1e9,
        "copy_seconds": copySeconds,
        "copy_gb_per_second": 2 * bytes / copySeconds / 1e9,
        "bf16_gemm_side": side,
        "bf16_gemm_seconds": gemmSeconds,
        "bf16_tflop_per_second": gemmFlop / gemmSeconds / 1e12,
    ]
    if let entryTemp { record["gpu_temp_entry_c"] = entryTemp }
    if let exit = e83GPUTemperature() { record["gpu_temp_exit_c"] = exit }
    return record
}
