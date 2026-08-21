import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXFastModel

// E91 -- the 512-token seed prefill, attacked on two axes.
//
// `begin()` is 8.59 % of the ranked candidate leg and 0 % of the ranked serial
// numerator, so every second removed from it is a pure ranked gain. E83 already
// showed the seed leg is essentially all quantized GEMM. E91 asks the two
// questions E83 left open.
//
//   RUNG 1, the schedule axis. `Qwen35TextModelInner` fires `asyncEval` at
//   `{0} + every i % 3 == 2` when the input is at least 512 rows. Nobody chose
//   stride 3 for prefill; it was scaled from a decode-width Laguna receipt. The
//   sweep drives the rung set from stride 1 to stride 16 and off, over one
//   resident model, and reads absolute `begin` wall time. Bit-exactness is
//   asserted across arms: `asyncEval` moves an enqueue boundary, never an op.
//
//   RUNG 2, the kernel ceiling. At M = 512 every prefill cell has an arithmetic
//   intensity above 1000 FLOP/byte, so the honest ceiling is arithmetic, not
//   bandwidth. The ceiling probe measures each cell three ways -- shipped
//   `affine_qmm_t`, dense bf16 `matmul` at the same shape, and the machine's own
//   measured read bandwidth and bf16 GEMM peak -- and weights the gap by the
//   cell's measured share of the seed leg.
//
// Timing-only instrument. Nothing here is on the submitted surface: Yukon never
// packages `Tests/`. The rung-1 arms are exact; the rung-2 probe runs synthetic
// weights and makes no token claim.
//
// Enable rung 1 with `MLXFAST_RUN_E91_LADDER=1` and rung 2 with
// `MLXFAST_RUN_E91_CEILING=1`.

@Suite(.serialized)
struct E91PrefillLadderTests {

    /// The shipped schedule, asserted as a property of the stride helper rather
    /// than as a literal set, so a future stride change cannot pass silently.
    @Test
    func shippedPrefillLadderIsStrideThree() {
        let shipped = qwen35PrefillLadderStride(3)
        for layer in 0..<64 {
            #expect(shipped.contains(layer) == (layer == 0 || layer % 3 == 2))
        }
        #expect(shipped.filter { $0 < 64 }.count == 22)
        #expect(qwen35PrefillLadderStride(1).filter { $0 < 64 }.count == 64)
        #expect(qwen35PrefillLadderStride(2).filter { $0 < 64 }.count == 33)
        #expect(qwen35PrefillLadderStride(64).filter { $0 < 64 }.count == 2)
        #expect(e91LadderRungs("ship") == shipped)
        #expect(e91LadderRungs("off") == [])
        #expect(e91LadderRungs("s4") == qwen35PrefillLadderStride(4))
        #expect(e91LadderRungs("nonsense") == nil)
    }

    // MARK: - rung 1

    private static var ladderEnabled: Bool {
        ProcessInfo.processInfo.environment["MLXFAST_RUN_E91_LADDER"] == "1"
    }

    @Test(.enabled(if: E91PrefillLadderTests.ladderEnabled))
    func sweepThePrefillLadderStride() throws {
        let env = ProcessInfo.processInfo.environment
        let outPath = try #require(
            env["MLXFAST_E91_OUT"], "MLXFAST_E91_OUT must name the JSON destination")

        let weightsPath = env["MLXFAST_E91_WEIGHTS"] ?? "weights"
        let promptPath =
            env["MLXFAST_E91_PROMPT"]
            ?? "correctness_prompts/public_longcopy_gate_english_512_1024.json"
        let seedLength = Int(env["MLXFAST_E91_SEED_LEN"] ?? "") ?? 512
        let reps = Int(env["MLXFAST_E91_REPS"] ?? "") ?? 3
        let warmup = Int(env["MLXFAST_E91_WARMUP"] ?? "") ?? 2
        // `reps == 0` is the census-only session. No timed arm runs, so the
        // selector swizzle may be installed before MLX builds its first
        // pipeline, which is the only ordering that resolves a dispatch to a
        // real kernel name.
        let censusOnly = reps == 0
        let runCensus = env["MLXFAST_E91_CENSUS"].map { $0 == "1" } ?? censusOnly
        let armNames =
            (env["MLXFAST_E91_ARMS"]?
                .split(separator: ",")
                .map { $0.trimmingCharacters(in: .whitespaces) })
            .flatMap { $0.isEmpty ? nil : $0 }
            ?? ["s1", "s2", "s4", "s6", "s8", "s12", "s16", "off"]
        let censusArms =
            (env["MLXFAST_E91_CENSUS_ARMS"]?
                .split(separator: ",")
                .map { $0.trimmingCharacters(in: .whitespaces) })
            .flatMap { $0.isEmpty ? nil : $0 } ?? ["ship", "off"]

        for name in armNames + censusArms + ["ship"] {
            #expect(e91LadderRungs(name) != nil, "unknown E91 ladder arm \(name)")
        }

        var swizzleInstalled = false
        if runCensus {
            #expect(censusOnly, "a census session must set MLXFAST_E91_REPS=0")
            swizzleInstalled = e83InstallSwizzles()
            #expect(swizzleInstalled, "selector swizzle install failed")
        }

        let tokens = try e83LoadPromptTokens(promptPath)
        #expect(tokens.count >= seedLength)

        let config = try Qwen35Config.load(from: weightsPath)
        let loader = try Qwen35WeightLoader(weightsPath: weightsPath)
        let loadStart = DispatchTime.now().uptimeNanoseconds
        let runtime = Qwen35RuntimeWeightCache(loader: loader, config: config)
        let model = try runtime.requireLibraryModel()
        let loadSeconds = Double(DispatchTime.now().uptimeNanoseconds - loadStart) / 1e9
        let layerCount = config.numHiddenLayers

        let shipped = qwen35PrefillLadderStride(3)
        #expect(
            model.model.prefillLadderRungs == shipped,
            "a scored run must start on the shipped schedule")

        let harness = E83Harness(model: model, tokens: tokens, seedLength: seedLength)
        var blocks: [[String: Any]] = []
        var fingerprints: [String: Set<String>] = [:]

        /// One `begin()` under the named schedule. The model instance carries
        /// the rung set, so no environment variable and no rebuild is involved
        /// and every arm sees the same resident weights.
        func measure(_ label: String, pairArm: String, position: Int) -> [String: Any] {
            let rungs = e91LadderRungs(label) ?? shipped
            model.model.prefillLadderRungs = rungs
            // Host CPU actually burned over the same interval. A stuck or
            // spinning host shows up here and nowhere else in the wall clock.
            let cpuStart = clock_gettime_nsec_np(CLOCK_THREAD_CPUTIME_ID)
            var block = harness.begin(arm: .baseline, phased: false)
            block["host_thread_cpu_ns"] =
                Int(clock_gettime_nsec_np(CLOCK_THREAD_CPUTIME_ID) - cpuStart)
            let armed = rungs.filter { $0 < layerCount }.count
            block["arm"] = label
            block["ladder_label"] = label
            block["ladder_stride"] = e91LadderStride(label) ?? -1
            block["forced_eval_points"] = armed
            block["pair_arm"] = pairArm
            block["pair_position"] = position
            block["order"] = blocks.count
            let fingerprint = e91Fingerprint(block)
            block["token_fingerprint"] = fingerprint
            fingerprints[label, default: []].insert(fingerprint)
            return block
        }

        harness.warmLikeTheSession()
        model.model.prefillLadderRungs = shipped
        for _ in 0..<warmup { _ = harness.begin(arm: .baseline, phased: false) }

        // Rung 0. Kernel names, dispatch counts, per-kernel grid shapes and
        // command-buffer commits per phase of one `begin()`. Untimed: the
        // selector swizzle perturbs the clock, so this never shares a session
        // with a timed arm.
        if runCensus && swizzleInstalled {
            for label in censusArms {
                model.model.prefillLadderRungs = e91LadderRungs(label) ?? shipped
                var census = harness.censusBoundaries()
                census["ladder_label"] = label
                census["ladder_stride"] = e91LadderStride(label) ?? -1
                census["forced_eval_points"] =
                    (e91LadderRungs(label) ?? shipped).filter { $0 < layerCount }.count
                census["order"] = blocks.count
                blocks.append(e83Emit(census))
            }
        }

        // Rung 1. ABBA inside every pair, so monotone thermal or clock drift
        // cancels to first order in the ship-minus-arm difference. The final
        // pair of each rep is ship against ship: the null that says how much of
        // any measured difference the instrument invents.
        for rep in 0..<reps {
            for arm in armNames + ["ship_null"] {
                let candidate = arm == "ship_null" ? "ship" : arm
                for (position, label) in ["ship", candidate, candidate, "ship"].enumerated() {
                    var block = measure(label, pairArm: arm, position: position)
                    block["rep"] = rep
                    blocks.append(e83Emit(block))
                }
            }
        }
        model.model.prefillLadderRungs = shipped

        // `asyncEval` changes when work is enqueued, never what is computed. If
        // any arm moves the tail row's top-2 evidence, the knob is not a pure
        // scheduling change and no timing on this page is usable.
        let allFingerprints = Set(fingerprints.values.flatMap { $0 })
        if !censusOnly {
            #expect(
                allFingerprints.count == 1,
                "prefill ladder stride changed the emitted evidence: \(fingerprints)")
        }

        let payload: [String: Any] = [
            "schema": 1,
            "experiment": env["MLXFAST_CENSUS_EXPERIMENT"] ?? "e91-prefill-ladder",
            "harness": "local",
            "cool_gate_passed_real_gate": false,
            "gate_qualified_for_timing": false,
            "official_or_ranked_score": false,
            "identity": [
                "weights_path": weightsPath,
                "prompt_path": promptPath,
                "seed_length": seedLength,
                "reps": reps,
                "warmup": warmup,
                "arms": armNames,
                "census_only": censusOnly,
                "census_arms": runCensus ? censusArms : [],
                "swizzle_installed": swizzleInstalled,
                "shipped_forced_eval_points": shipped.filter { $0 < layerCount }.count,
                "model_load_seconds": loadSeconds,
                "num_hidden_layers": layerCount,
                "hidden_size": config.hiddenSize,
                "intermediate_size": config.intermediateSize,
                "vocab_size": config.vocabSize,
                "device": e83Device(),
                "host": ProcessInfo.processInfo.hostName,
            ],
            "token_fingerprints": fingerprints.mapValues { Array($0).sorted() },
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

/// `nil` for an unparseable label, so a typo in the arm list fails the test
/// instead of silently measuring the shipped schedule twice.
func e91LadderRungs(_ label: String) -> Set<Int>? {
    switch label {
    case "ship", "default": return qwen35PrefillLadderStride(3)
    case "off": return []
    default:
        guard label.hasPrefix("s"), let stride = Int(label.dropFirst()), stride > 0 else {
            return nil
        }
        return qwen35PrefillLadderStride(stride)
    }
}

func e91LadderStride(_ label: String) -> Int? {
    switch label {
    case "ship", "default": return 3
    case "off": return 0
    default: return label.hasPrefix("s") ? Int(label.dropFirst()) : nil
    }
}

/// The tail row's exact top-2 evidence, in a form that a one-ulp move breaks.
func e91Fingerprint(_ block: [String: Any]) -> String {
    let primary = (block["first_primary"] as? Int).map(String.init) ?? "?"
    let values =
        (block["top2_values"] as? [Double])?
        .map { String(format: "%a", $0) }
        .joined(separator: ",") ?? "?"
    return "\(primary)|\(values)"
}

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
