import Foundation
import MLX
import MLXFast
import MLXLLM
import MLXLMCommon
import MLXNN
import Metal
import Testing

@testable import MLXFastModel

// E83 -- decomposition of the charged 512-token seed prefill.
//
// THE QUESTION. `begin()` is 8.6-9.4 % of the ranked candidate leg and no board
// submission has ever changed it. How much of it is quantized GEMM running on
// its own roofline, and how much is something else?
//
// THE METHOD, three independent instruments over one resident model:
//
//   1. PHASE CLOCKS. `begin()` is replayed twice per block: once exactly as the
//      session runs it (one blocking eval, `arm = whole`) and once with an eval
//      boundary after each phase (`arm = phased`). The whole arm is the
//      reference wall clock; the phased arm attributes it. Their difference is
//      the pipelining credit the single-eval form earns and is reported, not
//      hidden.
//
//   2. IN-SITU FAMILY TAX. E71's module surgery, re-pointed at M = 512. A
//      family is pinned to one row while every other family keeps its 512-row
//      shape, so `tax(F) = T(512) - T(512, F pinned to 1)`. At prefill the
//      reachable set is much larger than at decode width: `Qwen35FusedMLP`
//      gates its fused gate_up GEMM on `x.dim(-2) <= 16` and
//      `Qwen35GatedDeltaNet` gates its fused in-projection on `S <= 9`, so at
//      512 rows both dispatch through their child `Linear` modules and both
//      become interceptable. Only `fa.qkv_packed` stays fused and unreachable.
//
//   3. DISPATCH AND COMMAND-BUFFER CENSUS. Metal selector swizzling counts
//      kernel dispatches, distinct kernel names and command-buffer commits per
//      phase. This is the boundary count H-221 needs; it runs in its own
//      untimed block because the swizzle perturbs the clock.
//
// Plus an isolated `quantizedMM` sweep at every prefill shape, which prices the
// unreachable families and calibrates the roofline against the reachable ones.
//
// Timing-only instrument. The pinned arms produce WRONG TOKENS by construction.
// Nothing here is on the submitted surface: Yukon never packages `Tests/`.
//
// Enable with `MLXFAST_RUN_E83_PREFILL=1` and point `MLXFAST_E83_OUT` at the
// JSON destination.

@Suite(.serialized)
struct E83PrefillDecompositionTests {
    private static var enabled: Bool {
        ProcessInfo.processInfo.environment["MLXFAST_RUN_E83_PREFILL"] == "1"
    }

    @Test(.enabled(if: E83PrefillDecompositionTests.enabled))
    func decomposeTheSeedPrefill() throws {
        let env = ProcessInfo.processInfo.environment
        let outPath = try #require(
            env["MLXFAST_E83_OUT"], "MLXFAST_E83_OUT must name the JSON destination")

        let weightsPath = env["MLXFAST_E83_WEIGHTS"] ?? "weights"
        let promptPath =
            env["MLXFAST_E83_PROMPT"]
            ?? "correctness_prompts/public_longcopy_gate_english_512_1024.json"
        let seedLength = Int(env["MLXFAST_E83_SEED_LEN"] ?? "") ?? 512
        let reps = Int(env["MLXFAST_E83_REPS"] ?? "") ?? 5
        let warmup = Int(env["MLXFAST_E83_WARMUP"] ?? "") ?? 2
        let stallMillis = Int(env["MLXFAST_E83_STALL_MS"] ?? "") ?? 20
        let armNames =
            (env["MLXFAST_E83_ARMS"]?.split(separator: ",").map {
                $0.trimmingCharacters(in: .whitespaces)
            }).flatMap { $0.isEmpty ? nil : $0 }
            ?? [
                "null", "mlp_all", "mlp_gate", "mlp_up", "mlp_down",
                "fa_o_proj", "gdn_out_proj", "gdn_in_qkv", "gdn_in_z",
                "gdn_in_ba", "all_interceptable",
            ]

        let tokens = try e83LoadPromptTokens(promptPath)
        #expect(tokens.count >= seedLength)

        let config = try Qwen35Config.load(from: weightsPath)
        let loader = try Qwen35WeightLoader(weightsPath: weightsPath)
        let loadStart = DispatchTime.now().uptimeNanoseconds
        let runtime = Qwen35RuntimeWeightCache(loader: loader, config: config)
        let model = try runtime.requireLibraryModel()
        let loadSeconds = Double(DispatchTime.now().uptimeNanoseconds - loadStart) / 1e9

        let harness = E83Harness(model: model, tokens: tokens, seedLength: seedLength)

        // The session runs `begin()` after an untimed decode warm, so the fused
        // decode packs are already built and every prefill shape has a trace.
        harness.warmLikeTheSession()
        for _ in 0..<warmup { _ = harness.begin(arm: .baseline, phased: false) }

        var blocks: [[String: Any]] = []

        // Rung 1 -- phase clocks. ABBA over (whole, phased) so monotone drift
        // cancels to first order in their difference.
        for rep in 0..<reps {
            for phased in [false, true, true, false] {
                var block = harness.begin(arm: .baseline, phased: phased)
                block["rep"] = rep
                block["order"] = blocks.count
                blocks.append(e83Emit(block))
            }
        }

        // Rung 1 positive control. A deliberate 20 ms host stall in exactly one
        // phase must appear in that phase and nowhere else.
        for phase in ["p1_cache_alloc", "p4_tail_norm_lmhead"] {
            for _ in 0..<2 {
                var block = harness.begin(
                    arm: .baseline, phased: true,
                    stallPhase: phase, stallMillis: stallMillis)
                block["order"] = blocks.count
                blocks.append(e83Emit(block))
            }
        }

        // Rung 1 boundary census. Untimed: the swizzle perturbs the clock.
        blocks.append(e83Emit(harness.censusBoundaries()))

        // Rung 2 -- in-situ family tax at M = 512, ABBA within each arm.
        for name in armNames {
            guard let arm = E83Arm.named(name) else {
                Issue.record("unknown E83 arm \(name)")
                continue
            }
            for phased in [false] {
                for pattern in [E83Arm.baseline, arm, arm, E83Arm.baseline] {
                    var block = harness.begin(arm: pattern, phased: phased)
                    block["pair_arm"] = name
                    block["order"] = blocks.count
                    blocks.append(e83Emit(block))
                }
            }
        }

        // H-221 at prefill width, as a regression discontinuity. The prefill
        // ladder arms at exactly `dim(1) >= 512`, so widths 496...511 force no
        // evaluation point and widths 512...528 force 22, while arithmetic
        // moves by at most 6%. Fit seconds against width on the ladder-off
        // side, extrapolate across the step, and the residual on the
        // ladder-on side is the net cost of 22 boundaries. Interleaved
        // low/high so thermal drift cannot masquerade as the step.
        let ladderWidths = [496, 512, 504, 520, 511, 528]
        for rep in 0..<max(reps, 5) {
            for width in (rep % 2 == 0 ? ladderWidths : ladderWidths.reversed()) {
                var block = harness.begin(arm: .baseline, phased: false, width: width)
                block["kind"] = "ladder_step"
                block["rep"] = rep
                block["order"] = blocks.count
                blocks.append(e83Emit(block))
            }
        }

        // Rung 2 -- isolated roofline at the exact prefill shapes.
        let shapes = e83IsolatedShapes(seed: seedLength)
        for cell in shapes {
            var block = e83MeasureQuantizedShape(cell, reps: max(reps, 5))
            block["order"] = blocks.count
            blocks.append(e83Emit(block))
        }
        var sdpaBlock = e83MeasureSdpa(seed: seedLength, reps: max(reps, 5))
        sdpaBlock["order"] = blocks.count
        blocks.append(e83Emit(sdpaBlock))

        let payload: [String: Any] = [
            "schema": 1,
            "experiment": env["MLXFAST_CENSUS_EXPERIMENT"] ?? "e83-prefill-decomposition",
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
                "stall_millis": stallMillis,
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
        print("E83_OUT \(outPath)")
    }
}

// MARK: - harness

/// One resident model, replayed as `begin()` blocks.
private final class E83Harness {
    let model: Qwen35TextModel
    let tokens: [Int32]
    let seedLength: Int

    init(model: Qwen35TextModel, tokens: [Int], seedLength: Int) {
        self.model = model
        self.tokens = tokens.map(Int32.init)
        self.seedLength = seedLength
    }

    private func seedInput(_ width: Int) -> LMInput.Text {
        LMInput.Text(tokens: MLXArray(Array(tokens[0..<width])).reshaped([1, width]))
    }

    /// The untimed warm the trusted driver runs before it starts the clock:
    /// decode-width forwards build the fused decode packs and their shape
    /// traces, so `begin()` never pays a first-touch cost for them.
    func warmLikeTheSession() {
        let cache = model.newCache(parameters: nil)
        for width in [1, 2, 4, 6, 9] {
            let ids = MLXArray(Array(tokens[0..<width])).reshaped([1, width])
            let (logits, hidden) = model.callWithHidden(
                input: LMInput.Text(tokens: ids), cache: cache, nConfirmed: 0)
            eval(cache.flatMap { $0.state } + [logits, hidden])
        }
        Memory.clearCache()
    }

    /// One `begin()`. `phased == false` is the session's exact form: build the
    /// whole graph, then one blocking eval, then the host readback.
    func begin(
        arm: E83Arm, phased: Bool, width: Int? = nil,
        stallPhase: String? = nil, stallMillis: Int = 0
    ) -> [String: Any] {
        let width = width ?? seedLength
        let entryTemp = e83GPUTemperature()
        let restore = arm.install(model, width)
        defer { restore() }

        var phases: [String: Double] = [:]
        var mark = DispatchTime.now().uptimeNanoseconds
        func close(_ name: String) {
            let now = DispatchTime.now().uptimeNanoseconds
            phases[name] = Double(now - mark) / 1e9
            mark = now
        }
        func stall(_ name: String) {
            guard stallPhase == name, stallMillis > 0 else { return }
            usleep(UInt32(stallMillis) * 1000)
        }

        let total0 = mark

        // p1 -- cache construction.
        let cache = model.newCache(parameters: nil)
        stall("p1_cache_alloc")
        if phased {
            eval(cache.flatMap { $0.state })
            close("p1_cache_alloc")
        }

        // p2 -- the target forward's host graph build. `seedLogits` projects
        // lm_head over all 512 seed rows and is deliberately never evaluated.
        let (seedLogits, hidden) = model.callWithHidden(
            input: seedInput(width), cache: cache, nConfirmed: 0)
        _ = seedLogits
        stall("p2_target_forward_build")
        if phased { close("p2_target_forward_build") }

        // p3 -- the target forward's device work.
        if phased {
            eval(cache.flatMap { $0.state } + [hidden])
            stall("p3_target_forward_eval")
            close("p3_target_forward_eval")
        }

        // p4 -- the tail row: final norm, then one row of 5120 -> 248320.
        let tailRow = hidden[0..., (hidden.dim(1) - 1) ..< hidden.dim(1), 0...]
        let pendingHidden = model.applyFinalNorm(tailRow)
        let lastLogits = model.applyLMHead(pendingHidden)
        stall("p4_tail_norm_lmhead")
        if phased {
            eval(pendingHidden, lastLogits)
            close("p4_tail_norm_lmhead")
        }

        // p5 -- exact top-2 evidence for the tail row.
        let (tailIDs, tailValues) = Qwen36MTPBlockSession.linearTopTwoRows(lastLogits)
        stall("p5_top_two")
        if phased {
            eval(tailIDs, tailValues)
            close("p5_top_two")
        }

        // The session's single blocking eval over every root it needs.
        let buildDone = DispatchTime.now().uptimeNanoseconds
        eval(cache.flatMap { $0.state } + [tailIDs, tailValues, pendingHidden, hidden])
        let evalDone = DispatchTime.now().uptimeNanoseconds
        if phased { close("p6_final_eval") }

        // p7 -- host readback of the first primary and its top-2 evidence.
        let ids = tailIDs.asArray(Int32.self).map { Int($0) }
        let values = tailValues.asArray(Float.self).map { Double($0) }
        if phased { close("p7_host_readback") }
        let total1 = DispatchTime.now().uptimeNanoseconds

        let exitTemp = e83GPUTemperature()
        var record: [String: Any] = [
            "kind": "begin",
            "arm": arm.name,
            "pin_rows": arm.pinRows(width),
            "phased": phased,
            "seed_length": width,
            // `Qwen35TextModelInner.callAsFunction` arms the prefill ladder at
            // `dim(1) >= 512` and fires `asyncEval` at `i == 0 || i % 3 == 2`.
            // Below 512 (and above the 9-wide decode ladder) nothing is forced,
            // so the 511/512 step is a regression discontinuity in boundary
            // count with only 1/512 more arithmetic on the high side.
            "forced_eval_points": width >= 512 ? 22 : (width <= 9 ? 8 : 0),
            "begin_seconds": Double(total1 - total0) / 1e9,
            "build_seconds": Double(buildDone - total0) / 1e9,
            "final_eval_seconds": Double(evalDone - buildDone) / 1e9,
            "readback_seconds": Double(total1 - evalDone) / 1e9,
            "first_primary": ids.first ?? -1,
            "top2_values": values,
            "cache_offset": cache.map(\.offset).max() ?? 0,
        ]
        if !phases.isEmpty {
            record["phases"] = phases
            record["phase_sum_seconds"] = phases.values.reduce(0, +)
        }
        if let stallPhase { record["stall_phase"] = stallPhase }
        if stallMillis > 0, stallPhase != nil { record["stall_millis"] = stallMillis }
        if let entryTemp { record["gpu_temp_entry_c"] = entryTemp }
        if let exitTemp { record["gpu_temp_exit_c"] = exitTemp }

        Memory.clearCache()
        return record
    }

    /// Kernel dispatches, distinct kernel names and command-buffer commits per
    /// phase of one `begin()`. Untimed by design.
    func censusBoundaries() -> [String: Any] {
        guard e83InstallSwizzles() else {
            return ["kind": "boundary_census", "error": "swizzle install failed"]
        }
        var perPhase: [String: Any] = [:]
        func snap(_ name: String) {
            perPhase[name] = E83DispatchLedger.shared.snapshot()
        }

        E83DispatchLedger.shared.start()
        let cache = model.newCache(parameters: nil)
        eval(cache.flatMap { $0.state })
        snap("p1_cache_alloc")

        let (seedLogits, hidden) = model.callWithHidden(
            input: seedInput(seedLength), cache: cache, nConfirmed: 0)
        _ = seedLogits
        snap("p2_target_forward_build")

        eval(cache.flatMap { $0.state } + [hidden])
        snap("p3_target_forward_eval")

        let tailRow = hidden[0..., (hidden.dim(1) - 1) ..< hidden.dim(1), 0...]
        let pendingHidden = model.applyFinalNorm(tailRow)
        let lastLogits = model.applyLMHead(pendingHidden)
        eval(pendingHidden, lastLogits)
        snap("p4_tail_norm_lmhead")

        let (tailIDs, tailValues) = Qwen36MTPBlockSession.linearTopTwoRows(lastLogits)
        eval(tailIDs, tailValues)
        snap("p5_top_two")

        _ = tailIDs.asArray(Int32.self)
        _ = tailValues.asArray(Float.self)
        snap("p7_host_readback")
        let totals = E83DispatchLedger.shared.stop()

        Memory.clearCache()
        return [
            "kind": "boundary_census",
            "seed_length": seedLength,
            "per_phase": perPhase,
            "totals": totals,
        ]
    }
}

// MARK: - arms

/// A pinned-width ablation: which modules get wrapped, and at what pin width.
private struct E83Arm {
    let name: String
    /// Rows the wrapped family sees. `nil` means "the caller's width", i.e. the
    /// null control: same wrapper, same launches, same copy, full-width family.
    let pin: Int?
    let wrap: (Qwen35TextModel, Int) -> () -> Void

    func pinRows(_ width: Int) -> Int { pin ?? width }
    func install(_ model: Qwen35TextModel, _ width: Int) -> () -> Void {
        wrap(model, pin ?? width)
    }

    nonisolated(unsafe) static let baseline = E83Arm(name: "baseline", pin: nil) { _, _ in {} }

    static func named(_ name: String) -> E83Arm? {
        switch name {
        case "baseline":
            return baseline

        // Control. The wrapper runs around `mlp` at FULL width, so anything it
        // measures is harness overhead and nothing else.
        case "null":
            return E83Arm(name: "null", pin: nil) { model, pin in
                e83WrapUnary(model, ["mlp"], pin)
            }

        // 64 layers, 70.25 % of the prefill GEMM FLOP.
        case "mlp_all":
            return E83Arm(name: "mlp_all", pin: 1) { model, pin in
                e83WrapUnary(model, ["mlp"], pin)
            }
        case "mlp_gate":
            return E83Arm(name: "mlp_gate", pin: 1) { model, pin in
                e83WrapLinear(model, ["mlp", "gate_proj"], pin)
            }
        case "mlp_up":
            return E83Arm(name: "mlp_up", pin: 1) { model, pin in
                e83WrapLinear(model, ["mlp", "up_proj"], pin)
            }
        case "mlp_down":
            return E83Arm(name: "mlp_down", pin: 1) { model, pin in
                e83WrapLinear(model, ["mlp", "down_proj"], pin)
            }

        // 16 full-attention layers. The only attention projection that still
        // dispatches through its child module; q/k/v are packed and fused.
        case "fa_o_proj":
            return E83Arm(name: "fa_o_proj", pin: 1) { model, pin in
                e83WrapLinear(model, ["self_attn", "o_proj"], pin)
            }

        // 48 GDN layers. At S > 9 all four in-projections and the out
        // projection dispatch through their child `Linear`.
        case "gdn_out_proj":
            return E83Arm(name: "gdn_out_proj", pin: 1) { model, pin in
                e83WrapLinear(model, ["linear_attn", "out_proj"], pin)
            }
        case "gdn_in_qkv":
            return E83Arm(name: "gdn_in_qkv", pin: 1) { model, pin in
                e83WrapLinear(model, ["linear_attn", "in_proj_qkv"], pin)
            }
        case "gdn_in_z":
            return E83Arm(name: "gdn_in_z", pin: 1) { model, pin in
                e83WrapLinear(model, ["linear_attn", "in_proj_z"], pin)
            }
        case "gdn_in_ba":
            return E83Arm(name: "gdn_in_ba", pin: 1) { model, pin in
                let restores = [
                    e83WrapLinear(model, ["linear_attn", "in_proj_b"], pin),
                    e83WrapLinear(model, ["linear_attn", "in_proj_a"], pin),
                ]
                return { restores.reversed().forEach { $0() } }
            }

        // Every reachable family at once. What remains at full width is
        // `fa.qkv_packed`, SDPA, the GDN conv and scan, the norms and gates,
        // the embedding gather and the host.
        case "all_interceptable":
            return E83Arm(name: "all_interceptable", pin: 1) { model, pin in
                let restores = [
                    e83WrapUnary(model, ["mlp"], pin),
                    e83WrapLinear(model, ["self_attn", "o_proj"], pin),
                    e83WrapLinear(model, ["linear_attn", "out_proj"], pin),
                    e83WrapLinear(model, ["linear_attn", "in_proj_qkv"], pin),
                    e83WrapLinear(model, ["linear_attn", "in_proj_z"], pin),
                    e83WrapLinear(model, ["linear_attn", "in_proj_b"], pin),
                    e83WrapLinear(model, ["linear_attn", "in_proj_a"], pin),
                ]
                return { restores.reversed().forEach { $0() } }
            }

        default:
            return nil
        }
    }
}

// MARK: - module surgery

/// Slice to `pinRows`, run the family, then rebuild the caller's width from row
/// zero. Identical launches for every pin width, so the null control subtracts
/// the wrapper exactly.
@inline(__always)
private func e83Pinned(
    _ x: MLXArray, pinRows: Int, _ body: (MLXArray) -> MLXArray
) -> MLXArray {
    let rows = x.dim(1)
    let take = min(max(pinRows, 1), rows)
    let y = body(x[0..., ..<take])
    return broadcast(y[0..., ..<1], to: [y.dim(0), rows, y.dim(2)]).contiguous()
}

private final class E83PinnedUnary: Module, UnaryLayer {
    let inner: Module
    let pinRows: Int

    init(inner: Module, pinRows: Int) {
        self.inner = inner
        self.pinRows = pinRows
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        e83Pinned(x, pinRows: pinRows) { (inner as! UnaryLayer)($0) }
    }
}

/// Shares the wrapped layer's weight, scales and biases arrays, so installing it
/// costs no extra resident bytes and cannot perturb the memory profile.
private final class E83PinnedQuantizedLinear: QuantizedLinear {
    let pinRows: Int

    init(_ other: QuantizedLinear, pinRows: Int) {
        self.pinRows = pinRows
        super.init(
            weight: other.weight, bias: other.bias, scales: other.scales,
            biases: other.biases, groupSize: other.groupSize, bits: other.bits,
            mode: other.mode)
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        e83Pinned(x, pinRows: pinRows) { super.callAsFunction($0) }
    }
}

private func e83DecoderLayers(_ model: Qwen35TextModel) -> [Module] {
    guard case .array(let items)? = model.model.children()["layers"] else { return [] }
    return items.compactMap { item in
        if case .value(let module) = item { return module }
        return nil
    }
}

private func e83Child(_ module: Module, _ key: String) -> Module? {
    guard case .value(let child)? = module.children()[key] else { return nil }
    return child
}

private func e83Replace(_ parent: Module, _ key: String, _ replacement: Module) {
    var update = NestedDictionary<String, Module>()
    update[key] = .value(replacement)
    parent.update(modules: update)
}

private func e83Resolve(
    _ layer: Module, _ childPath: [String]
) -> (parent: Module, key: String, original: Module)? {
    var parent = layer
    for key in childPath.dropLast() {
        guard let next = e83Child(parent, key) else { return nil }
        parent = next
    }
    guard let key = childPath.last, let original = e83Child(parent, key) else { return nil }
    return (parent, key, original)
}

private func e83WrapUnary(
    _ model: Qwen35TextModel, _ childPath: [String], _ pinRows: Int
) -> () -> Void {
    var restores: [() -> Void] = []
    for layer in e83DecoderLayers(model) {
        guard let (parent, key, original) = e83Resolve(layer, childPath) else { continue }
        e83Replace(parent, key, E83PinnedUnary(inner: original, pinRows: pinRows))
        restores.append { e83Replace(parent, key, original) }
    }
    return { restores.forEach { $0() } }
}

private func e83WrapLinear(
    _ model: Qwen35TextModel, _ childPath: [String], _ pinRows: Int
) -> () -> Void {
    var restores: [() -> Void] = []
    for layer in e83DecoderLayers(model) {
        guard let (parent, key, original) = e83Resolve(layer, childPath),
            let quantized = original as? QuantizedLinear
        else { continue }
        e83Replace(parent, key, E83PinnedQuantizedLinear(quantized, pinRows: pinRows))
        restores.append { e83Replace(parent, key, original) }
    }
    return { restores.forEach { $0() } }
}

// MARK: - isolated roofline

private struct E83Shape {
    let family: String
    let m: Int
    let k: Int
    let n: Int
    let layers: Int
}

/// Every quantized GEMM the 512-token seed executes, at its exact scored shape
/// and multiplicity, plus the two shapes the brief asked about that prefill does
/// NOT run: the fused `5120 -> 34816` gate_up and the M = 1 residue every pinned
/// arm still pays.
private func e83IsolatedShapes(seed: Int) -> [E83Shape] {
    var cells: [E83Shape] = [
        E83Shape(family: "gdn.in_proj_qkv", m: seed, k: 5120, n: 10240, layers: 48),
        E83Shape(family: "gdn.in_proj_z", m: seed, k: 5120, n: 6144, layers: 48),
        E83Shape(family: "gdn.in_proj_b", m: seed, k: 5120, n: 48, layers: 48),
        E83Shape(family: "gdn.in_proj_a", m: seed, k: 5120, n: 48, layers: 48),
        E83Shape(family: "gdn.out_proj", m: seed, k: 6144, n: 5120, layers: 48),
        E83Shape(family: "fa.qkv_packed", m: seed, k: 5120, n: 14336, layers: 16),
        E83Shape(family: "fa.o_proj", m: seed, k: 6144, n: 5120, layers: 16),
        E83Shape(family: "mlp.gate_proj", m: seed, k: 5120, n: 17408, layers: 64),
        E83Shape(family: "mlp.up_proj", m: seed, k: 5120, n: 17408, layers: 64),
        E83Shape(family: "mlp.down_proj", m: seed, k: 17408, n: 5120, layers: 64),
        E83Shape(family: "lm_head.tail_row", m: 1, k: 5120, n: 248320, layers: 1),
        E83Shape(family: "mlp.gate_up_fused_unused", m: seed, k: 5120, n: 34816, layers: 64),
    ]
    // The M = 1 residue of every pinned family: what a pinned arm still pays.
    for family in ["mlp.gate_proj", "mlp.down_proj", "gdn.in_proj_qkv", "gdn.out_proj"] {
        guard let base = cells.first(where: { $0.family == family }) else { continue }
        cells.append(
            E83Shape(
                family: base.family + ".pin1", m: 1, k: base.k, n: base.n,
                layers: base.layers))
    }
    return cells
}

private struct E83QuantWeight {
    let w: MLXArray
    let scales: MLXArray
    let biases: MLXArray
}

private func e83QuantWeight(k: Int, n: Int, bits: Int = 4, group: Int = 64) -> E83QuantWeight {
    let words = k / (32 / bits)
    let tile = (0..<words).map { index -> UInt32 in
        UInt32(truncatingIfNeeded: index &* 2_654_435_761) ^ 0x9E37_79B9
    }
    let w = MLXArray(tile).reshaped([1, words]) + arange(0, n, dtype: .uint32).reshaped([n, 1])
    let groups = k / group
    let rowJitter = arange(0, n, dtype: .float32).reshaped([n, 1]) * 1e-6
    let scaleTile: [Float] = (0..<groups).map { i in
        0.006 + 0.004 * Float((i &* 37) % 61) / 61.0
    }
    let biasTile: [Float] = (0..<groups).map { i in
        -0.05 - 0.02 * Float((i &* 23) % 53) / 53.0
    }
    let scales = (MLXArray(scaleTile).reshaped([1, groups]) + rowJitter).asType(.bfloat16)
    let biases = (MLXArray(biasTile).reshaped([1, groups]) + rowJitter).asType(.bfloat16)
    let weight = E83QuantWeight(w: w, scales: scales, biases: biases)
    eval(weight.w, weight.scales, weight.biases)
    return weight
}

private func e83Activations(m: Int, k: Int) -> MLXArray {
    let tile: [Float] = (0..<k).map { i in Float((i &* 131) % 251) / 251.0 - 0.5 }
    let rowJitter = arange(0, m, dtype: .float32).reshaped([m, 1]) * 0.01
    let x = (MLXArray(tile).reshaped([1, k]) + rowJitter).asType(.bfloat16)
    eval(x)
    return x
}

private func e83MeasureQuantizedShape(_ cell: E83Shape, reps: Int) -> [String: Any] {
    let entryTemp = e83GPUTemperature()
    let weight = e83QuantWeight(k: cell.k, n: cell.n)
    let x = e83Activations(m: cell.m, k: cell.k)
    func call() -> MLXArray {
        quantizedMM(
            x, weight.w, scales: weight.scales, biases: weight.biases,
            transpose: true, groupSize: 64, bits: 4)
    }
    for _ in 0..<3 { eval(call()) }
    var samples: [Double] = []
    for _ in 0..<reps {
        let start = DispatchTime.now().uptimeNanoseconds
        eval(call())
        samples.append(Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9)
    }
    samples.sort()
    let median = samples[samples.count / 2]
    let flop = 2.0 * Double(cell.m) * Double(cell.k) * Double(cell.n)
    let weightBytes = Double(cell.n * cell.k / 2 + 4 * cell.n * cell.k / 64)
    Memory.clearCache()
    var record: [String: Any] = [
        "kind": "isolated_quantized_matmul",
        "family": cell.family,
        "m": cell.m, "k": cell.k, "n": cell.n, "layers": cell.layers,
        "reps": reps,
        "seconds": samples,
        "seconds_median": median,
        "seconds_min": samples[0],
        "seconds_max": samples[samples.count - 1],
        "flop_per_call": flop,
        "tflop_per_second": flop / median / 1e12,
        "weight_bytes_per_call": weightBytes,
        "gb_per_second": weightBytes / median / 1e9,
        "modelled_prefill_seconds": median * Double(cell.layers),
    ]
    if let entryTemp { record["gpu_temp_entry_c"] = entryTemp }
    if let exit = e83GPUTemperature() { record["gpu_temp_exit_c"] = exit }
    return record
}

private func e83MeasureSdpa(seed: Int, reps: Int) -> [String: Any] {
    let qHeads = 24, kvHeads = 4, headDim = 256
    let scale = 1.0 / Foundation.sqrt(Float(headDim))
    let q = MLXArray.zeros([1, qHeads, seed, headDim], dtype: .bfloat16)
    let k = MLXArray.zeros([1, kvHeads, seed, headDim], dtype: .bfloat16)
    let v = MLXArray.zeros([1, kvHeads, seed, headDim], dtype: .bfloat16)
    eval(q, k, v)
    func call() -> MLXArray {
        MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: .causal)
    }
    for _ in 0..<3 { eval(call()) }
    var samples: [Double] = []
    for _ in 0..<reps {
        let start = DispatchTime.now().uptimeNanoseconds
        eval(call())
        samples.append(Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9)
    }
    samples.sort()
    let median = samples[samples.count / 2]
    let flop = 2.0 * 2.0 * Double(qHeads) * Double(seed) * Double(seed) * Double(headDim) / 2.0
    Memory.clearCache()
    return [
        "kind": "isolated_sdpa",
        "family": "fa.sdpa_causal",
        "q_heads": qHeads, "kv_heads": kvHeads, "length": seed, "head_dim": headDim,
        "layers": 16,
        "reps": reps,
        "seconds": samples,
        "seconds_median": median,
        "flop_per_call": flop,
        "tflop_per_second": flop / median / 1e12,
        "modelled_prefill_seconds": median * 16.0,
    ]
}

// MARK: - dispatch and command-buffer census

private final class E83DispatchLedger: @unchecked Sendable {
    static let shared = E83DispatchLedger()

    private let lock = NSLock()
    private var pipelineNames: [ObjectIdentifier: String] = [:]
    private var encoderBinding: [ObjectIdentifier: String] = [:]
    private var counts: [String: Int] = [:]
    private var dispatches = 0
    private var commits = 0
    private var recording = false

    func note(pipeline: AnyObject, name: String) {
        lock.lock()
        pipelineNames[ObjectIdentifier(pipeline)] = name
        lock.unlock()
    }

    func bind(encoder: AnyObject, pipeline: AnyObject) {
        lock.lock()
        encoderBinding[ObjectIdentifier(encoder)] =
            pipelineNames[ObjectIdentifier(pipeline)] ?? "<unmapped>"
        lock.unlock()
    }

    func dispatch(encoder: AnyObject) {
        lock.lock()
        if recording {
            let name = encoderBinding[ObjectIdentifier(encoder)] ?? "<unbound>"
            counts[name, default: 0] += 1
            dispatches += 1
        }
        lock.unlock()
    }

    func commit() {
        lock.lock()
        if recording { commits += 1 }
        lock.unlock()
    }

    func start() {
        lock.lock()
        counts = [:]
        dispatches = 0
        commits = 0
        recording = true
        lock.unlock()
    }

    /// Cumulative counts so far, so a caller can difference two snapshots.
    func snapshot() -> [String: Any] {
        lock.lock()
        let value: [String: Any] = [
            "dispatches": dispatches,
            "command_buffer_commits": commits,
            "kernels": counts,
        ]
        lock.unlock()
        return value
    }

    func stop() -> [String: Any] {
        let value = snapshot()
        lock.lock()
        recording = false
        lock.unlock()
        return value
    }
}

private typealias E83DispatchIMP = @convention(c) (AnyObject, Selector, MTLSize, MTLSize) -> Void
private typealias E83SetPipelineIMP = @convention(c) (AnyObject, Selector, AnyObject) -> Void
private typealias E83VoidIMP = @convention(c) (AnyObject, Selector) -> Void
private typealias E83NewPipelineIMP = @convention(c) (
    AnyObject, Selector, AnyObject, UnsafeMutableRawPointer?
) -> UnsafeMutableRawPointer?

private func e83SwizzleDispatch(_ cls: AnyClass, _ name: String) -> Bool {
    let selector = NSSelectorFromString(name)
    guard let method = class_getInstanceMethod(cls, selector) else { return false }
    let original = unsafeBitCast(method_getImplementation(method), to: E83DispatchIMP.self)
    let replacement: @convention(block) (AnyObject, MTLSize, MTLSize) -> Void = {
        encoder, grid, threadgroup in
        E83DispatchLedger.shared.dispatch(encoder: encoder)
        original(encoder, selector, grid, threadgroup)
    }
    method_setImplementation(method, imp_implementationWithBlock(replacement))
    return true
}

private func e83SwizzleSetPipeline(_ cls: AnyClass) -> Bool {
    let selector = NSSelectorFromString("setComputePipelineState:")
    guard let method = class_getInstanceMethod(cls, selector) else { return false }
    let original = unsafeBitCast(method_getImplementation(method), to: E83SetPipelineIMP.self)
    let replacement: @convention(block) (AnyObject, AnyObject) -> Void = { encoder, pipeline in
        E83DispatchLedger.shared.bind(encoder: encoder, pipeline: pipeline)
        original(encoder, selector, pipeline)
    }
    method_setImplementation(method, imp_implementationWithBlock(replacement))
    return true
}

private func e83SwizzleCommit(_ cls: AnyClass) -> Bool {
    let selector = NSSelectorFromString("commit")
    guard let method = class_getInstanceMethod(cls, selector) else { return false }
    let original = unsafeBitCast(method_getImplementation(method), to: E83VoidIMP.self)
    let replacement: @convention(block) (AnyObject) -> Void = { buffer in
        E83DispatchLedger.shared.commit()
        original(buffer, selector)
    }
    method_setImplementation(method, imp_implementationWithBlock(replacement))
    return true
}

private func e83SwizzleNewPipeline(_ cls: AnyClass) -> Bool {
    let selector = NSSelectorFromString("newComputePipelineStateWithFunction:error:")
    guard let method = class_getInstanceMethod(cls, selector) else { return false }
    let original = unsafeBitCast(method_getImplementation(method), to: E83NewPipelineIMP.self)
    let replacement: @convention(block) (AnyObject, AnyObject, UnsafeMutableRawPointer?)
        -> UnsafeMutableRawPointer? = { device, function, errorOut in
            let result = original(device, selector, function, errorOut)
            if let result {
                let pipeline = Unmanaged<AnyObject>.fromOpaque(result).takeUnretainedValue()
                let name = (function as? MTLFunction)?.name ?? "<unnamed>"
                E83DispatchLedger.shared.note(pipeline: pipeline, name: name)
            }
            return result
        }
    method_setImplementation(method, imp_implementationWithBlock(replacement))
    return true
}

nonisolated(unsafe) private var e83SwizzlesInstalled = false

private func e83InstallSwizzles() -> Bool {
    if e83SwizzlesInstalled { return true }
    guard let device = MTLCreateSystemDefaultDevice(),
        let queue = device.makeCommandQueue(),
        let buffer = queue.makeCommandBuffer(),
        let encoder = buffer.makeComputeCommandEncoder()
    else { return false }
    let encoderClass: AnyClass = type(of: encoder as AnyObject)
    let deviceClass: AnyClass = type(of: device as AnyObject)
    let bufferClass: AnyClass = type(of: buffer as AnyObject)
    encoder.endEncoding()

    var ok = e83SwizzleNewPipeline(deviceClass)
    ok = e83SwizzleSetPipeline(encoderClass) && ok
    ok = e83SwizzleDispatch(encoderClass, "dispatchThreadgroups:threadsPerThreadgroup:") && ok
    ok = e83SwizzleDispatch(encoderClass, "dispatchThreads:threadsPerThreadgroup:") && ok
    ok = e83SwizzleCommit(bufferClass) && ok
    e83SwizzlesInstalled = ok
    return ok
}

// MARK: - support

private func e83LoadPromptTokens(_ path: String) throws -> [Int] {
    let data = try Data(contentsOf: URL(fileURLWithPath: path))
    guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any],
        let cases = object["cases"] as? [[String: Any]],
        let first = cases.first,
        let prompt = first["prompt_tokens"] as? [Int]
    else {
        throw E83Failure("E83: \(path) is not a correctness prompt fixture")
    }
    return prompt
}

private struct E83Failure: Error, CustomStringConvertible {
    let description: String
    init(_ description: String) { self.description = description }
}

/// One macmon sample. The census runs with `MLXFAST_LOCAL_COOL_GATE=0`, so the
/// entry and exit temperature of every block is the thermal record.
private func e83GPUTemperature() -> Double? {
    let binary =
        ProcessInfo.processInfo.environment["MLXFAST_E83_MACMON"] ?? "/opt/homebrew/bin/macmon"
    guard FileManager.default.isExecutableFile(atPath: binary) else { return nil }
    let process = Process()
    process.executableURL = URL(fileURLWithPath: binary)
    process.arguments = ["pipe", "-s1"]
    let pipe = Pipe()
    process.standardOutput = pipe
    process.standardError = FileHandle.nullDevice
    do { try process.run() } catch { return nil }
    let data = pipe.fileHandleForReading.readDataToEndOfFile()
    process.waitUntilExit()
    for line in String(decoding: data, as: UTF8.self).split(separator: "\n") {
        guard let object = try? JSONSerialization.jsonObject(with: Data(line.utf8)),
            let root = object as? [String: Any],
            let temp = root["temp"] as? [String: Any],
            let gpu = temp["gpu_temp_avg"] as? Double
        else { continue }
        return gpu
    }
    return nil
}

private func e83Device() -> [String: Any] {
    let device = MLX.GPU.deviceInfo()
    return [
        "architecture": device.architecture,
        "max_buffer_size": device.maxBufferSize,
        "max_recommended_working_set_size": Int(device.maxRecommendedWorkingSetSize),
        "memory_size": device.memorySize,
    ]
}

/// Stream one block to stdout as it completes, so a long session is observable
/// and a crash does not lose everything measured before it.
@discardableResult
private func e83Emit(_ record: [String: Any]) -> [String: Any] {
    let json =
        (try? JSONSerialization.data(withJSONObject: record, options: [.sortedKeys]))
        .flatMap { String(data: $0, encoding: .utf8) } ?? "{}"
    print("E83_BLOCK \(json)")
    fflush(stdout)
    return record
}
