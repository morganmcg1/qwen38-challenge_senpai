import Foundation
import MLX
@testable import MLXLLM
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXFastModel

// E169 -- in-situ census of the marginal verified row.
//
// PROVENANCE. This suite is the E71 width-tax census, deleted in `92b8fe70`
// ("E116 cleanup") and revived here for E169. The only changes are the removed
// `E58DispatchCensus` hooks, whose implementation went in the same commit, and
// the `E169` environment prefix. The measurement design is unchanged, so E71
// results and E169 results are directly comparable on a matched host.
//
// THE QUESTION. A verify round reads the same 14,412,349,440 bytes of quantized
// backbone weight whether it checks 1 row or 9. E1 still measured
// `T(6) - T(1) = 69.659 ms`. This suite attributes that width tax to kernel
// families IN SITU, at real occupancy, with the real checkpoint resident.
//
// WHAT E169 ADDS. The `baseline` width curve is the in-situ row law of the
// verify forward alone: fit `T(M) = a + b*M + c*(ceil(M/5) - 1)` and `b` is the
// marginal verified row. Each family arm then prices one family's contribution
// to `b`, and the sum of the arms against `b` is the closure gap.
//
// THE METHOD. For a family `F`, run `F` at width 1 while every other family
// keeps its exact width-M shape schedule. The data dependency is broken by
// slicing `F`'s input to one row and broadcasting `F`'s output back to M rows,
// so `tax(F) = T(M) - T(M, F pinned to 1)`.
//
// The tokens every arm produces are WRONG BY CONSTRUCTION. That is the design,
// not a defect: this is a timing-only instrument. Nothing here is on the
// submitted surface -- Yukon never packages `Tests/`.
//
// The wrapper is symmetric on purpose. Both the pinned arm and the null control
// slice, take row 0, broadcast to M rows and materialise. The ONLY difference
// between them is the width the wrapped family itself sees, so the wrapper's
// own launches, copies and host indirection cancel in the subtraction.
//
// WHAT IS NOT REACHABLE FROM HERE, and why the closure gap has a named floor:
// `Qwen35Attention` and `Qwen35GatedDeltaNet` are `final class`, and their
// q/k/v and in-projection weights are FUSED into raw `quantizedMM` calls that
// never dispatch through the child `Linear` (`Qwen35.swift:677-689`, `:1707-1712`).
// Those families, SDPA, the GDN scan and every compiled fusion can only be
// pinned by editing `Vendor/mlx-swift-lm/.../Qwen35.swift`, which ships. They
// are therefore reported as unattributable-by-construction, not as measurement
// failure.
//
// Enable with `MLXFAST_RUN_E169_WIDTH_TAX=1` and point `MLXFAST_E169_OUT` at the
// JSON destination.

@Suite(.serialized)
struct E169WidthTaxCensusTests {
    private static var enabled: Bool {
        ProcessInfo.processInfo.environment["MLXFAST_RUN_E169_WIDTH_TAX"] == "1"
    }

    @Test(.enabled(if: E169WidthTaxCensusTests.enabled))
    func censusTheWidthTax() throws {
        let env = ProcessInfo.processInfo.environment
        let outPath = try #require(
            env["MLXFAST_E169_OUT"], "MLXFAST_E169_OUT must name the JSON destination")

        let weightsPath = env["MLXFAST_E169_WEIGHTS"] ?? "weights"
        let promptPath =
            env["MLXFAST_E169_PROMPT"]
            ?? "correctness_prompts/public_longcopy_gate_english_512_1024.json"
        let seedLength = Int(env["MLXFAST_E169_SEED_LEN"] ?? "") ?? 768
        let reps = Int(env["MLXFAST_E169_REPS"] ?? "") ?? 12
        let warmup = Int(env["MLXFAST_E169_WARMUP"] ?? "") ?? 3
        let curveWidths = parseWidths(env["MLXFAST_E169_CURVE_WIDTHS"]) ?? Array(1...9)
        // M=6 is the headline; M=4 and M=9 are the assignment's shape checks.
        // M=5 is added because it is the last width BELOW the SDPA split: the
        // helper cuts 6..9 query rows into two <=5-row calls, so M=5 against
        // M=6 is the only pair that isolates that branch, and M=5 also carries
        // 0.241 of the ranked width mixture against M=6's 0.334.
        let armWidths = parseWidths(env["MLXFAST_E169_ARM_WIDTHS"]) ?? [4, 5, 6, 9]
        let armNames =
            (env["MLXFAST_E169_ARMS"]?.split(separator: ",").map {
                $0.trimmingCharacters(in: .whitespaces)
            }).flatMap { $0.isEmpty ? nil : $0 }
            ?? [
                "null", "lm_head", "mlp_all", "mlp_down", "fa_o_proj",
                "gdn_out_proj", "all_interceptable",
            ]

        let tokens = try loadPromptTokens(promptPath)
        #expect(tokens.count >= seedLength + 256)

        let config = try Qwen35Config.load(from: weightsPath)
        let loader = try Qwen35WeightLoader(weightsPath: weightsPath)
        let loadStart = DispatchTime.now().uptimeNanoseconds
        let runtime = Qwen35RuntimeWeightCache(loader: loader, config: config)
        let model = try runtime.requireLibraryModel()
        let loadSeconds =
            Double(DispatchTime.now().uptimeNanoseconds - loadStart) / 1e9

        let harness = E169Harness(
            model: model, tokens: tokens, seedLength: seedLength,
            reps: reps, warmup: warmup)

        // One resident model, one process, one thermal history. Every block
        // below is timed inside it, and the arm order is ABBA-counterbalanced
        // so monotone drift cancels to first order.
        var blocks: [[String: Any]] = []

        // Rung 1 -- the harness gate. The width curve on the CURRENT base.
        for width in curveWidths {
            blocks.append(harness.run(arm: "baseline", width: width, order: blocks.count))
        }
        for width in curveWidths.reversed() {
            blocks.append(harness.run(arm: "baseline", width: width, order: blocks.count))
        }

        // Rungs 2 and 3 -- controls first, then the family arms, ABBA within
        // each (arm, width) pair: baseline, arm, arm, baseline.
        for width in armWidths {
            for name in armNames {
                guard let arm = e169Arm(name) else {
                    Issue.record("unknown E169 arm \(name)")
                    continue
                }
                blocks.append(harness.run(arm: "baseline", width: width, order: blocks.count))
                blocks.append(harness.run(arm: arm, width: width, order: blocks.count))
                blocks.append(harness.run(arm: arm, width: width, order: blocks.count))
                blocks.append(harness.run(arm: "baseline", width: width, order: blocks.count))
            }
        }

        let payload: [String: Any] = [
            "schema": 1,
            "experiment": ProcessInfo.processInfo.environment["MLXFAST_CENSUS_EXPERIMENT"]
                ?? "e169-in-situ-width-tax-census",
            "harness": "local",
            "cool_gate_passed_real_gate": false,
            "gate_qualified_for_timing": false,
            "official_or_ranked_score": false,
            "identity": [
                "weights_path": weightsPath,
                "prompt_path": promptPath,
                "seed_length": seedLength,
                "reps_per_block": reps,
                "warmup_per_block": warmup,
                "curve_widths": curveWidths,
                "arm_widths": armWidths,
                "arms": armNames,
                "model_load_seconds": loadSeconds,
                "num_hidden_layers": config.numHiddenLayers,
                "hidden_size": config.hiddenSize,
                "vocab_size": config.vocabSize,
                "device": describeE169Device(),
                "host": ProcessInfo.processInfo.hostName,
            ],
            "blocks": blocks,
        ]
        let data = try JSONSerialization.data(
            withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: URL(fileURLWithPath: outPath))
        print("E169_OUT \(outPath)")
    }
}

// MARK: - harness

/// One resident model, replayed as a sequence of matched timing blocks.
private final class E169Harness {
    let model: Qwen35TextModel
    let tokens: [Int32]
    let seedLength: Int
    let reps: Int
    let warmup: Int

    init(model: Qwen35TextModel, tokens: [Int], seedLength: Int, reps: Int, warmup: Int) {
        self.model = model
        self.tokens = tokens.map(Int32.init)
        self.seedLength = seedLength
        self.reps = reps
        self.warmup = warmup
    }

    func run(arm: String, width: Int, order: Int) -> [String: Any] {
        run(arm: e169Arm(arm) ?? E169Arm.baseline(), width: width, order: order)
    }

    func run(arm: E169Arm, width: Int, order: Int) -> [String: Any] {
        let entryTemp = e169GPUTemperature()
        let restore = arm.install(model, width)
        defer { restore() }

        var caches = model.newCache(parameters: nil)
        let seed = MLXArray(Array(tokens[0..<seedLength])).reshaped([1, seedLength])
        let (seedLogits, _) = model.callWithHidden(
            input: LMInput.Text(tokens: seed), cache: caches, nConfirmed: 0)
        eval(caches.flatMap { $0.state } + [seedLogits])

        var cursor = seedLength
        let klStart = caches.map(\.offset).max() ?? seedLength

        for _ in 0..<warmup {
            _ = round(width: width, cursor: &cursor, caches: caches)
        }

        var samples: [Double] = []
        samples.reserveCapacity(reps)
        for rep in 0..<reps {
            let start = DispatchTime.now().uptimeNanoseconds
            _ = round(width: width, cursor: &cursor, caches: caches)
            samples.append(Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9)
        }
        let klEnd = caches.map(\.offset).max() ?? cursor

        // The caches hold one full-attention KV window plus 48 fp32 recurrent
        // states. Drop them before the next block so peak residency is one
        // model plus one cache, not one model plus every block's cache.
        caches = []
        Memory.clearCache()

        let sorted = samples.sorted()
        let exitTemp = e169GPUTemperature()
        var record: [String: Any] = [
            "arm": arm.name,
            "width": width,
            "order": order,
            "pin_rows": arm.pinRows(width),
            "reps": reps,
            "warmup": warmup,
            "seconds": sorted,
            "seconds_median": sorted[sorted.count / 2],
            "seconds_min": sorted[0],
            "seconds_max": sorted[sorted.count - 1],
            "seconds_mean": sorted.reduce(0, +) / Double(sorted.count),
            "kl_start": klStart,
            "kl_end": klEnd,
        ]
        // JSONSerialization rejects NaN, so a missing macmon must drop the key
        // rather than poison the whole block record.
        if let entryTemp { record["gpu_temp_entry_c"] = entryTemp }
        if let exitTemp { record["gpu_temp_exit_c"] = exitTemp }
        print(
            "E169_BLOCK "
                + (String(
                    data: (try? JSONSerialization.data(
                        withJSONObject: record, options: [.sortedKeys])) ?? Data(),
                    encoding: .utf8) ?? "{}"))
        // stdout is block-buffered behind the W&B streamer's pipe. Without this
        // flush the blocks would only reach W&B in 4 KB batches, which is
        // logging at session end wearing a live-logging costume.
        fflush(stdout)
        return record
    }

    /// One verify round, shaped exactly like the scored session's: the same
    /// `callWithHiddenAndNormed(nConfirmed: 1)` call, the same exact top-2
    /// readout kernel pair, and the same single blocking eval over the cache
    /// roots plus the readout.
    private func round(width: Int, cursor: inout Int, caches: [any KVCache]) -> Double {
        var ids: [Int32] = []
        ids.reserveCapacity(width)
        for i in 0..<width { ids.append(tokens[(cursor + i) % tokens.count]) }
        cursor += width
        let input = MLXArray(ids).reshaped([1, width])
        let (logits, _, _) = model.callWithHiddenAndNormed(
            input: LMInput.Text(tokens: input), cache: caches, nConfirmed: 1)
        let (top2IDs, top2Values) = Qwen36MTPBlockSession.linearTopTwoRows(logits)
        eval(caches.flatMap { $0.state } + [top2IDs, top2Values])
        return 0
    }
}

// MARK: - arms

/// A pinned-width ablation: which modules get wrapped, and at what pin width.
private struct E169Arm {
    let name: String
    /// Rows the wrapped family sees. `nil` means "the caller's width", i.e. the
    /// null control: same wrapper, same launches, same copy, full-width family.
    let pin: Int?
    let wrap: (Qwen35TextModel, Int) -> () -> Void

    func pinRows(_ width: Int) -> Int { pin ?? width }

    func install(_ model: Qwen35TextModel, _ width: Int) -> () -> Void {
        wrap(model, pin ?? width)
    }

    static func baseline() -> E169Arm {
        E169Arm(name: "baseline", pin: nil) { _, _ in {} }
    }
}

private func e169Arm(_ name: String) -> E169Arm? {
    switch name {
    case "baseline":
        return .baseline()

    // Rung 2 control. The wrapper runs around `mlp` with the family at FULL
    // width, so anything it measures is harness overhead and nothing else.
    case "null":
        return E169Arm(name: "null", pin: nil) { model, pin in
            e169WrapUnary(model, childPath: ["mlp"], pinRows: pin)
        }

    // Rung 2 positive control. E63 measured this shape standalone at 92.1 % of
    // peak bandwidth at NA=4 -- the highest fraction of any scored shape -- so
    // its tax is predictable from independent data.
    case "lm_head":
        return E169Arm(name: "lm_head", pin: 1) { model, pin in
            e169WrapLMHead(model, pinRows: pin)
        }

    // Rung 3. 64 layers, 9.626 GB of the 14.412 GB weight stream: the fused
    // gate/up QMV, the compiled SwiGLU and `down_proj` together.
    case "mlp_all":
        return E169Arm(name: "mlp_all", pin: 1) { model, pin in
            e169WrapUnary(model, childPath: ["mlp"], pinRows: pin)
        }

    // 64 layers, 3.209 GB. `mlp_all` minus this isolates fused gate/up + SwiGLU.
    case "mlp_down":
        return E169Arm(name: "mlp_down", pin: 1) { model, pin in
            e169WrapQuantizedLinear(model, childPath: ["mlp", "down_proj"], pinRows: pin)
        }

    // 16 full-attention layers, 0.283 GB. The only attention projection that
    // still dispatches through its child module.
    case "fa_o_proj":
        return E169Arm(name: "fa_o_proj", pin: 1) { model, pin in
            e169WrapQuantizedLinear(model, childPath: ["self_attn", "o_proj"], pinRows: pin)
        }

    // 48 GDN layers, 0.849 GB. Same seam on the recurrent side.
    case "gdn_out_proj":
        return E169Arm(name: "gdn_out_proj", pin: 1) { model, pin in
            e169WrapQuantizedLinear(model, childPath: ["linear_attn", "out_proj"], pinRows: pin)
        }

    // Every reachable family pinned at once, 11.473 GB or 79.61 % of the weight
    // stream. This is the additivity test the closure gap needs: if it equals
    // the sum of the disjoint single-family arms, the remaining gap is families
    // this harness cannot reach; if it does not, the families interact and a
    // per-family map is incomplete on its own.
    case "all_interceptable":
        return E169Arm(name: "all_interceptable", pin: 1) { model, pin in
            let restores = [
                e169WrapUnary(model, childPath: ["mlp"], pinRows: pin),
                e169WrapQuantizedLinear(
                    model, childPath: ["self_attn", "o_proj"], pinRows: pin),
                e169WrapQuantizedLinear(
                    model, childPath: ["linear_attn", "out_proj"], pinRows: pin),
                e169WrapLMHead(model, pinRows: pin),
            ]
            return { restores.reversed().forEach { $0() } }
        }

    default:
        return nil
    }
}

// MARK: - module surgery

/// Slice to `pinRows`, run the family, then rebuild the caller's width from row
/// zero. Identical launches for every pin width, so the null control subtracts
/// the wrapper exactly.
@inline(__always)
private func e169Pinned(
    _ x: MLXArray, pinRows: Int, _ body: (MLXArray) -> MLXArray
) -> MLXArray {
    let rows = x.dim(1)
    let take = min(max(pinRows, 1), rows)
    let y = body(x[0..., ..<take])
    return broadcast(y[0..., ..<1], to: [y.dim(0), rows, y.dim(2)]).contiguous()
}

private final class E169PinnedUnary: Module, UnaryLayer {
    let inner: Module
    let pinRows: Int

    init(inner: Module, pinRows: Int) {
        self.inner = inner
        self.pinRows = pinRows
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        e169Pinned(x, pinRows: pinRows) { (inner as! UnaryLayer)($0) }
    }
}

/// Shares the wrapped layer's weight, scales and biases arrays, so installing it
/// costs no extra resident bytes and cannot perturb the memory profile.
private final class E169PinnedQuantizedLinear: QuantizedLinear {
    let pinRows: Int

    init(_ other: QuantizedLinear, pinRows: Int) {
        self.pinRows = pinRows
        super.init(
            weight: other.weight, bias: other.bias, scales: other.scales,
            biases: other.biases, groupSize: other.groupSize, bits: other.bits,
            mode: other.mode)
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        e169Pinned(x, pinRows: pinRows) { super.callAsFunction($0) }
    }
}

private func e169DecoderLayers(_ model: Qwen35TextModel) -> [Module] {
    guard case .array(let items)? = model.model.children()["layers"] else { return [] }
    return items.compactMap { item in
        if case .value(let module) = item { return module }
        return nil
    }
}

private func e169Child(_ module: Module, _ key: String) -> Module? {
    guard case .value(let child)? = module.children()[key] else { return nil }
    return child
}

private func e169Replace(_ parent: Module, _ key: String, _ replacement: Module) {
    var update = NestedDictionary<String, Module>()
    update[key] = .value(replacement)
    parent.update(modules: update)
}

/// Wrap a `Module`-typed child that the layer calls through `UnaryLayer`.
private func e169WrapUnary(
    _ model: Qwen35TextModel, childPath: [String], pinRows: Int
) -> () -> Void {
    var restores: [() -> Void] = []
    for layer in e169DecoderLayers(model) {
        guard let (parent, key, original) = e169Resolve(layer, childPath) else { continue }
        e169Replace(parent, key, E169PinnedUnary(inner: original, pinRows: pinRows))
        restores.append { e169Replace(parent, key, original) }
    }
    return { restores.forEach { $0() } }
}

private func e169WrapQuantizedLinear(
    _ model: Qwen35TextModel, childPath: [String], pinRows: Int
) -> () -> Void {
    var restores: [() -> Void] = []
    for layer in e169DecoderLayers(model) {
        guard let (parent, key, original) = e169Resolve(layer, childPath),
            let quantized = original as? QuantizedLinear
        else { continue }
        e169Replace(parent, key, E169PinnedQuantizedLinear(quantized, pinRows: pinRows))
        restores.append { e169Replace(parent, key, original) }
    }
    return { restores.forEach { $0() } }
}

private func e169WrapLMHead(_ model: Qwen35TextModel, pinRows: Int) -> () -> Void {
    guard let head = e169Child(model, "lm_head") as? QuantizedLinear else { return {} }
    e169Replace(model, "lm_head", E169PinnedQuantizedLinear(head, pinRows: pinRows))
    return { e169Replace(model, "lm_head", head) }
}

/// Walk `childPath` from a decoder layer and return the parent that owns the
/// last component, so the replacement lands on a `@ModuleInfo` setter.
private func e169Resolve(
    _ layer: Module, _ childPath: [String]
) -> (parent: Module, key: String, original: Module)? {
    var parent = layer
    for key in childPath.dropLast() {
        guard let next = e169Child(parent, key) else { return nil }
        parent = next
    }
    guard let key = childPath.last, let original = e169Child(parent, key) else { return nil }
    return (parent, key, original)
}

// MARK: - support

private func parseWidths(_ raw: String?) -> [Int]? {
    guard let raw, !raw.isEmpty else { return nil }
    let widths = raw.split(separator: ",").compactMap {
        Int($0.trimmingCharacters(in: .whitespaces))
    }
    return widths.isEmpty ? nil : widths
}

private func loadPromptTokens(_ path: String) throws -> [Int] {
    let data = try Data(contentsOf: URL(fileURLWithPath: path))
    guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any],
        let cases = object["cases"] as? [[String: Any]],
        let first = cases.first,
        let prompt = first["prompt_tokens"] as? [Int],
        let expected = first["expected_tokens"] as? [Int]
    else {
        throw E169Failure("E169: \(path) is not a correctness prompt fixture")
    }
    return prompt + expected
}

/// One macmon sample. The census runs with `MLXFAST_LOCAL_COOL_GATE=0`, so the
/// entry and exit temperature of every block is the thermal record.
private func e169GPUTemperature() -> Double? {
    let binary =
        ProcessInfo.processInfo.environment["MLXFAST_E169_MACMON"] ?? "/opt/homebrew/bin/macmon"
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

private struct E169Failure: Error, CustomStringConvertible {
    let description: String
    init(_ description: String) { self.description = description }
}

// MARK: - E169 bottom-up estimator

// The in-situ census above cannot reach `linear_attn.in_proj`, `self_attn.qkv`,
// SDPA or the gated-delta scan: those weights are fused into raw `quantizedMM`
// calls inside `final class` modules. This second estimator prices the same
// forward from below instead. Every wide QMV shape of one verify forward is
// timed THROUGH THE SCORED ROUTED ENTRY POINT `qwen35RoutedQuantizedMM`, which
// is what the worker calls, and then weighted by that shape's call count per
// forward: 48 + 48 + 16 + 16 + 64 + 64 + 1 = 257 wide QMV calls.
//
// TWO KNOWN BIASES, both in the intercept rather than the slope:
//   1. A synthetic sweep reuses one weight tensor, so a small shape can sit in
//      the system cache where the in-situ call streams it cold. Weight traffic
//      does not change with width, so this moves `a`, not `b`.
//   2. In situ the fused-norm producer often publishes the chunk-sum table
//      (`Qwen35XSumsSidecar`), and a standalone sweep always pays the fill,
//      measured at 4 to 6 us per call and flat in width.
// Both are reported, not hidden. The quantity E169 needs is the width slope.

private struct E169Shape {
    let name: String
    let k: Int
    let n: Int
    let callsPerVerify: Int
    let family: String
}

private let e169ScoredShapes: [E169Shape] = [
    .init(name: "linear_attn.in_proj_fused_qkvzba", k: 5120, n: 16480,
        callsPerVerify: 48, family: "gdn"),
    .init(name: "linear_attn.out_proj", k: 6144, n: 5120,
        callsPerVerify: 48, family: "gdn"),
    .init(name: "full_attn.qkv_proj_fused", k: 5120, n: 14336,
        callsPerVerify: 16, family: "attn"),
    .init(name: "full_attn.o_proj", k: 6144, n: 5120,
        callsPerVerify: 16, family: "attn"),
    .init(name: "mlp.gate_up_fused", k: 5120, n: 34816,
        callsPerVerify: 64, family: "mlp"),
    .init(name: "mlp.down", k: 17408, n: 5120,
        callsPerVerify: 64, family: "mlp"),
    .init(name: "head.lm_head", k: 5120, n: 248320,
        callsPerVerify: 1, family: "lm_head"),
]

@Suite(.serialized)
struct E169BottomUpShapeCurveTests {
    private static var enabled: Bool {
        ProcessInfo.processInfo.environment["MLXFAST_RUN_E169_BOTTOM_UP"] == "1"
    }

    @Test(.enabled(if: E169BottomUpShapeCurveTests.enabled))
    func priceEveryRoutedShapeOverWidth() throws {
        let env = ProcessInfo.processInfo.environment
        let outPath = try #require(
            env["MLXFAST_E169_BOTTOM_UP_OUT"],
            "MLXFAST_E169_BOTTOM_UP_OUT must name the JSON destination")
        let widths = parseWidths(env["MLXFAST_E169_BOTTOM_UP_WIDTHS"]) ?? Array(1...9)
        let reps = Int(env["MLXFAST_E169_BOTTOM_UP_REPS"] ?? "") ?? 9
        let inner = Int(env["MLXFAST_E169_BOTTOM_UP_INNER"] ?? "") ?? 8

        var payload: [String: Any] = [
            "experiment": "e169-bottom-up-routed-shape-curve",
            "harness": "local",
            "entry_point": "qwen35RoutedQuantizedMM",
            "widths": widths,
            "reps": reps,
            "inner_calls_per_timed_region": inner,
            "device": describeE169Device(),
            "custom_qmv_arm": String(describing: Qwen35CustomQMV.arm),
        ]
        if let temp = e169GPUTemperature() { payload["gpu_temp_entry_c"] = temp }

        var shapeRecords: [[String: Any]] = []
        for shape in e169ScoredShapes {
            shapeRecords.append(
                e169SweepRoutedShape(shape, widths: widths, reps: reps, inner: inner))
            Memory.clearCache()
        }
        payload["shapes"] = shapeRecords
        payload["gated_delta_recurrence"] =
            e169SweepGatedDelta(widths: widths, reps: reps, inner: inner)
        payload["top_two_readout"] =
            e169SweepTopTwo(widths: widths, reps: reps, inner: max(2, inner / 4))
        if let temp = e169GPUTemperature() { payload["gpu_temp_exit_c"] = temp }

        let data = try JSONSerialization.data(
            withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: URL(fileURLWithPath: outPath))
        print("E169_BOTTOM_UP_OUT \(outPath)")
    }
}

private struct E169QuantWeight {
    let w: MLXArray
    let scales: MLXArray
    let biases: MLXArray
}

/// Deterministic packed affine-4 g64 weights at the scored `[n, k]`. Values do
/// not change kernel time; shape, dtype and contiguity do.
private func e169QuantWeight(k: Int, n: Int) -> E169QuantWeight {
    let words = k / 8
    let tile = (0..<words).map { index -> UInt32 in
        UInt32(truncatingIfNeeded: index &* 2_654_435_761) ^ 0x9E37_79B9
    }
    let w = MLXArray(tile).reshaped([1, words]) + arange(0, n, dtype: .uint32).reshaped([n, 1])
    let groups = k / 64
    let jitter = arange(0, n, dtype: .float32).reshaped([n, 1]) * 1e-6
    let scaleTile: [Float] = (0..<groups).map { 0.006 + 0.004 * Float(($0 &* 37) % 61) / 61.0 }
    let biasTile: [Float] = (0..<groups).map { -0.05 - 0.02 * Float(($0 &* 23) % 53) / 53.0 }
    let scales = (MLXArray(scaleTile).reshaped([1, groups]) + jitter).asType(.bfloat16)
    let biases = (MLXArray(biasTile).reshaped([1, groups]) + jitter).asType(.bfloat16)
    eval(w, scales, biases)
    return E169QuantWeight(w: w, scales: scales, biases: biases)
}

private func e169Activations(m: Int, k: Int, salt: Int) -> MLXArray {
    let row = (0..<k).map { index -> Float in
        Float(((index &* 2_246_822_519) &+ salt &* 374_761_393) % 1_009) / 1_009.0 - 0.5
    }
    let base = MLXArray(row).reshaped([1, 1, k])
    let ramp = (arange(0, m, dtype: .float32) * 1e-3).reshaped([1, m, 1])
    let x = (base + ramp).asType(.bfloat16)
    eval(x)
    return x
}

/// Median seconds per call over `reps` timed regions of `inner` chained calls.
/// The chain matters: MLX encodes independent kernels concurrently, which would
/// understate the cost of the strictly dependent scored chain.
private func e169MedianSecondsPerCall(
    reps: Int, inner: Int, warmup: Int = 3, _ body: () -> [MLXArray]
) -> [Double] {
    for _ in 0..<warmup { eval(body()) }
    var samples: [Double] = []
    samples.reserveCapacity(reps)
    for _ in 0..<reps {
        let start = DispatchTime.now().uptimeNanoseconds
        eval(body())
        let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
        samples.append(elapsed / Double(inner))
    }
    return samples.sorted()
}

private func e169SweepRoutedShape(
    _ shape: E169Shape, widths: [Int], reps: Int, inner: Int
) -> [String: Any] {
    let weight = e169QuantWeight(k: shape.k, n: shape.n)
    var rows: [[String: Any]] = []

    // Ascending then descending, pooled: monotone thermal drift cancels to
    // first order inside one shape.
    for (pass, order) in [("up", widths), ("down", widths.reversed().map { $0 })] {
        for m in order {
            let xs = (0..<inner).map { e169Activations(m: m, k: shape.k, salt: $0) }
            let routed =
                Qwen35CustomQMV.routable(
                    xs[0], weight.w, scales: weight.scales, biases: weight.biases,
                    groupSize: 64, bits: 4, mode: .affine) != nil
            let samples = e169MedianSecondsPerCall(reps: reps, inner: inner) {
                var outs: [MLXArray] = []
                outs.reserveCapacity(inner)
                var x = xs[0]
                for index in 0..<inner {
                    let out = qwen35RoutedQuantizedMM(
                        x, weight.w, scales: weight.scales, biases: weight.biases,
                        groupSize: 64, bits: 4, mode: .affine)
                    outs.append(out)
                    if index + 1 < inner { x = xs[index + 1] + out[0..., 0..<1, 0..<1] * 1e-30 }
                }
                return outs
            }
            rows.append([
                "m": m,
                "pass": pass,
                "routed_to_custom_qmv": routed,
                "seconds_per_call": samples[samples.count / 2],
                "seconds_per_call_min": samples[0],
                "seconds_per_call_max": samples[samples.count - 1],
                "flops_per_call": 2 * m * shape.k * shape.n,
            ])
        }
    }
    return [
        "name": shape.name,
        "family": shape.family,
        "k": shape.k,
        "n": shape.n,
        "calls_per_verify": shape.callsPerVerify,
        "rows": rows,
    ]
}

/// The 48 gated-delta scans. Same head geometry as the checkpoint, same public
/// entry point the layer calls.
private func e169SweepGatedDelta(widths: [Int], reps: Int, inner: Int) -> [String: Any] {
    let hk = 16, dk = 128, hv = 48, dv = 128
    let aLog = (zeros([hv], dtype: .float32) + Float(-0.5)).asType(.bfloat16)
    let dtBias = (zeros([hv], dtype: .float32) + Float(0.1)).asType(.bfloat16)
    let state0 = zeros([1, hv, dk, dv], dtype: .float32)
    eval(aLog, dtBias, state0)

    var rows: [[String: Any]] = []
    for (pass, order) in [("up", widths), ("down", widths.reversed().map { $0 })] {
        for m in order {
            let q = e169GDNInput([1, m, hk, dk], salt: 1)
            let k = e169GDNInput([1, m, hk, dk], salt: 2)
            let v = e169GDNInput([1, m, hv, dv], salt: 3)
            let a = e169GDNInput([1, m, hv], salt: 4)
            let b = e169GDNInput([1, m, hv], salt: 5)
            eval(q, k, v, a, b)
            let samples = e169MedianSecondsPerCall(reps: reps, inner: inner) {
                var outs: [MLXArray] = []
                var state = state0
                for _ in 0..<inner {
                    let (y, next) = gatedDeltaUpdate(
                        q: q, k: k, v: v, a: a, b: b,
                        aLog: aLog, dtBias: dtBias, state: state)
                    outs.append(y)
                    state = next
                }
                outs.append(state)
                return outs
            }
            rows.append([
                "m": m,
                "pass": pass,
                "seconds_per_call": samples[samples.count / 2],
                "seconds_per_call_min": samples[0],
                "seconds_per_call_max": samples[samples.count - 1],
                "state_bytes_per_layer": hv * dk * dv * 4,
            ])
        }
    }
    return [
        "name": "linear_attn.gated_delta_recurrence",
        "family": "gdn",
        "calls_per_verify": 48,
        "rows": rows,
    ]
}

private func e169GDNInput(_ shape: [Int], salt: Int) -> MLXArray {
    let count = shape.reduce(1, *)
    let values = (0..<count).map { index -> Float in
        Float(((index &* 1_664_525) &+ salt &* 1_013_904_223) % 2_003) / 2_003.0 - 0.5
    }
    return MLXArray(values).reshaped(shape).asType(.bfloat16)
}

/// The exact top-two readout the session runs once per verify block over
/// `[1, M, 248320]` logits. One call per forward, but its cost scales with rows.
private func e169SweepTopTwo(widths: [Int], reps: Int, inner: Int) -> [String: Any] {
    var rows: [[String: Any]] = []
    for (pass, order) in [("up", widths), ("down", widths.reversed().map { $0 })] {
        for m in order {
            let logits = e169Activations(m: m, k: 248_320, salt: 7)
            let samples = e169MedianSecondsPerCall(reps: reps, inner: inner) {
                var outs: [MLXArray] = []
                for _ in 0..<inner {
                    let (ids, values) = Qwen36MTPBlockSession.linearTopTwoRows(logits)
                    outs.append(ids)
                    outs.append(values)
                }
                return outs
            }
            rows.append([
                "m": m,
                "pass": pass,
                "seconds_per_call": samples[samples.count / 2],
                "seconds_per_call_min": samples[0],
                "seconds_per_call_max": samples[samples.count - 1],
            ])
        }
    }
    return [
        "name": "session.linear_top_two_rows",
        "family": "readout",
        "calls_per_verify": 1,
        "rows": rows,
    ]
}

private func describeE169Device() -> [String: Any] {
    let device = MLX.GPU.deviceInfo()
    return [
        "architecture": device.architecture,
        "max_buffer_size": device.maxBufferSize,
        "max_recommended_working_set_size": Int(device.maxRecommendedWorkingSetSize),
        "memory_size": device.memorySize,
    ]
}
