import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXFastModel

// E71 -- in-situ width-tax census.
//
// THE QUESTION. A verify round reads the same 14,412,349,440 bytes of quantized
// backbone weight whether it checks 1 row or 9. E1 still measured
// `T(6) - T(1) = 69.659 ms`. This suite attributes that width tax to kernel
// families IN SITU, at real occupancy, with the real checkpoint resident.
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
// Enable with `MLXFAST_RUN_E71_WIDTH_TAX=1` and point `MLXFAST_E71_OUT` at the
// JSON destination.

@Suite(.serialized)
struct E71WidthTaxCensusTests {
    private static var enabled: Bool {
        ProcessInfo.processInfo.environment["MLXFAST_RUN_E71_WIDTH_TAX"] == "1"
    }

    @Test(.enabled(if: E71WidthTaxCensusTests.enabled))
    func censusTheWidthTax() throws {
        let env = ProcessInfo.processInfo.environment
        let outPath = try #require(
            env["MLXFAST_E71_OUT"], "MLXFAST_E71_OUT must name the JSON destination")

        let weightsPath = env["MLXFAST_E71_WEIGHTS"] ?? "weights"
        let promptPath =
            env["MLXFAST_E71_PROMPT"]
            ?? "correctness_prompts/public_longcopy_gate_english_512_1024.json"
        let seedLength = Int(env["MLXFAST_E71_SEED_LEN"] ?? "") ?? 768
        let reps = Int(env["MLXFAST_E71_REPS"] ?? "") ?? 12
        let warmup = Int(env["MLXFAST_E71_WARMUP"] ?? "") ?? 3
        let curveWidths = parseWidths(env["MLXFAST_E71_CURVE_WIDTHS"]) ?? Array(1...9)
        let armWidths = parseWidths(env["MLXFAST_E71_ARM_WIDTHS"]) ?? [4, 6, 9]
        let armNames =
            (env["MLXFAST_E71_ARMS"]?.split(separator: ",").map {
                $0.trimmingCharacters(in: .whitespaces)
            }).flatMap { $0.isEmpty ? nil : $0 }
            ?? ["null", "lm_head", "mlp_all", "mlp_down", "fa_o_proj", "gdn_out_proj"]

        let tokens = try loadPromptTokens(promptPath)
        #expect(tokens.count >= seedLength + 256)

        let config = try Qwen35Config.load(from: weightsPath)
        let loader = try Qwen35WeightLoader(weightsPath: weightsPath)
        let loadStart = DispatchTime.now().uptimeNanoseconds
        let runtime = Qwen35RuntimeWeightCache(loader: loader, config: config)
        let model = try runtime.requireLibraryModel()
        let loadSeconds =
            Double(DispatchTime.now().uptimeNanoseconds - loadStart) / 1e9

        let harness = E71Harness(
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
                guard let arm = e71Arm(name) else {
                    Issue.record("unknown E71 arm \(name)")
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
            "experiment": "e71-in-situ-width-tax-census",
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
                "device": describeE71Device(),
                "host": ProcessInfo.processInfo.hostName,
            ],
            "blocks": blocks,
        ]
        let data = try JSONSerialization.data(
            withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: URL(fileURLWithPath: outPath))
        print("E71_OUT \(outPath)")
    }
}

// MARK: - harness

/// One resident model, replayed as a sequence of matched timing blocks.
private final class E71Harness {
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
        run(arm: e71Arm(arm) ?? E71Arm.baseline(), width: width, order: order)
    }

    func run(arm: E71Arm, width: Int, order: Int) -> [String: Any] {
        let entryTemp = e71GPUTemperature()
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
        for _ in 0..<reps {
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
        let exitTemp = e71GPUTemperature()
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
            "E71_BLOCK "
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
private struct E71Arm {
    let name: String
    /// Rows the wrapped family sees. `nil` means "the caller's width", i.e. the
    /// null control: same wrapper, same launches, same copy, full-width family.
    let pin: Int?
    let wrap: (Qwen35TextModel, Int) -> () -> Void

    func pinRows(_ width: Int) -> Int { pin ?? width }

    func install(_ model: Qwen35TextModel, _ width: Int) -> () -> Void {
        wrap(model, pin ?? width)
    }

    static func baseline() -> E71Arm {
        E71Arm(name: "baseline", pin: nil) { _, _ in {} }
    }
}

private func e71Arm(_ name: String) -> E71Arm? {
    switch name {
    case "baseline":
        return .baseline()

    // Rung 2 control. The wrapper runs around `mlp` with the family at FULL
    // width, so anything it measures is harness overhead and nothing else.
    case "null":
        return E71Arm(name: "null", pin: nil) { model, pin in
            e71WrapUnary(model, childPath: ["mlp"], pinRows: pin)
        }

    // Rung 2 positive control. E63 measured this shape standalone at 92.1 % of
    // peak bandwidth at NA=4 -- the highest fraction of any scored shape -- so
    // its tax is predictable from independent data.
    case "lm_head":
        return E71Arm(name: "lm_head", pin: 1) { model, pin in
            e71WrapLMHead(model, pinRows: pin)
        }

    // Rung 3. 64 layers, 9.626 GB of the 14.412 GB weight stream: the fused
    // gate/up QMV, the compiled SwiGLU and `down_proj` together.
    case "mlp_all":
        return E71Arm(name: "mlp_all", pin: 1) { model, pin in
            e71WrapUnary(model, childPath: ["mlp"], pinRows: pin)
        }

    // 64 layers, 3.209 GB. `mlp_all` minus this isolates fused gate/up + SwiGLU.
    case "mlp_down":
        return E71Arm(name: "mlp_down", pin: 1) { model, pin in
            e71WrapQuantizedLinear(model, childPath: ["mlp", "down_proj"], pinRows: pin)
        }

    // 16 full-attention layers, 0.283 GB. The only attention projection that
    // still dispatches through its child module.
    case "fa_o_proj":
        return E71Arm(name: "fa_o_proj", pin: 1) { model, pin in
            e71WrapQuantizedLinear(model, childPath: ["self_attn", "o_proj"], pinRows: pin)
        }

    // 48 GDN layers, 0.849 GB. Same seam on the recurrent side.
    case "gdn_out_proj":
        return E71Arm(name: "gdn_out_proj", pin: 1) { model, pin in
            e71WrapQuantizedLinear(model, childPath: ["linear_attn", "out_proj"], pinRows: pin)
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
private func e71Pinned(
    _ x: MLXArray, pinRows: Int, _ body: (MLXArray) -> MLXArray
) -> MLXArray {
    let rows = x.dim(1)
    let take = min(max(pinRows, 1), rows)
    let y = body(x[0..., ..<take])
    return broadcast(y[0..., ..<1], to: [y.dim(0), rows, y.dim(2)]).contiguous()
}

private final class E71PinnedUnary: Module, UnaryLayer {
    let inner: Module
    let pinRows: Int

    init(inner: Module, pinRows: Int) {
        self.inner = inner
        self.pinRows = pinRows
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        e71Pinned(x, pinRows: pinRows) { (inner as! UnaryLayer)($0) }
    }
}

/// Shares the wrapped layer's weight, scales and biases arrays, so installing it
/// costs no extra resident bytes and cannot perturb the memory profile.
private final class E71PinnedQuantizedLinear: QuantizedLinear {
    let pinRows: Int

    init(_ other: QuantizedLinear, pinRows: Int) {
        self.pinRows = pinRows
        super.init(
            weight: other.weight, bias: other.bias, scales: other.scales,
            biases: other.biases, groupSize: other.groupSize, bits: other.bits,
            mode: other.mode)
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        e71Pinned(x, pinRows: pinRows) { super.callAsFunction($0) }
    }
}

private func e71DecoderLayers(_ model: Qwen35TextModel) -> [Module] {
    guard case .array(let items)? = model.model.children()["layers"] else { return [] }
    return items.compactMap { item in
        if case .value(let module) = item { return module }
        return nil
    }
}

private func e71Child(_ module: Module, _ key: String) -> Module? {
    guard case .value(let child)? = module.children()[key] else { return nil }
    return child
}

private func e71Replace(_ parent: Module, _ key: String, _ replacement: Module) {
    var update = NestedDictionary<String, Module>()
    update[key] = .value(replacement)
    parent.update(modules: update)
}

/// Wrap a `Module`-typed child that the layer calls through `UnaryLayer`.
private func e71WrapUnary(
    _ model: Qwen35TextModel, childPath: [String], pinRows: Int
) -> () -> Void {
    var restores: [() -> Void] = []
    for layer in e71DecoderLayers(model) {
        guard let (parent, key, original) = e71Resolve(layer, childPath) else { continue }
        e71Replace(parent, key, E71PinnedUnary(inner: original, pinRows: pinRows))
        restores.append { e71Replace(parent, key, original) }
    }
    return { restores.forEach { $0() } }
}

private func e71WrapQuantizedLinear(
    _ model: Qwen35TextModel, childPath: [String], pinRows: Int
) -> () -> Void {
    var restores: [() -> Void] = []
    for layer in e71DecoderLayers(model) {
        guard let (parent, key, original) = e71Resolve(layer, childPath),
            let quantized = original as? QuantizedLinear
        else { continue }
        e71Replace(parent, key, E71PinnedQuantizedLinear(quantized, pinRows: pinRows))
        restores.append { e71Replace(parent, key, original) }
    }
    return { restores.forEach { $0() } }
}

private func e71WrapLMHead(_ model: Qwen35TextModel, pinRows: Int) -> () -> Void {
    guard let head = e71Child(model, "lm_head") as? QuantizedLinear else { return {} }
    e71Replace(model, "lm_head", E71PinnedQuantizedLinear(head, pinRows: pinRows))
    return { e71Replace(model, "lm_head", head) }
}

/// Walk `childPath` from a decoder layer and return the parent that owns the
/// last component, so the replacement lands on a `@ModuleInfo` setter.
private func e71Resolve(
    _ layer: Module, _ childPath: [String]
) -> (parent: Module, key: String, original: Module)? {
    var parent = layer
    for key in childPath.dropLast() {
        guard let next = e71Child(parent, key) else { return nil }
        parent = next
    }
    guard let key = childPath.last, let original = e71Child(parent, key) else { return nil }
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
        throw E71Failure("E71: \(path) is not a correctness prompt fixture")
    }
    return prompt + expected
}

/// One macmon sample. The census runs with `MLXFAST_LOCAL_COOL_GATE=0`, so the
/// entry and exit temperature of every block is the thermal record.
private func e71GPUTemperature() -> Double? {
    let binary =
        ProcessInfo.processInfo.environment["MLXFAST_E71_MACMON"] ?? "/opt/homebrew/bin/macmon"
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

private struct E71Failure: Error, CustomStringConvertible {
    let description: String
    init(_ description: String) { self.description = description }
}

private func describeE71Device() -> [String: Any] {
    let device = MLX.GPU.deviceInfo()
    return [
        "architecture": device.architecture,
        "max_buffer_size": device.maxBufferSize,
        "max_recommended_working_set_size": Int(device.maxRecommendedWorkingSetSize),
        "memory_size": device.memorySize,
    ]
}
