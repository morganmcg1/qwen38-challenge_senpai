import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXRandom
import Testing

@testable import MLXLLM

// E186 -- which kernel family carries E177's 0.576*M^2 ms round term?
//
// Isolated per-family cost curves us(m) at evaluation width m = 1...9, run in
// a test process instead of the timed worker. Two sections:
//
//   * `dispatch` proves which GDN branch and which QMV dispatch the decode
//     widths actually take, from runtime side effects and counter deltas
//     rather than from a source read alone;
//   * `probe` times each family at every width with ABBA-counterbalanced
//     block ordering and writes raw per-block samples as JSON.
//
// Research instrument. Off unless the section knob is set. `Tests/` is never
// packaged into a submission. Within-session relative measurement,
// harness=local, no thermal gate, no score.

private struct E186Sample: Encodable {
    var family: String
    var cell: String
    var m: Int
    var block: Int
    var ascending: Bool
    var microseconds: Double
    var reps: Int
}

private struct E186QmvCell {
    var name: String
    var k: Int
    var n: Int
}

@Suite("E186 decode width kernel probes")
struct E186DecodeWidthProbeTests {
    static let dispatchEnabled =
        ProcessInfo.processInfo.environment["MLXFAST_RUN_E186_DISPATCH"] == "1"
    static let probeEnabled =
        ProcessInfo.processInfo.environment["MLXFAST_RUN_E186_PROBE"] == "1"

    static let widths = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    // MARK: - target configuration

    /// Decode the pinned target's own `config.json` rather than typing the
    /// architecture in. Every shape in this probe then comes from the artifact.
    static func configuration() throws -> Qwen35TextConfiguration {
        let env = ProcessInfo.processInfo.environment
        let explicit = env["MLXFAST_E186_CONFIG"]
        let candidates: [String]
        if let explicit {
            candidates = [explicit]
        } else {
            let home = FileManager.default.homeDirectoryForCurrentUser.path
            let root =
                "\(home)/.cache/huggingface/hub/models--EigenLabs--Qwen3.8-27B-4bit/snapshots"
            let snapshots =
                (try? FileManager.default.contentsOfDirectory(atPath: root)) ?? []
            candidates = snapshots.map { "\(root)/\($0)/config.json" }
        }
        let path = try #require(
            candidates.first { FileManager.default.fileExists(atPath: $0) },
            "no target config.json found; set MLXFAST_E186_CONFIG")
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        let outer = try #require(
            try JSONSerialization.jsonObject(with: data) as? [String: Any])
        let text = try #require(outer["text_config"] as? [String: Any])
        let textData = try JSONSerialization.data(withJSONObject: text)
        return try JSONDecoder().decode(
            Qwen35TextConfiguration.self, from: textData)
    }

    // MARK: - helpers

    static func bf16(_ module: Module) {
        module.update(
            parameters: module.parameters().mapValues {
                $0.dtype == .float32 ? $0.asType(.bfloat16) : $0
            })
    }

    static func quantizedBF16(_ module: Module) {
        bf16(module)
        quantize(model: module, groupSize: 64, bits: 4)
        eval(module)
    }

    /// One blocking `eval` per call, so every cell carries the same fixed
    /// per-eval cost. `evalFloor` measures that cost directly.
    static func timed(reps: Int, _ body: () -> [MLXArray]) -> Double {
        let start = DispatchTime.now().uptimeNanoseconds
        for _ in 0 ..< reps {
            eval(body())
        }
        return Double(DispatchTime.now().uptimeNanoseconds - start) / 1e3
            / Double(reps)
    }

    static func evalFloor(reps: Int) -> Double {
        let a = MLXArray.zeros([16], dtype: .float32)
        return timed(reps: reps) { [a + 1] }
    }

    static func gpuTemperature() -> Double? {
        for path in [
            ProcessInfo.processInfo.environment["MLXFAST_MACMON_BIN"] ?? "",
            "\(FileManager.default.homeDirectoryForCurrentUser.path)/bin/macmon",
            "/opt/homebrew/bin/macmon", "/usr/local/bin/macmon",
        ] where !path.isEmpty && FileManager.default.isExecutableFile(atPath: path) {
            let process = Process()
            process.executableURL = URL(fileURLWithPath: path)
            process.arguments = ["pipe", "-s1"]
            let pipe = Pipe()
            process.standardOutput = pipe
            process.standardError = FileHandle.nullDevice
            guard (try? process.run()) != nil else { continue }
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            process.waitUntilExit()
            guard
                let object = try? JSONSerialization.jsonObject(with: data)
                    as? [String: Any],
                let temp = object["temp"] as? [String: Any],
                let gpu = temp["gpu_temp_avg"] as? Double
            else { continue }
            return gpu
        }
        return nil
    }

    static func write(_ payload: [String: Any], to key: String) throws {
        let path = try #require(
            ProcessInfo.processInfo.environment[key],
            "\(key) must name the JSON destination")
        let data = try JSONSerialization.data(
            withJSONObject: payload,
            options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
        try data.write(to: URL(fileURLWithPath: path))
    }

    static func counters() -> [String: Int] {
        [
            "kernel_config_cache_hits": qwen35KernelConfigCacheHits,
            "kernel_config_cache_misses": qwen35KernelConfigCacheMisses,
            "xsums_sidecar_hits": qwen35XSumsSidecarHits,
            "xsums_standalone_fills": qwen35XSumsStandaloneFills,
            "xsums_fill_distinct": qwen35XSumsFillDistinct,
            "row_top32_fused_drafts": qwen35RowTop32FusedDrafts,
        ]
    }

    static func counterDelta(_ before: [String: Int], _ after: [String: Int])
        -> [String: Int]
    {
        after.reduce(into: [String: Int]()) { out, entry in
            out[entry.key] = entry.value - (before[entry.key] ?? 0)
        }
    }

    // MARK: - synthetic quantized cells

    /// affine 4-bit group-64 packed weights at a scored cell shape. The probe
    /// times dispatch, not arithmetic, so the packed bits are random.
    static func packedCell(k: Int, n: Int) -> (MLXArray, MLXArray, MLXArray) {
        let w = MLXRandom.uniform(low: 0.0, high: 4.2e9, [n, k / 8])
            .asType(.uint32)
        let scales = MLXRandom.uniform(low: 0.01, high: 0.02, [n, k / 64])
            .asType(.bfloat16)
        let biases = MLXRandom.uniform(low: -0.02, high: 0.02, [n, k / 64])
            .asType(.bfloat16)
        eval(w, scales, biases)
        return (w, scales, biases)
    }

    static func activations(width: Int, k: Int) -> MLXArray {
        let x = MLXRandom.normal([1, width, k]).asType(.bfloat16)
        eval(x)
        return x
    }

    static let qmvCells = [
        E186QmvCell(name: "mlp.gate_up", k: 5120, n: 34816),
        E186QmvCell(name: "mlp.down", k: 17408, n: 5120),
        E186QmvCell(name: "gdn.in_proj", k: 5120, n: 16480),
        E186QmvCell(name: "gdn.out_proj", k: 6144, n: 5120),
        E186QmvCell(name: "fa.qkv", k: 5120, n: 14336),
        E186QmvCell(name: "fa.o_proj", k: 6144, n: 5120),
        E186QmvCell(name: "lm_head", k: 5120, n: 248_320),
    ]

    // MARK: - dispatch proof

    @Test(.enabled(if: E186DecodeWidthProbeTests.dispatchEnabled))
    func dispatchProof() throws {
        let config = try configuration()
        var payload: [String: Any] = [
            "harness": "local",
            "cool_gate_passed_real_gate": false,
            "gate_qualified_for_timing": false,
            "official_or_ranked_score": false,
            "compiled_decode_supported": MLXHardwareInfo.isCompiledDecodeSupported,
            "qmv_arm": Qwen35CustomQMV.arm.rawValue,
            "kernel_config_cache_enabled": Qwen35KernelConfigCache.enabled,
        ]
        payload["config"] = [
            "hidden_size": config.hiddenSize,
            "hidden_layers": config.hiddenLayers,
            "intermediate_size": config.intermediateSize,
            "attention_heads": config.attentionHeads,
            "kv_heads": config.kvHeads,
            "head_dim": config.headDim ?? -1,
            "vocab_size": config.vocabularySize,
            "full_attention_interval": config.fullAttentionInterval,
            "linear_num_key_heads": config.linearNumKeyHeads,
            "linear_num_value_heads": config.linearNumValueHeads,
            "linear_key_head_dim": config.linearKeyHeadDim,
            "linear_value_head_dim": config.linearValueHeadDim,
            "linear_conv_kernel_dim": config.linearConvKernelDim,
        ]

        // 1. QMV routing witness: does the candidate replica actually take the
        // cell at this width, and with how many input groups?
        var routing: [[String: Any]] = []
        for cell in qmvCells {
            let (w, scales, biases) = packedCell(k: cell.k, n: cell.n)
            for m in widths {
                let x = activations(width: m, k: cell.k)
                let routed = Qwen35CustomQMV.matmul(
                    x, w, scales: scales, biases: biases,
                    groupSize: 64, bits: 4, mode: .affine)
                if let routed { eval(routed) }
                routing.append([
                    "cell": cell.name, "k": cell.k, "n": cell.n, "m": m,
                    "replica_taken": routed != nil,
                    "active_input_groups": (2 ... 9).contains(m)
                        ? Qwen35CustomQMV.activeInputGroups(m) : 0,
                    "table_pays": Qwen35CustomQMV.tablePays(m: m),
                ])
            }
        }
        payload["qmv_routing"] = routing

        // 2. GDN branch witness. The three decode branches leave different,
        // observable marks on the cache: the S == 2 mid-kernel branch writes
        // one rollback checkpoint per boundary and no replay tape; the
        // S >= 3 branch writes a replay tape and no checkpoints.
        let gdn = Qwen35GatedDeltaNet(config)
        quantizedBF16(gdn)
        let convDim = config.linearKeyHeadDim * config.linearNumKeyHeads * 2
            + config.linearValueHeadDim * config.linearNumValueHeads
        var branches: [[String: Any]] = []
        for m in widths {
            let cache = MambaCache()
            cache[0] = MLXRandom.normal([1, config.linearConvKernelDim - 1, convDim])
                .asType(.bfloat16)
            cache[1] = MLXRandom.normal([
                1, config.linearNumValueHeads, config.linearValueHeadDim,
                config.linearKeyHeadDim,
            ])
            let x = activations(width: m, k: config.hiddenSize)
            let before = counters()
            let out = gdn(x, mask: nil, cache: cache, nConfirmed: 1)
            eval(out)
            let after = counters()
            branches.append([
                "m": m,
                "output_shape": out.shape,
                "rollback_checkpoints": cache.rollbackCheckpoints.count,
                "has_replay_tape": cache.prefixReplayTape != nil,
                "replay_tape_rows": cache.prefixReplayTape?.rowCount ?? -1,
                "has_rollback_state": cache.rollbackState != nil,
                "counter_delta": counterDelta(before, after),
            ])
        }
        payload["gdn_branch_witness"] = branches

        // 3. Optional GPU capture: names the kernels directly when the
        // environment allows a capture.
        if ProcessInfo.processInfo.environment["MLXFAST_E186_CAPTURE"] == "1",
           let capturePath = ProcessInfo.processInfo.environment[
               "MLXFAST_E186_CAPTURE_PATH"]
        {
            var captured: [String: Any] = ["requested_path": capturePath]
            let url = URL(fileURLWithPath: capturePath)
            GPU.startCapture(url: url)
            for m in [2, 5, 8] {
                let cache = MambaCache()
                cache[0] = MLXRandom.normal(
                    [1, config.linearConvKernelDim - 1, convDim]
                ).asType(.bfloat16)
                cache[1] = MLXRandom.normal([
                    1, config.linearNumValueHeads, config.linearValueHeadDim,
                    config.linearKeyHeadDim,
                ])
                let x = activations(width: m, k: config.hiddenSize)
                eval(gdn(x, mask: nil, cache: cache, nConfirmed: 1))
            }
            GPU.stopCapture()
            captured["exists"] = FileManager.default.fileExists(
                atPath: capturePath)
            payload["gpu_capture"] = captured
        }

        try write(payload, to: "MLXFAST_E186_DISPATCH_OUT")
    }

    // MARK: - width sweep

    @Test(.enabled(if: E186DecodeWidthProbeTests.probeEnabled))
    func widthSweep() throws {
        let env = ProcessInfo.processInfo.environment
        let blocks = Int(env["MLXFAST_E186_BLOCKS"] ?? "") ?? 12
        let warmup = Int(env["MLXFAST_E186_WARMUP"] ?? "") ?? 8
        let config = try configuration()
        let hidden = config.hiddenSize
        let headDim = config.headDim ?? 256
        let kvLengths = (env["MLXFAST_E186_KV"]?.split(separator: ",")
            .compactMap { Int($0) }).flatMap { $0.isEmpty ? nil : $0 } ?? [512, 1024]

        var samples: [E186Sample] = []
        var notes: [[String: Any]] = []
        var temperatures: [[String: Any]] = []

        func recordTemperature(_ label: String) {
            temperatures.append([
                "label": label,
                "gpu_temp_c": gpuTemperature() ?? -1,
                "seconds": Date().timeIntervalSince1970,
            ])
        }

        // --- family construction -------------------------------------------
        // Every family is a closure list indexed by width, built once so that
        // no allocation of a weight tensor lands inside a timed block.
        var families: [(family: String, cell: String, reps: Int,
                        call: [Int: () -> [MLXArray]])] = []

        // 1. Routed QMV cells, in-path dispatch (replica where routable,
        // MLX quantizedMM otherwise) and the MLX arm alone for contrast.
        for cell in qmvCells {
            let (w, scales, biases) = packedCell(k: cell.k, n: cell.n)
            let x = Dictionary(
                uniqueKeysWithValues: widths.map {
                    ($0, activations(width: $0, k: cell.k))
                })
            var inpath: [Int: () -> [MLXArray]] = [:]
            var mlxArm: [Int: () -> [MLXArray]] = [:]
            for m in widths {
                let xm = x[m]!
                inpath[m] = {
                    if let y = Qwen35CustomQMV.matmul(
                        xm, w, scales: scales, biases: biases,
                        groupSize: 64, bits: 4, mode: .affine)
                    {
                        return [y]
                    }
                    return [
                        quantizedMM(
                            xm, w, scales: scales, biases: biases,
                            transpose: true, groupSize: 64, bits: 4,
                            mode: .affine)
                    ]
                }
                mlxArm[m] = {
                    [
                        quantizedMM(
                            xm, w, scales: scales, biases: biases,
                            transpose: true, groupSize: 64, bits: 4,
                            mode: .affine)
                    ]
                }
            }
            let reps = cell.n * cell.k > 100_000_000 ? 6 : 20
            families.append(("qmv_inpath", cell.name, reps, inpath))
            families.append(("qmv_mlx", cell.name, reps, mlxArm))
        }

        // 2. GDN mixer, exactly as a verify round calls it.
        let gdn = Qwen35GatedDeltaNet(config)
        quantizedBF16(gdn)
        let convDim = config.linearKeyHeadDim * config.linearNumKeyHeads * 2
            + config.linearValueHeadDim * config.linearNumValueHeads
        let gdnConv = MLXRandom.normal(
            [1, config.linearConvKernelDim - 1, convDim]
        ).asType(.bfloat16)
        let gdnState = MLXRandom.normal([
            1, config.linearNumValueHeads, config.linearValueHeadDim,
            config.linearKeyHeadDim,
        ])
        eval(gdnConv, gdnState)
        var gdnCalls: [Int: () -> [MLXArray]] = [:]
        for m in widths {
            let x = activations(width: m, k: hidden)
            gdnCalls[m] = {
                // A fresh cache per call keeps the recurrent state fixed, so
                // every replicate measures the same arithmetic.
                let cache = MambaCache()
                cache[0] = gdnConv
                cache[1] = gdnState
                return [gdn(x, mask: nil, cache: cache, nConfirmed: 1)]
            }
        }
        families.append(("gdn_layer", "nConfirmed=1", 20, gdnCalls))

        // 3. Isolated GDN recurrence at T = m, the vendored public entry.
        let rq = Dictionary(
            uniqueKeysWithValues: widths.map { m in
                (m,
                 MLXRandom.normal([
                    1, m, config.linearNumKeyHeads, config.linearKeyHeadDim,
                 ]).asType(.bfloat16))
            })
        let rk = Dictionary(
            uniqueKeysWithValues: widths.map { m in
                (m,
                 MLXRandom.normal([
                    1, m, config.linearNumKeyHeads, config.linearKeyHeadDim,
                 ]).asType(.bfloat16))
            })
        let rv = Dictionary(
            uniqueKeysWithValues: widths.map { m in
                (m,
                 MLXRandom.normal([
                    1, m, config.linearNumValueHeads, config.linearValueHeadDim,
                 ]).asType(.bfloat16))
            })
        let ra = Dictionary(
            uniqueKeysWithValues: widths.map { m in
                (m,
                 MLXRandom.normal([1, m, config.linearNumValueHeads])
                    .asType(.bfloat16))
            })
        let rb = Dictionary(
            uniqueKeysWithValues: widths.map { m in
                (m,
                 MLXRandom.normal([1, m, config.linearNumValueHeads])
                    .asType(.bfloat16))
            })
        let aLog = MLXRandom.uniform(
            low: 0.1, high: 2.0, [config.linearNumValueHeads]
        ).asType(.bfloat16)
        let dtBias = MLXRandom.uniform(
            low: 0.1, high: 1.0, [config.linearNumValueHeads]
        ).asType(.bfloat16)
        eval(aLog, dtBias)
        for table in [rq, rk, rv, ra, rb] {
            for value in table.values { eval(value) }
        }
        var recurrenceCalls: [Int: () -> [MLXArray]] = [:]
        for m in widths {
            recurrenceCalls[m] = {
                let pair = gatedDeltaUpdate(
                    q: rq[m]!, k: rk[m]!, v: rv[m]!, a: ra[m]!, b: rb[m]!,
                    aLog: aLog, dtBias: dtBias, state: gdnState, mask: nil)
                return [pair.0, pair.1]
            }
        }
        families.append(("gdn_recurrence", "T=m", 20, recurrenceCalls))

        // 4. Full-attention mixer with a populated KV cache.
        let attention = Qwen35Attention(config)
        quantizedBF16(attention)
        for kv in kvLengths {
            let cache = KVCacheSimple()
            let seed = activations(width: kv, k: hidden)
            eval(attention(seed, mask: .causal, cache: cache))
            var calls: [Int: () -> [MLXArray]] = [:]
            for m in widths {
                let x = activations(width: m, k: hidden)
                let mask: MLXFast.ScaledDotProductAttentionMaskMode =
                    m == 1 ? .none : .causal
                calls[m] = {
                    cache.offset = kv
                    return [attention(x, mask: mask, cache: cache)]
                }
            }
            families.append(("fa_layer", "kv=\(kv)", 20, calls))
        }

        // 5. Isolated SDPA at the GQA decode shape.
        for kv in kvLengths {
            let keys = MLXRandom.normal([1, config.kvHeads, kv, headDim])
                .asType(.bfloat16)
            let values = MLXRandom.normal([1, config.kvHeads, kv, headDim])
                .asType(.bfloat16)
            eval(keys, values)
            var calls: [Int: () -> [MLXArray]] = [:]
            let scale = 1.0 / Foundation.sqrt(Float(headDim))
            for m in widths {
                let q = MLXRandom.normal([1, config.attentionHeads, m, headDim])
                    .asType(.bfloat16)
                eval(q)
                let mask: MLXFast.ScaledDotProductAttentionMaskMode =
                    m == 1 ? .none : .causal
                calls[m] = {
                    [
                        MLXFast.scaledDotProductAttention(
                            queries: q, keys: keys, values: values,
                            scale: scale, mask: mask)
                    ]
                }
            }
            families.append(("sdpa", "kv=\(kv)", 20, calls))
        }

        // 6. MLP block.
        let mlp = Qwen35FusedMLP(
            dimensions: hidden, hiddenDimensions: config.intermediateSize)
        quantizedBF16(mlp)
        var mlpCalls: [Int: () -> [MLXArray]] = [:]
        for m in widths {
            let x = activations(width: m, k: hidden)
            mlpCalls[m] = { [mlp(x)] }
        }
        families.append(("mlp", "fused", 20, mlpCalls))

        notes.append([
            "families": families.map { ["family": $0.family, "cell": $0.cell] },
            "blocks": blocks,
            "warmup_evals_per_cell": warmup,
        ])

        // --- warmup ---------------------------------------------------------
        recordTemperature("session_entry")
        for family in families {
            for m in widths {
                for _ in 0 ..< warmup { eval(family.call[m]!()) }
            }
        }
        recordTemperature("after_warmup")

        // --- ABBA blocks ----------------------------------------------------
        for block in 0 ..< blocks {
            let ascending = block % 2 == 0
            let order = ascending ? widths : widths.reversed().map { $0 }
            recordTemperature("block_\(block)_entry")
            for family in families {
                for m in order {
                    let us = timed(reps: family.reps, family.call[m]!)
                    samples.append(
                        E186Sample(
                            family: family.family, cell: family.cell, m: m,
                            block: block, ascending: ascending,
                            microseconds: us, reps: family.reps))
                }
            }
        }
        recordTemperature("session_exit")

        let floor = evalFloor(reps: 200)
        let encoded = try JSONEncoder().encode(samples)
        let sampleObjects = try JSONSerialization.jsonObject(with: encoded)

        var payload: [String: Any] = [
            "harness": "local",
            "cool_gate_passed_real_gate": false,
            "gate_qualified_for_timing": false,
            "official_or_ranked_score": false,
            "eval_floor_microseconds": floor,
            "widths": widths,
            "blocks": blocks,
            "warmup_evals_per_cell": warmup,
            "kv_lengths": kvLengths,
            "qmv_arm": Qwen35CustomQMV.arm.rawValue,
            "compiled_decode_supported": MLXHardwareInfo.isCompiledDecodeSupported,
            "samples": sampleObjects,
            "temperatures": temperatures,
            "notes": notes,
        ]
        payload["active_input_groups"] = Dictionary(
            uniqueKeysWithValues: widths.map {
                ("\($0)", (2 ... 9).contains($0)
                    ? Qwen35CustomQMV.activeInputGroups($0) : 0)
            })
        try write(payload, to: "MLXFAST_E186_OUT")
    }
}
