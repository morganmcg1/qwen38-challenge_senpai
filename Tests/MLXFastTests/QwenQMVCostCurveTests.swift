import CryptoKit
import Foundation
import MLX
import MLXLLM
import Testing

/// Cost curve of `quantized_matmul` at the exact scored Qwen 3.8 27B shapes,
/// swept over verify width M, through the VENDORED MLX the scored worker links.
///
/// This lives in the test target on purpose. Yukon never submits tests, so the
/// authoritative kernels can be timed without touching `Package.swift` or the
/// submitted surface. `research/qmv_cost_curve.py` runs the same sweep against
/// stock pip MLX 0.32.0; the two curves diverge wherever this checkout carries
/// a kernel the public release does not.
///
/// Enable with `MLXFAST_RUN_QMV_COST_CURVE=1` and point
/// `MLXFAST_QMV_COST_CURVE_OUT` at the JSON destination.
@Suite
struct QwenQMVCostCurveTests {
    private static var enabled: Bool {
        ProcessInfo.processInfo
            .environment["MLXFAST_RUN_QMV_COST_CURVE"] == "1"
    }

    @Test(.enabled(if: QwenQMVCostCurveTests.enabled))
    func sweepQuantizedMatmulOverVerifyWidth() throws {
        let env = ProcessInfo.processInfo.environment
        let outPath = try #require(
            env["MLXFAST_QMV_COST_CURVE_OUT"],
            "MLXFAST_QMV_COST_CURVE_OUT must name the JSON destination")
        let reps = Int(env["MLXFAST_QMV_COST_CURVE_REPS"] ?? "") ?? 15
        let inner = Int(env["MLXFAST_QMV_COST_CURVE_INNER"] ?? "") ?? 10

        var payload: [String: Any] = [
            "source": "vendored-mlx-swift",
            "reps": reps,
            "inner_calls_per_rep": inner,
            "widths": scoredWidths,
        ]
        payload["device"] = describeDispatchDevice()
        payload["roofline"] = measureRoofline(reps: reps)
        payload["shapes"] = scoredShapes.map { shape in
            sweep(shape: shape, widths: scoredWidths, reps: reps, inner: inner)
        }
        payload["dispatch_boundary_probes"] = dispatchBoundaryProbes.map { probe in
            sweep(shape: probe.shape, widths: probe.widths, reps: reps, inner: inner)
        }
        payload["fast_path_probes"] = fastPathProbes.map { shape in
            sweep(shape: shape, widths: [1, 4, 8, 9], reps: reps, inner: inner)
        }
        payload["head_fc_dtype_probe"] = measureHeadFCDtypes(reps: reps, inner: inner)
        payload["gdn_recurrence"] = sweepGatedDelta(
            widths: Array(1...12), reps: reps, inner: inner)

        let data = try JSONSerialization.data(
            withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: URL(fileURLWithPath: outPath))
        print("QMV_COST_CURVE_OUT \(outPath)")
    }

    /// Every scored projection must stay on `qmv_fast`. Falling off it is
    /// silent: the result is still correct, it just costs 1.64-1.80x more at
    /// M=9 (`fastprobe.k5184_n16480` vs the 512-aligned probes). Runs
    /// unconditionally because it is integer arithmetic, not a measurement.
    @Test
    func scoredShapesStayOnTheQMVFastPath() {
        for shape in scoredShapes + fastPathProbes.filter({ $0.k % 512 == 0 }) {
            #expect(
                shape.k % 512 == 0,
                "\(shape.name): reduction dim K=\(shape.k) leaves qmv_fast")
            #expect(
                shape.n % 8 == 0,
                "\(shape.name): output dim N=\(shape.n) leaves qmv_fast")
        }
    }

    /// `in_proj`'s 16480 and the compact draft head's 98336 satisfy only the
    /// `N % 8` half of the gate. They are safe as output dims, but a fusion or
    /// split that moved either into the reduction position would drop that
    /// matmul off `qmv_fast` at full numerical fidelity. Pinned so such a
    /// change has to delete this expectation deliberately.
    @Test
    func nOnlyDimsAreNotSafeAsReductionDims() {
        for n in [16480, 98336] {
            #expect(n % 8 == 0)
            #expect(n % 512 != 0, "\(n) is 512-aligned; re-measure the fast-path probes")
        }
    }

    /// Cost of the compact draft readout as a function of quantization bits.
    ///
    /// The crossrow specialization does NOT change which kernel the host
    /// launches. `qmv()` always names `affine_qmv_fast_<T>_gs_64_b_<bits>_
    /// batch_0` (`backend/metal/quantized.cpp:261-269`) and dispatches
    /// `grid_dims(M, ceil(N/8), B)` (`:254`), so `ntg.x == M`. The crossrow
    /// switch lives INSIDE that kernel, behind a bit-width gate, with
    /// `default: break` falling through to `qmv_fast_impl`
    /// (`backend/metal/kernels/quantized.h:1822-1955` at the time of writing).
    /// Which bit widths that gate admits is read back out of the live header by
    /// `crossrowGate()` rather than assumed here, so an arm is never labelled
    /// crossrow because of a hardcoded `bits == 4`.
    ///
    /// The M >= 2 arms are the empirical check on that reading: a discontinuity
    /// that appears at an admitted bit width and not at an excluded one is the
    /// specialization engaging, and its absence at M=1 is what the sweep needs
    /// to establish.
    ///
    /// Enable with `MLXFAST_RUN_QMV_BITS_SWEEP=1` and point
    /// `MLXFAST_QMV_BITS_SWEEP_OUT` at the JSON destination.
    @Test(.enabled(if: QwenQMVCostCurveTests.bitsSweepEnabled))
    func sweepCompactDraftReadoutOverBits() throws {
        let env = ProcessInfo.processInfo.environment
        let outPath = try #require(
            env["MLXFAST_QMV_BITS_SWEEP_OUT"],
            "MLXFAST_QMV_BITS_SWEEP_OUT must name the JSON destination")
        let reps = Int(env["MLXFAST_QMV_COST_CURVE_REPS"] ?? "") ?? 21
        let inner = Int(env["MLXFAST_QMV_COST_CURVE_INNER"] ?? "") ?? 10

        let k = 5120
        let n = 98_336
        let armSpecs: [(bits: Int, m: Int)] = [
            (4, 1), (3, 1), (2, 1),
            (4, 2), (3, 2),
            (4, 3), (3, 3),
            (4, 4), (3, 4),
            (4, 5), (3, 5),
        ]
        let gate = crossrowGate()
        var activations: [Int: [MLXArray]] = [:]
        for m in Set(armSpecs.map(\.m)) {
            let xs = (0..<inner).map { syntheticActivations(m: m, k: k, salt: $0) }
            eval(xs)
            activations[m] = xs
        }
        // One weight per bit width, shared across M, so the M probe costs
        // activations rather than another ~250 MB of resident head.
        var weights: [Int: QuantWeight] = [:]
        for bits in Set(armSpecs.map(\.bits)) {
            weights[bits] = lowBitQuantWeight(k: k, n: n, bits: bits)
        }

        // Every arm is resident before any of them is timed, and the reps below
        // are round-robin rather than blocked. This host holds a ~41C floor that
        // the 40C cool gate cannot clear, so an arm-blocked sweep would charge
        // whatever thermal drift happens during its own block to that arm alone.
        let bodies = armSpecs.map { spec -> () -> [MLXArray] in
            let weight = weights[spec.bits]!
            let xs = activations[spec.m]!
            return {
                var outs: [MLXArray] = []
                outs.reserveCapacity(inner)
                var x = xs[0]
                for index in 0..<inner {
                    let o = quantizedMM(
                        x, weight.w, scales: weight.scales, biases: weight.biases,
                        transpose: true, groupSize: 64, bits: spec.bits)
                    outs.append(o)
                    if index + 1 < inner { x = xs[index + 1] + o[0..<1, 0..<1] * 1e-30 }
                }
                return outs
            }
        }
        for body in bodies {
            for _ in 0..<3 { eval(body()) }
        }
        var samplesByArm = [[Double]](repeating: [], count: armSpecs.count)
        for _ in 0..<reps {
            for (index, body) in bodies.enumerated() {
                let start = DispatchTime.now().uptimeNanoseconds
                eval(body())
                let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
                samplesByArm[index].append(elapsed / Double(inner))
            }
        }

        var arms: [[String: Any]] = []
        for (index, spec) in armSpecs.enumerated() {
            let samples = samplesByArm[index].sorted()
            let median = samples[samples.count / 2]
            // Affine g64 with bf16 scale+bias is bits + 0.5 bits per weight.
            let bytes = Double(n) * Double(k) * (Double(spec.bits) + 0.5) / 8.0
            let path = gate.inKernelPath(bits: spec.bits, m: spec.m, n: n)
            let crossrow = path != "qmv_fast_impl"
            arms.append([
                "bits": spec.bits,
                "m": spec.m,
                "k": k,
                "n": n,
                "threadgroups_x": spec.m,
                "kernel": "affine_qmv_fast_bfloat16_t_gs_64_b_\(spec.bits)_batch_0",
                "in_kernel_path": path,
                "weight_bytes": bytes,
                "seconds_per_call": median,
                "seconds_per_call_min": samples[0],
                "seconds_per_call_max": samples[samples.count - 1],
                "achieved_gb_per_second": bytes / median / 1e9,
                "uses_crossrow_kernel": crossrow,
            ])
        }

        let payload: [String: Any] = [
            "source": "vendored-mlx-swift",
            "reps": reps,
            "inner_calls_per_rep": inner,
            "arm_order": "round-robin",
            "device": describeDispatchDevice(),
            "crossrow_gate": gate.describe(),
            "roofline": measureRoofline(reps: reps),
            "arms": arms,
        ]
        let data = try JSONSerialization.data(
            withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: URL(fileURLWithPath: outPath))
        print("QMV_BITS_SWEEP_OUT \(outPath)")
    }

    private static var bitsSweepEnabled: Bool {
        ProcessInfo.processInfo
            .environment["MLXFAST_RUN_QMV_BITS_SWEEP"] == "1"
    }
}

/// Bit-exactness of `quantized_matmul` across kernel arms, at the scored shapes
/// and every verify width the dispatch table distinguishes.
///
/// The cost curve's `row0_bitwise_matches_m1` only compares a build against
/// itself, so it cannot see an arm that is self-consistently wrong. This
/// digests the *whole* output of one deterministic call per (shape, width) and
/// writes it out, so `research/qmv_parity_compare.py` can diff one arm's build
/// against another's. Every input here is derived from integer arithmetic and
/// `arange`, so the digest depends on the kernel alone.
///
/// Untimed on purpose: one call per cell, no repetitions, no warmup.
///
/// The sweep covers every bit width in `parityBits`, not just 4. A gate that
/// only exercises 4-bit weights has *no power* over a change to the 3-bit
/// crossrow gate: none of its cells would execute a line of the new code, and
/// reporting it green would be a false assurance. `cells_by_bits` and
/// `covering_cells` in the payload make that power auditable rather than
/// assumed.
///
/// Enable with `MLXFAST_RUN_QMV_PARITY=1` and point `MLXFAST_QMV_PARITY_OUT`
/// at the JSON destination.
@Suite
struct QwenQMVParityTests {
    private static var enabled: Bool {
        ProcessInfo.processInfo.environment["MLXFAST_RUN_QMV_PARITY"] == "1"
    }

    @Test(.enabled(if: QwenQMVParityTests.enabled))
    func digestQuantizedMatmulOverVerifyWidth() throws {
        let env = ProcessInfo.processInfo.environment
        let outPath = try #require(
            env["MLXFAST_QMV_PARITY_OUT"],
            "MLXFAST_QMV_PARITY_OUT must name the JSON destination")
        let widths = Array(1...12)
        let parityBits = (env["MLXFAST_QMV_PARITY_BITS"] ?? "4,3")
            .split(separator: ",").compactMap { Int($0) }
        let gate = crossrowGate()

        var entries: [[String: Any]] = []
        for bits in parityBits {
            for shape in scoredShapes {
                let weight = lowBitQuantWeight(k: shape.k, n: shape.n, bits: bits)
                for m in widths {
                    let x = syntheticActivations(m: m, k: shape.k, salt: m)
                    let out = quantizedMM(
                        x, weight.w, scales: weight.scales, biases: weight.biases,
                        transpose: true, groupSize: 64, bits: bits)
                    eval(out)
                    // bf16 -> f32 is exact, so this digests the kernel's own bits.
                    let values = out.asType(.float32).asArray(Float.self)
                    entries.append([
                        "shape": shape.name,
                        "bits": bits,
                        "m": m,
                        "k": shape.k,
                        "n": shape.n,
                        "elements": values.count,
                        "in_kernel_path": gate.inKernelPath(
                            bits: bits, m: m, n: shape.n),
                        "digest": digest(values),
                    ])
                }
            }
        }

        // How many cells actually reach a crossrow body, per bit width. A zero
        // here for the bit width under test means the gate proves nothing.
        var covering: [String: Int] = [:]
        var byBits: [String: Int] = [:]
        for entry in entries {
            let bits = "\(entry["bits"] as? Int ?? -1)"
            byBits[bits, default: 0] += 1
            if (entry["in_kernel_path"] as? String) != "qmv_fast_impl" {
                covering[bits, default: 0] += 1
            }
        }

        let payload: [String: Any] = [
            "source": "vendored-mlx-swift",
            "widths": widths,
            "bits": parityBits,
            "device": describeDispatchDevice(),
            "crossrow_gate": gate.describe(),
            "cells_by_bits": byBits,
            "covering_cells_by_bits": covering,
            "entries": entries,
        ]
        let data = try JSONSerialization.data(
            withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: URL(fileURLWithPath: outPath))
        print(
            "QMV_PARITY_OUT \(outPath) cells=\(entries.count) "
                + "by_bits=\(byBits.sorted { $0.key < $1.key }) "
                + "covering=\(covering.sorted { $0.key < $1.key })")
    }
}

private func digest(_ values: [Float]) -> String {
    var hasher = SHA256()
    values.withUnsafeBytes { hasher.update(bufferPointer: $0) }
    return hasher.finalize().map { String(format: "%02x", $0) }.joined()
}

/// `syntheticQuantWeight` assumes a power-of-two `bits` when it derives the
/// packed width. Affine 3-bit packs a contiguous LSB-first bitstream whose row
/// width is `k * bits / 32` uint32 words (`mlx/ops.cpp:4862`), which is the
/// form the kernel's byte-stride view expects (`quantized.h:776`).
private func lowBitQuantWeight(k: Int, n: Int, bits: Int) -> QuantWeight {
    let words = k * bits / 32
    let tile = (0..<words).map { index -> UInt32 in
        UInt32(truncatingIfNeeded: index &* 2_654_435_761) ^ 0x9E37_79B9
    }
    let w = MLXArray(tile).reshaped([1, words]) + arange(0, n, dtype: .uint32).reshaped([n, 1])

    let groups = k / 64
    let rowJitter = arange(0, n, dtype: .float32).reshaped([n, 1]) * 1e-6
    let scaleTile: [Float] = (0..<groups).map { index -> Float in
        0.006 + 0.004 * Float((index &* 37) % 61) / 61.0
    }
    let biasTile: [Float] = (0..<groups).map { index -> Float in
        -0.05 - 0.02 * Float((index &* 23) % 53) / 53.0
    }
    let scales = (MLXArray(scaleTile).reshaped([1, groups]) + rowJitter).asType(.bfloat16)
    let biases = (MLXArray(biasTile).reshaped([1, groups]) + rowJitter).asType(.bfloat16)

    let weight = QuantWeight(w: w, scales: scales, biases: biases, bits: bits)
    eval(weight.w, weight.scales, weight.biases)
    return weight
}

// MARK: - what the scored verify actually calls

/// One affine 4-bit group-64 transposed projection the target runs per verify,
/// with the number of times a single verify pass reaches it.
private struct ScoredShape {
    let name: String
    let k: Int
    let n: Int
    let callsPerVerify: Int
}

/// Shapes read off the live target: `Qwen35.swift` (48 Gated DeltaNet layers,
/// 16 full-attention layers, fused gate/up MLP) plus the unconditional full
/// vocabulary readout. `out_proj`/`o_proj` share a shape and are listed
/// separately so the call mix stays legible.
/// `M = 1...12` brackets `vector_limit`; the decade points anchor `FLOPS_eff`
/// empirically and expose the bandwidth/compute knee instead of borrowing it
/// from the prefill measurement.
let scoredWidths: [Int] = Array(1...12) + [16, 32, 64, 128, 256, 512]

private let scoredShapes: [ScoredShape] = [
    .init(name: "linear_attn.in_proj_fused_qkvzba", k: 5120, n: 16480, callsPerVerify: 48),
    .init(name: "linear_attn.out_proj", k: 6144, n: 5120, callsPerVerify: 48),
    .init(name: "full_attn.qkv_proj_fused", k: 5120, n: 14336, callsPerVerify: 16),
    .init(name: "full_attn.o_proj", k: 6144, n: 5120, callsPerVerify: 16),
    .init(name: "mlp.gate_up_fused", k: 5120, n: 34816, callsPerVerify: 64),
    .init(name: "mlp.down", k: 17408, n: 5120, callsPerVerify: 64),
    .init(name: "head.lm_head", k: 5120, n: 248320, callsPerVerify: 1),
    // Draft-side, not part of the verify mix: the padded compact draft
    // vocabulary (`Qwen35.swift:2061-2066`, `applyDraftLMHead:2336`) runs once
    // per draft step at M=1. Carried here to settle whether its cost is the
    // ~0.6ms in the advisor's notes or the ~1.25ms its 283MB implies.
    .init(name: "head.compact_draft_vocab", k: 5120, n: 98336, callsPerVerify: 0),
]

/// `get_qmv_batch_limit` returns three different limits by (K, N) size class,
/// and the whole table shifts with the GPU architecture. Each probe range spans
/// every limit the dispatcher can return for its size class, so the boundary is
/// measured rather than assumed.
private struct BoundaryProbe {
    let shape: ScoredShape
    let widths: [Int]
}

private let dispatchBoundaryProbes: [BoundaryProbe] = [
    // small class: 32 ('d'), 18, or 14
    .init(
        shape: .init(name: "probe.k2048_n2048", k: 2048, n: 2048, callsPerVerify: 0),
        widths: Array(1...22)),
    // middle class: 18 ('d'), 12, or 10
    .init(
        shape: .init(name: "probe.k4096_n4096", k: 4096, n: 4096, callsPerVerify: 0),
        widths: Array(1...16)),
    // large class, the one every scored shape lands in: 12 ('d'), 10, or 6
    .init(
        shape: .init(name: "probe.k5120_n5120", k: 5120, n: 5120, callsPerVerify: 0),
        widths: Array(1...14)),
]

// MARK: - host dispatcher, reproduced

/// `get_qmv_batch_limit` (`mlx/backend/metal/quantized.cpp:84-124`) keys off the
/// last character of the Metal architecture name and the two digits before it.
/// The brief's `vector_limit ~ 10` is only the `arch_gen != 13, 14` row; a gen-13
/// or gen-14 part returns 6 for every scored shape, which would put verify
/// widths 7, 8 and 9 on `qmm_t_splitk` already. Reproducing the table here
/// records which row this host is actually on next to the measured curve.
private struct HostDispatch {
    let architecture: String
    let archClass: Character
    /// `Device::Device` (`device.cpp:565-572`) reads the generation from the two
    /// digits before the trailing size character, e.g. `applegpu_g15p` -> 15.
    let archGen: Int
    /// `is_nax_available` (`device.cpp:918-928`): macOS 26.2+ and a new enough
    /// generation. `_nax` kernels run on the ranked M5; a host that misses this
    /// gate is not executing the same quantized family the ranked run does.
    let naxAvailable: Bool

    init() {
        let arch = GPU.deviceInfo().architecture
        architecture = arch
        archClass = arch.last ?? " "
        var gen = 0
        if arch.count >= 3 {
            let chars = Array(arch)
            let tens = chars[chars.count - 3].wholeNumberValue ?? 0
            let ones = chars[chars.count - 2].wholeNumberValue ?? 0
            gen = (tens >= 0 && tens < 10 ? tens : 0) * 10 + (ones >= 0 && ones < 10 ? ones : 0)
        }
        archGen = gen
        var osReady = false
        if #available(macOS 26.2, *) { osReady = true }
        naxAvailable = osReady && gen >= (archClass == "p" ? 18 : 17)
    }

    func vectorLimit(k: Int, n: Int) -> Int {
        let small = k <= 2048 && n <= 2048
        let middle = k <= 4096 && n <= 4096
        if archClass == "d" { return small ? 32 : (middle ? 18 : 12) }
        if archGen == 13 || archGen == 14 { return small ? 14 : (middle ? 10 : 6) }
        return small ? 18 : (middle ? 12 : 10)
    }
}

private let hostDispatch = HostDispatch()

/// The crossrow gate as it actually reads in the vendored header, not as this
/// test remembers it.
///
/// `affine_qmv_fast` decides between the crossrow specializations and
/// `qmv_fast_impl` with a compile-time `if` on `group_size`, `bits`,
/// `batched`, and `out_vec_size`, then a `switch (ntg.x)` whose case labels
/// bound the row counts it covers. Hardcoding `bits == 4` into the reporting
/// side means any widening of that gate is silently mislabelled as a fallback,
/// which is exactly the mistake a bit-width experiment cannot afford. So the
/// gate is parsed back out of the header that this build compiled.
struct CrossrowGate {
    var headerPath: String
    var headerSha256: String
    /// Bit widths admitted by an `... group_size == 64 && bits == B ...` clause
    /// inside `affine_qmv_fast`, keyed by the innermost `out_vec_size >= bound`
    /// tier in effect, then by `case` label, then to the crossrow helper that
    /// case actually calls including its template arguments. The tier level
    /// matters: the promoted 4-bit gate calls a different helper for the same
    /// width above and below 4096 outputs.
    var coveredRows: [Int: [Int: [Int: String]]]
    var minOutVecSize: Int
    var gateLines: [Int]

    func inKernelPath(bits: Int, m: Int, n: Int) -> String {
        guard n >= minOutVecSize, let tiers = coveredRows[bits],
            let bound = tiers.keys.filter({ n >= $0 }).max(),
            let callee = tiers[bound]?[m]
        else { return "qmv_fast_impl" }
        return callee
    }

    func describe() -> [String: Any] {
        [
            "header_path": headerPath,
            "header_sha256": headerSha256,
            "min_out_vec_size": minOutVecSize,
            "gate_lines": gateLines,
            "covered_rows_by_bits_by_out_vec_tier": coveredRows
                .reduce(into: [String: [String: [String: String]]]()) { out, entry in
                    out["\(entry.key)"] = entry.value.reduce(into: [:]) { tiers, tier in
                        tiers["\(tier.key)"] = tier.value.reduce(into: [:]) {
                            $0["\($1.key)"] = $1.value
                        }
                    }
                },
        ]
    }
}

/// Locates the vendored header relative to this source file and extracts the
/// gate. Failing to find or parse it is a hard failure: silently defaulting to
/// "4 bits only" would reintroduce the bug this exists to prevent.
private func crossrowGate(file: StaticString = #filePath) -> CrossrowGate {
    var dir = URL(fileURLWithPath: "\(file)").deletingLastPathComponent()
    let suffix =
        "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h"
    var header: URL?
    while dir.path != "/" {
        let candidate = dir.appendingPathComponent(suffix)
        if FileManager.default.fileExists(atPath: candidate.path) {
            header = candidate
            break
        }
        dir = dir.deletingLastPathComponent()
    }
    guard let header, let text = try? String(contentsOf: header, encoding: .utf8)
    else { fatalError("cannot locate vendored \(suffix) from \(file)") }

    let lines = text.components(separatedBy: "\n")
    guard let kernelStart = lines.firstIndex(where: {
        $0.contains("affine_qmv_fast(") || $0.contains("[[kernel]] void affine_qmv_fast")
    }) else { fatalError("affine_qmv_fast not found in \(header.path)") }

    var coveredRows: [Int: [Int: [Int: String]]] = [:]
    var minOutVecSize = Int.max
    var gateLines: [Int] = []
    var activeBits: Int?
    var outerBound = 0
    var tier = 0
    var pendingLabel: Int?
    for offset in kernelStart..<lines.count {
        let line = lines[offset]
        if line.hasPrefix("}") && offset > kernelStart + 4 { break }
        if line.trimmingCharacters(in: .whitespaces).hasPrefix("//") { continue }
        if line.contains("group_size == 64"), line.contains("!batched"),
            let bits = firstInt(in: line, after: "bits == "),
            let bound = firstInt(in: line, after: "out_vec_size >= ") {
            activeBits = bits
            outerBound = bound
            tier = bound
            pendingLabel = nil
            gateLines.append(offset + 1)
            minOutVecSize = min(minOutVecSize, bound)
            coveredRows[bits, default: [:]][bound] = coveredRows[bits]?[bound] ?? [:]
            continue
        }
        guard let bits = activeBits else { continue }
        // A nested `if (out_vec_size >= HI)` narrows the tier for the switch it
        // guards; its `else` restores the enclosing gate bound.
        if line.contains("if ("), let hi = firstInt(in: line, after: "out_vec_size >= ") {
            tier = hi
            pendingLabel = nil
            coveredRows[bits, default: [:]][hi] = coveredRows[bits]?[hi] ?? [:]
            continue
        }
        if line.contains("} else {") {
            tier = outerBound
            pendingLabel = nil
            continue
        }
        if let label = firstInt(in: line, after: "case ") {
            pendingLabel = label
            continue
        }
        // The callee may sit many lines below its label behind a comment block,
        // so scan forward until the call rather than a fixed window.
        if let label = pendingLabel, let callee = crossrowCallee(in: line) {
            coveredRows[bits, default: [:]][tier, default: [:]][label] = callee
            pendingLabel = nil
        }
    }
    guard !coveredRows.isEmpty, minOutVecSize != Int.max else {
        fatalError("no crossrow gate parsed from \(header.path)")
    }
    return CrossrowGate(
        headerPath: suffix,
        headerSha256: sha256Hex(Data(text.utf8)),
        coveredRows: coveredRows,
        minOutVecSize: minOutVecSize,
        gateLines: gateLines)
}

private func firstInt(in line: String, after marker: String) -> Int? {
    guard let range = line.range(of: marker) else { return nil }
    let digits = line[range.upperBound...].prefix { $0.isNumber }
    return digits.isEmpty ? nil : Int(digits)
}

private func identifier(in line: String, startingWith prefix: String) -> String? {
    guard let range = line.range(of: prefix) else { return nil }
    let rest = line[range.lowerBound...].prefix { $0.isLetter || $0.isNumber || $0 == "_" }
    return String(rest)
}

/// The crossrow helper invoked on `line`, with its template arguments, because
/// the inputs-per-group and DIRECT_NIBBLES arguments distinguish call sites that
/// share a name.
private func crossrowCallee(in line: String) -> String? {
    guard let range = line.range(of: "qmv_fast_crossrow") else { return nil }
    let tail = line[range.lowerBound...]
    let name = tail.prefix { $0.isLetter || $0.isNumber || $0 == "_" }
    guard let args = tail.dropFirst(name.count).first, args == "<",
        let close = tail.firstIndex(of: ">")
    else { return String(name) }
    return String(tail[tail.startIndex...close])
}

private func sha256Hex(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}

private func describeDispatchDevice() -> [String: Any] {
    let info = GPU.deviceInfo()
    return [
        "architecture": hostDispatch.architecture,
        "architecture_class": String(hostDispatch.archClass),
        "architecture_gen": hostDispatch.archGen,
        "nax_available": hostDispatch.naxAvailable,
        "memory_size_bytes": info.memorySize,
        "max_buffer_size_bytes": info.maxBufferSize,
        "predicted_vector_limits": (scoredShapes + dispatchBoundaryProbes.map(\.shape)
            + fastPathProbes).map {
            [
                "shape": $0.name, "k": $0.k, "n": $0.n,
                "vector_limit": hostDispatch.vectorLimit(k: $0.k, n: $0.n),
            ].merging(qmvFastAlignment(k: $0.k, n: $0.n)) { a, _ in a }
        },
    ]
}

/// Vendored MLX 0.32.0 forks `qmv` on `N % 8 == 0 && K % 512 == 0` at
/// `quantized.cpp:259` (and `gather_qmv` at `:992`) with the 512 hardcoded for
/// every bit width. There is no bits-dependent `qmv_fast_k_alignment` helper in
/// this tree, so an 8-bit weight gets no alignment relief over a 4-bit one.
private func qmvFastAlignment(k: Int, n: Int) -> [String: Any] {
    [
        "k_mod_512": k % 512,
        "n_mod_8": n % 8,
        "qmv_fast": n % 8 == 0 && k % 512 == 0,
    ]
}

/// `qmv` picks its fast variant on `N % 8 == 0 && K % 512 == 0`. 5120 and 5632
/// are multiples of 512; 5184 is not. A step between them at fixed N is the
/// fast-path selection becoming visible.
private let fastPathProbes: [ScoredShape] = [
    .init(name: "fastprobe.k5120_n16480", k: 5120, n: 16480, callsPerVerify: 0),
    .init(name: "fastprobe.k5184_n16480", k: 5184, n: 16480, callsPerVerify: 0),
    .init(name: "fastprobe.k5632_n16480", k: 5632, n: 16480, callsPerVerify: 0),
]

// MARK: - measurement

private func sweep(
    shape: ScoredShape,
    widths: [Int],
    reps: Int,
    inner: Int
) -> [String: Any] {
    let weight = syntheticQuantWeight(k: shape.k, n: shape.n)
    var rows: [[String: Any]] = []
    var referenceRow0: [Float]? = nil

    for m in widths {
        // Wide widths are pure GPU work, so batching many of them per timed
        // region buys nothing and costs minutes. Shrink the batch and the
        // repetition count once the per-call cost dwarfs launch overhead.
        let innerForM = m <= 12 ? inner : max(2, (inner * 12) / m)
        let repsForM = m <= 12 ? reps : max(5, reps / 3)
        let xs = (0..<innerForM).map { syntheticActivations(m: m, k: shape.k, salt: $0) }
        eval(xs)

        // The scored verify pass is a strictly dependent chain: every layer's
        // quantized matmul consumes the previous layer's output. MLX only emits
        // a Metal barrier between kernels that actually share a buffer
        // (`CommandEncoder::maybeInsertBarrier`, device.cpp:324-364) and encodes
        // with `MTL::DispatchTypeConcurrent` (device.cpp:548), so independent
        // calls batched into one `eval` overlap and understate small-M cost --
        // exactly the regime this experiment is trying to measure. Threading a
        // 1x1 tap of each output into the next input restores that dependency.
        // The tap is scaled to 1e-30 so it vanishes in bf16 rounding: the graph
        // edge is real, the activations are bitwise unchanged.
        let chained = {
            var outs: [MLXArray] = []
            outs.reserveCapacity(innerForM)
            var x = xs[0]
            for i in 0..<innerForM {
                let o = quantizedMM(
                    x, weight.w, scales: weight.scales, biases: weight.biases,
                    transpose: true, groupSize: 64, bits: 4)
                outs.append(o)
                if i + 1 < innerForM { x = xs[i + 1] + o[0..<1, 0..<1] * 1e-30 }
            }
            return outs
        }
        // Same graph minus the matmuls: isolates the tap scaffolding so the
        // chained number can be reported with its own overhead quantified.
        let tapsOnly = {
            var outs: [MLXArray] = []
            outs.reserveCapacity(innerForM)
            var x = xs[0]
            for i in 0..<innerForM {
                outs.append(x)
                if i + 1 < innerForM { x = xs[i + 1] + x[0..<1, 0..<1] * 1e-30 }
            }
            return outs
        }
        let concurrent = {
            xs.map {
                quantizedMM(
                    $0, weight.w, scales: weight.scales, biases: weight.biases,
                    transpose: true, groupSize: 64, bits: 4)
            }
        }

        let samples = medianSpread(reps: repsForM, inner: innerForM, warmup: 3, body: chained)
        let conc = medianSpread(reps: repsForM, inner: innerForM, warmup: 3, body: concurrent)
        let taps = medianSpread(reps: repsForM, inner: innerForM, warmup: 3, body: tapsOnly)

        let row0 = quantizedMM(
            xs[0], weight.w, scales: weight.scales, biases: weight.biases,
            transpose: true, groupSize: 64, bits: 4)[0]
            .asType(.float32).asArray(Float.self)
        if referenceRow0 == nil { referenceRow0 = row0 }

        rows.append([
            "m": m,
            "seconds_per_call": samples[samples.count / 2],
            "seconds_per_call_min": samples[0],
            "seconds_per_call_max": samples[samples.count - 1],
            "seconds_per_call_concurrent": conc[conc.count / 2],
            "tap_overhead_seconds_per_call": taps[taps.count / 2],
            "calls_per_timed_region": innerForM,
            "timed_regions": repsForM,
            "row0_bitwise_matches_m1": row0 == referenceRow0!,
            "row0_max_abs_delta_vs_m1": maxAbsDelta(row0, referenceRow0!),
        ])
    }

    return [
        "name": shape.name,
        "k": shape.k,
        "n": shape.n,
        "calls_per_verify": shape.callsPerVerify,
        "weight_bytes": quantWeightBytes(k: shape.k, n: shape.n),
        "flops_per_row": 2 * shape.k * shape.n,
        "predicted_vector_limit": hostDispatch.vectorLimit(k: shape.k, n: shape.n),
        "rows": rows,
    ].merging(qmvFastAlignment(k: shape.k, n: shape.n)) { a, _ in a }
}

/// Affine group-64: packed weights plus a bf16 scale and bias per group.
private func quantWeightBytes(k: Int, n: Int, bits: Int = 4) -> Int {
    n * k * bits / 8 + 2 * (n * (k / 64) * 2)
}

// MARK: - proposal-head dtype

/// The head's widest projection, timed bf16 against affine group-64 at 8 and 4
/// bits, at the same shape and the same width.
///
/// Two artifacts claim to be the proposal head on this base and they disagree by
/// 3.556x in bytes: `fixtures/qwen3_8_27b_mtp_track.json` pins 849,398,784 bf16
/// tensor bytes and `setup-qwen-mtp.sh` fetches that tree unconditionally, while
/// `mtp-head.manifest.json` declares a 238,934,093-byte 4-bit group-64
/// requantization that nothing in the local path reads. Whether requantizing the
/// head buys its byte ratio back as time is a decode-side question this measures
/// directly instead of inferring from the backbone curve. The 8-bit point is
/// what decides whether the shipped 4-bit head is on the right side of the
/// nibble-unpacking tradeoff.
private let headFCShape = (k: 5120, n: 10240)

private func measureHeadFCDtypes(reps: Int, inner: Int) -> [String: Any] {
    let (k, n) = headFCShape
    let x = syntheticActivations(m: 1, k: k, salt: 3)
    eval(x)

    // [K, N] contiguous is the fastest legal bf16 layout, so the quantized side
    // is compared against the strongest dense form rather than a strided view.
    let dense = zeros([k, n], dtype: .bfloat16) + Float(0.01)
    eval(dense)
    let denseSamples = medianSpread(reps: reps, inner: inner, warmup: 3) {
        (0..<inner).map { _ in x.matmul(dense) }
    }
    let denseBytes = k * n * 2
    let denseSeconds = denseSamples[denseSamples.count / 2]

    var payload: [String: Any] = [
        "k": k,
        "n": n,
        "m": 1,
        "bf16_weight_bytes": denseBytes,
        "bf16_seconds_per_call": denseSeconds,
        "bf16_effective_bandwidth_bytes_per_second": Double(denseBytes) / denseSeconds,
    ]

    for bits in [8, 4] {
        let quant = syntheticQuantWeight(k: k, n: n, bits: bits)
        let samples = medianSpread(reps: reps, inner: inner, warmup: 3) {
            (0..<inner).map { _ in
                quantizedMM(
                    x, quant.w, scales: quant.scales, biases: quant.biases,
                    transpose: true, groupSize: 64, bits: bits)
            }
        }
        let bytes = quantWeightBytes(k: k, n: n, bits: bits)
        let seconds = samples[samples.count / 2]
        let tag = "q\(bits)g64"
        payload["\(tag)_weight_bytes"] = bytes
        payload["\(tag)_seconds_per_call"] = seconds
        payload["\(tag)_seconds_per_call_min"] = samples[0]
        payload["\(tag)_seconds_per_call_max"] = samples[samples.count - 1]
        payload["byte_ratio_bf16_over_\(tag)"] = Double(denseBytes) / Double(bytes)
        payload["time_ratio_bf16_over_\(tag)"] = denseSeconds / seconds
        payload["\(tag)_effective_bandwidth_bytes_per_second"] = Double(bytes) / seconds
    }

    if let q8 = payload["q8g64_seconds_per_call"] as? Double,
        let q4 = payload["q4g64_seconds_per_call"] as? Double
    {
        payload["time_ratio_q8g64_over_q4g64"] = q8 / q4
        payload["byte_ratio_q8g64_over_q4g64"] =
            Double(quantWeightBytes(k: k, n: n, bits: 8))
            / Double(quantWeightBytes(k: k, n: n, bits: 4))
    }
    return payload
}

// MARK: - Gated DeltaNet recurrence

/// The other half of every verify round: 48 recurrent layers whose cost is state
/// traffic, not projection traffic.
///
/// A GDN step reads and writes the whole `[1, Hv, Dk, Dv]` fp32 state regardless
/// of width, so its arithmetic intensity is roughly `2 * T` FLOP/byte against a
/// machine balance in the high twenties. That predicts an almost flat curve over
/// the verify widths, which is the opposite of a quantized projection and is why
/// the two are reported side by side rather than summed.
///
/// The chain is the real recurrence: each call consumes the previous call's
/// state, so nothing here overlaps that would not overlap in the scored round.
private func sweepGatedDelta(widths: [Int], reps: Int, inner: Int) -> [String: Any] {
    let hk = 16, dk = 128, hv = 48, dv = 128
    let aLog = (zeros([hv], dtype: .float32) + Float(-0.5)).asType(.bfloat16)
    let dtBias = (zeros([hv], dtype: .float32) + Float(0.1)).asType(.bfloat16)
    let state0 = zeros([1, hv, dk, dv], dtype: .float32)
    eval(aLog, dtBias, state0)

    var rows: [[String: Any]] = []
    for m in widths {
        let q = gdnInput([1, m, hk, dk], salt: 1)
        let kk = gdnInput([1, m, hk, dk], salt: 2)
        let v = gdnInput([1, m, hv, dv], salt: 3)
        let a = gdnInput([1, m, hv], salt: 4)
        let b = gdnInput([1, m, hv], salt: 5)
        eval(q, kk, v, a, b)

        let samples = medianSpread(reps: reps, inner: inner, warmup: 3) {
            var outs: [MLXArray] = []
            var state = state0
            for _ in 0..<inner {
                let (y, next) = gatedDeltaUpdate(
                    q: q, k: kk, v: v, a: a, b: b,
                    aLog: aLog, dtBias: dtBias, state: state)
                outs.append(y)
                state = next
            }
            outs.append(state)
            return outs
        }

        let stateBytes = hv * dk * dv * 4
        let ioBytes = 2 * m * hk * dk * 2 + m * hv * dv * 2 * 2
        rows.append([
            "m": m,
            "seconds_per_call": samples[samples.count / 2],
            "seconds_per_call_min": samples[0],
            "seconds_per_call_max": samples[samples.count - 1],
            "calls_per_timed_region": inner,
            "timed_regions": reps,
            "traffic_bytes": 2 * stateBytes + ioBytes,
            "flops": 4 * m * hv * dk * dv,
        ])
    }

    return [
        "name": "linear_attn.gated_delta_recurrence",
        "calls_per_verify": 48,
        "num_k_heads": hk,
        "num_v_heads": hv,
        "head_k_dim": dk,
        "head_v_dim": dv,
        "state_bytes_per_layer": hv * dk * dv * 4,
        "rows": rows,
    ]
}

private func gdnInput(_ shape: [Int], salt: Int) -> MLXArray {
    let count = shape.reduce(1, *)
    let values: [Float] = (0..<count).map { index in
        Float((index &* 97 &+ salt &* 6151) % 211) / 211.0 - 0.5
    }
    return MLXArray(values).reshaped(shape).asType(.bfloat16)
}

private func maxAbsDelta(_ a: [Float], _ b: [Float]) -> Double {
    guard a.count == b.count else { return .infinity }
    var worst = Float(0)
    for i in a.indices { worst = max(worst, abs(a[i] - b[i])) }
    return Double(worst)
}

private struct QuantWeight {
    let w: MLXArray
    let scales: MLXArray
    let biases: MLXArray
    let bits: Int
}

/// Builds a contiguous quantized weight without materializing a dense source.
/// A per-row broadcast add both varies the data across rows and forces the
/// broadcast tile into real row-contiguous storage, so `quantized_matmul` never
/// pays a hidden contiguity copy inside the timed region.
private func syntheticQuantWeight(k: Int, n: Int, bits: Int = 4) -> QuantWeight {
    let words = k / (32 / bits)
    let tile = (0..<words).map { index -> UInt32 in
        UInt32(truncatingIfNeeded: index &* 2_654_435_761) ^ 0x9E37_79B9
    }
    let w = MLXArray(tile).reshaped([1, words]) + arange(0, n, dtype: .uint32).reshaped([n, 1])

    let groups = k / 64
    let rowJitter = arange(0, n, dtype: .float32).reshaped([n, 1]) * 1e-6
    let scaleTile: [Float] = (0..<groups).map { index -> Float in
        let step = Float((index &* 37) % 61)
        return 0.006 + 0.004 * step / 61.0
    }
    let biasTile: [Float] = (0..<groups).map { index -> Float in
        let step = Float((index &* 23) % 53)
        return -0.05 - 0.02 * step / 53.0
    }
    let scales = (MLXArray(scaleTile).reshaped([1, groups]) + rowJitter).asType(.bfloat16)
    let biases = (MLXArray(biasTile).reshaped([1, groups]) + rowJitter).asType(.bfloat16)

    let weight = QuantWeight(w: w, scales: scales, biases: biases, bits: bits)
    eval(weight.w, weight.scales, weight.biases)
    return weight
}

private func syntheticActivations(m: Int, k: Int, salt: Int) -> MLXArray {
    let tile: [Float] = (0..<k).map { index -> Float in
        Float((index &* 131 &+ salt &* 7919) % 251) / 251.0 - 0.5
    }
    let rowJitter = arange(0, m, dtype: .float32).reshaped([m, 1]) * 0.01
    return (MLXArray(tile).reshaped([1, k]) + rowJitter).asType(.bfloat16)
}

/// Roofline constants for `qmv_tax`, measured on this host rather than quoted:
/// a large streaming elementwise op for bandwidth and a dense bf16 GEMM for the
/// arithmetic ceiling.
private func measureRoofline(reps: Int) -> [String: Any] {
    let streamElements = 256 * 1024 * 1024
    let stream = zeros([streamElements], dtype: .bfloat16)
    eval(stream)
    let streamSeconds = median(reps: reps) { eval(stream + Float(1)) }

    let gemmDim = 4096
    let a = zeros([gemmDim, gemmDim], dtype: .bfloat16)
    let b = zeros([gemmDim, gemmDim], dtype: .bfloat16)
    eval(a, b)
    let gemmSeconds = median(reps: reps) { eval(a.matmul(b)) }

    return [
        "stream_bytes": 2 * 2 * streamElements,
        "stream_seconds": streamSeconds,
        "peak_bandwidth_bytes_per_second": Double(2 * 2 * streamElements) / streamSeconds,
        "gemm_flops": 2 * gemmDim * gemmDim * gemmDim,
        "gemm_seconds": gemmSeconds,
        "peak_flops_per_second": Double(2 * gemmDim * gemmDim * gemmDim) / gemmSeconds,
    ]
}

/// Sorted per-call seconds over `reps` timed regions, each of which builds and
/// evaluates `inner` calls. Graph construction is inside the timed region for
/// every closure, so the taps-only closure subtracts it honestly.
private func medianSpread(
    reps: Int,
    inner: Int,
    warmup: Int,
    body: () -> [MLXArray]
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
    samples.sort()
    return samples
}

private func median(reps: Int, _ body: () -> Void) -> Double {
    for _ in 0..<3 { body() }
    var samples: [Double] = []
    for _ in 0..<reps {
        let start = DispatchTime.now().uptimeNanoseconds
        body()
        samples.append(Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9)
    }
    samples.sort()
    return samples[samples.count / 2]
}
