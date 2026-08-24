import Foundation
import MLX
import MLXRandom
import Testing

// E184 probe -- what does the Gated DeltaNet recurrent scan cost at the SEED
// width, and what would a chunk-parallel form cost instead?
//
// THE MECHANISM UNDER STUDY. `GatedDelta.swift:60` is a Metal kernel whose body
// is `for (int t = 0; t < T; ++t)`. Its launch grid is `(32, Dv, B * Hv)` with
// threadgroup `(32, 4, 1)` -- the grid carries NO time dimension, so the
// available parallelism at T = 512 is exactly the parallelism at T = 1 and the
// 512 timesteps run as one serial dependency chain inside a single launch. The
// seed prefill takes the "standard single-chunk path"
// (`Qwen35.swift:1291-1300`) at S = 512, so each of the 48 linear-attention
// layers pays one such 512-step chain per forward.
//
// WHY IT IS INVISIBLE TO FLOP ACCOUNTING. The scan's arithmetic is
// ~6 * T * Hv * Dv * Dk flops per layer = 2.4 GFLOP, against 24.9 TFLOP for
// the projection GEMMs of the whole seed forward: a 0.44 % static share. Only
// wall time can price it, which is what this probe measures.
//
// ARMS.
//   scan_T{n}     the shipped kernel at real T = n. The slope over n is the
//                 per-timestep serial cost; the intercept is launch + state
//                 load/store.
//   scan_T512_L1  inputs and output shaped for T = 512 but the kernel's `T`
//                 scalar set to 1. Identical dispatch, identical buffers, one
//                 loop iteration. `scan_T512 - scan_T512_L1` is the loop's own
//                 cost with launch, allocation and state traffic removed.
//                 Output rows 1... are left uninitialised: this arm is a TIMING
//                 arm and its values are never read.
//   chunk{C}_proxy   a batched-matmul lower bound on the chunk-parallel delta
//                 rule at chunk length C: the three per-chunk matmuls
//                 (intra-chunk QK^T and its V product, the cross-chunk query
//                 against the carried state, and the state update K^T dV).
//                 It EXCLUDES the WY / UT-transform triangular inverse a
//                 correct chunked delta rule also needs, so it is a lower
//                 bound on the alternative's cost, not an implementation.
//
// Research instrument. Off unless `MLXFAST_RUN_E184_PROBE=1`. `Tests/` is never
// packaged into a submission. Within-session relative measurement; every arm is
// run forward and reverse (ABBA) inside one session, and
// `cool_gate_passed_real_gate=false` / `gate_qualified_for_timing=false` hold
// for every number it prints.

private struct E184Cell: Codable {
    var arm: String
    var timesteps: Int
    var microseconds: Double
    var forwardMicroseconds: Double
    var reverseMicroseconds: Double
    var replicates: Int
}

struct E184GatedDeltaScanCostTests {
    static let enabled =
        ProcessInfo.processInfo.environment["MLXFAST_RUN_E184_PROBE"] == "1"

    // Qwen 3.8 27B linear-attention geometry (weights/config.json):
    // linear_num_key_heads 16, linear_num_value_heads 48, key/value head dim
    // 128, 48 of the 64 layers are linear_attention.
    static let kHeads = 16
    static let vHeads = 48
    static let headDim = 128
    static let linearLayers = 48
    static let seedLength = 512

    static var timesteps: [Int] {
        guard let raw = ProcessInfo.processInfo.environment["MLXFAST_E184_T"],
              !raw.isEmpty
        else { return [1, 2, 8, 32, 64, 128, 256, 512] }
        return raw.split(separator: ",").compactMap { Int($0) }
    }

    static var chunkLengths: [Int] {
        guard let raw = ProcessInfo.processInfo.environment["MLXFAST_E184_CHUNKS"],
              !raw.isEmpty
        else { return [64, 128] }
        return raw.split(separator: ",").compactMap { Int($0) }
    }

    static var rampSeconds: Double {
        Double(ProcessInfo.processInfo.environment["MLXFAST_E184_RAMP_S"] ?? "")
            ?? 0.30
    }

    static var targetMicroseconds: Double {
        Double(ProcessInfo.processInfo.environment["MLXFAST_E184_TARGET_US"] ?? "")
            ?? 60_000
    }

    /// Byte-identical copy of `makeGatedDeltaKernel(hasMask: false)`
    /// (`Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/GatedDelta.swift:27-95`).
    /// The function is internal to `MLXLLM`, so the probe rebuilds the same
    /// source string: same source, same JIT kernel.
    static let scanKernel = MLXFast.metalKernel(
        name: "e184_gated_delta_step",
        inputNames: ["q", "k", "v", "g", "beta", "state_in", "T"],
        outputNames: ["y", "state_out"],
        source: """
                auto n = thread_position_in_grid.z;
                auto b_idx = n / Hv;
                auto hv_idx = n % Hv;
                auto hk_idx = hv_idx / (Hv / Hk);
                constexpr int n_per_t = Dk / 32;

                // q, k: [B, T, Hk, Dk]
                auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
                auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

                // v, y: [B, T, Hv, Dv]
                auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
                y += b_idx * T * Hv * Dv + hv_idx * Dv;

                auto dk_idx = thread_position_in_threadgroup.x;
                auto dv_idx = thread_position_in_grid.y;

                // g: [B, T, Hv]
                auto g_ = g + b_idx * T * Hv;
                auto beta_ = beta + b_idx * T * Hv;

                // state_in, state_out: [B, Hv, Dv, Dk]
                auto i_state = state_in + (n * Dv + dv_idx) * Dk;
                auto o_state = state_out + (n * Dv + dv_idx) * Dk;

                float state[n_per_t];
                for (int i = 0; i < n_per_t; ++i) {
                  auto s_idx = n_per_t * dk_idx + i;
                  state[i] = static_cast<float>(i_state[s_idx]);
                }

                for (int t = 0; t < T; ++t) {
                  if (true) {
                    float kv_mem = 0.0f;
                    for (int i = 0; i < n_per_t; ++i) {
                      auto s_idx = n_per_t * dk_idx + i;
                      state[i] = state[i] * g_[hv_idx];
                      kv_mem += state[i] * k_[s_idx];
                    }
                    kv_mem = simd_sum(kv_mem);

                    auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

                    float out = 0.0f;
                    for (int i = 0; i < n_per_t; ++i) {
                      auto s_idx = n_per_t * dk_idx + i;
                      state[i] = state[i] + k_[s_idx] * delta;
                      out += state[i] * q_[s_idx];
                    }
                    out = simd_sum(out);
                    if (thread_index_in_simdgroup == 0) {
                      y[dv_idx] = static_cast<InT>(out);
                    }
                  } else {
                    y[dv_idx] = static_cast<InT>(0);
                  }
                  // Increment data pointers to next time step
                  q_ += Hk * Dk;
                  k_ += Hk * Dk;
                  v_ += Hv * Dv;
                  y += Hv * Dv;
                  g_ += Hv;
                  beta_ += Hv;
                }
                for (int i = 0; i < n_per_t; ++i) {
                  auto s_idx = n_per_t * dk_idx + i;
                  o_state[s_idx] = static_cast<StT>(state[i]);
                }
            """
    )

    private static func bf16(_ shape: [Int], _ seed: UInt64) -> MLXArray {
        MLXRandom.normal(shape, key: MLXRandom.key(seed)).asType(.bfloat16)
    }

    private static func fp32(_ shape: [Int], _ seed: UInt64) -> MLXArray {
        MLXRandom.normal(shape, key: MLXRandom.key(seed))
    }

    private static func timed(_ count: Int, _ body: () -> [MLXArray]) -> Double {
        let start = DispatchTime.now().uptimeNanoseconds
        for _ in 0 ..< count { eval(body()) }
        return Double(DispatchTime.now().uptimeNanoseconds - start) / 1e3
            / Double(count)
    }

    /// A cold GPU pays a fixed DVFS ramp that ABBA cannot cancel, so burn a
    /// fixed wall-clock duration first and discard it.
    private static func ramp(_ body: () -> [MLXArray], seconds: Double) {
        let start = DispatchTime.now().uptimeNanoseconds
        while Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9 < seconds {
            eval(body())
        }
    }

    /// One shipped-kernel dispatch at `outputTimesteps` output rows, running
    /// `loopTimesteps` iterations of the serial loop.
    private static func scanArm(
        outputTimesteps t: Int, loopTimesteps loop: Int
    ) -> () -> [MLXArray] {
        let q = bf16([1, t, kHeads, headDim], 0x5184_0001)
        let k = bf16([1, t, kHeads, headDim], 0x5184_0002)
        let v = bf16([1, t, vHeads, headDim], 0x5184_0003)
        let g = fp32([1, t, vHeads], 0x5184_0004)
        let beta = fp32([1, t, vHeads], 0x5184_0005)
        let state = fp32([1, vHeads, headDim, headDim], 0x5184_0006)
        let loopScalar = MLXArray(loop)
        eval(q, k, v, g, beta, state, loopScalar)
        return {
            scanKernel(
                [q, k, v, g, beta, state, loopScalar],
                template: [
                    ("InT", DType.bfloat16),
                    ("StT", DType.float32),
                    ("Dk", headDim),
                    ("Dv", headDim),
                    ("Hk", kHeads),
                    ("Hv", vHeads),
                ],
                grid: (32, headDim, vHeads),
                threadGroup: (32, 4, 1),
                outputShapes: [[1, t, vHeads, headDim], [1, vHeads, headDim, headDim]],
                outputDTypes: [.bfloat16, .float32]
            )
        }
    }

    /// Lower bound on a chunk-parallel delta rule over the same seed length:
    /// the three batched matmuls per chunk, without the triangular inverse.
    private static func chunkedProxyArm(chunk c: Int) -> () -> [MLXArray] {
        let chunks = seedLength / c
        // Head-major, key heads broadcast to value heads the way the recurrence
        // reads them (Hv / Hk = 3 value heads per key head).
        let q = bf16([vHeads, seedLength, headDim], 0x5184_0011)
        let k = bf16([vHeads, seedLength, headDim], 0x5184_0012)
        let v = bf16([vHeads, seedLength, headDim], 0x5184_0013)
        let state = bf16([vHeads, headDim, headDim], 0x5184_0014)
        eval(q, k, v, state)
        return {
            var outputs: [MLXArray] = []
            var carried = state
            for index in 0 ..< chunks {
                let lo = index * c
                let hi = lo + c
                let qc = q[0..., lo ..< hi, 0...]
                let kc = k[0..., lo ..< hi, 0...]
                let vc = v[0..., lo ..< hi, 0...]
                // intra-chunk score and its value product
                let scores = matmul(qc, kc.transposed(0, 2, 1))
                let intra = matmul(scores, vc)
                // cross-chunk contribution from the carried state
                let inter = matmul(qc, carried)
                // state update
                carried = carried + matmul(kc.transposed(0, 2, 1), vc)
                outputs.append(intra + inter)
            }
            outputs.append(carried)
            return outputs
        }
    }

    @Test(
        "E184: price the gated-delta serial scan at the 512-token seed width",
        .enabled(if: E184GatedDeltaScanCostTests.enabled))
    func scanCost() throws {
        var cells: [E184Cell] = []

        var arms: [(String, Int, () -> [MLXArray])] = []
        for t in Self.timesteps {
            arms.append(("scan_T\(t)", t, Self.scanArm(outputTimesteps: t, loopTimesteps: t)))
        }
        arms.append(
            ("scan_T512_L1", Self.seedLength,
             Self.scanArm(outputTimesteps: Self.seedLength, loopTimesteps: 1)))
        for c in Self.chunkLengths {
            arms.append(
                ("chunk\(c)_proxy", Self.seedLength, Self.chunkedProxyArm(chunk: c)))
        }

        Self.ramp(arms[arms.count - 1].2, seconds: Self.rampSeconds)

        // ABBA: forward pass over the arm list, then reverse, so monotone
        // thermal drift cancels to first order within this session.
        var forward: [String: Double] = [:]
        var reverse: [String: Double] = [:]
        var replicateCount: [String: Int] = [:]

        for (name, _, body) in arms {
            let probe = Self.timed(1, body)
            let replicates = max(3, min(200, Int(Self.targetMicroseconds / max(probe, 1))))
            replicateCount[name] = replicates
            forward[name] = Self.timed(replicates, body)
        }
        for (name, _, body) in arms.reversed() {
            reverse[name] = Self.timed(replicateCount[name] ?? 3, body)
        }

        for (name, t, _) in arms {
            let f = forward[name] ?? 0
            let r = reverse[name] ?? 0
            cells.append(
                E184Cell(
                    arm: name, timesteps: t, microseconds: (f + r) / 2,
                    forwardMicroseconds: f, reverseMicroseconds: r,
                    replicates: replicateCount[name] ?? 0))
        }

        let payload: [String: Any] = [
            "probe": "e184_gated_delta_scan_cost",
            "harness": "local",
            "cool_gate_passed_real_gate": false,
            "gate_qualified_for_timing": false,
            "linear_layers": Self.linearLayers,
            "seed_length": Self.seedLength,
            "cells": cells.map {
                [
                    "arm": $0.arm, "timesteps": $0.timesteps,
                    "microseconds": $0.microseconds,
                    "forward_microseconds": $0.forwardMicroseconds,
                    "reverse_microseconds": $0.reverseMicroseconds,
                    "replicates": $0.replicates,
                ]
            },
        ]
        let data = try JSONSerialization.data(
            withJSONObject: payload, options: [.sortedKeys, .prettyPrinted])
        let text = String(decoding: data, as: UTF8.self)
        if let path = ProcessInfo.processInfo.environment["MLXFAST_E184_PROBE_OUT"] {
            try text.write(toFile: path, atomically: true, encoding: .utf8)
        }
        print("E184-PROBE \(text)")
    }
}
