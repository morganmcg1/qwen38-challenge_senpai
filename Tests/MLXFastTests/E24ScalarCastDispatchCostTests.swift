import Foundation
import MLX
import Testing

// E24 Phase 1 — research-only microbenchmark. Never submitted.
//
// Measures the marginal cost of ONE size-1 `float32 -> bfloat16` `asType`, the
// exact operation the 48 GDN layers perform twice each on their `invScale`
// constants. Phase 0 proved (see research/e24-prereg.md) that this lowers to a
// real 1-thread `v_copy_float32_bfloat16` dispatch with no useful bandwidth,
// and that `MLXArray(Float)` itself is free, so a GDN forward pays exactly 96
// of these.
//
// The result is an UPPER BOUND on what hoisting can save, not a point estimate:
// a real forward can hide part of the host-side graph construction behind GPU
// work through the asyncEval ladder, and this benchmark cannot.
//
// Two arms, because a homogeneous loop of identical casts binds the Metal
// pipeline state once and therefore understates what the real layer pays, where
// each cast sits between unrelated kernels:
//
//   A  N independent casts, back to back            -> pure launch floor
//   B  N (cast, filler) pairs, fillers cycling       -> cast in a mixed stream
//   C  N fillers alone, same cycle                   -> the B subtrahend
//
// Removing one cast from the real forward removes one launch and collapses two
// pipeline switches back to one, which is exactly `slope(B) - slope(C)`.
@Suite
struct E24ScalarCastDispatchCostTests {
    private static var enabled: Bool {
        let env = ProcessInfo.processInfo.environment
        return env["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1"
            && env["E24_MICROBENCH"] == "1"
    }

    private static let counts = [0, 256, 512, 1024, 2048, 4096]
    private static let reps = 21
    private static let warmupReps = 5

    /// One size-1 bfloat16 unary, chosen by `i` so consecutive calls land on
    /// different Metal kernels and force a pipeline-state switch.
    private static func filler(_ i: Int, _ x: MLXArray) -> MLXArray {
        switch i % 4 {
        case 0: return MLX.abs(x)
        case 1: return MLX.negative(x)
        case 2: return MLX.sqrt(x)
        default: return MLX.floor(x)
        }
    }

    private static func armA(_ n: Int, _ seed: MLXArray) -> [MLXArray] {
        var out: [MLXArray] = []
        out.reserveCapacity(n)
        for i in 0 ..< n {
            out.append(MLXArray(Float(i) + 0.5).asType(.bfloat16))
        }
        return out
    }

    private static func armB(_ n: Int, _ seed: MLXArray) -> [MLXArray] {
        var out: [MLXArray] = []
        out.reserveCapacity(n * 2)
        for i in 0 ..< n {
            out.append(MLXArray(Float(i) + 0.5).asType(.bfloat16))
            out.append(filler(i, seed))
        }
        return out
    }

    private static func armC(_ n: Int, _ seed: MLXArray) -> [MLXArray] {
        var out: [MLXArray] = []
        out.reserveCapacity(n)
        for i in 0 ..< n {
            out.append(filler(i, seed))
        }
        return out
    }

    private struct Sample {
        var build: Double
        var evalOnly: Double
        var total: Double
        var evalIQR: Double
    }

    private static func measure(
        _ n: Int,
        _ seed: MLXArray,
        _ build: (Int, MLXArray) -> [MLXArray]
    ) -> Sample {
        var builds: [Double] = []
        var evals: [Double] = []
        for rep in 0 ..< (warmupReps + reps) {
            let t0 = DispatchTime.now().uptimeNanoseconds
            let arrays = build(n, seed)
            let t1 = DispatchTime.now().uptimeNanoseconds
            eval(arrays)
            let t2 = DispatchTime.now().uptimeNanoseconds
            guard rep >= warmupReps else { continue }
            builds.append(Double(t1 - t0) / 1e9)
            evals.append(Double(t2 - t1) / 1e9)
        }
        let b = median(builds)
        let e = median(evals)
        return Sample(build: b, evalOnly: e, total: b + e, evalIQR: iqr(evals))
    }

    private static func median(_ xs: [Double]) -> Double {
        let s = xs.sorted()
        guard !s.isEmpty else { return .nan }
        return s.count % 2 == 1
            ? s[s.count / 2]
            : (s[s.count / 2 - 1] + s[s.count / 2]) / 2
    }

    private static func iqr(_ xs: [Double]) -> Double {
        let s = xs.sorted()
        guard s.count >= 4 else { return .nan }
        return s[(s.count * 3) / 4] - s[s.count / 4]
    }

    /// Ordinary least squares of `y` on `x`, returning (slope, intercept, r2).
    private static func fit(_ x: [Double], _ y: [Double])
        -> (slope: Double, intercept: Double, r2: Double)
    {
        let n = Double(x.count)
        let mx = x.reduce(0, +) / n
        let my = y.reduce(0, +) / n
        var sxy = 0.0, sxx = 0.0, syy = 0.0
        for i in 0 ..< x.count {
            sxy += (x[i] - mx) * (y[i] - my)
            sxx += (x[i] - mx) * (x[i] - mx)
            syy += (y[i] - my) * (y[i] - my)
        }
        let slope = sxy / sxx
        return (slope, my - slope * mx, (sxy * sxy) / (sxx * syy))
    }

    @Test(.enabled(if: E24ScalarCastDispatchCostTests.enabled))
    func perDispatchSlopeOfTheSizeOneCast() throws {
        let seed = MLXArray(Float(1.25)).asType(.bfloat16)
        eval(seed)

        // Bind every kernel this benchmark uses before any timing, so JIT
        // compilation of `v_copy_float32_bfloat16` and the four unaries never
        // lands inside a measured window.
        eval(Self.armB(64, seed))

        var report: [String: Any] = [:]
        var arms: [String: Any] = [:]
        var slopes: [String: Double] = [:]

        for (name, dispatchesPerUnit, build) in [
            ("A_homogeneous", 1, Self.armA),
            ("B_cast_plus_filler", 2, Self.armB),
            ("C_filler_only", 1, Self.armC),
        ] as [(String, Int, (Int, MLXArray) -> [MLXArray])] {
            var rows: [[String: Any]] = []
            var xs: [Double] = []
            var ysTotal: [Double] = []
            var ysEval: [Double] = []
            for n in Self.counts {
                let s = Self.measure(n, seed, build)
                rows.append([
                    "n": n,
                    "dispatches": n * dispatchesPerUnit,
                    "build_seconds": s.build,
                    "eval_seconds": s.evalOnly,
                    "eval_iqr_seconds": s.evalIQR,
                    "total_seconds": s.total,
                ])
                xs.append(Double(n))
                ysTotal.append(s.total)
                ysEval.append(s.evalOnly)
            }
            let ft = Self.fit(xs, ysTotal)
            let fe = Self.fit(xs, ysEval)
            arms[name] = [
                "rows": rows,
                "slope_total_seconds_per_unit": ft.slope,
                "intercept_total_seconds": ft.intercept,
                "r2_total": ft.r2,
                "slope_eval_seconds_per_unit": fe.slope,
                "r2_eval": fe.r2,
            ]
            slopes["\(name)_total"] = ft.slope
            slopes["\(name)_eval"] = fe.slope
        }

        // Arm A is per-cast directly. Arm B minus Arm C is the marginal cast in
        // a stream that already pays a pipeline switch between neighbours.
        let armATotal = slopes["A_homogeneous_total"]!
        let armBMinusC = slopes["B_cast_plus_filler_total"]!
            - slopes["C_filler_only_total"]!
        let armAEval = slopes["A_homogeneous_eval"]!
        let armBMinusCEval = slopes["B_cast_plus_filler_eval"]!
            - slopes["C_filler_only_eval"]!

        // Phase 0 model: 96 casts per verify forward, flat in M; 246 rounds per
        // 512-token decode on the shipped CURVE default (E17 refit, `english`).
        let castsPerForward = 96.0
        let rounds = 246.0
        let localTrueDecodeSeconds = 19.698
        let removed = castsPerForward * rounds

        var projection: [String: Any] = [:]
        for (label, perCast) in [
            ("armA_floor", armATotal),
            ("armB_minus_C_realistic", armBMinusC),
            ("armA_floor_eval_only", armAEval),
            ("armB_minus_C_eval_only", armBMinusCEval),
        ] {
            let saved = removed * perCast
            projection[label] = [
                "seconds_per_cast": perCast,
                "microseconds_per_cast": perCast * 1e6,
                "projected_saved_seconds": saved,
                "projected_pct_of_local_true_decode":
                    100.0 * saved / localTrueDecodeSeconds,
            ]
        }

        report["schema"] = "e24-phase1-v1"
        report["generated_utc"] = ISO8601DateFormatter().string(from: Date())
        report["device"] = "\(Device.defaultDevice())"
        report["counts"] = Self.counts
        report["reps"] = Self.reps
        report["warmup_reps"] = Self.warmupReps
        report["arms"] = arms
        report["model"] = [
            "casts_per_verify_forward": castsPerForward,
            "rounds_per_512_token_decode": rounds,
            "dispatches_removed_per_run": removed,
            "local_mtp_true_decode_seconds": localTrueDecodeSeconds,
        ]
        report["projection"] = projection
        report["prereg_threshold_pct"] = 0.50

        let out = ProcessInfo.processInfo.environment["E24_OUT"]
            ?? "research/results/e24-phase1.json"
        let data = try JSONSerialization.data(
            withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: URL(fileURLWithPath: out))

        print("E24 phase1 written to \(out)")
        print(String(data: data, encoding: .utf8) ?? "")

        // The measurement is only usable if the sweep is actually linear in the
        // dispatch count; a low r2 means something other than launch count is
        // driving the time and the bound cannot be read off the slope.
        for (name, arm) in arms {
            let a = arm as! [String: Any]
            #expect(
                (a["r2_total"] as! Double) > 0.90,
                "arm \(name) is not linear in dispatch count (r2 = \(a["r2_total"]!)); "
                    + "the per-dispatch slope cannot be read from it"
            )
        }
    }
}
