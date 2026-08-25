import Foundation
import MLX
import MLXRandom
import Testing

@testable import MLXLLM

// E219 -- what is one wide-QMV pass physically made of?
//
// FINDING 559 fitted the width-cost law `R = G(m) x f(IPG)` and showed that a
// second weight-group pass costs 0.80-0.91 of the first while token columns are
// nearly free (`f(5)/f(2) = 1.32` for 2.5x the FMA). E216 (FINDING 565) removed
// one composition hypothesis: the per-pass fixed cost is not L1-dedupable
// redundant weight fetch. This file measures the composition directly, so the
// next mechanism wave can be priced instead of guessed:
//
//     cost(pass) = a_dispatch
//                + b x weight_bytes
//                + c x activation_bytes
//                + d x output_bytes
//
// The instrument JITs `qwen_e120_qmv_wide` from the SHIPPED header text, so it
// cannot drift from the scored kernel, but it supplies its own launch shim so
// `NA` (token columns per pass), `G` (weight-group passes), `n`, `k` and the
// dispatch count move one at a time. Under the shipped plan those knobs are
// locked together by the `(m, IPG)` table; here they are free.
//
// Two properties of `qwen_e120_qmv_wide` set the design:
//
//   * `rows_per_simd` is a compile-time 4 and the threadgroup is (32, 2, 1),
//     so one threadgroup always owns 8 output rows. The FINDING 549 register
//     grid therefore admits `NA <= 5` at `rows = 4` and nothing wider, and the
//     FINDING 552 `(NA=7, ROWS=2)` miscompile is unreachable from here.
//   * `NA` is at once the inputs-per-group of the width plan AND the activation
//     width of the pass. In this kernel the assignment's "IPG sweep" and its
//     "token-columns within a pass" sweep are the SAME knob; they are reported
//     once, not twice, and the `c` term is identified against the `k` and `n`
//     ladders instead.
//
// Timing is chain-slope, not per-call: `C` units are enqueued behind ONE
// `eval()` for `C` in {1, 2, 4, 8} and the per-unit cost is the regression
// slope. FINDING 544 measured the timed MTP leg at 98.34% GPU-busy, so the
// pipelined slope -- not the barrier-charged single call -- is what in-path
// decode pays.
//
// Every weight replica set is cycled so that each timed dispatch streams
// weights the previous dispatch did not touch. The hot arm keeps one replica
// and is reported alongside: the hot-minus-cold delta bounds the cacheable
// share of the stream, which is the quantity FINDING 560 showed a naive
// bytes-roofline gets wrong.
//
// Research instrument. `Tests/` is never packaged into a submission.
// harness=local-microbench. These are mechanism prices for one dispatch, never
// whole-leg or ranked numbers (RULE 79). No thermal gate, no score.

/// Live-path certification needle (RULE 384/387): a dedicated >= 16-byte
/// literal on the executed path of this probe, not on any failure path.
let e219LivePathNeedle = "E219-QMV-PASS-ANATOMY-CENSUS-LIVE-PATH-2026-08-25"

// MARK: - shapes

private struct E219Cell {
    var name: String
    var k: Int
    var n: Int
    /// Wide-QMV invocations of this cell in one decode round (FINDING 487).
    var invocations: Int
}

private let e219ScoredCells = [
    E219Cell(name: "mlp.gate_up", k: 5120, n: 34816, invocations: 64),
    E219Cell(name: "mlp.down", k: 17408, n: 5120, invocations: 64),
    E219Cell(name: "gdn.in_proj", k: 5120, n: 16480, invocations: 48),
]

/// Bytes one pass of this cell touches, split by the four fitted terms.
///
/// `weight` and `scaleBias` are unique device bytes: affine 4-bit group-64
/// carries 32 B packed plus 2 B of bf16 scale and 2 B of bf16 bias per 64
/// weights, so metadata is 1/8 of the packed stream and 11.1% of the whole.
/// `activationUnique` is the slab every threadgroup needs; `activationRead` is
/// what the `n / 8` row-slice threadgroups collectively request, which is what
/// the cache hierarchy -- not DRAM -- must serve.
private struct E219Bytes {
    var weight: Int
    var scaleBias: Int
    var activationUnique: Int
    var activationRead: Int
    var output: Int
    var xsumsUnique: Int
    var xsumsRead: Int
    var threadgroups: Int
    var simdgroups: Int
    var bytesPerRowPerKBlock: Double

    /// `rows` is `rows_per_simd`. One simdgroup owns `rows` output rows and
    /// reads the WHOLE activation slab, so the simdgroup count -- and with it
    /// every re-read term -- is inversely proportional to `rows` while the
    /// weight and metadata streams are invariant in it.
    init(k: Int, n: Int, na: Int, stride: Int, rows: Int = 4,
         activationBytesPerValue: Int = 2) {
        let simds = n / rows
        weight = n * (k / 2)
        scaleBias = n * (k / 64) * 4
        activationUnique = na * k * activationBytesPerValue
        activationRead = simds * activationUnique
        output = na * n * 2
        xsumsUnique = (k / 512) * 32 * stride * 4
        xsumsRead = simds * xsumsUnique
        threadgroups = n / (2 * rows)
        simdgroups = simds
        // Per output row per k-block, from the kernel body: `12 * rows` bytes
        // of weights and metadata against `36 * na` bytes of activation and
        // chunk sums, the second group divided by `rows`.
        bytesPerRowPerKBlock = 12.0
            + (16.0 * Double(activationBytesPerValue) + 4.0)
            * Double(na) / Double(rows)
    }

    var dictionary: [String: Int] {
        [
            "weight_bytes": weight, "scale_bias_bytes": scaleBias,
            "stream_bytes": weight + scaleBias,
            "activation_unique_bytes": activationUnique,
            "activation_read_bytes": activationRead,
            "output_bytes": output,
            "xsums_unique_bytes": xsumsUnique, "xsums_read_bytes": xsumsRead,
            "threadgroups_per_pass": threadgroups,
            "simdgroups_per_pass": simdgroups,
        ]
    }
}

// MARK: - E221: the rows_per_simd arm

/// `rows_per_simd` promoted to a template argument, from the SHIPPED text.
///
/// E221 asks whether the super-linear NA cost above NA=3 is register pressure.
/// The mechanism under test halves outputs per simdgroup, which halves live
/// accumulators at unchanged device weight bytes -- and doubles every activation
/// and chunk-sum re-read, because the simdgroup count doubles. The two premises
/// are separated only by measurement, so this arm compiles the real body with
/// one substituted line and times it beside the incumbent.
///
/// The substitution count is asserted, so a base move that renames or duplicates
/// any of these lines fails here instead of silently timing the wrong kernel.
/// `research/e221_register_probe.py` shows the parameterized body is register
/// identical to the shipped body at ROWS=4 on the AIR proxy and on both real
/// AGX backends, and the `rows4param` arm below is the timed form of the same
/// control.
func e221RowsParameterizedHeader() -> String {
    let substitutions = [
        ("template <int NA, bool USE_TABLE>",
         "template <int NA, int ROWS, bool USE_TABLE>"),
        ("constexpr int rows_per_simd = 4;",
         "constexpr int rows_per_simd = ROWS;"),
        ("template <int M, int IPG, bool USE_TABLE>",
         "template <int M, int IPG, int ROWS, bool USE_TABLE>"),
        ("qwen_e120_qmv_wide<IPG, USE_TABLE>(",
         "qwen_e120_qmv_wide<IPG, ROWS, USE_TABLE>("),
        ("qwen_e120_qmv_wide<(TAIL >= 2 ? TAIL : 2), USE_TABLE>(",
         "qwen_e120_qmv_wide<(TAIL >= 2 ? TAIL : 2), ROWS, USE_TABLE>("),
    ]
    var text = qwen35E120QMVHeader
    for (old, new) in substitutions {
        let occurrences = text.components(separatedBy: old).count - 1
        precondition(
            occurrences == 1,
            "[e221] expected one occurrence of \(old), found \(occurrences)")
        text = text.replacingOccurrences(of: old, with: new)
    }
    return text
}

// MARK: - E223: the activation dtype arm

/// The shipped gather, verbatim. The E223 transform replaces this exact text,
/// so a base move breaks this arm loudly instead of timing the wrong kernel.
private let e223ShippedGather = """
            for (int i = 0; i < 4; i++) {
                VF a0, a1, a2, a3;
                for (int m = 0; m < NA; m++) {
                    const device bfloat16_t* xm =
                        x + (first_m + m) * in_vec_size + k +
                        simd_lid * values_per_thread + 4 * i;
                    const vec<bfloat16_t, 4> xv =
                        *reinterpret_cast<const device vec<bfloat16_t, 4>*>(
                            xm);
                    a0[m] = static_cast<float>(xv[0]);
                    a1[m] = static_cast<float>(xv[1]);
                    a2[m] = static_cast<float>(xv[2]);
                    a3[m] = static_cast<float>(xv[3]);
                    if (!USE_TABLE) {
                        sums[m] += xv[0] + xv[1] + xv[2] + xv[3];
                    }
                }
"""

/// Activations already float32, in the SHIPPED `[m, k]` layout.
///
/// E223's census found that the addressable part of the per-column cost is not
/// the lane inserts, the address arithmetic or the load count -- the backend
/// removes or normalises all three -- but the 16 `bfloat16` to `float`
/// conversions each column pays per k-block. This arm is the only source form
/// that removes them: 12.6 of the 123.2 machine instructions per column per
/// k-block on `applegpu_g17s`, at the cost of issuing twice the activation
/// bytes.
///
/// `bfloat16` to `float` is exact and no other line moves, so this arm must
/// return values bit-identical to the shipped arm. `activationDTypeExactness`
/// proves that with a positive control before any arm is timed.
///
/// The chunk-sum table is unchanged, so this arm is only built at
/// `USE_TABLE = true`; the `!USE_TABLE` branch of the shipped gather is what
/// the replacement drops.
func e223ActivationF32Header() -> String {
    let gather = """
            for (int i = 0; i < 4; i++) {
                VF a0, a1, a2, a3;
                for (int m = 0; m < NA; m++) {
                    const device float* xm =
                        x + (first_m + m) * in_vec_size + k +
                        simd_lid * values_per_thread + 4 * i;
                    const vec<float, 4> xv =
                        *reinterpret_cast<const device vec<float, 4>*>(xm);
                    a0[m] = xv[0];
                    a1[m] = xv[1];
                    a2[m] = xv[2];
                    a3[m] = xv[3];
                }
"""
    // Both `qwen_e120_qmv_wide` and the `qwen_e120_qmv_m` wrapper declare the
    // activation pointer, and the wrapper forwards it, so both must move.
    let substitutions = [
        ("        const device bfloat16_t* x,\n",
         "        const device float* x,\n", 2),
        (e223ShippedGather, gather, 1),
    ]
    var text = qwen35E120QMVHeader
    for (old, new, expected) in substitutions {
        let occurrences = text.components(separatedBy: old).count - 1
        precondition(
            occurrences == expected,
            "[e223] expected \(expected) occurrences, found \(occurrences)")
        text = text.replacingOccurrences(of: old, with: new)
    }
    return text
}

/// Which output rows each simdgroup writes, by pure launch arithmetic.
///
/// RULE 403: a value comparison cannot prove write coverage, because a wrong
/// mapping leaves cells UNWRITTEN and MLX buffer recycling can return
/// stale-correct values. This enumerates the launched simdgroups and counts
/// writes per output row, so coverage is proved rather than inferred.
///
/// Check the returned KEY SET against `0 ..< n`, not its size: a stale
/// simdgroup stride can write exactly `n` distinct rows and still miss half the
/// output while addressing indices beyond `n - 1`.
///
/// `sgStride` is the per-simdgroup row stride the launch shim encodes. It
/// equals `rows` in every timed arm; passing a different value builds the
/// positive control that proves this census can fail.
func e221WriteCensus(n: Int, rows: Int, sgStride: Int? = nil) -> [Int: Int] {
    let stride = sgStride ?? rows
    var writes: [Int: Int] = [:]
    let threadgroupsY = n / (2 * rows)
    for tidY in 0 ..< threadgroupsY {
        for sgid in 0 ..< 2 {
            let outRow = tidY * (2 * stride) + sgid * stride
            for r in 0 ..< rows {
                writes[outRow + r, default: 0] += 1
            }
        }
    }
    return writes
}

private func e219SumsStride(_ m: Int) -> Int { Qwen35CustomQMV.sumsStride(m) }

// MARK: - the instrument kernel

/// One JIT pair over the SHIPPED header, with the launch shim this probe owns.
///
/// The shim mirrors `qwen35E120QMVSource` exactly except that `NA` and the
/// chunk-sum stride are compile-time literals instead of a switch on
/// `x_shape[x_ndim - 2]`, and `first_m` comes from the launched x extent rather
/// than from a width-plan table. `numericalSanity` proves the shim is
/// bit-exact against `Qwen35CustomQMV.matmulWithTable` wherever the shipped
/// plan reaches the same instantiation.
private struct E219Pipeline {
    let na: Int
    let stride: Int
    let table: Bool
    /// `rows_per_simd`. 4 is the shipped value.
    let rows: Int
    /// True when the body came from `e221RowsParameterizedHeader()`. At
    /// `rows == 4` this is the E221 control arm: same geometry, same
    /// arithmetic, one substituted line.
    let parameterized: Bool
    /// True when the body came from `e223ActivationF32Header()`, so the `x`
    /// argument is float32 in the shipped `[m, k]` layout.
    let activationF32: Bool
    let kernel: Qwen35CachedKernel

    init(na: Int, stride: Int, table: Bool, rows: Int = 4,
         parameterized: Bool = false, activationF32: Bool = false) {
        precondition(rows >= 1 && rows <= 4, "[e221] rows out of range")
        precondition(
            parameterized || rows == 4,
            "[e221] rows != 4 needs the parameterized header")
        precondition(
            !(activationF32 && parameterized),
            "[e223] the float32 activation arm and the rows arm are separate")
        precondition(
            !activationF32 || table,
            "[e223] the float32 activation arm drops the in-kernel sum path")
        // FINDING 552: `<NA=7, USE_TABLE=false, ROWS=2>` is miscompiled. It is
        // never timed, so the joint pair cannot reach a measurement.
        precondition(
            !(na == 7 && rows == 2 && !table),
            "[e221] FINDING 552 forbids (NA=7, rows=2, plain)")
        self.na = na
        self.stride = stride
        self.table = table
        self.rows = rows
        self.parameterized = parameterized
        self.activationF32 = activationF32
        let sums = table ? "xsums" : "qmv_null_sums"
        let flag = table ? "USE_TABLE" : "false"
        let nullDecl =
            table
            ? "" : "\n        const device float* qmv_null_sums = nullptr;"
        let arguments = parameterized
            ? "<\(na), \(rows), \(flag)>" : "<\(na), \(flag)>"
        let source = """
                const int qmv_k = x_shape[x_ndim - 1];
                const int qmv_n = w_shape[0];
                const uint3 qmv_tid = threadgroup_position_in_grid;
                const uint qmv_lid = thread_index_in_simdgroup;
                const uint qmv_sgid = simdgroup_index_in_threadgroup;
                const int qmv_out_row = int(qmv_tid.y) * \(2 * rows)
                    + int(qmv_sgid) * \(rows);
                const int qmv_first_m = int(qmv_tid.x) * \(na);\(nullDecl)
                qwen_e120_qmv_wide\(arguments)(
                    w, scales, biases, x, \(sums), y,
                    qmv_k, qmv_n, \(stride),
                    qmv_first_m, qmv_out_row, qmv_lid);
            """
        let tag = activationF32 ? "xf32" : (parameterized ? "r\(rows)p" : "ship")
        let header: String
        if activationF32 {
            header = e223ActivationF32Header()
        } else if parameterized {
            header = e221RowsParameterizedHeader()
        } else {
            header = qwen35E120QMVHeader
        }
        self.kernel = Qwen35CachedKernel(
            name: "e219_qmv_na\(na)_s\(stride)_\(table ? "tab" : "raw")_\(tag)",
            inputNames: table
                ? ["w", "scales", "biases", "x", "xsums"]
                : ["w", "scales", "biases", "x"],
            outputNames: ["y"],
            source: source,
            header: header)
    }

    /// One dispatch: `groups` weight-group passes over `n` output rows.
    ///
    /// One simdgroup owns `rows` rows and a threadgroup holds two simdgroups,
    /// so `n` rows need `n / rows` simdgroups. The y extent is in threads.
    func call(
        x: MLXArray, xsums: MLXArray, set: E219WeightSet, m: Int, groups: Int
    ) -> MLXArray {
        let launch = Qwen35KernelLaunch(
            grid: (groups * 32, set.n / rows, 1),
            threadGroup: (32, 2, 1),
            outputShape: [1, Int32(m), Int32(set.n)],
            outputDType: .bfloat16,
            useTable: table ? true : nil)
        if table {
            return kernel([set.w, set.scales, set.biases, x, xsums], launch)
        }
        return kernel([set.w, set.scales, set.biases, x], launch)
    }
}

private struct E219PipelineCache {
    private var pipelines: [String: E219Pipeline] = [:]

    mutating func get(
        na: Int, stride: Int, table: Bool, rows: Int = 4,
        parameterized: Bool = false, activationF32: Bool = false
    ) -> E219Pipeline {
        let key =
            "\(na)/\(stride)/\(table)/\(rows)/\(parameterized)/\(activationF32)"
        if let hit = pipelines[key] { return hit }
        let made = E219Pipeline(
            na: na, stride: stride, table: table, rows: rows,
            parameterized: parameterized, activationF32: activationF32)
        pipelines[key] = made
        return made
    }
}

// MARK: - weight replicas

private struct E219WeightSet {
    var w: MLXArray
    var scales: MLXArray
    var biases: MLXArray
    var k: Int
    var n: Int

    /// affine 4-bit group-64: 32 B packed plus 4 B of bf16 scale and bias per
    /// 64 weights, so the device stream is `n * k * 9 / 16`.
    static func bytes(k: Int, n: Int) -> Int { n * k * 9 / 16 }
    var bytes: Int { Self.bytes(k: k, n: n) }
}

/// One randomly packed affine-4/group-64 set at a scored cell shape. Nibble
/// values and scale magnitudes carry the range a decode round really sees, so
/// the numerical sanity check compares meaningful floats.
private func e219RandomSet(k: Int, n: Int, seed: UInt64) -> E219WeightSet {
    MLXRandom.seed(seed)
    let w = MLXRandom.uniform(low: 0.0, high: 4.2e9, [n, k / 8]).asType(.uint32)
    let scales = MLXRandom.uniform(low: 0.005, high: 0.03, [n, k / 64])
        .asType(.bfloat16)
    let biases = MLXRandom.uniform(low: -0.05, high: 0.05, [n, k / 64])
        .asType(.bfloat16)
    eval(w, scales, biases)
    return E219WeightSet(w: w, scales: scales, biases: biases, k: k, n: n)
}

/// Replica `i > 0` is a row rotation of replica 0. Dispatch cost is a function
/// of addresses and extents, not of nibble values, so a rotation buys a
/// genuinely distinct 50-100 MB device buffer for one copy instead of a second
/// multi-hundred-megabyte random draw.
private func e219RotatedSet(_ base: E219WeightSet, shift: Int)
    -> E219WeightSet
{
    let n = base.n
    let s = ((shift % n) + n) % n
    if s == 0 { return base }
    func roll(_ a: MLXArray) -> MLXArray {
        concatenated([a[s ..< n], a[0 ..< s]], axis: 0)
    }
    let set = E219WeightSet(
        w: roll(base.w), scales: roll(base.scales), biases: roll(base.biases),
        k: base.k, n: n)
    eval(set.w, set.scales, set.biases)
    return set
}

/// A cache-defeating replica set sized just past `targetBytes`, so every timed
/// dispatch streams weights the previous dispatch did not touch. The ring is
/// deliberately small: one replica already exceeds any Apple GPU cache, and a
/// large resident footprint reintroduces first-touch and residency noise.
private struct E219Replicas {
    var sets: [E219WeightSet]
    var requestedCount: Int
    var totalBytes: Int
    private var cursor = 0

    init(k: Int, n: Int, seed: UInt64, targetBytes: Int, hardCapBytes: Int) {
        let perSet = E219WeightSet.bytes(k: k, n: n)
        var wanted = (targetBytes + perSet - 1) / perSet
        wanted = max(2, min(wanted, max(2, hardCapBytes / perSet)))
        requestedCount = wanted
        let base = e219RandomSet(k: k, n: n, seed: seed)
        var built = [base]
        let step = max(8, (n / max(1, wanted)) / 8 * 8)
        for i in 1 ..< wanted {
            built.append(e219RotatedSet(base, shift: step * i))
        }
        sets = built
        totalBytes = perSet * built.count
        MLX.Memory.clearCache()
    }

    /// The cold arm walks the replica ring; the hot arm is the same ring
    /// pinned to one member.
    mutating func next(cold: Bool) -> E219WeightSet {
        guard cold else { return sets[0] }
        let set = sets[cursor]
        cursor = (cursor + 1) % sets.count
        return set
    }
}

// MARK: - timing

private struct E219Unit {
    var label: String
    var fields: [String: Any]
    /// Enqueues the unit's dispatches and returns their outputs unevaluated.
    var enqueue: () -> [MLXArray]
}

private func e219Int(_ key: String, _ fallback: Int) -> Int {
    guard let raw = ProcessInfo.processInfo.environment[key], let v = Int(raw)
    else { return fallback }
    return v
}

private func e219IntList(_ key: String, _ fallback: [Int]) -> [Int] {
    guard let raw = ProcessInfo.processInfo.environment[key], !raw.isEmpty
    else { return fallback }
    return raw.split(separator: ",").compactMap { Int($0) }
}

private func e219GpuTemperature() -> Double? {
    for path in [
        ProcessInfo.processInfo.environment["MLXFAST_MACMON_BIN"] ?? "",
        "\(ProcessInfo.processInfo.environment["HOME"] ?? "")/bin/macmon",
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
            let line = String(decoding: data, as: UTF8.self)
                .split(separator: "\n").first,
            let object = try? JSONSerialization.jsonObject(
                with: Data(line.utf8)) as? [String: Any],
            let temp = object["temp"] as? [String: Any],
            let gpu = temp["gpu_temp_avg"] as? Double
        else { continue }
        return gpu
    }
    return nil
}

/// One chain-slope block over `units`, palindromic in the block index so
/// monotone session drift cancels to first order between orientations.
private struct E219Session {
    let blocks: Int
    let chains: [Int]
    let warmup: Int
    let touch: Int
    let minimumReps: Int
    let targetMicroseconds: Double
    var temperatures: [String: Double?] = [:]
    var samples: [[String: Any]] = []

    init() {
        blocks = e219Int("MLX_E219_BLOCKS", 6)
        chains = e219IntList("MLX_E219_CHAINS", [1, 2, 3, 4])
        warmup = e219Int("MLX_E219_WARMUP", 4)
        touch = e219Int("MLX_E219_TOUCH", 24)
        minimumReps = e219Int("MLX_E219_MIN_REPS", 8)
        targetMicroseconds = Double(e219Int("MLX_E219_TARGET_US", 6000))
    }

    mutating func recordTemperature(_ label: String) {
        temperatures[label] = e219GpuTemperature()
    }

    private func time(_ unit: E219Unit, chain: Int, reps: Int) -> Double {
        let start = DispatchTime.now().uptimeNanoseconds
        for _ in 0 ..< reps {
            var outputs: [MLXArray] = []
            outputs.reserveCapacity(chain)
            for _ in 0 ..< chain { outputs.append(contentsOf: unit.enqueue()) }
            eval(outputs)
        }
        return Double(DispatchTime.now().uptimeNanoseconds - start) / 1e3
            / Double(reps)
    }

    /// Repetition count that puts one measurement near `targetMicroseconds`,
    /// so a 10 us unit and a 2 ms unit carry comparable clock resolution.
    private func calibrate(_ unit: E219Unit, chain: Int) -> Int {
        let pilot = time(unit, chain: chain, reps: 2)
        guard pilot > 0 else { return 64 }
        return max(
            minimumReps, min(400, Int((targetMicroseconds / pilot).rounded())))
    }

    /// Cycles the unit's whole replica ring several times before it is timed.
    /// Without this, a timed block pays first-touch residency cost for replicas
    /// the short warmup never reached, which dominates the wide cells.
    private func pretouch(_ unit: E219Unit) {
        for _ in 0 ..< touch { eval(unit.enqueue()) }
    }

    mutating func run(_ units: [E219Unit], settle: () -> Void) {
        recordTemperature("session_entry")
        for _ in 0 ..< 40 { settle() }
        for unit in units {
            for chain in chains {
                for _ in 0 ..< warmup {
                    var outputs: [MLXArray] = []
                    for _ in 0 ..< chain {
                        outputs.append(contentsOf: unit.enqueue())
                    }
                    eval(outputs)
                }
            }
        }
        recordTemperature("after_warmup")

        var reps: [String: Int] = [:]
        for unit in units {
            pretouch(unit)
            for chain in chains {
                reps["\(unit.label)/\(chain)"] = calibrate(unit, chain: chain)
            }
        }

        for block in 0 ..< blocks {
            let ascending = block % 2 == 0
            let order = ascending
                ? Array(units.indices) : Array(units.indices.reversed())
            recordTemperature("block_\(block)_entry")
            for _ in 0 ..< 4 { settle() }
            // Block prologue. The first two or three units measured after a
            // settle burst were reproducibly inflated, so every unit is cycled
            // once before any of them is timed and no position pays that cost.
            for index in order { pretouch(units[index]) }
            for (position, index) in order.enumerated() {
                let unit = units[index]
                pretouch(unit)
                for chain in ascending ? chains : chains.reversed() {
                    let n = reps["\(unit.label)/\(chain)"] ?? 16
                    let us = time(unit, chain: chain, reps: n)
                    var row = unit.fields
                    row["label"] = unit.label
                    row["block"] = block
                    row["ascending"] = ascending
                    row["position"] = position
                    row["chain"] = chain
                    row["reps"] = n
                    row["microseconds"] = us
                    row["microseconds_per_unit"] = us / Double(chain)
                    samples.append(row)
                }
            }
            recordTemperature("block_\(block)_exit")
        }
        recordTemperature("session_exit")
    }
}

private func e219Write(
    _ phase: String, _ body: [String: Any], session: E219Session?
) throws {
    var report: [String: Any] = [
        "probe": "e219-qmv-pass-anatomy",
        "phase": phase,
        "probe_needle": e219LivePathNeedle,
        "harness": "local-microbench",
        "cool_gate_passed_real_gate": false,
        "gate_qualified_for_timing": false,
        "official_or_ranked_score": false,
        "whole_leg_or_ranked_number": false,
        "active_plan_witness": qwen35QMVWidthPlanWitness,
        "qmv_arm": Qwen35CustomQMV.arm.rawValue,
        "config_cache_enabled": Qwen35KernelConfigCache.enabled,
        "shipped_rows_per_simd": 4,
        "threadgroup": [32, 2, 1],
    ]
    for (key, value) in body { report[key] = value }
    if let session {
        report["blocks"] = session.blocks
        report["chains"] = session.chains
        report["warmup"] = session.warmup
        report["pretouch_dispatches"] = session.touch
        report["minimum_reps"] = session.minimumReps
        report["target_microseconds"] = session.targetMicroseconds
        report["gpu_temperature_c"] = session.temperatures.mapValues {
            $0 ?? -1.0
        }
        report["samples"] = session.samples
    }
    let json = try JSONSerialization.data(
        withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
    if let path = ProcessInfo.processInfo.environment["MLX_E219_OUT"],
        !path.isEmpty
    {
        // One phase filter can hold several reports. Name the file after the
        // report's own phase so a second report cannot overwrite the first.
        let url = URL(fileURLWithPath: path)
            .deletingLastPathComponent()
            .appendingPathComponent("\(phase).json")
        try json.write(to: url)
        print("[e219] wrote \(url.path) (\(json.count) bytes)")
    } else {
        print(String(decoding: json, as: UTF8.self))
    }
}

private func e219PhaseEnabled(_ name: String) -> Bool {
    guard let raw = ProcessInfo.processInfo.environment["MLX_E219_PHASE"] else {
        return false
    }
    return raw == "all"
        || raw.split(separator: ",").map(String.init).contains(name)
}

private func e219Settle() -> () -> Void {
    let weights = MLXRandom.normal([2048, 2048]).asType(.bfloat16)
    let input = MLXRandom.normal([64, 2048]).asType(.bfloat16)
    eval(weights, input)
    return { eval(matmul(input, weights)) }
}

private func e219Activations(m: Int, k: Int, seed: UInt64) -> MLXArray {
    MLXRandom.seed(seed)
    let x = MLXRandom.normal([1, m, k]).asType(.bfloat16)
    eval(x)
    return x
}

// MARK: - Section 0: the shim drives the shipped kernel

@Suite(.serialized)
struct E219InstrumentSanityTests {
    /// The probe's launch shim must be bit-exact against the shipped routed
    /// dispatch wherever the width plan reaches the same instantiation. Without
    /// this, every coefficient below prices a kernel the model never runs.
    @Test(.enabled(if: e219PhaseEnabled("sanity")))
    func numericalSanity() throws {
        var cache = E219PipelineCache()
        var rows: [[String: Any]] = []

        for cell in e219ScoredCells {
            autoreleasepool {
                let set = e219RandomSet(k: cell.k, n: cell.n, seed: 0xE219)
                // Widths whose shipped staged plan is a whole number of equal
                // groups, so `first_m = tid.x * IPG` matches the shim exactly
                // and no tail instantiation is involved.
                for m in [4, 5, 8] {
                    autoreleasepool {
                        let variant = Qwen35CustomQMV.kernelVariant(
                            (m: m, k: cell.k, n: cell.n))
                        let ipg = Qwen35CustomQMV.inputsPerGroup(
                            m, variant: variant)
                        guard m % ipg == 0 else { return }
                        let groups = m / ipg
                        let x = e219Activations(
                            m: m, k: cell.k, seed: UInt64(0xE219_0000 + m))
                        let xsums = Qwen35CustomQMV.xsumsTable(x)
                        eval(xsums)
                        guard
                            let shipped = Qwen35CustomQMV.matmulWithTable(
                                x, set.w, scales: set.scales,
                                biases: set.biases, xsums: xsums,
                                groupSize: 64, bits: 4, mode: .affine)
                        else {
                            Issue.record(
                                "shipped path declined \(cell.name) m=\(m)")
                            return
                        }
                        let probe = cache.get(
                            na: ipg, stride: e219SumsStride(m), table: true)
                        let mine = probe.call(
                            x: x, xsums: xsums, set: set, m: m, groups: groups)
                        eval(shipped, mine)
                        let a = shipped.asType(.float32).asArray(Float.self)
                        let b = mine.asType(.float32).asArray(Float.self)
                        var differing = 0
                        var nonFinite = 0
                        for i in 0 ..< min(a.count, b.count) {
                            if a[i].bitPattern != b[i].bitPattern {
                                differing += 1
                            }
                            if !a[i].isFinite || !b[i].isFinite {
                                nonFinite += 1
                            }
                        }
                        rows.append([
                            "cell": cell.name, "k": cell.k, "n": cell.n,
                            "m": m, "ipg": ipg, "groups": groups,
                            "variant": variant.rawValue,
                            "elements": a.count, "differing": differing,
                            "non_finite": nonFinite,
                        ])
                        #expect(
                            differing == 0,
                            """
                            \(cell.name) m=\(m): shim differs from the shipped \
                            dispatch at \(differing) of \(a.count) outputs
                            """)
                        #expect(nonFinite == 0)
                    }
                }
            }
        }

        #expect(!rows.isEmpty)
        try e219Write(
            "sanity",
            [
                "deliverable":
                    "prove the E219 launch shim is bit-exact against "
                    + "Qwen35CustomQMV.matmulWithTable",
                "comparisons": rows,
            ], session: nil)
    }

    /// E221 arms: before a `rows_per_simd` arm may be timed, it must produce
    /// the shipped values AND cover every output row.
    ///
    /// Two independent gates, because neither alone is sufficient:
    ///  - values, against the shipped-header `rows == 4` pipeline, with a
    ///    perturbed positive control that proves the comparison can fail;
    ///  - write coverage by launch arithmetic (RULE 403), with a mismatched
    ///    simdgroup stride as the positive control, because MLX recycles output
    ///    buffers and an unwritten cell can read back stale-correct.
    @Test(.enabled(if: e219PhaseEnabled("sanity")))
    func rowsGeometryExactness() throws {
        var cache = E219PipelineCache()
        var valueRows: [[String: Any]] = []
        var censusRows: [[String: Any]] = []

        let geometries: [(rows: Int, parameterized: Bool)] =
            [(4, true), (2, true)]

        // A COUNT of written rows is not enough. At rows=2 with a stale
        // simdgroup stride of 4 the count is still exactly `n`, but the written
        // indices are the sparse set {0,1,4,5,...} running up to 2n-3. Coverage
        // must therefore be checked as a SET against 0 ..< n.
        func covers(_ census: [Int: Int], _ n: Int) -> Bool {
            census.count == n && census.keys.min() == 0
                && census.keys.max() == n - 1
                && census.values.allSatisfy { $0 == 1 }
        }

        for cell in e219ScoredCells {
            for rows in [4, 2] {
                let census = e221WriteCensus(n: cell.n, rows: rows)
                let broken = e221WriteCensus(
                    n: cell.n, rows: rows, sgStride: 4)
                censusRows.append([
                    "cell": cell.name, "n": cell.n, "rows": rows,
                    "rows_written": census.count,
                    "min_row": census.keys.min() ?? -1,
                    "max_row": census.keys.max() ?? -1,
                    "min_writes": census.values.min() ?? 0,
                    "max_writes": census.values.max() ?? 0,
                    "control_sg_stride": 4,
                    "control_rows_written": broken.count,
                    "control_max_row": broken.keys.max() ?? -1,
                    "control_covers": covers(broken, cell.n),
                ])
                #expect(
                    covers(census, cell.n),
                    """
                    \(cell.name) rows=\(rows): the launch writes \
                    \(census.count) rows over \
                    [\(census.keys.min() ?? -1), \(census.keys.max() ?? -1)], \
                    not exactly once over [0, \(cell.n - 1)]
                    """)
                if rows != 4 {
                    // The control must actually break, otherwise the coverage
                    // gate above is vacuous.
                    #expect(!covers(broken, cell.n))
                }
            }
        }

        for cell in e219ScoredCells {
            autoreleasepool {
                let set = e219RandomSet(k: cell.k, n: cell.n, seed: 0xE221)
                for m in [4, 5] {
                    autoreleasepool {
                        let stride = e219SumsStride(m)
                        let x = e219Activations(
                            m: m, k: cell.k, seed: UInt64(0xE221_0000 + m))
                        let xsums = Qwen35CustomQMV.xsumsTable(x)
                        eval(xsums)
                        let reference = cache.get(
                            na: m, stride: stride, table: true, rows: 4,
                            parameterized: false)
                        let want = reference.call(
                            x: x, xsums: xsums, set: set, m: m, groups: 1)
                        eval(want)
                        let a = want.asType(.float32).asArray(Float.self)
                        for geometry in geometries {
                            let probe = cache.get(
                                na: m, stride: stride, table: true,
                                rows: geometry.rows,
                                parameterized: geometry.parameterized)
                            let got = probe.call(
                                x: x, xsums: xsums, set: set, m: m, groups: 1)
                            eval(got)
                            let b = got.asType(.float32).asArray(Float.self)
                            var differing = 0
                            var nonFinite = 0
                            for i in 0 ..< min(a.count, b.count) {
                                if a[i].bitPattern != b[i].bitPattern {
                                    differing += 1
                                }
                                if !b[i].isFinite { nonFinite += 1 }
                            }
                            valueRows.append([
                                "cell": cell.name, "k": cell.k, "n": cell.n,
                                "m": m, "rows": geometry.rows,
                                "parameterized": geometry.parameterized,
                                "elements": b.count, "differing": differing,
                                "non_finite": nonFinite,
                            ])
                            #expect(
                                a.count == b.count,
                                """
                                \(cell.name) m=\(m) rows=\(geometry.rows): \
                                output element count changed
                                """)
                            #expect(
                                differing == 0,
                                """
                                \(cell.name) m=\(m) rows=\(geometry.rows): \
                                \(differing) of \(b.count) outputs differ from \
                                the shipped rows=4 pipeline
                                """)
                            #expect(nonFinite == 0)
                        }

                        // Positive control: one perturbed activation column
                        // must move the comparison off zero.
                        let xBad = e219Activations(
                            m: m, k: cell.k,
                            seed: UInt64(0xE221_9000 + m))
                        let xsumsBad = Qwen35CustomQMV.xsumsTable(xBad)
                        eval(xsumsBad)
                        let control = cache.get(
                            na: m, stride: stride, table: true, rows: 2,
                            parameterized: true)
                        let bad = control.call(
                            x: xBad, xsums: xsumsBad, set: set, m: m, groups: 1)
                        eval(bad)
                        let c = bad.asType(.float32).asArray(Float.self)
                        var controlDiffering = 0
                        for i in 0 ..< min(a.count, c.count) {
                            if a[i].bitPattern != c[i].bitPattern {
                                controlDiffering += 1
                            }
                        }
                        valueRows.append([
                            "cell": cell.name, "k": cell.k, "n": cell.n,
                            "m": m, "rows": 2, "parameterized": true,
                            "positive_control": true,
                            "elements": c.count,
                            "differing": controlDiffering,
                        ])
                        #expect(
                            controlDiffering > 0,
                            """
                            \(cell.name) m=\(m): the positive control did NOT \
                            differ, so the bit-exactness gate above is vacuous
                            """)
                    }
                }
            }
        }

        #expect(!valueRows.isEmpty)
        try e219Write(
            "sanity_rows",
            [
                "deliverable":
                    "E221 gate: every rows_per_simd arm is bit-exact against "
                    + "the shipped rows=4 pipeline and writes every output row",
                "value_comparisons": valueRows,
                "write_census": censusRows,
            ], session: nil)
    }

    /// E223 gate: the float32 activation arm must return the shipped bits.
    ///
    /// `bfloat16` to `float` widening is exact and the arm moves no other line,
    /// so `differing` must be zero at every scored width. A perturbed float32
    /// slab is the positive control: it changes the activation values inside the
    /// first k-block of the first column only, leaves the chunk-sum table
    /// alone, and must move the comparison off zero. Without it a gather that
    /// silently read nothing would pass.
    @Test(.enabled(if: e219PhaseEnabled("sanity")))
    func activationDTypeExactness() throws {
        var cache = E219PipelineCache()
        var valueRows: [[String: Any]] = []

        for cell in e219ScoredCells {
            autoreleasepool {
                let set = e219RandomSet(k: cell.k, n: cell.n, seed: 0xE223)
                for m in [2, 3, 4, 5] {
                    autoreleasepool {
                        let stride = e219SumsStride(m)
                        let x = e219Activations(
                            m: m, k: cell.k, seed: UInt64(0xE223_0000 + m))
                        let xsums = Qwen35CustomQMV.xsumsTable(x)
                        let xf32 = x.asType(.float32)
                        eval(xsums, xf32)

                        let reference = cache.get(
                            na: m, stride: stride, table: true)
                        let want = reference.call(
                            x: x, xsums: xsums, set: set, m: m, groups: 1)
                        eval(want)
                        let a = want.asType(.float32).asArray(Float.self)

                        let probe = cache.get(
                            na: m, stride: stride, table: true,
                            activationF32: true)
                        let got = probe.call(
                            x: xf32, xsums: xsums, set: set, m: m, groups: 1)
                        eval(got)
                        let b = got.asType(.float32).asArray(Float.self)

                        var differing = 0
                        var nonFinite = 0
                        for i in 0 ..< min(a.count, b.count) {
                            if a[i].bitPattern != b[i].bitPattern {
                                differing += 1
                            }
                            if !b[i].isFinite { nonFinite += 1 }
                        }
                        valueRows.append([
                            "cell": cell.name, "k": cell.k, "n": cell.n,
                            "m": m, "activation_dtype": "float32",
                            "elements": b.count, "differing": differing,
                            "non_finite": nonFinite,
                        ])
                        #expect(
                            a.count == b.count,
                            """
                            \(cell.name) m=\(m): the float32 activation arm \
                            changed the output element count
                            """)
                        #expect(
                            differing == 0,
                            """
                            \(cell.name) m=\(m): \(differing) of \(b.count) \
                            outputs differ from the shipped bfloat16 arm
                            """)
                        #expect(nonFinite == 0)

                        // Positive control: perturb the float32 slab only, and
                        // only inside the first k-block of the first column.
                        // The chunk-sum table stays the shipped one, so any
                        // output move comes from the gather under test.
                        var flat = xf32.asArray(Float.self)
                        for i in 0 ..< 64 { flat[i] += 0.25 }
                        let xBad = MLXArray(flat, [1, m, cell.k])
                        eval(xBad)
                        let bad = probe.call(
                            x: xBad, xsums: xsums, set: set, m: m, groups: 1)
                        eval(bad)
                        let c = bad.asType(.float32).asArray(Float.self)
                        var controlDiffering = 0
                        for i in 0 ..< min(a.count, c.count) {
                            if a[i].bitPattern != c[i].bitPattern {
                                controlDiffering += 1
                            }
                        }
                        valueRows.append([
                            "cell": cell.name, "k": cell.k, "n": cell.n,
                            "m": m, "activation_dtype": "float32",
                            "positive_control": true,
                            "elements": c.count,
                            "differing": controlDiffering,
                        ])
                        #expect(
                            controlDiffering > 0,
                            """
                            \(cell.name) m=\(m): the positive control did NOT \
                            differ, so the bit-exactness gate above is vacuous
                            """)
                    }
                }
            }
        }

        #expect(!valueRows.isEmpty)
        try e219Write(
            "sanity_xdtype",
            [
                "deliverable":
                    "E223 gate: the float32 activation arm is bit-exact "
                    + "against the shipped bfloat16 arm",
                "value_comparisons": valueRows,
            ], session: nil)
    }
}

// MARK: - Section 1: the sweeps

@Suite(.serialized)
struct E219PassAnatomyTests {
    private static var replicaTarget: Int {
        e219Int("MLX_E219_REPLICA_TARGET_MB", 192) * 1_048_576
    }
    private static var replicaCap: Int {
        e219Int("MLX_E219_REPLICA_CAP_MB", 768) * 1_048_576
    }

    /// One timed unit per thermal arm, for one (cell, width, geometry, dtype).
    private func sweepArm(
        cell: E219Cell,
        arm: (na: Int, groups: Int),
        m: Int,
        stride: Int,
        geometry: (rows: Int, parameterized: Bool),
        geometryTag: String,
        activationF32: Bool,
        activations: MLXArray,
        xsums: MLXArray,
        coldArms: [Bool],
        cache: inout E219PipelineCache,
        units: inout [E219Unit],
        ringBox: E219RingBox
    ) throws {
        let pipeline = cache.get(
            na: arm.na, stride: stride, table: true, rows: geometry.rows,
            parameterized: geometry.parameterized,
            activationF32: activationF32)
        let bytes = E219Bytes(
            k: cell.k, n: cell.n, na: arm.na, stride: stride,
            rows: geometry.rows,
            activationBytesPerValue: activationF32 ? 4 : 2)
        let dtypeTag = activationF32 ? "xf32" : "xbf16"
        for cold in coldArms {
            let key = cell.name
            var fields: [String: Any] = [
                "cell": cell.name, "k": cell.k, "n": cell.n,
                "na": arm.na, "groups": arm.groups, "m": m,
                "stride": stride, "cold": cold, "dispatches": 1,
                "rows_per_simd": geometry.rows,
                "parameterized_header": geometry.parameterized,
                "geometry": geometryTag,
                "activation_dtype": activationF32 ? "float32" : "bfloat16",
                "activation_bytes_per_value": activationF32 ? 4 : 2,
                "bytes_per_row_per_kblock": bytes.bytesPerRowPerKBlock,
                "invocations_per_round": cell.invocations,
                "replicas": ringBox.count(key),
                "replica_set_bytes": ringBox.bytes(key),
            ]
            for (bk, bv) in bytes.dictionary { fields[bk] = bv }
            fields["pass_stream_bytes"] =
                (bytes.weight + bytes.scaleBias) * arm.groups
            units.append(
                E219Unit(
                    label:
                        "\(cell.name)/na\(arm.na)/g\(arm.groups)/"
                        + "\(geometryTag)/\(dtypeTag)/"
                        + (cold ? "cold" : "hot"),
                    fields: fields,
                    enqueue: {
                        let set = ringBox.next(key, cold: cold)
                        return [
                            pipeline.call(
                                x: activations, xsums: xsums, set: set, m: m,
                                groups: arm.groups)
                        ]
                    }))
        }
    }

    /// Shared driver: build one replica ring per cell shape, then time every
    /// requested `(NA, G, geometry, dtype, cold)` unit on it.
    private func sweep(
        phase: String,
        shapes: [E219Cell],
        arms: [(na: Int, groups: Int)],
        coldArms: [Bool],
        deliverable: String,
        /// `rows_per_simd` arms. The default is the shipped geometry compiled
        /// from the shipped header, so every pre-E221 phase is unchanged.
        geometries: [(rows: Int, parameterized: Bool)] = [(4, false)],
        /// Activation dtype arms. The default is the shipped bfloat16 slab, so
        /// every pre-E223 phase is unchanged.
        activationDTypes: [Bool] = [false],
        extra: [String: Any] = [:]
    ) throws {
        var cache = E219PipelineCache()
        var session = E219Session()
        var units: [E219Unit] = []
        var ringFacts: [[String: Any]] = []
        // Boxed so the escaping timed closures share one cursor per ring.
        let ringBox = E219RingBox()

        for cell in shapes {
            let ring = E219Replicas(
                k: cell.k, n: cell.n, seed: 0xE219_5EED,
                targetBytes: Self.replicaTarget, hardCapBytes: Self.replicaCap)
            ringFacts.append([
                "cell": cell.name, "k": cell.k, "n": cell.n,
                "replicas": ring.sets.count,
                "replica_set_bytes": ring.totalBytes,
                "bytes_per_replica": ring.sets[0].bytes,
            ])
            ringBox.install(cell.name, ring)
        }

        for cell in shapes {
            for arm in arms {
                let m = arm.na * arm.groups
                guard m <= 16 else { continue }
                let stride = e219SumsStride(m)
                let x = e219Activations(
                    m: m, k: cell.k, seed: UInt64(0xE219_1000 + m))
                let xsums = Qwen35CustomQMV.xsumsTable(x)
                // bfloat16 to float32 is exact, so this is the same activation
                // slab in a wider container, not a different input.
                let xf32 = x.asType(.float32)
                eval(xsums, xf32)
                for geometry in geometries {
                    guard cell.n % (2 * geometry.rows) == 0 else { continue }
                    let geometryTag = geometry.parameterized
                        ? "rows\(geometry.rows)param" : "rows\(geometry.rows)"
                    for f32 in activationDTypes {
                        try sweepArm(
                            cell: cell, arm: arm, m: m, stride: stride,
                            geometry: geometry, geometryTag: geometryTag,
                            activationF32: f32,
                            activations: f32 ? xf32 : x, xsums: xsums,
                            coldArms: coldArms, cache: &cache,
                            units: &units, ringBox: ringBox)
                    }
                }
            }
        }

        session.run(units, settle: e219Settle())
        try e219Write(
            phase,
            [
                "deliverable": deliverable,
                "replica_rings": ringFacts,
                "cells": shapes.map {
                    ["name": $0.name, "k": $0.k, "n": $0.n,
                     "invocations_per_round": $0.invocations]
                },
            ].merging(extra) { a, _ in a },
            session: session)
    }

    /// `NA` is both the inputs-per-group and the activation width, so this one
    /// sweep is the standalone analogue of FINDING 559's `f(IPG)` AND the
    /// assignment's token-column sweep.
    @Test(.enabled(if: e219PhaseEnabled("ipg")))
    func inputsPerGroupSweep() throws {
        let nas = e219IntList("MLX_E219_NA", [2, 3, 4, 5])
        try sweep(
            phase: "ipg",
            shapes: e219ScoredCells,
            arms: nas.map { (na: $0, groups: 1) },
            coldArms: [true, false],
            deliverable:
                "standalone f(IPG) at G = 1, rows = 4, and the hot-vs-cold "
                + "delta that bounds the cacheable share of the stream")
    }

    /// The per-pass fixed cost, read directly: `G` moves and nothing else does.
    @Test(.enabled(if: e219PhaseEnabled("groups")))
    func weightPassSweep() throws {
        let groups = e219IntList("MLX_E219_GROUPS", [1, 2, 3])
        let nas = e219IntList("MLX_E219_GROUP_NA", [3, 5])
        var arms: [(na: Int, groups: Int)] = []
        for na in nas {
            for g in groups where na * g <= 16 { arms.append((na: na, groups: g)) }
        }
        try sweep(
            phase: "groups",
            shapes: e219ScoredCells,
            arms: arms,
            coldArms: [true, false],
            deliverable:
                "per-pass fixed cost from G at fixed NA; the G=2/G=1 ratio is "
                + "the FINDING 559 reconciliation gate")
    }

    /// Weight bytes per pass, at fixed output rows. `k` moves the stream and
    /// the unique activation slab; it does not move `output_bytes` or the
    /// threadgroup count, which is what identifies `b` against `d`.
    @Test(.enabled(if: e219PhaseEnabled("kext")))
    func weightExtentSweep() throws {
        let ks = e219IntList("MLX_E219_K", [2048, 4096, 8192, 12288, 17408])
        try sweep(
            phase: "kext",
            shapes: ks.map {
                E219Cell(name: "kext.k\($0)", k: $0, n: 5120, invocations: 0)
            },
            arms: [(na: 5, groups: 1)],
            coldArms: [true],
            deliverable:
                "weight-stream coefficient b: k ladder at n = 5120, NA = 5, "
                + "G = 1")
    }

    /// Output rows. `n` moves the stream, the output bytes and the threadgroup
    /// count together; the joint fit with the `k` ladder separates them.
    @Test(.enabled(if: e219PhaseEnabled("next")))
    func outputExtentSweep() throws {
        let ns = e219IntList("MLX_E219_N", [4096, 8192, 16480, 24576, 34816])
        try sweep(
            phase: "next",
            shapes: ns.map {
                E219Cell(name: "next.n\($0)", k: 5120, n: $0, invocations: 0)
            },
            arms: [(na: 5, groups: 1)],
            coldArms: [true],
            deliverable:
                "output and threadgroup coefficients: n ladder at k = 5120, "
                + "NA = 5, G = 1")
    }

    /// E221: outputs per simdgroup. Halving `rows_per_simd` halves the
    /// accumulator registers each lane holds AND doubles the simdgroup count,
    /// so the activation slab and the chunk-sum table are re-read twice
    /// (FINDING 576). The two premises therefore predict opposite signs, and
    /// this sweep reads which one governs the NA slope.
    ///
    /// Three arms, so a null result cannot be blamed on the substitution:
    /// `rows4` is the shipped header and reconciles against every earlier
    /// phase, `rows4param` is the same geometry through the parameterized
    /// header and must be timed-inert, and `rows2param` is the treatment.
    @Test(.enabled(if: e219PhaseEnabled("rows")))
    func rowsPerSimdgroupSweep() throws {
        let nas = e219IntList("MLX_E219_ROWS_NA", [4, 5])
        let groups = e219IntList("MLX_E219_ROWS_GROUPS", [1, 2])
        var arms: [(na: Int, groups: Int)] = []
        for na in nas {
            for g in groups where na * g <= 16 {
                arms.append((na: na, groups: g))
            }
        }
        try sweep(
            phase: "rows",
            shapes: e219ScoredCells,
            arms: arms,
            coldArms: [true, false],
            deliverable:
                "E221 step 1: does rows_per_simd 4 -> 2 relieve the NA slope, "
                + "or does the doubled activation and chunk-sum re-read "
                + "dominate it (FINDING 576)?",
            geometries: [(4, false), (4, true), (2, true)],
            extra: [
                "e221_write_census": e219ScoredCells.flatMap { cell in
                    [4, 2].map { rows -> [String: Any] in
                        let census = e221WriteCensus(n: cell.n, rows: rows)
                        return [
                            "cell": cell.name, "n": cell.n, "rows": rows,
                            "rows_written": census.count,
                            "min_row": census.keys.min() ?? -1,
                            "max_row": census.keys.max() ?? -1,
                            "min_writes": census.values.min() ?? 0,
                            "max_writes": census.values.max() ?? 0,
                        ]
                    }
                },
                "e221_register_witness":
                    "research/e221-artifacts/e221_register_probe.json",
            ])
    }

    /// E223 step 1: what is one removed non-arithmetic instruction worth?
    ///
    /// The float32 activation arm removes 12.6 of the 123.2 machine
    /// instructions each column runs per k-block on `applegpu_g17s`, all of
    /// them `bfloat16` to `float` conversions, and issues twice the activation
    /// bytes to do it. Both arms run the same geometry, the same weight stream,
    /// the same chunk-sum table and bit-identical arithmetic, so the measured
    /// delta separates the two readings the desk cannot separate:
    ///
    ///  - if cost is proportional to instructions, the arm is 10.2% faster per
    ///    column before the extra bytes are charged;
    ///  - if cost is limited by floating-point issue alone, removing a
    ///    conversion is worth nothing and only the extra bytes show up.
    ///
    /// The answer prices every future non-arithmetic instruction in this kernel,
    /// not just this arm.
    @Test(.enabled(if: e219PhaseEnabled("xdtype")))
    func activationDTypeSweep() throws {
        let nas = e219IntList("MLX_E219_XDTYPE_NA", [2, 3, 4, 5])
        let groups = e219IntList("MLX_E219_XDTYPE_GROUPS", [1, 2])
        var arms: [(na: Int, groups: Int)] = []
        for na in nas {
            for g in groups where na * g <= 16 {
                arms.append((na: na, groups: g))
            }
        }
        try sweep(
            phase: "xdtype",
            shapes: e219ScoredCells,
            arms: arms,
            coldArms: [true, false],
            deliverable:
                "E223 step 1: price one removed bfloat16-to-float conversion "
                + "per column per k-block against the doubled activation bytes "
                + "that removing it costs",
            activationDTypes: [false, true],
            extra: [
                "e223_census":
                    "research/e223-artifacts/e223-desk-pricing.json",
                "e223_insn_per_column_per_kblock_g17s": [
                    "bfloat16": 123.2, "float32": 110.6,
                ],
            ])
    }

    /// Identical total work in 1, 2 or 4 dispatches. Weight bytes, output
    /// bytes and threadgroup count are all conserved, so the slope in the
    /// dispatch count is `a_dispatch` with no byte term attached.
    @Test(.enabled(if: e219PhaseEnabled("dsplit")))
    func dispatchSplitSweep() throws {
        let splits = e219IntList("MLX_E219_SPLITS", [1, 2, 4, 8])
        let cellNames = Set(
            (ProcessInfo.processInfo.environment["MLX_E219_DSPLIT_CELLS"]
                ?? "mlp.down,gdn.in_proj")
                .split(separator: ",").map(String.init))
        let shapes = e219ScoredCells.filter { cellNames.contains($0.name) }
        let na = e219Int("MLX_E219_DSPLIT_NA", 5)
        let m = na
        let stride = e219SumsStride(m)

        var cache = E219PipelineCache()
        var session = E219Session()
        var units: [E219Unit] = []
        var ringFacts: [[String: Any]] = []
        let ringBox = E219RingBox()

        for cell in shapes {
            let pipeline = cache.get(na: na, stride: stride, table: true)
            let x = e219Activations(
                m: m, k: cell.k, seed: UInt64(0xE219_2000 + m))
            let xsums = Qwen35CustomQMV.xsumsTable(x)
            eval(xsums)
            for d in splits {
                let chunk = cell.n / d
                guard chunk % 8 == 0, chunk >= 8 else { continue }
                // One ring per (cell, d): `d` chunks of `n/d` rows per replica,
                // so total streamed bytes are identical across `d`.
                //
                // The ring byte total is held CONSTANT across `d`, not the
                // replica count: the cursor must walk the same number of
                // streamed bytes before it revisits a replica, otherwise a
                // high-`d` arm would run warmer than a low-`d` arm and the
                // fitted dispatch overhead would absorb a cache effect.
                let key = "\(cell.name)/d\(d)"
                let ring = E219Replicas(
                    k: cell.k, n: chunk, seed: 0xE219_7000 + UInt64(d),
                    targetBytes: Self.replicaTarget,
                    hardCapBytes: Self.replicaCap / max(1, splits.count))
                ringFacts.append([
                    "cell": cell.name, "dispatches": d, "chunk_n": chunk,
                    "replicas": ring.sets.count,
                    "replica_set_bytes": ring.totalBytes,
                ])
                ringBox.install(key, ring)
                let bytes = E219Bytes(
                    k: cell.k, n: chunk, na: na, stride: stride)
                var fields: [String: Any] = [
                    "cell": cell.name, "k": cell.k, "n": cell.n,
                    "chunk_n": chunk, "na": na, "groups": 1, "m": m,
                    "stride": stride, "cold": true, "dispatches": d,
                    "invocations_per_round": cell.invocations,
                    "replicas": ringBox.count(key),
                    "replica_set_bytes": ringBox.bytes(key),
                ]
                for (bk, bv) in bytes.dictionary { fields[bk] = bv * d }
                fields["chunk_weight_bytes"] = bytes.weight
                units.append(
                    E219Unit(
                        label: "\(cell.name)/d\(d)", fields: fields,
                        enqueue: {
                            var outputs: [MLXArray] = []
                            outputs.reserveCapacity(d)
                            for _ in 0 ..< d {
                                let set = ringBox.next(key, cold: true)
                                outputs.append(
                                    pipeline.call(
                                        x: x, xsums: xsums, set: set, m: m,
                                        groups: 1))
                            }
                            return outputs
                        }))
            }
        }

        session.run(units, settle: e219Settle())
        try e219Write(
            "dsplit",
            [
                "deliverable":
                    "dispatch-overhead coefficient a: identical total work in "
                    + "1, 2, 4 or 8 dispatches",
                "replica_rings": ringFacts,
                "na": na,
            ], session: session)
    }
}

/// Replica rings live behind a reference box because the timed closures
/// escape and must share one cursor per ring.
private final class E219RingBox: @unchecked Sendable {
    private var rings: [String: E219Replicas] = [:]

    func install(_ key: String, _ ring: E219Replicas) { rings[key] = ring }
    func count(_ key: String) -> Int { rings[key]?.sets.count ?? 0 }
    func bytes(_ key: String) -> Int { rings[key]?.totalBytes ?? 0 }

    func next(_ key: String, cold: Bool) -> E219WeightSet {
        guard var ring = rings[key] else {
            preconditionFailure("[e219] no replica ring for \(key)")
        }
        let set = ring.next(cold: cold)
        rings[key] = ring
        return set
    }
}
