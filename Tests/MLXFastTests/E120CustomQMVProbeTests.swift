import Foundation
import MLX
import MLXLLM
import MLXRandom
import Testing

// E120 -- own the wide affine-4/group-64 QMV dispatch.
//
// `quantized.cpp` is outside the editable surface, so the shipped wide kernel
// can never be given a tenth buffer and can never be launched on a grid the
// frozen launcher does not choose. `Qwen35CustomQMV` replaces the launcher, not
// the arithmetic: `MLXFast.metalKernel` binds exactly the buffers we name and
// `custom_kernel.cpp:113-117` dispatches exactly the grid we name.
//
// RUNG 1 GATE. Before any table or grid work is built on that capability, the
// replica must (1) be bit exact against `quantizedMM` on the same buffers and
// (2) match MLX end to end within about 1 %. A replica that cannot match MLX on
// identical arithmetic and identical geometry kills Route A outright.
//
// The third gate number is the per-dispatch host cost of the custom path. It is
// NOT only the `it->second != source_` comparison at `custom_kernel.cpp:57-71`:
// `metal_kernel.cpp:317-329` rebuilds the whole generated kernel source string
// on EVERY call, at graph-construction time, before any of it reaches the
// encoder. `hostBuild` below times graph construction with no `eval` at all, so
// it separates that host cost from every GPU effect, and the padded-source pair
// isolates the part of it that scales with source length.
//
// Research instrument. Timing is off unless `MLXFAST_RUN_E120_PROBE=1`;
// exactness runs with the standard `MLXFAST_RUN_MLX_RUNTIME_TESTS=1` sweep.
// `Tests/` is never packaged into a submission. Within-session relative
// measurement: no thermal gate, no score. `cool_gate_passed_real_gate=false`
// and `gate_qualified_for_timing=false` are recorded in the output.

private struct E120Weights {
    var packed: MLXArray
    var scales: MLXArray
    var biases: MLXArray
}

private struct SplitMix64 {
    var state: UInt64
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

private struct E120Arm {
    var name: String
    var body: () -> [MLXArray]
}

@Suite("E120 candidate-owned QMV dispatch")
struct E120CustomQMVProbeTests {
    static let runtimeEnabled =
        ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1"
    static let timingEnabled =
        ProcessInfo.processInfo.environment["MLXFAST_RUN_E120_PROBE"] == "1"

    static let groupSize = 64
    static let bits = 4

    static let presets: [String: (hidden: Int, outputs: Int)] = [
        "mlp.gate_up": (5120, 34816),
        "lm_head": (5120, 248_320),
        "gdn.in_proj": (5120, 16480),
        "fa.qkv": (5120, 14336),
        "mlp.down": (17408, 5120),
        // 24 query heads x 256. 12 k-blocks, a chunk count no other cell has.
        "fa.o_proj": (6144, 5120),
        // linear_num_value_heads 48 x 128 head_v_dim. Same shape as
        // `fa.o_proj`; both names are kept so the report names the call site.
        "gdn.out_proj": (6144, 5120),
        // Smallest wide-eligible output width at the shipped K. The fill cost
        // does not depend on N, so a cheap consumer resolves it better.
        "stream.small": (5120, 4096),
        // One k-block, so the fill has almost no work left and what remains is
        // the dispatch boundary itself.
        "control.small": (512, 4096),
    ]

    static var shapes: [(name: String, hidden: Int, outputs: Int)] {
        let requested =
            ProcessInfo.processInfo.environment["MLXFAST_E120_SHAPES"] ?? "mlp.gate_up"
        return requested.split(separator: ",").compactMap { token in
            let text = token.trimmingCharacters(in: .whitespaces)
            let parts = text.split(separator: ":")
            if parts.count == 3, let n = Int(parts[1]), let k = Int(parts[2]) {
                return (String(parts[0]), k, n)
            }
            guard let preset = presets[text] else { return nil }
            return (text, preset.hidden, preset.outputs)
        }
    }

    static var widths: [Int] {
        guard let requested = ProcessInfo.processInfo.environment["MLXFAST_E120_WIDTHS"],
            !requested.isEmpty
        else { return Array(3 ... 9) }
        return requested.split(separator: ",").compactMap { Int($0) }
    }

    static var blocks: Int {
        Int(ProcessInfo.processInfo.environment["MLXFAST_E120_BLOCKS"] ?? "") ?? 6
    }

    /// Dependent QMV dispatches per chain in the fill-cost probe. A chain makes
    /// the fill an in-stream dispatch with a real barrier in front of it, which
    /// is what a standalone probe cannot measure.
    static var layers: Int {
        Int(ProcessInfo.processInfo.environment["MLXFAST_E120_LAYERS"] ?? "") ?? 32
    }

    static let exactnessArms: [Qwen35CustomQMV.Arm] = [.replica, .fillNoConsume, .sumTable]

    static var rampSeconds: Double {
        Double(ProcessInfo.processInfo.environment["MLXFAST_E120_RAMP_S"] ?? "") ?? 0.30
    }

    static var targetMicroseconds: Double {
        Double(ProcessInfo.processInfo.environment["MLXFAST_E120_TARGET_US"] ?? "") ?? 100_000
    }

    // MARK: fixtures

    /// Full 32-bit packed nibbles, so every nibble position sees the whole
    /// 0...15 range. `MLXRandom.randInt` tops out below 2^31 and would leave the
    /// high nibble permanently small.
    fileprivate static func makeWeights(hidden: Int, outputs: Int, seed: UInt64) -> E120Weights {
        let words = outputs * hidden / 8
        var rng = SplitMix64(state: seed)
        var raw = [UInt32]()
        raw.reserveCapacity(words)
        for _ in 0 ..< words { raw.append(UInt32(truncatingIfNeeded: rng.next())) }
        let packed = MLXArray(raw).reshaped([outputs, hidden / 8])
        MLXRandom.seed(seed)
        let scales = MLXRandom.uniform(
            low: Float(0.004), high: Float(0.02), [outputs, hidden / groupSize]
        ).asType(.bfloat16)
        let biases = MLXRandom.uniform(
            low: Float(-0.06), high: Float(0.06), [outputs, hidden / groupSize]
        ).asType(.bfloat16)
        eval(packed, scales, biases)
        return E120Weights(packed: packed, scales: scales, biases: biases)
    }

    fileprivate static func mlx(_ x: MLXArray, _ w: E120Weights) -> MLXArray {
        quantizedMM(
            x, w.packed, scales: w.scales, biases: w.biases, transpose: true,
            groupSize: groupSize, bits: bits, mode: .affine)
    }

    fileprivate static func custom(_ x: MLXArray, _ w: E120Weights, _ arm: Qwen35CustomQMV.Arm)
        -> MLXArray?
    {
        Qwen35CustomQMV.matmul(
            x, w.packed, scales: w.scales, biases: w.biases,
            groupSize: groupSize, bits: bits, mode: .affine, arm: arm)
    }

    /// Number of output elements that differ. BF16 to float32 is exact, so a
    /// float32 inequality count is an exact bit comparison for finite values.
    static func mismatches(_ a: MLXArray, _ b: MLXArray) -> Int {
        (a .!= b).asType(.int32).sum().item(Int.self)
    }

    static func maxAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }

    // MARK: timing helpers

    static func timed(_ count: Int, _ body: () -> [MLXArray]) -> Double {
        let start = DispatchTime.now().uptimeNanoseconds
        for _ in 0 ..< count { eval(body()) }
        return Double(DispatchTime.now().uptimeNanoseconds - start) / 1e3 / Double(count)
    }

    static func rampBurst(_ body: () -> [MLXArray], seconds: Double) {
        let start = DispatchTime.now().uptimeNanoseconds
        while Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9 < seconds {
            eval(body())
        }
    }

    /// Brings every arm to the same thermal and allocator state before a block.
    ///
    /// HARNESS DEFECT. Ramping `arms[0]` alone put the whole burst's allocator
    /// growth immediately before the first timed arm, which is always `arms[0]`
    /// in the forward pass. ABBA cannot cancel that, because the ramp sits
    /// outside the counterbalanced pair. Measured on E129 at `b15aaf51`: the
    /// byte-identical M=5 control read +38.95 % for `a_compare`, and
    /// `a_compare`'s forward leg ran up to 12x its own reverse leg while
    /// `b_shipped` matched itself within 1 %.
    ///
    /// Each arm now gets an equal share of the burst, and every arm then takes
    /// a discarded timed run so the reclaim that follows the burst is charged
    /// to no arm in particular.
    static func settle(_ arms: [E120Arm], seconds: Double) {
        let share = seconds / Double(max(arms.count, 1))
        for arm in arms { rampBurst(arm.body, seconds: share) }
        for arm in arms { _ = timed(2, arm.body) }
    }

    /// Graph construction only: build the op and drop it without evaluating.
    /// This is the whole host-side price of a dispatch, including the source
    /// string that `write_signature` rebuilds on every call.
    static func hostBuild(_ count: Int, _ body: () -> MLXArray) -> Double {
        for _ in 0 ..< 200 { _ = body() }
        let start = DispatchTime.now().uptimeNanoseconds
        for _ in 0 ..< count { _ = body() }
        return Double(DispatchTime.now().uptimeNanoseconds - start) / 1e3 / Double(count)
    }

    // MARK: rung 0 dispatch table

    /// The dispatch table reaches the built worker only through interpolation
    /// into the Metal source, so `strings` cannot witness it and
    /// `rebuild-and-assert-worker.sh --require` silently passes on a table that
    /// is not there. `planWitness` is the literal that closes that hole, and it
    /// is only a witness while it renders the plan it claims to describe.
    @Test("every dispatch table matches its literal witness")
    func planWitnessMatchesWidthPlan() throws {
        #expect(
            Qwen35CustomQMV.renderPlan(Qwen35CustomQMV.shippedPlan)
                == Qwen35CustomQMV.shippedPlanWitness)
        #expect(
            Qwen35CustomQMV.renderPlan(Qwen35CustomQMV.onePassPlan)
                == Qwen35CustomQMV.onePassPlanWitness)
        #expect(
            Qwen35CustomQMV.renderPlan(Qwen35CustomQMV.widthPlan)
                == Qwen35CustomQMV.planWitness)
        for tier in [nil] + Qwen35CustomQMV.tiers.map(Optional.init) {
            for useTable in [true, false] {
                #expect(
                    Qwen35CustomQMV.pipelineSource(useTable: useTable, tier: tier)
                        .contains(Qwen35CustomQMV.planWitness),
                    "witness missing from tier \(tier as Any) table \(useTable)")
            }
        }
    }

    /// A table that breaks either kernel precondition compiles into a body the
    /// generator never validated, so check both here rather than at dispatch.
    @Test("every dispatch table routes each width to a legal case body")
    func widthPlansAreWellFormed() throws {
        let plans: [(String, [(m: Int, ipg: Int, rps: Int)])] = [
            ("shipped", Qwen35CustomQMV.shippedPlan),
            ("onepass", Qwen35CustomQMV.onePassPlan),
        ]
        for (name, plan) in plans {
            #expect(plan.map(\.m) == Array(Qwen35CustomQMV.widths), "\(name) width coverage")
            for cell in plan {
                #expect(cell.ipg >= 1 && cell.ipg <= cell.m, "\(name) M=\(cell.m) ipg range")
                // The kernel asserts `M % IPG != 1`: a one-row tail group would
                // read past the block.
                #expect(cell.m % cell.ipg != 1, "\(name) M=\(cell.m) tail group")
                #expect(cell.rps >= 1, "\(name) M=\(cell.m) rps range")
            }
        }
    }

    /// The one-pass arm moves exactly the widths that made more than one pass
    /// and the board can still reach. M=9 is above the ranked verify cap, so
    /// moving it would buy nothing and would add a seventh pipeline.
    @Test("the one-pass table moves exactly M=6, M=7 and M=8")
    func onePassTableMovesOnlyTheMultiPassWidths() throws {
        let shipped = Dictionary(
            uniqueKeysWithValues: Qwen35CustomQMV.shippedPlan.map { ($0.m, ($0.ipg, $0.rps)) })
        let onePass = Dictionary(
            uniqueKeysWithValues: Qwen35CustomQMV.onePassPlan.map { ($0.m, ($0.ipg, $0.rps)) })
        let moved = shipped.keys.filter { shipped[$0]! != onePass[$0]! }.sorted()
        #expect(moved == [6, 7, 8])
        for m in moved {
            #expect(onePass[m]!.0 == m, "M=\(m) makes one pass")
            // `rps = 4` costs the g17s entry point 114 registers at M=6 and
            // spills at M=7. Halving it is what makes one pass fit.
            #expect(onePass[m]!.1 == 2, "M=\(m) halves rows per simdgroup")
        }
        #expect(onePass[9]!.0 == shipped[9]!.0)
    }

    @Test("every tier of every table has its own entry-point name")
    func everyTierHasADistinctEntryPoint() throws {
        #expect(Set(Qwen35CustomQMV.shippedPlan.map(\.ipg)).sorted() == [3, 4, 5])
        #expect(Set(Qwen35CustomQMV.onePassPlan.map(\.ipg)).sorted() == [3, 4, 5, 6, 7, 8])

        // MLX keys its library cache by name and recompiles when one name is
        // seen with a different source, so two tiers sharing a name would
        // thrash the cache instead of specializing.
        var names: Set<String> = []
        for tier in [nil] + [3, 4, 5, 6, 7, 8].map(Optional.init) {
            for useTable in [true, false] {
                let name = Qwen35CustomQMV.pipelineName(useTable: useTable, tier: tier)
                #expect(!names.contains(name), "duplicate entry point \(name)")
                names.insert(name)
            }
        }
    }

    // MARK: rung 1 exactness

    @Test(
        "replica is bit exact against quantizedMM at every routed width",
        .enabled(if: E120CustomQMVProbeTests.runtimeEnabled))
    func replicaExactness() throws {
        var records: [[String: Any]] = []

        for shape in Self.shapes + [("control.small", 512, 4096)] {
            let w = Self.makeWeights(
                hidden: shape.hidden, outputs: shape.outputs,
                seed: UInt64(0xE120) &+ UInt64(shape.outputs))
            MLXRandom.seed(UInt64(0xB1F5) &+ UInt64(shape.outputs))
            let block = MLXRandom.normal([9, shape.hidden]).asType(.bfloat16)
            eval(block)

            // `meta_hit` control: metadata moved by one step. Proves the
            // scale/bias reads are live and not folded away.
            let wBumped = E120Weights(
                packed: w.packed,
                scales: (w.scales.asType(.float32) + Float(0.001)).asType(.bfloat16),
                biases: w.biases)
            eval(wBumped.scales)

            for width in 3 ... 9 {
                let x = contiguous(block[0 ..< width])
                eval(x)
                let reference = Self.mlx(x, w)
                eval(reference)

                // `x_hit` control: one activation moved by half a unit must
                // change the output, or the comparison cannot fail.
                let bumped = x + MLXArray(Float(0.5)).asType(.bfloat16)
                eval(bumped)

                let table = Qwen35CustomQMV.xsumsTable(x)
                eval(table)
                // `table_hit` control: one table entry moved must change the
                // output of the consuming arm, and restoring it must return the
                // output to bit equality. Entry 0 is (k_block 0, lane 0, m 0),
                // which every width reads.
                var oneHot = [Float](repeating: 0, count: table.size)
                oneHot[0] = 1024
                let badTable = table + MLXArray(oneHot)
                eval(badTable)

                for arm in Self.exactnessArms {
                    guard let candidate = Self.custom(x, w, arm) else {
                        Issue.record(
                            "\(shape.name) M=\(width) \(arm.rawValue): declined a routed cell")
                        continue
                    }
                    guard let xHit = Self.custom(bumped, w, arm),
                        let metaHit = Self.custom(x, wBumped, arm)
                    else {
                        Issue.record("\(shape.name) M=\(width) \(arm.rawValue): control declined")
                        continue
                    }
                    eval(candidate, xHit, metaHit)

                    var tableHit = -1
                    var restoredDiff = -1
                    if arm == .sumTable {
                        guard
                            let perturbed = Qwen35CustomQMV.matmulWithTable(
                                x, w.packed, scales: w.scales, biases: w.biases,
                                xsums: badTable, groupSize: Self.groupSize,
                                bits: Self.bits, mode: .affine),
                            let restored = Qwen35CustomQMV.matmulWithTable(
                                x, w.packed, scales: w.scales, biases: w.biases,
                                xsums: table, groupSize: Self.groupSize,
                                bits: Self.bits, mode: .affine)
                        else {
                            Issue.record("\(shape.name) M=\(width): table control declined")
                            continue
                        }
                        eval(perturbed, restored)
                        tableHit = Self.mismatches(reference, perturbed)
                        restoredDiff = Self.mismatches(reference, restored)
                    }

                    let differing = Self.mismatches(reference, candidate)
                    let xHitDiffering = Self.mismatches(reference, xHit)
                    let metaHitDiffering = Self.mismatches(reference, metaHit)
                    var record: [String: Any] = [
                        "shape": shape.name,
                        "outputs": shape.outputs,
                        "hidden": shape.hidden,
                        "width": width,
                        "arm": arm.rawValue,
                        "elements": reference.size,
                        "differing_elements": differing,
                        "max_abs_diff": Self.maxAbsDiff(reference, candidate),
                        "bit_exact": differing == 0,
                        "x_hit": xHitDiffering,
                        "meta_hit": metaHitDiffering,
                        "positive_control_can_fail": xHitDiffering > 0 && metaHitDiffering > 0,
                    ]
                    if arm == .sumTable {
                        record["table_hit"] = tableHit
                        record["restored_diff"] = restoredDiff
                        // 5b: the shipped gate is a pure function of the width,
                        // so record what it decided here.
                        record["table_routed"] = Qwen35CustomQMV.tablePays(m: width)
                        #expect(tableHit > 0)
                        #expect(restoredDiff == 0)
                    }
                    records.append(record)
                    print(
                        "E120_EXACT "
                            + (String(
                                data: (try? JSONSerialization.data(
                                    withJSONObject: record, options: [.sortedKeys])) ?? Data(),
                                encoding: .utf8) ?? "{}"))
                    fflush(stdout)
                    #expect(differing == 0)
                    #expect(xHitDiffering > 0)
                    #expect(metaHitDiffering > 0)
                }
            }

            // The guard must decline every width the incumbent owns.
            for width in [1, 2, 10, 12] {
                let x = contiguous(
                    MLXRandom.normal([width, shape.hidden]).asType(.bfloat16))
                eval(x)
                #expect(Self.custom(x, w, .replica) == nil)
            }

            // 5c: the left half of a [6, 2K] block keeps row stride 2K. The
            // kernels declare `ensureRowContiguous: true`, so MLX would copy
            // it and the answer would still be right; the guard exists so the
            // cell goes back to `quantizedMM`, which reads the stride without
            // a copy. This asserts the guard fires and that the dense copy of
            // the same values is still routed and bit exact.
            let wide = MLXRandom.normal([6, 2 * shape.hidden]).asType(.bfloat16)
            eval(wide)
            let leftHalf = wide[0 ..< 6, 0 ..< shape.hidden]
            eval(leftHalf)
            #expect(leftHalf.shape == [6, shape.hidden])
            let dense = Qwen35CustomQMV.rowContiguous(leftHalf, rowStride: shape.hidden)
            #expect(dense == false, "MLX materialised the slice; the stride case is untested")
            if !dense {
                #expect(Self.custom(leftHalf, w, .replica) == nil)
                #expect(Self.custom(leftHalf, w, .sumTable) == nil)
                // The same values, densely packed, must be routed and exact.
                let packed = contiguous(leftHalf)
                eval(packed)
                guard let routed = Self.custom(packed, w, .sumTable) else {
                    Issue.record("\(shape.name): declined the packed copy")
                    continue
                }
                let reference = Self.mlx(packed, w)
                eval(routed, reference)
                #expect(Self.mismatches(reference, routed) == 0)
            }
        }

        if let path = ProcessInfo.processInfo.environment["MLXFAST_E120_EXACT_OUT"], !path.isEmpty {
            let data = try JSONSerialization.data(
                withJSONObject: ["records": records], options: [.prettyPrinted, .sortedKeys])
            try data.write(to: URL(fileURLWithPath: path))
        }

        #expect(!records.isEmpty)
    }

    // MARK: rung 1 host cost

    @Test(
        "per-dispatch host cost of the custom path against MLX",
        .enabled(if: E120CustomQMVProbeTests.timingEnabled))
    func hostDispatchCost() throws {
        let hidden = 5120
        let outputs = 34816
        let w = Self.makeWeights(hidden: hidden, outputs: outputs, seed: 0xE120)
        let x = contiguous(MLXRandom.normal([8, hidden]).asType(.bfloat16))
        eval(x)
        _ = Self.custom(x, w, .replica).map { eval($0) }
        eval(Self.mlx(x, w))

        let count = 4000
        let mlxBuild = Self.hostBuild(count) { Self.mlx(x, w) }
        let customBuild = Self.hostBuild(count) { Self.custom(x, w, .replica)! }

        // Source-length slope. Two trivial kernels with identical bodies and
        // different source lengths; the difference is the part of the host cost
        // that scales with the generated source string.
        let padding = String(repeating: "// e120 source length probe padding\n", count: 220)
        let tiny = MLXArray(Array(repeating: Float(1), count: 32))
        eval(tiny)
        let shortKernel = MLXFast.metalKernel(
            name: "e120_len_probe_short", inputNames: ["a"], outputNames: ["o"],
            source: "o[thread_position_in_grid.x] = a[thread_position_in_grid.x];",
            header: "", ensureRowContiguous: false)
        let longKernel = MLXFast.metalKernel(
            name: "e120_len_probe_long", inputNames: ["a"], outputNames: ["o"],
            source: "o[thread_position_in_grid.x] = a[thread_position_in_grid.x];",
            header: padding, ensureRowContiguous: false)
        let shortBuild = Self.hostBuild(count) {
            shortKernel(
                [tiny], grid: (32, 1, 1), threadGroup: (32, 1, 1),
                outputShapes: [[32]], outputDTypes: [.float32])[0]
        }
        let longBuild = Self.hostBuild(count) {
            longKernel(
                [tiny], grid: (32, 1, 1), threadGroup: (32, 1, 1),
                outputShapes: [[32]], outputDTypes: [.float32])[0]
        }

        let record: [String: Any] = [
            "mlx_graph_build_us": mlxBuild,
            "custom_graph_build_us": customBuild,
            "custom_minus_mlx_us": customBuild - mlxBuild,
            "len_probe_short_us": shortBuild,
            "len_probe_long_us": longBuild,
            "len_probe_padding_bytes": padding.utf8.count,
            "len_probe_us_per_kib": (longBuild - shortBuild) / (Double(padding.utf8.count) / 1024.0),
            "replicates": count,
        ]
        print(
            "E120_HOSTCOST "
                + (String(
                    data: (try? JSONSerialization.data(
                        withJSONObject: record, options: [.sortedKeys])) ?? Data(),
                    encoding: .utf8) ?? "{}"))
        fflush(stdout)
        #expect(customBuild > 0)
    }

    // MARK: rung 1 matched timing

    @Test(
        "matched ABBA timing of the replica against MLX",
        .enabled(if: E120CustomQMVProbeTests.timingEnabled))
    func matchedTiming() throws {
        var cells: [[String: Any]] = []

        for shape in Self.shapes {
            let w = Self.makeWeights(
                hidden: shape.hidden, outputs: shape.outputs,
                seed: UInt64(0xE120) &+ UInt64(shape.outputs))
            let maxWidth = Self.widths.max() ?? 9
            MLXRandom.seed(UInt64(0xB1F5) &+ UInt64(shape.outputs))
            let block = MLXRandom.normal([maxWidth, shape.hidden]).asType(.bfloat16)
            eval(block)

            for width in Self.widths {
                let x = contiguous(block[0 ..< width])
                eval(x)

                var arms: [E120Arm] = [
                    E120Arm(name: "a_mlx", body: { [Self.mlx(x, w)] })
                ]
                if Self.custom(x, w, .replica) != nil {
                    arms.append(
                        E120Arm(
                            name: "b_replica",
                            body: { [Self.custom(x, w, .replica)!] }))
                }

                Self.rampBurst(arms[0].body, seconds: Self.rampSeconds)
                let probeUs = Self.timed(8, arms[0].body)
                let count = max(8, min(600, Int(Self.targetMicroseconds / max(probeUs, 1.0))))
                for arm in arms { _ = Self.timed(3, arm.body) }

                for blockIndex in 0 ..< Self.blocks {
                    let entryTemp = e120GPUTemperature()
                    Self.settle(arms, seconds: Self.rampSeconds)
                    let forward = arms.map { Self.timed(count, $0.body) }
                    let reverse = Array(
                        arms.reversed().map { Self.timed(count, $0.body) }.reversed())
                    let exitTemp = e120GPUTemperature()

                    var record: [String: Any] = [
                        "shape": shape.name,
                        "outputs": shape.outputs,
                        "hidden": shape.hidden,
                        "width": width,
                        "block": blockIndex,
                        "replicates": count,
                        "arms": arms.enumerated().map { index, arm in
                            [
                                "arm": arm.name,
                                "forward_us": forward[index],
                                "reverse_us": reverse[index],
                            ] as [String: Any]
                        },
                    ]
                    if let entryTemp { record["gpu_temp_entry_c"] = entryTemp }
                    if let exitTemp { record["gpu_temp_exit_c"] = exitTemp }
                    cells.append(record)
                    print(
                        "E120_BLOCK "
                            + (String(
                                data: (try? JSONSerialization.data(
                                    withJSONObject: record, options: [.sortedKeys])) ?? Data(),
                                encoding: .utf8) ?? "{}"))
                    fflush(stdout)
                }
                eval(MLXArray(0))
            }
            Memory.clearCache()
        }

        if let path = ProcessInfo.processInfo.environment["MLXFAST_E120_OUT"], !path.isEmpty {
            let payload: [String: Any] = [
                "group_size": Self.groupSize,
                "bits": Self.bits,
                "ramp_seconds": Self.rampSeconds,
                "target_us": Self.targetMicroseconds,
                "cool_gate_passed_real_gate": false,
                "gate_qualified_for_timing": false,
                "blocks": Self.blocks,
                "widths": Self.widths,
                "cells": cells,
            ]
            let data = try JSONSerialization.data(
                withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
            try data.write(to: URL(fileURLWithPath: path))
        }

        #expect(!cells.isEmpty)
    }

    // MARK: rung 2 in-stream fill cost

    /// The marginal cost of one chunk-sum fill dispatch, measured inside a
    /// dependent stream rather than on its own.
    ///
    /// A standalone probe charges a fill one whole command-buffer round trip,
    /// which is why askeladd's E118 read 117 to 452 microseconds for a fill
    /// whose real work is about 0.3. Here each chain is `layers` QMV dispatches
    /// that depend on one another, so a fill is one extra dispatch and one
    /// extra barrier inside a stream that is already running, exactly as it
    /// would be in a decode round.
    ///
    /// Three arms separate the two numbers the advisor asked to keep apart:
    ///
    /// - `a_replica`      no table at all
    /// - `b_fill_noconsume` table produced and bound, kernel does not read it
    /// - `c_sumtable`     table produced, bound and read
    ///
    /// `b - a` is the fill cost. `b - c` is the consumer gain. `a - c` is the
    /// net.
    @Test(
        "marginal cost of one in-stream chunk-sum fill dispatch",
        .enabled(if: E120CustomQMVProbeTests.timingEnabled))
    func fillCost() throws {
        var cells: [[String: Any]] = []
        let layers = Self.layers
        let zero = MLXArray(Float(0)).asType(.bfloat16)
        eval(zero)

        for shape in Self.shapes {
            let w = Self.makeWeights(
                hidden: shape.hidden, outputs: shape.outputs,
                seed: UInt64(0xE120) &+ UInt64(shape.outputs))
            let maxWidth = Self.widths.max() ?? 9
            MLXRandom.seed(UInt64(0xB1F5) &+ UInt64(shape.outputs))
            let block = MLXRandom.normal([maxWidth, shape.hidden]).asType(.bfloat16)
            eval(block)

            for width in Self.widths {
                let x0 = contiguous(block[0 ..< width])
                eval(x0)
                guard Self.custom(x0, w, .replica) != nil else { continue }

                // The tail of each layer feeds the next one so the whole chain
                // serialises. The dependency carries no signal: the slice is
                // multiplied by zero, so `x` is bit identical to `x0` at every
                // layer and every arm sees the same activations.
                func chain(_ arm: Qwen35CustomQMV.Arm) -> [MLXArray] {
                    var x = x0
                    for _ in 0 ..< layers {
                        let y = Self.custom(x, w, arm)!
                        x = x0 + y[0 ..< 1, 0 ..< 1] * zero
                    }
                    return [x]
                }

                let arms: [E120Arm] = [
                    E120Arm(name: "a_replica", body: { chain(.replica) }),
                    E120Arm(name: "b_fill_noconsume", body: { chain(.fillNoConsume) }),
                    E120Arm(name: "c_sumtable", body: { chain(.sumTable) }),
                ]

                Self.rampBurst(arms[0].body, seconds: Self.rampSeconds)
                let probeUs = Self.timed(3, arms[0].body)
                let count = max(4, min(400, Int(Self.targetMicroseconds / max(probeUs, 1.0))))
                for arm in arms { _ = Self.timed(2, arm.body) }

                for blockIndex in 0 ..< Self.blocks {
                    let entryTemp = e120GPUTemperature()
                    Self.settle(arms, seconds: Self.rampSeconds)
                    let forward = arms.map { Self.timed(count, $0.body) }
                    let reverse = Array(
                        arms.reversed().map { Self.timed(count, $0.body) }.reversed())
                    let exitTemp = e120GPUTemperature()

                    let mean = (0 ..< arms.count).map { 0.5 * (forward[$0] + reverse[$0]) }
                    var record: [String: Any] = [
                        "shape": shape.name,
                        "outputs": shape.outputs,
                        "hidden": shape.hidden,
                        "width": width,
                        "block": blockIndex,
                        "layers": layers,
                        "replicates": count,
                        "k_blocks": shape.hidden / 512,
                        "table_bytes": (shape.hidden / 512) * 32
                            * Qwen35CustomQMV.sumsStride(width) * 4,
                        "fill_us_per_dispatch": (mean[1] - mean[0]) / Double(layers),
                        "consumer_gain_us_per_matvec": (mean[1] - mean[2]) / Double(layers),
                        "net_us_per_matvec": (mean[0] - mean[2]) / Double(layers),
                        "arms": arms.enumerated().map { index, arm in
                            [
                                "arm": arm.name,
                                "forward_us": forward[index],
                                "reverse_us": reverse[index],
                            ] as [String: Any]
                        },
                    ]
                    if let entryTemp { record["gpu_temp_entry_c"] = entryTemp }
                    if let exitTemp { record["gpu_temp_exit_c"] = exitTemp }
                    cells.append(record)
                    print(
                        "E120_FILL "
                            + (String(
                                data: (try? JSONSerialization.data(
                                    withJSONObject: record, options: [.sortedKeys])) ?? Data(),
                                encoding: .utf8) ?? "{}"))
                    fflush(stdout)
                }
                eval(MLXArray(0))
            }
            Memory.clearCache()
        }

        if let path = ProcessInfo.processInfo.environment["MLXFAST_E120_FILL_OUT"], !path.isEmpty {
            let payload: [String: Any] = [
                "group_size": Self.groupSize,
                "bits": Self.bits,
                "layers": layers,
                "ramp_seconds": Self.rampSeconds,
                "target_us": Self.targetMicroseconds,
                "cool_gate_passed_real_gate": false,
                "gate_qualified_for_timing": false,
                "blocks": Self.blocks,
                "widths": Self.widths,
                "cells": cells,
            ]
            let data = try JSONSerialization.data(
                withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
            try data.write(to: URL(fileURLWithPath: path))
        }

        #expect(!cells.isEmpty)
    }

    // MARK: rung 2 entry-point width switch

    typealias WidthPlan = [(m: Int, ipg: Int, rps: Int)]

    /// The plan the candidate ships, read from the source under test.
    static let sourcePlan: WidthPlan = Qwen35CustomQMV.widthPlan

    /// Applies an `M:IPG:RPS` override spec on top of the shipped plan. Widths
    /// the spec omits keep their shipped values, so the two pipelines execute
    /// identical code there and those widths isolate the shared entry point's
    /// occupancy channel.
    static func plan(overriding spec: String) -> WidthPlan {
        var table = [Int: (ipg: Int, rps: Int)](
            uniqueKeysWithValues: sourcePlan.map { ($0.m, (ipg: $0.ipg, rps: $0.rps)) })
        for entry in spec.split(separator: ",") {
            let parts = entry.split(separator: ":").compactMap { Int($0) }
            guard parts.count == 3 else { continue }
            table[parts[0]] = (ipg: parts[1], rps: parts[2])
        }
        return sourcePlan.map {
            (m: $0.m, ipg: table[$0.m]?.ipg ?? $0.ipg, rps: table[$0.m]?.rps ?? $0.rps)
        }
    }

    /// The `b_shipped` arm. Defaults to the plan in the source under test.
    /// `MLXFAST_E129_SHIPPED_CASES` moves it so any two plans can be paired,
    /// which is what isolates one axis at a time: holding `rows_per_simd`
    /// equal in both arms leaves only the pass count changing.
    static let shippedCases: WidthPlan = plan(
        overriding: ProcessInfo.processInfo.environment["MLXFAST_E129_SHIPPED_CASES"] ?? "")

    /// The `a_compare` arm. The default reverses the shipped plan back to the
    /// promoted two-pass table, so `a_compare` is the incumbent.
    static let compareCases: WidthPlan = plan(
        overriding: ProcessInfo.processInfo.environment["MLXFAST_E129_CASES"]
            ?? "6:3:4,7:4:4,8:4:4")

    static func passes(_ cases: WidthPlan) -> [String: Int] {
        var out: [String: Int] = [:]
        for entry in cases { out["\(entry.m)"] = (entry.m + entry.ipg - 1) / entry.ipg }
        return out
    }

    static func caseTag(_ cases: WidthPlan) -> String {
        cases.map { "\($0.m)x\($0.ipg)x\($0.rps)" }.joined(separator: "_")
    }

    /// One side of the comparison: a width plan, an entry-point layout, and the
    /// pipelines they produce.
    ///
    /// A Metal entry point is allocated `max` registers over every case body it
    /// inlines. `shared` puts all seven widths in one switch, so the widest
    /// body sets the occupancy of every routed width. `tiered` emits one entry
    /// point per distinct `ipg`, which is a width's own maximum because a tail
    /// group is never wider than a full group. Both layouts come from the same
    /// generator and execute identical code.
    fileprivate struct Arm {
        let plan: WidthPlan
        let tiered: Bool
        let label: String
        /// Keyed by `ipg`, or by `0` for the single shared switch.
        private let kernels: [Int: MLXFast.MLXFastKernel]

        init(_ plan: WidthPlan, tiered: Bool, label: String) {
            self.plan = plan
            self.tiered = tiered
            self.label = label
            let tiers = tiered ? Set(plan.map(\.ipg)).sorted() : [0]
            self.kernels = Dictionary(
                uniqueKeysWithValues: tiers.map {
                    ($0, casesKernel(plan, tier: tiered ? $0 : nil, label: label))
                })
        }

        var pipelineCount: Int { kernels.count }

        func kernel(m: Int) -> MLXFast.MLXFastKernel {
            let key = tiered ? (plan.first { $0.m == m }?.ipg ?? 0) : 0
            guard let kernel = kernels[key] else {
                preconditionFailure("arm \(label) has no pipeline for width \(m)")
            }
            return kernel
        }

        /// The same buffers, grid and threadgroup the shipped call sites bind.
        func apply(_ x: MLXArray, _ w: E120Weights, _ xsums: MLXArray) -> MLXArray {
            let n = w.packed.dim(0)
            let m = x.dim(-2)
            let entry = plan.first { $0.m == m } ?? (m: m, ipg: m, rps: 4)
            var outShape = x.shape
            outShape[outShape.count - 1] = n
            return kernel(m: m)(
                [w.packed, w.scales, w.biases, x, xsums],
                template: [("USE_TABLE", true)],
                grid: (m * 32, n / entry.rps, 1),
                threadGroup: (32, 2, 1),
                outputShapes: [outShape],
                outputDTypes: [.bfloat16]
            )[0]
        }
    }

    /// The shipped generated source, rebuilt under a private name with the
    /// template arguments and `out_row` expression of the changed width cases
    /// replaced, and optionally restricted to one `ipg` tier.
    fileprivate static func casesKernel(
        _ cases: WidthPlan, tier: Int?, label: String
    ) -> MLXFast.MLXFastKernel {
        var source = Qwen35CustomQMV.generatedSource(table: true, tier: nil)
        for (index, entry) in cases.enumerated() {
            let shipped = sourcePlan[index]
            precondition(entry.m == shipped.m, "case plans must cover the same widths")
            let from = """
                qwen_e120_qmv_m<\(shipped.m), \(shipped.ipg), \(shipped.rps), USE_TABLE>
                """
            precondition(source.contains(from), "shipped case \(from) not in generated source")
            guard entry.ipg != shipped.ipg || entry.rps != shipped.rps else { continue }
            source = source.replacingOccurrences(
                of: from,
                with: "qwen_e120_qmv_m<\(entry.m), \(entry.ipg), \(entry.rps), USE_TABLE>")
            let rowFrom = "int(qmv_tid.y) * \(2 * shipped.rps) + int(qmv_sgid) * \(shipped.rps)"
            let rowTo = "int(qmv_tid.y) * \(2 * entry.rps) + int(qmv_sgid) * \(entry.rps)"
            source = replaceRowExpression(source, case: entry.m, from: rowFrom, to: rowTo)
        }
        if let tier {
            source = dropCases(source, keeping: cases.filter { $0.ipg == tier }.map(\.m))
        }
        return MLXFast.metalKernel(
            name: "e129_qmv_sums_\(label)\(tier.map { "_na\($0)" } ?? "")_\(caseTag(cases))",
            inputNames: ["w", "scales", "biases", "x", "xsums"],
            outputNames: ["y"],
            source: source,
            header: Qwen35CustomQMV.generatedHeader,
            ensureRowContiguous: true)
    }

    /// Removes every `case M:` block the tier does not carry, so the compiler
    /// inlines only the bodies this entry point can reach.
    private static func dropCases(_ source: String, keeping widths: [Int]) -> String {
        var out = source
        for entry in sourcePlan where !widths.contains(entry.m) {
            let head = "        case \(entry.m):\n"
            guard let start = out.range(of: head),
                let end = out.range(of: "break;\n", range: start.upperBound ..< out.endIndex)
            else { preconditionFailure("case \(entry.m) not found in generated source") }
            out = out.replacingCharacters(in: start.lowerBound ..< end.upperBound, with: "")
        }
        return out
    }

    /// `out_row` is written once per case, so replace only the occurrence that
    /// follows this case's `qwen_e120_qmv_m<...>` instantiation.
    private static func replaceRowExpression(
        _ source: String, case m: Int, from: String, to: String
    ) -> String {
        guard from != to,
            let anchor = source.range(of: "qwen_e120_qmv_m<\(m), "),
            let hit = source.range(of: from, range: anchor.upperBound ..< source.endIndex)
        else { return source }
        return source.replacingCharacters(in: hit, with: to)
    }

    /// `a_compare` and `b_shipped`. `MLXFAST_E129_TIERED` names the arms whose
    /// entry point is templated per `ipg`. The default tiers the compare arm
    /// and leaves the shipped arm on the shared switch, so with no case
    /// override the two arms run identical bodies and differ in the entry
    /// point's register maximum alone.
    fileprivate static var armPair: (compare: Arm, shipped: Arm) {
        let tiered = ProcessInfo.processInfo.environment["MLXFAST_E129_TIERED"] ?? "compare"
        return (
            Arm(compareCases, tiered: tiered.contains("compare"), label: "compare"),
            Arm(shippedCases, tiered: tiered.contains("shipped"), label: "shipped")
        )
    }

    @Test(
        "the width-table IPG choice changes no output element at any routed width",
        .enabled(if: E120CustomQMVProbeTests.runtimeEnabled))
    func m5IPGExactness() throws {
        var records: [[String: Any]] = []
        let (compare, shipped) = Self.armPair

        for shape in Self.shapes + [("control.small", 512, 4096)] {
            let w = Self.makeWeights(
                hidden: shape.hidden, outputs: shape.outputs,
                seed: UInt64(0xE120) &+ UInt64(shape.outputs))
            MLXRandom.seed(UInt64(0xB1F5) &+ UInt64(shape.outputs))
            let block = MLXRandom.normal([9, shape.hidden]).asType(.bfloat16)
            eval(block)

            for width in 3 ... 9 {
                let x = contiguous(block[0 ..< width])
                eval(x)
                let reference = Self.mlx(x, w)
                let table = Qwen35CustomQMV.xsumsTable(x)
                eval(reference, table)

                // `x_hit` control: one activation moved by half a unit must
                // change the output, or the comparison cannot fail.
                let bumped = x + MLXArray(Float(0.5)).asType(.bfloat16)
                eval(bumped)
                let bumpedTable = Qwen35CustomQMV.xsumsTable(bumped)
                eval(bumpedTable)

                let a = shipped.apply(x, w, table)
                let b = compare.apply(x, w, table)
                let hit = compare.apply(bumped, w, bumpedTable)
                eval(a, b, hit)

                let shippedVsMLX = Self.mismatches(reference, a)
                let compareVsMLX = Self.mismatches(reference, b)
                let crossDiffering = Self.mismatches(a, b)
                let xHit = Self.mismatches(reference, hit)
                records.append([
                    "shape": shape.name,
                    "outputs": shape.outputs,
                    "hidden": shape.hidden,
                    "width": width,
                    "elements": reference.size,
                    "cases_shipped": Self.caseTag(shipped.plan),
                    "cases_compare": Self.caseTag(compare.plan),
                    "tiered_shipped": shipped.tiered,
                    "tiered_compare": compare.tiered,
                    "pipelines_shipped": shipped.pipelineCount,
                    "pipelines_compare": compare.pipelineCount,
                    "passes_shipped": Self.passes(shipped.plan)["\(width)"] ?? 0,
                    "passes_compare": Self.passes(compare.plan)["\(width)"] ?? 0,
                    "shipped_vs_mlx": shippedVsMLX,
                    "compare_vs_mlx": compareVsMLX,
                    "differing_elements": crossDiffering,
                    "max_abs_diff": Self.maxAbsDiff(a, b),
                    "bit_exact": crossDiffering == 0,
                    "x_hit": xHit,
                    "positive_control_can_fail": xHit > 0,
                ])
                #expect(shippedVsMLX == 0)
                #expect(compareVsMLX == 0)
                #expect(crossDiffering == 0)
                #expect(xHit > 0)
                print("E129_M5IPG_EXACT " + e120JSON(records[records.count - 1]))
                fflush(stdout)
            }
        }

        if let path = ProcessInfo.processInfo.environment["MLXFAST_E129_M5IPG_EXACT_OUT"],
            !path.isEmpty
        {
            let data = try JSONSerialization.data(
                withJSONObject: ["records": records], options: [.prettyPrinted, .sortedKeys])
            try data.write(to: URL(fileURLWithPath: path))
        }
        #expect(!records.isEmpty)
    }

    @Test(
        "matched ABBA timing of the width-table IPG choice at every routed width",
        .enabled(if: E120CustomQMVProbeTests.timingEnabled))
    func m5IPGTiming() throws {
        var cells: [[String: Any]] = []
        let (compare, shipped) = Self.armPair

        for shape in Self.shapes {
            let w = Self.makeWeights(
                hidden: shape.hidden, outputs: shape.outputs,
                seed: UInt64(0xE120) &+ UInt64(shape.outputs))
            let maxWidth = Self.widths.max() ?? 9
            MLXRandom.seed(UInt64(0xB1F5) &+ UInt64(shape.outputs))
            let block = MLXRandom.normal([maxWidth, shape.hidden]).asType(.bfloat16)
            eval(block)

            for width in Self.widths {
                let x = contiguous(block[0 ..< width])
                eval(x)
                let table = Qwen35CustomQMV.xsumsTable(x)
                eval(table)

                let arms: [E120Arm] = [
                    E120Arm(
                        name: "a_compare",
                        body: { [compare.apply(x, w, table)] }),
                    E120Arm(
                        name: "b_shipped",
                        body: { [shipped.apply(x, w, table)] }),
                ]

                Self.rampBurst(arms[0].body, seconds: Self.rampSeconds)
                let probeUs = Self.timed(8, arms[0].body)
                let count = max(8, min(600, Int(Self.targetMicroseconds / max(probeUs, 1.0))))
                for arm in arms { _ = Self.timed(3, arm.body) }

                for blockIndex in 0 ..< Self.blocks {
                    let entryTemp = e120GPUTemperature()
                    Self.settle(arms, seconds: Self.rampSeconds)
                    let forward = arms.map { Self.timed(count, $0.body) }
                    let reverse = Array(
                        arms.reversed().map { Self.timed(count, $0.body) }.reversed())
                    let exitTemp = e120GPUTemperature()

                    let mean = (0 ..< arms.count).map { 0.5 * (forward[$0] + reverse[$0]) }
                    var record: [String: Any] = [
                        "shape": shape.name,
                        "outputs": shape.outputs,
                        "hidden": shape.hidden,
                        "width": width,
                        "block": blockIndex,
                        "replicates": count,
                        "cases_shipped": Self.caseTag(shipped.plan),
                        "cases_compare": Self.caseTag(compare.plan),
                        "tiered_shipped": shipped.tiered,
                        "tiered_compare": compare.tiered,
                        "pipelines_shipped": shipped.pipelineCount,
                        "pipelines_compare": compare.pipelineCount,
                        "passes_shipped": Self.passes(shipped.plan)["\(width)"] ?? 0,
                        "passes_compare": Self.passes(compare.plan)["\(width)"] ?? 0,
                        "saved_us_per_matvec": mean[0] - mean[1],
                        "saved_fraction": (mean[0] - mean[1]) / mean[0],
                        "arms": arms.enumerated().map { index, arm in
                            [
                                "arm": arm.name,
                                "forward_us": forward[index],
                                "reverse_us": reverse[index],
                            ] as [String: Any]
                        },
                    ]
                    if let entryTemp { record["gpu_temp_entry_c"] = entryTemp }
                    if let exitTemp { record["gpu_temp_exit_c"] = exitTemp }
                    cells.append(record)
                    print("E129_M5IPG_BLOCK " + e120JSON(record))
                    fflush(stdout)
                }
                eval(MLXArray(0))
            }
            Memory.clearCache()
        }

        if let path = ProcessInfo.processInfo.environment["MLXFAST_E129_M5IPG_OUT"], !path.isEmpty {
            let payload: [String: Any] = [
                "group_size": Self.groupSize,
                "bits": Self.bits,
                "ramp_seconds": Self.rampSeconds,
                "target_us": Self.targetMicroseconds,
                "cool_gate_passed_real_gate": false,
                "gate_qualified_for_timing": false,
                "blocks": Self.blocks,
                "widths": Self.widths,
                "cases_shipped": Self.caseTag(shipped.plan),
                "cases_compare": Self.caseTag(compare.plan),
                "passes_shipped": Self.passes(shipped.plan),
                "passes_compare": Self.passes(compare.plan),
                "tiered_shipped": shipped.tiered,
                "tiered_compare": compare.tiered,
                "pipelines_shipped": shipped.pipelineCount,
                "pipelines_compare": compare.pipelineCount,
                "cells": cells,
            ]
            let data = try JSONSerialization.data(
                withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
            try data.write(to: URL(fileURLWithPath: path))
        }
        #expect(!cells.isEmpty)
    }
}

private func e120JSON(_ object: [String: Any]) -> String {
    guard let data = try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
    else { return "{}" }
    return String(decoding: data, as: UTF8.self)
}

/// One macmon sample. The probe runs under no thermal gate, so the entry and
/// exit temperature of every block is the thermal record.
private func e120GPUTemperature() -> Double? {
    let binary =
        ProcessInfo.processInfo.environment["MLXFAST_E120_MACMON"]
        ?? "/opt/homebrew/bin/macmon"
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
