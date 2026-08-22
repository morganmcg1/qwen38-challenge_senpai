import Foundation
import MLX
import MLXLLM
import MLXRandom
import Testing

// E135 -- delete the no-op threadgroups from the wide QMV launch grid.
//
// `Qwen35CustomQMV.launch` under `.wide` launches `m` threadgroup columns for a
// width-`m` cell, but the kernel gives column `c` the input rows starting at
// `c * IPG` and returns at once when that start is at or past `M`. Only
// `ceil(m / IPG)` columns ever load a weight. `.tight` launches exactly those.
//
// The claim under test is that the two settings are bit-identical, so the whole
// question is a timing question. Two gates check it:
//
//  1. `tightGridKeepsExactlyTheWorkingColumns` mechanizes the argument. It
//     proves, per table and per width, that every column `.tight` drops
//     satisfies the kernel's own early-return predicate and every column it
//     keeps does not.
//  2. `tightGridIsBitIdenticalOnTheScoredShapes` runs both grids on real
//     buffers and counts differing output elements. Its positive control
//     perturbs one input element and requires the same comparison to fail, so
//     a comparison that cannot detect a difference cannot pass this suite.
//
// The runtime gate needs `MLXFAST_RUN_MLX_RUNTIME_TESTS=1`. `Tests/` is never
// packaged into a submission.

private struct E135Weights {
    var packed: MLXArray
    var scales: MLXArray
    var biases: MLXArray
}

private struct E135SplitMix64 {
    var state: UInt64
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

@Suite("E135 tight QMV launch grid")
struct E135TightLaunchGridTests {
    static let runtimeEnabled =
        ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1"

    static let groupSize = 64
    static let bits = 4

    /// The seven shapes that make up all 257 wide QMV calls of one decode
    /// round, with the layer count each carries. `lm_head` is omitted from the
    /// runtime gate only because its 636 MB of packed weights buy no argument
    /// the other six do not already make; it is present here because the
    /// threadgroup census below is the complete round.
    static let roundShapes: [(name: String, k: Int, n: Int, layers: Int)] = [
        ("gdn.in_proj", 5120, 16480, 48),
        ("fa.qkv", 5120, 14336, 16),
        ("mlp.gate_up", 5120, 34816, 64),
        ("lm_head", 5120, 248_320, 1),
        ("gdn.out_proj", 6144, 5120, 48),
        ("fa.o_proj", 6144, 5120, 16),
        ("mlp.down", 17408, 5120, 64),
    ]

    static var runtimeShapes: [(name: String, k: Int, n: Int, layers: Int)] {
        guard
            ProcessInfo.processInfo.environment["MLXFAST_E135_ALL_SHAPES"] == "1"
        else { return roundShapes.filter { $0.name != "lm_head" } }
        return roundShapes
    }

    fileprivate static func makeWeights(k: Int, n: Int, seed: UInt64) -> E135Weights {
        var rng = E135SplitMix64(state: seed)
        var raw = [UInt32]()
        raw.reserveCapacity(n * k / 8)
        for _ in 0 ..< (n * k / 8) { raw.append(UInt32(truncatingIfNeeded: rng.next())) }
        let packed = MLXArray(raw).reshaped([n, k / 8])
        MLXRandom.seed(seed)
        let scales = MLXRandom.uniform(
            low: Float(0.004), high: Float(0.02), [n, k / groupSize]
        ).asType(.bfloat16)
        let biases = MLXRandom.uniform(
            low: Float(-0.06), high: Float(0.06), [n, k / groupSize]
        ).asType(.bfloat16)
        eval(packed, scales, biases)
        return E135Weights(packed: packed, scales: scales, biases: biases)
    }

    /// Number of output elements that differ. BF16 widens to float32 exactly,
    /// so a float32 inequality count is an exact bit comparison here.
    static func mismatches(_ a: MLXArray, _ b: MLXArray) -> Int {
        (a .!= b).asType(.int32).sum().item(Int.self)
    }

    // MARK: rung 0 -- the exactness argument, mechanized

    @Test("tight keeps exactly the columns that load a weight")
    func tightGridKeepsExactlyTheWorkingColumns() throws {
        for table in Qwen35CustomQMV.Table.allCases {
            for entry in table.plan {
                let working = (entry.m + entry.ipg - 1) / entry.ipg
                // The kernel's own predicate, `Qwen35.swift` qwen_e120_qmv_m:
                // `first_m = group_x * IPG; if (first_m >= M) return;`.
                for column in 0 ..< working {
                    #expect(
                        column * entry.ipg < entry.m,
                        "\(table.rawValue) M=\(entry.m): tight keeps no-op column \(column)")
                }
                for column in working ..< entry.m {
                    #expect(
                        column * entry.ipg >= entry.m,
                        "\(table.rawValue) M=\(entry.m): tight drops working column \(column)")
                }
                #expect(working <= entry.m)
            }
        }
    }

    @Test("the two grids differ only in the x column count")
    func gridsDifferOnlyInColumnCount() throws {
        for entry in Qwen35CustomQMV.widthPlan {
            for n in [5120, 14336, 16480, 34816, 248_320] {
                let wide = Qwen35CustomQMV.launch(m: entry.m, n: n, using: .wide)
                let tight = Qwen35CustomQMV.launch(m: entry.m, n: n, using: .tight)
                #expect(wide.grid.0 == entry.m * 32)
                #expect(tight.grid.0 == ((entry.m + entry.ipg - 1) / entry.ipg) * 32)
                #expect(wide.grid.1 == tight.grid.1)
                #expect(wide.grid.2 == tight.grid.2)
                #expect(wide.threadGroup == tight.threadGroup)
                #expect(tight.grid.0 <= wide.grid.0)
            }
        }
    }

    /// The §C census the assignment asks to be reproduced from source rather
    /// than quoted. Threadgroup `y` count is `n / rps / 2`, because `launch`
    /// asks for `n / rps` threads against a threadgroup of `(32, 2, 1)`.
    @Test("one column of one round launches 519,040 threadgroups")
    func perRoundThreadgroupCensus() throws {
        var total = 0
        var layers = 0
        for shape in Self.roundShapes {
            let launch = Qwen35CustomQMV.launch(m: 6, n: shape.n, using: .tight)
            let perColumn = launch.grid.1 / launch.threadGroup.1
            #expect(perColumn == shape.n / 8, "\(shape.name) tg per column")
            total += perColumn * shape.layers
            layers += shape.layers
        }
        #expect(layers == 257)
        #expect(total == 519_040)
    }

    // MARK: rung 0 -- bit exactness on real buffers

    @Test(
        "tight and wide agree bit for bit on the scored shapes",
        .enabled(if: E135TightLaunchGridTests.runtimeEnabled))
    func tightGridIsBitIdenticalOnTheScoredShapes() throws {
        var controlFired = false
        for shape in Self.runtimeShapes {
            let w = Self.makeWeights(k: shape.k, n: shape.n, seed: 0xE135 &+ UInt64(shape.n))
            for m in Qwen35CustomQMV.widths {
                MLXRandom.seed(UInt64(m) &+ 0x135)
                let x = MLXRandom.normal([m, shape.k]).asType(.bfloat16)
                eval(x)
                for arm in [Qwen35CustomQMV.Arm.replica, .sumTable] {
                    let wide = try #require(
                        Qwen35CustomQMV.matmul(
                            x, w.packed, scales: w.scales, biases: w.biases,
                            groupSize: Self.groupSize, bits: Self.bits, mode: .affine,
                            arm: arm, using: .wide))
                    let tight = try #require(
                        Qwen35CustomQMV.matmul(
                            x, w.packed, scales: w.scales, biases: w.biases,
                            groupSize: Self.groupSize, bits: Self.bits, mode: .affine,
                            arm: arm, using: .tight))
                    eval(wide, tight)
                    #expect(
                        Self.mismatches(wide, tight) == 0,
                        "\(shape.name) M=\(m) arm=\(arm.rawValue)")

                    // Positive control. One perturbed input row must make the
                    // same comparison fail, or the comparison proves nothing.
                    if !controlFired {
                        let perturbed = x.asType(.float32)
                        perturbed[0, 0] = MLXArray(Float(1.5)) + perturbed[0, 0]
                        let other = try #require(
                            Qwen35CustomQMV.matmul(
                                perturbed.asType(.bfloat16), w.packed, scales: w.scales,
                                biases: w.biases, groupSize: Self.groupSize,
                                bits: Self.bits, mode: .affine, arm: arm, using: .tight))
                        eval(other)
                        #expect(Self.mismatches(wide, other) > 0, "positive control")
                        controlFired = true
                    }
                }
            }
        }
        #expect(controlFired)
    }
}
