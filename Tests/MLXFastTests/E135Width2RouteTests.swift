import Foundation
import MLX
import MLXLLM
import MLXRandom
import Testing

// E135 / advisor F18 -- route verify width 2 to the candidate QMV.
//
// `Qwen35CustomQMV.widths` was `3 ... 9`. Width 2 fell through `default: break`
// and MLX served it with the library pair kernel, whose host launches `M = 2`
// x-groups even though the kernel body returns at once for every group whose
// `first_m` reaches `M`. Routing the width hands the cell to the same entry
// point every other width already uses and lets `Grid.tight` launch the
// `ceil(2 / 2) = 1` column that does the work.
//
// Width 2 is a *routing* change, not a launcher change: it also swaps the
// library kernel body for ours at that width. The gates below are therefore
// split into the geometry claim and the arithmetic claim, and the arithmetic
// claim is checked against the library kernel the route displaces.
//
//  1. `widthTwoIsRoutedInEveryTable` and `widthTwoBuildsALegalEntryPoint` pin
//     the table entry and prove the Metal `static_assert(M % IPG != 1)` holds.
//  2. `tierTwoCarriesOnlyWidthTwo` is the static half of the Rule 82 register
//     argument: an entry point is allocated for the widest body inlined into
//     it, so tiers 3, 4 and 5 can only change if width 2 reaches them. It does
//     not. The measured half is `senpai/entry-point-cliff-census.sh`, which the
//     gate chain runs against the same base.
//  3. `widthTwoMatchesTheLibraryKernelOnTheScoredShapes` compares our m = 2
//     output with `quantizedMM` element for element on the scored shapes, and
//     its positive control perturbs one activation so the same comparison must
//     fail.
//
// The runtime gate needs `MLXFAST_RUN_MLX_RUNTIME_TESTS=1`. `Tests/` is never
// packaged into a submission.

private struct E135W2SplitMix64 {
    var state: UInt64
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

@Suite("E135 width 2 routing")
struct E135Width2RouteTests {
    static let runtimeEnabled =
        ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1"

    static let groupSize = 64
    static let bits = 4
    static let width = 2

    /// The seven shapes of one decode round. `lm_head` is excluded from the
    /// runtime gate for the same reason as in `E135TightLaunchGridTests`: its
    /// 636 MB of packed weights buy no argument the other six do not make.
    static let roundShapes: [(name: String, k: Int, n: Int)] = [
        ("gdn.in_proj", 5120, 16480),
        ("fa.qkv", 5120, 14336),
        ("mlp.gate_up", 5120, 34816),
        ("lm_head", 5120, 248_320),
        ("gdn.out_proj", 6144, 5120),
        ("fa.o_proj", 6144, 5120),
        ("mlp.down", 17408, 5120),
    ]

    static var runtimeShapes: [(name: String, k: Int, n: Int)] {
        guard ProcessInfo.processInfo.environment["MLXFAST_E135_ALL_SHAPES"] == "1"
        else { return roundShapes.filter { $0.name != "lm_head" } }
        return roundShapes
    }

    static func mismatches(_ a: MLXArray, _ b: MLXArray) -> Int {
        (a .!= b).asType(.int32).sum().item(Int.self)
    }

    // MARK: the table entry

    @Test("width 2 is routed in every table at ipg 2 and rps 4")
    func widthTwoIsRoutedInEveryTable() throws {
        #expect(Qwen35CustomQMV.widths.lowerBound == 2)
        #expect(Qwen35CustomQMV.widths.upperBound == 9)

        for table in Qwen35CustomQMV.Table.allCases {
            let entry = try #require(table.plan.first { $0.m == 2 },
                                     "\(table.rawValue) has no width-2 entry")
            #expect(entry.ipg == 2)
            #expect(entry.rps == 4)
            #expect(table.plan.first?.m == 2, "width 2 must lead the plan")
            #expect(table.witness.contains("2:2:4,"))
            #expect(table.plan.map(\.m) == Array(Qwen35CustomQMV.widths))
        }

        #expect(Qwen35CustomQMV.tier(m: 2) == 2)
        #expect(Qwen35CustomQMV.tiers.contains(2))
        if ProcessInfo.processInfo.environment["MLX_E120_QMV_TABLE"] == nil {
            #expect(Qwen35CustomQMV.tiers == [2, 3, 4, 5])
        }
    }

    /// `qwen_e120_qmv_m` asserts `M % IPG != 1`, because a one-input tail group
    /// has no built body. Width 2 on tier 2 has `TAIL = 0`, so it takes the
    /// full-group branch `qwen_e120_qmv_wide<2, 4, USE_TABLE>` and never needs
    /// a tail specialization at all.
    @Test("width 2 satisfies the kernel's own tail assertion")
    func widthTwoBuildsALegalEntryPoint() throws {
        for table in Qwen35CustomQMV.Table.allCases {
            for entry in table.plan {
                #expect(entry.m % entry.ipg != 1,
                        "\(table.rawValue) M=\(entry.m) builds a one-input tail")
                #expect(entry.ipg <= entry.m)
            }
        }
        #expect(2 % Qwen35CustomQMV.tier(m: 2) == 0)
    }

    @Test("width 2 keeps the width-independent host constants")
    func widthTwoKeepsTheHostConstants() throws {
        // `minimumTableWidth` is 4, so width 2 takes the replica arm and never
        // asks for a chunk-sum table.
        #expect(!Qwen35CustomQMV.tablePays(m: 2))
        // The kernel reads `qmv_stride = qmv_m <= 8 ? 8 : 16`; the host copy
        // must agree at the new width.
        #expect(Qwen35CustomQMV.sumsStride(2) == 8)
    }

    // MARK: launch geometry

    @Test("width 2 launches one tight column and two wide columns")
    func widthTwoLaunchesOneTightColumn() throws {
        for n in [5120, 14336, 16480, 34816, 248_320] {
            let tight = Qwen35CustomQMV.launch(m: 2, n: n, using: .tight)
            let wide = Qwen35CustomQMV.launch(m: 2, n: n, using: .wide)
            #expect(tight.grid.0 == 32, "one 32-wide column")
            #expect(wide.grid.0 == 64, "the library's two columns")
            #expect(tight.grid.1 == n / 4)
            #expect(tight.grid.1 == wide.grid.1)
            #expect(tight.threadGroup == wide.threadGroup)
        }
    }

    /// Under `.shipped` plus width 2 the whole routed set now launches
    /// `{2:1, 3:1, 4:1, 5:1, 6:2, 7:2, 8:2, 9:3}` columns. The pipeline log
    /// census in `research/e135_columns_check.py` reads the same numbers off a
    /// real leg; this pins the arithmetic they are compared against.
    @Test("the shipped column census gains one single-column width")
    func shippedColumnCensusGainsWidthTwo() throws {
        var census: [Int: Int] = [:]
        for entry in Qwen35CustomQMV.Table.shipped.plan {
            census[entry.m] = (entry.m + entry.ipg - 1) / entry.ipg
        }
        #expect(census == [2: 1, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3])
    }

    // MARK: tier isolation -- the static half of the Rule 82 argument

    @Test("tier 2 carries width 2 and no other tier mentions it")
    func tierTwoCarriesOnlyWidthTwo() throws {
        for useTable in [true, false] {
            let tierTwo = Qwen35CustomQMV.generatedSource(table: useTable, tier: 2)
            #expect(tierTwo.contains("case 2:"))
            for m in 3 ... 9 {
                #expect(!tierTwo.contains("case \(m):"),
                        "tier 2 also carries width \(m)")
            }

            for tier in Qwen35CustomQMV.tiers where tier != 2 {
                let other = Qwen35CustomQMV.generatedSource(table: useTable, tier: tier)
                #expect(!other.contains("case 2:"),
                        "tier \(tier) gained width 2 and may reallocate registers")
            }

            // The shared switch is the union and must gain the case.
            #expect(Qwen35CustomQMV.generatedSource(table: useTable, tier: nil)
                .contains("case 2:"))
        }
    }

    /// `qwen35E120QMVName` is a total switch over a hand-written list and traps
    /// on a tier it does not name. The list did not carry tier 2, so before
    /// this experiment added it the first width-2 verify would have killed the
    /// worker. No local leg reaches width 2 on the public fixture, so only this
    /// test can find that class of defect. It asks for the name of every tier
    /// of every table, not just the compiled one.
    @Test("every tier of every table has a pipeline name, and they are distinct")
    func everyTierOfEveryTableHasAPipelineName() throws {
        var names: Set<String> = []
        var tiers: Set<Int> = []
        for table in Qwen35CustomQMV.Table.allCases {
            tiers.formUnion(table.plan.map(\.ipg))
        }
        #expect(tiers.contains(2))

        for tier in [nil] + tiers.sorted().map(Optional.init) {
            for useTable in [true, false] {
                let name = Qwen35CustomQMV.pipelineName(useTable: useTable, tier: tier)
                #expect(!name.isEmpty)
                #expect(!names.contains(name), "duplicate entry point \(name)")
                names.insert(name)
            }
        }
        #expect(Qwen35CustomQMV.pipelineName(useTable: false, tier: 2)
            == "qwen35_custom_affine4_g64_qmv_wide_na2_v2")
        #expect(Qwen35CustomQMV.pipelineName(useTable: true, tier: 2)
            == "qwen35_custom_affine4_g64_qmv_wide_sums_na2_v2")
    }

    // MARK: the numeric guards `routable` applies

    /// `routable` rejects a cell unless `k % 512 == 0`, `n % 8 == 0` and
    /// `n >= 4096`. None of those depend on `m`, so every scored shape that
    /// routes at width 3 also routes at width 2. This states it per shape so a
    /// future shape that does not qualify is named.
    @Test("every scored shape clears the width-independent routing guards")
    func scoredShapesClearTheRoutingGuards() throws {
        for shape in Self.roundShapes {
            #expect(shape.k % 512 == 0, "\(shape.name) k")
            #expect(shape.n % 8 == 0, "\(shape.name) n")
            #expect(shape.n >= 4096, "\(shape.name) out_vec_size")
            #expect(Qwen35CustomQMV.widths.contains(2), "\(shape.name) width")
        }
    }

    // MARK: arithmetic -- our m = 2 body against the library body it displaces

    @Test(
        "width 2 is bit exact against quantizedMM on the scored shapes",
        .enabled(if: E135Width2RouteTests.runtimeEnabled))
    func widthTwoMatchesTheLibraryKernelOnTheScoredShapes() throws {
        var controlFired = false
        var routedCells = 0

        for shape in Self.runtimeShapes {
            var rng = E135W2SplitMix64(state: 0xE135_0002 &+ UInt64(shape.n))
            var raw = [UInt32]()
            raw.reserveCapacity(shape.n * shape.k / 8)
            for _ in 0 ..< (shape.n * shape.k / 8) {
                raw.append(UInt32(truncatingIfNeeded: rng.next()))
            }
            let packed = MLXArray(raw).reshaped([shape.n, shape.k / 8])
            MLXRandom.seed(0xE135_0002 &+ UInt64(shape.n))
            let scales = MLXRandom.uniform(
                low: Float(0.004), high: Float(0.02),
                [shape.n, shape.k / Self.groupSize]
            ).asType(.bfloat16)
            let biases = MLXRandom.uniform(
                low: Float(-0.06), high: Float(0.06),
                [shape.n, shape.k / Self.groupSize]
            ).asType(.bfloat16)
            let x = MLXRandom.normal([Self.width, shape.k]).asType(.bfloat16)
            eval(packed, scales, biases, x)

            // The library body this route displaces.
            let reference = quantizedMM(
                x, packed, scales: scales, biases: biases, transpose: true,
                groupSize: Self.groupSize, bits: Self.bits, mode: .affine)
            eval(reference)
            #expect(reference.dim(-2) == Self.width)

            // `tablePays(m: 2)` is false, so `.sumTable` reaches the same
            // replica dispatch as `.replica`. Both are run anyway: the arm
            // selector must not be able to send width 2 somewhere else.
            for arm in [Qwen35CustomQMV.Arm.replica, .sumTable] {
                let ours = try #require(
                    Qwen35CustomQMV.matmul(
                        x, packed, scales: scales, biases: biases,
                        groupSize: Self.groupSize, bits: Self.bits, mode: .affine,
                        arm: arm),
                    "\(shape.name) width 2 is not routed on arm \(arm.rawValue)")
                eval(ours)
                #expect(
                    Self.mismatches(reference, ours) == 0,
                    "\(shape.name) width 2 arm=\(arm.rawValue) differs from quantizedMM")
                routedCells += 1

                // Rule 101 positive control. One activation moved by half a
                // unit must break the same comparison, or a route that
                // returned the reference unchanged would pass.
                if !controlFired {
                    let bumped = x + MLXArray(Float(0.5)).asType(.bfloat16)
                    eval(bumped)
                    let hit = try #require(
                        Qwen35CustomQMV.matmul(
                            bumped, packed, scales: scales, biases: biases,
                            groupSize: Self.groupSize, bits: Self.bits,
                            mode: .affine, arm: arm))
                    eval(hit)
                    #expect(Self.mismatches(reference, hit) > 0,
                            "positive control could not fail")
                    controlFired = true
                }
            }
        }

        #expect(controlFired)
        #expect(routedCells == Self.runtimeShapes.count * 2)
    }

    /// Both launch grids must agree at the new width for the same reason they
    /// agree at every other width: the columns `.tight` removes write nothing.
    @Test(
        "width 2 is grid invariant",
        .enabled(if: E135Width2RouteTests.runtimeEnabled))
    func widthTwoIsGridInvariant() throws {
        let k = 5120
        let n = 16480
        var rng = E135W2SplitMix64(state: 0xE135_0003)
        var raw = [UInt32]()
        raw.reserveCapacity(n * k / 8)
        for _ in 0 ..< (n * k / 8) { raw.append(UInt32(truncatingIfNeeded: rng.next())) }
        let packed = MLXArray(raw).reshaped([n, k / 8])
        MLXRandom.seed(0xE135_0003)
        let scales = MLXRandom.uniform(
            low: Float(0.004), high: Float(0.02), [n, k / Self.groupSize]
        ).asType(.bfloat16)
        let biases = MLXRandom.uniform(
            low: Float(-0.06), high: Float(0.06), [n, k / Self.groupSize]
        ).asType(.bfloat16)
        let x = MLXRandom.normal([Self.width, k]).asType(.bfloat16)
        eval(packed, scales, biases, x)

        for arm in [Qwen35CustomQMV.Arm.replica, .sumTable] {
            let tight = try #require(
                Qwen35CustomQMV.matmul(
                    x, packed, scales: scales, biases: biases,
                    groupSize: Self.groupSize, bits: Self.bits, mode: .affine,
                    arm: arm, using: .tight))
            let wide = try #require(
                Qwen35CustomQMV.matmul(
                    x, packed, scales: scales, biases: biases,
                    groupSize: Self.groupSize, bits: Self.bits, mode: .affine,
                    arm: arm, using: .wide))
            eval(tight, wide)
            #expect(Self.mismatches(tight, wide) == 0, "arm=\(arm.rawValue)")
        }
    }
}
