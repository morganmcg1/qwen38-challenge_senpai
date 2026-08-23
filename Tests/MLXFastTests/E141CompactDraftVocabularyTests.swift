import Foundation
import MLX
import MLXLLM
import Testing

/// E141: the compact draft vocabulary cannot propose one target token in two
/// hundred, and widening it needs no new shipped data.
///
/// Two properties are checked here.
///
/// **The count family.** One prefix bound drives the control block, the real
/// count and the padded count (`Qwen35.swift:4224-4239`). Widening the prefix
/// must never duplicate a row, must keep the padded count a multiple of eight,
/// and must collapse the appended control block once the prefix already covers
/// it. This part needs no GPU and no weights.
///
/// **The licensing gate, rung 1.** `Qwen35.swift:5940-5943` claims the shipped
/// `draft_lm_head.*` is exactly `quantize(dequantize(exact compact lm_head,
/// 64, 4), 64, 2)`. If that holds bit for bit, the coarse shortlist table is a
/// pure function of the target's own fixed lm_head, so
/// `deriveCompactCoarseTable` can produce it at ANY prefix with zero shipped
/// bytes and no head re-declaration. This part reads the two artifacts the
/// process already loads and applies the same MLX ops the derivation calls, so
/// enable it with `MLXFAST_RUN_MLX_RUNTIME_TESTS=1`.
@Suite
struct E141CompactDraftVocabularyTests {
    private static let vocabulary = 248_320
    private static let shippedPrefix = 98_304
    private static let controlStart = 248_044
    private static let controlEnd = 248_070

    private static var runtimeEnabled: Bool {
        ProcessInfo.processInfo
            .environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1"
    }

    /// `FileManager.homeDirectoryForCurrentUser` reads the passwd database, so
    /// it misses the role-scoped `HOME` this campaign's caches live under.
    private static var home: URL {
        guard let path = ProcessInfo.processInfo.environment["HOME"] else {
            return FileManager.default.homeDirectoryForCurrentUser
        }
        return URL(fileURLWithPath: path)
    }

    /// The test bundle's working directory is not the checkout, so resolve the
    /// weights against this source file instead.
    private static var repositoryRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    /// Candidate bounds the census names, plus both corners of the control
    /// block, which are the only places the collapse rule can be wrong.
    private static let bounds = [
        1_024, 98_304, 114_688, 131_072, 147_456, 163_840, 196_608,
        248_043, 248_044, 248_050, 248_069, 248_070, 248_320,
    ]

    @Test
    func theShippedBoundReproducesTheShippedCounts() {
        let counts = qwen35CompactDraftCounts(prefix: Self.shippedPrefix)
        #expect(counts.controlStart == Self.controlStart)
        #expect(counts.controlEnd == Self.controlEnd)
        #expect(counts.real == 98_330)
        #expect(counts.padded == 98_336)
    }

    /// The default build must stay the shipped arm. A leg that exports nothing
    /// reads `source=unset`, which is the bare-leg proof that widening is
    /// opt-in rather than the compiled default.
    @Test
    func theCompiledDefaultIsTheShippedArm() {
        let live = qwen35CompactDraftGeometry
        guard live.source == "unset" else { return }
        #expect(live.prefix == Self.shippedPrefix)
        #expect(live.real == 98_330)
        #expect(live.padded == 98_336)
    }

    /// Arm A pins the absolute probe count so only the centroid pass grows.
    /// The byte table and the `plan.perThread` reported to the advisor are
    /// arithmetic over these four numbers, so pin them against the source
    /// rather than against a remembered figure.
    @Test
    func theArmGeometryMatchesTheReportedByteTable() {
        #expect(qwen35E141ProbeCountOverride == nil,
                "a test process must not carry MLX_E141_PROBES")

        let rowsPerLeaf = 8
        let stride = 32 * 256  // qwen35RowTop32Tiles * qwen35Top32TG
        func probes(_ padded: Int, _ fraction: Double) -> Int {
            max(1, Int((fraction * Double(padded / rowsPerLeaf)).rounded(.up)))
        }
        func perThread(_ rows: Int) -> Int { (rows + stride - 1) / stride }

        let shipped = qwen35CompactDraftCounts(prefix: Self.shippedPrefix).padded
        let full = qwen35CompactDraftCounts(prefix: Self.vocabulary).padded
        #expect(full == Self.vocabulary)

        #expect(probes(shipped, 0.25) == 3_073)
        #expect(probes(shipped, 0.15) == 1_844)
        #expect(probes(full, 0.25) == 7_760)

        // Holding the probe count keeps the row pass and therefore perThread.
        #expect(perThread(probes(shipped, 0.25) * rowsPerLeaf) == 4)
        #expect(perThread(probes(full, 0.25) * rowsPerLeaf) == 8)
        #expect(perThread(probes(shipped, 0.15) * rowsPerLeaf) == 2)
        #expect(perThread(3_073 * rowsPerLeaf) <= 32)
    }

    /// Arm B widens the leaf instead of the probe list, so the leaf count and
    /// the centroid-pass bytes stay near today's. The width must divide the
    /// padded row count exactly, which is why it has to stay arm-scoped: the
    /// shipped 98,336 rows do not divide by 20.
    @Test
    func theWidenedLeafOnlyDividesTheArmsThatUseIt() {
        #expect(qwen35E141RowsPerLeafOverride == nil,
                "a test process must not carry MLX_E141_ROWS_PER_LEAF")

        let shipped = qwen35CompactDraftCounts(prefix: Self.shippedPrefix).padded
        let full = qwen35CompactDraftCounts(prefix: Self.vocabulary).padded
        #expect(shipped % 20 != 0, "the shipped table must not take a 20-row leaf")
        #expect(full % 16 == 0)
        #expect(full % 20 == 0)

        // The reported arm B byte table: leaves x 1600 B centroid pass plus
        // probes x rowsPerLeaf x 1600 B row pass, against a shipped
        // 12,292-leaf / 24,584-row readout.
        let shippedLeaves = shipped / 8
        let shippedRows = 3_073 * 8
        #expect(shippedLeaves == 12_292)
        #expect(shippedRows == 24_584)
        for (rowsPerLeaf, probes) in [(8, 3_073), (16, 1_536), (20, 1_229)] {
            let leaves = full / rowsPerLeaf
            let deltaMB =
                Double((leaves - shippedLeaves) + (probes * rowsPerLeaf - shippedRows))
                * 1600 / 1_000_000
            switch rowsPerLeaf {
            case 8: #expect(leaves == 31_040 && abs(deltaMB - 30.00) < 0.01)
            case 16: #expect(leaves == 15_520 && abs(deltaMB - 5.15) < 0.01)
            default: #expect(leaves == 12_416 && abs(deltaMB - 0.19) < 0.01)
            }
            #expect(probes >= 1 && probes <= leaves)
            #expect(probes * rowsPerLeaf > 32, "the shortlist must exceed the rerank width")
        }
    }

    /// The generalised leaf QMV must agree with the `gatherQuantizedMM` it
    /// replaces at every leaf width an arm can ask for, and no worse than the
    /// shipped eight-row width already does.
    @Test
    func theGeneralisedLeafQMVMatchesTheGatherItReplaces() {
        guard Self.runtimeEnabled else { return }
        let clusters = 512, probes = 64
        for rowsPerCluster in [8, 16, 20] {
            let got = qwen35VerifyClusterRowQMV(
                clusters: clusters, rowsPerCluster: rowsPerCluster, probes: probes)
            #expect(got.routed, "leaf width \(rowsPerCluster) did not reach the kernel")
            #expect(got.maxAbsDiff == 0,
                    "leaf width \(rowsPerCluster) drifts \(got.maxAbsDiff) from the gather")

            // Rule 101: the same comparison over damaged weights must move the
            // fused scores. That both proves the inputs reach the kernel and
            // demonstrates the difference is not structurally pinned at zero.
            let damaged = qwen35VerifyClusterRowQMV(
                clusters: clusters, rowsPerCluster: rowsPerCluster, probes: probes,
                damageRow: true)
            #expect(damaged.checksum != got.checksum,
                    "the damaged control did not move the fused scores")
            #expect(damaged.maxAbsDiff <= 1e-3,
                    "damaged leaf width \(rowsPerCluster) disagrees by \(damaged.maxAbsDiff)")
        }

        // A width the dispatch must refuse rather than mis-address.
        #expect(!qwen35VerifyClusterRowQMV(
            clusters: clusters, rowsPerCluster: 6, probes: probes).routed)
    }

    /// The fused row top-32 address arithmetic already threads the leaf width,
    /// so arm B must not need a second selection path.
    @Test
    func theRowTop32SelectionFollowsTheWidenedLeaf() {
        guard Self.runtimeEnabled else { return }
        for (rowsPerCluster, clusters, probes) in
            [(8, 12_292, 3_073), (20, 12_416, 1_229)]
        {
            let (checked, bad, firstBad) = qwen35VerifyRowTop32(
                clusters: clusters, rowsPerCluster: rowsPerCluster, probes: probes,
                trials: 8)
            #expect(checked == 8)
            #expect(bad == 0, "row top-32 mismatch at leaf width \(rowsPerCluster), first bad trial \(firstBad)")
        }
        #expect(qwen35RowTop32PositiveControl(
            clusters: 12_416, rowsPerCluster: 20, probes: 1_229))
    }

    @Test
    func everyBoundKeepsTheLeafGroupingAndTheRowAccounting() {
        for prefix in Self.bounds {
            let c = qwen35CompactDraftCounts(prefix: prefix)
            #expect(c.padded % 8 == 0, "padded count not a leaf multiple at \(prefix)")
            #expect(c.padded >= c.real, "padding is negative at \(prefix)")
            #expect(c.padded - c.real < 8, "over-padded at \(prefix)")
            #expect(c.real <= Self.vocabulary, "real count exceeds the vocabulary at \(prefix)")
            #expect(c.controlEnd >= c.controlStart, "inverted control block at \(prefix)")
            #expect(c.real == prefix + c.controlEnd - c.controlStart)
        }
    }

    /// The appended control block must never repeat a prefix row, and it must
    /// vanish once the prefix reaches it. A duplicated row that survives to the
    /// argmax is a proposal for the wrong token.
    @Test
    func theControlBlockNeverDuplicatesAPrefixRow() {
        for prefix in Self.bounds {
            let c = qwen35CompactDraftCounts(prefix: prefix)
            #expect(c.controlStart >= prefix, "control block overlaps the prefix at \(prefix)")
            if prefix >= Self.controlEnd {
                #expect(c.controlEnd == c.controlStart, "control block survives at \(prefix)")
                #expect(c.real == prefix, "real count carries a stale block at \(prefix)")
            }
        }
        // The full-vocabulary arm is the one rung 3 measures: every id
        // proposable, no control block, and no padding row at all.
        let full = qwen35CompactDraftCounts(prefix: Self.vocabulary)
        #expect(full.real == Self.vocabulary)
        #expect(full.padded == Self.vocabulary)
    }

    /// Compact-to-full id mapping, as `mapDraftTokenIds` computes it. Every
    /// compact row must map to a distinct, in-range target id.
    @Test
    func theIdMappingIsInjectiveAndInRangeAtEveryBound() {
        for prefix in Self.bounds {
            let c = qwen35CompactDraftCounts(prefix: prefix)
            let offset = c.controlStart - prefix
            var seen = Set<Int>()
            for id in 0 ..< c.real {
                let mapped = id < prefix ? id : id + offset
                #expect(mapped >= 0 && mapped < Self.vocabulary,
                        "id \(id) maps out of range at prefix \(prefix)")
                #expect(seen.insert(mapped).inserted,
                        "id \(id) collides at prefix \(prefix)")
            }
            #expect(seen.count == c.real)
        }
    }

    /// Rung 1. The declared coarse table is `quantize(dequantize(exact compact
    /// lm_head, 64, 4), 64, 2)` bit for bit, so deriving it at a wider prefix
    /// creates no parameter and ships no bytes.
    @Test
    func theDeclaredCoarseTableIsAPureFunctionOfTheTargetLMHead() throws {
        guard Self.runtimeEnabled else { return }
        let targetPath = ProcessInfo.processInfo
            .environment["MLXFAST_E141_TARGET_SHARD"]
            ?? Self.repositoryRoot
            .appendingPathComponent("weights/model-00003-of-00003.safetensors").path
        let headPath = ProcessInfo.processInfo
            .environment["MLXFAST_E141_DECLARED_HEAD"]
            ?? Self.home.appendingPathComponent(
                ".cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head-declared/model.safetensors"
            ).path
        // A silently skipped licensing gate reads exactly like a passing one,
        // so demand the artifacts once the runtime flag asks for the check.
        try #require(FileManager.default.fileExists(atPath: targetPath),
                     "missing target shard \(targetPath)")
        try #require(FileManager.default.fileExists(atPath: headPath),
                     "missing declared head \(headPath)")

        let target = try MLX.loadArrays(url: URL(fileURLWithPath: targetPath))
        let head = try MLX.loadArrays(url: URL(fileURLWithPath: headPath))
        let counts = qwen35CompactDraftCounts(prefix: Self.shippedPrefix)

        func compactRows(_ a: MLXArray) -> MLXArray {
            concatenated(
                [
                    a[0 ..< Self.shippedPrefix],
                    a[counts.controlStart ..< counts.controlEnd],
                    a[0 ..< (counts.padded - counts.real)],
                ], axis: 0)
        }

        let exactWeight = compactRows(try #require(target["language_model.lm_head.weight"]))
        let exactScales = compactRows(try #require(target["language_model.lm_head.scales"]))
        let exactBiases = compactRows(try #require(target["language_model.lm_head.biases"]))
        let declaredWeight = try #require(head["draft_lm_head.weight"])
        let declaredScales = try #require(head["draft_lm_head.scales"])
        let declaredBiases = try #require(head["draft_lm_head.biases"])

        func derive(_ scales: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
            let rows = dequantized(
                exactWeight, scales: scales, biases: exactBiases,
                groupSize: 64, bits: 4, mode: .affine)
            let coarse = quantized(rows, groupSize: 64, bits: 2, mode: .affine)
            let biases = coarse.biases ?? MLXArray(Float(0))
            eval(coarse.wq, coarse.scales, biases)
            return (coarse.wq, coarse.scales, biases)
        }

        func differingElements(_ got: MLXArray, _ want: MLXArray) -> Int {
            #expect(got.shape == want.shape)
            #expect(got.dtype == want.dtype)
            let n = MLX.sum((got .!= want).asType(.int32))
            eval(n)
            return n.item(Int.self)
        }

        let (w, s, z) = derive(exactScales)
        #expect(differingElements(w, declaredWeight) == 0,
                "the derived 2-bit rows differ from the declared table")
        #expect(differingElements(s, declaredScales) == 0,
                "the derived 2-bit scales differ from the declared table")
        #expect(differingElements(z, declaredBiases) == 0,
                "the derived 2-bit biases differ from the declared table")

        // Rule 101 positive control: perturb one input row and require the
        // comparison to fail. A check that cannot fail is not a check.
        let perturbed = concatenated(
            [
                exactScales[0 ..< 7],
                (exactScales[7 ..< 8].asType(.float32) * 1.5)
                    .asType(exactScales.dtype),
                exactScales[8...],
            ], axis: 0)
        eval(perturbed)
        let (cw, cs, cz) = derive(perturbed)
        let controlDiffered =
            differingElements(cw, declaredWeight) > 0
            || differingElements(cs, declaredScales) > 0
            || differingElements(cz, declaredBiases) > 0
        #expect(controlDiffered, "the rung 1 comparison cannot fail")
    }
}
