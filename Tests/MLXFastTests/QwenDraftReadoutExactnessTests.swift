import Foundation
import MLX
import MLXLLM
import Testing

// E28 / PR #33 -- exactness audit of the 2-bit coarse draft readout we
// inherited at the frontier.
//
// `qmv_fast_singlerow_affine2_g64` is a hand-written arm that fires for exactly
// one shape in the scored path: affine, group_size 64, bits 2, out_vec_size
// 98336, M == 1. That is the coarse readout of the declared proposal head. It
// partitions K across 32 lanes (values_per_thread 32, rows_per_simd 4) where
// the generic `qmv_fast_impl<T, 64, 2>` partitions across 16 (values_per_thread
// 16). Source inspection says the elementary products and the bf16 quad `sum`
// used for the bias term are identical in both arms, so the ONLY difference is
// FP32 reassociation of a 5120-term dot product, landing in a bf16 output that
// carries 8 mantissa bits.
//
// Reassociation is not free of consequence, it is just bounded, so the audit
// measures it instead of asserting it. The measurement exploits the gate
// itself: the special arm keys on `out_vec_size == 98336`, so the SAME rows
// evaluated as row-slices of the head fall through to the generic arm. Each
// output row is an independent dot product, so the two arms are two orderings
// of one mathematical quantity and any difference is pure numerics.
//
// Containment matters more than the raw cell delta. The coarse readout only
// nominates 32 candidates; the emitted token is the argmax of an exact affine-4
// rerank over those candidates. An identical candidate SET therefore implies an
// identical token no matter how the coarse scores wobbled, so
// P(token changes) <= P(candidate set differs). The numeric suite reports both,
// plus the true reranked top-1 change rate when the transformed lm_head is
// available locally.

// MARK: - Shared source text helpers

enum QwenDraftReadoutSource {
    static let headerPath =
        "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h"
    /// The runtime-effective twin. The scored worker JIT-compiles the source
    /// string embedded in this C++ file, not the readable header, so a gate
    /// premise proven against one file only is proven against nothing.
    static let twinPath = "Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp"

    /// Collapse every run of whitespace to a single space so a reformat cannot
    /// silently defeat a literal match.
    static func squeeze(_ text: String) -> String {
        text.split(whereSeparator: { $0.isWhitespace }).joined(separator: " ")
    }

    static func occurrences(of needle: String, in haystack: String) -> Int {
        guard !needle.isEmpty else { return 0 }
        var count = 0
        var cursor = haystack.startIndex
        while let found = haystack.range(of: needle, range: cursor ..< haystack.endIndex) {
            count += 1
            cursor = found.upperBound
        }
        return count
    }
}

// MARK: - Gate premise (ungated: this is the permanent regression gate)

/// The numeric audit below is only meaningful while the dispatch gate keeps its
/// current shape. This suite pins that premise so the experiment stays
/// self-auditing: if someone widens, narrows or removes the 2-bit special case,
/// the recorded E28 evidence stops applying and this fails loudly rather than
/// ageing into a stale claim.
@Suite
struct QwenDraftReadoutGatePremiseTests {
    private typealias Src = QwenDraftReadoutSource

    private static let gate =
        "!batched && group_size == 64 && bits == 2 && out_vec_size == 98336 && ntg.x == 1"

    @Test
    func theTwoBitSinglerowArmIsStillGatedOnExactlyTheDeclaredHeadWidth() throws {
        for path in [Src.headerPath, Src.twinPath] {
            let text = try DFlashGateTextSupport.text(path)
            let flat = Src.squeeze(text)

            #expect(
                flat.contains(Self.gate),
                """
                E28 / PR #33: the 2-bit singlerow dispatch gate changed in \
                \(path). The recorded exactness evidence for the coarse draft \
                readout was measured against '\(Self.gate)' and does not \
                transfer to a different gate. Re-run the E28 audit before \
                relying on it.
                """
            )
            #expect(
                Src.occurrences(of: "out_vec_size == 98336", in: text) == 1,
                """
                E28 / PR #33: expected exactly one out_vec_size == 98336 \
                special case in \(path). More than one means a second bespoke \
                2-bit arm was added and is unaudited.
                """
            )
            #expect(
                Src.occurrences(of: "qmv_fast_singlerow_affine2_g64", in: text) == 2,
                """
                E28 / PR #33: expected exactly one definition and one call of \
                qmv_fast_singlerow_affine2_g64 in \(path).
                """
            )
        }
    }

    @Test
    func rowSlicesOfTheHeadStillFallThroughToTheGenericArm() throws {
        // The audit's decisive trick: because the gate keys on out_vec_size,
        // slicing the head into 98336/n row chunks reaches the generic
        // template for the same rows. That only holds while the generic
        // fallthrough exists and the special arm is the sole 2-bit exception.
        for path in [QwenDraftReadoutSource.headerPath, QwenDraftReadoutSource.twinPath] {
            let flat = QwenDraftReadoutSource.squeeze(try DFlashGateTextSupport.text(path))
            #expect(
                flat.contains("qmv_fast_impl<T, group_size, bits>"),
                """
                E28 / PR #33: the generic qmv_fast_impl fallthrough is gone \
                from \(path). The E28 differential arm depended on it.
                """
            )
        }
    }
}

// MARK: - Numeric audit

private struct QuantizedRows {
    let weight: MLXArray
    let scales: MLXArray
    let biases: MLXArray

    func rows(_ range: Range<Int>) -> QuantizedRows {
        QuantizedRows(
            weight: weight[range, axis: 0],
            scales: scales[range, axis: 0],
            biases: biases[range, axis: 0])
    }

    func gather(_ ids: MLXArray) -> QuantizedRows {
        QuantizedRows(
            weight: MLX.take(weight, ids, axis: 0),
            scales: MLX.take(scales, ids, axis: 0),
            biases: MLX.take(biases, ids, axis: 0))
    }
}

/// splitmix64. Deterministic and independent of MLXRandom's global state, so a
/// reported cell count is reproducible from the seed alone.
private struct SplitMix {
    private var state: UInt64
    init(_ seed: UInt64) { state = seed }

    mutating func next() -> UInt64 {
        state = state &+ 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    mutating func unit() -> Double { Double(next() >> 11) * (1.0 / 9_007_199_254_740_992.0) }

    mutating func normal() -> Float {
        let u = Swift.max(unit(), 1e-12)
        let v = unit()
        return Float((-2.0 * Foundation.log(u)).squareRoot() * Foundation.cos(2.0 * Double.pi * v))
    }
}

private struct TrialTotals {
    var trials = 0
    var cellsCompared = 0
    var differingCells = 0
    var maxAbsDelta: Float = 0
    var maxRelDelta: Float = 0
    var top32ChurnTrials = 0
    var top32SymmetricDifference = 0
    var coarseArgmaxChanges = 0
    var rerankTop1Changes = 0
    var rerankTrials = 0
    var chunkControlMismatches = 0
    var determinismMismatches = 0

    mutating func absorb(_ other: TrialTotals) {
        trials += other.trials
        cellsCompared += other.cellsCompared
        differingCells += other.differingCells
        maxAbsDelta = Swift.max(maxAbsDelta, other.maxAbsDelta)
        maxRelDelta = Swift.max(maxRelDelta, other.maxRelDelta)
        top32ChurnTrials += other.top32ChurnTrials
        top32SymmetricDifference += other.top32SymmetricDifference
        coarseArgmaxChanges += other.coarseArgmaxChanges
        rerankTop1Changes += other.rerankTop1Changes
        rerankTrials += other.rerankTrials
        chunkControlMismatches += other.chunkControlMismatches
        determinismMismatches += other.determinismMismatches
    }

    var payload: [String: Any] {
        [
            "trials": trials,
            "cells_compared": cellsCompared,
            "differing_cells": differingCells,
            "differing_cell_rate": cellsCompared == 0
                ? 0 : Double(differingCells) / Double(cellsCompared),
            "max_abs_delta": Double(maxAbsDelta),
            "max_rel_delta": Double(maxRelDelta),
            "top32_set_churn_trials": top32ChurnTrials,
            "top32_set_churn_rate": trials == 0 ? 0 : Double(top32ChurnTrials) / Double(trials),
            "top32_symmetric_difference": top32SymmetricDifference,
            "coarse_argmax_changes": coarseArgmaxChanges,
            "rerank_trials": rerankTrials,
            "rerank_top1_changes": rerankTop1Changes,
            "rerank_top1_change_rate": rerankTrials == 0
                ? 0 : Double(rerankTop1Changes) / Double(rerankTrials),
            "chunk_control_mismatches": chunkControlMismatches,
            "determinism_mismatches": determinismMismatches,
        ]
    }
}

/// Differential audit of the inherited 2-bit coarse readout against the generic
/// MLX arm, on the real declared proposal head.
///
/// Enable with `MLXFAST_RUN_DRAFT_READOUT_EXACTNESS=1` and point
/// `MLXFAST_DRAFT_READOUT_EXACTNESS_OUT` at the JSON destination. The head
/// defaults to the declared cache path and can be overridden with
/// `MLXFAST_QWEN_MTP_HEAD_DIR`; the transformed backbone directory defaults to
/// `weights` and can be overridden with `MLXFAST_QWEN_MTP_WEIGHTS_DIR`.
@Suite
struct QwenDraftReadoutExactnessTests {
    private static var enabled: Bool {
        ProcessInfo.processInfo.environment["MLXFAST_RUN_DRAFT_READOUT_EXACTNESS"] == "1"
    }

    private static let paddedCount = 98_336
    private static let realCount = 98_330
    private static let prefixCount = 98_304
    private static let controlOffset = 248_044 - 98_304
    private static let candidateCount = 32
    private static let hidden = 5_120

    private static var env: [String: String] { ProcessInfo.processInfo.environment }

    private static var headURL: URL {
        let dir =
            env["MLXFAST_QWEN_MTP_HEAD_DIR"]
            ?? FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent(".cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head-declared")
                .path
        return URL(fileURLWithPath: dir).appendingPathComponent("model.safetensors")
    }

    private static var backboneShardURL: URL {
        let dir = env["MLXFAST_QWEN_MTP_WEIGHTS_DIR"] ?? "weights"
        return URL(fileURLWithPath: dir)
            .appendingPathComponent("model-00003-of-00003.safetensors")
    }

    /// Compact draft id -> transformed-checkpoint vocabulary id. Mirrors
    /// `Qwen35`'s `mapDraftTokenIds`, which is what makes it legitimate to
    /// gather the exact rerank rows straight out of `language_model.lm_head`
    /// instead of materializing the compact affine-4 head.
    private static func mapDraftID(_ id: Int32) -> Int32 {
        id < Int32(prefixCount) ? id : id + Int32(controlOffset)
    }

    private static func coarse(_ x: MLXArray, _ head: QuantizedRows) -> MLXArray {
        quantizedMM(
            x, head.weight, scales: head.scales, biases: head.biases,
            transpose: true, groupSize: 64, bits: 2, mode: .affine
        ).reshaped([-1])
    }

    private static func coarseChunked(_ x: MLXArray, _ head: QuantizedRows, chunks: Int)
        -> MLXArray
    {
        let rows = paddedCount / chunks
        let parts = (0 ..< chunks).map { c in
            coarse(x, head.rows((c * rows) ..< ((c + 1) * rows)))
        }
        return concatenated(parts, axis: 0)
    }

    private static func top32(_ scores: MLXArray) -> [Int32] {
        let real = scores[0 ..< realCount, axis: 0]
        let order = MLX.argPartition(real, kth: realCount - candidateCount, axis: -1)
        return order[(realCount - candidateCount) ..< realCount, axis: 0].asArray(Int32.self)
    }

    /// Exact affine-4 rerank over the nominated candidates, with the ties
    /// broken toward the lower vocabulary id so the comparison between arms is
    /// deterministic on its own terms.
    private static func rerankWinner(
        _ x: MLXArray, exact: QuantizedRows, candidates: [Int32]
    ) -> Int32 {
        let mapped = candidates.map(mapDraftID)
        let rows = exact.gather(MLXArray(mapped))
        let logits = quantizedMM(
            x, rows.weight, scales: rows.scales, biases: rows.biases,
            transpose: true, groupSize: 64, bits: 4, mode: .affine
        ).reshaped([-1]).asType(.float32).asArray(Float.self)

        var best = 0
        for i in 1 ..< logits.count {
            if logits[i] > logits[best] || (logits[i] == logits[best] && mapped[i] < mapped[best]) {
                best = i
            }
        }
        return mapped[best]
    }

    private static func vector(_ values: [Float]) -> MLXArray {
        MLXArray(values).reshaped([1, hidden]).asType(.bfloat16)
    }

    private static func family(_ name: String, rng: inout SplitMix) -> [Float] {
        switch name {
        case "outlier":
            // Activation outliers are the realistic worst case for a 2-bit
            // readout: a handful of channels dominate the dot product, so the
            // FP32 partial sums differ far more in magnitude between the two
            // lane partitions than they do for a well-conditioned input.
            var v = (0 ..< hidden).map { _ in rng.normal() }
            for _ in 0 ..< 32 { v[Int(rng.next() % UInt64(hidden))] *= 50 }
            return v
        case "tied":
            // Coarse-quantized activations make exact score ties common, which
            // is where candidate-set membership is most fragile.
            return (0 ..< hidden).map { _ in (rng.normal() * 4).rounded() / 4 }
        default:
            return (0 ..< hidden).map { _ in rng.normal() }
        }
    }

    /// Walk the input toward a near-tie between the last included and first
    /// excluded candidate. This is the input class where a 1-ulp coarse wobble
    /// has the best chance of flipping the candidate set, so it is measured
    /// deliberately rather than left to luck.
    private static func adversarial(
        base: [Float], direction: [Float], head: QuantizedRows
    ) -> [Float] {
        func at(_ t: Float) -> MLXArray {
            vector(zip(base, direction).map { $0 + t * $1 })
        }
        func gap(_ t: Float, _ a: Int, _ b: Int) -> Float {
            let s = coarse(at(t), head).asType(.float32)
            return s[a].item(Float.self) - s[b].item(Float.self)
        }

        let scores = coarse(vector(base), head)
        let ranked = top32(scores)
        let member = Set(ranked)
        let realScores = scores[0 ..< realCount, axis: 0].asType(.float32).asArray(Float.self)
        // Weakest included candidate, and the strongest excluded one: the pair
        // that decides membership.
        guard let a = ranked.min(by: { realScores[Int($0)] < realScores[Int($1)] }) else {
            return base
        }
        var outside: Int32 = -1
        var outsideBest = -Float.greatestFiniteMagnitude
        for i in 0 ..< realCount where !member.contains(Int32(i)) {
            if realScores[i] > outsideBest {
                outsideBest = realScores[i]
                outside = Int32(i)
            }
        }
        guard outside >= 0 else { return base }

        var lo: Float = 0
        var hi: Float = 0.25
        var f0 = gap(lo, Int(a), Int(outside))
        guard f0 != 0 else { return base }
        var found = false
        for _ in 0 ..< 10 {
            if gap(hi, Int(a), Int(outside)) * f0 < 0 {
                found = true
                break
            }
            hi *= 2
        }
        guard found else { return base }
        for _ in 0 ..< 24 {
            let mid = (lo + hi) / 2
            let fm = gap(mid, Int(a), Int(outside))
            if fm == 0 { lo = mid; break }
            if fm * f0 < 0 { hi = mid } else { lo = mid; f0 = fm }
        }
        let t = (lo + hi) / 2
        return zip(base, direction).map { $0 + t * $1 }
    }

    private static func measure(
        x: MLXArray, head: QuantizedRows, exact: QuantizedRows?
    ) -> TrialTotals {
        var out = TrialTotals()
        out.trials = 1

        let fast = coarse(x, head)
        let fastRepeat = coarse(x, head)
        let generic2 = coarseChunked(x, head, chunks: 2)
        let generic4 = coarseChunked(x, head, chunks: 4)
        eval(fast, fastRepeat, generic2, generic4)

        out.determinismMismatches =
            (fast .!= fastRepeat).asType(.int32).sum().item(Int.self)
        out.chunkControlMismatches =
            (generic2 .!= generic4).asType(.int32).sum().item(Int.self)

        out.cellsCompared = paddedCount
        out.differingCells = (fast .!= generic2).asType(.int32).sum().item(Int.self)

        let a = fast.asType(.float32)
        let b = generic2.asType(.float32)
        let delta = MLX.abs(a - b)
        out.maxAbsDelta = delta.max().item(Float.self)
        let denom = MLX.maximum(MLX.abs(a), MLX.abs(b)) + MLXArray(Float(1e-30))
        out.maxRelDelta = (delta / denom).max().item(Float.self)

        let fastTop = Set(top32(fast))
        let genericTop = Set(top32(generic2))
        let symmetric = fastTop.symmetricDifference(genericTop).count
        out.top32SymmetricDifference = symmetric
        out.top32ChurnTrials = symmetric == 0 ? 0 : 1

        let fastArg = fast[0 ..< realCount, axis: 0].argMax().item(Int32.self)
        let genericArg = generic2[0 ..< realCount, axis: 0].argMax().item(Int32.self)
        out.coarseArgmaxChanges = fastArg == genericArg ? 0 : 1

        if let exact {
            out.rerankTrials = 1
            let fastWinner = rerankWinner(x, exact: exact, candidates: Array(fastTop).sorted())
            let genericWinner = rerankWinner(
                x, exact: exact, candidates: Array(genericTop).sorted())
            out.rerankTop1Changes = fastWinner == genericWinner ? 0 : 1
        }
        return out
    }

    @Test(.enabled(if: QwenDraftReadoutExactnessTests.enabled))
    func theSinglerowAffine2ArmAgreesWithTheGenericArmOnTheDeclaredHead() throws {
        let outPath = try #require(
            Self.env["MLXFAST_DRAFT_READOUT_EXACTNESS_OUT"],
            "set MLXFAST_DRAFT_READOUT_EXACTNESS_OUT to the JSON destination")
        let headURL = Self.headURL
        try #require(
            FileManager.default.fileExists(atPath: headURL.path),
            "declared proposal head not found; set MLXFAST_QWEN_MTP_HEAD_DIR")

        let raw = try loadArrays(url: headURL)
        let head = QuantizedRows(
            weight: try #require(raw["draft_lm_head.weight"]),
            scales: try #require(raw["draft_lm_head.scales"]),
            biases: try #require(raw["draft_lm_head.biases"]))
        #expect(head.weight.shape == [Self.paddedCount, 320])
        #expect(head.scales.shape == [Self.paddedCount, 80])

        var exact: QuantizedRows?
        if FileManager.default.fileExists(atPath: Self.backboneShardURL.path) {
            let shard = try loadArrays(url: Self.backboneShardURL)
            if let w = shard["language_model.lm_head.weight"],
                let s = shard["language_model.lm_head.scales"],
                let z = shard["language_model.lm_head.biases"]
            {
                exact = QuantizedRows(weight: w, scales: s, biases: z)
            }
        }

        let perFamily = Int(Self.env["MLXFAST_DRAFT_READOUT_TRIALS"] ?? "") ?? 16
        var rng = SplitMix(UInt64(Self.env["MLXFAST_DRAFT_READOUT_SEED"] ?? "") ?? 0x5E_1D_28)

        var families: [[String: Any]] = []
        var grand = TrialTotals()
        for name in ["normal", "outlier", "tied", "adversarial"] {
            var totals = TrialTotals()
            for _ in 0 ..< perFamily {
                let values: [Float]
                if name == "adversarial" {
                    let base = Self.family("normal", rng: &rng)
                    let direction = Self.family("normal", rng: &rng)
                    values = Self.adversarial(base: base, direction: direction, head: head)
                } else {
                    values = Self.family(name, rng: &rng)
                }
                totals.absorb(Self.measure(x: Self.vector(values), head: head, exact: exact))
            }
            var entry = totals.payload
            entry["family"] = name
            families.append(entry)
            grand.absorb(totals)
        }

        let payload: [String: Any] = [
            "schema": "e28.draft_readout_exactness.v1",
            "mechanism": "qmv_fast_singlerow_affine2_g64",
            "reference_arm": "qmv_fast_impl<T,64,2> via 2x49168 row slices",
            "self_consistency_arm": "qmv_fast_impl<T,64,2> via 4x24584 row slices",
            "head": headURL.path,
            "exact_rerank_available": exact != nil,
            "trials_per_family": perFamily,
            "families": families,
            "totals": grand.payload,
        ]
        let data = try JSONSerialization.data(
            withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: URL(fileURLWithPath: outPath))
        print("E28_DRAFT_READOUT_EXACTNESS \(outPath) totals=\(grand.payload)")

        // Hard invariants. Everything else is measured and reported, not
        // asserted: a nonzero differing-cell count between two orderings of a
        // 5120-term FP32 dot product is expected, and the audit's job is to
        // quantify it and prove containment.
        #expect(
            grand.determinismMismatches == 0,
            "E28 / PR #33: the inherited 2-bit arm is not deterministic run to run")
        #expect(
            grand.chunkControlMismatches == 0,
            """
            E28 / PR #33: two different row chunkings of the generic arm \
            disagreed. The differential comparison assumes each output row is \
            an independent dot product; that assumption just failed.
            """
        )
    }
}

// MARK: - Custom top-32 selection kernel

/// `Qwen35` ships a bespoke top-32 selection kernel on the scored draft path
/// and a self-check for it, but nothing called that self-check. Wiring it here
/// turns dormant validation code into a standing gate, and it is also what
/// licenses the exactness suite above to use `argPartition` as its
/// candidate-set oracle.
@Suite
struct QwenDraftTop32SelectionTests {
    private static var enabled: Bool {
        ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1"
    }

    @Test(.enabled(if: QwenDraftTop32SelectionTests.enabled))
    func theCustomTop32KernelMatchesArgPartitionIncludingHeavilyTiedRows() {
        let (trials, bad, firstBad) = qwen35VerifyDraftTop32(trials: 64, seed: 1)
        #expect(trials == 64)
        #expect(
            bad == 0 && firstBad == -1,
            """
            E28 / PR #33: qwen35DraftTop32 disagreed with argPartition on \
            \(bad) of \(trials) rows (first at \(firstBad)). Every fourth \
            trial is heavily tied, so a failure here most likely means the \
            candidate-set tie-break drifted.
            """
        )
    }
}
