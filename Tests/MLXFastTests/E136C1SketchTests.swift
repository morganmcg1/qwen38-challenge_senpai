import Foundation
import MLX
import MLXLLM
import Testing

// E136 / arm C1 -- risk gate for the widened sketch shortlist.
//
// The shipped readout narrows 98,330 compact rows to 32 candidates with an
// affine-2 centroid scan, a 0.25 probe fraction, an affine-2 row scan over
// 24,584 rows and a two-dispatch top-32. Arm C1 replaces the affine-2 row scan
// with a rank-256 int8 sketch at a 0.35 probe fraction, keeps 4,096 survivors
// and rescores only those rows with affine-2. That trades 37.5 MB of read
// traffic per draft step for a wider but cheaper first pass.
//
// The sketch is an approximation, so the arm cannot claim that the ORDER of
// the 32 candidates is unchanged. What it must prove is narrower and testable:
// every dispatch computes the arithmetic it claims to compute, the threshold
// is the exact boundary bin of the sketch scores, the compaction loses no row
// at or above that bin, and the emitted 32 ids are the exact affine-2 top 32
// OF THE SURVIVORS under the same total order the shipped path uses. Whether
// those 32 ids are the same 32 the shipped path emits is an acceptance
// question, priced end to end in the rung-2 timing arm, not an exactness one.
//
// The shortlist never reaches the target verifier, the row ledger or the
// top-two evidence. It only chooses which rows the exact affine-4 rerank looks
// at, and that rerank is unchanged.
@Suite
struct E136C1SketchTests {
    private static var env: [String: String] { ProcessInfo.processInfo.environment }

    private static var enabled: Bool {
        env["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1"
    }

    /// A synthetic index of the shipped shape family: eight rows per leaf,
    /// `qwen35C1ProbeFraction` of the leaves probed, hidden 5,120, and more
    /// probed rows than survivors so the threshold has real work to do.
    private static let clusters = 2_048
    private static let hidden = 5_120

    private static func emit(_ name: String, _ payload: [String: Any]) throws {
        print("E136_C1 \(name) \(payload)")
        guard let dir = env["MLXFAST_C1_OUT_DIR"] else { return }
        let data = try JSONSerialization.data(
            withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
        try data.write(
            to: URL(fileURLWithPath: dir).appendingPathComponent("\(name).json"))
    }

    @Test(.enabled(if: E136C1SketchTests.enabled))
    func theC1ShortlistMatchesItsReferenceAtEveryBoundary() throws {
        let seed = UInt64(Self.env["MLXFAST_C1_SEED"] ?? "") ?? 1_136
        let r = qwen35VerifyC1Selection(
            clusters: Self.clusters, hidden: Self.hidden, seed: seed)
        try Self.emit(
            "verify",
            [
                "schema": "e136.c1_selection_verify.v1",
                "entry_point": "qwen35VerifyC1Selection",
                "clusters": Self.clusters,
                "hidden": Self.hidden,
                "seed": Int(seed),
                "project_rel_error": Double(r.projectRelError),
                "centroid_sketch_rel_error": Double(r.centroidSketchRelError),
                "row_sketch_rel_error": Double(r.rowSketchRelError),
                "positive_control_rel_error": Double(r.positiveControlRelError),
                "rescore_contiguous_mismatches": r.rescoreContiguousMismatches,
                "rescore_gathered_mismatches": r.rescoreGatheredMismatches,
                "tau_is_exact": r.tauIsExact,
                "above_is_exact": r.aboveIsExact,
                "cursor_is_exact": r.cursorIsExact,
                "survivor_capacity": r.survivorCapacity,
                "survivors_distinct": r.survivorsDistinct,
                "survivors_below_tau": r.survivorsBelowTau,
                "sketch_top32_recall": r.sketchTop32Recall,
                "shortlist_mismatches": r.shortlistMismatches,
            ])

        // 1. Arithmetic. fp32 reassociation only, so a tight relative bound.
        #expect(
            r.projectRelError < 1e-5,
            """
            E136 C1: qwen_mtp_c1_project disagreed with B^T (x - mu) by \
            \(r.projectRelError) relative. The kernel stages x - mu in \
            threadgroup memory and reduces one basis row per simdgroup, so \
            only fp32 reassociation should differ.
            """
        )
        #expect(
            r.centroidSketchRelError < 1e-5 && r.rowSketchRelError < 1e-5,
            """
            E136 C1: the sketch scan disagreed with scale * (codes . q) + \
            offset by \(r.centroidSketchRelError) dense and \
            \(r.rowSketchRelError) gathered. A gathered failure alone points \
            at the probe address, not at the dot product.
            """
        )

        // 2. The positive control. The comparison above must be able to fail.
        #expect(
            r.positiveControlRelError > 1e-2,
            """
            E136 C1: comparing the sketch scan against a ROLLED reference \
            reported only \(r.positiveControlRelError) relative error, so the \
            comparison cannot distinguish the right answer from the wrong \
            one. Fix the comparison before trusting the checks above.
            """
        )

        // 3. The gathered rescore. Exact, against the shipped affine-2 kernel.
        #expect(
            r.rescoreContiguousMismatches == 0,
            """
            E136 C1: qwen_e136_a2_qmv4_gathered differed from the shipped \
            qwen_mtp_cluster_centroid_qmv_a2g64_v1 on \
            \(r.rescoreContiguousMismatches) of \(r.survivorCapacity) rows \
            under an identity gather. The two must be bit identical: only the \
            row address changed.
            """
        )
        #expect(
            r.rescoreGatheredMismatches == 0,
            """
            E136 C1: the gathered rescore was not a permutation of itself on \
            \(r.rescoreGatheredMismatches) of \(r.survivorCapacity) rows, so \
            the row address is not being read where the survivor list says.
            """
        )

        // 4. The threshold and the compaction.
        #expect(
            r.tauIsExact,
            """
            E136 C1: the two-level histogram scan did not choose the exact \
            boundary bin. tau must be the largest bin where at least \
            \(r.survivorCapacity) rows sit at or above it and fewer than that \
            sit strictly above it.
            """
        )
        #expect(
            r.aboveIsExact && r.cursorIsExact,
            """
            E136 C1: the survivor accounting is wrong. ctl[1] must equal the \
            number of rows strictly above tau, cursor[0] must equal it too, \
            and cursor[0] + cursor[1] must equal the number of rows at or \
            above tau. A single shared cursor passes the recall check by luck \
            and fails here.
            """
        )
        #expect(
            r.survivorsDistinct == r.survivorCapacity,
            """
            E136 C1: the compaction filled \(r.survivorsDistinct) distinct \
            slots of \(r.survivorCapacity). Every slot must hold a different \
            row, so a shortfall is either an unfilled slot or two rows \
            sharing one slot.
            """
        )
        #expect(
            r.survivorsBelowTau == 0,
            """
            E136 C1: \(r.survivorsBelowTau) survivors came from below the \
            boundary bin, so the compaction admitted rows the threshold had \
            already rejected.
            """
        )

        // 5. The recall invariant the arm's cost model depends on.
        #expect(
            r.sketchTop32Recall == 1.0,
            """
            E136 C1: only \(r.sketchTop32Recall) of the sketch top 32 \
            survived. With \(r.survivorCapacity) survivors chosen by the same \
            total order this is exact by construction, so a shortfall is a \
            compaction defect and not an approximation effect.
            """
        )

        // 6. The emitted ids.
        #expect(
            r.shortlistMismatches == 0,
            """
            E136 C1: \(r.shortlistMismatches) of \(r.topK) emitted ids \
            differed from the host affine-2 top 32 of the same survivors. The \
            selection ties on the permuted row id and writes rank r to slot \
            31 - r, so an off-by-one here reverses or rotates the shortlist.
            """
        )
    }
}
