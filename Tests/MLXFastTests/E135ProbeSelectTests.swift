import Foundation
import MLX
import MLXLLM
import Testing

// E135 / PR #135 -- exactness gate for the restored E87 probe select kernel.
//
// `qwen_mtp_e87_probe_select` replaces
// `MLX.sorted(MLX.argPartition(score, kth: C - P)[(C - P)...])` with one
// dispatch. The frontier snapshot that ships this kernel ships no test for it,
// and this kernel needs one more than the compaction kernel it supersedes.
//
// The compaction kernel consumes an index permutation, so its inputs are
// distinct by construction and it cannot inherit a tie decision. This kernel
// consumes float scores, so the tie rule is load bearing: among elements equal
// to the threshold key it must keep the highest indices, which is what
// `argPartition` plus an ascending sort produces. The trial mix is therefore
// weighted toward ties, and one trial is an all-zero row where every element
// ties and the index rule alone decides the answer.
//
// The 16-bit key the kernel packs is exact only for bf16 scores. Both live
// producers satisfy that: `qwen35ClusterCentroidQMV` declares
// `outputDTypes: [.bfloat16]`, and the `quantizedMM` fallback returns bf16.
@Suite
struct E135ProbeSelectTests {
    private static var env: [String: String] { ProcessInfo.processInfo.environment }

    private static var enabled: Bool {
        env["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1"
    }

    /// The live arm C geometry: `derivedClusterRowsPerLeaf` 8 over 98,336
    /// compact rows at `qwen35DerivedClusterProbeFraction` 0.25.
    private static let liveClusters = 12_292
    private static let liveProbes = 3_073

    private static func emit(_ name: String, _ payload: [String: Any]) throws {
        print("E135_PROBE_SELECT \(name) \(payload)")
        guard let dir = env["MLXFAST_PROBE_SELECT_OUT_DIR"] else { return }
        let data = try JSONSerialization.data(
            withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: URL(fileURLWithPath: dir).appendingPathComponent("\(name).json"))
    }

    @Test(.enabled(if: E135ProbeSelectTests.enabled))
    func theSelectKernelMatchesSortedArgPartitionIncludingTiedAndAllEqualScores() throws {
        let requested = Int(Self.env["MLXFAST_PROBE_SELECT_TRIALS"] ?? "") ?? 64
        let seed = UInt64(Self.env["MLXFAST_PROBE_SELECT_SEED"] ?? "") ?? 1
        let (trials, bad, firstBad) = qwen35VerifyProbeSelect(
            clusters: Self.liveClusters, probes: Self.liveProbes,
            trials: requested, seed: seed)
        try Self.emit(
            "verify",
            [
                "schema": "e135.probe_select_verify.v1",
                "entry_point": "qwen35VerifyProbeSelect",
                "kernel": "qwen_mtp_e87_probe_select",
                "clusters": Self.liveClusters,
                "probes": Self.liveProbes,
                "trials": trials,
                "seed": Int(seed),
                "coarse_tied_trials": trials / 4,
                "all_equal_trials": trials / 4,
                "fine_tied_trials": trials / 4,
                "mismatches": bad,
                "first_bad_trial": firstBad,
            ])
        #expect(trials == requested)
        #expect(
            bad == 0 && firstBad == -1,
            """
            E135: qwen_mtp_e87_probe_select disagreed with \
            sorted(argPartition(score)) on \(bad) of \(trials) rows (first at \
            \(firstBad)). The two arms must agree on every row, including the \
            all-equal row where the index rule alone decides the selection.
            """
        )
    }

    /// The gate above would pass just as happily against a comparison that
    /// cannot fail. This proves it can fail, using the smallest damage that
    /// changes the answer: promote one rejected score so exactly one index
    /// enters the selected set.
    @Test(.enabled(if: E135ProbeSelectTests.enabled))
    func theSelectGateRejectsASinglePromotedIndex() throws {
        let seed = UInt64(Self.env["MLXFAST_PROBE_SELECT_CONTROL_SEED"] ?? "") ?? 7
        let caught = qwen35ProbeSelectPositiveControl(
            clusters: Self.liveClusters, probes: Self.liveProbes, seed: seed)
        try Self.emit(
            "positive_control",
            [
                "schema": "e135.probe_select_positive_control.v1",
                "entry_point": "qwen35ProbeSelectPositiveControl",
                "kernel": "qwen_mtp_e87_probe_select",
                "clusters": Self.liveClusters,
                "probes": Self.liveProbes,
                "seed": Int(seed),
                "damage": "raise one rejected score above every selected score",
                "detected": caught,
            ])
        #expect(
            caught,
            """
            E135: the select gate accepted a row with one extra index promoted \
            into the selected set. The gate cannot fail, so its passes prove \
            nothing. Fix the comparison before trusting the verify test.
            """
        )
    }
}
