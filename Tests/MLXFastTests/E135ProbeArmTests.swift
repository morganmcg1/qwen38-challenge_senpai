import Foundation
import MLXLLM
import Testing

// E135 follow-up -- make the leaf pre-selection fraction an addressable arm.
//
// `qwen35DerivedClusterProbeFraction` was a plain compiled constant, so every
// rung of the ladder needed its own build and no counterbalanced session could
// hold two rungs. It now reads `Qwen35CustomQMV.ProbeArm`, which one process
// selects from `MLX_E135_PROBE_ARM`.
//
// The safety claim is narrow: a run that exports nothing -- the ranked run --
// must take `ProbeArm.compiledDefault`, and the shipped witness literal must
// name that same rung. These tests are pure Swift and need no GPU.

@Suite("E135 probe-fraction arm")
struct E135ProbeArmTests {

    /// The ranked runner sets no environment, so the fraction it takes is
    /// whatever `probeArm` falls back to. Every raw value is compiled in
    /// whichever rung ships, so `"p15"` cannot witness the fallback on its
    /// own; this whole literal exists only for the rung selected.
    @Test("the default-probe witness names the compiled default and can fail")
    func defaultProbeWitnessNamesTheCompiledDefault() throws {
        #expect(
            Qwen35CustomQMV.defaultProbeWitness
                == "e135_default_probe/"
                    + Qwen35CustomQMV.ProbeArm.compiledDefault.rawValue)

        // The witness is worthless if the optimizer can fold it out of a
        // shorter shipped string, and worthless if the fallback ignores it.
        #expect(Qwen35CustomQMV.defaultProbeWitness.utf8.count >= 16)
        #expect(Qwen35CustomQMV.ProbeArm.compiledDefault == .p15)

        if ProcessInfo.processInfo.environment["MLX_E135_PROBE_ARM"] == nil {
            #expect(Qwen35CustomQMV.probeArm == Qwen35CustomQMV.ProbeArm.compiledDefault)
            #expect(qwen35DerivedClusterProbeFraction == 0.15)
        }
    }

    /// Each rung must name the fraction its label claims, and the ladder must
    /// stay ordered, or a leg would time a rung other than the one its tag
    /// records.
    @Test("every rung parses to the fraction its name claims")
    func everyRungParsesToItsFraction() throws {
        for (raw, fraction) in [("p25", 0.25), ("p15", 0.15), ("p10", 0.10)] {
            let parsed = try #require(Qwen35CustomQMV.ProbeArm(rawValue: raw))
            #expect(parsed.fraction == fraction)
        }
        #expect(Qwen35CustomQMV.ProbeArm.allCases.count == 3)

        let fractions = Qwen35CustomQMV.ProbeArm.allCases.map(\.fraction)
        #expect(fractions == fractions.sorted(by: >))
        #expect(
            qwen35DerivedClusterProbeFraction == Qwen35CustomQMV.probeArm.fraction)
    }

    /// The declared head has 98,336 padded rows in leaves of 8, so 12,292
    /// leaves. Every rung must derive a different integer, or the trace could
    /// not tell the arms apart, and the shipped rung must derive the exact
    /// integer the runtime gate reads back off the leg's own trace.
    ///
    /// The name states no rung on purpose. The compiled default has moved
    /// twice on advisor evidence, and a name that carries the integer goes
    /// stale silently on every move.
    @Test("every rung derives a distinct probe count and the shipped one is pinned")
    func everyRungDerivesADistinctProbeCount() throws {
        let leaves = 98_336 / 8
        #expect(leaves == 12_292)

        #expect(Qwen35CustomQMV.ProbeArm.p25.probes(leaves: leaves) == 3073)
        #expect(Qwen35CustomQMV.ProbeArm.p15.probes(leaves: leaves) == 1844)
        #expect(Qwen35CustomQMV.ProbeArm.p10.probes(leaves: leaves) == 1230)

        let derived = Qwen35CustomQMV.ProbeArm.allCases.map { $0.probes(leaves: leaves) }
        #expect(Set(derived).count == derived.count)

        // Move this integer with `compiledDefault`; the gate asserts the same
        // one against the trace.
        #expect(
            Qwen35CustomQMV.ProbeArm.compiledDefault.probes(leaves: leaves) == 1844)

        // The rule never returns zero, whatever the head size, because a step
        // that scores no leaf cannot propose.
        for arm in Qwen35CustomQMV.ProbeArm.allCases {
            #expect(arm.probes(leaves: 1) == 1)
            #expect(arm.probes(leaves: 0) == 1)
        }
    }

    /// An unknown or empty override must fall back rather than crash or select
    /// a rung by accident, because the fallback is the ranked path.
    @Test("an unknown rung name is not a rung")
    func anUnknownRungNameIsNotARung() throws {
        #expect(Qwen35CustomQMV.ProbeArm(rawValue: "") == nil)
        #expect(Qwen35CustomQMV.ProbeArm(rawValue: "0.10") == nil)
        #expect(Qwen35CustomQMV.ProbeArm(rawValue: "p20") == nil)
    }

    /// A rung name in the JIT text would give each rung its own pipeline set
    /// and charge the ladder a compile it is trying to measure around. The
    /// fraction changes how many leaves are scored, not the kernel text.
    @Test("no probe rung reaches a pipeline cache key")
    func noProbeRungReachesAPipelineCacheKey() throws {
        for arm in Qwen35CustomQMV.ProbeArm.allCases {
            for useTable in [true, false] {
                #expect(
                    !Qwen35CustomQMV.pipelineName(useTable: useTable, tier: nil)
                        .contains(arm.rawValue))
                for tier in Qwen35CustomQMV.tiers {
                    #expect(
                        !Qwen35CustomQMV.pipelineName(useTable: useTable, tier: tier)
                            .contains(arm.rawValue))
                }
            }
        }
    }
}
