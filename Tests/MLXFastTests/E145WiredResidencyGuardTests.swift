import Foundation
@testable import MLXFastModel
import Testing

// E145 R6: the residency-sizing guard's physical-memory floor.
//
// `wireResidentWeightsIfEnabled()` refuses below 96 GiB, so a development host
// can never execute the wiring path and no local measurement of it exists.
// `MLX_E145_WIRED_MIN_GIB` lowers the floor for a diagnostic session. These
// tests own three properties: the floor is the shipped 96 GiB unless the
// variable names a number, the predicate the guard evaluates changes outcome
// on this host when it does, and both guard branches leave a witness.
@Suite("E145 wired residency guard")
struct E145WiredResidencyGuardTests {
    private static let developmentHostBytes = UInt64(48) << 30
    private static let rankedRunnerBytes = UInt64(128) << 30

    /// The default-off gate. It fails the moment the diagnostic knob acquires
    /// a compiled default other than the shipped floor.
    @Test("the floor is the shipped 96 GiB unless the variable is set")
    func floorDefaultsToShippedGuard() {
        #expect(ProcessInfo.processInfo
            .environment["MLX_E145_WIRED_MIN_GIB"] == nil)
        #expect(qwen36WiredMinimumPhysicalGiB([:]) == 96)
        #expect(
            qwen36WiredMinimumPhysicalGiB(
                ProcessInfo.processInfo.environment) == 96)
    }

    /// A value that is not a non-negative integer must not silently open the
    /// guard. `UInt64.init` rejects each of these, so the floor stays shipped.
    @Test("an unparsable value leaves the shipped floor in place")
    func unparsableValueKeepsShippedFloor() {
        for value in ["", "32.5", "-32", "thirty-two", "0x20", "32 "] {
            #expect(
                qwen36WiredMinimumPhysicalGiB(
                    ["MLX_E145_WIRED_MIN_GIB": value]) == 96,
                "\(value) must not lower the floor")
        }
    }

    /// Rule 101: both polarities of the predicate the guard actually
    /// evaluates, on the two hosts that matter.
    ///
    /// The first two expectations are the failing control. If the override did
    /// nothing, or if the shipped floor were already below this host's memory,
    /// one of them would fail and the diagnostic would be measuring nothing.
    @Test("the override flips the guard on a development host only")
    func overrideFlipsTheGuardOnADevelopmentHostOnly() {
        func admits(_ environment: [String: String], _ bytes: UInt64) -> Bool {
            bytes >= (qwen36WiredMinimumPhysicalGiB(environment) << 30)
        }
        let relaxed = ["MLX_E145_WIRED_MIN_GIB": "32"]

        #expect(!admits([:], Self.developmentHostBytes))
        #expect(admits(relaxed, Self.developmentHostBytes))

        // The ranked runner clears both floors, so the knob cannot change what
        // the scored M5 run does.
        #expect(admits([:], Self.rankedRunnerBytes))
        #expect(admits(relaxed, Self.rankedRunnerBytes))

        // A floor above the ranked runner still closes the guard, which proves
        // the comparison is live rather than short-circuited to true.
        #expect(
            !admits(["MLX_E145_WIRED_MIN_GIB": "256"], Self.rankedRunnerBytes))
    }

    /// Rule 114: a timed leg can only witness the residency path from the
    /// line the session writes, because `mtp-timed` swallows worker stderr.
    /// Both the taken branch and both refusal branches must emit, or an absent
    /// line would be ambiguous between "declined" and "never ran".
    @Test("every guard branch leaves a witness line")
    func everyGuardBranchLeavesAWitnessLine() throws {
        let source = try String(
            contentsOfFile:
                "Sources/MLXFastModel/Qwen36MTPBlockSession.swift",
            encoding: .utf8
        )
        #expect(source.contains("emitWiredTelemetry(\n                \"skipped=disabled"))
        #expect(source.contains("emitWiredTelemetry(\n                \"skipped=guard"))
        #expect(source.contains("emitWiredTelemetry(body)"))
        #expect(source.contains("MLX_E145_WIRED_LOG"))

        // The guard must read the resolved floor, not a second hardcoded
        // literal. This is the failing control for the refactor itself.
        #expect(!source.contains("physicalMemory >= (UInt64(96) << 30)"))
        #expect(
            source.contains("physical >= (wiredMinimumPhysicalGiB << 30)"))
    }
}
