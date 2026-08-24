//
//  E184PrefillProfile.swift
//  RESEARCH INSTRUMENT — NEVER SUBMITTED.
//
//  This file is deliberately placed OUTSIDE `benchmark.json` `editablePaths`
//  (the manifest names only `Qwen35.swift`, `Qwen35MTP.swift` and
//  `Qwen35MoE.swift` in this directory), so a submission archive cannot carry
//  it. Every call site in `Qwen35.swift` is guarded by `E184Prefill.active`,
//  which is false unless `MLXFAST_E184_PREFILL_PROFILE=1` is exported AND the
//  forward being executed is a seed-width forward.
//
//  WHAT IT MEASURES. E83 attributed the 512-token seed prefill by summing
//  ISOLATED kernel costs and reached 100.2 % of the measured `begin()`, which
//  leaves no room for the non-GEMM terms that provably exist. This instrument
//  measures the same phases IN PATH instead: each `step` forces an eval
//  boundary and charges the wall time since the previous boundary to one
//  phase label, so the phase table closes to the bracketed total by
//  construction and the only reconciliation left is bracketed total versus the
//  trusted `seed_prefill_seconds` of an uninstrumented run.
//
//  WHAT IT COSTS. An eval boundary drains the pipeline and stops the host from
//  running ahead of the GPU, so the bracketed total is an UPPER bound on the
//  natural prefill wall time. `MLXFAST_E184_PREFILL_PROFILE=coarse` keeps only
//  the layer-level boundaries (64 of them instead of ~400) so the distortion
//  can itself be measured.
//
import Foundation
import MLX

public enum E184Prefill {
    public enum Granularity: String, Sendable {
        case off
        case fine
        case coarse
    }

    static let granularity: Granularity = {
        let raw = ProcessInfo.processInfo.environment[
            "MLXFAST_E184_PREFILL_PROFILE"] ?? ""
        switch raw {
        case "1", "fine": return .fine
        case "coarse": return .coarse
        default: return .off
        }
    }()

    /// Smallest sequence length treated as a seed-width forward.
    static let minSequenceLength = Int(
        ProcessInfo.processInfo.environment["MLXFAST_E184_PREFILL_MIN_S"] ?? "")
        ?? 512

    static let outputPath = ProcessInfo.processInfo.environment[
        "MLXFAST_E184_PREFILL_PROFILE_OUT"]

    /// True only inside a seed-width forward while profiling is enabled.
    nonisolated(unsafe) public private(set) static var active = false
    /// True only when every sub-phase boundary is wanted.
    nonisolated(unsafe) public private(set) static var fine = false

    nonisolated(unsafe) private static var lastMark: UInt64 = 0
    nonisolated(unsafe) private static var order: [String] = []
    nonisolated(unsafe) private static var totals: [String: UInt64] = [:]
    nonisolated(unsafe) private static var counts: [String: Int] = [:]
    nonisolated(unsafe) private static var forwardIndex = 0
    nonisolated(unsafe) private static var forwardStart: UInt64 = 0
    nonisolated(unsafe) private static var sequenceLength = 0

    @inline(__always)
    public static func beginForward(sequenceLength length: Int) {
        guard granularity != .off, length >= minSequenceLength else {
            active = false
            fine = false
            return
        }
        active = true
        fine = granularity == .fine
        order.removeAll(keepingCapacity: true)
        totals.removeAll(keepingCapacity: true)
        counts.removeAll(keepingCapacity: true)
        sequenceLength = length
        forwardStart = DispatchTime.now().uptimeNanoseconds
        lastMark = forwardStart
    }

    /// Close the current phase: evaluate `arrays`, then charge every
    /// nanosecond since the previous boundary to `label`.
    @inline(__always)
    public static func step(_ label: String, _ arrays: [MLXArray]) {
        guard active else { return }
        if !arrays.isEmpty { eval(arrays) }
        let now = DispatchTime.now().uptimeNanoseconds
        let elapsed = now - lastMark
        lastMark = now
        if totals[label] == nil {
            order.append(label)
            totals[label] = 0
            counts[label] = 0
        }
        totals[label]! += elapsed
        counts[label]! += 1
    }

    /// Sub-phase boundary: present only in `fine` mode.
    @inline(__always)
    public static func fineStep(_ label: String, _ arrays: [MLXArray]) {
        guard active, fine else { return }
        step(label, arrays)
    }

    @inline(__always)
    public static func endForward(_ arrays: [MLXArray]) {
        guard active else { return }
        step("99_tail", arrays)
        let total = DispatchTime.now().uptimeNanoseconds - forwardStart
        forwardIndex += 1
        var phases: [String] = []
        for label in order {
            let ns = totals[label] ?? 0
            phases.append(
                "{\"phase\":\"\(label)\",\"seconds\":\(Double(ns) / 1e9),"
                    + "\"calls\":\(counts[label] ?? 0)}")
        }
        let line = "{\"e184_prefill_profile\":1,"
            + "\"pid\":\(ProcessInfo.processInfo.processIdentifier),"
            + "\"forward_index\":\(forwardIndex),"
            + "\"sequence_length\":\(sequenceLength),"
            + "\"granularity\":\"\(granularity.rawValue)\","
            + "\"bracketed_total_seconds\":\(Double(total) / 1e9),"
            + "\"phases\":[\(phases.joined(separator: ","))]}\n"
        write(line)
        active = false
        fine = false
    }

    private static func write(_ line: String) {
        if let outputPath, let handle = FileHandle(forWritingAtPath: outputPath)
            ?? {
                FileManager.default.createFile(atPath: outputPath, contents: nil)
                return FileHandle(forWritingAtPath: outputPath)
            }()
        {
            handle.seekToEndOfFile()
            handle.write(Data(line.utf8))
            try? handle.close()
            return
        }
        FileHandle.standardError.write(Data(line.utf8))
    }
}
