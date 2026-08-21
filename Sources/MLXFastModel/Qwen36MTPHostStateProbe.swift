import Darwin
import Foundation
import MLX

// E89 host-state probe. LOCAL RESEARCH INSTRUMENT ONLY.
//
// E86 and E85 found, on two different Macs, that a drafting leg lands in one
// of two host states. In the slow state every one of the eight once-per-round
// host phases costs 2.9x to 7.9x more, while the GPU phases do not move at
// all. The ranked board shows the same signature: a binary per-run mode worth
// 1.17 % of leg time that touches only the drafting prompts, and neither the
// prefill phase nor the runner-owned serial leg.
//
// This probe answers one question: is the host thread RUNNING SLOWER, or is
// it DOING MORE WORK? The two readings imply completely different fixes.
//
//   `cpuProbeNanos`     fixed 20,000-iteration dependent integer chain, no
//                       memory traffic. Its wall time is a direct readout of
//                       issue latency, which tracks core type and clock.
//   `ri_instructions`   exact retired-instruction count for the round. If the
//                       thread does more work, this rises. If the thread is
//                       merely slower, this does not move.
//   `ri_cycles`         with `ri_user_time`, an effective-clock estimate.
//   per-QoS CPU time    which service class actually accrued the round's CPU
//                       nanoseconds.
//   thread policy       QoS class, Darwin role, current scheduling priority,
//                       and the mach id of the thread that submits.
//   MLX allocator       active, cache and peak bytes, to test whether the two
//                       states sit at different allocator steady states.
//
// Every field is gated behind `MLX_E89_PROBE=1`, which the ranked path never
// sets. The `MLX_` prefix is required because the worker environment
// sanitizer drops `MLXFAST_*`.
enum Qwen36MTPHostStateProbe {
    static let enabled =
        ProcessInfo.processInfo.environment["MLX_E89_PROBE"] == "1"

    /// `PRIO_DARWIN_ROLE`. The Darwin headers define it as a macro, so it does
    /// not survive into Swift.
    private static let prioDarwinRole: Int32 = 6

    private static let machTimebase: (numer: UInt64, denom: UInt64) = {
        var info = mach_timebase_info_data_t()
        mach_timebase_info(&info)
        return (UInt64(info.numer), UInt64(info.denom))
    }()

    static func machTicksToNanos(_ ticks: UInt64) -> UInt64 {
        ticks * machTimebase.numer / machTimebase.denom
    }

    // MARK: - resource usage

    struct Usage {
        var instructions: UInt64 = 0
        var cycles: UInt64 = 0
        var userTicks: UInt64 = 0
        var systemTicks: UInt64 = 0
        var qosDefault: UInt64 = 0
        var qosUtility: UInt64 = 0
        var qosUserInitiated: UInt64 = 0
        var qosUserInteractive: UInt64 = 0
        var qosBackground: UInt64 = 0
        var qosMaintenance: UInt64 = 0

        static func - (lhs: Usage, rhs: Usage) -> Usage {
            Usage(
                instructions: lhs.instructions &- rhs.instructions,
                cycles: lhs.cycles &- rhs.cycles,
                userTicks: lhs.userTicks &- rhs.userTicks,
                systemTicks: lhs.systemTicks &- rhs.systemTicks,
                qosDefault: lhs.qosDefault &- rhs.qosDefault,
                qosUtility: lhs.qosUtility &- rhs.qosUtility,
                qosUserInitiated: lhs.qosUserInitiated &- rhs.qosUserInitiated,
                qosUserInteractive: lhs.qosUserInteractive
                    &- rhs.qosUserInteractive,
                qosBackground: lhs.qosBackground &- rhs.qosBackground,
                qosMaintenance: lhs.qosMaintenance &- rhs.qosMaintenance)
        }
    }

    static func usage() -> Usage {
        var info = rusage_info_v4()
        let rc = withUnsafeMutablePointer(to: &info) { pointer -> Int32 in
            pointer.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) {
                proc_pid_rusage(getpid(), RUSAGE_INFO_V4, $0)
            }
        }
        guard rc == 0 else { return Usage() }
        return Usage(
            instructions: info.ri_instructions,
            cycles: info.ri_cycles,
            userTicks: info.ri_user_time,
            systemTicks: info.ri_system_time,
            qosDefault: info.ri_cpu_time_qos_default,
            qosUtility: info.ri_cpu_time_qos_utility,
            qosUserInitiated: info.ri_cpu_time_qos_user_initiated,
            qosUserInteractive: info.ri_cpu_time_qos_user_interactive,
            qosBackground: info.ri_cpu_time_qos_background,
            qosMaintenance: info.ri_cpu_time_qos_maintenance)
    }

    // MARK: - dependent-chain CPU speed probe

    /// Written so the optimizer cannot delete the loop. Never read.
    nonisolated(unsafe) static var probeSink: UInt64 = 0

    /// About 39 us on a fast host thread and about 79 us on a slow one, with
    /// nothing in between, measured on this M4 Pro. 20,000 iterations cost
    /// under 0.05 % of a 13 second leg.
    @inline(never)
    static func cpuProbeNanos() -> UInt64 {
        let t0 = DispatchTime.now().uptimeNanoseconds
        var x: UInt64 = 0x9E37_79B9_7F4A_7C15
        for _ in 0 ..< 20_000 {
            x = x &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            x ^= x >> 31
        }
        probeSink = x
        return DispatchTime.now().uptimeNanoseconds - t0
    }

    // MARK: - thread policy

    struct ThreadState {
        var qos: UInt32 = 0
        var qosRelativePriority: Int32 = 0
        var role: Int32 = 0
        var currentPriority: Int32 = 0
        var basePriority: Int32 = 0
        var runState: Int32 = 0
        var machID: UInt64 = 0
    }

    static func threadState() -> ThreadState {
        var state = ThreadState()
        var qos = QOS_CLASS_UNSPECIFIED
        var relative: Int32 = 0
        pthread_get_qos_class_np(pthread_self(), &qos, &relative)
        state.qos = qos.rawValue
        state.qosRelativePriority = relative
        errno = 0
        state.role = getpriority(prioDarwinRole, 0)
        var threadID: UInt64 = 0
        pthread_threadid_np(nil, &threadID)
        state.machID = threadID
        var info = proc_threadinfo()
        let size = Int32(MemoryLayout<proc_threadinfo>.size)
        if proc_pidinfo(getpid(), PROC_PIDTHREADID64INFO, threadID, &info, size)
            == size
        {
            state.currentPriority = info.pth_curpri
            state.basePriority = info.pth_priority
            state.runState = info.pth_run_state
        }
        return state
    }

    // MARK: - forced policy, for the positive and negative controls

    /// `MLX_E89_FORCE_QOS=background|utility|default|userinitiated|userinteractive`.
    /// A control only. It must produce a distinct, reproducible signature, or
    /// the instrument cannot detect a state it did not cause.
    private static let forcedQoSRequest =
        ProcessInfo.processInfo.environment["MLX_E89_FORCE_QOS"]?.lowercased()

    nonisolated(unsafe) private static var forcedQoSOutcome: String?

    @discardableResult
    static func applyForcedQoS() -> String {
        if let outcome = forcedQoSOutcome { return outcome }
        guard let request = forcedQoSRequest, !request.isEmpty else {
            forcedQoSOutcome = "none"
            return "none"
        }
        let table: [String: qos_class_t] = [
            "background": QOS_CLASS_BACKGROUND,
            "utility": QOS_CLASS_UTILITY,
            "default": QOS_CLASS_DEFAULT,
            "userinitiated": QOS_CLASS_USER_INITIATED,
            "userinteractive": QOS_CLASS_USER_INTERACTIVE,
        ]
        guard let target = table[request] else {
            forcedQoSOutcome = "unknown:\(request)"
            return forcedQoSOutcome!
        }
        let rc = pthread_set_qos_class_self_np(target, 0)
        forcedQoSOutcome = rc == 0 ? request : "\(request)-failed-rc\(rc)"
        return forcedQoSOutcome!
    }

    // MARK: - trace fields

    /// One-time header so the analysis does not have to guess the units of the
    /// raw counters, and so a leg records the thread it started on.
    static func headerLine(tag: String) -> String {
        let state = threadState()
        let usage = usage()
        return "e89-probe: header tag=\(tag) "
            + "timebase_numer=\(machTimebase.numer) "
            + "timebase_denom=\(machTimebase.denom) "
            + "forced_qos=\(applyForcedQoS()) "
            + "qos=\(state.qos) qos_rel=\(state.qosRelativePriority) "
            + "role=\(state.role) curpri=\(state.currentPriority) "
            + "basepri=\(state.basePriority) tid=\(state.machID) "
            + "instr0=\(usage.instructions) cycles0=\(usage.cycles) "
            + "user_ticks0=\(usage.userTicks)\n"
    }

    /// Per-round fields. Emitted on every round, because E86 leg
    /// `e86r1-default-1` changed state at round 34 and a two-point sample at
    /// rounds 3 and 40 would have read it as uniformly slow.
    static func roundFields(
        probeNanos: UInt64, delta: Usage, legWallNanos: UInt64
    ) -> String {
        let state = threadState()
        return "e89_probe_ns=\(probeNanos) "
            + "e89_instr=\(delta.instructions) "
            + "e89_cycles=\(delta.cycles) "
            + "e89_user_ns=\(machTicksToNanos(delta.userTicks)) "
            + "e89_sys_ns=\(machTicksToNanos(delta.systemTicks)) "
            + "e89_qos_def=\(delta.qosDefault) "
            + "e89_qos_util=\(delta.qosUtility) "
            + "e89_qos_ui=\(delta.qosUserInitiated) "
            + "e89_qos_uix=\(delta.qosUserInteractive) "
            + "e89_qos_bg=\(delta.qosBackground) "
            + "e89_qos_maint=\(delta.qosMaintenance) "
            + "e89_qos=\(state.qos) e89_role=\(state.role) "
            + "e89_curpri=\(state.currentPriority) "
            + "e89_runstate=\(state.runState) e89_tid=\(state.machID) "
            + "e89_active_mb=\(Memory.activeMemory >> 20) "
            + "e89_cache_mb=\(Memory.cacheMemory >> 20) "
            + "e89_peak_mb=\(Memory.peakMemory >> 20) "
            + "e89_leg_ms=\(legWallNanos / 1_000_000) "
    }
}
