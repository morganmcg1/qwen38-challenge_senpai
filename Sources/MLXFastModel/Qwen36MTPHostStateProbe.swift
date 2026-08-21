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

    /// THE INSTRUMENT IS NOT PASSIVE. With every component on, a probed leg
    /// sits at about 2,900 us of host phase sum per round while the same
    /// worker binary with `MLX_E89_PROBE=0` sits at about 570 us, which is the
    /// clean level. That is the signature the probe was built to observe, so
    /// the probe must be decomposed before any of its readings mean anything.
    ///
    /// `MLX_E89_PARTS` selects components from `marks`, `probe`, `rusage`,
    /// `thread` and `mem`. Unset means every component, which is the
    /// state-inducing configuration. Each component alone is one ablation arm.
    static let parts: Set<String> = {
        guard let raw = ProcessInfo.processInfo.environment["MLX_E89_PARTS"],
              !raw.isEmpty
        else { return ["marks", "probe", "rusage", "thread", "mem"] }
        return Set(raw.lowercased().split(separator: ",").map(String.init))
    }()

    @inline(__always)
    static func on(_ part: String) -> Bool { enabled && parts.contains(part) }

    /// The MLX allocator counters take the allocator lock, so an ablation arm
    /// that excludes `mem` must not evaluate them at all.
    @inline(__always)
    private static func memoryMB(_ value: @autoclosure () -> Int) -> Int {
        on("mem") ? value() >> 20 : 0
    }

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

    // MARK: - thread CPU time

    /// Nanoseconds this THREAD has spent on-core. Bracketing a phase with two
    /// marks gives `cpu / wall` for that phase, which separates "the thread is
    /// running slowly" (cpu inflates with wall) from "the thread is not
    /// running" (cpu flat while wall inflates).
    @inline(__always)
    static func cpuMark() -> UInt64 {
        on("marks") ? clock_gettime_nsec_np(CLOCK_THREAD_CPUTIME_ID) : 0
    }

    /// Wall cost of the twelve `cpuMark` calls a traced round adds, measured
    /// on the same thread in the same round, so the analysis can subtract the
    /// instrument from the phase totals instead of assuming it is free.
    static func cpuMarkOverheadNanos() -> UInt64 {
        guard on("marks") else { return 0 }
        let t0 = DispatchTime.now().uptimeNanoseconds
        for _ in 0 ..< 12 { overheadSink &+= clock_gettime_nsec_np(CLOCK_THREAD_CPUTIME_ID) }
        return DispatchTime.now().uptimeNanoseconds - t0
    }

    nonisolated(unsafe) static var overheadSink: UInt64 = 0

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
        var pageins: UInt64 = 0
        var voluntarySwitches: UInt64 = 0
        var involuntarySwitches: UInt64 = 0
        var minorFaults: UInt64 = 0
        var majorFaults: UInt64 = 0
        var compressions: UInt64 = 0
        var decompressions: UInt64 = 0
        var swapins: UInt64 = 0
        var footprintBytes: UInt64 = 0
        var compressedBytes: UInt64 = 0
        var hostFreePages: UInt64 = 0
        var hostCompressorPages: UInt64 = 0
        var hostPageins: UInt64 = 0
        var hostDecompressions: UInt64 = 0
        var hostSwapins: UInt64 = 0

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
                qosMaintenance: lhs.qosMaintenance &- rhs.qosMaintenance,
                pageins: lhs.pageins &- rhs.pageins,
                voluntarySwitches: lhs.voluntarySwitches
                    &- rhs.voluntarySwitches,
                involuntarySwitches: lhs.involuntarySwitches
                    &- rhs.involuntarySwitches,
                minorFaults: lhs.minorFaults &- rhs.minorFaults,
                majorFaults: lhs.majorFaults &- rhs.majorFaults,
                compressions: lhs.compressions &- rhs.compressions,
                decompressions: lhs.decompressions &- rhs.decompressions,
                swapins: lhs.swapins &- rhs.swapins,
                // Levels, not counters: report the value at the end of the round.
                footprintBytes: lhs.footprintBytes,
                compressedBytes: lhs.compressedBytes,
                hostFreePages: lhs.hostFreePages,
                hostCompressorPages: lhs.hostCompressorPages,
                hostPageins: lhs.hostPageins &- rhs.hostPageins,
                hostDecompressions: lhs.hostDecompressions
                    &- rhs.hostDecompressions,
                hostSwapins: lhs.hostSwapins &- rhs.hostSwapins)
        }
    }

    static func usage() -> Usage {
        guard on("rusage") else { return Usage() }
        var out = Usage()

        var info = rusage_info_v4()
        let rc = withUnsafeMutablePointer(to: &info) { pointer -> Int32 in
            pointer.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) {
                proc_pid_rusage(getpid(), RUSAGE_INFO_V4, $0)
            }
        }
        guard rc == 0 else { return out }
        out.instructions = info.ri_instructions
        out.cycles = info.ri_cycles
        out.userTicks = info.ri_user_time
        out.systemTicks = info.ri_system_time
        out.qosDefault = info.ri_cpu_time_qos_default
        out.qosUtility = info.ri_cpu_time_qos_utility
        out.qosUserInitiated = info.ri_cpu_time_qos_user_initiated
        out.qosUserInteractive = info.ri_cpu_time_qos_user_interactive
        out.qosBackground = info.ri_cpu_time_qos_background
        out.qosMaintenance = info.ri_cpu_time_qos_maintenance
        out.pageins = info.ri_pageins

        // Voluntary and involuntary switches separate the two scheduling
        // stories. A thread that waits on the GPU raises `ru_nvcsw`. A thread
        // that the scheduler takes off core raises `ru_nivcsw`. Pooling them
        // would hide exactly the distinction the round needs.
        var ru = rusage()
        if getrusage(RUSAGE_SELF, &ru) == 0 {
            out.voluntarySwitches = UInt64(bitPattern: Int64(ru.ru_nvcsw))
            out.involuntarySwitches = UInt64(bitPattern: Int64(ru.ru_nivcsw))
            out.minorFaults = UInt64(bitPattern: Int64(ru.ru_minflt))
            out.majorFaults = UInt64(bitPattern: Int64(ru.ru_majflt))
        }

        var vm = task_vm_info_data_t()
        var vmCount = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let vmrc = withUnsafeMutablePointer(to: &vm) { pointer -> kern_return_t in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(vmCount)) {
                task_info(
                    mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &vmCount)
            }
        }
        if vmrc == KERN_SUCCESS {
            // `task_vm_info` has no compression or swap-out counter. It does
            // carry a lifetime compressed-byte total, which rises by exactly
            // the bytes the compressor took in, so it serves as the counter.
            out.compressions = vm.compressed_lifetime
            out.decompressions = UInt64(bitPattern: Int64(vm.decompressions))
            out.swapins = UInt64(bitPattern: vm.ledger_swapins)
            out.footprintBytes = vm.phys_footprint
            out.compressedBytes = vm.compressed
        }

        // System-wide memory pressure. This is the direct replacement for the
        // refused two-model-holder arm: it observes the whole machine without
        // adding a second 27 GB resident process.
        var stats = vm_statistics64_data_t()
        var statCount = mach_msg_type_number_t(
            MemoryLayout<vm_statistics64_data_t>.size / MemoryLayout<integer_t>.size)
        let hostrc = withUnsafeMutablePointer(to: &stats) { pointer -> kern_return_t in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(statCount)) {
                host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &statCount)
            }
        }
        if hostrc == KERN_SUCCESS {
            out.hostFreePages = UInt64(stats.free_count)
            out.hostCompressorPages = UInt64(stats.compressor_page_count)
            out.hostPageins = stats.pageins
            out.hostDecompressions = UInt64(stats.decompressions)
            out.hostSwapins = stats.swapins
        }
        return out
    }

    // MARK: - dependent-chain CPU speed probe

    /// Written so the optimizer cannot delete the loop. Never read.
    nonisolated(unsafe) static var probeSink: UInt64 = 0

    /// About 39 us on a fast host thread and about 79 us on a slow one, with
    /// nothing in between, measured on this M4 Pro. 20,000 iterations cost
    /// under 0.05 % of a 13 second leg.
    @inline(never)
    static func cpuProbeNanos() -> UInt64 {
        guard on("probe") else { return 0 }
        let t0 = DispatchTime.now().uptimeNanoseconds
        var x: UInt64 = 0x9E37_79B9_7F4A_7C15
        for _ in 0 ..< 20_000 {
            x = x &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            x ^= x >> 31
        }
        probeSink = x
        return DispatchTime.now().uptimeNanoseconds - t0
    }

    /// The chain above runs a fixed and known instruction stream, so measuring
    /// cycles across it converts a wall time into a core clock. This separates
    /// the two ways a round can lose time: a lower clock at the same work per
    /// cycle, or the same clock at a lower work per cycle.
    struct ProbeClock {
        var nanos: UInt64 = 0
        var spanNanos: UInt64 = 0
        var cycles: UInt64 = 0
        var instructions: UInt64 = 0
    }

    /// `nanos` covers the chain alone and stays comparable with earlier rungs.
    /// `spanNanos` covers exactly the window the cycle and instruction deltas
    /// cover, which is the chain plus the two counter reads, so the derived
    /// clock is `cycles / spanNanos` with no unaccounted work between them.
    @inline(never)
    static func cpuProbeClock() -> ProbeClock {
        guard on("probe") else { return ProbeClock() }
        let t0 = DispatchTime.now().uptimeNanoseconds
        let before = usage()
        let nanos = cpuProbeNanos()
        let after = usage()
        let t1 = DispatchTime.now().uptimeNanoseconds
        return ProbeClock(
            nanos: nanos,
            spanNanos: t1 &- t0,
            cycles: after.cycles &- before.cycles,
            instructions: after.instructions &- before.instructions)
    }

    // MARK: - core identity

    /// The logical CPU the calling thread runs on right now. A move between
    /// the performance and efficiency clusters changes the clock and the work
    /// per cycle together, so the core number tells the two apart directly.
    @inline(__always)
    static func coreNumber() -> Int {
        guard on("thread") else { return -1 }
        var cpu = 0
        return pthread_cpu_number_np(&cpu) == 0 ? cpu : -1
    }

    static let performanceCoreCount: Int = {
        var value: Int32 = 0
        var size = MemoryLayout<Int32>.size
        if sysctlbyname("hw.perflevel0.logicalcpu", &value, &size, nil, 0) == 0 {
            return Int(value)
        }
        return -1
    }()

    static let efficiencyCoreCount: Int = {
        var value: Int32 = 0
        var size = MemoryLayout<Int32>.size
        if sysctlbyname("hw.perflevel1.logicalcpu", &value, &size, nil, 0) == 0 {
            return Int(value)
        }
        return -1
    }()

    // MARK: - thread policy

    /// `proc_pid_rusage` reports the WHOLE PROCESS, and the MLX worker and
    /// completion threads contribute to it. `pth_user_time` and
    /// `pth_system_time` are the calling thread's own CPU nanoseconds, which
    /// is what separates a thread that runs slowly from a thread that blocks:
    /// a slow thread burns CPU for the whole inflated phase, a blocked thread
    /// does not.
    struct ThreadState {
        var qos: UInt32 = 0
        var qosRelativePriority: Int32 = 0
        var role: Int32 = 0
        var currentPriority: Int32 = 0
        var basePriority: Int32 = 0
        var runState: Int32 = 0
        var machID: UInt64 = 0
        var userNanos: UInt64 = 0
        var systemNanos: UInt64 = 0
        var cpuUsage: Int32 = 0
    }

    static func threadState() -> ThreadState {
        guard on("thread") else { return ThreadState() }
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
            state.userNanos = info.pth_user_time
            state.systemNanos = info.pth_system_time
            state.cpuUsage = info.pth_cpu_usage
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
            + "parts=\(parts.sorted().joined(separator: "+")) "
            + "forced_qos=\(applyForcedQoS()) "
            + "qos=\(state.qos) qos_rel=\(state.qosRelativePriority) "
            + "role=\(state.role) curpri=\(state.currentPriority) "
            + "basepri=\(state.basePriority) tid=\(state.machID) "
            + "instr0=\(usage.instructions) cycles0=\(usage.cycles) "
            + "user_ticks0=\(usage.userTicks) "
            + "pcores=\(performanceCoreCount) ecores=\(efficiencyCoreCount) "
            + "core0=\(coreNumber())\n"
    }

    /// Per-round fields. Emitted on every round, because E86 leg
    /// `e86r1-default-1` changed state at round 34 and a two-point sample at
    /// rounds 3 and 40 would have read it as uniformly slow.
    static func roundFields(
        probe: ProbeClock, delta: Usage, legWallNanos: UInt64,
        threadStart: ThreadState, coreStart: Int, coreEnd: Int
    ) -> String {
        let state = threadState()
        return "e89_thr_user_ns=\(state.userNanos &- threadStart.userNanos) "
            + "e89_thr_sys_ns=\(state.systemNanos &- threadStart.systemNanos) "
            + "e89_thr_cpu=\(state.cpuUsage) "
            + "e89_core_a=\(coreStart) e89_core_b=\(coreEnd) "
            + "e89_probe_ns=\(probe.nanos) "
            + "e89_probe_span_ns=\(probe.spanNanos) "
            + "e89_probe_cyc=\(probe.cycles) "
            + "e89_probe_ins=\(probe.instructions) "
            + "e89_nvcsw=\(delta.voluntarySwitches) "
            + "e89_nivcsw=\(delta.involuntarySwitches) "
            + "e89_minflt=\(delta.minorFaults) "
            + "e89_majflt=\(delta.majorFaults) "
            + "e89_pageins=\(delta.pageins) "
            + "e89_vm_comp=\(delta.compressions) "
            + "e89_vm_decomp=\(delta.decompressions) "
            + "e89_vm_swapin=\(delta.swapins) "
            + "e89_vm_footprint_mb=\(delta.footprintBytes >> 20) "
            + "e89_vm_compressed_mb=\(delta.compressedBytes >> 20) "
            + "e89_host_free_mb=\((delta.hostFreePages &* 16384) >> 20) "
            + "e89_host_compressor_mb=\((delta.hostCompressorPages &* 16384) >> 20) "
            + "e89_host_pageins=\(delta.hostPageins) "
            + "e89_host_decomp=\(delta.hostDecompressions) "
            + "e89_host_swapin=\(delta.hostSwapins) "
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
            + "e89_active_mb=\(memoryMB(Memory.activeMemory)) "
            + "e89_cache_mb=\(memoryMB(Memory.cacheMemory)) "
            + "e89_peak_mb=\(memoryMB(Memory.peakMemory)) "
            + "e89_leg_ms=\(legWallNanos / 1_000_000) "
    }
}
