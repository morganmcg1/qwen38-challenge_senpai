"""E89: what distinguishes a leg before the scheduler demotes it?

    usage: research/e89_predecision_duty.py PREFIX [DECISION_ROUND [JSON_OUT]]

Rung A located the placement decision at round 3 or 4 of the timed window.
Rung C assumes duty controls placement. That assumption is testable with no
GPU: if duty is the trigger, legs that go on to be demoted must already show
lower duty in the rounds *before* the decision.

The comparison uses only pre-decision rounds, so the regressor cannot be
contaminated by the demotion it is meant to predict. A demoted thread runs
slower and therefore looks busier afterwards, which would invert the effect.

Instrument scopes, which are not the same and must not be divided into each
other. `Qwen36MTPHostStateProbe` takes `e89_instr` and `e89_cycles` from
`proc_pid_rusage(getpid(), ...)`, so both count the WHOLE PROCESS including
the MLX worker and completion threads. `round_thread_cpu_ns` and `e89_probe_ns`
belong to the drafting thread alone. Dividing process cycles by thread
nanoseconds produced an apparent 5.02 GHz on a part whose performance cores
stop near 4.5, which is how the error was caught.

The clock instrument here is therefore `e89_probe_ns`: the wall time of a fixed
20,000-iteration dependent integer chain that touches no memory and makes no
system call. It measures issue latency, so it tracks core clock and nothing
else. Process IPC is still reported, because numerator and denominator share
the same scope, but it describes the process rather than the thread.
"""

import glob
import itertools
import json
import os
import re
import statistics
import sys

PROBE_ITERATIONS = 20_000


def fnum(line, key):
    m = re.search(rf"(?:^|[ =]){re.escape(key)}=(-?[0-9.]+)", line)
    return float(m.group(1)) if m else None


def med(rows, field):
    vals = [x[field] for x in rows if x[field]]
    return statistics.median(vals) if vals else None


def cluster_cost(rows):
    """Per-round medians for one cluster's share of a leg's late rounds."""
    if not rows:
        return None
    return {
        "rounds": len(rows),
        "process_instructions": med(rows, "instr"),
        "process_ipc": statistics.median(
            x["instr"] / x["cycles"] for x in rows if x["cycles"]),
        "probe_ns": med(rows, "probe_ns"),
        "thread_cpu_ns": med(rows, "cpu_ns"),
        "round_us": med(rows, "round_us"),
    }


def load(prefix, decision_round):
    legs = []
    for d in sorted(glob.glob(f"research/out/{prefix}-*")):
        trace = d + "/trace.txt"
        if not os.path.isfile(trace):
            continue
        rounds = []
        for line in open(trace):
            if not line.startswith("mtp-trace: round="):
                continue
            r = fnum(line, "round")
            if r is None:
                continue
            rounds.append({
                "round": int(r),
                "core": int(fnum(line, "e89_core_a") or -1),
                "round_us": fnum(line, "round_us"),
                "cpu_ns": fnum(line, "round_thread_cpu_ns"),
                "instr": fnum(line, "e89_instr"),
                "cycles": fnum(line, "e89_cycles"),
                "probe_ns": fnum(line, "e89_probe_ns"),
            })
        if not rounds:
            continue
        rounds.sort(key=lambda x: x["round"])
        pre = [x for x in rounds if x["round"] < decision_round]
        if not pre:
            continue
        wall = sum(x["round_us"] or 0 for x in pre) * 1000.0
        cpu = sum(x["cpu_ns"] or 0 for x in pre)
        # Round 1 carries the warm path's counters, so the work comparison
        # uses only whole timed rounds after it. The probe is a self-contained
        # chain inside the round, so it stays valid in round 1.
        body = [x for x in pre if x["round"] > 1]
        instr = sum(x["instr"] or 0 for x in body)
        cycles = sum(x["cycles"] or 0 for x in body)
        demoted = next((x["round"] for x in rounds if 0 <= x["core"] <= 3), None)
        post = [x for x in rounds if x["round"] > 8]
        late = {c: [x for x in rounds if x["round"] > 8
                    and (x["core"] <= 3) == (c == "e")] for c in "ep"}
        legs.append({
            "tag": os.path.basename(d),
            "arm": os.path.basename(d)[len(prefix) + 1:].rsplit("-", 1)[0],
            "pre_rounds": len(pre),
            "pre_duty": cpu / wall if wall else None,
            "pre_cores": sorted({x["core"] for x in pre}),
            "pre_instructions": instr,
            "pre_process_ipc": instr / cycles if cycles else None,
            "pre_probe_ns": med(body, "probe_ns"),
            "demoted_at": demoted,
            "late_p_probe_ns": med(late["p"], "probe_ns"),
            "late_p_ipc": (statistics.median(
                x["instr"] / x["cycles"] for x in late["p"] if x["cycles"])
                if any(x["cycles"] for x in late["p"]) else None),
            "late_p_rounds": len(late["p"]),
            "late_by_cluster": {c: cluster_cost(v) for c, v in late.items()},
            "stuck": sum(1 for x in post if 0 <= x["core"] <= 3) / len(post) > 0.5,
        })
    return legs


def exact_one_sided(a, b):
    """Exact permutation p for 'group a has the lower mean'."""
    values = a + b
    n, na = len(values), len(a)
    total = sum(values)
    obs = sum(a) / na - (total - sum(a)) / (n - na)
    hits = draws = 0
    for idx in itertools.combinations(range(n), na):
        s = sum(values[i] for i in idx)
        draws += 1
        if s / na - (total - s) / (n - na) <= obs + 1e-12:
            hits += 1
    return {"statistic": obs, "relabelings": draws, "p_one_sided": hits / draws}


def compare(legs, field, label, unit=""):
    """Demoted legs against the rest on one pre-decision quantity."""
    a = [L[field] for L in legs if L["demoted_at"] and L[field]]
    b = [L[field] for L in legs if not L["demoted_at"] and L[field]]
    if len(a) < 2 or len(b) < 2:
        return None
    lower = exact_one_sided(a, b)
    out = {"label": label, "n_demoted": len(a), "n_rest": len(b),
           "demoted_mean": statistics.mean(a), "rest_mean": statistics.mean(b),
           "demoted_median": statistics.median(a),
           "rest_median": statistics.median(b),
           "ratio": statistics.median(a) / statistics.median(b),
           "p_demoted_lower": lower["p_one_sided"],
           "p_demoted_higher": exact_one_sided(b, a)["p_one_sided"],
           "relabelings": lower["relabelings"]}
    print(f"  {label:<34} demoted {out['demoted_median']:>14.4f}  "
          f"rest {out['rest_median']:>14.4f}  ratio {out['ratio']:6.4f}  "
          f"p(lower)={out['p_demoted_lower']:.4f} "
          f"p(higher)={out['p_demoted_higher']:.4f} {unit}")
    return out


def stall_persistence(legs):
    """Does the pre-decision slowdown stay with the leg after it recovers?

    A leg that was demoted and then returned to a performance core is its own
    control. If its late performance-core numbers match the never-demoted
    legs, the slowdown passed through; if they stay poor, it belongs to the
    leg.
    """
    print("\nIS THE SLOWDOWN A PROPERTY OF THE LEG, OR A PASSING DISTURBANCE?")
    groups = {
        "recovered": [L for L in legs if L["demoted_at"] and not L["stuck"]],
        "never demoted": [L for L in legs if not L["demoted_at"]],
        "permanently stuck": [L for L in legs if L["stuck"]],
    }
    out = {}
    print(f"  {'leg':<16} {'group':<18} {'pre_probe_ns':>12} "
          f"{'late_P_probe_ns':>16} {'late_P_rounds':>13}")
    for name, members in groups.items():
        for L in sorted(members, key=lambda x: x["tag"]):
            late = (f"{L['late_p_probe_ns']:.0f}"
                    if L["late_p_probe_ns"] else "none")
            print(f"  {L['tag']:<16} {name:<18} {L['pre_probe_ns']:>12.0f} "
                  f"{late:>16} {L['late_p_rounds']:>13}")
        vals = [L["late_p_probe_ns"] for L in members if L["late_p_probe_ns"]]
        out[name.replace(" ", "_")] = {
            "n": len(members),
            "n_with_late_performance_rounds": len(vals),
            "pre_probe_ns_median": statistics.median(
                [L["pre_probe_ns"] for L in members]) if members else None,
            "late_performance_probe_ns_median":
                statistics.median(vals) if vals else None,
        }
    for name, d in out.items():
        if d["late_performance_probe_ns_median"]:
            print(f"  {name:<18} n={d['n_with_late_performance_rounds']:<3} "
                  f"pre-decision median {d['pre_probe_ns_median']:.0f} ns  "
                  f"late performance-core median "
                  f"{d['late_performance_probe_ns_median']:.0f} ns")
    return out


def demotion_price(legs):
    """Split the demoted round's extra host CPU time into clock and the rest.

    The probe chain gives the clock ratio between the two clusters directly.
    Whatever the thread CPU time ratio exceeds it by is work the efficiency
    core does not lose to clock alone.
    """
    print("\nWHAT IS THE EFFICIENCY-CORE PENALTY MADE OF?")
    stuck = [L["late_by_cluster"]["e"] for L in legs
             if L["stuck"] and L["late_by_cluster"]["e"]]
    clean = [L["late_by_cluster"]["p"] for L in legs
             if not L["demoted_at"] and L["late_by_cluster"]["p"]]
    if not (stuck and clean):
        print("  one cluster is unrepresented; comparison skipped")
        return None
    out = {"n_stuck_legs": len(stuck), "n_clean_legs": len(clean)}
    print(f"  {'quantity':<22} {'stuck E-core':>14} {'clean P-core':>14} "
          f"{'ratio':>8}")
    for field in ("process_instructions", "process_ipc", "probe_ns",
                  "thread_cpu_ns", "round_us"):
        a = statistics.median(x[field] for x in stuck)
        b = statistics.median(x[field] for x in clean)
        out[field] = {"stuck_efficiency": a, "clean_performance": b,
                      "ratio": a / b if b else None}
        print(f"  {field:<22} {a:>14.4f} {b:>14.4f} {a / b:>8.4f}")
    clock = out["probe_ns"]["ratio"]
    cpu = out["thread_cpu_ns"]["ratio"]
    out["clock_slowdown"] = clock
    out["residual_slowdown_beyond_clock"] = cpu / clock
    for name, rows in (("stuck efficiency", stuck), ("clean performance", clean)):
        ns = statistics.median(x["probe_ns"] for x in rows) / PROBE_ITERATIONS
        out[f"{name.split()[0]}_ns_per_chain_iteration"] = ns
        print(f"  {name:<22} {ns:.4f} ns per dependent chain iteration")
    print(f"  thread CPU time ratio {cpu:.4f} = clock {clock:.4f} "
          f"times {cpu / clock:.4f} of everything else")
    return out


def main():
    prefix = sys.argv[1] if len(sys.argv) > 1 else "e89rA"
    decision = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    out_json = sys.argv[3] if len(sys.argv) > 3 else None

    legs = load(prefix, decision)
    doc = {"prefix": prefix, "decision_round": decision,
           "question": "what distinguishes a leg before the scheduler demotes it",
           "probe_iterations": PROBE_ITERATIONS,
           "process_scope_fields": ["e89_instr", "e89_cycles"],
           "thread_scope_fields": ["round_thread_cpu_ns", "e89_probe_ns"],
           "harness": "local", "sandbox": "off",
           "official_or_ranked_score": False}

    print(f"pre-decision window: rounds 1 to {decision - 1}\n")
    print(f"{'leg':<16} {'stuck':>6} {'demoted':>8} {'duty':>7} "
          f"{'probe_ns':>9} {'instr_M':>8} {'procIPC':>8}  {'pre_cores'}")
    for L in sorted(legs, key=lambda x: (x["demoted_at"] is None, x["tag"])):
        print(f"{L['tag']:<16} {str(L['stuck']):>6} "
              f"{str(L['demoted_at']):>8} {L['pre_duty']:>7.4f} "
              f"{L['pre_probe_ns']:>9.0f} {L['pre_instructions'] / 1e6:>8.1f} "
              f"{L['pre_process_ipc']:>8.3f}  {L['pre_cores']}")

    for name, hit in (("permanently stuck", lambda L: L["stuck"]),
                      ("ever demoted", lambda L: L["demoted_at"] is not None)):
        a = [L["pre_duty"] for L in legs if hit(L)]
        b = [L["pre_duty"] for L in legs if not hit(L)]
        print(f"\nPRE-DECISION DUTY, {name} against the rest")
        if not (a and b):
            print("  one group is empty; test skipped")
            continue
        key = name.replace(" ", "_")
        doc[key] = {
            "n_hit": len(a), "n_rest": len(b),
            "hit_mean_duty": statistics.mean(a),
            "rest_mean_duty": statistics.mean(b),
            "hit_values": a,
            "test_hit_lower": exact_one_sided(a, b),
            "rest_legs_at_or_below_highest_hit":
                sum(1 for v in b if v <= max(a)),
        }
        t = doc[key]["test_hit_lower"]
        print(f"  {name:<18} n={len(a)} mean {statistics.mean(a):.4f} "
              f"values {[round(v, 4) for v in a]}")
        print(f"  {'the rest':<18} n={len(b)} mean {statistics.mean(b):.4f} "
              f"min {min(b):.4f} max {max(b):.4f}")
        print(f"  difference {t['statistic']:+.4f}  "
              f"exact one-sided p={t['p_one_sided']:.4f} for "
              f"'{name} has the lower duty' over {t['relabelings']} relabelings")
        print(f"  legs in the rest at or below the highest {name} leg: "
              f"{doc[key]['rest_legs_at_or_below_highest_hit']} of {len(b)}")

    print("\nWAS THE DEMOTED LEG ALREADY SLOWER, AND WAS IT DOING MORE WORK?")
    doc["pre_decision"] = {
        "probe_ns": compare(legs, "pre_probe_ns",
                            "dependent-chain time, thread", "[clock, ns]"),
        "instructions": compare(legs, "pre_instructions",
                                "retired instructions, process", "[work]"),
        "process_ipc": compare(legs, "pre_process_ipc",
                               "instructions per cycle, process", "[process]"),
        "duty": compare(legs, "pre_duty", "thread duty", "[thread]"),
    }
    doc["stall_persistence"] = stall_persistence(legs)
    doc["demotion_price"] = demotion_price(legs)

    if out_json:
        json.dump(doc, open(out_json, "w"), indent=2, sort_keys=True)
        print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
