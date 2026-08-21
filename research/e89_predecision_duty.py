"""E89: is thread duty before the demotion round what selects the cluster?

    usage: research/e89_predecision_duty.py PREFIX [DECISION_ROUND [JSON_OUT]]

Rung A located the placement decision at round 3 or 4 of the timed window.
Rung C assumes duty controls placement. That assumption is testable with no
GPU: if duty is the trigger, legs that go on to be demoted must already show
lower duty in the rounds *before* the decision.

The comparison uses only pre-decision rounds, so the regressor cannot be
contaminated by the demotion it is meant to predict. A demoted thread runs
slower and therefore looks busier afterwards, which would invert the effect.
"""

import glob
import itertools
import json
import os
import re
import statistics
import sys


def fnum(line, key):
    m = re.search(rf"(?:^|[ =]){re.escape(key)}=(-?[0-9.]+)", line)
    return float(m.group(1)) if m else None


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
                "user_ns": fnum(line, "e89_thr_user_ns"),
                "sys_ns": fnum(line, "e89_thr_sys_ns"),
                "instr": fnum(line, "e89_instr"),
                "cycles": fnum(line, "e89_cycles"),
            })
        if not rounds:
            continue
        rounds.sort(key=lambda x: x["round"])
        pre = [x for x in rounds if x["round"] < decision_round]
        if not pre:
            continue
        wall = sum(x["round_us"] or 0 for x in pre) * 1000.0
        cpu = sum(x["cpu_ns"] or 0 for x in pre)
        user = sum(x["user_ns"] or 0 for x in pre)
        # Round 1 carries the warm path's counters, so the work comparison
        # uses only whole timed rounds after it.
        body = [x for x in pre if x["round"] > 1]
        instr = sum(x["instr"] or 0 for x in body)
        cycles = sum(x["cycles"] or 0 for x in body)
        body_cpu = sum(x["cpu_ns"] or 0 for x in body)
        demoted = next((x["round"] for x in rounds if 0 <= x["core"] <= 3), None)
        post = [x for x in rounds if x["round"] > 8]
        legs.append({
            "tag": os.path.basename(d),
            "arm": os.path.basename(d)[len(prefix) + 1:].rsplit("-", 1)[0],
            "pre_rounds": len(pre),
            "pre_duty": cpu / wall if wall else None,
            "pre_user_duty": user / wall if wall else None,
            "pre_wall_us": wall / 1000.0,
            "pre_cores": sorted({x["core"] for x in pre}),
            "pre_instructions": instr,
            "pre_cycles": cycles,
            "pre_ipc": instr / cycles if cycles else None,
            "pre_clock_ghz": cycles / body_cpu if body_cpu else None,
            "demoted_at": demoted,
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


def work_or_speed(legs, label):
    """Split the pre-decision cost of the demoted legs into work and speed."""
    hit = [L for L in legs if L["demoted_at"] is not None]
    rest = [L for L in legs if L["demoted_at"] is None]
    out = {"label": label, "n_demoted": len(hit), "n_rest": len(rest),
           "legs": [L["tag"] for L in legs]}
    print(f"\n  {label}: {len(hit)} demoted against {len(rest)}")
    if len(hit) < 2 or len(rest) < 2:
        print("    too few legs on one side; test skipped")
        return out
    for field, unit in (("pre_instructions", "instructions"),
                        ("pre_cycles", "cycles"),
                        ("pre_ipc", "instructions per cycle"),
                        ("pre_clock_ghz", "GHz")):
        a = [L[field] for L in hit if L[field]]
        b = [L[field] for L in rest if L[field]]
        ma, mb = statistics.median(a), statistics.median(b)
        out[field] = {"demoted_median": ma, "rest_median": mb,
                      "ratio": ma / mb if mb else None,
                      "test_demoted_lower": exact_one_sided(a, b)}
        t = out[field]["test_demoted_lower"]
        print(f"    {field:<18} demoted {ma:>16.4f}  rest {mb:>16.4f}  "
              f"ratio {ma / mb:6.4f}  p(demoted lower)={t['p_one_sided']:.4f}"
              f"  [{unit}]")
    return out


def main():
    prefix = sys.argv[1] if len(sys.argv) > 1 else "e89rA"
    decision = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    out_json = sys.argv[3] if len(sys.argv) > 3 else None

    legs = load(prefix, decision)
    doc = {"prefix": prefix, "decision_round": decision,
           "question": "does duty before the demotion round predict demotion",
           "harness": "local", "sandbox": "off",
           "official_or_ranked_score": False}

    print(f"pre-decision window: rounds 1 to {decision - 1}\n")
    print(f"{'leg':<16} {'stuck':>6} {'demoted':>8} {'duty':>7} "
          f"{'instr_M':>8} {'IPC':>6} {'GHz':>5}  {'pre_cores'}")
    for L in sorted(legs, key=lambda x: (x["demoted_at"] is None, x["tag"])):
        print(f"{L['tag']:<16} {str(L['stuck']):>6} "
              f"{str(L['demoted_at']):>8} {L['pre_duty']:>7.4f} "
              f"{L['pre_instructions'] / 1e6:>8.1f} {L['pre_ipc']:>6.3f} "
              f"{L['pre_clock_ghz']:>5.2f}  {L['pre_cores']}")

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

    print("\nIS THE HIGHER DUTY MORE WORK, OR SLOWER WORK?")
    print("  duty is CPU time over wall time, so a thread that runs slower "
          "looks busier\n  for identical work. Instructions settle it.")
    doc["work_or_speed"] = work_or_speed(legs, "all legs")
    doc["work_or_speed_cluster2_only"] = work_or_speed(
        [L for L in legs if min(L["pre_cores"]) > 8],
        "sensitivity: legs confined to performance cores 9 to 13")

    if out_json:
        json.dump(doc, open(out_json, "w"), indent=2, sort_keys=True)
        print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
