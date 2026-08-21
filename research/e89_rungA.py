"""E89 rung A: does a 200 ms warm-path spin latch the performance cluster?

    usage: research/e89_rungA.py PREFIX [WARMUP [JSON_OUT]]

The advisor fixed the quarantine before the numbers were seen: legs that
overlap the launching agent's own shell activity are dropped, and the
remaining legs are primary. The quarantine is defined by a timestamped
mechanism, host contention, and not by any outcome.

Prevalence is continuous when legs do not latch, so the arm contrast is a
Wilcoxon rank-sum with an exact permutation p-value over leg relabelings.
"""

import datetime
import glob
import itertools
import json
import os
import re
import statistics
import sys

HOST = ["d_pre_us", "d_flush_us", "d_head1_us", "d_submit1_us",
        "d_chain_us", "readout_us", "commit_us", "upkeep_us"]

SLOW_CUT_US = 1246.0  # frozen by the rung-2 session so the label is comparable
QUARANTINE = {"e89rA-ctl-1", "e89rA-spin-1", "e89rA-spin-2"}


def fnum(line, key):
    m = re.search(rf"(?:^|[ =]){re.escape(key)}=(-?[0-9.]+)", line)
    return float(m.group(1)) if m else None


def load(prefix, warmup):
    legs = []
    for d in sorted(glob.glob(f"research/out/{prefix}-*")):
        trace = d + "/trace.txt"
        if not os.path.isfile(trace):
            continue
        rows, spins, every = [], [], []
        for line in open(trace):
            m = re.search(r"warm_spin=ms=(\d+),blocks=(\d+)", line)
            if m:
                spins.append((int(m.group(1)), int(m.group(2))))
            if not line.startswith("mtp-trace: round="):
                continue
            r = fnum(line, "round")
            if r is None:
                continue
            core = fnum(line, "e89_core_a")
            host = sum(fnum(line, k) or 0 for k in HOST)
            every.append((int(r), int(core) if core is not None else -1, host))
            if r <= warmup:
                continue
            rows.append({"host": host, "core": core,
                         "round_us": fnum(line, "round_us")})
        if not rows:
            continue
        every.sort()
        tag = os.path.basename(d)
        metrics = json.load(open(d + "/score.json"))["metrics"]
        legs.append({
            "tag": tag,
            "arm": tag[len(prefix) + 1:].rsplit("-", 1)[0],
            "start": os.stat(d).st_birthtime,
            "n": len(rows),
            "prev": sum(1 for r in rows if r["host"] > SLOW_CUT_US) / len(rows),
            "ecore": sum(1 for r in rows
                         if r["core"] is not None and r["core"] <= 3) / len(rows),
            "cores": sorted({int(r["core"]) for r in rows if r["core"] is not None}),
            "median_host_us": statistics.median(r["host"] for r in rows),
            "spt": metrics["mtp_seconds_per_token"],
            "matched": metrics.get("all_tokens_matched"),
            "draftlen": metrics.get("effective_mean_draft_len"),
            "blocks_per_ms": [b / ms for ms, b in spins if ms],
            "spin_ms": sorted({ms for ms, _ in spins}),
            "round1_core": every[0][1],
            "first_efficiency_round": next(
                (r for r, c, _ in every if 0 <= c <= 3), None),
            "efficiency_rounds": sum(1 for _, c, _ in every if 0 <= c <= 3),
            "total_rounds": len(every),
        })
    legs.sort(key=lambda L: L["start"])
    return legs


def ranksum(a, b):
    """Wilcoxon rank-sum statistic for `a`, with midranks for ties."""
    pooled = sorted(a + b)
    ranks = {}
    i = 0
    while i < len(pooled):
        j = i
        while j + 1 < len(pooled) and pooled[j + 1] == pooled[i]:
            j += 1
        for v in {pooled[i]}:
            ranks[v] = (i + j) / 2 + 1
        i = j + 1
    return sum(ranks[v] for v in a)


def exact_perm(a, b):
    """Exact two-sided permutation p for the rank-sum, over all relabelings."""
    values = a + b
    n, na = len(values), len(a)
    obs = ranksum(a, b)
    centre = na * (n + 1) / 2
    hits = draws = 0
    for idx in itertools.combinations(range(n), na):
        sub = [values[i] for i in idx]
        rest = [values[i] for i in range(n) if i not in set(idx)]
        draws += 1
        if abs(ranksum(sub, rest) - centre) >= abs(obs - centre) - 1e-12:
            hits += 1
    return {"ranksum": obs, "null_mean": centre,
            "relabelings": draws, "p_two_sided": hits / draws}


def mean_diff_perm(a, b):
    """Exact one-sided permutation p for 'a has the lower mean'."""
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


def contrast(legs, label):
    ctl = [L["prev"] for L in legs if L["arm"] == "ctl"]
    spin = [L["prev"] for L in legs if L["arm"] == "spin"]
    out = {
        "label": label,
        "n_ctl": len(ctl), "n_spin": len(spin),
        "ctl_prevalence_mean": statistics.mean(ctl),
        "spin_prevalence_mean": statistics.mean(spin),
        "ctl_prevalence_median": statistics.median(ctl),
        "spin_prevalence_median": statistics.median(spin),
        "ranksum_exact": exact_perm(spin, ctl),
        "mean_diff_exact": mean_diff_perm(spin, ctl),
        "ctl_stuck_legs": sum(1 for v in ctl if v > 0.5),
        "spin_stuck_legs": sum(1 for v in spin if v > 0.5),
        "ctl_clean_legs": sum(1 for v in ctl if v < 0.10),
        "spin_clean_legs": sum(1 for v in spin if v < 0.10),
    }
    print(f"\n=== ARM CONTRAST, {label} ===")
    print(f"  ctl  n={len(ctl)}  prevalence mean {statistics.mean(ctl):.4f}  "
          f"median {statistics.median(ctl):.4f}  "
          f"stuck {out['ctl_stuck_legs']}  clean {out['ctl_clean_legs']}")
    print(f"  spin n={len(spin)}  prevalence mean {statistics.mean(spin):.4f}  "
          f"median {statistics.median(spin):.4f}  "
          f"stuck {out['spin_stuck_legs']}  clean {out['spin_clean_legs']}")
    r, m = out["ranksum_exact"], out["mean_diff_exact"]
    print(f"  rank-sum W={r['ranksum']:.1f} null {r['null_mean']:.1f}  "
          f"exact two-sided p={r['p_two_sided']:.4f} over {r['relabelings']}")
    print(f"  spin minus ctl mean {m['statistic']:+.4f}  "
          f"exact one-sided p={m['p_one_sided']:.4f}")
    return out


def main():
    prefix = sys.argv[1] if len(sys.argv) > 1 else "e89rA"
    warmup = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    out_json = sys.argv[3] if len(sys.argv) > 3 else None

    legs = load(prefix, warmup)
    doc = {"prefix": prefix, "warmup_rounds": warmup, "decode_tokens": 512,
           "slow_cut_us": SLOW_CUT_US, "harness": "local", "sandbox": "off",
           "cool_gate_passed_real_gate": False,
           "gate_qualified_for_timing": False,
           "official_or_ranked_score": False,
           "quarantined_legs": sorted(QUARANTINE),
           "quarantine_reason": "overlapped the agent's own shell activity, "
                                "fixed before the numbers were seen"}

    print("harness=local, sandbox=off, ungated, "
          "cool_gate_passed_real_gate=false, gate_qualified_for_timing=false, "
          "official_or_ranked_score=false")
    print(f"slow cut {SLOW_CUT_US:.0f} us, frozen from rung 2\n")
    print(f"{'pos':>3} {'leg':<16} {'start':<9} {'n':>3} {'prev':>6} "
          f"{'ecore':>6} {'medhost':>8} {'s/token':>10} {'blocks/ms':>18} Q")
    doc["legs"] = []
    for i, L in enumerate(legs):
        ts = datetime.datetime.fromtimestamp(L["start"], datetime.UTC).strftime("%H:%M:%S")
        bp = ",".join(f"{v:.1f}" for v in L["blocks_per_ms"]) or "-"
        q = "Q" if L["tag"] in QUARANTINE else ""
        print(f"{i:>3} {L['tag']:<16} {ts:<9} {L['n']:>3} {L['prev']:>6.3f} "
              f"{L['ecore']:>6.3f} {L['median_host_us']:>8.0f} "
              f"{L['spt']:>10.7f} {bp:>18} {q}")
        doc["legs"].append({k: L[k] for k in
                            ("tag", "arm", "n", "prev", "ecore", "cores",
                             "median_host_us", "spt", "matched", "draftlen",
                             "blocks_per_ms", "spin_ms")})

    kept = [L for L in legs if L["tag"] not in QUARANTINE]
    doc["primary"] = contrast(kept, "PRIMARY, quarantine applied")
    doc["secondary"] = contrast(legs, "SECONDARY, all legs")

    print("\n=== WHERE THE CLUSTER IS DECIDED ===")
    on_p = [L for L in legs if L["round1_core"] > 3]
    stuck_legs = [L for L in legs if L["prev"] > 0.9]
    doc["placement_decision"] = {
        "legs_starting_on_performance_core": len(on_p),
        "legs_total": len(legs),
        "stuck_legs": [L["tag"] for L in stuck_legs],
        "stuck_first_efficiency_round": [L["first_efficiency_round"]
                                         for L in stuck_legs],
        "stuck_round1_core": [L["round1_core"] for L in stuck_legs],
        "clean_legs_that_ever_migrated": sum(
            1 for L in legs
            if L["prev"] < 0.10 and L["first_efficiency_round"] is not None),
    }
    print(f"  round 1 ran on a performance core in "
          f"{len(on_p)} of {len(legs)} legs")
    for L in stuck_legs:
        print(f"  stuck {L['tag']:<16} round1_core={L['round1_core']} "
              f"first_efficiency_round={L['first_efficiency_round']} "
              f"efficiency_rounds={L['efficiency_rounds']}/{L['total_rounds']}")
    print(f"  legs that stayed clean and never touched an efficiency core: "
          f"{sum(1 for L in legs if L['prev'] < 0.10 and L['efficiency_rounds'] == 0)}"
          f" of {sum(1 for L in legs if L['prev'] < 0.10)}")

    print("\n=== CONTAMINATION CHECK, warm-thread clock in quarantined legs ===")
    qs = [v for L in legs if L["tag"] in QUARANTINE for v in L["blocks_per_ms"]]
    ks = [v for L in kept for v in L["blocks_per_ms"]]
    if qs and ks:
        doc["contamination_check"] = {
            "quarantined_n": len(qs), "quarantined_median": statistics.median(qs),
            "kept_n": len(ks), "kept_median": statistics.median(ks),
            "ratio": statistics.median(qs) / statistics.median(ks)}
        print(f"  quarantined n={len(qs)} median {statistics.median(qs):.1f} "
              f"blocks/ms  min {min(qs):.1f}")
        print(f"  kept        n={len(ks)} median {statistics.median(ks):.1f} "
              f"blocks/ms  min {min(ks):.1f}")
        print(f"  ratio {statistics.median(qs) / statistics.median(ks):.4f}")
    else:
        print("  no spun legs on one side; check skipped")

    print("\n=== ABSOLUTE s/token, CLEAN LEGS ONLY, prevalence below 0.10 ===")
    doc["clean_leg_spt"] = {}
    for arm in ("ctl", "spin"):
        g = [L["spt"] for L in kept if L["arm"] == arm and L["prev"] < 0.10]
        if not g:
            print(f"  {arm:<5} no clean leg")
            continue
        sd = statistics.stdev(g) if len(g) > 1 else 0.0
        doc["clean_leg_spt"][arm] = {"n": len(g), "mean": statistics.mean(g),
                                     "sd": sd}
        print(f"  {arm:<5} n={len(g)}  mean {statistics.mean(g):.7f}  sd {sd:.7f}")
    if {"ctl", "spin"} <= set(doc["clean_leg_spt"]):
        c, s = (doc["clean_leg_spt"][a]["mean"] for a in ("ctl", "spin"))
        doc["clean_leg_spin_minus_ctl_pct"] = 100 * (s - c) / c
        print(f"  spin minus ctl {s - c:+.7f} ({100 * (s - c) / c:+.3f} %)")

    print("\n=== PRICE OF THE STATE ITSELF, arms pooled, quarantine applied ===")
    clean = [L["spt"] for L in kept if L["prev"] < 0.10]
    stuck = [L["spt"] for L in kept if L["prev"] > 0.90]
    if clean:
        doc["price"] = {"clean_n": len(clean), "clean_mean": statistics.mean(clean)}
        print(f"  clean n={len(clean)} mean {statistics.mean(clean):.7f}")
        if stuck:
            pen = 100 * (statistics.mean(stuck) - statistics.mean(clean)) \
                / statistics.mean(clean)
            doc["price"].update({"stuck_n": len(stuck),
                                 "stuck_mean": statistics.mean(stuck),
                                 "stuck_penalty_pct": pen})
            print(f"  stuck n={len(stuck)} mean {statistics.mean(stuck):.7f} "
                  f"penalty {pen:+.3f} %")
        else:
            print("  no fully stuck leg after quarantine")

    print("\n=== CORRECTNESS ===")
    matched = sum(1 for L in legs if L["matched"])
    lens = {L["draftlen"] for L in legs}
    doc["correctness"] = {"matched_legs": matched, "total_legs": len(legs),
                          "distinct_draft_lengths": len(lens)}
    print(f"  all_tokens_matched true on {matched} of {len(legs)} legs; "
          f"{len(lens)} distinct draft lengths")

    if out_json:
        json.dump(doc, open(out_json, "w"), indent=2, sort_keys=True)
        print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
