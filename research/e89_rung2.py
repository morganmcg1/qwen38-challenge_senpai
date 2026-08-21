"""E89 rung 2: did claiming user-interactive QoS remove the efficiency-core state?

    usage: research/e89_rung2.py PREFIX [WARMUP [JSON_OUT]]

Reports the leg-level stuck rate per arm with a Wilson interval, absolute
candidate seconds per token restricted to clean legs, the price of the state
itself pooled over arms, and the drafting against non-drafting host phase
ratio that the plutarch reconciliation needs.
"""

import glob
import itertools
import json
import math
import os
import re
import statistics
import sys

HOST = ["d_pre_us", "d_flush_us", "d_head1_us", "d_submit1_us",
        "d_chain_us", "readout_us", "commit_us", "upkeep_us"]


def fnum(line, key):
    m = re.search(r"\b" + key + r"=(-?\d+)", line)
    return int(m.group(1)) if m else None


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - hw), min(1.0, c + hw)


def load(prefix, warmup):
    legs = {}
    for d in sorted(glob.glob(f"research/out/{prefix}-*")):
        tag = os.path.basename(d)
        rows = []
        for line in open(d + "/trace.txt"):
            if not line.startswith("mtp-trace: round="):
                continue
            r = fnum(line, "round")
            if r is None or r <= warmup:
                continue
            rows.append({
                "round": r,
                "host": sum(fnum(line, k) or 0 for k in HOST),
                "depth": fnum(line, "d"),
                "core": fnum(line, "e89_core_a"),
                "round_us": fnum(line, "round_us"),
            })
        if not rows:
            continue
        metrics = json.load(open(d + "/score.json"))["metrics"]
        legs[tag] = {
            "arm": tag[len(prefix) + 1:].rsplit("-", 1)[0],
            "rows": rows,
            "spt": metrics["mtp_seconds_per_token"],
            "matched": metrics.get("all_tokens_matched"),
            "draftlen": metrics.get("effective_mean_draft_len"),
        }
    return legs


def permutation(legs, key):
    """Exact one-sided permutation p for 'fix lowers key' over ctl and fix."""
    obs = [(L["arm"], L[key]) for L in legs.values() if L["arm"] in ("ctl", "fix")]
    values = [v for _, v in obs]
    nfix = sum(1 for a, _ in obs if a == "fix")
    total = sum(values)
    n = len(values)
    observed = (sum(v for a, v in obs if a == "fix") / nfix
                - (total - sum(v for a, v in obs if a == "fix")) / (n - nfix))
    hits = draws = 0
    for idx in itertools.combinations(range(n), nfix):
        s = sum(values[i] for i in idx)
        d = s / nfix - (total - s) / (n - nfix)
        draws += 1
        if d <= observed + 1e-12:
            hits += 1
    return {"statistic": observed, "relabelings": draws, "p_one_sided": hits / draws}


def main():
    prefix = sys.argv[1]
    warmup = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    out_json = sys.argv[3] if len(sys.argv) > 3 else None
    doc = {"prefix": prefix, "warmup_rounds": warmup, "decode_tokens": 512,
           "harness": "local", "sandbox": "off",
           "cool_gate_passed_real_gate": False,
           "gate_qualified_for_timing": False,
           "official_or_ranked_score": False}
    legs = load(prefix, warmup)
    allhost = sorted(r["host"] for L in legs.values() for r in L["rows"])
    cut = 2.0 * statistics.median(allhost[: len(allhost) // 2])
    for L in legs.values():
        L["flags"] = [r["host"] > cut for r in L["rows"]]
        L["prev"] = sum(L["flags"]) / len(L["flags"])

    print(f"slow cut {cut:.0f} us over {len(allhost)} post-warmup rounds, "
          f"{len(legs)} legs")
    print("harness=local, sandbox=off, ungated, "
          "cool_gate_passed_real_gate=false, gate_qualified_for_timing=false, "
          "official_or_ranked_score=false\n")

    doc["slow_cut_us"] = cut
    doc["post_warmup_rounds"] = len(allhost)
    doc["legs"] = len(legs)
    doc["per_leg_prevalence"] = {t: L["prev"] for t, L in sorted(legs.items())}

    arms = sorted({L["arm"] for L in legs.values()})
    doc["stuck_rate"] = {}
    print("LEG-LEVEL STUCK RATE, a leg is stuck when its post-warmup "
          "prevalence exceeds 0.5")
    for arm in arms:
        g = [L for L in legs.values() if L["arm"] == arm]
        k = sum(1 for L in g if L["prev"] > 0.5)
        lo, hi = wilson(k, len(g))
        doc["stuck_rate"][arm] = {
            "stuck": k, "legs": len(g), "rate": k / len(g),
            "wilson95_lo": lo, "wilson95_hi": hi,
            "mean_prevalence": statistics.mean(L["prev"] for L in g)}
        print(f"  {arm:<5} {k}/{len(g)}  rate {k / len(g):.3f}  "
              f"Wilson95 [{lo:.3f}, {hi:.3f}]  "
              f"mean prevalence {statistics.mean(L['prev'] for L in g):.3f}")

    print("\nEXACT PERMUTATION TEST, one sided, hypothesis 'fix lowers the value'")
    doc["permutation"] = {}
    for key, label in (("prev", "prevalence"), ("spt", "mtp_seconds_per_token")):
        r = permutation(legs, key)
        doc["permutation"][label] = r
        print(f"  {label:<24} fix minus ctl {r['statistic']:+.6f}  "
              f"p={r['p_one_sided']:.3f} over {r['relabelings']} relabelings")

    print("\nABSOLUTE s/token, CLEAN LEGS ONLY, prevalence below 0.10")
    means = {}
    for arm in arms:
        g = [L["spt"] for L in legs.values()
             if L["arm"] == arm and L["prev"] < 0.10]
        if not g:
            print(f"  {arm:<5} no clean leg")
            continue
        means[arm] = statistics.mean(g)
        sd = statistics.stdev(g) if len(g) > 1 else 0.0
        doc.setdefault("clean_leg_spt", {})[arm] = {
            "n": len(g), "mean": means[arm], "sd": sd}
        print(f"  {arm:<5} n={len(g)}  mean {means[arm]:.7f}  sd {sd:.7f}")
    if "ctl" in means and "fix" in means:
        d = means["fix"] - means["ctl"]
        doc["clean_leg_fix_minus_ctl"] = d
        doc["clean_leg_fix_minus_ctl_pct"] = 100 * d / means["ctl"]
        print(f"  fix minus ctl {d:+.7f} ({100 * d / means['ctl']:+.3f} %)")

    print("\nPRICE OF THE STATE ITSELF, leg level, arms pooled")
    obs = [L for L in legs.values() if L["arm"] != "bg"]
    clean = [L["spt"] for L in obs if L["prev"] < 0.10]
    stuck = [L["spt"] for L in obs if L["prev"] > 0.90]
    price = {"clean_n": len(clean), "clean_mean": statistics.mean(clean)}
    print(f"  clean n={len(clean)} mean {statistics.mean(clean):.7f}")
    if stuck:
        price["stuck_n"] = len(stuck)
        price["stuck_mean"] = statistics.mean(stuck)
        price["stuck_penalty_pct"] = 100 * (
            statistics.mean(stuck) - statistics.mean(clean)) / statistics.mean(clean)
        print(f"  stuck n={len(stuck)} mean {statistics.mean(stuck):.7f}")
        print(f"  penalty {price['stuck_penalty_pct']:+.3f} %")
    bg = [L["spt"] for L in legs.values() if L["arm"] == "bg"]
    if bg:
        price["forced_background_mean"] = statistics.mean(bg)
        price["forced_background_penalty_pct"] = 100 * (
            statistics.mean(bg) - statistics.mean(clean)) / statistics.mean(clean)
        print(f"  forced background n={len(bg)} mean {statistics.mean(bg):.7f} "
              f"penalty {price['forced_background_penalty_pct']:+.3f} %")
    doc["price_of_state"] = price

    print("\nCORRECTNESS")
    doc["correctness"] = {
        "all_tokens_matched_legs": sum(1 for L in legs.values() if L["matched"]),
        "total_legs": len(legs),
        "distinct_effective_mean_draft_len":
            len({L["draftlen"] for L in legs.values()})}
    print(f"  all_tokens_matched true on {sum(1 for L in legs.values() if L['matched'])}"
          f" of {len(legs)} legs; "
          f"{len({L['draftlen'] for L in legs.values()})} distinct draft lengths")

    print("\nTASK B, host phase sum by draft depth")
    hist = {}
    for L in legs.values():
        for r in L["rows"]:
            hist[r["depth"]] = hist.get(r["depth"], 0) + 1
    doc["depth_histogram"] = {str(k): v for k, v in hist.items()}
    doc["depth_split"] = {}
    print("  depth histogram:", dict(sorted(hist.items(), key=lambda kv: -kv[1])))
    for sel, name in ((lambda d: d == 0, "non-drafting depth 0"),
                      (lambda d: d and d > 0, "drafting depth above 0")):
        fast, slow = [], []
        for L in legs.values():
            if L["arm"] == "bg":
                continue
            for r, flag in zip(L["rows"], L["flags"]):
                if r["depth"] is None or not sel(r["depth"]):
                    continue
                (slow if flag else fast).append(r["host"])
        entry = {"fast_n": len(fast), "slow_n": len(slow)}
        if fast and slow:
            entry["fast_median_us"] = statistics.median(fast)
            entry["slow_median_us"] = statistics.median(slow)
            entry["ratio"] = statistics.median(slow) / statistics.median(fast)
            print(f"  {name:<22} fast n={len(fast)} med {statistics.median(fast):.0f} us"
                  f"   slow n={len(slow)} med {statistics.median(slow):.0f} us"
                  f"   ratio {entry['ratio']:.2f}")
        else:
            entry["insufficient"] = True
            print(f"  {name:<22} fast n={len(fast)} slow n={len(slow)}, insufficient")
        doc["depth_split"][name] = entry

    if out_json:
        json.dump(doc, open(out_json, "w"), indent=2, sort_keys=True)
        print(f"\nwrote {out_json}")


main()
