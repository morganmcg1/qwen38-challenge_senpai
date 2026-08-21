"""E89 rung 2: did claiming user-interactive QoS remove the efficiency-core state?

    usage: research/e89_rung2.py PREFIX [WARMUP]

Reports the leg-level stuck rate per arm with a Wilson interval, absolute
candidate seconds per token restricted to clean legs, the price of the state
itself pooled over arms, and the drafting against non-drafting host phase
ratio that the plutarch reconciliation needs.
"""

import glob
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


def main():
    prefix = sys.argv[1]
    warmup = int(sys.argv[2]) if len(sys.argv) > 2 else 8
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

    arms = sorted({L["arm"] for L in legs.values()})
    print("LEG-LEVEL STUCK RATE, a leg is stuck when its post-warmup "
          "prevalence exceeds 0.5")
    for arm in arms:
        g = [L for L in legs.values() if L["arm"] == arm]
        k = sum(1 for L in g if L["prev"] > 0.5)
        lo, hi = wilson(k, len(g))
        print(f"  {arm:<5} {k}/{len(g)}  rate {k / len(g):.3f}  "
              f"Wilson95 [{lo:.3f}, {hi:.3f}]  "
              f"mean prevalence {statistics.mean(L['prev'] for L in g):.3f}")

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
        print(f"  {arm:<5} n={len(g)}  mean {means[arm]:.7f}  sd {sd:.7f}")
    if "ctl" in means and "fix" in means:
        d = means["fix"] - means["ctl"]
        print(f"  fix minus ctl {d:+.7f} ({100 * d / means['ctl']:+.3f} %)")

    print("\nPRICE OF THE STATE ITSELF, leg level, arms pooled")
    obs = [L for L in legs.values() if L["arm"] != "bg"]
    clean = [L["spt"] for L in obs if L["prev"] < 0.10]
    stuck = [L["spt"] for L in obs if L["prev"] > 0.90]
    print(f"  clean n={len(clean)} mean {statistics.mean(clean):.7f}")
    if stuck:
        print(f"  stuck n={len(stuck)} mean {statistics.mean(stuck):.7f}")
        print(f"  penalty {100 * (statistics.mean(stuck) - statistics.mean(clean)) / statistics.mean(clean):+.3f} %")
    bg = [L["spt"] for L in legs.values() if L["arm"] == "bg"]
    if bg:
        print(f"  forced background n={len(bg)} mean {statistics.mean(bg):.7f} "
              f"penalty {100 * (statistics.mean(bg) - statistics.mean(clean)) / statistics.mean(clean):+.3f} %")

    print("\nCORRECTNESS")
    print(f"  all_tokens_matched true on {sum(1 for L in legs.values() if L['matched'])}"
          f" of {len(legs)} legs; "
          f"{len({L['draftlen'] for L in legs.values()})} distinct draft lengths")

    print("\nTASK B, host phase sum by draft depth")
    hist = {}
    for L in legs.values():
        for r in L["rows"]:
            hist[r["depth"]] = hist.get(r["depth"], 0) + 1
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
        if fast and slow:
            print(f"  {name:<22} fast n={len(fast)} med {statistics.median(fast):.0f} us"
                  f"   slow n={len(slow)} med {statistics.median(slow):.0f} us"
                  f"   ratio {statistics.median(slow) / statistics.median(fast):.2f}")
        else:
            print(f"  {name:<22} fast n={len(fast)} slow n={len(slow)}, insufficient")


main()
