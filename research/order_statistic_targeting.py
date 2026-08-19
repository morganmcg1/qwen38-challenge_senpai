#!/usr/bin/env python3
"""Which prompts actually set the published median, and where do we lose to the frontier?

The published score is median(raw_1..raw_8) = mean of the 4th and 5th order
statistics after sorting.  A gain on a prompt that never occupies rank 4 or 5 is
worth exactly zero.  This instrument identifies, per submission, which prompts
occupy the central pair, and then decomposes our deficit to the promoted frontier
prompt by prompt.

Read-only.  Input: research/e53-board-facts.json (edward's E53 board pull).
harness=ranked throughout: every number here is an official per-prompt raw_ratio.
"""

import json
import os
import statistics
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
FACTS = os.path.join(HERE, "e53-board-facts.json")

FRONTIER = "59b321ee"
OURS = {
    "ca9251b8": ("2b0c36a0", 3.23250848263467),
    "2c766441": ("e277c57b", 3.07213258255320),
    "9197ed62": ("dbf91c6c", 3.06938159465413),
    "4437d061": ("95f8311d", 2.86126590369985),
}


def load():
    with open(FACTS) as f:
        raw = json.load(f)
    by_sub = defaultdict(dict)
    meta = {}
    for prompt, rows in raw["telemetry"].items():
        for r in rows:
            sub = str(r["submission"])[:8]
            if r.get("raw_ratio") is None:
                continue
            by_sub[sub][prompt] = float(r["raw_ratio"])
            meta.setdefault(sub, {
                "solver": r.get("solver"),
                "score": r.get("score"),
                "commit": str(r.get("commit") or "")[:8],
                "created": r.get("created"),
            })
    return by_sub, meta


def central_pair(prompt_to_rr):
    """Return (sorted list of (prompt, rr), the two prompts at ranks 4 and 5)."""
    ordered = sorted(prompt_to_rr.items(), key=lambda kv: kv[1])
    if len(ordered) != 8:
        return ordered, None
    return ordered, (ordered[3][0], ordered[4][0])


def main():
    by_sub, meta = load()
    full = {s: p for s, p in by_sub.items() if len(p) == 8}
    print(f"submissions with all 8 prompts: {len(full)} of {len(by_sub)}")

    prompts = sorted({p for v in full.values() for p in v})
    print(f"prompts: {prompts}\n")

    # --- 1. Which prompts ever occupy the central pair, across the whole board?
    pair_count = Counter()
    member_count = Counter()
    for s, p in full.items():
        _, pair = central_pair(p)
        if pair:
            pair_count[tuple(sorted(pair))] += 1
            for x in pair:
                member_count[x] += 1
    n = sum(pair_count.values())
    print("=== central-pair membership across the board (rank 4 or 5 of 8) ===")
    print(f"{'prompt':<14}{'times in pair':>14}{'share of subs':>16}")
    for p in prompts:
        c = member_count.get(p, 0)
        print(f"{p:<14}{c:>14}{c / n * 100:>15.1f}%")
    print("\n=== most common exact central pairs ===")
    for pair, c in pair_count.most_common(6):
        print(f"  {pair[0]:<12} + {pair[1]:<12}  {c:>4}  ({c / n * 100:.1f}%)")

    # --- 2. Per-prompt ceiling and floor: is the pool bimodal?
    print("\n=== per-prompt distribution over the board (raw_ratio) ===")
    print(f"{'prompt':<14}{'best':>9}{'p90':>9}{'median':>9}{'worst':>9}")
    for p in prompts:
        vals = sorted(v[p] for v in full.values())
        k = int(0.9 * (len(vals) - 1))
        print(f"{p:<14}{vals[-1]:>9.4f}{vals[k]:>9.4f}"
              f"{statistics.median(vals):>9.4f}{vals[0]:>9.4f}")

    # --- 3. Frontier vs our best: prompt-by-prompt deficit decomposition
    print("\n=== frontier vs our best submission, prompt by prompt ===")
    fr = full.get(FRONTIER)
    best_ours = max(
        (s for s in OURS if s in full),
        key=lambda s: OURS[s][1], default=None)
    if fr is None or best_ours is None:
        print("  missing frontier or our rows; cannot decompose")
        return
    ours = full[best_ours]
    fr_ord, fr_pair = central_pair(fr)
    our_ord, our_pair = central_pair(ours)
    print(f"  frontier  {FRONTIER} central pair: {fr_pair}")
    print(f"  ours      {best_ours} central pair: {our_pair}\n")
    print(f"{'prompt':<14}{'ours':>9}{'frontier':>10}{'delta':>9}"
          f"{'delta %':>9}  {'our rank':>8}{'fr rank':>8}  in-pair")
    for p in prompts:
        o, f_ = ours[p], fr[p]
        orank = [q for q, _ in our_ord].index(p) + 1
        frank = [q for q, _ in fr_ord].index(p) + 1
        tag = []
        if our_pair and p in our_pair:
            tag.append("OURS")
        if fr_pair and p in fr_pair:
            tag.append("FRONTIER")
        print(f"{p:<14}{o:>9.4f}{f_:>10.4f}{o - f_:>9.4f}"
              f"{(o - f_) / f_ * 100:>8.2f}%{orank:>9}{frank:>8}  "
              f"{','.join(tag)}")

    om = statistics.median(ours.values())
    fm = statistics.median(fr.values())
    print(f"\n  our median      {om:.14f}")
    print(f"  frontier median {fm:.14f}")
    print(f"  deficit         {fm - om:.14f} = {(fm - om) / om * 100:.4f}%")

    # --- 4. Counterfactual: what does a uniform x% candidate-leg gain buy us?
    #        A uniform multiplicative gain moves every raw_p by the same factor,
    #        so it moves the median by exactly that factor.  A TARGETED gain on
    #        only the central pair moves the median by half as much per prompt
    #        but costs work on 2 prompts instead of 8.
    print("\n=== what would close the deficit ===")
    need = (fm - om) / om * 100
    print(f"  uniform candidate-leg speedup needed on all 8 prompts: {need:.4f}%")
    # targeted: raise only the two central prompts until the median matches.
    ov = sorted(ours.values())
    target = fm
    # median = (ov[3]+ov[4])/2 ; raising both by factor g and re-sorting
    g_lo, g_hi = 1.0, 2.0
    for _ in range(200):
        g = (g_lo + g_hi) / 2
        trial = sorted(ov[:3] + [ov[3] * g, ov[4] * g] + ov[5:])
        m = (trial[3] + trial[4]) / 2
        if m < target:
            g_lo = g
        else:
            g_hi = g
    print(f"  speedup needed on ONLY the 2 central prompts:              "
          f"{(g_lo - 1) * 100:.4f}%")
    print("  (a targeted gain saturates once the raised prompt overtakes rank 6:")
    print(f"   our rank-6 raw_ratio is {ov[5]:.4f} vs rank-5 {ov[4]:.4f}, "
          f"headroom {(ov[5] / ov[4] - 1) * 100:.2f}% before the pair changes)")

    # --- 5. Does the frontier win by being uniformly good, or by a tail?
    print("\n=== how the frontier's own profile is shaped ===")
    fr_ranks = {}
    for p in prompts:
        vals = sorted((v[p] for v in full.values()), reverse=True)
        fr_ranks[p] = vals.index(fr[p]) + 1
    for p in prompts:
        print(f"  {p:<14} frontier board rank {fr_ranks[p]:>4} of {len(full)}")

    # --- 6. What makes each prompt mechanically different?  mean_draft_len is
    #        the schedule's realised depth, so it reads the acceptance regime.
    print("\n=== mechanical signature of each prompt (over the whole board) ===")
    with open(FACTS) as f:
        raw = json.load(f)
    print(f"{'prompt':<14}{'mdl med':>9}{'mdl best10':>11}"
          f"{'nondraft%':>11}{'mtp_spt med':>13}{'rr/mdl':>9}")
    for p in prompts:
        rows = raw["telemetry"][p]
        mdl = [r["mean_draft_len"] for r in rows
               if r.get("mean_draft_len") is not None]
        top = sorted(rows, key=lambda r: r["raw_ratio"] or 0, reverse=True)[:10]
        mdl_top = [r["mean_draft_len"] for r in top
                   if r.get("mean_draft_len") is not None]
        nd = [r for r in rows if (r.get("non_drafting_rounds") or 0) > 0]
        spt = [r["mtp_spt"] for r in rows if r.get("mtp_spt") is not None]
        med_mdl = statistics.median(mdl)
        print(f"{p:<14}{med_mdl:>9.3f}"
              f"{statistics.median(mdl_top):>11.3f}"
              f"{len(nd) / len(rows) * 100:>10.1f}%"
              f"{statistics.median(spt):>13.6f}"
              f"{statistics.median(v[p] for v in full.values()) / med_mdl:>9.3f}")

    # --- 7. Frontier vs us on the pivotal prompt: is it depth or speed?
    print("\n=== frontier vs us, mechanical, on the two central prompts ===")
    print(f"{'prompt':<10}{'who':<12}{'mean_draft_len':>15}"
          f"{'mtp_spt':>12}{'serial_spt':>12}{'raw_ratio':>11}")
    for p in ("beagle", "medicine"):
        for label, sub in (("frontier", FRONTIER), ("ours", best_ours)):
            r = next(x for x in raw["telemetry"][p]
                     if str(x["submission"]).startswith(sub))
            print(f"{p:<10}{label:<12}{r['mean_draft_len']:>15.4f}"
                  f"{r['mtp_spt']:>12.6f}{r['serial_spt']:>12.6f}"
                  f"{r['raw_ratio']:>11.4f}")

    # --- 8. Across the board, does beagle reward depth or reward speed?
    print("\n=== beagle: what separates the top decile from the rest ===")
    rows = [r for r in raw["telemetry"]["beagle"]
            if r.get("raw_ratio") and r.get("mean_draft_len")]
    rows.sort(key=lambda r: r["raw_ratio"], reverse=True)
    k = max(1, len(rows) // 10)
    for label, grp in (("top decile", rows[:k]), ("bottom decile", rows[-k:]),
                       ("middle", rows[k:-k])):
        print(f"  {label:<15} n={len(grp):>4}"
              f"  mean_draft_len {statistics.median(r['mean_draft_len'] for r in grp):>6.3f}"
              f"  mtp_spt {statistics.median(r['mtp_spt'] for r in grp):>9.6f}"
              f"  raw {statistics.median(r['raw_ratio'] for r in grp):>6.3f}")


if __name__ == "__main__":
    main()
