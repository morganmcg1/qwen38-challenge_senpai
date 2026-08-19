#!/usr/bin/env python3
"""The valid ranked test of E34's direction, restricted to the declared head.

E34 wants sdpaWidthWallDepthCap 5 -> 4, which pins capped rounds at M=5 (one
weight pass) instead of M=6 (two), at the cost of ~0.49 fewer accepted tokens per
round on beagle (n 4.533 -> ~4.04, -10.3%).

Every cap=4 row on the board ran a different head artifact, so those scores
cannot test it. But the declared-head population (559b24eb, n=94) contains rows
whose acceptance differs from the 8/8 default fingerprint. Split them by
DIRECTION: did anyone on the right head already run SHALLOWER on beagle, and what
did they score?
"""
import json
import statistics as st

NAMES = {"919318e1": "beagle", "192fb621": "botany", "4b9e88cd": "drama",
         "a2ea8b60": "essays", "00142a44": "medicine", "c1ec5866": "plutarch",
         "ea82dcb5": "republic", "3b10cb4d": "travel"}
ORDER = ["plutarch", "drama", "travel", "beagle", "medicine", "essays", "republic", "botany"]
DECLARED = "559b24ebca354018e4402fdb1f5af1afe5a0721bd2ebf04133500d846f7d5f71"

rows = json.load(open("/tmp/rows_live.json"))
scored = [r for r in rows if isinstance(r.get("officialScore"), (int, float))
          and r.get("officialMetrics", {}).get("prompt_count") == 8]


def pp(r):
    return {NAMES[p["prompt_sha256"][:8]]: p for p in r["officialMetrics"]["per_prompt"]}


def head_of(r):
    hs = {p.get("head_provenance_sha256") for p in r["officialMetrics"]["per_prompt"]}
    hs.discard(None)
    return next(iter(hs)) if len(hs) == 1 else None


ours = [r for r in rows if r["id"].startswith("ca9251b8")][0]
o = pp(ours)
DEF = {n: o[n]["effective_mean_draft_len"] for n in ORDER}
print("default fingerprint (ours == top-12):")
print("  " + "  ".join(f"{n}={DEF[n]:.4f}" for n in ORDER))

dh = [r for r in scored if head_of(r) == DECLARED]
print(f"\nrows on the declared head: {len(dh)}")

same, deeper, shallower, mixed = [], [], [], []
for r in dh:
    q = pp(r)
    d = {n: q[n]["effective_mean_draft_len"] - DEF[n] for n in ORDER}
    nz = [n for n in ORDER if abs(d[n]) > 1e-9]
    if not nz:
        same.append(r)
    elif all(d[n] > 0 for n in nz):
        deeper.append((r, d))
    elif all(d[n] < 0 for n in nz):
        shallower.append((r, d))
    else:
        mixed.append((r, d))

print(f"  identical acceptance (default schedule) : {len(same)}")
print(f"  strictly DEEPER on every changed prompt : {len(deeper)}")
print(f"  strictly SHALLOWER on every changed prompt: {len(shallower)}")
print(f"  mixed direction                          : {len(mixed)}")

if same:
    s = [r["officialScore"] for r in same]
    print(f"\ndefault-schedule cluster: n={len(s)} best={max(s):.6f} "
          f"median={st.median(s):.6f} min={min(s):.6f}")

print()
print("=" * 104)
print("NON-DEFAULT SCHEDULES ON THE DECLARED HEAD, by direction")
for label, group in (("DEEPER", deeper), ("SHALLOWER", shallower), ("MIXED", mixed)):
    if not group:
        print(f"\n  {label}: none")
        continue
    print(f"\n  {label}  (n={len(group)})")
    group.sort(key=lambda t: -t[0]["officialScore"])
    for r, d in group[:14]:
        q = pp(r)
        chg = ", ".join(f"{n} {DEF[n]:.3f}->{q[n]['effective_mean_draft_len']:.3f}"
                        for n in ORDER if abs(d[n]) > 1e-9)[:96]
        print(f"    {r['officialScore']:.6f} {r['status']:<11} {r['id'][:8]} "
              f"{r.get('solverUsername','?'):<16} {chg}")

print()
print("=" * 104)
print("BEAGLE-SPECIFIC: every declared-head row whose beagle n differs from 4.5327")
cand = []
for r in dh:
    q = pp(r)
    b = q["beagle"]["effective_mean_draft_len"]
    if abs(b - DEF["beagle"]) > 1e-9:
        cand.append((r, b, q["beagle"]["raw_ratio_of_means"],
                     q["beagle"]["mtp_seconds_per_token_mean"]))
print(f"  n = {len(cand)}")
print(f"  {'score':>11} {'status':<11} {'id':<10} {'beagle_n':>9} {'beagle_raw':>11} "
      f"{'beagle_s/tok':>13} {'vs ours raw':>12}")
ourraw = o["beagle"]["raw_ratio_of_means"]
cand.sort(key=lambda t: t[1])
for r, b, raw, spt in cand:
    print(f"  {r['officialScore']:>11.6f} {r['status']:<11} {r['id'][:8]:<10} "
          f"{b:>9.4f} {raw:>11.5f} {spt:>13.8f} {100*(raw/ourraw-1):>+11.3f}%")
print(f"  {'OURS':>11} {'':<11} {'ca9251b8':<10} {DEF['beagle']:>9.4f} "
      f"{ourraw:>11.5f} {o['beagle']['mtp_seconds_per_token_mean']:>13.8f} {0.0:>+11.3f}%")

print()
print("  ==> If any row ran beagle SHALLOWER on this head and its beagle raw_p rose,")
print("      E34's direction has ranked support. If shallower rows lost, it is dead.")
