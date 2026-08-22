#!/usr/bin/env python3
"""How reproducible is the crown?  Find every row whose note declares itself a
resample / restoration of bc070b7 (or of the fac135f frontier) and read its score."""
import json, re, sys

d = json.load(open("/tmp/yukon-board/full.json"))
rows = d["submissions"]

# the F76 mode index
W = {"plut": -0.3852, "dram": +0.0215, "trav": +0.4945, "beag": +0.2068,
     "medi": -0.1480, "repu": -0.0917, "essa": -0.0041, "bota": -0.0939}
ORDER = ["plut", "dram", "trav", "beag", "medi", "repu", "essa", "bota"]

# learn the prompt_sha256 -> slot map from the crown row, using the known
# ordering of median seconds/token (plutarch slowest ... botany fastest is NOT
# the order; use the campaign's fixed per-prompt medians instead).
MED = {"plut": 0.030310, "dram": 0.020127, "trav": 0.017810, "beag": 0.012375,
       "medi": 0.011564, "repu": 0.011324, "essa": 0.011339, "bota": 0.011301}


def slots(row):
    pp = row.get("officialMetrics", {}).get("per_prompt")
    if not pp or len(pp) != 8:
        return None
    out = {}
    for e in pp:
        t = e.get("mtp_seconds_per_token_mean")
        if t is None:
            return None
        best = min(MED, key=lambda k: abs(__import__("math").log(t / MED[k])))
        out[best] = t
    return out if len(out) == 8 else None


def index(row):
    s = slots(row)
    if not s:
        return None
    import math
    return sum(W[k] * 100.0 * math.log(s[k]) for k in ORDER)


pat = re.compile(r"bc070b7|fac135f", re.I)
hits = []
for r in rows:
    note = (r.get("note") or "")
    sc = r.get("officialScore")
    if sc is None:
        continue
    if pat.search(note):
        hits.append(r)

print(f"{len(hits)} scored rows whose note names bc070b7 or fac135f\n")
print(f"{'id':<10} {'score':>12} {'idx':>9} {'status':<10} {'when':<20} note")
for r in sorted(hits, key=lambda x: -x["officialScore"])[:24]:
    i = index(r)
    print(f"{r['id'][:8]:<10} {r['officialScore']:>12.8f} "
          f"{('%9.4f' % i) if i is not None else '        -'} "
          f"{str(r.get('status'))[:10]:<10} {str(r.get('createdAt'))[:19]:<20} "
          f"{(r.get('note') or '')[:70].splitlines()[-1] if r.get('note') else ''}")

sc = sorted(r["officialScore"] for r in hits)
if sc:
    import statistics
    print(f"\nn={len(sc)}  min {sc[0]:.6f}  median {statistics.median(sc):.6f}  max {sc[-1]:.6f}")
    print(f"crown published 3.35922017  is the {sum(1 for x in sc if x < 3.35922017)}/{len(sc)} order statistic")
