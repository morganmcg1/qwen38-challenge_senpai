#!/usr/bin/env python3
"""Did de7981ae's +73.6% plutarch fix buy any score at all?

Counterfactual: replace their plutarch raw_p with OUR (absorbing-state broken)
value and recompute the median-of-two-central-order-statistics score.
"""
import json

NAMES = {"919318e1": "beagle", "192fb621": "botany", "4b9e88cd": "drama",
         "a2ea8b60": "essays", "00142a44": "medicine", "c1ec5866": "plutarch",
         "ea82dcb5": "republic", "3b10cb4d": "travel"}

rows = json.load(open("/tmp/rows_live.json"))


def pp(r):
    return {NAMES[p["prompt_sha256"][:8]]: p for p in r["officialMetrics"]["per_prompt"]}


def med(v):
    v = sorted(v)
    return (v[3] + v[4]) / 2.0


a = [r for r in rows if r["id"].startswith("de7981ae")][0]
o = [r for r in rows if r["id"].startswith("ca9251b8")][0]
qa, qo = pp(a), pp(o)
raws = {n: qa[n]["raw_ratio_of_means"] for n in NAMES.values()}

print("de7981ae reported score        ", repr(a["officialScore"]))
print("identity from their 8 raw_p    ", repr(med(list(raws.values()))))

cf = dict(raws)
cf["plutarch"] = qo["plutarch"]["raw_ratio_of_means"]
print()
print("COUNTERFACTUAL: their plutarch reverted to our broken (absorbing) value")
print("  their plutarch raw_p         ", round(raws["plutarch"], 6))
print("  our   plutarch raw_p         ", round(cf["plutarch"], 6))
print("  plutarch improvement         %+.2f%%" % (100 * (raws["plutarch"] / cf["plutarch"] - 1)))
print("  score WITH their fix         ", repr(med(list(raws.values()))))
print("  score WITHOUT their fix      ", repr(med(list(cf.values()))))
print("  score value of the fix       %+.6f%%"
      % (100 * (med(list(raws.values())) / med(list(cf.values())) - 1)))

print()
print("their raw_p sorted (order statistic -> value):")
for i, (n, v) in enumerate(sorted(raws.items(), key=lambda t: t[1]), 1):
    mark = "  <-- CENTRAL PAIR (weight 0.5)" if i in (4, 5) else ""
    print(f"  {i}. {n:<9} {v:.6f}{mark}")

print()
print("ours sorted:")
ro = {n: qo[n]["raw_ratio_of_means"] for n in NAMES.values()}
for i, (n, v) in enumerate(sorted(ro.items(), key=lambda t: t[1]), 1):
    mark = "  <-- CENTRAL PAIR (weight 0.5)" if i in (4, 5) else ""
    print(f"  {i}. {n:<9} {v:.6f}{mark}")
