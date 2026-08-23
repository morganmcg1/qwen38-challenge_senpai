"""F258  price 0cf1637e against the crown on the fair (de-lucked) median."""
import json, statistics as st

BOARD = "/tmp/yukon-board/full.json"
rows = json.load(open(BOARD))["submissions"]
by = {}
for r in rows:
    pp = (r.get("officialMetrics") or {}).get("per_prompt")
    if not pp:
        continue
    by[r["id"][:8]] = r

NAME = {"919318e1": "beagle", "192fb621": "botany", "4b9e88cd": "drama",
        "a2ea8b60": "essays", "00142a44": "medicine", "c1ec5866": "plutarch",
        "ea82dcb5": "republic", "3b10cb4d": "travel"}


def vec(idx):
    d = {}
    for e in by[idx]["officialMetrics"]["per_prompt"]:
        d[NAME[e["prompt_sha256"][:8]]] = e
    return d


def med(raws):
    s = sorted(raws.values())
    return (s[3] + s[4]) / 2.0, s


# population serial medians, per prompt, over every scored row
pool = {n: [] for n in NAME.values()}
for r in rows:
    pp = (r.get("officialMetrics") or {}).get("per_prompt")
    if not pp:
        continue
    for e in pp:
        pool[NAME[e["prompt_sha256"][:8]]].append(e["serial_seconds_per_token_mean"])
SMED = {n: st.median(v) for n, v in pool.items()}
N = len(pool["essays"])

OURS, CROWN = "0cf1637e", "684821ed"
o, c = vec(OURS), vec(CROWN)

print("=" * 92)
print("F258   0cf1637e   published 3.68278758   vs the crown 684821ed 3.71959723")
print("=" * 92)
print(f"serial population medians drawn from {N} scored rows\n")
print("prompt     ours serial   %vs med   crown serial  %vs med    cand ours vs crown")
tot = 0.0
W = {"beagle": .5000, "essays": .4474, "republic": .0329, "medicine": .0197,
     "botany": 0.0, "drama": 0.0, "travel": 0.0, "plutarch": 0.0}
wsum = 0.0
for n in ["beagle", "essays", "medicine", "republic", "botany", "drama", "travel", "plutarch"]:
    so, sc = o[n]["serial_seconds_per_token_mean"], c[n]["serial_seconds_per_token_mean"]
    co, cc = o[n]["mtp_seconds_per_token_mean"], c[n]["mtp_seconds_per_token_mean"]
    dc = (co / cc - 1) * 100
    tot += dc
    wsum += W[n] * dc
    print(f"{n:10s} {so:.9f} {(so/SMED[n]-1)*100:+8.4f}  {sc:.9f} {(sc/SMED[n]-1)*100:+8.4f}    {dc:+8.4f} %")
print(f"\n  candidate-leg gap to the crown: mean8 {tot/8:+.4f} %   Rule148-weighted {wsum:+.4f} %")
print("  (was  mean8 +0.8221 %  Rule148 +0.9511 %  at 572b2cc4)")

# fair medians
for tag, v in (("0cf1637e OURS", o), ("684821ed CROWN", c)):
    asp = {n: v[n]["serial_seconds_per_token_mean"] / v[n]["mtp_seconds_per_token_mean"] for n in NAME.values()}
    fair = {n: SMED[n] / v[n]["mtp_seconds_per_token_mean"] for n in NAME.values()}
    ma, sa = med(asp)
    mf, sf = med(fair)
    up_a = [k for k in asp if abs(asp[k] - sa[4]) < 1e-12][0]
    up_f = [k for k in fair if abs(fair[k] - sf[4]) < 1e-12][0]
    print(f"\n{tag}")
    print(f"   as published        {ma:.8f}   upper slot = {up_a}")
    print(f"   fair serial vector  {mf:.8f}   upper slot = {up_f}   ({(mf/ma-1)*100:+.4f} %)")

# head to head on fair medians
of = {n: SMED[n] / o[n]["mtp_seconds_per_token_mean"] for n in NAME.values()}
cf = {n: SMED[n] / c[n]["mtp_seconds_per_token_mean"] for n in NAME.values()}
mo, _ = med(of)
mc, _ = med(cf)
print("\n" + "=" * 92)
print("HEAD TO HEAD ON THE FAIR MEDIAN  (both trees given the population serial vector)")
print("=" * 92)
print(f"   crown  684821ed  {mc:.8f}")
print(f"   ours   0cf1637e  {mo:.8f}     gap {(mc/mo-1)*100:+.4f} %")
print(f"\n   published gap                        {(3.71959723/3.68278758-1)*100:+.4f} %")
print(f"   of which pure serial lottery         {(3.71959723/3.68278758-1)*100 - (mc/mo-1)*100:+.4f} pp")

# what a uniform candidate speedup buys us
print("\n" + "=" * 92)
print("REQUIRED UNIFORM CANDIDATE-LEG SPEEDUP, from 0cf1637e")
print("=" * 92)
for target, lab in ((mc, "the crown's FAIR median"), (3.71959723, "the bar AS PUBLISHED")):
    lo, hi = 0.0, 0.10
    for _ in range(80):
        mid = (lo + hi) / 2
        test = {n: o[n]["serial_seconds_per_token_mean"] / (o[n]["mtp_seconds_per_token_mean"] * (1 - mid))
                for n in NAME.values()}
        m, _ = med(test)
        if m < target:
            lo = mid
        else:
            hi = mid
    # fair-serial variant
    lo2, hi2 = 0.0, 0.10
    for _ in range(80):
        mid = (lo2 + hi2) / 2
        test = {n: SMED[n] / (o[n]["mtp_seconds_per_token_mean"] * (1 - mid)) for n in NAME.values()}
        m, _ = med(test)
        if m < target:
            lo2 = mid
        else:
            hi2 = mid
    print(f"   to reach {lab:26s}  own serial draw {lo*100:+.4f} %   fair serial {lo2*100:+.4f} %")

# essays counterfactual: what if our essays serial had drawn like the bar's
alt = {n: o[n]["serial_seconds_per_token_mean"] / o[n]["mtp_seconds_per_token_mean"] for n in NAME.values()}
alt["essays"] = c["essays"]["serial_seconds_per_token_mean"] / o["essays"]["mtp_seconds_per_token_mean"]
ma2, sa2 = med(alt)
print("\n" + "=" * 92)
print("COUNTERFACTUAL  our tree, with the BAR'S essays serial leg")
print("=" * 92)
print(f"   0cf1637e would have published  {ma2:.8f}   ({(ma2/3.68278758-1)*100:+.4f} %)")
print(f"   vs the bar 3.71959723           {(ma2/3.71959723-1)*100:+.4f} %")
