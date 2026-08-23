#!/usr/bin/env python3
"""F257: the exact ranked value of a candidate seed-prefill saving.

Finding 227's value model was raw' = (S - dP)/(C - dP), which subtracts the
saving from the SERIAL leg as well.  That is the LOCAL harness, where both legs
run the candidate binary.  In the RANKED harness the serial leg runs the pinned
baseline build, so a candidate prefill saving changes the DENOMINATOR ONLY.
Re-derive it from real receipts and compare.
"""
import json, statistics as st

BOARD = "/tmp/yukon-board/full.json"
PFX = {"919318e1": "beagle", "192fb621": "botany", "4b9e88cd": "drama",
       "a2ea8b60": "essays", "00142a44": "medicine", "c1ec5866": "plutarch",
       "ea82dcb5": "republic", "3b10cb4d": "travel"}
CAND = "mtp_seconds_per_token_mean"
SER = "serial_seconds_per_token_mean"
PRE = "prefill_seconds_per_token"
ORDER = ["beagle", "essays", "medicine", "republic", "botany", "drama", "travel", "plutarch"]

with open(BOARD) as f:
    doc = json.load(f)
rows = doc["submissions"] if isinstance(doc, dict) else doc
B = {}
for r in rows:
    om = r.get("officialMetrics") or {}
    if not om.get("per_prompt"):
        continue
    d = {PFX[e["prompt_sha256"][:8]]: e for e in om["per_prompt"]
         if e["prompt_sha256"][:8] in PFX}
    if len(d) == 8 and r.get("officialScore"):
        B[r["id"][:8]] = (d, r["officialScore"], r.get("solverUsername"))

bar, barscore, _ = B["684821ed"]

print("=" * 92)
print("F257  the candidate seed prefill is a bigger share of exactly the prompts")
print("      that decide the median")
print("=" * 92)
print(f"{'prompt':9} {'prefill spt':>13} {'cand spt':>12} {'prefill share':>14}"
      f" {'Rule148 wt':>11}")
share = {}
W148 = {"beagle": 0.5000, "essays": 0.4474, "republic": 0.0329, "medicine": 0.0197}
for p in ORDER:
    s = bar[p][PRE] / bar[p][CAND]
    share[p] = s
    print(f"{p:9} {bar[p][PRE]:13.9f} {bar[p][CAND]:12.9f} {100*s:13.4f} %"
          f" {W148.get(p, 0.0):11.4f}")
wshare = sum(W148[p] * share[p] for p in W148) / sum(W148.values())
print(f"\n  Rule 148 weighted prefill share of the candidate leg: {100*wshare:.4f} %")
print(f"  unweighted eight-prompt mean:                         "
      f"{100*st.mean([share[p] for p in ORDER]):.4f} %")


def median_after(f):
    """published median after cutting the CANDIDATE prefill by fraction f"""
    raw = {}
    for p in ORDER:
        c = bar[p][CAND] - f * bar[p][PRE]
        raw[p] = bar[p][SER] / c
    v = sorted(raw.values())
    up = min(("essays", "republic", "medicine", "botany"), key=lambda q: raw[q])
    return 0.5 * (v[3] + v[4]), up


def median_after_bothlegs(f):
    """F227's model: subtract from both legs"""
    raw = {}
    for p in ORDER:
        d = f * bar[p][PRE]
        raw[p] = (bar[p][SER] - d) / (bar[p][CAND] - d)
    v = sorted(raw.values())
    return 0.5 * (v[3] + v[4])


print()
print("=" * 92)
print("RANKED value of a candidate-only prefill saving  (denominator only)")
print("=" * 92)
print(f"  {'prefill cut':>12} {'median':>12} {'% of median':>13} {'F227 both-legs':>16}"
      f"  {'ratio':>7}  upper slot")
for f in (0.02, 0.05, 0.0497, 0.10, 0.15, 0.20, 0.30, 0.50, 1.00):
    m, up = median_after(f)
    pct = 100 * (m / barscore - 1)
    mb = median_after_bothlegs(f)
    pb = 100 * (mb / barscore - 1)
    ratio = pct / pb if pb else float("nan")
    print(f"  {100*f:11.2f}% {m:12.8f} {pct:12.4f}% {pb:15.4f}%  {ratio:7.2f}x  {up}")

print()
print("  the two proven ranked prefill wins on the board:")
for rid in ("5cdc9c17", "43925f29", "a9dd132a"):
    if rid in B:
        d, sc, who = B[rid]
        pp = 100 * (st.mean([d[p][PRE] for p in ORDER])
                    / st.mean([bar[p][PRE] for p in ORDER]) - 1)
        print(f"    {rid} {who:>14s}  prefill {pp:+.4f} % vs the bar   published {sc:.8f}")

print()
print("=" * 92)
print("WHAT THE CAMPAIGN NEEDS  (Finding 254)")
print("=" * 92)
print("  +0.9475 % to the crown's fair median   -> prefill cut needed: ", end="")
lo, hi = 0.0, 1.0
for _ in range(100):
    mid = 0.5 * (lo + hi)
    if 100 * (median_after(mid)[0] / barscore - 1) < 0.9475:
        lo = mid
    else:
        hi = mid
print(f"{100*0.5*(lo+hi):.2f} %")
print("  +1.3997 % to the bar as published      -> prefill cut needed: ", end="")
lo, hi = 0.0, 1.0
for _ in range(100):
    mid = 0.5 * (lo + hi)
    if 100 * (median_after(mid)[0] / barscore - 1) < 1.3997:
        lo = mid
    else:
        hi = mid
print(f"{100*0.5*(lo+hi):.2f} %")
print("\n  best prefill result anyone has ever published: -4.97 % (5cdc9c17, BitWonka)")
print("  our own prefill sits at about -0.01 % vs the bar: we have never shipped one.")
