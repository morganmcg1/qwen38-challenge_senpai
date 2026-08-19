#!/usr/bin/env python3
"""Board-wide analysis of the SERIAL LEG as a score determinant.

The list payload carries officialMetrics for every scored row, so we get
414 independent measurements of `baseline_serial_seconds_per_token_mean`
(the mean of the 8 per-prompt serial numerators) for free.

Questions:
 1. How wide is the serial draw, and how much score does it buy?
 2. Is OUR serial draw systematically low across our own submission history
    (=> tree-attributable) or low only on the scored row (=> session noise)?
 3. Does selection-on-score enrich the board top in high serial draws?
 4. Which explains score better: candidate speed or serial draw?
"""
import json
import statistics as st
from collections import defaultdict

PIN = 0.037994794617407023
rows = json.load(open("/tmp/rows_live.json"))
sc = [r for r in rows if isinstance(r.get("officialScore"), (int, float))
      and r.get("officialMetrics")]
print("scored rows with metrics:", len(sc))

def f(r, k):
    return r["officialMetrics"].get(k)

# keep only rows with both legs and the standard geometry
good = [r for r in sc
        if isinstance(f(r, "baseline_serial_seconds_per_token_mean"), float)
        and isinstance(f(r, "candidate_mtp_seconds_per_token_mean"), float)
        and f(r, "prompt_count") == 8 and f(r, "decode_tokens") == 512]
print("usable rows:", len(good))

ser = [f(r, "baseline_serial_seconds_per_token_mean") for r in good]
cand = [f(r, "candidate_mtp_seconds_per_token_mean") for r in good]
score = [r["officialScore"] for r in good]

print()
print("=" * 100)
print("1. SERIAL DRAW DISTRIBUTION over %d scored submissions" % len(good))
q = sorted(ser)
def pc(p):
    return q[int(p * (len(q) - 1))]
print(f"   mean   {st.mean(ser):.9f}   ({100*(st.mean(ser)/PIN-1):+.3f}% vs fixture pin)")
print(f"   sd     {st.stdev(ser):.9f}   = {100*st.stdev(ser)/st.mean(ser):.3f}% of mean")
print(f"   min    {q[0]:.9f}  ({100*(q[0]/PIN-1):+.3f}%)")
print(f"   p05    {pc(.05):.9f}  ({100*(pc(.05)/PIN-1):+.3f}%)")
print(f"   p50    {pc(.50):.9f}  ({100*(pc(.50)/PIN-1):+.3f}%)")
print(f"   p95    {pc(.95):.9f}  ({100*(pc(.95)/PIN-1):+.3f}%)")
print(f"   max    {q[-1]:.9f}  ({100*(q[-1]/PIN-1):+.3f}%)")
print(f"   p95-p05 span = {100*(pc(.95)/pc(.05)-1):.3f}% of score, for IDENTICAL candidate code")

print()
print("=" * 100)
print("2. OUR OWN SUBMISSION HISTORY (solverUsername == ours) - serial draw per row")
mine = [r for r in good if r.get("solverUsername") == good and False]
# identify our account from the known row
ourrow = [r for r in rows if r["id"].startswith("ca9251b8")][0]
acct = ourrow.get("solverAccountId")
user = ourrow.get("solverUsername")
print(f"   our account {acct}  username {user!r}")
mine = [r for r in good if r.get("solverAccountId") == acct]
mine.sort(key=lambda r: r["createdAt"])
print(f"   our scored rows: {len(mine)}")
for r in mine[-18:]:
    s = f(r, "baseline_serial_seconds_per_token_mean")
    c = f(r, "candidate_mtp_seconds_per_token_mean")
    print(f"     {r['createdAt'][:19]}  {r['id'][:8]}  score {r['officialScore']:.6f}"
          f"  serial {s:.8f} ({100*(s/PIN-1):+.3f}%)  cand {c:.8f}  {r['status']}")
if mine:
    ms = [f(r, "baseline_serial_seconds_per_token_mean") for r in mine]
    others = [f(r, "baseline_serial_seconds_per_token_mean") for r in good
              if r.get("solverAccountId") != acct]
    print(f"   OUR serial mean   {st.mean(ms):.9f}  n={len(ms)}  sd={st.stdev(ms) if len(ms)>1 else 0:.9f}")
    print(f"   OTHERS serial mean{st.mean(others):.9f}  n={len(others)}  sd={st.stdev(others):.9f}")
    d = st.mean(ms) - st.mean(others)
    se = (st.variance(ms) / len(ms) + st.variance(others) / len(others)) ** 0.5
    print(f"   difference {100*d/st.mean(others):+.4f}%   t = {d/se:+.2f}")

print()
print("=" * 100)
print("3. SELECTION EFFECT: serial draw by score decile")
pairs = sorted(zip(score, ser), key=lambda t: -t[0])
n = len(pairs)
for i in range(10):
    a, b = i * n // 10, (i + 1) * n // 10
    chunk = pairs[a:b]
    print(f"   decile {i+1} (score {chunk[0][0]:.4f}..{chunk[-1][0]:.4f})  "
          f"serial mean {st.mean([c[1] for c in chunk]):.8f} "
          f"({100*(st.mean([c[1] for c in chunk])/PIN-1):+.3f}%)")

print()
print("   TOP-20 rows vs REST:")
top20 = pairs[:20]
rest = pairs[20:]
print(f"     top20 serial {st.mean([c[1] for c in top20]):.9f}"
      f"  ({100*(st.mean([c[1] for c in top20])/PIN-1):+.3f}%)")
print(f"     rest  serial {st.mean([c[1] for c in rest]):.9f}"
      f"  ({100*(st.mean([c[1] for c in rest])/PIN-1):+.3f}%)")
d = st.mean([c[1] for c in top20]) - st.mean([c[1] for c in rest])
se = (st.variance([c[1] for c in top20]) / 20
      + st.variance([c[1] for c in rest]) / len(rest)) ** 0.5
print(f"     enrichment {100*d/st.mean([c[1] for c in rest]):+.4f}%   t = {d/se:+.2f}")

print()
print("=" * 100)
print("4. WHAT EXPLAINS SCORE? restrict to the competitive cluster (score > 3.15)")
comp = [r for r in good if r["officialScore"] > 3.15]
print("   n =", len(comp))
S = [r["officialScore"] for r in comp]
X = [f(r, "baseline_serial_seconds_per_token_mean") for r in comp]
Y = [f(r, "candidate_mtp_seconds_per_token_mean") for r in comp]

def corr(a, b):
    ma, mb = st.mean(a), st.mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5
    return num / den if den else float("nan")

print(f"   corr(score, serial_leg)     = {corr(S, X):+.4f}")
print(f"   corr(score, candidate_leg)  = {corr(S, Y):+.4f}")
print(f"   corr(serial_leg, cand_leg)  = {corr(X, Y):+.4f}")
print("   -> if the box were simply fast/slow, serial and candidate would move")
print("      together (positive corr) and pairing would cancel it.")

print()
print("=" * 100)
print("5. TIME DRIFT of the serial draw (is the box getting faster/slower?)")
byday = defaultdict(list)
for r in good:
    byday[r["createdAt"][:10]].append(f(r, "baseline_serial_seconds_per_token_mean"))
for d_ in sorted(byday):
    v = byday[d_]
    print(f"   {d_}  n={len(v):4d}  serial mean {st.mean(v):.9f} "
          f"({100*(st.mean(v)/PIN-1):+.3f}%)  sd {100*(st.stdev(v)/st.mean(v) if len(v)>1 else 0):.3f}%")

print()
print("=" * 100)
print("6. THE de7981ae ANOMALY: 10% faster candidate, still below #1")
NAMES = {"919318e1": "beagle", "192fb621": "botany", "4b9e88cd": "drama",
         "a2ea8b60": "essays", "00142a44": "medicine", "c1ec5866": "plutarch",
         "ea82dcb5": "republic", "3b10cb4d": "travel"}
ORDER = ["plutarch", "drama", "travel", "beagle", "medicine", "essays", "republic", "botany"]
def pp(r):
    return {NAMES[p["prompt_sha256"][:8]]: p for p in r["officialMetrics"]["per_prompt"]}
a = [r for r in good if r["id"].startswith("de7981ae")][0]
b = ourrow
pa, pb = pp(a), pp(b)
print(f"   {'prompt':<10} {'ours_raw':>10} {'de79_raw':>10} {'raw_%':>9} "
      f"{'ours_cand':>11} {'de79_cand':>11} {'cand_%':>9} {'ours_n':>8} {'de79_n':>8} {'ours_nd':>8} {'de79_nd':>8}")
for nm in ORDER:
    print(f"   {nm:<10} {pb[nm]['raw_ratio_of_means']:>10.5f} {pa[nm]['raw_ratio_of_means']:>10.5f} "
          f"{100*(pa[nm]['raw_ratio_of_means']/pb[nm]['raw_ratio_of_means']-1):>+8.2f}% "
          f"{pb[nm]['mtp_seconds_per_token_mean']:>11.8f} {pa[nm]['mtp_seconds_per_token_mean']:>11.8f} "
          f"{100*(pb[nm]['mtp_seconds_per_token_mean']/pa[nm]['mtp_seconds_per_token_mean']-1):>+8.2f}% "
          f"{pb[nm]['effective_mean_draft_len']:>8.3f} {pa[nm]['effective_mean_draft_len']:>8.3f} "
          f"{pb[nm]['non_drafting_round_count']:>8d} {pa[nm]['non_drafting_round_count']:>8d}")
print(f"   de7981ae score {a['officialScore']:.8f} status {a['status']} user {a.get('solverUsername')}")
print("   note:", (a.get("note") or "")[:600].replace("\n", " | "))
