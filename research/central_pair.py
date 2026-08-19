#!/usr/bin/env python3
"""The pooled legs are garbage regressors for score because plutarch (zero weight)
dominates them. Redo everything on the CENTRAL PAIR only.

Also: is the beagle-vs-medicine asymmetry real across all top rivals?
beagle n=4.53 -> we are ~0.35% behind.  medicine n=4.77 -> we are at parity.
That breaks a monotone-in-n story and is the key puzzle.
"""
import json
import statistics as st

NAMES = {"919318e1": "beagle", "192fb621": "botany", "4b9e88cd": "drama",
         "a2ea8b60": "essays", "00142a44": "medicine", "c1ec5866": "plutarch",
         "ea82dcb5": "republic", "3b10cb4d": "travel"}
ORDER = ["plutarch", "drama", "travel", "beagle", "medicine", "essays", "republic", "botany"]
PIN = 0.037994794617407023

rows = json.load(open("/tmp/rows_live.json"))
good = [r for r in rows if isinstance(r.get("officialScore"), (int, float))
        and r.get("officialMetrics")
        and r["officialMetrics"].get("prompt_count") == 8]
ours = [r for r in rows if r["id"].startswith("ca9251b8")][0]


def pp(r):
    return {NAMES[p["prompt_sha256"][:8]]: p for p in r["officialMetrics"]["per_prompt"]}


def median_even(v):
    v = sorted(v)
    return (v[3] + v[4]) / 2.0


def corr(a, b):
    ma, mb = st.mean(a), st.mean(b)
    n = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    d = (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5
    return n / d if d else float("nan")


print("=" * 100)
print("A. PER-PROMPT SERIAL DRAW: distribution across all %d scored rows" % len(good))
print(f"   {'prompt':<10} {'mean':>13} {'vs_pin_%':>9} {'sd_%':>7} {'p05_%':>8} {'p95_%':>8}")
serials = {}
for nm in ORDER:
    v = [pp(r)[nm]["serial_seconds_per_token_mean"] for r in good]
    serials[nm] = v
    q = sorted(v)
    print(f"   {nm:<10} {st.mean(v):>13.9f} {100*(st.mean(v)/PIN-1):>+8.3f}% "
          f"{100*st.stdev(v)/st.mean(v):>6.3f}% "
          f"{100*(q[int(.05*(len(q)-1))]/PIN-1):>+7.3f}% {100*(q[int(.95*(len(q)-1))]/PIN-1):>+7.3f}%")
allser = [x for nm in ORDER for x in serials[nm]]
print(f"   {'ALL':<10} {st.mean(allser):>13.9f} {100*(st.mean(allser)/PIN-1):>+8.3f}% "
      f"{100*st.stdev(allser)/st.mean(allser):>6.3f}%")

print()
print("=" * 100)
print("B. SCORE NOISE FROM THE SERIAL DRAW ALONE")
print("   score = (serial_b/cand_b + serial_m/cand_m)/2 when the central pair is")
print("   beagle+medicine. Holding the candidate leg fixed, resample the two serial")
print("   readings from their empirical board-wide distributions.")
sb, sm = serials["beagle"], serials["medicine"]
cvb, cvm = st.stdev(sb) / st.mean(sb), st.stdev(sm) / st.mean(sm)
o = pp(ours)
rb, rm = o["beagle"]["raw_ratio_of_means"], o["medicine"]["raw_ratio_of_means"]
w = rb / (rb + rm), rm / (rb + rm)
sd_score = ((w[0] * cvb) ** 2 + (w[1] * cvm) ** 2) ** 0.5
print(f"   beagle serial CV {100*cvb:.3f}%   medicine serial CV {100*cvm:.3f}%")
print(f"   => sd of score from serial draw alone = {100*sd_score:.3f}%  (if independent)")
print(f"   corr(serial_beagle, serial_medicine) across rows = {corr(sb, sm):+.4f}")
cc = corr(sb, sm)
sd_corr = ((w[0]*cvb)**2 + (w[1]*cvm)**2 + 2*cc*w[0]*w[1]*cvb*cvm) ** 0.5
print(f"   => with that correlation: sd = {100*sd_corr:.3f}%")
print(f"   organizer paired sigma_score (fixture calibration) = 0.078%")
print("   NOTE: the two must be reconciled by serial/candidate common-mode within a")
print("         session, which pairing cancels. Check that directly:")
for nm in ["beagle", "medicine"]:
    s = [pp(r)[nm]["serial_seconds_per_token_mean"] for r in good]
    c = [pp(r)[nm]["mtp_seconds_per_token_mean"] for r in good]
    print(f"      {nm:<9} corr(serial, candidate) across rows = {corr(s, c):+.4f}")
    comp = [r for r in good if r["officialScore"] > 3.20]
    s2 = [pp(r)[nm]["serial_seconds_per_token_mean"] for r in comp]
    c2 = [pp(r)[nm]["mtp_seconds_per_token_mean"] for r in comp]
    print(f"      {nm:<9} corr restricted to score>3.20 (n={len(comp)}) = {corr(s2, c2):+.4f}")

print()
print("=" * 100)
print("C. DOES THE CENTRAL-PAIR SERIAL DRAW PREDICT SCORE? (competitive cluster)")
comp = [r for r in good if r["officialScore"] > 3.20]
S = [r["officialScore"] for r in comp]
cps = [(pp(r)["beagle"]["serial_seconds_per_token_mean"]
        + pp(r)["medicine"]["serial_seconds_per_token_mean"]) / 2 for r in comp]
cpc = [(pp(r)["beagle"]["mtp_seconds_per_token_mean"]
        + pp(r)["medicine"]["mtp_seconds_per_token_mean"]) / 2 for r in comp]
pool = [r["officialMetrics"]["baseline_serial_seconds_per_token_mean"] for r in comp]
print(f"   n={len(comp)}")
print(f"   corr(score, central-pair serial)     = {corr(S, cps):+.4f}   <-- the real channel")
print(f"   corr(score, central-pair candidate)  = {corr(S, cpc):+.4f}")
print(f"   corr(score, POOLED serial mean)      = {corr(S, pool):+.4f}   (diluted)")
mb = st.mean(cps)
b = sum((x-mb)*(y-st.mean(S)) for x, y in zip(cps, S))/sum((x-mb)**2 for x in cps)
print(f"   slope: +0.1% central-pair serial -> {100*(b*mb*0.001)/st.mean(S):+.4f}% score")
print("   (theory says exactly +0.100%)")

print()
print("=" * 100)
print("D. THE BEAGLE/MEDICINE PUZZLE: every top rival vs us, wide prompts only")
tops = sorted(good, key=lambda r: -r["officialScore"])[:12]
print(f"   {'id':<10} {'score':>11} " + "".join(f"{n[:8]:>10}" for n in
      ["beagle", "medicine", "essays", "republic", "botany", "drama", "travel"]))
print("   candidate s/tok advantage over us, %  (positive = rival faster)")
for r in tops:
    q = pp(r)
    line = f"   {r['id'][:8]:<10} {r['officialScore']:>11.6f} "
    for nm in ["beagle", "medicine", "essays", "republic", "botany", "drama", "travel"]:
        adv = 100*(o[nm]["mtp_seconds_per_token_mean"]/q[nm]["mtp_seconds_per_token_mean"] - 1)
        line += f"{adv:>+9.3f}%"
    print(line)
print()
print("   our n / their n (effective_mean_draft_len) -- is the deficit an ACCEPTANCE")
print("   difference or a SPEED difference at equal acceptance?")
print(f"   {'id':<10} " + "".join(f"{n[:8]:>10}" for n in
      ["beagle", "medicine", "essays", "republic", "botany"]))
for r in [ours] + tops[:6]:
    q = pp(r)
    tag = "US" if r is ours else r["id"][:8]
    print(f"   {tag:<10} " + "".join(f"{q[nm]['effective_mean_draft_len']:>10.4f}"
                                     for nm in ["beagle", "medicine", "essays", "republic", "botany"]))
print()
print("   n IDENTICAL to ours on 8/8 for these rows?  (item 119 claim, re-verified)")
for r in tops[:8]:
    q = pp(r)
    same = sum(1 for nm in ORDER
               if q[nm]["effective_mean_draft_len"] == o[nm]["effective_mean_draft_len"])
    samend = sum(1 for nm in ORDER
                 if q[nm]["non_drafting_round_count"] == o[nm]["non_drafting_round_count"])
    print(f"     {r['id'][:8]}  n match {same}/8   non_drafting match {samend}/8")
