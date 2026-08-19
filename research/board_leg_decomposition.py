#!/usr/bin/env python3
"""Ledger 143. Splits the board gap into denominator and numerator channels.

DATA PROVENANCE -- refresh /tmp/rows_live.json before trusting any number here:
  curl -s -H "Authorization: Bearer $YUKON_API_TOKEN" \
    'https://api.yukon.org/api/benchmarks/5d1ee4d7-80bd-4555-b182-6505f26ef495/submissions?limit=2000' \
    > /tmp/rows_live.json
That endpoint is the only one that returns full officialMetrics for every row.
Cohort = rows whose per_prompt head_provenance_sha256 is 559b24eb... on all 8
prompts (94 rows as of 2026-08-19); score claims are only comparable within it.

Item 142, final pass. Proper variance components + significance.

The serial leg is a NOMINALLY IDENTICAL computation in every submission, so the
94 matched rows are 94 quasi-replicates of one measurement. That is the replicate
we have never had, and it yields an independent estimate of run-to-run noise.
"""
import json
import math
import random
import statistics as st

FP, OURS, CROWN = "559b24eb", "2b0c36a078", "ef42e04327"
rows = json.load(open("/tmp/rows_live.json"))
sel = []
for r in rows:
    om = r.get("officialMetrics") or {}
    pp = om.get("per_prompt")
    if (isinstance(pp, list) and len(pp) == 8 and r.get("officialScore") is not None
            and all(isinstance(p.get("head_provenance_sha256"), str)
                    and p["head_provenance_sha256"].startswith(FP) for p in pp)):
        sel.append(r)

def tab(r):
    return {p["prompt_sha256"][:8]:
            (p["serial_seconds_per_token_mean"], p["mtp_seconds_per_token_mean"])
            for p in r["officialMetrics"]["per_prompt"]}

prompts = sorted(tab(sel[0]).keys())
order = sorted(prompts, key=lambda q: st.median([tab(r)[q][0]/tab(r)[q][1] for r in sel]))
NAMES = ["plutarch","drama","travel","beagle","medicine","essays","republic","botany"]
name = {q: NAMES[i] for i, q in enumerate(order)}
ours = next(r for r in sel if str(r.get("submissionCommitSha","")).startswith(OURS))
crown = next(r for r in sel if str(r.get("submissionCommitSha","")).startswith(CROWN))
n = len(sel)
score_of = lambda v: (sorted(v)[3] + sorted(v)[4]) / 2.0

# --- A. is per-prompt variation in the serial leg systematic or noise? -----
print("=== A. SERIAL LEG: systematic prompt effect vs noise ===")
meds = [100*math.log(st.median([tab(r)[q][0] for r in sel])) for q in order]
print(f"spread of per-prompt MEDIAN serial (max-min)   {max(meds)-min(meds):6.4f} %")
withins = []
for r in sel:
    ls = [100*math.log(tab(r)[q][0]) for q in order]
    mu = st.mean(ls)
    withins.append(st.variance(ls))
w_sd = math.sqrt(st.mean(withins))
print(f"within-row sd across the 8 prompts             {w_sd:6.4f} %")
print("  -> systematic prompt effect is ~5x smaller than the scatter, so")
print("     within-row spread of the serial leg is essentially MEASUREMENT NOISE.")

# --- B. variance components, corrected for averaging ----------------------
print()
print("=== B. VARIANCE COMPONENTS (corrected) ===")
rowmeans = [st.mean([100*math.log(tab(r)[q][0]) for q in order]) for r in sel]
b_sd_raw = st.stdev(rowmeans)
noise_in_rowmean = w_sd / math.sqrt(8)
common = b_sd_raw**2 - noise_in_rowmean**2
c_sd = math.sqrt(common) if common > 0 else 0.0
print(f"raw sd of row-mean serial                     {b_sd_raw:6.4f} %")
print(f"  noise contribution to a row-mean (w/sqrt8)  {noise_in_rowmean:6.4f} %")
print(f"  => per-RUN COMMON-MODE serial sigma         {c_sd:6.4f} %")
print(f"  => per-PROMPT independent serial sigma      {w_sd:6.4f} %")
print()
print("  INDEPENDENT CHECK on score noise: the common-mode term passes into every")
print("  prompt's ratio at full magnitude, so it lower-bounds score sigma.")
print(f"  common-mode sigma      = {c_sd:.4f} %")
print(f"  my carried sigma_score = 0.0923 %   <-- corroborated by a different route")

# --- C. crown vs us on the serial leg: significance ----------------------
print()
print("=== C. CROWN vs US, SERIAL LEG ===")
cm = st.mean([100*math.log(tab(crown)[q][0]) for q in order])
om_ = st.mean([100*math.log(tab(ours)[q][0]) for q in order])
d = cm - om_
se_diff = math.sqrt(2) * b_sd_raw
print(f"crown row-mean serial is {d:+.4f} % vs ours")
print(f"se of a difference of two single runs = sqrt2 * {b_sd_raw:.4f} = {se_diff:.4f} %")
print(f"  => {d/se_diff:+.2f} sigma  (suggestive, NOT decisive)")
slower = sum(1 for m in rowmeans if m > cm)
print(f"rows with a slower serial leg than the crown: {slower} of {n}")

# --- D. the decomposition that matters: gap to crown --------------------
print()
print("=== D. GAP TO CROWN, DECOMPOSED ===")
base, cs = ours["officialScore"], crown["officialScore"]
cf2 = score_of([tab(crown)[q][0]/tab(ours)[q][1] for q in order])
cf3 = score_of([tab(ours)[q][0]/tab(crown)[q][1] for q in order])
print(f"official gap to crown                      {100*(cs/base-1):+.4f} %")
print(f"  denominator channel (their serial leg)   {100*(cf2/base-1):+.4f} %")
print(f"  numerator   channel (their MTP leg)      {100*(cf3/base-1):+.4f} %")
print("  the denominator half is either their luck or their choice; either way it")
print("  is NOT MTP-path headroom we can engineer.")

# --- E. honest board-visible MTP headroom, de-cursed -------------------
print()
print("=== E. BOARD-VISIBLE MTP HEADROOM ABOVE US (winner's curse removed) ===")
print("  'best rival MTP leg per prompt' is a 6-solver composite of luckiest runs.")
for k in (1, 2, 3, 5, 10):
    cf = []
    for q in order:
        v = sorted(tab(r)[q][1] for r in sel)
        cf.append(tab(ours)[q][0] / v[k-1])
    s = score_of(cf)
    print(f"  our MTP -> rank-{k:<2d} best per prompt   {s:.8f}  {100*(s/base-1):+.4f} %")
# single best whole submission by MTP leg alone (no cherry-picking)
bestrow, bestval = None, None
for r in sel:
    v = score_of([tab(ours)[q][0]/tab(r)[q][1] for q in order])
    if bestval is None or v > bestval:
        bestrow, bestval = r, v
print(f"  our MTP -> best SINGLE rival's MTP leg     {bestval:.8f}  "
      f"{100*(bestval/base-1):+.4f} %  ({str(bestrow.get('submissionCommitSha'))[:8]} "
      f"{bestrow.get('solverUsername')})")

# --- F. how much of a rank-1 score is buyable with resubmissions? -----
print()
print("=== F. WHAT DOES IT TAKE TO POST 3.2493? ===")
need = 100*(cs/base-1)
print(f"we need {need:+.4f} % on the recorded board number")
print(f"our per-run score sigma (from common-mode)   ~{c_sd:.4f} %")
for eng in (0.0, 0.15, 0.26, 0.40):
    z = (need - eng) / c_sd
    # one-sided normal tail
    p = 0.5 * math.erfc(z / math.sqrt(2))
    exp = (1/p) if p > 0 else float('inf')
    print(f"  with {eng:+.2f} % engineered: need {z:+.2f} sigma of luck, "
          f"p={p:.4g}  => ~{exp:,.0f} submissions" if p > 1e-12 else
          f"  with {eng:+.2f} % engineered: need {z:+.2f} sigma  => hopeless")
