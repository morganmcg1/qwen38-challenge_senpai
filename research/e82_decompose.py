#!/usr/bin/env python3
"""Decompose the E82 island timed session into head-step vs target-step cost."""
import json
import statistics as st

DECLARED_BYTES = 427_742_600
d = json.load(open("research/e82-headcost-islands.json"))
arms = {}
for r in d["legs"]:
    arms.setdefault(r["arm"], []).append(r)


def m(a, k):
    return st.mean(r[k] for r in arms[a])


hdr = ("arm", "rounds", "rows/tok", "dbuild/rd", "vbuild/rd",
       "round_us tot", "dbuild tot", "vbuild tot", "other tot")
print("{:<11}{:>7}{:>10}{:>11}{:>11}{:>14}{:>12}{:>12}{:>11}".format(*hdr))
for a in ("declared", "noislands", "qonly"):
    ro, rpt = m(a, "rounds"), m(a, "rows_per_token")
    db, vb, tot = m(a, "draft_build_us_per_round"), m(a, "verify_build_us_per_round"), m(a, "round_us_total")
    print("{:<11}{:>7.0f}{:>10.4f}{:>11.0f}{:>11.0f}{:>14.0f}{:>12.0f}{:>12.0f}{:>11.0f}".format(
        a, ro, rpt, db, vb, tot, db * ro, vb * ro, tot - db * ro - vb * ro))

dro, nro = m("declared", "rounds"), m("noislands", "rounds")
ddb, ndb = m("declared", "draft_build_us_per_round"), m("noislands", "draft_build_us_per_round")
dtot, ntot = m("declared", "round_us_total"), m("noislands", "round_us_total")
dpct = (ndb - ddb) / ddb * 100.0
conv = dpct / -7.357
share = ddb * dro / dtot

print()
print(f"draft_build per round   {ddb:.0f} -> {ndb:.0f} = {dpct:+.2f}%   (head bytes -7.357%)")
print(f"bytes -> draft_build conversion       = {conv:.4f} x")
print(f"draft_build TOTAL       {ddb*dro:.0f} -> {ndb*nro:.0f} = {(ndb*nro-ddb*dro)/(ddb*dro)*100:+.2f}%")
print(f"round_us TOTAL          {dtot:.0f} -> {ntot:.0f} = {(ntot-dtot)/dtot*100:+.2f}%")
print(f"draft_build share of round_us (declared) = {share*100:.2f}%")

print()
COMP = ("d_pre_us", "d_flush_us", "d_head1_us", "d_submit1_us", "d_chain_us", "d_submit2_us")
print("draft_build components, mean us per round:")
print("{:<11}".format("arm") + "".join(f"{c.replace('_us',''):>13}" for c in COMP))
for a in ("declared", "noislands", "qonly"):
    print("{:<11}".format(a) + "".join(f"{m(a, c):>13.0f}" for c in COMP))
print("{:<11}".format("nois-decl") + "".join(
    f"{m('noislands', c) - m('declared', c):>+13.0f}" for c in COMP))
print("{:<11}".format("qonly-decl") + "".join(
    f"{m('qonly', c) - m('declared', c):>+13.0f}" for c in COMP))

print()
print("per-leg d_submit1_us / d_submit2_us mean (mean is outlier sensitive at n=78..80):")
for r in sorted(d["legs"], key=lambda r: r["started"]):
    print(f"  {r['tag']:<28} d_submit1={r['d_submit1_us']:>8.0f}  d_submit2={r['d_submit2_us']:>8.0f}")

print()
print("--- projection for a BIT-IDENTICAL byte removal (rounds and rows/tok unchanged) ---")
for name, by in (("askeladd mechanism A (target-side)", 5_898_240),
                 ("arm (a) k/v bf16 swap", 5_906_432)):
    p = by / DECLARED_BYTES * 100.0
    dp = p * conv
    print(f"{name:<36} bytes -{p:.3f}%  draft_build {dp:+.3f}%  candidate {dp*share:+.4f}%")
