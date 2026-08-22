#!/usr/bin/env python3
"""F201 - decompose every published-median move into its candidate leg and its
serial leg, under the Rule 116 median-pair identity.

The serial leg is the runner-owned pinned build.  Candidate edits cannot move
it (senpai/verify-ranked-score-boundary.sh enforces exactly this).  Any serial
component of a published-median difference is therefore run-to-run noise on the
runner and must NOT be attributed to the mechanism.
"""
import json
import math
import sys

BOARD = "/tmp/yukon-board/full.json"
NAME = {"919318e1": "beagle", "192fb621": "botany", "4b9e88cd": "drama",
        "a2ea8b60": "essays", "00142a44": "medicine", "c1ec5866": "plutarch",
        "ea82dcb5": "republic", "3b10cb4d": "travel"}
W = {"beagle": 0.478, "essays": 0.522}


def load():
    with open(BOARD) as fh:
        return json.load(fh)["submissions"]


def find(subs, p):
    for s in subs:
        if s["id"].startswith(p):
            return s
    raise SystemExit("not found " + p)


def pp(row):
    return {NAME[e["prompt_sha256"][:8]]: e
            for e in row["officialMetrics"]["per_prompt"]
            if e["prompt_sha256"][:8] in NAME}


def contrast(a8, b8, label):
    subs = load()
    A, B = find(subs, a8), find(subs, b8)
    pa, pb = pp(A), pp(B)
    cm = sm = 0.0
    cmed = smed = 0.0
    for nm in pa:
        c = 100.0 * (pb[nm]["mtp_seconds_per_token_mean"]
                     / pa[nm]["mtp_seconds_per_token_mean"] - 1.0)
        s = 100.0 * (pb[nm]["serial_seconds_per_token_mean"]
                     / pa[nm]["serial_seconds_per_token_mean"] - 1.0)
        cm += c / 8.0
        sm += s / 8.0
        if nm in W:
            cmed += W[nm] * c
            smed += W[nm] * s
    pub = 100.0 * (B["officialScore"] / A["officialScore"] - 1.0)
    print("%-34s cand8 %+8.4f  candmed %+8.4f | ser8 %+8.4f  sermed %+8.4f"
          " | pub %+8.4f   check %+8.4f"
          % (label, cm, cmed, sm, smed, pub, smed - cmed))
    return dict(cand8=cm, candmed=cmed, ser8=sm, sermed=smed, pub=pub)


print("=" * 128)
print("F201  EVERY RANKED CONTRAST, SPLIT INTO CANDIDATE LEG AND SERIAL LEG")
print("      positive candidate = slower.  'check' = sermed - candmed, which"
      " should reproduce the published median move.")
print("=" * 128)
o = contrast("623e77af", "572b2cc4", "OURS   tight grid, onePass67")
r = contrast("02742bf0", "ed608e64", "RIVAL  tight grid, shipped")
t = contrast("ed608e64", "0b2f0014", "F194   add onePass67 to tight base")
p = contrast("ed608e64", "08b67f12", "F192   probe 0.15 on tight base")
n = contrast("8849fad7"[:0] or "ed608e64", "115c5c50", "F197   prefill swizzle, null")
g = contrast("572b2cc4", "08b67f12", "GAP    ours -> crown")

print()
print("=" * 128)
print("THE TWO ESTIMATES OF THE onePass67 COST UNDER A TIGHT GRID")
print("=" * 128)
print("  F194 direct isolation, one contrast")
print("      candidate 8-prompt mean %+8.4f     candidate medpair %+8.4f"
      % (t["cand8"], t["candmed"]))
print("  ours, difference of two independent tight-grid contrasts")
print("      candidate 8-prompt mean %+8.4f     candidate medpair %+8.4f"
      % (o["cand8"] - r["cand8"], o["candmed"] - r["candmed"]))
print()
print("  Rule 112 null sd, candidate 8-prompt mean, single contrast   0.067 %")
print("  difference of two contrasts                                  0.095 %")
print("  medpair uses 2 of 8 prompts, so its null sd is about 2x that")
print()
print("  BEST ESTIMATE = the direct isolation, %+.4f %% candidate medpair"
      % t["candmed"])
print("  removing the table therefore gains %+.4f %% of published median"
      % (100.0 * (1.0 / (1.0 - t["candmed"] / 100.0) - 1.0)))

print()
print("=" * 128)
print("CORRECTED COMPOSITION LADDER, CANDIDATE-LEG PRICED ONLY")
print("=" * 128)
base = 3.66218564
crown = 3.69071883
gains = [("pb6 tier 1.45, held out replay", 0.024683),
         ("probe 0.25 -> 0.10, 0.5103 gross x 0.95", 0.004848),
         ("Table .onePass67 -> .shipped, F194 candidate leg",
          1.0 / (1.0 - t["candmed"] / 100.0) - 1.0)]
v = base
print("  %-54s %10.5f" % ("572b2cc4 measured", v))
for nm, gg in gains:
    v *= 1.0 + gg
    print("  %-54s %10.5f   (+%.4f %%)" % (nm, v, 100 * gg))
print("  %-54s %10.5f" % ("crown 08b67f12", crown))
print("  margin %+.4f %%" % (100.0 * (v / crown - 1.0)))

print()
print("=" * 128)
print("THE GAP TO THE CROWN, PRICED ON THE CANDIDATE LEG ONLY")
print("=" * 128)
print("  ours -> crown, candidate medpair %+8.4f  (crown is faster by that)"
      % g["candmed"])
print("  ours -> crown, serial   medpair %+8.4f  (runner noise, unattributable)"
      % g["sermed"])
print("  ours -> crown, published median  %+8.4f" % g["pub"])
exp = (1.0 - t["candmed"] / 100.0) * (1.0 + p["candmed"] / 100.0)
print("  explained on the candidate leg: remove table %+.4f, probe 0.15 %+.4f"
      % (-t["candmed"], p["candmed"]))
print("  product %+8.4f   residual on the candidate leg %+8.4f"
      % (100.0 * (exp - 1.0), g["candmed"] - 100.0 * (exp - 1.0)))
