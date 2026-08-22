"""What does a real measurement-mode flip look like, per prompt?

Find declared byte-identical resamples on the board and read the per-prompt
time difference directly.  That gives an empirical mode/noise null with no
model.  Then test whether cf9a9eda's per-prompt deltas match that null shape
or are a real regression on the drafting path.
"""
import json
import math
import statistics as st

BOARD = "/tmp/yukon-board/full.json"
SHA = {"919318e1": "beagle", "192fb621": "botany", "4b9e88cd": "drama",
       "a2ea8b60": "essays", "00142a44": "medicine", "c1ec5866": "plutarch",
       "ea82dcb5": "republic", "3b10cb4d": "travel"}
ORDER = ["plutarch", "drama", "travel", "beagle", "medicine", "essays",
         "republic", "botany"]
W76 = {"plutarch": -0.3852, "drama": 0.0215, "travel": 0.4945,
       "beagle": 0.2068, "medicine": -0.1480, "republic": -0.0917,
       "essays": -0.0041, "botany": -0.0939}
W83 = {"beagle": 0.4862, "medicine": 0.2508, "essays": 0.1598,
       "botany": 0.0124, "republic": 0.0100, "plutarch": 0.0,
       "drama": 0.0, "travel": 0.0}

rows = json.load(open(BOARD))["submissions"]
by = {}
for r in rows:
    pp = (r.get("officialMetrics") or {}).get("per_prompt")
    if not pp:
        continue
    d = {}
    for e in pp:
        nm = SHA.get((e.get("prompt_sha256") or "")[:8])
        if nm:
            d[nm] = e["mtp_seconds_per_token_mean"] * 1000.0
    if len(d) != 8:
        continue
    idx = sum(W76[k] * 100.0 * math.log(v / 1000.0) for k, v in d.items())
    by[r["id"][:8]] = {"p": d, "score": r.get("officialScore"),
                       "note": (r.get("note") or "").replace("\n", " ")[:120],
                       "idx": idx, "at": (r.get("resolvedAt") or "")[:19]}

print("A. declared resamples / redraws on the board")
print()
hits = [(k, v) for k, v in by.items()
        if any(w in v["note"].lower() for w in
               ("resample", "redraw", "variance", "identical", "replay of"))]
hits.sort(key=lambda kv: kv[1]["at"])
for k, v in hits:
    print("  %s  %.8f  idx %8.4f  %s"
          % (k, v["score"] or -1, v["idx"], v["note"][:92]))
print()

print("=" * 78)
print("B. crown bc070b7b vs its declared byte-identical resample ec778a91")
print()
a, b = by["bc070b7b"], by["ec778a91"]
print("  prompt      crown_ms  resample_ms    delta %    w76      w83")
tot76 = 0.0
for nm in ORDER:
    d = 100.0 * (b["p"][nm] / a["p"][nm] - 1.0)
    tot76 += W76[nm] * d
    print("  %-9s %9.4f  %10.4f   %+8.3f  %+.4f  %.4f"
          % (nm, a["p"][nm], b["p"][nm], d, W76[nm], W83[nm]))
print("  index delta %+.4f (measured %+.4f)" % (tot76, b["idx"] - a["idx"]))
print()

print("=" * 78)
print("C. cf9a9eda against our own fast receipts, and the resample null")
print()
print("  prompt      w83     vs 44559d02  vs b8b8b860  vs bc070b7b   "
      "crown->resample")
for nm in ORDER:
    c = by["cf9a9eda"]["p"][nm]
    d1 = 100.0 * (c / by["44559d02"]["p"][nm] - 1.0)
    d2 = 100.0 * (c / by["b8b8b860"]["p"][nm] - 1.0)
    d3 = 100.0 * (c / by["bc070b7b"]["p"][nm] - 1.0)
    dr = 100.0 * (by["ec778a91"]["p"][nm] / by["bc070b7b"]["p"][nm] - 1.0)
    print("  %-9s %.4f    %+8.3f    %+8.3f    %+8.3f       %+8.3f"
          % (nm, W83[nm], d1, d2, d3, dr))
print()

print("=" * 78)
print("D. is the cf9a9eda shape proportional to DRAFTING work?")
print()
F12_R = {"plutarch": 487, "drama": 252, "travel": 212, "beagle": 110,
         "republic": 93, "essays": 92, "medicine": 90, "botany": 81}
NONDRAFT = {"plutarch": 449, "drama": 0, "travel": 0, "beagle": 0,
            "republic": 0, "essays": 0, "medicine": 0, "botany": 0}
DEPTH = {"plutarch": 0.154, "drama": 2.298, "travel": 2.656, "beagle": 4.382,
         "medicine": 5.256, "essays": 5.087, "republic": 4.989,
         "botany": 6.148}
print("  prompt     rounds  draft  width   observed   mode-flip   unit/round"
      "  unit/row")
obs, mfl, prd, pwd = [], [], [], []
for nm in ORDER:
    R, nd = F12_R[nm], NONDRAFT[nm]
    dr, w = R - nd, DEPTH[nm] + 1.0
    tot = 512.0 * by["cf9a9eda"]["p"][nm]
    o = 100.0 * (by["cf9a9eda"]["p"][nm] / by["44559d02"]["p"][nm] - 1.0)
    m = 100.0 * 0.82 * dr / tot
    pr, pw = 100.0 * dr / tot, 100.0 * dr * w / tot
    obs.append(o)
    mfl.append(m)
    prd.append(pr)
    pwd.append(pw)
    print("  %-9s  %4d   %4d  %5.3f   %+7.3f    %+7.3f     %7.4f   %7.4f"
          % (nm, R, dr, w, o, m, pr, pw))


def fit(x, y):
    mx, my = st.mean(x), st.mean(y)
    sxx = sum((v - mx) ** 2 for v in x)
    sxy = sum((v - mx) * (w - my) for v, w in zip(x, y))
    b1 = sxy / sxx
    b0 = my - b1 * mx
    ss = sum((w - (b0 + b1 * v)) ** 2 for v, w in zip(x, y))
    tt = sum((w - my) ** 2 for w in y)
    return b0, b1, 1.0 - ss / tt


print()
for lbl, x in (("per drafting round", prd), ("per verified row", pwd),
               ("the mode-flip shape", mfl)):
    b0, b1, r2 = fit(x, obs)
    print("  observed ~ %-20s intercept %+7.4f  slope %+8.4f  R2 %.4f"
          % (lbl, b0, b1, r2))
print()
b0, b1, _ = fit(prd, obs)
print("  implied %.4f ms per drafting round (a mode flip is 0.82 ms)" % b1)
b0, b1, _ = fit(pwd, obs)
print("  implied %.4f ms per verified row" % b1)

print()
print("=" * 78)
print("E. F83-weighted candidate-leg regression of cf9a9eda")
print()
sw = sum(W83.values())
for R in ("44559d02", "b8b8b860", "bc070b7b", "f04b102e", "7bef7d4c"):
    tot = sum(W83[nm] * 100.0 * (by["cf9a9eda"]["p"][nm] / by[R]["p"][nm] - 1.0)
              for nm in ORDER) / sw
    print("  vs %-9s  %+7.3f %%  (their published %.8f)"
          % (R, tot, by[R]["score"]))
