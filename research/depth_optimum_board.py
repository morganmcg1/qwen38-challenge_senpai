"""Model-free identification of the RANKED optimal draft depth.

The board is a natural experiment: hundreds of solvers ship different depth
schedules, and every scored row reports, per hidden prompt, both the realised
`effective_mean_draft_len` and the realised `mtp_seconds_per_token_mean`.

If a shallower schedule were faster on the ranked M5 host, rows that draft
shallower on a given prompt would decode faster on that prompt.  That is a
direct test of the E127 claim with no cost-curve model and no acceptance model.

Confound: solver quality.  A better solver is faster at every depth.  Handle it
three ways -- restrict to a narrow score band, correct for the F76 measurement
mode, and run a within-solver fixed-effect regression.
"""
import json
import math
import statistics as st

BOARD = "/tmp/yukon-board/full.json"
SHA = {"919318e1": "beagle", "192fb621": "botany", "4b9e88cd": "drama",
       "a2ea8b60": "essays", "00142a44": "medicine", "c1ec5866": "plutarch",
       "ea82dcb5": "republic", "3b10cb4d": "travel"}
RANKED = ["beagle", "medicine", "essays", "botany", "republic"]
W76 = {"plutarch": -0.3852, "drama": 0.0215, "travel": 0.4945,
       "beagle": 0.2068, "medicine": -0.1480, "republic": -0.0917,
       "essays": -0.0041, "botany": -0.0939}

rows = json.load(open(BOARD))["submissions"]
recs = []
for r in rows:
    sc = r.get("officialScore")
    om = r.get("officialMetrics") or {}
    pp = om.get("per_prompt")
    if sc is None or not pp:
        continue
    d = {}
    for e in pp:
        nm = SHA.get((e.get("prompt_sha256") or "")[:8])
        if nm and e.get("mtp_seconds_per_token_mean") and \
                e.get("effective_mean_draft_len") is not None:
            d[nm] = (float(e["effective_mean_draft_len"]),
                     float(e["mtp_seconds_per_token_mean"]))
    if len(d) != 8:
        continue
    idx = sum(W76[k] * 100.0 * math.log(v[1]) for k, v in d.items())
    recs.append({"id": r.get("id", "")[:8], "solver": r.get("solverName", "?"),
                 "score": float(sc), "p": d, "mode": idx,
                 "fast": idx <= -12.9})

print("rows with a full 8-prompt vector: %d" % len(recs))
fast = [r for r in recs if r["fast"]]
print("  fast-mode rows: %d   slow-mode rows: %d"
      % (len(fast), len(recs) - len(fast)))
print()

print("=" * 76)
print("A. depth vs decode time, per ranked prompt, FAST-MODE rows only")
print()
for nm in RANKED:
    pts = [(r["p"][nm][0], r["p"][nm][1] * 1000.0, r["score"]) for r in fast]
    print("  %-9s n=%d  depth range %.2f .. %.2f"
          % (nm, len(pts), min(p[0] for p in pts), max(p[0] for p in pts)))
    edges = [0, 2, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 99]
    for a, b in zip(edges, edges[1:]):
        sel = [p for p in pts if a <= p[0] < b]
        if len(sel) < 4:
            continue
        print("      d in [%4.1f,%4.1f)  n=%3d  median %7.4f ms  "
              "best %7.4f  median score %.4f"
              % (a, b, len(sel), st.median([s[1] for s in sel]),
                 min(s[1] for s in sel), st.median([s[2] for s in sel])))
    print()

print("=" * 76)
print("B. restricted to the TOP score band (officialScore >= 3.28)")
print()
top = [r for r in fast if r["score"] >= 3.28]
print("  n = %d rows" % len(top))
for nm in RANKED:
    pts = [(r["p"][nm][0], r["p"][nm][1] * 1000.0) for r in top]
    edges = [0, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 99]
    out = []
    for a, b in zip(edges, edges[1:]):
        sel = [p for p in pts if a <= p[0] < b]
        if len(sel) < 3:
            continue
        out.append("[%.1f,%.1f) n=%d med %.4f"
                   % (a, b, len(sel), st.median([s[1] for s in sel])))
    print("  %-9s  %s" % (nm, " | ".join(out)))

print()
print("=" * 76)
print("C. WITHIN-SOLVER fixed effect: does one solver get faster when it")
print("   drafts shallower on the same prompt?")
print()
by_solver = {}
for r in fast:
    by_solver.setdefault(r["solver"], []).append(r)
print("  solver                 prompt     n  span   slope ms/depth   corr")
tot_slope = []
for s, rs in sorted(by_solver.items()):
    if len(rs) < 6:
        continue
    for nm in RANKED:
        xs = [r["p"][nm][0] for r in rs]
        ys = [r["p"][nm][1] * 1000.0 for r in rs]
        span = max(xs) - min(xs)
        if span < 0.4:
            continue
        mx, my = st.mean(xs), st.mean(ys)
        sxx = sum((x - mx) ** 2 for x in xs)
        sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        syy = sum((y - my) ** 2 for y in ys)
        if sxx <= 0 or syy <= 0:
            continue
        slope, corr = sxy / sxx, sxy / math.sqrt(sxx * syy)
        tot_slope.append((slope, len(rs), nm, s))
        print("  %-22s %-9s %3d  %5.2f   %+8.4f      %+.3f"
              % (s[:22], nm, len(rs), span, slope, corr))
print()
if tot_slope:
    sl = [t[0] for t in tot_slope]
    pos = sum(1 for x in sl if x > 0)
    print("  %d within-solver slopes: %d positive (deeper=slower), %d negative"
          % (len(sl), pos, len(sl) - pos))
    print("  median slope %+.4f ms per unit depth, mean %+.4f"
          % (st.median(sl), st.mean(sl)))

print()
print("=" * 76)
print("D. the fastest rows on the two heaviest prompts")
print()
for nm in ("beagle", "medicine"):
    pts = [(r["p"][nm][0], r["p"][nm][1] * 1000.0, r["solver"], r["id"])
           for r in fast]
    print("  %s: 12 fastest rows" % nm)
    for d, t, s, i in sorted(pts, key=lambda x: x[1])[:12]:
        print("      %7.4f ms   depth %5.3f   %-18s %s" % (t, d, s[:18], i))
    print()
