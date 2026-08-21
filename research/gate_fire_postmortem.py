#!/usr/bin/env python3
"""87b654b2 post-mortem: reconstruct the per-prompt margin-gate fire rate.

The gate clamps the offered draft count to marginGateDepth = 3.  With a fire
rate f, the mean offered draft length is approximately

    d_gated = f * 3 + (1 - f) * d_crown        (clamped rounds offer AT MOST 3,
                                                so this f is a LOWER bound)
"""

# f04b102e (crown, gate absent) vs 87b654b2 (gate on, t = 9.4375, depth 3)
rows = [
    # prompt      d_crown       d_gated     cand_delta_pct   M_crown
    ("plutarch",  0.1540041068, 0.1540041068,  -0.0425),
    ("drama",     2.2976190476, 2.2301587302,  -0.3548),
    ("travel",    2.6556603774, 2.4930232558,  -0.8505),
    ("beagle",    4.3818181818, 3.9745762712,  -3.0920),
    ("republic",  4.9892473118, 3.9819819820,  -5.6023),
    ("essays",    5.0869565217, 4.1792452830,  -3.2155),
    ("medicine",  5.2555555556, 2.8652482270,  -9.9662),
    ("botany",    6.1481481481, 3.8547008547,  -8.7379),
]
CLAMP = 3.0

print("=" * 84)
print("87b654b2  margin gate t=9.4375 depth=3   vs  f04b102e crown")
print("=" * 84)
print(f"{'prompt':<10} {'M_crown':>8} {'d_crown':>8} {'d_gated':>8} "
      f"{'d drop %':>9} {'fire f':>8} {'cand %':>8}")

pts = []
for name, da, db, cand in rows:
    drop = 100 * (db / da - 1)
    if da > CLAMP:
        f = (da - db) / (da - CLAMP)
    else:
        f = float("nan")          # clamp does not bind
    pts.append((name, f, cand, 1 + da))
    fs = f"{min(f,1.0):>8.3f}" if f == f else f"{'n/a':>8}"
    print(f"{name:<10} {1+da:>8.3f} {da:>8.4f} {db:>8.4f} "
      f"{drop:>8.1f} % {fs} {cand:>7.2f} %")

print()
print("  LOCAL fired share at t=9.4375 (E99, one public fixture):")
print("     cap 8 -> 21/81 = 0.259     cap 6 -> 22/90 = 0.244")
print("     cap 5 -> 21/98 = 0.214     cap 4 -> 21/113 = 0.186")
print()
usable = [(n, min(f, 1.0), c) for n, f, c, _ in pts if f == f]
mean_f = sum(f for _, f, _ in usable) / len(usable)
print(f"  RANKED fire share on the five clamp-binding prompts: mean {mean_f:.3f}, "
      f"range {min(f for _,f,_ in usable):.3f} to {max(f for _,f,_ in usable):.3f}")
print(f"  => the gate fired about {mean_f/0.259:.1f}x more often than the "
      "public fixture predicted")

print()
print("=" * 84)
print("damage tracks fire rate")
print("=" * 84)
import statistics
xs = [f for _, f, _ in usable]
ys = [c for _, _, c in usable]
n = len(xs)
mx, my = sum(xs)/n, sum(ys)/n
sxy = sum((a-mx)*(b-my) for a, b in zip(xs, ys))
sxx = sum((a-mx)**2 for a in xs)
slope = sxy/sxx
r = sxy / (sxx**0.5 * sum((b-my)**2 for b in ys)**0.5)
print(f"  candidate slowdown per unit fire rate : {slope:+.2f} % ")
print(f"  correlation r                          : {r:+.3f}  (n={n})")
print(f"  intercept at f=0                       : {my - slope*mx:+.3f} %")

print()
print("=" * 84)
print("WHY: the threshold was placed in a GAP of the public margin distribution")
print("=" * 84)
print("""  E99 measured the top-2 margin distribution on ONE public fixture:

      leg cap    p05     p25     p50     p75     p95
            8   0.731   9.906  14.250  15.922  17.591
            6   0.825   9.250  14.250  15.750  18.100
            5   1.050  11.125  14.625  16.000  18.325
            4   1.056  11.906  14.906  16.125  18.519

  Edward recorded that the distribution is strongly bimodal and that
  t = 9.4375 "sits in a gap in the data".  The fired share was flat across
  t = 8.25 and t = 9.4375 (both +3.22 % at cap 8), and that flatness was read
  as robustness.

  It is the opposite.  A threshold in a gap means a LARGE MASS sits just on
  one side of it.  Local insensitivity to t is therefore evidence of
  FRAGILITY to any shift of the distribution, not of robustness: a small
  shift carries that whole mass across the threshold at once.

  The hidden prompts have systematically lower top-2 margins than the public
  fixture, so the mass crossed, and the gate fired on 30 % to 100 % of rounds
  instead of 26 %.""")

print()
print("=" * 84)
print("SCORE ARITHMETIC")
print("=" * 84)
print("  f04b102e  published 3.32824629   serial-free 3.33711595")
print("  87b654b2  published 3.12600524   serial-free 3.13029605")
print("  gap       published -6.0765 %    serial-free -6.1976 %")
print("  candidate leg slower by 3.98 % (sd 3.80); serial leg +0.11 % (sd 0.33)")
print()
print("  This is a real mechanism failure, not the serial lottery:")
print("  the schedule differs on 7 of 8 prompts and the candidate leg carries")
print("  the whole loss.")
