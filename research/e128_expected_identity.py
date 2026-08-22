"""The profile-free identity that pins R, and the shape the assumed R demands.

costModelDepth accumulates
    reach_k = prod_{j<=k} p_j          expected += reach_k  for each accepted level
so at termination
    expected = sum_{k=0}^{d-1} prod_{j=0}^{k} p_j = E[accepted drafts this round]

That is EXACTLY the expected accepted-draft count under the same p vector.  No
flatness assumption anywhere.  Combined with the window identity 512 = R + A:

    R = 512 / (1 + E[expected at termination])

So the scheduler's own internal variable pins the ranked round count, provided
the acceptance EMA is calibrated.  Testing that calibration IS E128.

Second result: what positional shape does the ASSUMED R require?  The stopping
rule at depth d gives reach_d <= 0.18(1+e_d)/(1+0.18 d), while e_d = sum of the
first d reaches is fixed by the accounting.  Because reach is non-increasing,
this brackets the profile hard.
"""

HEAD_STEP = 0.18

CASES = [("beagle", 4.3818, 110, 0.4862), ("medicine", 5.2556, 90, 0.2508),
         ("essays", 5.0870, 92, 0.1598), ("botany", 6.1481, 81, 0.0124),
         ("republic", 4.9892, 93, 0.0100), ("drama", 2.2976, 252, 0.0),
         ("travel", 2.6557, 212, 0.0), ("plutarch", 0.1540, 487, 0.0)]

print("=" * 98)
print("IDENTITY  R = 512 / (1 + E[accepted drafts per round])")
print("and E[accepted drafts per round] == the scheduler's own `expected` at termination")
print("=" * 98)
print(f"{'prompt':10s} {'eff':>7} {'assumed R':>10} {'A=512-R':>8} "
      f"{'e_d = A/R':>10} {'implied E[expected]':>20}")
print("-" * 98)
for name, eff, R, w in CASES:
    A = 512.0 - R
    print(f"{name:10s} {eff:7.4f} {R:10d} {A:8.0f} {A/R:10.4f} {A/R:20.4f}")

print()
print("=" * 98)
print("WHAT POSITIONAL SHAPE DOES THE ASSUMED R DEMAND?")
print("=" * 98)
print("reach is non-increasing.  With d levels summing to e_d and each <= 1:")
print("   reach_{d-1} >= d*avg - (d-1)*1        (lowest possible last reach)")
print("The stop rule caps the NEXT reach:")
print("   reach_d <= 0.18*(1+e_d)/(1+0.18*d)")
print("so the acceptance at the stopping position obeys p_d <= reach_d / reach_{d-1}.")
print()
print(f"{'prompt':10s} {'d':>3} {'e_d':>7} {'avg reach':>10} {'reach_{d-1}>=':>13} "
      f"{'geo-mean p_0..p_{d-1}>=':>23} {'reach_d<=':>10} {'p_d <=':>8}")
print("-" * 98)
for name, eff, R, w in CASES:
    if eff < 1.0:
        continue
    d = int(round(eff))
    e_d = (512.0 - R) / R
    if e_d > d:
        print(f"{name:10s} {d:3d} {e_d:7.4f}  IMPOSSIBLE: needs mean reach > 1")
        continue
    avg = e_d / d
    r_last = max(0.0, d * avg - (d - 1) * 1.0)
    geo = r_last ** (1.0 / d) if r_last > 0 else 0.0
    reach_d = HEAD_STEP * (1.0 + e_d) / (1.0 + HEAD_STEP * d)
    p_d = reach_d / r_last if r_last > 0 else float("inf")
    print(f"{name:10s} {d:3d} {e_d:7.4f} {avg:10.4f} {r_last:13.4f} "
          f"{geo:23.4f} {reach_d:10.4f} {p_d:8.4f}")

print()
print("=" * 98)
print("PRE-REGISTERED FALSIFIER FOR E128 RUNG 1 (forced depth 7, uncensored p_0..p_7)")
print("=" * 98)
for name, eff, R, w in CASES:
    if w <= 0.0 or eff < 1.0:
        continue
    d = int(round(eff))
    e_d = (512.0 - R) / R
    if e_d > d:
        continue
    avg = e_d / d
    r_last = max(0.0, d * avg - (d - 1) * 1.0)
    geo = r_last ** (1.0 / d) if r_last > 0 else 0.0
    reach_d = HEAD_STEP * (1.0 + e_d) / (1.0 + HEAD_STEP * d)
    p_d = reach_d / r_last if r_last > 0 else float("inf")
    print(f"  {name:9s} weight {w:.4f}:  the assumed R={R} survives ONLY IF the")
    print(f"             measured profile has geo-mean(p_0..p_{d-1}) >= {geo:.4f}")
    print(f"             AND a drop to p_{d} <= {min(p_d,1.0):.4f} at position {d}.")
    print(f"             A flat or slowly-declining profile FALSIFIES R={R}.")
