"""Pin the unpublished ranked round count R from the shipped scheduler itself.

Two independent readings of the same per-step acceptance p must agree.

  READING 1 (accounting).  Every round emits 1 committed token plus its
  accepted drafts, so over a 512-token window R = 512 - A and
      accept_rate = (512 - R) / (R * eff)
  Inverting E[accepted] = p(1-p^d)/(1-p) at d = eff gives p_acct(R).

  READING 2 (policy).  costModelDepth is deterministic.  With the shipped
  uniform 0.18 price, a flat per-position acceptance p makes the scheduler
  settle at an equilibrium depth d*(p).  Requiring d*(p) == eff gives
  p_policy, which does not depend on R at all.

R is whatever makes the two agree.  Anything else says the flat-p model is
wrong, which is itself the finding.
"""

CAP = 7
HEAD_STEP = 0.18


def sched_depth(pvec):
    """Exact transcription of costModelDepth with the shipped uniform price.

    Margin gates are omitted: top1 - top2 >= 0 always, so sigmoid(m/2) >= 0.5
    and sigmoid(m/3) >= 0.5, and the depth-0 and depth-1 thresholds are 0.18
    and 0.18*(1+reach0)/1.18 < 0.31.  Neither gate can ever fire.  That is
    also why non_drafting_round_count is 0 on all seven drafting prompts.
    """
    reach, expected, depth = 1.0, 0.0, 0
    while depth < CAP:
        reach *= pvec[depth]
        marginal = HEAD_STEP
        cumulative = 1.0 + HEAD_STEP * depth
        if not (reach > marginal * (1.0 + expected) / cumulative):
            break
        expected += reach
        depth += 1
    return depth


def sched_depth_frac(p):
    """Equilibrium depth as a continuous function of flat p.

    Returns the fractional depth at which reach exactly equals threshold, so
    the integer policy is bracketed and eff can be matched off-grid.
    """
    reach, expected = 1.0, 0.0
    for depth in range(CAP):
        reach *= p
        thr = HEAD_STEP * (1.0 + expected) / (1.0 + HEAD_STEP * depth)
        if reach <= thr:
            return depth + (reach / thr if thr > 0 else 0.0) - 1.0 + 1.0 \
                if False else float(depth)
        expected += reach
    return float(CAP)


def eacc(p, d):
    if d <= 0:
        return 0.0
    if abs(1.0 - p) < 1e-12:
        return float(d)
    return p * (1.0 - p ** d) / (1.0 - p)


def solve_p_acct(eff, rate):
    """p such that E[accepted | depth eff] = rate * eff."""
    target = rate * eff
    lo, hi = 1e-6, 0.999999
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if eacc(mid, eff) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def round_us(M):
    return 27215.4 + 3966.4 * M if M <= 4 else 17020.7 + 7154.2 * M


CASES = [("beagle", 4.3818, 110, 0.4862), ("medicine", 5.2556, 90, 0.2508),
         ("essays", 5.0870, 92, 0.1598), ("republic", 4.9892, 93, 0.0100),
         ("botany", 6.1481, 81, 0.0124), ("drama", 2.2976, 252, 0.0),
         ("travel", 2.6557, 212, 0.0)]

print("=" * 96)
print("STEP 1  policy reading: flat p -> equilibrium depth, shipped uniform 0.18 price")
print("=" * 96)
print(f"{'p':>8} {'d*':>4}   {'p':>8} {'d*':>4}   {'p':>8} {'d*':>4}")
grid = [0.50, 0.60, 0.70, 0.75, 0.80, 0.82, 0.84, 0.85, 0.86, 0.88,
        0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
cells = [(p, sched_depth([p] * CAP)) for p in grid]
for i in range(0, len(cells), 3):
    row = cells[i:i + 3]
    print("   ".join(f"{p:8.3f} {d:4d}" for p, d in row))

print()
print("The policy is a STEP function of p.  Read the thresholds:")
prev = None
for k in range(0, 2001):
    p = k / 2000.0
    d = sched_depth([p] * CAP)
    if prev is not None and d != prev:
        print(f"   d* moves {prev} -> {d} at p = {p:.4f}")
    prev = d

print()
print("=" * 96)
print("STEP 2  the two readings, swept over R")
print("=" * 96)
print("p_acct falls as R rises.  p_policy is the flat p whose equilibrium depth")
print("brackets the observed eff, and does not move with R.  Where they cross is R.")
print()
for name, eff, R_assumed, w in CASES:
    lo_d, hi_d = int(eff), int(eff) + 1
    band = []
    for k in range(1, 2000):
        p = k / 2000.0
        if sched_depth([p] * CAP) == lo_d:
            band.append(p)
    p_lo = min(band) if band else float("nan")
    p_hi = max(band) if band else float("nan")
    Rmin = int(512.0 / (1.0 + eff)) + 1
    hit = None
    for R in range(Rmin, 513):
        rate = (512.0 - R) / (R * eff)
        if not (0.0 < rate <= 1.0):
            continue
        pa = solve_p_acct(eff, rate)
        if p_lo <= pa <= p_hi and hit is None:
            hit = (R, pa, rate)
    print(f"--- {name:9s} eff {eff:6.4f}  weight {w:.4f}  assumed R {R_assumed}")
    print(f"      policy band for d*={lo_d}: p in [{p_lo:.4f}, {p_hi:.4f}]")
    pa_assumed = solve_p_acct(eff, (512.0 - R_assumed) / (R_assumed * eff))
    print(f"      p_acct at the assumed R           = {pa_assumed:.4f}"
          f"   -> policy would pick d* = {sched_depth([pa_assumed]*CAP)}")
    if hit:
        R, pa, rate = hit
        print(f"      CONSISTENT at R = {R:3d}  (p {pa:.4f}, accept rate {rate:.4f})")
        base = round_us(eff + 1.0) / (1.0 + eacc(pa, eff))
        bd, bc = eff, base
        for kk in range(0, CAP + 1):
            c = round_us(kk + 1.0) / (1.0 + eacc(pa, float(kk)))
            if c < bc:
                bd, bc = float(kk), c
        print(f"      at that R the depth optimum is {bd:.1f} "
              f"({'DEEPER' if bd > eff else 'SHALLOWER' if bd < eff else 'same'}),"
              f" gain {100.0*(base-bc)/base:.3f} %")
    else:
        print("      NO R makes the two readings agree -> the flat-p model is refuted")
    print()
