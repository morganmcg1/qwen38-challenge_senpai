"""A hard feasibility bound on the unpublished ranked round count R.

The shipped costModelDepth with the uniform 0.18 price is a STEP function of a
flat per-position acceptance p, with thresholds t_1..t_7.  Therefore

    E[depth] = sum_k P(p > t_k)

Given the published effective_mean_draft_len = E[depth], the largest possible
E[p] is a small linear program: put mass only at the threshold points, because
mass just below a threshold buys the most mean p per unit of depth.  Solve the
upper concave envelope of the points (k, t_{k+1}).

Separately the accounting identity forces
    accept_rate(R) = (512 - R) / (R * eff),  p_acct(R) from E[acc] = rate * eff.
p_acct falls monotonically in R.  Any R whose p_acct exceeds the LP bound is
arithmetically impossible under a flat acceptance profile.

Both readings assume a flat profile.  A declining profile relaxes the bound,
which is exactly what E128 rung 1 measures.  Report the bound as conditional.
"""

CAP = 7
HEAD_STEP = 0.18


def sched_depth(p):
    reach, expected, depth = 1.0, 0.0, 0
    while depth < CAP:
        reach *= p
        if not (reach > HEAD_STEP * (1.0 + expected) / (1.0 + HEAD_STEP * depth)):
            break
        expected += reach
        depth += 1
    return depth


def thresholds():
    ts, prev = [], sched_depth(0.0)
    for k in range(1, 1000001):
        p = k / 1000000.0
        d = sched_depth(p)
        if d != prev:
            ts.append((d, p))
            prev = d
    return ts


TS = thresholds()
# Support points for the LP: depth value s, and the largest p attaining it.
PTS = [(0, TS[0][1] - 1e-6)]
for i, (d, p) in enumerate(TS):
    hi = TS[i + 1][1] - 1e-6 if i + 1 < len(TS) else 1.0
    PTS.append((d, hi))


def max_mean_p(eff):
    """Upper concave envelope of PTS evaluated at E[s] = eff."""
    best = -1.0
    for i in range(len(PTS)):
        for j in range(len(PTS)):
            s0, v0 = PTS[i]
            s1, v1 = PTS[j]
            if s0 <= eff <= s1 and s1 > s0:
                lam = (eff - s0) / (s1 - s0)
                best = max(best, (1 - lam) * v0 + lam * v1)
            if s0 == eff:
                best = max(best, v0)
    return best


def eacc(p, d):
    if d <= 0:
        return 0.0
    if abs(1.0 - p) < 1e-12:
        return float(d)
    return p * (1.0 - p ** d) / (1.0 - p)


def solve_p_acct(eff, rate):
    target = rate * eff
    lo, hi = 1e-9, 1.0 - 1e-9
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if eacc(mid, eff) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def round_us(M):
    return 27215.4 + 3966.4 * M if M <= 4 else 17020.7 + 7154.2 * M


CASES = [("beagle", 4.3818, 110, 0.4862), ("medicine", 5.2556, 90, 0.2508),
         ("essays", 5.0870, 92, 0.1598), ("botany", 6.1481, 81, 0.0124),
         ("republic", 4.9892, 93, 0.0100), ("drama", 2.2976, 252, 0.0),
         ("travel", 2.6557, 212, 0.0)]

print("=" * 100)
print("SHIPPED POLICY STEP THRESHOLDS (flat p -> equilibrium depth)")
print("=" * 100)
print("  " + "   ".join(f"d>={d} at p>{p:.4f}" for d, p in TS))
print()
print("=" * 100)
print("HARD FEASIBILITY BOUND ON R, ASSUMING A FLAT PER-POSITION ACCEPTANCE")
print("=" * 100)
print(f"{'prompt':10s} {'eff':>7} {'assumed R':>10} {'p_acct':>8} "
      f"{'max E[p]':>9} {'verdict':>12} {'R_min':>6} {'rate':>7} {'opt d':>6} {'gain%':>7}")
print("-" * 100)
rows = []
for name, eff, R_assumed, w in CASES:
    cap_p = max_mean_p(eff)
    pa = solve_p_acct(eff, (512.0 - R_assumed) / (R_assumed * eff))
    R_min = None
    for R in range(int(512.0 / (1.0 + eff)) + 1, 513):
        rate = (512.0 - R) / (R * eff)
        if rate <= 0:
            break
        if solve_p_acct(eff, rate) <= cap_p:
            R_min = R
            break
    verdict = "INFEASIBLE" if pa > cap_p else "feasible"
    rate_m = (512.0 - R_min) / (R_min * eff)
    p_m = solve_p_acct(eff, rate_m)
    base = round_us(eff + 1.0) / (1.0 + eacc(p_m, eff))
    bd, bc = eff, base
    for k in range(0, CAP + 1):
        c = round_us(k + 1.0) / (1.0 + eacc(p_m, float(k)))
        if c < bc:
            bd, bc = float(k), c
    gain = 100.0 * (base - bc) / base
    rows.append((name, w, eff, R_assumed, R_min, p_m, bd, gain))
    print(f"{name:10s} {eff:7.4f} {R_assumed:10d} {pa:8.4f} {cap_p:9.4f} "
          f"{verdict:>12} {R_min:6d} {rate_m:7.4f} {bd:6.1f} {gain:7.3f}")

print()
print("R_min is the SMALLEST round count compatible with the shipped policy.")
print("The true R is at least R_min, so the assumed vector is a lower bound that")
print("the policy itself refutes on every median-carrying prompt.")
print()
print("=" * 100)
print("WHAT THIS DOES TO THE FITTED RANKED ROUND-COST CURVE")
print("=" * 100)
SPT = {"beagle": 0.012375, "botany": 0.011301, "drama": 0.020127,
       "essays": 0.011339, "medicine": 0.011564, "republic": 0.011324,
       "travel": 0.017810}
print(f"{'prompt':10s} {'Mbar':>6} {'round us @assumed R':>20} {'round us @R_min':>17} {'change':>9}")
print("-" * 100)
for name, w, eff, Ra, Rm, p_m, bd, gain in rows:
    tot = 512.0 * SPT[name] * 1e6
    a, b = tot / Ra, tot / Rm
    print(f"{name:10s} {eff+1:6.3f} {a:20,.1f} {b:17,.1f} {100*(b-a)/a:8.1f} %")
print()
print("The deep prompts get CHEAPER per round and the shallow ones barely move,")
print("so the fitted cost-versus-width curve FLATTENS.  A flatter curve makes")
print("deeper drafting look better, which pushes back against the shallower")
print("optimum above.  The two effects fight; only measured p_0..p_7 settles it.")
