"""The depth-4 dominance theorem and its true margin, on M4 Pro and on M5.

Decision: a depth-4 round beats a depth-3 round iff  Y4/C4 > Y3/C3,
which rearranges to        r4 / Y3 > m4 / C3
with r4 = Y4 - Y3 = prod_{j<=4} q_j and Y3 = 1 + a1 + a2 + a3, a_i = prod_{j<=i} q_j.

Because a1 >= a2 >= a3 >= a4 = r4, we have Y3 >= 1 + 3*r4, hence

    r4 / Y3 <= r4 / (1 + 3*r4) <= 1/4,   attained only as q -> 1.

So depth 4 is dominated for EVERY acceptance profile whenever m4/C3 > 1/4.
Three different margins fall out of that, and they are not interchangeable.
"""

# E92 production_pure, rescaled by makeMeasuredDepthPrice to 8h = 1.44.
M4PRO = [0.054987, 0.051612, 0.118217, 0.411272,
         0.121110, 0.129871, 0.139537, 0.413395]

# Edward's raw production round-busy microseconds, verify width 1..9.
C_RAW = {1: 64445.4, 2: 69775.5, 3: 74778.4, 4: 86237.4, 5: 126103.1,
         6: 137842.6, 7: 150431.4, 8: 163957.1, 9: 204028.5}

M5_CLIFF_FLATTENING = 1.126


def C(m, d):
    return 1.0 + sum(m[:d])


def yflat(q, d):
    t, p = 1.0, 1.0
    for _ in range(d):
        p *= q
        t += p
    return t


def report(label, m):
    c3, m4 = C(m, 3), m[3]
    ratio = m4 / c3
    print("== %s" % label)
    print("   m4 = %.6f   C3 = %.6f   m4/C3 = %.6f" % (m4, c3, ratio))
    print("   achievable max of r4/Y3 over EVERY profile = 0.250000")
    if ratio > 0.25:
        print("   -> depth 4 is dominated UNCONDITIONALLY")
    else:
        print("   -> depth 4 can win at high acceptance")
    # margin in r4 units: required r4 at the coupled worst case q -> 1 (Y3 = 4)
    req_r4 = 4.0 * ratio
    print("   margin in r4 units : required r4 = %.4f against a hard ceiling"
          " of 1.0  ->  %.1f %%" % (req_r4, 100.0 * (req_r4 - 1.0)))
    # margin in m4 units: m4 would have to fall to C3/4
    m4_crit = c3 / 4.0
    print("   margin in m4 units : m4 must fall %.6f -> %.6f  ->  %.1f %%" % (
        m4, m4_crit, 100.0 * (m4 - m4_crit) / m4))
    # the near-degenerate coefficient that produced the 0.735 %% reading
    coef = c3 - 3.0 * m4
    print("   the C3-3*m4 coefficient = %+.6f (%.3f %% of C3)  <- NOT the margin"
          % (coef, 100.0 * abs(coef) / c3))
    print("   margin at realistic flat acceptance:")
    for q in (0.90, 0.93, 0.955, 0.97, 0.98):
        y3, r4 = yflat(q, 3), q ** 4
        need = r4 * c3 / y3
        print("      q=%.3f  Y3=%.4f  r4=%.4f  m4 must fall to %.6f -> %.1f %%"
              % (q, y3, r4, need, 100.0 * (m4 - need) / m4))
    print()


def main():
    report("M4 Pro, measured directly by E92", M4PRO)

    m5 = list(M4PRO)
    m5[3] = M4PRO[3] / M5_CLIFF_FLATTENING
    report("M5, cliff assumed %.3fx flatter" % M5_CLIFF_FLATTENING, m5)

    print("== how flat would the M5 cliff have to be to overturn the theorem?")
    c3 = C(M4PRO, 3)
    m4_crit = c3 / 4.0
    total = M4PRO[3] / m4_crit
    print("   m4 must reach C3/4 = %.6f, so the cliff must flatten by %.3fx"
          % (m4_crit, total))
    print("   already assumed %.3fx, so a further %.3fx is required"
          % (M5_CLIFF_FLATTENING, total / M5_CLIFF_FLATTENING))
    print()

    # The raw form uses only two measured numbers and one exact bound.
    # Y4/Y3 = 1 + a4/(1+a1+a2+a3) <= 1 + a4/(1+3*a4) <= 1.25, for ANY profile.
    print("== raw round-busy check: no rescaling, no fitted price array")
    ceiling = 1.25
    marg4 = C_RAW[5] - C_RAW[4]
    print("   hard ceiling on Y(4)/Y(3) for any acceptance profile = %.4f"
          % ceiling)
    for name, c5 in (("M4 Pro measured", C_RAW[5]),
                     ("M5 at %.3fx flatter cliff" % M5_CLIFF_FLATTENING,
                      C_RAW[4] + marg4 / M5_CLIFF_FLATTENING)):
        ratio = c5 / C_RAW[4]
        crit_c5 = ceiling * C_RAW[4]
        crit_marg = crit_c5 - C_RAW[4]
        this_marg = c5 - C_RAW[4]
        print("   %-32s C(w5)/C(w4) = %.4f   margin %.1f %%" % (
            name, ratio, 100.0 * (ratio / ceiling - 1.0)))
        print("       marginal step into width 5 = %8.1f us, must fall to"
              " %8.1f us  ->  %.1f %%" % (
                  this_marg, crit_marg,
                  100.0 * (this_marg - crit_marg) / this_marg))


if __name__ == "__main__":
    main()
