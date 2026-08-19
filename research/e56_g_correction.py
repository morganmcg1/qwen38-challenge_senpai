"""Per-depth transfer correction for the E1 depth ladder, M4 Pro -> ranked M5.

The identified transfer is two-parameter: the ranked depth-0 round costs c1, and
the ladder's marginal increments carry a slope factor g relative to that base.

    round_M5(d) = c1 * (1 + g * cum_M4(d) / C0_M4)

A uniform scaling of the marginal ladder cancels in a marginal/cumulative ratio,
which is why the mechanism was first thought immune. It does not cancel here,
because the cumulative denominator contains the unscaled depth-0 round.
"""

C0_M4 = 65.0094  # ms, depth-0 round on M4 Pro (E1, N=1530, sd 0.16 %)
C1_M5 = 30.402  # ms, prefill-corrected ranked depth-0 round (ledger 184)

# E1 cumulative drafting cost above the depth-0 round, ms.
CUM_M4 = {0: 0.0, 1: 5.47, 2: 10.51, 3: 26.28, 4: 50.68,
          5: 69.66, 6: 89.16, 7: 107.82, 8: 133.23}
MARG_M4 = {d: CUM_M4[d] - CUM_M4[d - 1] for d in range(1, 9)}

G_BAND = (0.7388, 0.7778)  # ledger 184, prefill-corrected
G_BAND_STALE = (0.8331, 0.8792)  # prefill-blind, superseded


def h_ratio(d, g):
    """costModelDepth's marginal/cumulative ratio at depth d under slope g."""
    marg = g * MARG_M4[d] / C0_M4
    cum = 1.0 + g * CUM_M4[d] / C0_M4
    return marg / cum


def report():
    print(f"C0_M4 = {C0_M4:.4f} ms   c1_M5 = {C1_M5:.4f} ms   "
          f"host scale = {C1_M5 / C0_M4:.4f}")
    print(f"corrected g band = {G_BAND}   (superseded: {G_BAND_STALE})\n")

    hdr = (f"{'d':>2} {'marg_M4':>8} {'h_M4':>8} {'h_M5 lo':>8} {'h_M5 hi':>8} "
           f"{'r lo':>7} {'r hi':>7}")
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for d in range(1, 9):
        h4 = h_ratio(d, 1.0)
        h5lo = h_ratio(d, G_BAND[0])
        h5hi = h_ratio(d, G_BAND[1])
        rows.append((d, h4, h5lo, h5hi))
        print(f"{d:>2} {MARG_M4[d]:>8.2f} {h4:>8.4f} {h5lo:>8.4f} {h5hi:>8.4f} "
              f"{h5lo / h4:>7.4f} {h5hi / h4:>7.4f}")

    # After mean-pinning, only the shape survives. Compare normalised tables.
    print("\nMean-pinned to 0.18 (what the scheduler actually consumes):")
    print(f"{'d':>2} {'shipped':>9} {'M5 lo':>9} {'M5 hi':>9} {'delta lo':>9} "
          f"{'delta hi':>9}")
    print("-" * 60)
    m4 = [r[1] for r in rows]
    lo = [r[2] for r in rows]
    hi = [r[3] for r in rows]
    s4 = 0.18 * len(m4) / sum(m4)
    slo = 0.18 * len(lo) / sum(lo)
    shi = 0.18 * len(hi) / sum(hi)
    for i, d in enumerate(range(1, 9)):
        a, b, c = m4[i] * s4, lo[i] * slo, hi[i] * shi
        print(f"{d:>2} {a:>9.4f} {b:>9.4f} {c:>9.4f} "
              f"{(b - a) / a * 100:>8.2f}% {(c - a) / a * 100:>8.2f}%")

    # Marginal cost of one more draft, relative to the round it sits in.
    print("\nMarginal cost of step 3->4 relative to own depth-0 round:")
    print(f"  M4 Pro      {MARG_M4[4] / C0_M4:.4f}")
    for g in G_BAND:
        print(f"  ranked M5   {g * MARG_M4[4] / C0_M4:.4f}   (g={g})")

    print("\nDirection of the policy error:")
    print("  threshold = h*(1+expected)/(1+depth*h); lower h -> lower threshold")
    print("  -> deeper drafting. g<1 lowers every h, so the ranked optimum h is")
    print("  BELOW the shipped 0.18, and the shipped table under-drafts at rank.")


if __name__ == "__main__":
    report()
