#!/usr/bin/env python3
"""Recompute the two-regime roofline derivation behind PR #10 and the FACT 1 banner.

This is a GENERATOR, not a measurement. It consumes numbers already recorded in
merged artifacts and does arithmetic on them. It exists so that no claim in
`ESTABLISHED_FACTS.md` FACT 1 or `CURRENT_RESEARCH_STATE.md` prediction 8 has to
be taken on trust -- run this and compare.

Inputs (both from merged PRs, cited inline):
  * PR #8 NA=4 control, stream-corrected GB/s at M = 4,5,8,9
    -> research/ESTABLISHED_FACTS.md:1192-1195
  * PR #5 `implied_streams` curve for lm_head at M = 1..9
    -> research/ESTABLISHED_FACTS.md:171

Definitions being inverted (research/qmv_cost_curve_summary.py):
    gbps_nominal          = weight_bytes / seconds_per_call / 1e9        (:274-278)
    gbps_stream_corrected = weight_streams(m) * gbps_nominal             (:274-278)
    weight_streams(m)     = ceil(m / 4)                                  (:132-136)

Usage:  python3 research/roofline_regime_check.py
"""

import math
import statistics as st

# --- Input 1: PR #8 NA=4 control (ESTABLISHED_FACTS.md:1192-1195) -------------
# (M, gbps_stream_corrected)
PR8_NA4_CONTROL = [(4, 165.6), (5, 262.1), (8, 183.0), (9, 239.5)]

# --- Input 2: PR #5 implied_streams, lm_head (ESTABLISHED_FACTS.md:171) -------
PR5_IMPLIED_STREAMS = {
    1: 1.00, 2: 0.99, 3: 1.01, 4: 1.24, 5: 1.64,
    6: 1.90, 7: 2.17, 8: 2.44, 9: 2.87,
}
PLATEAU_MAX_M = 3  # PR #5: "the measured plateau ends at M = 1-3 on every shape"


def weight_streams(m: int) -> int:
    """qmv_cost_curve_summary.py:132-136."""
    return math.ceil(m / 4)


def invert_stream_correction():
    """Recover gbps_nominal and test which quantity is actually invariant."""
    rows = []
    for m, corrected in PR8_NA4_CONTROL:
        s = weight_streams(m)
        nominal = corrected / s
        rows.append((m, s, corrected, nominal, nominal * m))

    print("Inverting the ceil(M/4) stream correction over PR #8's NA=4 control")
    print()
    print("   M  streams  stream-corr    nominal   nominal*M")
    for m, s, corrected, nominal, prod in rows:
        print(f"  {m:2d}  {s:7d}  {corrected:11.1f}  {nominal:9.2f}  {prod:10.1f}")
    print()

    noms = [r[3] for r in rows]
    prods = [r[4] for r in rows]

    nom_span = max(noms) / min(noms)
    nom_rel = 100 * st.stdev(noms) / st.mean(noms)
    prod_mean, prod_sd = st.mean(prods), st.stdev(prods)
    prod_rel = 100 * prod_sd / prod_mean

    # A bandwidth-bound kernel holds gbps_nominal flat.
    # An ALU-bound kernel (seconds_per_call ~ M) holds gbps_nominal * M flat.
    print(f"  gbps_nominal      : span {nom_span:.3f}x  "
          f"({min(noms):.1f} -> {max(noms):.1f}), rel. sd {nom_rel:.1f}%")
    print(f"  gbps_nominal * M  : {prod_mean:.1f} +/- {prod_rel:.1f}%  "
          f"(stdev {prod_sd:.1f}, n={len(prods)})")
    print()
    verdict = "ALU-bound" if prod_rel < nom_rel else "bandwidth-bound"
    print(f"  => the invariant is the one with the smaller spread: {verdict}")
    print(f"     (nominal*M is {nom_rel / prod_rel:.1f}x tighter than nominal)")
    return prod_mean, prod_rel


def _ols(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    slope = (sum((x - mx) * (y - my) for x, y in zip(xs, ys))
             / sum((x - mx) ** 2 for x in xs))
    return slope, my - slope * mx


def knee_locations():
    """Where does the ramp meet the flat floor? Test method-robustness."""
    imp = PR5_IMPLIED_STREAMS
    floor = st.mean([imp[m] for m in range(1, PLATEAU_MAX_M + 1)])

    print()
    print(f"PR #5 ramp vs the flat floor c = {floor:.4f} (mean of M=1..{PLATEAU_MAX_M})")
    print()
    print("  method                                  slope  intercept    knee")

    results = []

    # Endpoint slope, proportional line (intercept forced to zero).
    # This is what the PR #10 brief computed: (2.87 - 1.24) / (9 - 4).
    ep = (imp[9] - imp[4]) / (9 - 4)
    k = floor / ep
    results.append(k)
    print(f"  endpoint M=4->9, proportional c=B*M    {ep:6.3f}   (forced 0)  {k:6.3f}")

    # Endpoint slope carrying its own intercept.
    ic = imp[4] - ep * 4
    k = (floor - ic) / ep
    results.append(k)
    print(f"  endpoint M=4->9, own intercept         {ep:6.3f}     {ic:+7.3f}  {k:6.3f}")

    # OLS over several plausible windows.
    for lo in (3, 4, 5, 6):
        xs = list(range(lo, 10))
        ys = [imp[m] for m in xs]
        slope, ic = _ols(xs, ys)
        k = (floor - ic) / slope
        results.append(k)
        print(f"  OLS M={lo}..9                            "
              f"{slope:6.3f}     {ic:+7.3f}  {k:6.3f}")

    print()
    print(f"  => knee in [{min(results):.2f}, {max(results):.2f}] "
          f"across all methods and windows")
    print(f"     The BAND is the load-bearing claim. Any single figure "
          f"(e.g. 3.07) is method-dependent.")

    # The integer-stream model would demand steps; the data is continuous.
    print()
    print("  Integer-stream model vs measurement (why the correction is suspect):")
    print("     M            " + "".join(f"{m:6d}" for m in sorted(imp)))
    print("     measured     " + "".join(f"{imp[m]:6.2f}" for m in sorted(imp)))
    print("     ceil(M/4)    " + "".join(f"{weight_streams(m):6d}" for m in sorted(imp)))
    print("     => measured is continuous where the integer model demands steps.")
    return min(results), max(results)


def main():
    print("=" * 74)
    print("Two-regime roofline check: t(M) = max(t_bandwidth, beta*M)")
    print("=" * 74)
    print()
    invert_stream_correction()
    lo, hi = knee_locations()
    print()
    print("=" * 74)
    print("Conclusion (derived from merged artifacts, NOT separately measured):")
    print(f"  Both datasets agree on a knee near M = 3 (band [{lo:.2f}, {hi:.2f}]).")
    print("  Above it, seconds_per_call ~ M, i.e. cross-row weight reuse buys ~0.")
    print("  If confirmed by PR #10, the '% of peak' headroom for M=4 and M=8 is")
    print("  an artifact of the ceil(M/4) correction and the memory-side lever")
    print("  family (stream count, tiling, cache blocking) is dead.")
    print("  PR #10 settles it by experiment; this script only states the case.")
    print("=" * 74)


if __name__ == "__main__":
    main()
