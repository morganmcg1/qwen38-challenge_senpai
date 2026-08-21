"""Check the depth-4 dominance margins, and correct the required cliff
flattening reported in advisor feedback `e92-f7`.

`e92-f7` gives 1.343x as the flattening that would overturn the theorem. That
number divides the RESCALED `m4` by its critical value without re-rescaling
the curve, so it is not a statement about the raw width-5 step. Shrinking the
raw cliff also raises the `1.44 / sum(raw)` scale factor, which partly
compensates. This script reports all three normalisations so the record is
unambiguous.

Run: python3 research/e94-artifacts/depth4_flattening_check.py
"""

# E92 rung 2 production sweep, median round GPU-busy microseconds by verify
# width. Job 2694a061, 512 decode tokens, pin_purity 1.000 on every leg.
C_RAW = {1: 64445.4, 2: 69775.5, 3: 74778.4, 4: 86237.4, 5: 126103.1,
         6: 137842.6, 7: 150431.4, 8: 163957.1, 9: 204028.5}

# E92 `production_pure`: marginal round cost of the step into verify width
# `index + 2`, normalised by the width-1 round cost, no additive h.
RAW_PRICE = [0.08270645, 0.07762945, 0.17781030, 0.61859579,
             0.18216268, 0.19533968, 0.20987759, 0.62178894]

SHIPPED_TOTAL = 8 * 0.18
YIELD_CEILING = 0.25  # sup of a4 / (1 + 3*a4) over a4 in [0, 1]
M5_ASSUMED_FLATTENING = 1.126


def rescaled(raw):
    scale = SHIPPED_TOTAL / sum(raw)
    return [v * scale for v in raw]


def m4_over_c3(raw):
    m = rescaled(raw)
    return m[3] / (1.0 + sum(m[:3]))


def flattening_for_rescaled_model():
    """Raw cliff shrink factor that drives m4/C3 down to 1/4, re-rescaling."""
    lo, hi = 1.0, 8.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        probe = list(RAW_PRICE)
        probe[3] = RAW_PRICE[3] / mid
        if m4_over_c3(probe) > YIELD_CEILING:
            lo = mid
        else:
            hi = mid
    return lo


def main():
    step = C_RAW[5] - C_RAW[4]
    ratio = C_RAW[5] / C_RAW[4]
    # Largest C(w5) that any acceptance profile could ever justify.
    crit_c5 = (1.0 + YIELD_CEILING) * C_RAW[4]
    crit_step = crit_c5 - C_RAW[4]

    print("== assumption-free form, raw round-busy only")
    print("   Y(4)/Y(3) <= %.4f for every acceptance profile"
          % (1.0 + YIELD_CEILING))
    print("   C(w5)/C(w4) = %.5f    margin %.2f %%"
          % (ratio, 100.0 * (ratio / (1.0 + YIELD_CEILING) - 1.0)))
    print("   step must fall %.1f -> %.1f us = %.2f %%  (flatten %.4fx)"
          % (step, crit_step, 100.0 * (1.0 - crit_step / step),
             step / crit_step))

    s5 = step / M5_ASSUMED_FLATTENING
    r5 = (C_RAW[4] + s5) / C_RAW[4]
    print("\n== M5 at a %.3fx flatter cliff" % M5_ASSUMED_FLATTENING)
    print("   step %.1f us   C(w5)/C(w4) = %.5f   margin %.2f %%"
          % (s5, r5, 100.0 * (r5 / (1.0 + YIELD_CEILING) - 1.0)))
    print("   step must still fall %.2f %%, a further %.4fx"
          % (100.0 * (1.0 - crit_step / s5), s5 / crit_step))

    m = rescaled(RAW_PRICE)
    c3, m4 = 1.0 + sum(m[:3]), m[3]
    print("\n== rescaled price array")
    print("   m4/C3 = %.6f against the %.2f ceiling   margin %.1f %%"
          % (m4 / c3, YIELD_CEILING,
             100.0 * (m4 / c3 - YIELD_CEILING) / (m4 / c3)))
    print("   the C3 - 3*m4 coefficient = %+.6f, which is %.3f %% of C3."
          % (c3 - 3.0 * m4, 100.0 * abs(c3 - 3.0 * m4) / c3))
    print("   That coefficient is NOT the decision margin.")

    print("\n== required flattening of the RAW width-5 step, three models")
    print("   raw round-busy, assumption-free      %.4fx  <- the sound one"
          % (step / crit_step))
    print("   rescaled price, re-rescaled to 1.44  %.4fx"
          % flattening_for_rescaled_model())
    print("   rescaled m4 alone, no re-rescale     %.4fx  <- e92-f7"
          % (m4 / (c3 * YIELD_CEILING)))
    print("   all three exceed the %.3fx M5 assumption, so snap4 transfers"
          % M5_ASSUMED_FLATTENING)


if __name__ == "__main__":
    main()
