#!/usr/bin/env python3
"""The ranked pricer is CONCAVE, so scalar shrinkage must be applied to the leg
gain BEFORE pricing, never to the score afterwards.

`research/e54_gap_decomposition.py:262-265` does the latter:

    surviving = escape["net_cell_win_pct"] / CELL_WIN_PCT[5]
    naive = E54_PRICE_SINGLE_CELL[mixture][5] * surviving
    # "This keeps the nonlinear ranked pricing thorfinn measured and
    #  only rescales the mechanism size."

`E54_PRICE_SINGLE_CELL[mixture][5]` is a SCORE that has already passed through
`qmv_score_leverage.score_pct_from_leg_gains`, which prices by re-sorting the
eight ranked order statistics.  Multiplying that score by `surviving` does not
"keep the nonlinear pricing"; it linearises it, which is exactly what ledger
187(H) and rule 189(J)/L11009 forbid.

`f` is concave and `f(0) = 0`, so `f(a*x) >= a*f(x)` for `a` in `[0, 1]`.  The
published order therefore UNDER-prices every direction whose full-size gain sits
above the kink and whose shrunken gain falls below it.  The `r=2` row-block
route is exactly that case.

This is a SECOND, independent under-pricing on top of the prefill
double-dilution that ledger 189(D) found.  They compound.

Run `--self-test` for the checks, no argument for the report.  Exit 0 means
every check passed.

Sources
-------
research/qmv_score_leverage.py:141-198  score_from_leg_gains, by re-sorting
research/qmv_score_leverage.py:47       PSI_MTP = 0.6736  (NOT 0.693391)
research/qmv_score_leverage.py:218-247  substitution_headroom, kink_pct
research/e54_gap_decomposition.py:262-265  the mis-ordered rescale
ledger 187(L) :10798-10801  published r=2 route prices
ledger 188(E) :11122-11127  same route on the calibrated g/h union
ledger 189(E) :11364-11369  the prefill-corrected column
"""

import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import qmv_score_leverage as L                                  # noqa: E402

# ------------------------------------------------------------------ mechanism
CELL_WIN_PCT_M5 = -20.253      # E54 measured, <T,5,5> alone, negative = faster
R2_TAX_PCT = 10.54             # E44 measured x re-read tax at NA=4
SHARE_M5 = {"e48": 12.1744, "e53_mid": 22.3363}    # % of scored QMV cost

# ------------------------------------------------------------------ transfers
TRANSFER_UNION = (0.7388, 0.8617)      # 188(D): report the g/h union
DILUTION_MEDIAN_PAIR = 0.9125          # 186(B), round basis -> leg basis
REBASE = (1.17776, 1.19108)            # 189(D), local leg share -> ranked

# --------------------------------------------------------------------- bars
DEFICIT_PCT = 0.5367
MDE_PCT = 0.283

SURVIVING = (CELL_WIN_PCT_M5 + R2_TAX_PCT) / CELL_WIN_PCT_M5     # 0.47958


def both(x):
    return {"beagle": x, "medicine": x}


def f(leg_pct):
    """The ranked pricer: uniform scored-pair leg gain -> % score change."""
    return L.score_pct_from_leg_gains(both(leg_pct))


def full_leg_gain(mixture, psi=L.PSI_MTP):
    """Leg gain of the UNTAXED <T,5,5> cell, before any shrinkage."""
    removed = SHARE_M5[mixture] / 100.0 * abs(CELL_WIN_PCT_M5)
    return psi * removed


def price_then_shrink(mixture, transfer, rebase=1.0):
    """The published order: price the full cell, then multiply the SCORE."""
    return f(full_leg_gain(mixture)) * SURVIVING * rebase * transfer


def shrink_then_price(mixture, transfer, rebase=1.0):
    """The correct order: shrink the LEG GAIN, then price once."""
    return f(full_leg_gain(mixture) * SURVIVING * rebase * transfer)


def slope_below():
    return f(0.5) / 0.5


def slope_above():
    k = L.kink_pct()
    return (f(k + 1.0) - f(k)) / 1.0


# ------------------------------------------------------------------ self-tests

def self_test():
    checks = []

    def ck(name, cond, detail=""):
        checks.append((name, bool(cond), detail))

    k = L.kink_pct()
    ck("kink reproduces the documented +1.0551 %", abs(k - 1.0551) < 1e-3,
       f"kink={k:.4f} %")
    ck("slope BELOW the kink is exactly 1.0 (a uniform pair gain is the score)",
       abs(slope_below() - 1.0) < 1e-9, f"{slope_below():.6f}")
    ck("slope ABOVE the kink is about 0.4837 (only rank 4 still pays)",
       abs(slope_above() - 0.4837) < 1e-3, f"{slope_above():.6f}")
    ck("the pricer uses psi = 0.6736, NOT the 0.693391 my files label it",
       abs(L.PSI_MTP - 0.6736) < 1e-9, f"PSI_MTP={L.PSI_MTP}")

    # The leg reconstruction must reproduce E54's published single-cell prices,
    # otherwise nothing below is about the same quantity.
    ck("reconstruction reproduces E54's published e48 M=5 price +1.3481",
       abs(f(full_leg_gain("e48")) - 1.3481) < 2e-4,
       f"{f(full_leg_gain('e48')):.4f}")
    ck("reconstruction reproduces E54's published e53_mid M=5 price +2.0187",
       abs(f(full_leg_gain("e53_mid")) - 2.0187) < 2e-4,
       f"{f(full_leg_gain('e53_mid')):.4f}")
    ck("r=2 surviving fraction reproduces the published 0.47958",
       abs(SURVIVING - 0.47958) < 1e-5, f"{SURVIVING:.5f}")

    # Reproduce the two published columns before correcting them.  188(E)
    # applied the round-basis dilution; 189(D) replaced it with the rebase.
    lo = price_then_shrink("e48", TRANSFER_UNION[0], DILUTION_MEDIAN_PAIR)
    hi = price_then_shrink("e48", TRANSFER_UNION[1], DILUTION_MEDIAN_PAIR)
    ck("reproduces 188(E)'s published e48 r=2 band 0.4359..0.5084",
       abs(lo - 0.4359) < 2e-3 and abs(hi - 0.5084) < 2e-3,
       f"{lo:.4f}..{hi:.4f}")
    lo9 = price_then_shrink("e48", TRANSFER_UNION[0], REBASE[0])
    hi9 = price_then_shrink("e48", TRANSFER_UNION[1], REBASE[1])
    ck("reproduces 189(E)'s prefill-corrected e48 r=2 band 0.5626..0.6636",
       abs(lo9 - 0.5626) < 2e-3 and abs(hi9 - 0.6636) < 2e-3,
       f"{lo9:.4f}..{hi9:.4f}")

    # POSITIVE CONTROL: concavity must bite, and in the stated direction.
    x = full_leg_gain("e48")
    ck("POSITIVE CONTROL: the untaxed e48 gain sits ABOVE the kink",
       x > k, f"leg={x:.4f} % vs kink {k:.4f} %")
    ck("POSITIVE CONTROL: f(a*x) > a*f(x) strictly, for a=SURVIVING, x>kink",
       f(SURVIVING * x) > SURVIVING * f(x) * (1 + 1e-9),
       f"{f(SURVIVING*x):.4f} > {SURVIVING*f(x):.4f}")

    # NEGATIVE CONTROL 1: below the kink the two orders must agree EXACTLY.
    small = k / 4.0
    ck("NEGATIVE CONTROL: below the kink the two orders agree to 1e-12",
       abs(f(SURVIVING * small) - SURVIVING * f(small)) < 1e-12,
       f"delta={abs(f(SURVIVING*small) - SURVIVING*f(small)):.2e}")

    # NEGATIVE CONTROL 2: at a = 1 the two orders are the same expression.
    ck("NEGATIVE CONTROL: at shrinkage 1.0 the two orders agree to 1e-12",
       abs(f(1.0 * x) - 1.0 * f(x)) < 1e-12)

    # NEGATIVE CONTROL 3: a LINEAR pricer would show no ordering effect at all.
    lin = L.CROWN_ORDER_STATS
    flat = tuple((n, 3.0) for n, _ in lin)     # all eight ratios equal
    def f_flat(v):
        return L.score_pct_from_leg_gains(both(v), stats=flat)
    ck("NEGATIVE CONTROL: with eight equal ratios the pricer is linear and the "
       "ordering effect vanishes",
       abs(f_flat(SURVIVING * x) - SURVIVING * f_flat(x)) < 1e-9,
       f"delta={abs(f_flat(SURVIVING*x) - SURVIVING*f_flat(x)):.2e}")

    # The corrected numbers.
    c_lo = shrink_then_price("e48", TRANSFER_UNION[0], REBASE[0])
    c_hi = shrink_then_price("e48", TRANSFER_UNION[1], REBASE[1])
    ck("corrected e48 r=2 band is 0.6931..0.8176",
       abs(c_lo - 0.6931) < 2e-3 and abs(c_hi - 0.8176) < 2e-3,
       f"{c_lo:.4f}..{c_hi:.4f}")
    d_lo = shrink_then_price("e53_mid", TRANSFER_UNION[0], REBASE[0])
    d_hi = shrink_then_price("e53_mid", TRANSFER_UNION[1], REBASE[1])
    ck("corrected e53_mid r=2 band is 1.1598..1.2703",
       abs(d_lo - 1.1598) < 2e-3 and abs(d_hi - 1.2703) < 2e-3,
       f"{d_lo:.4f}..{d_hi:.4f}")

    ck("total e48 under-pricing versus 188(E) is 1.59x..1.61x",
       1.58 < c_lo / lo < 1.62 and 1.58 < c_hi / hi < 1.62,
       f"{c_lo/lo:.4f}x .. {c_hi/hi:.4f}x")

    # The decision the correction changes.
    ck("DECISION: the r=2 route clears the 0.5367 % deficit at the LOW end of "
       "both mixtures and both transfer ends",
       min(c_lo, d_lo) > DEFICIT_PCT,
       f"worst corner {min(c_lo, d_lo):.4f} % = "
       f"{min(c_lo, d_lo)/DEFICIT_PCT:.2f}x deficit")
    ck("DECISION: worst corner is above 2x the +0.283 % ranked MDE",
       min(c_lo, d_lo) / MDE_PCT > 2.0,
       f"{min(c_lo, d_lo)/MDE_PCT:.2f}x MDE")

    # The e53_mid gain stays above the kink even after shrinkage, so its
    # ordering gap is smaller than e48's. That asymmetry is a real prediction.
    ck("e53_mid stays above the kink after shrinkage; e48 falls below it",
       (full_leg_gain("e53_mid") * SURVIVING * REBASE[0] * TRANSFER_UNION[0]) > k
       and (full_leg_gain("e48") * SURVIVING * REBASE[1] * TRANSFER_UNION[1]) < k)

    width = max(len(n) for n, _, _ in checks)
    bad = 0
    for name, ok, detail in checks:
        bad += 0 if ok else 1
        print(f"[{'PASS' if ok else 'FAIL'}] {name.ljust(width)}  {detail}")
    print(f"\n{len(checks) - bad}/{len(checks)} checks passed")
    return 1 if bad else 0


def report():
    k = L.kink_pct()
    print("Pricing ORDER: shrink the leg gain, then price. Never the reverse.")
    print("=" * 78)
    print(f"\nThe ranked pricer f is piecewise linear and CONCAVE, f(0) = 0:\n")
    print(f"   kink at            {k:.4f} % uniform scored-pair leg gain")
    print(f"   slope below kink   {slope_below():.6f}   (both scored prompts pay)")
    print(f"   slope above kink   {slope_above():.6f}   (medicine ejects; only "
          "beagle pays)")
    print(f"\n   Concavity with f(0)=0 gives f(a*x) >= a*f(x) for a in [0,1].")
    print("   So multiplying a SCORE by a shrinkage factor UNDER-prices "
          "whenever the\n   full-size gain is above the kink and the shrunken "
          "gain is below it.")

    print(f"\nMechanism: <T,5,5> at rows_per_simd=2.")
    print(f"   cell win           {CELL_WIN_PCT_M5:+.3f} %")
    print(f"   r=2 x re-read tax  {R2_TAX_PCT:+.2f} %")
    print(f"   surviving fraction {SURVIVING:.5f}")

    print(f"\n{'mixture':<10} {'full leg':>10} {'f(full)':>10} "
          f"{'188(E) pub':>18} {'189(E) corr':>18} {'CORRECT ORDER':>18}")
    for m in ("e48", "e53_mid"):
        x = full_leg_gain(m)
        pub = (price_then_shrink(m, TRANSFER_UNION[0], DILUTION_MEDIAN_PAIR),
               price_then_shrink(m, TRANSFER_UNION[1], DILUTION_MEDIAN_PAIR))
        c89 = (price_then_shrink(m, TRANSFER_UNION[0], REBASE[0]),
               price_then_shrink(m, TRANSFER_UNION[1], REBASE[1]))
        cor = (shrink_then_price(m, TRANSFER_UNION[0], REBASE[0]),
               shrink_then_price(m, TRANSFER_UNION[1], REBASE[1]))
        print(f"{m:<10} {x:>10.4f} {f(x):>10.4f} "
              f"{f'{pub[0]:.4f}..{pub[1]:.4f}':>18} "
              f"{f'{c89[0]:.4f}..{c89[1]:.4f}':>18} "
              f"{f'{cor[0]:.4f}..{cor[1]:.4f}':>18}")

    c_lo = shrink_then_price("e48", TRANSFER_UNION[0], REBASE[0])
    d_lo = shrink_then_price("e53_mid", TRANSFER_UNION[0], REBASE[0])
    d_hi = shrink_then_price("e53_mid", TRANSFER_UNION[1], REBASE[1])
    worst = min(c_lo, d_lo)
    print(f"\nTwo compounding corrections, both mine:")
    print(f"   189(D) prefill double dilution   x1.2907 .. x1.3053")
    print(f"   this item, pricing order         x1.2320 (e48; e53_mid smaller)")
    print(f"   combined on the e48 r=2 route    "
          f"x{c_lo / price_then_shrink('e48', TRANSFER_UNION[0], DILUTION_MEDIAN_PAIR):.4f}")

    print(f"\nDECISION")
    print(f"   worst corner  {worst:.4f} %  =  "
          f"{worst/DEFICIT_PCT:.2f}x the {DEFICIT_PCT} % deficit, "
          f"{worst/MDE_PCT:.2f}x the {MDE_PCT} % MDE")
    print(f"   best corner   {d_hi:.4f} %  =  {d_hi/DEFICIT_PCT:.2f}x deficit")
    print("\n   The r=2 row-block route closes the deficit at the LOW end of "
          "both\n   mixtures and both ends of the transfer band. Its ceiling "
          "tax is zero\n   by construction, so it is also immune to the "
          "additive-versus-multiplicative\n   question. It is the campaign's "
          "highest-value live experiment (E59, PR #62).")

    print("\nUNRESOLVED: psi ambiguity")
    print(f"   The live pricer runs PSI_MTP = {L.PSI_MTP}. Ledger 189(D) and "
          "my own\n   e54_gap_decomposition.py label the same quantity "
          "0.693391. If 0.693391 is\n   correct, every number in this table "
          "rises by a further x1.029. Resolve\n   before quoting these figures "
          "as final.")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.exit(self_test())
    report()
