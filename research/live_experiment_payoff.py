#!/usr/bin/env python3
"""Ranked payoff curves for the two live experiments that can move the frontier.

E56 (edward, PR #59)  -- stream-aware draft-depth schedule. Step 0 counterfactual
                         gives per-prompt LEG gains on beagle and medicine.
E59 (thorfinn, PR #62) -- <T,5,5> at rows_per_simd=2. Rung 1 passed with ZERO
                         ceiling dose, so the whole promotion case now turns on
                         the measured r=2 x re-read tax.

Both are priced on OUR OWN published board row `ca9251b8` (score
3.23250848263467), not on the crown tree, because that is the row a new
submission actually replaces.

harness=ranked throughout. Every input is a ranked-leg quantity or is labelled.

Run:  python3 research/live_experiment_payoff.py
"""

import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import pricing_order as P                                       # noqa: E402
import qmv_score_leverage as L                                  # noqa: E402

# ------------------------------------------------------------------ our board
# `ca9251b8`, candidate 2b0c36a0. Per-prompt raw_p from the official receipt.
OUR_RAW = {
    "plutarch": 1.2528,
    "drama": 1.9167,
    "travel": 2.1798,
    "beagle": 3.1202,
    "medicine": 3.3449,
    "essays": 3.3661,
    "republic": 3.3940,
    "botany": 3.4254,
}
OUR_SCORE = 3.23250848263467
FRONTIER = 3.24985583421771
DEFICIT_PCT = 0.5367
# RETRACTED by ledger 193(E): this is 2 sd of the SERIAL leg's jitter applied to the
# score, and the median over eight prompts does not average the candidate-leg common
# mode away. The measured single-pair ranked MDE is 2.10 %, 7.4x larger. The value
# below is kept so this module's published arithmetic stays reproducible; import
# research/ranked_noise.py for any NEW ranked pricing.
MDE_PCT = 0.283

# Ranked mean draft length per prompt, from the same receipt. Used only to
# extrapolate a depth-policy gain from the two prompts edward simulated.
MEAN_DRAFT = {
    "plutarch": 0.1540,
    "drama": 2.2976,
    "travel": 2.6557,
    "beagle": 4.5327,
    "medicine": 4.7677,
    "republic": 5.2697,
    "essays": 5.4253,
    "botany": 5.7765,
}

# --------------------------------------------------------------- E56 evidence
# edward's Step 0 counterfactual, PR #59 comment 2. Signs as he reported them:
# negative = less time. LEG basis (decode basis is the larger pair).
E56_LEG_GAIN_PCT = {"beagle": 2.84, "medicine": 2.03}          # E46-ratio shape
E56_LEG_GAIN_PCT_CONSERVATIVE = {                              # measured-marginal shape
    # decode basis -1.73 / -1.80 rebased to the leg by the per-prompt dilution
    "beagle": 1.73 * 0.91552,
    "medicine": 1.80 * 0.90953,
}


def median_of_eight(vals):
    s = sorted(vals)
    return 0.5 * (s[3] + s[4])


def score_from_leg_gains(gains_pct):
    """gains_pct: prompt -> % REDUCTION in that prompt's timed leg.

    raw_p is inversely proportional to candidate leg time, so a g% reduction
    multiplies raw_p by 1/(1-g/100).
    """
    out = []
    for p, r in OUR_RAW.items():
        g = gains_pct.get(p, 0.0) / 100.0
        out.append(r / (1.0 - g))
    return median_of_eight(out)


def score_pct(gains_pct):
    base = median_of_eight(OUR_RAW.values())
    return 100.0 * (score_from_leg_gains(gains_pct) - base) / base


def score_from_raw_changes(raw_pct):
    """raw_pct: prompt -> % change in that prompt's raw_p. NEGATIVE = regression.

    Separate entry point from score_from_leg_gains on purpose. A leg REDUCTION
    and a raw_p INCREASE are the same event with opposite signs, and mixing the
    two conventions by hand is the single error I have made most often in this
    campaign. Keep them in different named functions.
    """
    return median_of_eight(r * (1.0 + raw_pct.get(p, 0.0) / 100.0)
                           for p, r in OUR_RAW.items())


def score_pct_from_raw_changes(raw_pct):
    base = median_of_eight(OUR_RAW.values())
    return 100.0 * (score_from_raw_changes(raw_pct) - base) / base


def headroom_pct(prompt):
    """How far this prompt's raw_p can rise before it stops being decision-
    relevant, i.e. before the next order statistic above it takes its place."""
    s = sorted(OUR_RAW.values())
    r = OUR_RAW[prompt]
    above = [v for v in s if v > r]
    if not above:
        return float("inf")
    return 100.0 * (above[0] - r) / r


def e56_scenarios(gain):
    """Three ways the two simulated prompts could generalise to all eight."""
    scen = {}
    # S1: only the two prompts edward simulated move.
    scen["S1 simulated pair only"] = dict(gain)
    # S2: NEAREST-ANCHOR, never extrapolated.
    #
    # A two-point fit in mean draft length CANNOT be extrapolated here. The two
    # anchors sit 0.235 apart and the fitted slope is negative, so a linear
    # extrapolation to plutarch (4.38 below the nearest anchor, a 19x reach)
    # returns +17.9 %, which is absurd. The self-test negative control catches
    # it. So: interpolate strictly inside the anchor interval, hold the nearest
    # anchor above it, and credit nothing below it -- a prompt whose mean draft
    # is under 4 rarely reaches the 4->5 boundary this mechanism corrects.
    x0, x1 = MEAN_DRAFT["beagle"], MEAN_DRAFT["medicine"]
    y0, y1 = gain["beagle"], gain["medicine"]
    s2 = {}
    for p, md in MEAN_DRAFT.items():
        if md < x0:
            s2[p] = 0.0
        elif md > x1:
            s2[p] = y1
        else:
            s2[p] = y0 + (y1 - y0) * (md - x0) / (x1 - x0)
    scen["S2 nearest anchor, no extrap"] = s2
    # S3: uniform at the SMALLER of the two measured gains.
    scen["S3 uniform at min"] = {p: min(gain.values()) for p in MEAN_DRAFT}
    return scen


# --------------------------------------------------- E56 -> E59 cross-pricing
# edward's PR #59 comment 8 simulates the RANKED width mixture directly from the
# published receipt moments, and reports M=5 QMV *time* share on the two
# score-setting prompts:
#     beagle   21.82 - 26.39 %
#     medicine 19.35 - 21.55 %
# The union is the honest corner set for the one width E59 changes. Compare with
# the two mixtures pricing_order ships: e48 12.1744 %, e53_mid 22.3363 %.
E56_RANKED_SHARE_M5 = (19.35, 26.39)


def e59_tax_curve_at_share(share_m5, taxes=(0.0, 4.0, 8.0, 10.54, 14.253, 16.0)):
    """E59 payoff using an explicit M=5 QMV time share rather than a mixture."""
    rows = []
    for tax in taxes:
        surviving = (abs(P.CELL_WIN_PCT_M5) - tax) / abs(P.CELL_WIN_PCT_M5)
        removed = share_m5 / 100.0 * abs(P.CELL_WIN_PCT_M5)
        full = L.PSI_MTP * removed
        lo = P.f(full * surviving * P.REBASE[0] * P.TRANSFER_UNION[0])
        hi = P.f(full * surviving * P.REBASE[1] * P.TRANSFER_UNION[1])
        rows.append({"tax": tax, "surviving": surviving, "lo": lo, "hi": hi})
    return rows


def e59_tax_curve():
    """Ranked score gain of the <T,5,5> r=2 route vs the measured r=2 tax.

    Rung 1 measured a ZERO ceiling dose, so the only remaining discount is the
    x re-read tax. surviving = (|win| - tax) / |win|.
    """
    rows = []
    for tax in (0.0, 2.0, 4.0, 6.0, 8.0, 10.54, 13.0, 16.0, 20.253):
        surviving = (abs(P.CELL_WIN_PCT_M5) - tax) / abs(P.CELL_WIN_PCT_M5)
        row = {"tax": tax, "surviving": surviving}
        for mixture in ("e48", "e53_mid"):
            full = P.full_leg_gain(mixture)
            lo = P.f(full * surviving * P.REBASE[0] * P.TRANSFER_UNION[0])
            hi = P.f(full * surviving * P.REBASE[1] * P.TRANSFER_UNION[1])
            row[mixture] = (lo, hi)
        rows.append(row)
    return rows


def self_test():
    checks = []

    def chk(name, got, want, tol=1e-9):
        ok = abs(got - want) <= tol
        checks.append((name, ok, got, want))
        return ok

    # 1-2. Our board row reproduces from the eight raw_p values.
    chk("median of our eight raw_p == published score", median_of_eight(OUR_RAW.values()),
        OUR_SCORE, 5e-5)
    chk("deficit to frontier", 100.0 * (FRONTIER - OUR_SCORE) / OUR_SCORE,
        DEFICIT_PCT, 2e-3)

    # 3. Zero gain is a zero score change.
    chk("f(0) == 0", score_pct({}), 0.0, 1e-12)

    # 4. A uniform leg gain g moves the median by exactly g/(1-g).
    g = 3.0
    chk("uniform gain is exact", score_pct({p: g for p in OUR_RAW}),
        100.0 * (1.0 / (1.0 - g / 100.0) - 1.0), 1e-9)

    # 5. NEGATIVE CONTROL: improving only plutarch (rank 1) must move nothing.
    chk("plutarch alone is inert", score_pct({"plutarch": 50.0}), 0.0, 1e-12)

    # 6. NEGATIVE CONTROL: improving only republic (rank 8) must move nothing.
    chk("republic alone is inert", score_pct({"republic": 50.0}), 0.0, 1e-12)

    # 7. POSITIVE CONTROL: improving beagle alone (rank 4) must move the median.
    v = score_pct({"beagle": 5.0})
    checks.append(("beagle alone moves the median", v > 0.05, v, ">0.05"))

    # 8. POSITIVE CONTROL: the pricer must SATURATE -- a huge beagle gain cannot
    #    keep paying, because beagle stops being the 4th order statistic.
    small = score_pct({"beagle": 5.0})
    huge = score_pct({"beagle": 60.0})
    checks.append(("beagle gain saturates", huge < 12.0 * small, (small, huge), "sublinear"))

    # 9. Monotone in the tax: less tax is never worth less.
    curve = e59_tax_curve()
    mono = all(curve[i]["e48"][0] >= curve[i + 1]["e48"][0] - 1e-12
               for i in range(len(curve) - 1))
    checks.append(("e59 payoff monotone decreasing in tax", mono, mono, True))

    # 10. At the full 20.253 tax the route is exactly worthless.
    chk("tax == |win| gives zero", curve[-1]["e48"][0], 0.0, 1e-12)

    # 11. The published E44 tax row reproduces ledger 191's corrected band.
    row = [r for r in curve if abs(r["tax"] - 10.54) < 1e-9][0]
    chk("e48 at 10.54 tax lo == ledger 191", row["e48"][0], 0.6931, 2e-3)
    chk("e48 at 10.54 tax hi == ledger 191", row["e48"][1], 0.8175, 2e-3)
    chk("e53_mid at 10.54 tax lo == ledger 191", row["e53_mid"][0], 1.1598, 2e-3)
    chk("e53_mid at 10.54 tax hi == ledger 191", row["e53_mid"][1], 1.2702, 2e-3)

    # 15. NEGATIVE CONTROL: the wrong pricing order must NOT reproduce 191.
    wrong = P.price_then_shrink("e48", P.TRANSFER_UNION[0], P.REBASE[0])
    checks.append(("wrong order does not reproduce 191", abs(wrong - 0.6931) > 0.05,
                   wrong, "!= 0.6931"))

    # 16. E56 S3 uniform must equal the closed form for a uniform gain.
    s = e56_scenarios(E56_LEG_GAIN_PCT)["S3 uniform at min"]
    gmin = min(E56_LEG_GAIN_PCT.values())
    chk("E56 S3 == closed form", score_pct(s),
        100.0 * (1.0 / (1.0 - gmin / 100.0) - 1.0), 1e-9)

    # 17. E56 S1 must be smaller than S3, because a pair-only gain is taxed by
    #     substitution while a uniform gain is not.
    s1 = score_pct(e56_scenarios(E56_LEG_GAIN_PCT)["S1 simulated pair only"])
    s3 = score_pct(s)
    checks.append(("E56 pair-only < uniform (substitution tax)", s1 < s3, (s1, s3), True))

    # 18. NEGATIVE CONTROL: S2 must never extrapolate. plutarch is clamped to
    #     zero and no prompt may exceed the larger anchor.
    s2 = e56_scenarios(E56_LEG_GAIN_PCT)["S2 nearest anchor, no extrap"]
    chk("E56 S2 clamps plutarch to zero", s2["plutarch"], 0.0, 1e-12)
    hi = max(E56_LEG_GAIN_PCT.values())
    checks.append(("E56 S2 never exceeds the larger anchor",
                   max(s2.values()) <= hi + 1e-12, max(s2.values()), f"<={hi}"))

    # 19-20. Sanity on the bars themselves.
    chk("MDE bar", MDE_PCT, 0.283, 1e-12)
    chk("deficit bar", DEFICIT_PCT, 0.5367, 1e-12)

    bad = [c for c in checks if not c[1]]
    for name, ok, got, want in checks:
        print(f"  [{'ok' if ok else 'FAIL'}] {name}: got={got} want={want}")
    print(f"  {len(checks) - len(bad)}/{len(checks)} checks passed")
    return not bad


def report():
    print("=" * 78)
    print("LIVE EXPERIMENT PAYOFF  (harness=ranked, priced on OUR row ca9251b8)")
    print("=" * 78)
    print(f"our published score {OUR_SCORE:.14f}   frontier {FRONTIER:.14f}")
    print(f"deficit {DEFICIT_PCT:.4f} %   ranked MDE at 2 sd {MDE_PCT:.3f} %")

    print("\n--- E56  edward PR #59  stream-aware draft-depth schedule -------------")
    for label, gains in (("E46-ratio shape (his headline)", E56_LEG_GAIN_PCT),
                         ("measured-marginal shape (his conservative)",
                          E56_LEG_GAIN_PCT_CONSERVATIVE)):
        print(f"\n  {label}: " + ", ".join(f"{k} -{v:.3f} %" for k, v in gains.items()))
        for name, sc in e56_scenarios(gains).items():
            pct = score_pct(sc)
            new = score_from_leg_gains(sc)
            verdict = ("PROMOTES" if new > FRONTIER else
                       "closes deficit" if pct > DEFICIT_PCT else
                       "over MDE" if pct > MDE_PCT else "under MDE")
            print(f"    {name:28s}  score {new:.6f}  {pct:+.4f} %  "
                  f"{pct / MDE_PCT:5.1f}x MDE  {verdict}")

    print("\n--- E59  thorfinn PR #62  <T,5,5> at rows_per_simd=2 ------------------")
    print("  Rung 1 measured ZERO ceiling dose, so the tax is the only discount.")
    print(f"  {'r=2 tax %':>10s} {'surviving':>10s} {'e48 lo..hi':>20s} "
          f"{'e53_mid lo..hi':>20s}  verdict")
    for r in e59_tax_curve():
        lo48, hi48 = r["e48"]
        lo53, hi53 = r["e53_mid"]
        worst = lo48
        verdict = ("closes deficit at worst corner" if worst > DEFICIT_PCT else
                   "over MDE at worst corner" if worst > MDE_PCT else
                   "under MDE at worst corner")
        tag = "  <- E44 inherited" if abs(r["tax"] - 10.54) < 1e-9 else ""
        print(f"  {r['tax']:10.2f} {r['surviving']:10.4f} "
              f"{lo48:8.4f}..{hi48:-7.4f} {lo53:11.4f}..{hi53:-7.4f}  {verdict}{tag}")

    print("\n  Break-even taxes (worst corner e48 lo):")
    for bar, name in ((MDE_PCT, "MDE"), (DEFICIT_PCT, "deficit")):
        lo, hi = 0.0, abs(P.CELL_WIN_PCT_M5)
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            surv = (abs(P.CELL_WIN_PCT_M5) - mid) / abs(P.CELL_WIN_PCT_M5)
            v = P.f(P.full_leg_gain("e48") * surv * P.REBASE[0] * P.TRANSFER_UNION[0])
            if v > bar:
                lo = mid
            else:
                hi = mid
        print(f"    tax below {lo:6.2f} % keeps the worst corner above the {name}")


if __name__ == "__main__":
    print("self-test:")
    ok = self_test()
    print()
    report()
    sys.exit(0 if ok else 1)
