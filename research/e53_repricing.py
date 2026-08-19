#!/usr/bin/env python3
"""E53 Part 3: re-price old-coefficient verdicts at psi_mtp = +0.693391.

Every conversion below goes through `research/qmv_score_leverage.py` (campaign
gate 26). No constant from that module is re-inlined here; the module's public
entry points take `psi` explicitly, which is how the superseding measurement is
threaded through without editing the gate.

NEW INPUT (the only constant this file introduces):
    psi_mtp = 0.693391, interval [0.692292, 0.694490] -- askeladd, two doses,
    measured on this base (45b7c6a4), quoted from the E53 assignment brief.
    It supersedes PSI_MTP = 0.6736 (askeladd E42, tree 04ad6bf1), which is
    what the module still carries as its default.

Kink discipline (ledger 177 / module order-statistics section): any score
price above kink_pct() (~+1.0551 %) is re-derived with
score_pct_from_leg_gains() on the scored pair, never by multiplication.
Prices above saturation_cap_pct() (~+4.7156 %) are unreachable for a
scored-pair-only mechanism; target_for() refuses them.

Output feeds research/e53_repricing.md. Run: python3 research/e53_repricing.py
"""

import contextlib
import io
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import qmv_score_leverage as qsl  # noqa: E402
from noise_floors import SCORE_BETWEEN_SUBMISSION  # noqa: E402

PSI_NEW = 0.693391          # askeladd, two doses, base 45b7c6a4 (E53 brief)
PSI_LO, PSI_HI = 0.692292, 0.694490
FLOOR = SCORE_BETWEEN_SUBMISSION.pct   # 0.7678 %, ledger 166, 17 sets, dof 23


def run_gate_selftest():
    """The gate must pass before any of its outputs are quoted."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        qsl.selftest()
    n_pass = buf.getvalue().count("PASS ")
    assert "SELFTEST PASSED" in buf.getvalue()
    return n_pass


def scored_pair_pct(leg_gain_pct):
    """Score %% when both scored prompts' legs gain leg_gain_pct."""
    return qsl.score_pct_from_leg_gains(
        {p: leg_gain_pct for p in qsl.SCORED_PROMPTS})


def kinked(linear_pct):
    """Order-statistics price for a linear (leg-gain) estimate.

    Below the kink the conversion is 1:1 by construction, so this is the
    identity there; above it the re-sort does the piecewise work.
    """
    return scored_pair_pct(linear_pct)


def price_per_width(wins, psi):
    """(linear, order-stat) score %% for a per-width kernel win map."""
    linear = qsl.mechanism_value_per_width(wins, gated=True, psi=psi)
    return linear, kinked(linear)


def interval(fn):
    """Evaluate fn(psi) at the measured value and both interval ends."""
    return fn(PSI_NEW), fn(PSI_LO), fn(PSI_HI)


def invert_scored_pair(score_pct):
    """Leg gain needed for a scored-pair score target, piecewise via bisection."""
    if score_pct > qsl.saturation_cap_pct():
        return None
    lo, hi = 0.0, 50.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if scored_pair_pct(mid) < score_pct:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def fmt3(tup):
    return "%+.4f [%+.4f, %+.4f]" % tup


def main():
    n = run_gate_selftest()
    print("gate 26 selftest: PASSED (%d PASS lines)" % n)
    print("floor = %.4f %%  kink = %+.4f %%  cap = %+.4f %%"
          % (FLOOR, qsl.kink_pct(), qsl.saturation_cap_pct()))
    print("module default psi (superseded) = %.4f; new psi = %.6f [%.6f, %.6f]"
          % (qsl.PSI_MTP, PSI_NEW, PSI_LO, PSI_HI))
    print("module f{7,8} = %.4f (HIST x T(M); askeladd census carried 0.1225)"
          % qsl.width_set_share((7, 8)))
    print()

    print("1 %% kernel-wide QMV win (all verify widths), score %%:")
    allw = tuple(sorted(m for m in qsl.HIST if m in qsl.T_BY_WIDTH))
    print("  ", fmt3(interval(
        lambda p: qsl.mechanism_value(allw, 1.0, gated=True, psi=p))))
    print()

    print("E44 r2 narrow M in {7,8} (r2 measured wins, weight-then-sum):")
    mixes = {
        "mlp_down only ": {7: 4.596, 8: 12.649},
        "equal shape mix": {7: (11.389 + 4.596) / 2, 8: (17.050 + 12.649) / 2},
        "attn_out only ": {7: 11.389, 8: 17.050},
    }
    for label, wins in mixes.items():
        lin = interval(lambda p: price_per_width(wins, p)[0])
        ost = interval(lambda p: price_per_width(wins, p)[1])
        print("  %s linear %s  order-stat %s" % (label, fmt3(lin), fmt3(ost)))
    print()

    print("E44 r1 all-width MMA, measured cells only {4,7,8}, equal shape mix:")
    r1 = {4: -(41.72 + 52.39) / 2, 7: (10.46 + 4.46) / 2, 8: (16.65 + 13.05) / 2}
    print("  linear", fmt3(interval(lambda p: price_per_width(r1, p)[0])),
          " (M in {5,6,9} cells not in the r1 table; pooled net was -7.341 %)")
    print()

    print("simdgroup MMA at M=9 (173(D) measured -10.37/-11.66):")
    print("  linear", fmt3(interval(
        lambda p: price_per_width({9: -(10.37 + 11.66) / 2}, p)[0])))
    print()

    print("M=9 two-stream prize (<T,9,5> at <=108 regs), corpus HIST:")
    w9 = 100.0 * qsl.STREAM_MS / qsl.T_BY_WIDTH[9]
    print("  win at M=9 = %.3f %% of width-9 cost; qmv_share(9) = %.4f"
          % (w9, qsl.qmv_share(9)))
    lin = interval(lambda p: price_per_width({9: w9}, p)[0])
    ost = interval(lambda p: price_per_width({9: w9}, p)[1])
    print("  linear %s  order-stat (scored-pair-only) %s" % (fmt3(lin), fmt3(ost)))
    print()

    print("E27 residual (register-step price), linear decomposition:")
    expected = interval(lambda p: 9.134 * p)
    print("  expected stream wins %s; observed -0.3321;" % fmt3(expected))
    print("  residual", fmt3(tuple(-0.3321 - e for e in expected)))
    print()

    print("E44 ceiling bound (uniform register-allocation term, bound 0.663 %):")
    print("  |dScore| <=", fmt3(interval(lambda p: 0.663 * p)),
          " (%.1f %% of floor)" % (100 * 0.663 * PSI_NEW / FLOOR))
    print()

    print("E49 c_ceiling = +10.6 % uniform QMV slowdown:")
    print("  leg %s -> score %s"
          % (fmt3(interval(lambda p: -10.6 * p)),
             fmt3(interval(lambda p: scored_pair_pct(-10.6 * p)))))
    print()

    print("<T,8,3> code-match edit (E46 measured +18.72 % slower at M=8):")
    print("  score", fmt3(interval(
        lambda p: price_per_width({8: -18.72}, p)[0])))
    print()

    print("Advisor QMV targets (kernel-wide gated win needed):")
    for label, tgt in (("crown gap 0.5193 %", 0.5193),
                       ("1 sd 0.7678 %", FLOOR),
                       ("2 sd 1.5356 %", 2 * FLOOR)):
        if tgt <= qsl.kink_pct():
            need = interval(lambda p: qsl.target_for(tgt, gated=True, psi=p))
        else:
            g = invert_scored_pair(tgt)
            need = tuple(g / p for p in (PSI_NEW, PSI_LO, PSI_HI))
        print("  %-20s %s %% QMV cost" % (label, fmt3(need)))


if __name__ == "__main__":
    main()
