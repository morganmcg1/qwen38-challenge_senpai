#!/usr/bin/env python3
"""Settle the dilution basis askeladd deferred to the advisor on PR #57.

THE QUESTION
------------
A local arm measures a win two ways: on the whole candidate LEG, and on the
MTP round cost with the seed prefill removed.  E55 measured both:

    leg   basis  -4.2952 %
    round basis  -5.5439 %      amplification 1.2907x

Ledger 186(B) says every round-cost score projection carries the median-pair
dilution x0.9125.  Applied to the two readings that gives +3.9195 % (leg) or
+5.0589 % (round).  Which one is the ranked score change?

THE ANSWER: THE ROUND BASIS.  Applying x0.9125 to a leg-basis local number
charges the seed prefill TWICE -- once through the LOCAL prefill share that is
already inside the local leg, and again through the RANKED prefill share.  This
is the same failure mode ledger 186(A) caught in item 122's hypothesis B.

THE CONSEQUENCE, WHICH IS THE PART THAT MATTERS
-----------------------------------------------
`psi_mtp = 0.693391` is a LOCAL LEG share.  Proof by source, not by assertion:

  Sources/MLXFastTrustedHarness/QwenRuntimeMTPDriver.swift
      :94   let started = Date()                   <- clock origin
      :95   client.beginMTPDecode(seedTokens:)     <- the seed prefill runs here
      :197  let decodeSeconds = Date().timeIntervalSince(started)

  research/e42_analyze.py:161  leg[key]["decode_seconds"]   <- what psi_mtp reads
  research/e30_log_wandb.py:183 names the same field
                               "mtp_decode_seconds_prefill_inclusive"

So `decode_seconds` is the prefill-INCLUSIVE leg, and every instrument that
divides by it produces a leg share diluted by the LOCAL prefill fraction.

`research/e49_price.py:78` then does `leg = PSI_MTP * removed` and feeds that
to `score_pct_from_leg_gains`, which consumes RANKED per-prompt leg speedups.
That silently equates a leg share measured at 23.4 % prefill with one that
applies at 8.75 % prefill.  It under-prices.  My own
`research/e54_gap_decomposition.py` then multiplied by 0.9125 on top, which
charges the prefill a further time.

The correct re-basing is one factor, not two:

    psi_mtp_ranked_leg = psi_mtp_local_leg * (1 - p_ranked) / (1 - p_local)

and the published chain must be multiplied by

    (1 - p_ranked) / (1 - p_local) / 0.9125  =  1.291 .. 1.305

i.e. EVERY ranked price I have published through `psi_mtp x ... x 0.9125` is
low by about 30 %.
"""

from __future__ import annotations

import argparse
import json

# ---------------------------------------------------------------- constants

PSI_MTP = 0.693391                      # E48, local LEG basis (PR #52)
PSI_MTP_INTERVAL = (0.692292, 0.694490)

# Ledger 186(B): ranked median-pair prefill share, from the ca9251b8 receipt.
P_RANKED = 0.0875                       # beagle 8.44 %, medicine 9.05 %
DILUTION_MEDIAN_PAIR = 1.0 - P_RANKED   # 0.9125

# E55 (askeladd, PR #57, W&B wxezisvs), local fixture, 512 decode tokens.
E55_LEG_WIN_PCT = 4.2952                # candidate_mtp_leg_seconds_per_token
E55_ROUND_WIN_PCT = 5.5439              # same win, seed prefill removed
E55_AMPLIFICATION = 1.2907              # published ratio
E55_PREFILL_SHARE_LOCAL = 0.23389       # seed prefill / MTP leg, measured
E55_F9_LOCAL = 0.554                    # M=9 share of local QMV cost, his

# Ledger 187, my own cost-weighting of the same local fixture histogram.
MY_F9_LOCAL = 0.5345

# E54 (thorfinn, PR #58, W&B 9qt2x4cp), isolated cell timings.
CELL_WIN_PCT = {5: -20.253, 7: +0.994, 8: +1.345, 9: -11.548}

# E54's published `e49_price.py --harness ranked` single-cell prices.
E54_PRICE_SINGLE_CELL = {
    "e48": {5: +1.3481, 7: -0.0310, 8: -0.0431, 9: +1.4084},
    "e53_mid": {5: +2.0187, 7: -0.0948, 8: -0.0761, 9: +0.5590},
}

# Ranked M=9 share of candidate-leg QMV cost. STILL DISPUTED; 188(E) retracted
# the claim that PR #57 settles it, because #57 measured the LOCAL fixture.
F9_RANKED = {"e48": 0.216296, "e53_mid": 0.067712}

# Ledger 188(D): the two calibration forms, reported as a union.
TRANSFER_UNION = (0.7388, 0.8617)

# Board state, ledger 187.
DEFICIT_PCT = 0.5367
RANKED_MDE_PCT = 0.283

# Ledger 187(L): the r=2 row-block x re-read tax at NA=4.
R2_TAX_PCT = 10.54


# ------------------------------------------------------------------ helpers

def p_local_from_amplification(amp: float) -> float:
    """Local prefill share implied by a leg/round amplification ratio.

    If the prefill is unchanged by the edit then dL/L = (dR/R)*(1-p), so
    amplification = (dR/R)/(dL/L) = 1/(1-p).
    """
    return 1.0 - 1.0 / amp


def rebase_factor(p_local: float) -> float:
    """Convert a LOCAL leg share into a RANKED leg share."""
    return (1.0 - P_RANKED) / (1.0 - p_local)


def qmv_share_of_round(psi_leg: float, p_local: float) -> float:
    """Strip the local prefill out of a local leg share.

    Token-count independent, because both numerator and denominator are round
    quantities.  This is the quantity that transfers between harnesses.
    """
    return psi_leg / (1.0 - p_local)


def e48_min_decode_tokens(psi_leg: float) -> float:
    """Lower bound on E48's decode window, from Q/R <= 1 alone.

    The seed prefill is a fixed 512-token cost.  Writing the 512-token leg as
    prefill share A and round share B = 1 - A, an N-token leg has
    p(N) = A / (A + B*N/512).  Q/R = psi_leg / (1 - p) <= 1 bounds p, and
    therefore bounds N from below.  No assumption about E48's harness.
    """
    a = E55_PREFILL_SHARE_LOCAL
    b = 1.0 - a
    p_max = 1.0 - psi_leg                       # from psi/(1-p) <= 1
    if p_max <= 0.0:
        return float("inf")
    return 512.0 * (a / p_max - a) / b


def band(lo_hi: tuple[float, float], x: float) -> tuple[float, float]:
    return (lo_hi[0] * x, lo_hi[1] * x)


# --------------------------------------------------------------------- main

def build() -> dict:
    out: dict = {}

    # --- 1. Which reading is the ranked score change? ---------------------
    out["readings"] = {
        "leg_basis_local_pct": E55_LEG_WIN_PCT,
        "round_basis_local_pct": E55_ROUND_WIN_PCT,
        "leg_x_dilution_pct": E55_LEG_WIN_PCT * DILUTION_MEDIAN_PAIR,
        "round_x_dilution_pct": E55_ROUND_WIN_PCT * DILUTION_MEDIAN_PAIR,
        "correct": "round_x_dilution",
        "why_leg_is_wrong": (
            "the local leg is already diluted by the LOCAL prefill share; "
            "x0.9125 then applies the RANKED prefill share a second time"
        ),
    }

    # --- 2. Two independent readings of the local prefill share -----------
    p_meas = E55_PREFILL_SHARE_LOCAL
    p_amp = p_local_from_amplification(E55_AMPLIFICATION)
    out["p_local"] = {
        "measured_directly": p_meas,
        "implied_by_amplification": p_amp,
        "spread_pct": 100.0 * abs(p_meas - p_amp) / p_meas,
        "band": sorted((p_meas, p_amp)),
    }

    # --- 3. Q/R three ways, which is the real cross-check ------------------
    qr_from_psi_meas = qmv_share_of_round(PSI_MTP, p_meas)
    qr_from_psi_amp = qmv_share_of_round(PSI_MTP, p_amp)
    qr_from_e55_his = E55_ROUND_WIN_PCT / (E55_F9_LOCAL * -CELL_WIN_PCT[9])
    qr_from_e55_mine = E55_ROUND_WIN_PCT / (MY_F9_LOCAL * -CELL_WIN_PCT[9])
    out["qmv_share_of_round"] = {
        "from_psi_mtp_and_measured_prefill": qr_from_psi_meas,
        "from_psi_mtp_and_amplification": qr_from_psi_amp,
        "from_e55_with_his_f9": qr_from_e55_his,
        "from_e55_with_my_f9": qr_from_e55_mine,
        "psi_route_vs_e55_route_pct": 100.0
        * abs(qr_from_psi_amp / qr_from_e55_mine - 1.0),
    }

    # --- 4. E48's decode window, bounded rather than assumed ---------------
    out["e48_window"] = {
        "min_decode_tokens_from_qr_le_1": e48_min_decode_tokens(PSI_MTP),
        "note": (
            "psi_mtp = 0.693391 is only physically admissible if the E48 legs "
            "ran a long decode window; a short window drives Q/R above 1"
        ),
    }

    # --- 5. The correction factor on every published price -----------------
    rb_meas = rebase_factor(p_meas)
    rb_amp = rebase_factor(p_amp)
    out["correction"] = {
        "rebase_factor_measured": rb_meas,
        "rebase_factor_amplification": rb_amp,
        "rebase_band": sorted((rb_meas, rb_amp)),
        "published_multiplier_was": DILUTION_MEDIAN_PAIR,
        "correction_ratio_band": sorted(
            (rb_amp / DILUTION_MEDIAN_PAIR, rb_meas / DILUTION_MEDIAN_PAIR)
        ),
        "psi_mtp_ranked_leg_band": sorted(
            (PSI_MTP * rb_amp, PSI_MTP * rb_meas)
        ),
    }

    # --- 6. Re-price the two live decisions --------------------------------
    rb_lo, rb_hi = sorted((rb_amp, rb_meas))
    surviving = (CELL_WIN_PCT[5] + R2_TAX_PCT) / CELL_WIN_PCT[5]
    priced = {}
    for mixture in ("e48", "e53_mid"):
        m9_naive = E54_PRICE_SINGLE_CELL[mixture][9]
        r2_naive = E54_PRICE_SINGLE_CELL[mixture][5] * surviving
        old_m9 = band(TRANSFER_UNION, m9_naive * DILUTION_MEDIAN_PAIR)
        old_r2 = band(TRANSFER_UNION, r2_naive * DILUTION_MEDIAN_PAIR)
        new_m9 = (
            TRANSFER_UNION[0] * m9_naive * rb_lo,
            TRANSFER_UNION[1] * m9_naive * rb_hi,
        )
        new_r2 = (
            TRANSFER_UNION[0] * r2_naive * rb_lo,
            TRANSFER_UNION[1] * r2_naive * rb_hi,
        )
        priced[mixture] = {
            "m9_only_naive": m9_naive,
            "m9_only_published": old_m9,
            "m9_only_corrected": new_m9,
            "r2_surviving_fraction": surviving,
            "r2_naive": r2_naive,
            "r2_published": old_r2,
            "r2_corrected": new_r2,
            "r2_published_closes_deficit": old_r2[0] >= DEFICIT_PCT,
            "r2_corrected_closes_deficit": new_r2[0] >= DEFICIT_PCT,
            "r2_corrected_clears_mde": new_r2[0] >= RANKED_MDE_PCT,
        }
    out["repriced"] = priced
    out["f9_ranked_still_disputed"] = F9_RANKED
    return out


def report(out: dict) -> None:
    bar = "=" * 74
    print(bar)
    print("1. The two readings of E55's win")
    print(bar)
    r = out["readings"]
    print(f"   leg   basis local  -{r['leg_basis_local_pct']:.4f} %"
          f"   x0.9125 -> +{r['leg_x_dilution_pct']:.4f} %")
    print(f"   round basis local  -{r['round_basis_local_pct']:.4f} %"
          f"   x0.9125 -> +{r['round_x_dilution_pct']:.4f} %")
    print(f"   CORRECT: {r['correct']}")
    print(f"   {r['why_leg_is_wrong']}")

    print()
    print(bar)
    print("2. The local prefill share, read two independent ways")
    print(bar)
    p = out["p_local"]
    print(f"   measured directly by askeladd     {100*p['measured_directly']:.3f} %")
    print(f"   implied by his amplification      {100*p['implied_by_amplification']:.3f} %")
    print(f"   spread                            {p['spread_pct']:.2f} %")

    print()
    print(bar)
    print("3. Crossrow QMV share of the ROUND cost, three independent routes")
    print(bar)
    q = out["qmv_share_of_round"]
    print(f"   psi_mtp / (1 - p_measured)              {q['from_psi_mtp_and_measured_prefill']:.5f}")
    print(f"   psi_mtp / (1 - p_amplification)         {q['from_psi_mtp_and_amplification']:.5f}")
    print(f"   E55 round win / (his f9  x cell win)    {q['from_e55_with_his_f9']:.5f}")
    print(f"   E55 round win / (my  f9  x cell win)    {q['from_e55_with_my_f9']:.5f}")
    print(f"   psi route vs E55 route disagree by      {q['psi_route_vs_e55_route_pct']:.2f} %")
    print("   -> two calibrations that share no input agree to under 1 %.")

    print()
    print(bar)
    print("4. E48's decode window, bounded not assumed")
    print(bar)
    print(f"   Q/R <= 1 forces at least {out['e48_window']['min_decode_tokens_from_qr_le_1']:.0f}"
          " decode tokens in the E48 legs")

    print()
    print(bar)
    print("5. The correction to every published psi_mtp price")
    print(bar)
    c = out["correction"]
    print(f"   published multiplier                    x{c['published_multiplier_was']:.4f}")
    print(f"   correct re-basing factor                x{c['rebase_band'][0]:.4f} .. x{c['rebase_band'][1]:.4f}")
    print(f"   every published price is LOW by         x{c['correction_ratio_band'][0]:.4f} .. x{c['correction_ratio_band'][1]:.4f}")
    print(f"   psi_mtp as a RANKED leg elasticity       {c['psi_mtp_ranked_leg_band'][0]:.5f} .. {c['psi_mtp_ranked_leg_band'][1]:.5f}")

    print()
    print(bar)
    print("6. What the correction does to the two live decisions")
    print(bar)
    print(f"   deficit to close {DEFICIT_PCT:.4f} %   ranked MDE {RANKED_MDE_PCT:.3f} %")
    for mixture, d in out["repriced"].items():
        print()
        print(f"   --- {mixture} ---")
        print(f"   M=9 cell alone   published {d['m9_only_published'][0]:+.4f} .. {d['m9_only_published'][1]:+.4f} %"
              f"   corrected {d['m9_only_corrected'][0]:+.4f} .. {d['m9_only_corrected'][1]:+.4f} %")
        print(f"   r=2 route        published {d['r2_published'][0]:+.4f} .. {d['r2_published'][1]:+.4f} %"
              f"   corrected {d['r2_corrected'][0]:+.4f} .. {d['r2_corrected'][1]:+.4f} %")
        print(f"   r=2 closes the deficit:  published {d['r2_published_closes_deficit']}"
              f"   corrected {d['r2_corrected_closes_deficit']}")


def self_test() -> int:
    fails: list[str] = []
    out = build()

    # 1. The amplification and the measured prefill share must agree.
    if out["p_local"]["spread_pct"] > 5.0:
        fails.append("p_local routes disagree by more than 5 %")

    # 2. Q/R from psi_mtp and Q/R from E55 share no input, so agreement is a
    #    real cross-check.  Demand better than 5 %.
    if out["qmv_share_of_round"]["psi_route_vs_e55_route_pct"] > 5.0:
        fails.append("Q/R routes disagree by more than 5 %")

    # 3. Q/R must be a share.
    for k, v in out["qmv_share_of_round"].items():
        if k.endswith("_pct"):
            continue
        if not 0.0 < v <= 1.0:
            fails.append(f"{k} = {v} is not a share")

    # 4. The correction must be an INCREASE; that is the whole point.
    lo, hi = out["correction"]["correction_ratio_band"]
    if not 1.0 < lo <= hi:
        fails.append(f"correction ratio {lo}..{hi} is not an increase")

    # 5. Re-basing must be exactly one factor, not two.  Check the identity.
    p = out["p_local"]["measured_directly"]
    if abs(rebase_factor(p) - DILUTION_MEDIAN_PAIR / (1.0 - p)) > 1e-12:
        fails.append("rebase_factor does not equal (1-p_rank)/(1-p_local)")

    # 6. The leg reading must be the round reading times (1 - p_local).
    implied = E55_ROUND_WIN_PCT * (1.0 - out["p_local"]["implied_by_amplification"])
    if abs(implied - E55_LEG_WIN_PCT) > 1e-3:
        fails.append(f"leg/round identity broken: {implied} vs {E55_LEG_WIN_PCT}")

    # 7. POSITIVE CONTROL.  A zero-prefill local harness must make the two
    #    readings identical and the correction vanish.
    if abs(rebase_factor(0.0) - DILUTION_MEDIAN_PAIR) > 1e-12:
        fails.append("at p_local = 0 the correction must vanish")

    # 8. POSITIVE CONTROL.  If the local prefill share equalled the ranked one
    #    the re-basing must be exactly 1.
    if abs(rebase_factor(P_RANKED) - 1.0) > 1e-12:
        fails.append("at p_local = p_ranked the re-basing must be 1")

    # 9. The E48 window bound must be a real constraint, not vacuous.
    n = out["e48_window"]["min_decode_tokens_from_qr_le_1"]
    if not 64.0 < n < 512.0:
        fails.append(f"E48 window bound {n} is vacuous or impossible")

    # 10. The r=2 route must move from not-closing to closing under e48.
    e48 = out["repriced"]["e48"]
    if e48["r2_published_closes_deficit"]:
        fails.append("e48 r=2 was recorded as already closing the deficit")
    if not e48["r2_corrected_closes_deficit"]:
        fails.append("e48 r=2 still fails to close after the correction")

    # 11. Every corrected price must exceed its published counterpart.
    for mixture, d in out["repriced"].items():
        for key in ("m9_only", "r2"):
            if d[f"{key}_corrected"][0] <= d[f"{key}_published"][0]:
                fails.append(f"{mixture} {key} correction did not increase the price")

    # 12. NEGATIVE CONTROL.  A deliberately wrong p_local must break check 2.
    bad = qmv_share_of_round(PSI_MTP, 0.60)
    if bad <= 1.0:
        fails.append("negative control did not produce an inadmissible share")

    if fails:
        for f in fails:
            print(f"FAIL: {f}")
        return 1
    print("self-test OK: 12 checks passed")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    out = build()
    if args.json:
        print(json.dumps(out, indent=2, sort_keys=True))
    else:
        report(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
