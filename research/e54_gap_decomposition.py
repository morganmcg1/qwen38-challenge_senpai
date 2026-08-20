#!/usr/bin/env python3
"""Decompose E54's local-cell-to-ranked-score gap into named terms.

E54 (thorfinn, PR #58) measured E27's exact composite on the real shipped QMV
table and priced it at +2.04..2.53 % of score against a board-observed
-0.3321 %.  This tool applies the two corrections that landed after
`research/e49_price.py` was written -- ledger 186(B)'s prefill dilution and
186(D)'s transfer law -- and reports what residual survives.

It also answers the question that decides the next experiment: is the residual
ADDITIVE (a fixed shared-register-ceiling tax charged once, however many cells
you edit) or MULTIPLICATIVE (a transfer factor that scales with the cell win)?
The two hypotheses make opposite predictions for a single-cell edit, and
thorfinn's register census gives the discriminating input for free.

Run:  python3 research/e54_gap_decomposition.py
      python3 research/e54_gap_decomposition.py --self-test
"""

from __future__ import annotations

import argparse
import json
import sys

# ---------------------------------------------------------------- E54 inputs

# Cell timing deltas, per cent, negative = faster.  E54 primary table.
CELL_WIN_PCT = {5: -20.253, 7: +0.994, 8: +1.345, 9: -11.548}

# Score-weighted per-width QMV shares, per cent.  E54 "per-width price" table.
SHARES = {
    "e48": {5: 12.1744, 7: 4.6307, 8: 4.7603, 9: 21.6296},
    "e53_mid": {5: 22.3363, 7: 14.1607, 8: 8.3949, 9: 6.7712},
}

# E54's own published score prices, per cent of published median, from
# `research/e49_price.py`.  🔴 That script has NO argparse: the `--harness
# ranked` flag this comment used to cite does not exist and was silently
# ignored; the harness is hard-coded at `e49_price.py:55`.  It also runs
# `qmv_score_leverage.PSI_MTP = 0.6736`, not the 0.693391 named below --
# see ledger 191(E), unresolved.  That tool recomputes the eight
# per-prompt raw ratios and re-medians them, so its output is NOT a linear
# function of the cell wins: the per-cell prices do not sum to the composite
# price, and the effective multiplier varies with effect size.  We therefore
# consume its published numbers as inputs instead of reproducing its algebra.
E54_PRICE_SINGLE_CELL = {
    "e48": {5: +1.3481, 7: -0.0310, 8: -0.0431, 9: +1.4084},
    "e53_mid": {5: +2.0187, 7: -0.0948, 8: -0.0761, 9: +0.5590},
}
E54_PRICE_COMPOSITE = {"e48": +2.2118, "e53_low": +2.0447,
                       "e53_mid": +2.2890, "e53_high": +2.5334}

# QMV share of the candidate MTP leg (askeladd E48, ledger 177).
PSI_MTP = 0.693391

# Ledger 186(B): median-pair round-cost -> score conversion.
DILUTION_MEDIAN_PAIR = 0.9125

# Ledger 186(D): a memory-traffic-bound local win transfers no better than
# this divisor.  The prefill/decode-round host-advantage ratio 7.58 / 2.14.
TRANSFER_DIVISOR_TRAFFIC = 3.55

# Ledger 184/186(E) two-parameter depth transfer, mean-pinned h ratio at the
# depth that a M=5 verify width sits on (depth 4).  research/e56_g_correction.py
TRANSFER_H_RATIO_D4 = (0.8343, 0.8617)

# Board anchor: E27's observed published-score change.
E27_BOARD_PCT = -0.3321

# Register census, E54.  (kernel max, production entry) for
# affine_qmv_fast<bfloat16_t,64,4,false>.
CENSUS = {
    "shipped": (108, 163),
    "e27_m5_only": (125, 182),
    "e27_m9_only": (129, 181),
    "e27_full": (129, 183),
}

# E54's own instrument limit: +4 constant over-count on mixed-NA cells,
# exact on single-group cells.
CENSUS_OVERCOUNT_MIXED = 4

# Local fixture width histogram, in VERIFY WIDTH units M (E49/E27, 78 rounds).
LOCAL_FIXTURE_HIST = {2: 1, 4: 5, 5: 5, 6: 23, 7: 4, 8: 6, 9: 34}

# thorfinn E46 refit QMV cost model, milliseconds.
QMV_T0, QMV_T_STREAM, QMV_T_ROW = 16.757, 27.532, 9.624
SHIPPED_IPG = {2: 2, 3: 3, 4: 4, 5: 3, 6: 3, 7: 4, 8: 4, 9: 3}

# End-to-end local null floor (askeladd E48 base2), per cent.
LOCAL_NULL_FLOOR_PCT = 0.0629

# Ranked minimum detectable effect at 2 sd, per cent of published median.
# RETRACTED by ledger 193(E): this is 2 sd of the SERIAL leg's jitter applied to the
# score, and the median over eight prompts does not average the candidate-leg common
# mode away. The measured single-pair ranked MDE is 2.10 %, 7.4x larger. The value
# below is kept so this module's published arithmetic stays reproducible; import
# research/ranked_noise.py for any NEW ranked pricing.
RANKED_MDE_PCT = 0.283
RANKED_MDE_WORST_PCT = 0.527

# Our deficit to the live promoted frontier, per cent.
DEFICIT_PCT = 0.5367

# Measured rows_per_simd=2 tax (x re-read per row block), per cent, at NA=4.
R2_TAX_PCT = 10.54

# Affine register ladders (E32, confirmed out of sample by E54's census).
# 🔴 E55 (askeladd) closed the law with zero fitted parameters:
#     peak_live_regs = 20 + 21*max(NA) + 4*[the cell has two distinct NA groups]
# The bare ladder below is valid ONLY on uniform cells.  The shipped maximum is
# set by M=7 `<T,7,4>`, whose only legal split {4,3} is MIXED, so it reads 108,
# not the 104 this file published before ledger 187(P)/189(C).
SHIPPED_TRUE_MAX_REGS = 108


def regs_r4(na: int, mixed: bool = False) -> int:
    return 20 + 21 * na + (4 if mixed else 0)


def regs_r2(na: int) -> int:
    return 16 + 15 * na


def qmv_cost_ms(m: int) -> float:
    """thorfinn E46 refit: T(M) = t0 + t_stream*ceil(M/IPG) + t_row*M."""
    import math

    streams = math.ceil(m / SHIPPED_IPG[m])
    return QMV_T0 + QMV_T_STREAM * streams + QMV_T_ROW * m


def local_fixture_shares() -> dict[int, float]:
    """Cost-weighted width shares of the LOCAL fixture, per cent."""
    weights = {m: n * qmv_cost_ms(m) for m, n in LOCAL_FIXTURE_HIST.items()}
    total = sum(weights.values())
    return {m: 100.0 * w / total for m, w in sorted(weights.items())}


def naive_score_pct(cells: tuple[int, ...], mixture: str) -> float:
    """E54's published `e49_price.py --harness ranked` price, per cent of score.

    `cells` selects a published row: the two-element key (5, 9) is the E27
    composite, and a one-element key is a single cell.
    """
    if tuple(sorted(cells)) == (5, 9):
        return E54_PRICE_COMPOSITE[mixture]
    if len(cells) == 1:
        return E54_PRICE_SINGLE_CELL[mixture][cells[0]]
    raise KeyError(f"no published E54 price for cells {cells}")


def linear_score_pct(cells: tuple[int, ...], mixture: str) -> float:
    """psi_mtp * sum(share * -win): the LINEARISED price, for contrast only.

    E54's tool is nonlinear, so this differs from `naive_score_pct`.  Keeping
    both makes the nonlinearity visible instead of hiding it in one number.
    """
    share = SHARES[mixture]
    return PSI_MTP * sum(share[m] * (-CELL_WIN_PCT[m]) / 100.0 for m in cells)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    out: dict = {}

    # --- 1. Reproduce E54's own pricing, then correct it -------------------
    composite = (5, 9)
    rows = []
    for mixture in ("e48", "e53_mid"):
        naive = naive_score_pct(composite, mixture)
        diluted = naive * DILUTION_MEDIAN_PAIR
        traffic = diluted / TRANSFER_DIVISOR_TRAFFIC
        hlo = diluted * TRANSFER_H_RATIO_D4[0]
        hhi = diluted * TRANSFER_H_RATIO_D4[1]
        rows.append(
            {
                "mixture": mixture,
                "naive": naive,
                "diluted": diluted,
                "transfer_traffic_3p55": traffic,
                "transfer_h_ratio_lo": hlo,
                "transfer_h_ratio_hi": hhi,
                "residual_vs_board_traffic": traffic - E27_BOARD_PCT,
                "residual_vs_board_h_lo": hlo - E27_BOARD_PCT,
                "residual_vs_board_h_hi": hhi - E27_BOARD_PCT,
            }
        )
    out["e27_composite"] = rows

    # --- 2. Single-cell M=5, priced the same way --------------------------
    m5_rows = []
    for mixture in ("e48", "e53_mid"):
        naive = naive_score_pct((5,), mixture)
        diluted = naive * DILUTION_MEDIAN_PAIR
        m5_rows.append(
            {
                "mixture": mixture,
                "naive": naive,
                "diluted": diluted,
                "transfer_traffic_3p55": diluted / TRANSFER_DIVISOR_TRAFFIC,
                "transfer_h_ratio_lo": diluted * TRANSFER_H_RATIO_D4[0],
                "transfer_h_ratio_hi": diluted * TRANSFER_H_RATIO_D4[1],
            }
        )
    out["m5_only"] = m5_rows

    # --- 3. Additive versus multiplicative residual -----------------------
    # Additive: a fixed ceiling tax A charged once, whatever the cell edit.
    #   board = corrected_prediction + A   ->   A = board - corrected
    # Multiplicative: board = k * corrected  ->  k = board / corrected
    add_mult = []
    for r in out["e27_composite"]:
        for key in ("transfer_traffic_3p55", "transfer_h_ratio_lo", "transfer_h_ratio_hi"):
            corrected = r[key]
            add_mult.append(
                {
                    "mixture": r["mixture"],
                    "transfer": key,
                    "corrected_prediction": corrected,
                    "additive_tax_pct": E27_BOARD_PCT - corrected,
                    "multiplicative_k": E27_BOARD_PCT / corrected if corrected else None,
                }
            )
    out["residual_model"] = add_mult

    # M=5-only outcome under each residual model, using the e48/e53 corrected
    # predictions and the tax or factor inferred from the E27 anchor.
    m5_forecast = []
    for m5 in out["m5_only"]:
        for comp, am in zip(out["e27_composite"] * 3, add_mult):
            if am["mixture"] != m5["mixture"]:
                continue
            corrected_m5 = m5[am["transfer"]]
            m5_forecast.append(
                {
                    "mixture": m5["mixture"],
                    "transfer": am["transfer"],
                    "m5_corrected": corrected_m5,
                    "under_additive": corrected_m5 + am["additive_tax_pct"],
                    "under_multiplicative": corrected_m5 * am["multiplicative_k"],
                }
            )
    out["m5_forecast"] = m5_forecast

    # --- 4. Ceiling dose carried by each single-cell edit -----------------
    ship_max, ship_entry = CENSUS["shipped"]
    dose = {}
    for arm, (kmax, entry) in CENSUS.items():
        dose[arm] = {
            "kernel_max": kmax,
            "entry": entry,
            "d_kernel_max": kmax - ship_max,
            "d_entry": entry - ship_entry,
            "share_of_full_entry_dose_pct": (
                100.0 * (entry - ship_entry) / (CENSUS["e27_full"][1] - ship_entry)
            ),
        }
    out["ceiling_dose"] = dose

    # --- 5. The rows_per_simd=2 escape ------------------------------------
    # <T,5,5> at r=4 needs 125 regs, above the shipped 104 true max.
    # <T,5,5> at r=2 needs 91 regs, BELOW it -> zero ceiling dose.
    escape = {
        "shipped_true_max_r4_na4": SHIPPED_TRUE_MAX_REGS,
        "t5_5_r4": regs_r4(5),
        "t5_5_r2": regs_r2(5),
        "raises_ceiling_r4": regs_r4(5) > SHIPPED_TRUE_MAX_REGS,
        "raises_ceiling_r2": regs_r2(5) > SHIPPED_TRUE_MAX_REGS,
        "r2_tax_pct_at_na4": R2_TAX_PCT,
        "net_cell_win_pct": CELL_WIN_PCT[5] + R2_TAX_PCT,
    }
    # 🔴 Ledger 191 corrects this block twice over.  The previous version read
    #     naive = E54_PRICE_SINGLE_CELL[mixture][5] * surviving
    # and claimed it "keeps the nonlinear ranked pricing".  It does the
    # opposite: that input is a SCORE that already passed through the concave
    # re-sorting pricer, so scaling it linearises exactly what 187(H) forbids.
    # It then multiplied by DILUTION_MEDIAN_PAIR, double-charging the prefill
    # per 189(D).  Both corrections now live in `research/pricing_order.py`,
    # which shrinks the LEG GAIN and prices once.
    import pricing_order as PO
    for mixture in ("e48", "e53_mid"):
        surviving = escape["net_cell_win_pct"] / CELL_WIN_PCT[5]
        escape["surviving_fraction_of_cell_win"] = surviving
        escape[f"score_{mixture}_traffic"] = (
            PO.shrink_then_price(mixture, 1.0 / TRANSFER_DIVISOR_TRAFFIC,
                                 PO.REBASE[0]))
        escape[f"score_{mixture}_h_lo"] = PO.shrink_then_price(
            mixture, TRANSFER_H_RATIO_D4[0], PO.REBASE[0])
        escape[f"score_{mixture}_h_hi"] = PO.shrink_then_price(
            mixture, TRANSFER_H_RATIO_D4[1], PO.REBASE[1])
        escape[f"score_{mixture}_union_lo"] = PO.shrink_then_price(
            mixture, PO.TRANSFER_UNION[0], PO.REBASE[0])
        escape[f"score_{mixture}_union_hi"] = PO.shrink_then_price(
            mixture, PO.TRANSFER_UNION[1], PO.REBASE[1])
    out["r2_escape"] = escape

    # --- 6. Local-fixture sensitivity, which is the measurement bias ------
    lf = local_fixture_shares()
    local = {
        "fixture_shares_pct": lf,
        "m5_local_share_pct": lf[5],
        "m5_ranked_share_pct": {k: SHARES[k][5] for k in SHARES},
        "amplification_e48": SHARES["e48"][5] / lf[5],
        "amplification_e53_mid": SHARES["e53_mid"][5] / lf[5],
        "local_leg_effect_pct": PSI_MTP * lf[5] * CELL_WIN_PCT[5] / 100.0,
        "null_floor_pct": LOCAL_NULL_FLOOR_PCT,
    }
    local["sd_above_floor"] = abs(local["local_leg_effect_pct"]) / LOCAL_NULL_FLOOR_PCT
    # E27 composite on the local fixture, for comparison with its -6.56 %.
    e27_local = PSI_MTP * sum(lf[m] * CELL_WIN_PCT[m] / 100.0 for m in (5, 9))
    local["e27_composite_local_leg_pct"] = e27_local
    local["e27_composite_local_observed_pct"] = -6.56
    out["local_fixture"] = local

    if args.json:
        print(json.dumps(out, indent=2, sort_keys=True))
        return 0

    print("=" * 78)
    print("E54 gap decomposition: from a QMV cell win to a ranked score change")
    print("=" * 78)

    print("\n1. E27 composite, E54's pricing corrected by ledger 186(B) and 186(D)")
    print(f"   {'mixture':9s} {'naive':>8s} {'x0.9125':>9s} {'/3.55':>8s} "
          f"{'xh_lo':>8s} {'xh_hi':>8s}   board={E27_BOARD_PCT:+.4f}")
    for r in out["e27_composite"]:
        print(f"   {r['mixture']:9s} {r['naive']:+8.4f} {r['diluted']:+9.4f} "
              f"{r['transfer_traffic_3p55']:+8.4f} "
              f"{r['transfer_h_ratio_lo']:+8.4f} {r['transfer_h_ratio_hi']:+8.4f}")
    print("   residual after correction, points of score:")
    for r in out["e27_composite"]:
        print(f"   {r['mixture']:9s} traffic {r['residual_vs_board_traffic']:+7.4f}   "
              f"h_ratio {r['residual_vs_board_h_lo']:+7.4f} .. "
              f"{r['residual_vs_board_h_hi']:+7.4f}")

    print("\n2. <T,5,5> alone, same corrections")
    print(f"   {'mixture':9s} {'naive':>8s} {'x0.9125':>9s} {'/3.55':>8s} "
          f"{'xh_lo':>8s} {'xh_hi':>8s}")
    for r in out["m5_only"]:
        print(f"   {r['mixture']:9s} {r['naive']:+8.4f} {r['diluted']:+9.4f} "
              f"{r['transfer_traffic_3p55']:+8.4f} "
              f"{r['transfer_h_ratio_lo']:+8.4f} {r['transfer_h_ratio_hi']:+8.4f}")
    print(f"   ranked MDE 2sd {RANKED_MDE_PCT:+.3f} (worst {RANKED_MDE_WORST_PCT:+.3f}), "
          f"deficit {DEFICIT_PCT:+.4f}")

    print("\n3. Is the residual ADDITIVE or MULTIPLICATIVE?")
    for am in out["residual_model"]:
        k = am["multiplicative_k"]
        print(f"   {am['mixture']:9s} {am['transfer']:22s} "
              f"corrected {am['corrected_prediction']:+7.4f}  "
              f"additive tax {am['additive_tax_pct']:+7.4f}  "
              f"mult k {k:+7.4f}")

    print("\n4. What <T,5,5> alone delivers under each residual model")
    for f in out["m5_forecast"]:
        print(f"   {f['mixture']:9s} {f['transfer']:22s} "
              f"corrected {f['m5_corrected']:+7.4f}  "
              f"additive {f['under_additive']:+7.4f}  "
              f"mult {f['under_multiplicative']:+7.4f}")

    print("\n5. Ceiling dose carried by each single-cell edit (E54 census)")
    print(f"   {'arm':14s} {'kmax':>5s} {'entry':>6s} {'dkmax':>6s} {'dentry':>7s} "
          f"{'% of full dose':>15s}")
    for arm, d in out["ceiling_dose"].items():
        print(f"   {arm:14s} {d['kernel_max']:5d} {d['entry']:6d} "
              f"{d['d_kernel_max']:+6d} {d['d_entry']:+7d} "
              f"{d['share_of_full_entry_dose_pct']:14.1f} %")

    print("\n6. The rows_per_simd=2 escape: a NA=5 cell BELOW the shipped ceiling")
    e = out["r2_escape"]
    print(f"   shipped true max, <T,7,4> MIXED {4,3} {e['shipped_true_max_r4_na4']:4d} regs")
    print(f"   <T,5,5> at r=4                        {e['t5_5_r4']:4d} regs  "
          f"raises ceiling: {e['raises_ceiling_r4']}")
    print(f"   <T,5,5> at r=2, two row blocks        {e['t5_5_r2']:4d} regs  "
          f"raises ceiling: {e['raises_ceiling_r2']}")
    print(f"   measured r=2 x re-read tax at NA=4    {e['r2_tax_pct_at_na4']:+.2f} %")
    print(f"   net cell win after that tax           {e['net_cell_win_pct']:+.3f} %")
    for mixture in ("e48", "e53_mid"):
        print(f"   score, {mixture:8s} union "
              f"{e[f'score_{mixture}_union_lo']:+7.4f}.."
              f"{e[f'score_{mixture}_union_hi']:+7.4f}  "
              f"/3.55 {e[f'score_{mixture}_traffic']:+7.4f}  "
              f"xh {e[f'score_{mixture}_h_lo']:+7.4f}.."
              f"{e[f'score_{mixture}_h_hi']:+7.4f}")
    print("   (shrink-then-price per ledger 191; see research/pricing_order.py)")

    print("\n7. The LOCAL fixture under-weights M=5, so a local test is biased against it")
    lo = out["local_fixture"]
    print("   local fixture cost-weighted width shares, per cent:")
    print("   " + "  ".join(f"M{m}={s:5.2f}" for m, s in lo["fixture_shares_pct"].items()))
    print(f"   M=5 local {lo['m5_local_share_pct']:.2f} % vs ranked "
          f"e48 {lo['m5_ranked_share_pct']['e48']:.2f} % "
          f"e53_mid {lo['m5_ranked_share_pct']['e53_mid']:.2f} %")
    print(f"   ranked/local amplification  e48 {lo['amplification_e48']:.2f}x   "
          f"e53_mid {lo['amplification_e53_mid']:.2f}x")
    print(f"   predicted LOCAL MTP-leg effect of <T,5,5> alone "
          f"{lo['local_leg_effect_pct']:+.4f} %  "
          f"= {lo['sd_above_floor']:.1f}x the {LOCAL_NULL_FLOOR_PCT} % null floor")
    print(f"   E27 composite on the local fixture: predicted "
          f"{lo['e27_composite_local_leg_pct']:+.3f} % vs observed "
          f"{lo['e27_composite_local_observed_pct']:+.3f} %")
    print()
    return 0


def self_test() -> int:
    fails = []

    # 1. Cost model reproduces the recorded M=9 QMV cost.
    if abs(qmv_cost_ms(9) - 185.969) > 5e-3:
        fails.append(f"qmv_cost_ms(9)={qmv_cost_ms(9)} != 185.969")

    # 2. Local fixture M=9 share reproduces the recorded 53.45 %.
    lf = local_fixture_shares()
    if abs(lf[9] - 53.45) > 0.05:
        fails.append(f"local M=9 share {lf[9]:.3f} != 53.45")

    # 3. Affine ladders reproduce E54's census on single-group cells exactly.
    for na, want in ((2, 62), (3, 83), (4, 104), (5, 125)):
        if regs_r4(na) != want:
            fails.append(f"regs_r4({na})={regs_r4(na)} != {want}")
    if regs_r2(12) != 196:
        fails.append(f"regs_r2(12)={regs_r2(12)} != 196")

    # 4. The r=2 escape must NOT raise the shipped ceiling; the r=4 form must.
    if not regs_r4(5) > regs_r4(4):
        fails.append("r=4 NA=5 should raise the ceiling")
    if regs_r2(5) >= regs_r4(4):
        fails.append(f"r=2 NA=5 ({regs_r2(5)}) should sit below {regs_r4(4)}")

    # 5. E54's published pricing is NONLINEAR: the per-cell prices must not sum
    #    to the composite price, and the linearised model must disagree with it.
    #    If either ever became true, this tool's decision to consume published
    #    numbers instead of recomputing them would no longer be justified.
    for mixture in ("e48", "e53_mid"):
        summed = sum(E54_PRICE_SINGLE_CELL[mixture][m] for m in (5, 9))
        if abs(summed - E54_PRICE_COMPOSITE[mixture]) < 0.05:
            fails.append(f"{mixture}: per-cell sum {summed:.4f} now matches the "
                         f"composite {E54_PRICE_COMPOSITE[mixture]:.4f}; the "
                         f"nonlinearity assumption needs re-checking")
        lin = linear_score_pct((5, 9), mixture)
        if abs(lin - E54_PRICE_COMPOSITE[mixture]) < 0.05:
            fails.append(f"{mixture}: linearised {lin:.4f} now matches published")

    # 6. Every correction must move the composite prediction DOWN but keep it
    #    positive, so the sign flip is never explained by the corrections alone.
    for mixture in ("e48", "e53_mid"):
        naive = naive_score_pct((5, 9), mixture)
        worst = naive * DILUTION_MEDIAN_PAIR / TRANSFER_DIVISOR_TRAFFIC
        if not 0.0 < worst < naive:
            fails.append(f"{mixture}: corrected {worst:.4f} not in (0, {naive:.4f})")
        if worst <= E27_BOARD_PCT:
            fails.append(f"{mixture}: corrections alone should not reach the board sign")

    # 7. Single-cell M=5 must carry most of the full entry-register dose.
    d_m5 = CENSUS["e27_m5_only"][1] - CENSUS["shipped"][1]
    d_full = CENSUS["e27_full"][1] - CENSUS["shipped"][1]
    if d_m5 / d_full < 0.9:
        fails.append(f"m5_only entry dose {d_m5}/{d_full} unexpectedly small")

    # 8. The local fixture must under-weight M=5 relative to both ranked
    #    mixtures, which is the measurement bias the brief relies on.
    for mixture in SHARES:
        if SHARES[mixture][5] <= lf[5]:
            fails.append(f"{mixture} M=5 share should exceed local {lf[5]:.2f}")

    if fails:
        for f in fails:
            print(f"FAIL: {f}")
        return 1
    print("self-test OK: 8 checks passed")
    return 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        raise SystemExit(self_test())
    raise SystemExit(main())
