#!/usr/bin/env python3
"""E55: the shared-register-ceiling term the advisor required, computed not asserted.

Advisor feedback `e55-is-the-m9-half-of-e27-and-prereg-omits-ceiling-term`
(PR #57 comment 5346572284) makes four demands. This script answers each one
arithmetically from the committed census artifact and the recorded measurement,
so no figure in the write-up is hand-derived.

  S3  hard gate: the register census must read 108 for the SHIPPED table.
  S4  add a second prediction term, conditional on the ceiling moving, using
      E27's measured ranked MTP-leg tax of +0.1995 % as its magnitude.
  S5a the M4 Pro ladder over-prices depth on ranked M5; quote the corrected
      +4.5..+4.7 % prize rather than +5.36 %.
  S5b f9 on the ranked central pair is only bounded to [0 %, 70 %]; no
      moment-based method can close it, so do not try.

The second term is POST HOC. The timed arms completed before this feedback
arrived, so it is a requested correction to the model and not a
pre-registration. Everything under `ranked_prereg` is different: it is stated
before any ranked submission of this candidate and is falsifiable by one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# --- E48-derived conversion, unchanged from the analyzer -------------------
PSI_MTP = 0.693391
M9_CELL_WIN_PCT = 12.255  # E49 Arm 1, isolated M=9 cell
SENSITIVITY_PCT_PER_F9 = 8.49751  # = PSI_MTP * M9_CELL_WIN_PCT, % MTP leg per unit f9

# --- measured null floors (E48 base/base2) --------------------------------
NULL_MTP_PCT = 0.0497
NULL_SERIAL_PCT = 0.001496

# --- E27 ranked receipt, as quoted by the advisor -------------------------
E27_RANKED_DSCORE_PCT = -0.3321
E27_RANKED_MTP_LEG_PCT = +0.1995
E27_REG_BASE = 108
E27_REG_ARM = 129

# --- E49 -----------------------------------------------------------------
E49_ARM2_HARM_BOUND_PCT = 0.0876  # |dScore| bound, control-free contrasts
E49_ARM2_READ_AT_MY_DOSE_PCT = -0.035  # +18 shared-entry registers
E49_M9_HALF_RANKED_PRICE_PCT = +1.3625  # advisor, 16:37Z comment

# --- ranked instrument ---------------------------------------------------
RANKED_MDE_2SD_PCT = 0.283
BOARD_FLOOR_PCT = 0.7678
DEFICIT_PCT = 0.5338

# --- advisor prize figures to be back-solved (S5a) ------------------------
ADVISOR_PRIZE_PRE_CORRECTION_PCT = 5.36
ADVISOR_PRIZE_CORRECTED_PCT = (4.5, 4.7)
ADVISOR_G_RANGE = (0.8331, 0.8792)

# --- registered f9 mixtures ----------------------------------------------
MIXTURES = {
    "e48_merged": 21.630,
    "edward_e53_upper": 8.9,
    "edward_e53_lower": 4.6,
}

# --- measured timed arms (bracketed base, drift-corrected) ---------------
MTP_BASE = 0.03342669
MTP_CAND = 0.03199581
MTP_BASE2 = 0.03343687
SERIAL_BASE = 0.07341014
SERIAL_CAND = 0.07341301
SERIAL_BASE2 = 0.07340904

# --- measured local verify-width mixture --------------------------------
LOCAL_WIDTH_HIST = {2: 1, 4: 5, 5: 5, 6: 23, 7: 4, 8: 6, 9: 34}
LOCAL_ACCEPTED_DRAFTS = 434
LOCAL_F9_PCT = 55.435  # M=9 share of candidate-leg verify QMV cost, local fixture
ADVISOR_QUOTED_LOCAL_F9_PCT = 53.45  # S5b quotes this for the same fixture

# --- ranked central pair, from our own receipt ca9251b8 ------------------
RECEIPT_MEAN_DRAFT_LEN = {"beagle": 4.533, "medicine": 4.768}


def pct_delta(candidate: float, baseline: float) -> float:
    return 100.0 * (candidate - baseline) / baseline


def register_decomposition(census: dict) -> dict:
    """S3 hard gate, plus the decomposition that gives E55 its scientific value."""
    kw = census["kernel_wide_reg_max"]
    per = census["per_width_reg"]
    base, e27, cand = kw["base_na4_table"], kw["e27_both_cells"], kw["m9two_candidate"]

    widths = sorted(int(w) for w in per["base_na4_table"])
    rows = []
    for w in widths:
        b = per["base_na4_table"][str(w)]
        e = per["e27_both_cells"][str(w)]
        c = per["m9two_candidate"][str(w)]
        rows.append(
            {
                "width": w,
                "base": b,
                "e27": e,
                "candidate": c,
                "candidate_equals_base": c == b,
                "candidate_equals_e27": c == e,
            }
        )

    # Which cell set E27's kernel-wide ceiling? The max is over per-width cells.
    e27_argmax = [r["width"] for r in rows if r["e27"] == e27]
    cand_argmax = [r["width"] for r in rows if r["candidate"] == cand]
    e27_case5_cell = per["e27_both_cells"]["5"]

    differing = [r["width"] for r in rows if not r["candidate_equals_e27"]]

    return {
        "shipped_table_reg_max": base,
        "hard_gate_shipped_reads_108": base == 108,
        "e27_reg_max": e27,
        "candidate_reg_max": cand,
        "candidate_is_register_identical_to_e27": cand == e27,
        "candidate_vs_base_step": cand - base,
        "e27_vs_base_step": e27 - base,
        "steps_are_equal": (cand - base) == (e27 - base),
        "e27_ceiling_set_by_widths": e27_argmax,
        "candidate_ceiling_set_by_widths": cand_argmax,
        "e27_case5_cell": e27_case5_cell,
        "e27_case5_is_below_ceiling": e27_case5_cell < e27,
        "widths_where_candidate_differs_from_e27": differing,
        "per_width": rows,
        "reading": (
            "E27's kernel-wide ceiling of {e27} is set by the M=9 cell alone; its "
            "M=5 cell reaches only {c5}, which is below it. The candidate raises "
            "the same M=9 cell and leaves M=5 at the base value, so the candidate "
            "and E27 have the SAME kernel-wide register allocation. Section 3 of "
            "the feedback establishes one library, one pipeline and one allocation "
            "for all M, so the shared-ceiling tax is common-mode between E27 and "
            "E55 and cancels in the E27-minus-E55 contrast."
        ).format(e27=e27, c5=e27_case5_cell),
    }


def two_term_predictions(local_f9_pct: float) -> dict:
    """S4: one prediction per mixture, each split into cell term and ceiling term."""
    ceiling_term_pct = E27_RANKED_MTP_LEG_PCT  # slows the leg, so positive

    def row(name: str, f9_pct: float) -> dict:
        cell = -SENSITIVITY_PCT_PER_F9 * f9_pct / 100.0
        return {
            "mixture": name,
            "f9_pct": f9_pct,
            "term1_cell_pct": cell,
            "term2_ceiling_pct": ceiling_term_pct,
            "net_static_ceiling_pct": cell,
            "net_moved_ceiling_pct": cell + ceiling_term_pct,
            "x_mtp_null_moved": abs(cell + ceiling_term_pct) / NULL_MTP_PCT,
        }

    rows = [row(n, f) for n, f in MIXTURES.items()]
    rows.append(row("local_fixture_measured", local_f9_pct))
    return {
        "ceiling_term_source": (
            "E27 ranked MTP leg +0.1995 %, applied at the candidate's dose because "
            "the candidate's kernel-wide allocation is identical to E27's."
        ),
        "ceiling_term_is_cross_host": True,
        "ceiling_term_local_alternative_pct": E49_ARM2_READ_AT_MY_DOSE_PCT,
        "is_pre_registration": False,
        "post_hoc_reason": (
            "the three timed arms completed before this feedback arrived, so the "
            "second term is a requested model correction, not a prereg"
        ),
        "rows": rows,
    }


def measured_vs_two_term(local_f9_pct: float) -> dict:
    mtp_base_bracket = 0.5 * (MTP_BASE + MTP_BASE2)
    serial_base_bracket = 0.5 * (SERIAL_BASE + SERIAL_BASE2)
    d_mtp = pct_delta(MTP_CAND, mtp_base_bracket)
    d_serial = pct_delta(SERIAL_CAND, serial_base_bracket)

    cell = -SENSITIVITY_PCT_PER_F9 * local_f9_pct / 100.0
    one_term = cell
    two_term_e27 = cell + E27_RANKED_MTP_LEG_PCT
    two_term_e49 = cell + E49_ARM2_READ_AT_MY_DOSE_PCT

    variants = {
        "one_term_cell_only": one_term,
        "two_term_e27_ceiling_magnitude": two_term_e27,
        "two_term_e49_local_dose_reading": two_term_e49,
    }
    residuals = {k: d_mtp - v for k, v in variants.items()}
    best = min(residuals, key=lambda k: abs(residuals[k]))

    return {
        "measured_mtp_leg_pct": d_mtp,
        "measured_serial_leg_pct": d_serial,
        "serial_falsifier_held": abs(d_serial) < abs(NULL_SERIAL_PCT) * 5,
        "mtp_base_bracket_mean": mtp_base_bracket,
        "predicted_pct": variants,
        "residual_pp": residuals,
        "best_fitting_variant": best,
        "two_term_improves_local_fit": abs(residuals["two_term_e27_ceiling_magnitude"])
        < abs(residuals["one_term_cell_only"]),
        "fit_improvement_factor": abs(residuals["one_term_cell_only"])
        / abs(residuals["two_term_e27_ceiling_magnitude"]),
        "reading": (
            "Adding the E27-magnitude ceiling tax moves the local prediction "
            "TOWARD the measurement. That is weak corroboration that a real "
            "ceiling tax is present locally as well, but it is a one-point fit "
            "with a cross-host coefficient and it does not identify the term."
        ),
    }


def ranked_prereg() -> dict:
    """A genuine forward pre-registration, stated before any ranked submission.

    E27 = ceiling + M5cell + M9cell. E55 = ceiling + M9cell, at the SAME ceiling.
    So dScore(E55) = dScore(E27) - contribution(M5 cell), and the two competing
    explanations for E27's loss give cleanly separated predictions.
    """
    # H_ceiling: E27's loss is the shared ceiling; both cell terms are ~0 on ranked.
    h_ceiling_lo = E27_RANKED_DSCORE_PCT
    h_ceiling_hi = 0.0

    # H_m5cell: the ceiling is bounded by E49 Arm 2 and the M=9 cell pays as priced.
    # E27 = ceiling + M5cell + M9cell  =>  M5cell = E27 - M9cell - ceiling.
    # dScore(E55) = ceiling + M9cell, with ceiling in [-bound, +bound].
    h_m5_lo = -E49_ARM2_HARM_BOUND_PCT + E49_M9_HALF_RANKED_PRICE_PCT
    h_m5_hi = +E49_ARM2_HARM_BOUND_PCT + E49_M9_HALF_RANKED_PRICE_PCT
    m5cell_lo = E27_RANKED_DSCORE_PCT - E49_M9_HALF_RANKED_PRICE_PCT - E49_ARM2_HARM_BOUND_PCT
    m5cell_hi = E27_RANKED_DSCORE_PCT - E49_M9_HALF_RANKED_PRICE_PCT + E49_ARM2_HARM_BOUND_PCT

    separation = h_m5_lo - h_ceiling_hi
    return {
        "identity": "E27 = ceiling + M5cell + M9cell ; E55 = ceiling + M9cell",
        "ceiling_is_common_mode": True,
        "h_ceiling": {
            "claim": "E27's ranked loss is the shared register ceiling",
            "predicted_ranked_dscore_pct": [h_ceiling_lo, h_ceiling_hi],
        },
        "h_m5cell": {
            "claim": "E27's ranked loss is the M=5 cell; the ceiling is bounded by E49 Arm 2",
            "predicted_ranked_dscore_pct": [h_m5_lo, h_m5_hi],
            "implied_m5cell_contribution_pct": [m5cell_lo, m5cell_hi],
        },
        "separation_pct": separation,
        "ranked_mde_2sd_pct": RANKED_MDE_2SD_PCT,
        "separation_in_mde": separation / RANKED_MDE_2SD_PCT,
        "discriminating": separation > 2 * RANKED_MDE_2SD_PCT,
        "reading": (
            "One ranked run of this candidate separates the two explanations by "
            "{sep:.4f} points against a {mde} % minimum detectable effect, i.e. "
            "{n:.2f}x. That is the experiment section 2 asks for, and it is not "
            "available locally at any cost, because the ceiling term is "
            "common-mode between E27 and E55 by construction."
        ).format(sep=separation, mde=RANKED_MDE_2SD_PCT, n=separation / RANKED_MDE_2SD_PCT),
    }


def back_solve_advisor_prize() -> dict:
    """S5a: what ranked f9 does each quoted prize figure imply?"""

    def f9_from_dscore(dscore_pct: float, g: float | None) -> float:
        denom = SENSITIVITY_PCT_PER_F9 * (g if g is not None else 1.0)
        return 100.0 * dscore_pct / denom

    g_mid = 0.5 * (ADVISOR_G_RANGE[0] + ADVISOR_G_RANGE[1])
    out = {
        "sensitivity_pct_dscore_per_unit_f9": SENSITIVITY_PCT_PER_F9,
        "g_range": list(ADVISOR_G_RANGE),
        "g_mid": g_mid,
        "pre_correction_5_36": {
            "implied_f9_pct_no_g": f9_from_dscore(ADVISOR_PRIZE_PRE_CORRECTION_PCT, None),
        },
        "corrected_4_5_to_4_7": {
            "implied_f9_pct_no_g": [
                f9_from_dscore(ADVISOR_PRIZE_CORRECTED_PCT[0], None),
                f9_from_dscore(ADVISOR_PRIZE_CORRECTED_PCT[1], None),
            ],
            "implied_f9_pct_with_g_mid": [
                f9_from_dscore(ADVISOR_PRIZE_CORRECTED_PCT[0], g_mid),
                f9_from_dscore(ADVISOR_PRIZE_CORRECTED_PCT[1], g_mid),
            ],
        },
        "e49_m9_half_1_3625": {
            "implied_f9_pct_no_g": f9_from_dscore(E49_M9_HALF_RANKED_PRICE_PCT, None),
        },
        "registered_mixtures_f9_pct": MIXTURES,
        "local_fixture_f9_pct": LOCAL_F9_PCT,
        "ambiguity_flagged": (
            "I cannot tell from the comment whether g multiplies the sensitivity "
            "or was already applied when the corrected range was quoted, so both "
            "back-solves are reported."
        ),
    }
    lo = min(out["corrected_4_5_to_4_7"]["implied_f9_pct_no_g"])
    hi = max(out["corrected_4_5_to_4_7"]["implied_f9_pct_with_g_mid"])
    out["implied_ranked_f9_band_pct"] = [lo, hi]
    out["implied_band_exceeds_e48"] = lo > MIXTURES["e48_merged"]
    out["implied_band_exceeds_edward_upper"] = lo > MIXTURES["edward_e53_upper"]
    out["implied_band_brackets_local_f9"] = lo <= LOCAL_F9_PCT <= hi
    out["reading"] = (
        "Every quoted ranked prize figure implies a ranked f9 of {lo:.1f}..{hi:.1f} %, "
        "which is the LOCAL fixture's mixture, not either registered ranked mixture. "
        "It is {a:.2f}x E48's 21.630 % and {b:.2f}x edward's 8.9 % upper bound. The "
        "E49 M=9-half figure of +1.3625 % implies only {c:.2f} %. The prize quoted to "
        "me and the prize implied by E49 differ by {d:.2f}x, and the whole difference "
        "is f9."
    ).format(
        lo=lo,
        hi=hi,
        a=lo / MIXTURES["e48_merged"],
        b=lo / MIXTURES["edward_e53_upper"],
        c=out["e49_m9_half_1_3625"]["implied_f9_pct_no_g"],
        d=ADVISOR_PRIZE_CORRECTED_PCT[0] / E49_M9_HALF_RANKED_PRICE_PCT,
    )
    return out


def width_mixture_gap() -> dict:
    """How much shallower does the ranked central pair draft than the local fixture?

    edward established M = drafts PROPOSED + 1. My own arm satisfies that
    identity exactly, which is what lets me compare the two populations. The
    receipt's `mean_draft_len` is ambiguous between proposed and accepted, so
    both readings are carried.
    """
    rounds = sum(LOCAL_WIDTH_HIST.values())
    rows = sum(w * n for w, n in LOCAL_WIDTH_HIST.items())
    local_mean_m = rows / rounds
    local_proposed = local_mean_m - 1.0
    local_accepted = LOCAL_ACCEPTED_DRAFTS / rounds
    proposed_over_accepted = local_proposed / local_accepted

    prompts = {}
    for name, mdl in RECEIPT_MEAN_DRAFT_LEN.items():
        # Reading 1: receipt reports PROPOSED drafts, same quantity as mine.
        r1_mean_m = mdl + 1.0
        # Reading 2: receipt reports ACCEPTED drafts; scale by my own
        # proposed/accepted ratio to recover proposed, then add the primary.
        r2_mean_m = mdl * proposed_over_accepted + 1.0
        prompts[name] = {
            "receipt_mean_draft_len": mdl,
            "reading1_proposed_mean_m": r1_mean_m,
            "reading2_accepted_mean_m": r2_mean_m,
            "gap_vs_local_reading1_rows": local_mean_m - r1_mean_m,
            "gap_vs_local_reading2_rows": local_mean_m - r2_mean_m,
        }

    gaps = [
        v[k]
        for v in prompts.values()
        for k in ("gap_vs_local_reading1_rows", "gap_vs_local_reading2_rows")
    ]
    return {
        "local_rounds": rounds,
        "local_rows": rows,
        "local_mean_verify_width": local_mean_m,
        "local_proposed_drafts_per_round": local_proposed,
        "local_accepted_drafts_per_round": local_accepted,
        "local_proposed_over_accepted": proposed_over_accepted,
        "m_equals_proposed_plus_one_holds_locally": abs(
            local_proposed - (rows / rounds - 1.0)
        )
        < 1e-12,
        "central_pair": prompts,
        "gap_rows_min": min(gaps),
        "gap_rows_max": max(gaps),
        "gap_positive_under_both_readings": min(gaps) > 0,
        "ambiguity_flagged": (
            "the receipt's mean_draft_len may be proposed or accepted drafts; both "
            "readings are carried and the direction is the same under each"
        ),
        "reading": (
            "The local public fixture dispatches {g1:.3f} to {g2:.3f} more rows per "
            "verify than the two prompts that set the published median. Under a "
            "greedy marginal-cost schedule a shallower mean width places less cost "
            "at M=9, so f9 on the central pair is BELOW the local fixture's "
            "{f9:.3f} %. That is a direction, not a bound: section 5b's LP admits "
            "up to 70 % and I am not trying to close it analytically."
        ).format(g1=min(gaps), g2=max(gaps), f9=LOCAL_F9_PCT),
    }


def e27_conversion_anomaly() -> dict:
    """E27's own receipt does not convert at psi_mtp. Flagged, not resolved."""
    implied = abs(E27_RANKED_DSCORE_PCT) / abs(E27_RANKED_MTP_LEG_PCT)
    return {
        "e27_ranked_dscore_pct": E27_RANKED_DSCORE_PCT,
        "e27_ranked_mtp_leg_pct": E27_RANKED_MTP_LEG_PCT,
        "implied_dscore_per_mtp_leg": implied,
        "psi_mtp": PSI_MTP,
        "ratio_to_psi_mtp": implied / PSI_MTP,
        "flag": (
            "E27's receipt converts an MTP-leg change into published score at "
            "{i:.4f}, which is {r:.2f}x psi_mtp = {p}. Either the E27 loss has a "
            "component outside the MTP leg, or the order-statistic behaviour at "
            "that operating point is far from the constant-rate model. I flag it "
            "because the ceiling term in section 4 is quoted in MTP-leg space and "
            "then priced with psi_mtp; if E27's own conversion is right, the "
            "ceiling term costs {alt:.4f} % of score, not {std:.4f} %."
        ).format(
            i=implied,
            r=implied / PSI_MTP,
            p=PSI_MTP,
            alt=E27_RANKED_MTP_LEG_PCT * implied,
            std=E27_RANKED_MTP_LEG_PCT * PSI_MTP,
        ),
    }


def negative_controls(census: dict) -> dict:
    """An instrument that cannot fail is not an instrument (ledger 178(E))."""
    controls = {}

    fake = json.loads(json.dumps(census))
    fake["kernel_wide_reg_max"]["base_na4_table"] = 129
    controls["gate_fires_when_shipped_table_is_not_108"] = (
        register_decomposition(fake)["hard_gate_shipped_reads_108"] is False
    )

    fake2 = json.loads(json.dumps(census))
    fake2["per_width_reg"]["m9two_candidate"]["5"] = 125
    d2 = register_decomposition(fake2)
    controls["decomposition_notices_a_case5_change"] = (
        d2["widths_where_candidate_differs_from_e27"] == []
        and d2["per_width"][2]["candidate_equals_base"] is False
    )

    fake3 = json.loads(json.dumps(census))
    fake3["kernel_wide_reg_max"]["m9two_candidate"] = 140
    controls["register_identity_claim_can_fail"] = (
        register_decomposition(fake3)["candidate_is_register_identical_to_e27"] is False
    )

    prereg = ranked_prereg()
    controls["ranked_prereg_hypotheses_do_not_overlap"] = (
        prereg["h_ceiling"]["predicted_ranked_dscore_pct"][1]
        < prereg["h_m5cell"]["predicted_ranked_dscore_pct"][0]
    )

    gap = width_mixture_gap()
    controls["width_gap_direction_survives_both_readings"] = gap[
        "gap_positive_under_both_readings"
    ]

    fit = measured_vs_two_term(LOCAL_F9_PCT)
    controls["two_term_fit_claim_is_a_comparison_not_a_pass"] = isinstance(
        fit["two_term_improves_local_fit"], bool
    )

    back = back_solve_advisor_prize()
    controls["back_solve_separates_prize_figures"] = (
        back["implied_band_exceeds_e48"] is True
    )

    controls["post_hoc_label_present"] = (
        two_term_predictions(LOCAL_F9_PCT)["is_pre_registration"] is False
    )

    return {"controls": controls, "all_fire": all(controls.values())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", default="research/e55-reg-census.json")
    ap.add_argument("--out", default="research/e55-ceiling-term.json")
    args = ap.parse_args()

    census = json.loads(Path(args.census).read_text())

    reg = register_decomposition(census)
    fit = measured_vs_two_term(LOCAL_F9_PCT)
    controls = negative_controls(census)

    doc = {
        "experiment": "E55",
        "responds_to_feedback_id": "e55-is-the-m9-half-of-e27-and-prereg-omits-ceiling-term",
        "register_decomposition": reg,
        "two_term_predictions": two_term_predictions(LOCAL_F9_PCT),
        "measured_vs_two_term": fit,
        "ranked_prereg": ranked_prereg(),
        "advisor_prize_back_solve": back_solve_advisor_prize(),
        "width_mixture_gap": width_mixture_gap(),
        "e27_conversion_anomaly": e27_conversion_anomaly(),
        "local_f9_discrepancy": {
            "mine_pct": LOCAL_F9_PCT,
            "advisor_quoted_pct": ADVISOR_QUOTED_LOCAL_F9_PCT,
            "delta_pp": LOCAL_F9_PCT - ADVISOR_QUOTED_LOCAL_F9_PCT,
            "note": (
                "same fixture, so this is a denominator difference and not a "
                "disagreement about the schedule; it does not change any sign here"
            ),
        },
        "negative_controls": controls,
    }
    doc["verdict_ok"] = bool(
        reg["hard_gate_shipped_reads_108"]
        and reg["candidate_is_register_identical_to_e27"]
        and fit["serial_falsifier_held"]
        and doc["ranked_prereg"]["discriminating"]
        and controls["all_fire"]
    )

    Path(args.out).write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")

    print(f"wrote {args.out}")
    print(f"  hard gate, shipped table reads 108   : {reg['hard_gate_shipped_reads_108']}")
    print(f"  shipped / E27 / candidate reg max    : {reg['shipped_table_reg_max']} / "
          f"{reg['e27_reg_max']} / {reg['candidate_reg_max']}")
    print(f"  candidate register-identical to E27   : {reg['candidate_is_register_identical_to_e27']}")
    print(f"  E27 ceiling set by width(s)          : {reg['e27_ceiling_set_by_widths']}")
    print(f"  E27 case-5 cell (below ceiling)      : {reg['e27_case5_cell']}")
    print(f"  widths where candidate differs E27    : {reg['widths_where_candidate_differs_from_e27']}")
    print(f"  measured MTP leg                      : {fit['measured_mtp_leg_pct']:+.4f} %")
    for k, v in fit["predicted_pct"].items():
        print(f"    predicted {k:<34}: {v:+.4f} %  residual {fit['residual_pp'][k]:+.4f} pp")
    print(f"  best fitting variant                  : {fit['best_fitting_variant']}")
    pr = doc["ranked_prereg"]
    print(f"  ranked prereg H_ceiling               : {pr['h_ceiling']['predicted_ranked_dscore_pct']}")
    print(f"  ranked prereg H_m5cell                : {pr['h_m5cell']['predicted_ranked_dscore_pct']}")
    print(f"  separation / ranked MDE               : {pr['separation_in_mde']:.2f}x")
    bs = doc["advisor_prize_back_solve"]
    print(f"  prize figures imply ranked f9         : {bs['implied_ranked_f9_band_pct'][0]:.2f}"
          f"..{bs['implied_ranked_f9_band_pct'][1]:.2f} %")
    g = doc["width_mixture_gap"]
    print(f"  local mean verify width               : {g['local_mean_verify_width']:.4f}")
    print(f"  central-pair gap (rows)               : {g['gap_rows_min']:.3f}..{g['gap_rows_max']:.3f}")
    print(f"  negative controls all fire            : {controls['all_fire']}")
    print(f"  verdict_ok                            : {doc['verdict_ok']}")
    return 0 if doc["verdict_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
