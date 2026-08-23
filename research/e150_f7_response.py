"""Answer advisor F7 sections 2 and 3.

Section 2 reprices the E150 A0 width-6 admissibility boundary on F7's clean
input. F7 retracts the `c47b45be` reading (+1.1455 %) used by F6 and by
`research/e150_a0_transfer.py`. The clean receipt for the identical two-integer
source edit is `24fb4012`, which reads -0.0390 % +/- 0.0799. Only the
ranked-percent input changes, so this reprices the same machinery.

Section 2 also identifies where F7's quoted width-6 and width-5 round costs
come from. They are the `us` basis of `research/e145-artifacts/curve.json`,
while the compiled Swift curve and every E150 price read the
`us_mean_from_blocks` basis. That is harness defect (f). The boundary is
recomputed on both bases to show the conclusion does not depend on the choice.

Section 3 pre-registers the run-level nuisance-tail fields that F7 requires
before the receipt for submission 749da2cf lands, and implements the exposure
regression so the fit can be run on the receipt without further design freedom.

harness=local, gpu_used=false.
"""

from __future__ import annotations

import json
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import e150_lib  # noqa: E402

VOID_ROW = "c47b45be"
CLEAN_ROW = "24fb4012"
CLEAN_PCT = -0.0390
CLEAN_SE = 0.0799

# F7 quotes these two curve values while doing the transfer arithmetic. They are
# not the values in this experiment's curve, so record both and flag the delta.
F7_QUOTED_W6_US = 124436.0
F7_QUOTED_W5_US = 95301.5

# F7 section 2: the rival's isolated microbenchmark of both M=6 partitions on all
# seven real verify-leg shapes at real (N, K).
ISOLATED_M6_IPG3_X2_US = 137111.0
ISOLATED_M6_IPG6_X1_US = 125723.0

# F7 section 3: one state step, in seconds.
STATE_STEP_US = 903.0
DECODE_TOKENS = 512

BOARD = Path("/tmp/yukon-board/full.json")
MY_SUBMISSION = "749da2cf"
ANCHOR_SUBMISSION = "0cf1637e"
BAR_SUBMISSION = "ec24d591"

# Pre-registered before the receipt lands. See the report text for provenance.
PREREG = {
    "e150_expected_gain_pp": 1.2716,
    "e150_expected_gain_pp_conservative_floor": 0.5737,
    "e150_expected_gain_pp_advisor_min_over_curves": 0.8543,
    "e150_expected_ranked_score": 3.729619558407483,
    "e150_required_uniform_candidate_leg_gain_pct_vs_bar": 1.0042,
    "e150_nuisance_tail_acknowledged_pp": 1.10,
    "e150_receipt_is_underpowered_for_this_effect": True,
}


def recover_round_count(edl: float, tokens: int = DECODE_TOKENS) -> int | None:
    """Recover the round count from a published effective mean draft length.

    thorfinn's E135 method. `edl` is drafted tokens per round, so
    `Fraction(edl).limit_denominator(600)` fixes the round count up to an
    integer multiple. Emitted tokens obey `rounds + accepted = tokens`, and
    accepted cannot exceed drafted, which prunes the small multiples. The
    smallest surviving multiple is canonical.
    """
    if edl <= 0:
        return None
    frac = Fraction(edl).limit_denominator(600)
    base = frac.denominator
    rounds = base
    while rounds <= tokens:
        accepted = tokens - rounds
        drafted = frac * rounds
        if 0 <= accepted <= drafted:
            return rounds
        rounds += base
    return None


def exposure(row: dict) -> dict:
    """F7's exposure shape for one published per-prompt receipt row."""
    edl = row["effective_mean_draft_len"]
    rounds = recover_round_count(edl)
    non_drafting = row["non_drafting_round_count"]
    decode_seconds = row["mtp_seconds_per_token_mean"] * DECODE_TOKENS
    drafting_rounds = None if rounds is None else rounds - non_drafting
    s_p = (
        None
        if drafting_rounds is None
        else (STATE_STEP_US * 1e-6) * drafting_rounds / decode_seconds
    )
    return {
        "prompt_sha256": row["prompt_sha256"][:8],
        "effective_mean_draft_len": edl,
        "recovered_round_count": rounds,
        "non_drafting_round_count": non_drafting,
        "drafting_round_count": drafting_rounds,
        "decode_seconds": decode_seconds,
        "mtp_seconds_per_token_mean": row["mtp_seconds_per_token_mean"],
        "s_p": s_p,
    }


def fit_k(contrast_pct: list[float], s: list[float]) -> dict:
    """Least squares through the origin of contrast_pct on 100 * s_p."""
    x = [100.0 * v for v in s]
    sxx = sum(v * v for v in x)
    sxy = sum(a * b for a, b in zip(x, contrast_pct))
    k = sxy / sxx if sxx else 0.0
    resid = [c - k * v for c, v in zip(contrast_pct, x)]
    mean = sum(contrast_pct) / len(contrast_pct)
    sst = sum((c - mean) ** 2 for c in contrast_pct)
    sse = sum(r * r for r in resid)
    return {
        "k_state_steps": k,
        "us_per_drafting_round": k * STATE_STEP_US,
        "r_squared": (1.0 - sse / sst) if sst else None,
        "residual_mean_pp": sum(resid) / len(resid),
        "residuals_pp": resid,
    }


def board_rows() -> dict:
    if not BOARD.exists():
        return {}
    raw = json.loads(BOARD.read_text())
    rows = raw["submissions"] if isinstance(raw, dict) else raw
    return {r["id"][:8]: r for r in rows}


def exposure_table(sub_id: str, rows: dict) -> dict | None:
    row = rows.get(sub_id)
    if row is None:
        return None
    metrics = row.get("officialMetrics") or {}
    per_prompt = metrics.get("per_prompt")
    if not per_prompt:
        return None
    return {
        "submission_id": row["id"],
        "official_score": row.get("officialScore"),
        "status": row.get("status"),
        "per_prompt": [exposure(p) for p in per_prompt],
    }


def nuisance_pair(a_id: str, b_id: str, rows: dict, f7_k: float, f7_r2: float) -> dict | None:
    """Measure one run-level nuisance draw between two receipts.

    The candidate-leg contrast is model free. The exposure regression is F7's
    attribution of that contrast to drafting rounds, and is only meaningful
    when both receipts ran the same schedule.
    """
    ea = exposure_table(a_id, rows)
    eb = exposure_table(b_id, rows)
    if ea is None or eb is None:
        return None
    pa = {r["prompt_sha256"]: r for r in ea["per_prompt"]}
    pb = {r["prompt_sha256"]: r for r in eb["per_prompt"]}
    keys = sorted(pa)
    schedule_identical = all(
        pa[k]["effective_mean_draft_len"] == pb[k]["effective_mean_draft_len"]
        and pa[k]["recovered_round_count"] == pb[k]["recovered_round_count"]
        and pa[k]["non_drafting_round_count"] == pb[k]["non_drafting_round_count"]
        for k in keys
    )
    contrast = [
        100.0 * (pa[k]["mtp_seconds_per_token_mean"] - pb[k]["mtp_seconds_per_token_mean"])
        / pb[k]["mtp_seconds_per_token_mean"]
        for k in keys
    ]
    fit = fit_k(contrast, [pb[k]["s_p"] for k in keys])
    ordered = sorted(contrast)
    drafting = [c for c, k in zip(contrast, keys) if pb[k]["non_drafting_round_count"] == 0]
    mostly_non_drafting = [
        c
        for c, k in zip(contrast, keys)
        if pb[k]["non_drafting_round_count"] > 0.5 * (pb[k]["recovered_round_count"] or 1)
    ]
    return {
        "pair": f"{a_id} vs {b_id}",
        "schedule_identical": schedule_identical,
        "official_score_a": ea["official_score"],
        "official_score_b": eb["official_score"],
        "per_prompt": {
            pa[k]["prompt_sha256"]: {
                "contrast_pct": c,
                "s_p": pb[k]["s_p"],
                "drafting_round_count": pb[k]["drafting_round_count"],
            }
            for k, c in zip(keys, contrast)
        },
        "contrast_mean_pp": sum(contrast) / len(contrast),
        "contrast_median_pp": 0.5 * (ordered[3] + ordered[4]),
        "contrast_min_pp": min(contrast),
        "contrast_max_pp": max(contrast),
        "drafting_prompt_min_pp": min(drafting) if drafting else None,
        "drafting_prompt_max_pp": max(drafting) if drafting else None,
        "mostly_non_drafting_pp": mostly_non_drafting,
        "fit": fit,
        "f7_reported_k_state_steps": f7_k,
        "f7_reported_r_squared": f7_r2,
        "k_ratio_mine_over_f7": abs(fit["k_state_steps"]) / f7_k,
        "r_squared_reproduced": abs(fit["r_squared"] - f7_r2) < 0.05,
    }


def band(void_pct: float, void_rung_us: float, c6_us: float, c5_us: float, pct: float) -> dict:
    rung = void_rung_us * (pct / void_pct)
    post = c6_us - rung
    return {
        "ranked_pct": pct,
        "rung_local_us": rung,
        "c6_post_us": post,
        "c6_post_per_token_us": post / 6.0,
        "admissible": bool(post / 6.0 < c5_us / 5.0),
    }


def main() -> int:
    art = json.loads((e150_lib.ARTIFACTS / "a0_transfer.json").read_text())
    b = art["e150_a0_admissibility_boundary"]
    pre = art["e150_a0_curve_pre"]

    void_pct = b["measured_ranked_pct"]
    void_rung = b["point_estimate_local_us_removed"]
    step_us = b["step_us_used_as_divisor"]
    required = b["required_local_us_to_admit_width6"]
    breakeven_pct = b["breakeven_ranked_pct"]
    c6 = pre["round_us"]["6"]
    c5 = pre["round_us"]["5"]

    point = band(void_pct, void_rung, c6, c5, CLEAN_PCT)
    lo = band(void_pct, void_rung, c6, c5, CLEAN_PCT - CLEAN_SE)
    hi = band(void_pct, void_rung, c6, c5, CLEAN_PCT + CLEAN_SE)

    sigma = (breakeven_pct - CLEAN_PCT) / CLEAN_SE

    # F7's quoted round costs are the `us` basis of the same E145 R2 curve, not
    # the `us_mean_from_blocks` basis this experiment compiles. Confirm that,
    # then redo the boundary entirely inside the `us` basis. The transfer
    # constant k was itself fitted on the `us` basis, so that frame is the
    # internally consistent one and is the stronger test of the conclusion.
    e145 = json.loads(Path("research/e145-artifacts/curve.json").read_text())["measured"]
    us_c6 = e145["6"]["us"]
    us_c5 = e145["5"]["us"]
    basis_match = (
        abs(us_c6 - F7_QUOTED_W6_US) < 0.05 and abs(us_c5 - F7_QUOTED_W5_US) < 0.05
    )
    us_required = us_c6 - 6.0 * us_c5 / 5.0
    us_rung_per_pct = void_rung / void_pct
    us_breakeven_pct = us_required / us_rung_per_pct
    us_rung_clean = us_rung_per_pct * CLEAN_PCT
    us_basis = {
        "width6_us": us_c6,
        "width5_us": us_c5,
        "required_local_us_to_admit_width6": us_required,
        "breakeven_ranked_pct": us_breakeven_pct,
        "sigma_to_boundary": (us_breakeven_pct - CLEAN_PCT) / CLEAN_SE,
        "point_estimate_local_us_removed": us_rung_clean,
        "shortfall_local_us": us_required - us_rung_clean,
        "c6_post_per_token_us": (us_c6 - us_rung_clean) / 6.0,
        "width5_per_token_us": us_c5 / 5.0,
        "admissible": bool((us_c6 - us_rung_clean) / 6.0 < us_c5 / 5.0),
    }

    payload = {
        "e150_a0_f7_harness": "local",
        "e150_a0_f7_gpu_used": False,
        "e150_a0_f7_void_row": VOID_ROW,
        "e150_a0_f7_clean_row": CLEAN_ROW,
        "e150_a0_f7_void_ranked_pct": void_pct,
        "e150_a0_f7_clean_ranked_pct": CLEAN_PCT,
        "e150_a0_f7_clean_ranked_se": CLEAN_SE,
        "e150_a0_f7_rows_disagree_pp": void_pct - CLEAN_PCT,
        "e150_a0_f7_breakeven_ranked_pct": breakeven_pct,
        "e150_a0_f7_sigma_to_boundary": sigma,
        "e150_a0_f7_sigma_to_boundary_on_void_input": b["sigma_to_boundary"],
        "e150_a0_f7_width6_question_resolved": bool(sigma >= 3.0),
        "e150_a0_f7_width6_admissible": point["admissible"],
        "e150_a0_f7_required_local_us_to_admit_width6": required,
        "e150_a0_f7_point_estimate_local_us_removed": point["rung_local_us"],
        "e150_a0_f7_shortfall_local_us": required - point["rung_local_us"],
        "e150_a0_f7_share_of_cliff": point["rung_local_us"] / step_us,
        "e150_a0_f7_share_of_cliff_on_void_input": void_rung / step_us,
        "e150_a0_f7_band": {"minus_1se": lo, "point": point, "plus_1se": hi},
        "e150_a0_f7_curve_value_flag": {
            "f7_quoted_width6_us": F7_QUOTED_W6_US,
            "e150_width6_us": c6,
            "width6_delta_us": F7_QUOTED_W6_US - c6,
            "width6_delta_pct": 100.0 * (F7_QUOTED_W6_US - c6) / c6,
            "f7_quoted_width5_us": F7_QUOTED_W5_US,
            "e150_width5_us": c5,
            "width5_delta_us": F7_QUOTED_W5_US - c5,
            "width5_delta_pct": 100.0 * (F7_QUOTED_W5_US - c5) / c5,
            "curves_match": False,
            "f7_basis": "us",
            "e150_basis": "us_mean_from_blocks",
            "f7_values_are_exactly_the_us_basis": basis_match,
            "root_cause": "harness defect (f)",
        },
        "e150_a0_f7_boundary_on_us_basis": us_basis,
        "e150_a0_f7_isolated_m6": {
            "ipg3_x2_us": ISOLATED_M6_IPG3_X2_US,
            "ipg6_x1_us": ISOLATED_M6_IPG6_X1_US,
            "isolated_one_pass_saving_pct": 100.0
            * (ISOLATED_M6_IPG3_X2_US - ISOLATED_M6_IPG6_X1_US)
            / ISOLATED_M6_IPG3_X2_US,
            "in_situ_ranked_pct": CLEAN_PCT,
            "transfer_ratio": CLEAN_PCT
            / (
                100.0
                * (ISOLATED_M6_IPG3_X2_US - ISOLATED_M6_IPG6_X1_US)
                / ISOLATED_M6_IPG3_X2_US
            ),
        },
        "e150_a0_f7_admissible_widths_unchanged": [1, 2, 3, 4, 5],
    }

    rows = board_rows()
    my_receipt = exposure_table(MY_SUBMISSION, rows)
    prereg = dict(PREREG)
    prereg.update(
        {
            "e150_nuisance_state_step_us": STATE_STEP_US,
            "e150_nuisance_exposure_shape": (
                "s_p = 903 us * drafting_rounds_p / decode_seconds_p; "
                "contrast_pct_p = 100 * k * s_p; k in state steps; "
                "k * 903 = us per drafting round"
            ),
            "e150_nuisance_round_recovery": (
                "rounds_p = smallest multiple of "
                "denominator(Fraction(effective_mean_draft_len).limit_denominator(600)) "
                "with rounds <= 512 and 0 <= 512 - rounds <= drafted"
            ),
            "e150_nuisance_sp_must_come_from_own_receipt": True,
            "e150_nuisance_sp_confounded_by_schedule_change": True,
            "e150_nuisance_null_pair": nuisance_pair(
                "c47b45be", "24fb4012", rows, f7_k=0.611, f7_r2=0.938
            ),
            "e150_nuisance_second_pair": nuisance_pair(
                "43925f29", "a9dd132a", rows, f7_k=1.074, f7_r2=0.907
            ),
            "e150_nuisance_anchor_submission": ANCHOR_SUBMISSION,
            "e150_nuisance_bar_submission": BAR_SUBMISSION,
            "e150_nuisance_my_submission": MY_SUBMISSION,
            "e150_nuisance_my_receipt_available": my_receipt is not None,
            "e150_anchor_exposure": exposure_table(ANCHOR_SUBMISSION, rows),
            "e150_bar_exposure": exposure_table(BAR_SUBMISSION, rows),
            "e150_my_exposure": my_receipt,
        }
    )
    pairs = [
        p
        for p in (prereg["e150_nuisance_null_pair"], prereg["e150_nuisance_second_pair"])
        if p
    ]
    prereg.update(
        {
            "e150_nuisance_tail_observed_max_pp": max(
                abs(p["contrast_mean_pp"]) for p in pairs
            ),
            "e150_nuisance_tail_observed_max_single_prompt_pp": max(
                max(abs(p["contrast_min_pp"]), abs(p["contrast_max_pp"])) for p in pairs
            ),
            "e150_nuisance_tail_observed_exceeds_f7_value": bool(
                max(abs(p["contrast_mean_pp"]) for p in pairs)
                > PREREG["e150_nuisance_tail_acknowledged_pp"]
            ),
            # Both pairs share an intended mechanism but not a commit, so each
            # contrast is nuisance plus any real implementation difference.
            "e150_nuisance_pairs_are_byte_replicates": False,
            # A seed-prefill or verify-width mechanism would move every prompt.
            # The near-zero move on the mostly non-drafting prompt rules that
            # out and leaves drafting-round exposure as the carrier.
            "e150_nuisance_carrier_is_drafting_rounds": all(
                all(abs(c) < 0.10 for c in p["mostly_non_drafting_pp"])
                for p in pairs
                if p["mostly_non_drafting_pp"]
            ),
            "e150_nuisance_f7_fit_reproduced": all(p["r_squared_reproduced"] for p in pairs),
            "e150_nuisance_f7_k_ratio_mine_over_f7": [
                p["k_ratio_mine_over_f7"] for p in pairs
            ],
            "e150_nuisance_exposure_scale_convention_unresolved": True,
        }
    )
    payload["e150_f7_nuisance_prereg"] = prereg

    e150_lib.write_artifact("a0_f7_correction.json", payload)

    print("-- e150 A0 width-6 boundary on F7's clean input --")
    print(f"  void row {VOID_ROW}   {void_pct:+.4f} %  (retracted)")
    print(f"  clean row {CLEAN_ROW}  {CLEAN_PCT:+.4f} % +/- {CLEAN_SE:.4f}")
    print(f"  break-even ranked pct        {breakeven_pct:+.4f} %")
    print(f"  sigma to boundary            {sigma:.2f}   (was {b['sigma_to_boundary']:.3f})")
    print(f"  required local us to admit   {required:9.1f}")
    print(f"  point-estimate us removed    {point['rung_local_us']:+9.1f}")
    print(f"  shortfall local us           {required - point['rung_local_us']:9.1f}")
    print(f"  share of the 5->6 cliff      {point['rung_local_us'] / step_us:+.4f}")
    print(
        f"  C(6) post per token          {point['c6_post_per_token_us']:9.1f}"
        f"   vs width-5 minimum {c5 / 5.0:9.1f}"
    )
    print(f"  width 6 admissible           {point['admissible']}")
    print(f"  +1se     C(6)/6 {hi['c6_post_per_token_us']:9.1f} admissible {hi['admissible']}")
    print(f"  -1se     C(6)/6 {lo['c6_post_per_token_us']:9.1f} admissible {lo['admissible']}")
    print(
        f"  F7 quotes width6 {F7_QUOTED_W6_US:.1f} vs mine {c6:.1f}"
        f" ({100.0 * (F7_QUOTED_W6_US - c6) / c6:+.3f} %)"
    )
    print(
        f"  F7 quotes width5 {F7_QUOTED_W5_US:.1f} vs mine {c5:.1f}"
        f" ({100.0 * (F7_QUOTED_W5_US - c5) / c5:+.3f} %)"
    )
    print(f"  F7 values are exactly the `us` basis: {basis_match}")
    print("-- same boundary redone entirely inside the `us` basis --")
    print(f"  required local us to admit   {us_basis['required_local_us_to_admit_width6']:9.1f}")
    print(f"  break-even ranked pct        {us_basis['breakeven_ranked_pct']:+.4f} %")
    print(f"  sigma to boundary            {us_basis['sigma_to_boundary']:.2f}")
    print(
        f"  C(6) post per token          {us_basis['c6_post_per_token_us']:9.1f}"
        f"   vs width-5 minimum {us_basis['width5_per_token_us']:9.1f}"
    )
    print(f"  width 6 admissible           {us_basis['admissible']}")

    print("-- F7 section 3 nuisance-tail pre-registration --")
    print(f"  expected gain pp                    {PREREG['e150_expected_gain_pp']:+.4f}")
    print(
        f"  conservative floor pp               "
        f"{PREREG['e150_expected_gain_pp_conservative_floor']:+.4f}"
    )
    print(f"  expected ranked score               {PREREG['e150_expected_ranked_score']:.8f}")
    print(
        f"  required uniform gain vs bar pct    "
        f"{PREREG['e150_required_uniform_candidate_leg_gain_pct_vs_bar']:+.4f}"
    )
    print(f"  nuisance tail acknowledged pp       {PREREG['e150_nuisance_tail_acknowledged_pp']:.2f}")
    print(
        f"  receipt underpowered                "
        f"{PREREG['e150_receipt_is_underpowered_for_this_effect']}"
    )
    print(f"  my receipt available yet            {my_receipt is not None}")
    print("-- nuisance draws measured directly from published receipts --")
    print(
        "  pair                  sched_id  mean_pp  median_pp   drafting range   "
        "k_mine  k_f7   R2_mine  R2_f7"
    )
    print(
        "\n".join(
            f"  {p['pair']:20s}  {str(p['schedule_identical']):>8}"
            f"  {p['contrast_mean_pp']:+7.4f}  {p['contrast_median_pp']:+9.4f}"
            f"  {p['drafting_prompt_min_pp']:+6.3f} to {p['drafting_prompt_max_pp']:+6.3f}"
            f"  {p['fit']['k_state_steps']:+6.3f} {p['f7_reported_k_state_steps']:+5.3f}"
            f"   {p['fit']['r_squared']:6.4f} {p['f7_reported_r_squared']:6.3f}"
            for p in (prereg["e150_nuisance_null_pair"], prereg["e150_nuisance_second_pair"])
            if p
        )
    )
    print(
        f"  observed tail max mean pp           "
        f"{prereg['e150_nuisance_tail_observed_max_pp']:.4f}"
        f"  (F7 assumed {PREREG['e150_nuisance_tail_acknowledged_pp']:.2f})"
    )
    print(
        f"  observed tail max single prompt pp  "
        f"{prereg['e150_nuisance_tail_observed_max_single_prompt_pp']:.4f}"
    )
    print(f"  carrier is drafting rounds          {prereg['e150_nuisance_carrier_is_drafting_rounds']}")
    print(f"  F7 fit reproduced                   {prereg['e150_nuisance_f7_fit_reproduced']}")
    anchor = prereg["e150_anchor_exposure"]
    if anchor:
        print(f"  anchor {anchor['submission_id'][:8]} exposure (s_p from its own receipt):")
        print("    prompt      edl   rounds  drafting   decode_s      s_p    100*s_p")
        print(
            "\n".join(
                f"    {r['prompt_sha256']}  {r['effective_mean_draft_len']:6.3f}"
                f"  {str(r['recovered_round_count']):>6}  {str(r['drafting_round_count']):>8}"
                f"  {r['decode_seconds']:9.4f}  {(r['s_p'] or 0):7.5f}  {(r['s_p'] or 0) * 100:7.4f}"
                for r in anchor["per_prompt"]
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
