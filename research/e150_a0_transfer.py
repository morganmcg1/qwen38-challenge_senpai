#!/usr/bin/env python3
"""E150 F6 section 3: is the measured 5->6 cliff physics, or `onePass6`?

`harness=local` for every curve and price. `harness=ranked` for the three
receipt-derived inputs, which are labelled at the point of use. Zero GPU.
Zero Swift. Rule 79 is not engaged: this module publishes no timing contrast.

THE QUESTION
------------
FINDING 259 measured, on the ranked runner, that routing verify width 6
through the one-pass template costs `+1.1455 %` of the candidate leg,
Rule 148 weighted. Our `rankedMeasuredRoundMicroseconds` was read off a tree
that HAS that rung, so the width-6 point of our cost curve, and therefore the
5->6 cliff that makes widths 6 and 7 inadmissible, contains the cost of a code
path the campaign is about to delete.

If the rung explains most of the cliff, Rule 138's admissible set is a
property of deletable code and the linearised policy must be re-derived. If it
explains a minority, the cliff is mostly real.

THE FRAME TRAP THIS MODULE EXISTS TO AVOID
------------------------------------------
F6 quotes "measured curve marginals 4->5 5,759.0  5->6 13,851.1  6->7
12,295.1  7->8 1,875.1" and asks for those to be compared against a ranked
microsecond figure after applying a local-to-ranked transfer constant.

Those four numbers are ALREADY in the ranked frame. Each one is the E145 R2
measured step divided by the fitted level transfer `k = 2.1034238`, and
`check_advisor_marginals_are_ranked_frame` proves it to six significant
figures on three of the four steps. Applying `k` again would inflate the
answer by `k` and would report the rung as explaining two thirds of the cliff
instead of one third.

So the comparison is done once, in one named frame, and the module refuses to
run if the identity check fails.

Usage:
  python3 research/e150_a0_transfer.py --json a0_transfer.json
"""
from __future__ import annotations

import argparse
import copy
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import e150_lib  # noqa: E402
from e128_replay import SEGMENTED_VERIFY_DEPTH_CAP  # noqa: E402
from e140_lookahead import curve_price  # noqa: E402
from e145_r3 import installable  # noqa: E402
from e145_r7 import admissibility  # noqa: E402
from e145_r7_state import curves_and_prices  # noqa: E402
from e150_curve_bracket import price_cell, solve_mu  # noqa: E402
from e150_lib import build_env, write_artifact  # noqa: E402

CURVE_JSON = HERE.parent / "research/e145-artifacts/curve.json"

# ------------------------------------------------------- ranked-frame inputs
# harness=ranked. Every value here comes from a resolved Yukon receipt or from
# a campaign rule fitted on receipts. None of it is measurable locally.

# FINDING 259. `684821ed -> c47b45be`, candidate leg, positive = slower, with
# `effective_mean_draft_len` digit-identical on all eight ranked prompts, so
# the schedule is held fixed and the contrast is a pure width-table cost.
F259_CANDIDATE_MEAN_PCT = 1.0583
F259_CANDIDATE_SD_PCT = 0.7190
F259_CANDIDATE_SE_PCT = 0.2542
F259_RULE148_WEIGHTED_PCT = 1.1455

# Rule 134, thorfinn's corrected constant, total-leg frame: how many ranked
# microseconds per round buy one percent of published median.
RULE134_US_PER_ROUND_PER_PCT = 524.5

# Rule 148 weighted ranked width-6 round mass. The rung is paid only on rounds
# that actually run at width 6, so the per-round average must be divided by
# this to reach a per-width-6-round figure.
RANKED_WIDTH6_MASS_RULE148 = 0.1397

# What F6 published after doing the division itself, kept so this module
# reproduces the advisor's arithmetic rather than asserting agreement.
F6_RANKED_US_PER_WIDTH6_ROUND = 4301.0

# The four F6 "measured curve marginals". Proved below to be ranked-frame.
F6_MEASURED_MARGINALS = {
    "4->5": 5759.0, "5->6": 13851.1, "6->7": 12295.1, "7->8": 1875.1,
}

# Six significant figures is the precision F6 published to; the identity is
# accepted when every step agrees with `measured_us / k` inside this.
FRAME_CHECK_TOLERANCE_PCT = 0.10

# A0 changes the width-6 route only: F6 states the fix adds a `Table.onePass7`
# rung and drops `onePass6`, so widths 7 through 9 keep the route they were
# measured with.
A0_AFFECTED_WIDTH = 6


def ranked_us_per_width6_round(pct: float) -> float:
    """Ranked microseconds the `onePass6` rung costs on a width-6 round."""
    return pct * RULE134_US_PER_ROUND_PER_PCT / RANKED_WIDTH6_MASS_RULE148


def load_transfer() -> dict:
    """The E145 R2 level transfer and step table, with their own provenance."""
    doc = json.loads(CURVE_JSON.read_text())
    if doc.get("harness") != "local":
        raise SystemExit("e145 curve artifact is not labelled harness=local")
    transfer = doc["level_transfer"]
    return {
        "k": transfer["k"],
        "per_width_ratio": {int(w): v
                            for w, v in transfer["per_width_ratio"].items()},
        "residual_pct": {int(w): v
                         for w, v in transfer["residual_pct"].items()},
        "worst_residual_pct": transfer["worst_residual_pct"],
        "worst_width": transfer["worst_width"],
        "steps_us": {name: step["measured_us"]
                     for name, step in doc["steps"].items()},
        "replayed_ranked_us": {int(w): v for w, v
                               in doc["replayed_ranked_us"].items()},
    }


def check_advisor_marginals_are_ranked_frame(transfer: dict) -> dict:
    """Prove F6's measured marginals are `measured_us / k`, not local.

    This is the positive control for the whole module. If it fails, the frame
    is not what this module assumes and every ratio below is wrong by a factor
    of `k`, so the run stops instead of reporting a number.
    """
    k = transfer["k"]
    rows = []
    worst = 0.0
    for name, published in sorted(F6_MEASURED_MARGINALS.items()):
        measured = transfer["steps_us"][name]
        implied = measured / k
        error_pct = 100.0 * (implied - published) / published
        worst = max(worst, abs(error_pct))
        rows.append({"step": name, "measured_local_us": measured,
                     "measured_over_k_us": implied,
                     "f6_published_us": published,
                     "error_pct": error_pct,
                     "implied_k": measured / published})
    return {"rows": rows, "worst_abs_error_pct": worst,
            "frame_is_ranked": worst <= FRAME_CHECK_TOLERANCE_PCT,
            "tolerance_pct": FRAME_CHECK_TOLERANCE_PCT}


def rung_share(ranked_us: float, k: float, local_step_us: float) -> float:
    """Fraction of one local cliff explained by a ranked per-round cost.

    Both routes give the same number, and the module reports both so a reader
    can see the transfer is applied exactly once:

        local frame:  (ranked_us * k) / local_step_us
        ranked frame: ranked_us / (local_step_us / k)
    """
    return ranked_us * k / local_step_us


def transfer_uncertainty(ranked_us: float, transfer: dict,
                         local_step_us: float) -> dict:
    """The share under every defensible choice of the transfer constant."""
    ratios = transfer["per_width_ratio"]
    named = {
        "pooled_fit_k": transfer["k"],
        "width6_specific": ratios[6],
        "width5_specific": ratios[5],
        "min_over_widths": min(ratios.values()),
        "max_over_widths": max(ratios.values()),
    }
    return {name: {"k": k, "share": rung_share(ranked_us, k, local_step_us)}
            for name, k in named.items()}


def ranked_uncertainty(transfer: dict, local_step_us: float) -> dict:
    """The share under the ranked measurement's own one-sigma band.

    F259 publishes a standard error for the UNWEIGHTED eight-prompt mean, not
    for the Rule 148 weighted figure. The weighted figure is used as the point
    estimate and the unweighted standard error is carried as a proxy for its
    spread, which is stated rather than hidden.
    """
    k = transfer["k"]
    out = {}
    for name, pct in (
        ("rule148_weighted", F259_RULE148_WEIGHTED_PCT),
        ("rule148_weighted_minus_1se",
         F259_RULE148_WEIGHTED_PCT - F259_CANDIDATE_SE_PCT),
        ("rule148_weighted_plus_1se",
         F259_RULE148_WEIGHTED_PCT + F259_CANDIDATE_SE_PCT),
        ("unweighted_mean", F259_CANDIDATE_MEAN_PCT),
    ):
        us = ranked_us_per_width6_round(pct)
        out[name] = {"pct": pct, "ranked_us_per_width6_round": us,
                     "share": rung_share(us, k, local_step_us)}
    return out


def post_a0_points(points: dict, local_saving_us: float) -> dict:
    """The measured curve with the `onePass6` rung removed from width 6."""
    out = dict(points)
    out[A0_AFFECTED_WIDTH] = points[A0_AFFECTED_WIDTH] - local_saving_us
    return out


def curve_summary(points: dict) -> dict:
    """Steps, cost per token and the admissible set for one curve."""
    widths = sorted(points)
    adm = admissibility(points)
    return {
        "round_us": {str(w): points[w] for w in widths},
        "step_us": {"%d->%d" % (w, w + 1): points[w + 1] - points[w]
                    for w in widths if w + 1 in points},
        "cost_per_token_us": {str(w): points[w] / w for w in widths},
        "admissible_widths": sorted(adm["admissible_widths"]),
        "running_minimum_width": min(
            (w for w in widths), key=lambda w: points[w] / w),
    }


def admissibility_boundary(points: dict, transfer: dict,
                           local_step_us: float) -> dict:
    """How far is the post-A0 width-6 cost from becoming a running minimum?

    The point estimate says width 6 stays inadmissible after A0, and that is
    the answer the ordering decision would be built on. It is worth almost
    nothing on its own, because the margin is far smaller than the uncertainty
    already reported on the rung.

    Width 6 is admissible when `C(6)/6 < C(5)/5`, so the rung has to remove

        C(6) - 6 * C(5) / 5

    local microseconds. This function reports that requirement, the point
    estimate against it, and the ranked percentage at which the two are equal.
    That last number is the useful one: it converts the whole question into a
    single ranked measurement with a known standard error, so the distance to
    the boundary can be stated in sigma instead of in a verdict.
    """
    c5, c6 = points[5], points[A0_AFFECTED_WIDTH]
    required_local_us = c6 - A0_AFFECTED_WIDTH * c5 / 5.0
    k = transfer["k"]
    us_per_pct = ranked_us_per_width6_round(1.0)
    breakeven_ranked_pct = required_local_us / k / us_per_pct

    pct = F259_RULE148_WEIGHTED_PCT
    se = F259_CANDIDATE_SE_PCT
    sigma_to_boundary = (breakeven_ranked_pct - pct) / se

    # The same boundary expressed in the transfer constant, holding the ranked
    # measurement fixed. k is the other input and it is the less well
    # determined of the two.
    breakeven_k = required_local_us / (pct * us_per_pct)
    k_values = list(transfer["per_width_ratio"].values())

    return {
        "required_local_us_to_admit_width6": required_local_us,
        "point_estimate_local_us_removed": rung_local_us(pct, k),
        "shortfall_local_us": required_local_us - rung_local_us(pct, k),
        "breakeven_ranked_pct": breakeven_ranked_pct,
        "measured_ranked_pct": pct,
        "measured_ranked_se": se,
        "sigma_to_boundary": sigma_to_boundary,
        "breakeven_k": breakeven_k,
        "k_used": k,
        "k_min_over_widths": min(k_values),
        "k_max_over_widths": max(k_values),
        "admissible_at_k_max": rung_local_us(pct, max(k_values))
        >= required_local_us,
        "admissible_at_ranked_plus_1se": rung_local_us(pct + se, k)
        >= required_local_us,
        "admissible_at_point_estimate": rung_local_us(pct, k)
        >= required_local_us,
        "step_us_used_as_divisor": local_step_us,
    }


def rung_local_us(ranked_pct: float, k: float) -> float:
    return ranked_us_per_width6_round(ranked_pct) * k


def score_curve(env, label: str, curve, price, admitted: set,
                windows: int) -> dict:
    """Solve `mu*` on one measured curve and price both rules against it."""
    env.measured = curve
    env.measured_price = price
    env.admitted = admitted
    env._base = {}

    shipped = price_cell(env, "measured", "measured", "shipped", "greedy",
                         None, None)
    solved = solve_mu(env, "measured", "shipped", "ratio", {})
    mu = solved["mu_star"]
    policy = price_cell(env, "measured", "measured", "shipped", "ratio",
                        {}, mu)
    return {
        "label": label,
        "mu_star": mu,
        "lambda_star": 1.0 / mu,
        "mu_converged": solved["converged"],
        "shipped_pct": shipped["median_pct_mean"],
        "shipped_pct_sd": shipped["median_pct_sd"],
        "shipped_mean_depth": shipped["weighted_mean_depth"],
        "policy_pct": policy["median_pct_mean"],
        "policy_pct_sd": policy["median_pct_sd"],
        "policy_mean_depth": policy["weighted_mean_depth"],
        "policy_frac_inadmissible": policy["frac_rounds_inadmissible"],
        "policy_width_histogram": policy["width_histogram"],
        "gain_pp": policy["median_pct_mean"] - shipped["median_pct_mean"],
        "windows": windows,
        "cap_limit": SEGMENTED_VERIFY_DEPTH_CAP,
    }


def verdict(share: float, band: tuple) -> str:
    low, high = band
    if low >= 0.60:
        return ("the rung explains most of the cliff: Rule 138's admissible "
                "set is a property of deletable code and the policy must be "
                "re-derived on a post-A0 curve")
    if high <= 0.50:
        return ("the rung explains a minority of the cliff: the cliff is "
                "mostly real and the policy is mostly safe, but the width-6 "
                "cost carries a correction of this size")
    return ("the band straddles half the cliff and the transfer constant is "
            "too uncertain to decide; name the measurement that settles it")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--json", type=str, default="a0_transfer.json")
    ap.add_argument("--arithmetic-only", action="store_true")
    args = ap.parse_args()

    print("E150 F6 section 3 - is the 5->6 cliff physics or onePass6?")
    print("  harness=local for curves and prices")
    print("  harness=ranked for FINDING 259, Rule 134 and Rule 148 inputs")
    print("  gpu_used=False  rule_79=not_engaged")

    transfer = load_transfer()
    frame = check_advisor_marginals_are_ranked_frame(transfer)
    print("\n-- positive control: which frame are F6's marginals in? --")
    for r in frame["rows"]:
        print("  %-6s local %10.1f  /k %10.1f  F6 %10.1f  err %+7.4f %%"
              % (r["step"], r["measured_local_us"], r["measured_over_k_us"],
                 r["f6_published_us"], r["error_pct"]))
    print("  worst |error| %.4f %%  frame_is_ranked=%s"
          % (frame["worst_abs_error_pct"], frame["frame_is_ranked"]))
    if not frame["frame_is_ranked"]:
        raise SystemExit(
            "F6's marginals are not measured_us/k; the frame assumption this "
            "module rests on is false and no ratio below would be readable")

    k = transfer["k"]
    ranked_us = ranked_us_per_width6_round(F259_RULE148_WEIGHTED_PCT)
    reproduce_error_pct = 100.0 * (
        ranked_us - F6_RANKED_US_PER_WIDTH6_ROUND
    ) / F6_RANKED_US_PER_WIDTH6_ROUND
    local_saving_us = ranked_us * k

    # Two bases exist for the same measured step and they differ by 2.4 %.
    # `us` is the per-leg mean the E145 R2 step table and the level transfer
    # were both fitted on. `us_mean_from_blocks` is what `measured_points`
    # reads, so it is the basis of the curve compiled into Swift and of every
    # price in this experiment. Both are reported; the headline uses `us`
    # because that is the basis `k` belongs to.
    step_us_basis = transfer["steps_us"]["5->6"]
    print("\n-- the transfer --")
    print("  ranked us per width-6 round        %10.1f  (F6 says %.0f, "
          "err %+0.3f %%)"
          % (ranked_us, F6_RANKED_US_PER_WIDTH6_ROUND, reproduce_error_pct))
    print("  transfer constant k                %10.6f  local us per ranked us"
          % k)
    print("  local us the rung costs            %10.1f" % local_saving_us)
    print("  measured 5->6 step, us basis       %10.1f" % step_us_basis)

    # The curve itself is cheap; only the acceptance transfer cache behind
    # `build_env` is expensive, and the admissibility boundary does not need
    # it. So the boundary is computed in every mode.
    points = dict(curves_and_prices("measured")[0])
    step_blocks = points[6] - points[5]
    print("  measured 5->6 step, blocks basis   %10.1f" % step_blocks)

    env = None
    pre = post = None
    if not args.arithmetic_only:
        env = build_env(windows=args.windows, seeds=args.seeds)

    share_us = rung_share(ranked_us, k, step_us_basis)
    band = transfer_uncertainty(ranked_us, transfer, step_us_basis)
    ranked_band = ranked_uncertainty(transfer, step_us_basis)
    shares = ([v["share"] for v in band.values()]
              + [v["share"] for v in ranked_band.values()])
    band_low, band_high = min(shares), max(shares)

    print("\n-- the ratio --")
    print("  share of the 5->6 cliff explained by onePass6  %.4f  (%.2f %%)"
          % (share_us, 100.0 * share_us))
    print("  transfer-constant band:")
    for name, entry in band.items():
        print("    %-20s k %8.6f  share %.4f" % (name, entry["k"],
                                                 entry["share"]))
    print("  ranked-measurement band:")
    for name, entry in ranked_band.items():
        print("    %-28s %+0.4f %%  share %.4f"
              % (name, entry["pct"], entry["share"]))
    print("  combined band  [%.4f, %.4f]" % (band_low, band_high))
    print("  VERDICT: %s" % verdict(share_us, (band_low, band_high)))

    boundary = admissibility_boundary(points, transfer, step_us_basis)
    print("\n-- does A0 make width 6 admissible? --")
    print("  local us the rung must remove      %10.1f"
          % boundary["required_local_us_to_admit_width6"])
    print("  local us the point estimate removes%10.1f"
          % boundary["point_estimate_local_us_removed"])
    print("  shortfall                          %10.1f"
          % boundary["shortfall_local_us"])
    print("  the boundary as a ranked measurement:")
    print("    break-even ranked pct            %+10.4f %%"
          % boundary["breakeven_ranked_pct"])
    print("    measured ranked pct              %+10.4f %%  se %.4f"
          % (boundary["measured_ranked_pct"], boundary["measured_ranked_se"]))
    print("    DISTANCE TO THE BOUNDARY         %10.3f sigma"
          % boundary["sigma_to_boundary"])
    print("  the boundary as a transfer constant:")
    print("    break-even k                     %10.6f  (used %.6f, "
          "per-width range %.6f..%.6f)"
          % (boundary["breakeven_k"], boundary["k_used"],
             boundary["k_min_over_widths"], boundary["k_max_over_widths"]))
    print("  admissible at point estimate  %s"
          % boundary["admissible_at_point_estimate"])
    print("  admissible at ranked +1se     %s"
          % boundary["admissible_at_ranked_plus_1se"])
    print("  admissible at k_max           %s"
          % boundary["admissible_at_k_max"])
    boundary_resolved = not (
        boundary["admissible_at_ranked_plus_1se"]
        or boundary["admissible_at_k_max"])
    print("  WIDTH-6 ADMISSIBILITY RESOLVED  %s" % boundary_resolved)

    payload = {
        "e150_a0_admissibility_boundary": boundary,
        "e150_a0_width6_admissibility_resolved": boundary_resolved,
        "e150_a0_breakeven_ranked_pct": boundary["breakeven_ranked_pct"],
        "e150_a0_sigma_to_admissibility_boundary": boundary[
            "sigma_to_boundary"],
        "e150_a0_harness": "local",
        "e150_a0_ranked_inputs_harness": "ranked",
        "e150_a0_gpu_used": False,
        "e150_a0_frame_control": frame,
        "e150_a0_transfer_constant_k": k,
        "e150_a0_transfer_provenance": (
            "research/e145-artifacts/curve.json level_transfer.k, one scalar "
            "fitted across measured widths 2..8 of the E145 R2 live pinned "
            "decode on an M4 Pro against the ranked replayed curve"),
        "e150_a0_transfer_per_width_ratio": transfer["per_width_ratio"],
        "e150_a0_transfer_residual_pct": transfer["residual_pct"],
        "e150_a0_transfer_worst_residual_pct": transfer["worst_residual_pct"],
        "e150_a0_transfer_worst_width": transfer["worst_width"],
        "e150_a0_ranked_us_per_width6_round": ranked_us,
        "e150_a0_ranked_us_reproduce_error_pct": reproduce_error_pct,
        "e150_a0_local_us_of_rung": local_saving_us,
        "e150_a0_measured_step_5_to_6_us_basis": step_us_basis,
        "e150_a0_measured_step_5_to_6_blocks_basis": step_blocks,
        "e150_a0_rung_share_of_cliff": share_us,
        "e150_a0_rung_share_band": [band_low, band_high],
        "e150_a0_rung_share_by_transfer_choice": band,
        "e150_a0_rung_share_by_ranked_input": ranked_band,
        "e150_a0_verdict": verdict(share_us, (band_low, band_high)),
    }

    if not args.arithmetic_only:
        pre_points = dict(env.points)
        post_points = post_a0_points(pre_points, local_saving_us)
        pre_summary = curve_summary(pre_points)
        post_summary = curve_summary(post_points)
        print("\n-- what A0 does to the curve --")
        print("  pre-A0  admissible widths %s" %
              pre_summary["admissible_widths"])
        print("  post-A0 admissible widths %s" %
              post_summary["admissible_widths"])
        for w in sorted(pre_points):
            print("    w%d  cost/token  pre %10.1f  post %10.1f"
                  % (w, pre_summary["cost_per_token_us"][str(w)],
                     post_summary["cost_per_token_us"][str(w)]))

        pre_curve = installable(pre_points, "prea0")
        post_curve = installable(post_points, "posta0")
        pre_price = curve_price(pre_curve)
        post_price = curve_price(post_curve)

        print("\n-- re-pricing the policy on each curve --")
        pre = score_curve(env, "pre_a0", pre_curve, pre_price,
                          set(pre_summary["admissible_widths"]), args.windows)
        print("  pre-A0   mu* %.6f  shipped %+0.4f  policy %+0.4f  "
              "gain %+0.4f  depth %.4f  inadm %.4f"
              % (pre["mu_star"], pre["shipped_pct"], pre["policy_pct"],
                 pre["gain_pp"], pre["policy_mean_depth"],
                 pre["policy_frac_inadmissible"]))
        post = score_curve(env, "post_a0", post_curve, post_price,
                           set(post_summary["admissible_widths"]),
                           args.windows)
        print("  post-A0  mu* %.6f  shipped %+0.4f  policy %+0.4f  "
              "gain %+0.4f  depth %.4f  inadm %.4f"
              % (post["mu_star"], post["shipped_pct"], post["policy_pct"],
                 post["gain_pp"], post["policy_mean_depth"],
                 post["policy_frac_inadmissible"]))

        payload.update({
            "e150_a0_curve_pre": pre_summary,
            "e150_a0_curve_post": post_summary,
            "e150_a0_width6_admissible_pre":
                6 in pre_summary["admissible_widths"],
            "e150_a0_width6_admissible_post":
                6 in post_summary["admissible_widths"],
            "e150_a0_price_pre": {"marginal": pre_price[0],
                                  "cumulative": pre_price[1]},
            "e150_a0_price_post": {"marginal": post_price[0],
                                   "cumulative": post_price[1]},
            "e150_a0_repriced_pre": pre,
            "e150_a0_repriced_post": post,
            "e150_a0_mu_star_shift": post["mu_star"] - pre["mu_star"],
            "e150_a0_policy_gain_shift_pp": post["gain_pp"] - pre["gain_pp"],
            "e150_a0_mean_depth_shift": (post["policy_mean_depth"]
                                         - pre["policy_mean_depth"]),
            "e150_a0_identity": env.identity(),
        })

    if args.arithmetic_only:
        # The arithmetic pass does not produce the repriced sections, so it
        # merges rather than replaces. Replacing would silently delete the
        # expensive half of an existing artifact and leave a file that still
        # looks complete.
        existing = e150_lib.ARTIFACTS / args.json
        if existing.exists():
            merged = json.loads(existing.read_text())
            merged.update(payload)
            payload = merged

    write_artifact(args.json, payload)
    print("\nwrote research/e150-artifacts/%s" % args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
