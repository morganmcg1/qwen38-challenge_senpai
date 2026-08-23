#!/usr/bin/env python3
"""E140: publish every replayed artifact to W&B as one run.

`harness=local instrument`. Zero GPU, so there is no timing, no thermal gate
and no `gate_qualified_for_timing` claim to make.

The primary metric follows CAMPAIGN RULE 116: it is the actual median of the
eight replayed per-prompt raw ratios, obtained by sorting, never a weighted
sum. `e140_rule116.py` proves that by re-implementing the sort independently
and comparing, and the gap it reports is logged here as evidence rather than
asserted.

Usage:
  python3 e140_wandb_log.py --run-name e140-lookahead-argmax
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import wandb  # noqa: E402

PROJECT = "qwen38-mlx-challenge-senpai"
ENTITY = "wandb-applied-ai-team"
ARTIFACTS = HERE / "e140-artifacts"
BEST_FORM = "per_drafting_round"
# The receipt E128, E134 and E140 replay against, and the two live trees the
# result must also be priced on because their fourth and fifth prompts differ.
CONTINUITY_RECEIPT = "d3c491b5"
FRONTIER_RECEIPT = "623e77af"
CROWN_RECEIPT = "08b67f12"


def load(name: str) -> dict | None:
    path = ARTIFACTS / name
    return json.loads(path.read_text()) if path.exists() else None


def git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE.parent,
                          capture_output=True, text=True,
                          check=True).stdout.strip()


def table(columns, rows):
    return wandb.Table(columns=list(columns), data=[list(r) for r in rows])


def finite(value):
    """`None` rather than an infinity, which W&B cannot plot or aggregate.

    Item D can legitimately produce an infinite bound: if the receipt needs
    more width-2 mass than the whole replayed tree holds above width 2, no
    shallow-end bias can explain the saving at all.
    """
    return value if math.isfinite(value) else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="e140-lookahead-depth-argmax")
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()

    gate = load("gate-cellC.json")
    cells = load("cells.json")
    rule116 = load("rule116.json")
    perturb = load("perturb.json")
    oracle = load("oracle-state.json")
    posttight = load("posttight.json")
    itemab = load("itemab.json")
    r1r2 = load("r1r2.json")
    r2b = load("r2b.json")
    if gate is None or cells is None:
        raise SystemExit("the cell C gate and the 2x2 must both be present")

    run = wandb.init(
        entity=ENTITY, project=PROJECT, name=args.run_name,
        job_type="replay", mode="offline" if args.offline else "online",
        tags=["e140", "lookahead", "argmax", "replay", "zero-gpu",
              "harness=local"],
        config={
            "experiment": "E140",
            "hypothesis": "H140: replacing the first-break cost-model walk "
                          "with an explicit argmax over feasible depths, "
                          "paired with the E134 item 2 measured cost curve, "
                          "beats pb6 at +2.4683 percent held out",
            "harness": "local instrument",
            "gpu_used": False,
            "commit": git_sha(),
            "base_sha": "6115a8ad406dc56310fbba8ee8f6991ad843a047",
            "pr": 140,
            "scoring_rule": "CAMPAIGN RULE 116, sorted median of eight "
                            "per-prompt raw ratios",
            "replay_receipt": cells["receipt"],
            "replay_receipt_score": cells["receipt_score"],
            "windows": cells["windows"],
            "fit_windows": cells["fit_windows"],
            "seeds": cells["seeds"],
            "best_curve_form": cells["best_form"],
            "tier_grid": cells["tier_grid"],
            "advance_bar_pct": cells["reference"]["E_pb6"],
            "close_bar_pct": 1.5,
            "advisor_prediction_pct": [3.0, 6.0],
        })

    def cell(name, form=BEST_FORM):
        return cells["cells"]["%s|%s" % (form, name)]

    d_cell = cell("D_curvelook")
    e_cell = cell("E_pb6")
    f_cell = cell("F_pb6look")

    summary = {
        # The four metrics the assignment names.
        "e140_replayed_ranked_median_pct": d_cell["curve_lopo_mean"],
        "e140_cellC_depth_disagreements": gate[
            "e140_cellC_depth_disagreements"],
        "e140_deeper_round_count": d_cell["tally"]["deeper_round_count"],
        "e140_shallower_round_count": d_cell["tally"]["shallower_round_count"],

        # The gate, on the recorded traces and again inside the replayer.
        "e140_cellC_scored_rounds_recorded": gate["scored_rounds"],
        "e140_cellC_disagreements_replayer": cells[
            "e140_cellC_depth_disagreements_replayer"],
        "e140_cellC_replayed_rounds": cells["cells"][
            "%s|C_flatlook" % BEST_FORM]["tally"]["rounds"],
        "e140_cellC_median_pct": cells["cells"][
            "%s|C_flatlook" % BEST_FORM]["curve_lopo_mean"],
        "e140_positive_control_disagreements": min(
            gate["positive_control_disagreements"].values()),
        "e140_positive_control_can_fail": gate["positive_control_can_fail"],

        # The 2x2 and the two reference arms on the E134 best form.
        "e140_cellA_ship_pct": cell("A_ship")["curve_lopo_mean"],
        "e140_cellB_rankedprice_pct": cell("B_rankedprice")["curve_lopo_mean"],
        "e140_cellD_curvelook_pct": d_cell["curve_lopo_mean"],
        "e140_cellD_curvelook_sd": d_cell["curve_lopo_sd"],
        "e140_cellD_in_sample_pct": d_cell["in_sample_mean"],
        "e140_cellE_pb6_pct": e_cell["curve_lopo_mean"],
        "e140_cellF_pb6look_pct": f_cell["curve_lopo_mean"],
        "e140_cellF_deeper_round_count": f_cell["tally"]["deeper_round_count"],
        "e140_cellF_shallower_round_count": f_cell["tally"][
            "shallower_round_count"],
        "e140_cellF_extra_tokens_per_deeper_round": f_cell["tally"][
            "extra_accepted_per_deeper_round"],
        "e140_cellF_p_first_declined_draft_accepted": f_cell["tally"][
            "p_first_declined_draft_accepted_deeper"],
        "e140_cellF_changed_round_us_per_token_greedy": f_cell["tally"][
            "changed_us_per_token_greedy"],
        "e140_cellF_changed_round_us_per_token_lookahead": f_cell["tally"][
            "changed_us_per_token_lookahead"],

        # Item 4, the cap.
        "e140_cap_share_ship": cell("A_ship")["tally"]["cap_bound_share"],
        "e140_cap_share_cellD": d_cell["tally"]["cap_bound_share"],
        "e140_cap_share_cellE": e_cell["tally"]["cap_bound_share"],
        "e140_cap_share_cellF": f_cell["tally"]["cap_bound_share"],

        # Item 3, curve-form robustness of the hypothesis cell.
        "e140_cellD_form_spread_pp": (
            max(cell("D_curvelook", f)["in_sample_mean"]
                for f in ("pre_arm", "per_round", BEST_FORM, "proportional"))
            - min(cell("D_curvelook", f)["in_sample_mean"]
                  for f in ("pre_arm", "per_round", BEST_FORM,
                            "proportional"))),
        "e140_cellD_forms_above_advance_bar": sum(
            1 for f in ("pre_arm", "per_round", BEST_FORM, "proportional")
            if cell("D_curvelook", f)["in_sample_mean"]
            > cells["reference"]["E_pb6"]),

        # Calibration against the advisor's published reference points.
        "e140_pb6_tier_lopo_held_out_pct": cells["tier_lopo"][BEST_FORM][
            "held_out"],
        "e140_pb6_tier_lopo_sd": cells["tier_lopo"][BEST_FORM]["held_out_sd"],
        "e140_pb6_calibration_gap_pp": (
            e_cell["in_sample_mean"] - cells["reference"]["E_pb6"]),
        "e140_rankedprice_calibration_gap_pp": (
            cell("B_rankedprice")["in_sample_mean"]
            - cells["reference"]["B_rankedprice"]),

        "e140_advance_bar_pct": cells["reference"]["E_pb6"],
        "e140_close_bar_pct": 1.5,
        "e140_verdict": ("close" if d_cell["curve_lopo_mean"] < 1.5
                         else "revise"),
    }

    if rule116 is not None:
        summary["e140_finding196_worst_identity_error"] = rule116[
            "finding196_worst_error"]
        summary["e140_finding196_rows_checked"] = len(
            rule116["finding196_rows"])
        rows = []
        for name, entry in rule116["receipts"].items():
            for arm, value in entry.get("arms", {}).items():
                rows.append([name, entry["score"], arm,
                             value["median_pct_sorted"],
                             value["median_pct_f196_weighted"],
                             value["f196_error_pp"],
                             value["median_pct_harness"],
                             "%s + %s" % (value["fourth"], value["fifth"]),
                             value["reordered"],
                             value["headroom"]["fifth_to_sixth_pct"]]
                            + [value["per_prompt_raw_gain_pct"][p]
                               for p in ("plutarch", "drama", "travel",
                                         "beagle", "essays", "republic",
                                         "medicine", "botany")])
                summary["e140_%s_on_%s_pct" % (arm, name)] = value[
                    "median_pct_sorted"]
        run.log({"rule116_per_receipt": table(
            ["receipt", "receipt_score", "cell", "median_sorted_pct",
             "median_f196_weighted_pct", "f196_error_pp",
             "median_harness_pct", "pair_after", "reordered",
             "fifth_to_sixth_headroom_pct", "plutarch", "drama", "travel",
             "beagle", "essays", "republic", "medicine", "botany"], rows)})
        # The largest error the withdrawn weighted model would still make on a
        # reordering mechanism, which is the case Rule 116 has to cover.
        summary["e140_f196_weighted_worst_error_pp"] = max(
            abs(v["f196_error_pp"])
            for e in rule116["receipts"].values()
            for v in e.get("arms", {}).values())
        summary["e140_rule116_independent_sort_gap_pp"] = max(
            abs(v["median_pct_sorted"] - v["median_pct_harness"])
            for e in rule116["receipts"].values()
            for v in e.get("arms", {}).values())

    run.log({"cells": table(
        ["curve_form", "cell", "in_sample_pct", "in_sample_sd",
         "curve_lopo_pct", "curve_lopo_sd", "disagreements", "deeper",
         "shallower", "cap_bound_share", "f83_weighted_mean_depth"],
        [[form, name, entry["in_sample_mean"], entry["in_sample_sd"],
          entry["curve_lopo_mean"], entry["curve_lopo_sd"],
          entry["tally"]["disagreements"],
          entry["tally"]["deeper_round_count"],
          entry["tally"]["shallower_round_count"],
          entry["tally"]["cap_bound_share"],
          entry["f83_weighted_mean_depth"]]
         for key, entry in sorted(cells["cells"].items())
         for form, name in [key.split("|")]])})

    run.log({"round_start_ema": table(
        ["cell"] + ["p%d" % i for i in range(8)],
        [[name] + cell(name)["tally"]["mean_round_start_ema"]
         for name in ("A_ship", "B_rankedprice", "C_flatlook", "D_curvelook",
                      "E_pb6", "F_pb6look")]
        + [["EMA_PRIOR"] + [0.85 * 0.98 ** i for i in range(8)]])})

    audit_keys = ("recorded_acc_changed", "recorded_acc_unchanged",
                  "recorded_acc_population",
                  "p_first_declined_draft_accepted_changed",
                  "extra_accepted_per_changed_round",
                  "changed_shipped_depth_mean", "changed_censored",
                  "changed_censored_share", "changed_extra_is_lower_bound")
    run.log({"gate_walk_comparison": table(
        ["price", "rounds", "disagreements", "deeper", "shallower",
         "non_quasiconcave", "transitions"]
        + list(audit_keys)
        + ["greedy_d%d" % d for d in range(9)]
        + ["argmax_d%d" % d for d in range(9)],
        [[name, row["rounds"], row["disagreements"], row["deeper"],
          row["shallower"], row["non_quasiconcave"], json.dumps(row["pairs"])]
         + [row.get(k) for k in audit_keys]
         + list(row["greedy_depths"]) + list(row["argmax_depths"])
         for name, row in sorted(gate["walk_comparison"].items())])})

    # The advisor's diagnosis of the defect and the fate of his fix are two
    # different questions. This counts the rounds on which the ratio really is
    # non-quasiconcave, so the diagnosis is scored separately from the cure.
    measured = gate["walk_comparison"][
        "measured %s (cell B/D price)" % BEST_FORM]
    flat = gate["walk_comparison"]["flat 0.18 (cell A/C price)"]
    summary["e140_non_quasiconcave_rounds_measured_price"] = measured[
        "non_quasiconcave"]
    summary["e140_non_quasiconcave_rounds_flat_price"] = flat[
        "non_quasiconcave"]
    summary["e140_recorded_trace_disagreements_measured_price"] = measured[
        "disagreements"]
    # Item 2 on the recorded traces. Every round the argmax changes under the
    # measured price is a round the shipped flat price already drafted to the
    # cap and had fully accepted, so the argmax repairs the measured price
    # rather than finding value the shipped walk misses.
    summary["e140_recorded_changed_acc_mean"] = measured[
        "recorded_acc_changed"]
    summary["e140_recorded_population_acc_mean"] = measured[
        "recorded_acc_population"]
    summary["e140_recorded_changed_shipped_depth_mean"] = measured[
        "changed_shipped_depth_mean"]
    summary["e140_recorded_changed_censored_share"] = measured[
        "changed_censored_share"]
    summary["e140_recorded_changed_extra_is_lower_bound"] = measured[
        "changed_extra_is_lower_bound"]
    summary["e140_recorded_extra_tokens_per_changed_round"] = measured[
        "extra_accepted_per_changed_round"]

    if perturb is not None:
        summary["e140_cliff_step_us"] = perturb["step_us"]
        summary["e140_alphonse_fraction_of_step"] = perturb[
            "alphonse_fraction_of_step"]
        rows = []
        for key, entry in sorted(perturb["arms"].items()):
            form, name, fraction = key.split("|")
            rows.append([form, name, float(fraction), entry["in_sample"][0],
                         entry["curve_lopo"][0] if entry["curve_lopo"]
                         else None, entry["mean_depth"]]
                        + list(entry["depth_share"]))
        run.log({"cliff_perturbation": table(
            ["curve_form", "cell", "cliff_cut", "in_sample_pct",
             "curve_lopo_pct", "f83_weighted_mean_depth"]
            + ["depth_share_%d" % d for d in range(9)], rows)})
        if perturb.get("tier_sweep"):
            run.log({"cliff_perturbation_tier": table(
                ["cliff_cut", "walk", "best_tier", "best_pct",
                 "at_shipped_1p45_pct", "at_tier_1p0_pct"],
                [[float(cut), walk, entry["best_tier"], entry["best"],
                  entry["at_shipped"], entry["at_one"]]
                 for cut, by_walk in sorted(perturb["tier_sweep"].items())
                 for walk, entry in by_walk.items()])})
            worst = max(perturb["tier_sweep"])
            summary["e140_pb6_degradation_at_max_cut_pp"] = (
                perturb["tier_sweep"][worst]["greedy"]["at_shipped"]
                - perturb["tier_sweep"]["0.000000"]["greedy"]["at_shipped"])
            summary["e140_pb6look_degradation_at_max_cut_pp"] = (
                perturb["tier_sweep"][worst]["argmax"]["at_shipped"]
                - perturb["tier_sweep"]["0.000000"]["argmax"]["at_shipped"])
        # F3's falsifier: if pb6 and the argmax layer degrade at the same rate
        # the adaptivity claim is refuted. The ratio of the two least-squares
        # slopes over the cut grid is that test, one number per curve form.
        def slope(form, name):
            xs = perturb["fractions"]
            ys = []
            for fraction in xs:
                entry = perturb["arms"]["%s|%s|%.6f" % (form, name, fraction)]
                ys.append((entry["curve_lopo"] or entry["in_sample"])[0])
            mx = sum(xs) / len(xs)
            my = sum(ys) / len(ys)
            return (sum((x - mx) * (y - my) for x, y in zip(xs, ys))
                    / sum((x - mx) ** 2 for x in xs))

        forms = ("pre_arm", "per_round", BEST_FORM, "proportional")
        run.log({"cliff_degradation_slope": table(
            ["curve_form", "cell", "pp_per_10pct_of_cut"],
            [[form, name, slope(form, name) * 0.10]
             for form in forms
             for name in ("D_curvelook", "E_pb6", "F_pb6look")])})
        for form in forms:
            summary["e140_slope_ratio_pb6look_over_pb6_%s" % form] = (
                slope(form, "F_pb6look") / slope(form, "E_pb6"))
        summary["e140_pb6_slope_pp_per_10pct"] = slope(BEST_FORM,
                                                       "E_pb6") * 0.10
        summary["e140_pb6look_slope_pp_per_10pct"] = slope(BEST_FORM,
                                                           "F_pb6look") * 0.10

    if posttight is not None:
        # F3 arm (a) and F4 items 3, 4 and 8. `log_shipped` is the curve the
        # next ranked receipt will walk, so it is the one that decides.
        rows = []
        for key, entry in sorted(posttight["arms"].items()):
            form, variant, name = key.split("|")
            rows.append([form, variant, name, entry["in_sample"][0],
                         entry["curve_lopo"][0] if entry["curve_lopo"]
                         else None, entry["mean_depth"]]
                        + list(entry["depth_share"]))
        run.log({"posttight_arms": table(
            ["curve_form", "curve_variant", "cell", "in_sample_pct",
             "curve_lopo_pct", "f83_weighted_mean_depth"]
            + ["depth_share_%d" % d for d in range(9)], rows)})
        run.log({"posttight_steps": table(
            ["curve_variant"] + ["step_into_%d" % m for m in range(2, 10)],
            [[name] + list(steps)
             for name, steps in sorted(posttight["steps"].items())])})
        run.log({"posttight_tier8_fit": table(
            ["curve_variant", "best_tier8", "best_pct", "at_tier8_1_pct"],
            [[name, entry["best_tier8"],
              entry["by_tier"]["%.2f" % entry["best_tier8"]],
              entry["at_one"]]
             for name, entry in sorted(posttight["tier8_fits"].items())])})
        run.log({"posttight_cap_probe": table(
            ["curve_variant", "rounds", "wants_past_cap_share",
             "mean_capped_depth", "mean_uncapped_depth"]
            + ["uncapped_depth_share_%d" % d for d in range(9)],
            [[name, entry["rounds"], entry["wants_past_cap_share"],
              entry["mean_capped_depth"], entry["mean_uncapped_depth"]]
             + list(entry["uncapped_depth_share"])
             for name, entry in sorted(posttight["cap_probe"].items())])})

        def post(variant, name, form=BEST_FORM):
            entry = posttight["arms"]["%s|%s|%s" % (form, variant, name)]
            return (entry["curve_lopo"] or entry["in_sample"])[0]

        summary["e140_launch_coef_us"] = posttight["launch_coef"]
        for variant in ("base", "uniform", "log_onePass67", "log_shipped"):
            for name in ("D_curvelook", "E_pb6", "F_pb6look", "G_pb68",
                         "H_pb68look"):
                summary["e140_%s_%s_pct" % (variant, name)] = post(variant,
                                                                   name)
        summary["e140_posttight_shipped_cellD_pct"] = post("log_shipped",
                                                           "D_curvelook")
        summary["e140_posttight_shipped_cellD_minus_pb6_pp"] = (
            post("log_shipped", "D_curvelook") - post("log_shipped", "E_pb6"))
        summary["e140_posttight_shipped_cellD_minus_pb68_pp"] = (
            post("log_shipped", "D_curvelook") - post("log_shipped", "G_pb68"))
        summary["e140_pb68_best_gain_over_pb6_pp"] = max(
            entry["by_tier"]["%.2f" % entry["best_tier8"]] - entry["at_one"]
            for entry in posttight["tier8_fits"].values())
        summary["e140_cellD_wants_past_cap_share"] = max(
            entry["wants_past_cap_share"]
            for entry in posttight["cap_probe"].values())
        summary["e140_uniform_shift_pb6_move_pp"] = (
            post("uniform", "E_pb6") - post("base", "E_pb6"))

    if oracle is not None:
        # Replaces the EMA with the true per-position acceptance vector, so a
        # perfect stationary estimator is priced separately from the argmax.
        widths = {len(e["depth_share"]) for e in oracle["cells"].values()}
        if len(widths) != 1:
            raise SystemExit("oracle depth_share widths disagree: %s" % widths)
        run.log({"oracle_state": table(
            ["cell", "in_sample_pct", "in_sample_sd", "curve_lopo_pct",
             "unweighted_mean_depth"]
            + ["depth_share_%d" % d for d in range(widths.pop())],
            [[name, entry["in_sample_mean"], entry["in_sample_sd"],
              entry["curve_lopo_mean"], entry["unweighted_mean_depth"]]
             + list(entry["depth_share"])
             for name, entry in sorted(oracle["cells"].items())])})
        run.log({"oracle_p_target": table(
            ["prompt"] + ["p%d" % i for i in range(8)],
            [[p] + list(v) for p, v in sorted(oracle["p_target"].items())])})
        summary["e140_oracle_estimator_worth_pp"] = oracle[
            "estimator_worth_pp"]
        summary["e140_oracle_argmax_worth_pp"] = oracle[
            "argmax_worth_with_perfect_estimator_pp"]
        for name, entry in oracle["cells"].items():
            summary["e140_oracle_%s_pct" % name] = entry["in_sample_mean"]

    if itemab is not None:
        # Item A, the shallow end of the tree under both live policies.
        run.log({"itemab_width_histogram": table(
            ["cell", "prompt", "p_width2", "shallow_mass_width_1_to_4",
             "mean_width", "mean_width_sd", "rounds_per_window"]
            + ["width_share_%d" % w for w in range(1, 10)],
            [[cell, prompt, e["p_width2"], e["shallow_mass_width_1_to_4"],
              e["mean_width"], e["mean_width_sd"], e["rounds_per_window"]]
             + [e["width_share"].get(str(w), 0.0) for w in range(1, 10)]
             for cell, hist in sorted(itemab["item_a"]["histograms"].items())
             for prompt, e in sorted(hist.items())])})
        summary["e140_medpair_p_width2_ship"] = itemab["item_a"][
            "medpair_p_width2_ship"]
        summary["e140_medpair_p_width2_pb6"] = itemab["item_a"][
            "medpair_p_width2_pb6"]

        # Item D, the width-2 launch receipt inverted against that histogram.
        d = itemab["item_d"]
        run.log({"itemab_width2_receipt": table(
            ["prompt", "receipt_mbar", "replay_mean_width", "receipt_rounds",
             "replay_rounds", "receipt_delta_us", "replay_p_width2",
             "implied_round_cost_us", "implied_over_law", "required_p_width2",
             "min_forced_mbar_shift", "observed_mbar_error", "seed_sd"],
            [[prompt,
              d["board_receipt"][prompt]["mbar"],
              itemab["item_a"]["histograms"]["A_ship"][prompt]["mean_width"],
              d["board_receipt"][prompt]["rounds_exact"],
              itemab["item_a"]["histograms"]["A_ship"][prompt][
                  "rounds_per_window"],
              d["board_receipt"][prompt]["delta_us_per_token"],
              b["replayed_p_width2"],
              finite(d["implied_cost_us"][prompt]),
              finite(d["implied_cost_us"][prompt]
                     / itemab["law_cost_width2_us"]),
              b["required_p_width2"],
              finite(b["min_forced_mbar_shift"]),
              b["observed_mbar_error"], b["seed_sd"]]
             for prompt, b in sorted(d["shallow_bias_bound"].items())])})
        run.log({"itemab_saving_regressions": table(
            ["regressor", "slope", "intercept_us", "r2"],
            [[name, f["slope"], f["intercept"], f["r2"]]
             for name, f in sorted(d["fits"].items())])})
        summary["e140_law_cost_width2_us"] = itemab["law_cost_width2_us"]
        summary["e140_width2_implied_cost_cv"] = d["spread"]["cv"]
        summary["e140_width2_implied_over_law_mean"] = (
            d["spread"]["mean"] / itemab["law_cost_width2_us"])
        summary["e140_width2_mbar_error_mean"] = d["mbar_error_mean"]
        summary["e140_width2_mbar_error_sd"] = d["mbar_error_sd"]
        summary["e140_width2_round_rate_error_pct"] = d[
            "round_rate_control"]["plutarch"]["error_pct"]
        summary["e140_width2_geometry_share"] = d["geometry_share_of_total"]
        summary["e140_width2_residual_mean_us"] = d["residual_mean_us"]
        summary["e140_width2_residual_sd_us"] = d["residual_sd_us"]
        summary["e140_width2_saving_r2_on_p_width2"] = d["fits"][
            "on_p_width2"]["r2"]
        summary["e140_width2_max_forced_mbar_shift"] = finite(max(
            b["min_forced_mbar_shift"]
            for b in d["shallow_bias_bound"].values()))

        # Item B, pb6's tier re-fitted on each launch table. `log_shipped` is
        # the table the next ranked receipt walks, so it is the one to ship.
        run.log({"itemab_tier_grid": table(
            ["curve_variant", "tier", "curve_lopo_pct"],
            [[variant, float(tier), value]
             for variant, entry in sorted(itemab["item_b"]["tiers"].items())
             for tier, value in sorted(entry["held_out"].items())])})
        run.log({"itemab_tier_best": table(
            ["curve_variant", "best_tier", "best_pct", "at_shipped_1_45_pct",
             "gain_over_shipped_pp", "plateau_low", "plateau_high",
             "sd_at_best"],
            [[variant, e["best_tier"], e["best"], e["at_shipped"],
              e["gain_over_shipped"], e["plateau"][0], e["plateau"][1],
              e["sd_at_best"]]
             for variant, e in sorted(itemab["item_b"]["tiers"].items())])})
        run.log({"itemab_tier_depth": table(
            ["curve_variant", "tier", "mean_depth"]
            + ["width_share_%d" % w for w in range(1, 10)],
            [[key.split("|")[0], float(key.split("|")[1]), e["mean_depth"]]
             + [e["width_share"].get(str(w), 0.0) for w in range(1, 10)]
             for key, e in sorted(itemab["item_b"]["depth"].items())])})
        for variant, e in itemab["item_b"]["tiers"].items():
            summary["e140_tierfit_%s_best_tier" % variant] = e["best_tier"]
            summary["e140_tierfit_%s_best_pct" % variant] = e["best"]
            summary["e140_tierfit_%s_gain_pp" % variant] = e[
                "gain_over_shipped"]
        shipped = itemab["item_b"]["tiers"]["log_shipped"]
        summary["e140_tierfit_shipped_best_tier"] = shipped["best_tier"]
        summary["e140_tierfit_shipped_gain_over_1_45_pp"] = shipped[
            "gain_over_shipped"]

    if r1r2 is not None:
        # F7 R1. The shipped `pb6` arm is a boundary spike composed with a
        # global depth subsidy, and the ranked pair 572b2cc4 -> e003a86d is the
        # first out-of-sample test the replayer has ever had.
        run.log({"r1r2_threshold_table": table(
            ["depth", "ship_threshold", "pb6_threshold", "ratio",
             "subsidy_pct", "advisor_ratio"],
            [[r["depth"], r["ship"], r["pb6"], r["ratio"],
              (1.0 - r["ratio"]) * 100.0, r["advisor_ratio"]]
             for r in r1r2["threshold_table"]])})
        summary["e140_threshold_table_max_error"] = r1r2[
            "threshold_worst_error"]

        r1 = r1r2["r1"]
        run.log({"r1r2_arm_per_prompt": table(
            ["curve_form", "arm", "prompt", "mean_depth", "ratio",
             "non_drafting_share", "rounds_per_512", "control_depth",
             "measured_depth", "measured_ratio"],
            [[form, arm, prompt, row["mean_depth"], row["ratio"],
              row["non_drafting_share"], row["rounds_per_512"],
              r1["control_depth"][prompt], r1["measured_depth"][prompt],
              r1["measured_ratio"][prompt]]
             for form, arms in sorted(r1["forms"].items())
             for arm, entry in sorted(arms.items())
             for prompt, row in sorted(entry["per_prompt"].items())])})
        run.log({"r1r2_arm_median": table(
            ["curve_form", "arm", "median_pct", "median_pct_lopo",
             "reordered", "median_pair", "measured_median_pct"],
            [[form, arm, entry["median_pct"],
              entry.get("median_pct_lopo"), entry["rank"]["reordered"],
              "/".join(entry["rank"]["median_pair"]),
              r1["measured_median_pct"]]
             for form, arms in sorted(r1["forms"].items())
             for arm, entry in sorted(arms.items())])})

        best = r1["forms"][BEST_FORM]
        pred = best["pb6"].get("median_pct_lopo", best["pb6"]["median_pct"])
        summary["e140_pb6_predicted_median_pct"] = pred
        summary["e140_pb6_measured_median_pct"] = r1["measured_median_pct"]
        summary["e140_pb6_prediction_error_pp"] = (
            pred - r1["measured_median_pct"])
        summary["e140_subsidy_median_pct"] = best["S_subsidy"]["median_pct"]
        summary["e140_spike_median_pct"] = best["P_spike"]["median_pct"]
        summary["e140_control_binding_gap_pct"] = r1[
            "control_binding_gap_pct"]

        run.log({"r1r2_unlock_curve": table(
            ["depth0_threshold", "prompt", "non_drafting_share", "mean_depth"],
            [[float(level), prompt, row["non_drafting_share"],
              row["mean_depth"]]
             for level, entry in sorted(r1r2["unlock_curve"].items())
             for prompt, row in sorted(entry.items())])})
        ship_level, sub_level = "%.6f" % 0.18, "%.6f" % (8 * 0.18 / 8.45)
        summary["e140_plutarch_unlock_from_subsidy"] = (
            r1r2["unlock_curve"][sub_level]["plutarch"]["non_drafting_share"]
            - r1r2["unlock_curve"][ship_level]["plutarch"][
                "non_drafting_share"])

        run.log({"r1r2_probe_sweep": table(
            ["forced_draft_rate", "prompt", "non_drafting_share", "mean_depth",
             "ratio", "median_pct"],
            [[float(rate), prompt, row["non_drafting_share"],
              row["mean_depth"], row["ratio"], entry["median_pct"]]
             for rate, entry in sorted(r1r2["probe_sweep"].items())
             for prompt, row in sorted(entry["per_prompt"].items())])})
        summary["e140_plutarch_unlock_from_probe_002"] = (
            r1r2["probe_sweep"]["0.0200"]["per_prompt"]["plutarch"][
                "non_drafting_share"]
            - r1r2["probe_sweep"]["0.0000"]["per_prompt"]["plutarch"][
                "non_drafting_share"])
        summary["e140_probe_030_median_pct"] = r1r2["probe_sweep"]["0.3000"][
            "median_pct"]

        # F7 R2. Campaign Rule 121: a replayed median without its rank vector
        # is not decision-grade.
        run.log({"r1r2_rank_cells": table(
            ["cell", "curve_lopo_pct", "reordered", "pair_changed",
             "median_pair"],
            [[key, row["curve_lopo"], row["reordered"], row["pair_changed"],
              "/".join(row["median_pair"])]
             for key, row in sorted(r1r2["r2_cells"].items())])})
        run.log({"r1r2_rank_tier": table(
            ["tier", "median_pct", "reordered", "pair_changed", "pair_churn",
             "median_pair"],
            [[float(key), row["median_pct"], row["reordered"],
              row["pair_changed"], row["pair_churn"],
              "/".join(row["median_pair"])]
             for key, row in sorted(r1r2["r2_tier_grid"].items(),
                                    key=lambda kv: float(kv[0]))])})
        summary["e140_rank_flag_cells"] = sum(
            1 for r in r1r2["r2_cells"].values() if r["reordered"])
        summary["e140_rank_flag_tier"] = sum(
            1 for r in r1r2["r2_tier_grid"].values() if r["reordered"])
        summary["e140_binding_gap_pct"] = r1r2["binding_gap_pct"]

        if r1r2.get("r2_perturbation"):
            cellrows = r1r2["r2_perturbation"]["cells"]
            run.log({"r1r2_rank_perturbation": table(
                ["curve_form", "cell", "cliff_cut", "in_sample_pct",
                 "curve_lopo_pct", "reordered", "pair_changed", "pair_churn",
                 "median_pair"],
                [[key.split("|")[0], key.split("|")[1],
                  float(key.split("|")[2]), row["in_sample"],
                  row.get("curve_lopo"), row["reordered"],
                  row["pair_changed"], row["pair_churn"],
                  "/".join(row["median_pair"])]
                 for key, row in sorted(cellrows.items())])})
            summary["e140_rank_flag_perturbation"] = sum(
                1 for r in cellrows.values() if r["reordered"])
            summary["e140_rank_pair_changed_perturbation"] = sum(
                1 for r in cellrows.values() if r["pair_changed"])

    if r2b is not None:
        # F8 Rule 123: the median is beagle plus the minimum of the four heavy
        # prompts, so a gain that lands on plutarch, drama or travel is unpaid.
        # `zero_weight_gain_share` and `beagle_cost_pct` price that directly.
        r2b_rows = ([("posttight|" + k, v) for k, v in r2b["posttight"].items()]
                    + [("oracle|" + k, v) for k, v in r2b["oracle"].items()])
        run.log({"r2b_rank_reflag": table(
            ["group_cell", "replayed_median_pct", "artifact_median_pct",
             "curve_lopo_pct", "reproduces_artifact", "reordered",
             "pair_changed", "median_pair", "rank_stable", "rank_agreement",
             "beagle_cost_pct", "zero_weight_gain_share", "plutarch_unlock"],
            [[key, row["in_sample"], row["artifact_in_sample"],
              row["curve_lopo"], row["reproduces_artifact"], row["reordered"],
              row["pair_changed"], "/".join(row["median_pair"]),
              row["rank_stable"], row["rank_agreement"],
              row["beagle_cost_pct"], row["zero_weight_gain_share"],
              row["plutarch_unlock"]]
             for key, row in sorted(r2b_rows)])})
        for key, value in r2b["summary"].items():
            if isinstance(value, (int, float, bool)):
                summary["e140_r2b_" + key] = value
        summary["e140_r2b_plutarch_unlock_cells"] = len(
            r2b["summary"]["plutarch_unlock_cells"])
        summary["e140_r2b_worst_zero_weight_share"] = max(
            row["zero_weight_gain_share"] for _, row in r2b_rows)

    run.summary.update(summary)
    print("run id   %s" % run.id)
    print("run url  %s" % run.url)
    for key in sorted(summary):
        print("  %-52s %s" % (key, summary[key]))
    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
