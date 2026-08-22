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

    if oracle is not None:
        # Replaces the EMA with the true per-position acceptance vector, so a
        # perfect stationary estimator is priced separately from the argmax.
        run.log({"oracle_state": table(
            ["cell", "in_sample_pct", "in_sample_sd", "curve_lopo_pct",
             "unweighted_mean_depth"]
            + ["depth_share_%d" % d for d in range(9)],
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

    run.summary.update(summary)
    print("run id   %s" % run.id)
    print("run url  %s" % run.url)
    for key in sorted(summary):
        print("  %-52s %s" % (key, summary[key]))
    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
