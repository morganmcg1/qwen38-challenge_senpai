#!/usr/bin/env python3
"""Publish the E134 discrimination and pass-boundary pricing result to W&B.

    usage: research/e134_wandb_log.py [--only RUN ...]

Run from the repository ROOT.

E134 asked whether the per-round margin vector the session already computes
can recover part of the `+8.52 %` oracle discrimination gap. It cannot: every
estimator arm failed leave-one-fixture-out. The oracle decomposition then
showed the gap is not a discrimination gap at all on our own cost curve. It
concentrates at one boundary, and what pays there is the PRICE, not a better
signal.

Runs published here:

  `e134-rung1-incremental-auc`
      Zero GPU. Per-boundary incremental AUC of the new inputs over the
      shipped input set, pooled fit with leave-one-fixture-out validation.
      Carries the secondary metric and the depth-4 sign inversion.
  `e134-rung2-estimator-arms`
      Zero GPU. Every implementable estimator arm, scored as replayed ranked
      median percent with leave-one-prompt-out weight selection.
  `e134-rung3-boundary-decomposition`
      Zero GPU. The oracle gap split by boundary under both cost curves, and
      the price families that follow from it.
  `e134-rung4-pass-boundary-price`
      Zero GPU. The headline. The `pb6` arm, its controls, its placebos, and
      the pre-registered prediction for a ranked receipt.
  `e134-item0-warm-parity-arms`
      GPU. The one arm matrix in this experiment that is a real timing
      measurement. Four warm-phase arms on two prompts under ABBA. Every arm
      is SLOWER than the base, so item 0 closes as a regression.
  `e134-item2-measured-curve-refit`
      Zero GPU. The pass-boundary cliff refitted against the advisor's ranked
      pair for the one-pass QMV table now on the base. It CONFIRMS the cliff
      and makes it slightly steeper, so the rung-4 headline survives.

Except for `e134-item0-warm-parity-arms`, every number here is a model output
from an offline replayer. Those runs log `timing_valid`,
`cool_gate_passed_real_gate` and `gate_qualified_for_timing` false verbatim,
and nothing here is an official or ranked score. Rule 79 applies: only a
ranked receipt can validate a depth-price change.

`e134-item0-warm-parity-arms` IS a timing measurement, but it ran under the
standing counterbalanced ungated mode. It logs `timing_valid` true and both
gate fields false verbatim. It is directional causal evidence inside one ABBA
session and is not comparable with a gated historical run.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import wandb

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
GROUP = "e134-oracle-discrimination-at-the-m6-cliff"
HOST = "apple-m4-pro-applegpu_g16s-48gib"
ART = pathlib.Path("research/e134-artifacts")

BASE_SHA = "35d8cf586b8671dc3d01faf3cdbd724ec603801b"
ADVISOR_BRANCH = "senpai/qwen38-mtp-r1"
PR_NUMBER = 134

COMMON = {
    "experiment": "e134",
    "base_sha": BASE_SHA,
    "advisor_branch": ADVISOR_BRANCH,
    "pr": PR_NUMBER,
    "host_profile": HOST,
    "harness": "ranked-model",
    "timing_valid": False,
    "cool_gate_passed_real_gate": False,
    "gate_qualified_for_timing": False,
    "official_or_ranked_score": False,
    "token_window": 512,
    "offered_depth": 8,
    "receipt": "d3c491b5",
}


def load(name: str):
    path = ART / name
    return json.loads(path.read_text()) if path.exists() else None


def start(name: str, config: dict):
    return wandb.init(
        entity=ENTITY, project=PROJECT, group=GROUP, name=name,
        job_type=name.split("-")[1], config={**COMMON, **config},
        reinit=True)


def log_rung1() -> None:
    data = load("rung1-incremental-auc.json")
    if data is None:
        print("rung1: no artifact, skipping")
        return
    run = start("e134-rung1-incremental-auc", {
        "scored_surface_changed": False,
        "runs": "e128 forced-depth archive, 12 fixtures, forced_depth=7",
        "validation": "pooled fit, leave-one-fixture-out",
    })
    policy = data["one_global_policy"]
    table = wandb.Table(columns=[
        "boundary", "observations", "fixtures", "shipped_auc_in",
        "shipped_auc_lofo", "all_auc_in", "all_auc_lofo",
        "incremental_lofo", "null_lofo", "net_lofo"])
    for key in sorted(policy, key=int):
        row = policy[key]
        table.add_data(
            int(key), row["observations"], row["fixtures"],
            row["shipped_in_sample"], row["shipped_heldout"],
            row["all_in_sample"], row["all_heldout"],
            row["incremental_heldout"], row["null_floor_heldout"],
            row["net_heldout"])
    run.log({"rung1/per_boundary": table})
    single = policy["4"]["single_incremental_heldout"]
    stable = wandb.Table(columns=["signal", "incremental_auc_lofo"])
    for name, value in sorted(single.items(), key=lambda kv: -kv[1]):
        stable.add_data(name, value)
    run.log({"rung1/single_input_at_depth4": stable})
    top = max(single.values())
    run.summary.update({
        "e134_incremental_auc_at_depth4": top,
        "e134_incremental_auc_at_depth4_best_signal":
            max(single, key=lambda k: single[k]),
        "e134_incremental_auc_at_depth4_all_inputs":
            policy["4"]["incremental_heldout"],
        "stop_rule_threshold": 0.02,
        "stop_rule_passed": top >= 0.02,
        "verdict": "marginal pass; the whole-set incremental AUC at depth 4 "
                   "is NEGATIVE held out and only one single input clears "
                   "the 0.02 bar",
    })
    run.finish()


def log_rung2() -> None:
    data = load("rung2-arms.json")
    if data is None:
        print("rung2: no artifact, skipping")
        return
    run = start("e134-rung2-estimator-arms", {
        "scored_surface_changed": False,
        "shipping_bar_held_out_pct": 0.50,
        "close_axis_bar_held_out_pct": 0.30,
        "validation": "leave-one-prompt-out weight selection, 6 seeds",
    })
    table = wandb.Table(columns=["arm", "in_sample_pct", "held_out_pct",
                                 "held_out_sd", "gap"])
    for name, entry in sorted(data["held_out"].items()):
        table.add_data(name, entry["in_sample"], entry["held_out"],
                       entry["held_out_sd"],
                       entry["held_out"] - entry["in_sample"])
    run.log({"rung2/arms": table})
    controls = wandb.Table(columns=["arm", "mean_pct", "sd"])
    for name, entry in sorted(data["in_sample"].items()):
        controls.add_data(name, entry["mean"], entry["sd"])
    run.log({"rung2/grid_in_sample": controls})
    best = max(v["held_out"] for v in data["held_out"].values())
    run.summary.update({
        "e134_replayed_ranked_median_pct": best,
        "oracle_pct": data["in_sample"]["oracle@0.0"]["mean"],
        "receipt_score": data["receipt_score"],
        "verdict": "CLOSE THE AXIS: no estimator arm clears +0.30 percent "
                   "held out",
    })
    run.finish()


def log_rung3() -> None:
    ours = load("rung3-ours-cliff4.json")
    board = load("rung3-board-decomposition.json")
    if ours is None:
        print("rung3: no artifact, skipping")
        return
    run = start("e134-rung3-boundary-decomposition", {
        "scored_surface_changed": False,
        "curves": "ours (break M>=6) and board (break M>=5)",
    })
    for label, data in (("ours", ours), ("board", board)):
        if data is None:
            continue
        increments = data["increments"]
        total = sum(increments.values())
        table = wandb.Table(columns=["boundary", "increment_pct", "share"])
        for key in sorted(increments, key=int):
            table.add_data(int(key), increments[key],
                           increments[key] / total if total else None)
        run.log({"rung3/increments_%s" % label: table})
        run.summary["oracle_sum_%s" % label] = total
    run.summary.update({
        "our_curve_boundary4_share": 0.262,
        "board_curve_boundary4_share": 0.193,
        "even_split_share": 1.0 / 7.0,
        "verdict": "on the BOARD curve the oracle gap is nearly evenly "
                   "spread, which fires the rung-3 stop rule; on OUR curve "
                   "boundary 4 carries 26 percent and is the only boundary "
                   "where the cost-aware oracle declines an acceptable draft",
    })
    run.finish()


def log_rung4() -> None:
    fine = load("rung3-ours-tier4-fine.json")
    predict = load("rung4-prediction.json")
    if fine is None or predict is None:
        print("rung4: no artifact, skipping")
        return
    run = start("e134-rung4-pass-boundary-price", {
        "scored_surface_changed": True,
        "arm": "pb6",
        "verify_width": predict["width"],
        "tier": predict["tier"],
        "curve": predict["curve"],
        "validation": "leave-one-prompt-out tier selection, 6 seeds",
        "rule_79": "a local timing leg cannot validate a depth price change",
    })
    grid = wandb.Table(columns=["tier", "median_pct", "sd"])
    for name, entry in sorted(fine["arms"].items()):
        if name.startswith("tierprice@"):
            grid.add_data(float(name.split("@")[1]), entry["mean"],
                          entry["sd"])
    run.log({"rung4/tier_grid": grid})

    arms = wandb.Table(columns=["arm", "median_pct", "sd"])
    for name in ("ship", "pb5", "pb6", "pb7", "pbfit", "rankedprice"):
        entry = fine["arms"].get(name)
        if entry:
            arms.add_data(name, entry["mean"], entry["sd"])
    run.log({"rung4/existing_arms": arms})

    prompts = wandb.Table(columns=["prompt", "candidate_time_ratio", "sd",
                                   "ship_raw", "arm_raw",
                                   "median_pct_if_this_prompt_fails"])
    for name, entry in sorted(predict["per_prompt"].items()):
        prompts.add_data(name, entry["ratio"], entry["sd"],
                         entry.get("ship_raw"), entry.get("arm_raw"),
                         predict["jackknife_median_pct"].get(name))
    run.log({"rung4/per_prompt_prediction": prompts})

    lofo = fine.get("cliff_lofo", {})
    run.summary.update({
        "e134_replayed_ranked_median_pct": lofo.get("held_out"),
        "e134_replayed_ranked_median_pct_in_sample": lofo.get("in_sample"),
        "e134_replayed_ranked_median_pct_sd": lofo.get("sd"),
        "predicted_receipt_score": predict["arm_median_raw"],
        "shipped_receipt_score": predict["ship_median_raw"],
        "predicted_published_median_pct":
            predict["published_median_pct"]["mean"],
        "predicted_published_median_z": predict["published_median_pct"]["z"],
        "predicted_unweighted_mean_pct":
            predict["unweighted_mean_pct"]["mean"],
        "predicted_unweighted_mean_z": predict["unweighted_mean_pct"]["z"],
        "worst_single_prompt_jackknife_pct":
            min(predict["jackknife_median_pct"].values()),
        "flat_level_control_pct": -0.2970,
        "placebo_boundary3_held_out_pct": 0.0,
        "placebo_boundary5_held_out_pct": 0.0,
        "board_curve_at_tier_1p50_pct": -1.6469,
        "onepass67_curve_at_tier_1p50_pct": -0.8963,
        "anchor_pbfit_replayed_pct": fine["arms"]["pbfit"]["mean"],
        "anchor_pbfit_measured_gpu_pct": -3.500,
        "anchor_rankedprice_replayed_pct": fine["arms"]["rankedprice"]["mean"],
        "anchor_rankedprice_e128_published_pct": -2.8508,
        "verdict": "pb6 is the implementable proposal. It is fitted to one "
                   "QMV dispatch table: the same arm reads -0.90 percent on "
                   "the curve that follows thorfinn's one-pass table. The "
                   "replayer reproduces E128 rankedprice to 0.25 pp but does "
                   "NOT reproduce E68's measured pbfit result, so the "
                   "instrument is not externally calibrated.",
    })
    run.finish()


def log_item0() -> None:
    data = load("item0-warm-arms.json")
    if data is None:
        print("item0: no artifact, skipping")
        return
    legs = data["legs"]
    records = [rec for arms in legs.values() for rec in arms.values()]
    entry = [float(r["gpu_temp_entry_c"]) for r in records]
    exit_ = [float(r["gpu_temp_exit_c"]) for r in records]
    run = start("e134-item0-warm-parity-arms", {
        "scored_surface_changed": True,
        "timing_valid": True,
        "harness": "local",
        "arms": "base, wnorm, wprefetch, all",
        "prompts": "beagle_a, plutarch_lives",
        "reps": 2,
        "counterbalancing": "ABBA within one session",
        "cool_gate_mode": "MLXFAST_LOCAL_COOL_GATE=0, standing permitted mode",
        "gpu_temp_entry_c_min": min(entry),
        "gpu_temp_entry_c_max": max(entry),
        "gpu_temp_entry_c_spread": max(entry) - min(entry),
        "gpu_temp_exit_c_min": min(exit_),
        "gpu_temp_exit_c_max": max(exit_),
        "legs": len(records),
    })

    metrics = ("spt", "E1_width_matched_us", "first_block_s", "p50_block_s",
               "seed_prefill_s")
    contrasts = wandb.Table(columns=[
        "contrast", "metric", "mean_delta", "mean_pct", "sd_pct", "se_pct",
        "pairs", "all_same_sign"])
    for name in ("wnorm-base", "wprefetch-base", "all-base", "all-wnorm"):
        block = data["contrasts"].get(name)
        if block is None:
            continue
        for metric in metrics:
            row = block.get(metric)
            if row is None:
                continue
            contrasts.add_data(name, metric, row["mean_delta"],
                               row["mean_pct"], row["sd_pct"], row["se_pct"],
                               row["n"], row["all_same_sign"])
    run.log({"item0/contrasts": contrasts})

    per_leg = wandb.Table(columns=[
        "leg", "rep", "arm", "seconds_per_token", "round1_excess_us",
        "first_block_s", "p50_block_s", "mean_draft", "rounds",
        "gpu_temp_entry_c", "gpu_temp_exit_c", "all_tokens_matched",
        "residual_divergence_count"])
    for key in sorted(legs):
        for arm in sorted(legs[key]):
            rec = legs[key][arm]
            per_leg.add_data(
                rec["leg"], rec["rep"], arm, rec["spt"],
                rec["E1_width_matched_us"], rec["first_block_s"],
                rec["p50_block_s"], rec["mean_draft"], rec["round_count"],
                float(rec["gpu_temp_entry_c"]), float(rec["gpu_temp_exit_c"]),
                rec["all_tokens_matched"], rec["residual_divergence_count"])
    run.log({"item0/legs": per_leg})

    spt = {k: v["spt"] for k, v in data["contrasts"].items()
           if "spt" in v}
    run.summary.update({
        "wnorm_minus_base_spt_pct": spt["wnorm-base"]["mean_pct"],
        "wnorm_minus_base_spt_sd_pct": spt["wnorm-base"]["sd_pct"],
        "wprefetch_minus_base_spt_pct": spt["wprefetch-base"]["mean_pct"],
        "wprefetch_minus_base_spt_sd_pct": spt["wprefetch-base"]["sd_pct"],
        "all_minus_base_spt_pct": spt["all-base"]["mean_pct"],
        "all_minus_base_spt_sd_pct": spt["all-base"]["sd_pct"],
        "wnorm_minus_base_round1_excess_pct":
            data["contrasts"]["wnorm-base"]["E1_width_matched_us"]["mean_pct"],
        "wprefetch_minus_base_round1_excess_pct":
            data["contrasts"]["wprefetch-base"]["E1_width_matched_us"]
            ["mean_pct"],
        "exactness_failures": len(data["exactness_failures"]),
        "f8_item4_regression_gate_pct": 0.30,
        "f8_item4_regression_gate_fired": True,
        "verdict": "CLOSE ITEM 0 AS A REGRESSION. Every warm-parity arm is "
                   "slower than the base on absolute candidate seconds per "
                   "token, and W-NORM makes round 1 WORSE by +24.9 percent, "
                   "which is the opposite sign to the hypothesis. The "
                   "W-PREFETCH regression of +0.48 percent fires the item-4 "
                   "gate. The regression is not drafting-scaled: it is larger "
                   "on plutarch_lives, which barely drafts, so it is not "
                   "confined to the prefix-reject restore path.",
    })
    run.finish()


def log_item2() -> None:
    curve = load("item2-measured-curve.json")
    tier = load("item2-tier4-measured-per_drafting_round.json")
    predict = load("item2-prediction-measured.json")
    if curve is None or tier is None or predict is None:
        print("item2: no artifact, skipping")
        return
    best = curve["best_form"]
    run = start("e134-item2-measured-curve-refit", {
        "scored_surface_changed": False,
        "receipt": curve["receipt"],
        "receipt_note": curve["receipt_note"],
        "best_form": best,
        "arm": "pb6",
        "verify_width": predict["width"],
        "tier": predict["tier"],
        "validation": "leave-one-prompt-out tier selection, 6 seeds; "
                      "leave-one-prompt-out and bootstrap refits",
        "attachment_gate": curve["attachment_gate"],
    })

    fits = wandb.Table(columns=[
        "form", "uniform_us", "uniform_se", "delta_round_us_6",
        "delta_round_us_6_se", "delta_round_us_7", "delta_round_us_7_se",
        "rms_residual_us", "residual_sigma_us", "r2"])
    for name, fit in sorted(curve["fits"].items()):
        beta, se = fit["beta"], fit["se"]
        fits.add_data(name, beta[0], se[0], beta[1], se[1], beta[2], se[2],
                      fit["rms_residual_us"], fit["residual_sigma_us"],
                      fit["r2"])
    run.log({"item2/inversion": fits})

    shapes = wandb.Table(columns=[
        "form", "rows", "round_us", "step_us", "step_ratio_to_shallow"])
    for name in sorted(curve["curves"]):
        shape = curve["curves"][name]["shape"]
        rows = shape["rows"]
        steps = shape["steps"]
        ratios = shape["ratios"]
        for index, row in enumerate(rows):
            shapes.add_data(
                name, row, shape["round_us"][index],
                steps[index - 1] if index else None,
                ratios[index - 1] if index else None)
    run.log({"item2/curves": shapes})

    masses = wandb.Table(columns=["prompt", "rounds", "drafting_share",
                                  "mass_5", "mass_6", "mass_7", "mass_8"])
    for name, block in sorted(curve["width_masses"].items()):
        mass = block["by_width"]
        masses.add_data(name, block["rounds"], block["drafting_share"],
                        mass.get("5"), mass.get("6"), mass.get("7"),
                        mass.get("8"))
    run.log({"item2/replayed_width_mass": masses})

    lopo_table = wandb.Table(columns=[
        "form", "held_out_prompt", "argmax_boundary", "ratio_at_4",
        "uniform_us", "delta_round_us_6", "delta_round_us_7"])
    for name, block in sorted(curve["leave_one_prompt_out"].items()):
        for refit in block:
            lopo_table.add_data(
                name, refit["held_out"], refit["argmax_boundary"],
                refit["ratio_at_4"], refit["uniform"],
                refit["delta_round_us_6"], refit["delta_round_us_7"])
    run.log({"item2/leave_one_prompt_out_refits": lopo_table})

    grid = wandb.Table(columns=["tier", "median_pct", "sd"])
    for name, arm in sorted(tier["arms"].items()):
        if name.startswith("tierprice@"):
            grid.add_data(float(name.split("@")[1]), arm["mean"], arm["sd"])
    run.log({"item2/tier_grid": grid})

    prompts = wandb.Table(columns=["prompt", "candidate_time_ratio", "sd",
                                   "median_pct_if_this_prompt_fails"])
    for name, entry in sorted(predict["per_prompt"].items()):
        prompts.add_data(name, entry["ratio"], entry["sd"],
                         predict["jackknife_median_pct"].get(name))
    run.log({"item2/per_prompt_prediction": prompts})

    robust = wandb.Table(columns=["form", "noise", "p_argmax_is_4",
                                  "ratio_at_4_p2.5", "ratio_at_4_p50",
                                  "ratio_at_4_p97.5"])
    for name, block in sorted(curve["bootstrap"].items()):
        for noise in ("leg_noise", "residual_noise"):
            draw = block.get(noise)
            if draw is None:
                continue
            ratio = draw["ratio_at_4"]
            robust.add_data(name, noise, draw["p_argmax_is_4"],
                            ratio["p2.5"], ratio["p50"], ratio["p97.5"])
    run.log({"item2/argmax_robustness": robust})

    lofo = tier["cliff_lofo"]
    shape = curve["curves"][best]["shape"]
    pre = curve["curves"]["pre_arm"]["shape"]
    run.summary.update({
        "e134_replayed_ranked_median_pct": lofo["held_out"],
        "e134_replayed_ranked_median_pct_in_sample": lofo["in_sample"],
        "e134_replayed_ranked_median_pct_sd": lofo["sd"],
        "pre_arm_replayed_ranked_median_pct": 2.3422,
        "argmax_boundary": shape["argmax_boundary"],
        "pre_arm_argmax_boundary": pre["argmax_boundary"],
        "step_ratio_at_boundary4": shape["ratios"][3],
        "pre_arm_step_ratio_at_boundary4": pre["ratios"][3],
        "leave_one_prompt_out_refits_keeping_boundary4": 24,
        "leave_one_prompt_out_refits": 24,
        "predicted_receipt_score": predict["arm_median_raw"],
        "shipped_receipt_score": predict["ship_median_raw"],
        "predicted_published_median_pct":
            predict["published_median_pct"]["mean"],
        "predicted_published_median_z": predict["published_median_pct"]["z"],
        "predicted_unweighted_mean_pct":
            predict["unweighted_mean_pct"]["mean"],
        "predicted_unweighted_mean_z": predict["unweighted_mean_pct"]["z"],
        "worst_single_prompt_jackknife_pct":
            min(predict["jackknife_median_pct"].values()),
        "flat_level_control_pct": -0.2987,
        "placebo_boundary3_held_out_pct": 0.0,
        "placebo_boundary5_held_out_pct": 0.0,
        "board_curve_at_tier_pct": -1.6469,
        "verdict": "CONFIRM. The boundary-4 cliff kept 110 to 114 percent of "
                   "its pre-arm magnitude, because width 7 became cheaper "
                   "while width 6 did not. Boundary 4 is the argmax in 24 of "
                   "24 leave-one-prompt-out refits and in 5997 of 6000 "
                   "bootstrap draws. Tier stays 1.45. Two honest limits: the "
                   "individual width-6 and width-7 coefficients are NOT "
                   "resolved, only their difference, and the fitted residual "
                   "sigma of 133 us exceeds the 40 to 58 us leg noise, so the "
                   "three-parameter model is misspecified.",
    })
    run.finish()


def log_rung6() -> None:
    data = load("rung6-query-contiguity.json")
    if data is None:
        print("rung6: no artifact, skipping")
        return
    run = start("e134-rung6-query-contiguity", {
        "harness": data["harness"],
        "scored_surface_changed": False,
        "runs": "no GPU legs; Swift runtime test reads real MLX strides",
        "validation": "two positive controls that prove the predicate "
                      "can fail",
    })
    table = wandb.Table(columns=[
        "site", "qL", "shape", "strides", "row_contiguous",
        "q_copy_unless", "query_transposed", "kernel"])
    for row in data["measurements"]:
        table.add_data(
            row["site"], str(row["qL"]), str(row["shape"]),
            str(row["strides"]), row["row_contiguous"],
            row["q_copy_unless"], row["query_transposed"], row["kernel"])
    run.log({"rung6/query_contiguity": table})
    controls = wandb.Table(columns=["control", "detail", "passed"])
    for row in data["controls"]:
        controls.add_data(row["name"], row["detail"], row["passed"])
    run.log({"rung6/controls": controls})
    follow = data["named_follow_up"]
    run.summary.update({
        "hypothesis": data["hypothesis"],
        "verdict": data["verdict"],
        "distinct_kernels_observed": 1,
        "all_widths_compile_qnt": True,
        "qL23_later_window_warm_value_pct": 0.0,
        "gpu_legs": 0,
        "follow_up_hidden_copies_per_round": 32,
        "follow_up_rough_order_pct_of_round":
            follow["rough_order_pct_of_round"],
        "follow_up_priced": follow["priced"],
        "follow_up_in_scope": follow["in_scope"],
        "verdict_text": data["conclusion"],
    })
    run.finish()


def log_item4() -> None:
    data = load("item4-shipped-population.json")
    if data is None:
        print("item4: no artifact, skipping")
        return
    run = start("e134-item4-shipped-population", {
        "harness": "local",
        "scored_surface_changed": False,
        "runs": "2 new shipped-policy legs, medicine_hist and "
                "essays_montaigne, against the 12-fixture forced-depth "
                "archive",
        "validation": "fit-free AUC with exact positive and negative counts",
    })
    pop = wandb.Table(columns=[
        "fixture", "prompt", "rounds", "mean_offered_depth",
        "mean_accepted", "row_count_bad", "sched_max_abs_error"])
    for name in sorted(data["shipped"]):
        rec = data["shipped"][name]
        pop.add_data(
            name, str(rec["prompt"]), rec["rounds"],
            rec["mean_offered_depth"], rec["mean_accepted"],
            rec["gate"]["row_count_bad"], rec["gate"]["sched_max_abs_error"])
    run.log({"item4/population": pop})

    inputs = data["reported_inputs"]
    short = ["margin", "ema_d", "reach_shipped", "km_reach", "pm_min",
             "pm_at_d"]
    table = wandb.Table(columns=[
        "population", "fixture", "boundary", "observations", "npos", "nneg",
        "accept_rate", "resolved"] + short)
    for pop_name in ("shipped", "forced"):
        for fixture in data["focus"]:
            rec = data[pop_name].get(fixture)
            if rec is None:
                continue
            for row in rec["boundaries"]:
                cells = [row["auc"].get(k) for k in short]
                first = next((c for c in cells if c is not None), None)
                table.add_data(
                    pop_name, fixture, row["depth"], row["obs"],
                    first["npos"] if first else 0,
                    first["nneg"] if first else 0,
                    row["accept_rate"], row["usable"],
                    *[c["value"] if c else None for c in cells])
    run.log({"item4/per_boundary_auc": table})

    med_s = {r["depth"]: r for r in data["shipped"]["medicine_hist"]
             ["boundaries"]}
    med_f = {r["depth"]: r for r in data["forced"]["medicine_hist"]
             ["boundaries"]}
    ess_s = {r["depth"]: r for r in data["shipped"]["essays_montaigne"]
             ["boundaries"]}
    run.summary.update({
        "reported_inputs": inputs,
        "min_obs_for_resolution": data["min_obs"],
        "shipped_medicine_b4_nneg":
            med_s[4]["auc"]["margin"]["nneg"],
        "shipped_medicine_b4_margin_auc":
            med_s[4]["auc"]["margin"]["value"],
        "shipped_medicine_b4_margin_lo":
            med_s[4]["auc"]["margin"]["lo"],
        "shipped_medicine_b4_margin_hi":
            med_s[4]["auc"]["margin"]["hi"],
        "forced_medicine_b4_margin_auc":
            med_f[4]["auc"]["margin"]["value"],
        "forced_medicine_b4_margin_hi":
            med_f[4]["auc"]["margin"]["hi"],
        "shipped_essays_rounds_reaching_depth5": ess_s[5]["obs"],
        "shipped_essays_rounds_reaching_depth6": ess_s[6]["obs"],
        "verdict": "The shipped population does NOT rescue the information "
                   "hypothesis; it starves it. Where both populations "
                   "resolve a boundary they agree. The shipped population "
                   "cannot resolve the deep boundary at all: "
                   "essays_montaigne reaches depth 5 once in 146 rounds and "
                   "never reaches 6, and medicine_hist carries only 2 "
                   "negatives at boundary 4, so its AUC spans "
                   "[-0.13, 0.70]. The one boundary-4 cell that IS resolved "
                   "is the forced medicine_hist cell, and it stays "
                   "anti-predictive at 0.2281 [0.0934, 0.3627]. Rung 1 "
                   "stands.",
    })
    run.finish()


def log_item5() -> None:
    diff = load("item5-rule108-diff.json")
    presubmit = load("item5-presubmit.json")
    if diff is None:
        print("item5: no artifact, skipping")
        return
    run = start("e134-item5-pb6-candidate", {
        "harness": "ranked-model",
        "scored_surface_changed": True,
        "arm": "pb6",
        "pass_boundary_verify_width": 6,
        "pass_boundary_tier_factor": 1.45,
        "runs": "no ranked receipt yet; Rule 79 means only a ranked run "
                "validates this change",
        "validation": "static gates plus a 512-token exactness check",
    })
    table = wandb.Table(columns=[
        "direction", "mechanism", "our_price_pct", "confidence", "credited",
        "our_price_source"])
    for key in ("deletes", "adds"):
        for row in diff[key]:
            table.add_data(
                key, row["mechanism"], row["our_price_pct"],
                row.get("confidence", "n/a"), row["credited"],
                row["our_price_source"])
    run.log({"item5/rule108_diff": table})

    frontier, crown = diff["frontier"], diff["our_crown"]
    summary = {
        "rule108_held_value_of_deleted_set_pct":
            diff["held_value_of_deleted_set_pct"],
        "rule108_effect_of_deleting_them_pct":
            diff["effect_of_deleting_them_pct"],
        "rule108_effect_of_adding_them_pct":
            diff["effect_of_adding_them_pct"],
        "rule108_net_our_instrument_pct": diff["net_pct"],
        "rule108_net_conservative_pct": diff["net_conservative_pct"],
        "frontier_submission": frontier["id"],
        "frontier_solver": frontier["solver"],
        "frontier_score": frontier["score"],
        "frontier_source_ref": frontier["source_ref"],
        "our_crown_submission": crown["id"],
        "our_crown_score": crown["score"],
        "gap_to_frontier_pct":
            100.0 * (frontier["score"] - crown["score"]) / crown["score"],
        "frontier_contains_our_one_pass_qmv": False,
        "source_diff_available": diff["source_diff_available"],
        "predicted_published_median_pct": 2.4683,
        "predicted_worst_prompt_jackknife_pct": 1.13,
        "submitted_to_yukon": False,
    }
    if presubmit is not None:
        summary.update(presubmit["summary"])
        legs = wandb.Table(columns=[
            "fixture", "tokens", "all_tokens_matched",
            "residual_divergence_count", "rounds", "row_count_bad",
            "mean_offered_depth", "mean_accepted"])
        for row in presubmit["legs"]:
            legs.add_data(
                row["fixture"], row["tokens"], row["all_tokens_matched"],
                row["residual_divergence_count"], row["rounds"],
                row["row_count_bad"], row["mean_offered_depth"],
                row["mean_accepted"])
        run.log({"item5/exactness_legs": legs})
    run.summary.update(summary)
    run.finish()


def log_item1() -> None:
    fm1 = load("fm1-flipped-round-audit.json")
    grid = load("tier-grid.json")
    legs = load("leg-prediction.json")
    if fm1 is None:
        print("item1: no artifact, skipping")
        return
    run = start("e134-item1-fm1-flipped-round-audit", {
        "harness": fm1["harness"],
        "scored_surface_changed": False,
        "runs": "no new GPU legs; replay over recorded round-start state "
                "of the archived shipped legs",
        "validation": "replayed ship depth equals the recorded depth on "
                      "every scored round, and the tier-1.0 control flips 0",
        "cliff": fm1["cliff"],
        "tier": fm1["tier"],
    })

    decisive = fm1["decisive"]["5"]
    flipped = wandb.Table(columns=[
        "leg", "round", "offer", "d_ship", "d_cand", "acc", "tokens_before"])
    for row in fm1["flipped_rounds"]:
        flipped.add_data(
            row["leg"], row["round"], row["offer"], row["d_ship"],
            row["d_cand"], row["acc"], row["tokens_before"])
    run.log({"item1/flipped_rounds": flipped})

    per_leg = wandb.Table(columns=[
        "leg", "k", "flipped_n", "flipped_rate", "kept_n", "kept_rate",
        "rounds", "row_count_bad", "margin_identity_bad",
        "sched_max_abs_error"])
    for name, row in fm1["per_leg"].items():
        gate = fm1["legs"][name]
        per_leg.add_data(
            name, row["k"], row["flipped"]["n"], row["flipped"]["rate"],
            row["kept"]["n"], row["kept"]["rate"], gate["rounds"],
            gate["row_count_bad"], gate["margin_identity_bad"],
            gate["sched_max_abs_error"])
    run.log({"item1/per_leg": per_leg})

    priced = wandb.Table(columns=[
        "curve", "component", "pct", "spt_ship_us", "spt_cand_us"])
    for curve, parts in fm1["priced"].items():
        for component, row in parts.items():
            priced.add_data(curve, component, row["pct"], row["spt_ship"],
                            row["spt_cand"])
    run.log({"item1/priced_decomposition": priced})

    summary = {
        "hypothesis": "FM1: the rounds pb6 suppresses accept ABOVE the "
                      "population rate, so the replayer overpays pb6",
        "verdict": "REFUTED: the suppressed rounds accept far BELOW the "
                   "population rate, so the replayed price is a lower bound",
        "n_flipped": fm1["n_flipped"],
        "n_deepened": fm1["n_deepened"],
        "rounds_replayed": fm1["rounds"],
        "gates_pass": fm1["gates_pass"],
        "sched_fidelity_bad": fm1["sched_fidelity_bad"],
        "control_flips": fm1["control_flips"],
        "control_deepened": fm1["control_deepened"],
        "p_acc_ge5_given_flipped": decisive["conditional_flipped"]["rate"],
        "p_acc_ge5_given_kept": decisive["kept"]["rate"],
        "p_acc_ge5_eligible_population":
            fm1["replayer_booked"]["_summary"]["recorded_eligible"],
        "excess_vs_kept": decisive["excess_vs_kept"],
        "excess_vs_shipped": decisive["excess_vs_shipped"],
        "z_flipped_minus_kept": decisive["z_flipped_minus_kept"],
        "replayer_booked_equal_weight":
            fm1["replayer_booked"]["_summary"]["equal_weight"],
        "replayer_booked_asked_weight":
            fm1["replayer_booked"]["_summary"]["asked_weight"],
        "replayer_booked_like_for_like":
            fm1["replayer_booked"]["_summary"]["like_for_like"],
        "max_ship_pb6_booked_drift":
            fm1["replayer_booked"]["_summary"]["max_ship_pb6_drift"],
        "trajectory_fixed_suppression_pct":
            fm1["priced"]["measured"]["suppression_only"]["pct"],
        "trajectory_fixed_deepening_pct":
            fm1["priced"]["measured"]["deepening_only"]["pct"],
        "trajectory_fixed_net_pct": fm1["priced"]["measured"]["both"]["pct"],
        "f12_discount_limb_triggered": False,
    }

    if grid is not None:
        table = wandb.Table(columns=[
            "tier", "median_pct", "sd", "mean_depth", "accept_rate"])
        for row in grid["grid"]:
            table.add_data(row["weight"], row["median_pct"], row["sd"],
                           row["mean_depth"], row["accept_rate"])
        run.log({"item1/tier_grid": table})
        lofo = grid["cliff_lofo"]
        chosen = sorted({v for picks in lofo["chosen"].values()
                         for v in picks})
        summary.update({
            "tier_grid_points": [row["weight"] for row in grid["grid"]],
            "tier_grid_median_pct": [row["median_pct"] for row in grid["grid"]],
            "tier_lofo_chosen_values": chosen,
            "tier_lofo_unanimous_at_1_45": chosen == [1.45],
            "tier_lofo_held_out_pct": lofo["held_out"],
            "tier_lofo_sd": lofo["sd"],
            "tier_upper_side_monotone_decline": True,
        })

    if legs is not None:
        table = wandb.Table(columns=[
            "leg", "model", "predicted_pct", "observed_pct"])
        for model, rows in legs["models"].items():
            for leg, value in rows.items():
                table.add_data(leg, model, value, legs["observed"][leg])
        run.log({"item1/leg_model_discrimination": table})
        measured = legs["models"]["measured"]
        summary.update({
            "legs_ranked_weighted_observed_pct":
                legs["ranked_weighted_observed"],
            "legs_ranked_weight_held": legs["ranked_weight_held"],
            "legs_thermally_matched": legs["thermally_matched"],
            "legs_abba_counterbalanced": legs["abba_counterbalanced"],
            "legs_rows_only_gets_medicine_sign_wrong": True,
            "legs_every_width_priced_model_overshoots": all(
                measured[leg] > legs["observed"][leg]
                for leg in ("beagle_a", "benchfixture", "essays_montaigne")),
            "legs_note": legs["note"],
        })

    run.summary.update(summary)
    run.finish()


def log_item5b() -> None:
    data = load("abba-s1.json")
    prereg = load("abba-preregistration.json")
    if data is None:
        print("item5b: no artifact, skipping")
        return
    run = start("e134-item5b-abba-same-binary-arm", {
        "harness": data["harness"],
        "scored_surface_changed": True,
        "arm_selector": "MLX_E134_DEPTH_PRICE_ARM",
        "compiled_default_arm": "pb6",
        "fixture": data["fixture"],
        "design": "ship pb6 pb6 ship, within-replicate contrast",
        "runs": "one worker binary and one commit across every leg",
        "validation": "Rule 114 witness: the arm is read from the run's own "
                      "round count, never from the environment",
    })

    legs = wandb.Table(columns=[
        "slot", "arm", "replicate", "position", "rounds", "mean_draft",
        "declared_rows", "decode_spt_us", "ranked_spt_us",
        "seed_prefill_s", "entry_c", "exit_c", "matched", "divergence",
        "witness"])
    for row in data["legs"]:
        legs.add_data(
            row["slot"], row["arm"], row["replicate"], row["position"],
            row["round_count"], row["mean_draft"], row["declared_rows"],
            row["decode_spt"] * 1e6, row["ranked_spt"] * 1e6,
            row["seed_prefill_seconds"], row["entry_c"], row["exit_c"],
            row["all_tokens_matched"], row["residual_divergence_count"],
            row["witness"])
    run.log({"item5b/legs": legs})

    contrasts = wandb.Table(columns=[
        "replicate", "ship_decode_us", "pb6_decode_us", "decode_pct",
        "ranked_pct", "delta_rounds"])
    for row in data["contrasts"]:
        contrasts.add_data(
            row["replicate"], row["ship_decode_spt"] * 1e6,
            row["pb6_decode_spt"] * 1e6, row["decode_pct"],
            row["ranked_pct"], row["delta_rounds"])
    run.log({"item5b/contrasts": contrasts})

    entries = data["entry_temp_c"]
    summary = {
        "hypothesis": "pb6 is faster than ship on one worker binary, when "
                      "the arm is chosen at run time and the order cancels "
                      "linear thermal drift",
        "verdict": data["verdict"],
        "decode_pct_mean": data["decode_pct_mean"],
        "decode_pct_sd": data["decode_pct_sd"],
        "ranked_pct_mean": data["ranked_pct_mean"],
        "ranked_pct_sd": data["ranked_pct_sd"],
        "replicates": len(data["contrasts"]),
        "identity_one_binary_one_commit": data["identity_ok"],
        "exactness_on_every_leg": data["exactness_ok"],
        "witness_mismatches": data["witness_mismatches"],
        "session_commit": " ".join(data["session_commit"]),
        "session_worker_sha256": " ".join(data["session_worker_sha256"]),
        "entry_temp_min_c": min(entries) if entries else None,
        "entry_temp_max_c": max(entries) if entries else None,
        "entry_temp_range_c": (max(entries) - min(entries)) if entries
                              else None,
        "timing_valid": True,
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "ruling": "one-sided: a local win proves nothing because this host's "
                  "width-6 cliff is steeper than the runner's; a local loss "
                  "blocks",
    }
    if prereg is not None:
        priors = prereg["priors_and_predictions"]
        summary.update({
            "prereg_written_before_any_leg_ran":
                prereg["written_before_any_leg_ran"],
            "prior_traced_screen_pct":
                priors["traced_screen_benchfixture_pct"],
            "prior_expectation": priors["expectation"],
            "prereg_decision_rule": json.dumps(prereg["decision_rule"]),
        })
    run.summary.update(summary)
    run.finish()


RUNS = {"rung1": log_rung1, "rung2": log_rung2, "rung3": log_rung3,
        "rung4": log_rung4, "item0": log_item0, "item2": log_item2,
        "rung6": log_rung6, "item4": log_item4, "item1": log_item1,
        "item5": log_item5, "item5b": log_item5b}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=tuple(RUNS))
    args = ap.parse_args()
    if not ART.exists():
        raise SystemExit("run from the repository root: %s is missing" % ART)
    for name in args.only or tuple(RUNS):
        RUNS[name]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
