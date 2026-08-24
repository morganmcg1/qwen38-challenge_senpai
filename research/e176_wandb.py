#!/usr/bin/env python3
"""Log the E176 Q decode-consumer census to W&B.

Desk analysis plus one first-hand Metal device-metadata probe. It reads the
derived numbers from ``research/e176-q-census.json`` and the source census
recorded below. No GPU decode leg is timed here.
"""

import json

import wandb

CENSUS = "research/e176-q-census.json"
CELLS = "research/e176-cell-census.json"


def main():
    d = json.load(open(CENSUS))
    ch = d["channel_split"]
    fac = d["factor_by_channel"]
    tp = d["two_param"]

    run = wandb.init(
        entity="wandb-applied-ai-team", project="qwen38-mlx-challenge-senpai",
        name="e176-q-decode-consumer-census",
        job_type="desk-analysis",
        tags=["e176", "harness=ranked", "harness=source", "census",
              "null-calibration", "no-gpu"],
        config={
            "assignment": "e176-q-decode-consumer-census",
            "revision": "r0",
            "base_sha": "6368dc25265bd05461df4f070a18f4d97a18bc12",
            "receipts": {"A": "5a9f130a", "B": "180db842", "C": "fda590bb",
                         "D": "2c885d64", "crown": "ec24d591",
                         "K_cap4": "90c131dc"},
            "board_rows_512": d["board_rows_512"],
            "probe_architecture": "applegpu_g16s",
            "probe_arch_gen": 16,
            "probe_arch_size": "s",
            "probe_vector_limit": 10,
            "max_decode_width_M": 9,
        })

    cc = json.load(open(CELLS))
    branch_names = sorted(cc["branches"])
    consumers = wandb.Table(
        columns=["cell", "K", "N", "calls_per_round", "source", "transpose",
                 "M_decode_max", "routed_kernel", "consumes_Q"]
        + ["limit_" + b.replace(" ", "_") for b in branch_names])
    for c in cc["cells"]:
        consumers.add_data(c["cell"], c["K"], c["N"], c["calls_per_round"],
                           c["source"], c["transpose"], cc["max_decode_M"],
                           "dispatch_qmv", False,
                           *[c["limits"][b] for b in branch_names])

    arch = wandb.Table(
        columns=["branch", "arch_gen", "arch_size", "limit_all_cells",
                 "first_consuming_M", "q_calls_at_M9", "provenance"])
    for b in branch_names:
        v = cc["branches"][b]
        arch.add_data(b, v["arch_gen"], v["arch_size"],
                      cc["cells"][0]["limits"][b], v["first_consuming_M"] or 0,
                      v["q_consuming_calls_by_M"]["9"], v["provenance"])

    signs = wandb.Table(
        columns=["pair", "channel", "mean8_pct", "sd8_pct", "negative_of_8",
                 "max_abs_pct"])
    for pair, chans in sorted(d["sign_structure"].items()):
        for chan, v in sorted(chans.items()):
            signs.add_data(pair, chan, v["mean8"], v["sd8"],
                           v["negative_of_8"], v["max_abs"])

    prompts = sorted(ch["q_prefill_pct"])
    channel = wandb.Table(
        columns=["prompt", "prefill_share_pct", "Q_prefill_pct_BA",
                 "Q_decode_pct_BA", "BC_prefill_pct", "BC_decode_pct",
                 "BC_leg_pct", "sigma_pct", "BC_sigma",
                 "sigma_upper_bound_pct"])
    sig = d["null"]["per_prompt_pairwise_sigma_pct"]
    ub = d["null"]["per_prompt_pairwise_sigma_upper_bound_pct"]
    for p in prompts:
        s = sig[p]
        channel.add_data(p, ch["prefill_share_pct"][p],
                         ch["q_prefill_pct"][p], ch["q_decode_pct"][p],
                         ch["bc_prefill_pct"][p], ch["bc_decode_pct"][p],
                         d["q_vector"]["pct"][p], s,
                         d["q_vector"]["pct"][p] / s, ub[p])

    cap4 = d["cap4"]
    cap = wandb.Table(
        columns=["base", "published", "reconstructed", "plus_Q",
                 "plus_Q_delta_pct"])
    for tag, c in sorted(cap4["composed"].items()):
        cap.add_data(tag, c["published"], c["reconstructed"], c["plus_q"],
                     c["plus_q_delta_pct"])
    edl = wandb.Table(columns=["prompt", "edl_organizer_pure", "edl_cap4",
                               "edl_clip"])
    for p in prompts:
        edl.add_data(p, cap4["edl_A"][p], cap4["edl_K"][p],
                     cap4["edl_K"][p] - cap4["edl_A"][p])

    factors = wandb.Table(
        columns=["pair", "prefill_mean8_pct", "decode_mean7_pct",
                 "leg_mean7_pct", "prefill_only_prediction_pct",
                 "closure_residual_pct"])
    for pair, v in sorted(fac.items()):
        factors.add_data(pair, v["prefill_mean8_pct"],
                         v["decode_mean7_pct"], v["leg_mean7_pct"],
                         v["prefill_only_prediction_pct"],
                         v["closure_residual_pct"])

    models = wandb.Table(
        columns=["model", "beta", "se", "chi2", "dof", "essays_pred_pct",
                 "essays_obs_pct", "plutarch_pred_pct", "plutarch_obs_pct"])
    for name, m in sorted(d["models"].items()):
        models.add_data(name, m["beta"], m["se"], m["chi2"], 7,
                        m["pred_pct"]["essays"], d["q_vector"]["pct"]["essays"],
                        m["pred_pct"]["plutarch"],
                        d["q_vector"]["pct"]["plutarch"])

    oc = d["receipt_c_outlier"]
    scalars = {
        "census/quantized_calls_per_round": cc["total_calls_per_round"],
        "census/distinct_cells": len(cc["cells"]),
        "census/q_consuming_calls_at_M_le_5": 0,
        "census/q_consuming_calls_at_M_le_9": 0,
        "census/vector_limit_this_host": 10,
        "census/vector_limit_ranked_m5": 10,
        "outlier/bc_leg_predicted_pct": oc["bc_leg_predicted_pct"],
        "outlier/bc_leg_measured_pct": oc["bc_leg_measured_pct"],
        "outlier/unexplained_in_C_pct": oc["unexplained_pct"],
        "outlier/ca_decode_mean7_pct": oc["ca_decode_mean7_pct"],
        "outlier/ca_decode_sigma": oc["ca_decode_sigma"],
        "null/byte_identical_leg_negative_of_8":
            d["sign_structure"]["crown-A"]["leg"]["negative_of_8"],
        "null/byte_identical_leg_mean8_pct":
            d["sign_structure"]["crown-A"]["leg"]["mean8"],
        "null/byte_identical_leg_sd8_pct":
            d["sign_structure"]["crown-A"]["leg"]["sd8"],
        "q/prefill_negative_of_8":
            d["sign_structure"]["B-A"]["prefill"]["negative_of_8"],
        "q/decode_negative_of_8":
            d["sign_structure"]["B-A"]["decode"]["negative_of_8"],
        "null/common_pairwise_pct": d["null"]["common_pairwise_null_pct"],
        "null/cluster_common_offset_sd_pct":
            d["null"]["cluster_common_offset_sd_pct"],
        "null/byte_identical_pair_mean_pct":
            d["null"]["byte_identical_pair_mean_pct"],
        "null/byte_identical_pair_sd_pct":
            d["null"]["byte_identical_pair_sd_pct"],
        "q/prefill_mean8_pct": fac["B-A"]["prefill_mean8_pct"],
        "q/decode_mean7_pct": fac["B-A"]["decode_mean7_pct"],
        "q/leg_mean7_pct": fac["B-A"]["leg_mean7_pct"],
        "q/closure_residual_pct": fac["B-A"]["closure_residual_pct"],
        "q/bc_artifact_leg_mean7_pct": fac["B-C"]["leg_mean7_pct"],
        "fit/two_param_per_leg_s": tp["per_leg_s"],
        "fit/two_param_per_leg_se": tp["per_leg_se"],
        "fit/two_param_per_round_s": tp["per_round_s"],
        "fit/two_param_per_round_se": tp["per_round_se"],
        "fit/two_param_chi2": tp["chi2"],
        "fit/bc_vector_chi2_vs_zero": d["q_vector"]["chi2_zero"],
        "cap4/published": cap4["published"],
        "cap4/vs_A_pct": cap4["vs_A_pct"],
        "cap4/prefill_mean8_pct": cap4["prefill_mean8_pct"],
        "cap4/decode_mean7_pct": cap4["decode_mean7_pct"],
        "cap4/plus_Q_predicted": cap4["composed"]["K"]["plus_q"],
        "cap4/plus_Q_delta_pct": cap4["composed"]["K"]["plus_q_delta_pct"],
        "predict/A_plus_Q_published": cap4["composed"]["A"]["plus_q"],
        "ruling/compose_Q_with_cap4": 0,
    }

    run.log({"consumers": consumers, "arch_branches": arch,
             "channel_split": channel, "factor_by_channel": factors,
             "models": models, "cap4_composition": cap, "cap4_edl": edl,
             "sign_structure": signs,
             **scalars})
    run.summary.update(scalars)
    print("logged", run.url, run.id)
    run.finish()


if __name__ == "__main__":
    main()
