#!/usr/bin/env python3
"""Log the E176 Q decode-consumer census to W&B.

Desk analysis plus one first-hand Metal device-metadata probe. It reads the
derived numbers from ``research/e176-q-census.json`` and the source census
recorded below. No GPU decode leg is timed here.
"""

import json

import wandb

CENSUS = "research/e176-q-census.json"

# harness=source. Wide quantized matmul call sites reached once per decode
# round by the scored worker, from Sources/MLXFastModel/Qwen35.swift.
CONSUMERS = [
    ("mlp.gate_up", 64, 5120, 34816, "Qwen35.swift:1737"),
    ("mlp.down", 64, 17408, 5120, "Qwen35.swift:1741"),
    ("gdn.in_proj", 48, 5120, 16480, "Qwen35.swift:1743"),
    ("gdn.out_proj", 48, 6144, 5120, "Qwen35.swift:1744"),
    ("fa.qkv", 16, 5120, 14336, "Qwen35.swift:1746"),
    ("fa.o_proj", 16, 6144, 5120, "Qwen35.swift:1747"),
    ("lm_head", 1, 5120, 248320, "Qwen35.swift:1737"),
]

# harness=source. get_qmv_batch_limit, backend/metal/quantized.cpp:84-125.
ARCH_TABLE = [
    ("arch_gen 13/14 (M3/M4 family)", "6/10/14", 6, "not the ranked runner"),
    ("arch_gen 16 size s (this host, probed)", "10/12/18", 10, "first-hand"),
    ("arch_gen >= 17 (ranked M5, _nax required)", "10/12/18", 10, "inferred"),
    ("arch_size d", "32/18/12", 12, "not this chip"),
]


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

    consumers = wandb.Table(
        columns=["site", "calls_per_round", "K", "N", "source",
                 "transpose", "M_decode_max", "vector_limit_min",
                 "routed_kernel", "consumes_Q"])
    for name, calls, k, n, src in CONSUMERS:
        consumers.add_data(name, calls, k, n, src, True, 9, 10,
                           "dispatch_qmv", False)

    arch = wandb.Table(
        columns=["branch", "limits_str", "vector_limit_min", "provenance"])
    for row in ARCH_TABLE:
        arch.add_data(*row)

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

    scalars = {
        "census/wide_qmv_calls_per_round": sum(c[1] for c in CONSUMERS),
        "census/distinct_shapes": len(CONSUMERS),
        "census/q_consuming_cells_at_M_le_5": 0,
        "census/q_consuming_cells_at_M_le_9": 0,
        "census/vector_limit_min_over_all_branches": 10,
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
             **scalars})
    run.summary.update(scalars)
    print("logged", run.url, run.id)
    run.finish()


if __name__ == "__main__":
    main()
