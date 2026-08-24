#!/usr/bin/env python3
"""Publish the E178 ranked receipt-channel distribution and replay pricing.

    usage: python3 research/e178_receipt_channel.py && python3 research/e178_wandb_log.py

Every number is a DESK estimate from official M5 receipts on the public Yukon
board. No GPU ran, no local timing leg was measured, and no value here is a
gate-qualified local measurement. harness=ranked on every metric.
"""
import json
import subprocess

import wandb

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
GROUP = "e178-receipt-channel-statistics"
BASE_SHA = "8da0b874d24bb2d627fc726aae7a31d90a9eeae8"
REPORT = "research/e178-receipt-channel.json"


def git(*args):
    return subprocess.run(["git"] + list(args), capture_output=True, text=True,
                          check=True).stdout.strip()


def main():
    d = json.load(open(REPORT))
    p, strict = d["primary"], d["strict"]
    sel, het, bands = d["selection"], d["heteroskedasticity"], d["sigma_by_score_band"]
    sched, drift, pr = d["schedule_reproducibility"], d["serial_drift"], d["replay_option"]
    impl = d["campaign_implications"]
    dirty = git("status", "--porcelain")

    run = wandb.init(
        entity=ENTITY, project=PROJECT, group=GROUP,
        name="e178-receipt-channel-statistics", job_type="desk-analysis",
        config={
            "experiment": "E178",
            "assignment_pr": 177,
            "base_sha": BASE_SHA,
            "commit": git("rev-parse", "HEAD"),
            "dirty_files": len(dirty.split("\n")) if dirty else 0,
            "harness": "ranked",
            "leg_kind": "desk-estimate-from-paid-receipts",
            "gpu_seconds_spent": 0,
            "timed_local_legs": 0,
            "cool_gate_passed_real_gate": None,
            "gate_qualified_for_timing": False,
            "board_rows_scored": d["board_rows_scored"],
            "identity_primary": "sha256 over benchmark.json editablePaths blobs",
            "identity_strict": "git tree sha of the whole submitted snapshot",
            "crown": pr["crown"], "crown_id8": "ec24d591",
            "our_A": pr["ours_A"], "our_A_id8": "5a9f130a",
            "decode_tokens": 512,
        })

    groups = wandb.Table(columns=[
        "group", "n", "mean_score", "date", "score_sd_pct", "mean_gap_h",
        "same_solver", "members"])
    for r in het["per_group"]:
        groups.add_data(r["group"], r["n"], r["mean_score"], r["date"],
                        r["sd_pct"], r["mean_gap_h"], r["same_solver"],
                        ",".join(r["members"]))
    run.log({"identical_tree_groups": groups})

    receipts = wandb.Table(columns=[
        "group", "n", "id8", "solver", "createdAt", "score", "score_dev_pct",
        "mtp_offset_pct", "serial_offset_pct", "decode_offset_pct",
        "raw_offset_pct", "mtp_per_prompt_spread_pct", "promotion"])
    for r in p["per_receipt"]:
        receipts.add_data(
            r["group"], r["n"], r["id8"], r["solver"], r["createdAt"], r["score"],
            100 * r["score_dev"], 100 * r["mtp"], 100 * r["serial"],
            100 * r["decode"] if r["decode"] is not None else None,
            100 * r["raw"], 100 * r["mtp_spread"], r["promotion"])
    run.log({"receipt_offsets": receipts})

    chan = wandb.Table(columns=[
        "identity", "channel", "sigma_pct", "dof", "per_prompt_spread_pct"])
    for name, res in (("editable-archive", p), ("exact-tree", strict)):
        for c, o in res["sigma"].items():
            chan.add_data(name, c, o["sigma_pct"], o["dof"],
                          o.get("per_prompt_spread_pct"))
    run.log({"channel_sigma": chan})

    band = wandb.Table(columns=["score_band", "n_groups", "n_receipts", "dof",
                                "score_sigma_pct", "ci95_lo", "ci95_hi",
                                "mtp_sigma_pct"])
    for b, o in bands.items():
        band.add_data(b, o["n_groups"], o["n_receipts"], o["dof"],
                      o["score_sigma_pct"], o["ci95_pct"][0], o["ci95_pct"][1],
                      o["mtp_sigma_pct"])
    run.log({"sigma_by_score_band": band})

    struct = wandb.Table(columns=["channel", "test", "slope", "t", "n"])
    for c, tests in p["structure"].items():
        if c == "adjacent_pairs_diff_group":
            continue
        for name, o in tests.items():
            struct.add_data(c, name, o["slope"], o["t"], o["n"])
    for name in ("log_sd_vs_score_level", "log_sd_vs_day", "log_sd_vs_mean_gap_h"):
        o = het[name]
        struct.add_data("group_sd", name, o["slope"], o["t"], o["n"])
    o = het["pairs"]["abs_diff_vs_gap"]
    struct.add_data("pair_abs_diff", "vs_gap_hours", o["slope"], o["t"], o["n"])
    run.log({"structure_tests": struct})

    day = wandb.Table(columns=["date", "n", "serial_ms_per_token", "sd_pct"])
    for date, v in drift["by_day"].items():
        day.add_data(date, v["n"], v["mean_ms_per_token"], v["sd_pct"])
    run.log({"serial_numerator_by_day": day})

    opt = wandb.Table(columns=["scenario", "mu", "sigma_pct", "z_to_crown",
                               "p_single", "p_best_of_2", "p_best_of_3",
                               "e_best_of_1", "e_best_of_2", "e_best_of_3"])
    for name, sc in pr["scenarios"].items():
        opt.add_data(name, sc["mu"], sc["sigma_pct"], sc["z_to_target"],
                     sc["p_single"], sc["p_best_of_k"]["2"], sc["p_best_of_k"]["3"],
                     sc["e_best_of_k"]["1"], sc["e_best_of_k"]["2"],
                     sc["e_best_of_k"]["3"])
    for name, sc in pr["mechanism_comparison"]["band"].items():
        opt.add_data("xsums_" + name, sc["mu"], sc["sigma_pct"], sc["z_to_target"],
                     sc["p_single"], sc["p_best_of_k"]["2"], None,
                     sc["e_best_of_k"]["1"], sc["e_best_of_k"]["2"], None)
    run.log({"replay_option_pricing": opt})

    ladder = wandb.Table(columns=["receipt", "id8", "score", "createdAt",
                                  "pct_vs_A_level", "z_vs_A_level",
                                  "resolved_at_2sigma"])
    for name, o in impl["receipts"].items():
        ladder.add_data(name, o["id8"], o["score"], o["createdAt"],
                        o["pct_vs_A_level"], o["z_vs_A_level"],
                        o["resolved_at_2sigma"])
    run.log({"ladder_vs_channel": ladder})

    draws = wandb.Table(columns=["id8", "solver", "score", "createdAt", "promotion"])
    for x in pr["draws"]:
        draws.add_data(x["id8"], x["solver"], x["score"], x["createdAt"],
                       x["promotion"] or "rejected")
    run.log({"a_lineage_draws": draws})

    head = pr["scenarios"]["drift_adjusted"]
    run.summary.update({
        "harness": "ranked",
        "channel/identity_groups": p["n_groups"],
        "channel/identity_receipts": p["n_receipts"],
        "channel/score_sigma_pct": p["sigma"]["score_dev"]["sigma_pct"],
        "channel/score_sigma_dof": p["sigma"]["score_dev"]["dof"],
        "channel/score_sigma_ci95_lo_pct": bands["3.6-9.9"]["ci95_pct"][0],
        "channel/mtp_sigma_pct": p["sigma"]["mtp"]["sigma_pct"],
        "channel/serial_sigma_pct": p["sigma"]["serial"]["sigma_pct"],
        "channel/decode_sigma_pct": p["sigma"]["decode"]["sigma_pct"],
        "channel/per_prompt_spread_pct": p["sigma"]["mtp"]["per_prompt_spread_pct"],
        "channel/sign_coherence_of_8": p["mtp_sign_coherence"]["mean_same_sign_of_8"],
        "channel/score_vs_raw_slope": p["score_vs_raw_slope"]["slope"],
        "channel/skew": p["standardised"]["score_dev"]["skew"],
        "channel/excess_kurtosis": p["standardised"]["score_dev"]["excess_kurtosis"],
        "channel/frac_abs_z_gt_2": p["standardised"]["score_dev"]["frac_abs_gt_2"],
        "channel/top_band_sigma_pct": bands["3.6-9.9"]["score_sigma_pct"],
        "channel/copier_only_sigma_pct": sel["copier_only_sigma"]["score"]["sigma_pct"],
        "selection/first_vs_rest_score_pct": sel["first_vs_rest_score_dev"]["diff_pct"],
        "selection/first_vs_rest_t": sel["first_vs_rest_score_dev"]["t"],
        "schedule/groups_with_identical_counters": sched["deterministic_schedule"]["n_groups"],
        "schedule/groups_with_varying_counters": sched["adaptive_schedule"]["n_groups"],
        "structure/adjacency_corr": p["structure"]["adjacent_pairs_diff_group"]["corr_all"],
        "drift/serial_pct_per_day_since_0821":
            100 * drift["multivariate_since_0821"]["coef"]["day"]["beta"],
        "drift/serial_t_since_0821": drift["multivariate_since_0821"]["coef"]["day"]["t"],
        "drift/serial_pct_per_day_campaign":
            100 * drift["multivariate"]["coef"]["day"]["beta"],
        "replay/a_draws": pr["n"],
        "replay/a_level_unselected": pr["mean_copier_draws"],
        "replay/a_level_today": head["mu"],
        "replay/p_single_beats_crown": head["p_single"],
        "replay/p_best_of_2": head["p_best_of_k"]["2"],
        "replay/p_best_of_3": head["p_best_of_k"]["3"],
        "replay/e_best_of_3": head["e_best_of_k"]["3"],
        "replay/nonparametric_upper_bound": pr["nonparametric_upper_bound"]["p_single_replay_is_max"],
        "mechanism/xsums_upper_p_single":
            pr["mechanism_comparison"]["band"]["upper"]["p_single"],
        "mechanism/gain_pct_for_p80": pr["mechanism_size_for_confidence"]["gain_pct_for_p80"],
        "implication/pair_contrast_sd_pct": impl["pair_contrast_sd_pct"],
        "implication/min_resolvable_2sigma_pct": impl["min_resolvable_2sigma_pct"]
        if "min_resolvable_2sigma_pct" in impl else impl["min_resolvable_2sigma_pair_pct"],
        "implication/finding452_z": impl["finding_452_recheck"]["z"],
        "implication/finding453_z": impl["finding_453_recheck"]["z"],
        "implication/per_prompt_understatement_factor":
            impl["per_prompt_understatement_factor"],
        "implication/e175_z_vs_A_level": impl["receipts"]["E175"]["z_vs_A_level"],
    })
    print("run:", run.url)
    run.finish()


if __name__ == "__main__":
    main()
