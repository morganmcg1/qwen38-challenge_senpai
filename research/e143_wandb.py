#!/usr/bin/env python3
"""Log the E143 channel census to W&B.

One run holds the whole E143 record: the assignment's named metrics, the
per-carrier channel table, the arm S recall curve, the Rule 101 positive
control dose-response, the arm F sigma sweep, the C2 fork's live ABBA
island-arm session, and the source JSON files as an artifact.

Usage: research/e143_wandb.py [--name ...] [--resume RUN_ID] [--c2 PATH]
                              [--offline]
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import wandb

PROJECT = "qwen38-mlx-challenge-senpai"
ENTITY = "wandb-applied-ai-team"
R0 = Path("research/e143-r0.json")
R1 = Path("research/e143-r1.json")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True,
                          check=True).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="e143-r0r1-channel-census")
    parser.add_argument("--resume", default=None,
                        help="extend this existing run instead of creating one")
    parser.add_argument("--c2", default=None,
                        help="research/e143-c2.json from the live ABBA session")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    r0 = json.loads(R0.read_text())
    r1 = json.loads(R1.read_text())
    c2 = json.loads(Path(args.c2).read_text()) if args.c2 else None
    carriers = r1["per_carrier"]

    config = {
        "experiment": "E143",
        "stage": "R0+R1",
        "harness": "local",
        "hypothesis": ("the first divergent position of an MTP round splits "
                       "into C-a unproposable, C-b dropped by the coarse "
                       "screen, C-c mis-ranked by the exact rerank, and C-d "
                       "the head preferring another continuation"),
        "base_sha": "892dc5e16c3ec741287b3ef213b52b97fd01a6c6",
        "commit": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "pr": 143,
        "capture": "e142/verifyrows, 6 seeds, 800 rounds, 3072 emitted tokens",
        "head_sha256": r0["seed_counters"][0]["head_sha256"],
        "miss_to_score_pct": 203.0,
        "gpu_seconds": r1["elapsed_seconds"],
        "arm_f_calibrated_sigma": r1["arm_f_calibrated_sigma"],
        "rule_121_anchor": "572b2cc4",
        "value_model": "research/e143_value.py, reproduces every f209 figure",
    }

    summary = {
        # the assignment's named metrics
        "e143_reachable_acceptance_pct_beagle":
            r1["primary"]["e143_reachable_acceptance_pct_beagle"],
        "e143_reachable_acceptance_median_pct":
            r1["primary"]["median_pct_rule121"],
        "e143_channel_pct_ca": carriers["beagle"]["ca"]["raw_ratio_pct"],
        "e143_channel_pct_cb": carriers["beagle"]["cb"]["raw_ratio_pct"],
        "e143_channel_pct_cc": carriers["beagle"]["cc"]["raw_ratio_pct"],
        "e143_channel_pct_cd": carriers["beagle"]["cd"]["raw_ratio_pct"],
        "e143_unresolved_fraction": r0["stage_b"]["unresolved_fraction_all"],
        # the fork
        "e143_fork_cb_plus_cc_median_pct": r1["fork"]["cb_plus_cc_median_pct"],
        "e143_fork_cb_upper_95_median_pct": r1["fork"]["cb_upper_95_median_pct"],
        "e143_fork_take_c2": int(r1["fork"]["take_c2_fallback"]),
        "e143_channel_d_share_of_in_vocab_misses":
            r1["fork"]["channel_d_share_of_in_vocabulary_misses"],
        "e143_channel_d_closes_acceptance_axis":
            int(r1["fork"]["channel_d_closes_acceptance_axis"]),
        # the instrument
        "e143_screen_recall_at_32": r1["arm_s_true_hidden"]["recall_at_32"],
        "e143_screen_loss_at_32": r1["arm_s_true_hidden"]["screen_loss_at_32"],
        "e143_coarse_rank_of_exact_argmax_max":
            r1["arm_s_true_hidden"]["coarse_rank_of_exact_argmax_max"],
        "e143_margin_in_coarse_error_sigmas_median":
            r1["arm_s_true_hidden"]["mechanism"][
                "margin_in_coarse_error_sigmas_median"],
        "e143_margin_in_coarse_error_sigmas_min":
            r1["arm_s_true_hidden"]["mechanism"][
                "margin_in_coarse_error_sigmas_min"],
        "e143_offline_argmax_matches_device":
            r1["arm_s_true_hidden"]["arm_p_offline_argmax_matches_device"],
        "e143_surrogate_usable": int(r1["arm_f_validation"]["surrogate_usable"]),
        # the corpus
        "e143_draft_trials": r0["emitted_token_census"]["draft_trial_total"],
        "e143_first_divergences": r0["emitted_token_census"]["divergent_rounds"],
        "e143_emitted_tokens": r0["emitted_token_census"]["emitted_token_total"],
    }
    for name, entry in carriers.items():
        summary[f"e143_{name}_per_step_p"] = entry["measured_per_step_p"]
        summary[f"e143_{name}_trials"] = entry["trials"]
        for tag in ("ca", "cb", "cc", "cd"):
            summary[f"e143_{name}_{tag}_rate"] = entry[tag]["rate"]
            summary[f"e143_{name}_{tag}_events"] = entry[tag]["events"]
            summary[f"e143_{name}_{tag}_raw_ratio_pct"] = entry[tag]["raw_ratio_pct"]

    tags = ["e143", "channel-census", "acceptance", "r0", "r1", "harness=local"]
    tables = {}
    if c2:
        config["stage"] = "R0+R1+C2"
        config["c2_design"] = c2["design"]
        config["c2_decode_tokens"] = c2["decode_tokens"]
        config["c2_gate_qualified_for_timing"] = c2["legs"][0][
            "gate_qualified_for_timing"]
        tags += ["c2", "island-arm", "abba"]
        summary.update({
            "e143_c2_realised_acceptance_delta_pp":
                c2["e143_c2_realised_acceptance_delta_pp"],
            "e143_c2_realised_per_step_p_delta_pp":
                c2["e143_c2_realised_per_step_p_delta_pp"],
            "e143_c2_time_pct_of_candidate_leg":
                c2["e143_c2_time_pct_of_candidate_leg"],
            "e143_c2_ranked_pct": c2["e143_c2_ranked_pct"],
            "e143_c2_replicate_spread_pct": c2["replicate_spread_pct"],
            "e143_c2_verdict": c2["verdict"],
            "e143_exactness_divergences": c2["e143_exactness_divergences"],
            "e143_c2_all_tokens_matched": int(c2["all_tokens_matched"]),
            "e143_c2_cross_arm_token_mismatches": sum(
                v["token_mismatches"] for v in
                c2["cross_arm_token_agreement"].values()),
            "e143_c2_rule_101_control_passed":
                int(c2["rule_101_wrong_arm_control_passed"]),
        })
        for arm, entry in c2["arm_summary"].items():
            summary[f"e143_c2_{arm}_mtp_seconds_per_token"] = entry[
                "mtp_seconds_per_token_mean"]
            summary[f"e143_c2_{arm}_accepted_draft_rate"] = entry[
                "accepted_draft_rate_mean"]
            summary[f"e143_c2_{arm}_rounds"] = entry["rounds_mean"]
            summary[f"e143_c2_{arm}_effective_mean_draft_len"] = entry[
                "effective_mean_draft_len_mean"]
        tables["c2_legs"] = wandb.Table(
            columns=["tag", "arm", "arm_witness", "mtp_seconds_per_token",
                     "serial_seconds_per_token", "mtp_decode_speedup", "rounds",
                     "effective_mean_draft_len", "accepted_draft_rate",
                     "per_step_p", "all_tokens_matched",
                     "residual_divergence_count", "gpu_temp_entry_c",
                     "gpu_temp_exit_c", "gate_qualified_for_timing"],
            data=[[e["tag"], e["arm"], e["arm_witness"],
                   e["mtp_seconds_per_token"], e["serial_seconds_per_token"],
                   e["mtp_decode_speedup"], e["rounds"],
                   e["effective_mean_draft_len"], e["accepted_draft_rate"],
                   e.get("per_step_p"), int(e["all_tokens_matched"]),
                   e["residual_divergence_count"], e["gpu_temp_entry_c"],
                   e["gpu_temp_exit_c"], e["gate_qualified_for_timing"]]
                  for e in c2["legs"]])

    init = dict(project=PROJECT, entity=ENTITY, name=args.name,
                job_type="analysis", tags=tags,
                mode="offline" if args.offline else "online", config=config)
    if args.resume:
        init.update(id=args.resume, resume="must")
    run = wandb.init(**init)

    tables.update({
        "positive_control": wandb.Table(
            columns=["extra_error_in_own_sigmas", "screen_loss_at_32",
                     "screen_loss_at_32_events", "coarse_rank_median"],
            data=[[r["extra_error_in_own_sigmas"], r["screen_loss_at_32"],
                   r["screen_loss_at_32_events"], r["coarse_rank_median"]]
                  for r in r1["positive_control_rule101"]]),
        "arm_f_sigma_sweep": wandb.Table(
            columns=["sigma", "simulated_miss_rate", "screen_loss_at_32",
                     "channel_b_rate", "channel_c_rate", "channel_d_rate",
                     "channel_b_events"],
            data=[[r["sigma"], r["simulated_miss_rate"], r["screen_loss_at_32"],
                   r["channel_b_rate"], r["channel_c_rate"], r["channel_d_rate"],
                   r["channel_b_events"]] for r in r1["arm_f_sweep"]]),
        "screen_recall_curve": wandb.Table(
            columns=["k", "recall"],
            data=[[k, r1["arm_s_true_hidden"][f"recall_at_{k}"]]
                  for k in (8, 16, 32, 64, 128, 256, 512, 1024)]),
        "channel_table": wandb.Table(
            columns=["carrier", "trials", "per_step_p", "ca_rate", "cb_rate",
                     "cc_rate", "cd_rate", "ca_raw_pct", "cb_raw_pct",
                     "cc_raw_pct", "cd_raw_pct"],
            data=[[name, e["trials"], e["measured_per_step_p"],
                   e["ca"]["rate"], e["cb"]["rate"], e["cc"]["rate"],
                   e["cd"]["rate"], e["ca"]["raw_ratio_pct"],
                   e["cb"]["raw_ratio_pct"], e["cc"]["raw_ratio_pct"],
                   e["cd"]["raw_ratio_pct"]]
                  for name, e in carriers.items()]),
        "per_seed": wandb.Table(
            columns=["seed", "carrier", "rounds", "divergences", "accepted",
                     "per_step_p", "effective_mean_draft_len",
                     "accepted_draft_rate", "all_tokens_matched",
                     "residual_divergence_count"],
            data=[[s["seed"], s["carrier"], s["round_count"],
                   s["divergent_round_count"], s["accepted_draft_total"],
                   s["per_step_p"], s["effective_mean_draft_len"],
                   s["reported_accepted_draft_rate"],
                   int(s["all_tokens_matched"]), s["residual_divergence_count"]]
                  for s in r0["seed_counters"]]),
    })
    run.log(tables)
    run.summary.update(summary)

    artifact = wandb.Artifact("e143-channel-census", type="analysis")
    artifact.add_file(str(R0))
    artifact.add_file(str(R1))
    if args.c2:
        artifact.add_file(args.c2)
    run.log_artifact(artifact)
    print(f"e143-wandb: {run.url}")
    print(f"e143-wandb: run_id {run.id}")
    run.finish()


if __name__ == "__main__":
    main()
