#!/usr/bin/env python3
"""Log the E37 r2 dispatched-verify-width (M) census to W&B.

E37 measures no time, so there is no metric series to stream. The durable
record is the per-arm width census, the traced-vs-untraced control, the exact
ranked round resolution, the assumption-free M>=6 bracket, the maxent scoring,
and the score payoff on OUR ranked row.

  python3 research/e37_wandb_log.py research/results/e37/r2-census.json
"""

from __future__ import annotations

import json
import pathlib
import sys

import wandb

PROJECT = "qwen38-mlx-challenge-senpai"
ENTITY = "wandb-applied-ai-team"
EVIDENCE = pathlib.Path("research/results/e37")
ORDER = ["plutarch", "drama", "travel", "beagle", "medicine", "essays",
         "republic", "botany"]


def main() -> None:
    d = json.loads(pathlib.Path(sys.argv[1]).read_text())
    census, metas, pay = d["census"], d["run_meta"], d["payoff"]
    m0 = metas["benchfixture"]

    run = wandb.init(
        entity=ENTITY, project=PROJECT,
        name="e37r2-draft-width-census-benchfixture",
        job_type="census",
        tags=["e37", "r2", "counts-only", "no-timing-claim", "qwen38-mtp-v1"],
        config={
            "assignment": "qwen38-r1-e37-draft-width-census-beagle-medicine",
            "revision": "r2", "pr": 42,
            "base_sha": "0491f9e5",
            "head_sha": m0["head_sha"], "worktree_dirty": m0["dirty"],
            "host": "Apple M4 Pro Mac16,11 applegpu_g16s (NOT ranked M5)",
            "physical_memory_gib": 48,
            "startup_memory_policy_applied": True,
            "decode_tokens": d["decode_tokens"],
            "offered_depth": d["offered_max_depth"],
            "phase_trace": 0,
            "head_tree_digest":
                "559b24ebca354018e4402fdb1f5af1afe5a0721bd2ebf04133500d846f7d5f71",
            "head_safetensors_sha256": m0["head_safetensors_sha256"],
            "worker_sha256": m0["worker_sha256"], "cli_sha256": m0["cli_sha256"],
            "metallib_fingerprint": m0["metallib_fingerprint"],
            "sdpa_width_wall_depth_cap": 5, "segmented_verify_depth_cap": 8,
            "segmented_streak_gate": 2, "max_legal_q_len": 5,
            "sdpa_wall_bites_at_q_len": 6,
            # Preserved verbatim: this run is not gate-qualified and no number
            # it produces may be compared as a timing measurement.
            "cool_gate_passed_real_gate": False,
            "gate_qualified_for_timing": False,
            "timing_claims_permitted": False,
        })

    width = wandb.Table(columns=[
        "arm", "M", "rounds", "round_share", "rows", "row_share"])
    arm_tbl = wandb.Table(columns=[
        "arm", "prompt_file", "rounds", "mean_M", "max_M", "offered",
        "accepted", "rejected", "accept_rate", "round_share_ge6",
        "row_share_ge6", "w6_round", "w6_row", "non_drafting_rounds",
        "R_plus_A", "all_tokens_matched", "residual_divergence_count",
        "started", "gpu_temp_in_out"])
    for arm in sorted(census):
        c, m = census[arm], metas[arm]
        rows_total = sum(int(k) * v for k, v in c["M_hist"].items())
        for k in sorted(c["M_hist"], key=int):
            n = c["M_hist"][k]
            width.add_data(arm, int(k), n, n / c["round_count"],
                           int(k) * n, int(k) * n / rows_total)
        arm_tbl.add_data(
            arm, m["prompt_file"], c["round_count"], c["mean_M"], c["max_M"],
            c["offered"], c["accepted"], c["rejected"], c["accept_rate"],
            c["round_share_ge6"], c["row_share_ge6"], c["w6_round"],
            c["w6_row"], c["non_drafting_round_count"],
            c["identity_R_plus_A"], c["all_tokens_matched"],
            c["residual_divergence_count"], m["started"],
            "%s -> %s" % (m["thermal_before"].split()[0],
                          m["thermal_after"].split()[0]))

    control = wandb.Table(columns=["arm", "rounds_untraced", "rounds_traced",
                                   "element_wise_identical"])
    for arm, t in d["traced_control"].items():
        control.add_data(arm, len(t["untraced"]), len(t["traced"]),
                         t["untraced"] == t["traced"])

    a, b = d["rho_fit"]["a"], d["rho_fit"]["b"]
    ranked = wandb.Table(columns=[
        "prompt", "n", "R", "D", "A", "alpha", "rho", "mean_M",
        "tokens_per_round", "rho_residual", "rejected_readings"])
    for nm in ORDER:
        r = d["ranked"][nm]
        ranked.add_data(nm, "%d/%d" % tuple(r["reduced"]),
                        r["R"], r["D"], r["A"], r["alpha"],
                        r["rho"], r["mean_M"], d["decode_tokens"] / r["R"],
                        r["rho"] - (a + b * r["mean_M"]),
                        str([x[0] for x in r["rejected"]]))

    bracket = wandb.Table(columns=[
        "prompt", "round_min", "round_max", "row_min", "row_max",
        "time_min", "time_max"])
    for nm, bk in d["bracket"].items():
        bracket.add_data(nm, bk["round"][0], bk["round"][1], bk["row"][0],
                         bk["row"][1], bk["time"][0], bk["time"][1])

    payoff = wandb.Table(columns=[
        "prompt", "our_raw_p", "top_raw_p", "headroom_pct",
        "saturated_score_pct"])
    for nm in ORDER:
        payoff.add_data(nm, pay["our_raw_p"][nm], pay["top_raw_p"][nm],
                        pay["headroom_pct"].get(nm),
                        pay["saturated_score_pct"].get(nm))

    phi = wandb.Table(columns=["time_share_phi", "score_gain_pct_per_1pct",
                               "sigmas"])
    for k, v in sorted(pay["phi_to_score_pct"].items()):
        phi.add_data(float(k), v, v / pay["sigma_score_pct"])

    run.log({"width_census": width, "arm_summary": arm_tbl,
             "trace_control": control, "ranked_resolution": ranked,
             "ranked_ge6_bracket": bracket, "score_payoff": payoff,
             "phi_sensitivity": phi})

    bf, med, nh = (census["benchfixture"], census["medicine"],
                   census["natural_history"])
    run.summary.update({
        "local/benchfixture_mean_M": bf["mean_M"],
        "local/benchfixture_max_M": bf["max_M"],
        "local/benchfixture_accept_rate": bf["accept_rate"],
        "local/benchfixture_row_share_ge6": bf["row_share_ge6"],
        "local/benchfixture_w6_row": bf["w6_row"],
        "local/benchfixture_rounds_at_M9": bf["M_hist"].get("9", 0),
        "local/medicine_proxy_mean_M": med["mean_M"],
        "local/medicine_proxy_row_share_ge6": med["row_share_ge6"],
        "local/beagle_proxy_mean_M": nh["mean_M"],
        "local/beagle_proxy_row_share_ge6": nh["row_share_ge6"],
        "local/proxy_fidelity_beagle": nh["mean_depth"] / 4.532710280374,
        "local/proxy_fidelity_medicine": med["mean_depth"] / 4.767676767677,
        "ranked/beagle_R": d["ranked"]["beagle"]["R"],
        "ranked/beagle_alpha": d["ranked"]["beagle"]["alpha"],
        "ranked/medicine_R": d["ranked"]["medicine"]["R"],
        "ranked/medicine_alpha": d["ranked"]["medicine"]["alpha"],
        "ranked/rho_fit_r2": d["rho_fit"]["r2"],
        "ranked/beagle_min_row_share_ge6": d["bracket"]["beagle"]["row"][0],
        "ranked/beagle_min_time_share_ge6": d["bracket"]["beagle"]["time"][0],
        "ranked/medicine_min_row_share_ge6": d["bracket"]["medicine"]["row"][0],
        "score/our_row": pay["our_score"],
        "score/board_top": pay["top_score"],
        "score/gap_R_pct": pay["gap_R_pct"],
        "score/gap_Rprime_pct": pay["gap_Rprime_pct"],
        "score/sigma_score_pct": pay["sigma_score_pct"],
        "score/identical_tree_pairs_on_board": pay["identical_tree_pairs"],
        "score/beagle_headroom_pct": pay["headroom_pct"]["beagle"],
        "score/medicine_headroom_pct": pay["headroom_pct"]["medicine"],
        "score/medicine_saturated_pct": pay["saturated_score_pct"]["medicine"],
        "score/beagle_saturated_pct": pay["saturated_score_pct"]["beagle"],
        "verdict/H1_falsified_on_prose_proxies": True,
        "verdict/M_ge_6_reachable_locally": True,
        "verdict/M7_9_reachable_locally": True,
        "verdict/r1_structural_claim_withdrawn": True,
        "verdict/trace_perturbs_counts": False,
    })

    art = wandb.Artifact("e37-width-census", type="census")
    for p in sorted(EVIDENCE.iterdir()):
        art.add_file(str(p))
    art.add_file("research/results/"
                 "qwen38-r1-e37-draft-width-census-beagle-medicine.md")
    run.log_artifact(art)
    print("logged %s" % run.url)
    run.finish()


if __name__ == "__main__":
    main()
