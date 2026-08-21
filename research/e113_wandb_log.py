#!/usr/bin/env python3
"""Publish the E113 depth-price boundary evidence to W&B.

E113 asked whether the post-E100 QMV dispatch group boundary at verify width 6
should be taught to `costModelDepth()`. Nothing was timed: every number is
arithmetic over published Yukon per-prompt receipts, the reused E106 per-width
census, and a replay of already-recorded rounds. `timing_valid`,
`cool_gate_passed_real_gate` and `gate_qualified_for_timing` are therefore
logged false verbatim and no leg is a score.

    usage: research/e113_wandb_log.py
"""

from __future__ import annotations

import json
import pathlib

import wandb

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
GROUP = "e113-depth-price-boundary"
NAME = "e113-rung01-depth-price-boundary"
HOST = "apple-m4-pro-applegpu_g16s-20core-48gib"

BASE_SHA = "25c33465efe6fc45123b69b2accd547bb56e0642"
RESULTS = pathlib.Path("research/e113-artifacts/rung01.json")
CONSOLE = pathlib.Path("research/e113-artifacts/rung01.txt")

RECEIPTS = ("b8b8b860", "44559d02")
TRACE = "e101ctl512"
ASSIGNED_ARMS = ("pb6", "pb6fit", "look6")


def gate_flags() -> dict[str, object]:
    return {
        "timing_valid": False,
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "gpu_seconds_used": 0,
        "instrument": "arithmetic over published receipts and recorded rounds",
    }


def identity() -> dict[str, object]:
    return {
        "experiment": GROUP,
        "base_sha": BASE_SHA,
        "host": HOST,
        "candidate_diff": "none, research/ only",
        "ranked_receipts": list(RECEIPTS),
        "replay_trace": TRACE,
        "replay_trace_base_sha": "9d837fc2",
        "replay_trace_tokens": 512,
        "local_cost_census_run": "19kgn6xi",
        "local_cost_census_timing_valid": False,
    }


def main() -> None:
    d = json.loads(RESULTS.read_text())
    pub = d["rung1b_published"]

    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        group=GROUP,
        job_type="depth-price-boundary-analysis",
        name=NAME,
        config={
            "question": (
                "E100 moved the QMV dispatch group boundary from verify width "
                "5 to width 6. Does refitting the uniform depth price in "
                "costModelDepth() to that boundary buy published score?"
            ),
            "kill_rule_0": (
                "stop if the realised schedule is within 0.20 % of the "
                "per-prompt cost-per-token optimum on 6 of 8 prompts"
            ),
            "kill_rule_1": (
                "stop if the best arm's predicted published gain is below "
                "0.20 %"
            ),
            "assigned_arms": list(ASSIGNED_ARMS),
            **identity(),
            **gate_flags(),
        },
        reinit=True,
    )

    # ---- deliverable: the two-frame marginal cost table ------------------
    marg = wandb.Table(columns=[
        "harness", "M", "partition", "round_us", "marginal_us",
        "vs_g1_step", "dispatch_groups", "provenance", "n_measured"])
    part = {1: "[1]", 2: "[2]", 3: "[3]", 4: "[4]", 5: "[5]",
            6: "[3+3]", 7: "[4+3]", 8: "[4+4]"}
    groups = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2}
    measured = set(d["local_measured_widths"])
    n_by_width = {int(k): v for k, v in d["e106_n_per_width"].items()}
    for harness, key in (("local", "local_current_curve"),
                         ("ranked", "ranked_f31_curve")):
        curve = {int(k): v for k, v in d[key].items()}
        step = curve[2] - curve[1]
        for M in range(1, 9):
            prev = curve.get(M - 1)
            marg.add_data(
                harness, M, part[M], curve[M],
                None if prev is None else curve[M] - prev,
                None if prev is None else (curve[M] - prev) / step,
                groups[M],
                "measured" if (harness == "local" and M in measured)
                else "INFERRED",
                n_by_width.get(M) if harness == "local" else None)
    run.log({"deliverable/marginal_cost_by_harness": marg})

    # ---- where the ranked cost boundary actually is -----------------------
    fits = wandb.Table(columns=["break_at_M", "a1", "c1", "a2", "c2",
                                "step_us", "max_abs_residual_pct"])
    for k, f in sorted(d["ranked_route_b_fits"].items()):
        fits.add_data(int(k.replace("break_M", "")), f["a1"], f["c1"],
                      f["a2"], f["c2"], f["step"], 100 * f["maxres"])
    sl = d["ranked_route_b_single_line"]
    fits.add_data(None, sl["a"], sl["c"], None, None, None,
                  100 * sl["maxres"])
    run.log({"rung0/ranked_route_b_two_line_fits": fits})

    # ---- rung 0: per-prompt optimum against the realised schedule ---------
    opt = wandb.Table(columns=[
        "harness", "prompt", "accept", "realised_mean_width", "d_star",
        "d_realised", "cpt_at_d_star_us", "cpt_at_d_realised_us", "gap_pct"])
    for harness, per in sorted(d["rung0_per_prompt"].items()):
        for prompt, r in sorted(per.items(), key=lambda kv: kv[1]["accept"]):
            opt.add_data(harness, prompt, r["accept"],
                         r["realised_mean_width"], r["d_star"],
                         r["d_realised"], r["cpt_at_d_star"],
                         r["cpt_at_d_realised"], r["gap_pct"])
    run.log({"rung0/per_prompt_optimum": opt})

    # ---- rung 1: trace replay --------------------------------------------
    tr = wandb.Table(columns=["treatment", "curve", "arm", "us_per_token",
                              "mean_depth", "vs_ship_pct"])
    for treatment, by_curve in sorted(d["rung1_trace"][TRACE].items()):
        for curve, by_arm in sorted(by_curve.items()):
            for arm, r in by_arm.items():
                tr.add_data(treatment, curve, arm, r["us_per_token"],
                            r["mean_depth"], r["vs_ship_pct"])
    run.log({"rung1/trace_replay": tr})

    # ---- rung 1b: per-prompt replay and predicted published median --------
    pp = wandb.Table(columns=["prompt", "arm", "mean_depth",
                              "ranked_us_per_token", "vs_ship_pct",
                              "clamp_share"])
    for prompt, by_arm in sorted(d["rung1b_per_prompt"].items()):
        for arm, r in by_arm.items():
            pp.add_data(prompt, arm, r["mean_depth"],
                        r["ranked_us_per_token"], r["vs_ship_pct"],
                        r["clamp_share"])
    run.log({"rung1b/per_prompt": pp})

    med = wandb.Table(columns=["arm", "predicted_published_median",
                               "vs_ship_pct", "is_assigned_arm"])
    for arm, r in pub.items():
        med.add_data(arm, r["median"], r["vs_ship_pct"], arm in ASSIGNED_ARMS)
    run.log({"rung1b/predicted_published_median": med})

    ctl = d["rung1_control"][TRACE]
    best_assigned = max(pub[a]["vs_ship_pct"] for a in ASSIGNED_ARMS)
    summary = {
        "verdict": "kill_rule_1_fires",
        "result_label": "not useful",
        "kill_rule_0_fires": any(
            v["kill_rule_0_fires"] for v in d["rung0_gaps"].values()),
        "kill_rule_1_fires": bool(d["kill_rule_1_fires"]),
        "best_assigned_arm_vs_ship_pct": best_assigned,
        "ship_predicted_published_median": pub["ship"]["median"],
        "official_median_of_receipt_b8b8b860": 3.33412148,
        "local_tier_ratio_into_width_6": d["local_tier_ratio_into_6"],
        "ranked_tier_ratio_into_width_6": d["ranked_tier_ratio_into_6"],
        "ranked_tier_ratio_into_width_5": d["ranked_tier_ratio_into_5"],
        "ranked_break_M5_max_residual_pct":
            100 * d["ranked_route_b_fits"]["break_M5"]["maxres"],
        "ranked_break_M6_max_residual_pct":
            100 * d["ranked_route_b_fits"]["break_M6"]["maxres"],
        "positive_control_rounds": ctl["rounds"],
        "positive_control_mismatches": ctl["mismatches"],
        "e106_census_n_at_M2": n_by_width.get(2),
        "e106_census_n_at_M5": n_by_width.get(5),
    }
    for arm, r in pub.items():
        summary["arm/%s/vs_ship_pct" % arm] = r["vs_ship_pct"]
        summary["arm/%s/predicted_published_median" % arm] = r["median"]
    run.summary.update(summary)

    art = wandb.Artifact("e113-depth-price-boundary", type="analysis")
    art.add_file(str(RESULTS))
    art.add_file(str(CONSOLE))
    art.add_file("research/e113_depth_boundary.py")
    art.add_file("research/e113-results.md")
    run.log_artifact(art)

    print("run", run.id, run.url)
    run.finish()


if __name__ == "__main__":
    main()
