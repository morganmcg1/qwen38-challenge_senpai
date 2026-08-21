#!/usr/bin/env python3
"""Publish the E114 operating-point evidence to W&B.

E114 asked whether the standing NA weights `0.024/0.275/0.667/0.034` are the
LOCAL operating point wearing a ranked label, and whether re-weighting the
recorded kernel arms to the published operating point changes any ranking.
Nothing was timed. Every number is arithmetic over published Yukon per-prompt
receipts, three traced histograms already in the repository, and recorded arm
tables, so `timing_valid`, `cool_gate_passed_real_gate` and
`gate_qualified_for_timing` are logged false verbatim and no leg is a score.

    usage: research/e114_wandb_log.py
"""

from __future__ import annotations

import json
import pathlib

import wandb

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
GROUP = "e114-operating-point"
NAME = "e114-rung01-arm-reweight-at-published-operating-point"
HOST = "apple-m4-pro-applegpu_g16s-20core-48gib"

BASE_SHA = "770a3ff2f8fbd1bb75d15e3c37ae3c5b076ebbcf"
ART = pathlib.Path("research/e114-artifacts")
RECEIPTS = ("b8b8b860", "44559d02", "51b9bf85")
NA_CELLS = ("2", "3", "4", "5")
SHAPES = ("maxent", "gt1", "gt2")


def gate_flags() -> dict[str, object]:
    return {
        "timing_valid": False,
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "gpu_seconds_used": 0,
        "instrument": "arithmetic over published receipts and recorded arms",
    }


def identity() -> dict[str, object]:
    return {
        "experiment": GROUP,
        "base_sha": BASE_SHA,
        "host": HOST,
        "candidate_diff": "none, research/ only",
        "ranked_receipt_priced": RECEIPTS[0],
        "ranked_receipts_cross_checked": list(RECEIPTS),
        "ground_truth_gt1_run": "19kgn6xi",
        "ground_truth_gt2_artifact": "research/out/e109-witness-w512",
        "ground_truth_gt3_artifact": "research/e99-artifacts/rung67.json",
        "standing_weights": "NA2 0.024 NA3 0.275 NA4 0.667 NA5 0.034",
    }


def rung0_tables(run, d: dict) -> None:
    rc = wandb.Table(columns=[
        "prompt", "chosen_R", "admissible_R", "cost_curve_pick", "agrees",
        "chosen_resid_pct"])
    for r in d["round_counts"]:
        rc.add_data(r["prompt"], r["chosen"], json.dumps(r["admissible"]),
                    r["cost_curve_pick"], r["agrees"], r["chosen_resid_pct"])
    run.log({"rung0/round_count_admissibility": rc})

    board = wandb.Table(columns=[
        "prompt", "rounds", "mean_draft_len", "mean_width", "accept",
        "zero_draft_rounds", "p_width1", "round_us", "mtp_us_per_token",
        "raw_ratio_of_means"])
    for name, p in sorted(d["prompts"].items()):
        board.add_data(name, p["rounds"], p["mean_draft_len"],
                       p["mean_width"], p["accept"], p["zero_draft_rounds"],
                       p["p_width1"], p["round_us"], p["mtp_us_per_token"],
                       p["raw"])
    run.log({"rung0/ranked_board_inputs": board})

    val = wandb.Table(columns=[
        "rate_table_harness", "ground_truth", "rounds", "mean_width",
        "estimator", "NA2", "NA3", "NA4", "NA5", "max_abs_err", "tol", "pass"])
    for frame, v in sorted(d["validation"].items()):
        for gt, cov in sorted(v["coverage"].items()):
            t = cov["w_true"]
            val.add_data(frame, gt, cov["rounds"], cov["mean_width"],
                         "TRACED TRUTH", t["2"], t["3"], t["4"], t["5"],
                         0.0, None, True)
            for est in ("maxent", "transport"):
                e = v["points"][est][gt]
                w = e["w"]
                val.add_data(frame, gt, cov["rounds"], cov["mean_width"], est,
                             w["2"], w["3"], w["4"], w["5"],
                             e["max_abs_err"], e["tol"], e["pass"])
            for edge in ("lo", "hi"):
                b = cov[edge]
                val.add_data(frame, gt, cov["rounds"], cov["mean_width"],
                             "identified bound %s" % edge,
                             *[b[na] for na in NA_CELLS],
                             None, None, cov["covered"])
    run.log({"rung0/validation_against_traced_truth": val})

    rec = wandb.Table(columns=[
        "rate_table_harness", "constraints", "prompt", "mean_width",
        "p_width1", "vertices", "widest_band",
        "maxent_NA2", "maxent_NA3", "maxent_NA4", "maxent_NA5",
        "transport_NA2", "transport_NA3", "transport_NA4", "transport_NA5",
        "lo_NA2", "hi_NA2", "lo_NA3", "hi_NA3", "lo_NA4", "hi_NA4",
        "lo_NA5", "hi_NA5"])
    for key, per in sorted(d["recovery"].items()):
        frame, constraints = key.split("_", 1)
        for prompt, r in sorted(per.items()):
            rec.add_data(
                frame, constraints, prompt, r["mean_width"], r["p_width1"],
                r["vertices"], r["widest_band"],
                *[r["maxent"][na] for na in NA_CELLS],
                *[r["transport"][na] for na in NA_CELLS],
                *[x for na in NA_CELLS for x in (r["lo"][na], r["hi"][na])])
    run.log({"rung0/recovered_na_weights_by_prompt": rec})


def rung1_tables(run, d: dict) -> None:
    mix = wandb.Table(columns=["prompt", "sensitivity_weight", "share"])
    tot = sum(d["prompt_mix"].values())
    for name, w in sorted(d["prompt_mix"].items(), key=lambda kv: -kv[1]):
        mix.add_data(name, w, w / tot)
    run.log({"rung1/published_score_sensitivity_mix": mix})

    wt = wandb.Table(columns=["rate_table_harness", "vector", "NA2", "NA3",
                              "NA4", "NA5"])
    st = d["standing_weights"]
    wt.add_data("local", "STANDING", *[st[na] for na in NA_CELLS])
    for frame, f in sorted(d["frames"].items()):
        for shape in SHAPES:
            w = f["points"][shape]
            wt.add_data(frame, "published/%s" % shape,
                        *[w[na] for na in NA_CELLS])
    for shape, dw in sorted(d["delta_weights"].items()):
        wt.add_data("ranked", "dW/%s (published - standing)" % shape,
                    *[dw[na] for na in NA_CELLS])
    run.log({"deliverable/na_weight_vectors": wt})

    arms = wandb.Table(columns=[
        "rate_table_harness", "arm", "source", "role",
        "NA2_pct", "NA3_pct", "NA4_pct", "NA5_pct",
        "standing_pct", "published_lo", "published_hi",
        "move_lo_pp", "move_hi_pp", "guaranteed_move_pp", "max_move_pp",
        "sign_identified", "immaterial_proven",
        "point_maxent", "point_gt1", "point_gt2"])
    for frame, f in sorted(d["frames"].items()):
        for r in sorted(f["arms"], key=lambda r: -r["max_move_pp"]):
            arms.add_data(
                frame, r["arm"], r["src"], r["role"],
                *[r["na"][na] for na in NA_CELLS],
                r["standing_pct"], r["published_lo"], r["published_hi"],
                r["move_lo_pp"], r["move_hi_pp"], r["guaranteed_move_pp"],
                r["max_move_pp"], r["sign_identified"],
                r["immaterial_proven"],
                *[r["points"][s] for s in SHAPES])
    run.log({"deliverable/arm_rerank": arms})

    un = wandb.Table(columns=["arm", "why_it_cannot_be_reweighted"])
    for arm, why in sorted(d["unresolved"].items()):
        un.add_data(arm, why)
    run.log({"rung1/arms_without_per_na_cells": un})


def item5_tables(run, d: dict) -> dict:
    it5 = d["item5"]
    sh = wandb.Table(columns=["prompt", "maxent", "gt1", "gt2",
                              "identified_lo", "identified_hi"])
    for name, r in sorted(it5["per_prompt_m5_time_share"].items()):
        sh.add_data(name, *[r["points"][s] for s in SHAPES], r["lo"], r["hi"])
    ag = it5["published_weighted_share"]
    sh.add_data("COMBINED published", *[ag["points"][s] for s in SHAPES],
                ag["lo"], ag["hi"])
    run.log({"item5/m5_share_of_candidate_time": sh})

    nets = wandb.Table(columns=[
        "collapse_price", "tax_source", "gain_pct", "tax_pct", "net_point_pct",
        "net_lo_pct", "net_hi_pct", "band_excludes_zero"])
    for k, v in sorted(it5["nets"].items()):
        price, tax = k.split("/")
        nets.add_data(price, tax, v["gain_pct"], v["tax_pct"], v["net_point"],
                      v["net_lo"], v["net_hi"],
                      bool(v["net_lo"] > 0 or v["net_hi"] < 0))
    run.log({"item5/collapse_net_of_register_tax": nets})

    cd = it5["curve_difference_price"]
    pl = wandb.Table(columns=["M", "partition_changed_by_E100", "pre_E100_us",
                              "post_E100_us", "difference_pct"])
    pl.add_data(5, True, cd["pre_us"], cd["post_us"], cd["pct"])
    for M, v in sorted(cd["placebo_unchanged_partition"].items(),
                       key=lambda kv: int(kv[0])):
        pl.add_data(int(M), False, v["pre_us"], v["post_us"], v["pct"])
    run.log({"item5/curve_difference_placebo": pl})
    return it5


def main() -> None:
    d0 = json.loads((ART / "rung0.json").read_text())
    d1 = json.loads((ART / "rung1.json").read_text())
    pm = d1["primary_metric"]

    run = wandb.init(
        entity=ENTITY, project=PROJECT, group=GROUP,
        job_type="operating-point-analysis", name=NAME,
        config={
            "question": (
                "Are the standing NA weights 0.024/0.275/0.667/0.034 the "
                "local operating point wearing a ranked label, and does "
                "re-weighting the recorded kernel arms to the published "
                "operating point change any ranking?"
            ),
            "kill_rule": (
                "stop if no arm moves by more than 0.05 pp under the "
                "published weights"
            ),
            "primary_metric": pm["name"],
            "primary_metric_definition": pm["definition"],
            "pre_registered_gates": (
                "GT2 max|err| <= 0.05, GT1 <= 0.10, GT3 <= 0.08; calibration "
                "may use only (mean draft length, accept, R0/R); the shape is "
                "never fitted to the ground truth"
            ),
            **identity(), **gate_flags(),
        },
        reinit=True,
    )

    rung0_tables(run, d0)
    rung1_tables(run, d1)
    it5 = item5_tables(run, d1)

    v = d0["validation"]["ranked"]
    ag = it5["published_weighted_share"]
    cd = it5["curve_difference_price"]
    e110 = [r for r in d1["frames"]["ranked"]["arms"] if r["src"] == "E110 A"]
    summary = {
        "verdict": "kill_rule_does_not_fire_but_weights_are_not_identified",
        "result_label": "unclear",
        pm["name"]: pm["value"],
        "kill_rule_pp": pm["kill_rule_pp"],
        "kill_rule_fires": bool(pm["value"] < pm["kill_rule_pp"]),
        "identified_set_upper_bound_pp": pm["identified_set_upper_bound"],
        "guaranteed_lower_bound_pp": pm["guaranteed_lower_bound"],
        "shape_ensemble_lo_pp": pm["shape_ensemble_lo"],
        "shape_ensemble_hi_pp": pm["shape_ensemble_hi"],
        "weights_point_identified": False,
        "free_dimensions_after_constraints": 4,
        "bound_coverage_pass": bool(v["bound_pass"]),
        "point_estimator_maxent_gate_pass":
            bool(v["points"]["maxent"]["pass"]),
        "point_estimator_transport_gate_pass":
            bool(v["points"]["transport"]["pass"]),
        "round_counts_all_agree_with_finding_18b":
            all(r["agrees"] for r in d0["round_counts"]),
        "arms_reweighted": len(e110),
        "arms_without_per_na_cells": len(d1["unresolved"]),
        "arms_sign_identified": sum(1 for r in e110 if r["sign_identified"]),
        "item5_m5_time_share_maxent": ag["points"]["maxent"],
        "item5_m5_time_share_lo": ag["lo"],
        "item5_m5_time_share_hi": ag["hi"],
        "item5_curve_difference_usable_as_price":
            bool(cd["usable_as_price"]),
        "item5_curve_difference_treatment_pct": cd["pct"],
        "item5_curve_difference_worst_placebo_pct": cd["worst_placebo_pct"],
    }
    for k, net in it5["nets"].items():
        summary["item5/net/%s/point_pct" % k] = net["net_point"]
        summary["item5/net/%s/lo_pct" % k] = net["net_lo"]
        summary["item5/net/%s/hi_pct" % k] = net["net_hi"]
    for r in e110:
        summary["arm/%s/max_move_pp" % r["arm"]] = r["max_move_pp"]
        summary["arm/%s/sign_identified" % r["arm"]] = r["sign_identified"]
    run.summary.update(summary)

    art = wandb.Artifact("e114-operating-point", type="analysis")
    for f in ("rung0.json", "rung0.txt", "rung1.json", "rung1.txt",
              "selftest.txt"):
        art.add_file(str(ART / f))
    for f in ("scoring_weights.py", "scoring_weights_selftest.py",
              "e114_width_recovery.py", "e114_rerank.py", "e114-results.md"):
        art.add_file("research/%s" % f)
    run.log_artifact(art)

    print("run", run.id, run.url)
    run.finish()


if __name__ == "__main__":
    main()
