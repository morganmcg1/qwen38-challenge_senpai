#!/usr/bin/env python3
"""Log the E34 ranked-operating-point analysis to W&B.

Reads research/e34-ranked-operating-point.json (produced by
research/e34_ranked_operating_point.py) and publishes the full record:
config, scalar summary metrics, tables, and the artifact files.

E34 is an analysis experiment with zero GPU timing (thorfinn holds the
precision-timing slot), so every number logged here is either a replay of
already-measured telemetry or a model prediction.  Predictions are named
`predicted_*` so no downstream reader mistakes them for measurements.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib

import wandb

REPO = pathlib.Path(__file__).resolve().parents[1]
ARTIFACT = REPO / "research" / "e34-ranked-operating-point.json"

PROJECT = "qwen38-mlx-challenge-senpai"
ENTITY = "wandb-applied-ai-team"

PROMPTS = (
    "plutarch",
    "drama",
    "travel",
    "beagle",
    "medicine",
    "republic",
    "essays",
    "botany",
)

ARMS = ("w3_s3", "w4_s4", "w4_s8", "w5_s5", "w5_s8", "w8_s8")
MODELS = ("scaled_local", "ranked_linear", "ranked_quadratic")


def _clean(x):
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x


def _tables(d: dict) -> dict[str, wandb.Table]:
    out: dict[str, wandb.Table] = {}

    disp = d["dispatch"]
    out["e34/dispatch_table"] = wandb.Table(
        columns=["width_M", "ipg_now", "passes_now", "ipg_pre_e27", "passes_pre_e27"],
        data=[
            [
                m,
                disp["ipg"].get(str(m)),
                disp["weight_passes"][str(m)],
                disp["ipg_pre_e27"].get(str(m)),
                disp["weight_passes_pre_e27"][str(m)],
            ]
            for m in range(1, 10)
        ],
    )

    lc = d["local_curve"]
    out["e34/local_forced_depth_curve"] = wandb.Table(
        columns=["width_M", "post_e27_ms", "pre_e27_ms", "delta_ms", "sem_ms", "passes_now"],
        data=[
            [
                m,
                lc["post_e27_ms"][str(m)],
                lc["pre_e27_ms"][str(m)],
                lc["post_e27_ms"][str(m)] - lc["pre_e27_ms"][str(m)],
                lc["sem_ms"][str(m)],
                lc["passes_post"][str(m)],
            ]
            for m in range(1, 9)
        ],
    )

    out["e34/cost_model_fits"] = wandb.Table(
        columns=[
            "fit",
            "max_abs_residual_ms",
            "r_squared",
            "intercept_ms",
            "per_row_ms",
            "per_row2_ms",
            "per_weight_pass_ms",
        ],
        data=[
            [
                name,
                f["max_abs_residual_ms"],
                f["r_squared"],
                f["intercept_ms"],
                f["per_row_ms"],
                f["per_row2_ms"],
                f["per_weight_pass_ms"],
            ]
            for name, f in lc["fits"].items()
        ],
    )

    rk = d["ranked"]["prompts"]
    out["e34/ranked_ledger"] = wandb.Table(
        columns=[
            "prompt",
            "multiple",
            "rounds",
            "drafts_proposed",
            "drafts_accepted",
            "accept_rate",
            "mean_width_M",
            "tokens_per_round",
            "ratio_R",
            "round_ms",
            "serial_ms_per_token",
            "mtp_ms_per_token",
        ],
        data=[
            [
                p,
                rk[p]["multiple"],
                rk[p]["rounds"],
                rk[p]["drafts_proposed"],
                rk[p]["drafts_accepted"],
                rk[p]["accept_rate"],
                rk[p]["mean_width"],
                rk[p]["tokens_per_round"],
                rk[p]["ratio"],
                rk[p]["round_ms"],
                rk[p]["serial_ms_per_token"],
                rk[p]["mtp_ms_per_token"],
            ]
            for p in PROMPTS
        ],
    )

    ident = d["identity_check"]["prompts"]
    out["e34/identity_check"] = wandb.Table(
        columns=[
            "prompt",
            "n",
            "h_bar",
            "implied_alpha",
            "assumed_tokens_per_round",
            "exact_tokens_per_round",
            "tokens_per_round_error",
            "replay_residual",
        ],
        data=[
            [
                p,
                ident[p]["n"],
                ident[p]["h_bar"],
                ident[p]["implied_alpha"],
                ident[p]["assumed_tokens_per_round"],
                ident[p]["exact_tokens_per_round"],
                ident[p]["tokens_per_round_error"],
                ident[p]["replay_residual"],
            ]
            for p in PROMPTS
        ],
    )

    me = d["max_entropy"]
    cal = d["calibration"]
    out["e34/wall_binding"] = wandb.Table(
        columns=[
            "prompt",
            "frac_M_ge_6_policy_sim",
            "frac_M_ge_6_max_entropy",
            "mean_width_M_ledger",
            "mean_passes_policy_sim",
        ],
        data=[
            [
                p,
                cal[p]["fraction_ge_M6"],
                me[p]["fraction_ge_M6"],
                rk[p]["mean_width"],
                cal[p]["mean_passes"],
            ]
            for p in PROMPTS
        ],
    )

    cf = d["counterfactual"]

    def ratio(prompt: str, model: str, arm: str) -> float:
        return cf[prompt][model][arm]["predicted_ratio"]

    out["e34/counterfactual_central_pair"] = wandb.Table(
        columns=["arm", *MODELS],
        data=[
            [a, *[(ratio("beagle", m, a) + ratio("medicine", m, a)) / 2 for m in MODELS]]
            for a in ARMS
        ],
    )

    out["e34/counterfactual_per_prompt"] = wandb.Table(
        columns=["prompt", "cost_model", "measured_R", *ARMS],
        data=[
            [p, m, d["ranked"]["prompts"][p]["ratio"], *[ratio(p, m, a) for a in ARMS]]
            for p in PROMPTS
            for m in MODELS
        ],
    )

    out["e34/counterfactual_widths"] = wandb.Table(
        columns=["prompt", "arm", "mean_width_M", "mean_passes", "tokens_per_round",
                 "round_ms", "accept_rate", "fraction_ge_M6", "predicted_R"],
        data=[
            [
                p,
                a,
                e["mean_width"],
                e["mean_passes"],
                e["tokens_per_round"],
                e["round_ms"],
                e["accept_rate"],
                e["fraction_ge_M6"],
                e["predicted_ratio"],
            ]
            for p in PROMPTS
            for a in ARMS
            for e in [cf[p]["ranked_linear"][a]]
        ],
    )

    os_ = d["score_order_statistics"]
    out["e34/order_statistics"] = wandb.Table(
        columns=["source", "cost_model", "arm", "median", "central_prompts", "order"],
        data=[
            ["measured", "-", "-", os_["measured"]["median"],
             ",".join(os_["measured"]["central_prompts"]), ",".join(os_["measured"]["order"])]
        ]
        + [
            ["board_kernel_replay", m, "w5_s8", v["median"],
             ",".join(v["central_prompts"]), ",".join(v["order"])]
            for m, v in os_["board_kernel_replay_shipped"].items()
        ]
        + [
            ["our_kernel", m, a, os_["our_kernel"][m][a]["median"],
             ",".join(os_["our_kernel"][m][a]["central_prompts"]),
             ",".join(os_["our_kernel"][m][a]["order"])]
            for m in MODELS
            for a in ARMS
        ],
    )

    lcnt = d["local_counters"]["per_prompt"]
    out["e34/local_realised_depth"] = wandb.Table(
        columns=["prompt", "leg", "effective_max_draft_len", "effective_mean_draft_len",
                 "max_width_M", "mean_width_M", "accepted_draft_rate", "round_count",
                 "replayed_rounds"],
        data=[
            [
                p,
                leg,
                v["effective_max_draft_len"],
                v["effective_mean_draft_len"],
                v["effective_max_draft_len"] + 1,
                v["effective_mean_draft_len"] + 1,
                v["accepted_draft_rate"],
                v["round_count"],
                v["replayed_rounds"],
            ]
            for leg, prompts in lcnt.items()
            for p, v in prompts.items()
        ],
    )

    sens = d["sensitivity"]
    out["e34/sensitivity_to_p"] = wandb.Table(
        columns=[
            "prompt",
            "q0",
            "gamma",
            "margin_mean",
            "mean_draft_len",
            "accept_rate",
            "shipped_ratio",
            "w4_s4_ratio",
            "relative_gain",
            "shipped_fraction_ge_M6",
        ],
        data=[
            [
                p,
                v["q0"],
                v["gamma"],
                v["margin_mean"],
                v["mean_draft_len"],
                v["accept_rate"],
                v["shipped_ratio"],
                v["best_ratio"],
                v["relative_gain"],
                v["shipped_fraction_ge_M6"],
            ]
            for p in ("beagle", "medicine")
            for v in sens[p]["variants"].values()
        ],
    )

    rows = []
    for p in ("beagle", "medicine"):
        ev = d["ranked_depth_evidence"][p]
        b = ev["best_on_default"]
        rows.append(
            [
                p,
                "default",
                ev["default_row_count"],
                ev["default_median_accept_rate"],
                b["ratio"],
                b["id"],
                b["ms_per_token"],
            ]
        )
        for cohort in ("deeper_than_default", "shallower_than_default"):
            c = ev[cohort]
            best = max(c["rows"], key=lambda r: r["ratio"]) if c["rows"] else None
            rows.append(
                [
                    p,
                    cohort,
                    c["count"],
                    c["median_accept_rate"],
                    c["best_ratio"],
                    best["id"] if best else None,
                    c["best_ms_per_token"],
                ]
            )
    out["e34/ranked_depth_evidence"] = wandb.Table(
        columns=[
            "prompt",
            "cohort",
            "rows",
            "median_accept_rate",
            "best_R",
            "best_row",
            "best_ms_per_token",
        ],
        data=rows,
    )

    out["e34/width_distribution_policy_sim"] = wandb.Table(
        columns=["prompt", *[f"M{m}" for m in range(1, 10)]],
        data=[
            [p, *[cal[p]["width_distribution"].get(str(m), 0.0) for m in range(1, 10)]]
            for p in PROMPTS
        ],
    )

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()

    d = json.loads(ARTIFACT.read_text())

    disp = d["dispatch"]
    lc = d["local_curve"]
    sp = d["score_prediction"]
    prov = d["cap_provenance"]
    head = d["head"]

    config = {
        "experiment": "e34-ranked-operating-point-depth-cap",
        "assignment_id": "qwen38-r1-e34-ranked-operating-point-depth-cap",
        "revision": "r1",
        "pr": 39,
        "base_sha": "4e5dc2bdc9ed7b89c1b3c75a7fc0620e97d43549",
        "gpu_timing_runs": 0,
        "analysis_only": True,
        "shipped_sdpaWidthWallDepthCap": 5,
        "shipped_segmentedVerifyDepthCap": 8,
        "shipped_segmentedStreakGate": 2,
        "shipped_headStepCostRatio": 0.18,
        "shipped_acceptEMAAlpha": 0.15,
        "candidate_arm": "w4_s4",
        "candidate_sdpaWidthWallDepthCap": 4,
        "candidate_segmentedVerifyDepthCap": 4,
        "single_pass_top_width_now": disp["single_pass_top_width_now"],
        "single_pass_top_width_pre_e27": disp["single_pass_top_width_pre_e27"],
        "single_pass_top_depth_now": disp["single_pass_top_depth_now"],
        "single_pass_top_depth_pre_e27": disp["single_pass_top_depth_pre_e27"],
        "cap_organizer_value": prov["organizer_value"],
        "cap_raised_to": prov["raised_to"],
        "cap_raised_by_submission": prov["raised_by_submission"],
        "cap_raised_by_solver": prov["raised_by_solver"],
        "cap_raised_by_score": prov["raised_by_score"],
        "cap_raised_composite_arms": prov["composite_arms"],
        "cap_crossed_pass_boundary_when_raised": prov["crossed_a_pass_boundary_when_raised"],
        "cap_crosses_pass_boundary_now": prov["crosses_a_pass_boundary_now"],
        "board_top_row": d["ranked"]["id"],
        "board_top_solver": d["ranked"]["solver"],
        "board_top_score": d["ranked"]["score"],
        "head_verified": head["run_tree_matches_manifest"],
        "head_tree_digest": head["manifest"]["sha256"],
        "head_bytes": head["manifest"]["bytes"],
        "head_file_digest": head["file_digest_of_model_safetensors"],
    }

    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        job_type="analysis",
        name="e34-ranked-operating-point-depth-cap",
        notes=(
            "E34: ranked operating point + pass-count-aware cost model. "
            "Zero GPU timing; replays measured E25r3 forced-depth curves and "
            "the ranked row ledger of board top 0cd0a6b4. The primary metric "
            "is a model PREDICTION, not a measurement."
        ),
        tags=["e34", "qwen38-mtp", "analysis", "no-gpu", "cost-model", "depth-cap"],
        config=config,
        mode="offline" if args.offline else "online",
    )

    summary = {
        "e34/predicted_ranked_central_pair_at_best_cap": sp[
            "predicted_ranked_central_pair_at_best_cap"
        ],
        "e34/predicted_interval_low": sp["interval"][0],
        "e34/predicted_interval_high": sp["interval"][1],
        "e34/baseline_board_top_score": sp["board_top_score"],
        "e34/measured_central_pair": sp["measured_central_pair"],
        "e34/predicted_delta_vs_baseline": (
            sp["predicted_ranked_central_pair_at_best_cap"] - sp["board_top_score"]
        ),
        "e34/causal_one_weight_pass_cost_ms": lc["causal_pass_cost_ms"],
        "e34/host_transfer_scale_kappa": d["host_transfer"]["final"]["scale"],
        "e34/host_transfer_max_resid_ms": d["host_transfer"]["final"]["max_abs_residual_ms"],
        "e34/host_transfer_r_squared": d["host_transfer"]["final"]["r_squared"],
        "e34/identity_max_replay_residual": d["identity_check"]["max_abs_replay_residual"],
        "e34/identity_max_tokens_per_round_error": d["identity_check"][
            "max_abs_tokens_per_round_error"
        ],
        "e34/depth_echo_distinct_pairs": len(d["depth_echo"]["distinct_depth_pairs"]),
        "e34/depth_echo_scored_rows": d["depth_echo"]["scored_rows"],
        "e34/depth_is_config_echo": int(bool(d["depth_echo"]["is_config_echo"])),
        "e34/depth_echo_zero_draft_controls": d["depth_echo"]["zero_draft_control_count"],
    }

    for name, f in lc["fits"].items():
        key = name.replace("+", "_")
        summary[f"e34/fit/{key}/max_abs_residual_ms"] = f["max_abs_residual_ms"]
        summary[f"e34/fit/{key}/r_squared"] = f["r_squared"]
        summary[f"e34/fit/{key}/per_weight_pass_ms"] = f["per_weight_pass_ms"]
        summary[f"e34/fit/{key}/per_row_ms"] = f["per_row_ms"]

    for name in ("linear", "quadratic"):
        f = d["ranked_cost_fit"][name]
        summary[f"e34/ranked_fit/{name}/per_row_ms"] = f["per_row_ms"]
        summary[f"e34/ranked_fit/{name}/per_weight_pass_ms"] = f["per_weight_pass_ms"]
        summary[f"e34/ranked_fit/{name}/r_squared"] = f["r_squared"]
        summary[f"e34/ranked_fit/{name}/max_abs_residual_ms"] = f["max_abs_residual_ms"]

    for model in MODELS:
        summary[f"e34/board_kernel_replay/{model}"] = sp["board_kernel_replay"][model]
        summary[f"e34/shipped_replay_error/{model}"] = sp["shipped_replay_error"][model]
        summary[f"e34/best_arm/{model}"] = sp["best_arm"][model]
        for arm in ARMS:
            summary[f"e34/counterfactual/{model}/{arm}"] = sp["modelled_our_kernel"][model][arm]

    for name, dec in sp["decomposition"].items():
        for model, value in dec.items():
            summary[f"e34/decomposition/{name}/{model}"] = value

    pm = sp["published_median"]
    summary["e34/predicted_published_median_at_best_cap"] = pm["predicted_at_best_arm"]
    summary["e34/measured_published_median"] = pm["measured"]
    summary["e34/published_median_replay_error"] = pm["board_kernel_replay_error"]
    summary["e34/central_pair_minus_median"] = pm["central_pair_minus_median"]
    summary["e34/central_pair_is_still_median_at_best_arm"] = int(
        bool(pm["central_pair_is_still_the_median_at_best_arm"])
    )
    for model in MODELS:
        for arm in ARMS:
            summary[f"e34/median/{model}/{arm}"] = d["score_order_statistics"]["our_kernel"][
                model
            ][arm]["median"]

    lsum = d["local_counters"]["summary"]
    for leg, v in lsum.items():
        summary[f"e34/local_depth/{leg}/max_width_M_reached"] = v["max_width_M_reached"]
        summary[f"e34/local_depth/{leg}/mean_width_M_min"] = v["mean_width_M_range"][0]
        summary[f"e34/local_depth/{leg}/mean_width_M_max"] = v["mean_width_M_range"][1]
        summary[f"e34/local_depth/{leg}/reaches_shipped_wall_M6"] = int(
            bool(v["reaches_shipped_wall_M6"])
        )
    summary["e34/ranked_mean_width_min"] = d["local_counters"]["ranked_mean_width_range"][0]
    summary["e34/ranked_mean_width_max"] = d["local_counters"]["ranked_mean_width_range"][1]

    for p in PROMPTS:
        summary[f"e34/wall_binding/{p}/policy_sim"] = d["calibration"][p]["fraction_ge_M6"]
        summary[f"e34/wall_binding/{p}/max_entropy"] = d["max_entropy"][p]["fraction_ge_M6"]
        summary[f"e34/ledger/{p}/mean_width"] = d["ranked"]["prompts"][p]["mean_width"]
        summary[f"e34/ledger/{p}/ratio"] = d["ranked"]["prompts"][p]["ratio"]
        summary[f"e34/ledger/{p}/accept_rate"] = d["ranked"]["prompts"][p]["accept_rate"]

    for p in ("beagle", "medicine"):
        s = d["sensitivity"][p]
        summary[f"e34/sensitivity/{p}/admissible_count"] = s["admissible_count"]
        summary[f"e34/sensitivity/{p}/relative_gain_min"] = s["relative_gain_min"]
        summary[f"e34/sensitivity/{p}/relative_gain_max"] = s["relative_gain_max"]
        summary[f"e34/sensitivity/{p}/relative_gain_median"] = s["relative_gain_median"]
        ev = d["ranked_depth_evidence"][p]
        summary[f"e34/depth_evidence/{p}/default_best_R"] = ev["best_on_default"]["ratio"]
        summary[f"e34/depth_evidence/{p}/deeper_best_R"] = ev["deeper_than_default"]["best_ratio"]
        summary[f"e34/depth_evidence/{p}/shallower_best_R"] = ev["shallower_than_default"][
            "best_ratio"
        ]
        summary[f"e34/depth_evidence/{p}/shallow_is_valid_control"] = int(
            bool(ev["shallow_rows_are_a_valid_cap_control"])
        )
        summary[f"e34/depth_evidence/{p}/deeper_ever_beat_default"] = int(
            bool(ev["deeper_ever_beat_default_ratio"])
        )

    summary = {k: _clean(v) for k, v in summary.items()}
    run.summary.update(summary)
    run.log({**{k: v for k, v in summary.items() if not isinstance(v, str)}, **_tables(d)})

    art = wandb.Artifact("e34-ranked-operating-point", type="analysis")
    art.add_file(str(ARTIFACT))
    art.add_file(str(REPO / "research" / "e34_ranked_operating_point.py"))
    art.add_file(str(REPO / "research" / "e34_cost_model.py"))
    run.log_artifact(art)

    print(f"run_id={run.id}")
    print(f"run_url={run.url}")
    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
