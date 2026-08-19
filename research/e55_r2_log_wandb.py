#!/usr/bin/env python3
"""Push the E55 revision e55-r2 readings to the headline bracket run.

Two independent blocks, so either can be pushed as soon as it exists:

  python3 research/e55_r2_log_wandb.py --run wxezisvs --reconcile
  python3 research/e55_r2_log_wandb.py --run wxezisvs --local-submit

The 15:02Z retag destroyed a peer's workspace mid-session, so nothing waits for
the end of the turn.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import wandb

PROJECT = "wandb-applied-ai-team/qwen38-mlx-challenge-senpai"
ROOT = pathlib.Path(__file__).resolve().parents[1]


def reconcile_summary() -> dict:
    d = json.loads((ROOT / "research" / "e55-e54-reconcile.json").read_text())
    bim = d["r_is_bimodal_in_the_group_count_change"]
    verdict = d["verdict"]
    eta = d["eta_curve"]
    rep = d["eta_replication"]
    flat = d["ladder_versus_flat_at_group_preserving_cells"]
    out = {
        "recon/retracted": verdict["retracted"],
        "recon/worst_oos_residual_points_stream_model":
            verdict["worst_out_of_sample_residual_points"]["my_e55_stream_model"],
        "recon/worst_oos_residual_points_third_structure":
            verdict["worst_out_of_sample_residual_points"]["third_structure"],
        "recon/worst_oos_residual_points_flat_law_a_prime":
            flat["worst_abs_points"]["flat"],
        "recon/r_mean_delta_n_minus_1": bim["delta_n_minus_1"]["mean"],
        "recon/r_spread_pct_delta_n_minus_1": bim["delta_n_minus_1"]["spread_pct"],
        "recon/r_mean_delta_n_zero": bim["delta_n_zero"]["mean"],
        "recon/r_spread_pct_delta_n_zero": bim["delta_n_zero"]["spread_pct"],
        "recon/r_separation_factor": bim["separation_factor"],
        "recon/eta_1": eta["eta_1"],
        "recon/eta_2": eta["eta_2"],
        "recon/eta_3": eta["eta_3"],
        "recon/eta_1_over_2_spread_pct": rep["eta(1)/eta(2)"]["spread_pct"],
        "recon/eta_2_over_3_spread_pct": rep["eta(2)/eta(3)"]["spread_pct"],
        "recon/identity_max_residual_points": max(
            v["abs_residual_points"]
            for v in d["thorfinn_rate_agreement_is_an_identity"]["pairs"].values()
        ),
        "recon/flat_refuted_at": json.dumps(flat["flat_is_refuted_at"]),
        "recon/negative_controls_fired": sum(
            1 for v in d["negative_controls"].values() if v["fired"]
        ),
        "recon/negative_controls_total": len(d["negative_controls"]),
        "recon/self_tests_passed": d["self_tests"]["passed"],
    }
    for key, row in d["third_structure_out_of_sample"].items():
        out[f"recon/{key}/measured_delta_pct"] = row["measured_delta_pct"]
        out[f"recon/{key}/third_predicted_delta_pct"] = row["predicted_delta_pct"]
        out[f"recon/{key}/third_residual_points"] = row["residual_points"]
    for key, row in d["stream_model_out_of_sample"]["holdout"].items():
        out[f"recon/{key}/stream_predicted_delta_pct"] = row["predicted_delta_pct"]
        out[f"recon/{key}/stream_residual_points"] = row["residual_points"]
    for na, gbps in d["lone_group_rate_ladder_gbps"].items():
        out[f"recon/lone_group_gbps_na{na}"] = gbps
    return out


def local_submit_summary() -> dict:
    path = ROOT / "research" / "e55-local-submit-score.json"
    s = json.loads(path.read_text())
    out = {"submit/score_json": json.dumps(s, sort_keys=True)}
    for key, value in s.items():
        if isinstance(value, (int, float, bool, str)):
            out[f"submit/{key}"] = value
    for key, value in s.get("metrics", {}).items():
        if isinstance(value, (int, float, bool, str)):
            out[f"submit/metrics/{key}"] = value
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--reconcile", action="store_true")
    ap.add_argument("--local-submit", action="store_true")
    args = ap.parse_args()
    if not (args.reconcile or args.local_submit):
        ap.error("choose at least one of --reconcile or --local-submit")

    summary: dict = {}
    if args.reconcile:
        summary.update(reconcile_summary())
    if args.local_submit:
        summary.update(local_submit_summary())

    run = wandb.init(project=PROJECT.split("/")[1], entity=PROJECT.split("/")[0],
                     id=args.run, resume="must")
    run.summary.update(summary)
    run.finish()
    for key in sorted(summary):
        print(f"{key} = {summary[key]}")
    print(f"logged {len(summary)} keys to {PROJECT}/runs/{args.run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
