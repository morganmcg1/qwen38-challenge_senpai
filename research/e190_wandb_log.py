#!/usr/bin/env python3
"""Publish the E190 cap-5 ranked receipt and its predeclared law test to W&B.

    usage: research/e190_wandb_log.py --receipt /tmp/e190_receipt.json

harness=ranked for every prediction and for the receipt itself. The one
harness=local block is the 512-token `--local-submit` confirmation that
qualified the candidate for submission; it is a correctness and acceptance
record, not a ranked speed claim.

The predictions are recomputed here from `e190_predict`, not copied, so the
logged numbers cannot drift from the predeclared ones. They were posted to
PR #188 before the submission was sent.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

import wandb

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import e177_ranked_depth_law as e177  # noqa: E402
import e190_predict as e190  # noqa: E402

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
GROUP = "e190-cap5-ranked-receipt"

BENCHMARK_ID = "5d1ee4d7-80bd-4555-b182-6505f26ef495"
SUBMISSION_ID = "04710829-3c7e-4ace-9751-f1fda9e368e9"
GUARD_BASE_SHA = "770a3ff2f8fbd1bb75d15e3c37ae3c5b076ebbcf"
SCIENTIFIC_BASE_SHA = "e0c0dec79f0b813e8c9a7caef07e49149e99176e"

# Model-free primary rule: the paid cap-4 receipt separates the two bands.
PRIMARY_BOUNDARY = e177.CAP4_PUBLISHED

# 512-token --local-submit confirmation, job 9e45aac4, worker rebuilt 15:07:35.
CONFIRMATION = {
    "decode_tokens": 512,
    "effective_mean_draft_len": 4.804123711340206,
    "all_tokens_matched": True,
    "reference_checked_rows_mtp": 563,
    "reference_checked_rows_mtp_total": 563,
    "reference_checked_rows_serial": 512,
    "reference_checked_rows_serial_total": 512,
    "chain_contradictions": 0,
    "residual_divergence_count": 0,
    "public_drift_tripwire_passed": True,
    "uses_pinned_mtp_head": True,
    "head_provenance_sha256":
        "6fe9db14b777d8c73f8daf9f2997bce126626bf0372126942e5732467aec69f6",
    "serial_seconds_per_token": 0.073149556759744883,
    "mtp_seconds_per_token": 0.033621271606534719,
    "local_ratio": 2.17569274641972,
    "rounds": 97,
    "accepted_draft_rate": 0.89270386266094426,
}


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def predictions():
    """Recompute the predeclared cap-5 predictions. Mirrors e190_predict.main."""
    data, surv, bounds, mu, fits, core = e190.build()
    quad = fits["quadratic"]
    stairs, shape = e190.staircase_variants(quad)
    stairs["staircase-ranked-refit"] = e190.refit_shape_on_ranked(core, shape)

    fitted = {n: list(surv[n]) for n in e177.ORDER}
    preds = {"quadratic": []}
    for key in stairs:
        preds[key] = []
    for anchor in ("A", "C"):
        preds["quadratic"].append(
            e190.score_of(data, fitted, mu, quad, e190.TARGET_CAP, anchor))
        for key, model in stairs.items():
            preds[key].append(
                e190.score_of(data, fitted, mu, model, e190.TARGET_CAP, anchor))
    means = {k: sum(v) / len(v) for k, v in preds.items()}
    return quad, stairs, shape, means


def verdict(score, means):
    """Primary model-free rule, then the secondary 1-sigma window rule."""
    if score is None:
        return "no-score", "no-verdict"
    primary = "quadratic-band" if score > PRIMARY_BOUNDARY else "staircase-band"

    hits = []
    for key, mean in means.items():
        lo = mean * (1 - e190.SIGMA_PUBLISHED)
        hi = mean * (1 + e190.SIGMA_PUBLISHED)
        if lo <= score <= hi:
            hits.append(key)
    if len(hits) == 1:
        secondary = hits[0]
    elif not hits:
        secondary = "outside-every-window"
    else:
        secondary = "inside-multiple-windows"
    return primary, secondary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", default="/tmp/e190_receipt.json")
    parser.add_argument("--inversion", default="research/out/e190/inversion.json")
    args = parser.parse_args()

    receipt = json.loads(pathlib.Path(args.receipt).read_text())
    status = receipt.get("status")
    score = receipt.get("officialScore")
    metrics = receipt.get("officialMetrics") or {}
    inversion = json.loads(pathlib.Path(args.inversion).read_text())

    quad, stairs, shape, means = predictions()
    pq = means["quadratic"]
    primary, secondary = verdict(score, means)

    config = {
        "experiment": "e190-cap5-ranked-receipt",
        "question": "is the ranked round-cost law the bare quadratic or the "
                    "affine-rescaled staircase",
        "harness": "ranked",
        "official_score": True,
        "purpose": "law discrimination, not a promotion attempt",
        "draft_cap": e190.TARGET_CAP,
        "candidate_head": git("rev-parse", "HEAD"),
        "worktree_dirty": bool(git("status", "--porcelain")),
        "scientific_base_sha": SCIENTIFIC_BASE_SHA,
        "guard_base_sha": GUARD_BASE_SHA,
        "guard_base_is_merge_base_with_origin_main": True,
        "candidate_diff": "Qwen36MTPBlockSession.swift segmentedVerifyDepthCap "
                          "7 -> 5, one file",
        "benchmark_id": BENCHMARK_ID,
        "submission_id": SUBMISSION_ID,
        "ranked_host": "m5-qwen38-27b-mtp",
        "sigma_published": e190.SIGMA_PUBLISHED,
        "mue_ms": e190.MUE_MS,
        "primary_decision_boundary": PRIMARY_BOUNDARY,
        "crown_score": e177.CROWN,
        "receipt_a_cap7": e177.BEST_A,
        "receipt_c_cap4": e177.CAP4_PUBLISHED,
        "predeclared_before_submission": True,
        **{f"confirmation_{k}": v for k, v in CONFIRMATION.items()},
        "confirmation_harness": "local",
        "confirmation_tokens": 512,
        "confirmation_is_speed_claim": False,
    }

    run = wandb.init(
        entity=ENTITY, project=PROJECT, group=GROUP,
        name="e190-cap5-receipt", job_type="analysis", config=config,
    )

    pred_cols = ["law", "predicted_published", "sigma_lo", "sigma_hi",
                 "delta_vs_quadratic", "sigma_from_quadratic", "harness"]
    pred_table = wandb.Table(columns=pred_cols)
    for key, mean in means.items():
        pred_table.add_data(
            key, mean,
            mean * (1 - e190.SIGMA_PUBLISHED),
            mean * (1 + e190.SIGMA_PUBLISHED),
            mean - pq,
            abs(mean - pq) / (e190.SIGMA_PUBLISHED * pq),
            "ranked",
        )

    law_cols = ["rows", "quadratic_ms"] + [f"{k}_ms" for k in stairs] + [
        "local_shape_ms", "harness"]
    law_table = wandb.Table(columns=law_cols)
    for i, m in enumerate(e190.WIDTHS):
        row = [m, 1000 * e177.cost_at_width(quad, m)]
        row += [1000 * e177.cost_at_width(model, m) for model in stairs.values()]
        row += [shape[i] if i < len(shape) else None, "ranked"]
        law_table.add_data(*row)

    res_cols = ["law", "predicted", "observed", "residual",
                "residual_sigma", "harness"]
    res_table = wandb.Table(columns=res_cols)
    if score is not None:
        for key, mean in means.items():
            res_table.add_data(
                key, mean, score, score - mean,
                abs(score - mean) / (e190.SIGMA_PUBLISHED * mean), "ranked")

    dr6_cols = ["law", "predicted_dr6_ms", "measured_dr6_ms", "error_ms",
                "error_mue", "c_anchored_score", "harness"]
    dr6_table = wandb.Table(columns=dr6_cols)
    for key, law in inversion["laws"].items():
        dr6_table.add_data(
            key, law["dr6_ms"], inversion["dr6_ms"], law["err_ms"],
            law["err_mue"], law["c_anchored_score"], "ranked")

    pp_cols = ["prompt", "raw_ratio", "effective_mean_draft_len",
               "mtp_seconds_per_token_mean", "serial_seconds_per_token_mean",
               "non_drafting_round_count", "parity_ok", "harness"]
    pp_table = wandb.Table(columns=pp_cols)
    for i, p in enumerate(metrics.get("per_prompt", [])):
        pp_table.add_data(
            p.get("prompt_sha256", "")[:12], p.get("raw_ratio_of_means"),
            p.get("effective_mean_draft_len"),
            p.get("mtp_seconds_per_token_mean"),
            p.get("serial_seconds_per_token_mean"),
            p.get("non_drafting_round_count"), p.get("parity_ok"), "ranked")

    summary = {
        "receipt/status": status,
        "receipt/official_score": score,
        "receipt/submission_id": SUBMISSION_ID,
        "verdict/primary_rule": primary,
        "verdict/secondary_window_rule": secondary,
        "verdict/primary_boundary": PRIMARY_BOUNDARY,
        "prediction/quadratic": means["quadratic"],
        "prediction/staircase_published": means["staircase-published"],
        "prediction/staircase_corrected": means["staircase-corrected"],
        "prediction/staircase_ranked_refit": means["staircase-ranked-refit"],
        "fit/quadratic_wrmse_ms": quad["wrmse_ms"],
        "fit/quadratic_aic": quad["aic"],
        "fit/staircase_refit_wrmse_ms": stairs["staircase-ranked-refit"]["wrmse_ms"],
        "fit/staircase_refit_aic": stairs["staircase-ranked-refit"]["aic"],
        "inversion/dr6_ms": inversion["dr6_ms"],
        "inversion/dr6_ms_sigma_lo": inversion["dr6_ms_sigma_band"][0],
        "inversion/dr6_ms_sigma_hi": inversion["dr6_ms_sigma_band"][1],
        "inversion/dr6_mue": inversion["dr6_mue"],
        "inversion/bisection_closure": inversion["bisection_closure"],
    }
    for key, law in inversion["laws"].items():
        summary[f"dr6_error_mue/{key}"] = law["err_mue"]
        summary[f"dr6_predicted_ms/{key}"] = law["dr6_ms"]
    if score is not None:
        for key, mean in means.items():
            summary[f"residual/{key}"] = score - mean
            summary[f"residual_sigma/{key}"] = abs(score - mean) / (
                e190.SIGMA_PUBLISHED * mean)
    for key, value in metrics.items():
        if isinstance(value, (int, float, str, bool)) or value is None:
            summary[f"receipt_metrics/{key}"] = value

    run.log({
        "predictions": pred_table,
        "round_cost_laws": law_table,
        "residuals": res_table,
        "six_row_cell_price": dr6_table,
        "receipt_per_prompt": pp_table,
    })
    run.summary.update(summary)
    print("run:", run.url)
    print("status:", status, "score:", score)
    print("primary:", primary, "secondary:", secondary)
    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
