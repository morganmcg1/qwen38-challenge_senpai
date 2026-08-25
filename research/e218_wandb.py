#!/usr/bin/env python3
"""Publish the E218 non-QMV width-tax census to W&B.

    usage: research/e218_wandb.py research/e218-artifacts/width-tax-census.json
                                  [--name NAME] [--notes TEXT]

Every number in the document is `harness=local` and ungated: the legs run with
the per-round phase trace on and `MLXFAST_LOCAL_COOL_GATE=0` under the standing
counterbalanced conditions. Band-arm legs are ATTRIBUTION ONLY — their round
time is inflated by the per-boundary device drains and is never a round cost.
No number here is an official or ranked score, and no absolute leg time is a
candidate speed claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import wandb

PROJECT = "qwen38-mlx-challenge-senpai"
ENTITY = "wandb-applied-ai-team"
EXPERIMENT = "e218-width-tax-census"
GROUP = "qwen38-r1-e218-width-tax-census"

BANDS = ["band_pre_us", "band_gdn_mixer_us", "band_gdn_mlp_us",
         "band_fa_mixer_us", "band_fa_mlp_us"]
PHASE_COLUMNS = ["round_us", "draft_build_us", "verify_build_us",
                 "eval_wall_us", "readout_us", "commit_us", "upkeep_us"]


def leg_table(report: dict) -> wandb.Table:
    columns = ["leg", "session", "arm", "width_m", "band_sync", "leg_index",
               "rounds_at_width", "round_ms_median", "round_ms_ci_lo",
               "round_ms_ci_hi", "gpu_temp_entry_c", "gpu_temp_exit_c",
               "all_tokens_matched", "residual_divergence_count",
               "mtp_seconds_per_token", "serial_seconds_per_token",
               "effective_mean_draft_len", "accepted_draft_rate",
               "worker_digest_stable", "worker_sha256", "base_sha"]
    rows = []
    for leg in report["legs"]:
        ci = leg["phase_ci_ms"].get("round_us", [None, None])
        rows.append([
            leg["leg"], leg["session"], leg["arm"], leg["width_m"],
            leg["band_sync"], leg["leg_index"], leg["rounds_at_width"],
            leg["phases_ms"].get("round_us", {}).get("median"), ci[0], ci[1],
            leg["gpu_temp_entry_c"], leg["gpu_temp_exit_c"],
            leg["all_tokens_matched"], leg["residual_divergence_count"],
            leg["mtp_seconds_per_token"], leg["serial_seconds_per_token"],
            leg["effective_mean_draft_len"], leg["accepted_draft_rate"],
            leg["worker_digest_stable"], leg["worker_sha256"], leg["base_sha"],
        ])
    return wandb.Table(columns=columns, data=rows)


def width_table(report: dict) -> wandb.Table:
    columns = (["m", "rounds_at_width", "leg_spread_ms", "over_noise_floor"]
               + [f"{p[:-3]}_ms" for p in PHASE_COLUMNS])
    rows = []
    for key, entry in sorted(report["width_table_untraced"].items(),
                             key=lambda kv: int(kv[0])):
        phases = entry["phase_ms"]
        rows.append([
            int(key), entry["rounds_at_width"],
            phases["round_us"]["leg_spread"], phases["round_us"]["over_noise_floor"],
        ] + [phases.get(p, {}).get("value") for p in PHASE_COLUMNS])
    return wandb.Table(columns=columns, data=rows)


def band_table(report: dict) -> wandb.Table:
    columns = (["m"] + [f"{b[5:-3]}_ms" for b in BANDS]
               + ["post_band_ms", "band_forward_ms", "round_ms",
                  "gdn_mixer_per_layer_ms", "gdn_mlp_per_layer_ms",
                  "fa_mixer_per_layer_ms", "fa_mlp_per_layer_ms"])
    rows = []
    for key, entry in sorted(report["width_table_band_arm"].items(),
                             key=lambda kv: int(kv[0])):
        phases = entry["phase_ms"]
        per_layer = entry.get("band_per_layer_ms", {})
        rows.append([int(key)]
                    + [phases.get(b, {}).get("value") for b in BANDS]
                    + [phases["eval_wall_us"]["value"],
                       entry["band_sum_ms"] + phases["eval_wall_us"]["value"],
                       phases["round_us"]["value"],
                       per_layer.get("band_gdn_mixer_us"),
                       per_layer.get("band_gdn_mlp_us"),
                       per_layer.get("band_fa_mixer_us"),
                       per_layer.get("band_fa_mlp_us")])
    return wandb.Table(columns=columns, data=rows)


def slope_table(report: dict) -> wandb.Table:
    columns = ["boundary", "true_round_step_ms", "true_forward_step_ms",
               "gdn_mixer_step_ms", "gdn_mlp_step_ms", "fa_mixer_step_ms",
               "fa_mlp_step_ms", "pre_step_ms", "lm_head_readout_step_ms",
               "draft_head_step_ms", "other_round_step_ms",
               "closure_residual_ms", "residual_over_2ms",
               "cells_changing_passes", "nominal_weight_gb_step"]
    rows = []
    for row in report["slope_census"]:
        fam = row["family_step_ms"]
        rows.append([
            row["boundary"], row["true_round_step_ms"],
            row["true_forward_step_ms"],
            fam.get("band_gdn_mixer_us"), fam.get("band_gdn_mlp_us"),
            fam.get("band_fa_mixer_us"), fam.get("band_fa_mlp_us"),
            fam.get("band_pre_us"), fam.get("lm_head_readout"),
            row["draft_head_step_ms"], row["other_round_step_ms"],
            row["closure_residual_ms"], row["closure_residual_over_2ms"],
            ",".join(row["cells_changing_passes"]) or "none",
            row["nominal_weight_gb_step"],
        ])
    return wandb.Table(columns=columns, data=rows)


def perturbation_table(report: dict) -> wandb.Table:
    columns = ["m", "w_round_ms", "b_round_ms", "round_inflation_ms",
               "round_inflation_x", "w_forward_ms", "b_forward_ms",
               "forward_inflation_x"]
    rows = []
    for key, entry in sorted(report["perturbation"].items(),
                             key=lambda kv: int(kv[0])):
        rows.append([int(key), entry["w_round_ms"], entry["b_round_ms"],
                     entry["round_inflation_ms"], entry["round_inflation_x"],
                     entry["w_forward_ms"], entry["b_forward_ms"],
                     entry["forward_inflation_x"]])
    return wandb.Table(columns=columns, data=rows)


def cell_table(report: dict) -> wandb.Table | None:
    decomposition = report.get("e186_decomposition")
    if not decomposition:
        return None
    columns = ["family", "variant", "m", "whole_ms", "part_a_ms", "part_b_ms",
               "part_c_ms", "remainder_ms", "parts"]
    rows = []
    for key, by_m in decomposition.get("gdn_layer", {}).items():
        rows.append(["gdn_layer", "nConfirmed=1", int(key), by_m["layer_ms"],
                     by_m["in_proj_ms"], by_m["out_proj_ms"],
                     by_m["recurrence_ms"], by_m["remainder_ms"],
                     "in_proj|out_proj|recurrence"])
    for kv, table in decomposition.get("fa_layer", {}).items():
        for key, by_m in table.items():
            rows.append(["fa_layer", f"kv={kv}", int(key), by_m["layer_ms"],
                         by_m["qkv_ms"], by_m["o_proj_ms"], by_m["sdpa_ms"],
                         by_m["remainder_ms"], "qkv|o_proj|sdpa_unsplit"])
    for key, by_m in decomposition.get("mlp", {}).items():
        rows.append(["mlp", "fused", int(key), by_m["mlp_ms"],
                     by_m["gate_up_ms"], by_m["down_ms"], None,
                     by_m["remainder_ms"], "gate_up|down"])
    for key, value in decomposition.get("lm_head", {}).items():
        rows.append(["lm_head", "qmv_inpath", int(key), value, None, None,
                     None, None, ""])
    return wandb.Table(columns=columns, data=rows)


def reconciliation_table(report: dict) -> wandb.Table | None:
    recon = report.get("insitu_reconciliation")
    if not recon:
        return None
    columns = ["m", "modeled_forward_ms", "insitu_forward_band_arm_ms",
               "modeled_over_insitu_band_arm", "within_15pct",
               "unperturbed_w_forward_ms", "band_arm_over_w_forward",
               "band", "layer_curve", "layer_count", "isolated_layer_ms",
               "phi_call_ms", "modeled_ms", "modeled_qmv_ms",
               "modeled_non_qmv_ms", "modeled_in_layer_other_ms", "insitu_ms",
               "modeled_over_insitu"]
    rows = []
    for row in recon["per_width"]:
        for band, entry in row["bands"].items():
            rows.append([
                row["m"], row["modeled_forward_ms"],
                row["insitu_forward_band_arm_ms"],
                row["modeled_over_insitu_band_arm"], row["within_15pct"],
                row["unperturbed_w_forward_ms"], row["band_arm_over_w_forward"],
                band, entry["layer_curve"], entry["layer_count"],
                entry["isolated_layer_ms"], entry["phi_call_ms"],
                entry["modeled_ms"], entry["modeled_qmv_ms"],
                entry["modeled_non_qmv_ms"],
                entry["modeled_in_layer_other_ms"], entry["insitu_ms"],
                entry["modeled_over_insitu"],
            ])
    return wandb.Table(columns=columns, data=rows)


def slope_closure_table(report: dict) -> wandb.Table | None:
    closure = report.get("slope_closure")
    if not closure:
        return None
    columns = ["boundary", "modeled_forward_step_ms",
               "true_w_leg_forward_step_ms", "residual_ms",
               "residual_over_tolerance", "gdn_mixer_step_ms",
               "gdn_mlp_step_ms", "fa_mixer_step_ms", "fa_mlp_step_ms",
               "lm_head_readout_step_ms"]
    rows = []
    for row in closure:
        band = row["modeled_band_step_ms"]
        rows.append([
            row["boundary"], row["modeled_forward_step_ms"],
            row["true_w_leg_forward_step_ms"], row["residual_ms"],
            row["residual_over_tolerance"],
            band.get("band_gdn_mixer_us"), band.get("band_gdn_mlp_us"),
            band.get("band_fa_mixer_us"), band.get("band_fa_mlp_us"),
            band.get("lm_head_readout"),
        ])
    return wandb.Table(columns=columns, data=rows)


def finding571_table(report: dict) -> wandb.Table | None:
    alloc = report.get("finding571_allocation")
    if not alloc:
        return None
    columns = ["band", "pass_plan_sensitive_ms", "pass_plan_invariant_ms",
               "allocated_to_non_qmv_ms", "if_inside_qmv_ms",
               "residual_ms", "weight_pass_ms", "total_ms"]
    rows = []
    for band, sensitive in alloc["pass_plan_sensitive_ms"].items():
        rows.append([
            band, sensitive, alloc["pass_plan_invariant_ms"][band],
            alloc["allocation_to_non_qmv_families_ms"][band],
            alloc["if_inside_qmv_by_band_ms"][band],
            alloc["non_qmv_residual_ms_per_round"],
            alloc["finding_570_weight_pass_ms_per_round"],
            alloc["finding_543_total_ms_per_round"],
        ])
    return wandb.Table(columns=columns, data=rows)


def pass_contrast_table(report: dict) -> wandb.Table | None:
    alloc = report.get("finding571_allocation")
    if not alloc or "pass_count_contrast" not in alloc:
        return None
    columns = ["cell", "count_per_forward", "groups_m6", "groups_m7",
               "adds_a_pass", "isolated_step_ms", "round_scaled_step_ms",
               "extra_pass_weight_gb"]
    rows = [[r["cell"], r["count_per_forward"], r["groups_m6"], r["groups_m7"],
             r["adds_a_pass"], r["isolated_step_ms"],
             r["round_scaled_step_ms"], r["extra_pass_weight_gb"]]
            for r in alloc["pass_count_contrast"]["cells"]]
    return wandb.Table(columns=columns, data=rows)


def pricing_table(report: dict) -> wandb.Table | None:
    pricing = report.get("census_pricing")
    if not pricing:
        return None
    labels = list(pricing["round_total"])
    columns = ["family", "m1_ms"] + [f"tax_{label}_ms" for label in labels]
    rows = [[family, entry["m1_ms"]]
            + [entry["weighted_tax_ms"][label] for label in labels]
            for family, entry in pricing["families"].items()]
    rows.append(["ROUND_TOTAL", None]
                + [pricing["round_total"][label] for label in labels])
    rows.append(["MEAN_SERVED_WIDTH", None]
                + [sum(int(m) * w for m, w in pricing["shares"][label].items())
                   for label in labels])
    return wandb.Table(columns=columns, data=rows)


def shortlist_table(report: dict) -> wandb.Table | None:
    shortlist = report.get("priced_shortlist")
    if not shortlist:
        return None
    labels = list(shortlist["items"][0]["weighted_tax_ms"])
    columns = (["rank", "mechanism", "shape", "cells"]
               + [f"tax_{label}_ms" for label in labels])
    rows = [[item["rank"], item["mechanism"], item["shape"],
             ",".join(item["cells"])]
            + [item["weighted_tax_ms"][label] for label in labels]
            for item in shortlist["items"]]
    return wandb.Table(columns=columns, data=rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("report")
    ap.add_argument("--name", default=None)
    ap.add_argument("--notes", default=None)
    args = ap.parse_args()

    report = json.loads(Path(args.report).read_text())
    identity = report["identity"]

    config = {
        "experiment": EXPERIMENT,
        "harness": "local",
        "gate_qualified_for_timing": False,
        "cool_gate_passed_real_gate": False,
        "official_or_ranked_score": False,
        "timing_claims_permitted": False,
        "sessions": report["sessions"],
        "noise_floor_ms": report["noise_floor_ms"],
        "base_sha": identity["base_sha"],
        "worker_sha256": identity["worker_sha256"],
        "head_provenance_sha256": identity["head_provenance_sha256"],
        "host": identity["host"],
        "chip": identity["chip"],
        "all_legs_exact": identity["all_legs_exact"],
        "residual_divergence_total": identity["residual_divergence_total"],
        "digest_stable_all_legs": identity["digest_stable_all_legs"],
        "tokens_per_leg": 512,
        "fixture": "public_longcopy_gate_english_512_1024",
    }
    if "e186" in report:
        config["e186_blocks"] = report["e186"].get("blocks")
        config["e186_qmv_arm"] = report["e186"].get("qmv_arm")
        config["e186_eval_floor_us"] = report["e186"].get("eval_floor_us")

    run = wandb.init(
        entity=ENTITY, project=PROJECT, group=GROUP, job_type="census",
        name=args.name or "e218-width-tax-census",
        notes=args.notes or ("E218: per-family dR/dm census of the verify "
                             "round. harness=local, ungated, attribution only."),
        config=config,
    )

    tables = {
        "legs": leg_table(report),
        "width_table": width_table(report),
        "band_table": band_table(report),
        "family_slopes": slope_table(report),
        "perturbation": perturbation_table(report),
    }
    optional = {
        "isolated_cells": cell_table(report),
        "insitu_reconciliation": reconciliation_table(report),
        "slope_closure": slope_closure_table(report),
        "finding571_allocation": finding571_table(report),
        "pass_count_contrast": pass_contrast_table(report),
        "census_pricing": pricing_table(report),
        "priced_shortlist": shortlist_table(report),
    }
    tables.update({k: v for k, v in optional.items() if v is not None})
    run.log(tables)
    if "e186" in report:
        recurrence = report["e186_decomposition"].get("recurrence_fit", {})
        alloc = report.get("finding571_allocation", {})
        run.summary.update({
            "recurrence_ms_per_row_per_layer":
                recurrence.get("linear", {}).get("slope_per_row"),
            "recurrence_ms_per_round_per_row":
                (recurrence.get("linear", {}).get("slope_per_row") or 0) * 48,
            "reconciliation_max_abs_deviation": max(
                abs(row["modeled_over_insitu_band_arm"] - 1.0)
                for row in report["insitu_reconciliation"]["per_width"]),
            "slope_closure_max_abs_residual_ms": max(
                abs(row["residual_ms"]) for row in report["slope_closure"]),
            "finding571_pass_plan_invariant_total_ms":
                alloc.get("pass_plan_invariant_total_ms"),
            "finding571_uncovered_ms":
                alloc.get("uncovered_by_non_qmv_families_ms"),
        })

    artifact = wandb.Artifact("e218-width-tax-census", type="census")
    artifact.add_file(args.report)
    run.log_artifact(artifact)
    print(f"e218: wandb run {run.id} {run.url}")
    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
