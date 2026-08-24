#!/usr/bin/env python3
"""Publish the E182 round-cost phase decomposition to W&B.

    usage: research/e182_wandb_log.py --report research/out/e182/report.json
                                      --fold research/out/e182/ranked_fold.json

Two harnesses live in one run and every table says which one it belongs to.

`harness=local`: every timing table from the pinned-depth ladder. The legs ran
with `MLXFAST_LOCAL_COOL_GATE=0` under the standing counterbalanced-session
allowance, and the per-round trace is on in every leg, so no number here is a
gate-qualified timing claim or a candidate speed claim. Only the SHAPE of a
phase against verify width M is read from them.

`harness=ranked`: the FINDING 456 convex law, the E177 priced cells, and the
model comparison that folds the measured local shape back into the ranked
estimator. Only shapes cross the boundary; every ranked coefficient is refitted
on ranked data.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

import wandb

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
GROUP = "e182-round-cost-phase-decomposition"

CROWN = 3.7291100105909
BEST_A = 3.70784519415395

PHASES = [
    "round_us",
    "draft_build_us",
    "d_head1_us",
    "d_chain_us",
    "d_submit1_us",
    "d_submit2_us",
    "verify_build_us",
    "eval_wall_us",
    "readout_us",
    "commit_us",
    "upkeep_us",
    "band_pre_us",
    "band_gdn_mixer_us",
    "band_gdn_mlp_us",
    "band_fa_mixer_us",
    "band_fa_mlp_us",
]


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="research/out/e182/report.json")
    parser.add_argument("--fold", default="research/out/e182/ranked_fold.json")
    parser.add_argument("--bands", default="research/out/e182/bands.json")
    parser.add_argument("--mechanism", default="research/out/e182/mechanism.json")
    args = parser.parse_args()

    report = json.loads(pathlib.Path(args.report).read_text())
    fold = json.loads(pathlib.Path(args.fold).read_text())
    bands = json.loads(pathlib.Path(args.bands).read_text())
    mech = json.loads(pathlib.Path(args.mechanism).read_text())

    legs = report["legs"]
    config = {
        "experiment": "e182-round-cost-phase-decomposition",
        "question": "which phase carries the round cost's M dependence",
        "harness": "local",
        "leg_kind": "pinned-depth phase ladder, palindromic session",
        "mode": "qwen-mtp-local-iterate",
        "tokens": 256,
        "legs": len(legs),
        "arms": "tN trace only; bN trace + head sync + per-band device sync",
        "pin": "MLX_E159_FIXED_DRAFT_DEPTH, M = 1 + depth",
        "band_instrument": "MLX_E182_BAND_SYNC, research-only, default off",
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "timing_claims_permitted": False,
        "trace_perturbs_timing": True,
        "band_arm_round_time_is_inflated_by_construction": True,
        "official_score": False,
        "candidate_head": git("rev-parse", "HEAD"),
        "worktree_dirty": bool(git("status", "--porcelain")),
        "host": "apple-m4-pro",
        "ranked_host": "m5-qwen38-27b-mtp",
        "crown_score": CROWN,
        "receipt_a_score": BEST_A,
    }

    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        group=GROUP,
        name="e182-phase-ladder",
        job_type="analysis",
        config=config,
    )

    leg_cols = [
        "leg", "arm", "m", "band_sync", "rounds", "entry_c", "exit_c",
        "exit_code", "round_us_p50", "harness",
    ]
    leg_table = wandb.Table(columns=leg_cols)
    for leg in legs:
        leg_table.add_data(
            leg["leg"], leg["arm"], leg["m"], leg["band_sync"], leg["rounds"],
            leg["entry_c"], leg["exit_c"], leg["exit_code"],
            leg.get("round_us_p50"), "local",
        )

    phase_tables = {}
    for kind, table in report["tables"].items():
        cols = ["m", "groups", "n", "harness"] + [
            f for f in PHASES if any(f in row for row in table)
        ]
        wt = wandb.Table(columns=cols)
        for row in table:
            wt.add_data(
                row["m"], row["groups"], row["n"], "local",
                *[row.get(f) for f in cols[4:]],
            )
        phase_tables[f"phase_table_{kind}"] = wt

    shape_table = wandb.Table(
        columns=[
            "key", "preferred", "smooth_rmse_us", "step_rmse_us",
            "aicc_smooth", "aicc_step", "smooth_coeffs", "step_coeffs",
            "harness",
        ]
    )
    for key, block in sorted(report["shapes"].items()):
        shape_table.add_data(
            key, block["preferred"], block["smooth_rmse_us"],
            block["step_rmse_us"], block["aicc_smooth"], block["aicc_step"],
            json.dumps([round(c, 4) for c in block["smooth_coeffs"]]),
            json.dumps([round(c, 4) for c in block["step_coeffs"]]),
            "local",
        )

    closure_table = wandb.Table(
        columns=["m", "band_sum_us", "verify_device_us", "verify_device_phase",
                 "closure_frac", "harness"]
    )
    for row in report["band_closure"]:
        closure_table.add_data(
            row["m"], row["band_sum_us"], row["verify_device_us"],
            row["verify_device_phase"], row["closure_frac"], "local",
        )

    drift_table = wandb.Table(
        columns=["arm", "m", "leg_first", "leg_second", "round_us_first",
                 "round_us_second", "delta_frac", "entry_c_first",
                 "entry_c_second", "harness"]
    )
    for row in report["abba_drift"]:
        drift_table.add_data(
            row["arm"], row["m"], row["leg_first"], row["leg_second"],
            row["round_us_first"], row["round_us_second"], row["delta_frac"],
            row["entry_c_first"], row["entry_c_second"], "local",
        )

    model_cols = [
        "model", "wrmse_ms", "max_resid_ms", "aic", "r1_anchor_err_ms",
        "a_only_cap4_err_pct", "priced_cap5", "priced_cap6", "priced_cap8",
        "params", "harness",
    ]
    model_table = wandb.Table(columns=model_cols)
    for key, block in sorted(
        fold["models"].items(), key=lambda kv: kv[1]["wrmse_ms"]
    ):
        model_table.add_data(
            key, block["wrmse_ms"], block["max_resid_ms"], block["aic"],
            block["r1_anchor_err_ms"], block.get("a_only_cap4_err_pct"),
            block["priced"]["5"], block["priced"]["6"], block["priced"]["8"],
            json.dumps(
                dict(zip(block["names"], [round(b, 4) for b in block["beta_ms"]]))
            ),
            "ranked",
        )

    width_cols = ["m"] + sorted(fold["models"]) + ["harness"]
    width_table = wandb.Table(columns=width_cols)
    for m in range(1, 10):
        width_table.add_data(
            m,
            *[fold["models"][k]["r_by_width_ms"][str(m)]
              for k in sorted(fold["models"])],
            "ranked",
        )

    band_cols = [
        "m", "gdn_mixer_ms", "gdn_mlp_ms", "fa_mixer_ms", "fa_mlp_ms",
        "pre_ms", "verify_ms", "head_ms", "host_ms", "gdn_per_layer_ms",
        "fa_per_layer_ms", "gdn_over_fa_per_layer", "gdn_probe_ms",
        "gdn_probe_share_of_band", "harness",
    ]
    band_table = wandb.Table(columns=band_cols)
    for m in range(1, 10):
        b = bands["bands_ms"][str(m)]
        pl = bands["per_layer_ms"][str(m)]
        probe = bands["gdn_probe_ms"][str(m)]
        band_table.add_data(
            m, b["gdn_mixer"], b["gdn_mlp"], b["fa_mixer"], b["fa_mlp"],
            b["pre"], b["verify"], b["head"], b["host"], pl["gdn"], pl["fa"],
            pl["ratio"], probe, probe / b["gdn_mixer"], "local",
        )

    shipped_table = wandb.Table(
        columns=["m", "groups", "round_ms", "draft_ms", "device_ms",
                 "host_ms", "harness"]
    )
    groups_by_m = {row["m"]: row["groups"] for row in report["tables"]["trace"]}
    for m in range(1, 10):
        s = bands["shipped_round_ms"][str(m)]
        shipped_table.add_data(
            m, groups_by_m[m], s["round"], s["draft"], s["device"], s["host"],
            "local",
        )

    mech_table = wandb.Table(
        columns=["law", "saved_m6_ms", "saved_m7_ms", "saved_m8_ms",
                 "published_base", "published_ceiling", "gain_pct",
                 "wide_round_share", "crown", "clears_crown", "harness"]
    )
    for law, block in sorted(
        mech["laws"].items(), key=lambda kv: kv[1]["gain_pct"]
    ):
        mech_table.add_data(
            law, block["saved_ms"]["6"], block["saved_ms"]["7"],
            block["saved_ms"]["8"], block["published_base"],
            block["published_ceiling"], block["gain_pct"],
            block["wide_round_share"], mech["crown"],
            block["published_ceiling"] > mech["crown"], "ranked",
        )

    payload = {
        "legs": leg_table,
        "shape_verdicts": shape_table,
        "band_closure": closure_table,
        "abba_drift": drift_table,
        "ranked_model_comparison": model_table,
        "ranked_r_by_width_ms": width_table,
        "band_decomposition": band_table,
        "shipped_round_by_width": shipped_table,
        "mechanism_ceiling": mech_table,
    }
    payload.update(phase_tables)
    run.log(payload)

    trace_by_m = {row["m"]: row for row in report["tables"]["trace"]}
    band_by_m = {row["m"]: row for row in report["tables"]["band"]}
    summary = {
        "local/round_us_m1": trace_by_m[1]["round_us"],
        "local/round_us_m9": trace_by_m[9]["round_us"],
        "local/round_growth_m1_to_m9": (
            trace_by_m[9]["round_us"] / trace_by_m[1]["round_us"]
        ),
        "local/band_closure_min": min(
            row["closure_frac"] for row in report["band_closure"]
        ),
        "local/band_closure_max": max(
            row["closure_frac"] for row in report["band_closure"]
        ),
        "local/abba_max_abs_drift_frac": max(
            abs(row["delta_frac"]) for row in report["abba_drift"]
        ),
        "ranked/quadratic_wrmse_ms": fold["models"]["quadratic"]["wrmse_ms"],
        "ranked/best_model": min(
            fold["models"], key=lambda k: fold["models"][k]["wrmse_ms"]
        ),
        "crown_score": CROWN,
        "receipt_a_score": BEST_A,
    }
    for key in ("phase-local", "phase-local+rows", "phase-split"):
        if key in fold["models"]:
            summary[f"ranked/{key}_wrmse_ms"] = fold["models"][key]["wrmse_ms"]
            summary[f"ranked/{key}_aic"] = fold["models"][key]["aic"]
    for field in ("band_gdn_mlp_us", "band_gdn_mixer_us", "band_fa_mixer_us"):
        if field in band_by_m[9] and field in band_by_m[1]:
            summary[f"local/{field}_m9_over_m1"] = (
                band_by_m[9][field] / band_by_m[1][field]
            )

    b1, b9 = bands["bands_ms"]["1"], bands["bands_ms"]["9"]
    growth = b9["verify"] + b9["head"] + b9["host"] - b1["verify"]
    summary.update({
        "local/growth_m1_to_m9_ms": growth,
        "local/growth_share_verify": (b9["verify"] - b1["verify"]) / growth,
        "local/growth_share_head": b9["head"] / growth,
        "local/growth_share_host": b9["host"] / growth,
        "local/gdn_probe_share_of_gdn_mixer_growth": (
            (bands["gdn_probe_ms"]["9"] - bands["gdn_probe_ms"]["1"])
            / (b9["gdn_mixer"] - b1["gdn_mixer"])
        ),
        "local/per_layer_gdn_over_fa_m1": bands["per_layer_ms"]["1"]["ratio"],
        "local/per_layer_gdn_over_fa_m9": bands["per_layer_ms"]["9"]["ratio"],
        "ranked/mechanism_wide_round_share": mech["wide_round_share"],
        "ranked/mechanism_paid_cap7": mech["paid_cap7"],
    })
    for law, block in mech["laws"].items():
        summary[f"ranked/mechanism_ceiling_{law}"] = block["published_ceiling"]
        summary[f"ranked/mechanism_gain_pct_{law}"] = block["gain_pct"]
    run.summary.update(summary)

    print(f"logged {run.url}")
    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
