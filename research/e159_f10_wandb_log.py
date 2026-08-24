#!/usr/bin/env python3
"""E159 F10: publish the width-6 decomposition and the NA register census.

Two tables and one verdict, both zero GPU.

  decomposition   the three- and four-parameter fits of the local dense sweep,
                  their per-width residuals and leverages, and the exact
                  collinearity that decides what is identified.
  register_census registers and spill for every shipped and one-pass QMV cell,
                  read from the live Qwen35.swift header through the AGX
                  backend for g16s and g17s.

harness=local for the sweep rows, harness=static_compile for the census rows.
Nothing here is a ranked score.

  python3 research/e159_f10_wandb_log.py --run-name e159-r2-f10-decomposition
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

import wandb

HERE = pathlib.Path(__file__).resolve().parent
ARTIFACTS = HERE / "e159-artifacts"
PROJECT = "qwen38-mlx-challenge-senpai"
ENTITY = "wandb-applied-ai-team"

ASSIGNMENT_ID = "qwen38-r1-e159-every-arm-that-ever-lost-moved-the-same-one-number"
BASE_SHA = "74b5137e583d526fa86d859ea36166a8e2174d85"


def git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=HERE.parent,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def fit_rows(decomp: dict) -> wandb.Table:
    table = wandb.Table(
        columns=[
            "grid",
            "fit",
            "param",
            "estimate_ms",
            "se_ms",
            "t",
            "resid_sd_ms",
            "dof",
            "r2",
            "condition_number",
        ]
    )
    blocks = [
        ("M=2..9", "fit3_a_b_c", decomp["widths_2_to_9"]["fit3_a_b_c"]),
        ("M=2..9", "fit4_a_b_c_s", decomp["widths_2_to_9"]["fit4_a_b_c_s"]),
        ("M=2..9", "fit_no_stream_a_c_s", decomp["widths_2_to_9"]["fit_split_only_a_c_s"]),
        (
            "M=2..8 reachable",
            "fit_a_c_combined_step",
            decomp["drop_the_identifying_point"]["fit_a_c_combined_step"],
        ),
    ]
    for grid, name, fit in blocks:
        for param, beta, se, t in zip(fit["names"], fit["beta"], fit["se"], fit["t"]):
            table.add_data(
                grid,
                name,
                param,
                beta,
                se,
                t,
                fit["resid_sd"],
                fit["dof"],
                fit["r2"],
                fit["condition_number"],
            )
    return table


def residual_rows(decomp: dict) -> wandb.Table:
    full = decomp["widths_2_to_9"]
    table = wandb.Table(
        columns=[
            "M",
            "R_dec_ms",
            "G_streams",
            "split_indicator",
            "resid_3param_ms",
            "resid_4param_ms",
            "resid_no_stream_ms",
            "leverage_4param",
        ]
    )
    for i, m in enumerate(full["widths"]):
        table.add_data(
            m,
            full["R_dec_ms"][i],
            full["G"][i],
            full["split_indicator"][i],
            full["fit3_a_b_c"]["resid"][i],
            full["fit4_a_b_c_s"]["resid"][i],
            full["fit_split_only_a_c_s"]["resid"][i],
            full["fit4_leverage"][str(m)],
        )
    return table


def census_rows(census: dict) -> wandb.Table:
    table = wandb.Table(
        columns=[
            "use_table",
            "cell",
            "M",
            "IPG",
            "NA",
            "role",
            "streams",
            "g16s_registers",
            "g16s_spill_bytes",
            "g17s_registers",
            "g17s_spill_bytes",
            "g17s_text_bytes",
            "g17s_text_sha8",
        ]
    )
    for key, block in census["cells"].items():
        if "error" in block:
            continue
        use_table = key.endswith("True")
        for cell, rec in block.items():
            g16 = rec.get("g16s", {})
            g17 = rec.get("g17s", {})
            table.add_data(
                use_table,
                cell,
                rec["M"],
                rec["IPG"],
                rec["NA"],
                rec["role"],
                rec["streams_ceil_m_over_ipg"],
                g16.get("registers"),
                g16.get("spill_bytes"),
                g17.get("registers"),
                g17.get("spill_bytes"),
                g17.get("text_bytes"),
                g17.get("text_sha8"),
            )
    return table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="e159-r2-f10-decomposition")
    args = parser.parse_args()

    decomp = json.loads((ARTIFACTS / "e159_width6_decomposition.json").read_text())
    census = json.loads((ARTIFACTS / "e159_na_register_census.json").read_text())

    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        name=args.run_name,
        job_type="analysis",
        tags=["e159", "r2", "f10", "width-wall", "register-census", "no-gpu-leg"],
        config={
            "assignment_id": ASSIGNMENT_ID,
            "revision": "r2",
            "pr": 159,
            "base_sha": BASE_SHA,
            "head_sha": git_sha(),
            "law": decomp["law"],
            "gpu_seconds": 0,
            "official_or_ranked_score": False,
            "rule79_not_evidence": True,
            "collinearity_identity": decomp["collinearity"]["identity"],
        },
    )

    run.log(
        {
            "decomposition_fits": fit_rows(decomp),
            "decomposition_residuals": residual_rows(decomp),
            "register_census": census_rows(census),
        }
    )

    full4 = decomp["widths_2_to_9"]["fit4_a_b_c_s"]
    combined = decomp["drop_the_identifying_point"]["fit_a_c_combined_step"]
    tbl = census["cells"]["use_table_True"]
    rec = census["cells"]["use_table_False"]
    run.summary.update(
        {
            "e159_f10_b_stream_ms": full4["beta"][1],
            "e159_f10_b_stream_se": full4["se"][1],
            "e159_f10_s_split_ms": full4["beta"][3],
            "e159_f10_s_split_se": full4["se"][3],
            "e159_f10_s_split_sigma": abs(full4["beta"][3]) / full4["se"][3],
            "e159_f10_leverage_at_M9": decomp["widths_2_to_9"]["fit4_leverage"]["9"],
            "e159_f10_combined_step_ms": combined["beta"][2],
            "e159_f10_combined_step_se": combined["se"][2],
            "e159_f10_c_row_ms": combined["beta"][1],
            "e159_f10_world": "b_and_s_not_separably_identified_on_reachable_grid",
            "e159_f10_na3_g17s_registers_table": tbl["m6_ipg3"]["g17s"]["registers"],
            "e159_f10_na6_g17s_registers_table": tbl["m6_ipg6"]["g17s"]["registers"],
            "e159_f10_na6_g17s_spill_table": tbl["m6_ipg6"]["g17s"]["spill_bytes"],
            "e159_f10_na3_g17s_registers_recompute": rec["m6_ipg3"]["g17s"]["registers"],
            "e159_f10_na6_g17s_registers_recompute": rec["m6_ipg6"]["g17s"]["registers"],
            "e159_f10_na6_g17s_spill_recompute": rec["m6_ipg6"]["g17s"]["spill_bytes"],
            "e159_f10_na6_fits_without_spill": True,
            "e159_f10_rule155_register_figures_reproduced": True,
            "e159_f10_old_census_void": True,
        }
    )

    artifact = wandb.Artifact("e159-f10-analysis", type="analysis")
    for name in (
        "e159_width6_decomposition.json",
        "e159_na_register_census.json",
    ):
        artifact.add_file(str(ARTIFACTS / name), name=name)
    for script in ("e159_width6_decomposition.py", "e159_na_register_census.py"):
        artifact.add_file(str(HERE / script), name=script)
    run.log_artifact(artifact)

    print(json.dumps({"run_id": run.id, "url": run.url}, indent=1))
    run.finish()


if __name__ == "__main__":
    main()
