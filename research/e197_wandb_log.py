#!/usr/bin/env python3
"""Publish the E197 saturating ranked-law refit and cap-8 pricing to W&B.

    usage: research/e197_wandb_log.py [--refit research/out/e197/refit.json]

Every number here is harness=ranked unless the column or key says `local`.
The one harness=local block is the E186 round-shape reconstruction, which is
used only to transfer a 9-row step scenario at the measured 6-row ranked/local
ratio. It is never used as a ranked speed claim.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

import wandb

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
GROUP = "e197-saturating-ranked-law-cap8"


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refit", default="research/out/e197/refit.json")
    args = parser.parse_args()
    d = json.loads(pathlib.Path(args.refit).read_text())

    c = d["constants"]
    mc, mc_a = d["montecarlo"], d["montecarlo_aicc_weights"]
    cells = d["cell_inversion"]
    order = sorted(d["fits"], key=lambda k: d["fits"][k]["aicc"])
    best = order[0]

    config = {
        "experiment": "e197-saturating-ranked-law-cap8",
        "question": "does a saturating ranked round-cost law price cap 6 or "
                    "cap 8 above the paid cap-7 receipt",
        "harness": "ranked",
        "official_score": False,
        "purpose": "desk refit and depth-axis pricing, no GPU run",
        "candidate_head": git("rev-parse", "HEAD"),
        "worktree_dirty": bool(git("status", "--porcelain")),
        "base_sha": "ebed4f803a399ae49df5e957c15772ab9fc7a6b3",
        "paid_receipts": "cap4 90c131dc, cap5 04710829, cap7 5a9f130a",
        "observations": 24,
        "crown_score": c["crown"],
        "receipt_a_cap7": c["receiptA"],
        "receipt_c_cap4": c["cap4"],
        "receipt_e_cap5": c["cap5"],
        "sigma_published": c["sigma_published"],
        "mue_ms": c["mue_ms"],
        "dr6_measured_ms": c["dr6_measured_ms"],
        "best_family_by_aicc": best,
        "candidate_diff_if_paid": "Qwen36MTPBlockSession.swift "
                                  "segmentedVerifyDepthCap 7 -> 8, one line, "
                                  "defined but not built in this experiment",
    }

    run = wandb.init(
        entity=ENTITY, project=PROJECT, group=GROUP,
        name="e197-ranked-law-refit-cap8", job_type="analysis", config=config,
    )

    fit_cols = ["family", "npar", "chi2", "aicc", "wrmse_ms", "dR6_ms",
                "dR9_ms", "R1_ms", "check_grade", "sample_weight",
                "loo_mean_abs_error_pct", "priced_cap6", "priced_cap8",
                "harness"]
    fit_table = wandb.Table(columns=fit_cols)
    for k in order:
        f = d["fits"][k]
        loo = d["loo_percent_error"][k]
        fit_table.add_data(
            k, f["npar"], f["chi2"], f["aicc"], f["wrmse_ms"], f["dR6_ms"],
            f["dR9_ms"], f["R_ms"][0], f["check_grade"], f["sample_weight"],
            sum(abs(v) for v in loo.values()) / len(loo),
            d["priced"][k]["6"], d["priced"][k]["8"], "ranked")

    law_cols = ["rows"] + [f"{k}_ms" for k in order] + ["local_shape_ms",
                                                        "harness"]
    law_table = wandb.Table(columns=law_cols)
    for i in range(len(d["fits"][best]["R_ms"])):
        row = [i + 1] + [d["fits"][k]["R_ms"][i] for k in order]
        row += [d["local_shape_ms"][i], "ranked"]
        law_table.add_data(*row)

    cell_cols = ["cap", "unknown_cell", "breakeven_vs_receiptA_ms",
                 "breakeven_vs_crown_ms", "local_ms", "local_transferred_ms",
                 "harness"]
    cell_table = wandb.Table(columns=cell_cols)
    for cap in ("6", "8"):
        e = cells[cap]
        cell_table.add_data(int(cap), e["cell"], e["thresholds"]["receipt A"],
                            e["thresholds"]["crown"], e["local_ms"],
                            e["local_transferred_ms"], "ranked")

    scen_cols = ["family", "dR9_ms_ranked_fit", "priced_cap8_flat",
                 "dR9_ms_transferred_local", "priced_cap8_stepped", "harness"]
    scen_table = wandb.Table(columns=scen_cols)
    for k in order:
        s = d["step9_scenarios"][k]
        scen_table.add_data(k, s["flat_dr9_ms"], s["flat_cap8"],
                            s["step_dr9_ms"], s["step_cap8"], "ranked")

    s2_cols = ["family", "shave_ms", "cap5", "cap6", "cap7", "cap8",
               "best_cap", "harness"]
    s2_table = wandb.Table(columns=s2_cols)
    for row in d["stage2"]:
        m = row["medians"]
        s2_table.add_data(row["family"], row["shave_ms"], m["5"], m["6"],
                          m["7"], m["8"], row["best_cap"], "ranked")

    run.log({
        "fit_comparison": fit_table,
        "round_cost_law": law_table,
        "single_cell_inversion": cell_table,
        "step9_scenarios": scen_table,
        "stage2_conditional_repricing": s2_table,
        "priced_cap6_median_ranked": mc["6"]["median"],
        "priced_cap8_median_ranked": mc["8"]["median"],
        "priced_cap8_p05": mc["8"]["p05"],
        "priced_cap8_p95": mc["8"]["p95"],
        "p_cap8_beats_crown": mc["8"]["p_beat_crown"],
        "p_cap8_beats_receiptA": mc["8"]["p_beat_A"],
        "p_cap6_beats_crown": mc["6"]["p_beat_crown"],
        "p_cap6_beats_receiptA": mc["6"]["p_beat_A"],
        "p_cap8_beats_crown_aicc_weights": mc_a["8"]["p_beat_crown"],
        "priced_cap8_median_aicc_weights": mc_a["8"]["median"],
        "dR9_breakeven_vs_receiptA_ms": cells["8"]["thresholds"]["receipt A"],
        "dR9_breakeven_vs_crown_ms": cells["8"]["thresholds"]["crown"],
        "dR8_breakeven_vs_receiptA_ms": cells["6"]["thresholds"]["receipt A"],
        "dR9_transferred_local_ms": cells["8"]["local_transferred_ms"],
        "oracle_per_prompt_cap_median": d["oracle_median"],
        "best_family_dR6_ms": d["fits"][best]["dR6_ms"],
    })
    print("wandb run id:", run.id)
    print("wandb run url:", run.url)
    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
