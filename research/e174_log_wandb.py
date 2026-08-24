#!/usr/bin/env python3
"""E174: publish the fill census and the epilogue-extension price reconciliation.

Two records go to W&B. The first is the live `xs_hit`/`xs_fill` census from this
base, which ledger 332.3 asked for and nobody had read. The second is the
reconciliation that decides the experiment: three matched end-to-end campaign
measurements against the source-derived E173 row that motivated it.

harness=local throughout. No ranked claim is made or implied.
"""

from __future__ import annotations

import json
import pathlib
import subprocess

import wandb

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "out"
PROJECT = "qwen38-mlx-challenge-senpai"
ENTITY = "wandb-applied-ai-team"

# The round the percentages are taken against, `harness=local`, M4 Pro class.
# E135/F41 states it directly for its own session; E160's legs reconstruct to
# the same scale (0.028767 s/token x 512 tokens / 78 rounds = 188.8 ms).
E135_LOCAL_ROUND_US = 198_671.0
E160_LOCAL_ROUND_US = 188_800.0

# Matched end-to-end measurements already in the ledger on this base. Every row
# is a gated ABBA session with a real 40 C gate, quoted with its own interval.
PRIOR_MEASUREMENTS = [
    {
        "source": "E135 / F41 (ledger 57705)",
        "contrast": "fill_noconsume - replica",
        "mechanism": "add a live, never-read chunk-sum fill for all 257 "
        "table-paying cells: the GPU dispatch and its host kernel record",
        "cells": 257,
        "effect_pct_of_candidate_mtp": 0.2213,
        "se_pp": 0.0942,
        "legs": 4,
        "note": "upper bound: the two arms run different pipeline objects",
    },
    {
        "source": "E160 replicate 1 (FINDING 350, ledger 65434)",
        "contrast": "fuse - off",
        "mechanism": "remove 64 of 130 standalone fills by publishing xsums "
        "from the mlp.down producer epilogue",
        "cells": 64,
        "effect_pct_of_candidate_mtp": 0.0214,
        "se_pp": 0.09833,
        "legs": 6,
        "note": "2 sigma [-0.1752, +0.2181]",
    },
    {
        "source": "E160 replicate 2 (FINDING 359, ledger 65873)",
        "contrast": "fuse - off",
        "mechanism": "same mechanism, second session, 12 legs",
        "cells": 64,
        "effect_pct_of_candidate_mtp": -0.0015,
        "se_pp": 0.1022,
        "legs": 12,
        "note": "2 sigma [-0.2059, +0.2029]",
    },
]

# The E173 row that motivated the assignment, and its method tag. RULE 160
# requires a 0.21 factor on this class until a matched end-to-end number
# exists; the rows above are that number.
E173_SOURCE_DERIVED = {
    "gpu_xsums_standalone_fills_ms_per_round": 0.650,
    "host_kernel_record_share_ms_per_round": 0.430,
    "cells": 130,
    "method": "source-derived (producer/consumer census) x in-source measured "
    "fill cost of 5 us each, Qwen35.swift:1732",
}

MINIMUM_USEFUL_EFFECT_PCT = 0.30
LOCAL_PER_LEG_NOISE_FLOOR_PCT = 0.120


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT.parent, capture_output=True, text=True, check=True
    ).stdout.strip()


def main() -> int:
    census_path = next(iter(sorted(OUT.glob("e174-census-*.json"))), None)
    census = json.loads(census_path.read_text()) if census_path else None

    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        name="e174-xsums-epilogue-extension-census-and-verdict",
        job_type="analysis",
        tags=[
            "e174", "qwen-alphonse", "harness-local", "fill-census",
            "no-ranked-claim", "not-gate-qualified", "closed-axis",
        ],
        config={
            "experiment": "e174-xsums-epilogue-extension",
            "assignment_pr": 173,
            "revision": "r0",
            "harness": "local",
            "base_sha": "2b8037c3e5207fa2f5633e61d013a779492b05c2",
            "candidate_sha": git("rev-parse", "HEAD"),
            "branch": "qwen-alphonse/e174-xsums-epilogue-extension",
            "arm_switch": "MLX_E174_XSUMS_SIDECAR",
            "routed_table_paying_cells_per_weight_pass": 257,
            "minimum_useful_effect_pct": MINIMUM_USEFUL_EFFECT_PCT,
            "local_per_leg_noise_floor_pct": LOCAL_PER_LEG_NOISE_FLOOR_PCT,
            "e173_source_derived_row": E173_SOURCE_DERIVED,
        },
    )

    logs: dict[str, object] = {}
    summary: dict[str, object] = {}

    # 1. The live census.
    if census:
        table = wandb.Table(
            columns=[
                "arm", "tag", "hit_per_round", "fill_per_round",
                "table_paying_cells_per_round", "rounds_differenced",
                "all_tokens_matched", "effective_mean_draft_len",
                "accepted_draft_rate", "worker_sha256", "host",
            ]
        )
        for arm, payload in census["arms"].items():
            c = payload["census"]
            table.add_data(
                arm, payload["tag"], c.get("hit_per_round_median"),
                c.get("fill_per_round_median"),
                c.get("table_paying_cells_median"),
                c.get("rounds_differenced"), payload.get("all_tokens_matched"),
                payload.get("effective_mean_draft_len"),
                payload.get("accepted_draft_rate"),
                payload.get("worker_sha256"), payload.get("host"),
            )
        logs["census/arms"] = table
        on = census["arms"]["on"]["census"]
        summary.update(
            {
                "census_hit_per_round": on.get("hit_per_round_median"),
                "census_fill_per_round": on.get("fill_per_round_median"),
                "census_table_paying_cells_per_round": on.get(
                    "table_paying_cells_median"
                ),
                "census_unserved_share": census.get(
                    "unserved_share_of_table_paying_surface"
                ),
                "census_witness_passed": census["passed"],
            }
        )
        for name, ok in census["checks"].items():
            summary[f"census_check_{name}"] = ok

    # 2. The reconciliation that decides the experiment. Every prior row is
    # rescaled to the 130 cells E174 targets, on that row's own cell count.
    table = wandb.Table(
        columns=[
            "source", "contrast", "cells", "effect_pct", "se_pp", "legs",
            "pct_per_cell", "scaled_to_130_cells_pct",
            "scaled_2se_upper_pct", "note", "mechanism",
        ]
    )
    scaled = []
    for row in PRIOR_MEASUREMENTS:
        per_cell = row["effect_pct_of_candidate_mtp"] / row["cells"]
        s = per_cell * 130
        upper = (row["effect_pct_of_candidate_mtp"] + 2 * row["se_pp"]) / row[
            "cells"
        ] * 130
        scaled.append(s)
        table.add_data(
            row["source"], row["contrast"], row["cells"],
            row["effect_pct_of_candidate_mtp"], row["se_pp"], row["legs"],
            per_cell, s, upper, row["note"], row["mechanism"],
        )
    logs["verdict/prior_measurements_scaled_to_130_cells"] = table

    e173_total_ms = (
        E173_SOURCE_DERIVED["gpu_xsums_standalone_fills_ms_per_round"]
        + E173_SOURCE_DERIVED["host_kernel_record_share_ms_per_round"]
    )
    e173_pct = e173_total_ms * 1e3 / E160_LOCAL_ROUND_US * 100
    # E135 measured the same composite for 257 cells. Rescaled to 130 it is the
    # tightest available ceiling on what removing those fills can pay, and it
    # still assumes the replacement epilogue is free, which E160 disproves.
    ceiling_pct = 0.2213 / 257 * 130

    summary.update(
        {
            "e173_predicted_pct_130_cells": e173_pct,
            "e173_predicted_ms_per_round_130_cells": e173_total_ms,
            "measured_ceiling_pct_130_cells_from_e135": ceiling_pct,
            "measured_central_pct_130_cells_from_e160_rep1": scaled[1],
            "measured_central_pct_130_cells_from_e160_rep2": scaled[2],
            "e173_overprediction_factor_vs_e135": e173_pct / ceiling_pct,
            "ceiling_below_minimum_useful_effect": ceiling_pct
            < MINIMUM_USEFUL_EFFECT_PCT,
            "ceiling_below_per_leg_noise_floor": ceiling_pct
            < LOCAL_PER_LEG_NOISE_FLOOR_PCT,
            "e135_local_round_us": E135_LOCAL_ROUND_US,
            "e160_local_round_us": E160_LOCAL_ROUND_US,
            "verdict": "not useful",
            "gpu_timed_legs_spent": 0,
        }
    )

    run.log(logs)
    for key, value in summary.items():
        run.summary[key] = value
    print(f"run_id {run.id}")
    print(f"run_url {run.url}")
    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
