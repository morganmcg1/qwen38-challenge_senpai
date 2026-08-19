#!/usr/bin/env python3
"""Log the E32 register-budget grid to W&B.

E32 measures no time, so there is no metric series to stream: the durable record
is the grid itself, the gate-control verdicts, and the decision table.

  python3 research/e32_wandb_log.py research/e32-rps-grid.json
"""

from __future__ import annotations

import json
import math
import pathlib
import sys

import wandb

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from e32_analysis import alu_per_tile, fit  # noqa: E402

PROJECT = "qwen38-mlx-challenge-senpai"
ENTITY = "wandb-applied-ai-team"
SHIPPED_IPG = {3: 3, 4: 4, 5: 5, 6: 3, 7: 4, 8: 4, 9: 5}


def main() -> None:
    path = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "research/e32-rps-grid.json")
    data = json.loads(path.read_text())
    cells = [c for c in data["cells"] if c["status"] == "ok"]
    relaxed = {(c["na"], c["r"]): c for c in cells if c["arm"] == "grid_relaxed"}
    blocked = {(c["na"], c["r"]): c for c in cells if c["arm"] == "coverage_preserving"}

    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        name="e32-crossrow-register-budget-trade",
        job_type="static-analysis",
        tags=["e32", "crossrow-qmv", "register-budget", "zero-gpu", "compile-only"],
        config={
            "assignment_id": "qwen38-r1-e32-crossrow-register-budget-trade",
            "base_sha": "d2fcebb0796926962016b87060f9580b9bca89d4",
            "student": "qwen-askeladd",
            "host": "Apple M4 Pro",
            "metal_version": "32023.883",
            "pipeline": "metal -std=metal3.1 -O2 -S | metal-opt -passes=default<O3>",
            "gpu_timing_performed": False,
            "shipped_static_assert": "NA >= 2 && NA <= 5",
            "probe_static_assert": "NA >= 2 && NA <= 16",
            "host_dispatch_bn": 8,
            "host_dispatch_num_simdgroups": 2,
            "host_dispatch_editable": False,
            "mlp_out_vec_size": 17408,
            "shipped_ipg_table": SHIPPED_IPG,
        },
    )

    grid = wandb.Table(columns=["cell", "arm", "NA", "rows_per_simd", "acc_floats",
                                "peak_live_regs", "allocas", "alloca_types",
                                "acc_spill", "air_lines", "device_loads",
                                "threadgroup_refs"])
    for c in sorted(cells, key=lambda c: (c["arm"], c["na"], c["r"])):
        grid.add_data(c["name"], c["arm"], c["na"], c["r"], c["na"] * c["r"],
                      c["peak_live_regs"], c["allocas"], ",".join(c["alloca_types"]),
                      c["acc_spill"], c["air_lines"], c["device_loads"],
                      c["threadgroup_refs"])

    controls = wandb.Table(columns=["cell", "kind", "expected_spill", "observed_spill", "pass"])
    for c in cells:
        if "expect_acc_spill" in c:
            controls.add_data(c["name"], c["arm"], c["expect_acc_spill"],
                              c["acc_spill"], c["expect_acc_spill"] == c["acc_spill"])

    decision = wandb.Table(columns=["M", "shipped_IPG", "shipped_passes", "best_NA",
                                    "best_rows_per_simd", "passes", "alu_per_tile",
                                    "shipped_alu_per_tile", "alu_ratio", "spill_free"])
    max_na = {r: max((na for (na, rr), c in relaxed.items() if rr == r and not c["acc_spill"]),
                     default=0) for r in (1, 2, 3, 4)}
    improved_M = []
    for m in range(3, 10):
        sipg, spasses = SHIPPED_IPG[m], math.ceil(m / SHIPPED_IPG[m])
        shipped_alu = sum(alu_per_tile(min(sipg, m - g * sipg), 4) for g in range(spasses))
        best = None
        for ipg in range(2, m + 1):
            for r in (1, 2, 4):
                c = blocked.get((ipg, r))
                if m % ipg == 1 or ipg > max_na[r] or c is None or c["acc_spill"]:
                    continue
                passes = math.ceil(m / ipg)
                alu = sum(alu_per_tile(min(ipg, m - g * ipg), r) for g in range(passes))
                if best is None or (passes, alu) < best[0]:
                    best = ((passes, alu), ipg, r)
        (passes, alu), ipg, r = best
        decision.add_data(m, sipg, spasses, ipg, r, passes, alu, shipped_alu,
                          alu / shipped_alu, True)
        if passes < spasses:
            improved_M.append(m)

    product_pts = [(na * r, c["peak_live_regs"]) for (na, r), c in relaxed.items()
                   if not c["acc_spill"]]
    p0, p1, p_res = fit(product_pts)
    slopes = {}
    for r in (1, 2, 3, 4):
        pts = [(na, c["peak_live_regs"]) for (na, rr), c in relaxed.items()
               if rr == r and not c["acc_spill"]]
        if len(pts) >= 3:
            slopes[r] = fit(pts)[1]
    s_int, s_slope, s_res = fit(list(slopes.items()))

    hist = {1: 1, 3: 5, 4: 5, 5: 23, 6: 4, 7: 6, 8: 34}
    covered = sum(n for depth, n in hist.items() if depth + 1 in improved_M)

    run.log({"register_grid": grid, "gate_controls": controls, "decision_table": decision})
    run.summary.update({
        "max_spill_free_NA_r1": max_na[1],
        "max_spill_free_NA_r2": max_na[2],
        "max_spill_free_NA_r3": max_na[3],
        "max_spill_free_NA_r4": max_na[4],
        "max_spill_free_IPG_under_frozen_grid": max(improved_M) if improved_M else 5,
        "shipped_max_IPG": 5,
        "regs_na9_r2_measured": relaxed[(9, 2)]["peak_live_regs"],
        "regs_na9_r2_advisor_predicted": 125,
        "regs_na9_r2_product_model": p0 + p1 * 18,
        "product_model_intercept": p0,
        "product_model_slope": p1,
        "product_model_max_residual": p_res,
        "slope_model_na_only_per_NA": s_int,
        "slope_model_per_row_per_NA": s_slope,
        "slope_model_max_residual": s_res,
        "gate_control_failures": len(data["gate_validation_failures"]),
        "gate_controls_checked": sum(1 for c in cells if "expect_acc_spill" in c),
        "rounds_with_fewer_weight_passes": covered,
        "rounds_total_in_quoted_histogram": sum(hist.values()),
        "max_total_threads_per_threadgroup_all_cells": 1024,
        "static_threadgroup_memory_bytes_all_cells": 0,
        "prediction_1_na6_r2_spill_free": "CORRECT",
        "prediction_2_na9_r2_spill_free": "CORRECT",
        "prediction_3_affine_in_product": "FALSIFIED",
    })
    print(f"logged {run.url}")
    run.finish()


if __name__ == "__main__":
    main()
