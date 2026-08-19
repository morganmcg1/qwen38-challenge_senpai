#!/usr/bin/env python3
"""Log the E36 values_per_thread x NA x rows_per_simd grid to W&B.

E36 measures no time, so there is no metric series to stream: the durable record
is the grid, the gate-control verdicts, the K-coverage table, the ranked-board
prior art and the composition decision.

  python3 research/e36_wandb_log.py research/e36-vpt-grid.json
  python3 research/e36_wandb_log.py research/e36-vpt-grid.json \
      --board research/e36-board-vpt-evidence.json --resume <run_id>
"""

from __future__ import annotations

import json
import math
import pathlib
import sys

import wandb

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from e36_analysis import SCORED_SHAPES, fit  # noqa: E402

PROJECT = "qwen38-mlx-challenge-senpai"
ENTITY = "wandb-applied-ai-team"
V_VALUES = [8, 16, 32, 64]
SHIPPED_IPG = {3: 3, 4: 4, 5: 5, 6: 3, 7: 4, 8: 4, 9: 5}
E27_LADDER = {2: 62, 3: 83, 4: 104, 5: 125}


def _flag(name: str) -> str | None:
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else None


def board_table(path: pathlib.Path):
    """Ranked prior art: the 13 board notes that mention values_per_thread."""
    doc = json.loads(path.read_text())
    table = wandb.Table(columns=[
        "submission_id", "solver", "created_at", "official_score", "status",
        "promotion_status", "rejection_reason", "excerpt",
    ])
    for s in doc["submissions"]:
        table.add_data(
            s["id"][:8], s["solver"], s["created_at"][:19],
            s["official_score"] if s["official_score"] is not None else float("nan"),
            s["status"], str(s["promotion_status"]),
            (s["rejection_reason"] or "")[:180],
            (s["excerpts"][0] if s["excerpts"] else "")[:900],
        )
    failures = [f["id"][:8] for f in doc["verify_path_parity_failures"]]
    return table, doc, failures


def main() -> None:
    positional = [a for a in sys.argv[1:] if not a.startswith("--")
                  and a not in {_flag("--board"), _flag("--resume")}]
    path = pathlib.Path(positional[0] if positional else "research/e36-vpt-grid.json")
    data = json.loads(path.read_text())
    cells = [c for c in data["cells"] if c["status"] == "ok"]
    by = {(c["arm"], c["na"], c["r"], c["v"]): c for c in cells}
    relaxed = {(c["na"], c["r"], c["v"]): c for c in cells if c["arm"] == "grid_relaxed"}
    blocked = {(c["na"], c["r"], c["v"]): c for c in cells
               if c["arm"] == "coverage_preserving"}

    anchors = {na: by[("shipped_anchor", na, 4, 16)]["peak_live_regs"] for na in E27_LADDER}
    if anchors != E27_LADDER:
        sys.exit(f"E27 anchors did not reproduce: {anchors} vs {E27_LADDER}; nothing to log")

    k_legal = {v: all(k % (v * 32) == 0 for _, k, _, _ in SCORED_SHAPES) for v in V_VALUES}

    resume_id = _flag("--resume")
    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        id=resume_id,
        resume="must" if resume_id else None,
        name="e36-values-per-thread-composes-with-na",
        job_type="static-analysis",
        tags=["e36", "crossrow-qmv", "values-per-thread", "register-budget",
              "zero-gpu", "compile-only"],
        config={
            "assignment_id": "qwen38-r1-e36-values-per-thread-composes-with-na",
            "base_sha": "4e5dc2bdc9ed7b89c1b3c75a7fc0620e97d43549",
            "student": "qwen-askeladd",
            "host": "Apple M4 Pro",
            "metal_version": "32023.883",
            "pipeline": "metal -std=metal3.1 -O2 -S | metal-opt -passes=default<O3>",
            "gpu_timing_performed": False,
            "shipped_values_per_thread": 16,
            "shipped_bytes_per_lane": 8,
            "swept_values_per_thread": V_VALUES,
            "swept_NA": sorted({c["na"] for c in cells if c["arm"] == "grid_relaxed"}),
            "swept_rows_per_simd": sorted({c["r"] for c in cells}),
            "generator_rewrites": 10,
            "extraction_reused_from": "research/crossrow_rps_gen.py:extract (E32, unchanged)",
            "host_dispatch_editable": False,
            "host_fast_gate": "N % 8 == 0 && K % 512 == 0",
            "scored_K_values": sorted({k for _, k, _, _ in SCORED_SHAPES}),
            "hidden_size": 5120,
            "intermediate_size": 17408,
            "vocab_size": 248320,
            "quant_group_size": 64,
            "quant_bits": 4,
        },
    )

    grid = wandb.Table(columns=[
        "cell", "arm", "NA", "rows_per_simd", "values_per_thread", "bytes_per_lane",
        "peak_live_regs", "allocas", "alloca_types", "private_bytes", "stage_bytes",
        "acc_spill", "air_lines", "device_loads", "threadgroup_refs",
        "k_blocks_K5120", "k_blocks_K17408"])
    for c in sorted(cells, key=lambda c: (c["arm"], c["v"], c["na"], c["r"])):
        block = c["v"] * 32
        grid.add_data(
            c["name"], c["arm"], c["na"], c["r"], c["v"], c["v"] // 2,
            c["peak_live_regs"], c["allocas"], ",".join(c["alloca_types"]),
            c["private_bytes"], c["stage_bytes"], c["acc_spill"], c["air_lines"],
            c["device_loads"], c["threadgroup_refs"],
            5120 / block, 17408 / block)

    controls = wandb.Table(columns=["cell", "kind", "field", "expected", "observed", "pass"])
    for c in cells:
        for key in [k for k in c if k.startswith("expect_")]:
            field = key[len("expect_"):]
            controls.add_data(c["name"], c["arm"], field, str(c[key]),
                              str(c[field]), c[key] == c[field])

    # Registers are invariant to values_per_thread; private bytes are not. Log
    # both so the separability claim is checkable from the run alone.
    invariance = wandb.Table(columns=["arm", "NA", "rows_per_simd", "regs_v8",
                                      "regs_v16", "regs_v32", "regs_v64",
                                      "max_abs_delta_vs_v16", "priv_v8", "priv_v16",
                                      "priv_v32", "priv_v64"])
    max_delta_ge4 = 0
    max_delta_v32_ge4 = 0
    for arm, src in (("grid_relaxed", relaxed), ("coverage_preserving", blocked)):
        for na in sorted({k[0] for k in src}):
            for r in sorted({k[1] for k in src if k[0] == na}):
                row = [src.get((na, r, v)) for v in V_VALUES]
                if any(c is None for c in row) or row[1]["acc_spill"]:
                    continue
                regs = [c["peak_live_regs"] for c in row]
                delta = max(abs(x - regs[1]) for x in regs)
                if na >= 4:
                    max_delta_ge4 = max(max_delta_ge4, delta)
                    max_delta_v32_ge4 = max(max_delta_v32_ge4, abs(regs[2] - regs[1]))
                invariance.add_data(arm, na, r, *regs, delta,
                                    *[c["private_bytes"] for c in row])

    coverage = wandb.Table(columns=["N", "K", "projection", "K_mod_512", "K_mod_1024",
                                    "K_mod_2048", "v8_ok", "v16_ok", "v32_ok", "v64_ok"])
    for n, k, name, _src in SCORED_SHAPES:
        coverage.add_data(n, k, name, k % 512, k % 1024, k % 2048,
                          *[k % (v * 32) == 0 for v in V_VALUES])

    max_na = {}
    for arm, src in (("grid_relaxed", relaxed), ("coverage_preserving", blocked)):
        for r in sorted({k[1] for k in src}):
            for v in V_VALUES:
                best = 0
                for na in sorted({k[0] for k in src}):
                    c = src.get((na, r, v))
                    if c is None:
                        continue
                    if c["acc_spill"]:
                        break
                    best = na
                max_na[(arm, r, v)] = best

    decision = wandb.Table(columns=["M", "shipped_IPG", "shipped_passes", "best_NA",
                                    "passes", "widest_K_legal_vpt", "bytes_per_lane",
                                    "peak_live_regs", "private_bytes", "spill_free",
                                    "recommended_vpt", "recommendation_reason"])
    best_v = max(v for v in V_VALUES if k_legal[v])
    for m in (6, 9):
        sipg = SHIPPED_IPG[m]
        na = max(n for n in range(2, min(m, max_na[("coverage_preserving", 2, 32)]) + 1)
                 if m % n != 1)
        c = blocked[(na, 2, best_v)]
        decision.add_data(m, sipg, math.ceil(m / sipg), na, math.ceil(m / na), best_v,
                          best_v // 2, c["peak_live_regs"], c["private_bytes"],
                          not c["acc_spill"], 16,
                          "lane->K repartition breaks the kernel's own stated "
                          "K-accumulation-order invariant (quantized.h:966) and "
                          "desynchronises M>=2 from the M==1 qmv_fast_impl path")

    slopes = {}
    for r in (1, 2, 3, 4):
        for v in V_VALUES:
            avail = [na for na in range(2, 11)
                     if (na, r, v) in relaxed and not relaxed[(na, r, v)]["acc_spill"]]
            use = [na for na in avail if na >= 4]
            if len(use) < 3:
                use = avail
            if len(use) >= 3:
                slopes[(r, v)] = fit(use, [relaxed[(na, r, v)]["peak_live_regs"]
                                           for na in use])[1]
    s_int, s_slope, s_res = fit([r for r in (1, 2, 3, 4) if (r, 16) in slopes],
                                [slopes[(r, 16)] for r in (1, 2, 3, 4) if (r, 16) in slopes])
    stage_exact = all(
        c["stage_bytes"] == c["r"] * c["v"] // 2
        for c in cells if c["arm"] in ("grid_relaxed", "coverage_preserving")
        and not c["acc_spill"] and c["stage_bytes"] > 0)

    advisor_slope = 8.36 * 2 + 3.19 * 2

    run.log({"register_grid": grid, "gate_controls": controls,
             "vpt_invariance": invariance, "k_coverage": coverage,
             "decision_table": decision})
    run.summary.update({
        # Primary metric.
        "e36/max_spill_free_NA_at_values_per_thread_32":
            max_na[("coverage_preserving", 2, 32)],
        "e36/max_spill_free_NA_at_values_per_thread_16":
            max_na[("coverage_preserving", 2, 16)],
        "e36/max_spill_free_NA_baseline_shipped_r4_v16":
            max_na[("coverage_preserving", 4, 16)],
        "e36/vpt_attributable_delta_in_max_spill_free_NA":
            max_na[("coverage_preserving", 2, 32)] - max_na[("coverage_preserving", 2, 16)],
        "max_spill_free_NA_grid_relaxed_r2_v32": max_na[("grid_relaxed", 2, 32)],
        # (d) composition in the register file.
        "max_abs_reg_delta_vs_v16_NA_ge4": max_delta_ge4,
        "max_abs_reg_delta_v32_vs_v16_NA_ge4": max_delta_v32_ge4,
        "regs_na6_r2_v16": relaxed[(6, 2, 16)]["peak_live_regs"],
        "regs_na6_r2_v32_measured": relaxed[(6, 2, 32)]["peak_live_regs"],
        "regs_na6_r2_v32_advisor_predicted_quoted": 196,
        "regs_na6_r2_v32_advisor_formula": 16.0 + advisor_slope * 6,
        "slope_model_na_only_per_NA": s_int,
        "slope_model_per_row_per_NA": s_slope,
        "slope_model_max_residual": s_res,
        "stage_bytes_equals_r_times_vpt_over_2": stage_exact,
        # (c) host legality.
        "host_fast_gate_guarantees_K_mod": 512,
        "all_scored_K_cover_v32": k_legal[32],
        "all_scored_K_cover_v64": k_legal[64],
        "widest_K_legal_values_per_thread": best_v,
        "K_values_breaking_v64": sorted({k for _, k, _, _ in SCORED_SHAPES
                                         if k % 2048 != 0}),
        "host_derives_size_from_values_per_thread": False,
        "output_grid_depends_on_values_per_thread": False,
        # Gates.
        "gate_control_failures": len(data["gate_validation_failures"]),
        "gate_controls_checked": sum(
            1 for c in cells if any(k.startswith("expect_") for k in c)),
        "e27_anchors_reproduced": anchors == E27_LADDER,
        "threadgroup_memory_bytes_all_cells": 0,
        # Verdict.
        "prediction_advisor_xside_scales_with_vpt": "FALSIFIED",
        "prediction_axes_contend_for_registers": "FALSIFIED",
        "verdict_axes_compose_in_registers": True,
        "verdict_vpt_recommended_for_verify_path": False,
        "verdict_blocking_constraint": "fp32 accumulation order, not registers",
    })

    board = _flag("--board")
    if board:
        table, doc, failures = board_table(pathlib.Path(board))
        run.log({"board_prior_art": table})
        run.summary.update({
            "board_rows_total": doc["board_rows_total"],
            "board_notes_mentioning_values_per_thread":
                doc["notes_mentioning_values_per_thread"],
            # The decisive external evidence: same axis, same kernel, ranked runner.
            "ranked_verify_path_parity_failures": len(failures),
            "ranked_verify_path_parity_failure_ids": ",".join(failures),
            "ranked_failure_gate":
                "Qwen-MTP correctness and parity gate (untimed)",
            "ranked_failure_direction_includes_lower_vpt": True,
            "lesson_row_score":
                doc["lesson_recorded_by_next_accepted_row"]["official_score"],
            "verdict_axis_closed_by_ranked_evidence": True,
        })

    print(f"logged {run.url}")
    run.finish()


if __name__ == "__main__":
    main()
