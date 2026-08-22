#!/usr/bin/env python3
"""Publish the E138 plan surface to W&B, one run per rung.

    usage: research/e138_wandb_log.py --rung a|b|c [--dry]

E138 asks whether any bit-exact `(IPG, RPS)` plan beats the promoted
`onePass67` table, and in particular whether any plan flattens the width-6
step.

No rung here is a gated measurement or a score. Rung `a` is an offline
compiler census and touches no GPU. Rungs `b` and `c` use the GPU under the
standing ungated local-arm conditions, so every run publishes
`cool_gate_passed_real_gate`, `gate_qualified_for_timing` and
`official_or_ranked_score` verbatim as false, together with the entry and exit
GPU temperature of each block.

FINDING 181 and CAMPAIGN RULE 83 apply to every number below. This host is
`applegpu_g16s` with a 96-register ceiling; the ranked host is g17s at 126.
Any cell whose g16s and g17s spill state differ is labelled
`transfer_safe = false` and must not be promoted on this evidence alone.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import wandb

import e138_plan_analysis as analysis

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
GROUP = "e138-plan-surface-at-the-width-6-cliff"
HOST = "apple-m4-pro-applegpu_g16s-20core-48gib"
BASE_SHA = "328c4b9eac1b386f0c0913afcf0c7a64c232e5c0"
ART = pathlib.Path("research/e138-artifacts")

SHIPPED = {5: "5:5:4", 6: "6:6:4", 7: "7:7:4", 8: "8:4:4", 9: "9:3:4"}

# Pre-registered before any cell was measured, in the `e138-prereg-1` comment.
PREREG_STUDENT_GLOBAL_PCT = 0.0
PREREG_STUDENT_KEYED_PCT = 11.0
PREREG_STUDENT_KEYED_LO = 4.0
PREREG_STUDENT_KEYED_HI = 20.0
# The advisor took the other side in feedback F1 section 4.
PREREG_ADVISOR_KEYED_LO = 15.0
PREREG_ADVISOR_KEYED_HI = 30.0
# Both sides agreed this closes the plan axis.
PLAN_AXIS_CLOSE_PCT = 3.0

RUNGS = {
    "a": {
        "run_name": "e138-runga-offline-plan-register-census",
        "job_type": "offline-census",
        "question":
            "what does every legal (M, IPG, RPS) cell cost in registers and "
            "spill on applegpu_g16s and applegpu_g17s",
        "command": "python3 research/e138_plan_census.py",
        "uses_gpu": False,
    },
    "b": {
        "run_name": "e138-rungb-isolated-plan-surface-m5-m7",
        "job_type": "isolated-kernel-timing",
        "question":
            "at the shipped tight launch, is there an (IPG, RPS) plan for "
            "widths 5 to 7 that beats the promoted onePass67 plan, and does "
            "any plan flatten the width-6 step",
        "command":
            "research/e138_plan_surface.sh OUT CELLS '' 31 12 tight",
        "artifacts": [
            "item1-decisive-cells.json",
            "item1a-m5-m7-rows.json",
        ],
        "uses_gpu": True,
    },
    "c": {
        "run_name": "e138-rungc-spill-control-and-stock-reference",
        "job_type": "isolated-kernel-timing",
        "question":
            "is the M=7 two-pass win caused by the NA=7 register spill on "
            "this host, and was the E137 fallback column a plan effect or a "
            "replica-versus-stock code difference",
        "command":
            "research/e138_plan_surface.sh OUT CELLS '' 31 12 tight",
        "artifacts": ["item1c-spill-control-and-stock.json"],
        "uses_gpu": True,
    },
}


def flatten(prefix: str, value, out: dict) -> None:
    if isinstance(value, dict):
        for key, sub in value.items():
            flatten("%s/%s" % (prefix, key) if prefix else str(key), sub, out)
    elif isinstance(value, (int, float, str, bool)) or value is None:
        out[prefix] = value


def table(columns: list[str], rows: list[dict]) -> wandb.Table:
    t = wandb.Table(columns=columns)
    for row in rows:
        t.add_data(*[row.get(c) for c in columns])
    return t


def thermal(result: dict) -> dict:
    """The three standing conditions, verbatim, plus the entry spread."""
    entries = [
        s["gpu_temp_entry_c"]
        for s in result["sessions"]
        if s and s.get("gpu_temp_entry_c") is not None
    ]
    exits = [
        s["gpu_temp_exit_c"]
        for s in result["sessions"]
        if s and s.get("gpu_temp_exit_c") is not None
    ]
    out = {
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "arms_abba_counterbalanced_within_block": True,
    }
    if entries:
        out["gpu_temp_entry_c_min"] = min(entries)
        out["gpu_temp_entry_c_max"] = max(entries)
        out["gpu_temp_entry_c_spread"] = max(entries) - min(entries)
    if exits:
        out["gpu_temp_exit_c_max"] = max(exits)
    return out


def rung_a() -> tuple[dict, dict]:
    census = json.loads((ART / "stage-a-plan-census.json").read_text())
    cells = census["cells"]
    rows = []
    for name, cell in sorted(cells.items()):
        row = {
            "cell": name.replace("e138_qmv_", ""),
            "m": cell["m"],
            "ipg": cell["ipg"],
            "rps": cell["rps"],
            "passes": cell["passes"],
            "launched_columns": cell["launched_columns"],
            "built": cell["built"],
        }
        for arch in ("applegpu_g16s", "applegpu_g17s"):
            block = cell.get(arch) or {}
            for key in ("registers", "spill_bytes", "resident_simdgroups",
                        "text_bytes"):
                row["%s_%s" % (arch, key)] = block.get(key)
        # RULE 83: a cell that spills on one host and not the other cannot
        # carry a closure from that host to the other.
        g16 = (cell.get("applegpu_g16s") or {}).get("spill_bytes")
        g17 = (cell.get("applegpu_g17s") or {}).get("spill_bytes")
        row["spill_state_matches_across_hosts"] = (
            None if g16 is None or g17 is None else (g16 > 0) == (g17 > 0)
        )
        rows.append(row)

    spill16 = [r for r in rows if (r["applegpu_g16s_spill_bytes"] or 0) > 0]
    spill17 = [r for r in rows if (r["applegpu_g17s_spill_bytes"] or 0) > 0]
    summary = {
        "census_cells": len(rows),
        "census_cells_built": sum(1 for r in rows if r["built"]),
        "census_cells_spilling_g16s": len(spill16),
        "census_cells_spilling_g17s": len(spill17),
        "census_cells_spill_state_disagrees": sum(
            1 for r in rows if r["spill_state_matches_across_hosts"] is False
        ),
        "census_max_registers_g16s": max(
            r["applegpu_g16s_registers"] for r in rows
        ),
        "census_max_registers_g17s": max(
            r["applegpu_g17s_registers"] for r in rows
        ),
        "uses_gpu": False,
    }
    return summary, {
        "plan_register_census": table(sorted(rows[0]), rows),
    }


def rung_timing(files: list[str]) -> tuple[dict, dict]:
    paths = [ART / f for f in files]
    result = analysis.analyse(analysis.load(paths), SHIPPED)

    cell_rows = []
    for key, entry in result["cells"].items():
        row = dict(entry)
        row["key"] = key
        row["calls_per_verify"] = result["shapes"][entry["shape"]][
            "calls_per_verify"
        ]
        row["weighted_us"] = entry["us_per_call"] * row["calls_per_verify"]
        cell_rows.append(row)

    width_rows = []
    for m, block in sorted(result["per_width"].items()):
        for cell, total in sorted(block["global_totals_us"].items()):
            width_rows.append({
                "m": m,
                "cell": cell,
                "weighted_total_us": total,
                "is_shipped": cell == block["shipped_cell"],
                "is_stock": cell.endswith(":stock"),
                "delta_vs_shipped_us": (
                    None
                    if block["shipped_total_us"] is None
                    else total - block["shipped_total_us"]
                ),
            })

    summary = {}
    flatten("per_width", {
        str(m): {
            k: v
            for k, v in block.items()
            if k not in ("cells", "stock_cells", "global_totals_us",
                         "shape_keyed_choice", "shape_keyed_distinct_plans",
                         "stock_total_us")
        }
        for m, block in result["per_width"].items()
    }, summary)
    flatten("step", result.get("step", {}), summary)
    flatten("ranked_pricing", result["ranked_pricing"], summary)
    flatten("pipeline_count", {
        k: v
        for k, v in result["pipeline_count"].items()
        if not isinstance(v, list)
    }, summary)
    summary.update(thermal(result))

    if "e138_best_plan_isolated_step_reduction_pct" in result:
        best = result["e138_best_plan_isolated_step_reduction_pct"]
        summary["e138_best_plan_isolated_step_reduction_pct"] = best
        summary["e138_cliff_is_plan_invariant"] = result[
            "e138_cliff_is_plan_invariant"
        ]
        summary["prereg_student_keyed_pct"] = PREREG_STUDENT_KEYED_PCT
        summary["prereg_student_keyed_hit"] = (
            PREREG_STUDENT_KEYED_LO <= best <= PREREG_STUDENT_KEYED_HI
        )
        summary["prereg_advisor_keyed_hit"] = (
            PREREG_ADVISOR_KEYED_LO <= best <= PREREG_ADVISOR_KEYED_HI
        )
        summary["plan_axis_closed"] = best < PLAN_AXIS_CLOSE_PCT

    summary["cells_measured"] = len(cell_rows)
    summary["cells_bitwise_identical_to_incumbent"] = sum(
        1 for r in cell_rows if r["matches_incumbent_bitwise"]
    )
    summary["max_abs_delta_vs_incumbent"] = max(
        r["max_abs_delta_vs_incumbent"] for r in cell_rows
    )
    summary["problems"] = len(result["problems"])
    summary["uses_gpu"] = True

    return summary, {
        "plan_cells": table(sorted(cell_rows[0]), cell_rows),
        "weighted_totals": table(sorted(width_rows[0]), width_rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rung", required=True, choices=sorted(RUNGS))
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()

    spec = RUNGS[args.rung]
    if args.rung == "a":
        summary, tables = rung_a()
    else:
        summary, tables = rung_timing(spec["artifacts"])

    config = {
        "experiment": "e138",
        "rung": args.rung,
        "question": spec["question"],
        "command": spec["command"],
        "base_sha": BASE_SHA,
        "host": HOST,
        "chip": "apple-m4-pro",
        "ranked_chip": "apple-m5-applegpu_g17s",
        "harness": "local",
        "transfer_safe": False,
        "toolchain": "apple-swift-6.3.3-metal-32023.883",
        "instrument": "Tests/MLXFastTests/E138PlanSurfaceTests.swift",
        "dispatch_weights": "48/48/16/16/64/64/1 = 257 per verify forward",
        "rule_115_in_situ_transfer": analysis.IN_SITU_TRANSFER,
        "rule_115_host_transfer": analysis.HOST_TRANSFER,
        "rule_115_ranked_round_us": analysis.RANKED_ROUND_US,
    }

    if args.dry:
        print(json.dumps({"config": config, "summary": summary}, indent=2,
                         sort_keys=True, default=str))
        return 0

    run = wandb.init(
        entity=ENTITY, project=PROJECT, group=GROUP,
        name=spec["run_name"], job_type=spec["job_type"], config=config,
    )
    run.log(tables)
    run.summary.update(summary)
    print("%s  %s" % (run.id, run.url))
    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
