#!/usr/bin/env python3
"""E138: the launch-grid x plan factorial.

    usage: research/e138_grid_control.py WIDE.json TIGHT.json [--baseline=PLAN]

Advisor feedback F2 section 4 asks for the sweep to run as a two-by-plan
factorial: every plan cell under both `wide` and `tight`, the plan ranking
under each grid, and the interaction term stated explicitly. This script is
that report. It supersedes the narrower two-anchor control approved in F1
section 6, which measured only `(5,5,4)` and `(6,6,4)`.

It remains a control handed to the advisor, not a launch-column law. thorfinn
owns the launch column law under E135.

Both sessions must anchor on `6:stock`. The stock MLX quantized matmul does not
read `MLX_E120_QMV_GRID`, so it is the only reference cell whose cost is
identical in both sessions, which makes the two sets of drift-corrected
microseconds comparable in absolute terms.

Interaction, per shape and per plan, against the one-pass baseline plan:

    advantage(plan, grid) = us(baseline, grid) - us(plan, grid)
    interaction(plan)     = advantage(plan, tight) - advantage(plan, wide)

A positive interaction means the plan buys more under the tight grid than it
buys under the wide grid. A null interaction means the plan axis and the grid
axis are separable, so the plan ranking transfers between grids.
"""

from __future__ import annotations

import json
import pathlib
import statistics
import sys


def corrected(path: pathlib.Path, expect_grid: str) -> dict:
    payload = json.loads(path.read_text())
    if payload["grid"] != expect_grid:
        raise SystemExit(f"{path} is grid={payload['grid']}, expected {expect_grid}")
    if payload["reference_cell"] != "6:stock":
        raise SystemExit(
            f"{path} anchors on {payload['reference_cell']}; the factorial "
            "requires the grid-independent 6:stock anchor"
        )
    values, dispatch, launch = {}, {}, {}
    for shape in payload["shapes"]:
        if not shape["exactness_positive_control_rejects"]:
            raise SystemExit(f"{path}: {shape['name']} positive control passed")
        dispatch[shape["name"]] = shape["calls_per_verify"]
        anchor = statistics.median(
            [v for row in shape["rows"] for v in row["reference_samples"]]
        )
        for row in shape["rows"]:
            if not row["matches_incumbent_bitwise"]:
                raise SystemExit(f"{path}: {shape['name']} {row['cell']} not exact")
            ratio = statistics.median(row["samples"]) / statistics.median(
                row["reference_samples"]
            )
            values[(shape["name"], row["cell"])] = ratio * anchor * 1e6
            launch[(shape["name"], row["cell"])] = (
                row["launched_columns"] * row["threadgroups_per_column"]
            )
    return {
        "us": values,
        "dispatch": dispatch,
        "launch": launch,
        "n": {s["name"]: s["n"] for s in payload["shapes"]},
        "k": {s["name"]: s["k"] for s in payload["shapes"]},
    }


def factorial(wide_path: pathlib.Path, tight_path: pathlib.Path,
              baseline: str | None = None) -> dict:
    wide = corrected(wide_path, "wide")
    tight = corrected(tight_path, "tight")

    cells = sorted(
        {c for _, c in wide["us"]} & {c for _, c in tight["us"]},
        key=lambda c: (c.split(":")[1] == "stock", c),
    )
    plans = [c for c in cells if not c.endswith(":stock")]
    if not plans:
        raise SystemExit("no plan cell is present in both artifacts")
    baseline = baseline or max(plans, key=lambda c: int(c.split(":")[1]))
    if baseline not in plans:
        raise SystemExit(f"baseline {baseline} is not in both artifacts")
    dispatch = wide["dispatch"]
    names = sorted(dispatch, key=lambda s: (wide["n"][s], s))

    shapes = []
    for name in names:
        bw, bt = wide["us"][(name, baseline)], tight["us"][(name, baseline)]
        rows = []
        for cell in cells:
            w, t = wide["us"][(name, cell)], tight["us"][(name, cell)]
            row = {
                "shape": name,
                "n": wide["n"][name],
                "k": wide["k"][name],
                "plan": cell,
                "wide_us": w,
                "tight_us": t,
                "tight_minus_wide_pct": 100.0 * (t - w) / w,
                "launched_threadgroups_wide": wide["launch"][(name, cell)],
                "launched_threadgroups_tight": tight["launch"][(name, cell)],
            }
            if not cell.endswith(":stock"):
                row["advantage_wide_pct"] = 100.0 * (bw - w) / bw
                row["advantage_tight_pct"] = 100.0 * (bt - t) / bt
                row["interaction_pp"] = (
                    row["advantage_tight_pct"] - row["advantage_wide_pct"]
                )
                row["interaction_us"] = (bt - t) - (bw - w)
            rows.append(row)
        rank_w = sorted(plans, key=lambda c: wide["us"][(name, c)])
        rank_t = sorted(plans, key=lambda c: tight["us"][(name, c)])
        shapes.append({
            "name": name,
            "n": wide["n"][name],
            "dispatch": dispatch[name],
            "rows": rows,
            "rank_wide": rank_w,
            "rank_tight": rank_t,
            "order_same": rank_w == rank_t,
            "best_wide": rank_w[0],
            "best_tight": rank_t[0],
        })

    tot = {
        (c, g): sum(src["us"][(n, c)] * dispatch[n] for n in names)
        for c in cells
        for g, src in (("wide", wide), ("tight", tight))
    }
    totals = []
    for cell in cells:
        w, t = tot[(cell, "wide")], tot[(cell, "tight")]
        entry = {
            "plan": cell,
            "wide_us": w,
            "tight_us": t,
            "tight_minus_wide_pct": 100.0 * (t - w) / w,
        }
        if not cell.endswith(":stock"):
            entry["advantage_wide_us"] = tot[(baseline, "wide")] - w
            entry["advantage_tight_us"] = tot[(baseline, "tight")] - t
            entry["interaction_us"] = (
                entry["advantage_tight_us"] - entry["advantage_wide_us"]
            )
        totals.append(entry)

    out = {
        "harness": "local",
        "baseline_plan": baseline,
        "plans": plans,
        "cells": cells,
        "shapes": shapes,
        "totals": totals,
        "best_plan_wide": min(plans, key=lambda c: tot[(c, "wide")]),
        "best_plan_tight": min(plans, key=lambda c: tot[(c, "tight")]),
        "shapes_with_order_change": [s["name"] for s in shapes
                                     if not s["order_same"]],
    }
    out["grid_changes_the_global_winner"] = (
        out["best_plan_wide"] != out["best_plan_tight"]
    )
    out["plan_axis_and_grid_axis_are_separable"] = not out[
        "shapes_with_order_change"
    ]
    out["max_abs_interaction_pp"] = max(
        abs(r["interaction_pp"])
        for s in shapes
        for r in s["rows"]
        if "interaction_pp" in r
    )
    if "5:5:4" in plans and "6:6:4" in plans:
        sw = tot[("6:6:4", "wide")] - tot[("5:5:4", "wide")]
        st = tot[("6:6:4", "tight")] - tot[("5:5:4", "tight")]
        out["step_wide_us"] = sw
        out["step_tight_us"] = st
        out["step_tight_vs_wide_pct"] = 100.0 * (st - sw) / sw
    return out


def main() -> int:
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = dict(
        a.split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a
    )
    if len(argv) != 2:
        raise SystemExit(__doc__)
    r = factorial(pathlib.Path(argv[0]), pathlib.Path(argv[1]),
                  flags.get("--baseline"))

    print("E138 launch-grid x plan factorial   harness=local")
    print("baseline plan for every advantage and interaction: %s"
          % r["baseline_plan"])
    print()
    print("launched threadgroups per call (Qwen35.swift:1957-1961, columns x per column)")
    print("%-34s %7s %6s %10s %10s" % ("shape", "N", "plan", "wide", "tight"))
    for shape in r["shapes"]:
        for row in shape["rows"]:
            if row["plan"].endswith(":stock"):
                continue
            print("%-34s %7d %6s %10d %10d"
                  % (shape["name"], shape["n"], row["plan"],
                     row["launched_threadgroups_wide"],
                     row["launched_threadgroups_tight"]))
    print()

    print("per shape, drift-corrected microseconds per call")
    for shape in r["shapes"]:
        print("%s   N=%d   dispatched %dx per verify"
              % (shape["name"], shape["n"], shape["dispatch"]))
        print("    %6s %10s %10s %8s %10s %10s %9s"
              % ("plan", "wide", "tight", "T-W %", "advW %", "advT %",
                 "inter pp"))
        for row in shape["rows"]:
            if "interaction_pp" not in row:
                print("    %6s %10.1f %10.1f %7.1f %%%10s %10s %9s"
                      % (row["plan"], row["wide_us"], row["tight_us"],
                         row["tight_minus_wide_pct"], "-", "-", "-"))
                continue
            print("    %6s %10.1f %10.1f %7.1f %% %9.2f %9.2f %+8.2f"
                  % (row["plan"], row["wide_us"], row["tight_us"],
                     row["tight_minus_wide_pct"], row["advantage_wide_pct"],
                     row["advantage_tight_pct"], row["interaction_pp"]))
        print("    rank wide  %s" % " < ".join(shape["rank_wide"]))
        print("    rank tight %s%s"
              % (" < ".join(shape["rank_tight"]),
                 "" if shape["order_same"] else "   <- ORDER CHANGES"))
        print()

    print("dispatch-weighted totals over the whole 64-layer step")
    print("%6s %12s %12s %9s %12s %12s %11s"
          % ("plan", "wide us", "tight us", "T-W %", "advW us", "advT us",
             "inter us"))
    for entry in r["totals"]:
        if "interaction_us" not in entry:
            print("%6s %12.0f %12.0f %8.2f %%%12s %12s %11s"
                  % (entry["plan"], entry["wide_us"], entry["tight_us"],
                     entry["tight_minus_wide_pct"], "-", "-", "-"))
            continue
        print("%6s %12.0f %12.0f %8.2f %% %11.0f %12.0f %+11.0f"
              % (entry["plan"], entry["wide_us"], entry["tight_us"],
                 entry["tight_minus_wide_pct"], entry["advantage_wide_us"],
                 entry["advantage_tight_us"], entry["interaction_us"]))

    print()
    print("global best plan   wide %s   tight %s%s"
          % (r["best_plan_wide"], r["best_plan_tight"],
             "   <- GRID CHANGES THE WINNER"
             if r["grid_changes_the_global_winner"] else ""))
    print("largest absolute per-shape interaction: %.2f pp"
          % r["max_abs_interaction_pp"])
    if r["shapes_with_order_change"]:
        print("per-shape plan order changes under: %s"
              % ", ".join(r["shapes_with_order_change"]))
    else:
        print("per-shape plan order is identical under both grids for every shape:")
        print("    the plan axis and the grid axis are separable on this evidence")

    if "step_wide_us" in r:
        print()
        print("isolated dispatch-weighted M=5 -> M=6 step")
        print("    wide  %10.1f us" % r["step_wide_us"])
        print("    tight %10.1f us   %+.2f %% against wide"
              % (r["step_tight_us"], r["step_tight_vs_wide_pct"]))

    print()
    print("harness=local. Both sessions anchor on the grid-independent 6:stock")
    print("cell, so these microseconds are comparable across the two grids.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
