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
    }


def main() -> int:
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = dict(
        a.split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a
    )
    if len(argv) != 2:
        raise SystemExit(__doc__)
    wide = corrected(pathlib.Path(argv[0]), "wide")
    tight = corrected(pathlib.Path(argv[1]), "tight")

    cells = sorted(
        {c for _, c in wide["us"]} & {c for _, c in tight["us"]},
        key=lambda c: (c.split(":")[1] == "stock", c),
    )
    plans = [c for c in cells if not c.endswith(":stock")]
    if not plans:
        raise SystemExit("no plan cell is present in both artifacts")
    baseline = flags.get("--baseline", max(plans, key=lambda c: int(c.split(":")[1])))
    if baseline not in plans:
        raise SystemExit(f"baseline {baseline} is not in both artifacts")
    dispatch = wide["dispatch"]
    names = sorted(dispatch, key=lambda s: (wide["n"][s], s))

    print("E138 launch-grid x plan factorial   harness=local")
    print(f"baseline plan for every advantage and interaction: {baseline}")
    print()
    print("launched threadgroups per call (Qwen35.swift:1957-1961, columns x per column)")
    print("%-34s %7s %6s %10s %10s" % ("shape", "N", "plan", "wide", "tight"))
    for name in names:
        for cell in plans:
            print(
                "%-34s %7d %6s %10d %10d"
                % (
                    name,
                    wide["n"][name],
                    cell,
                    wide["launch"][(name, cell)],
                    tight["launch"][(name, cell)],
                )
            )
    print()

    print("per shape, drift-corrected microseconds per call")
    reversals = []
    for name in names:
        print(
            "%s   N=%d   dispatched %dx per verify"
            % (name, wide["n"][name], dispatch[name])
        )
        print(
            "    %6s %10s %10s %8s %10s %10s %9s"
            % ("plan", "wide", "tight", "T-W %", "advW %", "advT %", "inter pp")
        )
        for cell in cells:
            w, t = wide["us"][(name, cell)], tight["us"][(name, cell)]
            if cell.endswith(":stock"):
                print(
                    "    %6s %10.1f %10.1f %7.1f %%%10s %10s %9s"
                    % (cell, w, t, 100.0 * (t - w) / w, "-", "-", "-")
                )
                continue
            bw, bt = wide["us"][(name, baseline)], tight["us"][(name, baseline)]
            aw, at = 100.0 * (bw - w) / bw, 100.0 * (bt - t) / bt
            print(
                "    %6s %10.1f %10.1f %7.1f %% %9.2f %9.2f %+8.2f"
                % (cell, w, t, 100.0 * (t - w) / w, aw, at, at - aw)
            )
        rank_w = sorted(plans, key=lambda c: wide["us"][(name, c)])
        rank_t = sorted(plans, key=lambda c: tight["us"][(name, c)])
        same = rank_w == rank_t
        reversals.append((name, same))
        print("    rank wide  %s" % " < ".join(rank_w))
        print(
            "    rank tight %s%s"
            % (" < ".join(rank_t), "" if same else "   <- ORDER CHANGES")
        )
        print()

    print("dispatch-weighted totals over the whole 64-layer step")
    print(
        "%6s %12s %12s %9s %12s %12s %11s"
        % ("plan", "wide us", "tight us", "T-W %", "advW us", "advT us", "inter us")
    )
    tot = {
        (c, g): sum(src["us"][(n, c)] * dispatch[n] for n in names)
        for c in cells
        for g, src in (("wide", wide), ("tight", tight))
    }
    for cell in cells:
        w, t = tot[(cell, "wide")], tot[(cell, "tight")]
        if cell.endswith(":stock"):
            print(
                "%6s %12.0f %12.0f %8.2f %%%12s %12s %11s"
                % (cell, w, t, 100.0 * (t - w) / w, "-", "-", "-")
            )
            continue
        aw = tot[(baseline, "wide")] - w
        at = tot[(baseline, "tight")] - t
        print(
            "%6s %12.0f %12.0f %8.2f %% %11.0f %12.0f %+11.0f"
            % (cell, w, t, 100.0 * (t - w) / w, aw, at, at - aw)
        )

    print()
    best_w = min(plans, key=lambda c: tot[(c, "wide")])
    best_t = min(plans, key=lambda c: tot[(c, "tight")])
    print(
        "global best plan   wide %s   tight %s%s"
        % (best_w, best_t, "" if best_w == best_t else "   <- GRID CHANGES THE WINNER")
    )
    changed = [n for n, same in reversals if not same]
    if changed:
        print("per-shape plan order changes under: %s" % ", ".join(changed))
    else:
        print("per-shape plan order is identical under both grids for every shape:")
        print("    the plan axis and the grid axis are separable on this evidence")

    if "5:5:4" in plans and "6:6:4" in plans:
        sw = tot[("6:6:4", "wide")] - tot[("5:5:4", "wide")]
        st = tot[("6:6:4", "tight")] - tot[("5:5:4", "tight")]
        print()
        print("isolated dispatch-weighted M=5 -> M=6 step")
        print("    wide  %10.1f us" % sw)
        print(
            "    tight %10.1f us   %+.2f %% against wide"
            % (st, 100.0 * (st - sw) / sw)
        )

    print()
    print("harness=local. Both sessions anchor on the grid-independent 6:stock")
    print("cell, so these microseconds are comparable across the two grids.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
