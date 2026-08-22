#!/usr/bin/env python3
"""E138: the declared two-anchor launch-grid control.

    usage: research/e138_grid_control.py WIDE.json TIGHT.json

Scope, as approved by the advisor in feedback F1 section 6: exactly the two
shipped anchors `(5,5,4)` and `(6,6,4)`, measured on both launch grids, handed
to the advisor rather than turned into a law. This is NOT a column-count
ladder; thorfinn owns the launch column law under E135.

Why it exists: the E137 denominator this experiment is scored against was
measured under the wide grid, which is still the compiled default at
`Qwen35.swift:1952`, while the E138 sweep runs under the tight grid now in
flight. Without these two cells a change in the step cannot be attributed to
the plan rather than to the grid.

Both sessions anchor on `6:stock`. The stock MLX quantized matmul does not read
`MLX_E120_QMV_GRID`, so it is the only reference cell whose cost is identical
in both sessions, which makes the two sets of drift-corrected microseconds
comparable in absolute terms.
"""

from __future__ import annotations

import json
import pathlib
import statistics
import sys

ANCHORS = ("5:5:4", "6:6:4")


def corrected(path: pathlib.Path, expect_grid: str) -> tuple[dict, dict]:
    payload = json.loads(path.read_text())
    if payload["grid"] != expect_grid:
        raise SystemExit(f"{path} is grid={payload['grid']}, expected {expect_grid}")
    if payload["reference_cell"] != "6:stock":
        raise SystemExit(
            f"{path} anchors on {payload['reference_cell']}; the grid control "
            "requires the grid-independent 6:stock anchor"
        )
    values, dispatch = {}, {}
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
    return values, dispatch


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    wide, dispatch = corrected(pathlib.Path(sys.argv[1]), "wide")
    tight, _ = corrected(pathlib.Path(sys.argv[2]), "tight")

    names = sorted(dispatch)
    header = "%-34s %4s" % ("shape", "disp")
    for cell in ANCHORS:
        header += "%10s W%10s T%9s" % (cell, cell, "delta")
    print(header)

    totals = {(cell, grid): 0.0 for cell in ANCHORS for grid in ("wide", "tight")}
    for name in names:
        line = "%-34s %4d" % (name, dispatch[name])
        for cell in ANCHORS:
            w, t = wide[(name, cell)], tight[(name, cell)]
            totals[(cell, "wide")] += w * dispatch[name]
            totals[(cell, "tight")] += t * dispatch[name]
            line += "%11.1f%11.1f%8.1f %%" % (w, t, 100.0 * (t - w) / w)
        print(line)

    print()
    for cell in ANCHORS:
        w, t = totals[(cell, "wide")], totals[(cell, "tight")]
        print(
            "weighted %s   wide %10.0f   tight %10.0f   tight is %+6.2f %%"
            % (cell, w, t, 100.0 * (t - w) / w)
        )

    step_wide = totals[("6:6:4", "wide")] - totals[("5:5:4", "wide")]
    step_tight = totals[("6:6:4", "tight")] - totals[("5:5:4", "tight")]
    print()
    print("isolated dispatch-weighted M=5 -> M=6 step")
    print("    wide  %10.1f us" % step_wide)
    print("    tight %10.1f us   %+.2f %% against wide"
          % (step_tight, 100.0 * (step_tight - step_wide) / step_wide))
    print()
    print("harness=local. Both sessions anchor on the grid-independent 6:stock")
    print("cell, so these microseconds are comparable across the two grids.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
