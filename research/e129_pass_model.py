#!/usr/bin/env python3
"""E129 rung 2e -- fit the per-pass and per-row cost of the wide QMV.

    usage: research/e129_pass_model.py OUT_DIR [--histogram 4:16,5:20,...]

E63 modelled one routed cell as a fixed cost paid once per pass over the
weight matrix plus an arithmetic cost paid once per input row:

    t(M) = d + a * G(M) + b * sum(NA_g)

`G` is the number of threadgroups in x that do work, `ceil(M / IPG)`, and each
one re-reads the whole weight matrix. `NA_g` is the rows that group carries.
Every shipped case has `sum(NA_g) == M`, so the shipped plan gives seven
`(G, M)` points per shape and the fit is well determined:

    M        3  4  5  6  7  8  9
    G        1  1  1  2  2  2  3

`d` is the dispatch and launch cost that neither scales with passes nor with
rows. Fitting it separately keeps it out of `a`, which would otherwise
overstate what one-pass recovers.

The one-pass prediction is `d + a + b * M`. The residual of a measured
one-pass cell against that prediction is the spill penalty of the wide body at
that width, in microseconds, measured rather than assumed.

`harness=local`, ungated. Nothing here is a ranked or official score.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics

FALLBACK_PLAN = {3: 3, 4: 4, 5: 5, 6: 3, 7: 4, 8: 4, 9: 3}

CALLS_PER_ROUND = {
    "mlp.gate_up": 64,
    "mlp.down": 64,
    "gdn.in_proj": 48,
    "gdn.out_proj": 48,
    "fa.qkv": 16,
    "fa.o_proj": 16,
    "lm_head": 1,
}
SHAPE_ALIAS = {"gdn.out_proj": "fa.o_proj"}
DEFAULT_HISTOGRAM = {4: 16, 5: 20, 6: 20, 7: 12, 8: 240}


def groups(m: int, ipg: int) -> int:
    return (m + ipg - 1) // ipg


def parse_plan(tag) -> dict:
    """`3x3x4_4x4x4_...`, or the older `3x3_4x4_...`, to `{M: IPG}`."""
    if not tag:
        return dict(FALLBACK_PLAN)
    plan = {}
    for entry in str(tag).split("_"):
        parts = entry.split("x")
        if len(parts) >= 2:
            plan[int(parts[0])] = int(parts[1])
    return plan or dict(FALLBACK_PLAN)


def solve(rows: list[tuple[list[float], float]]) -> list[float]:
    """Least squares by normal equations; the design is 3x3 at most."""
    n = len(rows[0][0])
    ata = [[sum(r[0][i] * r[0][j] for r in rows) for j in range(n)] for i in range(n)]
    atb = [sum(r[0][i] * r[1] for r in rows) for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(ata[r][col]))
        ata[col], ata[pivot] = ata[pivot], ata[col]
        atb[col], atb[pivot] = atb[pivot], atb[col]
        head = ata[col][col]
        if abs(head) < 1e-12:
            raise SystemExit("singular design matrix; check the width coverage")
        for row in range(n):
            if row == col:
                continue
            factor = ata[row][col] / head
            for k in range(col, n):
                ata[row][k] -= factor * ata[col][k]
            atb[row] -= factor * atb[col]
    return [atb[i] / ata[i][i] for i in range(n)]


def arm_medians(payload: dict) -> dict[tuple[str, int], dict[str, float]]:
    grouped: dict[tuple[str, int], dict[str, list[float]]] = {}
    for record in payload["cells"]:
        key = (record["shape"], record["width"])
        slot = grouped.setdefault(key, {})
        for arm in record["arms"]:
            mean = 0.5 * (arm["forward_us"] + arm["reverse_us"])
            slot.setdefault(arm["arm"], []).append(mean)
    return {
        key: {name: statistics.median(values) for name, values in slot.items()}
        for key, slot in grouped.items()
    }


def fit_shape(shape: str, medians: dict, widths: list[int], plan: dict) -> dict | None:
    rows = []
    for width in widths:
        cell = medians.get((shape, width))
        if cell is None or "b_shipped" not in cell or width not in plan:
            continue
        rows.append(([1.0, float(groups(width, plan[width])), float(width)],
                     cell["b_shipped"]))
    if len(rows) < 4:
        return None
    d, a, b = solve(rows)
    residuals = [value - (d + a * design[1] + b * design[2]) for design, value in rows]
    mean = statistics.fmean(value for _, value in rows)
    ss_tot = sum((value - mean) ** 2 for _, value in rows)
    ss_res = sum(r * r for r in residuals)
    return {
        "shape": shape,
        "points": len(rows),
        "dispatch_us": d,
        "per_pass_us": a,
        "per_row_us": b,
        "r_squared": 1 - ss_res / ss_tot if ss_tot else None,
        "max_abs_residual_us": max(abs(r) for r in residuals),
        "rel_residual": max(abs(r) for r in residuals) / mean if mean else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=pathlib.Path)
    parser.add_argument("--histogram", default=None)
    args = parser.parse_args()

    payload = json.loads((args.out_dir / "m5ipg.json").read_text())
    medians = arm_medians(payload)
    widths = sorted({width for _, width in medians})
    shapes = sorted({shape for shape, _ in medians})
    histogram = DEFAULT_HISTOGRAM
    if args.histogram:
        histogram = {int(t.split(":")[0]): int(t.split(":")[1])
                     for t in args.histogram.split(",")}

    plan = parse_plan(payload.get("cases_shipped"))
    fits = [f for f in (fit_shape(s, medians, widths, plan) for s in shapes) if f]
    print("fit t = d + a*G + b*M on the shipped arm, per shape")
    print("%-14s %6s %10s %10s %10s %8s %9s"
          % ("shape", "pts", "d us", "a us/pass", "b us/row", "R2", "maxresid"))
    for fit in fits:
        print("%-14s %6d %10.3f %10.3f %10.3f %8.5f %9.3f"
              % (fit["shape"], fit["points"], fit["dispatch_us"], fit["per_pass_us"],
                 fit["per_row_us"], fit["r_squared"], fit["max_abs_residual_us"]))

    print("\none-pass prediction d + a + b*M against the measured compare arm")
    print("%-14s %3s %10s %10s %10s %9s %9s"
          % ("shape", "M", "shipped", "predicted", "measured", "resid us", "resid %"))
    spill = []
    for fit in fits:
        for width in widths:
            cell = medians.get((fit["shape"], width))
            if cell is None or "a_compare" not in cell:
                continue
            if groups(width, plan[width]) == 1:
                continue
            predicted = fit["dispatch_us"] + fit["per_pass_us"] + fit["per_row_us"] * width
            measured = cell["a_compare"]
            residual = measured - predicted
            spill.append({
                "shape": fit["shape"], "width": width, "predicted_us": predicted,
                "measured_us": measured, "residual_us": residual,
                "residual_fraction": residual / predicted,
                "shipped_us": cell["b_shipped"],
                "measured_gain": (cell["b_shipped"] - measured) / cell["b_shipped"],
                "ideal_gain": (cell["b_shipped"] - predicted) / cell["b_shipped"],
            })
            print("%-14s %3d %10.3f %10.3f %10.3f %9.3f %9.3f"
                  % (fit["shape"], width, cell["b_shipped"], predicted, measured,
                     residual, 100 * residual / predicted))

    print("\nspill residual by width, averaged over shapes "
          "(positive = the wide body costs more than the spill-free line)")
    for width in sorted({row["width"] for row in spill}):
        rows = [r for r in spill if r["width"] == width]
        print("  M=%d  n=%d  residual %+7.3f %%   measured gain %+7.3f %%   "
              "spill-free gain %+7.3f %%"
              % (width, len(rows),
                 100 * statistics.fmean(r["residual_fraction"] for r in rows),
                 100 * statistics.fmean(r["measured_gain"] for r in rows),
                 100 * statistics.fmean(r["ideal_gain"] for r in rows)))

    by_key = {(row["shape"], row["width"]): row for row in spill}
    base = saved = ideal = 0.0
    for width, rounds in sorted(histogram.items()):
        for shape, calls in CALLS_PER_ROUND.items():
            key = (SHAPE_ALIAS.get(shape, shape), width)
            cell = medians.get((key[0], width))
            if cell is None:
                continue
            base += rounds * calls * cell["b_shipped"]
            row = by_key.get(key)
            if row is None:
                continue
            saved += rounds * calls * (row["shipped_us"] - row["measured_us"])
            ideal += rounds * calls * (row["shipped_us"] - row["predicted_us"])
    print("\nrouted-round model over %d rounds, wide QMV %.0f us shipped"
          % (sum(histogram.values()), base))
    print("  measured one-pass gain    %+7.3f %%" % (100 * saved / base))
    print("  spill-free one-pass gain  %+7.3f %%" % (100 * ideal / base))

    out = args.out_dir / "pass-model.json"
    out.write_text(json.dumps({
        "harness": "local",
        "gate_qualified_for_timing": False,
        "cool_gate_passed_real_gate": False,
        "official_or_ranked_score": False,
        "cases_shipped": payload.get("cases_shipped"),
        "cases_compare": payload.get("cases_compare"),
        "fits": fits,
        "one_pass_cells": spill,
        "histogram": {str(k): v for k, v in sorted(histogram.items())},
        "wide_qmv_us_shipped": base,
        "measured_gain_fraction": saved / base if base else None,
        "spill_free_gain_fraction": ideal / base if base else None,
    }, indent=2, sort_keys=True) + "\n")
    print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
