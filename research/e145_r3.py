#!/usr/bin/env python3
"""E145 R3: re-price the E140 and E134 decision cells on the LIVE curve.

`harness=local instrument`. Zero GPU.

WHY THIS RUNS. Every cell in `e140_cells.py` reads one cost curve twice: once
as the COST MODEL that turns a simulated round sequence into microseconds, and
once as the PRICE TABLE the depth walk consults. Until R2 that curve was
rebuilt, not measured. R2 pinned the drafted depth and read the round cost off
a live decode at every width from 2 to 8, and R2b added the width-1 anchor the
price table divides by. The two curves disagree by far more than the 5 percent
the assignment set as the skip threshold, so the cells have to be re-priced.

THE TWO ROLES ARE SEPARATED. A cell can be re-priced in three ways and they
answer different questions:

  cost=replayed price=replayed   the published E140 result, reproduced here as
                                 the control.
  cost=measured price=replayed   what the published policies are really worth,
                                 if the live curve is the better cost model.
                                 The POLICY is untouched, so this isolates the
                                 cost model.
  cost=measured price=measured   the policy a campaign that trusted the live
                                 curve would actually install.

Reporting only the third would confound a changed policy with a changed
scorer. Reporting all three separates them.

WHAT IS AND IS NOT CLAIMED. The R2 curve is LOCAL, measured on this M4 Pro. The
E140 replayer scores in ranked microseconds. A level factor cannot be avoided
here, because the price table normalises by width 1 and the cost model enters
`us_per_token` as a ratio against an unpriced arm run on the SAME curve. Both
uses are scale free: multiplying the whole curve by a constant leaves
`cumulative[d] = round_us(d+1) / round_us(1)` and the ratio
`run.us_per_token / base.us_per_token` unchanged. So the level factor cancels
and only the SHAPE transfers. That is stated as an assumption and gated:
`--check-scale` multiplies the measured curve by an arbitrary constant and
requires every reported number to be unchanged.

The shape itself still has to transfer from M4 Pro to M5. It is not proven
that it does. This module reports what the live shape implies; it does not
claim the ranked machine has the same shape.

Usage:
  python3 research/e145_r3.py --json research/e145-artifacts/r3.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import e128_price  # noqa: E402
from e128_price import (  # noqa: E402
    MAX_DEPTH, RANKED_PROMPTS, load_board_receipt,
)
from e134_rung2 import build_legs, median_pct, simulate  # noqa: E402
from e140_cells import (  # noqa: E402
    CELLS, greedy_walker, install, lookahead_walker, price_for,
    transfer_cache,
)
from e140_lookahead import load_curves  # noqa: E402
from e145_curve import BASIS, LEGS_JSON, measured_curve  # noqa: E402

# The replayed ranked curve the campaign has priced everything through, in the
# form `e134_item2_refit.measured_curve` installs. R3 does not refit it; it is
# loaded from the same artifact E140 read, so the control is bit identical.
REPLAYED_LABEL = "replayed"
MEASURED_LABEL = "measured"


def measured_points(legs: list[dict]) -> dict[int, float]:
    """Width to live round cost, microseconds, on the R2 session level.

    R2 swept pins 1 to 7 inside one palindrome, so widths 2 to 8 are directly
    comparable. R2b measured width 1 in a SECOND session and repeated width 5
    as a tie point, so the width-1 anchor is rescaled by the ratio of the two
    sessions' width-5 readings before it is used. Without that rescale the
    anchor would carry the second session's own thermal level into the
    denominator of every entry in the price table.
    """
    points = {w: entry[BASIS] for w, entry in measured_curve(legs).items()}

    r2b: dict[int, list[float]] = {}
    for leg in legs:
        if not leg["slot"].startswith("r2b-"):
            continue
        if leg["pin"] in (None, "", "none"):
            continue
        width = int(leg["pin"]) + 1
        r2b.setdefault(width, []).append(leg["round_us_from_blocks"])

    anchor = {"available": False}
    if 1 in r2b:
        raw = statistics.fmean(r2b[1])
        tie_in = [w for w in r2b if w != 1 and w in points]
        if tie_in:
            tie = tie_in[0]
            scale = points[tie] / statistics.fmean(r2b[tie])
        else:
            tie, scale = None, 1.0
        anchor = {"available": True, "raw_us": raw, "tie_width": tie,
                  "session_scale": scale, "rescaled_us": raw * scale,
                  "r2b_legs": {w: len(v) for w, v in r2b.items()}}
    return points, anchor


def width1_candidates(points: dict[int, float], anchor: dict,
                      serial: float | None) -> dict:
    """Every width-1 value the evidence offers, with what each one assumes.

    The price table divides by width 1, so this single number scales the whole
    table. It is reported three ways rather than chosen silently.
    """
    lo = min(points)
    out = {}
    if anchor.get("available"):
        out["measured"] = anchor["rescaled_us"]
    if lo + 1 in points:
        out["extrapolated"] = points[lo] - (points[lo + 1] - points[lo])
    if serial is not None:
        out["serial"] = serial
    return out


def serial_round_us(legs: list[dict]) -> float | None:
    """The `--mtp-depth 0` control's round cost, for the width-1 comparison."""
    rows = [leg["round_us_from_blocks"] for leg in legs
            if leg["slot"].startswith("r1-") and leg["arm"] == "serial"
            and leg["fixture"] == "beagle_a"]
    return statistics.fmean(rows) if rows else None


def installable(points: dict[int, float], label: str) -> dict:
    """A `e128_price.CURVE` that returns the measured value at every width.

    The two-segment `lo`/`hi` fit is kept at zero and the whole curve is
    carried in `per_width`, so no fitted line can smooth away the step the
    experiment exists to measure. Widths outside the measured range are
    extrapolated from the last measured step and flagged, because the walk
    reads `cumulative` up to `MAX_DEPTH + 1` whatever was measured.
    """
    lo = min(points)
    hi = max(points)
    per_width = dict(points)
    extrapolated = []
    for width in range(1, MAX_DEPTH + 2):
        if width in per_width:
            continue
        if width < lo:
            step = points[lo + 1] - points[lo]
            per_width[width] = points[lo] - step * (lo - width)
        else:
            step = points[hi] - points[hi - 1]
            per_width[width] = points[hi] + step * (width - hi)
        extrapolated.append(width)
    return {"name": "e145_r2_live_%s" % label,
            "breakpoint": 1, "lo": (0.0, 0.0), "hi": (0.0, 0.0),
            "per_width": per_width,
            "provenance": {"measured_widths": sorted(points),
                           "extrapolated_widths": extrapolated}}


def shape(curve: dict) -> dict:
    saved = e128_price.CURVE
    e128_price.CURVE = curve
    widths = list(range(1, MAX_DEPTH + 2))
    values = [e128_price.ranked_round_us(w) for w in widths]
    marginal, cumulative = e128_price.ranked_price_table()
    e128_price.CURVE = saved
    steps = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    return {"widths": widths, "round_us": values, "steps": steps,
            "marginal": list(marginal), "cumulative": list(cumulative),
            "argmax_step_index": max(range(len(steps)),
                                     key=lambda i: steps[i])}


def tier_implied(curve: dict, cliff: int = 4) -> dict:
    """What one-boundary tier factor the curve itself asks for.

    `boundary_price` raises exactly one marginal entry and lowers the rest to
    hold the total. The tier the curve implies is therefore the cliff step
    divided by the mean of the other steps, not divided by the width-1 step.
    Both are reported because the campaign has quoted the second form.
    """
    sh = shape(curve)
    steps = sh["steps"]
    others = [s for i, s in enumerate(steps) if i != cliff]
    mean_other = statistics.fmean(others)
    return {"cliff_step_us": steps[cliff],
            "mean_other_step_us": mean_other,
            "tier_against_mean_other": steps[cliff] / mean_other,
            "tier_against_first_step": steps[cliff] / steps[0],
            "argmax_step_index": sh["argmax_step_index"],
            "argmax_is_the_shipped_cliff": sh["argmax_step_index"] == cliff}


def boundary_table(curve: dict) -> list[dict]:
    """Every boundary's step, as a share of the curve's own mean step.

    The campaign prices ONE boundary because the replayed curve has one big
    step. Whether that is still the right shape is a property of the curve, so
    it is tabulated rather than assumed.
    """
    sh = shape(curve)
    steps = sh["steps"]
    mean_step = statistics.fmean(steps)
    return [{"boundary": i, "from_width": i + 1, "to_width": i + 2,
             "step_us": s, "share_of_mean_step": s / mean_step}
            for i, s in enumerate(steps)]


def compare_shapes(a: dict, b: dict) -> dict:
    """Per-width disagreement, as the assignment's skip rule states it."""
    sa, sb = shape(a), shape(b)
    # Both curves are scale free in every use, so the comparison is made after
    # normalising each to its own width-1 cost. Comparing raw microseconds
    # would report the local-to-ranked level factor as a shape disagreement.
    na = [v / sa["round_us"][0] for v in sa["round_us"]]
    nb = [v / sb["round_us"][0] for v in sb["round_us"]]
    rows = []
    for i, width in enumerate(sa["widths"]):
        rows.append({"width": width, "a_norm": na[i], "b_norm": nb[i],
                     "pct": 100.0 * (nb[i] / na[i] - 1.0)})
    worst = max(rows, key=lambda r: abs(r["pct"]))
    return {"per_width": rows, "worst_width": worst["width"],
            "worst_pct": worst["pct"],
            "agrees_within_5_pct": abs(worst["pct"]) <= 5.0}


def run_grid(cache, seeds, receipt, windows, cost_curves, price_curves,
             cells) -> dict:
    """`median_pct` for every (cost model, price curve, cell)."""
    out = {}
    for cost_label, cost_curve in cost_curves.items():
        for price_label, price_curve in price_curves.items():
            for cell in cells:
                values, per_prompt, depths = [], {}, []
                for seed in seeds:
                    ratios = {}
                    for prompt in RANKED_PROMPTS:
                        install(cost_curve)
                        row = run_cell_split(cache, seed, prompt, cell,
                                             cost_curve, price_curve, windows)
                        ratios[prompt] = row
                        per_prompt.setdefault(prompt, []).append(row["ratio"])
                        depths.append(RANKED_PROMPTS[prompt]["weight"]
                                      * row["mean_depth"])
                    values.append(median_pct(receipt, ratios))
                key = "%s|%s|%s" % (cost_label, price_label, cell)
                out[key] = {
                    "cost_curve": cost_label, "price_curve": price_label,
                    "cell": cell,
                    "median_pct_mean": statistics.fmean(values),
                    "median_pct_sd": (statistics.stdev(values)
                                      if len(values) > 1 else 0.0),
                    "median_pct_values": values,
                    "f83_weighted_mean_depth": sum(depths) / len(seeds),
                    "per_prompt_ratio": {p: statistics.fmean(v)
                                         for p, v in per_prompt.items()},
                }
    return out


def run_cell_split(cache, seed, prompt, cell, cost_curve, price_curve,
                   windows):
    """`run_cell` with the cost model and the price table decoupled.

    `e140_cells.run_cell` installs ONE curve and then derives the price from
    that same curve, which is right for E140's question and wrong for this
    one. Here the installed curve is always the cost model, and the price is
    built from whichever curve the grid cell names.
    """
    entry = cache[(seed, prompt)]
    install(cost_curve)
    base = simulate(None, entry["factory"](entry["p_target"]), windows)
    spec = CELLS[cell]
    price = price_for(spec["price"], price_curve)
    walker = (lookahead_walker() if spec["walk"] == "argmax"
              else greedy_walker())
    run = simulate(None, entry["factory"](entry["p_target"]), windows,
                   price=price, walker=walker)
    return {"ratio": run["us_per_token"] / base["us_per_token"],
            "mean_depth": run["mean_depth"],
            "accept_rate": run["accept_rate"],
            "us_per_token": run["us_per_token"],
            "base_us_per_token": base["us_per_token"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=pathlib.Path,
                    default=HERE.parent / ".mlxfast-private/e128/runs-forced")
    ap.add_argument("--accept", type=pathlib.Path,
                    default=HERE / "e128-artifacts/rung1-forced.json")
    ap.add_argument("--board", type=pathlib.Path,
                    default=pathlib.Path("/tmp/yukon-board/full.json"))
    ap.add_argument("--receipt", default="d3c491b5")
    ap.add_argument("--windows", type=int, default=200)
    ap.add_argument("--fit-windows", type=int, default=60)
    ap.add_argument("--seed", type=int, default=128)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--anchor", default="measured",
                    choices=("measured", "extrapolated", "serial"))
    ap.add_argument("--check-scale", action="store_true")
    ap.add_argument("--json", type=pathlib.Path,
                    default=HERE / "e145-artifacts/r3.json")
    args = ap.parse_args()

    print("harness=local instrument  E145 R3 re-pricing  zero GPU")

    legs = json.loads(LEGS_JSON.read_text())["legs"]
    points, anchor = measured_points(legs)
    if not points:
        print("e145_r3: no R2 timed legs")
        return 1
    serial = serial_round_us(legs)
    candidates = width1_candidates(points, anchor, serial)
    if args.anchor not in candidates:
        print("e145_r3: the %s width-1 anchor is not available; have %s"
              % (args.anchor, sorted(candidates)))
        return 1

    print("\n## the width-1 anchor, which divides every price table entry")
    for name, value in sorted(candidates.items()):
        mark = "  <- used" if name == args.anchor else ""
        print("  %-13s %9.1f us%s" % (name, value, mark))
    if anchor.get("available"):
        print("  R2b raw %.1f us, tie width %s, cross-session scale %.5f"
              % (anchor["raw_us"], anchor["tie_width"],
                 anchor["session_scale"]))
    if serial is not None and "measured" in candidates:
        print("  the serial control is %+.3f %% against the pinned width-1"
              " round; it runs no head step, so it is a lower bound"
              % (100.0 * (serial / candidates["measured"] - 1.0)))

    points[1] = candidates[args.anchor]

    replayed_curves, best_form = load_curves()
    replayed = replayed_curves[best_form]
    measured = installable(points, best_form)

    diff = compare_shapes(replayed, measured)
    print("\n## shape disagreement, each curve normalised to its own width 1")
    for row in diff["per_width"]:
        print("  width %d  replayed %7.4f  measured %7.4f  %+8.3f %%"
              % (row["width"], row["a_norm"], row["b_norm"], row["pct"]))
    print("  worst width %d at %+.3f %%   agrees within 5 %%: %s"
          % (diff["worst_width"], diff["worst_pct"],
             diff["agrees_within_5_pct"]))
    if diff["agrees_within_5_pct"]:
        print("  the assignment's skip rule fires: R3 was not needed")

    tiers = {REPLAYED_LABEL: tier_implied(replayed),
             MEASURED_LABEL: tier_implied(measured)}
    print("\n## the one-boundary tier each curve asks for")
    for label, row in tiers.items():
        print("  %-9s cliff step %9.1f us   mean other %9.1f us"
              "   tier %.4f   against first step %.4f   argmax boundary %d"
              % (label, row["cliff_step_us"], row["mean_other_step_us"],
                 row["tier_against_mean_other"],
                 row["tier_against_first_step"], row["argmax_step_index"]))

    bounds = {REPLAYED_LABEL: boundary_table(replayed),
              MEASURED_LABEL: boundary_table(measured)}
    print("\n## every boundary, as a multiple of the curve's own mean step")
    print("  %-9s %s" % ("boundary", "  ".join(
        "%d->%d" % (r["from_width"], r["to_width"])
        for r in bounds[REPLAYED_LABEL])))
    for label, rows in bounds.items():
        print("  %-9s %s" % (label, "  ".join(
            "%5.2f" % r["share_of_mean_step"] for r in rows)))
    extrap = measured["provenance"]["extrapolated_widths"]
    if extrap:
        print("  widths %s are extrapolated from the last measured step, so"
              " every boundary that touches them is not a measurement"
              % ", ".join(str(w) for w in extrap))

    receipt = load_board_receipt(args.board, args.receipt)
    seeds = [args.seed + i for i in range(args.seeds)]
    legs_e128, gate = build_legs(args.accept, args.runs)
    print("\n## attachment gate")
    print("  legs %d ; attached %d ; accept mismatches %d ; margin mismatches"
          " %d ; unmatched %d"
          % (gate["legs"], gate["attached"], gate["accept_mismatch"],
             gate["margin_mismatch"], gate["unmatched"]))
    if gate["accept_mismatch"] or gate["margin_mismatch"] or gate["unmatched"]:
        raise SystemExit("attachment is not proven; every number would be void")

    cache = transfer_cache(legs_e128, args.windows, args.fit_windows, seeds)
    cost_curves = {REPLAYED_LABEL: replayed, MEASURED_LABEL: measured}
    price_curves = {REPLAYED_LABEL: replayed, MEASURED_LABEL: measured}
    grid = run_grid(cache, seeds, receipt, args.windows,
                    cost_curves, price_curves, list(CELLS))

    print("\n## cells, replayed median gain in percent against the shipped arm")
    print("  %-14s %10s %10s %10s" % ("cell", "cost=rep", "cost=meas",
                                      "cost=meas"))
    print("  %-14s %10s %10s %10s" % ("", "price=rep", "price=rep",
                                      "price=meas"))
    for cell in CELLS:
        row = []
        for cost, price in ((REPLAYED_LABEL, REPLAYED_LABEL),
                            (MEASURED_LABEL, REPLAYED_LABEL),
                            (MEASURED_LABEL, MEASURED_LABEL)):
            row.append(grid["%s|%s|%s" % (cost, price, cell)]
                       ["median_pct_mean"])
        print("  %-14s %+10.4f %+10.4f %+10.4f" % (cell, row[0], row[1],
                                                   row[2]))

    scale_gate = None
    if args.check_scale:
        scaled = installable({w: 3.7 * v for w, v in points.items()},
                             best_form)
        small = run_grid(cache, seeds[:1], receipt, args.windows,
                         {MEASURED_LABEL: scaled}, {MEASURED_LABEL: scaled},
                         list(CELLS))
        worst = 0.0
        for cell in CELLS:
            a = grid["%s|%s|%s" % (MEASURED_LABEL, MEASURED_LABEL, cell)]
            b = small["%s|%s|%s" % (MEASURED_LABEL, MEASURED_LABEL, cell)]
            worst = max(worst, abs(a["median_pct_values"][0]
                                   - b["median_pct_values"][0]))
        scale_gate = {"factor": 3.7, "worst_abs_median_pct_shift": worst,
                      "scale_free": worst < 1e-9}
        print("\n## scale-free gate")
        print("  the measured curve times 3.7 moves the worst cell by %.3e"
              " percentage points   scale free: %s"
              % (worst, scale_gate["scale_free"]))
        if not scale_gate["scale_free"]:
            raise SystemExit("the cells read the curve level; the local"
                             " measurement cannot be transferred by shape")

    blob = {
        "harness": "local instrument",
        "gpu_used": False,
        "best_form": best_form,
        "receipt": args.receipt,
        "seeds": seeds,
        "windows": args.windows,
        "measured_points_us": points,
        "width1_anchor": anchor,
        "width1_candidates_us": candidates,
        "width1_anchor_used": args.anchor,
        "serial_control_round_us": serial,
        "shape_disagreement": diff,
        "tier_implied": tiers,
        "boundary_table": bounds,
        "cells": grid,
        "scale_gate": scale_gate,
        "replayed_shape": shape(replayed),
        "measured_shape": shape(measured),
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(blob, indent=2, sort_keys=True) + "\n")
    print("\nwrote %s" % args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
