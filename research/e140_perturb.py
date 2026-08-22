#!/usr/bin/env python3
"""E140 F1 item 3: does a flatter width-6 cliff cost `pb6` more than lookahead?

`harness=local instrument`. Zero GPU.

The advisor's claim is that `pb6` and alphonse's plan table attack the same
cliff from opposite sides and therefore do not compose. `pb6` declines to enter
width 6 because entering is expensive; a plan table that makes width 6 cheaper
turns that decline into a loss. A parameter-free argmax over the measured curve
is supposed to re-optimise instead.

The falsifier the advisor pre-registered: if `pb6` and cell D degrade at the
same rate, the adaptivity claim is wrong.

The perturbation scales the step into verify width 6 down by a fraction and
leaves every other step exactly as measured. Subtracting the same absolute
microsecond amount from every width at or above 6 does that: only the 6 step
moves, and widths 7, 8 and 9 keep the marginal costs E134 item 2 measured.

The perturbed curve is installed as the TRUE round cost for every arm, so the
scored world really is flatter. Only cell D's price table reads it; `pb6` and
the shipped flat table are constants and cannot see the change. That asymmetry
is the whole experiment.

The grid is the requested 0, 25 and 50 percent, plus one point calibrated to
alphonse's actual measurement of 4.6 percent of an M = 6 round, so the headline
is not read off a perturbation nobody has observed.

Usage:
  python3 e140_perturb.py --json e140-artifacts/perturb.json
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
from e128_price import MAX_DEPTH, RANKED_PROMPTS, load_board_receipt  # noqa: E402
from e134_item2_refit import FORMS  # noqa: E402
from e134_rung2 import build_legs, median_pct  # noqa: E402
from e140_cells import (  # noqa: E402
    TIER_GRID, install, load_masses, lookahead_walker, lopo_curves, run_cell,
    summarise, tier_ratios, transfer_cache,
)
from e140_lookahead import CURVE_FORMS, SHIPPED_TIER, load_curves  # noqa: E402

# `A_ship` is the denominator of every ratio, so it is 0.0000 by construction
# at every perturbation and is printed as the anchor rather than measured.
ARMS = ("A_ship", "C_flatlook", "D_curvelook", "E_pb6", "F_pb6look")

CLIFF_WIDTH = 6
# Alphonse's isolated measurement, as a share of one M = 6 round.
ALPHONSE_ROUND_SHARE = 0.046


def cliff_step(curve, width: int = CLIFF_WIDTH) -> tuple[float, float]:
    """`(step into width, cost of one round at width)` under `curve`."""
    saved = e128_price.CURVE
    try:
        install(curve)
        at = e128_price.ranked_round_us(width)
        below = e128_price.ranked_round_us(width - 1)
    finally:
        e128_price.CURVE = saved
    return at - below, at


def perturb(curve, fraction: float, width: int = CLIFF_WIDTH) -> dict:
    """Shrink the step into `width` by `fraction`, leaving later steps fixed."""
    step, _ = cliff_step(curve, width)
    out = dict(curve)
    per_width = {int(w): v for w, v in (curve.get("per_width") or {}).items()}
    for rows in range(width, MAX_DEPTH + 2):
        per_width[rows] = per_width.get(rows, 0.0) - fraction * step
    out["per_width"] = per_width
    out["name"] = "%s|cliff-%.1f%%" % (curve.get("name", "?"), fraction * 100)
    return out


def arm_row(cache, seeds, arms_curve, lopo, receipt, windows, cell):
    """One arm at one perturbation: in-sample, curve-LOPO, and depth mass."""
    in_sample, held_out = [], []
    depth_counts = {}
    depths = []
    for seed in seeds:
        ratios = {}
        for prompt in RANKED_PROMPTS:
            row = run_cell(cache, seed, prompt, cell, arms_curve, windows)
            ratios[prompt] = row
            depths.append(RANKED_PROMPTS[prompt]["weight"] * row["mean_depth"])
            for depth, count in enumerate(row["depth_counts"]):
                depth_counts[depth] = depth_counts.get(depth, 0) + count
        in_sample.append(median_pct(receipt, ratios))
        if lopo is not None:
            folded = {p: run_cell(cache, seed, p, cell, lopo[p], windows)
                      for p in RANKED_PROMPTS}
            held_out.append(median_pct(receipt, folded))
    total = sum(depth_counts.values()) or 1
    return {
        "in_sample": summarise(in_sample),
        "curve_lopo": summarise(held_out) if held_out else None,
        "mean_depth": sum(depths) / len(seeds),
        "depth_share": [depth_counts.get(d, 0) / total
                        for d in range(MAX_DEPTH + 1)],
    }


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
    ap.add_argument("--forms", default=",".join(CURVE_FORMS))
    ap.add_argument("--tier-form", default="per_drafting_round")
    ap.add_argument("--skip-tier-sweep", action="store_true")
    ap.add_argument("--json", type=pathlib.Path,
                    default=HERE / "e140-artifacts/perturb.json")
    args = ap.parse_args()

    print("harness=local instrument  E140 F1 item 3  cliff perturbation  "
          "zero GPU")
    curves, _ = load_curves()
    masses = load_masses()
    receipt = load_board_receipt(args.board, args.receipt)
    seeds = [args.seed + i for i in range(args.seeds)]
    legs, gate = build_legs(args.accept, args.runs)
    if gate["accept_mismatch"] or gate["margin_mismatch"] or gate["unmatched"]:
        raise SystemExit("attachment is not proven; every number would be void")
    cache = transfer_cache(legs, args.windows, args.fit_windows, seeds)

    forms = [f for f in args.forms.split(",") if f]
    step, round_us = cliff_step(curves[args.tier_form])
    alphonse = ALPHONSE_ROUND_SHARE * round_us / step
    # The requested 0, 25 and 50 percent, alphonse's calibrated point, and
    # three fill-in points, because the smoke run put the `pb6`-to-`pb6look`
    # crossover below 25 percent and left cell D non-monotone across it.
    fractions = sorted([0.0, 0.05, 0.10, alphonse, 0.25, 0.375, 0.50])
    print("\n## the perturbation grid, on form %s" % args.tier_form)
    print("step into width 6           %.1f us" % step)
    print("one round at width 6        %.1f us" % round_us)
    print("alphonse 4.6%% of an M=6 round = %.1f us = %.1f%% of the step"
          % (ALPHONSE_ROUND_SHARE * round_us, alphonse * 100))

    results = {}
    for form in forms:
        base_curve = curves[form]
        base_lopo = lopo_curves(masses, form) if form in FORMS else None
        print("\n## form %s" % form)
        header = "  %-10s " % "cliff-cut" + " ".join(
            "%-14s" % a for a in ARMS)
        print(header)
        for fraction in fractions:
            curve = perturb(base_curve, fraction)
            lopo = ({p: perturb(c, fraction) for p, c in base_lopo.items()}
                    if base_lopo is not None else None)
            row = {}
            for cell in ARMS:
                row[cell] = arm_row(cache, seeds, curve, lopo, receipt,
                                    args.windows, cell)
            results[(form, round(fraction, 6))] = row
            cells = " ".join(
                "%+7.4f/%s" % (
                    row[a]["in_sample"][0],
                    "%+6.4f" % row[a]["curve_lopo"][0]
                    if row[a]["curve_lopo"] else "  n/a ")
                for a in ARMS)
            print("  %-10.4f %s" % (fraction, cells))

    tier_sweep = {}
    if not args.skip_tier_sweep:
        # Comparing a frozen constant with an adaptive rule would prove only
        # that constants are frozen. The re-tuned tier is the strongest form
        # `pb6` can take at each perturbation, so it is the honest opponent.
        print("\n## pb6's optimal tier as the cliff flattens (form %s)"
              % args.tier_form)
        base_curve = curves[args.tier_form]
        for fraction in fractions:
            curve = perturb(base_curve, fraction)
            entry = {}
            for label, walker in (("greedy", None),
                                  ("argmax", lookahead_walker())):
                by_tier = {}
                for tier in TIER_GRID:
                    values = [median_pct(receipt,
                                         tier_ratios(cache, seed, curve, tier,
                                                     args.windows, walker))
                              for seed in seeds]
                    by_tier[tier] = statistics.fmean(values)
                best_tier = max(by_tier, key=by_tier.get)
                entry[label] = {"by_tier": by_tier, "best_tier": best_tier,
                                "best": by_tier[best_tier],
                                "at_shipped": by_tier[SHIPPED_TIER],
                                "at_one": by_tier[1.0]}
                print("  cut %-8.4f %-7s best tier %-7.4f -> %+7.4f   "
                      "shipped 1.45 -> %+7.4f   tier 1.0 -> %+7.4f"
                      % (fraction, label, best_tier, by_tier[best_tier],
                         by_tier[SHIPPED_TIER], by_tier[1.0]))
            tier_sweep[round(fraction, 6)] = entry

    print("\n## depth distribution the argmax chooses (form %s)"
          % args.tier_form)
    print("  %-10s %-14s %-7s %s" % ("cliff-cut", "arm", "depth",
                                     " ".join("d%d" % d for d in
                                              range(MAX_DEPTH + 1))))
    for fraction in fractions:
        row = results.get((args.tier_form, round(fraction, 6)))
        if row is None:
            continue
        for cell in ("A_ship", "D_curvelook", "E_pb6", "F_pb6look"):
            print("  %-10.4f %-14s %-7.4f %s"
                  % (fraction, cell, row[cell]["mean_depth"],
                     " ".join("%.3f" % v for v in row[cell]["depth_share"])))

    print("\n## degradation rate, curve-LOPO, relative to cut 0")
    for form in forms:
        zero = results[(form, 0.0)]
        for cell in ARMS:
            if cell == "A_ship":
                continue
            base_value = (zero[cell]["curve_lopo"] or zero[cell]["in_sample"])[0]
            deltas = []
            for fraction in fractions[1:]:
                row = results[(form, round(fraction, 6))][cell]
                value = (row["curve_lopo"] or row["in_sample"])[0]
                deltas.append("cut %.2f %+7.4f (%+7.4f)"
                              % (fraction, value, value - base_value))
            print("  %-20s %-14s at 0 %+7.4f | %s"
                  % (form, cell, base_value, "  ".join(deltas)))

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps({
        "harness": "local instrument",
        "windows": args.windows, "seeds": seeds,
        "cliff_width": CLIFF_WIDTH,
        "step_us": step, "round_us_at_width": round_us,
        "alphonse_fraction_of_step": alphonse,
        "fractions": fractions,
        "arms": {"%s|%s|%.6f" % (form, cell, fraction): row[cell]
                 for (form, fraction), row in results.items()
                 for cell in row},
        "tier_sweep": {"%.6f" % k: v for k, v in tier_sweep.items()},
    }, indent=2, default=list))
    print("\nwrote %s" % args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
