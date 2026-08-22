#!/usr/bin/env python3
"""E140 section 4: is the walk rule wrong, or is the estimator starved?

`harness=local instrument`. Zero GPU.

The assignment names one mechanism that would make the hypothesis fail: "the
reach estimator is biased, and lookahead consumes more of it". Cell D is worse
than that. It fires on no round at all, because the measured price parks the
walk at depth 4 and `recordAcceptOutcome` only updates a position the round
actually reached, so positions 5 to 7 stay frozen at `EMA_PRIOR`.

Bias and starvation are different defects and they imply different follow-ups.
This file separates them by replacing the estimator with the sampler's own
per-position conditional acceptance vector, which is the quantity the EMA is
trying to estimate. Nothing else changes: the price, the caps, the margin
clamps at depths 0 and 1, and the round trajectory are all as shipped.

    cell                       price     walk     acceptance state
    A_ship                     flat      greedy   EMA, the anchor
    B_rankedprice              measured  greedy   EMA
    D_curvelook                measured  argmax   EMA
    Boracle_rankedprice        measured  greedy   TRUE per-position vector
    Doracle_curvelook          measured  argmax   TRUE per-position vector
    E_pb6                      pb6       greedy   EMA
    Foracle_pb6look            pb6       argmax   TRUE per-position vector

The oracle arms are diagnostics, not candidates. No shipped policy can read
the true vector, so their score is an upper bound on what fixing the estimator
could buy, not a result that could be submitted.

Usage:
  python3 e140_oracle_state.py --json e140-artifacts/oracle-state.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from e128_price import MAX_DEPTH, RANKED_PROMPTS, load_board_receipt  # noqa: E402
from e134_item2_refit import FORMS  # noqa: E402
from e134_rung2 import build_legs, median_pct, simulate, walk  # noqa: E402
from e140_cells import (  # noqa: E402
    install, load_masses, lopo_curves, price_for, summarise, transfer_cache,
)
from e140_lookahead import load_curves, walk_argmax  # noqa: E402

CELLS = {
    "A_ship": ("flat", "greedy", False),
    "B_rankedprice": ("curve", "greedy", False),
    "D_curvelook": ("curve", "argmax", False),
    "Boracle_rankedprice": ("curve", "greedy", True),
    "Doracle_curvelook": ("curve", "argmax", True),
    "E_pb6": ("pb6", "greedy", False),
    "Foracle_pb6look": ("pb6", "argmax", True),
}


def make_walker(kind: str, oracle_vector=None):
    """The shipped rule or the argmax, optionally on the true accept vector."""
    if kind == "argmax":
        def rule(state, margin, offer, ctx, price):
            return walk_argmax(state, margin, offer, price=price)
    else:
        def rule(state, margin, offer, ctx, price):
            return walk(state, margin, offer, None, ctx, None, price)
    vector = (None if oracle_vector is None else
              [oracle_vector[min(i, len(oracle_vector) - 1)]
               for i in range(MAX_DEPTH)])

    def chooser(ema, margin, offer, adjust=None, ctx=None, force=None,
                price=None):
        return rule(vector if vector is not None else ema, margin, offer,
                    ctx, price)
    return chooser


def run_cell(cache, seed, prompt, cell, curve, windows):
    entry = cache[(seed, prompt)]
    install(curve)
    base = simulate(None, entry["factory"](entry["p_target"]), windows)
    kind, walk_kind, oracle = CELLS[cell]
    walker = make_walker(walk_kind,
                         entry["p_target"] if oracle else None)
    run = simulate(None, entry["factory"](entry["p_target"]), windows,
                   price=price_for(kind, curve), walker=walker)
    return {"ratio": run["us_per_token"] / base["us_per_token"],
            "mean_depth": run["mean_depth"],
            "accept_rate": run["accept_rate"],
            "depth_counts": run["depth_counts"]}


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
    ap.add_argument("--form", default="per_drafting_round")
    ap.add_argument("--json", type=pathlib.Path,
                    default=HERE / "e140-artifacts/oracle-state.json")
    args = ap.parse_args()

    print("harness=local instrument  E140 estimator starvation  zero GPU")
    curves, _ = load_curves()
    masses = load_masses()
    receipt = load_board_receipt(args.board, args.receipt)
    seeds = [args.seed + i for i in range(args.seeds)]
    legs, gate = build_legs(args.accept, args.runs)
    if gate["accept_mismatch"] or gate["margin_mismatch"] or gate["unmatched"]:
        raise SystemExit("attachment is not proven; every number would be void")
    cache = transfer_cache(legs, args.windows, args.fit_windows, seeds)
    curve = curves[args.form]
    lopo = lopo_curves(masses, args.form) if args.form in FORMS else None

    print("\n## the true per-position acceptance vector each prompt is "
          "sampled from")
    print("%-10s %s" % ("prompt", " ".join("%7s" % ("p%d" % i)
                                           for i in range(MAX_DEPTH))))
    for prompt in RANKED_PROMPTS:
        vector = cache[(seeds[0], prompt)]["p_target"]
        print("%-10s %s" % (prompt, " ".join(
            "%7.4f" % vector[i] for i in range(min(MAX_DEPTH, len(vector))))))
    print("%-10s %s" % ("EMA_PRIOR", " ".join(
        "%7.4f" % (0.85 * 0.98 ** i) for i in range(MAX_DEPTH))))

    results = {}
    print("\n## replayed ranked median percent, form %s" % args.form)
    print("%-22s %10s %8s %12s %8s %10s" % (
        "cell", "in-sample", "sd", "curve-lopo", "sd", "mean depth"))
    for cell in CELLS:
        in_sample, held_out, depths = [], [], []
        counts = [0] * (MAX_DEPTH + 2)
        for seed in seeds:
            ratios = {}
            for prompt in RANKED_PROMPTS:
                row = run_cell(cache, seed, prompt, cell, curve, args.windows)
                ratios[prompt] = row
                depths.append(row["mean_depth"])
                for index, value in enumerate(row["depth_counts"]):
                    counts[index] += value
            in_sample.append(median_pct(receipt, ratios))
            if lopo is not None:
                held_out.append(median_pct(receipt, {
                    p: run_cell(cache, seed, p, cell, lopo[p], args.windows)
                    for p in RANKED_PROMPTS}))
        total = sum(counts) or 1
        results[cell] = {
            "in_sample": summarise(in_sample),
            "curve_lopo": summarise(held_out) if held_out else None,
            "unweighted_mean_depth": statistics.fmean(depths),
            "depth_share": [c / total for c in counts],
        }
        print("%-22s %+10.4f %8.4f %12s %8s %10.4f" % (
            cell, results[cell]["in_sample"][0], results[cell]["in_sample"][1],
            "%+.4f" % results[cell]["curve_lopo"][0] if held_out else "n/a",
            "%.4f" % results[cell]["curve_lopo"][1] if held_out else "",
            results[cell]["unweighted_mean_depth"]))

    print("\n## depth distribution")
    print("%-22s %s" % ("cell", " ".join("%7s" % ("d%d" % d)
                                         for d in range(MAX_DEPTH + 1))))
    for cell in CELLS:
        print("%-22s %s" % (cell, " ".join(
            "%7.4f" % v for v in results[cell]["depth_share"][:MAX_DEPTH + 1])))

    def value(cell):
        entry = results[cell]
        return (entry["curve_lopo"] or entry["in_sample"])[0]

    print("\n## what the estimator costs, held out where a fit exists")
    print("argmax on the measured price, EMA state          %+8.4f"
          % value("D_curvelook"))
    print("argmax on the measured price, TRUE state         %+8.4f"
          % value("Doracle_curvelook"))
    print("  the estimator is worth                         %+8.4f pp"
          % (value("Doracle_curvelook") - value("D_curvelook")))
    print("greedy on the measured price, TRUE state         %+8.4f"
          % value("Boracle_rankedprice"))
    print("  the argmax is worth, with a perfect estimator  %+8.4f pp"
          % (value("Doracle_curvelook") - value("Boracle_rankedprice")))
    print("pb6, EMA state                                   %+8.4f"
          % value("E_pb6"))
    print("pb6 plus argmax, TRUE state                      %+8.4f"
          % value("Foracle_pb6look"))

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps({
        "harness": "local instrument", "gpu_used": False,
        "form": args.form, "windows": args.windows, "seeds": seeds,
        "receipt": receipt["id"],
        "p_target": {p: list(cache[(seeds[0], p)]["p_target"])
                     for p in RANKED_PROMPTS},
        "cells": {cell: {
            "in_sample_mean": entry["in_sample"][0],
            "in_sample_sd": entry["in_sample"][1],
            "curve_lopo_mean": (entry["curve_lopo"][0]
                                if entry["curve_lopo"] else None),
            "curve_lopo_sd": (entry["curve_lopo"][1]
                              if entry["curve_lopo"] else None),
            "unweighted_mean_depth": entry["unweighted_mean_depth"],
            "depth_share": entry["depth_share"],
        } for cell, entry in results.items()},
        "estimator_worth_pp": (value("Doracle_curvelook")
                               - value("D_curvelook")),
        "argmax_worth_with_perfect_estimator_pp": (
            value("Doracle_curvelook") - value("Boracle_rankedprice")),
    }, indent=1, sort_keys=True) + "\n")
    print("\nwrote %s" % args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
