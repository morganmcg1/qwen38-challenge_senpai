#!/usr/bin/env python3
"""E92: refit `measuredRawDepthPrice` from the production width sweep.

Two conventions, because the E68 one is now suspect:

    pure   marginal ROUND GPU busy normalised by C(1). No additive term.
    hadd   `headStepCostRatio + marginal VERIFY / C(1)`, the E68 convention,
           kept only so the two shapes can be compared cell for cell.

`makeMeasuredDepthPrice` rescales whatever array it is given to the shipped
total `maxDepth * headStepCostRatio`, so only the shape reaches the scheduler.
Both rescaled forms are printed for that reason.

Also prints cost per token over the measured curve and the measured pinned
acceptance profile, and the chosen-depth histogram of the unpinned legs.

    usage: research/e92_depth_price.py [--output PATH]
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from e90_intervals import read_rounds

HEAD_STEP_COST_RATIO = 0.18
MAX_DEPTH = 8
ARTIFACTS = Path("research/e92-artifacts")
UNPINNED_LEGS = ["e90-rung0b-prod", "e90-rung0b-synchead", "e90-lhsindices"]


def load(name):
    table = json.loads((ARTIFACTS / name).read_text())["table"]
    return {row["M"]: row for row in table}


def rescale(raw):
    scale = MAX_DEPTH * HEAD_STEP_COST_RATIO / sum(raw)
    return [value * scale for value in raw]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    production = load("rung2-production-widths.json")
    synchead = load("rung2-widths.json")
    acceptance = json.loads((ARTIFACTS / "acceptance-profile.json").read_text())

    forms = {}
    for label, table in (("production", production), ("sync_head", synchead)):
        anchor = table[1]["round_gpu_busy_us"]
        anchor_verify = table[1]["verify_gpu_busy_us"]
        forms[label + "_pure"] = [
            table[m]["marginal_round_gpu_busy_us"] / anchor
            for m in range(2, 10)]
        forms[label + "_hadd"] = [
            HEAD_STEP_COST_RATIO
            + table[m]["marginal_verify_gpu_busy_us"] / anchor_verify
            for m in range(2, 10)]

    shipped = [0.26300121724709807, 0.29195567495854047, 0.34642143034825884,
               0.40231023217247086, 0.63287276451077956, 0.43601634825870655,
               0.35457813598673293, 0.42510483416251998]
    forms["e68_shipped_pbfit"] = shipped

    print("raw arrays, step into width 2..9")
    for label, raw in forms.items():
        print("%-22s [%s]" % (label, ", ".join("%.8f" % v for v in raw)))
    print()
    print("after makeMeasuredDepthPrice rescaling to %.2f"
          % (MAX_DEPTH * HEAD_STEP_COST_RATIO))
    for label, raw in forms.items():
        print("%-22s [%s]" % (label, ", ".join("%.6f" % v
                                               for v in rescale(raw))))

    print()
    print("cost per token over the measured production curve")
    print("  d  width  C(d+1) us   Y_pin   C/Y_pin   vs best"
          "    Y_pool  C/Y_pool   vs best")
    costs = {m: production[m]["round_gpu_busy_us"] for m in production}
    pinned = acceptance["per_pinned_depth"]
    pooled = acceptance["pooled_expected_tokens"]
    rows = []
    for depth in range(0, 9):
        width = depth + 1
        pin_y = 1.0 if depth == 0 else pinned[str(depth)]["expected_tokens"]
        rows.append({"depth": depth, "width": width,
                     "round_gpu_busy_us": costs[width],
                     "expected_tokens_pinned": pin_y,
                     "us_per_token_pinned": costs[width] / pin_y,
                     "expected_tokens_pooled": pooled[depth],
                     "us_per_token_pooled": costs[width] / pooled[depth]})
    best_pin = min(row["us_per_token_pinned"] for row in rows)
    best_pool = min(row["us_per_token_pooled"] for row in rows)
    for row in rows:
        print("%3d %6d %11.1f %7.4f %9.1f %8.1f %% %8.4f %9.1f %8.1f %%"
              % (row["depth"], row["width"], row["round_gpu_busy_us"],
                 row["expected_tokens_pinned"], row["us_per_token_pinned"],
                 (row["us_per_token_pinned"] / best_pin - 1.0) * 100.0,
                 row["expected_tokens_pooled"], row["us_per_token_pooled"],
                 (row["us_per_token_pooled"] / best_pool - 1.0) * 100.0))

    print()
    print("chosen-depth histogram, unpinned legs, shipped schedule")
    histograms = {}
    for tag in UNPINNED_LEGS:
        if not (Path("research/out") / tag / "trace.txt").exists():
            continue
        depths = [row["d"] for row in read_rounds(tag)]
        histogram = collections.Counter(depths)
        total = len(depths)
        histograms[tag] = {
            "rounds": total,
            "histogram": {str(k): v for k, v in sorted(histogram.items())},
            "mass_at_depth_4": histogram[4] / total,
            "mass_at_depth_5": histogram[5] / total,
            "mean_depth": sum(depths) / total,
        }
        print("%-22s rounds=%3d mean_d=%.3f  %s"
              % (tag, total, histograms[tag]["mean_depth"],
                 dict(sorted(histogram.items()))))

    result = {
        "raw": forms,
        "rescaled": {k: rescale(v) for k, v in forms.items()},
        "cost_per_token": rows,
        "chosen_depth_histograms": histograms,
    }
    if arguments.output:
        arguments.output.write_text(json.dumps(result, indent=2,
                                               sort_keys=True))


if __name__ == "__main__":
    main()
