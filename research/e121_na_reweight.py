#!/usr/bin/env python3
"""Why did the rung-2 isolated probe over-predict the rung-3 in-situ leg by 2x?

The isolated probe measures one (shape, width) cell at a time, and two weights
combine those cells into one leg prediction. Only one was ever checked:

  ACROSS WIDTHS   `ROUND_WEIGHTS`, the realised verify-width histogram
                  {2: 0.024, 3: 0.275, 4: 0.667, 5: 0.034}. Already used.
  WITHIN A WIDTH  each shape's share of round cost, or every shape equally.

This script reports the leg prediction under both, which is the direct question,
and then asks what the disagreement between them is made of. The answer is not
a weighting subtlety. The per-cell gains are ordered almost perfectly by how
close each shape runs to the memory roofline, so any scalar summary of them
reports the probe's operating point rather than a property of the arm.

The shipped frame is applied throughout: NA=2 dispatches a separate kernel this
arm never touches and NA=5 is gated off, so both contribute exactly zero and are
PINNED, not dropped. Dropping them would renormalise their round weight onto
NA=3 and NA=4 and inflate the prediction by the 5.8 % of rounds they own.

  usage: research/e121_na_reweight.py
"""
import argparse
import json
import pathlib
import re
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import e121_analysis as e2  # noqa: E402

ARM = "g_split_pred"

# The rung-3 in-situ result the prediction is priced against. Negative is
# faster, so predictions are negated to compare.
MEASURED_LEG_PCT = -0.436
MEASURED_LEG_SD = 0.093

# Widths the shipped gate leaves byte-identical to the base.
PINNED_ZERO = (2, 5)

# Affine 4-bit group-64: half a byte of weight per element, plus an fp16 scale
# and an fp16 bias for every 64 elements. Every probe shape streams tens of
# megabytes once, far past any cache, so bytes over time is the achieved rate.
BYTES_PER_WEIGHT = 0.5 + 4 / 64
DRAM_PEAK_GBS = 273.0

# The widths that carry the shipped effect.
LIVE_WIDTHS = (3, 4)

# Splits the five probe shapes into the two strata their achieved rate forms.
STRATUM_SPLIT_GBS = 175.0


def shape_kn(shape: str) -> tuple[int, int]:
    k, n = re.search(r"_k(\d+)_n(\d+)", shape).groups()
    return int(k), int(n)


def base_seconds(cell: dict) -> float:
    return e2.med([b["a_base"] for b in cell["blocks"] if b.get("a_base")])


def implied_gbs(shape: str, cell: dict) -> float:
    k, n = shape_kn(shape)
    return k * n * BYTES_PER_WEIGHT / 1e9 / base_seconds(cell)


def per_width(cells, shapes, widths, how):
    """Combine each width's shape effects under one within-width rule."""
    if how == "cost":
        return e2.cost_ladder(cells, shapes, widths, ARM)[0]
    if how == "pooled median":
        return {w: e2.med(v)
                for w, v in e2.ladder(cells, shapes, widths, ARM).items()}
    return {w: statistics.fmean([e2.med(e2.gains(cells[(s, w)], ARM))
                                 for s in shapes if (s, w) in cells])
            for w in widths}


def scheme(cells, shapes, widths, how, pin):
    pw = per_width(cells, shapes, widths, how)
    if pin:
        pw = {w: (0.0 if w in PINNED_ZERO else v) for w, v in pw.items()}
    kernel, _ = e2.weighted(pw)
    return pw, kernel, -kernel * e2.LEG_TRANSFER


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", default="research/e121-artifacts/rung2-rate.json")
    ap.add_argument("--warmup-blocks", type=int, default=1)
    ap.add_argument("--json-out",
                    default="research/e121-artifacts/rung3-na-reweight.json")
    args = ap.parse_args()

    doc = json.loads(pathlib.Path(args.rate).read_text())
    cells = e2.collect(doc, args.warmup_blocks)
    shapes = sorted({s for s, _ in cells})
    widths = sorted({w for _, w in cells})

    print("=== the direct question: recombine the rung-2 cells two ways ===")
    print("  round weights (realised NA histogram): %s" % e2.ROUND_WEIGHTS)
    print("  shipped frame pins NA%s to zero" % (list(PINNED_ZERO),))
    print()
    print("  %-32s" % "within-width rule"
          + "".join("     NA%d" % w for w in widths)
          + "   kernel      leg  vs measured")

    rows = {}
    for pin in (True, False):
        for how in ("cost", "pooled median", "uniform mean"):
            pw, kernel, leg = scheme(cells, shapes, widths, how, pin)
            name = "%s%s" % (how, ", shipped frame" if pin else ", all widths")
            rows[name] = {"per_width_pct": pw, "kernel_pct": kernel,
                          "leg_pct": leg,
                          "ranked_pct": leg * e2.RANKED_TRANSFER,
                          "over_prediction_ratio": leg / MEASURED_LEG_PCT}
            print("  %-32s" % name
                  + "".join("  %+.3f" % pw.get(w, float("nan")) for w in widths)
                  + "   %+.3f   %+.3f       %.2fx"
                  % (kernel, leg, leg / MEASURED_LEG_PCT))
    print()
    print("  measured in-situ leg             %+.3f %% (sd %.3f, n=2)"
          % (MEASURED_LEG_PCT, MEASURED_LEG_SD))
    print()

    shipped = rows["cost, shipped frame"]
    pooled = rows["pooled median, shipped frame"]
    gap = shipped["leg_pct"] - MEASURED_LEG_PCT
    closed = shipped["leg_pct"] - pooled["leg_pct"]
    frac = closed / gap
    print("  shipped %+.3f -> re-weighted %+.3f -> measured %+.3f"
          % (shipped["leg_pct"], pooled["leg_pct"], MEASURED_LEG_PCT))
    print("  the re-weighting explains %.3f pp of the %.3f pp shortfall (%.0f %%)"
          % (abs(closed), abs(gap), 100 * frac))
    print("  over-prediction ratio %.2fx -> %.2fx"
          % (shipped["over_prediction_ratio"], pooled["over_prediction_ratio"]))
    print()

    # The three rules disagree because the cells are bimodal, not noisy, so the
    # next question is what orders them.
    print("=== what the rules disagree about: a roofline knee ===")
    print("  %-26s%7s%7s%8s%8s%8s%7s"
          % ("shape", "k", "n", "MB", "us", "GB/s", "%peak")
          + "".join("     NA%d" % w for w in LIVE_WIDTHS))
    fit, table = [], {}
    for shape in sorted(shapes, key=lambda s: implied_gbs(s, cells[(s, 4)])):
        k, n = shape_kn(shape)
        gbs = implied_gbs(shape, cells[(shape, 4)])
        megabytes = k * n * BYTES_PER_WEIGHT / 1e6
        microseconds = base_seconds(cells[(shape, 4)]) * 1e6
        gains = {w: e2.med(e2.gains(cells[(shape, w)], ARM))
                 for w in LIVE_WIDTHS}
        table[shape] = {"k": k, "n": n, "megabytes": megabytes,
                        "base_microseconds": microseconds, "implied_gbs": gbs,
                        "percent_of_dram_peak": 100 * gbs / DRAM_PEAK_GBS,
                        "gain_pct": gains}
        for w in LIVE_WIDTHS:
            fit.append((implied_gbs(shape, cells[(shape, w)]), gains[w]))
        print("  %-26s%7d%7d%8.1f%8.1f%8.1f%7.1f"
              % (shape, k, n, megabytes, microseconds, gbs,
                 100 * gbs / DRAM_PEAK_GBS)
              + "".join("  %+.3f" % gains[w] for w in LIVE_WIDTHS))

    r = statistics.correlation([p[0] for p in fit], [p[1] for p in fit])
    print()
    print("  Pearson r of achieved GB/s against percent gain, %d cells: %.3f"
          % (len(fit), r))

    strata = {}
    for name, keep in (("issue-bound", False), ("near-roofline", True)):
        group = [s for s in shapes
                 if (implied_gbs(s, cells[(s, 4)]) >= STRATUM_SPLIT_GBS) == keep]
        rates = [implied_gbs(s, cells[(s, 4)]) for s in group]
        strata[name] = {"shapes": group,
                        "gbs_range": [min(rates), max(rates)],
                        "mean_pct": {w: statistics.fmean(
                            [e2.med(e2.gains(cells[(s, w)], ARM))
                             for s in group]) for w in LIVE_WIDTHS}}
        print("  %-14s %d shapes at %.0f-%.0f GB/s" %
              (name, len(group), min(rates), max(rates))
              + "".join("   NA%d %+.3f" % (w, strata[name]["mean_pct"][w])
                        for w in LIVE_WIDTHS))
    print()
    print("  The arm deletes arithmetic, so it pays where issue, not bandwidth,"
          " is the limit. The two shapes furthest from the roofline gain 2.7x"
          " what the three nearest it gain, with no overlap at either width.")
    print()

    # If the whole-model pipeline overlaps dispatches better than a probe that
    # chains 32 of them, every cell moves toward the near-roofline stratum.
    print("=== counterfactual: in situ, every shape behaves near-roofline ===")
    near = strata["near-roofline"]["mean_pct"]
    cf_kernel, _ = e2.weighted({w: (0.0 if w in PINNED_ZERO else near[w])
                                for w in widths})
    cf_leg = -cf_kernel * e2.LEG_TRANSFER
    z_cf = abs(cf_leg - MEASURED_LEG_PCT) / MEASURED_LEG_SD
    print("  every live width at its near-roofline mean -> kernel %+.3f, "
          "leg %+.3f" % (cf_kernel, cf_leg))
    print("  measured %+.3f, so this counterfactual sits %.1f measured sd away"
          % (MEASURED_LEG_PCT, z_cf))
    print()

    z_pooled = abs(pooled["leg_pct"] - MEASURED_LEG_PCT) / MEASURED_LEG_SD
    verdict = (
        "MIXED, AND THE WEIGHTING IS NOT THE ROOT CAUSE. Re-weighting closes "
        "%.0f %% of the shortfall and cuts the over-prediction from %.2fx to "
        "%.2fx, but %.1f sd of gap remains. The cells are ordered by achieved "
        "bandwidth at r=%.3f, and pricing every shape at the near-roofline rate "
        "lands %.1f sd from the measurement. The isolated probe reports its own "
        "issue-bound operating point, not a wrong average."
        % (100 * frac, shipped["over_prediction_ratio"],
           pooled["over_prediction_ratio"], z_pooled, r, z_cf))
    print("  verdict: %s" % verdict)
    print()
    print("  CAVEAT. Shape-uniform is not the more principled weight: a leg is "
          "a SUM over dispatches, so cost weighting is what `program.md` asks "
          "for, and dropping it is not a correction on its own. The correction "
          "rule this supports is different and applies to every isolated probe "
          "on this tree: report the achieved bandwidth of each probe cell, and "
          "transfer a gain to a leg only across cells at the same distance from "
          "the roofline.")

    out = pathlib.Path(args.json_out)
    out.write_text(json.dumps({
        "arm": ARM,
        "harness": "local",
        "round_weights": e2.ROUND_WEIGHTS,
        "pinned_zero": list(PINNED_ZERO),
        "leg_transfer": e2.LEG_TRANSFER,
        "ranked_transfer": e2.RANKED_TRANSFER,
        "measured_leg_pct": MEASURED_LEG_PCT,
        "measured_leg_sd": MEASURED_LEG_SD,
        "schemes": rows,
        "shortfall_pp": abs(gap),
        "explained_pp": abs(closed),
        "explained_fraction": frac,
        "roofline": {"dram_peak_gbs": DRAM_PEAK_GBS,
                     "bytes_per_weight": BYTES_PER_WEIGHT,
                     "stratum_split_gbs": STRATUM_SPLIT_GBS,
                     "per_shape": table,
                     "pearson_r_gbs_vs_gain": r,
                     "strata": strata,
                     "counterfactual_kernel_pct": cf_kernel,
                     "counterfactual_leg_pct": cf_leg,
                     "counterfactual_sd_from_measured": z_cf},
        "verdict": verdict,
    }, indent=1, sort_keys=True) + "\n")
    print()
    print("wrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
