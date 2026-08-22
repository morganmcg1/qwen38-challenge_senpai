#!/usr/bin/env python3
"""E140 F5/F6 items A, B and D: the shallow end, and pb6's tier after the grid.

`harness=local instrument`. Zero GPU.

Item B is the part with shipping value. `passBoundaryTierFactor = 1.45` was
fitted on the pre-tight curve, whose step into verify width 6 was 16241.3 us.
On `log_shipped` that step is 16903.8 us, 4.1 percent steeper, so CAMPAIGN
RULE 117 requires the constant to be re-priced. The grid runs on every curve
so the constant's sensitivity to the launch table is measured rather than
assumed, and it is reported held out with the same leave-one-prompt-out
discipline as the original fit.

Item A reads `P(M = 2)` per prompt under `ship` and under `pb6`. A round that
drafts `d` tokens verifies `d + 1` rows, so `P(M = 2) = P(depth = 1)`.

Item D inverts the width-2 launch receipt `08b67f12 -> 1760479a`. If the
replayed histogram and the launch law are both right, the eight implied
per-round costs must agree. The decisive control on the histogram is its
first moment: the receipt publishes `Mbar` and `R` per prompt, and the replay
predicts both, so a histogram that is biased at the shallow end has to show
up as a biased `Mbar` unless the bias is exactly compensating.

Usage:
  python3 e140_itemab.py --json e140-artifacts/itemab.json
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from e128_price import RANKED_PROMPTS, load_board_receipt  # noqa: E402
from e134_item2_refit import FORMS  # noqa: E402
from e134_rung2 import build_legs, median_pct  # noqa: E402
from e140_cells import (  # noqa: E402
    load_masses, lopo_curves, run_cell, summarise, tier_ratios, transfer_cache,
)
from e140_lookahead import SHIPPED_TIER, load_curves  # noqa: E402
from e140_posttight import LAUNCH_COEF, VARIANTS  # noqa: E402

# `08b67f12 -> 1760479a`. One file, width 2 routed through the custom QMV so
# the host launches one threadgroup column instead of two. R and
# `effective_mean_draft_len` are digit-identical on all eight prompts, so this
# is a pure time contrast at exactly one verify width.
WIDTH2_RECEIPT = {
    "plutarch": {"mbar": 1.156, "rounds": 487, "delta_us": -70.0},
    "drama": {"mbar": 3.298, "rounds": 252, "delta_us": -361.6},
    "travel": {"mbar": 3.648, "rounds": 212, "delta_us": -80.3},
    "beagle": {"mbar": 5.382, "rounds": 110, "delta_us": -208.7},
    "republic": {"mbar": 5.989, "rounds": 93, "delta_us": -42.1},
    "essays": {"mbar": 6.087, "rounds": 92, "delta_us": -289.3},
    "medicine": {"mbar": 6.256, "rounds": 90, "delta_us": -39.0},
    "botany": {"mbar": 7.148, "rounds": 81, "delta_us": -165.2},
}

# The crown's median pair, from the F2 audit of receipt `08b67f12`.
MEDPAIR = {"beagle": 0.4782, "essays": 0.5218}

TIER_GRID_FINE = (1.00, 1.10, 1.20, 1.25, 1.30, 1.35, 1.40, 1.45, 1.50, 1.55,
                  1.60, 1.65, 1.70, 1.80, 1.90, 2.00, 2.25, 2.50)


def prompt_histograms(cache, seeds, curve, windows, cell) -> dict:
    """Per-prompt depth histogram, mean verify width and rounds per window."""
    out = {}
    for prompt in RANKED_PROMPTS:
        counts = None
        depths, rounds = [], []
        for seed in seeds:
            row = run_cell(cache, seed, prompt, cell, curve, windows)
            if counts is None:
                counts = [0] * len(row["depth_counts"])
            for depth, count in enumerate(row["depth_counts"]):
                counts[depth] += count
            depths.append(row["mean_depth"])
            rounds.append(sum(row["depth_counts"]) / windows)
        total = sum(counts) or 1
        share = [c / total for c in counts]
        out[prompt] = {
            "depth_share": share,
            "width_share": {str(d + 1): share[d] for d in range(len(share))},
            "p_width2": share[1],
            "shallow_mass_width_1_to_4": sum(share[0:4]),
            "mean_depth": statistics.fmean(depths),
            "mean_width": statistics.fmean(depths) + 1.0,
            "mean_width_sd": (statistics.stdev(depths)
                              if len(depths) > 1 else 0.0),
            "rounds_per_window": statistics.fmean(rounds),
        }
    return out


def min_mbar_shift(share, extra_mass):
    """Smallest |change in mean width| that adds `extra_mass` at width 2.

    This is the falsifier for explanation (b). If the replay under-counts
    width 2, the missing mass has to come from somewhere above it, and every
    donor width drags the mean. Taking the mass from width 3 first, then 4,
    and so on, is the cheapest possible bias, so the value returned is a lower
    bound on how wrong the replayed mean width would have to be.
    """
    need, shift = extra_mass, 0.0
    for depth in range(2, len(share)):
        if need <= 0:
            break
        take = min(share[depth], need)
        shift += take * ((depth + 1) - 2)
        need -= take
    if need > 1e-12:
        return float("inf")
    return shift


def ols(xs, ys):
    """Slope, intercept and R-squared of a one-regressor least squares fit."""
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return {"slope": float("nan"), "intercept": my, "r2": 0.0}
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    intercept = my - slope * mx
    sst = sum((y - my) ** 2 for y in ys)
    ssr = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    return {"slope": slope, "intercept": intercept,
            "r2": 1.0 - ssr / sst if sst > 0 else 1.0}


def tier_row(cache, seeds, curve, lopo, receipt, windows, tier, base_cache):
    in_sample, held_out = [], []
    for seed in seeds:
        in_sample.append(median_pct(receipt, tier_ratios(
            cache, seed, curve, tier, windows,
            base_cache=base_cache, curve_key="in_sample")))
        if lopo is not None:
            folded = {}
            for prompt in RANKED_PROMPTS:
                folded[prompt] = tier_ratios(
                    cache, seed, lopo[prompt], tier, windows,
                    prompts=[prompt], base_cache=base_cache,
                    curve_key=("lopo", prompt))[prompt]
            held_out.append(median_pct(receipt, folded))
    return {"in_sample": summarise(in_sample),
            "curve_lopo": summarise(held_out) if held_out else None}


def plateau(by_tier, best, tolerance=0.05):
    """The contiguous tier range whose value is within `tolerance` pp of best."""
    tiers = sorted(by_tier)
    target = by_tier[best] - tolerance
    low = high = best
    index = tiers.index(best)
    for tier in reversed(tiers[:index]):
        if by_tier[tier] < target:
            break
        low = tier
    for tier in tiers[index + 1:]:
        if by_tier[tier] < target:
            break
        high = tier
    return low, high


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
                    default=HERE / "e140-artifacts/itemab.json")
    args = ap.parse_args()

    print("harness=local instrument  E140 items A, B and D  zero GPU")
    curves, _ = load_curves()
    masses = load_masses()
    receipt = load_board_receipt(args.board, args.receipt)
    seeds = [args.seed + i for i in range(args.seeds)]
    legs, gate = build_legs(args.accept, args.runs)
    if gate["accept_mismatch"] or gate["margin_mismatch"] or gate["unmatched"]:
        raise SystemExit("attachment is not proven; every number would be void")
    cache = transfer_cache(legs, args.windows, args.fit_windows, seeds)
    base_curve = curves[args.form]
    base_lopo = lopo_curves(masses, args.form) if args.form in FORMS else None

    # ------------------------------------------------------------- item A
    print("\n## Item A: P(M = 2) per prompt, by arm (form %s)" % args.form)
    hist = {cell: prompt_histograms(cache, seeds, base_curve, args.windows,
                                    cell)
            for cell in ("A_ship", "E_pb6")}
    print("  %-10s %8s %8s %9s %9s %9s %9s"
          % ("prompt", "ship", "pb6", "d(pp)", "ship m1-4", "pb6 m1-4",
             "medpair"))
    for prompt in RANKED_PROMPTS:
        ship = hist["A_ship"][prompt]
        pb6 = hist["E_pb6"][prompt]
        print("  %-10s %8.4f %8.4f %+9.4f %9.4f %9.4f %9.4f"
              % (prompt, ship["p_width2"], pb6["p_width2"],
                 100 * (pb6["p_width2"] - ship["p_width2"]),
                 ship["shallow_mass_width_1_to_4"],
                 pb6["shallow_mass_width_1_to_4"],
                 MEDPAIR.get(prompt, 0.0)))
    medpair_ship = sum(w * hist["A_ship"][p]["p_width2"]
                       for p, w in MEDPAIR.items())
    medpair_pb6 = sum(w * hist["E_pb6"][p]["p_width2"]
                      for p, w in MEDPAIR.items())
    print("  medpair-weighted P(M = 2)   ship %.4f   pb6 %.4f   "
          "amplification %.4f"
          % (medpair_ship, medpair_pb6,
             medpair_pb6 / medpair_ship if medpair_ship else float("nan")))

    # ------------------------------------------------------------- item D
    print("\n## Item D: inverting the width-2 launch receipt "
          "`08b67f12 -> 1760479a`")
    law_cost = LAUNCH_COEF * math.log(2.0)
    print("  the launch law removes %.1f us per round at width 2" % law_cost)
    print("  %-10s %8s %8s %9s %9s %10s %10s"
          % ("prompt", "Mbar rx", "Mbar us", "R rx", "R us", "P(M=2) us",
             "implied us"))
    implied, mbar_err, rounds_err = {}, [], []
    for prompt, entry in WIDTH2_RECEIPT.items():
        ship = hist["A_ship"][prompt]
        cost = (-entry["delta_us"] / ship["p_width2"]
                if ship["p_width2"] else float("inf"))
        implied[prompt] = cost
        mbar_err.append(ship["mean_width"] - entry["mbar"])
        rounds_err.append(ship["rounds_per_window"] - entry["rounds"])
        print("  %-10s %8.3f %8.3f %9d %9.1f %10.4f %10.1f"
              % (prompt, entry["mbar"], ship["mean_width"], entry["rounds"],
                 ship["rounds_per_window"], ship["p_width2"], cost))
    values = [v for v in implied.values() if math.isfinite(v)]
    spread = {"min": min(values), "max": max(values),
              "mean": statistics.fmean(values),
              "sd": statistics.stdev(values) if len(values) > 1 else 0.0}
    spread["cv"] = spread["sd"] / spread["mean"] if spread["mean"] else 0.0
    print("  implied per-round cost at width 2: min %.1f max %.1f mean %.1f "
          "sd %.1f CV %.3f"
          % (spread["min"], spread["max"], spread["mean"], spread["sd"],
             spread["cv"]))
    print("  the law predicts %.1f us; ratio of mean implied to law %.3f"
          % (law_cost, spread["mean"] / law_cost))
    print("  first-moment control: mean(Mbar replay - Mbar receipt) %+.4f  "
          "sd %.4f" % (statistics.fmean(mbar_err),
                       statistics.stdev(mbar_err)))
    print("  round-count control:  mean(R replay - R receipt) %+.2f  sd %.2f"
          % (statistics.fmean(rounds_err), statistics.stdev(rounds_err)))

    # Explanation (b) says the replay under-counts width 2. That is not an
    # opinion: the missing mass has to be taken from wider rounds, so it moves
    # the replayed mean width by a computable minimum. Compare that forced
    # shift with the shift actually observed, and with its across-seed noise.
    print("\n  (b) biased shallow end, as a falsifiable bound:")
    print("    %-10s %9s %9s %10s %10s %9s %8s"
          % ("prompt", "need P2", "have P2", "min dMbar", "obs dMbar",
             "seed sd", "b ratio"))
    bias = {}
    for prompt, entry in WIDTH2_RECEIPT.items():
        ship = hist["A_ship"][prompt]
        need = -entry["delta_us"] / law_cost
        forced = min_mbar_shift(ship["depth_share"], max(0.0, need
                                                         - ship["p_width2"]))
        observed = ship["mean_width"] - entry["mbar"]
        sd = ship["mean_width_sd"]
        bias[prompt] = {"required_p_width2": need,
                        "replayed_p_width2": ship["p_width2"],
                        "min_forced_mbar_shift": forced,
                        "observed_mbar_error": observed,
                        "seed_sd": sd,
                        "ratio": forced / abs(observed) if observed else
                        float("inf")}
        print("    %-10s %9.4f %9.4f %10.4f %+10.4f %9.4f %8.1f"
              % (prompt, need, ship["p_width2"], forced, observed, sd,
                 bias[prompt]["ratio"]))
    worst = max(bias, key=lambda p: bias[p]["min_forced_mbar_shift"])
    print("    (b) needs the replayed mean width to be wrong by up to %.4f "
          "(%s); it is wrong by %+.4f"
          % (bias[worst]["min_forced_mbar_shift"], worst,
             bias[worst]["observed_mbar_error"]))

    # (c) says the launch coefficient is simply different at width 2. That is
    # a common scale error, so it predicts a tight spread of implied costs.
    # (a) says the saving is not carried by width-2 rounds at all, so it
    # predicts no relation between the saving and width-2 occupancy.
    prompts = list(WIDTH2_RECEIPT)
    saving = [-WIDTH2_RECEIPT[p]["delta_us"] for p in prompts]
    fits = {
        "on_p_width2": ols([hist["A_ship"][p]["p_width2"] for p in prompts],
                           saving),
        "on_shallow_mass_1_to_4": ols(
            [hist["A_ship"][p]["shallow_mass_width_1_to_4"] for p in prompts],
            saving),
        "on_mean_width": ols([hist["A_ship"][p]["mean_width"]
                              for p in prompts], saving),
    }
    print("\n  what explains the eight savings? one regressor at a time:")
    for name, fit in fits.items():
        print("    %-24s slope %12.1f  intercept %9.1f us  R2 %+.4f"
              % (name, fit["slope"], fit["intercept"], fit["r2"]))
    print("    geometry predicts slope %.1f and intercept 0 on p_width2"
          % law_cost)
    print("    a pure body effect predicts R2 near 0 and a positive "
          "intercept near the mean saving %.1f us"
          % statistics.fmean(saving))

    # ------------------------------------------------------------- item B
    print("\n## Item B: pb6's tier re-fitted on every launch table (form %s)"
          % args.form)
    tiers = {}
    for variant, make in VARIANTS.items():
        curve = make(base_curve)
        lopo = ({p: make(c) for p, c in base_lopo.items()}
                if base_lopo is not None else None)
        base_cache = {}
        rows = {tier: tier_row(cache, seeds, curve, lopo, receipt,
                               args.windows, tier, base_cache)
                for tier in TIER_GRID_FINE}

        def value(tier, rows=rows):
            row = rows[tier]
            return (row["curve_lopo"] or row["in_sample"])[0]

        by_tier = {tier: value(tier) for tier in TIER_GRID_FINE}
        best = max(by_tier, key=by_tier.get)
        low, high = plateau(by_tier, best)
        tiers[variant] = {
            "by_tier": {"%.2f" % t: {"in_sample": rows[t]["in_sample"],
                                     "curve_lopo": rows[t]["curve_lopo"]}
                        for t in TIER_GRID_FINE},
            "held_out": {"%.2f" % t: by_tier[t] for t in TIER_GRID_FINE},
            "best_tier": best, "best": by_tier[best],
            "at_shipped": by_tier[SHIPPED_TIER],
            "gain_over_shipped": by_tier[best] - by_tier[SHIPPED_TIER],
            "plateau": [low, high],
            "sd_at_best": (rows[best]["curve_lopo"]
                           or rows[best]["in_sample"])[1],
        }
        print("  %-14s best tier %.2f -> %+7.4f   at 1.45 -> %+7.4f   "
              "gain %+7.4f pp   plateau %.2f to %.2f   sd %.4f"
              % (variant, best, by_tier[best], by_tier[SHIPPED_TIER],
                 by_tier[best] - by_tier[SHIPPED_TIER], low, high,
                 tiers[variant]["sd_at_best"]))
        print("     held out: %s" % "  ".join(
            "%.2f %+.4f" % (t, by_tier[t]) for t in TIER_GRID_FINE))

    print("\n## Item B, the depth histogram at 1.45 and at the new tier "
          "(form %s)" % args.form)
    depth_rows = {}
    for variant, make in VARIANTS.items():
        curve = make(base_curve)
        for tier in sorted({SHIPPED_TIER, tiers[variant]["best_tier"]}):
            counts, depths = None, []
            for seed in seeds:
                rows = tier_ratios(cache, seed, curve, tier, args.windows)
                for prompt in RANKED_PROMPTS:
                    row = rows[prompt]
                    depths.append(row["mean_depth"])
                    if counts is None:
                        counts = [0] * len(row["depth_counts"])
                    for depth, count in enumerate(row["depth_counts"]):
                        counts[depth] += count
            total = sum(counts) or 1
            share = [c / total for c in counts]
            depth_rows["%s|%.2f" % (variant, tier)] = {
                "mean_depth": statistics.fmean(depths),
                "depth_share": share,
                "width_share": {str(d + 1): s for d, s in enumerate(share)},
            }
            print("  %-14s tier %.2f  mean depth %.4f  width share %s"
                  % (variant, tier, statistics.fmean(depths),
                     " ".join("%d:%.4f" % (d + 1, s)
                              for d, s in enumerate(share) if s)))

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps({
        "harness": "local instrument", "gpu_used": False,
        "form": args.form, "windows": args.windows, "seeds": seeds,
        "receipt": receipt["id"], "law_cost_width2_us": law_cost,
        "medpair": MEDPAIR,
        "item_a": {"histograms": hist,
                   "medpair_p_width2_ship": medpair_ship,
                   "medpair_p_width2_pb6": medpair_pb6},
        "item_d": {"receipt": WIDTH2_RECEIPT, "implied_cost_us": implied,
                   "spread": spread, "shallow_bias_bound": bias, "fits": fits,
                   "mbar_error_mean": statistics.fmean(mbar_err),
                   "mbar_error_sd": statistics.stdev(mbar_err),
                   "rounds_error_mean": statistics.fmean(rounds_err),
                   "rounds_error_sd": statistics.stdev(rounds_err)},
        "item_b": {"tiers": tiers, "depth": depth_rows,
                   "grid": list(TIER_GRID_FINE)},
    }, indent=2, default=list) + "\n")
    print("\nwrote %s" % args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
