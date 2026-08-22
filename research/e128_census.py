#!/usr/bin/env python3
"""E128-F6 - test the advisor's NA census against our fitted ranked curve.

harness=ranked. Zero GPU.

The advisor built an independent model of the extra-pass cost by classifying
each work class in `qwen_e120_qmv_wide<NA, USE_TABLE>` by whether it scales
with NA. A second pass repeats the NA-independent part `F` in full and none
of the NA-scaling part, so the extra pass costs a CONSTANT and its share of
the round must SHRINK as M grows. Our two-segment fit put the tier in the
slope, so its excess GROWS with M. The two models disagree in a testable
direction.

Three questions are answered here.

1. Does a pooled fit with an explicit pass term recover the census constant?
   The model is

       T(M) = c + f * (G(M) - 1) + b * M + k * (G(M) - 1) * M

   with `G(M)` read off the shipped dispatch table, not from `ceil(M / 4)`.
   `f` is the flat per-pass cost the census predicts and `k` is the per-row
   tier our earlier fit preferred. Both are constrained non-negative, because
   a negative per-pass cost is not physical and the unconstrained fit wants
   exactly that. The eight observations are the raw receipt-derived per-prompt
   round costs, taken in width-histogram expectation, so nothing here is
   fitted to another fit.

2. What is the census excess profile against ours, in the same units?

3. What does the arm price become with the real width histograms instead of
   the advisor's two-point approximation?
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from e128_ourcurve import (
    F83_WEIGHT,
    MAX_ROWS,
    build_points,
    fixture_histograms,
    load_receipt,
    prompt_probs,
    r_scenarios,
)
from e128_slopes import ROWS, design, passes_with_boundary

FIRST_TWO_PASS = 6

# The advisor's census, taken at NA=4, in census units.
CENSUS_F = 30.486          # NA-independent, duplicated by a second pass
CENSUS_V = 15.833          # per NA lane, not duplicated

# Pass count per width, read off the dispatch table thorfinn located at
# `Qwen35.swift:1565`: one pass at M = 3,4,5, two at M = 6,7,8, three at
# M = 9. This is NOT `ceil(M/4)`, which would put the boundary at M = 5.
SOURCE_PASSES = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3}

# The advisor's published census excess, share of the one-pass cost.
CENSUS_EXCESS_PCT = {6: 24.29, 7: 21.57, 8: 19.40, 9: 35.25}
# The advisor's published census saving, share of the two-pass cost.
CENSUS_SAVING_FRAC = {6: 0.1955, 7: 0.1774, 8: 0.1625, 9: 0.2606}

QMV_SHARE_OF_LEG = 0.8735  # E105 0.9443 x verify share 0.925


def passes(m: int) -> int:
    return SOURCE_PASSES[int(m)]


def census_round_units(m: int) -> float:
    """Census cost of a width-M round body, in census units."""
    return passes(m) * CENSUS_F + CENSUS_V * m


def solve_at_f(cols: np.ndarray, x: np.ndarray, y: np.ndarray, f: float,
               grid: int = 1201) -> dict:
    """Best `(c, b, k)` with `k >= 0` for a fixed flat pass cost `f`.

    `x` holds the histogram expectation of each per-width regressor column,
    so the non-integer per-prompt mean width never has to be rounded.
    """
    best = None
    k_hi = float(y.max() - y.min()) / 3.0
    sub = x[:, [0, 2]]
    for k in np.linspace(0.0, k_hi, grid):
        target = y - f * x[:, 1] - k * x[:, 3]
        beta, *_ = np.linalg.lstsq(sub, target, rcond=None)
        rss = float(((target - sub @ beta) ** 2).sum())
        if best is None or rss < best["rss"]:
            best = {"c": float(beta[0]), "b": float(beta[1]),
                    "f": float(f), "k": float(k), "rss": rss}
    best["rmse_us"] = math.sqrt(best["rss"] / len(y))
    return best


def profile_f(cols: np.ndarray, x: np.ndarray, y: np.ndarray, f_max: float,
              nodes: int = 121) -> list:
    """Profile the residual sum of squares over the flat pass cost `f`."""
    return [solve_at_f(cols, x, y, f) for f in np.linspace(0.0, f_max, nodes)]


def curve_of(fit: dict):
    return lambda m: (fit["c"] + fit["f"] * (passes(m) - 1)
                      + fit["b"] * m + fit["k"] * (passes(m) - 1) * m)


def median_of(values) -> float:
    ordered = sorted(values)
    return 0.5 * (ordered[3] + ordered[4])


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--board", type=Path,
                        default=Path("/tmp/yukon-board/full.json"))
    parser.add_argument("--identity", type=Path,
                        default=here / "e128-artifacts/rung0-identity.json")
    parser.add_argument("--shipped", type=Path,
                        default=here / "e128-artifacts/rung1-shipped.json")
    parser.add_argument("--curves", type=Path,
                        default=here / "e128-artifacts/"
                                       "f4-candidate-curves.json")
    parser.add_argument("--receipt", default="d3c491b5")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    curves = json.loads(args.curves.read_text())["curves"]

    def seg(curve, m):
        part = curve["lo"] if m < curve["breakpoint"] else curve["hi"]
        return part[0] + part[1] * m

    hists = fixture_histograms(args.shipped)
    scen = r_scenarios(args.identity)
    receipt = load_receipt(args.board, args.receipt)
    point_list = build_points(receipt, scen["assumed"], hists)
    points = {p["prompt"]: p for p in point_list}
    for prompt, point in points.items():
        point["raw"] = receipt["per_prompt"][prompt]["raw"]

    print("harness=ranked  E128-F6 - the NA census against our fitted curve")
    print("pass count from the dispatch table: %s"
          % {m: SOURCE_PASSES[m] for m in range(1, 10)})
    print("that boundary is M >= 6, NOT the ceil(M/4) boundary at M >= 5\n")

    # Census `f` in microseconds. One census unit is priced from the round
    # the census and our headline admissible curve both describe, so the
    # comparison stays in our units and does not assume the census scale.
    anchor_m = 6
    us_per_unit = (seg(curves["slopeonly_b6"], anchor_m) * QMV_SHARE_OF_LEG
                   / census_round_units(anchor_m))
    f_census_us = CENSUS_F * us_per_unit

    # -- question 1: profile the flat pass term against the data ----------
    # Fit the raw receipt-derived per-prompt round costs, not a fitted curve.
    # Columns are [1, G-1, M, (G-1)M] taken in histogram expectation, so the
    # non-integer per-prompt mean width is never rounded to a lattice width.
    g = passes_with_boundary(FIRST_TWO_PASS)
    cols = np.stack([np.ones_like(ROWS), g - 1.0, ROWS, (g - 1.0) * ROWS],
                    axis=1)
    x = design(point_list, cols)
    y = np.array([p["round_us"] for p in point_list], dtype=float)
    prof = profile_f(cols, x, y, f_max=2.5 * f_census_us)
    best = min(prof, key=lambda r: r["rss"])
    at_census = solve_at_f(cols, x, y, f_census_us)
    n, p = len(y), 4
    # F test of the fixed-f fit against the free-f optimum, 1 vs n - p dof.
    f_stat = ((at_census["rss"] - best["rss"]) / 1.0) / (best["rss"] / (n - p))
    print("## 1. pooled fit  T = c + f*(G-1) + b*M + k*(G-1)*M,  f,k >= 0")
    print("best      f = %8.1f us  k = %7.1f us/row  b = %7.1f  rmse %7.1f"
          % (best["f"], best["k"], best["b"], best["rmse_us"]))
    print("at census f = %8.1f us  k = %7.1f us/row  b = %7.1f  rmse %7.1f"
          % (at_census["f"], at_census["k"], at_census["b"],
             at_census["rmse_us"]))
    print("census unit priced at M=%d: %.2f us/unit  (QMV share %.2f %%)"
          % (anchor_m, us_per_unit, 100 * QMV_SHARE_OF_LEG))
    print("F(1,%d) for holding f at the census value = %.3f" % (n - p, f_stat))
    # The set of f the data cannot reject at the 5 % level, F(1,4) = 7.709.
    crit = 7.709
    ok = [r["f"] for r in prof
          if (r["rss"] - best["rss"]) / (best["rss"] / (n - p)) <= crit]
    print("f values the data do not reject at 5 %%: [%.0f, %.0f] us"
          % (min(ok), max(ok)))
    print("census f = %.1f us is %s that interval"
          % (f_census_us, "INSIDE" if min(ok) <= f_census_us <= max(ok)
             else "OUTSIDE"))
    print("\n%-12s %10s %10s %10s" % ("f (us)", "k (us/row)", "rmse", "dRSS/s2"))
    for r in prof[::12]:
        print("%-12.0f %10.1f %10.1f %10.2f"
              % (r["f"], r["k"], r["rmse_us"],
                 (r["rss"] - best["rss"]) / (best["rss"] / (n - p))))
    fit = at_census

    # -- question 2: excess profiles, same units --------------------------
    print("\n## 2. excess of the two-pass cost over a one-pass counterfactual")
    print("%-4s %10s %10s %10s %10s %10s" % (
        "M", "census %", "pooled %", "affine %", "slopeonly %", "freeslope %"))
    rows = {}
    for m in (6, 7, 8):
        one_pooled = fit["c"] + fit["b"] * m
        pooled = 100.0 * (curve_of(fit)(m) / one_pooled - 1.0)
        aff = curves["affine_b6"]
        one_aff = aff["lo"][0] + aff["lo"][1] * m
        affine = 100.0 * (seg(aff, m) / one_aff - 1.0)
        slo = curves["slopeonly_b6"]
        one_slo = slo["lo"][0] + slo["lo"][1] * m
        slope = 100.0 * (seg(slo, m) / one_slo - 1.0)
        fre = curves["freeslope_b6"]
        one_fre = fre["lo"][0] + fre["lo"][1] * m
        free = 100.0 * (seg(fre, m) / one_fre - 1.0)
        rows[m] = {"census": CENSUS_EXCESS_PCT[m], "pooled": pooled,
                   "affine": affine, "slopeonly": slope, "freeslope": free}
        print("%-4d %10.2f %10.2f %10.2f %10.2f %10.2f"
              % (m, CENSUS_EXCESS_PCT[m], pooled, affine, slope, free))
    for name in ("census", "pooled", "affine", "slopeonly", "freeslope"):
        d = rows[8][name] - rows[6][name]
        print("  %-10s M=8 minus M=6: %+7.2f points  -> %s"
              % (name, d, "SHRINKS" if d < 0 else "GROWS"))

    # -- question 3: the arm price on the real histograms ------------------
    print("\n## 3. census arm price on the real width histograms")
    print("%-10s %7s %8s %11s %11s %11s" % (
        "prompt", "w83", "mean M", "2-point %", "real hist %", "delta"))

    def saving_of(probs: np.ndarray) -> float:
        return float(sum(probs[m - 1] * CENSUS_SAVING_FRAC.get(m, 0.0)
                         for m in range(1, MAX_ROWS + 1)
                         if m - 1 < len(probs)))

    per_prompt, weighted, two_point_wtd = {}, 0.0, 0.0
    for prompt in sorted(points, key=lambda p: -F83_WEIGHT[p]):
        probs = np.array(points[prompt]["hist"]["probs"])
        saving = saving_of(probs)
        lattice = saving_of(prompt_probs(points[prompt], False))
        mean_m = float(sum(probs[i] * (i + 1) for i in range(len(probs))))
        per_prompt[prompt] = {"qmv_saving_frac": saving, "mean_m": mean_m,
                              "two_point_frac": lattice,
                              "f83": F83_WEIGHT[prompt]}
        weighted += F83_WEIGHT[prompt] * saving
        two_point_wtd += F83_WEIGHT[prompt] * lattice
        print("%-10s %7.4f %8.3f %11.2f %11.2f %+11.2f" % (
            prompt, F83_WEIGHT[prompt], mean_m, 100 * lattice, 100 * saving,
            100 * (saving - lattice)))
    print("%-10s %7.4f %8s %11.2f %11.2f %+11.2f" % (
        "F83-WTD", sum(F83_WEIGHT.values()), "-", 100 * two_point_wtd,
        100 * weighted, 100 * (weighted - two_point_wtd)))
    print("advisor published 12.73 % on his own two-point approximation")
    leg = weighted * QMV_SHARE_OF_LEG
    print("leg frame: %.2f %% x %.2f %% QMV share = %.2f %%   "
          "(advisor 11.12 %%)" % (100 * weighted, 100 * QMV_SHARE_OF_LEG,
                                  100 * leg))

    # Rule 67 exact median, which the weighted mean above is not.
    new_raw = {}
    for prompt, point in points.items():
        s = per_prompt[prompt]["qmv_saving_frac"] * QMV_SHARE_OF_LEG
        new_raw[prompt] = point["raw"] / (1.0 - s)
    old_med = median_of([p["raw"] for p in points.values()])
    new_med = median_of(list(new_raw.values()))
    med_pct = 100.0 * (new_med / old_med - 1.0)
    print("Rule 67 exact median: %.8f -> %.8f  = %+.4f %%"
          % (old_med, new_med, med_pct))

    # -- table design: rank by absolute saving times round share ----------
    print("\n## table design: absolute us saved per average round")
    print("%-4s %9s %14s %14s %12s %12s" % (
        "M", "F83 share", "census us", "slopeonly us", "census wtd",
        "slopeonly wtd"))
    share = {m: sum(F83_WEIGHT[p]
                    * np.array(points[p]["hist"]["probs"])[m - 1]
                    for p in points) for m in (6, 7, 8)}
    slo = curves["slopeonly_b6"]
    table_design = {}
    for m in (6, 7, 8):
        c_us = f_census_us * (passes(m) - 1)
        s_us = seg(slo, m) - (slo["lo"][0] + slo["lo"][1] * m)
        table_design[m] = {"f83_share": share[m], "census_us": c_us,
                     "slopeonly_us": s_us,
                     "census_weighted": share[m] * c_us,
                     "slopeonly_weighted": share[m] * s_us}
        print("%-4d %9.4f %14.1f %14.1f %12.1f %12.1f"
              % (m, share[m], c_us, s_us, share[m] * c_us, share[m] * s_us))
    order_c = sorted((6, 7, 8),
                     key=lambda m: -table_design[m]["census_weighted"])
    order_s = sorted((6, 7, 8),
                     key=lambda m: -table_design[m]["slopeonly_weighted"])
    print("ranked build order, census profile:    %s" % (order_c,))
    print("ranked build order, slopeonly profile: %s" % (order_s,))

    # -- M=9 reachability --------------------------------------------------
    m9 = {p: (float(np.array(points[p]["hist"]["probs"])[8])
              if len(points[p]["hist"]["probs"]) > 8 else 0.0)
          for p in points}
    print("\n## M=9 mass in the ranked histograms: %.6f (max over prompts)"
          % max(m9.values()))

    if args.json:
        args.json.write_text(json.dumps({
            "harness": "ranked",
            "pooled_fit": fit, "pooled_best": best,
            "pooled_profile": [{k: v for k, v in r.items()} for r in prof],
            "census": {"F": CENSUS_F, "V": CENSUS_V,
                       "first_two_pass": FIRST_TWO_PASS,
                       "us_per_unit": us_per_unit, "f_us": f_census_us,
                       "excess_pct": CENSUS_EXCESS_PCT,
                       "saving_frac": CENSUS_SAVING_FRAC},
            "excess_profiles": rows,
            "arm_price": {"per_prompt": per_prompt,
                          "f83_weighted_qmv": weighted,
                          "leg_frame": leg,
                          "rule67_median_pct": med_pct},
            "table_design": table_design,
            "m9_mass": m9,
        }, indent=2) + "\n")
        print("\nwrote %s" % args.json)


if __name__ == "__main__":
    main()
