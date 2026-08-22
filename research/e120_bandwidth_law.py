#!/usr/bin/env python3
"""Does the E120 chunk-sum gain depend on achieved bandwidth? (E120 F7)

Alphonse's E121 measured the gain of a GATED CROSS-SIMDGROUP SHARE of the same
activation chunk-sums and found it bimodal and ordered by achieved bandwidth,
r = -0.938 over 10 cells drawn from 5 shapes split 2 against 3 with no overlap.
He reads that as arithmetic deletion paying where instruction issue binds and
not where bandwidth binds.

E120 measured the HOIST of the same term over a 7 shape x 7 width grid and
found fractional gain flat from 74 to 246 GB/s.

Both cannot describe the same dependence. This settles it from data already on
disk. It costs no GPU time.

Four questions, in the order the advisor asked them:

  1. Within each width, regress gain percent on achieved GB/s. Seven points per
     width, seven widths. Slope, standard error, 95 % interval.
  2. Pooled over all live cells with a width fixed effect, so width-to-width
     bandwidth differences cannot drive the fit.
  3. Is r = -0.938 reproduced, weakened, or rejected on this grid?
  4. Is a large r even INFORMATIVE at n=5 with a 2-against-3 split? This grid
     is measured flat, so any large correlation drawn from it under alphonse's
     sampling structure is manufactured by that structure. Resampling gives the
     rate directly.

Question 4 is the one that decides between the advisor's hypothesis 1 (the two
mechanisms genuinely differ) and hypothesis 2 (a two-group split at n=5
produces a large correlation with almost any covariate that separates the same
way).

    usage: research/e120_bandwidth_law.py [--additivity PATH] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

# Alphonse's reported correlation, E121 rung 2, 10 live cells.
ALPHONSE_R = -0.938
# The bandwidth window his five shapes spanned.
ALPHONSE_BANDWIDTH_WINDOW = (162.0, 203.0)
# His sampling structure: 5 shapes, split 2 against 3, no overlap in bandwidth.
ALPHONSE_SPLIT = (2, 3)

# `stream.small` and `control.small` are probe controls, not round consumers.
CONTROL_SHAPES = {"stream.small", "control.small"}

RNG_SEED = 20260822


def ols(x: np.ndarray, y: np.ndarray) -> dict:
    """Slope, intercept and the usual textbook standard error."""
    n = len(x)
    if n < 3:
        return {"n": n, "slope": None}
    design = np.column_stack([np.ones(n), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coef
    dof = n - 2
    sigma2 = float(resid @ resid) / dof
    cov = sigma2 * np.linalg.inv(design.T @ design)
    slope_se = float(np.sqrt(cov[1, 1]))
    r = float(np.corrcoef(x, y)[0, 1])
    return {
        "n": n,
        "intercept": float(coef[0]),
        "slope": float(coef[1]),
        "slope_se": slope_se,
        "t": float(coef[1] / slope_se) if slope_se else None,
        "r": r,
        "r2": r * r,
        "residual_sd": float(np.sqrt(sigma2)),
        "dof": dof,
    }


def bootstrap_slope_ci(
    x: np.ndarray, y: np.ndarray, draws: int = 20000, seed: int = RNG_SEED
) -> list[float] | None:
    """Percentile interval. No scipy on this host, and n=7 does not justify a
    normal approximation anyway."""
    n = len(x)
    if n < 3:
        return None
    rng = np.random.default_rng(seed)
    slopes = np.empty(draws)
    for i in range(draws):
        idx = rng.integers(0, n, n)
        xi, yi = x[idx], y[idx]
        if np.ptp(xi) == 0:
            slopes[i] = np.nan
            continue
        slopes[i] = np.polyfit(xi, yi, 1)[0]
    slopes = slopes[np.isfinite(slopes)]
    return [float(np.percentile(slopes, 2.5)), float(np.percentile(slopes, 97.5))]


def permutation_p(x: np.ndarray, y: np.ndarray, draws: int = 20000) -> float | None:
    """Two-sided p for |r|, shuffling the covariate."""
    n = len(x)
    if n < 3:
        return None
    observed = abs(np.corrcoef(x, y)[0, 1])
    rng = np.random.default_rng(RNG_SEED)
    hits = 0
    for _ in range(draws):
        if abs(np.corrcoef(rng.permutation(x), y)[0, 1]) >= observed:
            hits += 1
    return (hits + 1) / (draws + 1)


def alphonse_structure_null(
    bandwidth: np.ndarray,
    gain: np.ndarray,
    draws: int = 20000,
    seed: int = RNG_SEED,
) -> dict:
    """How often does THIS grid, which is measured flat, produce |r| >= 0.938
    when sampled the way alphonse sampled?

    His structure is five shapes in two bandwidth groups with no overlap,
    two against three. Reproduce exactly that: draw 5 cells, sort by
    bandwidth, require a clean 2/3 separation, then correlate.

    A high hit rate means r = -0.938 is what that sampling structure returns
    from flat data, and is therefore not evidence of a bandwidth law.
    """
    rng = np.random.default_rng(seed)
    n = len(bandwidth)
    kept = 0
    hits = 0
    magnitudes = []
    while kept < draws:
        idx = rng.choice(n, 5, replace=False)
        b, g = bandwidth[idx], gain[idx]
        order = np.argsort(b)
        b, g = b[order], g[order]
        # A clean 2-against-3 split: the gap between cell 2 and cell 3 must be
        # the largest gap, which is what "no overlap at either width" means.
        gaps = np.diff(b)
        if np.argmax(gaps) != ALPHONSE_SPLIT[0] - 1:
            continue
        if np.ptp(b) == 0 or np.ptp(g) == 0:
            continue
        kept += 1
        r = abs(np.corrcoef(b, g)[0, 1])
        magnitudes.append(r)
        if r >= abs(ALPHONSE_R):
            hits += 1
    magnitudes = np.array(magnitudes)
    return {
        "draws": kept,
        "structure": "5 cells, clean 2-against-3 bandwidth split, as E121",
        "p_abs_r_at_least_alphonse": hits / kept,
        "median_abs_r": float(np.median(magnitudes)),
        "p95_abs_r": float(np.percentile(magnitudes, 95)),
        "mean_abs_r": float(np.mean(magnitudes)),
    }


def pooled_with_width_fixed_effect(
    cells: list[dict], key: str, shape_fixed_effect: bool = False
) -> dict:
    """gain ~ bandwidth + width dummies, optionally + shape dummies.

    The shape fixed effect is the honest test and it cuts both ways. Achieved
    bandwidth is largely a property of the shape, so a bandwidth slope that
    survives only a width control may be shape identity wearing a bandwidth
    costume -- which is exactly the objection raised against E121's n=5 split.
    Applying it to E120's own grid keeps the criticism symmetric.
    """
    widths = sorted({c["m"] for c in cells})
    shapes = sorted({c["shape"] for c in cells})
    rows = []
    for cell in cells:
        row = [1.0, cell["gb_per_s"]]
        row += [1.0 if cell["m"] == w else 0.0 for w in widths[1:]]
        if shape_fixed_effect:
            row += [1.0 if cell["shape"] == s else 0.0 for s in shapes[1:]]
        rows.append(row)
    design = np.array(rows)
    y = np.array([c[key] for c in cells])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coef
    dof = len(y) - int(np.linalg.matrix_rank(design))
    sigma2 = float(resid @ resid) / dof
    cov = sigma2 * np.linalg.pinv(design.T @ design)
    slope, slope_se = float(coef[1]), float(np.sqrt(cov[1, 1]))
    return {
        "n": len(y),
        "widths": widths,
        "shape_fixed_effect": shape_fixed_effect,
        "slope_pct_per_gbps": slope,
        "slope_se": slope_se,
        "t": slope / slope_se if slope_se else None,
        # n is large enough here that 1.96 is honest.
        "ci95": [slope - 1.96 * slope_se, slope + 1.96 * slope_se],
        "dof": dof,
        "residual_sd": float(np.sqrt(sigma2)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--additivity",
        type=pathlib.Path,
        default=pathlib.Path("research/out/e120-additivity.json"),
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path("research/out/e120-bandwidth-law.json"),
    )
    args = parser.parse_args()

    raw = json.load(args.additivity.open())["cells"]
    # PRIMARY MEASURE is the consumer gain, `gain_us / base_us`. That is the
    # kernel-level effect of not recomputing the term, which is the quantity
    # alphonse's share arm also reports. `net_us` subtracts a standalone fill
    # dispatch that his mechanism does not pay, so comparing net against his
    # gain would compare two different things.
    cells = []
    for c in raw:
        if c["shape"] in CONTROL_SHAPES or not c["base_us"]:
            continue
        cells.append(
            {
                "shape": c["shape"],
                "m": c["m"],
                "gb_per_s": c["gb_per_s"],
                "gain_pct": 100.0 * c["gain_us"] / c["base_us"],
                "net_pct": 100.0 * c["net_us"] / c["base_us"],
            }
        )

    result: dict = {
        "question": (
            "Does the E120 chunk-sum hoist gain depend on achieved bandwidth, "
            "as E121 r=-0.938 implies?"
        ),
        "harness": "local",
        "primary_measure": "gain_pct = 100 * gain_us / base_us (consumer gain)",
        "primary_measure_note": (
            "matches what E121's share arm reports; net_pct subtracts a "
            "standalone fill dispatch that the share arm never pays"
        ),
        "cells": len(cells),
        "control_shapes_excluded": sorted(CONTROL_SHAPES),
        "alphonse_r": ALPHONSE_R,
        "gpu_seconds_used": 0.0,
        "instrument": "regression over existing 5d tables, no measurement",
    }

    # --- 1. within each width -------------------------------------------------
    per_width = {}
    for width in sorted({c["m"] for c in cells}):
        group = [c for c in cells if c["m"] == width]
        x = np.array([c["gb_per_s"] for c in group])
        for key in ("gain_pct", "net_pct"):
            y = np.array([c[key] for c in group])
            fit = ols(x, y)
            fit["ci95_bootstrap"] = bootstrap_slope_ci(x, y)
            fit["permutation_p"] = permutation_p(x, y)
            fit["bandwidth_span"] = [float(x.min()), float(x.max())]
            fit["gain_span"] = [float(y.min()), float(y.max())]
            per_width.setdefault(key, {})[width] = fit
    result["per_width"] = per_width

    # --- 2. pooled, four specifications --------------------------------------
    # SHIPPED is the decision-relevant subset: `tablePays(m:) = m >= 4`, so the
    # gate never routes M=3 and M=3 bandwidth behaviour cannot reach the score.
    shipped = [c for c in cells if c["m"] >= 4]
    result["pooled_width_fixed_effect"] = {
        key: pooled_with_width_fixed_effect(cells, key)
        for key in ("gain_pct", "net_pct")
    }
    result["pooled_specifications"] = {
        "all_widths_width_fe": pooled_with_width_fixed_effect(cells, "gain_pct"),
        "all_widths_width_and_shape_fe": pooled_with_width_fixed_effect(
            cells, "gain_pct", shape_fixed_effect=True
        ),
        "shipped_m4plus_width_fe": pooled_with_width_fixed_effect(
            shipped, "gain_pct"
        ),
        "shipped_m4plus_width_and_shape_fe": pooled_with_width_fixed_effect(
            shipped, "gain_pct", shape_fixed_effect=True
        ),
    }

    # --- 3. the naive pooled correlation, directly comparable to E121 --------
    bandwidth = np.array([c["gb_per_s"] for c in cells])
    for key in ("gain_pct", "net_pct"):
        values = np.array([c[key] for c in cells])
        result.setdefault("pooled_naive", {})[key] = {
            "r": float(np.corrcoef(bandwidth, values)[0, 1]),
            "n": len(values),
            "note": (
                "no width control; width and bandwidth are strongly "
                "confounded in this grid, so this is reported only because it "
                "is the statistic E121 reports"
            ),
        }

    # --- 3b. restricted to alphonse's own bandwidth window -------------------
    low, high = ALPHONSE_BANDWIDTH_WINDOW
    window = [c for c in cells if low <= c["gb_per_s"] <= high]
    if len(window) >= 3:
        x = np.array([c["gb_per_s"] for c in window])
        y = np.array([c["gain_pct"] for c in window])
        fit = ols(x, y)
        fit["ci95_bootstrap"] = bootstrap_slope_ci(x, y)
        fit["permutation_p"] = permutation_p(x, y)
        fit["window"] = [low, high]
        fit["shapes"] = sorted({c["shape"] for c in window})
        result["alphonse_window"] = fit

    # --- 4. is a large r informative at n=5 with a 2-against-3 split? --------
    result["alphonse_structure_null"] = alphonse_structure_null(
        bandwidth, np.array([c["gain_pct"] for c in cells])
    )

    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    # --- report ---------------------------------------------------------------
    print(f"live cells                {len(cells)} (7 shapes x 7 widths)")
    print(f"bandwidth span            "
          f"{bandwidth.min():.0f} to {bandwidth.max():.0f} GB/s")
    print()
    print("1. WITHIN EACH WIDTH   gain_pct ~ achieved GB/s")
    print(f"   {'M':>2} {'n':>2} {'GB/s span':>12} {'gain span':>12} "
          f"{'slope':>10} {'se':>8} {'95% CI (bootstrap)':>24} {'r':>7} {'perm p':>7}")
    for width, fit in sorted(per_width["gain_pct"].items()):
        ci = fit.get("ci95_bootstrap")
        ci_text = f"[{ci[0]:+.4f}, {ci[1]:+.4f}]" if ci else "n/a"
        print(
            f"   {width:>2} {fit['n']:>2} "
            f"{fit['bandwidth_span'][0]:>5.0f}-{fit['bandwidth_span'][1]:<6.0f} "
            f"{fit['gain_span'][0]:>5.2f}-{fit['gain_span'][1]:<6.2f} "
            f"{fit['slope']:>+10.5f} {fit['slope_se']:>8.5f} {ci_text:>24} "
            f"{fit['r']:>+7.3f} {fit['permutation_p']:>7.3f}"
        )
    print()
    print("2. POOLED, FOUR SPECIFICATIONS   gain_pct ~ GB/s + fixed effects")
    print(f"   {'specification':<38} {'n':>3} {'slope':>10} {'se':>8} "
          f"{'t':>6} {'95% CI':>22}")
    for label, fit in result["pooled_specifications"].items():
        print(
            f"   {label:<38} {fit['n']:>3} "
            f"{fit['slope_pct_per_gbps']:>+10.6f} {fit['slope_se']:>8.6f} "
            f"{fit['t']:>+6.2f} "
            f"[{fit['ci95'][0]:+.6f}, {fit['ci95'][1]:+.6f}]"
        )
    print("   The shipped gate routes only M >= 4, so the two `shipped` rows")
    print("   are the only ones that can reach the score.")
    print()
    print("3. NAIVE POOLED r, the statistic E121 reports")
    print(f"   E120 r = {result['pooled_naive']['gain_pct']['r']:+.3f} "
          f"over {len(cells)} cells,  E121 r = {ALPHONSE_R:+.3f} over 10 cells")
    if "alphonse_window" in result:
        win = result["alphonse_window"]
        print(f"   restricted to his {low:.0f}-{high:.0f} GB/s window: "
              f"n={win['n']} r={win['r']:+.3f} slope={win['slope']:+.5f} "
              f"perm p={win['permutation_p']:.3f}")
    print()
    null = result["alphonse_structure_null"]
    print("4. IS |r| >= 0.938 INFORMATIVE AT n=5 WITH A 2-AGAINST-3 SPLIT?")
    print(f"   Drawing that structure from THIS measured-flat grid:")
    print(f"   P(|r| >= 0.938) = {null['p_abs_r_at_least_alphonse']:.3f}   "
          f"median |r| = {null['median_abs_r']:.3f}   "
          f"95th pct |r| = {null['p95_abs_r']:.3f}   "
          f"({null['draws']} draws)")
    print()
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
