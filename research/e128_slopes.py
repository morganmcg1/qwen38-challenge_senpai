#!/usr/bin/env python3
"""E128-F4 section 3c - is the tier-2 slope identified, or is the fit absorbing?

harness=ranked. Zero GPU.

The advisor's pass-count model is `T(M, G) = G * a + b * M`: the number of
passes over the weight matrix sets the intercept, the FMA count is
proportional to `M` regardless of grouping, so the slope inside a segment must
be the same in both segments. Our fit reports 3388.3 below the break and
6167.5 above it, a ratio of 1.82, and the board curve reports the same shape.

This script answers three separate questions and keeps them separate.

1. Are the two slopes identified at all? Leave-one-prompt-out on the SLOPES,
   analytic standard errors from the residual variance, and the condition
   number of the design matrix. If the slopes are not identified, the anomaly
   is a fitting artefact and nothing may be priced from the slopes.

2. Does an equal-slope model actually fit worse? The pass-count model is
   nested inside the free-slope model, so an exact F test on eight points is
   available. The comparison uses the CORRECTED post-E100 grouping
   `G = 1 for M <= 5, G = 2 for M in 6..8, G = 3 for M = 9`, not F97's
   pre-E100 labels.

3. If the equal-slope model is rejected, what shape replaces it? A pass that
   carries its own per-row work gives `T = c + G * (a + b * M)`, whose slope
   ratio is exactly `G` and needs no free slope parameter at all.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from e128_ourcurve import (
    BOARD_CURVE,
    MAX_ROWS,
    build_points,
    curve_us,
    fit,
    fixture_histograms,
    load_receipt,
    prompt_probs,
    r_scenarios,
)

OUR_BREAK = 6
ROWS = np.arange(0, MAX_ROWS, dtype=float) + 1.0

# Post-E100 dispatch grouping. E100 ships `<T,5,5>`, so one pass covers M <= 5.
# F97 labelled G=1 for M=1..4 and G=2 for M=5..8, which is the pre-E100 table.
PASSES_POST_E100 = np.array([1, 1, 1, 1, 1, 2, 2, 2, 3], dtype=float)
PASSES_PRE_E100 = np.array([1, 1, 1, 1, 2, 2, 2, 2, 3], dtype=float)


# ------------------------------------------------------------------- models

def basis(name: str, passes: np.ndarray) -> callable:
    """Per-width regressor rows for one model, evaluated at M = 1..MAX_ROWS.

    Every model here is linear in its parameters, so the expected cost over a
    per-prompt width histogram is the histogram expectation of these rows.
    """
    g = passes
    m = ROWS
    if name == "line":  # T = a + b M
        return np.stack([np.ones_like(m), m], axis=1)
    if name == "passcount":  # T = G a + b M   (the advisor's model)
        return np.stack([g, m], axis=1)
    if name == "passcount_affine":  # T = c + G a + b M
        return np.stack([np.ones_like(m), g, m], axis=1)
    if name == "passcount_scaled":  # T = G (a + b M), slope ratio exactly G
        return np.stack([g, g * m], axis=1)
    if name == "passcount_scaled_affine":  # T = c + G (a + b M)
        return np.stack([np.ones_like(m), g, g * m], axis=1)
    if name == "passcount_freeslope":  # T = c + G a + b M + db (G - 1) M
        return np.stack([np.ones_like(m), g, m, (g - 1.0) * m], axis=1)
    if name == "passcount_slopeonly":  # T = c + b M + k (G - 1) M
        return np.stack([np.ones_like(m), m, (g - 1.0) * m], axis=1)
    if name == "quadratic":  # T = a + b M + c M^2, no tier at all
        return np.stack([np.ones_like(m), m, m * m], axis=1)
    raise SystemExit("unknown model %r" % name)


def design(points: list[dict], cols: np.ndarray) -> np.ndarray:
    return np.array([
        np.array(prompt_probs(p, True)) @ cols for p in points])


def ols(a: np.ndarray, y: np.ndarray) -> dict:
    beta, *_ = np.linalg.lstsq(a, y, rcond=None)
    resid = y - a @ beta
    rss = float(resid @ resid)
    n, k = len(y), a.shape[1]
    dof = n - k
    aicc = n * math.log(max(rss, 1e-12) / n) + 2 * k
    aicc += 2 * k * (k + 1) / (n - k - 1) if n - k - 1 > 0 else float("inf")
    out = {"beta": beta, "rss": rss, "rmse": math.sqrt(rss / n), "aicc": aicc,
           "params": k, "dof": dof,
           "cond": float(np.linalg.cond(a)),
           "residuals": resid}
    if dof > 0:
        sigma2 = rss / dof
        cov = sigma2 * np.linalg.pinv(a.T @ a)
        out["se"] = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        out["cov"] = cov
    return out


def f_test(restricted: dict, full: dict) -> dict:
    """Exact F statistic for a nested restriction, with its p value.

    scipy is not available here, so the tail probability of F(q, dof) is
    evaluated through the regularised incomplete beta function, which numpy
    does not ship either. `betainc` is computed by its continued fraction.
    """
    q = full["params"] - restricted["params"]
    dof = full["dof"]
    if q <= 0 or dof <= 0 or full["rss"] <= 0:
        return {"f": None, "p": None, "q": q, "dof": dof}
    f = ((restricted["rss"] - full["rss"]) / q) / (full["rss"] / dof)
    x = dof / (dof + q * f)
    p = betainc(dof / 2.0, q / 2.0, x)
    return {"f": float(f), "p": float(p), "q": q, "dof": dof}


def betainc(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log1p(-x) - lbeta)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * betacf(a, b, x) / a
    return 1.0 - math.exp(b * math.log1p(-x) + a * math.log(x) - lbeta) \
        * betacf(b, a, 1.0 - x) / b


def betacf(a: float, b: float, x: float, iters: int = 300) -> float:
    tiny = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (tiny if abs(d) < tiny else d)
    h = d
    for m in range(1, iters + 1):
        m2 = 2 * m
        num = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + num * d
        c = 1.0 + num / c
        d = 1.0 / (tiny if abs(d) < tiny else d)
        c = tiny if abs(c) < tiny else c
        h *= d * c
        num = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + num * d
        c = 1.0 + num / c
        d = 1.0 / (tiny if abs(d) < tiny else d)
        c = tiny if abs(c) < tiny else c
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-12:
            break
    return h


# ----------------------------------------------------- question 1: slopes

def slope_loo(points: list[dict], breakpoint: int) -> dict:
    """Leave-one-prompt-out on the two slopes at a FIXED tier break.

    The break is held fixed on purpose. F3 already showed the break is robust
    under leave-one-out; the open question is whether the slopes are, and
    re-selecting the break on each drop would confound the two.
    """
    full = fit(points, breakpoint, "piece", True)
    rows = []
    for drop in range(len(points)):
        subset = [p for i, p in enumerate(points) if i != drop]
        got = fit(subset, breakpoint, "piece", True)
        rows.append({
            "dropped": points[drop]["prompt"],
            "b_lo": got["lo"][1], "b_hi": got["hi"][1],
            "ratio": got["hi"][1] / got["lo"][1] if got["lo"][1] else None,
            "jump_us": got["jump_us"], "rmse": got["rmse"],
        })
    lo = np.array([r["b_lo"] for r in rows])
    hi = np.array([r["b_hi"] for r in rows])
    ratio = np.array([r["ratio"] for r in rows])
    return {
        "breakpoint": breakpoint,
        "full": {"b_lo": full["lo"][1], "b_hi": full["hi"][1],
                 "ratio": full["hi"][1] / full["lo"][1],
                 "rmse": full["rmse"]},
        "rows": rows,
        "spread": {
            "b_lo": [float(lo.min()), float(lo.max()),
                     float(lo.max() - lo.min())],
            "b_hi": [float(hi.min()), float(hi.max()),
                     float(hi.max() - hi.min())],
            "ratio": [float(ratio.min()), float(ratio.max())],
            "b_lo_rel_range": float((lo.max() - lo.min()) / lo.mean()),
            "b_hi_rel_range": float((hi.max() - hi.min()) / hi.mean()),
            "ratio_crosses_one": bool(ratio.min() <= 1.0 <= ratio.max()),
        },
    }


def slope_errors(points: list[dict], breakpoint: int) -> dict:
    """Analytic standard errors on the two slopes of the unconstrained fit.

    The constrained fit used for the campaign curve has all sign constraints
    inactive at the selected break, so the unconstrained covariance applies.
    """
    cols = np.stack([
        np.ones_like(ROWS), ROWS,
        (ROWS >= breakpoint).astype(float),
        np.where(ROWS >= breakpoint, ROWS - breakpoint, 0.0)], axis=1)
    a = design(points, cols)
    y = np.array([p["round_us"] for p in points])
    got = ols(a, y)
    beta, se = got["beta"], got["se"]
    b_lo, db = float(beta[1]), float(beta[3])
    se_lo, se_db = float(se[1]), float(se[3])
    cov = got["cov"]
    var_hi = cov[1, 1] + 2 * cov[1, 3] + cov[3, 3]
    se_hi = math.sqrt(max(var_hi, 0.0))
    # ratio = 1 + db / b_lo; the delta method needs cov(b_lo, db).
    ratio = 1.0 + db / b_lo
    grad = np.array([-db / b_lo ** 2, 1.0 / b_lo])
    sub = np.array([[cov[1, 1], cov[1, 3]], [cov[3, 1], cov[3, 3]]])
    se_ratio = math.sqrt(max(float(grad @ sub @ grad), 0.0))
    tcrit = 2.776  # two-sided 95 % at 4 degrees of freedom
    return {
        "breakpoint": breakpoint, "cond": got["cond"], "dof": got["dof"],
        "rmse": got["rmse"],
        "b_lo": b_lo, "se_b_lo": se_lo,
        "b_hi": b_lo + db, "se_b_hi": se_hi,
        "dslope": db, "se_dslope": se_db,
        "dslope_t": db / se_db if se_db else None,
        "dslope_ci95": [db - tcrit * se_db, db + tcrit * se_db],
        "dslope_ci95_excludes_zero": bool(abs(db) > tcrit * se_db),
        "ratio": ratio, "se_ratio": se_ratio,
        "ratio_ci95": [ratio - tcrit * se_ratio, ratio + tcrit * se_ratio],
        "ratio_ci95_excludes_one": bool(abs(ratio - 1.0) > tcrit * se_ratio),
        "ratio_ci95_contains_two": bool(
            abs(ratio - 2.0) <= tcrit * se_ratio),
    }


def leverage(points: list[dict], breakpoint: int) -> list:
    """Which prompt actually carries each slope."""
    cols = np.stack([
        np.ones_like(ROWS), ROWS,
        (ROWS >= breakpoint).astype(float),
        np.where(ROWS >= breakpoint, ROWS - breakpoint, 0.0)], axis=1)
    a = design(points, cols)
    hat = a @ np.linalg.pinv(a.T @ a) @ a.T
    out = []
    for i, point in enumerate(points):
        probs = np.array(point["hist"]["probs"])
        out.append({
            "prompt": point["prompt"], "mbar": point["mbar"],
            "hat": float(hat[i, i]),
            "mass_above_break": float(probs[ROWS >= breakpoint].sum()),
            "mean_M_above_break": float(
                (probs[ROWS >= breakpoint] * ROWS[ROWS >= breakpoint]).sum()
                / max(probs[ROWS >= breakpoint].sum(), 1e-12)),
        })
    return out


# ------------------------------------------- question 2 and 3: model shapes

def model_table(points: list[dict], passes: np.ndarray) -> list:
    y = np.array([p["round_us"] for p in points])
    names = ["line", "passcount", "passcount_affine", "passcount_scaled",
             "passcount_scaled_affine", "passcount_freeslope", "quadratic"]
    out = []
    for name in names:
        cols = basis(name, passes)
        a = design(points, cols)
        got = ols(a, y)
        entry = {"model": name, "params": got["params"], "dof": got["dof"],
                 "rss": got["rss"], "rmse_us": got["rmse"],
                 "aicc": got["aicc"], "cond": got["cond"],
                 "beta": [float(v) for v in got["beta"]]}
        entry["curve"] = [float(row @ got["beta"]) for row in cols]
        entry["implied_slope_ratio"] = implied_ratio(name, got["beta"], passes)
        out.append(entry)
    out.sort(key=lambda row: row["aicc"])
    return out


def passes_with_boundary(first_two_pass: int) -> np.ndarray:
    """Pass count when the one-pass tile ends just below `first_two_pass`.

    M=9 always needs a third pass under any tile width of 4 or 5, so it is
    held at G=3 and only the G=1 / G=2 boundary moves.
    """
    g = np.where(ROWS >= first_two_pass, 2.0, 1.0)
    g[ROWS >= 9] = 3.0
    return g


def boundary_sweep(points: list[dict]) -> list:
    """Where does each model family put the pass boundary?

    The tier break was selected inside one model family in F3. If the
    boundary moves when the family changes, then the break is a property of
    the family and not only of the hardware, and it must be reported as such.
    """
    y = np.array([p["round_us"] for p in points])
    out = []
    for name in ("passcount_affine", "passcount_scaled_affine",
                 "passcount_freeslope"):
        for boundary in range(3, 9):
            cols = basis(name, passes_with_boundary(boundary))
            got = ols(design(points, cols), y)
            out.append({"model": name, "boundary_M": boundary,
                        "params": got["params"], "rmse_us": got["rmse"],
                        "aicc": got["aicc"]})
    return out


TIER_FAMILIES = ("passcount_affine", "passcount_slopeonly",
                 "passcount_scaled_affine", "passcount_freeslope")


def unpack(name: str, beta: np.ndarray) -> dict:
    """Read `c` fixed, `a` per-pass overhead, `b` per-row, `k` per-row-per-pass.

    Every tier family is a restriction of `T = c + a (G-1) + b M + k (G-1) M`,
    so all four are reported in the same units and can be compared directly.
    `a` is a fixed cost paid for the extra pass and cannot physically be
    negative; `k` is extra per-row work that only the multi-pass regime pays.
    """
    if name == "passcount_affine":  # T = c + G a + b M
        return {"c": float(beta[0] + beta[1]), "a": float(beta[1]),
                "b": float(beta[2]), "k": 0.0}
    if name == "passcount_slopeonly":  # T = c + b M + k (G-1) M
        return {"c": float(beta[0]), "a": 0.0,
                "b": float(beta[1]), "k": float(beta[2])}
    if name == "passcount_scaled_affine":  # T = c + G (a + b M)
        return {"c": float(beta[0] + beta[1]), "a": float(beta[1]),
                "b": float(beta[2]), "k": float(beta[2])}
    if name == "passcount_freeslope":  # T = c + G a + b M + db (G-1) M
        return {"c": float(beta[0] + beta[1]), "a": float(beta[1]),
                "b": float(beta[2]), "k": float(beta[3])}
    raise SystemExit("cannot unpack %r" % name)


def tier_decomposition(points: list[dict]) -> list:
    """Is the tier a fixed cost, a per-row cost, or both?

    Each candidate is checked for two physical constraints that a least
    squares fit does not enforce: the curve must rise with width, and the
    extra pass must not make the round cheaper at the boundary.
    """
    y = np.array([p["round_us"] for p in points])
    out = []
    for name in TIER_FAMILIES:
        for boundary in range(3, 9):
            passes = passes_with_boundary(boundary)
            cols = basis(name, passes)
            got = ols(design(points, cols), y)
            vals = cols @ got["beta"]
            single = basis(name, np.ones_like(passes))
            step = float((cols[boundary - 1] - single[boundary - 1])
                         @ got["beta"])
            terms = unpack(name, got["beta"])
            out.append({
                "name": "%s@M>=%d" % (name, boundary),
                "family": name, "boundary_M": boundary,
                "params": got["params"], "rmse_us": got["rmse"],
                "aicc": got["aicc"], **terms,
                "tier_step_us": step,
                "monotone": bool(np.all(np.diff(vals[:8]) > 0)),
                "pass_overhead_nonnegative": bool(terms["a"] >= 0.0),
                "step_nonnegative": bool(step >= 0.0),
            })
    out.sort(key=lambda row: row["aicc"])
    return out


def candidate_curves(points: list[dict]) -> list:
    """Every (family, boundary) pair within 5 aicc of the winner.

    Each is re-expressed as an ordinary two-segment linear curve over the
    reachable widths M = 1..8, which is the shape `e128_price.py` consumes.
    The shipped cap keeps the draft depth at or below 7, so M never reaches
    the third pass and a two-segment form is exact over the scored range.
    """
    y = np.array([p["round_us"] for p in points])
    grid = []
    for name in ("passcount_affine", "passcount_scaled_affine",
                 "passcount_freeslope"):
        for boundary in range(3, 9):
            passes = passes_with_boundary(boundary)
            cols = basis(name, passes)
            got = ols(design(points, cols), y)
            vals = cols @ got["beta"]
            lo_i = boundary - 2  # index of M = boundary - 1
            slope_lo = float(vals[1] - vals[0])
            slope_hi = float(vals[min(7, len(vals) - 1)] - vals[lo_i + 1]) \
                / max(min(8, MAX_ROWS) - boundary, 1)
            grid.append({
                "name": "%s@M>=%d" % (name, boundary),
                "family": name, "breakpoint": boundary,
                "aicc": got["aicc"], "rmse_us": got["rmse"],
                "params": got["params"],
                "lo": [float(vals[0]) - slope_lo, slope_lo],
                "hi": [float(vals[boundary - 1]) - slope_hi * boundary,
                       slope_hi],
            })
    best = min(row["aicc"] for row in grid)
    keep = [row for row in grid if row["aicc"] - best <= 5.0]
    keep.sort(key=lambda row: row["aicc"])
    return keep


def joint_selection(points: list[dict], drop: int | None = None) -> dict:
    """Best (family, boundary) pair by aicc over the whole grid."""
    subset = points if drop is None else [
        p for i, p in enumerate(points) if i != drop]
    y = np.array([p["round_us"] for p in subset])
    best = None
    for name in ("line", "quadratic", "passcount_affine",
                 "passcount_scaled_affine", "passcount_freeslope"):
        boundaries = [0] if name in ("line", "quadratic") else range(3, 9)
        for boundary in boundaries:
            passes = (PASSES_POST_E100 if boundary == 0
                      else passes_with_boundary(boundary))
            cols = basis(name, passes)
            got = ols(design(subset, cols), y)
            entry = {"model": name, "boundary_M": boundary or None,
                     "aicc": got["aicc"], "rmse_us": got["rmse"],
                     "params": got["params"],
                     "slope_ratio": implied_ratio(name, got["beta"], passes)}
            if best is None or entry["aicc"] < best["aicc"]:
                best = entry
    return best


def implied_ratio(name: str, beta: np.ndarray, passes: np.ndarray) -> float:
    """Slope of the G=2 segment divided by the slope of the G=1 segment."""
    if name in ("line", "passcount", "passcount_affine", "quadratic"):
        return 1.0
    if name in ("passcount_scaled", "passcount_scaled_affine"):
        return 2.0
    if name == "passcount_freeslope":
        return float((beta[2] + beta[3]) / beta[2])
    return float("nan")


def board_pseudo_points(curve: dict, passes: np.ndarray) -> tuple:
    """Fit the same model family to F97's PUBLISHED board curve.

    F97's 147 rows are not available here, so the board curve is refitted at
    its own eight published widths. That cannot recover F97's residuals, but
    it does show whether the published board coefficients are themselves
    consistent with an equal-slope pass-count model.
    """
    y = np.array([curve_us(curve, m) for m in ROWS])
    out = []
    for name in ["line", "passcount", "passcount_affine", "passcount_scaled",
                 "passcount_scaled_affine", "passcount_freeslope",
                 "quadratic"]:
        cols = basis(name, passes)
        got = ols(cols, y)
        out.append({"model": name, "params": got["params"],
                    "rmse_us": got["rmse"], "aicc": got["aicc"],
                    "beta": [float(v) for v in got["beta"]]})
    out.sort(key=lambda row: row["aicc"])
    return y, out


def within_segment_inflation(curve_vals: np.ndarray, passes: np.ndarray,
                             label: str) -> list:
    """F4 section 3d: cost growth inside a fixed pass count is arithmetic only.

    Bytes read are constant inside a segment, so the whole rise from the first
    to the last width of a segment is non-byte work. This replaces the 542.8
    GB/s achieved-rate figure, which was computed when M=5 still read the
    weights twice and is not a ceiling.
    """
    out = []
    for level in sorted(set(passes.tolist())):
        idx = np.where(passes == level)[0]
        if len(idx) < 2:
            continue
        lo_m, hi_m = ROWS[idx[0]], ROWS[idx[-1]]
        lo_v, hi_v = curve_vals[idx[0]], curve_vals[idx[-1]]
        out.append({
            "curve": label, "passes": int(level),
            "m_lo": float(lo_m), "m_hi": float(hi_m),
            "us_lo": float(lo_v), "us_hi": float(hi_v),
            "inflation_pct": 100.0 * (hi_v / lo_v - 1.0),
            "saving_pct_if_arithmetic_free": 100.0 * (1.0 - lo_v / hi_v),
        })
    return out


# ------------------------------------------------------------------- report

def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--board", type=Path,
                        default=Path("/tmp/yukon-board/full.json"))
    parser.add_argument("--identity", type=Path,
                        default=here / "e128-artifacts/rung0-identity.json")
    parser.add_argument("--shipped", type=Path,
                        default=here / "e128-artifacts/rung1-shipped.json")
    parser.add_argument("--receipt", default="d3c491b5")
    parser.add_argument("--scenario", default="assumed")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--curve-json", type=Path,
                        help="write the surviving curves in e128_price shape")
    args = parser.parse_args()

    hists = fixture_histograms(args.shipped)
    scenarios = r_scenarios(args.identity)
    receipt = load_receipt(args.board, args.receipt)
    points = build_points(receipt, scenarios[args.scenario], hists)

    print("harness=ranked  E128-F4 3c - tier-2 slope identifiability\n")
    print("receipt %s  score %.8f  R scenario %s  break M>=%d"
          % (receipt["id"][:8], receipt["score"], args.scenario, OUR_BREAK))

    print("\n## corrected pass counts (post-E100 <T,5,5>)")
    print("M         " + " ".join("%5.0f" % m for m in ROWS))
    print("G post    " + " ".join("%5.0f" % g for g in PASSES_POST_E100))
    print("G pre/F97 " + " ".join("%5.0f" % g for g in PASSES_PRE_E100))

    loo = slope_loo(points, OUR_BREAK)
    print("\n## 1a. leave-one-prompt-out on the SLOPES, break held at M>=%d"
          % OUR_BREAK)
    print("%-10s %10s %10s %8s %9s %8s"
          % ("dropped", "b_lo", "b_hi", "ratio", "jump us", "rmse"))
    for row in loo["rows"]:
        print("%-10s %10.1f %10.1f %8.3f %9.0f %8.0f"
              % (row["dropped"], row["b_lo"], row["b_hi"], row["ratio"],
                 row["jump_us"], row["rmse"]))
    full = loo["full"]
    print("%-10s %10.1f %10.1f %8.3f %9s %8.0f"
          % ("(none)", full["b_lo"], full["b_hi"], full["ratio"], "-",
             full["rmse"]))
    sp = loo["spread"]
    print("b_lo range %.1f to %.1f (%.1f wide, %.1f %% of mean)"
          % (sp["b_lo"][0], sp["b_lo"][1], sp["b_lo"][2],
             100.0 * sp["b_lo_rel_range"]))
    print("b_hi range %.1f to %.1f (%.1f wide, %.1f %% of mean)"
          % (sp["b_hi"][0], sp["b_hi"][1], sp["b_hi"][2],
             100.0 * sp["b_hi_rel_range"]))
    print("ratio range %.3f to %.3f  crosses 1.0: %s"
          % (sp["ratio"][0], sp["ratio"][1], sp["ratio_crosses_one"]))

    err = slope_errors(points, OUR_BREAK)
    print("\n## 1b. analytic standard errors, unconstrained 4-parameter fit")
    print("cond(A) %.1f  dof %d  rmse %.0f us" % (err["cond"], err["dof"],
                                                  err["rmse"]))
    print("b_lo   %9.1f +/- %8.1f" % (err["b_lo"], err["se_b_lo"]))
    print("b_hi   %9.1f +/- %8.1f" % (err["b_hi"], err["se_b_hi"]))
    print("dslope %9.1f +/- %8.1f  t=%.2f  95%% [%.0f, %.0f]  excl 0: %s"
          % (err["dslope"], err["se_dslope"], err["dslope_t"],
             err["dslope_ci95"][0], err["dslope_ci95"][1],
             err["dslope_ci95_excludes_zero"]))
    print("ratio  %9.3f +/- %8.3f  95%% [%.3f, %.3f]  excl 1: %s  incl 2: %s"
          % (err["ratio"], err["se_ratio"], err["ratio_ci95"][0],
             err["ratio_ci95"][1], err["ratio_ci95_excludes_one"],
             err["ratio_ci95_contains_two"]))

    lev = leverage(points, OUR_BREAK)
    print("\n## 1c. leverage - which prompt carries the upper segment")
    print("%-10s %8s %8s %10s %10s"
          % ("prompt", "Mbar", "hat", "mass>=6", "E[M|>=6]"))
    for row in lev:
        print("%-10s %8.3f %8.3f %10.4f %10.3f"
              % (row["prompt"], row["mbar"], row["hat"],
                 row["mass_above_break"], row["mean_M_above_break"]))

    table = model_table(points, PASSES_POST_E100)
    by_name = {row["model"]: row for row in table}
    print("\n## 2. model shapes on OUR points, corrected G labels")
    print("%-24s %6s %9s %9s %9s %10s"
          % ("model", "params", "rmse us", "aicc", "cond", "slope x"))
    for row in table:
        print("%-24s %6d %9.0f %9.2f %9.1f %10.2f"
              % (row["model"], row["params"], row["rmse_us"], row["aicc"],
                 row["cond"], row["implied_slope_ratio"]))

    tests = {}
    for restricted in ("passcount_affine", "passcount_scaled_affine"):
        tests[restricted] = f_test(by_name[restricted],
                                   by_name["passcount_freeslope"])
        got = tests[restricted]
        print("F test %s inside passcount_freeslope: F(%d,%d)=%.3f p=%.4f"
              % (restricted, got["q"], got["dof"], got["f"], got["p"]))

    scen_rows = []
    for name in ("assumed", "predicted", "band_lo", "band_hi"):
        pts = build_points(receipt, scenarios[name], hists)
        got = slope_errors(pts, OUR_BREAK)
        tbl = model_table(pts, PASSES_POST_E100)
        scen_rows.append({
            "scenario": name, "b_lo": got["b_lo"], "b_hi": got["b_hi"],
            "ratio": got["ratio"], "se_ratio": got["se_ratio"],
            "ratio_ci95": got["ratio_ci95"],
            "excludes_one": got["ratio_ci95_excludes_one"],
            "contains_two": got["ratio_ci95_contains_two"],
            "best_model": tbl[0]["model"],
        })
    print("\n## 1d. is the ratio robust to the unpinned R vector?")
    print("%-10s %9s %9s %8s %18s %7s %7s %-24s"
          % ("R vector", "b_lo", "b_hi", "ratio", "95 % CI", "excl 1",
             "incl 2", "best model"))
    for row in scen_rows:
        print("%-10s %9.1f %9.1f %8.3f  [%6.3f, %6.3f] %7s %7s %-24s"
              % (row["scenario"], row["b_lo"], row["b_hi"], row["ratio"],
                 row["ratio_ci95"][0], row["ratio_ci95"][1],
                 row["excludes_one"], row["contains_two"],
                 row["best_model"]))

    sweep = boundary_sweep(points)
    print("\n## 2a. where each family puts the G=1 -> G=2 boundary (aicc)")
    print("%-24s " % "model"
          + " ".join("%9s" % ("M>=%d" % b) for b in range(3, 9)))
    for name in ("passcount_affine", "passcount_scaled_affine",
                 "passcount_freeslope"):
        rows = [r for r in sweep if r["model"] == name]
        best = min(rows, key=lambda r: r["aicc"])
        print("%-24s " % name
              + " ".join("%9.2f" % r["aicc"] for r in rows)
              + "   best M>=%d rmse %.0f" % (best["boundary_M"],
                                             best["rmse_us"]))

    joint = {"full": joint_selection(points),
             "loo": [dict(dropped=points[i]["prompt"],
                          **joint_selection(points, i))
                     for i in range(len(points))]}
    print("\n## 2a-loo. joint (family, boundary) selection, leave one out")
    print("%-10s %-24s %8s %9s %9s"
          % ("dropped", "model", "boundary", "aicc", "slope x"))
    print("%-10s %-24s %8s %9.2f %9.2f"
          % ("(none)", joint["full"]["model"],
             "M>=%s" % joint["full"]["boundary_M"], joint["full"]["aicc"],
             joint["full"]["slope_ratio"]))
    for row in joint["loo"]:
        print("%-10s %-24s %8s %9.2f %9.2f"
              % (row["dropped"], row["model"], "M>=%s" % row["boundary_M"],
                 row["aicc"], row["slope_ratio"]))
    picks = [(r["model"], r["boundary_M"]) for r in joint["loo"]]
    print("distinct joint picks over 8 drops: %d  %s"
          % (len(set(picks)), sorted(set(picks))))

    pre = model_table(points, PASSES_PRE_E100)
    pre_by_name = {row["model"]: row for row in pre}
    print("\n## 2b. same models under F97's PRE-E100 G labels, as a control")
    print("%-24s %9s %9s" % ("model", "rmse us", "aicc"))
    for row in pre:
        print("%-24s %9.0f %9.2f" % (row["model"], row["rmse_us"],
                                     row["aicc"]))
    print("corrected labels beat pre-E100 labels on passcount_affine by "
          "%.0f us rmse"
          % (pre_by_name["passcount_affine"]["rmse_us"]
             - by_name["passcount_affine"]["rmse_us"]))

    board_vals, board_table = board_pseudo_points(BOARD_CURVE,
                                                  PASSES_PRE_E100)
    print("\n## 2c. F97's published board curve refitted at its own widths")
    print("%-24s %9s %9s" % ("model", "rmse us", "aicc"))
    for row in board_table:
        print("%-24s %9.0f %9.2f" % (row["model"], row["rmse_us"],
                                     row["aicc"]))

    ours_vals = np.array([by_name["passcount_freeslope"]["curve"][i]
                          for i in range(MAX_ROWS)])
    piece = fit(points, OUR_BREAK, "piece", True)
    piece_vals = np.array([curve_us(piece, m) for m in ROWS])
    infl = (within_segment_inflation(piece_vals, PASSES_POST_E100, "ours")
            + within_segment_inflation(board_vals, PASSES_PRE_E100,
                                       "board_f97_labels")
            + within_segment_inflation(board_vals, PASSES_POST_E100,
                                       "board_corrected_labels"))
    print("\n## 3d. within-segment inflation - the correct bandwidth "
          "instrument")
    print("%-24s %6s %6s %6s %10s %10s %11s %12s"
          % ("curve", "G", "M lo", "M hi", "us lo", "us hi", "inflation %",
             "free-arith %"))
    for row in infl:
        print("%-24s %6d %6.0f %6.0f %10.0f %10.0f %11.1f %12.1f"
              % (row["curve"], row["passes"], row["m_lo"], row["m_hi"],
                 row["us_lo"], row["us_hi"], row["inflation_pct"],
                 row["saving_pct_if_arithmetic_free"]))

    decomp = tier_decomposition(points)
    print("\n## 2e. is the tier a fixed cost or a per-row cost?")
    print("T = c + a (G-1) + b M + k (G-1) M, in us")
    print("%-32s %6s %8s %8s %9s %8s %8s %9s %5s %5s %5s"
          % ("model@break", "params", "rmse", "aicc", "c", "a", "b", "k",
             "mono", "a>=0", "step"))
    for row in decomp[:12]:
        print("%-32s %6d %8.0f %8.2f %9.0f %8.0f %8.0f %9.0f %5s %5s %5s"
              % (row["name"], row["params"], row["rmse_us"], row["aicc"],
                 row["c"], row["a"], row["b"], row["k"], row["monotone"],
                 row["pass_overhead_nonnegative"], row["step_nonnegative"]))
    valid = [r for r in decomp if r["monotone"] and r["step_nonnegative"]]
    print("monotone and non-decreasing at the step: %d of %d candidates"
          % (len(valid), len(decomp)))
    print("best physically valid: %s  rmse %.0f  aicc %.2f  a=%.0f  k=%.0f"
          % (valid[0]["name"], valid[0]["rmse_us"], valid[0]["aicc"],
             valid[0]["a"], valid[0]["k"]))
    with_a = [r for r in valid if r["pass_overhead_nonnegative"]]
    print("of those, with a NON-NEGATIVE per-pass overhead: %d"
          % len(with_a))
    if with_a:
        print("best with a>=0: %s rmse %.0f aicc %.2f"
              % (with_a[0]["name"], with_a[0]["rmse_us"], with_a[0]["aicc"]))

    candidates = candidate_curves(points)
    print("\n## 2d. the curves that survive, in price-tool shape")
    print("%-26s %6s %11s %11s %11s %11s  " % (
        "curve", "break", "lo int", "lo slope", "hi int", "hi slope")
        + " ".join("%7s" % ("M=%d" % m) for m in range(1, 9)))
    for row in candidates:
        print("%-26s %6d %11.1f %11.1f %11.1f %11.1f  " % (
            row["name"], row["breakpoint"], row["lo"][0], row["lo"][1],
            row["hi"][0], row["hi"][1])
            + " ".join("%7.0f" % curve_us(row, m) for m in range(1, 9)))

    payload = {
        "harness": "ranked",
        "candidate_curves": candidates,
        "tier_decomposition": decomp,
        "receipt": {"id": receipt["id"], "score": receipt["score"],
                    "prefix": args.receipt},
        "r_scenario": args.scenario,
        "breakpoint": OUR_BREAK,
        "passes_post_e100": PASSES_POST_E100.tolist(),
        "passes_pre_e100": PASSES_PRE_E100.tolist(),
        "slope_loo": loo,
        "slope_errors": err,
        "leverage": lev,
        "model_table_corrected_labels": table,
        "model_table_pre_e100_labels": pre,
        "f_tests": tests,
        "ratio_by_r_scenario": scen_rows,
        "boundary_sweep": sweep,
        "joint_selection": joint,
        "board_curve_refit": {"values_us": board_vals.tolist(),
                              "table": board_table},
        "our_piece_curve_us": piece_vals.tolist(),
        "our_freeslope_curve_us": ours_vals.tolist(),
        "within_segment_inflation": infl,
        "retracted": {
            "figure": "542.8 GB/s ranked stream rate",
            "reason": "advisor error 107: 2 * 14.4123 GB / 53108 us is an "
                      "achieved rate measured when M=5 still read the weights "
                      "twice, so it is not a ceiling",
            "appears_in_e128_work": False,
            "appears_elsewhere_in_research": [
                "research/finding22_reprice.py:8",
                "research/finding22_reprice.py:21",
            ],
            "replacement_instrument": "within-segment inflation",
        },
    }
    if args.curve_json:
        key = lambda name: name.replace("passcount_", "").replace(
            "@M>=", "_b")
        head = candidates[0]
        curve_payload = {
            "harness": "ranked", "source": "e128_slopes.py",
            "receipt": args.receipt, "r_scenario": args.scenario,
            "model": head["family"], "breakpoint": head["breakpoint"],
            "lo": head["lo"], "hi": head["hi"], "rmse_us": head["rmse_us"],
            "curves": {key(row["name"]): {
                "name": row["name"], "model": row["family"],
                "breakpoint": row["breakpoint"], "lo": row["lo"],
                "hi": row["hi"], "rmse_us": row["rmse_us"],
                "aicc": row["aicc"]} for row in candidates},
        }
        args.curve_json.parent.mkdir(parents=True, exist_ok=True)
        args.curve_json.write_text(
            json.dumps(curve_payload, indent=2, sort_keys=True))
        print("wrote %s with keys %s"
              % (args.curve_json, sorted(curve_payload["curves"])))

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, sort_keys=True))
        print("\nwrote %s" % args.json)


if __name__ == "__main__":
    main()
