#!/usr/bin/env python3
"""The campaign-standard `value ~ arm + position` estimator, shared.

Both E59 measurement layers need the same fit: the whole-table cell palindrome
and the end-to-end leg palindrome. A difference of arm means confounds the arm
with where in the session it ran; carrying position as a covariate separates
the two and reports the residual degrees of freedom honestly.

A palindrome makes arm and linear position exactly orthogonal, so the arm
contrasts equal the plain means. The fit is still worth running: it prices the
drift, and its residual sd is the session's own noise estimate.

Plain normal equations by Gaussian elimination. The design is tiny and a SciPy
dependency inside a timed workspace is not worth it.
"""

from __future__ import annotations

import math


def mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs)


def fit_arm_position(observations: list[dict], base_arm: str) -> dict:
    """`value ~ arm + position` with `base_arm` as the reference level.

    Each observation needs `arm`, `position`, `value`, and a `label` used only
    to attribute residuals.
    """
    arms = sorted({obs["arm"] for obs in observations})
    if base_arm not in arms:
        return {"fitted": False, "reason": f"no {base_arm} observations"}
    others = [a for a in arms if a != base_arm]
    names = ["intercept"] + [f"arm[{a}]" for a in others] + ["position"]

    rows, ys = [], []
    for obs in observations:
        row = [1.0]
        row += [1.0 if obs["arm"] == a else 0.0 for a in others]
        row.append(float(obs["position"]))
        rows.append(row)
        ys.append(float(obs["value"]))

    k = len(names)
    n = len(rows)
    dof = n - k
    if dof <= 0:
        return {"fitted": False, "reason": f"{n} observations cannot fit {k} parameters"}

    xtx = [[sum(rows[i][a] * rows[i][b] for i in range(n)) for b in range(k)]
           for a in range(k)]
    xty = [sum(rows[i][a] * ys[i] for i in range(n)) for a in range(k)]

    inv = [[1.0 if i == j else 0.0 for j in range(k)] for i in range(k)]
    work = [row[:] for row in xtx]
    rhs = xty[:]
    for col in range(k):
        pivot = max(range(col, k), key=lambda r: abs(work[r][col]))
        if abs(work[pivot][col]) < 1e-18:
            return {"fitted": False, "reason": "singular design"}
        work[col], work[pivot] = work[pivot], work[col]
        inv[col], inv[pivot] = inv[pivot], inv[col]
        rhs[col], rhs[pivot] = rhs[pivot], rhs[col]
        scale = work[col][col]
        work[col] = [v / scale for v in work[col]]
        inv[col] = [v / scale for v in inv[col]]
        rhs[col] /= scale
        for r in range(k):
            if r == col:
                continue
            factor = work[r][col]
            work[r] = [v - factor * w for v, w in zip(work[r], work[col])]
            inv[r] = [v - factor * w for v, w in zip(inv[r], inv[col])]
            rhs[r] -= factor * rhs[col]
    beta = rhs

    resid = [ys[i] - sum(rows[i][a] * beta[a] for a in range(k)) for i in range(n)]
    sigma2 = sum(r * r for r in resid) / dof

    base_level = beta[0] + beta[k - 1] * mean(
        float(obs["position"]) for obs in observations)
    terms = {}
    for i, name in enumerate(names):
        se = math.sqrt(max(sigma2 * inv[i][i], 0.0))
        terms[name] = {
            "estimate": beta[i],
            "std_error": se,
            "t": beta[i] / se if se > 0 else None,
            "estimate_pct_of_base": (
                100.0 * beta[i] / base_level if name.startswith("arm[") else None),
        }
    return {
        "fitted": True,
        "n": n,
        "parameters": k,
        "residual_dof": dof,
        "residual_sd": math.sqrt(sigma2),
        "residual_sd_pct_of_base": 100.0 * math.sqrt(sigma2) / base_level,
        "reference_arm": base_arm,
        "base_level_at_mean_position": base_level,
        "position_drift_pct_of_base_per_leg": 100.0 * beta[k - 1] / base_level,
        "terms": terms,
        "names": names,
        "covariance": [[sigma2 * inv[i][j] for j in range(k)] for i in range(k)],
        "residuals": [
            {"label": obs["label"], "arm": obs["arm"],
             "position": obs["position"],
             "residual_pct_of_base": 100.0 * resid[i] / base_level}
            for i, obs in enumerate(observations)
        ],
    }


def arm_contrast(fit: dict, left: str, right: str) -> dict:
    """`arm[left] - arm[right]` as a percent of the base level, with its error.

    Either side may be the reference arm, which has no column: its coefficient
    and every covariance entry involving it are zero by construction.
    """
    if not fit.get("fitted"):
        return {"available": False}
    names = fit["names"]
    reference = fit["reference_arm"]
    index = {}
    for side in (left, right):
        if side == reference:
            index[side] = None
        elif f"arm[{side}]" in names:
            index[side] = names.index(f"arm[{side}]")
        else:
            return {"available": False}

    cov = fit["covariance"]

    def coef(side: str) -> float:
        i = index[side]
        return 0.0 if i is None else fit["terms"][names[i]]["estimate"]

    def cov_at(a: str, b: str) -> float:
        i, j = index[a], index[b]
        return 0.0 if i is None or j is None else cov[i][j]

    estimate = coef(left) - coef(right)
    variance = cov_at(left, left) + cov_at(right, right) - 2.0 * cov_at(left, right)
    se = math.sqrt(max(variance, 0.0))
    base_level = fit["base_level_at_mean_position"]
    return {
        "available": True,
        "estimate_pct_of_base": 100.0 * estimate / base_level,
        "std_error_pct_of_base": 100.0 * se / base_level,
        "t": estimate / se if se > 0 else None,
        "residual_dof": fit["residual_dof"],
    }
