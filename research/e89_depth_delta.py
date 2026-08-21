#!/usr/bin/env python3
"""Task B: does the slow-state host penalty scale with the round's draft depth?

Regresses the per-round host phase sum on chosen depth, host-state and their
interaction. The intercept difference is the part a prompt that almost never
drafts would still pay; the slope difference is the per-draft-step part.

Rounds are labelled by the core the round ran on, not by the host phase sum, so
the label is independent of the regressand. cpu0-3 are the efficiency cores on
this host. The host-sum cut is reported as a cross-check.
"""
import json
import os
import sys

import numpy as np

PREFIX = sys.argv[1] if len(sys.argv) > 1 else "research/out/e89r2"
WARMUP = int(sys.argv[2]) if len(sys.argv) > 2 else 8
SLOW_CUT_US = 1246
ECORE_MAX = 3

FIELDS = ["round", "d", "host_sum_us", "e89_core_a", "e89_core_b"]


def parse(path):
    rows = []
    with open(path) as handle:
        for line in handle:
            if "mtp-trace:" not in line:
                continue
            row = {}
            for token in line.split():
                key, _, value = token.partition("=")
                if key in FIELDS:
                    try:
                        row[key] = int(value)
                    except ValueError:
                        pass
            if len(row) == len(FIELDS):
                rows.append(row)
    return rows


def load(prefix):
    parent = os.path.dirname(prefix) or "."
    stem = os.path.basename(prefix)
    legs = []
    for name in sorted(os.listdir(parent)):
        if not name.startswith(stem) or "-bg-" in name:
            continue
        trace = os.path.join(parent, name, "trace.txt")
        if os.path.exists(trace):
            legs.append((name, [r for r in parse(trace) if r["round"] > WARMUP]))
    return legs


def cluster_ols(X, y, groups):
    """OLS with cluster-robust (CR1) covariance, clustered on `groups`."""
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    resid = y - X @ beta
    meat = np.zeros((X.shape[1], X.shape[1]))
    unique = np.unique(groups)
    for g in unique:
        sel = groups == g
        Xg, ug = X[sel], resid[sel]
        s = Xg.T @ ug
        meat += np.outer(s, s)
    n, k, m = X.shape[0], X.shape[1], len(unique)
    scale = (m / (m - 1.0)) * ((n - 1.0) / (n - k))
    cov = xtx_inv @ (scale * meat) @ xtx_inv
    return beta, np.sqrt(np.diag(cov)), m


def fit(rows, legs, label_slow, name):
    depth = np.array([r["d"] for r in rows], dtype=float)
    host = np.array([r["host_sum_us"] for r in rows], dtype=float)
    slow = np.array([1.0 if label_slow(r) else 0.0 for r in rows])
    groups = np.array(legs)
    X = np.column_stack([np.ones_like(depth), depth, slow, slow * depth])
    beta, se, n_clusters = cluster_ols(X, host, groups)
    names = ["intercept_fast", "slope_fast", "delta_intercept", "delta_slope"]
    out = {
        "label": name,
        "n_rounds": int(len(rows)),
        "n_legs": int(n_clusters),
        "n_slow_rounds": int(slow.sum()),
        "depth_histogram": {
            str(int(d)): int((depth == d).sum()) for d in sorted(set(depth))
        },
        "coefficients": {
            key: {
                "value": float(beta[i]),
                "se": float(se[i]),
                "ci95": [float(beta[i] - 1.96 * se[i]), float(beta[i] + 1.96 * se[i])],
                "t": float(beta[i] / se[i]) if se[i] else None,
            }
            for i, key in enumerate(names)
        },
    }
    predict = lambda d: beta[2] + beta[3] * d
    out["delta_us_at_depth"] = {
        f"{d:g}": float(predict(d)) for d in (1.0, 1.15, 2.3, 4.38, 5.09, 7.0)
    }
    beagle, plutarch = predict(4.382), predict(1.15)
    out["beagle_over_plutarch_predicted"] = (
        float(beagle / plutarch) if plutarch else None
    )
    out["beagle_over_plutarch_ranked"] = 0.641 / 0.077
    out["per_depth_medians"] = {}
    for d in sorted(set(depth)):
        sel = depth == d
        f = host[sel & (slow == 0)]
        s = host[sel & (slow == 1)]
        out["per_depth_medians"][str(int(d))] = {
            "n_fast": int(f.size),
            "n_slow": int(s.size),
            "median_fast_us": float(np.median(f)) if f.size else None,
            "median_slow_us": float(np.median(s)) if s.size else None,
            "delta_us": float(np.median(s) - np.median(f))
            if f.size and s.size
            else None,
        }
    return out


def median_delta_fit(rows, leg_ids, label_slow):
    """Fit the per-depth median delta on depth, so the d=7 mass cannot dominate.

    The confidence interval resamples whole legs, because the host state is a
    per-leg draw and rounds inside a leg are not independent.
    """
    depth = np.array([r["d"] for r in rows], dtype=float)
    host = np.array([r["host_sum_us"] for r in rows], dtype=float)
    slow = np.array([label_slow(r) for r in rows])
    groups = np.array(leg_ids)
    depths = np.array(sorted(set(depth)))

    def fit_from(index):
        d_sub, h_sub, s_sub = depth[index], host[index], slow[index]
        xs, ys = [], []
        for d in depths:
            sel = d_sub == d
            fast, slw = h_sub[sel & ~s_sub], h_sub[sel & s_sub]
            if fast.size and slw.size:
                xs.append(d)
                ys.append(np.median(slw) - np.median(fast))
        if len(xs) < 3:
            return None
        return np.polyfit(np.array(xs), np.array(ys), 1)

    point = fit_from(np.arange(len(depth)))
    unique = np.unique(groups)
    index_by_leg = {leg: np.flatnonzero(groups == leg) for leg in unique}
    rng = np.random.default_rng(20260821)
    draws = []
    for _ in range(2000):
        picked = rng.choice(unique, size=len(unique), replace=True)
        index = np.concatenate([index_by_leg[leg] for leg in picked])
        fitted = fit_from(index)
        if fitted is not None:
            draws.append(fitted)
    draws = np.array(draws)
    return {
        "slope_us_per_draft_step": float(point[0]),
        "intercept_us": float(point[1]),
        "slope_ci95": [
            float(np.percentile(draws[:, 0], 2.5)),
            float(np.percentile(draws[:, 0], 97.5)),
        ],
        "intercept_ci95": [
            float(np.percentile(draws[:, 1], 2.5)),
            float(np.percentile(draws[:, 1], 97.5)),
        ],
        "intercept_share_at_depth_4.38": float(
            point[1] / (point[1] + point[0] * 4.382)
        ),
        "beagle_over_plutarch_predicted": float(
            (point[1] + point[0] * 4.382) / (point[1] + point[0] * 1.15)
        ),
        "bootstrap_draws": int(draws.shape[0]),
    }


def main():
    legs = load(PREFIX)
    rows, leg_ids = [], []
    for name, leg_rows in legs:
        for row in leg_rows:
            rows.append(row)
            leg_ids.append(name)
    report = {
        "prefix": PREFIX,
        "warmup_rounds": WARMUP,
        "primary_core_label": fit(
            rows, leg_ids, lambda r: r["e89_core_a"] <= ECORE_MAX, "core<=3 is slow"
        ),
        "crosscheck_hostsum_label": fit(
            rows, leg_ids, lambda r: r["host_sum_us"] > SLOW_CUT_US,
            f"host_sum>{SLOW_CUT_US} is slow",
        ),
        "median_fit_core_label": median_delta_fit(
            rows, leg_ids, lambda r: r["e89_core_a"] <= ECORE_MAX
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


main()
