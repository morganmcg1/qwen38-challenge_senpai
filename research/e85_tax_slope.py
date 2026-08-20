#!/usr/bin/env python3
"""Price one materialised intermediate from a buffer-tax dose-response session.

    usage: research/e85_tax_slope.py SESSION_DIR [--json OUT]

`research/e85_tax_session.sh` adds K zero-valued elementwise results to the
per-draft proposal path and times the same candidate at several K. The slope of
decode time against K prices one added intermediate directly, instead of
inferring it from the six intermediates E85 removes, whose combined effect sits
under one third of a single leg's noise.

Two drift-free estimators are reported.

`pass`  Each palindromic pass visits every level twice, at positions j and
        2L-1-j. Those positions sum to a constant, so the mean of a level's two
        legs is free of any linear drift. One least-squares slope is fitted per
        pass, and the spread across passes gives the interval.

`ols`   One fit of value ~ 1 + leg_order + K over every leg. The optional
        covariate form adds the unchanged serial leg of the same wrapper run,
        which absorbs host-level noise shared by both legs.

One tax unit is one extra dispatch AND one extra materialised buffer, so the
slope is an upper bound on the buffer price. A slope at or below the 0.66-1.55
us E80 dispatch cost means the buffer itself is close to free.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e85_abba_stats import drafts_per_token, t95  # noqa: E402

CLAIMED_LO_US = 13.0
CLAIMED_HI_US = 16.0


def read_legs(path: Path) -> list[dict]:
    with path.open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    for row in rows:
        for key in ("mtp_s_per_tok", "serial_s_per_tok", "ratio",
                    "mean_draft_len", "accepted_rate", "temp_in", "temp_out"):
            row[key] = float(row[key]) if row[key] not in ("", "None") else math.nan
        row["leg"] = int(row["leg"])
        row["tax"] = int(row["tax"])
        row["seconds"] = int(row["seconds"])
    return rows


def solve(xtx: list[list[float]], xty: list[float]) -> tuple[list[float], list[list[float]]]:
    k = len(xty)
    aug = [xtx[i][:] + [1.0 if i == j else 0.0 for j in range(k)] + [xty[i]]
           for i in range(k)]
    for col in range(k):
        pivot = max(range(col, k), key=lambda r: abs(aug[r][col]))
        aug[col], aug[pivot] = aug[pivot], aug[col]
        div = aug[col][col]
        aug[col] = [v / div for v in aug[col]]
        for r in range(k):
            if r == col:
                continue
            factor = aug[r][col]
            aug[r] = [v - factor * w for v, w in zip(aug[r], aug[col])]
    beta = [aug[i][-1] for i in range(k)]
    inv = [aug[i][k:k + k] for i in range(k)]
    return beta, inv


def ols_slope(rows: list[dict], field: str, covariate: str | None = None) -> dict:
    """Fit value ~ 1 + leg_order [+ covariate] + tax; return the tax slope."""
    n = len(rows)

    def design(i: int, r: dict) -> list[float]:
        head = [1.0, float(i)]
        if covariate is not None:
            head.append(r[covariate])
        head.append(float(r["tax"]))
        return head

    x = [design(i, r) for i, r in enumerate(rows)]
    y = [r[field] for r in rows]
    k = len(x[0])
    slope_col = k - 1
    xtx = [[sum(x[i][a] * x[i][b] for i in range(n)) for b in range(k)] for a in range(k)]
    xty = [sum(x[i][a] * y[i] for i in range(n)) for a in range(k)]
    beta, inv = solve(xtx, xty)

    resid = [y[i] - sum(beta[a] * x[i][a] for a in range(k)) for i in range(n)]
    dof = n - k
    sigma2 = sum(r * r for r in resid) / dof
    se = math.sqrt(sigma2 * inv[slope_col][slope_col])
    slope = beta[slope_col]
    return {
        "slope": slope,
        "stderr": se,
        "t": slope / se if se > 0 else math.nan,
        "dof": dof,
        "ci95_lo": slope - t95(dof) * se,
        "ci95_hi": slope + t95(dof) * se,
        "intercept": beta[0],
        "drift_per_leg": beta[1],
        "resid_sd": math.sqrt(sigma2),
        "covariate": covariate,
    }


def pass_slopes(rows: list[dict], field: str, n_levels: int) -> dict:
    """One drift-free least-squares slope per complete palindromic pass."""
    width = 2 * n_levels
    slopes: list[float] = []
    for start in range(0, len(rows) - width + 1, width):
        block = rows[start:start + width]
        by_level: dict[int, list[float]] = {}
        for r in block:
            by_level.setdefault(r["tax"], []).append(r[field])
        if len(by_level) != n_levels or any(len(v) != 2 for v in by_level.values()):
            continue
        ks = sorted(by_level)
        means = [statistics.fmean(by_level[k]) for k in ks]
        kbar = statistics.fmean(float(k) for k in ks)
        ybar = statistics.fmean(means)
        num = sum((k - kbar) * (m - ybar) for k, m in zip(ks, means))
        den = sum((k - kbar) ** 2 for k in ks)
        slopes.append(num / den)
    if not slopes:
        return {"slopes": [], "slope": math.nan}
    mean = statistics.fmean(slopes)
    sd = statistics.stdev(slopes) if len(slopes) > 1 else math.nan
    sem = sd / math.sqrt(len(slopes)) if len(slopes) > 1 else math.nan
    dof = len(slopes) - 1
    return {
        "slopes": slopes,
        "slope": mean,
        "sd": sd,
        "stderr": sem,
        "t": mean / sem if sem and sem > 0 else math.nan,
        "dof": dof,
        "ci95_lo": mean - t95(dof) * sem if sem and sem > 0 else math.nan,
        "ci95_hi": mean + t95(dof) * sem if sem and sem > 0 else math.nan,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    root = Path(args.session)
    rows = read_legs(root / "legs.tsv")
    if not rows:
        raise SystemExit(f"{root}/legs.tsv has no legs")

    levels = sorted({r["tax"] for r in rows})
    dpt = statistics.fmean(drafts_per_token(r) for r in rows)

    report: dict = {
        "session": str(root),
        "legs": len(rows),
        "tax_levels": levels,
        "drafts_per_token": dpt,
        "all_tokens_matched": all(r["matched"] == "True" for r in rows),
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "temp_in_min": min(r["temp_in"] for r in rows),
        "temp_in_max": max(r["temp_in"] for r in rows),
        "per_level": {},
    }
    report["temp_in_spread"] = report["temp_in_max"] - report["temp_in_min"]

    for k in levels:
        sub = [r for r in rows if r["tax"] == k]
        report["per_level"][str(k)] = {
            "legs": len(sub),
            "mtp_s_per_tok_mean": statistics.fmean(r["mtp_s_per_tok"] for r in sub),
            "mtp_s_per_tok_sd": statistics.stdev(r["mtp_s_per_tok"] for r in sub)
            if len(sub) > 1 else math.nan,
            "serial_s_per_tok_mean": statistics.fmean(r["serial_s_per_tok"] for r in sub),
            "mean_draft_len": statistics.fmean(r["mean_draft_len"] for r in sub),
            "accepted_rate": statistics.fmean(r["accepted_rate"] for r in sub),
        }

    baseline = report["per_level"][str(levels[0])]["mtp_s_per_tok_mean"]
    report["baseline_mtp_s_per_tok"] = baseline

    estimators = {
        "pass": pass_slopes(rows, "mtp_s_per_tok", len(levels)),
        "ols": ols_slope(rows, "mtp_s_per_tok"),
        "cov": ols_slope(rows, "mtp_s_per_tok", covariate="serial_s_per_tok"),
    }
    # The unchanged serial leg must show no tax response. It is the control.
    estimators["control_serial"] = ols_slope(rows, "serial_s_per_tok")

    for name, est in estimators.items():
        for key in ("slope", "ci95_lo", "ci95_hi"):
            value = est.get(key, math.nan)
            est[f"{key}_us_per_token_per_unit"] = value * 1e6
            est[f"{key}_us_per_buffer"] = value * 1e6 / dpt if dpt else math.nan
        est["percent_of_candidate_per_unit"] = (
            100.0 * est.get("slope", math.nan) / baseline if baseline else math.nan)
    report["estimators"] = estimators

    point = estimators["ols"]["slope_us_per_buffer"]
    hi = estimators["ols"]["ci95_hi_us_per_buffer"]
    report["claimed_us_per_buffer"] = [CLAIMED_LO_US, CLAIMED_HI_US]
    report["claim_excluded_by_ci"] = bool(hi < CLAIMED_LO_US)
    report["overprediction_factor"] = (
        CLAIMED_LO_US / point if point and point > 0 else math.nan)

    if report["claim_excluded_by_ci"] and point < 5.0:
        verdict = ("terminal-negative: the 95 percent interval for one added "
                   "intermediate excludes the claimed 13-16 us and sits under "
                   "the 5 us stop-rule floor")
    elif hi < CLAIMED_LO_US:
        verdict = ("law-does-not-hold: the interval excludes 13-16 us but the "
                   "point estimate is above the 5 us floor")
    elif estimators["ols"]["ci95_lo_us_per_buffer"] > CLAIMED_HI_US:
        verdict = "law-understates: the interval sits above the claimed range"
    else:
        verdict = "inconclusive: the interval still contains the claimed range"
    report["verdict"] = verdict

    print(json.dumps(report, indent=2, default=str))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
