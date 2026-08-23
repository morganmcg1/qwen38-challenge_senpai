#!/usr/bin/env python3
"""Compare the four-leg pairwise fits with the full six-leg model.

ADVISOR F36. The merged palindrome shares its `c67pb6` legs between the two
contrasts, so the contrasts are correlated. The four-leg fit that produces each
published interval spends 3 of its 4 observations on parameters and leaves one
residual degree of freedom, which makes its residual standard deviation a very
noisy estimate of the leg noise. The six-leg model estimates one common
residual over 2 degrees of freedom from all six legs, so it is the better noise
estimate, and rescaling the four-leg standard errors with it is the honest
interval.

    python3 research/e135_f34_residuals.py --label f34
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e135_report as report

ARMS = ("c67ship", "c67pb6", "c678pb6")
PAIRS = (
    ("c67ship", "c67pb6", "e135_pb6_under_tight_pct"),
    ("c67pb6", "c678pb6", "e135_onepass678_local_pct"),
    ("c67ship", "c678pb6", "e135_f34_full_move_pct"),
)


def design(rows, arms):
    """Intercept, one dummy per arm after the first, and centred leg index."""
    index = np.array([r["idx"] for r in rows], dtype=float)
    cols = [np.ones(len(rows)), index - index.mean()]
    for arm in arms[1:]:
        cols.append(np.array([1.0 if r["arm"] == arm else 0.0 for r in rows]))
    return np.column_stack(cols)


def fit(rows, arms, key="mtp_seconds_per_token"):
    y = np.array([report.fnum(r["metrics"][key]) for r in rows])
    x = design(rows, arms)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    dof = len(y) - x.shape[1]
    sigma2 = float(resid @ resid) / dof
    cov = sigma2 * np.linalg.inv(x.T @ x)
    return {
        "beta": beta, "mean": float(y.mean()), "dof": dof,
        "sigma": float(np.sqrt(sigma2)), "cov": cov, "n": len(y),
        "drift_per_leg": float(beta[1]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="f34")
    args = ap.parse_args()
    report.configure("f34")
    report.ARMS = ARMS
    rows = [r for r in report.legs(args.label)
            if r["metrics"].get("mtp_seconds_per_token")]

    full = fit(rows, ARMS)
    print(f"FULL SIX-LEG MODEL  n {full['n']}  dof {full['dof']}")
    print(f"  residual sd  {full['sigma']:.3e} s/tok"
          f"  = {100 * full['sigma'] / full['mean']:.4f} % of the arm mean")
    print(f"  drift        {full['drift_per_leg']:+.3e} s/tok per leg"
          f"  = {100 * full['drift_per_leg'] / full['mean']:+.4f} % per leg")
    print("  T44's twelve-leg composition model had residual sd 0.0650 %.")

    print()
    print("CONTRASTS. Positive means the SECOND arm is FASTER.")
    print(f"{'contrast':<34}{'pct':>10}{'se4':>9}{'se6':>9}"
          f"{'sd4':>9}{'legs':>6}")
    for ref, cand, key in PAIRS:
        sub = [r for r in rows if r["arm"] in (ref, cand)]
        pair = fit(sub, (ref, cand))
        pct = -100.0 * pair["beta"][2] / pair["mean"]
        se4 = 100.0 * float(np.sqrt(pair["cov"][2, 2])) / pair["mean"]
        # Same design, same leverage, but the variance comes from all six legs.
        scale = full["sigma"] / pair["sigma"]
        se6 = se4 * scale
        sd4 = 100.0 * pair["sigma"] / pair["mean"]
        print(f"{key:<34}{pct:>+10.4f}{se4:>9.4f}{se6:>9.4f}{sd4:>9.4f}"
              f"{pair['n']:>6}")

    print()
    print("se4 uses the four legs of that contrast and one residual degree of"
          " freedom.")
    print("se6 rescales the same standard error with the six-leg residual, "
          "which has two degrees of freedom and uses every leg.")
    print("Report se6 as the interval and se4 beside it. The four-leg residual"
          " is too noisy to trust on its own.")

    print()
    print("PER LEG")
    print(f"{'i':>3}{'arm':<10}{'mtp s/tok':>12}{'resid s/tok':>14}"
          f"{'entry C':>9}{'exit C':>8}")
    y = np.array([report.fnum(r["metrics"]["mtp_seconds_per_token"])
                  for r in rows])
    resid = y - design(rows, ARMS) @ full["beta"]
    for r, obs, res in zip(rows, y, resid):
        print(f"{r['idx']:>3}{r['arm']:<10}{obs:>12.6f}{res:>+14.3e}"
              f"{report.fnum(r['meta'].get('gpu_temp_entry_c')):>9.1f}"
              f"{report.fnum(r['meta'].get('gpu_temp_exit_c')):>8.1f}")

    entry = [report.fnum(r["meta"]["gpu_temp_entry_c"]) for r in rows]
    print()
    print(f"entry temperature  min {min(entry):.1f} C  max {max(entry):.1f} C"
          f"  spread {max(entry) - min(entry):.1f} C")
    for arm in ARMS:
        v = [report.fnum(r["meta"]["gpu_temp_entry_c"])
             for r in rows if r["arm"] == arm]
        print(f"  {arm:<10} entry {[round(x, 2) for x in v]}"
              f"  mean {sum(v) / len(v):.2f} C")

    # The arms did not enter at the same temperature, so bound the confound
    # from the data rather than assuming it away. The two legs of one arm are
    # the cleanest probe: same code, same schedule, different entry.
    print()
    print("ENTRY TEMPERATURE CONFOUND")
    for arm in ARMS:
        pair = [(report.fnum(r["meta"]["gpu_temp_entry_c"]),
                 report.fnum(r["metrics"]["mtp_seconds_per_token"]))
                for r in rows if r["arm"] == arm]
        (t0, y0), (t1, y1) = pair
        if t0 == t1:
            continue
        slope = (y1 - y0) / (t1 - t0)
        print(f"  {arm:<10} dT {t1 - t0:+6.2f} C  dy {y1 - y0:+.3e} s/tok"
              f"  slope {slope:+.3e} s/tok per C"
              f"  = {100 * slope / full['mean']:+.4f} % per C")
    print("  A positive slope means a hotter entry ran slower. A sign that"
          " flips between arms, or a slope too small to matter, means entry"
          " temperature is not driving the contrast.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
