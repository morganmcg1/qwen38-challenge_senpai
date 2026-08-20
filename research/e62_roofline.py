#!/usr/bin/env python3
"""Reconcile the E62 ladder with the ledger 199(A) roofline ceiling.

Three jobs, all requested in the advisor's roofline comment:

1. Rescale the fitted slope from the student's pooled covariate to a real
   per-commit cost, using the candidate share of dispatches per round.
2. Report the design's detectable effect against the physical ceiling, so a
   null is distinguishable from "cost at its physical maximum".
3. Fit the ladder in two segments, because the linear model that the ceiling
   arithmetic assumes is not what the measurement shows.

  research/e62_roofline.py --out research/e62-artifacts/e62-roofline.json
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib

import numpy as np

# Ledger 199(A) and the advisor's roofline arithmetic.
SERIAL_DISPATCHES_PER_ROUND = 1706.5
CANDIDATE_DISPATCHES_PER_ROUND = 1045.3
POOLED_DISPATCHES_PER_ROUND = 1617.15
CANDIDATE_SHARE = CANDIDATE_DISPATCHES_PER_ROUND / POOLED_DISPATCHES_PER_ROUND
TOKENS_PER_CANDIDATE_ROUND = 6.4
NON_BANDWIDTH_BUDGET_S = 1.770e-3
SERIAL_COMMITS_AT_SHIPPED = SERIAL_DISPATCHES_PER_ROUND / 30.95
CEILING_C_SECONDS = NON_BANDWIDTH_BUDGET_S / SERIAL_COMMITS_AT_SHIPPED


def student_t_975(dof: int) -> float:
    table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
             7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179}
    return table.get(dof, 1.960 + 2.4 / dof) if dof > 0 else float("nan")


def fit(legs: list[dict], metric: str) -> dict | None:
    """OLS of metric on pooled commits/round, controlling for leg position."""
    if len({leg["commits_per_round"] for leg in legs}) < 2:
        return None
    commits = np.array([leg["commits_per_round"] for leg in legs])
    positions = np.array([leg["position"] for leg in legs], dtype=float)
    columns = [np.ones(len(legs)), commits - commits.mean()]
    if len(legs) > 3:
        columns.append(positions - positions.mean())
    design = np.column_stack(columns)
    response = np.array([leg[metric] for leg in legs])
    coefficients, *_ = np.linalg.lstsq(design, response, rcond=None)
    residuals = response - design @ coefficients
    dof = len(legs) - design.shape[1]
    if dof <= 0:
        return None
    residual_var = float(residuals @ residuals) / dof
    covariance = residual_var * np.linalg.pinv(design.T @ design)
    se = float(np.sqrt(np.diag(covariance))[1])
    slope = float(coefficients[1])
    critical = student_t_975(dof)
    mean = float(response.mean())
    return {
        "n_legs": len(legs),
        "dof": dof,
        "commits_per_round_range": [float(commits.min()), float(commits.max())],
        "slope_b": slope,
        "se": se,
        "t": slope / se if se else float("nan"),
        "ci95": [slope - critical * se, slope + critical * se],
        "ci95_halfwidth": critical * se,
        "residual_sd_percent": 100.0 * math.sqrt(residual_var) / mean,
        # b is s/token per pooled commit/round; c is seconds per candidate commit
        "per_commit_cost_seconds": slope * TOKENS_PER_CANDIDATE_ROUND
        / CANDIDATE_SHARE,
        "per_commit_cost_ci95_seconds": [
            (slope - critical * se) * TOKENS_PER_CANDIDATE_ROUND / CANDIDATE_SHARE,
            (slope + critical * se) * TOKENS_PER_CANDIDATE_ROUND / CANDIDATE_SHARE,
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", default="research/e62-artifacts/e62-r1ops.json")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    payload = json.loads(pathlib.Path(args.session).read_text())
    legs = [leg for leg in payload["legs"] if leg.get("commits_per_round")]
    metric = payload["regression"]["metric"]

    # b <= ceiling_c * candidate_share / tokens_per_candidate_round
    ceiling_b = CEILING_C_SECONDS * CANDIDATE_SHARE / TOKENS_PER_CANDIDATE_ROUND

    ladder = [leg for leg in legs if leg["mb"] == 4096]
    shipped_commits = 52.2
    low = [leg for leg in ladder if leg["commits_per_round"] >= shipped_commits]
    high = [leg for leg in ladder if leg["commits_per_round"] <= shipped_commits]
    ship = [leg for leg in legs if leg["mb"] == 512]
    # The shipped arm anchors both segments; it is the same geometry as `null`.
    low_with_ship = low + ship
    high_with_ship = high + ship

    result = {
        "constants": {
            "candidate_share_of_dispatches": CANDIDATE_SHARE,
            "tokens_per_candidate_round": TOKENS_PER_CANDIDATE_ROUND,
            "non_bandwidth_budget_per_serial_round_s": NON_BANDWIDTH_BUDGET_S,
            "serial_commits_at_shipped": SERIAL_COMMITS_AT_SHIPPED,
            "ceiling_per_commit_cost_s": CEILING_C_SECONDS,
            "ceiling_b_s_per_token_per_commit_per_round": ceiling_b,
        },
        "full_ladder": fit(ladder, metric),
        "low_segment_above_shipped_commits": fit(low_with_ship, metric),
        "high_segment_below_shipped_commits": fit(high_with_ship, metric),
    }

    for key in ("full_ladder", "low_segment_above_shipped_commits",
                "high_segment_below_shipped_commits"):
        block = result[key]
        if not block:
            continue
        block["fraction_of_ceiling"] = block["slope_b"] / ceiling_b
        block["ci95_halfwidth_as_fraction_of_ceiling"] = (
            block["ci95_halfwidth"] / ceiling_b
        )
        # Two-sided 5 %, 80 % power needs about 2.80 standard errors.
        block["detectable_effect_b_80pct_power"] = 2.80 * block["se"]
        block["detectable_effect_as_fraction_of_ceiling"] = (
            2.80 * block["se"] / ceiling_b
        )

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
