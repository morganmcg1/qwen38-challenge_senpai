#!/usr/bin/env python3
"""E155 bonus -- within-prompt round-cost versus verified width.

Fits `round_us = a_p + b_p * M` inside each prompt, where M = d + 1 is the
number of target rows the round verified. No new GPU time: the inputs are the
per-round `block_request_seconds` and `effective_draft_lengths` already stored
in earlier E153 reports.

Sign convention: b_p > 0 means a wider round costs more microseconds, so b_p is
the marginal price of one extra verified row inside one prompt.

The published cross-prompt slope this is compared against was fitted BETWEEN
prompt means. Within-prompt and between-prompt slopes answer different
questions, and a large gap between them is confounding evidence, not noise.

A second fit adds the round index as a drift covariate:
`round_us = a_p + b_p * M + c_p * index`. These legs did not take the real cool
gate, so monotone thermal drift is a live alternative explanation for any
apparent width effect; c_p measures it directly.

harness=local. These are Apple M4 Pro legs, not the ranked M5 runner, and they
carry `cool_gate_passed_real_gate=false`, so no ranked slope may be read from
them.

usage: research/e155_width_cost_fit.py OUT.json REPORT.json [REPORT.json ...]
"""

from __future__ import annotations

import json
import os
import sys


def ols(rows: list[list[float]], targets: list[float]) -> list[float]:
    """Small dense least squares by Gaussian elimination on the normal
    equations. The designs here are 2 or 3 columns over ~120 rows."""
    width = len(rows[0])
    gram = [[0.0] * (width + 1) for _ in range(width)]
    for row, target in zip(rows, targets):
        for i in range(width):
            for j in range(width):
                gram[i][j] += row[i] * row[j]
            gram[i][width] += row[i] * target
    for col in range(width):
        pivot = max(range(col, width), key=lambda r: abs(gram[r][col]))
        gram[col], gram[pivot] = gram[pivot], gram[col]
        head = gram[col][col]
        if abs(head) < 1e-18:
            raise ValueError("singular design")
        gram[col] = [value / head for value in gram[col]]
        for other in range(width):
            if other == col:
                continue
            factor = gram[other][col]
            gram[other] = [
                value - factor * base
                for value, base in zip(gram[other], gram[col])
            ]
    return [gram[i][width] for i in range(width)]


def r_squared(rows, targets, beta) -> float:
    mean = sum(targets) / len(targets)
    ss_tot = sum((t - mean) ** 2 for t in targets)
    ss_res = sum(
        (t - sum(b * x for b, x in zip(beta, row))) ** 2
        for row, t in zip(rows, targets)
    )
    return 1.0 - ss_res / ss_tot if ss_tot else float("nan")


def slope_stderr(rows, targets, beta, index: int) -> float:
    """Standard error of one coefficient under homoskedastic residuals."""
    n, k = len(rows), len(rows[0])
    if n <= k:
        return float("nan")
    ss_res = sum(
        (t - sum(b * x for b, x in zip(beta, row))) ** 2
        for row, t in zip(rows, targets)
    )
    sigma2 = ss_res / (n - k)
    gram = [[0.0] * k for _ in range(k)]
    for row in rows:
        for i in range(k):
            for j in range(k):
                gram[i][j] += row[i] * row[j]
    # Invert the k x k Gram matrix by Gauss-Jordan.
    aug = [gram[i] + [1.0 if i == j else 0.0 for j in range(k)] for i in range(k)]
    for col in range(k):
        pivot = max(range(col, k), key=lambda r: abs(aug[r][col]))
        aug[col], aug[pivot] = aug[pivot], aug[col]
        head = aug[col][col]
        aug[col] = [value / head for value in aug[col]]
        for other in range(k):
            if other == col:
                continue
            factor = aug[other][col]
            aug[other] = [
                value - factor * base for value, base in zip(aug[other], aug[col])
            ]
    return (sigma2 * aug[index][k + index]) ** 0.5


def analyse(path: str) -> dict:
    report = json.load(open(path))
    seconds = report["block_request_seconds"]
    drafts = report["effective_draft_lengths"]
    if len(seconds) != len(drafts):
        raise ValueError(f"{path}: round vector length mismatch")
    # Drop round 0: it carries the seed flush and is ~35% slower than the
    # steady state on every leg, so keeping it biases the intercept.
    pairs = [
        (index, draft + 1, second * 1e6)
        for index, (draft, second) in enumerate(zip(drafts, seconds))
    ][1:]
    widths = sorted({width for _, width, _ in pairs})

    design = [[1.0, float(width)] for _, width, _ in pairs]
    targets = [micros for _, _, micros in pairs]
    beta = ols(design, targets)
    drift_design = [
        [1.0, float(width), float(index)] for index, width, _ in pairs
    ]
    drift_beta = ols(drift_design, targets)

    by_width: dict[int, list[float]] = {}
    for _, width, micros in pairs:
        by_width.setdefault(width, []).append(micros)

    return {
        "prompt": os.path.basename(os.path.dirname(path)),
        "rounds_used": len(pairs),
        "widths_observed": widths,
        "mean_round_us": sum(targets) / len(targets),
        "fit_round_us_vs_M": {
            "intercept_a_us": beta[0],
            "slope_b_us_per_row": beta[1],
            "slope_stderr_us": slope_stderr(design, targets, beta, 1),
            "r_squared": r_squared(design, targets, beta),
        },
        "fit_round_us_vs_M_and_round_index": {
            "intercept_a_us": drift_beta[0],
            "slope_b_us_per_row": drift_beta[1],
            "slope_stderr_us": slope_stderr(
                drift_design, targets, drift_beta, 1
            ),
            "drift_c_us_per_round": drift_beta[2],
            "r_squared": r_squared(drift_design, targets, drift_beta),
        },
        "mean_round_us_by_width": {
            str(width): {
                "rounds": len(values),
                "mean_us": sum(values) / len(values),
            }
            for width, values in sorted(by_width.items())
        },
    }


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    out_path = sys.argv[1]
    fits = [analyse(path) for path in sys.argv[2:]]
    slopes = [fit["fit_round_us_vs_M"]["slope_b_us_per_row"] for fit in fits]
    drift_slopes = [
        fit["fit_round_us_vs_M_and_round_index"]["slope_b_us_per_row"]
        for fit in fits
    ]
    report = {
        "harness": "local",
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "cross_prompt_reference_slope_us_per_row": 4291.0,
        "within_prompt_fits": fits,
        "within_prompt_slope_mean_us": sum(slopes) / len(slopes),
        "within_prompt_slope_mean_us_drift_controlled": sum(drift_slopes)
        / len(drift_slopes),
    }
    with open(out_path, "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
