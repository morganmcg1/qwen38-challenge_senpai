#!/usr/bin/env python3
"""E173 -- the `tablePays` natural experiment, at zero GPU seconds.

`Qwen35CustomQMV.tablePays(m) = m >= 4` gates BOTH the campaign's wide-QMV
chunk-sum table path AND the two research counters FINDING 413 isolated. Every
drafting round routes the same 257 wide QMV cells whatever `m` is, and
`groups(M) = 1` for every M in 2...5, so rounds at M = 3 and M = 4 differ by:

  * one head step (the step law says the increment is `h`), and
  * the `tablePays` branch (table path, sidecar lookup, counter increments).

FINDING 415 reports a +6.94 ms M = 3 -> M = 4 increment against +1.43 ms one
step earlier, from medians binned by M. That binning has no position control and
no acceptance control, and the advisor's brief asks whether the excess survives
them.

This script answers that from per-round traces already committed to this
repository: `research/results/e37/{medicine,natural_history}-rounds.txt`, 493
rounds of a 512-token traced census on Apple M4 Pro, the same host generation as
FINDING 415's `g16s`.

PROVENANCE AND LIMITS OF THE INPUT DATA (research/results/e37/*-meta.txt):
`dirty=1`, `cool_gate_passed_real_gate=false`, `gate_qualified_for_timing=false`,
`timing_claims_permitted=false`, `trace_perturbs_timing=true`, started
2026-08-19. So these rounds support a WITHIN-RUN differential in M at fixed
routed-cell count, and nothing else. No absolute timing claim, no ranked claim,
no comparison against any other run.

Usage:
  python3 research/e173_tablepays_natural_experiment.py [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path

TRACE = re.compile(r"^mtp-trace: round=(\d+) (.*)$")
FIELD = re.compile(r"([a-z_0-9]+)=([-0-9.]+)")

FILES = {
    "medicine": "research/results/e37/medicine-rounds.txt",
    "natural_history": "research/results/e37/natural_history-rounds.txt",
}

PHASES = [
    "draft_build_us",
    "verify_build_us",
    "eval_wall_us",
    "readout_us",
    "commit_us",
    "upkeep_us",
    "round_us",
]


def parse(path: Path, prompt: str) -> list[dict]:
    rounds = []
    position = 0
    for line in path.read_text().splitlines():
        match = TRACE.match(line)
        if not match:
            continue
        fields = dict(FIELD.findall(match.group(2)))
        row = {"prompt": prompt, "round": int(match.group(1))}
        for key in ("d", "acc"):
            row[key] = int(float(fields[key]))
        for key in PHASES:
            row[key] = float(fields[key])
        row["m_verify"] = row["d"] + 1
        row["shortfall"] = row["d"] - row["acc"]
        # Cache length at the START of this round: the seed plus every token
        # committed so far. One round commits `acc + 1` tokens.
        row["cache_pos"] = 512 + position
        position += row["acc"] + 1
        rounds.append(row)
    return rounds


def ols(y: list[float], columns: dict[str, list[float]]):
    """Least squares with heteroskedasticity-naive standard errors."""
    names = list(columns)
    n = len(y)
    p = len(names) + 1
    design = [[1.0] + [columns[name][i] for name in names] for i in range(n)]
    # Normal equations, small p: solve with Gauss-Jordan on the augmented matrix.
    xtx = [[sum(design[i][a] * design[i][b] for i in range(n)) for b in range(p)] for a in range(p)]
    xty = [sum(design[i][a] * y[i] for i in range(n)) for a in range(p)]
    aug = [row[:] + [xty[a]] for a, row in enumerate(xtx)]
    for col in range(p):
        pivot = max(range(col, p), key=lambda r: abs(aug[r][col]))
        aug[col], aug[pivot] = aug[pivot], aug[col]
        scale = aug[col][col]
        aug[col] = [value / scale for value in aug[col]]
        for row in range(p):
            if row == col:
                continue
            factor = aug[row][col]
            aug[row] = [value - factor * aug[col][index] for index, value in enumerate(aug[row])]
    beta = [aug[a][p] for a in range(p)]
    fitted = [sum(design[i][a] * beta[a] for a in range(p)) for i in range(n)]
    residuals = [y[i] - fitted[i] for i in range(n)]
    dof = n - p
    sigma2 = sum(value * value for value in residuals) / dof
    # Inverse of XtX from the same elimination, recomputed for the diagonal.
    inverse = invert(xtx)
    se = [math.sqrt(sigma2 * inverse[a][a]) for a in range(p)]
    total = sum((value - statistics.fmean(y)) ** 2 for value in y)
    r2 = 1.0 - sum(value * value for value in residuals) / total
    return {
        "n": n,
        "r2": r2,
        "sigma_us": math.sqrt(sigma2),
        "terms": [
            {
                "name": name,
                "coef": beta[index],
                "se": se[index],
                "t": beta[index] / se[index] if se[index] else float("nan"),
            }
            for index, name in enumerate(["intercept"] + names)
        ],
    }


def invert(matrix: list[list[float]]) -> list[list[float]]:
    size = len(matrix)
    aug = [row[:] + [1.0 if i == j else 0.0 for j in range(size)] for i, row in enumerate(matrix)]
    for col in range(size):
        pivot = max(range(col, size), key=lambda r: abs(aug[r][col]))
        aug[col], aug[pivot] = aug[pivot], aug[col]
        scale = aug[col][col]
        aug[col] = [value / scale for value in aug[col]]
        for row in range(size):
            if row == col:
                continue
            factor = aug[row][col]
            aug[row] = [value - factor * aug[col][index] for index, value in enumerate(aug[row])]
    return [row[size:] for row in aug]


def boundary_contrast(rounds: list[dict], draws: int = 4000, seed: int = 173) -> dict:
    """Bootstrap the width increments, their second differences, and the
    boundary-against-neighbour contrast, resampling rounds within each
    (prompt, M) stratum."""
    import random

    generator = random.Random(seed)
    strata: dict[tuple[str, int], list[float]] = {}
    for row in rounds:
        strata.setdefault((row["prompt"], row["m_verify"]), []).append(row["round_us"])
    widths = sorted({width for _, width in strata})

    def medians(sample: dict[tuple[str, int], list[float]]) -> dict[int, float]:
        pooled: dict[int, list[float]] = {}
        for (_, width), values in sample.items():
            pooled.setdefault(width, []).extend(values)
        return {width: statistics.median(values) for width, values in pooled.items()}

    def summarise(sample: dict[tuple[str, int], list[float]]) -> dict:
        table = medians(sample)
        increments = {
            width: (table[width + 1] - table[width]) / 1e3
            for width in widths
            if width + 1 in table
        }
        curvature = {
            width: increments[width] - increments[width - 1]
            for width in increments
            if width - 1 in increments
        }
        contrast = (
            curvature.get(3, float("nan")) - curvature.get(4, float("nan"))
            if 3 in curvature and 4 in curvature
            else float("nan")
        )
        return {"increments_ms": increments, "curvature_ms": curvature, "contrast_ms": contrast}

    point = summarise(strata)
    draws_contrast: list[float] = []
    draws_curvature: dict[int, list[float]] = {}
    for _ in range(draws):
        sample = {
            key: [generator.choice(values) for _ in values] for key, values in strata.items()
        }
        result = summarise(sample)
        if not math.isnan(result["contrast_ms"]):
            draws_contrast.append(result["contrast_ms"])
        for width, value in result["curvature_ms"].items():
            draws_curvature.setdefault(width, []).append(value)

    def interval(values: list[float]) -> list[float]:
        ordered = sorted(values)
        low = ordered[int(0.025 * len(ordered))]
        high = ordered[int(0.975 * len(ordered)) - 1]
        return [low, high]

    return {
        "definition": "curvature(M) = inc(M -> M+1) - inc(M-1 -> M); "
        "contrast = curvature(3) - curvature(4); "
        "curvature(3) is the second difference that straddles tablePays",
        "point": point,
        "bootstrap_draws": len(draws_contrast),
        "contrast_ci95_ms": interval(draws_contrast) if draws_contrast else None,
        "curvature_ci95_ms": {
            width: interval(values) for width, values in sorted(draws_curvature.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path)
    parser.add_argument(
        "--drop-first",
        type=int,
        default=2,
        help="drop this many leading rounds per prompt (round 1 carries the "
        "first-round excess and warm-up compilation)",
    )
    args = parser.parse_args()

    rounds: list[dict] = []
    for prompt, relative in FILES.items():
        rows = parse(Path(relative), prompt)
        rounds.extend(row for row in rows if row["round"] > args.drop_first)

    report: dict = {
        "experiment": "e173-tablepays-natural-experiment",
        "harness": "local",
        "host_generation": "Apple M4 Pro",
        "source_files": FILES,
        "gate_qualified_for_timing": False,
        "cool_gate_passed_real_gate": False,
        "timing_claims_permitted": False,
        "trace_perturbs_timing": True,
        "rounds_used": len(rounds),
        "drop_first_rounds": args.drop_first,
    }

    # --- per-M summary, the same statistic FINDING 415 used.
    widths = sorted({row["m_verify"] for row in rounds})
    per_width = []
    for width in widths:
        subset = [row for row in rounds if row["m_verify"] == width]
        per_width.append(
            {
                "m_verify": width,
                "n": len(subset),
                "table_path": width >= 4,
                "median_round_ms": statistics.median(row["round_us"] for row in subset) / 1e3,
                "mean_round_ms": statistics.fmean(row["round_us"] for row in subset) / 1e3,
                "median_eval_wall_ms": statistics.median(row["eval_wall_us"] for row in subset)
                / 1e3,
                "median_verify_build_ms": statistics.median(
                    row["verify_build_us"] for row in subset
                )
                / 1e3,
                "median_host_tail_ms": statistics.median(
                    row["readout_us"] + row["commit_us"] + row["upkeep_us"] for row in subset
                )
                / 1e3,
                "mean_cache_pos": statistics.fmean(row["cache_pos"] for row in subset),
                "mean_acc": statistics.fmean(row["acc"] for row in subset),
                "mean_shortfall": statistics.fmean(row["shortfall"] for row in subset),
            }
        )
    report["per_width"] = per_width
    report["uncontrolled_increments_ms"] = [
        {
            "from": per_width[index]["m_verify"],
            "to": per_width[index + 1]["m_verify"],
            "median_increment_ms": per_width[index + 1]["median_round_ms"]
            - per_width[index]["median_round_ms"],
            "crosses_table_boundary": per_width[index]["m_verify"] == 3,
        }
        for index in range(len(per_width) - 1)
    ]

    # --- controlled step test. `step4` is the tablePays indicator. A cost that
    # switches on at m >= 4 shows up in `step4` and nowhere else.
    subset = [row for row in rounds if 2 <= row["m_verify"] <= 6]
    columns = {
        "m_verify": [float(row["m_verify"]) for row in subset],
        "step4": [1.0 if row["m_verify"] >= 4 else 0.0 for row in subset],
        "cache_pos": [float(row["cache_pos"]) for row in subset],
        "shortfall": [float(row["shortfall"]) for row in subset],
        "medicine": [1.0 if row["prompt"] == "medicine" else 0.0 for row in subset],
    }
    report["controlled_fits"] = {}
    for response in ("round_us", "eval_wall_us"):
        y = [row[response] for row in subset]
        report["controlled_fits"][response] = {
            "with_step": ols(y, columns),
            "without_step": ols(y, {k: v for k, v in columns.items() if k != "step4"}),
        }

    # --- host-side only. The tablePays branch adds host work (sidecar lookup,
    # table dispatch record, counter writes) inside the verify graph build.
    host = [row["draft_build_us"] + row["readout_us"] + row["commit_us"] + row["upkeep_us"] for row in subset]
    report["controlled_fits"]["host_phases_us"] = {"with_step": ols(host, columns)}

    # --- the local contrast. `step4` above is not separately identified from
    # the curvature of the width cost, because the M curve is convex over
    # 2...6. So compare the boundary increment with its NEIGHBOURING
    # increments: a cost that switches on at m >= 4 raises the 3 -> 4 second
    # difference above the 4 -> 5 one, whatever the smooth width curve does.
    report["boundary_contrast"] = boundary_contrast(rounds)

    # --- the same contrast with controls, restricted to M in {3, 4}: the two
    # bins that differ by one head step and the tablePays branch alone.
    pair = [row for row in rounds if row["m_verify"] in (3, 4)]
    pair_columns = {
        "is_m4": [1.0 if row["m_verify"] == 4 else 0.0 for row in pair],
        "cache_pos": [float(row["cache_pos"]) for row in pair],
        "shortfall": [float(row["shortfall"]) for row in pair],
        "medicine": [1.0 if row["prompt"] == "medicine" else 0.0 for row in pair],
    }
    report["restricted_m3_m4"] = {
        response: ols([row[response] for row in pair], pair_columns)
        for response in ("round_us", "eval_wall_us")
    }

    print(json.dumps(report, indent=2, sort_keys=True))
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
