#!/usr/bin/env python3
"""Analyse one E62 session with the campaign's standard estimator.

The model is `time ~ arm + leg_position`, fitted by ordinary least squares over
the legs of one counterbalanced session. `leg_position` is centred, so the arm
coefficients are the contrast at the middle of the session and monotone thermal
drift is absorbed by one slope instead of being confounded with an arm.

  research/e62_analyze.py --session r1-ops --reference ops50 \
      --out research/e62-artifacts/e62-r1-ops.json

The null floor is not a single number. It scales with leg separation, so this
also reports the observed same-arm spread at every separation present in the
session; a contrast is only interesting when it clears the same-arm spread at
the separation its own legs actually had.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import pathlib

import numpy as np

OUT_ROOT = pathlib.Path("research/out")


def read_meta(leg: pathlib.Path) -> dict:
    meta: dict[str, str] = {}
    path = leg / "meta.txt"
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                meta[key.strip()] = value.strip()
    return meta


def load_session(session: str) -> list[dict]:
    legs = []
    for directory in sorted(OUT_ROOT.glob(f"e62-{session}-*")):
        meta = read_meta(directory)
        score = directory / "score.json"
        if not score.exists():
            print(f"skip {directory.name}: no score.json")
            continue
        metrics = json.loads(score.read_text())["metrics"]
        position = int(directory.name.split("-")[2])
        legs.append(
            {
                "tag": directory.name,
                "arm": meta.get("label", "?"),
                "position": position,
                "mb": int(meta.get("mlx_max_mb_per_buffer", 0) or 0),
                "ops": int(meta.get("mlx_max_ops_per_buffer", 0) or 0),
                "wired_residency_active": meta.get("wired_residency_active") == "true",
                "gpu_temp_entry_c": float(meta.get("gpu_temp_entry_c", "nan")),
                "gpu_temp_exit_c": float(meta.get("gpu_temp_exit_c", "nan")),
                "worker_peak_rss_gb": float(meta.get("worker_peak_rss_gb", "nan")),
                "mtp_seconds_per_token": metrics["mtp_seconds_per_token"],
                "serial_seconds_per_token": metrics["serial_seconds_per_token"],
                "mtp_decode_speedup": metrics["mtp_decode_speedup"],
                "all_tokens_matched": metrics["all_tokens_matched"],
                "residual_divergence_count": metrics["residual_divergence_count"],
                "effective_mean_draft_len": metrics["effective_mean_draft_len"],
                "accepted_draft_rate": metrics["accepted_draft_rate"],
            }
        )
    return legs


def student_t_975(dof: int) -> float:
    """Two-sided 95 % t quantile without a SciPy dependency."""
    table = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
        7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
        13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
        19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064,
        25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
    }
    if dof <= 0:
        return float("nan")
    if dof in table:
        return table[dof]
    return 1.960 + 2.4 / dof


def regress(legs: list[dict], reference: str, metric: str) -> dict:
    arms = sorted({leg["arm"] for leg in legs})
    if reference not in arms:
        raise SystemExit(f"e62: reference arm {reference!r} not in {arms}")
    others = [arm for arm in arms if arm != reference]
    positions = np.array([leg["position"] for leg in legs], dtype=float)
    centred = positions - positions.mean()
    columns = [np.ones(len(legs)), centred]
    names = ["intercept", "leg_position"]
    for arm in others:
        columns.append(
            np.array([1.0 if leg["arm"] == arm else 0.0 for leg in legs])
        )
        names.append(f"arm[{arm}]")
    design = np.column_stack(columns)
    response = np.array([leg[metric] for leg in legs], dtype=float)

    coefficients, *_ = np.linalg.lstsq(design, response, rcond=None)
    residuals = response - design @ coefficients
    dof = len(legs) - design.shape[1]
    if dof <= 0:
        raise SystemExit("e62: not enough legs to fit this model")
    residual_var = float(residuals @ residuals) / dof
    covariance = residual_var * np.linalg.pinv(design.T @ design)
    standard_errors = np.sqrt(np.diag(covariance))
    critical = student_t_975(dof)

    reference_mean = float(
        np.mean([leg[metric] for leg in legs if leg["arm"] == reference])
    )
    terms = {}
    for index, name in enumerate(names):
        estimate = float(coefficients[index])
        se = float(standard_errors[index])
        terms[name] = {
            "estimate": estimate,
            "se": se,
            "t": estimate / se if se else float("nan"),
            "ci95_low": estimate - critical * se,
            "ci95_high": estimate + critical * se,
        }
        if name.startswith("arm["):
            terms[name]["percent_of_reference"] = 100.0 * estimate / reference_mean
            terms[name]["percent_ci95_low"] = (
                100.0 * (estimate - critical * se) / reference_mean
            )
            terms[name]["percent_ci95_high"] = (
                100.0 * (estimate + critical * se) / reference_mean
            )
    return {
        "metric": metric,
        "reference_arm": reference,
        "reference_mean": reference_mean,
        "n_legs": len(legs),
        "dof": dof,
        "residual_sd": math.sqrt(residual_var),
        "residual_sd_percent_of_reference": 100.0
        * math.sqrt(residual_var)
        / reference_mean,
        "t_critical_95": critical,
        "terms": terms,
    }


def load_census(path: pathlib.Path) -> dict[tuple[int, int], float]:
    """Map (MB, OPS) to the measured commits per decode round."""
    payload = json.loads(path.read_text())
    return {
        (leg["mb"], leg["ops"]): leg["commits_per_round"]
        for leg in payload["legs"]
    }


def trend(legs: list[dict], metric: str, mb: int) -> dict:
    """Fit `time ~ commits_per_round + leg_position` across one MB ladder.

    The covariate is the *measured* commit count, not `log2(OPS)`. The census
    shows total dispatches are invariant to 0.010 % across a 24x change in
    commit count, so a geometry change repackages identical work. If a command
    buffer commit carries a fixed cost `c`, then decode time is linear in the
    number of commits and this slope estimates `c` directly. `log2(OPS)` has no
    such reading, and the OPS-to-commit map is not even log-linear.

    Seven arms with one replicate pair each give a weak pairwise test but a
    well-powered test of a monotone tilt, because every leg informs the one
    slope. This is the primary screen; the per-arm contrasts stay descriptive.
    """
    ladder = [
        leg for leg in legs
        if leg["mb"] == mb and leg.get("commits_per_round") is not None
    ]
    distinct = sorted({leg["ops"] for leg in ladder})
    if len(distinct) < 3:
        return {"skipped": f"need >=3 distinct ops at mb={mb}, saw {distinct}"}
    positions = np.array([leg["position"] for leg in ladder], dtype=float)
    commits = np.array([leg["commits_per_round"] for leg in ladder])
    design = np.column_stack(
        [np.ones(len(ladder)), commits - commits.mean(),
         positions - positions.mean()]
    )
    response = np.array([leg[metric] for leg in ladder], dtype=float)
    coefficients, *_ = np.linalg.lstsq(design, response, rcond=None)
    residuals = response - design @ coefficients
    dof = len(ladder) - design.shape[1]
    if dof <= 0:
        return {"skipped": "not enough ladder legs"}
    residual_var = float(residuals @ residuals) / dof
    covariance = residual_var * np.linalg.pinv(design.T @ design)
    se = float(np.sqrt(np.diag(covariance))[1])
    estimate = float(coefficients[1])
    critical = student_t_975(dof)
    mean = float(response.mean())
    span = float(commits.max() - commits.min())
    return {
        "mb": mb,
        "covariate": "commits_per_round (measured by census)",
        "ops_points": distinct,
        "commits_per_round_span": [float(commits.min()), float(commits.max())],
        "n_legs": len(ladder),
        "dof": dof,
        "mean": mean,
        "residual_sd_percent": 100.0 * math.sqrt(residual_var) / mean,
        "seconds_per_token_per_commit_per_round": estimate,
        "se": se,
        "t": estimate / se if se else float("nan"),
        "ci95_low": estimate - critical * se,
        "ci95_high": estimate + critical * se,
        "percent_across_full_ladder": 100.0 * estimate * span / mean,
        "percent_across_full_ladder_ci95_low":
            100.0 * (estimate - critical * se) * span / mean,
        "percent_across_full_ladder_ci95_high":
            100.0 * (estimate + critical * se) * span / mean,
    }


def same_arm_spreads(legs: list[dict], metric: str) -> dict:
    """Observed |a-b|/mean for every same-arm pair, keyed by leg separation."""
    by_separation: dict[int, list[float]] = {}
    for left, right in itertools.combinations(legs, 2):
        if left["arm"] != right["arm"]:
            continue
        separation = abs(left["position"] - right["position"])
        mean = 0.5 * (left[metric] + right[metric])
        spread = 100.0 * abs(left[metric] - right[metric]) / mean
        by_separation.setdefault(separation, []).append(spread)
    return {
        str(separation): {
            "n_pairs": len(values),
            "max_percent": max(values),
            "mean_percent": sum(values) / len(values),
            "values_percent": values,
        }
        for separation, values in sorted(by_separation.items())
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--metric", default="mtp_seconds_per_token")
    parser.add_argument("--drop", nargs="*", default=[],
                        help="leg tags to exclude, e.g. a declared warm-up")
    parser.add_argument("--trend-mb", type=int, default=0,
                        help="fit the ladder slope across legs at this MB")
    parser.add_argument("--census", default="research/e62-artifacts/e62-census.json",
                        help="census file supplying measured commits per round")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    legs = [leg for leg in load_session(args.session) if leg["tag"] not in args.drop]
    census_path = pathlib.Path(args.census)
    if census_path.exists():
        census = load_census(census_path)
        missing = set()
        for leg in legs:
            key = (leg["mb"], leg["ops"])
            leg["commits_per_round"] = census.get(key)
            if key not in census:
                missing.add(key)
        if missing:
            print(f"warning: no census geometry for {sorted(missing)}")
    if not legs:
        raise SystemExit(f"e62: no legs found for session {args.session}")
    entry = [leg["gpu_temp_entry_c"] for leg in legs]
    payload = {
        "session": args.session,
        "harness": "local",
        "local_mode": "--local-iterate",
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "dropped_legs": args.drop,
        "entry_temperatures_c": entry,
        "entry_temperature_spread_c": max(entry) - min(entry),
        "exactness": {
            leg["tag"]: {
                "all_tokens_matched": leg["all_tokens_matched"],
                "residual_divergence_count": leg["residual_divergence_count"],
            }
            for leg in legs
        },
        "legs": legs,
        "arm_means": {},
        "same_arm_spreads_percent": same_arm_spreads(legs, args.metric),
        "regression": regress(legs, args.reference, args.metric),
    }
    if args.trend_mb:
        payload["ladder_trend"] = trend(legs, args.metric, args.trend_mb)
    for arm in sorted({leg["arm"] for leg in legs}):
        values = [leg[args.metric] for leg in legs if leg["arm"] == arm]
        payload["arm_means"][arm] = {
            "n": len(values),
            "mean": sum(values) / len(values),
            "values": values,
        }
    reference_mean = payload["arm_means"][args.reference]["mean"]
    for arm, stats in payload["arm_means"].items():
        stats["percent_vs_reference"] = 100.0 * (stats["mean"] - reference_mean) / reference_mean

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1) + "\n")
    print(json.dumps(payload["regression"], indent=1))
    print(json.dumps(payload["arm_means"], indent=1))
    print(json.dumps(payload["same_arm_spreads_percent"], indent=1))
    if "ladder_trend" in payload:
        print(json.dumps(payload["ladder_trend"], indent=1))
    print(f"entry temperature spread: {payload['entry_temperature_spread_c']:.3f} C")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
