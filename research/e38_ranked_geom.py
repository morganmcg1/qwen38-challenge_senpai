#!/usr/bin/env python3
"""E38 deliverable (d): does the ranked runner's MLX buffer geometry move the
shape-level cost curve?

Two `--shapes-only` cost curves at the same `BASE_SHA`, same host, same session:

  * `e38-base-r1`     -- architecture-default MLX buffer geometry
  * `e38-base-rg-r1`  -- ranked geometry (MLX_MAX_MB_PER_BUFFER=512,
                         MLX_MAX_OPS_PER_BUFFER=50)

If the curves agree, every local cost curve in this campaign -- including the
E38 arms -- carries over to the ranked runner at the shape level, and no future
experiment needs to pay for a ranked-geometry replay of a shape probe.

Three independent readings, because a bare paired mean cannot separate a real
geometry effect from monotone session drift:

  1. paired test over widths M>=3 (M<=2 excluded: measured warmup contamination)
  2. width trend -- buffer pressure grows with M, so a genuine geometry effect
     must correlate with M; pure drift must not
  3. comparison against the independently measured cross-session drift envelope

Self-test:  python3 research/e38_ranked_geom.py --self-test
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

CURVE_ROOT = pathlib.Path(".mlxfast-private/qmv-curve")

# Cross-session drift envelope, measured in `research/e38_anchor_check.py`:
# worst |delta| over M>=3 between this session's base curve and E33's.
DRIFT_ENVELOPE_PCT = 0.311

# Cost-curve minimum detectable effect from research/e39_mde.py, the regime
# this comparison lives in (per-width sd 0.344 %, 1 treated vs 6 controls).
COST_CURVE_MDE_NORMAL_PCT = 1.041
COST_CURVE_MDE_EXACT_PCT = 1.304

# Widths at or below this are warmup/JIT contaminated (E38 pre-flight: M=1
# spread 13.8-39.5 %, M=2 5.8-11.1 %) and are excluded from every estimate.
MIN_TRUSTED_WIDTH = 3


def load_summary(tag: str, root: pathlib.Path = CURVE_ROOT) -> dict:
    return json.loads((root / tag / "summary.json").read_text())


def ladder_ms(summary: dict) -> dict[int, float]:
    """{verify_width: per-round GEMM milliseconds}."""
    return {
        int(r["verify_width"]): 1000.0 * float(r["gemm_seconds"])
        for r in summary["round_cost_model"]["rows"]
    }


def per_shape_us(summary: dict, width: int) -> dict[str, float]:
    return {
        row["name"]: 1e6 * float(row["seconds_per_call"])
        for row in summary["per_shape_curve"]
        if int(row["m"]) == width
    }


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def sample_sd(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mu = mean(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1))


def pearson_r(xs: list[float], ys: list[float]) -> float:
    """Pearson correlation; 0.0 when either side is constant."""
    mx, my = mean(xs), mean(ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0.0 or syy <= 0.0:
        return 0.0
    return sxy / math.sqrt(sxx * syy)


def compare(arch: dict, ranked: dict) -> dict:
    la, lr = ladder_ms(arch), ladder_ms(ranked)
    widths = sorted(set(la) & set(lr))

    per_width = []
    for m in widths:
        delta_pct = 100.0 * (lr[m] / la[m] - 1.0)
        per_width.append(
            {
                "verify_width": m,
                "arch_ms": la[m],
                "ranked_ms": lr[m],
                "ratio": lr[m] / la[m],
                "delta_pct": delta_pct,
                "trusted": m >= MIN_TRUSTED_WIDTH,
            }
        )

    trusted = [r for r in per_width if r["trusted"]]
    deltas = [r["delta_pct"] for r in trusted]
    ms = [float(r["verify_width"]) for r in trusted]

    n = len(deltas)
    mu = mean(deltas)
    sd = sample_sd(deltas)
    se = sd / math.sqrt(n) if n > 1 else float("inf")
    t_stat = mu / se if se > 0 else float("inf")
    n_positive = sum(1 for d in deltas if d > 0.0)

    r_width = pearson_r(ms, deltas)

    return {
        "per_width": per_width,
        "trusted_widths": [int(m) for m in ms],
        "n_trusted": n,
        "mean_delta_pct": mu,
        "sd_delta_pp": sd,
        "se_delta_pp": se,
        "t_stat": t_stat,
        "df": n - 1,
        "abs_max_delta_pct": max(abs(d) for d in deltas),
        "n_positive_of_n": [n_positive, n],
        "width_trend_pearson_r": r_width,
        "drift_envelope_pct": DRIFT_ENVELOPE_PCT,
        "within_drift_envelope": max(abs(d) for d in deltas) <= DRIFT_ENVELOPE_PCT
        or abs(mu) <= DRIFT_ENVELOPE_PCT,
        "cost_curve_mde_normal_pct": COST_CURVE_MDE_NORMAL_PCT,
        "cost_curve_mde_exact_pct": COST_CURVE_MDE_EXACT_PCT,
        "mean_over_mde_normal": abs(mu) / COST_CURVE_MDE_NORMAL_PCT,
        "base_sha_match": arch.get("base_sha") == ranked.get("base_sha"),
        "host_match": arch.get("host") == ranked.get("host"),
        "arch_base_sha": arch.get("base_sha"),
        "ranked_base_sha": ranked.get("base_sha"),
        "host": arch.get("host"),
    }


def verdict(res: dict) -> str:
    if not res["base_sha_match"] or not res["host_match"]:
        return "INVALID: curves differ in base_sha or host"
    if abs(res["mean_delta_pct"]) > res["cost_curve_mde_normal_pct"]:
        return "GEOMETRY MATTERS: shape-level curves do not transfer"
    if abs(res["width_trend_pearson_r"]) >= 0.7:
        return "AMBIGUOUS: sub-MDE mean but a width-ordered residual"
    return (
        "GEOMETRY-INVARIANT: shape-level cost curves transfer to ranked geometry; "
        "residual is drift-shaped (no width ordering) and far below the curve MDE"
    )


def self_test() -> int:
    checks: list[tuple[str, bool]] = []

    checks.append(("mean", abs(mean([1.0, 2.0, 3.0]) - 2.0) < 1e-12))
    checks.append(("sd", abs(sample_sd([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]) - 2.13809) < 1e-4))
    checks.append(("sd_singleton", sample_sd([3.0]) == 0.0))
    checks.append(("pearson_perfect", abs(pearson_r([1, 2, 3], [2, 4, 6]) - 1.0) < 1e-12))
    checks.append(("pearson_anti", abs(pearson_r([1, 2, 3], [6, 4, 2]) + 1.0) < 1e-12))
    checks.append(("pearson_flat", pearson_r([1, 2, 3], [5, 5, 5]) == 0.0))

    # A synthetic pure width-proportional effect must be caught by the trend arm.
    arch = {
        "base_sha": "x",
        "host": "h",
        "round_cost_model": {
            "rows": [{"verify_width": m, "gemm_seconds": 0.001 * m} for m in range(1, 10)]
        },
        "per_shape_curve": [],
    }
    ranked_trend = {
        "base_sha": "x",
        "host": "h",
        "round_cost_model": {
            "rows": [
                {"verify_width": m, "gemm_seconds": 0.001 * m * (1.0 + 0.001 * m)}
                for m in range(1, 10)
            ]
        },
        "per_shape_curve": [],
    }
    r_trend = compare(arch, ranked_trend)
    checks.append(("trend_detected", r_trend["width_trend_pearson_r"] > 0.99))
    checks.append(("trend_flagged", "AMBIGUOUS" in verdict(r_trend)))

    # A flat 5 % offset must trip the MDE arm regardless of trend.
    ranked_big = {
        "base_sha": "x",
        "host": "h",
        "round_cost_model": {
            "rows": [{"verify_width": m, "gemm_seconds": 0.00105 * m} for m in range(1, 10)]
        },
        "per_shape_curve": [],
    }
    r_big = compare(arch, ranked_big)
    checks.append(("offset_mean", abs(r_big["mean_delta_pct"] - 5.0) < 1e-9))
    checks.append(("offset_flagged", "GEOMETRY MATTERS" in verdict(r_big)))

    # Identical curves must be declared invariant, with M<=2 excluded.
    r_same = compare(arch, arch)
    checks.append(("null_mean_zero", abs(r_same["mean_delta_pct"]) < 1e-12))
    checks.append(("null_invariant", "GEOMETRY-INVARIANT" in verdict(r_same)))
    checks.append(("excludes_low_widths", r_same["trusted_widths"] == [3, 4, 5, 6, 7, 8, 9]))
    checks.append(("mismatched_base_invalid", "INVALID" in verdict(compare(arch, {**arch, "base_sha": "y"}))))

    for name, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    failed = [n for n, ok in checks if not ok]
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed")
    return 1 if failed else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="e38-base-r1")
    ap.add_argument("--ranked", default="e38-base-rg-r1")
    ap.add_argument("--json-out")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    res = compare(load_summary(args.arch), load_summary(args.ranked))

    print(f"arch curve   : {args.arch}")
    print(f"ranked curve : {args.ranked}")
    print(f"base_sha     : {res['arch_base_sha']}  (match={res['base_sha_match']})")
    print(f"host         : {res['host']}  (match={res['host_match']})\n")
    print(f"{'M':>3} {'arch ms':>10} {'ranked ms':>10} {'delta%':>9}  trusted")
    for row in res["per_width"]:
        print(
            f"{row['verify_width']:>3} {row['arch_ms']:10.3f} {row['ranked_ms']:10.3f} "
            f"{row['delta_pct']:+9.3f}  {'yes' if row['trusted'] else 'no (warmup)'}"
        )
    print(
        f"\nM>={MIN_TRUSTED_WIDTH}: mean {res['mean_delta_pct']:+.3f}%  sd {res['sd_delta_pp']:.3f} pp  "
        f"se {res['se_delta_pp']:.3f} pp  t {res['t_stat']:+.2f} (df {res['df']})"
    )
    print(f"  worst |delta|          {res['abs_max_delta_pct']:.3f}%")
    print(f"  positive signs         {res['n_positive_of_n'][0]}/{res['n_positive_of_n'][1]}")
    print(f"  width-trend Pearson r  {res['width_trend_pearson_r']:+.3f}")
    print(f"  drift envelope (E33)   {res['drift_envelope_pct']:.3f}%")
    print(
        f"  cost-curve MDE         {res['cost_curve_mde_normal_pct']:.3f}% normal / "
        f"{res['cost_curve_mde_exact_pct']:.3f}% exact  "
        f"-> mean is {res['mean_over_mde_normal']:.2f}x the normal MDE"
    )
    print(f"\nVERDICT: {verdict(res)}")

    if args.json_out:
        out = pathlib.Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({**res, "verdict": verdict(res)}, indent=2) + "\n")
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
