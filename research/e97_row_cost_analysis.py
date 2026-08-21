#!/usr/bin/env python3
"""Fit the E97 rung-1 per-row slopes and price what a marginal verify row buys.

`Tests/MLXFastTests/E97VerifyRowCostTests.swift` writes one JSON of cells. Each
cell is a mean over `replicates` blocking evals of one (kernel, shape, width) at
one counterbalanced block. This script fits

    us(M) = intercept + slope * M

separately for each kernel, shape and dispatch regime, and reports the slope,
its standard error, the ratio between the quantized and the bf16 arm, and the
per-row arithmetic rate each slope implies.

The intercepts of the two arms are NOT comparable: the bf16 arm streams four
times the weight bytes. The slopes are, because in both arms the weight stream
is amortised across the input rows of one dispatch.

  usage: research/e97_row_cost_analysis.py [PATH] [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

HIDDEN = 5120
VECTOR_LIMIT = 10


def inputs_per_group(width: int) -> int:
    """The shipped rule in `kernels/quantized.h`: IPG = ceil(M / ceil(M/4))."""
    return -(-width // -(-width // 4))


def groups(width: int) -> int:
    return -(-width // inputs_per_group(width))


def fit(points: list[tuple[float, float]]) -> dict:
    """OLS of y on x with the textbook standard errors."""
    n = len(points)
    if n < 3:
        raise ValueError("need at least three points")
    mean_x = sum(x for x, _ in points) / n
    mean_y = sum(y for _, y in points) / n
    sxx = sum((x - mean_x) ** 2 for x, _ in points)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in points)
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    rss = sum((y - intercept - slope * x) ** 2 for x, y in points)
    sigma2 = rss / (n - 2)
    se_slope = math.sqrt(sigma2 / sxx)
    se_intercept = math.sqrt(sigma2 * (1.0 / n + mean_x**2 / sxx))
    tss = sum((y - mean_y) ** 2 for _, y in points)
    return {
        "n": n,
        "slope_us_per_row": slope,
        "se_slope_us_per_row": se_slope,
        "intercept_us": intercept,
        "se_intercept_us": se_intercept,
        "residual_sd_us": math.sqrt(sigma2),
        "r_squared": 1.0 - rss / tss if tss > 0 else float("nan"),
    }


def flops_per_row(outputs: int) -> float:
    """One multiply and one add for every weight, for one input row."""
    return 2.0 * HIDDEN * outputs


def analyse(payload: dict) -> dict:
    cells = payload["cells"]
    overhead = payload["eval_overhead_us"]
    shapes = sorted({c["outputs"] for c in cells})
    kernels = sorted({c["kernel"] for c in cells})

    fits: dict[str, dict] = {}
    for outputs in shapes:
        for kernel in kernels:
            for regime, keep in (
                ("vector", lambda m: m < VECTOR_LIMIT),
                ("matrix", lambda m: m >= VECTOR_LIMIT),
            ):
                selected = [
                    c
                    for c in cells
                    if c["outputs"] == outputs
                    and c["kernel"] == kernel
                    and keep(c["m"])
                ]
                if len(selected) < 3:
                    continue
                key = f"{kernel}/{outputs}/{regime}"
                record = fit([(float(c["m"]), c["us"]) for c in selected])
                record.update(
                    kernel=kernel,
                    outputs=outputs,
                    regime=regime,
                    widths=sorted({c["m"] for c in selected}),
                    flops_per_row=flops_per_row(outputs),
                    tflop_per_s=flops_per_row(outputs)
                    / (record["slope_us_per_row"] * 1e-6)
                    / 1e12,
                    intercept_net_us=record["intercept_us"] - overhead,
                )
                for direction, up in (("ascending", True), ("descending", False)):
                    part = [c for c in selected if c["ascending"] == up]
                    if len(part) >= 3:
                        record[f"slope_{direction}"] = fit(
                            [(float(c["m"]), c["us"]) for c in part]
                        )["slope_us_per_row"]
                if "slope_ascending" in record and "slope_descending" in record:
                    mean_slope = 0.5 * (
                        record["slope_ascending"] + record["slope_descending"]
                    )
                    record["drift_gap_pct"] = (
                        100.0
                        * (record["slope_ascending"] - record["slope_descending"])
                        / mean_slope
                    )
                fits[key] = record

    ratios = {}
    for outputs in shapes:
        for regime in ("vector", "matrix"):
            quantized = fits.get(f"affine4/{outputs}/{regime}")
            dense = fits.get(f"bf16/{outputs}/{regime}")
            if not quantized or not dense:
                continue
            q = quantized["slope_us_per_row"]
            d = dense["slope_us_per_row"]
            ratio = q / d
            rel = math.sqrt(
                (quantized["se_slope_us_per_row"] / q) ** 2
                + (dense["se_slope_us_per_row"] / d) ** 2
            )
            ratios[f"{outputs}/{regime}"] = {
                "outputs": outputs,
                "regime": regime,
                "slope_affine4_us": q,
                "slope_bf16_us": d,
                "ratio": ratio,
                "se_ratio": ratio * rel,
                "gap_pct": 100.0 * (q - d) / d,
                # Rung 1 stop rule: agreement within 15 % refutes (A).
                "agree_within_15pct": abs(100.0 * (q - d) / d) <= 15.0,
            }

    # Rung 3. The ranked curve fits one line per group band, so the local slope
    # has to be read the same way. M = 1 is excluded: `ntg.x == 1` has no case
    # in the WIDE switch, so it runs the generic qmv, not the cross-row body.
    band_fits: dict[str, dict] = {}
    for outputs in shapes:
        for kernel in kernels:
            for band, widths in (("G1", [2, 3, 4]), ("G2", [5, 6, 7, 8])):
                selected = [
                    c
                    for c in cells
                    if c["outputs"] == outputs
                    and c["kernel"] == kernel
                    and c["m"] in widths
                ]
                if len(selected) < 3:
                    continue
                record = fit([(float(c["m"]), c["us"]) for c in selected])
                record.update(
                    kernel=kernel,
                    outputs=outputs,
                    band=band,
                    widths=widths,
                    tflop_per_s=flops_per_row(outputs)
                    / (record["slope_us_per_row"] * 1e-6)
                    / 1e12,
                )
                band_fits[f"{kernel}/{outputs}/{band}"] = record

    band_step = {}
    for outputs in shapes:
        for kernel in kernels:
            g1 = band_fits.get(f"{kernel}/{outputs}/G1")
            g2 = band_fits.get(f"{kernel}/{outputs}/G2")
            if not g1 or not g2:
                continue
            band_step[f"{kernel}/{outputs}"] = {
                "slope_g1_us": g1["slope_us_per_row"],
                "se_slope_g1_us": g1["se_slope_us_per_row"],
                "slope_g2_us": g2["slope_us_per_row"],
                "se_slope_g2_us": g2["se_slope_us_per_row"],
                "g2_over_g1": g2["slope_us_per_row"] / g1["slope_us_per_row"],
                "intercept_g1_us": g1["intercept_us"],
                "intercept_g2_us": g2["intercept_us"],
                "intercept_step_us": g2["intercept_us"] - g1["intercept_us"],
            }

    regime_step = {}
    for outputs in shapes:
        for kernel in kernels:
            vector = fits.get(f"{kernel}/{outputs}/vector")
            matrix = fits.get(f"{kernel}/{outputs}/matrix")
            if not vector or not matrix:
                continue
            regime_step[f"{kernel}/{outputs}"] = {
                "slope_vector_us": vector["slope_us_per_row"],
                "slope_matrix_us": matrix["slope_us_per_row"],
                "matrix_over_vector": matrix["slope_us_per_row"]
                / vector["slope_us_per_row"],
                "tflop_per_s_vector": vector["tflop_per_s"],
                "tflop_per_s_matrix": matrix["tflop_per_s"],
            }

    per_width: dict[str, dict] = {}
    for outputs in shapes:
        for kernel in kernels:
            for width in sorted({c["m"] for c in cells}):
                selected = [
                    c
                    for c in cells
                    if c["outputs"] == outputs
                    and c["kernel"] == kernel
                    and c["m"] == width
                ]
                if not selected:
                    continue
                values = sorted(c["us"] for c in selected)
                mean = sum(values) / len(values)
                spread = (
                    max(values) - min(values)
                ) / mean * 100.0 if mean else float("nan")
                per_width[f"{kernel}/{outputs}/{width}"] = {
                    "kernel": kernel,
                    "outputs": outputs,
                    "m": width,
                    "groups": groups(width) if width < VECTOR_LIMIT else 0,
                    "inputs_per_group": (
                        inputs_per_group(width) if width < VECTOR_LIMIT else 0
                    ),
                    "mean_us": mean,
                    "net_us": mean - overhead,
                    "range_pct": spread,
                    "blocks": len(values),
                }

    nulls = {}
    for outputs in shapes:
        opened = next(
            (
                n
                for n in payload["nulls"]
                if n["outputs"] == outputs and n["label"] == "session_open"
            ),
            None,
        )
        closed = next(
            (
                n
                for n in payload["nulls"]
                if n["outputs"] == outputs and n["label"] == "session_close"
            ),
            None,
        )
        if opened and closed:
            nulls[str(outputs)] = {
                "open_us": opened["us"],
                "close_us": closed["us"],
                "drift_pct": 100.0 * (closed["us"] - opened["us"]) / opened["us"],
            }

    return {
        "eval_overhead_us": overhead,
        "eval_overhead_close_us": payload["eval_overhead_close_us"],
        "reads": payload["reads"],
        "fits": fits,
        "band_fits": band_fits,
        "band_step": band_step,
        "ratios": ratios,
        "regime_step": regime_step,
        "session_null": nulls,
        "per_width_mean_us": per_width,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path", nargs="?", default="research/out/e97-row-cost-r1/row-cost.json"
    )
    parser.add_argument("--json", default="")
    args = parser.parse_args()

    payload = json.loads(pathlib.Path(args.path).read_text())
    result = analyse(payload)

    print(f"eval_overhead_us={result['eval_overhead_us']:.2f} "
          f"close={result['eval_overhead_close_us']:.2f}")
    print("\nread rate of each working set")
    for read in result["reads"]:
        print(
            f"  {read['kernel']:<8} O={read['outputs']:<7} "
            f"bytes={read['bytes']:>11} net_us={read['net_us']:>9.1f} "
            f"GB/s={read['gb_per_s']:>7.1f}"
        )

    print("\nper-row slope")
    header = (
        f"  {'arm':<26}{'slope us':>10}{'se':>8}{'TFLOP/s':>9}"
        f"{'intercept':>11}{'R2':>7}{'drift %':>9}"
    )
    print(header)
    for key in sorted(result["fits"]):
        f = result["fits"][key]
        print(
            f"  {key:<26}{f['slope_us_per_row']:>10.2f}"
            f"{f['se_slope_us_per_row']:>8.2f}{f['tflop_per_s']:>9.2f}"
            f"{f['intercept_us']:>11.1f}{f['r_squared']:>7.3f}"
            f"{f.get('drift_gap_pct', float('nan')):>9.2f}"
        )

    print("\nmean over blocks, per width")
    print(
        f"  {'arm':<20}{'M':>4}{'G':>3}{'IPG':>5}{'mean us':>11}"
        f"{'net us':>11}{'range %':>9}"
    )
    for key in sorted(
        result["per_width_mean_us"],
        key=lambda k: (
            result["per_width_mean_us"][k]["kernel"],
            result["per_width_mean_us"][k]["outputs"],
            result["per_width_mean_us"][k]["m"],
        ),
    ):
        w = result["per_width_mean_us"][key]
        print(
            f"  {w['kernel'] + '/' + str(w['outputs']):<20}{w['m']:>4}"
            f"{w['groups']:>3}{w['inputs_per_group']:>5}{w['mean_us']:>11.1f}"
            f"{w['net_us']:>11.1f}{w['range_pct']:>9.2f}"
        )

    print("\nper-row slope inside each group band (rung 3)")
    for key in sorted(result["band_step"]):
        b = result["band_step"][key]
        print(
            f"  {key:<18} G1={b['slope_g1_us']:.2f}+/-{b['se_slope_g1_us']:.2f} us "
            f"G2={b['slope_g2_us']:.2f}+/-{b['se_slope_g2_us']:.2f} us "
            f"G2/G1={b['g2_over_g1']:.3f} "
            f"intercept step={b['intercept_step_us']:+.1f} us"
        )

    print("\naffine4 over bf16")
    for key in sorted(result["ratios"]):
        r = result["ratios"][key]
        print(
            f"  {key:<18} ratio={r['ratio']:.3f} +/- {r['se_ratio']:.3f} "
            f"gap={r['gap_pct']:+.1f} % "
            f"{'AGREE<=15%' if r['agree_within_15pct'] else 'DIFFER>15%'}"
        )

    print("\nvector regime versus split-K matrix regime")
    for key in sorted(result["regime_step"]):
        s = result["regime_step"][key]
        print(
            f"  {key:<18} vector={s['slope_vector_us']:.2f} us "
            f"matrix={s['slope_matrix_us']:.2f} us "
            f"ratio={s['matrix_over_vector']:.3f} "
            f"({s['tflop_per_s_vector']:.2f} -> {s['tflop_per_s_matrix']:.2f} TFLOP/s)"
        )

    print("\nsame-arm session null (affine4, M=4)")
    for key, null in result["session_null"].items():
        print(
            f"  O={key:<8} open={null['open_us']:.1f} us "
            f"close={null['close_us']:.1f} us drift={null['drift_pct']:+.2f} %"
        )

    if args.json:
        out = pathlib.Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
