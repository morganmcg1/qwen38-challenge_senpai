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
            # M = 1 is excluded from the vector regime in BOTH arms. The bf16
            # arm leaves gemv for the steel GEMM at M = 2 and the affine-4 arm
            # has no `ntg.x == 1` case in the WIDE switch, so a fit that spans
            # M = 1 straddles a kernel change and its slope is meaningless.
            for regime, keep in (
                ("vector2_9", lambda m: 2 <= m < VECTOR_LIMIT),
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
        for regime in ("vector2_9", "matrix"):
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
            vector = fits.get(f"{kernel}/{outputs}/vector2_9")
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

    # Model-free rung-3 evidence. A band fit averages over the template changes
    # inside the band, so the step at a group boundary is only visible width by
    # width. Each increment is also priced two ways: as arithmetic, and as a
    # fraction of the single-group weight read measured at M = 1.
    increments: dict[str, dict] = {}
    for outputs in shapes:
        for kernel in kernels:
            means = {
                w["m"]: w["mean_us"]
                for key, w in per_width.items()
                if w["kernel"] == kernel and w["outputs"] == outputs
            }
            base = means.get(1)
            for width in sorted(means):
                if width - 1 not in means:
                    continue
                step = means[width] - means[width - 1]
                record = {
                    "kernel": kernel,
                    "outputs": outputs,
                    "from_m": width - 1,
                    "to_m": width,
                    "step_us": step,
                    "crosses_group": groups(width) != groups(width - 1)
                    if width < VECTOR_LIMIT
                    else None,
                    "ipg_from": inputs_per_group(width - 1)
                    if width - 1 < VECTOR_LIMIT
                    else None,
                    "ipg_to": inputs_per_group(width)
                    if width < VECTOR_LIMIT
                    else None,
                    "tflop_per_s_if_arithmetic": flops_per_row(outputs)
                    / (step * 1e-6)
                    / 1e12
                    if step > 0
                    else float("nan"),
                }
                if base:
                    record["fraction_of_m1_weight_read"] = step / (
                        base - overhead
                    )
                increments[f"{kernel}/{outputs}/{width - 1}to{width}"] = record

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
        "increments": increments,
    }


def peak_ceilings(peak: dict) -> dict:
    """Rung 0. The best rate this GPU actually reached, per weight form.

    `affine4` is the ceiling the scored kernel is allowed to be compared
    against, because it already pays dequantisation; `dense` is the ceiling of
    the same machine with no unpacking at all.
    """
    best: dict[str, dict] = {}
    for record in peak["records"]:
        form = "affine4" if record["label"] == "affine4_batch" else "dense"
        if form not in best or record["tflop_per_s"] > best[form]["tflop_per_s"]:
            best[form] = record
    return {
        form: {
            "tflop_per_s": record["tflop_per_s"],
            "label": record["label"],
            "shape": record["shape"],
            "dtype": record["dtype"],
        }
        for form, record in best.items()
    }


def against_peak(band_fits: dict, ceilings: dict) -> dict:
    """Read every band slope as a fraction of a measured ceiling.

    A marginal row buys `2 * K * N` multiply-accumulates and nothing else, so
    `slope / (2 K N)` is directly comparable with a GEMM rate on the same GPU.
    A high fraction means the per-row cost IS arithmetic and no kernel rewrite
    can return much; a low fraction means the row is paying for something else.
    """
    out = {}
    for key, fit_record in band_fits.items():
        if fit_record["kernel"] != "affine4":
            continue
        rate = fit_record["tflop_per_s"]
        out[key] = {
            "tflop_per_s": rate,
            "fraction_of_affine4_peak": rate / ceilings["affine4"]["tflop_per_s"],
            "fraction_of_dense_peak": rate / ceilings["dense"]["tflop_per_s"],
            "headroom_pct_vs_affine4_peak": 100.0
            * (1.0 - rate / ceilings["affine4"]["tflop_per_s"]),
        }
    return out


def analyse_shape(payload: dict) -> dict:
    """Rung 2. Split the in-band per-row slope into K-proportional work and
    K-independent per-row overhead.

    For each (N, K) the G == 2 band slope is fitted over M = 5..8. Regressing
    those slopes on K gives `s(K) = a + b * K`. `b * K` is the reduction-scaled
    term, which is the multiply-accumulate chain plus the activation reads that
    ride with it (hypotheses B and D). `a` is what a row costs before any
    reduction runs: launch, group setup and register allocation (hypothesis C).
    """
    cells = payload["cells"]
    overhead = payload["eval_overhead_us"]
    shapes = sorted({c["outputs"] for c in cells})

    band: dict[str, dict] = {}
    for outputs in shapes:
        for k in sorted({c["k"] for c in cells if c["outputs"] == outputs}):
            selected = [
                c for c in cells if c["outputs"] == outputs and c["k"] == k
            ]
            record = fit([(float(c["m"]), c["us"]) for c in selected])
            record.update(
                outputs=outputs,
                k=k,
                flops_per_row=2.0 * k * outputs,
                tflop_per_s=(2.0 * k * outputs)
                / (record["slope_us_per_row"] * 1e-6)
                / 1e12,
                intercept_net_us=record["intercept_us"] - overhead,
            )
            band[f"{outputs}/{k}"] = record

    # The matched NA contrast. M 5 -> 6 adds a row at IPG 3 (NA 2 -> 3); M 7 -> 8
    # adds a row at IPG 4 (NA 3 -> 4). Both steps add exactly one row, one idle
    # x-group and the same multiply-accumulate count, so any gap between them is
    # register pressure at the wider accumulator, not work.
    na_contrast: dict[str, dict] = {}
    for key in band:
        outputs = band[key]["outputs"]
        k = band[key]["k"]
        means = {}
        for width in (5, 6, 7, 8):
            values = [
                c["us"]
                for c in cells
                if c["outputs"] == outputs and c["k"] == k and c["m"] == width
            ]
            means[width] = sum(values) / len(values)
        step_ipg3 = means[6] - means[5]
        step_ipg4 = means[8] - means[7]
        na_contrast[key] = {
            "outputs": outputs,
            "k": k,
            "step_na2_to_3_us": step_ipg3,
            "step_na3_to_4_us": step_ipg4,
            "excess_pct": 100.0 * (step_ipg4 - step_ipg3) / step_ipg3,
        }

    slope_in_k: dict[str, dict] = {}
    for outputs in shapes:
        points = [
            (float(record["k"]), record["slope_us_per_row"])
            for record in band.values()
            if record["outputs"] == outputs
        ]
        if len(points) < 3:
            continue
        record = fit(sorted(points))
        reference = max(k for k, _ in points)
        predicted = record["intercept_us"] + record["slope_us_per_row"] * reference
        slope_in_k[str(outputs)] = {
            "outputs": outputs,
            "k_points": sorted(k for k, _ in points),
            "us_per_row_per_k": record["slope_us_per_row"],
            "se_us_per_row_per_k": record["se_slope_us_per_row"],
            "k_independent_us_per_row": record["intercept_us"],
            "se_k_independent_us_per_row": record["se_intercept_us"],
            "r_squared": record["r_squared"],
            "reference_k": reference,
            "k_independent_share_at_reference_k": record["intercept_us"]
            / predicted,
        }

    nulls = []
    for null in payload["nulls"]:
        nulls.append(null)
    drift = {}
    for outputs in shapes:
        for k in sorted({c["k"] for c in cells if c["outputs"] == outputs}):
            opened = next(
                (
                    n
                    for n in nulls
                    if n["outputs"] == outputs
                    and n["k"] == k
                    and n["label"] == "open"
                ),
                None,
            )
            closed = next(
                (
                    n
                    for n in nulls
                    if n["outputs"] == outputs
                    and n["k"] == k
                    and n["label"] == "close"
                ),
                None,
            )
            if opened and closed:
                drift[f"{outputs}/{k}"] = {
                    "open_us": opened["us"],
                    "close_us": closed["us"],
                    "drift_pct": 100.0
                    * (closed["us"] - opened["us"])
                    / opened["us"],
                }

    return {
        "eval_overhead_us": overhead,
        "eval_overhead_close_us": payload["eval_overhead_close_us"],
        "band_slope_per_k": band,
        "na_contrast": na_contrast,
        "slope_in_k": slope_in_k,
        "session_null": drift,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path", nargs="?", default="research/out/e97-row-cost-r1/row-cost.json"
    )
    parser.add_argument("--json", default="")
    parser.add_argument(
        "--peak",
        default="",
        help="rung-0 peak.json; reads every band slope as a fraction of it",
    )
    parser.add_argument(
        "--shape", default="", help="rung-2 shape.json; the K sweep"
    )
    args = parser.parse_args()

    payload = json.loads(pathlib.Path(args.path).read_text())
    result = analyse(payload)

    if args.peak:
        ceilings = peak_ceilings(json.loads(pathlib.Path(args.peak).read_text()))
        result["peak_ceilings"] = ceilings
        result["against_peak"] = against_peak(result["band_fits"], ceilings)

    if args.shape:
        result["shape"] = analyse_shape(
            json.loads(pathlib.Path(args.shape).read_text())
        )

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

    print("\nmarginal cost of each single added row (rung 3)")
    print(
        f"  {'arm':<20}{'step':>9}{'us':>9}{'G?':>5}{'IPG':>9}"
        f"{'TFLOP/s if FMA':>16}{'x M=1 read':>12}"
    )
    for key in sorted(
        result["increments"],
        key=lambda k: (
            result["increments"][k]["kernel"],
            result["increments"][k]["outputs"],
            result["increments"][k]["to_m"],
        ),
    ):
        i = result["increments"][key]
        arm = f"{i['kernel']}/{i['outputs']}"
        step = f"{i['from_m']}->{i['to_m']}"
        crosses = "" if i["crosses_group"] is None else (
            "NEW" if i["crosses_group"] else "-"
        )
        ipg = (
            f"{i['ipg_from']}->{i['ipg_to']}"
            if i["ipg_from"] and i["ipg_to"]
            else ""
        )
        print(
            f"  {arm:<20}{step:>9}{i['step_us']:>9.1f}{crosses:>5}{ipg:>9}"
            f"{i['tflop_per_s_if_arithmetic']:>16.2f}"
            f"{i.get('fraction_of_m1_weight_read', float('nan')):>12.3f}"
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

    if "against_peak" in result:
        print("\nmeasured ceiling of this GPU (rung 0)")
        for form, ceiling in sorted(result["peak_ceilings"].items()):
            print(
                f"  {form:<8} {ceiling['tflop_per_s']:.3f} TFLOP/s "
                f"({ceiling['label']} {ceiling['shape']} {ceiling['dtype']})"
            )
        print("\nband slope as a fraction of that ceiling")
        for key in sorted(result["against_peak"]):
            a = result["against_peak"][key]
            print(
                f"  {key:<24} {a['tflop_per_s']:>6.2f} TFLOP/s "
                f"affine4_peak={100 * a['fraction_of_affine4_peak']:>5.1f} % "
                f"dense_peak={100 * a['fraction_of_dense_peak']:>5.1f} % "
                f"headroom={a['headroom_pct_vs_affine4_peak']:>5.1f} %"
            )

    if "shape" in result:
        shape = result["shape"]
        print("\nrung 2: G == 2 band slope against the reduction length")
        print(
            f"  {'N/K':<16}{'slope us':>10}{'se':>8}{'TFLOP/s':>9}"
            f"{'R2':>7}{'intercept':>11}"
        )
        for key in sorted(
            shape["band_slope_per_k"],
            key=lambda k: (
                shape["band_slope_per_k"][k]["outputs"],
                shape["band_slope_per_k"][k]["k"],
            ),
        ):
            b = shape["band_slope_per_k"][key]
            print(
                f"  {key:<16}{b['slope_us_per_row']:>10.2f}"
                f"{b['se_slope_us_per_row']:>8.2f}{b['tflop_per_s']:>9.2f}"
                f"{b['r_squared']:>7.3f}{b['intercept_us']:>11.1f}"
            )
        print("\n  K-proportional versus K-independent per-row cost")
        for key in sorted(shape["slope_in_k"]):
            s = shape["slope_in_k"][key]
            print(
                f"    O={key:<8} b={1e3 * s['us_per_row_per_k']:.4f} "
                f"+/- {1e3 * s['se_us_per_row_per_k']:.4f} ns/row/K   "
                f"a={s['k_independent_us_per_row']:+.2f} "
                f"+/- {s['se_k_independent_us_per_row']:.2f} us/row   "
                f"R2={s['r_squared']:.4f}   "
                f"a is {100 * s['k_independent_share_at_reference_k']:.1f} % "
                f"of the slope at K={s['reference_k']}"
            )
        print("\n  matched NA step: 2->3 at IPG 3 versus 3->4 at IPG 4")
        for key in sorted(
            shape["na_contrast"],
            key=lambda k: (
                shape["na_contrast"][k]["outputs"],
                shape["na_contrast"][k]["k"],
            ),
        ):
            c = shape["na_contrast"][key]
            print(
                f"    {key:<16} na2_3={c['step_na2_to_3_us']:>9.2f} us "
                f"na3_4={c['step_na3_to_4_us']:>9.2f} us "
                f"excess={c['excess_pct']:+7.1f} %"
            )
        print("\n  rung-2 session null")
        for key in sorted(shape["session_null"]):
            n = shape["session_null"][key]
            print(f"    {key:<16} drift={n['drift_pct']:+.2f} %")

    if args.json:
        out = pathlib.Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
