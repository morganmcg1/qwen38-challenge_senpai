#!/usr/bin/env python3
"""Turns raw verify-width cost curves into the width-9 verdict.

Reads the vendored (scored-kernel) sweep, optionally the stock pip-MLX control,
and reports for every scored `quantized_matmul` shape:

  cost(M)/cost(1)                       the raw width penalty
  qmv_tax(M) = cost(M) / roofline(M)    how much of that penalty is kernel
                                        inefficiency rather than real work

then weights the per-shape curves by the scored call mix to predict the
verify-cost multiplier at width 9 relative to width 1, and evaluates the
assignment's stop rule against it.

Research-only: never packaged into a submission.
"""

import argparse
import json
import os
import sys

STOP_RETIRE_BELOW = 1.5
STOP_FULL_ABOVE = 3.0
ADVISOR_EVAL_WALL_MS = {7: 79.0, 8: 89.0, 9: 106.0}


def load(path):
    with open(path) as f:
        return json.load(f)


def row(shape, m):
    for r in shape["rows"]:
        if r["m"] == m:
            return r
    return None


def roofline_seconds(shape, m, rf):
    return (
        shape["weight_bytes"] / rf["peak_bandwidth_bytes_per_second"]
        + m * shape["flops_per_row"] / rf["peak_flops_per_second"]
    )


def per_shape_curve(shape, rf, widths):
    base = row(shape, 1)["seconds_per_call"]
    out = []
    for m in widths:
        r = row(shape, m)
        if r is None:
            continue
        floor = max(base, roofline_seconds(shape, m, rf))
        out.append(
            {
                "name": shape["name"],
                "k": shape["k"],
                "n": shape["n"],
                "calls_per_verify": shape["calls_per_verify"],
                "m": m,
                "seconds_per_call": r["seconds_per_call"],
                "cost_ratio_vs_m1": r["seconds_per_call"] / base,
                "roofline_seconds": roofline_seconds(shape, m, rf),
                "qmv_tax": r["seconds_per_call"] / floor,
                "row0_bitwise_matches_m1": r["row0_bitwise_matches_m1"],
                "row0_max_abs_delta_vs_m1": r["row0_max_abs_delta_vs_m1"],
            }
        )
    return out


def weighted_verify_seconds(shapes, m):
    total = 0.0
    for s in shapes:
        r = row(s, m)
        if r is None:
            return None
        total += s["calls_per_verify"] * r["seconds_per_call"]
    return total


def crossover(shape):
    """Largest M whose cost is still below the step to the next kernel."""
    rows = sorted(shape["rows"], key=lambda r: r["m"])
    worst_step, worst_m = 0.0, None
    for prev, cur in zip(rows, rows[1:]):
        step = cur["seconds_per_call"] / prev["seconds_per_call"]
        if step > worst_step:
            worst_step, worst_m = step, cur["m"]
    return {"largest_step_at_m": worst_m, "largest_step_ratio": worst_step}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vendored", required=True)
    ap.add_argument("--stock")
    ap.add_argument("--out", required=True)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--tag", default="local")
    ap.add_argument("--host", default="unknown")
    ap.add_argument("--base-sha", default="unknown")
    args = ap.parse_args()

    vend = load(args.vendored)
    rf = vend["roofline"]
    widths = vend["widths"]
    shapes = vend["shapes"]

    curves = []
    for s in shapes:
        curves.extend(per_shape_curve(s, rf, widths))

    verify = {m: weighted_verify_seconds(shapes, m) for m in widths}
    base_verify = verify[1]
    multiplier = {m: verify[m] / base_verify for m in widths if verify[m]}

    weighted_9 = multiplier[9]
    if weighted_9 < STOP_RETIRE_BELOW:
        branch = "retire"
        decision = (
            f"weighted cost(9)/cost(1) = {weighted_9:.3f}x < {STOP_RETIRE_BELOW}x: "
            "small-M qmv inefficiency is NOT the width-9 cost driver. "
            "Hypothesis retired, Part B not run."
        )
    elif weighted_9 > STOP_FULL_ABOVE:
        branch = "part_b_full"
        decision = (
            f"weighted cost(9)/cost(1) = {weighted_9:.3f}x > {STOP_FULL_ABOVE}x: "
            "run Part B (padding and kernel retune)."
        )
    else:
        branch = "part_b_a_only"
        decision = (
            f"weighted cost(9)/cost(1) = {weighted_9:.3f}x is between "
            f"{STOP_RETIRE_BELOW}x and {STOP_FULL_ABOVE}x: Part B(a) padding only."
        )

    # Does padding 9 -> 10 (into the qmm_splitk regime) buy anything?
    pad_gain = None
    if verify.get(10):
        pad_gain = {
            "verify_seconds_m9": verify[9],
            "verify_seconds_m10": verify[10],
            "pad_9_to_10_speedup": verify[9] / verify[10],
        }

    # Modelled verify wall against the advisor's observed eval_wall.
    advisor = {}
    for m, ms in ADVISOR_EVAL_WALL_MS.items():
        if verify.get(m):
            advisor[m] = {
                "advisor_eval_wall_ms": ms,
                "modelled_qmm_wall_ms": verify[m] * 1e3,
                "qmm_share_of_eval_wall": verify[m] * 1e3 / ms,
                "advisor_ratio_vs_m7": ms / ADVISOR_EVAL_WALL_MS[7],
                "modelled_ratio_vs_m7": verify[m] / verify[7],
            }

    stock_vs_vendored = None
    if args.stock:
        stock = load(args.stock)
        stock_vs_vendored = []
        for sv in shapes:
            ss = next((z for z in stock["shapes"] if z["name"] == sv["name"]), None)
            if ss is None:
                continue
            for m in widths:
                rv, rs = row(sv, m), row(ss, m)
                if rv is None or rs is None:
                    continue
                stock_vs_vendored.append(
                    {
                        "name": sv["name"],
                        "m": m,
                        "vendored_seconds": rv["seconds_per_call"],
                        "stock_seconds": rs["seconds_per_call"],
                        "vendored_speedup_vs_stock": rs["seconds_per_call"]
                        / rv["seconds_per_call"],
                    }
                )

    out = {
        "host": args.host,
        "base_sha": args.base_sha,
        "roofline": rf,
        "widths": widths,
        "per_shape_curve": curves,
        "weighted_verify_seconds": verify,
        "weighted_cost_multiplier_vs_m1": multiplier,
        "weighted_cost_9_over_1": weighted_9,
        "stop_rule_branch": branch,
        "decision": decision,
        "pad_9_to_10": pad_gain,
        "advisor_eval_wall_comparison": advisor,
        "crossover_probes": [
            {
                "name": p["name"],
                "predicted_vector_limit": p["predicted_vector_limit"],
                **crossover(p),
            }
            for p in vend["dispatch_boundary_probes"]
        ],
        "fast_path_probes": [
            {
                "name": p["name"],
                "k": p["k"],
                "seconds_per_call_by_m": {
                    r["m"]: r["seconds_per_call"] for r in p["rows"]
                },
            }
            for p in vend["fast_path_probes"]
        ],
        "stock_vs_vendored": stock_vs_vendored,
    }

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print(f"host {args.host}   base {args.base_sha}")
    print(
        f"roofline: {rf['peak_bandwidth_bytes_per_second']/1e9:.1f} GB/s, "
        f"{rf['peak_flops_per_second']/1e12:.2f} TFLOP/s\n"
    )
    print(f"{'shape':36s} {'K':>6s} {'N':>7s} {'calls':>5s}  " + "".join(
        f"{'M'+str(m):>7s}" for m in widths))
    for s in shapes:
        base = row(s, 1)["seconds_per_call"]
        ratios = "".join(
            f"{row(s, m)['seconds_per_call']/base:7.2f}" for m in widths
        )
        print(f"{s['name']:36s} {s['k']:6d} {s['n']:7d} "
              f"{s['calls_per_verify']:5d}  {ratios}")
    print("\nqmv_tax(M) = cost(M) / max(cost(1), roofline(M))")
    for s in shapes:
        taxes = "".join(
            f"{c['qmv_tax']:7.2f}"
            for c in per_shape_curve(s, rf, widths)
        )
        print(f"{s['name']:36s} {'':6s} {'':7s} {'':5s}  {taxes}")
    print("\ncall-mix-weighted verify cost, relative to width 1")
    for m in widths:
        print(f"  M={m:2d}  {verify[m]*1e3:8.3f} ms  {multiplier[m]:6.3f}x")
    print(f"\n{decision}")
    if pad_gain:
        print(f"pad 9->10 speedup: {pad_gain['pad_9_to_10_speedup']:.3f}x")
    print("\ndispatch-boundary probes (largest cost step = kernel change)")
    for p in out["crossover_probes"]:
        print(f"  {p['name']:24s} predicted limit {p['predicted_vector_limit']:2d}  "
              f"largest step at M={p['largest_step_at_m']} "
              f"({p['largest_step_ratio']:.2f}x)")
    print("\nfast-path probes (K % 512 == 0 selects qmv_fast)")
    for p in out["fast_path_probes"]:
        vals = "  ".join(f"M{m}={s*1e3:.3f}ms" for m, s in p["seconds_per_call_by_m"].items())
        print(f"  {p['name']:24s} {vals}")
    if stock_vs_vendored:
        print("\nvendored vs stock pip MLX (speedup > 1 means this checkout is faster)")
        for s in shapes:
            vals = "".join(
                f"{e['vendored_speedup_vs_stock']:7.2f}"
                for e in stock_vs_vendored
                if e["name"] == s["name"]
            )
            print(f"  {s['name']:36s}{vals}")

    if args.wandb:
        import wandb

        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "qwen38-mlx-challenge-senpai"),
            entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
            name=f"qmv-cost-curve-{args.tag}",
            job_type="analysis",
            group="qwen38-r1-e5-qmv-small-m-retune",
            config={
                "host": args.host,
                "base_sha": args.base_sha,
                "reps": vend["reps"],
                "inner_calls_per_rep": vend["inner_calls_per_rep"],
                "peak_bandwidth_gb_s": rf["peak_bandwidth_bytes_per_second"] / 1e9,
                "peak_tflops": rf["peak_flops_per_second"] / 1e12,
            },
        )
        curve_table = wandb.Table(
            columns=[
                "shape", "k", "n", "calls_per_verify", "m", "seconds_per_call",
                "cost_ratio_vs_m1", "roofline_seconds", "qmv_tax",
                "row0_bitwise_matches_m1",
            ]
        )
        for c in curves:
            curve_table.add_data(
                c["name"], c["k"], c["n"], c["calls_per_verify"], c["m"],
                c["seconds_per_call"], c["cost_ratio_vs_m1"], c["roofline_seconds"],
                c["qmv_tax"], c["row0_bitwise_matches_m1"],
            )
        verify_table = wandb.Table(
            columns=["m", "verify_seconds", "verify_ms", "multiplier_vs_m1"]
        )
        for m in widths:
            verify_table.add_data(m, verify[m], verify[m] * 1e3, multiplier[m])

        run.log(
            {
                "qmv/cost_curve": curve_table,
                "qmv/weighted_verify": verify_table,
                "qmv/cost_ratio_by_shape": wandb.plot.line(
                    curve_table, "m", "cost_ratio_vs_m1",
                    stroke="shape", title="quantized_matmul cost(M)/cost(1)"),
                "qmv/tax_by_shape": wandb.plot.line(
                    curve_table, "m", "qmv_tax",
                    stroke="shape", title="qmv_tax(M) vs measured roofline"),
                "qmv/weighted_verify_multiplier": wandb.plot.line(
                    verify_table, "m", "multiplier_vs_m1",
                    title="call-mix-weighted verify cost vs width 1"),
            }
        )
        flat = {
            "qmv/weighted_cost_9_over_1": weighted_9,
            "qmv/stop_rule_branch": branch,
            "qmv/peak_bandwidth_gb_s": rf["peak_bandwidth_bytes_per_second"] / 1e9,
            "qmv/peak_tflops": rf["peak_flops_per_second"] / 1e12,
        }
        flat |= {f"qmv/weighted_multiplier_m{m}": multiplier[m] for m in widths}
        flat |= {f"qmv/verify_ms_m{m}": verify[m] * 1e3 for m in widths}
        if pad_gain:
            flat["qmv/pad_9_to_10_speedup"] = pad_gain["pad_9_to_10_speedup"]
        for m, d in advisor.items():
            flat[f"qmv/qmm_share_of_eval_wall_m{m}"] = d["qmm_share_of_eval_wall"]
        run.summary.update(flat)
        run.log({k: v for k, v in flat.items() if not isinstance(v, str)})
        print(f"WANDB_RUN_URL {run.url}", file=sys.stderr)
        print(f"WANDB_RUN_ID {run.id}", file=sys.stderr)
        run.finish()


if __name__ == "__main__":
    main()
