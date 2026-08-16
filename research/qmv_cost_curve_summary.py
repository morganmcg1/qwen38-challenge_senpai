#!/usr/bin/env python3
"""Turns raw verify-width cost curves into the width-9 verdict.

Reads the vendored (scored-kernel) sweep, optionally the stock pip-MLX control,
and reports for every scored `quantized_matmul` shape:

  cost(M)/cost(1)                       the raw width penalty
  qmv_tax(M) = cost(M) / roofline(M)    how much of that penalty is kernel
                                        inefficiency rather than real work

The floor is calibrated per shape from its own endpoints -- BW_eff from M=1,
FLOPS_eff from M=512 -- and the two terms are combined with max(), not sum(),
because a GPU overlaps streaming and arithmetic. The raw ratio is host-specific
(it tracks where each machine's roofline knee falls) while the normalised tax
transfers across hosts, so the tax is the headline verdict.

Both are then weighted by the scored call mix to predict the verify-cost
multiplier at width 9 relative to width 1, and evaluated against the stop rule.

Research-only: never packaged into a submission.
"""

import argparse
import json
import os
import sys

STOP_RETIRE_BELOW = 1.5
STOP_FULL_ABOVE = 3.0
NORM_RETIRE_BELOW = 1.35
NORM_FULL_ABOVE = 2.7
FLOPS_REF_WIDTH = 512
ADVISOR_EVAL_WALL_MS = {7: 79.0, 8: 89.0, 9: 106.0}


def load(path):
    with open(path) as f:
        return json.load(f)


def row(shape, m):
    for r in shape["rows"]:
        if r["m"] == m:
            return r
    return None


def shape_roofline(shape):
    """Per-shape effective peaks measured from this shape's own curve.

    BW_eff comes from M=1 (pure weight streaming) and FLOPS_eff from the widest
    sampled M (compute saturated), so the floor is calibrated on the same kernel,
    host and thermal state as the measurement it normalises.
    """
    bytes_ = shape["weight_bytes"]
    flops_row = shape["flops_per_row"]
    c1 = row(shape, 1)["seconds_per_call"]
    ref = row(shape, FLOPS_REF_WIDTH)
    bw_eff = bytes_ / c1
    flops_eff = (
        flops_row * FLOPS_REF_WIDTH / ref["seconds_per_call"] if ref else None
    )
    # bytes/BW_eff and flops_row*M/FLOPS_eff both scale with K*N, so the knee is
    # a property of the machine alone: M* = 0.5625 * FLOPS_eff / (2 * BW_eff).
    knee = bytes_ * flops_eff / (bw_eff * flops_row) if flops_eff else None
    return {"bw_eff": bw_eff, "flops_eff": flops_eff, "knee_m": knee}


def roofline_seconds(shape, m, eff):
    """max(bandwidth floor, compute floor) -- the two are overlapped, not summed."""
    bw = shape["weight_bytes"] / eff["bw_eff"]
    if not eff["flops_eff"]:
        return bw
    return max(bw, m * shape["flops_per_row"] / eff["flops_eff"])


def hw_roofline_seconds(shape, m, rf):
    return max(
        shape["weight_bytes"] / rf["peak_bandwidth_bytes_per_second"],
        m * shape["flops_per_row"] / rf["peak_flops_per_second"],
    )


def empirical_knee(shape, widths):
    """Where the measured curve leaves its flat, bandwidth-bound plateau."""
    base = row(shape, 1)["seconds_per_call"]
    small = [m for m in sorted(widths) if row(shape, m)]
    plateau_end = 1
    for m in small:
        if row(shape, m)["seconds_per_call"] > 1.10 * base:
            break
        plateau_end = m
    hi, lo = row(shape, FLOPS_REF_WIDTH), row(shape, 256)
    slope = None
    marginal = None
    if hi and lo:
        slope = (hi["seconds_per_call"] - lo["seconds_per_call"]) / (
            FLOPS_REF_WIDTH - 256
        )
        for prev, cur in zip(small, small[1:]):
            if cur - prev != 1:
                break
            step = (
                row(shape, cur)["seconds_per_call"]
                - row(shape, prev)["seconds_per_call"]
            )
            if step >= 0.5 * slope:
                marginal = cur
                break
    return {
        "plateau_end_m": plateau_end,
        "asymptotic_seconds_per_row": slope,
        "marginal_knee_m": marginal,
    }


def per_shape_curve(shape, rf, widths):
    base = row(shape, 1)["seconds_per_call"]
    eff = shape_roofline(shape)
    out = []
    for m in widths:
        r = row(shape, m)
        if r is None:
            continue
        floor = roofline_seconds(shape, m, eff)
        hw_floor = hw_roofline_seconds(shape, m, rf)
        out.append(
            {
                "name": shape["name"],
                "k": shape["k"],
                "n": shape["n"],
                "calls_per_verify": shape["calls_per_verify"],
                "m": m,
                "seconds_per_call": r["seconds_per_call"],
                "cost_ratio_vs_m1": r["seconds_per_call"] / base,
                # What the *next* row costs in units of the first row. A shape
                # still on its bandwidth plateau charges ~0 for another row; one
                # past its knee charges ~1/knee_m. This is the number a depth
                # scheduler needs, and averaging it into cost(M)/cost(1) hides it.
                "incremental_tax": (
                    (r["seconds_per_call"] - row(shape, m - 1)["seconds_per_call"]) / base
                    if row(shape, m - 1) is not None
                    else None
                ),
                "roofline_seconds": floor,
                "qmv_tax": r["seconds_per_call"] / floor,
                "hw_roofline_seconds": hw_floor,
                "hw_efficiency": hw_floor / r["seconds_per_call"],
                "concurrent_speedup": (
                    r["seconds_per_call"] / r["seconds_per_call_concurrent"]
                    if r.get("seconds_per_call_concurrent")
                    else None
                ),
                "tap_overhead_fraction": (
                    r["tap_overhead_seconds_per_call"] / r["seconds_per_call"]
                    if r.get("tap_overhead_seconds_per_call") is not None
                    else None
                ),
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


def weighted_verify_roofline(shapes, m):
    total = 0.0
    for s in shapes:
        if not s["calls_per_verify"] or row(s, m) is None:
            continue
        total += s["calls_per_verify"] * roofline_seconds(s, m, shape_roofline(s))
    return total


def crossover(shape):
    """Locate the kernel switch by the shape of the cost curve itself.

    `M >= vector_limit` leaves `qmv` for `qmm_t_splitk`. A limit that is set too
    high shows up as a *drop*: the wider kernel is cheaper than the narrower one
    it replaced. `largest_drop_at_m` is therefore the measurement that decides
    whether padding the verify width across the limit can pay.
    """
    rows = sorted(shape["rows"], key=lambda r: r["m"])
    steps = {
        cur["m"]: cur["seconds_per_call"] / prev["seconds_per_call"]
        for prev, cur in zip(rows, rows[1:])
    }
    if not steps:
        return {}
    up_m = max(steps, key=steps.get)
    down_m = min(steps, key=steps.get)
    limit = shape.get("predicted_vector_limit")
    return {
        "largest_step_at_m": up_m,
        "largest_step_ratio": steps[up_m],
        "largest_drop_at_m": down_m,
        "largest_drop_ratio": steps[down_m],
        "step_at_predicted_limit": steps.get(limit),
    }


def gdn_curve(gdn, rf):
    """The 48 recurrent layers, normalised the same way as the projections.

    A GDN step moves the whole fp32 state in and out whatever the width, so its
    arithmetic intensity is ~2*M FLOP/byte against a machine balance in the high
    twenties. If the projections steepen with M while this stays flat, the round
    has two cost families and one draft-depth policy cannot be tuned against a
    single curve.
    """
    if not gdn:
        return None
    base = gdn["rows"][0]["seconds_per_call"]
    bw = rf["peak_bandwidth_bytes_per_second"]
    flops = rf["peak_flops_per_second"]
    rows = []
    prev = None
    for r in gdn["rows"]:
        floor = max(r["traffic_bytes"] / bw, r["flops"] / flops)
        rows.append(
            {
                "m": r["m"],
                "seconds_per_call": r["seconds_per_call"],
                "cost_ratio_vs_m1": r["seconds_per_call"] / base,
                "incremental_tax": (
                    (r["seconds_per_call"] - prev) / base if prev is not None else None
                ),
                "hw_roofline_seconds": floor,
                "hw_efficiency": floor / r["seconds_per_call"],
                "arithmetic_intensity_flop_per_byte": r["flops"] / r["traffic_bytes"],
                "layer_seconds_per_verify": r["seconds_per_call"] * gdn["calls_per_verify"],
            }
        )
        prev = r["seconds_per_call"]
    return {
        "name": gdn["name"],
        "calls_per_verify": gdn["calls_per_verify"],
        "state_bytes_per_layer": gdn["state_bytes_per_layer"],
        "state_bytes_all_layers": gdn["state_bytes_per_layer"] * gdn["calls_per_verify"],
        "machine_balance_flop_per_byte": flops / bw,
        "cost_9_over_1": next(
            (r["cost_ratio_vs_m1"] for r in rows if r["m"] == 9), None),
        "rows": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vendored", required=True)
    ap.add_argument("--stock")
    ap.add_argument("--out", required=True)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--tag", default="local")
    ap.add_argument("--host", default="unknown")
    ap.add_argument("--base-sha", default="unknown")
    ap.add_argument("--head-provenance")
    args = ap.parse_args()

    head_prov = {}
    if args.head_provenance and os.path.exists(args.head_provenance):
        head_prov = load(args.head_provenance)

    vend = load(args.vendored)
    rf = vend["roofline"]
    widths = vend["widths"]
    shapes = vend["shapes"]

    curves = []
    for s in shapes:
        curves.extend(per_shape_curve(s, rf, widths))

    verify = {m: weighted_verify_seconds(shapes, m) for m in widths}
    verify_floor = {m: weighted_verify_roofline(shapes, m) for m in widths}
    base_verify = verify[1]
    multiplier = {m: verify[m] / base_verify for m in widths if verify[m]}
    weighted_tax = {
        m: verify[m] / verify_floor[m] for m in widths if verify_floor.get(m)
    }

    # fb1: the raw ratio is host-specific (it tracks each machine's roofline
    # knee); the normalised tax is what transfers to the ranked M5.
    weighted_9 = multiplier[9]
    tax_9 = weighted_tax[9]
    if tax_9 < NORM_RETIRE_BELOW:
        branch = "retire"
        decision = (
            f"weighted qmv_tax(9) = {tax_9:.3f}x < {NORM_RETIRE_BELOW}x: "
            "width-9 verify cost is real roofline work, not small-M kernel "
            "inefficiency. Hypothesis retired, Part B not run."
        )
    elif tax_9 > NORM_FULL_ABOVE:
        branch = "part_b_full"
        decision = (
            f"weighted qmv_tax(9) = {tax_9:.3f}x > {NORM_FULL_ABOVE}x: "
            "run Part B (padding and kernel retune)."
        )
    else:
        branch = "part_b_a_only"
        decision = (
            f"weighted qmv_tax(9) = {tax_9:.3f}x is between "
            f"{NORM_RETIRE_BELOW}x and {NORM_FULL_ABOVE}x: Part B(a) padding only."
        )
    raw_branch = (
        "retire" if weighted_9 < STOP_RETIRE_BELOW
        else "part_b_full" if weighted_9 > STOP_FULL_ABOVE
        else "part_b_a_only"
    )

    knees = []
    for s in shapes:
        eff = shape_roofline(s)
        knees.append(
            {
                "name": s["name"],
                "k": s["k"],
                "n": s["n"],
                "bw_eff_gb_s": eff["bw_eff"] / 1e9,
                "flops_eff_tflop_s": (
                    eff["flops_eff"] / 1e12 if eff["flops_eff"] else None
                ),
                "predicted_knee_m": eff["knee_m"],
                **empirical_knee(s, widths),
            }
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

    gdn = gdn_curve(vend.get("gdn_recurrence"), rf)

    out = {
        "host": args.host,
        "base_sha": args.base_sha,
        "head_provenance": head_prov,
        "device": vend.get("device", {}),
        "roofline": rf,
        "widths": widths,
        "per_shape_curve": curves,
        "per_shape_roofline": knees,
        "weighted_verify_seconds": verify,
        "weighted_verify_roofline_seconds": verify_floor,
        "weighted_cost_multiplier_vs_m1": multiplier,
        "weighted_qmv_tax": weighted_tax,
        "weighted_cost_9_over_1": weighted_9,
        "weighted_qmv_tax_9": tax_9,
        "stop_rule_branch": branch,
        "stop_rule_branch_raw_thresholds": raw_branch,
        "decision": decision,
        "pad_9_to_10": pad_gain,
        "advisor_eval_wall_comparison": advisor,
        "crossover_probes": [
            {
                "name": p["name"],
                "predicted_vector_limit": p["predicted_vector_limit"],
                **crossover(p),
            }
            for p in vend["dispatch_boundary_probes"] + shapes
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
        "head_fc_dtype_probe": vend.get("head_fc_dtype_probe"),
        "gdn_recurrence": gdn,
        "stock_vs_vendored": stock_vs_vendored,
    }

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print(f"host {args.host}   base {args.base_sha}")
    dev = out["device"]
    if dev:
        limits = {e["vector_limit"] for e in dev["predicted_vector_limits"]
                  if e["shape"] in {s["name"] for s in shapes}}
        print(f"gpu {dev['architecture']} (class '{dev['architecture_class']}', "
              f"gen {dev['architecture_gen']}), _nax available: {dev['nax_available']}")
        print(f"get_qmv_batch_limit -> vector_limit {sorted(limits)} "
              f"for the scored shapes")
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
    print("\nqmv_tax(M) = cost(M) / max(bytes/BW_eff, 2*K*N*M/FLOPS_eff)")
    for s in shapes:
        taxes = "".join(
            f"{c['qmv_tax']:7.2f}"
            for c in per_shape_curve(s, rf, widths)
        )
        print(f"{s['name']:36s} {'':6s} {'':7s} {'':5s}  {taxes}")
    print("\nmeasurement-validity guards: concurrent/chained speedup (GPU idle at")
    print("small M if >> 1) and tap scaffolding as a fraction of the reported cost")
    for s in shapes:
        curve = per_shape_curve(s, rf, widths)
        conc = "".join(
            f"{c['concurrent_speedup']:7.2f}" if c["concurrent_speedup"] else f"{'n/a':>7s}"
            for c in curve
        )
        print(f"  conc {s['name']:31s}{conc}")
    for s in shapes:
        curve = per_shape_curve(s, rf, widths)
        tapf = "".join(
            f"{c['tap_overhead_fraction']*100:6.1f}%" if c["tap_overhead_fraction"] is not None
            else f"{'n/a':>7s}"
            for c in curve
        )
        print(f"  tap  {s['name']:31s}{tapf}")
    print("\nper-shape measured peaks and roofline knee")
    print(f"  {'shape':36s} {'BW_eff':>10s} {'FLOPS_eff':>11s} {'M*':>6s} "
          f"{'plateau':>8s} {'marginal':>9s}")
    for kn in knees:
        fl = f"{kn['flops_eff_tflop_s']:11.3f}" if kn["flops_eff_tflop_s"] else f"{'n/a':>11s}"
        km = f"{kn['predicted_knee_m']:6.2f}" if kn["predicted_knee_m"] else f"{'n/a':>6s}"
        mk = kn["marginal_knee_m"]
        print(f"  {kn['name']:36s} {kn['bw_eff_gb_s']:9.1f}G {fl} {km} "
              f"{kn['plateau_end_m']:8d} {(mk if mk else 0):9d}")
    print("\ncall-mix-weighted verify cost, relative to width 1")
    for m in widths:
        tax = weighted_tax.get(m)
        tx = f"{tax:6.3f}x" if tax else f"{'n/a':>7s}"
        print(f"  M={m:3d}  {verify[m]*1e3:8.3f} ms  raw {multiplier[m]:6.3f}x"
              f"  tax {tx}")
    print(f"\n{decision}")
    print(f"raw-threshold cross-check: cost(9)/cost(1) = {weighted_9:.3f}x "
          f"-> branch '{raw_branch}' under the original 1.5x/3.0x rule")
    if pad_gain:
        print(f"pad 9->10 speedup: {pad_gain['pad_9_to_10_speedup']:.3f}x")
    print("\ndispatch boundary: a cost DROP at M means the wider kernel that takes")
    print("over there is cheaper, so padding up to it can pay")
    print(f"  {'shape':36s} {'limit':>5s} {'step@limit':>11s} "
          f"{'max step':>16s} {'max drop':>16s}")
    for p in out["crossover_probes"]:
        at = p["step_at_predicted_limit"]
        at = f"{at:11.2f}" if at else f"{'n/a':>11s}"
        print(f"  {p['name']:36s} {p['predicted_vector_limit']:5d} {at} "
              f"{p['largest_step_ratio']:9.2f}x @M{p['largest_step_at_m']:<4d} "
              f"{p['largest_drop_ratio']:9.2f}x @M{p['largest_drop_at_m']:<4d}")
    print("\nfast-path probes (K % 512 == 0 selects qmv_fast)")
    for p in out["fast_path_probes"]:
        vals = "  ".join(f"M{m}={s*1e3:.3f}ms" for m, s in p["seconds_per_call_by_m"].items())
        print(f"  {p['name']:24s} {vals}")

    hp = out["head_fc_dtype_probe"]
    if hp:
        print(f"\nproposal-head fc, K={hp['k']} N={hp['n']}, M=1: does 4-bit "
              "requantisation buy its byte ratio back as time?")
        print(f"  bf16    {hp['bf16_weight_bytes']/1e6:8.1f} MB  "
              f"{hp['bf16_seconds_per_call']*1e3:7.3f} ms  "
              f"{hp['bf16_effective_bandwidth_bytes_per_second']/1e9:6.1f} GB/s")
        print(f"  q4/g64  {hp['q4g64_weight_bytes']/1e6:8.1f} MB  "
              f"{hp['q4g64_seconds_per_call']*1e3:7.3f} ms  "
              f"{hp['q4g64_effective_bandwidth_bytes_per_second']/1e9:6.1f} GB/s")
        print(f"  bytes {hp['byte_ratio_bf16_over_q4g64']:.3f}x  ->  "
              f"time {hp['time_ratio_bf16_over_q4g64']:.3f}x  "
              f"({100*hp['time_ratio_bf16_over_q4g64']/hp['byte_ratio_bf16_over_q4g64']:.0f}% "
              "of the byte ratio realised)")

    if out["gdn_recurrence"]:
        g = out["gdn_recurrence"]
        print(f"\ngated-delta recurrence, {g['calls_per_verify']} layers x "
              f"{g['state_bytes_per_layer']/2**20:.1f} MiB state "
              f"({g['state_bytes_all_layers']/2**20:.0f} MiB total); "
              f"machine balance {g['machine_balance_flop_per_byte']:.1f} FLOP/byte")
        print(f"  {'M':>3s} {'ms/layer':>9s} {'ms/verify':>10s} {'cost/M1':>8s} "
              f"{'incr':>7s} {'FLOP/B':>7s} {'hw eff':>7s}")
        for r in g["rows"]:
            incr = f"{r['incremental_tax']:7.3f}" if r["incremental_tax"] is not None else f"{'':>7s}"
            print(f"  {r['m']:3d} {r['seconds_per_call']*1e3:9.4f} "
                  f"{r['layer_seconds_per_verify']*1e3:10.3f} "
                  f"{r['cost_ratio_vs_m1']:8.3f} {incr} "
                  f"{r['arithmetic_intensity_flop_per_byte']:7.2f} "
                  f"{r['hw_efficiency']:7.3f}")
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
                "head_dir": head_prov.get("head_dir"),
                "head_repo_declares_proposal_head": head_prov.get(
                    "repo_declares_proposal_head"
                ),
                "head_provenance_files": head_prov.get("files", []),
                "head_provenance_sha256": next(
                    (
                        f["sha256"]
                        for f in head_prov.get("files", [])
                        if f["name"] == "model.safetensors"
                    ),
                    None,
                ),
                "head_provenance_bytes": next(
                    (
                        f["bytes"]
                        for f in head_prov.get("files", [])
                        if f["name"] == "model.safetensors"
                    ),
                    None,
                ),
                "head_dtype": "bf16",
                "reps": vend["reps"],
                "inner_calls_per_rep": vend["inner_calls_per_rep"],
                "peak_bandwidth_gb_s": rf["peak_bandwidth_bytes_per_second"] / 1e9,
                "peak_tflops": rf["peak_flops_per_second"] / 1e12,
                "gpu_architecture": dev.get("architecture"),
                "gpu_architecture_class": dev.get("architecture_class"),
                "gpu_architecture_gen": dev.get("architecture_gen"),
                "nax_available": dev.get("nax_available"),
                "scored_shape_vector_limits": sorted(
                    {
                        e["vector_limit"]
                        for e in dev.get("predicted_vector_limits", [])
                        if e["shape"] in {s["name"] for s in shapes}
                    }
                ),
            },
        )
        curve_table = wandb.Table(
            columns=[
                "shape", "k", "n", "calls_per_verify", "m", "seconds_per_call",
                "cost_ratio_vs_m1", "incremental_tax", "roofline_seconds", "qmv_tax",
                "hw_roofline_seconds", "hw_efficiency", "concurrent_speedup",
                "tap_overhead_fraction", "row0_bitwise_matches_m1",
            ]
        )
        for c in curves:
            curve_table.add_data(
                c["name"], c["k"], c["n"], c["calls_per_verify"], c["m"],
                c["seconds_per_call"], c["cost_ratio_vs_m1"], c["incremental_tax"],
                c["roofline_seconds"],
                c["qmv_tax"], c["hw_roofline_seconds"], c["hw_efficiency"],
                c["concurrent_speedup"], c["tap_overhead_fraction"],
                c["row0_bitwise_matches_m1"],
            )
        verify_table = wandb.Table(
            columns=[
                "m", "verify_seconds", "verify_ms", "multiplier_vs_m1", "qmv_tax",
            ]
        )
        for m in widths:
            verify_table.add_data(
                m, verify[m], verify[m] * 1e3, multiplier[m], weighted_tax.get(m)
            )
        knee_table = wandb.Table(
            columns=[
                "shape", "k", "n", "bw_eff_gb_s", "flops_eff_tflop_s",
                "predicted_knee_m", "plateau_end_m", "marginal_knee_m",
            ]
        )
        for kn in knees:
            knee_table.add_data(
                kn["name"], kn["k"], kn["n"], kn["bw_eff_gb_s"],
                kn["flops_eff_tflop_s"], kn["predicted_knee_m"],
                kn["plateau_end_m"], kn["marginal_knee_m"],
            )

        gdn_table = wandb.Table(
            columns=[
                "m", "seconds_per_call", "ms_per_verify", "cost_ratio_vs_m1",
                "incremental_tax", "arithmetic_intensity_flop_per_byte",
                "hw_efficiency",
            ]
        )
        for r in (out["gdn_recurrence"] or {}).get("rows", []):
            gdn_table.add_data(
                r["m"], r["seconds_per_call"], r["layer_seconds_per_verify"] * 1e3,
                r["cost_ratio_vs_m1"], r["incremental_tax"],
                r["arithmetic_intensity_flop_per_byte"], r["hw_efficiency"],
            )

        run.log(
            {
                "qmv/cost_curve": curve_table,
                "qmv/weighted_verify": verify_table,
                "qmv/per_shape_roofline": knee_table,
                "qmv/gdn_recurrence": gdn_table,
                "qmv/gdn_cost_ratio": wandb.plot.line(
                    gdn_table, "m", "cost_ratio_vs_m1",
                    title="gated-delta recurrence cost(M)/cost(1)"),
                "qmv/incremental_tax_by_shape": wandb.plot.line(
                    curve_table, "m", "incremental_tax",
                    stroke="shape", title="marginal cost of row M, in units of cost(1)"),
                "qmv/cost_ratio_by_shape": wandb.plot.line(
                    curve_table, "m", "cost_ratio_vs_m1",
                    stroke="shape", title="quantized_matmul cost(M)/cost(1)"),
                "qmv/tax_by_shape": wandb.plot.line(
                    curve_table, "m", "qmv_tax",
                    stroke="shape", title="qmv_tax(M) vs measured roofline"),
                "qmv/hw_efficiency_by_shape": wandb.plot.line(
                    curve_table, "m", "hw_efficiency",
                    stroke="shape", title="fraction of hardware roofline attained"),
                "qmv/weighted_verify_multiplier": wandb.plot.line(
                    verify_table, "m", "multiplier_vs_m1",
                    title="call-mix-weighted verify cost vs width 1"),
                "qmv/weighted_verify_tax": wandb.plot.line(
                    verify_table, "m", "qmv_tax",
                    title="call-mix-weighted qmv_tax vs width"),
            }
        )
        flat = {
            "qmv/weighted_cost_9_over_1": weighted_9,
            "qmv/weighted_qmv_tax_9": tax_9,
            "qmv/stop_rule_branch": branch,
            "qmv/stop_rule_branch_raw_thresholds": raw_branch,
            "qmv/peak_bandwidth_gb_s": rf["peak_bandwidth_bytes_per_second"] / 1e9,
            "qmv/peak_tflops": rf["peak_flops_per_second"] / 1e12,
        }
        flat |= {f"qmv/weighted_multiplier_m{m}": multiplier[m] for m in widths}
        flat |= {f"qmv/verify_ms_m{m}": verify[m] * 1e3 for m in widths}
        flat |= {
            f"qmv/weighted_tax_m{m}": t for m, t in weighted_tax.items()
        }
        flat |= {
            f"qmv/knee_m_{kn['name']}": kn["predicted_knee_m"]
            for kn in knees if kn["predicted_knee_m"]
        }
        if hp:
            flat |= {
                "head_fc/bf16_ms": hp["bf16_seconds_per_call"] * 1e3,
                "head_fc/q4g64_ms": hp["q4g64_seconds_per_call"] * 1e3,
                "head_fc/byte_ratio": hp["byte_ratio_bf16_over_q4g64"],
                "head_fc/time_ratio": hp["time_ratio_bf16_over_q4g64"],
                "head_fc/byte_ratio_realised": (
                    hp["time_ratio_bf16_over_q4g64"] / hp["byte_ratio_bf16_over_q4g64"]),
            }
        if out["gdn_recurrence"]:
            g = out["gdn_recurrence"]
            flat |= {
                f"gdn/ms_per_verify_m{r['m']}": r["layer_seconds_per_verify"] * 1e3
                for r in g["rows"]
            }
            flat["gdn/cost_9_over_1"] = g["cost_9_over_1"]
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
