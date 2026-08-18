#!/usr/bin/env python3
"""E22 Q1: the C(M) verify-width cost curve at bits == 4.

Reads one or two `vendored.json` payloads written by
`Tests/MLXFastTests/QwenQMVCostCurveTests.swift` and reports, for every scored
projection and for the round-weighted aggregate:

    C(M)          seconds per call at verify width M
    C(M)/C(1)     what the round actually pays for one extra draft
    C(M)/M        seconds per verified row; flat means M is free per row

alongside the callee the live switch selects, whether it is a crossrow kernel,
its `inputs_per_group`, and the weight streams that implies.

The round-weighted column is the one a draft-depth policy would consume:
each shape contributes `calls_per_verify` launches, so `C_round(M)` is the
whole target-verification cost of a round that checks M rows.
"""

import argparse
import json
import math
import os

# Rounds observed at each accepted depth in a 512-token measurement under
# E17's S18 policy. A round that drafts to depth d verifies M = d + 1 rows,
# so this is the width distribution the verify cost is actually paid at.
S18_DEPTH_HISTOGRAM = {1: 19, 2: 138, 3: 67, 4: 21}

HEADLINE = ("head.lm_head", "mlp.gate_up_fused")


def load(path):
    with open(path) as f:
        return json.load(f)


def rows_by_m(shape):
    return {r["m"]: r for r in shape["rows"]}


def shapes_by_name(payload):
    return {s["name"]: s for s in payload["shapes"]}


def curve(shape, widths):
    """C(M), C(M)/C(1) and C(M)/M with the live dispatch labels."""
    rows = rows_by_m(shape)
    base = rows[1]["seconds_per_call"]
    out = []
    for m in widths:
        r = rows.get(m)
        if r is None:
            continue
        c = r["seconds_per_call"]
        out.append(
            {
                "shape": shape["name"],
                "k": shape["k"],
                "n": shape["n"],
                "m": m,
                "c_seconds": c,
                "c_over_c1": c / base,
                "c_over_m": c / m,
                "c_over_m_norm": (c / m) / base,
                "kernel": r["in_kernel_path"],
                "crossrow": r["crossrow"],
                "inputs_per_group": r["inputs_per_group"],
                "weight_streams": r["weight_streams"],
                "spread": (
                    (r["seconds_per_call_max"] - r["seconds_per_call_min"]) / c
                ),
                "row0_bitwise_matches_m1": r["row0_bitwise_matches_m1"],
            }
        )
    return out


def round_curve(payload, widths):
    """Whole-round verify cost, weighting each shape by its launches."""
    shapes = payload["shapes"]
    out = {}
    for m in widths:
        total = 0.0
        complete = True
        for s in shapes:
            n_calls = s["calls_per_verify"]
            if not n_calls:
                continue
            r = rows_by_m(s).get(m)
            if r is None:
                complete = False
                break
            total += n_calls * r["seconds_per_call"]
        if complete:
            out[m] = total
    return out


def dispatch_labels(payload, widths):
    """The live switch's choice per width, asserted identical across shapes.

    Every scored projection has n >= 5120, so all of them land in the same
    `out_vec_size` tier. A disagreement here means a shape fell into the
    narrow tier and the aggregate would be mixing two dispatch families.
    """
    labels, conflicts = {}, []
    for s in payload["shapes"]:
        for m, r in rows_by_m(s).items():
            if m not in widths:
                continue
            key = (r["in_kernel_path"], r["crossrow"], r["inputs_per_group"],
                   r["weight_streams"])
            if m in labels and labels[m] != key:
                conflicts.append({"m": m, "shape": s["name"],
                                  "a": labels[m], "b": key})
            labels.setdefault(m, key)
    return labels, conflicts


def predicted_streams_brief(m):
    """The assignment brief's model: IPG = ceil(M/ceil(M/4)), streams = ceil(M/4)."""
    return math.ceil(m / 4)


def adjudicate(cost, labels, widths):
    """Score the pre-registered predictions against one measured curve."""
    incr = {m: cost[m] / cost[m - 1] for m in widths if m - 1 in cost and m in cost}
    live_bnd = [
        m for m in widths
        if m - 1 in labels and m in labels
        and labels[m][3] is not None and labels[m - 1][3] is not None
        and labels[m][3] != labels[m - 1][3]
    ]
    brief_bnd = [
        m for m in widths
        if m > 1 and predicted_streams_brief(m) != predicted_streams_brief(m - 1)
    ]
    order = sorted(incr, key=lambda m: incr[m], reverse=True)
    return {
        "c2_over_c1": cost[2] / cost[1] if 2 in cost else None,
        "step_ratios": incr,
        "increment_rank": {str(m): order.index(m) + 1 for m in incr},
        "live_stream_boundaries": live_bnd,
        "brief_stream_boundaries": brief_bnd,
        "live_boundary_steps": {str(m): incr[m] for m in live_bnd if m in incr},
        "brief_boundary_steps": {str(m): incr[m] for m in brief_bnd if m in incr},
        "largest_step_at": order[0] if order else None,
        "largest_step": incr[order[0]] if order else None,
        # P3: the in-source comment at quantized.h case 8 claims M=9 profiles
        # cheaper than M=8. A pass model forbids that.
        "monotone_non_decreasing": all(
            cost[m] >= cost[m - 1] for m in widths if m - 1 in cost and m in cost
        ),
        "c9_over_c8": (cost[9] / cost[8]) if 9 in cost and 8 in cost else None,
        "c8_over_c7": (cost[8] / cost[7]) if 8 in cost and 7 in cost else None,
    }


def s18_pricing(cost):
    """Price the depth histogram against constant-cost and true-C(M) verify.

    `Qwen36MTPDepthPolicy` compares a fixed `headStepCostRatio` against a
    threshold that has no C(M) term, so it implicitly treats the verify round
    as costing C(1) whatever the depth. This is the size of that error.
    """
    rounds = sum(S18_DEPTH_HISTOGRAM.values())
    modelled = sum(n * cost[1] for n in S18_DEPTH_HISTOGRAM.values())
    actual = sum(
        n * cost[d + 1] for d, n in S18_DEPTH_HISTOGRAM.items() if d + 1 in cost
    )
    covered = sum(n for d, n in S18_DEPTH_HISTOGRAM.items() if d + 1 in cost)
    # A scalar h fitted end-to-end cannot be wrong about the mean: it absorbs
    # whatever constant best prices the observed mix. The C(1) anchor therefore
    # overstates the correctable error. What a scalar provably cannot absorb is
    # the dispersion of C(M) around that fitted mean, so report it separately.
    fitted = actual / covered if covered else None
    dispersion = (
        {str(d + 1): cost[d + 1] / fitted for d in sorted(S18_DEPTH_HISTOGRAM) if d + 1 in cost}
        if fitted
        else {}
    )
    return {
        "rounds": rounds,
        "rounds_priced": covered,
        "width_mix": {str(d + 1): n / rounds for d, n in S18_DEPTH_HISTOGRAM.items()},
        "verify_seconds_if_constant": modelled,
        "verify_seconds_actual": actual,
        "understatement": actual / modelled if modelled else None,
        "understatement_vs_c2": (
            actual / (covered * cost[2]) if covered and 2 in cost else None
        ),
        "best_fit_constant_seconds": fitted,
        "residual_after_best_fit_constant": dispersion,
        "residual_span": (
            max(dispersion.values()) / min(dispersion.values()) if dispersion else None
        ),
        "marginal_cost_of_one_more_draft": {
            str(d): (cost[d + 2] / cost[d + 1])
            for d in sorted(S18_DEPTH_HISTOGRAM)
            if d + 2 in cost and d + 1 in cost
        },
    }


def reproducibility(a, b, widths):
    out = {}
    for name, sa in shapes_by_name(a).items():
        sb = shapes_by_name(b).get(name)
        if sb is None:
            continue
        ra, rb = rows_by_m(sa), rows_by_m(sb)
        deltas = {
            m: abs(rb[m]["seconds_per_call"] - ra[m]["seconds_per_call"])
            / ra[m]["seconds_per_call"]
            for m in widths
            if m in ra and m in rb
        }
        if deltas:
            out[name] = {
                "max_abs_rel_delta": max(deltas.values()),
                "per_m": deltas,
            }
    return out


def fmt_curve(rows):
    head = (f"  {'M':>2s} {'C(M) us':>9s} {'C/C(1)':>7s} {'C/M us':>8s} "
            f"{'C/M norm':>9s} {'str':>3s} {'ipg':>3s} {'spread':>7s}  kernel")
    lines = [head]
    for r in rows:
        s = r["weight_streams"]
        g = r["inputs_per_group"]
        lines.append(
            f"  {r['m']:2d} {r['c_seconds']*1e6:9.2f} {r['c_over_c1']:7.4f} "
            f"{r['c_over_m']*1e6:8.2f} {r['c_over_m_norm']:9.4f} "
            f"{('-' if s is None else s):>3} {('-' if g is None else g):>3} "
            f"{r['spread']*100:6.2f}%  {r['kernel']}"
        )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r1", required=True, help="vendored.json for repeat 1")
    ap.add_argument("--r2", help="vendored.json for repeat 2")
    ap.add_argument("--identity", action="append", default=[],
                    help="identity.txt to record verbatim, repeatable")
    ap.add_argument("--out", help="write the report JSON here")
    ap.add_argument("--max-m", type=int, default=12)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-name", default="e22-cm-staircase")
    args = ap.parse_args()

    a = load(args.r1)
    b = load(args.r2) if args.r2 else None
    widths = [m for m in a["widths"] if m <= args.max_m]

    labels, conflicts = dispatch_labels(a, widths)
    per_shape = {s["name"]: curve(s, widths) for s in a["shapes"]}
    rc = round_curve(a, widths)
    rc_adj = adjudicate(rc, labels, widths)

    identities = []
    for p in args.identity:
        if os.path.exists(p):
            with open(p) as f:
                identities.append({"path": p, "text": f.read().strip()})

    report = {
        "experiment": "E22-Q1-cm-staircase",
        "bits": 4,
        "widths": widths,
        "device": a.get("device"),
        "reps": a.get("reps"),
        "inner_calls_per_rep": a.get("inner_calls_per_rep"),
        "identities": identities,
        "dispatch": {
            str(m): {
                "kernel": labels[m][0], "crossrow": labels[m][1],
                "inputs_per_group": labels[m][2], "weight_streams": labels[m][3],
                "brief_predicted_streams": predicted_streams_brief(m),
            }
            for m in sorted(labels)
        },
        "dispatch_conflicts": conflicts,
        "per_shape_curve": per_shape,
        "per_shape_adjudication": {
            name: adjudicate(
                {r["m"]: r["c_seconds"] for r in rows}, labels, widths)
            for name, rows in per_shape.items()
        },
        "round_curve_seconds": rc,
        "round_curve_over_c1": {m: rc[m] / rc[1] for m in rc},
        "round_curve_over_m": {m: rc[m] / m for m in rc},
        "round_adjudication": rc_adj,
        "s18_pricing": s18_pricing(rc),
        "s18_provenance": (
            "a 512-token measurement under E17's S18 policy; depth histogram "
            f"{S18_DEPTH_HISTOGRAM}, M = depth + 1"
        ),
    }
    if b is not None:
        report["reproducibility"] = reproducibility(a, b, widths)
        report["r2_round_curve_seconds"] = round_curve(b, widths)

    print(f"E22 Q1: C(M) at bits=4, {a.get('device', {}).get('name', '?')}, "
          f"reps={a.get('reps')} inner={a.get('inner_calls_per_rep')}")
    for ident in identities:
        print(f"\n{ident['path']}:")
        for line in ident["text"].splitlines():
            print(f"  {line}")

    print("\nlive dispatch (all scored shapes agree; conflicts: "
          f"{len(conflicts)})")
    print(f"  {'M':>2s} {'str':>3s} {'ipg':>3s} {'brief':>5s}  kernel")
    for m in sorted(labels):
        k, cr, g, s = labels[m]
        print(f"  {m:2d} {('-' if s is None else s):>3} "
              f"{('-' if g is None else g):>3} {predicted_streams_brief(m):5d}  {k}")

    for name in HEADLINE:
        if name in per_shape:
            s = shapes_by_name(a)[name]
            print(f"\n{name}  K={s['k']} N={s['n']} "
                  f"weight_bytes={s['weight_bytes']} "
                  f"calls_per_verify={s['calls_per_verify']}")
            print(fmt_curve(per_shape[name]))

    print("\nround-weighted verify cost (sum over shapes of "
          "calls_per_verify * C(M))")
    print(f"  {'M':>2s} {'C_round ms':>11s} {'/C(1)':>7s} {'/M ms':>8s} "
          f"{'step':>7s}")
    for m in widths:
        if m not in rc:
            continue
        step = rc_adj["step_ratios"].get(m)
        print(f"  {m:2d} {rc[m]*1e3:11.3f} {rc[m]/rc[1]:7.4f} "
              f"{rc[m]/m*1e3:8.3f} {('-' if step is None else f'{step:.4f}'):>7}")

    print("\nadjudication (round-weighted)")
    print(f"  P1  C(2)/C(1) = {rc_adj['c2_over_c1']:.4f}")
    print(f"  live stream boundaries  M={rc_adj['live_stream_boundaries']} "
          f"steps={ {k: round(v, 4) for k, v in rc_adj['live_boundary_steps'].items()} }")
    print(f"  brief stream boundaries M={rc_adj['brief_stream_boundaries']} "
          f"steps={ {k: round(v, 4) for k, v in rc_adj['brief_boundary_steps'].items()} }")
    print(f"  largest single step at M={rc_adj['largest_step_at']} "
          f"({rc_adj['largest_step']:.4f}x)")
    print(f"  P3  monotone non-decreasing = {rc_adj['monotone_non_decreasing']}, "
          f"C(8)/C(7)={rc_adj['c8_over_c7']}, C(9)/C(8)={rc_adj['c9_over_c8']}")

    sp = report["s18_pricing"]
    print(f"\nS18 width mix ({report['s18_provenance']})")
    print(f"  verify seconds if cost were constant in depth: "
          f"{sp['verify_seconds_if_constant']*1e3:.2f} ms")
    print(f"  verify seconds at the measured C(M):           "
          f"{sp['verify_seconds_actual']*1e3:.2f} ms")
    print(f"  the constant-cost model understates verify by "
          f"{(sp['understatement']-1)*100:.1f}% anchored at C(1), "
          f"{(sp['understatement_vs_c2']-1)*100:.1f}% anchored at C(2)")
    print(f"  best-fit constant (what a scalar h already absorbs): "
          f"{sp['best_fit_constant_seconds']*1e3:.2f} ms/round")
    print("  residual a scalar cannot absorb, C(M)/best-fit: " + "  ".join(
        f"M={m}:{v:.3f}" for m, v in sorted(
            sp["residual_after_best_fit_constant"].items(), key=lambda kv: int(kv[0]))))
    print(f"  residual span (max/min over the mix): {sp['residual_span']:.3f}x")
    print("  marginal cost of one more draft, C(M+1)/C(M): " + "  ".join(
        f"depth {d}->{int(d)+1}:{v:.3f}" for d, v in sorted(
            sp["marginal_cost_of_one_more_draft"].items(), key=lambda kv: int(kv[0]))))

    if b is not None:
        worst = max(
            (v["max_abs_rel_delta"], k) for k, v in report["reproducibility"].items()
        )
        print(f"\nr1 vs r2: worst per-point |delta| = {worst[0]*100:.3f}% "
              f"({worst[1]})")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2, sort_keys=True)
        print(f"\nwrote {args.out}")

    if args.wandb:
        log_wandb(report, per_shape, rc, args.wandb_name)


def log_wandb(report, per_shape, rc, name):
    import wandb

    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "qwen38-mlx-challenge-senpai"),
        entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
        name=name,
        job_type="qmv-cost-curve",
        config={
            "experiment": report["experiment"],
            "bits": report["bits"],
            "widths": report["widths"],
            "device": report["device"],
            "reps": report["reps"],
            "inner_calls_per_rep": report["inner_calls_per_rep"],
            "s18_provenance": report["s18_provenance"],
            "host_caveat": "Apple M4 Pro, not the ranked M5; directional only",
        },
    )
    cols = ["shape", "m", "c_seconds", "c_over_c1", "c_over_m", "c_over_m_norm",
            "kernel", "crossrow", "inputs_per_group", "weight_streams", "spread"]
    curve_table = wandb.Table(columns=cols)
    for rows in per_shape.values():
        for r in rows:
            curve_table.add_data(*[r[c] for c in cols])

    round_table = wandb.Table(
        columns=["m", "c_round_seconds", "c_round_over_c1", "c_round_over_m",
                 "weight_streams", "brief_streams"])
    for m in sorted(rc):
        d = report["dispatch"].get(str(m), {})
        round_table.add_data(
            m, rc[m], rc[m] / rc[1], rc[m] / m,
            d.get("weight_streams"), d.get("brief_predicted_streams"))

    dispatch_table = wandb.Table(
        columns=["m", "kernel", "crossrow", "inputs_per_group",
                 "weight_streams", "brief_predicted_streams"])
    for m in sorted(report["dispatch"], key=int):
        d = report["dispatch"][m]
        dispatch_table.add_data(
            int(m), d["kernel"], d["crossrow"], d["inputs_per_group"],
            d["weight_streams"], d["brief_predicted_streams"])

    adj = report["round_adjudication"]
    sp = report["s18_pricing"]
    run.log({
        "e22/cost_curve": curve_table,
        "e22/round_curve": round_table,
        "e22/dispatch": dispatch_table,
        "e22/c_over_c1_by_shape": wandb.plot.line(
            curve_table, "m", "c_over_c1", stroke="shape",
            title="C(M)/C(1) per scored shape, bits=4"),
        "e22/c_over_m_by_shape": wandb.plot.line(
            curve_table, "m", "c_over_m_norm", stroke="shape",
            title="C(M)/M normalised by C(1): flat means width is free per row"),
        "e22/round_c_over_c1": wandb.plot.line(
            round_table, "m", "c_round_over_c1",
            title="round-weighted verify cost relative to M=1"),
        "e22/p1_c2_over_c1": adj["c2_over_c1"],
        "e22/largest_step": adj["largest_step"],
        "e22/largest_step_at_m": adj["largest_step_at"],
        "e22/monotone_non_decreasing": adj["monotone_non_decreasing"],
        "e22/c8_over_c7": adj["c8_over_c7"],
        "e22/c9_over_c8": adj["c9_over_c8"],
        "e22/s18_verify_understatement": sp["understatement"],
    })
    for m, c in sorted(rc.items()):
        run.log({"e22/m": m, "e22/c_round_seconds": c,
                 "e22/c_round_over_c1": c / rc[1], "e22/c_round_over_m": c / m})
    if "reproducibility" in report:
        run.summary["e22/worst_r1_r2_rel_delta"] = max(
            v["max_abs_rel_delta"] for v in report["reproducibility"].values())
    run.summary.update({
        "e22/live_stream_boundaries": adj["live_stream_boundaries"],
        "e22/brief_stream_boundaries": adj["brief_stream_boundaries"],
        "e22/step_ratios": {str(k): v for k, v in adj["step_ratios"].items()},
    })
    print(f"wandb run: {run.url}  id={run.id}")
    run.finish()


if __name__ == "__main__":
    main()
