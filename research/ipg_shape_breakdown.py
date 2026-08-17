#!/usr/bin/env python3
"""Per-shape breakdown of a QMV cost-curve arm against a reference (E14).

    research/ipg_shape_breakdown.py --ref e14-ref1 --arm e14-armB [--json-out F]

The aggregate weighted-verify ratio hides where a dispatch-table change costs
time.  This splits it per shape so a weight-traffic effect (which scales with
the weight footprint and appears at every shape) can be told apart from a
warmup or scheduling artifact (which appears on one shape only).

`seconds_per_call_min` is the default estimator: it is the best of the timed
regions for that row and therefore strips the clock-ramp transient that lands
on the first rows of the first shape of every sweep.
"""
import argparse
import json
import statistics
from pathlib import Path

CURVE_DIR = Path(__file__).resolve().parent.parent / ".mlxfast-private" / "qmv-curve"
WIDTHS = [1, 2, 3, 4, 5, 6, 7, 8, 9]


def load(tag):
    path = CURVE_DIR / tag / "vendored.json"
    if not path.is_file():
        raise SystemExit(f"missing curve for tag {tag}: {path}")
    return json.loads(path.read_text())


def rows_by_m(shape):
    return {r["m"]: r for r in shape["rows"]}


def weighted(doc, key):
    out = {}
    for m in WIDTHS:
        out[m] = sum(rows_by_m(s)[m][key] * s["calls_per_verify"] for s in doc["shapes"])
    return out


def h_vector(w):
    return [(w[d + 1] - w[d]) / w[1] for d in range(1, max(WIDTHS))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--estimator", default="seconds_per_call_min",
                    choices=["seconds_per_call", "seconds_per_call_min"])
    ap.add_argument("--changed", type=int, default=4,
                    help="verify width whose dispatch entry the arm changed")
    ap.add_argument("--json-out")
    args = ap.parse_args()

    ref, arm = load(args.ref), load(args.arm)
    key = args.estimator
    ref_by_name = {s["name"]: s for s in ref["shapes"]}

    controls = [m for m in WIDTHS if m != args.changed]
    wr, wa = weighted(ref, key), weighted(arm, key)
    ctl_ratios = [wa[m] / wr[m] for m in controls if m > 2]
    drift = statistics.median(ctl_ratios)

    shapes = []
    for order, sa in enumerate(arm["shapes"]):
        sr = ref_by_name[sa["name"]]
        rr, ra = rows_by_m(sr), rows_by_m(sa)
        m = args.changed
        shapes.append({
            "name": sa["name"],
            "sweep_order": order,
            "n": sa["n"],
            "k": sa["k"],
            "weight_mib": sa["weight_bytes"] / 2 ** 20,
            "calls_per_verify": sa["calls_per_verify"],
            "ref_seconds_per_call": rr[m][key],
            "arm_seconds_per_call": ra[m][key],
            "excess_fraction": ra[m][key] / rr[m][key] - 1,
            "weighted_excess_seconds": (ra[m][key] - rr[m][key]) * sa["calls_per_verify"],
            "ref_intra_run_spread": rr[m]["seconds_per_call_max"] / rr[m]["seconds_per_call_min"],
            "arm_intra_run_spread": ra[m]["seconds_per_call_max"] / ra[m]["seconds_per_call_min"],
            "ratio_by_width": {str(w): ra[w][key] / rr[w][key] for w in WIDTHS},
        })

    total_excess = sum(s["weighted_excess_seconds"] for s in shapes)
    out = {
        "ref_tag": args.ref,
        "arm_tag": args.arm,
        "estimator": key,
        "changed_width": args.changed,
        "control_widths": [m for m in controls if m > 2],
        "control_ratio_min": min(ctl_ratios),
        "control_ratio_max": max(ctl_ratios),
        "median_control_drift": drift,
        "noise_floor_halfwidth_pct": 100 * max(abs(r / drift - 1) for r in ctl_ratios),
        "weighted_verify_seconds_ref": wr,
        "weighted_verify_seconds_arm": wa,
        "changed_width_ratio": wa[args.changed] / wr[args.changed],
        "changed_width_ratio_drift_adjusted": (wa[args.changed] / wr[args.changed]) / drift,
        "changed_width_excess_seconds": wa[args.changed] - wr[args.changed],
        "changed_width_excess_in_h_units": (wa[args.changed] - wr[args.changed]) / wr[1],
        "h_ref": h_vector(wr),
        "h_arm": h_vector(wa),
        "shapes": shapes,
    }
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(out, indent=2) + "\n")

    print(f"{args.arm} / {args.ref}   estimator={key}   changed width M={args.changed}")
    print(f"  control widths {out['control_widths']}: "
          f"[{out['control_ratio_min']:.4f}, {out['control_ratio_max']:.4f}] "
          f"median drift {drift:.4f}, noise floor +/-{out['noise_floor_halfwidth_pct']:.2f}%")
    print(f"  M={args.changed} ratio {out['changed_width_ratio']:.4f} -> drift-adjusted "
          f"{out['changed_width_ratio_drift_adjusted']:.4f} "
          f"({out['changed_width_excess_in_h_units']:+.4f} h-units)")
    print()
    print(f"  {'shape':<34}{'W MiB':>9}{'calls':>7}{'excess':>9}{'share':>8}{'spread':>8}")
    for s in sorted(shapes, key=lambda x: -x["weighted_excess_seconds"]):
        share = s["weighted_excess_seconds"] / total_excess if total_excess else 0.0
        print(f"  {s['name']:<34}{s['weight_mib']:9.2f}{s['calls_per_verify']:>7}"
              f"{s['excess_fraction'] * 100:8.2f}%{share * 100:7.1f}%"
              f"{s['arm_intra_run_spread']:8.3f}")
    print()
    print("  h_ref = [" + ", ".join(f"{x:.4f}" for x in out["h_ref"]) + "]")
    print("  h_arm = [" + ", ".join(f"{x:.4f}" for x in out["h_arm"]) + "]")


if __name__ == "__main__":
    main()
