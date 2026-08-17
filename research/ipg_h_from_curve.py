#!/usr/bin/env python3
"""Derive the scheduler's marginal depth-cost vector h(d) from a QMV curve.

The base's forced-depth env hooks were removed when the winning schedule was
merged, so h(d) can no longer be measured by end-to-end depth arms. It can
still be measured directly, and more precisely, from the weighted verify cost
the curve already reports:

    h(d) = (weighted_verify_seconds[d+1] - weighted_verify_seconds[d])
           / weighted_verify_seconds[1]

Cross-checked against the shipped h vector on `e10-control`: h(4) = 0.3713 vs
0.3754 shipped (1.1%), whole-vector mean ratio 0.998.

    research/ipg_h_from_curve.py SUMMARY.json [--label NAME] [--json-out F]
    research/ipg_h_from_curve.py A.json --vs B.json
"""
from __future__ import annotations

import argparse
import json
import pathlib

SHIPPED_H = [0.0842, 0.0775, 0.2426, 0.3754, 0.2919, 0.3000, 0.2870, 0.3909]


def load_widths(path: pathlib.Path, estimator: str = "seconds_per_call") -> dict[int, float]:
    doc = json.loads(path.read_text())
    if "shapes" not in doc:
        return {int(k): v for k, v in doc["weighted_verify_seconds"].items()}
    widths = {}
    for m in range(1, 10):
        widths[m] = sum(
            {r["m"]: r for r in s["rows"]}[m][estimator] * s["calls_per_verify"]
            for s in doc["shapes"]
        )
    return widths


def h_vector(widths: dict[int, float]) -> list[float]:
    return [(widths[d + 1] - widths[d]) / widths[1] for d in range(1, 9)]


def depth_threshold(h: list[float], depth: int) -> dict[str, float]:
    """Whether stepping to `depth` can ever pay for itself.

    A depth-(d-1) round costs 1 + sum(h[:d-1]) and yields at most d tokens, so
    its best achievable cost per token is that sum over d. Stepping to depth d
    buys one further token for h[d-1]. Even at perfect acceptance the step only
    helps when that marginal cost is under the average already achieved, so
    `marginal / breakeven` above 1 means the step is unreachable at any q.
    """
    cum_before = sum(h[: depth - 1])
    marginal = h[depth - 1]
    breakeven = (1.0 + cum_before) / depth
    return {
        "depth": depth,
        "cum_h_before": cum_before,
        "marginal_h": marginal,
        "breakeven_marginal_h": breakeven,
        "threshold_ratio_at_q1": marginal / breakeven,
        "reachable_at_q1": marginal <= breakeven,
        "required_h_cut_fraction": (
            (marginal - breakeven) / marginal if marginal > breakeven else 0.0
        ),
    }


def report(label: str, path: pathlib.Path, estimator: str = "seconds_per_call") -> dict:
    widths = load_widths(path, estimator)
    h = h_vector(widths)
    out = {
        "label": label,
        "summary": str(path),
        "weighted_verify_seconds": {str(m): widths[m] for m in range(1, 10)},
        "h": h,
        "h_shipped": SHIPPED_H,
        "h_ratio_vs_shipped": [a / b for a, b in zip(h, SHIPPED_H)],
        "depth4": depth_threshold(h, 4),
    }
    print(f"== {label}  ({path})")
    print("  d   h_measured  h_shipped  ratio")
    for d, (a, b) in enumerate(zip(h, SHIPPED_H), start=1):
        mark = "  <-- depth-4 step" if d == 4 else ""
        print(f"  {d}   {a:10.4f}  {b:9.4f}  {a / b:5.3f}{mark}")
    d4 = out["depth4"]
    print(f"  cumH(h1..h3)={d4['cum_h_before']:.4f}  h4={d4['marginal_h']:.4f}"
          f"  breakeven h4<={d4['breakeven_marginal_h']:.4f}")
    print(f"  depth-4 reachable at q=1: {d4['reachable_at_q1']}"
          f"  (needs {100 * d4['required_h_cut_fraction']:.2f}% h4 cut)")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("summary", type=pathlib.Path)
    ap.add_argument("--label", default="curve")
    ap.add_argument("--vs", type=pathlib.Path, default=None)
    ap.add_argument("--vs-label", default="reference")
    ap.add_argument("--json-out", type=pathlib.Path, default=None)
    ap.add_argument("--estimator", default="seconds_per_call",
                    choices=["seconds_per_call", "seconds_per_call_min"],
                    help="only used when the input is a raw vendored.json")
    args = ap.parse_args()

    out = {"candidate": report(args.label, args.summary, args.estimator)}
    if args.vs is not None:
        print()
        out["reference"] = report(args.vs_label, args.vs, args.estimator)
        hc = out["candidate"]["h"]
        hr = out["reference"]["h"]
        wc = load_widths(args.summary, args.estimator)
        wr = load_widths(args.vs, args.estimator)
        print("\n== candidate vs reference")
        print("  d   h_ref     h_cand    delta      pct")
        for d, (a, b) in enumerate(zip(hr, hc), start=1):
            print(f"  {d}   {a:8.4f}  {b:8.4f}  {b - a:+8.4f}  {100 * (b - a) / a:+7.2f}%")
        print("\n  width  verify_ref_s  verify_cand_s   ratio")
        for m in range(1, 10):
            print(f"  {m:5d}  {wr[m]:12.6f}  {wc[m]:13.6f}  {wc[m] / wr[m]:6.4f}")
        out["h_delta"] = [b - a for a, b in zip(hr, hc)]
        out["verify_ratio"] = {str(m): wc[m] / wr[m] for m in range(1, 10)}

    if args.json_out is not None:
        args.json_out.write_text(json.dumps(out, indent=2) + "\n")
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
