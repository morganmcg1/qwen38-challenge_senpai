#!/usr/bin/env python3
"""Research-only: score the depth policy offline against a measured cost curve.

`costModelDepth` is a pure function of the acceptance estimates, the offered
cap and the cost constants, so once `h(d)` is measured the whole policy can be
replayed in arithmetic. That answers two questions without spending a
benchmark run:

  * how far does the *chosen* depth actually move when the scalar
    `headStepCostRatio` is replaced by the measured vector, and
  * does the vector still win at acceptance levels the local fixture never
    reaches?

The local fixture is a long-copy gate and accepts essentially every draft. The
ranked pool is natural prose, where the organizer's own calibration record
shows effective depth 1 and a depth-1 accept rate near 0.70. A policy tuned at
acceptance 1.0 can be strictly worse there, because a rejected draft still pays
its build, its verify row and its rollback. So every profile below is scored
against the *same* measured cost curve; only the policy's depth choice differs.
"""

import argparse
import json
import sys
from pathlib import Path

SCALAR = 0.20


def choose_depth(accept, cap, h):
    """Replay `costModelDepth`'s greedy walk.

    `h` is a per-step cost list (`h[k]` prices the (k+1)-th draft). A scalar
    model is just a constant list. The generalised threshold keeps the exact
    algebra of the shipped rule: extending is worth it when the probability of
    reaching the new token exceeds the relative cost increase, so with a
    per-step cost the denominator is the cumulative cost already committed
    rather than `depth * h`.
    """
    reach, expected, cumulative, depth = 1.0, 0.0, 0.0, 0
    while depth < cap:
        reach *= accept(depth)
        step = h[depth]
        if not reach > step * (1.0 + expected) / (1.0 + cumulative):
            break
        expected += reach
        cumulative += step
        depth += 1
    return depth, expected


def expected_tokens(accept, depth):
    reach, total = 1.0, 0.0
    for k in range(depth):
        reach *= accept(k)
        total += reach
    return 1.0 + total


def round_cost(h, depth):
    return 1.0 + sum(h[:depth])


def per_token(accept, h_true, depth):
    return round_cost(h_true, depth) / expected_tokens(accept, depth)


PROFILES = {
    "longcopy(1.00)": lambda d: 1.0,
    "flat 0.95": lambda d: 0.95,
    "flat 0.85": lambda d: 0.85,
    "flat 0.70": lambda d: 0.70,
    "0.70*0.95^d": lambda d: 0.70 * 0.95 ** d,
    "0.90*0.90^d": lambda d: 0.90 * 0.90 ** d,
    # Ranked-pool anchor: the unmodified tree measured depth-1 accept 0.699 on
    # the eight hidden natural-prose prompts and settled at effective depth 1.
    "ranked 0.699": lambda d: 0.699,
    "0.699*0.85^d": lambda d: 0.699 * 0.85 ** d,
    "0.699*0.70^d": lambda d: 0.699 * 0.70 ** d,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curve", type=Path, default=None,
                    help="curve.json from depth_cost_curve.py")
    ap.add_argument("--h", nargs="*", type=float, default=None,
                    help="explicit 8-entry measured h vector (overrides --curve)")
    ap.add_argument("--scalar", type=float, default=SCALAR)
    ap.add_argument("--caps", nargs="*", type=int, default=[4, 8])
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    if args.h:
        h_meas = list(args.h)
    elif args.curve:
        curve = json.loads(args.curve.read_text())["curve"]
        h_meas = [curve[str(d)]["h"] for d in range(1, 9)
                  if str(d) in curve and curve[str(d)]["h"] is not None]
    else:
        print("need --curve or --h", file=sys.stderr)
        return 2
    if len(h_meas) < 8:
        # The greedy walk indexes h[depth] up to cap-1; hold the last measured
        # step flat rather than silently extrapolating a trend.
        h_meas = h_meas + [h_meas[-1]] * (8 - len(h_meas))
    h_scalar = [args.scalar] * 8

    print(f"measured h = [{', '.join('%.4f' % v for v in h_meas)}]")
    print(f"scalar   h = {args.scalar}\n")

    rows = []
    for cap in args.caps:
        print(f"cap = {cap}   (cost of every row priced by the MEASURED curve)")
        print("  old  = shipped: greedy walk + scalar h")
        print("  grd  = greedy walk + measured vector (constant change only)")
        print("  CAND = argmin + measured vector (the shipped candidate)")
        print(f"{'profile':<14} {'d*old':>5} {'d*grd':>5} {'dCAND':>5} "
              f"{'T old':>6} {'T grd':>6} {'c/tok old':>10} {'c/tok grd':>10} "
              f"{'c/tokCAND':>10} {'grd vs old':>10} {'CANDvsold':>10}")
        for name, accept in PROFILES.items():
            d_old, _ = choose_depth(accept, cap, h_scalar)
            d_new, _ = choose_depth(accept, cap, h_meas)
            costs = [per_token(accept, h_meas, d) for d in range(cap + 1)]
            d_opt = min(range(cap + 1), key=lambda d: costs[d])
            c_old, c_new, c_opt = costs[d_old], costs[d_new], costs[d_opt]
            gain = 100.0 * (c_old - c_new) / c_old
            gain_opt = 100.0 * (c_old - c_opt) / c_old
            row = {"cap": cap, "profile": name, "d_old": d_old, "d_new": d_new,
                   "d_opt": d_opt,
                   "T_old": expected_tokens(accept, d_old),
                   "T_new": expected_tokens(accept, d_new),
                   "cost_per_token_old": c_old, "cost_per_token_new": c_new,
                   "cost_per_token_opt": c_opt,
                   "pct_new_vs_old": gain, "pct_opt_vs_old": gain_opt}
            rows.append(row)
            print(f"{name:<14} {d_old:>5} {d_new:>5} {d_opt:>5} "
                  f"{row['T_old']:>6.3f} {row['T_new']:>6.3f} "
                  f"{c_old:>10.4f} {c_new:>10.4f} {c_opt:>10.4f} "
                  f"{gain:>9.2f}% {gain_opt:>9.2f}%")
        print()

    # "A curve that fits beautifully but never changes the selected depth is a
    # negative result." This sweep is the direct answer to that.
    sweep = []
    print("flat-acceptance sweep: does the curve ever move the selected depth?")
    header = f"{'p':>5}"
    for cap in args.caps:
        header += (f" | cap{cap}: {'old':>3} {'grd':>3} {'grdgain':>8}"
                   f" {'CAND':>4} {'CANDgain':>9}")
    print(header)
    for i in range(11):
        p = 0.50 + 0.05 * i
        accept = (lambda q: (lambda d: q))(p)
        line = f"{p:>5.2f}"
        entry = {"accept": p, "caps": {}}
        for cap in args.caps:
            d_old, _ = choose_depth(accept, cap, h_scalar)
            d_new, _ = choose_depth(accept, cap, h_meas)
            costs = [per_token(accept, h_meas, d) for d in range(cap + 1)]
            d_cand = min(range(cap + 1), key=lambda d: costs[d])
            c_old, c_new, c_cand = costs[d_old], costs[d_new], costs[d_cand]
            gain = 100.0 * (c_old - c_new) / c_old
            gain_cand = 100.0 * (c_old - c_cand) / c_old
            entry["caps"][str(cap)] = {"d_old": d_old, "d_new": d_new,
                                       "d_cand": d_cand, "pct": gain,
                                       "pct_cand": gain_cand}
            line += (f" | {'':<5}{d_old:>3} {d_new:>3} {gain:>7.2f}%"
                     f" {d_cand:>4} {gain_cand:>8.2f}%")
        sweep.append(entry)
        print(line)
    print()

    # Feasibility cross-check from the advisor: raw = G * T / (1 + h*d).
    print("implied G for the promoted frontier score 2.9042 (raw = G*T/(1+H_d))")
    print(f"{'profile':<14} {'d':>3} {'T':>6} {'1+H_d':>7} {'G_needed':>9}")
    for name, accept in PROFILES.items():
        d, _ = choose_depth(accept, 8, h_meas)
        t = expected_tokens(accept, d)
        cost = round_cost(h_meas, d)
        print(f"{name:<14} {d:>3} {t:>6.3f} {cost:>7.4f} "
              f"{2.9042110287045 * cost / t:>9.3f}")
    h_bound = (3.099 - 1.0) / 8.0
    print(f"\nadvisor bound at G=1, T=9, d=8: mean h <= {h_bound:.4f}; "
          f"measured mean h = {sum(h_meas) / len(h_meas):.4f}")

    if args.json:
        args.json.write_text(json.dumps(
            {"h_measured": h_meas, "h_scalar": args.scalar, "rows": rows,
             "flat_sweep": sweep, "h_bound_at_G1": h_bound},
            indent=2, sort_keys=True))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
