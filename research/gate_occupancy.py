#!/usr/bin/env python3
"""Occupancy and blended-ratio table for the deep-round streak gate.

Answers the advisor's FB2 request directly: for a per-draft acceptance q and a
streak-gate value G, what fraction of rounds actually run at the deep cap, and
what blended decode speedup does that produce under the *measured* round-cost
curve C(d)?

Model.  The schedule is a two-phase renewal process.  After any rejection the
streak is zero and the cap is the shallow SDPA width wall, so the session must
land G consecutive fully-accepted shallow rounds before the deep cap reopens.
It then stays deep until the next rejection.  Both phase lengths are stopping
times for i.i.d. rounds, so Wald's identity gives the expected tokens per phase
as (expected rounds) x (expected tokens per round).

Cost is paid per verified row regardless of acceptance, which is what makes the
gate a real trade rather than free insurance.
"""

from __future__ import annotations

import argparse
import json

SERIAL_MS = 65.58  # one serial target forward, measured on this host
SHALLOW_DEPTH = 4

# Measured round cost, fully-accepted rounds only, run C (M4 Pro, 256 tokens).
MEASURED_C = {2: 79.7, 4: 126.4, 5: 146.5, 6: 168.3, 7: 189.7, 8: 217.4}
# Linear fit over the qmv-linear region, used where no direct measurement exists.
FIT_FIXED_MS = 12.21
FIT_SLOPE_MS = 22.49

# Realised per-position acceptance, run C, draft index 0..7.
MEASURED_P = [1.0000, 0.9737, 1.0000, 0.9722, 1.0000, 1.0000, 0.9444, 0.8462]


def cost(depth: int, *, no_kink: bool = False) -> float:
    """Round cost in ms at drafting depth `depth` (verify width depth+1)."""
    if no_kink:
        return FIT_FIXED_MS + FIT_SLOPE_MS * (depth + 1)
    if depth in MEASURED_C:
        return MEASURED_C[depth]
    return FIT_FIXED_MS + FIT_SLOPE_MS * (depth + 1)


def accept_profile(depth: int, q: float | None) -> tuple[float, float]:
    """(expected tokens emitted, probability the whole round is accepted)."""
    if q is None:
        prefix, total = 1.0, 1.0
        for i in range(depth):
            prefix *= MEASURED_P[i]
            total += prefix
        return total, prefix
    total, prefix = 1.0, 1.0
    for _ in range(depth):
        prefix *= q
        total += prefix
    return total, prefix


def rounds_to_streak(a: float, gate: int) -> float:
    """Expected rounds to collect `gate` consecutive successes at rate `a`."""
    if gate <= 0:
        return 0.0
    if a <= 0.0:
        return float("inf")
    if a >= 1.0:
        return float(gate)
    return (1.0 - a**gate) / (a**gate * (1.0 - a))


def blended(deep_depth: int, gate: int, q: float | None, *, no_kink: bool = False) -> dict:
    t_deep, a_deep = accept_profile(deep_depth, q)
    t_shal, a_shal = accept_profile(SHALLOW_DEPTH, q)
    c_deep, c_shal = cost(deep_depth, no_kink=no_kink), cost(SHALLOW_DEPTH, no_kink=no_kink)

    n_shal = rounds_to_streak(a_shal, gate)
    n_deep = float("inf") if a_deep >= 1.0 else 1.0 / (1.0 - a_deep)

    if n_deep == float("inf"):
        deep_share, tokens, ms = 1.0, t_deep, c_deep
    elif n_shal == float("inf"):
        deep_share, tokens, ms = 0.0, t_shal, c_shal
    else:
        total_rounds = n_shal + n_deep
        deep_share = n_deep / total_rounds
        tokens = n_shal * t_shal + n_deep * t_deep
        ms = n_shal * c_shal + n_deep * c_deep

    return {
        "deep_depth": deep_depth,
        "gate": gate,
        "deep_round_share": deep_share,
        "tokens_per_deep_round": t_deep,
        "occupancy_deep": t_deep / (deep_depth + 1),
        "full_accept_prob_deep": a_deep,
        "full_accept_prob_shallow": a_shal,
        "expected_shallow_rounds": n_shal,
        "expected_deep_rounds": n_deep,
        "ms_per_token": ms / tokens,
        "blended_ratio": SERIAL_MS * tokens / ms,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--acceptance",
        default="0.70,0.80,0.85,0.90,0.93,0.95,0.96,0.98",
        help="comma-separated per-draft acceptance values; 'measured' adds the realised profile",
    )
    ap.add_argument("--gates", default="0,1,2,3,4")
    ap.add_argument("--depths", default="4,5,6,7,8")
    ap.add_argument("--no-kink", action="store_true", help="use the linear fit at width 9 too")
    ap.add_argument("--out")
    args = ap.parse_args()

    qs: list[float | None] = [float(x) for x in args.acceptance.split(",") if x]
    qs.append(None)  # measured per-position profile
    gates = [int(x) for x in args.gates.split(",") if x]
    depths = [int(x) for x in args.depths.split(",") if x]

    rows = []
    for q in qs:
        for gate in gates:
            for depth in depths:
                r = blended(depth, gate, q, no_kink=args.no_kink)
                r["acceptance"] = "measured" if q is None else q
                rows.append(r)

    out = {
        "serial_ms_per_token": SERIAL_MS,
        "shallow_depth": SHALLOW_DEPTH,
        "cost_curve_ms": {str(d): cost(d, no_kink=args.no_kink) for d in depths},
        "measured_per_position_acceptance": MEASURED_P,
        "no_kink": args.no_kink,
        "rows": rows,
    }

    label = lambda q: "measured" if q is None else f"{q:.2f}"
    print(f"serial forward {SERIAL_MS} ms/token; shallow cap depth {SHALLOW_DEPTH}")
    print("cost C(d) ms:", {d: round(cost(d, no_kink=args.no_kink), 1) for d in depths})
    print()
    for q in qs:
        sub = [r for r in rows if r["acceptance"] == (("measured") if q is None else q)]
        best = max(sub, key=lambda r: r["blended_ratio"])
        print(f"q={label(q)}   best: depth {best['deep_depth']} gate {best['gate']}"
              f"  ratio {best['blended_ratio']:.4f}")
        print("   gate | depth | deep_share | occupancy | tok/round |  ms/tok | ratio")
        for r in sub:
            star = " *" if r is best else "  "
            print(f"   {r['gate']:4d} | {r['deep_depth']:5d} | {r['deep_round_share']:10.4f} |"
                  f" {r['occupancy_deep']:9.4f} | {r['tokens_per_deep_round']:9.4f} |"
                  f" {r['ms_per_token']:7.2f} | {r['blended_ratio']:.4f}{star}")
        print()

    # Decision rule: for each gate, the acceptance at which depth 8 overtakes depth 7.
    print("crossover: smallest q where deep depth 8 beats deep depth 7, per gate")
    for gate in gates:
        cross = None
        for i in range(700, 1000):
            q = i / 1000.0
            if blended(8, gate, q, no_kink=args.no_kink)["blended_ratio"] > blended(
                7, gate, q, no_kink=args.no_kink
            )["blended_ratio"]:
                cross = q
                break
        print(f"   gate {gate}: q* = {cross if cross is not None else '>0.999'}")
        out.setdefault("depth8_over_depth7_crossover", {})[str(gate)] = cross

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
