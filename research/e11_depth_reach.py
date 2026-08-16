#!/usr/bin/env python3
"""Research-only (qwen38-r1-e11-depth-lever-showdown): can the shipped rule
ever reach depth 4 under the measured curve?

The advisor's prediction 3 (`H8` == `H`) rests on an analytic claim: under the
measured per-depth curve the greedy walk stops at 3 for EVERY acceptance
profile, so `segmentedVerifyDepthCap` is never read and opening it 7 -> 8
cannot change a schedule. If that is true, `H8` != `H` would mean the arms are
misbuilt rather than that the caps matter -- which is a very different headline.
This reproduces the walk in `costModelDepth` exactly and reports the tightest
threshold each step must clear.

usage: research/e11_depth_reach.py
"""

import itertools
import random

CURVE = [0.0842, 0.0775, 0.2426, 0.3754, 0.2919, 0.3000, 0.2870, 0.3909]
FLAT = [0.20] * 8


def walk(p, h, cap=8):
    """`costModelDepth` with per-position acceptance `p`; returns the depth."""
    reach, expected, cum_h, depth = 1.0, 0.0, 0.0, 0
    while depth < cap:
        reach *= p[depth]
        if not reach > h[depth] * (1.0 + expected) / (1.0 + cum_h):
            break
        expected += reach
        cum_h += h[depth]
        depth += 1
    return depth


def threshold_at(step, h):
    """Best-case (p == 1 everywhere) threshold the `step -> step+1` test faces.

    `reach` is a product of probabilities, so it is bounded by 1. `expected`
    and `cum_h` are both maximised at p == 1, and the threshold is increasing
    in `expected`, so this is the easiest the test can ever be.
    """
    expected, cum_h = 0.0, 0.0
    for d in range(step):
        expected += 1.0
        cum_h += h[d]
    return h[step] * (1.0 + expected) / (1.0 + cum_h)


def main():
    for name, h in (("measured curve", CURVE), ("flat 0.20", FLAT)):
        print(f"\n=== {name} ===")
        print(f"  depth at p=1.0 (best case, cap 8): {walk([1.0] * 8, h)}")
        for step in range(8):
            thr = threshold_at(step, h)
            print(f"  step {step} -> {step + 1}: best-case threshold "
                  f"{thr:.4f}  reachable={'yes' if thr < 1.0 else 'NO'}")

    print("\n=== constant-q sweep (both curves, cap 8) ===")
    for q in (0.70, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.99, 1.00):
        print(f"  q={q:.2f}  curve={walk([q] * 8, CURVE)}  "
              f"flat={walk([q] * 8, FLAT)}")

    print("\n=== adversarial search for curve depth >= 4 ===")
    worst = 0
    rng = random.Random(11)
    # Monotone-decreasing profiles are the realistic shape, but the claim is
    # for ANY profile, so search unconstrained vectors too.
    for _ in range(400000):
        p = [rng.random() for _ in range(8)]
        worst = max(worst, walk(p, CURVE))
    for combo in itertools.product((0.5, 0.9, 0.99, 1.0), repeat=4):
        p = list(combo) + [1.0] * 4
        worst = max(worst, walk(p, CURVE))
    worst = max(worst, walk([1.0] * 8, CURVE))
    print(f"  max depth found over 400k random + grid profiles: {worst}")
    print(f"  => segmentedVerifyDepthCap (7 or 8) is "
          f"{'UNREACHABLE' if worst <= 4 else 'REACHABLE'}; "
          f"width cap binds at {min(worst + 1, 8)}")


if __name__ == "__main__":
    main()
