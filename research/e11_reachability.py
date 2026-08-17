#!/usr/bin/env python3
"""Research-only (qwen38-r1-e11-depth-lever-showdown Q3): decide whether the
width wall can bind under a given per-depth cost model, without running the GPU.

`costModelDepth` in Qwen36MTPBlockSession.swift extends a round from depth d to
d+1 iff

    reach > h[d] * (1 + expected) / (1 + cumH)

where `reach` is a running product of per-position acceptance probabilities,
`expected` is the sum of those reaches, and `cumH` is the sum of h[0..d-1].
Two properties make the wall question decidable rather than empirical:

  * `reach` is a product of probabilities, so reach <= 1 always; and
  * the two confidence clamps can only LOWER p (`p = min(p, sigmoid(...))`),
    so reach = 1 is the supremum over every acceptance profile and every prompt.

So evaluating the rule with every p pinned at 1.0 gives the maximum depth the
model can EVER select. If that maximum is below the width cap, the cap is
unreachable and the wall constant is inert -- the levers are substitutes, not
complements.
"""

import argparse

CURVE = [0.0842, 0.0775, 0.2426, 0.3754, 0.2919, 0.3000, 0.2870, 0.3909]
MAX_DEPTH = 8


def walk(h, cap, q):
    """Replicate costModelDepth with constant per-position acceptance q.

    Returns (depth, trace) where trace records the decision at each step.
    """
    reach, expected, cum_h, depth = 1.0, 0.0, 0.0, 0
    trace = []
    while depth < cap:
        reach *= q
        threshold = h[depth] * (1.0 + expected) / (1.0 + cum_h)
        take = reach > threshold
        trace.append((depth, depth + 1, reach, threshold, take))
        if not take:
            break
        expected += reach
        cum_h += h[depth]
        depth += 1
    return depth, trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scalar", type=float, default=0.18,
                    help="flat headStepCostRatio baseline to contrast")
    ap.add_argument("--wall", type=int, default=5,
                    help="sdpaWidthWallDepthCap (streak below the gate)")
    ap.add_argument("--segmented", type=int, default=8,
                    help="segmentedVerifyDepthCap (streak at or above gate)")
    args = ap.parse_args()

    models = {
        "curve": CURVE,
        f"scalar {args.scalar}": [args.scalar] * MAX_DEPTH,
    }

    for cap_name, cap in (("wall", args.wall), ("segmented", args.segmented)):
        cap = min(cap, MAX_DEPTH)
        print(f"\n=== widthCap = {cap} ({cap_name}) ===")
        print(f"{'q':>6}  " + "  ".join(f"{n:>14}" for n in models))
        for q in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0):
            row = [f"{walk(h, cap, q)[0]:>14}" for h in models.values()]
            print(f"{q:>6.2f}  " + "  ".join(row))

    # The supremum case is the only one that proves anything universal.
    print("\n=== supremum: every p pinned at 1.0, cap released to 8 ===")
    for name, h in models.items():
        depth, trace = walk(h, MAX_DEPTH, 1.0)
        print(f"\n{name}: max depth over ALL acceptance profiles = {depth}")
        for d, to, reach, thr, take in trace:
            print(f"  step {d} -> {to}: reach={reach:.4f} thr={thr:.4f} "
                  f"take={take}")
        if depth < args.wall:
            print(f"  => width cap {args.wall} is UNREACHABLE; the wall "
                  f"constant is inert and cannot compose with this model.")
        else:
            print(f"  => width cap {args.wall} CAN bind; the wall constant is "
                  f"live for this model.")


if __name__ == "__main__":
    main()
