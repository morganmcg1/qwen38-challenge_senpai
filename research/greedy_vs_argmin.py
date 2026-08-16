#!/usr/bin/env python3
"""Check whether the shipped greedy depth walk and the argmin rule agree.

The shipped rule extends depth while ``reach > h*(1+expected)/(1+depth*h)``.
That test is exactly "f(d+1) < f(d)" for f(d) = (1 + sum h_i) / T(d), so greedy
returns the first local minimum of f.  The candidate rule returns the global
argmin of the same f.  They coincide iff f is unimodal over 0..cap.

Usage:
  research/greedy_vs_argmin.py --h 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 --trials 200000
"""
import argparse
import random


def greedy(h, p, cap):
    """Transcription of the shipped rule at dbed6c2:Qwen36MTPBlockSession.swift.

    Note `expected` counts EXTRA accepted drafts (starts at 0) and `reach` is
    updated before the threshold test.
    """
    reach = 1.0
    expected = 0.0
    depth = 0
    while depth < cap:
        hd = h[depth]
        reach *= p[depth]
        threshold = hd * (1.0 + expected) / (1.0 + depth * hd)
        if not reach > threshold:
            break
        expected += reach
        depth += 1
    return depth


def argmin(h, p, cap):
    reach = 1.0
    expected = 1.0
    cost = 1.0
    best = 0
    best_cpt = 1.0
    for depth in range(cap):
        reach *= p[depth]
        expected += reach
        cost += h[depth]
        cpt = cost / expected
        if cpt < best_cpt:
            best_cpt = cpt
            best = depth + 1
    return best


def cost_per_token(h, p, cap, d):
    reach = 1.0
    expected = 1.0
    cost = 1.0
    for i in range(d):
        reach *= p[i]
        expected += reach
        cost += h[i]
    return cost / expected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", nargs="+", type=float, required=True)
    ap.add_argument("--caps", nargs="+", type=int, default=[2, 4, 8])
    ap.add_argument("--trials", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--monotone", action="store_true",
                    help="only sample non-increasing acceptance profiles")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    h = args.h
    n = len(h)

    for cap in args.caps:
        cap = min(cap, n)
        mismatch = 0
        worst_gap = 0.0
        worst = None
        for _ in range(args.trials):
            if args.monotone:
                p = []
                cur = rng.random()
                for _ in range(n):
                    p.append(cur)
                    cur *= rng.random() ** 0.25
            else:
                p = [rng.random() for _ in range(n)]
            g = greedy(h, p, cap)
            a = argmin(h, p, cap)
            if g != a:
                mismatch += 1
                cg = cost_per_token(h, p, cap, g)
                ca = cost_per_token(h, p, cap, a)
                gap = (cg - ca) / cg
                if gap > worst_gap:
                    worst_gap = gap
                    worst = (tuple(round(x, 4) for x in p), g, a, cg, ca)
        print(f"cap={cap} trials={args.trials} mismatches={mismatch} "
              f"({100.0*mismatch/args.trials:.4f}%) worst_gap={100*worst_gap:.3f}%")
        if worst:
            print(f"  worst: p={worst[0]} greedy={worst[1]} argmin={worst[2]} "
                  f"cpt_greedy={worst[3]:.4f} cpt_argmin={worst[4]:.4f}")


if __name__ == "__main__":
    main()
